"""Track 2 enhancement runner (stage 1 of 2): enhance every input clip and write the output WAVs.

Stage 2 (inference/sravaani_asr.py) transcribes these WAVs with the mandated SraVaani-1.0 call.
main.py starts this file in fresh processes (at most --chunk clips each) and sets its options with environment
variables (T2_*). The input clips are WAV files anywhere below T2_INPUT/T2_INPUT_SLUG; the model folders are found
below T2_INPUT by name (find_input).
The two systems:
  T4 = T2_ENHANCER 'ravg', T2_RAVG_THR 2.0 (the fusion path for every clip), T2_PP '' (no pass-through),
       T2_ZIP_TTA 's50f' (4 ZipEnhancer views averaged), T2_NO_RAW_FALLBACK 1;
  E3 = the defaults of this file (route rule with threshold 0.20, pass-through pt20, one ZipEnhancer view).
The file also keeps options that T4 and E3 do not use (enhancers 'frcrn' / 'mossgan', the routes 'route' /
'route_pt', a residual TF post-filter, an input-level option, a raw dump). They need files that this release does
not include.
Outputs (T2_OUT): enhanced_wavs.zip (PCM16 16 kHz mono, same length, same file names, stored),
  manifest.json (name, samples, route, input sha1, output sha1), sisdr.json (only when clean references exist).
No labels, no other clip and no test statistics are used to process a clip.
"""
import hashlib
import importlib
import importlib.machinery
import json
import os
import subprocess
import sys
import time
import types
import wave
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- configuration (T4: main.py sets T2_*; E3 = the defaults)
INPUT_SLUG = os.environ.get('T2_INPUT_SLUG') or 'validation'   # folder below T2_INPUT that holds the input clips
CKPT_KERNEL = ''                                # FRCRN option only (not used by T4 / E3): folder with a fine-tuned best.pt; '' = pretrained
ZIP_CKPT_KERNEL = os.environ.get('T2_ZIP_CKPT_KERNEL', 'ckpt_zipenhancer')  # folder with the fine-tuned ZipEnhancer final.pt (G3-ASR,
                                                #   sha256 1ec3186c...); '' = official pretrained weights
TFG_CKPT_KERNEL = os.environ.get('T2_TFG_CKPT_KERNEL', 'ckpt_tfgridnet')   # folder with the fine-tuned TF-GridNet final.pt (v8r-M,
                                                #   sha256 75538ba7...); '' = pretrained URGENT 2025 weights
GATE = ''                                       # '' or 'res_c2_q10' (residual TF post-filter; not used by T4 / E3)
PP = os.environ.get('T2_PP', 'pt20')                                      # '' (off; T4) | 'pt20' (pass-through, T 20 dB; E3) | 'full20' (+ quiet-frame muting; not used)
FIT_PEAK = {'1': True, '0': False}.get(os.environ.get('T2_FIT_PEAK', ''), True)                                  # True: a clip whose final output exceeds int16 is SCALED to fit (not clipped)
SAVE_RAW = os.environ.get('T2_SAVE_RAW', '') == '1'   # True + PP: also write the enhancer output before PP (enhanced_raw.zip)
IN_RMS_DBFS = os.environ.get('T2_IN_RMS_DBFS', '')     # input RMS target in dBFS (not used by T4 / E3); '' = off
ZIP_TTA = os.environ.get('T2_ZIP_TTA', '')       # ZipEnhancer test-time views, averaged: '' off (E3) | 's50' | 'f' | 's50f' (T4)
TTA_VIEWS = {'': (), 's50': ('s50',), 'f': ('f',), 's50f': ('s50', 'f', 's50f')}
assert ZIP_TTA in TTA_VIEWS, ZIP_TTA
TTA_SHIFT = 50
NO_RAW_FALLBACK = os.environ.get('T2_NO_RAW_FALLBACK', '') == '1'   # T4: stop on an enhancement failure instead of writing the raw input
ENHANCER = os.environ.get('T2_ENHANCER', 'ravg')                                # 'ravg' (T4, E3) = ZipEnhancer (ZIP_CKPT_KERNEL) + TF-GridNet (TFG_CKPT_KERNEL), fused; other options (not used): 'frcrn' | 'mossgan' | 'zip_whole' | 'tfg' | 'route' | 'route_pt'
RAVG_THR = float(os.environ.get('T2_RAVG_THR', '0.20'))   # route rule of 'ravg': zip_whole alone if its pass-through share >= RAVG_THR (E3: 0.20),
                                                #   else the fusion of zip and g tfg (g = LS gain); a value > 1 (T4: 2.0) = the fusion path for every clip
RAVG_W = float(os.environ.get('T2_RAVG_W', '0.70'))                                    # Zip weight of the plain average (used only when FUSE_HEAD = '')
RAVG_EDGE = int(os.environ.get('T2_RAVG_EDGE', '64'))                                   # the last RAVG_EDGE samples of the fused output = zip, and g without them (TF-GridNet iSTFT end spike); 0 = off
FUSE_HEAD = os.environ.get('T2_FUSE_HEAD', 'ckpt_fusion_head')         # folder (or file) with head.pt of training/fusion_head.py: the Zip weight per TF bin; '' = the plain average with RAVG_W
WIN_MAX_S = float(os.environ.get('T2_WIN_MAX_S', '20.0'))                                 # long clips: zip_whole / tfg on a clip longer than WIN_MAX_S s (or after an
WIN_S, WIN_OV_S = float(os.environ.get('T2_WIN_S', '20.0')), float(os.environ.get('T2_WIN_OV_S', '1.0'))                     #   out-of-memory error) run in WIN_S-s windows with WIN_OV_S-s linear cross-fades; <= 0 = off.
                                                #   20 s: the longest validation clip is 18.4 s
RPT_FIRST = 'tfg'                               # route_pt (not used by T4 / E3): FIRST on every clip; if its
RPT_SECOND = 'zip_whole'                        #   passthrough share (PP pt20 frames) >= RPT_THR the clip is 'easy'
RPT_THR = 0.20                                  #   and SECOND is used instead (the clip's own audio only)
ROUTE_A = 'frcrn'                               # route (not used by T4 / E3): per clip A if A's removed-energy ratio
ROUTE_B = 'zip_whole'                           #   10log10(|y_A|^2/|x-y_A|^2) >= ROUTE_THR dB, else B (the clip's
ROUTE_THR = 4.0                                 #   own audio only)
LIMIT = int(os.environ.get('T2_LIMIT', '0'))
T0 = time.time()
OUT = Path(os.environ.get('T2_OUT', '/kaggle/working'))
TMP = Path(os.environ.get('T2_TMP', '/tmp/t2'))
IN_ROOT = Path(os.environ.get('T2_INPUT', '/kaggle/input'))
TMP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SR = 16000


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def find_input(slug):
    for base in (IN_ROOT,):
        if (base / slug).exists():
            return base / slug
    hits = [p for p in IN_ROOT.rglob(slug) if p.is_dir()]
    if hits:
        return hits[0]
    raise SystemExit(f'input {slug} not found')


EN = find_input('helpers')                 # optional wheels (empty folder is fine)
NEED = {ROUTE_A, ROUTE_B} if ENHANCER == 'route' else ({RPT_FIRST, RPT_SECOND} if ENHANCER == 'route_pt' else
                                                        ({'zip_whole', 'tfg'} if ENHANCER == 'ravg' else {ENHANCER}))
ZD = find_input('zipenhancer') if 'zip_whole' in NEED else None
UD = find_input('espnet_universal') if 'tfg' in NEED else None
INP = find_input(INPUT_SLUG)
log('inputs', EN, INP)

# ---------------------------------------------------------------- environment (wheels from the helper folder if missing, stub modules)
whls = sorted(str(p_) for p_ in EN.glob('*.whl'))
for mod, key in (('einops', 'einops'), ('rotary_embedding_torch', 'rotary_embedding_torch')):
    try:
        importlib.import_module(mod)
    except ImportError:
        w = [x for x in whls if Path(x).name.startswith(key)]
        log('pip install', w)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', *w], check=True)


def _stub_call(*a, **k):
    raise RuntimeError('stub: this code path must not run')


for mod in ('yamlargparse', 'pydub', 'librosa'):
    try:
        importlib.import_module(mod)
    except ImportError:
        m = types.ModuleType(mod)
        m.__spec__ = importlib.machinery.ModuleSpec(mod, None)
        if mod == 'pydub':
            m.AudioSegment = None
        if mod == 'librosa':
            for sub in ('util', 'filters'):
                sm = types.ModuleType(f'librosa.{sub}')
                sm.__spec__ = importlib.machinery.ModuleSpec(f'librosa.{sub}', None)
                sm.normalize = sm.mel = sm.find_files = _stub_call
                setattr(m, sub, sm)
                sys.modules[f'librosa.{sub}'] = sm
            m.resample = m.load = m.stft = _stub_call
        sys.modules[mod] = m
        log('stub module', mod)

import torch  # noqa: E402

DEV = torch.device(os.environ.get('T2_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))   # Mac: mps
if DEV.type == 'cpu':
    torch.set_num_threads(os.cpu_count() or 4)
log('torch', torch.__version__, 'device', DEV, torch.cuda.get_device_name(0) if DEV.type == 'cuda' else '')


def unpacked(zip_name, marker):
    z = EN / zip_name
    if z.exists():
        d = TMP / zip_name.replace('.zip', '')
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        return next(d.rglob(marker)).parent
    return next(EN.rglob(marker)).parent


if NEED & {'frcrn', 'mossgan'}:              # ClearerVoice-Studio code: only for the FRCRN / MossFormerGAN options (not T4 / E3)
    CVZ = unpacked('clearvoice_pkg.zip', 'clearvoice/__init__.py').parent
    sys.path.insert(0, str(CVZ))
    from clearvoice.utils.decode import decode_one_audio  # noqa: E402
    from clearvoice.dataloader.dataloader import audio_norm  # noqa: E402
    from clearvoice.models.frcrn_se.frcrn import FRCRN_SE_16K  # noqa: E402
    from clearvoice.models.mossformer_gan_se.generator import MossFormerGAN_SE_16K  # noqa: E402

# ---------------------------------------------------------------- input clips
root = INP
for z in sorted(INP.rglob('*.zip')):
    d = TMP / ('in_' + z.stem)
    with zipfile.ZipFile(z) as f:
        f.extractall(d)
    root = TMP
    log('extracted', z.name)
wavs = sorted(p for p in (TMP if root == TMP else INP).rglob('*.wav') if not p.name.startswith('._'))
clean_dirs = {'syntheticCleanRefAudio'}
inputs = [p for p in wavs if p.parent.name not in clean_dirs]
CLEAN = {p.stem: p for p in wavs if p.parent.name in clean_dirs}
names = [p.name for p in inputs]
assert len(names) == len(set(names)), 'duplicate clip file names below the input root'
if LIMIT:
    inputs = inputs[:LIMIT]
log('input clips', len(inputs), 'clean references', len(CLEAN))


def read_wav(p):
    with wave.open(str(p), 'rb') as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR), p
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)


def wav_bytes(y_int_units):
    q = np.clip(np.rint(y_int_units), -32768, 32767).astype('<i2')
    return q


def si_sdr(reference, enhanced):
    ref = np.asarray(reference, dtype=np.float64)
    enh = np.asarray(enhanced, dtype=np.float64)
    n = min(len(ref), len(enh))
    ref, enh = ref[:n], enh[:n]
    scale = np.dot(enh, ref) / np.dot(ref, ref)
    s_target = scale * ref
    e_noise = enh - s_target
    with np.errstate(divide='ignore', invalid='ignore'):
        value = 10.0 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise))
    return float(np.clip(value, -100.0, 100.0))


# ---------------------------------------------------------------- enhancer
class Args:
    network = 'FRCRN_SE_16K'; sampling_rate = SR; one_time_decode_length = 120; decode_window = 1
    win_type = 'hanning'; win_len = 640; win_inc = 320; fft_len = 640


class MArgs:
    network = 'MossFormerGAN_SE_16K'; mode = 'inference'; sampling_rate = SR
    one_time_decode_length = 10; decode_window = 10          # the shipped inference yaml
    win_type = 'hamming'; win_len = 400; win_inc = 100; fft_len = 400


def load_pretrained(model, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck
    st = model.state_dict()
    miss = 0
    for k in st:
        for cand in (k, k.replace('module.', ''), 'module.' + k):
            if cand in sd and sd[cand].shape == st[k].shape:
                st[k] = sd[cand]
                break
        else:
            miss += 1
    assert miss == 0, f'{miss} pretrained tensors not loaded'
    model.load_state_dict(st)


MODELS = {}
if 'frcrn' in NEED:
    net = FRCRN_SE_16K(Args)
    load_pretrained(net.model, EN / 'frcrn_last_best_checkpoint.pt')
    if CKPT_KERNEL:
        ckp = next(find_input(CKPT_KERNEL).rglob('best.pt'))
        net.model.load_state_dict(torch.load(ckp, map_location='cpu'))
        log('fine-tuned FRCRN state_dict', ckp, hashlib.sha1(ckp.read_bytes()).hexdigest())
    MODELS['frcrn'] = net.model.to(DEV).eval()
if 'mossgan' in NEED:
    mnet = MossFormerGAN_SE_16K(MArgs)
    load_pretrained(mnet.model, EN / 'mossgan_last_best_checkpoint.pt')
    MODELS['mossgan'] = mnet.model.to(DEV).eval()


def _unz(root, zip_name, marker):
    z = root / zip_name
    if z.exists() and z.is_file():
        d = TMP / zip_name.replace('.zip', '')
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        return next(d.rglob(marker)).parent
    return next(root.rglob(marker)).parent


if 'zip_whole' in NEED:
    sys.path.insert(0, str(_unz(ZD, 'zipenhancer_pkg.zip', 'zip_infer.py')))
    from zip_infer import ZipEnh
    zw = str(next(ZD.rglob('zipenhancer_pytorch_model.bin')))
    if ZIP_CKPT_KERNEL:
        _kd = find_input(ZIP_CKPT_KERNEL)          # the training scripts write final.pt (best or its WiSE-FT average)
        zw = str(next(_kd.rglob('final.pt'), None) or next(_kd.rglob('best.pt')))
        log('fine-tuned ZipEnhancer state_dict', zw, hashlib.sha1(Path(zw).read_bytes()).hexdigest())
    MODELS['zip_whole'] = ZipEnh(DEV, weights=zw, mode='whole')
if 'tfg' in NEED:
    sys.path.insert(0, str(_unz(UD, 'espnet_enh_pkg.zip', 'infer.py')))
    from infer import UniEnh
    MODELS['tfg'] = UniEnh('tfgridnet_urgent25', str(DEV), lstm_chunk=128)
    if TFG_CKPT_KERNEL:                    # final.pt = model.state_dict() (the LSTM chunk keeps the keys)
        tw = str(next(find_input(TFG_CKPT_KERNEL).rglob('final.pt')))
        MODELS['tfg'].model.load_state_dict(torch.load(tw, map_location=DEV))
        MODELS['tfg'].model.eval()
        log('fine-tuned TF-GridNet state_dict', tw, hashlib.sha1(Path(tw).read_bytes()).hexdigest())
log('enhancer', ENHANCER, 'models', list(MODELS))
FHEAD = None
if FUSE_HEAD and ENHANCER == 'ravg':
    from torch import nn

    class _FHead(nn.Module):                 # arch 'e3' = training/fusion_head.py Head (same keys)
        def __init__(self, arch='e3'):
            super().__init__()
            assert arch in ('e3', 'g', 'c', 'gc'), arch
            self.arch, self.gain = arch, arch in ('g', 'gc')
            nout = 2 if self.gain else 1
            if arch in ('e3', 'g'):
                ch, layers = 32, [nn.Conv2d(4, 32, 3, padding=1), nn.ReLU()]
                for _ in range(3):
                    layers += [nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()]
            else:
                ch, layers, cin = 48, [], 5
                for dt, df in [(1, 1), (1, 1), (2, 2), (4, 2), (8, 1), (1, 1)]:   # (time, freq); layout (B, C, F, T)
                    layers += [nn.Conv2d(cin, 48, 3, padding=(df, dt), dilation=(df, dt)), nn.ReLU()]
                    cin = 48
            layers.append(nn.Conv2d(ch, nout, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, X, Z, G):
            eps = 1e-5
            lx, lz, lg = (torch.log(a.abs() + eps) for a in (X, Z, G))
            mu = lx.mean(dim=(1, 2), keepdim=True)
            f = [lx - mu, lz - mu, lg - mu, torch.cos(torch.angle(Z) - torch.angle(G))]
            if self.arch in ('c', 'gc'):
                f.append(torch.cos(torch.angle(X) - torch.angle(Z)))
            o = self.net(torch.stack(f, 1))
            w = torch.sigmoid(o[:, 0])
            Y = w * Z + (1 - w) * G
            if self.gain:
                Y = 1.25 * torch.sigmoid(o[:, 1]) * Y
            return Y

    _hp = Path(FUSE_HEAD) if Path(FUSE_HEAD).is_file() else next(find_input(FUSE_HEAD).rglob('head.pt'))
    _ck = torch.load(str(_hp), map_location='cpu')
    FHEAD = _FHead(_ck.get('arch', 'e3'))
    FHEAD.load_state_dict(_ck['state_dict'])
    FHEAD = FHEAD.to(DEV).eval()
    log('E3 fusion head', _hp, hashlib.sha256(_hp.read_bytes()).hexdigest(), 'arch', FHEAD.arch)


def _stft(a):
    return torch.stft(a, 512, 128, window=torch.hann_window(512, device=a.device), center=True, return_complex=True)


@torch.no_grad()
def fuse_head(x_int, ya, gyb):
    """istft(w Z + (1 - w) G), the head at its training scale (float audio in [-1, 1]); = fuse() of training/fusion_head.py."""
    xt, zt, gt = (torch.from_numpy((a / 32768.0).astype(np.float32))[None].to(DEV) for a in (x_int, ya, gyb))
    y = torch.istft(FHEAD(_stft(xt), _stft(zt), _stft(gt)), 512, 128, window=torch.hann_window(512, device=DEV),
                    center=True, length=len(ya))[0]
    return y.cpu().double().numpy() * 32768.0


def _zip_view(x, v, n):                            # T4 test-time view
    """One test-time view of the ZipEnhancer on float x in [-1, 1]: 's50' = 50 zeros in front (output samples 50 ..
    50 + n), 'f' = sign flip of input and output, 's50f' = both. Same length n as x."""
    xi = np.concatenate([np.zeros(TTA_SHIFT), x]) if v in ('s50', 's50f') else x
    if v in ('f', 's50f'):
        xi = -xi
    y = np.asarray(MODELS['zip_whole'](xi), dtype=np.float64).reshape(-1)
    if v in ('f', 's50f'):
        y = -y
    y = y[TTA_SHIFT:TTA_SHIFT + n] if v in ('s50', 's50f') else y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y


@torch.no_grad()
def _enhance_one_whole(name, x_int):
    x = x_int / 32768.0
    n = len(x)
    if name == 'frcrn':
        xn, scalar = audio_norm(x.astype(np.float64))
        y = decode_one_audio(MODELS['frcrn'], DEV, xn.astype(np.float32)[None], Args)
        y = np.asarray(y, dtype=np.float64).reshape(-1)[:n] * scalar
    elif name == 'mossgan':
        y = decode_one_audio(MODELS['mossgan'], DEV, x.astype(np.float32)[None], MArgs)
        y = np.asarray(y, dtype=np.float64).reshape(-1)[:n]
    else:                                   # zip_whole, tfg: float in [-1, 1] -> same length, input level
        y = np.asarray(MODELS[name](x), dtype=np.float64).reshape(-1)[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    if name == 'zip_whole' and ZIP_TTA:     # T4: the same views on every clip, summed in the order v0, s50, f, s50f
        for v_ in TTA_VIEWS[ZIP_TTA]:
            y = y + _zip_view(x, v_, n)
        y = y / (1 + len(TTA_VIEWS[ZIP_TTA]))
    return y * 32768.0


WINFLAG = [False]                           # set when the current clip used the window path (manifest 'windowed')


def _free_dev():
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()
    elif DEV.type == 'mps':
        torch.mps.empty_cache()


def enhance_one(name, x_int):
    """Long clips: zip_whole / tfg on a clip longer than WIN_MAX_S s, or after an out-of-memory error, in WIN_S-s windows with
    WIN_OV_S-s linear cross-fades (the clip's own audio only; the weights sum to 1). Every other call = the whole clip."""
    n = len(x_int)
    if name not in ('zip_whole', 'tfg') or WIN_MAX_S <= 0:
        return _enhance_one_whole(name, x_int)
    if n <= int(WIN_MAX_S * SR):
        try:
            return _enhance_one_whole(name, x_int)
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise
            log('OOM -> windows', name, n, repr(e)[:160])
            _free_dev()
    err = None
    for wl in (WIN_S, WIN_S / 2, WIN_S / 4):          # on an out-of-memory error: half-size windows (20 -> 10 -> 5 s)
        try:
            y = _windowed(name, x_int, int(wl * SR), int(WIN_OV_S * SR))
            WINFLAG[0] = True
            return y
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise
            err = e
            log('OOM in windows of', wl, 's', name, n, repr(e)[:120])
            _free_dev()
    raise err


def _windowed(name, x_int, W, O):
    n = len(x_int)
    starts = [0] if n <= W else sorted(set(list(range(0, n - W, W - O)) + [n - W]))   # equal window lengths (one
    y, wsum = np.zeros(n), np.zeros(n)                                                  # graph shape on MPS)
    for k, s0 in enumerate(starts):
        e = min(n, s0 + W)
        seg = _enhance_one_whole(name, x_int[s0:e])
        w = np.ones(e - s0)
        if k > 0:
            ov = min(n, starts[k - 1] + W) - s0                # the overlap with the previous window
            w[:ov] = np.linspace(0.0, 1.0, ov + 2)[1:-1]
        if k + 1 < len(starts):
            ov = e - starts[k + 1]                             # the overlap with the next window
            w[-ov:] = np.linspace(1.0, 0.0, ov + 2)[1:-1]
        y[s0:e] += w * seg
        wsum[s0:e] += w
        _free_dev()
    return y / np.maximum(wsum, 1e-9)


ROUTE = {}


def enhance(x_int):
    if ENHANCER == 'ravg':                  # PP (pt20) is applied afterwards in the main loop
        try:
            ya = enhance_one('zip_whole', x_int)
        except Exception as e:  # noqa: BLE001  (safety net: TF-GridNet alone; the raw input only if both fail)
            log('ZIP FAIL (tfg alone)', repr(e)[:200])
            if DEV.type == 'cuda':
                torch.cuda.empty_cache()
            ROUTE['last'], ROUTE['model'] = None, 'tfg_zipfail'
            return enhance_one('tfg', x_int)
        _, _, share = post_pp(ya, x_int, gate=False, T=20.0)
        ROUTE['last'] = float(share)
        if share >= RAVG_THR:
            ROUTE['model'] = 'zip_whole'
            return ya
        _free_dev()                         # release the Zip pass cache before the TF-GridNet pass (no numeric change)
        try:
            yb = enhance_one('tfg', x_int)
        except Exception as e:  # noqa: BLE001  (safety net: ZipEnhancer + pt instead of the raw input)
            log('TFG FAIL (zip_whole kept)', repr(e)[:200])
            if DEV.type == 'cuda':
                torch.cuda.empty_cache()
            ROUTE['model'] = 'zip_whole_tfgfail'
            return ya
        n_ = len(ya)
        if RAVG_EDGE and n_ > 4 * RAVG_EDGE:     # the TF-GridNet output can hold an iSTFT spike in its last samples
            e_ = n_ - RAVG_EDGE
            g = float(np.dot(ya[:e_], yb[:e_]) / (np.dot(yb[:e_], yb[:e_]) + 1e-9))
            y_ = fuse_head(x_int, ya, g * yb) if FHEAD is not None else RAVG_W * ya + (1 - RAVG_W) * g * yb
            y_[e_:] = ya[e_:]
            ROUTE['model'] = 'avg'
            return y_
        g = float(np.dot(ya, yb) / (np.dot(yb, yb) + 1e-9))
        ROUTE['model'] = 'avg'
        return fuse_head(x_int, ya, g * yb) if FHEAD is not None else RAVG_W * ya + (1 - RAVG_W) * g * yb
    if ENHANCER == 'route_pt':
        y1 = enhance_one(RPT_FIRST, x_int)
        _, _, share = post_pp(y1, x_int, gate=False, T=20.0)
        ROUTE['last'] = float(share)
        ROUTE['model'] = RPT_SECOND if share >= RPT_THR else RPT_FIRST
        return enhance_one(RPT_SECOND, x_int) if share >= RPT_THR else y1
    if ENHANCER != 'route':
        return enhance_one(ENHANCER, x_int)
    ya = enhance_one(ROUTE_A, x_int)
    r = 10 * np.log10((np.dot(ya, ya) + 1e-9) / (np.dot(x_int - ya, x_int - ya) + 1e-9))
    ROUTE['last'] = float(r)
    return ya if r >= ROUTE_THR else enhance_one(ROUTE_B, x_int)


if GATE:
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import istft, stft

    def _spec(x):
        return stft(x, fs=SR, window='hann', nperseg=512, noverlap=384, boundary='zeros', padded=True)[2]

    def gate(x, yh):
        c, q = (float(v) for v in GATE[5:].split('_q'))
        Rm = _spec(x - yh)
        P = uniform_filter1d(np.abs(Rm) ** 2, size=3, axis=1)
        D = np.percentile(P, q, axis=1)[:, None]
        G = np.clip(1.0 - c * D / (P + 1e-9), 0.0, 1.0)
        r = istft(G * Rm, fs=SR, window='hann', nperseg=512, noverlap=384, boundary=True)[1]
        out = np.zeros(len(x))
        out[:min(len(x), len(r))] = r[:min(len(x), len(r))]
        return x - out


# ---------------------------------------------------------------- PP (= training/pp_lib.py pp(); the clip's own audio only)
PP_L = 256


def _pp_smooth(mf, n):
    gm = np.convolve(np.repeat(mf.astype(float), PP_L), np.ones(81) / 81, mode='same')
    o = np.zeros(n)
    o[:len(gm)] = gm[:n]
    return o


def post_pp(y, x, gate, T, t1=30.0, t2=20.0):
    """Passthrough: frames where 10log10(|g x|^2/|g x - y|^2) > T take g x (g = LS gain over the least-changed loud
    frames), 5 ms fades. With the quiet-frame option, frames > t1 dB under the output p95 frame where > t2 dB was
    removed are first set to zero (not used by T4 / E3)."""
    n = len(y)
    nf = n // PP_L
    if nf < 8:
        return y, 0.0, 0.0
    yg, gshare = y, 0.0
    if gate:
        fy = 10 * np.log10(np.mean(y[:nf * PP_L].reshape(nf, PP_L) ** 2, 1) + 1e-6)
        fx = 10 * np.log10(np.mean(x[:nf * PP_L].reshape(nf, PP_L) ** 2, 1) + 1e-6)
        z = ((fy - np.percentile(fy, 95)) < -t1) & ((fx - fy) > t2)
        gshare = float(z.mean())
        yg = y * (1 - _pp_smooth(z, n))
    X = x[:nf * PP_L].reshape(nf, PP_L)
    Y = yg[:nf * PP_L].reshape(nf, PP_L)
    ex = (X ** 2).sum(1) + 1e-9
    g0 = (yg @ x) / (x @ x + 1e-9)
    rel = ((Y - g0 * X) ** 2).sum(1) / ex
    hi = ex >= np.median(ex)
    cand = hi & (rel <= np.quantile(rel[hi], 0.3)) if hi.any() else hi
    g = float((Y[cand] * X[cand]).sum() / ((X[cand] ** 2).sum() + 1e-9)) if cand.any() else g0
    ch = 10 * np.log10(((g * X) ** 2).sum(1) / (((g * X - Y) ** 2).sum(1) + 1e-9) + 1e-12)
    un = ch > T
    w = _pp_smooth(un, n)
    w[nf * PP_L:] = 0
    return (1 - w) * yg + w * (g * x), gshare, float(un.mean())


# ---------------------------------------------------------------- run
man, sis, fails = [], {}, 0
n_fit = 0
t = time.time()
audio_s = 0.0
zraw = zipfile.ZipFile(OUT / 'enhanced_raw.zip', 'w', zipfile.ZIP_STORED) if (PP and SAVE_RAW) else None
with zipfile.ZipFile(OUT / 'enhanced_wavs.zip', 'w', zipfile.ZIP_STORED) as zf:
    for i, p in enumerate(inputs):
        raw = p.read_bytes()
        x = read_wav(p)
        _in_c = 1.0
        if IN_RMS_DBFS:
            _in_c = float(32768.0 * 10 ** (float(IN_RMS_DBFS) / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9))
            x = x * _in_c
        WINFLAG[0] = False
        try:
            y = enhance(x)
            if GATE:
                y = gate(x, y)
        except Exception as e:  # noqa: BLE001
            fails += 1
            log('ENH FAIL', p.name, repr(e)[:300])
            if NO_RAW_FALLBACK:
                raise SystemExit(f'ENH FAIL on {p.name}: no raw-input fallback (T2_NO_RAW_FALLBACK=1)')
            y = x
        pp_info = None
        if PP:
            if zraw is not None:
                tmpr = TMP / 'raw.wav'
                with wave.open(str(tmpr), 'wb') as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(wav_bytes(y).tobytes())
                zraw.write(tmpr, p.name)
            try:
                y, gsh, psh = post_pp(y, x, gate=PP.startswith('full'), T=float(PP.lstrip('ptful')))
                pp_info = [round(gsh, 4), round(psh, 4)]
            except Exception as e:  # noqa: BLE001
                log('PP FAIL (enhancer output kept)', p.name, repr(e)[:300])
        if _in_c != 1.0:
            y = np.asarray(y, dtype=np.float64) / _in_c
            x = x / _in_c
        if FIT_PEAK:
            _pk = float(np.max(np.abs(y))) if len(y) else 0.0
            if _pk > 32767.0:
                y = np.asarray(y, dtype=np.float64) * (32767.0 / _pk)   # scale the whole clip (SI-SDR is scale-invariant)
                n_fit += 1
        q = wav_bytes(y)
        assert len(q) == len(x)
        tmpw = TMP / 'one.wav'
        with wave.open(str(tmpw), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(q.tobytes())
        zf.write(tmpw, p.name)
        if DEV.type == 'mps':
            torch.mps.empty_cache()
        man.append({'name': p.name, 'samples': int(len(x)), 'route_ratio_db': ROUTE.get('last'), 'route_model': ROUTE.get('model'), 'windowed': WINFLAG[0], 'pp_gate_pt': pp_info, 'in_sha1': hashlib.sha1(raw).hexdigest(),
                    'out_sha1': hashlib.sha1(tmpw.read_bytes()).hexdigest()})
        if p.stem in CLEAN:
            sis[p.stem] = si_sdr(read_wav(CLEAN[p.stem]), q.astype(np.float64))
        audio_s += len(x) / SR
        if (i + 1) % 250 == 0:
            log(f'{i + 1}/{len(inputs)} clips, RTF {(time.time() - t) / audio_s:.4f}')
if zraw is not None:
    zraw.close()
(OUT / 'manifest.json').write_text(json.dumps({'input_slug': INPUT_SLUG, 'pp': PP, 'no_raw_fallback': NO_RAW_FALLBACK, 'zip_tta': ZIP_TTA, 'ckpt_kernel': CKPT_KERNEL, 'zip_ckpt_kernel': ZIP_CKPT_KERNEL, 'tfg_ckpt_kernel': TFG_CKPT_KERNEL, 'gate': GATE, 'enhancer': ENHANCER, 'route_a': ROUTE_A, 'route_b': ROUTE_B, 'route_thr': ROUTE_THR, 'rpt': [RPT_FIRST, RPT_SECOND, RPT_THR], 'ravg_thr': RAVG_THR, 'ravg_w': RAVG_W, 'ravg_edge': RAVG_EDGE, 'fuse_head': FUSE_HEAD, 'win': [WIN_MAX_S, WIN_S, WIN_OV_S], 'windowed_clips': sum(1 for f_ in man if f_.get('windowed')), 'fit_peak': FIT_PEAK, 'peak_fitted': n_fit,
                                              'device': str(DEV), 'clips': len(man), 'fails': fails,
                                              'audio_h': round(audio_s / 3600, 4), 'rtf': (time.time() - t) / max(audio_s, 1),
                                              'files': man}, indent=1))
if sis:
    (OUT / 'sisdr.json').write_text(json.dumps(sis))
    log(f'SI-SDR mean on {len(sis)} clips with clean references: {np.mean(list(sis.values())):.4f} dB')
log('DONE', len(man), 'clips, fails', fails, 'peak-fitted clips', n_fit)
