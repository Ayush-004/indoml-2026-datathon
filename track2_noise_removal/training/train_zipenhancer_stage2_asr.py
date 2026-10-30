import os  # run defaults of stage 1 (stage 2 sets its own values with environment variables; see the README)
# ZipEnhancer stage 2 (G3-ASR): the exact file that trained the stage-2 model = the stage-1 file
# + the ASR-aware loss terms in the micro-batch path
os.environ.setdefault('T2_BEDS', 'clean')
os.environ.setdefault('T2_TARGET_G', '0.0')
os.environ.setdefault('T2_W_EVENT', '0.0')
os.environ.setdefault('T2_SEL_PP', 'pt20')
os.environ.setdefault('T2_W_HARD_SEL', '2.0')
os.environ.setdefault('T2_STOP_STEP', '1000')
os.environ.setdefault('T2_EVAL_VAL', '0')
os.environ.setdefault('T2_FINAL_NAT_STRIDE', '2')
os.environ.setdefault('T2_WISE_ALPHA', '0.5')
os.environ.setdefault('T2_BATCH', '4')
os.environ.setdefault('T2_CROP', '3.0')
os.environ.setdefault('T2_LR', '5e-5')
os.environ.setdefault('T2_STEPS', '2000')
os.environ.setdefault('T2_EVAL_EVERY', '500')
os.environ.setdefault('T2_BUDGET_H', '2.0')
os.environ.setdefault('T2_WORKERS', '3')
os.environ.setdefault('T2_MICRO', '1')
"""Track 2 fine-tuning of ZipEnhancer, stage 2 (SI-SNR + ASR-aware loss), started from the stage-1 model.

This file = train_zipenhancer_stage1.py + two ASR-aware loss terms in the micro-batch path: an L1 loss
between the SraVaani-1.0 front-end features of the output and of the target, and an L1 loss between the
outputs of the frozen SraVaani encoder on these features. During the first T2_CAL_STEPS batches only the
SI-SNR loss trains; then each term gets the weight T2_W_FE (T2_W_ENC) x mean |SI-SNR loss| / mean term of
these batches. With T2_W_FE = T2_W_ENC = 0 the file trains like the stage-1 file.
Data (built by training/data/): targets = clean speech beds (DeepFilterNet3, then a Silero VAD mask: pauses are
exact zeros, as in the organizer clean references); noise = Vaani noise events whose Silero VAD speech share is
< 0.2, procedural stationary noise (engine, machine, fan, rumble, hiss, tones, music; the generator is inlined
below), public noise (DEMAND, ESC-50 folds 1-4, MUSAN) and point tags; 1-3 events per mixture, event SNR ~ N(-8, 6) dB
clipped to [-20, 12]. Training audio = the public Vaani Noise-Event training release only (sessions of the organizer
validation set dropped, training/data/build_banks.py) + public and synthetic noise. No test data.
Selection: the organizer validation pairs at the val-hard level (clean + 1.68 x noise; SI-SDR + SraVaani WER of the
deployed output, eval subset SYN[::3]) and val-newlib (organizer clean speech + held-out public noise, SYN[::6]),
weighted 2:1; stop check at step T2_STOP_STEP; WiSE-FT candidate (0.5 init + 0.5 best) at the end.
Inputs: folders below T2_INPUT, found by name (find_input): vaani_inputs (or T2_LOCAL_VAANI = a folder with
manifest.jsonl + audio/), helpers (val_order.json), noise_bank, noise_bank_vad, clean_beds, public_noise, validation,
val_newlib, sravaani (a snapshot of ARTPARK-IISc/SraVaani-1.0) and the model package; see the README.
Outputs (T2_OUT): best.pt / last.pt / final.pt (state_dict), history.json, sisdr_*.json, hyps_*.jsonl,
wav_*_best.zip (raw outputs of the final model on val-hard, val-newlib and natural clips).
"""
import bisect
import importlib
import importlib.machinery
import json
import math
import os
import random
import subprocess
import sys
import time
import types
import wave
import zipfile
from pathlib import Path

import numpy as np

T0 = time.time()
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
E = os.environ.get
CFG = {
    'steps': int(E('T2_STEPS', 3000)),
    'batch': int(E('T2_BATCH', 4)),
    'crop_s': float(E('T2_CROP', 3.0)),
    'lr': float(E('T2_LR', 3e-5)),
    'weight_decay': 1e-5,
    'clip_grad': 5.0,
    'eval_every': int(E('T2_EVAL_EVERY', 500)),
    'eval_stride': 3,                 # periodic evals on SYN[::3] (232 clips); all 695 at the end
    'snr_mu': float(E('T2_SNR_MU', -8.0)), 'snr_sd': 6.0, 'snr_lo': -20.0, 'snr_hi': 12.0,
    'n_events': (1, 3),
    'p_src': json.loads(E('T2_P_SRC', '{"env": 0.15, "proc": 0.20, "pub": 0.40, "pts": 0.25}')),
    'proc_dur_s': (0.3, 3.0),
    'p_no_noise': 0.05,
    'fade_ms': 5.0,
    'silver_bed_hours': 14.0,
    'beds': E('T2_BEDS', 'clean'),            # 'clean' (DFN3 + VAD mask, training/data/build_clean_beds.py) | 'raw'
    'max_point_tags': int(E('T2_MAX_PTS', 6000)),
    'budget_h': float(E('T2_BUDGET_H', 1.6)),
    'target_g': float(E('T2_TARGET_G', 0.0)),   # 0 = full removal target; 0.1 = s + 0.1 n
    'w_event': float(E('T2_W_EVENT', 0.0)),     # in-event -SNR weight (0 = off)
    'sel_pp': E('T2_SEL_PP', 'pt20'),           # selection on the deployed output: '' | 'pt20' | 'full20'
    'w_hard_sel': float(E('T2_W_HARD_SEL', 2.0)),  # selection = (w val-hard + val-newlib) / (w + 1)
    'stop_step': int(E('T2_STOP_STEP', 1000)),  # stop check at the first eval >= this step (0 = off)
    'eval_val': E('T2_EVAL_VAL', '0') == '1',   # validation-level eval (not in the selection)
    'final_nat_stride': int(E('T2_FINAL_NAT_STRIDE', 2)),  # final stage natural clips NAT[::k] (0 = none)
    'final_force': E('T2_FINAL_FORCE', '0') == '1',  # smoke tests only: run the final stage even if best = step 0
    'wise_alpha': float(E('T2_WISE_ALPHA', 0.5)),  # final stage: WiSE-FT candidate (1-a) init + a best (0 = off)
    'w_fe': float(E('T2_W_FE', 0.0)),           # ASR-aware: front-end L1 share of |SI-SNR loss| (0 = off)
    'w_enc': float(E('T2_W_ENC', 0.0)),         # ASR-aware: frozen-encoder L1 share (0 = off)
    'cal_steps': int(E('T2_CAL_STEPS', 20)),    # batches that calibrate lam_fe / lam_enc (SI-SNR only there)
    'swoosh_fix': E('T2_SWOOSH_FIX', '1') == '1',  # SwooshL/R forward via softplus (finite gradients; same forward)
    'grad_sanitize': E('T2_GRAD_SANITIZE', '1') == '1',  # zero non-finite gradient entries before clipping
    'seed': int(E('T2_SEED', 1234)),
    'hard_gain': 1.68,                          # val-hard: clean + 1.68 x noise (+4.5 dB), peak rule 32,440
    'workers': int(E('T2_WORKERS', 3)),
    'micro': int(E('T2_MICRO', 0)),             # >0: gradient accumulation, micro-batches of this many clips (memory)
    'final_eval': E('T2_FINAL', '1') == '1',
    'probe': E('T2_PROBE', '0') == '1',         # speed probe: no evaluations, a few steps, timing only
}
SR = 16000
INIT_KERNEL = E('T2_INIT_KERNEL', '')            # start from another run's best.pt
OUT = Path(E('T2_OUT', '/kaggle/working'))
TMP = Path(E('T2_TMP', '/tmp/t2'))
IN_ROOT = Path(E('T2_INPUT', '/kaggle/input'))
LIMIT_VAL = int(E('T2_LIMIT_VAL', 0))
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
random.seed(CFG['seed'])
np.random.seed(CFG['seed'])


def log(*a):
    print(f'[{time.time() - T0:8.1f}s]', *a, flush=True)


def find_input(slug):
    for base in (IN_ROOT,):
        if (base / slug).exists():
            return base / slug
    hits = [p for p in IN_ROOT.rglob(slug) if p.is_dir()]
    if hits:
        return hits[0]
    raise SystemExit(f'input {slug} not found')


LOCAL_VAANI = E('T2_LOCAL_VAANI')        # local Vaani copy: <folder with manifest.jsonl + audio/>
LIMIT_BANK = int(E('T2_LIMIT_BANK', 0))  # local smoke test only
VA = Path(LOCAL_VAANI) if LOCAL_VAANI else find_input('vaani_inputs')
EN = find_input('helpers')
BK = find_input('noise_bank')
VD = find_input('validation')
SV = find_input('sravaani')
ZD = find_input('zipenhancer')
log('inputs', VA, EN, BK, VD, SV, ZD)
log('cfg', json.dumps(CFG))

whls = sorted(str(p) for p in EN.glob('*.whl'))
for mod, key in (('sentencepiece', 'sentencepiece'),):
    try:
        importlib.import_module(mod)
    except ImportError:
        w = [x for x in whls if Path(x).name.startswith(key)]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', *w], check=True)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

DEV = torch.device(E('T2_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))   # 'mps' on an Apple M-series Mac
if DEV.type == 'cpu':
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - (1 if CFG['workers'] > 0 else 0)))
torch.manual_seed(CFG['seed'])
log('torch', torch.__version__, DEV, torch.cuda.get_device_name(0) if torch.cuda.is_available() else
    f'cpu threads {torch.get_num_threads()}')


def _unz(root, zip_name, marker):
    z = root / zip_name
    if z.exists() and z.is_file():
        d = TMP / zip_name.replace('.zip', '')
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        return next(d.rglob(marker)).parent
    return next(root.rglob(marker)).parent


sys.path.insert(0, str(_unz(ZD, 'zipenhancer_pkg.zip', 'zip_infer.py')))
from zip_infer import build_model, _norm_gains  # noqa: E402
from zipenhancer_ms import pipeline as ZP  # noqa: E402
import zipenhancer_ms.zipenhancer_layers.scaling as _ZSC  # noqa: E402


def _swoosh_l_safe(x):
    return torch.nn.functional.softplus(x - 4.0) - 0.08 * x - 0.035


def _swoosh_r_safe(x):
    return torch.nn.functional.softplus(x - 1.0) - 0.08 * x - 0.313261687


if CFG['swoosh_fix']:   # the vendored log(1+exp) + torch.where gives NaN gradients for z > ~88.7
    _ZSC.SwooshLForward, _ZSC.SwooshRForward = _swoosh_l_safe, _swoosh_r_safe

# ---------------------------------------------------------------- procedural noise (inlined generator)
# ---- begin inlined procedural noise generator

import numpy as np
from scipy.signal import butter, sosfilt

SR = 16000
KINDS = ('engine', 'machine', 'fan', 'rumble', 'hiss', 'tones', 'music')


def _unit(x):
    x = x - np.mean(x)
    r = np.sqrt(np.mean(x ** 2)) + 1e-12
    return (x / r).astype(np.float32)


def _shaped_noise(n, rng, env_fn):
    """White noise shaped in the frequency domain by env_fn(f) (amplitude)."""
    m = 1 << int(np.ceil(np.log2(max(n, 256))))
    X = rng.standard_normal(m // 2 + 1) + 1j * rng.standard_normal(m // 2 + 1)
    f = np.fft.rfftfreq(m, 1 / SR)
    X *= env_fn(f)
    return np.fft.irfft(X, m)[:n]


def _slow(n, rng, rate_hz, depth):
    """Slow random amplitude envelope, 1 +- depth, band-limited to about rate_hz."""
    k = max(2, int(n / SR * rate_hz) + 2)
    pts = rng.uniform(-1, 1, k)
    env = np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
    return np.clip(1 + depth * env, 0.05, None)


def _harmonics(n, rng, f0, n_harm, decay, glide=0.0, jitter=0.0):
    t = np.arange(n) / SR
    f = f0 * (1 + glide * np.interp(t, [0, t[-1] + 1e-9], rng.uniform(-1, 1, 2)))
    if jitter:
        f = f * (1 + jitter * _slow(n, rng, 3, 1.0) - jitter)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.zeros(n)
    for k in range(1, n_harm + 1):
        if k * f0 > SR / 2 - 200:
            break
        y += (k ** -decay) * rng.uniform(0.3, 1.0) * np.sin(k * ph + rng.uniform(0, 2 * np.pi))
    return y


def _bp(x, lo, hi, order=2):
    lo, hi = max(lo, 20.0), min(hi, SR / 2 - 100)
    if hi <= lo * 1.05:
        return x
    return sosfilt(butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x)


def _lp(x, fc, order=4):
    return sosfilt(butter(order, min(fc, SR / 2 - 100), btype='low', fs=SR, output='sos'), x)


def engine(n, rng):
    f0 = rng.uniform(22, 95)
    h = _harmonics(n, rng, f0, int(rng.integers(8, 40)), rng.uniform(0.3, 1.2), glide=rng.uniform(0, 0.3),
                   jitter=rng.uniform(0, 0.03))
    nz = _lp(_shaped_noise(n, rng, lambda f: 1 / np.sqrt(np.maximum(f, 20))), rng.uniform(250, 1500))
    y = _unit(h) * rng.uniform(0.3, 1.0) + _unit(nz) * rng.uniform(0.3, 1.0)
    return _unit(y * _slow(n, rng, rng.uniform(0.3, 4), rng.uniform(0, 0.5)))


def machine(n, rng):
    f0 = rng.choice([50, 60, 100, 120]) if rng.random() < 0.3 else rng.uniform(40, 400)
    y = _unit(_harmonics(n, rng, f0, int(rng.integers(3, 25)), rng.uniform(0.2, 1.5), jitter=rng.uniform(0, 0.01)))
    if rng.random() < 0.6:   # whine
        y += _unit(_harmonics(n, rng, rng.uniform(600, 5000), int(rng.integers(1, 4)), 1.0,
                              glide=rng.uniform(0, 0.1))) * rng.uniform(0.1, 0.8)
    if rng.random() < 0.7:   # broadband part
        lo = rng.uniform(80, 1500)
        y += _unit(_bp(rng.standard_normal(n), lo, lo * rng.uniform(2, 8))) * rng.uniform(0.2, 1.0)
    if rng.random() < 0.35:  # clatter / grinding impulses
        rate = rng.uniform(4, 60)
        imp = (rng.random(n) < rate / SR).astype(float) * rng.uniform(0.5, 1.5, n)
        dec = np.exp(-np.arange(int(0.02 * SR)) / (rng.uniform(0.001, 0.008) * SR))
        cl = np.convolve(imp, dec)[:n] * rng.standard_normal(n)
        y += _unit(_bp(cl, rng.uniform(300, 2000), rng.uniform(3000, 7500))) * rng.uniform(0.2, 1.0)
    return _unit(y * _slow(n, rng, rng.uniform(0.2, 3), rng.uniform(0, 0.3)))


def fan(n, rng):
    slope = rng.uniform(0.0, 1.2)   # amplitude ~ f^-slope/2
    y = _unit(_shaped_noise(n, rng, lambda f: np.maximum(f, 30) ** (-slope / 2)))
    for _ in range(int(rng.integers(0, 4))):
        c = rng.uniform(100, 3000)
        y += _unit(_bp(rng.standard_normal(n), c / 1.15, c * 1.15, order=2)) * rng.uniform(0.2, 1.0)
    if rng.random() < 0.6:   # blade-pass tone
        y += _unit(_harmonics(n, rng, rng.uniform(20, 200), int(rng.integers(2, 12)), 1.0)) * rng.uniform(0.1, 0.7)
    return _unit(y * _slow(n, rng, rng.uniform(0.2, 2), rng.uniform(0, 0.2)))


def rumble(n, rng):
    y = _shaped_noise(n, rng, lambda f: 1 / np.maximum(f, 15))
    y = _lp(y, rng.uniform(80, 700))
    return _unit(y * _slow(n, rng, rng.uniform(0.2, 3), rng.uniform(0, 0.6)))


def hiss(n, rng):
    k = int(rng.integers(5, 14))
    lf = np.linspace(np.log(40), np.log(7900), k)
    ldb = np.cumsum(rng.normal(0, 6, k))
    y = _shaped_noise(n, rng, lambda f: 10 ** (np.interp(np.log(np.maximum(f, 40)), lf, ldb) / 20))
    return _unit(y * _slow(n, rng, rng.uniform(0.2, 6), rng.uniform(0, 0.4)))


def tones(n, rng):
    y = np.zeros(n)
    for _ in range(int(rng.integers(1, 4))):
        d = int(min(n, rng.uniform(0.1, 1.5) * SR))
        if d < 400:
            continue
        a = int(rng.integers(0, n - d + 1))
        f0 = rng.uniform(250, 1200)
        seg = _harmonics(d, rng, f0, int(rng.integers(1, 12)), rng.uniform(0.2, 1.5), glide=rng.uniform(0, 0.05))
        if rng.random() < 0.3:   # two-tone horn / siren
            seg += _harmonics(d, rng, f0 * rng.uniform(1.1, 1.5), int(rng.integers(1, 8)), 1.0)
        r = min(int(0.01 * SR), d // 4)
        w = np.ones(d)
        w[:r] = np.linspace(0, 1, r)
        w[-r:] = np.linspace(1, 0, r)
        y[a:a + d] += seg * w
    if not np.any(y):
        y = _harmonics(n, rng, rng.uniform(250, 1200), 4, 1.0)
    return _unit(y)


def music(n, rng):
    y = np.zeros(n)
    base = rng.uniform(110, 440)
    scale = np.array([0, 2, 4, 5, 7, 9, 11, 12])
    rate = rng.uniform(1, 5)
    t = 0
    while t < n:
        d = int(SR / rate * rng.uniform(0.5, 1.5))
        notes = base * 2 ** (rng.choice(scale, size=int(rng.integers(1, 4))) / 12) * rng.choice([0.5, 1, 2])
        e = min(n, t + int(d * rng.uniform(1.0, 2.5)))
        L = e - t
        if L > 200:
            env = np.exp(-np.arange(L) / (rng.uniform(0.1, 0.8) * SR)) if rng.random() < 0.6 else np.ones(L)
            for f in notes:
                y[t:e] += env * _harmonics(L, rng, f, int(rng.integers(2, 10)), rng.uniform(0.8, 2.0))
        t += d
    if rng.random() < 0.5:  # percussion
        imp = np.zeros(n)
        imp[::max(1, int(SR / (rate * rng.choice([1, 2]))))] = 1
        y += _unit(np.convolve(imp, np.exp(-np.arange(800) / 120.0) * rng.standard_normal(800))[:n]) * 0.4
    return _unit(y)


GEN = {'engine': engine, 'machine': machine, 'fan': fan, 'rumble': rumble, 'hiss': hiss, 'tones': tones,
       'music': music}


def gen(kind, n, rng):
    return GEN[kind](int(n), rng)


# ---- end inlined procedural noise generator

# ---------------------------------------------------------------- Vaani audio access (layout of the inputs dataset)
def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


if LOCAL_VAANI:
    ROW = {r['clip_id']: r['audio_path'] for r in read_jsonl(VA / 'manifest.jsonl')}

    def vaani_path(cid):
        return VA / ROW[cid]
else:
    contract = json.loads((VA / 'full-input-contract.json').read_text())
    training = read_jsonl(VA / 'training.jsonl')
    shards, cursor = [], 0
    for entry in contract['archives']:
        if 'training_clips' not in entry:
            continue
        stem = Path(entry['path']).stem
        rel = training[cursor]['audio_path']
        prefix = next(p for p in ('', stem) if (VA / p / rel).is_file())
        shards.append((cursor + entry['training_clips'], prefix))
        cursor += entry['training_clips']
    ENDS = [e for e, _ in shards]
    ROW = {r['clip_id']: i for i, r in enumerate(training)}

    def vaani_path(cid):
        i = ROW[cid]
        return VA / shards[bisect.bisect_right(ENDS, i)][1] / training[i]['audio_path']


# ---------------------------------------------------------------- banks
noise = read_jsonl(BK / 'noise_bank.jsonl')
crop = int(CFG['crop_s'] * SR)
if CFG['beds'] == 'clean':
    # clean beds (training/data/build_clean_beds.py): DeepFilterNet3 + Silero VAD mask, int16 shards
    CB = find_input('clean_beds')
    beds = [b for b in read_jsonl(CB / 'index.jsonl') if b['length'] >= crop]
    SHARDS = {n: np.load(CB / n, mmap_mode='r') for n in sorted({b['shard'] for b in beds})}
else:
    beds = read_jsonl(BK / 'speech_beds.jsonl')
    beds = [b for b in beds if (b['end'] - b['start']) >= CFG['crop_s'] and b['clip_id'] in ROW]
    gold = [b for b in beds if b['tier'] == 'ver']
    silver = [b for b in beds if b['tier'] == 'unv']
    random.shuffle(silver)
    keep, acc = [], 0.0
    for b in silver:
        if acc >= CFG['silver_bed_hours'] * 3600:
            break
        keep.append(b)
        acc += b['end'] - b['start']
    beds = gold + keep
def held_district(cid):
    """Hold-out districts of an optional proxy set (only with T2_HOLDOUT=1; not used): never a training noise source."""
    import hashlib as _hl
    p = cid.split('_')
    return int(_hl.sha1((p[4] if len(p) > 4 else cid).encode()).hexdigest(), 16) % 5 == 0


_n0 = len(noise)
_VADS = {}
for _r in read_jsonl(find_input('noise_bank_vad') / 'noise_bank_vad.jsonl'):
    _VADS[_r['i']] = _r['vad_speech']
for _i, _n in enumerate(noise):
    _n['vad_speech'] = _VADS.get(_i)
_VMAX = float(os.environ.get('T2_VAD_MAX', 0.2))
_HOLD = os.environ.get('T2_HOLDOUT', '0') == '1'
noise = [n for n in noise if n['clip_id'] in ROW and (n['end'] - n['start']) >= 0.05
         and n['vad_speech'] is not None and n['vad_speech'] < _VMAX and not (_HOLD and held_district(n['clip_id']))]
log(f'noise events after the length / VAD < {_VMAX} / hold-out ({_HOLD}) filter: {len(noise)} of {_n0}; '
    f'kinds {dict((k, sum(1 for n in noise if n["kind"] == k)) for k in ("empty_wrap", "point_tag"))}')
pts_all = [n for n in noise if n['kind'] == 'point_tag']
random.shuffle(pts_all)
noise = [n for n in noise if n['kind'] != 'point_tag'] + pts_all[:CFG['max_point_tags']]
if LIMIT_BANK:
    random.shuffle(beds)
    beds = beds[:LIMIT_BANK]
    random.shuffle(noise)
    noise = noise[:LIMIT_BANK * 4]
log(f'beds ({CFG["beds"]}) {len(beds)}, noise events {len(noise)}')

SPANS = {}


def read_clip(cid):
    # float read, then int16 units: exact for the Kaggle FLAC PCM_16 files, and correct for float WAV copies
    # (a dtype='int16' read of a FLOAT WAV would truncate every sample to 0 / +-1)
    x, sr = sf.read(str(vaani_path(cid)), dtype='float32', always_2d=False)
    assert sr == SR
    x = x if x.ndim == 1 else x[:, 0]
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)


t = time.time()
need = {}
if CFG['beds'] != 'clean':
    for i, b in enumerate(beds):
        need.setdefault(b['clip_id'], []).append(('bed', i, b['start'], b['end']))
for i, n in enumerate(noise):
    need.setdefault(n['clip_id'], []).append(('noise', i, n['start'], n['end']))
for j, (cid, items) in enumerate(need.items()):
    x = read_clip(cid)
    for kind, i, a, b in items:
        seg = x[int(a * SR):int(b * SR)].copy()
        if len(seg):
            SPANS[(kind, i)] = seg
    if (j + 1) % 10000 == 0:
        log(f'read {j + 1} of {len(need)} clips')
log(f'spans in RAM: {len(SPANS)}, {sum(len(v) for v in SPANS.values()) / SR / 3600:.2f} h, {time.time() - t:.0f}s')
if CFG['beds'] == 'clean':
    beds_idx = list(range(len(beds)))
else:
    beds_idx = [i for i in range(len(beds)) if ('bed', i) in SPANS and len(SPANS[('bed', i)]) >= crop]


def span(kind, i):
    if kind == 'bed' and CFG['beds'] == 'clean':
        b = beds[i]
        return np.asarray(SHARDS[b['shard']][b['offset']:b['offset'] + b['length']], dtype=np.float32) / 32768.0
    return SPANS[(kind, i)].astype(np.float32) / 32768.0


SRC = {'env': [i for i, n in enumerate(noise) if n['kind'] == 'empty_wrap' and ('noise', i) in SPANS],
       'pts': [i for i, n in enumerate(noise) if n['kind'] == 'point_tag' and ('noise', i) in SPANS]}
log(f'noise: environmental {len(SRC["env"])}, point tags {len(SRC["pts"])}, procedural kinds {KINDS}')
PUB = {}
try:
    PD = find_input('public_noise')      # DEMAND / ESC-50 / MUSAN (training/data/build_public_noise.py)
    for r in read_jsonl(PD / 'index.jsonl'):
        PUB.setdefault(r['source'], []).append(r)
    PUBSH = {n: np.load(PD / n, mmap_mode='r') for n in sorted({r['shard'] for v in PUB.values() for r in v})}
    log('public noise:', {k: len(v) for k, v in PUB.items()})
except SystemExit:
    log('public noise dataset not attached')
PUB_KEYS = sorted(PUB)
_PW = json.loads(os.environ.get('T2_PUB_W', '{"esc50": 0.50, "musan_noise": 0.25, "demand": 0.15, "musan_music": 0.10}'))
PUB_W = [float(_PW.get(k, 0.05)) for k in PUB_KEYS]
PUB_W = [w / sum(PUB_W) for w in PUB_W] if PUB_W else []
if PUB_KEYS:
    log('public noise source weights', dict(zip(PUB_KEYS, [round(w, 3) for w in PUB_W])))
P_SRC = {k: v for k, v in CFG['p_src'].items() if v > 0 and (k == 'proc' or (k == 'pub' and PUB) or SRC.get(k))}
log('noise source weights', P_SRC)


def pub_segment(rng, n_max):
    e = PUB[rng.choices(PUB_KEYS, weights=PUB_W)[0]]
    r = e[rng.randrange(len(e))]
    d = min(n_max, int(SR * math.exp(rng.uniform(math.log(CFG['proc_dur_s'][0]), math.log(4.0)))))
    best, be = None, -1.0
    for _ in range(4):                      # loudest of 4 offsets (no silent segments)
        o = rng.randint(0, max(0, r['length'] - d))
        seg = np.asarray(PUBSH[r['shard']][r['offset'] + o:r['offset'] + o + min(d, r['length'])], dtype=np.float32)
        en = float(np.dot(seg, seg))
        if en > be:
            best, be = seg, en
    return best / 32768.0


class MixSet(torch.utils.data.IterableDataset):
    """(x, target, s): x = s + sum of events; target = s + g * events (g = CFG target_g)."""

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        seed = CFG['seed'] + (wi.id if wi else 0) * 7919 + int(time.time())
        rng = random.Random(seed)
        nrng = np.random.default_rng(seed)
        fade = int(CFG['fade_ms'] * SR / 1000)
        keys, probs = list(P_SRC), np.array(list(P_SRC.values()), float)
        probs /= probs.sum()
        while True:
            bi = rng.choice(beds_idx)
            full = span('bed', bi)
            o = rng.randint(0, len(full) - crop)
            s = full[o:o + crop]
            ps = float(np.mean(s ** 2)) + 1e-10
            n_mix = np.zeros(crop, np.float32)
            if rng.random() >= CFG['p_no_noise']:
                for _ in range(rng.randint(*CFG['n_events'])):
                    src = keys[int(nrng.choice(len(keys), p=probs))]
                    if src == 'pub':
                        seg = pub_segment(rng, crop)
                    elif src == 'proc':
                        d = int(SR * math.exp(rng.uniform(math.log(CFG['proc_dur_s'][0]), math.log(CFG['proc_dur_s'][1]))))
                        seg = gen(KINDS[rng.randrange(len(KINDS))], min(d, crop), nrng)
                    else:
                        ei = rng.choice(SRC[src])
                        if noise[ei]['clip_id'] == beds[bi]['clip_id']:
                            continue
                        seg = span('noise', ei)
                    if len(seg) < 400:
                        continue
                    if len(seg) > crop:
                        o2 = rng.randint(0, len(seg) - crop)
                        seg = seg[o2:o2 + crop]
                    seg = np.array(seg, np.float32)
                    pn = float(np.mean(seg ** 2)) + 1e-10
                    snr = min(CFG['snr_hi'], max(CFG['snr_lo'], rng.gauss(CFG['snr_mu'], CFG['snr_sd'])))
                    seg = seg * math.sqrt(ps / (pn * 10 ** (snr / 10)))
                    k = min(fade, len(seg) // 4)
                    if k > 0:
                        ramp = np.linspace(0, 1, k, dtype=np.float32)
                        seg[:k] *= ramp
                        seg[-k:] *= ramp[::-1]
                    pos = rng.randint(0, crop - len(seg))
                    n_mix[pos:pos + len(seg)] += seg
            x = s + n_mix + 1e-6 * nrng.standard_normal(crop).astype(np.float32)   # -120 dBFS: no exact zeros
            gain = 10 ** (rng.uniform(-12, 6) / 20) * 0.1 / (float(np.sqrt(np.mean(x ** 2))) + 1e-8)
            peak = float(np.max(np.abs(x))) * gain
            if peak > 0.99:
                gain *= 0.99 / peak
            tgt = s + CFG['target_g'] * n_mix
            if CFG['w_event'] > 0:
                _ev = np.convolve((np.abs(n_mix) > 0).astype(np.float32), np.ones(641, np.float32), 'same') > 0
                yield (torch.from_numpy((x * gain).astype(np.float32)), torch.from_numpy((tgt * gain).astype(np.float32)),
                       torch.from_numpy(_ev.astype(np.float32)))
                continue
            yield (torch.from_numpy((x * gain).astype(np.float32)), torch.from_numpy((tgt * gain).astype(np.float32)))


def si_snr_loss(est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) * ref / ((ref ** 2).sum(-1, keepdim=True) + eps)
    e = est - s
    return -(10 * torch.log10((s ** 2).sum(-1) / ((e ** 2).sum(-1) + eps) + eps)).mean()


def event_loss(est, ref, m, eps=1e-8):
    """-SNR inside the event regions (m = 1) with the whole-crop projection scale; per item, clipped to [-30, 60] dB."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + eps)
    t = a * ref
    e2 = (((est - t) * m) ** 2).sum(-1)
    t2 = ((t * m) ** 2).sum(-1)
    has = (m.sum(-1) > 0) & (t2 > 1e-6 * (t ** 2).sum(-1))
    if not bool(has.any()):
        return est.new_zeros(())
    snr = (10 * torch.log10((t2 + eps) / (e2 + eps))).clamp(-30.0, 60.0)
    return -snr[has].mean()


# ---- SraVaani front end + frozen-encoder mimic (differentiable copy of the SraVaani-1.0 front end; matches the model code to 1e-6)
class SraVaaniFrontend(torch.nn.Module):
    def __init__(self, preproc_path, log_floor=None):
        super().__init__()
        pp = torch.load(preproc_path, map_location='cpu', weights_only=False)
        self.p = pp['params']
        self.register_buffer('window', pp['window'].float(), persistent=False)
        self.register_buffer('fb', pp['fb'].float(), persistent=False)          # (1, 128, 257)
        self.log_floor = log_floor

    def forward(self, wav):
        """wav: (B, T) float -> features (B, 128, frames), all frames valid (same length per batch)."""
        p = self.p
        x = torch.cat((wav[:, :1], wav[:, 1:] - p['preemph'] * wav[:, :-1]), dim=1)
        spec = torch.stft(x, n_fft=p['n_fft'], hop_length=p['hop_length'], win_length=p['win_length'],
                          window=self.window, center=True, pad_mode='constant', return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2                                   # mag_power 2
        mel = torch.matmul(self.fb.to(power.dtype), power)                       # (B, 128, frames)
        logmel = torch.log(mel + p['log_zero_guard_value'])
        if self.log_floor is not None:
            logmel = torch.clamp(logmel, min=self.log_floor)
        seq_len = wav.shape[1] // p['hop_length']
        logmel = logmel[..., :seq_len]
        mean = logmel.mean(dim=2, keepdim=True)
        std = torch.sqrt(((logmel - mean) ** 2).sum(dim=2, keepdim=True) / (seq_len - 1.0)) + p['CONSTANT']
        return (logmel - mean) / std


def frontend_l1(fe, y, s):
    return (fe(y) - fe(s)).abs().mean()


class EncoderMimic(torch.nn.Module):
    """Frozen SraVaani encoder (TorchScript) as a feature extractor for a mimic loss."""

    def __init__(self, ts_path, device):
        super().__init__()
        m = torch.jit.load(str(ts_path), map_location='cpu').eval()   # G3-ASR: MPS cannot hold the file's float64 constants; the float32 encoder goes to DEV with .to()
        self.encoder = m.encoder.float()
        for q in self.encoder.parameters():
            q.requires_grad_(False)

    def forward(self, feats):
        lengths = torch.full((feats.shape[0],), feats.shape[-1], dtype=torch.long, device=feats.device)
        enc, _ = self.encoder(feats, lengths)
        return enc


# ---------------------------------------------------------------- validation (official SI-SDR) and val-hard
VAL = _unz(VD, 'validation.zip', 'validationMetadata.json')
META = {Path(r['segmentFileName']).stem: r for r in json.loads((VAL / 'validationMetadata.json').read_text())}
ORDER = json.loads((EN / 'val_order.json').read_text())
SYN = [Path(n).stem for n in ORDER['syn_noisy']]
if LIMIT_VAL:
    SYN = SYN[:LIMIT_VAL]
SYN_EVAL = SYN[::CFG['eval_stride']]
NAT = [Path(n).stem for n in ORDER['nat_noisy']]
NAT = NAT[::CFG['final_nat_stride']] if CFG['final_nat_stride'] > 0 else []
if LIMIT_VAL:
    NAT = NAT[:LIMIT_VAL]


def read_wav(p):
    with wave.open(str(p), 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)


def write_wav(p, y_int_units):
    q = np.clip(np.rint(y_int_units), -32768, 32767).astype('<i2')
    with wave.open(str(p), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(q.tobytes())


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


VAL_X = {k: read_wav(VAL / 'syntheticNoiseAudio' / f'{k}.wav') for k in SYN}
VAL_S = {k: read_wav(VAL / 'syntheticCleanRefAudio' / f'{k}.wav') for k in SYN}


def _hard(k):
    y = VAL_S[k] + CFG['hard_gain'] * (VAL_X[k] - VAL_S[k])
    pk = float(np.max(np.abs(y)))
    return np.rint(y * (32440.0 / pk)) if pk > 32440.0 else np.rint(y)


VAL_XH = {k: _hard(k) for k in SYN}
UNZ = _unz

# ---------------------------------------------------------------- extra proxy set (val-newlib)
EXTRA = {}
for _tg, _slug, _zip, _lvl in (('new', 'val_newlib', 'valnewlib.zip', 'newlib_hard'),):
    try:
        _root = UNZ(find_input(_slug), _zip, _lvl)
        EXTRA[_tg] = {k: read_wav(_root / _lvl / 'syntheticNoiseAudio' / f'{k}.wav') for k in SYN}
        log(f'extra proxy set {_tg} ({_lvl}): {len(EXTRA[_tg])} clips')
    except (SystemExit, StopIteration, FileNotFoundError) as _e:
        log(f'extra proxy set {_tg} not used:', type(_e).__name__, _e)
SYN_EVAL_X = SYN[::int(os.environ.get('T2_EVAL_STRIDE_X', 6))]
EVWX = {_tg: TMP / f'eval_{_tg}_wav' for _tg in EXTRA}
for _d in EVWX.values():
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- model
model, wpath = build_model(str(next(ZD.rglob('zipenhancer_pytorch_model.bin'))), 'cpu')
if INIT_KERNEL:
    ckp = next(find_input(INIT_KERNEL).rglob('best.pt'))
    model.load_state_dict(torch.load(ckp, map_location='cpu'))
    log('init from', ckp)
model = model.to(DEV).eval()          # eval(): the vendored layers have no train-mode forward; no BatchNorm here
INIT_SD = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # for the WiSE-FT candidate
for p_ in model.parameters():
    p_.requires_grad_(True)
log('ZipEnhancer parameters', sum(p_.numel() for p_ in model.parameters()))


@torch.no_grad()
def enhance(x_int):
    """Official ZipEnhancer path in whole-clip mode: audio_norm gains -> one pass -> input level."""
    x = x_int / 32768.0
    n = len(x)
    if n == 0 or float(np.dot(x, x)) <= 0:
        return np.zeros(n)
    s1, s2 = _norm_gains(x)
    xn = ((x * s1) * s2).astype(np.float32)[None]
    y = ZP.whole_forward(model, xn, n, DEV) / (s1 * s2)
    if DEV.type == 'mps':
        torch.mps.empty_cache()   # variable clip lengths: do not keep cached blocks of every size
    return np.asarray(y, dtype=np.float64).reshape(-1)[:n] * 32768.0


# ---------------------------------------------------------------- PP (the same code as post_pp of inference/enhance.py)
PP_L = 256


def _pp_smooth(mf, n):
    gm = np.convolve(np.repeat(mf.astype(float), PP_L), np.ones(81) / 81, mode='same')
    o = np.zeros(n)
    o[:len(gm)] = gm[:n]
    return o


def post_pp(y, x, gate, T, t1=30.0, t2=20.0):
    """Passthrough: frames where 10log10(|g x|^2/|g x - y|^2) > T take g x (g = LS gain over the least-changed loud
    frames), 5 ms fades. With the quiet-frame option, frames > t1 dB under the output p95 frame where > t2 dB was
    removed are first set to zero (not used)."""
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


def apply_pp(y, x):
    if not CFG['sel_pp']:
        return y
    return np.clip(np.rint(post_pp(y, x, gate=CFG['sel_pp'].startswith('full'),
                                   T=float(CFG['sel_pp'].lstrip('ptful')))[0]), -32768, 32767)


EVAL_LAST = {}


def evaluate(hard=False, keys=None, wav_dir=None, src=None, tag=None):
    """Mean SI-SDR of the DEPLOYED output (enhancer + PP if CFG['sel_pp']); per-clip raw / deployed values in
    EVAL_LAST[tag]; the WAVs written for the ASR proxy are the deployed output."""
    ZP.set_attention_budget(model, 512 if DEV.type in ('cuda', 'mps') else 256)
    raw, dep = {}, {}
    src = src if src is not None else (VAL_XH if hard else VAL_X)
    for k in (keys if keys is not None else SYN):
        y = np.clip(np.rint(enhance(src[k])), -32768, 32767)
        raw[k] = si_sdr(VAL_S[k], y)
        y = apply_pp(y, src[k])
        dep[k] = si_sdr(VAL_S[k], y)
        if wav_dir is not None:
            write_wav(wav_dir / f'{k}.wav', y)
    ZP.set_attention_budget(model, None)
    if tag:
        EVAL_LAST[tag] = {'raw': raw, 'dep': dep}
    return float(np.mean(list(dep.values())))


import re as _re  # noqa: E402
import unicodedata as _ud  # noqa: E402

_TAG = _re.compile(r"</?[^<>]*>|\[[^\[\]]*\]")


def norm_text(text):
    """Verbatim copy of the official text normaliser."""
    t = _TAG.sub(" ", text or "")
    t = "".join(" " if _ud.category(c).startswith("P") else c for c in t)
    return " ".join(t.lower().split())


def _edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)]


def pooled_wer(keys, hyp):
    e = n = 0
    for k in keys:
        ref = norm_text(META[k]['transcript']).split()
        if not ref:
            continue
        h = norm_text(hyp.get(k, '')).split() or ['@']
        e += _edits(ref, h)
        n += len(ref)
    return e / max(n, 1)


def asr_texts(wav_dir, keys):
    from transformers import AutoModel
    if DEV.type == 'mps':
        torch.mps.empty_cache()   # free the cached MPS blocks before the CPU ASR loads
    asr = AutoModel.from_pretrained(str(SV), trust_remote_code=True).to('cpu' if DEV.type == 'mps' else DEV).eval()
    out = {}
    with torch.no_grad():
        for s0 in range(0, len(keys), 64):
            part = keys[s0:s0 + 64]
            hyps = asr.transcribe([str(wav_dir / f'{k}.wav') for k in part], batch_size=8, return_hypotheses=True)
            for k, h in zip(part, hyps):
                out[k] = h.text.strip()
    del asr
    import gc
    gc.collect()
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()
    if DEV.type == 'mps':
        torch.mps.empty_cache()
    return out


EVW = TMP / 'eval_hard_wav'
EVW.mkdir(parents=True, exist_ok=True)
HIST = {'cfg': CFG, 'init_kernel': INIT_KERNEL, 'eval': [], 'loss': []}


def proxy_eval(step):
    t = time.time()
    v = evaluate(keys=SYN_EVAL, tag='val') if CFG['eval_val'] else float('nan')
    vh = evaluate(hard=True, keys=SYN_EVAL, wav_dir=EVW, tag='hard')
    wh = pooled_wer(SYN_EVAL, asr_texts(EVW, SYN_EVAL))
    pxh = vh + 32.03 * (HIST['wer_hard_noisy'] - wh)          # SI-SDR + 0.3203 x dWER-syn (pp)
    rec = {'step': step, 'si_sdr': v, 'si_sdr_hard': vh, 'wer_hard': wh, 'proxy_hard': pxh,
           'si_sdr_hard_raw': float(np.mean(list(EVAL_LAST['hard']['raw'].values())))}
    if CFG['eval_val']:
        rec['si_sdr_raw'] = float(np.mean(list(EVAL_LAST['val']['raw'].values())))
    if 'hard_keys' not in HIST:      # hard clips: raw val-hard SI-SDR of the step-0 model < 20 dB
        HIST['hard_keys'] = [k for k in SYN_EVAL if EVAL_LAST['hard']['raw'][k] < 20.0]
    hk = HIST['hard_keys']
    rec['si_sdr_hard_hardclips'] = float(np.mean([EVAL_LAST['hard']['dep'][k] for k in hk])) if hk else 0.0
    pxs, ws, msg = [pxh], [CFG['w_hard_sel']], ''
    for tg, src_x in EXTRA.items():
        vx = evaluate(keys=SYN_EVAL_X, wav_dir=EVWX[tg], src=src_x, tag=tg)
        wx = pooled_wer(SYN_EVAL_X, asr_texts(EVWX[tg], SYN_EVAL_X))
        pxx = vx + 32.03 * (HIST[f'wer_{tg}_noisy'] - wx)
        pxs.append(pxx)
        ws.append(1.0)
        rec.update({f'si_sdr_{tg}': vx, f'wer_{tg}': wx, f'proxy_{tg}': pxx,
                    f'si_sdr_{tg}_raw': float(np.mean(list(EVAL_LAST[tg]['raw'].values())))})
        msg += f'; {tg} {vx:.4f} dB, WER {100 * wx:.2f} % (noisy {100 * HIST[f"wer_{tg}_noisy"]:.2f}), proxy {pxx:.4f}'
    px = float(np.dot(ws, pxs) / np.sum(ws))
    rec.update({'proxy': px, 'sec': time.time() - t})
    HIST['eval'].append(rec)
    log(f'step {step}: [PP {CFG["sel_pp"] or "off"}] val-hard raw {rec["si_sdr_hard_raw"]:.4f} dB, '
        f'hard clips ({len(hk)}) {rec["si_sdr_hard_hardclips"]:.4f} dB; '
        f'validation SI-SDR {v:.4f} dB, val-hard {vh:.4f} dB, val-hard WER {100 * wh:.2f} % '
        f'(noisy {100 * HIST["wer_hard_noisy"]:.2f}), proxy {pxh:.4f}{msg}; SELECTION {px:.4f} ({time.time() - t:.0f}s)')
    return px


opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 150) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / CFG['steps'])))))

if not CFG['probe']:
    NZW = TMP / 'noisy_hard_wav'
    NZW.mkdir(parents=True, exist_ok=True)
    for k in SYN_EVAL:
        write_wav(NZW / f'{k}.wav', VAL_XH[k])
    HIST['wer_hard_noisy'] = pooled_wer(SYN_EVAL, asr_texts(NZW, SYN_EVAL))
    log(f'val-hard noisy WER on the eval subset: {100 * HIST["wer_hard_noisy"]:.2f} %')
    for _tg, _src in EXTRA.items():
        _nz = TMP / f'noisy_{_tg}_wav'
        _nz.mkdir(parents=True, exist_ok=True)
        for k in SYN_EVAL_X:
            write_wav(_nz / f'{k}.wav', _src[k])
        HIST[f'wer_{_tg}_noisy'] = pooled_wer(SYN_EVAL_X, asr_texts(_nz, SYN_EVAL_X))
        log(f'{_tg} noisy WER on the eval subset: {100 * HIST[f"wer_{_tg}_noisy"]:.2f} %')
    best = proxy_eval(0)
    torch.save(model.state_dict(), OUT / 'best.pt')
    HIST['best_step'] = 0
else:
    best = -1e9

FE = SraVaaniFrontend(str(SV / 'preproc.pt')).to(DEV) if (CFG['w_fe'] > 0 or CFG['w_enc'] > 0) else None
ENC = EncoderMimic(SV / 'model-asr.fp16.ts', DEV).to(DEV) if CFG['w_enc'] > 0 else None
LAM, CAL = {'fe': 0.0, 'enc': 0.0}, {'si': [], 'fe': [], 'enc': []}
if FE is not None:
    log(f'ASR-aware loss on: w_fe {CFG["w_fe"]}, w_enc {CFG["w_enc"]}, calibration {CFG["cal_steps"]} batches')
NW = CFG['workers']
loader = torch.utils.data.DataLoader(MixSet(), batch_size=CFG['batch'], num_workers=NW, persistent_workers=NW > 0)
it = iter(loader)
run = 0.0
BAD = {'loss': 0, 'grad': 0}
HIST['bad_steps'] = BAD
MICRO = CFG['micro']
if MICRO:
    assert CFG['w_event'] == 0, 'T2_MICRO supports SI-SNR (+ the ASR-aware terms) only'
    log(f'gradient accumulation: micro-batches of {MICRO} clip(s) per step of {CFG["batch"]} '
        '(same gradient as one pass up to rounding)')


def train_forward(xb):
    """Training forward of one (micro-)batch: whole clip, same math as inference."""
    return ZP.forward_batch(model, xb, fix_length=True, mode='whole')


def micro_backward(x, tgt, step):
    """Batch loss back-propagated micro-batch by micro-batch (same gradient as one pass up to float rounding: no batch
    coupling, every term is a mean over items). Loss = SI-SNR, + the ASR-aware terms (G3-ASR) after the calibration
    batches. Returns the batch loss, or None when a micro-batch loss is not finite (the step is then skipped)."""
    opt.zero_grad(set_to_none=True)
    n = x.shape[0]
    tot = torch.zeros((), device=x.device)
    cal = {'si': 0.0, 'fe': 0.0, 'enc': 0.0}
    for j0 in range(0, n, MICRO):
        xb, tb = x[j0:j0 + MICRO], tgt[j0:j0 + MICRO]
        w = xb.shape[0] / n
        est = train_forward(xb)
        l_ = si_snr_loss(est, tb)
        if FE is not None:
            cal['si'] += abs(float(l_)) * w
            if step <= CFG['cal_steps']:
                with torch.no_grad():
                    fy, fs = FE(est), FE(tb)
                    cal['fe'] += float((fy - fs).abs().mean()) * w
                    if ENC is not None:
                        cal['enc'] += float((ENC(fy) - ENC(fs)).abs().mean()) * w
            else:
                fy = FE(est)
                with torch.no_grad():
                    fs = FE(tb)
                    e_t = ENC(fs) if ENC is not None else None
                l_fe = (fy - fs).abs().mean()
                l_enc = (ENC(fy) - e_t).abs().mean() if ENC is not None else torch.zeros((), device=DEV)
                cal['fe'] += float(l_fe) * w
                cal['enc'] += float(l_enc) * w
                l_ = l_ + LAM['fe'] * l_fe + LAM['enc'] * l_enc
        l_ = l_ * w
        if not torch.isfinite(l_):
            opt.zero_grad(set_to_none=True)
            return None
        l_.backward()
        tot = tot + l_.detach()
    if FE is not None:
        if step % 50 == 0 or step <= CFG['cal_steps']:
            HIST.setdefault('asr_terms', []).append([step, round(cal['si'], 4), round(cal['fe'], 5), round(cal['enc'], 5)])
        if step <= CFG['cal_steps']:
            CAL['si'].append(cal['si']); CAL['fe'].append(cal['fe']); CAL['enc'].append(cal['enc'])
            if step == CFG['cal_steps']:
                si_m = float(np.mean(CAL['si']))
                LAM['fe'] = CFG['w_fe'] * si_m / max(float(np.mean(CAL['fe'])), 1e-6)
                LAM['enc'] = CFG['w_enc'] * si_m / max(float(np.mean(CAL['enc'])), 1e-6) if ENC is not None else 0.0
                HIST['lambda'] = {**LAM, 'calib': {k: float(np.mean(v)) for k, v in CAL.items()}}
                log(f'ASR-aware loss weights: {HIST["lambda"]}')
    return tot


t_steps = time.time()
for step in range(1, CFG['steps'] + 1):
    batch = next(it)
    x, tgt = batch[0].to(DEV), batch[1].to(DEV)
    if MICRO:
        loss = micro_backward(x, tgt, step)
        if loss is None:
            BAD['loss'] += 1
            sched.step()
            continue
    else:
        est = ZP.forward_batch(model, x, fix_length=True, mode='whole')
        loss = si_snr_loss(est, tgt)
        if FE is not None:
            l_si = float(loss)
            fy = FE(est)
            with torch.no_grad():
                fs = FE(tgt)
                e_t = ENC(fs) if ENC is not None else None
            l_fe = (fy - fs).abs().mean()
            l_enc = (ENC(fy) - e_t).abs().mean() if ENC is not None else torch.zeros((), device=DEV)
            if step <= CFG['cal_steps']:
                CAL['si'].append(abs(l_si)); CAL['fe'].append(float(l_fe)); CAL['enc'].append(float(l_enc))
                if step == CFG['cal_steps']:
                    si_m = float(np.mean(CAL['si']))
                    LAM['fe'] = CFG['w_fe'] * si_m / max(float(np.mean(CAL['fe'])), 1e-6)
                    LAM['enc'] = CFG['w_enc'] * si_m / max(float(np.mean(CAL['enc'])), 1e-6) if ENC is not None else 0.0
                    HIST['lambda'] = {**LAM, 'calib': {k: float(np.mean(v)) for k, v in CAL.items()}}
                    log(f'ASR-aware loss weights: {HIST["lambda"]}')
            else:
                loss = loss + LAM['fe'] * l_fe + LAM['enc'] * l_enc
        if CFG['w_event'] > 0:
            loss = loss + CFG['w_event'] * event_loss(est, tgt, batch[2].to(DEV))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            BAD['loss'] += 1
            sched.step()
            continue
        loss.backward()
    if CFG['grad_sanitize']:
        _nf = 0
        for p_ in model.parameters():
            if p_.grad is not None and not bool(torch.isfinite(p_.grad).all()):
                _nf += 1
                torch.nan_to_num_(p_.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if _nf:
            BAD['sanitized'] = BAD.get('sanitized', 0) + 1
            BAD['sanitized_params_max'] = max(BAD.get('sanitized_params_max', 0), _nf)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
    if not torch.isfinite(gn):                   # skip a step with a non-finite gradient (BiasNorm on flat frames)
        BAD['grad'] += 1
        opt.zero_grad(set_to_none=True)
        sched.step()
        if BAD['grad'] + BAD['loss'] > 50:
            log('too many non-finite steps', BAD)
            break
        continue
    opt.step()
    sched.step()
    run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
    if step in (5, 20) or (CFG['probe'] and step % 5 == 0):
        mem = f', GPU max {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB' if DEV.type == 'cuda' else ''
        log(f'step {step}: {(time.time() - t_steps) / step:.2f} s/step{mem}, loss {float(loss):.3f}, grad norm {float(gn):.2f}')
    if step % 100 == 0:
        HIST['loss'].append({'step': step, 'loss_ema': run, 'loss': float(loss), 'grad_norm': float(gn),
                             's_per_step': (time.time() - t_steps) / step})
        log(f'step {step} loss(ema) {run:.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t_steps) / step:.2f} s/step')
    if not CFG['probe'] and (step % CFG['eval_every'] == 0 or step == CFG['steps']):
        v = proxy_eval(step)
        if v > best:
            best = v
            torch.save(model.state_dict(), OUT / 'best.pt')
            HIST['best_step'] = step
            log(f'new best proxy {best:.4f} at step {step}')
        torch.save(model.state_dict(), OUT / 'last.pt')
        (OUT / 'history.json').write_text(json.dumps(HIST, indent=1))
        if CFG['stop_step'] and step >= CFG['stop_step'] and 'stop_check' not in HIST:
            e0, e1 = HIST['eval'][0], HIST['eval'][-1]
            g_sel = e1['proxy'] - e0['proxy']
            g_hard = e1['si_sdr_hard_hardclips'] - e0['si_sdr_hard_hardclips']
            HIST['stop_check'] = {'step': step, 'selection_gain': g_sel, 'hard_clip_gain_db': g_hard}
            log(f'stop check at step {step}: selection gain {g_sel:+.4f}, hard-clip gain {g_hard:+.4f} dB')
            if g_sel <= 0.2 and g_hard < 0.3:
                HIST['early_stop'] = step
                log('EARLY STOP (rule: selection gain <= 0.2 and hard-clip gain < 0.3 dB)')
                (OUT / 'history.json').write_text(json.dumps(HIST, indent=1))
                break
    if (time.time() - T0) / 3600 > CFG['budget_h']:
        log('budget reached at step', step)
        if not CFG['probe'] and step % CFG['eval_every'] != 0:
            v = proxy_eval(step)
            if v > best:
                best = v
                torch.save(model.state_dict(), OUT / 'best.pt')
                HIST['best_step'] = step
                log(f'new best proxy {best:.4f} at step {step}')
            torch.save(model.state_dict(), OUT / 'last.pt')
        break
(OUT / 'history.json').write_text(json.dumps(HIST, indent=1))
if CFG['probe'] or not CFG['final_eval']:
    log('DONE (probe / no final eval)', best)
    raise SystemExit(0)

# ---------------------------------------------------------------- final: best checkpoint (raw + deployed PP output)
if HIST.get('best_step', 0) == 0 and not INIT_KERNEL and not CFG['final_force']:
    log('best checkpoint = step 0 (the pretrained model): final stage skipped')
    raise SystemExit(0)
_sd_best = torch.load(OUT / 'best.pt', map_location='cpu')
_final_sd, HIST['final_model'] = _sd_best, 'best'
if CFG['wise_alpha'] > 0 and (HIST.get('best_step', 0) != 0 or CFG['final_force']) and not CFG['probe']:  # force: smoke
    _a = CFG['wise_alpha']
    _sd_w = {k: ((1 - _a) * INIT_SD[k] + _a * v) if v.is_floating_point() else v for k, v in _sd_best.items()}
    model.load_state_dict(_sd_w)
    _vw = proxy_eval(-1)                     # WiSE-FT candidate, same proxy and selection rule
    _vb = max(e['proxy'] for e in HIST['eval'] if e['step'] >= 0 and e['step'] == HIST['best_step'])
    HIST['wise'] = {'alpha': _a, 'selection': _vw, 'best_selection': _vb}
    log(f'WiSE-FT alpha {_a}: selection {_vw:.4f} vs best checkpoint (step {HIST["best_step"]}) {_vb:.4f}')
    if _vw > _vb:
        _final_sd, HIST['final_model'] = _sd_w, f'wise{_a}'
torch.save(_final_sd, OUT / 'final.pt')
(OUT / 'history.json').write_text(json.dumps(HIST, indent=1))
log('final model:', HIST['final_model'])
model.load_state_dict({k: v.to(DEV) for k, v in _final_sd.items()})
ZP.set_attention_budget(model, 512 if DEV.type in ('cuda', 'mps') else 256)
SETS = [('hard', VAL_XH, SYN)] + [(t, x, SYN) for t, x in EXTRA.items()]
if CFG['eval_val']:
    SETS.append(('val', VAL_X, SYN))
if NAT:
    NAT_X = {k: read_wav(VAL / 'naturalNoisyAudio' / f'{k}.wav') for k in NAT}
    SETS.append(('nat', NAT_X, NAT))
DIRS = {}
for tg, src, keys in SETS:
    d_raw, d_pp = TMP / f'enh_best_{tg}', TMP / f'enh_best_{tg}_pp'
    d_raw.mkdir(parents=True, exist_ok=True)
    d_pp.mkdir(parents=True, exist_ok=True)
    DIRS[tg] = (d_raw, d_pp, keys)
    v_raw, v_pp = {}, {}
    for k in keys:
        y = np.clip(np.rint(enhance(src[k])), -32768, 32767)
        write_wav(d_raw / f'{k}.wav', y)
        yp = apply_pp(y, src[k]) if CFG['sel_pp'] else np.clip(np.rint(post_pp(y, src[k], gate=False, T=20.0)[0]), -32768, 32767)
        write_wav(d_pp / f'{k}.wav', yp)
        if tg != 'nat':
            v_raw[k] = si_sdr(VAL_S[k], y)
            v_pp[k] = si_sdr(VAL_S[k], yp)
    if tg != 'nat':
        (OUT / f'sisdr_{tg}_best.json').write_text(json.dumps(v_raw))
        (OUT / f'sisdr_{tg}_best_pp.json').write_text(json.dumps(v_pp))
        log(f'best checkpoint: {tg} SI-SDR raw {np.mean(list(v_raw.values())):.4f} dB, '
            f'PP {np.mean(list(v_pp.values())):.4f} dB ({len(keys)} clips)')
    log(f'final {tg}: {len(keys)} clips enhanced')
for tg, (d_raw, d_pp, keys) in DIRS.items():
    for suf, d in (('', d_raw), ('_pp', d_pp)):
        tx = asr_texts(d, keys)
        with open(OUT / f'hyps_{tg}_best{suf}.jsonl', 'w') as f:
            for k in keys:
                f.write(json.dumps({'clip_id': k, 'text': tx[k]}, ensure_ascii=False) + '\n')
        log(f'hyps_{tg}_best{suf}.jsonl', 'WER', round(100 * pooled_wer(keys, tx), 3))
    with zipfile.ZipFile(OUT / f'wav_{tg}_best.zip', 'w', zipfile.ZIP_STORED) as z:
        for k in keys:
            z.write(d_raw / f'{k}.wav', f'{k}.wav')
log('DONE best', best, 'step', HIST.get('best_step'), 'final model', HIST.get('final_model'))
