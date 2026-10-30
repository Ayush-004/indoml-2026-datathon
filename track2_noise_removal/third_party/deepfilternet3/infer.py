#!/usr/bin/env python3
"""DeepFilterNet3 (48 kHz) for IndoML 2026 Track 2, without the deepfilternet / deepfilterlib packages.

Written in ASD-STE100 Simplified Technical English.

Model: DeepFilterNet3 (H. Schroeter et al., Interspeech 2023), 48 kHz, 2,135,484 parameters. ERB-band gains
plus a deep filter (order 5) on the lowest 96 FFT bins. Weights: models/DeepFilterNet3.zip of
github.com/Rikorose/DeepFilterNet (commit d375b2d8), MIT OR Apache-2.0. Training data (paper and README):
DNS Challenge data (speech of several languages, noise, RIRs), full band 48 kHz.
Code: code/df/ = the unmodified model files of PyPI deepfilternet 0.5.6 (only df/__init__.py reduced),
code/libdf_np.py = a NumPy translation of the used DeepFilterLib 0.5.6 functions (STFT with Vorbis window,
ERB features, normalisations), code/shims/loguru = a logging stand-in (used only if loguru is missing).
Only torch, numpy and soxr (or scipy) are necessary.

Use:
    import sys; sys.path.insert(0, '/path/to/folder/with/infer.py')
    from infer import DeepFilterNet3
    enh = DeepFilterNet3('cpu')                    # full suppression; atten_lim_db=20 limits it to 20 dB
    y = enh.enhance(x)    # x: float64 (N,), int16-range values, 16 kHz mono -> y: float64 (N,), same scale
    q = np.clip(np.rint(y), -32768, 32767).astype(np.int16)

Path (df.enhance.enhance of deepfilternet 0.5.6, whole clip): 16 kHz -> 48 kHz (soxr HQ, the same
resampler as the MossFormer2 wrapper), float32 in [-1, 1], zero pad of n_fft (960) at the end,
libdf analysis (Vorbis window, n_fft 960, hop 480), ERB features (32 bands, dB, mean norm alpha 0.99) and
unit-normalised complex features (96 bins), network, optional attenuation limit (mix of the noisy and the
enhanced spectrum: lim = 10^(-atten/20)), libdf synthesis, drop n_fft - hop samples of delay, 48 -> 16 kHz.
The network output keeps the input scale (gains and filters), so output_level='raw' (default) needs no gain.
"""
import glob
import hashlib
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, 'code')
SHIMS = os.path.join(CODE, 'shims')
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if importlib.util.find_spec('loguru') is None and SHIMS not in sys.path:
    sys.path.insert(0, SHIMS)

import libdf_np  # noqa: E402

GH_REPO = 'Rikorose/DeepFilterNet'
GH_COMMIT = 'd375b2d8309e0935d165700c91da9de862a99c31'
CKPT_NAME = 'model_120.ckpt.best'
CKPT_BYTES = 8714073
CKPT_SHA256 = '23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003'
FS_IN = 16000
FS_MODEL = 48000
MAX_WAV_VALUE = 32768.0
OUTPUT_LEVELS = ('raw', 'input')
RESAMPLERS = ('soxr', 'poly', 'poly_sharp')
_SHARP_FIR = {}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _ok_dir(d):
    return (os.path.isfile(os.path.join(d, 'config.ini'))
            and os.path.isfile(os.path.join(d, 'checkpoints', CKPT_NAME))
            and os.path.getsize(os.path.join(d, 'checkpoints', CKPT_NAME)) == CKPT_BYTES)


def find_model_dir(weights=None):
    """Model folder (config.ini + checkpoints/model_120.ckpt.best): argument (the folder or the checkpoint
    file), $DFN3_DIR, next to this file (weights/DeepFilterNet3), up to 3 parent folders, /kaggle/input."""
    if weights:
        d = weights
        if os.path.isfile(d):
            d = os.path.dirname(os.path.dirname(os.path.abspath(d)))
        if not _ok_dir(d):
            raise FileNotFoundError('config.ini + checkpoints/%s (%d bytes) not found in %s' % (CKPT_NAME, CKPT_BYTES, d))
        return d
    env = os.environ.get('DFN3_DIR')
    cands = ([env] if env else []) + [os.path.join(HERE, 'weights', 'DeepFilterNet3'), os.path.join(HERE, 'DeepFilterNet3')]
    up = HERE
    for _ in range(3):
        up = os.path.dirname(up)
        cands += [os.path.join(up, 'DeepFilterNet3', 'weights', 'DeepFilterNet3'), os.path.join(up, 'DeepFilterNet3')]
    for d in cands:
        if d and _ok_dir(d):
            return d
    if os.path.isdir('/kaggle/input'):
        for p in sorted(glob.glob(os.path.join('/kaggle/input', '**', 'DeepFilterNet3', 'config.ini'), recursive=True)):
            if _ok_dir(os.path.dirname(p)):
                return os.path.dirname(p)
    raise FileNotFoundError('DeepFilterNet3 model folder not found; give weights=<folder>')


def _sharp_fir():
    if 'h' not in _SHARP_FIR:
        from scipy.signal import firwin
        _SHARP_FIR['h'] = firwin(961, 7800.0, fs=FS_MODEL, window=('kaiser', 10.0))
    return _SHARP_FIR['h']


def _resample(x, fs_from, fs_to, method):
    x = np.asarray(x, dtype=np.float64)
    if fs_from == fs_to:
        return x
    if method == 'soxr':
        import soxr  # quality 'HQ' = librosa res_type 'soxr_hq'
        return np.asarray(soxr.resample(x, fs_from, fs_to, quality='HQ'), dtype=np.float64)
    from scipy.signal import resample_poly
    g = np.gcd(fs_from, fs_to)
    if method == 'poly':
        return resample_poly(x, fs_to // g, fs_from // g)
    if method == 'poly_sharp':
        return resample_poly(x, fs_to // g, fs_from // g, window=_sharp_fir())
    raise ValueError('resampler must be one of %s' % (RESAMPLERS,))


class DeepFilterNet3:
    """DeepFilterNet3 enhancer. enhance(x) -> float64 array, same length, same scale as x.

    Args:
        device: 'cpu', 'cuda', ...
        weights: optional model folder (config.ini + checkpoints/) or the checkpoint file.
        atten_lim_db: None (default, full suppression) or a limit in dB (for example 20).
        post_filter: False (default) or True (the DeepFilterNet post-filter, mask_pf).
        resampler: 'soxr' (default, soxr HQ), 'poly' (scipy resample_poly, default filter) or 'poly_sharp'.
        backend: 'numpy' (default, code/libdf_np.py) or 'libdf' (the DeepFilterLib package, if installed).
        output_level: 'raw' (default) or 'input' (least-squares gain onto the noisy input).
        max_whole_s, overlap_s: None (default) = whole clip; a number = chunks with overlap-add (linear
            cross-fade over overlap_s seconds). Our clips are at most 18.4 s; whole clip is used.
    After each call: self.last_ls_gain (<x, y> / <y, y>), self.last_gain (applied gain).
    """

    def __init__(self, device='cpu', weights=None, atten_lim_db=None, post_filter=False, resampler='soxr',
                 backend='numpy', output_level='raw', max_whole_s=None, overlap_s=2.0, check_sha256=False):
        if output_level not in OUTPUT_LEVELS:
            raise ValueError('output_level must be one of %s' % (OUTPUT_LEVELS,))
        if resampler not in RESAMPLERS:
            raise ValueError('resampler must be one of %s' % (RESAMPLERS,))
        self.device = torch.device(device)
        self.atten_lim_db = atten_lim_db
        self.post_filter = bool(post_filter)
        self.resampler = resampler
        self.output_level = output_level
        self.max_whole_s = max_whole_s
        self.overlap_s = float(overlap_s)
        self.model_dir = find_model_dir(weights)
        self.ckpt_path = os.path.join(self.model_dir, 'checkpoints', CKPT_NAME)
        if check_sha256:
            got = sha256_file(self.ckpt_path)
            if got != CKPT_SHA256:
                raise ValueError('SHA-256 mismatch for %s: %s' % (self.ckpt_path, got))
        if backend == 'numpy':
            self.L = libdf_np
            if importlib.util.find_spec('libdf') is None:
                sys.modules['libdf'] = libdf_np   # the vendored model files import libdf (DF, unit_norm_init)
        elif backend == 'libdf':
            import libdf
            self.L = libdf
        else:
            raise ValueError("backend must be 'numpy' or 'libdf'")
        self.backend = backend
        from df.config import config
        config.load(os.path.join(self.model_dir, 'config.ini'), config_must_exist=True, allow_defaults=True,
                    allow_reload=True)
        from df.model import ModelParams
        if self.post_filter:
            config.set('mask_pf', True, bool, ModelParams().section)
        from df.deepfilternet3 import init_model
        from df.utils import get_norm_alpha
        p = ModelParams()
        self.p = p
        self.df_state = self.L.DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb,
                                  min_nb_erb_freqs=p.min_nb_freqs)
        self.alpha = get_norm_alpha(False)
        model = init_model(self.df_state, run_df=True, train_mask=True)
        sd = torch.load(self.ckpt_path, map_location='cpu', weights_only=True)
        sd = {k.replace('clc', 'df'): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)   # as df.checkpoint.read_cp
        self.missing_keys = list(missing)
        self.unexpected_keys = [k for k in unexpected if not (k.endswith('.h0') or 'erb_comp' in k)]
        if self.missing_keys or self.unexpected_keys:
            raise RuntimeError('checkpoint mismatch: missing %s unexpected %s' % (self.missing_keys, self.unexpected_keys))
        self.model = model.to(self.device).eval()
        self.nb_df = getattr(self.model, 'nb_df', getattr(self.model, 'df_bins', p.nb_df))
        self.n_params = sum(q.numel() for q in self.model.parameters())
        self.last_ls_gain = None
        self.last_gain = None

    # ------------------------------------------------------------------ 48 kHz core (df.enhance.enhance)
    @torch.no_grad()
    def enhance48(self, audio):
        """audio: float array (T,) in [-1, 1] at 48 kHz. Returns float32 numpy (T,) (pad=True path)."""
        from df.utils import as_complex, as_real
        L = self.L
        audio = torch.as_tensor(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        if hasattr(self.model, 'reset_h0'):
            self.model.reset_h0(batch_size=1, device=self.device)
        orig_len = audio.shape[-1]
        n_fft, hop = self.df_state.fft_size(), self.df_state.hop_size()
        audio = torch.nn.functional.pad(audio, (0, n_fft))
        spec = self.df_state.analysis(audio.numpy())
        erb_fb = self.df_state.erb_widths()
        erb_feat = torch.as_tensor(L.erb_norm(L.erb(spec, erb_fb), self.alpha)).unsqueeze(1)
        spec_feat = as_real(torch.as_tensor(L.unit_norm(spec[..., :self.nb_df], self.alpha)).unsqueeze(1))
        spec = as_real(torch.as_tensor(spec).unsqueeze(1))
        spec, erb_feat, spec_feat = spec.to(self.device), erb_feat.to(self.device), spec_feat.to(self.device)
        enhanced = self.model(spec.clone(), erb_feat, spec_feat)[0].cpu()
        enhanced = as_complex(enhanced.squeeze(1))
        if self.atten_lim_db is not None and abs(self.atten_lim_db) > 0:
            lim = 10 ** (-abs(self.atten_lim_db) / 20)
            enhanced = as_complex(spec.squeeze(1).cpu()) * lim + enhanced * (1 - lim)
        out = np.asarray(self.df_state.synthesis(enhanced.numpy()))
        assert n_fft % hop == 0
        d = n_fft - hop
        return out[0, d:orig_len + d].astype(np.float32)

    # ------------------------------------------------------------------ 16 kHz wrapper
    def _enhance_whole(self, x16):
        n = x16.shape[0]
        x48 = _resample(x16 / MAX_WAV_VALUE, FS_IN, FS_MODEL, self.resampler)
        y48 = self.enhance48(x48).astype(np.float64)
        y16 = _resample(y48, FS_MODEL, FS_IN, self.resampler)
        if y16.shape[0] < n:
            y16 = np.pad(y16, (0, n - y16.shape[0]))
        return y16[:n] * MAX_WAV_VALUE

    def _enhance_chunks(self, x16):
        n = x16.shape[0]
        c = int(round(self.max_whole_s * FS_IN))
        o = int(round(self.overlap_s * FS_IN))
        hop = c - o
        acc = np.zeros(n)
        wsum = np.zeros(n)
        for s in range(0, max(n - o, 1), hop):
            e = min(n, s + c)
            y = self._enhance_whole(x16[s:e])
            w = np.ones(e - s)
            if s > 0:
                k = min(o, e - s)
                w[:k] = np.linspace(0.0, 1.0, k + 2)[1:-1]
            if e < n:
                k = min(o, e - s)
                w[-k:] = np.minimum(w[-k:], np.linspace(1.0, 0.0, k + 2)[1:-1])
            acc[s:e] += w * y
            wsum[s:e] += w
            if e >= n:
                break
        return acc / np.maximum(wsum, 1e-12)

    def enhance(self, x):
        """x: 1-D float array of int16-range values (16 kHz mono, whole clip). Returns float64, same length."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError('x must be 1-D (mono), got shape %s' % (x.shape,))
        n = x.shape[0]
        if n == 0 or not np.any(x):
            self.last_ls_gain, self.last_gain = None, 1.0
            return np.zeros(n)
        if self.max_whole_s is not None and n > self.max_whole_s * FS_IN:
            y = self._enhance_chunks(x)
        else:
            y = self._enhance_whole(x)
        yy = float(np.dot(y, y))
        self.last_ls_gain = float(np.dot(x, y) / yy) if yy > 0 else None
        g = 1.0
        if self.output_level == 'input' and self.last_ls_gain is not None and self.last_ls_gain > 0:
            g = self.last_ls_gain
        self.last_gain = g
        return y * g

    __call__ = enhance

    def info(self):
        return {'model': 'DeepFilterNet3', 'github': GH_REPO, 'commit': GH_COMMIT, 'model_dir': self.model_dir,
                'params': self.n_params, 'device': str(self.device), 'atten_lim_db': self.atten_lim_db,
                'post_filter': self.post_filter, 'resampler': self.resampler, 'backend': self.backend,
                'output_level': self.output_level, 'alpha': self.alpha,
                'erb_widths': [int(v) for v in self.df_state.erb_widths()], 'torch': torch.__version__}


if __name__ == '__main__':
    import time
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    rng = np.random.default_rng(0)
    enh = DeepFilterNet3(dev)
    x = rng.normal(0, 1000, FS_IN * 2)
    t = time.time()
    y = enh.enhance(x)
    print(enh.info())
    print('in', x.shape, 'out', y.shape, y.dtype, 'finite', bool(np.isfinite(y).all()),
          'ls_gain', enh.last_ls_gain, '%.2f s' % (time.time() - t))
