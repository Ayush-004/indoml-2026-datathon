#!/usr/bin/env python3
"""UniEnh: ESPnet "universal" speech enhancers for IndoML 2026 Track 2, without the espnet package.

Written in ASD-STE100 Simplified Technical English.

Models (weights CC-BY-4.0, code Apache-2.0, see README.md and SOURCES.json):
  tfgridnet_urgent25     kohei0209/tfgridnet_urgent25 (TF-GridNet v3, URGENT 2025 baseline), 8,524,352 params
  bsrnn_large_noncausal  wyz/vctk_dns2020_whamr_bsrnn_large_noncausal (BSRNN large), 63,133,324 params

Use:
    import sys; sys.path.insert(0, '/path/to/folder/with/infer.py')
    from infer import UniEnh
    enh = UniEnh('tfgridnet_urgent25', 'cuda')      # or 'bsrnn_large_noncausal', 'cpu'
    y = enh(x)       # x: float numpy (N,), in [-1, 1], 16 kHz mono -> y: float32 numpy (N,)
    q = np.clip(np.rint(y * 32768), -32768, 32767).astype(np.int16)

Step 1 (the official path, espnet2.bin.enh_inference.SeparateSpeech, whole clip by default): float32,
x / std(x) (unbiased std, "normalize_variance_per_ch"), STFT with the window and hop scaled to 16 kHz
(TF-GridNet 512/256, BSRNN 320/160, Hann, centre, reflect padding), separator, iSTFT with length N,
output * std(x). This is the "raw" output (SeparateSpeech with normalize_output_wav=False).

Step 2 (output level, one scalar gain per clip). Both models were trained with scale-invariant losses
(mr_l1_tfd with normalize_variance, SI-SNR), so the level of the raw output is arbitrary. For BSRNN the raw
output is 56 to 66 times the speech level of the input (raw peaks 4 to 65 for input peaks 0.99), and int16
rounding then clips it (SI-SDR falls to about 0 dB). The model cards therefore use
normalize_output_wav=True. Choices (output_level):
  'input' (default)  y * g, g = <x, y> / <y, y> (least-squares gain onto the noisy input; if g <= 0, the
                     RMS ratio). The output has the level of the speech in the input, the same as the
                     masking enhancers (FRCRN, MossFormerGAN). Blends with the noisy input and the
                     removed-energy router need this level.
  'peak'             y / max|y| * 0.9 (the official normalize_output_wav=True).
  'raw'              no gain (SeparateSpeech normalize_output_wav=False). BSRNN clips in int16.
  'rms'              y * rms(x) / rms(y).
SI-SDR does not change with the gain, only int16 rounding and clipping can change it.

Only torch, numpy and PyYAML are necessary. All paths are relative to this file. The Kaggle kernels
can import this file from the dataset folder.
"""
import glob
import hashlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from espnet_min import (  # noqa: E402
    build_model_from_file,
    enhance_tensor,
    separate_speech,
    set_bsrnn_groupnorm_eps,
    set_lstm_batch_chunk,
)

MODELS = {
    'tfgridnet_urgent25': {
        'folder': 'tfgridnet_urgent25',
        'config': 'config.yaml',
        'weights': 'valid.loss.ave_5best.pth',
        'hf_repo': 'kohei0209/tfgridnet_urgent25',
        'hf_commit': 'daad000927daa131e7692376a71c8144bcbfa6f8',
        'weights_sha256': '8350b6f84bb5de01646b7cebe9d19d5b1fc4318cd85c31d242caccc2442d276e',
        'params': 8524352,
    },
    'bsrnn_large_noncausal': {
        'folder': 'bsrnn_large_noncausal',
        'config': 'config.yaml',
        'weights': '97epoch.pth',
        'hf_repo': 'wyz/vctk_dns2020_whamr_bsrnn_large_noncausal',
        'hf_commit': '05f45f26b889822091410d9f1f2c1025f84d93bb',
        'weights_sha256': '15fe051146b4ea9e3a589f45c1bb12bb8699a380a01664f25e1dce0350ee93a3',
        'params': 63133324,
    },
}
ALIASES = {
    'tfgridnet_urgent25': 'tfgridnet_urgent25', 'tfgridnet': 'tfgridnet_urgent25',
    'urgent25': 'tfgridnet_urgent25', 'kohei0209/tfgridnet_urgent25': 'tfgridnet_urgent25',
    'bsrnn_large_noncausal': 'bsrnn_large_noncausal', 'bsrnn': 'bsrnn_large_noncausal',
    'bsrnn_large': 'bsrnn_large_noncausal',
    'wyz/vctk_dns2020_whamr_bsrnn_large_noncausal': 'bsrnn_large_noncausal',
}
# GroupNorm eps of the BSRNN BandSplit and MaskDecoder layers (see espnet_min.set_bsrnn_groupnorm_eps):
# 'train'  = 1e-5, the value of the code that trained the weights (github.com/Emrys365/se-scaling)
# 'espnet' = 1e-8, the value that ESPnet >= v.202409 uses when it loads the same weights
BSRNN_GN_EPS = {'train': 1e-5, 'espnet': 1e-8}
OUTPUT_LEVELS = ('input', 'peak', 'raw', 'rms')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def output_gain(x, y, mode):
    """Scalar gain for the raw output y of input x (see the module docstring). Returns float."""
    if mode == 'raw':
        return 1.0
    yd = np.asarray(y, dtype=np.float64)
    xd = np.asarray(x, dtype=np.float64)
    yy = float(np.dot(yd, yd))
    if yy <= 0.0 or not np.isfinite(yy):
        return 0.0
    if mode == 'peak':
        # the same float32 operation as SeparateSpeech (w / max|w| * 0.9) is done in apply_output_level
        return 0.9 / float(np.max(np.abs(yd)))
    rms_gain = float(np.sqrt(np.dot(xd, xd) / yy))
    if mode == 'rms':
        return rms_gain
    if mode == 'input':
        g = float(np.dot(xd, yd) / yy)
        return g if (np.isfinite(g) and g > 0.0) else rms_gain
    raise ValueError('output_level must be one of %s' % (OUTPUT_LEVELS,))


def apply_output_level(x, y, mode):
    """Return (float32 output at the chosen level, gain)."""
    y = np.asarray(y, dtype=np.float32)
    if mode == 'peak':
        m = np.max(np.abs(y))
        if m > 0:
            return (y / m * np.float32(0.9)).astype(np.float32), float(0.9 / m)
        return y, 1.0
    g = output_gain(x, y, mode)
    return (y.astype(np.float64) * g).astype(np.float32), g


def find_file(folder, fname, env=None):
    """Find folder/fname: $env, next to this file, below this file, up to 3 parents, /kaggle/input."""
    if env and os.environ.get(env) and os.path.isfile(os.environ[env]):
        return os.environ[env]
    for c in (os.path.join(HERE, folder, fname), os.path.join(HERE, fname)):
        if os.path.isfile(c):
            return c
    hits = sorted(glob.glob(os.path.join(HERE, '**', folder, fname), recursive=True))
    if hits:
        return hits[0]
    up = HERE
    for _ in range(3):
        up = os.path.dirname(up)
        c = os.path.join(up, folder, fname)
        if os.path.isfile(c):
            return c
    if os.path.isdir('/kaggle/input'):
        hits = sorted(glob.glob(os.path.join('/kaggle/input', '**', folder, fname), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError('%s/%s not found (searched next to %s and /kaggle/input)' % (folder, fname, HERE))


class UniEnh:
    """ESPnet universal enhancer. __call__(x) -> enhanced waveform of the same length.

    Args:
        name: 'tfgridnet_urgent25' or 'bsrnn_large_noncausal' (aliases: 'tfgridnet', 'bsrnn').
        device: 'cpu', 'cuda', 'cuda:0', ...
        weights, config: optional explicit paths (else they are found next to this file).
        output_level: 'input' (default), 'peak' (official normalize_output_wav=True), 'raw' or 'rms'.
        segment_size, hop_size: seconds; if both are set and the clip is longer than segment_size,
            use the segment-wise path of SeparateSpeech (overlap-add, average in the overlap).
            Default None = whole clip in one pass (the SeparateSpeech default). CAUTION: the output level
            of these models is arbitrary per pass, so the segments can get different levels; on our
            8.66 s check clip TF-GridNet fell from 45.1 dB (whole) to 21.1 dB (4 s / 3 s segments).
        normalize_segment_scale: SeparateSpeech option for the segment path (scale each segment output to
            the RMS of its input segment).
        bsrnn_gn_eps: 'train' (default, 1e-5) or 'espnet' (1e-8); only for the BSRNN model.
        lstm_chunk: None (default, normal nn.LSTM) or N = run each LSTM on at most N sequences at a time
            (same math, lower peak memory on CPU; see espnet_min.set_lstm_batch_chunk).
        check_sha256: True = compare the weight file with the known SHA-256 before loading.
    After each call: self.last_gain (the output gain) and self.last_raw_peak (max |raw output|).
    """

    def __init__(self, name, device='cpu', weights=None, config=None, output_level='input',
                 segment_size=None, hop_size=None, normalize_segment_scale=False, bsrnn_gn_eps='train',
                 lstm_chunk=None, check_sha256=False):
        key = ALIASES.get(name, name)
        if key not in MODELS:
            raise ValueError('unknown model %r; use one of %s' % (name, sorted(MODELS)))
        if output_level not in OUTPUT_LEVELS:
            raise ValueError('output_level must be one of %s' % (OUTPUT_LEVELS,))
        spec = MODELS[key]
        self.name = key
        self.device = str(device)
        self.fs = 16000
        self.output_level = output_level
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.normalize_segment_scale = bool(normalize_segment_scale)
        env = 'UNIENH_' + key.upper()
        self.config_path = config or find_file(spec['folder'], spec['config'], env + '_CONFIG')
        self.weights_path = weights or find_file(spec['folder'], spec['weights'], env + '_WEIGHTS')
        if check_sha256:
            got = sha256_file(self.weights_path)
            if got != spec['weights_sha256']:
                raise ValueError('SHA-256 mismatch for %s: %s' % (self.weights_path, got))
        self.model, self.train_args = build_model_from_file(self.config_path, self.weights_path, self.device)
        self.bsrnn_gn_eps = None
        if self.train_args['separator'] == 'bsrnn':
            self.set_bsrnn_gn_eps(bsrnn_gn_eps)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.lstm_chunk = None
        self.set_lstm_chunk(lstm_chunk)
        self.last_gain = None
        self.last_raw_peak = None

    def set_bsrnn_gn_eps(self, mode):
        """Change the BSRNN GroupNorm eps mode ('train' or 'espnet') of the loaded model."""
        if self.train_args['separator'] != 'bsrnn':
            raise ValueError('not a BSRNN model')
        if mode not in BSRNN_GN_EPS:
            raise ValueError('bsrnn_gn_eps must be one of %s' % sorted(BSRNN_GN_EPS))
        self.n_gn_changed = set_bsrnn_groupnorm_eps(self.model, BSRNN_GN_EPS[mode])
        self.bsrnn_gn_eps = mode

    def set_lstm_chunk(self, chunk):
        """None = normal nn.LSTM; N = at most N sequences per LSTM call (same math, less CPU memory)."""
        set_lstm_batch_chunk(self.model, chunk)
        self.lstm_chunk = chunk

    def enhance_raw(self, x, fs=16000):
        """Step 1 only: SeparateSpeech output with normalize_output_wav=False (float32, same length)."""
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError('x must be 1-D (mono), got shape %s' % (x.shape,))
        if fs != self.fs:
            raise ValueError('this wrapper expects 16 kHz input (got %s)' % fs)
        if x.size == 0 or not np.any(x):
            # all-zero input: SeparateSpeech would divide by std = 0 (NaN); give zeros
            return np.zeros(x.shape[0], dtype=np.float32)
        y = separate_speech(self.model, x[None, :], fs=fs, segment_size=self.segment_size,
                            hop_size=self.hop_size, normalize_segment_scale=self.normalize_segment_scale,
                            normalize_output_wav=False, device=self.device)[0][0]
        if y.shape[0] != x.shape[0]:
            raise RuntimeError('output length %d != input length %d' % (y.shape[0], x.shape[0]))
        return y.astype(np.float32, copy=False)

    def __call__(self, x, fs=16000, output_level=None):
        """x: 1-D float array in [-1, 1] (16 kHz mono). Returns float32 numpy array, same length."""
        y = self.enhance_raw(x, fs)
        self.last_raw_peak = float(np.max(np.abs(y))) if y.size else 0.0
        y, self.last_gain = apply_output_level(x, y, output_level or self.output_level)
        return y

    def forward_tensor(self, x, fs=16000):
        """Differentiable whole-clip path (step 1 only, raw level). x: (B, N) tensor -> (B, N)."""
        return enhance_tensor(self.model, x, fs=fs)

    def info(self):
        return {'name': self.name, 'hf_repo': MODELS[self.name]['hf_repo'],
                'hf_commit': MODELS[self.name]['hf_commit'], 'config': self.config_path,
                'weights': self.weights_path, 'params': self.n_params, 'device': self.device,
                'output_level': self.output_level, 'segment_size': self.segment_size,
                'hop_size': self.hop_size, 'normalize_segment_scale': self.normalize_segment_scale,
                'bsrnn_gn_eps': self.bsrnn_gn_eps,
                'lstm_chunk': self.lstm_chunk, 'torch': torch.__version__}


if __name__ == '__main__':
    # tiny self-test: python infer.py [name] [device]
    import time
    name = sys.argv[1] if len(sys.argv) > 1 else 'tfgridnet_urgent25'
    dev = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
    torch.manual_seed(0)
    enh = UniEnh(name, dev)
    x = (0.05 * torch.randn(16000 * 2)).numpy()
    t = time.time()
    y = enh(x)
    print(enh.info())
    print('in', x.shape, 'out', y.shape, y.dtype, 'finite', bool(np.isfinite(y).all()),
          'gain %.4g raw peak %.4g' % (enh.last_gain, enh.last_raw_peak), '%.2f s' % (time.time() - t))
