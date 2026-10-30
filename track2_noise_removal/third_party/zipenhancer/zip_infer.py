"""ZipEnhancer (ModelScope iic/speech_zipenhancer_ans_multiloss_16k_base) in plain PyTorch, offline.

IndoML 2026 Track 2. No ModelScope import: the model code is vendored in code/zipenhancer_ms/
(Apache-2.0, see code/LICENSE and README.md).

    import sys; sys.path.insert(0, '<folder that holds zip_infer.py>')
    from zip_infer import ZipEnh, forward_train
    enh = ZipEnh('cpu')      # or 'cuda'; the weights are found automatically (see find_weights)
    y = enh(x)               # x: float numpy (T,), [-1, 1], 16 kHz mono -> y: float64 (T,)

Inference (ZipEnh.__call__) = the official ModelScope pipeline (ANSZipEnhancerPipeline):
    audio_norm (float64) -> float32 -> T < 2 s: zero pad to 2 s, one pass | 2 s <= T <= 6 s: one pass over
    the whole clip | T > 6 s: 2 s windows, 1.5 s hop, 0.25 s dropped at inner edges -> per window: unit-RMS
    norm, STFT (n_fft 400, hop 100, hann 400, center, magnitude ** 0.3), network, iSTFT, undo norm.
Differences from the official pipeline (all optional, defaults shown):
    fix_length=True    output has exactly T samples (official: 100 * floor(T / 100) samples for 2-6 s clips;
                       zeros in the last 4000 - pad samples for > 6 s clips when pad < 4000).
    level='input'      output at the input level (official: at the audio_norm level, int16 by truncation).
    attn_budget_mb     attention computed in batch pieces of at most this size ('auto': 128 MB on CPU,
                       512 MB on CUDA) - same numbers up to float rounding, less memory.
    conv_budget_mb     unpadded convolutions computed in frequency pieces with at most this im2col buffer
                       ('auto': 64 MB on CPU, off on CUDA) - same numbers up to float rounding, less memory.
    mode='official'    'whole' = one pass for any length (not official for T > 6 s; needs more memory; on the
                       Mac CPU faster than the 2 s windows and better on the one clip tested, see README).
"""
import glob
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(HERE, 'code')
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)  # torch.cuda.amp.custom_fwd in scaling.py
    from zipenhancer_ms import pipeline as P  # noqa: E402
    from zipenhancer_ms.zipenhancer import AttrDict, ZipEnhancer  # noqa: E402

SAMPLE_RATE = P.SAMPLE_RATE
WEIGHT_NAMES = ('zipenhancer_pytorch_model.bin', 'pytorch_model.bin')
OFFICIAL_WEIGHTS_SHA256 = 'b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f'
# "model" section of the official configuration.json (sha256 877861f9...), used when the file is absent
DEFAULT_CONFIG = {
    'dense_channel': 64,
    'num_tsconformers': 4,
    'former_name': 'Zipformer2DualPathEncoder',
    'former_conf': {
        'num_encoder_layers': [1, 1, 1, 1],
        'downsampling_factor': [1, 2, 2, 1],
        'f_downsampling_factor': [1, 2, 2, 1],
        'encoder_dim': [64, 64, 64, 64],
        'pos_dim': 24,
        'num_heads': 4,
        'query_head_dim': 12,
        'pos_head_dim': 4,
        'value_head_dim': 8,
        'feedforward_dim': [256, 256, 256, 256],
        'cnn_module_kernel': 15,
        'causal': False,
        'encoder_unmasked_dim': 64,
        'warmup_batches': 4000.0,
    },
    'batch_first': True,
    'model_num_spks': 1,
}


def _search_dirs():
    dirs = [HERE, os.path.join(HERE, 'weights')]
    d = HERE
    for _ in range(3):
        d = os.path.dirname(d)
        dirs += [d, os.path.join(d, 'weights')]
    return dirs


def find_weights(weights=None):
    """weights argument > $ZIPENH_WEIGHTS > next to this file (or weights/) > up to 3 parent
    folders > anywhere below /kaggle/input."""
    if weights:
        return str(weights)
    env = os.environ.get('ZIPENH_WEIGHTS')
    if env:
        return env
    for n in WEIGHT_NAMES:  # the specific name first, in all folders
        for d in _search_dirs():
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    if os.path.isdir('/kaggle/input'):
        for n in WEIGHT_NAMES:
            hits = sorted(glob.glob(os.path.join('/kaggle/input', '**', n), recursive=True))
            if hits:
                return hits[0]
    raise FileNotFoundError('ZipEnhancer weights not found (%s)' % ', '.join(WEIGHT_NAMES))


def load_config(weights_path=None):
    """The "model" part of configuration.json next to the weights, else DEFAULT_CONFIG."""
    cands = []
    if weights_path:
        cands.append(os.path.join(os.path.dirname(weights_path), 'configuration.json'))
    cands += [os.path.join(d, 'configuration.json') for d in _search_dirs()[:2]]
    for c in cands:
        if os.path.isfile(c):
            with open(c) as f:
                cfg = json.load(f).get('model', {})
            return {k: v for k, v in cfg.items() if k != 'type'}
    return dict(DEFAULT_CONFIG)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def extract_state_dict(ckpt):
    """Official file {'generator': sd}; ZipenhancerDecorator {'state_dict': 'model.*'}; or a plain sd."""
    if isinstance(ckpt, dict) and 'generator' in ckpt:
        return ckpt['generator']
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if all(k.startswith('model.') for k in ckpt):
        ckpt = {k[len('model.'):]: v for k, v in ckpt.items()}
    return ckpt


def build_model(weights=None, device='cpu', config=None):
    """ZipEnhancer network with the weights loaded (strict), in eval() mode, on device."""
    path = find_weights(weights)
    cfg = config or load_config(path)
    h = AttrDict(dict(num_tsconformers=cfg['num_tsconformers'], dense_channel=cfg['dense_channel'],
                      former_conf=cfg['former_conf'], batch_first=cfg['batch_first'],
                      model_num_spks=cfg['model_num_spks']))
    model = ZipEnhancer(h)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model.eval()
    return model.to(device), path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _norm_gains(x):
    """The two factors of the official audio_norm, same arithmetic: audio_norm(x) == (x * s1) * s2."""
    with np.errstate(all='ignore'):
        rms = (x**2).mean()**0.5
        s1 = 10**(-25 / 20) / rms
        p = (x * s1)**2
        m = p > p.mean()
        s2 = 10**(-25 / 20) / (p[m].mean()**0.5) if m.any() else 1.0
    return s1, s2


class ZipEnh:
    """ZipEnh(device)(x) -> enhanced waveform, same length as x (see the module docstring)."""

    def __init__(self, device='cpu', weights=None, fix_length=True, level='input',
                 attn_budget_mb='auto', conv_budget_mb='auto', mode='official'):
        assert level in ('input', 'official') and mode in ('official', 'whole')
        self.device = torch.device(device)
        self.model, self.weights_path = build_model(weights, self.device)
        self.weights_sha256 = sha256_file(self.weights_path)
        if self.weights_sha256 != OFFICIAL_WEIGHTS_SHA256:
            warnings.warn('ZipEnhancer weights are not the official file: %s' % self.weights_sha256)
        self.fix_length, self.level, self.mode = fix_length, level, mode
        cpu = self.device.type == 'cpu'
        self.set_budgets(attn_mb=(128 if cpu else 512) if attn_budget_mb == 'auto' else attn_budget_mb,
                         conv_mb=(64 if cpu else None) if conv_budget_mb == 'auto' else conv_budget_mb)
        self.n_params = count_parameters(self.model)

    def set_budgets(self, attn_mb=None, conv_mb=None):
        """Memory budgets in MB (None = upstream call, no split)."""
        self.attn_budget_mb, self.conv_budget_mb = attn_mb, conv_mb
        P.set_attention_budget(self.model, attn_mb)
        P.set_conv_budget(self.model, conv_mb)

    def enhance_normed(self, xn, nsamples=None, fix_length=None):
        """Network part of the official pipeline on an audio_norm-ed float32 array (T,) or (1, T)."""
        xn = np.asarray(xn, dtype=np.float32).reshape(1, -1)
        n = xn.shape[1] if nsamples is None else nsamples
        fix = self.fix_length if fix_length is None else fix_length
        if self.mode == 'whole':
            return P.whole_forward(self.model, xn, n, self.device)
        return P.pipeline_forward(self.model, xn, n, self.device, fix_length=fix)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n = x.shape[0]
        if n == 0:
            return x.copy()
        e = float(np.dot(x, x))
        if not np.isfinite(e) or e <= 0.0:
            return np.zeros(n) if np.isfinite(e) else x.copy()
        s1, s2 = _norm_gains(x)
        xn = ((x * s1) * s2).astype(np.float32)  # == official audio_norm(x).astype(np.float32)
        y = self.enhance_normed(xn, n)
        if self.level == 'input':
            y = y / (s1 * s2)
        if self.fix_length:
            assert y.shape[0] == n, (y.shape, n)
        return y


def forward_train(model, x_batch, fix_length=True, mode='official', window_batch=None):
    """Differentiable forward: (B, T) float tensor at any level -> (B, T) enhanced, at the input level.

    Same path as ZipEnh.__call__ (per-row norm, same windows) without audio_norm (the network input is
    normalised per window, so a global gain only scales the output) and without int16 rounding.
    T < 2 s is zero padded to 2 s (as upstream); 2-6 s is one pass; > 6 s uses the 2 s windows
    (window_batch = windows per network call, None = all at once).
    Keep model.eval(): the vendored ActivationDropoutAndLinear has no train-mode forward (it returns None);
    eval() only turns off dropout / layer skipping, InstanceNorm2d has no running statistics.
    """
    if model.training:
        raise RuntimeError('call model.eval() first: ZipEnhancer has no train-mode forward (see docstring)')
    return P.forward_batch(model, x_batch, fix_length=fix_length, mode=mode, window_batch=window_batch)
