"""NumPy copy of the DeepFilterLib 0.5.6 Python API subset that DeepFilterNet3 inference uses.

Written in ASD-STE100 Simplified Technical English.

Source: PyPI DeepFilterLib 0.5.6 sdist (sha256 38fa2334c5b19385ca1f747ca9636f8016119005b3a12046d56ddc054a3807e5),
files src/lib.rs (the Python bindings) and local_dependencies/deep_filter/src/{lib.rs, transforms.rs}
(Rust, MIT OR Apache-2.0, Copyright Hendrik Schroeter). This file translates these Rust functions to NumPy:
  DFState::new (Vorbis window, wnorm, erb_fb), frame_analysis, frame_synthesis (with state reset per call,
  as the Python binding does by default), compute_band_corr + 10*log10(x + 1e-10) (erb, db=True),
  band_mean_norm_erb (erb_norm), band_unit_norm (unit_norm), unit_norm_init.
The recursions run in float32, as in Rust. The FFTs run in float64 (Rust: f32), then the result is cast to
float32. The differences to the Rust library are float rounding only (see parity_df.json).
[IndoML-T2 change]: new file (a translation), not an upstream file.
"""
import math

import numpy as np

MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)


_C = np.float32(24.7) * np.float32(9.265)   # f32 product, as the Rust constant expression


def _freq2erb(freq_hz):
    return np.float32(9.265) * np.log1p(np.float32(freq_hz) / _C, dtype=np.float32)


def _erb2freq(n_erb):
    return _C * (np.exp(np.float32(n_erb) / np.float32(9.265), dtype=np.float32) - np.float32(1.0))


def _linspace_f32(a, b, n):
    """ndarray Array1::linspace for f32: a + step * i, step = (b - a) / (n - 1), all in f32."""
    a = np.float32(a)
    step = (np.float32(b) - a) / np.float32(n - 1)
    return (a + step * np.arange(n, dtype=np.float32)).astype(np.float32)


def erb_fb_widths(sr, fft_size, nb_bands, min_nb_freqs):
    """libDF erb_fb(): number of FFT bins per ERB band (f32 math as in Rust)."""
    nyq_freq = sr // 2
    freq_width = np.float32(sr) / np.float32(fft_size)
    erb_low = _freq2erb(0.0)
    erb_high = _freq2erb(float(nyq_freq))
    erb = [0] * nb_bands
    step = (erb_high - erb_low) / np.float32(nb_bands)
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = _erb2freq(erb_low + np.float32(i) * step)
        fb = int(np.round(np.float32(f / freq_width)))  # Rust f32::round: half away from zero
        v = float(f / freq_width)
        if abs(v - math.floor(v) - 0.5) < 1e-12:
            fb = int(math.floor(v) + 1) if v > 0 else int(math.ceil(v) - 1)
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        erb[i - 1] = nb_freqs
        prev_freq = fb
    erb[nb_bands - 1] += 1
    too_large = sum(erb) - (fft_size // 2 + 1)
    if too_large > 0:
        erb[nb_bands - 1] -= too_large
    assert sum(erb) == fft_size // 2 + 1
    return np.asarray(erb, dtype=np.uint64)


class DF:
    """libdf.DF: STFT analysis/synthesis state (Vorbis window) and ERB band widths."""

    def __init__(self, sr, fft_size, hop_size, nb_bands=32, min_nb_erb_freqs=1):
        assert hop_size * 2 <= fft_size
        self._sr = int(sr)
        self._fft = int(fft_size)
        self._hop = int(hop_size)
        self._freq = self._fft // 2 + 1
        self._erb = erb_fb_widths(self._sr, self._fft, int(nb_bands), int(min_nb_erb_freqs))
        h = self._fft // 2
        i = np.arange(self._fft, dtype=np.float64)
        s = np.sin(0.5 * np.pi * (i + 0.5) / h)
        self._window = np.sin(0.5 * np.pi * s * s).astype(np.float32)
        self._wnorm = np.float32(1.0) / (np.float32(self._fft ** 2) / np.float32(2 * self._hop))

    def sr(self):
        return self._sr

    def fft_size(self):
        return self._fft

    def hop_size(self):
        return self._hop

    def nb_erb(self):
        return len(self._erb)

    def erb_widths(self):
        return self._erb.copy()

    def fft_window(self):
        return self._window.copy()

    def reset(self):
        pass  # the state is reset in each analysis/synthesis call (binding default reset=True)

    def analysis(self, input, reset=True):
        """[C, T] float32 -> [C, T // hop, fft/2 + 1] complex64 (state reset per channel)."""
        x = np.asarray(input, dtype=np.float32)
        C, T = x.shape
        n = T // self._hop
        mem = self._fft - self._hop
        out = np.zeros((C, n, self._freq), dtype=np.complex64)
        for c in range(C):
            xp = np.concatenate([np.zeros(mem, dtype=np.float32), x[c, : n * self._hop]])
            idx = np.arange(n)[:, None] * self._hop + np.arange(self._fft)[None, :]
            frames = xp[idx] * self._window[None, :]             # float32 products, as in Rust
            spec = np.fft.rfft(frames.astype(np.float64), axis=-1)
            out[c] = (spec.astype(np.complex64) * self._wnorm).astype(np.complex64)
        return out

    def synthesis(self, input, reset=True):
        """[C, T, fft/2 + 1] complex64 -> [C, T * hop] float32 (state reset per channel)."""
        X = np.asarray(input, dtype=np.complex64)
        C, n, F = X.shape
        assert F == self._freq
        out = np.zeros((C, n * self._hop), dtype=np.float32)
        hop = self._hop
        assert self._fft == 2 * hop, 'this copy supports 50 % overlap only (DeepFilterNet3: 960 / 480)'
        for c in range(C):
            # realfft inverse is not normalized: x = N * irfft(X)
            fr = (np.fft.irfft(X[c].astype(np.complex128), n=self._fft, axis=-1) * self._fft).astype(np.float32)
            fr = fr * self._window[None, :]
            first = fr[:, :hop]
            second = fr[:, hop:]
            prev = np.concatenate([np.zeros((1, hop), dtype=np.float32), second[:-1]], axis=0)
            out[c] = (first + prev).reshape(-1)
        return out


def erb(input, erb_fb, db=True):
    """libdf.erb: mean power per ERB band ([..., T, F] complex -> [..., T, E]); 10*log10(x + 1e-10) if db."""
    x = np.asarray(input, dtype=np.complex64)
    widths = np.asarray(erb_fb, dtype=np.int64)
    p = (x.real.astype(np.float32) ** 2 + x.imag.astype(np.float32) ** 2).astype(np.float32)
    out = np.zeros(x.shape[:-1] + (len(widths),), dtype=np.float32)
    start = 0
    for b, w in enumerate(widths):
        k = np.float32(1.0) / np.float32(w)
        acc = np.zeros(x.shape[:-1], dtype=np.float32)
        for j in range(w):                      # sequential f32 sum, as in Rust
            acc = acc + p[..., start + j] * k
        out[..., b] = acc
        start += w
    if db:
        out = (np.log10(out + np.float32(1e-10), dtype=np.float32) * np.float32(10.0)).astype(np.float32)
    return out


def erb_norm(erb, alpha, state=None):
    """libdf.erb_norm: exponential mean normalisation over time ([C, T, E] float32)."""
    x = np.array(erb, dtype=np.float32, copy=True)
    C, T, E = x.shape
    a = np.float32(alpha)
    b = np.float32(1.0) - a
    if state is None:
        s0 = _linspace_f32(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], E)
        s = np.tile(s0[None, :], (C, 1))
    else:
        s = np.array(state, dtype=np.float32, copy=True)
    for t in range(T):
        s = x[:, t] * b + s * a
        x[:, t] = (x[:, t] - s) / np.float32(40.0)
    return x


def unit_norm(spec, alpha, state=None):
    """libdf.unit_norm: complex spec divided by sqrt of an exponential mean of |x| ([C, T, F] complex64)."""
    x = np.array(spec, dtype=np.complex64, copy=True)
    C, T, F = x.shape
    a = np.float32(alpha)
    b = np.float32(1.0) - a
    if state is None:
        s0 = _linspace_f32(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], F)
        s = np.tile(s0[None, :], (C, 1))
    else:
        s = np.array(state, dtype=np.float32, copy=True)
    for t in range(T):
        s = np.abs(x[:, t]).astype(np.float32) * b + s * a
        r = np.sqrt(s)
        x[:, t] = (x[:, t].real / r) + 1j * (x[:, t].imag / r)   # component-wise, as Complex32 / f32
    return x


def unit_norm_init(num_freq_bins):
    return _linspace_f32(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], num_freq_bins).reshape(1, -1)
