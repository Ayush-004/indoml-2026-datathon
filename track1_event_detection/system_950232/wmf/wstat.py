"""Detection statistics for a known, exactly pasted waveform (template) with unknown gain
under loud, coloured, non-stationary speech.

Full-lag statistics (FFT, one clip against one kernel, every lag t):
  S0  plain NCC  <y_t, x> / (|y_t| |x|)                       (the base detector, signed)
  S1  whitened NCC: both y and x filtered by the same zero-phase filter |H(f)| =
      1/sqrt(smoothed clip PSD + floor) (per clip, fitted on the noisy clip)
  S2  pre-emphasis NCC: both filtered by 1 - 0.97 z^-1

Candidate-lag statistics (at a given lag t; STFT tiles = frequency bands x time blocks):
  S3  tile GLRT  Z = sqrt(n) * sum_i (c_i / e_i) / sqrt(sum_i ex_i / e_i)
      c_i = Re sum Y X*, e_i = sum |Y|^2 (local clip power of the tile), ex_i = sum |X|^2;
      the matched filter after a per-tile power normalisation (local whitening at the tile
      level). Under H0 (Gaussian, locally stationary) Z ~ N(0, 1).
  S4  the same with the tile power smoothed over the neighbouring tiles (3 x 3).
  S5  exact-subtraction score: g = weighted LS gain (GLRT gain of S3), residual R = Y - g X;
      template-energy weighted mean over tiles of log(e_i / r_i) (energy removed where the
      template has energy; a true paste falls to the speech floor, a false lag adds energy).
All statistics are signed: a paste has a positive gain (2,213/2,213 validation events).
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

SR = 16000
PSD_NPER = 512
PSD_HOP = 256
PRE = 0.97


def nfft_for(n):
    return 1 << int(np.ceil(np.log2(max(n, 2))))


def clip_psd(y, nper=PSD_NPER, hop=PSD_HOP):
    """Welch power spectrum (periodic Hann, 50 % overlap) on nper//2+1 bins."""
    if len(y) < nper:
        y = np.concatenate([y, np.zeros(nper - len(y))])
    w = np.hanning(nper + 1)[:-1]
    nfr = 1 + (len(y) - nper) // hop
    idx = np.arange(nper)[None, :] + hop * np.arange(nfr)[:, None]
    F = sfft.rfft(y[idx] * w[None, :], axis=1)
    return np.mean(np.abs(F) ** 2, axis=0) / np.sum(w ** 2)


def whitening_gain2(y, n, smooth_bins=5, floor_rel=1e-4, psd=None):
    """|H(f)|^2 on the n-point rfft grid: 1 / (smoothed clip PSD + floor)."""
    P = clip_psd(y) if psd is None else psd
    if smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        P = np.convolve(np.pad(P, smooth_bins // 2, mode='edge'), k, mode='valid')
    P = P + floor_rel * float(np.median(P)) + 1e-20
    f_src = np.linspace(0, SR / 2, len(P))
    f_dst = np.linspace(0, SR / 2, n // 2 + 1)
    logP = np.interp(f_dst, f_src, np.log(P))
    H2 = np.exp(-logP)
    return H2 / np.mean(H2)


def preemph_gain2(n, a=PRE):
    w = np.linspace(0, np.pi, n // 2 + 1)
    return np.abs(1 - a * np.exp(-1j * w)) ** 2


def rfft_energy(X2, n):
    """sum of squares of the length-n real signal whose rfft power is X2 (Parseval)."""
    s = X2[0] + X2[-1] + 2 * np.sum(X2[1:-1]) if n % 2 == 0 else X2[0] + 2 * np.sum(X2[1:])
    return float(s) / n


class ClipScan:
    """One clip prepared for full-lag scans with several spectral weightings."""

    def __init__(self, y, weights=('plain', 'white', 'pre'), white_kw=None):
        self.y = np.asarray(y, np.float64)
        self.N = len(self.y)
        self.n = nfft_for(self.N)
        self.Y = sfft.rfft(self.y, self.n)
        self.W = {}
        for name in weights:
            if name == 'plain':
                self.W[name] = None
            elif name == 'white':
                self.W[name] = whitening_gain2(self.y, self.n, **(white_kw or {}))
            elif name == 'pre':
                self.W[name] = preemph_gain2(self.n)
            else:
                raise ValueError(name)
        self._cum = {}

    def cumenergy(self, name):
        if name not in self._cum:
            W = self.W[name]
            yf = self.y if W is None else sfft.irfft(self.Y * np.sqrt(W), self.n)[:self.N]
            self._cum[name] = np.concatenate([[0.0], np.cumsum(yf * yf)])
        return self._cum[name]

    def ncc(self, name, X, L, kernel_energy_w=None):
        """signed normalised correlation for every lag t = 0..N-L.
        X: rfft(kernel, n) (not conjugated); L kernel length."""
        W = self.W[name]
        P = self.Y * np.conj(X)
        if W is not None:
            P = P * W
        c = sfft.irfft(P, self.n)[:self.N - L + 1]
        if W is None:
            ex = rfft_energy(np.abs(X) ** 2, self.n)
        else:
            ex = rfft_energy(np.abs(X) ** 2 * W, self.n)
        C = self.cumenergy(name)
        e = C[L:] - C[:-L]
        e = e[:self.N - L + 1]
        floor = 1e-10 * L          # ~ -100 dBFS mean power: digital silence
        return c / np.sqrt(np.maximum(e, floor) * max(ex, 1e-30))


def block_peaks(v, L, top=4, min_block=400):
    """top local maxima of a trace, one per block of length max(L//2, min_block)."""
    B = max(L // 2, min_block)
    m = len(v)
    nb = (m + B - 1) // B
    pad = nb * B - m
    vv = np.concatenate([v, np.full(pad, -np.inf)]) if pad else v
    R = vv.reshape(nb, B)
    j = np.argmax(R, axis=1)
    vals = R[np.arange(nb), j]
    pos = j + B * np.arange(nb)
    order = np.argsort(-vals)[:top]
    return [(int(pos[i]), float(vals[i])) for i in order if np.isfinite(vals[i])]


# ------------------------------------------------------------------ candidate-lag statistics

FRAME = 256
HOP = 128
_WIN = np.hanning(FRAME + 1)[:-1]


def stft(v):
    """frames at 0, HOP, ... covering len(v) (zero padded); returns (T, F) complex."""
    n = len(v)
    nfr = max(1, int(np.ceil((n - FRAME) / HOP)) + 1) if n > FRAME else 1
    need = (nfr - 1) * HOP + FRAME
    if need > n:
        v = np.concatenate([v, np.zeros(need - n)])
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(nfr)[:, None]
    return sfft.rfft(v[idx] * _WIN[None, :], axis=1)


def tile_sums(A, fb, tb):
    """sum a (T, F) array over tiles of tb frames x fb bins (bin 0 dropped; last partial kept)."""
    A = A[:, 1:]
    T, F = A.shape
    nt = (T + tb - 1) // tb
    nf = (F + fb - 1) // fb
    pt = nt * tb - T
    pf = nf * fb - F
    if pt or pf:
        A = np.pad(A, ((0, pt), (0, pf)))
    return A.reshape(nt, tb, nf, fb).sum(axis=(1, 3))


def smooth3(E):
    """3 x 3 mean over tiles (edge replicate)."""
    P = np.pad(E, 1, mode='edge')
    return (P[:-2, :-2] + P[:-2, 1:-1] + P[:-2, 2:] + P[1:-1, :-2] + P[1:-1, 1:-1] + P[1:-1, 2:]
            + P[2:, :-2] + P[2:, 1:-1] + P[2:, 2:]) / 9.0


def tile_stats(yseg, Xs, fb=8, tb=4, floor=1e-9):
    """yseg: clip segment aligned with the kernel; Xs: stft(kernel) (precomputed).
    Returns dict(S3, S4, S5, g, dof) — see module doc."""
    Ys = stft(yseg)
    T = min(len(Ys), len(Xs))
    Ys = Ys[:T]
    Xs = Xs[:T]
    c = tile_sums(np.real(Ys * np.conj(Xs)), fb, tb)
    e = tile_sums(np.abs(Ys) ** 2, fb, tb)
    ex = tile_sums(np.abs(Xs) ** 2, fb, tb)
    # floor: an absolute power floor per tile (digital silence) scaled by the tile size
    fl = floor * fb * tb * FRAME
    e = np.maximum(e, fl)
    dof = fb * tb  # complex STFT values per tile at 50 % overlap ~ fb*tb real dof
    num = np.sum(c / e)
    den = np.sqrt(np.sum(ex / e))
    S3 = np.sqrt(dof) * num / max(den, 1e-30)
    es = np.maximum(smooth3(e), fl)
    S4 = np.sqrt(dof) * np.sum(c / es) / max(np.sqrt(np.sum(ex / es)), 1e-30)
    g = num / max(np.sum(ex / e), 1e-30)
    r = e - 2 * g * c + g * g * ex          # tile energy of Y - g X
    r = np.maximum(r, fl)
    wts = ex / max(np.sum(ex), 1e-30)
    S5 = float(np.sum(wts * np.log(e / r)))
    return {'S3': float(S3), 'S4': float(S4), 'S5': S5, 'g': float(g)}
