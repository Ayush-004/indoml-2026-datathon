"""Scan of arbitrary kernels (natural cuts of labelled validation events, Q2) with the WMF
statistics. The statistic is wstat.tile_stats (S3 tile GLRT z with tiles 8 bins x 4 frames, and the
exact-subtraction score S5), computed here for a BATCH of candidate lags at once (same arithmetic,
checked against wstat.tile_stats by `python kscan.py check`).

Stage 1 per (clip, kernel): full-lag whitened NCC S1 and plain NCC S0 (wstat.ClipScan, as w02_scan);
candidates = top-5 block peaks of S1 U top-3 block peaks of S0, computed on the traces restricted
to the ALLOWED lags (a lag p is allowed when the kernel [p, p + L) fits the allowed sample range
of the clip: the whole clip for a null clip, the generator-legal free regions for a synthetic clip).
Stage 2: S3, S5, g at the candidates (+ optional requested lags).
One clip at a time; no clip is compared with another clip.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
from scipy import fft as sfft  # noqa: E402

import wstat  # noqa: E402

FRAME = wstat.FRAME
HOP = wstat.HOP
WIN = wstat._WIN
TOP_S1 = 5
TOP_S0 = 3


def kernel_prep(x, keep_stft=True):
    """per kernel: waveform (float32 holds int16 audio exactly) and, optionally, its STFT (T, 129).
    Without keep_stft the STFT is computed when needed (memory: 2,849 kernels)."""
    x = np.asarray(x, np.float32)
    kp = {'x': x, 'L': len(x)}
    if keep_stft:
        kp['Xs'] = wstat.stft(x.astype(np.float64))
    return kp


def kernel_stft(kp):
    return kp['Xs'] if 'Xs' in kp else wstat.stft(kp['x'].astype(np.float64))


def _tile_sums_b(A, fb, tb):
    """batch version of wstat.tile_sums: A (B, T, F) -> (B, nt, nf)."""
    A = A[:, :, 1:]
    B, T, Fq = A.shape
    nt = (T + tb - 1) // tb
    nf = (Fq + fb - 1) // fb
    pt = nt * tb - T
    pf = nf * fb - Fq
    if pt or pf:
        A = np.pad(A, ((0, 0), (0, pt), (0, pf)))
    return A.reshape(B, nt, tb, nf, fb).sum(axis=(2, 4))


def tile_s3_batch(y, lags, kp, fb=8, tb=4, floor=1e-9):
    """S3, S5, g of wstat.tile_stats(y[p:p+L], Xs, fb, tb) for every p in lags (vectorised)."""
    L = kp['L']
    Xs = kernel_stft(kp)
    lags = np.asarray(lags, np.int64)
    n = L
    nfr = max(1, int(np.ceil((n - FRAME) / HOP)) + 1) if n > FRAME else 1
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(nfr)[:, None]          # (nfr, FRAME) within segment
    valid = idx < L
    idc = np.minimum(idx, L - 1)
    fr = y[lags[:, None, None] + idc[None, :, :]] * valid[None, :, :]         # zero padding past L
    Ys = sfft.rfft(fr * WIN[None, None, :], axis=-1)
    T = min(Ys.shape[1], len(Xs))
    Ys = Ys[:, :T]
    X = Xs[:T]
    c = _tile_sums_b(np.real(Ys * np.conj(X)[None]), fb, tb)
    e = _tile_sums_b(np.abs(Ys) ** 2, fb, tb)
    ex = wstat.tile_sums(np.abs(X) ** 2, fb, tb)[None]
    fl = floor * fb * tb * FRAME
    e = np.maximum(e, fl)
    dof = fb * tb
    num = np.sum(c / e, axis=(1, 2))
    sx = np.sum(ex / e, axis=(1, 2))
    S3 = np.sqrt(dof) * num / np.maximum(np.sqrt(sx), 1e-30)
    g = num / np.maximum(sx, 1e-30)
    r = e - 2 * g[:, None, None] * c + (g * g)[:, None, None] * ex
    r = np.maximum(r, fl)
    wts = ex / max(float(np.sum(ex)), 1e-30)
    S5 = np.sum(wts * np.log(e / r), axis=(1, 2))
    return S3, S5, g


def masked_peaks(v, L, top, allowed=None, min_block=400):
    """wstat.block_peaks on a trace whose disallowed lags are set to -inf."""
    if allowed is not None:
        v = np.where(allowed, v, -np.inf)
    return [p for p, _ in wstat.block_peaks(v, L, top, min_block)]


def allowed_mask(nlags, L, ranges):
    """lags p (0..nlags-1) with [p, p + L) inside one of the sample ranges [(u, v)]."""
    m = np.zeros(nlags, bool)
    for u, v in ranges:
        lo = max(0, int(u))
        hi = min(nlags - 1, int(v) - L)
        if hi >= lo:
            m[lo:hi + 1] = True
    return m


class Clip:
    def __init__(self, y):
        self.y = np.asarray(y, np.float64)
        self.cs = wstat.ClipScan(self.y, weights=('plain', 'white'))


def scan_kernel(clip, kp, ranges=None, extra_lags=(), use_s0=True):
    """candidates of one kernel in one clip -> (lags, S0, S1, S3, S5, g, from_peaks) arrays."""
    cs = clip.cs
    L = kp['L']
    if L >= cs.N:
        return None
    nl = cs.N - L + 1
    allowed = None if ranges is None else allowed_mask(nl, L, ranges)
    extra = [int(p) for p in extra_lags if 0 <= p <= cs.N - L]
    open_ = allowed is None or bool(allowed.any())
    if not open_ and not extra:
        return None
    X = sfft.rfft(kp['x'].astype(np.float64), cs.n)
    s1 = cs.ncc('white', X, L)
    s0 = cs.ncc('plain', X, L) if use_s0 else None
    peaks = set()
    if open_:
        peaks |= set(masked_peaks(s1, L, TOP_S1, allowed))
        if use_s0:
            peaks |= set(masked_peaks(s0, L, TOP_S0, allowed))
    cands = peaks | set(extra)
    if not cands:
        return None
    lags = np.array(sorted(cands), np.int64)
    S3, S5, g = tile_s3_batch(clip.y, lags, kp)
    S0 = s0[lags] if use_s0 else np.full(len(lags), np.nan)
    from_peaks = np.array([int(p) in peaks for p in lags])
    return lags, S0, s1[lags], S3, S5, g, from_peaks


def check(n=40, seed=3):
    """vectorised S3/S5/g == wstat.tile_stats on random segments of a real clip."""
    import nclib as N
    rng = np.random.default_rng(seed)
    idx = N.store_index('valsyn')
    clips = sorted(idx)
    worst = 0.0
    for t in range(n):
        c = clips[rng.integers(len(clips))]
        y = N.read_clip('valsyn', c, idx)
        L = int(rng.integers(300, min(40000, len(y) // 2)))
        x = y[rng.integers(0, len(y) - L):][:L].copy() + 1e-3 * rng.standard_normal(L)
        kp = kernel_prep(x)
        lags = rng.integers(0, len(y) - L, size=5)
        S3, S5, g = tile_s3_batch(y, lags, kp)
        for j, p in enumerate(lags):
            ref = wstat.tile_stats(y[p:p + L], wstat.stft(kp['x'].astype(np.float64)), fb=8, tb=4)
            worst = max(worst, abs(ref['S3'] - S3[j]) / max(1.0, abs(ref['S3'])),
                        abs(ref['S5'] - S5[j]) / max(1.0, abs(ref['S5'])), abs(ref['g'] - g[j]) / max(1e-6, abs(ref['g'])))
    print('kscan check: max relative difference %.3g over %d x 5 lags' % (worst, n))
    return worst


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        w = check()
        assert w < 1e-6
