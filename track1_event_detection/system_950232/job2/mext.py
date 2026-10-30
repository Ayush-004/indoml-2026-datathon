"""Master-based extent of a pasted cut (JOB 2, v2).

The generator cuts every long noise file at a random length, always from the file
start. Library templates of one file are prefixes of each other (16 kHz, lag <= 13
samples, NCC 1.000; see m01_masters.py). The longest cut is a speech-free copy of the
file ("master"). For a clip x that carries a cut of unknown length L at a known
alignment, x(u) = s(u) + g m(u) for u < L and x(u) = s(u) after it (s = speech and
everything else). The least-squares change point

    G(E) = sum_{u<E} (2 g x(u) m(u) - g^2 m(u)^2),      L_hat = argmax_E G(E)

gains g^2 m^2 per sample inside the paste and loses g^2 m^2 per sample after it, so it
integrates the evidence instead of thresholding a local coherence (no hole closing, no
over-extension from coherence blips). Optional pre-whitening (a smoothed inverse of
the clip spectrum) down-weights the bands where speech dominates.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import correlate
SR = 16000


def align(x, m, c0, W, maxlag=24):
    """Refine the clip position of master sample 0 by a small lag search on the first W samples."""
    best = (-1.0, c0)
    mm = m[:W]
    nm = np.linalg.norm(mm) + 1e-12
    for d in range(-maxlag, maxlag + 1):
        a = c0 + d
        if a < 0 or a + W > len(x): continue
        w = x[a:a + W]
        v = abs(float(np.dot(w, mm))) / (nm * (np.linalg.norm(w) + 1e-12))
        if v > best[0]: best = (v, a)
    return best[1], best[0]


def whiten_filter(x, nfft=1024, smooth=9):
    """Minimum-effort whitening: inverse of the smoothed clip magnitude spectrum,
    applied in the frequency domain to both signals (zero phase)."""
    from scipy.signal import welch
    f, p = welch(x, fs=SR, nperseg=nfft)
    a = np.sqrt(np.maximum(p, 1e-20))
    k = np.ones(smooth) / smooth
    a = np.convolve(a, k, mode='same')
    return f, 1.0 / np.maximum(a, a.max() * 1e-4)


def apply_filter(v, f, h):
    n = 1 << int(np.ceil(np.log2(len(v) + 1)))
    V = np.fft.rfft(v, n)
    fr = np.fft.rfftfreq(n, 1 / SR)
    H = np.interp(fr, f, h)
    return np.fft.irfft(V * H, n)[:len(v)]


def estimate(x, m, c0, min_len_s=1.5, whiten=False, gain_s=1.5, lagsearch=True):
    """x: clip (float64); m: master (float64); c0: clip sample where master sample 0 sits.
    Returns dict(L=seconds, ncc_head, g, prom) or None."""
    W = int(gain_s * SR)
    if lagsearch:
        c0, v = align(x, m, c0, min(W, len(m)))
    else:
        v = float('nan')
    if c0 < 0: return None
    U = min(len(m), len(x) - c0)
    if U < W: return None
    y = x[c0:c0 + U].astype(np.float64)
    mm = m[:U].astype(np.float64)
    if whiten:
        f, h = whiten_filter(x)
        y = apply_filter(y, f, h); mm = apply_filter(mm, f, h)
    g = float(np.dot(y[:W], mm[:W]) / (np.dot(mm[:W], mm[:W]) + 1e-18))
    inc = 2 * g * y * mm - g * g * mm * mm
    G = np.cumsum(inc)
    lo = int(min_len_s * SR)
    j = lo + int(np.argmax(G[lo:]))
    E = j + 1
    # prominence: how much G drops 0.25 s either side of the maximum (in units of the in-paste slope)
    slope = g * g * float(np.mean(mm[:W] ** 2)) + 1e-18
    q = int(0.25 * SR)
    left = G[j] - G[max(lo, j - q)] if j - q >= lo else np.nan
    right = G[j] - G[min(len(G) - 1, j + q)] if j + q < len(G) else np.nan
    return {'L': E / SR, 'c0': int(c0), 'ncc_head': v, 'g': g, 'U': U / SR,
            'at_end': bool(E >= U - 1), 'prom_left': float(left / (slope * q)) if left == left else None,
            'prom_right': float(right / (slope * q)) if right == right else None}
