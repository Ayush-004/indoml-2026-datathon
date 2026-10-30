"""PP = label-free post-processing of an enhancer output y for the noisy clip x (pass-through).

Uses only the clip's own input x and output y. No reference, no metadata, no other clip. numpy only.
Step 1 (optional quiet-frame muting; off in our use): 16 ms frames; zero a frame when the output frame level is more
  than t1 dB under the output p95 frame level AND the enhancer removed more than t2 dB there. 5 ms fades.
Step 2 (gain-matched pass-through): g = least-squares gain of y onto x over the least-changed loud frames; a frame is
  'unchanged' when 10 log10(|g x|^2 / |g x - y|^2) > T; unchanged frames take g x (5 ms cross-fades).
training/fusion_head.py uses the third return value (the pass-through share at T = 20 dB) to pick the training crops.
"""
import numpy as np

L = 256


def smooth(mf, n):
    gm = np.convolve(np.repeat(mf.astype(float), L), np.ones(81) / 81, mode='same')
    o = np.zeros(n)
    o[:len(gm)] = gm
    return o


def pp(y, x, gate=True, t1=30, t2=20, T=25):
    """Return (int16-rounded output, share of muted frames, share of passthrough frames). T=None: step 1 only."""
    n = len(y)
    nf = n // L
    yg = y
    gshare = 0.0
    if gate:
        fy = 10 * np.log10(np.mean(y[:nf * L].reshape(nf, L) ** 2, 1) + 1e-6)
        fx = 10 * np.log10(np.mean(x[:nf * L].reshape(nf, L) ** 2, 1) + 1e-6)
        z = ((fy - np.percentile(fy, 95)) < -t1) & ((fx - fy) > t2)
        gshare = float(z.mean())
        yg = y * (1 - smooth(z, n))
    if T is None:
        return np.rint(yg), gshare, 0.0
    X = x[:nf * L].reshape(nf, L)
    Y = yg[:nf * L].reshape(nf, L)
    ex = (X ** 2).sum(1) + 1e-9
    g0 = (yg @ x) / (x @ x + 1e-9)
    rel = ((Y - g0 * X) ** 2).sum(1) / ex
    hi = ex >= np.median(ex)
    cand = hi & (rel <= np.quantile(rel[hi], 0.3)) if hi.any() else hi
    g = float((Y[cand] * X[cand]).sum() / ((X[cand] ** 2).sum() + 1e-9)) if cand.any() else g0
    ch = 10 * np.log10(((g * X) ** 2).sum(1) / (((g * X - Y) ** 2).sum(1) + 1e-9) + 1e-12)
    un = ch > T
    w = smooth(un, n)
    w[nf * L:] = 0
    return np.rint((1 - w) * yg + w * (g * x)), gshare, float(un.mean())
