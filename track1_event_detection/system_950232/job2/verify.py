"""Numerically safe verification of a candidate repeat at 16 kHz.

The running-energy normaliser of a long cumsum loses all precision in near-silent
windows (it can even go negative), which turns digital silence into a fake NCC of 1.
Here the peak position comes from a rough pass and the score is recomputed exactly
from the raw slices, with an absolute amplitude floor.
"""
import numpy as np
from scipy.signal import correlate, oaconvolve
SR=16000
RMS_FLOOR=3e-4        # -70 dBFS: below this a window is silence, not a noise event

def exact_ncc(y, k, j):
    w=y[j:j+len(k)]
    if len(w)<len(k): return 0.0
    w=w-w.mean(); kk=k-k.mean()
    nw=np.linalg.norm(w); nk=np.linalg.norm(kk)
    if nw<RMS_FLOOR*np.sqrt(len(w)) or nk<RMS_FLOOR*np.sqrt(len(kk)): return 0.0
    return float(abs(np.dot(w,kk))/(nw*nk))

def best_match(y, k, topk=3):
    """Best NCC of kernel k inside y, computed safely."""
    kk=k-k.mean(); nk=np.linalg.norm(kk)
    if nk<RMS_FLOOR*np.sqrt(len(k)) or len(y)<=len(k): return 0.0, 0
    kn=kk/nk; L=len(kn)
    yy=(y-y.mean()).astype(np.float64)
    e=oaconvolve(yy*yy, np.ones(L), mode='valid')        # stable running energy
    cc=np.abs(correlate(yy,kn,mode='valid',method='fft'))
    m=min(len(cc),len(e))
    good=e[:m]>(RMS_FLOOR**2)*L
    score=np.where(good, cc[:m]/np.sqrt(np.maximum(e[:m],1e-12)), 0.0)
    best=(0.0,0)
    for j in np.argsort(score)[::-1][:topk]:
        v=exact_ncc(y,k,int(j))
        if v>best[0]: best=(v,int(j))
    return best
