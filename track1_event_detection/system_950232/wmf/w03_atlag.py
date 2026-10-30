"""Statistics at given lags (not only at the scan candidates): used for the audit of the base
hits and for the true-lag reference of the validation events.

    python w03_atlag.py STORE REQUEST.pkl OUT.pkl WORKERS
REQUEST.pkl: {clip: [(k, p), ...]}.  For each request the lag is refined inside +-16 samples
by the whitened NCC (S1); the tile statistics are computed at the refined lag.
OUT.pkl: {clip: [(k, p_req, p_ref, S0, S1, S3, S4, S5, g, S3b, S5b)]}"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
from scipy import fft as sfft  # noqa: E402

import wlib  # noqa: E402
import wstat  # noqa: E402

_EXS = None
_XS = {}


def _init():
    global _EXS
    EX = wlib.exemplars()
    _EXS = [[np.asarray(e[1], np.float64) for e in t['exs']] for t in EX]


def _xs(k, i):
    if (k, i) not in _XS:
        _XS[(k, i)] = wstat.stft(_EXS[k][i])
    return _XS[(k, i)]


def at_lags(y, clip, reqs, loo, refine=16):
    cs = wstat.ClipScan(y, weights=('plain', 'white'))
    out = []
    bytemp = {}
    for k, p in reqs:
        bytemp.setdefault(int(k), []).append(int(p))
    for k, ps in bytemp.items():
        i = wlib.kernel_index(k, clip) if loo else 0
        if i is None:
            for p in ps:
                out.append((k, p, -1) + (np.nan,) * 8)
            continue
        x = _EXS[k][i]
        L = len(x)
        if L >= cs.N:
            for p in ps:
                out.append((k, p, -1) + (np.nan,) * 8)
            continue
        X = sfft.rfft(x, cs.n)
        s0 = cs.ncc('plain', X, L)
        s1 = cs.ncc('white', X, L)
        for p in ps:
            lo = max(0, p - refine)
            hi = min(len(s1), p + refine + 1)
            if lo >= hi:
                out.append((k, p, -1) + (np.nan,) * 8)
                continue
            q = lo + int(np.argmax(s1[lo:hi]))
            seg = y[q:q + L]
            a = wstat.tile_stats(seg, _xs(k, i), fb=8, tb=4)
            b = wstat.tile_stats(seg, _xs(k, i), fb=16, tb=8)
            out.append((k, p, q, float(s0[q]), float(s1[q]), a['S3'], a['S4'], a['S5'], a['g'], b['S3'], b['S5']))
    return out


def _work(args):
    store, items, loo = args
    idx = wlib.store_index(store)
    res = {}
    for c, reqs in items:
        y = wlib.read_clip(store, c, idx)
        res[c] = at_lags(y, c, reqs, loo)
    return res


def run(store, request, workers=3, chunk=16):
    loo = store in ('valsyn', 'valclean')
    items = sorted(request.items())
    jobs = [(store, items[s:s + chunk], loo) for s in range(0, len(items), chunk)]
    out = {}
    if workers <= 1:
        _init()
        for j in jobs:
            out.update(_work(j))
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(workers, initializer=_init) as pool:
            for r in pool.imap_unordered(_work, jobs):
                out.update(r)
    return out


if __name__ == '__main__':
    store, req, outp, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    R = wlib.load_pickle(req)
    wlib.dump_pickle(outp, run(store, R, workers))
    print('done', outp)
