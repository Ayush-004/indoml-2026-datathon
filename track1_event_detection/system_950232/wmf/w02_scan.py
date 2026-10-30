"""Production scan: every clip of a store against every library template.

Stage 1 (full lag, FFT): S0 plain NCC and S1 per-clip whitened NCC (signed).
  candidates = top-5 block peaks of S1  U  top-3 block peaks of S0.
Stage 2 (at each candidate lag): tile statistics of wstat.tile_stats (S3 tile GLRT z,
  S4 smoothed-power GLRT z, S5 exact-subtraction log drop, g LS gain), for two tile sizes.

Kernels: exemplar of templates.py, leave-one-clip-out (a validation synthetic clip is never
scanned with an exemplar cut from itself; for every other set exemplar 0 is used).

    python w02_scan.py STORE WORKERS [CHUNK]
Output: cache/scan_STORE/chunk_XXXX.npz (resumable) -> merged by load_scan(STORE).
Row fields: k, p, S0, S1, S3, S4, S5, g, S3b, S5b (float32; k, p as float)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
from scipy import fft as sfft  # noqa: E402

import wlib  # noqa: E402
import wstat  # noqa: E402

FIELDS = ('k', 'p', 'S0', 'S1', 'S3', 'S4', 'S5', 'g', 'S3b', 'S5b')
TOP_S1 = 5
TOP_S0 = 3

_EXS = None
_XS = {}
_FFT = {}


def _init():
    global _EXS
    EX = wlib.exemplars()
    _EXS = [[np.asarray(e[1], np.float64) for e in t['exs']] for t in EX]
    for k, lst in enumerate(_EXS):
        for i, x in enumerate(lst):
            _XS[(k, i)] = wstat.stft(x)


def _kfft(k, i, n):
    key = n
    if key not in _FFT:
        if len(_FFT) >= 1:
            _FFT.clear()
        _FFT[key] = {}
    d = _FFT[key]
    if (k, i) not in d:
        d[(k, i)] = sfft.rfft(_EXS[k][i], n).astype(np.complex64)
    return d[(k, i)]


def scan_clip(y, clip, loo):
    cs = wstat.ClipScan(y, weights=('plain', 'white'))
    rows = []
    for k in range(len(_EXS)):
        i = wlib.kernel_index(k, clip) if loo else 0
        if i is None:
            continue
        x = _EXS[k][i]
        L = len(x)
        if L >= cs.N:
            continue
        X = _kfft(k, i, cs.n).astype(np.complex128)
        s0 = cs.ncc('plain', X, L)
        s1 = cs.ncc('white', X, L)
        cands = {p for p, _ in wstat.block_peaks(s1, L, TOP_S1)} | {p for p, _ in wstat.block_peaks(s0, L, TOP_S0)}
        Xs = _XS[(k, i)]
        for p in sorted(cands):
            seg = y[p:p + L]
            if len(seg) < L:
                continue
            a = wstat.tile_stats(seg, Xs, fb=8, tb=4)
            b = wstat.tile_stats(seg, Xs, fb=16, tb=8)
            rows.append((k, p, s0[p], s1[p], a['S3'], a['S4'], a['S5'], a['g'], b['S3'], b['S5']))
    return np.array(rows, np.float64).astype(np.float32) if rows else np.zeros((0, len(FIELDS)), np.float32)


def _work(args):
    store, clips, out, loo = args
    if Path(out).exists():
        return out, 0.0
    t0 = time.time()
    idx = wlib.store_index(store)
    res = {}
    for c in clips:
        y = wlib.read_clip(store, c, idx)
        res[c] = scan_clip(y, c, loo)
    tmp = out + '.tmp.npz'
    np.savez_compressed(tmp, clips=np.array(list(res)), **{f'r{j}': v for j, v in enumerate(res.values())})
    os.replace(tmp, out)
    return out, time.time() - t0


def run(store, workers, chunk=24, clips=None):
    idx = wlib.store_index(store)
    clips = sorted(idx) if clips is None else clips
    clips = sorted(clips, key=lambda c: idx[c][1])          # similar lengths share an FFT size
    loo = store in ('valsyn', 'valclean')
    d = wlib.CACHE / f'scan_{store}'
    d.mkdir(exist_ok=True)
    jobs = [(store, clips[s:s + chunk], str(d / f'chunk_{s // chunk:04d}.npz'), loo)
            for s in range(0, len(clips), chunk)]
    todo = [j for j in jobs if not Path(j[2]).exists()]
    print(store, len(clips), 'clips', len(jobs), 'chunks', len(todo), 'to do', flush=True)
    t0 = time.time()
    if workers <= 1:
        _init()
        for n, j in enumerate(todo):
            _work(j)
            print(n + 1, '/', len(todo), round(time.time() - t0), 's', flush=True)
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(workers, initializer=_init) as pool:
            for n, (o, dt) in enumerate(pool.imap_unordered(_work, todo)):
                if (n + 1) % 5 == 0 or n + 1 == len(todo):
                    print(n + 1, '/', len(todo), 'chunks', round(time.time() - t0), 's', flush=True)
    return jobs


def load_scan(store):
    d = wlib.CACHE / f'scan_{store}'
    out = {}
    for f in sorted(d.glob('chunk_*.npz')):
        z = np.load(f, allow_pickle=False)
        for j, c in enumerate(z['clips']):
            out[str(c)] = z[f'r{j}']
    return out


if __name__ == '__main__':
    store = sys.argv[1]
    workers = int(sys.argv[2])
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    run(store, workers, chunk)
