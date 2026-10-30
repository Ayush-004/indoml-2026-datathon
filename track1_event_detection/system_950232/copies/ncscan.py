"""The WMF production scan (../wmf/w02_scan.py, scan_clip), re-implemented here with
the same logic so that every cache lands in this folder:

Stage 1 (full lag, FFT): S0 plain NCC and S1 per-clip whitened NCC (signed);
  candidates = top-5 block peaks of S1  U  top-3 block peaks of S0.
Stage 2 (at each candidate lag): wstat.tile_stats (S3 tile GLRT z, S4, S5, g) for tiles 8 x 4 and 16 x 8.
Kernel: exemplar 0 of every library template (natural stores and test stores are never scanned
leave-one-out; the same choice as w02_scan for 'valnat', 'train', 'testsyn').

One clip at a time; no clip is compared with another clip.

    python ncscan.py build-testnat            (int16 store of the 3,435 natural-routed test clips)
    python ncscan.py scan STORE WORKERS [CHUNK]
    python ncscan.py selfcheck               (re-scan 30 natural validation clips; must equal the WMF scan_valnat)
Output: cache/scan_STORE/chunk_XXXX.npz (resumable)."""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

os.environ.setdefault('OMP_NUM_THREADS', '1')
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import fft as sfft  # noqa: E402

import nclib as N  # noqa: E402
import wstat  # noqa: E402

FIELDS = N.FIELDS
TOP_S1 = 5
TOP_S0 = 3

_EXS = None
_XS = {}
_FFT = {}


def _init():
    global _EXS
    EX = N.exemplars()
    _EXS = [np.asarray(t['exs'][0][1], np.float64) for t in EX]
    for k, x in enumerate(_EXS):
        _XS[k] = wstat.stft(x)


def _kfft(k, n):
    if n not in _FFT:
        _FFT.clear()
        _FFT[n] = {}
    d = _FFT[n]
    if k not in d:
        d[k] = sfft.rfft(_EXS[k], n).astype(np.complex64)
    return d[k]


def scan_clip(y):
    cs = wstat.ClipScan(y, weights=('plain', 'white'))
    rows = []
    for k in range(len(_EXS)):
        x = _EXS[k]
        L = len(x)
        if L >= cs.N:
            continue
        X = _kfft(k, cs.n).astype(np.complex128)
        s0 = cs.ncc('plain', X, L)
        s1 = cs.ncc('white', X, L)
        cands = {p for p, _ in wstat.block_peaks(s1, L, TOP_S1)} | {p for p, _ in wstat.block_peaks(s0, L, TOP_S0)}
        Xs = _XS[k]
        for p in sorted(cands):
            seg = y[p:p + L]
            if len(seg) < L:
                continue
            a = wstat.tile_stats(seg, Xs, fb=8, tb=4)
            b = wstat.tile_stats(seg, Xs, fb=16, tb=8)
            rows.append((k, p, s0[p], s1[p], a['S3'], a['S4'], a['S5'], a['g'], b['S3'], b['S5']))
    return np.array(rows, np.float64).astype(np.float32) if rows else np.zeros((0, len(FIELDS)), np.float32)


def _work(args):
    store, clips, out = args
    if Path(out).exists():
        return out, 0.0
    t0 = time.time()
    idx = N.store_index(store)
    res = {}
    for c in clips:
        y = N.read_clip(store, c, idx)
        res[c] = scan_clip(y)
    tmp = out + '.tmp.npz'
    np.savez_compressed(tmp, clips=np.array(list(res)), **{f'r{j}': v for j, v in enumerate(res.values())})
    os.replace(tmp, out)
    return out, time.time() - t0


def run(store, workers, chunk=24, clips=None):
    idx = N.store_index(store)
    clips = sorted(idx) if clips is None else clips
    clips = sorted(clips, key=lambda c: idx[c][1])
    d = N.CACHE / f'scan_{store}'
    d.mkdir(exist_ok=True)
    jobs = [(store, clips[s:s + chunk], str(d / f'chunk_{s // chunk:04d}.npz')) for s in range(0, len(clips), chunk)]
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
        with get_context('spawn').Pool(workers, initializer=_init, maxtasksperchild=20) as pool:
            for n, (o, dt) in enumerate(pool.imap_unordered(_work, todo)):
                if (n + 1) % 5 == 0 or n + 1 == len(todo):
                    print(n + 1, '/', len(todo), 'chunks', round(time.time() - t0), 's', flush=True)
    return jobs


def build_testnat():
    nat = set(N.test_nat_clips())
    assert len(nat) == 3435, len(nat)
    with zipfile.ZipFile(N.TEST_ZIP) as z:
        members = {}
        for n in z.namelist():
            if n.endswith('.wav') and '__MACOSX' not in n:
                b = n.rsplit('/', 1)[-1]
                if b in nat:
                    members[b] = n
        assert len(members) == len(nat), (len(members), len(nat))
        items = [(c, N.zip_loader(z, members[c])) for c in sorted(nat)]
        idx = N.build_store('testnat', items)
    tot = sum(v[1] for v in idx.values())
    d = N.test_durations()
    bad = [c for c, (o, n) in idx.items() if abs(n / N.SR - d[c]) > 1e-3]
    print('testnat', len(idx), 'clips', round(tot / N.SR / 3600, 3), 'h ; duration mismatches', len(bad), flush=True)


def selfcheck(n=30):
    """re-scan n natural validation clips with this code; the rows must equal the WMF scan_valnat."""
    _init()
    ref = N.load_wmf_scan('valnat')
    idx = N.store_index('valnat')
    rng = np.random.default_rng(7)
    clips = sorted(ref)
    pick = [clips[i] for i in rng.choice(len(clips), n, replace=False)]
    worst = 0.0
    same = 0
    for c in pick:
        y = N.read_clip('valnat', c, idx)
        r = scan_clip(y)
        a = ref[c]
        if r.shape == a.shape:
            dmax = float(np.nanmax(np.abs(r - a))) if len(r) else 0.0
            worst = max(worst, dmax)
            same += dmax == 0.0
        else:
            worst = np.inf
    print('selfcheck: %d clips, identical %d, max abs diff %.3g' % (n, same, worst), flush=True)
    return {'clips': n, 'identical': same, 'max_abs_diff': worst}


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build-testnat':
        build_testnat()
    elif cmd == 'scan':
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 24)
    elif cmd == 'selfcheck':
        res = selfcheck()
        N.dump_json(N.HERE / 'selfcheck_scan.json', res)
    else:
        raise SystemExit('unknown command')
