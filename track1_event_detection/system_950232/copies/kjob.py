"""Run a kernel set (natural cuts of labelled VALIDATION natural events) over an audio store, one clip
at a time, with the kscan statistics. Resumable chunks in cache/k_<SET>_<STORE>[_<MODE>]/.

    python kjob.py SET STORE MODE WORKERS [CHUNK]
SET    'src71'  the 71 labelled natural events that hold a library template (q1_val.json)
       'cand'   the Q2 candidate templates (q2_kernels.py -> cache/kernels_cand.json)
STORE  valsyn | valclean | train | testsyn (read-only stores of the WMF cache)
MODE   'full'  every lag of the clip (w02-style candidates) + the true lags of the validation pastes
               of the kernel's templates (valsyn only, for the positives)
       'free'  only lags whose kernel extent lies inside a generator-legal free region:
               valsyn: free of every LABELLED event (null); testsyn: free of every event of the WMF
               file test_synthetic_predictions_wmf.jsonl (the application). No label of a test clip.
Row fields: j (kernel), p (lag), S0, S1, S3, S5, g, src (1 = stage-1 peak, 2 = requested true-lag window, 3 = both)."""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

os.environ.setdefault('OMP_NUM_THREADS', '1')
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import kscan  # noqa: E402
import nclib as N  # noqa: E402

SR = N.SR
KF = ('j', 'p', 'S0', 'S1', 'S3', 'S5', 'g', 'src')
REFINE = 32          # true-lag window (samples) around the expected lag of a validation paste

_K = None
_EXTRA = None
_RANGES = None


def kernel_specs(name):
    p = N.CACHE / f'kernels_{name}.json'
    return json.loads(p.read_text())


def cut_kernels(specs):
    idx = N.store_index('valnat')
    out = []
    keep = len(specs) <= 200            # large kernel sets: STFT on demand (worker memory)
    for s in specs:
        y = N.read_clip('valnat', s['clip'], idx)
        x = y[s['a']:s['b']].copy()
        out.append(kscan.kernel_prep(x, keep_stft=keep))
    return out


def free_ranges(events, dur, vocal):
    """sample ranges [(u, v)] where a NEW paste may lie: inside [EDGE_ON, dur - EDGE_OFF] (noise) or
    [0, dur] (vocal), at least GAP from every existing event; none when 5 events exist."""
    if len(events) >= N.MAX_EVENTS:
        return []
    lo, hi = (0.0, dur) if vocal else (N.EDGE_ON - N.SLACK, dur - N.EDGE_OFF + N.SLACK)
    cuts = sorted((a - N.GAP + N.SLACK, b + N.GAP - N.SLACK) for a, b in events)
    out = []
    cur = lo
    for a, b in cuts:
        if a > cur:
            out.append((cur, min(a, hi)))
        cur = max(cur, b)
        if cur >= hi:
            break
    if cur < hi:
        out.append((cur, hi))
    return [(int(np.ceil(u * SR)), int(np.floor(v * SR))) for u, v in out if v > u]


def ranges_for(store):
    """{clip: {'noise': ranges, 'vocal': ranges}} for MODE free."""
    if store == 'valsyn':
        lab = N.val_labels()
        vd = N.val_durations()
        ev = {c: [(a, b) for a, b, t in lab[c]] for c in N.val_syn_clips()}
        dur = vd
    elif store == 'testsyn':
        pred = N.read_pred_jsonl(N.WMF_SYN)
        td = N.test_durations()
        ev = {c: pred[N.stem(c)] for c in N.test_syn_clips()}
        dur = td
    else:
        raise SystemExit('free mode only for valsyn / testsyn')
    return {c: {'noise': free_ranges(e, dur[c], False), 'vocal': free_ranges(e, dur[c], True)} for c, e in ev.items()}


def extra_for(store, specs):
    """valsyn, mode full: {clip: [(j, true lag)]} of the pastes of each kernel's templates."""
    if store != 'valsyn':
        return {}
    E = N.load_pickle(N.WMF_CACHE / 'events.pkl')['events']
    by_k = {}
    for j, s in enumerate(specs):
        for k, off in s.get('templates', {}).items():
            by_k.setdefault(int(k), []).append((j, int(off)))
    out = {}
    for e in E:
        for j, off in by_k.get(e['k'], []):
            out.setdefault(e['clip'], []).append((j, int(e['paste_start']) - off))
    return out


def _init(name, store, mode):
    global _K, _EXTRA, _RANGES
    specs = kernel_specs(name)
    _K = [(s, kp) for s, kp in zip(specs, cut_kernels(specs))]
    _EXTRA = extra_for(store, specs) if mode == 'full' else {}
    _RANGES = ranges_for(store) if mode == 'free' else None


def scan_one(y, clip):
    C = kscan.Clip(y)
    rows = []
    ex = {}
    for j, p in _EXTRA.get(clip, []):
        ex.setdefault(j, []).extend(range(p - REFINE, p + REFINE + 1))
    for j, (s, kp) in enumerate(_K):
        rng = None
        if _RANGES is not None:
            rng = _RANGES[clip]['vocal' if s['vocal'] else 'noise']
            if not rng:
                continue
        res = kscan.scan_kernel(C, kp, ranges=rng, extra_lags=ex.get(j, ()))
        if res is None:
            continue
        lags, S0, S1, S3, S5, g, fp = res
        xs = set(ex.get(j, ()))
        for i in range(len(lags)):
            # src: 1 = stage-1 peak, 2 = requested true-lag window only, 3 = both
            src = (1 if fp[i] else 0) + (2 if int(lags[i]) in xs else 0)
            rows.append((j, lags[i], S0[i], S1[i], S3[i], S5[i], g[i], float(src)))
    return np.array(rows, np.float64).astype(np.float32) if rows else np.zeros((0, len(KF)), np.float32)


def _work(args):
    store, clips, out = args
    if Path(out).exists():
        return out, 0.0
    t0 = time.time()
    idx = N.store_index(store)
    res = {}
    for c in clips:
        y = N.read_clip(store, c, idx)
        res[c] = scan_one(y, c)
    tmp = out + '.tmp.npz'
    np.savez_compressed(tmp, clips=np.array(list(res)), **{f'r{j}': v for j, v in enumerate(res.values())})
    os.replace(tmp, out)
    return out, time.time() - t0


def outdir(name, store, mode):
    return N.CACHE / f'k_{name}_{store}_{mode}'


def run(name, store, mode, workers, chunk=16, clips=None):
    idx = N.store_index(store)
    if clips is None:
        clips = sorted(idx)
        if store == 'testsyn':
            assert set(clips) == set(N.test_syn_clips())
    clips = sorted(clips, key=lambda c: idx[c][1])
    d = outdir(name, store, mode)
    d.mkdir(exist_ok=True)
    jobs = [(store, clips[s:s + chunk], str(d / f'chunk_{s // chunk:04d}.npz')) for s in range(0, len(clips), chunk)]
    todo = [j for j in jobs if not Path(j[2]).exists()]
    print(name, store, mode, len(clips), 'clips', len(jobs), 'chunks', len(todo), 'to do', flush=True)
    t0 = time.time()
    if workers <= 1:
        _init(name, store, mode)
        for n, j in enumerate(todo):
            _work(j)
            print(n + 1, '/', len(todo), round(time.time() - t0), 's', flush=True)
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(workers, initializer=_init, initargs=(name, store, mode),
                                       maxtasksperchild=25) as pool:
            for n, (o, dt) in enumerate(pool.imap_unordered(_work, todo)):
                if (n + 1) % 10 == 0 or n + 1 == len(todo):
                    print(n + 1, '/', len(todo), 'chunks', round(time.time() - t0), 's', flush=True)


def load(name, store, mode):
    return N.load_scan_dir(outdir(name, store, mode))


if __name__ == '__main__':
    name, store, mode, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    run(name, store, mode, workers, chunk)
