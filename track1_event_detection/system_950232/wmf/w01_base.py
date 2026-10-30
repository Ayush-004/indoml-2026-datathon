"""Reproduce the base system: the plain-NCC template decode of the 695 validation synthetic
clips with the honest 5-fold geography cross-fit of the per-template thresholds
(validation synthetic: 1.9584, TP 2155 / FP 8 / FN 58).

Also rebuilds the shipped test decode (tau_syn fitted on all 695 clips) with the template
index of every accepted hit, and maps it onto the uploaded job2only file.
Output: cache/base.pkl"""
from collections import defaultdict

import numpy as np

import wlib

SR = wlib.SR
EX = wlib.exemplars()
K = len(EX)
DUR = np.array([t['dur'] for t in EX])
GRID = np.round(np.arange(0.2, 0.96, 0.025), 3)
TOL = 0.02
FLOOR = 0.3


def tau_nat_from(hits, clips):
    hi = np.zeros(K)
    for c in clips:
        for k, p, v in hits[c]:
            hi[k] = max(hi[k], v)
    return np.maximum(hi + 0.03, 0.25)


def fit(hits, ref, clips, tau_nat):
    lab = defaultdict(list)
    for c in clips:
        for k, p, v in hits[c]:
            on = p / SR
            off = on + DUR[k]
            ok = any(abs(on - a) <= TOL and abs(off - b) <= TOL for a, b in ref[c])
            lab[k].append((v, ok))
    tau = np.array(tau_nat)
    for k in range(K):
        vals = lab.get(k, [])
        if not any(o for _, o in vals):
            continue
        best = None
        for t in GRID:
            tp = sum(1 for v, o in vals if o and v >= t)
            fp = sum(1 for v, o in vals if (not o) and v >= t)
            m = tp - fp
            if best is None or m > best[0] or (m == best[0] and t > best[1]):
                best = (m, t)
        tau[k] = max(min(best[1], tau_nat[k]), FLOOR)
    return tau


def decode_full(hits, tau):
    """greedy non-overlapping decode; returns [(onset, offset, k, ncc, sample)] sorted by onset."""
    cands = sorted(((v, p / SR, p / SR + DUR[k], k, p) for k, p, v in hits if v >= tau[k]), reverse=True)
    acc = []
    for v, on, off, k, p in cands:
        if all(off <= a or on >= b for _, a, b, _, _ in acc):
            acc.append((v, on, off, k, p))
    return sorted(((on, off, int(k), float(v), int(p)) for v, on, off, k, p in acc))


def main():
    ref = wlib.val_reference()
    hv = wlib.val_hits()
    vsyn = wlib.val_syn_clips()
    vnat = wlib.val_nat_clips()
    fold = wlib.val_folds()
    tau_nat = tau_nat_from(hv, vnat)
    taus = {}
    for f in range(5):
        taus[f] = fit(hv, ref, [c for c in vsyn if fold[c] != f], tau_nat)
    tau_in = fit(hv, ref, vsyn, tau_nat)
    base_cf = {c: decode_full(hv[c], taus[fold[c]]) for c in vsyn}
    base_in = {c: decode_full(hv[c], tau_in) for c in vsyn}
    s_cf = wlib.score(ref, {c: [(a, b) for a, b, *_ in v] for c, v in base_cf.items()}, vsyn)
    s_in = wlib.score(ref, {c: [(a, b) for a, b, *_ in v] for c, v in base_in.items()}, vsyn)
    print('base 5-fold cross-fit', wlib.fmt(s_cf))
    print('base in-sample       ', wlib.fmt(s_in))
    tr = wlib.tau_router()
    print('tau_in equals the shipped tau_syn:', bool(np.allclose(tau_in, tr['tau_syn'])),
          '; tau_nat equals shipped:', bool(np.allclose(tau_nat, tr['tau_nat'])))

    # test: shipped decode with template ids, mapped onto job2only
    ht = wlib.test_hits()
    tsyn = wlib.test_syn_clips()
    td = wlib.test_durations()
    job2 = wlib.read_pred_jsonl(wlib.JOB2ONLY)
    assert sorted(job2) == sorted(c[:-4] for c in tsyn)
    test_dec = {c: decode_full(ht[c], tr['tau_syn']) for c in tsyn}
    # map every job2only event to a decoded hit (same onset within 1 ms)
    mapped = {}
    n_same = n_ext = n_new = 0
    for c in tsyn:
        rows = []
        dec = test_dec[c]
        for a, b in job2[c[:-4]]:
            m = [d for d in dec if abs(d[0] - a) <= 1e-3]
            if m:
                d = m[0]
                kind = 'same' if abs(min(d[1], td[c]) - b) <= 1e-3 else 'job2_extent'
                rows.append({'on': a, 'off': b, 'k': d[2], 'ncc': d[3], 'p': d[4], 'kind': kind})
                n_same += kind == 'same'
                n_ext += kind == 'job2_extent'
            else:
                rows.append({'on': a, 'off': b, 'k': None, 'ncc': None, 'p': None, 'kind': 'job2_new'})
                n_new += 1
        mapped[c] = rows
    print('job2only events: %d same as the shipped decode, %d job2 extents, %d job2 new ; shipped decode events %d' % (
        n_same, n_ext, n_new, sum(len(v) for v in test_dec.values())))
    wlib.dump_pickle(wlib.CACHE / 'base.pkl', {
        'tau_nat': tau_nat, 'taus': taus, 'tau_in': tau_in, 'fold': fold,
        'base_cf': base_cf, 'base_in': base_in, 'score_cf': s_cf, 'score_in': s_in,
        'test_dec': test_dec, 'test_job2': mapped})


if __name__ == '__main__':
    main()
