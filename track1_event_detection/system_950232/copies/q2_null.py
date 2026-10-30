"""Q2 decision rule and thresholds from VALIDATION only (written before any test row is read).

Kernels: the 2,849 candidate templates (cache/kernels_cand.json).
Null: the 695 validation synthetic clips,
  (a) free mode: lags whose kernel extent lies in the regions free of every LABELLED event
      (clean speech; the exact analogue of the test search),
  (b) full mode: every lag, including the pastes of the 203 known noise files.
A null row counts as a FALSE detection when the kernel extent at its lag does NOT match a labelled
event of the clip within the scorer tolerance (a row that matches a labelled paste would add a
correct event; it is reported apart as 'correct-extent').

Finding on the null (see q2_theta.json): for natural cuts, S3 alone has heavy, kernel-specific tails
(stationary tones of horns, birds, beeps correlate with other noise files at fixed offsets, up to
S3 21 on false extents), while the whitened NCC S1 of these rows stays low (max 0.377 at S3 >= 12).
True pastes of a kernel's own noise file (validation check q2_val.json, 650 pastes) have S1 median
0.71 and q05 0.39.
Fixed rule for Q2 (stricter than the WMF rule, because the S3 tail is heavier here):
  add (kernel j, lag p) when S3 >= theta_j AND S1 >= S1_MIN, with
  theta_j = max(15, false-extent chance max_j + 5) and S1_MIN = 0.3;
  the 16 kernels that overlap a known source event are not searched (their audio holds part of a
  known noise file; a hit would give a wrong extent).
Checks: false detections of the rule on the whole null (in-sample for theta) and on a 2-fold split
of the null clips (theta from one half, false detections counted on the other half); recall of the
rule on the 650 validation pastes of the 71 source kernels (q2_val.json).
Output: q2_theta.json"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

import kjob  # noqa: E402
import nclib as N  # noqa: E402

SR = N.SR
KF = {f: i for i, f in enumerate(kjob.KF)}
THETA_G = 15.0
MARGIN = 5.0
S1_MIN = 0.3


def main():
    specs = kjob.kernel_specs('cand')
    K = len(specs)
    Lk = np.array([s['b'] - s['a'] for s in specs])
    near = np.array([s['near_source'] for s in specs])
    lab = N.val_labels()
    nf = kjob.load('cand', 'valsyn', 'free')
    nu = kjob.load('cand', 'valsyn', 'full')
    assert len(nf) == 695 and len(nu) == 695, (len(nf), len(nu))
    vsyn = sorted(nu)
    rng = np.random.default_rng(20260923)
    half = {c: int(h) for c, h in zip(vsyn, rng.integers(0, 2, len(vsyn)))}

    def correct_extent(c, j, p):
        on, off = p / SR, (p + Lk[j]) / SR
        for a, b, t in lab[c]:
            tol = max(0.2 * (b - a), 0.05)
            if abs(on - a) <= tol and abs(off - b) <= tol:
                return True
        return False

    M = np.full(K, -np.inf)                       # false-extent chance max (all null clips)
    Mh = np.full((2, K), -np.inf)
    arg = [None] * K
    S1_at_s3 = []                                  # (S3, S1) of false-extent rows with S3 >= 12 (non-near kernels)
    correct_rows = []
    rows_n = {'free': 0, 'full': 0}
    keep_rows = []                                 # candidate rows for the rule checks (S3 >= 12 or S1 >= S1_MIN)
    for name, S in (('free', nf), ('full', nu)):
        for c, R in S.items():
            if not len(R):
                continue
            rows_n[name] += len(R)
            j = R[:, KF['j']].astype(int)
            s3 = R[:, KF['S3']]
            s1 = R[:, KF['S1']]
            hi = np.where(((s3 >= 8) | (s1 >= S1_MIN)) & ~near[j])[0]
            ok = np.ones(len(R), bool)
            for i in hi:
                if correct_extent(c, j[i], int(R[i, KF['p']])):
                    ok[i] = False
                    correct_rows.append((name, N.stem(c), int(j[i]), int(R[i, KF['p']]), float(s3[i]), float(s1[i])))
            sel = ok & ~near[j]
            cm = np.full(K, -np.inf)
            np.maximum.at(cm, j[sel], s3[sel])
            for jj in np.where(cm > M)[0]:
                arg[jj] = (name, N.stem(c), float(cm[jj]))
            M = np.maximum(M, cm)
            Mh[half[c]] = np.maximum(Mh[half[c]], cm)
            for i in hi:
                if ok[i]:
                    keep_rows.append((half[c], int(j[i]), float(s3[i]), float(s1[i]), name, N.stem(c), int(R[i, KF['p']])))
                    if s3[i] >= 12:
                        S1_at_s3.append((float(s3[i]), float(s1[i])))
    fin = np.isfinite(M)
    theta = np.maximum(THETA_G, np.where(fin, M, 0.0) + MARGIN)
    theta[near] = np.inf
    KR = np.array([(h, j, s3, s1) for h, j, s3, s1, *_ in keep_rows]) if keep_rows else np.zeros((0, 4))
    # in-sample false detections of the rule (theta from all null clips)
    jj = KR[:, 1].astype(int)
    fd_all = int(np.sum((KR[:, 2] >= theta[jj]) & (KR[:, 3] >= S1_MIN)))
    # S3-only false detections at the global floor, and the S1 threshold alone
    thr_table = {f'S3>={x}_S1>={y}': int(np.sum((KR[:, 2] >= x) & (KR[:, 3] >= y))) for x in (12, 15, 20) for y in (0.0, 0.2, 0.3)}
    # 2-fold split
    split = {}
    for h in (0, 1):
        th_h = np.maximum(THETA_G, np.where(np.isfinite(Mh[h]), Mh[h], 0.0) + MARGIN)
        th_h[near] = np.inf
        o = KR[:, 0] == 1 - h
        split[f'theta_from_half{h}'] = {
            'false_detections_other_half_rule': int(np.sum(o & (KR[:, 2] >= th_h[jj]) & (KR[:, 3] >= S1_MIN))),
            'false_detections_other_half_S3_only': int(np.sum(o & (KR[:, 2] >= th_h[jj])))}
    A = np.array(S1_at_s3)
    # recall of the rule on the validation pastes of the 71 source kernels
    q2v = json.loads((N.HERE / 'q2_val.json').read_text())
    P = [p for p in q2v['positives'] if p['fit'] and p['S3'] is not None and not p['master']]
    th_src = np.array(q2v['theta_j'])
    chance_src = np.array(q2v['chance_max'])
    rec_rule = sum(1 for p in P if p['S3'] >= th_src[p['j']] and p['S1'] >= S1_MIN)
    rec_rule15 = sum(1 for p in P if p['S3'] >= 15 and p['S1'] >= S1_MIN)
    idx = N.store_index('valsyn')
    R_v = kjob.ranges_for('valsyn')
    out = {'written': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'rule': {'THETA_G': THETA_G, 'MARGIN': MARGIN, 'S1_MIN': S1_MIN,
                                                                      'kernels_not_searched_near_source': int(near.sum())},
           'kernels': K, 'null_clips': len(vsyn), 'null_hours_full': round(sum(idx[c][1] for c in vsyn) / SR / 3600, 3),
           'null_hours_free': {k: round(sum(v - u for c in R_v for u, v in R_v[c][k]) / SR / 3600, 3) for k in ('noise', 'vocal')},
           'rows': rows_n, 'correct_extent_rows': len(correct_rows),
           'correct_extent_rows_passing_rule': sum(1 for n_, c, j, p, s3, s1 in correct_rows if s3 >= theta[j] and s1 >= S1_MIN),
           'correct_extent_examples': sorted(correct_rows, key=lambda r: -r[4])[:8],
           'false_extent_rows_S3_ge_12': len(A),
           'false_extent_S1_quantiles_at_S3_ge_12': dict(zip(['q50', 'q90', 'q99', 'max'], np.round(np.quantile(A[:, 1], [.5, .9, .99, 1]), 3))) if len(A) else None,
           'chance_max_quantiles': dict(zip(['q50', 'q90', 'q99', 'max'], np.round(np.quantile(M[fin & ~near], [.5, .9, .99, 1]), 2))),
           'kernels_theta_above_15': int(np.sum((theta > THETA_G) & np.isfinite(theta))),
           'theta_quantiles': dict(zip(['q50', 'q90', 'q99', 'max'], np.round(np.quantile(theta[np.isfinite(theta)], [.5, .9, .99, 1]), 2))),
           'false_detections_rule_in_sample': fd_all, 'threshold_table_false_extent_rows': thr_table, 'split_half_check': split,
           'positives_src71_nonmaster': len(P), 'positives_recall_rule': rec_rule, 'positives_recall_S3ge15_S1ge0.3': rec_rule15,
           'top_false_extent_chance': sorted([(round(float(M[j]), 2), j, specs[j]['tag'], specs[j]['dur'], arg[j][0], arg[j][1][-40:])
                                              for j in range(K) if arg[j] and not near[j]], reverse=True)[:12],
           'theta': [float(x) if np.isfinite(x) else None for x in theta],
           'chance_max': [float(x) if np.isfinite(x) else None for x in M]}
    print(json.dumps({k: v for k, v in out.items() if k not in ('theta', 'chance_max')}, indent=1, default=str))
    N.dump_json(N.HERE / 'q2_theta.json', out)


if __name__ == '__main__':
    main()
