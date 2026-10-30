"""Master pastes found by the prefix scan: validation recall/precision against the labels,
and on test the pastes that no shipped master-group event covers (new events).
Event rule (validated in m03, 33/33 validation events): [p, min(p + FULL, clip end)].
Threshold per master: max(0.5, highest natural-clip peak + 0.03). This script takes the natural peaks of m05
(validation natural and test natural clips, label-free). The deployed build used the same rule on validation
natural clips and the training release only (no test audio); both give 0.5 for all three masters."""
import os, sys, json, pickle, numpy as np
import lib, j2_lib as J
from mgroups import LIB, DUR, tau, GROUPS, group_of
SR = 16000
HERE = os.path.join(os.environ.get('T1_JOB2_OUT', '.'), '')   # where the m0x outputs are written and read
S = pickle.load(open(HERE + 'm05_prefix_scan.pkl', 'rb'))['res']
FULL = {125: 5.437, 88: 5.211, 66: 9.077}
nat_max = {m: max(max([p[1] for p in d[m]], default=0.0) for s in ('val_nat', 'test_nat') for d in S[s].values()) for m in GROUPS}
THR = {m: max(0.5, nat_max[m] + 0.03) for m in GROUPS}
print('natural max NCC per master', {m: round(v, 3) for m, v in nat_max.items()}, '-> threshold', {m: round(v, 3) for m, v in THR.items()})

def cands(R, dur):
    out = {}
    for c, d in R.items():
        L = []
        for m in GROUPS:
            for j, v in d[m]:
                if v >= THR[m]:
                    on = j / SR; L.append((m, on, min(on + FULL[m], dur[c]), v))
        out[c] = L
    return out

# ---------------- validation ----------------
ref = lib.val_reference(); vd = lib.val_durations(); hv = lib.val_hits()
vsyn = sorted(c for c in ref if lib.val_is_synthetic()[c])
acc = J.accepted(hv, vsyn, tau, DUR)
C = cands(S['val_syn'], vd)
tp = fp = 0; rows = []
for c in vsyn:
    for m, on, off, v in C[c]:
        lab = [(a, b) for a, b in ref[c] if abs(a - on) <= 0.05]
        dec = [(k, o) for k, o, vv in acc[c] if abs(o - on) <= 0.05 and k in group_of]
        ok_end = bool(lab and abs(off - lab[0][1]) <= max(0.2 * (lab[0][1] - lab[0][0]), 0.05))
        rows.append((c, m, on, off, v, bool(lab), ok_end, bool(dec)))
n_master_lab = 0
for c in vsyn:
    for a, b in ref[c]:
        # a labelled master event: the residual matches a master cut (use the library ownership)
        pass
lab_hit = [r for r in rows if r[5]]
print('VALIDATION synthetic: %d prefix candidates ; on a labelled onset %d (end correct %d) ; not on a label %d ; not decoded by the shipped decode %d (of which on a label %d, end correct %d)' % (
    len(rows), len(lab_hit), sum(r[6] for r in rows), len(rows) - len(lab_hit),
    sum(1 for r in rows if not r[7]), sum(1 for r in rows if not r[7] and r[5]), sum(1 for r in rows if not r[7] and r[6])))
for r in rows:
    if not r[5] or not r[6] or not r[7]:
        print('   ', r[0][-40:], 'master', r[1], 'on %.3f off %.3f ncc %.3f label %s end_ok %s decoded %s' % r[2:])
# ---------------- test ----------------
tsyn = lib.test_synthetic_clips(); td = lib.test_durations(); ht = lib.test_hits()
tacc = J.accepted(ht, tsyn, tau, DUR)
C = cands(S['test_syn'], td)
new = []; known = 0
for c in tsyn:
    for m, on, off, v in C[c]:
        dec = [(k, o) for k, o, vv in tacc[c] if abs(o - on) <= 0.05 and k in group_of]
        if dec: known += 1; continue
        others = [(int(k), round(o, 3), round(min(o + DUR[k], td[c]), 3), round(vv, 3)) for k, o, vv in tacc[c]
                  if o < off + 0.2 and o + DUR[k] > on - 0.2]
        new.append({'clip': c, 'master': int(m), 'on': on, 'off': off, 'ncc': v, 'clip_dur': td[c], 'conflicts': others})
print('TEST synthetic: %d prefix candidates ; on a shipped master-group event %d ; new %d (with conflicts %d)' % (
    known + len(new), known, len(new), sum(1 for r in new if r['conflicts'])))
for r in new:
    print('   %-45s master %3d on %.3f off %.3f (len %.3f) ncc %.3f clip %.3f conflicts %s' % (
        r['clip'][-45:], r['master'], r['on'], r['off'], r['off'] - r['on'], r['ncc'], r['clip_dur'], r['conflicts']))
json.dump({'thr': THR, 'nat_max': nat_max, 'val_rows': rows, 'test_new': new, 'test_known': known},
          open(HERE + 'm06_prefix_events.json', 'w'), indent=1, default=float)
