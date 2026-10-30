"""Apply the master-based extent estimator to every shipped test event of a master group
(machine noise, child talking, bird sound). No label is involved.
Writes m04_test_extent.json: one row per master-group event, with the conflicts against
the other shipped events of the clip."""
import os, sys, json, numpy as np
import lib, audio_store, j2_lib as J, mext
from mgroups import EX, LIB, DUR, tau, GROUPS, group_of, MU, reference
SR = 16000
HERE = os.path.join(os.environ.get('T1_JOB2_OUT', '.'), '')   # where the m0x outputs are written and read
tsyn = lib.test_synthetic_clips(); ht = lib.test_hits(); tdur = lib.test_durations()
acc = J.accepted(ht, tsyn, tau, DUR)
shipped = {r['clip_id']: [(e['onset'], e['offset']) for e in r['events']]
           for r in (json.loads(l) for l in open(lib.WORK / 'test' / 'predictions.jsonl') if l.strip())}
# the shipped synthetic events are the greedy decode: check it, so every event maps to a template
mismatch = 0
for c in tsyn:
    dec = sorted((on, on + DUR[k]) for k, on, v in acc[c])
    shp = sorted(shipped[c[:-4]])
    if len(dec) != len(shp) or any(abs(a - x) > 1e-3 or abs(min(b, tdur[c]) - y) > 1e-3 for (a, b), (x, y) in zip(dec, shp)):
        mismatch += 1
print('clips where the shipped events differ from the decode: %d of %d' % (mismatch, len(tsyn)))
idx, mm = audio_store.load('test')
REF = {g: reference(g) for g in GROUPS}
rows = []
for c in tsyn:
    hits = [(k, on, v) for k, on, v in acc[c] if k in group_of]
    if not hits: continue
    x = audio_store.get(idx, mm, c).astype(np.float64)
    for k, on, v in hits:
        g = group_of[k]; w, mu_ref, cens = REF[g]
        c0 = int(round(on * SR)) - MU[k] + mu_ref
        e = mext.estimate(x, w, c0)
        if e is None: continue
        L = min(e['L'], tdur[c] - on)
        others = [(kk, oo, oo + DUR[kk], vv) for kk, oo, vv in acc[c] if not (kk == k and abs(oo - on) < 1e-9)]
        conf = [(int(kk), round(oo, 3), round(min(ff, tdur[c]), 3), round(vv, 3), kk in group_of) for kk, oo, ff, vv in others
                if oo < on + L + 0.2 and ff > on]
        tolv = max(0.2 * L, 0.05)
        rows.append({'clip': c, 'k': int(k), 'group': int(g), 'on': on, 'ncc': v, 'tmpl_dur': float(DUR[k]),
                     'L': L, 'at_end': e['at_end'], 'clip_end': tdur[c] - on, 'ncc_head': e['ncc_head'], 'g': e['g'],
                     'prom_left': e['prom_left'], 'prom_right': e['prom_right'],
                     'flag': bool(abs(on + DUR[k] - (on + L)) > tolv), 'delta': L - float(DUR[k]), 'conflicts': conf})
json.dump(rows, open(HERE + 'm04_test_extent.json', 'w'), indent=1)
for g in GROUPS:
    R = [r for r in rows if r['group'] == g]
    if not R: continue
    d = np.array([r['delta'] for r in R])
    print('group %d (%s, ref %.3f s): %d events ; flagged (template outside the tolerance of the estimate) %d : longer %d shorter %d ; at ref end %d ; with conflicts %d' % (
        g, LIB[g]['tag'], len(REF[g][0]) / SR, len(R), sum(r['flag'] for r in R), sum(r['flag'] and r['delta'] > 0 for r in R),
        sum(r['flag'] and r['delta'] < 0 for r in R), sum(r['at_end'] for r in R), sum(1 for r in R if r['conflicts'])))
    print('   delta quantiles', np.quantile(d, [0, .1, .25, .5, .75, .9, 1]).round(3))
    Ls = np.array([r['L'] for r in R]); print('   L_hat values (sorted):', np.sort(Ls).round(3).tolist())
for r in rows:
    if r['flag'] or r['conflicts']:
        print('  %-45s k %3d on %6.3f tmpl %.3f -> L %.3f %s head %.2f prom %s/%s conf %s' % (
            r['clip'][-45:], r['k'], r['on'], r['tmpl_dur'], r['L'], 'END' if r['at_end'] else '   ', r['ncc_head'],
            None if r['prom_left'] is None else round(r['prom_left'], 2), None if r['prom_right'] is None else round(r['prom_right'], 2),
            r['conflicts']))
