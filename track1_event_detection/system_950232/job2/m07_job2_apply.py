"""JOB 2 result on test: the extent rule for master-group events and the prefix-scan events.

Rule (generator, validated on 33/33 decoded + 3/3 undecoded validation master events):
a paste of a long noise file starts at the file start and ends at
min(onset + full file length, clip end); full length machine 5.437 s, child 5.211 s,
bird >= 9.077 s (every bird paste seen is cut by the clip end).
Every correction is cross-checked by the label-free change-point estimator of m04
(agreement within 50 ms required).
Writes base_test_job2.pkl ({clip: events} for the 2,082 test synthetic clips = shipped
events with the corrections and the new master events) and job2_changes.json."""
import os, sys, json, pickle, numpy as np
import lib
HERE = os.path.join(os.environ.get('T1_JOB2_OUT', '.'), '')   # where the m0x outputs are written and read
FULL = {125: 5.437, 88: 5.211, 66: 9.077}
tsyn = lib.test_synthetic_clips(); td = lib.test_durations()
shipped = {r['clip_id']: [(e['onset'], e['offset']) for e in r['events']]
           for r in (json.loads(l) for l in open(lib.WORK / 'test' / 'predictions.jsonl') if l.strip())}
E = json.load(open(HERE + 'm04_test_extent.json'))
P = json.load(open(HERE + 'm06_prefix_events.json'))
base = {c: sorted(shipped[c[:-4]]) for c in tsyn}
changes = []
for r in E:
    c = r['clip']; on = r['on']
    new_off = min(on + FULL[r['group']], td[c])
    est_off = on + r['L']
    assert abs(new_off - est_off) <= 0.05, (c, new_off, est_off)
    ev = base[c]
    i = min(range(len(ev)), key=lambda n: abs(ev[n][0] - on))
    a, b = ev[i]; assert abs(a - on) < 1e-3
    tol_new = max(0.2 * (new_off - a), 0.05)
    if abs(new_off - b) > 1e-3:
        ev[i] = (a, round(new_off, 4))
        changes.append({'clip': c, 'kind': 'extent', 'group': r['group'], 'k': r['k'], 'old': [a, b], 'new': [a, round(new_off, 4)],
                        'estimator_off': est_off, 'outside_tolerance': bool(abs(new_off - b) > tol_new)})
    base[c] = sorted(ev)
for r in P['test_new']:
    c = r['clip']; on = round(r['on'], 4); off = round(min(r['on'] + FULL[r['master']], td[c]), 4)
    assert all(off + 0.2 <= a or on >= b + 0.2 for a, b in base[c]), c
    base[c] = sorted(base[c] + [(on, off)])
    changes.append({'clip': c, 'kind': 'new_master_event', 'group': r['master'], 'k': None, 'old': None, 'new': [on, off],
                    'prefix_ncc': r['ncc'], 'outside_tolerance': True})
# geometry check after the changes
bad = [(c, a, b) for c in tsyn for (a, b), (x, y) in zip(base[c][:-1], base[c][1:]) if x < b + 0.2 - 1e-6]
print('changes: %d extents (outside the tolerance of the old end: %d) + %d new master events ; gap violations after: %d' % (
    sum(1 for c in changes if c['kind'] == 'extent'), sum(1 for c in changes if c['kind'] == 'extent' and c['outside_tolerance']),
    sum(1 for c in changes if c['kind'] == 'new_master_event'), len(bad)))
by = {}
for ch in changes: by.setdefault((ch['kind'], ch['group'], ch['outside_tolerance']), 0); by[(ch['kind'], ch['group'], ch['outside_tolerance'])] += 1
print(by)
pickle.dump(base, open(HERE + 'base_test_job2.pkl', 'wb'))
json.dump(changes, open(HERE + 'job2_changes.json', 'w'), indent=1)
