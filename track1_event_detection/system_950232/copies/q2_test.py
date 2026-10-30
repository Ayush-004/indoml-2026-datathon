"""Q2 on TEST: apply the rule of q2_theta.json (validation only) to the scan of the 2,082
synthetic-routed test clips (kjob cand testsyn free: only lags whose kernel extent lies in the
generator-legal regions that the WMF decode leaves free), one clip at a time.

Rule: (kernel j, lag p) with S3 >= theta_j AND S1 >= S1_MIN; near-source kernels are not searched.
Decode: strongest S3 first; the kernel extent [p, p + L] is added when the generator allows it next
to the WMF events and the events added before (no overlap, gap >= 0.2 s, <= 5 events, edge margins
0.250 / 0.251 s for noise kernels). No test clip sets or checks anything.
Outputs: q2_test.json, q2_added.json, q2_test_near.json (label-free near-threshold rows),
test_synthetic_predictions_wmf_q2.jsonl (only when events are added: the WMF file + new events)."""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

import kjob  # noqa: E402
import nclib as N  # noqa: E402

SR = N.SR
KF = {f: i for i, f in enumerate(kjob.KF)}


def allowed(on, off, vocal, events, dur):
    if len(events) >= N.MAX_EVENTS:
        return False
    for a, b in events:
        if not (off + N.GAP - N.SLACK <= a or on >= b + N.GAP - N.SLACK):
            return False
    if not vocal and (on < N.EDGE_ON - N.SLACK or off > dur - N.EDGE_OFF + N.SLACK):
        return False
    return True


def main():
    specs = kjob.kernel_specs('cand')
    Lk = np.array([s['b'] - s['a'] for s in specs])
    T = json.loads((N.HERE / 'q2_theta.json').read_text())
    theta = np.array([np.inf if x is None else x for x in T['theta']])
    s1min = T['rule']['S1_MIN']
    tf = kjob.load('cand', 'testsyn', 'free')
    assert len(tf) == 2082, len(tf)
    td = N.test_durations()
    wmf = N.read_pred_jsonl(N.WMF_SYN)

    cands, near = [], []
    n_rows = 0
    s3_only = 0
    for c, R in tf.items():
        if not len(R):
            continue
        n_rows += len(R)
        j = R[:, KF['j']].astype(int)
        s3 = R[:, KF['S3']]
        s1 = R[:, KF['S1']]
        th = theta[j]
        s3_only += int(np.sum(s3 >= th))
        for i in np.where((s3 >= th) & (s1 >= s1min))[0]:
            cands.append((float(s3[i]), c, int(j[i]), int(R[i, KF['p']]), R[i]))
        for i in np.where(((s3 >= np.minimum(th, 12.0)) & np.isfinite(th)) | ((s1 >= 0.2) & (s3 >= 8)))[0]:
            near.append({'clip': N.stem(c), 'kernel': int(j[i]), 'tag': specs[j[i]]['tag'], 'S3': round(float(s3[i]), 2),
                         'S1': round(float(s1[i]), 3), 'S0': round(float(R[i, KF['S0']]), 3), 'S5': round(float(R[i, KF['S5']]), 3),
                         'theta_j': round(float(th[i]), 2), 'onset': round(R[i, KF['p']] / SR, 4)})
    cands.sort(key=lambda x: -x[0])
    events = {N.stem(c): list(wmf[N.stem(c)]) for c in tf}
    added, blocked = [], []
    for s3, c, j, p, r in cands:
        st = N.stem(c)
        on, off = p / SR, (p + Lk[j]) / SR
        rec = {'clip': st, 'onset': round(on, 4), 'offset': round(off, 4), 'p': p, 'L': int(Lk[j]), 'kernel': j, 'kernel_tag': specs[j]['tag'],
               'kernel_source_clip': N.stem(specs[j]['clip']), 'kernel_label': [specs[j]['on'], specs[j]['off']],
               'S3': round(s3, 2), 'theta_j': round(float(theta[j]), 2), 'S1': round(float(r[KF['S1']]), 3),
               'S0': round(float(r[KF['S0']]), 3), 'S5': round(float(r[KF['S5']]), 3), 'g': float(r[KF['g']])}
        if allowed(on, off, specs[j]['vocal'], events[st], td[c]):
            events[st] = sorted(events[st] + [(on, off)])
            added.append(rec)
        else:
            blocked.append(rec)
    near.sort(key=lambda r: -r['S3'])
    res = {'rule': T['rule'], 'clips_scanned': len(tf), 'rows': n_rows,
           'rows_S3_ge_theta_any_S1': s3_only, 'rows_passing_rule': len(cands),
           'clips_with_rows_passing_rule': len({x[1] for x in cands}),
           'added': len(added), 'clips_added': len({a['clip'] for a in added}), 'blocked_by_generator_rules': len(blocked),
           'added_tags': Counter(a['kernel_tag'] for a in added).most_common(),
           'near_threshold_rows': len(near)}
    print(json.dumps(res, indent=1))
    for a in added:
        print('ADDED', a)
    for b in blocked[:10]:
        print('BLOCKED', b)
    for n_ in near[:15]:
        print('NEAR', n_)
    N.dump_json(N.HERE / 'q2_test.json', res)
    N.dump_json(N.HERE / 'q2_added.json', {'added': added, 'blocked': blocked})
    N.dump_json(N.HERE / 'q2_test_near.json', near[:500])
    if added:
        rows = []
        for line in open(N.WMF_SYN):
            r = json.loads(line)
            st = r['clip_id']
            new = [a for a in added if a['clip'] == st]
            if new:
                d = round(td[st + '.wav'], 4)
                ev = r['events'] + [{'onset': a['onset'], 'offset': min(a['offset'], d)} for a in new]
                ev = sorted(ev, key=lambda e: (e['onset'], e['offset']))
                rows.append(json.dumps({'clip_id': st, 'events': ev}))
            else:
                rows.append(line.rstrip('\n'))
        (N.HERE / 'test_synthetic_predictions_wmf_q2.jsonl').write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
