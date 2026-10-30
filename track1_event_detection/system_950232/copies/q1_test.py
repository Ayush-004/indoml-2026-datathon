"""Q1 on TEST: the same detector (tile-GLRT S3 >= theta_k, 203 validation templates, exemplar 0) and
the same fixed rule R1 (drop every incumbent event that overlaps a copy, add the copy extent) on the
3,435 natural-routed test clips, one clip at a time. No test label exists or is used; no test clip is
compared with another; theta_k comes from ../wmf/theta.json (natural null, set before).

Incumbent test natural predictions: the natural-routed lines of an earlier build (nclib.BASE_ZIP;
checked against nclib.SUB08_TEST).

Estimate of the LB change (natural side): the validation effect per copy event (in-sample, see
q1_val.json) scaled to the test copy count, plus a label-free PROXY that uses the copy extent as the
label of the copied event (validation: 71/71 copy extents match their label within tolerance, median
|d_on| 0.6 ms, |d_off| 1.0 ms): the proxy counts on test how many copied events the incumbent already
matches and how many incumbent events R1 drops; the unknown part (a dropped event that matched ANOTHER
label; the Dice change) takes the validation rates.

Outputs: test_natural_predictions_copies.jsonl (3,435 natural-routed rows, zip order, stem ids; rows
without a copy are the raw incumbent lines), test_copies.json (every copy: clip, template, tag, S3, S0,
extent, incumbent events it replaces), q1_test.json (counts, estimate)."""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

import nclib as N  # noqa: E402

SR = N.SR


def read_zip_lines(path):
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ['predictions.jsonl']
        blob = z.read('predictions.jsonl')
    out = []
    for raw in blob.split(b'\n'):
        if raw.strip():
            r = json.loads(raw)
            out.append((N.stem(r['clip_id']), raw, r))
    return out


def proxy_effect(pred, copies_by_clip):
    """label-free proxy per clip under R1: the copy extent stands for the label of the copied event.
    Returns per copy: matched_before (an incumbent event matches the copy extent within the scorer
    tolerance), and per clip: dropped incumbent events (overlapping a copy)."""
    rows = []
    n_drop = 0
    n_drop_matching_copy = 0
    for c, cps in copies_by_clip.items():
        pr = pred[c]
        ext = [(a, b) for a, b, *_ in cps]
        mr = N.matched_pairs(ext, pr)            # proxy labels = copy extents
        dropped = [i for i, e in enumerate(pr) if any(N.overlaps(e, x) for x in ext)]
        n_drop += len(dropped)
        n_drop_matching_copy += sum(1 for i in dropped if i in mr.values())
        for ci, cp in enumerate(cps):
            rows.append({'clip': c, 'matched_before': ci in mr})
    n_c = len(rows)
    n_m = sum(r['matched_before'] for r in rows)
    return {'copies': n_c, 'copies_matched_by_incumbent': n_m, 'copies_unmatched': n_c - n_m,
            'incumbent_events_dropped': n_drop, 'dropped_that_match_the_copy': n_drop_matching_copy,
            'dropped_other': n_drop - n_drop_matching_copy}


def val_calibration():
    """the proxy on VALIDATION next to the true label effect: rates for the unknown parts."""
    th = N.theta()
    scan = N.load_wmf_scan('valnat')
    vd = N.val_durations()
    ref = N.val_reference_stem()
    pred = N.read_pred_jsonl(N.SUB08_VAL)
    copies = {}
    for c in N.val_nat_clips():
        ev = N.copy_events(scan.get(c), vd[c], th)
        if ev:
            copies[N.stem(c)] = ev
    px = proxy_effect(pred, copies)
    # true status of the dropped 'other' events: matched to another label (TP lost) or FP
    tp_lost = fp_removed = 0
    for c, cps in copies.items():
        pr = pred[c]
        ext = [(a, b) for a, b, *_ in cps]
        mr_proxy = N.matched_pairs(ext, pr)
        mr_true = N.matched_pairs(ref[c], pr)
        matched_pred_true = set(mr_true.values())
        for i, e in enumerate(pr):
            if any(N.overlaps(e, x) for x in ext) and i not in mr_proxy.values():
                if i in matched_pred_true:
                    # it matched a label: the copied label itself (then the copy takes over) or another one
                    lab = [r for r, p in mr_true.items() if p == i][0]
                    a, b = ref[c][lab]
                    is_copy_label = any(N.iou((a, b), x) > 0.5 for x in ext)
                    if not is_copy_label:
                        tp_lost += 1
                else:
                    fp_removed += 1
    px['dropped_other_true_status'] = {'matched_another_label': tp_lost, 'false_positive': fp_removed}
    return px, copies


def main():
    th = N.theta()
    EX = N.exemplars()
    td = N.test_durations()
    nat = N.test_nat_clips()
    scan = N.load_scan_dir(N.CACHE / 'scan_testnat')
    assert set(scan) == set(nat) and len(nat) == 3435, (len(scan), len(nat))

    base = read_zip_lines(N.BASE_ZIP)
    assert len(base) == 5517
    nat_s = {N.stem(c) for c in nat}
    base_nat = [(s, raw, r) for s, raw, r in base if s in nat_s]
    assert len(base_nat) == 3435
    inc = {s: [(float(e['onset']), float(e['offset'])) for e in r['events']] for s, raw, r in base_nat}
    paired = N.read_pred_jsonl(N.SUB08_TEST)
    same_as_paired = sum(1 for s in inc if paired.get(s) == inc[s])

    # ---- detection (one clip at a time)
    copies = {}
    raw_rows = []
    s3_all_max = []
    for c in nat:
        R = scan[c]
        if len(R):
            k = R[:, N.F['k']].astype(int)
            above = R[R[:, N.F['S3']] >= th[k]]
            for r in above:
                raw_rows.append((N.stem(c), int(r[N.F['k']]), int(r[N.F['p']]), float(r[N.F['S3']])))
            # the largest S3 of the clip below threshold (diagnostic of the gap)
            below = R[R[:, N.F['S3']] < th[k]]
            if len(below):
                s3_all_max.append(float(below[:, N.F['S3']].max()))
        ev = N.copy_events(R, td[c], th)
        if ev:
            copies[N.stem(c)] = ev
    n_cp = sum(len(v) for v in copies.values())
    s3_cp = np.array([e[3] for v in copies.values() for e in v])
    below_max = np.array(s3_all_max)
    print('test natural: template-level detections %d (templates %d) ; copy events %d in %d clips' % (
        len(raw_rows), len({x[1] for x in raw_rows}), n_cp, len(copies)))

    # ---- R1 on the incumbent
    out_lines = []
    table = []
    n_changed = 0
    for s, raw, r in base_nat:
        if s not in copies:
            out_lines.append(raw)
            continue
        d = td[s + '.wav']
        new = N.normalise(N.rule_override(inc[s], copies[s]), d)
        if new != inc[s]:
            n_changed += 1
        out_lines.append(json.dumps({'clip_id': s, 'events': [{'onset': a, 'offset': b} for a, b in new]}).encode())
        for on, off, k, s3, s0, p in copies[s]:
            table.append({'clip': s, 'k': k, 'tag': EX[k]['tag'], 'master': N.MASTER_OF.get(k),
                          'S3': round(s3, 2), 'S0': round(s0, 3), 'onset': round(on, 4), 'offset': round(min(off, d), 4),
                          'dur': round(min(off, d) - on, 4), 'clip_dur': d,
                          'incumbent_overlapping': [[a, b] for a, b in inc[s] if N.overlaps((a, b), (on, off))],
                          'shipped_override_already': bool(s0 >= 0.9)})
    p = N.HERE / 'test_natural_predictions_copies.jsonl'
    p.write_bytes(b''.join(x + b'\n' for x in out_lines))

    # ---- checks
    chk = N.read_pred_jsonl(p)
    ids = [json.loads(x)['clip_id'] for x in out_lines]
    bad = [(s, a, b) for s, ev in chk.items() for a, b in ev if not (0 <= a < b <= td[s + '.wav'])]
    checks = {'rows': len(out_lines), 'ids_in_zip_order': ids == [s for s, _, _ in base_nat],
              'ids_with_wav': sum(1 for x in ids if x.endswith('.wav')), 'events_out_of_bounds': len(bad),
              'clips_changed': n_changed, 'events_before': sum(len(v) for v in inc.values()),
              'events_after': sum(len(v) for v in chk.values()),
              'incumbent_equals_sub08_test_paired_natural_lines': same_as_paired}
    print(checks)

    # ---- proxy on test and its validation calibration
    px_test = proxy_effect(inc, copies)
    px_val, val_copies = val_calibration()
    q1v = json.loads((N.HERE / 'q1_val.json').read_text())
    print('proxy test', px_test)
    print('proxy val ', px_val)

    res = {'detections': {'template_level': len(raw_rows), 'templates': len({x[1] for x in raw_rows}),
                          'copy_events': n_cp, 'copy_clips': len(copies),
                          'S3_quantiles_of_copies': np.round(np.quantile(s3_cp, [0, .1, .5, .9, 1]), 2).tolist() if n_cp else [],
                          'per_clip_max_S3_below_theta_quantiles': np.round(np.quantile(below_max, [.5, .99, .999, 1]), 2).tolist(),
                          'clips_with_below_theta_max_ge_12': int(np.sum(below_max >= 12)),
                          'tags': Counter(t['tag'] for t in table).most_common(),
                          'templates_hit': sorted({t['k'] for t in table}),
                          'masters': Counter(t['master'] for t in table if t['master']).most_common(),
                          'S0_ge_0.9_already_in_shipped_override': sum(t['shipped_override_already'] for t in table)},
           'checks': checks, 'proxy_test': px_test, 'proxy_val': px_val,
           'val_true_effect_R1': q1v['rules']['R1_override']}
    N.dump_json(N.HERE / 'test_copies.json', table)
    N.dump_json(N.HERE / 'q1_test.json', res)


if __name__ == '__main__':
    main()
