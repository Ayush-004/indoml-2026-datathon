"""Apply the final rules to the 2,082 synthetic-routed test clips, starting from the
uploaded synthetic file test_synthetic_predictions_v2_job2only.jsonl (203 validation templates +
job2 master extents; no test-discovered template).

Inputs: cache/base.pkl (job2only events mapped to their decoded template / lag), cache/scan_testsyn
(stage-1 candidates with the tile statistics), cache/atlag_testsyn_base.pkl (statistics at the lag
of every job2only event), theta.json (null thresholds; natural clips only), level ranges of all
validation events. No test clip is used to set anything; no test clip is compared with another.

    python w09_test.py CONFIG_JSON
CONFIG_JSON e.g. '{"W": 6.0, "theta_r": null, "factor": null, "masters": true}'
Output: test_synthetic_predictions_wmf.jsonl, test_changes.json"""
import json
import sys

import numpy as np

import wlib
import w06_rules as R
from w02_scan import load_scan

SR = wlib.SR


def impossible_flags(e, ev, s3, clip_dur, n_clip, verified=20.0):
    """Label-free generator checks of an event that REMOVE takes out: can ANY legal reference
    event [a, b] match it within the scorer tolerance t(d) = max(0.2 d, 0.05 s), d = b - a?
    Legal (validation generator rules, used label-free as in v3): a non-master noise paste has
    a >= 0.250 s and b <= clip end - 0.251 s (R6); a reference keeps a gap >= 0.200 s to every
    other reference (R8); a verified paste (another event of the clip with S3 >= 20, an exact copy
    of its template) is a reference; a clip holds at most 5 references (R8).
    The search runs over d = 0.01 .. 12 s in 1 ms steps."""
    on, off, k = e[0], e[1], e[2]
    lo_a, hi_b = 0.0, clip_dur
    if not R.VOCAL[k] and R.MASTER_OF.get(k) is None:
        lo_a, hi_b = R.EDGE_ON, clip_dur - R.EDGE_OFF
    ver = [f for f, s in zip(ev, s3) if f is not e and s is not None and np.isfinite(s) and s >= verified]

    def feasible(lo_a, hi_b, blocks):
        d = np.arange(0.01, 12.0, 0.001)
        t = np.maximum(0.2 * d, 0.05)
        lo = np.maximum(np.maximum(lo_a, on - t), off - t - d)
        hi = np.minimum(on + t, np.minimum(hi_b, off + t) - d)
        ok = lo <= hi
        if not ok.any():
            return False
        if not blocks:
            return True
        # a reference must also keep 0.2 s from every verified paste: check the admissible a-range
        for di, l, h in zip(d[ok], lo[ok], hi[ok]):
            a = np.linspace(l, h, 5)
            b = a + di
            good = np.ones(len(a), bool)
            for (x, y) in blocks:
                good &= (b + R.GAP <= x + 1e-9) | (a >= y + R.GAP - 1e-9)
            if good.any():
                return True
        return False

    edge_ok = feasible(lo_a, hi_b, [])
    gap_ok = feasible(lo_a, hi_b, [(f[0], f[1]) for f in ver])
    count = (n_clip > R.MAX_EVENTS) and (len(ver) >= R.MAX_EVENTS)
    return {'edge_impossible': bool(not edge_ok), 'gap_impossible': bool(edge_ok and not gap_ok),
            'in_clip_with_more_than_5': bool(count), 'certain_fp': bool((not gap_ok) or count)}


def main(cfg):
    B = wlib.load_pickle(wlib.CACHE / 'base.pkl')
    E = wlib.load_pickle(wlib.CACHE / 'events.pkl')
    A = wlib.load_pickle(wlib.CACHE / 'atlag_testsyn_base.pkl')
    T = json.loads((wlib.HERE / 'theta.json').read_text())
    theta = np.array(T['theta'])
    tsyn = wlib.test_syn_clips()
    td = wlib.test_durations()
    job2_lines = {}
    for line in open(wlib.JOB2ONLY):
        r = json.loads(line)
        job2_lines[r['clip_id']] = r
    scan = load_scan('testsyn')
    ranges = R.level_ranges(E['events'])
    out_rows = []
    changes = {'added': [], 'removed': [], 'rechosen': [], 'config': cfg}
    n_before = n_after = 0
    for c in tsyn:
        stem = c[:-4]
        rows = B['test_job2'][c]
        # events: (on, off, k, src) with src = index into the verbatim job2only events
        ev = []
        s3 = []
        at = {(int(r[0]), int(r[1])): r for r in A.get(c, [])}
        for j, r in enumerate(rows):
            k = r['k'] if r['k'] is not None else -1
            ev.append((r['on'], r['off'], k, ('job2', j)))
            if r['k'] is not None and (r['k'], r['p']) in at:
                s3.append(float(at[(r['k'], r['p'])][5]))
            else:
                s3.append(None)
        n_before += len(ev)
        # REMOVE
        if cfg.get('theta_r') is not None:
            keep, ks3 = [], []
            n_clip = len(ev)
            for e, s in zip(ev, s3):
                if (e[2] >= 0 and s is not None and np.isfinite(s) and s < cfg['theta_r']
                        and R.MASTER_OF.get(e[2]) is None):
                    row = {'clip': stem, 'onset': e[0], 'offset': e[1], 'k': e[2],
                           'tag': R.EX[e[2]]['tag'], 'S3': round(s, 2)}
                    row.update(impossible_flags(e, ev, s3, td[c], n_clip))
                    changes['removed'].append(row)
                else:
                    keep.append(e)
                    ks3.append(s)
            ev, s3 = keep, ks3
        cands = R.candidates(scan.get(c), c, False, min_s3=5.0)
        if cfg.get('factor') is not None:
            ev2, ch = R.apply_rechoose(ev, s3, cands, td[c], cfg['factor'], ranges, cfg.get('W'))
            for old, new, so, sn in ch:
                changes['rechosen'].append({'clip': stem, 'old': [old[0], old[1]], 'old_k': old[2], 'new': [round(new[0], 4), round(new[1], 4)],
                                            'new_k': new[2], 'S3_old': round(so, 2), 'S3_new': round(sn, 2)})
            ev = ev2
        ev, added = R.apply_add(ev, cands, td[c], theta, ranges, cfg.get('W'), cfg.get('masters', True))
        for a in added:
            changes['added'].append({'clip': stem, 'onset': round(a['on'], 4), 'offset': round(min(a['off'], td[c]), 4),
                                     'k': a['k'], 'tag': R.EX[a['k']]['tag'], 'S3': round(a['S3'], 2),
                                     'S0': round(a['S0'], 3), 'S1': round(a['S1'], 3), 'S5': round(a['S5'], 3),
                                     'level_dB': round(a['level'], 2),
                                     'level_range_dB': [round(x, 2) for x in ranges.get(a['k'], (np.nan, np.nan))],
                                     'theta_k': round(float(theta[a['k']]), 2)})
        # serialise: verbatim job2only events where unchanged, new events rounded like the job2only file
        evs = []
        orig = job2_lines[stem]['events']
        for on, off, k, src in sorted(ev):
            if isinstance(src, tuple) and src[0] == 'job2':
                evs.append(orig[src[1]])
            else:
                a = max(0.0, round(float(on), 4))
                b = min(round(td[c], 4), round(float(off), 4))
                if b > a:
                    evs.append({'onset': a, 'offset': b})
        evs = sorted(evs, key=lambda e: (e['onset'], e['offset']))
        n_after += len(evs)
        out_rows.append({'clip_id': stem, 'events': evs})
    out_rows.sort(key=lambda r: r['clip_id'])
    p = wlib.HERE / 'test_synthetic_predictions_wmf.jsonl'
    with open(p, 'w') as f:
        for r in out_rows:
            f.write(json.dumps(r) + '\n')
    # checks against job2only
    same = sum(1 for r in out_rows if r['events'] == job2_lines[r['clip_id']]['events'])
    gaps = sum(1 for r in out_rows for x, y in zip(r['events'][:-1], r['events'][1:]) if y['onset'] < x['offset'] + 0.2 - 1e-6)
    gaps0 = sum(1 for r in job2_lines.values() for x, y in zip(r['events'][:-1], r['events'][1:]) if y['onset'] < x['offset'] + 0.2 - 1e-6)
    changes['summary'] = {'clips': len(out_rows), 'events_before': n_before, 'events_after': n_after,
                          'added': len(changes['added']), 'removed': len(changes['removed']),
                          'rechosen': len(changes['rechosen']), 'clips_identical_to_job2only': same,
                          'clips_changed': len(out_rows) - same, 'gaps_below_0.2s_after': gaps,
                          'gaps_below_0.2s_job2only': gaps0,
                          'max_events_per_clip': max(len(r['events']) for r in out_rows)}
    wlib.dump_json(wlib.HERE / 'test_changes.json', changes)
    print(json.dumps(changes['summary'], indent=1))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
