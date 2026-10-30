"""vote.py: ONE pipeline for EVERY clip (natural and synthetic together). No router, no template, no library copy.

It turns the per-frame outputs of the member models (export_frames.py) into predictions.jsonl:

  N1  per member and input shift: frames -> P2 decoder at 0.75 (nms 0, min 0.08 s) -> refiner (prune 0.70,
      cross-channel NMS 0.7 keep, empty-clip cascade 0.65) -> remove the shift lag.
      LTL6 (--c4, 950232 only): in clips with wav D >= 6.0 s every candidate >= 2.0 s with min(score, peak) >= 0.60
      gets score and peak >= 0.75.
  N2  the member-shift sets (4 per member; a member listed twice counts twice): class-agnostic clustering at IoU 0.55
      (pipeline fuse.fuse_group); a cluster is kept when best score x (supporting sets / all sets) >= 0.5. The refined
      events carry no score, so every score is 1.0: a cluster needs support from half of the sets, and its endpoints
      are the plain means of its members.
      950232 (--c4): the 24 sets of P2, S1, R10, R10, PSR10B, PSR10B; accept support >= 12 of 24.
      LACC (--c4): clusters >= 1.0 s need 11. LPAD8 (--c4): clips D >= 8.0 s, accepted events >= 1.0 s get 7.5 % of
      their length at each end.
  N3  packaging: clamp to [0, D], round to 0.1 ms.
  N4  E0 endpoint pairing (950232 only; off with --no-e0): IoU >= 0.5 one-to-one, 50/50 average for base events < 2 s.
  N6  empty clip -> the fallback donors in order (each: its 4 shift runs decoded at 0.55 and fused, support >= 2 of 4)
      -> floor [0, D] for a clip that is still empty. C8: R10-L2, then PSR10B-L2. 950232: R10, R10S1, SOUP2.
  post LD1R (--c4): clips D >= 8.0 s whose line has no event >= 0.9 D: if >= 6 of the 8 P2/S1 member-shift sets have
      a pre-NMS candidate >= 0.9 D_in with min(score, peak) >= 0.65, the longest event becomes [0, D].

The 950232 natural lines also have a library-copy step (Q1, ../system_950232/copies) between N4 and N6; it is not
part of this file. Labels are never read here. CPU only, 2 threads, numpy only (no torch). Writes only where --out
points.

  python vote.py --split test --clips test_clips.txt --durations test-durations.json --frames-root frames \
      --members P2L3,S1L3,R10L2,PSR10BL2 --no-e0 --fb-donors R10L2,PSR10BL2 --out C8.predictions.jsonl
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '2'
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from indoml_pipeline import common, fuse as FU, refine as RF  # noqa: E402

assert 'torch' not in sys.modules
SHIFTS = (0, 160, 320, 480)
DECODER = {'threshold': 0.75, 'nms_iou': 0.0, 'min_duration': 0.08}
PARAMS = {'boundary': None, 'cascade': 0.65, 'cross': [0.7, 'keep'], 'prune': [0.7, False, 1e9], 'union': None}
CASCADE = 0.65
FTWO = ['P2', 'S1', 'R10', 'R10', 'PSR10B', 'PSR10B']
NSETS = 24
FUSE_IOU = 0.55
FB_THR = 0.55
FB_PARAMS = {'boundary': None, 'cascade': None, 'cross': [0.7, 'keep'], 'prune': None, 'union': None}
LTL6 = {'q': 0.60, 'min_d': 6.0, 'min_len': 2.0, 'floor': 0.75}
LACC = {'min_s': 1.0, 'k': 11}
LPAD8 = {'p': 0.075, 'min_d': 8.0, 'min_s': 1.0}
LD1R = {'min_d': 8.0, 'whole': 0.9, 'min_sp': 0.65, 'min_sets': 6}
ORIG_CI = RF.candidate_intervals

# Inputs (set by main() from the command line; audit_guard.py re-keys DURS and E0):
#   frame stores  <frames root>/<member>/<split>/frames_shift<k>.npz, k in SHIFTS (export_frames.py writes them)
#   DURS[split]   json clip -> wav duration in seconds (from the wav headers)
#   E0[split]     the E0 predictions jsonl (950232 only; not read with --no-e0)
FRAMES_ROOT = Path(os.environ.get('T1_FRAMES', 'frames'))
E0 = {}
DURS = {}


def stem(c):
    return c[:-4] if c.endswith('.wav') else c


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def store_dirs(split, member):
    full = FRAMES_ROOT / member / split
    if not all((full / f'frames_shift{s}.npz').exists() for s in SHIFTS):
        raise SystemExit(f'STOP: member {member} has no frames for {split} ({full})')
    return [full]


# ------------------------------------------------------------------ N1
def hooked_candidates(dur_of, q):
    def hooked(row, presence, distances, *args, **kwargs):
        cats = ORIG_CI(row, presence, distances, *args, **kwargs)
        if dur_of[stem(str(row['clip_id']))] < LTL6['min_d']:
            return cats
        out = []
        for group in cats:
            new_group, changed = [], False
            for x in group:
                if (x['offset'] - x['onset']) >= LTL6['min_len'] and min(x['score'], x['peak']) >= q:
                    y = dict(x)
                    y['score'] = max(x['score'], LTL6['floor'])
                    y['peak'] = max(x['peak'], LTL6['floor'])
                    if y['score'] != x['score'] or y['peak'] != x['peak']:
                        changed = True
                    x = y
                new_group.append(x)
            if changed:
                new_group = sorted(new_group, key=lambda z: (-z['score'], -z['peak'], z['onset']))
            out.append(new_group)
        return out
    return hooked


def refine_store(npz, shift, want, done, dur_of, ltl6_q):
    """N1 for one frame store and shift: decode + refine the clips of `want` (stems) not in `done`."""
    lag = shift / common.SR
    blob = np.load(npz, allow_pickle=True)
    ids = [str(c) for c in blob['clip_ids']]
    durations = np.asarray(blob['durations'], dtype=np.float64)
    bounds = np.asarray(blob['offsets'], dtype=np.int64)
    presence = blob['presence']
    distances = blob['distances']
    RF.P2_DECODER = DECODER
    RF.candidate_intervals = hooked_candidates(dur_of, ltl6_q) if ltl6_q is not None else ORIG_CI
    try:
        for k in sorted(range(len(ids)), key=lambda j: ids[j]):
            s = stem(ids[k])
            if s not in want or s in done:
                continue
            a, b = int(bounds[k]), int(bounds[k + 1])
            pr = np.asarray(presence[a:b], np.float32)
            ds = np.asarray(distances[a:b], np.float32)
            row = {'clip_id': ids[k], 'duration': float(durations[k]), 'frame_hop': common.FRAME_HOP,
                   'frame_offset': common.FRAME_OFFSET}
            cats = RF.candidate_intervals(row, pr, ds)
            decoded = RF.decode_candidates(row, cats, RF.P2_DECODER, keep_meta=True)
            entry = {'duration': float(durations[k]), 'presence': pr, 'distances': ds,
                     'candidates': [dict(c) for g in cats for c in g], 'p2_events': decoded['events']}
            pairs = RF.run_pipeline(entry, PARAMS, RF.cascade_sets(entry, (CASCADE,)))
            duration = entry['duration'] - lag
            events = []
            for x, y in pairs:
                x = max(0.0, x - lag)
                y = min(duration, y - lag)
                if y > x:
                    events.append({'onset': x, 'offset': y})
            done[s] = {'clip_id': ids[k], 'duration': duration, 'events': events}
    finally:
        RF.candidate_intervals = ORIG_CI
    return done


def member_sets(split, clips, dur_of, ltl6_q, log):
    cache = {}
    for m in dict.fromkeys(FTWO):
        per = []
        for s in SHIFTS:
            done = {}
            for d in store_dirs(split, m):
                refine_store(Path(d) / f'frames_shift{s}.npz', s, clips, done, dur_of, ltl6_q)
            missing = clips - set(done)
            if missing:
                raise SystemExit(f'STOP: member {m} shift {s}: {len(missing)} clips have no frames, e.g. '
                                 f'{sorted(missing)[:2]}')
            per.append(done)
        cache[m] = per
        log(f'N1 {m} refined ({len(clips)} clips x 4 shifts)')
    return [cache[m][i] for m in FTWO for i in range(len(SHIFTS))]


# ------------------------------------------------------------------ N2 (clustering + acceptance)
def clusters_of_clip(sets, clip):
    duration = next(float(g[clip]['duration']) for g in sets if clip in g)
    items = []
    for index, g in enumerate(sets):
        for e in g.get(clip, {}).get('events', []):
            items.append((float(e['onset']), float(e['offset']), float(e.get('score', 1.0)), index))
    clusters = []
    for onset, offset, score, index in sorted(items, key=lambda x: (-x[2], x[0], x[1])):
        target = None
        for cluster in clusters:
            if FU.iou_1d((onset, offset), (cluster['onset'], cluster['offset'])) >= FUSE_IOU:
                target = cluster
                break
        weight = max(score, 1e-6)
        if target is None:
            clusters.append({'onset': onset, 'offset': offset, 'best': score, 'weight_sum': weight,
                             'on_sum': onset * weight, 'off_sum': offset * weight, 'sets': {index}})
            continue
        target['weight_sum'] += weight
        target['on_sum'] += onset * weight
        target['off_sum'] += offset * weight
        target['onset'] = target['on_sum'] / target['weight_sum']
        target['offset'] = target['off_sum'] / target['weight_sum']
        target['best'] = max(target['best'], score)
        target['sets'].add(index)
    return duration, clusters


def fuse_clip(duration, clusters, D, lacc, lpad8):
    band = lacc
    pad = lpad8 and D >= LPAD8['min_d']
    events = []
    for c in clusters:
        support = len(c['sets'])
        score = float(c['best'] * (min(support, NSETS) / NSETS))
        on, off = c['onset'], c['offset']
        if band:
            k = LACC['k'] if (off - on) >= LACC['min_s'] else 12
            if support < k:
                continue
        elif score < 0.5:
            continue
        if pad and (off - on) >= LPAD8['min_s']:
            w = off - on
            on, off = on - LPAD8['p'] * w, off + LPAD8['p'] * w
        onset = max(0., min(on, duration))
        offset = max(0., min(off, duration))
        if offset <= onset:
            continue
        events.append((float(onset), float(offset)))
    return sorted(events)


# ------------------------------------------------------------------ N3, N4
def package(events, D):
    out = [(max(0.0, round(a, 4)), min(D, round(b, 4))) for a, b in events if b > a]
    for a, b in out:
        assert 0.0 <= a < b <= D, (a, b, D)
    return out


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def gpair(s, p, w=0.5, max_len=2.0, thr=0.5):
    cand = sorted(((iou(a, b), i, j) for i, a in enumerate(s) for j, b in enumerate(p)), reverse=True)
    ui, uj, mate = set(), set(), {}
    for v, i, j in cand:
        if v < thr:
            break
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); mate[i] = j
    res = []
    for i, a in enumerate(s):
        if i in mate and (max_len is None or (a[1] - a[0]) < max_len):
            b = p[mate[i]]
            res.append(((1 - w) * a[0] + w * b[0], (1 - w) * a[1] + w * b[1]))
        else:
            res.append(a)
    return sorted(res)


def read_events(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[stem(r['clip_id'])] = [(float(e['onset']), float(e['offset'])) for e in r['events']]
    return out


# ------------------------------------------------------------------ N6 (fallback donors)
def fb_decode(split, donor, clips, durs):
    sets = []
    for s in SHIFTS:
        lag = s / common.SR
        recs = {}
        for d in store_dirs(split, donor):
            p = Path(d) / f'frames_shift{s}.npz'
            blob = np.load(p, allow_pickle=True)
            ids = [str(c) for c in blob['clip_ids']]
            want = {c for c in ids if stem(c) in clips and stem(c) not in {stem(x) for x in recs}}
            if not want:
                continue
            saved = RF.P2_DECODER
            RF.P2_DECODER = {'threshold': FB_THR, 'nms_iou': 0.0, 'min_duration': 0.08}
            try:
                entries = RF.entries_from_frames(p, clips=want)
            finally:
                RF.P2_DECODER = saved
            for clip, entry in sorted(entries.items()):
                pairs = RF.run_pipeline(entry, FB_PARAMS, None)
                duration = entry['duration'] - lag
                ev = []
                for a, b in pairs:
                    a = max(0.0, a - lag)
                    b = min(duration, b - lag)
                    if b > a:
                        ev.append({'onset': a, 'offset': b})
                recs[stem(clip)] = {'clip_id': stem(clip), 'duration': duration, 'events': ev}
        missing = set(clips) - set(recs)
        if missing:
            return None, missing
        sets.append(recs)
    fused, _ = FU.fuse(sets, iou_threshold=0.55, support_scale=True, min_score=0.5, use_channel=False)
    out = {}
    for c in clips:
        raw = [(e['onset'], e['offset']) for e in fused[c]['events']]
        ev = [(max(0.0, round(a, 4)), min(durs[c], round(b, 4))) for a, b in raw if b > a]
        out[c] = sorted((a, b) for a, b in ev if b > a)
    return out, set()


# ------------------------------------------------------------------ LD1R
def best_whole(npz, clips):
    blob = np.load(npz, allow_pickle=True)
    ids = [str(c) for c in blob['clip_ids']]
    durs = np.asarray(blob['durations'], float)
    b = np.asarray(blob['offsets'], np.int64)
    pres, dist = blob['presence'], blob['distances']
    out = {}
    for k, c in enumerate(ids):
        s = stem(c)
        if s not in clips or s in out:
            continue
        D = durs[k]
        row = {'clip_id': c, 'duration': D, 'frame_hop': common.FRAME_HOP, 'frame_offset': common.FRAME_OFFSET}
        cands = ORIG_CI(row, np.asarray(pres[b[k]:b[k + 1]], np.float32), np.asarray(dist[b[k]:b[k + 1]], np.float32))
        best = 0.0
        for g in cands:
            for x in g:
                if x['offset'] - x['onset'] >= LD1R['whole'] * D:
                    best = max(best, min(x['score'], x['peak']))
        out[s] = float(best)
    return out


def ld1r_counts(split, clips):
    cnt = Counter()
    for m in ('P2', 'S1'):
        for s in SHIFTS:
            got = {}
            for d in store_dirs(split, m):
                for c, v in best_whole(Path(d) / f'frames_shift{s}.npz', clips - set(got)).items():
                    got[c] = v
            assert set(got) == clips
            for c, v in got.items():
                cnt[c] += int(v >= LD1R['min_sp'])
    return cnt


def extend_longest(ev, D):
    ev = [(float(a), float(b)) for a, b in ev]
    if not ev:
        return [(0.0, float(D))]
    i = max(range(len(ev)), key=lambda j: ev[j][1] - ev[j][0])
    ev[i] = (0.0, float(D))
    return sorted(ev)


# ------------------------------------------------------------------ the pipeline
def run(split, clips, c4, log, fb_donors=('R10', 'R10S1', 'SOUP2'), e0_on=True, e0_w=0.5, e0_max_len=2.0, e0_thr=0.5):
    durs = {stem(k): float(v) for k, v in json.loads(DURS[split].read_text()).items()}
    clips = set(clips)
    sets = member_sets(split, clips, durs, LTL6['q'] if c4 else None, log)
    stages = {}
    fused = {}
    for c in sorted(clips):
        duration, cl = clusters_of_clip(sets, c)
        fused[c] = fuse_clip(duration, cl, durs[c], c4, c4)
    stages['N2'] = {c: list(v) for c, v in fused.items()}
    n3 = {c: package(fused[c], durs[c]) for c in clips}
    stages['N3'] = n3
    e0 = read_events(E0[split]) if e0_on else {}
    n4 = {}
    for c in clips:
        n4[c] = package(gpair(n3[c], e0.get(c, []), w=e0_w, max_len=e0_max_len, thr=e0_thr), durs[c]) if e0_on else list(n3[c])
    stages['N4'] = n4
    final = dict(n4)
    tail = []
    for donor in fb_donors:
        empty = {c for c in clips if not final[c]}
        if not empty:
            break
        got, missing = fb_decode(split, donor, empty, durs)
        if got is None:
            tail.append({'donor': donor, 'targets': len(empty), 'status': f'NO FRAMES for {len(missing)} clips',
                         'missing_examples': sorted(missing)[:5]})
            log(f'N6 {donor}: no frames for {len(missing)} empty clips -> step skipped for ALL targets')
            continue
        filled = [c for c in empty if got[c]]
        for c in filled:
            final[c] = got[c]
        tail.append({'donor': donor, 'targets': len(empty), 'filled': len(filled)})
    floor = sorted(c for c in clips if not final[c])
    for c in floor:
        final[c] = [(0.0, durs[c])]
    stages['N6'] = dict(final)
    info = {'tail': tail, 'A6_floor': len(floor)}
    if c4:
        scope = {c for c in clips if durs[c] >= LD1R['min_d']}
        cnt = ld1r_counts(split, scope) if scope else Counter()
        changed = []
        for c in sorted(scope):
            if cnt[c] >= LD1R['min_sets']:
                if any(b - a >= LD1R['whole'] * durs[c] for a, b in final[c]):
                    continue
                final[c] = extend_longest(final[c], durs[c])
                changed.append(c)
        info['LD1R'] = {'scope': len(scope), 'changed': len(changed)}
    return final, stages, info


def dump(final, path):
    blob = b''.join(json.dumps({'clip_id': c, 'events': [{'onset': a, 'offset': b} for a, b in final[c]]}).encode()
                    + b'\n' for c in sorted(final))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(blob)
    return sha_bytes(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('val', 'test'), required=True)
    ap.add_argument('--clips', required=True, help='file with one stem per line')
    ap.add_argument('--c4', action='store_true', help='LTL6 + LACC + LPAD8 + LD1R ON (the 950232 rules)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--durations', required=True, help='json: clip -> wav duration in seconds (wav headers)')
    ap.add_argument('--frames-root', default=None, help='<root>/<member>/<split>/frames_shift<k>.npz '
                    '(default: $T1_FRAMES or ./frames)')
    ap.add_argument('--e0', default=None, help='E0 predictions jsonl (needed unless --no-e0)')
    ap.add_argument('--no-e0', action='store_true', help='skip the E0 endpoint pairing (N4) on every clip')
    ap.add_argument('--fb-donors', default='R10,R10S1,SOUP2', help='fallback donors, in order (950232: R10,R10S1,SOUP2)')
    ap.add_argument('--e0-thr', type=float, default=0.5, help='pairing IoU threshold (950232: 0.5)')
    ap.add_argument('--e0-w', type=float, default=0.5, help='partner weight (950232: 0.5)')
    ap.add_argument('--members', default=None, help='comma list of member roles (default: the 950232 order '
                    'P2,S1,R10,R10,PSR10B,PSR10B); a role may repeat (= vote weight). Acceptance = half the sets.')
    a = ap.parse_args()
    global FTWO, NSETS, FRAMES_ROOT
    if a.frames_root:
        FRAMES_ROOT = Path(a.frames_root)
    DURS[a.split] = Path(a.durations)
    if not a.no_e0:
        if not a.e0:
            raise SystemExit('--e0 is required unless --no-e0')
        E0[a.split] = Path(a.e0)
    if a.members:
        assert not a.c4, 'the --c4 rules are defined for the 24-set 950232 stack only'
        FTWO = a.members.split(',')
        NSETS = 4 * len(FTWO)
    t0 = time.time()

    def log(msg):
        print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)
    clips = [x.strip() for x in Path(a.clips).read_text().splitlines() if x.strip()]
    final, stages, info = run(a.split, clips, a.c4, log, fb_donors=tuple(a.fb_donors.split(',')), e0_on=not a.no_e0, e0_w=a.e0_w, e0_thr=a.e0_thr)
    h = dump(final, a.out)
    for k in ('N3', 'N4'):
        dump(stages[k], str(a.out).replace('.jsonl', f'.{k}.jsonl'))
    info.update({'split': a.split, 'clips': len(clips), 'c4': a.c4, 'e0': not a.no_e0, 'e0_thr': a.e0_thr, 'e0_w': a.e0_w, 'members': FTWO, 'out': a.out, 'sha256': h,
                 'events': sum(len(v) for v in final.values()), 'seconds': round(time.time() - t0, 1)})
    Path(str(a.out).replace('.jsonl', '.info.json')).write_text(json.dumps(info, indent=1) + '\n')
    log(json.dumps(info))


if __name__ == '__main__':
    main()
