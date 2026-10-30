"""Decode rules on top of a base decode, shared by the validation cross-fit (w07) and the test
application (w09).  No label is read here.

Generator constraints (measured on the validation labels, used label-free):
  * no overlap, gap >= 0.200 s between events, at most 5 events per clip;
  * non-master NOISE pastes start >= 0.250 s and end <= clip end - 0.251 s (R6);
  * a master file (machine 5.437 s, child 5.211 s, bird >= 9.077 s) is pasted from its first
    sample and cut at the clip end: event = [onset, min(onset + file length, clip end)] (job2).
Rules:
  ADD      a scan candidate of template k with tile-GLRT z S3 >= theta_k (null-derived), a
           plausible paste level (inside the validated level range of k +- W dB) and a place the
           generator allows is added, strongest S3 first; existing events are never moved.
  REMOVE   a base event whose S3 at its own lag is below theta_r is removed (the exact copy of
           the template is not there).
  RECHOOSE a base event is replaced by an overlapping candidate of another template when that
           candidate explains the audio much better (S3 larger by a factor) and the new extent
           is allowed.
"""
from __future__ import annotations

import numpy as np

import wlib
from w02_scan import FIELDS

SR = wlib.SR
F = {f: i for i, f in enumerate(FIELDS)}
EX = wlib.exemplars()
K = len(EX)
DUR = np.array([t['dur'] for t in EX])
VOCAL = np.array([t['tag'].startswith('[') for t in EX])
MASTER = {'machine': list(range(103, 126)), 'child': list(range(84, 89)), 'bird': [64, 65, 66]}
MASTER_OF = {k: m for m, ks in MASTER.items() for k in ks}
MASTER_LEN = {'machine': 5.437, 'child': 5.211, 'bird': 9.077}
GAP = 0.200
SLACK = 0.005
EDGE_ON = 0.250
EDGE_OFF = 0.251
MAX_EVENTS = 5


def kernel_len(k, clip, loo):
    i = wlib.kernel_index(k, clip) if loo else 0
    return None if i is None else len(EX[k]['exs'][i][1])


def extent(k, on, clip_dur):
    m = MASTER_OF.get(k)
    if m is not None:
        return on, min(on + MASTER_LEN[m], clip_dur)
    return on, min(on + DUR[k], clip_dur)


def allowed(on, off, k, events, clip_dur, edge=True):
    if len(events) >= MAX_EVENTS:
        return False
    for a, b in events:
        if not (off + GAP - SLACK <= a or on >= b + GAP - SLACK):
            return False
    if edge and (not VOCAL[k]) and MASTER_OF.get(k) is None:
        if on < EDGE_ON - SLACK or off > clip_dur - EDGE_OFF + SLACK:
            return False
    return True


def candidates(rows, clip, loo, min_s3=0.0):
    """scan rows -> list of dicts (deduplicated per template within 8 samples, max S3 kept)."""
    out = []
    if rows is None or not len(rows):
        return out
    R = rows[rows[:, F['S3']] >= min_s3]
    R = R[np.argsort(-R[:, F['S3']])]
    seen = []
    for r in R:
        k = int(r[F['k']])
        p = int(r[F['p']])
        if any(kk == k and abs(pp - p) <= 8 for kk, pp in seen):
            continue
        seen.append((k, p))
        L = kernel_len(k, clip, loo)
        g = float(r[F['g']])
        lvl = 10 * np.log10(max(g * g / L, 1e-20)) if (L and g > 0) else -200.0
        out.append({'k': k, 'p': p, 'on': p / SR, 'S0': float(r[F['S0']]), 'S1': float(r[F['S1']]),
                    'S3': float(r[F['S3']]), 'S5': float(r[F['S5']]), 'g': g, 'level': lvl,
                    'S3b': float(r[F['S3b']])})
    return out


def level_ranges(events, exclude_fold=None):
    """per template (min, max) paste level (dBFS) of the labelled validation events."""
    d = {}
    for e in events:
        if e['k'] is None or (exclude_fold is not None and e['fold'] == exclude_fold):
            continue
        lo, hi = d.get(e['k'], (np.inf, -np.inf))
        d[e['k']] = (min(lo, e['level']), max(hi, e['level']))
    return d


def level_ok(k, lvl, ranges, W):
    if W is None:
        return True
    r = ranges.get(k)
    if r is None:
        return True        # no validated level for this template: no constraint
    return r[0] - W <= lvl <= r[1] + W


def apply_add(events, cands, clip_dur, theta, ranges, W, masters=True):
    """events: [(on, off, k, tag)] current; returns (new events, added list)."""
    ev = list(events)
    added = []
    for c in cands:
        k = c['k']
        if c['S3'] < theta[k]:
            continue
        if MASTER_OF.get(k) is not None and not masters:
            continue
        if not level_ok(k, c['level'], ranges, W):
            continue
        on, off = extent(k, c['on'], clip_dur)
        if off - on < 0.05:
            continue
        if allowed(on, off, k, [(a, b) for a, b, *_ in ev], clip_dur):
            ev.append((on, off, k, 'add'))
            added.append(dict(c, off=off))
    return sorted(ev), added


def apply_remove(events, s3_of_event, theta_r):
    """s3_of_event: list aligned with events (None = unknown / keep)."""
    keep, removed = [], []
    for e, s in zip(events, s3_of_event):
        if s is not None and np.isfinite(s) and s < theta_r and MASTER_OF.get(e[2]) is None:
            removed.append((e, s))
        else:
            keep.append(e)
    return keep, removed


def apply_rechoose(events, s3_of_event, cands, clip_dur, factor, ranges, W):
    """replace a base event by an overlapping candidate of another (non-master) template whose
    S3 >= factor * S3(event) when the new extent is allowed next to the other events."""
    ev = list(events)
    changed = []
    for i, (e, s) in enumerate(zip(events, s3_of_event)):
        if s is None or not np.isfinite(s) or MASTER_OF.get(e[2]) is not None:
            continue
        best = None
        for c in cands:
            k = c['k']
            if k == e[2] or MASTER_OF.get(k) is not None:
                continue
            on, off = extent(k, c['on'], clip_dur)
            if not (on < e[1] and off > e[0]):
                continue
            if c['S3'] < factor * max(s, 1e-6):
                continue
            if not level_ok(k, c['level'], ranges, W):
                continue
            if best is None or c['S3'] > best[0]['S3']:
                best = (c, on, off)
        if best is None:
            continue
        c, on, off = best
        others = [(a, b) for j, (a, b, *_) in enumerate(ev) if j != i]
        if allowed(on, off, c['k'], others, clip_dur):
            changed.append((ev[i], (on, off, c['k'], 'rechoose'), s, c['S3']))
            ev[i] = (on, off, c['k'], 'rechoose')
    return sorted(ev), changed
