"""P2 decoder replay and the label-free endpoint/pruning refiner.

`candidate_intervals` and `decode_candidates` are transcribed verbatim from the
inference source of P2 (atst_red/indoml/intervals.py), so this module turns the
per-frame P2 outputs back into exactly the events P2 itself emits.

`run_pipeline` is the label-free refiner. Its default parameters (common.REFINER_PARAMS)
come from a cross-fitted study over the natural validation clips:

  prune    drop every decoded event with min(score, peak) < 0.70
  cross    class-agnostic NMS at IoU 0.6, keeping the higher-confidence event
  cascade  if a clip is left with nothing, re-decode it at threshold 0.55

Every operator reads only P2's own per-frame outputs. No label is read here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import common
from .common import CATEGORIES, FRAME_HOP, FRAME_OFFSET, P2_DECODER, REFINER_PARAMS, SR


# ------------------------------------------------------------------ exact P2 decoder

def candidate_intervals(row, presence, distances, max_candidates=100):
    """One continuous interval per candidate time, ranked by event confidence."""
    result = []
    centers = np.arange(len(presence)) * row['frame_hop'] + row['frame_offset']
    for category in range(len(CATEGORIES)):
        values = presence[:, category]
        prefix = np.concatenate([[0.], values.cumsum(dtype=np.float64)])
        padded = np.pad(values, (2, 2), constant_values=-np.inf)
        peaks = np.logical_and.reduce([values >= padded[shift:shift + len(values)]
                                       for shift in range(5)])
        indices = np.flatnonzero(peaks)
        if len(indices):
            groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            indices = np.asarray([group[len(group) // 2] for group in groups])
        indices = indices[np.argsort(-values[indices], kind='stable')[:max_candidates]]
        candidates = []
        for index in indices:
            onset = max(0., centers[index] - float(distances[index, category, 0]))
            offset = min(row['duration'], centers[index] + float(distances[index, category, 1]))
            if offset <= onset:
                continue
            left = max(0, int(np.searchsorted(centers, onset, side='left')))
            right = min(len(values), int(np.searchsorted(centers, offset, side='left')))
            confidence = (float((prefix[right] - prefix[left]) / (right - left))
                          if right > left else float(values[index]))
            candidates.append({'onset': onset, 'offset': offset, 'score': confidence,
                               'peak': float(values[index]), 'anchor': int(index),
                               'channel': category})
        result.append(sorted(candidates, key=lambda x: (-x['score'], -x['peak'], x['onset'])))
    return result


def decode_candidates(row, categories, decoder=None, keep_meta=False):
    decoder = decoder or P2_DECODER
    events = []
    for candidates in categories:
        candidates = [c for c in candidates
                      if min(c['score'], c['peak']) >= decoder['threshold']
                      and c['offset'] - c['onset'] >= decoder['min_duration']]
        boxes = np.asarray([[c['onset'], c['offset']] for c in candidates], dtype=np.float64)
        order = list(range(len(candidates)))
        selected = []
        while len(boxes) and len(selected) < 50:
            first = boxes[0]
            event = {'onset': float(first[0]), 'offset': float(first[1])}
            if keep_meta:
                source = candidates[order[0]]
                event.update({'score': source['score'], 'peak': source['peak'],
                              'anchor': source['anchor'], 'channel': source['channel']})
            selected.append(event)
            others = boxes[1:]
            rest = order[1:]
            intersection = np.maximum(0., np.minimum(first[1], others[:, 1])
                                      - np.maximum(first[0], others[:, 0]))
            union = np.maximum(first[1], others[:, 1]) - np.minimum(first[0], others[:, 0])
            keep = (intersection <= 0) | (intersection / np.maximum(union, 1e-9) <= decoder['nms_iou'])
            boxes = others[keep]
            order = [rest[i] for i in np.flatnonzero(keep)]
        events.extend(selected)
    return {'clip_id': row['clip_id'], 'duration': row['duration'],
            'events': sorted(events, key=lambda event: (event['onset'], event['offset']))}


# ------------------------------------------------------------------ refiner operators

def isolated_flags(events):
    """True where the event overlaps no other event in the same clip."""
    count = len(events)
    flags = [True] * count
    for i in range(count):
        for j in range(i + 1, count):
            if min(events[i]['offset'], events[j]['offset']) > max(events[i]['onset'],
                                                                   events[j]['onset']):
                flags[i] = flags[j] = False
    return flags


def cross_channel_resolve(events, iou_threshold, mode):
    """Class-agnostic suppression or merge over decoded events carrying score/peak."""
    if mode == 'off' or len(events) <= 1:
        return [(e['onset'], e['offset']) for e in events]
    order = sorted(range(len(events)),
                   key=lambda i: (-min(events[i]['score'], events[i]['peak']), events[i]['onset']))
    alive = [True] * len(events)
    out = []
    for i in order:
        if not alive[i]:
            continue
        alive[i] = False
        a, b = events[i]['onset'], events[i]['offset']
        weight = min(events[i]['score'], events[i]['peak'])
        acc_on, acc_off, acc_w = a * weight, b * weight, weight
        for j in order:
            if not alive[j]:
                continue
            intersection = max(0., min(b, events[j]['offset']) - max(a, events[j]['onset']))
            union = max(b, events[j]['offset']) - min(a, events[j]['onset'])
            if union > 0 and intersection / union >= iou_threshold:
                alive[j] = False
                if mode == 'merge':
                    w = min(events[j]['score'], events[j]['peak'])
                    acc_on += events[j]['onset'] * w
                    acc_off += events[j]['offset'] * w
                    acc_w += w
        if mode == 'merge' and acc_w > 0:
            out.append((acc_on / acc_w, acc_off / acc_w))
        else:
            out.append((a, b))
    return sorted(out)


def presence_run_events(entry, threshold, gap_frames, min_duration):
    """Contiguous presence runs, used only by the optional `union` stage."""
    presence, duration = entry['presence'], entry['duration']
    events = []
    for channel in range(presence.shape[1]):
        mask = presence[:, channel] >= threshold
        if not mask.any():
            continue
        padded = np.concatenate([[False], mask, [False]])
        change = np.diff(padded.astype(np.int8))
        spans = np.stack([np.flatnonzero(change == 1), np.flatnonzero(change == -1)], axis=1)
        if gap_frames > 0 and len(spans) > 1:
            merged = [spans[0].copy()]
            for span in spans[1:]:
                if span[0] - merged[-1][1] <= gap_frames:
                    merged[-1][1] = span[1]
                else:
                    merged.append(span.copy())
            spans = np.asarray(merged)
        for a, b in zip(np.clip(spans[:, 0] * FRAME_HOP, 0., duration),
                        np.clip(spans[:, 1] * FRAME_HOP, 0., duration)):
            if b > a and b - a >= min_duration:
                events.append((float(a), float(b)))
    return sorted(events)


def cascade_sets(entry, thresholds):
    """Re-decode the same candidate set at lower confidence thresholds."""
    result = {}
    for threshold in thresholds:
        groups = [[] for _ in range(len(CATEGORIES))]
        for candidate in entry['candidates']:
            groups[candidate['channel']].append(candidate)
        for group in groups:
            group.sort(key=lambda x: (-x['score'], -x['peak'], x['onset']))
        decoded = decode_candidates({'clip_id': '', 'duration': entry['duration']}, groups,
                                    {'threshold': threshold, 'nms_iou': 0.0, 'min_duration': 0.08})
        result[threshold] = [(e['onset'], e['offset']) for e in decoded['events']]
    return result


def run_pipeline(entry, params, cascade=None):
    """P2's own detection set with only pruning, cross-channel NMS and a cascade applied."""
    events = [dict(e) for e in entry['p2_events']]
    if params.get('boundary') is not None:
        raise ValueError('the boundary stage is not part of this pipeline')
    if params.get('prune') is not None:
        score_gate, isolated_only, max_duration = params['prune']
        flags = isolated_flags(events)
        events = [e for e, isolated in zip(events, flags)
                  if not (min(e['score'], e['peak']) < score_gate
                          and (e['offset'] - e['onset']) <= max_duration
                          and (isolated or not isolated_only))]
    if params.get('cross') is not None:
        iou, mode = params['cross']
        pairs = cross_channel_resolve(events, iou, mode)
    else:
        pairs = [(e['onset'], e['offset']) for e in events]
    if not pairs and params.get('cascade') is not None and cascade is not None:
        pairs = list(cascade[params['cascade']])
    if params.get('union') is not None:
        threshold, min_duration = params['union']
        for a, b in presence_run_events(entry, threshold, 2, min_duration):
            if not any(min(b, d) > max(a, c) for c, d in pairs):
                pairs.append((a, b))
    duration = entry['duration']
    out = []
    for a, b in pairs:
        a = float(max(0.0, min(a, duration)))
        b = float(max(0.0, min(b, duration)))
        if b > a:
            out.append((a, b))
    return sorted(out)


# ------------------------------------------------------------------ frames -> events

def entries_from_frames(path, clips=None):
    """Read a frames npz (clip_ids, durations, offsets, presence, distances)."""
    blob = np.load(path, allow_pickle=True)
    clip_ids = [str(c) for c in blob['clip_ids']]
    durations = np.asarray(blob['durations'], dtype=np.float64)
    bounds = np.asarray(blob['offsets'], dtype=np.int64)
    presence = np.asarray(blob['presence'], dtype=np.float32)
    distances = np.asarray(blob['distances'], dtype=np.float32)
    out = {}
    for k, clip in enumerate(clip_ids):
        if clips is not None and clip not in clips:
            continue
        a, b = int(bounds[k]), int(bounds[k + 1])
        row = {'clip_id': clip, 'duration': float(durations[k]),
               'frame_hop': FRAME_HOP, 'frame_offset': FRAME_OFFSET}
        categories = candidate_intervals(row, presence[a:b], distances[a:b])
        decoded = decode_candidates(row, categories, P2_DECODER, keep_meta=True)
        out[clip] = {'duration': float(durations[k]), 'presence': presence[a:b],
                     'distances': distances[a:b],
                     'candidates': [dict(c) for group in categories for c in group],
                     'p2_events': decoded['events']}
    return out


def p2_events_from_frames(path, clips=None):
    """The unrefined P2 events, for verifying an inference run against a reference file."""
    return {clip: sorted((e['onset'], e['offset']) for e in entry['p2_events'])
            for clip, entry in entries_from_frames(path, clips).items()}


def refine_frames(path, params=None, clips=None, shift=0):
    """frames npz -> [{clip_id, duration, events}], with the input shift removed."""
    params = params or REFINER_PARAMS
    thresholds = tuple(t for t in (0.45, 0.50, 0.55, 0.60) if params.get('cascade') == t) \
        or (0.55, 0.60)
    lag = shift / SR
    records = []
    for clip, entry in sorted(entries_from_frames(path, clips).items()):
        pairs = run_pipeline(entry, params, cascade_sets(entry, thresholds))
        duration = entry['duration'] - lag
        events = []
        for a, b in pairs:
            a = max(0.0, a - lag)
            b = min(duration, b - lag)
            if b > a:
                events.append({'onset': a, 'offset': b})
        records.append({'clip_id': clip, 'duration': duration, 'events': events})
    return records


# ------------------------------------------------------------------ CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--params', type=Path, default=None,
                        help='refiner parameters json (default: the built-in best-pipeline)')
    parser.add_argument('--shift', type=int, default=0,
                        help='input shift in SAMPLES that was prepended before inference')
    parser.add_argument('--clips', type=Path, default=None)
    parser.add_argument('--verify', type=Path, default=None,
                        help='reference P2 jsonl the re-derived (unrefined) events must reproduce')
    args = parser.parse_args(argv)

    clips = None
    if args.clips:
        clips = {line.strip() for line in args.clips.read_text().splitlines() if line.strip()}
    params = json.loads(args.params.read_text()) if args.params else REFINER_PARAMS

    if args.verify:
        derived = p2_events_from_frames(args.frames, clips)
        expected = common.events_of(args.verify)
        worst, mismatch = 0.0, 0
        for clip, got in derived.items():
            if clip not in expected:
                continue
            want = sorted(expected[clip])
            if len(got) != len(want):
                mismatch += 1
                continue
            for (a, b), (c, d) in zip(got, want):
                worst = max(worst, abs(a - c), abs(b - d))
        print(f'verify: {mismatch} event-count mismatches, max endpoint error {worst:.3e} s',
              flush=True)

    records = refine_frames(args.frames, params, clips, args.shift)
    common.write_jsonl(args.out, records)
    print(f'wrote {args.out}: {len(records)} clips, '
          f'{sum(len(r["events"]) for r in records)} events', flush=True)


if __name__ == '__main__':
    main()
