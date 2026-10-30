"""Weighted interval fusion (1-D weighted box fusion) over several prediction sets.

Used to combine the four input-shift views of the same clip. Per clip, over all
intervals of all sets (class-agnostic here, so every interval competes):

  1. sort every interval by score, high first;
  2. add each to the first open cluster whose fused interval has IoU >= `iou` with
     it, else open a new cluster;
  3. a cluster's endpoints are the score-weighted mean of its members and its score
     the maximum member score;
  4. scale that score by support / number_of_sets (`support_scale`), so an interval
     only one shift saw keeps a quarter of its score;
  5. drop fused intervals below `min_score`.

With four sets, scores of 1.0 and min_score 0.5, that keeps exactly the intervals at
least two of the four shifts agree on. Nothing here reads a label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import common


def iou_1d(a, b):
    intersection = min(a[1], b[1]) - max(a[0], b[0])
    if intersection <= 0:
        return 0.
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.


def fuse_group(items, iou_threshold, unmatched_weight, support_scale, sets):
    """items: list of (onset, offset, score, set_index). Returns fused intervals."""
    clusters = []
    for onset, offset, score, index in sorted(items, key=lambda x: (-x[2], x[0], x[1])):
        target = None
        for cluster in clusters:
            if iou_1d((onset, offset), (cluster['onset'], cluster['offset'])) >= iou_threshold:
                target = cluster
                break
        weight = max(score, 1e-6)
        if target is None:
            clusters.append({'onset': onset, 'offset': offset, 'best': score,
                             'weight_sum': weight, 'on_sum': onset * weight,
                             'off_sum': offset * weight, 'sets': {index}, 'members': 1})
            continue
        target['weight_sum'] += weight
        target['on_sum'] += onset * weight
        target['off_sum'] += offset * weight
        target['onset'] = target['on_sum'] / target['weight_sum']
        target['offset'] = target['off_sum'] / target['weight_sum']
        target['best'] = max(target['best'], score)
        target['sets'].add(index)
        target['members'] += 1
    out = []
    for cluster in clusters:
        support = len(cluster['sets'])
        scale = (min(support, sets) / sets) if support_scale else (1. if support >= 2
                                                                  else unmatched_weight)
        out.append({'onset': float(cluster['onset']), 'offset': float(cluster['offset']),
                    'score': float(cluster['best'] * scale), 'support': support,
                    'members': cluster['members']})
    return out


def fuse(prediction_sets, *, iou_threshold=.5, unmatched_weight=.5, support_scale=False,
         min_score=0., use_channel=True, default_score=1.):
    """prediction_sets: list of {clip_id: record}. Returns ({clip_id: record}, audit)."""
    clips = sorted({clip for group in prediction_sets for clip in group})
    channels_available = use_channel and all(
        all('channel' in event for record in group.values() for event in record['events'])
        for group in prediction_sets)
    fused = {}
    for clip in clips:
        duration = next(float(group[clip]['duration']) for group in prediction_sets
                        if clip in group)
        buckets = {}
        for index, group in enumerate(prediction_sets):
            for event in group.get(clip, {}).get('events', []):
                key = event.get('channel') if channels_available else None
                buckets.setdefault(key, []).append((float(event['onset']), float(event['offset']),
                                                    float(event.get('score', default_score)),
                                                    index))
        events = []
        for key, items in sorted(buckets.items(), key=lambda kv: (kv[0] is not None, kv[0])):
            for item in fuse_group(items, iou_threshold, unmatched_weight, support_scale,
                                   len(prediction_sets)):
                if item['score'] < min_score:
                    continue
                onset = max(0., min(item['onset'], duration))
                offset = max(0., min(item['offset'], duration))
                if offset <= onset:
                    continue
                item.update(onset=onset, offset=offset)
                if key is not None:
                    item['channel'] = key
                events.append(item)
        fused[clip] = {'clip_id': clip, 'duration': duration,
                       'events': sorted(events, key=lambda e: (e['onset'], e['offset']))}
    audit = {'clips': len(clips), 'sets': len(prediction_sets),
             'per_channel': bool(channels_available), 'iou_threshold': iou_threshold,
             'unmatched_weight': unmatched_weight, 'support_scale': bool(support_scale),
             'min_score': min_score, 'events': sum(len(r['events']) for r in fused.values())}
    return fused, audit


def load_sets(paths):
    return [{record['clip_id']: record for record in common.read_jsonl(path)} for path in paths]


def fuse_files(paths, out, *, iou_threshold=.6, support_scale=True, min_score=.5,
               class_agnostic=True):
    """The four-shift fusion: the four unshifted refined sets, shift 0 first."""
    fused, audit = fuse(load_sets(paths), iou_threshold=iou_threshold,
                        support_scale=support_scale, min_score=min_score,
                        use_channel=not class_agnostic)
    common.write_jsonl(out, [fused[c] for c in sorted(fused)])
    return fused, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs='+', required=True,
                        help='refined prediction sets; the FIRST one supplies each clip duration')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--iou', type=float, default=common.FUSE_PARAMS['iou'])
    parser.add_argument('--min-score', type=float, default=common.FUSE_PARAMS['min_score'])
    parser.add_argument('--no-support-scale', action='store_true')
    parser.add_argument('--per-channel', action='store_true')
    args = parser.parse_args(argv)
    _, audit = fuse_files(args.input, args.out, iou_threshold=args.iou,
                          support_scale=not args.no_support_scale, min_score=args.min_score,
                          class_agnostic=not args.per_channel)
    print(f'wrote {args.out}: {audit["clips"]} clips, {audit["events"]} events '
          f'(iou {audit["iou_threshold"]}, min-score {audit["min_score"]}, '
          f'support-scale {audit["support_scale"]})', flush=True)


if __name__ == '__main__':
    main()
