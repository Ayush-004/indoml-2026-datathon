"""The organizer metric, plus a CLI for scoring a predictions file on the validation set.

The three functions below (`match_events`, `events_to_frames`, `clip_dice`) and the
combination in `score` are an exact copy of the organizer reference code
(reference_event_f1.py, reference_segment_dice.py); only the counters and the
`clips` / `weights` arguments are added, and they change nothing when unused.

Score = event F1 (tolerance max(20% of the reference duration, 50 ms), greedy
closest-first matching, micro-averaged over all events) + segment Dice (10 ms
frames, macro-averaged over clips).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ------------------------------------------------------------------ organizer scorer

def match_events(ref_events, pred_events, tolerance_frac=0.20):
    matched_ref, matched_pred = set(), set()
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp


def events_to_frames(events, max_time, frame_len=0.01):
    n_frames = int(max_time / frame_len) + 1
    mask = [0] * n_frames
    for on, off in events:
        for i in range(int(on / frame_len), min(int(off / frame_len) + 1, n_frames)):
            mask[i] = 1
    return mask


def clip_dice(ref_events, pred_events):
    all_events = ref_events + pred_events
    if not all_events:
        return 1.0
    max_time = max(off for _, off in all_events) + 0.5
    rm = events_to_frames(ref_events, max_time)
    pm = events_to_frames(pred_events, max_time)
    inter = sum(r & p for r, p in zip(rm, pm))
    total = sum(rm) + sum(pm)
    return 1.0 if total == 0 else 2.0 * inter / total


def score(ref, pred, clips=None, weights=None):
    """ref/pred: clip_id -> [(on, off)]. weights: clip_id -> w (to reweight subsets)."""
    clips = clips if clips is not None else list(ref)
    TP = FP = FN = 0.0
    dsum = 0.0
    wsum = 0.0
    for c in clips:
        w = 1.0 if weights is None else weights[c]
        tp, fp, fn = match_events(ref[c], pred.get(c, []))
        TP += w * tp
        FP += w * fp
        FN += w * fn
        dsum += w * clip_dice(ref[c], pred.get(c, []))
        wsum += w
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    dice = dsum / wsum if wsum else 0.0
    return dict(f1=f1, dice=dice, combined=f1 + dice, tp=TP, fp=FP, fn=FN, precision=p, recall=r)


# ------------------------------------------------------------------ validation report

def _round(result, digits=4):
    return {k: round(v, digits) for k, v in result.items()
            if k in ('f1', 'dice', 'combined', 'tp', 'fp', 'fn')}


def validation_report(predictions, digits=4):
    """All / natural / synthetic / test-mix-reweighted scores over the 2,501 validation clips."""
    from . import common
    records = common.validation_records()
    reference = common.validation_reference()
    synthetic = {r['segmentFileName']: bool(r['syntheticData']) for r in records}
    natural_clips = [c for c in reference if not synthetic[c]]
    synthetic_clips = [c for c in reference if synthetic[c]]
    weights = {c: (common.TEST_NATURAL / len(natural_clips) if not synthetic[c]
                   else common.TEST_SYNTHETIC / len(synthetic_clips)) for c in reference}
    missing = [c for c in reference if c not in predictions]
    return {
        'clips': len(reference),
        'clips_missing_from_predictions': len(missing),
        'predicted_events': sum(len(v) for v in predictions.values()),
        'all_2501': _round(score(reference, predictions), digits),
        'natural_1806': _round(score(reference, predictions, natural_clips), digits),
        'synthetic_695': _round(score(reference, predictions, synthetic_clips), digits),
        'test_mix_reweighted': _round(score(reference, predictions, weights=weights), digits),
    }


def main(argv=None):
    from . import common
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions', type=Path,
                        help='jsonl with clip_id / events[{onset, offset}]')
    parser.add_argument('--json-out', type=Path, default=None)
    parser.add_argument('--digits', type=int, default=4)
    args = parser.parse_args(argv)
    report = validation_report(common.events_of(args.predictions), args.digits)
    report['predictions'] = str(args.predictions)
    print(json.dumps(report, indent=1))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1) + '\n')
    return report


if __name__ == '__main__':
    main()
