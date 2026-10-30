"""Organizer-published Track 1 metric semantics, verified 2026-09-07.

Source: https://www.codabench.org/competitions/17825/ Evaluation tab.
This is a local reproduction of the published snippets, not the hidden scorer.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def event_map(records):
    result = {}
    for row in records:
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id or clip_id in result:
            raise ValueError(f"Missing, invalid or duplicate clip_id: {clip_id!r}")
        if not isinstance(row.get("events"), list):
            raise ValueError(f"{clip_id}: events must be a list; unknown labels cannot be scored")
        events = []
        for event in row["events"]:
            on, off = event["onset"], event["offset"]
            if (isinstance(on, bool) or isinstance(off, bool)
                    or not isinstance(on, (int, float)) or not isinstance(off, (int, float))
                    or not math.isfinite(on) or not math.isfinite(off)
                    or not 0 <= on < off):
                raise ValueError(f"{clip_id}: invalid event {event}")
            events.append((on, off))
        result[clip_id] = events
    return result


def match_events(ref_events, pred_events):
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tolerance = max(0.20 * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tolerance and abs(p_off - r_off) <= tolerance:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    matched_ref, matched_pred = set(), set()
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp


def clip_dice(ref_events, pred_events):
    combined = ref_events + pred_events
    if not combined:
        return 1.0
    n_frames = int((max(off for _, off in combined) + 0.5) / 0.01) + 1
    masks = []
    for events in (ref_events, pred_events):
        mask = np.zeros(n_frames, dtype=bool)
        for on, off in events:
            mask[int(on / 0.01):min(int(off / 0.01) + 1, n_frames)] = True
        masks.append(mask)
    total = int(masks[0].sum() + masks[1].sum())
    return 1.0 if total == 0 else 2.0 * int((masks[0] & masks[1]).sum()) / total


def score_records(references, predictions):
    ref, pred = event_map(references), event_map(predictions)
    tp = fp = fn = 0
    dice = []
    for clip_id, events in ref.items():
        counts = match_events(events, pred.get(clip_id, []))
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
        dice.append(clip_dice(events, pred.get(clip_id, [])))
    fp += sum(len(events) for clip_id, events in pred.items() if clip_id not in ref)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    segment_dice = sum(dice) / len(dice) if dice else 0.0
    return {"event_f1": f1, "segment_dice": segment_dice,
            "combined": f1 + segment_dice, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "reference_clips": len(ref)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scores = score_records(read(args.reference), read(args.predictions))
    report = json.dumps(scores, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
