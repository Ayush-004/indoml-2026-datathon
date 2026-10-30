"""t1score.py - organizer scorer logic (copied from the organizer reference_event_f1.py / reference_segment_dice.py,
unchanged math) with per-clip statistics, so subsets (natural / synthetic) and pooled mixes can be computed.
Read-only. No label is used for anything but scoring.

  T1_VALIDATION_DIR=<extracted validation folder with validationMetadata.json> python t1score.py <predictions.jsonl> ...
"""
from __future__ import annotations
import json, os
from pathlib import Path

VAL = Path(os.environ.get('T1_VALIDATION_DIR', 'validation'))


def stem(c):
    return c[:-4] if c.endswith('.wav') else c


def load_refs():
    meta = json.load(open(VAL / 'validationMetadata.json'))
    refs, kind, dur = {}, {}, {}
    for r in meta:
        s = stem(r['segmentFileName'])
        refs[s] = [(float(e['start']), float(e['end'])) for e in r['NoiseSubCategoryTimeStamp']]
        kind[s] = 'syn' if r['syntheticData'] else 'nat'
        dur[s] = float(r['duration'])
    return refs, kind, dur


# ---- official math (reference_event_f1.match_events, reference_segment_dice) ----
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
            matched_ref.add(ri); matched_pred.add(pi)
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


def per_clip(refs, preds):
    """clip -> (tp, fp, fn, dice) for every reference clip."""
    out = {}
    for c, r in refs.items():
        p = preds.get(c, [])
        tp, fp, fn = match_events(r, p)
        out[c] = (tp, fp, fn, clip_dice(r, p))
    return out


def summarize(stats, clips):
    TP = sum(stats[c][0] for c in clips); FP = sum(stats[c][1] for c in clips); FN = sum(stats[c][2] for c in clips)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * P * R / (P + R) if P + R else 0.0
    dice = sum(stats[c][3] for c in clips) / len(clips)
    return {'clips': len(clips), 'tp': TP, 'fp': FP, 'fn': FN, 'precision': round(P, 5), 'recall': round(R, 5),
            'f1': round(f1, 5), 'dice': round(dice, 5), 'combined': round(f1 + dice, 5)}


def read_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[stem(r['clip_id'])] = [(float(e['onset']), float(e['offset'])) for e in r['events']]
    return out


def report(preds, refs=None, kind=None):
    if refs is None:
        refs, kind, _ = load_refs()
    st = per_clip(refs, preds)
    nat = [c for c in refs if kind[c] == 'nat']; syn = [c for c in refs if kind[c] == 'syn']
    return st, {'all': summarize(st, list(refs)), 'natural': summarize(st, nat), 'synthetic': summarize(st, syn)}


if __name__ == '__main__':
    import sys
    refs, kind, _ = load_refs()
    for p in sys.argv[1:]:
        st, rep = report(read_preds(p), refs, kind)
        print(p)
        for k, v in rep.items():
            print('  ', k, v)
