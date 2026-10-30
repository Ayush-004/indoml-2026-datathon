"""Data layer for the SILVER-STRONG arms (S1, S2).

Built on `c2_local.py` (imported, never modified): the row shapes, the
caps, the evaluation populations, the organizer scorer and the duration diagnostics are
the same ones the C2/CTRL arms used, so the new arms are comparable checkpoint for
checkpoint with the recorded CTRL trajectory.

What is new here:

  * a STRONG Silver pool - the 61,640 `unverified_timestamps` clips of
    `vaani-full/manifest.jsonl`, whose events carry onset/offset/category. They have never
    been used as strong supervision at scale; the organizer baseline uses them at 0.4.
  * leak hygiene - every Silver/Bronze row whose SESSION token (the 6th underscore field
    of the clip id) also appears in one of the 1,806 organizer natural VALIDATION clip ids
    is dropped from every pool, and the count is reported.
  * the long-event coverage diagnostic - for organizer references of 2 s or more whose tag
    does not start with '[', the coverage of the reference by the UNION of the predictions.
  * prediction at several decoder thresholds from ONE forward pass.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from collections import Counter
import json
from pathlib import Path

C2MAC = Path(__file__).resolve().parent
if str(C2MAC) not in sys.path:
    sys.path.insert(0, str(C2MAC))

import c2_local as L  # noqa: E402

VAANI = L.VAANI
KAGGLE_INPUT = L.KAGGLE_INPUT
REBUILD = L.REBUILD
COMMON = L.COMMON
CATEGORIES = L.CATEGORIES
DECODERS = {'t065': {'threshold': .65, 'nms_iou': 0., 'min_duration': .08},
            't075': {'threshold': .75, 'nms_iou': 0., 'min_duration': .08}}

# Per-update clip counts. S1 keeps CTRL's budget: 8 strong clips and 16 weak clips.
CAPS = {'gold': 20, 'silver': 20, 'weak': 20}
MIX = {'S1': {'gold': 4, 'silver': 4, 'weak': 16},
       'S2': {'gold': 4, 'silver': 4, 'weak': 16}}
SILVER_WEIGHT = {'S1': 0.4, 'S2': 0.4}
# S2: the Silver timestamps are a different distribution from Gold and from the organizer
# references (median event 0.98 s against 0.45 s; 31 % of Silver onsets sit on a 100 ms
# grid, an error the size of the 50-65 ms tolerance), so Silver teaches WHERE and WHICH
# CLASS only - boundary and IoU weights 0 on the Silver forward. Gold keeps P2's objective.
SILVER_LOSS = {'S1': {'boundary_weight': 100.0, 'iou_weight': 1.0},
               'S2': {'boundary_weight': 0.0, 'iou_weight': 0.0}}
SEEDS = {'gold': 149, 'silver': 649, 'weak': 51}
HELD_OUT_FOLD = L.HELD_OUT_FOLD

# The CTRL arm of the C2 trainer, scored by this same code on the same population.
CTRL_REFERENCE = {0: 1.0758464842826536, 1000: 1.049658, 2000: 1.057873,
                  3000: 1.061895, 4000: 1.063276}


def session_token(clip_id):
    """IISc_VaaniProject_<L>_<State>_<District>_<SESSION>_...  -> the 6th field."""
    parts = clip_id.split('_')
    return parts[5] if len(parts) > 5 else clip_id


def build_pools(*, arm='S1', need_training=True, gold_scope='train', gold_leak_filter=False):
    """`gold_scope` 'train' is the Gold-train split the C2/CTRL/S1/S2 arms used; 'all' is
    every Gold row, which is what parent P2 itself was trained on (Gold-val was inside its
    training set, so it is not a clean metric). `gold_leak_filter` also drops Gold rows that
    share a session token with an organizer natural validation clip, as the Kaggle kernels do.
    """
    census = {'dropped': {}}

    def drop(key, count=1):
        census['dropped'][key] = census['dropped'].get(key, 0) + count

    natural_ids = sorted(json.loads((REBUILD / 'folds.json').read_text())['clip_fold'])
    validation_sessions = {session_token(clip) for clip in natural_ids}
    census['validation_session_tokens'] = len(validation_sessions)

    gold_rows, silver_rows, weak_rows = [], [], []
    if need_training:
        local = {row['clip_id']: row for row in L.read_jsonl(VAANI / 'manifest.jsonl')}
        gold_train = set(json.loads((REBUILD / 'gold-split.json').read_text())['train'])
        leaked_silver = leaked_bronze = leaked_gold = 0
        silver_usable_ids = set()

        # --- Gold-train: exactly the pool the C2/CTRL arms used (from training.jsonl).
        for row in L.read_jsonl(KAGGLE_INPUT / 'training.jsonl'):
            if row['clip_id'] not in local:
                drop('no_local_audio_' + row['supervision'])
                continue
            if row['supervision'] != 'strong_gold':
                continue
            if gold_scope == 'train' and row['clip_id'] not in gold_train:
                drop('gold_not_in_gold_train')
                continue
            if gold_leak_filter and session_token(row['clip_id']) in validation_sessions:
                leaked_gold += 1
                continue
            if row['native_windows'] > CAPS['gold']:
                drop('gold_too_many_windows')
                continue
            events = [{'onset': e['onset'], 'offset': e['offset'], 'category': e['category']}
                      for e in (row['events'] or [])]
            if not events or any(e['category'] not in CATEGORIES for e in events):
                drop('gold_unusable_events')
                continue
            if any(e['offset'] <= e['onset'] for e in events):
                drop('gold_degenerate_events')
                continue
            gold_rows.append(L.training_row(row['clip_id'], VAANI / local[row['clip_id']]['audio_path'],
                                            row['duration'], events, row['positive_categories'],
                                            windows=row['native_windows']))

        # --- Silver: strong supervision from the unverified timestamps themselves.
        for clip_id, record in local.items():
            if record['quality'] != 'unverified_timestamps':
                continue
            if session_token(clip_id) in validation_sessions:
                leaked_silver += 1
                continue
            events = [{'onset': float(e['onset']), 'offset': float(e['offset']),
                       'category': e['category']}
                      for e in (record.get('events') or [])
                      if e['category'] in CATEGORIES and float(e['offset']) > float(e['onset'])
                      and float(e['offset']) <= record['duration'] + 1e-6]
            if not events:
                drop('silver_no_usable_event')
                continue
            row = L.training_row(clip_id, VAANI / record['audio_path'], record['duration'], events,
                                 {e['category'] for e in events}, quality='unverified_timestamps')
            if row['native_windows'] > CAPS['silver']:
                drop('silver_too_many_windows')
                continue
            silver_rows.append(row)
            silver_usable_ids.add(clip_id)

        # --- Weak: Bronze, plus any Silver row the strong stream does not take.
        for row in L.read_jsonl(KAGGLE_INPUT / 'training.jsonl'):
            if row['supervision'] != 'weak_clip' or row['clip_id'] not in local:
                continue
            if row['clip_id'] in silver_usable_ids:
                continue
            if session_token(row['clip_id']) in validation_sessions:
                leaked_bronze += 1
                continue
            categories = [c for c in (row['positive_categories'] or []) if c in CATEGORIES]
            if not categories:
                drop('weak_no_category')
                continue
            if row['native_windows'] > CAPS['weak']:
                drop('weak_too_many_windows')
                continue
            weak_rows.append(L.training_row(row['clip_id'], VAANI / local[row['clip_id']]['audio_path'],
                                            row['duration'], None, categories,
                                            supervision='weak_clip', quality=row['quality'],
                                            windows=row['native_windows']))
        census['leak_dropped_gold'] = leaked_gold
        census['gold_scope'] = gold_scope
        census['leak_dropped_silver'] = leaked_silver
        census['leak_dropped_weak_bronze_or_unused_silver'] = leaked_bronze
        census['leak_dropped_total'] = leaked_silver + leaked_bronze + leaked_gold
        census['weak_quality'] = dict(Counter(r['quality'] for r in weak_rows))
        census['silver_events'] = sum(len(r['events']) for r in silver_rows)
        census['gold_events'] = sum(len(r['events']) for r in gold_rows)

    # ------------------------------------------------------------- evaluation sets
    duration_of = {row['clip_id']: row['duration']
                   for row in L.read_jsonl(KAGGLE_INPUT / 'organizer-inference.jsonl')}
    meta = json.loads((COMMON / 'val_meta.json').read_text())
    reference, tagged = {}, {}
    for record in meta:
        if record.get('syntheticData'):
            continue
        clip = record['segmentFileName']
        reference[clip] = [(float(e['start']), float(e['end']))
                           for e in record['NoiseSubCategoryTimeStamp']]
        tagged[clip] = [(float(e['start']), float(e['end']), e.get('tag') or '')
                        for e in record['NoiseSubCategoryTimeStamp']]
    eval_rows, eval_reference, eval_tagged = [], {}, {}
    for clip in natural_ids:
        eval_rows.append(L.inference_row(clip, L.PIPELINE / 'val' / clip, duration_of[clip]))
        eval_reference[clip] = reference[clip]
        eval_tagged[clip] = tagged[clip]

    goldval = json.loads((REBUILD / 'goldval-500.json').read_text())
    goldval_rows, goldval_reference = [], {}
    wanted = set(goldval['clips'])
    paths, durations = {}, {}
    for row in L.read_jsonl(KAGGLE_INPUT / 'training.jsonl'):
        if row['clip_id'] in wanted:
            durations[row['clip_id']] = row['duration']
    for record in L.read_jsonl(VAANI / 'manifest.jsonl'):
        if record['clip_id'] in wanted:
            paths[record['clip_id']] = VAANI / record['audio_path']
    for clip in goldval['clips']:
        goldval_rows.append(L.inference_row(clip, paths[clip], durations[clip]))
        goldval_reference[clip] = [(float(a), float(b)) for a, b in goldval['reference'][clip]]

    pools = {'gold': gold_rows, 'silver': silver_rows, 'weak': weak_rows}
    census.update({name: len(rows) for name, rows in pools.items()},
                  eval_clips=len(eval_rows), goldval_clips=len(goldval_rows), arm=arm)
    return {'pools': pools, 'census': census, 'eval_rows': eval_rows,
            'eval_reference': eval_reference, 'eval_tagged': eval_tagged,
            'goldval_rows': goldval_rows, 'goldval_reference': goldval_reference}


# ------------------------------------------------------- long-event coverage

def union_of(intervals):
    merged = []
    for on, off in sorted(intervals):
        if merged and on <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], off)
        else:
            merged.append([on, off])
    return merged


def coverage_report(tagged_reference, predicted, clips, minimum_seconds=2.0):
    """Median coverage of long, NON-bracketed references by the union of predictions."""
    values = []
    for clip in clips:
        merged = union_of(predicted.get(clip, []))
        for on, off, tag in tagged_reference.get(clip, []):
            if off - on < minimum_seconds or str(tag).startswith('['):
                continue
            covered = sum(max(0., min(off, b) - max(on, a)) for a, b in merged)
            values.append(covered / (off - on))
    values.sort()
    if not values:
        return {'references': 0, 'median_coverage': None, 'fraction_below_half': None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else .5 * (values[middle - 1] + values[middle])
    return {'references': len(values), 'median_coverage': median,
            'fraction_below_half': sum(1 for v in values if v < .5) / len(values),
            'mean_coverage': sum(values) / len(values),
            'fraction_above_0.9': sum(1 for v in values if v > .9) / len(values)}


# --------------------------------------------------------------- endpoints

SHORT_SECONDS = 0.5
LONG_SECONDS = 2.0


def endpoint_report(reference, predicted, clips):
    """Endpoint error on SHORT references, and precision/recall split by duration.

    * the endpoint errors pair every reference under 0.5 s with the prediction that
      OVERLAPS it most (no tolerance involved), so they measure the error even when the
      pair fails the organizer's max(0.2 d, 0.05 s) test;
    * recall is split by the REFERENCE duration, precision by the PREDICTION duration,
      which is the only way a prediction can be assigned to a band at all.
    """
    onset_errors, offset_errors = [], []
    counts = {'short_reference': 0, 'short_reference_matched': 0,
              'long_reference': 0, 'long_reference_matched': 0,
              'short_prediction': 0, 'short_prediction_matched': 0,
              'long_prediction': 0, 'long_prediction_matched': 0}
    for clip in clips:
        refs = list(reference.get(clip, []))
        preds = list(predicted.get(clip, []))
        matched_ref, matched_pred = L.match_indices(refs, preds)
        for index, (on, off) in enumerate(refs):
            length = off - on
            if length < SHORT_SECONDS:
                counts['short_reference'] += 1
                counts['short_reference_matched'] += index in matched_ref
                best, overlap = None, 0.
            elif length >= LONG_SECONDS:
                counts['long_reference'] += 1
                counts['long_reference_matched'] += index in matched_ref
                continue
            else:
                continue
            for p_on, p_off in preds:
                value = min(off, p_off) - max(on, p_on)
                if value > overlap:
                    best, overlap = (p_on, p_off), value
            if best is not None:
                onset_errors.append(abs(best[0] - on))
                offset_errors.append(abs(best[1] - off))
        for index, (on, off) in enumerate(preds):
            length = off - on
            if length < SHORT_SECONDS:
                counts['short_prediction'] += 1
                counts['short_prediction_matched'] += index in matched_pred
            elif length >= LONG_SECONDS:
                counts['long_prediction'] += 1
                counts['long_prediction_matched'] += index in matched_pred

    def median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else .5 * (values[middle - 1] + values[middle])

    def ratio(a, b):
        return (counts[a] / counts[b]) if counts[b] else None

    return {'short_seconds': SHORT_SECONDS, 'long_seconds': LONG_SECONDS,
            'median_abs_onset_error_short': median(onset_errors),
            'median_abs_offset_error_short': median(offset_errors),
            'overlap_pairs_short': len(onset_errors),
            'recall_short_references': ratio('short_reference_matched', 'short_reference'),
            'precision_short_predictions': ratio('short_prediction_matched', 'short_prediction'),
            'recall_long_references': ratio('long_reference_matched', 'long_reference'),
            'precision_long_predictions': ratio('long_prediction_matched', 'long_prediction'),
            'counts': counts}


BUCKETS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9))
BUCKET_NAMES = ('<0.5', '0.5-1', '1-2', '>=2')


def bucket_endpoints(reference, predicted, clips):
    """Recall and endpoint error per reference-duration bucket.

    Recall uses the organizer tolerance; the endpoint errors pair a reference with the
    prediction that overlaps it most, so a long dose that damages short events shows up
    as a larger |onset| / |offset| error in the short buckets even when recall holds.
    """
    totals = [0] * len(BUCKETS)
    hits = [0] * len(BUCKETS)
    onset = [[] for _ in BUCKETS]
    offset = [[] for _ in BUCKETS]
    for clip in clips:
        refs = list(reference.get(clip, []))
        preds = list(predicted.get(clip, []))
        matched_ref, _ = L.match_indices(refs, preds)
        for index, (on, off) in enumerate(refs):
            length = off - on
            bucket = next(i for i, (low, high) in enumerate(BUCKETS) if low <= length < high)
            totals[bucket] += 1
            if index in matched_ref:
                hits[bucket] += 1
            best, overlap = None, 0.
            for p_on, p_off in preds:
                value = min(off, p_off) - max(on, p_on)
                if value > overlap:
                    best, overlap = (p_on, p_off), value
            if best is not None:
                onset[bucket].append(abs(best[0] - on))
                offset[bucket].append(abs(best[1] - off))

    def median(values):
        if not values:
            return None
        values = sorted(values)
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else .5 * (values[middle - 1] + values[middle])

    return {name: {'references': totals[i], 'matched': hits[i],
                   'recall': (hits[i] / totals[i]) if totals[i] else None,
                   'overlap_pairs': len(onset[i]),
                   'median_abs_onset_error': median(onset[i]),
                   'median_abs_offset_error': median(offset[i])}
            for i, name in enumerate(BUCKET_NAMES)}


# ------------------------------------------------------------------- prediction

def predict_multi(model, api, rows, decoders, cap=4):
    """One forward pass, decoded at every threshold in `decoders`."""
    import torch
    previous = getattr(model, 'frequency_augmentation', None)
    model.eval()
    model.frequency_augmentation = None
    out = {name: {} for name in decoders}
    with torch.inference_mode():
        for selected in L.packs(rows, cap):
            for row, values in api.infer_outputs(model, selected, len(selected)):
                candidates = api.candidate_intervals(row, values['presence'], values['distances'])
                for name, decoder in decoders.items():
                    record = api.decode_candidates(row, candidates, dict(decoder))
                    out[name][record['clip_id']] = [(e['onset'], e['offset'])
                                                    for e in record['events']]
    model.frequency_augmentation = previous
    return out
