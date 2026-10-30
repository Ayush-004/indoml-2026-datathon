"""Local (Mac) data layer for the C2 consistency recipe (S1 uses only its row helpers; see train_s1.py).

This is the /kaggle/input half of the C2 consistency kernel, re-pointed at local files. Row shapes, pool filters, caps and
the evaluation populations are the kernel's, unchanged; only the paths differ.

  training.jsonl          the same manifest the Kaggle dataset carries, read locally
                          (T1_VAANI_INPUTS = data_prep/vaani/stage_full_inputs.py output)
  audio                   T1_VAANI_FULL/audio/... (WAV, data_prep/vaani/prepare_full.py) instead of the FLAC
  generated               gold_synth.py WAVs, filtered by the SAME 6,000-row gen-manifest.jsonl the kernel used
  unlabelled domain       T1_ORGANIZER_AUDIO/test/*.wav, ONLY for the C2 consistency arm (build_pools with
                          need_unlabelled=True). S1 does not call build_pools of this file and reads no test audio.
  evaluation              all 1,806 organizer natural VALIDATION clips (never trained on)
                          + the fixed 500-clip Gold-val slice
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from collections import Counter
import json
import math
from pathlib import Path

import os

HERE = Path(__file__).resolve().parent
# Local inputs (environment variables; see the README, 'Training of the members'):
VAANI = Path(os.environ.get('T1_VAANI_FULL', 'vaani-full'))                 # manifest.jsonl + audio/ (prepare_full.py)
KAGGLE_INPUT = Path(os.environ.get('T1_VAANI_INPUTS', 'vaani-inputs'))      # training.jsonl, organizer-*.jsonl
REBUILD = Path(os.environ.get('T1_GOLD_POOLS', 'gold-pools'))               # prepare_gold_pools.py output
GEN_MANIFEST_LOCAL = Path(os.environ.get('T1_GOLD_SYNTH_MANIFEST', 'gold-synth/manifest.jsonl'))
PIPELINE = Path(os.environ.get('T1_ORGANIZER_AUDIO', 'organizer-audio'))    # val/*.wav (organizer validation audio)
COMMON = Path(os.environ.get('T1_COMMON', 'common'))                        # val_meta.json (= validationMetadata.json),
                                                                            # scorer.py (organizer scorer copy, eval only)
DURATION_CACHE = HERE / 'test-durations.json'

CATEGORIES = ('animal', 'vehicle_traffic', 'baby_child', 'singing_music',
              'phone_signal_alarm', 'appliance_machine', 'human_non_speech')
DECODER = {'threshold': .65, 'nms_iou': 0., 'min_duration': .08}

# Kernel constants. CAPS is the per-clip native-window cap of each pool AND the window
# budget of its batch; MIX_MAC is the reduced per-update clip count for this machine.
CAPS = {'gold': 20, 'generated': 8, 'weak': 20, 'unlab_domain': 8, 'unlab_weak': 8}
MIX_MAC = {'gold': 6, 'generated': 2, 'weak': 16, 'unlab_domain': 2, 'unlab_weak': 2}
SEEDS = {'gold': 149, 'generated': 249, 'weak': 51, 'unlab_domain': 449, 'unlab_weak': 549}
HELD_OUT_FOLD = 1

PARENT_NATURAL_1806 = {'combined': 1.0758464842826536, 'f1': 0.3958299397295976,
                       'dice': 0.680016544553056}
PARENT_GOLDVAL500 = 1.3273


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def load_scorer():
    import importlib.util
    spec = importlib.util.spec_from_file_location('indoml_scorer', COMMON / 'scorer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_windows(duration):
    from indoml.pretrained import window_starts
    return len(window_starts(round(duration * 16000)))


def training_row(clip_id, audio_path, duration, events, categories, *, supervision='strong_gold',
                 quality='verified_timestamps', windows=None):
    return {'clip_id': clip_id, 'audio_path': str(audio_path), 'duration': float(duration),
            'num_frames': (round(duration * 16000) + 639) // 640,
            'frame_hop': .04, 'frame_offset': .02, 'feature_dim': 768,
            'events': events, 'positive_categories': sorted(categories),
            'quality': quality, 'supervision': supervision,
            'native_windows': windows if windows is not None else native_windows(duration)}


def inference_row(clip_id, audio_path, duration):
    return {'clip_id': clip_id, 'audio_path': str(audio_path), 'duration': float(duration),
            'num_frames': (round(duration * 16000) + 639) // 640,
            'frame_hop': .04, 'frame_offset': .02, 'feature_dim': 768}


# ---------------------------------------------------------------------- populations

def build_pools(*, need_training=True, need_unlabelled=True):
    census = {'dropped': {}}

    def drop(key):
        census['dropped'][key] = census['dropped'].get(key, 0) + 1

    local_audio = {row['clip_id']: VAANI / row['audio_path']
                   for row in read_jsonl(VAANI / 'manifest.jsonl')}
    gold_rows, weak_rows = [], []
    if need_training:
        gold_train = set(json.loads((REBUILD / 'gold-split.json').read_text())['train'])
        for row in read_jsonl(KAGGLE_INPUT / 'training.jsonl'):
            if row['clip_id'] not in local_audio:
                # 5 of the 90,635 uploaded rows have no row in the local vaani-full
                # manifest. They are dropped from every pool and counted here.
                drop('no_local_audio_' + row['supervision'])
                continue
            if row['supervision'] == 'strong_gold':
                if row['clip_id'] not in gold_train:
                    drop('gold_not_in_gold_train')
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
                gold_rows.append(training_row(row['clip_id'], local_audio[row['clip_id']],
                                              row['duration'], events, row['positive_categories'],
                                              windows=row['native_windows']))
            elif row['supervision'] == 'weak_clip':
                categories = [c for c in (row['positive_categories'] or []) if c in CATEGORIES]
                if not categories:
                    drop('weak_no_category')
                    continue
                if row['native_windows'] > CAPS['weak']:
                    drop('weak_too_many_windows')
                    continue
                weak_rows.append(training_row(row['clip_id'], local_audio[row['clip_id']],
                                              row['duration'], None, categories,
                                              supervision='weak_clip', quality=row['quality'],
                                              windows=row['native_windows']))

    gen_rows = []
    if need_training:
        local_gen = {row['clip_id']: Path(row['audio_path'])
                     for row in read_jsonl(GEN_MANIFEST_LOCAL)}
        for record in read_jsonl(REBUILD / 'gen-manifest.jsonl'):
            path = local_gen.get(record['clip_id'])
            if path is None or not path.is_file():
                drop('generated_missing_audio')
                continue
            row = training_row(record['clip_id'], path, record['duration'], record['events'],
                               record['positive_categories'])
            if row['native_windows'] > CAPS['generated']:
                drop('generated_too_many_windows')
                continue
            row['recipe'] = record['recipe']
            gen_rows.append(row)

    unlab_domain, unlab_weak = [], []
    if need_unlabelled:
        import soundfile as sf
        cache = json.loads(DURATION_CACHE.read_text()) if DURATION_CACHE.is_file() else {}
        dirty = False
        for path in sorted((PIPELINE / 'test').glob('*.wav')):
            key = path.name
            if key not in cache:
                info = sf.info(str(path))
                if info.samplerate != 16000:
                    raise ValueError('Expected 16 kHz test audio: ' + key)
                cache[key] = info.frames / 16000.
                dirty = True
            row = inference_row(path.stem + '.wav', path, cache[key])
            row['native_windows'] = native_windows(row['duration'])
            row['quality'] = row['origin'] = 'organizer_test_natural'
            if row['native_windows'] > CAPS['unlab_domain']:
                drop('unlabelled_test_too_many_windows')
                continue
            unlab_domain.append(row)
        if dirty:
            DURATION_CACHE.write_text(json.dumps(cache, sort_keys=True))
        for row in weak_rows:
            if row['native_windows'] <= CAPS['unlab_weak']:
                item = inference_row(row['clip_id'], row['audio_path'], row['duration'])
                item['native_windows'] = row['native_windows']
                item['quality'] = item['origin'] = 'vaani_' + row['quality']
                unlab_weak.append(item)

    # ------------------------------------------------------------- evaluation sets
    duration_of = {row['clip_id']: row['duration']
                   for row in read_jsonl(KAGGLE_INPUT / 'organizer-inference.jsonl')}
    natural_ids = sorted(json.loads((REBUILD / 'folds.json').read_text())['clip_fold'])
    meta = json.loads((COMMON / 'val_meta.json').read_text())
    reference_meta = {}
    for record in meta:
        if record.get('syntheticData'):
            continue
        reference_meta[record['segmentFileName']] = [
            (float(e['start']), float(e['end'])) for e in record['NoiseSubCategoryTimeStamp']]
    reference_kaggle = {row['clip_id']: [(float(e['onset']), float(e['offset']))
                                         for e in row['events']]
                        for row in read_jsonl(KAGGLE_INPUT / 'organizer-references.jsonl')}
    disagreements = sum(1 for clip in natural_ids
                        if [tuple(round(v, 6) for v in pair) for pair in reference_meta[clip]]
                        != [tuple(round(v, 6) for v in pair) for pair in reference_kaggle[clip]])
    census['val_meta_vs_organizer_references_disagreements'] = disagreements
    eval_rows, eval_reference = [], {}
    for clip in natural_ids:
        path = PIPELINE / 'val' / clip
        if not path.is_file():
            raise SystemExit('Missing organizer validation audio: ' + clip)
        eval_rows.append(inference_row(clip, path, duration_of[clip]))
        eval_reference[clip] = reference_meta[clip]

    goldval = json.loads((REBUILD / 'goldval-500.json').read_text())
    goldval_rows, goldval_reference = [], {}
    kaggle_duration = {}
    if goldval['clips']:
        wanted = set(goldval['clips'])
        for row in read_jsonl(KAGGLE_INPUT / 'training.jsonl'):
            if row['clip_id'] in wanted:
                kaggle_duration[row['clip_id']] = row['duration']
    for clip in goldval['clips']:
        goldval_rows.append(inference_row(clip, local_audio[clip], kaggle_duration[clip]))
        goldval_reference[clip] = [(float(a), float(b)) for a, b in goldval['reference'][clip]]

    pools = {'gold': gold_rows, 'generated': gen_rows, 'weak': weak_rows,
             'unlab_domain': unlab_domain, 'unlab_weak': unlab_weak}
    census.update({name: len(rows) for name, rows in pools.items()},
                  eval_clips=len(eval_rows), goldval_clips=len(goldval_rows),
                  generated_recipes=dict(Counter(r.get('recipe') for r in gen_rows)),
                  unlabelled_origins=dict(Counter(r['origin'] for r in unlab_domain)))
    return {'pools': pools, 'census': census, 'eval_rows': eval_rows,
            'eval_reference': eval_reference, 'goldval_rows': goldval_rows,
            'goldval_reference': goldval_reference}


# ------------------------------------------------------------------- diagnostics

DURATION_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9))
BUCKET_NAMES = ('<0.25', '0.25-0.5', '0.5-1', '1-2', '2-5', '5+')


def bucket_of(duration):
    for index, (low, high) in enumerate(DURATION_BUCKETS):
        if low <= duration < high:
            return index
    return len(DURATION_BUCKETS) - 1


def match_indices(ref_events, pred_events, tolerance_frac=0.20):
    matched_ref, matched_pred, candidates = set(), set(), []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    return matched_ref, matched_pred


def diagnostics(reference, predicted, clips):
    hits = [0] * len(DURATION_BUCKETS)
    totals = [0] * len(DURATION_BUCKETS)
    ref_seconds = pred_seconds = 0.0
    ref_lengths, pred_lengths = [], []
    for clip in clips:
        refs = list(reference.get(clip, []))
        preds = list(predicted.get(clip, []))
        matched, _ = match_indices(refs, preds)
        for index, (on, off) in enumerate(refs):
            length = off - on
            totals[bucket_of(length)] += 1
            ref_seconds += length
            ref_lengths.append(length)
            if index in matched:
                hits[bucket_of(length)] += 1
        for on, off in preds:
            pred_seconds += off - on
            pred_lengths.append(off - on)
    ref_lengths.sort()
    pred_lengths.sort()

    def median(values):
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else .5 * (values[middle - 1] + values[middle])

    return {'bucket_names': list(BUCKET_NAMES), 'reference_events_per_bucket': totals,
            'matched_events_per_bucket': hits,
            'recall_per_bucket': [(hits[i] / totals[i]) if totals[i] else None
                                  for i in range(len(totals))],
            'reference_seconds': ref_seconds, 'predicted_seconds': pred_seconds,
            'predicted_over_reference_seconds': (pred_seconds / ref_seconds) if ref_seconds else None,
            'median_reference_duration': median(ref_lengths),
            'median_predicted_duration': median(pred_lengths),
            'reference_events': len(ref_lengths), 'predicted_events': len(pred_lengths),
            'clips': len(clips)}


# ------------------------------------------------------------------- prediction

def packs(rows, cap=4):
    current, windows = [], 0
    for row in rows:
        count = 1 + max(0, (row['num_frames'] - 250 + 124) // 125)
        if current and windows + count > cap:
            yield current
            current, windows = [], 0
        current.append(row)
        windows += count
    if current:
        yield current


def predict(model, api, rows, cap=4, decoder=None):
    import torch
    decoder = dict(DECODER if decoder is None else decoder)
    previous = getattr(model, 'frequency_augmentation', None)
    model.eval()
    model.frequency_augmentation = None
    out = []
    with torch.inference_mode():
        for selected in packs(rows, cap):
            for row, values in api.infer_outputs(model, selected, len(selected)):
                candidates = api.candidate_intervals(row, values['presence'], values['distances'])
                out.append(api.decode_candidates(row, candidates, decoder))
    model.frequency_augmentation = previous
    return {record['clip_id']: [(e['onset'], e['offset']) for e in record['events']]
            for record in out}


def score_population(scorer, reference, predicted):
    clips = sorted(reference)
    value = scorer.score(reference, predicted, clips=clips)
    return {'f1': value['f1'], 'dice': value['dice'], 'combined': value['f1'] + value['dice'],
            'tp': value.get('tp'), 'fp': value.get('fp'), 'fn': value.get('fn'),
            'precision': value.get('precision'), 'recall': value.get('recall')}
