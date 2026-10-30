"""Build the Kaggle input dataset "gold-pools" of the training kernels.

Contents:
  folds.json         grouped 5-fold split of the 1,806 organizer natural validation clips (make_folds.py, byte copy)
  gold-split.json    Gold-train / Gold-val clip_id lists (T1_WORK/vaani-gold/{train,val}.jsonl)
  goldval-500.json   a fixed 500-clip Gold-val evaluation slice with its reference events
  gen-manifest.jsonl the fold-safe generated rows of gold_synth.py (recipe B + recipe A on Gold beds; clips built on
                     validation clean-reference beds are excluded), audio_path rewritten to `gen-audio/<stem>.flac`
  gen-audio.zip      those clips as FLAC (lossless, verified sample-exact)

The organizer labels are not part of this dataset: the "vaani-inputs" dataset (stage_full_inputs.py) carries
organizer-references.jsonl (2,501 rows) and organizer-inference.jsonl.

Inputs: T1_WORK (vaani-gold/, full-input-staging/), T1_FOLDS (folds.json), T1_GOLD_SYNTH_MANIFEST (manifest.jsonl of
gold_synth.py). Output: T1_WORK/gold-pools/dataset/.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import zipfile

WORK = Path(os.environ.get('T1_WORK', '.'))
HERE = WORK / 'gold-pools'
OUT = HERE / 'dataset'
STAGE = HERE / 'stage-gen-audio'
FOLDS_SRC = Path(os.environ.get('T1_FOLDS', 'folds.json'))
GOLD_TRAIN = WORK / 'vaani-gold/train.jsonl'
GOLD_VAL = WORK / 'vaani-gold/val.jsonl'
KAGGLE_TRAINING = WORK / 'full-input-staging/input-upload/training.jsonl'
SYNTH = Path(os.environ.get('T1_GOLD_SYNTH_MANIFEST', 'gold-synth/manifest.jsonl'))
CATEGORIES = ('animal', 'vehicle_traffic', 'baby_child', 'singing_music',
              'phone_signal_alarm', 'appliance_machine', 'human_non_speech')
GOLDVAL_CLIPS = 500
PREDICT_CAP = 4


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def convert(item):
    """int16 WAV -> FLAC, then read back and require identical samples."""
    import numpy as np
    import soundfile as sf
    source, destination = item
    data, rate = sf.read(source, dtype='int16', always_2d=False)
    if rate != 16000 or data.ndim != 1:
        return {'ok': False, 'source': source, 'why': 'geometry'}
    sf.write(destination, data, rate, format='FLAC', subtype='PCM_16')
    back, rate_back = sf.read(destination, dtype='int16', always_2d=False)
    if rate_back != 16000 or not np.array_equal(back, data):
        return {'ok': False, 'source': source, 'why': 'roundtrip'}
    return {'ok': True, 'source': source, 'samples': int(data.shape[0]),
            'bytes': Path(destination).stat().st_size}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---------------------------------------------------------------- folds
    folds = json.loads(FOLDS_SRC.read_text())
    shutil.copyfile(FOLDS_SRC, OUT / 'folds.json')
    report['folds'] = {'clips': len(folds['clip_fold']), 'groups': len(folds['group_fold']),
                       'clips_per_fold': folds['clips_per_fold']}

    # ------------------------------------------------------------ gold split
    kaggle_rows = {row['clip_id']: row for row in jsonl(KAGGLE_TRAINING)}
    gold_train = [row['clip_id'] for row in jsonl(GOLD_TRAIN)]
    gold_val = [row['clip_id'] for row in jsonl(GOLD_VAL)]
    missing = [c for c in gold_train + gold_val if c not in kaggle_rows]
    if missing:
        raise SystemExit(f'{len(missing)} Gold clip ids are absent from the Kaggle manifest')
    write_json(OUT / 'gold-split.json', {
        'source': {'train': str(GOLD_TRAIN), 'val': str(GOLD_VAL)},
        'note': 'Geographic Gold split; ids index '
                'training.jsonl of the vaani-inputs dataset.',
        'train': gold_train, 'val': gold_val})
    report['gold_split'] = {'train': len(gold_train), 'val': len(gold_val)}

    # -------------------------------------------------------- gold-val slice
    items = []
    for record in jsonl(GOLD_VAL):
        events = [(float(e['onset']), float(e['offset'])) for e in record.get('events') or []]
        if not events or any(b <= a for a, b in events):
            continue
        kaggle = kaggle_rows[record['clip_id']]
        if kaggle['native_windows'] > PREDICT_CAP:
            continue
        items.append((record['clip_id'], events))
    items.sort(key=lambda item: item[0])
    if GOLDVAL_CLIPS and len(items) > GOLDVAL_CLIPS:
        items = items[::max(1, len(items) // GOLDVAL_CLIPS)][:GOLDVAL_CLIPS]
    write_json(OUT / 'goldval-500.json', {
        'selection': 'vaani-gold/val.jsonl, usable events, native_windows <= 4, '
                     'sorted by clip_id, strided to 500',
        'parent_p2_combined_recorded_locally': 1.3273,
        'clips': [c for c, _ in items],
        'reference': {c: [[a, b] for a, b in events] for c, events in items}})
    report['goldval'] = {'clips': len(items)}

    # ------------------------------------------------------- generated rows
    kept, dropped = [], {}

    def drop(key):
        dropped[key] = dropped.get(key, 0) + 1

    for record in jsonl(SYNTH):
        source = record.get('source', {})
        recipe = source.get('recipe')
        bed_kind = (source.get('bed') or {}).get('kind')
        if bed_kind == 'valref':
            drop('valref_bed')
            continue
        if recipe == 'A' and bed_kind != 'gold':
            drop('recipe_a_non_gold_bed')
            continue
        if recipe not in ('A', 'B'):
            drop('unknown_recipe')
            continue
        keys = sorted(set(source.get('fold_keys') or []))
        if not keys:
            drop('no_fold_keys')
            continue
        events = [{'onset': float(e['onset']), 'offset': float(e['offset']),
                   'category': e['category']} for e in record['events']]
        if not events or any(e['category'] not in CATEGORIES for e in events):
            drop('unusable_events')
            continue
        if any(e['offset'] <= e['onset'] for e in events):
            drop('degenerate_events')
            continue
        path = Path(record['audio_path'])
        if not path.is_file():
            drop('missing_audio')
            continue
        kept.append({'clip_id': record['clip_id'], 'audio_path': 'gen-audio/' + path.stem + '.flac',
                     'duration': float(record['duration']), 'events': events,
                     'fold_keys': keys, 'recipe': recipe,
                     'positive_categories': sorted({e['category'] for e in events}),
                     'quality': 'verified_timestamps', 'supervision': 'strong_gold',
                     'source_wav': str(path)})
    report['generated'] = {'kept': len(kept), 'dropped': dropped}
    print(f'generated rows kept {len(kept)} dropped {dropped}', flush=True)

    # ------------------------------------------------------------ FLAC pass
    jobs = [(row.pop('source_wav'), str(STAGE / Path(row['audio_path']).name)) for row in kept]
    todo = [job for job in jobs if not Path(job[1]).is_file()]
    print(f'converting {len(todo)} of {len(jobs)} clips to FLAC', flush=True)
    results = []
    if todo:
        with mp.Pool(3) as pool:
            for index, result in enumerate(pool.imap_unordered(convert, todo, chunksize=16)):
                results.append(result)
                if (index + 1) % 500 == 0:
                    print(f'  {index + 1}/{len(todo)}', flush=True)
    bad = [r for r in results if not r['ok']]
    if bad:
        raise SystemExit(f'{len(bad)} clips failed the lossless round trip: {bad[:3]}')
    total_bytes = sum(Path(destination).stat().st_size for _, destination in jobs)
    report['flac'] = {'files': len(jobs), 'bytes': total_bytes,
                      'verified_sample_exact': len(results)}

    with (OUT / 'gen-manifest.jsonl').open('w') as stream:
        for row in kept:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    # ---------------------------------------------------------------- zip
    archive = OUT / 'gen-audio.zip'
    print(f'writing {archive} ({total_bytes / 2**30:.2f} GiB of FLAC)', flush=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as zf:
        for _, destination in sorted(jobs, key=lambda job: job[1]):
            zf.write(destination, 'gen-audio/' + Path(destination).name)
    report['archive'] = {'path': str(archive), 'bytes': archive.stat().st_size}

    write_json(OUT / 'in-domain-contract.json', {
        'kind': 'indoml_in_domain_inputs_v1',
        'scope': 'Fold split, Gold split, Gold-val evaluation slice and fold-safe '
                 'generated training clips for the Kaggle P2 rebuild.',
        'organizer_labels': 'not included here; organizer-references.jsonl already lives in '
                            'the vaani-inputs dataset',
        **report})
    write_json(OUT / 'dataset-metadata.json', {
        'title': 'IndoML Gold Pools',
        'id': os.environ.get('KAGGLE_DATASET_OWNER', '<owner>') + '/gold-pools',
        'licenses': [{'name': 'other'}]})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
