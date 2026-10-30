"""Create the shared inner geographic split without changing original manifests."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import os

BASE = Path(os.environ.get('T1_WORK', '.'))          # work folder (vaani-full, atst-gold-train, ...)
PROJECT = Path(__file__).resolve().parents[2] / 'atst_red'
sys.path.insert(0, str(PROJECT))
from indoml.data import read_manifest, resolve_audio_path
from indoml.prepare import _write_json, _write_manifest, audit_rows
from indoml.split_geographic import geographic_key, split_geographic

SOURCE = BASE / 'atst-gold-train/feature_manifest.jsonl'
OUTER = BASE / 'atst-gold-val/feature_manifest.jsonl'
OUT = BASE / 'inner-split'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def resolve_features(row, manifest):
    path = Path(row['feature_path'])
    return path.resolve() if path.is_absolute() else (Path(manifest).parent / path).resolve()


def summary(rows):
    categories, durations = Counter(), Counter()
    for row in rows:
        for event in row['events']:
            categories[event['category']] += 1
            durations[event['category']] += event['offset'] - event['onset']
    return {**audit_rows(rows), 'category_event_counts': dict(sorted(categories.items())),
            'category_event_duration_seconds': dict(sorted(durations.items())),
            'geographic_groups': len({geographic_key(row) for row in rows}),
            'feature_frames': sum(row['num_frames'] for row in rows),
            'feature_cache_ids': sorted({row['feature_cache_id'] for row in rows}),
            'clip_id_set_sha256': hashlib.sha256('\n'.join(sorted(row['clip_id'] for row in rows)).encode()).hexdigest()}


def assert_disjoint(first, second, first_manifest, second_manifest):
    checks = {
        'clip_ids': lambda row, path: row['clip_id'],
        'audio_paths': lambda row, path: str(resolve_audio_path(row, path).resolve()),
        'feature_paths': lambda row, path: str(resolve_features(row, path)),
        'geographic_groups': lambda row, path: geographic_key(row),
        'original_audio_filenames': lambda row, path: Path(row['source']['audio_path']).name,
    }
    for name, key in checks.items():
        assert {key(row, first_manifest) for row in first}.isdisjoint({key(row, second_manifest) for row in second}), name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, outer = read_manifest(SOURCE), read_manifest(OUTER)
    retained, excluded = [], []
    for original in source:
        declared = set(original.get('source', {}).get('metadata', {}).get('NoiseCategory') or [])
        observed = {event['category'] for event in original['events'] or []}
        reason = None
        if original['quality'] != 'verified_timestamps' or original['events'] is None:
            reason = 'not_gold_timestamped'
        elif declared - observed:
            reason = 'gold_positive_category_without_timestamp'
        if reason:
            excluded.append({'clip_id': original['clip_id'], 'reason': reason})
            continue
        row = dict(original)
        row['audio_path'] = str(resolve_audio_path(row, SOURCE).resolve())
        row['feature_path'] = str(resolve_features(row, SOURCE))
        assert Path(row['audio_path']).is_file() and Path(row['feature_path']).is_file()
        assert isinstance(row.get('feature_cache_id'), str) and row['feature_cache_id']
        retained.append(row)
    assert len(source) == 9663 and len(retained) == 9652 and len(excluded) == 11
    cache_ids = {row['feature_cache_id'] for row in retained + outer}
    assert len(cache_ids) == 1
    assert_disjoint(retained, outer, SOURCE, OUTER)
    clean = OUT / 'clean-gold-source.jsonl'
    _write_manifest(clean, retained)
    report = split_geographic(clean, OUT, val_fraction=.125, min_fraction=.10,
                              max_fraction=.15, seed=43, search_trials=512)
    train, validation = read_manifest(OUT / 'train.jsonl'), read_manifest(OUT / 'val.jsonl')
    for name, values in [('train.jsonl', train), ('val.jsonl', validation)]:
        for row in values:
            row['audio_path'] = str(resolve_audio_path(row, OUT / name).resolve())
            row['feature_path'] = str(resolve_features(row, OUT / name))
            assert Path(row['audio_path']).is_absolute() and Path(row['feature_path']).is_absolute()
            assert Path(row['audio_path']).is_file() and Path(row['feature_path']).is_file()
        _write_manifest(OUT / name, values)
    assert {row['clip_id'] for row in train + validation} == {row['clip_id'] for row in retained}
    assert len(train) + len(validation) == len(retained)
    assert_disjoint(train, validation, OUT / 'train.jsonl', OUT / 'val.jsonl')
    assert_disjoint(train, outer, OUT / 'train.jsonl', OUTER)
    assert_disjoint(validation, outer, OUT / 'val.jsonl', OUTER)
    audit = {'purpose': 'Shared inner development split for controlled model/weight comparisons; outer1443 remains excluded from training and inner calibration.',
             'outer_notice': 'The original outer set has already been used as development data and is not an independent test set.',
             'seed': 43, 'target_gold_validation_fraction': .125,
             'actual_gold_validation_fraction': len(validation) / len(retained),
             'source_feature_manifest_sha256': digest(SOURCE), 'outer_feature_manifest_sha256': digest(OUTER),
             'source_rows': len(source), 'clean_gold_rows': len(retained), 'excluded_source_rows': excluded,
             'all_paths_absolute': True, 'train_calibration_outer_geographic_disjoint': True,
             'feature_cache_id': next(iter(cache_ids)),
             'inner_train': summary(train), 'inner_validation': summary(validation), 'outer': summary(outer),
             'manifest_sha256': {'train': digest(OUT / 'train.jsonl'), 'validation': digest(OUT / 'val.jsonl')},
             'held_out_groups': report['held_out_groups'],
             'training_filter': 'Equivalent to the existing Gold FeatureDataset policy: exclude declared positive categories without timestamps.'}
    report['path_policy'] = 'audio_path and feature_path are both absolute'
    report['output_manifest_sha256'] = audit['manifest_sha256']
    _write_json(OUT / 'split-report.json', report)
    _write_json(OUT / 'audit.json', audit)
    print(json.dumps({key: audit[key] for key in ('seed', 'clean_gold_rows', 'actual_gold_validation_fraction',
          'feature_cache_id', 'manifest_sha256', 'inner_train', 'inner_validation')}, indent=2))


if __name__ == '__main__':
    main()
