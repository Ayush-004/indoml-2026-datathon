"""Shared loaders of the synthetic-branch scripts (job2, masters, noise tracks).

Read-only. Every artefact comes from the pipeline work folder INDOML_WORK of inference/indoml_pipeline:
templates/{exemplars,library,tau_router,hits_val,hits_test}.pkl (templates.py), test-durations.json and
test/predictions.jsonl (the template-decoded synthetic lines).
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'inference'))
from indoml_pipeline import common as _common  # noqa: E402
from indoml_pipeline.score import score as _score  # noqa: E402

WORK = _common.WORK
TEMPL = WORK / 'templates'
SR = 16000
VAL_ZIP = _common.VALIDATION_ZIP
TEST_ZIP = _common.TEST_ZIP


def score(reference, prediction, clips=None, weights=None):
    return _score(reference, prediction, clips, weights)


def load_pickle(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def exemplars():
    return load_pickle(TEMPL / 'exemplars.pkl')


def library():
    return load_pickle(TEMPL / 'library.pkl')


def tau():
    return load_pickle(TEMPL / 'tau_router.pkl')


def val_meta():
    return _common.validation_records()


def val_zip_name(record):
    prefix = ('validation/syntheticNoiseAudio/' if record['syntheticData']
              else 'validation/naturalNoisyAudio/')
    return prefix + record['segmentFileName']


def val_hits():
    """clip id -> [(template, sample, ncc)] for all 2,501 validation clips."""
    raw = load_pickle(TEMPL / 'hits_val.pkl')
    return {r['segmentFileName']: raw[val_zip_name(r)] for r in val_meta()}


def test_hits():
    """clip id (basename with .wav) -> [(template, sample, ncc)]."""
    raw = load_pickle(TEMPL / 'hits_test.pkl')
    return {Path(name).name: value for name, value in raw.items()}


def val_reference():
    return {r['segmentFileName']: [(float(e['start']), float(e['end']))
                                   for e in r['NoiseSubCategoryTimeStamp']] for r in val_meta()}


def val_is_synthetic():
    return {r['segmentFileName']: bool(r['syntheticData']) for r in val_meta()}


def val_durations():
    return {r['segmentFileName']: float(r['duration']) for r in val_meta()}


def test_durations():
    return {k: float(v) for k, v in json.loads((WORK / 'test-durations.json').read_text()).items()}


PATTERN = re.compile(r'_(\d+)_(\d+)\.wav$')


def is_synthetic_name(clip_id, duration, tolerance=1e-3):
    match = PATTERN.search(clip_id)
    if not match:
        return False
    span = (int(match.group(2)) - int(match.group(1))) / 1000.0
    return abs(span - duration) > tolerance


def test_synthetic_clips():
    durations = test_durations()
    return sorted(c for c, d in durations.items() if is_synthetic_name(c, d))


def folds():
    """clip id (without .wav? -> as stored) -> fold index."""
    import os
    path = Path(os.environ.get('T1_FOLDS', 'folds.json'))   # data_prep/kaggle_inputs/make_folds.py
    return json.loads(path.read_text())


def decode(hits, tau_vector, durations_of_template, blacklist=None):
    """Greedy non-overlapping decode, strongest NCC first."""
    candidates = []
    for index, position, value in hits:
        if blacklist is not None and index in blacklist:
            continue
        if value >= tau_vector[index]:
            onset = position / SR
            candidates.append((value, onset, onset + durations_of_template[index], index))
    candidates.sort(reverse=True)
    accepted = []
    for value, onset, offset, index in candidates:
        if all(offset <= a or onset >= b for _, a, b, _ in accepted):
            accepted.append((value, onset, offset, index))
    return sorted((onset, offset) for _, onset, offset, _ in accepted)


def fmt(result):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()
            if k in ('f1', 'dice', 'combined', 'tp', 'fp', 'fn')}
