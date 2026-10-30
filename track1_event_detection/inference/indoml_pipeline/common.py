"""Shared paths, constants and small I/O helpers for the inference pipeline.

Every path can be overridden with an environment variable, so the package runs
on a different machine without an edit.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import zipfile

SR = 16000
FRAME_HOP = 0.04
FRAME_OFFSET = 0.02
SHIFTS_MS = (0, 10, 20, 30)
SHIFTS = tuple(round(ms * SR / 1000) for ms in SHIFTS_MS)  # (0, 160, 320, 480) samples

# The seven P2 presence channels, in checkpoint order.
CATEGORIES = ('animal', 'vehicle_traffic', 'baby_child', 'human_non_speech',
              'phone_signal_alarm', 'singing_music', 'appliance_machine')

# The P2 decoder that produced the reference P2 predictions.
P2_DECODER = {'threshold': 0.65, 'nms_iou': 0.0, 'min_duration': 0.08}

# Default label-free refiner of refine.py (vote.py sets its own values).
REFINER_PARAMS = {'boundary': None, 'cascade': 0.55, 'cross': [0.6, 'keep'],
                  'prune': [0.7, False, 1e9], 'union': None}

# Weighted interval fusion over the four input shifts.
FUSE_PARAMS = {'iou': 0.6, 'support_scale': True, 'min_score': 0.5, 'class_agnostic': True}

# Template events at or above this NCC replace overlapping P2 events on natural clips (build_submission.py).
TEMPLATE_OVERRIDE_NCC = 0.9

# Test-set population (the organizer-stated counts), used by score.py to reweight the
# validation mix into a test-mix estimate.
TEST_NATURAL = 3437
TEST_SYNTHETIC = 2080


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default).expanduser()


PACKAGE = Path(__file__).resolve().parent
PIPELINE = PACKAGE.parent

VALIDATION_ZIP = _env_path('INDOML_VALIDATION_ZIP', 'validation.zip')
TEST_ZIP = _env_path('INDOML_TEST_ZIP', 'input_data.zip')
WORK = _env_path('INDOML_WORK', PIPELINE / 'work')

# Read-only model inputs that are not part of this package (see README, 'Setup'):
#   T1_NATIVE_BUNDLE  folder with asset-manifest.json, assets/ATST-F_strong_1.pt, assets/parent-native-e5.pt and
#                     source/PretrainedSED (the layout is in the README, 'Setup')
#   T1_P2_CHECKPOINT  the P2 checkpoint parent-p2.pt
NATIVE_BUNDLE = _env_path('T1_NATIVE_BUNDLE', 'native-bundle')
P2_CHECKPOINT = _env_path('T1_P2_CHECKPOINT', 'parent-p2.pt')

# The validation metadata comes out of validation.zip itself. INDOML_VAL_META can
# point at an extracted copy instead.
VAL_META_MEMBER = 'validation/validationMetadata.json'
VAL_META = Path(os.environ['INDOML_VAL_META']).expanduser() if os.environ.get('INDOML_VAL_META') \
    else None


VAL_NATURAL_PREFIX = 'validation/naturalNoisyAudio/'
VAL_SYNTHETIC_PREFIX = 'validation/syntheticNoiseAudio/'
VAL_CLEAN_PREFIX = 'validation/syntheticCleanRefAudio/'
TEST_PREFIX = 'audio/'


def threads():
    """Worker/thread budget, capped to keep the machine usable for other jobs."""
    return max(1, int(os.environ.get('INDOML_THREADS', '4')))


# ------------------------------------------------------------------ jsonl

def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    temporary.replace(path)


def events_of(path):
    """clip_id -> [(onset, offset)] from a predictions jsonl."""
    return {record['clip_id']: [(float(e['onset']), float(e['offset'])) for e in record['events']]
            for record in read_jsonl(path)}


# ------------------------------------------------------------------ metadata

_VALIDATION_RECORDS = None


def validation_records(validation_zip=None):
    """The 2,501 organizer validation records, read from validation.zip (cached in memory)."""
    global _VALIDATION_RECORDS
    if _VALIDATION_RECORDS is None:
        if VAL_META is not None:
            _VALIDATION_RECORDS = json.loads(VAL_META.read_text())
        else:
            with zipfile.ZipFile(validation_zip or VALIDATION_ZIP) as archive:
                _VALIDATION_RECORDS = json.loads(archive.read(VAL_META_MEMBER))
    return _VALIDATION_RECORDS


def validation_reference():
    """clip_id -> [(onset, offset)] organizer labels."""
    return {r['segmentFileName']: [(float(e['start']), float(e['end']))
                                   for e in r['NoiseSubCategoryTimeStamp']]
            for r in validation_records()}


def validation_zip_name(record):
    prefix = VAL_SYNTHETIC_PREFIX if record['syntheticData'] else VAL_NATURAL_PREFIX
    return prefix + record['segmentFileName']


# ------------------------------------------------------------------ audio

def zip_wav_names(zip_path, prefix=''):
    """Sorted wav members under `prefix`, without the macOS resource-fork entries."""
    with zipfile.ZipFile(zip_path) as archive:
        return sorted(n for n in archive.namelist()
                      if n.endswith('.wav') and n.startswith(prefix)
                      and '__MACOSX' not in n.split('/')
                      and not Path(n).name.startswith('._'))


def wav_duration(blob):
    import soundfile as sf
    info = sf.info(io.BytesIO(blob) if isinstance(blob, (bytes, bytearray)) else blob)
    return info.frames / info.samplerate


def durations_from_zip(zip_path, names):
    """name -> duration in seconds, read from the wav header inside the zip."""
    out = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in names:
            out[name] = wav_duration(archive.read(name))
    return out


def extract_audio(zip_path, names, destination):
    """Extract `names` from a zip into `destination`/<basename>, skipping files already there."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in names:
            target = destination / Path(name).name
            if target.exists():
                continue
            target.write_bytes(archive.read(name))
            written += 1
    return written
