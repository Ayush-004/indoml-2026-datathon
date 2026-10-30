"""Shared loaders of the whitened matched filter (WMF) scripts w00-w09 (whitened / exact-subtraction template
statistics).

Read-only against its inputs. Every cache lives in T1_WMF_CACHE (default: ./cache in this folder).
Audio stores are flat int16 files + a json index; a clip is read with np.fromfile(offset,
count), so no process maps or holds the whole store.

Sets
  valsyn   695 validation synthetic clips (noisy)            validation.zip
  valclean 695 clean references of valsyn (analysis only)   validation.zip
  valnat   1,806 validation natural clips (null)             validation.zip
  train    sampled natural clips of the training release (null)  vaani-full/audio
  testsyn  2,082 synthetic-routed test clips (application only)  input_data.zip
"""
from __future__ import annotations

import io
import json
import os
import pickle
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('T1_WMF_CACHE', HERE / 'cache'))
CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parents[1] / 'inference'))
from indoml_pipeline import common as _common  # noqa: E402
from indoml_pipeline.score import score as _score, clip_dice  # noqa: E402,F401

TEMPL = _common.WORK / 'templates'
HEAD = Path(os.environ.get('T1_JOB2_OUT', '.'))          # the job2 outputs (../job2)
VAL_ZIP = _common.VALIDATION_ZIP
TEST_ZIP = _common.TEST_ZIP
TRAIN_DIR = Path(os.environ.get('T1_WORK', '.')) / 'vaani-full'
JOB2ONLY = HEAD / 'test_synthetic_predictions_v2_job2only.jsonl'   # s40_emit_v2.py output

PAT = re.compile(r'_(\d+)_(\d+)\.wav$')
GP = re.compile(r'IISc_VaaniProject_[A-Z]_([A-Za-z]+)_([A-Za-z]+)_')


# ------------------------------------------------------------------ small io

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def dump_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------ validation metadata

_META = None


def val_meta():
    global _META
    if _META is None:
        p = CACHE / 'val_meta.json'
        if not p.exists():
            with zipfile.ZipFile(VAL_ZIP) as z:
                p.write_bytes(z.read('validation/validationMetadata.json'))
        _META = json.loads(p.read_text())
    return _META


def val_reference():
    return {r['segmentFileName']: [(float(e['start']), float(e['end'])) for e in r['NoiseSubCategoryTimeStamp']]
            for r in val_meta()}


def val_tags():
    return {r['segmentFileName']: [e['tag'] for e in r['NoiseSubCategoryTimeStamp']] for r in val_meta()}


def val_is_synthetic():
    return {r['segmentFileName']: bool(r['syntheticData']) for r in val_meta()}


def val_durations():
    return {r['segmentFileName']: float(r['duration']) for r in val_meta()}


def val_syn_clips():
    return sorted(r['segmentFileName'] for r in val_meta() if r['syntheticData'])


def val_nat_clips():
    return sorted(r['segmentFileName'] for r in val_meta() if not r['syntheticData'])


def geo_group(c):
    m = GP.match(c)
    return (m.group(1), m.group(2)) if m else ('?', '?')


def val_folds():
    """5 geography folds of the 695 validation synthetic clips (cross-fit rule: whole
    (state, district) groups, largest first into the smallest fold). Checked against the
    stored val_syn_folds.json, when that file is present."""
    vsyn = val_syn_clips()
    G = defaultdict(list)
    for c in vsyn:
        G[geo_group(c)].append(c)
    fold = {}
    load = [0] * 5
    for g, cs in sorted(G.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        f = int(np.argmin(load))
        load[f] += len(cs)
        for c in cs:
            fold[c] = f
    if (HEAD / 'val_syn_folds.json').is_file():
        stored = json.loads((HEAD / 'val_syn_folds.json').read_text())
        assert stored == fold, 'fold assignment differs from val_syn_folds.json'
    return fold


# ------------------------------------------------------------------ test

def test_durations():
    return {k: float(v) for k, v in json.loads((_common.WORK / 'test-durations.json').read_text()).items()}


def is_synthetic_name(clip_id, duration, tolerance=1e-3):
    m = PAT.search(clip_id)
    if not m:
        return False
    span = (int(m.group(2)) - int(m.group(1))) / 1000.0
    return abs(span - duration) > tolerance


def test_syn_clips():
    d = test_durations()
    return sorted(c for c, v in d.items() if is_synthetic_name(c, v))


def read_pred_jsonl(path):
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r['clip_id']] = [(float(e['onset']), float(e['offset'])) for e in r['events']]
    return out


# ------------------------------------------------------------------ library

_EX = None


def exemplars():
    global _EX
    if _EX is None:
        _EX = load_pickle(TEMPL / 'exemplars.pkl')
    return _EX


def library():
    return load_pickle(TEMPL / 'library.pkl')


def tau_router():
    return load_pickle(TEMPL / 'tau_router.pkl')


def kernel_index(k, clip):
    """index of the exemplar used for template k when clip is scanned (leave-one-clip-out,
    the same rule as templates._scan_one): the first exemplar not cut from clip; None if none."""
    for i, (src, _) in enumerate(exemplars()[k]['exs']):
        if src != clip:
            return i
    return None


def val_hits():
    raw = load_pickle(TEMPL / 'hits_val.pkl')
    out = {}
    for r in val_meta():
        pre = 'validation/syntheticNoiseAudio/' if r['syntheticData'] else 'validation/naturalNoisyAudio/'
        out[r['segmentFileName']] = raw[pre + r['segmentFileName']]
    return out


def test_hits():
    raw = load_pickle(TEMPL / 'hits_test.pkl')
    return {Path(n).name: v for n, v in raw.items()}


# ------------------------------------------------------------------ scoring

def score(ref, pred, clips=None):
    return _score(ref, pred, clips)


def fmt(r):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
            if k in ('f1', 'dice', 'combined', 'tp', 'fp', 'fn')}


def ok_pair(on, off, a, b):
    tol = max(0.2 * (b - a), 0.05)
    return abs(on - a) <= tol and abs(off - b) <= tol


# ------------------------------------------------------------------ audio stores

def _store_paths(name):
    return CACHE / f'{name}.i16', CACHE / f'{name}_index.json'


def store_index(name):
    return json.loads(_store_paths(name)[1].read_text())


def read_clip(name, clip, index=None):
    """float64 waveform in [-1, 1) of one clip of a store (no mapping of the store)."""
    dat, _ = _store_paths(name)
    index = index or store_index(name)
    off, n = index[clip]
    x = np.fromfile(dat, dtype=np.int16, count=n, offset=2 * off)
    return x.astype(np.float64) / 32768.0


def read_clip_i16(name, clip, index=None):
    dat, _ = _store_paths(name)
    index = index or store_index(name)
    off, n = index[clip]
    return np.fromfile(dat, dtype=np.int16, count=n, offset=2 * off)


def build_store(name, items):
    """items: iterable of (clip_id, loader) where loader() -> int16 array. Idempotent."""
    import soundfile as sf  # noqa: F401
    dat, idx = _store_paths(name)
    if dat.exists() and idx.exists():
        return store_index(name)
    index = {}
    off = 0
    tmp = dat.with_suffix('.tmp')
    with open(tmp, 'wb') as out:
        for clip, loader in items:
            x = loader()
            assert x.dtype == np.int16 and x.ndim == 1, clip
            out.write(x.tobytes())
            index[clip] = [off, int(len(x))]
            off += len(x)
    tmp.replace(dat)
    dump_json(idx, index)
    return index


def zip_loader(archive, member):
    import soundfile as sf

    def f():
        x, sr = sf.read(io.BytesIO(archive.read(member)), dtype='int16')
        assert sr == SR and x.ndim == 1, (member, sr, x.shape)
        return x
    return f


def file_loader(path):
    import soundfile as sf

    def f():
        x, sr = sf.read(str(path), dtype='int16')
        if x.ndim > 1:
            x = x[:, 0]
        assert sr == SR, (path, sr)
        return x
    return f


def train_null_sample(n=4000, seed=20260923, min_dur=3.0, max_dur=15.0):
    """A seeded sample of natural training-release clips for the null (router-natural only,
    duration inside the synthetic range), stratified by quality tier in proportion."""
    p = CACHE / f'train_sample_{n}_{seed}.json'
    if p.exists():
        return json.loads(p.read_text())
    rows = []
    with open(TRAIN_DIR / 'manifest.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if not (min_dur <= r['duration'] <= max_dur):
                continue
            if is_synthetic_name(r['clip_id'], r['duration']):
                continue
            rows.append((r['clip_id'], r['audio_path'], r['quality'], r['duration']))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=n, replace=False)
    sample = sorted([rows[i] for i in pick])
    dump_json(p, sample)
    return sample
