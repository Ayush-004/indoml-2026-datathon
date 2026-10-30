"""Shared helpers of the library-copy steps Q1 and Q2 (read-only against their inputs).

Q1  copies of the 203 VALIDATION library templates inside NATURAL clips mark labelled natural
    events (the organizers cut their noise files from labelled events of their natural recordings).
Q2  labelled events of the 1,806 validation natural clips, cut from the natural audio at the label
    extent, as candidate templates for the synthetic-routed test clips.

Every cache and output lives in this folder (cache/ and the json outputs). Inputs are only READ:
  ../wmf cache (T1_WMF_CACHE): {valnat,valsyn,valclean,train}.i16 + index (audio stores, np.fromfile),
  scan_{valnat,valsyn,valclean,train}/ (stage-1/2 scans of the 203 templates), {events,null}.pkl;
  ../wmf/theta.json (null thresholds theta_k), INDOML_WORK/templates/exemplars.pkl (the 203 validation templates),
  the organizer scorer files reference_event_f1.py and reference_segment_dice.py (T1_OFFICIAL_SCORER_DIR, executed
  verbatim; not included here),
  T1_BASE_PREDICTIONS: the earlier natural-side predictions that Q1 edits (val / test predictions.jsonl, base zip).
The detector (wstat.py here) is a byte-identical copy of ../wmf/wstat.py.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SR = 16000
HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'
WMF = HERE.parent / 'wmf'
WMF_CACHE = Path(os.environ.get('T1_WMF_CACHE', WMF / 'cache'))
WORK = Path(os.environ.get('INDOML_WORK', HERE.parents[1] / 'inference' / 'work'))
TEMPL = WORK / 'templates'
OFFICIAL = Path(os.environ.get('T1_OFFICIAL_SCORER_DIR', 'official'))
VAL_ZIP = Path(os.environ.get('INDOML_VALIDATION_ZIP', 'validation.zip'))
TEST_ZIP = Path(os.environ.get('INDOML_TEST_ZIP', 'input_data.zip'))
TRAIN_DIR = Path(os.environ.get('T1_WORK', '.')) / 'vaani-full'
BASE = Path(os.environ.get('T1_BASE_PREDICTIONS', 'base'))
SUB08_VAL = BASE / 'val' / 'predictions.jsonl'
SUB08_TEST = BASE / 'test' / 'predictions.jsonl'
BASE_ZIP = BASE / 'base.zip'
WMF_SYN = WMF / 'test_synthetic_predictions_wmf.jsonl'

FIELDS = ('k', 'p', 'S0', 'S1', 'S3', 'S4', 'S5', 'g', 'S3b', 'S5b')
F = {f: i for i, f in enumerate(FIELDS)}
PAT = re.compile(r'_(\d+)_(\d+)(?:\.wav)?$')

# generator facts (job2 / wmf w06_rules, used label-free)
MASTER = {'machine': list(range(103, 126)), 'child': list(range(84, 89)), 'bird': [64, 65, 66]}
MASTER_OF = {k: m for m, ks in MASTER.items() for k in ks}
MASTER_LEN = {'machine': 5.437, 'child': 5.211, 'bird': 9.077}
GAP = 0.200
SLACK = 0.005
EDGE_ON = 0.250
EDGE_OFF = 0.251
MAX_EVENTS = 5


# ------------------------------------------------------------------ small io

def stem(c):
    return c[:-4] if c.endswith('.wav') else c


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(path, obj):
    path = Path(path)
    assert HERE in path.resolve().parents, path
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def dump_json(path, obj):
    path = Path(path)
    assert HERE in path.resolve().parents, path
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def read_pred_jsonl(path):
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[stem(r['clip_id'])] = [(float(e['onset']), float(e['offset'])) for e in r['events']]
    return out


# ------------------------------------------------------------------ official scorer (verbatim snippets)

def _load_official():
    ns = {}
    for f in ('reference_event_f1.py', 'reference_segment_dice.py'):
        p = OFFICIAL / f
        exec(compile(p.read_text(), str(p), 'exec'), ns)
    return ns


OFF = _load_official()


def official(ref, pred, clips):
    """official event F1 + segment Dice on the subset `clips` (ref restricted to clips; pred
    restricted to the same clips, as the natural-subset calibration assumes)."""
    r = {c: list(ref[c]) for c in clips}
    p = {c: list(pred.get(c, [])) for c in clips}
    f1 = OFF['event_based_f1'](r, p)
    dice = OFF['segment_dice'](r, p)
    TP = FP = FN = 0
    for c in clips:
        tp, fp, fn = OFF['match_events'](r[c], p[c])
        TP += tp
        FP += fp
        FN += fn
    return {'f1': f1, 'dice': dice, 'combined': f1 + dice, 'tp': TP, 'fp': FP, 'fn': FN}


def clip_counts(ref_ev, pred_ev):
    tp, fp, fn = OFF['match_events'](list(ref_ev), list(pred_ev))
    d = OFF['segment_dice']({'c': list(ref_ev)}, {'c': list(pred_ev)})
    return tp, fp, fn, d


def matched_pairs(ref_ev, pred_ev):
    """the official greedy closest-first matching, returning {ref index: pred index}."""
    cands = []
    for ri, (a, b) in enumerate(ref_ev):
        tol = max(0.2 * (b - a), 0.05)
        for pi, (x, y) in enumerate(pred_ev):
            if abs(x - a) <= tol and abs(y - b) <= tol:
                cands.append((abs(x - a) + abs(y - b), ri, pi))
    mr, mp = {}, set()
    for _, ri, pi in sorted(cands):
        if ri not in mr and pi not in mp:
            mr[ri] = pi
            mp.add(pi)
    return mr


def fmt(r, d=4):
    return {k: (round(v, d) if isinstance(v, float) else v) for k, v in r.items()}


LB_NAT = (0.4484, 0.6228)
LB_SYN = (0.5427, 0.3770)


# ------------------------------------------------------------------ validation metadata (read only)

_META = None


def val_meta():
    global _META
    if _META is None:
        _META = json.loads((WMF_CACHE / 'val_meta.json').read_text())
    return _META


def val_labels():
    """clip (with .wav) -> [(on, off, tag)]"""
    return {r['segmentFileName']: [(float(e['start']), float(e['end']), e['tag'])
                                   for e in r['NoiseSubCategoryTimeStamp']] for r in val_meta()}


def val_reference_stem():
    return {stem(r['segmentFileName']): [(float(e['start']), float(e['end']))
                                         for e in r['NoiseSubCategoryTimeStamp']] for r in val_meta()}


def val_nat_clips():
    return sorted(r['segmentFileName'] for r in val_meta() if not r['syntheticData'])


def val_syn_clips():
    return sorted(r['segmentFileName'] for r in val_meta() if r['syntheticData'])


def val_durations():
    return {r['segmentFileName']: float(r['duration']) for r in val_meta()}


# ------------------------------------------------------------------ router / test

def test_durations():
    return {k: float(v) for k, v in json.loads((WORK / 'test-durations.json').read_text()).items()}


def is_synthetic_name(clip_id, duration, tolerance=1e-3):
    m = PAT.search(clip_id)
    if not m:
        return False
    span = (int(m.group(2)) - int(m.group(1))) / 1000.0
    return abs(span - duration) > tolerance


def test_nat_clips():
    d = test_durations()
    return sorted(c for c, v in d.items() if not is_synthetic_name(c, v))


def test_syn_clips():
    d = test_durations()
    return sorted(c for c, v in d.items() if is_synthetic_name(c, v))


# ------------------------------------------------------------------ library

_EX = None


def exemplars():
    global _EX
    if _EX is None:
        _EX = load_pickle(TEMPL / 'exemplars.pkl')
    return _EX


def theta():
    return np.array(json.loads((WMF / 'theta.json').read_text())['theta'])


def template_extent(k, on, clip_dur):
    """the paste / copy extent of template k starting at `on` (master files: the full master length,
    cut at the clip end), as wmf w06_rules.extent."""
    EX = exemplars()
    m = MASTER_OF.get(k)
    L = MASTER_LEN[m] if m is not None else EX[k]['dur']
    return on, min(on + L, clip_dur)


# ------------------------------------------------------------------ audio stores

def store_paths(name):
    local = CACHE / f'{name}.i16'
    if local.exists():
        return local, CACHE / f'{name}_index.json'
    return WMF_CACHE / f'{name}.i16', WMF_CACHE / f'{name}_index.json'


def store_index(name):
    return json.loads(store_paths(name)[1].read_text())


def read_clip(name, clip, index=None):
    dat, _ = store_paths(name)
    index = index or store_index(name)
    off, n = index[clip]
    x = np.fromfile(dat, dtype=np.int16, count=n, offset=2 * off)
    return x.astype(np.float64) / 32768.0


def build_store(name, items):
    """items: iterable of (clip_id, loader); writes cache/<name>.i16 + index (this folder only)."""
    dat = CACHE / f'{name}.i16'
    idx = CACHE / f'{name}_index.json'
    if dat.exists() and idx.exists():
        return json.loads(idx.read_text())
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


# ------------------------------------------------------------------ scans

def load_scan_dir(d):
    out = {}
    for f in sorted(Path(d).glob('chunk_*.npz')):
        z = np.load(f, allow_pickle=False)
        for j, c in enumerate(z['clips']):
            out[str(c)] = z[f'r{j}']
    return out


def load_wmf_scan(store):
    return load_scan_dir(WMF_CACHE / f'scan_{store}')


# ------------------------------------------------------------------ copy decode (Q1)

def copy_events(rows, clip_dur, th, extra_min=None):
    """Copies of library templates in one natural clip: scan rows with S3 >= theta_k, strongest
    first, template extent (master files: full master length, cut at the clip end), greedy
    non-overlapping. Returns [(on, off, k, S3, S0, p)] sorted by onset. No label is read."""
    if rows is None or not len(rows):
        return []
    k = rows[:, F['k']].astype(int)
    s3 = rows[:, F['S3']]
    thr = th[k] if extra_min is None else np.maximum(th[k], extra_min)
    R = rows[s3 >= thr]
    R = R[np.argsort(-R[:, F['S3']], kind='stable')]
    acc = []
    for r in R:
        kk = int(r[F['k']])
        p = int(r[F['p']])
        on, off = template_extent(kk, p / SR, clip_dur)
        if off - on <= 0:
            continue
        if all(off <= a or on >= b for a, b, *_ in acc):
            acc.append((on, off, kk, float(r[F['S3']]), float(r[F['S0']]), p))
    return sorted(acc)


# ------------------------------------------------------------------ injection rules (Q1)

def overlaps(a, b):
    return a[0] < b[1] and a[1] > b[0]


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def rule_override(pred, copies):
    """R1 (the primary rule; the shipped pipeline's template override on natural clips,
    build_submission.combine, extended to the S3 copies): every predicted event that overlaps a
    copy is dropped, the copy extent is added."""
    cps = [(a, b) for a, b, *_ in copies]
    kept = [e for e in pred if not any(overlaps(e, c) for c in cps)]
    return sorted(kept + cps)


def rule_replace_one(pred, copies):
    """R2 (secondary, reported only): the predicted event with the largest IoU with the copy is
    replaced by the copy extent; other predictions are kept; no overlap -> the copy is added."""
    out = list(pred)
    for c in [(a, b) for a, b, *_ in copies]:
        best, bi = 0.0, None
        for i, e in enumerate(out):
            v = iou(e, c)
            if v > best:
                best, bi = v, i
        if bi is not None:
            out[bi] = c
        else:
            out.append(c)
    return sorted(set(out))


def normalise(events, duration):
    out = [(max(0.0, round(a, 4)), min(duration, round(b, 4))) for a, b in events if b > a]
    out = [(a, b) for a, b in out if b > a]
    for a, b in out:
        assert np.isfinite(a) and np.isfinite(b) and 0.0 <= a < b <= duration, (a, b, duration)
    return out
