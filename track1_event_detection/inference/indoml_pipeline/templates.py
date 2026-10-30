"""Matched-filter detector for the synthetic clips.

The organizer built every synthetic clip by pasting a noise waveform into a clean
speech segment, and the same few hundred waveforms are reused across the set. The
validation release ships both the noisy clip and its clean reference, so the exact
noise waveform of every labelled synthetic event is recoverable as
(noisy - clean) cut at the labelled onset/offset.

Stages
  build_library     cut every labelled synthetic event out of the residual and
                    greedily cluster (same tag, duration within 3 ms, NCC >= 0.97)
                    -> ~203 distinct noise waveforms
  build_exemplars   keep up to two exemplars per template, each from a different
                    clip, L2-normalised (two so that a scan of a validation clip can
                    use an exemplar taken from a DIFFERENT clip: leave-one-clip-out)
  scan              normalised cross-correlation of every exemplar against every
                    clip (scipy.signal.correlate 'valid' with a running-energy
                    normaliser), peak-picked -> raw hits (template, sample, ncc)
  fit_thresholds    per-template NCC threshold, fitted on the validation hits
  decode            greedy non-overlapping decode of the hits above threshold

Nothing here reads a label at inference time: `decode` sees only the hits.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
from scipy.signal import correlate, find_peaks

from . import common
from .common import SR

CLUSTER_NCC = 0.97
CLUSTER_DURATION_TOLERANCE = 0.003
CLUSTER_LAG_SAMPLES = 320          # +/- 20 ms
PEAK_HEIGHT = 0.2
HIT_TOLERANCE = 0.02               # a raw hit counts as true within 20 ms at both ends
THRESHOLD_GRID = np.round(np.arange(0.2, 0.96, 0.025), 3)
THRESHOLD_FLOOR = 0.3
NATURAL_MARGIN = 0.03
NATURAL_MINIMUM = 0.25


# ------------------------------------------------------------------ library

def _residual_events(validation_zip, records):
    """Every labelled synthetic event as a residual (noisy - clean) waveform."""
    events = []
    with zipfile.ZipFile(validation_zip) as archive:
        import soundfile as sf
        for record in records:
            name = record['segmentFileName']
            noisy, _ = sf.read(io.BytesIO(archive.read(common.VAL_SYNTHETIC_PREFIX + name)),
                               dtype='float32')
            clean, _ = sf.read(io.BytesIO(archive.read(common.VAL_CLEAN_PREFIX + name)),
                               dtype='float32')
            residual = noisy - clean
            for event in record['NoiseSubCategoryTimeStamp']:
                start, end = float(event['start']), float(event['end'])
                segment = residual[int(round(start * SR)):int(round(end * SR))]
                events.append({'tag': event['tag'], 'dur': end - start, 'seg': segment,
                               'clip': name, 'start': start})
    return events


def _ncc(a, b):
    """Peak normalised cross-correlation of two segments over +/- 20 ms of lag."""
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    full = correlate(a, b, mode='full', method='fft')
    middle = length - 1
    window = full[max(0, middle - CLUSTER_LAG_SAMPLES):middle + CLUSTER_LAG_SAMPLES + 1]
    return float(np.max(np.abs(window)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_library(validation_zip=None, records=None):
    """Greedy clustering of the residual events into distinct noise waveforms."""
    validation_zip = validation_zip or common.VALIDATION_ZIP
    records = [r for r in (records or common.validation_records()) if r['syntheticData']]
    events = _residual_events(validation_zip, records)
    templates = []
    for event in sorted(events, key=lambda e: (e['tag'], e['dur'])):
        match = None
        for template in templates:
            if template['tag'] != event['tag']:
                continue
            if abs(template['dur'] - event['dur']) > CLUSTER_DURATION_TOLERANCE:
                continue
            if _ncc(event['seg'], template['seg']) >= CLUSTER_NCC:
                match = template
                break
        if match is None:
            templates.append({'tag': event['tag'], 'dur': event['dur'], 'seg': event['seg'],
                              'members': [event]})
        else:
            match['members'].append(event)
    return [{'tag': t['tag'], 'dur': t['dur'], 'seg': t['seg'], 'n': len(t['members']),
             'clips': [m['clip'] for m in t['members']]} for t in templates]


# ------------------------------------------------------------------ exemplars

def _residual_segment(archive, name, start, end):
    import soundfile as sf
    noisy, _ = sf.read(io.BytesIO(archive.read(common.VAL_SYNTHETIC_PREFIX + name)), dtype='float64')
    clean, _ = sf.read(io.BytesIO(archive.read(common.VAL_CLEAN_PREFIX + name)), dtype='float64')
    return (noisy - clean)[int(round(start * SR)):int(round(end * SR))]


def build_exemplars(library, validation_zip=None, records=None):
    """Up to two L2-normalised exemplars per template, each from a different clip."""
    validation_zip = validation_zip or common.VALIDATION_ZIP
    records = records or common.validation_records()
    by_clip = {r['segmentFileName']: r['NoiseSubCategoryTimeStamp']
               for r in records if r['syntheticData']}
    out = []
    with zipfile.ZipFile(validation_zip) as archive:
        for template in library:
            clips = []
            for clip in template['clips']:
                if clip not in clips:
                    clips.append(clip)
            exemplars = []
            for clip in clips[:2]:
                best = None
                for event in by_clip[clip]:
                    start, end = float(event['start']), float(event['end'])
                    if event['tag'] != template['tag']:
                        continue
                    if abs((end - start) - template['dur']) > CLUSTER_DURATION_TOLERANCE:
                        continue
                    segment = _residual_segment(archive, clip, start, end)
                    length = min(len(segment), len(template['seg']))
                    value = (abs(np.dot(segment[:length], template['seg'][:length]))
                             / (np.linalg.norm(segment[:length])
                                * np.linalg.norm(template['seg'][:length]) + 1e-12))
                    if best is None or value > best[0]:
                        best = (value, segment)
                exemplars.append((clip, best[1] / np.linalg.norm(best[1])))
            out.append({'tag': template['tag'], 'dur': template['dur'], 'exs': exemplars})
    return out


# ------------------------------------------------------------------ scan

_EXEMPLARS = None


def _worker_init(exemplars_path):
    global _EXEMPLARS
    with open(exemplars_path, 'rb') as stream:
        _EXEMPLARS = pickle.load(stream)


def _scan_one(task):
    """(zip path, member name, clip id) -> (member name, [(template, sample, ncc)])."""
    import soundfile as sf
    zip_path, name, clip = task
    with zipfile.ZipFile(zip_path) as archive:
        signal, _ = sf.read(io.BytesIO(archive.read(name)), dtype='float64')
    energy = np.concatenate([[0.0], np.cumsum(signal * signal)])
    hits = []
    for index, template in enumerate(_EXEMPLARS):
        exemplars = [e for e in template['exs'] if e[0] != clip]   # leave-one-clip-out
        if not exemplars:
            continue
        kernel = exemplars[0][1]
        length = len(kernel)
        if len(signal) < length:
            continue
        window_energy = np.maximum(energy[length:] - energy[:-length], 1e-12)
        ncc = np.abs(correlate(signal, kernel, mode='valid', method='fft')) / np.sqrt(window_energy)
        peaks, _ = find_peaks(ncc, height=PEAK_HEIGHT, distance=max(1, length // 2))
        for peak in peaks:
            hits.append((index, int(peak), float(ncc[peak])))
    return name, hits


def scan(tasks, exemplars_file, workers=None):
    """Run the matched filter over `tasks` = [(zip, member, clip_id)] with a process pool."""
    from multiprocessing import get_context
    workers = workers or common.threads()
    if workers <= 1:
        _worker_init(exemplars_file)
        return dict(_scan_one(task) for task in tasks)
    context = get_context('spawn')
    with context.Pool(workers, initializer=_worker_init, initargs=(str(exemplars_file),)) as pool:
        return dict(pool.map(_scan_one, tasks, chunksize=4))


def validation_tasks(validation_zip=None, records=None):
    validation_zip = str(validation_zip or common.VALIDATION_ZIP)
    records = records or common.validation_records()
    return [(validation_zip, common.validation_zip_name(r), r['segmentFileName']) for r in records]


def test_tasks(test_zip=None):
    test_zip = str(test_zip or common.TEST_ZIP)
    return [(test_zip, name, Path(name).name)
            for name in common.zip_wav_names(test_zip, common.TEST_PREFIX)]


# ------------------------------------------------------------------ thresholds

def fit_thresholds(hits_by_clip, exemplars, records=None):
    """Per-template NCC thresholds, fitted on the validation hits only.

    tau_nat  the level above which a template never fires on a NATURAL validation
             clip (its highest natural hit + 0.03, floored at 0.25); this is the
             conservative threshold used when a natural-routed clip is scanned.
    tau_syn  for every template with positive evidence on the synthetic clips, the
             grid threshold maximising (true hits - false hits), capped by tau_nat
             and floored at 0.3; otherwise tau_nat.
    """
    records = records or common.validation_records()
    reference = {r['segmentFileName']: [(float(e['start']), float(e['end']))
                                        for e in r['NoiseSubCategoryTimeStamp']] for r in records}
    synthetic = {r['segmentFileName']: bool(r['syntheticData']) for r in records}
    natural_clips = [c for c in reference if not synthetic[c]]
    synthetic_clips = sorted(c for c in reference if synthetic[c])
    count = len(exemplars)

    highest = np.zeros(count)
    for clip in natural_clips:
        for index, _, value in hits_by_clip[clip]:
            highest[index] = max(highest[index], value)
    tau_nat = np.maximum(highest + NATURAL_MARGIN, NATURAL_MINIMUM)

    labels = {k: [] for k in range(count)}
    for clip in synthetic_clips:
        for index, position, value in hits_by_clip[clip]:
            onset = position / SR
            offset = onset + exemplars[index]['dur']
            hit = any(abs(onset - a) <= HIT_TOLERANCE and abs(offset - b) <= HIT_TOLERANCE
                      for a, b in reference[clip])
            labels[index].append((value, hit))

    tau_syn = np.array(tau_nat)
    for index in range(count):
        values = labels[index]
        if not any(hit for _, hit in values):
            continue
        best = None
        for threshold in THRESHOLD_GRID:
            true_hits = sum(1 for v, hit in values if hit and v >= threshold)
            false_hits = sum(1 for v, hit in values if (not hit) and v >= threshold)
            margin = true_hits - false_hits
            if best is None or margin > best[0] or (margin == best[0] and threshold > best[1]):
                best = (margin, threshold)
        tau_syn[index] = max(min(best[1], tau_nat[index]), THRESHOLD_FLOOR)
    return {'tau_nat': tau_nat, 'tau_syn': tau_syn}


# ------------------------------------------------------------------ decode

def decode(hits, tau, exemplars):
    """Greedy non-overlapping decode: strongest NCC first, drop anything that overlaps."""
    candidates = sorted(((value, position / SR, position / SR + exemplars[index]['dur'], index)
                         for index, position, value in hits if value >= tau[index]), reverse=True)
    accepted = []
    for value, onset, offset, index in candidates:
        if all(offset <= a or onset >= b for _, a, b, _ in accepted):
            accepted.append((value, onset, offset, index))
    return sorted((onset, offset) for _, onset, offset, _ in accepted)


# ------------------------------------------------------------------ artefact cache

def library_path(work=None):
    return Path(work or common.WORK) / 'templates' / 'library.pkl'


def exemplars_path(work=None):
    return Path(work or common.WORK) / 'templates' / 'exemplars.pkl'


def hits_path(split, work=None):
    return Path(work or common.WORK) / 'templates' / f'hits_{split}.pkl'


def thresholds_path(work=None):
    return Path(work or common.WORK) / 'templates' / 'tau_router.pkl'


def load_pickle(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def dump_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as stream:
        pickle.dump(payload, stream)


def ensure_library(work=None, validation_zip=None):
    path = library_path(work)
    if not path.exists():
        dump_pickle(path, build_library(validation_zip))
    return load_pickle(path)


def ensure_exemplars(work=None, validation_zip=None):
    path = exemplars_path(work)
    if not path.exists():
        dump_pickle(path, build_exemplars(ensure_library(work, validation_zip), validation_zip))
    return load_pickle(path)


def ensure_hits(split, work=None, validation_zip=None, test_zip=None, workers=None):
    path = hits_path(split, work)
    if not path.exists():
        ensure_exemplars(work, validation_zip)
        tasks = validation_tasks(validation_zip) if split == 'val' else test_tasks(test_zip)
        dump_pickle(path, scan(tasks, exemplars_path(work), workers))
    return load_pickle(path)


def ensure_thresholds(work=None, validation_zip=None, workers=None):
    path = thresholds_path(work)
    if not path.exists():
        hits = ensure_hits('val', work, validation_zip, workers=workers)
        exemplars = ensure_exemplars(work, validation_zip)
        by_clip = {r['segmentFileName']: hits[common.validation_zip_name(r)]
                   for r in common.validation_records()}
        dump_pickle(path, fit_thresholds(by_clip, exemplars))
    return load_pickle(path)


# ------------------------------------------------------------------ CLI

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('library', 'exemplars', 'scan', 'thresholds', 'all'))
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--work', default=None)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args(argv)
    work = Path(args.work) if args.work else common.WORK

    if args.stage in ('library', 'all'):
        library = ensure_library(work)
        print(f'library: {len(library)} templates -> {library_path(work)}', flush=True)
    if args.stage in ('exemplars', 'all'):
        exemplars = ensure_exemplars(work)
        print(f'exemplars: {len(exemplars)} -> {exemplars_path(work)}', flush=True)
    if args.stage in ('scan', 'all'):
        hits = ensure_hits(args.split, work, workers=args.workers)
        print(f'hits[{args.split}]: {len(hits)} clips, '
              f'{sum(len(v) for v in hits.values())} raw peaks -> {hits_path(args.split, work)}',
              flush=True)
    if args.stage in ('thresholds', 'all'):
        tau = ensure_thresholds(work, workers=args.workers)
        lowered = int(np.sum(tau['tau_syn'] < tau['tau_nat'] - 1e-9))
        print(f'thresholds: {len(tau["tau_syn"])} templates, {lowered} lowered below the '
              f'natural-derived level -> {thresholds_path(work)}', flush=True)


if __name__ == '__main__':
    main()
