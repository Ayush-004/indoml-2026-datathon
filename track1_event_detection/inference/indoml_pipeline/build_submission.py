"""Assemble the final per-clip events and write the submission.

Per clip:

  1. ROUTE on the filename and the wav duration (router.py). No label, no model.
  2. SYNTHETIC-routed -> template decode at the fitted per-template thresholds
     (templates.py, tau_syn).
  3. NATURAL-routed  -> the fused P2 stack for that clip. If the template scan also
     fires on it at NCC >= 0.9 (a synthetic-style noise pasted into a natural clip,
     or a clip the router called natural), those template events are kept and every
     overlapping P2 event is dropped.
  4. Clamp to [0, duration], round to 0.1 ms, drop empty intervals.

Output: predictions.jsonl (clip_id, events[{onset, offset}]) and, with --zip, the
submission archive holding that file at the archive root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from . import common, router, templates
from .common import TEMPLATE_OVERRIDE_NCC


def durations_for(split, work=None, validation_zip=None, test_zip=None):
    """clip_id -> wav duration, cached as json under work/."""
    work = Path(work or common.WORK)
    path = work / f'{split}-durations.json'
    if path.exists():
        return {k: float(v) for k, v in json.loads(path.read_text()).items()}
    if split == 'val':
        records = common.validation_records()
        names = {r['segmentFileName']: common.validation_zip_name(r) for r in records}
        raw = common.durations_from_zip(validation_zip or common.VALIDATION_ZIP, names.values())
        out = {clip: raw[name] for clip, name in names.items()}
    else:
        zip_path = test_zip or common.TEST_ZIP
        names = common.zip_wav_names(zip_path, common.TEST_PREFIX)
        raw = common.durations_from_zip(zip_path, names)
        out = {Path(name).name: raw[name] for name in names}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def hits_by_clip(split, work=None, **kwargs):
    """The raw template hits, keyed by clip id rather than by zip member name."""
    hits = templates.ensure_hits(split, work, **kwargs)
    if split == 'val':
        return {r['segmentFileName']: hits[common.validation_zip_name(r)]
                for r in common.validation_records()}
    return {Path(name).name: value for name, value in hits.items()}


def combine(clip_id, duration, hits, natural_events, tau_syn, exemplars,
            synthetic=None, override_ncc=TEMPLATE_OVERRIDE_NCC):
    """Return (events, route) for one clip. `synthetic` overrides the filename router."""
    is_synthetic = (router.is_synthetic(clip_id, duration) if synthetic is None else synthetic)
    if is_synthetic:
        return templates.decode(hits, tau_syn, exemplars), 'syn'
    strong = templates.decode(hits, np.full(len(exemplars), override_ncc), exemplars)
    if strong:
        kept = [(a, b) for a, b in natural_events if all(b <= x or a >= y for x, y in strong)]
        return sorted(strong + kept), 'nat+tmpl'
    return natural_events, 'nat'


def build(split, natural_predictions, *, work=None, validation_zip=None, test_zip=None,
          route='filename', workers=None):
    """Return (rows, audit). `route='label'` uses the organizer syntheticData flag (validation only)."""
    work = Path(work or common.WORK)
    exemplars = templates.ensure_exemplars(work, validation_zip)
    tau = templates.ensure_thresholds(work, validation_zip, workers=workers)
    tau_syn = tau['tau_syn']
    durations = durations_for(split, work, validation_zip, test_zip)
    hits = hits_by_clip(split, work, validation_zip=validation_zip, test_zip=test_zip,
                        workers=workers)
    natural = common.events_of(natural_predictions)

    labels = None
    if route == 'label':
        if split != 'val':
            raise SystemExit('--route label needs organizer labels and so works only on validation')
        labels = {r['segmentFileName']: bool(r['syntheticData'])
                  for r in common.validation_records()}

    rows, routes, uncovered = [], {'syn': 0, 'nat+tmpl': 0, 'nat': 0}, []
    for clip in sorted(durations):
        duration = durations[clip]
        forced = labels[clip] if labels else None
        source = natural.get(clip)
        events, which = combine(clip, duration, hits[clip], source or [], tau_syn, exemplars,
                                synthetic=forced)
        if which != 'syn' and source is None:
            uncovered.append(clip)
        routes[which] += 1
        events = [(max(0.0, round(a, 4)), min(duration, round(b, 4)))
                  for a, b in events if b > a]
        for a, b in events:
            assert np.isfinite(a) and np.isfinite(b) and 0.0 <= a and b <= duration and b > a, \
                f'invalid event on {clip}: {(a, b)}'
        rows.append({'clip_id': clip, 'events': [{'onset': a, 'offset': b} for a, b in events]})
    audit = {'split': split, 'route': route, 'clips': len(rows), 'routes': routes,
             'total_events': sum(len(r['events']) for r in rows),
             'empty_clips': sum(1 for r in rows if not r['events']),
             'natural_routed_clips_without_p2_events': len(uncovered),
             'natural_source': str(natural_predictions)}
    return rows, audit


def submission_id(clip_id):
    """The organizer scorer keys clips by the audio file name WITHOUT the '.wav' suffix.

    Submission 927540 (ids with '.wav') scored 0.00 because the scorer silently drops
    unknown ids; the same predictions with the suffix removed (927559) scored 1.5412.
    Internally the pipeline keeps the file name with '.wav'; only the written file differs.
    """
    return clip_id[:-4] if clip_id.endswith('.wav') else clip_id


def write_submission(rows, jsonl_path, zip_path=None):
    payload = ''.join(json.dumps({'clip_id': submission_id(row['clip_id']),
                                  'events': row['events']}) + '\n' for row in rows).encode()
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_bytes(payload)
    out = {'predictions': str(jsonl_path),
           'predictions_sha256': hashlib.sha256(payload).hexdigest()}
    if zip_path:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('predictions.jsonl', payload)
        out['zip'] = str(zip_path)
        out['zip_sha256'] = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        out['zip_bytes'] = zip_path.stat().st_size
    return out


def check_format(rows, durations, expected_clips=None):
    """The organizer format checks: unique ids, finite times inside [0, duration]."""
    problems = []
    if expected_clips is not None and len(rows) != expected_clips:
        problems.append(f'{len(rows)} clips, expected {expected_clips}')
    ids = [r['clip_id'] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append('duplicate clip ids')
    for row in rows:
        duration = durations[row['clip_id']]
        for event in row['events']:
            a, b = event['onset'], event['offset']
            if not (np.isfinite(a) and np.isfinite(b)):
                problems.append(f'{row["clip_id"]}: non-finite time')
            elif not (0.0 <= a < b <= duration + 1e-9):
                problems.append(f'{row["clip_id"]}: {a}..{b} outside [0, {duration}]')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--natural', type=Path, required=True,
                        help='fused P2-stack predictions for the natural-routed clips')
    parser.add_argument('--out', type=Path, required=True, help='predictions.jsonl to write')
    parser.add_argument('--zip', type=Path, default=None, help='submission archive to write')
    parser.add_argument('--work', type=Path, default=None)
    parser.add_argument('--route', choices=('filename', 'label'), default='filename')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--score', action='store_true', help='score against the organizer labels')
    parser.add_argument('--compare', type=Path, default=None,
                        help='an existing submission zip or jsonl to diff against')
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args(argv)

    rows, audit = build(args.split, args.natural, work=args.work, route=args.route,
                        workers=args.workers)
    durations = durations_for(args.split, args.work)
    problems = check_format(rows, durations, 5517 if args.split == 'test' else 2501)
    audit['format_problems'] = problems
    audit.update(write_submission(rows, args.out, args.zip))

    if args.score:
        from .score import validation_report
        predictions = {r['clip_id']: [(e['onset'], e['offset']) for e in r['events']]
                       for r in rows}
        audit['validation'] = validation_report(predictions)

    if args.compare:
        other = args.compare
        if other.suffix == '.zip':
            with zipfile.ZipFile(other) as archive:
                lines = archive.read('predictions.jsonl').decode().splitlines()
            reference = {json.loads(l)['clip_id']: [(e['onset'], e['offset'])
                                                    for e in json.loads(l)['events']]
                         for l in lines if l.strip()}
        else:
            reference = common.events_of(other)
        # old archives (sub01-sub04) carry '.wav' ids, new ones do not: compare by stem
        reference = {submission_id(k): v for k, v in reference.items()}
        worst, mismatches, missing = 0.0, [], []
        for row in rows:
            want = reference.get(submission_id(row['clip_id']))
            if want is None:
                missing.append(row['clip_id'])
                continue
            got = [(e['onset'], e['offset']) for e in row['events']]
            if len(got) != len(want):
                mismatches.append(row['clip_id'])
                continue
            for (a, b), (c, d) in zip(got, want):
                worst = max(worst, abs(a - c), abs(b - d))
        audit['compare'] = {'reference': str(other), 'clips_missing': len(missing),
                            'event_count_mismatches': len(mismatches),
                            'max_abs_endpoint_error_seconds': worst,
                            'identical_within_1e-6': not missing and not mismatches
                            and worst <= 1e-6}
    print(json.dumps(audit, indent=1))
    if args.json_out:
        args.json_out.write_text(json.dumps(audit, indent=1) + '\n')
    return audit


if __name__ == '__main__':
    main()
