"""Build the Kaggle input dataset "silver-pools": the Silver timestamps and the generated-clip provenance.

The "vaani-inputs" dataset (stage_full_inputs.py) carries the Silver AUDIO (61,640 clips,
`quality == 'unverified_timestamps'`) but its manifest sets `events: null` for every weak row, so the Silver
timestamps are not in it. They are in T1_WORK/vaani-full/manifest.jsonl (prepare_full.py). This script exports just
the labels - no audio - as a small dataset that the kernels join to the existing rows by clip_id:

    silver-events.jsonl   {clip_id, duration, events[], positive_categories[]}
    gen-sessions.json     {generated clip_id: [session tokens of its bed + pieces]}
    silver-contract.json  counts, digests and the exact filters applied

Inputs: T1_WORK, T1_GOLD_SYNTH_MANIFEST (manifest.jsonl of gold_synth.py), T1_VAL_META (validationMetadata.json).
Only T1_WORK/silver-pools/dataset-silver/ is written.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

WORK = Path(os.environ.get('T1_WORK', '.'))
HERE = WORK / 'silver-pools'
OUT = HERE / 'dataset-silver'
VAANI = WORK / 'vaani-full/manifest.jsonl'
KAGGLE_TRAINING = WORK / 'full-input-staging/input-upload/training.jsonl'
GEN_MANIFEST = WORK / 'gold-pools/dataset/gen-manifest.jsonl'
SYNTH_SOURCE = Path(os.environ.get('T1_GOLD_SYNTH_MANIFEST', 'gold-synth/manifest.jsonl'))
VAL_META = Path(os.environ.get('T1_VAL_META', 'validationMetadata.json'))
CATEGORIES = ('animal', 'vehicle_traffic', 'baby_child', 'singing_music',
              'phone_signal_alarm', 'appliance_machine', 'human_non_speech')
SESSION_FIELD = 5          # the 6th underscore-separated field of a Vaani clip id


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def session_token(clip_id):
    parts = str(clip_id).split('_')
    return parts[SESSION_FIELD] if len(parts) > SESSION_FIELD else None


def sha256_of(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    natural = [record['segmentFileName'] for record in json.loads(VAL_META.read_text())
               if record['syntheticData'] is False]
    assert len(natural) == 1806, len(natural)
    natural_sessions = {session_token(clip) for clip in natural}

    kaggle = {row['clip_id']: row for row in read_jsonl(KAGGLE_TRAINING)}
    silver_ids = {clip for clip, row in kaggle.items()
                  if row['quality'] == 'unverified_timestamps'}

    counts = {'kaggle_rows': len(kaggle), 'kaggle_silver_rows': len(silver_ids),
              'natural_sessions': len(natural_sessions)}
    dropped = {}

    def drop(key):
        dropped[key] = dropped.get(key, 0) + 1

    exported, events_total, seconds = [], 0, 0.
    for record in read_jsonl(VAANI):
        if record.get('quality') != 'unverified_timestamps':
            continue
        clip_id = record['clip_id']
        if clip_id not in silver_ids:
            drop('not_in_kaggle_manifest')
            continue
        duration = float(kaggle[clip_id]['duration'])
        events = []
        for event in record.get('events') or []:
            onset, offset = float(event['onset']), min(float(event['offset']), duration)
            if event['category'] not in CATEGORIES:
                drop('unknown_category')
                continue
            if not (0. <= onset < offset):
                drop('degenerate_event')
                continue
            events.append({'onset': onset, 'offset': offset, 'category': event['category'],
                           'tag': event.get('tag') or ''})
        if not events:
            drop('no_usable_event')
            continue
        if abs(duration - float(record['duration'])) > .01:
            drop('duration_mismatch')
            continue
        exported.append({'clip_id': clip_id, 'duration': duration, 'events': events,
                         'positive_categories': sorted({e['category'] for e in events}),
                         'session_in_natural': session_token(clip_id) in natural_sessions})
        events_total += len(events)
        seconds += duration

    exported.sort(key=lambda row: row['clip_id'])
    with (OUT / 'silver-events.jsonl').open('w') as stream:
        for row in exported:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')

    # Generated clips: their bed and donor pieces are real Vaani clips, so they carry
    # session tokens even though the generated clip id does not.
    wanted = {row['clip_id'] for row in read_jsonl(GEN_MANIFEST)}
    sessions = {}
    for record in read_jsonl(SYNTH_SOURCE):
        if record['clip_id'] not in wanted:
            continue
        source = record.get('source') or {}
        ids = [source.get('bed', {}).get('clip_id')]
        ids += [piece.get('clip_id') for piece in (source.get('pieces') or [])]
        tokens = sorted({session_token(clip) for clip in ids if clip})
        sessions[record['clip_id']] = tokens
    (OUT / 'gen-sessions.json').write_text(json.dumps(sessions, sort_keys=True) + '\n')

    hit_silver = sum(1 for row in exported if row['session_in_natural'])
    hit_gen = sum(1 for tokens in sessions.values()
                  if natural_sessions & set(tokens))
    by_tier = {}
    for clip, row in kaggle.items():
        if session_token(clip) in natural_sessions:
            key = f"{row['supervision']}/{row['quality']}"
            by_tier[key] = by_tier.get(key, 0) + 1
    contract = {
        'kind': 'indoml_silver_labels_v1',
        'source': str(VAANI), 'source_sha256': sha256_of(VAANI),
        'joined_to': str(KAGGLE_TRAINING),
        'categories': list(CATEGORIES), 'session_field': SESSION_FIELD,
        'counts': dict(counts, exported_clips=len(exported), exported_events=events_total,
                       exported_hours=round(seconds / 3600, 2),
                       mean_duration=round(seconds / max(1, len(exported)), 3),
                       events_per_clip=round(events_total / max(1, len(exported)), 3),
                       generated_clips=len(sessions)),
        'dropped': dropped,
        'session_overlap_with_natural': {
            'kaggle_rows_by_tier': by_tier, 'kaggle_rows_total': sum(by_tier.values()),
            'silver_export_rows': hit_silver, 'generated_clips': hit_gen,
            'note': 'the kernel drops these rows itself and logs the same counts'},
        'files': {'silver-events.jsonl': sha256_of(OUT / 'silver-events.jsonl'),
                  'gen-sessions.json': sha256_of(OUT / 'gen-sessions.json')},
    }
    (OUT / 'silver-contract.json').write_text(json.dumps(contract, indent=1) + '\n')
    (OUT / 'dataset-metadata.json').write_text(json.dumps(
        {'title': 'IndoML Silver Timestamps',
         'id': os.environ.get('KAGGLE_DATASET_OWNER', '<owner>') + '/silver-pools',
         'licenses': [{'name': 'other'}]}, indent=2) + '\n')
    print(json.dumps(contract, indent=1))


if __name__ == '__main__':
    main()
