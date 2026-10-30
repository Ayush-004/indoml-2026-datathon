"""Stage the unlabelled organizer test natural clips as the Kaggle input dataset "test-natural".

The training kernels use this dataset for two things only: they assert that no test clip is in any training
stream, and at the end of a run they write test predictions and frames for these clips. No test clip is ever a
training row, and no test label exists here.

Source: INDOML_TEST_ZIP (input_data.zip of the organizers: audio/*.wav, 5,517 clips, no labels).
File-name router (2,500/2,501 correct on validation): a clip is NATURAL when
|(end_ms - start_ms)/1000 - wav_duration| <= 1 ms, the milliseconds coming from the file-name suffix
_<start_ms>_<end_ms>.wav. Only the natural clips are staged (3,435 expected). This router only selects the clips
of this dataset; the submitted predictions do not use it.

Writes T1_WORK/test-natural/dataset-test-audio/ with
    dataset-metadata.json         id <owner>/test-natural
    test-natural-manifest.jsonl   clip_id, audio_path (test-audio/<name>.flac), duration
    test-audio.zip                the clips as FLAC, verified sample-exact
and then:
    kaggle datasets create -p dataset-test-audio --dir-mode skip
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(os.environ.get('T1_WORK', '.')) / 'test-natural'
OUT = HERE / 'dataset-test-audio'
STAGE = HERE / 'stage-test-audio'
SOURCE_ZIP = Path(os.path.expanduser(os.environ.get('INDOML_TEST_ZIP', 'input_data.zip')))
TOLERANCE = 0.001


def is_natural(name, duration):
    parts = Path(name).stem.rsplit('_', 2)
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        return None
    span = (int(parts[2]) - int(parts[1])) / 1000.0
    return abs(span - duration) <= TOLERANCE


def convert(item):
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
    return {'ok': True, 'samples': int(data.shape[0])}


def main():
    import soundfile as sf
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    extracted = HERE / 'stage-test-wav'
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith('.wav')]
        print(f'{len(members)} wav members in {SOURCE_ZIP}', flush=True)
        for member in members:
            target = extracted / Path(member).name
            if not target.is_file():
                with archive.open(member) as stream, target.open('wb') as out:
                    out.write(stream.read())

    rows, synthetic, unknown = [], 0, 0
    for path in sorted(extracted.glob('*.wav')):
        info = sf.info(str(path))
        if info.samplerate != 16000:
            raise SystemExit(f'Unexpected sample rate in {path}')
        duration = info.frames / 16000.0
        verdict = is_natural(path.name, duration)
        if verdict is None:
            unknown += 1
            continue
        if not verdict:
            synthetic += 1
            continue
        rows.append({'clip_id': path.name, 'audio_path': 'test-audio/' + path.stem + '.flac',
                     'duration': duration, 'source_wav': str(path)})
    print(f'natural {len(rows)}  synthetic {synthetic}  unrouted {unknown}', flush=True)

    jobs = [(row.pop('source_wav'), str(STAGE / Path(row['audio_path']).name)) for row in rows]
    todo = [job for job in jobs if not Path(job[1]).is_file()]
    results = []
    if todo:
        with mp.Pool(3) as pool:
            for index, result in enumerate(pool.imap_unordered(convert, todo, chunksize=16)):
                results.append(result)
                if (index + 1) % 500 == 0:
                    print(f'  {index + 1}/{len(todo)}', flush=True)
    bad = [r for r in results if not r['ok']]
    if bad:
        raise SystemExit(f'{len(bad)} clips failed the lossless round trip')

    with (OUT / 'test-natural-manifest.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    archive_path = OUT / 'test-audio.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for _, destination in sorted(jobs, key=lambda job: job[1]):
            zf.write(destination, 'test-audio/' + Path(destination).name)
    (OUT / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'IndoML Organizer Test Natural Audio',
        'id': os.environ.get('KAGGLE_DATASET_OWNER', '<owner>') + '/test-natural',
        'licenses': [{'name': 'other'}]}, indent=2) + '\n')
    (OUT / 'test-audio-contract.json').write_text(json.dumps({
        'kind': 'indoml_unlabelled_test_audio_v1',
        'use': 'UNLABELLED audio only: no-overlap check and test predictions. No label, '
               'pseudo-label or organizer annotation of the test set is used or uploaded.',
        'router': 'filename span vs decoded duration, tolerance 1 ms',
        'natural_clips': len(rows), 'synthetic_clips_excluded': synthetic,
        'archive_bytes': archive_path.stat().st_size,
        'verified_sample_exact': len(results)}, indent=2) + '\n')
    print(f'staged {len(rows)} natural test clips, archive '
          f'{archive_path.stat().st_size / 2**20:.0f} MiB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
