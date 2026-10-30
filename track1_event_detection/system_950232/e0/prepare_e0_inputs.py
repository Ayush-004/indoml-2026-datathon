"""Package the existing WavLM head/source and lossless official test audio.

This builds the input dataset of full_wavlm_job.py (E0 stage 2): bundle/ (source/, initial-head.pt,
initial-selection.json, test-inference.jsonl, full-wavlm-contract.json) and input-upload/ (test-audio.zip,
full-wavlm-code.zip, dataset-metadata.json). The code is the original packaging script; only the input locations
are arguments here:

  --out            output folder (bundle/, input-upload/, input-preparation.json, test-conversion-ledger.jsonl)
  --head           the E0 stage-1 head folder: <frozen_wavlm_red.py --output>/head-seed73 (selected.pt of epoch 5)
  --test-audit     folder with the organizer test manifests: inference.jsonl (clip_id, audio_path, duration of the
                   5,517 test clips), audio-inventory.jsonl (clip_id, sha256) and input-audit.json (archive_sha256)
  T1_WORK          holds full-input-staging/input-upload (stage_full_inputs.py) and full-release-audit
                   (audit_full_release.py)

  python prepare_e0_inputs.py --out e0-inputs --head e0-stage1/head-seed73 --test-audit organizer-audit/test
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = Path(os.environ.get('T1_WORK', '.'))
sys.path.insert(0, str(REPO / 'data_prep' / 'vaani'))
from lossless_audio import encode


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(OUT, head, TEST):
    bundle, upload = OUT / 'bundle', OUT / 'input-upload'
    source = bundle / 'source'
    source.mkdir(parents=True, exist_ok=True)
    upload.mkdir(exist_ok=True)
    shutil.copytree(REPO / 'atst_red' / 'indoml',
                    source / 'indoml', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('instance_pilot.py', 'wavlm_adapter.py'):
        shutil.copy2(HERE / name, source / name)
    shutil.copy2(REPO / 'training/p2/input_paths.py', source / 'input_paths.py')
    shutil.copy2(REPO / 'atst_red/mixed_training.py', source / 'mixed_training.py')
    shutil.copy2(head / 'selected.pt', bundle / 'initial-head.pt')
    shutil.copy2(head / 'selection.json', bundle / 'initial-selection.json')
    selection = json.loads((bundle / 'initial-selection.json').read_text())
    assert selection['epoch'] == 5

    inference = read(TEST / 'inference.jsonl')
    inventory = {row['clip_id']: row for row in read(TEST / 'audio-inventory.jsonl')}
    assert len(inference) == len(inventory) == 5517
    assert all(set(row) == {'clip_id', 'audio_path', 'duration'} for row in inference)
    for row in inference:
        row['audio_sha256'] = inventory[row['clip_id']]['sha256']
    full_input = WORK / 'full-input-staging/input-upload'
    training_ids = {row['clip_id'] for row in read(full_input / 'training.jsonl')}
    organizer_ids = {row['clip_id'] for row in read(full_input / 'organizer-inference.jsonl')}
    assert len(training_ids) == 90635 and len(organizer_ids) == 2501
    assert not training_ids & organizer_ids
    assert not set(inventory) & (training_ids | organizer_ids)

    test_rows, ledger = [], []
    archive = upload / 'test-audio.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as packed:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for index, (row, encoded) in enumerate(zip(inference, pool.map(encode, inference), strict=True), 1):
                payload, extension, evidence = encoded
                name = 'test/audio/' + hashlib.sha256(row['clip_id'].encode()).hexdigest()[:24] + '.' + extension
                packed.writestr(name, payload)
                test_rows.append({'clip_id': row['clip_id'], 'audio_path': name, 'duration': row['duration']})
                ledger.append({'clip_id': row['clip_id'], **evidence})
                if index % 1000 == 0 or index == len(inference):
                    print(json.dumps({'stage': 'test_packaging', 'completed': index, 'total': len(inference)}), flush=True)
    (bundle / 'test-inference.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in test_rows))
    (OUT / 'test-conversion-ledger.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in ledger))
    assert all(row['waveform_bit_exact'] for row in ledger)
    contract = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'objective': 'Full released Vaani training, organizer model selection, official test predictions',
        'training_clips': 90635, 'strong_gold': 11095, 'weak_clips': 79540,
        'weak_gold': 16, 'weak_silver': 61640, 'weak_bronze': 17884,
        'organizer_clips': 2501, 'test_clips': 5517, 'custom_holdout': False,
        'weak_passes': 8, 'strong_batch': 16, 'weak_batch': 32, 'max_windows': 48,
        'weak_weight': .1, 'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip': 5.,
        'seed': 73, 'initial_selected_epoch': selection['epoch'],
        'initial_checkpoint_sha256': sha(bundle / 'initial-head.pt'),
        'optimizer': 'fresh AdamW for the corrected full-data objective',
        'selection': 'Earliest strict maximum organizer F1+Dice after a complete weak pass; all training clips must have been seen',
        'encoder_compute': 'float32 fixed uniform mean of all 25 frozen WavLM Large hidden states',
        'cache_storage': 'float32 when actual scratch space permits; float16 otherwise, expanded to float32 by the existing loader',
        'full_input_contract_sha256': sha(full_input / 'full-input-contract.json'),
        'training_manifest_sha256': sha(full_input / 'training.jsonl'),
        'organizer_inference_sha256': sha(full_input / 'organizer-inference.jsonl'),
        'organizer_references_sha256': sha(full_input / 'organizer-references.jsonl'),
        'source_training_audit_sha256': sha(WORK / 'full-release-audit/training-audit.jsonl'),
        'source_test_manifest_sha256': sha(TEST / 'inference.jsonl'),
        'source_test_archive_sha256': json.loads((TEST / 'input-audit.json').read_text())['archive_sha256'],
        'test_audio_archive_sha256': sha(archive), 'test_audio_archive_bytes': archive.stat().st_size,
        'test_waveforms_bit_exact': True, 'test_labels_present': False,
        'source_files': {str(p.relative_to(bundle)): sha(p) for p in sorted(source.rglob('*.py'))},
    }
    write(bundle / 'full-wavlm-contract.json', contract)
    with zipfile.ZipFile(upload / 'full-wavlm-code.zip', 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as packed:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                packed.write(path, path.relative_to(bundle))
    write(upload / 'dataset-metadata.json', {
        'title': 'IndoML WavLM Full Training Inputs',
        'id': os.environ.get('KAGGLE_DATASET_OWNER', '<owner>') + '/wavlm-full-inputs',
        'licenses': [{'name': 'other'}],
        'description': 'Private source, saved head initialization and lossless Codabench test audio for the full Vaani training and prediction run. Training and organizer validation audio reuse the existing private full input dataset. Original authors retain applicable rights. No public redistribution or test labels.',
    })
    write(OUT / 'input-preparation.json', {'status': 'complete', **contract,
        'code_archive_bytes': (upload / 'full-wavlm-code.zip').stat().st_size,
        'code_archive_sha256': sha(upload / 'full-wavlm-code.zip')})
    print(json.dumps({'stage': 'input_ready', 'test_clips': len(test_rows), 'test_audio_bytes': archive.stat().st_size,
                      'code_bytes': (upload / 'full-wavlm-code.zip').stat().st_size}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--test-audit', type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve(), args.head.resolve(), args.test_audit.resolve())
