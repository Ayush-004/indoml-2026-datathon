"""Predict the official test set from a validated, full-coverage v2 selected head."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
OUTPUT = Path('/kaggle/working/wavlm-predict-only')
SCRATCH = Path('/kaggle/temp/wavlm-predict-only')
PARENT_RUN_ID = 349143477
PARENT_SOURCE_SHA256 = 'c553ebc401caf158e315f2788f067cf67c9722fa7ad0a1e2a2a234bef58cb07e'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def emit(stage, **values):
    record = {'stage': stage, 'utc': now(), **values}
    print(json.dumps(record, allow_nan=False), flush=True)
    if OUTPUT.is_dir():
        write(OUTPUT / 'progress.json', record)


def only_input(name):
    paths = list(Path('/kaggle/input').rglob(name))
    assert len(paths) == 1, f'Expected one {name}, found {len(paths)}'
    return paths[0]


def load_selected(torch, marker, contract, representation):
    """Validate prediction provenance without a training-resume checkpoint."""
    marker = Path(marker)
    origin = json.loads(marker.read_text())
    assert origin['source_run_id'] == PARENT_RUN_ID
    assert origin['source_script_sha256'] == PARENT_SOURCE_SHA256
    assert set(origin['files']) == {'selected.pt', 'protocol.json', 'selection.json'}
    for name, digest in origin['files'].items():
        assert sha(marker.parent / name) == digest, name
    protocol = json.loads((marker.parent / 'protocol.json').read_text())
    selected = json.loads((marker.parent / 'selection.json').read_text())
    # V2 stored TorchVersion in its protocol. Allow only that metadata class.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(marker.parent / 'selected.pt', map_location='cpu', weights_only=True)
    assert checkpoint['protocol'] == protocol and checkpoint['selected'] == selected
    # V2 copies the contract, then sets created_utc to the run's creation time.
    # The full original contract is still bound by representation's file hash.
    training_fields = {key: value for key, value in contract.items() if key != 'created_utc'}
    assert {key: protocol[key] for key in training_fields} == training_fields, 'Training contract changed'
    contract_created = datetime.fromisoformat(contract['created_utc'])
    run_created = datetime.fromisoformat(protocol['created_utc'])
    assert contract_created.tzinfo is not None and run_created.tzinfo is not None
    assert run_created >= contract_created, 'Run creation precedes its input contract'
    assert protocol['representation'] == representation, 'Frozen representation changed'
    expected_cache_id = hashlib.sha256(json.dumps(representation, sort_keys=True).encode()).hexdigest()
    assert protocol['feature_cache_id'] == expected_cache_id
    completed = selected['weak_pass']
    assert type(completed) is int and 1 <= completed <= contract['weak_passes']
    coverage = selected['training_coverage']
    assert coverage['strong_unique'] == contract['strong_gold'] == 11095
    assert coverage['weak_unique'] == contract['weak_clips'] == 79540
    assert coverage['weak_draws'] == {
        'no_timestamps': completed * contract['weak_bronze'],
        'unverified_timestamps': completed * contract['weak_silver'],
        'verified_timestamps': completed * contract['weak_gold']}
    assert selected['strong_completed_passes'] >= 1
    assert selected['updates'] >= completed * math.ceil(contract['weak_clips'] / contract['weak_batch'])
    decoder, grid = selected['decoder'], protocol['decoder_grid']
    assert decoder['threshold'] in grid['thresholds'] and decoder['nms_iou'] in grid['nms']
    assert decoder['min_duration'] == grid['min_duration']
    metrics = selected['metrics']
    assert metrics['reference_clips'] == contract['organizer_clips'] == 2501
    assert all(math.isfinite(metrics[key]) for key in ('event_f1', 'segment_dice', 'combined'))
    assert abs(metrics['combined'] - metrics['event_f1'] - metrics['segment_dice']) < 1e-12
    assert checkpoint['model_state'], 'Missing selected model weights'
    return origin, protocol, selected, checkpoint['model_state']


def main():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    SCRATCH.mkdir(parents=True, exist_ok=False)
    emit('starting_prediction_only', source_run_id=PARENT_RUN_ID)
    if importlib.metadata.version('torch').split('+')[0] != '2.8.0':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir',
                        '--index-url', 'https://download.pytorch.org/whl/cu128',
                        'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0'], check=True)
    if importlib.metadata.version('transformers') != '4.57.1':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir',
                        'transformers==4.57.1'], check=True)
    contract_path = only_input('full-wavlm-contract.json')
    bundle = contract_path.parent
    contract = json.loads(contract_path.read_text())
    for name, digest in contract['source_files'].items():
        assert sha(bundle / name) == digest, name
    sys.path.insert(0, str(bundle / 'source'))
    import numpy as np
    import soundfile as sf
    import torch
    import instance_pilot as pilot
    import wavlm_adapter as wa

    # Only manifests are read for population identity. No training/organizer
    # waveform is loaded, encoded, cached, calibrated, or used for an update.
    full_contract_path = only_input('full-input-contract.json')
    full_root = full_contract_path.parent
    assert sha(full_contract_path) == contract['full_input_contract_sha256']
    for name, key in [('training.jsonl', 'training_manifest_sha256'),
                      ('organizer-inference.jsonl', 'organizer_inference_sha256'),
                      ('organizer-references.jsonl', 'organizer_references_sha256')]:
        assert sha(full_root / name) == contract[key]
    training = read(full_root / 'training.jsonl')
    organizer = read(full_root / 'organizer-inference.jsonl')
    test = read(bundle / 'test-inference.jsonl')
    assert (len(training), len(organizer), len(test)) == (90635, 2501, 5517)
    populations = [{row['clip_id'] for row in rows} for rows in (training, organizer, test)]
    assert [len(ids) for ids in populations] == [90635, 2501, 5517]
    assert all(not populations[i] & populations[j] for i in range(3) for j in range(i))
    assert Counter((row['quality'], row['supervision']) for row in training) == {
        ('verified_timestamps', 'strong_gold'): 11095,
        ('verified_timestamps', 'weak_clip'): 16,
        ('unverified_timestamps', 'weak_clip'): 61640,
        ('no_timestamps', 'weak_clip'): 17884}
    del training, organizer
    test_roots = [root for parent in (bundle, bundle.parent) for root in (parent, parent / 'test-audio')
                  if (root / test[0]['audio_path']).is_file()]
    assert len(set(test_roots)) == 1, 'Expected one ordinary test audio archive layout'
    for row in test:
        assert set(row) == {'clip_id', 'audio_path', 'duration'}
        row['audio_path'] = str(test_roots[0] / row['audio_path'])
    assert all(Path(row['audio_path']).is_file() for row in test)
    cache_dtype = np.dtype('float16')
    representation = {'encoder': 'microsoft/wavlm-large', 'revision': wa.REVISION,
        'asset_sha256': wa.ASSET_HASHES, 'encoder_dtype': 'float32', 'storage_dtype': 'float16',
        'head_input_dtype': 'float32', 'representation': 'fixed uniform mean of all 25 hidden states',
        'encoder_trainable': False, 'layer_weights_trainable': False, 'feature_dim': 1024,
        'frame_grid': wa.FRAME_GRID, 'source_contract_sha256': sha(contract_path)}
    marker = only_input('full-wavlm-predict-input.json')
    origin, source_protocol, selected, model_state = load_selected(torch, marker, contract, representation)
    assert str(source_protocol['compute']['torch']) == str(torch.__version__)
    assert source_protocol['decoder_grid'] == {
        'thresholds': pilot.THRESHOLDS, 'nms': pilot.NMS, 'min_duration': pilot.MIN_DURATION}
    assert torch.cuda.is_available(), 'Prediction requires an explicitly launched CUDA job'
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # JSON round-trip normalizes historical TorchVersion values to plain strings.
    protocol = {'mode': 'prediction_only',
        'source_training_protocol': json.loads(json.dumps(source_protocol, allow_nan=False)),
        'prediction_execution': {**origin, 'source_sha256': sha(Path(__file__)),
            'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(device),
            'training_updates_performed': 0, 'training_or_organizer_features_extracted': 0,
            'selected_checkpoint_completed_weak_passes': selected['weak_pass'], 'created_utc': now()}}
    write(OUTPUT / 'protocol.json', protocol)
    write(OUTPUT / 'selection.json', selected)
    shutil.copy2(marker.parent / 'selected.pt', OUTPUT / 'selected.pt')
    emit('selected_checkpoint_verified', selected_weak_pass=selected['weak_pass'],
         selected_updates=selected['updates'], training_coverage=selected['training_coverage'],
         selected_sha256=origin['files']['selected.pt'], test_clips=len(test))
    model = pilot.IntervalModel(feature_dim=1024, mode='red', epn_context='per_class').to(device)
    model.load_state_dict(model_state, strict=True)
    del model_state
    model.eval()
    assets = SCRATCH / 'assets'
    assets.mkdir()
    for name, digest in wa.ASSET_HASHES.items():
        destination = assets / name
        urllib.request.urlretrieve(f'https://huggingface.co/microsoft/wavlm-large/resolve/{wa.REVISION}/{name}', destination)
        assert sha(destination) == digest, name
    assert sum(path.stat().st_size for path in assets.iterdir() if path.is_file()) <= 2 * 1024**3
    maxima = {}

    def resources():
        values = {'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  'cuda_allocated_bytes': torch.cuda.memory_allocated(device),
                  'cuda_reserved_bytes': torch.cuda.memory_reserved(device)}
        for key, value in values.items():
            maxima[key] = max(maxima.get(key, 0), value)
        assert values['peak_process_rss_bytes'] <= 14 * 1024**3
        assert values['cuda_allocated_bytes'] <= 16 * 1024**3
        return values

    def encoder_model():
        value = wa.NativeWavLMIntervals.from_assets(assets)
        assert torch.count_nonzero(value.layer_weights).item() == 0
        value.requires_grad_(False).eval().set_device(device, 'cpu')
        assert not any(parameter.requires_grad for parameter in value.parameters())
        return value

    def encode_row(encoder, row):
        waveform, rate = sf.read(row['audio_path'], dtype='float32')
        assert rate == 16000 and waveform.ndim == 1 and len(waveform) == row['native_sample_count']
        with torch.inference_mode():
            values, lengths, _ = encoder.encode(torch.from_numpy(waveform)[None], torch.tensor([len(waveform)]))
            result = values[0, :int(lengths[0])].float().cpu().numpy().astype(cache_dtype, copy=True)
        assert result.shape == (row['num_frames'], 1024) and np.isfinite(result).all()
        return result

    encoder = encoder_model()
    # The test loop, decoder, endpoint/ID validation, and ZIP writer below are
    # copied unchanged from the running v2 source.
    predictions = []
    inference_started = time.monotonic()
    for index, row in enumerate(test, 1):
        item = wa.native_row(row, round(row['duration'] * 16000))
        values = encode_row(encoder, item).astype(np.float32, copy=False)
        with torch.inference_mode():
            output = model(torch.from_numpy(values)[None].to(device), torch.tensor([item['num_frames']]))
        candidates = pilot.candidate_intervals(item, output['presence'][0].cpu().numpy(), output['distances'][0].cpu().numpy())
        decoded = pilot.decode_candidates(item, candidates, selected['decoder'])
        events = [{'onset': float(event['onset']), 'offset': float(event['offset'])} for event in decoded['events']]
        assert all(math.isfinite(event['onset']) and math.isfinite(event['offset'])
                   and 0 <= event['onset'] < event['offset'] <= row['duration'] + 1e-8 for event in events)
        predictions.append({'clip_id': row['clip_id'], 'events': events})
        del values, output
        if index == 1 or index % 100 == 0 or index == len(test):
            emit('test_inference_progress', completed=index, total=len(test),
                 elapsed_seconds=time.monotonic() - inference_started, resources=resources())
        torch.cuda.empty_cache()
    del encoder
    assert len(predictions) == len({row['clip_id'] for row in predictions}) == 5517
    assert {row['clip_id'] for row in predictions} == populations[2]
    prediction_path = OUTPUT / 'predictions.jsonl'
    pilot.write_predictions(prediction_path, predictions)
    submission = OUTPUT / 'full-wavlm-predictions.zip'
    with zipfile.ZipFile(submission, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, 'predictions.jsonl')
    with zipfile.ZipFile(submission) as archive:
        assert archive.namelist() == ['predictions.jsonl'] and archive.testzip() is None
        assert archive.read('predictions.jsonl') == prediction_path.read_bytes()
    verification = {'test_clips': 5517, 'unique_clip_ids': 5517, 'exact_test_id_coverage': True,
                    'finite_valid_endpoints': True, 'events': sum(len(row['events']) for row in predictions),
                    'empty_clips': sum(not row['events'] for row in predictions),
                    'predictions_sha256': sha(prediction_path), 'zip_sha256': sha(submission),
                    'zip_members': ['predictions.jsonl'], 'leaderboard_uploaded': False}
    write(OUTPUT / 'test-verification.json', verification)
    complete = {'status': 'complete', 'mode': 'prediction_only', 'selected': selected,
                'selected_checkpoint_completed_weak_passes': selected['weak_pass'],
                'training_updates_performed': 0, 'test_package': verification,
                'maxima': maxima, 'elapsed_seconds': time.monotonic() - started, 'completed_utc': now()}
    write(OUTPUT / 'complete.json', complete)
    emit('complete', **complete)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = {'error': traceback.format_exc(), 'utc': now()}
        if OUTPUT.is_dir():
            write(OUTPUT / 'failure.json', failure)
        emit('failed', **failure)
        raise
