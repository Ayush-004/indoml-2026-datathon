"""Train frozen WavLM + RED on the entire usable Vaani release, then predict test."""
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
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
OUTPUT = Path('/kaggle/working/full-wavlm')
SCRATCH = Path('/kaggle/temp/full-wavlm')
WORKING_FEATURES = OUTPUT / 'temporary-feature-cache'
WORKING_FEATURE_LIMIT = 14 * 1024**3
SCRATCH_FEATURE_LIMIT = 42 * 1024**3
working_features_owned = False


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


def main():
    global working_features_owned
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    SCRATCH.mkdir(parents=True, exist_ok=False)
    WORKING_FEATURES.mkdir()
    working_features_owned = True
    assert OUTPUT.stat().st_dev != SCRATCH.stat().st_dev, (
        'Working output and scratch must use separate filesystems for this bounded cache layout')
    emit('starting_full_release')
    if importlib.metadata.version('torch').split('+')[0] != '2.8.0':
        emit('installing_pinned_cuda_torch')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir',
                        '--index-url', 'https://download.pytorch.org/whl/cu128',
                        'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0'], check=True)
    if importlib.metadata.version('transformers') != '4.57.1':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir',
                        'transformers==4.57.1'], check=True)
    contract_path = only_input('full-wavlm-contract.json')
    bundle = contract_path.parent
    contract = json.loads(contract_path.read_text())
    assert sha(bundle / 'initial-head.pt') == contract['initial_checkpoint_sha256']
    for name, digest in contract['source_files'].items():
        assert sha(bundle / name) == digest, name
    sys.path.insert(0, str(bundle / 'source'))
    import numpy as np
    import soundfile as sf
    import torch
    import instance_pilot as pilot
    import wavlm_adapter as wa
    from input_paths import InputLayout
    from mixed_training import WindowStream
    from indoml.weak_supervision import positive_category_mask, weak_positive_loss

    assert torch.cuda.is_available(), 'This full-data run requires a Kaggle CUDA GPU'
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    emit('gpu_ready', gpu=torch.cuda.get_device_name(device), torch=torch.__version__)

    full_contract_path = only_input('full-input-contract.json')
    full_root = full_contract_path.parent
    assert sha(full_contract_path) == contract['full_input_contract_sha256']
    for name, key in [('training.jsonl', 'training_manifest_sha256'),
                      ('organizer-inference.jsonl', 'organizer_inference_sha256'),
                      ('organizer-references.jsonl', 'organizer_references_sha256')]:
        assert sha(full_root / name) == contract[key]
    training = read(full_root / 'training.jsonl')
    organizer = read(full_root / 'organizer-inference.jsonl')
    references = {row['clip_id']: row for row in read(full_root / 'organizer-references.jsonl')}
    test = read(bundle / 'test-inference.jsonl')
    assert (len(training), len(organizer), len(references), len(test)) == (90635, 2501, 2501, 5517)
    populations = [{row['clip_id'] for row in rows} for rows in (training, organizer, test)]
    assert [len(ids) for ids in populations] == [90635, 2501, 5517]
    assert all(not populations[i] & populations[j] for i in range(3) for j in range(i))
    assert set(references) == populations[1]
    counts = Counter((row['quality'], row['supervision']) for row in training)
    assert counts == {('verified_timestamps', 'strong_gold'): 11095,
                      ('verified_timestamps', 'weak_clip'): 16,
                      ('unverified_timestamps', 'weak_clip'): 61640,
                      ('no_timestamps', 'weak_clip'): 17884}
    layout = InputLayout(full_root, json.loads(full_contract_path.read_text()), training, organizer)
    for index, row in enumerate(training):
        row['audio_path'] = str(layout.training_path(index, row))
    for row in organizer:
        row['audio_path'] = str(layout.organizer_path(row))
        row['events'] = references[row['clip_id']]['events']
    test_roots = [root for parent in (bundle, bundle.parent) for root in (parent, parent / 'test-audio')
                  if (root / test[0]['audio_path']).is_file()]
    assert len(set(test_roots)) == 1, 'Expected one ordinary test audio archive layout'
    for row in test:
        assert set(row) == {'clip_id', 'audio_path', 'duration'}
        row['audio_path'] = str(test_roots[0] / row['audio_path'])
    assert all(Path(row['audio_path']).is_file() for rows in (training, organizer, test) for row in rows)

    # Cache training and organizer features once; test is encoded only after selection.
    native = [[wa.native_row(row, round(row['duration'] * 16000)) for row in rows]
              for rows in (training, organizer)]
    feature_values = sum(row['num_frames'] * 1024 for rows in native for row in rows)
    # Kaggle's root-overlay free-space reading does not expose its writable quota.
    # Keep feature payloads within explicit budgets on the two separate filesystems.
    free = shutil.disk_usage(SCRATCH).free
    working_free = shutil.disk_usage(OUTPUT).free
    storage = 'float16'
    cache_dtype = np.dtype(storage)
    reserve = (len(training) + len(organizer)) * 8192
    assert feature_values * cache_dtype.itemsize + reserve <= WORKING_FEATURE_LIMIT + SCRATCH_FEATURE_LIMIT
    assert working_free >= WORKING_FEATURE_LIMIT + 4 * 1024**3, 'Insufficient output-volume headroom'
    cache_bytes = {'working': 0, 'scratch': 0}
    representation = {'encoder': 'microsoft/wavlm-large', 'revision': wa.REVISION,
        'asset_sha256': wa.ASSET_HASHES, 'encoder_dtype': 'float32', 'storage_dtype': storage,
        'head_input_dtype': 'float32', 'representation': 'fixed uniform mean of all 25 hidden states',
        'encoder_trainable': False, 'layer_weights_trainable': False, 'feature_dim': 1024,
        'frame_grid': wa.FRAME_GRID, 'source_contract_sha256': sha(contract_path)}
    cache_id = hashlib.sha256(json.dumps(representation, sort_keys=True).encode()).hexdigest()
    features_dir = SCRATCH / 'features'
    features_dir.mkdir()
    for rows in native:
        for row in rows:
            row.update(feature_dim=1024, feature_cache_id=cache_id,
                       feature_path=str(features_dir / (hashlib.sha256(row['clip_id'].encode()).hexdigest() + '.npz')))
    protocol = {**contract, 'representation': representation, 'feature_cache_id': cache_id,
        'scratch': {'path': str(SCRATCH), 'reported_free_bytes_not_writable_quota': free,
                    'feature_array_bytes': feature_values * cache_dtype.itemsize, 'reserve_bytes': reserve,
                    'scratch_feature_limit_bytes': SCRATCH_FEATURE_LIMIT,
                    'working_feature_limit_bytes': WORKING_FEATURE_LIMIT,
                    'working_cache_path': str(WORKING_FEATURES), 'working_free_bytes': working_free,
                    'cache_compression': 'lossless ZIP deflate of FP16 arrays',
                    'cache_byte_accounting': 'max(file size, allocated stat.st_blocks * 512)',
                    'separate_filesystems_verified': True, 'test_features_cached': False},
        'compute': {'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__, 'head_dtype': 'float32'},
        'storage_rounding': 'FP32 encoder features rounded once to FP16, then expanded to FP32 for head input'
                           if storage == 'float16' else 'No reduced-precision feature storage',
        'initialization_is_new_full_data_run': True, 'custom_holdout': False,
        'decoder_grid': {'thresholds': pilot.THRESHOLDS, 'nms': pilot.NMS, 'min_duration': pilot.MIN_DURATION},
        'input_layout': layout.evidence(), 'created_utc': now()}
    write(OUTPUT / 'protocol.json', protocol)
    emit('full_inputs_ready', training_clips=len(training), organizer_clips=len(organizer), test_clips=len(test),
         supervision_counts=[{'quality': q, 'supervision': s, 'clips': n} for (q, s), n in counts.items()],
         storage_dtype=storage, scratch=protocol['scratch'])

    assets = SCRATCH / 'assets'
    assets.mkdir()
    for name, digest in wa.ASSET_HASHES.items():
        destination = assets / name
        urllib.request.urlretrieve(f'https://huggingface.co/microsoft/wavlm-large/resolve/{wa.REVISION}/{name}', destination)
        assert sha(destination) == digest, name
    model_asset_bytes = sum(path.stat().st_size for path in assets.iterdir() if path.is_file())
    assert model_asset_bytes <= 2 * 1024**3, 'Model assets exceed the reserved footprint'
    protocol['scratch']['model_asset_bytes'] = model_asset_bytes
    write(OUTPUT / 'protocol.json', protocol)
    emit('cache_storage_ready', storage_dtype=storage, working_cache_cap_bytes=WORKING_FEATURE_LIMIT,
         scratch_cache_cap_bytes=SCRATCH_FEATURE_LIMIT, model_asset_bytes=model_asset_bytes,
         writable_audio_bytes=0, metadata_and_atomic_write_reserve_bytes=reserve)
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
    extracted = 0
    extraction_started = time.monotonic()
    for split, rows in zip(('training', 'organizer'), native, strict=True):
        for index, row in enumerate(rows, 1):
            values = encode_row(encoder, row)
            metadata = {'cache_id': cache_id, 'clip_id': row['clip_id'], 'num_frames': row['num_frames'],
                        **wa.FRAME_GRID, 'storage_dtype': storage, 'encoder_dtype': 'float32'}
            buffer = io.BytesIO()
            np.savez_compressed(buffer, features=values, metadata=np.array(json.dumps(metadata)))
            payload = buffer.getbuffer()
            size = len(payload)
            allocation_bound = size + 8192
            volume = 'working' if cache_bytes['working'] + allocation_bound <= WORKING_FEATURE_LIMIT else 'scratch'
            limit = WORKING_FEATURE_LIMIT if volume == 'working' else SCRATCH_FEATURE_LIMIT
            assert cache_bytes[volume] + allocation_bound <= limit, f'{volume} feature-cache byte budget exceeded'
            directory = WORKING_FEATURES if volume == 'working' else features_dir
            destination = directory / Path(row['feature_path']).name
            row['feature_path'] = str(destination)
            temporary = destination.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                stream.write(payload)
            temporary.replace(destination)
            allocated = max(size, destination.stat().st_blocks * 512)
            assert allocated <= allocation_bound, 'Unexpected filesystem allocation overhead'
            cache_bytes[volume] += allocated
            del payload, buffer, values
            extracted += 1
            if extracted == 1 or extracted % 100 == 0 or index == len(rows):
                emit('extraction_progress', split=split, split_completed=index, split_total=len(rows),
                     completed=extracted, total=len(training) + len(organizer),
                     elapsed_seconds=time.monotonic() - extraction_started, resources=resources(),
                     feature_cache_bytes=dict(cache_bytes))
                assert shutil.disk_usage(OUTPUT).free >= 2 * 1024**3, 'Output-volume reserve exhausted'
            torch.cuda.empty_cache()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    emit('full_cache_complete', training_clips=len(native[0]), organizer_clips=len(native[1]),
         storage_dtype=storage, elapsed_seconds=time.monotonic() - extraction_started,
         feature_cache_bytes=dict(cache_bytes))

    class CachedIntervals(pilot.IntervalDataset):
        def __init__(self, rows, mode):
            self.rows = self.all_rows = rows
            self.mode, self.feature_dim, self.feature_cache_id = mode, 1024, cache_id
            self.manifest = OUTPUT / 'protocol.json'
            self.frame_hop, self.frame_offset = .02, .0125
            self.silver_positive_weight, self.silver_negative_weight = .3, .05
            self.excluded = []
            self.index = {row['clip_id']: index for index, row in enumerate(rows)}

        def batch(self, rows):
            return pilot.collate_intervals([self[self.index[row['clip_id']]] for row in rows])

    strong_rows = [row for row in native[0] if row['supervision'] == 'strong_gold']
    weak_rows = [row for row in native[0] if row['supervision'] == 'weak_clip']
    assert (len(strong_rows), len(weak_rows)) == (11095, 79540)
    assert all(row['events'] and row['quality'] == 'verified_timestamps' for row in strong_rows)
    assert all(row['events'] is None and row['positive_categories'] for row in weak_rows)
    strong, weak = CachedIntervals(strong_rows, 'train'), CachedIntervals(weak_rows, 'predict')
    val = CachedIntervals(native[1], 'predict')
    positive_weights, positive_weight_details = strong.positive_weights(5.)
    positive_weights = positive_weights.to(device)
    write(OUTPUT / 'strong-positive-weights.json', positive_weight_details)
    torch.manual_seed(contract['seed'])
    np.random.seed(contract['seed'])
    model = pilot.IntervalModel(feature_dim=1024, mode='red', epn_context='per_class').to(device)
    initial = torch.load(bundle / 'initial-head.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(initial['model_state'], strict=True)
    del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract['learning_rate'], weight_decay=contract['weight_decay'])
    torch.manual_seed(contract['seed'] + 1)
    torch.cuda.manual_seed_all(contract['seed'] + 1)
    strong_stream = WindowStream(strong_rows, contract['strong_batch'], contract['max_windows'], contract['seed'] + 2)
    weak_stream = WindowStream(weak_rows, contract['weak_batch'], contract['max_windows'], contract['seed'] + 3)

    def forward(current, batch, arm):
        output = current(batch['features'].to(device), batch['lengths'])
        return output if current.training else {key: value.cpu() for key, value in output.items()}

    pilot.forward = forward
    history, selected, updates = [], None, 0
    training_started = time.monotonic()
    losses = Counter()
    updates_in_pass = 0

    def save_last():
        pilot.save(OUTPUT / 'last.pt', {'protocol': protocol, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'updates': updates, 'history': history, 'selected': selected,
            'strong_stream': strong_stream.state_dict(), 'weak_stream': weak_stream.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})

    emit('full_training_started', strong_clips=len(strong), weak_clips=len(weak), weak_passes=contract['weak_passes'],
         initial_selected_epoch=contract['initial_selected_epoch'], fresh_optimizer=True, weak_weight=contract['weak_weight'])
    while weak_stream.completed_cycles < contract['weak_passes']:
        previous_passes = weak_stream.completed_cycles
        gold_batch_rows, weak_batch_rows = strong_stream.take(), weak_stream.take()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = strong.batch(gold_batch_rows)
        output = forward(model, batch, 'red')
        loss, components = pilot.interval_loss(output, batch, positive_weights, device, 'red', 100., 1., collar_weight=0.)
        assert torch.isfinite(loss), 'Nonfinite strong training loss'
        loss.backward()
        strong_value = float(loss.detach())
        del loss, output, batch, components
        batch = weak.batch(weak_batch_rows)
        output = forward(model, batch, 'red')
        mask = positive_category_mask([row['positive_categories'] for row in weak_batch_rows], device=device)
        loss, stats = weak_positive_loss(output['presence'], batch['lengths'], mask)
        assert torch.isfinite(loss), 'Nonfinite weak training loss'
        (contract['weak_weight'] * loss).backward()
        weak_value = float(loss.detach())
        del loss, output, batch, mask, stats
        torch.nn.utils.clip_grad_norm_(model.parameters(), contract['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        updates_in_pass += 1
        losses.update(strong=strong_value, weak=weak_value)
        if updates == 1 or updates % 100 == 0:
            emit('full_training_progress', updates=updates, weak_completed_passes=weak_stream.completed_cycles,
                 weak_cursor=weak_stream.cursor, weak_unique_seen=len(weak_stream.seen), strong_unique_seen=len(strong_stream.seen),
                 weak_draws=dict(weak_stream.draws), strong_loss=strong_value, weak_loss=weak_value,
                 elapsed_seconds=time.monotonic() - training_started, resources=resources())
        if updates % 500 == 0:
            save_last()
        if weak_stream.completed_cycles > previous_passes:
            assert len(strong_stream.seen) == 11095 and len(weak_stream.seen) == 79540
            before = time.monotonic()
            choice, trials, predictions = pilot.validate(model, val, 'red', 16)
            entry = {'weak_pass': weak_stream.completed_cycles, 'updates': updates,
                     'strong_completed_passes': strong_stream.completed_cycles,
                     'training_coverage': {'strong_unique': len(strong_stream.seen), 'weak_unique': len(weak_stream.seen),
                                           'weak_draws': dict(weak_stream.draws)},
                     'mean_strong_loss': losses['strong'] / updates_in_pass,
                     'mean_weak_loss': losses['weak'] / updates_in_pass,
                     'validation_seconds': time.monotonic() - before, **choice}
            history.append(entry)
            write(OUTPUT / 'history.json', history)
            write(OUTPUT / f'calibration-pass-{entry["weak_pass"]:02d}.json', trials)
            if selected is None or choice['metrics']['combined'] > selected['metrics']['combined']:
                selected = entry
                write(OUTPUT / 'selection.json', selected)
                pilot.save(OUTPUT / 'selected.pt', {'protocol': protocol, 'selected': selected, 'model_state': model.state_dict()})
                pilot.write_predictions(OUTPUT / 'selected-organizer-predictions.jsonl', predictions)
            if weak_stream.completed_cycles == contract['weak_passes']:
                pilot.write_predictions(OUTPUT / 'final-organizer-predictions.jsonl', predictions)
            save_last()
            emit('full_training_pass_complete', **entry, selected_weak_pass=selected['weak_pass'])
            losses.clear()
            updates_in_pass = 0

    assert selected is not None and weak_stream.completed_cycles == 8
    terminal = {'status': 'training_complete', 'completed_weak_passes': weak_stream.completed_cycles,
                'strong_completed_passes': strong_stream.completed_cycles, 'updates': updates,
                'selected': selected, 'final': history[-1], 'maxima': maxima,
                'training_elapsed_seconds': time.monotonic() - training_started, 'completed_utc': now()}
    write(OUTPUT / 'training-complete.json', terminal)
    emit('training_complete', **terminal)
    shutil.rmtree(WORKING_FEATURES)
    working_features_owned = False
    shutil.rmtree(features_dir)
    emit('feature_cache_released', feature_cache_bytes=dict(cache_bytes))
    checkpoint = torch.load(OUTPUT / 'selected.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    del checkpoint, optimizer
    model.eval()
    encoder = encoder_model()
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
    complete = {'status': 'complete', 'selected': selected, 'final': history[-1], 'completed_weak_passes': 8,
                'test_package': verification, 'maxima': maxima, 'elapsed_seconds': time.monotonic() - started,
                'completed_utc': now()}
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
    finally:
        if working_features_owned:
            shutil.rmtree(WORKING_FEATURES)
