"""E0 stage 1 (local): one frozen WavLM-Large mean-layer feature cache of the inner split, followed by the RED head
recipe of instance_pilot.py. T1_WAVLM_ASSETS = a folder with the pinned microsoft/wavlm-large files;
T1_WORK/inner-split/{train,val}.jsonl = the inner split of prepare_inner.py."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('T1_WORK', '.')).resolve()
ASSETS = Path(os.environ.get('T1_WAVLM_ASSETS', 'wavlm-large-assets'))
SPLITS = {'train': ROOT / 'inner-split/train.jsonl',
          'inner': ROOT / 'inner-split/val.jsonl'}
CAPS = {'peak_process_rss_bytes': 14 * 1024**3,
        'mps_current_allocated_bytes': 16 * 1024**3,
        'mps_driver_allocated_bytes': 18 * 1024**3}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def setup(output):
    import wavlm_adapter as wa
    rows = {name: read_rows(path) for name, path in SPLITS.items()}
    assert [len(rows[name]) for name in SPLITS] == [8213, 1439]
    assert not {r['clip_id'] for r in rows['train']} & {r['clip_id'] for r in rows['inner']}
    contract = {'encoder': 'microsoft/wavlm-large', 'revision': wa.REVISION,
                'asset_sha256': wa.ASSET_HASHES, 'representation': 'fixed uniform mean of all 25 exposed hidden states',
                'encoder_trainable': False, 'layer_weights_trainable': False,
                'dtype': 'float32', 'feature_dim': 1024, 'frame_grid': wa.FRAME_GRID,
                'normalization': 'existing wavlm_adapter.normalize_real_audio',
                'adapter_sha256': sha(HERE / 'wavlm_adapter.py'),
                'source_manifests': {name: {'path': str(path), 'sha256': sha(path)} for name, path in SPLITS.items()}}
    cache_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cache = output / 'cache'
    (cache / 'features').mkdir(parents=True, exist_ok=True)
    identity = {**contract, 'cache_id': cache_id}
    identity_path = cache / 'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Existing cache belongs to another representation or input'
    else:
        write(identity_path, identity)
    native = {}
    for name, items in rows.items():
        native[name] = []
        for row in items:
            item = wa.native_row(row, round(row['duration'] * wa.SAMPLE_RATE))
            filename = hashlib.sha256(row['clip_id'].encode()).hexdigest() + '.npz'
            item.update(feature_path=str(cache / 'features' / filename), feature_cache_id=cache_id, feature_dim=1024)
            native[name].append(item)
    return wa, native, cache


def memory(device):
    import resource
    import torch
    result = {'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
              * (1 if sys.platform == 'darwin' else 1024)}
    if device.type == 'mps':
        result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    return result


def resource_check(output, device, maxima):
    # The resource measurement of the local native trainer (memory() of its runner, unchanged).
    values = memory(device)
    for key, value in values.items():
        maxima[key] = max(maxima.get(key, 0), value)
    if any(values.get(key, 0) > limit for key, limit in CAPS.items()):
        write(output / 'resource-stop.json', {'created_utc': now(), 'pid': os.getpid(), 'values': values, 'caps': CAPS})
        raise RuntimeError('Local resource cap exceeded; completed cache files and checkpoints are retained')
    return values


def extract(output):
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    wa, rows, cache = setup(output)
    ordered = rows['train'] + rows['inner']
    pending = [row for row in ordered if not Path(row['feature_path']).exists()]
    started = time.monotonic()
    maxima = {}
    device = torch.device('mps')
    assert torch.backends.mps.is_available()
    if pending:
        model = wa.NativeWavLMIntervals.from_assets(ASSETS)
        assert torch.count_nonzero(model.layer_weights).item() == 0
        model.requires_grad_(False).eval().set_device(device, 'cpu')
        assert not any(p.requires_grad for p in model.parameters())
        completed = len(ordered) - len(pending)
        print(json.dumps({'stage': 'extracting', 'pid': os.getpid(), 'cached': completed, 'total': len(ordered)}), flush=True)
        for row in pending:
            resource_check(output, device, maxima)
            audio, sample_rate = sf.read(row['audio_path'], dtype='float32')
            assert sample_rate == 16000 and audio.ndim == 1 and len(audio) == row['native_sample_count']
            with torch.inference_mode():
                features, lengths, mask = model.encode(torch.from_numpy(audio)[None], torch.tensor([len(audio)], dtype=torch.long))
                torch.mps.synchronize()
                values = features[0, :int(lengths[0])].float().cpu().numpy().copy()
            resource_check(output, device, maxima)
            assert values.shape == (row['num_frames'], 1024) and values.dtype == np.float32 and np.isfinite(values).all()
            metadata = {'cache_id': row['feature_cache_id'], 'clip_id': row['clip_id'],
                        'num_frames': row['num_frames'], **wa.FRAME_GRID, 'native_sample_count': len(audio),
                        'audio_sha256': sha(row['audio_path'])}
            destination = Path(row['feature_path'])
            temporary = destination.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, features=values, metadata=np.array(json.dumps(metadata, sort_keys=True)))
            temporary.replace(destination)
            completed += 1
            del features, lengths, mask, values, audio
            torch.mps.empty_cache()
            usage = resource_check(output, device, maxima)
            if completed % 25 == 0 or completed == len(ordered):
                state = {'stage': 'extracting', 'pid': os.getpid(), 'completed': completed, 'total': len(ordered),
                         'elapsed_seconds': time.monotonic() - started, 'maxima': maxima, 'updated_utc': now()}
                write(output / 'extraction-state.json', state)
                print(json.dumps(state), flush=True)
            # A real completed clip is retained before yielding on persistent driver growth.
            if usage.get('mps_driver_allocated_bytes', 0) >= 12 * 1024**3 and completed < len(ordered):
                write(output / 'extraction-state.json', {'stage': 'yield_after_completed_clip', 'completed': completed,
                                                       'total': len(ordered), 'maxima': maxima, 'updated_utc': now()})
                return 75
        del model
    for name, items in rows.items():
        assert all(Path(row['feature_path']).is_file() for row in items)
        destination = cache / f'{name}.jsonl'
        temporary = destination.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in items))
        temporary.replace(destination)
    write(cache / 'complete.json', {'completed_utc': now(), 'train_clips': 8213, 'inner_clips': 1439,
                                   'feature_dim': 1024, 'dtype': 'float32', 'maxima': maxima,
                                   'manifest_sha256': {name: sha(cache / f'{name}.jsonl') for name in rows}})
    print(json.dumps({'stage': 'cache_complete', 'clips': len(ordered), 'cache': str(cache)}), flush=True)
    return 0


def train(output):
    import torch
    import wavlm_adapter as wa
    import instance_pilot as pilot
    from indoml.data import validate_row
    from indoml.transfer import CATEGORIES, check_separate_splits
    from indoml.atst_finetune import check_geography
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cache = output / 'cache'
    complete = json.loads((cache / 'complete.json').read_text())
    assert complete['train_clips'] == 8213 and complete['inner_clips'] == 1439

    class NativeCachedIntervals(pilot.IntervalDataset):
        """Retain the existing loader/targets; validate native convolution lengths."""
        def __init__(self, manifest, mode):
            self.manifest = Path(manifest).resolve()
            self.feature_dim, self.mode = 1024, mode
            self.rows = read_rows(self.manifest)
            self.all_rows, self.excluded = self.rows, []
            self.silver_positive_weight, self.silver_negative_weight = .3, .05
            self.frame_hop, self.frame_offset = .02, .0125
            self.feature_cache_id = self.rows[0]['feature_cache_id']
            assert len({row['clip_id'] for row in self.rows}) == len(self.rows)
            for row in self.rows:
                validate_row(row, str(self.manifest))
                assert row['quality'] == 'verified_timestamps' and row['events'] is not None
                assert all(event['category'] in CATEGORIES for event in row['events'])
                declared = set(row.get('source', {}).get('metadata', {}).get('NoiseCategory') or [])
                assert not declared - {event['category'] for event in row['events']}
                assert row['feature_dim'] == 1024 and row['feature_cache_id'] == self.feature_cache_id
                assert (row['frame_hop'], row['frame_offset']) == (self.frame_hop, self.frame_offset)
                assert row['num_frames'] == int(wa.native_lengths(row['native_sample_count'])) > 0
                assert Path(row['feature_path']).is_file()

    def datasets():
        gold = NativeCachedIntervals(cache / 'train.jsonl', 'train')
        inner = NativeCachedIntervals(cache / 'inner.jsonl', 'val')
        assert len(gold) == 8213 and len(inner) == 1439
        check_separate_splits(gold, inner)
        check_geography(gold, inner)
        assert gold.feature_cache_id == inner.feature_cache_id
        return gold, inner

    pilot.datasets = datasets
    pilot.new_model = lambda arm: pilot.IntervalModel(feature_dim=1024, mode='red', epn_context='per_class')
    maximum = {}
    original_loss = pilot.loss_for
    original_validate = pilot.validate
    epoch = [0]
    head_output = output / 'head-seed73'

    def loss_for(*args, **kwargs):
        resource_check(output, torch.device('cpu'), maximum)
        return original_loss(*args, **kwargs)

    def validate(model, dataset, arm, batch_size, fixed_decoder=None):
        result = original_validate(model, dataset, arm, batch_size, fixed_decoder=fixed_decoder)
        resource_check(output, torch.device('cpu'), maximum)
        if getattr(dataset, 'manifest', None) == (cache / 'inner.jsonl').resolve():
            if epoch[0] in (0, 30):
                name = 'initial-predictions.jsonl' if epoch[0] == 0 else 'final-predictions.jsonl'
                pilot.write_predictions(head_output / name, result[2])
            epoch[0] += 1
        return result

    pilot.loss_for, pilot.validate = loss_for, validate
    write(output / 'experiment.json', {'created_utc': now(), 'scope': 'frozen WavLM uniform-mean features plus existing RED/BiGRU head',
                                      'source_sha256': sha(__file__), 'pilot_sha256': sha(HERE / 'instance_pilot.py'),
                                      'cache_identity': json.loads((cache / 'identity.json').read_text()),
                                      'seed': 73, 'epochs': 30, 'batch_size': 16, 'optimizer': 'AdamW',
                                      'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip': 5.,
                                      'objective': 'existing presence + 100*boundary + IoU, collar weight zero',
                                      'decoder': {'thresholds': pilot.THRESHOLDS, 'nms_ious': pilot.NMS, 'min_duration': pilot.MIN_DURATION},
                                      'comparison': {'cached_ATST_RED_selected_INNER': 1.0682449748617886,
                                                     'scope': 'same reused 1439 INNER clips; not organizer development or hidden test'},
                                      'native_length_adapter': 'Same IntervalDataset loading, targets and weights; native valid-convolution length replaces full decision-cell coverage check',
                                      'caps': CAPS, 'cpu_threads': 2})
    print(json.dumps({'stage': 'head_training', 'pid': os.getpid(), 'output': str(head_output)}), flush=True)
    pilot.train_arm(argparse.Namespace(arm='red', seed=73, output=head_output))
    assert epoch[0] == 31
    write(output / 'head-resources.json', {'maxima': maximum, 'caps': CAPS})


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'head-seed73').exists(), 'Preserve existing trained output'
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    state = {'status': 'starting', 'started_utc': now(), 'supervisor_pid': os.getpid(), 'output': str(output), 'children': []}
    active = [None]

    def stop(signum, frame):
        if active[0] is not None and active[0].poll() is None:
            os.killpg(active[0].pid, signal.SIGTERM)
        raise KeyboardInterrupt('Stop requested')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        stage = 'extract'
        while True:
            log_path = output / f'{stage}-{len(state["children"]):02d}.log'
            with log_path.open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), stage, '--output', str(output)],
                                         cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[0] = child
                state.update(status=stage, updated_utc=now())
                entry = {'stage': stage, 'pid': child.pid, 'log': str(log_path), 'started_utc': now()}
                state['children'].append(entry)
                write(output / 'supervision.json', state)
                print(json.dumps(entry), flush=True)
                code = child.wait()
                entry.update(exit_code=code, finished_utc=now())
                active[0] = None
            write(output / 'supervision.json', state)
            if stage == 'extract' and code == 75:
                continue
            if code != 0:
                raise RuntimeError(f'{stage} exited {code}; preserve output and inspect the actual error')
            if stage == 'train':
                break
            stage = 'train'
        state.update(status='complete', completed_utc=now())
        write(output / 'supervision.json', state)
    except BaseException as exc:
        state.update(status='stopped', error=repr(exc), updated_utc=now())
        write(output / 'supervision.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'extract', 'train'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.mode](args.output.resolve())
    if isinstance(result, int):
        raise SystemExit(result)
