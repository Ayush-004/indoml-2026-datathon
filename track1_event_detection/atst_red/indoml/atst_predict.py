"""Predict from a calibrated full-encoder ATST checkpoint and unlabeled audio.

The JSONL manifest needs only clip_id, audio_path and measured duration. The
trained checkpoint contains the complete GRU and encoder state; its original
training manifests, feature cache, initial GRU and probability archives are not
opened. Native source and released encoder paths may be relocated with overrides.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .atst_finetune import FineTuneATST, archive_probabilities, infer, load_checkpoint
from .data import resolve_audio_path
from .model import choose_device
from .pretrained import FrozenATST, FRAME_SAMPLES, SOURCE_FILES, file_sha256
from .submission import validate_predictions
from .train import write_json
from .transfer import CATEGORIES, TemporalHead, decoded_record, write_predictions


def require(condition, message):
    if not condition:
        raise ValueError(message)


class PredictionClips:
    """Read only audio identity and geometry; ignore labels and feature fields."""
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.rows = []
        seen = set()
        for number, line in enumerate(self.manifest.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            context = f'{self.manifest}:{number}'
            require(isinstance(value, dict), f'{context}: expected a JSON object')
            require(all(isinstance(value.get(key), str) and value[key].strip()
                        for key in ('clip_id', 'audio_path')), f'{context}: clip_id and audio_path are required')
            duration = value.get('duration')
            require(not isinstance(duration, bool) and isinstance(duration, (int, float))
                    and math.isfinite(duration) and duration > 0, f'{context}: duration must be finite and positive')
            require(value['clip_id'] not in seen, f'{context}: duplicate clip_id')
            seen.add(value['clip_id'])
            samples = round(duration * 16000)
            require(samples > 0, f'{context}: duration is shorter than one sample')
            frames = (samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES
            grid = {'frame_hop': .04, 'frame_offset': .02, 'num_frames': frames}
            require(all(name not in value or value[name] == expected for name, expected in grid.items()),
                    f'{context}: supplied frame geometry differs from the native 40 ms grid')
            self.rows.append({'clip_id': value['clip_id'], 'audio_path': str(resolve_audio_path(value, self.manifest).resolve()),
                              'duration': duration, **grid})
        require(bool(self.rows), 'Prediction manifest must contain at least one clip')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PredictionATST(FineTuneATST):
    """Reuse the trained forward path without reopening its initial private GRU."""
    def __init__(self, source_root, encoder_checkpoint, checkpoint):
        nn.Module.__init__(self)
        config = checkpoint['model_config']
        require(set(config) == {'train_blocks', 'head_config'}, 'Unsupported full-encoder model configuration')
        require(type(config['train_blocks']) is int and 1 <= config['train_blocks'] <= 12,
                'Invalid trained encoder block configuration')
        require(config['head_config'].get('feature_dim') == 768 and config['head_config'].get('head_type') == 'gru',
                'Expected a trained 768-dimensional GRU head')
        adapter = FrozenATST(source_root, encoder_checkpoint, device='cpu')
        expected = checkpoint['training']['native_provenance']
        # A copied source tree can lose Git metadata while retaining identical code.
        # Device is a runtime choice; all remaining recorded native semantics match.
        ignored = {'source_commit', 'device', 'include_logits'}
        require(all(adapter.provenance.get(key) == value for key, value in expected.items() if key not in ignored),
                'Native source, adapter, runtime, checkpoint or frame semantics differ from the trained model')
        self.frontend = adapter.wrapper.atst_mel
        self.encoder = adapter.wrapper.atst
        self.provenance = adapter.provenance
        self.head = TemporalHead(**config['head_config'])
        self.train_blocks = config['train_blocks']
        self.device = torch.device('cpu')
        self.head_device = torch.device('cpu')
        self.load_state_dict(checkpoint['model_state'], strict=True)
        require(all(torch.isfinite(value).all().item() for value in self.state_dict().values()),
                'Trained checkpoint contains nonfinite model tensors')
        self.requires_grad_(False).eval()


def predict(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    checkpoint_path = Path(args.checkpoint).resolve()
    payload = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(payload).hexdigest()
    require(args.checkpoint_sha256 is None or checkpoint_hash == args.checkpoint_sha256,
            'Full-encoder checkpoint SHA-256 differs from the requested identity')
    checkpoint = load_checkpoint(io.BytesIO(payload))
    del payload
    require(isinstance(checkpoint.get('decoder'), dict), 'Use a calibrated best/selected checkpoint, not optimizer last.pt')
    decoder = checkpoint['decoder']
    require(set(decoder) == {'threshold', 'min_duration', 'merge_gap'}, 'Unsupported saved ATST decoder configuration')
    require(all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                for value in decoder.values()) and 0 < decoder['threshold'] < 1
            and decoder['min_duration'] >= 0 and decoder['merge_gap'] >= 0, 'Invalid saved decoder values')
    selected = checkpoint.get('selection')
    if selected:
        require(selected['kind'] == 'finetuned_epoch' and selected['epoch'] == checkpoint['epoch']
                and selected['decoder'] == decoder and selected['metrics'] == checkpoint['validation'],
                'Checkpoint selection, decoder and epoch disagree')
    initial = checkpoint['training']['initialization']
    source_root = Path(args.source_root or initial['source_root']).resolve()
    encoder_checkpoint = Path(args.encoder_checkpoint or initial['encoder_checkpoint']).resolve()
    require(file_sha256(encoder_checkpoint) == initial['encoder_sha256'], 'Released encoder checkpoint SHA-256 differs')
    source_hashes = {name: file_sha256(source_root / name) for name in SOURCE_FILES}
    require(source_hashes == checkpoint['training']['native_provenance']['source_files_sha256'],
            'Native source file hashes differ from the trained checkpoint')
    data = PredictionClips(args.manifest)
    output = Path(args.output).resolve()
    sidecar = output.with_suffix(output.suffix + '.provenance.json')
    archive = Path(args.probabilities).resolve() if args.probabilities else None
    final_paths = [output, sidecar, *([archive] if archive else [])]
    require(len(set(final_paths)) == len(final_paths), 'Prediction, provenance and probability destinations must differ')
    destinations = set(final_paths)
    destinations.update(path.with_suffix(path.suffix + '.tmp') for path in final_paths)
    if archive:
        destinations.add(archive.with_suffix('.npz.tmp'))
    protected = {checkpoint_path, data.manifest, encoder_checkpoint,
                 *(source_root / name for name in SOURCE_FILES), *(Path(row['audio_path']) for row in data.rows)}
    require(not destinations & protected, 'Prediction output would overwrite an input')
    model = PredictionATST(source_root, encoder_checkpoint, checkpoint)
    require(file_sha256(encoder_checkpoint) == initial['encoder_sha256']
            and {name: file_sha256(source_root / name) for name in SOURCE_FILES} == source_hashes,
            'A native inference input changed while the model was loading')
    encoder_device, head_device = choose_device(args.encoder_device), choose_device(args.head_device)
    model.set_device(encoder_device, head_device)
    records = list(infer(model, data, args.batch_size, None))
    predictions = [decoded_record(row, values, decoder) for row, values in records]
    coverage = validate_predictions(predictions, data.rows)
    write_predictions(output, predictions)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive_probabilities(archive, records, checkpoint_hash, data.manifest, checkpoint['epoch'], decoder)
    report = {'scope': 'Inference only on unlabeled audio; no training manifests, feature caches, initial GRU, calibration or scoring',
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash, 'selected_epoch': checkpoint['epoch'],
              'manifest': str(data.manifest), 'manifest_sha256': file_sha256(data.manifest),
              'source_root': str(source_root), 'source_files_sha256': source_hashes,
              'encoder_checkpoint': str(encoder_checkpoint), 'encoder_checkpoint_sha256': initial['encoder_sha256'],
              'model_config': checkpoint['model_config'], 'categories': list(CATEGORIES), 'frame_grid': checkpoint['frame_grid'],
              'decoder': decoder, 'coverage': coverage, 'prediction_sha256': file_sha256(output),
              'probabilities': str(archive) if archive else None, 'encoder_device': str(encoder_device),
              'head_device': str(head_device), 'batch_size': args.batch_size, 'torch_version': str(torch.__version__),
              'predictor_sha256': file_sha256(Path(__file__)), 'trainer_forward_sha256': file_sha256(Path(__file__).with_name('atst_finetune.py')),
              'seconds': time.monotonic() - started}
    write_json(sidecar, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Calibrated best.pt or immutable selected checkpoint')
    parser.add_argument('--checkpoint-sha256', help='Optional exact full-checkpoint identity to require')
    parser.add_argument('--manifest', required=True, help='Unlabeled JSONL with clip_id, audio_path and duration')
    parser.add_argument('--output', required=True)
    parser.add_argument('--probabilities', help='Optional aligned FP32 probability archive for reproducibility')
    parser.add_argument('--source-root', help='Relocated exact native ATST source tree')
    parser.add_argument('--encoder-checkpoint', help='Relocated exact released ATST-F strong checkpoint')
    parser.add_argument('--encoder-device', choices=['auto', 'mps', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--head-device', choices=['auto', 'mps', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if min(args.threads, args.batch_size) <= 0:
        parser.error('threads and batch-size must be positive')
    predict(args)


if __name__ == '__main__':
    main()
