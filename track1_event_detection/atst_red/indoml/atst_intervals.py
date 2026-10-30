"""Adapt native ATST+RED with Gold intervals and optional training augmentation.

Selection uses INNER only. Every completed epoch, including the initial native
model, has an inference checkpoint; the controller owns any OUTER evaluation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .atst_finetune import FineTuneATST, prepare_clips, collate_clips, check_geography
from .atst_predict import PredictionClips
from .data import resolve_audio_path
from .intervals import (IntervalDataset, IntervalModel, interval_targets, interval_loss,
                        candidate_intervals, decode_candidates, select_decoder,
                        load_checkpoint as load_interval_checkpoint)
from .metrics import score_records
from .model import choose_device
from .pretrained import FrozenATST, FRAME_SAMPLES, WINDOW_SAMPLES, SOURCE_FILES, window_starts
from .submission import validate_predictions
from .train import check_separate_splits, rng_state, restore_rng, save_checkpoint, write_json
from .transfer import CATEGORIES, category_targets, file_hash, write_predictions

MODEL_TYPE = 'native_atst_interval_model'
FRAME_GRID = {'frame_hop': .04, 'frame_offset': .02}
GRID = {'thresholds': [.1, .2, .35, .5, .65, .8, .9, .95],
        'nms_ious': [0., .3, .5], 'min_duration': .08}
PROJECT_SOURCES = ('atst_intervals.py', 'atst_finetune.py', 'atst_predict.py',
                   'intervals.py', 'pretrained.py', 'mobile.py', 'transfer.py',
                   'train.py', 'model.py', 'data.py', 'metrics.py', 'submission.py',
                   'weak_supervision.py', 'frequency_warp.py', 'optimization.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in PROJECT_SOURCES}


def state_digest(state):
    """Hash names, geometry and tensor bytes without retaining a second model."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def check_head_config(config):
    require(isinstance(config, dict) and config.get('mode') == 'red'
            and config.get('feature_dim', 768) == 768
            and config.get('epn_context', 'per_class') == 'per_class',
            'Expected a trained 768-dimensional per-class RED interval head')


class NativeIntervalATST(FineTuneATST):
    """Reuse native differentiable stitching; replace only the downstream head."""
    def __init__(self, source_root, encoder_checkpoint, head_checkpoint, train_blocks=12):
        nn.Module.__init__(self)
        checkpoint = load_interval_checkpoint(head_checkpoint)
        check_head_config(checkpoint['model_config'])
        require(checkpoint.get('feature_grid') == FRAME_GRID, 'RED checkpoint frame grid differs')
        native = FrozenATST(source_root, encoder_checkpoint, device='cpu')
        require(checkpoint.get('feature_cache_id') == native.cache_id,
                'RED head and native encoder cache/source/runtime semantics differ')
        self._assemble(native, checkpoint['model_config'], train_blocks)
        self.head.load_state_dict(checkpoint['model_state'], strict=True)
        require(all(torch.isfinite(value).all().item() for value in self.head.state_dict().values()),
                'Initial RED checkpoint contains nonfinite tensors')
        self.head_checkpoint = checkpoint

    def _assemble(self, native, head_config, train_blocks):
        require(type(train_blocks) is int and train_blocks in (0, 12),
                'train_blocks must be 0 (frozen control) or 12 (full encoder)')
        check_head_config(head_config)
        self.frontend = native.wrapper.atst_mel
        self.encoder = native.wrapper.atst
        self.provenance = native.provenance
        self.initial_feature_cache_id = native.cache_id
        self.head = IntervalModel(**head_config)
        self.frontend.requires_grad_(False).eval()
        self.encoder.requires_grad_(train_blocks == 12).eval()
        # The release never uses this parameter in get_intermediate_layers.
        self.encoder.mask_embed.requires_grad_(False)
        self.train_blocks = train_blocks
        self.device = torch.device('cpu')
        self.head_device = torch.device('cpu')
        # Training-only state; no parameters/buffers and no inference dependency.
        self.frequency_augmentation = None

    @classmethod
    def from_checkpoint(cls, source_root, encoder_checkpoint, checkpoint):
        value = cls.__new__(cls)
        nn.Module.__init__(value)
        config = checkpoint['model_config']
        require(set(config) == {'train_blocks', 'head_config'}, 'Unexpected native RED configuration')
        native = FrozenATST(source_root, encoder_checkpoint, device='cpu')
        ignored = {'source_commit', 'device', 'include_logits'}
        expected = checkpoint['training']['native_provenance']
        require(all(native.provenance.get(k) == v for k, v in expected.items() if k not in ignored),
                'Native source, runtime, initial checkpoint or temporal semantics differ')
        value._assemble(native, config['head_config'], config['train_blocks'])
        # A relocated source tree may omit .git. The original cache identity is
        # historical; exact source/runtime/weight semantics were checked above.
        value.initial_feature_cache_id = checkpoint['initial_feature_cache_id']
        value.load_state_dict(checkpoint['model_state'], strict=True)
        require(all(torch.isfinite(tensor).all().item() for tensor in value.state_dict().values()),
                'Native RED checkpoint contains nonfinite tensors')
        return value

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # Evaluation mode does not disable gradients. Both arms use identical
        # deterministic native encoder semantics; only parameter updates differ.
        self.frontend.eval()
        self.encoder.eval()
        return self

    def encode_clips(self, windows, descriptions):
        augmentation = self.frequency_augmentation
        if not self.training or augmentation is None or augmentation.probability == 0:
            return super().encode_clips(windows, descriptions)
        # Keep the original path untouched for prediction and the control. The
        # augmented path changes only CPU mel values before the native encoder.
        if windows.device.type != 'cpu' or windows.ndim != 2 or windows.shape[1] != WINDOW_SAMPLES:
            raise ValueError('Expected CPU padded ten-second windows')
        with torch.no_grad():
            spec = torch.cat([self.frontend(window[None]) for window in windows], dim=0)
            if spec.shape != (len(windows), 1, 64, 1001):
                raise ValueError('Native ATST frontend geometry changed')
            spec = augmentation.apply(spec, descriptions)
        spec = spec.to(self.device)
        features = self.encoder.get_intermediate_layers(
            spec, torch.full((len(spec),), 1001, device=self.device), 1, scene=False)
        if features.shape != (len(windows), 250, 768):
            raise ValueError('Native ATST encoder geometry changed')
        clips = []
        for description in descriptions:
            parts = []
            for section in description['sections']:
                first, count = section['first'], section['count']
                value = features[section['window'], :count] * section['weight'].to(self.device)[:, None]
                parts.append(F.pad(value, (0, 0, first, description['frames'] - first - count)))
            clips.append(torch.stack(parts).sum(dim=0))
        return clips

    def forward(self, windows, descriptions, return_features=False):
        clips = self.encode_clips(windows, descriptions)
        lengths = torch.tensor([len(clip) for clip in clips], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(clips, batch_first=True).to(self.head_device)
        outputs = self.head(padded, lengths)
        return (outputs, clips) if return_features else outputs


class AudioIntervals(IntervalDataset):
    """Validate the original Gold feature manifest; read raw audio when batching."""
    def __init__(self, manifest, mode, test_only_clips=None):
        super().__init__(manifest, mode=mode)
        require(not self.excluded, 'Use the already-clean Gold manifest; no rows may be silently excluded')
        require((self.frame_hop, self.frame_offset) == (.04, .02), 'Native frame grid differs')
        for row in self.rows:
            require(row['quality'] == 'verified_timestamps' and row['events'] is not None,
                    'Native adaptation requires clean Gold timestamps')
            samples = round(row['duration'] * 16000)
            require(row['num_frames'] == (samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES,
                    'Raw-audio and feature frame counts differ')
            row['audio_path'] = str(resolve_audio_path(row, self.manifest).resolve())
        if test_only_clips is not None:
            self.rows = self.rows[:test_only_clips]
        self.num_windows = sum(len(window_starts(round(r['duration'] * 16000))) for r in self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_audio_intervals(rows):
    windows, descriptions = prepare_clips(rows)
    lengths = torch.tensor([item['frames'] for item in descriptions], dtype=torch.long)
    labels = [interval_targets(row) for row in rows]
    targets = [category_targets(row) for row in rows]
    require(all(torch.equal(target.bool(), label['event_weights'] > 0)
                for target, label in zip(targets, labels)), 'Interval assignment changed category targets')
    batch = {'windows': windows, 'descriptions': descriptions, 'rows': rows,
             'lengths': lengths, 'frames': int(lengths.sum()),
             'targets': nn.utils.rnn.pad_sequence(targets, batch_first=True),
             'mask': torch.arange(int(lengths.max()))[None] < lengths[:, None]}
    for key in ('distances', 'event_weights', 'onsets', 'offsets'):
        batch[key] = nn.utils.rnn.pad_sequence([label[key] for label in labels], batch_first=True)
    return batch


@torch.inference_mode()
def infer_outputs(model, dataset, batch_size):
    model.eval()
    # Validation/prediction must not advance the training or shuffle RNG streams.
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_clips, generator=generator)
    for batch in loader:
        outputs = {key: tensor.detach().float().cpu().numpy()
                   for key, tensor in model(batch['windows'], batch['descriptions']).items()}
        require(all(np.isfinite(value).all() for value in outputs.values()), 'Nonfinite native RED outputs')
        for index, description in enumerate(batch['descriptions']):
            yield description['row'], {key: value[index, :description['frames']] for key, value in outputs.items()}


def infer_candidates(model, dataset, batch_size):
    return [(row, candidate_intervals(row, values['presence'], values['distances']))
            for row, values in infer_outputs(model, dataset, batch_size)]


def decode_records(records, decoder):
    return [decode_candidates(row, candidates, decoder) for row, candidates in records]


def validate_decoder(decoder):
    require(isinstance(decoder, dict) and set(decoder) == {'threshold', 'nms_iou', 'min_duration'},
            'Expected a calibrated RED interval decoder')
    require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                for v in decoder.values()) and 0 < decoder['threshold'] < 1
            and 0 <= decoder['nms_iou'] < 1 and decoder['min_duration'] >= 0,
            'Invalid RED decoder values')


def load_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(value, dict) and value.get('format_version') == 1
            and value.get('model_type') == MODEL_TYPE and value.get('categories') == list(CATEGORIES)
            and value.get('frame_grid') == FRAME_GRID, 'Not a compatible native RED checkpoint')
    require(value.get('checkpoint_kind') in ('inference', 'resume'), 'Unknown native RED checkpoint kind')
    require(type(value.get('epoch')) is int and value['epoch'] >= 0, 'Invalid checkpoint epoch')
    check_head_config(value['model_config']['head_config'])
    if value['checkpoint_kind'] == 'inference':
        validate_decoder(value.get('decoder'))
        require(isinstance(value.get('validation'), dict), 'Inference checkpoint has no INNER metrics')
        require('optimizer_state' not in value and 'rng_state' not in value, 'Inference checkpoint carries optimizer state')
        require('frequency_warp_state' not in value, 'Inference checkpoint carries training augmentation state')
        require('optimization_state' not in value, 'Inference checkpoint carries optimizer schedule state')
    return value


def snapshot(model, epoch, metadata, decoder, metrics):
    return {'format_version': 1, 'model_type': MODEL_TYPE, 'checkpoint_kind': 'inference',
            'categories': list(CATEGORIES), 'frame_grid': FRAME_GRID, 'epoch': epoch,
            'model_config': {'train_blocks': model.train_blocks, 'head_config': model.head.config},
            'model_state': {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
            'initial_feature_cache_id': model.initial_feature_cache_id,
            'training': metadata, 'decoder': decoder, 'validation': metrics}


def audio_inventory(dataset):
    return [{'clip_id': row['clip_id'], 'audio_path': str(Path(row['audio_path']).resolve()),
             'duration': row['duration'], 'num_frames': row['num_frames'],
             'sha256': file_hash(Path(row['audio_path']))} for row in dataset.rows]


def inventory_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def copy_checkpoint(source, destination, expected_sha256, *, replace=False):
    """Publish an exact file, using a hard link when both outputs share a volume."""
    source, destination = Path(source), Path(destination)
    require(file_hash(source) == expected_sha256, 'Bound checkpoint bytes changed')
    if destination.exists():
        if file_hash(destination) == expected_sha256:
            return
        require(replace, f'Refusing to replace a retained checkpoint: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        with source.open('rb') as src, temporary.open('wb') as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(chunk)
    require(file_hash(temporary) == expected_sha256, 'Checkpoint copy changed bytes')
    temporary.replace(destination)


def rng_device(model):
    return torch.device('mps') if 'mps' in (model.device.type, model.head_device.type) else torch.device('cpu')


def seed_training(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def dropout_rng_state(device):
    state = {'torch_cpu': torch.get_rng_state()}
    if device.type == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_dropout_rng(state, device):
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'mps':
        torch.mps.set_rng_state(state['mps'])


class AuxiliaryStream:
    """Independent permutations, full-batch wrapping, and isolated dropout RNG."""
    def __init__(self, dataset, batch_size, shuffle_seed, training_rng_seed, device):
        require(len(dataset) > 0 and type(batch_size) is int and batch_size > 0,
                'Auxiliary pool and batch size must be positive')
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = torch.device(device)
        require(self.device.type in ('cpu', 'mps'), 'Auxiliary RNG supports CPU and MPS only')
        self.shuffle_seed, self.training_rng_seed = shuffle_seed, training_rng_seed
        self.base_order_sha256 = inventory_digest([row['clip_id'] for row in dataset.rows])
        self.generator = torch.Generator().manual_seed(shuffle_seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.cursor, self.cycles, self.batches_consumed, self.examples_consumed = 0, 0, 0, 0
        self.dropout_state = {'torch_cpu': torch.Generator().manual_seed(training_rng_seed).get_state()}
        if self.device.type == 'mps':
            original = dropout_rng_state(self.device)
            try:
                torch.mps.manual_seed(training_rng_seed)
                self.dropout_state['mps'] = torch.mps.get_rng_state()
            finally:
                restore_dropout_rng(original, self.device)

    def take(self):
        indices = []
        while len(indices) < self.batch_size:
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.dataset), generator=self.generator)
                self.cursor = 0
                self.cycles += 1
            count = min(self.batch_size - len(indices), len(self.order) - self.cursor)
            indices.extend(self.order[self.cursor:self.cursor + count].tolist())
            self.cursor += count
        self.batches_consumed += 1
        self.examples_consumed += len(indices)
        return [self.dataset[index] for index in indices]

    @contextmanager
    def use_rng(self):
        gold_state = dropout_rng_state(self.device)
        restore_dropout_rng(self.dropout_state, self.device)
        try:
            yield
        finally:
            self.dropout_state = dropout_rng_state(self.device)
            restore_dropout_rng(gold_state, self.device)

    def state_dict(self):
        return {'version': 1, 'pool_size': len(self.dataset), 'batch_size': self.batch_size,
                'base_order_sha256': self.base_order_sha256, 'device': self.device.type,
                'shuffle_seed': self.shuffle_seed, 'training_rng_seed': self.training_rng_seed,
                'order': self.order.clone(), 'cursor': self.cursor, 'cycles': self.cycles,
                'batches_consumed': self.batches_consumed, 'examples_consumed': self.examples_consumed,
                'shuffle_generator': self.generator.get_state(),
                'dropout_rng': {key: value.clone() for key, value in self.dropout_state.items()}}

    def load_state_dict(self, state):
        expected = self.state_dict()
        require(isinstance(state, dict) and state.keys() == expected.keys(), 'Incomplete auxiliary stream state')
        for key in ('version', 'pool_size', 'batch_size', 'base_order_sha256', 'device', 'shuffle_seed', 'training_rng_seed'):
            require(state[key] == expected[key], f'Auxiliary stream policy differs: {key}')
        order = state['order']
        require(isinstance(order, torch.Tensor) and order.dtype == torch.long and order.device.type == 'cpu'
                and order.shape == (len(self.dataset),)
                and torch.equal(order.sort().values, torch.arange(len(self.dataset))),
                'Auxiliary permutation is invalid')
        for key in ('cursor', 'cycles', 'batches_consumed', 'examples_consumed'):
            require(type(state[key]) is int and state[key] >= 0, f'Invalid auxiliary counter: {key}')
        require(state['cursor'] <= len(self.dataset)
                and state['examples_consumed'] == state['batches_consumed'] * self.batch_size
                and state['examples_consumed'] == state['cycles'] * len(self.dataset) + state['cursor'],
                'Auxiliary cursor and exposure counters disagree')
        require(state['dropout_rng'].keys() == self.dropout_state.keys(), 'Auxiliary dropout RNG devices differ')
        for key, value in state['dropout_rng'].items():
            require(isinstance(value, torch.Tensor) and value.dtype == self.dropout_state[key].dtype
                    and value.shape == self.dropout_state[key].shape, 'Invalid auxiliary dropout RNG state')
        self.generator.set_state(state['shuffle_generator'])
        self.order = order.clone()
        self.cursor, self.cycles = state['cursor'], state['cycles']
        self.batches_consumed, self.examples_consumed = state['batches_consumed'], state['examples_consumed']
        self.dropout_state = {key: value.clone() for key, value in state['dropout_rng'].items()}


def auxiliary_enabled(args):
    return bool(getattr(args, 'aux_manifest', None) and getattr(args, 'aux_weight', 0.) > 0)


WARP_COUNTS = ('batches', 'clips', 'windows', 'applied_clips', 'applied_windows',
               'feature_frames', 'applied_feature_frames', 'mel_time_frames', 'applied_mel_time_frames')
EMPTY_CHOICE_SHA256 = hashlib.sha256(b'').hexdigest()


def frequency_warp_enabled(args):
    return getattr(args, 'frequency_warp_probability', 0.) > 0


def validate_frequency_warp_args(args):
    probability = getattr(args, 'frequency_warp_probability', 0.)
    minimum = getattr(args, 'frequency_warp_min_scale', .6)
    maximum = getattr(args, 'frequency_warp_max_scale', 1.)
    seed = getattr(args, 'frequency_warp_seed', 48)
    require(all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                for value in (probability, minimum, maximum))
            and 0 <= probability <= 1 and 0 < minimum <= maximum <= 1,
            'Frequency warp requires probability in [0,1] and 0 < min_scale <= max_scale <= 1')
    require(type(seed) is int and 0 <= seed < 2 ** 63, 'Frequency-warp seed must be a nonnegative signed 64-bit integer')
    require(not probability or getattr(args, 'aux_weight', 0.) == 0,
            'The frequency-warp study requires auxiliary supervision OFF')


def warp_decision_bytes(decision):
    """Canonical ordered record shared by epoch and resumable cumulative audits."""
    return json.dumps(decision, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def frequency_warp_policy(args):
    return {
        'kind': 'frequency_crop_resize', 'version': 1,
        'probability': getattr(args, 'frequency_warp_probability', 0.),
        'min_scale': getattr(args, 'frequency_warp_min_scale', .6),
        'max_scale': getattr(args, 'frequency_warp_max_scale', 1.),
        'seed': getattr(args, 'frequency_warp_seed', 48),
        'sampling': 'Independent CPU Torch generator; float64 uniform application and scale draws per Gold clip; height=floor(64*scale) clamped to [1,64]; inclusive integer offset sampled only when height<64',
        'scope': 'One application choice and crop per original Gold clip, shared across all its native windows; auxiliary OFF',
        'placement': 'CPU normalized native mel after independent per-window frontend, before encoder transfer',
        'geometry': {'mel_bins': 64, 'mel_time_frames': 1001, 'virtual_crop_scale': [1., 1.], 'time_scale': [1., 1.]},
        'interpolation': 'bicubic', 'align_corners': True, 'post_interpolation_clamp': False,
        'evaluation': 'Original native path; no augmentation or private RNG draws',
        'choice_hash': 'Per-epoch SHA256 of canonical sorted-key JSON records plus newline in clip order; cumulative chain SHA256(previous digest bytes + next record), initialized as SHA256(empty)'}


class FrequencyWarpStream:
    """Private CPU randomness and exposure accounting; absent from model state."""
    def __init__(self, probability=.5, min_scale=.6, max_scale=1., seed=48):
        values = argparse.Namespace(frequency_warp_probability=probability,
                                    frequency_warp_min_scale=min_scale, frequency_warp_max_scale=max_scale,
                                    frequency_warp_seed=seed, aux_weight=0.)
        validate_frequency_warp_args(values)
        self.probability, self.min_scale, self.max_scale, self.seed = probability, min_scale, max_scale, seed
        self.policy = frequency_warp_policy(values)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.counts = dict.fromkeys(WARP_COUNTS, 0)
        self.choice_chain_sha256 = EMPTY_CHOICE_SHA256
        self.last_audit = None

    def apply(self, spec, descriptions):
        # The helper is a training dependency only. Prediction reconstructs the
        # unmodified model without importing it or constructing this stream.
        from .frequency_warp import frequency_warp
        transformed, audit = frequency_warp(spec, descriptions, generator=self.generator,
                                            probability=self.probability, min_scale=self.min_scale,
                                            max_scale=self.max_scale)
        require(len(audit['decisions']) == audit['clips'] == len(descriptions)
                and audit['windows'] == len(spec), 'Frequency-warp audit coverage differs')
        if self.probability == 0:
            self.last_audit = None
            return transformed
        self.counts['batches'] += 1
        for key in WARP_COUNTS[1:]:
            require(type(audit.get(key)) is int and audit[key] >= 0, f'Invalid frequency-warp audit count: {key}')
            self.counts[key] += audit[key]
        for decision in audit['decisions']:
            self.choice_chain_sha256 = hashlib.sha256(
                bytes.fromhex(self.choice_chain_sha256) + warp_decision_bytes(decision)).hexdigest()
        self.last_audit = audit
        return transformed

    def summary(self):
        return {**self.counts, 'choice_chain_sha256': self.choice_chain_sha256}

    def state_dict(self):
        return {'format_version': 1, 'policy': self.policy, 'generator_state': self.generator.get_state().clone(),
                'counts': dict(self.counts), 'choice_chain_sha256': self.choice_chain_sha256}

    def load_state_dict(self, state):
        require(isinstance(state, dict) and set(state) == {'format_version', 'policy', 'generator_state',
                                                        'counts', 'choice_chain_sha256'}
                and state['format_version'] == 1 and state['policy'] == self.policy,
                'Frequency-warp resume policy or state differs')
        counts = state['counts']
        require(isinstance(counts, dict) and set(counts) == set(WARP_COUNTS)
                and all(type(value) is int and value >= 0 for value in counts.values()),
                'Invalid frequency-warp resume counts')
        require(counts['batches'] <= counts['clips'] <= counts['windows']
                and counts['clips'] <= counts['feature_frames']
                and counts['mel_time_frames'] == 1001 * counts['windows']
                and counts['applied_mel_time_frames'] == 1001 * counts['applied_windows']
                and all(counts['applied_' + key] <= counts[key]
                        for key in ('clips', 'windows', 'feature_frames', 'mel_time_frames'))
                and counts['applied_clips'] <= counts['applied_windows']
                and (counts['batches'] > 0 or not any(counts.values())),
                'Inconsistent frequency-warp resume counts')
        digest = state['choice_chain_sha256']
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
                and (counts['clips'] > 0 or digest == EMPTY_CHOICE_SHA256), 'Invalid frequency-warp choice digest')
        rng = state['generator_state']
        expected = self.generator.get_state()
        require(isinstance(rng, torch.Tensor) and rng.device.type == 'cpu' and rng.dtype == torch.uint8
                and rng.shape == expected.shape, 'Invalid frequency-warp generator state')
        try:
            checked = torch.Generator(device='cpu')
            checked.set_state(rng)
        except (RuntimeError, TypeError) as exc:
            raise ValueError('Invalid frequency-warp generator state') from exc
        self.generator.set_state(checked.get_state())
        self.counts = dict(counts)
        self.choice_chain_sha256 = digest
        self.last_audit = None


def initialize_model(args):
    """A new stage can bind full native weights without importing optimizer/RNG."""
    parent_path = getattr(args, 'initial_native_checkpoint', None)
    parent = None
    if parent_path:
        payload = Path(parent_path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == args.initial_native_sha256, 'Initial native checkpoint SHA-256 differs')
        parent = load_checkpoint(io.BytesIO(payload))
        del payload
        require(parent['checkpoint_kind'] == 'inference', 'Native initialization requires a calibrated inference checkpoint')
        require(parent['training'].get('policy', {}).get('test_only_clips') is None
                or getattr(args, 'test_only_clips', None) is not None,
                'A test-only native parent cannot initialize a production training stage')
        old_sources = parent['training']['source_sha256']
        current_sources = source_hashes()
        # These two files describe the new training stage. All shared native,
        # RED-head, data and scorer implementations must retain their semantics.
        ignored = {'atst_intervals.py', 'weak_supervision.py', 'optimization.py'}
        # This training-only helper did not exist in the original native parent.
        # Existing parent bindings (including a later helper binding) stay strict.
        added_sources = ({'frequency_warp.py'} & set(current_sources)) - set(old_sources)
        require(all(old_sources.get(key) == value for key, value in current_sources.items()
                    if key not in ignored and key not in added_sources),
                'Parent native model and current forward/data implementations differ')
        model = NativeIntervalATST.from_checkpoint(args.source_root, args.encoder_checkpoint, parent)
        require(type(args.train_blocks) is int and args.train_blocks in (0, 12), 'Invalid new-stage encoder trainability')
        model.encoder.requires_grad_(args.train_blocks == 12).eval()
        model.encoder.mask_embed.requires_grad_(False)
        model.train_blocks = args.train_blocks
        # Keep only selection/data metadata, not another CPU copy of the encoder.
        model.head_checkpoint = {key: parent[key] for key in ('epoch', 'decoder', 'validation', 'training')}
        require(file_hash(parent_path) == digest, 'Native parent changed while loading')
    else:
        model = NativeIntervalATST(args.source_root, args.encoder_checkpoint, args.head_checkpoint, args.train_blocks)
    initial = {'encoder_checkpoint': str(Path(args.encoder_checkpoint).resolve()),
               'encoder_sha256': file_hash(args.encoder_checkpoint), 'source_root': str(Path(args.source_root).resolve()),
               'head_state_sha256': state_digest(model.head.state_dict()),
               'encoder_state_sha256': state_digest(model.encoder.state_dict())}
    if parent is None:
        initial.update(head_checkpoint=str(Path(args.head_checkpoint).resolve()),
                       head_sha256=file_hash(args.head_checkpoint), head_epoch=model.head_checkpoint['epoch'])
    else:
        initial.update(kind='native_weights_new_stage', parent_checkpoint=str(Path(parent_path).resolve()),
                       parent_checkpoint_sha256=digest, parent_epoch=parent['epoch'],
                       parent_decoder=parent['decoder'], parent_validation=parent['validation'],
                       parent_model_config=parent['model_config'], parent_train_blocks=parent['model_config']['train_blocks'],
                       new_train_blocks=args.train_blocks, parent_source_sha256=old_sources,
                       parent_native_provenance=parent['training']['native_provenance'],
                       parent_initialization=parent['training']['initialization'],
                       parent_manifest_sha256=parent['training']['manifest_sha256'],
                       fresh_optimizer=True, fresh_training_rng=True)
        if added_sources:
            initial['added_training_source_sha256'] = {key: current_sources[key] for key in sorted(added_sources)}
    return model, initial


def train_step(model, batch, positive_weights, optimizer, parameters, args, *, auxiliary=None, global_update=1,
               observe_update=False):
    """One Gold update; optional second backward before one clip and one step."""
    require(type(global_update) is int and global_update > 0, 'Gold update numbers start at one')
    augmentation = getattr(model, 'frequency_augmentation', None)
    augmented = augmentation is not None and augmentation.probability > 0
    if augmented:
        require(model.training and auxiliary is None and getattr(args, 'aux_weight', 0.) == 0,
                'Frequency augmentation is enabled only for isolated Gold training')
        previous_warp_batches = augmentation.counts['batches']
        augmentation.last_audit = None
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch['windows'], batch['descriptions'])
    if augmented:
        require(augmentation.counts['batches'] == previous_warp_batches + 1 and augmentation.last_audit is not None,
                'Each Gold update must consume exactly one frequency-warp batch')
    loss, components = interval_loss(outputs, batch, positive_weights, model.head_device,
                                     'red', args.boundary_weight, args.iou_weight, collar_weight=0.)
    require(bool(torch.isfinite(loss).item()), 'Nonfinite native RED loss')
    loss.backward()
    auxiliary_stats = None
    if auxiliary is not None and getattr(args, 'aux_weight', 0.) > 0 and global_update % args.aux_every == 0:
        from .weak_supervision import positive_category_mask, weak_positive_loss
        gold_loss_value = float(loss.detach().cpu())
        # Backward has released saved tensors; remove remaining references before
        # building the second native graph. Parameter gradients stay accumulated.
        del loss, outputs
        rows = auxiliary.take()
        auxiliary_batch = collate_clips(rows)
        lengths = torch.tensor([item['frames'] for item in auxiliary_batch['descriptions']], dtype=torch.long)
        mask = positive_category_mask([row['positive_categories'] for row in rows], device=model.head_device)
        with auxiliary.use_rng():
            presence = model(auxiliary_batch['windows'], auxiliary_batch['descriptions'])['presence']
            weak_loss, auxiliary_stats = weak_positive_loss(presence, lengths, mask)
            require(bool(torch.isfinite(weak_loss).item()), 'Nonfinite weak-positive loss')
            weighted = args.aux_weight * weak_loss
            weighted.backward()
            auxiliary_stats = {**auxiliary_stats, 'raw_loss': float(weak_loss.detach().cpu()),
                               'weighted_loss': float(weighted.detach().cpu()), 'clips': len(rows),
                               'clip_ids': [row['clip_id'] for row in rows]}
        del presence, weak_loss, weighted, auxiliary_batch
    else:
        gold_loss_value = None
    observation = None
    if observe_update:
        from .optimization import UpdateObservation
        observation = UpdateObservation(model)
    norm = nn.utils.clip_grad_norm_(parameters, args.gradient_clip, error_if_nonfinite=True)
    if observation is not None:
        observation.after_clip()
    optimizer.step()
    if gold_loss_value is None:
        gold_loss_value = float(loss.detach().cpu())
    result = {'gold_loss': gold_loss_value, 'gold_components': components,
              'gradient_norm': norm, 'auxiliary': auxiliary_stats}
    if augmented:
        result['augmentation'] = augmentation.last_audit
    if observation is not None:
        result['update_diagnostics'] = observation.finish()
    return result


def make_optimizer(model, args):
    groups = []
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder:
        groups.append({'params': encoder, 'lr': args.encoder_lr, 'name': 'encoder'})
    groups.append({'params': model.head.parameters(), 'lr': args.head_lr, 'name': 'head'})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def check_resume_policy(saved, metadata, args, *, checkpoint_sha256=None, historical=False):
    require(saved.get('checkpoint_kind') == 'resume' and 'optimizer_state' in saved
            and 'rng_state' in saved, 'Resume requires a full optimizer checkpoint')
    old = saved['training']
    from .optimization import legacy_checkpoint_allowed
    legacy = legacy_checkpoint_allowed(saved, metadata, checkpoint_sha256, historical=historical)
    for key in ('manifest_sha256', 'initialization', 'native_provenance', 'source_sha256',
                'audio_inventory_sha256', 'initial_feature_cache_id', 'policy'):
        if legacy and key in ('source_sha256', 'policy'):
            continue
        require(key in old and key in metadata and old[key] == metadata[key],
                f'Resume policy differs: {key}')
    if metadata['policy'].get('optimization') and not legacy:
        require(old.get('continuation_contract') == metadata['continuation_contract'],
                'Resume continuation contract differs')
        require(isinstance(saved.get('optimization_state'), dict), 'Resume requires saved optimization state')
    elif not legacy:
        require('optimization_state' not in saved, 'Inactive policy carries optimization state')
    if metadata['policy'].get('auxiliary') is not None:
        require('auxiliary' in old and 'auxiliary' in metadata and old['auxiliary'] == metadata['auxiliary'],
                'Resume auxiliary data/provenance differs')
        require(isinstance(saved.get('auxiliary_state'), dict) and type(saved.get('global_updates')) is int,
                'Resume requires auxiliary permutation, cursor and RNG state')
    else:
        require(old.get('auxiliary') is None and metadata.get('auxiliary') is None
                and saved.get('auxiliary_state') is None, 'Inactive auxiliary policy carries auxiliary state')
    warp_policy = metadata['policy'].get('frequency_warp')
    if warp_policy is not None:
        require(isinstance(saved.get('frequency_warp_state'), dict)
                and saved['frequency_warp_state'].get('policy') == warp_policy,
                'Resume requires the bound private frequency-warp RNG and counters')
    else:
        require(saved.get('frequency_warp_state') is None, 'Inactive frequency-warp policy carries augmentation state')
    require(args.epochs >= saved['epoch'], 'Requested horizon precedes the completed checkpoint')
    return legacy


def training_policy(args):
    policy = {key: getattr(args, key) for key in (
        'train_blocks', 'seed', 'training_rng_seed', 'shuffle_seed', 'batch_size',
        'inference_batch_size', 'threads', 'encoder_device', 'head_device', 'encoder_lr',
        'head_lr', 'weight_decay', 'pos_weight_cap', 'boundary_weight', 'iou_weight',
        'gradient_clip', 'test_only_clips')} | {
            'epn_context': 'per_class', 'collar_weight': 0., 'decoder_grid': GRID,
            'encoder_mode': 'eval in both arms; autograd and full parameter updates in the candidate',
            'head_mode': 'train during optimization, eval during inference',
            'selection': 'Epoch zero and completed epochs; INNER only; earliest epoch wins score ties'}
    if auxiliary_enabled(args):
        policy['auxiliary'] = {
            'objective': 'Mean declared-category -log(max real-frame presence), then mean supervised clips',
            'weight': args.aux_weight, 'every_gold_updates': args.aux_every,
            'cadence': 'Global completed Gold optimizer updates; auxiliary when the next update number is divisible by the frequency',
            'batch_size': args.aux_batch_size or args.batch_size,
            'shuffle_seed': args.aux_shuffle_seed, 'training_rng_seed': args.aux_training_rng_seed,
            'sampling': 'Independent permutation over the fixed base order; full batches wrap into a newly shuffled cycle',
            'normalization': 'Gold unchanged; auxiliary categories per clip, then supervised clips; one global clip and optimizer step',
            'dropout_rng': 'Independent CPU/MPS state; Gold RNG restored after each auxiliary forward/backward'}
    if frequency_warp_enabled(args):
        policy['frequency_warp'] = frequency_warp_policy(args)
    return policy


def fit(args):
    validate_frequency_warp_args(args)
    torch.set_num_threads(args.threads)
    seed_training(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(args.resume or not any((output / name).exists() for name in
                                  ('last.pt', 'best.pt', 'history.json', 'run-state.json')),
            'Use a new output directory or explicitly resume this run')
    training = AudioIntervals(args.train, 'train', args.test_only_clips)
    validation = AudioIntervals(args.val, 'val', args.test_only_clips)
    check_separate_splits(training, validation)
    check_geography(training, validation)
    model, initial = initialize_model(args)
    require(training.feature_cache_id == validation.feature_cache_id == model.initial_feature_cache_id,
            'Training, INNER and initial encoder feature identities differ')
    manifests = {'train': file_hash(training.manifest), 'validation': file_hash(validation.manifest)}
    require(model.head_checkpoint['training']['manifest_sha256'] == manifests,
            'Warm-start must use the original RED TRAIN/INNER split')
    auxiliary_data, auxiliary_metadata, auxiliary = None, None, None
    if auxiliary_enabled(args):
        from .weak_supervision import WeakAudioClips
        auxiliary_data = WeakAudioClips(args.aux_manifest,
                                       heldout_manifests=[args.val, *args.aux_heldout_manifests],
                                       order_seed=args.aux_shuffle_seed)
        auxiliary_metadata = {'manifest': str(auxiliary_data.manifest),
                              'manifest_sha256': auxiliary_data.manifest_sha256,
                              'heldout_manifest_sha256': auxiliary_data.heldout_manifest_sha256,
                              'base_order_sha256': auxiliary_data.order_sha256,
                              'base_order_seed': auxiliary_data.order_seed,
                              'base_order_version': auxiliary_data.order_version,
                              'supervision_counts': auxiliary_data.supervision_counts,
                              'heldout_use': 'Identity/geography exclusion only; no auxiliary heldout inference or scoring'}
    audio = {'train': audio_inventory(training), 'validation': audio_inventory(validation)}
    if auxiliary_data is not None:
        audio['auxiliary'] = audio_inventory(auxiliary_data)
    protected = {training.manifest, validation.manifest, Path(args.encoder_checkpoint).resolve(),
                 Path(args.head_checkpoint or args.initial_native_checkpoint).resolve(),
                 *(Path(r['audio_path']) for r in training.rows + validation.rows),
                 *(Path(__file__).with_name(name).resolve() for name in PROJECT_SOURCES)}
    if auxiliary_data is not None:
        protected.update([auxiliary_data.manifest, *(Path(p) for p in auxiliary_data.heldout_manifest_sha256),
                          *(Path(row['audio_path']) for row in auxiliary_data.rows)])
    require(not any(path == output or output in path.parents for path in protected),
            'Training output must not contain any protected input')
    model.set_device(choose_device(args.encoder_device), choose_device(args.head_device))
    seed_training(args.training_rng_seed)
    augmentation = None
    if frequency_warp_enabled(args):
        augmentation = FrequencyWarpStream(args.frequency_warp_probability, args.frequency_warp_min_scale,
                                           args.frequency_warp_max_scale, args.frequency_warp_seed)
    model.frequency_augmentation = augmentation
    optimizer = make_optimizer(model, args)
    generator = torch.Generator().manual_seed(args.shuffle_seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_audio_intervals, generator=generator)
    from .optimization import (continuation_config, check_legacy_training, check_populated_state,
                               check_optimizer_options, regroup_depth, ContinuationSchedule,
                               rates_for, schedule_phase)
    continuation, optimization = continuation_config(args, model, len(loader))
    schedule = None
    if auxiliary_data is not None:
        auxiliary = AuxiliaryStream(auxiliary_data, args.aux_batch_size or args.batch_size,
                                    args.aux_shuffle_seed, args.aux_training_rng_seed, rng_device(model))
    positive_weights, class_audit = training.positive_weights(args.pos_weight_cap)
    positive_weights = positive_weights.to(model.head_device)
    metadata = {'arguments': vars(args), 'manifest_sha256': manifests, 'initialization': initial,
                'native_provenance': model.provenance, 'initial_feature_cache_id': model.initial_feature_cache_id,
                'source_sha256': source_hashes(), 'audio_inventory_sha256': inventory_digest(audio),
                'policy': training_policy(args), 'train_clips': len(training),
                'train_windows': training.num_windows, 'validation_clips': len(validation),
                'validation_windows': validation.num_windows, 'steps_per_epoch': len(loader),
                'class_weight_audit': class_audit, 'parameters': {
                    'head': sum(p.numel() for p in model.head.parameters()),
                    'encoder': sum(p.numel() for p in model.encoder.parameters()),
                    'trainable_encoder': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)},
                'effective_learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
                'torch_version': str(torch.__version__),
                'frame_grid': FRAME_GRID, 'feature_mode': 'Live FP32 embeddings; no adapted-feature cache is reused',
                'rng_policy': 'Independent post-initialization Torch CPU/MPS seed and separate saved shuffle generator; validation uses a private generator',
                'selection_scope': 'INNER only, including native epoch zero; no automatic OUTER evaluation'}
    if continuation is not None:
        metadata['continuation_contract'] = continuation
        metadata['policy']['optimization'] = optimization
        metadata['effective_learning_rates'] = rates_for(optimization, optimization['origin_global_updates'])
    if auxiliary_metadata is not None:
        metadata['auxiliary'] = auxiliary_metadata
    warp_epoch_exposure = None
    if augmentation is not None:
        warp_epoch_exposure = {'batches': len(loader), 'clips': len(training), 'windows': training.num_windows,
                               'feature_frames': sum(row['num_frames'] for row in training.rows),
                               'mel_time_frames': training.num_windows * 1001}

    def check_augmentation_progress(epoch, stream=augmentation):
        if stream is not None:
            require(all(stream.counts[key] == epoch * count for key, count in warp_epoch_exposure.items()),
                    'Frequency-warp exposure differs from completed Gold epochs')

    def check_augmentation_checkpoint(checkpoint, stream=augmentation):
        if stream is not None:
            check_augmentation_progress(checkpoint['epoch'], stream)
            if checkpoint['epoch'] == 0:
                entry = checkpoint['baseline']
            elif checkpoint['validation_complete']:
                entry = checkpoint['history'][-1]
            else:
                entry = checkpoint['pending_validation']['details']
            require(entry.get('frequency_warp', {}).get('cumulative') == stream.summary(),
                    'Frequency-warp checkpoint and epoch audit differ')

    history, baseline, selected, first_epoch, pending = [], None, None, 0, None
    global_updates = 0
    state = {'status': 'initializing', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'requested_epochs': args.epochs, 'last_completed_epoch': 0, 'last_calibrated_epoch': None,
             'outer_evaluations_started': 0}
    epoch_dir = output / 'epoch-checkpoints'
    epoch_dir.mkdir(exist_ok=True)
    if args.resume:
        resumed = load_checkpoint(args.resume)
        resumed_sha256 = file_hash(args.resume)
        legacy_resume = check_resume_policy(resumed, metadata, args, checkpoint_sha256=resumed_sha256)
        for name in ('last.pt', 'best.pt', 'selected.pt'):
            if (output / name).exists():
                current = load_checkpoint(output / name)
                require(current['epoch'] <= resumed['epoch']
                        and current['training']['initialization'] == initial,
                        'Resume would overwrite a newer or unrelated run')
                if current['training']['policy'] != metadata['policy']:
                    require(continuation is not None, 'Resume would overwrite an unrelated policy')
                    if current['checkpoint_kind'] == 'resume':
                        check_resume_policy(current, metadata, args,
                                            checkpoint_sha256=file_hash(output / name), historical=True)
                    else:
                        require(continuation['inference_checkpoints'].get(str(current['epoch'])) == file_hash(output / name),
                                'Resume would overwrite an unbound historical inference model')
                        check_legacy_training(current['training'], metadata, continuation)
                else:
                    require(current['training']['source_sha256'] == metadata['source_sha256'],
                            'Resume would overwrite a run with different sources')
                del current
        model.load_state_dict(resumed['model_state'], strict=True)
        if optimization and not legacy_resume and optimization['mode'] == 'depth_cosine':
            regroup_depth(optimizer, model, continuation['parameter_name_groups'])
        optimizer.load_state_dict(resumed['optimizer_state'])
        history, baseline, selected, first_epoch = resumed['history'], resumed['baseline'], resumed['selected'], resumed['epoch']
        global_updates = resumed.get('global_updates', first_epoch * len(loader))
        require(type(global_updates) is int and global_updates == first_epoch * len(loader),
                'Completed Gold update count differs from the fixed batch/epoch policy')
        if optimization:
            check_populated_state(optimizer, model, global_updates)
            check_optimizer_options(optimizer, continuation)
            if legacy_resume and optimization['mode'] == 'depth_cosine':
                regroup_depth(optimizer, model, continuation['parameter_name_groups'])
            schedule = ContinuationSchedule(optimizer, model, optimization, global_updates)
            if not legacy_resume:
                schedule.restore(resumed['optimization_state'], global_updates)
            check_optimizer_options(optimizer, continuation)
            metadata['optimizer_parameter_names'] = schedule.layout
            metadata['continuation_origin'] = continuation['origin']
        if auxiliary is not None:
            auxiliary.load_state_dict(resumed['auxiliary_state'])
            require(auxiliary.batches_consumed == global_updates // args.aux_every,
                    'Auxiliary exposure count differs from the global update cadence')
        if augmentation is not None:
            augmentation.load_state_dict(resumed['frequency_warp_state'])
            check_augmentation_checkpoint(resumed)
        require([r['epoch'] for r in history] == list(range(1, first_epoch + int(resumed['validation_complete']))),
                'Resume history is not contiguous through its calibrated epoch')
        previous_output = Path(resumed['training']['arguments']['output']).resolve()
        # Preserve every completed inference checkpoint when moving a run.
        for entry in [baseline, *history]:
            source = Path(entry['checkpoint_path'])
            destination = epoch_dir / f"epoch-{entry['epoch']:03d}.pt"
            copy_checkpoint(source, destination, entry['checkpoint_sha256'])
            entry['checkpoint_path'] = str(destination)
        selected['checkpoint_path'] = str(epoch_dir / f"epoch-{selected['epoch']:03d}.pt")
        require(file_hash(selected['checkpoint_path']) == selected['checkpoint_sha256'],
                'Resumed selected checkpoint identity differs')
        # last.pt is published before the best.pt alias. If interrupted between
        # those atomic writes, the complete selected epoch in last.pt is itself
        # the exact full best state; no model/optimizer reconstruction is needed.
        best_source = Path(args.resume_best or (args.resume if resumed['validation_complete']
                           and selected['epoch'] == first_epoch else previous_output / 'best.pt'))
        require(best_source.is_file(), 'Resume requires the bound full best.pt; use --resume-best if relocated')
        best = load_checkpoint(best_source)
        check_resume_policy(best, metadata, args, checkpoint_sha256=file_hash(best_source), historical=True)
        if augmentation is not None:
            best_warp = FrequencyWarpStream(args.frequency_warp_probability, args.frequency_warp_min_scale,
                                            args.frequency_warp_max_scale, args.frequency_warp_seed)
            best_warp.load_state_dict(best['frequency_warp_state'])
            check_augmentation_checkpoint(best, best_warp)
            del best_warp
        require(best['validation_complete'] and best['epoch'] == selected['epoch']
                and best['training']['initialization'] == initial
                and state_digest(best['model_state']) == state_digest(load_checkpoint(selected['checkpoint_path'])['model_state']),
                'Full best checkpoint differs from the selected model')
        copy_checkpoint(best_source, output / 'best.pt', file_hash(best_source), replace=True)
        del best
        pending = resumed.get('pending_validation') if not resumed['validation_complete'] else None
        require(resumed['validation_complete'] or pending and pending['epoch'] == first_epoch,
                'Completed optimizer epoch lacks pending-validation metadata')
        state.update(last_completed_epoch=first_epoch, last_calibrated_epoch=history[-1]['epoch'] if history else 0)
        restore_rng(resumed['rng_state'], generator, rng_device(model))
        resumed.update(training=metadata, history=history, baseline=baseline, selected=selected)
        if schedule is not None:
            resumed.update(optimizer_state=optimizer.state_dict(), optimization_state=schedule.state_dict())
        require(file_hash(args.resume) == resumed_sha256, 'Resume input changed while loading')
        save_checkpoint(output / 'last.pt', resumed)
        del resumed
    write_json(output / 'audio-inputs.json', audio)
    write_json(output / 'training-audit.json', metadata)
    write_json(output / 'run-state.json', state)
    if baseline is not None:
        write_json(output / 'history.json', {'training': metadata, 'baseline': baseline, 'epochs': history, 'selected': selected})
    print(json.dumps({'setup': metadata}), flush=True)

    def full_state(completed, validation_complete, pending_validation=None):
        value = {**completed, 'checkpoint_kind': 'resume', 'optimizer_state': optimizer.state_dict(),
                 'rng_state': rng_state(generator, rng_device(model)), 'history': history,
                 'baseline': baseline, 'selected': selected, 'validation_complete': validation_complete,
                 'global_updates': global_updates, 'pending_validation': pending_validation}
        if auxiliary is not None:
            value['auxiliary_state'] = auxiliary.state_dict()
        if augmentation is not None:
            check_augmentation_progress(completed['epoch'])
            value['frequency_warp_state'] = augmentation.state_dict()
        if schedule is not None:
            require(schedule.completed == global_updates, 'Checkpoint and schedule completed counts differ')
            schedule.check_rates()
            value['optimization_state'] = schedule.state_dict()
        return value

    def finish_epoch(epoch, details, started):
        nonlocal baseline, selected
        state.update(status='calibrating_inner', current_epoch=epoch)
        write_json(output / 'run-state.json', state)
        records = infer_candidates(model, validation, args.inference_batch_size)
        if schedule is not None:
            require(schedule.completed == global_updates, 'INNER advanced the optimization schedule')
            schedule.check_rates()
        choice, trials = select_decoder(records, **GRID)
        entry = {'epoch': epoch, **details, 'validation': choice['metrics'], 'decoder': choice['decoder'],
                 'calibration': trials, 'seconds': time.monotonic() - started}
        if epoch == 0:
            if augmentation is not None:
                entry['frequency_warp'] = {**dict.fromkeys(WARP_COUNTS, 0),
                                           'choice_sha256': EMPTY_CHOICE_SHA256,
                                           'policy': augmentation.policy, 'cumulative': augmentation.summary()}
            original_decoder = model.head_checkpoint['decoder']
            entry['saved_decoder'] = original_decoder
            entry['saved_decoder_live_metrics'] = score_records(validation.rows, decode_records(records, original_decoder))
            if initial.get('kind') == 'native_weights_new_stage':
                entry['parent_native_validation'] = model.head_checkpoint['validation']
            else:
                entry['original_head_validation'] = model.head_checkpoint['validation']
        completed = snapshot(model, epoch, metadata, choice['decoder'], choice['metrics'])
        retained = epoch_dir / f'epoch-{epoch:03d}.pt'
        if retained.exists():
            # Recovery after the inference checkpoint was published but before
            # last.pt/history publication. Never silently replace its model.
            old = load_checkpoint(retained)
            require(old['epoch'] == epoch and old['decoder'] == completed['decoder']
                    and old['validation'] == completed['validation']
                    and old['training']['policy'] == metadata['policy']
                    and state_digest(old['model_state']) == state_digest(completed['model_state']),
                    'Retained completed-epoch checkpoint conflicts with resumed validation')
        else:
            save_checkpoint(retained, completed)
        entry.update(checkpoint_path=str(retained), checkpoint_sha256=file_hash(retained))
        if epoch == 0:
            baseline = entry
        else:
            require(not any(r['epoch'] == epoch for r in history), 'Epoch was already calibrated')
            history.append(entry)
        improved = selected is None or choice['metrics']['combined'] > selected['metrics']['combined']
        if improved:
            selected = {'kind': 'initial_native' if epoch == 0 else 'trained_epoch', 'epoch': epoch,
                        'decoder': choice['decoder'], 'metrics': choice['metrics'],
                        'checkpoint_path': str(retained), 'checkpoint_sha256': entry['checkpoint_sha256']}
        final = full_state(completed, True)
        save_checkpoint(output / 'last.pt', final)
        if improved:
            copy_checkpoint(output / 'last.pt', output / 'best.pt', file_hash(output / 'last.pt'), replace=True)
        write_json(output / 'history.json', {'training': metadata, 'baseline': baseline, 'epochs': history, 'selected': selected})
        state.update(status='epoch_complete', last_completed_epoch=epoch, last_calibrated_epoch=epoch, selected=selected)
        write_json(output / 'run-state.json', state)
        print(json.dumps(entry), flush=True)

    try:
        if baseline is None:
            finish_epoch(0, {'train_loss': None, 'loss_components': None, 'training_seconds': 0., 'updates': 0}, time.monotonic())
        if pending:
            finish_epoch(first_epoch, pending['details'], time.monotonic())
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        for epoch in range(first_epoch + 1, args.epochs + 1):
            started = time.monotonic()
            state.update(status='training', current_epoch=epoch)
            write_json(output / 'run-state.json', state)
            model.train()
            loss_sum = 0.
            component_sum = dict(presence=0., boundary=0., iou=0., collar=0.)
            auxiliary_totals = {key: 0 for key in ('batches', 'clips', 'supervised_clips', 'skipped_clips',
                                                   'positive_categories', 'supervised_category_frames', 'real_frames')}
            auxiliary_totals.update(raw_loss=0., weighted_loss=0.)
            auxiliary_ids, auxiliary_order = set(), hashlib.sha256()
            if augmentation is not None:
                warp_totals, warp_choices = dict.fromkeys(WARP_COUNTS, 0), hashlib.sha256()
            epoch_start_updates = global_updates
            if schedule is not None:
                raw_norms, update_samples = [], []
            for step, batch in enumerate(loader, 1):
                if schedule is not None:
                    schedule.before_update(global_updates + 1)
                result = train_step(model, batch, positive_weights, optimizer, parameters, args,
                                    auxiliary=auxiliary, global_update=global_updates + 1,
                                    observe_update=schedule is not None and step in (1, len(loader)))
                global_updates += 1
                if schedule is not None:
                    schedule.after_update()
                norm, components = result['gradient_norm'], result['gold_components']
                if schedule is not None:
                    raw_norms.append(float(norm.detach().cpu()))
                    if 'update_diagnostics' in result:
                        update_samples.append(dict(global_update=global_updates,
                                                   new_update=global_updates-optimization['origin_global_updates'],
                                                   phase=schedule_phase(optimization, global_updates-1),
                                                   applied_rates=rates_for(optimization, global_updates-1),
                                                   groups=result['update_diagnostics']))
                loss_sum += result['gold_loss']
                for name in component_sum:
                    component_sum[name] += components[name]
                if augmentation is not None:
                    audit = result['augmentation']
                    warp_totals['batches'] += 1
                    for key in WARP_COUNTS[1:]:
                        warp_totals[key] += audit[key]
                    for decision in audit['decisions']:
                        warp_choices.update(warp_decision_bytes(decision))
                if result['auxiliary'] is not None:
                    values = result['auxiliary']
                    auxiliary_totals['batches'] += 1
                    for name in auxiliary_totals:
                        if name != 'batches':
                            auxiliary_totals[name] += values[name]
                    auxiliary_ids.update(values['clip_ids'])
                    auxiliary_order.update(json.dumps(values['clip_ids'], separators=(',', ':')).encode() + b'\n')
                if args.log_every and step % args.log_every == 0:
                    print(json.dumps({'progress': {'epoch': epoch, 'step': step, 'steps': len(loader),
                          'train_loss': loss_sum / step, 'seconds': time.monotonic() - started,
                          'gradient_norm': float(norm.detach().cpu()), 'global_updates': global_updates,
                          'auxiliary_batches': auxiliary.batches_consumed if auxiliary else 0,
                          **({'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
                              'schedule_phase': schedule_phase(optimization, global_updates-1)} if schedule else {})}}), flush=True)
            details = {'train_loss': loss_sum / len(loader),
                       'loss_components': {k: v / len(loader) for k, v in component_sum.items()},
                       'training_seconds': time.monotonic() - started, 'updates': len(loader),
                       'global_updates_start': epoch_start_updates, 'global_updates_end': global_updates}
            if schedule is not None:
                details['optimization'] = dict(state=schedule.state_dict(),
                    first_applied_rates=rates_for(optimization, epoch_start_updates),
                    last_applied_rates=rates_for(optimization, global_updates-1),
                    raw_global_gradient_norm=dict(count=len(raw_norms), minimum=min(raw_norms), maximum=max(raw_norms),
                         mean=math.fsum(raw_norms)/len(raw_norms), above_clip=sum(n > args.gradient_clip for n in raw_norms)),
                    sampled_updates=update_samples)
            if augmentation is not None:
                require(all(warp_totals[key] == count for key, count in warp_epoch_exposure.items()),
                        'Frequency-warp epoch exposure differs from the Gold dataset')
                details['frequency_warp'] = {**warp_totals, 'choice_sha256': warp_choices.hexdigest(),
                                             'policy': augmentation.policy, 'cumulative': augmentation.summary()}
            if auxiliary is not None:
                count = auxiliary_totals['batches']
                details['auxiliary'] = {**{k: v for k, v in auxiliary_totals.items() if k not in ('raw_loss', 'weighted_loss')},
                                        'raw_loss_mean_per_aux_batch': auxiliary_totals['raw_loss'] / count if count else 0.,
                                        'weighted_loss_mean_per_aux_batch': auxiliary_totals['weighted_loss'] / count if count else 0.,
                                        'weighted_loss_per_gold_update': auxiliary_totals['weighted_loss'] / len(loader),
                                        'combined_objective_mean_per_gold_update': (loss_sum + auxiliary_totals['weighted_loss']) / len(loader),
                                        'unique_clips': len(auxiliary_ids), 'clip_order_sha256': auxiliary_order.hexdigest(),
                                        'cumulative_batches': auxiliary.batches_consumed,
                                        'cumulative_examples': auxiliary.examples_consumed,
                                        'pool_cycles': auxiliary.cycles, 'pool_cursor': auxiliary.cursor}
            completed = snapshot(model, epoch, metadata, None, None)
            save_checkpoint(output / 'last.pt', full_state(completed, False, {'epoch': epoch, 'details': details}))
            state['last_completed_epoch'] = epoch
            del completed
            finish_epoch(epoch, details, started)
        require(len(history) == args.epochs and [r['epoch'] for r in history] == list(range(1, args.epochs + 1)),
                'Final horizon is incomplete')
        require(source_hashes() == metadata['source_sha256'], 'Implementation changed while training')
        require({'train': file_hash(training.manifest), 'validation': file_hash(validation.manifest)} == manifests,
                'A training or INNER manifest changed during the run')
        final_audio = {'train': audio_inventory(training), 'validation': audio_inventory(validation)}
        if auxiliary_data is not None:
            final_audio['auxiliary'] = audio_inventory(auxiliary_data)
            require(file_hash(auxiliary_data.manifest) == auxiliary_metadata['manifest_sha256']
                    and all(file_hash(path) == digest for path, digest in auxiliary_metadata['heldout_manifest_sha256'].items()),
                    'Auxiliary or exclusion manifest changed during the run')
            require(auxiliary.batches_consumed == global_updates // args.aux_every,
                    'Final auxiliary exposure count differs from the global update cadence')
        require(inventory_digest(final_audio) == metadata['audio_inventory_sha256'], 'Raw audio changed during the run')
        copy_checkpoint(selected['checkpoint_path'], output / 'selected.pt', selected['checkpoint_sha256'], replace=True)
        report = {'status': 'complete', 'completed_epochs': args.epochs, 'requested_epochs': args.epochs,
                  'selected': selected, 'selected_epoch': selected['epoch'],
                  'selected_checkpoint': str(output / 'selected.pt'),
                  'selected_checkpoint_sha256': selected['checkpoint_sha256'],
                  'baseline': baseline, 'actual_inner_grid_trials': 24 * (args.epochs + 1),
                  'outer_evaluations_started': 0, 'outer_evaluations_completed': 0,
                  'selection_scope': metadata['selection_scope'], 'finished_utc': datetime.now(timezone.utc).isoformat()}
        report['global_updates'] = global_updates
        if schedule is not None:
            check_optimizer_options(optimizer, continuation)
            report.update(inherited_epochs=optimization['origin_epoch'],
                          new_epochs=args.epochs-optimization['origin_epoch'],
                          new_global_updates=global_updates-optimization['origin_global_updates'],
                          inherited_inner_grid_trials=24*(optimization['origin_epoch']+1),
                          new_inner_grid_trials=24*(args.epochs-optimization['origin_epoch']),
                          optimization_state=schedule.state_dict(), continuation_origin=continuation['origin'])
        if augmentation is not None:
            check_augmentation_progress(args.epochs)
            report['frequency_warp'] = {'policy': augmentation.policy, **augmentation.summary()}
        if auxiliary is not None:
            report['auxiliary'] = {'batches': auxiliary.batches_consumed, 'examples': auxiliary.examples_consumed,
                                   'pool_cycles': auxiliary.cycles, 'pool_cursor': auxiliary.cursor,
                                   'cadence': metadata['policy']['auxiliary']['cadence']}
        write_json(output / 'report.json', report)
        state.update(status='complete', report=report)
        write_json(output / 'run-state.json', state)
        print(json.dumps(report), flush=True)
        return report
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', error=str(exc))
        write_json(output / 'run-state.json', state)
        raise
    finally:
        lock.close()


def predict(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    checkpoint_path = Path(args.checkpoint).resolve()
    payload = checkpoint_path.read_bytes()
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    require(args.checkpoint_sha256 is None or args.checkpoint_sha256 == checkpoint_sha,
            'Requested native RED checkpoint SHA-256 differs')
    checkpoint = load_checkpoint(io.BytesIO(payload))
    del payload
    require(checkpoint['checkpoint_kind'] == 'inference', 'Prediction requires a retained/selected inference checkpoint')
    initial = checkpoint['training']['initialization']
    source_root = Path(args.source_root or initial['source_root']).resolve()
    encoder_checkpoint = Path(args.encoder_checkpoint or initial['encoder_checkpoint']).resolve()
    require(file_hash(encoder_checkpoint) == initial['encoder_sha256'], 'Initial released encoder bytes differ')
    require({name: file_hash(source_root / name) for name in SOURCE_FILES}
            == checkpoint['training']['native_provenance']['source_files_sha256'], 'Native source files differ')
    from .optimization import prediction_source_binding
    current_sources = source_hashes()
    inherited_binding = prediction_source_binding(checkpoint, checkpoint_sha, current_sources, args)
    manifest_sha = file_hash(args.manifest)
    dataset = PredictionClips(args.manifest)
    require(file_hash(dataset.manifest) == manifest_sha, 'Prediction manifest changed while loading')
    output = Path(args.output).resolve()
    sidecar = output.with_suffix(output.suffix + '.provenance.json')
    audio_path = output.with_suffix(output.suffix + '.audio-inputs.json')
    archive = Path(args.frame_outputs).resolve() if args.frame_outputs else None
    destinations = [output, sidecar, audio_path, *([archive] if archive else [])]
    require(len(set(destinations)) == len(destinations), 'Prediction destinations must differ')
    protected = {checkpoint_path, dataset.manifest, encoder_checkpoint,
                 *(source_root / name for name in SOURCE_FILES), *(Path(row['audio_path']) for row in dataset.rows),
                 *(Path(__file__).with_name(name).resolve() for name in PROJECT_SOURCES)}
    require(not set(destinations) & protected
            and not {p.with_suffix(p.suffix + '.tmp') for p in destinations} & protected,
            'Prediction output would overwrite input data')
    model = NativeIntervalATST.from_checkpoint(source_root, encoder_checkpoint, checkpoint)
    model.requires_grad_(False).eval().set_device(choose_device(args.encoder_device), choose_device(args.head_device))
    audio = audio_inventory(dataset)
    predictions, arrays, clip_offsets = [], [], [0]
    for row, values in infer_outputs(model, dataset, args.batch_size):
        candidates = candidate_intervals(row, values['presence'], values['distances'])
        predictions.append(decode_candidates(row, candidates, checkpoint['decoder']))
        if archive:
            arrays.append(values)
            clip_offsets.append(clip_offsets[-1] + row['num_frames'])
    coverage = validate_predictions(predictions, dataset.rows)
    require(file_hash(dataset.manifest) == manifest_sha and audio_inventory(dataset) == audio,
            'Prediction manifest or raw audio changed during inference')
    require(source_hashes() == current_sources
            and {name: file_hash(source_root / name) for name in SOURCE_FILES}
            == checkpoint['training']['native_provenance']['source_files_sha256'],
            'Prediction source changed during inference')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, predictions)
    write_json(audio_path, audio)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **{k: np.concatenate([v[k] for v in arrays]) for k in arrays[0]},
                                clip_offsets=np.asarray(clip_offsets, dtype=np.int64),
                                clip_ids=np.asarray([r['clip_id'] for r in dataset.rows]),
                                metadata=np.asarray(json.dumps({'model_type': MODEL_TYPE, 'checkpoint_sha256': checkpoint_sha,
                                                               'frame_grid': FRAME_GRID, 'decoder': checkpoint['decoder']})))
        temporary.replace(archive)
    require(file_hash(checkpoint_path) == checkpoint_sha and file_hash(encoder_checkpoint) == initial['encoder_sha256'],
            'A model input changed during prediction')
    report = {'scope': 'Raw-audio inference only; no original training manifests, initial RED head, feature cache, labels or scoring',
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_sha, 'selected_epoch': checkpoint['epoch'],
              'manifest': str(dataset.manifest), 'manifest_sha256': manifest_sha,
              'audio_inventory': str(audio_path), 'audio_inventory_sha256': inventory_digest(audio),
              'prediction_sha256': file_hash(output), 'coverage': coverage,
              'frame_outputs': str(archive) if archive else None,
              'frame_outputs_sha256': file_hash(archive) if archive else None,
              'model_config': checkpoint['model_config'], 'categories': list(CATEGORIES), 'frame_grid': FRAME_GRID,
              'decoder': checkpoint['decoder'], 'native_provenance': model.provenance,
              'source_sha256': source_hashes(), 'encoder_device': str(model.device), 'head_device': str(model.head_device),
              'batch_size': args.batch_size, 'torch_version': str(torch.__version__), 'seconds': time.monotonic() - started}
    if inherited_binding is not None:
        report['inherited_source_binding'] = inherited_binding
    write_json(sidecar, report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    for name in ('source-root', 'encoder-checkpoint', 'train', 'val', 'output'):
        training.add_argument('--' + name, required=True)
    initialization = training.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--head-checkpoint', help='Original trained frozen-feature RED head with the released encoder')
    initialization.add_argument('--initial-native-checkpoint', help='Calibrated complete native model; initialize weights only in a fresh training stage')
    training.add_argument('--initial-native-sha256', help='Required exact SHA-256 for --initial-native-checkpoint')
    training.add_argument('--train-blocks', type=int, choices=(0, 12), default=12)
    training.add_argument('--epochs', type=int, default=8, help='Total completed epochs, including resumed epochs')
    training.add_argument('--encoder-lr', type=float, default=1e-6)
    training.add_argument('--head-lr', type=float, default=1e-4)
    training.add_argument('--weight-decay', type=float, default=1e-4)
    training.add_argument('--pos-weight-cap', type=float, default=5.)
    training.add_argument('--boundary-weight', type=float, default=100.)
    training.add_argument('--iou-weight', type=float, default=1.)
    training.add_argument('--gradient-clip', type=float, default=5.)
    training.add_argument('--seed', type=int, default=45)
    training.add_argument('--training-rng-seed', type=int, default=45)
    training.add_argument('--shuffle-seed', type=int, default=45)
    training.add_argument('--inference-batch-size', type=int, default=8)
    training.add_argument('--log-every', type=int, default=100)
    training.add_argument('--resume', help='Full completed optimizer checkpoint, including pending INNER validation')
    training.add_argument('--resume-best', help='Relocated full best.pt for an exact resume')
    training.add_argument('--optimization-policy', choices=('constant', 'depth_cosine'),
                          help='Explicit true-continuation policy; requires the exact inherited-state contract')
    training.add_argument('--test-only-clips', type=int, help='Bounded QA subset of each original manifest; marked in all artifacts')
    training.add_argument('--aux-manifest', help='Optional raw-audio pool with declared clip-positive categories')
    training.add_argument('--aux-heldout-manifests', nargs='+', default=[], help='Additional exclusion manifests, including OUTER; INNER is always added')
    training.add_argument('--aux-weight', type=float, default=0., help='Clip-positive loss weight; zero leaves the Gold trajectory unchanged')
    training.add_argument('--aux-every', type=int, default=4, help='Auxiliary backward every N global Gold optimizer updates')
    training.add_argument('--aux-batch-size', type=int, help='Full auxiliary clip batch size; defaults to the Gold batch size')
    training.add_argument('--aux-shuffle-seed', type=int, default=46)
    training.add_argument('--aux-training-rng-seed', type=int, default=47, help='Independent CPU/MPS dropout stream for auxiliary audio')
    training.add_argument('--frequency-warp-probability', type=float, default=0.,
                          help='Gold clip probability for time-preserving frequency crop/resize; zero is the original training path')
    training.add_argument('--frequency-warp-min-scale', type=float, default=.6)
    training.add_argument('--frequency-warp-max-scale', type=float, default=1.)
    training.add_argument('--frequency-warp-seed', type=int, default=48,
                          help='Independent saved CPU Torch generator for per-clip crop choices')
    prediction = sub.add_parser('predict')
    for name in ('checkpoint', 'manifest', 'output'):
        prediction.add_argument('--' + name, required=True)
    prediction.add_argument('--checkpoint-sha256')
    prediction.add_argument('--source-root')
    prediction.add_argument('--encoder-checkpoint')
    prediction.add_argument('--frame-outputs', help='Optional aligned FP32 RED outputs; no calibration is performed')
    for command in (training, prediction):
        command.add_argument('--continuation-contract', help='Bound inherited-source/optimizer contract for this continuation')
        command.add_argument('--continuation-contract-sha256')
        command.add_argument('--batch-size', type=int, default=4 if command is training else 8)
        command.add_argument('--threads', type=int, default=2)
        command.add_argument('--encoder-device', choices=('cpu', 'mps'), default='mps')
        command.add_argument('--head-device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) <= 0:
        parser.error('Batch size and threads must be positive')
    if args.command == 'train':
        try:
            validate_frequency_warp_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.initial_native_checkpoint:
            if (not args.initial_native_sha256 or len(args.initial_native_sha256) != 64
                    or any(c not in '0123456789abcdef' for c in args.initial_native_sha256)):
                parser.error('--initial-native-checkpoint requires its exact lowercase SHA-256')
        elif args.initial_native_sha256:
            parser.error('--initial-native-sha256 requires --initial-native-checkpoint')
        if min(args.epochs, args.inference_batch_size) <= 0 or args.log_every < 0:
            parser.error('Epochs and inference batch size must be positive; log frequency nonnegative')
        if any(not math.isfinite(x) or x <= 0 for x in (args.encoder_lr, args.head_lr, args.gradient_clip)):
            parser.error('Learning rates and gradient clip must be positive and finite')
        if any(not math.isfinite(x) or x < 0 for x in (args.weight_decay, args.boundary_weight, args.iou_weight)):
            parser.error('Loss weights and weight decay must be finite and nonnegative')
        if not math.isfinite(args.pos_weight_cap) or args.pos_weight_cap < 1:
            parser.error('Positive weight cap must be finite and at least one')
        if args.test_only_clips is not None and args.test_only_clips <= 0:
            parser.error('test-only-clips must be positive')
        if not math.isfinite(args.aux_weight) or args.aux_weight < 0:
            parser.error('Auxiliary loss weight must be finite and nonnegative')
        if args.aux_every <= 0 or args.aux_batch_size is not None and args.aux_batch_size <= 0:
            parser.error('Auxiliary frequency and batch size must be positive')
        if min(args.aux_shuffle_seed, args.aux_training_rng_seed) < 0:
            parser.error('Auxiliary seeds must be nonnegative')
        if args.aux_weight > 0 and not args.aux_manifest:
            parser.error('Positive auxiliary weight requires --aux-manifest')
        if auxiliary_enabled(args) and not args.aux_heldout_manifests:
            parser.error('Auxiliary training requires OUTER in --aux-heldout-manifests; INNER is added automatically')
    return (fit if args.command == 'train' else predict)(args)


if __name__ == '__main__':
    main()
