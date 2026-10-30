"""Native WavLM Large representations with the existing masked RED interval head.

Preparation only: this module has no training entry point. It never resizes the
native sequence or changes original reference events. Public weights are loaded
locally and strictly, from a folder that holds the pinned microsoft/wavlm-large files (see ASSET_HASHES).
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel
from transformers.models.wavlm import modeling_wavlm

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / 'atst_red'          # the indoml package of local runs (the Kaggle job imports it from bundle/source)
sys.path.insert(0, str(FROZEN))
from indoml import intervals
from indoml.transfer import CATEGORIES, category_targets

SAMPLE_RATE = 16000
KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)
HOP_SAMPLES = 320
RECEPTIVE_SAMPLES = 400
FRAME_GRID = {'frame_hop': .02, 'frame_offset': .0125}
REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
ASSET_HASHES = {
    'config.json': 'a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d',
    'preprocessor_config.json': '60ca5a31e13f69ee2fbf147504c8676db5f6398fd7a6b12294341dff838edfcf',
    'pytorch_model.bin': 'fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d',
}
RUNTIME_SOURCE_SHA = '2d876ea90e1df2e049149a248b47c9f8ab496dbf5a95ee5da9222ed439b62cdf'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes():
    paths = [Path(__file__), Path(modeling_wavlm.__file__)]
    paths += sorted(path for path in (FROZEN / 'indoml').glob('*.py'))
    return {str(path): sha(path) for path in paths}


def native_lengths(sample_lengths):
    """Seven valid convolutions, including inputs with zero native outputs."""
    lengths = torch.as_tensor(sample_lengths, dtype=torch.long)
    require(bool((lengths >= 0).all()), 'Sample lengths must be nonnegative')
    for kernel, stride in zip(KERNELS, STRIDES):
        lengths = ((lengths - kernel) // stride + 1).clamp_min(0)
    return lengths


def frame_centers(frames, *, device='cpu'):
    """Center of sample intervals [i/sr, (i+1)/sr), in float64 seconds.

    A receptive field occupies 400 sample intervals, so the first center is
    200/16000 seconds. Sample-point centers would be earlier by 1/(2*16000).
    """
    require(frames >= 0, 'Negative frame count')
    # Use the same floating-point expression as the frozen target builder so
    # references exactly on a frame center receive the same half-open decision.
    return (torch.arange(frames, dtype=torch.float64, device=device) * FRAME_GRID['frame_hop']
            + FRAME_GRID['frame_offset'])


def normalize_real_audio(waveforms, sample_lengths):
    """Ignore supplied padding, normalize each real waveform, then pad zeros."""
    require(waveforms.ndim == 2 and waveforms.is_floating_point(), 'Expected floating mono [batch,samples] audio')
    require(sample_lengths.ndim == 1 and sample_lengths.dtype == torch.long
            and len(sample_lengths) == len(waveforms), 'Expected int64 sample length per waveform')
    lengths = sample_lengths.detach().cpu()
    require(bool((lengths >= RECEPTIVE_SAMPLES).all()) and bool((lengths <= waveforms.shape[1]).all()),
            'Each real waveform must contain at least 400 samples and fit the input')
    normalized = []
    for waveform, length in zip(waveforms, lengths.tolist()):
        real = waveform[:length].float()
        require(bool(torch.isfinite(real).all()), 'Nonfinite real audio')
        normalized.append((real - real.mean()) / torch.sqrt(real.var(unbiased=False) + 1e-7))
    values = nn.utils.rnn.pad_sequence(normalized, batch_first=True)
    mask = torch.arange(values.shape[1], device=values.device)[None] < lengths.to(values.device)[:, None]
    return values, mask


def native_row(row, samples):
    """Replace only cached feature geometry; preserve every reference instance."""
    require(samples >= RECEPTIVE_SAMPLES, 'No valid native output for this clip')
    require(abs(float(row['duration']) - samples / SAMPLE_RATE) <= 1 / SAMPLE_RATE + 1e-9,
            'Audio length and original clip duration differ')
    result = deepcopy(row)
    for key in ('feature_path', 'feature_cache_id', 'feature_dim'):
        result.pop(key, None)
    result.update(FRAME_GRID, num_frames=int(native_lengths(samples)))
    result['native_sample_count'] = int(samples)
    return result


def target_coverage(row):
    """Retain original event IDs and explain any event with zero owned cells."""
    times = frame_centers(row['num_frames']).numpy()
    owners = np.full((len(times), len(CATEGORIES)), -1, dtype=np.int64)
    spans = np.full(owners.shape, np.inf)
    covered_counts = []
    for event_id, event in enumerate(row['events']):
        category = CATEGORIES.index(event['category'])
        covered = (times >= event['onset']) & (times < event['offset'])
        duration = event['offset'] - event['onset']
        assigned = covered & (duration < spans[:, category])
        owners[assigned, category] = event_id
        spans[assigned, category] = duration
        covered_counts.append(int(covered.sum()))
    counts = np.bincount(owners[owners >= 0], minlength=len(row['events']))
    missing = []
    for event_id, owned in enumerate(counts):
        if owned:
            continue
        event = row['events'][event_id]
        covered = (times >= event['onset']) & (times < event['offset'])
        category = CATEGORIES.index(event['category'])
        missing.append({'clip_id': row['clip_id'], 'event_id': event_id, 'event': deepcopy(event),
                        'reason': 'no_native_frame_center' if not covered_counts[event_id] else 'same_category_ownership',
                        'covered_cells': covered_counts[event_id], 'owned_cells': 0,
                        'competing_event_ids': sorted(set(int(v) for v in owners[covered, category] if v >= 0))})
    return owners, counts, missing


def collate_native(rows, waveforms):
    """Whole raw clips and Gold targets; original labels stay in returned rows."""
    require(len(rows) == len(waveforms) and len(rows) > 0, 'Rows and audio must have equal nonzero batch size')
    require(all(w.ndim == 1 and w.dtype == torch.float32 for w in waveforms), 'Expected mono FP32 waveforms')
    sample_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    native_rows = [native_row(row, len(w)) for row, w in zip(rows, waveforms)]
    lengths = native_lengths(sample_lengths)
    labels = [intervals.interval_targets(row) for row in native_rows]
    targets = [category_targets(row) for row in native_rows]
    require(all(torch.equal(t.bool(), label['event_weights'] > 0) for t, label in zip(targets, labels)),
            'Native interval ownership does not preserve presence targets')
    result = {'waveforms': nn.utils.rnn.pad_sequence(waveforms, batch_first=True),
              'sample_lengths': sample_lengths, 'lengths': lengths, 'rows': native_rows,
              'mask': torch.arange(int(lengths.max()))[None] < lengths[:, None],
              'targets': nn.utils.rnn.pad_sequence(targets, batch_first=True)}
    result.update({key: nn.utils.rnn.pad_sequence([label[key] for label in labels], batch_first=True)
                   for key in ('distances', 'event_weights', 'onsets', 'offsets')})
    return result


@contextmanager
def evaluation_scope(model):
    """Preserve mode and Python/NumPy/Torch RNG, including upstream eval draws."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    devices = {p.device.type for p in model.parameters()}
    mps_state = torch.mps.get_rng_state() if 'mps' in devices else None
    cuda_state = torch.cuda.get_rng_state_all() if 'cuda' in devices else None
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for module, training in modes:
            module.training = training


class NativeWavLMIntervals(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Hidden states: post-position input, outputs of blocks 1..23, and the
        # final-normalized output of block 24, exactly as exposed by this runtime.
        self.layer_weights = nn.Parameter(torch.zeros(25))
        self.head = intervals.IntervalModel(feature_dim=1024, projection_dim=256, hidden_size=128,
                                            dropout=.2, mode='red', duration_hidden=32, epn_context='per_class')
        self.provenance = None

    @classmethod
    def from_assets(cls, asset_dir):
        directory = Path(asset_dir).resolve()
        require(sha(modeling_wavlm.__file__) == RUNTIME_SOURCE_SHA, 'Installed WavLM implementation changed')
        require(str(torch.__version__).split('+')[0] == '2.8.0', 'Use the separately pinned WavLM PyTorch runtime')
        for name, expected in ASSET_HASHES.items():
            require(sha(directory / name) == expected, f'Pinned public asset differs: {name}')
        config = WavLMConfig.from_json_file(str(directory / 'config.json'))
        require(tuple(config.conv_kernel) == KERNELS and tuple(config.conv_stride) == STRIDES
                and config.num_hidden_layers == 24 and config.hidden_size == 1024
                and config.feat_extract_norm == 'layer' and config.do_stable_layer_norm and not config.add_adapter,
                'Unexpected WavLM Large architecture')
        # Meta construction plus assign avoids a second initialized FP32 model.
        with torch.device('meta'):
            encoder = WavLMModel(config)
        weights = torch.load(directory / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
        require(len(weights) == 488 and sum(t.numel() for t in weights.values()) == 315453120,
                'Public tensor inventory differs')
        encoder.load_state_dict(weights, strict=True, assign=True)
        del weights
        require(all(p.device.type == 'cpu' for p in encoder.parameters()), 'Encoder contains unmaterialized parameters')
        # Preserve and load the original masked embedding, but it is unused by
        # this fixed localization policy. All actual feature layers can learn.
        config.apply_spec_augment = False
        config.layerdrop = 0.
        encoder.masked_spec_embed.requires_grad_(False)
        model = cls(encoder)
        model.provenance = {'repository': 'microsoft/wavlm-large', 'revision': REVISION,
                            'asset_dir': str(directory), 'asset_sha256': ASSET_HASHES,
                            'runtime_source_sha256': RUNTIME_SOURCE_SHA, 'frame_grid': FRAME_GRID,
                            'frame_center_convention': 'sample intervals, 200/16000 + i*320/16000 seconds',
                            'sequence_lengths': 'seven valid convolutions; no interpolation',
                            'normalization': 'real-sample FP32 population variance plus 1e-7 before zero padding',
                            'representation': 'learned softmax mixture of all 25 exposed hidden states',
                            'effective_overrides': {'apply_spec_augment': False, 'layerdrop': 0.},
                            'trainable_encoder': 'all seven convolutions, projection, positional embedding and 24 blocks',
                            'unused_frozen_parameter': 'masked_spec_embed'}
        return model

    def set_device(self, encoder_device, head_device='cpu'):
        # Call before creating an optimizer. Mix on the encoder device, then
        # transfer one sequence to the head rather than all 25 hidden states.
        self.to(encoder_device)
        self.head.to(head_device)
        return self

    def encode(self, waveforms, sample_lengths):
        values, sample_mask = normalize_real_audio(waveforms, sample_lengths)
        device = next(self.encoder.parameters()).device
        encoded = self.encoder(values.to(device), attention_mask=sample_mask.to(device),
                               output_hidden_states=True, return_dict=True)
        require(len(encoded.hidden_states) == 25, 'Expected all 25 exposed representations')
        lengths = native_lengths(sample_lengths.detach().cpu())
        require(encoded.last_hidden_state.shape[1] == int(lengths.max()), 'Native output length mismatch')
        head_device = next(self.head.parameters()).device
        require(self.layer_weights.device == device, 'Layer mixture and encoder devices differ')
        coefficients = self.layer_weights.softmax(dim=0)
        # Sum without stacking another B*T*25*1024 tensor.
        mixed = encoded.hidden_states[0] * coefficients[0]
        for coefficient, hidden in zip(coefficients[1:], encoded.hidden_states[1:]):
            mixed = mixed + hidden * coefficient
        mixed = mixed.to(head_device)
        mask = torch.arange(mixed.shape[1], device=head_device)[None] < lengths.to(head_device)[:, None]
        return mixed.masked_fill(~mask[:, :, None], 0.), lengths, mask

    def forward(self, waveforms, sample_lengths, *, return_features=False):
        features, lengths, mask = self.encode(waveforms, sample_lengths)
        output = self.head(features, lengths)
        for key, value in output.items():
            expanded = mask.reshape(*mask.shape, *([1] * (value.ndim - 2)))
            output[key] = value.masked_fill(~expanded, 0.)
        if return_features:
            return output, {'features': features, 'lengths': lengths, 'mask': mask}
        return output
