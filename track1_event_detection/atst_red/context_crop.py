"""Sample-exact Gold context crops and the existing native ATST/RED update.

WindowStream must receive original rows. Only this collator transforms a draw,
after packing, and rebuilds the 40 ms native grid from its in-memory waveform.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 16000
PARENT_SHA = '419c441e911419b33fa066d67ffa43bb63a8c6b3c659175ccb565f0ecfb83373'
PARENT_STATE = '3fa453a39509b804f80e628c53643c346824373b63c81b0deb5d877bc14a64e9'
PARENT_DECODER = {'threshold': .65, 'nms_iou': 0., 'min_duration': .08}
EMPTY_CHAIN = hashlib.sha256(b'').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def chain(previous, value):
    return hashlib.sha256(bytes.fromhex(previous) + json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sample_count(row):
    n = round(row['duration'] * SAMPLE_RATE)
    require(n > 0 and abs(n / SAMPLE_RATE - row['duration']) < 1e-8,
            'Source duration must describe exact 16 kHz samples')
    return n


def crop_limits(row):
    """Bounds from the approved temporal audit; no WavLM frame code is reused."""
    require(row['supervision'] == 'strong_gold' and row['quality'] == 'verified_timestamps'
            and bool(row['events']), 'Only reliable Gold may receive temporal crops')
    n = sample_count(row)
    minimum = min(n, round(1.5 * SAMPLE_RATE))
    side_cap = min(math.floor(.15 * n), (n - minimum) // 2)
    first = min(event['onset'] for event in row['events'])
    last = max(event['offset'] for event in row['events'])
    left = max(0, min(side_cap, math.floor((first - .25) * SAMPLE_RATE)))
    right = max(0, min(side_cap, math.floor((row['duration'] - last - .25) * SAMPLE_RATE)))
    return n, minimum, left, right


def translate(row, left, right):
    """Translate every endpoint; keep annotation order, multiplicity and fields."""
    from indoml.pretrained import window_starts
    n, minimum, left_max, right_max = crop_limits(row)
    require(type(left) is int and type(right) is int and 0 <= left <= left_max
            and 0 <= right <= right_max, 'Crop exceeds audited bounds')
    if left == right == 0:
        return row
    retained = n - left - right
    require(retained >= minimum, 'Crop is shorter than the minimum retained duration')
    shift = left / SAMPLE_RATE
    events = [{**event, 'onset': event['onset'] - shift,
               'offset': event['offset'] - shift} for event in row['events']]
    for original, event in zip(row['events'], events, strict=True):
        require(0 <= event['onset'] < event['offset'] <= retained / SAMPLE_RATE + 1e-12,
                'Crop removed an event boundary')
        require(abs(event['onset'] + shift - original['onset']) < 1e-12
                and abs(event['offset'] + shift - original['offset']) < 1e-12,
                'Endpoint translation changed the event')
    return {**row, 'events': events, 'duration': retained / SAMPLE_RATE,
            'num_frames': (retained + 639) // 640, 'frame_hop': .04, 'frame_offset': .02,
            'native_windows': len(window_starts(retained)),
            'context_crop': {'source_samples': n, 'left_samples': left,
                             'right_samples': right, 'retained_samples': retained}}


class CropStream:
    """Private shadow decisions are identical in control and crop trajectories."""
    def __init__(self, seed=107, probability=.5):
        require(0 <= probability <= 1, 'Invalid crop probability')
        self.seed, self.probability = seed, probability
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.counts = Counter()
        self.choice_chain_sha256 = EMPTY_CHAIN

    def draw(self, row):
        n, minimum, left_max, right_max = crop_limits(row)
        selected = float(torch.rand((), dtype=torch.float64, generator=self.generator)) < self.probability
        # Always consume all three draws, including ineligible clips and control.
        left = int(torch.randint(left_max + 1, (), generator=self.generator))
        right = int(torch.randint(right_max + 1, (), generator=self.generator))
        decision = {'clip_id': row['clip_id'], 'selected': selected,
                    'left_samples': left if selected else 0,
                    'right_samples': right if selected else 0,
                    'left_max': left_max, 'right_max': right_max, 'source_samples': n}
        self.choice_chain_sha256 = chain(self.choice_chain_sha256, decision)
        self.counts['draws'] += 1
        self.counts['selected_draws'] += int(selected)
        self.counts['eligible_draws'] += int(left_max + right_max > 0)
        self.counts['source_samples'] += n
        self.counts['event_records'] += len(row['events'])
        return decision

    def waveforms(self, rows, apply_crop):
        from indoml.data import load_audio
        transformed, waveforms, decisions = [], [], []
        for row in rows:
            audio = load_audio(row['audio_path'], SAMPLE_RATE)
            require(len(audio) == sample_count(row), 'Original audio and sample geometry differ')
            decision = self.draw(row)
            left = decision['left_samples'] if apply_crop else 0
            right = decision['right_samples'] if apply_crop else 0
            new_row = translate(row, left, right)
            value = audio[left:len(audio) - right]
            require(len(value) == sample_count(new_row), 'Crop sample count differs')
            self.counts['applied_draws'] += int(left + right > 0)
            self.counts['retained_samples'] += len(value)
            transformed.append(new_row)
            waveforms.append(value)
            decisions.append(decision)
        return transformed, waveforms, decisions

    def summary(self):
        return {**self.counts, 'choice_chain_sha256': self.choice_chain_sha256,
                'retained_audio_fraction': self.counts['retained_samples'] / self.counts['source_samples']
                if self.counts['source_samples'] else 1.}

    def state_dict(self):
        return {'seed': self.seed, 'probability': self.probability,
                'generator_state': self.generator.get_state().clone(),
                'counts': dict(self.counts), 'choice_chain_sha256': self.choice_chain_sha256}

    def load_state_dict(self, state):
        require((state['seed'], state['probability']) == (self.seed, self.probability),
                'Crop policy changed on resume')
        self.generator.set_state(state['generator_state'])
        self.counts = Counter(state['counts'])
        self.choice_chain_sha256 = state['choice_chain_sha256']


def collate_waveforms(rows, waveforms, api):
    """The native prepare_clips/collate_audio_intervals logic, with audio supplied.

    The native constants, window starts, triangular weights and target functions
    are imported from the unchanged production bundle. No temporary crop files.
    """
    from indoml.atst_finetune import TRIANGULAR
    from indoml.pretrained import FRAME_SAMPLES, WINDOW_SAMPLES, WINDOW_FRAMES, window_starts
    require((FRAME_SAMPLES, WINDOW_SAMPLES, WINDOW_FRAMES) == (640, 160000, 250),
            'Native ATST geometry changed')
    windows, descriptions = [], []
    for row, audio in zip(rows, waveforms, strict=True):
        frames = (len(audio) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        require(frames == row['num_frames'] and len(audio) == sample_count(row),
                'Transformed native geometry differs')
        starts = [value // FRAME_SAMPLES for value in window_starts(len(audio))]
        total = np.zeros(frames, dtype=np.float32)
        for first in starts:
            count = min(WINDOW_FRAMES, frames - first)
            total[first:first + count] += TRIANGULAR[:count]
        require(bool(np.all(total > 0)), 'Native stitching has uncovered frames')
        sections = []
        for first in starts:
            count = min(WINDOW_FRAMES, frames - first)
            sections.append({'window': len(windows), 'first': first, 'count': count,
                             'weight': torch.from_numpy(TRIANGULAR[:count] / total[first:first + count])})
            value = torch.from_numpy(audio[first * FRAME_SAMPLES:first * FRAME_SAMPLES + WINDOW_SAMPLES])
            windows.append(F.pad(value, (0, WINDOW_SAMPLES - len(value))))
        descriptions.append({'row': row, 'frames': frames, 'sections': sections})
    lengths = torch.tensor([item['frames'] for item in descriptions], dtype=torch.long)
    labels = [api.interval_targets(row) for row in rows]
    targets = [api.category_targets(row) for row in rows]
    require(all(torch.equal(target.bool(), label['event_weights'] > 0)
                for target, label in zip(targets, labels)), 'Native target occupancy differs')
    batch = {'windows': torch.stack(windows), 'descriptions': descriptions, 'rows': rows,
             'lengths': lengths, 'frames': int(lengths.sum()),
             'targets': nn.utils.rnn.pad_sequence(targets, batch_first=True),
             'mask': torch.arange(int(lengths.max()))[None] < lengths[:, None]}
    for key in ('distances', 'event_weights', 'onsets', 'offsets'):
        batch[key] = nn.utils.rnn.pad_sequence([label[key] for label in labels], batch_first=True)
    return batch


@contextmanager
def forward_rng(device, base_seed, update, phase):
    """Common logical draws; altered shapes cannot shift the next forward's RNG."""
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        seed = base_seed + update * 2 + phase
        # torch.manual_seed also touches devices not owned by this CPU fixture.
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        yield


def schedule_rates(optimizer, schedule, update):
    require(0 <= update < schedule['total_updates'], 'Update outside frozen schedule')
    warm = schedule['warmup_updates']
    if update < warm:
        fraction = .1 + .9 * update / max(1, warm - 1)
    else:
        progress = (update - warm) / max(1, schedule['total_updates'] - warm - 1)
        fraction = .1 + .9 * (1 + math.cos(math.pi * progress)) / 2
    peaks = {str(depth): schedule['encoder_peak_top'] * schedule['depth_decay'] ** (11 - depth)
             for depth in range(12)}
    peaks['head'] = schedule['head_peak']
    require({group['name'] for group in optimizer.param_groups} == set(peaks),
            'Depth optimizer groups changed')
    rates = {name: peak * fraction for name, peak in peaks.items()}
    for group in optimizer.param_groups:
        group['lr'] = rates[group['name']]
    return rates


def mixed_crop_update(model, gold_rows, weak_rows, optimizer, scaler, warp, crop, api,
                      *, apply_crop, update, amp, base_seed=52, original_collator=False):
    """Production RED + .1 positive-only weak loss; one optimizer step.

    Only the strong collator and common logical-forward RNG differ from the
    production mixed_update helper. CPU FP32 is a bounded implementation fixture.
    """
    from indoml.weak_supervision import positive_category_mask, weak_positive_loss
    require(all(row['supervision'] == 'weak_clip' and row['events'] is None
                and row['positive_categories'] for row in weak_rows), 'Weak rows changed')
    require(model.device == model.head_device, 'Model devices must match')
    require(not amp or model.device.type == 'cuda', 'AMP requires the production CUDA host')
    require(getattr(model, '_full_release_fp32_head', False), 'Native head must remain FP32')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rows, audio, decisions = crop.waveforms(gold_rows, apply_crop)
    require(not original_collator or not apply_crop, 'Original collator cannot read cropped audio')
    batch = api.collate_audio_intervals(gold_rows) if original_collator else collate_waveforms(rows, audio, api)
    strong_windows = len(batch['windows'])
    strong_frames = batch['frames']
    model.frequency_augmentation = warp
    with forward_rng(model.device, base_seed, update, 0):
        with torch.autocast(device_type=model.device.type, dtype=torch.float16, enabled=amp):
            output = model(batch['windows'], batch['descriptions'])
        output = {key: value.float() for key, value in output.items()}
        weights = torch.full((7,), 5., dtype=torch.float32, device=model.head_device)
        loss, components = api.interval_loss(output, batch, weights, model.head_device,
                                             'red', 100., 1., collar_weight=0.)
        require(bool(torch.isfinite(loss)), 'Nonfinite strong loss')
        scaler.scale(loss).backward()
    strong_loss = float(loss.detach())
    del loss, components, output, batch, audio
    model.frequency_augmentation = None
    batch = api.collate_clips(weak_rows)
    lengths = torch.tensor([item['frames'] for item in batch['descriptions']], dtype=torch.long)
    mask = positive_category_mask([row['positive_categories'] for row in weak_rows], device=model.head_device)
    weak_windows = len(batch['windows'])
    with forward_rng(model.device, base_seed, update, 1):
        with torch.autocast(device_type=model.device.type, dtype=torch.float16, enabled=amp):
            presence = model(batch['windows'], batch['descriptions'])['presence']
        loss, stats = weak_positive_loss(presence.float(), lengths, mask)
        require(bool(torch.isfinite(loss)), 'Nonfinite weak loss')
        scaler.scale(.1 * loss).backward()
    weak_loss = float(loss.detach())
    del presence, loss, batch, mask, lengths
    before = scaler.get_scale()
    scaler.unscale_(optimizer)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norms = [torch.linalg.vector_norm(parameter.grad.detach(), 2)
             for parameter in parameters if parameter.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack(norms), 2)
    finite = bool(torch.isfinite(norm))
    require(amp or finite, 'FP32 nonfinite gradients')
    if finite:
        torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    else:
        require(any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
                    if parameter.grad is not None), 'Aggregate gradient norm overflow')
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < before
    require(finite or skipped, 'Nonfinite gradients were not skipped')
    model.frequency_augmentation = warp
    return {'strong_loss': strong_loss, 'weak_loss': weak_loss,
            'raw_gradient_norm': float(norm.detach()) if finite else None,
            'skipped_update': skipped, 'scaler_before': before, 'scaler_after': scaler.get_scale(),
            'strong_clips': len(rows), 'weak_clips': len(weak_rows),
            'strong_windows': strong_windows, 'weak_windows': weak_windows,
            'strong_frames': strong_frames, 'weak_supervised_clips': stats['supervised_clips'],
            'strong_clip_ids': [row['clip_id'] for row in rows],
            'weak_clip_ids': [row['clip_id'] for row in weak_rows], 'crop_decisions': decisions}


def load_parent(api, native_bundle, parent_path, device):
    """Reconstruct the original native architecture, then load all exact P2 bytes."""
    from mixed_training import enforce_fp32_head
    require(sha(parent_path) == PARENT_SHA, 'Wrong retained P2 checkpoint')
    assets = json.loads((native_bundle / 'asset-manifest.json').read_text())
    e5_path = native_bundle / 'assets/parent-native-e5.pt'
    require(sha(e5_path) == assets['parent_checkpoint_sha256'], 'Architecture source checkpoint changed')
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        e5 = torch.load(e5_path, map_location='cpu', weights_only=True, mmap=True)
        parent = torch.load(parent_path, map_location='cpu', weights_only=True, mmap=True)
    require(parent['kind'] == 'full_vaani_inference_v1' and parent['arm'] == 'atst'
            and parent['completed_updates'] == 4972 and parent['decoder'] == PARENT_DECODER
            and parent['model_state_digest'] == PARENT_STATE, 'Retained P2 identity differs')
    native = api.FrozenATST(native_bundle / 'source/PretrainedSED',
                            native_bundle / 'assets/ATST-F_strong_1.pt', device='cpu')
    model = api.NativeIntervalATST.__new__(api.NativeIntervalATST)
    nn.Module.__init__(model)
    model._assemble(native, e5['model_config']['head_config'], 12)
    model.load_state_dict(parent['model_state'], strict=True)
    require(api.state_digest(model.state_dict()) == PARENT_STATE, 'P2 tensors changed during reconstruction')
    metadata = {key: parent[key] for key in ('kind', 'arm', 'decoder', 'metrics',
                                           'completed_updates', 'model_state_digest', 'runtime')}
    model.set_device(torch.device(device), torch.device(device))
    enforce_fp32_head(model)
    return model, metadata
