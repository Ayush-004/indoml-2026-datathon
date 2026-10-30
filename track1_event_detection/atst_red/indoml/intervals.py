"""Predict temporal intervals from frozen features, with optional RED dynamics.

Original implementation of a bounded experiment inspired by anchor-free temporal
localization and Schmid et al., arXiv:2601.04178. It is not a paper reproduction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import resolve_audio_path
from .metrics import score_records
from .model import choose_device, seed_everything
from .train import restore_rng, rng_state, save_checkpoint, write_json
from .transfer import (CATEGORIES, FeatureDataset, check_separate_splits, collate_features,
                       feature_path, file_hash, write_predictions)

MODEL_TYPE = 'frozen_feature_interval_model'


def interval_targets(row):
    """Assign overlapping same-category cells to their shortest covering event."""
    times = np.arange(row['num_frames']) * row['frame_hop'] + row['frame_offset']
    distances = np.zeros((len(times), len(CATEGORIES), 2), np.float32)
    owners = np.full((len(times), len(CATEGORIES)), -1, np.int64)
    spans = np.full(owners.shape, np.inf)
    for event_id, event in enumerate(row.get('events') or []):
        category = CATEGORIES.index(event['category'])
        duration = event['offset'] - event['onset']
        active = (times >= event['onset']) & (times < event['offset']) & (duration < spans[:, category])
        distances[active, category, 0] = times[active] - event['onset']
        distances[active, category, 1] = event['offset'] - times[active]
        owners[active, category] = event_id
        spans[active, category] = duration
    weights = np.zeros(owners.shape, np.float32)
    for event_id in np.unique(owners[owners >= 0]):
        active = owners == event_id
        weights[active] = 1 / active.sum()  # Equal aggregate contribution per assigned event.
    presence = (owners >= 0).astype(np.float32)
    previous = np.concatenate([np.zeros_like(presence[:1]), presence[:-1]], axis=0)
    return {'distances': torch.from_numpy(distances), 'event_weights': torch.from_numpy(weights),
            'onsets': torch.from_numpy(presence * (1 - previous)),
            'offsets': torch.from_numpy(previous * (1 - presence))}


class IntervalDataset(FeatureDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.mode != 'predict':
            item.update(interval_targets(item['row']))
            if not torch.equal(item['targets'].bool(), item['event_weights'] > 0):
                raise ValueError('Interval assignment must preserve the category-presence targets')
        return item


def collate_intervals(items):
    batch = collate_features(items)
    if 'distances' in items[0]:
        for name in ('distances', 'event_weights', 'onsets', 'offsets'):
            batch[name] = nn.utils.rnn.pad_sequence([item[name] for item in items], batch_first=True)
    return batch


def red_probabilities(logits):
    conditional = logits.sigmoid().reshape(*logits.shape[:2], len(CATEGORIES), 2)
    previous = torch.zeros_like(conditional[:, 0, :, 0])
    presence, onsets, offsets = [], [], []
    for step in conditional.unbind(dim=1):
        onset = step[:, :, 0] * (1 - previous)
        offset = step[:, :, 1] * previous
        previous = onset + previous - offset
        presence.append(previous)
        onsets.append(onset)
        offsets.append(offset)
    return tuple(torch.stack(values, dim=1) for values in (presence, onsets, offsets))


class IntervalModel(nn.Module):
    def __init__(self, feature_dim=768, projection_dim=256, hidden_size=128, dropout=.2,
                 mode='direct', duration_hidden=32, epn_context='per_class'):
        super().__init__()
        if mode not in {'direct', 'red'}:
            raise ValueError('mode must be direct or red')
        if epn_context not in {'per_class', 'joint'} or (mode != 'red' and epn_context != 'per_class'):
            raise ValueError('epn_context must be per_class, or joint for RED only')
        self.config = dict(feature_dim=feature_dim, projection_dim=projection_dim, hidden_size=hidden_size,
                           dropout=dropout, mode=mode, duration_hidden=duration_hidden)
        if epn_context != 'per_class':
            self.config['epn_context'] = epn_context
        self.mode = mode
        self.epn_context = epn_context
        self.project = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, projection_dim),
                                     nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(projection_dim, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden_size, len(CATEGORIES) * (2 if mode == 'red' else 1))
        if mode == 'red':
            context_classes = len(CATEGORIES) if epn_context == 'joint' else 1
            self.duration_rnn = nn.GRU(3 * context_classes, duration_hidden, num_layers=2, batch_first=True, bidirectional=True)
            self.duration_output = nn.Linear(2 * duration_hidden, 2 * context_classes)
            nn.init.constant_(self.classifier.bias, -3.)
        else:
            self.duration_output = nn.Linear(2 * hidden_size, 2 * len(CATEGORIES))
        nn.init.normal_(self.duration_output.weight, std=.01)
        nn.init.constant_(self.duration_output.bias, math.log(math.expm1(.2)))

    def forward(self, features, lengths):
        encoded = self.project(features)
        packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=features.shape[1])
        logits = self.classifier(encoded)
        if self.mode == 'red':
            presence, onsets, offsets = red_probabilities(logits)
            batch, frames, classes = presence.shape
            if self.epn_context == 'joint':
                temporal = torch.stack([presence, onsets, offsets], dim=-1).reshape(batch, frames, 3 * classes)
                duration_lengths = lengths.cpu()
            else:
                temporal = torch.stack([presence, onsets, offsets], dim=-1).permute(0, 2, 1, 3).reshape(batch * classes, frames, 3)
                duration_lengths = lengths.cpu().repeat_interleave(classes)
            packed = nn.utils.rnn.pack_padded_sequence(temporal, duration_lengths,
                                                      batch_first=True, enforce_sorted=False)
            packed, _ = self.duration_rnn(packed)
            temporal, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=frames)
            if self.epn_context == 'joint':
                distances = F.softplus(self.duration_output(temporal)).reshape(batch, frames, classes, 2)
            else:
                distances = F.softplus(self.duration_output(temporal)).reshape(batch, classes, frames, 2).permute(0, 2, 1, 3)
        else:
            presence = logits.sigmoid()
            previous = F.pad(presence[:, :-1], (0, 0, 1, 0))
            onsets, offsets = (presence - previous).relu(), (previous - presence).relu()
            distances = F.softplus(self.duration_output(encoded)).reshape(*presence.shape, 2)
        return dict(presence=presence, onsets=onsets, offsets=offsets, distances=distances + 1e-5, logits=logits)


def interval_loss(output, batch, positive_weights, device, mode, boundary_weight, iou_weight, collar_weight=0.):
    if not math.isfinite(collar_weight) or collar_weight < 0:
        raise ValueError('collar_weight must be finite and nonnegative')
    target, mask = batch['targets'].to(device), batch['mask'].to(device)[:, :, None]
    cells = mask.sum().clamp_min(1) * len(CATEGORIES)
    if mode == 'direct':
        presence = F.binary_cross_entropy_with_logits(output['logits'], target, pos_weight=positive_weights, reduction='none')
    else:
        probability = output['presence'].clamp(1e-6, 1 - 1e-6)
        presence = -(positive_weights * target * probability.log() + (1 - target) * torch.log1p(-probability))
    presence = (presence * mask).sum() / cells
    boundary = presence.new_zeros(())
    for name in ('onsets', 'offsets'):
        probability = output[name].clamp(1e-6, 1 - 1e-6)
        label = batch[name].to(device)
        focal = -(label * (1 - probability).square() * probability.log()
                  + (1 - label) * probability.square() * torch.log1p(-probability))
        boundary = boundary + (focal * mask).sum() / cells
    target_distance = batch['distances'].to(device)
    event_weight = batch['event_weights'].to(device) * mask
    intersection = torch.minimum(output['distances'], target_distance).sum(dim=-1)
    union = torch.maximum(output['distances'], target_distance).sum(dim=-1).clamp_min(1e-6)
    iou = ((1 - intersection / union) * event_weight).sum() / event_weight.sum().clamp_min(1)
    loss = presence + boundary_weight * boundary + iou_weight * iou
    collar = presence.new_zeros(())
    if collar_weight:
        # Only assigned, real cells contribute. The target distances recover the
        # assigned event's endpoints and duration; this is not scorer matching.
        assigned = event_weight > 0
        if assigned.any():
            frames = output['distances'].shape[1]
            centers = torch.stack([torch.arange(frames, device=device, dtype=target_distance.dtype)
                                   * row['frame_hop'] + row['frame_offset'] for row in batch['rows']])
            durations = target_distance.new_tensor([row['duration'] for row in batch['rows']])
            centers = centers[:, :, None].expand_as(event_weight)[assigned]
            durations = durations[:, None, None].expand_as(event_weight)[assigned]
            predicted = output['distances'][assigned]
            reference = target_distance[assigned]
            predicted_onsets = (centers - predicted[:, 0]).clamp_min(0.)
            predicted_offsets = torch.minimum(centers + predicted[:, 1], durations)
            reference_onsets, reference_offsets = centers - reference[:, 0], centers + reference[:, 1]
            tolerance = (.2 * reference.sum(dim=-1)).clamp_min(.05)
            outside = ((predicted_onsets - reference_onsets).abs() / tolerance - 1).relu()
            outside = outside + ((predicted_offsets - reference_offsets).abs() / tolerance - 1).relu()
            weights = event_weight[assigned]
            collar = (outside * weights).sum() / event_weight.sum().clamp_min(1)
        loss = loss + collar_weight * collar
    return loss, {'presence': float(presence.detach()), 'boundary': float(boundary.detach()),
                  'iou': float(iou.detach()), 'collar': float(collar.detach())}


def candidate_intervals(row, presence, distances, max_candidates=100):
    """One continuous interval per candidate time, ranked by event confidence."""
    result = []
    centers = np.arange(len(presence)) * row['frame_hop'] + row['frame_offset']
    for category in range(len(CATEGORIES)):
        values = presence[:, category]
        prefix = np.concatenate([[0.], values.cumsum(dtype=np.float64)])
        padded = np.pad(values, (2, 2), constant_values=-np.inf)
        peaks = np.logical_and.reduce([values >= padded[shift:shift + len(values)] for shift in range(5)])
        indices = np.flatnonzero(peaks)
        if len(indices):
            groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            indices = np.asarray([group[len(group) // 2] for group in groups])
        indices = indices[np.argsort(-values[indices], kind='stable')[:max_candidates]]
        candidates = []
        for index in indices:
            onset = max(0., centers[index] - float(distances[index, category, 0]))
            offset = min(row['duration'], centers[index] + float(distances[index, category, 1]))
            if offset <= onset:
                continue
            left = max(0, int(np.searchsorted(centers, onset, side='left')))
            right = min(len(values), int(np.searchsorted(centers, offset, side='left')))
            confidence = float((prefix[right] - prefix[left]) / (right - left)) if right > left else float(values[index])
            candidates.append({'onset': onset, 'offset': offset, 'score': confidence, 'peak': float(values[index])})
        result.append(sorted(candidates, key=lambda x: (-x['score'], -x['peak'], x['onset'])))
    return result


def decode_candidates(row, categories, decoder):
    events = []
    for candidates in categories:
        # Vectorized suppression keeps low-threshold validation searches bounded.
        candidates = [candidate for candidate in candidates
                      if min(candidate['score'], candidate['peak']) >= decoder['threshold']
                      and candidate['offset'] - candidate['onset'] >= decoder['min_duration']]
        boxes = np.asarray([[candidate['onset'], candidate['offset']] for candidate in candidates], dtype=np.float64)
        selected = []
        while len(boxes) and len(selected) < 50:
            first = boxes[0]
            selected.append({'onset': float(first[0]), 'offset': float(first[1])})
            others = boxes[1:]
            intersection = np.maximum(0., np.minimum(first[1], others[:, 1]) - np.maximum(first[0], others[:, 0]))
            union = np.maximum(first[1], others[:, 1]) - np.minimum(first[0], others[:, 0])
            boxes = others[(intersection <= 0) | (intersection / np.maximum(union, 1e-9) <= decoder['nms_iou'])]
        events.extend(selected)
    return {'clip_id': row['clip_id'], 'duration': row['duration'],
            'events': sorted(events, key=lambda event: (event['onset'], event['offset']))}


@torch.inference_mode()
def infer_candidates(model, loader, device):
    model.eval()
    records = []
    for batch in loader:
        output = model(batch['features'].to(device), batch['lengths'])
        presence, distances = output['presence'].cpu().numpy(), output['distances'].cpu().numpy()
        if not np.isfinite(presence).all() or not np.isfinite(distances).all():
            raise ValueError('Nonfinite interval predictions')
        for index, row in enumerate(batch['rows']):
            frames = row['num_frames']
            records.append((row, candidate_intervals(row, presence[index, :frames], distances[index, :frames])))
    return records


def select_decoder(records, thresholds, nms_ious, min_duration):
    reference = [row for row, _ in records]
    trials = []
    for threshold in thresholds:
        for nms_iou in nms_ious:
            decoder = dict(threshold=threshold, nms_iou=nms_iou, min_duration=min_duration)
            predictions = [decode_candidates(row, candidates, decoder) for row, candidates in records]
            trials.append({'decoder': decoder, 'metrics': score_records(reference, predictions)})
    best = max(trials, key=lambda result: (result['metrics']['combined'], result['decoder']['threshold'], -result['decoder']['nms_iou']))
    return best, trials


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('model_type') != MODEL_TYPE or checkpoint.get('categories') != list(CATEGORIES):
        raise ValueError('Not a compatible interval-model checkpoint')
    return checkpoint


def train(args):
    collar_weight = getattr(args, 'collar_weight', 0.)
    retain_epochs = getattr(args, 'retain_epoch_checkpoints', False)
    epn_context = getattr(args, 'epn_context', 'per_class')
    training_rng_seed = getattr(args, 'training_rng_seed', None)
    if not math.isfinite(collar_weight) or collar_weight < 0:
        raise ValueError('collar_weight must be finite and nonnegative')
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    training, validation = IntervalDataset(args.train, mode='train'), IntervalDataset(args.val, mode='val')
    check_separate_splits(training, validation)
    if training.feature_cache_id != validation.feature_cache_id or (training.frame_hop, training.frame_offset) != (validation.frame_hop, validation.frame_offset):
        raise ValueError('Training and validation feature identity/grid differ')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protected = {training.manifest, validation.manifest}
    for dataset in (training, validation):
        protected.update(feature_path(row, dataset.manifest) for row in dataset.rows)
        protected.update(resolve_audio_path(row, dataset.manifest).resolve() for row in dataset.rows)
    targets = [output / name for name in ('best.pt', 'last.pt', 'history.json', 'training-audit.json', 'inner-predictions.jsonl')]
    if retain_epochs:
        targets.extend(output / 'epoch-checkpoints' / f'epoch-{epoch:03d}.pt' for epoch in range(1, args.epochs + 1))
    if any(path.resolve() in protected or path.with_suffix(path.suffix + '.tmp').resolve() in protected for path in targets):
        raise ValueError('Training outputs must not overwrite input data')
    if not args.resume and any((output / name).exists() for name in ('best.pt', 'last.pt')):
        raise ValueError('Output already has checkpoints; use a new directory or resume')
    if args.resume:
        for name in ('best.pt', 'last.pt'):
            if (output / name).exists():
                load_checkpoint(output / name)  # Never replace a different model family's checkpoint.
    config = dict(mode=args.mode, projection_dim=args.projection_dim, hidden_size=args.hidden_size,
                  dropout=args.dropout, duration_hidden=args.duration_hidden)
    if epn_context != 'per_class':
        config['epn_context'] = epn_context
    model = IntervalModel(**config).to(device)
    if training_rng_seed is not None:
        torch.manual_seed(training_rng_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate_intervals)
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate_intervals)
    positive_weights, class_audit = training.positive_weights(args.pos_weight_cap)
    positive_weights = positive_weights.to(device)
    metadata = {'arguments': {**vars(args), 'collar_weight': collar_weight, 'retain_epoch_checkpoints': retain_epochs,
                             'epn_context': epn_context, 'training_rng_seed': training_rng_seed},
                'manifest_sha256': {'train': file_hash(training.manifest), 'validation': file_hash(validation.manifest)},
                'feature_cache_id': training.feature_cache_id, 'feature_grid': {'frame_hop': training.frame_hop, 'frame_offset': training.frame_offset},
                'train_clips': len(training), 'validation_clips': len(validation), 'excluded_training_rows': training.excluded,
                'class_weight_audit': class_audit, 'parameters': sum(p.numel() for p in model.parameters()),
                'torch_version': str(torch.__version__), 'interval_assignment': 'shortest containing event per category and cell',
                'implementation_sha256': file_hash(Path(__file__)),
                'regression_weighting': 'each event with assigned cells has total weight one',
                'candidate_selection': 'per-category five-frame local maxima with flat plateaus collapsed, top100 by presence; mean proposal confidence; within-category NMS; max50 events/category',
                'terminal_offset_supervision': 'no invented offset frame after the available sequence',
                'training_rng_policy': 'Optional torch reseed after model initialization; full checkpoint RNG overrides it on exact resume',
                'collar_loss': {'weight': collar_weight, 'tolerance': 'max(0.2 * assigned reference duration, 0.05) seconds',
                                'penalty': 'sum of relu(abs(endpoint error) / tolerance - 1) after inference clip-boundary clipping',
                                'weighting': 'same equal-event weights and positive-cell mask as IoU; no greedy matching'},
                'scope': 'epoch and decoder selection on supplied inner development partition'}
    first_epoch, history, best_score, best_checkpoint = 0, [], -1., None
    if args.resume:
        saved = load_checkpoint(args.resume)
        saved_config = {**saved['model_config'], 'epn_context': saved['model_config'].get('epn_context', 'per_class')}
        current_config = {**config, 'epn_context': epn_context}
        if saved_config != current_config or saved['training']['manifest_sha256'] != metadata['manifest_sha256']:
            raise ValueError('Resume model or manifests differ')
        for key in ('pos_weight_cap', 'boundary_weight', 'iou_weight', 'batch_size', 'thresholds', 'nms_ious', 'min_duration', 'seed'):
            if saved['training']['arguments'][key] != vars(args)[key]:
                raise ValueError(f'Resume policy differs: {key}')
        if saved['training']['arguments'].get('collar_weight', 0.) != collar_weight:
            raise ValueError('Resume policy differs: collar_weight')
        if saved['training']['arguments'].get('training_rng_seed') != training_rng_seed:
            raise ValueError('Resume policy differs: training_rng_seed')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        restore_rng(saved['rng_state'], generator, device)
        first_epoch, history, best_score, best_checkpoint = saved['epoch'], saved['history'], saved['best_combined'], saved['best_checkpoint']
        save_checkpoint(output / 'best.pt', best_checkpoint)
    metadata['effective_learning_rate'] = optimizer.param_groups[0]['lr']
    write_json(output / 'training-audit.json', metadata)
    print(json.dumps({'setup': metadata}), flush=True)
    for epoch in range(first_epoch + 1, args.epochs + 1):
        started = time.monotonic(); model.train(); loss_sum = 0.; components = dict(presence=0., boundary=0., iou=0., collar=0.)
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch['features'].to(device), batch['lengths'])
            loss, parts = interval_loss(prediction, batch, positive_weights, device, args.mode, args.boundary_weight, args.iou_weight,
                                        collar_weight=collar_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite interval loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
            loss_sum += float(loss.detach())
            for key in components:
                components[key] += parts[key]
            if step % 100 == 0:
                print(json.dumps({'progress': {'epoch': epoch, 'step': step, 'steps': len(loader), 'loss': loss_sum / step,
                                              'seconds': time.monotonic() - started}}), flush=True)
        records = infer_candidates(model, val_loader, device)
        selected, trials = select_decoder(records, args.thresholds, args.nms_ious, args.min_duration)
        entry = {'epoch': epoch, 'train_loss': loss_sum / len(loader), 'loss_components': {k: v / len(loader) for k, v in components.items()},
                 'validation': selected['metrics'], 'decoder': selected['decoder'], 'calibration': trials, 'seconds': time.monotonic() - started}
        history.append(entry)
        checkpoint = {'model_type': MODEL_TYPE, 'model_config': config, 'categories': list(CATEGORIES),
                      'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      'epoch': epoch, 'decoder': selected['decoder'], 'validation': selected['metrics'], 'training': metadata,
                      'feature_cache_id': training.feature_cache_id, 'feature_grid': metadata['feature_grid']}
        if selected['metrics']['combined'] > best_score:
            best_score, best_checkpoint = selected['metrics']['combined'], checkpoint
            save_checkpoint(output / 'best.pt', checkpoint)
            write_predictions(output / 'inner-predictions.jsonl', [decode_candidates(row, values, selected['decoder']) for row, values in records])
        save_checkpoint(output / 'last.pt', {**checkpoint, 'optimizer_state': optimizer.state_dict(), 'rng_state': rng_state(generator, device),
                                          'history': history, 'best_combined': best_score, 'best_checkpoint': best_checkpoint})
        if retain_epochs:
            epoch_path = output / 'epoch-checkpoints' / f'epoch-{epoch:03d}.pt'
            if epoch_path.exists():
                raise ValueError(f'Refusing to replace a retained epoch checkpoint: {epoch_path}')
            epoch_path.parent.mkdir(exist_ok=True)
            save_checkpoint(epoch_path, checkpoint)
        write_json(output / 'history.json', {'training': metadata, 'epochs': history, 'best_combined': best_score})
        print(json.dumps({k: v for k, v in entry.items() if k != 'calibration'}), flush=True)
        if args.patience and epoch - best_checkpoint['epoch'] >= args.patience:
            print(json.dumps({'early_stop': {'epoch': epoch, 'best_epoch': best_checkpoint['epoch'],
                                            'patience': args.patience}}), flush=True)
            break


def predict(args):
    torch.set_num_threads(args.threads)
    checkpoint = load_checkpoint(args.checkpoint)
    dataset = IntervalDataset(args.manifest, mode='predict')
    if dataset.feature_cache_id != checkpoint['feature_cache_id'] or checkpoint['feature_grid'] != {'frame_hop': dataset.frame_hop, 'frame_offset': dataset.frame_offset}:
        raise ValueError('Prediction features differ from the trained model')
    output = Path(args.output).resolve()
    protected = {Path(args.manifest).resolve(), Path(args.checkpoint).resolve()}
    protected.update(feature_path(row, dataset.manifest) for row in dataset.rows)
    protected.update(resolve_audio_path(row, dataset.manifest).resolve() for row in dataset.rows)
    if output in protected or output.with_suffix(output.suffix + '.tmp') in protected:
        raise ValueError('Prediction output must not overwrite inputs')
    device = choose_device(args.device)
    model = IntervalModel(**checkpoint['model_config']).to(device); model.load_state_dict(checkpoint['model_state'])
    records = infer_candidates(model, DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_intervals), device)
    write_predictions(output, [decode_candidates(row, candidates, checkpoint['decoder']) for row, candidates in records])
    print(json.dumps({'clips': len(dataset), 'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fitting = sub.add_parser('train')
    fitting.add_argument('--train', required=True); fitting.add_argument('--val', required=True)
    fitting.add_argument('--mode', choices=['direct', 'red'], default='direct')
    fitting.add_argument('--epochs', type=int, default=20); fitting.add_argument('--seed', type=int, default=42)
    fitting.add_argument('--patience', type=int, default=0, help='Stop after this many epochs without a new inner best; zero disables')
    fitting.add_argument('--lr', type=float, default=.001); fitting.add_argument('--pos-weight-cap', type=float, default=20.)
    fitting.add_argument('--projection-dim', type=int, default=256); fitting.add_argument('--hidden-size', type=int, default=128)
    fitting.add_argument('--duration-hidden', type=int, default=32); fitting.add_argument('--dropout', type=float, default=.2)
    fitting.add_argument('--epn-context', choices=['per_class', 'joint'], default='per_class', help='RED duration prediction uses separate per-class sequences or one joint seven-class sequence')
    fitting.add_argument('--training-rng-seed', type=int, help='Optional torch seed after model initialization; default preserves the original random stream')
    fitting.add_argument('--boundary-weight', type=float); fitting.add_argument('--iou-weight', type=float, default=1.)
    fitting.add_argument('--collar-weight', type=float, default=0., help='Assigned-event endpoint collar penalty; zero preserves the original loss')
    fitting.add_argument('--retain-epoch-checkpoints', action='store_true', help='Keep each completed calibrated epoch as an inference-only checkpoint')
    fitting.add_argument('--thresholds', nargs='+', type=float, default=[.1, .2, .35, .5, .65, .8, .9, .95])
    fitting.add_argument('--nms-ious', nargs='+', type=float, default=[0., .3, .5])
    fitting.add_argument('--min-duration', type=float, default=.08); fitting.add_argument('--resume')
    prediction = sub.add_parser('predict')
    prediction.add_argument('--checkpoint', required=True); prediction.add_argument('--manifest', required=True)
    for command in (fitting, prediction):
        command.add_argument('--output', required=True); command.add_argument('--batch-size', type=int, default=32)
        command.add_argument('--threads', type=int, default=2); command.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    if min(args.batch_size, args.threads) <= 0:
        parser.error('Batch size and threads must be positive')
    if args.command == 'train':
        if args.mode != 'red' and args.epn_context != 'per_class':
            parser.error('Joint EPN context is available for RED only')
        args.boundary_weight = (100. if args.mode == 'red' else 0.) if args.boundary_weight is None else args.boundary_weight
        if min(args.epochs, args.projection_dim, args.hidden_size, args.duration_hidden) <= 0 or args.patience < 0 or not 0 <= args.dropout < 1:
            parser.error('Invalid model dimensions, epochs or dropout')
        if any(not math.isfinite(x) or x < 0 for x in (args.boundary_weight, args.iou_weight, args.collar_weight, args.min_duration)) or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.pos_weight_cap) or args.pos_weight_cap < 1:
            parser.error('Invalid loss or training parameters')
        if any(not 0 < x < 1 for x in args.thresholds) or any(not 0 <= x < 1 for x in args.nms_ious):
            parser.error('Invalid proposal thresholds')
    (train if args.command == 'train' else predict)(args)


if __name__ == '__main__':
    main()
