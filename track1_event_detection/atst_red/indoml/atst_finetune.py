"""Fine-tune native ATST with stitched embeddings before a whole-clip GRU."""
from pathlib import Path
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import time
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_audio, resolve_audio_path
from .pretrained import (FrozenATST, FRAME_SAMPLES, WINDOW_SAMPLES, WINDOW_FRAMES,
                        window_starts, file_sha256)
from .transfer import (TemporalHead, load_transfer_checkpoint, category_targets,
                       decoded_record, write_predictions, CATEGORIES)
from .mobile import AudioWindows, DeadlineReached, parse_deadline, check_deadline
from .train import check_separate_splits, rng_state, restore_rng, save_checkpoint, write_json
from .metrics import score_records
from .submission import validate_predictions
from .ensemble import load_archive
from .transfer_calibrate import candidate_decoders
from torch.utils.data import DataLoader

TRIANGULAR = np.minimum((np.arange(WINDOW_FRAMES) + .5) / 125,
                       (WINDOW_FRAMES - np.arange(WINDOW_FRAMES) - .5) / 125).astype(np.float32)


def prepare_clips(rows):
    windows, descriptions = [], []
    for row in rows:
        audio = load_audio(Path(row['audio_path']), 16000)
        frames = (len(audio) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        if frames != row['num_frames'] or abs(len(audio) - round(row['duration'] * 16000)) > 1:
            raise ValueError(f"Audio duration changed for {row['clip_id']}")
        starts = [value // FRAME_SAMPLES for value in window_starts(len(audio))]
        total = np.zeros(frames, dtype=np.float32)
        for first in starts:
            count = min(WINDOW_FRAMES, frames - first)
            total[first:first + count] += TRIANGULAR[:count]
        assert np.all(total > 0)
        sections = []
        for first in starts:
            count = min(WINDOW_FRAMES, frames - first)
            sections.append({'window': len(windows), 'first': first, 'count': count,
                             'weight': torch.from_numpy(TRIANGULAR[:count] / total[first:first + count])})
            value = torch.from_numpy(audio[first * FRAME_SAMPLES:first * FRAME_SAMPLES + WINDOW_SAMPLES])
            windows.append(F.pad(value, (0, WINDOW_SAMPLES - len(value))))
        descriptions.append({'row': row, 'frames': frames, 'sections': sections})
    return torch.stack(windows), descriptions


class FineTuneATST(nn.Module):
    def __init__(self, source_root, encoder_checkpoint, head_checkpoint, train_blocks=12):
        super().__init__()
        adapter = FrozenATST(source_root, encoder_checkpoint, device='cpu')
        self.frontend = adapter.wrapper.atst_mel
        self.encoder = adapter.wrapper.atst
        self.provenance = adapter.provenance
        checkpoint = load_transfer_checkpoint(head_checkpoint)
        if checkpoint['model_config']['feature_dim'] != 768 or checkpoint['model_config']['head_type'] != 'gru':
            raise ValueError('Expected a trained 768-dimensional GRU head')
        if checkpoint['feature_cache_id'] != adapter.cache_id:
            raise ValueError('Initial head was not trained on these exact native ATST cache semantics')
        self.head = TemporalHead(**checkpoint['model_config'])
        self.head.load_state_dict(checkpoint['model_state'], strict=True)
        self.head_checkpoint = checkpoint
        self.frontend.requires_grad_(False).eval()
        self.encoder.requires_grad_(False)
        if not 1 <= train_blocks <= 12:
            raise ValueError('train_blocks must be between 1 and 12')
        for block in self.encoder.blocks[12 - train_blocks:]:
            block.requires_grad_(True)
        self.encoder.norm_frame.requires_grad_(True)
        if train_blocks == 12:
            self.encoder.patch_embed.requires_grad_(True)
            self.encoder.pos_embed.requires_grad_(True)
        # Native mask_embed is unused by get_intermediate_layers(mask=False).
        self.encoder.mask_embed.requires_grad_(False)
        self.train_blocks = train_blocks
        self.device = torch.device('cpu')
        self.head_device = torch.device('cpu')

    def set_device(self, device, head_device=None):
        self.device = torch.device(device)
        self.head_device = torch.device(head_device or device)
        self.encoder.to(self.device)
        self.head.to(self.head_device)
        self.frontend.cpu()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval()
        return self

    def encode_clips(self, windows, descriptions):
        if windows.device.type != 'cpu' or windows.ndim != 2 or windows.shape[1] != WINDOW_SAMPLES:
            raise ValueError('Expected CPU padded ten-second windows')
        # Native top_db clipping must be independent for each audio window.
        with torch.no_grad():
            spec = torch.cat([self.frontend(window[None]) for window in windows], dim=0)
        if spec.shape != (len(windows), 1, 64, 1001):
            raise ValueError('Native ATST frontend geometry changed')
        spec = spec.to(self.device)
        features = self.encoder.get_intermediate_layers(spec, torch.full((len(spec),), 1001, device=self.device), 1, scene=False)
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
        logits = self.head(padded, lengths)
        return (logits, clips) if return_features else logits


def supervised_loss(logits, descriptions, positive_weights=None):
    lengths = torch.tensor([item['frames'] for item in descriptions], device=logits.device)
    targets = nn.utils.rnn.pad_sequence([category_targets(item['row']) for item in descriptions], batch_first=True).to(logits.device)
    valid = torch.arange(logits.shape[1], device=logits.device)[None] < lengths[:, None]
    values = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weights, reduction='none')
    return (values * valid[:, :, None]).sum() / (valid.sum() * logits.shape[-1])


MODEL_TYPE = 'finetuned_native_atst_seven_category'
GRID = {'thresholds': [.2, .35, .5, .65, .8, .9, .95],
        'min_durations': [.04, .08, .12], 'merge_gaps': [0., .04, .08, .12]}


class AudioClips(AudioWindows):
    """Reuse the existing clean-Gold/geometry checks, but batch whole clips."""
    def __init__(self, manifest, *, supervised=True, test_only_clips=None):
        super().__init__(manifest, supervised=supervised)
        for row in self.rows:
            row['audio_path'] = str(resolve_audio_path(row, self.manifest))
        self.all_rows = self.rows
        if test_only_clips:
            self.rows = self.rows[:test_only_clips]
        self.num_windows = sum(len(window_starts(round(row['duration'] * 16000))) for row in self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_clips(rows):
    windows, descriptions = prepare_clips(rows)
    return {'windows': windows, 'descriptions': descriptions,
            'frames': sum(description['frames'] for description in descriptions)}


def check_geography(left, right):
    def keys(dataset):
        result = set()
        for row in dataset.rows:
            metadata = row.get('source', {}).get('metadata', {})
            values = [metadata.get(name) for name in ('state', 'district')]
            if all(isinstance(value, str) and value.strip() for value in values):
                result.add(tuple(' '.join(unicodedata.normalize('NFKC', value).split()).casefold() for value in values))
        return result
    if keys(left) & keys(right):
        raise ValueError('Fine-tuning and validation/outer share known geographic groups')


@torch.inference_mode()
def infer(model, dataset, batch_size, deadline):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_clips)
    for batch in loader:
        check_deadline(deadline)
        values = model(batch['windows'], batch['descriptions']).sigmoid().float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite fine-tuned ATST probabilities')
        for index, item in enumerate(batch['descriptions']):
            yield item['row'], values[index, :item['frames']]


def score(records, decoder):
    predictions = [decoded_record(row, values, decoder) for row, values in records]
    return score_records([row for row, _ in records], predictions)


def calibrate(records, deadline):
    decoders = candidate_decoders(None, **GRID)
    if len(decoders) != 84:
        raise ValueError('The predeclared inner grid must have exactly 84 candidates')
    candidates, selected = [], None
    for decoder in decoders:
        check_deadline(deadline)
        candidate = {'decoder': decoder, 'metrics': score(records, decoder)}
        candidates.append(candidate)
        if selected is None or candidate['metrics']['combined'] > selected['metrics']['combined'] + 1e-12:
            selected = candidate
    return selected, candidates


def archive_probabilities(path, records, checkpoint_hash, manifest, epoch, decoder):
    offsets = np.cumsum([0] + [len(values) for _, values in records], dtype=np.int64)
    metadata = {'model_type': MODEL_TYPE, 'checkpoint_sha256': checkpoint_hash,
                'manifest_sha256': file_sha256(Path(manifest)), 'categories': list(CATEGORIES),
                'frame_hop': .04, 'frame_offset': .02, 'selected_epoch': epoch, 'decoder': decoder}
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, probabilities=np.concatenate([values for _, values in records], axis=0),
                            offsets=offsets, clip_ids=np.asarray([row['clip_id'] for row, _ in records]),
                            metadata=np.asarray(json.dumps(metadata)))
    temporary.replace(path)


def load_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(value, dict) or value.get('format_version') != 1 or value.get('model_type') != MODEL_TYPE
            or value.get('categories') != list(CATEGORIES) or value.get('frame_grid') != {'frame_hop': .04, 'frame_offset': .02}):
        raise ValueError('Not a compatible native ATST fine-tuning checkpoint')
    return value


def snapshot(model, epoch, metadata, decoder, metrics):
    return {'format_version': 1, 'model_type': MODEL_TYPE, 'categories': list(CATEGORIES),
            'frame_grid': {'frame_hop': .04, 'frame_offset': .02}, 'epoch': epoch,
            'model_config': {'train_blocks': model.train_blocks, 'head_config': model.head.config},
            'model_state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            'training': metadata, 'decoder': decoder, 'validation': metrics}


def publish_best(output, checkpoint, selection):
    """Bind selection to immutable checkpoint bytes, independent of best.pt."""
    checkpoint['selection'] = dict(selection)
    save_checkpoint(output / 'best.pt', checkpoint)
    digest = file_sha256(output / 'best.pt')
    directory = output / 'selected-checkpoints'
    directory.mkdir(exist_ok=True)
    immutable = directory / f"epoch-{selection['epoch']:03d}-{digest[:16]}.pt"
    if immutable.exists():
        if file_sha256(immutable) != digest:
            raise ValueError('Immutable selected-checkpoint name collision')
    else:
        os.link(output / 'best.pt', immutable)
    return {**selection, 'checkpoint_path': str(immutable), 'checkpoint_sha256': digest}


def restore_selected(output, selection, expected_training, override=None):
    if selection['kind'] == 'frozen_baseline':
        return selection
    source = Path(override or selection.get('checkpoint_path', '')).resolve()
    expected = selection.get('checkpoint_sha256')
    if not expected or not source.is_file():
        raise ValueError('Resume requires the exact bound selected-best checkpoint; use --resume-best-checkpoint for a relocated copy')
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Selected-best checkpoint hash changed')
    checkpoint = load_checkpoint(io.BytesIO(payload))
    if (checkpoint['epoch'] != selection['epoch'] or checkpoint['decoder'] != selection['decoder']
            or checkpoint['validation'] != selection['metrics']
            or checkpoint['training']['manifest_sha256'] != expected_training['manifest_sha256']
            or checkpoint['training']['initialization'] != expected_training['initialization']):
        raise ValueError('Selected-best checkpoint identity differs from the resumed selection')
    directory = output / 'selected-checkpoints'
    directory.mkdir(exist_ok=True)
    destination = directory / f"epoch-{selection['epoch']:03d}-{expected[:16]}.pt"
    for path in (destination, output / 'best.pt'):
        if path.exists():
            existing_hash = file_sha256(path)
            if existing_hash == expected:
                continue
            if path != output / 'best.pt':
                raise ValueError(f'Immutable selected-checkpoint collision: {path}')
            existing = load_checkpoint(path)
            if (existing['training']['manifest_sha256'] != expected_training['manifest_sha256']
                    or existing['training']['initialization'] != expected_training['initialization']
                    or existing['training']['train_blocks'] != expected_training['train_blocks']):
                raise ValueError('Resume would overwrite an unrelated best checkpoint')
            preserved = directory / f"epoch-{existing['epoch']:03d}-{existing_hash[:16]}.pt"
            if not preserved.exists():
                os.link(path, preserved)
        temporary = path.with_suffix('.pt.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
    return {**selection, 'checkpoint_path': str(destination)}


def original_baseline(args, validation, head):
    expected = head.get('training', {}).get('manifest_sha256', {}).get('validation')
    validation_hash = file_sha256(validation.manifest)
    if not expected or expected != validation_hash:
        raise ValueError('Warmstart must use the exact original head validation manifest')
    specification = {'archive': {'path': str(Path(args.baseline_probabilities).resolve()),
                                 'sha256': file_sha256(Path(args.baseline_probabilities))},
                     'manifest': {'path': str(validation.manifest), 'sha256': validation_hash}}
    values = load_archive(specification, head, file_sha256(Path(args.head_checkpoint)), validation.all_rows)
    all_records = list(zip(validation.all_rows, values))
    measured = score(all_records, head['decoder'])
    if any(abs(measured[name] - head['validation'][name]) > 1e-12 for name in ('event_f1', 'segment_dice', 'combined')):
        raise ValueError('Original frozen baseline probability archive does not reproduce its saved score')
    lookup = {row['clip_id']: value for row, value in all_records}
    records = [(row, lookup[row['clip_id']]) for row in validation.rows]
    return records, score(records, head['decoder']), measured, specification


def fit(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    hard_deadline = parse_deadline(args.deadline_utc)
    # Keep time for selecting a completed epoch and its single outer comparison.
    train_deadline = None if hard_deadline is None else hard_deadline - args.finalization_reserve_seconds
    check_deadline(train_deadline)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and any((output / name).exists() for name in ('last.pt', 'best.pt', 'run-state.json')):
        raise ValueError('Use a fresh output directory or an explicit resume checkpoint')
    if args.test_only_clips and args.outer:
        raise ValueError('Test-only clipping must never request an outer comparison')
    if (output / 'outer-evaluation-state.json').exists():
        raise ValueError('This pilot already claimed its outer comparison; use a new explicit training stage')
    training = AudioClips(args.train, test_only_clips=args.test_only_clips)
    validation = AudioClips(args.val, test_only_clips=args.test_only_clips)
    check_separate_splits(training, validation)
    check_geography(training, validation)
    outer = None
    if args.outer:
        outer = AudioClips(args.outer, supervised=False)
        check_separate_splits(training, outer)
        check_geography(training, outer)
        check_separate_splits(validation, outer)
        check_geography(validation, outer)
    model = FineTuneATST(args.source_root, args.encoder_checkpoint, args.head_checkpoint, args.train_blocks)
    baseline_records, baseline_metrics, original_metrics, baseline_spec = original_baseline(args, validation, model.head_checkpoint)
    model.set_device(args.encoder_device, args.head_device)
    positive_weights, class_audit = training.positive_weights(args.pos_weight_cap)
    positive_weights = positive_weights.to(model.head_device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': args.encoder_lr},
                                  {'params': model.head.parameters(), 'lr': args.head_lr}], weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=collate_clips, generator=generator)
    hashes = {'train': file_sha256(training.manifest), 'validation': file_sha256(validation.manifest)}
    if outer:
        hashes['outer'] = file_sha256(outer.manifest)
    metadata = {'arguments': vars(args), 'manifest_sha256': hashes,
                'selection_scope': 'All model and decoder selection use INNER only; original frozen baseline wins ties',
                'initialization': {'encoder_checkpoint': str(Path(args.encoder_checkpoint).resolve()),
                                   'encoder_sha256': file_sha256(Path(args.encoder_checkpoint)),
                                   'head_checkpoint': str(Path(args.head_checkpoint).resolve()),
                                   'head_sha256': file_sha256(Path(args.head_checkpoint)),
                                   'source_root': str(Path(args.source_root).resolve()),
                                   'fresh_optimizer': True, 'fresh_rng_seed': args.seed},
                'native_provenance': model.provenance, 'trainer_sha256': file_sha256(Path(__file__)),
                'train_clips': len(training), 'train_windows': training.num_windows, 'validation_clips': len(validation),
                'train_blocks': args.train_blocks, 'positive_weight_cap': args.pos_weight_cap, 'class_weight_audit': class_audit,
                'trainable_encoder_parameters': sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                'head_parameters': sum(p.numel() for p in model.head.parameters()),
                'encoder_device': str(model.device), 'head_device': str(model.head_device), 'torch_version': str(torch.__version__),
                'stitching': 'Normalized triangular embeddings before whole-clip GRU; differentiable across CPU/MPS copy',
                'frontend': 'Original CPU ATSTMel, independent per window, no gradients or augmentation',
                'grid': GRID, 'baseline_specification': baseline_spec, 'original_frozen_metrics': original_metrics,
                'test_only_clips': args.test_only_clips, 'finalization_reserve_seconds': args.finalization_reserve_seconds}
    selected = {'kind': 'frozen_baseline', 'epoch': None, 'decoder': dict(model.head_checkpoint['decoder']), 'metrics': baseline_metrics}
    history, first_epoch, best_records, pending_validation = [], 0, None, None
    state = {'status': 'initializing', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'last_completed_epoch': 0, 'requested_epochs': args.epochs, 'hard_deadline_utc': args.deadline_utc}
    if args.resume:
        resume_payload = Path(args.resume).read_bytes()
        resume_hash = hashlib.sha256(resume_payload).hexdigest()
        resume = load_checkpoint(io.BytesIO(resume_payload))
        del resume_payload
        previous = resume['training']
        if (previous['manifest_sha256'] != hashes or previous['initialization'] != metadata['initialization']
                or previous['train_blocks'] != args.train_blocks or previous['positive_weight_cap'] != args.pos_weight_cap
                or previous['test_only_clips'] != args.test_only_clips or 'optimizer_state' not in resume):
            raise ValueError('Resume requires unchanged data, initialization, model and loss policy')
        model.load_state_dict(resume['model_state'], strict=True)
        optimizer.load_state_dict(resume['optimizer_state'])
        restore_rng(resume['rng_state'], generator, model.device)
        history, first_epoch, selected = resume['history'], resume['epoch'], resume['selected']
        metadata['initialization'] = previous['initialization']
        metadata['initial_live_baseline'] = previous.get('initial_live_baseline')
        metadata['resume_history'] = [*previous.get('resume_history', []),
                                      {'checkpoint': str(Path(args.resume).resolve()), 'sha256': resume_hash,
                                       'completed_epoch': first_epoch, 'destination': str(output),
                                       'utc': datetime.now(timezone.utc).isoformat()}]
        selected = restore_selected(output, selected, previous, args.resume_best_checkpoint)
        if not resume.get('validation_complete', False):
            pending_validation = resume.get('pending_validation')
            if not pending_validation or pending_validation['epoch'] != first_epoch:
                raise ValueError('Completed optimizer epoch lacks its pending-validation record')
        state['last_completed_epoch'] = first_epoch
        del resume
    metadata['effective_learning_rates'] = [group['lr'] for group in optimizer.param_groups]
    write_json(output / 'training-audit.json', metadata)
    write_json(output / 'run-state.json', state)
    print(json.dumps({'setup': metadata}), flush=True)
    interrupted_epoch = None

    def finish_validation(epoch, completed, train_loss, training_seconds, started):
        nonlocal selected, best_records
        if any(entry['epoch'] == epoch for entry in history):
            raise ValueError('A completed epoch would be calibrated twice in the same resumed history')
        state.update(status='calibrating_inner', last_completed_epoch=epoch)
        write_json(output / 'run-state.json', state)
        records = list(infer(model, validation, args.inference_batch_size, hard_deadline))
        candidate, candidates = calibrate(records, hard_deadline)
        write_json(output / f'epoch-{epoch:03d}-calibration.json', {'epoch': epoch, 'candidate_count': 84,
                   'selected': candidate, 'candidates': candidates})
        history.append({'epoch': epoch, 'train_loss': train_loss, 'validation': candidate['metrics'],
                        'decoder': candidate['decoder'], 'training_seconds': training_seconds,
                        'total_seconds': time.monotonic() - started})
        completed.update(decoder=candidate['decoder'], validation=candidate['metrics'])
        if candidate['metrics']['combined'] > selected['metrics']['combined'] + 1e-12:
            selected = publish_best(output, completed, {'kind': 'finetuned_epoch', 'epoch': epoch, **candidate})
            best_records = records
        save_checkpoint(output / 'last.pt', {**completed, 'optimizer_state': optimizer.state_dict(),
                        'rng_state': rng_state(generator, model.device), 'history': history, 'selected': selected,
                        'validation_complete': True})
        write_json(output / 'history.json', {'training': metadata, 'epochs': history, 'selected': selected})
        state.update(status='epoch_complete', last_calibrated_epoch=epoch, selected=selected)
        write_json(output / 'run-state.json', state)
        print(json.dumps(history[-1]), flush=True)

    try:
        if not args.resume:
            started = time.monotonic()
            live_records = list(infer(model, validation, args.inference_batch_size, hard_deadline))
            live_metrics = score(live_records, selected['decoder'])
            difference = max(float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(live_records, baseline_records))
            metadata['initial_live_baseline'] = {'frozen_metrics': baseline_metrics, 'live_metrics': live_metrics,
                       'live_minus_frozen_combined': live_metrics['combined'] - baseline_metrics['combined'],
                       'maximum_probability_difference': difference, 'decoder': selected['decoder'],
                       'inference_seconds': time.monotonic() - started,
                       'interpretation': 'Diagnostic of FP16 cache/runtime rounding; not an extra model-selection candidate'}
            del live_records
        if metadata.get('initial_live_baseline') is not None:
            write_json(output / 'initial-live-baseline.json', metadata['initial_live_baseline'])
        write_json(output / 'training-audit.json', metadata)
        if pending_validation:
            state['current_epoch'] = first_epoch
            finish_validation(first_epoch, snapshot(model, first_epoch, metadata, None, None),
                              pending_validation['train_loss'], pending_validation['training_seconds'], time.monotonic())
        for epoch in range(first_epoch + 1, args.epochs + 1):
            check_deadline(train_deadline)
            state.update(status='training', current_epoch=epoch)
            write_json(output / 'run-state.json', state)
            model.train()
            started = time.monotonic()
            total_loss, total_frames = 0., 0
            for step, batch in enumerate(loader, 1):
                check_deadline(train_deadline)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch['windows'], batch['descriptions'])
                loss = supervised_loss(logits, batch['descriptions'], positive_weights)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fine-tuning loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * batch['frames']
                total_frames += batch['frames']
                if args.log_every and step % args.log_every == 0:
                    print(json.dumps({'progress': {'epoch': epoch, 'step': step, 'steps': len(loader),
                          'train_loss': total_loss / total_frames, 'seconds': time.monotonic() - started,
                          'gradient_norm': float(norm.detach().cpu())}}), flush=True)
            training_seconds = time.monotonic() - started
            # Publish the complete optimizer epoch before spending validation time.
            completed = snapshot(model, epoch, metadata, None, None)
            save_checkpoint(output / 'last.pt', {**completed, 'optimizer_state': optimizer.state_dict(),
                            'rng_state': rng_state(generator, model.device), 'history': history, 'selected': selected,
                            'validation_complete': False, 'pending_validation': {'epoch': epoch,
                            'train_loss': total_loss / total_frames, 'training_seconds': training_seconds}})
            state['last_completed_epoch'] = epoch
            finish_validation(epoch, completed, total_loss / total_frames, training_seconds, started)
    except DeadlineReached:
        interrupted_epoch = state.get('current_epoch')
        state.update(status='training_cutoff', note='Incomplete optimizer epoch is discarded; completed checkpoint remains available')
        print(json.dumps({'training_cutoff': state}), flush=True)
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        write_json(output / 'run-state.json', state)
        raise
    state.update(status='finalizing', interrupted_epoch=interrupted_epoch, selected=selected)
    write_json(output / 'run-state.json', state)
    report = {'selection': selected, 'frozen_baseline_metrics': baseline_metrics,
              'original_full_inner_baseline_metrics': original_metrics, 'completed_epochs': state['last_completed_epoch'],
              'calibrated_epochs': [entry['epoch'] for entry in history], 'requested_epochs': args.epochs,
              'interrupted_epoch': interrupted_epoch, 'actual_inner_grid_trials': 84 * len(history),
              'test_only_clips': args.test_only_clips, 'outer_evaluations_started': 0, 'outer_evaluations_completed': 0,
              'selection_scope': metadata['selection_scope'], 'hard_deadline_utc': args.deadline_utc}
    try:
        if selected['kind'] == 'finetuned_epoch':
            selected = restore_selected(output, selected, metadata)
            report['selection'] = selected
            chosen_payload = Path(selected['checkpoint_path']).read_bytes()
            if hashlib.sha256(chosen_payload).hexdigest() != selected['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed before final inference')
            chosen = load_checkpoint(io.BytesIO(chosen_payload))
            del chosen_payload
            model.load_state_dict(chosen['model_state'], strict=True)
            if best_records is None:
                best_records = list(infer(model, validation, args.inference_batch_size, hard_deadline))
            selected_hash = selected['checkpoint_sha256']
            archive_probabilities(output / 'inner-probabilities.npz', best_records, selected_hash, validation.manifest,
                                  selected['epoch'], selected['decoder'])
            report['selected_checkpoint_sha256'] = selected_hash
            if outer:
                check_deadline(hard_deadline)
                guard = output / 'outer-evaluation-state.json'
                descriptor = os.open(guard, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, 'w') as stream:
                    json.dump({'status': 'running', 'checkpoint_sha256': selected_hash,
                               'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
                report['outer_evaluations_started'] = 1
                outer_records = list(infer(model, outer, args.inference_batch_size, hard_deadline))
                predictions = [decoded_record(row, values, selected['decoder']) for row, values in outer_records]
                validate_predictions(predictions, outer.rows)
                write_predictions(output / 'outer-predictions.jsonl', predictions)
                archive_probabilities(output / 'outer-probabilities.npz', outer_records, selected_hash, outer.manifest,
                                      selected['epoch'], selected['decoder'])
                report.update(outer_metrics=score_records(outer.rows, predictions), outer_evaluations_completed=1)
                write_json(guard, {'status': 'complete', **report})
        else:
            report['baseline_retained_reason'] = 'No completed trained epoch exceeded the original frozen INNER baseline; no new OUTER comparison'
        state.update(status='complete', report=report)
    except DeadlineReached:
        state.update(status='cutoff_during_finalization', report=report)
    except Exception as exc:
        state.update(status='failed', error=str(exc), report=report)
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'report.json', report)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'run-state.json', state)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'encoder-checkpoint', 'head-checkpoint', 'baseline-probabilities', 'train', 'val', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--outer', help='Optional original outer manifest; evaluated once only if a trained epoch wins INNER')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--encoder-lr', type=float, default=1e-6)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--pos-weight-cap', type=float, default=5.)
    parser.add_argument('--gradient-clip', type=float, default=5.)
    parser.add_argument('--train-blocks', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--inference-batch-size', type=int, default=8)
    parser.add_argument('--encoder-device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--head-device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--finalization-reserve-seconds', type=float, default=180.)
    parser.add_argument('--resume')
    parser.add_argument('--resume-best-checkpoint', help='Relocated exact selected-best checkpoint, validated against its bound SHA-256')
    parser.add_argument('--test-only-clips', type=int, help='Explicit smoke-test limit on train/val; prohibits OUTER and marks all outputs test-only')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.inference_batch_size, args.threads) <= 0 or not 1 <= args.train_blocks <= 12:
        parser.error('Epochs/batches/threads must be positive and train-blocks inside1..12')
    if any(not math.isfinite(value) or value <= 0 for value in (args.encoder_lr, args.head_lr, args.gradient_clip)):
        parser.error('Learning rates and gradient clip must be positive and finite')
    if args.pos_weight_cap < 1 or not math.isfinite(args.pos_weight_cap) or args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        parser.error('Invalid class cap or weight decay')
    if args.log_every < 0 or args.finalization_reserve_seconds < 0 or not math.isfinite(args.finalization_reserve_seconds):
        parser.error('Invalid log frequency or finalization reserve')
    if args.test_only_clips is not None and args.test_only_clips <= 0:
        parser.error('test-only-clips must be positive')
    return fit(args)


if __name__ == '__main__':
    main()
