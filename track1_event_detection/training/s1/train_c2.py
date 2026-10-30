"""C2 (corrected mean-teacher consistency) on the Mac GPU, at reduced scale.

The recipe is the C2 consistency kernel, arm C2 (not part of a final model; S1 = its CTRL arm, see train_s1.py):

    strong      Gold-train + generated variants, P2's interval loss (red, boundary 100,
                IoU 1, class positive weight 5), frequency warp
    weak        Silver/Bronze clip tags at weight 0.1
    consistency MSE between the student's and the EMA teacher's PRESENCE maps only
                (distance_weight 0), w_c ramped 0 -> 1 over the first 1,000 updates with
                the Tarvainen-Valpola sigmoid; the student view hides faint events (with
                probability 0.5 a random Silver/Bronze clip mixed in at SNR U[0, 15] dB,
                otherwise 1-3 zeroed spans of 0.2-0.6 s) and gets the frequency warp; the
                teacher sees the clean clip. EMA decay 0.99, applied after every step.

Reduced scale for this machine (the Kaggle ratios halved on the strong stream):
    8 strong (6 Gold-train + 2 generated), 16 weak at 0.1, 4 unlabelled
    (2 organizer TEST natural + 2 Silver/Bronze).

Everything else is P2's: AdamW, weight decay 1e-4, gradient clip 5, encoder 1e-6 /
head 1e-4 with warmup then cosine, depth decay 0.95.

Differences forced by the platform, and nothing else:
  * fp32 throughout (no CUDA autocast, no GradScaler): MPS has no fp16 GradScaler path.
  * encoder on MPS, head on CPU, which is what every previous Mac run of P2 uses.
  * evaluation does not happen here. The trainer writes checkpoint weights and
    eval_cpu.py scores them on the CPU in a separate process, so the GPU keeps training.
  * wavlm_mps_patch.py IS applied: on torch 2.8 MPS the fused LayerNorm
    backward returns garbage affine gradients whenever its input does not require grad.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_local as L  # noqa: E402

GPU_REQUEST = Path(os.environ.get('T1_GPU_REQUEST', str(HERE / 'GPU_REQUEST')))   # a file here pauses training
STARTED = time.monotonic()

ORIGINAL = {
    'optimizer': {'type': 'AdamW', 'weight_decay': 1e-4, 'gradient_norm_clip': 5.0},
    'construction_lr': {'encoder_lr': 1e-6, 'head_lr': 1e-4},
    'schedule': {'depth_decay': .95, 'warmup_fraction': 139 / 2776},
    'strong_loss': {'mode': 'red', 'class_positive_weight': 5.0, 'boundary_weight': 100.0,
                    'iou_weight': 1.0, 'collar_weight': 0.0},
    'weak_loss': {'lambda': 0.1},
    'frequency_warp': {'probability': .5, 'min_scale': .6, 'max_scale': 1., 'seed': 50},
}
CONSISTENCY = {
    'ema_decay': .99,
    'distance_weight': 0.,
    'ramp_updates': 1000,
    'teacher_presence_threshold': .5,
    'augmentation': {'gain_db': [0., 0.], 'shift_ms': 0., 'shift_probability': 0.,
                     'frequency_warp': True,
                     'mixup_probability': .5, 'mixup_snr_db': [0., 15.],
                     'time_mask_spans': [1, 3], 'time_mask_seconds': [.2, .6],
                     'augmentation_seed': 20260918, 'noise_seed': 20260919},
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(message, **details):
    item = {'utc': utc(), 'elapsed_s': round(time.monotonic() - STARTED, 1), 'message': message}
    item.update(details)
    print(json.dumps(item, allow_nan=False, default=str), flush=True)


# ------------------------------------------------------------------- memory guard

class Guard:
    """The memory guard of the earlier Mac fine-tune runs, unchanged thresholds."""
    POLICY = {'maximum_mps_driver_bytes': 15_000_000_000, 'minimum_system_free_percent': 15,
              'system_query_interval_seconds': 2.0}

    def __init__(self, torch_module, enabled=True):
        self.torch, self.enabled = torch_module, enabled
        self.last_query = -float('inf')
        self.maximum_driver_bytes = 0
        self.maximum_rss_bytes = 0
        self.snapshot = None

    def snapshot_now(self):
        free = None
        for attempt in range(3):
            try:
                free = subprocess.run(['/usr/bin/memory_pressure', '-Q'], check=True, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      timeout=30).stdout
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 2:
                    raise
                time.sleep(1.0)
        marker = 'System-wide memory free percentage:'
        line = next((x for x in free.splitlines() if marker in x), None)
        if line is None or not line.rstrip().endswith('%'):
            raise ValueError('memory_pressure output changed')
        percent = int(line.rsplit(':', 1)[1].strip().rstrip('%'))
        flag = int(subprocess.run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                                  check=True, text=True, stdout=subprocess.PIPE,
                                  timeout=30).stdout.strip())
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        driver = (self.torch.mps.driver_allocated_memory()
                  if self.torch.backends.mps.is_available() else 0)
        return {'utc': utc(), 'system_wide_free_percent': percent, 'pressure_flag': flag,
                'peak_rss_bytes': rss, 'mps_driver_bytes': driver,
                'mps_tensor_bytes': (self.torch.mps.current_allocated_memory()
                                     if self.torch.backends.mps.is_available() else 0)}

    def check(self, stage, force=False):
        if not self.enabled:
            return None
        if not force and time.monotonic() - self.last_query < self.POLICY['system_query_interval_seconds']:
            return self.snapshot
        self.snapshot = self.snapshot_now()
        self.last_query = time.monotonic()
        self.maximum_driver_bytes = max(self.maximum_driver_bytes, self.snapshot['mps_driver_bytes'])
        self.maximum_rss_bytes = max(self.maximum_rss_bytes, self.snapshot['peak_rss_bytes'])
        if self.snapshot['mps_driver_bytes'] > self.POLICY['maximum_mps_driver_bytes']:
            raise RuntimeError(f'guard stop at {stage}: mps_driver_over_15gb')
        if self.snapshot['system_wide_free_percent'] < self.POLICY['minimum_system_free_percent']:
            raise RuntimeError(f'guard stop at {stage}: system_free_below_15_percent')
        if self.snapshot['pressure_flag'] == 4:
            raise RuntimeError(f'guard stop at {stage}: critical_pressure_flag_4')
        return self.snapshot


# ------------------------------------------------------------------- optimization

def depth_optimizer(model, api, weight_decay, encoder_lr, head_lr):
    from types import SimpleNamespace
    from indoml.optimization import depth_parameter_names, regroup_depth, parameter_names
    settings = SimpleNamespace(encoder_lr=encoder_lr, head_lr=head_lr, weight_decay=weight_decay)
    optimizer = api.make_optimizer(model, settings)
    expected = {key: sorted(names) for key, names in depth_parameter_names(model).items()}
    regroup_depth(optimizer, model, expected)
    return optimizer, parameter_names(optimizer, model)


def schedule_for(total_updates, encoder_peak_top, head_peak):
    warm = max(1, round(ORIGINAL['schedule']['warmup_fraction'] * total_updates))
    return {'total_updates': total_updates,
            'warmup_updates': min(warm, max(1, total_updates - 1)),
            'encoder_peak_top': encoder_peak_top, 'head_peak': head_peak,
            'depth_decay': ORIGINAL['schedule']['depth_decay'],
            'initial_and_final_peak_fraction': .1}


def consistency_ramp(update, total):
    if total <= 0:
        return 1.
    fraction = min(1., max(0., update / total))
    return math.exp(-5. * (1. - fraction) ** 2)


# ------------------------------------------------------------- the student's view
# Verbatim from the kernel (indoml-p2-consistency-v2.py::augment_view /
# collate_from_waveforms / consistency_terms), with the distance term detached because
# C2 sets distance_weight to 0 and only reports it.

def load_waveforms(rows):
    from indoml.data import load_audio
    return [load_audio(row['audio_path'], 16000) for row in rows]


def noise_clip(pool, generator):
    import torch
    from indoml.data import load_audio
    index = int(torch.randint(len(pool), (), generator=generator))
    return load_audio(pool[index]['audio_path'], 16000), pool[index]['clip_id']


def augment_view(audio, generator, config, noise_pool=None, noise_generator=None):
    import numpy as np
    import torch
    value = audio.astype(np.float32)
    decision = {}
    low, high = config['gain_db']
    if high > low:
        gain_db = float(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator))
        value = value * np.float32(10. ** (gain_db / 20.))
        decision['gain_db'] = gain_db
    if config['shift_probability'] > 0 and float(torch.rand((), dtype=torch.float64, generator=generator)) < config['shift_probability']:
        limit = int(round(config['shift_ms'] * 16.))
        shift = int(torch.randint(-limit, limit + 1, (), generator=generator))
        if shift:
            value = np.roll(value, shift)
            if shift > 0:
                value[:shift] = 0.
            else:
                value[shift:] = 0.
            decision['shift_samples'] = shift
    mix = (noise_pool is not None
           and float(torch.rand((), dtype=torch.float64, generator=generator))
           < config.get('mixup_probability', 0.))
    if mix:
        noise, noise_id = noise_clip(noise_pool, noise_generator)
        if len(noise) < len(value):
            noise = np.tile(noise, int(np.ceil(len(value) / max(1, len(noise)))))
        start = 0
        if len(noise) > len(value):
            start = int(torch.randint(len(noise) - len(value) + 1, (), generator=generator))
        noise = noise[start:start + len(value)].astype(np.float32)
        low_db, high_db = config['mixup_snr_db']
        snr = float(torch.empty((), dtype=torch.float64).uniform_(low_db, high_db, generator=generator))
        signal_power = float(np.mean(value ** 2))
        noise_power = float(np.mean(noise ** 2))
        if noise_power > 1e-12 and signal_power > 1e-12:
            scale = np.sqrt(signal_power / (noise_power * 10. ** (snr / 10.)))
            value = value + np.float32(scale) * noise
            decision.update(mixed_clip=noise_id, snr_db=snr)
        else:
            decision['mix_skipped'] = True
    else:
        low_n, high_n = config['time_mask_spans']
        count = int(torch.randint(low_n, high_n + 1, (), generator=generator))
        low_s, high_s = config['time_mask_seconds']
        spans = []
        for _ in range(count):
            seconds = float(torch.empty((), dtype=torch.float64).uniform_(low_s, high_s, generator=generator))
            width = min(len(value), int(round(seconds * 16000)))
            if width <= 0 or width >= len(value):
                continue
            start = int(torch.randint(len(value) - width + 1, (), generator=generator))
            value[start:start + width] = 0.
            spans.append([start, start + width])
        decision['masked_spans'] = spans
    peak = float(np.max(np.abs(value))) if value.size else 0.
    if peak > 1.:
        value = value / np.float32(peak)
        decision['rescaled_peak'] = peak
    import numpy as _np
    return _np.ascontiguousarray(value, dtype=_np.float32), decision


def collate_from_waveforms(rows, waveforms):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from indoml.atst_finetune import TRIANGULAR
    from indoml.pretrained import FRAME_SAMPLES, WINDOW_SAMPLES, WINDOW_FRAMES, window_starts
    windows, descriptions = [], []
    for row, audio in zip(rows, waveforms):
        frames = (len(audio) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        if frames != row['num_frames']:
            raise ValueError('Audio duration changed for ' + row['clip_id'])
        starts = [value // FRAME_SAMPLES for value in window_starts(len(audio))]
        total = np.zeros(frames, dtype=np.float32)
        for first in starts:
            count = min(WINDOW_FRAMES, frames - first)
            total[first:first + count] += TRIANGULAR[:count]
        if not bool(np.all(total > 0)):
            raise ValueError('Native stitching has uncovered frames')
        sections = []
        for first in starts:
            count = min(WINDOW_FRAMES, frames - first)
            sections.append({'window': len(windows), 'first': first, 'count': count,
                             'weight': torch.from_numpy(TRIANGULAR[:count] / total[first:first + count])})
            value = torch.from_numpy(audio[first * FRAME_SAMPLES:first * FRAME_SAMPLES + WINDOW_SAMPLES])
            windows.append(F.pad(value, (0, WINDOW_SAMPLES - len(value))))
        descriptions.append({'row': row, 'frames': frames, 'sections': sections})
    lengths = torch.tensor([item['frames'] for item in descriptions], dtype=torch.long)
    return {'windows': torch.stack(windows), 'descriptions': descriptions, 'lengths': lengths,
            'frames': int(lengths.sum())}


def consistency_terms(student_output, teacher_output, lengths, device, threshold, distance_weight):
    """C2: MSE on presence over every real cell. The distance L1 is reported, not trained."""
    import torch
    presence_s = student_output['presence']
    presence_t = teacher_output['presence']
    frames = presence_s.shape[1]
    mask = torch.arange(frames, device=device)[None] < lengths.to(device)[:, None]
    cells = mask[..., None].to(presence_s.dtype)
    real = cells.sum() * presence_s.shape[-1]
    presence_loss = (((presence_s - presence_t) ** 2) * cells).sum() / real.clamp(min=1.)
    active = ((presence_t > threshold) & mask[..., None]).to(presence_s.dtype)
    with torch.no_grad():
        distance_l1 = (student_output['distances'].detach()
                       - teacher_output['distances']).abs().sum(-1)
        distance_report = float((distance_l1 * active).sum() / active.sum().clamp(min=1.))
    total = presence_loss if distance_weight == 0 else (
        presence_loss + distance_weight * (
            (student_output['distances'] - teacher_output['distances']).abs().sum(-1)
            * active).sum() / active.sum().clamp(min=1.))
    return total, {'distance_weight': distance_weight,
                   'presence_mse': float(presence_loss.detach()),
                   'distance_l1_reported': distance_report,
                   'active_cells': float(active.sum()), 'real_cells': float(real),
                   'teacher_mean_presence': float((presence_t * cells).sum() / real.clamp(min=1.)),
                   'student_mean_presence': float((presence_s.detach() * cells).sum()
                                                  / real.clamp(min=1.))}


# ------------------------------------------------------------------------ the teacher

def build_teacher(student, rt, encoder_device, head_device):
    import torch
    teacher, _ = rt.load_p2(encoder_device, head_device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.frequency_augmentation = None
    trainable = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    teacher_parameters = dict(teacher.named_parameters())
    groups = {}
    for name, parameter in student.named_parameters():
        if name not in trainable:
            continue
        other = teacher_parameters[name]
        if other.device != parameter.device or other.shape != parameter.shape:
            raise ValueError('Teacher and student differ at ' + name)
        groups.setdefault(str(parameter.device), ([], []))
        groups[str(parameter.device)][0].append(other)
        groups[str(parameter.device)][1].append(parameter)
    audit = {'averaged_tensors': sum(len(v[0]) for v in groups.values()),
             'averaged_parameters': int(sum(p.numel() for v in groups.values() for p in v[1])),
             'devices': {device: len(value[0]) for device, value in groups.items()},
             'copied_tensors': len(teacher.state_dict()) - sum(len(v[0]) for v in groups.values())}
    return teacher, groups, audit


def ema_step(groups, decay):
    import torch
    with torch.no_grad():
        for teacher_parameters, student_parameters in groups.values():
            torch._foreach_mul_(teacher_parameters, decay)
            torch._foreach_add_(teacher_parameters, student_parameters, alpha=1. - decay)


# ------------------------------------------------------------------------- one update

def one_update(model, teacher, api, strong_rows, weak_rows, unlabelled_rows, optimizer, warp, *,
               weak_weight, consistency_weight, generator, noise_pool, noise_generator):
    import torch
    from indoml.weak_supervision import positive_category_mask, weak_positive_loss
    device = model.head_device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch = api.collate_audio_intervals(strong_rows)
    strong_windows = len(batch['windows'])
    model.frequency_augmentation = warp
    output = model(batch['windows'], batch['descriptions'])
    output = {key: value.float() for key, value in output.items()}
    weights = torch.full((len(L.CATEGORIES),), ORIGINAL['strong_loss']['class_positive_weight'],
                         dtype=torch.float32, device=device)
    loss, components = api.interval_loss(output, batch, weights, device, 'red',
                                         ORIGINAL['strong_loss']['boundary_weight'],
                                         ORIGINAL['strong_loss']['iou_weight'], collar_weight=0.)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite strong loss')
    loss.backward()
    strong_value = float(loss.detach())
    del loss, output, batch
    model.frequency_augmentation = None

    weak_value, weak_windows = None, 0
    if weak_rows and weak_weight > 0:
        weak_batch = api.collate_clips(weak_rows)
        lengths = torch.tensor([item['frames'] for item in weak_batch['descriptions']],
                               dtype=torch.long)
        mask = positive_category_mask([row['positive_categories'] for row in weak_rows],
                                      device=device)
        weak_windows = len(weak_batch['windows'])
        presence = model(weak_batch['windows'], weak_batch['descriptions'])['presence']
        weak_loss, _ = weak_positive_loss(presence.float(), lengths, mask)
        if not bool(torch.isfinite(weak_loss)):
            raise RuntimeError('Nonfinite weak loss')
        (weak_weight * weak_loss).backward()
        weak_value = float(weak_loss.detach())
        del presence, weak_loss, weak_batch, mask, lengths

    parts, unlabelled_windows = {}, 0
    if teacher is not None and unlabelled_rows and consistency_weight > 0:
        clean = load_waveforms(unlabelled_rows)
        views, decisions = [], []
        for audio in clean:
            value, decision = augment_view(audio, generator, CONSISTENCY['augmentation'],
                                           noise_pool=noise_pool, noise_generator=noise_generator)
            views.append(value)
            decisions.append(decision)
        teacher_batch = collate_from_waveforms(unlabelled_rows, clean)
        student_batch = collate_from_waveforms(unlabelled_rows, views)
        unlabelled_windows = len(student_batch['windows'])
        with torch.no_grad():
            teacher_output = teacher(teacher_batch['windows'], teacher_batch['descriptions'])
            teacher_output = {key: value.float() for key, value in teacher_output.items()}
        model.frequency_augmentation = (warp if CONSISTENCY['augmentation']['frequency_warp']
                                        else None)
        student_output = model(student_batch['windows'], student_batch['descriptions'])
        student_output = {key: value.float() for key, value in student_output.items()}
        model.frequency_augmentation = None
        loss, parts = consistency_terms(student_output, teacher_output, student_batch['lengths'],
                                        device, CONSISTENCY['teacher_presence_threshold'],
                                        CONSISTENCY['distance_weight'])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite consistency loss')
        (consistency_weight * loss).backward()
        parts['consistency_loss'] = float(loss.detach())
        parts['mixed_views'] = sum(1 for d in decisions if 'snr_db' in d)
        parts['masked_views'] = sum(1 for d in decisions if d.get('masked_spans'))
        del loss, student_output, teacher_output, student_batch, teacher_batch, clean, views

    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, ORIGINAL['optimizer']['gradient_norm_clip'],
                                          error_if_nonfinite=True)
    optimizer.step()
    if teacher is not None:
        ema_step(TEACHER_GROUPS, CONSISTENCY['ema_decay'])
    return {'strong_loss': strong_value, 'strong_components': components,
            'weak_loss': weak_value, 'gradient_norm': float(norm.detach()),
            'strong_windows': strong_windows, 'weak_windows': weak_windows,
            'unlabelled_windows': unlabelled_windows,
            'consistency_weight': consistency_weight, **parts}


TEACHER_GROUPS = None


# ------------------------------------------------------------------------------ main

def main():
    global TEACHER_GROUPS
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--updates', type=int, default=6000)
    parser.add_argument('--consistency-weight', type=float, default=1.0)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--encoder-device', default='mps')
    parser.add_argument('--head-device', default='cpu')
    parser.add_argument('--deadline-hours', type=float, default=4.0)
    parser.add_argument('--checkpoint-dir', default=str(HERE / 'ckpt'))
    parser.add_argument('--tag', default='')
    parser.add_argument('--empty-cache-every', type=int, default=1)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import wavlm_mps_patch
    patch_report = wavlm_mps_patch.apply()
    import p2_runtime as rt
    api = rt.api_module()
    from mixed_training import WindowStream

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or out.name
    guard = Guard(torch)
    consistency_on = args.consistency_weight > 0

    log('runtime', torch=str(torch.__version__), mps=torch.backends.mps.is_available(),
        threads=args.threads, patch=patch_report, tag=tag,
        pid=os.getpid(), consistency_weight=args.consistency_weight)

    data = L.build_pools(need_training=True, need_unlabelled=consistency_on)
    log('pools', **data['census'])

    streams = {}
    for name in ('gold', 'generated', 'weak'):
        streams[name] = WindowStream(data['pools'][name], L.MIX_MAC[name], L.CAPS[name],
                                     L.SEEDS[name] + L.HELD_OUT_FOLD)
    if consistency_on:
        for name in ('unlab_domain', 'unlab_weak'):
            streams[name] = WindowStream(data['pools'][name], L.MIX_MAC[name], L.CAPS[name],
                                         L.SEEDS[name] + L.HELD_OUT_FOLD)

    guard.check('before_load', force=True)
    model, parent_metadata = rt.load_p2(args.encoder_device, args.head_device)
    teacher, groups, teacher_audit = (None, None, None)
    if consistency_on:
        teacher, groups, teacher_audit = build_teacher(model, rt, args.encoder_device,
                                                       args.head_device)
        TEACHER_GROUPS = groups
    guard.check('after_load', force=True)
    log('model', parent=parent_metadata.get('model_state_digest'), teacher=teacher_audit,
        mps_driver_gib=round(guard.snapshot['mps_driver_bytes'] / 2 ** 30, 2))

    optimizer, layout = depth_optimizer(model, api, ORIGINAL['optimizer']['weight_decay'],
                                        ORIGINAL['construction_lr']['encoder_lr'],
                                        ORIGINAL['construction_lr']['head_lr'])
    schedule = schedule_for(args.updates, ORIGINAL['construction_lr']['encoder_lr'],
                            ORIGINAL['construction_lr']['head_lr'])
    warp = api.FrequencyWarpStream(ORIGINAL['frequency_warp']['probability'],
                                   ORIGINAL['frequency_warp']['min_scale'],
                                   ORIGINAL['frequency_warp']['max_scale'],
                                   ORIGINAL['frequency_warp']['seed'] + L.HELD_OUT_FOLD)
    augmentation_generator = torch.Generator(device='cpu').manual_seed(
        CONSISTENCY['augmentation']['augmentation_seed'] + L.HELD_OUT_FOLD)
    noise_generator = torch.Generator(device='cpu').manual_seed(
        CONSISTENCY['augmentation']['noise_seed'] + L.HELD_OUT_FOLD)
    torch.manual_seed(20260917 + L.HELD_OUT_FOLD)

    contract = {'kind': 'c2_mac_pilot_v1', 'tag': tag, 'utc': utc(), 'arm': {
        'name': tag, 'updates': args.updates, 'consistency_weight': args.consistency_weight,
        'mix': L.MIX_MAC, 'caps': L.CAPS, 'seeds': L.SEEDS, 'weak_weight': ORIGINAL['weak_loss']['lambda']},
        'schedule': schedule, 'original': ORIGINAL,
        'consistency': CONSISTENCY if consistency_on else None,
        'decoder': L.DECODER, 'optimizer_groups': [g['name'] for g in optimizer.param_groups],
        'pool_sizes': {name: len(rows) for name, rows in data['pools'].items()},
        'census': data['census'], 'parent': parent_metadata, 'teacher': teacher_audit,
        'devices': {'encoder': args.encoder_device, 'head': args.head_device},
        'torch': str(torch.__version__), 'python': sys.version.split()[0],
        'layernorm_patch': patch_report}
    L.write_json(out / 'run-contract.json', contract)

    completed, trajectory = 0, []
    resume = out / 'last.pt'
    if resume.is_file():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state'], strict=True)
        model.set_device(torch.device(args.encoder_device), torch.device(args.head_device))
        if teacher is not None and saved.get('teacher_state') is not None:
            teacher.load_state_dict(saved['teacher_state'], strict=True)
            teacher.set_device(torch.device(args.encoder_device), torch.device(args.head_device))
            teacher, groups, teacher_audit = (teacher, *build_teacher_groups(teacher, model))
            TEACHER_GROUPS = groups
        optimizer.load_state_dict(saved['optimizer_state'])
        for name, stream in streams.items():
            if name in saved['streams']:
                stream.load_state_dict(saved['streams'][name])
        torch.set_rng_state(saved['rng'])
        augmentation_generator.set_state(saved['augmentation_generator_state'])
        noise_generator.set_state(saved['noise_generator_state'])
        warp.generator.set_state(saved['warp_state'])
        completed = saved['updates']
        trajectory = saved.get('trajectory', [])
        log('resumed', updates=completed)

    def save_checkpoint(update_index):
        payload = {'kind': 'c2_mac_checkpoint_v1', 'tag': tag, 'updates': update_index,
                   'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'decoder': L.DECODER, 'schedule': schedule}
        temporary = ckpt_dir / f'{tag}-{update_index:06d}-student.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(ckpt_dir / f'{tag}-{update_index:06d}-student.pt')
        if teacher is not None:
            payload['model_state'] = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
            payload['weights_from'] = 'teacher'
            temporary = ckpt_dir / f'{tag}-{update_index:06d}-teacher.pt.tmp'
            torch.save(payload, temporary)
            temporary.replace(ckpt_dir / f'{tag}-{update_index:06d}-teacher.pt')

    def save_last(update_index):
        payload = {'kind': 'c2_mac_last_v1', 'tag': tag, 'updates': update_index,
                   'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'teacher_state': ({k: v.detach().cpu() for k, v in teacher.state_dict().items()}
                                     if teacher is not None else None),
                   'optimizer_state': optimizer.state_dict(),
                   'streams': {name: stream.state_dict() for name, stream in streams.items()},
                   'rng': torch.get_rng_state(),
                   'augmentation_generator_state': augmentation_generator.get_state(),
                   'noise_generator_state': noise_generator.get_state(),
                   'warp_state': warp.generator.get_state(),
                   'trajectory': trajectory, 'schedule': schedule, 'decoder': L.DECODER,
                   'config': contract['arm']}
        temporary = out / 'last.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(out / 'last.pt')

    if completed == 0:
        save_checkpoint(0)
    losses, seconds = [], []
    stop_reason = 'complete'
    while completed < args.updates:
        if GPU_REQUEST.exists():
            stop_reason = 'gpu_request'
            break
        if time.monotonic() - STARTED > args.deadline_hours * 3600:
            stop_reason = 'deadline'
            break
        from context_crop import schedule_rates
        rates = schedule_rates(optimizer, schedule, completed)
        strong_rows = list(streams['gold'].take()) + list(streams['generated'].take())
        weak_rows = list(streams['weak'].take())
        unlabelled_rows = []
        if consistency_on:
            unlabelled_rows = list(streams['unlab_domain'].take()) + list(streams['unlab_weak'].take())
        weight = (args.consistency_weight
                  * consistency_ramp(completed, CONSISTENCY['ramp_updates']))
        step_started = time.monotonic()
        stats = one_update(model, teacher, api, strong_rows, weak_rows, unlabelled_rows,
                           optimizer, warp, weak_weight=ORIGINAL['weak_loss']['lambda'],
                           consistency_weight=weight, generator=augmentation_generator,
                           noise_pool=data['pools']['unlab_weak'] if consistency_on else None,
                           noise_generator=noise_generator)
        elapsed = time.monotonic() - step_started
        seconds.append(elapsed)
        losses.append(stats['strong_loss'])
        completed += 1
        if args.empty_cache_every and completed % args.empty_cache_every == 0 \
                and args.encoder_device == 'mps':
            torch.mps.empty_cache()
        guard.check('update')
        if completed % args.log_every == 0:
            window = seconds[-args.log_every:]
            item = {'update': completed, 'strong_loss': round(stats['strong_loss'], 5),
                    'weak_loss': (round(stats['weak_loss'], 5) if stats['weak_loss'] else None),
                    'consistency': (round(stats['consistency_loss'], 6)
                                    if stats.get('consistency_loss') is not None else None),
                    'presence_mse': (round(stats['presence_mse'], 6)
                                     if stats.get('presence_mse') is not None else None),
                    'distance_l1_reported': (round(stats['distance_l1_reported'], 4)
                                             if stats.get('distance_l1_reported') is not None else None),
                    'w_c': round(weight, 4), 'head_lr': rates['head'],
                    'gradient_norm': round(stats['gradient_norm'], 4),
                    'mean_seconds': round(sum(window) / len(window), 3),
                    'windows': [stats['strong_windows'], stats['weak_windows'],
                                stats['unlabelled_windows']],
                    'mps_driver_gib': round(guard.snapshot['mps_driver_bytes'] / 2 ** 30, 2),
                    'system_free_percent': guard.snapshot['system_wide_free_percent']}
            trajectory.append(item)
            log('update', **item)
            L.write_json(out / 'trajectory.json', {'tag': tag, 'updates': completed,
                                                   'log': trajectory})
        if completed % args.checkpoint_every == 0:
            save_checkpoint(completed)
            save_last(completed)
    save_checkpoint(completed)
    save_last(completed)
    summary = {'tag': tag, 'updates': completed, 'stop_reason': stop_reason,
               'mean_update_seconds': (sum(seconds) / len(seconds)) if seconds else None,
               'median_update_seconds': (sorted(seconds)[len(seconds) // 2] if seconds else None),
               'mean_strong_loss_last_100': (sum(losses[-100:]) / len(losses[-100:])) if losses else None,
               'mps_driver_peak_gib': guard.maximum_driver_bytes / 2 ** 30,
               'peak_rss_gib': guard.maximum_rss_bytes / 2 ** 30,
               'elapsed_hours': (time.monotonic() - STARTED) / 3600,
               'consistency_weight': args.consistency_weight, 'schedule': schedule}
    L.write_json(out / 'training-summary.json', summary)
    log('finished', **summary)


def build_teacher_groups(teacher, student):
    trainable = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    teacher_parameters = dict(teacher.named_parameters())
    groups = {}
    for name, parameter in student.named_parameters():
        if name not in trainable:
            continue
        groups.setdefault(str(parameter.device), ([], []))
        groups[str(parameter.device)][0].append(teacher_parameters[name])
        groups[str(parameter.device)][1].append(parameter)
    audit = {'averaged_tensors': sum(len(v[0]) for v in groups.values()),
             'devices': {device: len(value[0]) for device, value in groups.items()}}
    return groups, audit


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log('failed', error=repr(error), traceback=traceback.format_exc()[-3000:])
        raise
