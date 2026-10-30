"""Silver-strong arms on the Mac GPU.

    S1  4 Gold + 4 Silver per update, Silver weight 0.4, P2's full objective on both.
    S2  the same, but the Silver forward carries boundary weight 0 and IoU weight 0, so
        Silver teaches WHERE and WHICH CLASS and only Gold teaches endpoints. The reason is
        measured: Silver timestamps are a different distribution from Gold and from the
        organizer references (median event 0.98 s against 0.45 s, 31 % of onsets on a
        100 ms grid, which is the size of the 50-65 ms scoring tolerance).

This is the CTRL arm of `train_c2.py` (w_c = 0, no teacher, no unlabelled stream) with one
change: half the strong stream is SILVER - the 61,640 `unverified_timestamps` Vaani clips,
whose event timestamps have never been used as strong supervision at scale. The Silver part
of the strong loss is a second forward/backward multiplied by its weight, because
`collate_audio_intervals` cannot carry a per-clip loss weight.

Everything else is P2's and is imported from the C2 trainer so that it cannot drift: the
memory guard, the depth optimizer, the warmup-then-cosine schedule (encoder 1e-6 / head
1e-4, depth decay 0.95), AdamW with weight decay 1e-4, gradient clip 5, class positive
weight 5, boundary 100, IoU 1, the frequency warp, fp32 on MPS with the head on the CPU,
and the torch 2.8 MPS LayerNorm-backward patch.

Generated clips are dropped. The weak stream stays at 16 clips with weight 0.1 but is drawn
from Bronze plus the Silver rows the strong stream does not take. Every Silver/Bronze row
whose session token appears in an organizer natural VALIDATION clip id is removed first.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import s_local as S  # noqa: E402
import c2_local as L  # noqa: E402
import train_c2 as T  # noqa: E402  (Guard, depth_optimizer, schedule_for, ORIGINAL)

GPU_REQUEST = Path(os.environ.get('T1_GPU_REQUEST', str(HERE / 'GPU_REQUEST')))   # a file here pauses training
STARTED = time.monotonic()


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(message, **details):
    item = {'utc': utc(), 'elapsed_s': round(time.monotonic() - STARTED, 1), 'message': message}
    item.update(details)
    print(json.dumps(item, allow_nan=False, default=str), flush=True)


def strong_backward(model, api, rows, warp, weight, device, *, boundary_weight=None,
                    iou_weight=None):
    """P2's interval loss on one group of clips, scaled by `weight`.

    `boundary_weight` / `iou_weight` default to P2's own (100 / 1). S2 passes 0 / 0 for the
    Silver group, which leaves the presence term alone: Silver then teaches where and which
    class, and only Gold teaches endpoints.
    """
    import torch
    if boundary_weight is None:
        boundary_weight = T.ORIGINAL['strong_loss']['boundary_weight']
    if iou_weight is None:
        iou_weight = T.ORIGINAL['strong_loss']['iou_weight']
    batch = api.collate_audio_intervals(rows)
    model.frequency_augmentation = warp
    output = model(batch['windows'], batch['descriptions'])
    output = {key: value.float() for key, value in output.items()}
    weights = torch.full((len(L.CATEGORIES),),
                         T.ORIGINAL['strong_loss']['class_positive_weight'],
                         dtype=torch.float32, device=device)
    loss, _ = api.interval_loss(output, batch, weights, device, 'red',
                                boundary_weight, iou_weight, collar_weight=0.)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite strong loss')
    (weight * loss).backward()
    value = float(loss.detach())
    windows = len(batch['windows'])
    model.frequency_augmentation = None
    del loss, output, batch
    return value, windows


def one_update(model, api, gold_rows, silver_rows, weak_rows, optimizer, warp, *,
               silver_weight, weak_weight, silver_loss=None):
    import torch
    from indoml.weak_supervision import positive_category_mask, weak_positive_loss
    device = model.head_device
    model.train()
    optimizer.zero_grad(set_to_none=True)

    gold_value, gold_windows = strong_backward(model, api, gold_rows, warp, 1.0, device)
    silver_value, silver_windows = (None, 0)
    if silver_rows and silver_weight > 0:
        silver_loss = silver_loss or {}
        silver_value, silver_windows = strong_backward(
            model, api, silver_rows, warp, silver_weight, device,
            boundary_weight=silver_loss.get('boundary_weight'),
            iou_weight=silver_loss.get('iou_weight'))

    weak_value, weak_windows = None, 0
    if weak_rows and weak_weight > 0:
        batch = api.collate_clips(weak_rows)
        lengths = torch.tensor([item['frames'] for item in batch['descriptions']], dtype=torch.long)
        mask = positive_category_mask([row['positive_categories'] for row in weak_rows],
                                      device=device)
        weak_windows = len(batch['windows'])
        presence = model(batch['windows'], batch['descriptions'])['presence']
        loss, _ = weak_positive_loss(presence.float(), lengths, mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite weak loss')
        (weak_weight * loss).backward()
        weak_value = float(loss.detach())
        del presence, loss, batch, mask, lengths

    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters,
                                          T.ORIGINAL['optimizer']['gradient_norm_clip'],
                                          error_if_nonfinite=True)
    optimizer.step()
    return {'gold_loss': gold_value, 'silver_loss': silver_value, 'weak_loss': weak_value,
            'gradient_norm': float(norm.detach()),
            'windows': [gold_windows, silver_windows, weak_windows]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--tag', default='')
    parser.add_argument('--arm', default='S1', choices=('S1', 'S2'))
    parser.add_argument('--updates', type=int, default=4000)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--encoder-device', default='mps')
    parser.add_argument('--head-device', default='cpu')
    parser.add_argument('--deadline-hours', type=float, default=2.4)
    parser.add_argument('--checkpoint-dir', default=str(HERE / 'ckpt'))
    parser.add_argument('--empty-cache-every', type=int, default=1)
    args = parser.parse_args()
    tag = args.tag or args.arm

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
    guard = T.Guard(torch)
    mix = S.MIX[args.arm]
    silver_weight = S.SILVER_WEIGHT[args.arm]
    silver_loss = S.SILVER_LOSS[args.arm]
    log('runtime', torch=str(torch.__version__), mps=torch.backends.mps.is_available(),
        tag=tag, arm=args.arm, mix=mix, silver_weight=silver_weight, silver_loss=silver_loss,
        patch=patch_report)

    data = S.build_pools(arm=args.arm)
    log('pools', **data['census'])

    streams = {name: WindowStream(data['pools'][name], mix[name], S.CAPS[name],
                                  S.SEEDS[name] + S.HELD_OUT_FOLD)
               for name in ('gold', 'silver', 'weak') if mix[name] > 0}

    guard.check('before_load', force=True)
    model, parent_metadata = rt.load_p2(args.encoder_device, args.head_device)
    guard.check('after_load', force=True)
    optimizer, layout = T.depth_optimizer(model, api, T.ORIGINAL['optimizer']['weight_decay'],
                                          T.ORIGINAL['construction_lr']['encoder_lr'],
                                          T.ORIGINAL['construction_lr']['head_lr'])
    schedule = T.schedule_for(args.updates, T.ORIGINAL['construction_lr']['encoder_lr'],
                              T.ORIGINAL['construction_lr']['head_lr'])
    warp = api.FrequencyWarpStream(T.ORIGINAL['frequency_warp']['probability'],
                                   T.ORIGINAL['frequency_warp']['min_scale'],
                                   T.ORIGINAL['frequency_warp']['max_scale'],
                                   T.ORIGINAL['frequency_warp']['seed'] + S.HELD_OUT_FOLD)
    torch.manual_seed(20260917 + S.HELD_OUT_FOLD)

    contract = {'kind': 'silver_strong_v1', 'tag': tag, 'arm': args.arm, 'utc': utc(),
                'mix': mix, 'caps': S.CAPS, 'seeds': S.SEEDS, 'silver_weight': silver_weight,
                'silver_loss': silver_loss, 'weak_weight': T.ORIGINAL['weak_loss']['lambda'], 'schedule': schedule,
                'original': T.ORIGINAL, 'decoders': S.DECODERS,
                'pool_sizes': {name: len(rows) for name, rows in data['pools'].items()},
                'census': data['census'], 'parent': parent_metadata,
                'optimizer_groups': [g['name'] for g in optimizer.param_groups],
                'devices': {'encoder': args.encoder_device, 'head': args.head_device},
                'torch': str(torch.__version__), 'layernorm_patch': patch_report,
                'note': 'CTRL of the C2 trainer with the strong stream half Silver; the Silver '
                        'half is a separate forward/backward scaled by silver_weight.'}
    L.write_json(out / 'run-contract.json', contract)

    completed, trajectory = 0, []
    resume = out / 'last.pt'
    if resume.is_file():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state'], strict=True)
        model.set_device(torch.device(args.encoder_device), torch.device(args.head_device))
        optimizer.load_state_dict(saved['optimizer_state'])
        for name, stream in streams.items():
            if name in saved['streams']:
                stream.load_state_dict(saved['streams'][name])
        torch.set_rng_state(saved['rng'])
        warp.generator.set_state(saved['warp_state'])
        completed = saved['updates']
        trajectory = saved.get('trajectory', [])
        log('resumed', updates=completed)

    def save_checkpoint(update_index):
        payload = {'kind': 'silver_strong_checkpoint_v1', 'tag': tag, 'updates': update_index,
                   'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'decoders': S.DECODERS, 'schedule': schedule}
        temporary = ckpt_dir / f'{tag}-{update_index:06d}-student.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(ckpt_dir / f'{tag}-{update_index:06d}-student.pt')

    def save_last(update_index):
        payload = {'kind': 'silver_strong_last_v1', 'tag': tag, 'updates': update_index,
                   'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'optimizer_state': optimizer.state_dict(),
                   'streams': {name: stream.state_dict() for name, stream in streams.items()},
                   'rng': torch.get_rng_state(), 'warp_state': warp.generator.get_state(),
                   'trajectory': trajectory, 'schedule': schedule, 'config': contract['mix'],
                   'arm': args.arm, 'silver_weight': silver_weight}
        temporary = out / 'last.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(out / 'last.pt')

    from context_crop import schedule_rates
    seconds, stop_reason = [], 'complete'
    while completed < args.updates:
        if GPU_REQUEST.exists():
            stop_reason = 'gpu_request'
            break
        if time.monotonic() - STARTED > args.deadline_hours * 3600:
            stop_reason = 'deadline'
            break
        rates = schedule_rates(optimizer, schedule, completed)
        gold_rows = list(streams['gold'].take()) if 'gold' in streams else []
        silver_rows = list(streams['silver'].take()) if 'silver' in streams else []
        weak_rows = list(streams['weak'].take()) if 'weak' in streams else []
        step_started = time.monotonic()
        stats = one_update(model, api, gold_rows, silver_rows, weak_rows, optimizer, warp,
                           silver_weight=silver_weight, silver_loss=silver_loss,
                           weak_weight=T.ORIGINAL['weak_loss']['lambda'])
        seconds.append(time.monotonic() - step_started)
        completed += 1
        if args.empty_cache_every and completed % args.empty_cache_every == 0 \
                and args.encoder_device == 'mps':
            torch.mps.empty_cache()
        guard.check('update')
        if completed % args.log_every == 0:
            window = seconds[-args.log_every:]
            item = {'update': completed, 'gold_loss': round(stats['gold_loss'], 5),
                    'silver_loss': (round(stats['silver_loss'], 5)
                                    if stats['silver_loss'] is not None else None),
                    'weak_loss': (round(stats['weak_loss'], 5) if stats['weak_loss'] else None),
                    'head_lr': rates['head'], 'gradient_norm': round(stats['gradient_norm'], 4),
                    'mean_seconds': round(sum(window) / len(window), 3),
                    'windows': stats['windows'],
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
    summary = {'tag': tag, 'arm': args.arm, 'updates': completed, 'stop_reason': stop_reason,
               'mean_update_seconds': (sum(seconds) / len(seconds)) if seconds else None,
               'median_update_seconds': (sorted(seconds)[len(seconds) // 2] if seconds else None),
               'mps_driver_peak_gib': guard.maximum_driver_bytes / 2 ** 30,
               'peak_rss_gib': guard.maximum_rss_bytes / 2 ** 30,
               'elapsed_hours': (time.monotonic() - STARTED) / 3600,
               'silver_weight': silver_weight, 'silver_loss': silver_loss, 'mix': mix,
               'schedule': schedule}
    L.write_json(out / 'training-summary.json', summary)
    log('finished', **summary)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        log('failed', error=repr(error), traceback=traceback.format_exc()[-3000:])
        raise
