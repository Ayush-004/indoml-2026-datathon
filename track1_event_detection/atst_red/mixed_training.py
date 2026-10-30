"""Whole-clip strong/weak updates for the full-release CUDA path."""
from collections import Counter
from contextlib import contextmanager
import time
import torch


class WindowStream:
    """A saved permutation/cursor; packing never drops or truncates a clip."""
    def __init__(self, rows, max_clips, max_windows, seed, worst_first=False):
        assert rows and max_clips > 0 and max_windows >= max(r['native_windows'] for r in rows)
        assert len({r['clip_id'] for r in rows}) == len(rows)
        self.rows, self.max_clips, self.max_windows = rows, max_clips, max_windows
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.order = sorted(range(len(rows)), key=lambda i: (-rows[i]['native_windows'], -rows[i]['duration'], rows[i]['clip_id'])) if worst_first else torch.randperm(len(rows), generator=self.generator).tolist()
        self.cursor, self.completed_cycles = 0, 0
        self.draws, self.seen = Counter(), set()

    def take(self):
        if self.cursor == len(self.order):
            self.order = torch.randperm(len(self.rows), generator=self.generator).tolist()
            self.cursor = 0
        result, windows = [], 0
        while self.cursor < len(self.order) and len(result) < self.max_clips:
            row = self.rows[self.order[self.cursor]]
            if result and windows + row['native_windows'] > self.max_windows:
                break
            assert windows + row['native_windows'] <= self.max_windows
            self.cursor += 1
            result.append(row)
            windows += row['native_windows']
            self.draws[row['quality']] += 1
            self.seen.add(row['clip_id'])
        assert result
        if self.cursor == len(self.order):
            self.completed_cycles += 1
        return result

    def state_dict(self):
        return dict(order=self.order, cursor=self.cursor, completed_cycles=self.completed_cycles,
            generator_state=self.generator.get_state(), draws=dict(self.draws), seen=sorted(self.seen),
            max_clips=self.max_clips, max_windows=self.max_windows, pool_ids=[r['clip_id'] for r in self.rows])

    def load_state_dict(self, saved):
        assert saved['pool_ids'] == [r['clip_id'] for r in self.rows]
        assert (saved['max_clips'], saved['max_windows']) == (self.max_clips, self.max_windows)
        assert sorted(saved['order']) == list(range(len(self.rows)))
        assert 0 <= saved['cursor'] <= len(self.rows)
        self.order, self.cursor = saved['order'], saved['cursor']
        self.completed_cycles = saved['completed_cycles']
        self.generator.set_state(saved['generator_state'])
        self.draws, self.seen = Counter(saved['draws']), set(saved['seen'])


def enforce_fp32_head(model):
    """The encoder may autocast; recurrent/head probability math stays FP32."""
    assert not getattr(model, '_full_release_fp32_head', False)
    original = model.head.forward
    def full_precision(features, lengths):
        with torch.autocast(device_type='cuda', enabled=False):
            result = original(features.float(), lengths)
            assert all(value.dtype == torch.float32 for value in result.values())
            return result
    model.head.forward = full_precision
    model._full_release_fp32_head = True


class PhaseMeter:
    """Detailed timing only on requested steps; regular updates avoid extra syncs."""
    def __init__(self):
        self.enabled = False
        self.seconds, self.calls = Counter(), Counter()

    def reset(self, enabled):
        self.enabled = enabled
        self.seconds.clear(); self.calls.clear()

    @contextmanager
    def phase(self, name, cuda=False):
        if not self.enabled:
            yield
            return
        if cuda: torch.cuda.synchronize()
        before = time.perf_counter()
        yield
        if cuda: torch.cuda.synchronize()
        self.seconds[name] += time.perf_counter() - before
        self.calls[name] += 1

    def wrap(self, owner, attribute, name, cuda=False):
        original = getattr(owner, attribute)
        def measured(*args, **kwargs):
            with self.phase(name, cuda):
                return original(*args, **kwargs)
        setattr(owner, attribute, measured)


def mixed_update(model, gold_rows, weak_rows, optimizer, scaler, warp, api, *, amp, weak_weight, meter):
    """Release the strong graph before weak forward; unscale/clip/step once."""
    from indoml.weak_supervision import positive_category_mask, weak_positive_loss
    assert all(r['supervision'] == 'strong_gold' and r['events'] for r in gold_rows)
    assert all(r['supervision'] == 'weak_clip' and r['events'] is None and r['positive_categories'] for r in weak_rows)
    assert model.device.type == model.head_device.type == 'cuda'
    assert getattr(model, '_full_release_fp32_head', False), 'Enforce FP32 head math before mixed updates'
    device = model.head_device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with meter.phase('load_and_collate_strong'):
        batch = api.collate_audio_intervals(gold_rows)
    strong_windows = len(batch['windows'])
    model.frequency_augmentation = warp
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        output = model(batch['windows'], batch['descriptions'])
    with meter.phase('strong_loss_and_backward', cuda=True):
        # RED uses probabilities/logs and boundary ratios; keep these in FP32.
        output = {key: value.float() for key, value in output.items()}
        weights = torch.full((7,), 5., dtype=torch.float32, device=device)
        strong_loss, components = api.interval_loss(output, batch, weights, device, 'red', 100., 1., collar_weight=0.)
        assert torch.isfinite(strong_loss).item()
        scaler.scale(strong_loss).backward()
    strong_value = float(strong_loss.detach())
    del strong_loss, output, batch, components
    model.frequency_augmentation = None
    with meter.phase('load_and_collate_weak'):
        batch = api.collate_clips(weak_rows)
        lengths = torch.tensor([item['frames'] for item in batch['descriptions']], dtype=torch.long)
        mask = positive_category_mask([r['positive_categories'] for r in weak_rows], device=device)
    weak_windows = len(batch['windows'])
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        presence = model(batch['windows'], batch['descriptions'])['presence']
    with meter.phase('weak_loss_and_backward', cuda=True):
        weak_loss, stats = weak_positive_loss(presence.float(), lengths, mask)
        assert torch.isfinite(weak_loss).item()
        scaler.scale(weak_weight * weak_loss).backward()
    weak_value = float(weak_loss.detach())
    del presence, weak_loss, batch, mask, lengths
    with meter.phase('unscale_clip_optimizer', cuda=True):
        before_scale = scaler.get_scale()
        scaler.unscale_(optimizer)
        parameters = [p for p in model.parameters() if p.requires_grad]
        norms = [torch.linalg.vector_norm(p.grad.detach(), 2) for p in parameters if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack(norms), 2)
        finite = bool(torch.isfinite(norm).item())
        assert amp or finite, 'FP32 nonfinite gradients'
        if finite:
            torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        else:
            assert any(not torch.isfinite(p.grad).all().item() for p in parameters if p.grad is not None), 'Aggregate norm overflowed despite finite gradients; stop before changing weights'
        # GradScaler collected found-inf during unscale. Nonfinite gradients
        # are left untouched and the resulting skipped update is reported.
        scaler.step(optimizer)
        scaler.update()
        after_scale = scaler.get_scale()
        skipped = after_scale < before_scale
        assert finite or skipped, 'Nonfinite gradients were not safely skipped'
    model.frequency_augmentation = warp
    return dict(strong_loss=strong_value, weak_loss=weak_value, weak_weight=weak_weight,
        raw_gradient_norm=float(norm.detach()) if finite else None, skipped_update=skipped,
        scaler_before=before_scale, scaler_after=after_scale,
        strong_clips=len(gold_rows), weak_clips=len(weak_rows), strong_windows=strong_windows, weak_windows=weak_windows,
        weak_supervised_clips=stats['supervised_clips'], weak_unknown_labels_masked=True,
        strong_clip_ids=[r['clip_id'] for r in gold_rows], weak_clip_ids=[r['clip_id'] for r in weak_rows],
        phase_seconds=dict(meter.seconds), phase_calls=dict(meter.calls))


def rng_state():
    return {'torch_cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(saved):
    assert set(saved) == {'torch_cpu', 'cuda'}
    torch.set_rng_state(saved['torch_cpu'])
    torch.cuda.set_rng_state_all(saved['cuda'])
