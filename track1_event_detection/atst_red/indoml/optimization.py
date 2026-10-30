"""Bound full-encoder continuation, with update-indexed learning rates.

No model, data, loss, RNG or evaluation changes live here. AdamW state remains
keyed by the exact original Parameter objects when its groups are subdivided.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


TRAINING_KEYS = ('manifest_sha256', 'initialization', 'native_provenance',
                 'source_sha256', 'audio_inventory_sha256',
                 'initial_feature_cache_id', 'policy')
ORIGIN_SHA256 = '38e064407ff6241a0942581fe333ca69c7396f786cacbe79c28ec127d740198a'
BEST_SHA256 = '0ba14f005a2f9932bcf4961802c416694f587361c60de18b5b6d0595cbcf0769'
PRODUCTION_SCHEDULE = dict(warmup_updates=3081, hold_updates=3081,
                           decay_updates=18486, floor_fraction=.1,
                           encoder_peak_top=1e-5, head_peak=1e-4, depth_decay=.95)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_contract(path, expected_sha256):
    require(path and expected_sha256, 'Continuation requires an explicitly bound contract')
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha256,
            'Continuation contract SHA-256 differs')
    value = json.loads(payload)
    require(value.get('version') == 1 and value.get('scope') in ('production', 'train_only_qa'),
            'Invalid continuation contract version/scope')
    require(set(value['legacy_training']) == set(TRAINING_KEYS), 'Incomplete legacy training binding')
    origin = value['origin']
    require(type(origin['epoch']) is int and origin['epoch'] >= 1
            and type(origin['global_updates']) is int and origin['global_updates'] > 0,
            'Invalid completed continuation origin')
    schedule = value['schedule']
    require(set(schedule) == set(PRODUCTION_SCHEDULE), 'Unexpected schedule fields')
    require(all(type(schedule[k]) is int and schedule[k] >= 2
                for k in ('warmup_updates', 'hold_updates', 'decay_updates')),
            'Schedule phases must contain at least two updates')
    require(all(type(schedule[k]) in (int, float) and math.isfinite(schedule[k]) and schedule[k] > 0
                for k in ('floor_fraction', 'encoder_peak_top', 'head_peak', 'depth_decay'))
            and schedule['floor_fraction'] <= 1 and schedule['depth_decay'] <= 1,
            'Invalid schedule rates')
    require(set(value['parameter_name_groups']) == {str(k) for k in range(12)} | {'head'},
            'Expected all twelve encoder depths and the head')
    if value['scope'] == 'production':
        require(origin == dict(epoch=6, global_updates=6162, checkpoint_sha256=ORIGIN_SHA256)
                and value['best']['checkpoint_sha256'] == BEST_SHA256
                and value['best']['epoch'] == 5 and schedule == PRODUCTION_SCHEDULE,
                'Production continuation differs from the approved origin or schedule')
        require(value['legacy_training']['policy']['test_only_clips'] is None,
                'Production continuation cannot use a QA parent')
    else:
        require(value['legacy_training']['policy'].get('test_only_clips') is not None,
                'QA continuation requires an explicitly test-only origin')
    return value


def check_source_transition(old, current):
    require(set(current) == set(old) | {'optimization.py'}
            and all(current[k] == v for k, v in old.items() if k != 'atst_intervals.py'),
            'Continuation changed a forward, data, loss or scorer source')


def check_legacy_training(saved_training, metadata, contract):
    expected = contract['legacy_training']
    for key in TRAINING_KEYS:
        require(saved_training.get(key) == expected[key], f'Legacy continuation differs: {key}')
        if key not in ('source_sha256', 'policy'):
            require(metadata.get(key) == expected[key], f'Continuation changed: {key}')
    policy = {k: v for k, v in metadata['policy'].items() if k != 'optimization'}
    require(policy == expected['policy'], 'Continuation changed the original Gold policy')
    check_source_transition(expected['source_sha256'], metadata['source_sha256'])


def legacy_checkpoint_allowed(saved, metadata, checkpoint_sha256, *, historical=False):
    """Only the bound last checkpoint may start a trajectory; best is read-only."""
    policy = metadata.get('policy')
    spec = policy.get('optimization') if isinstance(policy, dict) else None
    if not spec:
        return False
    saved_policy = saved['training'].get('policy')
    require(isinstance(saved_policy, dict), 'Resume policy differs: policy')
    if saved_policy.get('optimization') is not None:
        return False
    contract = metadata['continuation_contract']
    expected = contract['best'] if historical else contract['origin']
    require(saved['checkpoint_kind'] == 'resume' and saved.get('validation_complete') is True
            and saved['epoch'] == expected['epoch']
            and checkpoint_sha256 == expected['checkpoint_sha256'],
            'Unbound legacy resume checkpoint; historical best is not a continuation origin')
    check_legacy_training(saved['training'], metadata, contract)
    require(saved.get('global_updates') == saved['epoch'] * metadata['steps_per_epoch'],
            'Legacy completed update count differs')
    require('optimization_state' not in saved, 'Legacy origin unexpectedly carries a schedule')
    return True


def prediction_source_binding(checkpoint, checkpoint_sha256, current_sources, args):
    old = checkpoint['training']['source_sha256']
    if old == current_sources:
        return None
    contract = load_contract(getattr(args, 'continuation_contract', None),
                             getattr(args, 'continuation_contract_sha256', None))
    require(checkpoint['checkpoint_kind'] == 'inference'
            and contract['inference_checkpoints'].get(str(checkpoint['epoch'])) == checkpoint_sha256,
            'Inference checkpoint is not bound in the inherited history')
    require(all(checkpoint['training'].get(k) == contract['legacy_training'][k] for k in TRAINING_KEYS),
            'Inherited inference provenance differs')
    check_source_transition(old, current_sources)
    return dict(contract_sha256=args.continuation_contract_sha256,
                inherited_epoch=checkpoint['epoch'], checkpoint_source_sha256=old)


def depth_parameter_names(model):
    groups = {str(k): [] for k in range(12)} | {'head': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('head.'):
            key = 'head'
        elif name == 'encoder.pos_embed' or name.startswith('encoder.patch_embed.'):
            key = '0'
        elif name.startswith('encoder.norm_frame.'):
            key = '11'
        elif name.startswith('encoder.blocks.'):
            key = name.split('.')[2]
            require(key in groups and key != 'head', 'Unexpected encoder block')
        else:
            raise ValueError(f'Unmapped trainable parameter: {name}')
        groups[key].append(name)
    require(all(groups.values()), 'All encoder depths and the head must be trainable')
    return groups


def parameter_names(optimizer, model):
    names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    flat = [p for group in optimizer.param_groups for p in group['params']]
    require(len(flat) == len(names) and len({id(p) for p in flat}) == len(names)
            and {id(p) for p in flat} == set(names), 'Dropped, duplicated or foreign optimizer parameter')
    return [{'name': g['name'], 'parameters': [names[id(p)] for p in g['params']]}
            for g in optimizer.param_groups]


def regroup_depth(optimizer, model, expected_names):
    """Subdivide the loaded optimizer in place. No state tensor is copied/reset."""
    layout = parameter_names(optimizer, model)
    require([g['name'] for g in layout] == ['encoder', 'head'], 'Expected original two AdamW groups')
    groups = depth_parameter_names(model)
    require({k: sorted(v) for k, v in groups.items()} == expected_names,
            'Encoder depth parameter names differ from the approved grouping')
    names = dict(model.named_parameters())
    original = {g['name']: g for g in optimizer.param_groups}
    old_states = {p: state for p, state in optimizer.state.items()}
    replacement = []
    for key, members in groups.items():
        parent = original['head' if key == 'head' else 'encoder']
        replacement.append({**parent, 'name': key, 'params': [names[name] for name in members]})
    optimizer.param_groups = replacement
    parameter_names(optimizer, model)
    require(set(old_states) == set(optimizer.state)
            and all(optimizer.state[p] is value for p, value in old_states.items()),
            'Regrouping changed per-parameter AdamW state')
    return parameter_names(optimizer, model)


def check_populated_state(optimizer, model, completed_updates):
    import torch
    layout = parameter_names(optimizer, model)
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    require(set(optimizer.state) == set(parameters), 'Continuation requires every populated AdamW state')
    for parameter in parameters:
        state = optimizer.state[parameter]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                and isinstance(state['step'], torch.Tensor) and state['step'].numel() == 1
                and float(state['step']) == completed_updates,
                'AdamW step or moment fields differ')
        require(all(state[key].shape == parameter.shape and state[key].dtype == parameter.dtype
                    and torch.isfinite(state[key]).all().item() for key in ('exp_avg', 'exp_avg_sq')),
                'Nonfinite or incompatible AdamW moments')
    return layout


def check_optimizer_options(optimizer, contract):
    for group in optimizer.param_groups:
        key = 'head' if group['name'] == 'head' else 'encoder'
        actual = {k: v for k, v in group.items() if k not in ('params', 'name', 'lr')}
        require(json.loads(json.dumps(actual)) == contract['adamw_non_lr_options'][key],
                'AdamW non-LR settings differ from the original optimizer')


def continuation_config(args, model, steps_per_epoch):
    mode = getattr(args, 'optimization_policy', None)
    if mode is None:
        require(not getattr(args, 'continuation_contract', None)
                and not getattr(args, 'continuation_contract_sha256', None),
                'A continuation contract requires its optimization policy')
        return None, None
    require(mode in ('constant', 'depth_cosine') and args.resume and args.train_blocks == 12
            and args.encoder_lr == 1e-6 and args.head_lr == 1e-4 and not args.aux_weight,
            'Optimization continuation requires the original full-model Gold resume')
    contract = load_contract(args.continuation_contract, args.continuation_contract_sha256)
    require((args.test_only_clips is None) == (contract['scope'] == 'production'),
            'QA and production continuation scopes differ')
    require(contract['origin']['global_updates'] == contract['origin']['epoch'] * steps_per_epoch,
            'Continuation origin and actual steps per epoch differ')
    require({k: sorted(v) for k, v in depth_parameter_names(model).items()} == contract['parameter_name_groups'],
            'Full-model parameter grouping differs from the bound contract')
    spec = dict(version=1, mode=mode, contract_sha256=args.continuation_contract_sha256,
                origin_epoch=contract['origin']['epoch'], origin_global_updates=contract['origin']['global_updates'],
                schedule=contract['schedule'],
                position='Completed global Gold updates before the next optimizer step; evaluation never advances it',
                diagnostics='Raw global norms on every new update; depth/head gradient and actual update sizes on first and last update of each new epoch')
    return contract, spec


def rates_for(spec, completed_global_updates):
    j = completed_global_updates - spec['origin_global_updates']
    require(type(completed_global_updates) is int and j >= 0, 'Schedule precedes the continuation origin')
    if spec['mode'] == 'constant':
        return dict(encoder=1e-6, head=1e-4)
    require(spec['mode'] == 'depth_cosine', 'Unknown optimization policy')
    s = spec['schedule']
    warm, hold, decay = (s[k] for k in ('warmup_updates', 'hold_updates', 'decay_updates'))
    peaks = {str(k): s['encoder_peak_top'] * s['depth_decay'] ** (11-k) for k in range(12)}
    peaks['head'] = s['head_peak']
    if j < warm:
        return {k: (1e-4 if k == 'head' else 1e-6)
                + (peak - (1e-4 if k == 'head' else 1e-6)) * j / (warm-1)
                for k, peak in peaks.items()}
    if j < warm + hold:
        return peaks
    fraction = s['floor_fraction']
    if j < warm + hold + decay:
        fraction += (1-s['floor_fraction']) / 2 * (1 + math.cos(math.pi * (j-warm-hold) / (decay-1)))
    return {k: peak * fraction for k, peak in peaks.items()}


def schedule_phase(spec, completed_global_updates):
    if spec['mode'] == 'constant':
        return 'constant'
    j = completed_global_updates - spec['origin_global_updates']
    s = spec['schedule']
    for phase, length in (('warmup', s['warmup_updates']), ('hold', s['hold_updates']), ('decay', s['decay_updates'])):
        if j < length:
            return phase
        j -= length
    return 'tail'


class ContinuationSchedule:
    def __init__(self, optimizer, model, spec, completed=None):
        self.optimizer, self.model, self.spec = optimizer, model, spec
        self.completed = spec['origin_global_updates'] if completed is None else completed
        self.layout = parameter_names(optimizer, model)
        require([g['name'] for g in self.layout] == list(rates_for(spec, self.completed)),
                'Optimizer and learning-rate groups differ')
        self.check_rates()

    def state_dict(self):
        return dict(version=1, policy=self.spec, completed_global_updates=self.completed,
                    completed_new_updates=self.completed - self.spec['origin_global_updates'],
                    optimizer_parameter_names=self.layout,
                    last_applied_rates=rates_for(self.spec, max(self.spec['origin_global_updates'], self.completed-1)),
                    next_rates=rates_for(self.spec, self.completed),
                    next_phase=schedule_phase(self.spec, self.completed))

    def check_rates(self):
        require({g['name']: g['lr'] for g in self.optimizer.param_groups} == self.state_dict()['last_applied_rates'],
                'Optimizer rates differ from the saved completed-update schedule')

    def restore(self, saved, completed):
        previous = self.completed
        self.completed = completed
        try:
            require(saved == self.state_dict(), 'Saved schedule state, group names or update position differs')
            self.check_rates()
        except BaseException:
            self.completed = previous
            raise

    def before_update(self, next_global_update):
        require(next_global_update == self.completed + 1, 'Optimizer update and schedule position differ')
        self.check_rates()
        rates = rates_for(self.spec, self.completed)
        for group in self.optimizer.param_groups:
            group['lr'] = rates[group['name']]

    def after_update(self):
        self.completed += 1
        self.check_rates()


class UpdateObservation:
    """First/last-update diagnostics; CPU copies add no RNG draws or gradients."""
    def __init__(self, model):
        self.groups = depth_parameter_names(model)
        self.parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.before = {name: p.detach().cpu().clone() for name, p in self.parameters.items()}
        self.raw = self.gradient_norms()
        self.clipped = None

    def gradient_norms(self):
        values = {}
        for key, names in self.groups.items():
            total = 0.
            for name in names:
                gradient = self.parameters[name].grad
                require(gradient is not None, f'Missing intended gradient: {name}')
                value = gradient.detach().cpu().double()
                total += float(value.square().sum())
            require(math.isfinite(total), 'Nonfinite diagnostic gradient')
            values[key] = math.sqrt(total)
        return values

    def after_clip(self):
        self.clipped = self.gradient_norms()

    def finish(self):
        result = {}
        for key, names in self.groups.items():
            weight, update, changed = 0., 0., 0
            for name in names:
                before = self.before[name].double()
                delta = self.parameters[name].detach().cpu().double() - before
                weight += float(before.square().sum())
                update += float(delta.square().sum())
                changed += int((delta != 0).sum())
            require(math.isfinite(update) and math.isfinite(weight), 'Nonfinite parameter update')
            result[key] = dict(raw_gradient_norm=self.raw[key], clipped_gradient_norm=self.clipped[key],
                               weight_norm=math.sqrt(weight), update_norm=math.sqrt(update),
                               update_to_weight=math.sqrt(update / weight) if weight else None,
                               changed_elements=changed, parameter_tensors=len(names))
        return result


def shared_horizon_decision(histories, horizon, arm_order=('constant', 'scheduled')):
    require(horizon in (30, 36, 42) and set(histories) == set(arm_order),
            'Both complete arms and an approved shared horizon are required')
    results = {}
    for arm in arm_order:
        rows = histories[arm]
        require([r['epoch'] for r in rows] == list(range(1, horizon+1)), 'Shared horizon history is incomplete')
        scores = [row['validation']['combined'] for row in rows]
        require(all(type(s) in (int, float) and math.isfinite(s) for s in scores), 'Nonfinite INNER history')
        best_epoch = max(range(horizon), key=lambda i: (scores[i], -i)) + 1
        gain = max(scores[-6:]) - max(scores[:-6])
        trend = math.fsum(scores[-3:]) / 3 - math.fsum(scores[-6:-3]) / 3
        conditions = dict(last_six_gain=gain <= .002, last_three_trend=trend <= .002,
                          trained_best_outside_last_three=best_epoch <= horizon-3)
        results[arm] = dict(earliest_best_trained_epoch=best_epoch, last_six_best_gain=gain,
                           last_three_mean_trend=trend, conditions=conditions, plateau=all(conditions.values()))
    plateau = all(row['plateau'] for row in results.values())
    stop = plateau or horizon == 42
    return dict(horizon=horizon, arms=results, stop=stop, next_horizon=None if stop else horizon+6,
                reason='both_plateau' if plateau else ('budget_cap_unresolved' if stop else 'extend_both'),
                optimization_adequacy='operational_plateau' if plateau else 'unresolved')
