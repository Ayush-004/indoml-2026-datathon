"""Small, explicit policies for the registered full-release comparison."""
from pathlib import Path
import hashlib
import json
import math


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load_protocol(path,digest):
    require(sha(path)==digest,'Registered protocol bytes differ')
    value=json.loads(Path(path).read_text())
    require(value['status']=='registered' and value['version']==1,'Full training requires the registered protocol')
    require(value['training']['usable_clips']==90635 and value['evaluation']['clips']==2501,'Population changed')
    require(value['initialization']['checkpoint_sha256']=='9bdb6ee9b5c1ddb3a0f11230776c04b3f91d4429a1857837971085766da33d6a','Common initialization changed')
    require(value['minimum_weak_passes']==8 and value['maximum_weak_passes']==12,'Exposure boundaries changed')
    return value


def depth_optimizer(model,api,policy):
    """Reuse the proven depth regrouping on the original native Parameters."""
    from types import SimpleNamespace
    from indoml.optimization import depth_parameter_names,regroup_depth,parameter_names
    native=getattr(model,'native',model)
    settings=SimpleNamespace(encoder_lr=1e-6,head_lr=1e-4,weight_decay=policy['weight_decay'])
    optimizer=api.make_optimizer(native,settings)
    expected={key:sorted(names) for key,names in depth_parameter_names(native).items()}
    regroup_depth(optimizer,native,expected)
    if hasattr(model,'side_projection'):
        optimizer.add_param_group({'params':list(model.side_norm.parameters())+list(model.side_projection.parameters()),
            'lr':1e-4,'name':'residual'})
    # Same Parameter identities must appear once, including the residual.
    layout=parameter_names(optimizer,model)
    return optimizer,layout


def apply_schedule(optimizer,schedule,completed_updates):
    from indoml.optimization import rates_for
    specification={'mode':'depth_cosine','origin_global_updates':0,'schedule':schedule}
    rates=rates_for(specification,completed_updates)
    if any(group['name']=='residual' for group in optimizer.param_groups):rates['residual']=rates['head']
    require({g['name'] for g in optimizer.param_groups}==set(rates),'Schedule groups differ')
    for group in optimizer.param_groups:group['lr']=rates[group['name']]
    return rates


def shared_horizon_decision(histories,horizon,policy):
    """A fixed operational plateau rule; capped runs are not called converged."""
    require(horizon in (8,10,12) and set(histories)=={'atst','beats'},'Need both completed arms at a registered boundary')
    checks={}
    for arm,history in histories.items():
        require([row['weak_pass_target'] for row in history]==[index/2 for index in range(horizon*2+1)],'Validation checkpoints are incomplete or unordered')
        trained=history[1:]
        scores=[row['metrics']['combined'] for row in trained]
        require(all(math.isfinite(score) for score in scores),'Nonfinite validation score')
        recent_peak=max(scores[-4:]);earlier_peak=max(scores[:-4])
        recent_mean=sum(scores[-2:])/2;previous_mean=sum(scores[-4:-2])/2
        best=max(scores);best_row=trained[scores.index(best)]
        result={'recent_two_pass_peak_gain':recent_peak-earlier_peak,
            'recent_one_pass_mean_gain':recent_mean-previous_mean,
            'earliest_best_trained_weak_pass':best_row['weak_pass_target'],
            'best_trained_age_passes':horizon-best_row['weak_pass_target'],
            'schedule_completed':horizon>=policy['schedule_completed_at_weak_pass']}
        result['plateau']=(result['recent_two_pass_peak_gain']<=policy['maximum_peak_gain']
            and result['recent_one_pass_mean_gain']<=policy['maximum_mean_gain']
            and result['best_trained_age_passes']>=policy['minimum_best_age_passes']
            and result['schedule_completed'])
        checks[arm]=result
    plateau=all(item['plateau'] for item in checks.values())
    return {'horizon':horizon,'checks':checks,'both_plateau':plateau,
        'decision':'stop_plateau' if plateau else 'stop_at_cap' if horizon==12 else 'extend_both',
        'next_horizon':horizon if plateau or horizon==12 else horizon+2,
        'convergence_claim':False,'budget_must_be_checked_before_any_extension':True}


def exposure(strong,weak):
    return {'strong_draws_by_tier':dict(strong.draws),'weak_draws_by_tier':dict(weak.draws),
        'strong_completed_passes':strong.completed_cycles,'weak_completed_passes':weak.completed_cycles,
        'strong_unique_clips':len(strong.seen),'weak_unique_clips':len(weak.seen),
        'strong_cursor':strong.cursor,'weak_cursor':weak.cursor}


def check_optimizer_state(optimizer,model,completed_updates,expected_layout):
    from indoml.optimization import parameter_names,check_populated_state
    require(parameter_names(optimizer,model)==expected_layout,'Optimizer parameter mapping changed')
    if completed_updates:
        check_populated_state(optimizer,model,completed_updates)
    else:
        require(not optimizer.state,'Fresh stage unexpectedly has optimizer history')


def selected_checkpoint_path(directory,selected):
    name=selected['checkpoint_file']
    require(name.startswith('best-') and name.endswith('.pt') and len(name)==72
        and all(char in '0123456789abcdef' for char in name[5:-3]),'Invalid selected checkpoint filename')
    path=Path(directory)/name
    require(path.is_file() and sha(path)==selected['checkpoint_sha256'],'Selected weights are missing or changed')
    return path


def restore_selected_artifacts(source,destination,selected,best_trained=None):
    import shutil
    source,destination=Path(source),Path(destination)
    selections={item['checkpoint_file']:item for item in (selected,best_trained) if item is not None}
    for selection in selections.values():
        best=selected_checkpoint_path(source,selection);target=destination/best.name
        if best.resolve()!=target.resolve():
            if target.exists():require(sha(target)==selection['checkpoint_sha256'],'Conflicting selected target')
            else:shutil.copyfile(best,target)
    for pattern in ('validation-*.json','validation-*-predictions.jsonl'):
        for path in source.glob(pattern):
            target=destination/path.name
            if path.resolve()!=target.resolve():
                if target.exists():require(sha(target)==sha(path),'Conflicting validation evidence')
                else:shutil.copyfile(path,target)


def prune_unselected_checkpoint_files(directory,selected,best_trained=None):
    """Call only after last.pt atomically commits both retained references."""
    keep={selected_checkpoint_path(directory,item) for item in (selected,best_trained) if item is not None}
    require(bool(keep),'A committed selection is required before pruning')
    for path in Path(directory).glob('best-*.pt'):
        if path in keep:continue
        token=path.name[5:-3]
        if len(token)==64 and all(char in '0123456789abcdef' for char in token):path.unlink()


def equal_state_tree(expected,actual,path='state'):
    """Exact restoration evidence for tensors and named optimizer/scaler state."""
    import torch
    if isinstance(expected,torch.Tensor):
        require(isinstance(actual,torch.Tensor) and expected.dtype==actual.dtype and expected.shape==actual.shape
            and torch.equal(expected.detach().cpu(),actual.detach().cpu()),'Restored tensor differs: '+path)
        return 1
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and expected.keys()==actual.keys(),'Restored mapping differs: '+path)
        return sum(equal_state_tree(expected[key],actual[key],path+'.'+str(key)) for key in expected)
    if isinstance(expected,(tuple,list)):
        require(type(expected)==type(actual) and len(expected)==len(actual),'Restored sequence differs: '+path)
        return sum(equal_state_tree(a,b,path+'.'+str(i)) for i,(a,b) in enumerate(zip(expected,actual,strict=True)))
    require(expected==actual,'Restored scalar differs: '+path)
    return 0
