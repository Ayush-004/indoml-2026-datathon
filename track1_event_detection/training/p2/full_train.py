"""One registered arm, with bounded segments and completed-update recovery."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
import traceback

from production_core import (require,sha,write_json,load_protocol,depth_optimizer,
    apply_schedule,exposure,check_optimizer_state,restore_selected_artifacts,
    prune_unselected_checkpoint_files,equal_state_tree)
from input_paths import InputLayout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--arm',choices=('atst','beats'),required=True)
    parser.add_argument('--native-bundle',type=Path,required=True)
    parser.add_argument('--mixed-bundle',type=Path,required=True)
    parser.add_argument('--beats-bundle',type=Path,required=True)
    parser.add_argument('--cache-manifest',type=Path)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-weak-passes',type=int,default=8)
    parser.add_argument('--max-seconds',type=int,default=37800)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after-updates',type=int)
    parser.add_argument('--extension-decision',type=Path)
    parser.add_argument('--extension-decision-sha256')
    args=parser.parse_args()
    protocol=load_protocol(args.protocol,args.protocol_sha256)
    require(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1'),'Use the owned CUDA worker')
    require(1<=args.target_weak_passes<=12 and 300<=args.max_seconds<=37800,'Invalid bounded segment')
    if args.stop_after_updates is not None:
        require(args.stop_after_updates==protocol['execution']['startup_checkpoint_updates'],'Only the registered completed-update startup boundary is allowed')
    if args.target_weak_passes>8:
        require(args.extension_decision and sha(args.extension_decision)==args.extension_decision_sha256,'An extension needs the registered joint decision')
        decision=json.loads(args.extension_decision.read_text())
        require(decision['decision']=='extend_both' and decision['next_horizon']==args.target_weak_passes,'Invalid extension decision')
        require(decision['protocol_sha256']==args.protocol_sha256,'Extension protocol differs')
    for name,digest in protocol['production_source_sha256'].items():
        require(sha(Path(__file__).parent/name)==digest,'Production source changed: '+name)
    native,mixed,beats,inputs=(p.resolve() for p in (args.native_bundle,args.mixed_bundle,args.beats_bundle,args.input_root))
    for directory,filename,key in ((native,'asset-manifest.json','native_manifest_sha256'),(mixed,'fixture-manifest.json','mixed_manifest_sha256'),
                                  (beats,'beats-manifest.json','beats_manifest_sha256'),(inputs,'full-input-contract.json','full_input_contract_sha256')):
        require(sha(directory/filename)==protocol['input_bindings'][key],'Input binding changed: '+key)
    full_inputs=json.loads((inputs/'full-input-contract.json').read_text())
    for filename,key in (('training.jsonl','training_manifest_sha256'),('organizer-inference.jsonl','organizer_inference_sha256'),('organizer-references.jsonl','organizer_references_sha256')):
        require(sha(inputs/filename)==full_inputs[key],'Input manifest changed: '+filename)
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'.training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(args.resume is not None or not (args.output/'last.pt').exists(),'Do not restart a trajectory without its checkpoint')
    sys.path[:0]=[str(native),str(native/'source'),str(mixed),str(beats)]
    import torch
    from cuda_adapter import runtime_fingerprint,require_cuda_host,verify_assets,load_model,cpu_tree
    from indoml import atst_intervals as api
    from indoml.train import save_checkpoint
    from indoml.optimization import parameter_names
    from mixed_training import WindowStream,PhaseMeter,mixed_update,rng_state,restore_rng,enforce_fp32_head
    from fusion import TrainableComplementaryATST,saved_model_state,restore_model_state
    torch.set_num_threads(protocol['execution']['cpu_threads_per_arm'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    runtime=runtime_fingerprint(torch,__import__('torchaudio'));require_cuda_host(runtime)
    require(runtime==protocol['runtime'],'Runtime differs from the measured and registered image')
    assets=json.loads((native/'asset-manifest.json').read_text());verify_assets(native,assets,content=True)
    training=[json.loads(line) for line in (inputs/'training.jsonl').open()]
    require(len(training)==90635,'Training population changed')
    inference=[json.loads(line) for line in (inputs/'organizer-inference.jsonl').open()]
    require(len(inference)==2501 and all(set(row)=={'clip_id','audio_path','duration'} for row in inference),'Predictor manifest must be unlabeled and complete')
    layout=InputLayout(inputs,full_inputs,training,inference)
    for index,row in enumerate(training):
        row['audio_path']=str(layout.training_path(index,row))
        require(inputs in Path(row['audio_path']).parents,'Audio path escaped input root')
    gold=[row for row in training if row['supervision']=='strong_gold']
    weak=[row for row in training if row['supervision']=='weak_clip']
    require(len(gold)==11095 and len(weak)==79540,'Strong/weak population changed')
    references=[json.loads(line) for line in (inputs/'organizer-references.jsonl').open()]
    reference_by_id={row['clip_id']:row for row in references}
    require(set(reference_by_id)=={row['clip_id'] for row in inference},'Organizer reference coverage changed')
    prediction_rows=[]
    for row in inference:
        audio=layout.organizer_path(row)
        require(inputs in audio.parents,'Organizer audio path escaped input root')
        prediction_rows.append({**row,'audio_path':str(audio),'num_frames':(round(row['duration']*16000)+639)//640,
            'frame_hop':.04,'frame_offset':.02,'feature_dim':768})
    class PredictClips:
        def __len__(self):return len(prediction_rows)
        def __getitem__(self,index):return prediction_rows[index]
    model,initialization=load_model(api,torch,native,assets)
    model.set_device(torch.device('cuda:0'),torch.device('cuda:0'));enforce_fp32_head(model)
    if args.arm=='beats':
        from beats_cache import CachedBEATs
        require(protocol['models']['beats']['feature_mode']=='cached_coarse_fp32','Registered BEATs feature mode differs')
        require(args.cache_manifest is not None,'The BEATs arm requires the complete verified cache')
        side=CachedBEATs(args.cache_manifest,protocol['models']['beats']['cache_manifest_sha256'])
        require(side.manifest['records']==93136 and side.manifest['windows']==101548
            and side.manifest['nodes']==3860480,'Full cache coverage changed')
        require(side.manifest['provenance']==protocol['models']['beats']['cache_provenance'],'Cache scientific provenance changed')
        model=TrainableComplementaryATST(model,side)
    input_layout_evidence=layout.evidence()
    optimizer,layout=depth_optimizer(model,api,protocol['optimizer'])
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=1024.,growth_interval=2000)
    warp=api.FrequencyWarpStream(.5,.6,1.,50)
    sampler=protocol['sampling']
    strong=WindowStream(gold,sampler['strong_max_clips'],sampler['max_windows_per_forward'],49)
    auxiliary=WindowStream(weak,sampler['weak_max_clips'],sampler['max_windows_per_forward'],51)
    meter=PhaseMeter()
    torch.manual_seed(52);torch.cuda.manual_seed_all(52)
    completed=attempted=0;history=[];selected=None;best_trained=None;next_validation_index=0;pending_validation=None
    train_seconds=validation_seconds=0.;skipped_attempts=0
    resumed_from=None
    if args.resume:
        source=args.resume.resolve();saved=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
        require(saved['kind']=='full_vaani_resume_v1' and saved['protocol_sha256']==args.protocol_sha256 and saved['arm']==args.arm,'Wrong full-data trajectory')
        require(saved['runtime']==runtime and saved['optimizer_layout']==layout,'Resume runtime or Parameter mapping changed')
        restore_model_state(model,saved['model_state']);optimizer.load_state_dict(saved['optimizer_state']);scaler.load_state_dict(saved['scaler_state'])
        warp.load_state_dict(saved['warp_state']);strong.load_state_dict(saved['strong_stream']);auxiliary.load_state_dict(saved['weak_stream'])
        completed,attempted=saved['completed_updates'],saved['attempted_updates']
        history,selected,best_trained=saved['history'],saved['selected'],saved['best_trained']
        next_validation_index,pending_validation=saved['next_validation_index'],saved['pending_validation']
        train_seconds,validation_seconds=saved['train_seconds'],saved['validation_seconds'];skipped_attempts=saved['skipped_attempts']
        # Completed validation journals can be reused if an interrupted segment
        # stopped after calibration but before publishing the updated checkpoint.
        restore_selected_artifacts(source.parent,args.output,selected,best_trained)
        restore_rng(saved['rng_state'])
        resumed_from={'path':str(source),'sha256':sha(source),'completed_updates':completed,'weak_completed_passes':auxiliary.completed_cycles}
        require(api.state_digest(saved_model_state(model))==saved['model_state_digest'],'Restored model values changed')
        check_optimizer_state(optimizer,model,completed,layout)
        verified_tensors=equal_state_tree(saved['optimizer_state'],optimizer.state_dict(),'optimizer')
        verified_tensors+=equal_state_tree(saved['scaler_state'],scaler.state_dict(),'scaler')
        verified_tensors+=equal_state_tree(saved['rng_state'],rng_state(),'rng')
        verified_tensors+=equal_state_tree(saved['warp_state'],warp.state_dict(),'warp')
        verified_tensors+=equal_state_tree(saved['strong_stream'],strong.state_dict(),'strong_stream')
        verified_tensors+=equal_state_tree(saved['weak_stream'],auxiliary.state_dict(),'weak_stream')
        write_json(args.output/f'restore-{completed:06d}.json',{'status':'passed','protocol_sha256':args.protocol_sha256,
            'arm':args.arm,'checkpoint_sha256':resumed_from['sha256'],'completed_updates':completed,
            'model_state_digest':saved['model_state_digest'],'additional_exact_restored_tensors':verified_tensors,
            'optimizer_layout':layout,'scaler_and_sampler_and_all_rng_exact':True})
        del saved
    else:
        check_optimizer_state(optimizer,model,0,layout)
    require(auxiliary.completed_cycles<args.target_weak_passes or pending_validation is not None,'Requested data exposure is already complete')
    model.frequency_augmentation=warp
    stopped={'requested':False}
    def stop(signum,frame):stopped['requested']=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    segment_started=time.monotonic()
    validation_directory=args.output
    state={'status':'starting','protocol_sha256':args.protocol_sha256,'arm':args.arm,'runtime':runtime,
        'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'resumed_from':resumed_from,'input_layout':input_layout_evidence,'segment_started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(args.output/'run-state.json',state)
    print(json.dumps({'status':'initialized','arm':args.arm,'completed_updates':completed,
        'resumed':resumed_from is not None,'protocol_sha256':args.protocol_sha256}),flush=True)
    def checkpoint():
        torch.cuda.synchronize()
        model_state=cpu_tree(saved_model_state(model),torch)
        value={'kind':'full_vaani_resume_v1','protocol_sha256':args.protocol_sha256,'arm':args.arm,'runtime':runtime,
            'completed_updates':completed,'attempted_updates':attempted,'model_state':model_state,
            'model_state_digest':api.state_digest(model_state),'optimizer_state':cpu_tree(optimizer.state_dict(),torch),
            'optimizer_layout':layout,'scaler_state':scaler.state_dict(),'warp_state':warp.state_dict(),
            'strong_stream':strong.state_dict(),'weak_stream':auxiliary.state_dict(),'rng_state':rng_state(),
            'history':history,'selected':selected,'best_trained':best_trained,'next_validation_index':next_validation_index,'pending_validation':pending_validation,
            'train_seconds':train_seconds,'validation_seconds':validation_seconds,'skipped_attempts':skipped_attempts,
            'source_initialization':initialization,'exposure':exposure(strong,auxiliary)}
        save_checkpoint(args.output/'last.pt',value)
        del value,model_state
    def validation(index):
        nonlocal selected,best_trained,next_validation_index,pending_validation,validation_seconds
        milestone=index/2
        if index:
            expected=sampler['milestones'][index-1]
            require(expected['weak_pass_target']==milestone and expected['completed_updates_if_no_scaler_skips']==completed,
                'Committed update count differs from the registered whole-clip sampler')
            require(expected['weak_draws']==dict(auxiliary.draws) and expected['strong_draws']==dict(strong.draws)
                and expected['weak_cursor']==auxiliary.cursor and expected['strong_cursor']==strong.cursor,
                'Actual per-tier data exposure differs from the registered sampler')
        digest=api.state_digest(saved_model_state(model))
        journal=validation_directory/f'validation-{index:03d}.json'
        pending_validation={'index':index,'weak_pass_target':milestone,'model_state_digest':digest,'completed_updates':completed}
        checkpoint()
        before=time.monotonic()
        print(json.dumps({'status':'validation_start','arm':args.arm,'weak_pass_target':milestone,
            'completed_updates':completed,'reuse_committed_journal':journal.exists()}),flush=True)
        if journal.exists():
            entry=json.loads(journal.read_text())
            require(entry['protocol_sha256']==args.protocol_sha256 and entry['arm']==args.arm and entry['model_state_digest']==digest
                and entry['completed_updates']==completed and entry['weak_pass_target']==milestone,'Pending calibration journal differs')
            require(sha(args.output/f'validation-{index:03d}-predictions.jsonl')==entry['prediction_sha256'],'Pending prediction evidence differs')
        else:
            before_rng=rng_state();before_warp=warp.state_dict()['generator_state'].clone()
            with torch.autocast(device_type='cuda',enabled=False):
                raw=api.infer_candidates(model,PredictClips(),protocol['evaluation']['batch_size'])
            require(len(raw)==2501,'Incomplete organizer forward pass')
            # Reference events enter only after the model has finished inference.
            records=[({**row,'events':reference_by_id[row['clip_id']]['events']},candidates) for row,candidates in raw]
            choice,trials=api.select_decoder(records,**protocol['evaluation']['decoder_grid'])
            predictions=api.decode_records(raw,choice['decoder'])
            coverage=api.validate_predictions(predictions,prediction_rows)
            require(api.score_records(references,predictions)==choice['metrics'],'Scorer parity changed')
            after_rng=rng_state()
            require(torch.equal(before_rng['torch_cpu'],after_rng['torch_cpu'])
                and all(torch.equal(a,b) for a,b in zip(before_rng['cuda'],after_rng['cuda'],strict=True))
                and torch.equal(before_warp,warp.state_dict()['generator_state']),'Validation advanced a training RNG')
            prediction_path=args.output/f'validation-{index:03d}-predictions.jsonl'
            prediction_path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in predictions))
            entry={'protocol_sha256':args.protocol_sha256,'arm':args.arm,'weak_pass_target':milestone,
                'completed_updates':completed,'model_state_digest':digest,'metrics':choice['metrics'],'decoder':choice['decoder'],
                'calibration':trials,'coverage':coverage,'prediction_sha256':sha(prediction_path),
                'inference_and_calibration_seconds':time.monotonic()-before,'exposure':exposure(strong,auxiliary),
                'scope':'Organizer validation used for model development; not an independent withheld test or leaderboard result'}
            write_json(journal,entry)
            del raw,records,predictions,trials
        new_global=selected is None or entry['metrics']['combined']>selected['metrics']['combined']
        new_trained=milestone>0 and (best_trained is None or entry['metrics']['combined']>best_trained['metrics']['combined'])
        if new_global or new_trained:
            best={'kind':'full_vaani_inference_v1','protocol_sha256':args.protocol_sha256,'arm':args.arm,'runtime':runtime,
                'model_state':cpu_tree(saved_model_state(model),torch),'decoder':entry['decoder'],'metrics':entry['metrics'],
                'weak_pass_target':milestone,'completed_updates':completed,'model_state_digest':digest}
            # A new immutable name leaves the previously committed selection
            # recoverable if the process stops before last.pt is published.
            artifact_id=hashlib.sha256(f'{digest}:{completed}:{milestone}'.encode()).hexdigest()
            best_file='best-'+artifact_id+'.pt'
            save_checkpoint(args.output/best_file,best)
            candidate={key:entry[key] for key in ('weak_pass_target','completed_updates','metrics','decoder','model_state_digest')}
            candidate.update(checkpoint_file=best_file,checkpoint_sha256=sha(args.output/best_file));del best
            if new_global:selected=dict(candidate)
            if new_trained:best_trained=dict(candidate)
        require(not history or history[-1]['weak_pass_target']<milestone,'Validation history would repeat a milestone')
        history.append(entry);next_validation_index=index+1;pending_validation=None
        validation_seconds+=time.monotonic()-before
        checkpoint()
        prune_unselected_checkpoint_files(args.output,selected,best_trained)
        write_json(args.output/'history.json',{'protocol_sha256':args.protocol_sha256,'arm':args.arm,'validations':history,'selected':selected,'best_trained':best_trained})
        print(json.dumps({'status':'validation_complete','arm':args.arm,'weak_pass_target':milestone,'updates':completed,'metrics':entry['metrics'],
            'decoder':entry['decoder'],'seconds':entry['inference_and_calibration_seconds'],'exposure':entry['exposure']}),flush=True)
    progress=args.output/'updates.jsonl'
    try:
        if pending_validation is not None:validation(pending_validation['index'])
        if not history:validation(0)
        checkpoint()
        with progress.open('a') as log:
            while auxiliary.completed_cycles<args.target_weak_passes:
                if args.stop_after_updates is not None and completed>=args.stop_after_updates:break
                if stopped['requested'] or time.monotonic()-segment_started>=args.max_seconds:break
                gold_rows,weak_rows=strong.take(),auxiliary.take()
                # Retry a scaler overflow on the same examples/random draws.
                # A full data pass therefore counts committed batches, not drops.
                before_rng=rng_state();before_warp=warp.state_dict()
                rates=apply_schedule(optimizer,protocol['schedule'],completed)
                before=time.monotonic();result=None
                for retry in range(8):
                    attempted+=1
                    result=mixed_update(model,gold_rows,weak_rows,optimizer,scaler,warp,api,amp=True,
                        weak_weight=protocol['loss']['weak_weight'],meter=meter)
                    if not result['skipped_update']:break
                    skipped_attempts+=1;restore_rng(before_rng);warp.load_state_dict(before_warp)
                require(result is not None and not result['skipped_update'],'Repeated scaler overflow; retain previous safe checkpoint')
                torch.cuda.synchronize();elapsed=time.monotonic()-before
                completed+=1;train_seconds+=elapsed
                record={key:value for key,value in result.items() if not key.endswith('_clip_ids') and not key.startswith('phase_')}
                record.update(completed_updates=completed,attempted_updates=attempted,seconds=elapsed,rates=rates,
                    strong_completed_passes=strong.completed_cycles,weak_completed_passes=auxiliary.completed_cycles)
                log.write(json.dumps(record,allow_nan=False)+'\n')
                if completed%100==0:
                    log.flush();state.update(status='training',completed_updates=completed,attempted_updates=attempted,exposure=exposure(strong,auxiliary),
                        elapsed_segment_seconds=time.monotonic()-segment_started,train_seconds=train_seconds,validation_seconds=validation_seconds)
                    write_json(args.output/'run-state.json',state)
                    print(json.dumps({'status':'training','arm':args.arm,'updates':completed,'weak_completed_passes':auxiliary.completed_cycles,
                        'weak_draws':sum(auxiliary.draws.values()),'strong_loss':result['strong_loss'],'weak_loss':result['weak_loss'],'seconds':elapsed}),flush=True)
                if completed%protocol['execution']['checkpoint_every_updates']==0:checkpoint()
                target=next_validation_index*39770
                if sum(auxiliary.draws.values())>=target and next_validation_index<=args.target_weak_passes*2:validation(next_validation_index)
                if stopped['requested'] or time.monotonic()-segment_started>=args.max_seconds:break
        checkpoint();check_optimizer_state(optimizer,model,completed,layout)
        status='segment_target_complete' if auxiliary.completed_cycles>=args.target_weak_passes else 'paused_at_completed_update'
        state.update(status=status,completed_updates=completed,attempted_updates=attempted,exposure=exposure(strong,auxiliary),selected=selected,best_trained=best_trained,
            finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_segment_seconds=time.monotonic()-segment_started,
            train_seconds=train_seconds,validation_seconds=validation_seconds,last_checkpoint_sha256=sha(args.output/'last.pt'),
            skipped_attempts=skipped_attempts,all_optimizer_states_checked=True)
        write_json(args.output/'run-state.json',state)
        print(json.dumps(state),flush=True)
    except BaseException:
        # Never label a partially executed/failed update as a resumable one.
        state.update(status='failed',error=traceback.format_exc(),previous_safe_checkpoint=str(args.output/'last.pt'),
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'run-state.json',state)
        raise


if __name__=='__main__':main()
