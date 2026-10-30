"""Local event-set versus RED pilot on the existing, shared frozen ATST cache.

`check` validates the new assignment/loss and measures cached-head cost only.
`train` is an explicit separate action, not triggered by check or this import.
No organizer inputs, remote jobs, new encoders, or downloads are used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('T1_WORK', '.'))
sys.path.insert(0, str(HERE.parents[1]/'atst_red'))
from indoml.intervals import (IntervalDataset, IntervalModel, collate_intervals,
                             interval_loss, candidate_intervals, select_decoder,
                             decode_candidates)
from indoml.metrics import score_records
from indoml.transfer import write_predictions, feature_path

THRESHOLDS = [.1, .2, .35, .5, .65, .8, .9, .95]
NMS = [0., .3, .5]
MIN_DURATION = .08


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def save(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);temp.replace(path)


class EventSet(nn.Module):
    """An unordered set of sixteen outputs, with objectness and two endpoints."""
    def __init__(self, queries=16, width=128):
        super().__init__()
        self.queries=queries;self.width=width
        self.project=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,width),nn.GELU())
        self.query=nn.Parameter(torch.randn(queries,width)*.02)
        layer=nn.TransformerDecoderLayer(width,4,dim_feedforward=256,dropout=.1,
                                        activation='gelu',batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,2,norm=nn.LayerNorm(width))
        self.objectness=nn.Linear(width,1)
        self.endpoints=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,2))
        self.register_buffer('initial_centers',torch.logit(torch.linspace(.03,.97,queries)))
        nn.init.constant_(self.objectness.bias,-2.)
        nn.init.normal_(self.endpoints[-1].weight,std=.01)
        nn.init.zeros_(self.endpoints[-1].bias)

    def forward(self,features,lengths,durations):
        memory=self.project(features)
        positions=torch.arange(features.shape[1],device=features.device).float()[None,:,None]
        # Position is explicit; masking excludes every padded frame from attention.
        scale=torch.exp(torch.arange(0,self.width,2,device=features.device).float()*(-math.log(10000.)/self.width))
        angles=positions*scale
        positional=torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
        memory=memory+positional
        mask=torch.arange(features.shape[1],device=features.device)[None]>=lengths[:,None]
        query=self.query[None].expand(features.shape[0],-1,-1)
        encoded=self.decoder(query,memory,memory_key_padding_mask=mask)
        raw=self.endpoints(encoded)
        center=(raw[...,0]+self.initial_centers).sigmoid()*durations[:,None]
        width=(raw[...,1]-2.2).sigmoid()*durations[:,None]
        onset=(center-width/2).clamp_min(0.)
        offset=torch.minimum(center+width/2,durations[:,None])
        return {'logits':self.objectness(encoded).squeeze(-1),'boxes':torch.stack((onset,offset),dim=-1)}


def pair_cost(boxes,logits,target):
    tolerance=(.2*(target[:,1]-target[:,0])).clamp_min(.05)
    endpoint=(boxes[:,None]-target[None]).abs().sum(-1)/(2*tolerance[None])
    intersection=(torch.minimum(boxes[:,None,1],target[None,:,1])-torch.maximum(boxes[:,None,0],target[None,:,0])).clamp_min(0.)
    union=(boxes[:,1]-boxes[:,0])[:,None]+(target[:,1]-target[:,0])[None]-intersection
    iou=intersection/union.clamp_min(1e-6)
    return -F.logsigmoid(logits)[:,None]+endpoint+2*(1-iou)


def assign(boxes,logits,target):
    if not len(target):return np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64)
    assert len(target)<=len(boxes),'Query capacity cannot discard a reference'
    return linear_sum_assignment(pair_cost(boxes,logits,target).detach().cpu().numpy())


def set_loss(output,rows):
    object_losses=[];boundary_losses=[];iou_losses=[]
    for boxes,logits,row in zip(output['boxes'],output['logits'],rows,strict=True):
        target=boxes.new_tensor([[e['onset'],e['offset']] for e in row['events']]).reshape(-1,2)
        pi,ri=assign(boxes,logits,target)
        label=torch.zeros_like(logits);label[pi]=1.
        weight=torch.where(label.bool(),1.,.1)
        object_losses.append((F.binary_cross_entropy_with_logits(logits,label,reduction='none')*weight).sum()/weight.sum())
        if len(ri):
            selected=boxes[pi];expected=target[ri]
            tolerance=(.2*(expected[:,1]-expected[:,0])).clamp_min(.05)
            boundary_losses.append(F.smooth_l1_loss((selected-expected)/tolerance[:,None],torch.zeros_like(expected),reduction='none').mean())
            intersection=(torch.minimum(selected[:,1],expected[:,1])-torch.maximum(selected[:,0],expected[:,0])).clamp_min(0.)
            union=(selected[:,1]-selected[:,0])+(expected[:,1]-expected[:,0])-intersection
            iou_losses.append((1-intersection/union.clamp_min(1e-6)).mean())
    objectness=torch.stack(object_losses).mean()
    boundary=torch.stack(boundary_losses).mean() if boundary_losses else objectness*0
    iou=torch.stack(iou_losses).mean() if iou_losses else objectness*0
    loss=objectness+boundary+2*iou
    return loss,{'objectness':float(objectness.detach()),'boundary':float(boundary.detach()),'iou':float(iou.detach())}


def datasets():
    train=IntervalDataset(WORK/'inner-split/train.jsonl',mode='train')
    val=IntervalDataset(WORK/'inner-split/val.jsonl',mode='val')
    assert len(train)==8213 and len(val)==1439
    assert not {r['clip_id'] for r in train.rows}&{r['clip_id'] for r in val.rows}
    assert train.feature_cache_id==val.feature_cache_id
    assert max(len(r['events']) for r in train.rows+val.rows)<=16
    return train,val


def new_model(arm):
    return EventSet() if arm=='set' else IntervalModel(mode='red',epn_context='per_class')


def forward(model,batch,arm):
    if arm=='set':
        durations=batch['features'].new_tensor([r['duration'] for r in batch['rows']])
        return model(batch['features'],batch['lengths'],durations)
    return model(batch['features'],batch['lengths'])


def loss_for(model,batch,arm,positive_weights):
    output=forward(model,batch,arm)
    return set_loss(output,batch['rows']) if arm=='set' else interval_loss(output,batch,positive_weights,'cpu','red',100.,1.)


def set_predictions(raw,threshold):
    result=[]
    for row,logits,boxes in raw:
        probability=1/(1+np.exp(-np.clip(logits,-60,60)))
        events=[{'onset':float(box[0]),'offset':float(box[1])} for box,p in zip(boxes,probability,strict=True)
                if p>=threshold and box[1]-box[0]>=MIN_DURATION]
        result.append({'clip_id':row['clip_id'],'duration':row['duration'],
                       'events':sorted(events,key=lambda e:(e['onset'],e['offset']))})
    return result


@torch.inference_mode()
def validate(model,val,arm,batch_size,fixed_decoder=None):
    model.eval();raw=[]
    loader=DataLoader(val,batch_size=batch_size,collate_fn=collate_intervals,generator=torch.Generator().manual_seed(0))
    for batch in loader:
        output=forward(model,batch,arm)
        if arm=='set':
            raw.extend((row,output['logits'][i].numpy(),output['boxes'][i].numpy()) for i,row in enumerate(batch['rows']))
        else:
            raw.extend((row,candidate_intervals(row,output['presence'][i,:row['num_frames']].numpy(),output['distances'][i,:row['num_frames']].numpy())) for i,row in enumerate(batch['rows']))
    if fixed_decoder is not None:
        predictions=set_predictions(raw,fixed_decoder['threshold']) if arm=='set' else [decode_candidates(row,candidates,fixed_decoder) for row,candidates in raw]
        choice={'decoder':fixed_decoder,'metrics':score_records(val.rows,predictions)};trials=[]
    elif arm=='set':
        trials=[{'decoder':{'threshold':t,'min_duration':MIN_DURATION,'nms':None},'metrics':score_records(val.rows,set_predictions(raw,t))} for t in THRESHOLDS]
        choice=max(trials,key=lambda t:(t['metrics']['combined'],t['decoder']['threshold']))
        predictions=set_predictions(raw,choice['decoder']['threshold'])
    else:
        choice,trials=select_decoder(raw,THRESHOLDS,NMS,MIN_DURATION)
        predictions=[decode_candidates(row,candidates,choice['decoder']) for row,candidates in raw]
    return choice,trials,predictions


def subset(dataset,indices):
    value=Subset(dataset,indices)
    value.rows=[dataset.rows[i] for i in indices]
    return value


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    # Distinct slots must preserve exact-coincident and overlapping references.
    boxes=torch.tensor([[.1,.4],[.1,.4],[.3,.7],[.8,.9]])
    target=boxes[:3].clone();logits=torch.ones(4)
    pi,ri=assign(boxes,logits,target)
    assert len(set(pi))==len(set(ri))==3 and torch.equal(boxes[pi],target[ri])
    perfect_rows=[{'events':[{'onset':float(a),'offset':float(b)} for a,b in target]}]
    perfect={'boxes':boxes[None].clone().requires_grad_(True),'logits':logits[None].clone().requires_grad_(True)}
    loss,_=set_loss(perfect,perfect_rows);loss.backward()
    assert torch.isfinite(loss) and perfect['logits'].grad is not None
    empty_loss,_=set_loss(perfect,[{'events':[]}]);assert torch.isfinite(empty_loss)
    permutation=[2,0,1]
    permuted_loss,_=set_loss(perfect,[{'events':[perfect_rows[0]['events'][i] for i in permutation]}])
    assert torch.allclose(loss,permuted_loss,atol=1e-7,rtol=1e-7)
    train,val=datasets();batch=collate_intervals([train[i] for i in range(16)])
    positive_weights,_=train.positive_weights(5.)
    measurements={}
    for arm in ('red','set'):
        torch.manual_seed(73);model=new_model(arm)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        started=time.perf_counter()
        for _ in range(3):
            model.train();optimizer.zero_grad(set_to_none=True)
            value,_=loss_for(model,batch,arm,positive_weights)
            assert torch.isfinite(value);value.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        train_seconds=(time.perf_counter()-started)/3
        model.eval()
        with torch.inference_mode():
            a=forward(model,batch,arm)
            if arm=='set':
                padded={**batch,'features':F.pad(batch['features'],(0,0,0,31),value=103.)}
                b=forward(model,padded,arm)
                assert torch.allclose(a['boxes'],b['boxes'],atol=2e-5,rtol=2e-5)
                assert torch.allclose(a['logits'],b['logits'],atol=2e-5,rtol=2e-5)
            timings=[]
            for _ in range(10):
                before=time.perf_counter();forward(model,batch,arm);timings.append(time.perf_counter()-before)
        parameters=sum(p.numel() for p in model.parameters())
        measurements[arm]={'head_parameters':parameters,'head_fp32_parameter_bytes':4*parameters,
                           'observed_mean_3_training_steps_seconds':train_seconds,
                           'median_10_cached_head_forwards_seconds':float(np.median(timings)),
                           'estimated_30epoch_training_seconds_excluding_validation':train_seconds*math.ceil(len(train)/16)*30}
        choice,_,predictions=validate(model,subset(val,list(range(4))),arm,4)
        fit,_,fit_predictions=validate(model,subset(train,list(range(4))),arm,4,fixed_decoder=choice['decoder'])
        assert len(predictions)==len(fit_predictions)==4 and fit['decoder']==choice['decoder']
        del optimizer,model
    report={'scope':'New head/loss smoke and tiny warm-cache benchmark only; no full pilot or raw-audio inference',
            'checks':['one-to-one coincident/overlapping target assignment','empty-target finite loss','target-order invariance','finite gradients and optimizer updates on16 TRAIN clips','padded-feature invariance','both-arm validation and fixed-decoder training-fit path on4 clips'],
            'hardware':{'platform':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'threads':torch.get_num_threads(),'device':'cpu'},
            'batch_size':16,'clip_durations_seconds':[r['duration'] for r in batch['rows']],
            'input_frames':[r['num_frames'] for r in batch['rows']],
            'cache':'Existing frozen pretrained ATST feature cache; first16 TRAIN clips, features resident in RAM',
            'excluded_costs':['audio read/decode','mel frontend','ATST encoder','feature-cache construction and disk read','event decoding and calibration'],
            'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'peak_memory_scope':'Whole check process including imports/datasets/two successive heads; not per-arm or end-to-end memory',
            'measurements':measurements,'train_clips':len(train),'inner_clips':len(val),'cache_id':train.feature_cache_id,
            'input_hashes':{str(p):sha(p) for p in [train.manifest,val.manifest,Path(__file__)]},
            'loaded_training_feature_sha256':{str(feature_path(r,train.manifest)):sha(feature_path(r,train.manifest)) for r in batch['rows']}}
    write_json(out/'check.json',report);print(json.dumps(report,indent=2))


def train_arm(args):
    # A new destination keeps previous evidence and interrupted runs intact.
    if args.output.exists():raise ValueError('Use a new output directory; existing pilot results are never overwritten')
    args.output.mkdir(parents=True)
    train,val=datasets();positive_weights,_=train.positive_weights(5.)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=new_model(args.arm)
    # Reset after architecture-specific initialization; both arms share shuffle order.
    torch.manual_seed(args.seed+1)
    generator=torch.Generator().manual_seed(args.seed+2)
    loader=DataLoader(train,batch_size=16,shuffle=True,generator=generator,collate_fn=collate_intervals)
    fit_indices=sorted(range(len(train)),key=lambda i:hashlib.sha256(train.rows[i]['clip_id'].encode()).hexdigest())[:512]
    fit_data=subset(train,fit_indices)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[];selected=None;start=time.monotonic()
    protocol={'arm':args.arm,'epochs':30,'seed':args.seed,'batch_size':16,'lr':.001,'weight_decay':.0001,
              'gradient_clip':5.,'selection':'earliest exact maximum INNER F1+Dice including epoch0',
              'scope':'Architecture check on reused INNER; no organizer, OUTER or raw-audio inference',
              'training_fit':'Fixed512 hashed TRAIN IDs at E0 and every5 epochs, scored with that epoch INNER-selected decoder; no TRAIN decoder search',
              'training_fit_clip_id_sha256':hashlib.sha256('\n'.join(r['clip_id'] for r in fit_data.rows).encode()).hexdigest(),
              'cache_id':train.feature_cache_id,'train_clips':len(train),'inner_clips':len(val),
              'source_sha256':sha(Path(__file__)),'train_manifest_sha256':sha(train.manifest),'inner_manifest_sha256':sha(val.manifest)}
    write_json(args.output/'protocol.json',protocol)
    for epoch in range(31):
        training_seconds=0.;loss_sum=0.;updates=0;component_sums={}
        if epoch:
            model.train();before=time.monotonic()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss,components=loss_for(model,batch,args.arm,positive_weights)
                if not torch.isfinite(loss):raise ValueError('Nonfinite pilot loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                loss_sum+=float(loss.detach());updates+=1
                for key,value in components.items():component_sums[key]=component_sums.get(key,0.)+value
            training_seconds=time.monotonic()-before
        before=time.monotonic();choice,trials,predictions=validate(model,val,args.arm,16)
        entry={'epoch':epoch,'updates':updates,'mean_training_loss':loss_sum/updates if updates else None,
               'loss_components':{key:value/updates for key,value in component_sums.items()},
               'training_seconds':training_seconds,'validation_seconds':time.monotonic()-before,**choice}
        if epoch%5==0:
            fit_start=time.monotonic()
            fit,_,_=validate(model,fit_data,args.arm,16,fixed_decoder=choice['decoder'])
            entry['training_fit_512']=fit['metrics'];entry['training_fit_seconds']=time.monotonic()-fit_start
        history.append(entry);write_json(args.output/'history.json',history)
        write_json(args.output/f'calibration-{epoch:02d}.json',trials)
        if selected is None or choice['metrics']['combined']>selected['metrics']['combined']:
            selected=entry;write_json(args.output/'selection.json',selected)
            save(args.output/'selected.pt',{'protocol':protocol,'selected':selected,'model_state':model.state_dict()})
            write_predictions(args.output/'selected-predictions.jsonl',predictions)
        save(args.output/'last.pt',{'protocol':protocol,'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                                  'torch_rng':torch.get_rng_state(),'shuffle_rng':generator.get_state(),'history':history,'selected':selected})
        print(json.dumps({'arm':args.arm,**entry,'elapsed_seconds':time.monotonic()-start}),flush=True)
    write_json(args.output/'complete.json',{'completed_epochs':30,'selected':selected,'elapsed_seconds':time.monotonic()-start,
                                         'parameters':sum(p.numel() for p in model.parameters()),
                                         'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    c=sub.add_parser('check');c.add_argument('--output',type=Path,default=HERE/'pilot-check')
    t=sub.add_parser('train');t.add_argument('--arm',choices=['red','set'],required=True);t.add_argument('--seed',type=int,default=73)
    t.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    check(args.output) if args.mode=='check' else train_arm(args)
