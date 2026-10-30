"""Audit every released training waveform and preserve justified supervision.

Reads T1_WORK/vaani-full (prepare_full.py) and T1_WORK/vaani-release (the parquet shards); writes
T1_WORK/full-release-audit/training-audit.jsonl (90,635 usable rows).
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import time
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

import os
WORK=Path(os.environ.get('T1_WORK','.'))
HERE=Path(__file__).resolve().parent
OUT=WORK/'full-release-audit'
RELEASE=WORK/'vaani-full'
CATEGORIES=('animal','vehicle_traffic','baby_child','singing_music','phone_signal_alarm','appliance_machine','human_non_speech')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def canonical(values):
    assert isinstance(values,list) and all(v in CATEGORIES for v in values)
    return [v for v in CATEGORIES if v in values]

def recover():
    quarantines=[json.loads(line) for line in (RELEASE/'quarantine.jsonl').read_text().splitlines()]
    recovered,excluded=[],[]
    bindings={}
    for q in quarantines:
        source=WORK/'vaani-release'/q['parquet']
        assert source.is_file(),'Required original quarantine parquet is unavailable'
        if str(source) not in bindings:
            bindings[str(source)]=sha(source)
        raw=pq.read_table(source,columns=['audio','NoiseCategory','NoiseSubCategoryTimeStamp','annotationQuality','duration']).slice(q['row_index'],1).to_pylist()[0]
        assert raw['annotationQuality']==q['quality'] and raw['audio']['path']==q['audio_path']
        payload=raw['audio']['bytes']
        evidence={**q,'embedded_bytes':len(payload),'embedded_sha256':hashlib.sha256(payload).hexdigest(),
                  'declared_categories':canonical(raw['NoiseCategory'])}
        # Both source decode failures contain a whole HTML document instead of audio.
        # No codec switch can recover a waveform from that non-audio payload.
        if payload.lstrip().lower().startswith(b'<!doctype html>'):
            evidence.update(status='excluded_non_audio_source_payload',reason='Embedded payload starts with HTML doctype; no audio waveform available in released source row',
                            magic_hex=payload[:16].hex(),bounded_repair='Inspected original embedded bytes; no replacement audio or reinterpretation of HTML')
            excluded.append(evidence)
            continue
        waveform,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
        assert waveform.size and np.isfinite(waveform).all()
        waveform=waveform.mean(axis=1,dtype=np.float32)
        if rate!=16000:
            divisor=math.gcd(rate,16000)
            waveform=resample_poly(waveform,16000//divisor,rate//divisor)
        assert len(waveform)>0 and np.isfinite(waveform).all()
        identity=f"{q['dataset']}\0{q['revision']}\0{q['parquet']}\0{q['row_index']}"
        audio=OUT/'recovered-audio'/(hashlib.sha256(identity.encode()).hexdigest()[:20]+'.wav')
        audio.parent.mkdir(exist_ok=True)
        assert not audio.exists()
        sf.write(audio,waveform,16000,subtype='FLOAT')
        evidence.update(status='recovered_as_weak_clip_labels',duration_seconds=len(waveform)/16000,
            audio_path=str(audio),audio_sha256=sha(audio),original_sample_rate=rate,
            invalid_timestamps_used=False,timestamps_clipped=False)
        recovered.append({'clip_id':Path(q['audio_path']).name,'audio_path':str(audio),'duration':len(waveform)/16000,
            'quality':q['quality'],'events':None,'language':'unneeded_for_training',
            'source':{'dataset':q['dataset'],'revision':q['revision'],'parquet':q['parquet'],'row_index':q['row_index'],
                      'metadata':{'NoiseCategory':raw['NoiseCategory']}},'_recovery_evidence':evidence})
    write(OUT/'quarantine-recovery.json',{'recovered':[r['_recovery_evidence'] for r in recovered],'excluded':excluded,
        'source_parquet_sha256':bindings,'original_quarantine_sha256':sha(RELEASE/'quarantine.jsonl')})
    return recovered,excluded

def inspect(row):
    audio=Path(row['audio_path'])
    if not audio.is_absolute(): audio=(RELEASE/audio).resolve()
    assert audio.is_file() and not audio.is_symlink(),str(audio)
    payload=audio.read_bytes()
    info=sf.info(io.BytesIO(payload))
    assert info.samplerate==16000 and info.channels==1 and info.frames>0 and info.subtype=='FLOAT'
    waveform,rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=False)
    assert rate==16000 and len(waveform)==info.frames and np.isfinite(waveform).all()
    duration=info.frames/16000
    assert abs(duration-row['duration'])<=1/16000
    frames=(info.frames+639)//640
    windows=1+max(0,math.ceil((frames-250)/125))
    tags=canonical(row['source']['metadata'].get('NoiseCategory'))
    assert tags,'Every included weak clip needs observed labels'
    events=row['events']
    timed={e.get('category') for e in events or []}
    assert timed<=set(CATEGORIES)
    is_recovered='_recovery_evidence' in row
    strong=row['quality']=='verified_timestamps' and bool(events) and set(tags)==timed and not is_recovered
    if strong:
        assert all(0<=e['onset']<e['offset']<=duration for e in events)
        reason='Verified Gold timestamps with matching clip categories'
    elif is_recovered:
        reason='Recovered decodable Gold with invalid original timestamp bounds; only clip tags used'
    elif row['quality']=='verified_timestamps':
        reason='Gold clip tags with missing timestamps' if not events else 'Gold clip and timestamp category disagreement'
    else:
        reason='Silver unverified timing ignored; declared clip tags only' if row['quality']=='unverified_timestamps' else 'Bronze clip tags only; timing unavailable'
    source=row['source']
    result={'clip_id':row['clip_id'],'audio_path':str(audio),'duration':duration,'quality':row['quality'],
        'supervision':'strong_gold' if strong else 'weak_clip','supervision_reason':reason,
        'positive_categories':tags,'events':events if strong else None,
        'frame_hop':.04,'frame_offset':.02,'num_frames':frames,'feature_dim':768,'native_windows':windows,
        'source':{k:source[k] for k in ('dataset','revision','parquet','row_index')},
        'audio_bytes':len(payload),'audio_sha256':hashlib.sha256(payload).hexdigest(),
        'decoded_float32_sha256':hashlib.sha256(waveform.astype('<f4',copy=False).tobytes()).hexdigest(),
        'source_samples':info.frames,'source_sample_rate':rate,'source_subtype':info.subtype,
        'source_timing_categories_match_declared':set(tags)==timed,
        'source_timestamp_event_count':len(events or []),'recovered_from_quarantine':is_recovered}
    return result

def main():
    assert not OUT.exists(),'Do not repeat the full audio audit'
    OUT.mkdir()
    started=time.monotonic()
    source_hash=sha(RELEASE/'manifest.jsonl')
    rows=[json.loads(line) for line in (RELEASE/'manifest.jsonl').open()]
    assert len(rows)==90630
    recovered,excluded=recover()
    rows.extend(recovered)
    assert len({r['clip_id'] for r in rows})==len(rows)
    assert len({(r['source']['parquet'],r['source']['row_index']) for r in rows})==len(rows)
    qualities=Counter(); supervision=Counter(); reasons=Counter(); labels=defaultdict(Counter)
    stats=defaultdict(lambda:{'clips':0,'seconds':0.,'bytes':0,'windows':0,'max_windows':0,'max_duration':0.,'durations':[]})
    hashes=defaultdict(list); decoded=defaultdict(list); largest=defaultdict(list)
    manifest=OUT/'training-audit.jsonl'
    with manifest.open('x') as stream, ThreadPoolExecutor(max_workers=2) as pool:
        for index,row in enumerate(pool.map(inspect,rows),1):
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            tier=row['quality']; qualities[tier]+=1; supervision[(tier,row['supervision'])]+=1; reasons[row['supervision_reason']]+=1
            labels[tier].update(row['positive_categories'])
            item=stats[tier]; item['clips']+=1; item['seconds']+=row['duration']; item['bytes']+=row['audio_bytes']; item['windows']+=row['native_windows']
            item['max_windows']=max(item['max_windows'],row['native_windows']); item['max_duration']=max(item['max_duration'],row['duration']); item['durations'].append(row['duration'])
            hashes[row['audio_sha256']].append(row['clip_id']); decoded[row['decoded_float32_sha256']].append(row['clip_id'])
            largest[tier].append({k:row[k] for k in ('clip_id','audio_path','duration','native_windows','audio_sha256','supervision')})
            if index%10000==0:
                stream.flush()
                print(json.dumps({'checked_clips':index,'elapsed_seconds':time.monotonic()-started}),flush=True)
    for tier,item in stats.items():
        durations=item.pop('durations')
        item['duration_quantiles']={str(q):float(np.quantile(durations,q)) for q in (.5,.9,.95,.99,1.)}
        largest[tier]=sorted(largest[tier],key=lambda r:(-r['native_windows'],-r['duration'],r['clip_id']))[:8]
        item['top8_total_windows']=sum(r['native_windows'] for r in largest[tier])
    assert sha(RELEASE/'manifest.jsonl')==source_hash
    summary={'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,
        'source_manifest':str(RELEASE/'manifest.jsonl'),'source_manifest_sha256':source_hash,
        'source_release_raw_rows':90637,'previous_imported_rows':90630,'recovered_gold_weak_rows':len(recovered),
        'included_audio_clips':len(rows),'excluded_rows':len(excluded),'excluded_reason':'2 original Silver payloads are HTML documents instead of audio',
        'quality_counts':dict(qualities),'supervision_counts':[{'quality':k[0],'supervision':k[1],'clips':v} for k,v in supervision.items()],
        'supervision_reasons':dict(reasons),'category_clip_counts':{k:dict(v) for k,v in labels.items()},'tier_audio':dict(stats),
        'largest_eight_by_tier':dict(largest),'waveform_bytes':sum(v['bytes'] for v in stats.values()),
        'raw_waveform_duplicate_groups':[v for v in hashes.values() if len(v)>1],
        'decoded_waveform_duplicate_groups':[v for v in decoded.values() if len(v)>1],
        'all_audio_decoded_finite':True,'all_audio_headers_and_content_hashed':True,'old_geographic_exclusions_applied':False,
        'old_gold_clean_total_restored':qualities['verified_timestamps']-len(recovered),
        'weak_unlisted_categories':'unknown; masked, never assumed negative',
        'strong_timestamp_rounding':'Uses historical importer audio bounds; 930 original boundaries were corrected by at most0.0005s; no new clipping',
        'training_manifest':str(manifest),'training_manifest_sha256':sha(manifest),
        'quarantine_recovery_sha256':sha(OUT/'quarantine-recovery.json'),'script_sha256':sha(__file__),'model_constructed':False}
    assert summary['included_audio_clips']+summary['excluded_rows']==90637
    write(OUT/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','included_audio_clips','recovered_gold_weak_rows','excluded_rows','quality_counts','supervision_counts','waveform_bytes','wall_seconds')}),flush=True)

if __name__=='__main__':
    main()
