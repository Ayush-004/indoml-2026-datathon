"""Stream losslessly encoded task inputs into bounded upload archives (the Kaggle input dataset "vaani-inputs").

Inputs: T1_WORK/full-release-audit/training-audit.jsonl (audit_full_release.py) and the organizer validation
manifests in T1_ORGANIZER_AUDIT/input-audit/ (inference.jsonl: clip_id, audio_path, duration of the 2,501 validation
clips; references.jsonl: their labelled events; official-audio.jsonl: the audio inventory). Output:
T1_WORK/full-input-staging/input-upload/ (training.jsonl, organizer-*.jsonl, audio zips, full-input-contract.json).
The organizer validation audio is packed only for in-session measurement; it is never a training row.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_STORED
import hashlib
import json
import time
from lossless_audio import encode

import os
HERE=Path(__file__).resolve().parent
WORK=Path(os.environ.get('T1_WORK','.'))
OUT=WORK/'full-input-staging'
OLD=Path(os.environ.get('T1_ORGANIZER_AUDIT','organizer-audit'))

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def add(archive,name,payload):
    info=ZipInfo(name,date_time=(1980,1,1,0,0,0))
    info.compress_type=ZIP_STORED;info.create_system=3;info.external_attr=0o100644<<16
    archive.writestr(info,payload)

def main():
    assert not OUT.exists(),'Do not repeat completed or active packaging'
    upload=OUT/'input-upload';upload.mkdir(parents=True)
    started=time.monotonic()
    source=WORK/'full-release-audit/training-audit.jsonl'
    assert sha(source)=='653a887af80b9d8c55994be6c5c5e599111da74520c4f2c640cbaefb015c4ad9'
    rows=[json.loads(line) for line in source.open()]
    assert len(rows)==90635
    types=Counter();input_bytes=output_bytes=0;archives=[];seen_paths=set()
    with (upload/'training.jsonl').open('x') as manifest,(OUT/'audio-conversion-ledger.jsonl').open('x') as ledger,ThreadPoolExecutor(max_workers=2) as pool:
        for shard_start in range(0,len(rows),20000):
            selected=rows[shard_start:shard_start+20000]
            path=upload/f'training-audio-{shard_start//20000:02d}.zip'
            with ZipFile(path,'x',compression=ZIP_STORED,allowZip64=True) as archive:
                for chunk_start in range(0,len(selected),256):
                    chunk=selected[chunk_start:chunk_start+256]
                    for row,(payload,extension,evidence) in zip(chunk,pool.map(encode,chunk),strict=True):
                        name='train/audio/'+row['audio_sha256'][:24]+'.'+extension
                        assert name not in seen_paths;seen_paths.add(name)
                        add(archive,name,payload)
                        exported={key:row[key] for key in ('clip_id','duration','quality','supervision','positive_categories','events','frame_hop','frame_offset','num_frames','feature_dim','native_windows')}
                        exported.update(audio_path=name,audio_sha256=evidence['packaged_audio_sha256'],decoded_float32_sha256=evidence['decoded_float32_sha256'])
                        manifest.write(json.dumps(exported,allow_nan=False)+'\n')
                        ledger.write(json.dumps({'population':'training','clip_id':row['clip_id'],'path':name,**evidence})+'\n')
                        types[evidence['subtype']]+=1;input_bytes+=evidence['original_bytes'];output_bytes+=len(payload)
                    checked=shard_start+chunk_start+len(chunk)
                    if checked%2048==0 or checked==len(rows):
                        print(json.dumps({'phase':'lossless_training_pack','clips':checked,'source_bytes':input_bytes,'packed_audio_bytes':output_bytes,'seconds':time.monotonic()-started}),flush=True)
            archives.append({'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path),'training_clips':len(selected)})
    references=OLD/'input-audit/references.jsonl'
    inventory=[json.loads(line) for line in (OLD/'input-audit/official-audio.jsonl').open()]
    noisy={row['clip_id']:row for row in inventory if row['kind']!='clean_reference'}
    inference=[json.loads(line) for line in (OLD/'input-audit/inference.jsonl').open()]
    assert len(noisy)==len(inference)==2501
    organizer_archive=upload/'organizer-audio.zip'
    with ZipFile(organizer_archive,'x',compression=ZIP_STORED,allowZip64=True) as archive,(upload/'organizer-inference.jsonl').open('x') as manifest,(OUT/'audio-conversion-ledger.jsonl').open('a') as ledger:
        for row in inference:
            assert set(row)=={'clip_id','audio_path','duration'}
            original=noisy[row['clip_id']]
            payload,extension,evidence=encode({**row,'audio_sha256':original['sha256']})
            name='organizer/audio/'+original['sha256'][:24]+'.'+extension
            assert name not in seen_paths;seen_paths.add(name)
            add(archive,name,payload)
            manifest.write(json.dumps({**row,'audio_path':name},allow_nan=False)+'\n')
            ledger.write(json.dumps({'population':'organizer_noisy','clip_id':row['clip_id'],'path':name,**evidence})+'\n')
    archives.append({'path':organizer_archive.name,'bytes':organizer_archive.stat().st_size,'sha256':sha(organizer_archive),'organizer_noisy_clips':2501})
    (upload/'organizer-references.jsonl').write_bytes(references.read_bytes())
    contract={'status':'ready_for_private_upload','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Full released training inputs and organizer noisy evaluation inputs; reference labels in a separate scoring file',
        'training_clips':90635,'strong_gold':11095,'weak_gold':16,'weak_silver':61640,'weak_bronze':17884,
        'excluded_original_rows':2,'organizer_noisy_clips':2501,'organizer_clean_reference_audio_included':False,
        'training_waveform_encoding_counts':dict(types),'training_source_audio_bytes':input_bytes,'training_packaged_audio_bytes':output_bytes,
        'all_packaged_waveforms_roundtrip_bit_exact':True,'existing_soundfile_loader_supported_formats':['FLAC','WAV'],
        'archives':archives,'source_training_audit_sha256':sha(source),'source_organizer_inference_sha256':sha(OLD/'input-audit/inference.jsonl'),
        'source_organizer_references_sha256':sha(references),'source_organizer_audio_inventory_sha256':sha(OLD/'input-audit/official-audio.jsonl'),
        'training_manifest_sha256':sha(upload/'training.jsonl'),'organizer_inference_sha256':sha(upload/'organizer-inference.jsonl'),
        'organizer_references_sha256':sha(upload/'organizer-references.jsonl'),
        'conversion_ledger_sha256':sha(OUT/'audio-conversion-ledger.jsonl'),
        'code_sha256':{p.name:sha(p) for p in (Path(__file__),HERE/'lossless_audio.py')},
        'temporary_local_staging_removable_after_remote_identity_verification':True,
        'local_original_training_data_preserved':True,'elapsed_seconds':time.monotonic()-started}
    write(upload/'full-input-contract.json',contract)
    write(upload/'dataset-metadata.json',{'title':'IndoML Full VAANI Training and Validation',
        'id':os.environ.get('KAGGLE_DATASET_OWNER','<owner>')+'/vaani-inputs','licenses':[{'name':'other'}],
        'description':'Private task inputs for authorized model development. Original released VAANI audio, annotations and organizer validation retain their authors and terms; no redistribution or relicensing. Training:90635 usable rows with audited strong/weak labels; organizer noisy inputs:2501, kept out of gradients. Reference labels are separate from predictor inputs; no clean reference audio or credentials. FLAC/WAV encoding preserves every float32 waveform sample exactly.'})
    write(OUT/'staging-result.json',{**contract,'upload_directory':str(upload),'upload_files':sorted(p.name for p in upload.iterdir()),
        'upload_total_bytes':sum(p.stat().st_size for p in upload.iterdir())})
    print(json.dumps({'status':'ready_for_private_upload','training_clips':90635,'organizer_noisy_clips':2501,
        'source_training_audio_bytes':input_bytes,'encoded_training_audio_bytes':output_bytes,'archive_bytes':sum(a['bytes'] for a in archives),
        'encoding_counts':dict(types),'elapsed_seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__':main()
