"""Narrow fresh-CUDA benchmark adapter; original native sources stay unchanged."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_runtime_transition(expected, current):
    """Only an explicit fresh environment may vary wheel suffix/git location."""
    ignored = {'source_commit','device','include_logits'}
    version_keys = {'torch_version','torchaudio_version'}
    require(set(current) == set(expected), 'Native provenance keys changed')
    for key, value in expected.items():
        if key in version_keys:
            require(str(value).split('+')[0] == '2.8.0' and str(current[key]) == '2.10.0+cu128', 'Explicit benchmark transition is parent2.8.0 to Torch/TorchAudio2.10.0+cu128')
        elif key not in ignored:
            require(current[key] == value, 'Native source/weight/temporal contract changed: '+key)
    return {'kind':'fresh_cuda_environment_from_native_weights','bitwise_MPS_continuation':False,
            'version_transition':{k:{'parent':expected[k],'current':current[k]} for k in version_keys},
            'all_non_runtime_native_fields_equal':True,'source_commit_may_be_absent_in_portable_bundle':True}


def rng_device(model, torch):
    types = {model.device.type,model.head_device.type}
    require(not {'cuda','mps'} <= types, 'A model may not mix CUDA and MPS')
    return torch.device('cuda:0' if 'cuda' in types else 'mps' if 'mps' in types else 'cpu')


def runtime_fingerprint(torch, torchaudio):
    cuda = torch.cuda.is_available()
    result = {'platform':platform.system(),'machine':platform.machine(),'python':platform.python_version(),
              'torch':str(torch.__version__),'torchaudio':str(torchaudio.__version__),
              'cuda_available':cuda,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
              'device_count':torch.cuda.device_count() if cuda else 0}
    if cuda:
        require(result['device_count']==1, 'Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0')
        properties = torch.cuda.get_device_properties(0)
        result.update(gpu_name=properties.name,gpu_memory_bytes=properties.total_memory,
                      compute_capability=[properties.major,properties.minor],compiled_architectures=torch.cuda.get_arch_list())
        driver = subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,capture_output=True)
        require(driver.returncode==0,'NVIDIA driver query failed')
        result['driver_version'] = sorted(set(driver.stdout.strip().splitlines()))
    return result


def require_cuda_host(fingerprint):
    require(fingerprint['platform']=='Linux' and fingerprint['machine']=='x86_64', 'Benchmark execution requires a Linux x86_64 host')
    require(fingerprint['cuda_available'] and fingerprint['device_count']==1,'Exactly one visible CUDA GPU is required')
    require(fingerprint['torch']==fingerprint['torchaudio']=='2.10.0+cu128','Match the measured Kaggle Torch and TorchAudio2.10.0+cu128')
    require(fingerprint['torch_cuda']=='12.8','This separately registered CUDA benchmark pins the measured cu128 environment')
    major,minor = fingerprint['compute_capability']
    exact = 'sm_'+str(major)+str(minor)
    require(exact in fingerprint['compiled_architectures'],'Installed wheel lacks the exact GPU architecture; choose a supported wheel before benchmarking')
    require(fingerprint['gpu_memory_bytes']>=14*2**30,'Less than the planned 16 GB GPU class; reduce scope before execution')


def load_model(api, torch, bundle, assets):
    """Construct only on the CUDA host; transfer weights, never the MPS state."""
    parent_path = bundle/'assets/parent-native-e5.pt'
    require(sha(parent_path)==assets['parent_checkpoint_sha256'],'Native parent bytes changed')
    parent = torch.load(parent_path,map_location='cpu',weights_only=True,mmap=True)
    require(parent['model_type']=='native_atst_interval_model' and parent['checkpoint_kind']=='inference'
            and parent['epoch']==5 and len(parent['model_state'])==172,'Expected the established E5 native inference parent')
    old_sources = parent['training']['source_sha256']
    current_sources = api.source_hashes()
    require(all(old_sources.get(k)==v for k,v in current_sources.items() if k not in {'atst_intervals.py','optimization.py'}),
            'Shared native forward/data/loss source changed')
    native = api.FrozenATST(bundle/'source/PretrainedSED',bundle/'assets/ATST-F_strong_1.pt',device='cpu')
    bridge = native_runtime_transition(parent['training']['native_provenance'],native.provenance)
    model = api.NativeIntervalATST.__new__(api.NativeIntervalATST)
    torch.nn.Module.__init__(model)
    model._assemble(native,parent['model_config']['head_config'],12)
    model.load_state_dict(parent['model_state'],strict=True)
    require(all(torch.equal(v,parent['model_state'][k]) for k,v in model.state_dict().items()),'Initial model tensors changed during reconstruction')
    bridge.update(parent_checkpoint_sha256=assets['parent_checkpoint_sha256'],parent_epoch=5,
                  parent_initial_feature_cache_id=parent['initial_feature_cache_id'],new_native_feature_cache_id=model.initial_feature_cache_id,
                  exact_172_initial_weight_tensors=True,fresh_optimizer=True,fresh_rng=True,
                  source_sha256=current_sources,native_provenance=native.provenance)
    model.head_checkpoint = {'epoch':5,'decoder':parent['decoder'],'validation':parent['validation']}
    del parent,native
    return model.set_device(torch.device('cuda:0'),torch.device('cpu')),bridge


def cpu_tree(value, torch):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu()
    if isinstance(value,dict):
        return {k:cpu_tree(v,torch) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_tree(v,torch) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(v,torch) for v in value)
    return value


def check_resume(saved, signature, fingerprint):
    require(saved.get('kind')=='train_only_cuda_benchmark_v1','Only a checkpoint from this separate CUDA benchmark may resume')
    require(saved['signature']==signature and saved['runtime']==fingerprint,'Benchmark data/code/runtime or GPU fingerprint changed')
    require('cuda' in saved['rng_state'] and 'mps' not in saved['rng_state'],'CUDA resume needs CUDA RNG, never an MPS substitute')
    require(type(saved['completed_updates']) is int and saved['completed_updates']>0,'Invalid benchmark update position')


def portable_case(bundle, relative):
    manifest = (bundle/relative).resolve()
    require(bundle.resolve() in manifest.parents,'Manifest escapes bundle')
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for row in records:
        audio = (manifest.parent/row['audio_path']).resolve()
        require(bundle.resolve() in audio.parents and audio.is_file(),'Audio path escapes bundle or is missing')
        require(row['quality']=='verified_timestamps' and row['events'] is not None,'Benchmark requires clean TRAIN labels')
        require((row['frame_hop'],row['frame_offset'],row['feature_dim'])==(.04,.02,768),'Native geometry changed')
        row['audio_path'] = str(audio)
    return records


def verify_assets(bundle, assets, content=True):
    for entry in assets['assets']:
        path = (bundle/entry['path']).resolve()
        require(bundle.resolve() in path.parents and path.is_file() and path.stat().st_size==entry['bytes'],'Bundle asset path/size differs')
        if content:
            require(sha(path)==entry['sha256'],'Bundle content changed: '+entry['path'])
