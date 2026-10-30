"""Build an int16 memmap store of the synthetic clips of a split (in T1_STORE)."""
import io, json, os, sys, zipfile
from pathlib import Path
import numpy as np, soundfile as sf
import lib
SCRATCH=Path(os.environ.get('T1_STORE', 'store'))/'synthstore'
SCRATCH.mkdir(parents=True, exist_ok=True)
def build(split):
    idx_path=SCRATCH/f'{split}_index.json'; dat_path=SCRATCH/f'{split}.i16'
    if idx_path.exists() and dat_path.exists(): return json.loads(idx_path.read_text()), dat_path
    if split=='val':
        recs=[r for r in lib.val_meta() if r['syntheticData']]
        names=[('validation/syntheticNoiseAudio/'+r['segmentFileName'], r['segmentFileName']) for r in recs]
        zp=lib.VAL_ZIP
    else:
        clips=set(lib.test_synthetic_clips()); zp=lib.TEST_ZIP
        with zipfile.ZipFile(zp) as z:
            names=[(n,Path(n).name) for n in sorted(z.namelist())
                   if n.endswith('.wav') and Path(n).name in clips and '__MACOSX' not in n]
    index={}; off=0
    with zipfile.ZipFile(zp) as z, open(dat_path,'wb') as out:
        for member,clip in names:
            x,sr=sf.read(io.BytesIO(z.read(member)),dtype='int16')
            assert sr==16000 and x.ndim==1, (clip,sr,x.shape)
            out.write(x.tobytes()); index[clip]=[off,len(x)]; off+=len(x)
    idx_path.write_text(json.dumps(index))
    return index, dat_path
def load(split):
    index,dat=build(split)
    mm=np.memmap(dat,dtype=np.int16,mode='r')
    return index, mm
def get(index, mm, clip):
    o,n=index[clip]; return np.asarray(mm[o:o+n],dtype=np.float32)/32768.0
if __name__=='__main__':
    for s in sys.argv[1:]:
        i,m=load(s); print(s, len(i), 'clips,', m.nbytes/1e6, 'MB')
