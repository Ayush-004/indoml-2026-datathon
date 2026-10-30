"""Training-release audio decoded exactly (float -> exact int16 = round(x * 32768)):
md5 of the int16 samples (exact-duplicate check vs val/test), peak (full-scale natural normalisation vs the synthetic
0.99 level 32440/32441), exactness of the int16 grid. One file at a time; label-free. Output: train_float.jsonl next
to this file (gen_libsyn.py reads the 'frames' field: the clip lengths of the training release)."""
import sys
sys.dont_write_bytecode = True
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import hashlib, json, time
from pathlib import Path
import numpy as np
import soundfile as sf
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('T1_WORK', '.')) / 'vaani-full'
out = HERE / 'train_float.jsonl'
if out.exists():
    raise SystemExit('exists')
rows = [json.loads(l) for l in open(ROOT / 'manifest.jsonl')]
t0 = time.time()
tmp = out.with_suffix('.tmp')
with open(tmp, 'w') as f:
    for i, r in enumerate(rows):
        x, sr = sf.read(str(ROOT / r['audio_path']), dtype='float64', always_2d=True)
        x = x[:, 0]
        q = np.round(x * 32768.0)
        exact = bool(np.max(np.abs(x * 32768.0 - q)) < 1e-6) if len(x) else True
        xi = np.clip(q, -32768, 32767).astype('<i2')
        f.write(json.dumps({'clip': r['clip_id'], 'quality': r['quality'], 'frames': int(len(xi)),
                            'md5': hashlib.md5(xi.tobytes()).hexdigest(), 'exact_int16': exact,
                            'peak': int(np.max(np.abs(xi.astype(np.int32)))) if len(xi) else 0}) + '\n')
        if i % 10000 == 0:
            print(i, round(time.time() - t0, 1), flush=True)
tmp.replace(out)
print('done', round(time.time() - t0, 1))
