"""Scan the first P seconds of each master (machine, child, bird) over every clip of four
sets: test synthetic, validation synthetic, validation natural (1,806), test natural (3,435).
A paste of a master always starts at the master start, so its prefix finds every paste,
also the truncated ones that no library cut fits into. Natural clips hold no pasted noise:
their peaks calibrate the threshold. Output: m05_prefix_scan.pkl
{set: {clip: {master: [(pos_samples, ncc)]}}}, top 3 peaks per clip, >= 1 s apart."""
import os, sys, pickle, time, numpy as np, soundfile as sf
from pathlib import Path
from scipy.signal import correlate, oaconvolve
import lib, audio_store, verify as VF
from mgroups import LIB, GROUPS
SR = 16000; P = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
HERE = os.path.join(os.environ.get('T1_JOB2_OUT', '.'), '')   # where the m0x outputs are written and read
L = int(P * SR)
KER = {}
for m in GROUPS:
    k = np.asarray(LIB[m]['seg'], np.float64)[:L]; k = k - k.mean(); KER[m] = k / np.linalg.norm(k)

def peaks(x):
    y = (x - x.mean()).astype(np.float64)
    if len(y) <= L: return {m: [] for m in KER}
    e = oaconvolve(y * y, np.ones(L), mode='valid')
    out = {}
    for m, kn in KER.items():
        cc = np.abs(correlate(y, kn, mode='valid', method='fft'))
        n = min(len(cc), len(e))
        good = e[:n] > (VF.RMS_FLOOR ** 2) * L
        sc = np.where(good, cc[:n] / np.sqrt(np.maximum(e[:n], 1e-12)), 0.0)
        res = []
        for _ in range(3):
            j = int(np.argmax(sc))
            if sc[j] <= 0.05: break
            res.append((j, VF.exact_ncc(x, kn, j)))
            sc[max(0, j - SR):j + SR] = 0.0
        out[m] = res
    return out

res = {}
t0 = time.time()
for split in ('test', 'val'):
    idx, mm = audio_store.load(split)
    clips = lib.test_synthetic_clips() if split == 'test' else sorted(c for c in lib.val_reference() if lib.val_is_synthetic()[c])
    R = {}
    for c in clips: R[c] = peaks(audio_store.get(idx, mm, c).astype(np.float64))
    res[split + '_syn'] = R
    print('%s_syn: %d clips (%.0fs)' % (split, len(R), time.time() - t0), flush=True)
for split in ('val', 'test'):
    files = sorted((lib.WORK / 'audio' / split).glob('*.wav'))
    R = {}
    for n, f in enumerate(files):
        x, sr = sf.read(f, dtype='float64')
        if x.ndim > 1: x = x[:, 0]
        R[f.name] = peaks(x)
        if n % 1000 == 0: print('  %s_nat %d/%d (%.0fs)' % (split, n, len(files), time.time() - t0), flush=True)
    res[split + '_nat'] = R
    print('%s_nat: %d clips (%.0fs)' % (split, len(R), time.time() - t0), flush=True)
pickle.dump({'P': P, 'res': res}, open(HERE + 'm05_prefix_scan.pkl', 'wb'))
for s, R in res.items():
    for m in KER:
        v = np.array([max([p[1] for p in d[m]], default=0.0) for d in R.values()])
        print('%-8s master %3d: clip max NCC quantiles %s' % (s, m, np.quantile(v, [0.5, 0.9, 0.99, 0.999, 1.0]).round(3)))
