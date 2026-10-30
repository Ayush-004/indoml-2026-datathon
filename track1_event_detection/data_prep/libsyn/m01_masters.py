"""16 kHz sub-segment relations between all library templates (the 4x-decimated
family scan missed lags of a few samples). Output: masters.pkl next to this file
{'pairs': [(short, long, ncc, pos_s)], 'master_of': {k: (master, pos_s, ncc)},
 'groups': {master: [members sorted by dur]}}"""
import sys, pickle, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'system_950232' / 'job2'))
import lib   # library() = INDOML_WORK/templates/library.pkl
from scipy.signal import correlate
LIB = lib.library(); N = len(LIB); SR = 16000
seg = [np.asarray(t['seg'], np.float64) for t in LIB]
dur = np.array([t['dur'] for t in LIB])

TRIM = 32   # 2 ms: cuts of one master start up to ~12 samples apart
def sub_ncc(s, l):
    s = s[TRIM:-TRIM] if len(s) > 4 * TRIM else s
    L = len(s)
    if len(l) < L: return 0.0, 0
    cs = np.concatenate([[0.0], np.cumsum(l * l)])
    we = np.maximum(cs[L:] - cs[:-L], 1e-18)
    cc = correlate(l, s, mode='valid', method='fft') / (np.sqrt(we) * np.linalg.norm(s))
    i = int(np.argmax(np.abs(cc)))
    return float(cc[i]), i - TRIM

pairs = []
order = np.argsort(dur)
for a in range(N):
    for b in range(N):
        if a == b or len(seg[a]) > len(seg[b]) or (len(seg[a]) == len(seg[b]) and a > b): continue
        v, i = sub_ncc(seg[a], seg[b])
        if abs(v) >= 0.90: pairs.append((a, b, v, i / SR))
print('16 kHz sub-segment pairs |ncc|>=0.90: %d ; >=0.97: %d' % (len(pairs), sum(1 for p in pairs if abs(p[2]) >= 0.97)))
# master of k = the LONGEST template that contains k with |ncc|>=0.97
master_of = {}
for a, b, v, pos in pairs:
    if abs(v) < 0.97: continue
    if a not in master_of or dur[b] > dur[master_of[a][0]]: master_of[a] = (b, pos, v)
# resolve chains to the top master
def top(k):
    seen = 0
    while k in master_of and seen < 50: k = master_of[k][0]; seen += 1
    return k
groups = {}
for k in range(N):
    m = top(k)
    if m != k or any(master_of.get(j, (None,))[0] is not None and top(j) == k for j in range(N) if j != k):
        groups.setdefault(m, set()).add(k)
groups = {m: sorted(v | {m}, key=lambda k: dur[k]) for m, v in groups.items()}
for m, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    rows = []
    for k in mem:
        if k == m: rows.append((int(k), round(dur[k], 3), LIB[k]['n'], 'MASTER')); continue
        v, i = sub_ncc(seg[k], seg[m])
        rows.append((int(k), round(dur[k], 3), LIB[k]['n'], 'pos %.4f ncc %.3f' % (i / SR, v)))
    print('master %d (%s, %.3f s): %d members' % (m, LIB[m]['tag'], dur[m], len(mem)))
    for r in rows: print('    ', r)
pickle.dump({'pairs': pairs, 'master_of': master_of, 'groups': groups}, open(Path(__file__).resolve().parent / 'masters.pkl', 'wb'))
