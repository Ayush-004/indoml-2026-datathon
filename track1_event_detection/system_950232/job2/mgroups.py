"""Master groups (start-aligned cuts of one noise file) and member offsets."""
import os, sys, pickle, numpy as np
import lib
from scipy.signal import correlate
MASTERS = os.environ.get('T1_MASTERS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                    'data_prep', 'libsyn', 'masters.pkl'))   # m01_masters.py
EX = lib.exemplars(); LIB = lib.library(); DUR = np.array([t['dur'] for t in EX]); tau = lib.tau()['tau_syn']
M = pickle.load(open(MASTERS, 'rb'))
GROUPS = {m: mem for m, mem in M['groups'].items() if m != 25}      # 25/161 is not a start-aligned cut pair
group_of = {k: m for m, mem in GROUPS.items() for k in mem}
def locate(short, long_):
    s = np.asarray(short, float)[32:-32]; l = np.asarray(long_, float); L = len(s)
    cs = np.concatenate([[0.0], np.cumsum(l * l)]); we = np.maximum(cs[L:] - cs[:-L], 1e-18)
    cc = correlate(l, s, mode='valid', method='fft') / (np.sqrt(we) * np.linalg.norm(s))
    i = int(np.argmax(np.abs(cc))); return i - 32
MU = {k: (0 if k == m else locate(LIB[k]['seg'], LIB[m]['seg'])) for m, mem in GROUPS.items() for k in mem}
def reference(master, clip=None, hidden=()):
    """(reference array, its start offset vs the canonical master, censored flag); never the
    residual of `clip` itself."""
    mem = [k for k in GROUPS[master] if k not in hidden]
    for k in sorted(mem, key=lambda k: -DUR[k]):
        for c, w in EX[k]['exs']:
            if c != clip: return np.asarray(w, np.float64), MU[k], k != master
    return None
