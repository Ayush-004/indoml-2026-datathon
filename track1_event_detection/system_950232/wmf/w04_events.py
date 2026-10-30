"""Validation event table (labels are used to SCORE and to learn generator facts only).

Per labelled synthetic event: clip, onset, offset, tag, template k (exact noise track mapping
of data_prep/libsyn/g01_val_tracks.py), paste start, paste level (dBFS of the exact residual), fold,
whether a leave-one-clip-out exemplar exists, master / vocal flags, base cross-fit status.
Also the base cross-fit decode with a TP/FP flag per accepted hit.
Output: cache/events.pkl"""
import os

import numpy as np

import wlib

SR = wlib.SR
MASTER = {'machine': list(range(103, 126)), 'child': list(range(84, 89)), 'bird': [64, 65, 66]}
MASTER_OF = {k: m for m, ks in MASTER.items() for k in ks}
MASTER_LEN = {'machine': 5.437, 'child': 5.211, 'bird': 9.077}


def is_vocal(k):
    return wlib.exemplars()[k]['tag'].startswith('[')


def match_decode(ref_events, dec):
    """organizer matching of a decode [(on, off, k, ncc, p)] against labels -> (tp_flags, ref_matched)."""
    cands = []
    for ri, (a, b) in enumerate(ref_events):
        tol = max(0.2 * (b - a), 0.05)
        for pi, d in enumerate(dec):
            if abs(d[0] - a) <= tol and abs(d[1] - b) <= tol:
                cands.append((abs(d[0] - a) + abs(d[1] - b), ri, pi))
    mr, mp = {}, {}
    for _, ri, pi in sorted(cands):
        if ri not in mr and pi not in mp:
            mr[ri] = pi
            mp[pi] = ri
    return mp, mr


def main():
    B = wlib.load_pickle(wlib.CACHE / 'base.pkl')
    G = wlib.load_pickle(os.environ.get('T1_G01', wlib.HERE.parents[1] / 'data_prep/libsyn/g01_val_tracks.pkl'))
    ref = wlib.val_reference()
    tags = wlib.val_tags()
    vd = wlib.val_durations()
    fold = B['fold']
    evk = {(e['clip'], round(e['on'], 3)): e for e in G['events']}
    rows = []
    dec_rows = []
    for c in wlib.val_syn_clips():
        dec = [(a, min(b, vd[c]), k, v, p) for a, b, k, v, p in B['base_cf'][c]]
        mp, mr = match_decode(ref[c], dec)
        for ri, ((a, b), tag) in enumerate(zip(ref[c], tags[c])):
            e = evk[(c, round(a, 3))]
            k = e['k']
            rows.append({'clip': c, 'i': ri, 'on': a, 'off': b, 'tag': tag, 'k': k, 'paste_start': e['paste_start'],
                         'level': float(10 * np.log10(max(e['p_noise'], 1e-20))), 'g01_ncc': e['ncc'],
                         'fold': fold[c], 'loo': wlib.kernel_index(k, c) is not None,
                         'master': MASTER_OF.get(k), 'vocal': tag.startswith('['),
                         'base_tp': ri in mr, 'dur_clip': vd[c]})
        for pi, d in enumerate(dec):
            dec_rows.append({'clip': c, 'on': d[0], 'off': d[1], 'k': d[2], 'ncc': d[3], 'p': d[4], 'tp': pi in mp,
                             'fold': fold[c]})
    n = len(rows)
    print('events', n, 'base TP', sum(r['base_tp'] for r in rows), 'no LOO kernel', sum(not r['loo'] for r in rows),
          '; decode', len(dec_rows), 'FP', sum(not d['tp'] for d in dec_rows))
    wlib.dump_pickle(wlib.CACHE / 'events.pkl', {'events': rows, 'dec': dec_rows})


if __name__ == '__main__':
    main()
