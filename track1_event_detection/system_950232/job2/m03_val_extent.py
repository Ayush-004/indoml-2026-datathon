"""Validation of the master-based extent estimator on the validation master-group events.

Scenarios (labels are read ONLY to score, never by the estimator):
  V1   in-sample decode (the shipped decode, every cut has its own template)
  LOCO leave-one-cut-out: the event's own template is hidden, the clip is re-decoded and
       the event re-fires with another member of the group; if the hidden template is the
       master itself, the reference becomes the longest remaining member (censored).
The master reference is never the event's own residual: for the clip a master exemplar
was cut from, another exemplar (or the next-longest member) is used.
"""
import os, sys, json, pickle, numpy as np
import lib, audio_store, j2_lib as J, mext
SR = 16000
HERE = os.path.join(os.environ.get('T1_JOB2_OUT', '.'), '')   # where the m0x outputs are written and read
from mgroups import EX, LIB, DUR, tau, GROUPS, group_of, MU, reference

ref = lib.val_reference(); hv = lib.val_hits()
vsyn = sorted(c for c in ref if lib.val_is_synthetic()[c])
idx, mm = audio_store.load('val')
full = J.accepted(hv, vsyn, tau, DUR)

def decode_excl(c, excl):
    cands = sorted(((v, p / SR, p / SR + DUR[k], k) for k, p, v in hv[c] if v >= tau[k] and k not in excl), reverse=True)
    acc = []
    for v, on, off, k in cands:
        if all(off <= a or on >= b for _, a, b, _ in acc): acc.append((v, on, off, k))
    return sorted(((k, on, v) for v, on, off, k in acc), key=lambda r: r[1])

rows = []
for c in vsyn:
    hits = [(k, on, v) for k, on, v in full[c] if k in group_of]
    if not hits: continue
    x = audio_store.get(idx, mm, c).astype(np.float64)
    for k0, on0, v0 in hits:
        lab = [(a, b) for a, b in ref[c] if abs(a - on0) <= 0.05]
        if not lab: rows.append({'clip': c, 'k': int(k0), 'label': None}); continue
        a, b = lab[0]; tol = max(0.2 * (b - a), 0.05)
        g = group_of[k0]
        for scen in ('V1', 'LOCO'):
            if scen == 'V1':
                k, on, hidden = k0, on0, ()
            else:
                hidden = (k0,)
                d = [r for r in decode_excl(c, set(hidden)) if abs(r[1] - on0) <= 0.05 and r[0] in group_of]
                if not d:
                    rows.append({'clip': c, 'scen': scen, 'k0': int(k0), 'refired': None}); continue
                k, on, _ = d[0]
            r = reference(g, c, hidden)
            if r is None: continue
            w, mu_ref, cens = r
            p = int(round(on * SR))
            c0 = p - MU[k] + mu_ref          # clip sample of reference-array sample 0
            out = {}
            for wh in (False, True):
                e = mext.estimate(x, w, c0, whiten=wh)
                out['wh' if wh else 'plain'] = e
            row = {'clip': c, 'scen': scen, 'k0': int(k0), 'k': int(k), 'group': int(g), 'censored': bool(cens),
                   'ref_len': len(w) / SR, 'true_on': a, 'true_dur': b - a, 'on': on, 'tmpl_dur': float(DUR[k]), 'tol': tol,
                   'was_correct': bool(abs(on - a) <= tol and abs(on + DUR[k] - b) <= tol)}
            for nm, e in out.items():
                if e is None: row[nm] = None; continue
                on_hat = (e['c0'] - mu_ref) / SR           # where master sample 0 (=cut start) sits
                row[nm] = {'L': e['L'], 'dur_err': e['L'] - (b - a), 'off_err': (on + e['L']) - b,
                           'on_hat_err': on_hat - a, 'ncc_head': e['ncc_head'], 'g': e['g'], 'at_end': e['at_end'],
                           'prom_left': e['prom_left'], 'prom_right': e['prom_right'],
                           'correct': bool(abs(on - a) <= tol and abs(on + e['L'] - b) <= tol)}
            rows.append(row)
json.dump(rows, open(HERE + 'm03_val_extent.json', 'w'), indent=1)
for scen in ('V1', 'LOCO'):
    R = [r for r in rows if r.get('scen') == scen and r.get('plain')]
    print('%s: %d events (censored refs %d)' % (scen, len(R), sum(r['censored'] for r in R)))
    for nm in ('plain', 'wh'):
        e = np.array([r[nm]['off_err'] for r in R if r[nm]])
        corr = sum(r[nm]['correct'] for r in R if r[nm]); was = sum(r['was_correct'] for r in R)
        fix = sum(1 for r in R if r[nm] and r[nm]['correct'] and not r['was_correct'])
        brk = sum(1 for r in R if r[nm] and r['was_correct'] and not r[nm]['correct'])
        print('   %-5s offset err ms: med %+.1f  |e|<=20ms %.3f  |e|<=50ms %.3f  max|e| %.0f ; correct %d (decode %d) fixed %d broken %d' % (
            nm, 1000 * np.median(e), np.mean(np.abs(e) <= 0.02), np.mean(np.abs(e) <= 0.05), 1000 * np.abs(e).max(), corr, was, fix, brk))
    for r in R:
        if not r['plain']['correct'] or abs(r['plain']['off_err']) > 0.05:
            print('     miss:', r['clip'][-40:], 'k0', r['k0'], 'k', r['k'], 'true %.3f tmpl %.3f est %.3f wh %.3f cens %s ref %.2f' % (
                r['true_dur'], r['tmpl_dur'], r['plain']['L'], r['wh']['L'] if r['wh'] else -1, r['censored'], r['ref_len']))
