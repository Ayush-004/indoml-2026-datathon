"""Validation of the new statistic in the decode: honest 5-fold geography cross-fit on the 695
validation synthetic clips (labels only score; every label-fitted piece is fitted out of fold).

Systems
  BASE        plain-NCC decode, per-template thresholds fitted on the 4 other folds (1.9584)
  BASE+J2     + master extent rule (event = [onset, min(onset + file length, clip end)]) = the
              logic of the uploaded synthetic file
  NEW(cfg)    BASE+J2 then REMOVE (theta_r) -> RECHOOSE (factor) -> ADD (theta_k from the natural
              null, level range of k out of fold +- W dB), see w06_rules.py
Selection: the a-priori configuration (ADD only, W = 6 dB, theta_k from theta.json) and a
nested choice over a small grid (per held-out fold, the configuration that maximises the combined
score on the other 4 folds).

    python w07_val.py
Output: val_results.json, cache/val_preds.pkl"""
import itertools
import json

import numpy as np

import wlib
import w06_rules as R
from w02_scan import load_scan

SR = wlib.SR


def j2(events, clip_dur):
    out = []
    for on, off, k, src in events:
        m = R.MASTER_OF.get(k)
        if m is not None:
            off = min(on + R.MASTER_LEN[m], clip_dur)
        out.append((on, off, k, src))
    return sorted(out)


def build(clips, base, cands_of, s3_at, clip_dur, theta, ranges_of, cfg):
    """cfg: dict(W, theta_r, factor, masters). base: {clip: [(on, off, k, src)]} (after J2).
    s3_at: {clip: [S3 at the lag of each base event]} ; ranges_of(clip) -> level ranges."""
    pred = {}
    log = {'added': [], 'removed': [], 'rechosen': []}
    for c in clips:
        ev = list(base[c])
        s3 = list(s3_at.get(c, [None] * len(ev)))
        rg = ranges_of(c)
        if cfg.get('theta_r') is not None:
            keep = []
            ks3 = []
            for e, s in zip(ev, s3):
                if s is not None and np.isfinite(s) and s < cfg['theta_r'] and R.MASTER_OF.get(e[2]) is None:
                    log['removed'].append((c, e, s))
                else:
                    keep.append(e)
                    ks3.append(s)
            ev, s3 = keep, ks3
        if cfg.get('factor') is not None:
            order = np.argsort([e[0] for e in ev])
            ev = [ev[i] for i in order]
            s3 = [s3[i] for i in order]
            ev2, ch = R.apply_rechoose(ev, s3, cands_of(c), clip_dur[c], cfg['factor'], rg, cfg.get('W'))
            for old, new, so, sn in ch:
                log['rechosen'].append((c, old, new, so, sn))
            ev = ev2
        ev, added = R.apply_add(ev, cands_of(c), clip_dur[c], theta, rg, cfg.get('W'), cfg.get('masters', True))
        for a in added:
            log['added'].append((c, a))
        pred[c] = sorted(ev)
    return pred, log


def as_pairs(pred):
    return {c: [(a, b) for a, b, *_ in v] for c, v in pred.items()}


def match_flags(ref_events, pred_events):
    cands = []
    for ri, (a, b) in enumerate(ref_events):
        tol = max(0.2 * (b - a), 0.05)
        for pi, (p, q) in enumerate(pred_events):
            if abs(p - a) <= tol and abs(q - b) <= tol:
                cands.append((abs(p - a) + abs(q - b), ri, pi))
    mr, mp = set(), set()
    for _, ri, pi in sorted(cands):
        if ri not in mr and pi not in mp:
            mr.add(ri)
            mp.add(pi)
    return mp


def change_report(ref, before, after, log):
    """added events true/false, removed events fixed (was FP) / broken (was TP)."""
    out = {'added_true': 0, 'added_false': 0, 'removed_fp': 0, 'removed_tp': 0, 'rechosen_fixed': 0,
           'rechosen_broken': 0, 'rechosen_neutral': 0, 'added_rows': [], 'removed_rows': [], 'rechosen_rows': []}
    for c, a in log['added']:
        pe = [(x, y) for x, y, *_ in after[c]]
        mp = match_flags(ref[c], pe)
        i = pe.index((a['on'], a['off'])) if (a['on'], a['off']) in pe else None
        ok = i in mp
        out['added_true' if ok else 'added_false'] += 1
        out['added_rows'].append({'clip': c, 'k': a['k'], 'on': round(a['on'], 4), 'off': round(a['off'], 4),
                                  'S3': round(a['S3'], 2), 'S0': round(a['S0'], 3), 'level': round(a['level'], 1),
                                  'true': bool(ok)})
    for c, e, s in log['removed']:
        pb = [(x, y) for x, y, *_ in before[c]]
        mp = match_flags(ref[c], pb)
        i = pb.index((e[0], e[1]))
        was_tp = i in mp
        out['removed_tp' if was_tp else 'removed_fp'] += 1
        out['removed_rows'].append({'clip': c, 'k': e[2], 'on': round(e[0], 4), 'off': round(e[1], 4),
                                    'S3': None if s is None else round(float(s), 2), 'was_tp': bool(was_tp)})
    for c, old, new, so, sn in log['rechosen']:
        pb = [(x, y) for x, y, *_ in before[c]]
        pa = [(x, y) for x, y, *_ in after[c]]
        tb = pb.index((old[0], old[1])) in match_flags(ref[c], pb)
        ta = (new[0], new[1]) in pa and pa.index((new[0], new[1])) in match_flags(ref[c], pa)
        key = 'rechosen_fixed' if (ta and not tb) else ('rechosen_broken' if (tb and not ta) else 'rechosen_neutral')
        out[key] += 1
        out['rechosen_rows'].append({'clip': c, 'old_k': old[2], 'new_k': new[2], 'old': [round(old[0], 4), round(old[1], 4)],
                                     'new': [round(new[0], 4), round(new[1], 4)], 'S3_old': so, 'S3_new': sn,
                                     'class': key})
    return out


def main():
    B = wlib.load_pickle(wlib.CACHE / 'base.pkl')
    E = wlib.load_pickle(wlib.CACHE / 'events.pkl')
    A = wlib.load_pickle(wlib.CACHE / 'atlag_valsyn_base.pkl')
    T = json.loads((wlib.HERE / 'theta.json').read_text())
    theta = np.array(T['theta'])
    ref = wlib.val_reference()
    vd = wlib.val_durations()
    vsyn = wlib.val_syn_clips()
    fold = B['fold']
    scan = load_scan('valsyn')
    cand_cache = {}

    def cands_of(c):
        if c not in cand_cache:
            cand_cache[c] = R.candidates(scan.get(c), c, True, min_s3=5.0)
        return cand_cache[c]

    base = {c: [(a, min(b, vd[c]), k, 'base') for a, b, k, v, p in B['base_cf'][c]] for c in vsyn}
    base_j2 = {c: j2(base[c], vd[c]) for c in vsyn}
    # S3 at the lag of each base event (w03), aligned with base_j2 order (same onsets)
    s3_at = {}
    for c in vsyn:
        rows = A.get(c, [])
        byk = {}
        for r in rows:
            byk[(int(r[0]), int(r[1]))] = r[5]
        s3_at[c] = []
        for on, off, k, src in base_j2[c]:
            hit = [h for h in B['base_cf'][c] if abs(h[0] - on) < 1e-9 and h[2] == k]
            s3_at[c].append(float(byk.get((k, hit[0][4]), np.nan)) if hit else None)
    ranges = {f: R.level_ranges(E['events'], exclude_fold=f) for f in range(5)}

    def ranges_of(c):
        return ranges[fold[c]]

    res = {'systems': {}}
    s_base = wlib.score(ref, as_pairs(base), vsyn)
    s_j2 = wlib.score(ref, as_pairs(base_j2), vsyn)
    res['systems']['BASE'] = wlib.fmt(s_base)
    res['systems']['BASE+J2'] = wlib.fmt(s_j2)
    print('BASE    ', wlib.fmt(s_base))
    print('BASE+J2 ', wlib.fmt(s_j2))
    grid = []
    for W, tr, fac, ms in itertools.product((None, 6.0, 3.0), (None, 3.0, 6.0, 10.0), (None, 1.5, 3.0), (True, False)):
        grid.append({'W': W, 'theta_r': tr, 'factor': fac, 'masters': ms})
    apriori = {'W': 6.0, 'theta_r': None, 'factor': None, 'masters': True}
    preds = {}
    scores = {}
    for i, cfg in enumerate(grid):
        pred, log = build(vsyn, base_j2, cands_of, s3_at, vd, theta, ranges_of, cfg)
        preds[i] = (pred, log)
        scores[i] = wlib.score(ref, as_pairs(pred), vsyn)
    for i, cfg in enumerate(grid):
        s = scores[i]
        print('%-62s F1 %.4f Dice %.4f comb %.4f TP %d FP %d FN %d' % (
            json.dumps(cfg), s['f1'], s['dice'], s['combined'], s['tp'], s['fp'], s['fn']))
    ia = grid.index(apriori)
    pa, la = preds[ia]
    rep = change_report(ref, base_j2, pa, la)
    res['systems']['NEW_apriori'] = dict(wlib.fmt(scores[ia]), cfg=apriori,
                                         changes={k: v for k, v in rep.items() if not k.endswith('_rows')})
    res['apriori_changes'] = rep
    print('NEW a priori', wlib.fmt(scores[ia]), {k: v for k, v in rep.items() if not k.endswith('_rows')})
    # nested selection
    nested = {}
    chosen = {}
    for f in range(5):
        tr_clips = [c for c in vsyn if fold[c] != f]
        best = None
        for i in range(len(grid)):
            s = wlib.score(ref, as_pairs({c: preds[i][0][c] for c in tr_clips}), tr_clips)['combined']
            if best is None or s > best[0] + 1e-12:
                best = (s, i)
        chosen[f] = grid[best[1]]
        for c in vsyn:
            if fold[c] == f:
                nested[c] = preds[best[1]][0][c]
    s_n = wlib.score(ref, as_pairs(nested), vsyn)
    nlog = {'added': [], 'removed': [], 'rechosen': []}
    for f in range(5):
        i = grid.index(chosen[f])
        lg = preds[i][1]
        for key in nlog:
            nlog[key] += [x for x in lg[key] if fold[x[0]] == f]
    rep_n = change_report(ref, base_j2, nested, nlog)
    res['systems']['NEW_nested'] = dict(wlib.fmt(s_n), chosen={str(f): chosen[f] for f in range(5)},
                                        changes={k: v for k, v in rep_n.items() if not k.endswith('_rows')})
    res['nested_changes'] = rep_n
    print('NEW nested ', wlib.fmt(s_n), {k: v for k, v in rep_n.items() if not k.endswith('_rows')})
    print('chosen per fold', chosen)
    res['grid'] = [dict(cfg, **wlib.fmt(scores[i])) for i, cfg in enumerate(grid)]
    wlib.dump_json(wlib.HERE / 'val_results.json', res)
    wlib.dump_pickle(wlib.CACHE / 'val_preds.pkl', {'base': base, 'base_j2': base_j2, 'apriori': pa, 'nested': nested,
                                                     's3_at': s3_at})


if __name__ == '__main__':
    main()
