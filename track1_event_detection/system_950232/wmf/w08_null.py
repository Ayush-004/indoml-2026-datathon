"""The null: tile-GLRT z (S3) on natural recordings, and the label-free per-template thresholds.

Null sets (the only sets used to set a threshold): the 1,806 natural validation clips and a
seeded sample of 4,000 natural clips of the training release (wlib.train_null_sample).
Not used for any threshold: synthetic clips (validation or test), test natural clips.

Real copies. The library noise files were cut out of labelled events of natural Vaani
recordings, so a natural clip can hold the template waveform itself (for example template 200
[lip smacking] in the natural validation clip ..._Etah_..._0723_498_7687 at 4.988 s, where the
clip has a [lip smacking] label 4.987-5.433 s). A candidate that overlaps a labelled natural
event of the SAME tag, or whose extent equals a labelled event within 30 ms, is a real copy,
not a chance match, and is removed from the chance null when its S3 >= COPY_MIN = 12 (above
the largest S3 at any unlabelled place of the natural validation clips, 11.0). Candidates below
12 always stay in the chance null. Clips without timestamps (Bronze) keep every candidate
(conservative: a real copy there can only raise a threshold).

Threshold of template k:  theta_k = max(THETA_G, chance_max_k + MARGIN), chance_max_k = the
largest S3 of k on all null clips after the real-copy exclusion.

    python w08_null.py
Output: cache/null.pkl (per store: clips x templates chance maxima, copies), theta.json"""
import json

import numpy as np

import wlib
from w02_scan import FIELDS, load_scan

SR = wlib.SR
F = {f: i for i, f in enumerate(FIELDS)}
EX = wlib.exemplars()
K = len(EX)
DUR = np.array([t['dur'] for t in EX])
THETA_G = 15.0
MARGIN = 5.0
COPY_MIN = 12.0
STATS = ('S0', 'S1', 'S3', 'S4', 'S5', 'S3b', 'S5b')


def labels_valnat():
    ref = wlib.val_reference()
    tags = wlib.val_tags()
    return {c: [(a, b, t) for (a, b), t in zip(ref[c], tags[c])] for c in wlib.val_nat_clips()}, set()


def labels_train():
    sample = wlib.train_null_sample()
    ids = {cid for cid, ap, q, d in sample}
    lab = {}
    nots = set()
    with open(wlib.TRAIN_DIR / 'manifest.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r['clip_id'] not in ids:
                continue
            if r['events'] is None:
                nots.add(r['clip_id'])
                lab[r['clip_id']] = []
            else:
                lab[r['clip_id']] = [(float(e['onset']), float(e['offset']), e['tag']) for e in r['events']]
    return lab, nots


def is_copy(k, on, off, labs):
    for a, b, t in labs:
        if t == EX[k]['tag'] and a < off and b > on:
            return True
        if abs(a - on) <= 0.03 and abs(b - off) <= 0.03:
            return True
    return False


def null_of(store, labels):
    scan = load_scan(store)
    clips = sorted(scan)
    M = {s: np.full((len(clips), K), -np.inf, np.float32) for s in STATS}
    copies = []
    hours = 0.0
    idx = wlib.store_index(store)
    for ci, c in enumerate(clips):
        hours += idx[c][1] / SR / 3600
        R = scan[c]
        if not len(R):
            continue
        labs = labels.get(c, [])
        keep = np.ones(len(R), bool)
        for j, r in enumerate(R):
            if r[F['S3']] < COPY_MIN:
                continue
            k = int(r[F['k']])
            on = r[F['p']] / SR
            if is_copy(k, on, on + DUR[k], labs):
                keep[j] = False
                copies.append((c, k, round(float(on), 3), float(r[F['S0']]), float(r[F['S3']]), float(r[F['S5']])))
        R = R[keep]
        k = R[:, F['k']].astype(int)
        for s in STATS:
            np.maximum.at(M[s][ci], k, R[:, F[s]])
    return {'clips': clips, 'M': M, 'copies': copies, 'hours': hours}


def main():
    out = {}
    for store, fn in (('valnat', labels_valnat), ('train', labels_train)):
        labels, nots = fn()
        out[store] = null_of(store, labels)
        out[store]['no_timestamps'] = len(nots)
        M = out[store]['M']['S3']
        v = M[np.isfinite(M)]
        print('%s: %d clips %.2f h ; real copies removed %d ; chance S3 clip x template max: q99 %.2f q999 %.2f max %.2f' % (
            store, len(out[store]['clips']), out[store]['hours'], len(out[store]['copies']),
            np.quantile(v, .99), np.quantile(v, .999), v.max()))
    allM = np.concatenate([out[s]['M']['S3'] for s in ('valnat', 'train')])
    chance_max = np.max(np.where(np.isfinite(allM), allM, -np.inf), axis=0)
    theta = np.maximum(THETA_G, chance_max + MARGIN)
    hours = sum(out[s]['hours'] for s in ('valnat', 'train'))
    order = np.argsort(-chance_max)[:15]
    print('null hours %.2f ; templates whose theta is above THETA_G: %d' % (hours, int(np.sum(theta > THETA_G))))
    for k in order:
        print('   k %3d %-18s chance max %.2f theta %.2f' % (k, EX[k]['tag'], chance_max[k], theta[k]))
    for th in (6, 8, 10, 12, 15, 20):
        n = int(np.sum(allM >= th))
        print('chance clip x template maxima >= %4.1f: %d (%.3f per null hour)' % (th, n, n / hours))
    wlib.dump_pickle(wlib.CACHE / 'null.pkl', out)
    wlib.dump_json(wlib.HERE / 'theta.json', {'THETA_G': THETA_G, 'MARGIN': MARGIN, 'null_hours': hours,
                                              'chance_max': chance_max.tolist(), 'theta': theta.tolist()})


if __name__ == '__main__':
    main()
