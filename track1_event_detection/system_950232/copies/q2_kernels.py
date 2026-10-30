"""Kernel sets for Q2 (natural cuts of labelled VALIDATION natural events, at the label extent).

src71  the 71 labelled natural events that hold a copy of a library template (q1_val.json; the
       template-level detections of the WMF scan_valnat with S3 >= theta_k). For each kernel the
       templates it holds and the template start inside the kernel (samples) are stored, so that the
       true lag of every validation synthetic paste of those templates is known (positives).
cand   every other labelled event of the 1,806 natural validation clips with duration >= 0.1 s
       (the shortest library template is 0.125 s), one kernel per distinct (clip, extent);
       events that overlap one of the 71 source events are flagged ('near_source').
Labels of validation natural clips are allowed here (the library is built from validation data).
Output: cache/kernels_src71.json, cache/kernels_cand.json"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

import nclib as N  # noqa: E402

SR = N.SR
MIN_DUR = 0.1


def main():
    EX = N.exemplars()
    th = N.theta()
    scan = N.load_wmf_scan('valnat')
    labels = N.val_labels()
    vd = N.val_durations()
    q1 = json.loads((N.HERE / 'q1_val.json').read_text())
    src = {}
    for r in q1['copies']:
        key = (r['clip'] + '.wav', r['label_i'])
        src[key] = r
    # template-level detections -> source event
    tmpl = {}
    for c in N.val_nat_clips():
        R = scan.get(c)
        if R is None or not len(R):
            continue
        k = R[:, N.F['k']].astype(int)
        for r in R[R[:, N.F['S3']] >= th[k]]:
            kk = int(r[N.F['k']])
            p = int(r[N.F['p']])
            on, off = N.template_extent(kk, p / SR, vd[c])
            best, bi = 0.0, None
            for i, (a, b, t) in enumerate(labels[c]):
                v = N.iou((on, off), (a, b))
                if v > best:
                    best, bi = v, i
            assert (c, bi) in src, (c, kk)
            tmpl.setdefault((c, bi), {})[kk] = p
    specs = []
    for (c, i), r in sorted(src.items()):
        a, b, t = labels[c][i]
        a_s, b_s = int(round(a * SR)), int(round(b * SR))
        specs.append({'clip': c, 'label_i': i, 'on': a, 'off': b, 'a': a_s, 'b': b_s, 'tag': t,
                      'vocal': t.startswith('['), 'dur': round(b - a, 4),
                      'templates': {str(k): p - a_s for k, p in sorted(tmpl[(c, i)].items())},
                      'S0_in_natural': r['S0'], 'S3_in_natural': r['S3']})
    N.dump_json(N.CACHE / 'kernels_src71.json', specs)
    print('src71 kernels', len(specs), 'templates', sum(len(s['templates']) for s in specs))

    cand = []
    seen = set()
    n_all = n_short = n_dup = n_src = 0
    for c in N.val_nat_clips():
        srcs = [(labels[c][i][0], labels[c][i][1]) for (cc, i) in src if cc == c]
        for i, (a, b, t) in enumerate(labels[c]):
            n_all += 1
            if (c, i) in src:
                n_src += 1
                continue
            if b - a < MIN_DUR:
                n_short += 1
                continue
            a_s, b_s = int(round(a * SR)), int(round(b * SR))
            if (c, a_s, b_s) in seen:
                n_dup += 1
                continue
            seen.add((c, a_s, b_s))
            cand.append({'clip': c, 'label_i': i, 'on': a, 'off': b, 'a': a_s, 'b': min(b_s, int(round(vd[c] * SR))),
                         'tag': t, 'vocal': t.startswith('['), 'dur': round(b - a, 4),
                         'near_source': any(N.overlaps((a, b), s) for s in srcs)})
    N.dump_json(N.CACHE / 'kernels_cand.json', cand)
    d = np.array([s['dur'] for s in cand])
    summary = {'labelled_events': n_all, 'source_events_excluded': n_src, 'shorter_than_0.1s': n_short,
               'duplicate_extent': n_dup, 'kernels': len(cand), 'near_source': sum(s['near_source'] for s in cand),
               'vocal': sum(s['vocal'] for s in cand), 'hours': round(float(d.sum()) / 3600, 3),
               'dur_quantiles': np.round(np.quantile(d, [0, .1, .25, .5, .75, .9, 1]), 3).tolist(),
               'tags_top': Counter(s['tag'] for s in cand).most_common(25)}
    print(json.dumps(summary))
    N.dump_json(N.HERE / 'q2_kernels_summary.json', summary)


if __name__ == '__main__':
    main()
