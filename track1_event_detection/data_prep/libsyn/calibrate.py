"""calibrate.py: check the calibration bands B1-B9 (see the README) on generated LIBSYN clips.

It reads the clips back from disk (FLAC), the kept beds (calib_beds/, written with --keep-beds), the manifest and
the provenance. It does not trust the generator's own numbers: levels, zeros, peaks and exactness are measured
again from the audio.

  python calibrate.py --dir libsyn/v1 --tag 000000-001999
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
LIBRARY = Path(os.environ.get('T1_LIBRARY', 'work/templates/library.pkl'))
BEDS = Path(os.environ.get('T1_BEDS', 'beds-clean'))
VAL_SHARES = np.array([110, 78, 124, 340, 43], float) / 695
VAL_META = Path(os.environ.get('T1_VAL_META', 'validationMetadata.json'))


def _np(o):
    return o.item() if hasattr(o, 'item') else str(o)


def zero_runs(x, min_len=16):
    z = (x == 0).astype(np.int8)
    d = np.diff(np.concatenate([[0], z, [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = (e - s) >= min_len
    return s[k], e[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    root = Path(a.dir)
    man = [json.loads(x) for x in open(root / f'manifest-{a.tag}.jsonl')]
    prov = {r['clip_id']: r for r in (json.loads(x) for x in open(root / f'provenance-{a.tag}.jsonl'))}
    lib = pickle.load(open(LIBRARY, 'rb'))
    idx_ids = {json.loads(x)['bed_i']: json.loads(x)['clip_id'] for x in open(BEDS / 'index.jsonl')}
    val_sessions = {r['segmentFileName'].split('_')[5] for r in json.load(open(VAL_META))}
    counts = np.zeros(5)
    legal_bad, vocal_zero_bad, vocal_lvl, noise_lvl, noise_by_t = [], 0, [], [], {}
    zero_share, empty_counts, peaks_bad, exact_bad, durs, bed_bad = [], [], 0, 0, [], 0
    for r in man:
        p = prov[r['clip_id']]
        D = r['duration']
        durs.append(D)
        ev = sorted(r['events'], key=lambda e: e['onset'])
        counts[len(ev) - 1] += 1
        # B2 legality (labels)
        for j, e in enumerate(ev):
            if e['onset'] < 0.250 - 1e-9:
                legal_bad.append((r['clip_id'], 'edge_on'))
            is_master = abs(e['offset'] - D) < 1e-6
            if not is_master and e['offset'] > D - 0.251 + 1e-9:
                legal_bad.append((r['clip_id'], 'edge_off'))
            if is_master and D - e['onset'] < 2.85 - 1e-6:
                legal_bad.append((r['clip_id'], 'master_inside'))
            if j and e['onset'] - ev[j - 1]['offset'] < 0.200 - 1e-9:
                legal_bad.append((r['clip_id'], 'gap'))
        # audio
        y, sr = sf.read(root / r['audio_path'], dtype='int16')
        b, _ = sf.read(root / 'calib_beds' / (r['clip_id'][:-4] + '.bed.flac'), dtype='int32')
        b = b.astype(np.int64) >> 8                     # the bed is stored as int32 << 8 in PCM_24
        assert sr == SR and len(y) == len(b) == round(D * SR)
        y = y.astype(np.int64)
        b = b.astype(np.int64)
        peaks_bad += int(abs(int(np.max(np.abs(y))) - 32440) > 1)
        res = y - b
        inside = np.zeros(len(y), bool)
        nzb = b[b != 0].astype(np.float64)
        p_sp = float(np.mean(nzb ** 2))
        for q in p['events']:
            s0, n0 = q['paste_start'], q['samples']
            inside[s0:s0 + n0] = True
            seg = res[s0:s0 + n0].astype(np.float64)
            w = np.asarray(lib[q['k']]['seg'][:n0], np.float64) * 32768.0
            expect = p['scale'] * q['gain'] * w
            if np.max(np.abs(seg - expect)) > 1.0 + 1e-6:
                exact_bad += 1
            lvl = 10 * np.log10(np.mean(seg ** 2) / p_sp)
            if q['kind'] == 'vocal':
                vocal_lvl.append(-lvl)
                if np.any(b[s0:s0 + n0] != 0):
                    vocal_zero_bad += 1
            else:
                noise_lvl.append(lvl)
                noise_by_t.setdefault(q['k'], []).append(lvl)
        if np.max(np.abs(res[~inside])) > 1:
            exact_bad += 1
        s, e = zero_runs(b)
        zero_share.append(float(np.sum(e - s)) / len(b))
        empty = 0
        for s1, e1 in zip(s, e):
            if not np.any(inside[s1:e1]):
                empty += 1
        empty_counts.append(empty)
        for bd in p['beds']:
            if idx_ids.get(bd['bed_i']) != bd['clip_id'] or bd['clip_id'].split('_')[5] in val_sessions:
                bed_bad += 1
    n = len(man)
    shares = counts / n
    within = [v - np.mean(vs) for vs in noise_by_t.values() if len(vs) >= 2 for v in vs]
    dof = sum(len(vs) - 1 for vs in noise_by_t.values() if len(vs) >= 2)
    sd_w = float(np.sqrt(np.sum(np.square(within)) / max(dof, 1)))
    durs = np.asarray(durs)
    gr = root / f'gen-report-{a.tag}.json'
    gen_report = json.loads(gr.read_text()) if gr.exists() else {}
    bands = {
        'B1 count': {'shares': shares.round(4).tolist(), 'val': VAL_SHARES.round(4).tolist(),
                     'pass': bool(np.all(np.abs(shares - VAL_SHARES) <= 0.03))},
        'B2 legality': {'violations': len(legal_bad), 'examples': legal_bad[:5], 'pass': not legal_bad},
        'B3 vocal': {'events': len(vocal_lvl), 'not_on_exact_zeros': vocal_zero_bad,
                     'median_db': round(float(np.median(vocal_lvl)), 3), 'sd_db': round(float(np.std(vocal_lvl)), 3),
                     'pass': (vocal_zero_bad == 0 and abs(np.median(vocal_lvl) - 3.97) <= 0.15
                              and 0.30 <= np.std(vocal_lvl) <= 0.60)},
        'B4 noise': {'events': len(noise_lvl), 'min': round(min(noise_lvl), 3), 'max': round(max(noise_lvl), 3),
                     'median_db': round(float(np.median(noise_lvl)), 3), 'pooled_within_sd_db': round(sd_w, 3),
                     'pass': (min(noise_lvl) >= -12.24 - 0.01 and max(noise_lvl) <= 15.49 + 0.01
                              and abs(np.median(noise_lvl) - 5.65) <= 0.5 and abs(sd_w - 4.56) <= 0.3)},
        'B5 zeros': {'median_zero_share': round(float(np.median(zero_share)), 4),
                     'median_empty_runs_per_clip': float(np.median(empty_counts)),
                     'pass': bool(0.05 <= np.median(zero_share) <= 0.11 and 2 <= np.median(empty_counts) <= 10)},
        'B6 peak': {'clips_off_32440': peaks_bad, 'pass': peaks_bad == 0},
        'B7 exactness': {'violations': exact_bad, 'pass': exact_bad == 0},
        'B8 length': {'min': float(durs.min()), 'max': float(durs.max()), 'share_ge_7s': round(float(np.mean(durs >= 7)), 4),
                      'pass': bool(durs.min() >= 3.0 and durs.max() <= 15.2 and abs(np.mean(durs >= 7) - 0.369) <= 0.03)},
        'B9 sources': {'bad_beds': bed_bad, 'generator_audit_hook': gen_report.get('audit_hook', 'see gen_libsyn.py _hook'),
                       'pass': bed_bad == 0},
    }
    for v in bands.values():
        v['pass'] = bool(v['pass'])
    rep = {'dir': str(root), 'tag': a.tag, 'clips': n, 'bands': bands, 'ALL_PASS': all(v['pass'] for v in bands.values()),
           'calibrate_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'generator_sha256': hashlib.sha256((Path(__file__).parent / 'gen_libsyn.py').read_bytes()).hexdigest()}
    (root / f'calibration-{a.tag}.json').write_text(json.dumps(rep, indent=1, default=_np) + '\n')
    print(json.dumps(rep, indent=1, default=_np))


if __name__ == '__main__':
    main()
