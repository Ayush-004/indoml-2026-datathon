"""valstats.py: generator statistics from the 695 organizer validation synthetic clips.
Reads validation.zip (clean references + noisy synthetic clips) and the noise-track table g01_val_tracks.pkl
(g01_val_tracks.py). Output: valstats.json next to this file. Validation data only. No test file is opened."""
import io, json, os, pickle, zipfile
import numpy as np, soundfile as sf
from pathlib import Path
SR = 16000
HERE = Path(__file__).resolve().parent
OUT = HERE / 'valstats.json'
g01 = pickle.load(open(os.environ.get('T1_G01', HERE / 'g01_val_tracks.pkl'), 'rb'))
LIB = pickle.load(open(os.environ.get('T1_LIBRARY', 'work/templates/library.pkl'), 'rb'))
META = {r['segmentFileName']: r for r in json.load(open(
    os.environ.get('T1_VAL_META', 'validationMetadata.json'))) if r['syntheticData']}
ev_by_clip = {}
for e in g01['events']:
    ev_by_clip.setdefault(e['clip'], []).append(e)


def runs_of_zero(x, min_len):
    z = (x == 0).astype(np.int8)
    d = np.diff(np.concatenate([[0], z, [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) >= min_len
    return s[keep], e[keep]


rows = []
clipdur = []
vocal = []
empty_runs = []
zero_share = []
nruns = []
with zipfile.ZipFile(os.environ.get('INDOML_VALIDATION_ZIP', 'validation.zip')) as z:
    for c, evs in sorted(ev_by_clip.items()):
        clean, sr = sf.read(io.BytesIO(z.read('validation/syntheticCleanRefAudio/' + c)), dtype='int16')
        assert sr == SR
        clean = clean.astype(np.int32)
        n = len(clean)
        s, e = runs_of_zero(clean, 16)            # exact-zero runs >= 1 ms
        zero_share.append(float(np.sum(e - s)) / n)
        nz = clean[clean != 0].astype(np.float64) / 32768.0
        p_sp = float(np.mean(nz ** 2))
        used = np.zeros(len(s), bool)
        for ev in evs:
            sa, sb = int(round(ev['on'] * SR)), int(round(ev['off'] * SR))
            is_vocal = ev['tag'].startswith('[')
            snr = 10 * np.log10(max(ev['p_noise'], 1e-30) / max(p_sp, 1e-30))
            cat = META[c]['NoiseSubCategoryTimeStamp'][ev['i']]['category']
            assert META[c]['NoiseSubCategoryTimeStamp'][ev['i']]['tag'] == ev['tag']
            rows.append({'clip': c, 'tag': ev['tag'], 'k': ev['k'], 'vocal': is_vocal, 'dur': ev['dur'], 'category': cat,
                         'snr_glob': snr, 'clip_dur': ev['clip_dur'], 'on': ev['on'], 'off': ev['off']})
            mid = (sa + sb) // 2
            j = np.flatnonzero((s <= mid) & (e > mid))
            if len(j):
                used[j[0]] = True
                if is_vocal:
                    vocal.append({'pre_ms': (sa - s[j[0]]) / 16.0, 'post_ms': (e[j[0]] - sb) / 16.0,
                                  'run_ms': (e[j[0]] - s[j[0]]) / 16.0, 'ev_ms': (sb - sa) / 16.0})
            elif is_vocal:
                vocal.append({'pre_ms': None, 'post_ms': None, 'run_ms': 0.0, 'ev_ms': (sb - sa) / 16.0})
        er = [(int(a), int(b)) for a, b, u in zip(s, e, used) if not u]
        nruns.append(len(er))
        clipdur.append(n / SR)
        empty_runs += [(b - a) / 16.0 for a, b in er]
        # where are the empty runs: share at the clip edges
for r in rows:
    pass


def q(x, qs=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    x = np.asarray([v for v in x if v is not None], float)
    return {str(k): round(float(np.quantile(x, k)), 3) for k in qs} | {'n': int(len(x)), 'mean': round(float(x.mean()), 3)}


snr_noise = [r['snr_glob'] for r in rows if not r['vocal']]
snr_vocal = [r['snr_glob'] for r in rows if r['vocal']]
per_t = {}
for r in rows:
    if not r['vocal'] and r['k'] is not None:
        per_t.setdefault(r['k'], []).append(r['snr_glob'])
within = [v - np.mean(vs) for vs in per_t.values() if len(vs) >= 2 for v in vs]
dof = sum(len(vs) - 1 for vs in per_t.values() if len(vs) >= 2)
sigma_w = float(np.sqrt(np.sum(np.square(within)) / max(dof, 1)))
out = {
    'clips': len(ev_by_clip), 'events': len(rows),
    'zero_share_runs_ge_1ms': q(zero_share),
    'empty_zero_runs_per_clip': q(nruns), 'empty_zero_run_ms': q(empty_runs),
    'vocal_slot': {'pre_ms': q([v['pre_ms'] for v in vocal]), 'post_ms': q([v['post_ms'] for v in vocal]),
                   'run_ms': q([v['run_ms'] for v in vocal]), 'ev_ms': q([v['ev_ms'] for v in vocal]),
                   'not_in_zero_run': sum(1 for v in vocal if v['pre_ms'] is None)},
    'snr_glob_noise': q(snr_noise) | {'min': round(min(snr_noise), 2), 'max': round(max(snr_noise), 2)},
    'minus_snr_glob_vocal': q([-v for v in snr_vocal]) | {'sd': round(float(np.std(snr_vocal)), 3)},
    'noise_templates_with_val_level': len(per_t), 'sigma_w_pooled_within_template_db': round(sigma_w, 3),
    'template_mean_snr': {str(k): round(float(np.mean(v)), 3) for k, v in sorted(per_t.items())},
    'template_n': {str(k): len(v) for k, v in sorted(per_t.items())},
    'library_templates': len(LIB), 'library_vocal': sum(1 for t in LIB if t['tag'].startswith('[')),
}
cats = {}
for r in rows:
    if r['k'] is not None:
        cats.setdefault(r['k'], {}).setdefault(r['category'], 0)
        cats[r['k']][r['category']] += 1
out['template_category'] = {str(k): max(v, key=v.get) for k, v in sorted(cats.items())}
out['template_category_conflicts'] = {str(k): v for k, v in cats.items() if len(v) > 1}
out['raw'] = {'vocal_margin_pairs_ms': [[round(v['pre_ms'], 4), round(v['post_ms'], 4)] for v in vocal if v['pre_ms'] is not None],
              'empty_runs_per_clip': nruns, 'clip_dur_s': [round(x, 4) for x in clipdur],
              'empty_run_ms': [round(x, 4) for x in empty_runs]}
OUT.write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: v for k, v in out.items() if k not in ('template_mean_snr', 'template_n', 'raw', 'template_category')}, indent=1))
