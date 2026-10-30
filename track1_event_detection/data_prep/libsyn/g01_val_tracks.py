"""The exact noise track of every validation synthetic clip (input of valstats.py).

noise = noisy - clean (validation.zip syntheticCleanRefAudio; int16 domain). Per clip and per
labelled event this script measures what the generator did:
  * support: is the residual exactly zero outside the labels?
  * template of each event (library member of the same tag, lag-searched NCC on the residual)
  * exact paste start (sample lag of the template against the residual) vs the label onset
  * paste gain against the template reference, noise RMS, SNR against the clean speech
    (whole clip and local span)
  * geometry: onsets, gaps, first onset, last end, events per clip, clip duration, name span
Labels and clean refs are used here only to learn the generator.
Output: g01_val_tracks.pkl next to this file (per-event rows + per-clip rows); the residuals are cached in
T1_STORE (val_res.i16)."""
import io, os, sys, json, pickle, re, zipfile, time
import numpy as np, soundfile as sf
from scipy.signal import correlate
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'system_950232', 'job2')
sys.path.insert(0, H)
import lib, audio_store
SR = 16000
V3 = os.path.dirname(os.path.abspath(__file__)) + '/'
STORE = os.path.join(os.environ.get('T1_STORE', 'store'), 'v3store', '')
os.makedirs(STORE, exist_ok=True)
LIB = lib.library(); DUR = np.array([t['dur'] for t in LIB])
meta = [r for r in lib.val_meta() if r['syntheticData']]
idx, mm = audio_store.load('val')
PAT = re.compile(r'_(\d+)_(\d+)\.wav$')

def best_lag_ncc(seg, tmpl, maxlag=480):
    """seg: residual around the label (padded by maxlag each side); tmpl: template seg.
    Returns (ncc, lag) where lag = template start relative to the label onset (samples)."""
    L = len(tmpl)
    if len(seg) < L: return 0.0, 0
    cc = correlate(seg, tmpl, mode='valid', method='fft')
    cs = np.concatenate([[0.0], np.cumsum(seg * seg)])
    e = np.maximum(cs[L:] - cs[:-L], 1e-18)[:len(cc)]
    v = np.abs(cc) / (np.sqrt(e) * (np.linalg.norm(tmpl) + 1e-18))
    j = int(np.argmax(v))
    return float(v[j]), j - maxlag

tag_members = {}
for k, t in enumerate(LIB): tag_members.setdefault(t['tag'], []).append(k)

res_index = {}; off = 0
rows_ev = []; rows_clip = []
t0 = time.time()
with zipfile.ZipFile(lib.VAL_ZIP) as z, open(STORE + 'val_res.i16', 'wb') as fout:
    for n, r in enumerate(meta):
        c = r['segmentFileName']
        noisy = np.asarray(mm[idx[c][0]:idx[c][0] + idx[c][1]], np.int32)
        clean, sr = sf.read(io.BytesIO(z.read('validation/syntheticCleanRefAudio/' + c)), dtype='int16')
        assert sr == SR
        clean = clean.astype(np.int32)
        if len(clean) != len(noisy):
            print('length mismatch', c, len(clean), len(noisy))
        Lm = min(len(clean), len(noisy))
        res = noisy[:Lm] - clean[:Lm]
        fout.write(np.clip(res, -32768, 32767).astype(np.int16).tobytes()); res_index[c] = [off, Lm, len(noisy), len(clean)]; off += Lm
        resf = res.astype(np.float64) / 32768.0; cleanf = clean[:Lm].astype(np.float64) / 32768.0
        noisyf = noisy[:Lm].astype(np.float64) / 32768.0
        evs = [(float(e['start']), float(e['end']), e['tag']) for e in r['NoiseSubCategoryTimeStamp']]
        inside = np.zeros(Lm, bool)
        for a, b, _ in evs: inside[int(round(a * SR)):int(round(b * SR))] = True
        tot = float(np.sum(resf ** 2)) + 1e-30
        nz = np.flatnonzero(res)
        m = PAT.search(c); span = (int(m.group(2)) - int(m.group(1))) / 1000.0
        # clip-level speech power (clean) and noisy power
        p_speech = float(np.mean(cleanf ** 2)); p_noisy = float(np.mean(noisyf ** 2))
        clip_row = {'clip': c, 'dur': float(r['duration']), 'nsamp': Lm, 'len_noisy': len(noisy), 'len_clean': len(clean),
                    'span': span, 'n_events': len(evs), 'res_energy_inside': float(np.sum(resf[inside] ** 2)) / tot,
                    'nz_outside': int(np.sum((res != 0) & ~inside)), 'max_abs_outside': int(np.max(np.abs(res[~inside]))) if (~inside).any() else 0,
                    'first_nz': int(nz[0]) if len(nz) else -1, 'last_nz': int(nz[-1]) if len(nz) else -1,
                    'p_speech': p_speech, 'p_noisy': p_noisy, 'peak_noisy': int(np.max(np.abs(noisy))), 'peak_clean': int(np.max(np.abs(clean))),
                    'transcript': r.get('transcript', ''), 'state': r['state'], 'district': r['district'], 'language': r.get('language')}
        rows_clip.append(clip_row)
        for ei, (a, b, tag) in enumerate(evs):
            sa = int(round(a * SR)); sb = int(round(b * SR))
            seg = resf[sa:sb]
            # template: same tag, NCC with a lag search of +-30 ms
            pad = 480
            lo = max(0, sa - pad); padl = sa - lo
            ext = np.concatenate([np.zeros(pad - padl), resf[lo:sb + pad]])
            ext = np.concatenate([ext, np.zeros(max(0, (sb - sa) + 2 * pad - len(ext)))])
            cand = []
            for k in tag_members.get(tag, []):
                t = np.asarray(LIB[k]['seg'], np.float64)
                Lt = min(len(t), sb - sa)
                if Lt < 160: continue
                v, lag = best_lag_ncc(ext[:Lt + 2 * pad], t[:Lt], pad)
                # prefer the template whose duration matches (cuts of one master all match the prefix)
                cand.append((v - 0.001 * abs(DUR[k] - (b - a)), v, k, lag))
            if cand:
                _, v, k, lag = max(cand)
            else:
                v, k, lag = 0.0, None, 0
            # paste gain against the template reference (the template seg is itself a residual)
            if k is not None:
                t = np.asarray(LIB[k]['seg'], np.float64)
                p0 = sa + lag; Lt = min(len(t), Lm - max(p0, 0))
                y = resf[max(p0, 0):max(p0, 0) + Lt]; tt = t[:len(y)]
                g = float(np.dot(y, tt) / (np.dot(tt, tt) + 1e-30))
                fit_res = float(np.sum((y - g * tt) ** 2) / (np.sum(y ** 2) + 1e-30))
            else:
                g = np.nan; fit_res = np.nan; p0 = sa
            ev_nz = np.flatnonzero(res[max(0, sa - 800):min(Lm, sb + 800)]) + max(0, sa - 800)
            p_ev = float(np.mean(seg ** 2)) if len(seg) else 0.0
            p_sp_local = float(np.mean(cleanf[sa:sb] ** 2)) if sb > sa else 0.0
            rows_ev.append({'clip': c, 'i': ei, 'tag': tag, 'on': a, 'off': b, 'dur': b - a, 'k': k, 'ncc': v, 'lag': int(lag),
                            'paste_start': int(p0), 'g': g, 'fit_res': fit_res,
                            'nz_first': int(ev_nz[0]) if len(ev_nz) else -1, 'nz_last': int(ev_nz[-1]) if len(ev_nz) else -1,
                            'p_noise': p_ev, 'p_speech_local': p_sp_local, 'p_speech_clip': p_speech,
                            'clip_dur': float(r['duration']), 'nsamp': Lm,
                            'peak_res': float(np.max(np.abs(seg))) if len(seg) else 0.0})
        if n % 100 == 0: print(n, round(time.time() - t0), flush=True)
json.dump(res_index, open(STORE + 'val_res_index.json', 'w'))
pickle.dump({'events': rows_ev, 'clips': rows_clip}, open(V3 + 'g01_val_tracks.pkl', 'wb'))
print('done', len(rows_ev), 'events', len(rows_clip), 'clips', round(time.time() - t0))
