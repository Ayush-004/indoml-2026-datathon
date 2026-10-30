"""One pass over the 11,106 Gold clips.

Produces
  * bed candidates  - maximal speech-only runs >= MINBED s, >= GUARD s away from every
    annotated event, with noisiness proxies (rms, dynamic range, spectral flatness,
    frame-energy stats).
  * noise pieces    - one per annotated Gold event (the organizers' recipe: the cut is
    the raw waveform between the annotated boundaries), with level stats relative to
    the same clip's speech-only material.
Metadata only; the audio is re-cut on demand by the generator. Output: gold_stats.pkl next to this file.

  python gold_stats.py [n_processes]
"""
import os, sys, json, time, pickle
import numpy as np
import soundfile as sf
from multiprocessing import Pool

SR = 16000
W = 400                      # 25 ms frames
GUARD = 0.30                 # s kept clear around every annotated event
MINBED = 1.50                # s minimum speech-only run
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('T1_WORK', '.')
# The vaani-gold rows hold absolute audio paths. If the audio moved after the split was written, set both prefixes.
OLD = os.environ.get('T1_AUDIO_PREFIX_OLD', '')
NEW = os.environ.get('T1_AUDIO_PREFIX_NEW', '')
GOLD = os.path.join(WORK, 'vaani-gold', '')


def load_gold():
    rows = []
    for split in ('train', 'val'):
        for line in open(GOLD + split + '.jsonl'):
            r = json.loads(line)
            r['_split'] = split
            r['_path'] = r['audio_path'].replace(OLD, NEW)
            rows.append(r)
    return rows


def frame_rms(x):
    n = len(x) // W
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(x[:n * W].reshape(-1, W) ** 2, axis=1) + 1e-20)


def flatness(x, nfft=512):
    if len(x) < nfft:
        return float('nan')
    n = (len(x) // nfft) * nfft
    fr = x[:n].reshape(-1, nfft) * np.hanning(nfft)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    return float(np.median(np.exp(np.mean(np.log(P), axis=1)) / np.mean(P, axis=1)))


def job(args):
    clip_id, path, dur, events, split = args
    try:
        x, sr = sf.read(path, dtype='float64')
    except Exception as e:
        return clip_id, None, None, str(e)
    if x.ndim > 1:
        x = x.mean(axis=1)
    N = len(x)
    ev_mask = np.zeros(N, bool)
    for e in events:
        a = max(0, int(round((e['onset'] - GUARD) * SR)))
        b = min(N, int(round((e['offset'] + GUARD) * SR)))
        if b > a:
            ev_mask[a:b] = True
    clean = ~ev_mask
    # ---- speech-only statistics for this clip (used as the reference level) ----
    if clean.sum() >= W:
        cf = frame_rms(x[clean])
        sp_rms = float(np.sqrt(np.mean(cf ** 2)))
        sp_act = float(np.sqrt(np.mean(cf[cf > np.quantile(cf, 0.60)] ** 2)))
        sp_flr = float(np.quantile(cf, 0.10))
    else:
        sp_rms = sp_act = sp_flr = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    # ---- bed candidates: maximal clean runs >= MINBED ----
    beds = []
    d = np.diff(np.concatenate([[0], clean.view(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    for a, b in zip(starts, ends):
        if (b - a) < MINBED * SR:
            continue
        seg = x[a:b]
        f = frame_rms(seg)
        if len(f) < 4:
            continue
        hi = float(np.sqrt(np.mean(f[f > np.quantile(f, 0.60)] ** 2)))
        lo = float(np.quantile(f, 0.10))
        beds.append(dict(clip_id=clip_id, split=split, start=round(a / SR, 4), end=round(b / SR, 4),
                         dur=round((b - a) / SR, 4),
                         rms=float(np.sqrt(np.mean(seg ** 2))), act_rms=hi, floor_rms=lo,
                         dr_db=float(20 * np.log10((hi + 1e-12) / (lo + 1e-12))),
                         flat=flatness(seg), peak=float(np.max(np.abs(seg))),
                         act_frac=float(np.mean(f > max(lo * 4, 1e-4)))))
    # ---- pieces: one per annotated event, cut at the annotated boundaries ----
    pieces = []
    for i, e in enumerate(events):
        a = max(0, int(round(e['onset'] * SR)))
        b = min(N, int(round(e['offset'] * SR)))
        if b - a < int(0.04 * SR):
            continue
        seg = x[a:b]
        r = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        f = frame_rms(seg)
        pieces.append(dict(clip_id=clip_id, split=split, idx=i, onset=e['onset'], offset=e['offset'],
                           dur=round(e['offset'] - e['onset'], 4), tag=e['tag'], category=e['category'],
                           rms=r, peak=float(np.max(np.abs(seg))),
                           ev_to_speech_db=float(20 * np.log10(r / (sp_rms + 1e-12))),
                           ev_to_act_db=float(20 * np.log10(r / (sp_act + 1e-12))),
                           crest_db=float(20 * np.log10(float(np.max(np.abs(seg)) + 1e-12) / r)),
                           flat=flatness(seg),
                           frame_dr_db=float(20 * np.log10((float(f.max()) + 1e-12) / (float(np.quantile(f, 0.10)) + 1e-12))) if len(f) >= 4 else float('nan')))
    info = dict(clip_id=clip_id, split=split, dur=dur, N=N, n_events=len(events),
                sp_rms=sp_rms, sp_act=sp_act, sp_floor=sp_flr,
                clean_frac=float(clean.mean()), peak=float(np.max(np.abs(x))),
                clipped=float(np.mean(np.abs(x) > 0.999)))
    return clip_id, dict(beds=beds, pieces=pieces, info=info), None, None


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = load_gold()
    jobs = [(r['clip_id'], r['_path'], r['duration'], r['events'], r['_split']) for r in rows]
    t0 = time.time()
    beds, pieces, infos, errs = [], [], [], {}
    with Pool(nproc) as pool:
        for i, (cid, res, _, err) in enumerate(pool.imap_unordered(job, jobs, chunksize=16)):
            if err:
                errs[cid] = err
                continue
            beds.extend(res['beds'])
            pieces.extend(res['pieces'])
            infos.append(res['info'])
            if (i + 1) % 2000 == 0:
                print(f'{i+1}/{len(jobs)} {time.time()-t0:.0f}s beds {len(beds)} pieces {len(pieces)}', flush=True)
    pickle.dump(dict(beds=beds, pieces=pieces, infos=infos, errors=errs,
                     params=dict(GUARD=GUARD, MINBED=MINBED, W=W)),
                open(os.path.join(HERE, 'gold_stats.pkl'), 'wb'))
    print(f'DONE {time.time()-t0:.0f}s  beds {len(beds)}  pieces {len(pieces)}  '
          f'clips {len(infos)}  errors {len(errs)}', flush=True)
    bd = np.array([b['dur'] for b in beds])
    print('bed run durations s q:', [round(float(np.quantile(bd, p)), 2) for p in (0, .25, .5, .75, .95, 1)],
          'total hours %.2f' % (bd.sum() / 3600))
    print('clips contributing >=1 bed:', len(set(b['clip_id'] for b in beds)))
    ed = np.array([p['ev_to_speech_db'] for p in pieces])
    print('piece event-to-speech dB q:', [round(float(np.quantile(ed, p)), 1) for p in (0, .05, .25, .5, .75, .95, 1)])
