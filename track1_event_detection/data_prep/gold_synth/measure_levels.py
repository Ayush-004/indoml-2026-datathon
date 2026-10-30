"""Measure the piece-to-bed level distribution of the organizer synthetic validation clips (val_levels.pkl).

For every synthetic validation clip: residual = noisy - cleanRef is the pasted piece.
For each annotated event window measure
   ratio_dB = 20*log10( rms(residual in window) / rms(cleanRef in window) )   [piece vs bed, same window]
   ratio_glob_dB = 20*log10( rms(residual in window) / rms(cleanRef whole clip) )
Also the bed noisiness proxies for the 695 clean refs.
"""
import zipfile, io, json, pickle, os
import numpy as np
import soundfile as sf

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
VAL_META = os.environ.get('T1_VAL_META', 'validationMetadata.json')
Z = os.environ.get('INDOML_VALIDATION_ZIP', 'validation.zip')


def flatness(x, nfft=512):
    if len(x) < nfft:
        return float('nan')
    n = (len(x) // nfft) * nfft
    fr = x[:n].reshape(-1, nfft) * np.hanning(nfft)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    gm = np.exp(np.mean(np.log(P), axis=1))
    am = np.mean(P, axis=1)
    return float(np.median(gm / am))


def main():
    meta = json.load(open(VAL_META))
    syn = [r for r in meta if r['syntheticData']]
    z = zipfile.ZipFile(Z)
    rows, beds = [], []
    for i, r in enumerate(syn):
        fn = r['segmentFileName']
        x, _ = sf.read(io.BytesIO(z.read('validation/syntheticNoiseAudio/' + fn)), dtype='float64')
        c, _ = sf.read(io.BytesIO(z.read('validation/syntheticCleanRefAudio/' + fn)), dtype='float64')
        res = x - c
        dur = len(c) / SR
        crms = float(np.sqrt(np.mean(c ** 2)) + 1e-12)
        # bed stats (frame energies, to find speech-active frames)
        W = 400
        nf = len(c) // W
        fe = np.sqrt(np.mean(c[:nf * W].reshape(-1, W) ** 2, axis=1) + 1e-20)
        act = fe > np.quantile(fe, 0.60)
        act_rms = float(np.sqrt(np.mean(fe[act] ** 2)))
        sil_rms = float(np.sqrt(np.mean(fe[fe <= np.quantile(fe, 0.20)] ** 2)))
        beds.append(dict(clip=fn, dur=dur, rms=crms, act_rms=act_rms, sil_rms=sil_rms,
                         dr_db=20 * np.log10(act_rms / (sil_rms + 1e-12)),
                         flat=flatness(c), peak=float(np.max(np.abs(c))),
                         clipped=float(np.mean(np.abs(x) > 0.999))))
        evs = sorted(r['NoiseSubCategoryTimeStamp'], key=lambda e: float(e['start']))
        ons = []
        for e in evs:
            s, t = float(e['start']), float(e['end'])
            a, b = int(round(s * SR)), int(round(t * SR))
            w_res, w_c = res[a:b], c[a:b]
            if len(w_res) < 16:
                continue
            rr = float(np.sqrt(np.mean(w_res ** 2)) + 1e-12)
            cr = float(np.sqrt(np.mean(w_c ** 2)) + 1e-12)
            rows.append(dict(clip=fn, tag=e['tag'], onset=s, offset=t, dur=t - s, clipdur=dur,
                             ratio_db=20 * np.log10(rr / cr),
                             ratio_glob_db=20 * np.log10(rr / crms),
                             ratio_act_db=20 * np.log10(rr / (act_rms + 1e-12)),
                             piece_rms=rr, bed_rms=cr))
            ons.append((s, t))
        # geometry
        if ons:
            rows[-1]['_geo'] = dict(clip=fn, n=len(ons), first=ons[0][0], last_end=ons[-1][1], dur=dur,
                                    gaps=[round(ons[k + 1][0] - ons[k][1], 4) for k in range(len(ons) - 1)])
    pickle.dump(dict(events=rows, beds=beds), open(os.path.join(HERE, 'val_levels.pkl'), 'wb'))
    q = lambda a, ps=(0, .05, .25, .5, .75, .95, 1): [round(float(np.quantile(a, p)), 2) for p in ps]
    rd = np.array([r['ratio_db'] for r in rows])
    print('synthetic val events:', len(rows), 'clips:', len(syn))
    print('piece/bed ratio dB (same window)   q0,5,25,50,75,95,100 =', q(rd))
    print('piece/bed-global ratio dB          =', q(np.array([r['ratio_glob_db'] for r in rows])))
    print('piece/bed-active ratio dB          =', q(np.array([r['ratio_act_db'] for r in rows])))
    print('mean %.2f sd %.2f' % (rd.mean(), rd.std()))
    geo = [r['_geo'] for r in rows if '_geo' in r]
    print('n events per clip:', np.bincount([g['n'] for g in geo]).tolist())
    print('first onset s   =', q(np.array([g['first'] for g in geo])))
    print('dur-last_end s  =', q(np.array([g['dur'] - g['last_end'] for g in geo])))
    gaps = np.array([x for g in geo for x in g['gaps']])
    print('gaps between pieces s =', q(gaps) if len(gaps) else 'none', 'n', len(gaps), 'min', round(float(gaps.min()), 4) if len(gaps) else '-')
    print('clip duration s =', q(np.array([g['dur'] for g in geo])))
    print('--- clean-ref beds (695) ---')
    print('bed rms         =', q(np.array([b['rms'] for b in beds])))
    print('bed dyn range dB=', q(np.array([b['dr_db'] for b in beds])))
    print('bed flatness    =', q(np.array([b['flat'] for b in beds])))
    print('bed sil_rms     =', q(np.array([b['sil_rms'] for b in beds])))
    print('bed peak        =', q(np.array([b['peak'] for b in beds])))
    print('noisy clipped frac>0 :', sum(1 for b in beds if b['clipped'] > 0))


if __name__ == '__main__':
    main()
