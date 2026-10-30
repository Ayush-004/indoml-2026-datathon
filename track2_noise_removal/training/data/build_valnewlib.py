"""val-newlib: a second proxy set = the organizer generator with a NEW noise library (measurement set; clean).

Why: the Phase-2 test uses a new noise library. val-hard keeps the Phase-1 noise library, so it cannot show how a
model does on unseen noise. Here every organizer event of the 695
validation synthetic clips keeps its time span and its noise ENERGY, but its waveform is replaced by a segment of a
HELD-OUT public noise file (build_public_noise.py heldout(): DEMAND TMETRO / NPARK and ESC-50 fold 5; MUSAN is
training-only; none of the held-out files is in the training bank). Outside the events the clip is the clean reference (as in the
organizer clips). Levels: 'newlib' = the organizer level; 'newlib_hard' = 1.68 x noise (val-hard level). Peak rule as
the organizer clips (a mixture above 32,440 LSB is scaled down). No test audio is used.

Output: OUT/{newlib,newlib_hard}/syntheticNoiseAudio/*.wav, OUT/events.jsonl (source, file, offset per event),
OUT/SUMMARY.json (mean input SI-SDR per level). The training scripts read OUT as the folder val_newlib.

usage: build_valnewlib.py --validation-dir VALIDATION --public-noise-src SRC --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_public_noise as BPN  # noqa: E402

VAL = OUT = None    # set from the command line (see usage)
SR = 16000
FADE = 80


def si_sdr(reference, enhanced):
    """Official SI-SDR snippet of the competition (float64, truncate, clip to [-100, 100])."""
    ref = np.asarray(reference, dtype=np.float64)
    enh = np.asarray(enhanced, dtype=np.float64)
    n = min(len(ref), len(enh))
    ref, enh = ref[:n], enh[:n]
    scale = np.dot(enh, ref) / np.dot(ref, ref)
    s_target = scale * ref
    e_noise = enh - s_target
    with np.errstate(divide='ignore', invalid='ignore'):
        value = 10.0 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise))
    return float(np.clip(value, -100.0, 100.0))


def pool():
    items = []
    for src, cat, f, lic, row in BPN.items():
        if BPN.heldout(src, cat, f, row):
            try:
                x, sr = sf.read(str(f), dtype='float32', always_2d=False)
            except Exception:
                continue
            x = BPN.to16(x.astype(np.float64), sr)
            if len(x) >= SR // 4 and np.isfinite(x).all() and np.max(np.abs(x)) > 1e-4:
                items.append((src, cat, str(f.relative_to(BPN.SRC)), x))
    return items


def main():
    P = pool()
    by_cat = {}                     # (source, category) -> files; a category is drawn uniformly per event
    for it in P:
        by_cat.setdefault((it[0], it[1]), []).append(it)
    keys = sorted(by_cat)
    by_src = {}
    for it in P:
        by_src[it[0]] = by_src.get(it[0], 0) + 1
    print('held-out pool:', by_src, len(keys), 'categories', flush=True)
    meta = json.loads((VAL / 'validationMetadata.json').read_text())
    for lvl in ('newlib', 'newlib_hard'):
        (OUT / lvl / 'syntheticNoiseAudio').mkdir(parents=True, exist_ok=True)
    ev_out = open(OUT / 'events.jsonl', 'w')
    si = {'newlib': [], 'newlib_hard': []}
    for r in meta:
        if not r['syntheticData']:
            continue
        name = r['segmentFileName']
        s = sf.read(str(VAL / 'syntheticCleanRefAudio' / name), dtype='int16')[0].astype(np.float64)
        x = sf.read(str(VAL / 'syntheticNoiseAudio' / name), dtype='int16')[0].astype(np.float64)
        n = min(len(s), len(x))
        s, x = s[:n], x[:n]
        a0 = np.dot(x, s) / np.dot(s, s) if np.dot(s, s) > 0 else 1.0
        n_org = x - a0 * s                                   # organizer noise at the organizer mixture scale
        n_new = np.zeros(n)
        for k, e in enumerate(r['NoiseSubCategoryTimeStamp'] or []):
            a = max(0, int(float(e['start']) * SR))
            b = min(n, int(float(e['end']) * SR))
            if b - a < 160:
                continue
            en = float(np.sum(n_org[a:b] ** 2))
            if en <= 0:
                continue
            h = int(hashlib.sha1(f'{name}|{k}'.encode()).hexdigest(), 16)
            cat = keys[h % len(keys)]
            it = by_cat[cat][(h // 53) % len(by_cat[cat])]
            w = it[3]
            L = b - a
            if len(w) < L:
                w = np.tile(w, int(np.ceil(L / len(w))))
            # 8 hashed offsets; keep the loudest segment (ESC-50 clips hold digital silence around short sounds)
            cands = [(h // (97 * (j + 1))) % (len(w) - L + 1) for j in range(8)]
            o = max(cands, key=lambda c: float(np.sum(w[c:c + L] ** 2)))
            seg = w[o:o + L].copy()
            f = min(FADE, L // 4)
            if f > 0:
                ramp = np.linspace(0, 1, f)
                seg[:f] *= ramp
                seg[-f:] *= ramp[::-1]
            seg *= np.sqrt(en / (np.sum(seg ** 2) + 1e-12))
            n_new[a:b] += seg
            ev_out.write(json.dumps({'clip': name, 'k': k, 'tag': e.get('tag'), 'source': it[0], 'category': it[1],
                                     'file': it[2], 'offset': int(o), 'len': int(L)}) + '\n')
        for lvl, g in (('newlib', 1.0), ('newlib_hard', 1.68)):
            y = a0 * s + g * n_new
            pk = np.max(np.abs(y))
            if pk > 32440:
                y *= 32440 / pk
            q = np.clip(np.rint(y), -32768, 32767).astype(np.int16)
            sf.write(str(OUT / lvl / 'syntheticNoiseAudio' / name), q, SR, subtype='PCM_16')
            si[lvl].append(si_sdr(s, q.astype(np.float64)))
    ev_out.close()
    summ = {lvl: {'clips': len(v), 'input_si_sdr_mean': round(float(np.mean(v)), 4)} for lvl, v in si.items()}
    summ['pool'] = by_src
    summ['categories'] = len(keys)
    (OUT / 'SUMMARY.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validation-dir', type=Path, required=True,
                    help='organizer validation folder: validationMetadata.json, syntheticNoiseAudio/, syntheticCleanRefAudio/')
    ap.add_argument('--public-noise-src', type=Path, required=True, help='the --src folder of build_public_noise.py')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    VAL, OUT, BPN.SRC = a.validation_dir, a.out, a.public_noise_src
    main()
