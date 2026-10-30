"""Clean speech beds: training targets that look like the organizer clean references.

Why: the organizer clean references are much cleaner than raw Vaani speech and their pauses are exact digital zeros
(median 10 % zero samples; 507 of 695 clips hold a zero run > 0.1 s). Raw Vaani beds keep their natural background,
so a model trained on them learns to KEEP stationary noise.
Recipe per bed: DeepFilterNet3 (full suppression, MIT/Apache-2.0, third_party/deepfilternet3) on the whole source clip
-> cut the bed span -> Silero VAD mask (p >= 0.35, +-128 ms, 40 ms raised-cosine edges) -> int16.
Sources: the beds of speech_beds.jsonl (build_banks.py; Vaani Noise-Event training release, validation sessions
already dropped). No validation or test audio is used.

usage: build_clean_beds.py --part i/n [--silver-hours 14] --vaani-dir V --beds B --silero-jit J --work W
           (writes W/part_i.npy + W/part_i.jsonl)
       build_clean_beds.py --merge --work W --out OUT     (writes OUT/beds_clean_XX.npy, OUT/index.jsonl, OUT/BEDS_CLEAN.json)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DFN3 = Path(__file__).resolve().parents[2] / 'third_party' / 'deepfilternet3'   # infer.py + code/ (+ the weights)
V = BEDS = WORK = OUT = SILERO = None    # set from the command line (see usage)
SR = 16000
MIN_S = 2.0
P_SPEECH = 0.35
DIL = 4            # frames of 32 ms on each side (128 ms)
RAMP = 640         # 40 ms raised-cosine edge


def floor_db(x):
    if len(x) < 2048:
        return None
    fr = np.lib.stride_tricks.sliding_window_view(x, 512)[::256]
    e = np.mean(fr ** 2, axis=1) + 1e-12
    p10, p90 = np.percentile(e, [10, 90])
    return round(float(10 * np.log10(p90 / p10)), 2)


def select(silver_hours):
    beds = [json.loads(l) for l in open(BEDS)]
    for i, b in enumerate(beds):
        b['i'] = i
    beds = [b for b in beds if b['end'] - b['start'] >= MIN_S]
    gold = [b for b in beds if b['tier'] == 'ver']
    silver = [b for b in beds if b['tier'] == 'unv']
    random.Random(1234).shuffle(silver)
    keep, acc = [], 0.0
    for b in silver:
        if acc >= silver_hours * 3600:
            break
        keep.append(b)
        acc += b['end'] - b['start']
    return gold + keep


def part(i, n, silver_hours):
    sys.path.insert(0, str(DFN3))
    from infer import DeepFilterNet3
    torch.set_num_threads(2)
    enh = DeepFilterNet3('cpu')
    vad = torch.jit.load(str(SILERO)).eval()
    man = {}
    for l in open(V / 'manifest.jsonl'):
        r = json.loads(l)
        man[r['clip_id']] = r['audio_path']
    beds = select(silver_hours)
    by_clip = {}
    for b in beds:
        by_clip.setdefault(b['clip_id'], []).append(b)
    clips = sorted(c for c in by_clip if c in man)
    clips = [c for k, c in enumerate(clips) if k % n == i]
    WORK.mkdir(parents=True, exist_ok=True)
    chunks, rows, off = [], [], 0
    win = np.hanning(2 * RAMP + 1)
    win /= win.sum()
    t0 = time.time()
    for j, cid in enumerate(clips):
        x = sf.read(str(V / man[cid]), dtype='float32', always_2d=False)[0]
        x = (x if x.ndim == 1 else x[:, 0]).astype(np.float64) * 32768.0
        y = enh.enhance(x)
        for b in by_clip[cid]:
            a, e = int(b['start'] * SR), int(b['end'] * SR)
            seg = y[a:e]
            nfr = len(seg) // 512
            if nfr < 8:
                continue
            vad.reset_states()
            with torch.no_grad():
                p = np.array([float(vad(torch.from_numpy((seg[k * 512:(k + 1) * 512] / 32768.0).astype(np.float32)), SR))
                              for k in range(nfr)])
            g = p >= P_SPEECH
            gd = g.copy()
            for d in range(1, DIL + 1):
                gd[d:] |= g[:-d]
                gd[:-d] |= g[d:]
            gs = np.repeat(gd.astype(np.float64), 512)
            gs = np.concatenate([gs, np.full(len(seg) - len(gs), gs[-1] if len(gs) else 0.0)])
            gs = np.clip(np.convolve(gs, win, mode='same'), 0.0, 1.0)
            out = np.clip(np.rint(seg * gs), -32768, 32767).astype(np.int16)
            share = float(np.mean(gd))
            if share < 0.3:
                continue
            chunks.append(out)
            rows.append({'bed_i': b['i'], 'clip_id': cid, 'tier': b['tier'], 'start': b['start'], 'end': b['end'],
                         'offset': off, 'length': len(out), 'speech_share': round(share, 3),
                         'zero_share': round(float(np.mean(out == 0)), 4),
                         'floor_db_raw': floor_db(x[a:e]), 'floor_db_clean': floor_db(out.astype(np.float64))})
            off += len(out)
        if (j + 1) % 500 == 0:
            print(f'part {i}/{n}: {j + 1}/{len(clips)} clips, {off / SR / 3600:.2f} h, {time.time() - t0:.0f}s', flush=True)
    np.save(WORK / f'part_{i}.npy', np.concatenate(chunks) if chunks else np.zeros(0, np.int16))
    with open(WORK / f'part_{i}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'DONE part {i}/{n}: {len(rows)} beds, {off / SR / 3600:.2f} h, {time.time() - t0:.0f}s', flush=True)


def merge():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = sorted(WORK.glob('part_*.jsonl'))
    index, summ = [], {'beds': 0, 'hours': 0.0}
    for pj in parts:
        k = int(pj.stem.split('_')[1])
        arr = np.load(WORK / f'part_{k}.npy')
        name = f'beds_clean_{k:02d}.npy'
        np.save(OUT / name, arr)
        for l in open(pj):
            r = json.loads(l)
            r['shard'] = name
            index.append(r)
        summ['beds'] = len(index)
    summ['hours'] = round(sum(r['length'] for r in index) / SR / 3600, 3)
    for key in ('speech_share', 'zero_share', 'floor_db_raw', 'floor_db_clean'):
        v = [r[key] for r in index if r[key] is not None]
        summ[key + '_q10_50_90'] = np.percentile(v, [10, 50, 90]).round(3).tolist()
    summ['by_tier'] = {t: sum(1 for r in index if r['tier'] == t) for t in ('ver', 'unv')}
    with open(OUT / 'index.jsonl', 'w') as f:
        for r in index:
            f.write(json.dumps(r) + '\n')
    (OUT / 'BEDS_CLEAN.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--silver-hours', type=float, default=14.0)
    ap.add_argument('--vaani-dir', type=Path, help='local Vaani copy: manifest.jsonl + audio/')
    ap.add_argument('--beds', type=Path, help='speech_beds.jsonl of build_banks.py')
    ap.add_argument('--silero-jit', type=Path, help='silero_vad.jit (Silero VAD v5)')
    ap.add_argument('--work', type=Path, required=True, help='folder for the part files')
    ap.add_argument('--out', type=Path, help='output folder of --merge')
    a = ap.parse_args()
    V, BEDS, WORK, OUT, SILERO = a.vaani_dir, a.beds, a.work, a.out, a.silero_jit
    if a.merge:
        merge()
    else:
        i, n = (int(v) for v in a.part.split('/'))
        part(i, n, a.silver_hours)
