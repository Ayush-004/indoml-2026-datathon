"""Public noise bank (noise source 'pub' of the training mixtures): public corpora only, no test data.

Sources (SRC = the downloaded corpora, see the README for the official URLs and licences):
  DEMAND 16 kHz (CC-BY-4.0): 12 environments without babble (kitchen, living room, washing machine, office, hallway,
      river, park, field, bus, car, metro, traffic); channels 1 and 9.
  ESC-50 (CC-BY-NC-3.0): 2,000 x 5 s environmental sounds, 50 classes (no speech words), 44.1 -> 16 kHz.
  MUSAN (CC-BY-4.0; music files carry their own CC licences): noise/ (free-sound, sound-bible) without files whose
      annotation mentions voices or whose Silero VAD speech share is >= 0.2 (SRC/musan_noise_vad.json, optional);
      music/ with vocals = N only, at most MUSIC_H hours.
Held out of training (used only by build_valnewlib.py): DEMAND TMETRO and NPARK, ESC-50 fold 5.
Output: OUT/pub_XX.npy (int16 shards, 16 kHz) + OUT/index.jsonl (source, category, file, offset, length, licence) +
OUT/NOISE_PUBLIC.json.

usage: build_public_noise.py --src SRC --out OUT
       SRC layout: demand/<ENV>_16k/.../ch01.wav, esc50/<...>/meta/esc50.csv + audio/, musan/musan/{noise,music}/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SRC = OUT = None    # set from the command line (see usage); build_valnewlib.py sets SRC
SR = 16000
MUSIC_H = 5.0
VOICE = re.compile(r'voice|talk|speech|speak|crowd|people|child|kid|babble|chatter|conversation|singing|sing |'
                   r'vocal|laugh|shout|scream|yell|announce|radio|tv|television', re.I)


HELDOUT_DEMAND = {'TMETRO', 'NPARK'}           # whole environments kept out of training (val-newlib proxy)


def heldout(src, cat, f, row=None):
    """Deterministic split: True = kept out of the training bank (used only by sim/build_valnewlib.py)."""
    if src == 'demand':
        return cat in HELDOUT_DEMAND
    if src == 'esc50':
        return row is not None and str(row.get('fold')) == '5'
    return False          # MUSAN: all files go to training (val-newlib uses ESC-50 + DEMAND only)


def to16(x, sr):
    x = x if x.ndim == 1 else x[:, 0]
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x


def items():
    # DEMAND
    for d in sorted((SRC / 'demand').glob('*')):
        if not d.is_dir():
            continue
        for ch in ('ch01.wav', 'ch09.wav'):
            f = next(d.rglob(ch), None)
            if f:
                yield 'demand', d.name.split('_')[0], f, 'CC-BY-4.0', None
    # ESC-50
    meta = next((SRC / 'esc50').rglob('esc50.csv'), None)
    if meta:
        root = meta.parent.parent / 'audio'
        for r in csv.DictReader(open(meta)):
            yield 'esc50', r['category'], root / r['filename'], 'CC-BY-NC-3.0', r
    # MUSAN noise
    for sub in ('free-sound', 'sound-bible'):
        d = SRC / 'musan' / 'musan' / 'noise' / sub
        ann = {}
        a = d / 'ANNOTATIONS'
        if a.exists():
            for line in open(a, errors='ignore'):
                p = line.strip().split(None, 1)
                if p:
                    ann[p[0]] = p[1] if len(p) > 1 else ''
        vad = json.loads((SRC / 'musan_noise_vad.json').read_text()) if (SRC / 'musan_noise_vad.json').exists() else {}
        for f in sorted(d.glob('*.wav')):
            desc = ann.get(f.stem, '') + ' ' + f.stem
            if VOICE.search(desc) or vad.get(f.name, 0.0) >= 0.2:     # Silero VAD speech share (5 of 930 files)
                continue
            yield 'musan_noise', sub, f, 'CC-BY-4.0 (MUSAN)', None
    # MUSAN music without vocals
    acc = 0.0
    for d in sorted((SRC / 'musan' / 'musan' / 'music').glob('*')):
        a = d / 'ANNOTATIONS'
        if not a.exists():
            continue
        for line in open(a, errors='ignore'):
            p = line.strip().split()
            if len(p) < 3 or p[2].upper() != 'N':
                continue
            f = d / f'{p[0]}.wav'
            if f.exists() and acc < MUSIC_H * 3600:
                acc += sf.info(str(f)).duration
                yield 'musan_music', p[1] if len(p) > 1 else 'music', f, 'CC (MUSAN music, per file)', None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index, buf, off, shard, total = [], [], 0, 0, 0
    limit = 700 * 1024 * 1024 // 2          # samples per shard (about 700 MB)

    def flush():
        nonlocal buf, off, shard
        if buf:
            np.save(OUT / f'pub_{shard:02d}.npy', np.concatenate(buf))
            shard += 1
            buf, off = [], 0

    for src, cat, f, lic, row in items():
        if heldout(src, cat, f, row):
            continue
        try:
            x, sr = sf.read(str(f), dtype='float32', always_2d=False)
        except Exception as e:
            print('skip', f, e)
            continue
        x = to16(x.astype(np.float64), sr)
        if len(x) < SR // 4 or not np.isfinite(x).all() or np.max(np.abs(x)) < 1e-4:
            continue
        x = x / max(1e-9, np.max(np.abs(x))) * 0.9          # peak 0.9 (the mixer sets the level)
        q = np.clip(np.rint(x * 32767), -32768, 32767).astype(np.int16)
        if off + len(q) > limit:
            flush()
        index.append({'source': src, 'category': cat, 'file': str(f.relative_to(SRC)), 'shard': f'pub_{shard:02d}.npy',
                      'offset': off, 'length': int(len(q)), 'licence': lic})
        buf.append(q)
        off += len(q)
        total += len(q)
    flush()
    with open(OUT / 'index.jsonl', 'w') as fo:
        for r in index:
            fo.write(json.dumps(r) + '\n')
    summ = {'files': len(index), 'hours': round(total / SR / 3600, 3),
            'by_source_hours': {s: round(sum(r['length'] for r in index if r['source'] == s) / SR / 3600, 3)
                                for s in sorted({r['source'] for r in index})}}
    (OUT / 'NOISE_PUBLIC.json').write_text(json.dumps(summ, indent=1))
    dl = SRC / 'DOWNLOADS.tsv'
    if dl.exists():
        (OUT / 'DOWNLOADS.tsv').write_text(dl.read_text())
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True, help='folder with the downloaded DEMAND, ESC-50 and MUSAN data')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    SRC, OUT = a.src, a.out
    main()
