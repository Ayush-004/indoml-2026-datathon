"""Silero VAD speech share of every bank noise event (noise_bank.jsonl of build_banks.py).

Why: the organizer noise pieces hold few speech frames, but many Vaani noise events hold speech (the noise is under
or next to the speaker). Training on speech-bearing noise teaches the enhancer to remove speech. The training scripts
keep only events with vad_speech < T2_VAD_MAX (default 0.2).
Method: the event span of the source clip (local Vaani copy), peak-normalised to 0.5, Silero VAD v5 in
32 ms frames (reset per event), share of frames with p >= 0.5. Events shorter than one frame get vad_speech = 1.0.
Output: OUT/noise_bank_vad.jsonl (one line per noise_bank.jsonl line: index i, clip_id, start, end, tag, kind,
vad_speech) + OUT/BANKVAD.json.

usage: bank_vad.py --vaani-dir VAANI --bank NOISE_BANK/noise_bank.jsonl --silero-jit silero_vad.jit --out OUT
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

V = BANK = SILERO = OUT = None    # set from the command line (see usage)


def main():
    torch.set_num_threads(2)
    vad = torch.jit.load(str(SILERO)).eval()
    man = {}
    for l in open(V / 'manifest.jsonl'):
        r = json.loads(l)
        man[r['clip_id']] = r['audio_path']
    bank = [json.loads(l) for l in open(BANK)]
    by_clip = defaultdict(list)
    for i, r in enumerate(bank):
        by_clip[r['clip_id']].append(i)
    res = [None] * len(bank)
    t0 = time.time()
    for j, (cid, idx) in enumerate(by_clip.items()):
        if cid not in man:
            continue
        x = sf.read(str(V / man[cid]), dtype='float32', always_2d=False)[0]
        x = x if x.ndim == 1 else x[:, 0]
        for i in idx:
            r = bank[i]
            seg = x[int(r['start'] * 16000):int(r['end'] * 16000)].astype(np.float32)
            nfr = len(seg) // 512
            if nfr < 1 or float(np.max(np.abs(seg))) <= 0:
                res[i] = 1.0
                continue
            seg = seg / float(np.max(np.abs(seg))) * 0.5
            vad.reset_states()
            with torch.no_grad():
                p = [float(vad(torch.from_numpy(seg[k * 512:(k + 1) * 512]), 16000)) for k in range(nfr)]
            res[i] = round(float(np.mean(np.array(p) >= 0.5)), 4)
        if (j + 1) % 2000 == 0:
            print(f'{j + 1}/{len(by_clip)} clips, {time.time() - t0:.0f}s', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'noise_bank_vad.jsonl', 'w') as f:
        for i, r in enumerate(bank):
            f.write(json.dumps({'i': i, 'clip_id': r['clip_id'], 'start': r['start'], 'end': r['end'], 'tag': r['tag'],
                                'kind': r['kind'], 'vad_speech': res[i]}) + '\n')
    summ = {'events': len(bank), 'missing_audio': sum(v is None for v in res)}
    for kind in ('empty_wrap', 'point_tag'):
        v = np.array([res[i] for i, r in enumerate(bank) if r['kind'] == kind and res[i] is not None])
        summ[kind] = {'n': int(len(v)), 'mean': round(float(v.mean()), 4), 'lt_0.2': int((v < 0.2).sum()),
                      'lt_0.1': int((v < 0.1).sum()), 'gt_0.5': int((v > 0.5).sum())}
    (OUT / 'BANKVAD.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vaani-dir', type=Path, required=True, help='local Vaani copy: manifest.jsonl + audio/')
    ap.add_argument('--bank', type=Path, required=True, help='noise_bank.jsonl of build_banks.py')
    ap.add_argument('--silero-jit', type=Path, required=True, help='silero_vad.jit (Silero VAD v5)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    V, BANK, SILERO, OUT = a.vaani_dir, a.bank, a.silero_jit, a.out
    sys.exit(main())
