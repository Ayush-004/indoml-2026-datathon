"""SraVaani-1.0 transcription with the mandated call (Track 2).

Mandated call (organizer instructions): AutoModel.from_pretrained(<SraVaani-1.0>, trust_remote_code=True),
then model.transcribe(wav_paths, return_hypotheses=True), then h.text.strip().
--model-dir is the Hugging Face id (default) or a local snapshot folder of it (we used commit f5dd5358).

Output: JSONL lines {"clip_id": <wav stem>, "text": <text>} (+ "sec" timing in a side file).
Resume: clip ids that are already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

MODEL_DIR = Path('ARTPARK-IISc/SraVaani-1.0')   # Hugging Face id (access request, approved automatically) or a local snapshot folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-dir', type=Path, help='folder of .wav files')
    ap.add_argument('--list', type=Path, help='text file: one wav path per line (overrides --wav-dir)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=8, help='transcribe() batch_size (default of the model code: 8)')
    ap.add_argument('--chunk', type=int, default=64, help='clips per transcribe() call (write + resume unit)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-per-run', type=int, default=0, help='exit after this many clips (fresh process = no '
                    'allocator growth); run it again to resume')
    ap.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel
    torch.set_num_threads(a.threads)

    if a.list:
        paths = [Path(l.strip()) for l in a.list.read_text().splitlines() if l.strip()]
    else:
        paths = sorted(a.wav_dir.glob('*.wav'))
    if a.limit:
        paths = paths[:a.limit]
    done = set()
    if a.out.exists():
        for l in a.out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)['clip_id'])
    todo = [p for p in paths if p.stem not in done]
    if a.max_per_run:
        todo = todo[:a.max_per_run]
    print(f'[asr] {len(paths)} clips, {len(done)} done, {len(todo)} to do, device={a.device}, threads={a.threads}, '
          f'batch={a.batch_size}', flush=True)
    if not todo:
        return

    t0 = time.time()
    model = AutoModel.from_pretrained(str(a.model_dir), trust_remote_code=True).to(a.device).eval()
    print(f'[asr] model loaded in {time.time() - t0:.1f}s', flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    timing = a.out.with_suffix('.timing.jsonl')
    n = 0
    t_all = time.time()
    with open(a.out, 'a') as fo, open(timing, 'a') as ft:
        for s in range(0, len(todo), a.chunk):
            part = todo[s:s + a.chunk]
            t1 = time.time()
            hyps = model.transcribe([str(p) for p in part], batch_size=a.batch_size, return_hypotheses=True)
            dt = time.time() - t1
            for p, h in zip(part, hyps):
                fo.write(json.dumps({'clip_id': p.stem, 'text': h.text.strip()}, ensure_ascii=False) + '\n')
            fo.flush()
            ft.write(json.dumps({'first': part[0].stem, 'n': len(part), 'sec': round(dt, 3)}) + '\n')
            ft.flush()
            n += len(part)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS: bytes -> MB
            print(f'[asr] {n}/{len(todo)} clips, {dt:.1f}s for this chunk, maxrss {rss:.0f} MB', flush=True)
    print(f'[asr] done {n} clips in {time.time() - t_all:.1f}s', flush=True)


if __name__ == '__main__':
    main()
