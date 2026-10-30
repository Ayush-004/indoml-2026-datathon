"""Build a Track 2 submission ZIP from a folder of enhanced WAVs (this script does not upload anything).

Contract (Codabench 17835 Evaluation page):
  * one enhanced WAV per test clip, PCM 16-bit, 16 kHz, mono, the SAME number of samples as the input clip,
    file name = clip id (input file name); a missing WAV costs -50 dB for that clip;
  * transcripts.jsonl with {"clip_id", "text"} for EVERY clip, from the mandated ASR ARTPARK-IISc/SraVaani-1.0 on the
    enhanced WAV (the hidden audit transcribes a subset again); an empty or missing transcript counts as WER 1.0;
  * all files at the ZIP root.
Transcripts: taken from --hyp (JSONL made by inference/sravaani_asr.py on these exact enhanced WAVs) after a check that
every clip is covered; the builder refuses a hyp file that is older than any enhanced WAV.

usage: build_submission.py --input-dir TEST_WAVS --enhanced-dir ENH --hyp HYP.jsonl --out OUT.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
import zipfile
from pathlib import Path

import numpy as np


def wav_info(p):
    with wave.open(str(p), 'rb') as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate()), w.getnframes(), w.readframes(w.getnframes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--enhanced-dir', type=Path, required=True)
    ap.add_argument('--hyp', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    inputs = sorted(a.input_dir.glob('*.wav'))
    if not inputs:
        raise SystemExit('no input WAVs')
    hyp = {}
    for l in a.hyp.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            hyp[r['clip_id']] = r['text']
    problems, report = [], {'clips': len(inputs), 'empty_text': 0, 'silent_wavs': 0, 'max_abs': 0}
    newest_wav = 0.0
    for p in inputs:
        e = a.enhanced_dir / p.name
        if not e.exists():
            problems.append(f'missing enhanced WAV {p.name}')
            continue
        newest_wav = max(newest_wav, e.stat().st_mtime)
        fi, ni, _ = wav_info(p)
        fe, ne, raw = wav_info(e)
        if fe != (1, 2, 16000):
            problems.append(f'bad format {p.name}: {fe}')
        if ne != ni:
            problems.append(f'length {p.name}: enhanced {ne} vs input {ni}')
        y = np.frombuffer(raw, dtype='<i2')
        if not np.any(y):
            report['silent_wavs'] += 1
        report['max_abs'] = max(report['max_abs'], int(np.max(np.abs(y.astype(np.int32)))) if len(y) else 0)
        if p.stem not in hyp:
            problems.append(f'no transcript for {p.stem}')
        elif not hyp[p.stem].strip():
            report['empty_text'] += 1
    extra = sorted(set(hyp) - {p.stem for p in inputs})
    if extra:
        problems.append(f'{len(extra)} transcripts for clips that are not in the input (e.g. {extra[:2]})')
    if a.hyp.stat().st_mtime < newest_wav:
        problems.append('the transcript file is older than an enhanced WAV: transcribe the final WAVs again')
    if problems:
        print('\n'.join(problems[:30]))
        raise SystemExit(f'{len(problems)} problems: no ZIP written')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix('.tmp.zip')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in inputs:
            z.write(a.enhanced_dir / p.name, p.name)
        lines = [json.dumps({'clip_id': p.stem, 'text': hyp[p.stem]}, ensure_ascii=False) for p in inputs]
        z.writestr('transcripts.jsonl', '\n'.join(lines) + '\n')
    tmp.replace(a.out)
    h = hashlib.sha256()
    with open(a.out, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    report.update({'zip': str(a.out), 'zip_bytes': a.out.stat().st_size, 'zip_sha256': h.hexdigest()})
    with zipfile.ZipFile(a.out) as z:
        names = z.namelist()
    report['zip_members'] = len(names)
    report['root_only'] = all('/' not in n for n in names)
    a.out.with_suffix('.report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
