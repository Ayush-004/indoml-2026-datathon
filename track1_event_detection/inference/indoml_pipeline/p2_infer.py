"""P2 per-frame inference on raw audio, optionally at a prepended-zero input shift.

The P2 model is a fine-tuned ATST-Frame encoder with a RED-style interval head on
seven noise-category channels. This module loads the checkpoint through its original
loader (atst_red/context_crop.load_parent), runs the inference path of the `indoml`
package (atst_red/indoml), and writes the raw per-frame outputs to an npz that
`refine.py` consumes:

  clip_ids   (N,)      object       durations  (N,)     float64 (of the SHIFTED wav)
  offsets    (N+1,)    int64        presence   (F, 7)   float32
  distances  (F, 7, 2) float32      offsets[k]:offsets[k+1] is clip k's frame range

Run this module with torch 2.8 / torchaudio 2.8 (see requirements.txt), with T1_NATIVE_BUNDLE and
T1_P2_CHECKPOINT set (common.py):

  python -m indoml_pipeline.p2_infer --audio DIR --out frames.npz [--shift 160]

A shift of k samples prepends k zeros to every waveform before inference; the shift
is removed again in refine.py, which is where the unshifted events are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import common
from .common import SR

ATST_RED = Path(__file__).resolve().parents[2] / 'atst_red'   # indoml package, context_crop.py, mixed_training.py
NATIVE = common.NATIVE_BUNDLE
CHECKPOINT = common.P2_CHECKPOINT
CHECKPOINT_SHA = '419c441e911419b33fa066d67ffa43bb63a8c6b3c659175ccb565f0ecfb83373'
STATE_SHA = '3fa453a39509b804f80e628c53643c346824373b63c81b0deb5d877bc14a64e9'

WINDOW_FRAMES = 250
STRIDE_FRAMES = 125
WINDOW_CAP = 4


def _prepare_sys_path():
    for entry in (str(NATIVE / 'source'), str(ATST_RED)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def api_module():
    _prepare_sys_path()
    from indoml import atst_intervals as api
    return api


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_p2(device='cpu'):
    """Return (model, metadata) for the frozen P2 checkpoint, via its original loader."""
    _prepare_sys_path()
    import context_crop
    api = api_module()
    model, metadata = context_crop.load_parent(api, NATIVE, CHECKPOINT, device)
    assert api.state_digest(model.state_dict()) == STATE_SHA, 'P2 weights differ'
    model.eval()
    model.frequency_augmentation = None
    return model, metadata


def row_for(clip_id, audio_path, duration):
    """The exact inference row shape the organizer inference path builds."""
    return {'clip_id': clip_id, 'audio_path': str(audio_path), 'duration': duration,
            'num_frames': (round(duration * SR) + 639) // 640,
            'frame_hop': 0.04, 'frame_offset': 0.02, 'feature_dim': 768}


def packs(rows, cap=WINDOW_CAP):
    """At most `cap` native 10 s windows per forward pass."""
    current, windows = [], 0
    for row in rows:
        count = 1 + max(0, (row['num_frames'] - WINDOW_FRAMES + STRIDE_FRAMES - 1) // STRIDE_FRAMES)
        assert count <= cap, 'clip exceeds the native window cap'
        if current and windows + count > cap:
            yield current
            current, windows = [], 0
        current.append(row)
        windows += count
    if current:
        yield current


def write_shifted(source_wavs, destination, shift):
    """Prepend `shift` zero samples to each wav and write a PCM_16 copy into `destination`."""
    import soundfile as sf
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for wav in source_wavs:
        target = destination / Path(wav).name
        if not target.exists():
            signal, _ = sf.read(wav, dtype='float32')
            if shift:
                signal = np.concatenate([np.zeros(shift, dtype=np.float32), signal])
            sf.write(target, signal, SR, subtype='PCM_16')
        paths.append(target)
    return paths


def frames_for(audio_paths, out_npz, *, shift=0, threads=None, cap=WINDOW_CAP, device='cpu'):
    """Run P2 over `audio_paths` (already shifted) and write the per-frame npz."""
    import soundfile as sf
    import torch
    threads = threads or common.threads()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    api = api_module()
    started = time.time()
    model, _ = load_p2(device)
    load_seconds = time.time() - started

    rows = []
    for path in audio_paths:
        info = sf.info(str(path))
        rows.append(row_for(Path(path).name, path, info.frames / info.samplerate))

    ids, durations, presence, distances = [], [], [], []
    started = time.time()
    with torch.inference_mode():
        for selected in packs(rows, cap):
            for row, values in api.infer_outputs(model, selected, len(selected)):
                ids.append(row['clip_id'])
                durations.append(float(row['duration']))
                presence.append(np.asarray(values['presence'], dtype=np.float32))
                distances.append(np.asarray(values['distances'], dtype=np.float32))
    seconds = time.time() - started

    offsets = np.concatenate([[0], np.cumsum([len(p) for p in presence])]).astype(np.int64)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, clip_ids=np.asarray(ids, dtype=object),
             durations=np.asarray(durations), offsets=offsets,
             presence=np.concatenate(presence), distances=np.concatenate(distances))
    return {'npz': str(out_npz), 'clips': len(ids), 'frames': int(offsets[-1]), 'shift': shift,
            'threads': threads, 'model_load_seconds': round(load_seconds, 3),
            'inference_seconds': round(seconds, 1), 'torch_version': torch.__version__,
            'checkpoint_sha256': CHECKPOINT_SHA}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audio', type=Path, required=True,
                        help='folder of unshifted 16 kHz wav files')
    parser.add_argument('--out', type=Path, required=True, help='frames npz to write')
    parser.add_argument('--shift', type=int, default=0, help='zero samples to prepend')
    parser.add_argument('--shift-dir', type=Path, default=None,
                        help='where the shifted wav copies go (default: <out>.shifted)')
    parser.add_argument('--clips', type=Path, default=None,
                        help='optional file listing the clip filenames to run')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--keep-shifted', action='store_true',
                        help='keep the shifted wav copies instead of deleting them afterwards')
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args(argv)

    wavs = sorted(p for p in Path(args.audio).rglob('*.wav'))
    if args.clips:
        wanted = {line.strip() for line in args.clips.read_text().splitlines() if line.strip()}
        wavs = [p for p in wavs if p.name in wanted]
    if not wavs:
        raise SystemExit(f'no wav files under {args.audio}')

    if args.shift:
        shift_dir = args.shift_dir or Path(str(args.out) + '.shifted')
        paths = write_shifted(wavs, shift_dir, args.shift)
    else:
        shift_dir, paths = None, wavs

    report = frames_for(paths, args.out, shift=args.shift, threads=args.threads)
    if shift_dir is not None and not args.keep_shifted:
        for path in paths:
            path.unlink(missing_ok=True)
        try:
            shift_dir.rmdir()
        except OSError:
            pass
        report['shifted_audio'] = 'deleted after inference'
    print(json.dumps(report, indent=1), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
