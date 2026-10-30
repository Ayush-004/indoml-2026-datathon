"""Mac frame exporter: per-frame outputs of a Mac checkpoint at the 4 input shifts, in the P2 member format.

The inference path is the pipeline's (inference/indoml_pipeline/p2_infer.py, as inference/export_frames.py
uses it): P2's architecture from its own loader (p2_infer.load_p2 -> context_crop.load_parent), then the
checkpoint's model_state (fp32 state files; fp16 checkpoints are cast to fp32), strict. For each split and shift:
p2_infer.write_shifted (zeros prepended, PCM_16), p2_infer.row_for with the shifted wav's header duration,
p2_infer.packs (at most 4 native windows per forward), api.infer_outputs, and the npz written exactly as
p2_infer.frames_for writes it:
    clip_ids (N,) object, durations (N,) float64 (of the SHIFTED wav), offsets (N+1,) int64,
    presence (F, 7) float32, distances (F, 7, 2) float32
so that the member fusion (refine_one -> fuse -> build_submission) reads the folder as a member.
Clip lists: val = every wav in INDOML_WORK/audio/val (1,807: the 1,806 organizer natural clips + 1 synthetic clip
the file-name router sends to the natural branch; the list of INDOML_WORK/val and of every Kaggle member), test =
every wav in INDOML_WORK/audio/test (3,435 natural-routed clips; the list of the R10 member's test frames,
T1_TEST_REFERENCE_FRAMES). The fold split belongs to the twin arms P2T / S1T (twin_train.py, not part of this
release; no released model uses them). Shift 0 reads the file, shifts 160/320/480 read the PCM_16 copies of
p2_infer.write_shifted.
Default splits: 'val,fold' for a --run-dir whose training-summary.json arm is a twin arm (P2T, S1T),
else 'val,test' (unchanged). --splits overrides the default.
Encoder on MPS (--encoder-device mps), head on the CPU, fp32, inference mode.

Precision: --amp none|fp16|bf16 (default none). Exports stay fp32 unless tested equal: with --amp fp16|bf16 the
exporter first computes the frames of the first --amp-probe-clips clips (first split, shift 0) in fp32 and with
autocast (psr10_port.encoder_autocast: the encoder only, the head fp32); it uses autocast only when both agree
within EXPORT_EQUAL (the tolerance of tests/test_export.py x1: max |presence| and |distance| difference <= 1e-4),
else it exports in fp32. The decision and the numbers are in export-report.json ('amp'). The supervisor always
exports with --amp none.

  export_frames.py --checkpoint PATH|p2 --out DIR [--splits val,test] [--shifts 0,160,320,480] --encoder-device mps
  export_frames.py --run-dir runs/M0 --which D,T --encoder-device mps     (the supervisor form: plateau-<D>.pt and
                   final-<T>.pt of the run -> runs/M0/export/u<update>/{val,test}/)
Each (split, shift) npz is written atomically and skipped when present (resumable). Exit codes: 0 done,
12 memory guard, 13 GPU_REQUEST, 14 another MPS process, 16 STOP, 2 refused, 1 error.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psr10_port as P  # noqa: E402

PIPE = HERE.parents[1] / 'inference'
WORK = Path(os.environ.get('INDOML_WORK', PIPE / 'work'))
AUDIO = WORK / 'audio'
V2FOLD = Path(os.environ.get('T1_TWIN_FOLD', 'twin-fold'))          # twin arms only (not part of this release)
SPLIT_AUDIO = {'val': AUDIO / 'val', 'test': AUDIO / 'test', 'fold': V2FOLD / 'audio-fold'}
REFERENCE_LISTS = {'val': WORK / 'val' / 'frames_shift0.npz',
                   'test': Path(os.environ.get('T1_TEST_REFERENCE_FRAMES', WORK / 'test' / 'frames_shift0.npz')),
                   'fold': V2FOLD / 'fold_ids.txt'}
TWIN_ARMS = ('P2T', 'S1T')          # twin runs export val + fold by default
SHIFTS = (0, 160, 320, 480)
EXPORT_EQUAL = {'presence': 1e-4, 'distances': 1e-4}     # "equal frames" (tests/test_export.py x1)
EXIT = {'complete': 0, 'guard': 12, 'gpu_request': 13, 'mps_busy': 14, 'stop_file': 16, 'refused': 2, 'error': 1}
STARTED = time.monotonic()


def log(message, **details):
    item = {'utc': P.utc(), 'elapsed_s': round(time.monotonic() - STARTED, 1), 'message': message}
    item.update(details)
    print(json.dumps(P.clean(item), allow_nan=False, default=str), flush=True)


class Pause(Exception):
    def __init__(self, code, reason):
        super().__init__(reason)
        self.code, self.reason = code, reason


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default=None, help="a state/checkpoint .pt, or 'p2' for the parent weights")
    parser.add_argument('--out', default=None)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--which', default='D,T')
    parser.add_argument('--splits', default=None,
                        help="default: 'val,fold' for a twin run (P2T, S1T), else 'val,test'")
    parser.add_argument('--shifts', default='0,160,320,480')
    parser.add_argument('--encoder-device', required=True, choices=('mps', 'cpu'))
    parser.add_argument('--head-device', default='cpu', choices=('cpu',))
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--clips', default=None, help='tests: a file with the clip names to export')
    parser.add_argument('--amp', default='none', choices=P.AMP_MODES,
                        help='autocast for the export; used only if the probe frames equal fp32 (else fp32)')
    parser.add_argument('--amp-probe-clips', type=int, default=20)
    parser.add_argument('--stop-path', default=str(P.STOP))
    parser.add_argument('--gpu-request-path', default=str(P.GPU_REQUEST))
    return parser.parse_args(argv)


def clip_paths(split, wanted=None):
    wavs = sorted(p for p in SPLIT_AUDIO[split].rglob('*.wav'))
    if wanted is not None:
        wavs = [p for p in wavs if p.name in wanted]
    return wavs


def reference_list(split):
    import numpy as np
    path = REFERENCE_LISTS[split]
    if split == 'fold':
        import twin_train
        if P.sha256_file(path) != twin_train.FOLD_IDS_SHA256:
            raise RuntimeError(f'{path} is not the pinned fold id file (twin_train.FOLD_IDS_SHA256)')
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [str(c) for c in np.load(path, allow_pickle=True)['clip_ids']]


def default_splits(run_dir):
    """'val,fold' for a twin run folder (training-summary.json arm P2T or S1T), else 'val,test'."""
    if run_dir:
        summary = Path(run_dir) / 'training-summary.json'
        if summary.is_file() and json.loads(summary.read_text()).get('arm') in TWIN_ARMS:
            return 'val,fold'
    return 'val,test'


def frames_of(model, api, p2_infer, rows, amp):
    """{clip: (presence, distances)} of rows (inference mode, packs of at most 4 native windows)."""
    import numpy as np
    import torch
    out = {}
    with torch.inference_mode(), P.encoder_autocast(model, amp):
        for pack in p2_infer.packs(rows):
            for row, values in api.infer_outputs(model, pack, len(pack)):
                out[row['clip_id']] = (np.asarray(values['presence'], dtype=np.float32),
                                       np.asarray(values['distances'], dtype=np.float32))
    return out


def amp_probe(model, api, p2_infer, split, wanted, amp, clips):
    """fp32 vs autocast frames on the first `clips` clips of `split` (shift 0). Equal -> the export may use amp."""
    import soundfile as sf
    rows = []
    for path in clip_paths(split, wanted)[:clips]:
        header = sf.info(str(path))
        rows.append(p2_infer.row_for(path.name, path, header.frames / header.samplerate))
    started = time.time()
    reference = frames_of(model, api, p2_infer, rows, 'none')
    fp32_seconds = time.time() - started
    started = time.time()
    candidate = frames_of(model, api, p2_infer, rows, amp)
    amp_seconds = time.time() - started
    presence = max(float(abs(reference[c][0] - candidate[c][0]).max()) for c in reference)
    distances = max(float(abs(reference[c][1] - candidate[c][1]).max()) for c in reference)
    equal = presence <= EXPORT_EQUAL['presence'] and distances <= EXPORT_EQUAL['distances']
    return {'requested': amp, 'split': split, 'clips': len(rows), 'max_abs_presence': presence,
            'max_abs_distances': distances, 'tolerance': EXPORT_EQUAL, 'equal': equal,
            'fp32_seconds': round(fp32_seconds, 2), 'amp_seconds': round(amp_seconds, 2),
            'used': amp if equal else 'none'}


def load_model(checkpoint, encoder_device, head_device):
    import torch
    sys.path.insert(0, str(PIPE))
    from indoml_pipeline import p2_infer
    model, metadata = p2_infer.load_p2('cpu')                 # P2 architecture + P2 bytes (digest asserted)
    api = p2_infer.api_module()
    info = {'source': 'parent-p2.pt', 'state_digest': api.state_digest(model.state_dict())}
    if checkpoint != 'p2':
        payload = P.load_weights(model, checkpoint)
        info = {'source': str(checkpoint), 'kind': payload.get('kind'), 'updates': payload.get('updates'),
                'dtype': payload.get('dtype', 'float32'), 'sha256': P.sha256_file(checkpoint),
                'state_digest': api.state_digest(model.state_dict())}
    model.eval()
    model.frequency_augmentation = None
    if encoder_device != 'cpu':
        model.set_device(torch.device(encoder_device), torch.device(head_device))
    return model, api, p2_infer, info


def export(model, api, p2_infer, info, out, splits, shifts, *, mps, wanted, guard, stop_path, gpu_request,
           amp='none'):
    import numpy as np
    import soundfile as sf
    import torch
    reports = {}
    for split in splits:
        wavs = clip_paths(split, wanted)
        names = [p.name for p in wavs]
        if wanted is None and names != reference_list(split):
            raise RuntimeError(f'{split}: the audio folder list differs from the reference member list')
        for shift in shifts:
            target = out / split / f'frames_shift{shift}.npz'
            if target.is_file():
                reports[f'{split}/{shift}'] = 'present'
                continue
            if stop_path.exists():
                raise Pause(EXIT['stop_file'], 'STOP file')
            if mps and gpu_request.exists():
                raise Pause(EXIT['gpu_request'], 'GPU_REQUEST')
            target.parent.mkdir(parents=True, exist_ok=True)
            shift_dir = out / split / f'shifted{shift}'
            if shift:
                if shift_dir.exists():
                    shutil.rmtree(shift_dir)
                paths = p2_infer.write_shifted(wavs, shift_dir, shift)
            else:
                paths = wavs
            rows = []
            for path in paths:
                header = sf.info(str(path))
                rows.append(p2_infer.row_for(Path(path).name, path, header.frames / header.samplerate))
            ids, durations, presence, distances = [], [], [], []
            started = time.time()
            with torch.inference_mode(), P.encoder_autocast(model, amp):
                for index, pack in enumerate(p2_infer.packs(rows)):
                    for row, values in api.infer_outputs(model, pack, len(pack)):
                        ids.append(row['clip_id'])
                        durations.append(float(row['duration']))
                        presence.append(np.asarray(values['presence'], dtype=np.float32))
                        distances.append(np.asarray(values['distances'], dtype=np.float32))
                    if index % 50 == 0:
                        if stop_path.exists():
                            raise Pause(EXIT['stop_file'], 'STOP file')
                        if mps:
                            torch.mps.empty_cache()
                            reason = guard.check('export')
                            if reason:
                                raise Pause(EXIT['guard'], reason)
            seconds = time.time() - started
            offsets = np.concatenate([[0], np.cumsum([len(p) for p in presence])]).astype(np.int64)
            temporary = target.parent / (target.stem + '.tmp.npz')
            np.savez(temporary, clip_ids=np.asarray(ids, dtype=object), durations=np.asarray(durations),
                     offsets=offsets, presence=np.concatenate(presence), distances=np.concatenate(distances))
            temporary.replace(target)
            if shift:
                shutil.rmtree(shift_dir, ignore_errors=True)
            report = {'npz': str(target), 'clips': len(ids), 'frames': int(offsets[-1]), 'shift': shift,
                      'inference_seconds': round(seconds, 1), 'device': 'mps' if mps else 'cpu',
                      'torch_version': torch.__version__, 'checkpoint': info, 'amp': amp}
            (target.parent / f'p2-shift{shift}.json').write_text(json.dumps(report, indent=1) + '\n')
            reports[f'{split}/{shift}'] = report
            log('exported', split=split, shift=shift, clips=len(ids), seconds=round(seconds, 1),
                mps_driver_gib=((guard.snapshot or {}).get('mps_driver_bytes', 0) / 2 ** 30) if guard else None)
    return reports


def run_targets(run_dir, which):
    """[(update, state path)] for D and/or T of a finished run."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'training-summary.json').read_text())
    T, D = summary['T'], summary['D']
    out = []
    for key in which:
        if key == 'D' and D < T:
            out.append((D, run_dir / f'plateau-{D:06d}.pt'))
        elif key == 'T':
            out.append((T, run_dir / f'final-{T:06d}.pt'))
    return out


def main(argv=None):
    args = parse(argv)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    mps = args.encoder_device == 'mps'
    stop_path, gpu_request = Path(args.stop_path), Path(args.gpu_request_path)
    if bool(args.checkpoint) == bool(args.run_dir):
        log('refused', error='give --checkpoint (+ --out) or --run-dir')
        return EXIT['refused']
    if stop_path.exists():
        log('paused', reason='STOP file present at start')
        return EXIT['stop_file']
    if mps and gpu_request.exists():
        log('paused', reason='GPU_REQUEST present at start')
        return EXIT['gpu_request']
    if args.run_dir:
        targets = [(u, path, Path(args.run_dir) / 'export' / f'u{u:06d}')
                   for u, path in run_targets(args.run_dir, [w.strip() for w in args.which.split(',') if w.strip()])]
    else:
        targets = [(None, args.checkpoint, Path(args.out))]
    for _, path, _ in targets:
        if path != 'p2' and not Path(path).is_file():
            log('refused', error=f'missing checkpoint {path}')
            return EXIT['refused']
    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    guard = P.Guard(torch, use_mps=mps) if mps else None
    if mps:
        busy = P.other_mps_processes({os.getpid()})
        if busy:
            time.sleep(5)
            again = P.other_mps_processes({os.getpid()})
            if {b['pid'] for b in busy} & {b['pid'] for b in again}:
                log('paused', reason='another MPS process at start', processes=again)
                return EXIT['mps_busy']
        reason = guard.check('export_start', force=True)
        if reason:
            log('paused', reason=reason)
            return EXIT['guard']
    wanted = None
    if args.clips:
        wanted = {line.strip() for line in Path(args.clips).read_text().splitlines() if line.strip()}
    splits_text = args.splits if args.splits is not None else default_splits(args.run_dir)
    splits = [s.strip() for s in splits_text.split(',') if s.strip()]
    if any(s not in SPLIT_AUDIO for s in splits):
        log('refused', error=f'splits must be in {sorted(SPLIT_AUDIO)}')
        return EXIT['refused']
    shifts = [int(s) for s in args.shifts.split(',') if s.strip()]
    if any(s not in SHIFTS for s in shifts):
        log('refused', error=f'shifts must be in {SHIFTS}')
        return EXIT['refused']
    for update, path, out in targets:
        done = all((out / split / f'frames_shift{k}.npz').is_file() for split in splits for k in shifts)
        if done:
            log('already_exported', out=str(out))
            continue
        model, api, p2_infer, info = load_model(path, args.encoder_device, args.head_device)
        log('model_ready', checkpoint=str(path), update=update, info=info, out=str(out))
        precision = {'requested': 'none', 'used': 'none', 'probe': None}
        if args.amp != 'none':
            probe = amp_probe(model, api, p2_infer, splits[0], wanted, args.amp, args.amp_probe_clips)
            precision = {'requested': args.amp, 'used': probe['used'], 'probe': probe}
            log('amp_probe', **probe)
        try:
            reports = export(model, api, p2_infer, info, out, splits, shifts, mps=mps, wanted=wanted, guard=guard,
                             stop_path=stop_path, gpu_request=gpu_request, amp=precision['used'])
        except Pause as pause:
            log('paused', reason=pause.reason)
            return pause.code
        P.write_json(out / 'export-report.json', {'utc': P.utc(), 'checkpoint': info, 'update': update,
                                                  'amp': precision,
                                                  'splits': splits, 'shifts': shifts, 'reports': reports,
                                                  'apply_multi': 'member fusion: <split> <out_dir> '
                                                                 f'{out}/<split> [other member folders]'})
        del model
        if mps:
            torch.mps.empty_cache()
    log('done', targets=[str(t[2]) for t in targets])
    return EXIT['complete']


if __name__ == '__main__':
    try:
        code = main()
    except Exception as error:
        log('failed', error=repr(error), traceback=traceback.format_exc()[-3000:])
        code = EXIT['error']
    sys.exit(code)
