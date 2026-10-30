"""Mac trainer (the R10S1 fallback donor of 950232): ONE kernel arm per process on the Mac (encoder on MPS, head on CPU; fp32 by default).

Recipe = the kernel's arm (training/kernels/kernel_psr10b.py, pinned by sha256; psr10_port.py lists what is used unchanged):
P2 init; per update 8 Gold + 2 generated (weight 1.0, interval_loss), 8 Silver (weight 0.4, the arm's Silver mode:
'full' for PSR10/PSR10B, 'R2' for the K-A arms R10/A3/A), 32 weak clip tags (weight 0.1; the arms with Bronze -
PSR10B, R10, A3, A - also the Bronze wrap-all Silver rows and the Bronze negative 0.1), the kernel's packing caps;
class weight 5, boundary 100, IoU 1, collar 0; AdamW wd 1e-4, 13 depth groups; encoder top peak 1e-5 (A3 3e-6,
A 1e-6) x 0.95^(11-depth) with warmup .1 -> 1 over 1,000 updates (A3, A: 200); head 1e-4 with warmup over 200;
plateau to D, cosine to .05 x peak at T; gradient clip 5; frequency warp p .5 scale .6-1; seeds gold 149, generated
249, silver 349, weak 51, torch 1052, warp 50. Full original clips, native overlapping 10 s windows stitched before
the recurrent head.

Run options: --arm PSR10|PSR10B|R10|A3|A (psr10_port.TRAIN_ARMS; the arm spec is the kernel's ARM_SPECS entry),
--seed-offset N (adds N to all six seeds), --silver-length-mix (M1, R10-L),
--silver-wholeclip-odds X (R10-W: genuine Silver clips with an event >= 90 % of the clip get X times the draw
probability of the other clips of their fixed clip-length bin, 1 < X <= 2; psr10_port.WholeClipOddsSilver; not with
--silver-length-mix), --encoder-peak X (M2; same fractional warmup and depth decay),
--generated-presence-only (R10G1: in the strong pass the generated rows keep their presence term and their boundary and
interval-IoU terms are 0; the Gold terms, the logical normalizers, the draws and the Silver/weak/Bronze terms do not
change; psr10_port.strong_group_loss_generated_presence_only; not with --silver-length-mix or --silver-wholeclip-odds),
--silver-targets FILE (R10-TC: the events and R2 flags of the genuine Silver rows come from FILE, a targets file
(not part of this release); the draws, the Bronze rows and the Gold/generated/weak/Bronze terms do not change; the file's sha256
is in the run config; K-A arms (Silver mode R2) only; psr10_port.apply_silver_targets; not with --silver-length-mix,
--silver-wholeclip-odds or --generated-presence-only),
--continue-from <plateau state> --T --D (M4:
restores model, AdamW, streams, warp, RNG and the draw chain at the saved update, then runs to the new T with the
new plateau end D), --max-updates N (stop at update N), --smoke (the supervisor's first stage).

Precision: --amp none|fp16|bf16 (default none = fp32). fp16 = autocast float16 on the encoder device + the kernel's
GradScaler(1024, growth 2000); bf16 = autocast bfloat16, no scaler (psr10_port.mac_update). --precision-file
runs/PRECISION.json (written ONCE by precision.py decide) sets the mode for every M run; a different --amp is
refused, and a full run in the real runs folder needs the file. The mode is part of the resume/continuation
contract, so a run never changes precision. --smoke --amp fp16|bf16 --ab-reference <fp32 smoke-report.json> is the
precision A/B: the same init and draws as that fp32 smoke, no evaluation, N updates, ab-<amp>/ab-report.json with
the fixed rule's numbers (the supervisor then runs precision.py decide).

Outputs in <root>/<run>/ (smoke: <root>/<run>/smoke/): run-contract.json, draws.jsonl, trajectory.json,
last.pt (every --last-every updates; full state), ckpt-<u>.pt (fp16 weights every --ckpt-every updates and at D
and T, as the kernel's checkpoints), plateau-<D>.pt (full state at D), final-<T>.pt (full state at T),
eval/eval-<u>.json + eval/pred-<u>-t065|t075.jsonl + eval/frames-<u>.npz (the kernel's evaluate on the 1,806
natural val clips at every --eval-every updates and at D and T; the kernel's kill rule raw t0.65 < 1.00 from u500
for the arms with kill_below: PSR10, PSR10B, R10),
training-summary.json at T. Smoke: smoke-report.json and <root>/<run>/SMOKE_OK when it passes. A/B:
<root>/<run>/ab-<amp>/ab-report.json (exit 0 whenever the report is written; the rule decides, not the exit code).

Exit codes: 0 complete (or --max-updates reached), 12 memory guard pause, 13 GPU_REQUEST pause, 14 another MPS
process, 15 session limit (tests), 16 STOP file, 17 draws differ from the kernel replay (fatal), 18 kill rule
(recipe failed; run ended), 20 smoke failed, 2 refused to start, 1 error. A pause saves last.pt first.

Twin arms: a command line with '--arm P2T' or '--arm S1T' goes to
twin_train.main() (not part of this release; no released model uses it) (the P2 full-release recipe and the S1 recipe, INNER-TRAIN Gold, no fold id in any stream, the K1
filter; same exit codes, files and smoke contract). Every other command line takes the unchanged path below.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import psr10_port as P  # noqa: E402

STARTED = time.monotonic()
EXIT = {'complete': 0, 'max_updates': 0, 'guard': 12, 'gpu_request': 13, 'mps_busy': 14, 'session_limit': 15,
        'stop_file': 16, 'draws_differ': 17, 'kill_rule': 18, 'smoke_failed': 20, 'refused': 2, 'error': 1}
RUN_NAME = re.compile(r'^[A-Z0-9_]+$')
TWIN_ARMS = ('P2T', 'S1T')          # twin arms: twin_train.py (the kernel arms never go there)


def twin_arm(argv):
    """True when the command line asks for a twin arm ('--arm P2T', '--arm=S1T')."""
    argv = list(argv)
    for index, item in enumerate(argv):
        if item == '--arm' and index + 1 < len(argv):
            return argv[index + 1] in TWIN_ARMS
        if item.startswith('--arm='):
            return item.split('=', 1)[1] in TWIN_ARMS
    return False


def log(message, **details):
    item = {'utc': P.utc(), 'elapsed_s': round(time.monotonic() - STARTED, 1), 'message': message}
    item.update(details)
    print(json.dumps(P.clean(item), allow_nan=False, default=str), flush=True)


class Refused(RuntimeError):
    pass


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='run name, [A-Z0-9_]+ (e.g. M0)')
    parser.add_argument('--arm', required=True, choices=P.TRAIN_ARMS)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--silver-length-mix', action='store_true')
    parser.add_argument('--silver-wholeclip-odds', type=float, default=None,
                        help='R10-W: whole-clip odds of the genuine Silver clips inside fixed clip-length bins '
                             '(e.g. 1.5; 1 < X <= 2; default: off = the kernel Silver stream)')
    parser.add_argument('--encoder-peak', type=float, default=None,
                        help='encoder top peak (default: the arm spec: 1e-5; A3 3e-6, A 1e-6)')
    parser.add_argument('--generated-presence-only', action='store_true',
                        help='R10G1: the generated rows of the strong pass keep their presence term; their boundary '
                             'and interval-IoU terms are 0 (default: off = the kernel loss)')
    parser.add_argument('--silver-targets', default=None,
                        help='R10-TC: a targets file with the events and R2 flags of the '
                             'genuine Silver rows (default: off = the kernel Silver pool)')
    parser.add_argument('--continue-from', default=None, help='a plateau-<D>.pt full state of a finished run')
    parser.add_argument('--T', type=int, default=15000)
    parser.add_argument('--D', type=int, default=None, help="default: the kernel's plateau_end(T)")
    parser.add_argument('--max-updates', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-updates', type=int, default=60)
    parser.add_argument('--root', default=str(P.REAL_RUNS))
    parser.add_argument('--encoder-device', required=True, choices=('mps', 'cpu'))
    parser.add_argument('--head-device', default='cpu', choices=('cpu',))
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--max-forward-windows', type=int, default=24,
                        help='microbatch budget in native windows per forward (0 = one group per pass)')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--last-every', type=int, default=250)
    parser.add_argument('--ckpt-every', type=int, default=1000)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--mps-check-every', type=int, default=100)
    parser.add_argument('--empty-cache-every', type=int, default=1)
    parser.add_argument('--stop-path', default=str(P.STOP))
    parser.add_argument('--gpu-request-path', default=str(P.GPU_REQUEST))
    parser.add_argument('--amp', default=None, choices=P.AMP_MODES,
                        help='none (fp32, default), fp16 (autocast + GradScaler) or bf16 (autocast); '
                             'default: the mode of --precision-file, else none')
    parser.add_argument('--precision-file', default=None, help='runs/PRECISION.json (precision.py decide)')
    parser.add_argument('--ab-reference', default=None,
                        help='with --smoke: the precision A/B against this fp32 smoke-report.json')
    # test hooks: refused for the real runs folder
    parser.add_argument('--test-no-autocast', action='store_true',
                        help='tests: the scaler path of --amp with an fp32 forward (CPU autocast is very slow)')
    parser.add_argument('--test-mix', default=None, choices=('probe',), help="tests: the kernel's PROBE_MIX")
    parser.add_argument('--test-eval-clips', type=int, default=0, help='tests: evaluate the first N val clips')
    parser.add_argument('--test-stop-after', type=int, default=0, help='tests: stop after N updates (exit 15)')
    parser.add_argument('--test-no-kernel-check', action='store_true')
    parser.add_argument('--test-u0-reference', default=None, help='tests: CPU u0 predictions to compare')
    parser.add_argument('--test-fail-smoke', action='store_true', help='tests: the smoke reports a failure')
    return parser.parse_args(argv)


def resolve_precision(args, real):
    """Set args.amp from --amp / --precision-file and return the precision-file record (or None)."""
    record = None
    if args.precision_file:
        path = Path(args.precision_file)
        if not path.is_file():
            raise Refused(f'--precision-file {path} does not exist')
        try:
            decision, digest = P.read_precision_file(path)
        except (ValueError, json.JSONDecodeError) as error:
            raise Refused(str(error))
        if args.amp is not None and args.amp != decision['amp']:
            raise Refused(f"--amp {args.amp} differs from {path} (amp {decision['amp']}): all M runs use one "
                          'precision')
        args.amp = decision['amp']
        record = {'path': str(path.resolve()), 'sha256': digest, 'decision': decision}
    elif args.amp is None:
        args.amp = 'none'
    if args.ab_reference:
        if not args.smoke:
            raise Refused('--ab-reference is the precision A/B of the smoke stage (needs --smoke)')
        if args.amp == 'none':
            raise Refused('the A/B needs --amp fp16 or bf16')
        if args.precision_file:
            raise Refused('the precision is already decided (--precision-file); no A/B after the decision')
        if not Path(args.ab_reference).is_file():
            raise Refused(f'--ab-reference {args.ab_reference} does not exist')
    if real:
        if not args.smoke and record is None:
            raise Refused('a full run in the real runs folder needs --precision-file (one precision for all M runs)')
        if args.smoke and record is None and not args.ab_reference and args.amp != 'none':
            raise Refused('a smoke before the precision decision is the fp32 reference (--amp none)')
        if args.ab_reference and P.PRECISION_FILE.exists():
            raise Refused(f'{P.PRECISION_FILE} exists: the precision is decided, no new A/B')
    return record


def check_arguments(args):
    root = Path(args.root).resolve()
    if not RUN_NAME.match(args.run):
        raise Refused('run names are [A-Z0-9_]+ (the supervisor final-line rule)')
    hooks = (args.test_mix or args.test_eval_clips or args.test_stop_after or args.test_no_kernel_check
             or args.test_fail_smoke or args.test_u0_reference or args.test_no_autocast)
    real = root == P.REAL_RUNS.resolve()
    args.precision_record = resolve_precision(args, real)
    if real:
        problems = []
        if args.encoder_device != 'mps':
            problems.append('real runs use --encoder-device mps')
        if hooks:
            problems.append('test hooks are refused for the real runs')
        if args.threads > 3:
            problems.append('at most 3 CPU threads')
        if problems:
            raise Refused('; '.join(problems))
    else:
        if P.TEST_ROOT.resolve() not in root.parents and root != P.TEST_ROOT.resolve():
            raise Refused('non-real runs must live under tests/')
        if args.encoder_device == 'mps':
            raise Refused('the tests never use MPS')
    if args.D is not None and not (0 < args.D <= args.T):
        raise Refused('need 0 < D <= T')
    odds = getattr(args, 'silver_wholeclip_odds', None)
    if odds is not None:
        if not P.WHOLECLIP['min_odds'] < odds <= P.WHOLECLIP['max_odds']:
            raise Refused(f"--silver-wholeclip-odds must be in ({P.WHOLECLIP['min_odds']}, {P.WHOLECLIP['max_odds']}]")
        if args.silver_length_mix:
            raise Refused('--silver-wholeclip-odds and --silver-length-mix are not combined (not tested)')
    if getattr(args, 'generated_presence_only', False) and (args.silver_length_mix or odds is not None):
        raise Refused('--generated-presence-only is not combined with --silver-length-mix or --silver-wholeclip-odds '
                      '(not tested)')
    if args.continue_from and not Path(args.continue_from).is_file():
        raise Refused(f'--continue-from {args.continue_from} does not exist')
    targets = getattr(args, 'silver_targets', None)
    if targets is not None:                             # R10-TC only (every other run: nothing below runs)
        if not Path(targets).is_file():
            raise Refused(f'--silver-targets {targets} does not exist')
        if args.arm not in P.KA_ARMS:
            raise Refused('--silver-targets needs a K-A arm with Silver mode R2 (R10, A3 or A)')
        if args.silver_length_mix or odds is not None or getattr(args, 'generated_presence_only', False):
            raise Refused('--silver-targets is not combined with --silver-length-mix, --silver-wholeclip-odds or '
                          '--generated-presence-only (not tested)')
        args.silver_targets = str(Path(targets).resolve())
        args.silver_targets_sha256 = P.sha256_file(args.silver_targets)


def resolved_config(ps, args, spec, seeds, pool_digest, mix, D):
    config = {'arm': args.arm, 'spec': ps.spec_core(spec), 'seed_offset': args.seed_offset, 'seeds': seeds,
              'silver_length_mix': (dict(P.LENGTH_MIX) if args.silver_length_mix else None),
              'mix': mix, 'caps': ps.CAPS, 'pool_digest': pool_digest, 'kernel_sha256': P.KERNEL_SHA256,
              'T': args.T, 'D': D, 'strong_weight': ps.STRONG_WEIGHT, 'weak_weight': ps.WEAK_WEIGHT,
              'bronze_negative_weight': ps.BRONZE_NEGATIVE_WEIGHT if spec['bronze'] else None,
              'original': ps.ORIGINAL, 'schedule': ps.SCHEDULE, 'test_mix': args.test_mix,
              'amp': args.amp, 'test_no_autocast': bool(args.test_no_autocast)}
    odds = getattr(args, 'silver_wholeclip_odds', None)
    if odds is not None:          # only when on: the config of every other run stays as before (resume contract)
        config['silver_wholeclip_odds'] = dict(P.WHOLECLIP, odds=odds, weight_seed_effective=(
            P.WHOLECLIP['weight_seed'] + args.seed_offset))
    if getattr(args, 'generated_presence_only', False):      # only when on (the same resume contract rule)
        config['generated_presence_only'] = dict(P.GENERATED_PRESENCE_ONLY)
    if getattr(args, 'silver_targets', None):                # R10-TC only (the same resume contract rule)
        digest = getattr(args, 'silver_targets_sha256', None) or P.sha256_file(args.silver_targets)
        config['silver_targets'] = dict(P.SILVER_TARGETS, sha256=digest)
    return config


def precision_contract(args, P):
    """The run-contract 'precision' field (also in every smoke and A/B report)."""
    autocast = None
    if args.amp != 'none' and not args.test_no_autocast:
        autocast = {'device_type': args.encoder_device, 'dtype': {'fp16': 'float16', 'bf16': 'bfloat16'}[args.amp],
                    'region': 'encoder.get_intermediate_layers (the ops the kernel autocast casts on CUDA)'}
    return {'amp': args.amp, 'autocast': autocast,
            'grad_scaler': (dict(P.KERNEL_GRAD_SCALER, device='head device (cpu)') if args.amp == 'fp16' else None),
            'test_no_autocast': bool(args.test_no_autocast),
            'head': 'fp32 (CPU; enforce_fp32_head)', 'evaluation': 'fp32 (the kernel evaluate has no autocast)',
            'kernel': 'CUDA fp16 autocast encoder + GradScaler(1024, growth 2000), fp32 head',
            'precision_file': args.precision_record, 'ab_reference': args.ab_reference}


def recipe_part(config):
    return {k: v for k, v in config.items() if k not in ('T', 'D')}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if twin_arm(argv):                   # twin arms; the kernel-arm path below does not change
        import twin_train
        return twin_train.main(argv)
    args = parse(argv)
    try:
        check_arguments(args)
    except Refused as error:
        log('refused', error=str(error))
        return EXIT['refused']
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    stop_path, gpu_request = Path(args.stop_path), Path(args.gpu_request_path)
    if stop_path.exists():
        log('paused', reason='STOP file present at start', path=str(stop_path))
        return EXIT['stop_file']
    mps = args.encoder_device == 'mps'
    if gpu_request.exists() and mps:
        log('paused', reason='GPU_REQUEST present at start', path=str(gpu_request))
        return EXIT['gpu_request']
    run_dir = Path(args.root) / args.run
    if args.ab_reference:
        out = run_dir / f'ab-{args.amp}'
    else:
        out = run_dir / 'smoke' if args.smoke else run_dir
    if not args.smoke:
        final_ready = (out / f'final-{args.T:06d}.pt').is_file() and (out / 'training-summary.json').is_file()
        if final_ready:
            log('already_complete', run=args.run, T=args.T)
            return EXIT['complete']
        if (out / 'killed.json').is_file():
            log('refused', error='this run ended by the kill rule (killed.json); start a new run name')
            return EXIT['kill_rule']
    if args.smoke and out.exists():
        shutil.rmtree(out)                                    # every smoke starts clean
    out.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import wavlm_mps_patch
    patch = wavlm_mps_patch.apply() if mps else {'applied': False, 'reason': 'CPU run'}
    guard = P.Guard(torch, use_mps=mps)
    if mps:
        reason = guard.check('before_load', force=True)
        if reason:
            log('paused', reason=reason)
            return EXIT['guard']

    # ------------------------------------------------------------------ kernel, pools, streams
    ps = P.load_kernel()
    started = time.monotonic()
    data, view = P.arm_data(ps, args.arm)
    census = view['arm_census']
    expected = P.POOL_SIZES[args.arm]
    if census['pool_digest'] != P.POOL_DIGESTS[args.arm] or any(census[k] != v for k, v in expected.items()):
        log('refused', error='arm pools differ from the kernel census', census=census)
        return EXIT['refused']
    pools = view['pools']
    targets_report = None
    if args.silver_targets:                             # R10-TC: the Silver events and R2 flags from the file
        try:
            view, targets_report = P.apply_silver_targets(ps, view, args.silver_targets)
        except ValueError as error:
            log('refused', error=f'--silver-targets: {error}')
            return EXIT['refused']
        if targets_report['sha256'] != args.silver_targets_sha256:
            log('refused', error='--silver-targets changed after the argument check')
            return EXIT['refused']
        pools = view['pools']
        log('silver_targets', **{k: v for k, v in targets_report.items() if k != 'rule'})
    missing_audio = sorted({r['clip_id'] for rows in pools.values() for r in rows
                            if not os.path.exists(r['audio_path'])})
    if missing_audio:
        log('refused', error='pool audio missing locally', clips=missing_audio[:20], count=len(missing_audio))
        return EXIT['refused']
    log('pools', seconds=round(time.monotonic() - started, 1), arm=args.arm, digest=census['pool_digest'],
        **{k: census[k] for k in ('gold', 'silver', 'generated', 'weak')})
    spec = P.effective_spec(ps, args.arm, args.encoder_peak)
    seeds = P.seeds_for(ps, args.seed_offset)
    mix = dict(ps.PROBE_MIX if args.test_mix == 'probe' else ps.MIXES['8+2+8'])
    D = args.D if args.D is not None else ps.plateau_end(args.T)
    config = resolved_config(ps, args, spec, seeds, census['pool_digest'], mix, D)
    streams = P.build_streams(ps, pools, seeds, args.seed_offset, args.silver_length_mix, spec['bronze'], mix,
                              wholeclip_odds=args.silver_wholeclip_odds)
    length_census = streams['silver'].census() if args.silver_length_mix else None
    kernel_check = None
    if not args.test_no_kernel_check:
        kernel_check = P.check_draws_against_kernel(ps, pools, args.arm, args.seed_offset, args.silver_length_mix,
                                                    wholeclip_odds=args.silver_wholeclip_odds)
        log('kernel_draw_check', **kernel_check)
        if not kernel_check['pass']:
            return EXIT['draws_differ']

    own = {os.getpid()}
    if mps:
        busy = P.other_mps_processes(own)
        if busy:
            time.sleep(5)
            again = P.other_mps_processes(own)
            if {b['pid'] for b in busy} & {b['pid'] for b in again}:
                log('paused', reason='another MPS process at start', processes=again)
                return EXIT['mps_busy']

    # ------------------------------------------------------------------ model, optimizer, warp
    model, api, init_meta = P.load_model(ps, args.encoder_device, args.head_device)
    if mps:
        reason = guard.check('after_model', force=True)
        if reason:
            log('paused', reason=reason)
            return EXIT['guard']
    optimizer, layout = ps.depth_optimizer(model, api, ps.ORIGINAL['optimizer']['weight_decay'],
                                           ps.ORIGINAL['construction_lr']['encoder_lr'],
                                           ps.ORIGINAL['construction_lr']['head_lr'])
    fw = ps.ORIGINAL['frequency_warp']
    warp = api.FrequencyWarpStream(fw['probability'], fw['min_scale'], fw['max_scale'], seeds['warp'])
    class_weights = torch.full((len(ps.CATEGORIES),), ps.ORIGINAL['strong_loss']['class_positive_weight'],
                               dtype=torch.float32, device=model.head_device)
    budget = args.max_forward_windows or None
    scaler = P.make_scaler(args.amp, model)             # the kernel's GradScaler for fp16, else None
    precision = precision_contract(args, P)
    update_options = {'amp': args.amp, 'scaler': scaler, 'autocast': not args.test_no_autocast}
    if args.generated_presence_only:                    # R10G1 (the loss only; the key exists only when on)
        update_options['generated_presence_only'] = True

    completed, chain = 0, P.CHAIN_START
    trajectory, evaluations, sessions = [], [], []
    accum, report = Counter(), Counter()
    skips = {'total': 0, 'consecutive': 0, 'updates': []}
    collisions = 0
    step_seconds = []
    continued = None
    last_path = out / 'last.pt'
    final_path = out / f'final-{args.T:06d}.pt'
    resume_path = last_path if last_path.is_file() else (final_path if final_path.is_file() else None)
    resumed = False

    def restore(saved):
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        P.restore_stream_state(streams, saved['streams'])
        warp.load_state_dict(saved['warp_state'])
        torch.set_rng_state(saved['rng']['cpu'])
        if mps and saved['rng'].get('mps') is not None:
            torch.mps.set_rng_state(saved['rng']['mps'])
        if scaler is not None:                          # fp16: the scale and its growth tracker (the kernel)
            if not saved.get('scaler_state'):
                raise RuntimeError('an fp16 run needs the saved GradScaler state')
            scaler.load_state_dict(saved['scaler_state'])

    if not args.smoke and resume_path is not None:
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        if saved.get('config') != config:
            differences = sorted(k for k in set(config) | set(saved.get('config') or {})
                                 if (saved.get('config') or {}).get(k) != config.get(k))
            log('refused', error='last.pt was written by a different configuration', differences=differences)
            return EXIT['refused']
        restore(saved)
        completed, chain = int(saved['updates']), saved['draw_chain']
        trajectory, evaluations = saved['trajectory'], saved['evaluations']
        sessions, accum, report = saved['sessions'], Counter(saved['accum']), Counter(saved['report'])
        skips, collisions, continued = saved['skips'], saved['collisions'], saved.get('continued_from')
        del saved
        resumed = True
        log('resumed', updates=completed, chain=chain, path=str(resume_path))
    elif args.continue_from:
        source = Path(args.continue_from)
        saved = torch.load(source, map_location='cpu', weights_only=False)
        mismatch = sorted(k for k in set(recipe_part(config)) | set(recipe_part(saved.get('config') or {}))
                          if recipe_part(saved.get('config') or {}).get(k) != recipe_part(config).get(k))
        if saved.get('kind') != 'mac_psr10_state_v1' or mismatch:
            log('refused', error='--continue-from is not a state of the same recipe', differences=mismatch)
            return EXIT['refused']
        start = int(saved['updates'])
        if not start < D <= args.T:
            log('refused', error=f'continuation needs saved update {start} < D {D} <= T {args.T}')
            return EXIT['refused']
        restore(saved)
        completed, chain = start, saved['draw_chain']
        evaluations = [dict(e, source=str(source)) for e in saved['evaluations']]
        skips, collisions = saved['skips'], saved['collisions']
        continued = {'path': str(source.resolve()), 'updates': start, 'source_T': saved['T'],
                     'source_D': saved['D'], 'source_run': saved['run'], 'source_utc': saved['utc'],
                     'new_T': args.T, 'new_D': D}
        del saved
        log('continued', **continued)
    if not args.smoke and not resumed:
        later = [p.name for p in out.glob('ckpt-*.pt')]
        rows_written = (out / 'draws.jsonl').is_file() and (out / 'draws.jsonl').stat().st_size > 0
        if later or rows_written:
            log('refused', error='checkpoints or draws exist without last.pt', files=later[:5])
            return EXIT['refused']
    sessions.append({'utc': P.utc(), 'pid': os.getpid(), 'started_at_update': completed, 'argv': sys.argv})

    draws_path = out / 'draws.jsonl'
    if draws_path.is_file():
        kept = []
        for line in draws_path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item['u'] <= completed:
                kept.append(line)
        first = continued['updates'] if continued else 0
        if len(kept) != completed - first:
            log('refused', error=f'draws.jsonl has {len(kept)} rows for updates {first + 1}..{completed}')
            return EXIT['refused']
        draws_path.write_text(''.join(line + '\n' for line in kept))
    elif completed and resumed:
        log('refused', error='draws.jsonl is missing for a resumed run')
        return EXIT['refused']

    source = {name: P.sha256_file(HERE / name) for name in
              ('mac_train.py', 'psr10_port.py', 'wavlm_mps_patch.py', 'export_frames.py', 'precision.py')
              if (HERE / name).is_file()}
    contract = {'kind': 'mac_psr10_run_v1', 'utc': P.utc(), 'run': args.run, 'smoke': args.smoke,
                'config': config, 'arm_spec_effective': spec, 'arm_census': census,
                'full_census': {k: v for k, v in data['census'].items() if k != 'dropped'},
                'dropped': data['census'].get('dropped'), 'silver_length_mix_census': length_census,
                'kernel': {'path': str(P.KERNEL_PATH), 'sha256': P.KERNEL_SHA256,
                           'original': str(P.KERNEL_ORIGINAL),
                           'original_sha256_now': (P.sha256_file(P.KERNEL_ORIGINAL)
                                                   if P.KERNEL_ORIGINAL.is_file() else None)},
                'kernel_draw_check': kernel_check, 'init': init_meta, 'parent_state_digest': P.PARENT_STATE,
                'optimizer_groups': [g['name'] for g in optimizer.param_groups], 'optimizer_layout': layout,
                'devices': {'encoder': args.encoder_device, 'head': args.head_device},
                'precision': precision,
                'microbatch_budget_windows': budget, 'threads': args.threads,
                'eval_every': args.eval_every, 'ckpt_every': args.ckpt_every, 'last_every': args.last_every,
                'torch': str(torch.__version__), 'python': sys.version.split()[0], 'layernorm_patch': patch,
                'source_sha256': source, 'sessions': sessions, 'resumed_at_update': completed if resumed else None,
                'continued_from': continued, 'argv': sys.argv, 'pid': os.getpid()}
    if args.silver_wholeclip_odds is not None:
        contract['silver_wholeclip_census'] = streams['silver'].census()
    if targets_report is not None:                      # R10-TC only
        contract['silver_targets'] = targets_report
    P.write_json(out / 'run-contract.json', contract)

    # ------------------------------------------------------------------ the kernel's startup checks
    t0 = time.monotonic()
    startup = ps.startup_checks(model, api, view, model.head_device, False, class_weights, spec)
    log('startup_checks', passed=startup['pass'], seconds=round(time.monotonic() - t0, 1),
        r2_identity=startup['r2_identity']['difference'], arm_objective=startup['arm_objective']['difference'],
        bronze_off=startup.get('bronze_off'), bronze_negative=(startup.get('bronze_negative') or {}).get('value'))
    if not startup['pass']:
        P.write_json(out / 'startup-checks.json', startup)
        log('failed', error='kernel startup checks failed')
        return EXIT['error']

    eval_rows = view['eval_rows'][:args.test_eval_clips] if args.test_eval_clips else view['eval_rows']
    eval_dir = out / 'eval'

    def run_eval(update, note=None):
        eval_dir.mkdir(parents=True, exist_ok=True)
        item = ps.evaluate(model, api, view, eval_rows, update, eval_dir)
        if note:
            item['note'] = note
        P.write_json(eval_dir / f'eval-{update:06d}.json', item)
        summary = {'update': update, 'clips': item['clips'], 'combined_t065': item['selection_combined'],
                   'f1_t065': item['decoders']['t065']['natural1806']['f1'],
                   'dice_t065': item['decoders']['t065']['natural1806']['dice'],
                   'combined_t075': item['decoders']['t075']['natural1806']['combined'],
                   'seconds': item['eval_seconds'], 'note': note}
        evaluations.append(summary)
        return item, summary

    # ================================================================== SMOKE (and the precision A/B)
    if args.smoke:
        return smoke(args, ps, api, P, torch, model, optimizer, warp, streams, spec, seeds, class_weights,
                     budget, guard, mps, out, run_dir, run_eval, continued, startup, config, own,
                     update_options, precision)

    # ================================================================== FULL RUN
    if completed == 0:
        if mps:
            torch.manual_seed(seeds['torch'])           # CPU (head dropout) and MPS generators, as the kernel
        else:
            torch.default_generator.manual_seed(seeds['torch'])

    def state_payload():
        return {'kind': 'mac_psr10_state_v1', 'run': args.run, 'arm': args.arm, 'config': config,
                'updates': completed, 'T': args.T, 'D': D,
                'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'optimizer_state': optimizer.state_dict(), 'streams': P.stream_state(streams),
                'warp_state': warp.state_dict(), 'amp': args.amp,
                'scaler_state': scaler.state_dict() if scaler is not None else None,
                'rng': {'cpu': torch.get_rng_state(), 'mps': torch.mps.get_rng_state() if mps else None},
                'draw_chain': chain, 'trajectory': trajectory, 'evaluations': evaluations, 'sessions': sessions,
                'accum': dict(accum), 'report': dict(report), 'skips': skips, 'collisions': collisions,
                'continued_from': continued, 'utc': P.utc()}

    def save_state(name):
        temporary = out / (name + '.tmp')
        torch.save(state_payload(), temporary)
        temporary.replace(out / name)

    def save_checkpoint(update, metrics=None):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload = {'kind': 'mac_psr10_checkpoint_v1', 'run': args.run, 'arm': args.arm, 'updates': update,
                   'dtype': 'float16', 'spec': spec, 'config': config, 'decoders': ps.DECODERS,
                   'metrics': metrics, 'state_digest_fp32': api.state_digest(state),
                   'model_state': {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()},
                   'architecture': 'NativeIntervalATST as parent-p2.pt; cast to float32 and load_state_dict',
                   'utc': P.utc()}
        temporary = out / f'ckpt-{update:06d}.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(out / f'ckpt-{update:06d}.pt')

    stop, session_updates = 'complete', 0
    driver_before_peak = 0
    final_name = final_path.name
    if not resumed:
        save_state('last.pt')           # a crash before the first periodic save resumes from here
    last_saved = completed
    with draws_path.open('a') as draws_stream:
        while completed < args.T:
            if args.max_updates is not None and completed >= args.max_updates:
                stop = 'max_updates'
                break
            if stop_path.exists():
                stop = 'stop_file'
                break
            if mps and gpu_request.exists():
                stop = 'gpu_request'
                break
            if args.test_stop_after and session_updates >= args.test_stop_after:
                stop = 'session_limit'
                break
            if mps and args.mps_check_every and session_updates and completed % args.mps_check_every == 0:
                busy = P.other_mps_processes(own)
                if busy:
                    time.sleep(5)
                    again = P.other_mps_processes(own)
                    if {b['pid'] for b in busy} & {b['pid'] for b in again}:
                        log('mps_busy', processes=again)
                        stop = 'mps_busy'
                        break
            rates = ps.apply_rates(optimizer, spec, completed, args.T, D)
            rows, skipped_ids, parts = P.draw(streams)
            tick = time.monotonic()
            stats = P.mac_update(ps, api, model, rows, optimizer, warp, class_weights=class_weights,
                                 silver_mode=spec['silver_mode'], bronze=spec['bronze'], budget=budget,
                                 **update_options)
            seconds = time.monotonic() - tick
            completed += 1
            session_updates += 1
            step_seconds.append(seconds)
            collisions += len(skipped_ids)
            record = P.draw_record(completed, rows, skipped_ids, parts, chain)
            chain = record['chain']
            draws_stream.write(json.dumps(record, separators=(',', ':')) + '\n')
            draws_stream.flush()
            if stats['skipped_update']:
                skips['total'] += 1
                skips['consecutive'] += 1
                skips['updates'].append(completed)
                log('nonfinite_gradients_skipped', update=completed, skips=skips, amp=args.amp,
                    scale_after=stats.get('scale'))
                if (skips['consecutive'] >= P.SKIP_LIMITS['consecutive']
                        or skips['total'] >= P.SKIP_LIMITS['total']):
                    raise RuntimeError(f'too many nonfinite-gradient updates: {skips}')
            else:
                skips['consecutive'] = 0
            accum['count'] += 1
            for key in ('strong_loss', 'silver_loss', 'weak_loss', 'bronze_negative', 'raw_gradient_norm'):
                if stats.get(key) is not None:
                    accum[key] += stats[key]
                    accum[key + '_n'] += 1
            for key, value in (stats.get('silver_parts') or {}).items():
                accum['silver_' + key] += value
            for key, value in (stats.get('silver_report') or {}).items():
                report[key] += value
            for key, value in (stats.get('generated_presence_only') or {}).items():     # R10G1 only
                accum['generated_presence_only_' + key] += value
            before_empty = torch.mps.driver_allocated_memory() if mps else 0
            driver_before_peak = max(driver_before_peak, before_empty)
            if mps and args.empty_cache_every and completed % args.empty_cache_every == 0:
                torch.mps.empty_cache()
            reason = guard.check('update') if mps else None
            if completed % args.log_every == 0 or completed == 1 or reason:
                window = step_seconds[-args.log_every:]
                snap = guard.snapshot or {}
                item = {'update': completed, 'strong_loss': stats['strong_loss'], 'silver_loss': stats['silver_loss'],
                        'silver_parts': stats['silver_parts'], 'weak_loss': stats['weak_loss'],
                        'bronze_negative': stats['bronze_negative'], 'gradient_norm': stats['raw_gradient_norm'],
                        'encoder_top_lr': rates['11'], 'encoder_bottom_lr': rates['0'], 'head_lr': rates['head'],
                        'seconds': seconds, 'mean_seconds': sum(window) / len(window),
                        'windows': stats['windows'], 'groups': stats['groups'], 'weak_collisions': collisions,
                        'mps_driver_gib': snap.get('mps_driver_bytes', 0) / 2 ** 30,
                        'mps_tensor_gib': snap.get('mps_tensor_bytes', 0) / 2 ** 30,
                        'mps_driver_peak_gib': guard.maximum_driver_bytes / 2 ** 30,
                        'mps_driver_before_empty_cache_gib': before_empty / 2 ** 30,
                        'system_free_percent': snap.get('system_wide_free_percent'), 'rss_peak_gib': P.rss_gib(),
                        'skips_total': skips['total'], 'amp': args.amp, 'scale': stats.get('scale')}
                if 'generated_presence_only' in stats:     # R10G1 only: rows and strong parts (kept and dropped)
                    item['generated_presence_only'] = stats['generated_presence_only']
                trajectory.append(item)
                log('update', **item)
                P.write_json(out / 'trajectory.json', {'run': args.run, 'updates': completed, 'T': args.T, 'D': D,
                                                       'log': trajectory, 'evaluations': evaluations})
            metrics = None
            at_eval = (args.eval_every and completed % args.eval_every == 0) or completed in (D, args.T)
            kill = None
            if at_eval:
                item, summary = run_eval(completed)
                count = max(1, accum.get('count', 0))
                summary['train_losses'] = {key: accum[key] / accum[key + '_n'] for key in
                                           ('strong_loss', 'silver_loss', 'weak_loss', 'bronze_negative',
                                            'raw_gradient_norm') if accum.get(key + '_n')}
                summary['train_losses'].update({key: accum[key] / count for key in
                                                ('silver_presence', 'silver_boundary', 'silver_iou') if key in accum})
                summary['train_losses'].update({key: accum[key] / count for key in sorted(accum)
                                                if key.startswith('generated_presence_only_')})     # R10G1 only
                summary['silver_report'] = dict(report)
                summary['rates'] = rates
                accum, report = Counter(), Counter()
                metrics = {'combined_t065': summary['combined_t065'], 'combined_t075': summary['combined_t075']}
                log('evaluated', **{k: v for k, v in summary.items() if k != 'silver_report'})
                threshold = spec.get('kill_below')
                if threshold is not None and completed >= ps.KILL_FROM and summary['combined_t065'] < threshold:
                    kill = f'kill_rule_raw_t065_below_{threshold}'
            if completed % args.ckpt_every == 0 or completed in (D, args.T):
                save_checkpoint(completed, metrics)
            if completed == D and D < args.T:
                save_state(f'plateau-{D:06d}.pt')
                log('plateau_saved', update=completed, path=str(out / f'plateau-{D:06d}.pt'))
            if completed == args.T:
                save_state(final_name)
                last_saved = completed
                if last_path.is_file():
                    last_path.unlink()                      # the final state replaces last.pt (disk)
            elif completed % args.last_every == 0 or kill:
                save_state('last.pt')
                last_saved = completed
            if kill:
                stop = 'kill_rule'
                P.write_json(out / 'killed.json', {'update': completed, 'reason': kill, 'metrics': metrics,
                                                   'utc': P.utc()})
                log('kill_rule', update=completed, combined_t065=metrics['combined_t065'],
                    threshold=spec.get('kill_below'))
                break
            if reason:
                stop = 'guard'
                log('guard', reason=reason, snapshot=guard.snapshot)
                break
    if completed != last_saved and completed < args.T:
        try:
            save_state('last.pt')
        except Exception as error:
            log('save_last_failed', error=repr(error))
    summary = {'run': args.run, 'arm': args.arm, 'updates': completed, 'T': args.T, 'D': D, 'stop_reason': stop,
               'session_updates': session_updates,
               'mean_update_seconds': (sum(step_seconds) / len(step_seconds)) if step_seconds else None,
               'median_update_seconds': sorted(step_seconds)[len(step_seconds) // 2] if step_seconds else None,
               'mps_driver_peak_gib': guard.maximum_driver_bytes / 2 ** 30,
               'mps_driver_peak_before_empty_cache_gib': driver_before_peak / 2 ** 30,
               'mps_tensor_peak_gib': guard.maximum_tensor_bytes / 2 ** 30,
               'rss_peak_gib': P.rss_gib(), 'weak_collisions': collisions, 'skips': skips, 'draw_chain': chain,
               'evaluations': evaluations, 'layernorm_patch_hits': wavlm_mps_patch.STATE['hits'],
               'session_hours': (time.monotonic() - STARTED) / 3600, 'continued_from': continued,
               'precision': precision, 'scale': scaler.get_scale() if scaler is not None else None}
    if completed >= args.T:
        P.write_json(out / 'training-summary.json', summary)
    log('stopped', **{k: v for k, v in summary.items() if k != 'evaluations'})
    return EXIT[stop]


# ====================================================================================== smoke

def smoke(args, ps, api, P, torch, model, optimizer, warp, streams, spec, seeds, class_weights, budget, guard,
          mps, out, run_dir, run_eval, continued, startup, config, own, update_options, precision):
    """The smoke: u0 (or the restored model) raw val eval vs the known score; N updates with finite losses and
    gradients; every encoder depth group moves; s/update, MPS memory, RSS; the time projection of the full run.
    Every update is recorded in updates.per_update (losses, the total objective, gradient norm, seconds, draw
    chain, scale, MPS memory). With --ab-reference (the precision A/B): no evaluation; N updates with --amp from
    the same init and the same draws as that fp32 smoke; ab-report.json holds the fixed rule's numbers
    (psr10_port.precision_conditions); an error or a guard stop inside the A/B is recorded, not raised."""
    import math
    ab = bool(args.ab_reference)
    report = {'run': args.run, 'arm': args.arm, 'utc': P.utc(), 'device': args.encoder_device,
              'startup_checks_pass': startup['pass'], 'config': config, 'microbatch_budget_windows': budget,
              'precision': precision, 'error': None, 'guard_stop': None}
    problems = []
    summary = None
    # ---- the u0 check (fresh) or the restored-state check (continuation); not in the A/B
    if not ab:
        tolerance = P.MPS_U0_TOLERANCE if mps else ps.U0_TOLERANCE
        item, summary = run_eval(continued['updates'] if continued else 0,
                                 note='smoke: restored plateau state' if continued else 'smoke: parent-p2.pt u0')
        combined = summary['combined_t065']
        if continued:
            source = Path(continued['path']).parent / 'eval' / f"eval-{continued['updates']:06d}.json"
            reference = None
            if source.is_file():
                reference = json.loads(source.read_text())['selection_combined']
            report['restored_check'] = {'combined_t065': combined, 'source_eval': str(source),
                                        'reference': reference, 'tolerance': tolerance,
                                        'status': ('no_reference' if reference is None else
                                                   'pass' if abs(combined - reference) <= tolerance else 'fail')}
            if report['restored_check']['status'] == 'fail':
                problems.append(f'restored model scores {combined:.6f}, source eval {reference:.6f}')
        else:
            full = item['clips'] == 1806
            delta = combined - ps.PARENT_NATURAL['combined']
            report['u0_check'] = {'combined_t065': combined, 'reference': ps.PARENT_NATURAL['combined'],
                                  'delta': delta, 'tolerance': tolerance, 'clips': item['clips'],
                                  'f1': summary['f1_t065'], 'dice': summary['dice_t065'],
                                  'combined_t075': summary['combined_t075'], 'eval_seconds': summary['seconds'],
                                  'status': ('pass' if abs(delta) <= tolerance else 'fail') if full else 'subset'}
            if report['u0_check']['status'] == 'fail':
                problems.append(f'u0 raw t0.65 {combined:.6f} is not P2 {ps.PARENT_NATURAL["combined"]:.6f} '
                                f'+/- {tolerance}')
            reference_path = Path(args.test_u0_reference) if args.test_u0_reference else (
                P.TEST_ROOT / 'ref' / 'u0-cpu' / 'pred-000000-t065.jsonl')
            if reference_path.is_file():
                mine = {json.loads(l)['clip_id']: json.loads(l)['events']
                        for l in (out / 'eval' / 'pred-000000-t065.jsonl').read_text().splitlines() if l.strip()}
                theirs = {json.loads(l)['clip_id']: json.loads(l)['events']
                          for l in reference_path.read_text().splitlines() if l.strip()}
                common = sorted(set(mine) & set(theirs))
                same = sum(1 for c in common if mine[c] == theirs[c])
                count_same = sum(1 for c in common if len(mine[c]) == len(theirs[c]))
                worst = 0.
                for c in common:
                    if len(mine[c]) == len(theirs[c]):
                        for a, b in zip(mine[c], theirs[c]):
                            worst = max(worst, abs(a['onset'] - b['onset']), abs(a['offset'] - b['offset']))
                report['u0_vs_cpu_reference'] = {'clips': len(common), 'identical_events': same,
                                                 'same_event_count': count_same,
                                                 'max_endpoint_diff_same_count_s': worst,
                                                 'reference': str(reference_path)}
    # ---- N updates
    if not continued:
        if mps:
            torch.manual_seed(seeds['torch'])
        else:
            torch.default_generator.manual_seed(seeds['torch'])
    watched = {}
    for group in optimizer.param_groups:
        first = group['params'][0]
        watched[group['name']] = first.detach().to('cpu').clone()
    completed = continued['updates'] if continued else 0
    T = args.T
    D = config['D']
    seconds, losses, norms, skipped, records = [], [], [], 0, []
    chain = P.CHAIN_START
    reason = None
    before_empty_peak = 0          # the MPS driver memory BEFORE empty_cache (the cache holds the update's peak)
    try:
        with (out / 'draws.jsonl').open('w') as stream:
            for index in range(args.smoke_updates):
                rates = ps.apply_rates(optimizer, spec, completed, T, D)
                rows, skipped_ids, parts = P.draw(streams)
                tick = time.monotonic()
                stats = P.mac_update(ps, api, model, rows, optimizer, warp, class_weights=class_weights,
                                     silver_mode=spec['silver_mode'], bronze=spec['bronze'], budget=budget,
                                     **update_options)
                seconds.append(time.monotonic() - tick)
                completed += 1
                record = P.draw_record(completed, rows, skipped_ids, parts, chain)
                chain = record['chain']
                stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                values = [stats['strong_loss'], stats['silver_loss'], stats['weak_loss']]
                if stats['bronze_negative'] is not None:
                    values.append(stats['bronze_negative'])
                losses.append(values)
                norms.append(stats['raw_gradient_norm'])
                skipped += int(stats['skipped_update'])
                before_empty = None
                if mps:
                    before_empty = torch.mps.driver_allocated_memory()
                    before_empty_peak = max(before_empty_peak, before_empty)
                    torch.mps.empty_cache()
                    reason = guard.check('smoke_update', force=True)
                snap = guard.snapshot or {}
                norm = stats['raw_gradient_norm']
                records.append({
                    'u': completed, 'strong': stats['strong_loss'], 'silver': stats['silver_loss'],
                    'weak': stats['weak_loss'], 'bronze_negative': stats['bronze_negative'],
                    'total': P.total_objective(ps, stats['strong_loss'], stats['silver_loss'], stats['weak_loss'],
                                               stats['bronze_negative']),
                    'gradient_norm': norm,
                    'finite': (all(v is None or math.isfinite(v) for v in values)
                               and norm is not None and math.isfinite(norm)),
                    'skipped': bool(stats['skipped_update']), 'seconds': seconds[-1], 'chain': chain,
                    'scale': stats.get('scale'), 'windows': stats['windows'],
                    'mps_driver_bytes': snap.get('mps_driver_bytes') if mps else None,
                    'mps_driver_bytes_before_empty_cache': before_empty,
                    'system_free_percent': snap.get('system_wide_free_percent') if mps else None})
                extra = {}
                if 'generated_presence_only' in stats:     # R10G1 only: rows and strong parts (kept and dropped)
                    records[-1]['generated_presence_only'] = stats['generated_presence_only']
                    extra = {'generated_presence_only': stats['generated_presence_only']}
                if index % 10 == 0 or index == args.smoke_updates - 1:
                    log('smoke_update', update=completed, losses=values, gradient_norm=norm,
                        seconds=round(seconds[-1], 3), windows=stats['windows'], groups=stats['groups'],
                        encoder_top_lr=rates['11'], head_lr=rates['head'], amp=args.amp, scale=stats.get('scale'),
                        mps_driver_gib=(guard.snapshot or {}).get('mps_driver_bytes', 0) / 2 ** 30, **extra)
                if reason:
                    problems.append(f'guard stop during the smoke: {reason}')
                    report['guard_stop'] = reason
                    break
    except Exception as error:
        if not ab:
            raise
        report['error'] = repr(error)[:600]
        log('ab_error', candidate=args.amp, error=report['error'], traceback=traceback.format_exc()[-2000:])
    moved = {}
    for group in optimizer.param_groups:
        now = group['params'][0].detach().to('cpu')
        moved[group['name']] = float((now - watched[group['name']]).abs().max())
    finite = bool(records) and all(r['finite'] for r in records)
    steady = seconds[10:] if len(seconds) > 20 else seconds
    per_update = sum(steady) / max(1, len(steady))
    reference = json.loads(Path(args.ab_reference).read_text()) if ab else None
    eval_seconds = summary['seconds'] if summary else (reference.get('projection') or {}).get('eval_seconds')
    evals = (T - completed) // max(1, args.eval_every) + 2
    report['updates'] = {'count': len(seconds), 'all_losses_finite': finite, 'skipped_updates': skipped,
                         'first_losses': losses[:3], 'last_losses': losses[-3:],
                         'gradient_norms_first_last': [norms[:3], norms[-3:]],
                         'encoder_groups_moved_max_abs': moved,
                         'all_encoder_groups_moved': all(moved[str(d)] > 0 for d in range(12)),
                         'head_moved': moved.get('head', 0) > 0,
                         'seconds_per_update_steady': per_update, 'seconds_all': [round(s, 3) for s in seconds],
                         'draw_chain': chain, 'per_update': records}
    peak_bytes = max(guard.maximum_driver_bytes, before_empty_peak)
    report['memory'] = {'mps_driver_peak_gib': peak_bytes / 2 ** 30,
                        'mps_driver_peak_bytes': peak_bytes if mps else None,
                        'mps_driver_peak_after_empty_cache_bytes': guard.maximum_driver_bytes if mps else None,
                        'mps_driver_peak_before_empty_cache_bytes': before_empty_peak if mps else None,
                        'mps_tensor_peak_gib': guard.maximum_tensor_bytes / 2 ** 30,
                        'rss_peak_gib': P.rss_gib(),
                        'system_free_percent_last': (guard.snapshot or {}).get('system_wide_free_percent'),
                        'system_free_percent_min': guard.minimum_free_percent if mps else None,
                        'guard_limits': P.Guard.POLICY}
    start = continued['updates'] if continued else 0
    report['projection'] = {'updates_left': T - start, 'seconds_per_update': per_update,
                            'eval_seconds': eval_seconds, 'evaluations': evals,
                            'hours': ((T - start) * per_update + evals * (eval_seconds or 0)) / 3600}
    import wavlm_mps_patch
    report['layernorm_patch_hits'] = wavlm_mps_patch.STATE['hits']
    if ab:
        report['ab'] = {'candidate': args.amp, 'reference': str(Path(args.ab_reference).resolve()),
                        'updates_requested': args.smoke_updates,
                        'reference_seconds_per_update_steady': reference['updates']['seconds_per_update_steady'],
                        'reference_memory': reference.get('memory')}
        evaluation = P.precision_conditions(reference, report)
        report['ab']['rule'] = evaluation
        report['pass'] = evaluation['pass']
        report['problems'] = problems + [f'rule: {name}' for name, c in evaluation['conditions'].items()
                                         if not c['pass']]
        P.write_json(out / 'ab-report.json', report)
        log('ab_done', candidate=args.amp, rule_pass=evaluation['pass'],
            conditions={k: [v['value'], v['pass']] for k, v in evaluation['conditions'].items()})
        return EXIT['complete']
    if not finite:
        problems.append('a nonfinite loss or gradient norm')
    if skipped:
        problems.append(f'{skipped} updates skipped (nonfinite gradients)')
    if not report['updates']['all_encoder_groups_moved']:
        problems.append('an encoder depth group did not change')
    if not report['updates']['head_moved']:
        problems.append('the head did not change')
    if len(seconds) < args.smoke_updates:
        problems.append(f'only {len(seconds)} of {args.smoke_updates} updates ran')
    if args.test_fail_smoke:
        problems.append('test hook: forced smoke failure')
    report['problems'] = problems
    report['pass'] = not problems
    P.write_json(out / 'smoke-report.json', report)
    marker = run_dir / 'SMOKE_OK'
    if report['pass']:
        marker.write_text(json.dumps({'utc': P.utc(), 'report': str(out / 'smoke-report.json'),
                                      'amp': args.amp, 'seconds_per_update': per_update,
                                      'projection_hours': report['projection']['hours']}) + '\n')
        log('smoke_passed', seconds_per_update=per_update, projection_hours=report['projection']['hours'],
            mps_driver_peak_gib=report['memory']['mps_driver_peak_gib'], u0=report.get('u0_check'), amp=args.amp)
        return EXIT['complete']
    if marker.exists():
        marker.unlink()
    log('smoke_failed', problems=problems)
    return EXIT['smoke_failed']


if __name__ == '__main__':
    try:
        code = main()
    except Exception as error:
        log('failed', error=repr(error), traceback=traceback.format_exc()[-3000:])
        code = EXIT['error']
    sys.exit(code)
