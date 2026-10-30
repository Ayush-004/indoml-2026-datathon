"""Mac port layer: the PS-R10 Kaggle kernel on the Mac (MPS encoder, CPU head; fp32 by default,
opt-in autocast fp16 + GradScaler or bf16 through `amp`).

REFERENCE = ../../training/kernels/kernel_psr10b.py, the PS-R10 Kaggle kernel (sha256 KERNEL_SHA256 = that file in
this release). load_kernel() refuses any other bytes. These kernel functions are USED UNCHANGED (called, not copied):
  build_pools, arm_view, pools_digest, check_rows, r2_tensors, r2_modes, bronze_absent_mask,
  load_parent_model (context_crop.load_parent + enforce_fp32_head), depth_optimizer, arm_rates, apply_rates,
  wsd_fraction, plateau_end, startup_checks, evaluate (predictions + fp16 frames + diagnostics), score,
  the constants (ARM_SPECS, SEEDS, MIXES, CAPS, ORIGINAL, STRONG_WEIGHT, WEAK_WEIGHT, ...), and
  mixed_training.WindowStream (the kernel's sampler) + api.FrequencyWarpStream (the kernel's warp).
The ONE kernel method that is replaced: InputLayout._path (it resolves symlinks and then refuses paths that
leave the dataset root; the local input tree is made of symlinks). Every local test of the kernel did this.

Arms (TRAIN_ARMS): the kernel's ARM_SPECS entries, used as the kernel runs them (arm_view, three_stream_update with
the spec's silver_mode and bronze, arm_rates with the spec's rates, startup_checks with the spec):
  PSR10, PSR10B  Silver mode 'full' (PSR10B with Bronze): this kernel's own arms (ARMS).
  R10, A3, A     the K-A arms (KA_ARMS): Silver mode 'R2' (the K1 per-event modes) + Bronze (the 1,887 wrap-all
                 Silver rows with their ignore channels, the Bronze absent-category negative 0.1 in the weak pass);
                 arm_view = the full build = K-A's pools (the kernel's KA_POOL_DIGEST febb87a3..., = PSR10B's pools),
                 so their streams are K-A's (and PSR10B's) streams. R10: encoder top 1e-5, warmup 0.1 -> 1 over 1,000
                 updates, head 1e-4 (warmup 200), kill rule raw t0.65 < 1.00; A3: encoder 3e-6, A: 1e-6 (warmup 200).
  startup_checks runs the Bronze negative check with torch.backends.cudnn.flags(enabled=False) (the kernel's fix
  for cuDNN's refusal of an RNN backward in evaluation mode). The port runs the GRU head in fp32 on the CPU, where
  there is no cuDNN, so that fix is not needed here (the context manager is a no-op on the CPU).

What this file ADDS (the Mac differences; each is covered by a test in tests/):
  * mac_update(): the kernel's three_stream_update, fp32 by default (amp='none': no autocast, no GradScaler), with
    optional MICROBATCHES. PRECISION (amp='fp16'|'bf16', opt-in): the kernel runs the model forward under
    torch.autocast('cuda', float16) + GradScaler(1024, growth 2000). On CUDA that autocast region touches only the
    encoder: the frontend runs on the CPU and enforce_fp32_head disables autocast in the head. encoder_autocast()
    applies autocast (fp16 or bf16) on the encoder device to exactly that part, the encoder's
    get_intermediate_layers (on MPS this is the same set of ops as autocast around the whole forward, because the
    frontend and the head run on the CPU; on the CPU tests it keeps the frontend and the head fp32 as on the Mac).
    fp16 uses the kernel's GradScaler (init 1024, growth 2, backoff .5, interval 2000) and the kernel's sequence:
    scale every pass loss, unscale_, the raw norm, clip only when finite, scaler.step, scaler.update, skipped =
    the scale dropped. bf16 has the fp32 exponent range: no scaler, a nonfinite norm skips the step (as fp32).
    Evaluation and frame exports stay fp32 (the kernel's evaluate has no autocast). Each pass (strong = Gold + generated, Silver, weak) may be split into consecutive
    clip groups of at most `budget` native windows. The loss of every group uses the LOGICAL batch's
    normalizers (the kernel's own denominators, computed from the logical batch before any forward), so the
    sum of the group losses IS the kernel's logical loss and the accumulated gradient is its gradient. With one
    group per pass, every expression is the kernel's expression (bit-identical on CPU, tests/test_loss.py).
    Nonfinite gradients: the kernel's fp32 path raises; its GPU (AMP) path skips the update. Here the update
    is skipped and counted (no optimizer step), and the trainer stops after 5 consecutive / 20 total skips.
    Gradient norm: the per-parameter norms are moved to the CPU before the stack (encoder MPS, head CPU).
  * Streams: the kernel's four WindowStreams with every seed + seed_offset (M3), or the Silver length mix
    (M1, R10-L): 4 uniform Silver draws (the arm's Silver pool, seed silver+offset, cap 12 windows) + 4 draws from
    the genuine Silver clips of 6-12 s (seed LENGTH_MIX['long_seed']+offset, cap 8). Bronze on (PSR10B and the
    K-A arms): each of the 4 long slots is a Bronze slot with probability p_B = (Bronze rows / Silver pool) (own
    generator), drawn from a Bronze-only stream, so the expected Bronze draws per update stay 8 p_B (as the
    kernel's), or the whole-clip odds (R10-W, WholeClipOddsSilver, option wholeclip_odds): the kernel's Silver
    WindowStream (same pool, seed, caps and per-cycle permutations) where, in every cycle and in every fixed
    clip-length bin (WHOLECLIP: [0,3), [3,4), [4,5), [5,6), [6,8), [8,10), [10,inf) s, each split by the native
    window count), some positions of genuine clips WITHOUT a >= 90 % event get an extra copy of a genuine clip
    WITH one (the kernel's R2 'whole' flag) of the same bin: whole-clip clips get `odds` x the draw probability
    of the other clips of their bin, each bin keeps its base share of the draws, the Bronze rows keep their
    positions. The clips stay intact (same rows, same labels, same losses). Off (None) = the kernel's stream.
  * Draw order per update = kernel Arm.step: gold.take() + generated.take(), silver, then weak.take() minus
    the clips already drawn (collisions). check_draws_against_kernel() replays the kernel's own Arm.step.
  * R10G1 (mac_update(..., generated_presence_only=True); mac_train.py --generated-presence-only; off by default):
    in the strong pass the generated rows keep their presence term, and their boundary (onset/offset focal) and
    interval-IoU terms are 0 (strong_group_loss_generated_presence_only). The Gold rows' terms and the LOGICAL
    normalizers of the strong batch (Gold + generated: strong_denominators) do not change, so every Gold row keeps
    its R10S1 terms and scale. The draws and the Silver, weak and Bronze terms do not change. Off = the kernel's loss.
  * R10-TC (apply_silver_targets; mac_train.py --silver-targets FILE; off by default): the
    'events' and 'r2' (the per-event R2 flags) of the genuine Silver rows come from a targets file (one line per
    genuine Silver clip; its builder is not part of this release). A row whose line has changed=false keeps its pool
    row object; the Bronze wrap-all rows never change. The clip ids, the pool order, the durations and the native
    windows do not change, so the draws, the Gold/generated/weak/Bronze terms and the kernel's r2_tensors code stay;
    only the Silver targets and masks of the changed rows change. Silver mode R2 only. Off = the kernel's pools.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT_DIR = HERE.parent
KERNEL_PATH = HERE.parents[1] / 'training' / 'kernels' / 'kernel_psr10b.py'
KERNEL_SHA256 = '78144d9b5d602d8fe197c5ed550df13f82d7e281fa8a62c4a3a77fd391f4538f'     # this release's kernel_psr10b.py
KERNEL_ORIGINAL = KERNEL_PATH
MOCK = Path(os.environ.get('T1_MAC_INPUT', HERE / 'input'))     # a folder laid out like /kaggle/input
REAL_RUNS = HERE / 'runs'
TEST_ROOT = HERE / 'tests'
STOP = HERE / 'STOP'
GPU_REQUEST = HERE / 'GPU_REQUEST'
PYTHON = Path(sys.executable)
PARENT_STATE = '3fa453a39509b804f80e628c53643c346824373b63c81b0deb5d877bc14a64e9'
POOL_DIGESTS = {'PSR10': '66a0d8854463ca18a48a8fe231c1d465aa8dd10a199e178346a7be49233952c2',
                'PSR10B': 'febb87a3be187cd53b4b4070526d1db54fcd68f4fb04771343b5eb046cc16d41',
                # the K-A arms: arm_view (Bronze on) = the full build = K-A's pools (the kernel's KA_POOL_DIGEST)
                'R10': 'febb87a3be187cd53b4b4070526d1db54fcd68f4fb04771343b5eb046cc16d41',
                'A3': 'febb87a3be187cd53b4b4070526d1db54fcd68f4fb04771343b5eb046cc16d41',
                'A': 'febb87a3be187cd53b4b4070526d1db54fcd68f4fb04771343b5eb046cc16d41'}
POOL_SIZES = {'PSR10': {'gold': 10745, 'silver': 60343, 'generated': 5391, 'weak': 78017},
              'PSR10B': {'gold': 10745, 'silver': 62230, 'generated': 5391, 'weak': 78017},
              'R10': {'gold': 10745, 'silver': 62230, 'generated': 5391, 'weak': 78017},
              'A3': {'gold': 10745, 'silver': 62230, 'generated': 5391, 'weak': 78017},
              'A': {'gold': 10745, 'silver': 62230, 'generated': 5391, 'weak': 78017}}
ARMS = ('PSR10', 'PSR10B')              # this kernel's own arms (the older tests iterate this pair)
KA_ARMS = ('R10', 'A3', 'A')            # the K-A arms of the kernel's ARM_SPECS (Silver 'R2' + Bronze)
TRAIN_ARMS = ARMS + KA_ARMS             # the arms that arm_data() and mac_train.py --arm accept
SEED_OFFSET_KEYS = ('gold', 'generated', 'silver', 'weak', 'torch', 'warp')
LENGTH_MIX = {'uniform_clips': 4, 'uniform_windows': 12, 'long_clips': 4, 'long_windows': 8,
              'min_seconds': 6.0, 'max_seconds': 12.0, 'long_seed': 3349, 'bronze_slot_seed': 3350,
              'bronze_stream_seed': 3351, 'bronze_windows': 16,
              'note': 'uniform part: the arm Silver pool (Bronze rows included when the arm has Bronze), the '
                      "kernel seed; long part: genuine Silver clips with 6.0 <= duration <= 12.0 s; the two caps "
                      "add up to the kernel's Silver cap (20 windows)"}
# R10-W (Mac run 4): whole-clip odds in the Silver stream (WholeClipOddsSilver). The bin of a genuine
# Silver clip = (its length bin by these edges, its native window count); every clip of a bin has one window count,
# so a replacement never changes the take packing (the per-update Silver clip and window counts stay the base's).
WHOLECLIP = {'edges_seconds': [0.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0], 'split_by_native_windows': True,
             'weight_seed': 4349, 'min_odds': 1.0, 'max_odds': 2.0,
             'note': 'bins [0,3), [3,4), [4,5), [5,6), [6,8), [8,10), [10,inf) s x native windows; whole-clip clip = '
                     "a genuine Silver clip with an event >= 90 % of the clip duration (the kernel's R2 'whole' "
                     'flag); per cycle and bin: K = floor(E) or ceil(E) (mean E = w o (odds - 1) / (odds w + o)) '
                     'non-whole clips give their positions to extra copies of K whole-clip clips (no clip twice '
                     'within max_clips positions); the base permutation, the bin sequence and the Bronze rows stay'}
# R10G1 (mac_train.py --generated-presence-only): the loss rule of the generated rows (the config value).
GENERATED_PRESENCE_ONLY = {'rows': "the generated rows of the strong pass (rows['generated'], origin 'generated_*')",
                           'kept': ['presence'], 'dropped': ['boundary', 'iou'],
                           'normalizers': 'the logical strong batch (Gold + generated), unchanged',
                           'source': 'R10G1'}
# R10-TC (mac_train.py --silver-targets FILE): the Silver-target rule of the option (the config value; the file's
# sha256 is added by the trainer). The targets builder is not part of this release.
SILVER_TARGETS = {'rows': "the genuine Silver rows of the arm's Silver pool (quality 'unverified_timestamps')",
                  'changed': "a row whose file line has changed=true gets the line's events and R2 flags (a shallow "
                             'copy of the pool row); every other row object stays',
                  'unchanged': 'the Bronze wrap-all rows, the pool order, the clip ids, durations and native windows '
                               '(so the draws are the draws without the option), the Gold/generated/weak/Bronze terms',
                  'flags': "the kernel's r2_tensors reads the file's per-event flags (mode full/presence/ignore, "
                           'onset_mask); Silver mode R2 only',
                  'checks': "one line per genuine Silver clip, same duration; a changed=false line = the row; 'kept' "
                            "and 'masked' events are the row's own human events; 'teacher' events are 'full'",
                  'source': 'R10-TC'}
SILVER_TARGET_SOURCES = ('kept', 'teacher', 'masked')
SILVER_TARGET_FLAG_KEYS = ('kind', 'mode', 'onset_mask', 'whole')
MPS_U0_TOLERANCE = 1e-3          # raw t0.65 at u0 on MPS vs P2 1.0758465. Evidence: pl-mac's P2 frames on MPS
                                 # (encoder MPS, head CPU) score exactly 1.0758464842826536 (tests/calibrate_mps_u0);
                                 # one decode flip moves the score by ~3.6e-4 (the kernel's CUDA tolerance: 2e-4)
SKIP_LIMITS = {'consecutive': 5, 'total': 20}   # K-A R10 (CUDA fp16 + GradScaler) skipped 3 of 15,500 updates

# ---- precision (item 6). fp32 stays the default; autocast is opt-in and decided ONCE for all M runs.
AMP_MODES = ('none', 'fp16', 'bf16')
KERNEL_GRAD_SCALER = {'init_scale': 1024., 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000}
PRECISION_FILE = REAL_RUNS / 'PRECISION.json'
# The fixed A/B rule for autocast. The supervisor applies it after the fp32 smoke of the first queue
# run: a candidate is used for ALL M runs only if every condition holds; the candidates are tried in this order;
# when none passes, every M run uses fp32.
PRECISION_RULE = {
    'candidates': ['fp16', 'bf16'],
    'ab_updates': 40,                          # same P2 init, same draws (seeds S0 / the run's seeds) as the smoke
    'finite': 'every A/B update has finite losses and a finite gradient norm, and no update is skipped',
    'loss_updates': [1, 2, 3, 4, 5],
    'max_relative_loss_difference': 0.02,      # |L_amp - L_fp32| / |L_fp32|, L = the total weighted objective
    'min_speedup': 1.25,                       # mean fp32 s/update / mean candidate s/update, the same updates
    'speed_updates_from': 11,                  # updates 11..40 (both runs); all updates when fewer than 20 ran
    'max_mps_driver_bytes': 11_000_000_000,    # the job's MPS memory budget (~11 GB) ...
    'min_system_free_percent': 17,             # ... and >= ~4 GB of the 24 GB free at every poll, no guard stop
    'draws': 'the per-update draw chain of the candidate equals the fp32 smoke chain (same draws)',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=1, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30      # macOS: bytes


# ============================================================================ the kernel

def _mock_path(self, relative, prefix):
    """InputLayout._path without the symlink resolution (the local tree is symlinks); same other checks."""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Input path must stay relative')
    return self.root / prefix / relative


def load_kernel(module_name='psr10_kernel', check_sha=True):
    """Import the pinned kernel copy as a module (no side effects: torch is imported inside functions)."""
    digest = sha256_file(KERNEL_PATH)
    if check_sha and digest != KERNEL_SHA256:
        raise RuntimeError(f'kernel copy changed: {digest} != {KERNEL_SHA256}')
    os.environ['ATST_SCALE_INPUT_ROOTS'] = str(MOCK)
    os.environ.setdefault('ATST_SCALE_WORKING', str(HERE / 'runs' / '.kernel-working-unused'))
    os.environ.setdefault('ATST_SCALE_TMP', str(HERE / 'runs' / '.kernel-tmp-unused'))
    spec = importlib.util.spec_from_file_location(module_name, KERNEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.InputLayout._path = _mock_path
    module.setup_paths()                       # indoml source + the parent package on sys.path
    return module


def arm_data(ps, arm):
    """The kernel's full pool build and the arm's view (census, digest, eval rows, test rows)."""
    if arm not in TRAIN_ARMS:
        raise ValueError(f'unknown arm {arm}')
    data = ps.build_pools('full')
    view = ps.arm_view(data, ps.ARM_SPECS[arm])
    return data, view


def effective_spec(ps, arm, encoder_peak=None):
    spec = dict(ps.ARM_SPECS[arm])
    if encoder_peak is not None and encoder_peak != spec['encoder_peak_top']:
        spec['encoder_peak_top'] = float(encoder_peak)
        spec['note'] = spec['note'] + f' | Mac option: encoder top peak {encoder_peak:g} (same warmup/decay)'
    return ps.check_spec(spec)


def seeds_for(ps, offset):
    seeds = {name: value + offset for name, value in ps.SEEDS.items()}
    seeds['warp'] = ps.ORIGINAL['frequency_warp']['seed'] + offset
    return seeds


# ============================================================================ streams

def take_n(stream, count):
    """WindowStream.take() with max_clips = count for this call only (the stream reads it per call)."""
    saved = stream.max_clips
    stream.max_clips = count
    try:
        return stream.take()
    finally:
        stream.max_clips = saved


def long_pool(pool):
    return [r for r in pool if r['quality'] != 'bronze_wrap_all'
            and LENGTH_MIX['min_seconds'] <= r['duration'] <= LENGTH_MIX['max_seconds']]


class LengthMixSilver:
    """M1: 4 uniform Silver draws + 4 draws from the genuine Silver clips of 6-12 s (Bronze kept at p_B)."""

    def __init__(self, pool, seeds, offset, bronze, window_stream_class):
        import torch
        WS = window_stream_class
        self.pool = pool
        self.uniform = WS(pool, LENGTH_MIX['uniform_clips'], LENGTH_MIX['uniform_windows'], seeds['silver'])
        self.long_rows = long_pool(pool)
        self.long = WS(self.long_rows, LENGTH_MIX['long_clips'], LENGTH_MIX['long_windows'],
                       LENGTH_MIX['long_seed'] + offset)
        self.bronze = None
        self.slot_generator = None
        self.bronze_rows = [r for r in pool if r['quality'] == 'bronze_wrap_all']
        self.p_bronze = len(self.bronze_rows) / len(pool)
        if bronze:
            if not self.bronze_rows:
                raise ValueError('Bronze on but no Bronze row in the Silver pool')
            self.bronze = WS(self.bronze_rows, LENGTH_MIX['long_clips'], LENGTH_MIX['bronze_windows'],
                             LENGTH_MIX['bronze_stream_seed'] + offset)
            self.slot_generator = torch.Generator(device='cpu').manual_seed(LENGTH_MIX['bronze_slot_seed'] + offset)
        elif self.bronze_rows:
            raise ValueError('Bronze off but Bronze rows in the Silver pool')
        self.last = None

    def take(self):
        import torch
        uniform = list(self.uniform.take())
        slots = LENGTH_MIX['long_clips']
        bronze_slots = 0
        if self.slot_generator is not None:
            draws = torch.rand(slots, dtype=torch.float64, generator=self.slot_generator)
            bronze_slots = int((draws < self.p_bronze).sum())
        long = list(take_n(self.long, slots - bronze_slots)) if slots - bronze_slots > 0 else []
        bronze = list(take_n(self.bronze, bronze_slots)) if bronze_slots else []
        self.last = {'uniform': len(uniform), 'long': len(long), 'bronze_slots': bronze_slots}
        return uniform + long + bronze

    def census(self):
        rows = self.long_rows
        whole = sum(1 for r in rows for m in r['r2'] if m['whole'])
        events = sum(len(r['r2']) for r in rows)
        return {'long_pool_clips': len(rows),
                'long_pool_mean_duration': sum(r['duration'] for r in rows) / max(1, len(rows)),
                'long_pool_whole_clip_event_share': whole / max(1, events),
                'silver_pool_clips': len(self.pool),
                'silver_pool_mean_duration': sum(r['duration'] for r in self.pool) / max(1, len(self.pool)),
                'bronze_rows': len(self.bronze_rows), 'p_bronze': self.p_bronze,
                'bronze_slots_on': self.slot_generator is not None}

    def sub_streams(self):
        out = {'silver_uniform': self.uniform, 'silver_long': self.long}
        if self.bronze is not None:
            out['silver_bronze'] = self.bronze
        return out


def wholeclip_bin(row):
    """The R10-W bin of a Silver row: (length bin index by WHOLECLIP['edges_seconds'], native window count)."""
    index = 0
    for number, edge in enumerate(WHOLECLIP['edges_seconds']):
        if row['duration'] >= edge:
            index = number
    return index, int(row['native_windows'])


def is_whole_clip(row):
    """A genuine Silver clip with an event >= 90 % of the clip duration (the kernel's R2 'whole' flag, r2_modes)."""
    return row['quality'] != 'bronze_wrap_all' and any(m['whole'] for m in row['r2'])


class WholeClipOddsSilver:
    """R10-W: the kernel's Silver WindowStream with whole-clip odds x `odds` inside fixed clip-length bins.

    `base` IS the kernel's stream (the same pool, max_clips, max_windows and seed as the base run); its generator makes
    only the per-cycle permutations, so they are the base run's permutations. At the start of every cycle a SEPARATE
    generator (weight_seed) changes that permutation: in each bin (w whole-clip clips, o other genuine clips),
    K = floor(E) + Bernoulli(frac(E)) of the o other clips (uniform, without replacement) give their positions to
    extra copies of K of the w whole-clip clips (uniform, without replacement; random matching), with
    E = w o (odds - 1) / (odds w + o). Per cycle, a whole-clip clip of the bin is then drawn 1 + E / w times on average
    and another clip 1 - E / o times: the ratio is `odds` (per draw, whole-clip odds x `odds` inside the bin), and the
    bin keeps its w + o positions (the base bin probability). The Bronze rows and all other positions keep the base
    permutation. Every clip of a bin has one native window count, so the take packing (Silver clips and windows per
    update) is the base run's. An extra copy never lands within max_clips positions of the clip's own position (a
    take is at most max_clips consecutive positions: no clip twice in one update). The rows are the pool rows: the
    clips stay intact (all their events, the kernel's R2 labels and losses)."""

    def __init__(self, pool, max_clips, max_windows, seed, weight_seed, odds, bronze, window_stream_class):
        import torch
        odds = float(odds)
        if not WHOLECLIP['min_odds'] < odds <= WHOLECLIP['max_odds']:
            raise ValueError(f"the whole-clip odds must be in ({WHOLECLIP['min_odds']}, {WHOLECLIP['max_odds']}]")
        if not {r['quality'] for r in pool} <= {'unverified_timestamps', 'bronze_wrap_all'}:
            raise ValueError('unexpected quality in the Silver pool')
        self.pool, self.odds, self.weight_seed, self.min_gap = pool, odds, int(weight_seed), int(max_clips)
        self.index_of = {row['clip_id']: index for index, row in enumerate(pool)}
        self.bronze = [index for index, row in enumerate(pool) if row['quality'] == 'bronze_wrap_all']
        if bool(bronze) != bool(self.bronze):
            raise ValueError('Bronze on/off does not match the Bronze rows of the Silver pool')
        self.bins = {}
        for index, row in enumerate(pool):
            if row['quality'] != 'bronze_wrap_all':
                part = self.bins.setdefault(wholeclip_bin(row), {'whole': [], 'other': []})
                part['whole' if is_whole_clip(row) else 'other'].append(index)
        self.keys = sorted(self.bins)
        self.expected_extra = {}
        for key in self.keys:
            whole, other = len(self.bins[key]['whole']), len(self.bins[key]['other'])
            self.expected_extra[key] = whole * other * (odds - 1.) / (odds * whole + other) if whole and other else 0.
        self.bin_of = {index: key for key in self.keys for part in self.bins[key].values() for index in part}
        self.whole = {index for key in self.keys for index in self.bins[key]['whole']}
        self.generator = torch.Generator(device='cpu').manual_seed(self.weight_seed)
        self.base = window_stream_class(pool, max_clips, max_windows, seed)
        self.base.order = self.perturb(self.base.order)

    def take(self):
        import torch
        base = self.base
        if base.cursor == len(base.order):      # here the kernel's stream starts its next cycle, with this call
            base.order = self.perturb(torch.randperm(len(base.rows), generator=base.generator).tolist())
            base.cursor = 0
        return base.take()

    def perturb(self, permutation):
        """One cycle's order from the base permutation (class docstring). Uses only self.generator."""
        import torch
        order = list(permutation)
        position = [0] * len(order)
        for place, index in enumerate(order):
            position[index] = place
        for key in self.keys:
            whole, other = self.bins[key]['whole'], self.bins[key]['other']
            expected = self.expected_extra[key]
            count = math.floor(expected)
            if float(torch.rand(1, dtype=torch.float64, generator=self.generator)) < expected - count:
                count += 1
            if not count:
                continue
            dropped = [other[i] for i in torch.randperm(len(other), generator=self.generator)[:count].tolist()]
            doubled = [whole[i] for i in torch.randperm(len(whole), generator=self.generator)[:count].tolist()]
            slots = sorted(position[index] for index in dropped)
            for place, index in zip(slots, self.separate(slots, doubled, position, whole)):
                if index is not None:
                    order[place] = index
        return order

    def separate(self, slots, doubled, position, whole):
        """The extra copies in slot order, swapped (deterministically) so that no copy is within min_gap positions of
        its clip's own position; a slot without such a clip keeps its own clip (None; not seen with these pools)."""
        doubled = list(doubled)

        def fits(slot, index):
            return abs(slot - position[index]) >= self.min_gap

        for i, slot in enumerate(slots):
            if fits(slot, doubled[i]):
                continue
            for j in list(range(i + 1, len(slots))) + list(range(i)):
                if doubled[j] is not None and fits(slot, doubled[j]) and fits(slots[j], doubled[i]):
                    doubled[i], doubled[j] = doubled[j], doubled[i]
                    break
            else:
                chosen = set(doubled)
                doubled[i] = next((index for index in whole if index not in chosen and fits(slot, index)), None)
        return doubled

    def signature(self):
        return {'odds': self.odds, 'weight_seed': self.weight_seed, 'min_gap': self.min_gap,
                'edges_seconds': list(WHOLECLIP['edges_seconds']),
                'bins': [[list(key), len(self.bins[key]['whole']), len(self.bins[key]['other'])] for key in self.keys]}

    def extra_state(self):
        return {'generator_state': self.generator.get_state().clone(), 'signature': self.signature()}

    def check_extra_state(self, saved):
        if not saved or 'generator_state' not in saved:
            raise ValueError('stream silver: the saved state has no whole-clip odds state')
        if saved.get('signature') != self.signature():
            raise ValueError('stream silver: the whole-clip odds rule differs from the saved state')

    def order_problem(self, order):
        """None when `order` can be a cycle order of this sampler, else the reason."""
        size = len(self.pool)
        if len(order) != size or not all(isinstance(i, int) and 0 <= i < size for i in order):
            return 'invalid cycle order'
        counts = Counter(order)
        if any(counts[i] != 1 for i in self.bronze):
            return 'a Bronze row is not in the cycle order exactly once'
        if any(count > (2 if index in self.whole else 1) for index, count in counts.items()):
            return 'a clip is in the cycle order too often'
        per_bin = Counter(self.bin_of[i] for i in order if i in self.bin_of)
        if any(per_bin[key] != len(self.bins[key]['whole']) + len(self.bins[key]['other']) for key in self.keys):
            return 'the bin totals of the cycle order differ from the pool'
        return None

    def compare_with_base(self, rows, base_ids):
        """(problem or None, replaced positions): `rows` must be the base run's draw `base_ids`, except positions where
        a genuine clip without a >= 90 % event is replaced by a whole-clip clip of the same bin."""
        if len(rows) != len(base_ids):
            return 'count', 0
        replaced = 0
        for row, base_id in zip(rows, base_ids):
            if row['clip_id'] == base_id:
                continue
            mine, theirs = self.index_of.get(row['clip_id']), self.index_of.get(base_id)
            if (mine is None or theirs is None or mine not in self.whole or theirs in self.whole
                    or theirs not in self.bin_of or self.bin_of[mine] != self.bin_of[theirs]):
                return 'replacement', replaced
            replaced += 1
        return None, replaced

    def census(self):
        edges = WHOLECLIP['edges_seconds']
        total = len(self.pool)
        genuine = sum(len(part['whole']) + len(part['other']) for part in self.bins.values())
        rows = []
        for key in self.keys:
            w, o = len(self.bins[key]['whole']), len(self.bins[key]['other'])
            extra = self.expected_extra[key]
            rows.append({'bin': list(key), 'seconds': [edges[key[0]], edges[key[0] + 1] if key[0] + 1 < len(edges)
                                                       else None], 'native_windows': key[1],
                         'clips': w + o, 'whole_clips': w, 'other_clips': o,
                         'draw_probability': (w + o) / total, 'expected_extra_copies_per_cycle': extra,
                         'whole_multiplicity': 1 + extra / w if w else None,
                         'other_multiplicity': 1 - extra / o if o else None,
                         'whole_share': {'base': w / (w + o), 'weighted': (w + extra) / (w + o)}})
        extra = sum(self.expected_extra.values())
        return {'odds': self.odds, 'bins': rows, 'silver_pool_clips': total, 'genuine_clips': genuine,
                'whole_clip_clips': len(self.whole), 'bronze_rows': len(self.bronze),
                'p_bronze': len(self.bronze) / total, 'expected_extra_copies_per_cycle': extra,
                'whole_share_of_genuine_draws': {'base': len(self.whole) / genuine,
                                                 'weighted': (len(self.whole) + extra) / genuine},
                'min_gap_positions': self.min_gap, 'weight_seed': self.weight_seed, 'rule': WHOLECLIP['note']}


def build_streams(ps, pools, seeds, offset, length_mix, bronze, mix=None, wholeclip_odds=None):
    """{'gold','generated','silver','weak'} -> sampler. 'silver' is a WindowStream, a LengthMixSilver (length_mix)
    or a WholeClipOddsSilver (wholeclip_odds = the odds; None = off, the kernel's stream)."""
    from mixed_training import WindowStream
    mix = dict(mix or ps.MIXES['8+2+8'])
    if length_mix and wholeclip_odds is not None:
        raise ValueError('the Silver length mix and the whole-clip odds are not combined (not tested)')
    streams = {}
    for name in ('gold', 'generated', 'silver', 'weak'):
        if mix.get(name, 0) <= 0:
            continue
        if name == 'silver' and length_mix:
            if mix['silver'] != 8:
                raise ValueError('the Silver length mix is defined for 8 Silver clips per update')
            streams[name] = LengthMixSilver(pools['silver'], seeds, offset, bronze, WindowStream)
        elif name == 'silver' and wholeclip_odds is not None:
            streams[name] = WholeClipOddsSilver(pools['silver'], mix['silver'], ps.CAPS['silver'], seeds['silver'],
                                                WHOLECLIP['weight_seed'] + offset, wholeclip_odds, bronze,
                                                WindowStream)
        else:
            streams[name] = WindowStream(pools[name], mix[name], ps.CAPS[name], seeds[name])
    return streams


def draw(streams):
    """One update's rows, in the kernel's Arm.step order. Returns (rows, skipped weak ids, silver parts)."""
    strong = {'gold': [], 'generated': []}
    for name in ('gold', 'generated'):
        if name in streams:
            strong[name] = list(streams[name].take())
    silver = list(streams['silver'].take()) if 'silver' in streams else []
    parts = getattr(streams.get('silver'), 'last', None)
    drawn = {row['clip_id'] for row in strong['gold'] + strong['generated'] + silver}
    weak, skipped = [], []
    if 'weak' in streams:
        for row in streams['weak'].take():
            if row['clip_id'] in drawn:
                skipped.append(row['clip_id'])
                continue
            weak.append(row)
    return {'gold': strong['gold'], 'generated': strong['generated'], 'silver': silver, 'weak': weak}, skipped, parts


def flat_streams(streams):
    out = {}
    for name, stream in streams.items():
        if isinstance(stream, LengthMixSilver):
            out.update(stream.sub_streams())
        elif isinstance(stream, WholeClipOddsSilver):
            out[name] = stream.base                    # the kernel's stream with this cycle's changed order
        else:
            out[name] = stream
    return out


def stream_state(streams):
    out = {}
    for name, stream in flat_streams(streams).items():
        out[name] = {'order': list(stream.order), 'cursor': stream.cursor,
                     'completed_cycles': stream.completed_cycles,
                     'generator_state': stream.generator.get_state().clone(),
                     'draws': dict(stream.draws), 'seen': sorted(stream.seen),
                     'max_clips': stream.max_clips, 'max_windows': stream.max_windows,
                     'pool_sha256': sha_json([r['clip_id'] for r in stream.rows])}
    silver = streams.get('silver')
    if isinstance(silver, LengthMixSilver) and silver.slot_generator is not None:
        out['_bronze_slot_generator'] = silver.slot_generator.get_state().clone()
    if isinstance(silver, WholeClipOddsSilver):
        out['silver']['wholeclip'] = silver.extra_state()     # the weight generator + the rule signature
    return out


def restore_stream_state(streams, saved):
    from collections import Counter as _Counter
    flat = flat_streams(streams)
    names = set(saved) - {'_bronze_slot_generator'}
    if names != set(flat):
        raise ValueError(f'stream set differs: saved {sorted(names)} vs {sorted(flat)}')
    silver = streams.get('silver')
    weighted = silver if isinstance(silver, WholeClipOddsSilver) else None
    if weighted is not None:
        weighted.check_extra_state(saved['silver'].get('wholeclip'))
    for name, stream in flat.items():
        state = saved[name]
        if state['pool_sha256'] != sha_json([r['clip_id'] for r in stream.rows]):
            raise ValueError(f'stream {name}: pool differs from the saved state')
        if (state['max_clips'], state['max_windows']) != (stream.max_clips, stream.max_windows):
            raise ValueError(f'stream {name}: caps differ from the saved state')
        if weighted is not None and name == 'silver':
            problem = weighted.order_problem(state['order'])     # a changed cycle order, not a permutation
            if problem:
                raise ValueError(f'stream {name}: {problem}')
        elif 'wholeclip' in state:
            raise ValueError(f'stream {name}: saved by the whole-clip odds sampler, which is off now')
        elif sorted(state['order']) != list(range(len(stream.rows))):
            raise ValueError(f'stream {name}: invalid permutation')
        stream.order = list(state['order'])
        stream.cursor = int(state['cursor'])
        stream.completed_cycles = int(state['completed_cycles'])
        stream.generator.set_state(state['generator_state'])
        stream.draws = _Counter(state['draws'])
        stream.seen = set(state['seen'])
    if weighted is not None:
        weighted.generator.set_state(saved['silver']['wholeclip']['generator_state'])
    if isinstance(silver, LengthMixSilver) and silver.slot_generator is not None:
        silver.slot_generator.set_state(saved['_bronze_slot_generator'])


def draw_record(update, rows, skipped, parts, chain):
    record = {'u': update}
    for name in ('gold', 'generated', 'silver', 'weak'):
        record[name] = [[r['clip_id'], r['native_windows']] for r in rows[name]]
    record['weak_skipped'] = skipped
    if parts is not None:
        record['silver_parts'] = parts
    record['sha'] = sha_json(record)
    record['chain'] = hashlib.sha256((chain + record['sha']).encode()).hexdigest()
    return record


CHAIN_START = hashlib.sha256(b'mac-psr10 draws v1').hexdigest()


# ============================================================================ kernel draw replay

class _FakeOptimizer:
    def __init__(self):
        self.param_groups = [{'name': str(d), 'lr': 0.} for d in range(12)] + [{'name': 'head', 'lr': 0.}]


def kernel_step_replay(pools, arm, updates, seed_offset=0, T=15000, D=12000, module_name='psr10_kernel_replay'):
    """The kernel's OWN Arm.step on a fake arm (no model): streams built as the kernel builds them, with every
    stream seed + seed_offset. three_stream_update of this separate kernel instance only records the rows."""
    replay = load_kernel(module_name)
    from mixed_training import WindowStream
    calls = []

    def recorder(model, strong, silver, weak, optimizer, scaler, warp, api, **kwargs):
        calls.append({'strong': [r['clip_id'] for r in strong], 'silver': [r['clip_id'] for r in silver],
                      'weak': [r['clip_id'] for r in weak],
                      'windows': [r['native_windows'] for r in strong + silver + weak],
                      'kwargs': dict(kwargs), 'rates': {g['name']: g['lr'] for g in optimizer.param_groups}})
        return {'strong_loss': 1., 'silver_loss': 1., 'weak_loss': 1., 'bronze_negative': None,
                'raw_gradient_norm': 1., 'silver_parts': {}, 'silver_report': {}, 'bronze_stats': None,
                'skipped_update': False, 'windows': {}}

    replay.three_stream_update = recorder
    spec = replay.ARM_SPECS[arm]
    arm_object = object.__new__(replay.Arm)
    arm_object.spec = dict(spec)
    arm_object.optimizer = _FakeOptimizer()
    arm_object.completed, arm_object.total, arm_object.plateau = 0, T, D
    mix = replay.MIXES['8+2+8']
    arm_object.streams = {n: WindowStream(pools[n], c, replay.CAPS[n], replay.SEEDS[n] + seed_offset)
                          for n, c in mix.items() if c > 0}
    arm_object.collisions = 0
    arm_object.model = arm_object.scaler = arm_object.warp = arm_object.api = None
    arm_object.amp, arm_object.class_weights = False, None
    arm_object.step_sum, arm_object.step_times = 0., []
    arm_object.accum, arm_object.report = Counter(), Counter()
    arm_object.silver_mode, arm_object.bronze = spec['silver_mode'], spec['bronze']
    for _ in range(updates):
        replay.Arm.step(arm_object)
    return calls, arm_object.collisions


def check_draws_against_kernel(ps, pools, arm, seed_offset, length_mix, updates=200, wholeclip_odds=None):
    """Startup self-check: this port's draw() equals the kernel's Arm.step replay (all streams; with the
    length mix: Gold/generated and the weak stream's raw takes; with the whole-clip odds: Gold/generated, the weak
    stream's raw takes, and the Silver draws = the kernel's Silver draws except the replaced positions, each a
    genuine clip without a >= 90 % event replaced by a whole-clip clip of the same bin). Returns a dict with 'pass'."""
    from mixed_training import WindowStream
    kernel_calls, kernel_collisions = kernel_step_replay(pools, arm, updates, seed_offset)
    seeds = seeds_for(ps, seed_offset)
    streams = build_streams(ps, pools, seeds, seed_offset, length_mix, ps.ARM_SPECS[arm]['bronze'],
                            wholeclip_odds=wholeclip_odds)
    raw = WindowStream(pools['weak'], ps.MIXES['8+2+8']['weak'], ps.CAPS['weak'], seeds['weak'])
    mismatch, long_bad = [], 0
    weighted = streams['silver'] if wholeclip_odds is not None else None
    replaced = 0
    for index in range(updates):
        rows, skipped, parts = draw(streams)
        k = kernel_calls[index]
        strong = [r['clip_id'] for r in rows['gold'] + rows['generated']]
        if strong != k['strong']:
            mismatch.append((index + 1, 'strong'))
        mine_weak = [r['clip_id'] for r in rows['weak']]
        raw_ids = [r['clip_id'] for r in raw.take()]
        if weighted is not None:
            drawn = set(strong) | {r['clip_id'] for r in rows['silver']}
            if mine_weak != [c for c in raw_ids if c not in drawn] or not set(k['weak']) <= set(raw_ids):
                mismatch.append((index + 1, 'weak_raw'))
            problem, count = weighted.compare_with_base(rows['silver'], k['silver'])
            replaced += count
            if problem:
                mismatch.append((index + 1, 'silver_' + problem))
        elif not length_mix:
            if [r['clip_id'] for r in rows['silver']] != k['silver']:
                mismatch.append((index + 1, 'silver'))
            if mine_weak != k['weak']:
                mismatch.append((index + 1, 'weak'))
        else:
            # the weak stream's raw take is the kernel's; each filtered list = raw minus its own collisions
            drawn = set(strong) | {r['clip_id'] for r in rows['silver']}
            if mine_weak != [c for c in raw_ids if c not in drawn] or not set(k['weak']) <= set(raw_ids):
                mismatch.append((index + 1, 'weak_raw'))
            uniform = rows['silver'][:parts['uniform']]
            long = rows['silver'][parts['uniform']:parts['uniform'] + parts['long']]
            if (len(uniform) + parts['long'] + parts['bronze_slots'] != len(rows['silver'])
                    or not all(LENGTH_MIX['min_seconds'] <= r['duration'] <= LENGTH_MIX['max_seconds']
                               and r['quality'] != 'bronze_wrap_all' for r in long)):
                long_bad += 1
    result = {'updates': updates, 'arm': arm, 'seed_offset': seed_offset, 'length_mix': bool(length_mix),
              'kernel_collisions': kernel_collisions, 'first_mismatches': mismatch[:10],
              'compared': (['strong', 'weak raw takes', 'length-mix silver invariants'] if length_mix
                           else ['strong', 'silver', 'weak'])}
    if length_mix:
        result['length_mix_invariant_failures'] = long_bad
    if weighted is not None:
        result['silver_wholeclip_odds'] = wholeclip_odds
        result['silver_positions_replaced'] = replaced
        result['compared'] = ['strong', 'weak raw takes', 'Silver = the kernel draws except the replaced positions']
    result['pass'] = not mismatch and not long_bad
    return result


# ============================================================================ R10-TC: Silver targets from a file

def apply_silver_targets(ps, view, path):
    """R10-TC (--silver-targets): the arm view with the events and R2 flags of the genuine Silver rows taken from the
    targets file `path`. Raises ValueError (the trainer then refuses to start) unless: the file has one line per
    genuine Silver row of the view (the same clip ids) with the row's duration; every line has events aligned with
    its flags, every flag set has the keys mode/kind/whole/onset_mask with mode full, presence or ignore, every event
    has a source kept/teacher/masked, a known category and 0 <= onset < offset <= duration; a changed=false line has
    exactly the row's events and flags; on a changed=true line every 'kept' or 'masked' event is one of the row's own
    human events (at most as often as the row has it), a 'kept' event has the row's own R2 flags (mode full), every
    human event with R2 mode full is kept, a 'masked' event is a human event whose R2 mode is not full and has mode
    ignore, a 'teacher' event has mode full, and per category the teacher events are at most the human events whose R2
    mode is not full (the short-event inventory). Returns (new view, report). The new view has
    the new Silver pool: the same order and clip ids; the pool row object for a Bronze wrap-all row or a
    changed=false line; for a changed=true line a shallow copy of the row with new 'events' (onset, offset, category,
    tag) and 'r2' (the file's flags). The pool digest (clip ids) is asserted equal. check_rows is recomputed from the
    new pools, so the kernel's startup checks run on rows that the student trains on."""
    path = Path(path)
    digest = sha256_file(path)
    pools = view['pools']
    silver = pools['silver']
    genuine = {r['clip_id'] for r in silver if r['quality'] == 'unverified_timestamps'}
    lines = {}
    with path.open() as stream:
        for number, text in enumerate(stream, 1):
            if not text.strip():
                continue
            item = json.loads(text)
            clip = item.get('clip_id')
            if clip in lines:
                raise ValueError(f'{path.name}:{number}: clip {clip} twice')
            lines[clip] = item
    if set(lines) != genuine:
        raise ValueError(f'the file clips differ from the genuine Silver rows: {len(genuine - set(lines))} missing '
                         f'(e.g. {sorted(genuine - set(lines))[:2]}), {len(set(lines) - genuine)} extra '
                         f'(e.g. {sorted(set(lines) - genuine)[:2]})')
    report = Counter()
    modes = Counter()
    new_silver = []
    for row in silver:
        if row['quality'] != 'unverified_timestamps':
            new_silver.append(row)
            report['bronze_wrap_all_rows_kept'] += 1
            continue
        clip = row['clip_id']
        item = lines[clip]
        if float(item['duration']) != float(row['duration']):
            raise ValueError(f"{clip}: duration {item['duration']} != the pool row's {row['duration']}")
        events, flags = item['events'], item['r2']
        if not events or len(events) != len(flags) or not isinstance(item.get('changed'), bool):
            raise ValueError(f'{clip}: events and r2 must be non-empty and aligned, changed must be true or false')
        plain, clean_flags = [], []
        for event, flag in zip(events, flags):
            if (tuple(sorted(flag)) != SILVER_TARGET_FLAG_KEYS or flag['mode'] not in ('full', 'presence', 'ignore')
                    or not isinstance(flag['whole'], bool) or not isinstance(flag['onset_mask'], bool)):
                raise ValueError(f'{clip}: bad flags {flag}')
            if event.get('source') not in SILVER_TARGET_SOURCES or event.get('category') not in ps.CATEGORIES:
                raise ValueError(f'{clip}: bad event {event}')
            onset, offset = float(event['onset']), float(event['offset'])
            if not (0. <= onset < offset <= float(row['duration'])):
                raise ValueError(f'{clip}: event outside the clip {event}')
            plain.append({'onset': onset, 'offset': offset, 'category': event['category'],
                          'tag': str(event.get('tag') or '')})
            clean_flags.append({'mode': flag['mode'], 'kind': flag['kind'], 'whole': flag['whole'],
                                'onset_mask': flag['onset_mask']})
        if not item['changed']:
            if plain != row['events'] or clean_flags != row['r2']:
                raise ValueError(f"{clip}: a changed=false line differs from the pool row's events or flags")
            new_silver.append(row)
            report['rows_unchanged'] += 1
            modes.update(m['mode'] for m in row['r2'])
            continue
        own = Counter((e['onset'], e['offset'], e['category'], e['tag']) for e in row['events'])
        own_flags = {(e['onset'], e['offset'], e['category'], e['tag']): m for e, m in zip(row['events'], row['r2'])}
        full_own = Counter((e['onset'], e['offset'], e['category'], e['tag'])
                           for e, m in zip(row['events'], row['r2']) if m['mode'] == 'full')
        short_inventory = Counter(e['category'] for e, m in zip(row['events'], row['r2']) if m['mode'] != 'full')
        teacher_count, kept = Counter(), Counter()
        used = Counter()
        for event, flag, simple in zip(events, clean_flags, plain):
            source = event['source']
            if source == 'teacher':
                if flag['mode'] != 'full':
                    raise ValueError(f'{clip}: a teacher event must have mode full: {event} {flag}')
                teacher_count[simple['category']] += 1
                report['teacher_events'] += 1
                continue
            key = (simple['onset'], simple['offset'], simple['category'], simple['tag'])
            used[key] += 1
            if used[key] > own[key]:
                raise ValueError(f"{clip}: a '{source}' event is not one of the row's human events: {event}")
            if source == 'kept' and (flag != own_flags[key] or flag['mode'] != 'full'):
                raise ValueError(f"{clip}: a kept event must keep the row's R2 full flags: {event} {flag}")
            if source == 'kept':
                kept[key] += 1
            if source == 'masked' and (flag['mode'] != 'ignore' or own_flags[key]['mode'] == 'full'):
                raise ValueError(f'{clip}: a masked event must be a short human event with mode ignore: {event} {flag}')
            report[source + '_events'] += 1
        if kept != full_own:
            raise ValueError(f'{clip}: every human event with R2 mode full must be kept once')
        if any(count > short_inventory[category] for category, count in teacher_count.items()):
            raise ValueError(f'{clip}: more teacher events than short human events in a category: {dict(teacher_count)}')
        report['replaced_human_events'] += sum(own.values()) - sum(used.values())
        new_silver.append(dict(row, events=plain, r2=clean_flags))
        report['rows_changed'] += 1
        modes.update(m['mode'] for m in clean_flags)
    new_pools = dict(pools, silver=new_silver)
    if ps.pools_digest(new_pools) != view['pool_digest']:
        raise ValueError('the pool digest changed (it must not: the clip ids and order stay)')
    if [(r['clip_id'], r['native_windows'], r['duration'], r['quality']) for r in new_silver] != \
            [(r['clip_id'], r['native_windows'], r['duration'], r['quality']) for r in silver]:
        raise ValueError('a Silver row changed its id, windows, duration or quality')
    new_view = dict(view, pools=new_pools, check_rows=ps.check_rows(new_pools))
    result = {'path': str(path.resolve()), 'sha256': digest, 'lines': len(lines), **dict(report),
              'genuine_modes_after': dict(modes),
              'check_rows_silver': [r['clip_id'] for r in new_view['check_rows']['silver']],
              'pool_digest': view['pool_digest'], 'rule': SILVER_TARGETS}
    return new_view, result


# ============================================================================ losses (sum form)

def strong_denominators(api, rows, device):
    """interval_loss's normalizers for the LOGICAL strong batch, with the kernel's own tensor operations:
    cells = mask.sum().clamp_min(1) * 7, IoU = (event_weights * mask).sum().clamp_min(1)."""
    import torch
    from torch import nn
    lengths = torch.tensor([row['num_frames'] for row in rows], dtype=torch.long)
    mask = (torch.arange(int(lengths.max()))[None] < lengths[:, None]).to(device)[:, :, None]
    labels = [api.interval_targets(row) for row in rows]
    event_weights = nn.utils.rnn.pad_sequence([label['event_weights'] for label in labels], batch_first=True)
    cells = mask.sum().clamp_min(1) * 7
    event_weight = event_weights.to(device) * mask
    return {'cells': cells, 'event_weight': event_weight.sum().clamp_min(1)}


def strong_group_loss(output, batch, den, class_weights, device, boundary_weight, iou_weight):
    """interval_loss (mode 'red', collar 0) of one clip group, divided by the LOGICAL normalizers."""
    import torch
    target, mask = batch['targets'].to(device), batch['mask'].to(device)[:, :, None]
    cells = den['cells']
    probability = output['presence'].clamp(1e-6, 1 - 1e-6)
    presence = -(class_weights * target * probability.log() + (1 - target) * torch.log1p(-probability))
    presence = (presence * mask).sum() / cells
    boundary = presence.new_zeros(())
    for name in ('onsets', 'offsets'):
        probability = output[name].clamp(1e-6, 1 - 1e-6)
        label = batch[name].to(device)
        focal = -(label * (1 - probability).square() * probability.log()
                  + (1 - label) * probability.square() * torch.log1p(-probability))
        boundary = boundary + (focal * mask).sum() / cells
    target_distance = batch['distances'].to(device)
    event_weight = batch['event_weights'].to(device) * mask
    intersection = torch.minimum(output['distances'], target_distance).sum(dim=-1)
    union = torch.maximum(output['distances'], target_distance).sum(dim=-1).clamp_min(1e-6)
    iou = ((1 - intersection / union) * event_weight).sum() / den['event_weight']
    loss = presence + boundary_weight * boundary + iou_weight * iou
    return loss, {'presence': presence.detach(), 'boundary': boundary.detach(), 'iou': iou.detach()}


def strong_group_loss_generated_presence_only(output, batch, den, class_weights, device, boundary_weight, iou_weight,
                                              keep):
    """R10G1: strong_group_loss with the boundary (onset/offset focal) and interval-IoU terms of the generated rows set
    to 0. keep: [clips] float32 in the clip order of `batch`, 1. = a Gold row, 0. = a generated row. The presence term
    (every row) and the LOGICAL normalizers `den` (strong_denominators of ALL strong rows, Gold + generated) are
    strong_group_loss's, so every Gold row adds exactly its terms at its scale; with keep all 1. the loss is
    strong_group_loss's bit for bit. Returns (loss, parts); the parts are detached: presence, boundary, iou (what the
    loss holds) and boundary_dropped, iou_dropped (the generated rows' terms that strong_group_loss would add)."""
    import torch
    target, mask = batch['targets'].to(device), batch['mask'].to(device)[:, :, None]
    cells = den['cells']
    probability = output['presence'].clamp(1e-6, 1 - 1e-6)
    presence = -(class_weights * target * probability.log() + (1 - target) * torch.log1p(-probability))
    presence = (presence * mask).sum() / cells
    keep = keep.to(device=device, dtype=torch.float32)[:, None, None]
    kept, dropped = mask * keep, mask * (1 - keep)
    boundary = presence.new_zeros(())
    boundary_dropped = presence.new_zeros(())
    for name in ('onsets', 'offsets'):
        probability = output[name].clamp(1e-6, 1 - 1e-6)
        label = batch[name].to(device)
        focal = -(label * (1 - probability).square() * probability.log()
                  + (1 - label) * probability.square() * torch.log1p(-probability))
        boundary = boundary + (focal * kept).sum() / cells
        boundary_dropped = boundary_dropped + (focal.detach() * dropped).sum() / cells
    target_distance = batch['distances'].to(device)
    event_weight = batch['event_weights'].to(device) * kept
    dropped_weight = batch['event_weights'].to(device) * dropped
    intersection = torch.minimum(output['distances'], target_distance).sum(dim=-1)
    union = torch.maximum(output['distances'], target_distance).sum(dim=-1).clamp_min(1e-6)
    iou = ((1 - intersection / union) * event_weight).sum() / den['event_weight']
    iou_dropped = ((1 - intersection.detach() / union.detach()) * dropped_weight).sum() / den['event_weight']
    loss = presence + boundary_weight * boundary + iou_weight * iou
    return loss, {'presence': presence.detach(), 'boundary': boundary.detach(), 'iou': iou.detach(),
                  'boundary_dropped': boundary_dropped, 'iou_dropped': iou_dropped}


def generated_keep_masks(rows, parts):
    """R10G1: one keep vector per strong clip group (split_groups of rows['gold'] + rows['generated'], clip order kept):
    float32, 1. for a Gold row, 0. for a generated row. Refuses rows that are not the Gold and generated pool rows."""
    import torch
    gold, generated = list(rows['gold']), list(rows['generated'])
    if (any(str(r.get('origin', '')).startswith('generated_') for r in gold)
            or not all(str(r.get('origin', '')).startswith('generated_') for r in generated)):
        raise ValueError("generated_presence_only: rows['gold'] / rows['generated'] are not Gold / generated rows")
    strong = gold + generated
    out, start = [], 0
    for group in parts:
        if any(a is not b for a, b in zip(group, strong[start:start + len(group)])) or start + len(group) > len(strong):
            raise RuntimeError('generated_presence_only: a clip group is not the next part of the strong batch')
        out.append(torch.tensor([1. if start + i < len(gold) else 0. for i in range(len(group))], dtype=torch.float32))
        start += len(group)
    if start != len(strong):
        raise RuntimeError('generated_presence_only: the clip groups do not cover the strong batch')
    return out


def silver_denominators(ps, api, rows, force_full, device):
    """r2_silver_loss's normalizers for the LOGICAL Silver batch (the kernel's r2_tensors on all its rows)."""
    frames_max = max(row['num_frames'] for row in rows)
    r2 = ps.r2_tensors(rows, frames_max, api, force_full=force_full)
    keep = r2['bi_keep'].to(device)
    return {'presence': r2['presence_keep'].to(device).sum().clamp_min(1),
            'cells': keep.sum().clamp_min(1),
            'event_weight': (r2['full']['event_weights'].to(device) * keep).sum().clamp_min(1),
            'has_boundary': float(r2['bi_keep'].sum()) > 0}


def silver_group_loss(output, batch, r2, den, class_weights, device, boundary_weight, iou_weight):
    """masked_presence + boundary_iou_terms (kernel r2_silver_loss) of one group, LOGICAL normalizers."""
    import torch
    target = batch['targets'].to(device)
    keep = r2['presence_keep'].to(device)
    probability = output['presence'].clamp(1e-6, 1 - 1e-6)
    presence = -(class_weights * target * probability.log() + (1 - target) * torch.log1p(-probability))
    presence = (presence * keep).sum() / den['presence']
    loss = presence
    parts = {'presence': presence.detach(), 'boundary': None, 'iou': None}
    if den['has_boundary']:
        bi = r2['bi_keep'].to(device)
        onset = bi * r2['onset_keep'].to(device)
        cells = den['cells']
        boundary = output['presence'].new_zeros(())
        for name, mask in (('onsets', onset), ('offsets', bi)):
            probability = output[name].clamp(1e-6, 1 - 1e-6)
            label = r2['full'][name].to(device)
            focal = -(label * (1 - probability).square() * probability.log()
                      + (1 - label) * probability.square() * torch.log1p(-probability))
            boundary = boundary + (focal * mask).sum() / cells
        target_distance = r2['full']['distances'].to(device)
        event_weight = r2['full']['event_weights'].to(device) * bi
        intersection = torch.minimum(output['distances'], target_distance).sum(dim=-1)
        union = torch.maximum(output['distances'], target_distance).sum(dim=-1).clamp_min(1e-6)
        iou = ((1 - intersection / union) * event_weight).sum() / den['event_weight']
        loss = loss + boundary_weight * boundary + iou_weight * iou
        parts.update(boundary=boundary.detach(), iou=iou.detach())
    return loss, parts


def weak_group_terms(presence, lengths, positive_mask):
    """indoml.weak_supervision.weak_positive_loss up to its final mean: the per supervised clip values."""
    import torch
    batch, frames, _ = presence.shape
    lengths = lengths.to(device=presence.device)
    if not bool(((lengths > 0) & (lengths <= frames)).all()):
        raise ValueError('Lengths must be within the real time axis')
    positive_mask = positive_mask.to(device=presence.device)
    pairs = positive_mask.nonzero(as_tuple=False)
    category_counts = positive_mask.sum(dim=1)
    supervised = category_counts > 0
    if not len(pairs):
        return presence.reshape(-1)[:0], 0
    values = presence[pairs[:, 0], :, pairs[:, 1]]
    valid = torch.arange(frames, device=presence.device)[None] < lengths[pairs[:, 0], None]
    observed = values[valid]
    if not (bool(torch.isfinite(observed).all()) and bool(((observed >= 0) & (observed <= 1)).all())):
        raise ValueError('Supervised real-frame presence must be finite probabilities in [0, 1]')
    maxima = values.masked_fill(~valid, -torch.inf).amax(dim=1)
    terms = -maxima.clamp(min=1e-6, max=1.).log()
    sums = presence.new_zeros(batch).index_add(0, pairs[:, 0], terms)
    return (sums / category_counts.clamp_min(1))[supervised], int(supervised.sum())


def bronze_group_terms(presence, lengths, absent):
    """kernel bronze_negative_loss up to its final mean: the per Bronze clip values."""
    import torch
    batch, frames, _ = presence.shape
    lengths = lengths.to(presence.device)
    per_clip = absent.sum(dim=1)
    supervised = per_clip > 0
    if not bool(supervised.any()):
        return presence.reshape(-1)[:0], 0
    valid = torch.arange(frames, device=presence.device)[None, :, None] < lengths[:, None, None]
    maxima = presence.masked_fill(~valid, 0.).amax(dim=1)
    terms = -torch.log1p(-maxima.clamp(0., 1 - 1e-6))
    sums = (terms * absent).sum(dim=1)
    return sums[supervised] / per_clip[supervised], int(supervised.sum())


def group_mean(values, total, single):
    """The kernel's .mean() when the pass is one group, else sum / (the logical count)."""
    if single:
        return values.mean()
    return values.sum() / total


# ============================================================================ precision

def amp_dtype(amp):
    import torch
    return {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[amp]


@contextmanager
def encoder_autocast(model, amp, enabled=True):
    """Autocast (amp dtype, device type of the ENCODER) around the encoder's get_intermediate_layers only.

    The kernel wraps model(...) in torch.autocast('cuda', float16). The ops that this region really casts on CUDA
    are the encoder's: the frontend (mel) runs on the CPU and the head disables autocast (enforce_fp32_head).
    The encoder ends with norm_frame (LayerNorm, an autocast fp32 op), so the stitched features are fp32 in both.
    On the Mac (encoder MPS, frontend and head CPU) this region is the same set of ops as autocast around the
    whole forward. On the CPU tests (encoder CPU) it keeps the frontend and the head in fp32, as on the Mac.
    amp 'none' (or enabled False, a test hook) is a plain no-op context."""
    if amp == 'none' or not enabled:
        yield
        return
    import torch
    dtype = amp_dtype(amp)
    device_type = model.device.type
    encoder = model.encoder
    if 'get_intermediate_layers' in encoder.__dict__:
        raise RuntimeError('encoder_autocast is already active')
    inner = encoder.get_intermediate_layers

    def get_intermediate_layers(*args, **kwargs):
        with torch.autocast(device_type=device_type, dtype=dtype):
            return inner(*args, **kwargs)

    encoder.get_intermediate_layers = get_intermediate_layers
    try:
        yield
    finally:
        del encoder.get_intermediate_layers              # the class method again


def make_scaler(amp, model):
    """The kernel's GradScaler (init 1024, growth interval 2000, defaults growth 2 / backoff .5) for fp16;
    None for fp32 and bf16. The losses are computed on the head device (CPU), so the scale tensor lives there;
    unscale_ handles the encoder (MPS) and head (CPU) gradients per device."""
    if amp != 'fp16':
        return None
    import torch
    return torch.amp.GradScaler(model.head_device.type, **KERNEL_GRAD_SCALER, enabled=True)


def total_objective(ps, strong, silver, weak, bronze_negative=None):
    """The weighted objective of one update: 1.0 strong + 0.4 Silver + 0.1 weak (+ 0.1 Bronze negative)."""
    total = ps.STRONG_WEIGHT['gold'] * strong
    if silver is not None:
        total += ps.STRONG_WEIGHT['silver'] * silver
    if weak is not None:
        total += ps.WEAK_WEIGHT * weak
    if bronze_negative is not None:
        total += ps.BRONZE_NEGATIVE_WEIGHT * bronze_negative
    return total


def read_precision_file(path):
    """(decision dict, sha256) of a PRECISION.json written by precision.py decide."""
    path = Path(path)
    decision = json.loads(path.read_text())
    if decision.get('kind') != 'mac_psr10_precision_v1' or decision.get('amp') not in AMP_MODES:
        raise ValueError(f'{path} is not a mac-psr10 precision decision')
    return decision, sha256_file(path)


def precision_conditions(reference, candidate, rule=None):
    """Evaluate the fixed A/B rule. reference = the fp32 smoke report, candidate = an A/B report (both written
    by mac_train.py; their 'updates.per_update' lists hold u, total, finite, skipped, seconds, chain).
    Returns {'conditions': {name: {'value', 'limit', 'pass'}}, 'pass': bool}. Recomputed from the raw numbers."""
    rule = dict(PRECISION_RULE, **(rule or {}))
    out = {}
    ab = candidate.get('ab') or {}
    requested = int(ab.get('updates_requested') or 0)
    mine = {int(r['u']): r for r in candidate.get('updates', {}).get('per_update', [])}
    theirs = {int(r['u']): r for r in reference.get('updates', {}).get('per_update', [])}
    first = min(mine) if mine else 1
    wanted = list(range(first, first + requested))
    ran = [u for u in wanted if u in mine]
    out['reference_is_fp32_smoke'] = {'value': reference.get('precision', {}).get('amp'), 'limit': 'none',
                                      'pass': reference.get('precision', {}).get('amp') == 'none'
                                      and not reference.get('ab') and bool(reference.get('pass'))}
    out['ran_without_error'] = {'value': candidate.get('error'), 'limit': None,
                                'pass': candidate.get('error') is None and candidate.get('guard_stop') is None}
    finite = [u for u in ran if mine[u].get('finite') and not mine[u].get('skipped')]
    out['finite_updates'] = {'value': f'{len(finite)}/{requested}', 'limit': f'{requested}/{requested}',
                             'pass': requested > 0 and len(finite) == requested}
    same = [u for u in ran if u in theirs and mine[u].get('chain') == theirs[u].get('chain')]
    out['same_draws'] = {'value': f'{len(same)}/{len(ran)}', 'limit': 'all',
                         'pass': bool(ran) and len(same) == len(ran) == requested}
    differences = {}
    compared = [first + u - 1 for u in rule['loss_updates'] if u <= requested]     # updates 1-5 (fewer: tests)
    for u in compared:
        if u in mine and u in theirs and mine[u].get('total') is not None and theirs[u].get('total'):
            differences[u] = abs(mine[u]['total'] - theirs[u]['total']) / abs(theirs[u]['total'])
    worst = max(differences.values()) if differences else None
    out['loss_difference_updates_1_5'] = {
        'value': worst, 'per_update': differences, 'limit': rule['max_relative_loss_difference'],
        'pass': worst is not None and bool(compared) and len(differences) == len(compared)
        and math.isfinite(worst) and worst <= rule['max_relative_loss_difference']}
    steady = [u for u in ran if u >= first + rule['speed_updates_from'] - 1] if requested >= 20 else list(ran)
    steady = [u for u in steady if u in theirs and mine[u].get('seconds') and theirs[u].get('seconds')]
    speedup = None
    if steady:
        speedup = (sum(theirs[u]['seconds'] for u in steady) / len(steady)) / (
            sum(mine[u]['seconds'] for u in steady) / len(steady))
    out['speedup'] = {'value': speedup, 'updates': [steady[0], steady[-1]] if steady else None,
                      'limit': rule['min_speedup'], 'pass': speedup is not None and speedup >= rule['min_speedup']}
    memory = candidate.get('memory') or {}
    driver, free = memory.get('mps_driver_peak_bytes'), memory.get('system_free_percent_min')
    if candidate.get('device') == 'mps':
        memory_ok = (driver is not None and driver <= rule['max_mps_driver_bytes'] and free is not None
                     and free >= rule['min_system_free_percent'] and candidate.get('guard_stop') is None)
    else:                                            # CPU tests: no MPS memory to measure
        memory_ok = candidate.get('guard_stop') is None
    out['memory'] = {'value': {'mps_driver_peak_bytes': driver, 'system_free_percent_min': free,
                               'guard_stop': candidate.get('guard_stop'), 'device': candidate.get('device')},
                     'limit': {'mps_driver_peak_bytes': rule['max_mps_driver_bytes'],
                               'system_free_percent_min': rule['min_system_free_percent']},
                     'pass': bool(memory_ok)}
    return {'conditions': out, 'pass': all(c['pass'] for c in out.values()), 'rule': rule}


# ============================================================================ microbatches

def split_groups(rows, budget=None, groups=None):
    """Consecutive clip groups (clip order kept): at most `budget` native windows each (one clip that is larger
    forms its own group), or exactly `groups` groups of near-equal clip counts (tests)."""
    if groups:
        groups = min(groups, len(rows))
        size, extra = divmod(len(rows), groups)
        out, start = [], 0
        for index in range(groups):
            end = start + size + (1 if index < extra else 0)
            out.append(rows[start:end])
            start = end
        return [g for g in out if g]
    if not budget:
        return [list(rows)]
    out, current, windows = [], [], 0
    for row in rows:
        count = row['native_windows']
        if current and windows + count > budget:
            out.append(current)
            current, windows = [], 0
        current.append(row)
        windows += count
    if current:
        out.append(current)
    return out


def mac_update(ps, api, model, rows, optimizer, warp, *, class_weights, silver_mode, bronze,
               budget=None, groups=None, step=True, max_norm=None, amp='none', scaler=None, autocast=True,
               generated_presence_only=False):
    """The kernel's three_stream_update with optional clip groups (see the module docstring).
    rows: {'gold','generated','silver','weak'}. Returns the kernel's stats keys plus 'groups', 'amp', 'scale'.
    amp 'none' (fp32, scaler None), 'fp16' (scaler = make_scaler('fp16', model), the kernel's AMP sequence) or
    'bf16' (scaler None). autocast=False is a TEST hook: the scaler path of `amp` with an fp32 forward (this is
    what the kernel's own AMP code does on a machine without CUDA, where its autocast('cuda') is disabled).
    step=False (tests) leaves the unscaled gradients in .grad; a scaler used so must not be used again.
    generated_presence_only=True (R10G1): the strong pass uses strong_group_loss_generated_presence_only (the generated
    rows keep their presence term; their boundary and IoU terms are 0; the Gold rows' terms and the logical
    normalizers do not change), and the stats get the key 'generated_presence_only' (rows and the strong parts).
    False (default): the kernel's loss, the code path and the stats of before."""
    import torch
    from indoml.weak_supervision import positive_category_mask
    if amp not in AMP_MODES:
        raise ValueError(f'unknown amp {amp!r}')
    scaling = scaler is not None and scaler.is_enabled()
    if (amp == 'fp16') != scaling:
        raise ValueError('amp fp16 needs an enabled GradScaler; amp none/bf16 use no scaler')

    def scaled(value):
        return scaler.scale(value) if scaling else value

    strong_rows = list(rows['gold']) + list(rows['generated'])
    silver_rows, weak_rows = list(rows['silver']), list(rows['weak'])
    assert silver_mode in ps.SILVER_MODES, f'Unknown Silver mode {silver_mode!r}'
    assert all(r['supervision'] == 'strong_gold' and r['events'] for r in strong_rows)
    assert all(r['supervision'] == 'strong_gold' and r['events'] and 'r2' in r for r in silver_rows)
    assert all(r['supervision'] == 'weak_clip' and r['events'] is None
               and r['positive_categories'] for r in weak_rows)
    assert bronze or not any(r['quality'] == 'bronze_wrap_all' for r in silver_rows), 'Bronze row, Bronze off'
    assert bronze or not any(r.get('bronze_absent') for r in weak_rows), 'Bronze negative list, Bronze off'
    assert getattr(model, '_full_release_fp32_head', False), 'Enforce FP32 head math first'
    device = model.head_device
    settings = ps.ORIGINAL['strong_loss']
    boundary_weight, iou_weight = settings['boundary_weight'], settings['iou_weight']
    max_norm = ps.ORIGINAL['optimizer']['gradient_norm_clip'] if max_norm is None else max_norm
    model.train()
    optimizer.zero_grad(set_to_none=True)
    windows, split = {}, {}

    # ---- strong pass: Gold + generated, weight 1.0, P2's objective (interval_loss), warp on
    den = strong_denominators(api, strong_rows, device)
    parts = split_groups(strong_rows, budget, groups)
    split['strong'] = [len(g) for g in parts]
    strong_value, windows['strong'] = 0., 0
    keeps, presence_only = [None] * len(parts), None
    if generated_presence_only:                       # R10G1: 1. = Gold row, 0. = generated row, per clip group
        keeps = generated_keep_masks(rows, parts)
        presence_only = {'gold_rows': len(rows['gold']), 'generated_rows': len(rows['generated']), 'presence': 0.,
                         'boundary': 0., 'iou': 0., 'boundary_dropped': 0., 'iou_dropped': 0.}
    for group, keep in zip(parts, keeps):
        batch = api.collate_audio_intervals(group)
        windows['strong'] += len(batch['windows'])
        model.frequency_augmentation = warp
        with encoder_autocast(model, amp, autocast):
            output = model(batch['windows'], batch['descriptions'])
        output = {key: value.float() for key, value in output.items()}
        if keep is None:
            loss, _ = strong_group_loss(output, batch, den, class_weights, device, boundary_weight, iou_weight)
        else:
            loss, terms = strong_group_loss_generated_presence_only(output, batch, den, class_weights, device,
                                                                    boundary_weight, iou_weight, keep)
            for key, value in terms.items():
                presence_only[key] += float(value)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite strong loss')
        scaled(ps.STRONG_WEIGHT['gold'] * loss).backward()
        strong_value += float(loss.detach())
        del loss, output, batch

    # ---- Silver pass: weight 0.4, the arm's Silver mode ('full' = interval_loss on every event), warp on
    silver_value, silver_parts, silver_report = None, None, None
    if silver_rows:
        force_full = silver_mode == 'full'
        den = silver_denominators(ps, api, silver_rows, force_full, device)
        parts = split_groups(silver_rows, budget, groups)
        split['silver'] = [len(g) for g in parts]
        silver_value, windows['silver'] = 0., 0
        silver_parts = {'presence': 0., 'boundary': 0., 'iou': 0.}
        silver_report = Counter()
        for group in parts:
            batch = api.collate_audio_intervals(group)
            windows['silver'] += len(batch['windows'])
            r2 = ps.r2_tensors(group, int(batch['targets'].shape[1]), api, force_full=force_full)
            model.frequency_augmentation = warp
            with encoder_autocast(model, amp, autocast):
                output = model(batch['windows'], batch['descriptions'])
            output = {key: value.float() for key, value in output.items()}
            loss, terms = silver_group_loss(output, batch, r2, den, class_weights, device,
                                            boundary_weight, iou_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite silver loss')
            scaled(ps.STRONG_WEIGHT['silver'] * loss).backward()
            silver_value += float(loss.detach())
            for key, value in terms.items():
                if value is not None:
                    silver_parts[key] += float(value)
            silver_report.update(r2['report'])
            del loss, output, batch, r2
        silver_report = dict(silver_report)

    # ---- weak pass: 0.1 x positive-only clip-tag term (+ 0.1 x Bronze negative only when the arm has Bronze)
    weak_value, bronze_value, bronze_stats = None, None, None
    if weak_rows:
        model.frequency_augmentation = None
        all_mask = positive_category_mask([r['positive_categories'] for r in weak_rows])
        weak_total = int((all_mask.sum(dim=1) > 0).sum())
        bronze_total = 0
        if bronze:
            bronze_total = int((ps.bronze_absent_mask(weak_rows, 'cpu').sum(dim=1) > 0).sum())
        parts = split_groups(weak_rows, budget, groups)
        single = len(parts) == 1
        split['weak'] = [len(g) for g in parts]
        weak_value, windows['weak'] = 0., 0
        bronze_value = 0. if bronze else None
        bronze_stats = Counter() if bronze else None
        for group in parts:
            batch = api.collate_clips(group)
            lengths = torch.tensor([item['frames'] for item in batch['descriptions']], dtype=torch.long)
            mask = positive_category_mask([r['positive_categories'] for r in group], device=device)
            windows['weak'] += len(batch['windows'])
            with encoder_autocast(model, amp, autocast):
                presence = model(batch['windows'], batch['descriptions'])['presence']
            presence = presence.float()
            values, _ = weak_group_terms(presence, lengths, mask)
            weak_loss = group_mean(values, weak_total, single)
            if not bool(torch.isfinite(weak_loss)):
                raise RuntimeError('Nonfinite weak loss')
            total = ps.WEAK_WEIGHT * weak_loss
            if bronze:
                absent = ps.bronze_absent_mask(group, device)
                negative_values, count = bronze_group_terms(presence, lengths, absent)
                if count:
                    negative = group_mean(negative_values, bronze_total, single)
                else:
                    negative = presence.reshape(-1)[:0].sum()
                if not bool(torch.isfinite(negative)):
                    raise RuntimeError('Nonfinite Bronze negative loss')
                total = ps.WEAK_WEIGHT * weak_loss + ps.BRONZE_NEGATIVE_WEIGHT * negative
                bronze_value += float(negative.detach())
                bronze_stats['bronze_clips'] += count
                bronze_stats['absent_channels'] += int(absent.sum())
            scaled(total).backward()
            weak_value += float(weak_loss.detach())
            del presence, weak_loss, total, batch, mask, lengths
        bronze_stats = dict(bronze_stats) if bronze_stats is not None else None

    # ---- clip the global gradient norm at 5 and step once (a nonfinite norm skips the step)
    parameters = [p for p in model.parameters() if p.requires_grad]
    before_scale = None
    if scaling:                                       # the kernel: before_scale, unscale_ before the norm
        before_scale = scaler.get_scale()
        scaler.unscale_(optimizer)
    norms = [torch.linalg.vector_norm(p.grad.detach(), 2).to('cpu') for p in parameters if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack(norms), 2)
    finite = bool(torch.isfinite(norm).item())
    skipped = False
    if step and scaling:                              # the kernel's AMP sequence
        if finite:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
        scaler.step(optimizer)                        # no optimizer step when a gradient is inf/NaN
        scaler.update()                               # backoff .5 on a skip, growth 2 after 2,000 clean updates
        skipped = scaler.get_scale() < before_scale
        if not (finite or skipped):
            raise RuntimeError('Nonfinite gradients were not safely skipped')
    elif step:
        if finite:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            skipped = True
    model.frequency_augmentation = warp
    stats = {'strong_loss': strong_value, 'silver_loss': silver_value, 'silver_parts': silver_parts,
             'silver_report': silver_report, 'weak_loss': weak_value, 'bronze_negative': bronze_value,
             'bronze_stats': bronze_stats, 'strong_clips': len(strong_rows), 'silver_clips': len(silver_rows),
             'weak_clips': len(weak_rows), 'windows': windows, 'groups': split,
             'raw_gradient_norm': float(norm) if finite else None, 'skipped_update': skipped, 'amp': amp,
             'scale': scaler.get_scale() if scaling else None}
    if presence_only is not None:                     # R10G1 only: the strong parts (the key exists only when on)
        stats['generated_presence_only'] = presence_only
    return stats


# ============================================================================ model

def load_model(ps, encoder_device='cpu', head_device='cpu'):
    """The kernel's load_parent_model (context_crop.load_parent: P2 bytes, digest, enforce_fp32_head) on the
    CPU, then the encoder moved to `encoder_device` (the head stays on `head_device`)."""
    import torch
    model, meta = ps.load_parent_model('cpu')
    from indoml import atst_intervals as api
    digest = api.state_digest(model.state_dict())
    if digest != PARENT_STATE:
        raise RuntimeError(f'u0 model is not P2: {digest}')
    if str(encoder_device) != 'cpu' or str(head_device) != 'cpu':
        model.set_device(torch.device(encoder_device), torch.device(head_device))
    meta = dict(meta, u0_state_digest=digest)
    return model, api, meta


def load_weights(model, path):
    """Weights from a state file (model_state fp32) or a checkpoint (fp16 -> fp32), strict."""
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('model_state', payload)
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return payload


# ============================================================================ guards

class Guard:
    """du-mac/kv-mac Guard with this job's limits: pause (exit 12) when the MPS driver holds more than 14 GB,
    the system-wide free memory is below 17 % (about 4 GB of 24 GB) or the pressure flag is 4."""
    POLICY = {'maximum_mps_driver_bytes': 14_000_000_000, 'minimum_system_free_percent': 17,
              'system_query_interval_seconds': 2.0}

    def __init__(self, torch_module, use_mps=True):
        self.torch = torch_module
        self.use_mps = bool(use_mps)
        self.last_query = -float('inf')
        self.maximum_driver_bytes = 0
        self.maximum_tensor_bytes = 0
        self.minimum_free_percent = None
        self.snapshot = None

    def snapshot_now(self):
        text = subprocess.run(['/usr/bin/memory_pressure', '-Q'], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30).stdout
        marker = 'System-wide memory free percentage:'
        line = next((x for x in text.splitlines() if marker in x), None)
        if line is None:
            raise ValueError('memory_pressure output changed')
        percent = int(line.rsplit(':', 1)[1].strip().rstrip('%'))
        flag = int(subprocess.run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                                  check=True, text=True, stdout=subprocess.PIPE, timeout=30).stdout.strip())
        mps = self.use_mps
        return {'utc': utc(), 'system_wide_free_percent': percent, 'pressure_flag': flag,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'mps_driver_bytes': self.torch.mps.driver_allocated_memory() if mps else 0,
                'mps_tensor_bytes': self.torch.mps.current_allocated_memory() if mps else 0}

    def check(self, stage, force=False):
        if not force and time.monotonic() - self.last_query < self.POLICY['system_query_interval_seconds']:
            return None
        try:
            self.snapshot = self.snapshot_now()
        except Exception as error:
            return f'guard_query_failed_at_{stage}: {error!r}'
        self.last_query = time.monotonic()
        self.maximum_driver_bytes = max(self.maximum_driver_bytes, self.snapshot['mps_driver_bytes'])
        self.maximum_tensor_bytes = max(self.maximum_tensor_bytes, self.snapshot['mps_tensor_bytes'])
        free = self.snapshot['system_wide_free_percent']
        self.minimum_free_percent = free if self.minimum_free_percent is None else min(self.minimum_free_percent,
                                                                                       free)
        if self.snapshot['mps_driver_bytes'] > self.POLICY['maximum_mps_driver_bytes']:
            return f'mps_driver_over_14gb_at_{stage}'
        if self.snapshot['system_wide_free_percent'] < self.POLICY['minimum_system_free_percent']:
            return f'system_free_below_17_percent_at_{stage}'
        if self.snapshot['pressure_flag'] == 4:
            return f'critical_pressure_flag_4_at_{stage}'
        return None


def other_mps_processes(own_pids=()):
    """Python processes with '--encoder-device mps' on the command line (the project marker), except own."""
    text = subprocess.run(['/bin/ps', '-axo', 'pid=,command='], check=True, text=True,
                          stdout=subprocess.PIPE, timeout=30).stdout
    own = {int(p) for p in own_pids}
    found = []
    for line in text.splitlines():
        pid_text, _, command = line.strip().partition(' ')
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in own:
            continue
        if '--encoder-device mps' in command and 'python' in command.lower():
            found.append({'pid': pid, 'command': command.strip()[:200]})
    return found
