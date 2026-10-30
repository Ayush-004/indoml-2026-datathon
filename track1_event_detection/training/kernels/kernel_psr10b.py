"""IndoML 2026 Track 1 - PSR10B training kernel (Kaggle, 2 x T4; the PSR10B member is arm PSR10B at update 15,000).
"PS-R10": P2 continued with the FULL Silver objective (as P2-silver) at the K-A R10 encoder rate; arm PSR10 without
Bronze, arm PSR10B with Bronze; two arms on two T4s.

Code base: kernel_r10.py (K-A), which is the P2-silver kernel for mounting, pools, the P2 model, the depth
optimizer, the three-stream update, evaluation and outputs, plus the Silver R2 masks of K1.
Changes from K-A: an explicit per-arm spec (encoder schedule, Silver mode, Bronze on/off) in
ARM_SPECS; per-arm pools (arm_view); the Bronze negative term only in an arm with Bronze;
test natural predictions + frames at D as well as at T (the plan keeps time for them).

Model: P2 unchanged (NativeIntervalATST loaded from parent-p2.pt by context_crop.load_parent,
FP32 head, fp16 autocast encoder + GradScaler(1024, growth 2000), AdamW wd 1e-4, 13 depth groups
with depth decay .95, gradient-norm clip 5, frequency warp p .5 scale .6-1.0 seed 50, organizer
decoder {0.65, NMS 0, 0.08 s}).  Every arm starts from parent-p2.pt.

Streams per update (seeds as P2-silver and K-A: gold 149, generated 249, silver 349, weak 51;
torch 1052):
    strong A   8 Vaani Gold + 2 generated                     weight 1.0, P2 objective
    strong B   8 Silver-stream clips                          weight 0.4, Silver mode of the arm
    weak      32 Bronze + Silver clip tags                    weight 0.1, positive-only term
                                                              (+ Bronze absent-category negative
                                                              only in an arm with Bronze)
Silver mode 'full' (both PS-R10 arms): every Silver event is a full event = P2's interval_loss
(presence + boundary 100 + IoU 1), which is the P2-silver Silver objective.  Silver mode 'R2'
(K-A, kept for the arms A/R10/A3) = the per-event modes of K1:
    full      whole-clip (>= 90 % of the clip) or >= 2.0 s: presence + boundary 100 + IoU 1
    presence  point tag < 2 s, or any event of 1-2 s: presence only, presence ignored within
              +/-80 ms of each endpoint, removed from the boundary/IoU mask of its channel
    ignore    wrap < 1 s: presence masked over [onset - .08, offset + .08] on its channel and
              removed from the boundary/IoU mask
    onset     full event, not whole-clip, onset < .02 s: no onset boundary term (to onset + .08)
    Clips without a full event carry the presence term only.  Offsets are clipped to the clip end.
Bronze (arm spec 'bronze': True): the 1,887 wrap-all rows outside the val sessions
(recipe_bronze_strong.jsonl) join the Silver pool as one full whole-clip event [0, end] on their
channel; their `ignore_channels` are masked over the whole clip (presence and boundary/IoU) in
both Silver modes.  Bronze negatives (recipe_bronze_neg.jsonl): for the Bronze rows of the weak
batch, mean over clips of mean over absent channels of -log(1 - max_t p_c), x 0.1.
Bronze off: the Silver pool has no Bronze row (= P2-silver's Silver pool, 60,343 rows in the
same order) and the weak rows carry no absent-category list (the weak pass = P2-silver's).
Uniform clip sampling in every stream.
Session filter: every training row whose session token (6th '_' field) occurs among the 1,806
natural val clips is dropped (every tier, generated beds/donors too).  NO organizer clip (val or
test, natural or synthetic) and no test clip enters any stream (asserted for every arm's pools).

Arms (one process per GPU):
    PSR10   cuda:0  Silver 'full', NO Bronze (no Bronze Silver rows, no Bronze negatives)
    PSR10B  cuda:1  Silver 'full', WITH Bronze (1,887 wrap-all rows, ignore channels masked,
                    Bronze negatives), exactly as K-A
    both:   encoder top 1e-5 with warmup .1 -> 1 over 1,000 updates (1e-6 -> 1e-5), plateau,
            cosine to .05 x peak at T (= K-A arm R10); head 1e-4, warmup .1 -> 1 over 200, WSD
            as K-A; kill rule raw t0.65 < 1.00 from u500 (as K-A R10)
What is the same and what differs: the Gold, generated and weak pools are the same in both arms
(and in K-A), so these three streams give the same clip order; PSR10B's four pools are K-A's
pools (the same pool digest), so all its streams are K-A's streams.  The Silver-stream pool of
PSR10 has 1,887 fewer rows, so its Silver order (a permutation of a shorter pool with seed 349)
differs from PSR10B/K-A; it is the Silver order of P2-silver.  The weak collision skips, the
dropout RNG and the warp draws follow the Silver draws, so they also differ between the arms.
T = min(T_max, what fits the session) is fixed by the parent from arm PSR10's speed at u300;
D = T - 500 * round(.2 T / 500) (the plateau end on the evaluation grid).

Measurement: all 1,806 organizer natural val clips at u0 (must re-measure P2 1.075846 +/- 2e-4)
and every 500 updates, decoders t0.65 and t0.75, with the diagnostics; predictions + fp16 frames
at every evaluation; fp16 weights at fixed points; last.pt + results at every evaluation;
plateau-last.pt at D; test natural predictions + frames at D and at T.  Self-stop 25 minutes
before SESSION_LIMIT_HOURS (the planned session; the push --timeout is one hour longer).

CPU session (no GPU): PROBE mode, AMP off: real mounts, pools and census, both arms as two CPU
processes with a tiny mix, R2 identity, arm-objective and Bronze checks, a forced stop + resume
equality check, evaluations on a val subset, test predictions on a test subset at D (u3) and T.

Outputs (/kaggle/working; Kaggle publishes them only when the session ends):
    results.json                  contract, census, per-arm pools, plan, checks, every trajectory
    plan.json                     T, D and their basis
    <arm>/arm-results.json        the arm's trajectory, checks and timing (merged into results.json)
    <arm>/pred-<u>-t065.jsonl     1,806-clip predictions at 0.65, and -t075.jsonl at 0.75
    <arm>/frames-<u>.npz          per-frame presence [sum T, 7] + distances [sum T, 14], fp16
    <arm>/ckpt-<u>.pt             fp16 weights at fixed points
    <arm>/last.pt                 resumable state (model + optimizer + scaler + streams + warp + RNG)
    <arm>/plateau-last.pt         resumable state at D (for a second session)
    <arm>/test-pred-<u>-t065.jsonl, -t075.jsonl, test-frames-<u>.npz   test natural at D and at T
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

# ------------------------------------------------------------------ run switches (per copy)
RUN_MODE = 'full'                  # 'full' = the run, 'smoke' = GPU smoke; no GPU => 'probe'
CONFIG = 'L'                       # 'S' (T_max 8,000) or 'L' (T_max 18,000); the plan fits T to the session
GOLD_MIX = '8+2+8'                 # '8+2+8' or '4+2+12'
ARMS = ('PSR10', 'PSR10B')         # arm on cuda:0, arm on cuda:1; each arm's Silver mode + Bronze: ARM_SPECS
SESSION_LIMIT_HOURS = 11.0         # planned session (push --timeout = this + 1 h); self-stop 25 min before
P2_SILVER_SECONDS_PER_UPDATE = 1.7424292343290109  # P2-silver results.json mean s/update (J2/J3)

RUN_TAG = 'PS-R10'
KERNEL_MARKER = 'indoml-ps-r10-kernel-v1'
# The environment names stay those of K-A (ATST_SCALE_*), so the K-A local harness works unchanged.
THREADS = int(os.environ.get('ATST_SCALE_THREADS', '2'))   # torch threads per arm process
WORKING = Path(os.environ.get('ATST_SCALE_WORKING', '/kaggle/working'))
SCRATCH = Path(os.environ.get('ATST_SCALE_TMP', '/tmp/ps-r10'))
WALL_START = time.time()
STARTED = time.monotonic()
SELF_STOP_MARGIN_S = 25 * 60       # stop 25 min before the session limit
FINAL_WORK_S = 15 * 60             # final evaluation + test predictions + saves after training
D_TEST_ALLOWANCE_S = 5 * 60        # test predictions at D (K-A: 264 s for the 3,435 test clips at T)
CONTAINER_OFFSET_S = 90            # session time spent before this script started (estimate)

T_MAX = {'S': 8000, 'L': 18000}
MIN_T = 1000
EVAL_EVERY = 500
PLAN_AT = {'full': 300, 'smoke': 50, 'probe': 2}
PLAN_REQUIRED_BY = 1000            # a child waits for plan.json at this update if it is missing
SPEED_GUARD = {'at': 300, 'window': (100, 300), 'ratio': 1.4, 'refit_at': 600, 'refit_window': (400, 600)}
U0_TOLERANCE = 2e-4
KILL_FROM = 500                    # the R10 kill rule applies from this update on
HISTORY_EVERY, LOG_EVERY_FULL, PLAN_POLL = 50, 100, (10, 50)
SMOKE_UPDATES, SMOKE_EVAL_CLIPS, SMOKE_TEST_CLIPS, SMOKE_SPEED_FROM = 100, 80, 40, 20
SMOKE_EVAL_EVERY, SMOKE_PLATEAU = 50, 50           # smoke: evaluation + D test predictions at u50
PROBE_UPDATES, PROBE_EVAL_EVERY, PROBE_STOP_AT, PROBE_PLATEAU = 6, 3, 4, 3
PROBE_EVAL_CLIPS, PROBE_TEST_CLIPS = 16, 8

DECODERS = {'t065': {'threshold': .65, 'nms_iou': 0., 'min_duration': .08},
            't075': {'threshold': .75, 'nms_iou': 0., 'min_duration': .08}}
SELECTION_DECODER = 't065'
CATEGORIES = ('animal', 'vehicle_traffic', 'baby_child', 'singing_music',
              'phone_signal_alarm', 'appliance_machine', 'human_non_speech')
PARENT_SHA = '419c441e911419b33fa066d67ffa43bb63a8c6b3c659175ccb565f0ecfb83373'
PARENT_NATURAL = {'f1': 0.3958304, 'dice': 0.6800166, 'combined': 1.0758464842826536}
FRAME_HOP, FRAME_OFFSET = .04, .02

# P2's registered constants (the training protocol of the P2 parent package), as P2-silver.
ORIGINAL = {
    'optimizer': {'type': 'AdamW', 'weight_decay': 1e-4, 'fresh_state': True, 'gradient_norm_clip': 5.0},
    'construction_lr': {'encoder_lr': 1e-6, 'head_lr': 1e-4},
    'depth_decay': .95,
    'strong_loss': {'mode': 'red', 'class_positive_weight': 5.0, 'boundary_weight': 100.0,
                    'iou_weight': 1.0, 'collar_weight': 0.0},
    'weak_loss': {'lambda': 0.1},
    'frequency_warp': {'probability': .5, 'min_scale': .6, 'max_scale': 1., 'seed': 50},
}
STRONG_WEIGHT = {'gold': 1.0, 'generated': 1.0, 'silver': 0.4}
WEAK_WEIGHT = 0.1
BRONZE_NEGATIVE_WEIGHT = 0.1
MIXES = {'8+2+8': {'gold': 8, 'generated': 2, 'silver': 8, 'weak': 32},
         '4+2+12': {'gold': 4, 'generated': 2, 'silver': 12, 'weak': 32}}
PROBE_MIX = {'gold': 2, 'generated': 1, 'silver': 2, 'weak': 4}
CAPS = {'gold': 20, 'generated': 8, 'silver': 20, 'weak': 48}      # native 10 s windows
SEEDS = {'gold': 149, 'generated': 249, 'silver': 349, 'weak': 51, 'torch': 1052}
SESSION_FIELD = 5              # the 6th underscore-separated field of a Vaani clip id
DURATION_CAP = {'silver': 25.}
R2 = {'whole_fraction': .9, 'long_seconds': 2.0, 'presence_min_seconds': 1.0,
      'band_seconds': .08, 'onset_edge_seconds': .02}
SCHEDULE = {'floor': .1, 'final': .05, 'depth_decay': .95}
SILVER_MODES = ('R2', 'full')      # 'full' = interval_loss on every Silver event (P2-silver); 'R2' = K1 modes
# One explicit spec per arm: encoder schedule, head schedule, kill rule, Silver mode, Bronze on/off.
# A, R10 and A3 are the K-A arms unchanged (K-A ran them with Silver R2 + Bronze); PSR10 and PSR10B
# are this kernel's arms.
ARM_SPECS = {
    'PSR10': {'name': 'PSR10', 'encoder_peak_top': 1e-5, 'encoder_warmup': 1000, 'head_peak': 1e-4,
              'head_warmup': 200, 'kill_below': 1.00, 'silver_mode': 'full', 'bronze': False,
              'note': 'P2-silver recipe (full Silver objective, no Bronze) at the K-A R10 encoder schedule'},
    'PSR10B': {'name': 'PSR10B', 'encoder_peak_top': 1e-5, 'encoder_warmup': 1000, 'head_peak': 1e-4,
               'head_warmup': 200, 'kill_below': 1.00, 'silver_mode': 'full', 'bronze': True,
               'note': 'as PSR10 plus the 1,887 Bronze wrap-all Silver rows and the Bronze negatives (as K-A)'},
    'A': {'name': 'A', 'encoder_peak_top': 1e-6, 'encoder_warmup': 200, 'head_peak': 1e-4,
          'head_warmup': 200, 'kill_below': None, 'silver_mode': 'R2', 'bronze': True,
          'note': 'K-A arm A: P2 init, encoder top 1e-6 (as every Silver continuation so far)'},
    'R10': {'name': 'R10', 'encoder_peak_top': 1e-5, 'encoder_warmup': 1000, 'head_peak': 1e-4,
            'head_warmup': 200, 'kill_below': 1.00, 'silver_mode': 'R2', 'bronze': True,
            'note': "K-A arm R10: P2's own final-stage encoder peak 1e-5, warmup 1e-6 -> 1e-5 over 1,000"},
    'A3': {'name': 'A3', 'encoder_peak_top': 3e-6, 'encoder_warmup': 200, 'head_peak': 1e-4,
           'head_warmup': 200, 'kill_below': None, 'silver_mode': 'R2', 'bronze': True,
           'note': 'K-A fallback for R10: encoder top 3e-6 (designer arm A3)'},
}
SPEC_KEYS = ('name', 'encoder_peak_top', 'encoder_warmup', 'head_peak', 'head_warmup', 'kill_below',
             'silver_mode', 'bronze')
KA_POOL_DIGEST = 'febb87a3be187cd53b4b4070526d1db54fcd68f4fb04771343b5eb046cc16d41'  # K-A pools (Bronze on)

# Attached input datasets (folder names under /kaggle/input or ATST_SCALE_INPUT_ROOTS). Each name can be changed
# with an environment variable T1_DS_<KEY> (e.g. T1_DS_INPUTS). The README says which script builds each folder.
DATASETS = {key: os.environ.get('T1_DS_' + key.upper(), name) for key, name in {
    'inputs': 'vaani-inputs',          # data_prep/vaani/stage_full_inputs.py
    'bundle': 'native-bundle',         # README 'Setup': the native bundle layout
    'parent': 'p2-parent',             # README 'Setup': context_crop.py, mixed_training.py, parent-p2.pt
    'indomain': 'gold-pools',          # data_prep/kaggle_inputs/prepare_gold_pools.py
    'silver': 'silver-pools',          # data_prep/kaggle_inputs/prepare_silver.py
    'test': 'test-natural',            # data_prep/kaggle_inputs/prepare_test_audio.py (no-overlap check, test predictions)
    'recipe': 'bronze-recipe',         # data_prep/recipe/build_recipe.py
}.items()}
RECIPE_SHA = {'recipe_bronze_strong.jsonl': 'd84e2bf5b94b718394090036bbeba5205158f891f010d199a47fd80b3c44d14c',
              'recipe_bronze_neg.jsonl': 'eca7efd4845dd9d3d1f3f71b6575cfb2cbb2065f8c9dd6a54b7954ad2168ca39'}
RESUME_DATASET = os.environ.get('T1_DS_RESUME', 'psr10-resume')   # optional: <arm>/last.pt of a previous session


# ------------------------------------------------------------------------ utilities

def utc():
    return datetime.now(timezone.utc).isoformat()


def session_t0():
    value = os.environ.get('ATST_SCALE_T0')
    return float(value) if value else WALL_START - CONTAINER_OFFSET_S


def session_elapsed():
    return time.time() - session_t0()


def session_limit_s():
    return SESSION_LIMIT_HOURS * 3600.


def self_stop_s():
    return session_limit_s() - SELF_STOP_MARGIN_S


def train_budget_s():
    return self_stop_s() - FINAL_WORK_S


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def log(message, **details):
    item = {'utc': utc(), 'elapsed_s': round(session_elapsed(), 1), 'message': message}
    arm = os.environ.get('ATST_SCALE_CHILD')
    if arm:
        item['arm'] = arm
    item.update(details)
    print(json.dumps(clean(item), allow_nan=False, default=str), flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=1, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strip_wav(clip_id):
    clip_id = str(clip_id)
    return clip_id[:-4] if clip_id.endswith('.wav') else clip_id


def rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except Exception:
        return None


# ------------------------------------------------------------------ organizer scorer
# Exact copy of the organizer reference code (reference_event_f1.py,
# reference_segment_dice.py), the same copy as P2-silver and the run-A/C/W kernels.

def match_events(ref_events, pred_events, tolerance_frac=0.20):
    matched_ref, matched_pred, candidates = set(), set(), []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    return matched_ref, matched_pred


def events_to_frames(events, max_time, frame_len=0.01):
    n_frames = int(max_time / frame_len) + 1
    mask = [0] * n_frames
    for on, off in events:
        for i in range(int(on / frame_len), min(int(off / frame_len) + 1, n_frames)):
            mask[i] = 1
    return mask


def clip_dice(ref_events, pred_events):
    all_events = ref_events + pred_events
    if not all_events:
        return 1.0
    max_time = max(off for _, off in all_events) + 0.5
    rm = events_to_frames(ref_events, max_time)
    pm = events_to_frames(pred_events, max_time)
    inter = sum(r & p for r, p in zip(rm, pm))
    total = sum(rm) + sum(pm)
    return 1.0 if total == 0 else 2.0 * inter / total


def score(ref, pred, clips=None):
    clips = clips if clips is not None else list(ref)
    TP = FP = FN = 0.0
    dsum = 0.0
    for clip in clips:
        matched_ref, matched_pred = match_events(ref[clip], pred.get(clip, []))
        TP += len(matched_ref)
        FP += len(pred.get(clip, [])) - len(matched_pred)
        FN += len(ref[clip]) - len(matched_ref)
        dsum += clip_dice(ref[clip], pred.get(clip, []))
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    dice = dsum / len(clips) if clips else 0.0
    return {'f1': f1, 'dice': dice, 'combined': f1 + dice, 'tp': TP, 'fp': FP, 'fn': FN,
            'precision': p, 'recall': r}


def match_pairs(ref_events, pred_events, tolerance_frac=0.20):
    """The organizer matcher (same greedy order), returning the matched (ref, pred) pairs."""
    matched_ref, matched_pred, candidates, pairs = set(), set(), [], []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
            pairs.append((ri, pi))
    return pairs


DURATION_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9))
BUCKET_NAMES = ('<0.25', '0.25-0.5', '0.5-1', '1-2', '2-5', '5+')
TYPES = ('W', 'L', 'M', 'S', 'XS')
LONG_SECONDS = 2.0
SHORT_SECONDS = 0.5


def bucket_of(duration):
    for index, (low, high) in enumerate(DURATION_BUCKETS):
        if low <= duration < high:
            return index
    return len(DURATION_BUCKETS) - 1


def ref_type(on, off, clip_duration):
    """Reference-event type: W >= 90 % of the clip, L >= 2 s, M 1-2, S .5-1, XS."""
    d = off - on
    if clip_duration > 0 and d / clip_duration >= 0.9:
        return 'W'
    if d >= 2.0:
        return 'L'
    if d >= 1.0:
        return 'M'
    if d >= 0.5:
        return 'S'
    return 'XS'


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else .5 * (ordered[middle - 1] + ordered[middle])


def coverage_of(reference, predicted):
    on, off = reference
    span = off - on
    if span <= 0:
        return None
    pieces = sorted((max(on, a), min(off, b)) for a, b in predicted if b > on and a < off)
    covered, cursor = 0., on
    for a, b in pieces:
        if b <= cursor:
            continue
        covered += b - max(a, cursor)
        cursor = b
    return min(1., covered / span)


def diagnostics(reference, predicted, clips, tags=None, durations=None):
    """P2-silver's diagnostics plus type x kind recall, long predictions and endpoint errors."""
    hits = [0] * len(DURATION_BUCKETS)
    totals = [0] * len(DURATION_BUCKETS)
    ref_seconds = pred_seconds = 0.0
    ref_lengths, pred_lengths = [], []
    long_cov, long_cov_plain, long_hits, long_total = [], [], 0, 0
    long_hits_plain, long_total_plain = 0, 0
    short_pred_total = short_pred_hit = 0
    long_pred_total = long_pred_hit = 0
    type_kind_total, type_kind_hit = Counter(), Counter()
    onset_errors = [[] for _ in DURATION_BUCKETS]
    offset_errors = [[] for _ in DURATION_BUCKETS]
    for clip in clips:
        refs = list(reference.get(clip, []))
        preds = list(predicted.get(clip, []))
        clip_tags = list((tags or {}).get(clip) or [''] * len(refs))
        clip_duration = float((durations or {}).get(clip) or 0.)
        pairs = match_pairs(refs, preds)
        matched = {ri for ri, _ in pairs}
        matched_pred = {pi for _, pi in pairs}
        for ri, pi in pairs:
            bucket = bucket_of(refs[ri][1] - refs[ri][0])
            onset_errors[bucket].append(abs(preds[pi][0] - refs[ri][0]))
            offset_errors[bucket].append(abs(preds[pi][1] - refs[ri][1]))
        for index, (on, off) in enumerate(refs):
            length = off - on
            bucket = bucket_of(length)
            totals[bucket] += 1
            ref_seconds += length
            ref_lengths.append(length)
            if index in matched:
                hits[bucket] += 1
            tag = (clip_tags[index] if index < len(clip_tags) else '') or ''
            kind = 'point' if tag.strip().startswith('[') else 'wrap'
            key = ref_type(on, off, clip_duration) + '|' + kind
            type_kind_total[key] += 1
            type_kind_hit[key] += int(index in matched)
            if length >= LONG_SECONDS:
                cover = coverage_of((on, off), preds)
                long_total += 1
                long_hits += int(index in matched)
                if cover is not None:
                    long_cov.append(cover)
                if not tag.strip().startswith('['):
                    long_total_plain += 1
                    long_hits_plain += int(index in matched)
                    if cover is not None:
                        long_cov_plain.append(cover)
        for index, (on, off) in enumerate(preds):
            pred_seconds += off - on
            pred_lengths.append(off - on)
            if off - on < SHORT_SECONDS:
                short_pred_total += 1
                short_pred_hit += int(index in matched_pred)
            if off - on >= LONG_SECONDS:
                long_pred_total += 1
                long_pred_hit += int(index in matched_pred)
    return {'bucket_names': list(BUCKET_NAMES), 'reference_events_per_bucket': totals,
            'matched_events_per_bucket': hits,
            'recall_per_bucket': [(hits[i] / totals[i]) if totals[i] else None for i in range(len(totals))],
            'type_kind_references': dict(type_kind_total),
            'type_kind_recall': {k: type_kind_hit[k] / type_kind_total[k] for k in sorted(type_kind_total)},
            'median_abs_onset_error_matched_per_bucket': [median(v) for v in onset_errors],
            'median_abs_offset_error_matched_per_bucket': [median(v) for v in offset_errors],
            'long_reference_seconds_threshold': LONG_SECONDS,
            'long_references': long_total,
            'long_median_coverage': median(long_cov),
            'long_recall': (long_hits / long_total) if long_total else None,
            'long_references_non_bracketed': long_total_plain,
            'long_median_coverage_non_bracketed': median(long_cov_plain),
            'long_recall_non_bracketed': (long_hits_plain / long_total_plain) if long_total_plain else None,
            'long_predictions': long_pred_total,
            'long_prediction_precision': (long_pred_hit / long_pred_total) if long_pred_total else None,
            'short_prediction_seconds_threshold': SHORT_SECONDS,
            'short_predictions': short_pred_total,
            'short_prediction_precision': (short_pred_hit / short_pred_total) if short_pred_total else None,
            'reference_seconds': ref_seconds, 'predicted_seconds': pred_seconds,
            'predicted_over_reference_seconds': (pred_seconds / ref_seconds) if ref_seconds else None,
            'median_reference_duration': median(ref_lengths),
            'median_predicted_duration': median(pred_lengths),
            'reference_events': len(ref_lengths), 'predicted_events': len(pred_lengths),
            'clips': len(clips)}


# ------------------------------------------------------------------------- mounting

def input_roots():
    roots = [Path(p) for p in os.environ.get('ATST_SCALE_INPUT_ROOTS', '').split(':') if p]
    return roots + [Path('/kaggle/input')]


def dataset_root(slug, required=True):
    for base in input_roots():
        candidate = base / slug
        if candidate.is_dir():
            return candidate.resolve()
    if required:
        raise SystemExit(f'Dataset is not mounted: {slug}')
    return None


def find_one(root, name):
    found = [root / name] if (root / name).is_file() else sorted(root.rglob(name))
    require(len(found) == 1, f'Expected exactly one {name} under {root}, found {len(found)}')
    return found[0]


def setup_paths():
    bundle = dataset_root(DATASETS['bundle'])
    native = bundle / 'bundle' if (bundle / 'bundle/asset-manifest.json').is_file() else bundle
    for entry in (str(native / 'source'), str(dataset_root(DATASETS['parent']))):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return native


class InputLayout:
    """Kaggle expands each archive under its own stem, or flat. Detect which (P2-silver)."""

    def __init__(self, root, contract, training, inference):
        self.root = Path(root).resolve()
        self.shards, cursor = [], 0
        for entry in contract['archives']:
            if 'training_clips' not in entry:
                continue
            count = entry['training_clips']
            stem = Path(entry['path']).stem
            prefix = self._choose_prefix(training[cursor]['audio_path'], stem)
            self.shards.append({'first_row': cursor, 'end_row': cursor + count,
                                'archive_stem': stem, 'prefix': prefix})
            cursor += count
        if cursor != len(training) or len(self.shards) != 5:
            raise ValueError('Training archive population changed')
        organizer = [e for e in contract['archives'] if 'organizer_noisy_clips' in e]
        if len(organizer) != 1 or organizer[0]['organizer_noisy_clips'] != len(inference):
            raise ValueError('Organizer archive population changed')
        self.organizer_prefix = self._choose_prefix(inference[0]['audio_path'],
                                                    Path(organizer[0]['path']).stem)
        self.ends = [item['end_row'] for item in self.shards]

    def _path(self, relative, prefix):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Input path must stay relative')
        path = (self.root / prefix / relative).resolve()
        if self.root not in path.parents:
            raise ValueError('Resolved input escaped its dataset root')
        return path

    def _choose_prefix(self, relative, stem):
        choices = [prefix for prefix in ('', stem) if self._path(relative, prefix).is_file()]
        if len(choices) != 1:
            raise ValueError(f'Expected one archive-layout sentinel for {relative}')
        return choices[0]

    def training_path(self, index, row):
        shard = bisect_right(self.ends, index)
        return self._path(row['audio_path'], self.shards[shard]['prefix'])

    def organizer_path(self, row):
        return self._path(row['audio_path'], self.organizer_prefix)


def session_token(clip_id):
    parts = str(clip_id).split('_')
    return parts[SESSION_FIELD] if len(parts) > SESSION_FIELD else None


# ----------------------------------------------------------------------------- data

def native_windows(duration):
    from indoml.pretrained import window_starts
    return len(window_starts(round(duration * 16000)))


def training_row(clip_id, audio_path, duration, events, categories, *, supervision='strong_gold',
                 quality='verified_timestamps', origin='vaani_gold', windows=None):
    return {'clip_id': clip_id, 'audio_path': str(audio_path), 'duration': float(duration),
            'num_frames': (round(duration * 16000) + 639) // 640,
            'frame_hop': FRAME_HOP, 'frame_offset': FRAME_OFFSET, 'feature_dim': 768,
            'events': events, 'positive_categories': sorted(categories or []),
            'quality': quality, 'supervision': supervision, 'origin': origin,
            'native_windows': windows if windows is not None else native_windows(duration)}


def inference_row(clip_id, audio_path, duration):
    return {'clip_id': clip_id, 'audio_path': str(audio_path), 'duration': float(duration),
            'num_frames': (round(duration * 16000) + 639) // 640,
            'frame_hop': FRAME_HOP, 'frame_offset': FRAME_OFFSET, 'feature_dim': 768}


def r2_modes(row):
    """Per-event Silver mode of the R2 recipe (K1 kernel-wl-silver, unchanged)."""
    duration = row['duration']
    modes = []
    for event in row['events']:
        length = event['offset'] - event['onset']
        whole = length >= R2['whole_fraction'] * duration
        tag = str(event.get('tag') or '')
        kind = 'point' if tag.startswith('[') else 'wrap'
        if whole or length >= R2['long_seconds']:
            mode = 'full'
        elif kind == 'point' or length >= R2['presence_min_seconds']:
            mode = 'presence'
        else:
            mode = 'ignore'
        onset_mask = mode == 'full' and not whole and event['onset'] < R2['onset_edge_seconds']
        modes.append({'mode': mode, 'kind': kind, 'whole': whole, 'onset_mask': onset_mask})
    return modes


def verify_recipe():
    root = dataset_root(DATASETS['recipe'])
    files, digests = {}, {}
    for name, expected in RECIPE_SHA.items():
        path = find_one(root, name)
        digests[name] = sha256_of(path)
        require(digests[name] == expected, f'{name}: sha256 {digests[name]} != {expected}')
        files[name] = path
    return files, digests


def build_pools(mode):
    """P2-silver's pools (unchanged Gold/generated/weak/Silver rules) + Bronze wrap-all rows in
    the Silver pool + Bronze absent-category lists on the weak Bronze rows + test rows."""
    inputs = dataset_root(DATASETS['inputs'])
    indomain = dataset_root(DATASETS['indomain'])
    silver_root = dataset_root(DATASETS['silver'])
    test_root = dataset_root(DATASETS['test'])
    recipe_files, recipe_digests = verify_recipe()
    contract = json.loads((inputs / 'full-input-contract.json').read_text())
    training = list(read_jsonl(inputs / 'training.jsonl'))
    inference = list(read_jsonl(inputs / 'organizer-inference.jsonl'))
    references = {row['clip_id']: row['events']
                  for row in read_jsonl(inputs / 'organizer-references.jsonl')}
    require((len(training), len(inference), len(references)) == (90635, 2501, 2501), 'Manifest sizes changed')
    layout = InputLayout(inputs, contract, training, inference)
    folds = json.loads((indomain / 'folds.json').read_text())
    census = {'dropped': {}, 'recipe_sha256': recipe_digests}

    def drop(key, count=1):
        census['dropped'][key] = census['dropped'].get(key, 0) + count

    natural_fold = folds['clip_fold']
    require(len(natural_fold) == 1806, f'Expected 1,806 natural clips, got {len(natural_fold)}')
    natural_sessions = {session_token(clip_id) for clip_id in natural_fold}
    natural_sessions.discard(None)
    census['natural_sessions'] = len(natural_sessions)
    organizer_ids = {row['clip_id'] for row in inference}
    require(set(natural_fold) <= organizer_ids, 'Natural clips must be organizer clips')

    test_manifest = list(read_jsonl(test_root / 'test-natural-manifest.jsonl'))
    test_prefixes = [p for p in ('', 'test-audio') if (test_root / p / test_manifest[0]['audio_path']).is_file()]
    require(len(test_prefixes) == 1, 'Expected exactly one test-audio layout sentinel')
    test_ids = {row['clip_id'] for row in test_manifest}
    require(len(test_manifest) == 3435 == len(test_ids), 'Expected 3,435 test natural clips')
    test_rows = [inference_row(r['clip_id'], test_root / test_prefixes[0] / r['audio_path'], r['duration'])
                 for r in test_manifest]

    silver_labels = {row['clip_id']: row for row in read_jsonl(silver_root / 'silver-events.jsonl')}
    gen_sessions = json.loads((silver_root / 'gen-sessions.json').read_text())
    census['silver_labels'] = len(silver_labels)
    bronze_strong = {strip_wav(r['clip_id']): r for r in read_jsonl(recipe_files['recipe_bronze_strong.jsonl'])}
    bronze_neg = {strip_wav(r['clip_id']): r for r in read_jsonl(recipe_files['recipe_bronze_neg.jsonl'])}
    census['recipe_rows'] = {'bronze_strong': len(bronze_strong), 'bronze_neg': len(bronze_neg)}

    def usable_events(events, duration, key):
        out = []
        for event in events or []:
            onset, offset = float(event['onset']), min(float(event['offset']), float(duration))
            if event['category'] not in CATEGORIES:
                drop(key + '_unknown_category')
                return None
            if not (0. <= onset < offset):
                drop(key + '_degenerate_event')
                return None
            out.append({'onset': onset, 'offset': offset, 'category': event['category'],
                        'tag': str(event.get('tag') or '')})
        return out or None

    gold_rows, silver_rows, bronze_rows, weak_rows = [], [], [], []
    bronze_seen = Counter()
    for index, row in enumerate(training):
        clip_id = row['clip_id']
        require(clip_id not in organizer_ids and clip_id not in test_ids,
                f'Training row is an organizer/test clip: {clip_id}')
        if session_token(clip_id) in natural_sessions:
            drop('session_in_natural_' + str(row['quality']))
            continue
        if row['supervision'] == 'strong_gold':
            # EVERY Gold row outside the val sessions, as P2-silver.
            if row['native_windows'] > CAPS['gold']:
                drop('gold_too_many_windows')
                continue
            events = usable_events(row['events'], row['duration'], 'gold')
            if not events:
                drop('gold_unusable_events')
                continue
            gold_rows.append(training_row(clip_id, layout.training_path(index, row),
                                          row['duration'], events, row['positive_categories'],
                                          windows=row['native_windows']))
        elif row['supervision'] == 'weak_clip':
            categories = [c for c in (row['positive_categories'] or []) if c in CATEGORIES]
            label = silver_labels.get(clip_id)
            if label is not None:
                if row['native_windows'] > CAPS['silver'] or row['duration'] > DURATION_CAP['silver']:
                    drop('silver_too_long')
                else:
                    events = usable_events(label['events'], row['duration'], 'silver')
                    if events:
                        item = training_row(clip_id, layout.training_path(index, row), row['duration'],
                                            events, label.get('positive_categories') or categories,
                                            quality='unverified_timestamps', origin='vaani_silver_strong',
                                            windows=row['native_windows'])
                        item['r2'] = r2_modes(item)
                        item['ignore_channels'] = []
                        silver_rows.append(item)
            key = strip_wav(clip_id)
            absent = None
            if row['quality'] == 'no_timestamps':
                bronze_seen['rows'] += 1
                strong = bronze_strong.get(key)
                if strong is not None:
                    bronze_seen['wrap_all_rows'] += 1
                    events = usable_events(strong['events'], row['duration'], 'bronze_strong')
                    if strong.get('val_session'):
                        drop('bronze_strong_val_session_flag')
                    elif abs(float(strong['duration']) - float(row['duration'])) > .011:
                        drop('bronze_strong_duration_mismatch')
                    elif row['native_windows'] > CAPS['silver'] or row['duration'] > DURATION_CAP['silver']:
                        drop('bronze_strong_too_long')
                    elif not events:
                        drop('bronze_strong_unusable')
                    else:
                        item = training_row(clip_id, layout.training_path(index, row), row['duration'],
                                            events, sorted({e['category'] for e in events}),
                                            quality='bronze_wrap_all', origin='vaani_bronze_wrap_all',
                                            windows=row['native_windows'])
                        item['r2'] = r2_modes(item)
                        own = {e['category'] for e in events}
                        item['ignore_channels'] = sorted(c for c in (strong.get('ignore_channels') or [])
                                                         if c in CATEGORIES and c not in own)
                        if all(m['mode'] == 'full' and m['whole'] for m in item['r2']):
                            bronze_rows.append(item)
                        else:
                            drop('bronze_strong_not_whole_clip')
                neg = bronze_neg.get(key)
                if neg is None:
                    bronze_seen['without_negative_record'] += 1
                else:
                    listed = [c for c in (neg.get('absent_negative_everywhere') or []) if c in CATEGORIES]
                    absent = sorted(c for c in listed if c not in categories)
                    bronze_seen['negative_conflicts_removed'] += len(listed) - len(absent)
            if not categories:
                drop('weak_no_category')
                continue
            if row['native_windows'] > CAPS['weak']:
                drop('weak_too_many_windows')
                continue
            item = training_row(clip_id, layout.training_path(index, row), row['duration'], None, categories,
                                supervision='weak_clip', quality=row['quality'],
                                origin='vaani_' + str(row['quality']), windows=row['native_windows'])
            if absent:
                item['bronze_absent'] = absent
            weak_rows.append(item)

    # Organizer clips are the MEASUREMENT set only.
    eval_rows, eval_reference, eval_tags, eval_durations = [], {}, {}, {}
    duration_of = {row['clip_id']: row['duration'] for row in inference}
    path_of = {row['clip_id']: layout.organizer_path(row) for row in inference}
    for clip_id in sorted(duration_of):
        if clip_id not in natural_fold:
            drop('organizer_synthetic_excluded_everywhere')
            continue
        eval_rows.append(inference_row(clip_id, path_of[clip_id], duration_of[clip_id]))
        events = references.get(clip_id) or []
        eval_reference[clip_id] = [(float(e['onset']), float(e['offset'])) for e in events]
        eval_tags[clip_id] = [str(e.get('tag') or '') for e in events]
        eval_durations[clip_id] = float(duration_of[clip_id])
    require(len(eval_rows) == 1806, f'Expected 1,806 natural clips, got {len(eval_rows)}')

    gen_rows = []
    gen_manifest = list(read_jsonl(indomain / 'gen-manifest.jsonl'))
    gen_prefixes = [p for p in ('', 'gen-audio') if (indomain / p / gen_manifest[0]['audio_path']).is_file()]
    require(len(gen_prefixes) == 1, 'Expected exactly one generated-audio layout sentinel')
    gen_prefix = gen_prefixes[0]
    census['generated_audio_prefix'] = gen_prefix
    for record in gen_manifest:
        path = indomain / gen_prefix / record['audio_path']
        if not path.is_file():
            drop('generated_missing_audio')
            continue
        tokens = gen_sessions.get(record['clip_id'])
        if tokens is None:
            drop('generated_without_session_record')
            continue
        if natural_sessions & set(tokens):
            drop('generated_session_in_natural')
            continue
        row = training_row(record['clip_id'], path, record['duration'], record['events'],
                           record['positive_categories'], origin='generated_' + record['recipe'])
        if row['native_windows'] > CAPS['generated']:
            drop('generated_too_many_windows')
            continue
        row['recipe'] = record['recipe']
        gen_rows.append(row)

    pools = {'gold': gold_rows, 'silver': silver_rows + bronze_rows, 'generated': gen_rows, 'weak': weak_rows}
    train_ids = {row['clip_id'] for rows in pools.values() for row in rows}
    require(not train_ids & organizer_ids, 'An organizer clip is in a training stream')
    require(not train_ids & test_ids, 'A test clip is in a training stream')
    require(not {session_token(r['clip_id']) for n in ('gold', 'silver', 'weak') for r in pools[n]}
            & natural_sessions, 'Session filter failed')
    for name, rows in pools.items():
        require(len({r['clip_id'] for r in rows}) == len(rows), f'Duplicate clip in pool {name}')
    silver_modes = Counter(m['mode'] for r in silver_rows for m in r['r2'])
    census.update({name: len(rows) for name, rows in pools.items()},
                  silver_only=len(silver_rows), bronze_wrap_all_in_silver=len(bronze_rows),
                  bronze_ignore_channel_rows=sum(1 for r in bronze_rows if r['ignore_channels']),
                  bronze=dict(bronze_seen),
                  weak_bronze_rows_with_negatives=sum(1 for r in weak_rows if r.get('bronze_absent')),
                  weak_bronze_absent_channels=sum(len(r.get('bronze_absent') or []) for r in weak_rows),
                  silver_r2_modes=dict(silver_modes),
                  silver_r2_onset_masked=sum(m['onset_mask'] for r in silver_rows for m in r['r2']),
                  silver_r2_kinds=dict(Counter(m['kind'] for r in silver_rows for m in r['r2'])),
                  silver_clips_with_full_event=sum(any(m['mode'] == 'full' for m in r['r2']) for r in silver_rows),
                  silver_events=sum(len(r['events']) for r in silver_rows),
                  eval_clips=len(eval_rows), test_clips=len(test_rows),
                  eval_reference_events=sum(len(v) for v in eval_reference.values()),
                  gold_clips_in_manifest=sum(1 for r in training if r['supervision'] == 'strong_gold'),
                  generated_recipes=dict(Counter(r['recipe'] for r in gen_rows)),
                  weak_quality=dict(Counter(r['quality'] for r in weak_rows)),
                  silver_pool_quality=dict(Counter(r['quality'] for r in pools['silver'])),
                  caps=CAPS, session_field=SESSION_FIELD,
                  organizer_clips_in_training_streams=0, test_clips_in_training_streams=0,
                  windows_per_clip={name: round(sum(r['native_windows'] for r in rows) / max(1, len(rows)), 3)
                                    for name, rows in pools.items()},
                  mean_duration={name: round(sum(r['duration'] for r in rows) / max(1, len(rows)), 3)
                                 for name, rows in pools.items()})
    digest = pools_digest(pools)
    census['pool_digest'] = digest
    census['same_pools_as_ka'] = digest == KA_POOL_DIGEST
    return {'pools': pools, 'census': census, 'eval_rows': eval_rows, 'eval_reference': eval_reference,
            'eval_tags': eval_tags, 'eval_durations': eval_durations, 'test_rows': test_rows,
            'check_rows': check_rows(pools), 'pool_digest': digest,
            'organizer_ids': sorted(organizer_ids), 'natural_sessions': sorted(natural_sessions)}


def pools_digest(pools):
    """sha256 of every pool's clip ids in pool order (K-A's pool digest)."""
    return hashlib.sha256(json.dumps({name: [r['clip_id'] for r in rows] for name, rows in sorted(pools.items())}
                                     ).encode()).hexdigest()


def check_spec(spec):
    """An arm spec must say every schedule value, its Silver mode and Bronze on/off."""
    missing = sorted(set(SPEC_KEYS) - set(spec))
    require(not missing, f'Arm spec {spec.get("name")} misses {missing}')
    require(spec['silver_mode'] in SILVER_MODES, f"Arm {spec['name']}: unknown Silver mode {spec['silver_mode']!r}")
    require(isinstance(spec['bronze'], bool), f"Arm {spec['name']}: bronze must be True or False")
    require(all(spec[k] > 0 for k in ('encoder_peak_top', 'encoder_warmup', 'head_peak', 'head_warmup')),
            f"Arm {spec['name']}: rates and warmups must be positive")
    return spec


def spec_core(spec):
    return {key: spec[key] for key in SPEC_KEYS}


def arm_view(data, spec):
    """The pools of one arm.  Bronze on: the full build (= K-A's pools).  Bronze off: the Silver pool
    without the Bronze wrap-all rows (the other rows keep their order: P2-silver's Silver pool) and the
    same weak rows in the same order without their absent-category lists (P2-silver's weak pool).  The
    leak checks run again on the arm's pools; the arm's census, pool digest and check rows go with it."""
    check_spec(spec)
    full = data['pools']
    pools = dict(full)
    if not spec['bronze']:
        pools['silver'] = [r for r in full['silver'] if r['quality'] != 'bronze_wrap_all']
        pools['weak'] = [{k: v for k, v in r.items() if k != 'bronze_absent'} if 'bronze_absent' in r else r
                         for r in full['weak']]
    name = spec['name']
    organizer_ids, natural_sessions = set(data['organizer_ids']), set(data['natural_sessions'])
    test_ids = {r['clip_id'] for r in data['test_rows']}
    train_ids = {r['clip_id'] for rows in pools.values() for r in rows}
    require(not train_ids & organizer_ids, f'{name}: an organizer clip is in a training stream')
    require(not train_ids & test_ids, f'{name}: a test clip is in a training stream')
    require(not {session_token(r['clip_id']) for n in ('gold', 'silver', 'weak') for r in pools[n]}
            & natural_sessions, f'{name}: session filter failed')
    for pool, rows in pools.items():
        require(len({r['clip_id'] for r in rows}) == len(rows), f'{name}: duplicate clip in pool {pool}')
    absent_rows = sum(1 for r in pools['weak'] if r.get('bronze_absent'))
    census = {'arm': name, 'silver_mode': spec['silver_mode'], 'bronze': spec['bronze'],
              **{pool: len(rows) for pool, rows in pools.items()},
              'silver_pool_quality': dict(Counter(r['quality'] for r in pools['silver'])),
              'silver_bronze_rows': sum(1 for r in pools['silver'] if r['quality'] == 'bronze_wrap_all'),
              'silver_ignore_channel_rows': sum(1 for r in pools['silver'] if r.get('ignore_channels')),
              'weak_rows_with_bronze_absent': absent_rows,
              'weak_bronze_absent_channels': sum(len(r.get('bronze_absent') or []) for r in pools['weak']),
              'removed_silver_rows': len(full['silver']) - len(pools['silver']),
              'removed_weak_absent_lists': sum(1 for r in full['weak'] if r.get('bronze_absent')) - absent_rows,
              'silver_events': sum(len(r['events']) for r in pools['silver']),
              'weak_quality': dict(Counter(r['quality'] for r in pools['weak'])),
              'organizer_clips_in_training_streams': 0, 'test_clips_in_training_streams': 0,
              'val_sessions_in_training_streams': 0, 'pool_digest': pools_digest(pools)}
    census['same_pools_as_ka'] = census['pool_digest'] == KA_POOL_DIGEST
    return dict(data, pools=pools, check_rows=check_rows(pools), pool_digest=census['pool_digest'],
                arm_census=census)


def check_rows(pools):
    """Deterministic small row sets for the startup checks."""
    def key(row):
        return hashlib.sha256(row['clip_id'].encode()).hexdigest()
    silver = sorted((r for r in pools['silver'] if r['quality'] == 'unverified_timestamps'), key=key)
    chosen, flags_needed = [], ['full', 'presence', 'ignore', 'onset']
    for row in silver:
        if row['native_windows'] > 2:
            continue
        flags = {m['mode'] for m in row['r2']} | ({'onset'} if any(m['onset_mask'] for m in row['r2']) else set())
        if flags_needed and flags & set(flags_needed):
            chosen.append(row)
            for flag in flags:
                if flag in flags_needed:
                    flags_needed.remove(flag)
        if len(chosen) == 4:
            break
    for row in silver:
        if len(chosen) >= 4:
            break
        if row not in chosen and row['native_windows'] == 1:
            chosen.append(row)
    bronze = sorted((r for r in pools['silver'] if r['quality'] == 'bronze_wrap_all'
                     and r['ignore_channels'] and r['native_windows'] == 1), key=key)[:2]
    weak_bronze = sorted((r for r in pools['weak'] if r.get('bronze_absent') and r['native_windows'] == 1),
                         key=key)[:4]
    return {'silver': chosen, 'bronze_strong': bronze, 'weak_bronze': weak_bronze}


# ---------------------------------------------------------------------------- model

def load_parent_model(device):
    import context_crop
    import mixed_training
    from indoml import atst_intervals as api
    native = setup_paths()
    parent = dataset_root(DATASETS['parent']) / 'parent-p2.pt'
    model, metadata = context_crop.load_parent(api, native, parent, device)
    if not getattr(model, '_full_release_fp32_head', False):
        mixed_training.enforce_fp32_head(model)      # P2's own precondition for a mixed update
    return model, {'initialisation': 'parent_p2', 'parent_metadata': metadata, 'parent_sha256': PARENT_SHA,
                   'fp32_head': bool(getattr(model, '_full_release_fp32_head', False))}


def depth_optimizer(model, api, weight_decay, encoder_lr, head_lr):
    from types import SimpleNamespace
    from indoml.optimization import depth_parameter_names, regroup_depth, parameter_names
    settings = SimpleNamespace(encoder_lr=encoder_lr, head_lr=head_lr, weight_decay=weight_decay)
    optimizer = api.make_optimizer(model, settings)
    expected = {key: sorted(names) for key, names in depth_parameter_names(model).items()}
    regroup_depth(optimizer, model, expected)
    return optimizer, parameter_names(optimizer, model)


def plateau_end(total):
    return total - 500 * round(0.2 * total / 500)


def wsd_fraction(update, warmup, plateau, total):
    """Warmup floor -> 1 (linear), 1 until the plateau end D, cosine to `final` at T."""
    if update < warmup:
        return SCHEDULE['floor'] + (1 - SCHEDULE['floor']) * update / warmup
    if update < plateau:
        return 1.0
    progress = min(1.0, (update - plateau) / max(1, total - plateau))
    return SCHEDULE['final'] + (1 - SCHEDULE['final']) * .5 * (1 + math.cos(math.pi * progress))


def arm_rates(spec, update, total, plateau):
    encoder = wsd_fraction(update, spec['encoder_warmup'], plateau, total)
    head = wsd_fraction(update, spec['head_warmup'], plateau, total)
    rates = {str(depth): spec['encoder_peak_top'] * SCHEDULE['depth_decay'] ** (11 - depth) * encoder
             for depth in range(12)}
    rates['head'] = spec['head_peak'] * head
    return rates


def apply_rates(optimizer, spec, update, total, plateau):
    rates = arm_rates(spec, update, total, plateau)
    require({group['name'] for group in optimizer.param_groups} == set(rates), 'Depth optimizer groups changed')
    for group in optimizer.param_groups:
        group['lr'] = rates[group['name']]
    return rates


def weight_points(total, plateau):
    points = {1000, 2000, 3000, 4000, plateau, total}
    points |= set(range(2000, plateau + 1, 2000))
    points |= {u for u in range(1000, total + 1, 1000) if u > plateau}
    return sorted(p for p in points if 0 < p <= total)


def compute_plan(rate_s, eval_s, completed, elapsed_s, t_max, basis):
    """T that fits the training budget; the test predictions at D (D_TEST_ALLOWANCE_S) come off it."""
    per_update = rate_s + (eval_s + 25.) / EVAL_EVERY
    remaining = train_budget_s() - elapsed_s - D_TEST_ALLOWANCE_S
    fit = max(0, int(remaining / per_update)) if per_update > 0 else 0
    total = min(t_max, 500 * ((completed + fit) // 500))
    total = max(total, MIN_T, 500 * ((completed + 499) // 500 + 1))
    total = min(total, t_max) if completed < t_max else total
    return {'T': int(total), 'D': int(plateau_end(total)), 'rate_s': rate_s, 'eval_s': eval_s,
            'per_update_s': per_update, 'completed_at_plan': completed, 'elapsed_s_at_plan': elapsed_s,
            'train_budget_s': train_budget_s(), 'd_test_allowance_s': D_TEST_ALLOWANCE_S, 'fit_updates': fit,
            't_max': t_max, 'basis': basis, 'utc': utc()}


# ----------------------------------------------------------------------- prediction

def packs(rows, cap=4):
    current, windows = [], 0
    for row in rows:
        count = 1 + max(0, (row['num_frames'] - 250 + 124) // 125)
        if current and windows + count > cap:
            yield current
            current, windows = [], 0
        current.append(row)
        windows += count
    if current:
        yield current


def predict_outputs(model, api, rows, cap=4, keep_frames=False):
    """P2's inference chain, decoder-independent: candidates per row (+ raw frames)."""
    import numpy as np
    import torch
    previous = getattr(model, 'frequency_augmentation', None)
    model.eval()
    model.frequency_augmentation = None

    def as_array(value):
        if hasattr(value, 'detach'):
            value = value.detach().to('cpu').float().numpy()
        return np.asarray(value, dtype=np.float32)

    records, frames = [], {}
    with torch.inference_mode():
        for selected in packs(rows, cap):
            for row, values in api.infer_outputs(model, selected, len(selected)):
                presence = as_array(values['presence'])
                distances = as_array(values['distances'])
                records.append((row, api.candidate_intervals(row, presence, distances)))
                if keep_frames:
                    frames[row['clip_id']] = (presence.astype(np.float16), distances.astype(np.float16))
    model.frequency_augmentation = previous
    return records, frames


def decode_all(api, records, decoder):
    out = {}
    for row, candidates in records:
        record = api.decode_candidates(row, candidates, decoder)
        out[record['clip_id']] = [(e['onset'], e['offset']) for e in record['events']]
    return out


def save_frames(path, frames, rows):
    import numpy as np
    clip_ids, counts, presence, distances = [], [], [], []
    for row in rows:
        item = frames.get(row['clip_id'])
        if item is None:
            continue
        clip_ids.append(row['clip_id'])
        counts.append(item[0].shape[0])
        presence.append(item[0])
        distances.append(item[1].reshape(item[1].shape[0], -1))
    np.savez_compressed(path, clip_ids=np.array(clip_ids), frame_counts=np.array(counts, np.int32),
                        presence=np.concatenate(presence, axis=0),
                        distances=np.concatenate(distances, axis=0),
                        frame_hop=np.float32(FRAME_HOP), frame_offset=np.float32(FRAME_OFFSET),
                        categories=np.array(CATEGORIES),
                        layout=np.array('presence [sum(T), 7]; distances [sum(T), 14] = '
                                        '(7 categories x [onset_distance, offset_distance]); '
                                        'split by frame_counts in clip_ids order'))


def write_predictions(path, clips, predicted):
    with Path(path).open('w') as stream:
        for clip in clips:
            stream.write(json.dumps({'clip_id': clip, 'events': [{'onset': a, 'offset': b}
                                                                 for a, b in predicted.get(clip, [])]},
                                    allow_nan=False) + '\n')


def evaluate(model, api, data, rows, update, arm_dir):
    started = time.monotonic()
    records, frames = predict_outputs(model, api, rows, keep_frames=True)
    clips = sorted(row['clip_id'] for row in rows)
    reference = {clip: data['eval_reference'][clip] for clip in clips}
    item = {'update': update, 'utc': utc(), 'clips': len(clips), 'decoders': {}}
    for name, decoder in DECODERS.items():
        predicted = decode_all(api, records, decoder)
        values = score(reference, predicted, clips=clips)
        item['decoders'][name] = {
            'decoder': decoder,
            'natural1806': {k: values[k] for k in ('f1', 'dice', 'combined', 'tp', 'fp', 'fn',
                                                   'precision', 'recall')},
            'natural1806_vs_parent': values['combined'] - PARENT_NATURAL['combined'],
            'natural1806_diagnostics': diagnostics(reference, predicted, clips, data['eval_tags'],
                                                   data['eval_durations'])}
        path = arm_dir / f'pred-{update:06d}-{name}.jsonl'
        write_predictions(path, clips, predicted)
        item['decoders'][name]['predictions'] = path.name
    save_frames(arm_dir / f'frames-{update:06d}.npz', frames, rows)
    item['frames'] = f'frames-{update:06d}.npz'
    item['selection_decoder'] = SELECTION_DECODER
    item['selection_combined'] = item['decoders'][SELECTION_DECODER]['natural1806']['combined']
    item['eval_seconds'] = time.monotonic() - started
    primary = item['decoders'][SELECTION_DECODER]
    diag = primary['natural1806_diagnostics']
    log('evaluation', update=update, clips=len(clips),
        natural_t065=round(primary['natural1806']['combined'], 6),
        f1=round(primary['natural1806']['f1'], 6), dice=round(primary['natural1806']['dice'], 6),
        natural_t075=round(item['decoders']['t075']['natural1806']['combined'], 6),
        recall_per_bucket=[None if v is None else round(v, 4) for v in diag['recall_per_bucket']],
        long_coverage=diag['long_median_coverage_non_bracketed'],
        short_precision=diag['short_prediction_precision'],
        pred_over_ref=diag['predicted_over_reference_seconds'], seconds=round(item['eval_seconds'], 1))
    return item


def exact_test_rows(rows):
    """Durations from the audio headers (the manifest is sample-exact; this only guards it)."""
    import soundfile
    out, corrections = [], 0
    for row in rows:
        frames = soundfile.info(row['audio_path']).frames
        if frames != round(row['duration'] * 16000):
            corrections += 1
            row = inference_row(row['clip_id'], row['audio_path'], frames / 16000.)
        out.append(row)
    return out, corrections


def predict_test(model, api, rows, update, arm_dir):
    started = time.monotonic()
    rows, corrections = exact_test_rows(rows)
    records, frames = predict_outputs(model, api, rows, keep_frames=True)
    clips = sorted(row['clip_id'] for row in rows)
    files = {}
    for name, decoder in DECODERS.items():
        predicted = decode_all(api, records, decoder)
        path = arm_dir / f'test-pred-{update:06d}-{name}.jsonl'
        write_predictions(path, clips, predicted)
        files[name] = path.name
    save_frames(arm_dir / f'test-frames-{update:06d}.npz', frames, rows)
    item = {'update': update, 'clips': len(rows), 'predictions': files, 'frames': f'test-frames-{update:06d}.npz',
            'duration_corrections': corrections, 'seconds': time.monotonic() - started, 'utc': utc()}
    log('test_predictions', **item)
    return item


# ------------------------------------------------------------------- R2 + Bronze losses

def span_frames(low, high, frames):
    """Inclusive frame range whose centres fall near [low, high] (P2-silver-lb / K1 convention)."""
    first = max(0, int(math.floor((low - FRAME_OFFSET) / FRAME_HOP)))
    last = min(frames - 1, int(math.ceil((high - FRAME_OFFSET) / FRAME_HOP)))
    return first, last


def r2_tensors(rows, frames_max, api, force_full=False):
    """Per-channel masks of the R2 Silver objective and the full-event-only targets
    (K1 r2_tensors on the ATST grid, plus the Bronze ignore channels)."""
    import torch
    batch_size, classes = len(rows), len(CATEGORIES)
    presence_keep = torch.zeros(batch_size, frames_max, classes)
    bi_keep = torch.zeros(batch_size, frames_max, classes)
    onset_keep = torch.ones(batch_size, frames_max, classes)
    full_items = []
    band = R2['band_seconds']
    report = Counter()
    for b, row in enumerate(rows):
        frames = row['num_frames']
        presence_keep[b, :frames] = 1.
        modes = row['r2']
        if force_full:
            modes = [dict(m, mode='full', onset_mask=False) for m in modes]
        has_full = any(m['mode'] == 'full' for m in modes)
        if has_full:
            bi_keep[b, :frames] = 1.
        for event, m in zip(row['events'], modes):
            c = CATEGORIES.index(event['category'])
            report['events_' + m['mode']] += 1
            if m['mode'] == 'presence':
                for endpoint in (event['onset'], event['offset']):
                    first, last = span_frames(endpoint - band, endpoint + band, frames)
                    presence_keep[b, first:last + 1, c] = 0.
            elif m['mode'] == 'ignore':
                first, last = span_frames(event['onset'] - band, event['offset'] + band, frames)
                presence_keep[b, first:last + 1, c] = 0.
            if m['mode'] != 'full' and has_full:
                first, last = span_frames(event['onset'] - band, event['offset'] + band, frames)
                bi_keep[b, first:last + 1, c] = 0.
            if m['mode'] == 'full' and m['onset_mask']:
                first, last = span_frames(event['onset'] - band, event['onset'] + band, frames)
                onset_keep[b, first:last + 1, c] = 0.
                report['onset_masked_events'] += 1
        for category in row.get('ignore_channels') or []:
            c = CATEGORIES.index(category)
            presence_keep[b, :, c] = 0.
            bi_keep[b, :, c] = 0.
            report['ignore_channel_cells'] += frames
        if row.get('quality') == 'bronze_wrap_all':
            report['bronze_wrap_all_clips'] += 1
        full_events = [e for e, m in zip(row['events'], modes) if m['mode'] == 'full']
        full_items.append(api.interval_targets(dict(row, events=full_events)))
    full = {name: torch.nn.utils.rnn.pad_sequence([item[name] for item in full_items], batch_first=True)
            for name in ('distances', 'event_weights', 'onsets', 'offsets')}
    for name in full:
        require(full[name].shape[1] == frames_max, 'R2 target length mismatch')
    report['presence_cells_kept'] = int(presence_keep.sum())
    report['presence_cells_real'] = int(sum(r['num_frames'] for r in rows) * classes)
    report['bi_cells_kept'] = int(bi_keep.sum())
    report['clips_with_full'] = int(sum(any(m['mode'] == 'full' for m in r['r2']) or force_full for r in rows))
    report['clips'] = batch_size
    return {'presence_keep': presence_keep, 'bi_keep': bi_keep, 'onset_keep': onset_keep,
            'full': full, 'report': dict(report)}


def masked_presence(output, targets, keep, class_weights, device):
    """interval_loss's RED presence term with a per-channel keep mask [B, T, 7]."""
    import torch
    target = targets.to(device)
    keep = keep.to(device)
    probability = output['presence'].clamp(1e-6, 1 - 1e-6)
    presence = -(class_weights * target * probability.log() + (1 - target) * torch.log1p(-probability))
    return (presence * keep).sum() / keep.sum().clamp_min(1)


def boundary_iou_terms(output, full, bi_keep, onset_keep, device):
    """interval_loss's boundary (focal) and IoU terms on the full-event targets, per-channel masks."""
    import torch
    keep = bi_keep.to(device)
    onset = keep * onset_keep.to(device)
    cells = keep.sum().clamp_min(1)
    boundary = output['presence'].new_zeros(())
    for name, mask in (('onsets', onset), ('offsets', keep)):
        probability = output[name].clamp(1e-6, 1 - 1e-6)
        label = full[name].to(device)
        focal = -(label * (1 - probability).square() * probability.log()
                  + (1 - label) * probability.square() * torch.log1p(-probability))
        boundary = boundary + (focal * mask).sum() / cells
    target_distance = full['distances'].to(device)
    event_weight = full['event_weights'].to(device) * keep
    intersection = torch.minimum(output['distances'], target_distance).sum(dim=-1)
    union = torch.maximum(output['distances'], target_distance).sum(dim=-1).clamp_min(1e-6)
    iou = ((1 - intersection / union) * event_weight).sum() / event_weight.sum().clamp_min(1)
    return boundary, iou


def r2_silver_loss(output, batch, r2, class_weights, device):
    settings = ORIGINAL['strong_loss']
    presence = masked_presence(output, batch['targets'], r2['presence_keep'], class_weights, device)
    loss = presence
    parts = {'presence': float(presence.detach()), 'boundary': 0., 'iou': 0.}
    if float(r2['bi_keep'].sum()) > 0:
        boundary, iou = boundary_iou_terms(output, r2['full'], r2['bi_keep'], r2['onset_keep'], device)
        loss = loss + settings['boundary_weight'] * boundary + settings['iou_weight'] * iou
        parts.update(boundary=float(boundary.detach()), iou=float(iou.detach()))
    return loss, parts


def bronze_absent_mask(rows, device):
    import torch
    mask = torch.zeros((len(rows), len(CATEGORIES)), dtype=torch.bool)
    for index, row in enumerate(rows):
        for category in row.get('bronze_absent') or []:
            mask[index, CATEGORIES.index(category)] = True
    return mask.to(device)


def bronze_negative_loss(presence, lengths, absent):
    """Mean over Bronze clips of mean over absent channels of -log(1 - max_t p_c) (real frames)."""
    import torch
    batch, frames, _ = presence.shape
    lengths = lengths.to(presence.device)
    per_clip = absent.sum(dim=1)
    supervised = per_clip > 0
    stats = {'bronze_clips': int(supervised.sum()), 'absent_channels': int(absent.sum())}
    if not bool(supervised.any()):
        return presence.reshape(-1)[:0].sum(), stats
    valid = torch.arange(frames, device=presence.device)[None, :, None] < lengths[:, None, None]
    maxima = presence.masked_fill(~valid, 0.).amax(dim=1)                       # [B, 7]
    terms = -torch.log1p(-maxima.clamp(0., 1 - 1e-6))
    sums = (terms * absent).sum(dim=1)
    loss = (sums[supervised] / per_clip[supervised]).mean()
    stats['mean_absent_max_probability'] = float((maxima.detach() * absent).sum() / absent.sum().clamp_min(1))
    return loss, stats


# ---------------------------------------------------------------------------- update

def three_stream_update(model, strong_rows, silver_rows, weak_rows, optimizer, scaler, warp, api,
                        *, amp, class_weights, silver_mode, bronze):
    """P2-silver's three_stream_update (P2's mixed_update + a Silver pass).  The Silver pass uses the
    arm's Silver mode: 'full' = interval_loss on every event (the P2-silver objective), 'R2' = the K1
    masks (K-A); the Bronze ignore channels are masked in both modes.  Only an arm with Bronze adds
    the Bronze absent-category negative to the weak pass; without Bronze the weak pass is P2-silver's."""
    import torch
    from indoml.weak_supervision import positive_category_mask, weak_positive_loss
    assert silver_mode in SILVER_MODES, f'Unknown Silver mode {silver_mode!r}'
    assert all(r['supervision'] == 'strong_gold' and r['events'] for r in strong_rows)
    assert all(r['supervision'] == 'strong_gold' and r['events'] and 'r2' in r for r in silver_rows)
    assert all(r['supervision'] == 'weak_clip' and r['events'] is None
               and r['positive_categories'] for r in weak_rows)
    assert bronze or not any(r['quality'] == 'bronze_wrap_all' for r in silver_rows), 'Bronze row, Bronze off'
    assert bronze or not any(r.get('bronze_absent') for r in weak_rows), 'Bronze negative list, Bronze off'
    assert getattr(model, '_full_release_fp32_head', False), 'Enforce FP32 head math first'
    device = model.head_device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    windows = {}
    settings = ORIGINAL['strong_loss']

    batch = api.collate_audio_intervals(strong_rows)
    windows['strong'] = len(batch['windows'])
    model.frequency_augmentation = warp
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        output = model(batch['windows'], batch['descriptions'])
    output = {key: value.float() for key, value in output.items()}
    loss, _ = api.interval_loss(output, batch, class_weights, device, 'red', settings['boundary_weight'],
                                settings['iou_weight'], collar_weight=settings['collar_weight'])
    assert torch.isfinite(loss).item(), 'Nonfinite strong loss'
    scaler.scale(STRONG_WEIGHT['gold'] * loss).backward()
    strong_value = float(loss.detach())
    del loss, output, batch

    silver_value, silver_parts, silver_report = None, None, None
    if silver_rows:
        batch = api.collate_audio_intervals(silver_rows)
        windows['silver'] = len(batch['windows'])
        r2 = r2_tensors(silver_rows, int(batch['targets'].shape[1]), api, force_full=silver_mode == 'full')
        model.frequency_augmentation = warp
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            output = model(batch['windows'], batch['descriptions'])
        output = {key: value.float() for key, value in output.items()}
        loss, silver_parts = r2_silver_loss(output, batch, r2, class_weights, device)
        assert torch.isfinite(loss).item(), 'Nonfinite silver loss'
        scaler.scale(STRONG_WEIGHT['silver'] * loss).backward()
        silver_value = float(loss.detach())
        silver_report = r2['report']
        del loss, output, batch, r2

    weak_value, bronze_value, bronze_stats = None, None, None
    if weak_rows:
        model.frequency_augmentation = None
        batch = api.collate_clips(weak_rows)
        lengths = torch.tensor([item['frames'] for item in batch['descriptions']], dtype=torch.long)
        mask = positive_category_mask([r['positive_categories'] for r in weak_rows], device=device)
        windows['weak'] = len(batch['windows'])
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            presence = model(batch['windows'], batch['descriptions'])['presence']
        presence = presence.float()
        weak_loss, _ = weak_positive_loss(presence, lengths, mask)
        assert torch.isfinite(weak_loss).item(), 'Nonfinite weak loss'
        if bronze:
            absent = bronze_absent_mask(weak_rows, device)
            negative, bronze_stats = bronze_negative_loss(presence, lengths, absent)
            assert torch.isfinite(negative).item(), 'Nonfinite Bronze negative loss'
            scaler.scale(WEAK_WEIGHT * weak_loss + BRONZE_NEGATIVE_WEIGHT * negative).backward()
            bronze_value = float(negative.detach())
            del negative, absent
        else:                      # P2-silver's weak pass: the positive-only term, nothing else
            scaler.scale(WEAK_WEIGHT * weak_loss).backward()
        weak_value = float(weak_loss.detach())
        del presence, weak_loss, batch, mask, lengths
    before_scale = scaler.get_scale()
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norms = [torch.linalg.vector_norm(p.grad.detach(), 2) for p in parameters if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack(norms), 2)
    finite = bool(torch.isfinite(norm).item())
    if finite:
        torch.nn.utils.clip_grad_norm_(parameters, ORIGINAL['optimizer']['gradient_norm_clip'],
                                       error_if_nonfinite=True)
    else:
        assert amp, 'FP32 nonfinite gradients'
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < before_scale
    assert finite or skipped, 'Nonfinite gradients were not safely skipped'
    model.frequency_augmentation = warp
    return {'strong_loss': strong_value, 'silver_loss': silver_value, 'silver_parts': silver_parts,
            'silver_report': silver_report, 'weak_loss': weak_value, 'bronze_negative': bronze_value,
            'bronze_stats': bronze_stats, 'strong_clips': len(strong_rows), 'silver_clips': len(silver_rows),
            'weak_clips': len(weak_rows), 'windows': windows,
            'raw_gradient_norm': float(norm.detach()) if finite else None,
            'skipped_update': skipped, 'scale': scaler.get_scale()}


# -------------------------------------------------------------------- startup checks

def startup_checks(model, api, data, device, amp, class_weights, spec):
    """On 4 real Silver clips: the R2 identity (every event 'full', no masks == interval_loss), the
    arm's own Silver objective (Silver mode 'full' must equal interval_loss; 'R2' the real R2 value)
    and the real R2 masks finite.  Bronze on: the Bronze wrap-all ignore channels masked (the same
    tensors in both Silver modes) and the Bronze negative finite with a nonzero head gradient (cuDNN
    off for that one backward).  Bronze off: no Bronze row and no absent-category list in the arm's
    pools.  Evaluation mode (no dropout, no warp): no RNG stream is consumed."""
    import torch
    settings = ORIGINAL['strong_loss']
    rows = data['check_rows']
    force_full = spec['silver_mode'] == 'full'
    result = {'silver_mode': spec['silver_mode'], 'bronze': spec['bronze']}
    previous = model.frequency_augmentation
    model.frequency_augmentation = None
    model.eval()
    with torch.no_grad():
        silver = [dict(r, ignore_channels=[]) for r in rows['silver']]
        batch = api.collate_audio_intervals(silver)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            output = model(batch['windows'], batch['descriptions'])
        output = {key: value.float() for key, value in output.items()}
        reference, parts = api.interval_loss(output, batch, class_weights, device, 'red',
                                             settings['boundary_weight'], settings['iou_weight'], collar_weight=0.)
        frames_max = int(batch['targets'].shape[1])
        all_full = [dict(r, r2=[dict(m, mode='full', onset_mask=False) for m in r['r2']]) for r in silver]
        mine, mine_parts = r2_silver_loss(output, batch, r2_tensors(all_full, frames_max, api), class_weights, device)
        forced, _ = r2_silver_loss(output, batch, r2_tensors(silver, frames_max, api, force_full=True),
                                   class_weights, device)
        real_r2 = r2_tensors(silver, frames_max, api)
        real, real_parts = r2_silver_loss(output, batch, real_r2, class_weights, device)
        ref_value, mine_value = float(reference), float(mine)
        tolerance = 1e-5 * max(1., abs(ref_value))
        result['r2_identity'] = {
            'clips': [r['clip_id'] for r in silver], 'modes': [[m['mode'] for m in r['r2']] for r in silver],
            'interval_loss': ref_value, 'interval_parts': parts, 'r2_all_full': mine_value,
            'r2_all_full_parts': mine_parts, 'r2_force_full_flag': float(forced),
            'difference': abs(mine_value - ref_value), 'tolerance': tolerance,
            'pass': abs(mine_value - ref_value) <= tolerance and abs(float(forced) - ref_value) <= tolerance,
            'r2_real': float(real), 'r2_real_parts': real_parts, 'r2_real_finite': math.isfinite(float(real)),
            'r2_real_report': real_r2['report']}
        own, own_parts = r2_silver_loss(output, batch, r2_tensors(silver, frames_max, api, force_full=force_full),
                                        class_weights, device)
        expected = ref_value if force_full else float(real)
        result['arm_objective'] = {
            'silver_mode': spec['silver_mode'], 'value': float(own), 'parts': own_parts, 'expected': expected,
            'expected_is': 'interval_loss (the P2-silver Silver objective)' if force_full else 'the real R2 loss',
            'difference': abs(float(own) - expected), 'tolerance': tolerance,
            'pass': math.isfinite(float(own)) and abs(float(own) - expected) <= tolerance}
        del batch, output
        bronze = rows['bronze_strong']
        if spec['bronze']:
            require(bronze and rows['weak_bronze'], f"Arm {spec['name']} has Bronze on but no Bronze check rows")
            batch = api.collate_audio_intervals(bronze)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                output = model(batch['windows'], batch['descriptions'])
            output = {key: value.float() for key, value in output.items()}
            tensors = r2_tensors(bronze, int(batch['targets'].shape[1]), api, force_full=force_full)
            other = r2_tensors(bronze, int(batch['targets'].shape[1]), api, force_full=not force_full)
            loss, loss_parts = r2_silver_loss(output, batch, tensors, class_weights, device)
            other_loss, _ = r2_silver_loss(output, batch, other, class_weights, device)
            expected_masked = sum(r['num_frames'] * len(r['ignore_channels']) for r in bronze)
            result['bronze_strong'] = {
                'clips': [r['clip_id'] for r in bronze], 'ignore_channels': [r['ignore_channels'] for r in bronze],
                'loss': float(loss), 'parts': loss_parts, 'report': tensors['report'],
                'ignore_cells_expected': expected_masked, 'loss_other_silver_mode': float(other_loss),
                'pass': (math.isfinite(float(loss)) and tensors['report'].get('ignore_channel_cells') == expected_masked
                         and float(other_loss) == float(loss))}
            del batch, output
    if spec['bronze']:
        weak = rows['weak_bronze']
        model.zero_grad(set_to_none=True)
        batch = api.collate_clips(weak)
        lengths = torch.tensor([item['frames'] for item in batch['descriptions']], dtype=torch.long)
        absent = bronze_absent_mask(weak, device)
        # Evaluation mode keeps dropout and the RNG streams untouched; cuDNN refuses an RNN backward in
        # evaluation mode, so this one check runs the GRUs through the native (non-cuDNN) kernels.
        with torch.backends.cudnn.flags(enabled=False):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                presence = model(batch['windows'], batch['descriptions'])['presence']
            negative, stats = bronze_negative_loss(presence.float(), lengths, absent)
            negative.backward()
        head_grad = math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in model.head.parameters()
                                  if p.grad is not None))
        encoder_grad = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                                     for p in model.encoder.parameters() if p.grad is not None))
        result['bronze_negative'] = {
            'clips': [r['clip_id'] for r in weak], 'absent': [r['bronze_absent'] for r in weak],
            'value': float(negative.detach()), 'stats': stats, 'head_gradient_norm': head_grad,
            'encoder_gradient_norm': encoder_grad,
            'pass': math.isfinite(float(negative.detach())) and head_grad > 0 and math.isfinite(head_grad)}
        model.zero_grad(set_to_none=True)
        del batch, presence, negative
        bronze_pass = result['bronze_strong']['pass'] and result['bronze_negative']['pass']
    else:
        pools = data['pools']
        off = {'silver_bronze_rows': sum(1 for r in pools['silver'] if r['quality'] == 'bronze_wrap_all'),
               'silver_ignore_channel_rows': sum(1 for r in pools['silver'] if r.get('ignore_channels')),
               'weak_rows_with_bronze_absent': sum(1 for r in pools['weak'] if r.get('bronze_absent')),
               'check_rows_bronze_strong': len(rows['bronze_strong']),
               'check_rows_weak_bronze': len(rows['weak_bronze'])}
        result['bronze_off'] = dict(off, negative_term='not computed (three_stream_update bronze=False)',
                                    **{'pass': all(value == 0 for value in off.values())})
        bronze_pass = result['bronze_off']['pass']
    model.frequency_augmentation = previous
    result['pass'] = bool(result['r2_identity']['pass'] and result['r2_identity']['r2_real_finite']
                          and result['arm_objective']['pass'] and bronze_pass)
    return result


# ------------------------------------------------------------------------------ arm

def stream_states(streams):
    return {name: {'order': stream.order, 'cursor': stream.cursor,
                   'completed_cycles': stream.completed_cycles,
                   'generator_state': stream.generator.get_state().clone(),
                   'draws': dict(stream.draws), 'seen': sorted(stream.seen)}
            for name, stream in streams.items()}


def restore_streams(streams, saved):
    for name, stream in streams.items():
        state = saved[name]
        stream.order = list(state['order'])
        stream.cursor = state['cursor']
        stream.completed_cycles = state['completed_cycles']
        stream.generator.set_state(state['generator_state'])
        stream.draws = Counter(state['draws'])
        stream.seen = set(state['seen'])


class Arm:
    """One arm in its own process: model, optimizer, scaler, warp, streams, trajectory."""

    def __init__(self, name, data, mode):
        import torch
        from indoml import atst_intervals as api
        from mixed_training import WindowStream
        self.torch, self.api = torch, api
        self.name, self.data, self.mode = name, data, mode
        self.spec = check_spec(dict(ARM_SPECS[name]))
        self.silver_mode, self.bronze = self.spec['silver_mode'], self.spec['bronze']
        require(data.get('arm_census', {}).get('arm') == name, f'Pools are not the arm view of {name}')
        self.dir = WORKING / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.amp = self.device.startswith('cuda')
        self.mix = PROBE_MIX if mode == 'probe' else MIXES[GOLD_MIX]
        self.streams = {n: WindowStream(data['pools'][n], c, CAPS[n], SEEDS[n])
                        for n, c in self.mix.items() if c > 0}
        self.model, self.init_meta = load_parent_model(self.device)
        self.optimizer, self.layout = depth_optimizer(self.model, api, ORIGINAL['optimizer']['weight_decay'],
                                                      ORIGINAL['construction_lr']['encoder_lr'],
                                                      ORIGINAL['construction_lr']['head_lr'])
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1024., growth_interval=2000, enabled=self.amp)
        fw = ORIGINAL['frequency_warp']
        self.warp = api.FrequencyWarpStream(fw['probability'], fw['min_scale'], fw['max_scale'], fw['seed'])
        self.class_weights = torch.full((len(CATEGORIES),), ORIGINAL['strong_loss']['class_positive_weight'],
                                        dtype=torch.float32, device=self.model.head_device)
        self.completed = 0
        self.total = T_MAX[CONFIG] if mode == 'full' else (SMOKE_UPDATES if mode == 'smoke' else PROBE_UPDATES)
        self.plateau = (plateau_end(self.total) if mode == 'full' else
                        (SMOKE_PLATEAU if mode == 'smoke' else PROBE_PLATEAU))
        self.plan_version = None
        self.trajectory, self.checks, self.u0_check = [], {}, None
        self.collisions = 0
        self.accum = Counter()
        self.report = Counter()
        self.history = []            # [completed, step_seconds_sum, train_wall_sum, epoch time]
        self.step_sum = self.train_wall_sum = 0.
        self.step_times = []
        self.status = 'starting'
        self.test_item = None        # the test predictions at the end (T, or the stop point)
        self.tests = []              # every test prediction item (D and T)
        self.events = []
        self.resumed_from = None

    # ----------------------------------------------------------------- state
    def state_payload(self):
        torch = self.torch
        return {'kind': 'indoml_atst_scale_state_v1', 'run': RUN_TAG, 'arm': self.name, 'spec': self.spec,
                'updates': self.completed, 'total': self.total, 'plateau': self.plateau,
                'plan_version': self.plan_version,
                'model_state': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                'optimizer_state': self.optimizer.state_dict(), 'scaler_state': self.scaler.state_dict(),
                'warp_state': self.warp.state_dict(), 'streams': stream_states(self.streams),
                'rng': {'cpu': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else []},
                'trajectory': self.trajectory, 'weak_collisions': self.collisions,
                'accum': dict(self.accum), 'report': dict(self.report), 'history': self.history,
                'step_sum': self.step_sum, 'train_wall_sum': self.train_wall_sum,
                'mix': self.mix, 'caps': CAPS, 'seeds': SEEDS, 'r2': R2, 'silver_mode': self.silver_mode,
                'bronze': self.bronze, 'tests': self.tests, 'events': self.events,
                'pool_digest': self.data['pool_digest'], 'utc': utc()}

    def save_state(self, name='last.pt'):
        started = time.monotonic()
        temporary = self.dir / (name + '.tmp')
        self.torch.save(self.state_payload(), temporary)
        temporary.replace(self.dir / name)
        return time.monotonic() - started

    def load_state(self, path, model=None, optimizer=None, scaler=None, warp=None, streams=None):
        torch = self.torch
        saved = torch.load(path, map_location='cpu', weights_only=False)
        require(saved['kind'] == 'indoml_atst_scale_state_v1' and saved['arm'] == self.name, 'Not this arm')
        require(saved['pool_digest'] == self.data['pool_digest'], 'Resume pools differ')
        require(saved['mix'] == self.mix and saved['seeds'] == SEEDS and saved['r2'] == R2, 'Resume contract differs')
        require(saved.get('silver_mode') == self.silver_mode and saved.get('bronze') == self.bronze
                and spec_core(saved['spec']) == spec_core(self.spec), 'Resume arm spec differs')
        model = model or self.model
        model.load_state_dict(saved['model_state'], strict=True)
        model.set_device(torch.device(self.device), torch.device(self.device))
        (optimizer or self.optimizer).load_state_dict(saved['optimizer_state'])
        (scaler or self.scaler).load_state_dict(saved['scaler_state'])
        (warp or self.warp).load_state_dict(saved['warp_state'])
        restore_streams(streams or self.streams, saved['streams'])
        torch.set_rng_state(saved['rng']['cpu'])
        if self.device.startswith('cuda') and saved['rng']['cuda']:
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        return saved

    def resume(self, path):
        saved = self.load_state(path)
        self.completed = saved['updates']
        self.total, self.plateau, self.plan_version = saved['total'], saved['plateau'], saved['plan_version']
        self.trajectory = saved['trajectory']
        self.collisions = saved['weak_collisions']
        self.accum, self.report = Counter(saved['accum']), Counter(saved['report'])
        self.history = saved['history']
        self.step_sum, self.train_wall_sum = saved['step_sum'], saved['train_wall_sum']
        self.tests = list(saved.get('tests') or [])
        self.events = list(saved.get('events') or [])
        self.resumed_from = {'updates': self.completed, 'saved_utc': saved['utc'], 'path': str(path)}
        log('resumed', updates=self.completed, total=self.total, plateau=self.plateau)

    def resume_equality(self):
        """Save last.pt, rebuild every object from scratch, load it and compare (resume check)."""
        torch, api = self.torch, self.api
        from mixed_training import WindowStream
        rng_saved = torch.get_rng_state().clone()
        cuda_saved = torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else []
        seconds = self.save_state()
        model, _ = load_parent_model(self.device)
        optimizer, _ = depth_optimizer(model, api, ORIGINAL['optimizer']['weight_decay'],
                                       ORIGINAL['construction_lr']['encoder_lr'],
                                       ORIGINAL['construction_lr']['head_lr'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024., growth_interval=2000, enabled=self.amp)
        fw = ORIGINAL['frequency_warp']
        warp = api.FrequencyWarpStream(fw['probability'], fw['min_scale'], fw['max_scale'], fw['seed'])
        streams = {n: WindowStream(self.data['pools'][n], c, CAPS[n], SEEDS[n]) for n, c in self.mix.items() if c > 0}
        saved = self.load_state(self.dir / 'last.pt', model, optimizer, scaler, warp, streams)
        mine, theirs = self.model.state_dict(), model.state_dict()
        weights_equal = all(torch.equal(mine[k].cpu(), theirs[k].cpu()) for k in mine)
        a, b = self.optimizer.state_dict(), optimizer.state_dict()
        optimizer_equal = (len(a['state']) == len(b['state']) and all(
            all(torch.equal(torch.as_tensor(a['state'][i][key]).cpu(), torch.as_tensor(b['state'][i][key]).cpu())
                for key in a['state'][i]) for i in a['state'])
            and [g['lr'] for g in a['param_groups']] == [g['lr'] for g in b['param_groups']])
        scaler_equal = self.scaler.state_dict() == scaler.state_dict()
        warp_equal = (torch.equal(self.warp.generator.get_state(), warp.generator.get_state())
                      and self.warp.counts == warp.counts)
        streams_equal = all(s.order == streams[n].order and s.cursor == streams[n].cursor
                            and torch.equal(s.generator.get_state(), streams[n].generator.get_state())
                            for n, s in self.streams.items())
        cuda_now = torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else []
        rng_equal = (torch.equal(rng_saved, torch.get_rng_state()) and len(cuda_now) == len(cuda_saved)
                     and all(torch.equal(a, b) for a, b in zip(cuda_now, cuda_saved)))
        result = {'updates': saved['updates'], 'save_seconds': seconds,
                  'bytes': (self.dir / 'last.pt').stat().st_size, 'weights_equal': weights_equal,
                  'optimizer_equal': optimizer_equal, 'scaler_equal': scaler_equal, 'warp_equal': warp_equal,
                  'streams_equal': streams_equal, 'rng_equal': rng_equal}
        result['pass'] = all(result[k] for k in ('weights_equal', 'optimizer_equal', 'scaler_equal',
                                                 'warp_equal', 'streams_equal', 'rng_equal'))
        log('resume_equality', **result)
        return result, (model, optimizer, scaler, warp, streams)

    # ----------------------------------------------------------------- files
    def progress(self, extra=None):
        item = {'arm': self.name, 'pid': os.getpid(), 'status': self.status, 'completed': self.completed,
                'total': self.total, 'plateau': self.plateau, 'plan_version': self.plan_version,
                'history': self.history[-400:], 'utc': utc(), 'epoch': time.time(),
                'u0_eval_seconds': self.checks.get('u0_eval_seconds'),
                'train_started_elapsed': self.checks.get('train_started_elapsed'),
                'last_eval_seconds': self.trajectory[-1]['eval_seconds'] if self.trajectory else None}
        if extra:
            item.update(extra)
        write_json(self.dir / 'progress.json', item)

    def write_results(self, error=None):
        mean_step = (sum(self.step_times) / len(self.step_times)) if self.step_times else None
        item = {'arm': self.name, 'spec': self.spec, 'silver_mode': self.silver_mode, 'bronze': self.bronze,
                'arm_census': self.data.get('arm_census'), 'status': self.status, 'mode': self.mode,
                'device': self.device, 'mix': self.mix, 'completed': self.completed, 'total': self.total,
                'plateau': self.plateau, 'plan_version': self.plan_version, 'init': self.init_meta,
                'checks': self.checks, 'u0_check': self.u0_check, 'checkpoints': self.trajectory,
                'weak_collisions': self.collisions, 'stream_draws': {k: dict(s.draws) for k, s in self.streams.items()},
                'stream_cycles': {k: s.completed_cycles for k, s in self.streams.items()},
                'warp': self.warp.summary(), 'test': self.test_item, 'tests': self.tests, 'events': self.events,
                'resumed_from': self.resumed_from,
                'seconds_per_update_this_session': {'mean_step': mean_step, 'updates': len(self.step_times)},
                'history': self.history, 'utc': utc()}
        if error:
            item['error'] = error
        write_json(self.dir / 'arm-results.json', item)

    # ----------------------------------------------------------------- plan
    def refresh_plan(self):
        if self.mode != 'full':
            return
        plan = read_json(WORKING / 'plan.json')
        if not plan or plan.get('version') == self.plan_version:
            return
        total, plateau = int(plan['T']), int(plan['D'])
        require(total >= self.completed, f'Plan T {total} is behind update {self.completed}')
        if self.plan_version is not None and self.completed > self.plateau:
            log('plan_ignored_after_plateau', version=plan.get('version'))
            self.plan_version = plan.get('version')
            return
        self.total, self.plateau, self.plan_version = total, plateau, plan.get('version')
        self.events.append({'plan': plan.get('version'), 'T': total, 'D': plateau, 'at': self.completed})
        log('plan_adopted', version=plan.get('version'), T=total, D=plateau, completed=self.completed)

    def wait_for_plan(self):
        if self.mode != 'full' or self.plan_version is not None:
            return
        started = time.monotonic()
        while self.plan_version is None and time.monotonic() - started < 20 * 60:
            if (WORKING / name_stop(self.name)).exists():
                return
            self.refresh_plan()
            if self.plan_version is None:
                time.sleep(5)
        if self.plan_version is None:
            rate = self.train_wall_sum / max(1, self.completed)
            plan = compute_plan(rate, self.checks.get('u0_eval_seconds') or 90., self.completed,
                                session_elapsed(), T_MAX[CONFIG], f'self-computed by {self.name} (no plan.json)')
            self.total, self.plateau, self.plan_version = plan['T'], plan['D'], 'self'
            self.events.append({'plan': 'self', 'T': plan['T'], 'D': plan['D'], 'at': self.completed})
            log('plan_self', **plan)

    # ----------------------------------------------------------------- evaluation
    def eval_rows(self):
        rows = self.data['eval_rows']
        if self.mode == 'probe':
            return rows[:PROBE_EVAL_CLIPS]
        if self.mode == 'smoke' and self.completed > 0:
            return rows[:SMOKE_EVAL_CLIPS]
        return rows

    def test_rows(self):
        rows = self.data['test_rows']
        if self.mode == 'probe':
            return rows[:PROBE_TEST_CLIPS]
        if self.mode == 'smoke':
            return rows[:SMOKE_TEST_CLIPS]
        return rows

    def run_test_predictions(self, note):
        item = predict_test(self.model, self.api, self.test_rows(), self.completed, self.dir)
        item['note'] = note
        self.tests.append(item)
        return item

    def run_evaluation(self, note=None):
        torch = self.torch
        item = evaluate(self.model, self.api, self.data, self.eval_rows(), self.completed, self.dir)
        if note:
            item['note'] = note
        count = max(1, self.accum.get('count', 0))
        losses = {}
        for key in ('strong_loss', 'silver_loss', 'weak_loss', 'bronze_negative', 'raw_gradient_norm'):
            if self.accum.get(key + '_n'):
                losses[key] = self.accum[key] / self.accum[key + '_n']
        for key in ('silver_presence', 'silver_boundary', 'silver_iou'):
            if key in self.accum:
                losses[key] = self.accum[key] / max(1, self.accum.get('silver_loss_n', 0))
        losses['skipped_updates'] = self.accum.get('skipped', 0)
        item['train_losses'] = losses
        item['train_updates_in_period'] = self.accum.get('count', 0)
        item['r2_silver_report'] = dict(self.report)
        item['weak_collisions'] = self.collisions
        item['rates'] = arm_rates(self.spec, max(0, self.completed - 1), self.total, self.plateau) if self.completed else None
        item['plan'] = {'T': self.total, 'D': self.plateau, 'version': self.plan_version}
        item['scale'] = self.scaler.get_scale() if self.amp else None
        item['gpu_gb'] = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                          if self.device.startswith('cuda') else None)
        self.accum, self.report = Counter(), Counter()
        saves = {}
        if self.mode == 'full' and self.completed in weight_points(self.total, self.plateau):
            started = time.monotonic()
            payload = {'kind': 'indoml_atst_scale_checkpoint_v1', 'arm': self.name, 'updates': self.completed,
                       'dtype': 'float16', 'spec': self.spec, 'config': self.spec, 'decoders': DECODERS,
                       'metrics': {'natural1806_combined_t065': item['selection_combined'],
                                   'natural1806_combined_t075':
                                       item['decoders']['t075']['natural1806']['combined']},
                       'model_state': {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
                                       for k, v in self.model.state_dict().items()},
                       'architecture': 'NativeIntervalATST as parent-p2.pt (head_config of parent-native-e5.pt, '
                                       'train_blocks 12); cast to float32 and load_state_dict'}
            torch.save(payload, self.dir / f'ckpt-{self.completed:06d}.pt')
            saves['weights'] = f'ckpt-{self.completed:06d}.pt'
            saves['weights_seconds'] = time.monotonic() - started
        self.trajectory.append(item)
        if self.completed > 0:
            saves['last_seconds'] = self.save_state()
        if self.completed == self.plateau and self.completed < self.total:
            shutil.copyfile(self.dir / 'last.pt', self.dir / 'plateau-last.pt')
            saves['plateau_last'] = 'plateau-last.pt'
            log('plateau_saved', update=self.completed)
        item['saves'] = saves
        self.write_results()
        self.progress()
        return item

    # ----------------------------------------------------------------- one update
    def step(self):
        rates = apply_rates(self.optimizer, self.spec, self.completed, self.total, self.plateau)
        strong = []
        for name in ('gold', 'generated'):
            if name in self.streams:
                strong.extend(self.streams[name].take())
        silver = list(self.streams['silver'].take()) if 'silver' in self.streams else []
        drawn = {row['clip_id'] for row in strong + silver}
        weak = []
        if 'weak' in self.streams:
            for row in self.streams['weak'].take():
                if row['clip_id'] in drawn:          # a clip already drawn as strong
                    self.collisions += 1
                    continue
                weak.append(row)
        started = time.monotonic()
        stats = three_stream_update(self.model, strong, silver, weak, self.optimizer, self.scaler, self.warp,
                                    self.api, amp=self.amp, class_weights=self.class_weights,
                                    silver_mode=self.silver_mode, bronze=self.bronze)
        seconds = time.monotonic() - started
        self.completed += 1
        self.step_sum += seconds
        self.step_times.append(seconds)
        self.accum['count'] += 1
        for key in ('strong_loss', 'silver_loss', 'weak_loss', 'bronze_negative', 'raw_gradient_norm'):
            if stats.get(key) is not None:
                self.accum[key] += stats[key]
                self.accum[key + '_n'] += 1
        for key, value in (stats.get('silver_parts') or {}).items():
            self.accum['silver_' + key] += value
        self.accum['skipped'] += int(stats['skipped_update'])
        for key, value in (stats.get('silver_report') or {}).items():
            self.report[key] += value
        for key, value in (stats.get('bronze_stats') or {}).items():
            if key != 'mean_absent_max_probability':
                self.report['weak_' + key] += value
        self.report['silver_bronze_rows_drawn'] += sum(1 for r in silver if r['quality'] == 'bronze_wrap_all')
        return stats, rates, seconds


def name_stop(arm):
    return f'{arm}/STOP'


def run_child(name):
    import torch
    mode = os.environ.get('ATST_SCALE_MODE', 'probe')
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    setup_paths()
    with open(os.environ['ATST_SCALE_POOLS'], 'rb') as stream:
        data = pickle.load(stream)
    arm = None
    try:
        data = arm_view(data, ARM_SPECS[name])           # this arm's pools (Bronze on/off)
        require(data['pool_digest'] == (data.get('arm_digests') or {}).get(name),
                f'{name}: pool digest differs from the parent census')
        log('arm_pools', **data['arm_census'])
        gpu = None
        if mode != 'probe' and not os.environ.get('ATST_SCALE_FORCE_MODE'):
            require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                    f'Child {name} must see exactly one GPU (CUDA_VISIBLE_DEVICES='
                    f"{os.environ.get('CUDA_VISIBLE_DEVICES')})")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            props = torch.cuda.get_device_properties(0)
            gpu = {'name': props.name, 'total_gb': round(props.total_memory / 2 ** 30, 2),
                   'uuid': str(getattr(props, 'uuid', '')),
                   'pci_bus_id': getattr(props, 'pci_bus_id', None),
                   'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        log('child_start', mode=mode, gpu=gpu, torch=str(torch.__version__), threads=torch.get_num_threads())
        arm = Arm(name, data, mode)
        arm.checks['gpu'] = gpu
        arm.checks['load_seconds'] = round(session_elapsed(), 1)
        write_json(arm.dir / 'run-contract.json', {
            'kind': 'indoml_atst_scale_v1', 'run': RUN_TAG, 'arm': arm.spec, 'mode': mode, 'config': CONFIG,
            'gold_mix': GOLD_MIX, 'mix': arm.mix, 'caps': CAPS, 'seeds': SEEDS, 'silver_mode': arm.silver_mode,
            'bronze': arm.bronze, 'r2': R2, 'schedule': SCHEDULE, 'strong_weight': STRONG_WEIGHT,
            'weak_weight': WEAK_WEIGHT,
            'bronze_negative_weight': BRONZE_NEGATIVE_WEIGHT if arm.bronze else None,
            'original': ORIGINAL, 'decoders': DECODERS,
            'init': arm.init_meta, 'optimizer_groups': [g['name'] for g in arm.optimizer.param_groups],
            'optimizer_layout': arm.layout, 'pool_digest': data['pool_digest'], 'arm_census': data['arm_census'],
            'gpu': gpu, 'session_limit_hours': SESSION_LIMIT_HOURS, 'utc': utc()})
        resume_path = arm.dir / 'last.pt'
        if resume_path.is_file():
            arm.resume(resume_path)
        arm.status = 'checks'
        arm.progress()
        started = time.monotonic()
        arm.checks['startup'] = startup_checks(arm.model, arm.api, data, arm.model.head_device, arm.amp,
                                               arm.class_weights, arm.spec)
        arm.checks['startup_seconds'] = time.monotonic() - started
        startup = arm.checks['startup']
        log('startup_checks', passed=startup['pass'], silver_mode=arm.silver_mode, bronze=arm.bronze,
            r2_identity=startup['r2_identity']['difference'],
            interval_loss=startup['r2_identity']['interval_loss'],
            arm_objective=startup['arm_objective']['value'],
            arm_objective_difference=startup['arm_objective']['difference'],
            bronze_negative=(startup.get('bronze_negative') or {}).get('value'),
            bronze_head_grad=(startup.get('bronze_negative') or {}).get('head_gradient_norm'),
            bronze_off=startup.get('bronze_off'))
        arm.write_results()
        require(startup['pass'], 'Startup checks failed (R2 identity / arm objective / Bronze terms)')
        if arm.completed == 0:
            arm.status = 'u0'
            arm.progress()
            item = arm.run_evaluation(note='parent-p2.pt before any update')
            combined = item['selection_combined']
            delta = combined - PARENT_NATURAL['combined']
            full_population = item['clips'] == 1806
            arm.u0_check = {'combined_t065': combined, 'delta': delta, 'clips': item['clips'],
                            'tolerance': U0_TOLERANCE,
                            'status': ('pass' if abs(delta) <= U0_TOLERANCE else 'fail') if full_population
                            else 'subset_not_comparable'}
            arm.checks['u0_eval_seconds'] = item['eval_seconds']
            log('u0_check', **arm.u0_check)
            arm.write_results()
            require(arm.u0_check['status'] != 'fail', f'u0 {combined:.6f} is not P2 1.075846 +/- 2e-4')
            torch.manual_seed(SEEDS['torch'])
            if arm.device.startswith('cuda'):
                torch.cuda.manual_seed_all(SEEDS['torch'])
        train(arm)
    except BaseException as error:
        if arm is not None:
            arm.status = 'failed'
            try:
                arm.write_results(error=traceback.format_exc()[-6000:])
                arm.progress()
            except Exception:
                traceback.print_exc()
        else:
            try:
                write_json(WORKING / name / 'arm-results.json',
                           {'arm': name, 'status': 'failed', 'error': traceback.format_exc()[-6000:], 'utc': utc()})
            except Exception:
                traceback.print_exc()
        log('arm_failed', error=repr(error), traceback=traceback.format_exc()[-3000:])
        raise


def train(arm):
    torch = arm.torch
    mode = arm.mode
    arm.status = 'training'
    arm.checks['train_started_elapsed'] = session_elapsed()
    arm.progress({'train_started_elapsed': arm.checks['train_started_elapsed']})
    train_started = time.monotonic()
    last_mark = time.monotonic()
    stop_reason = None
    eval_every = EVAL_EVERY if mode == 'full' else (PROBE_EVAL_EVERY if mode == 'probe' else SMOKE_EVAL_EVERY)
    log_every = LOG_EVERY_FULL if mode == 'full' else (10 if mode == 'smoke' else 1)
    while arm.completed < arm.total:
        if (WORKING / name_stop(arm.name)).exists():
            stop_reason = (WORKING / name_stop(arm.name)).read_text().strip() or 'stop_file'
            break
        if session_elapsed() > self_stop_s():
            stop_reason = 'deadline'
            break
        if mode == 'full':
            if arm.plan_version is None and arm.completed >= PLAN_AT['full'] and arm.completed % PLAN_POLL[0] == 0:
                arm.refresh_plan()
            elif arm.completed % PLAN_POLL[1] == 0:
                arm.refresh_plan()
            if arm.completed >= PLAN_REQUIRED_BY and arm.plan_version is None:
                arm.wait_for_plan()
            if arm.completed >= arm.total:
                break
        tick = time.monotonic()
        stats, rates, seconds = arm.step()
        arm.train_wall_sum += time.monotonic() - tick
        if arm.completed % HISTORY_EVERY == 0 or mode != 'full':
            arm.history.append([arm.completed, round(arm.step_sum, 3), round(arm.train_wall_sum, 3),
                                round(time.time(), 1)])
        if arm.completed % log_every == 0 or arm.completed == 1:
            window = arm.step_times[-100:]
            log('update', completed=arm.completed, total=arm.total, plateau=arm.plateau,
                strong_loss=round(stats['strong_loss'], 5),
                silver_loss=None if stats['silver_loss'] is None else round(stats['silver_loss'], 5),
                weak_loss=None if stats['weak_loss'] is None else round(stats['weak_loss'], 5),
                bronze_negative=None if stats['bronze_negative'] is None else round(stats['bronze_negative'], 5),
                encoder_top_lr=rates['11'], head_lr=rates['head'], seconds=round(seconds, 3),
                mean_seconds=round(sum(window) / len(window), 3), windows=stats['windows'],
                gradient_norm=(None if stats['raw_gradient_norm'] is None else round(stats['raw_gradient_norm'], 3)),
                skipped=stats['skipped_update'], scale=stats['scale'], weak_collisions=arm.collisions,
                gpu_gb=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                        if arm.device.startswith('cuda') else None), rss_gb=rss_gb())
            arm.progress()
        if mode == 'probe' and arm.completed == PROBE_STOP_AT:
            result, objects = arm.resume_equality()
            arm.checks['resume_equality'] = result
            require(result['pass'], 'Resume did not restore the saved state')
            arm.model, arm.optimizer, arm.scaler, arm.warp, arm.streams = objects
            arm.class_weights = arm.class_weights.to(arm.model.head_device)
            arm.events.append({'probe_resume_swap': arm.completed})
        if arm.completed % eval_every == 0 or arm.completed == arm.total:
            item = arm.run_evaluation()
            kill = arm.spec.get('kill_below')
            if (mode == 'full' and kill is not None and arm.completed >= KILL_FROM
                    and item['selection_combined'] < kill):
                stop_reason = f'kill_rule_raw_below_{kill}'
                arm.events.append({'kill': arm.completed, 'combined': item['selection_combined']})
                break
            if arm.completed == arm.plateau and arm.completed < arm.total:
                left = self_stop_s() - session_elapsed()
                if left > D_TEST_ALLOWANCE_S + FINAL_WORK_S:
                    arm.run_test_predictions(f'test predictions at D = u{arm.completed}')
                else:
                    arm.events.append({'d_test_skipped': arm.completed, 'seconds_to_self_stop': left})
                    log('d_test_skipped', update=arm.completed, seconds_to_self_stop=round(left, 1))
                arm.write_results()
            if mode == 'full':
                arm.refresh_plan()
    arm.checks['train_seconds'] = time.monotonic() - train_started
    finish(arm, stop_reason)


def finish(arm, stop_reason):
    evaluated = {item['update'] for item in arm.trajectory}
    time_left = session_limit_s() - session_elapsed()
    if stop_reason:
        arm.status = 'stopped'
        arm.events.append({'stop': stop_reason, 'at': arm.completed, 'utc': utc()})
        log('self_stop', reason=stop_reason, update=arm.completed, minutes_left=round(time_left / 60, 1))
        if arm.completed not in evaluated and time_left > 12 * 60 and arm.completed > 0:
            arm.run_evaluation(note=f'final evaluation after {stop_reason}')
        else:
            arm.save_state()
    time_left = session_limit_s() - session_elapsed()
    if arm.mode == 'smoke':
        result, _ = arm.resume_equality()
        arm.checks['resume_equality'] = result
    want_test = arm.completed == arm.total or stop_reason == 'deadline'
    done = [item for item in arm.tests if item['update'] == arm.completed]
    if want_test and done:
        arm.test_item = done[-1]                 # a deadline stop right after the D test predictions
    elif want_test and arm.completed > 0 and time_left > 9 * 60:
        arm.test_item = arm.run_test_predictions(
            'test predictions at T' if arm.completed == arm.total else f'test predictions after {stop_reason}')
    arm.status = 'complete' if not stop_reason else 'stopped'
    arm.write_results()
    arm.progress()
    mean_step = (sum(arm.step_times) / len(arm.step_times)) if arm.step_times else None
    log('arm_done', status=arm.status, updates=arm.completed, mean_step_seconds=mean_step)


# ---------------------------------------------------------------------------- parent

def script_path():
    candidates = []
    main = sys.modules.get('__main__')
    if getattr(main, '__file__', None):
        candidates.append(Path(main.__file__))
    candidates.append(Path('/kaggle/src/script.py'))
    for path in candidates:
        try:
            if path.is_file() and KERNEL_MARKER in path.read_text():
                return path.resolve()
        except OSError:
            continue
    return None


def nvidia_smi():
    """GPU index, UUID, memory in use and utilisation of every GPU (None without nvidia-smi)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()] or None
    except Exception:
        return None


def gpu_count():
    import torch
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def adopt_resume():
    root = dataset_root(RESUME_DATASET, required=False)
    adopted = []
    if root is None:
        return adopted
    for arm in ARMS:
        for candidate in (root / arm / 'last.pt', root / f'{arm}-last.pt'):
            target = WORKING / arm / 'last.pt'
            if candidate.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, target)
                adopted.append(str(candidate))
                break
    if adopted:
        log('adopted_resume_states', files=adopted)
    return adopted


def mean_between(history, low, high, index):
    """Mean seconds/update between two completed counts from the cumulative history."""
    points = {int(item[0]): item for item in history}
    if low not in points or high not in points:
        before = [item for item in history if item[0] <= low]
        after = [item for item in history if item[0] >= high]
        if not before or not after:
            return None
        a, b = before[-1], after[0]
    else:
        a, b = points[low], points[high]
    if b[0] <= a[0]:
        return None
    return (b[index] - a[index]) / (b[0] - a[0])


def projection(rate_s, eval_s, setup_s):
    """Session hours a full run of each configuration needs at the measured speed."""
    setup_s = setup_s or 15 * 60.
    per_update = rate_s + (eval_s + 25.) / EVAL_EVERY
    out = {}
    for cfg, total in T_MAX.items():
        train_s = total * per_update
        needed = setup_s + train_s + D_TEST_ALLOWANCE_S + FINAL_WORK_S + SELF_STOP_MARGIN_S
        out[cfg] = {'T_max': total, 'per_update_s': per_update, 'setup_s': setup_s,
                    'train_hours': train_s / 3600, 'session_hours_needed': needed / 3600,
                    'timeout_hours_rule': math.ceil((needed / 3600 + 1.0) * 4) / 4}
    return out


def merge_results(results, arms):
    for name in arms:
        payload = read_json(WORKING / name / 'arm-results.json')
        if payload:
            results['arms'][name] = payload
    results['plan'] = read_json(WORKING / 'plan.json')
    results['updated_utc'] = utc()
    write_json(WORKING / 'results.json', results)


def parent_main():
    import torch
    torch.set_num_threads(THREADS)
    WORKING.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    count = gpu_count()
    mode = os.environ.get('ATST_SCALE_FORCE_MODE') or (RUN_MODE if count > 0 else 'probe')
    try:
        memory = {line.split(':')[0]: line.split(':')[1].strip()
                  for line in Path('/proc/meminfo').read_text().splitlines()[:3]}
    except OSError:
        memory = None
    specs = {name: check_spec(dict(ARM_SPECS[name])) for name in ARMS}
    log('runtime', torch=str(torch.__version__), cuda=torch.version.cuda, device_count=count, mode=mode,
        config=CONFIG, gold_mix=GOLD_MIX, arms=list(ARMS),
        arm_specs={name: spec_core(spec) for name, spec in specs.items()},
        session_limit_hours=SESSION_LIMIT_HOURS, cpu_count=os.cpu_count(), memory=memory,
        python=sys.version.split()[0])
    results = {'kind': 'indoml_atst_scale_results_v1', 'run': RUN_TAG, 'mode': mode, 'started_utc': utc(),
               'config': CONFIG, 't_max': T_MAX[CONFIG], 'gold_mix': GOLD_MIX, 'mix': MIXES[GOLD_MIX],
               'silver_modes': {name: spec['silver_mode'] for name, spec in specs.items()},
               'bronze': {name: spec['bronze'] for name, spec in specs.items()}, 'arms_requested': list(ARMS),
               'arm_specs': specs, 'device_count': count,
               'session_limit_hours': SESSION_LIMIT_HOURS, 'self_stop_hours': self_stop_s() / 3600,
               'train_budget_hours': train_budget_s() / 3600,
               'p2_silver_seconds_per_update': P2_SILVER_SECONDS_PER_UPDATE,
               'model_family': 'P2 (ATST-Frame + RED head) continued with the full Silver objective '
                               '(P2-silver) at the R10 encoder schedule, without / with Bronze',
               'evaluation_population': 'all 1,806 organizer natural validation clips',
               'parent_reference': {'natural1806': PARENT_NATURAL}, 'decoders': DECODERS,
               'session_filter': 'every training row whose 6th underscore field (session token) occurs in the '
                                 '1,806 natural clip ids is dropped, and every generated clip whose bed or donor '
                                 'piece comes from such a session', 'arms': {}}
    write_json(WORKING / 'results.json', results)
    setup_paths()
    adopted = adopt_resume()
    started = time.monotonic()
    data = build_pools(mode)
    results['census'] = data['census']
    results['pool_seconds'] = time.monotonic() - started
    results['adopted_resume_states'] = adopted
    log('pools', seconds=round(results['pool_seconds'], 1),
        **{k: v for k, v in data['census'].items() if k not in ('dropped',)})
    log('dropped', **data['census']['dropped'])
    results['arm_pools'] = {name: arm_view(data, spec)['arm_census'] for name, spec in specs.items()}
    data['arm_digests'] = {name: census['pool_digest'] for name, census in results['arm_pools'].items()}
    for name, census in results['arm_pools'].items():
        log('arm_pools', child=name, **census)
    write_json(WORKING / 'results.json', results)
    pools_path = SCRATCH / 'pools.pkl'
    with pools_path.open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    del data

    if mode == 'probe' or count == 0:
        arms = list(ARMS)
        devices = {name: None for name in arms}
    elif count >= 2:
        arms = list(ARMS)
        devices = {name: str(index) for index, name in enumerate(arms)}
    else:
        arms = [ARMS[0]]
        devices = {ARMS[0]: '0'}
        results['single_gpu'] = True
    results['arms_run'] = arms
    results['devices'] = devices
    script = script_path()
    require(script is not None, 'Cannot find this script file to start the arm processes')
    results['script'] = {'path': str(script), 'sha256': sha256_of(script)}
    for name in arms:
        stop = WORKING / name_stop(name)
        if stop.exists():
            stop.unlink()
    children = {}
    for name in arms:
        env = dict(os.environ)
        env.update(ATST_SCALE_CHILD=name, ATST_SCALE_MODE=mode, ATST_SCALE_POOLS=str(pools_path),
                   ATST_SCALE_T0=repr(session_t0()), OMP_NUM_THREADS=str(THREADS),
                   MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS), PYTHONUNBUFFERED='1')
        if devices[name] is None:
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            env['CUDA_VISIBLE_DEVICES'] = devices[name]
        children[name] = subprocess.Popen([sys.executable, str(script)], env=env)
        log('child_spawned', child=name, pid=children[name].pid, cuda_visible_devices=devices[name])
    results['nvidia_smi_at_spawn'] = nvidia_smi()
    write_json(WORKING / 'results.json', results)
    monitor(children, arms, mode, results)
    pools_path.unlink(missing_ok=True)
    merge_results(results, arms)
    if mode in ('smoke', 'probe'):
        results['verdict'] = verdict(results, arms, mode, count)
        log('verdict', **results['verdict'])
    results['finished_utc'] = utc()
    results['elapsed_hours'] = session_elapsed() / 3600
    write_json(WORKING / 'results.json', results)
    log('done', elapsed_hours=results['elapsed_hours'],
        exit_codes={name: child.returncode for name, child in children.items()})


def monitor(children, arms, mode, results):
    plan_version = 0
    guard_done = refit_pending = refit_done = False
    stop_sent = False
    last_merge = time.monotonic()
    results['speed_guard'] = None
    while any(child.poll() is None for child in children.values()):
        time.sleep(15 if mode != 'probe' else 5)
        progress = {name: read_json(WORKING / name / 'progress.json') or {} for name in arms}
        alive = [name for name in arms if children[name].poll() is None]
        reference = arms[0] if arms[0] in alive else (alive[0] if alive else None)
        ref = progress.get(reference) or {}
        done = int(ref.get('completed') or 0)
        history = ref.get('history') or []
        if plan_version == 0 and reference and done >= PLAN_AT[mode]:
            low = 100 if mode == 'full' else (SMOKE_SPEED_FROM if mode == 'smoke' else 0)
            rate = mean_between(history, low, done - done % HISTORY_EVERY if mode == 'full' else done, 2)
            if rate is None and history:
                rate = history[-1][2] / max(1, history[-1][0])
            eval_s = ref.get('u0_eval_seconds') or 90.
            plan = compute_plan(rate or 2.0, eval_s, done, session_elapsed(), T_MAX[CONFIG],
                                f'arm {reference} u{done}')
            plan_version = 1
            plan['version'] = plan_version
            if mode == 'smoke':
                plan['projection'] = projection(rate or 2.0, eval_s, ref.get('train_started_elapsed'))
            write_json(WORKING / 'plan.json', plan)
            log('plan_written', **plan)
            results['nvidia_smi_at_plan'] = nvidia_smi()
            log('nvidia_smi', gpus=results['nvidia_smi_at_plan'])
        if mode == 'full' and not guard_done and reference == arms[0] and done >= SPEED_GUARD['at']:
            guard_done = True
            step = mean_between(history, *SPEED_GUARD['window'], 1)
            guard = {'arm': reference, 'mean_step_seconds_100_300': step,
                     'p2_silver_seconds_per_update': P2_SILVER_SECONDS_PER_UPDATE}
            if step is not None and P2_SILVER_SECONDS_PER_UPDATE:
                guard['ratio'] = step / P2_SILVER_SECONDS_PER_UPDATE
                guard['fail'] = guard['ratio'] > SPEED_GUARD['ratio']
                if guard['fail'] and len(arms) > 1 and children[arms[1]].poll() is None:
                    (WORKING / name_stop(arms[1])).write_text('speed_guard')
                    guard['stopped'] = arms[1]
                    refit_pending = True
            results['speed_guard'] = guard
            log('speed_guard', **guard)
        if mode == 'full' and refit_pending and not refit_done and done >= SPEED_GUARD['refit_at']:
            refit_done = True
            rate = mean_between(history, *SPEED_GUARD['refit_window'], 2)
            if rate is not None:
                plan = compute_plan(rate, ref.get('u0_eval_seconds') or 90., done, session_elapsed(),
                                    T_MAX[CONFIG], f'refit after the speed guard, arm {reference} u{done}')
                plan_version += 1
                plan['version'] = plan_version
                write_json(WORKING / 'plan.json', plan)
                log('plan_written', **plan)
        if not stop_sent and session_elapsed() > self_stop_s():
            stop_sent = True
            for name in alive:
                (WORKING / name_stop(name)).write_text('deadline')
            log('deadline_stop_sent', arms=alive)
        if session_elapsed() > session_limit_s() - 4 * 60:
            for name in alive:
                children[name].terminate()
            log('children_terminated', arms=alive)
            break
        if time.monotonic() - last_merge > 600:
            last_merge = time.monotonic()
            results['nvidia_smi_last'] = nvidia_smi()
            merge_results(results, arms)
    for name, child in children.items():
        try:
            child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            child.kill()
    results['exit_codes'] = {name: child.returncode for name, child in children.items()}


def verdict(results, arms, mode, count):
    out = {'mode': mode, 'arms': arms}
    checks = {}
    full_census = results.get('census') or {}
    for name in arms:
        arm = results['arms'].get(name) or {}
        spec = ARM_SPECS[name]
        startup = (arm.get('checks') or {}).get('startup') or {}
        u0 = arm.get('u0_check') or {}
        resume = (arm.get('checks') or {}).get('resume_equality') or {}
        seconds = arm.get('history') or []
        pools = (results.get('arm_pools') or {}).get(name) or {}
        speed = mean_between(seconds, SMOKE_SPEED_FROM, SMOKE_UPDATES, 1) if mode == 'smoke' else None
        wall = mean_between(seconds, SMOKE_SPEED_FROM, SMOKE_UPDATES, 2) if mode == 'smoke' else None
        ratio = (speed / P2_SILVER_SECONDS_PER_UPDATE) if speed and P2_SILVER_SECONDS_PER_UPDATE else None
        tests = arm.get('tests') or []
        if spec['bronze']:
            pools_ok = bool(pools.get('same_pools_as_ka'))
        else:
            pools_ok = (pools.get('silver_bronze_rows') == 0 and pools.get('weak_rows_with_bronze_absent') == 0
                        and pools.get('removed_silver_rows') == full_census.get('bronze_wrap_all_in_silver')
                        and pools.get('removed_weak_absent_lists') == full_census.get('weak_bronze_rows_with_negatives'))
        checks[name] = {
            'status': arm.get('status'), 'exit_code': (results.get('exit_codes') or {}).get(name),
            'silver_mode': spec['silver_mode'], 'bronze': spec['bronze'], 'pools_ok': pools_ok,
            'pool_digest': pools.get('pool_digest'), 'silver_pool': pools.get('silver'),
            'u0': u0.get('combined_t065'), 'u0_delta': u0.get('delta'), 'u0_status': u0.get('status'),
            'r2_identity_difference': (startup.get('r2_identity') or {}).get('difference'),
            'r2_identity_pass': (startup.get('r2_identity') or {}).get('pass'),
            'arm_objective_difference': (startup.get('arm_objective') or {}).get('difference'),
            'arm_objective_pass': (startup.get('arm_objective') or {}).get('pass'),
            'bronze_strong_pass': (startup.get('bronze_strong') or {}).get('pass'),
            'bronze_negative_value': (startup.get('bronze_negative') or {}).get('value'),
            'bronze_negative_head_gradient': (startup.get('bronze_negative') or {}).get('head_gradient_norm'),
            'bronze_negative_pass': (startup.get('bronze_negative') or {}).get('pass'),
            'bronze_off_pass': (startup.get('bronze_off') or {}).get('pass'),
            'resume_pass': resume.get('pass'), 'updates': arm.get('completed'),
            'mean_step_seconds': speed, 'mean_wall_seconds': wall, 'speed_ratio_vs_p2_silver': ratio,
            'gpu': (arm.get('checks') or {}).get('gpu'),
            'test_predictions': (arm.get('test') or {}).get('clips'),
            'test_updates': [item.get('update') for item in tests],
            'test_at_d_and_t': ({arm.get('plateau'), arm.get('total')} <= {item.get('update') for item in tests})}
    out['arms_detail'] = checks
    uuids = {c['gpu'].get('uuid') or c['gpu'].get('pci_bus_id') or c['gpu'].get('visible_devices')
             for c in checks.values() if c.get('gpu')}
    out['nvidia_smi_at_plan'] = results.get('nvidia_smi_at_plan')
    run_checks = {
        'finished': all(c['status'] == 'complete' and c['exit_code'] == 0 for c in checks.values()),
        'pools': all(c['pools_ok'] for c in checks.values()),
        'r2_identity': all(c['r2_identity_pass'] for c in checks.values()),
        'arm_objective': all(c['arm_objective_pass'] for c in checks.values()),
        'bronze': all((c['bronze_negative_pass'] is True and c['bronze_strong_pass'] is True) if c['bronze']
                      else c['bronze_off_pass'] is True for c in checks.values()),
        'resume': all(c['resume_pass'] for c in checks.values()),
        'test_at_d_and_t': all(c['test_at_d_and_t'] for c in checks.values()),
    }
    if mode == 'smoke':
        run_checks['u0'] = all(c['u0_status'] == 'pass' for c in checks.values())
        run_checks['both_gpus'] = count >= 2 and len(arms) == 2 and len(uuids) == 2
        run_checks['speed'] = (all(c['speed_ratio_vs_p2_silver'] is not None and c['speed_ratio_vs_p2_silver'] <= 1.4
                             for c in checks.values()) if P2_SILVER_SECONDS_PER_UPDATE else None)
    out['run_checks'] = run_checks
    out['all_pass'] = all(v is not False and v is not None for v in run_checks.values())
    return out


def main():
    child = os.environ.get('ATST_SCALE_CHILD')
    if child:
        run_child(child)
        return
    try:
        parent_main()
    except BaseException as error:
        log('parent_failed', error=repr(error), traceback=traceback.format_exc()[-3000:])
        payload = read_json(WORKING / 'results.json', {}) or {}
        payload['error'] = traceback.format_exc()[-6000:]
        payload['status'] = 'failed'
        try:
            write_json(WORKING / 'results.json', payload)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
