"""SR1 label-grid library: the fit and the apply step of SR1.

The rule:
    synthetic onset  = round((p + c_k)/16000, 3)
    synthetic offset = round((p + e_k)/16000, 3)
p   = the exact scan lag of the template hit that made the event (hits_*.pkl of the shipped pipeline);
c_k = per-kernel start offset, e_k = per-kernel end offset, both fit on the val exemplar lags.

Test functions (map_event, apply_line) read no label. The fit functions (fit_rows, fit_table) read the
val labels and the val clean references: they are the calibration step. The val-only adapter
(own_lag) reads only the noisy val audio.

Everything here is read-only against its inputs. CPU only; the callers set the threads.
INPUTS lists the files of the deployed run and PINS their sha256 (check_pins). The batch driver that applied
apply_line to the stored synthetic lines is not part of this release.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1] / 'inference'
WORK = Path(os.environ.get('INDOML_WORK', PIPE / 'work'))
TPL = WORK / 'templates'
WMF = HERE.parent / 'wmf'
WMF_CACHE = Path(os.environ.get('T1_WMF_CACHE', WMF / 'cache'))
SROUTER = Path(os.environ.get('T1_SR1_INPUTS', 'sr1-inputs'))   # stored records of the deployed run (see INPUTS)

sys.path.insert(0, str(PIPE))
from indoml_pipeline import router as ROUTER  # noqa: E402
from indoml_pipeline.score import clip_dice, match_events, score  # noqa: E402,F401  (official scorer, unchanged)

SR = 16000
MASTER = frozenset(range(103, 126)) | frozenset(range(84, 89)) | frozenset({64, 65, 66})
ONSET_TOL = 0.00007        # s: hit lag vs deployed onset
OFFSET_TOL = 0.0021        # s: template extent vs deployed offset
SEARCH = 48                # samples: exact-lag search half width (fit rows and the val adapter)
WINDOW = 40                # samples: vernier search half width around b
NCC_MIN = 0.9              # fit rows: kernel-0 correlation at the exact lag
CLIP_END = 0.0015          # s: clip-end rows (fit) and clip-end offsets (apply)
MIN_ROWS = 2               # a constant is applied only with >= 2 fit rows
MAX_SHIFT = 0.0015         # s: check C5

INPUTS = {
    'sub16_zip': SROUTER / 'sub16-ftwo-fb-cascade-floor.zip',
    'hits_test': TPL / 'hits_test.pkl',
    'hits_val': TPL / 'hits_val.pkl',
    'exemplars': TPL / 'exemplars.pkl',
    'test_durations': WORK / 'test-durations.json',
    'val_durations': WORK / 'val-durations.json',
    'val_meta': WMF_CACHE / 'val_meta.json',
    'val_preds': WMF_CACHE / 'val_preds.pkl',
    'valsyn_i16': WMF_CACHE / 'valsyn.i16',
    'valsyn_index': WMF_CACHE / 'valsyn_index.json',
    'valclean_i16': WMF_CACHE / 'valclean.i16',
    'valclean_index': WMF_CACHE / 'valclean_index.json',
    'val_folds': SROUTER / 'val_syn_folds.json',
    'q2_added': HERE.parent / 'copies' / 'q2_added.json',
    'router_fix': SROUTER / 'router_fix_results.json',
    'score_py': PIPE / 'indoml_pipeline/score.py',
    'router_py': PIPE / 'indoml_pipeline/router.py',
    's14_calibration': SROUTER / 's14_calibration.json',
    's08_rows': SROUTER / 's08_rows.json',
    's18_fold_cv': SROUTER / 's18_fold_cv.json',
    's14_test_candidate': SROUTER / 's14_test_calibrated_synthetic.json',
    'sub16_val_file': SROUTER / 'sub16_val_natural.jsonl',
    'wmf_base': WMF_CACHE / 'base.pkl',                       # information: provenance of the job2 events
    'wmf_test_changes': WMF / 'test_changes.json',            # information: provenance of the WMF ADD events
    'y01_diff': SROUTER / 'y01_sr1_diff.json',
}
PINS = {
    'sub16_zip': '14487ad87fe52c111c8efeb91dcdb0770b3850a1b7468936071f207f1961d2e9',
    'hits_test': 'da8d71459e0676b327b439c0d5521b8ea6f85273a021cef3503b70d1e5949a01',
    'hits_val': '3d39648b006a9d4ef32b2ca612e3ec31ebcf419098170b02a9432e4fe39c42e7',
    'exemplars': 'b69195ffc020e7cda08eff78b9dce3ed76b9e68dbf112f37bca3c5d6a2ca7cff',
    'test_durations': '96e696eabf81310167f06695e9a1206eb64aee6376dec7cdd991a3e4ff4c8cb0',
    'val_durations': '9ad3ec87ebd30992ec7e41893650586e27f3c6b3f97de620a554beaf520e23a4',
    'val_meta': 'b3bf7d4ffa3d0663ce56aec444dcead8f64f0e21877894fe279645c8e0871311',
    'val_preds': '40a93b656d6c50b7fd5fd22808907abe0605e5f521c87dc3d99b2f158aa7dc64',
    'valsyn_i16': '42f48b18180b692a45d9e31cb299bcfd380be0e10cce1efdee0f5e0126fce7ed',
    'valsyn_index': '03a7603680d4517023be8d5ea5a2b0e2c676150671903b3d9c5b40f0e81bf3a2',
    'valclean_i16': '40c497dc14c1f8621ae53a777147699abe5ec44b96a0419722ca3d1480db2053',
    'valclean_index': '03a7603680d4517023be8d5ea5a2b0e2c676150671903b3d9c5b40f0e81bf3a2',
    'val_folds': 'f55f76578fc078643adce3ffeb48d06de24e2a4d831d08173be9b469fa1a28fa',
    'q2_added': 'a5d8e7e772af245e45e13a90a8f5457b7f00bba9be6c0e6c81663b1da7616aaf',
    'router_fix': '6ee4e8c94aeabb57161769262ddc85f3f8fc8d489dd0bc6ac412123eb81d2b7d',
    'score_py': '3f9cd6ccee4eff677bd1db4ff29c1b417f28d2ff06afc8f9a11fc9c1f2f9ef45',
    'router_py': '201acb5c3aab6dcdb8e200f0307cf389c91937e9ccf49880907ce3f46550beed',
    's14_calibration': 'c33450f78bae17543c9471d11cef35e17efe00a7837fa349c2b623027df0a0af',
    's08_rows': '8ee3281adb11b616fc2780e4ef3a62dfcc4876d7dfd799784bb5dc628ecc98dd',
    's18_fold_cv': '36e218e095e514d9b2d14b70c0dcfea57bd6777fe4cd2ed2a40b3737dfd3084e',
    's14_test_candidate': '89a57158da7fd94aaa4301a2c63d3bbec90afb7fad9a2eed4ba700c355745e8d',
    'sub16_val_file': '0ab7f3d6ca74aad4a8c512285b5839c96cced9ee26976650ab27c9455d94dc4e',   # = build_sub16.json val output
    'wmf_base': '98b5c8e6e1025dd678c75822c35a487c74e8361864e7f91fba6a9f89ac675f32',
    'wmf_test_changes': 'd3f9752018cb5a345bdb4af2cbdfbfacf4ce0eca72ccb8419eaf34ce18407e16',
    'y01_diff': 'fb467ea45e3b0d94883d56ac1cc05512bfbea67c2a947dd81384c55910d01fc8',
}
SUB16_PRED_SHA = 'ed665ab122846cf1f91337f46bdda35dcd64594a095cd20ab21fc7fccf220816'


# ------------------------------------------------------------------ small helpers

def sha256_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def check_pins(names=None):
    """C1: the sha256 of every pinned input. Returns {name: {path, sha256, ok}}."""
    out = {}
    for name in (names or INPUTS):
        path = INPUTS[name]
        got = sha256_file(path)
        want = PINS.get(name)
        out[name] = {'path': str(path), 'sha256': got, 'pinned': want, 'ok': (want is None or got == want)}
    return out


def load_pickle(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def stem(clip):
    return clip[:-4] if clip.endswith('.wav') else clip


def F(s):
    """label ms of paste sample s: round(round(s/16000, 3) * 1000) (organizer label rule, s11/s12)."""
    return int(round(round(s / SR, 3) * 1000))


def grid_value(s):
    """round(s/16000, 3) as written in the line; checked against F(s)/1000; below 0 -> 0.0."""
    v = round(s / SR, 3)
    assert v == F(s) / 1000, (s, v, F(s))
    if v <= 0:
        return 0.0
    return v


def frame(t):
    """the Dice frame index of the organizer scorer (events_to_frames: int(t / 0.01))."""
    return int(t / 0.01)


# ------------------------------------------------------------------ templates and hits

class Templates:
    def __init__(self):
        self.EX = load_pickle(INPUTS['exemplars'])
        self.dur = [float(t['dur']) for t in self.EX]
        self.L0 = [len(t['exs'][0][1]) for t in self.EX]
        self.src0 = [t['exs'][0][0] for t in self.EX]


def load_hits(split):
    """clip file name (with '.wav') -> [(k, p, ncc)] of the shipped scan."""
    raw = load_pickle(INPUTS['hits_test' if split == 'test' else 'hits_val'])
    out = {}
    for name, hits in raw.items():
        base = name.split('/')[-1]
        assert base not in out, ('duplicate clip name in hits', base)
        out[base] = hits
    return out


def map_event(hits, onset, offset, T):
    """M1 (fixed in advance): the scan hit (k, p, ncc) that made the event, or None. Label-free.
    Returns (hit or None, number of hits at the onset)."""
    cand = [(k, p, v) for k, p, v in hits if abs(p / SR - onset) < ONSET_TOL]
    best = None
    for k, p, v in cand:
        if (k in MASTER or abs(p / SR + T.dur[k] - offset) < OFFSET_TOL
                or abs(p / SR + T.L0[k] / SR - offset) < OFFSET_TOL):
            if best is None or v > best[2]:
                best = (k, p, v)
    return best, len(cand)


# ------------------------------------------------------------------ the val audio stores (read-only)

class Store:
    def __init__(self, name):
        self.dat = INPUTS[f'{name}_i16']
        self.index = json.loads(INPUTS[f'{name}_index'].read_text())

    def i16(self, clip):
        off, n = self.index[clip]
        return np.fromfile(self.dat, dtype=np.int16, count=n, offset=2 * off)


# ------------------------------------------------------------------ fit rows (val exemplars) and the fit

def val_labels(meta):
    """clip -> [(start, end, tag)] of the labelled val synthetic clips; clip -> metadata duration."""
    lab = {r['segmentFileName']: [(float(e['start']), float(e['end']), e['tag']) for e in r['NoiseSubCategoryTimeStamp']]
           for r in meta if r['syntheticData']}
    cd = {r['segmentFileName']: float(r['duration']) for r in meta if r['syntheticData']}
    return lab, cd


def fit_rows(meta, nested, T):
    """The val exemplar rows (rule fit_rows_val_exemplars; the logic of the earlier vernier script s08_vernier.py).
    Reads the val labels and the val clean references (calibration step only)."""
    lab, cd = val_labels(meta)
    syn, cln = Store('valsyn'), Store('valclean')
    rows = []
    for c in sorted(lab):
        R = lab[c]
        pe = [(float(a), float(b), int(k)) for a, b, k, *_ in nested.get(c, [])]
        if not pe:
            continue
        res = syn.i16(c).astype(np.float64) - cln.i16(c).astype(np.float64)
        cands = []
        for ri, (ra, rb, _) in enumerate(R):
            tol = max(0.2 * (rb - ra), 0.05)
            for pi, (pa, pb, k) in enumerate(pe):
                if abs(pa - ra) <= tol and abs(pb - rb) <= tol:
                    cands.append((abs(pa - ra) + abs(pb - rb), ri, pi))
        mr, mp = set(), set()
        for _, ri, pi in sorted(cands):
            if ri in mr or pi in mp:
                continue
            mr.add(ri)
            mp.add(pi)
            ra, rb, t = R[ri]
            pa, pb, k = pe[pi]
            src, ker = T.EX[k]['exs'][0]
            L = len(ker)
            p0 = int(round(pa * SR))
            best = None
            for p in range(p0 - SEARCH, p0 + SEARCH + 1):
                if p < 0 or p + L > len(res):
                    if p < 0:
                        continue
                    seg = res[p:]
                    kk = ker[:len(seg)]
                else:
                    seg = res[p:p + L]
                    kk = ker
                nrm = np.linalg.norm(seg) * np.linalg.norm(kk) + 1e-12
                v = float(np.dot(seg, kk)) / nrm
                if best is None or v > best[0]:
                    best = (v, p)
            rows.append(dict(c=c, k=k, t=t, ra=ra, rb=rb, pa=pa, pb=pb, p=best[1], ncc=best[0], own=(src == c),
                             L=L, cd=cd[c], l_on=int(round(ra * 1000)), l_off=int(round(rb * 1000)),
                             clip_end_row=abs(rb - cd[c]) < CLIP_END))
    return rows


def compare_rows_with_s08(rows):
    """C2 part: the recomputed rows equal the stored fit rows s08_rows.json (order, k, p, labels, ncc to 1e-9)."""
    s08 = json.loads(INPUTS['s08_rows'].read_text())
    same_len = len(rows) == len(s08)
    bad = []
    for i, (a, b) in enumerate(zip(rows, s08)):
        ok = (a['c'] == b['c'] and a['k'] == b['k'] and a['p'] == b['p'] and a['ra'] == b['ra'] and a['rb'] == b['rb']
              and a['pa'] == b['pa'] and a['pb'] == b['pb'] and a['own'] == b['own'] and a['L'] == b['L']
              and a['cd'] == b['cd'] and abs(a['ncc'] - b['ncc']) < 1e-9)
        if not ok:
            bad.append(i)
    return {'rows': len(rows), 's08_rows': len(s08), 'same_length': same_len, 'rows_differing': len(bad),
            'first_differing': bad[:5], 'equal': same_len and not bad}


def fit_const(pairs):
    """The vernier fit (rule fit_c_k): pairs [(p, label_ms)] in row order.
    Returns (c, violations, number of equally good c)."""
    base = pairs[0][1] * 16 - pairs[0][0]
    sc = [(sum(1 for p, l in pairs if F(p + c) != l), c) for c in range(base - WINDOW, base + WINDOW + 1)]
    best = min(sc)[0]
    cs = [c for n, c in sc if n == best]
    return cs[len(cs) // 2], best, len(cs)


def fit_table(rows):
    """k -> {c, n_on, viol_on, width_on, e, n_off, viol_off, width_off}; rows with ncc > 0.9 only."""
    by = defaultdict(list)
    for r in rows:
        if r['ncc'] > NCC_MIN:
            by[r['k']].append(r)
    table = {}
    for k, v in by.items():
        c, vo, wo = fit_const([(o['p'], o['l_on']) for o in v])
        offs = [(o['p'], o['l_off']) for o in v if not o['clip_end_row']]
        if offs:
            e, vf, wf = fit_const(offs)
        else:
            e, vf, wf = None, None, None
        table[k] = dict(c=c, n_on=len(v), viol_on=vo, width_on=wo, e=e, n_off=len(offs), viol_off=vf, width_off=wf)
    return table


def s14_view(table):
    """The table in the s14_calibration.json format {str(k): [c_k, e_k, n]}."""
    return {str(k): [t['c'], t['e'], t['n_on']] for k, t in table.items()}


# ------------------------------------------------------------------ apply (test and val: the same function)

def apply_line(clip, events, hits, table, duration, T, q2_onsets=frozenset(), own_lag=None, min_rows=MIN_ROWS,
               master_offsets_fixed=True):
    """events: [(onset, offset)] as written in the line. Returns (new events, per-event log).
    own_lag(clip, k, p) -> p in the kernel-0 frame (val only; None on test)."""
    new, log = [], []
    for i, (on, off) in enumerate(events):
        rec = {'i': i, 'onset': on, 'offset': off, 'k': None, 'p': None, 'ncc': None, 'p_used': None,
               'onset_new': on, 'offset_new': off, 'why_onset': None, 'why_offset': None, 'n_hits_at_onset': 0}
        if on in q2_onsets:
            rec['why_onset'] = rec['why_offset'] = 'q2_event'
            new.append((on, off))
            log.append(rec)
            continue
        hit, nc = map_event(hits, on, off, T)
        rec['n_hits_at_onset'] = nc
        if hit is None:
            rec['why_onset'] = rec['why_offset'] = 'unmapped'
            new.append((on, off))
            log.append(rec)
            continue
        k, p, v = hit
        rec.update(k=int(k), p=int(p), ncc=float(v))
        pu = p
        if own_lag is not None and T.src0[k] == clip:
            pu = own_lag(clip, k, p)
            rec['own'] = True
        rec['p_used'] = int(pu)
        t = table.get(k)
        n_on = t['n_on'] if t else 0
        if t is None:
            rec['why_onset'] = rec['why_offset'] = 'template_without_val_rows'
        else:
            if n_on >= min_rows:
                rec['onset_new'] = grid_value(pu + t['c'])
                rec['why_onset'] = 'calibrated'
            else:
                rec['why_onset'] = f'template_with_{n_on}_val_row'
            if master_offsets_fixed and k in MASTER:
                rec['why_offset'] = 'master_template'
            elif t['e'] is None or t['n_off'] < min_rows:
                rec['why_offset'] = f"template_with_{t['n_off']}_offset_rows"
            elif not (duration - off > CLIP_END):
                rec['why_offset'] = 'offset_at_clip_end'
            else:
                rec['offset_new'] = grid_value(pu + t['e'])
                rec['why_offset'] = 'calibrated'
        new.append((rec['onset_new'], rec['offset_new']))
        log.append(rec)
    return new, log


class OwnLag:
    """Val-only adapter: the val scan is leave-one-clip-out, so an 'own' event (kernel 0 of k was cut from this
    clip) has its lag in the kernel-1 frame. Measure the kernel-0 lag on the NOISY val audio in +-48 samples
    around it with the statistic of the shipped scan (|correlation| / sqrt(window energy)). No label."""

    def __init__(self, T):
        self.T = T
        self.store = Store('valsyn')
        self.log = []

    def __call__(self, clip, k, p1):
        x = self.store.i16(clip).astype(np.float64) / 32768.0
        ker = self.T.EX[k]['exs'][0][1]
        L = len(ker)
        lo = max(0, p1 - SEARCH)
        hi = min(len(x) - L, p1 + SEARCH)
        assert hi >= lo, (clip, k, p1)
        seg = x[lo:hi + L]
        corr = np.abs(np.correlate(seg, ker, mode='valid'))
        e = np.concatenate([[0.0], np.cumsum(seg * seg)])
        win = np.maximum(e[L:] - e[:-L], 1e-12)
        ncc = corr / np.sqrt(win)
        j = int(np.argmax(ncc))
        p0 = lo + j
        self.log.append({'clip': clip, 'k': int(k), 'p_kernel1': int(p1), 'p_kernel0': int(p0), 'ncc0': float(ncc[j])})
        return p0


# ------------------------------------------------------------------ lines

def parse_zip_lines(blob):
    """[(clip id, raw bytes)] of a predictions.jsonl payload, file order; checks the exact reconstruction."""
    assert blob.endswith(b'\n')
    parts = blob.split(b'\n')
    assert parts[-1] == b''
    lines = parts[:-1]
    assert all(line.strip() for line in lines)
    assert b''.join(line + b'\n' for line in lines) == blob
    return [(json.loads(line)['clip_id'], line) for line in lines]


def dump_line(clip_id, events):
    """the shipped writer (build_submission.write_submission): json.dumps with default separators."""
    return json.dumps({'clip_id': clip_id, 'events': [{'onset': a, 'offset': b} for a, b in events]}).encode()


def events_of_line(raw):
    r = json.loads(raw)
    return [(e['onset'], e['offset']) for e in r['events']]


def line_format_problems(events, duration):
    out = []
    for a, b in events:
        if not (isinstance(a, float) or isinstance(a, int)) or not (isinstance(b, float) or isinstance(b, int)):
            out.append(('type', a, b))
        elif not (np.isfinite(a) and np.isfinite(b)):
            out.append(('nonfinite', a, b))
        elif not (0.0 <= a < b <= duration):
            out.append(('range', a, b, duration))
    for (a, b), (c, d) in zip(events[:-1], events[1:]):
        if c < a:
            out.append(('order', a, c))
    return out


def overlaps(events):
    return sum(1 for (a, b), (c, d) in zip(events[:-1], events[1:]) if c < b)


def frames_changed(old, new):
    n = 0
    for (a, b), (c, d) in zip(old, new):
        n += frame(a) != frame(c)
        n += frame(b) != frame(d)
    return n


def summarize_shifts(pairs):
    """pairs: [(old, new)] endpoint values."""
    sh = sorted(abs(n - o) for o, n in pairs)
    if not sh:
        return {'endpoints': 0}
    return {'endpoints': len(sh), 'moved': sum(1 for x in sh if x > 0), 'max_ms': round(1000 * sh[-1], 4),
            'p99_ms': round(1000 * sh[int(0.99 * len(sh))], 4), 'above_1ms': sum(1 for x in sh if x > 0.001 + 1e-12),
            'above_1.5ms': sum(1 for x in sh if x > MAX_SHIFT + 1e-12)}


# ------------------------------------------------------------------ scoring helpers (official scorer)

def score_sets(ref, pred, clips):
    s = score(ref, pred, clips)
    return {'clips': len(clips), 'f1': s['f1'], 'dice': s['dice'], 'combined': s['combined'], 'tp': s['tp'],
            'fp': s['fp'], 'fn': s['fn']}


def delta(ref, before, after, clips):
    a = score_sets(ref, before, clips)
    b = score_sets(ref, after, clips)
    per = [clip_dice(ref[c], after.get(c, [])) - clip_dice(ref[c], before.get(c, [])) for c in clips]
    match_changed = sum(1 for c in clips if match_events(ref[c], before.get(c, [])) != match_events(ref[c], after.get(c, [])))
    return {'before': a, 'after': b, 'dDice': b['dice'] - a['dice'], 'dF1': b['f1'] - a['f1'],
            'dice_clip_units': (b['dice'] - a['dice']) * len(clips), 'sum_per_clip_delta': float(np.sum(per)),
            'tp_fp_fn_identical': ((a['tp'], a['fp'], a['fn']) == (b['tp'], b['fp'], b['fn'])) and match_changed == 0,
            'clips_with_changed_tp_fp_fn': match_changed,
            'clips_better': int(sum(1 for x in per if x > 1e-15)), 'clips_worse': int(sum(1 for x in per if x < -1e-15))}


def pseudo_gain(old, new, clips):
    """label-free: sum over clips of 1 - Dice(new line, old line)."""
    return float(sum(1 - clip_dice(new[c], old[c]) for c in clips))
