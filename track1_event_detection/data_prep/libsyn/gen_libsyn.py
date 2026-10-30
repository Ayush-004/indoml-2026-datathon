"""gen_libsyn.py: the LIBSYN generator (rules L1-L11, see the README).

Inputs:
  templates  T1_LIBRARY: library.pkl of inference/indoml_pipeline/templates.py (203 exact noise tracks from the
             organizer validation synthetic clips)
  masters    T1_MASTERS: masters.pkl of m01_masters.py (groups 66, 88, 125: master = the group key template)
  val laws   valstats.json next to this file (valstats.py: validation synthetic statistics: levels, zero runs,
             vocal slots, categories)
  beds       T1_BEDS: the clean speech bed folder (index.jsonl + int16 shards, mmap) of the Track 2 scripts
             build_banks.py and build_clean_beds.py (training release only)
  lengths    T1_TRAIN_LENGTHS: train_float.jsonl of train_float.py (training-release clip lengths)
No test file and no validation audio file is opened (an audit hook stops the run if one is).
Clip i depends only on (seed, i): np.random.default_rng(SeedSequence(seed, spawn_key=(i,))).

  python gen_libsyn.py --out libsyn/v1 --start 0 --count 2000 [--workers 4] [--keep-beds]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
PEAK = 32440
EDGE_ON, EDGE_OFF, MIN_GAP = 0.250, 0.251, 0.201      # 0.201 s between pastes -> label gaps >= 0.200 s
MASTER_MIN_INSIDE = 2.85
DMIN, DMAX = 3.0, 15.2
COUNT_W = np.array([110, 78, 124, 340, 43], float)
CLASS_W = {'vocal': 670., 'noise': 1506., 'master': 37.}
MASTERS = (66, 88, 125)
VOCAL_DB = (3.97, 0.44)
SNR_RANGE = (-12.24, 15.49)
SIGMA_W = 4.956
JOIN_MIN_MS = 10.0
LOW_ENERGY_SHARE = 0.20
SEED = 20261006

HERE = Path(__file__).resolve().parent
LIBRARY = Path(os.environ.get('T1_LIBRARY', 'work/templates/library.pkl'))
MASTERS_PKL = Path(os.environ.get('T1_MASTERS', HERE / 'masters.pkl'))
VALSTATS = HERE / 'valstats.json'
BEDS = Path(os.environ.get('T1_BEDS', 'beds-clean'))
TRAIN_DUR = Path(os.environ.get('T1_TRAIN_LENGTHS', HERE / 'train_float.jsonl'))
FORBIDDEN = ('input_data.zip', 'validation.zip', '/work/test', 'audio_test', 'hits_test', 'test-durations',
             'test_all_clips', 'test-natural', 'test_syn', 'test_nat', 'test_missing', 'exemplars.pkl',
             'tau_router', 'routes')

S = {}          # worker state


def _hook(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if any(f in p for f in FORBIDDEN) and '/site-packages/' not in p and '/lib/python' not in p:
            raise RuntimeError(f'gen_libsyn audit: forbidden file opened: {p}')


def session_of(clip_id):
    parts = clip_id.split('_')
    return parts[5] if len(parts) > 5 else None


def district_of(clip_id):
    parts = clip_id.split('_')
    return parts[3] + '|' + parts[4] if len(parts) > 4 else 'UNKNOWN'


def load_state():
    lib = pickle.load(open(LIBRARY, 'rb'))
    mast = pickle.load(open(MASTERS_PKL, 'rb'))
    vs = json.loads(VALSTATS.read_text())
    cuts = set()
    for g in MASTERS:
        cuts |= set(mast['groups'][g]) - {g}
    vocal = [k for k, t in enumerate(lib) if t['tag'].startswith('[')]
    noise = [k for k, t in enumerate(lib) if t['tag'].startswith('<') and k not in cuts and k not in MASTERS]
    assert len(vocal) == 58 and len(noise) == 145 - len(cuts) - len(MASTERS) == 114, (len(vocal), len(noise))
    cat = {int(k): v for k, v in vs['template_category'].items()}
    mu = {int(k): v for k, v in vs['template_mean_snr'].items()}
    nn = {int(k): v for k, v in vs['template_n'].items()}
    for g in MASTERS:                       # a master's level: the mean over the events of its group
        mem = [k for k in mast['groups'][g] if k in mu]
        mu[g] = float(np.sum([mu[k] * nn[k] for k in mem]) / np.sum([nn[k] for k in mem]))
        cat.setdefault(g, cat[mast['groups'][g][0]])
    assert all(k in mu and k in cat for k in noise + list(MASTERS)) and all(k in cat for k in vocal)
    w = {k: np.asarray(lib[k]['seg'], np.float64) for k in vocal + noise + list(MASTERS)}
    for k, x in w.items():
        assert np.allclose(x * 32768, np.round(x * 32768)), f'template {k} is not on the int16 grid'
    raw = vs['raw']
    margins = np.maximum(0.0, np.asarray(raw['vocal_margin_pairs_ms'], float))       # (670, 2) ms
    rate = np.asarray(raw['empty_runs_per_clip'], float) / np.asarray(raw['clip_dur_s'], float)
    runs_ms = np.asarray(raw['empty_run_ms'], float)
    joins_ms = runs_ms[runs_ms >= JOIN_MIN_MS]
    idx = [json.loads(line) for line in open(BEDS / 'index.jsonl')]
    shards = {name: np.load(BEDS / name, mmap_mode='r') for name in sorted({r['shard'] for r in idx})}
    by_session, by_district = {}, {}
    for j, r in enumerate(idx):
        by_session.setdefault(session_of(r['clip_id']), []).append(j)
        by_district.setdefault(district_of(r['clip_id']), []).append(j)
    d = np.asarray([json.loads(line)['frames'] for line in open(TRAIN_DUR)], np.int64)
    dlaw = d[(d >= int(DMIN * SR)) & (d <= int(DMAX * SR))]
    S.update(lib=lib, w=w, vocal=vocal, noise=noise, cat=cat, mu=mu, margins=margins, rate=rate, runs_ms=runs_ms,
             joins_ms=joins_ms, idx=idx, shards=shards, by_session=by_session, by_district=by_district, dlaw=dlaw)


def bed_audio(j):
    r = S['idx'][j]
    a = S['shards'][r['shard']]
    return np.asarray(a[r['offset']:r['offset'] + r['length']], np.float64) / 32768.0


def place(foot, lo, hi, rng):
    """Footprints (samples) in [lo, hi) in this order, gaps >= MIN_GAP; Dirichlet split of the slack."""
    n = len(foot)
    gap = int(round(MIN_GAP * SR))
    slack = (hi - lo) - int(np.sum(foot)) - (n - 1) * gap
    if slack < 0:
        return None
    wts = np.diff(np.concatenate([[0.0], np.sort(rng.random(n)), [1.0]]))
    extra = np.floor(wts * slack).astype(np.int64)
    starts, t = [], lo + int(extra[0])
    for i in range(n):
        starts.append(t)
        t += int(foot[i]) + (gap + int(extra[i + 1]) if i < n - 1 else 0)
    return starts


def make_bed(L, rng):
    """L samples of training-release speech: beds of one session, then of its district, then any bed;
    exact-zero joins from the val law (runs >= 10 ms)."""
    first = int(rng.integers(len(S['idx'])))
    sess = session_of(S['idx'][first]['clip_id'])
    dist = district_of(S['idx'][first]['clip_id'])
    order = [first] + [j for j in rng.permutation(S['by_session'][sess]).tolist() if j != first]
    used, parts, total, joins = set(), [], 0, []
    tiers = [order, rng.permutation(S['by_district'][dist]).tolist()]
    tier = 0
    pos = 0
    while total < L:
        if tier < len(tiers) and pos >= len(tiers[tier]):
            tier, pos = tier + 1, 0
            continue
        j = tiers[tier][pos] if tier < len(tiers) else int(rng.integers(len(S['idx'])))
        pos += 1
        if j in used:
            continue
        used.add(j)
        if parts:
            g = int(round(float(rng.choice(S['joins_ms'])) * SR / 1000.0))
            parts.append(np.zeros(g))
            joins.append(g)
            total += g
        x = bed_audio(j)
        parts.append(x)
        total += len(x)
    x = np.concatenate(parts)
    off = int(rng.integers(0, len(x) - L + 1))
    beds = [{'bed_i': S['idx'][j]['bed_i'], 'clip_id': S['idx'][j]['clip_id']} for j in used]
    return x[off:off + L].copy(), beds, {'joins': joins, 'offset': off, 'tier_reached': tier}


def empty_runs(x, D, rng):
    """Val law of empty exact-zero runs: count = rate x D, lengths from the val list, in low-energy frames."""
    k = int(round(float(rng.choice(S['rate'])) * D))
    if k == 0 or len(x) < 160:
        return []
    fr = len(x) // 160
    e = np.mean(x[:fr * 160].reshape(fr, 160) ** 2, axis=1)
    low = np.argsort(e)[:max(1, int(LOW_ENERGY_SHARE * fr))]
    out = []
    for _ in range(k):
        n = max(16, int(round(float(rng.choice(S['runs_ms'])) * SR / 1000.0)))
        s = int(low[int(rng.integers(len(low)))]) * 160 + int(rng.integers(0, 160))
        s = min(s, len(x) - n) if len(x) > n else 0
        x[s:s + n] = 0.0
        out.append((s, n))
    return out


def gen_clip(i, seed=SEED, keep_bed=False):
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(i,)))
    lib, w = S['lib'], S['w']
    rejected = []
    Ds = int(rng.choice(S['dlaw']))                  # L1: D is drawn once; a failed layout redraws n and templates
    for attempt in range(200):
        if attempt and attempt % 50 == 0:            # practically never: a new D after 50 failed layouts
            Ds = int(rng.choice(S['dlaw']))
        D = Ds / SR
        n = 1 + int(rng.choice(5, p=COUNT_W / COUNT_W.sum()))
        kinds = list(rng.choice(['vocal', 'noise', 'master'], size=n,
                                p=np.array(list(CLASS_W.values())) / sum(CLASS_W.values())))
        if kinds.count('master') > 1:                  # one master at most; it is the last event
            first = kinds.index('master')
            kinds = [('noise' if (k == 'master' and j != first) else k) for j, k in enumerate(kinds)]
        ks = []
        for kd in kinds:
            pool = S['vocal'] if kd == 'vocal' else (S['noise'] if kd == 'noise' else list(MASTERS))
            ks.append(int(pool[int(rng.integers(len(pool)))]))
        normal = [(kd, k) for kd, k in zip(kinds, ks) if kd != 'master']
        rng.shuffle(normal)
        master = [(kd, k) for kd, k in zip(kinds, ks) if kd == 'master']
        evs = []
        for kd, k in normal:
            ln = len(w[k])
            pre = post = 0
            if kd == 'vocal':
                m = S['margins'][int(rng.integers(len(S['margins'])))]
                pre, post = int(round(m[0] * 16)), int(round(m[1] * 16))
            evs.append({'kind': kd, 'k': k, 'len': ln, 'pre': pre, 'post': post, 'foot': pre + ln + post})
        lo, hi = int(round(EDGE_ON * SR)), Ds - int(round(EDGE_OFF * SR))
        mev = None
        if master:
            k = master[0][1]
            room = Ds - lo - (sum(e['foot'] for e in evs) + len(evs) * int(round(MIN_GAP * SR)))
            top = min(len(w[k]), room)
            bot = int(round(MASTER_MIN_INSIDE * SR))
            if top < bot:
                rejected.append([n, Ds, 'master'])
                continue
            inside = int(rng.integers(bot, top + 1))
            mev = {'kind': 'master', 'k': k, 'start': Ds - inside, 'len': inside, 'pre': 0, 'post': 0}
            hi = mev['start'] - int(round(MIN_GAP * SR))
        starts = place([e['foot'] for e in evs], lo, hi, rng) if evs else []
        if starts is None:
            rejected.append([n, Ds, 'layout'])
            continue
        for e, s in zip(evs, starts):
            e['start'] = s + e['pre']                  # paste start (samples, final timeline)
        allev = evs + ([mev] if mev else [])
        # bed: speech for every non-slot sample; vocal slots are inserted exact zeros
        slots = sorted((e['start'] - e['pre'], e['start'] + e['len'] + e['post']) for e in evs if e['kind'] == 'vocal')
        L = Ds - sum(b - a for a, b in slots)
        speech, beds, bedinfo = make_bed(L, rng)
        zr = empty_runs(speech, D, rng)
        bed = np.zeros(Ds)
        src, dst = 0, 0
        for a, b in slots:
            bed[dst:a] = speech[src:src + (a - dst)]
            src += a - dst
            dst = b
        bed[dst:] = speech[src:src + (Ds - dst)]
        assert src + (Ds - dst) == L
        nz = bed[bed != 0]
        if len(nz) < SR:                               # under 1 s of speech: draw again
            rejected.append([n, Ds, 'speech'])
            continue
        p_sp = float(np.mean(nz ** 2))
        add = np.zeros(Ds)
        out_ev, prov = [], []
        for e in sorted(allev, key=lambda z: z['start']):
            x = w[e['k']][:e['len']]
            if e['kind'] == 'vocal':
                lvl = -float(rng.normal(*VOCAL_DB))
            else:
                lvl = float(rng.normal(S['mu'][e['k']], SIGMA_W))
                while not (SNR_RANGE[0] <= lvl <= SNR_RANGE[1]):      # truncated normal: draw again
                    lvl = float(rng.normal(S['mu'][e['k']], SIGMA_W))
            g = float(np.sqrt(p_sp * 10 ** (lvl / 10.0) / np.mean(x ** 2)))
            add[e['start']:e['start'] + len(x)] += g * x
            on = np.floor(e['start'] / 16.0) / 1000.0
            off = D if e['kind'] == 'master' else on + round(lib[e['k']]['dur'], 3)
            out_ev.append({'onset': round(on, 3), 'offset': D if e['kind'] == 'master' else round(min(off, D), 4),
                           'category': S['cat'][e['k']], 'tag': lib[e['k']]['tag']})
            prov.append({'k': e['k'], 'kind': e['kind'], 'paste_start': int(e['start']), 'samples': int(len(x)),
                         'pre': e['pre'], 'post': e['post'], 'level_db': round(lvl, 4), 'gain': g})
        y = bed + add
        c = (PEAK / 32768.0) / float(np.max(np.abs(y)))
        yi = np.round(y * c * 32768.0)
        assert int(np.max(np.abs(yi))) == PEAK
        rec = {'clip_id': f'libsyn_v1_{i:06d}.wav', 'audio_path': f'audio/libsyn_v1_{i:06d}.flac',
               'duration': Ds / SR, 'events': out_ev, 'fold_keys': sorted({district_of(b['clip_id']) for b in beds}),
               'recipe': 'LIBSYN', 'positive_categories': sorted({e['category'] for e in out_ev}),
               'quality': 'verified_timestamps', 'supervision': 'strong_gold'}
        meta = {'clip_id': rec['clip_id'], 'index': i, 'seed': seed, 'attempt': attempt, 'rejected': rejected,
                'samples': Ds, 'beds': beds, 'bed_info': bedinfo, 'empty_runs': zr, 'slots': slots,
                'p_speech_nonzero': p_sp, 'scale': c, 'events': prov,
                'sessions': sorted({session_of(b['clip_id']) for b in beds})}
        # the scaled bed alone can pass the int16 range where an event cancels part of it: keep it as int32
        bed_i32 = np.round(bed * c * 32768.0).astype(np.int32) if keep_bed else None
        return rec, meta, yi.astype(np.int16), bed_i32
    raise RuntimeError(f'clip {i}: no valid layout in 200 attempts')


def work(args):
    i0, i1, out, keep, seed = args
    import soundfile as sf
    if not S:
        sys.addaudithook(_hook)
        load_state()
    res = []
    for i in range(i0, i1):
        rec, meta, y, bed = gen_clip(i, seed, keep)
        sf.write(out / rec['audio_path'], y, SR, subtype='PCM_16', format='FLAC')
        if bed is not None:
            sf.write(out / 'calib_beds' / f'libsyn_v1_{i:06d}.bed.flac', (bed.astype(np.int32) << 8), SR,
                     subtype='PCM_24', format='FLAC')       # exact: the int32 bed in the top bits of PCM_24
        res.append((rec, meta))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--keep-beds', action='store_true')
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    if a.keep_beds:
        (out / 'calib_beds').mkdir(exist_ok=True)
    chunk = 50
    jobs = [(s, min(s + chunk, a.start + a.count), out, a.keep_beds, a.seed)
            for s in range(a.start, a.start + a.count, chunk)]
    t0 = time.time()
    rows = []
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(a.workers) as pool:
            for r in pool.imap(work, jobs):
                rows += r
    else:
        for j in jobs:
            rows += work(j)
    tag = f'{a.start:06d}-{a.start + a.count - 1:06d}'
    with open(out / f'manifest-{tag}.jsonl', 'w') as f:
        for rec, _ in rows:
            f.write(json.dumps(rec) + '\n')
    with open(out / f'provenance-{tag}.jsonl', 'w') as f:
        for _, meta in rows:
            f.write(json.dumps(meta) + '\n')
    sess = {rec['clip_id']: meta['sessions'] for rec, meta in rows}
    (out / f'sessions-{tag}.json').write_text(json.dumps(sess) + '\n')
    import hashlib
    rep = {'tag': tag, 'clips': len(rows), 'seed': a.seed, 'start': a.start, 'count': a.count,
           'seconds': round(time.time() - t0, 1), 'audio_hours': round(sum(r['duration'] for r, _ in rows) / 3600, 3),
           'rejected_draws': sum(len(m['rejected']) for _, m in rows),
           'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'valstats_sha256': hashlib.sha256(VALSTATS.read_bytes()).hexdigest(),
           'audit_hook': 'every worker ran with an audit hook that stops on any forbidden path: ' + ', '.join(FORBIDDEN),
           'keep_beds': a.keep_beds}
    (out / f'gen-report-{tag}.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
