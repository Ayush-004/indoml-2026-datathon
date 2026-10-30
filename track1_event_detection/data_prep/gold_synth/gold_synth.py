#!/usr/bin/env python
"""gold_synth.py - synthetic training clips with EXACT labels, built from Gold material.

Recipe A ("organizer-like"): clean speech bed + 1-5 Gold event cuts pasted additively.
Recipe B ("natural-plus") : a Gold clip kept as it is, with 1-2 extra Gold event cuts
                            pasted on its speech-only regions.

The recipe copies what the organizers measurably did in their synthetic validation clips:
  * noisy = bed + pieces, additive, pieces confined to the labelled windows;
  * a piece is the RAW waveform cut between two annotated boundaries of a source
    recording (so it carries the source background and often the source speech);
  * 1-5 pieces per clip, gaps >= 0.200 s, >= 0.2 s of margin at both ends;
  * one target piece/bed level per clip, with a floor when the bed window is silent.

Everything is deterministic in `seed`: clip i is built from
np.random.default_rng(SeedSequence(seed, spawn_key=(i,))) alone, so the output does
not depend on the number of workers.

Inputs: T1_WORK/vaani-gold/{train,val}.jsonl, gold_stats.pkl (gold_stats.py), val_levels.pkl (measure_levels.py:
the piece-to-bed levels of the 695 synthetic validation clips), --val-meta (validationMetadata.json) and, for beds
of kind 'valref', the clean references in INDOML_VALIDATION_ZIP. prepare_gold_pools.py later drops every clip
built on a 'valref' bed, so the training clips use Gold beds only. piece_speech.pkl is optional; with the
default --piece-max-words -1 it only labels the pieces and does not change the clips.

  python gold_synth.py --out gold-synth --seed 20260916 --count-a 6000 --count-b 3000
"""
import os, sys, json, time, math, argparse, pickle, zipfile, io
import numpy as np
import soundfile as sf
from multiprocessing import Pool

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('T1_WORK', '.')
# The vaani-gold rows hold absolute audio paths. If the audio moved after the split was written, set both prefixes.
OLD = os.environ.get('T1_AUDIO_PREFIX_OLD', '')
NEW = os.environ.get('T1_AUDIO_PREFIX_NEW', '')
GOLD = os.path.join(WORK, 'vaani-gold', '')
VALZIP = os.environ.get('INDOML_VALIDATION_ZIP', 'validation.zip')

MIN_GAP = 0.200          # s, measured minimum gap between organizer pieces
EDGE = 0.200             # s, margin kept at both ends
XFADE = 0.010            # s, crossfade when beds are concatenated
PEAK_CEIL = 0.99

# ---------------------------------------------------------------- shared state
CFG = None


def fold_of_name(clip_id):
    """IISc_VaaniProject_<L>_<State>_<District>_...  -> 'State|District'"""
    p = clip_id.split('_')
    return (p[3] + '|' + p[4]) if len(p) > 4 else 'UNKNOWN|UNKNOWN'


def gold_rows():
    rows = []
    for split in ('train', 'val'):
        for line in open(GOLD + split + '.jsonl'):
            r = json.loads(line)
            r['_split'] = split
            r['_path'] = r['audio_path'].replace(OLD, NEW)
            m = r['source']['metadata']
            r['_fold'] = m['state'] + '|' + m['district']
            rows.append(r)
    return rows


# ---------------------------------------------------------------- audio access
_CACHE = {}
_ZIP = None


def read_gold(path):
    a = _CACHE.get(path)
    if a is None:
        a, _ = sf.read(path, dtype='float64')
        if a.ndim > 1:
            a = a.mean(axis=1)
        if len(_CACHE) > 96:
            _CACHE.clear()
        _CACHE[path] = a
    return a


def read_valref(clip):
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(VALZIP)
    a, _ = sf.read(io.BytesIO(_ZIP.read('validation/syntheticCleanRefAudio/' + clip)), dtype='float64')
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a


# ---------------------------------------------------------------- helpers
def rms(a):
    return float(np.sqrt(np.mean(a * a)) + 1e-12) if len(a) else 1e-12


def xfade_concat(a, b, n):
    n = min(n, len(a), len(b))
    if n <= 1:
        return np.concatenate([a, b])
    w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    out = np.concatenate([a[:-n], a[-n:] * (1 - w) + b[:n] * w, b[n:]])
    return out


def place(n, durs, D, rng):
    """n non-overlapping windows of lengths `durs` inside [0, D] with MIN_GAP gaps
    and EDGE margins; the slack is split over the n+1 gaps by a uniform Dirichlet."""
    need = float(np.sum(durs)) + (n - 1) * MIN_GAP + 2 * EDGE
    slack = D - need
    if slack < 0:
        return None
    cuts = np.sort(rng.random(n))          # n breakpoints -> n+1 Dirichlet weights
    w = np.diff(np.concatenate([[0.0], cuts, [1.0]]))
    extra = w * slack
    ons, t = [], EDGE + extra[0]
    for i in range(n):
        ons.append(t)
        t += durs[i] + (MIN_GAP + extra[i + 1] if i < n - 1 else 0.0)
    return ons


def pick_pool(lib, fold, args):
    """Tag -> piece-index map to sample from.  With --fold-safe the pieces must come from
    the same Gold state|district as the bed/host, so dropping one fold drops every scrap
    of material from it.  Folds too small to fill a clip fall back to the whole split."""
    if args.fold_safe:
        p = lib['by_fold'].get(fold)
        if p and len(p) >= args.fold_min_pieces:
            return lib['by_fold_tag'][fold]
    return lib['by_tag']


def draw_piece(lib, pool, rng, max_dur, taken, forbid_clip, tries=40):
    """Sample a piece: first a tag from the empirical Gold tag mix, then a piece with
    that tag.  Sampling the tag first keeps the generated tag mix on the Gold mix even
    though the piece pool is quality-filtered."""
    tags, cdf = lib['tags'], lib['tag_cdf']
    for _ in range(tries):
        t = tags[int(np.searchsorted(cdf, rng.random()))]
        idxs = pool.get(t)
        if not idxs:
            continue
        p = lib['pieces'][idxs[int(rng.integers(len(idxs)))]]
        if p['clip_id'] == forbid_clip or p['pid'] in taken or p['dur'] > max_dur:
            continue
        return p
    return None


def fit_clip(y):
    """Scale the finished mix so it fits in [-1, 1].

    One constant is applied to bed AND pieces together, so every level relation inside
    the clip - piece against bed, piece against piece - is exactly the one that was
    asked for.  The output is peak-normalised, which is what PCM16 wants anyway; only
    the absolute level differs from the organizers, and no detector can see that.
    """
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    c = 1.0 if pk <= PEAK_CEIL else PEAK_CEIL / pk
    return y * c, c


# ---------------------------------------------------------------- library
def build_library(args):
    """Load bed / piece indices and the organizer level statistics."""
    st = pickle.load(open(args.gold_stats, 'rb'))
    lv = pickle.load(open(args.val_levels, 'rb'))
    rows = {r['clip_id']: r for r in gold_rows()}
    spw = pickle.load(open(args.piece_speech, 'rb')) if os.path.exists(args.piece_speech) else {}

    splits = tuple(args.gold_split.split(','))

    # ---- pieces: one per Gold event, cut at the annotated boundaries -----------
    pieces = []
    for p in st['pieces']:
        r = rows[p['clip_id']]
        if r['_split'] not in splits:
            continue
        if not (args.piece_min_dur <= p['dur'] <= args.piece_max_dur):
            continue
        # keep only cuts whose window really stands above the source clip's own speech
        # level; a cut well below it is mostly the source speech, not the event.
        if p['ev_to_speech_db'] < args.piece_min_snr_db:
            continue
        if p['rms'] < 1e-4:
            continue
        # words spoken inside the annotated span of the source recording: the cut is the
        # raw waveform, so a span with words carries the source speaker's voice too.
        nw = spw.get(p['clip_id'], {}).get(p['idx'], -1)
        if args.piece_max_words >= 0 and nw > args.piece_max_words:
            continue
        pieces.append(dict(pid='%s#%d' % (p['clip_id'], p['idx']), clip_id=p['clip_id'],
                           path=r['_path'], onset=p['onset'], offset=p['offset'], dur=p['dur'],
                           tag=p['tag'], category=p['category'], rms=p['rms'],
                           fold=r['_fold'], split=r['_split'], speech_words=nw,
                           ev_to_speech_db=p['ev_to_speech_db']))

    # ---- beds -----------------------------------------------------------------
    beds = []
    if 'gold' in args.bed_kinds:
        for b in st['beds']:
            r = rows[b['clip_id']]
            if r['_split'] not in splits:
                continue
            if b['dur'] < args.bed_min_dur:
                continue
            if b['dr_db'] < args.bed_min_dr_db or b['flat'] > args.bed_max_flat:
                continue
            if b['act_frac'] < args.bed_min_act_frac or b['peak'] > 0.995:
                continue
            beds.append(dict(bid='%s@%.3f-%.3f' % (b['clip_id'], b['start'], b['end']),
                             kind='gold', clip_id=b['clip_id'], path=r['_path'],
                             start=b['start'], end=b['end'], dur=b['dur'],
                             fold=r['_fold'], split=r['_split'],
                             language=r['language'], rms=b['rms']))
    if 'valref' in args.bed_kinds:
        vm = {r['segmentFileName']: r for r in json.load(open(args.val_meta))}
        for b in lv['beds']:
            if b['dur'] < args.bed_min_dur:
                continue
            v = vm.get(b['clip'])
            fold = (v['state'] + '|' + v['district']) if v else fold_of_name(b['clip'])
            beds.append(dict(bid=b['clip'], kind='valref', clip_id=b['clip'], path=None,
                             start=0.0, end=b['dur'], dur=b['dur'],
                             fold=fold, split='organizer_validation',
                             language=(v or {}).get('language'), rms=b['rms']))

    # ---- organizer level statistics -------------------------------------------
    # snr_pool  : one value per validation synthetic clip - the piece/bed ratio measured
    #             in the LOCAL event window (constant within a clip, SD 0.00 dB).
    # sil_pool  : the floor used when the bed window is silent, relative to the bed's
    #             whole-clip rms (tight, median -3.3 dB).
    ev = lv['events']
    per_clip_snr, npc, cdur = {}, {}, {}
    for e in ev:
        npc[e['clip']] = npc.get(e['clip'], 0) + 1
        cdur[e['clip']] = e['clipdur']
        if e['bed_rms'] >= 1e-4:
            per_clip_snr.setdefault(e['clip'], e['ratio_db'])
    snr_pool = np.array(sorted(per_clip_snr.values()))
    sil_pool = np.array(sorted(e['ratio_glob_db'] for e in ev if e['bed_rms'] < 1e-4))
    npiece_pool = np.array(sorted(npc.values()))
    dp = np.array(sorted(cdur.values()))
    dur_pool = dp[(dp >= args.clip_min_dur) & (dp <= args.clip_max_dur)]

    # index pieces by fold (fold-safe sampling) and by tag (tag-mix matching)
    by_fold, by_tag, by_fold_tag = {}, {}, {}
    for i, p in enumerate(pieces):
        by_fold.setdefault(p['fold'], []).append(i)
        by_tag.setdefault(p['tag'], []).append(i)
        by_fold_tag.setdefault(p['fold'], {}).setdefault(p['tag'], []).append(i)
    # target tag mix: 'gold' = empirical Gold event mix of the chosen split(s)
    #                 'val-synth' = the mix the organizers used in their synthetic clips
    cnt = {}
    if args.tag_mix == 'val-synth':
        for r in json.load(open(args.val_meta)):
            if r['syntheticData']:
                for e in r['NoiseSubCategoryTimeStamp']:
                    cnt[e['tag']] = cnt.get(e['tag'], 0) + 1
    else:
        for r in rows.values():
            if r['_split'] in splits:
                for e in r['events']:
                    cnt[e['tag']] = cnt.get(e['tag'], 0) + 1
    tags = sorted(t for t in cnt if t in by_tag)
    w = np.array([cnt[t] for t in tags], float)
    tag_cdf = np.cumsum(w / w.sum())
    return dict(pieces=pieces, beds=beds, by_fold=by_fold, by_tag=by_tag,
                by_fold_tag=by_fold_tag, tags=tags, tag_cdf=tag_cdf,
                gold_tag_counts=cnt, snr_pool=snr_pool,
                sil_pool=sil_pool, npiece_pool=npiece_pool, dur_pool=dur_pool,
                gold=rows, args=vars(args))


# ---------------------------------------------------------------- recipe A
def make_bed(rng, bed, D, lib):
    """Return a bed waveform of exactly round(D*SR) samples."""
    need = int(round(D * SR))
    used = [bed['bid']]
    if bed['kind'] == 'valref':
        x = read_valref(bed['clip_id'])
    else:
        a = read_gold(bed['path'])
        x = a[int(round(bed['start'] * SR)):int(round(bed['end'] * SR))]
    # extend by crossfading further beds - same fold first, then any bed of the same kind
    same = lib['bed_by_fold'].get((bed['kind'], bed['fold'])) or []
    anyk = lib['bed_by_kind'][bed['kind']]
    guard = 0
    while len(x) < need and guard < 12:
        guard += 1
        src = same if (len(same) > 1 and guard <= 8) else anyk
        cand = src[int(rng.integers(len(src)))]
        if cand['bid'] == bed['bid'] and len(src) > 1:
            continue
        y = read_valref(cand['clip_id']) if cand['kind'] == 'valref' else \
            read_gold(cand['path'])[int(round(cand['start'] * SR)):int(round(cand['end'] * SR))]
        x = xfade_concat(x, y, int(XFADE * SR))
        used.append(cand['bid'])
    while len(x) < need:                    # last resort: crossfade the bed onto itself
        x = xfade_concat(x, x, int(XFADE * SR))
    off = int(rng.integers(0, len(x) - need + 1))
    return np.array(x[off:off + need], dtype=np.float64), used, off / SR


def gen_A(i, lib, args):
    rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(1, i)))
    pieces = lib['pieces']
    # deterministic bed-kind schedule: a `valref_share` fraction of the indices use the
    # organizer clean references, the rest use Gold speech-only regions.
    kinds = [k for k in ('gold', 'valref') if lib['bed_by_kind'][k]]
    if len(kinds) == 2:
        # 7919 is coprime with 1000, so the two kinds interleave instead of forming
        # blocks while the share over any 1000 consecutive indices stays exact.
        kind = 'valref' if ((i * 7919) % 1000) < int(round(args.valref_share * 1000)) else 'gold'
    else:
        kind = kinds[0]
    cands = lib['bed_by_kind'][kind]
    for _attempt in range(12):
        bed = cands[int(rng.integers(len(cands)))]
        D = float(lib['dur_pool'][int(rng.integers(len(lib['dur_pool'])))])
        pool = pick_pool(lib, bed['fold'], args)
        if not pool:
            continue
        n = int(lib['npiece_pool'][int(rng.integers(len(lib['npiece_pool'])))])
        n = max(1, min(5, n))
        budget = D - 2 * EDGE - (n - 1) * MIN_GAP
        sel, taken = [], set()
        for _ in range(n):
            ok = draw_piece(lib, pool, rng, budget - sum(s['dur'] for s in sel),
                            taken, bed['clip_id'])
            if ok is None:
                break
            sel.append(ok)
            taken.add(ok['pid'])
        if not sel:
            continue
        ons = place(len(sel), [s['dur'] for s in sel], D, rng)
        if ons is None:
            continue
        x, bed_used, bed_off = make_bed(rng, bed, D, lib)
        bglob = rms(x)
        snr = float(lib['snr_pool'][int(rng.integers(len(lib['snr_pool'])))])
        sil = float(lib['sil_pool'][int(rng.integers(len(lib['sil_pool'])))])
        add = np.zeros_like(x)
        events, srcs = [], []
        for s, on in zip(sel, ons):
            a = int(round(on * SR))
            seg = read_gold(s['path'])[int(round(s['onset'] * SR)):int(round(s['offset'] * SR))]
            b = a + len(seg)
            if b > len(add):
                seg = seg[:len(add) - a]
                b = len(add)
            if b - a < 160:                       # < 10 ms left: drop the piece
                continue
            local = rms(x[a:b])
            target = max(local * 10 ** (snr / 20.0), bglob * 10 ** (sil / 20.0))
            g = target / rms(seg)
            g = float(min(g, args.max_gain))
            add[a:b] += g * seg
            events.append(dict(onset=round(a / SR, 4), offset=round(b / SR, 4),
                               category=s['category'], tag=s['tag']))
            srcs.append(dict(pid=s['pid'], clip_id=s['clip_id'], fold=s['fold'],
                             split=s['split'], onset=s['onset'], offset=s['offset'],
                             tag=s['tag'], speech_words=s['speech_words'], gain=round(g, 6),
                             piece_rms=round(rms(seg) * g, 6), bed_local_rms=round(local, 6)))
        if not events:
            continue
        y, cscale = fit_clip(x + add)
        return dict(audio=y, duration=round(len(y) / SR, 4), events=events,
                    language=bed.get('language') or 'unknown',
                    holdout_fold=bed['fold'],
                    source=dict(recipe='A', seed=args.seed, index=i,
                                bed=dict(bid=bed['bid'], kind=bed['kind'], clip_id=bed['clip_id'],
                                         fold=bed['fold'], split=bed['split'],
                                         parts=bed_used, offset=round(bed_off, 4)),
                                pieces=srcs, snr_db=round(snr, 3), sil_floor_db=round(sil, 3),
                                clip_scale=round(cscale, 6),
                                fold_keys=sorted({bed['fold']} | {s['fold'] for s in srcs}),
                                source_clip_ids=sorted({bed['clip_id']} | {s['clip_id'] for s in srcs})))
    return None


# ---------------------------------------------------------------- recipe B
def gen_B(i, lib, args):
    rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(2, i)))
    hosts = lib['hosts']
    for _attempt in range(12):
        h = hosts[int(rng.integers(len(hosts)))]
        x = read_gold(h['_path']).copy()
        D = len(x) / SR
        # free regions: speech-only runs of the host, EDGE from the clip ends
        free = []
        for b in h['_beds']:
            s, e = max(b['start'], EDGE), min(b['end'], D - EDGE)
            if e - s >= args.b_min_free:
                free.append((s + MIN_GAP, e - MIN_GAP))
        free = [(s, e) for s, e in free if e - s >= 0.10]
        if not free:
            continue
        pool = pick_pool(lib, h['_fold'], args)
        if not pool:
            continue
        n = 1 + int(rng.integers(2))
        xglob = rms(x)
        snr = float(lib['snr_pool'][int(rng.integers(len(lib['snr_pool'])))])
        sil = float(lib['sil_pool'][int(rng.integers(len(lib['sil_pool'])))])
        add = np.zeros_like(x)
        events = [dict(onset=e['onset'], offset=e['offset'], category=e['category'], tag=e['tag'])
                  for e in h['events']]
        srcs = []
        for _k in range(n):
            if not free:
                break
            fi = int(rng.integers(len(free)))
            fs, fe = free[fi]
            room = fe - fs
            cand = draw_piece(lib, pool, rng, room, {s['pid'] for s in srcs}, h['clip_id'], tries=50)
            if cand is None:
                free.pop(fi)
                continue
            on = fs + float(rng.random()) * (room - cand['dur'])
            a = int(round(on * SR))
            seg = read_gold(cand['path'])[int(round(cand['onset'] * SR)):int(round(cand['offset'] * SR))]
            b = min(a + len(seg), len(add))
            seg = seg[:b - a]
            if b - a < 160:
                free.pop(fi)
                continue
            local = rms(x[a:b])
            target = max(local * 10 ** (snr / 20.0), xglob * 10 ** (sil / 20.0))
            g = float(min(target / rms(seg), args.max_gain))
            add[a:b] += g * seg
            events.append(dict(onset=round(a / SR, 4), offset=round(b / SR, 4),
                               category=cand['category'], tag=cand['tag']))
            srcs.append(dict(pid=cand['pid'], clip_id=cand['clip_id'], fold=cand['fold'],
                             split=cand['split'], onset=cand['onset'], offset=cand['offset'],
                             tag=cand['tag'], speech_words=cand['speech_words'], gain=round(g, 6),
                             piece_rms=round(rms(seg) * g, 6), bed_local_rms=round(local, 6)))
            # consume the used region
            free.pop(fi)
            if on - MIN_GAP - fs >= 0.15:
                free.append((fs, on - MIN_GAP))
            if fe - (on + cand['dur'] + MIN_GAP) >= 0.15:
                free.append((on + cand['dur'] + MIN_GAP, fe))
        if not srcs:
            continue
        y, cscale = fit_clip(x + add)
        events.sort(key=lambda e: e['onset'])
        return dict(audio=y, duration=round(len(y) / SR, 4), events=events,
                    language=h['language'], holdout_fold=h['_fold'],
                    source=dict(recipe='B', seed=args.seed, index=i,
                                host=dict(clip_id=h['clip_id'], fold=h['_fold'], split=h['_split'],
                                          n_original_events=len(h['events'])),
                                pieces=srcs, snr_db=round(snr, 3), sil_floor_db=round(sil, 3),
                                clip_scale=round(cscale, 6),
                                fold_keys=sorted({h['_fold']} | {s['fold'] for s in srcs}),
                                source_clip_ids=sorted({h['clip_id']} | {s['clip_id'] for s in srcs})))
    return None


# ---------------------------------------------------------------- driver
LIB = None
ARGS = None


def _init(libpath, argsd):
    global LIB, ARGS
    LIB = pickle.load(open(libpath, 'rb'))
    ARGS = argparse.Namespace(**argsd)


def _work(spec):
    recipe, i, cid = spec
    r = gen_A(i, LIB, ARGS) if recipe == 'A' else gen_B(i, LIB, ARGS)
    if r is None:
        return None
    path = os.path.join(ARGS.out, 'audio', cid)
    sf.write(path, np.clip(r['audio'], -1.0, 1.0).astype(np.float32), SR, subtype='PCM_16')
    # a handful of Gold host annotations end a fraction of a millisecond past the file;
    # clamp so every label lies inside the clip.  Nested/overlapping host events are
    # kept as they are - that is Gold ground truth, not a generator artefact.
    evs = []
    for e in r['events']:
        on = max(0.0, round(e['onset'], 4))
        off = min(r['duration'], round(e['offset'], 4))
        if off - on >= 0.010:
            evs.append(dict(onset=on, offset=off, category=e['category'], tag=e['tag']))
    r['events'] = evs
    m = dict(clip_id=cid, audio_path=os.path.abspath(path), duration=r['duration'],
             events=r['events'], quality='verified_timestamps', language=r['language'],
             holdout_fold=r['holdout_fold'], source=r['source'])
    m['source']['dataset'] = 'gold_synth'
    m['source']['metadata'] = dict(
        state=r['holdout_fold'].split('|')[0], district=r['holdout_fold'].split('|')[1],
        duration=r['duration'], language=r['language'],
        annotationQuality='verified_timestamps', isTranscriptionAvailable=False,
        transcript='', imageFileName=None,
        NoiseCategory=sorted({e['category'] for e in r['events']}),
        NoiseSubCategoryTimeStamp=[dict(start=e['onset'], end=e['offset'], tag=e['tag'])
                                   for e in r['events']])
    return json.dumps(m, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--count-a', type=int, default=6000)
    ap.add_argument('--count-b', type=int, default=3000)
    ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--gold-stats', default=os.path.join(HERE, 'gold_stats.pkl'))
    ap.add_argument('--val-levels', default=os.path.join(HERE, 'val_levels.pkl'))
    ap.add_argument('--val-meta', default=os.environ.get('T1_VAL_META', 'validationMetadata.json'))
    ap.add_argument('--gold-split', default='train', help='which Gold split supplies beds/pieces/hosts')
    ap.add_argument('--bed-kinds', default='gold,valref')
    ap.add_argument('--valref-share', type=float, default=0.5)
    ap.add_argument('--tag-mix', default='gold', choices=('gold', 'val-synth'))
    ap.add_argument('--fold-safe', type=int, default=1)
    ap.add_argument('--fold-min-pieces', type=int, default=40)
    ap.add_argument('--clip-min-dur', type=float, default=3.0)
    ap.add_argument('--clip-max-dur', type=float, default=8.0)
    ap.add_argument('--bed-min-dur', type=float, default=1.5)
    ap.add_argument('--bed-min-dr-db', type=float, default=18.0)
    ap.add_argument('--bed-max-flat', type=float, default=0.02)
    ap.add_argument('--bed-min-act-frac', type=float, default=0.25)
    ap.add_argument('--piece-min-dur', type=float, default=0.08)
    ap.add_argument('--piece-max-dur', type=float, default=6.0)
    ap.add_argument('--piece-min-snr-db', type=float, default=-20.0,
                    help='floor on the cut level relative to the source clip speech rms; '
                         'keeps the re-scaling gain reasonable')
    ap.add_argument('--piece-max-words', type=int, default=-1,
                    help='-1 keeps every cut; 0 keeps only cuts with no spoken word '
                         'inside the annotated span of the source recording')
    ap.add_argument('--piece-speech', default=os.path.join(HERE, 'piece_speech.pkl'))
    ap.add_argument('--b-min-free', type=float, default=1.0)
    ap.add_argument('--max-gain', type=float, default=20.0)
    args = ap.parse_args()
    args.bed_kinds = args.bed_kinds.split(',')

    os.makedirs(os.path.join(args.out, 'audio'), exist_ok=True)
    t0 = time.time()
    lib = build_library(args)
    lib['bed_by_kind'] = {k: [b for b in lib['beds'] if b['kind'] == k] for k in ('gold', 'valref')}
    bbf = {}
    for b in lib['beds']:
        bbf.setdefault((b['kind'], b['fold']), []).append(b)
    lib['bed_by_fold'] = bbf
    # hosts for recipe B: Gold clips of the chosen split with >=1 usable free region
    st = pickle.load(open(args.gold_stats, 'rb'))
    bybed = {}
    for b in st['beds']:
        bybed.setdefault(b['clip_id'], []).append(b)
    hosts = []
    for r in gold_rows():
        if r['_split'] not in args.gold_split.split(','):
            continue
        bs = bybed.get(r['clip_id'])
        if not bs or not r['events']:
            continue
        if max(b['dur'] for b in bs) < args.b_min_free:
            continue
        r['_beds'] = bs
        hosts.append(r)
    lib['hosts'] = hosts
    print('library: %d pieces, %d beds (gold %d / valref %d), %d hosts  [%.1fs]' % (
        len(lib['pieces']), len(lib['beds']), len(lib['bed_by_kind']['gold']),
        len(lib['bed_by_kind']['valref']), len(hosts), time.time() - t0), flush=True)

    libpath = os.path.join(args.out, '_lib.pkl')
    pickle.dump(lib, open(libpath, 'wb'))

    specs = []
    for i in range(args.count_a):
        specs.append(('A', i, 'goldsynthA_%s_%06d.wav' % (args.seed, i)))
    for i in range(args.count_b):
        specs.append(('B', i, 'goldsynthB_%s_%06d.wav' % (args.seed, i)))

    argsd = vars(args)
    out_lines, fails = [], 0
    with Pool(args.nproc, initializer=_init, initargs=(libpath, argsd)) as pool:
        for k, line in enumerate(pool.imap(_work, specs, chunksize=16)):
            if line is None:
                fails += 1
            else:
                out_lines.append(line)
            if (k + 1) % 1000 == 0:
                print('%d/%d  %.0fs  fails %d' % (k + 1, len(specs), time.time() - t0, fails), flush=True)
    man = os.path.join(args.out, 'manifest.jsonl')
    with open(man, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    os.remove(libpath)
    meta = dict(args=argsd, n_written=len(out_lines), n_failed=fails,
                seconds=round(time.time() - t0, 1),
                n_pieces=len(lib['pieces']), n_beds=len(lib['beds']),
                n_beds_gold=len(lib['bed_by_kind']['gold']),
                n_beds_valref=len(lib['bed_by_kind']['valref']), n_hosts=len(hosts))
    json.dump(meta, open(os.path.join(args.out, 'gen_meta.json'), 'w'), indent=1)
    print('WROTE %d clips (%d failed) in %.0fs -> %s' % (len(out_lines), fails, time.time() - t0, man))


if __name__ == '__main__':
    main()
