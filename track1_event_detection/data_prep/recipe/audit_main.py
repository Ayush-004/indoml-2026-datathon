"""Tag vocabulary and mapping, event-type prevalence per tier against the organizer natural validation clips,
and transcript-derived targets.

build_recipe.py reads the vocabulary part of out/audit_main.json: a transcript tag name is kept when the timestamped
tiers (Gold, Silver and the natural validation clips, pooled) have >= 0.5 events per transcript occurrence of the
name, and dropped below 0.05. Read-only on the data. CPU, numpy only. Writes out/audit_main.json, out/events.npz
and out/clips.json.
"""
import collections, json, math, os, sys
import numpy as np
from common_io import load_manifest, load_val, CATS
from transcript import parse, link_events

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
TIERS = ['gold', 'silver', 'bronze', 'val']
TS_TIERS = ['gold', 'silver', 'val']


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def family(name, cat):
    n = name.lower()
    if cat == 'human_non_speech':
        if 'breath' in n:
            return 'hns_breathing'
        if 'inhal' in n:
            return 'hns_inhaling'
        return 'hns_other(lip/cough/throat/sneeze...)'
    if cat == 'vehicle_traffic':
        return 'veh_horn/honk' if ('horn' in n or 'honk' in n) else 'veh_other(vehicle/traffic/motor...)'
    if cat == 'animal':
        if 'bird' in n or 'crow' in n or 'cuckoo' in n or 'rooster' in n or 'hen' in n or 'peacock' in n:
            return 'ani_bird'
        if 'bark' in n or 'dog' in n:
            return 'ani_dog'
        if 'insect' in n or 'cricket' in n or 'bee' in n or 'mosquito' in n:
            return 'ani_insect'
        return 'ani_other'
    if cat == 'baby_child':
        return 'child_talking' if 'talk' in n else 'child_cry/yell/other'
    if cat == 'singing_music':
        return 'music/singing'
    if cat == 'phone_signal_alarm':
        return 'phone/bell/beep/siren'
    if cat == 'appliance_machine':
        return 'appliance(fan/tv/machine)'
    return 'other:' + str(cat)


def bucket(dur, clipdur):
    if dur >= 0.9 * clipdur:
        return 'whole'
    if dur >= 2.0:
        return 'ge2'
    if dur >= 1.0:
        return '1-2'
    if dur >= 0.5:
        return '0.5-1'
    return 'lt0.5'


BUCKETS = ['whole', 'ge2', '1-2', '0.5-1', 'lt0.5']


def on_grid(x, step=0.1):
    return abs(x / step - round(x / step)) < 1e-3


def main():
    rows = load_manifest() + load_val()
    by = collections.defaultdict(list)
    for r in rows:
        by[r['tier']].append(r)

    # ---------------------------------------------------------------- vocabulary
    occ_count = {t: collections.Counter() for t in TIERS}
    ev_count = {t: collections.Counter() for t in TIERS}
    tag_cat = {t: collections.defaultdict(collections.Counter) for t in TIERS}
    kind_count = collections.defaultdict(collections.Counter)
    parsed = {}
    for t in TIERS:
        for r in by[t]:
            occ, nw = parse(r['transcript'])
            parsed[(t, r['clip_id'])] = (occ, nw)
            for o in occ:
                occ_count[t][o['name']] += 1
                kind_count[o['name']][o['kind']] += 1
            for e in r['events']:
                ev_count[t][e['tag']] += 1
                tag_cat[t][e['tag']][e['category']] += 1
    names = set()
    for t in TIERS:
        names |= set(occ_count[t]) | set(ev_count[t])
    # kept / dropped status from the timestamped tiers pooled
    status = {}
    for n in names:
        o = sum(occ_count[t][n] for t in TS_TIERS)
        e = sum(ev_count[t][n] for t in TS_TIERS)
        if o == 0 and e == 0:
            status[n] = 'bronze_only'
        elif o == 0:
            status[n] = 'event_without_transcript_tag'
        else:
            ratio = e / o
            status[n] = 'kept' if ratio >= 0.5 else ('dropped' if ratio < 0.05 else 'partial')
    # category of each kept name (pooled over timestamped tiers)
    name_cat = {}
    multi_cat = {}
    for n in names:
        c = collections.Counter()
        for t in TS_TIERS:
            c.update(tag_cat[t][n])
        if c:
            name_cat[n] = c.most_common(1)[0][0]
            if len(c) > 1:
                multi_cat[n] = {t: dict(tag_cat[t][n]) for t in TS_TIERS if tag_cat[t][n]}
    # tier differences in mapping: a name whose top category differs between tiers
    tier_diff = {}
    for n in names:
        tops = {t: tag_cat[t][n].most_common(1)[0][0] for t in TS_TIERS if tag_cat[t][n]}
        if len(set(tops.values())) > 1:
            tier_diff[n] = tops
    # per-tier drop rate of names that are kept elsewhere
    vocab_rows = []
    for n in sorted(names, key=lambda k: -sum(occ_count[t][k] + ev_count[t][k] for t in TIERS)):
        vocab_rows.append({'name': n, 'status': status[n], 'category': name_cat.get(n),
                           'family': family(n, name_cat.get(n)) if name_cat.get(n) else None,
                           'kinds': dict(kind_count[n]),
                           **{t + '_occ': occ_count[t][n] for t in TIERS},
                           **{t + '_events': ev_count[t][n] for t in TS_TIERS}})
    dropped = [v for v in vocab_rows if v['status'] == 'dropped']
    partial = [v for v in vocab_rows if v['status'] == 'partial']
    bronze_only = [v for v in vocab_rows if v['status'] == 'bronze_only']

    # ---------------------------------------------------------------- events and clips
    ev_rows = []      # one per timestamped event
    clip_rows = []    # one per clip, all tiers
    for t in TIERS:
        for r in by[t]:
            occ, nw = parsed[(t, r['clip_id'])]
            kept_occ = [o for o in occ if status.get(o['name']) in ('kept', 'partial') and o['kind'] != 'marker']
            dropped_occ = [o for o in occ if status.get(o['name']) == 'dropped' or o['kind'] == 'marker']
            evs = r['events']
            links, unl_ev, unl_occ = link_events(evs, occ)
            unl_kept_occ = [j for j in unl_occ if status.get(occ[j]['name']) in ('kept', 'partial')
                            and occ[j]['kind'] != 'marker']
            d = r['duration']
            wrap_all = any(o['kind'] in ('wrap', 'unclosed') and o['w_before'] == 0 and o['w_inside'] == nw and nw > 0
                           for o in kept_occ)
            kept_cats = sorted({name_cat[o['name']] for o in kept_occ if o['name'] in name_cat})
            cr = {'tier': t, 'clip_id': r['clip_id'], 'duration': d, 'n_events': len(evs),
                  'n_kept_tags': len(kept_occ), 'n_point_tags': sum(o['kind'] == 'point' for o in kept_occ),
                  'n_wrap_tags': sum(o['kind'] in ('wrap', 'unclosed') for o in kept_occ),
                  'n_dropped_tags': len(dropped_occ), 'words': nw, 'wrap_all_tag': wrap_all,
                  'noise_category': sorted(r['noise_category']), 'kept_tag_categories': kept_cats,
                  'unlinked_events': len(unl_ev), 'unlinked_kept_tags': len(unl_kept_occ),
                  'dropped_names': sorted({o['name'] for o in dropped_occ})}
            occ_of = {k: j for k, j in links}
            iv = [(e['onset'], e['offset']) for e in evs]
            whole_any = False
            for k, e in enumerate(evs):
                on, off = e['onset'], e['offset']
                dur = off - on
                o = occ[occ_of[k]] if k in occ_of else None
                overlap = any(i != k and min(off, iv[i][1]) - max(on, iv[i][0]) > 1e-6 for i in range(len(iv)))
                contained = any(i != k and iv[i][0] <= on + 1e-6 and iv[i][1] >= off - 1e-6 and
                                (iv[i][1] - iv[i][0]) > dur + 1e-6 for i in range(len(iv)))
                b = bucket(dur, d) if dur > 0 else 'degenerate'
                whole_any |= (b == 'whole')
                ev_rows.append({
                    'tier': t, 'clip_id': r['clip_id'], 'k': k, 'category': e['category'], 'tag': e['tag'],
                    'family': family(e['tag'], e['category']), 'onset': on, 'offset': off, 'dur': dur,
                    'clipdur': d, 'bucket': b,
                    'kind': (o['kind'] if o else 'unlinked'),
                    'w_before': (o['w_before'] if o else -1), 'w_inside': (o['w_inside'] if o else -1),
                    'w_total': nw,
                    'wrap_all': bool(o and o['kind'] in ('wrap', 'unclosed') and o['w_before'] == 0
                                     and o['w_inside'] == nw and nw > 0),
                    'in_dropped_wrap': bool(o and any(status.get(x) == 'dropped' for x in o['enclosed_by'])),
                    'in_kept_wrap': bool(o and any(status.get(x) in ('kept', 'partial') for x in o['enclosed_by'])),
                    'onset0': on < 0.005, 'end': off >= d - 0.01, 'd300': abs(dur - 0.3) < 5e-4,
                    'grid100': on_grid(on) and on_grid(off), 'overlap': overlap, 'contained': contained,
                    'clip_mismatch': (len(unl_ev) + len(unl_kept_occ)) > 0,
                    'clip_n_events': len(evs)})
            cr['has_whole'] = whole_any
            clip_rows.append(cr)

    # ---------------------------------------------------------------- aggregates
    res = {'tiers': {t: len(by[t]) for t in TIERS}}
    res['vocabulary'] = {
        'n_names': len(names),
        'n_kept': sum(1 for v in vocab_rows if v['status'] == 'kept'),
        'dropped': dropped, 'partial': partial,
        'bronze_only_top': bronze_only[:40],
        'multi_category_names': multi_cat, 'tier_top_category_differs': tier_diff,
        'top_kept': [v for v in vocab_rows if v['status'] == 'kept'][:80]}
    # drop rate of dropped names per tier (occurrences)
    res['dropped_occurrences_per_tier'] = {t: {v['name']: occ_count[t][v['name']] for v in dropped} for t in TIERS}
    # val events by family
    fam = {t: collections.Counter() for t in TS_TIERS}
    fam_sec = {t: collections.Counter() for t in TS_TIERS}
    for e in ev_rows:
        fam[e['tier']][e['family']] += 1
        fam_sec[e['tier']][e['family']] += e['dur']
    res['events_by_family'] = {t: {k: {'events': v, 'share': r4(v / max(1, sum(fam[t].values()))),
                                       'seconds_share': r4(fam_sec[t][k] / max(1e-9, sum(fam_sec[t].values())))}
                                   for k, v in fam[t].most_common()} for t in TS_TIERS}

    # mapping exactness per clip
    mp = {}
    for t in TS_TIERS:
        cl = [c for c in clip_rows if c['tier'] == t]
        mp[t] = {'clips': len(cl),
                 'exact': sum(1 for c in cl if c['unlinked_events'] == 0 and c['unlinked_kept_tags'] == 0),
                 'tag_without_event_clips': sum(1 for c in cl if c['unlinked_kept_tags'] > 0),
                 'tag_without_event_count': sum(c['unlinked_kept_tags'] for c in cl),
                 'event_without_tag_clips': sum(1 for c in cl if c['unlinked_events'] > 0),
                 'event_without_tag_count': sum(c['unlinked_events'] for c in cl),
                 'zero_event_clips': sum(1 for c in cl if c['n_events'] == 0),
                 'zero_event_clips_with_kept_tag': sum(1 for c in cl if c['n_events'] == 0 and c['n_kept_tags'] > 0),
                 'noisecategory_not_equal_event_categories': sum(
                     1 for c in cl if set(c['noise_category']) != {e for e in []} and False)}
    # NoiseCategory vs event categories
    evcats = collections.defaultdict(set)
    for e in ev_rows:
        evcats[(e['tier'], e['clip_id'])].add(e['category'])
    for t in TS_TIERS:
        cl = [c for c in clip_rows if c['tier'] == t]
        eq = sum(1 for c in cl if set(c['noise_category']) == evcats[(t, c['clip_id'])])
        extra_nc = sum(1 for c in cl if set(c['noise_category']) - evcats[(t, c['clip_id'])])
        missing_nc = sum(1 for c in cl if evcats[(t, c['clip_id'])] - set(c['noise_category']))
        mp[t].update({'noisecategory_equals_event_categories': eq,
                      'noisecategory_has_category_without_event': extra_nc,
                      'event_category_missing_from_noisecategory': missing_nc})
        del mp[t]['noisecategory_not_equal_event_categories']
    res['mapping_exactness'] = mp

    # ---- task 2: prevalence
    prev = {}
    for t in TIERS:
        cl = [c for c in clip_rows if c['tier'] == t]
        ev = [e for e in ev_rows if e['tier'] == t]
        durs = np.array([c['duration'] for c in cl])
        minutes = durs.sum() / 60.0
        p = {'clips': len(cl), 'hours': r4(durs.sum() / 3600), 'clip_dur_mean': r4(durs.mean()),
             'clip_dur_median': r4(np.median(durs)), 'clip_dur_p10': r4(np.percentile(durs, 10)),
             'clip_dur_p90': r4(np.percentile(durs, 90)),
             'kept_tags_per_clip': r4(np.mean([c['n_kept_tags'] for c in cl])),
             'point_tags_per_clip': r4(np.mean([c['n_point_tags'] for c in cl])),
             'wrap_tags_per_clip': r4(np.mean([c['n_wrap_tags'] for c in cl])),
             'kept_tags_per_minute': r4(sum(c['n_kept_tags'] for c in cl) / minutes),
             'clips_with_wrap_all_tag': r4(np.mean([c['wrap_all_tag'] for c in cl])),
             'clips_with_zero_kept_tags': r4(np.mean([c['n_kept_tags'] == 0 for c in cl])),
             'words_per_second': r4(sum(c['words'] for c in cl) / durs.sum())}
        if t in TS_TIERS:
            p.update({'events': len(ev), 'events_per_clip': r4(len(ev) / len(cl)),
                      'events_per_minute': r4(len(ev) / minutes),
                      'event_seconds_per_clip_second': r4(sum(e['dur'] for e in ev) / durs.sum()),
                      'clips_with_whole_clip_event': r4(np.mean([c['has_whole'] for c in cl])),
                      'bucket_share': {b: r4(np.mean([e['bucket'] == b for e in ev])) for b in BUCKETS},
                      'bucket_per_clip': {b: r4(sum(e['bucket'] == b for e in ev) / len(cl)) for b in BUCKETS},
                      'bucket_per_minute': {b: r4(sum(e['bucket'] == b for e in ev) / minutes) for b in BUCKETS},
                      'kind_share': {k: r4(np.mean([e['kind'] == k for e in ev]))
                                     for k in ['point', 'wrap', 'unclosed', 'unlinked']},
                      'wrap_empty_share': r4(np.mean([e['kind'] == 'wrap' and e['w_inside'] == 0 for e in ev])),
                      'wrap_all_share': r4(np.mean([e['wrap_all'] for e in ev])),
                      'overlap_event_share': r4(np.mean([e['overlap'] for e in ev])),
                      'contained_event_share': r4(np.mean([e['contained'] for e in ev])),
                      'clips_with_overlap': r4(np.mean([any(e['overlap'] for e in ev_c) for ev_c in
                                                        _group(ev).values()] + [False] * (len(cl) - len(_group(ev))))),
                      'onset0_share': r4(np.mean([e['onset0'] for e in ev])),
                      'end_share': r4(np.mean([e['end'] for e in ev])),
                      'd300_share': r4(np.mean([e['d300'] for e in ev])),
                      'grid100_share': r4(np.mean([e['grid100'] for e in ev])),
                      'median_event_dur': r4(np.median([e['dur'] for e in ev])),
                      'in_dropped_wrap_share': r4(np.mean([e['in_dropped_wrap'] for e in ev])),
                      'in_kept_wrap_share': r4(np.mean([e['in_kept_wrap'] for e in ev]))})
            percat = {}
            for c in CATS:
                ec = [e for e in ev if e['category'] == c]
                if not ec:
                    continue
                percat[c] = {'events': len(ec), 'share': r4(len(ec) / len(ev)),
                             'per_minute': r4(len(ec) / minutes),
                             'seconds_share': r4(sum(e['dur'] for e in ec) / max(1e-9, sum(e['dur'] for e in ev))),
                             'median_dur': r4(np.median([e['dur'] for e in ec])),
                             'bucket_share': {b: r4(np.mean([e['bucket'] == b for e in ec])) for b in BUCKETS},
                             'whole_share': r4(np.mean([e['bucket'] == 'whole' for e in ec])),
                             'point_share': r4(np.mean([e['kind'] == 'point' for e in ec])),
                             'onset0_share': r4(np.mean([e['onset0'] for e in ec])),
                             'end_share': r4(np.mean([e['end'] for e in ec])),
                             'd300_share': r4(np.mean([e['d300'] for e in ec])),
                             'grid100_share': r4(np.mean([e['grid100'] for e in ec]))}
            p['per_category'] = percat
            # whole-clip by clip-duration band
            bands = [(0, 4), (4, 6), (6, 10), (10, 1e9)]
            p['whole_clip_clips_by_duration'] = {
                '%g-%g' % (a, b if b < 1e8 else 99): {'clips': int(sum(a <= c['duration'] < b for c in cl)),
                                                    'with_whole': r4(np.mean([c['has_whole'] for c in cl if a <= c['duration'] < b]) if any(a <= c['duration'] < b for c in cl) else float('nan'))}
                for a, b in bands}
        prev[t] = p
    res['prevalence'] = prev

    # ---- transcript -> time relations (task 4 support), timestamped tiers
    rel = {}
    for t in TS_TIERS:
        ev = [e for e in ev_rows if e['tier'] == t and e['kind'] in ('wrap', 'point', 'unclosed') and e['w_total'] >= 3]
        on_err = np.array([e['onset'] - e['w_before'] / e['w_total'] * e['clipdur'] for e in ev])
        wr = [e for e in ev if e['kind'] in ('wrap', 'unclosed') and e['w_inside'] > 0]
        off_err = np.array([e['offset'] - (e['w_before'] + e['w_inside']) / e['w_total'] * e['clipdur'] for e in wr])
        wi = np.array([e['w_inside'] for e in wr], float)
        du = np.array([e['dur'] for e in wr], float)
        sec_per_word = np.array([e['clipdur'] / e['w_total'] for e in wr])
        pred_dur = wi * sec_per_word
        rel[t] = {'events_linked_w3': len(ev),
                  'onset_from_word_position_median_abs_err_s': r4(np.median(np.abs(on_err))),
                  'onset_from_word_position_p75_abs_err_s': r4(np.percentile(np.abs(on_err), 75)),
                  'wrap_events_with_words': len(wr),
                  'offset_from_word_position_median_abs_err_s': r4(np.median(np.abs(off_err))) if len(wr) else None,
                  'wrap_duration_vs_words_r': r4(np.corrcoef(wi, du)[0, 1]) if len(wr) > 2 else None,
                  'wrap_duration_vs_predicted_r': r4(np.corrcoef(pred_dur, du)[0, 1]) if len(wr) > 2 else None,
                  'wrap_duration_over_predicted_median': r4(np.median(du / np.maximum(pred_dur, 1e-3))) if len(wr) else None,
                  'wrap_all_events': int(sum(e['wrap_all'] for e in ev)),
                  'wrap_all_events_that_are_whole_clip': r4(np.mean([e['bucket'] == 'whole' for e in ev if e['wrap_all']])) if any(e['wrap_all'] for e in ev) else None,
                  'whole_clip_events_that_are_wrap_all': r4(np.mean([e['wrap_all'] for e in ev_rows if e['tier'] == t and e['bucket'] == 'whole'])),
                  'empty_wrap_median_dur': r4(np.median([e['dur'] for e in ev_rows if e['tier'] == t and e['kind'] == 'wrap' and e['w_inside'] == 0])),
                  'point_median_dur': r4(np.median([e['dur'] for e in ev_rows if e['tier'] == t and e['kind'] == 'point']))}
    res['transcript_time_relations'] = rel

    # ---- task 4: Bronze targets
    br = [c for c in clip_rows if c['tier'] == 'bronze']
    nc_vs_tags = collections.Counter()
    for c in br:
        a, b = set(c['noise_category']), set(c['kept_tag_categories'])
        nc_vs_tags['equal' if a == b else ('nc_superset' if a > b else ('nc_subset' if a < b else 'differ'))] += 1
    cnt_hist = collections.Counter(min(c['n_kept_tags'], 6) for c in br)
    pt_hist = collections.Counter(min(c['n_point_tags'], 6) for c in br)
    cat_counts = collections.Counter(x for c in br for x in c['noise_category'])
    res['bronze'] = {
        'clips': len(br), 'hours': r4(sum(c['duration'] for c in br) / 3600),
        'noisecategory_vs_tag_categories': dict(nc_vs_tags),
        'clips_with_empty_noisecategory': sum(1 for c in br if not c['noise_category']),
        'clips_per_category': dict(cat_counts.most_common()),
        'kept_tag_count_hist(6=6+)': dict(sorted(cnt_hist.items())),
        'point_tag_count_hist(6=6+)': dict(sorted(pt_hist.items())),
        'clips_with_wrap_all_tag': sum(c['wrap_all_tag'] for c in br),
        'clips_with_no_kept_tag': sum(c['n_kept_tags'] == 0 for c in br),
        'clips_with_only_point_tags': sum(c['n_kept_tags'] > 0 and c['n_wrap_tags'] == 0 for c in br),
        'clips_with_multi_category': sum(len(c['noise_category']) > 1 for c in br),
        'clips_with_dropped_wrap(static/noise/wind...)': sum(bool(c['dropped_names']) for c in br)}
    # same targets on timestamped tiers, for comparison and to validate the targets
    for t in TS_TIERS:
        cl = [c for c in clip_rows if c['tier'] == t]
        res['bronze']['compare_' + t] = {
            'kept_tag_count_hist(6=6+)': dict(sorted(collections.Counter(min(c['n_kept_tags'], 6) for c in cl).items())),
            'clips_with_wrap_all_tag_share': r4(np.mean([c['wrap_all_tag'] for c in cl])),
            'P(whole_clip_event | wrap_all_tag)': r4(np.mean([c['has_whole'] for c in cl if c['wrap_all_tag']])) if any(c['wrap_all_tag'] for c in cl) else None,
            'P(wrap_all_tag | whole_clip_event)': r4(np.mean([c['wrap_all_tag'] for c in cl if c['has_whole']])) if any(c['has_whole'] for c in cl) else None,
            'P(n_events == n_kept_tags)': r4(np.mean([c['n_events'] == c['n_kept_tags'] for c in cl])),
            'clips_with_no_kept_tag_share': r4(np.mean([c['n_kept_tags'] == 0 for c in cl]))}
    res['bronze']['compare_bronze'] = {'clips_with_wrap_all_tag_share': r4(np.mean([c['wrap_all_tag'] for c in br])),
                                       'clips_with_no_kept_tag_share': r4(np.mean([c['n_kept_tags'] == 0 for c in br]))}

    json.dump(res, open(os.path.join(OUT, 'audit_main.json'), 'w'), indent=1, ensure_ascii=False, default=str)
    # event table for the Silver-quality script
    keys = ['tier', 'clip_id', 'k', 'category', 'tag', 'family', 'onset', 'offset', 'dur', 'clipdur', 'bucket', 'kind',
            'w_before', 'w_inside', 'w_total', 'wrap_all', 'in_dropped_wrap', 'in_kept_wrap', 'onset0', 'end', 'd300',
            'grid100', 'overlap', 'contained', 'clip_mismatch', 'clip_n_events']
    np.savez_compressed(os.path.join(OUT, 'events.npz'), **{k: np.array([e[k] for e in ev_rows]) for k in keys})
    json.dump(clip_rows, open(os.path.join(OUT, 'clips.json'), 'w'), ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ['tiers', 'mapping_exactness', 'transcript_time_relations']}, indent=1))
    for t in TIERS:
        p = prev[t]
        print(t, {k: v for k, v in p.items() if not isinstance(v, dict)})
        for k in ['bucket_share', 'bucket_per_minute', 'kind_share']:
            if k in p:
                print('   ', k, p[k])


def _group(ev):
    g = collections.defaultdict(list)
    for e in ev:
        g[e['clip_id']].append(e)
    return g


if __name__ == '__main__':
    main()
