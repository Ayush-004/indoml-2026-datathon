"""Build the data recipe files of the R10 / PSR10B training kernels (read-only on the data).

Run audit_main.py first (it writes out/audit_main.json). The four out/recipe_* files form the Kaggle input
dataset "bronze-recipe" of the kernels.

out/recipe_silver.jsonl        one row per usable Silver clip: clip_id, sampling weight, events with a
                               per-event 'mode' ('full' | 'presence' | 'ignore') and a reason
out/recipe_bronze_strong.jsonl one row per Bronze clip with a wrap-all tag: a whole-clip event of the
                               wrap-all category ('full'), channels to ignore, channels that are
                               negative everywhere
out/recipe_bronze_neg.jsonl    one row per Bronze clip: absent categories (negative everywhere) and
                               present categories (unknown position)
out/recipe_summary.json        counts, stream composition before/after, the rules

Session filter: rows whose session token (6th '_' field) occurs in the 1,806 natural validation clips
are flagged 'val_session': true. The kernels already drop them for model selection; the FINAL
submission run may keep them (0 of them share a session with a test clip).

Rules:
  Silver clip dropped   if a kept transcript tag has no event (label list incomplete)
  Silver event ignore   grid100 (both endpoints on a 100 ms grid) | wrap-all tag but event not whole-clip |
                        onset >= 1.5 s away from its word-position estimate | short WRAP (< 2 s, not whole)
  Silver event presence point tag ([breathing] ...): presence only, boundary/IoU masked, +/-80 ms band
  Silver event full     whole-clip (>= 90 % of the clip) or >= 2 s
  Silver clip weight    WHOLE_W for clips with a whole-clip event, 1.0 otherwise
"""
import collections, json, os
import numpy as np
from common_io import load_manifest, load_val, CATS
from transcript import parse, link_events

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
WHOLE_W = 0.40          # sampling weight of a Silver clip that has a whole-clip event (others 1.0)
LONG = 2.0
POS_ERR = 1.5


def main():
    main_res = json.load(open(os.path.join(OUT, 'audit_main.json')))
    dropped = {v['name'] for v in main_res['vocabulary']['dropped']}
    name_cat = {}
    for v in main_res['vocabulary']['top_kept']:
        name_cat[v['name']] = v['category']
    rows = load_manifest()
    # name -> category from all timestamped events (complete map)
    for r in rows:
        for e in r['events']:
            name_cat.setdefault(e['tag'], e['category'])
    val_sessions = {r['clip_id'].split('_')[5] for r in load_val()}

    silver_out, bronze_strong, bronze_neg = [], [], []
    cnt = collections.Counter()
    comp_before = collections.Counter()
    comp_after = collections.Counter()
    for r in rows:
        sess = r['clip_id'].split('_')[5]
        vs = sess in val_sessions
        occ, nw = parse(r['transcript'])
        D = r['duration']
        if r['tier'] == 'silver':
            # offsets past the clip end are rounding (<= 1 ms, measured): clip them, as the Kaggle kernels do
            evs = [dict(e, offset=min(e['offset'], D)) for e in r['events']]
            evs = [e for e in evs if e['category'] in CATS and e['offset'] > e['onset']]
            if not evs:
                cnt['silver_clip_no_usable_event'] += 1
                continue
            links, unl_ev, unl_occ = link_events(evs, occ)
            unl_kept = [j for j in unl_occ if occ[j]['kind'] != 'marker' and occ[j]['name'] not in dropped]
            if unl_kept:
                cnt['silver_clip_dropped_tag_without_event'] += 1
                continue
            occ_of = dict(links)
            out_ev = []
            has_whole = False
            for k, e in enumerate(evs):
                dur = e['offset'] - e['onset']
                o = occ[occ_of[k]] if k in occ_of else None
                whole = dur >= 0.9 * D
                has_whole |= whole
                grid = all(abs(x * 10 - round(x * 10)) < 1e-3 for x in (e['onset'], e['offset']))
                wrap_all = bool(o and o['kind'] in ('wrap', 'unclosed') and o['w_before'] == 0 and o['w_inside'] == nw and nw > 0)
                pos_err = None
                if o and nw >= 3 and not wrap_all:
                    pos_err = e['onset'] - o['w_before'] / nw * D
                if grid:
                    mode, why = 'ignore', 'grid100'
                elif wrap_all and not whole:
                    mode, why = 'ignore', 'wrap_all_tag_not_whole_event'
                elif pos_err is not None and abs(pos_err) >= POS_ERR:
                    mode, why = 'ignore', 'far_from_word_position'
                elif whole:
                    mode, why = 'full', 'whole_clip'
                elif dur >= LONG:
                    mode, why = 'full', 'ge2s'
                elif o and o['kind'] == 'point':
                    mode, why = 'presence', 'point_tag_short'
                else:
                    mode, why = 'ignore', 'short_wrap_style_bias'
                cnt['silver_event_' + mode] += 1
                cnt['silver_event_reason_' + why] += 1
                out_ev.append({'onset': round(e['onset'], 4), 'offset': round(e['offset'], 4), 'category': e['category'],
                               'tag': e['tag'], 'mode': mode, 'reason': why})
            w = WHOLE_W if has_whole else 1.0
            silver_out.append({'clip_id': r['clip_id'], 'duration': D, 'weight': w, 'has_whole': has_whole,
                               'val_session': vs, 'events': out_ev})
            cnt['silver_clips_kept'] += 1
            if not vs:
                comp_before['clips'] += 1
                comp_before['whole'] += has_whole
                comp_after['clips'] += w
                comp_after['whole'] += w * has_whole
                for ev in out_ev:
                    comp_before['ev_' + ev['mode']] += 1
                    comp_after['ev_' + ev['mode']] += w
                    if ev['mode'] == 'full':
                        comp_before['full_long_events'] += 1
                        comp_after['full_long_events'] += w
        elif r['tier'] == 'bronze':
            kept = [o for o in occ if o['kind'] != 'marker' and o['name'] not in dropped]
            present = sorted(set(r['noise_category']) | {name_cat[o['name']] for o in kept if o['name'] in name_cat})
            absent = [c for c in CATS if c not in present]
            bronze_neg.append({'clip_id': r['clip_id'], 'duration': D, 'val_session': vs, 'present': present,
                               'absent_negative_everywhere': absent,
                               'n_point_tags': sum(o['kind'] == 'point' for o in kept),
                               'n_kept_tags': len(kept)})
            wa = [o for o in kept if o['kind'] in ('wrap', 'unclosed') and o['w_before'] == 0 and o['w_inside'] == nw and nw > 0
                  and o['name'] in name_cat]
            if wa:
                cat = name_cat[wa[0]['name']]
                others = sorted(set(present) - {cat})      # other present categories: position unknown -> ignore
                bronze_strong.append({'clip_id': r['clip_id'], 'duration': D, 'val_session': vs, 'weight': WHOLE_W,
                                      'events': [{'onset': 0.0, 'offset': round(D, 4), 'category': cat, 'tag': wa[0]['name'],
                                                  'mode': 'full', 'reason': 'bronze_wrap_all_tag'}],
                                      'ignore_channels': others, 'negative_channels': [c for c in CATS if c not in present]})
                cnt['bronze_wrap_all_clips'] += 1
                cnt['bronze_wrap_all_clips_not_val_session'] += (not vs)
    with open(os.path.join(OUT, 'recipe_silver.jsonl'), 'w') as f:
        for x in silver_out:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    with open(os.path.join(OUT, 'recipe_bronze_strong.jsonl'), 'w') as f:
        for x in bronze_strong:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    with open(os.path.join(OUT, 'recipe_bronze_neg.jsonl'), 'w') as f:
        for x in bronze_neg:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    b = comp_before['clips']; a = comp_after['clips']
    summary = {
        'whole_clip_weight': WHOLE_W, 'long_seconds': LONG, 'pos_err_seconds': POS_ERR,
        'counts': dict(cnt),
        'silver_stream_uniform(after session filter)': {
            'clips': b, 'share_clips_with_whole_event': round(comp_before['whole'] / b, 4),
            'full_events_per_clip': round(comp_before['full_long_events'] / b, 4),
            'presence_events_per_clip': round(comp_before['ev_presence'] / b, 4),
            'ignored_events_per_clip': round(comp_before['ev_ignore'] / b, 4)},
        'silver_stream_weighted': {
            'effective_clips': round(a, 1), 'share_clips_with_whole_event': round(comp_after['whole'] / a, 4),
            'full_events_per_clip': round(comp_after['full_long_events'] / a, 4),
            'presence_events_per_clip': round(comp_after['ev_presence'] / a, 4),
            'ignored_events_per_clip': round(comp_after['ev_ignore'] / a, 4),
            'target(val re-weighted to test clip durations)_share_clips_with_whole_event': 0.121},
        'bronze_neg_clips': len(bronze_neg),
        'bronze_neg_absent_channel_count_hist': dict(collections.Counter(len(x['absent_negative_everywhere']) for x in bronze_neg)),
    }
    json.dump(summary, open(os.path.join(OUT, 'recipe_summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
