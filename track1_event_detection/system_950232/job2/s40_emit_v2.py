"""Write a v2 synthetic-side prediction file for the 2,082 synthetic-routed test clips.
    python s40_emit_v2.py RESCAN.pkl OUT.jsonl
RESCAN.pkl is an s39 result whose base was v2/base_test_job2.pkl (shipped events with the
JOB 2 extent corrections and the new master events). Format = v1 file: one JSON object per
line, clip id without .wav, events sorted, 4 decimals, clamped to [0, clip duration],
clips sorted by id."""
import sys, json, pickle
import lib
rp, outp = sys.argv[1], sys.argv[2]
R = pickle.load(open(rp, 'rb')); pred = R['pred']
tsyn = lib.test_synthetic_clips(); td = lib.test_durations()
shipped = {r['clip_id']: [(e['onset'], e['offset']) for e in r['events']]
           for r in (json.loads(l) for l in open(lib.WORK / 'test' / 'predictions.jsonl') if l.strip())}
rows = []; n_ev = 0; bad = 0
for c in sorted(tsyn):
    stem = c[:-4]; d = td[c]; ev = []
    for a, b in sorted(pred[c]):
        aa = max(0.0, round(float(a), 4)); bb = min(round(d, 4), round(float(b), 4))
        if bb > aa: ev.append((aa, bb))
        else: bad += 1
    rows.append({'clip_id': stem, 'events': [{'onset': a, 'offset': b} for a, b in ev]}); n_ev += len(ev)
with open(outp, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
same = sum(1 for r in rows if [(e['onset'], e['offset']) for e in r['events']] == sorted(shipped[r['clip_id']]))
kept = sum(1 for r in rows for e in shipped[r['clip_id']] if (e[0], e[1]) in {(x['onset'], x['offset']) for x in r['events']})
gaps = sum(1 for r in rows for x, y in zip(r['events'][:-1], r['events'][1:]) if y['onset'] < x['offset'] + 0.2 - 1e-6)
print('%s: %d clips, %d events (shipped %d) ; clips identical to shipped %d ; shipped events kept verbatim %d ; '
      'dropped degenerate %d ; gaps < 0.2 s: %d ; max events/clip %d' % (
      outp, len(rows), n_ev, sum(len(shipped[c[:-4]]) for c in tsyn),
      same, kept, bad, gaps, max(len(r['events']) for r in rows)))
