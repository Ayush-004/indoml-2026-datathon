"""Speech-bed and noise-event banks for Track 2 enhancer fine-tuning.

Sources: the public Vaani Noise-Event training release only (local manifest of 90,630 rows, revision be488e2a;
manifest.jsonl with clip_id, audio_path, duration, quality, events, language, source.metadata.transcript).
Leakage rule: every clip whose recording-session id (field 5 of the clip id) occurs in the organizer validation
set (695 synthetic + 1,806 natural) is dropped from both banks, because the 695 validation pairs are our only
unbiased SI-SDR / dWER measurement. No test audio, no test metadata, no validation audio is used.

speech_beds.jsonl : event-free spans of Gold (verified, pad 0.10 s) and Silver (unverified, pad 0.25 s) clips,
                    length >= MIN_BED_S, cut into pieces of at most MAX_BED_S. Target speech keeps its own background
                    (the organizer clean references are Vaani recordings too).
noise_bank.jsonl  : (a) paired tags <x> ... </x> whose transcript wrap holds no word (noise in a pause),
                    matched to the timed events of the clip in order (only clips where the tag sequences agree);
                    (b) point tags [x] (breathing, inhaling, cough, ...), timed events of Gold and Silver.
BANKS.json        : counts, hours, tag histogram, dropped sessions.

usage: build_banks.py --manifest VAANI/manifest.jsonl --val-meta VALIDATION/validationMetadata.json --out OUT
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

MANIFEST = VALMETA = OUT = None    # set from the command line (see usage)
MIN_BED_S = {'verified_timestamps': 1.5, 'unverified_timestamps': 2.0}
PAD_S = {'verified_timestamps': 0.10, 'unverified_timestamps': 0.25}
MAX_BED_S = 8.0
PAIR = re.compile(r"<([^<>/]+)>(.*?)</\1>", re.S)


def session(cid):
    p = str(cid).split('_')
    return p[5] if len(p) > 5 else None


def n_words(s):
    s = re.sub(r"</?[^<>]*>|\[[^\[\]]*\]|\{[^{}]*\}", " ", s)
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return len(s.split())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val = json.loads(VALMETA.read_text())
    val_sessions = {session(r['segmentFileName']) for r in val}
    val_sessions.discard(None)
    rows = [json.loads(l) for l in open(MANIFEST)]
    beds, noise, noise_words = [], [], []
    census = collections.Counter()
    tag_hist = collections.Counter()
    for r in rows:
        q = r['quality']
        if q == 'no_timestamps':
            continue
        cid, dur = r['clip_id'], float(r['duration'])
        if session(cid) in val_sessions:
            census[f'dropped_val_session_{q}'] += 1
            continue
        evs = sorted(r['events'] or [], key=lambda e: e['onset'])
        # speech beds
        pad = PAD_S[q]
        iv = sorted((max(0.0, e['onset'] - pad), min(dur, e['offset'] + pad)) for e in evs)
        t, gaps = 0.0, []
        for a, b in iv:
            if a > t:
                gaps.append((t, a))
            t = max(t, b)
        if dur > t:
            gaps.append((t, dur))
        for a, b in gaps:
            if b - a < MIN_BED_S[q]:
                continue
            n = max(1, int((b - a) // MAX_BED_S) + (1 if (b - a) % MAX_BED_S > MIN_BED_S[q] else 0))
            step = (b - a) / n
            for i in range(n):
                s0, s1 = a + i * step, min(b, a + (i + 1) * step)
                beds.append({'clip_id': cid, 'start': round(s0, 4), 'end': round(s1, 4), 'tier': q[:3],
                             'language': r.get('language')})
        # noise events
        transcript = (r['source']['metadata'].get('transcript') or '')
        pairs = PAIR.findall(transcript)
        paired_evs = [e for e in evs if e['tag'].startswith('<')]
        aligned = [e['tag'].strip('<>').strip() for e in paired_evs] == [p[0].strip() for p in pairs]
        if aligned:
            census['clips_tags_aligned'] += 1
            for e, p in zip(paired_evs, pairs):
                if n_words(p[1]) == 0 and e['offset'] - e['onset'] >= 0.05:
                    noise.append({'clip_id': cid, 'start': e['onset'], 'end': e['offset'], 'tag': e['tag'],
                                  'category': e['category'], 'kind': 'empty_wrap', 'tier': q[:3]})
                    tag_hist[e['tag']] += 1
        # (c) all paired-tag events with their words (not read by the training scripts); words_inside = -1 when unknown
        wlist = [n_words(p_[1]) for p_ in pairs] if aligned else None
        for j, e in enumerate(paired_evs):
            d = e['offset'] - e['onset']
            if 0.1 <= d <= 10.0:
                w = wlist[j] if aligned else -1
                if w != 0:          # word-free spans are already in (a)
                    noise_words.append({'clip_id': cid, 'start': e['onset'], 'end': e['offset'], 'tag': e['tag'],
                                        'category': e['category'], 'kind': 'with_words' if w > 0 else 'unknown_words',
                                        'words_inside': w, 'tier': q[:3]})
        for e in evs:
            if e['tag'].startswith('[') and e['offset'] - e['onset'] >= 0.05:
                noise.append({'clip_id': cid, 'start': e['onset'], 'end': e['offset'], 'tag': e['tag'],
                              'category': e['category'], 'kind': 'point_tag', 'tier': q[:3]})
                tag_hist[e['tag']] += 1
    with open(OUT / 'speech_beds.jsonl', 'w') as f:
        for b in beds:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')
    with open(OUT / 'noise_bank.jsonl', 'w') as f:
        for n in noise:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
    with open(OUT / 'noise_bank_words.jsonl', 'w') as f:
        for n in noise_words:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
    hours = lambda xs: round(sum(x['end'] - x['start'] for x in xs) / 3600, 3)  # noqa: E731
    by = lambda xs, key: {k: {'n': len(v), 'hours': hours(v)}  # noqa: E731
                          for k, v in collections.defaultdict(list, {kk: [x for x in xs if x[key] == kk]
                                                                      for kk in {x[key] for x in xs}}).items()}
    summary = {
        'source_manifest': str(MANIFEST), 'validation_sessions': len(val_sessions), 'census': dict(census),
        'speech_beds': {'n': len(beds), 'hours': hours(beds), 'by_tier': by(beds, 'tier')},
        'noise_bank': {'n': len(noise), 'hours': hours(noise), 'by_kind': by(noise, 'kind'), 'by_tier': by(noise, 'tier'),
                       'top_tags': tag_hist.most_common(40)},
        'noise_bank_words': {'n': len(noise_words), 'hours': hours(noise_words), 'by_kind': by(noise_words, 'kind'),
                             'by_category': by(noise_words, 'category')},
        'rules': {'min_bed_s': MIN_BED_S, 'pad_s': PAD_S, 'max_bed_s': MAX_BED_S},
    }
    (OUT / 'BANKS.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'noise_bank'}, ensure_ascii=False)[:1500])
    print(json.dumps(summary['noise_bank'], ensure_ascii=False)[:1500])
    print(json.dumps(summary['noise_bank_words'], ensure_ascii=False)[:2500])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True, help='manifest.jsonl of the local Vaani copy')
    ap.add_argument('--val-meta', type=Path, required=True, help='validationMetadata.json of the organizer validation set')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    MANIFEST, VALMETA, OUT = a.manifest, a.val_meta, a.out
    main()
