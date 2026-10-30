"""Shared loaders of audit_main.py and build_recipe.py (read-only on the data).

T1_WORK/vaani-full/manifest.jsonl is the Vaani manifest of prepare_full.py. T1_VAL_META is validationMetadata.json
of the organizer validation release (load_val reads its natural clips)."""
import json, re, os
MANIFEST = os.path.join(os.environ.get('T1_WORK', '.'), 'vaani-full', 'manifest.jsonl')
VAL_META = os.environ.get('T1_VAL_META', 'validationMetadata.json')
TIER = {'verified_timestamps': 'gold', 'unverified_timestamps': 'silver', 'no_timestamps': 'bronze'}
CATS = ('animal', 'vehicle_traffic', 'baby_child', 'singing_music', 'phone_signal_alarm',
        'appliance_machine', 'human_non_speech')
TAG_RE = re.compile(r'<(/?)([^<>]+)>|\[([^\[\]]+)\]')


def norm_id(c):
    c = os.path.basename(str(c))
    return c[:-4] if c.endswith('.wav') else c


def session_token(clip_id):
    p = norm_id(clip_id).split('_')
    return p[5] if len(p) > 5 else None


def parse_tags(transcript):
    """Return a list of tag tokens in order: ('open'|'close'|'point', name, char_pos).
    name is '<x>' for angle tags (open and close both named '<x>') and '[x]' for square tags."""
    out = []
    for m in TAG_RE.finditer(transcript or ''):
        if m.group(2) is not None:
            name = '<' + m.group(2).strip() + '>'
            out.append(('close' if m.group(1) else 'open', name, m.start(), m.end()))
        else:
            out.append(('point', '[' + m.group(3).strip() + ']', m.start(), m.end()))
    return out


def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            md = r['source']['metadata']
            evs = []
            for e in md.get('NoiseSubCategoryTimeStamp') or []:
                try:
                    evs.append({'onset': float(e['start']), 'offset': float(e['end']),
                                'category': e.get('category'), 'tag': (e.get('tag') or '').strip()})
                except Exception:
                    evs.append({'onset': None, 'offset': None, 'category': e.get('category'),
                                'tag': (e.get('tag') or '').strip(), 'bad': True})
            rows.append({'clip_id': norm_id(r['clip_id']), 'tier': TIER[r['quality']],
                         'duration': float(r['duration']), 'transcript': md.get('transcript') or '',
                         'noise_category': list(md.get('NoiseCategory') or []), 'events': evs,
                         'manifest_events': r.get('events'), 'state': md.get('state'),
                         'language': md.get('language'), 'audio_path': r['audio_path']})
    return rows


def load_val(natural_only=True):
    meta = json.load(open(VAL_META))
    out = []
    for m in meta:
        if natural_only and m.get('syntheticData'):
            continue
        evs = []
        for e in m.get('NoiseSubCategoryTimeStamp') or []:
            evs.append({'onset': float(e['start']), 'offset': float(e['end']),
                        'category': e.get('category'), 'tag': (e.get('tag') or '').strip()})
        out.append({'clip_id': norm_id(m['segmentFileName']), 'tier': 'val_synth' if m.get('syntheticData') else 'val',
                    'duration': float(m['duration']), 'transcript': m.get('transcript') or '',
                    'noise_category': list(m.get('NoiseCategory') or []), 'events': evs,
                    'state': m.get('state'), 'language': m.get('language')})
    return out
