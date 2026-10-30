"""Transcript parser and event linker (used by audit_main.py and build_recipe.py).

A transcript holds words, angle tags that wrap words (<bird sound> ... </bird sound>),
unpaired angle markers (<PAUSE>, <UNKNOWN SEGMENT>), square point tags ([breathing]) and
curly glosses ({Cement}, an English gloss of the word before it; not a spoken word).
Tags can be misnested (<music> a <noise> b </music> c </noise>), so wraps are paired PER NAME
(last unclosed opening of the same name).

link_events() pairs the k-th transcript occurrence of a tag name with the k-th event of that
tag in list order (the organizer event list follows the transcript order of the openings).
"""
import re

TOKEN_RE = re.compile(r'<[^<>]*>|\[[^\[\]]*\]|\{[^{}]*\}|[^\s<>\[\]{}]+')
WORDCHAR_RE = re.compile(r'\w', re.UNICODE)
MARKERS = {'<PAUSE>', '<UNKNOWN SEGMENT>'}


def parse(transcript):
    """Return (occurrences, total_words).

    occurrence = dict(name, kind in {'wrap', 'point', 'unclosed', 'marker'}, w_before, w_inside,
                      open_index (order of the opening among all tag openings), depth_names)
    """
    items = []   # sequence of ('word',) or ('open', name) / ('close', name) / ('point', name) / ('marker', name)
    for m in TOKEN_RE.finditer(transcript or ''):
        tok = m.group(0)
        if tok.startswith('<'):
            inner = tok[1:-1].strip()
            if inner.startswith('/'):
                items.append(('close', '<' + inner[1:].strip() + '>'))
            else:
                name = '<' + inner + '>'
                items.append(('marker', name) if name in MARKERS else ('open', name))
        elif tok.startswith('['):
            items.append(('point', '[' + tok[1:-1].strip() + ']'))
        elif tok.startswith('{'):
            continue
        elif WORDCHAR_RE.search(tok):
            items.append(('word', None))
    words = 0
    occ = []
    stacks = {}
    open_now = []          # indices into occ of wraps currently open (any name)
    for it in items:
        kind, name = it
        if kind == 'word':
            words += 1
            continue
        if kind == 'open':
            o = {'name': name, 'kind': 'wrap', 'w_before': words, 'w_inside': None,
                 'enclosed_by': [occ[j]['name'] for j in open_now]}
            occ.append(o)
            stacks.setdefault(name, []).append(len(occ) - 1)
            open_now.append(len(occ) - 1)
        elif kind == 'close':
            st = stacks.get(name)
            if st:
                j = st.pop()
                occ[j]['w_inside'] = words - occ[j]['w_before']
                if j in open_now:
                    open_now.remove(j)
            # a stray closing tag without an opening is ignored
        elif kind in ('point', 'marker'):
            occ.append({'name': name, 'kind': kind, 'w_before': words, 'w_inside': 0,
                        'enclosed_by': [occ[j]['name'] for j in open_now]})
    for o in occ:
        if o['kind'] == 'wrap' and o['w_inside'] is None:
            o['kind'] = 'unclosed'
            o['w_inside'] = words - o['w_before']
    for i, o in enumerate(occ):
        o['order'] = i
    return occ, words


def link_events(events, occ):
    """Return (links, unmatched_events, unmatched_occurrences).

    links: list of (event_index, occurrence_index)."""
    by_name = {}
    for i, o in enumerate(occ):
        by_name.setdefault(o['name'], []).append(i)
    used = {}
    links, unmatched_ev = [], []
    for k, e in enumerate(events):
        name = e.get('tag') or ''
        lst = by_name.get(name, [])
        u = used.get(name, 0)
        if u < len(lst):
            links.append((k, lst[u]))
            used[name] = u + 1
        else:
            unmatched_ev.append(k)
    linked_occ = {j for _, j in links}
    return links, unmatched_ev, [j for j in range(len(occ)) if j not in linked_occ]
