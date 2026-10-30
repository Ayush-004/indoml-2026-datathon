"""audit_guard.py: run vote.py under an audit, and prove three facts for the output.

  1. No forbidden input. An audit hook records every file that the process opens. The run fails if a path
     matches a forbidden pattern: template / master / router / route files, Q1 / Q2 / copy files, clip
     lists made by the peak router, organizer labels, and ANY audio file (.wav / .flac / audio zips).
     The pipeline reads frames and wav-header durations only, so it cannot read an audio peak.
  2. No forbidden code. An import blocker makes the template, router, Q1 / Q2 and matched-filter modules
     impossible to import; after the run none of them is in sys.modules.
  3. No use of the file name. The same pipeline runs a second time with every clip id replaced by an opaque
     token (sha1 of the id; the token order differs from the name order). Frame stores, durations, the E0
     file and the clip list are all re-keyed. After the tokens are mapped back, the output must be
     byte-identical to the normal output. So no step reads state, district, session or any other part
     of the name, and no step depends on the clip order.

  python audit_guard.py <the vote.py arguments>      (writes <out>.guard.json next to the output)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import importlib.abc  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent

BLOCKED_MODULES = ('indoml_pipeline.templates', 'indoml_pipeline.router', 'peak_router', 'nclib', 'ncscan',
                   'sr1lib', 'templates', 'router', 'q1_val', 'q1_test', 'q2_kernels', 'j2_lib', 'lib_wmf')
FORBIDDEN = ('exemplars', 'tau_router', '/templates/', 'masters.pkl', 'routes', 'peak_router', 'router.py',
             'system_950232', 'q1_', 'q2_', 'test_copies', 'library.pkl', 'theta.json', 'kernels_cand',
             'natural-clips.txt', '_syn_clips', '_nat_clips', 'test_missing', 'validationMetadata',
             'organizer-references', '.wav', '.flac', 'input_data.zip', 'validation.zip', 'audio_test')
OPENED = []
BLOCKED_TRIES = []


def _hook(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        OPENED.append(p)


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED_MODULES or name.split('.')[-1] in ('templates', 'router', 'peak_router', 'nclib'):
            BLOCKED_TRIES.append(name)
            raise ImportError(f'audit_guard: module {name} is blocked (template / router / copy code)')
        return None


def tok(stem):
    return 'x' + hashlib.sha1(stem.encode()).hexdigest()[:24]


def main():
    sys.meta_path.insert(0, _Blocker())
    sys.addaudithook(_hook)
    sys.path.insert(0, str(HERE))
    import numpy as np
    import vote as OP

    argv = sys.argv[1:]
    sys.argv = ['vote.py'] + argv
    n0 = len(OPENED)
    OP.main()                                   # the normal run: writes out, .N3, .N4, .info.json
    opened_normal = OPENED[n0:]
    out = Path(argv[argv.index('--out') + 1])
    info = json.loads(Path(str(out).replace('.jsonl', '.info.json')).read_text())
    normal_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert normal_sha == info['sha256']

    # ---- the opaque-id run (same parameters, every id re-keyed) ----
    split = info['split']
    clips = [x.strip() for x in Path(argv[argv.index('--clips') + 1]).read_text().splitlines() if x.strip()]
    stem = OP.stem
    back = {}

    def T(c):
        s = stem(str(c))
        t = tok(s)
        back[t] = s
        return t + ('.wav' if str(c).endswith('.wav') else '')

    work = Path(os.environ.get('T1_GUARD_TMP', str(out.parent / 'guard_tmp')))
    work.mkdir(parents=True, exist_ok=True)
    durs = json.loads(OP.DURS[split].read_text())
    (work / f'{split}-durations.tok.json').write_text(json.dumps({T(k): v for k, v in durs.items()}))
    OP.DURS = dict(OP.DURS, **{split: work / f'{split}-durations.tok.json'})
    if OP.E0.get(split) is not None:                     # E0 is read only when the pairing is on (950232)
        e0_lines = []
        for line in Path(OP.E0[split]).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r['clip_id'] = T(r['clip_id'])
                e0_lines.append(json.dumps(r))
        (work / f'{split}-e0.tok.jsonl').write_text('\n'.join(e0_lines) + '\n')
        OP.E0 = dict(OP.E0, **{split: work / f'{split}-e0.tok.jsonl'})

    real_load = np.load

    class TokBlob:
        def __init__(self, blob):
            self.blob = blob

        def __getitem__(self, key):
            v = self.blob[key]
            if key == 'clip_ids':
                return np.asarray([T(c) for c in v], dtype=object)
            return v

    def tok_load(path, *a, **k):
        blob = real_load(path, *a, **k)
        return TokBlob(blob) if str(path).endswith('.npz') else blob

    donors = argv[argv.index('--fb-donors') + 1] if '--fb-donors' in argv else 'R10,R10S1,SOUP2'
    np.load = tok_load
    try:
        final, _, _ = OP.run(split, [T(c) for c in clips], info['c4'], lambda m: None,
                             fb_donors=tuple(donors.split(',')),
                             e0_on=info['e0'], e0_w=info['e0_w'], e0_thr=info['e0_thr'])
    finally:
        np.load = real_load
    mapped = {back[c]: v for c, v in final.items()}
    opaque_sha = OP.dump(mapped, work / f'{out.stem}.opaque-mapped.jsonl')

    data_opened = sorted({p for p in opened_normal if not p.endswith(('.py', '.pyc', '.so', '.dylib'))
                          and '/site-packages/' not in p and '/lib/python' not in p})
    hits = sorted({p for p in OPENED if any(f in p for f in FORBIDDEN)})
    loaded_blocked = sorted(m for m in sys.modules if m in BLOCKED_MODULES)
    report = {
        'output': str(out), 'output_sha256': normal_sha, 'lines': len(out.read_text().splitlines()),
        'vote_args': argv,
        'fb_donors': donors.split(','),
        'forbidden_patterns': list(FORBIDDEN), 'forbidden_hits': hits,
        'blocked_modules': list(BLOCKED_MODULES), 'blocked_import_attempts': BLOCKED_TRIES,
        'blocked_modules_loaded': loaded_blocked,
        'audio_files_opened': sorted(p for p in OPENED if p.endswith(('.wav', '.flac'))),
        'data_files_opened_normal_run': data_opened,
        'name_blind': {'opaque_mapped_sha256': opaque_sha, 'normal_sha256': normal_sha,
                       'identical': opaque_sha == normal_sha},
        'code_sha256': {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                        for n in ('vote.py', 'audit_guard.py')},
    }
    report['PASS'] = (not hits and not BLOCKED_TRIES and not loaded_blocked and not report['audio_files_opened']
                      and report['name_blind']['identical'])
    Path(str(out).replace('.jsonl', '.guard.json')).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('output_sha256', 'lines', 'forbidden_hits', 'blocked_import_attempts',
                                             'audio_files_opened', 'name_blind', 'PASS')}, indent=1))
    if not report['PASS']:
        raise SystemExit('AUDIT FAIL')


if __name__ == '__main__':
    main()
