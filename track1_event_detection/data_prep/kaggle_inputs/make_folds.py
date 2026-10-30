"""Write folds.json: a deterministic grouped 5-fold split of the 1,806 organizer natural validation clips.

The training kernels read only the clip ids of folds.json (clip_fold), to drop every training row that shares a
recording session with a natural validation clip. The functions below are unchanged from the fine-tuning script
that wrote the file.

  python make_folds.py --val-meta validation/validationMetadata.json --out folds.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

FOLDS = 5


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=False) + '\n')
    temporary.replace(path)


def natural_records(val_meta):
    records = [r for r in json.loads(Path(val_meta).read_text()) if r['syntheticData'] is False]
    if len(records) != 1806:
        raise ValueError(f'Expected 1,806 natural clips, found {len(records)}')
    return records


def group_key(record, level):
    if level == 'state':
        return record['state']
    return record['state'] + '|' + record['district']


def build_folds(records, folds=FOLDS, level='state_district'):
    """Deterministic grouped split: whole groups, largest first, into the lightest fold."""
    members = {}
    for record in records:
        members.setdefault(group_key(record, level), []).append(record['segmentFileName'])
    order = sorted(members, key=lambda key: (-len(members[key]), key))
    assignment, sizes = {}, [0] * folds
    groups = {}
    for key in order:
        fold = min(range(folds), key=lambda index: (sizes[index], index))
        groups[key] = fold
        sizes[fold] += len(members[key])
        for clip in members[key]:
            assignment[clip] = fold
    events = Counter()
    for record in records:
        events[assignment[record['segmentFileName']]] += len(record['NoiseSubCategoryTimeStamp'])
    return {'version': 'natural-grouped-5fold-v1', 'folds': folds, 'group_level': level,
            'rule': 'Whole (state, district) groups, largest first, always into the currently smallest fold; '
                    'ties by fold index. No randomness, no clip split across folds.',
            'groups': len(members), 'clip_fold': assignment, 'group_fold': groups,
            'clips_per_fold': sizes, 'events_per_fold': [events[i] for i in range(folds)],
            'states_per_fold': [sorted({r['state'] for r in records
                                        if assignment[r['segmentFileName']] == i}) for i in range(folds)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-meta', required=True, help='validationMetadata.json of the organizer validation release')
    parser.add_argument('--out', required=True)
    parser.add_argument('--group-level', default='state_district', choices=('state', 'state_district'))
    args = parser.parse_args()
    folds = build_folds(natural_records(args.val_meta), FOLDS, args.group_level)
    write_json(args.out, folds)
    print(json.dumps({'folds': args.out, 'clips_per_fold': folds['clips_per_fold'],
                      'events_per_fold': folds['events_per_fold'], 'groups': folds['groups']}, indent=2))


if __name__ == '__main__':
    main()
