"""apply_q1.py: step N5 of the 950232 natural branch (Q1 library copies).

In the deployed build, N5 sits between the E0 pairing (N4) and the empty-clip fallback (N6):

    for every natural clip c (peak router):
        cp = nclib.copy_events(WMF scan rows of c, duration of c, theta)      # rows with S3 >= theta_k
        if cp:
            events[c] = nclib.normalise(nclib.rule_override(events[c], cp), duration of c)

A clip with a copy is never empty after N5, so N6 does not act on it, and the post rule LD1R does not act on a
Q1 clip either. vote.py has no N5. This script therefore takes the vote.py --c4 run (its final file and its .N4
file) and replaces the line of every clip with a copy by the N5 result of its N4 line. The other lines stay as
vote.py wrote them. On the test set: 87 copies in 77 natural clips.

The loop above is the deployed N5 code with its inputs turned into arguments. Inputs:
  --final, --n4    predictions.jsonl and predictions.N4.jsonl of one vote.py --c4 run
  --durations      json: clip -> wav duration in seconds (the same file as for vote.py)
  --theta          theta.json of the WMF null (../wmf/w08_null.py): theta_k per template
  --scan-stores    folders with chunk_*.npz files of the WMF production scan of the natural clips
                   (z['clips'] = wav names, z['r<j>'] = the scan rows of clip j; ncscan.py / ../wmf/w02_scan.py)
nclib reads the organizer scorer files at import (T1_OFFICIAL_SCORER_DIR); see nclib.py.

  python apply_q1.py --final nat.predictions.jsonl --n4 nat.predictions.N4.jsonl --durations test-durations.json \
      --theta ../wmf/theta.json --scan-stores scan_testnat --out nat_q1.predictions.jsonl --copies-out q1_copies.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nclib  # noqa: E402


def read_lines(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r['clip_id']] = [(float(e['onset']), float(e['offset'])) for e in r['events']]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--final', required=True)
    ap.add_argument('--n4', required=True)
    ap.add_argument('--durations', required=True)
    ap.add_argument('--theta', required=True)
    ap.add_argument('--scan-stores', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--copies-out', default=None)
    a = ap.parse_args()

    final, n4 = read_lines(a.final), read_lines(a.n4)
    assert set(final) == set(n4), 'the final and the N4 file must hold the same clips'
    raw = json.loads(Path(a.durations).read_text())
    durations = {nclib.stem(k): float(v) for k, v in raw.items()}
    nat = sorted(final)
    nat_wav = {c + '.wav' for c in nat}
    theta = np.array(json.loads(Path(a.theta).read_text())['theta'])

    scan, scan_src = {}, collections.Counter()
    for d in a.scan_stores:
        for f in sorted(Path(d).glob('chunk_*.npz')):
            with np.load(f, allow_pickle=False) as z:
                for j, cw in enumerate(z['clips']):
                    cw = str(cw)
                    if cw in nat_wav:
                        if cw in scan:
                            raise SystemExit(f'STOP N5: {cw} is in two scan stores')
                        scan[cw] = z[f'r{j}']
                        scan_src[d] += 1
    no_scan = sorted(nat_wav - set(scan))
    if no_scan:
        raise SystemExit(f'STOP N5: {len(no_scan)} natural clips have no WMF scan rows (e.g. {no_scan[:2]})')

    out, copies = dict(final), {}
    for c in nat:
        cp = nclib.copy_events(scan[c + '.wav'], durations[c], theta)
        if cp:
            copies[c] = cp
            out[c] = nclib.normalise(nclib.rule_override(n4[c], cp), durations[c])

    blob = ''.join(json.dumps({'clip_id': c, 'events': [{'onset': x, 'offset': y} for x, y in out[c]]}) + '\n'
                   for c in sorted(out))
    Path(a.out).write_text(blob)
    if a.copies_out:
        Path(a.copies_out).write_text(json.dumps(
            [{'clip': c, 'k': int(k), 'onset': round(float(on), 4), 'offset': round(float(off), 4)}
             for c in sorted(copies) for on, off, k, *_ in copies[c]], indent=1) + '\n')
    print(json.dumps({'natural_clips': len(nat), 'q1_clips': len(copies),
                      'q1_copies': sum(len(v) for v in copies.values()),
                      'lines_changed': sum(1 for c in nat if out[c] != final[c]),
                      'scan_rows_from': dict(scan_src)}, indent=1))


if __name__ == '__main__':
    main()
