"""make_zip.py: put a test prediction file into a Codabench zip and check the official format.

Official format (the organizer evaluation page): a ZIP with ONE file predictions.jsonl; one JSON object per line
{"clip_id": ..., "events": [{"onset": float, "offset": float}, ...]}; every test clip exactly once; [] for a clip
without events. Conventions of our accepted entries: clip_id = the audio stem without ".wav", lines sorted by
clip_id, json.dumps default separators.

The check reads the zip back from disk. Clip durations come from the WAV headers inside the organizer test zip
(an independent source), and are compared with the durations json given to vote.py.
This script never uploads anything.

  python make_zip.py --pred <predictions jsonl> --zip <out zip> --audio-zip input_data.zip --durations test-durations.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path

EXPECTED = 5517


def wav_duration(head):
    """Duration from a RIFF/WAVE header (fmt + data chunk sizes)."""
    assert head[:4] == b'RIFF' and head[8:12] == b'WAVE', 'not a RIFF/WAVE file'
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(head):
        cid, size = head[pos:pos + 4], struct.unpack('<I', head[pos + 4:pos + 8])[0]
        if cid == b'fmt ':
            ch, sr = struct.unpack('<HI', head[pos + 10:pos + 16])
            bits = struct.unpack('<H', head[pos + 22:pos + 24])[0]
            fmt = (ch, sr, bits)
        elif cid == b'data':
            data = size
            break
        pos += 8 + size + (size & 1)
    ch, sr, bits = fmt
    return data / (ch * (bits // 8)) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--zip', required=True)
    ap.add_argument('--audio-zip', required=True, help='the organizer test zip (input_data.zip)')
    ap.add_argument('--durations', required=True, help='the durations json given to vote.py')
    a = ap.parse_args()
    raw = Path(a.pred).read_bytes()
    pred_sha = hashlib.sha256(raw).hexdigest()

    # audio stems and header durations from the organizer test zip
    hdr = {}
    with zipfile.ZipFile(a.audio_zip) as z:
        for info in z.infolist():
            base = info.filename.split('/')[-1]
            if not base.endswith('.wav') or '__MACOSX' in info.filename or base.startswith('._'):
                continue
            with z.open(info) as f:
                hdr[base[:-4]] = wav_duration(f.read(4096))
    pipe_d = {(k[:-4] if k.endswith('.wav') else k): float(v)
              for k, v in json.loads(Path(a.durations).read_text()).items()}

    # write the zip (one member, fixed time stamp), then read it back from disk
    zinfo = zipfile.ZipInfo('predictions.jsonl', date_time=(2026, 10, 6, 0, 0, 0))
    zinfo.compress_type = zipfile.ZIP_DEFLATED
    zinfo.external_attr = 0o644 << 16
    Path(a.zip).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.zip, 'w') as z:
        z.writestr(zinfo, raw)
    with zipfile.ZipFile(a.zip) as z:
        members = z.namelist()
        back = z.read('predictions.jsonl')
        crc_ok = z.testzip() is None

    problems, ids, n_ev, empty, order_ok, style_ok = [], [], 0, 0, True, True
    text = back.decode('utf-8')
    lines = text.split('\n')
    trailing_newline = text.endswith('\n')
    if trailing_newline:
        lines = lines[:-1]
    for i, line in enumerate(lines):
        if not line.strip():
            problems.append(f'line {i + 1}: empty line')
            continue
        r = json.loads(line)
        if set(r) != {'clip_id', 'events'} or not isinstance(r['clip_id'], str) or not isinstance(r['events'], list):
            problems.append(f'line {i + 1}: bad keys or types')
            continue
        if json.dumps(r) != line:
            style_ok = False
        c = r['clip_id']
        ids.append(c)
        if c.endswith('.wav'):
            problems.append(f'{c}: .wav suffix')
        d = hdr.get(c)
        if d is None:
            problems.append(f'{c}: not a test audio stem')
            continue
        empty += not r['events']
        for e in r['events']:
            n_ev += 1
            if set(e) != {'onset', 'offset'}:
                problems.append(f'{c}: bad event keys')
                continue
            on, off = e['onset'], e['offset']
            if not (isinstance(on, (int, float)) and isinstance(off, (int, float)) and math.isfinite(on)
                    and math.isfinite(off) and 0.0 <= on < off <= d + 1e-9):
                problems.append(f'{c}: {on}..{off} outside [0, {d}]')
    order_ok = ids == sorted(ids)
    dmax = max(abs(hdr[k] - pipe_d[k]) for k in hdr if k in pipe_d)
    rep = {
        'pred': a.pred, 'pred_sha256': pred_sha, 'zip': a.zip,
        'zip_sha256': hashlib.sha256(Path(a.zip).read_bytes()).hexdigest(), 'zip_bytes': Path(a.zip).stat().st_size,
        'zip_members': members, 'zip_crc_ok': crc_ok,
        'predictions_in_zip_equal_pred_file': back == raw,
        'rows': len(lines), 'expected_rows': EXPECTED, 'unique_ids': len(set(ids)),
        'ids_equal_test_audio_stems': set(ids) == set(hdr), 'test_audio_stems': len(hdr),
        'ids_with_wav_suffix': sum(1 for x in ids if x.endswith('.wav')),
        'sorted_by_clip_id': order_ok, 'json_dumps_default_style': style_ok, 'trailing_newline': trailing_newline,
        'events': n_ev, 'empty_clips': empty,
        'events_violating_0_le_onset_lt_offset_le_duration': len(problems), 'problems_first_10': problems[:10],
        'duration_source': 'WAV headers in the organizer test zip',
        'max_abs_diff_header_vs_pipeline_durations_s': dmax,
    }
    rep['ok'] = (members == ['predictions.jsonl'] and crc_ok and back == raw and len(lines) == EXPECTED
                 and len(set(ids)) == EXPECTED and set(ids) == set(hdr) and not problems and order_ok
                 and style_ok and trailing_newline)
    Path(str(a.zip).replace('.zip', '.format.json')).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))
    if not rep['ok']:
        raise SystemExit('FORMAT CHECK FAIL')


if __name__ == '__main__':
    main()
