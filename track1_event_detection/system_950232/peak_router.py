"""Audio-only peak router of 950232.

RULE (one number per clip, read from the waveform only):
    natural   <=>  x.max() >= 32767  or  x.min() <= -32768     (x = the int16 samples, cast to int32 first)
    synthetic <=>  otherwise
WHY: the organizers peak-normalised every natural clip to full scale. Val: 1,806 of 1,806 natural clips peak at
32767 or 32768; the 695 synthetic mixtures peak at 32441 or lower; 0 of the 8,018 val + test clips have a peak in
32442..32766. This is an artefact of how the organizers normalised the data; we disclose it and do not claim it
as novelty.

WHAT THE RULE DOES NOT READ: the file name, the name span, a header field other than the sample format, a label,
a metadata file, or another clip. The clip id is used ONLY as the key of the output row.

PITFALL: np.abs on int16 data overflows at -32768 (it stays -32768). A router that uses np.abs(x) on
int16 sends 695 test natural clips to the synthetic branch. This module never calls abs on int16 data: it casts
the samples to int32 and compares max and min.

Own RIFF/WAVE parser (standard library + numpy): the 'fmt ' chunk must say PCM (tag 1), 1 channel, 16 bits.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np

FULL_SCALE_POS = 32767
FULL_SCALE_NEG = -32768
GAP = (32442, 32766)          # the empty peak band between the two classes (information / check only)
SR = 16000


def parse_pcm16(blob: bytes):
    """RIFF/WAVE bytes -> (sample_rate, int16 numpy array). Stops on any format other than PCM16 mono."""
    if len(blob) < 12 or blob[:4] != b'RIFF' or blob[8:12] != b'WAVE':
        raise ValueError('not a RIFF/WAVE file')
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = struct.unpack('<I', blob[pos + 4:pos + 8])[0]
        body = blob[pos + 8:pos + 8 + size]
        if cid == b'fmt ':
            tag, channels, rate, _byte_rate, _align, bits = struct.unpack('<HHIIHH', body[:16])
            fmt = (tag, channels, rate, bits)
        elif cid == b'data':
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError('missing fmt or data chunk')
    tag, channels, rate, bits = fmt
    if (tag, channels, bits) != (1, 1, 16):
        raise ValueError(f'unsupported format tag={tag} channels={channels} bits={bits}')
    n = len(data) // 2
    return rate, np.frombuffer(data[:2 * n], dtype='<i2')


def peak_route(samples: np.ndarray):
    """int16 samples -> (route, max, min). The only routing decision of the release."""
    if samples.dtype != np.dtype('<i2'):
        raise TypeError(f'expected int16 samples, got {samples.dtype}')
    x = samples.astype(np.int32)                      # cast first: never abs() on int16 (audit B-4)
    mx, mn = int(x.max()), int(x.min())
    natural = mx >= FULL_SCALE_POS or mn <= FULL_SCALE_NEG
    return ('natural' if natural else 'synthetic'), mx, mn


def wav_members(archive: zipfile.ZipFile, prefix='audio/'):
    """The test wav members (as common.zip_wav_names: no __MACOSX, no '._' resource forks), sorted."""
    return sorted(n for n in archive.namelist()
                  if n.endswith('.wav') and n.startswith(prefix)
                  and '__MACOSX' not in n.split('/') and not Path(n).name.startswith('._'))


def route_zip(zip_path, keep_audio=None):
    """Route every test wav of the zip. keep_audio: optional set of clip ids whose int16 samples are returned too.
    Returns ({clip_id: {route, max, min, peak, n_samples, sample_rate, duration}}, {clip_id: int16 samples})."""
    rows, audio = {}, {}
    with zipfile.ZipFile(zip_path) as archive:
        for member in wav_members(archive):
            clip = Path(member).name[:-4]                 # key only; the route below never reads it
            rate, x = parse_pcm16(archive.read(member))
            if rate != SR:
                raise ValueError(f'{member}: sample rate {rate}')
            route, mx, mn = peak_route(x)
            peak = max(mx, -mn)                           # Python ints: no overflow
            rows[clip] = {'route': route, 'max': mx, 'min': mn, 'peak': peak, 'n_samples': int(len(x)),
                          'sample_rate': rate, 'duration': len(x) / rate}
            if keep_audio is not None and clip in keep_audio:
                audio[clip] = x.copy()
    return rows, audio


def selftest():
    """Unit checks of the rule, including the int16 overflow pitfall (B-4)."""
    z = np.zeros(10, dtype='<i2')
    a = z.copy(); a[3] = 32767
    b = z.copy(); b[3] = -32768                          # np.abs(b) would give -32768 (overflow)
    c = z.copy(); c[3] = 32441; c[4] = -32441
    d = z.copy(); d[3] = -32767
    assert peak_route(a)[0] == 'natural'
    assert peak_route(b)[0] == 'natural'
    assert int(np.abs(b)[3]) == -32768 and int(np.abs(b).max()) < FULL_SCALE_POS   # the pitfall this module avoids:
    # np.abs(int16 -32768) stays -32768, so a naive max|x| rule would call clip b synthetic
    assert peak_route(c)[0] == 'synthetic'
    assert peak_route(d)[0] == 'synthetic'               # -32767 is not full scale on the negative side
    return {'natural_at_+32767': True, 'natural_at_-32768': True, 'synthetic_at_32441': True,
            'synthetic_at_-32767': True, 'np_abs_int16_overflow_documented': True}


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(selftest()))
    if len(sys.argv) > 1:
        rows, _ = route_zip(sys.argv[1])
        n = sum(1 for r in rows.values() if r['route'] == 'natural')
        print(json.dumps({'clips': len(rows), 'natural': n, 'synthetic': len(rows) - n,
                          'peaks_in_gap': sum(1 for r in rows.values() if GAP[0] <= r['peak'] <= GAP[1])}))
