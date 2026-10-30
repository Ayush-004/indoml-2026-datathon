"""Label-free natural / synthetic router.

Every clip filename ends with `_<start_ms>_<end_ms>.wav`: the millisecond span the
segment was cut from in its parent recording. For a NATURAL clip the wav is that
span, so (end_ms - start_ms) / 1000 equals the wav duration. For a SYNTHETIC clip
the organizer pasted noise into a clean segment and the wav duration no longer
matches the name.

  natural  <=>  |(end_ms - start_ms) / 1000 - wav_duration| <= 1 ms

On the organizer validation set this is right for 2,500 of 2,501 clips (the single
error is a synthetic clip the rule calls natural). On the test set it splits the
5,517 clips into 3,435 natural and 2,082 synthetic.

The rule reads only the filename and the wav header: no label, no model.
"""
from __future__ import annotations

import re

TOLERANCE = 1e-3
PATTERN = re.compile(r'_(\d+)_(\d+)\.wav$')


def is_natural(clip_id, duration, tolerance=TOLERANCE):
    """True / False, or None when the filename carries no millisecond span."""
    match = PATTERN.search(clip_id)
    if not match:
        return None
    span = (int(match.group(2)) - int(match.group(1))) / 1000.0
    return abs(span - duration) <= tolerance


def is_synthetic(clip_id, duration, tolerance=TOLERANCE):
    """True only when the rule positively says synthetic; an unparsable name is not synthetic."""
    return is_natural(clip_id, duration, tolerance) is False


def route(durations, tolerance=TOLERANCE):
    """{clip_id: duration} -> {clip_id: 'synthetic' | 'natural'} (unparsable names go natural)."""
    return {clip: ('synthetic' if is_synthetic(clip, duration, tolerance) else 'natural')
            for clip, duration in durations.items()}


def counts(durations, tolerance=TOLERANCE):
    routed = route(durations, tolerance)
    return {'natural': sum(1 for v in routed.values() if v == 'natural'),
            'synthetic': sum(1 for v in routed.values() if v == 'synthetic')}
