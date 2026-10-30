"""Convert probabilities at centered STFT frames into noise intervals."""

import math

import numpy as np


def decode_events(probabilities, frame_hop, duration, threshold=0.5,
                  min_duration=0.06, merge_gap=0.04, offset_threshold=None,
                  frame_offset=0.0, onset_backtrack=False):
    """Hysteresis, gap merging, then duration filtering (all times in seconds).

    Frame i is centered at frame_offset + i * frame_hop; its decision cell
    extends half a hop either side. STFT uses offset 0; ATST interval cells use
    offset half a hop. Padding never extends a prediction past the audio.
    This binary decoder emits a union timeline, not overlapping instances.
    With onset_backtrack=True, a high-threshold seed also extends left through
    contiguous frames meeting offset_threshold, symmetrically with its offset.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be a finite 1D array in [0, 1]")
    low = threshold if offset_threshold is None else offset_threshold
    parameters = [frame_hop, duration, threshold, low, min_duration, merge_gap, frame_offset]
    if not all(math.isfinite(x) for x in parameters):
        raise ValueError("Decoder parameters must be finite")
    if frame_hop <= 0 or duration < 0 or min_duration < 0 or merge_gap < 0:
        raise ValueError("Invalid duration, hop, or postprocessing times")
    if not 0 <= frame_offset < frame_hop:
        raise ValueError("frame_offset must lie in [0, frame_hop)")
    if not 0 <= low <= threshold <= 1:
        raise ValueError("Require 0 <= offset_threshold <= threshold <= 1")
    if not isinstance(onset_backtrack, bool):
        raise ValueError("onset_backtrack must be a boolean")
    runs, start = [], None
    for index, prob in enumerate(p):
        if start is None and prob >= threshold:
            start = index
            if onset_backtrack:
                while start > 0 and p[start - 1] >= low:
                    start -= 1
        elif start is not None and prob < low:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(p)))
    intervals = []
    for start, end in runs:
        on = max(0.0, frame_offset + (start - 0.5) * frame_hop)
        # The last available center owns the remaining partial frame too.
        off = duration if end == len(p) else min(duration, frame_offset + (end - 0.5) * frame_hop)
        if off <= on:
            continue
        if intervals and on - intervals[-1][1] <= merge_gap + 1e-12:
            intervals[-1][1] = off
        else:
            intervals.append([on, off])
    return [{"onset": on, "offset": off} for on, off in intervals
            if off - on + 1e-12 >= min_duration]
