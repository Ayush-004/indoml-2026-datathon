"""Frequency-only crop/resize of native ATST mel windows using a private RNG.

This adapts the pinned CPJKU/PretrainedSED RandomResizeCrop configuration with
virtual_crop_scale=(1, 1), freq_scale=(.6, 1), time_scale=(1, 1). Positive scale
is truncated with int(), then clipped to [1, 64]; integer heights are NOT drawn
uniformly. Crops use bicubic interpolation with align_corners=True.

Torch/per-clip random choices are distribution-equivalent to that configured
crop, not the released Python/NumPy random sequence or batch-wide behavior.
The caller owns generator state, training/evaluation mode and history counters.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _layout(descriptions, windows):
    _require(isinstance(descriptions, (list, tuple)) and bool(descriptions),
             "Descriptions must be a nonempty list/tuple")
    layout, clip_ids, seen_windows = [], set(), set()
    for description in descriptions:
        _require(isinstance(description, dict) and isinstance(description.get("row"), dict),
                 "Each description needs a row with clip_id")
        clip_id = description["row"].get("clip_id")
        _require(isinstance(clip_id, str) and bool(clip_id.strip()) and clip_id not in clip_ids,
                 "Description clip IDs must be nonempty and unique")
        clip_ids.add(clip_id)
        frames = description.get("frames")
        _require(type(frames) is int and frames > 0, "Description frames must be a positive integer")
        sections = description.get("sections")
        _require(isinstance(sections, (list, tuple)) and bool(sections), "Each clip needs window sections")
        indices = []
        for section in sections:
            _require(isinstance(section, dict), "Each section must be a mapping")
            index = section.get("window")
            _require(type(index) is int and 0 <= index < windows, "Section window index is out of range")
            _require(index not in seen_windows, "Duplicate window: each input window must have exactly one clip/section owner")
            seen_windows.add(index)
            indices.append(index)
        layout.append((clip_id, frames, indices))
    _require(len(seen_windows) == windows, "Descriptions must cover every input window exactly once")
    return layout


def frequency_warp(spec, descriptions, *, generator, probability, min_scale, max_scale):
    """Return frequency-warped CPU float32 [windows, 1, 64, T] and a small audit.

    Every clip shares one crop across all of its windows. When probability is
    positive, draw one float64 Bernoulli variate per clip; an applied clip also
    draws one float64 scale, even for a degenerate scale range. Draw an inclusive
    integer offset only when the crop height is less than 64. Probability zero
    returns the SAME tensor and consumes no random values.

    Skipped decisions use low=None/height=None. An applied full-height crop uses
    low=0/height=64. Unapplied windows are bitwise preserved; spec is never mutated.
    Time coordinates, frame counts and descriptions are unchanged. Input hashes
    and cumulative history/RNG accounting belong to QA/the caller, not this path.
    """
    _require(isinstance(spec, torch.Tensor) and spec.device.type == "cpu"
             and spec.dtype == torch.float32 and spec.ndim == 4
             and spec.shape[0] > 0 and spec.shape[1] == 1 and spec.shape[2] == 64
             and spec.shape[3] > 0,
             "Expected nonempty CPU float32 spectrogram [windows, 1, 64, T]")
    _require(isinstance(generator, torch.Generator) and generator.device.type == "cpu",
             "A private CPU torch.Generator is required")
    for name, value in (("probability", probability), ("min_scale", min_scale), ("max_scale", max_scale)):
        _require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value),
                 f"{name} must be a finite number")
    _require(0 <= probability <= 1, "Probability must be in [0, 1]")
    _require(0 < min_scale <= max_scale <= 1, "Frequency scales must satisfy 0 < min_scale <= max_scale <= 1")
    windows, _, frequencies, time_frames = spec.shape
    layout = _layout(descriptions, windows)
    _require(bool(torch.isfinite(spec).all()), "Input spectrogram contains nonfinite values")
    audit = {"decisions": [], "clips": len(layout), "windows": windows,
             "applied_clips": 0, "applied_windows": 0,
             "feature_frames": sum(frames for _, frames, _ in layout),
             "applied_feature_frames": 0, "mel_time_frames": windows * time_frames,
             "applied_mel_time_frames": 0, "input_shape": list(spec.shape)}
    result = None
    for clip_id, frames, indices in layout:
        applied = probability > 0 and float(torch.rand((), dtype=torch.float64, generator=generator)) < probability
        low = height = None
        if applied:
            scale = float(torch.empty((), dtype=torch.float64).uniform_(min_scale, max_scale, generator=generator))
            height = min(frequencies, max(1, int(scale * frequencies)))
            low = int(torch.randint(0, frequencies - height + 1, (), generator=generator)) if height < frequencies else 0
            if result is None:
                result = spec.clone()
            # A full crop is already the identity; keep its bytes untouched.
            if height < frequencies:
                selected = torch.tensor(indices, dtype=torch.long)
                crop = spec.index_select(0, selected)[:, :, low:low + height, :]
                warped = F.interpolate(crop, size=(frequencies, time_frames), mode="bicubic", align_corners=True)
                result.index_copy_(0, selected, warped)
            audit["applied_clips"] += 1
            audit["applied_windows"] += len(indices)
            audit["applied_feature_frames"] += frames
            audit["applied_mel_time_frames"] += len(indices) * time_frames
        audit["decisions"].append({"clip_id": clip_id, "applied": bool(applied), "low": low, "height": height})
    if result is None:
        result = spec
    _require(bool(torch.isfinite(result).all()), "Warped spectrogram contains nonfinite values")
    return result, audit
