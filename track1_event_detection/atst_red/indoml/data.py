"""Small, dependency-light audio and JSONL manifest utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

QUALITY_TIERS = frozenset(
    {"verified_timestamps", "unverified_timestamps", "no_timestamps", "preview_timestamps"}
)
VALIDATION_TIERS = frozenset({"verified_timestamps", "preview_timestamps"})


def validate_row(row: dict, context: str = "manifest row") -> None:
    """Validate the canonical schema without silently repairing labels."""
    for name in ("clip_id", "audio_path", "quality", "language"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"{context}: {name} must be a nonempty string")
    quality = row["quality"]
    if quality not in QUALITY_TIERS:
        raise ValueError(f"{context}: unsupported quality {quality!r}")
    duration = row.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(f"{context}: duration must be a positive JSON number")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{context}: duration must be finite and positive")
    if "events" not in row:
        raise ValueError(f"{context}: events is required (null for no_timestamps)")
    events = row["events"]
    if quality == "no_timestamps":
        if events is not None:
            raise ValueError(f"{context}: no_timestamps must have events=null, not negative labels")
    elif not isinstance(events, list):
        raise ValueError(f"{context}: timestamped data must have an events list")
    for event in events or []:
        if not isinstance(event, dict):
            raise ValueError(f"{context}: each event must be an object")
        onset, offset = event.get("onset"), event.get("offset")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (onset, offset)):
            raise ValueError(f"{context}: events need JSON numbers for onset and offset")
        if not (math.isfinite(onset) and math.isfinite(offset) and 0 <= onset < offset <= duration):
            raise ValueError(f"{context}: invalid event [{onset}, {offset}] for duration {duration}")
    if row.get("group_id") is not None and (
        not isinstance(row["group_id"], str) or not row["group_id"].strip()
    ):
        raise ValueError(f"{context}: group_id must be a nonempty string when present")


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    rows, seen = [], set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            context = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{context}: expected an object")
            validate_row(row, context)
            if row["clip_id"] in seen:
                raise ValueError(f"{context}: duplicate clip_id {row['clip_id']!r}")
            seen.add(row["clip_id"])
            rows.append(row)
    return rows


def resolve_audio_path(row: dict, manifest_path: str | Path) -> Path:
    path = Path(row["audio_path"]).expanduser()
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode audio to finite float32 mono samples at the requested sample rate."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    waveform, original_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    waveform = waveform.mean(axis=1, dtype=np.float32)
    if original_rate != sample_rate:
        divisor = math.gcd(int(original_rate), int(sample_rate))
        waveform = resample_poly(waveform, sample_rate // divisor, original_rate // divisor)
    return np.asarray(waveform, dtype=np.float32)
