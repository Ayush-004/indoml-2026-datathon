"""Positive-only clip supervision for a separate native-audio auxiliary stream.

Timestamp fields are never converted to auxiliary targets. Unlisted canonical
categories are unknown; malformed or noncanonical declared labels are errors.
The trainer owns sampler permutations, cycle reshuffling and RNG/cursor state.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .data import resolve_audio_path
from .transfer import CATEGORIES, file_hash

ORDER_VERSION = "weak-aux-order-v1"
FRAME_GRID = {"frame_hop": .04, "frame_offset": .02}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical_categories(categories):
    if categories is None:
        return ()
    _require(isinstance(categories, (list, tuple)), "Positive categories must be a list/tuple or null")
    _require(all(isinstance(category, str) and category in CATEGORIES for category in categories),
             "Positive categories must use the seven canonical category strings")
    present = set(categories)
    return tuple(category for category in CATEGORIES if category in present)


def positive_category_mask(categories, *, device=None):
    """Convert one declared category collection per clip to a bool [B, 7] mask."""
    _require(isinstance(categories, (list, tuple)), "Expected one category collection per clip")
    mask = torch.zeros((len(categories), len(CATEGORIES)), dtype=torch.bool, device=device)
    for index, declared in enumerate(categories):
        for category in _canonical_categories(declared):
            mask[index, CATEGORIES.index(category)] = True
    return mask


def weak_positive_loss(presence, lengths, positive_mask):
    """Mean -log(max real-frame presence), per category then supervised clip.

Only supervised real-frame probabilities are read. Padding and unlisted cells
have zero gradient. Empty label collections skip the clip; an entirely skipped
batch returns a differentiable zero, including when ignored cells are nonfinite.
"""
    _require(isinstance(presence, torch.Tensor) and presence.is_floating_point()
             and presence.ndim == 3 and presence.shape[-1] == len(CATEGORIES),
             "Presence must be a floating [batch, time, 7] tensor")
    batch, frames, _ = presence.shape
    _require(isinstance(lengths, torch.Tensor) and lengths.shape == (batch,)
             and lengths.dtype in (torch.int32, torch.int64),
             "Lengths must be an integral [batch] tensor")
    _require(isinstance(positive_mask, torch.Tensor) and positive_mask.dtype == torch.bool
             and positive_mask.shape == (batch, len(CATEGORIES)),
             "Positive mask must be a bool [batch, 7] tensor")
    lengths = lengths.to(device=presence.device)
    _require(bool(((lengths > 0) & (lengths <= frames)).all()), "Lengths must be within the real time axis")
    positive_mask = positive_mask.to(device=presence.device)
    pairs = positive_mask.nonzero(as_tuple=False)
    category_counts = positive_mask.sum(dim=1)
    supervised = category_counts > 0
    stats = {"supervised_clips": int(supervised.sum()),
             "skipped_clips": batch - int(supervised.sum()),
             "positive_categories": len(pairs),
             "supervised_category_frames": int(lengths[pairs[:, 0]].sum()),
             "real_frames": int(lengths.sum())}
    if not len(pairs):
        # An empty differentiable slice avoids reading ignored NaN/Inf values.
        return presence.reshape(-1)[:0].sum(), stats
    values = presence[pairs[:, 0], :, pairs[:, 1]]
    valid = torch.arange(frames, device=presence.device)[None] < lengths[pairs[:, 0], None]
    observed = values[valid]
    _require(bool(torch.isfinite(observed).all()) and bool(((observed >= 0) & (observed <= 1)).all()),
             "Supervised real-frame presence must be finite probabilities in [0, 1]")
    maxima = values.masked_fill(~valid, -torch.inf).amax(dim=1)
    terms = -maxima.clamp(min=1e-6, max=1.).log()
    sums = presence.new_zeros(batch).index_add(0, pairs[:, 0], terms)
    loss = (sums / category_counts.clamp_min(1))[supervised].mean()
    return loss, stats


def clip_order_key(clip_id, seed=46):
    _require(isinstance(clip_id, str) and bool(clip_id), "Clip ID must be a nonempty string")
    _require(type(seed) is int and seed >= 0, "Order seed must be a nonnegative integer")
    value = hashlib.sha256(f"{ORDER_VERSION}|{seed}|{clip_id}".encode("utf-8")).hexdigest()
    return value, clip_id


def _read_rows(manifest):
    payload = manifest.read_bytes()
    rows = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        _require(isinstance(row, dict), f"{manifest}:{number}: expected a JSON object")
        rows.append(row)
    _require(bool(rows), f"Empty auxiliary/exclusion manifest: {manifest}")
    return rows, hashlib.sha256(payload).hexdigest()


def _identity(row, manifest):
    clip_id = row.get("clip_id")
    _require(isinstance(clip_id, str) and bool(clip_id.strip()), "Clip ID is required")
    _require(isinstance(row.get("audio_path"), str) and bool(row["audio_path"].strip()),
             f"{clip_id}: audio_path is required")
    source = row.get("source")
    _require(isinstance(source, dict) and isinstance(source.get("metadata"), dict),
             f"{clip_id}: source metadata is required for holdout exclusion")
    metadata = source["metadata"]
    geography = [metadata.get(key) for key in ("state", "district")]
    _require(all(isinstance(value, str) and value.strip() for value in geography),
             f"{clip_id}: state and district are required for holdout exclusion")
    geography = tuple(" ".join(unicodedata.normalize("NFKC", value).split()).casefold()
                      for value in geography)
    source_row = (source.get("revision"), source.get("parquet"), source.get("row_index"))
    _require(all(isinstance(value, str) and value.strip() for value in source_row[:2])
             and type(source_row[2]) is int and source_row[2] >= 0,
             f"{clip_id}: revision, parquet and row_index are required for holdout exclusion")
    return {"clip_id": clip_id, "audio_path": str(resolve_audio_path(row, manifest).resolve()),
            "geography": geography, "source_row": source_row}


def _audio_geometry(row, audio_path):
    duration = row.get("duration")
    _require(not isinstance(duration, bool) and isinstance(duration, (int, float))
             and math.isfinite(duration) and duration > 0, "Audio duration must be finite and positive")
    declared_samples = round(duration * 16000)
    _require(declared_samples > 0, "Audio duration must span at least one native sample")
    # Header-only check; prepare_clips later checks decoded/resampled finite audio.
    try:
        info = sf.info(audio_path)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{row['clip_id']}: cannot read audio header: {audio_path}") from exc
    _require(info.frames > 0 and info.samplerate > 0 and info.channels > 0, "Empty or invalid audio header")
    native_samples = (int(info.frames) * 16000 + int(info.samplerate) - 1) // int(info.samplerate)
    frames = (native_samples + 639) // 640
    _require(abs(native_samples - declared_samples) <= 1
             and frames == (declared_samples + 639) // 640,
             f"{row['clip_id']}: audio header duration differs from native manifest geometry")
    grid = {**FRAME_GRID, "num_frames": frames}
    _require(all(key not in row or type(row[key]) in (int, float) and row[key] == value
                 for key, value in FRAME_GRID.items())
             and ("num_frames" not in row or type(row["num_frames"]) is int and row["num_frames"] == frames),
             f"{row['clip_id']}: supplied frame geometry differs from the native 40 ms grid")
    return grid, {"source_frames": int(info.frames), "sample_rate": int(info.samplerate),
                  "channels": int(info.channels), "native_samples": native_samples}


class WeakAudioClips(Dataset):
    """Raw-audio rows with positive tags, checked against INNER and OUTER identity.

    ``heldout_manifests`` must include both INNER and OUTER (distinct paths).
    Base rows have a reproducible seed-derived order. Actual batch permutations,
    pool-cycle reshuffles and resume cursor state belong to the native trainer.
    No feature file or timestamp event is read to construct a target.
    """
    def __init__(self, manifest, *, heldout_manifests, order_seed=46):
        self.manifest = Path(manifest).resolve()
        _require(type(order_seed) is int and order_seed >= 0, "Order seed must be a nonnegative integer")
        _require(isinstance(heldout_manifests, (list, tuple)), "Pass INNER and OUTER exclusion manifests")
        holdouts = [Path(path).resolve() for path in heldout_manifests]
        _require(len(set(holdouts)) >= 2, "Both distinct INNER and OUTER exclusion manifests are required")
        _require(self.manifest not in holdouts, "Auxiliary source cannot be an exclusion manifest")
        excluded = {key: set() for key in ("clip_id", "audio_path", "geography", "source_row")}
        self.heldout_manifest_sha256 = {}
        for path in dict.fromkeys(holdouts):
            rows, digest = _read_rows(path)
            self.heldout_manifest_sha256[str(path)] = digest
            for row in rows:
                for key, value in _identity(row, path).items():
                    excluded[key].add(value)
        raw, self.manifest_sha256 = _read_rows(self.manifest)
        seen = {key: set() for key in ("clip_id", "audio_path", "source_row")}
        self.rows, self.audio_headers = [], {}
        counts = Counter()
        for row in raw:
            identity = _identity(row, self.manifest)
            clip_id = identity["clip_id"]
            for key, value in identity.items():
                _require(value not in excluded[key], f"{clip_id}: auxiliary {key} overlaps held-out data")
                if key in seen:
                    _require(value not in seen[key], f"{clip_id}: duplicate auxiliary {key}")
                    seen[key].add(value)
            categories = _canonical_categories(row["source"]["metadata"].get("NoiseCategory"))
            grid, header = _audio_geometry(row, identity["audio_path"])
            self.rows.append({**row, "audio_path": identity["audio_path"], **grid,
                              "positive_categories": list(categories)})
            self.audio_headers[clip_id] = header
            counts.update(categories)
        self.rows.sort(key=lambda row: clip_order_key(row["clip_id"], order_seed))
        self.all_rows = self.rows
        self.order_seed, self.order_version = order_seed, ORDER_VERSION
        self.order_sha256 = hashlib.sha256(json.dumps([row["clip_id"] for row in self.rows],
                                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        self.supervision_counts = {"clips": len(self.rows),
                                   "supervised_clips": sum(bool(row["positive_categories"]) for row in self.rows),
                                   "skipped_clips": sum(not row["positive_categories"] for row in self.rows),
                                   "category_clip_counts": dict(counts)}
        self.frame_hop, self.frame_offset = FRAME_GRID["frame_hop"], FRAME_GRID["frame_offset"]
        protected = {str(self.manifest): self.manifest_sha256, **self.heldout_manifest_sha256}
        _require(all(file_hash(path) == digest for path, digest in protected.items()),
                 "Auxiliary/exclusion manifest changed while it was validated")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]
