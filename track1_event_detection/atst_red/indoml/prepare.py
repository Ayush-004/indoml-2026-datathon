"""Prepare the Vaani parquet release and make an explicit, auditable holdout."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .data import QUALITY_TIERS, VALIDATION_TIERS, read_manifest, resolve_audio_path, validate_row

DEFAULT_DATASET = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
PREVIEW_DATASET = "PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp"
# Annotator rounding may extend a boundary by a few milliseconds. Larger errors fail.
BOUNDARY_TOLERANCE_SECONDS = 0.020


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_manifest(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def audit_rows(rows: list[dict]) -> dict:
    return {
        "clips": len(rows),
        "duration_seconds": sum(float(row["duration"]) for row in rows),
        "quality_counts": dict(sorted(Counter(row["quality"] for row in rows).items())),
        "language_counts": dict(sorted(Counter(row["language"] for row in rows).items())),
        "events": sum(len(row["events"]) for row in rows if row["events"] is not None),
        "unknown_timestamp_clips": sum(row["events"] is None for row in rows),
        "known_empty_event_clips": sum(row["events"] == [] for row in rows),
        "missing_group_id": sum(not row.get("group_id") for row in rows),
        "unique_groups": len({row["group_id"] for row in rows if row.get("group_id")}),
    }


def _events(record: dict, quality: str, duration: float, audit: dict) -> list[dict] | None:
    raw = record.get("NoiseSubCategoryTimeStamp")
    if quality == "no_timestamps":
        # These clips contain noise, but have no usable temporal supervision.
        if raw:
            audit["ignored_no_timestamp_annotation_rows"] += 1
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("NoiseSubCategoryTimeStamp must be a list or JSON list") from exc
    if not isinstance(raw, list):
        raise ValueError("Timestamped row lacks a NoiseSubCategoryTimeStamp list")
    result = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each timestamp entry must be an object")
        # The official Gold example has this status object after the event list.
        # It is metadata, and never establishes a row's annotationQuality itself.
        meaningful_keys = {key for key, value in entry.items() if value is not None}
        if (meaningful_keys == {"Verification_status"}
                and set(entry) <= {"Verification_status", "category", "tag", "start", "end"}
                and isinstance(entry["Verification_status"], str)):
            audit["verification_status_metadata_entries"] += 1
            continue
        if isinstance(entry.get("start"), bool) or isinstance(entry.get("end"), bool):
            raise ValueError("Timestamp start/end cannot be booleans")
        try:
            onset, offset = float(entry["start"]), float(entry["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid timestamp entry: numeric start/end values are required") from exc
        if not (math.isfinite(onset) and math.isfinite(offset) and onset < offset):
            raise ValueError("Invalid timestamp interval: require finite start < end")
        if onset < -BOUNDARY_TOLERANCE_SECONDS or offset > duration + BOUNDARY_TOLERANCE_SECONDS:
            raise ValueError(f"Timestamp [{onset}, {offset}] outside audio duration {duration}")
        clipped_onset, clipped_offset = max(0.0, onset), min(duration, offset)
        if clipped_onset >= clipped_offset:
            raise ValueError("Timestamp interval has no audio overlap")
        if clipped_onset != onset or clipped_offset != offset:
            audit["clipped_event_boundaries"] += 1
            audit["max_boundary_correction_seconds"] = max(
                audit["max_boundary_correction_seconds"], abs(clipped_onset - onset), abs(clipped_offset - offset)
            )
        # Individual intervals and their provenance remain separate even if they overlap.
        event = {"onset": clipped_onset, "offset": clipped_offset}
        for key in ("category", "tag"):
            if entry.get(key) is not None:
                event[key] = str(entry[key])
        result.append(event)
    return result


def _decode_audio(record: dict, parquet_path: Path) -> tuple[np.ndarray, int, str | None]:
    audio = record.get("audio")
    if isinstance(audio, dict):
        original_path = audio.get("path")
        if audio.get("bytes") is not None:
            if not isinstance(audio["bytes"], (bytes, bytearray, memoryview)):
                raise ValueError("audio.bytes must contain binary audio data")
            source = io.BytesIO(audio["bytes"])
        elif original_path:
            source = Path(original_path).expanduser()
            if not source.is_absolute():
                source = parquet_path.parent / source
        else:
            raise ValueError("audio has neither embedded bytes nor a file path")
    elif isinstance(audio, (str, Path)):
        original_path = str(audio)
        source = Path(audio).expanduser()
        if not source.is_absolute():
            source = parquet_path.parent / source
    else:
        raise ValueError("Expected parquet audio={bytes, path}, or a local audio path")
    if isinstance(source, Path):
        # Opening paths with Python keeps permission and I/O failures as OSError;
        # libsndfile would otherwise hide them inside a generic decode exception.
        with source.open("rb") as stream:
            waveform, rate = sf.read(stream, dtype="float32", always_2d=True)
    else:
        waveform, rate = sf.read(source, dtype="float32", always_2d=True)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    return waveform.mean(axis=1, dtype=np.float32), int(rate), str(original_path) if original_path else None


def _group_metadata(record: dict, group_field: str | None) -> tuple[str | None, str | None]:
    # Speaker is preferred to session: splitting sessions can leak the same speaker.
    candidates = [group_field] if group_field else [
        "group_id", "speaker_id", "speakerId", "speakerID", "speaker", "participant_id",
        "participantId", "session_id", "sessionId", "recording_session_id",
    ]
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            if isinstance(value, (list, dict)):
                raise ValueError(f"Group field {key} must be a scalar")
            return f"{key}:{value}", key
    return None, None


def _quarantine_record(record: dict, *, dataset: str, revision: str | None,
                       source_name: str, row_index: int, quality: str,
                       error: Exception) -> dict:
    audio = record.get("audio")
    audio_path = audio.get("path") if isinstance(audio, dict) else audio
    if isinstance(error, sf.LibsndfileError):
        reason = f"Audio decode failed: {error.error_string} (libsndfile code {error.code})"
    else:
        reason = str(error)
    # Include neither raw annotations, transcripts, nor embedded payloads. Error
    # messages from our validators deliberately avoid dumping source records.
    return {"dataset": dataset, "revision": revision, "parquet": source_name,
            "row_index": row_index, "audio_path": str(audio_path) if isinstance(audio_path, (str, Path)) else None,
            "quality": quality, "error_type": type(error).__name__,
            "error_reason": " ".join(reason.split())[:500]}


def prepare_parquets(
    parquets: Iterable[tuple[Path, str]], output: Path, *, dataset: str,
    revision: str | None = None, default_quality: str | None = None,
    limit: int | None = None, sample_rate: int = 16000, group_field: str | None = None,
    invalid_rows: str = "error",
) -> dict:
    import pyarrow.parquet as pq

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if invalid_rows not in {"error", "quarantine"}:
        raise ValueError("invalid_rows must be error or quarantine")
    if group_field in {"imageFileName", "image_file_name", "language", "district", "state"}:
        raise ValueError("--group-field must identify a real speaker/session, not an image prompt or demographic field")
    if default_quality is not None and default_quality not in QUALITY_TIERS:
        raise ValueError("Unsupported default quality")
    if dataset == PREVIEW_DATASET and default_quality != "preview_timestamps":
        raise ValueError("The public preview lacks verified quality labels; explicitly use --default-quality preview_timestamps")
    output = Path(output).resolve()
    audio_directory = output / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    rows, candidate_counts = [], Counter()
    input_quality_counts, quarantine_quality_counts = Counter(), Counter()
    input_rows_seen, quarantined_rows = 0, 0
    quarantine_path = output / "quarantine.jsonl"
    if invalid_rows == "quarantine":
        quarantine_path.write_text("", encoding="utf-8")
    audit = {"clipped_event_boundaries": 0, "max_boundary_correction_seconds": 0.0,
             "metadata_duration_mismatches": 0, "source_parquets": [],
             "verification_status_metadata_entries": 0, "ignored_no_timestamp_annotation_rows": 0}
    for parquet_path, source_name in parquets:
        parquet_path = Path(parquet_path)
        audit["source_parquets"].append(source_name)
        row_index = 0
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=8):
            for record in batch.to_pylist():
                source_row_index = row_index
                row_index += 1
                input_rows_seen += 1
                quality = record.get("annotationQuality") or default_quality
                quality_label = quality if isinstance(quality, str) else "missing_or_invalid_quality"
                input_quality_counts[quality_label] += 1
                row_audit = {"clipped_event_boundaries": 0, "max_boundary_correction_seconds": 0.0,
                             "metadata_duration_mismatches": 0, "verification_status_metadata_entries": 0,
                             "ignored_no_timestamp_annotation_rows": 0}
                try:
                    if not isinstance(quality, str) or quality not in QUALITY_TIERS:
                        raise ValueError("Missing/unsupported annotationQuality; supply --default-quality only when its provenance is known")
                    waveform, original_rate, original_path = _decode_audio(record, parquet_path)
                    original_duration = len(waveform) / original_rate
                    if original_rate != sample_rate:
                        divisor = math.gcd(original_rate, sample_rate)
                        waveform = resample_poly(waveform, sample_rate // divisor, original_rate // divisor)
                    duration = len(waveform) / sample_rate
                    metadata_duration = record.get("duration")
                    if metadata_duration is not None:
                        try:
                            metadata_duration = float(metadata_duration)
                        except (TypeError, ValueError) as exc:
                            raise ValueError("Source duration metadata must be numeric") from exc
                        if not math.isfinite(metadata_duration):
                            raise ValueError("Source duration metadata must be finite")
                        if abs(metadata_duration - original_duration) > .002:
                            row_audit["metadata_duration_mismatches"] += 1
                    identity = f"{dataset}\0{revision or 'local'}\0{source_name}\0{source_row_index}"
                    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
                    fallback_id = f"{dataset.split('/')[-1]}-{digest}"
                    candidate = Path(original_path).name if original_path else None
                    if candidate in (None, "", ".", ".."):
                        candidate = fallback_id
                    group_id, selected_group_field = _group_metadata(record, group_field)
                    metadata = {key: value for key, value in record.items() if key != "audio"}
                    row = {
                        "clip_id": candidate, "audio_path": f"audio/{digest}.wav",
                        "duration": duration, "events": _events(record, quality, duration, row_audit),
                        "quality": quality, "language": str(record.get("language") or "unknown"),
                        "source": {"dataset": dataset, "revision": revision, "parquet": source_name,
                                   "row_index": source_row_index, "audio_path": original_path,
                                   "original_sample_rate": original_rate, "original_duration": original_duration,
                                   "group_field": selected_group_field, "metadata": metadata},
                        "_fallback_id": fallback_id,
                    }
                    if group_id is not None:
                        row["group_id"] = group_id
                    validate_row(row, f"{source_name}:row {source_row_index}")
                except (ValueError, sf.LibsndfileError) as exc:
                    # Public libsndfile codes 1/3/4 are bad format, malformed file,
                    # and unsupported encoding. Code 2 is a system/I/O error.
                    if isinstance(exc, sf.LibsndfileError) and exc.code not in {1, 3, 4}:
                        raise
                    if invalid_rows == "error":
                        raise ValueError(f"Failed preparing {source_name}:row {source_row_index}: {exc}") from exc
                    entry = _quarantine_record(record, dataset=dataset, revision=revision,
                                               source_name=source_name, row_index=source_row_index,
                                               quality=quality_label, error=exc)
                    # Append immediately so already-discovered corrupt records
                    # remain auditable if a later operational failure stops work.
                    with quarantine_path.open("a", encoding="utf-8") as quarantine:
                        quarantine.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    quarantined_rows += 1
                    quarantine_quality_counts[quality_label] += 1
                    continue
                # Output writes and filesystem errors are never quarantineable.
                sf.write(audio_directory / f"{digest}.wav", waveform, sample_rate, subtype="FLOAT")
                rows.append(row)
                candidate_counts[candidate] += 1
                for key, value in row_audit.items():
                    if key == "max_boundary_correction_seconds":
                        audit[key] = max(audit[key], value)
                    else:
                        audit[key] += value
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break
    if not input_rows_seen:
        raise ValueError("No records found in the parquet files")
    for row in rows:
        fallback_id = row.pop("_fallback_id")
        if candidate_counts[row["clip_id"]] > 1:
            row["clip_id"] = fallback_id
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("Could not derive unique clip IDs from the source records")
    report = {"dataset": dataset, "revision": revision, "sample_rate": sample_rate,
              "default_quality": default_quality, "requested_limit": limit,
              "invalid_rows_policy": invalid_rows, "input_rows_seen": input_rows_seen,
              "input_quality_counts": dict(sorted(input_quality_counts.items())),
              "quarantined_rows": quarantined_rows,
              "quarantine_quality_counts": dict(sorted(quarantine_quality_counts.items())),
              "boundary_tolerance_seconds": BOUNDARY_TOLERANCE_SECONDS,
              **audit_rows(rows), **audit}
    _write_manifest(output / "manifest.jsonl", rows)
    _write_json(output / "audit.json", report)
    if not rows:
        raise ValueError(f"All {input_rows_seen} input rows were quarantined; inspect {quarantine_path}")
    return report


def fetch_dataset(args: argparse.Namespace) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    # Pin every download and the manifest to one repository commit, including when
    # the caller supplied a movable branch name or omitted --revision.
    revision = api.dataset_info(args.dataset, revision=args.revision).sha
    files = sorted(name for name in api.list_repo_files(args.dataset, repo_type="dataset", revision=revision)
                   if name.endswith(".parquet"))
    if not files:
        raise ValueError("Dataset repository contains no parquet files; use the parquet command for a local export")

    def downloaded():
        for name in files:
            path = hf_hub_download(args.dataset, name, repo_type="dataset", revision=revision)
            yield Path(path), name

    return prepare_parquets(downloaded(), Path(args.output), dataset=args.dataset, revision=revision,
                            default_quality=args.default_quality, limit=args.limit,
                            sample_rate=args.sample_rate, group_field=args.group_field,
                            invalid_rows=args.invalid_rows)


def split_manifest(manifest: str | Path, output: str | Path, val_fraction: float = .2,
                   seed: int = 42, allow_clip_split: bool = False) -> dict:
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1")
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError("Cannot split an empty manifest")
    missing = sum(not row.get("group_id") for row in rows)
    if missing and not allow_clip_split:
        raise ValueError(f"{missing} clips lack real speaker/session group_id. Add those IDs, or explicitly use --allow-clip-split for a smoke test with possible speaker leakage.")
    # Known groups remain intact even when the caller accepts missing group metadata.
    def unit(row):
        return "group:" + row["group_id"] if row.get("group_id") else "clip:" + row["clip_id"]

    candidates = sorted({unit(row) for row in rows if row["quality"] in VALIDATION_TIERS})
    if len(candidates) < 2:
        raise ValueError("Need at least two eligible groups (or clips) with verified/preview timestamps to split")
    random.Random(seed).shuffle(candidates)
    number_held_out = max(1, min(len(candidates) - 1, round(len(candidates) * val_fraction)))
    held_out = set(candidates[:number_held_out])
    train, validation, excluded = [], [], []
    for original in rows:
        row = dict(original)
        row["audio_path"] = os.path.relpath(resolve_audio_path(row, manifest), output)
        if unit(row) in held_out:
            if row["quality"] in VALIDATION_TIERS:
                validation.append(row)
            else:
                excluded.append(row)
        else:
            train.append(row)
    if not train or not validation:
        raise ValueError("Split produced an empty train or validation set")
    report = {
        "source_manifest": str(manifest), "seed": seed, "requested_val_fraction": val_fraction,
        "split_unit": "speaker_or_session_group" if not missing else "available_groups_and_ungrouped_clips",
        "speaker_disjoint_verified": not missing and all(
            (row.get("source", {}).get("group_field") or "") in {
                "speaker_id", "speakerId", "speakerID", "speaker", "participant_id", "participantId"
            } for row in rows
        ),
        "group_disjoint": True, "allow_clip_split": allow_clip_split,
        "warning": "Missing speaker/session IDs: this split cannot establish speaker-disjoint generalization." if missing else
                   "Group-disjoint split; speaker disjointness depends on the meaning and completeness of supplied group IDs.",
        "eligible_validation_units": len(candidates), "held_out_units": sorted(held_out),
        "validation_quality_policy": sorted(VALIDATION_TIERS),
        "train": audit_rows(train), "validation": audit_rows(validation),
        "excluded_heldout_other_tiers": audit_rows(excluded),
        "excluded_clip_ids": [row["clip_id"] for row in excluded],
    }
    _write_manifest(output / "train.jsonl", train)
    _write_manifest(output / "val.jsonl", validation)
    _write_json(output / "split-report.json", report)
    return report


def prepare_audio_directory(audio_directory: str | Path, output: str | Path,
                            id_mode: str = "filename") -> dict:
    """Create minimal unlabeled inference rows; never fabricate event labels."""
    audio_directory, output = Path(audio_directory).resolve(), Path(output).resolve()
    if not audio_directory.is_dir():
        raise ValueError(f"Audio directory does not exist: {audio_directory}")
    if id_mode not in {"stem", "filename"}:
        raise ValueError("id_mode must be stem or filename")
    files = sorted(path for path in audio_directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".wav", ".flac"})
    if not files:
        raise ValueError(f"No WAV/FLAC files found in {audio_directory}")
    rows, identifiers = [], {}
    for path in files:
        clip_id = path.stem if id_mode == "stem" else path.name
        if clip_id in identifiers:
            raise ValueError(f"Duplicate clip ID {clip_id!r}: {identifiers[clip_id]} and {path}; use the official clip-ID manifest")
        identifiers[clip_id] = str(path)
        info = sf.info(path)
        if info.samplerate <= 0 or info.frames <= 0:
            raise ValueError(f"Empty or invalid audio: {path}")
        rows.append({"clip_id": clip_id, "audio_path": os.path.relpath(path, output.parent),
                     "duration": info.frames / info.samplerate})
    _write_manifest(output, rows)
    return {"manifest": str(output), "clips": len(rows),
            "duration_seconds": sum(row["duration"] for row in rows), "id_mode": id_mode,
            "warning": "Match clip_id to the official evaluation manifest. Filename/stem inference is only valid when official naming agrees."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download and prepare a Hugging Face parquet dataset")
    fetch.add_argument("--dataset", default=DEFAULT_DATASET)
    fetch.add_argument("--revision", default=None, help="Commit, tag or branch; resolved commit is recorded")
    local = commands.add_parser("parquet", help="Prepare one or more local parquet files")
    local.add_argument("--parquet", nargs="+", required=True, type=Path)
    local.add_argument("--dataset", default="local-parquet", help="Source dataset identifier for provenance")
    for command in (fetch, local):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--limit", type=int)
        command.add_argument("--sample-rate", type=int, default=16000)
        command.add_argument("--default-quality", choices=sorted(QUALITY_TIERS), default=None,
                             help="Only for sources missing annotationQuality; preview needs preview_timestamps")
        command.add_argument("--group-field", help="Actual speaker/session metadata field; never an image prompt ID")
        command.add_argument("--invalid-rows", choices=["error", "quarantine"], default="error",
                             help="Opt in to excluding corrupt records with an audit; I/O/write errors always stop")
    split = commands.add_parser("split", help="Hold out groups using verified/preview labels only")
    split.add_argument("--manifest", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--val-fraction", default=.2, type=float)
    split.add_argument("--seed", default=42, type=int)
    split.add_argument("--allow-clip-split", action="store_true",
                       help="Accept possible speaker leakage for ungrouped clips (e.g. preview smoke test)")
    audio = commands.add_parser("audio-dir", help="Make an unlabeled inference manifest from WAV/FLAC files")
    audio.add_argument("--audio-dir", required=True, type=Path)
    audio.add_argument("--output", required=True, type=Path, help="Output JSONL path (not a directory)")
    audio.add_argument("--id-mode", choices=["filename", "stem"], default="filename",
                       help="Use official IDs when supplied; filename/stem must agree with official naming")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            report = fetch_dataset(args)
        elif args.command == "parquet":
            report = prepare_parquets(((path.resolve(), str(path.resolve())) for path in args.parquet),
                                      args.output, dataset=args.dataset, default_quality=args.default_quality,
                                      limit=args.limit, sample_rate=args.sample_rate, group_field=args.group_field,
                                      invalid_rows=args.invalid_rows)
        elif args.command == "split":
            report = split_manifest(args.manifest, args.output, args.val_fraction, args.seed, args.allow_clip_split)
        else:
            report = prepare_audio_directory(args.audio_dir, args.output, args.id_mode)
    except Exception as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
