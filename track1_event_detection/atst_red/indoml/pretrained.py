"""Frozen AudioSet-Strong ATST-F extraction using the original PyTorch model.

The upstream label encoder defines 40 ms decision intervals [i*.04, (i+1)*.04).
Our feature timestamps are their centers: frame_offset=.02, frame_hop=.04.
These are label-grid timestamps, not a claim about the STFT receptive-field center.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn

from .data import load_audio, resolve_audio_path, validate_row
from .model import choose_device

SAMPLE_RATE = 16000
FRAME_SAMPLES = 640
FRAME_HOP = 0.04
FRAME_OFFSET = 0.02
WINDOW_SAMPLES = 160000
WINDOW_FRAMES = 250
STRIDE_FRAMES = 125
FEATURE_DIM = 768
CLASS_COUNT = 447
SOURCE_FILES = (
    "models/atstframe/ATSTF_wrapper.py", "models/atstframe/audio_transformer.py",
    "models/atstframe/transformer.py", "models/transformer_wrapper.py",
    "helpers/encode.py", "data_util/audioset_classes.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def iter_manifest(path: Path, limit: int | None = None):
    """Stream canonical training rows or minimal unlabeled inference rows."""
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if limit is not None and len(seen) >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key]
                                                for key in ("clip_id", "audio_path")):
                raise ValueError(f"{path}:{number}: clip_id and audio_path must be nonempty strings")
            duration = row.get("duration")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{path}:{number}: duration must be finite and positive")
            if "events" in row or "quality" in row:
                validate_row(row, f"{path}:{number}")
            if row["clip_id"] in seen:
                raise ValueError(f"{path}:{number}: duplicate clip_id {row['clip_id']!r}")
            seen.add(row["clip_id"])
            yield row


def window_starts(num_samples: int) -> list[int]:
    """Sample offsets aligned to both the 5 s stride and 40 ms label grid."""
    if num_samples <= 0:
        raise ValueError("Audio must have at least one sample")
    frames = (num_samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    starts = [0]
    while starts[-1] // FRAME_SAMPLES + WINDOW_FRAMES < frames:
        starts.append(starts[-1] + STRIDE_FRAMES * FRAME_SAMPLES)
    return starts


class FrozenATST:
    """Original ATST encoder and frontend, with explicit safe checkpoint loading.

    Only the encoder and optional linear head move to the requested device.
    The original torchaudio frontend runs independently on each CPU window so
    its top-dB clipping cannot couple different clips within an inference batch.
    """

    def __init__(self, source_root: str | Path, checkpoint: str | Path, device="cpu", include_logits=False):
        self.source_root = Path(source_root).resolve()
        checkpoint = Path(checkpoint).resolve()
        self.device = choose_device(str(device))
        self.include_logits = include_logits
        source_hashes = {name: file_sha256(self.source_root / name) for name in SOURCE_FILES}
        checkpoint_hash = file_sha256(checkpoint)
        sys.path.insert(0, str(self.source_root))
        module = importlib.import_module("models.atstframe.ATSTF_wrapper")
        for name in ("models.atstframe.ATSTF_wrapper", "models.atstframe.audio_transformer",
                     "models.atstframe.transformer", "models.transformer_wrapper"):
            actual = Path(sys.modules[name].__file__).resolve()
            expected = self.source_root / (name.replace(".", "/") + ".py")
            if actual != expected:
                raise ValueError(f"Imported {name} from {actual}, not the requested source root")
        self.wrapper = module.ATSTWrapper(atst_dropout=0.0)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("Expected the released ATST-F tensor state dictionary")
        head_keys = {"strong_head.weight", "strong_head.bias", "weak_head.weight", "weak_head.bias"}
        if {key for key in state if not key.startswith("model.")} != head_keys:
            raise ValueError("Unexpected ATST checkpoint keys outside the encoder and two released heads")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Checkpoint contains nonfinite tensors")
        encoder_state = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
        expected = set(self.wrapper.state_dict())
        frontend_buffers = {key for key in expected if key.startswith("atst_mel.mel_transform.")}
        missing = expected - set(encoder_state)
        unexpected = set(encoder_state) - expected
        if unexpected or not missing.issubset(frontend_buffers):
            raise ValueError(f"ATST encoder key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        loaded_missing, loaded_unexpected = self.wrapper.load_state_dict(encoder_state, strict=False)
        if set(loaded_missing) != missing or loaded_unexpected:
            raise ValueError("Unexpected ATST encoder state-loading result")
        for prefix in ("strong_head", "weak_head"):
            if state[prefix + ".weight"].shape != (CLASS_COUNT, FEATURE_DIM) or state[prefix + ".bias"].shape != (CLASS_COUNT,):
                raise ValueError(f"Unexpected shape for {prefix}; expected the 447-class strong release")
        self.strong_head = nn.Linear(FEATURE_DIM, CLASS_COUNT)
        self.strong_head.load_state_dict({"weight": state["strong_head.weight"], "bias": state["strong_head.bias"]}, strict=True)
        del state, encoder_state
        self.wrapper.requires_grad_(False).eval()
        self.wrapper.atst.to(self.device)
        self.strong_head.requires_grad_(False).eval().to(self.device)
        class_tree = ast.parse((self.source_root / "data_util/audioset_classes.py").read_text())
        self.classes = next(ast.literal_eval(node.value) for node in class_tree.body
                            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                            and target.id == "as_strong_train_classes" for target in node.targets))
        if len(self.classes) != CLASS_COUNT or len(set(self.classes)) != CLASS_COUNT:
            raise ValueError("Unexpected AudioSet-Strong class vocabulary")
        revision = subprocess.run(["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
        import torchaudio
        self.provenance = {
            "format_version": 1, "model": "ATST-F_strong_1", "checkpoint_sha256": checkpoint_hash,
            "source_files_sha256": source_hashes, "source_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "adapter_sha256": file_sha256(Path(__file__)), "sample_rate": SAMPLE_RATE,
            "frame_hop": FRAME_HOP, "frame_offset": FRAME_OFFSET, "feature_dim": FEATURE_DIM,
            "window_seconds": 10, "window_stride_seconds": 5,
            "stitching": "positive triangular weights at cell centers, normalized over overlapping windows",
            "grid": "cell i=[i*0.04,min((i+1)*0.04,duration)); frame_offset=0.02 denotes its nominal center",
            "grid_source": "helpers/encode.py ManyHotEncoder: frame_hop=160,net_pooling=4,fs=16000; onset floor,offset ceil",
            "frontend": "original ATSTMel, CPU and independently per padded 10-second window",
            "feature_dtype": "float16", "include_logits": include_logits,
            "torch_version": str(torch.__version__), "torchaudio_version": str(torchaudio.__version__),
            "device": str(self.device),
        }
        # Diagnostic logits and execution device do not change feature semantics.
        identity = {key: value for key, value in self.provenance.items() if key not in {"include_logits", "device"}}
        self.cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @torch.inference_mode()
    def encode_windows(self, windows: list[torch.Tensor]):
        if not windows or any(window.shape != (WINDOW_SAMPLES,) or window.device.type != "cpu" for window in windows):
            raise ValueError("Expected a nonempty list of exactly 160,000 CPU samples per window")
        # Per-window processing matches the upstream single-window inference;
        # AmplitudeToDB(top_db=80) can otherwise share clipping across a batch.
        spectrograms = torch.cat([self.wrapper.mel_forward(window[None]) for window in windows], dim=0)
        if spectrograms.shape != (len(windows), 1, 64, 1001):
            raise ValueError(f"Unexpected native ATST frontend shape: {spectrograms.shape}")
        features = self.wrapper(spectrograms.to(self.device))
        if features.shape != (len(windows), WINDOW_FRAMES, FEATURE_DIM) or not torch.isfinite(features).all():
            raise ValueError("Invalid ATST frame embeddings")
        logits = self.strong_head(features) if self.include_logits else None
        if logits is not None and not torch.isfinite(logits).all():
            raise ValueError("Nonfinite ATST class logits")
        return features.float().cpu().numpy(), None if logits is None else logits.float().cpu().numpy()


def read_valid_cache(path: Path, clip_id: str, audio_hash: str, cache_id: str, include_logits: bool):
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if (not isinstance(metadata, dict) or isinstance(metadata.get("num_samples"), bool)
                    or not isinstance(metadata.get("num_samples"), int) or metadata["num_samples"] <= 0
                    or metadata.get("frame_hop") != FRAME_HOP or metadata.get("frame_offset") != FRAME_OFFSET):
                return None
            frames = (metadata["num_samples"] + FRAME_SAMPLES - 1) // FRAME_SAMPLES
            if (metadata["clip_id"] != clip_id or metadata["audio_sha256"] != audio_hash
                    or metadata["cache_id"] != cache_id or metadata["num_frames"] != frames or frames <= 0):
                return None
            features = data["features"]
            if features.shape != (frames, FEATURE_DIM) or features.dtype != np.float16 or not np.isfinite(features).all():
                return None
            if include_logits:
                logits = data["logits"]
                if logits.shape != (frames, CLASS_COUNT) or logits.dtype != np.float16 or not np.isfinite(logits).all():
                    return None
            return metadata
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None


def feature_row(row: dict, audio_path: Path, feature_path: Path, output: Path, metadata: dict):
    result = dict(row)
    duration = metadata["num_samples"] / SAMPLE_RATE
    if abs(duration - float(row["duration"])) > 0.05:
        raise ValueError(f"Manifest/audio duration mismatch for {row['clip_id']}: {row['duration']} vs {duration}")
    if row.get("events") and any(float(event["offset"]) > duration + 0.02 for event in row["events"]):
        raise ValueError(f"Event extends beyond measured audio for {row['clip_id']}")
    if duration != row["duration"]:
        result["source_manifest_duration"] = row["duration"]
    result.update(audio_path=str(audio_path.resolve()), duration=duration,
                  feature_path=str(feature_path.relative_to(output)), num_frames=metadata["num_frames"],
                  frame_hop=FRAME_HOP, frame_offset=FRAME_OFFSET, feature_dim=FEATURE_DIM,
                  feature_cache_id=metadata["cache_id"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="mps", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=4, help="Number of fixed 10-second windows per encoder batch")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="TEST ONLY: extract at most this many input manifest rows")
    parser.add_argument("--include-logits", action="store_true", help="Also store the 447 AudioSet-Strong class logits")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) <= 0 or args.log_every < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("batch-size, threads and limit must be positive; log-every must be nonnegative")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    target_manifest = output / "feature_manifest.jsonl"
    info_path = output / "cache.json"
    protected = {manifest, args.checkpoint.resolve()}
    if any(path.resolve() in protected or path.with_suffix(path.suffix + ".tmp").resolve() in protected
           for path in (target_manifest, info_path)):
        raise ValueError("Cache outputs must not overwrite the input manifest or checkpoint")
    adapter = FrozenATST(args.source_root, args.checkpoint, args.device, args.include_logits)
    output.mkdir(parents=True, exist_ok=True)
    if info_path.exists() and json.loads(info_path.read_text())["cache_id"] != adapter.cache_id:
        raise ValueError("Output cache was created with a different model/configuration; use a new output directory")
    (output / "features").mkdir(exist_ok=True)
    info = {"cache_id": adapter.cache_id, "provenance": adapter.provenance,
            "source_root": str(args.source_root.resolve()), "checkpoint_path": str(args.checkpoint.resolve()),
            "source_manifest": str(manifest), "source_manifest_sha256": file_sha256(manifest),
            "classes": adapter.classes, "test_only_limit": args.limit, "complete": False}
    atomic_json(info_path, info)
    if args.limit is not None:
        print(f"TEST-ONLY LIMIT: extracting at most {args.limit} clips.", flush=True)
    counts = {"clips": 0, "reused": 0, "extracted": 0, "windows": 0}
    audio_seconds = 0.0
    peak_mps_bytes = 0
    jobs = []
    triangular = np.minimum((np.arange(WINDOW_FRAMES) + 0.5) / 125,
                            (WINDOW_FRAMES - np.arange(WINDOW_FRAMES) - 0.5) / 125).astype(np.float32)
    temporary_manifest = target_manifest.with_suffix(".jsonl.tmp")
    extraction_started = time.monotonic()
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        def emit(row):
            nonlocal audio_seconds
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            counts["clips"] += 1
            audio_seconds += row["duration"]
            if args.log_every and counts["clips"] % args.log_every == 0:
                elapsed = time.monotonic() - extraction_started
                print(json.dumps({"progress": counts, "seconds": round(elapsed, 2),
                      "clips_per_second": round(counts["clips"] / max(elapsed, 1e-9), 2)}), flush=True)

        def flush():
            nonlocal peak_mps_bytes
            if not jobs:
                return
            features, logits = adapter.encode_windows([job[2] for job in jobs])
            counts["windows"] += len(jobs)
            if adapter.device.type == "mps":
                peak_mps_bytes = max(peak_mps_bytes, torch.mps.driver_allocated_memory())
            for index, (work, start_frame, _) in enumerate(jobs):
                length = min(WINDOW_FRAMES, work["num_frames"] - start_frame)
                selection = slice(start_frame, start_frame + length)
                weight = triangular[:length]
                work["features"][selection] += features[index, :length] * weight[:, None]
                work["weights"][selection] += weight
                if logits is not None:
                    work["logits"][selection] += logits[index, :length] * weight[:, None]
                work["remaining"] -= 1
                if work["remaining"] == 0:
                    if (work["weights"] <= 0).any():
                        raise ValueError("Uncovered feature cells during window stitching")
                    arrays = {"features": (work["features"] / work["weights"][:, None]).astype(np.float16)}
                    if logits is not None:
                        arrays["logits"] = (work["logits"] / work["weights"][:, None]).astype(np.float16)
                    if any(not np.isfinite(array).all() for array in arrays.values()):
                        raise ValueError("Nonfinite float16 features after stitching")
                    metadata = {"cache_id": adapter.cache_id, "clip_id": work["row"]["clip_id"],
                                "audio_sha256": work["audio_hash"], "num_samples": work["num_samples"],
                                "num_frames": work["num_frames"], "frame_hop": FRAME_HOP, "frame_offset": FRAME_OFFSET}
                    temporary = work["path"].with_suffix(".npz.tmp")
                    with temporary.open("wb") as destination:
                        np.savez(destination, **arrays, metadata=np.asarray(json.dumps(metadata)))
                    temporary.replace(work["path"])
                    counts["extracted"] += 1
                    emit(feature_row(work["row"], work["audio_path"], work["path"], output, metadata))
            jobs.clear()

        for row in iter_manifest(manifest, args.limit):
            audio_path = Path(resolve_audio_path(row, manifest)).resolve()
            audio_hash = file_sha256(audio_path)
            path = output / "features" / (hashlib.sha256(row["clip_id"].encode()).hexdigest()[:24] + ".npz")
            if audio_path in {path, path.with_suffix(".npz.tmp"), target_manifest, temporary_manifest, info_path}:
                raise ValueError("Cache output would overwrite input audio")
            cached = read_valid_cache(path, row["clip_id"], audio_hash, adapter.cache_id, args.include_logits)
            if cached is not None:
                flush()  # Preserve input manifest order across partially cached runs.
                counts["reused"] += 1
                emit(feature_row(row, audio_path, path, output, cached))
                continue
            audio = load_audio(audio_path, sample_rate=SAMPLE_RATE)
            num_samples = len(audio)
            starts = window_starts(num_samples)
            frames = (num_samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES
            work = {"row": row, "audio_path": audio_path, "audio_hash": audio_hash, "path": path,
                    "num_samples": num_samples, "num_frames": frames, "remaining": len(starts),
                    "features": np.zeros((frames, FEATURE_DIM), dtype=np.float32), "weights": np.zeros(frames, dtype=np.float32)}
            if args.include_logits:
                work["logits"] = np.zeros((frames, CLASS_COUNT), dtype=np.float32)
            # Validate duration before spending encoder work or saving a cache file.
            feature_row(row, audio_path, path, output, {"num_samples": num_samples, "num_frames": frames, "cache_id": adapter.cache_id})
            for sample_start in starts:
                window = torch.from_numpy(audio[sample_start:sample_start + WINDOW_SAMPLES])
                window = torch.nn.functional.pad(window, (0, WINDOW_SAMPLES - window.numel()))
                jobs.append((work, sample_start // FRAME_SAMPLES, window))
                if len(jobs) == args.batch_size:
                    flush()
        flush()
    if not counts["clips"]:
        raise ValueError("Input manifest contains no clips")
    temporary_manifest.replace(target_manifest)
    elapsed = time.monotonic() - extraction_started
    info.update(complete=True, counts=counts, audio_seconds=audio_seconds,
                extraction_seconds=elapsed, total_seconds=time.monotonic() - started,
                clips_per_second=counts["clips"] / elapsed, audio_seconds_per_second=audio_seconds / elapsed,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024),
                sampled_peak_mps_driver_mib=peak_mps_bytes / 1024 ** 2)
    atomic_json(info_path, info)
    print(json.dumps({"feature_manifest": str(target_manifest), "counts": counts,
                      "seconds": round(elapsed, 3), "clips_per_second": round(counts["clips"] / elapsed, 3),
                      "peak_rss_mib": info["peak_rss_mib"], "sampled_peak_mps_driver_mib": info["sampled_peak_mps_driver_mib"]}), flush=True)


if __name__ == "__main__":
    main()
