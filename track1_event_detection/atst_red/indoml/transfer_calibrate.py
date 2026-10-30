"""Post hoc decoder selection on the checkpoint's original development holdout."""
from __future__ import annotations

import argparse
import hashlib
import io
from itertools import product
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .data import resolve_audio_path
from .metrics import score_records
from .model import choose_device
from .train import save_checkpoint, write_json
from .transfer import (FeatureDataset, TemporalHead, collate_features, decoded_record,
                       feature_path, file_hash, load_transfer_checkpoint, probabilities)

NOTICE = ("Post hoc decoder selection on the same development holdout used during training. "
          "These are development metrics, not an independent test or competition result.")
MAX_CANDIDATES = 1000


def candidate_decoders(original, thresholds, min_durations, merge_gaps, low_thresholds=None):
    """Retain the original first, then controls and optional symmetric seeds."""
    if not thresholds or not min_durations or not merge_gaps:
        raise ValueError("Decoder grid axes must be nonempty")
    if any(not math.isfinite(value) or not 0 < value < 1 for value in thresholds):
        raise ValueError("thresholds must be finite and strictly inside (0,1)")
    if any(not math.isfinite(value) or value < 0 for value in list(min_durations) + list(merge_gaps)):
        raise ValueError("duration and gap grids must be finite and nonnegative")
    if low_thresholds is not None:
        if not low_thresholds or any(not math.isfinite(value) or not 0 <= value < 1 for value in low_thresholds):
            raise ValueError("low thresholds must be finite and inside [0,1)")
        if any(value > max(thresholds) for value in low_thresholds):
            raise ValueError("Each low threshold must be <= at least one high threshold")
    result, seen = [], set()

    def append(decoder):
        key = json.dumps(decoder, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(decoder)
            if len(result) > MAX_CANDIDATES:
                raise ValueError(f"Decoder grid exceeds {MAX_CANDIDATES} unique candidates including the original")

    if original is not None:
        append(dict(original))
    # Original single-threshold controls remain in exactly their previous order.
    for threshold, minimum, gap in product(thresholds, min_durations, merge_gaps):
        append(dict(threshold=threshold, min_duration=minimum, merge_gap=gap))
    if low_thresholds is not None:
        for threshold, low, minimum, gap in product(thresholds, low_thresholds, min_durations, merge_gaps):
            if low <= threshold:
                append(dict(threshold=threshold, offset_threshold=low, onset_backtrack=True,
                            min_duration=minimum, merge_gap=gap))
    return result


def calibrate_checkpoint(args):
    checkpoint_path, manifest = Path(args.checkpoint).resolve(), Path(args.val).resolve()
    output = Path(args.output).resolve()
    # A single open/read observes one atomically published checkpoint even if a
    # training process replaces its path. The helper performs weights-only load.
    payload = checkpoint_path.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    checkpoint = load_transfer_checkpoint(io.BytesIO(payload))
    validation_sha = file_hash(manifest)
    expected_sha = checkpoint.get("training", {}).get("manifest_sha256", {}).get("validation")
    if not expected_sha or validation_sha != expected_sha:
        raise ValueError("Validation manifest must exactly match the checkpoint's original validation SHA-256")
    dataset = FeatureDataset(manifest, mode="val", feature_dim=checkpoint["model_config"]["feature_dim"])
    if dataset.feature_cache_id != checkpoint["feature_cache_id"]:
        raise ValueError("Validation feature_cache_id differs from checkpoint")
    if checkpoint["feature_grid"] != {"frame_hop": dataset.frame_hop, "frame_offset": dataset.frame_offset}:
        raise ValueError("Validation feature grid differs from checkpoint")
    if len(dataset.rows) != len(dataset.all_rows):
        raise ValueError("Calibration must not drop validation rows")
    protected = {checkpoint_path, manifest}
    protected.update(feature_path(row, manifest) for row in dataset.rows)
    protected.update(resolve_audio_path(row, manifest).resolve() for row in dataset.rows)
    destinations = {output / name for name in ("calibrated.pt", "calibrated.pt.tmp", "calibration.json", "calibration.json.tmp")}
    if {path.resolve() for path in destinations} & protected:
        raise ValueError("Calibration output must not overwrite input checkpoint, manifest, features or audio")
    original = dict(checkpoint["decoder"])
    low_thresholds = getattr(args, "low_thresholds", None)
    decoders = candidate_decoders(original, args.thresholds, args.min_durations,
                                 args.merge_gaps, low_thresholds)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model = TemporalHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_features)
    started = time.monotonic()
    records = list(probabilities(model, loader, device))
    inference_seconds = time.monotonic() - started
    references = [row for row, _ in records]
    candidates, selected = [], None
    for decoder in decoders:
        predictions = [decoded_record(row, values, decoder) for row, values in records]
        candidate = {"decoder": decoder, "metrics": score_records(references, predictions),
                     "prediction_events": sum(len(row["events"]) for row in predictions),
                     "original_decoder": decoder == original}
        candidates.append(candidate)
        # Evaluate the original first and retain it on ties. Small numerical
        # differences never justify replacing an equally scoring original.
        if selected is None or candidate["metrics"]["combined"] > selected["metrics"]["combined"] + 1e-12:
            selected = candidate
    before, after = candidates[0]["metrics"], selected["metrics"]
    if not math.isfinite(after["combined"]) or after["combined"] < before["combined"]:
        raise ValueError("Calibration would degrade the original decoder; no outputs were written")
    provenance = {"notice": NOTICE, "source_checkpoint_sha256": source_sha,
                  "validation_manifest_sha256": validation_sha, "feature_cache_id": dataset.feature_cache_id,
                  "checkpoint_epoch": checkpoint["epoch"], "original_decoder": original,
                  "selected_decoder": selected["decoder"], "before": before, "after": after,
                  "combined_delta": after["combined"] - before["combined"],
                  "original_decoder_retained": selected["original_decoder"],
                  "tie_policy": "original first; replacement requires combined gain greater than 1e-12"}
    # This is an inference checkpoint. Keep the model weights and training
    # provenance; discard resume-only optimizer/history state if last.pt was used.
    calibrated = {key: value for key, value in checkpoint.items()
                  if key not in {"optimizer_state", "rng_state", "history", "best_checkpoint", "best_combined"}}
    calibrated.update(decoder=selected["decoder"], validation=after, calibration=provenance)
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output / "calibrated.pt", calibrated)
    grid = {"thresholds": args.thresholds, "min_durations": args.min_durations, "merge_gaps": args.merge_gaps}
    if low_thresholds is not None:
        grid.update(low_thresholds=low_thresholds, onset_backtrack=True,
                    seeded_pair_policy="low <= high", max_candidates=MAX_CANDIDATES)
    report = {**provenance, "source_checkpoint": str(checkpoint_path), "validation_manifest": str(manifest),
              "calibrated_checkpoint_sha256": file_hash(output / "calibrated.pt"),
              "validation_clips": len(dataset), "validation_rows_dropped": 0,
              "grid": grid,
              "candidates": candidates, "candidate_count": len(candidates),
              "inference_seconds": inference_seconds, "total_seconds": time.monotonic() - started,
              "device": str(device), "threads": args.threads, "batch_size": args.batch_size,
              "torch_version": str(torch.__version__)}
    write_json(output / "calibration.json", report)
    print(json.dumps({key: report[key] for key in ("notice", "checkpoint_epoch", "source_checkpoint_sha256",
          "candidate_count", "original_decoder", "selected_decoder", "before", "after", "combined_delta")}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val", required=True, help="Original cached validation feature manifest")
    parser.add_argument("--output", required=True, help="Directory for calibrated.pt and calibration.json")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.65, .8, .85, .9, .95])
    parser.add_argument("--low-thresholds", type=float, nargs="+", default=None,
                        help="Optional low thresholds for symmetric onset/offset extension around high seeds; only low<=high pairs are evaluated alongside controls")
    parser.add_argument("--min-durations", type=float, nargs="+", default=[.04, .08, .12])
    parser.add_argument("--merge-gaps", type=float, nargs="+", default=[0., .04, .08, .12])
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) <= 0:
        parser.error("batch-size and threads must be positive")
    if any(not math.isfinite(value) or not 0 < value < 1 for value in args.thresholds):
        parser.error("thresholds must be finite and strictly inside (0,1)")
    if any(not math.isfinite(value) or value < 0 for value in args.min_durations + args.merge_gaps):
        parser.error("duration and gap grids must be finite and nonnegative")
    try:
        # Bound the grid before opening checkpoints or loading validation features.
        candidate_decoders(None, args.thresholds, args.min_durations, args.merge_gaps, args.low_thresholds)
    except ValueError as exc:
        parser.error(str(exc))
    return calibrate_checkpoint(args)


if __name__ == "__main__":
    main()
