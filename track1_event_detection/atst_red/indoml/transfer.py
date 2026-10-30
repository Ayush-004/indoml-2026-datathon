"""Train and apply seven temporal category heads on frozen cached embeddings."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import resolve_audio_path, validate_row
from .metrics import score_records
from .model import choose_device, seed_everything
from .postprocess import decode_events
from .train import restore_rng, rng_state, save_checkpoint, write_json

CATEGORIES = ("animal", "vehicle_traffic", "baby_child", "singing_music",
              "phone_signal_alarm", "appliance_machine", "human_non_speech")
MODEL_TYPE = "frozen_seven_category_temporal_head"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_path(row, manifest):
    path = Path(row["feature_path"]).expanduser()
    return path.resolve() if path.is_absolute() else (Path(manifest).parent / path).resolve()


def category_targets(row):
    centers = np.arange(row["num_frames"], dtype=np.float64) * row["frame_hop"] + row["frame_offset"]
    targets = np.zeros((len(centers), len(CATEGORIES)), dtype=np.float32)
    for event in row.get("events") or []:
        targets[(centers >= event["onset"]) & (centers < event["offset"]), CATEGORIES.index(event["category"])] = 1
    return torch.from_numpy(targets)


class FeatureDataset(Dataset):
    def __init__(self, manifest, *, feature_dim=768, mode="predict", tiers=("verified_timestamps",),
                 silver_positive_weight=.3, silver_negative_weight=.05):
        self.manifest = Path(manifest).resolve()
        self.feature_dim, self.mode = feature_dim, mode
        self.rows, self.all_rows, self.excluded = [], [], []
        self.silver_positive_weight, self.silver_negative_weight = silver_positive_weight, silver_negative_weight
        seen, grids, cache_ids = set(), set(), set()
        with self.manifest.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                context = f"{self.manifest}:{number}"
                if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                       for key in ("clip_id", "audio_path", "feature_path", "feature_cache_id")):
                    raise ValueError(f"{context}: clip_id, audio_path, feature_path and feature_cache_id are required")
                if row["clip_id"] in seen:
                    raise ValueError(f"{context}: duplicate clip_id")
                seen.add(row["clip_id"])
                cache_ids.add(row["feature_cache_id"])
                if row.get("feature_dim", feature_dim) != feature_dim:
                    raise ValueError(f"{context}: feature_dim differs from the head input dimension")
                for key in ("duration", "frame_hop", "frame_offset"):
                    value = row.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"{context}: {key} must be a finite JSON number")
                if row["duration"] <= 0 or row["frame_hop"] <= 0 or not 0 <= row["frame_offset"] < row["frame_hop"]:
                    raise ValueError(f"{context}: invalid duration or frame grid")
                frames = row.get("num_frames")
                if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
                    raise ValueError(f"{context}: num_frames must be a positive integer")
                if frames != math.ceil(row["duration"] / row["frame_hop"] - 1e-9):
                    raise ValueError(f"{context}: num_frames must describe cropped, complete decision-cell coverage")
                if not feature_path(row, self.manifest).is_file():
                    raise ValueError(f"{context}: feature file is missing")
                grids.add((float(row["frame_hop"]), float(row["frame_offset"])))
                if mode != "predict":
                    validate_row(row, context)
                    if any(event.get("category") not in CATEGORIES for event in row["events"] or []):
                        raise ValueError(f"{context}: timed events require one of the seven canonical categories")
                self.all_rows.append(row)
                if mode == "val" and (row["quality"] != "verified_timestamps" or row["events"] is None):
                    raise ValueError("Validation must contain Gold timestamps; validation rows are never dropped")
                reason = None
                if mode == "train":
                    if row["quality"] == "no_timestamps" or row["events"] is None or row["quality"] not in tiers:
                        reason = "tier_not_selected_or_unknown_timestamps"
                    elif row["quality"] == "verified_timestamps":
                        declared = set(row.get("source", {}).get("metadata", {}).get("NoiseCategory") or [])
                        if declared - {event["category"] for event in row["events"]}:
                            reason = "gold_positive_category_without_timestamp"
                if reason:
                    self.excluded.append({"clip_id": row["clip_id"], "quality": row["quality"], "reason": reason})
                else:
                    self.rows.append(row)
        if not self.rows or len(grids) != 1 or len(cache_ids) != 1:
            raise ValueError("Feature manifest must retain rows on one consistent temporal grid and feature_cache_id")
        self.frame_hop, self.frame_offset = next(iter(grids))
        self.feature_cache_id = next(iter(cache_ids))

    def __len__(self):
        return len(self.rows)

    def tier_weights(self, row):
        return (self.silver_positive_weight, self.silver_negative_weight) if row.get("quality") == "unverified_timestamps" else (1., 1.)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(feature_path(row, self.manifest), allow_pickle=False) as archive:
            if "features" not in archive:
                raise ValueError(f"{row['clip_id']}: NPZ needs a features array")
            if "metadata" in archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if not isinstance(metadata, dict):
                    raise ValueError(f"{row['clip_id']}: NPZ metadata must be a JSON object")
                for key, expected in (("cache_id", self.feature_cache_id), ("clip_id", row["clip_id"]),
                                      ("num_frames", row["num_frames"]), ("frame_hop", row["frame_hop"]), ("frame_offset", row["frame_offset"])):
                    if key in metadata and metadata[key] != expected:
                        raise ValueError(f"{row['clip_id']}: NPZ metadata {key} differs from feature manifest")
            features = np.asarray(archive["features"], dtype=np.float32)
        if features.shape != (row["num_frames"], self.feature_dim) or not np.isfinite(features).all():
            raise ValueError(f"{row['clip_id']}: features must be finite [{row['num_frames']},{self.feature_dim}]")
        targets = category_targets(row) if self.mode != "predict" else torch.zeros(row["num_frames"], len(CATEGORIES))
        return {"features": torch.from_numpy(features), "targets": targets,
                "weights": self.tier_weights(row), "row": row}

    def positive_weights(self, cap):
        positive, negative = np.zeros(len(CATEGORIES)), np.zeros(len(CATEGORIES))
        events_per_class, unmapped = Counter(), Counter()
        for row in self.rows:
            targets = category_targets(row).numpy()
            centers = np.arange(row["num_frames"], dtype=np.float64) * row["frame_hop"] + row["frame_offset"]
            for event in row["events"]:
                events_per_class[event["category"]] += 1
                unmapped[event["category"]] += int(not ((centers >= event["onset"]) & (centers < event["offset"])).any())
            pos, neg = self.tier_weights(row)
            positive += targets.sum(axis=0) * pos
            negative += (len(targets) - targets.sum(axis=0)) * neg
        raw = negative / np.maximum(positive, 1.)
        weights = np.clip(raw, 1., cap).astype(np.float32)
        return torch.from_numpy(weights), {category: {"weighted_positive_frames": float(positive[i]),
                "weighted_negative_frames": float(negative[i]), "raw_ratio": float(raw[i]),
                "timestamped_events": events_per_class[category], "events_without_positive_cell": unmapped[category],
                "positive_weight": float(weights[i])} for i, category in enumerate(CATEGORIES)}


def collate_features(clips):
    lengths = torch.tensor([len(clip["features"]) for clip in clips], dtype=torch.long)
    return {"features": nn.utils.rnn.pad_sequence([clip["features"] for clip in clips], batch_first=True),
            "targets": nn.utils.rnn.pad_sequence([clip["targets"] for clip in clips], batch_first=True),
            "mask": torch.arange(int(lengths.max()))[None] < lengths[:, None], "lengths": lengths,
            "weights": torch.tensor([clip["weights"] for clip in clips]), "rows": [clip["row"] for clip in clips]}


class TemporalHead(nn.Module):
    def __init__(self, feature_dim=768, projection_dim=256, hidden_size=128, dropout=.2, head_type="gru"):
        super().__init__()
        if head_type not in {"gru", "linear"}:
            raise ValueError("head_type must be gru or linear")
        self.head_type = head_type
        self.config = dict(feature_dim=feature_dim, projection_dim=projection_dim, hidden_size=hidden_size,
                           dropout=dropout, head_type=head_type)
        if head_type == "linear":
            self.project = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout))
            self.output = nn.Linear(feature_dim, len(CATEGORIES))
            return
        self.project = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, projection_dim), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(projection_dim, hidden_size, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * hidden_size, len(CATEGORIES))

    def forward(self, features, lengths):
        encoded = self.project(features)
        if self.head_type == "linear":
            return self.output(encoded)
        packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=features.shape[1])
        return self.output(encoded)


def masked_loss(logits, batch, positive_weights, device):
    targets, mask = batch["targets"].to(device), batch["mask"].to(device)
    tiers = batch["weights"].to(device)
    weights = torch.where(targets > 0, tiers[:, None, 0:1], tiers[:, None, 1:2])
    losses = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weights, reduction="none")
    denominator = mask.sum().clamp_min(1) * len(CATEGORIES)
    return (losses * weights * mask[:, :, None]).sum() / denominator, int(denominator)


def check_separate_splits(train, val):
    def geography(row):
        metadata = row.get("source", {}).get("metadata", {})
        values = [metadata.get(key) for key in ("state", "district")]
        if all(isinstance(value, str) and value.strip() for value in values):
            return tuple(" ".join(unicodedata.normalize("NFKC", value).split()).casefold() for value in values)
    checks = {
        "clip IDs": lambda row, dataset: row["clip_id"],
        "audio paths": lambda row, dataset: str(resolve_audio_path(row, dataset.manifest).resolve()),
        "feature paths": lambda row, dataset: str(feature_path(row, dataset.manifest)),
        "source audio filenames": lambda row, dataset: Path(row.get("source", {}).get("audio_path")).name if row.get("source", {}).get("audio_path") else None,
        "group IDs": lambda row, dataset: row.get("group_id"),
        "known state/district groups": lambda row, dataset: geography(row),
    }
    for name, key in checks.items():
        common = ({key(row, train) for row in train.all_rows} - {None}) & ({key(row, val) for row in val.all_rows} - {None})
        if common:
            raise ValueError(f"Train/validation leakage: overlapping {name}: {sorted(map(str, common))[:5]}")


def decoded_record(row, probabilities, decoder):
    options = dict(decoder)
    category_thresholds = options.pop("category_thresholds", None)
    if category_thresholds is not None:
        if (not isinstance(category_thresholds, dict) or set(category_thresholds) != set(CATEGORIES)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or not 0 < value < 1 for value in category_thresholds.values())):
            raise ValueError("category_thresholds must map all seven categories to finite thresholds inside (0,1)")
    events = []
    for index in range(len(CATEGORIES)):
        class_options = options if category_thresholds is None else {**options, "threshold": category_thresholds[CATEGORIES[index]]}
        events.extend(decode_events(probabilities[:, index], row["frame_hop"], row["duration"],
                                    frame_offset=row["frame_offset"], **class_options))
    # No cross-category union or deduplication: simultaneous instances must survive.
    return {"clip_id": row["clip_id"], "duration": row["duration"],
            "events": sorted(events, key=lambda event: (event["onset"], event["offset"]))}


@torch.inference_mode()
def probabilities(model, loader, device):
    model.eval()
    for batch in loader:
        values = model(batch["features"].to(device), batch["lengths"]).sigmoid().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite model probabilities")
        for index, row in enumerate(batch["rows"]):
            yield row, values[index, :row["num_frames"]]


def calibrate(records, thresholds, min_duration, merge_gap):
    references = [row for row, _ in records]
    candidates = []
    for threshold in thresholds:
        decoder = dict(threshold=threshold, min_duration=min_duration, merge_gap=merge_gap)
        predicted = [decoded_record(row, values, decoder) for row, values in records]
        candidates.append((score_records(references, predicted), decoder))
    best = max(candidates, key=lambda item: (item[0]["combined"], -abs(item[1]["threshold"] - .5), -item[1]["threshold"]))
    return *best, [{"decoder": decoder, "metrics": metrics} for metrics, decoder in candidates]


def write_predictions(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_transfer_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format_version") != 1 or checkpoint.get("model_type") != MODEL_TYPE or checkpoint.get("categories") != list(CATEGORIES)
            or not isinstance(checkpoint.get("feature_cache_id"), str) or not checkpoint["feature_cache_id"]):
        raise ValueError("Checkpoint is not a compatible frozen seven-category head")
    return checkpoint


def train(args):
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    resume = load_transfer_checkpoint(args.resume) if args.resume else None
    initialize_path = getattr(args, "initialize", None)
    if resume and initialize_path:
        raise ValueError("Choose either resume or weights-only initialization")
    initialize = load_transfer_checkpoint(initialize_path) if initialize_path else None
    source = resume or initialize
    config = source["model_config"] if source else dict(feature_dim=768, projection_dim=args.projection_dim,
                                                       hidden_size=args.hidden_size, dropout=args.dropout,
                                                       head_type=args.head_type)
    training = FeatureDataset(args.train, feature_dim=config["feature_dim"], mode="train", tiers=args.tiers,
                              silver_positive_weight=args.silver_positive_weight, silver_negative_weight=args.silver_negative_weight)
    validation = FeatureDataset(args.val, feature_dim=config["feature_dim"], mode="val")
    check_separate_splits(training, validation)
    grid = {"frame_hop": training.frame_hop, "frame_offset": training.frame_offset}
    if grid != {"frame_hop": validation.frame_hop, "frame_offset": validation.frame_offset}:
        raise ValueError("Training and validation feature grids differ")
    if training.feature_cache_id != validation.feature_cache_id:
        raise ValueError("Training and validation feature_cache_id values differ")
    if initialize and (initialize["feature_cache_id"] != training.feature_cache_id or initialize["feature_grid"] != grid):
        raise ValueError("Initialization requires the same pretrained feature cache and temporal grid")
    if initialize and initialize["training"]["manifest_sha256"]["validation"] != file_hash(validation.manifest):
        raise ValueError("Initialization requires the same validation manifest to preserve the calibration boundary")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise ValueError("Output already contains checkpoints; choose a new directory or use --resume")
    if resume:
        for name in ("best.pt", "last.pt"):
            if (output / name).exists():
                load_transfer_checkpoint(output / name)  # Never overwrite a baseline/other model's checkpoints.
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=collate_features, generator=generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate_features)
    model = TemporalHead(**config).to(device)
    if initialize:
        model.load_state_dict(initialize["model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positive_weights, class_audit = training.positive_weights(args.pos_weight_cap)
    positive_weights = positive_weights.to(device)
    policy = {"tiers": sorted(args.tiers), "silver_positive_weight": args.silver_positive_weight,
              "silver_negative_weight": args.silver_negative_weight, "positive_weight_cap": args.pos_weight_cap}
    metadata = {"arguments": vars(args), "seed": args.seed, "device": str(device), "feature_grid": grid,
                "feature_cache_id": training.feature_cache_id,
                "train_clips": len(training), "validation_clips": len(validation),
                "retained_train_quality_counts": dict(Counter(row["quality"] for row in training.rows)),
                "excluded_training_rows": training.excluded, "validation_rows_dropped": 0,
                "training_policy": policy, "class_weight_audit": class_audit,
                "events_without_positive_cell": sum(value["events_without_positive_cell"] for value in class_audit.values()),
                "loss_normalization": "unweighted_valid_frames_times_seven_classes",
                "label_grid": "positive_when_cell_center_is_inside_half_open_event_interval",
                "manifest_sha256": {"train": file_hash(training.manifest), "validation": file_hash(validation.manifest)},
                "parameters": sum(value.numel() for value in model.parameters()), "torch_version": str(torch.__version__)}
    if initialize:
        metadata["initialization"] = {
            "checkpoint": str(Path(initialize_path).resolve()), "checkpoint_sha256": file_hash(initialize_path),
            "epoch": initialize["epoch"], "source_training": initialize["training"],
            "policy": "model weights only; fresh optimizer, random state, epoch count and validation selection",
            "source_model_config_used": config,
        }
    elif resume and "initialization" in resume["training"]:
        metadata["initialization"] = resume["training"]["initialization"]
    history, best_score, first_epoch, best_checkpoint = [], float("-inf"), 0, None
    if resume:
        if ("optimizer_state" not in resume or metadata["manifest_sha256"] != resume["training"]["manifest_sha256"]
                or policy != resume["training"]["training_policy"] or resume["feature_cache_id"] != training.feature_cache_id):
            raise ValueError("Resume requires last.pt with unchanged manifests and supervision policy")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        restore_rng(resume["rng_state"], generator, device)
        history, best_score, first_epoch = resume["history"], resume["best_combined"], resume["epoch"]
        best_checkpoint = resume["best_checkpoint"]
        metadata["seed"] = resume["training"]["seed"]
        save_checkpoint(output / "best.pt", best_checkpoint)
        if Path(args.resume).resolve() != output / "last.pt":
            save_checkpoint(output / "last.pt", resume)
    metadata["effective_learning_rate"] = optimizer.param_groups[0]["lr"]
    write_json(output / "training-audit.json", metadata)
    if first_epoch >= args.epochs:
        write_json(output / "history.json", {"training": metadata, "epochs": history, "best_combined": best_score})
    print(json.dumps({"setup": {key: metadata[key] for key in ("train_clips", "validation_clips", "parameters", "class_weight_audit", "training_policy")},
                      "excluded_training_clips": len(training.excluded)}), flush=True)
    for epoch in range(first_epoch + 1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        total_loss = total_cells = 0.
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, cells = masked_loss(model(batch["features"].to(device), batch["lengths"]), batch, positive_weights, device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * cells
            total_cells += cells
            if args.log_every and step % args.log_every == 0:
                print(json.dumps({"progress": {"epoch": epoch, "step": step, "steps": len(loader),
                      "loss": total_loss / total_cells, "seconds": round(time.monotonic() - started, 2)}}), flush=True)
        records = list(probabilities(model, val_loader, device))
        metrics, decoder, calibration = calibrate(records, args.thresholds, args.min_duration, args.merge_gap)
        entry = {"epoch": epoch, "train_loss": total_loss / total_cells, "validation": metrics,
                 "decoder": decoder, "calibration": calibration, "seconds": round(time.monotonic() - started, 3)}
        history.append(entry)
        checkpoint = {"format_version": 1, "model_type": MODEL_TYPE, "model_config": config,
                      "categories": list(CATEGORIES), "feature_grid": grid, "decoder": decoder,
                      "feature_cache_id": training.feature_cache_id,
                      "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                      "epoch": epoch, "validation": metrics, "training": metadata}
        if metrics["combined"] > best_score:
            best_score, best_checkpoint = metrics["combined"], checkpoint
            save_checkpoint(output / "best.pt", checkpoint)
            write_predictions(output / "validation-predictions.jsonl", (decoded_record(row, values, decoder) for row, values in records))
        save_checkpoint(output / "last.pt", {**checkpoint, "optimizer_state": optimizer.state_dict(),
                        "rng_state": rng_state(generator, device), "history": history,
                        "best_combined": best_score, "best_checkpoint": best_checkpoint})
        write_json(output / "history.json", {"training": metadata, "epochs": history, "best_combined": best_score})
        print(json.dumps({key: entry[key] for key in ("epoch", "train_loss", "validation", "decoder", "seconds")}), flush=True)
    print(f"Best frozen-feature checkpoint: {output / 'best.pt'}", flush=True)


def predict(args):
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    checkpoint = load_transfer_checkpoint(args.checkpoint)
    dataset = FeatureDataset(args.manifest, feature_dim=checkpoint["model_config"]["feature_dim"])
    if checkpoint["feature_grid"] != {"frame_hop": dataset.frame_hop, "frame_offset": dataset.frame_offset}:
        raise ValueError("Prediction feature grid differs from checkpoint")
    if checkpoint["feature_cache_id"] != dataset.feature_cache_id:
        raise ValueError("Prediction feature_cache_id differs from checkpoint")
    output = Path(args.output).resolve()
    protected = {Path(args.checkpoint).resolve(), dataset.manifest}
    protected.update(feature_path(row, dataset.manifest) for row in dataset.rows)
    protected.update(resolve_audio_path(row, dataset.manifest).resolve() for row in dataset.rows)
    if output in protected:
        raise ValueError("Prediction output must not overwrite input data or checkpoint")
    model = TemporalHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    decoder = dict(checkpoint["decoder"])
    if args.threshold is not None:
        decoder.pop("category_thresholds", None)
        decoder["threshold"] = args.threshold
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_features)
    write_predictions(output, (decoded_record(row, values, decoder) for row, values in probabilities(model, loader, device)))
    print(f"Wrote {len(dataset)} class-agnostic interval predictions to {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--train", required=True)
    training.add_argument("--val", required=True)
    training.add_argument("--epochs", type=int, default=20, help="Total epochs, including completed epochs on resume")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--lr", type=float, default=1e-3, help="Fresh-run learning rate; resume restores optimizer settings")
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--projection-dim", type=int, default=256)
    training.add_argument("--hidden-size", type=int, default=128)
    training.add_argument("--dropout", type=float, default=.2)
    training.add_argument("--head-type", choices=("gru", "linear"), default="gru",
                          help="GRU temporal head or a small per-frame linear probe on contextual features")
    training.add_argument("--tiers", nargs="+", choices=("verified_timestamps", "unverified_timestamps"), default=["verified_timestamps"])
    training.add_argument("--silver-positive-weight", type=float, default=.3)
    training.add_argument("--silver-negative-weight", type=float, default=.05)
    training.add_argument("--pos-weight-cap", type=float, default=20.)
    training.add_argument("--thresholds", type=float, nargs="+", default=[.2, .35, .5, .65, .8])
    training.add_argument("--min-duration", type=float, default=.04)
    training.add_argument("--merge-gap", type=float, default=0.)
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--resume")
    training.add_argument("--initialize", help="Start a new training stage from compatible model weights with a fresh optimizer")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--manifest", required=True)
    prediction.add_argument("--threshold", type=float)
    for command in (training, prediction):
        command.add_argument("--output", required=True)
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) <= 0:
        parser.error("batch-size and threads must be positive")
    if args.command == "train":
        if min(args.epochs, args.projection_dim, args.hidden_size) <= 0 or args.log_every < 0:
            parser.error("epochs/dimensions must be positive; log-every must be nonnegative")
        if any(not math.isfinite(value) or value <= 0 for value in (args.lr, args.pos_weight_cap)) or args.pos_weight_cap < 1:
            parser.error("lr must be positive and pos-weight-cap must be at least one")
        if any(not math.isfinite(value) or value < 0 for value in (args.weight_decay, args.min_duration, args.merge_gap, args.silver_positive_weight, args.silver_negative_weight)) or not 0 <= args.dropout < 1:
            parser.error("Weights and decoder durations must be finite/nonnegative; dropout must be in [0,1)")
        if any(not 0 < threshold < 1 for threshold in args.thresholds):
            parser.error("thresholds must lie strictly inside (0,1)")
    elif args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("threshold must lie strictly inside (0,1)")
    (train if args.command == "train" else predict)(args)


if __name__ == "__main__":
    main()
