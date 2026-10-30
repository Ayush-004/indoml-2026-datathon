"""Fine-tune AudioSet-Strong Frame-MobileNet06 on an INNER Gold split only.

Native frontend: 16 kHz, pre-emphasis .97, 128 Kaldi mel bins, 25 ms window,
10 ms hop, FFT512, nonperiodic Hann, evaluation fmax=7000 Hz, (logmel+4.5)/5.
Fixed 10-second inputs produce 250 decision cells at 40 ms; centers are +20 ms.
Overlapping windows are weighted so each original clip cell has unit total loss
weight. Inference stitches logits using the same weights before applying sigmoid.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import load_audio, read_manifest, resolve_audio_path
from .model import choose_device, seed_everything
from .pretrained import (FRAME_HOP, FRAME_OFFSET, FRAME_SAMPLES, SAMPLE_RATE,
                         WINDOW_FRAMES, WINDOW_SAMPLES, file_sha256, window_starts)
from .train import check_separate_splits, restore_rng, rng_state, save_checkpoint, write_json
from .transfer import CATEGORIES, TemporalHead, calibrate, category_targets, decoded_record, write_predictions

MODEL_TYPE = "finetuned_frame_mn06_seven_category"
NATIVE_FILES = ("models/frame_mn/Frame_MN_wrapper.py", "models/frame_mn/model.py",
                "models/frame_mn/block_types.py", "models/frame_mn/utils.py",
                "models/frame_passt/preprocess.py", "models/transformer_wrapper.py", "helpers/encode.py")
TRIANGULAR = np.minimum((np.arange(WINDOW_FRAMES) + .5) / 125,
                       (WINDOW_FRAMES - np.arange(WINDOW_FRAMES) - .5) / 125).astype(np.float32)


class DeadlineReached(RuntimeError):
    pass


def check_deadline(deadline):
    # Reserve time for finishing the current small operation and writing state.
    if deadline is not None and time.time() >= deadline - 15:
        raise DeadlineReached("Configured deadline reached")


def parse_deadline(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline-utc must include a timezone")
    return parsed.timestamp()


class MobileSED(nn.Module):
    def __init__(self, source_root, *, pretrained=None, head_type="gru", projection_dim=256,
                 hidden_size=128, dropout=.2, bn_policy="frozen"):
        super().__init__()
        root = Path(source_root).resolve()
        hashes = {name: file_sha256(root / name) for name in NATIVE_FILES}
        sys.path.insert(0, str(root))
        module = importlib.import_module("models.frame_mn.Frame_MN_wrapper")
        if Path(module.__file__).resolve() != root / "models/frame_mn/Frame_MN_wrapper.py":
            raise ValueError("Frame-MobileNet imported from a different source root")
        with contextlib.redirect_stdout(io.StringIO()):
            native = module.FrameMNWrapper(width_mult=.6)
        self.frontend = native.mel.cpu().eval()
        self.encoder = native.frame_mn
        self.head = TemporalHead(feature_dim=576, projection_dim=projection_dim, hidden_size=hidden_size,
                                 dropout=dropout, head_type=head_type)
        self.bn_policy = bn_policy
        self.config = dict(head_type=head_type, projection_dim=projection_dim, hidden_size=hidden_size,
                           dropout=dropout, bn_policy=bn_policy)
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        self.provenance = {"source_root": str(root), "source_sha256": hashes,
                           "source_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                           "native_frontend": {"sample_rate": 16000, "preemphasis": .97, "n_mels": 128,
                               "window_samples": 400, "hop_samples": 160, "n_fft": 512,
                               "fmin": 0, "eval_fmax": 7000, "periodic_hann": False,
                               "normalization": "(natural_log(mel_power_clipped_1e-7)+4.5)/5"},
                           "bn_policy": "frozen running statistics; affine parameters trainable" if bn_policy == "frozen" else "update running statistics",
                           "frontend_policy": "CPU, eval mode, no augmentation, no gradients"}
        if pretrained:
            path = Path(pretrained).resolve()
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError("Expected the original Frame-MobileNet06 tensor state dictionary")
            prefix = "model.frame_mn."
            heads = {"strong_head.weight", "strong_head.bias", "weak_head.weight", "weak_head.bias"}
            if {key for key in state if not key.startswith(prefix)} != heads:
                raise ValueError("Unexpected released Frame-MobileNet checkpoint keys")
            for name in ("strong_head", "weak_head"):
                if state[name + ".weight"].shape != (447, 576) or state[name + ".bias"].shape != (447,):
                    raise ValueError("Expected the 447-class, width .6 release")
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Nonfinite pretrained parameters")
            self.encoder.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
            self.provenance.update(pretrained_path=str(path), pretrained_sha256=file_sha256(path),
                                   pretrained_bytes=path.stat().st_size, strict_pretrained_keys=len(state))
        self.frontend.requires_grad_(False)
        self.train(False)

    def set_device(self, device):
        self.encoder.to(device)
        self.head.to(device)
        self.frontend.cpu().eval()
        self.device = torch.device(device)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval()
        if self.bn_policy == "frozen":
            for layer in self.encoder.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def forward(self, waveform, lengths):
        if waveform.device.type != "cpu" or waveform.ndim != 2 or waveform.shape[1] != WINDOW_SAMPLES:
            raise ValueError("Expected a CPU batch of fixed 10-second 16 kHz windows")
        with torch.no_grad():
            spec = self.frontend(waveform)
        if spec.shape != (len(waveform), 1, 128, 1000):
            raise ValueError(f"Native frontend shape changed: {spec.shape}")
        features = self.encoder(spec.to(self.device))
        if features.shape != (len(waveform), WINDOW_FRAMES, 576):
            raise ValueError(f"Native encoder shape changed: {features.shape}")
        return self.head(features, lengths)


class AudioWindows(Dataset):
    def __init__(self, manifest, *, supervised=True):
        self.manifest = Path(manifest).resolve()
        self.rows = read_manifest(self.manifest)
        if not self.rows:
            raise ValueError("Empty audio manifest")
        self.windows, self.denominators, self.starts = [], [], []
        for row_index, row in enumerate(self.rows):
            if supervised:
                if row["quality"] != "verified_timestamps" or row["events"] is None:
                    raise ValueError("Fine-tuning and inner calibration require clean Gold timestamps")
                if any(event.get("category") not in CATEGORIES for event in row["events"]):
                    raise ValueError("Timed events need a canonical noise category")
                declared = set(row.get("source", {}).get("metadata", {}).get("NoiseCategory") or [])
                if declared - {event["category"] for event in row["events"]}:
                    raise ValueError("Clean INNER manifest still has a declared category without timestamps")
            samples = round(row["duration"] * SAMPLE_RATE)
            frames = (samples + FRAME_SAMPLES - 1) // FRAME_SAMPLES
            row.update(num_frames=frames, frame_hop=FRAME_HOP, frame_offset=FRAME_OFFSET)
            starts = window_starts(samples)
            denominator = np.zeros(frames, dtype=np.float32)
            for start in starts:
                first = start // FRAME_SAMPLES
                count = min(WINDOW_FRAMES, frames - first)
                denominator[first:first + count] += TRIANGULAR[:count]
                self.windows.append((row_index, start))
            self.starts.append(starts)
            self.denominators.append(denominator)
        self.supervised = supervised

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        row_index, start = self.windows[index]
        row = self.rows[row_index]
        audio = load_audio(resolve_audio_path(row, self.manifest), sample_rate=SAMPLE_RATE)
        if abs(len(audio) - round(row["duration"] * SAMPLE_RATE)) > 1:
            raise ValueError(f"Audio duration changed for {row['clip_id']}")
        waveform = torch.from_numpy(audio[start:start + WINDOW_SAMPLES])
        waveform = F.pad(waveform, (0, WINDOW_SAMPLES - len(waveform)))
        first = start // FRAME_SAMPLES
        count = min(WINDOW_FRAMES, row["num_frames"] - first)
        weights = np.zeros(WINDOW_FRAMES, dtype=np.float32)
        weights[:count] = TRIANGULAR[:count] / self.denominators[row_index][first:first + count]
        targets = torch.zeros(WINDOW_FRAMES, len(CATEGORIES))
        if self.supervised:
            full_targets = category_targets(row)
            targets[:count] = full_targets[first:first + count]
        return dict(waveform=waveform, targets=targets, weights=torch.from_numpy(weights), length=count,
                    row_index=row_index, start_frame=first)

    def positive_weights(self, cap=20):
        positive = np.zeros(len(CATEGORIES), dtype=np.float64)
        negative = np.zeros_like(positive)
        for row in self.rows:
            targets = category_targets(row).numpy()
            positive += targets.sum(axis=0)
            negative += len(targets) - targets.sum(axis=0)
        weights = np.clip(negative / np.maximum(positive, 1), 1, cap).astype(np.float32)
        return torch.from_numpy(weights), {name: dict(positive_frames=float(positive[i]), negative_frames=float(negative[i]),
                  positive_weight=float(weights[i])) for i, name in enumerate(CATEGORIES)}


def collate_windows(items):
    return {"waveform": torch.stack([item["waveform"] for item in items]),
            "targets": torch.stack([item["targets"] for item in items]),
            "weights": torch.stack([item["weights"] for item in items]),
            "lengths": torch.tensor([item["length"] for item in items]), "items": items}


def loss_for_batch(logits, batch, positive_weights, device):
    targets, weights = batch["targets"].to(device), batch["weights"].to(device)
    values = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weights, reduction="none")
    denominator = weights.sum().clamp_min(1) * len(CATEGORIES)
    return (values * weights[:, :, None]).sum() / denominator, float(denominator)


@torch.inference_mode()
def infer_records(model, dataset, batch_size, deadline=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_windows)
    active = {}
    for batch in loader:
        check_deadline(deadline)
        values = model(batch["waveform"], batch["lengths"]).float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite MobileNet logits")
        for i, item in enumerate(batch["items"]):
            index, first, count = item["row_index"], item["start_frame"], item["length"]
            row = dataset.rows[index]
            if index not in active:
                active[index] = [np.zeros((row["num_frames"], len(CATEGORIES)), dtype=np.float32), len(dataset.starts[index])]
            accumulator = active[index]
            weight = item["weights"][:count].numpy()
            accumulator[0][first:first + count] += values[i, :count] * weight[:, None]
            accumulator[1] -= 1
            if accumulator[1] == 0:
                probability = torch.from_numpy(accumulator[0]).sigmoid().numpy()
                yield row, probability
                del active[index]
    if active:
        raise ValueError("Incomplete window stitching")


def load_mobile_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1
            or checkpoint.get("model_type") != MODEL_TYPE or checkpoint.get("categories") != list(CATEGORIES)):
        raise ValueError("Not a compatible MobileNet06 checkpoint")
    return checkpoint


def train(args, deadline):
    seed_everything(args.seed)
    device = choose_device(args.device)
    training, validation = AudioWindows(args.train), AudioWindows(args.val)
    check_separate_splits(training, validation)
    def geography(row):
        metadata = row.get("source", {}).get("metadata", {})
        values = [metadata.get(name) for name in ("state", "district")]
        if all(isinstance(value, str) and value.strip() for value in values):
            return tuple(" ".join(unicodedata.normalize("NFKC", value).split()).casefold() for value in values)
    overlap = ({geography(row) for row in training.rows} - {None}) & ({geography(row) for row in validation.rows} - {None})
    if overlap:
        raise ValueError(f"Training/INNER calibration share state/district groups: {sorted(overlap)[:5]}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume = load_mobile_checkpoint(args.resume) if args.resume else None
    if not resume and any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise ValueError("Use a new output directory or --resume")
    if resume:
        for name in ("best.pt", "last.pt"):
            if (output / name).exists():
                load_mobile_checkpoint(output / name)
    config = resume["model_config"] if resume else dict(head_type=args.head_type, projection_dim=args.projection_dim,
                 hidden_size=args.hidden_size, dropout=args.dropout, bn_policy=args.bn_policy)
    model = MobileSED(args.source_root, pretrained=None if resume else args.pretrained_checkpoint, **config).set_device(device)
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.encoder_lr},
                                  {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=args.weight_decay)
    positive_weights, audit = training.positive_weights(args.pos_weight_cap)
    positive_weights = positive_weights.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=collate_windows, generator=generator)
    metadata = {"arguments": vars(args), "selection_data": "INNER geographic Gold calibration only; outer unused",
                "manifest_sha256": {"train": file_sha256(training.manifest), "inner_validation": file_sha256(validation.manifest)},
                "train_clips": len(training.rows), "train_windows": len(training), "inner_validation_clips": len(validation.rows),
                "inner_validation_windows": len(validation), "class_weight_audit": audit,
                "label_grid": "cell centers .02+i*.04 inside half-open events", "positive_weight_cap": args.pos_weight_cap,
                "window_loss_weighting": "triangular overlap weights normalized per original clip cell",
                "model_provenance": model.provenance, "head_config": model.head.config,
                "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                "head_parameters": sum(p.numel() for p in model.head.parameters()),
                "device": str(device), "torch_version": str(torch.__version__), "seed": args.seed,
                "deadline_utc": args.deadline_utc, "deadline_safety_margin_seconds": 15,
                "learning_rate_schedule": "constant"}
    history, best_score, first_epoch, best = [], -math.inf, 0, None
    if resume:
        if ("optimizer_state" not in resume or resume["training"]["manifest_sha256"] != metadata["manifest_sha256"]
                or resume["training"]["positive_weight_cap"] != args.pos_weight_cap
                or resume["training"]["model_provenance"]["source_sha256"] != model.provenance["source_sha256"]):
            raise ValueError("Resume requires last.pt with unchanged data, native source and loss policy")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        restore_rng(resume["rng_state"], generator, device)
        history, best_score, first_epoch, best = resume["history"], resume["best_combined"], resume["epoch"], resume["best_checkpoint"]
        metadata["seed"] = resume["training"]["seed"]
        metadata["model_provenance"] = resume["training"]["model_provenance"]
        save_checkpoint(output / "best.pt", best)
    metadata["effective_learning_rates"] = [group["lr"] for group in optimizer.param_groups]
    write_json(output / "training-audit.json", metadata)
    state = {"status": "training", "pid": os.getpid(), "last_completed_epoch": first_epoch,
             "command": [sys.executable, *sys.argv], "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "run-state.json", state)
    print(json.dumps({"setup": metadata}), flush=True)
    try:
        for epoch in range(first_epoch + 1, args.epochs + 1):
            check_deadline(deadline)
            started = time.monotonic()
            model.train()
            total_loss = total_weight = 0.
            for step, batch in enumerate(loader, 1):
                check_deadline(deadline)
                optimizer.zero_grad(set_to_none=True)
                loss, weight = loss_for_batch(model(batch["waveform"], batch["lengths"]), batch, positive_weights, device)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite fine-tuning loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item() * weight
                total_weight += weight
                if args.log_every and step % args.log_every == 0:
                    print(json.dumps({"progress": {"epoch": epoch, "step": step, "steps": len(loader),
                          "train_loss": total_loss / total_weight, "seconds": round(time.monotonic() - started, 2)}}), flush=True)
            records = list(infer_records(model, validation, args.batch_size, deadline))
            metrics, decoder, candidates = calibrate(records, args.thresholds, args.min_duration, args.merge_gap)
            entry = {"epoch": epoch, "train_loss": total_loss / total_weight, "validation": metrics,
                     "decoder": decoder, "calibration": candidates, "seconds": time.monotonic() - started}
            history.append(entry)
            checkpoint = {"format_version": 1, "model_type": MODEL_TYPE, "categories": list(CATEGORIES),
                          "model_config": config, "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "frame_grid": {"frame_hop": FRAME_HOP, "frame_offset": FRAME_OFFSET},
                          "epoch": epoch, "decoder": decoder, "validation": metrics, "training": metadata}
            if metrics["combined"] > best_score:
                best_score, best = metrics["combined"], checkpoint
                save_checkpoint(output / "best.pt", best)
                write_predictions(output / "inner-validation-predictions.jsonl", (decoded_record(row, prob, decoder) for row, prob in records))
            save_checkpoint(output / "last.pt", {**checkpoint, "optimizer_state": optimizer.state_dict(),
                            "rng_state": rng_state(generator, device), "history": history,
                            "best_combined": best_score, "best_checkpoint": best})
            write_json(output / "history.json", {"training": metadata, "epochs": history, "best_combined": best_score})
            state.update(last_completed_epoch=epoch, best_inner_combined=best_score,
                         updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "run-state.json", state)
            print(json.dumps({k: entry[k] for k in ("epoch", "train_loss", "validation", "decoder", "seconds")}), flush=True)
        state["status"] = "complete"
    except DeadlineReached:
        state.update(status="deadline_stopped", note="Partial epoch discarded; last.pt resumes the last completed epoch")
        print("Deadline reached; preserved the last complete epoch checkpoint.", flush=True)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run-state.json", state)


def predict(args, deadline):
    checkpoint = load_mobile_checkpoint(args.checkpoint)
    model = MobileSED(args.source_root, **checkpoint["model_config"]).set_device(choose_device(args.device))
    if model.provenance["source_sha256"] != checkpoint["training"]["model_provenance"]["source_sha256"]:
        raise ValueError("Native model source differs from the training checkpoint")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dataset = AudioWindows(args.manifest, supervised=False)
    output = Path(args.output).resolve()
    protected = {Path(args.checkpoint).resolve(), dataset.manifest}
    protected.update(Path(resolve_audio_path(row, dataset.manifest)).resolve() for row in dataset.rows)
    if output in protected or output.with_suffix(output.suffix + ".tmp") in protected:
        raise ValueError("Predictions would overwrite source data")
    write_predictions(output, (decoded_record(row, values, checkpoint["decoder"])
                              for row, values in infer_records(model, dataset, args.batch_size, deadline)))
    print(f"Wrote {len(dataset.rows)} predictions to {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--train", required=True)
    fitting.add_argument("--val", required=True, help="INNER calibration manifest, never the outer comparison split")
    fitting.add_argument("--pretrained-checkpoint", required=True)
    fitting.add_argument("--epochs", type=int, default=15)
    fitting.add_argument("--encoder-lr", type=float, default=1e-4)
    fitting.add_argument("--head-lr", type=float, default=1e-3)
    fitting.add_argument("--weight-decay", type=float, default=1e-4)
    fitting.add_argument("--pos-weight-cap", type=float, default=20.)
    fitting.add_argument("--head-type", choices=("gru", "linear"), default="gru")
    fitting.add_argument("--projection-dim", type=int, default=256)
    fitting.add_argument("--hidden-size", type=int, default=128)
    fitting.add_argument("--dropout", type=float, default=.2)
    fitting.add_argument("--bn-policy", choices=("frozen", "update"), default="frozen")
    fitting.add_argument("--thresholds", nargs="+", type=float, default=[.2, .35, .5, .65, .8])
    fitting.add_argument("--min-duration", type=float, default=.04)
    fitting.add_argument("--merge-gap", type=float, default=0.)
    fitting.add_argument("--log-every", type=int, default=100)
    fitting.add_argument("--seed", type=int, default=42)
    fitting.add_argument("--resume")
    inference = commands.add_parser("predict")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--manifest", required=True)
    for command in (fitting, inference):
        command.add_argument("--source-root", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", choices=("mps", "cpu", "cuda", "auto"), default="mps")
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--deadline-utc")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) <= 0:
        parser.error("batch-size and threads must be positive")
    if args.command == "train":
        if min(args.epochs, args.projection_dim, args.hidden_size) <= 0 or args.log_every < 0:
            parser.error("epochs/dimensions must be positive and log-every nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in (args.encoder_lr, args.head_lr, args.pos_weight_cap)) or args.pos_weight_cap < 1:
            parser.error("learning rates must be positive and pos-weight-cap at least one")
        if any(not math.isfinite(v) or v < 0 for v in (args.weight_decay, args.min_duration, args.merge_gap)) or not 0 <= args.dropout < 1:
            parser.error("Invalid weight decay, decoder durations, or dropout")
        if any(not 0 < v < 1 for v in args.thresholds):
            parser.error("thresholds must be in (0,1)")
    deadline = parse_deadline(args.deadline_utc)
    torch.set_num_threads(args.threads)
    check_deadline(deadline)
    (train if args.command == "train" else predict)(args, deadline)


if __name__ == "__main__":
    main()
