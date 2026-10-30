"""Train the CRNN and select decoding threshold exclusively on validation data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import VALIDATION_TIERS, resolve_audio_path
from .metrics import score_records
from .model import (ClipDataset, FrontendConfig, NoiseCRNN, choose_device,
                    collate_clips, seed_everything)
from .postprocess import decode_events


def check_separate_splits(train: ClipDataset, val: ClipDataset) -> None:
    for name, left, right in [
        ("clip IDs", {r["clip_id"] for r in train.rows}, {r["clip_id"] for r in val.rows}),
        ("audio paths", {Path(resolve_audio_path(r, train.manifest)).resolve() for r in train.rows},
         {Path(resolve_audio_path(r, val.manifest)).resolve() for r in val.rows}),
        ("group IDs", {r["group_id"] for r in train.rows if r.get("group_id")},
         {r["group_id"] for r in val.rows if r.get("group_id")}),
    ]:
        common = left & right
        if common:
            raise ValueError(f"Train/validation leakage: overlapping {name}: {sorted(map(str, common))[:5]}")


def masked_loss(logits, batch, positive_weight, device):
    targets = batch["targets"].to(device)
    mask = batch["mask"].to(device)
    weights = mask * batch["weights"].to(device)[:, None]
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=positive_weight)
    # Keep tier weights effective even when an entire batch has the same tier.
    denominator = mask.sum().clamp_min(1)
    return (loss * weights).sum() / denominator, denominator.item()


@torch.inference_mode()
def validation_probabilities(model, loader, device):
    model.eval()
    results = []
    for batch in loader:
        probabilities = model(batch["features"].to(device), batch["lengths"]).sigmoid().cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Validation produced nonfinite probabilities")
        for index, clip in enumerate(batch["clips"]):
            results.append({"clip_id": clip["clip_id"], "duration": clip["duration"],
                            "events": clip["events"],
                            "probabilities": probabilities[index, :int(batch["lengths"][index])].numpy()})
    return results


def tune_threshold(records, frame_hop, thresholds, min_duration, merge_gap):
    references = [{k: r[k] for k in ("clip_id", "duration", "events")} for r in records]
    candidates = []
    for threshold in thresholds:
        decoder = {"threshold": threshold, "min_duration": min_duration, "merge_gap": merge_gap}
        predictions = [{"clip_id": r["clip_id"], "duration": r["duration"],
                        "events": decode_events(r["probabilities"], frame_hop, r["duration"], **decoder)}
                       for r in records]
        candidates.append((score_records(references, predictions), decoder))
    # Prefer a threshold nearer 0.5 on exact ties, then the lower threshold.
    return max(candidates, key=lambda item: (item[0]["combined"],
                -abs(item[1]["threshold"] - 0.5), -item[1]["threshold"]))


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state(generator, device):
    state = {"torch_cpu": torch.get_rng_state(), "shuffle": generator.get_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, generator, device):
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["shuffle"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15, help="Total epochs, including epochs already completed on resume")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Fresh-run learning rate; resume restores saved optimizer settings")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--seed", type=int, default=42, help="Fresh-run seed; resume restores saved random states")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--min-duration", type=float, default=0.06)
    parser.add_argument("--merge-gap", type=float, default=0.04)
    parser.add_argument("--log-every", type=int, default=100, help="Training steps between progress reports; 0 disables")
    parser.add_argument("--resume", help="Resume from last.pt, including optimizer and shuffle state")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads) <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size, threads, and lr must be positive")
    if args.log_every < 0:
        parser.error("log-every must be nonnegative")
    if any(not 0 < t < 1 for t in args.thresholds) or any(
        not math.isfinite(value) or value < 0 for value in (args.min_duration, args.merge_gap)
    ):
        parser.error("thresholds must be in (0, 1); decoder durations must be nonnegative")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume is not None and (resume.get("format_version") != 1 or "optimizer_state" not in resume):
        raise ValueError("Resume requires last.pt with optimizer state; best.pt is for inference")
    config = FrontendConfig(**resume["frontend"]) if resume else FrontendConfig()
    train = ClipDataset(args.train, config)
    val = ClipDataset(args.val, config)
    if val.skipped or any(row["quality"] not in VALIDATION_TIERS for row in val.rows):
        raise ValueError("Validation requires verified_timestamps (or preview_timestamps for smoke tests)")
    check_separate_splits(train, val)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_clips,
                              generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_clips)
    model = (NoiseCRNN(**resume["model_config"]) if resume else NoiseCRNN(n_mels=config.n_mels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positive_weight = torch.tensor(train.positive_weight(), device=device)
    metadata = {"train_manifest": str(train.manifest), "val_manifest": str(val.manifest),
                "train_clips": len(train), "val_clips": len(val),
                "skipped_no_timestamps": {"train": train.skipped, "val": val.skipped},
                "positive_weight": float(positive_weight.cpu()),
                "seed": resume["training"]["seed"] if resume else args.seed,
                "device": str(device), "arguments": vars(args),
                "label_tier_weights": {"verified_timestamps": 1.0, "unverified_timestamps": 0.3,
                                       "preview_timestamps": 1.0},
                "loss_normalization": "unweighted_valid_frames",
                "recurrence": "packed_variable_length",
                "manifest_sha256": {name: hashlib.sha256(dataset.manifest.read_bytes()).hexdigest()
                                    for name, dataset in (("train", train), ("val", val))},
                "preview_only": all(r["quality"] == "preview_timestamps" for r in train.rows + val.rows),
                "parameters": sum(p.numel() for p in model.parameters()),
                "torch_version": str(torch.__version__)}
    history, best_score, start_epoch, best_checkpoint = [], float("-inf"), 0, None
    if resume:
        if metadata["manifest_sha256"] != resume["training"].get("manifest_sha256"):
            raise ValueError("Cannot resume: training or validation manifest content changed")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        restore_rng(resume["rng_state"], generator, device)
        history, best_score, start_epoch = resume["history"], resume["best_combined"], resume["epoch"]
        best_checkpoint = resume["best_checkpoint"]
        metadata["resumed_from"] = str(Path(args.resume).resolve())
        metadata["completed_epochs_at_start"] = start_epoch
        save_checkpoint(output / "best.pt", best_checkpoint)
        if Path(args.resume).resolve() != (output / "last.pt").resolve():
            save_checkpoint(output / "last.pt", resume)
    metadata["effective_learning_rate"] = optimizer.param_groups[0]["lr"]
    print(json.dumps({"setup": metadata}), flush=True)
    if metadata["preview_only"]:
        print("PREVIEW SMOKE TEST ONLY: these few clips do not estimate competition performance.", flush=True)
    if start_epoch >= args.epochs:
        write_json(output / "history.json", {"training": metadata, "epochs": history, "best_combined": best_score})
        print(f"Checkpoint already completed {start_epoch} epochs; no further training requested.", flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        weighted_loss = denominator = 0.0
        clips_seen = 0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"].to(device), batch["lengths"])
            loss, weight = masked_loss(logits, batch, positive_weight, device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; inspect audio and annotation scales")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            weighted_loss += loss.item() * weight
            denominator += weight
            clips_seen += len(batch["clips"])
            if args.log_every and step % args.log_every == 0:
                elapsed = time.monotonic() - started
                print(json.dumps({"progress": {"epoch": epoch, "step": step, "steps": len(train_loader),
                      "train_loss": weighted_loss / denominator, "seconds": round(elapsed, 2),
                      "clips_per_second": round(clips_seen / elapsed, 2)}}), flush=True)
        records = validation_probabilities(model, val_loader, device)
        metrics, decoder = tune_threshold(records, config.frame_hop, args.thresholds,
                                         args.min_duration, args.merge_gap)
        entry = {"epoch": epoch, "train_loss": weighted_loss / denominator,
                 "validation": metrics, "decoder": decoder,
                 "seconds": round(time.monotonic() - started, 3)}
        history.append(entry)
        checkpoint = {"format_version": 1,
                      "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                      "model_config": model.config, "frontend": asdict(config),
                      "decoder": decoder, "epoch": epoch, "validation": metrics, "training": metadata}
        if metrics["combined"] > best_score:
            best_score = metrics["combined"]
            best_checkpoint = checkpoint
            save_checkpoint(output / "best.pt", best_checkpoint)
        save_checkpoint(output / "last.pt", {**checkpoint, "optimizer_state": optimizer.state_dict(),
                        "rng_state": rng_state(generator, device), "history": history,
                        "best_combined": best_score, "best_checkpoint": best_checkpoint})
        write_json(output / "history.json", {"training": metadata, "epochs": history,
                                             "best_combined": best_score})
        print(json.dumps(entry, allow_nan=False), flush=True)
    print(f"Saved checkpoint: {(output / 'best.pt').resolve()}", flush=True)
    print(f"Resume checkpoint: {(output / 'last.pt').resolve()}", flush=True)


if __name__ == "__main__":
    main()
