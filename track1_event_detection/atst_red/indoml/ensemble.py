"""Select a fixed probability ensemble on INNER, then score OUTER exactly once.

Only cached FP32 probabilities are used. No model inference or training occurs.
The predeclared baseline and its original decoder win numerical ties.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import read_manifest
from .metrics import score_records
from .submission import validate_predictions
from .train import write_json
from .transfer import CATEGORIES, decoded_record, load_transfer_checkpoint, write_predictions
from .transfer_calibrate import candidate_decoders

GRID = {"thresholds": [.2, .35, .5, .65, .8, .9, .95],
        "min_durations": [.04, .08, .12], "merge_gaps": [0., .04, .08, .12]}
WEIGHTS = [.25, .5, .75]
FRAME_GRID = {"frame_hop": .04, "frame_offset": .02}
LOGIT_EPSILON = 1e-7
TIE_EPSILON = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(spec):
    path = Path(spec["path"]).resolve()
    require(sha256(path) == spec["sha256"], f"File hash changed: {path}")
    return path


def check_deadline(deadline):
    if time.time() >= deadline - 15:
        raise RuntimeError("Ensemble deadline reached; no further exploration or evaluation")


def canonical_rows(path, expected_count):
    rows = read_manifest(path)
    require(len(rows) == expected_count, f"Expected {expected_count} canonical clips")
    return [dict(row, **FRAME_GRID, num_frames=math.ceil(row["duration"] / .04 - 1e-9)) for row in rows]


def align_probabilities(probabilities, offsets, clip_ids, rows):
    """Return clip arrays in canonical order, never silently pairing by position."""
    require(probabilities.dtype == np.float32, "Probability archives must retain FP32")
    require(probabilities.ndim == 2 and probabilities.shape[1] == len(CATEGORIES), "Expected probabilities [frames, 7]")
    require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "Probabilities must be finite and inside [0,1]")
    require(clip_ids.ndim == 1 and clip_ids.dtype.kind == "U", "clip_ids must be a Unicode vector")
    ids = clip_ids.tolist()
    require(len(ids) == len(set(ids)), "Archive has duplicate clip IDs")
    require(offsets.ndim == 1 and offsets.dtype.kind in "iu" and len(offsets) == len(ids) + 1,
            "Offsets must contain one integer boundary per clip plus the endpoint")
    require(int(offsets[0]) == 0 and int(offsets[-1]) == len(probabilities)
            and np.all(offsets[1:] > offsets[:-1]), "Archive offsets do not partition all frames")
    expected_ids = [row["clip_id"] for row in rows]
    require(len(expected_ids) == len(set(expected_ids)), "Canonical rows have duplicate clip IDs")
    require(set(ids) == set(expected_ids), "Archive and canonical clip IDs differ")
    positions = {clip_id: index for index, clip_id in enumerate(ids)}
    result = []
    for row in rows:
        index = positions[row["clip_id"]]
        values = probabilities[int(offsets[index]):int(offsets[index + 1])]
        require(len(values) == row["num_frames"], f"Frame count differs for {row['clip_id']}")
        result.append(values)
    return result


def verify_manifest_rows(source, canonical):
    """Feature manifests may have different paths, but labels and time axes must agree."""
    lookup = {row["clip_id"]: row for row in source}
    require(set(lookup) == {row["clip_id"] for row in canonical}, "Source manifest IDs differ from canonical split")
    for row in canonical:
        other = lookup[row["clip_id"]]
        require(other["duration"] == row["duration"] and other["events"] == row["events"],
                f"Source manifest labels or duration differ for {row['clip_id']}")
        require(all(other.get(key, value) == value for key, value in FRAME_GRID.items()), "Source frame grid differs")
        require(other.get("num_frames", row["num_frames"]) == row["num_frames"], "Source frame count differs")


def load_archive(spec, checkpoint, checkpoint_hash, rows):
    path = checked_file(spec["archive"])
    manifest = checked_file(spec["manifest"])
    verify_manifest_rows(read_manifest(manifest), rows)
    # Read the same bytes that are hashed; replacement cannot decouple provenance.
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == spec["archive"]["sha256"], "Archive changed during loading")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require({"probabilities", "offsets", "clip_ids", "metadata"} <= set(archive.files), "Missing archive fields")
        metadata = json.loads(str(archive["metadata"].item()))
        require(metadata["checkpoint_sha256"] == checkpoint_hash, "Archive belongs to another checkpoint")
        require(metadata["manifest_sha256"] == spec["manifest"]["sha256"], "Archive source manifest hash differs")
        require(metadata["categories"] == list(CATEGORIES) == checkpoint["categories"], "Category order differs")
        require(all(metadata[key] == value for key, value in FRAME_GRID.items()), "Archive frame grid differs")
        require(metadata["feature_cache_id"] == checkpoint["feature_cache_id"], "Feature cache identity differs")
        require(metadata.get("epoch", metadata.get("selected_epoch", metadata.get("checkpoint_epoch"))) == checkpoint["epoch"], "Archive epoch differs")
        require(metadata["decoder"] == checkpoint["decoder"], "Archive original decoder differs")
        return align_probabilities(archive["probabilities"], archive["offsets"], archive["clip_ids"], rows)


def blend(left, right, left_weight, method):
    """FP32 output; logit arithmetic uses float64 and clips only mixed endpoints."""
    require(method in {"probability", "logit"}, "Unknown ensemble method")
    require(math.isfinite(left_weight) and 0 <= left_weight <= 1, "Weight must be inside [0,1]")
    require(left.shape == right.shape, "Ensemble probability shapes differ")
    if left_weight == 1:
        return left.copy()
    if left_weight == 0:
        return right.copy()
    a, b = left.astype(np.float64), right.astype(np.float64)
    if method == "probability":
        return (left_weight * a + (1 - left_weight) * b).astype(np.float32)
    a, b = np.clip(a, LOGIT_EPSILON, 1 - LOGIT_EPSILON), np.clip(b, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
    logits = left_weight * (np.log(a) - np.log1p(-a)) + (1 - left_weight) * (np.log(b) - np.log1p(-b))
    return (1 / (1 + np.exp(-logits))).astype(np.float32)


def recipes():
    return ([{"name": "left_control", "method": "probability", "weights": [1., 0.]},
             {"name": "right_control", "method": "probability", "weights": [0., 1.]}]
            + [{"name": f"{method}_left_{weight:g}", "method": method, "weights": [weight, 1 - weight]}
               for method in ("probability", "logit") for weight in WEIGHTS])


def blend_many(arrays, weights, method):
    require(len(arrays) == len(weights) and len(arrays) >= 2, "Recipe source count differs")
    require(method in {"probability", "logit"}, "Unknown ensemble method")
    require(all(math.isfinite(weight) and 0 <= weight <= 1 for weight in weights)
            and abs(sum(weights) - 1) <= TIE_EPSILON, "Recipe weights must be nonnegative and sum to one")
    require(all(array.shape == arrays[0].shape for array in arrays), "Ensemble probability shapes differ")
    active = [(array, weight) for array, weight in zip(arrays, weights, strict=True) if weight > 0]
    if len(active) == 1:
        return active[0][0].copy()
    if len(active) == 2:
        # Preserve byte-for-byte parity with the original two-source implementation.
        return blend(active[0][0], active[1][0], active[0][1], method)
    total = np.zeros_like(arrays[0], dtype=np.float64)
    for array, weight in active:
        values = array.astype(np.float64)
        if method == "logit":
            values = np.clip(values, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
            values = np.log(values) - np.log1p(-values)
        total += weight * values
    return (1 / (1 + np.exp(-total)) if method == "logit" else total).astype(np.float32)


def blended_records(values, recipe):
    weights = recipe.get("weights")
    if "left_weight" in recipe:
        require(len(values) == 2, "Legacy left_weight recipes require exactly two sources")
        legacy = [recipe["left_weight"], 1 - recipe["left_weight"]]
        require(weights is None or list(weights) == legacy, "Legacy and explicit recipe weights disagree")
        weights = legacy
    require(weights is not None, "Recipe requires weights or legacy left_weight")
    return [blend_many(arrays, weights, recipe["method"]) for arrays in zip(*values, strict=True)]


def decode(rows, values, decoder):
    return [decoded_record(row, probability, decoder) for row, probability in zip(rows, values, strict=True)]


def verify_control(rows, values, checkpoint):
    metrics = score_records(rows, decode(rows, values, checkpoint["decoder"]))
    expected = checkpoint["validation"]
    require(set(metrics) == set(expected), "Control metric fields differ")
    for key, value in metrics.items():
        require(abs(value - expected[key]) <= TIE_EPSILON, f"Original control score parity failed: {key}: {value} != {expected[key]}")
    return metrics


def select_inner(rows, values, checkpoints, deadline, on_recipe=None, *, recipe_grid=None, baseline=None):
    controls = [verify_control(rows, source_values, checkpoint) for source_values, checkpoint in zip(values, checkpoints, strict=True)]
    grid = candidate_decoders(None, **GRID)
    require(len(grid) == 84, "Expected the fixed 84-decoder grid")
    recipe_grid = recipes() if recipe_grid is None else recipe_grid
    baseline = ({"recipe": recipe_grid[0], "decoder": checkpoints[0]["decoder"], "metrics": controls[0]}
                if baseline is None else baseline)
    baseline = dict(baseline, baseline=True)
    require(baseline["recipe"] in recipe_grid, "Baseline recipe must be a predeclared candidate")
    baseline_actual = score_records(rows, decode(rows, blended_records(values, baseline["recipe"]), baseline["decoder"]))
    require(all(abs(value - baseline["metrics"][key]) <= TIE_EPSILON for key, value in baseline_actual.items()),
            "Predeclared baseline score parity failed")
    trials, selected = [baseline], baseline
    for recipe in recipe_grid:
        check_deadline(deadline)
        blended = blended_records(values, recipe)
        for decoder in grid:
            check_deadline(deadline)
            if recipe == baseline["recipe"] and decoder == baseline["decoder"]:
                continue
            metrics = score_records(rows, decode(rows, blended, decoder))
            candidate = {"recipe": recipe, "decoder": decoder, "metrics": metrics, "baseline": False}
            trials.append(candidate)
            if metrics["combined"] > selected["metrics"]["combined"] + TIE_EPSILON:
                selected = candidate
        if on_recipe:
            on_recipe(recipe, trials, selected)
    require(selected["metrics"]["combined"] + TIE_EPSILON >= baseline["metrics"]["combined"], "INNER nondegradation failed")
    return {"selected": selected, "baseline": baseline, "original_control_metrics": controls,
            "candidate_count": len(trials), "trials": trials,
            "inner_combined_gain": selected["metrics"]["combined"] - baseline["metrics"]["combined"]}


def run(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    deadline = datetime.fromisoformat(protocol["deadline_utc"].replace("Z", "+00:00")).timestamp()
    check_deadline(deadline)
    require(protocol["grid"] == GRID and protocol["left_weights"] == WEIGHTS, "Protocol differs from the fixed search")
    if "predeclaration" in protocol:
        declaration = json.loads(checked_file(protocol["predeclaration"]).read_text())
        require(all(protocol[key] == declaration[key] for key in ("recipes", "baseline", "grid", "deadline_utc")),
                "Recipe family changed after predeclaration")
        require([source["name"] for source in protocol["sources"]] == declaration["source_order"], "Source order changed")
    output = Path(protocol["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "process.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another ensemble finalizer owns this output") from exc
    state = {"status": "validating_inner", "pid": os.getpid(),
             "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(), "script_sha256": sha256(__file__),
             "started_utc": datetime.now(timezone.utc).isoformat()}
    guard = output / "outer-evaluation-state.json"
    try:
        require(not guard.exists(), "OUTER evaluation already claimed; never repeat automatically")
        implementation = Path(__file__).read_bytes()
        require(hashlib.sha256(implementation).hexdigest() == state["script_sha256"], "Implementation changed during startup")
        (output / "ensemble-implementation.py").write_bytes(implementation)
        write_json(output / "run-state.json", state)
        inner_path, outer_path = checked_file(protocol["inner_manifest"]), checked_file(protocol["outer_manifest"])
        inner = canonical_rows(inner_path, 1439)
        # No OUTER probabilities or scores are accessed before INNER selection.
        outer_ids = {row["clip_id"] for row in read_manifest(outer_path)}
        inner_ids = {row["clip_id"] for row in inner}
        require(len(outer_ids) == 1443 and not outer_ids & inner_ids, "INNER/OUTER split IDs overlap or counts differ")
        checkpoints, values = [], []
        for source in protocol["sources"]:
            check_deadline(deadline)
            checkpoint_path = checked_file(source["checkpoint"])
            payload = checkpoint_path.read_bytes()
            require(hashlib.sha256(payload).hexdigest() == source["checkpoint"]["sha256"], "Checkpoint changed during loading")
            checkpoint = load_transfer_checkpoint(io.BytesIO(payload))
            require(checkpoint["feature_grid"] == FRAME_GRID, "Checkpoint frame grid differs")
            require(checkpoint["calibration"]["validation_manifest_sha256"] == source["inner"]["manifest"]["sha256"],
                    "Checkpoint was not calibrated on this INNER source manifest")
            train_manifest = Path(checkpoint["training"]["arguments"]["train"])
            require(sha256(train_manifest) == checkpoint["training"]["manifest_sha256"]["train"], "Training manifest changed")
            training = read_manifest(train_manifest)
            train_ids = {row["clip_id"] for row in training}
            require(not train_ids & (inner_ids | outer_ids), "An ensemble checkpoint trained on INNER or OUTER clips")
            values.append(load_archive(source["inner"], checkpoint, source["checkpoint"]["sha256"], inner))
            checkpoints.append(checkpoint)
        require(len(checkpoints) in {2, 3, 4}, "Expected two, three, or four predeclared sources")

        def progress(recipe, trials, selected):
            state.update(status="selecting_inner", completed_recipe=recipe["name"], completed_trials=len(trials), current_best=selected)
            write_json(output / "run-state.json", state)
            write_json(output / "inner-trials-progress.json", trials)
            print(json.dumps({"completed_recipe": recipe["name"], "trials": len(trials), "current_best": selected}), flush=True)

        selection = select_inner(inner, values, checkpoints, deadline, progress,
                                 recipe_grid=protocol.get("recipes"), baseline=protocol.get("baseline"))
        if "predeclaration" in protocol:
            require(selection["candidate_count"] == declaration["candidate_count"], "Predeclared trial count differs")
        selected = selection["selected"]
        recipe = {"format_version": 1, "sources": protocol["sources"], "selected": selected,
                  "baseline": selection["baseline"],
                  "canonical_manifests": {"inner": protocol["inner_manifest"], "outer": protocol["outer_manifest"]},
                  "grid": GRID, "left_weights": WEIGHTS, "logit_epsilon": LOGIT_EPSILON,
                  "blend_precision": "float64 arithmetic then FP32 output; endpoints preserved exactly",
                  "tie_policy": "Predeclared original baseline first; replace only for combined gain greater than 1e-12; otherwise fixed recipe/grid order",
                  "selection_scope": "All ensemble and decoder selection used INNER 1439 only; OUTER 1443 evaluated once afterward",
                  "protocol_sha256": state["protocol_sha256"], "script_sha256": state["script_sha256"],
                  "candidate_count": selection["candidate_count"], "inner_combined_gain": selection["inner_combined_gain"]}
        write_json(output / "recipe.json", recipe)
        write_json(output / "inner-selection.json", selection)
        inner_predictions = decode(inner, blended_records(values, selected["recipe"]), selected["decoder"])
        validate_predictions(inner_predictions, inner)
        write_predictions(output / "inner-predictions.jsonl", inner_predictions)
        check_deadline(deadline)
        # Claim is exclusive even if a second process passed an earlier existence check.
        with guard.open("x") as stream:
            json.dump({"status": "running", "outer_evaluations_started": 1, "recipe_sha256": sha256(output / "recipe.json")}, stream)
        state.update(status="evaluating_outer")
        write_json(output / "run-state.json", state)
        outer = canonical_rows(outer_path, 1443)
        outer_values = [load_archive(source["outer"], checkpoint, source["checkpoint"]["sha256"], outer)
                        for source, checkpoint in zip(protocol["sources"], checkpoints, strict=True)]
        check_deadline(deadline)
        predictions = decode(outer, blended_records(outer_values, selected["recipe"]), selected["decoder"])
        validate_predictions(predictions, outer)
        write_predictions(output / "outer-predictions.jsonl", predictions)
        outer_metrics = score_records(outer, predictions)
        report = {"status": "complete", "selected": selected, "inner_combined_gain": selection["inner_combined_gain"],
                  "baseline": selection["baseline"], "source_order": [source["name"] for source in protocol["sources"]],
                  "original_control_inner_metrics": selection["original_control_metrics"], "outer_metrics": outer_metrics,
                  "validation_checks": {"fp32_archive_dtype": True, "checkpoint_archive_manifest_hashes": True,
                      "canonical_ids_labels_grid_and_category_order": True, "training_excludes_inner_outer_ids": True,
                      "original_control_inner_score_parity": True, "baseline_inner_score_parity": True},
                  "outer_evaluations_completed": 1, "recipe_sha256": sha256(output / "recipe.json"),
                  "protocol_sha256": state["protocol_sha256"], "candidate_count": selection["candidate_count"],
                  "selection_scope": recipe["selection_scope"], "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "interpretation": "Local geographic OUTER comparison, not an official or hidden-test result"}
        write_json(output / "report.json", report)
        write_json(guard, report)
        state.update(status="complete", result=report)
        write_json(output / "run-state.json", state)
        print(json.dumps(report), flush=True)
        return report
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        write_json(output / "run-state.json", state)
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.protocol)


if __name__ == "__main__":
    main()
