"""Create a geographic Gold holdout using the release's actual state/district fields.

This evaluates geographic generalization. District separation does not establish
speaker separation: the release has no documented speaker/session identifiers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import unicodedata

from .data import read_manifest, resolve_audio_path
from .prepare import _write_json, _write_manifest, audit_rows

GOLD = "verified_timestamps"
UNKNOWN_GEOGRAPHY = frozenset({"unknown", "none", "null", "nan", "n/a", "na", "-"})


def geographic_key(row: dict) -> tuple[str, str]:
    """Use supplied geography; normalize only Unicode, whitespace, and case."""
    metadata = row.get("source", {}).get("metadata", {})
    parts = []
    for field in ("state", "district"):
        value = metadata.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{row['clip_id']}: missing string source.metadata.{field}")
        value = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
        if not value or value in UNKNOWN_GEOGRAPHY:
            raise ValueError(f"{row['clip_id']}: unknown source.metadata.{field}")
        parts.append(value)
    return tuple(parts)


def _select_groups(groups: dict, *, val_fraction: float, min_fraction: float,
                   max_fraction: float, seed: int, search_trials: int) -> tuple[set, dict]:
    """Bounded seeded search over whole districts; no optimization dependencies.

    Prefer the requested size band, preserve Gold training languages, cover Gold
    validation languages available in multiple districts, then balance language
    counts around their targets. Equal candidates lose less non-Gold training data.
    This heuristic reports its result; it does not claim a globally optimal split.
    """
    candidates = sorted(key for key, group in groups.items() if group["gold_clips"])
    if len(candidates) < 2:
        raise ValueError("Need Gold clips in at least two different state/district groups")
    total_languages = Counter()
    language_groups = Counter()
    for key in candidates:
        total_languages.update(groups[key]["gold_languages"])
        language_groups.update(groups[key]["gold_languages"].keys())
    total_gold = sum(total_languages.values())
    target = val_fraction * total_gold
    lower, upper = min_fraction * total_gold, max_fraction * total_gold
    languages = sorted(total_languages)
    splittable_languages = {language for language in languages if language_groups[language] >= 2}
    best_selection, best_score = None, None
    evaluated = 0

    def score(selection):
        nonlocal evaluated
        evaluated += 1
        counts = Counter()
        excluded = 0
        for key in selection:
            counts.update(groups[key]["gold_languages"])
            excluded += groups[key]["clips"] - groups[key]["gold_clips"]
        count = sum(counts.values())
        if count == 0 or count == total_gold:
            return None
        outside_band = max(lower - count, count - upper, 0.0)
        lost_train = sum(counts[language] == total_languages[language] for language in languages)
        missing_val = sum(counts[language] == 0 for language in splittable_languages)
        language_error = sum(abs(counts[language] - val_fraction * total_languages[language])
                             for language in languages) / total_gold
        return (outside_band, lost_train, missing_val, language_error,
                abs(count - target) / total_gold, excluded, tuple(sorted(selection)))

    def consider(selection):
        nonlocal best_selection, best_score
        result = score(selection)
        if result is not None and (best_score is None or result < best_score):
            best_score, best_selection = result, set(selection)

    for key in candidates:
        consider({key})
    rng = random.Random(seed)
    # Random group order gives many distinct size-constrained subsets. Check each
    # prefix and its first oversized additions to handle coarse district sizes.
    for _ in range(search_trials):
        order = candidates.copy()
        rng.shuffle(order)
        selection, count = set(), 0
        for key in order:
            proposal = selection | {key}
            consider(proposal)
            if count + groups[key]["gold_clips"] <= upper:
                selection = proposal
                count += groups[key]["gold_clips"]
    if best_selection is None:
        raise ValueError("No nonempty train/validation Gold split is possible")
    # Refine the best candidate through a bounded number of additions, removals,
    # and swaps; the score only improves, and the random seed fixes all proposals.
    for _ in range(search_trials * 8):
        proposal = set(best_selection)
        key = rng.choice(candidates)
        if key in proposal:
            proposal.remove(key)
        else:
            proposal.add(key)
        if rng.random() < .5:
            other = rng.choice(candidates)
            if other in proposal:
                proposal.remove(other)
            else:
                proposal.add(other)
        consider(proposal)
    summary = {
        "algorithm": "seeded_district_permutations_with_local_refinement_v1",
        "search_trials": search_trials,
        "local_proposals": search_trials * 8,
        "candidates_evaluated": evaluated,
        "optimality_guaranteed": False,
        "objective_priority": ["distance_outside_preferred_gold_fraction_band",
                               "gold_languages_lost_from_training",
                               "missing_validation_languages_with_gold_in_multiple_districts",
                               "absolute_error_from_per_language_gold_count_targets",
                               "absolute_error_from_total_gold_count_target",
                               "excluded_non_gold_clips", "sorted_geographic_keys"],
        "winning_objective": list(best_score[:-1]),
        "gold_languages_in_one_district": sorted(language for language in languages if language_groups[language] == 1),
        "gold_languages_in_multiple_districts": sorted(splittable_languages),
    }
    return best_selection, summary


def split_geographic(manifest: str | Path, output: str | Path, *, val_fraction: float = .125,
                     min_fraction: float = .10, max_fraction: float = .15,
                     seed: int = 42, search_trials: int = 512) -> dict:
    if not 0 < min_fraction <= val_fraction <= max_fraction < 1:
        raise ValueError("Require 0 < min_fraction <= val_fraction <= max_fraction < 1")
    if search_trials <= 0:
        raise ValueError("search_trials must be positive")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if manifest in {output / name for name in ("train.jsonl", "val.jsonl", "excluded-heldout.jsonl")}:
        raise ValueError("Output filenames would overwrite the input manifest; choose another output directory")
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError("Cannot split an empty manifest")
    groups = defaultdict(lambda: {"clips": 0, "gold_clips": 0, "gold_languages": Counter(), "variants": set()})
    keys = {}
    for row in rows:
        key = geographic_key(row)  # Require geography across all tiers to exclude held-out districts fully.
        keys[row["clip_id"]] = key
        group = groups[key]
        group["clips"] += 1
        metadata = row["source"]["metadata"]
        group["variants"].add((metadata["state"], metadata["district"]))
        if row["quality"] == GOLD:
            group["gold_clips"] += 1
            group["gold_languages"][row["language"]] += 1
    held_out, search = _select_groups(groups, val_fraction=val_fraction, min_fraction=min_fraction,
                                      max_fraction=max_fraction, seed=seed, search_trials=search_trials)
    train, validation, excluded = [], [], []
    for original in rows:
        row = dict(original)
        row["audio_path"] = os.path.relpath(resolve_audio_path(row, manifest), output)
        if keys[row["clip_id"]] in held_out:
            (validation if row["quality"] == GOLD else excluded).append(row)
        else:
            train.append(row)
    train_gold = [row for row in train if row["quality"] == GOLD]
    if not train_gold or not validation:
        raise ValueError("Geographic split requires nonempty Gold training and validation sets")
    train_group_ids = {row["group_id"] for row in train if row.get("group_id")}
    validation_group_ids = {row["group_id"] for row in validation if row.get("group_id")}
    if train_group_ids & validation_group_ids:
        raise ValueError("Supplied real group_id values cross the selected districts. Use the group-aware prepare split command instead.")
    all_gold = [row for row in rows if row["quality"] == GOLD]
    total_languages = Counter(row["language"] for row in all_gold)
    train_languages = Counter(row["language"] for row in train_gold)
    validation_languages = Counter(row["language"] for row in validation)
    actual_fraction = len(validation) / len(all_gold)
    in_band = min_fraction <= actual_fraction <= max_fraction
    missing_train = sorted(set(total_languages) - set(train_languages))
    missing_validation = sorted(set(total_languages) - set(validation_languages))
    warnings = [
        "Geographic holdout only: speaker disjointness is not verified; no speaker IDs were inferred.",
        "District names are matched after Unicode/whitespace/case normalization; spelling aliases are not automatically reconciled.",
        "Holdout selection uses annotation counts and language metadata only, never model results.",
    ]
    if not in_band:
        warnings.append("The selected whole-district split is outside the preferred Gold fraction band. Inspect district sizes before training; this heuristic does not prove that the band is infeasible.")
    if missing_train:
        warnings.append("Some Gold languages are absent from Gold training; inspect language_coverage before training.")
    if missing_validation:
        warnings.append("Some Gold languages are absent from validation. Per-language generalization cannot be measured for them.")
    def group_record(key):
        group = groups[key]
        state, district = sorted(group["variants"])[0]
        return {"state": state, "district": district, "normalized_key": list(key),
                "source_spelling_variants": [list(pair) for pair in sorted(group["variants"])],
                "clips_all_tiers": group["clips"], "gold_clips": group["gold_clips"],
                "gold_language_counts": dict(sorted(group["gold_languages"].items()))}
    digest = hashlib.sha256()
    with manifest.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    source_hash = digest.hexdigest()
    report = {
        "source_manifest": str(manifest), "source_manifest_sha256": source_hash,
        "seed": seed, "split_unit": "source.metadata.state_and_district",
        "geographically_disjoint": True, "speaker_disjoint_verified": False,
        "known_group_id_overlap": [], "group_ids_inferred": False,
        "requested_val_fraction_of_gold": val_fraction,
        "preferred_gold_fraction_band": [min_fraction, max_fraction],
        "actual_val_fraction_of_gold": actual_fraction,
        "preferred_fraction_band_met": in_band,
        "total_geographic_groups": len(groups),
        "gold_geographic_groups": sum(bool(group["gold_clips"]) for group in groups.values()),
        "held_out_groups": [group_record(key) for key in sorted(held_out)],
        "all_geographic_groups": [group_record(key) for key in sorted(groups)],
        "validation_quality_policy": [GOLD],
        "train": audit_rows(train), "train_gold": audit_rows(train_gold),
        "validation": audit_rows(validation), "excluded_heldout_other_tiers": audit_rows(excluded),
        "language_coverage": {
            "missing_from_gold_training": missing_train,
            "missing_from_gold_validation": missing_validation,
            "per_language": [{"language": language, "total_gold": total_languages[language],
                              "train_gold": train_languages[language], "validation_gold": validation_languages[language],
                              "validation_fraction": validation_languages[language] / total_languages[language]}
                             for language in sorted(total_languages)],
        },
        "search": search, "warnings": warnings,
    }
    _write_manifest(output / "train.jsonl", train)
    _write_manifest(output / "val.jsonl", validation)
    _write_manifest(output / "excluded-heldout.jsonl", excluded)
    _write_json(output / "split-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=.125, help="Target fraction of Gold clips")
    parser.add_argument("--min-fraction", type=float, default=.10)
    parser.add_argument("--max-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=512)
    args = parser.parse_args()
    try:
        report = split_geographic(args.manifest, args.output, val_fraction=args.val_fraction,
                                  min_fraction=args.min_fraction, max_fraction=args.max_fraction,
                                  seed=args.seed, search_trials=args.search_trials)
    except Exception as exc:
        parser.exit(1, f"Geographic split failed: {exc}\n")
    compact = {key: report[key] for key in ("split_unit", "speaker_disjoint_verified",
               "actual_val_fraction_of_gold", "preferred_fraction_band_met", "held_out_groups",
               "train", "train_gold", "validation", "excluded_heldout_other_tiers", "language_coverage", "warnings")}
    compact["report"] = str(args.output.resolve() / "split-report.json")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
