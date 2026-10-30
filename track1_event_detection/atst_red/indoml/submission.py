"""Validate clip coverage and timestamps, then create the Codabench ZIP."""

import argparse
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from .metrics import event_map


def validate_predictions(records, expected):
    """Expected rows require clip_id and measured duration; labels not needed."""
    pred = event_map(records)
    ids, durations = set(), {}
    for row in expected:
        clip_id, duration = row["clip_id"], row["duration"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in ids:
            raise ValueError(f"Invalid or duplicate expected clip_id: {clip_id!r}")
        if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid audio duration for {clip_id}")
        ids.add(clip_id)
        durations[clip_id] = duration
    if not ids:
        raise ValueError("Expected manifest has no clips")
    missing, extra = ids - pred.keys(), pred.keys() - ids
    if missing or extra:
        raise ValueError(f"Clip coverage mismatch: missing={sorted(missing)[:10]}, "
                         f"extra={sorted(extra)[:10]}")
    for clip_id, events in pred.items():
        for _, offset in events:
            if offset > durations[clip_id] + 1e-6:
                raise ValueError(f"{clip_id}: event extends past audio duration")
    return {"clips": len(pred), "events": sum(map(len, pred.values())),
            "empty_clips": sum(not v for v in pred.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="All evaluation clip IDs and measured durations")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows, expected = read(args.predictions), read(args.manifest)
    report = validate_predictions(rows, expected)
    by_id = {row["clip_id"]: row["events"] for row in rows}
    payload = "".join(json.dumps({"clip_id": row["clip_id"], "events": [
        {"onset": event["onset"], "offset": event["offset"]}
        for event in by_id[row["clip_id"]]]}, allow_nan=False) + "\n" for row in expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("predictions.jsonl", payload)
    with ZipFile(args.output) as archive:
        assert archive.namelist() == ["predictions.jsonl"]
        assert archive.testzip() is None
    print(json.dumps({**report, "zip": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
