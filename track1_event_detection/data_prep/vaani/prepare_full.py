"""Prepare each verified-size shard as the pinned download makes it available (T1_WORK/vaani-release ->
T1_WORK/vaani-full: manifest.jsonl + 16 kHz WAV audio; rows that cannot be decoded are quarantined)."""
from pathlib import Path
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("T1_WORK", "."))
sys.path.insert(0, str(HERE.parents[1] / "atst_red"))
from indoml.prepare import prepare_parquets

metadata = json.loads((HERE / "full-release-source.json").read_text())
started = time.time()

def available_shards():
    for number, entry in enumerate(metadata["files"], 1):
        path = ROOT / "vaani-release" / entry["path"]
        deadline = time.monotonic() + 1800
        while not path.exists() or path.stat().st_size != entry["size"]:
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for shard " + entry["path"])
            time.sleep(1)
        print(json.dumps({"preparing_shard": number, "total_shards": len(metadata["files"]),
                          "path": entry["path"]}), flush=True)
        yield path, entry["path"]

report = prepare_parquets(available_shards(), ROOT / "vaani-full",
                          dataset=metadata["dataset"], revision=metadata["revision"],
                          invalid_rows="quarantine")
print(json.dumps({"complete": True, "clips": report["clips"],
                  "seconds": round(time.time() - started, 2),
                  "quality_counts": report["quality_counts"]}), flush=True)
