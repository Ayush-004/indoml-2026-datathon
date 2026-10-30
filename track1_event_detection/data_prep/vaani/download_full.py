"""Fetch the pinned Vaani training release (182 parquet shards) into the work folder T1_WORK (default: .).

Uses the standard Hugging Face login (huggingface-cli login) if the dataset asks for one.
"""
from pathlib import Path
import json
import os
import time

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("T1_WORK", "."))
metadata = json.loads((HERE / "full-release-source.json").read_text())
started = time.time()
print(json.dumps({"started": started, "revision": metadata["revision"],
                  "shards": len(metadata["files"]), "expected_bytes": metadata["expected_bytes"]}), flush=True)
snapshot_download(metadata["dataset"], repo_type="dataset", revision=metadata["revision"],
                  allow_patterns=[x["path"] for x in metadata["files"]],
                  local_dir=ROOT / "vaani-release", max_workers=8)
errors = []
for entry in metadata["files"]:
    path = ROOT / "vaani-release" / entry["path"]
    if not path.exists() or path.stat().st_size != entry["size"]:
        errors.append(entry["path"])
if errors:
    raise SystemExit("Download size verification failed for " + ", ".join(errors))
status = {"complete": True, "shards": len(metadata["files"]),
          "bytes": metadata["expected_bytes"], "seconds": round(time.time() - started, 2),
          "revision": metadata["revision"]}
(ROOT / "full-download-status.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status), flush=True)
