#!/usr/bin/env bash
# Q2: the 2,849 candidate kernels (natural cuts of labelled validation natural events) over
#   1. validation synthetic, generator-legal free regions (null, exact analogue of the test search)
#   2. validation synthetic, every lag (null incl. pastes of other noise files)
#   3. the 2,082 synthetic-routed test clips, free regions of the WMF file (application)
# 3 workers, one run after the other. Thresholds come only from runs 1-2.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
$PY kjob.py cand valsyn free 3 8
$PY kjob.py cand valsyn full 3 8
$PY kjob.py cand testsyn free 3 8
echo ALL_DONE
