#!/usr/bin/env python3
"""
Golden regression: the parser and metrics must reproduce the stored v0.1
results exactly.

    python validate.py        exit 0 if every stored metric reproduces, else 1

For each legacy/results_*.json this re-extracts metrics from the stored raw
text and compares every v0.1 metric field. It never rewrites those files.

History: the v0.1 version of this script printed "expected" values that
disagreed with its own output (e.g. baseline commitment latency "expected
0.40", measured 0.30) and never asserted anything, so it could not fail.
These are real assertions against the stored measurements.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from harness import extract_metrics

V01_KEYS = [
    "commitment_latency", "assume_count", "claim_count", "branch_count",
    "select_count", "conclude_count", "total_checkpoints",
    "tokens_per_checkpoint", "claim_select_ratio", "total_tokens",
]


def validate(directory: Path = Path(__file__).parent / "legacy") -> bool:
    files = sorted(directory.glob("results_*.json"))
    if not files:
        print(f"no results_*.json in {directory}", file=sys.stderr)
        return False
    ok = True
    for f in files:
        stored = json.loads(f.read_text(encoding="utf-8"))
        fresh = extract_metrics(stored["raw_text"], stored["variant"]).metrics
        bad = []
        for k in V01_KEYS:
            a, b = stored["metrics"][k], getattr(fresh, k)
            if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)):
                bad.append(f"{k}: stored={a} now={b}")
        n_cp = len(stored["checkpoints"])
        if n_cp != fresh.total_checkpoints:
            bad.append(f"checkpoint count: stored={n_cp} now={fresh.total_checkpoints}")
        print(f"  {'PASS' if not bad else 'FAIL'}  {f.name}")
        for line in bad:
            print(f"        {line}")
        ok &= not bad
    return ok


if __name__ == "__main__":
    print("GOLDEN REGRESSION (stored v0.1 results)")
    print("=" * 60)
    passed = validate()
    print("=" * 60)
    print("ALL REPRODUCED" if passed else "MISMATCH: parser/metric behavior changed")
    sys.exit(0 if passed else 1)
