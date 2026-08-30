#!/usr/bin/env python3
"""Check stored CPU decode baseline CSVs against regression thresholds.

This is the explicit regression gate for the two benchmark suites:

* ``probes/cpu_tiny_baseline.csv``: tiny Qwen3 toy model
* ``probes/qwen35_cpu_baseline.csv``: real Qwen3.5-0.8B on WSL CPU

For each file the script compares disk variants against the in-memory median
throughput and reports the slowdown.  Thresholds can be overridden on the
command line; defaults are intentionally loose first-pass gates so the current
known baseline passes while still catching order-of-magnitude regressions.

Example:

    python scripts/decode_baseline_check.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def slowdown(memory_tok_s: float, disk_tok_s: float) -> float:
    if disk_tok_s <= 0:
        return float("inf")
    return max(0.0, memory_tok_s / disk_tok_s - 1.0)


def check_file(
    path: Path,
    raw_limit: float,
    lru_limit: float,
) -> tuple[bool, list[str]]:
    rows = load_rows(path)
    by_label = {r["label"]: r for r in rows}
    if "memory" not in by_label:
        return False, [f"{path}: missing memory row"]
    mem_tok = float(by_label["memory"]["median_tok_s"])

    checks: list[str] = []
    ok = True
    for label in sorted(by_label):
        if not label.startswith("disk-"):
            continue
        disk_tok = float(by_label[label]["median_tok_s"])
        s = slowdown(mem_tok, disk_tok)
        if label == "disk-raw-cache0":
            limit = raw_limit
        else:
            limit = lru_limit
        status = "PASS" if s <= limit else "FAIL"
        if s > limit:
            ok = False
        checks.append(f"{status}: {path.name} {label} slowdown={s:.1%} (limit {limit:.1%})")
    return ok, checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny-csv", default="probes/cpu_tiny_baseline.csv")
    ap.add_argument("--qwen-csv", default="probes/qwen35_cpu_baseline.csv")
    ap.add_argument("--tiny-raw-limit", type=float, default=0.50)
    ap.add_argument("--tiny-lru-limit", type=float, default=0.75)
    ap.add_argument("--qwen-raw-limit", type=float, default=0.50)
    ap.add_argument("--qwen-lru-limit", type=float, default=0.50)
    args = ap.parse_args()

    all_ok = True
    for csv_path, raw_limit, lru_limit in [
        (Path(args.tiny_csv), args.tiny_raw_limit, args.tiny_lru_limit),
        (Path(args.qwen_csv), args.qwen_raw_limit, args.qwen_lru_limit),
    ]:
        if not csv_path.exists():
            print(f"SKIP: {csv_path} does not exist")
            continue
        ok, checks = check_file(csv_path, raw_limit, lru_limit)
        all_ok = all_ok and ok
        for line in checks:
            print(line)

    if not all_ok:
        raise SystemExit("GATE FAIL: decode baseline regression thresholds exceeded")
    print("GATE PASS: decode baseline thresholds satisfied")


if __name__ == "__main__":
    main()
