#!/usr/bin/env python3
"""Compute the closure rate of the **live ledger** -- and refuse to fake it.

Why this exists
---------------
roadmap §36.6 第六条 used to say "count research in the net-closure rate; two
negative rounds in a row means stop and subtract".  The diagnosis was right
(Sessions 42-43 really did add three documents and close zero acceptance items)
but it picked the wrong denominator: it counted every checkbox in the roadmap,
including ~40 generations of superseded plans.  That ratio can *only* go
negative -- it measures how many dead plans nobody has deleted yet -- so it was
retired in Session 44 and the ledger was consolidated (roadmap §0).

The replacement rule needs to be mechanical, or it will drift the same way.  So
this script defines the live ledger as exactly two tables in README §6.1 and
prints the rate.  Anything genuinely required must appear in one of those tables
(roadmap §36.5 holds the priority side).

Classification
--------------
A row is PARTIAL if it carries ⚠️ *anywhere in the status cell*, else DONE if it
carries ✅, else OPEN.  ⚠️ wins over ✅ on purpose: several rows cross-reference a
closed item while still being open themselves, and a naive ✅-first check
miscounts them (it did, the first time this was done by hand).

Usage::

    python scripts/ledger_rate.py                 # print the rate
    python scripts/ledger_rate.py --max-open 6    # fail if more than 6 open rows
    python scripts/ledger_rate.py --json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

ACCEPT_HEADER = "### 6.1 验收目标"
SUB_HEADER = "#### 「GPU 端 A/B ≤5%」的六条子条件"


def _table_rows(lines: list[str], start: int) -> list[tuple[int, str]]:
    """Rows of the first markdown table at or after ``start``.

    Stops at the first non-``|`` line *after* the table has begun, so trailing
    tables in the same section are not swallowed -- the bug that made the first
    hand count include the medium table, the doc-nav table and the geometry
    table.
    """
    rows, begun = [], False
    for n in range(start, len(lines)):
        line = lines[n]
        if line.startswith("|"):
            begun = True
            if re.match(r"^\|[\s:|-]+\|$", line):
                continue                      # separator
            rows.append((n + 1, line))
        elif begun:
            break
    return rows


def _first_cell(row: str) -> str:
    return row.strip().strip("|").split("|")[0].strip()


def _status_cell(row: str) -> str:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    return cells[-1] if cells else ""


def classify(readme: Path = README) -> dict:
    lines = readme.read_text().splitlines()

    def find(header: str) -> int:
        for n, line in enumerate(lines):
            if line.startswith(header):
                return n
        raise SystemExit(f"header not found in {readme}: {header!r}")

    acc_start = find(ACCEPT_HEADER)
    sub_start = find(SUB_HEADER)
    acc = [(n, r) for n, r in _table_rows(lines, acc_start) if "指标" not in r]
    # sub-condition rows are numbered (`| 1a |`, `| 4′ |`, ...)
    sub = [(n, r) for n, r in _table_rows(lines, sub_start)
           if re.match(r"^\d", _first_cell(r))]

    def bucket(rows):
        out = {"done": [], "partial": [], "open": []}
        for n, r in rows:
            cell = _status_cell(r)
            key = "partial" if "⚠️" in cell else ("done" if "✅" in cell else "open")
            out[key].append((n, r))
        return out

    a, s = bucket(acc), bucket(sub)
    tot = {k: len(a[k]) + len(s[k]) for k in a}
    n_total = sum(tot.values())
    return {
        "readme": str(readme),
        "acceptance_rows": len(acc),
        "sub_condition_rows": len(sub),
        "total": n_total,
        "done": tot["done"],
        "partial": tot["partial"],
        "open": tot["open"],
        "closed_rate_pct": round(tot["done"] / n_total * 100, 1) if n_total else 0.0,
        "not_open_rate_pct": round((tot["done"] + tot["partial"]) / n_total * 100, 1)
        if n_total else 0.0,
        "open_rows": [{"line": n, "text": r[:150]} for n, r in a["open"] + s["open"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--max-open", type=int, default=None,
                    help="exit 1 if the number of fully-open rows exceeds this")
    args = ap.parse_args()

    d = classify()
    if args.json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
    else:
        print(f"live ledger = {d['readme']} §6.1")
        print(f"  acceptance rows    : {d['acceptance_rows']}")
        print(f"  sub-condition rows : {d['sub_condition_rows']}")
        print(f"  total              : {d['total']}")
        print(f"  ✅ done            : {d['done']}")
        print(f"  ⚠️ partial         : {d['partial']}")
        print(f"  ❌ open            : {d['open']}")
        print(f"  closed rate        : {d['closed_rate_pct']}%")
        print(f"  not-open rate      : {d['not_open_rate_pct']}%")
        if d["open_rows"]:
            print("\n  fully-open rows:")
            for r in d["open_rows"]:
                print(f"    L{r['line']}: {r['text']}")

    if args.max_open is not None and d["open"] > args.max_open:
        print(f"\nFAIL: {d['open']} fully-open rows > --max-open {args.max_open}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
