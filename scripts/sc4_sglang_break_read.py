#!/usr/bin/env python3
"""Sub-condition 4 on SGLang: does a disk read run *inside* CUDA-graph replay?

Run one arm (or all three) and report the graph-mode storage number that vLLM
0.29.0 could not be made to produce.  See ``scripts/engramdb_sc4_inject.py`` for
the mechanism and for what each self-proof does and does not establish.

    python scripts/sc4_sglang_break_read.py --arm all \\
        --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \\
        --store /root/autodl-tmp/qwen35-ple/qwen38-rows \\
        --layer 1 --json-out probes/sc4_sglang_session43.json

Methodology is deliberately the same as ``scripts/serve_sglang_baseline.py``
(same offline ``sglang.Engine`` path, same prompt construction, same median over
iterations) so the numbers are comparable with the 440.4 tok/s breakable-era
baseline rather than merely plausible next to it.

What each arm is for
--------------------
``none``   engine floor, breakable decode graph, no injection.
``break``  the break runs, delta synthesised from tokens, **no disk I/O**.
``read``   the break performs the real 16-row read from the 47.7 GiB table.

``read - break`` is the disk read's cost inside a graph step; ``break - none``
is the cost of the mechanism that allows it.  Reporting only ``read - none``
would fold the two together and overstate the storage cost.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("none", "break", "read_static", "read")


def extract(o):
    """Pull generated token ids out of one SGLang output dict.

    The key name has moved between versions; an unparsed result would be a
    silent zero, which is worse than an error, so report the real keys instead.
    """
    meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
    for src, name in ((meta, "output_ids"), (meta, "output_token_ids"),
                      (meta, "completion_tokens_ids"), (o, "output_ids"),
                      (o, "token_ids")):
        if isinstance(src, dict) and name in src:
            return list(src[name])
    raise KeyError(
        f"no output token ids found. top-level={sorted(o.keys()) if isinstance(o, dict) else type(o)} "
        f"meta_info={sorted(meta.keys())}"
    )


def run_arm(args) -> dict:
    arm = args.arm
    counters_path = os.path.join(args.tmp, f"sc4_counters_{arm}.json")
    if os.path.exists(counters_path):
        os.unlink(counters_path)
    os.environ["ENGRAMDB_SC4_ARM"] = arm
    os.environ["ENGRAMDB_SC4_LAYER"] = str(args.layer)
    os.environ["ENGRAMDB_SC4_ROWS"] = str(args.rows)
    os.environ["ENGRAMDB_SC4_STORE"] = args.store
    os.environ["ENGRAMDB_SC4_COUNTERS"] = counters_path

    import sglang

    kwargs = dict(
        model_path=args.model,
        mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=8192,
        log_level="warning",
        cuda_graph_backend_decode="breakable",
        # Prefill graph capture is ~26 s of the engine start and it would also
        # run our break, muddying the decode counts.  Decode is the phase the
        # acceptance criterion is about, so prefill is left eager on purpose.
        cuda_graph_backend_prefill=args.prefill_backend,
    )
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    print(f"[{arm}] sglang {sglang.__version__} launching engine ...", flush=True)
    t0 = time.perf_counter()
    try:
        sys.stdout = devnull                      # engine start is very chatty
        engine = sglang.Engine(**kwargs)
    finally:
        sys.stdout = old_stdout
    print(f"[{arm}] engine up in {time.perf_counter() - t0:.1f}s", flush=True)

    if arm in ("read", "read_static") and not args.keep_warm:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from engramdb_sc4_inject import drop_page_cache

            drop_page_cache(args.store)
            print(f"[{arm}] dropped page cache for {args.store}", flush=True)
        except Exception as exc:
            print(f"[{arm}] WARNING could not drop cache: {exc}", flush=True)

    sp = {"temperature": 0.0, "max_new_tokens": args.max_tokens, "ignore_eos": True}
    prompts = [[1000 + (7 * i + j) % 200000 for j in range(args.prompt_len)]
               for i in range(args.batch)]

    runs, first = [], None
    for it in range(args.iterations):
        t0 = time.perf_counter()
        outs = engine.generate(input_ids=prompts, sampling_params=sp)
        dt = time.perf_counter() - t0
        got = [extract(o) for o in outs]
        ntok = sum(len(g) for g in got)
        runs.append({"iter": it, "seconds": dt, "tokens": ntok,
                     "tok_per_s": ntok / dt})
        if first is None:
            first = got
        elif got != first:
            runs[-1]["differs_from_iter0"] = True
        print(f"[{arm}] iter={it} tokens={ntok} s={dt:.3f} tok/s={ntok / dt:.1f}",
              flush=True)

    engine.shutdown()

    counters = None
    deadline = time.time() + 30
    while time.time() < deadline:
        # Every process that imports the model writes <base>.<pid>.json; the one
        # that matters is whichever actually ran the model (see _dump()).
        cands = sorted(glob.glob(counters_path + ".*.json"))
        if cands:
            best = None
            for c in cands:
                try:
                    d = json.loads(Path(c).read_text())
                except Exception:
                    continue
                score = (d.get("break_calls", 0) + d.get("backend_replay_calls", 0)
                         + d.get("forward_calls", 0))
                if best is None or score > best[0]:
                    best = (score, d, c)
            if best and best[0] > 0:
                counters = best[1]
                counters["_counters_file"] = best[2]
                counters["_n_counter_files"] = len(cands)
                break
            if best:
                counters = best[1]
                counters["_counters_file"] = best[2]
                counters["_n_counter_files"] = len(cands)
        time.sleep(0.5)

    vals = sorted(r["tok_per_s"] for r in runs)
    med = vals[len(vals) // 2]
    flat = [t for g in first for t in g]
    return {
        "arm": arm,
        "sglang": sglang.__version__,
        "cuda_graph_backend_decode": "breakable",
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "layer": args.layer,
        "rows_per_token": args.rows,
        "tok_per_s_median": med,
        "tok_per_s_best": vals[-1],
        "ms_per_token": 1000.0 / med,
        "runs": runs,
        "deterministic_within_arm": not any(r.get("differs_from_iter0") for r in runs),
        "first_tokens_sha1": hashlib.sha1(
            json.dumps(first).encode()).hexdigest()[:16],
        "first_tokens_head": flat[:12],
        "counters": counters,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=(*ARMS, "all"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.55)
    ap.add_argument("--prefill-backend", default="disabled",
                    choices=("disabled", "full", "breakable", "tc_piecewise"),
                    help="decode is always breakable; prefill defaults to eager"
                         " so the engine starts fast and decode counts stay clean")
    ap.add_argument("--keep-warm", action="store_true")
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.arm == "all":
        results = []
        for arm in ARMS:
            cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
                   "--model", args.model, "--store", args.store,
                   "--layer", str(args.layer), "--rows", str(args.rows),
                   "--iterations", str(args.iterations),
                   "--prompt-len", str(args.prompt_len),
                   "--max-tokens", str(args.max_tokens), "--batch", str(args.batch),
                   "--mem-fraction-static", str(args.mem_fraction_static),
                   "--prefill-backend", args.prefill_backend,
                   "--tmp", args.tmp]
            if args.keep_warm:
                cmd.append("--keep-warm")
            out = os.path.join(args.tmp, f"sc4_{arm}.json")
            cmd += ["--json-out", out]
            print(f"\n===== arm {arm} =====", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"arm {arm} FAILED rc={r.returncode}")
                return r.returncode
            results.append(json.loads(Path(out).read_text()))
        summary = compare(results)
        finish(summary, args)
        return 0

    res = run_arm(args)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n[{args.arm}] median {res['tok_per_s_median']:.1f} tok/s "
          f"= {res['ms_per_token']:.2f} ms/token")
    c = res.get("counters")
    if c:
        print(f"[{args.arm}] forward_calls={c.get('forward_calls')} "
              f"by_class={c.get('forward_by_class')} "
              f"wrapped={c.get('wrapped_classes')}")
        print(f"[{args.arm}] break_calls={c['break_calls']} "
              f"in_replay={c['break_calls_in_replay']} "
              f"in_capture={c['break_calls_in_capture']} "
              f"backend_replay={c['backend_replay_calls']} "
              f"bcg_replay={c['bcg_replay_calls']} "
              f"rows={c['rows_read']}")
        print(f"[{args.arm}] read_us_median={c.get('read_us_median')} "
              f"break_us_median={c.get('break_us_median')} "
              f"process={(c.get('process') or '')[:110]}")
        if c.get("skipped"):
            print(f"[{args.arm}] SKIPPED: {c['skipped'][:3]}")
        if c.get("errors"):
            print(f"[{args.arm}] ERRORS: {c['errors'][:3]}")
    else:
        print(f"[{args.arm}] NO COUNTERS FOUND under "
              f"{os.path.join(args.tmp, f'sc4_counters_{args.arm}.json')}.*.json")
    return 0


def compare(results: list[dict]) -> dict:
    by = {r["arm"]: r for r in results}
    out = {"arms": {r["arm"]: {
        "tok_per_s_median": r["tok_per_s_median"],
        "ms_per_token": r["ms_per_token"],
        "first_tokens_sha1": r["first_tokens_sha1"],
        "deterministic_within_arm": r["deterministic_within_arm"],
        "counters": r.get("counters"),
    } for r in results}}

    if "none" in by and "break" in by:
        out["break_minus_none_ms"] = by["break"]["ms_per_token"] - by["none"]["ms_per_token"]
    if "break" in by and "read" in by:
        out["read_minus_break_ms"] = by["read"]["ms_per_token"] - by["break"]["ms_per_token"]
    if "break" in by and "read_static" in by:
        # disk + H2D + add, with the device-to-host sync removed entirely
        out["readstatic_minus_break_ms"] = (by["read_static"]["ms_per_token"]
                                            - by["break"]["ms_per_token"])
    if "read_static" in by and "read" in by:
        # the D2H + rowid-computation term, isolated
        out["read_minus_readstatic_ms"] = (by["read"]["ms_per_token"]
                                           - by["read_static"]["ms_per_token"])
    if "none" in by and "read" in by:
        out["read_minus_none_ms"] = by["read"]["ms_per_token"] - by["none"]["ms_per_token"]
        out["read_pct_of_step"] = (out["read_minus_none_ms"]
                                   / by["none"]["ms_per_token"] * 100)

    # functional self-proof: an injected arm that matched the baseline would mean
    # the break never reached the model at replay.
    if "none" in by:
        for arm in ("break", "read_static", "read"):
            if arm in by:
                out[f"{arm}_differs_from_none"] = (
                    by[arm]["first_tokens_sha1"] != by["none"]["first_tokens_sha1"])
    return out


def finish(summary: dict, args) -> None:
    print("\n================ SUMMARY ================")
    for arm, v in summary["arms"].items():
        c = v.get("counters") or {}
        print(f"  {arm:6s} {v['tok_per_s_median']:8.1f} tok/s  "
              f"{v['ms_per_token']:6.3f} ms/tok  "
              f"break_in_replay={c.get('break_calls_in_replay', '-')}  "
              f"backend_replay={c.get('backend_replay_calls', '-')}")
    for k in ("break_minus_none_ms", "readstatic_minus_break_ms",
              "read_minus_readstatic_ms", "read_minus_break_ms",
              "read_minus_none_ms", "read_pct_of_step"):
        if k in summary:
            print(f"  {k:28s} {summary[k]:.4f}")
    for k in ("break_differs_from_none", "read_static_differs_from_none",
              "read_differs_from_none"):
        if k in summary:
            print(f"  {k:28s} {summary[k]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    raise SystemExit(main())
