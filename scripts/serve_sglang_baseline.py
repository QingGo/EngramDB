#!/usr/bin/env python3
"""SGLang baseline: engine-overhead floor on a second engine.

Why this exists
---------------
README §6.1 sub-condition 6 asks for a multi-engine number.  The PLE variant of
that question turned out to be **structurally impossible**, not merely undone:
SGLang 0.5.19 has no ``qwen4_exp`` and no ``ple_layer_ids`` anywhere, so there
is no PLE seam to attach to (see ``probes/ple_rowid_exactness_session42.md`` §5).

What *is* measurable on SGLang is the same thing we measured on vLLM: how much
of a decoding step is the engine itself, with the model fully in HBM.  That
number is the denominator of every storage claim we make, and having it from two
engines is what makes the "≤5% of the step" budget falsifiable rather than
vLLM-specific.

It also gives the CUDA-graph contrast directly: vLLM's 45-47 tok/s numbers are
``enforce_eager=True`` (a Python reader cannot enter a graph).  Running SGLang
with graphs on *and* off brackets how much of the step is graph-eligible work --
see roadmap §35.1 on why an eager denominator makes "≤5%" cheap.

Honest scope
------------
No disk arm.  This measures the engine floor only.  Injecting a disk reader into
SGLang requires a ``sitecustomize``/import-hook because SGLang starts its
scheduler with ``mp.set_start_method("spawn")``, so parent-process monkeypatches
are not inherited -- that is a separate piece of work, and the rowid and
embedding correctness it would test are already established on vLLM.

Usage
-----
    python serve_sglang_baseline.py --model /path/Qwen3.5-0.8B --iterations 5
    python serve_sglang_baseline.py ... --disable-cuda-graph
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.70)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import sglang

    kwargs = dict(
        # NOTE: the field is `model_path`, not `model` -- ServerArgs has 488
        # fields and the HTTP-server spelling differs from the CLI flag name.
        model_path=args.model,
        mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=8192,
        disable_cuda_graph=args.disable_cuda_graph,
        log_level="warning",
    )
    print(f"[sglang] version={sglang.__version__} cuda_graph={not args.disable_cuda_graph}")
    t0 = time.perf_counter()
    engine = sglang.Engine(**kwargs)
    print(f"[sglang] engine up in {time.perf_counter() - t0:.1f}s")

    sp = {"temperature": 0.0, "max_new_tokens": args.max_tokens, "ignore_eos": True}
    prompts = [[1000 + (7 * i + j) % 200000 for j in range(args.prompt_len)]
               for i in range(args.batch)]

    runs = []
    token_ids = None

    def extract(o):
        """Pull generated token ids out of one SGLang output dict.

        The key name has moved between versions; rather than guess, look for the
        first plausible name and report the real keys once if none is found --
        an unparsed result is a silent zero, which is worse than an error.
        """
        meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
        for src, name in ((meta, "output_ids"), (meta, "output_token_ids"),
                          (meta, "completion_tokens_ids"), (o, "output_ids"),
                          (o, "token_ids")):
            if isinstance(src, dict) and name in src:
                return list(src[name])
        raise KeyError(
            f"no output token ids found. top-level keys={sorted(o.keys()) if isinstance(o, dict) else type(o)} "
            f"meta_info keys={sorted(meta.keys())}"
        )

    for it in range(args.iterations):
        t0 = time.perf_counter()
        outs = engine.generate(input_ids=prompts, sampling_params=sp)
        dt = time.perf_counter() - t0
        got = [extract(o) for o in outs]
        ntok = sum(len(g) for g in got)
        runs.append({"iter": it, "seconds": dt, "tokens": ntok, "tok_per_s": ntok / dt})
        if token_ids is None:
            token_ids = got
        print(f"[run] iter={it} tokens={ntok} s={dt:.3f} tok/s={ntok / dt:.1f}")
        if token_ids is not None and got != token_ids:
            print(f"  >>> NON-DETERMINISTIC: iteration {it} differs from iteration 0")

    engine.shutdown()

    vals = sorted(r["tok_per_s"] for r in runs)
    med = vals[len(vals) // 2]
    step_ms = 1000.0 / med
    result = {
        "engine": "sglang",
        "sglang": sglang.__version__,
        "model": args.model,
        "cuda_graph": not args.disable_cuda_graph,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "tok_per_s_median": med,
        "tok_per_s_best": vals[-1],
        "ms_per_token": step_ms,
        "budget_500us_as_pct_of_step": 500.0 / (step_ms * 1000) * 100,
        "runs": runs,
        "greedy_deterministic_across_iterations": token_ids is not None,
    }
    print()
    print(f"  median {med:.1f} tok/s  = {step_ms:.2f} ms/token")
    print(f"  500 us budget = {result['budget_500us_as_pct_of_step']:.2f}% of one step")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
