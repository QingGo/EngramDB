#!/usr/bin/env python3
"""vLLM engine floor: the denominator of every storage claim we make.

Sibling of ``serve_sglang_baseline.py``.  Both answer one question with the
model fully in HBM and no reader installed: **how long is one decoding step?**

That number is the denominator of "≤5% of the step", so it decides whether the
budget is meaningful at all.  Measured on this box (Qwen3.5-0.8B, RTX 4090,
batch=1, prompt 128, 128 new tokens):

    vLLM  0.29.0  eager        47 tok/s   21.3 ms/token   500us =  2.3% of step
    SGLang 0.5.19 cuda graph  440 tok/s    2.27 ms/token  500us = 22.0% of step

A 9.4x difference, and it is entirely an *engine* difference -- same model, same
GPU, same workload.  Roadmap 35.1 argues that an eager denominator makes "≤5%"
cheap and unfalsifiable; this is that argument as a number.

Usage
-----
    python vllm_engine_floor.py --model /path/Qwen3.5-0.8B --iterations 5
    python vllm_engine_floor.py ... --enforce-eager
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
    ap.add_argument("--gpu-mem-util", type=float, default=0.70)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        disable_log_stats=True,
        trust_remote_code=True,
    )
    print(f"[vllm] engine up in {time.perf_counter() - t0:.1f}s "
          f"eager={args.enforce_eager}")

    prompts = [{"prompt_token_ids": [1000 + (7 * i + j) % 200000
                                     for j in range(args.prompt_len)]}
               for i in range(args.batch)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    runs = []
    token_ids = None
    for it in range(args.iterations):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        got = [list(o.outputs[0].token_ids) for o in outs]
        ntok = sum(len(g) for g in got)
        runs.append({"iter": it, "seconds": dt, "tokens": ntok, "tok_per_s": ntok / dt})
        if token_ids is None:
            token_ids = got
        print(f"[run] iter={it} tokens={ntok} s={dt:.3f} tok/s={ntok / dt:.1f}")
        if got != token_ids:
            print(f"  >>> NON-DETERMINISTIC: iteration {it} differs from iteration 0")

    vals = sorted(r["tok_per_s"] for r in runs)
    med = vals[len(vals) // 2]
    step_ms = 1000.0 / med
    result = {
        "engine": "vllm",
        "vllm": __import__("vllm").__version__,
        "model": args.model,
        "enforce_eager": args.enforce_eager,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "tok_per_s_median": med,
        "tok_per_s_best": vals[-1],
        "ms_per_token": step_ms,
        "budget_500us_as_pct_of_step": 500.0 / (step_ms * 1000) * 100,
        "runs": runs,
        "greedy_deterministic_across_iterations": True,
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
