#!/usr/bin/env python3
"""Sub-condition 4, stage 1: get the disk read into a CUDA-graph run at all.

Why the injection has to change shape
-------------------------------------
The eager A/B (``serve_ple_ab.py``) injected with a **forward hook on layer 2
that returned a modified output tensor**.  That cannot survive CUDA graphs:
during replay the graph executes captured kernels and never calls the hook.

Getting the patch *seen by the trace* is the whole game.  Three measurements pin
it down, each from a failed run of this script:

  * Patch the layer **instance** after ``LLM(...)``: nothing happens, silently.
  * Patch the **class** before ``LLM(...)`` but leave vLLM's compile cache on:
    still nothing.  ``~/.cache/vllm/torch_compile_cache`` held an 87 MB artifact
    compiled by an *unpatched* run, and every later run reused it.  Hence
    ``VLLM_DISABLE_COMPILE_CACHE=1`` below.
  * Put a **counter increment inside the traced forward**: torch refuses --
    "Assigning / modifying buffers of nn.Module during forward pass is not
    allowed when using cudagraph inside the compiler because it will cause
    silent errors."

So the traced forward is **side-effect free**, and the reader arm's delta is
added **unconditionally**.  It has to be: during the profiling trace
``inj.reader`` is still ``None``, so any ``if reader is None: return`` early-exit
would keep the add out of the captured graph entirely.  The no-reader arm is
made a true baseline by **zeroing the buffer** instead, so the two arms are
structurally identical and differ only in whether the disk read happens.

Honest scope of stage 1
-----------------------
The disk read stays **eager and serial** with the step.  This measures "storage
cost against a real graph-mode step" -- what sub-condition 4 asks for -- but it
does **not** show the read hidden inside a graph gap.  That is stage 2
(register the read as a splitting op the way
``vllm::qwen4_exp_compute_ple_ngram_ids`` does).

Self-proof (V175)
-----------------
No Python runs at replay, so a counter can never prove the *graph* consumed the
buffer -- and a counter inside the trace is rejected outright.  A functional test
can, and it did catch both silent failures:

  * the reader arm's greedy tokens must **differ** from the no-reader arm.  If
    the graph ignored the buffer they would be identical.
  * the reader arm must still be **deterministic across iterations**.

Mode is checkable from tok/s alone on this box: ~47 is eager, ~340-410 is graph.

Usage
-----
    python serve_ple_ab_graph.py --arms none,engram-i,none --iterations 3 --cold
    python serve_ple_ab_graph.py --arms none --eager       # control
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# vLLM persists compiled graphs under ~/.cache/vllm/torch_compile_cache and
# reuses them when the compilation hash matches.  A cache entry produced by a
# run WITHOUT the patch therefore silently defeats every later patched run:
# the replay never calls our code and the injection vanishes.  Both failed
# attempts of this script were explained by this (the cache was written by an
# unpatched run at 23:53 and reused afterwards).  Never measure with it on.
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from serve_ple_ab import (  # noqa: E402
    MODEL,
    ROWS_DIR,
    ROW_WIDTH,
    EngramStoreReader,
    MmapReader,
    ShmReader,
    drop_table_cache,
    find_model,
    verify_cold,
)

MAX_TOKENS = 4096
SHM_BUDGET_BYTES = 20 * 1024**3

# Eager-only instrumentation.  It MUST stay off in graph mode: any side effect
# inside the traced forward is rejected by torch's cudagraph-safety check.
_DIAG: dict = {"on": False, "done": False}

# "buffer" = stage 1 (plain add of a static buffer -- works eager, folded away
#            by Inductor under torch.compile)
# "op"     = stage 2 (registered custom op writing into a graph-created tensor,
#            listed in splitting_ops -- the mechanism the engine itself uses)
_MODE: dict = {"read": "buffer", "pending_tokens": None}
OP_NAME = "engramdb_ple_read"


def install_read_op():
    """Register the read as a vLLM custom op, before the engine is built.

    This copies ``vllm::qwen4_exp_compute_ple_ngram_ids`` (ple_layer.py:1191):
    a function that takes a tensor the *graph* created, mutates it in place, and
    is declared ``mutates_args=["output"]`` with a no-op fake impl.  Listing it
    in ``splitting_ops`` makes vLLM split the piecewise graph there and run it
    **eagerly** -- which is what lets it do disk I/O at all.

    Why stage 1's static buffer could not work: a tensor that is a Python
    closure variable or an ``nn.Module`` buffer becomes a constant/``get_attr``
    for Inductor, which folds ``hidden_states + 0`` away and never reads the
    original address again.  Only an op-visible, graph-created tensor survives.
    """
    from vllm.utils.torch_utils import direct_register_custom_op

    def _read(output: torch.Tensor, tag: str) -> None:
        inj = _STATE["inj"]
        reader = inj.reader if inj is not None else None
        toks = _MODE["pending_tokens"]
        if reader is None or not toks:
            output.zero_()
            return
        e = reader.fetch(toks)                  # [n, out_dim] float32 cpu
        out_dim = e.shape[1]
        n = min(output.shape[0], e.shape[0])
        proj = inj.proj
        delta = (e[:n].to(output.device, dtype=torch.float32) @ proj).to(output.dtype)
        output[:n].copy_(delta)
        if n < output.shape[0]:
            output[n:].zero_()
        inj.counters["op_calls"] = inj.counters.get("op_calls", 0) + 1
        inj.counters["op_rows"] = inj.counters.get("op_rows", 0) + n

    def _fake(output: torch.Tensor, tag: str) -> None:
        return

    direct_register_custom_op(
        op_name=OP_NAME,
        op_func=_read,
        mutates_args=["output"],
        fake_impl=_fake,
    )


def default_splitting_ops():
    """The engine's own default list, plus ours.

    Passing an explicit ``splitting_ops`` bypasses the ``is None`` branch that
    would also append the kv-cache-update ops, so reproduce them here or the
    compiled graph stops matching the baseline it is compared against.
    """
    from vllm.config.compilation import CompilationConfig

    return list(CompilationConfig._attention_ops) + [
        "vllm::unified_kv_cache_update",
        "vllm::unified_mla_kv_cache_update",
        f"vllm::{OP_NAME}",
    ]


def hs_shape_of(out):
    hs = out[0] if isinstance(out, tuple) else out
    return hs.shape

# Module-level because the layer patch must exist BEFORE LLM() is constructed.
_STATE: dict = {"inj": None}
_ORIG_LAYER_FORWARD = None
_ORIG_LAYER_INIT = None
_PATCHED = False


def install_layer_patch():
    """Wrap the decoder-layer forward at CLASS level, before the engine exists."""
    global _ORIG_LAYER_FORWARD, _ORIG_LAYER_INIT, _PATCHED
    if _PATCHED:
        return
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    _ORIG_LAYER_FORWARD = Qwen3_5DecoderLayer.forward
    _ORIG_LAYER_INIT = Qwen3_5DecoderLayer.__init__

    def patched_init(self, *a, **kw):
        _ORIG_LAYER_INIT(self, *a, **kw)
        inj = _STATE["inj"]
        if inj is None or getattr(self, "layer_idx", None) != inj.layer_idx:
            return
        # Register the shared tensor as a real module buffer.
        #
        # A plain closure tensor does NOT work: at trace time the buffer is all
        # zeros, so Inductor folds `hidden_states + 0` away and the add never
        # reaches the captured graph.  The run compiles cleanly, reports healthy
        # tok/s, and the reader arm comes out bit-identical to baseline.  As a
        # module buffer Dynamo must treat it as an input.
        #
        # persistent=False keeps it out of state_dict so weight loading is
        # unaffected.  vLLM sets cudagraph_copy_inputs=False, so replay reads
        # this exact address -- which is what the eager fill writes into.
        self.register_buffer("_engram_buf", inj.buf, persistent=False)

    def patched(self, hidden_states, residual, positions, **kwargs):
        out = _ORIG_LAYER_FORWARD(self, hidden_states, residual, positions, **kwargs)
        inj = _STATE["inj"]
        if inj is None or self.layer_idx != inj.layer_idx:
            return out
        buf = getattr(self, "_engram_buf", None)
        if _DIAG["on"] and not _DIAG["done"]:
            _DIAG["done"] = True
            print(f"[diag] layer={self.layer_idx} hs.shape={tuple(hs_shape_of(out))} "
                  f"hidden={inj.hidden} buf={'yes' if buf is not None else 'NO'} "
                  f"buf_shape={tuple(buf.shape) if buf is not None else None}")
        if buf is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        # NO side effects in this function.  torch's cudagraph-safety check
        # rejects mutating state during a forward pass:
        #   "Assigning / modifying buffers of nn.Module during forward pass is
        #    not allowed when using cudagraph inside the compiler because it
        #    will cause silent errors."
        # Proof that the graph consumed the buffer therefore comes from a
        # functional test -- the reader arm's tokens must differ -- not from a
        # counter.  Mutating a counter here cost one full run.
        if hs.shape[-1] != inj.hidden:
            return out
        # generic over leading dims: layer output is not always 2-D
        flat = hs.reshape(-1, hs.shape[-1])
        n = flat.shape[0]
        if n > MAX_TOKENS:
            return out
        if _MODE["read"] == "op":
            # Stage 2: a graph-created tensor the registered op mutates.  Being
            # an op argument is what keeps it out of Inductor's constant pool.
            delta = torch.empty(n, inj.hidden, dtype=flat.dtype, device=flat.device)
            torch.ops.vllm.engramdb_ple_read(delta, "ple")
        else:
            delta = buf[:n].to(flat.dtype)
        flat = flat + delta
        hs = flat.reshape(hs.shape)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    # Qwen3_5DecoderLayer defines neither of its own, so setting them here
    # shadows Qwen3NextDecoderLayer's for this model only.
    Qwen3_5DecoderLayer.__init__ = patched_init
    Qwen3_5DecoderLayer.forward = patched
    _PATCHED = True


class GraphInjector:
    """Eager fill into a static buffer; the compiled graph consumes that buffer."""

    def __init__(self, layer_idx: int, out_dim: int, hidden: int, dtype):
        self.layer_idx = layer_idx
        self.out_dim = out_dim
        self.hidden = hidden
        self.reader = None
        self.emb = None
        self._orig_emb_forward = None
        # Projection never enters the graph, so plain float32 is fine.
        self.proj = torch.randn(out_dim, hidden, device="cuda",
                                dtype=torch.float32) * (hidden ** -0.5)
        # STATIC buffer.  One address for the whole run; the graph captures it.
        self.buf = torch.zeros(MAX_TOKENS, hidden, device="cuda", dtype=dtype)
        self.counters = {
            "fill_calls": 0,
            "fill_tokens": 0,
        }

    def attach_embed(self, model):
        """Patch the embed_tokens INSTANCE (safe after LLM: it runs eagerly)."""
        emb = emb_name = None
        for name, mod in model.named_modules():
            if name.endswith("embed_tokens") and hasattr(mod, "weight"):
                emb, emb_name = mod, name
                break
        if emb is None:
            raise RuntimeError("no embed_tokens with a weight")
        if emb.weight.shape[1] != self.hidden:
            raise RuntimeError(
                f"hidden mismatch: buffer built for {self.hidden}, "
                f"embed_tokens has {emb.weight.shape[1]}"
            )
        self.emb = emb
        inj, cif = self, self.counters
        orig = emb.forward

        def fill_and_embed(self_emb, ids, *a, **kw):
            n = int(ids.numel())
            cif["fill_calls"] += 1
            cif["fill_tokens"] += n
            r = inj.reader
            if r is not None:
                tok = [int(t) for t in ids.reshape(-1).cpu().tolist()]
                if _MODE["read"] == "op":
                    # Stage 2: only hand the tokens over; the disk read itself
                    # moves into the graph gap via the registered op.
                    _MODE["pending_tokens"] = tok
                else:
                    e = r.fetch(tok)                   # [n, out_dim] float32 cpu
                    delta = (e.to("cuda", dtype=torch.float32) @ inj.proj).to(inj.buf.dtype)
                    inj.buf[:n].copy_(delta)
            else:
                _MODE["pending_tokens"] = None
            return orig(ids, *a, **kw)

        emb.forward = types.MethodType(fill_and_embed, emb)
        self._orig_emb_forward = orig
        print(f"[inject] class-level layer patch armed for layer={self.layer_idx}; "
              f"embed instance patch on {emb_name} "
              f"(hidden={self.hidden} out_dim={self.out_dim} "
              f"buf={tuple(self.buf.shape)} {self.buf.dtype})")

    def set_reader(self, reader):
        self.reader = reader
        if reader is None:
            # makes the no-reader arm a true baseline: identical structure, zero
            # delta.  bf16 +0.0 is exact, so this is bit-identical to no add.
            self.buf.zero_()

    def reset_counters(self):
        for k in self.counters:
            self.counters[k] = 0


def model_hidden_size(model_dir: str) -> int:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return int(cfg.get("text_config", cfg)["hidden_size"])


def make_reader(arm: str, rows_per_token: int):
    if arm == "engram-i":
        return EngramStoreReader(ROWS_DIR, rows_per_token)
    if arm == "mmap":
        return MmapReader(ROWS_DIR, rows_per_token)
    if arm == "shm":
        return ShmReader(ROWS_DIR, rows_per_token, SHM_BUDGET_BYTES)
    raise SystemExit(f"unknown arm {arm!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="none,engram-i,none")
    ap.add_argument("--rows-per-token", type=int, default=16)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--eager", action="store_true",
                    help="enforce_eager=True; control run")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--diag", action="store_true",
                    help="eager-only prints; do NOT combine with graph mode")
    ap.add_argument("--read", choices=("buffer", "op"), default="buffer",
                    help="buffer = stage 1 static buffer (Inductor folds it under "
                         "torch.compile); op = stage 2 registered splitting op")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    _DIAG["on"] = args.diag
    _MODE["read"] = args.read

    shards = sorted(Path(ROWS_DIR).glob("shard_*.bin"))
    if not shards:
        print(f"FATAL: no shards under {ROWS_DIR}", file=sys.stderr)
        return 2
    print(f"[table] {len(shards)} shards in {ROWS_DIR}")

    hidden = model_hidden_size(MODEL)
    out_dim = args.rows_per_token * ROW_WIDTH // 4      # float32 lanes per token
    inj = GraphInjector(args.layer, out_dim, hidden, torch.bfloat16)
    install_layer_patch()
    _STATE["inj"] = inj          # must be set BEFORE LLM(): the profiling
    comp_cfg = None
    if args.read == "op":
        install_read_op()
        comp_cfg = {"splitting_ops": default_splitting_ops()}
        print(f"[inject] registered vllm::{OP_NAME} and added it to splitting_ops")
    print(f"[inject] patch installed before engine construction "
          f"(layer={args.layer} hidden={hidden} out_dim={out_dim} "
          f"read={args.read})")

    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        enforce_eager=args.eager,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        disable_log_stats=True,
        trust_remote_code=True,
        **({"compilation_config": comp_cfg} if comp_cfg else {}),
    )
    print(f"[engine] up in {time.perf_counter() - t0:.1f}s "
          f"enforce_eager={args.eager} "
          f"(~47 tok/s = eager, ~340-410 tok/s = cuda graph on this box)")

    model = find_model(llm)
    if model is None:
        print("FATAL: could not reach the model in-process "
              "(VLLM_ENABLE_V1_MULTIPROCESSING=0 set?)", file=sys.stderr)
        return 3
    inj.attach_embed(model)
    if args.diag:
        found = [
            (n, m) for n, m in model.named_modules()
            if getattr(m, "layer_idx", None) == args.layer
        ]
        for n, m in found[:3]:
            print(f"[diag] module {n}: layer_idx={m.layer_idx} "
                  f"has_engram_buf={hasattr(m, '_engram_buf')} "
                  f"buf_is_shared={getattr(m, '_engram_buf', None) is inj.buf}")
        if not found:
            print(f"[diag] NO module with layer_idx == {args.layer}")

    prompts = [{"prompt_token_ids": [1000 + (7 * i + j) % 200000
                                     for j in range(args.prompt_len)]}
               for i in range(args.batch)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    results = []
    baseline_tokens = None
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        reader = make_reader(arm, args.rows_per_token) if arm != "none" else None
        inj.set_reader(reader)
        inj.reset_counters()
        runs, it_tokens = [], []
        for it in range(args.iterations):
            if args.cold and reader is not None:
                reader.release()
                n = drop_table_cache(shards)
                v = verify_cold(shards)
                print(f"[cold] {arm} iter={it}: fadvise {n}/{len(shards)} "
                      f"ratio={v['ratio']:.1f}x marginal={v['marginal_us_per_read']:.1f}us "
                      f"-> {v['verdict']}")
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            dt = time.perf_counter() - t0
            got = [list(o.outputs[0].token_ids) for o in outs]
            it_tokens.append(got)
            ntok = sum(len(g) for g in got)
            runs.append({"iter": it, "seconds": dt, "tokens": ntok,
                         "tok_per_s": ntok / dt})
            rs = reader.stats() if reader else {}
            print(f"[run] arm={arm} iter={it} tokens={ntok} s={dt:.3f} "
                  f"tok/s={ntok / dt:.1f}"
                  + (f" reader_us/call={rs['reader_us_per_call']:.1f}"
                     if rs.get("reader_us_per_call") else ""))
        if arm == "none" and baseline_tokens is None:
            baseline_tokens = it_tokens[0]
        vals = sorted(r["tok_per_s"] for r in runs)
        det = all(t == it_tokens[0] for t in it_tokens)
        same = (baseline_tokens is not None and it_tokens[0] == baseline_tokens)
        results.append({
            "arm": arm,
            "runs": runs,
            "tok_per_s_median": vals[len(vals) // 2],
            "tok_per_s_best": vals[-1],
            "reader": reader.stats() if reader else {},
            "counters": dict(inj.counters),
            "deterministic_across_iters": det,
            "tokens_identical_to_none": same,
        })
        print(f"  -> counters={inj.counters} det={det} identical_to_none={same}")

    # Proof of consumption cannot come from a counter: no Python runs at replay.
    # It comes from the functional test -- a non-zero delta must move the output.
    reader_reached_model = all(
        (r["arm"] == "none") or (not r["tokens_identical_to_none"]) for r in results
    )
    any_reader = any(r["arm"] != "none" for r in results)
    # eager is ~47 tok/s on this box, cuda graph ~340-410
    graph_mode = max(r["tok_per_s_median"] for r in results) > 150.0
    ok = reader_reached_model and (graph_mode or args.eager) and any_reader
    out = {
        "engine": "vllm",
        "vllm": __import__("vllm").__version__,
        "enforce_eager": args.eager,
        "mode": "eager" if args.eager else "cuda_graph",
        "injection": "eager fill (embed instance) -> static buffer -> "
                     "class-level layer patch consumed inside the graph",
        "read_mode": args.read,
        "stage": ("2 of 2: disk read inside a registered splitting op"
                  if args.read == "op" else
                  "1 of 2: static buffer add (folded away by Inductor under "
                  "torch.compile; valid in eager only)"),
        "layer": args.layer,
        "rows_per_token": args.rows_per_token,
        "payload_bytes_per_token": args.rows_per_token * ROW_WIDTH,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "table_shards": len(shards),
        "graph_mode_detected": graph_mode,
        "reader_arm_reached_model": reader_reached_model,
        "verdict": "valid" if ok else "VOID",
        "results": results,
    }
    print()
    for r in results:
        print(f"  {r['arm']:<10} median={r['tok_per_s_median']:>6.1f} tok/s  "
              f"identical_to_none={r['tokens_identical_to_none']}  "
              f"deterministic={r['deterministic_across_iters']}  "
              f"fill={r['counters']['fill_calls']}")
    print(f"  VERDICT: {out['verdict']}  (graph mode={graph_mode}, "
          f"reader reached model={reader_reached_model})")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0 if out["verdict"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
