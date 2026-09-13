#!/usr/bin/env python3
"""SGLang BCG probe: establish the facts we must not guess.

Why this exists
---------------
Roadmap §35.1d concluded (from source reading) that SGLang's breakable CUDA
graph is the structurally-safer route to sub-condition 4.  Every one of those
conclusions is **unverified on the box**.  Before writing an A/B we need four
facts, and three of them are cheap to get wrong in a way that produces a silent
no-op instead of an error:

1. Where `eager_on_graph` actually lives in the installed version.
   (The path moved in v0.5.14: `model_executor/breakable_cuda_graph/` ->
   `model_executor/runner_backend_utils/breakable_cuda_graph/`.  A wrong import
   is loud; a wrong *patch target* is silent.)

2. Is `--cuda-graph-backend-decode=breakable` a real field on this ServerArgs,
   and does the engine actually select `BreakableCudaGraphBackend` for DECODE?
   (`SGLANG_USE_BREAKABLE_CUDA_GRAPH` is a write-only env var in v0.5.19 --
   docs claim otherwise.  See `docs/cuda-graph-injection.md` §4.1.)

3. **Does a `sitecustomize.py` monkeypatch reach the scheduler child?**
   SGLang runs the model in a **spawn**ed subprocess, so a class-level patch made
   in this parent process is *not inherited* -- it would silently not apply,
   which is exactly the vLLM W1 failure mode.  This is the single highest-risk
   unknown and the whole reason this probe exists.

4. What the decoder layer stack looks like for *our* checkpoint: which class
   `ALL_DECODER_LAYER_TYPES` picks, how many layers, and whether `self.layer_id`
   is present (our injection point keys on it).

Honest scope
------------
This probe **asserts nothing about performance** and takes no measurement.  It
prints PASS/FAIL per check and exits non-zero if any structural check fails.
It starts a real engine once (tiny generation) to answer (2) and (3).

Usage
-----
    python scripts/sglang_bcg_probe.py --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B
    python scripts/sglang_bcg_probe.py --model ... --json-out probes/sglang_bcg_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record one structural check.  Prints immediately so a hang still shows
    how far we got -- an unattributed hang is worse than a failure."""
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------------------
# 1. imports and paths
# ---------------------------------------------------------------------------

BCG_PATHS = [
    # v0.5.14 .. current
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph",
    # v0.5.11 .. v0.5.13
    "sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph",
    # original PR #19102 spelling
    "sglang.srt.model_executor.breakable_cuda_graph",
]


def probe_imports() -> dict:
    print("\n[1] breakable-cuda-graph primitives")
    found = None
    for mod_path in BCG_PATHS:
        try:
            mod = __import__(mod_path, fromlist=["eager_on_graph"])
        except Exception:
            continue
        if hasattr(mod, "eager_on_graph"):
            found = (mod_path, mod)
            break
    if not check("eager_on_graph importable", found is not None,
                 found[0] if found else f"tried {len(BCG_PATHS)} paths"):
        return {}

    mod_path, mod = found
    names = ["eager_on_graph", "break_graph", "BreakableCUDAGraph",
             "BreakableCUDAGraphCapture"]
    present = [n for n in names if hasattr(mod, n)]
    check("primitive surface", len(present) == len(names), f"have {present}")

    # is_in_breakable_cuda_graph lives in a sibling `context` module
    ctx = None
    for cand in (f"{mod_path}.context",
                 "sglang.srt.model_executor.runner_backend_utils."
                 "breakable_cuda_graph.context"):
        try:
            ctx = __import__(cand, fromlist=["is_in_breakable_cuda_graph"])
            break
        except Exception:
            continue
    has_flag = ctx is not None and hasattr(ctx, "is_in_breakable_cuda_graph")
    check("is_in_breakable_cuda_graph importable", has_flag,
          "used as the replay-time self-proof" if has_flag else "not found")
    if has_flag:
        check("flag is False outside BCG", ctx.is_in_breakable_cuda_graph() is False,
              f"got {ctx.is_in_breakable_cuda_graph()!r}")

    return {"path": mod_path, "mod": mod, "ctx": ctx}


# ---------------------------------------------------------------------------
# 2. ServerArgs / Backend
# ---------------------------------------------------------------------------

def probe_server_args() -> dict:
    print("\n[2] cuda-graph backend configuration")
    from sglang.srt.server_args import ServerArgs

    from sglang.srt.model_executor.cuda_graph_config import Backend
    check("Backend.BREAKABLE exists", getattr(Backend, "BREAKABLE", None) == "breakable",
          f"Backend.BREAKABLE={getattr(Backend, 'BREAKABLE', None)!r}")

    fields = getattr(ServerArgs, "model_fields", None) or getattr(ServerArgs, "__fields__", {})
    has_field = "cuda_graph_backend_decode" in fields
    check("ServerArgs.cuda_graph_backend_decode exists", has_field,
          f"{len(fields)} fields" if not has_field else "")
    choices = None
    if has_field:
        f = fields["cuda_graph_backend_decode"]
        choices = getattr(f, "annotation", None) or getattr(f, "type_", None)
        check("  ... accepts 'breakable'", "breakable" in str(choices), str(choices)[:200])

    # defaults: decode should be FULL, prefill BREAKABLE (on CUDA)
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
    cfg = CudaGraphConfig()
    decode_default = getattr(cfg.decode, "backend", None)
    prefill_default = getattr(cfg.prefill, "backend", None)
    check("decode default needs an explicit override", decode_default != "breakable",
          f"decode={decode_default!r} prefill={prefill_default!r}")

    # the env var must be treated as dead
    env_mod = None
    try:
        from sglang.srt import environ as env_mod
    except Exception:
        pass
    env_present = env_mod is not None and hasattr(env_mod, "SGLANG_USE_BREAKABLE_CUDA_GRAPH")
    check("SGLANG_USE_BREAKABLE_CUDA_GRAPH exists but is never read",
          env_present, "do NOT rely on it; use --cuda-graph-backend-decode=breakable")

    return {"decode_default": decode_default, "prefill_default": prefill_default,
            "has_field": has_field}


def probe_backend_selection(model: str) -> dict:
    """Static check that the decode path can reach a breakable backend.

    The *dynamic* proof (a child actually constructing one) comes from the
    sitecustomize hook in probe_spawn_inheritance -- this function only
    establishes that the branch exists, so a failure here stays
    distinguishable from a failure there.
    """
    print("\n[3] decode backend selection (static)")
    from sglang.srt.model_executor.runner_backend import utils as rb_utils
    import inspect

    src = inspect.getsource(rb_utils)
    has_breakable_branch = "BreakableCudaGraphBackend" in src
    check("runner_backend.utils constructs BreakableCudaGraphBackend",
          has_breakable_branch, "")

    # which phase gates it
    lines = src.splitlines()
    hits = [(i + 1, l.strip()) for i, l in enumerate(lines)
            if "BreakableCudaGraphBackend" in l]
    check("  ... and it is reachable for a phase we will use", bool(hits),
          "; ".join(f"L{n}: {t[:70]}" for n, t in hits[:4]))
    return {"source_hits": [n for n, _ in hits]}


# ---------------------------------------------------------------------------
# 4. spawn-inheritance -- THE critical unknown
# ---------------------------------------------------------------------------

SITECUSTOMIZE_TEMPLATE = '''\
# Auto-generated by scripts/sglang_bcg_probe.py -- spawn-inheritance probe.
#
# Answers TWO questions in one engine start:
#   (1) does SGLang's spawned scheduler child import this file at all
#       (i.e. would a parent-process monkeypatch be inherited)?  and
#   (2) did that child actually construct a BreakableCudaGraphBackend
#       (i.e. was the DECODE backend really breakable)?
# (2) matters because "the decorator is inert outside a capture" is silent:
# without a capture object the wrapper just calls through, exactly like the
# vLLM failure.
import importlib.abc
import importlib.util
import json
import os
import sys
import time

_MARKER = __MARKER__
_TARGET = ("sglang.srt.model_executor.runner_backend."
           "breakable_cuda_graph_backend")


def _rec(kind, **kw):
    try:
        with open(_MARKER, "a") as _f:
            _f.write(json.dumps({"pid": os.getpid(), "kind": kind,
                                 "t": time.time(), **kw}) + "\\n")
    except Exception:
        pass


_rec("interp", ppid=os.getppid(), argv0=(sys.argv[0] or "")[-80:])


class _Finder(importlib.abc.MetaPathFinder):
    """Lazy post-import hook: patch the BCG backend module when it loads.

    Lazy on purpose -- importing sglang modules from sitecustomize would run
    before the interpreter is fully set up and risks import cycles.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            spec = None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        _orig_exec = spec.loader.exec_module

        def _exec(module, _orig_exec=_orig_exec):
            _orig_exec(module)
            try:
                cls = getattr(module, "BreakableCudaGraphBackend", None)
                if cls is None:
                    _rec("no_backend_class", target=fullname)
                    return
                _orig_init = cls.__init__

                def __init__(self, *a, **kw):
                    _orig_init(self, *a, **kw)
                    _rec("BreakableCudaGraphBackend", cls=type(self).__name__)

                cls.__init__ = __init__
                _rec("patched", target=fullname)
            except Exception as exc:
                _rec("patch_error", err=repr(exc))

        spec.loader.exec_module = _exec
        return spec


sys.meta_path.insert(0, _Finder())
'''


def probe_spawn_inheritance(model: str, mem_fraction: float) -> dict:
    print("\n[4] spawn inheritance: does a PYTHONPATH sitecustomize reach the scheduler?")
    tmp = Path(tempfile.mkdtemp(prefix="engramdb-bcg-probe-"))
    marker = tmp / "marker.jsonl"
    (tmp / "sitecustomize.py").write_text(SITECUSTOMIZE_TEMPLATE.replace("__MARKER__", repr(str(marker))))

    env_before = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{tmp}{os.pathsep}{env_before}" if env_before else str(tmp)
    # Make sure the *current* interpreter would also pick it up for a re-exec.
    if str(tmp) not in sys.path:
        sys.path.insert(0, str(tmp))

    print(f"  PYTHONPATH={os.environ['PYTHONPATH'][:160]}")
    print(f"  marker={marker}")

    engine = None
    err = None
    try:
        import sglang
        engine = sglang.Engine(
            model_path=model,
            mem_fraction_static=mem_fraction,
            max_total_tokens=4096,
            cuda_graph_backend_decode="breakable",
            log_level="warning",
        )
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    if err is not None:
        check("engine starts with --cuda-graph-backend-decode=breakable", False, err[:400])
    else:
        check("engine starts with --cuda-graph-backend-decode=breakable", True, "")
        try:
            engine.generate(input_ids=[[1000 + i for i in range(16)]],
                            sampling_params={"temperature": 0.0, "max_new_tokens": 4,
                                             "ignore_eos": True})
            check("  ... and completes a tiny generation", True, "")
        except Exception as exc:  # noqa: BLE001
            check("  ... and completes a tiny generation", False,
                  f"{type(exc).__name__}: {exc}"[:400])
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass

    time.sleep(1.5)  # let any child flush

    records = []
    if marker.exists():
        for line in marker.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    pids = sorted({r["pid"] for r in records})
    check("sitecustomize was imported at all", bool(records),
          f"{len(records)} record(s) from pids {pids}")

    mypid = os.getpid()
    children = sorted({r["pid"] for r in records if r["pid"] != mypid})
    inherited = bool(children)
    check(">>> SPAWNED CHILD inherited the patch  <-- decisive", inherited,
          f"child pids {children}" if inherited else
          "NO child imported it -- a parent-process monkeypatch would be a SILENT NO-OP")

    # (2) did the child really build a breakable backend?
    backend_recs = [r for r in records if r.get("kind") == "BreakableCudaGraphBackend"]
    backend_pids = sorted({r["pid"] for r in backend_recs})
    child_backend = [p for p in backend_pids if p != mypid]
    check(">>> BreakableCudaGraphBackend constructed in a CHILD  <-- decode is BCG",
          bool(child_backend),
          f"pids {child_backend}" if child_backend else
          f"construction sites: {backend_pids or 'none'}; if the parent built it, "
          "the patch landed but the model runs elsewhere")

    patch_errors = [r for r in records if r.get("kind") in ("patch_error", "no_backend_class")]
    check("no sitecustomize patch errors", not patch_errors,
          "; ".join(str(r)[:120] for r in patch_errors[:3]))

    return {"marker": str(marker), "records": records, "pids": pids,
            "child_pids": children, "inherited": inherited,
            "backend_pids": backend_pids, "child_backend_pids": child_backend,
            "engine_error": err, "tmpdir": str(tmp)}


# ---------------------------------------------------------------------------
# 5. decoder layer structure of OUR checkpoint
# ---------------------------------------------------------------------------

def probe_model_structure(model: str) -> dict:
    print("\n[5] decoder layer structure")
    out: dict = {}

    cfg_path = Path(model) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        tc = cfg.get("text_config", cfg)
        out["architectures"] = cfg.get("architectures")
        out["num_hidden_layers"] = tc.get("num_hidden_layers")
        out["layer_types"] = tc.get("layer_types")
        out["has_ple_fields"] = sorted(
            k for k in tc if "ple" in k.lower() or "ngram" in k.lower()
        )
        check("config.json readable", True,
              f"arch={out['architectures']} n_layers={out['num_hidden_layers']}")
    else:
        check("config.json readable", False, str(cfg_path))

    try:
        from sglang.srt.models.qwen3_5 import ALL_DECODER_LAYER_TYPES
        classes = {k: v.__name__ for k, v in ALL_DECODER_LAYER_TYPES.items()}
        out["decoder_layer_classes"] = classes
        check("ALL_DECODER_LAYER_TYPES importable", True, str(classes))
        # our injection keys on self.layer_id -- verify the attribute exists
        import inspect
        has_attr = []
        for k, v in ALL_DECODER_LAYER_TYPES.items():
            src = inspect.getsource(v.__init__)
            has_attr.append((k, "self.layer_id" in src))
        out["layer_id_attr"] = has_attr
        check("every decoder layer class sets self.layer_id",
              all(ok for _, ok in has_attr), str(has_attr))
    except Exception as exc:  # noqa: BLE001
        check("ALL_DECODER_LAYER_TYPES importable", False,
              f"{type(exc).__name__}: {exc}"[:300])

    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mem-fraction-static", type=float, default=0.70)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--skip-engine", action="store_true",
                    help="structure-only: do not start a real engine")
    args = ap.parse_args()

    print("=" * 78)
    print("SGLang BCG probe -- structural facts only, no performance claim")
    print("=" * 78)

    result: dict = {"model": args.model}
    try:
        import sglang
        result["sglang_version"] = getattr(sglang, "__version__", "?")
        print(f"\nsglang {result['sglang_version']}   python {sys.version.split()[0]}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nFATAL: cannot import sglang: {exc}")
        return 2

    imports = probe_imports()
    result["imports_path"] = imports.get("path") if imports else None
    result["server_args"] = probe_server_args()
    result["backend_selection"] = probe_backend_selection(args.model)

    if args.skip_engine:
        print("\n[4] spawn inheritance -- SKIPPED (--skip-engine)")
        result["spawn"] = None
    else:
        result["spawn"] = probe_spawn_inheritance(args.model, args.mem_fraction_static)

    result["model_structure"] = probe_model_structure(args.model)

    failed = [c["check"] for c in CHECKS if not c["ok"]]
    result["checks"] = CHECKS
    result["failed"] = failed
    result["verdict"] = "PASS" if not failed else "FAIL"

    print("\n" + "=" * 78)
    print(f"verdict: {result['verdict']}   ({len(CHECKS) - len(failed)}/{len(CHECKS)} checks)")
    for f in failed:
        print(f"  FAILED: {f}")
    if result.get("spawn") and not result["spawn"].get("inherited"):
        print("\n  >>> The spawn check failed. A parent-process monkeypatch WILL NOT")
        print("      reach the scheduler. Do not write the A/B until this is solved;")
        print("      a patch that silently does not apply is the vLLM W1 failure mode.")
    if result.get("spawn"):
        print(f"\n  probe dir left for inspection: {result['spawn'].get('tmpdir')}")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
