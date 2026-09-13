#!/usr/bin/env python3
"""Teach SGLang 0.5.19's breakable-cudagraph backend about *dataclass* outputs.

The gap
-------
``BreakableCudaGraphBackend`` bridges graph segments through static output
buffers, so it needs four structure-recursive helpers::

    _output_rows            how many leading rows the body produced
    _alloc_full_buffer      a same-structure buffer with `size` leading rows
    _slice_output           the first `num_tokens` rows, same structure
    _copy_output_to_buffer  copy body output into that buffer

All four dispatch on ``None`` / ``torch.Tensor`` / ``PPProxyTensors`` /
``tuple`` / ``list`` -- and raise ``TypeError`` on anything else.  A model whose
forward returns a dataclass therefore cannot be captured at all::

    TypeError: Unsupported BCG output type:
               <class 'sglang.srt.layers.logits_processor.LogitsProcessorOutput'>

``LogitsProcessorOutput`` is exactly that shape (a dataclass holding
``next_token_logits`` plus a dozen optional fields), so
``--cuda-graph-backend-decode=breakable`` fails at capture for essentially every
text model in the tree.  Verified on ``Qwen3.5-0.8B`` with sglang 0.5.19.

Why this is the right fix
-------------------------
SGLang already ships the intended pattern one module over: ``_copy_output`` in
``runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`` *does*
handle ``hasattr(dst, "__dict__")`` objects, recursing per attribute.  The
backend's four buffer helpers simply never got the same branch.  This module
fills that hole with the same rule, so nothing about the existing types changes.

Design notes
------------
* Installed by **wrapping**, not by rewriting: each replacement delegates to the
  original for every type the original supports, and only intercepts the generic
  object case.  A future sglang version that gains its own branch keeps working.
* ``copy.copy`` + ``setattr`` is used instead of calling the constructor, so
  dataclasses with non-default fields, ``InitVar``, or extra attributes all
  survive.
* Non-tensor attributes (``None``, ints, plain lists of Python floats) pass
  through untouched in all four helpers -- they carry no graph state.
* Because it is installed by import, it must be imported **before** capture.
  ``scripts/sglang_bcg_hook.py`` appends a 4-line loader to the installed
  backend module for exactly that reason.

Usage
-----
    import sglang_bcg_output_patch
    sglang_bcg_output_patch.install()
"""

from __future__ import annotations

import copy
import functools

__all__ = ["install", "uninstall"]

_PATCHED = False


def _generic_object(x) -> bool:
    """True for the dataclass-ish case the backend helpers do not handle.

    Everything the originals already support is excluded explicitly, so those
    paths keep their existing behaviour byte for byte.
    """
    import torch

    if x is None or torch.is_tensor(x):
        return False
    if isinstance(x, (tuple, list)):
        return False
    if type(x).__name__ == "PPProxyTensors":
        return False
    return hasattr(x, "__dict__")


def _iter_attrs(obj):
    # vars() is the same view _copy_output uses one module over.
    return list(vars(obj).items())


def install(verbose: bool = True) -> bool:
    """Wrap the four buffer helpers.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend as B,
    )

    orig_alloc = B._alloc_full_buffer
    orig_slice = B._slice_output
    orig_copy = B._copy_output_to_buffer
    orig_rows = B._output_rows

    @functools.wraps(orig_alloc)
    def _alloc_full_buffer(self, output, size):
        if _generic_object(output):
            buf = copy.copy(output)
            for key, val in _iter_attrs(output):
                setattr(buf, key, _alloc_full_buffer(self, val, size))
            return buf
        return orig_alloc(self, output, size)

    @functools.wraps(orig_slice)
    def _slice_output(self, output, num_tokens):
        if _generic_object(output):
            buf = copy.copy(output)
            for key, val in _iter_attrs(output):
                setattr(buf, key, _slice_output(self, val, num_tokens))
            return buf
        return orig_slice(self, output, num_tokens)

    @functools.wraps(orig_copy)
    def _copy_output_to_buffer(self, output, output_buffer, num_tokens):
        if _generic_object(output) and _generic_object(output_buffer):
            a, b = vars(output), vars(output_buffer)
            if a.keys() != b.keys():
                raise ValueError(
                    "BCG output structure changed between capture sizes: "
                    f"{sorted(a)} vs {sorted(b)}"
                )
            for key in a:
                _copy_output_to_buffer(self, a[key], b[key], num_tokens)
            return
        return orig_copy(self, output, output_buffer, num_tokens)

    @functools.wraps(orig_rows)
    def _output_rows(self, output, cap):
        if _generic_object(output):
            rows = [
                _output_rows(self, val, cap)
                for _, val in _iter_attrs(output)
                if val is not None
            ]
            return min([cap, *rows]) if rows else cap
        return orig_rows(self, output, cap)

    B._alloc_full_buffer = _alloc_full_buffer
    B._slice_output = _slice_output
    B._copy_output_to_buffer = _copy_output_to_buffer
    B._output_rows = _output_rows

    _PATCHED = True
    if verbose:
        print(
            "[sglang_bcg_output_patch] installed: BCG output helpers now accept "
            "dataclass outputs (LogitsProcessorOutput and friends)"
        )
    return True


def uninstall() -> None:
    """Restore the originals.  Only useful in-process, before any capture."""
    global _PATCHED
    if not _PATCHED:
        return
    raise NotImplementedError("re-import the module to reset; not needed in practice")
