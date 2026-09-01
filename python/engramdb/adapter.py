"""Generic engine-side serving adapters.

This module is the thin, engine-agnostic bridge between EngramDB's optional
serving layer and a PyTorch model or inference engine.

It intentionally does not import vLLM, SGLang, or any concrete Qwen reader.
The only heavyweight optional dependency is PyTorch, used by
``PleMemoryAdapter``; the target-reader hook itself works with any PyTorch
module (or compatible object exposing register hooks).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from .ple_memory import PleMemory, PleSequence, PleSequenceStore

__all__ = [
    "PleMemoryAdapter",
    "TargetReaderHook",
    "install_target_reader_hook",
    "install_bundle_adapter",
    "install_vllm_target_reader",
    "install_sglang_target_reader",
]


class PleMemoryAdapter(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Stateful PyTorch adapter backed by :class:`PleMemory`.

    It keeps one :class:`PleSequence` per request id and returns the PLE e_t
    tensor for each fed token batch.

    Example::

        adapter = PleMemoryAdapter(memory)
        e_t = adapter(input_ids, seq_ids=[0, 1])   # [B, T, heads, head_dim]
    """

    def __init__(
        self,
        memory: PleMemory,
        *,
        sequence_store: PleSequenceStore | None = None,
        keep_steps: int = 32,
    ) -> None:
        if nn is None:
            raise ImportError("PleMemoryAdapter requires PyTorch")
        super().__init__()
        self.memory = memory
        self.sequences = sequence_store or PleSequenceStore(
            memory,
            keep_steps=keep_steps,
        )

    def forward(
        self,
        input_ids: Any,
        seq_ids: Sequence[int] | None = None,
        *,
        as_tensor: bool = True,
    ) -> Any:
        if torch is None:
            raise ImportError("PleMemoryAdapter.forward requires PyTorch")
        was_1d = input_ids.dim() == 1
        if was_1d:
            input_ids = input_ids.unsqueeze(0)
        batch_size = int(input_ids.shape[0])
        if seq_ids is None:
            seq_ids = list(range(batch_size))
        else:
            seq_ids = [int(x) for x in seq_ids]
            if len(seq_ids) != batch_size:
                raise ValueError(
                    f"seq_ids length {len(seq_ids)} != batch size {batch_size}"
                )

        outputs: list[Any] = []
        for b, seq_id in enumerate(seq_ids):
            ids = input_ids[b].tolist()
            step = self.sequences.feed(seq_id, ids, as_tensor=as_tensor)
            if not as_tensor:
                # Raw mode is useful for non-torch consumers; return bytes.
                outputs.append(step.raw)
            else:
                if step.e_t is None:
                    raise RuntimeError("PleMemoryAdapter failed to produce e_t")
                outputs.append(step.e_t)

        if not outputs:
            return []
        if as_tensor and len(outputs) > 1:
            out = torch.stack(outputs, dim=0)
        else:
            out = outputs[0]
        if was_1d and as_tensor:
            out = out.squeeze(0)
        return out

    def reset(self, seq_id: Any | None = None) -> None:
        """Reset one sequence or all sequences."""
        if seq_id is None:
            self.sequences.clear()
        else:
            self.sequences.remove(seq_id)

    def close(self) -> None:
        self.sequences.clear()


class TargetReaderHook:
    """A small managed forward hook that invokes a target reader.

    The reader can be any object with ``on_forward(module, args, output_or_kwargs)``
    or simply a callable.
    """

    def __init__(
        self,
        reader: Any,
        *,
        target_path: str | None = None,
        mode: str = "post",
        name: str = "engramdb-target-reader",
    ) -> None:
        self.reader = reader
        self.target_path = target_path
        self.mode = mode
        self.name = name
        self.handle: Any = None
        self._target: Any = None

    def _callback(self) -> Callable[..., Any]:
        callback = getattr(self.reader, "on_forward", None)
        if callback is None:
            callback = self.reader
        if self.mode == "pre":
            def pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> Any:
                return callback(module, args, kwargs)
            return pre_hook

        def post_hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            return callback(module, args, output)
        return post_hook

    def attach(self, model: Any) -> "TargetReaderHook":
        target = model
        if self.target_path:
            target = model.get_submodule(self.target_path)
        if not hasattr(target, "register_forward_pre_hook") and not hasattr(
            target, "register_forward_hook"
        ):
            raise TypeError(
                f"{type(target).__name__} does not expose PyTorch forward hooks"
            )
        if self.mode == "pre":
            self.handle = target.register_forward_pre_hook(self._callback())
        elif self.mode == "post":
            self.handle = target.register_forward_hook(self._callback())
        else:
            raise ValueError("mode must be 'pre' or 'post'")
        self._target = target
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self._target = None

    def __enter__(self) -> "TargetReaderHook":
        if self._target is None and self.handle is None:
            raise RuntimeError("attach the hook before using it as a context manager")
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()


def install_target_reader_hook(
    model: Any,
    reader: Any,
    *,
    target_path: str | None = None,
    mode: str = "post",
    name: str = "engramdb-target-reader",
) -> TargetReaderHook:
    """Install a target-reader forward hook and return the handle."""
    return TargetReaderHook(
        reader,
        target_path=target_path,
        mode=mode,
        name=name,
    ).attach(model)


def install_vllm_target_reader(
    model: Any,
    reader: Any,
    **kwargs: Any,
) -> TargetReaderHook:
    """Convenience alias for installing a target reader in vLLM-style models."""
    return install_target_reader_hook(model, reader, **kwargs)


def install_sglang_target_reader(
    model: Any,
    reader: Any,
    **kwargs: Any,
) -> TargetReaderHook:
    """Convenience alias for installing a target reader in SGLang-style models."""
    return install_target_reader_hook(model, reader, **kwargs)


def install_bundle_adapter(
    model: Any,
    registry: Any,
    bundle: Any,
    *,
    memory: PleMemory | None = None,
    reader_index: int = 0,
    target_path: str | None = None,
    mode: str = "post",
) -> dict[str, Any]:
    """Create a memory + target reader from a bundle and attach it to a model.

    Returns a dict with ``memory``, ``reader``, and ``hook`` so callers can
    clean up afterward.
    """
    if memory is None:
        memory = bundle.open_memory()
    reader = registry.create_from_manifest(bundle, index=reader_index)
    hook = TargetReaderHook(
        reader,
        target_path=target_path,
        mode=mode,
    ).attach(model)
    return {"memory": memory, "reader": reader, "hook": hook}
