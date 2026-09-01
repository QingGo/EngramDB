"""Per-sequence PLE memory abstraction.

This module is the first piece of EngramDB's optional serving layer.

It separates the disk storage ("which rows/slots do I read?") from the
per-request state ("what is the current n-gram history for this sequence?").
It intentionally has no dependency on vLLM, SGLang, or a concrete reader
checkpoint; the only optional heavyweight dependency is PyTorch for tensor
conversion.

Public API
----------

``PleMemory``
    Owns one Store-I (raw row file) or Store-P view (+ slot index) and
    exposes deterministic rowid generation plus raw/tensor fetch.

``PleSequence``
    Mutable per-request state.  It keeps the last ``ngram_size - 1`` tokens,
    computes the rowid tuple for each new token batch, fetches the PLE record,
    and exposes ``current_e_t()``.

``PleStep``
    Lightweight result returned by ``PleSequence.feed()``.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "PleMemory",
    "PleSequence",
    "PleSequenceStore",
    "PleStep",
    "ple_memory_from_discovery",
]

# Qwen PLE defaults are duplicated here so importing this optional serving
# module does not import PyTorch via ple_adapter.  The reference rowid
# implementation is still loaded lazily on first use.
_PLE_BASE = 20_000_000
_PLE_DIVISOR = 128
_PLE_EOS = 248044
_PLE_HEADS_PER_NGRAM = 8
_PLE_HEADS = 16
_PLE_NGRAM_SIZE = 3
_DEFAULT_MULTIPLIERS = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]


def _ple_reference():
    """Import the pure-Python PLE reference on demand."""
    from .ple_math import (
        head_offsets,
        head_vocab_sizes,
        ple_rowids,
    )

    return head_offsets, head_vocab_sizes, ple_rowids


def _head_vocab_sizes(heads: int, base: int = _PLE_BASE) -> list[int]:
    _, head_vocab_sizes, _ = _ple_reference()
    return head_vocab_sizes(base=base, heads=heads)


def _head_offsets(sizes: list[int]) -> list[int]:
    head_offsets, _, _ = _ple_reference()
    return head_offsets(sizes)


def _ple_rowids(*args: Any, **kwargs: Any) -> list[list[int]]:
    _, _, ple_rowids = _ple_reference()
    return ple_rowids(*args, **kwargs)


def _as_int_list(values: Iterable[Any] | None) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


@dataclass
class PleStep:
    """One fed token batch and its PLE disk result."""

    tokens: list[int]
    rowids: list[list[int]]
    raw: bytes
    e_t: Any | None = None


class PleMemory:
    """Unified Store-I / Store-P PLE memory.

    Exactly one of ``store`` or ``view`` must be supplied.

    Store-I mode::

        mem = PleMemory(
            store=store,
            head_dim=160,       # bytes per individual head row
            num_heads=16,
        )

    Store-P mode::

        mem = PleMemory(
            view=view,
            slot_index=disk_index,
            num_heads=16,
        )

    ``slot_index`` may be any object with ``lookup(row) -> int``; both
    ``SlotIndex`` and ``DiskSlotIndex`` qualify.
    """

    def __init__(
        self,
        *,
        store: Any | None = None,
        view: Any | None = None,
        slot_index: Any | None = None,
        head_dim: int | None = None,
        num_heads: int | None = None,
        ngram_size: int | None = None,
        heads_per_ngram: int | None = None,
        multipliers: Sequence[int] | None = None,
        scale: float = 1.0,
        dtype: Any | None = None,
        out_dtype: Any | None = None,
        eos: int | None = None,
        prime_sizes: Sequence[int] | None = None,
        offsets: Sequence[int] | None = None,
        divisor: int = _PLE_DIVISOR,
        sequential_view: bool = False,
        start_slot: int = 0,
    ) -> None:
        if (store is None) == (view is None):
            raise ValueError("provide exactly one of store= or view=")

        self.store = store
        self.view = view
        self.slot_index = slot_index

        # Derive/validate shape.
        self.num_heads = int(num_heads or _PLE_HEADS)
        self.head_dim = int(head_dim if head_dim is not None else self._default_head_dim())
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.ngram_size = int(ngram_size or _PLE_NGRAM_SIZE)
        self.heads_per_ngram = int(heads_per_ngram or _PLE_HEADS_PER_NGRAM)
        if self.ngram_size < 2:
            raise ValueError("ngram_size must be >= 2")
        expected_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.num_heads != expected_heads:
            raise ValueError(
                f"num_heads={self.num_heads} does not match ngram_size={self.ngram_size} "
                f"* heads_per_ngram={self.heads_per_ngram} => {expected_heads}"
            )

        self.eos = int(eos if eos is not None else _PLE_EOS)
        self.multipliers = (
            [int(x) for x in multipliers]
            if multipliers is not None
            else list(_DEFAULT_MULTIPLIERS)
        )
        self.scale = float(scale)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.divisor = int(divisor)
        self.sequential_view = bool(sequential_view)
        self.start_slot = int(start_slot)

        self.prime_sizes = (
            [int(x) for x in prime_sizes]
            if prime_sizes is not None
            else _head_vocab_sizes(self.num_heads)
        )
        self.offsets = (
            [int(x) for x in offsets]
            if offsets is not None
            else _head_offsets(self.prime_sizes)
        )
        if len(self.prime_sizes) != self.num_heads:
            raise ValueError(
                f"prime_sizes length {len(self.prime_sizes)} != num_heads {self.num_heads}"
            )
        if len(self.offsets) != self.num_heads:
            raise ValueError(
                f"offsets length {len(self.offsets)} != num_heads {self.num_heads}"
            )

        # Source validation.
        if self.view is not None:
            if self.slot_index is None and not self.sequential_view:
                raise ValueError(
                    "Store-P mode requires slot_index= (or sequential_view=True)"
                )
        else:
            if self.slot_index is not None:
                # It is harmless to carry an index in Store-I mode, but it is
                # almost always a user error.  Do not silently use it.
                pass

    def _default_head_dim(self) -> int:
        if self.store is not None:
            width = getattr(self.store, "width", None)
            if width is None:
                # ctypes/PyO3 both expose width as an int attribute/property.
                width = int(getattr(self.store, "width", 0))
            return int(width)
        if self.view is not None:
            slot_bytes = int(self.view.slot_bytes)
            heads = int(self.num_heads or _PLE_HEADS)
            if slot_bytes % heads != 0:
                raise ValueError(
                    f"view slot_bytes {slot_bytes} is not divisible by num_heads {heads}"
                )
            return slot_bytes // heads
        return 160

    @property
    def source(self) -> str:
        return "view" if self.view is not None else "store"

    @property
    def record_bytes(self) -> int:
        """Bytes for one complete PLE token record."""
        return self.num_heads * self.head_dim

    def rowids_for_tokens(
        self,
        tokens: Iterable[Any],
        history: Iterable[Any] | None = None,
    ) -> list[list[int]]:
        """Return ``[len(tokens), num_heads]`` rowids for a token batch.

        ``history`` is the already-known previous context (normally the last
        ``ngram_size - 1`` tokens).  If omitted, a fresh EOS-padded context is
        used.
        """
        tok = _as_int_list(tokens)
        hist = _as_int_list(history)
        if hist:
            return _ple_rowids(
                tok,
                self.multipliers,
                self.eos,
                sizes=self.prime_sizes,
                offsets=self.offsets,
                ngram_size=self.ngram_size,
                heads_per_ngram=self.heads_per_ngram,
                history=hist,
            )
        return _ple_rowids(
            tok,
            self.multipliers,
            self.eos,
            sizes=self.prime_sizes,
            offsets=self.offsets,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
        )

    def _coerce_rows(
        self,
        rowid_tuples: Iterable[Sequence[int] | Any],
    ) -> list[tuple[int, ...]]:
        rows: list[tuple[int, ...]] = []
        for row in rowid_tuples:
            if hasattr(row, "tolist"):
                row = row.tolist()
            r = tuple(int(x) for x in row)
            if len(r) != self.num_heads:
                raise ValueError(
                    f"rowid tuple has {len(r)} heads, expected {self.num_heads}"
                )
            rows.append(r)
        return rows

    def fetch_raw(self, rowid_tuples: Iterable[Sequence[int] | Any]) -> bytes:
        """Return raw PLE bytes for rowid tuples.

        In Store-I mode this is the concatenation of individual head rows.  In
        Store-P mode this is the concatenation of full slot records.
        """
        rows = self._coerce_rows(rowid_tuples)
        if not rows:
            return b""
        if self.view is not None:
            slots: list[int]
            if self.slot_index is not None:
                slots = [self.slot_index.lookup(row) for row in rows]
            else:
                slots = [self.start_slot + i for i in range(len(rows))]
            read_records = getattr(self.view, "read_records", None)
            if callable(read_records):
                raw = read_records(slots)
            else:
                raw = b"".join(self.view.read_record(i) for i in slots)
            expected = len(rows) * self.record_bytes
            if len(raw) != expected:
                raise RuntimeError(
                    f"view returned {len(raw)} bytes for {len(rows)} records, "
                    f"expected {expected}"
                )
            return raw

        # Store-I path: flatten one head row per rowid.
        flat: list[int] = []
        for row in rows:
            flat.extend(row)
        raw = self.store.fetch(flat)
        expected = len(rows) * self.record_bytes
        if len(raw) != expected:
            raise RuntimeError(
                f"store returned {len(raw)} bytes for {len(rows)} tokens, "
                f"expected {expected}"
            )
        return raw

    def fetch_tensor(self, rowid_tuples: Iterable[Sequence[int] | Any]) -> Any:
        """Return a ``[N, num_heads, head_dim]`` torch tensor.

        Requires PyTorch.  The output dtype defaults to ``torch.float32`` and
        the raw dtype defaults to ``torch.float8_e4m3fn`` (both overridable at
        construction).
        """
        import torch

        rows = self._coerce_rows(rowid_tuples)
        if not rows:
            return torch.empty(
                (0, self.num_heads, self.head_dim),
                dtype=self.out_dtype or torch.float32,
            )
        raw = self.fetch_raw(rows)
        dtype = self.dtype or torch.float8_e4m3fn
        out_dtype = self.out_dtype or torch.float32
        arr = torch.frombuffer(bytearray(raw), dtype=dtype)
        if arr.dtype != out_dtype:
            arr = arr.to(out_dtype)
        arr = arr.reshape(len(rows), self.num_heads, self.head_dim)
        if self.scale != 1.0:
            arr = arr * self.scale
        return arr

    def fetch(
        self,
        rowid_tuples: Iterable[Sequence[int] | Any],
        *,
        as_tensor: bool = False,
    ) -> Any:
        """Fetch raw bytes by default, or a torch tensor when asked."""
        if as_tensor:
            return self.fetch_tensor(rowid_tuples)
        return self.fetch_raw(rowid_tuples)

    def new_sequence(
        self,
        initial_tokens: Iterable[Any] | None = None,
        *,
        history: Iterable[Any] | None = None,
    ) -> "PleSequence":
        """Create a per-request sequence backed by this memory."""
        seq = PleSequence(self, history=history)
        if initial_tokens is not None:
            seq.feed(initial_tokens)
        return seq

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "record_bytes": self.record_bytes,
            "ngram_size": self.ngram_size,
            "heads_per_ngram": self.heads_per_ngram,
            "scale": self.scale,
        }
        if hasattr(self.slot_index, "stats") and callable(self.slot_index.stats):
            out["slot_index"] = self.slot_index.stats()
        return out

    def close(self) -> None:
        """Close owned optional resources.

        The caller remains responsible for closing ``store`` and ``view``; this
        method only serves as a symmetric lifecycle hook for future ownership
        tracking.
        """


class PleSequence:
    """Per-request PLE history and current e_t.

    The sequence owns only Python-level state (last ``ngram_size - 1`` tokens).
    It does not own the underlying ``PleMemory``/Store/View.
    """

    def __init__(
        self,
        memory: PleMemory,
        history: Iterable[Any] | None = None,
        *,
        keep_steps: int = 32,
    ) -> None:
        self.memory = memory
        self.keep_steps = max(0, int(keep_steps))
        self._history: list[int] = (
            _as_int_list(history)
            if history is not None
            else [memory.eos] * (memory.ngram_size - 1)
        )
        self._tokens: list[int] = []
        self._steps: deque[PleStep] = deque(maxlen=self.keep_steps if self.keep_steps > 0 else 0)
        self._last: PleStep | None = None

    @property
    def history(self) -> list[int]:
        """Current n-gram history (usually last ``ngram_size - 1`` tokens)."""
        return list(self._history)

    @property
    def tokens(self) -> list[int]:
        """All tokens fed to this sequence so far."""
        return list(self._tokens)

    @property
    def length(self) -> int:
        return len(self._tokens)

    def feed(
        self,
        tokens: Iterable[Any],
        *,
        as_tensor: bool = False,
    ) -> PleStep:
        """Feed a token batch, update history, and return its PLE result.

        If ``as_tensor`` is true, ``step.e_t`` is populated with a torch tensor
        of shape ``[len(tokens), num_heads, head_dim]``.
        """
        tok = _as_int_list(tokens)
        if not tok:
            step = PleStep([], [], b"", None)
            self._steps.append(step)
            self._last = step
            return step

        rows = self.memory.rowids_for_tokens(tok, self._history)
        raw = self.memory.fetch_raw(rows)
        e_t = self.memory.fetch_tensor(rows) if as_tensor else None
        step = PleStep(tok, rows, raw, e_t)

        self._tokens.extend(tok)
        keep = max(1, self.memory.ngram_size - 1)
        self._history = (self._history + tok)[-keep:]
        self._steps.append(step)
        self._last = step
        return step

    # Serving-style alias.
    step = feed

    def current_rowids(self) -> list[list[int]]:
        """Rowids for the most recently fed token batch."""
        if self._last is None:
            raise RuntimeError("no tokens have been fed to this sequence")
        return [list(r) for r in self._last.rowids]

    def current_raw(self) -> bytes:
        """Raw PLE bytes for the most recently fed token batch."""
        if self._last is None:
            raise RuntimeError("no tokens have been fed to this sequence")
        return self._last.raw

    def current_e_t(self) -> Any:
        """Torch e_t tensor for the most recently fed token batch.

        If the step was originally fetched as raw bytes, this computes and
        caches the tensor on demand.
        """
        if self._last is None:
            raise RuntimeError("no tokens have been fed to this sequence")
        if self._last.e_t is None:
            self._last.e_t = self.memory.fetch_tensor(self._last.rowids)
        return self._last.e_t

    def reset(self) -> None:
        """Clear request-local state without closing the memory."""
        self._history = [self.memory.eos] * (self.memory.ngram_size - 1)
        self._tokens = []
        self._steps = deque(
            maxlen=self.keep_steps if self.keep_steps > 0 else 0
        )
        self._last = None

    def close(self) -> None:
        """Compatibility lifecycle hook; a sequence owns no disk resources."""
        self.reset()

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self):
        return iter(self._steps)


class PleSequenceStore:
    """Map sequence ids to :class:`PleSequence` instances.

    This is the continuous-batching state container.  It owns only the
    per-request histories; disk resources remain in the shared ``PleMemory``.
    """

    def __init__(
        self,
        memory: PleMemory,
        *,
        max_sequences: int | None = None,
        keep_steps: int = 32,
    ) -> None:
        self.memory = memory
        self.max_sequences = max_sequences
        self.keep_steps = max(0, int(keep_steps))
        self._seqs: OrderedDict[Any, PleSequence] = OrderedDict()

    def get_or_create(
        self,
        seq_id: Any,
        *,
        history: Iterable[Any] | None = None,
        keep_steps: int | None = None,
    ) -> PleSequence:
        if seq_id in self._seqs:
            self._seqs.move_to_end(seq_id)
            return self._seqs[seq_id]
        if self.max_sequences is not None and len(self._seqs) >= self.max_sequences:
            self._seqs.popitem(last=False)
        seq = PleSequence(
            self.memory,
            history=history,
            keep_steps=self.keep_steps if keep_steps is None else keep_steps,
        )
        self._seqs[seq_id] = seq
        return seq

    def feed(
        self,
        seq_id: Any,
        tokens: Iterable[Any],
        *,
        as_tensor: bool = False,
    ) -> PleStep:
        seq = self.get_or_create(seq_id)
        return seq.feed(tokens, as_tensor=as_tensor)

    def get(self, seq_id: Any) -> PleSequence:
        if seq_id not in self._seqs:
            raise KeyError(f"no PleSequence for id {seq_id!r}")
        self._seqs.move_to_end(seq_id)
        return self._seqs[seq_id]

    def current_e_t(self, seq_id: Any) -> Any:
        return self.get(seq_id).current_e_t()

    def current_raw(self, seq_id: Any) -> bytes:
        return self.get(seq_id).current_raw()

    def remove(self, seq_id: Any) -> None:
        self._seqs.pop(seq_id, None)

    def clear(self) -> None:
        self._seqs.clear()

    @property
    def sequences(self) -> dict[Any, PleSequence]:
        return dict(self._seqs)

    def __len__(self) -> int:
        return len(self._seqs)

    def stats(self) -> dict[str, Any]:
        return {
            "sequences": len(self._seqs),
            "max_sequences": self.max_sequences,
            "keep_steps": self.keep_steps,
            "memory_source": self.memory.source,
        }

    def close(self) -> None:
        self.clear()


def ple_memory_from_discovery(
    info: dict[str, Any],
    *,
    store: Any | None = None,
    view: Any | None = None,
    slot_index: Any | None = None,
    head_dim: int | None = None,
    scale: float | None = None,
    **kwargs: Any,
) -> PleMemory:
    """Build a :class:`PleMemory` from ``discover_ple`` metadata.

    This is the service-layer counterpart to ``disk_ple_from_discovery``; it
    accepts both Store-I and Store-P sources.
    """
    if not info:
        raise ValueError("PLE discovery returned no metadata")

    ngram_size = int(info.get("ngram_size") or _PLE_NGRAM_SIZE)
    heads_per_ngram = int(info.get("heads_per_ngram") or _PLE_HEADS_PER_NGRAM)
    num_heads = (ngram_size - 1) * heads_per_ngram
    ple_embed_dim = int(info.get("ple_embed_dim") or 0)
    if ple_embed_dim <= 0:
        raise ValueError("discovery metadata is missing ple_embed_dim")

    if head_dim is None:
        head_dim = ple_embed_dim // num_heads
    base = int(info.get("ngram_vocab_size_base") or _PLE_BASE)
    prime_sizes = _head_vocab_sizes(num_heads, base=base)
    multipliers = None
    for key in ("layer_multipliers", "rowid_multipliers", "multipliers"):
        if info.get(key) is not None:
            multipliers = [int(x) for x in info[key]]
            break
    if scale is None:
        scale = float(info.get("weight_scale") or 1.0)

    return PleMemory(
        store=store,
        view=view,
        slot_index=slot_index,
        head_dim=head_dim,
        num_heads=num_heads,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        multipliers=multipliers,
        scale=scale,
        prime_sizes=prime_sizes,
        offsets=_head_offsets(prime_sizes),
        **kwargs,
    )
