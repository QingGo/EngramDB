
"""EngramDB: disk-first storage engine for Engram/PLE n-gram memory tables.

**必须使用原生 PyO3 扩展（``engramdb._engramdb``），没有纯 Python 回退。**

这里曾经有一份 ctypes C-ABI 回退实现，已删除（见 ``docs/roadmap.md`` §34）：
它是同一套 API 的**第二份、且 CI 从未覆盖过**的实现，会在扩展导入失败时
**静默降级**成子集功能，而不是报错。

C ABI 本身仍然保留 —— 作为 **C/C++ 嵌入面**（``crates/engramdb-python``），
那是与「Python 后端」不同的产品。
"""

from __future__ import annotations

from typing import Any

from .addressable_memory import AddressableNgramMemory, MemoryMatch

__version__ = "0.3.0"

# PyO3 是唯一后端，恒为 True；保留该名字供外部代码探测。
_USING_PYO3 = True

try:
    from ._engramdb import Store, View, read_keys
except ImportError as _exc:  # pragma: no cover - 取决于运行环境
    raise ImportError(
        "无法导入 EngramDB 的原生扩展 `engramdb._engramdb`，而它是必需的"
        "（本包没有纯 Python 回退）。\n"
        "\n"
        "该扩展随 wheel 一起分发，所以通常只有两种情况：\n"
        "  * 在源码树里跑但没构建过扩展 —— 执行：\n"
        "        cd python && maturin develop --release\n"
        "  * wheel 与当前 Python ABI 不匹配 —— 重装：\n"
        "        python -m pip install --force-reinstall engramdb-python\n"
        f"\n底层错误：{_exc!r}"
    ) from _exc

# 以下两个是平台可选面：PageReader 仅 unix，IoUringPageReader 仅 linux。
try:
    from ._engramdb import PageReader
except ImportError:
    PageReader = None  # type: ignore[assignment]

try:
    from ._engramdb import IoUringPageReader
except ImportError:
    IoUringPageReader = None  # type: ignore[assignment]

from .tables import Database
from .server import EngramDBServer, EngramDBBinaryServer
from .service_client import EngramDBClient
from .ple_discovery import (
    discover_ple,
    load_ple_multipliers,
    load_ple_weight_scale,
)
from .vllm import fetch_e_t_tensor
from .pool import StorePool, ThreadLocalStore
try:  # SlotIndex/DiskSlotIndex are pure-Python and need numpy; keep optional for light installs.
    from .slot_index import SlotIndex
    from .disk_slot_index import DiskSlotIndex
except ImportError:  # pragma: no cover - depends on environment
    SlotIndex = None  # type: ignore[assignment,misc]
    DiskSlotIndex = None  # type: ignore[assignment,misc]

PLE_QWEN_V1 = 1
ENG_DEEPSEEK_V1 = 2


def abi_version() -> int:
    """Return the EngramDB C ABI version implemented by this package."""
    if hasattr(_engramdb, "abi_version"):
        return int(_engramdb.abi_version())
    return 1


def rowids_for_seq(
    tokens: Any,
    ple_spec: int = PLE_QWEN_V1,
    multipliers: list[int] | None = None,
    info: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Return PLE/Engram rowids for a token sequence.

    The returned shape is ``[len(tokens), 16]`` for ``PLE_QWEN_V1``.  The
    implementation prefers the native Rust path (PyO3 or the C ABI) and falls
    back to the pure-Python reference when no native binding is installed.

    ``multipliers`` may be supplied directly, or taken from ``info`` as returned
    by :func:`engramdb.ple_discovery.discover_ple` (keys ``layer_multipliers``,
    ``rowid_multipliers``, or ``multipliers``).  When a non-default multiplier
    source is provided, the pure-Python path is used because the current C
    ABI/PyO3 rowid entry points implement only the standard Qwen PLE constants.
    """
    if ple_spec != PLE_QWEN_V1:
        raise NotImplementedError(
            f"ple_spec {ple_spec} is not implemented (only PLE_QWEN_V1=1)"
        )
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    tok = [int(x) for x in tokens]

    info_multipliers: list[int] | None = None
    if info is not None:
        for key in ("layer_multipliers", "rowid_multipliers", "multipliers"):
            val = info.get(key)
            if val is not None:
                info_multipliers = [int(x) for x in val]
                break

    custom_multipliers = multipliers is not None or info_multipliers is not None
    if not custom_multipliers:
        if hasattr(_engramdb, "rowids_for_seq"):
            return [list(r) for r in _engramdb.rowids_for_seq(tok, ple_spec)]
        # 否则落到下面的纯 Python 参考实现（ple_adapter.ple_rowids）

    from .ple_adapter import ple_rowids

    if multipliers is not None:
        effective = [int(x) for x in multipliers]
    elif info_multipliers is not None:
        effective = info_multipliers
    else:
        effective = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    return ple_rowids(tok, effective)


def rowids_for_seq_with_history(
    history: list[int],
    tokens: list[int],
    ple_spec: int = PLE_QWEN_V1,
) -> list[list[int]]:
    """Return PLE rowids for a streamed sequence with explicit n-gram history.

    This is the native fast path for sequential decode: ``history`` is the
    already-known previous context (usually ``ngram_size - 1`` tokens), and
    ``tokens`` are the current step's input ids.  Falls back to the pure-Python
    reference when the native binding is unavailable.
    """
    if ple_spec != PLE_QWEN_V1:
        raise NotImplementedError(
            f"ple_spec {ple_spec} is not implemented (only PLE_QWEN_V1=1)"
        )
    hist = [int(x) for x in history]
    tok = [int(x) for x in tokens]
    if _USING_PYO3 and hasattr(_engramdb, "rowids_for_seq_with_history"):
        return [list(r) for r in _engramdb.rowids_for_seq_with_history(hist, tok, ple_spec)]
    from .ple_adapter import ple_rowids

    return ple_rowids(tok, [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071], history=hist)


def __getattr__(name: str):
    if name == "DiskPleNGramEmbedding":
        from .ple_adapter import DiskPleNGramEmbedding
        return DiskPleNGramEmbedding
    if name in (
        "PleMemory",
        "PleSequence",
        "PleSequenceStore",
        "PleStep",
        "ple_memory_from_discovery",
    ):
        from .ple_memory import (
            PleMemory,
            PleSequence,
            PleSequenceStore,
            PleStep,
            ple_memory_from_discovery,
        )
        return {
            "PleMemory": PleMemory,
            "PleSequence": PleSequence,
            "PleSequenceStore": PleSequenceStore,
            "PleStep": PleStep,
            "ple_memory_from_discovery": ple_memory_from_discovery,
        }[name]
    if name in ("BundleManifest", "bundle_manifest_from_path"):
        from .bundle import BundleManifest, bundle_manifest_from_path

        return {
            "BundleManifest": BundleManifest,
            "bundle_manifest_from_path": bundle_manifest_from_path,
        }[name]
    if name in ("TargetReaderRegistry", "ReaderSpec"):
        from .target_reader import ReaderSpec, TargetReaderRegistry

        return {
            "TargetReaderRegistry": TargetReaderRegistry,
            "ReaderSpec": ReaderSpec,
        }[name]
    if name in (
        "PleMemoryAdapter",
        "TargetReaderHook",
        "install_target_reader_hook",
        "install_bundle_adapter",
        "install_vllm_target_reader",
        "install_sglang_target_reader",
    ):
        from .adapter import (
            PleMemoryAdapter,
            TargetReaderHook,
            install_bundle_adapter,
            install_sglang_target_reader,
            install_target_reader_hook,
            install_vllm_target_reader,
        )
        return {
            "PleMemoryAdapter": PleMemoryAdapter,
            "TargetReaderHook": TargetReaderHook,
            "install_target_reader_hook": install_target_reader_hook,
            "install_bundle_adapter": install_bundle_adapter,
            "install_vllm_target_reader": install_vllm_target_reader,
            "install_sglang_target_reader": install_sglang_target_reader,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
