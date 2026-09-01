"""SGLang-compatible page-reader adapter.

SGLang's NVMe PLE work exposes a Rust ``IoUringReader.read_pages(fds, offsets)``
shape.  This module provides the same shape from EngramDB's Python package so a
small upstream patch can swap ``sglang_storage.IoUringReader`` for this class.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import Store

try:
    from . import PageReader, IoUringPageReader
except ImportError:
    # A source checkout without a built native extension may not provide these.
    import engramdb as _engramdb_root

    PageReader = getattr(_engramdb_root, "PageReader", None)
    IoUringPageReader = getattr(_engramdb_root, "IoUringPageReader", None)


class SGLangPageReader:
    """Drop-in replacement for ``sglang_storage.IoUringReader``.

    On Linux this prefers the io_uring-backed ``IoUringPageReader``; elsewhere it
    falls back to the portable pread ``PageReader``.
    """

    def __init__(self, page_size: int = 4096, **kwargs: object) -> None:
        self.page_size = page_size
        self._kwargs = kwargs
        if IoUringPageReader is not None:
            self._impl = IoUringPageReader(page_size=page_size)
        elif PageReader is not None:
            self._impl = PageReader(page_size=page_size)
        else:
            raise RuntimeError("no EngramDB page reader is available on this platform")

    def read_pages(
        self,
        file_descriptors: Sequence[int],
        offsets: Sequence[int],
    ) -> list[bytes]:
        return self._impl.read_pages(list(file_descriptors), list(offsets))

    def close(self) -> None:
        """Compatibility no-op; EngramDB readers do not own resources."""


def install_sglang_io_uring_reader() -> bool:
    """Replace SGLang's ``IoUringReader`` with EngramDB's adapter when available.

    Returns True if the monkeypatch was applied, False if the SGLang storage
    module could not be imported in this process.
    """
    try:
        import sglang_storage
    except Exception:
        return False

    if not hasattr(sglang_storage, "IoUringReader"):
        return False

    sglang_storage.IoUringReader = SGLangPageReader
    return True



def install_sglang_ple(
    model_class: type,
    store: Store,
    attr_name: str,
    embedding_dim: int,
    dtype: Any = None,
    cache_size: int = 4096,
) -> type:
    """Patch an SGLang model class to use EngramDB for its PLE table.

    This is the no-source-change entry point: call it before constructing the
    SGLang model/engine.  It wraps ``model_class.__init__`` and replaces the PLE
    embedding attribute on each instance after normal construction.

    Example::

        from engramdb.sglang import install_sglang_ple
        install_sglang_ple(
            Gemma4Model,
            store=store,
            attr_name="embed_tokens_per_layer",
            embedding_dim=hidden_size_per_layer_input,
        )
        # then start SGLang normally
    """
    from .vllm_plugin import patch_model_class_ple

    return patch_model_class_ple(
        model_class,
        store=store,
        attr_name=attr_name,
        embedding_dim=embedding_dim,
        dtype=dtype,
        cache_size=cache_size,
    )
