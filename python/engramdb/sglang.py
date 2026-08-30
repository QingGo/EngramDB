"""SGLang-compatible page-reader adapter.

SGLang's NVMe PLE work exposes a Rust ``IoUringReader.read_pages(fds, offsets)``
shape.  This module provides the same shape from EngramDB's Python package so a
small upstream patch can swap ``sglang_storage.IoUringReader`` for this class.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from . import PageReader, IoUringPageReader


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
