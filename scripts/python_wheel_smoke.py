#!/usr/bin/env python3
"""Smoke-test an installed engramdb-python wheel.

This intentionally avoids PyTorch/engram-peft so it can run in a plain Python
environment on every wheel platform. It exercises the native extension plus the
SGLang/vLLM-facing helpers that are new in 0.2.1.
"""

from __future__ import annotations

import os
import tempfile

import engramdb


def test_page_reader() -> None:
    readers = [
        name
        for name in ("PageReader", "IoUringPageReader")
        if getattr(engramdb, name, None) is not None
    ]
    if not readers:
        print("No page reader available on this platform; skipping")
        return

    page_size = 16
    payload = bytes(range(page_size))
    tmp = tempfile.NamedTemporaryFile(prefix="engramdb-page-", delete=False)
    tmp.write(payload)
    tmp.close()

    try:
        fd = os.open(tmp.name, os.O_RDONLY)
        try:
            for name in readers:
                reader = getattr(engramdb, name)(page_size=page_size)
                pages = reader.read_pages([fd], [0])
                assert len(pages) == 1
                assert pages[0] == payload
                print(f"{name} OK")
        finally:
            os.close(fd)
    finally:
        os.unlink(tmp.name)


def test_store_and_vllm_gather() -> None:
    row_width = 8
    rows = [bytes([i] * row_width) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="engramdb-store-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)

        store = engramdb.Store(directory, 1, len(rows), row_width)
        try:
            assert store.width == row_width
            raw = store.fetch([2, 0, 2, 1])
            assert raw == rows[2] + rows[0] + rows[2] + rows[1]

            from engramdb.vllm import PleDiskGather

            gather = PleDiskGather(store, row_bytes=row_width)
            expanded = gather.fetch([2, 0, 2, 1])
            assert expanded == rows[2] + rows[0] + rows[2] + rows[1]
            unique = gather.fetch_unique([2, 0, 2, 1])
            assert unique == rows[2] + rows[0] + rows[1]
            print("Store + PleDiskGather OK")
        finally:
            store.close()


def main() -> None:
    assert engramdb.__version__.startswith("0.2.1"), engramdb.__version__
    # Importing every public integration surface catches missing/renamed symbols.
    from engramdb.vllm import PleDiskGather  # noqa: F401
    try:
        from engramdb import integrations  # noqa: F401
        print("integrations import OK")
    except Exception as exc:  # optional torch/engram-peft dependency
        print(f"integrations skipped ({exc})")

    test_page_reader()
    test_store_and_vllm_gather()
    print("python wheel smoke OK")


if __name__ == "__main__":
    main()
