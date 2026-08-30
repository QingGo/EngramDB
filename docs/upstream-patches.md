# Upstream Patch Sketches: SGLang / vLLM

This file records the *intended* upstream integration points. It is not an
applied patch: the checked-out SGLang/vLLM repositories do not currently contain
the upstream PLE NVMe/disk code, and this sandbox cannot write to those repos.

## 1. SGLang

Target upstream work: [sgl-project/sglang#36567](https://github.com/sgl-project/sglang/pull/36567)
(`sglang-storage` Rust + PyO3 `IoUringReader`).

### 1.1 Python-side drop-in

Where SGLang currently does:

```python
from sglang_storage import IoUringReader

reader = IoUringReader(page_size=4096)
pages = reader.read_pages(file_descriptors, offsets)
```

replace with:

```python
from engramdb.sglang import SGLangPageReader as IoUringReader

reader = IoUringReader(page_size=4096)
pages = reader.read_pages(file_descriptors, offsets)
```

`engramdb.sglang.SGLangPageReader` chooses `IoUringPageReader` on Linux and
falls back to `PageReader` elsewhere.  An in-process helper is available:

```python
from engramdb.sglang import install_sglang_io_uring_reader
install_sglang_io_uring_reader()
```

### 1.2 Rust-side replacement

For a real upstream contribution, the PR's Rust reader should be replaced by an
`engramdb-io` adapter:

- `BadgeGather` / `ViewReader` already speak EngramDB's Store-I / Store-P format.
- Linux batch path: `engramdb_io::UringBatchBackend` (per-file batch).
- Python-facing page API: keep `read_pages(fds, offsets)` as the stable boundary.

## 2. vLLM

Target upstream work: [vllm-project/vllm#54070](https://github.com/vllm-project/vllm/pull/54070)
(disk-backed PLE) and the blazux `Qwen3_8FlashNextNGramEmbedding` mmap patch.

### 2.1 Python-side prototype

The prototype in `engramdb.vllm_plugin` replaces a named embedding attribute with
an EngramDB-backed module:

```python
from engramdb import Store
from engramdb.vllm_plugin import patch_named_embedding

store = Store("path/to/engram-store", shards=1, rows_per_shard=N, width=W)
patch_named_embedding(
    model,
    "embed_tokens_per_layer",
    store=store,
    embedding_dim=hidden_size_per_layer_input,
)
```

If the vLLM model uses a custom PLE op/class instead of `nn.Embedding`, port the
same pattern into that class:

- dedup row ids before disk fetch,
- batch `Store.fetch`,
- expand back to original order,
- keep GPU staging out of the repeated Python path (or register as a splitting
  op for CUDA-graph capture).

### 2.2 Not yet handled

- custom CUDA op registration,
- pinned staging + async H2D,
- `PREWARM` / readahead policy,
- cgroup memory limits and deployment recipes.

These remain for real vLLM integration and hardware A/B testing.

## 3. Linux real-machine verification

The 0.2.2 PyPI wheel includes the SGLang/vLLM adapter modules. On a Linux box
(WSL or Raspberry Pi) run:

```bash
python3 -m pip install --upgrade engramdb-python==0.2.2

curl -sL https://raw.githubusercontent.com/QingGo/EngramDB/master/scripts/python_wheel_smoke.py \
  -o /tmp/engramdb_smoke.py
python3 /tmp/engramdb_smoke.py
```

The smoke tests:

- `engramdb.Store` + `PleDiskGather`
- `PageReader` / Linux `IoUringPageReader`
- `SGLangPageReader`
- importing `vllm_plugin` (torch optional)

If a real PLE store is available, additionally test the vLLM patch prototype with
a real model attribute:

```python
from engramdb import Store
from engramdb.vllm_plugin import patch_named_embedding

store = Store("/path/to/engram-rows", shards=..., rows_per_shard=..., width=...)
patch_named_embedding(model, "embed_tokens_per_layer", store, embedding_dim=...)
```

