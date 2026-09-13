# EngramDB ⇄ SGLang: a file-backed PLE table on devices without HMM

Target tree: `sgl-project/sglang@14b647cf27d7f2c1a3764841f7d3770ff9f9e7d6` (main, 2026-09-13).

`0001-ple-offload-file-staged.patch` adds a third value to
`--ple-offload-backend` so that a disk-backed PLE n-gram table can be served by
a GPU that cannot dereference host memory — every discrete part, i.e. 4090,
H100, A100, and not GB10.

## The seam, in five files

SGLang `main` already ships a file-backed PLE table. `check_file_backend_supported`
(`models/qwen4_exp_ple_table.py:321`) refuses to load it unless the device
reports `cudaDevAttrPageableMemoryAccessUsesHostPageTables` (attr 100), because
the gather kernel `_gather_ple_embedding_from_pinned_kernel`
(`models/qwen4_exp.py:736`) dereferences the mapping through the device's
address space. That is the whole gap: the storage layer is done, the
*dereference* has exactly one implementation.

Upstream separates storage from nothing; this patch separates **storage** from
**access**:

| | storage | who dereferences | device requirement |
|---|---|---|---|
| `pinned` | pinned host RAM | device, via UVA | `cudaHostGetDevicePointer` on pinned memory |
| `file` | sparse file, mmap | device, via host page tables | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` |
| **`file-staged`** | **same sparse file** | **the host** | none beyond a readable file and pinned RAM |

```
                                    allocate_ple_host_table()        <- shared
                                              |
                        +---------------------+---------------------+
                        |                                           |
        Qwen4ExpPinnedHostEmbedding                 Qwen4ExpStagedFileEmbedding
        gather(): Triton kernel reads                gather(): host reads rows into
        the mapping through the device               pinned staging, one pinned->dev copy
        no graph break                               eager_on_graph -> a graph break
                        |                                           |
                        +---------------------+---------------------+
                                              |
                       PleFilePrefetcher (fadvise WILLNEED), PleFileRssTrimmer
                       (MADV_DONTNEED, budget)                       <- shared
```

Files touched:

| file | change |
|---|---|
| `models/qwen4_exp_ple_table.py` | `PLE_OFFLOAD_BACKENDS` + `file-staged`; `PLE_FILE_BACKENDS`; `ple_file_access_mode()`; `check_ple_offload_backend_supported()`; the `file` error now names the fallback |
| `models/qwen4_exp.py` | `Qwen4ExpStagedFileEmbedding`; `_resolve_gather_output()` extracted out of `gather` so both backends share it; backend selection in `Qwen4ExpPLELayer.__init__` |
| `arg_groups/fields/exec_.py` | `--ple-offload-backend` gains the value and documents the breakable-graph requirement |
| `arg_groups/memory_hook.py` | the "file requires --ple-offload-embedding" check covers both file backends |
| `model_executor/model_runner_components/load_model_utils.py` | the load-time check dispatches on the backend instead of assuming `file` |

## Why `gather` has to be a graph break

`Qwen4ExpPLELayer.start_prefetch` (`qwen4_exp.py:1145`) is called from the model
loop at `qwen4_exp.py:1688`, at the top of iteration *i*, for the PLE layer of
layer *i+1* — so the lead time is exactly **one decoder layer**.

The ids are produced on the device (`compute_ngram_ids` → `_hash_contexts`), so
a host read needs a device→host copy first, and a device→host copy is not
capturable. With the break disarmed, capture fails outright:

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

That is the honest statement of the cost: `file-staged` needs
`--cuda-graph-backend-decode breakable` (and the prefill equivalent), and the
read it performs is **not** hidden behind the preceding layer — see
`probes/ple_sglang_main_session45.md` for the measured exposure.

## Applying

```sh
git clone --depth 1 https://github.com/sgl-project/sglang
cd sglang && git checkout 14b647cf27d7f2c1a3764841f7d3770ff9f9e7d6
patch -p1 < .../integrations/sglang-main/0001-ple-offload-file-staged.patch
```

Run it straight out of the source tree; no rebuild is needed, because nothing
here is a kernel:

```sh
PYTHONPATH=$PWD/python python -m sglang.launch_server \
  --model-path <Qwen3.8-Flash-Next> \
  --ple-offload-embedding --ple-offload-backend file-staged \
  --cuda-graph-backend-decode breakable --cuda-graph-backend-prefill breakable
```

## What is *not* in this patch

The read is exposed, so a cold decode step pays it. Closing that needs the ids
on the host *before* the step's device work — which means computing the n-gram
hash host-side from the scheduler's token history instead of from the device
pool. That is the next patch, not this one; `probes/ple_sglang_main_session45.md`
records the measurement that motivates it.

---

# 0002 — the `engramdb` backend

`0002-ple-offload-engramdb-store.patch` (applies on top of 0001) adds a fourth
value, `--ple-offload-backend engramdb`, which reads PLE rows out of an existing
EngramDB store directory (`--ple-offload-dir`) instead of allocating a table of
its own.

It exists because the other three backends all need the table to be *one thing*:
`pinned` needs it resident, `file`/`file-staged` need it as one sparse file of
`padded_vocab × 160` B — 51.2 GB, which is exactly what a machine that cannot
hold the table also usually cannot stage to disk. A store is 128 shards that are
already there, and its own reader is concurrent.

```
                    Qwen4ExpPinnedHostEmbedding (device-side gather)
                              |
        Qwen4ExpStagedFileEmbedding (host gather + graph break)
                              |
        Qwen4ExpEngramDbEmbedding (host gather + graph break, rows from a store)
                              |
                    engramdb.Store.fetch(rowids) -> bytes
```

`open_ple_engramdb_store` reads `manifest.json` for `num_shards` and
`expected_shard_bytes` (falling back to globbing `shard_*.bin`), derives
`rows_per_shard = shard_bytes / row_bytes`, and **checks the store's shape
against the engine's** — `width` against `head_dim_per_ngram × dtype size`, and
`total_rows` against the engine's padded vocabulary — because a mismatch would
otherwise be served silently as the wrong rows. `engramdb` is imported lazily, so
the other three backends keep working without it installed.

## The one non-obvious part: where the fp8→bf16 cast happens

The staging slab is kept in the **table's** dtype and the cast happens on the
device, so the publish is two plain same-dtype copies rather than one that also
casts:

| publish, 2048 rows | µs |
|---|---|
| one cross-device copy that also casts | 456.5 |
| bf16 slab, same-dtype copy (CPU cast first: 272.3) | 41.6 |
| **two same-dtype copies, device cast** | **38.6** |

The cast itself is ~18 µs on the GPU. Making `copy_` do it is what costs. Getting
this backwards — moving the cast off the CPU but leaving it inside the
cross-device copy — is measurably *worse* than not optimizing at all
(1468 µs vs 1265 at 2048 rows).

Measured end to end on a 4090 against the real Qwen3.8-Flash-Next geometry and
the real 51.2 GB store: warm **196.5 / 263.9 / 560.3 / 1094.9 µs** for
1 / 8 / 32 / 128 tokens, all replay-correct against an independent `pread` of the
raw shards. See `probes/ple_sglang_main_engramdb_session46.md`.
