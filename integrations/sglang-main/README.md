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

---

# Using this: who should, and how

## First — most people should not use `engramdb`

| your machine | backend | why |
|---|---|---|
| ≥48 GiB of mlock-able host RAM | **`pinned`** (upstream default) | fastest by far: **~50 µs/step and flat in batch size** (30.1 / 67.4 / 48.1 / 51.7 µs at 1/8/32/128 tokens). Nothing here beats it. |
| GB10 / DGX Spark class (HMM) | **`file`** (upstream) | kernel walks host page tables, no host round trip |
| neither of the above, and the table must come off the device | **`engramdb`** (0002) or **`file-staged`** (0001) | the only options that work, and they cost a host round trip per gather |

The order matters: `engramdb` exists for machines where the table cannot be
resident *and* the GPU cannot dereference pageable host memory. If that is not
your machine, it is strictly slower than what you already have.

## The pipeline

1. **Get the checkpoint** — `Qwen3.8-Flash-Next-FP8` (the PLE table lives in
   `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`).
   Only those shards are needed for the store, but the model itself has to run
   somewhere.

2. **Extract the rows into a store** (~51.2 GB, sequential read+write, ~10 min):

   ```sh
   python scripts/extract_ple_rows.py <checkpoint_dir> <store_dir>
   # -> <store_dir>/shard_000.bin .. shard_127.bin, 400001920 B each
   ```

   Plus a `manifest.json` describing `num_shards` / `expected_shard_bytes`; the
   backend reads it to derive `rows_per_shard`, and falls back to globbing
   `shard_*.bin` if it is absent.

3. **Apply the patches to a source checkout of SGLang `main`** (`14b647c`):

   ```sh
   git clone --depth 1 https://github.com/sgl-project/sglang
   cd sglang && git checkout 14b647cf27d7f2c1a3764841f7d3770ff9f9e7d6
   patch -p1 < 0001-ple-offload-file-staged.patch
   patch -p1 < 0002-ple-offload-engramdb-store.patch
   ```

   No rebuild: nothing here is a kernel.

4. **Run** (needs `pip install engramdb`, v0.3.0 or later):

   ```sh
   PYTHONPATH=$PWD/python python -m sglang.launch_server \
     --model-path <checkpoint_dir> \
     --ple-offload-embedding \
     --ple-offload-backend engramdb \
     --ple-offload-dir <store_dir> \
     --cuda-graph-backend-decode breakable \
     --cuda-graph-backend-prefill breakable
   ```

   The `breakable` flags are **required**, not tuning: `gather` is a host round
   trip and capture fails outright without the break. Add `--disable-cuda-graph`
   instead if you want eager — that also works, and it is what the class-level
   hooks in the main README do.

## What to expect

Against τ(1) ≈ 325 µs (the lead time the model gives the PLE layer at index 1),
warm 1- and 8-token steps fit; everything else does not, and the cost lands on
the critical path in full:

| tokens | rows | warm | cold |
|---|---|---|---|
| 1 | 16 | 196.5 | 406.5 |
| 8 | 128 | 263.9 | 1247.5 |
| 32 | 512 | 560.3 | 3536.0 |
| 128 | 2048 | 1094.9 | 10603.8 |

That is a **service ceiling, not a speedup**: a 128-sequence decode step pays
1.095 ms, so PLE alone caps aggregate throughput near 117k tok/s; a single
sequence warm is ~5.1k tok/s, cold ~2.5k tok/s. It is usable; it does not feel
resident.

## Not yet

- **Nothing here is upstreamed.** Both patches are diffs against a checkout, and
  the machine that runs this must build a 51.2 GB store first. There is no
  released turnkey path today.
- Without the patches, EngramDB still works with vLLM and SGLang through the
  class-level hooks in the repository README (`DiskPleNGramEmbedding`,
  `install_real_qwen_ple_embedding`) — but **in eager mode**. vLLM's graph mode
  cannot execute a host-issued read at replay under `FULL_AND_PIECEWISE`, which
  is why this work moved to SGLang.
