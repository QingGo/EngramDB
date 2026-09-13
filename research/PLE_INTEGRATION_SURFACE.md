# Disk-backed PLE / n-gram gather: exact integration surface in vLLM and SGLang

Research date: HEADs of `vllm-project/vllm@main` and `sgl-project/sglang@main` as of this session.
All file paths are repo-relative. **[C]** = CONFIRMED (I read the file/diff/API payload). **[U]** = UNCONFIRMED (inferred).

---

## 0. Three premise corrections you need before reading the answers

**P0.1 — `vllm#54070` IS a disk PR, but there is a *second*, unrelated disk PR you did not name.** **[C]**

| PR | Title | Env var | State | Head (fork) | Created |
|---|---|---|---|---|---|
| [#53899](https://github.com/vllm-project/vllm/pull/53899) | Support PLE-Offload for Qwen3.8-Flash-Next | `VLLM_PLE_CPU_OFFLOAD` | open | `peakcrosser7:release/qwen38next_offload` | 2026-08-26 |
| [#54070](https://github.com/vllm-project/vllm/pull/54070) | **[Feature] Disk-backed PLE n-gram tables (`VLLM_PLE_DISK_OFFLOAD_DIR`)** | `VLLM_PLE_DISK_OFFLOAD_DIR` | open | `jagat-primitive-org/feat/ple-disk-offload` | 2026-08-27 |
| [#54129](https://github.com/vllm-project/vllm/pull/54129) | **[Model] Support disk-backed (mmap) PLE table … (`VLLM_PLE_MMAP`)** | `VLLM_PLE_MMAP` | open | `Trosfy:ple-mmap-upstream` | 2026-08-28 |
| [#54371](https://github.com/vllm-project/vllm/pull/54371) | [Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism | `VLLM_PLE_CPU_OFFLOAD` / `--engram-config` | **MERGED** 2026-09-09 | `peakcrosser7` | 2026-08-29 |

#54070 is 151 lines stacked on top of #53899 (`worker.py` + `envs.py`). #54129 is a **completely independent** disk design touching a disjoint file set. #54371's "Not a duplicate" section compares against **#54129**, not #54070 — so if you were matching PR numbers to designs from that body, the mapping is off by one PR.

**P0.2 — SGLang `main` has no `python/sglang/srt/layers/engram.py`.** **[C]**
`GET /git/trees/main?recursive=1` returns `truncated: false` and **zero** paths containing "engram"; `GET /contents/python/sglang/srt/layers/engram.py?ref=main` → HTTP 404. `EngramHasher` lives on the **`dsv4.1` branch** at `python/sglang/srt/layers/engram.py` (972 lines). Q6 is answered against that branch and labelled accordingly.

**P0.3 — SGLang `main` ALREADY ships a file-backed, disk-backed PLE table.** **[C]**
`python/sglang/srt/models/qwen4_exp_ple_table.py`, selected by `config.ple_offload_backend == "file"`, with `PleFilePrefetcher` (`posix_fadvise(WILLNEED)`) and `PleFileRssTrimmer` (`MADV_DONTNEED`, 8 GiB default budget). Details in Q7. #36567 is a *second, deeper* NVMe path on a side branch, not the first one.

---

## 1. vLLM: the class that owns the PLE gather, and where it sits vs CUDA graph capture

**Ownership chain (vLLM `main`, post-#54371).** **[C]**

```
vllm/models/qwen4_exp/nvidia/model.py:195-208
  Qwen4ExpPLELayer(...)                      # created when (layer_idx+1) in config.ple_layer_ids
    └─ vllm/models/qwen4_exp/nvidia/ple_layer.py:66  class Qwen4ExpPLELayer(nn.Module, MambaBase)
         self.ple_embedding = Qwen4ExpNGramEmbedding(...)      # ple_layer.py:96-105
         └─ vllm/models/qwen4_exp/nvidia/ngram_embedding.py:535  class Qwen4ExpNGramEmbedding(nn.Module)
              self.ngram_embedding ∈ {
                  Qwen4ExpPLEDeviceEmbedding      (ngram_embedding.py:325)   # GPU-resident
                  Qwen4ExpPLEPinnedHostEmbedding  (ngram_embedding.py:387)   # pinned CPU + UVA
              }                                  # both → Qwen4ExpPLEEmbedding(:52)
                                                 #      → PLEVocabParallelEmbedding (common/ple.py)
                                                 #      → VocabParallelEmbedding
```

**The two concrete gather sites.** **[C]**

*GPU-resident:* `Qwen4ExpPLEDeviceEmbedding.forward` (`ngram_embedding.py:345`) → `super().forward(gathered_ids)` → `VocabParallelEmbedding.forward` → `F.embedding`. **A plain host-side tensor index, captured inside the CUDA graph.** No custom op, no break.

*Pinned-host / UVA:* `Qwen4ExpPLEPinnedHostEmbedding._lookup` (`ngram_embedding.py:445`) launches the Triton kernel `_lookup_ple_embedding_from_pinned_kernel` (`ngram_embedding.py:357`) against `self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)` (`ngram_embedding.py:418`), where `self.weight` came from `torch.empty(..., device="cpu", pin_memory=True)` (`ngram_embedding.py:437-443`).

**Graph placement.** **[C]**

| function | file:line | decorator | in-graph? |
|---|---|---|---|
| `Qwen4ExpNGramEmbedding.forward` | `ngram_embedding.py:839` | none | in-graph (`if embedding.supports_prefetch: return embedding(hidden_states)`) |
| `Qwen4ExpPLEPinnedHostEmbedding.start_prefetch` | `ngram_embedding.py:492` | `@eager_break_during_capture` | **conditional break** |
| `Qwen4ExpPLEPinnedHostEmbedding._finalize_prefetch` | `ngram_embedding.py:508` | `@eager_break_during_capture` | **conditional break** |
| `Qwen4ExpPLEPinnedHostEmbedding.forward` | `ngram_embedding.py:526` | none | in-graph (joins side stream, ETP all-reduce, row select) |

`eager_break_during_capture` (`vllm/compilation/breakable_cudagraph.py:66`) is **not** an unconditional break. **[C]**

```python
def eager_break_during_capture(fn):
    if not is_breakable_cudagraph_enabled():   # line 97 — VLLM_USE_BREAKABLE_CUDAGRAPH
        return fn                              # line 98 — NO break at all
    ...
        if mode == CUDAGraphMode.FULL:
            return fn(*args, **kwargs)         # line 109-110 — NO break in FULL either
```

So: **with breakable CUDA graphs disabled (the default), the UVA gather path introduces no graph break whatsoever.** With breakable graphs on, `start_prefetch` becomes an eager segment only in non-FULL (PIECEWISE) modes.

**The offload path (#53899 / #54070) introduces no graph break either — and does not even use `eager_break_during_capture`.** **[C]** In the #53899 diff, `Qwen4ExpNGramEmbedding` is reparented from `nn.Module` to `PleOffloadLayer` (patch hunk `+class Qwen4ExpNGramEmbedding(PleOffloadLayer):`). On the GPU worker the layer's `__init__` is short-circuited (`PleOffloadLayer.__init_subclass__` → `guarded_init`, `ple_offload_layer.py`), and its `forward` becomes:

```python
def forward(self, hidden_states, input_ids, *args, **kwargs):
    if self._is_cpu_offloaded:
        torch.ops.vllm.ple_offload_wait(self._sem.flag_tensor, self._gpu_output_buffer, hidden_states)
        return self._gpu_output_buffer[: input_ids.shape[0]]
    return self.forward_impl(...)
```

`vllm::ple_offload_wait` is registered with `direct_register_custom_op(..., fake_impl=_ple_offload_wait_fake)` — a fake impl exists, so torch.compile treats it as an opaque node, and the docstring in that file states explicitly: `WaitValue32(flag==1)  <- in Graph`. The wait is a `cuStreamWaitValue32` **inside** the captured graph.

---

## 2. What UVA PLE-Offload (#54371) does — and whether it removes the need for disk backing

**It does NOT remove the need for disk backing. It requires the complete table resident in pinned host RAM.** **[C]**

Key sentences, quoted verbatim from the PR body ([API](https://api.github.com/repos/vllm-project/vllm/pulls/54371)):

> "`cpu_offload=true` stores each local PLE weight shard in pinned CPU memory. The GPU reads the required rows directly through CUDA Unified Virtual Addressing (UVA), using a dedicated CUDA stream."

> "`#53899` uses a dedicated CPU offload worker and transfers selected rows back to GPU workers. **This PR keeps the table in process-local pinned memory and lets the GPU read the required rows directly through UVA**, without an offload worker or IPC row transfer."

> "`#54129` uses disk-backed mmap storage to reduce host-memory residency. **This PR intentionally keeps the complete local shard pinned in RAM for direct UVA access.**"

Corroborated in code **[C]**:
- `Qwen4ExpPLEPinnedHostEmbedding.allocate_embedding_weight` returns `torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu", pin_memory=True)` — the *entire* table.
- `if not is_uva_available(): raise RuntimeError("Engram CPU offload requires UVA support")` where `is_uva_available()` is literally `is_pin_memory_available() or current_platform.is_cpu()` (`vllm/utils/platform_utils.py:51-57`).
- `get_accelerator_view_from_cpu_tensor` → `torch.ops._C.get_cuda_view_from_cpu_tensor` → `csrc/libtorch_stable/cuda_view.cu` calls **`cudaHostGetDevicePointer`**, and falls back to `cudaHostAlloc` + `cudaMemcpy` if `aten::is_pinned` is false. **[C]**

**Consequence for a disk-backed store [C for the mechanism, U for your deployment]:** a plain `torch.from_file` / `np.memmap` mapping is pageable, not pinned. Handing it to `get_cuda_view_from_cpu_tensor` hits the non-pinned branch, which `cudaHostAlloc`s a *new* full-size buffer and `cudaMemcpy`s the whole table into it — precisely the ~95 GB RAM cost you are trying to avoid. Making a file mapping UVA-visible therefore requires either `cudaHostRegister(..., cudaHostRegisterMapped)` (which pins it, defeating paging) **or** a platform whose pageable host memory is directly addressable via host page tables (HMM/ATS — what SGLang gates on with `cudaDevAttrPageableMemoryAccessUsesHostPageTables`). I did **not** find any vLLM code that registers a file mapping; I am labelling "vLLM #54371 cannot serve from a pageable file mapping" as **[C] for the code path, [U] for whether some CUDA/driver configuration makes `cudaHostGetDevicePointer` succeed on unregistered memory.**

---

## 3. What #54070 changes, mechanically

**The module that does the swap is `vllm/v1/ple_offload/worker.py`, in the CPU offload process — not in the gather path.** **[C]**

New symbols added by #54070 (all in `vllm/v1/ple_offload/worker.py`, visible in the [files patch](https://api.github.com/repos/vllm-project/vllm/pulls/54070/files)):

| symbol | role |
|---|---|
| `_ple_disk_dir()` | `return envs.VLLM_PLE_DISK_OFFLOAD_DIR or None` |
| `_ple_disk_shard_of(mapped_name)` | `"<layer>.a.b.shard_3.weight"` → `"<layer>.a.b"` via `re.match(r"^(.*)\.shard_\d+\.weight$", ...)` |
| `_disk_backed_tensor(path, shape, dtype, writable)` | `np.memmap(path, dtype=np_dtype, mode="r+" if writable else "c", shape=shape)` + `arr._mmap.madvise(_mmap.MADV_RANDOM)` + `torch.from_numpy(arr).view(dtype)`; keeps the array alive in `_PLE_DISK_MAPS` |
| `_ple_disk_attach(layer_name, layer, disk_dir)` | size gate `numel * element_size < (1 << 30)` → skip; picks the **largest** param; writes `<dir>/<layer>.<param>.bin` (+ `.done.json` sidecar); swaps in the mapping |
| `_ple_disk_finalize(...)` | `arr.flush()`, write `.done.json`, re-map with `writable=False` (copy-on-write) |

The swap itself, verbatim from the diff:

```python
    mapped = _disk_backed_tensor(bin_path, shape, dtype, writable=not complete)
    # Replace the parameter data in place; module structure and names are unchanged,
    # so load_weights and the gather path are untouched.
    owner = layer
    parts = pname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    getattr(owner, parts[-1]).data = mapped
```

Call sites inside `PleOffloadRunner.__init__` (weight discovery) and after `process_weights_after_loading`. Shard tensors that would overwrite an already-complete table are skipped in `offload_only_iter`:

```python
                    if disk_complete_tables:
                        table = _ple_disk_shard_of(mapped_name)
                        if table is not None and table.startswith(disk_complete_tables):
                            continue
```

**Gather path untouched: CONFIRMED.** **[C]** The disk change is exactly `worker.py` + `envs.py` (the PR body says "the disk-offload change itself is the final commit (151 insertions, worker.py + envs.py)"); the CPU-side gather is `Qwen4ExpNGramEmbedding.forward_impl` doing `torch.index_select(self.ngram_embedding.weight, 0, ngram_ids.reshape(-1), out=...)` — an ordinary CPU tensor index that happens to be file-backed. The GPU-side `PleOffloadLayer.forward` and the IPC row transfer are byte-identical to #53899.

Two dials in `vllm/envs.py` added by #54070: `VLLM_PLE_DISK_OFFLOAD_DIR: str = ""`, and (from #53899) `VLLM_PLE_CPU_OFFLOAD: bool = False`, `VLLM_PLE_OFFLOAD_READY_TIMEOUT: float = 600.0`. The env count delta (`envs.py +22` in #54070 vs `+11` in #53899) is exactly the disk variable's comment block.

---

## 4. Does the #54070 disk path depend on CUDA graph mode?

**No graph dependency of its own. There is no graph-related code or comment anywhere in the #54070-specific diff.** **[C]** I grepped the whole #54070 patch set for `cudagraph|cuda_graph|PIECEWISE|FULL|enforce_eager|graph` and the only hits outside `tests/` are in the files it *inherits* from #53899 (`model_runner.py`'s `if batch_desc.cg_mode == CUDAGraphMode.FULL:` context line, and `ple_offload_layer.py`'s `torch.compile(fullgraph=True)` comment).

**Does it work with `cudagraph_mode=FULL_AND_PIECEWISE` / `enforce_eager=False`?** **[C] for the mechanism, [U] for a direct #54070 test at FULL.**

Evidence that it does:
- The disk swap only changes *what backs* a CPU tensor in the offload process. Disk and RAM offload are indistinguishable to the GPU side, and the GPU side is designed for capture: `execute_model` calls `PleOffloadConnector.prepare_forward(...)` on the step stream **before** `if batch_desc.cg_mode == CUDAGraphMode.FULL: ... replay`, and `release_outputs()` after the model runs. `capture_model` calls `signal_dummy_outputs(self.max_num_tokens)` before decoder capture and `release_outputs()` after. The in-graph primitive is the fake-impl custom op `torch.ops.vllm.ple_offload_wait` doing `cuStreamWaitValue32(..., DONE_VALUE, EQ)`.
- #53899's test plan (which #54070 includes) says: "For each combination, verify storage and topology against the table, **exercise CUDA Graph capture and replay with uneven DP loads (including an idle replica)**, and run GSM8K." **[C]**
- #54070's own test plan (verified list) covers first boot, reboot, and gather equivalence against the BF16 path, and states "No existing pytest suite covers `v1/ple_offload` yet (it lands in #53899)". **[C]**

Caveat I could not resolve — flag it as **[U]**: the `VLLM_PLE_CPU_OFFLOAD` docstring added in #53899/#54070 says *"The initial implementation supports ModelRunner V1 and single-node TP only."* But the only runner file either PR modifies is `vllm/v1/worker/gpu/model_runner.py`, which is the **Model Runner V2** runner (`vllm_config.use_v2_model_runner`, `model_state.prepare_inputs`). I could not reconcile that comment with the code; treat the "V1 only" claim as stale/unverified. **[U]**

---

## 5. SGLang #36567 — what its NVMe path actually does

PR: [#36567](https://github.com/sgl-project/sglang/pull/36567) "feat(qwen4): stream PLE embeddings from NVMe", **open**, head `feat/qwen4-nvme-ple`, **base `qwen4-main-squashed`** (stacked on #36497, not on `main`). **[C]**

**Mechanism: `io_uring` (default) OR `mmap` (fallback/debug) — not `pread`.** **[C]**

- Rust extension `rust/sglang-storage/` (`Cargo.toml`, `src/lib.rs`, `src/io_uring_reader.rs`): a persistent `io_uring::IoUring`, a `posix_memalign`'d `AlignedBuffer` (page-aligned), fds opened `O_RDONLY | O_DIRECT`, `opcode::Read::new(Fd(fd), page_ptr, page_size).offset(offset)` submitted in `queue_depth` batches with `submit_and_wait(batch)`, GIL released, OS errno preserved into `PyOSError`. Exposed as `IoUringReader(queue_depth, max_batch, page_size)` via `sglang.srt.rust_extensions._storage`.
- Python `python/sglang/srt/models/qwen4_ple_nvme.py` (new, 626 lines):
  - `PLEManifest.from_snapshot()` parses `model.safetensors.index.json` + raw safetensors headers (`_read_safetensors_header`) without loading tensors; `locate(row_id) -> RowLocation(path, offset, nbytes)`.
  - `IoUringPageRowReader` reads 4096-byte pages through the ring, with an LRU page cache; `MMapRowReader` is the "portable correctness/debug backend backed by the page cache".
  - `NVMePLEEmbedding(nn.Module)` is the module swapped in for `VocabParallelEmbedding` in `Qwen4ExpNGramEmbedding.__init__` (`qwen4_exp.py` diff), gated on `SGLANG_QWEN4_PLE_NVME_PATH`. TP1 and `F8_E4M3` only, enforced at construction.
- Env vars (`python/sglang/srt/environ.py`): `SGLANG_QWEN4_PLE_NVME_PATH` (default `""`, opt-in), `_BACKEND` (default `io_uring`), `_QUEUE_DEPTH=512`, `_MAX_BATCH_PAGES=4096`, `_CACHE_PAGES=0`, `_LOG_INTERVAL=1000`.

**Reads are issued HOST-SIDE — yes, there is a break — and YES it requires the breakable CUDA graph backend.** **[C]**

```python
    @eager_on_graph(True, capture_stub=_capture_start_gather)
    def start_gather(self, input_ids):
        row_ids = input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64).tolist()
        return PendingGather(self._io_executor.submit(self._timed_read_rows, row_ids), ...)

    @eager_on_graph(True, capture_stub=_capture_finish_gather)
    def finish_gather(self, pending, device, out=None, stream=None):
        rows = pending.future.result()            # blocks on the ThreadPoolExecutor
        ...
        ctypes.memmove(stage.data_ptr(), raw, len(raw))   # pinned staging
        with stream_context:
            device_bytes = stage.to(device=device, non_blocking=True)
            ...
            if is_in_breakable_cuda_graph():
                self._stage_event.synchronize()
```

`eager_on_graph` is `sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph.eager_on_graph` — it reads `_current_capture_var` (a `ContextVar` set **only** by `BreakableCUDAGraphCapture.__enter__`) and, when set, ends the current segment, runs the body eagerly, and appends a `replay_fn` to `capture.cuda_graph._break_fns`. **[C]**

The wiring in `Qwen4ExpPLELayer.start_prefetch` / `_consume_prefetched_embeddings` confirms the intent: `pending_nvme = offloaded_embedding.start_gather(lookup_ids)` on the host, and `finish_gather(pending_nvme, ...)` at consume time, on a side stream, followed by `torch.cuda.current_stream().wait_stream(self._prefetch_stream)`.

**Important caveat [U]:** because `_current_capture_var` is `None` outside the breakable backend, calling `start_gather`/`finish_gather` under a *plain* (non-breakable) CUDA graph capture would execute the real body — a D2H `.tolist()`, a Python thread-pool submit, and a blocking `future.result()` — during capture. The PR's own end-to-end validation was run with **"eager execution"** (its words). So: the code is written to *depend* on the breakable graph backend, and the analogue PR #39205 states this requirement outright for the DeepSeek-V4.1 Engram path ("Keep hash/prefetch and waits outside breakable CUDA Graphs… both `--cuda-graph-backend-prefill breakable` / `--cuda-graph-backend-decode breakable`"), but I did **not** find a `FULL_AND_PIECEWISE` capture test in #36567's diff. **[U]**

---

## 6. SGLang `srt/layers/engram.py` / `EngramHasher`

**Location: `dsv4.1` branch, `python/sglang/srt/layers/engram.py` (972 lines). NOT on `main`.** **[C]** — see P0.2. Sibling kernels on the same branch: `python/sglang/kernels/ops/embeddings/engram_hash.py`, `engram_gather.py`, `engram_gate.py`.

**`EngramHasher`** — `python/sglang/srt/layers/engram.py:200`. **[C]**

> `"""Hash ids for every token of a forward batch, [T, n_engram_layers, n_hash_cols]."""`

Members: `__init__(layout, tokenizer, pad_id, compressed_vocab_size)` builds `token_map` (`build_compressed_token_map`), `multipliers` (`compute_hash_multipliers`), `primes`, `offsets` buffers; `init_history(num_req_slots, device)` allocates the `[num_req_slots + 1, max_ngram_size - 1]` int32 successor-history with a spare pad row; `from_config(config, layout)` (line 248); `forward(input_ids, forward_batch)` (line 271); `_torch_hash_ids` (line 377, non-CUDA fallback); `commit_after_verify` (line 444).

**Call site: `python/sglang/srt/models/deepseek_v4.py` (dsv4.1), Pre-LayerNorm / "hc_pre" model body — not a `Qwen4ExpPLELayer` analogue.** **[C]**

| line | code |
|---|---|
| 104-108 | `from sglang.srt.layers.engram import (..., EngramHasher, build_engram_layout, ...)` |
| 3418 | `self.engram_layout = build_engram_layout(config)` |
| 3452-3453 | `if self.engram_layout is not None: self.engram_hasher = EngramHasher.from_config(config, self.engram_layout)` |
| 2261-2268 | Engram modules injected per layer: `if engram_layout is not None and layer_id in engram_layout.layer_ids: self.engram = Engram(...)` |
| 3573-3595 | invocation inside `_forward_layers_hc_pre_from_prev` |
| 3597-3606 | prefetch of `self.layers[14].engram` on `self.engram_prefetch_stream` |
| 3633-3647 | per-layer `hidden_states = engram(hidden_states, hash_ids[:, engram.layer_hash_index], forward_batch, ...)` |

The invocation dispatch, verbatim:

```python
        if self.engram_hasher is not None:
            if cp_extend:
                ... hash_ids = self.engram_hasher(forward_batch.input_ids[:total], forward_batch) ...
            elif (forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph()):
                hash_ids = bcg_deepseek_v4_engram_hash_ids(self.engram_hasher, input_ids)
            else:
                hash_ids = self.engram_hasher(input_ids, forward_batch)
```

with

```python
def deepseek_v4_engram_hash_ids(hasher, input_ids: torch.Tensor) -> torch.Tensor:
    # The hasher reads per-request rows: run it outside the prefill CUDA graph
    # on the live batch.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    return hasher(input_ids, forward_batch)

bcg_deepseek_v4_engram_hash_ids = eager_on_graph(True)(deepseek_v4_engram_hash_ids)   # line 735
```

**Is that gather CUDA-graph safe? Two different things must be separated.** **[C]**

- **The hasher is NOT graph-safe for extend/decode with changing request layouts.** It reads `self.history[req_slots]`, does `torch.repeat_interleave` on `forward_batch.extend_seq_lens`, and *writes* `self.history[commit_rows] = ...`. The decode path calls `engram_hash_ids_and_commit(...)` (a fused kernel) and explicitly asserts `forward_batch.out_cache_loc is not None` so "out_cache_loc 0 marks the CUDA-graph padded rows that must not commit". This is exactly why the extend path is wrapped in `eager_on_graph(True)`.
- **The embedding gather itself IS graph-safe.** `EngramEmbedding` (`engram.py:687`) is a pure-device table: `forward` (762) → `_lookup` (797) → `_owned_rows` (812) → the `engram_gather` Triton kernel, with `tensor_model_parallel_all_reduce` when TP > 1. No host sync, no Python I/O, no `.item()`. The host-memory variants (`_HostTable` at line 549, layouts `shared`/private, THP-backed) are still only *memory placement*, not host round-trips.

---

## 7. Bottom line — the smallest change for a working disk-backed PLE store

### vLLM

There are now **three** seams, because three designs are in flight. Pick by which path is live in your tree.

**(a) If you are on merged `main` (#54371, UVA pinned host) — override the pinned backend.** **[C]**

The class to subclass is `Qwen4ExpPLEPinnedHostEmbedding` (`vllm/models/qwen4_exp/nvidia/ngram_embedding.py:387`) and the two methods that must change are:

- `allocate_embedding_weight(num_embeddings, embedding_dim, dtype)` (`:430`) — today returns one `torch.empty(..., pin_memory=True)`. Replace with your store's row source (e.g. a small pinned staging slab, not the full table).
- `_lookup(input_ids, output=None)` (`:445`) — today launches `_lookup_ple_embedding_from_pinned_kernel` against `self._uva_weight`. Replace with a kernel/op that reads from your own row cache.
- **And** `__init__:418` (`self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)`) must stop being applied to the whole table.

**Graph break: NOT needed**, provided your replacement is itself device-side and idempotent under replay — `start_prefetch` → `_finalize_prefetch` → `forward` all run inside the captured graph in FULL mode (see Q1: `eager_break_during_capture` is a no-op in FULL and a no-op entirely without `VLLM_USE_BREAKABLE_CUDAGRAPH`). If your store's read is issued from Python/host, you **do** need the break, and the existing `@eager_break_during_capture` on `start_prefetch`/`_finalize_prefetch` already gives it to you for free in breakable + PIECEWISE mode. **[U] I could not confirm that a pageable file mapping can be made UVA-visible without `cudaHostRegister`** (see Q2) — if it cannot on your hardware, this seam forces the whole table to stay pinned and is the wrong seam for you.

**(b) If you are on the offload-worker path (#53899/#54070) — this is the cheapest seam, and #54070 is a working proof.** **[C]**

The exact function to add/replace is the weight-discovery step of `PleOffloadRunner.__init__` in **`vllm/v1/ple_offload/worker.py`**, i.e. swap in a `_ple_disk_attach`-style hook. Your store needs one method with the shape of `forward_impl`, and the layer to target is `Qwen4ExpNGramEmbedding` (the `PleOffloadLayer` subclass) plus `PleOffloadLayer.forward_impl` (`vllm/model_executor/layers/ple_offload_layer.py`). The GPU-side contract you must preserve is exactly three things: `_gpu_output_buffer` (IPC-mapped CUDA tensor), `_sem` (`CpuGpuSemaphore`), and signalling `DONE_VALUE` on the copy stream after the H2D.

**Graph break: NOT needed.** The wait is `torch.ops.vllm.ple_offload_wait`, a `direct_register_custom_op` with a fake impl, issued *inside* the captured graph; the batch metadata is D2H'd in `PleOffloadConnector.prepare_forward` before replay. This is the design that already pairs `cpu_offload=true` with FULL capture.

**(c) If you are on #54129's V2 input-prep path — override `gather_into`.** **[C]**
`vllm/models/qwen4_exp/nvidia/ple_mmap.py:976  MmapNgramEmbedding.gather_into(ids, destination)`. It is already called from `Qwen4ExpModelState.prepare_inputs` via `Qwen4ExpNGramEmbedding.prepare_mmap_rows(...)` (`ple_layer.py` diff), i.e. **before** the compiled/captured forward. Requires Model Runner V2 (`check_cudagraph_safety`, `ple_mmap.py:1179`). **Graph break: NOT needed** — "Captured model code only ever reads that buffer … so there is no host work, D2H, or Python custom-op body left inside the traced/captured graph" and "This makes every V2 cudagraph mode safe, including the FULL-containing ones."

**(d) The disjoint alternative [C], listed for completeness:** the disk *side* is already solved by #54070 — `_ple_disk_attach` / `_disk_backed_tensor` in `vllm/v1/ple_offload/worker.py` — and it needs **no** graph change. If your problem is purely placement (page cache vs anonymous RAM) and you accept the kernel's page-fault path, you do not need to touch any gather at all.

### SGLang

**Note first that `main` already has a file-backed PLE table, and you may not need to write anything.** **[C]**
`python/sglang/srt/models/qwen4_exp_ple_table.py`:

- `allocate_ple_host_table(shape, dtype, backend, table_dir, tag)` — `backend="pinned"` (default) → `torch.empty(..., pin_memory=True)`; `backend="file"` → sparse file + `torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)` + `_madvise_random`, deterministic filename from `ple_table_file_name(shape, dtype, tag)`.
- Selected by `config.ple_offload_backend` / `config.ple_offload_dir` in `Qwen4ExpPLELayer.__init__` (`python/sglang/srt/models/qwen4_exp.py:933-938`).
- `check_file_backend_supported()` **fails fast** unless the device reports `cudaDevAttrPageableMemoryAccessUsesHostPageTables` (attr 100) — i.e. GB10 / DGX Spark-class HMM. `SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1` overrides.
- `PleFilePrefetcher` (`posix_fadvise(WILLNEED)` over the distinct 4 KiB pages of a gather, skips decode-sized gathers below `PLE_FILE_PREFETCH_MIN_ROWS = 2048` and skips during `torch.cuda.is_current_stream_capturing()`).
- `PleFileRssTrimmer` (`MADV_DONTNEED` in 1 GiB chunks on a daemon thread, budget `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=8.0`, interval `SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S=30.0`).
- Env: `SGLANG_QWEN4_PLE_FILE_DIR`, `SGLANG_QWEN4_PLE_FILE_PREFETCH`, `SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK`, `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB`, `SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S`.

**If you must plug in your own store, the smallest change is to subclass `Qwen4ExpPinnedHostEmbedding` and override `gather`.** **[C]**
`python/sglang/srt/models/qwen4_exp.py:768  class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding)`; the gather is

```python
    def gather(self, input_ids, out=None):        # :861
        ...
            _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight, flat_ids, out, self.embedding_dim,
                self.vocab_start, self.vocab_end)
```

It is installed by a one-line swap at `Qwen4ExpPLELayer.__init__:933-938`, and the whole prefetch/side-stream/buffer machinery around it (`_get_prefetch_buffer:1129`, `start_prefetch:1145`, `_consume_prefetched_embeddings:1183`, `_prefetch_stream`) is already there and already capture-aware (`_graph_prefetch_buffers` keyed by `lookup_tokens` when `get_is_capture_mode()`). **Graph break: NOT needed** for a device-side gather. If your store's read is host-issued, mirror what #36567 does — that is, a new class next to `NVMePLEEmbedding` plus an `is_nvme_ple_embedding`-style branch at `qwen4_exp.py:933` — and wrap the read in `eager_on_graph(True, capture_stub=...)`, which **requires** `--cuda-graph-backend-prefill breakable` / `--cuda-graph-backend-decode breakable`.

**For DeepSeek-V4.1 Engram (`dsv4.1` branch only): subclass `EngramEmbedding` and override `_owned_rows` / `_lookup`** (`python/sglang/srt/layers/engram.py:812` and `:797`) — that is the gather (`engram_gather` Triton kernel). `_HostTable.choose_layout` (`:604`) already picks between `shared` and per-rank private host mappings. **Graph break: NOT needed** for the gather; the *hasher* already carries its own `eager_on_graph(True)` wrapper at `deepseek_v4.py:735` and you should leave that alone.

---

## Appendix — everything I could NOT confirm

1. **[U]** Whether `cudaHostGetDevicePointer` can succeed on an unregistered, pageable file mapping on HMM/ATS hardware (GB10-class). If yes, vLLM's merged UVA seam (7a) could serve from disk without pinning the table; if no, it cannot. SGLang's `device_uses_host_page_tables` check exists precisely because that property is platform-dependent, but it is a different API (`cudaDeviceGetAttribute`, not `cudaHostGetDevicePointer`), so it does not settle the question.
2. **[U]** Which PR `#54129`'s body means by "without the previous whole-forward splitting op" — it never names one, and #53899's diff comments reference "the splitting op" (`torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids` writing into graph-owned storage) in the *resident* path, not in the offload path.
3. **[U]** The stale-sounding `VLLM_PLE_CPU_OFFLOAD` docstring "supports ModelRunner V1 … only" (see Q4) versus the fact that both #53899 and #54070 patch the **V2** runner file.
4. **[U]** Whether #54070 itself was captured end-to-end at `cudagraph_mode=FULL_AND_PIECEWISE`. #54070's stated tests are first-boot / reboot / gather-equivalence on hardware; the FULL-capture test plan lives in the #53899 commit it includes.
5. **[U]** Whether #36567 has a non-eager CUDA-graph validation run at all. Its own Speed Tests section says "eager execution"; the code is written for the breakable backend (`eager_on_graph`, `is_in_breakable_cuda_graph`) but I found no `FULL_AND_PIECEWISE`/breakable test in its file list (`.github/workflows/pr-test-rust-exts.yml` is the only CI change).
6. **[U]** `sgl-project/sglang#39205`'s Engram work (Mooncake backend) is on the `dsv4.1` base and is **not** merged; I read only its body and file list, not its diff, so its `engram_mooncake.py` internals are unverified beyond the body's claims.
