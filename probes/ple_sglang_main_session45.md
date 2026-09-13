# SGLang-main PLE offload: file-backed table on a non-HMM GPU (Session 45)

**Target tag** `sgl-project/sglang@14b647cf27d7f2c1a3764841f7d3770ff9f9e7d6` (main, 2026-09-13).
Everything under test — `Qwen4ExpNGramEmbedding`, `Qwen4ExpPinnedHostEmbedding`,
`Qwen4ExpStagedFileEmbedding`, `BreakableCUDAGraph`, `eager_on_graph` — is
imported from that tree. Hardware: RTX 4090 24 GB, 1 TB RAM, 124 GB `/root/autodl-tmp`.

**Deliverable** `integrations/sglang-main/0001-ple-offload-file-staged.patch`
(+ `README.md` in the same directory). Probe: `scripts/sc_main_staged_probe.py`,
driver `scripts/sc_main_staged_sweep.sh`, raw JSON in `probes/data/`.

---

## 1. What was built, and why it is the *target* seam

`main` already ships a file-backed PLE table; the storage layer
(`allocate_ple_host_table`, sparse file, deterministic name, `PleFilePrefetcher`,
`PleFileRssTrimmer`) is complete and untouched by the patch. What main has is
exactly **one** dereference implementation: a Triton kernel that reads the
mapping through the device's address space, gated at load time by
`check_file_backend_supported` on `cudaDevAttrPageableMemoryAccessUsesHostPageTables`.

The patch splits the decision in two — **where the table lives** and **which
side dereferences it** — and adds `--ple-offload-backend file-staged`: the same
sparse file, read by the host into a pinned staging slab, published to the
device by one pinned→device copy. `gather` is wrapped in
`eager_on_graph(True, capture_stub=...)`, which is main's own break mechanism.

This closes the debt recorded as **V187** ("SC4 apparatus was built on a seam
upstream does not use"). The measurement below is on `Qwen4ExpPLELayer`'s own
gather call site, and the patch is a diff against a real tree.

### The break is a precondition, not an optimisation

`Qwen4ExpPLELayer.start_prefetch` (`qwen4_exp.py:1145`) is called at
`qwen4_exp.py:1688`, at the top of iteration *i*, for the layer *i+1* PLE layer,
so the lead time is exactly one decoder layer. But the ids are produced on the
device, so the read cannot start until they have been copied back — and a
device→host copy is not capturable. With the decorator disarmed, capture dies:

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned.
```

So `file-staged` requires `--cuda-graph-backend-{decode,prefill} breakable`.

### The upstream gate prevents a real crash

Running upstream's `file` backend on this non-HMM device with the load-time
check bypassed (the `direct` arm), and gathering through it:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
  (cudaErrorIllegalAddress)
```

Full log: `probes/data/direct_arm_crash.txt`. `check_file_backend_supported` is
load-time fail-fast against a hard fault, not defensive caution — and this is
the first-hand evidence for that, which the patch's error message now names the
`file-staged` alternative.

---

## 2. Correctness

The table the probe builds is the shipped model's geometry, scaled in row count
only: **16 rows per token** (`ngram_size=3` × `heads_per_ngram=8`),
**160 B per row fp8**, `ple_embed_dim=2560`. 256 MiB = 1,679,360 rows;
**4096/4096 distinct rows** in the first 4096, `zero_frac` 0.0089 — real data,
not a sparse placeholder.

| check | result |
|---|---|
| eager `staged` gather vs an independent second mapping of the same file | **equal** (exact bf16 bytes) |
| graph replay produces the *current* step's rows, all 16 cells × 20 replays | **20/20 every cell** |
| `staged-nobreak` (decorator disarmed) | fails at capture, see above |

Freshness is checked by rewriting the id buffer between replays and comparing
the graph's consumed output against the oracle for *that* iteration, so a stale
replay would be caught: the `pinned` arm scores 20/20 as well, which is what
makes the check meaningful.

---

## 3. Cost: replay step time (µs, median of 20, RTX 4090)

Table 256 MiB fp8. `cold` = `posix_fadvise(DONTNEED)` + `madvise(DONTNEED)` on
the mapping immediately before the replay.

| tokens | rows | pinned warm | pinned cold | staged warm | staged cold | staged − pinned (warm) |
|---|---|---|---|---|---|---|
| 1 | 16 | 30.1 | 50.2 | **181.4** | **2297.7** | +151.3 |
| 8 | 128 | 67.4 | 63.8 | **378.4** | **13043.8** | +311.0 |
| 32 | 512 | 48.1 | 72.9 | **498.1** | **18990.5** | +450.0 |
| 128 | 2048 | 51.7 | 78.7 | **1494.1** | **6515.5** | +1442.4 |

`pinned` is flat in batch size and in coldness (it is RAM); it is the device-side
floor. Graph shape is identical across cells: `pinned` = 1 segment, 0 breaks;
`staged` = 2 segments, 1 break.

## 4. Where the staged time goes

Eager, per phase, prefetch hint **off** (so the decomposition closes):

| tokens | d2h ids | host read | h2d publish | sum | measured step |
|---|---|---|---|---|---|
| 1 | 20.5 | 87.1 | 16.9 | 124.5 | 123.3 ✓ |
| 128 | 28.1 | 513.1 | 46.3 | 587.5 | 590.5 ✓ |

Cooling the table moves only the middle column — 87.1 → **3335.3** µs at 16
rows, 513.1 → **17232.9** µs at 2048 rows (prefetch on) / 16136.8 (off). The
round trip itself (d2h + h2d, no storage at all) is **~37 µs**.

### The prefetch hint is a net loss at the sizes where it fires

`PLE_FILE_PREFETCH_MIN_ROWS = 2048`, and 128 tokens × 16 heads is exactly 2048,
so this is the regime it was written for. Same cell, hint on vs off:

| | hint on | hint off |
|---|---|---|
| step warm | 1421.1 | **590.5** |
| step cold | 18221.8 | **17161.7** |

It costs ~830 µs warm (2048 `posix_fadvise` syscalls on a background thread
competing for the GIL) and buys nothing cold. The reason is the same one that
limits everything else here: **the hint is issued microseconds before the read
it is meant to accelerate, so it has no lead time.** This is the same finding as
the kernel criterion's second clause — `issue_time ≤ consume_time − τ` — and it
is now measured on the target seam rather than inferred.

## 5. Verdict against τ(1)

τ(1) ≈ **325 µs** (Session 44, delay sweep; stand-in — see §6).

| case | step | vs τ(1) |
|---|---|---|
| staged warm, 1 token | 181 µs | **fits**, 0.56× |
| staged warm, 8 tokens | 378 µs | 1.16× |
| staged warm, 128 tokens | 1494 µs | 4.6× |
| staged cache-dropped, 1 token | 2298 µs | 7.1× |
| staged cache-dropped, 128 tokens | 6516 µs | 20× |

**The read is not hidden.** A break whose first act is a device→host copy cannot
start before the segment in front of it has drained, so "one decoder layer of
lead" buys nothing; the exposure is the full `d2h + read + h2d`.

What that means for the north star is not that the approach fails, but that it
has a **measured service ceiling**: at ~2.3 ms/step a single-sequence decode runs
at ≈430 tok/s and at 6.5 ms/step a 128-sequence batch at ≈20k tok/s aggregate.
"Serviceable when no memory tier fits the table" is satisfied; "indistinguishable
from a resident table" is not, and the gap is entirely lead time plus the
serialized fault cost of the host read.

## 6. Stand-ins and boundaries (discipline 第十条)

| what | target? |
|---|---|
| the seam (`Qwen4ExpPLELayer`, `Qwen4Exp*Embedding`, `BreakableCUDAGraph`, `eager_on_graph`) | **target**: main @14b647c |
| the patch | **target**: applies clean to a pristine 14b647c tree |
| dependency set | **stand-in**: sglang 0.5.19's site-packages, reached by `PYTHONPATH`. Nothing in the tested path is a kernel; `torch` is 2.13.0, the pin main declares |
| model body | **not instantiated**. `hidden_size=256`, 4 layers; only `Qwen4ExpNGramEmbedding` + the swapped embedding exist |
| table size | **stand-in**: 256 MiB, not 51.2 GB. Row *geometry* is the shipped one |
| the "preceding decoder layer" | **stand-in**: one 512×512 matmul, not a real Qwen4 layer |
| τ(1) | **stand-in**: Session 44, Qwen3.5-0.8B 24 layers |
| "cold" | **partially stand-in**: `fadvise`+`madvise` eviction on a box whose page cache is 796 GB. The min/median spread (0.5–88 ms) shows eviction is incomplete; the real cold number is therefore *worse* than reported, and the cold column is a lower bound |
| 4090 | **target device class** for the non-HMM branch |

Also noted: the `pinned --cold` cells ran while `table_path_of` could still
resolve a stale same-shaped file, so they cache-dropped against an unrelated
file. `madvise(MADV_DONTNEED)` on `pin_memory()` returns 0 without disturbing
the data (verified directly, and corroborated by 20/20 freshness), so those
cells are unaffected; the probe now returns no path for non-file backends.

## 7. What the numbers say to do next

1. **Get the ids on the host before the step.** The whole exposure is caused by
   the ids being device-side. The scheduler already has the token history on the
   host; hashing there (`_hash_contexts` is deterministic integer arithmetic
   over `layer_multipliers` / `ngram_heads_vocab_sizes` / `ngram_heads_offsets`,
   all of which are checkpoint constants) would let the read be *issued* a step
   ahead instead of merely *started* at consume time.
2. **Then** make the read fast and concurrent — `O_DIRECT` + `io_uring` at depth
   (SGLang #36567's reader), which is also what turns the 208 µs/row serialized
   cold fault into something closer to the device floor.
3. **Delete the hint, or give it lead time.** As written, `posix_fadvise(WILLNEED)`
   at consume time is a measurable loss; either drop it for `file-staged` or
   move it to the point where the ids are known early — item 1.
4. The `file` gate stays. `file-staged` should not silently replace it: the
   access mode determines whether a breakable graph is required, and that is a
   property of the deployment, not something to infer.
