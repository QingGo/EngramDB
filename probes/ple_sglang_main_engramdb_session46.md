# EngramDB inside SGLang `main`'s real PLE seam, on the real table (Session 46)

**Target tag** `sgl-project/sglang@14b647cf27d7f2c1a3764841f7d3770ff9f9e7d6` (main).
Everything under test is imported from that tree; the row source is a real
EngramDB store. Hardware: RTX 4090 24 GB, 1 TB RAM.

**Deliverables**
- `integrations/sglang-main/0002-ple-offload-engramdb-store.patch` — the `engramdb`
  backend: SGLang reads its PLE rows out of an EngramDB store directory.
- `scripts/sc_main_engramdb_probe.py` (+ `sc_main_engramdb_sweep.sh`),
  `scripts/engramdb_store_api_probe.py`
- raw JSON in `probes/data/main_engramdb/`

---

## 1. What is no longer a stand-in

Session 45 established the seam with a 256 MiB scaled-down table and a synthetic
config. This session removes all of that:

| | Session 45 | here |
|---|---|---|
| PLE config | hand-picked small values | **the checkpoint's own `text_config`** |
| vocabulary | 1,679,360 rows | **320,001,536 rows** — the engine's `padded_vocab` exactly |
| table | 256 MiB synthetic | **51,200,245,760 B**, 128 shards × 2,500,012 rows × 160 B |
| row source | upstream sparse-file mmap | **`engramdb.Store`** |
| rowids | random | **the engine's own `_hash_contexts`** |
| seam | `Qwen4ExpPLELayer` → `gather` → `BreakableCUDAGraph` | unchanged (target) |

Still stand-in, and tagged as such: the model body is never instantiated (the
module is built on the **meta** device), the "preceding decoder layer" is one
512×512 matmul, and τ(1) ≈ 325 µs is Session 44's, not this session's.

One deliberate deviation, recorded because it changes a config path: the
checkpoint reaches fp8 through its `quantization_config`, and building the real
quant config pulls in the model-loading stack for nothing here, so the probe
declares `ple_embedding_dtype="float8_e4m3fn"` instead. `_ple_table_is_fp8`
accepts either, and the resulting row width is **asserted against the store**
(`row_bytes_equals_store_width: true`).

## 2. Correctness, three independent ways

| check | result |
|---|---|
| engine `_hash_contexts` vs `engramdb.rowids_for_seq`, same tokens, real config | **equal**, 0 mismatches / 512×16 ids, `max_abs_diff = 0` |
| `Store.fetch` vs an independent `os.pread` of the same `shard_NNN.bin` | **byte-equal** (512 rows, 512 distinct, `zero_frac` 0.0) |
| every graph replay's output vs that same pread oracle, at 4 batch sizes | **5/5 every cell**, `mismatch_elems: []` |

The second is the one that matters for "is this *our* data": the oracle opens the
raw shard files itself and computes `shard = rowid // 2,500,012`,
`offset = (rowid % 2,500,012) × 160` — no engramdb code in the loop.

The engine's per-layer multipliers come out as
`[23703573157769, 20109073645365, 8052911324071]`, which is **literally the
constant set our native rowid path hardcodes** — the alignment is not a
coincidence we arranged, it is the checkpoint's.

The geometry is asserted, not assumed: engine `padded_vocab = 320,001,536` ==
store `total_rows`; `head_dim_per_ngram` 160 == store `width`; table dtype fp8 so
row bytes 160 == store width.

## 3. It runs in the graph, through the tree's own selection path

`--real-layer` builds `Qwen4ExpPLELayer` and lets `Qwen4ExpPLELayer.__init__`
pick the backend from the config; the probe asserts the result is a
`Qwen4ExpEngramDbEmbedding`. Graph shape is 2 segments / 1 break, i.e. the eager
`gather` really did split the capture.

## 4. Cost (µs, median of 25 replays, RTX 4090, real 51.2 GB table)

| tokens | rows | warm | warm min | cold | cold min |
|---|---|---|---|---|---|
| 1 | 16 | **196.5** | 178.7 | 406.5 | 304.8 |
| 8 | 128 | **263.9** | 249.2 | 1247.5 | 1091.8 |
| 32 | 512 | **560.3** | 529.0 | 3536.0 | 2985.9 |
| 128 | 2048 | **1094.9** | 1008.5 | 10603.8 | 9418.2 |

`cold` drops exactly the 4 KiB pages this step will read
(`posix_fadvise(DONTNEED)` per row) — dropping the whole 51.2 GB table first
leaves the kernel walking it while the read starts, and that contention, not the
storage, then dominates (that mistake produced a bogus 42 ms median earlier).

Versus Session 45's upstream-mmap numbers on the same seam and the same graph
shape (256 MiB table):

| tokens | mmap warm | store warm | mmap cache-dropped | store cold |
|---|---|---|---|---|
| 1 | 181.4 | 196.5 | 2297.7 | **406.5** |
| 8 | 378.4 | **263.9** | 13043.8 | **1247.5** |
| 32 | 498.1 | 560.3 | 18990.5 | **3536.0** |
| 128 | 1494.1 | **1094.9** | 6515.5 | 10603.8 |

Two caveats before reading that table as "we are faster": the mmap's cold
control was *partial* (a whole-mapping `madvise`+`fadvise` leaves pages behind)
while this one is surgical, so the cold columns are not the same experiment; and
the warm columns are within a factor of ~1.3 either way. What is not in doubt is
that a 51.2 GB store served this way is in the same cost class as a 256 MiB
mapping — which is the point, since the mapping is the thing that cannot exist
at 51.2 GB on a 36 GB-free disk.

Against τ(1) ≈ 325 µs: **warm 1 and 8 tokens fit** (196.5 / 263.9); everything
cold, and everything ≥32 tokens, does not.

## 5. Where the time goes, and the one optimization that worked

Attribution at 2048 rows, measured in place (same process, same buffers):

```
d2h_ids        2.0
read_rows    792.5      <- of which Store.fetch ~590-670
publish       38.6
gather_all   919.0      ... eager, no graph
graph step  1094.9      ... so the graph machinery costs ~36
```

`cProfile` was tried first and is **unusable here**: it reported 4.3 ms per
`Store.fetch` where wall clock says 0.59 ms, a 7× inflation, because the store's
worker threads and the tracer fight over the GIL. Noted so the next person does
not trust it.

### The publish was the surprise

| variant | publish at 2048 rows |
|---|---|
| one cross-device copy that also casts (fp8 slab → bf16 CUDA) | **456.5** |
| bf16 slab, same-dtype copy (cast done on the CPU first: 272.3) | 41.6 (+272.3) |
| **two same-dtype copies, cast on the device** | **38.6** |

PyTorch's `copy_` across devices *and* dtypes costs 12× what the same-dtype copy
does, and 25× what the cast itself costs on the GPU (18 µs flat). The fix is to
keep the staging slab in the table's dtype and split the publish:

```
pinned fp8 --(plain H2D)--> device fp8 --(device cast + copy)--> output bf16
```

End-to-end, `--staging` A/B on the probe class, warm:

| tokens | legacy publish | device cast | ratio |
|---|---|---|---|
| 1 | 183.9 | 196.5 | 1.07× |
| 8 | 455.0 | **263.9** | **0.58×** |
| 32 | 781.4 | **560.3** | **0.72×** |
| 128 | 1613.2 | **1094.9** | **0.68×** |

The tree's own class reproduces it (`real_1` 205.3, `real_128` 1145.1).

### Two things that did *not* work

- **Naive fp8 staging** — moving the cast off the CPU while leaving it inside the
  cross-device copy is a *pessimization*: 1468 µs vs 1265 at 2048 rows.
- **Thread count** — `Store` thread sweep at 16 / 128 / 2048 rows: 4–8 threads
  beats 32 by ~4 µs at 16 rows; 32 wins by 8% at 2048. Not a lever worth pulling.
- Also checked and rejected: `Store.fetch` accepts a numpy array (no faster than a
  list) but rejects a `torch.Tensor`
  (`TypeError: 'Tensor' object cannot be converted to 'Sequence'`); `.tolist()` on
  2048 ids is 34 µs, not the bottleneck it looked like.

## 6. What is left

The step is still **exposed**: the whole 919 µs at 2048 rows sits on the critical
path because the break's first act is a device→host copy. Nothing in this session
changed that, and no reader improvement will. The next move is unchanged from
Session 45 and now has its feasibility proof: **produce the ids on the host**, so
the read can be *issued* a step early instead of merely started at consume time.

Cheaper follow-ups, in order: (a) `Store.fetch` at 16 rows costs ~94 µs, almost
all fixed overhead, and a decode step is 16 rows — worth its own look;
(b) a `fetch_into(rowids, buffer)` that writes into a caller-owned pinned
`bytearray` would remove the `bytes` allocation and the `bytearray` copy that
`torch.frombuffer` forces on read-only input.
