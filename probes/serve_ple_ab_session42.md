# Session 42 · 真 serving A/B：vLLM 0.29.0 + Qwen3.5-0.8B + 真实 PLE 表

> 口径：**真引擎、真模型、真表、真 GPU**。所有数字来自一台机器一次运行，可复跑。
> 产物：`scripts/serve_ple_ab.py`、本文件、`docs/prefetch-lead-time.md`。

## 1. 环境【实测】

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GiB, driver 595.71.05, CUDA 13.2 |
| 引擎 | **vLLM 0.29.0**（offline `LLM` API：真调度器、真 KV cache、真 continuous batching） |
| torch / transformers | 2.13.0+cu130 / 5.17.0 |
| 模型 | `Qwen3.5-0.8B`（`Qwen3_5ForConditionalGeneration`，24 层，hidden 1024，混合 linear/full attention） |
| 表 | `qwen38-rows/` 65 shard × 400,001,920 B = **26 GB**，row width 160 B（Qwen PLE 几何） |
| rowid | **EngramDB 自己的 `rowids_for_seq(..., PLE_QWEN_V1)`**，16 heads/token |
| 注入点 | 第 2 层（`ple_layer_ids=[2]`），每 token 真取 16 行 = **2,560 B** |
| 主机 | 128 线程，1 TB RAM，XFS on RAID1 NVMe |

**必须声明的边界**：注入的是**随机投影**，输出是乱码（见 §5）。本实验只测**存储代价**，
不测质量。这不是一个训练过的 PLE 模型。

## 2. 结果【实测】batch=1, prompt 128, max_tokens 128, 3 次迭代

| 臂 | tok/s 中位 | 最好 | reader µs/次 | vs 首个 none | 增加 µs/token |
|---|---|---|---|---|---|
| `none`（无 reader） | 45.0 | 45.2 | — | 基准 | — |
| **`engram-i`（磁盘 Store-I）** | **43.9** | 44.3 | **913** | **−2.4%** | **+552** |
| `shm`（/dev/shm 同字节） | 45.7 | 45.8 | 116 | +1.6% | −359 |
| `mmap`（numpy memmap） | 46.0 | 46.4 | 338 | +2.2% | −485 |
| `prefetch-ub`（后台线程预取） | 40.4 | 40.4 | **2525** | **−10.2%** | +2522 |
| `none`（重复，测漂移） | 46.1 | 46.9 | — | **+2.5%** | −552 |

**噪声地板 = 2.5%**（首尾两次 `none` 的差）。

### 读数

> **⚠️ 本节已更正。** 初版把 −2.4% 归因给「磁盘读暴露在关键路径」。后续用
> `shm-keygen` 臂（真实 keygen + /dev/shm）把两者分开后，**该归因被实测否定**。
> 更正见 §2.1。

1. 磁盘臂的代价（−2.4%）与噪声地板（+2.5%）同量级 ⇒ 在 batch=1、eager 下
   **无法断言可分辨的性能回归**。
2. `shm` 与 `mmap` 都落在基准附近 ⇒ 116–338 µs 量级的取数可被掩盖。
3. `prefetch-ub` 比同步更慢 2.7× —— 见 §3。

## 2.1 更正：代价来自 **keygen**，不是磁盘【实测】

`shm-keygen` = 真实 EngramDB keygen + `/dev/shm` 取数。与 `engram-i`（同 keygen + 磁盘）
配对即可把 keygen 与介质分开。同一台机器、同参数、单进程六臂：

| 臂 | tok/s 中位 | vs none | µs/次 | 其中 keygen |
|---|---|---|---|---|
| `none` | 45.0 / 45.5（首尾） | ±0.6% | — | — |
| `shm`（固定 RNG + RAM） | 43.7 | −3.3% | 118 | 36 |
| **`shm-keygen`（真实 keygen + RAM）** | **43.0** | **−5.0%** | **900** | **812** |
| **`engram-i`（真实 keygen + NVMe）** | **43.4** | **−4.1%** | **921** | — |
| `mmap`（固定 RNG + NVMe） | 44.4 | −1.8% | 160 | — |

**两个结论：**

1. **`shm-keygen`(RAM) 43.0 ≈ `engram-i`(NVMe) 43.4** ⇒ **把介质从 RAM 换成 NVMe
   在噪声内无差别。磁盘不是代价。**
2. `shm-keygen` 的 900 µs 里 **812 µs 是 rowid 生成（90%）**。

独立分解（`scripts/ple_reader_decompose.py`，300 次，非 serving 场景）给出同一结论：

| 阶段 | 中位 | 占比 |
|---|---|---|
| **1 rowid 生成（PyO3）** | **940.7 µs** | **83.6%** |
| 2 rowid 折行（Python） | 2.2 µs | 0.2% |
| **3 `Store.fetch`（真 I/O）** | **175.7 µs** | **15.6%** |
| 4 张量构造（Python） | 6.1 µs | 0.5% |

⇒ 这是仓库自己记录的技术债 **V157**（roadmap §29.4 Phase 1）的独立复现：
「rowid 生成 / history / fetch / FP8 反量化全部下沉 PyO3，目标 500–625 → ≤50 µs/token」。

**因此要先做 V157，再谈掩盖。** 当前的 900 µs 里只有 176 µs 是 I/O；把 812 µs 的
keygen 留在 Python/PyO3 边界上，任何预取策略都无从下手。

## 2.2 根因：`PleSpec::real()` 每次调用都被重建【实测】

keygen 那 ~900 µs 与 token 数**无关**：

| 调用 | 中位 | tokens/call 扫描 |
|---|---|---|
| 1 | 939.1 µs | |
| 8 | 777.0 µs | |
| 64 | 915.1 µs | |
| 512 | 1601.4 µs | |

**空输入决定性验证**（`engramdb-python 0.3.0`，本机 venv）：

```
rowids_for_seq([])   = 941.4 us   <-- 空列表
rowids_for_seq([1])  = 942.4 us
rowids_for_seq(x64)  = 799.8 us
```

⇒ **100% 的固定开销来自 `crates/engramdb-pyo3/src/lib.rs:425`**：

```rust
fn rowids_for_seq(tokens: Vec<u32>, ple_spec: u32) -> PyResult<Vec<Vec<u32>>> {
    ...
    let spec = PleSpec::real();   // ← 每次调用重建
```

而 `PleSpec::real()`（`crates/engramdb-keygen/src/lib.rs:47-50`）对 `i in 0..16` 调用
`nth_prime_after(19_999_999, i + 1)` —— **每次调用做 16 次 2000 万量级的素数搜索**。
同样的模式出现在 `rowids_for_seq_with_history`。

**影响**：batch=1 时每步付 ~900 µs 纯浪费；128 token 的生成里累计 ~345 ms。
这解释了 §2.1 观察到的 −4~5%（部分与引擎重叠）。

**修法（未实施）**：把 spec 提到 `OnceLock`/`LazyLock`（或 `static`）里构造一次。
预期 `engram-i` 的每次调用从 ~920 µs 降到 ~85–180 µs（即真实 I/O），
届时 §2.1 的可分辨回归应当消失。

**注意**：这不是新债，是 **v0.3.0 已发布代码里的缺陷** —— 任何调用
`engramdb.rowids_for_seq` 的用户都在付这笔钱。它属于 V157 的处置范围，
但比 V157 描述的「下沉 Rust」更简单：**一行缓存即可**。

## 2.3 修复与验证【实测】

**改动**（`crates/engramdb-keygen/src/lib.rs` 新增 `real_spec()`，
`engramdb-pyo3` 两处、`engramdb-cabi` 一处改用之）：

```rust
pub fn real_spec() -> &'static PleSpec {
    static SPEC: std::sync::OnceLock<PleSpec> = std::sync::OnceLock::new();
    SPEC.get_or_init(PleSpec::real)
}
```

**逐位一致性**（修复前后同一组 32 token 的 rowid 全量对比）：`identical: True`。
`cargo test --workspace` 全绿（含新增 `real_spec_is_shared_and_matches_real`，
以及原有的 `matches_python_golden`）。`cargo fmt --check` / `clippy -D warnings` 干净。

### 单调用效果（远端 4090 主机，venv 内实测）

| 调用 | 修复前 | 修复后 | 加速 |
|---|---|---|---|
| `rowids_for_seq([])` | 758.9 µs | **0.9 µs** | **843×** |
| `rowids_for_seq([1])` | 759.7 µs | **1.8 µs** | **422×** |
| `rowids_for_seq(x64)` | 793.0 µs | **41.7 µs** | **19×** |

### serving 效果（同参数，`--iterations 3`）

| 臂 | it0 | it1 | it2 | µs/次 |
|---|---|---|---|---|
| `none` | 44.1 | 45.2 | 45.3 | — |
| `engram-i` | 37.5 | 42.8 | **45.7** | 185 |
| `shm-keygen` | 45.8 | 45.9 | **45.9** | **103**（原 900） |
| `mmap` | 36.3 | 45.8 | **46.2** | 1456 |
| `none`（重复） | 46.7 | 47.1 | 47.7 | — |

**结论：batch=1 的可分辨回归消失。** 第 3 次迭代 `engram-i` 45.7 ≈ `shm-keygen` 45.9
≈ `none` 45.2。`engram-i` 与 `mmap` 的 it0 低（37.5 / 36.3）是**冷启动**（页缓存），
两者同量级，与 keygen 无关 —— 若只看中位数会把这个冷启动误读成磁盘代价。

### 影响范围（重要边界，别扩大）

**受影响**：`engramdb.rowids_for_seq` / `rowids_for_seq_with_history` 的直接调用方，
以及 **C ABI** 的 rowid 入口 —— 两者都在每次调用时重建 spec。

**不受影响**：**`PleMemory` / `PleMemoryAdapter`**。它们走
`ple_memory.py:251,261 -> _ple_rowids -> ple_math`（纯 Python 参考实现），
素数表在 `PleMemory.__init__` 里**只算一次**（`self.prime_sizes` / `self.offsets`）。

⇒ 即 **Rust 「快路径」比它要取代的纯 Python 路径慢约 400×**（941 µs vs 数 µs），
因为 Python 侧在构造时缓存、Rust 侧每次重算。

实测确认（同一 sparse 完整 128 分片地址空间，前后各一次）：

| tokens/次 | 修复前 | 修复后 |
|---|---|---|
| 1 | 198.3 µs/token | 180.8 |
| 4 | 69.7 | 63.2 |
| 64 | 18.1 | 19.9 |
| 1000 | 11.6 | 11.2 |

**基本无变化 ⇒ 修复确实不经过 adapter，符合预期，不是零结果。**

### 对 V157 的影响：它的表述可能是误诊

V157 说 `PleMemoryAdapter` 真表热路径 500–625 µs/token，处置方向是
「rowid/history/fetch/dequant 下沉 Rust/PyO3」。但本次实测：

- adapter 的**计算路径**（keygen + 转换）在**介质免费**（sparse 空洞）时为
  **11.6–198 µs/token**（batch 1/4/64/1000），全部远低于 500–625。
- 既然介质是免费的还这么快，V157 的 500–625 只能是**被 I/O 主导**。
- 而 rowid 在 adapter 里**每次调用只算一次**，不是瓶颈。

⇒ **V157 提出的处置（把 rowid 下沉）对一个不是瓶颈的环节**；
真正的瓶颈是 fetch/介质，而 Store-P 折叠（7.69 µs/token，roadmap §32.3）已经是它的答案。
**建议重写 V157**，而不是继续按原方向投入。

**仍未做**：真表（非 sparse）的 adapter 绝对值 —— 本机只有 65/128 分片。
冷态 `fadvise`、batch ≥8 的 serving 规模化。




## 3. 核心发现：Python 级预取**不能**掩盖 I/O

`prefetch-ub` 是「把取数丢到后台线程、下一步再消费」的上界探针（值是陈旧的，只测计时）。
预期：取数躲进 token 之间的 22 ms 空隙 ⇒ 代价归零。

**实测：代价从 913 µs 涨到 2525 µs，端到端 −10.2%，比同步版更差。**

原因：这条取数路径是 **GIL 受限**的。后台线程做的是
rowid 生成（PyO3 调用 + Python 列表推导 + 取模）+ `torch.frombuffer` + reshape，
这些**都要拿 GIL**。与主线程的 Python 工作重叠时，它不是「并行」，而是**争用**，
再叠加线程调度开销。

> 注意：`Store.fetch` 本身**确实**释放 GIL（`crates/engramdb-pyo3/src/lib.rs:61`
> 的 `py.allow_threads`）。膨胀来自 **Python 侧的每次调用开销**，不是 Rust 的 I/O。

### 结论（可执行）

**能掩盖的前提是取数路径 GIL-free 或不经过 Python。** 具体三条：

1. **rowid 生成 + 取数 + 张量构造必须整体下沉到 PyO3 的一次调用**（一次调用内
   `allow_threads` 覆盖全程），而不是现在这样「PyO3 取数 + Python 组装」。
2. 或者：**预取在采样点发起**（引擎侧），此时主线程正阻塞在 GPU/调度上，才有真空隙。
   官方 V4.1 设计正是「stage 处理 micro-batch **之前**对整批发起预取」。
3. 反过来说，**在层的 forward hook 里做取数，lead time 只有该层之前的计算**
   —— 对 0.8B/4090 是微秒级，等于没有。我在 `docs/prefetch-lead-time.md` §6 的
   **预测 P1（引擎开销 O 会掩盖取数）被本轮实测否证**：O 在 forward **之外**，
   层内的 τ 只有 `(L/N)·C`。

## 4. 复跑

```bash
ssh -p 28326 root@connect.nmb1.seetacloud.com
cd /root/engram-serve
PATH=/root/engram-serve/venv/bin:$PATH ./venv/bin/python serve_ple_ab.py \
    --arms none,engram-i,prefetch-ub,mmap,shm,none \
    --iterations 3 --prompt-len 128 --max-tokens 128 \
    --shm-budget-gb 20 --json-out ab-warm.json
```

## 5. 反面证据（必须一起引用）

- **输出是乱码**：`sample_output` = `'e,\t}\t}\t}…'`。注入随机投影必然如此。
  本实验**不构成任何质量声明**。
- `enforce_eager=True`：Python reader 进不了 CUDA graph，四个臂同罚，A/B 公平，
  但**绝对 tok/s 不是 CUDA-graph 数字**。
- 表 26 GB、RAM 1 TB ⇒ 页缓存会吞下整张表，"冷"必须靠 `fadvise(DONTNEED)` 制造。
  本轮为**温态**运行；冷态未做。
- 只有 65/128 个 shard 在本地，rowid 对可用行数取了模。I/O 形状保持，表身份不完整。
- 三条 `diag` 计数（`stash_embed=384`、`layer_hits=384`）是本轮**唯一**证明 hook 真的
  触发过的东西。上一轮 `inject=0` 时四个臂跑的是同一份代码，产出了一份「零开销」的
  漂亮假结果 —— 没有这三个计数就无法发现。

## 6. 未做（不要当成已做）

- **冷态**（`--cold`）与冷/温自校验。
- **batch 8/32/512** 的规模化（`docs/prefetch-lead-time.md` §4.3 的「批量为敌」预测待验）。
- **Store-P** 臂（需先建视图；本机 26 GB 表 + 可用磁盘 28 GB，未做）。
- **CUDA graph 路径**（需要把 reader 做成 splitting op）。
- 多轮 counterbalance 到噪声地板以下（当前只有首尾 `none` 两次）。

## 8. batch=32 规模化：§4.3「批量为敌」被部分否证【实测】

同参数（prompt 64，max_tokens 128，3 次迭代），batch 1 -> 32：

| 臂 | it0 | it1 | it2 | 中位 | µs/次 |
|---|---|---|---|---|---|
| `none` | 1122.8 | 1220.9 | 1175.7 | 1175.7 | — |
| `engram-i` | 1129.4 | 1120.4 | 1189.1 | 1129.4 | 565–646 |
| `shm-keygen` | 1230.4 | 1203.9 | 1194.8 | 1203.9 | 860–899 |
| `none`（重复） | 1248.1 | 1257.9 | 1238.6 | 1248.1 | — |

**四臂在噪声内一致（1120–1258）⇒ batch=32 下存储路径依然测不出代价。**

机制（这才是预测错在哪）：

| | batch=1 | batch=32 | 倍数 |
|---|---|---|---|
| 每 step 产出 | 1 token | 32 token | 32× |
| step 时间 | 22.6 ms | 26.5 ms | **1.17×** |
| reader µs/次 | ~185 | ~600 | **3.2×** |
| `F/τ` | 0.8% | 2.3% | — |

⇒ `F` 随 batch **次线性**增长（3.2× 而非 32×），因为 `gather_pp` 的同页去重把
重复 rowid 折掉了；而 `τ` 几乎不变。**方向对、量级远小于担忧。**

`docs/prefetch-lead-time.md` §4.3 的「批量为敌」应改为：
**「F 随 batch 次线性增长（去重使其饱和），故在实测 batch 区间内仍被掩盖；
真正的敌人是吞吐目标 S（τ 随 S 线性缩小），不是 batch。」**

**仍未做**：冷态 `fadvise`；batch 512+（本机 KV cache 受 4090 24 GB 限制）。
