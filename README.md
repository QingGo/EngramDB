# EngramDB

> **消歧**：GitHub 上另有多个同名 "EngramDB" 项目，多为通用 Agent 记忆/语义检索产品。本项目与它们无关。
>
> **EngramDB = DeepSeek Engram / Qwen PLE（N-gram 嵌入记忆表）的磁盘优先存储引擎。**
> 不做向量检索、不做 ANN、不做通用 KV 数据库。
> 它把「确定性哈希寻址的 n-gram 嵌入表」变成像 DuckDB 一样可嵌入、可构建、可预取、可服务的本地数据库。

---

## 1. 解决什么问题

Qwen3.8-Flash-Next / DeepSeek-V4.1 一类模型里的 **PLE / Engram 表**有四个特点：

| 特点 | 含义 |
|---|---|
| **超大** | Qwen3.8 约 51.2 GB（FP8，128 shard × 2.5M 行 × 160 B）；V4.1 Engram 约 **202.8 GB** |
| **静态只读** | 训练完就固定，推理期不写 |
| **地址先验** | rowid 由 token 序列**确定性哈希**得到 ⇒ 要读哪些行在计算开始前就全部已知 |
| **读得碎** | 每个 token 要 16（Qwen）~ 48（V4.1）行，每行只有 160 B ~ 数 KB |

「地址先验」是最关键的一条：**因为地址已知，所以预取、批合并、按访问序重排都能在 token 生成时就做好**，而不是等到了 PLE 层再同步读盘。

EngramDB 要把它变成一条命令链：

```text
build  →  index  →  warm  →  serve
```

同时服务两条负载：

- **负载 A：训练 / 语料预处理** —— 高吞吐批量 e_t 生成；
- **负载 B：在线推理** —— 低延迟点查 + 与引擎计算重叠的预取。

### 1.1 位置：EngramDB 做什么，不做什么

| | |
|---|---|
| **做** | 表的物理布局、定址、构建/校验工具、读路径、预取计划、Store-P 物化视图、Python/Rust/C ABI 三面 API、上游引擎（vLLM/SGLang/engram-peft）接入层 |
| **不做** | ANN / 向量检索；通用 KV；**可写**的规范表（训练写路径归 DeepEP 与 3FS，见 `docs/roadmap.md` §29.10）；模型前向计算 |

---

## 2. 核心设计

### 2.1 两套存储视图

| 视图 | 内容 | 特点 |
|---|---|---|
| **Store-I** | 原始行表，按 `shard / badge` 分片存放 | 与上游引擎原生 gather 兼容、可位级审计；**N 行 = N 次独立 4 KiB 页读** |
| **Store-P** | 物化 e_t 视图：每个唯一 n-gram key 一条定长紧凑记录 | **N 次散读折叠为 1 次定长读** |

**这是本项目收益最大的一个设计决策。** 在 V4.1 几何（48 行/token）下、原生 NVMe 冷读实测：

| 形态 | @512 tokens, 32 线程 | 占 500 μs/token 预算 |
|---|---|---|
| Store-I（48 次散读） | 220 μs | 44% |
| **Store-P（1 次折叠读）** | **7.69 μs** | **1.5%** |
| **折叠增益** | **28.6×** | 页流量 196,608 B → 12,672 B（**15.5×**） |

代价是需要额外一份约等于原表的磁盘。**如果磁盘受限，应做部分物化或 FP8 视图，而不是默认全量。**

### 2.2 物理布局：直接寻址 + 页去重

```text
rowid → shard = rowid / rows_per_shard
        badge = (rowid % rows_per_shard) / badge_rows      // badge_rows = 4096 / row_bytes
```

- **直接寻址**：没有 B-Tree、没有扫描页结构，`rowid` 直接算出字节偏移；
- **badge 聚簇**：一个 badge 约 4 KiB 行数据（160 B 行 ⇒ 25 行/badge）；
- **页对齐读**（`BadgeGather::gather_pp`）：按 shard 分组 → shard 内按键升序 → 读 **4 KiB 对齐页**并在同页内去重。布局本身不保证页命中，页命中来自「把地址排序后再读」。

> ⚠️ 一个必须说清楚的事实：在真实 PLE 负载下 rowid 是哈希散开的，
> **实测 8192 个 key 命中 8192 个不同的 4 KiB 页 —— 恰好 1 行 1 页，零页共享。**
> 所以 Store-I 的成本就是「独立随机页读 × 行数」，没有「顺带命中」可捡。

### 2.3 读路径：并发度是主要杠杆

`pread` 路径的**队列深度 = 线程数**。原生 NVMe 上单线程 4 KiB 随机读延迟约 **77 μs**（QD1 由盘内流水线决定），
并发买到的是吞吐而不是单次延迟：

| 线程 | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| μs/行（Qwen 16 行/token，冷） | 85.7 | 22.9 | 12.8 | 7.5 | **4.7** |

两条随之而来的工程决策：

1. **默认线程数 = 32**（`engramdb_io::batch::DEFAULT_GATHER_THREADS`）。
   此前硬编码的 8 会让 V4.1 超预算（119–131%），32 把它压回 42–56%。
2. **常驻线程池**（`engramdb_io::pool`）取代每次调用 `std::thread::scope`。
   本机单次 `spawn` 实测 **30–35 μs**，t=32 时每次调用要起约 22 个线程 ⇒ **约 0.7 ms/次调用**。
   换池后小 batch 冷读 **1008 → 515 μs（1.9×）**。`ENGRAMDB_NO_POOL=1` 可退回旧行为。

> ⚠️ 池子大小**不要**用 `available_parallelism()`：它遵守 cgroup CPU 配额
> （实测在 128 核宿主上返回 **16**），会把 IO 并发度卡死。现取下界
> `max(available_parallelism(), 32)`，且任务数 > worker 数时自动回退到 `thread::scope`。

### 2.4 三级缓存与预取

```text
T1 RAM 热集   → 频率优先 + LRU，--ram-budget 可配
T2 OS 页缓存  → mmap / fadvise，主动批量预读
T3 块设备     → preadv（默认）；io_uring 为可插拔语义实现
```

- **主动预取，不靠被动 page fault**；rowid 确定性 ⇒ 预取计划可在 token 生成时产生；
- GPU 路径上预取起点应早于「到达 PLE 层」；
- 全表**顺序流**可达 930 MB/s，而随机序只有 88.7 MB/s —— **顺序化是最大的未兑现杠杆**（按访问序重排视图槽位）。

---

## 3. 实测性能

### 3.1 先读口径：本项目最容易出错的地方

这个项目在性能测量上踩过**同一类错误的四次**（三次是自己的），所以口径必须前置：

1. **冷读的合法性不能靠「表比内存大」论证。** 在 120 GiB cgroup 上限、表只有 25 GB 的机器上，
   一份被读过的数据会**永久**留在 page cache 里。
2. **`drop_caches` 在容器里通常不可用**（无 `cap_sys_admin`）。替代方案是**按文件 `fadvise(DONTNEED)`**。
3. 每个门禁都强制**冷热自校验**：同一批 key 立刻重读，**比值 ≥5× 或边际 ≥2 μs** 才算冷；
   否则打印 `>>> VOID` 并以退出码 3 终止，**数字不得引用**。
4. **共享机器上的 A/B 必须配对交替**（`A B B A`）并在同一次脚本内完成 ——
   顺序扫描里「先跑」的那组会吸收上一轮余波，本项目因此**两次**把 1.9× 的收益读成 1.7× 的损失。
5. 每个基准**自报实际走的代码路径**，否则「A/B 无差异」可能只是「两组跑的是同一条路径」。

**参考介质（除非另行标注）**：原生 NVMe，XFS on RAID1（2× Samsung PM9A3 7.68 TB，PCIe 4.0 x4），
真实 Qwen3.8 PLE 行，`fadvise` 冷读。
工具：`crates/engramdb-bench/src/bin/{nvme_gate,view_gate,call_cost}.rs`、`scripts/nvme_raw_probe.c`。

> 设备交叉验证：裸 C 探针 `O_DIRECT` 与 buffered-全新偏移给出**几乎相同**的数字
> （77.28 vs 78.35 μs/4 KiB 页），两条原理不同的路径互证冷读成立。

### 3.2 每 token 预算（核心指标）

预算为 **500 μs/token**（= 100 tok/s 下 5% 的每 token 时间，推导见 `docs/design.md` §7.3）。

**V4.1 Engram 几何（48 行/token，冷读，中位数）：**

| 形态 | 线程 | 16 tokens/次 | 64 | 512 | 4096 |
|---|---|---|---|---|---|
| Store-I | 8（旧默认） | 655 μs（131%） | 612（122%） | 593（119%） | 592（118%） |
| Store-I | **32（现默认）** | **282（56%）** | **239（48%）** | **220（44%）** | **212（42%）** |
| Store-P | 32 | 49.6（9.9%） | 24.6（4.9%） | **7.7（1.5%）** | **5.5（1.1%）** |

**Qwen PLE 几何（16 行/token，冷读）：** 8 线程 204 μs（41%）；**32 线程 76 μs（15%）**。

**单线程下 Store-I 与行数线性**（85 μs/行 × 行数）：V4.1 就是 4.1 ms/token（816%）。
⇒ **Store-I 必须靠并发 + 足够大的 batch；Store-P 在任意 batch 下都进预算**（batch=1 也只要 81 μs = 16%）。

### 3.3 介质对比（8 线程冷读 μs/行）

| 介质 | μs/行 | 比原生 NVMe 慢 |
|---|---|---|
| **原生 NVMe（RAID1 PM9A3）** | **12.6** | — |
| Mac + USB 外盘 | 30.0 | **2.4×** |
| WSL2 + VHDX | 55.9–57.9 | 4.4–4.6× |

- **真实倍率是 2.4×，不是 40×。** 早期文档曾宣称 NVMe 比 USB 快约 40×，那个数字来自
  复用同一批偏移的 warm 读数，**已撤回**。
- **WSL2/VHDX 不是生产介质的有效代理**：它的虚拟化存储栈引入的延迟超过了介质差异本身
  （比 Mac 的外接 USB 还慢约 2×）。
- 冷热比 **7.1–8.1×**，真冷单页延迟 **≈77 μs**。**缓存态与介质类别同等重要，前者不能被后者掩盖。**
- 同一 seed 复跑偏差 **≤2.4%**（宿主有其他租户，load ≈11–13，不影响 4 KiB 随机读延迟）。

### 3.4 引擎自身开销

冷读边际稳定在 **3.9–4.5 μs/行**（32 线程），与裸设备 QD32 的 2.74 μs/页一致（+43%，含排序/拷贝）；
单线程时引擎比裸设备只多 **11%**。**开销几乎全在介质上，引擎不是瓶颈。**

warm（页缓存命中）口径下的纯代码路径成本 —— 这些数字**只能说明代码路径便宜，不能作为预算结论**：

| 路径 | 批量 | μs/token |
|---|---|---|
| `Store.fetch` | 4096 tokens/次调用 | 9.5–10.9 |
| `PleMemory.fetch_raw` | 4096 | 11.8–13.2 |
| `PleMemoryAdapter`（torch） | 4096 | 20.4–21.0 |

> 早期把上面这组数字当作「有 24× 余量」是错的 —— 它们是 warm 口径。**已撤回。**

### 3.5 其它已固化指标

| 路径 | 环境 | 结果 |
|---|---|---|
| Store-P 紧凑槽吞吐 | Mac+USB，200K 热态，8 线程 | 4.50M 行/s，字节放大 **1.00×** |
| Store-P 全表顺序流 | 顺序序 | **930 MB/s** ⚠️ warm 口径，冷态复测见 §4.3 |
| Store-P 全表随机序 | 随机序 | 88.7 MB/s |
| DiskSlotIndex v3 build / verify | 本机，10M grams | 135.2 s / 87.0 s |
| 单记录延迟（warm） | 1 线程 | p50≈0.75–0.88 μs，p99≈1.4–12 μs |

数据固化在 `probes/*.json|csv`，门禁在 `scripts/{real_perf_gate,overhead_budget_check,decode_baseline_check}.py`。

---

## 4. 哪些有用，哪些没用

### 4.1 已被证明有用

1. **Store-P 物化视图** —— N 路 scatter → 1 次定长读；V4.1 几何下 **28.6–41.5×**，磁盘读放大 20× → 1.00×；
   也是唯一让 V4.1 在 batch=1 也进预算的路径。
2. **并发度 + 常驻线程池** —— 1→32 线程 **18.1×**；线程池再在小 batch 上拿 **1.9×**。
3. **页对齐读 + 同页去重**（`gather_pp`）—— 避免 llama.cpp 式「4.75M 次 gather 零同页」的反面路径；
   在顺序访问下有效，在哈希散列负载下收益为零（已实测）。
4. **主动预取 + 按访问序重排**（方向）—— 顺序序 930 MB/s vs 随机序 88.7 MB/s。
5. **紧凑定长槽（无 pad）** —— 相对 4 KiB 对齐槽放大 1.00× vs 1.60×（见 §4.2 第 1 条）。

### 4.2 已被证明没用 / 不值得投入

1. **4 KiB pad 视图槽** —— 初版对齐槽放大 1.60×、吞吐 0.97M；紧凑槽放大 1.00×、吞吐 4.50M。
2. **为大语料训练做热集 / 频率索引** —— 30M token 真实语料中 top-1000 覆盖率 **<6%**，Zipf 假设不成立。
   频率索引只对 agent 型负载有效（top-100 覆盖 99%）。
3. **在 USB / HDD / SD 上做性能采样** —— 那是介质上限，不是引擎设计问题。树莓派 SD 只做功能门禁。
4. **盲目「全量物化」** —— 视图需要额外一份等大磁盘，磁盘受限时应做部分物化。

### 4.3 未验证（不要当成已验证，也不要当成已否证）

1. **io_uring 的性能收益。** 早期结论（逐提交 0.97×、批量 0.94×）**测于 WSL/VHDX**，
   而 WSL 已判定不是有效介质代理 ⇒ **证据失效，结论回到「未验证」**。
   在原生 NVMe 上**无法验证**：Docker 默认 seccomp 让 `io_uring_setup` 返回 **`EPERM`**（已独立确证）。
   `crates/engramdb-bench/src/bin/uring_gate.rs` 保留，供有权限的机器一次跑出答案。
   **但队列深度问题是真实的**（8→32 线程快 2.7×），所以这条路值得在有 io_uring 权限时重开。
2. **端到端 GPU / 真机 decode**（vLLM/SGLang 的 tok/s 验收）—— 待硬件。
3. **训练流有效吞吐 ≥100K tok/s**。
4. **顺序化视图的大表冷态复测**（930 MB/s 是 warm 顺序流）。

---

## 5. 安装与使用

### 5.1 Python 包

已发布到 PyPI：`engramdb-python`（import 名是 `engramdb`），要求 Python ≥ 3.10，
提供 Linux x86_64/aarch64、macOS x86_64/arm64、Windows x86_64 wheel。

```bash
python3 -m pip install --upgrade engramdb-python
# 或
uv add engramdb-python
```

#### 核心存储与视图

```python
import engramdb

# Store-I：打开原始行表
store = engramdb.Store(
    "/path/to/rows",
    shards=128,
    rows_per_shard=2_500_012,
    width=160,
    # threads=32,   # 可选；默认 = engramdb_io::batch::DEFAULT_GATHER_THREADS（32）
)
data = store.fetch([rowid1, rowid2, rowid3])
store.close()

# Store-P：打开物化视图
view = engramdb.View("/path/to/view.bin")
rec = view.read_record(0)

# SGLang 兼容的低层页读取
reader = engramdb.PageReader(page_size=4096)
pages = reader.read_pages([fd0, fd1], [offset0, offset1])

# Linux 上还有 io_uring 版（注意：容器默认 seccomp 下不可用）
if hasattr(engramdb, "IoUringPageReader"):
    io_reader = engramdb.IoUringPageReader(page_size=4096)
    pages = io_reader.read_pages([fd0, fd1], [offset0, offset1])
```

线程安全句柄管理：

```python
from engramdb import StorePool, ThreadLocalStore

pool = StorePool("/path/to/rows", shards=128, rows_per_shard=2_500_012, width=160, pool_size=4)
with pool as store:                       # 借出，用完自动归还
    data = store.fetch(rowids)

tls = ThreadLocalStore(pool)              # 每线程一个句柄（多 worker / 服务线程）
handle = tls.get()
try:
    data = handle.fetch(rowids)
finally:
    tls.release_current()
```

#### PLE rowid 与自动发现

```python
from engramdb import rowids_for_seq, rowids_for_seq_with_history
from engramdb import discover_ple, load_ple_weight_scale, load_ple_multipliers

rows = rowids_for_seq([248044, 1000, 99999, 42])          # -> [T, 16]
rows = rowids_for_seq_with_history([eos, eos], [10, 11, 12])

info  = discover_ple("/path/to/Qwen3.8-Flash-Next")        # 自动读元数据 + FP8 scale + multipliers
scale = load_ple_weight_scale("/path/to/Qwen3.8-Flash-Next")
mult  = load_ple_multipliers("/path/to/Qwen3.8-Flash-Next")
rows  = rowids_for_seq([248044, 1000, 99999, 42], info=info)
```

#### 快速 e_t tensor 读取

训练/预计算不要用 Python 逐行拼 bytes，直接一次 `Store.fetch` + `torch.frombuffer`：

```python
import torch
from engramdb import Store, fetch_e_t_tensor

store = Store("/path/to/real-ple-rows", shards=128, rows_per_shard=2_500_012, width=160)
e_t = fetch_e_t_tensor(
    store,
    flat_rowids,                                # [T * 16] 扁平行列表
    scale=0.00019931793212890625,
    num_heads=16, head_dim=160,
    dtype=torch.float8_e4m3fn, out_dtype=torch.float32,
)
# e_t.shape == (T, 16, 160)
```

#### 多表 / Arrow / 最小服务

```python
from engramdb import Database
from engramdb.arrow_utils import store_fetch_arrow, table_to_ipc_bytes

db = Database("/path/to/tables-root")
print(db.list_tables())
raw = db.fetch("alpha", [1, 3], shards=1, rows_per_shard=100, width=256)
```

服务端与客户端详见 `python/README.md`。

#### Serving 层（按需加载，不阻塞核心导入）

```python
from engramdb import PleMemory, PleSequence, PleSequenceStore
from engramdb import BundleManifest, TargetReaderRegistry
from engramdb import PleMemoryAdapter, install_target_reader_hook

mem   = PleMemory(store=store, head_dim=160, num_heads=16)
seq   = mem.new_sequence(); seq.feed([10, 11])
states = PleSequenceStore(mem, max_sequences=4096)     # continuous batching
states.feed("req-1", [10, 11])

bundle   = BundleManifest.load("bundle.json")
registry = TargetReaderRegistry()                      # 通用 reader 注册协议

adapter = PleMemoryAdapter(mem)
e_t = adapter(input_ids, seq_ids=[0, 1])
hook = install_target_reader_hook(model, reader, mode="post")
```

### 5.2 vLLM / SGLang：不修改源码，启动前 patch

> ⚠️ **验证边界（别把本节读成「已验证的 serving 配方」）**
> **已在真实引擎里跑通**（4090 + vLLM 0.29.0 + Qwen3.5-0.8B + 真实 PLE 行几何，
> 第 2 层注入，`enforce_eager=True`）：batch=1 下磁盘臂 43.9 tok/s vs 无 reader 45.0，
> **−2.4%，与 2.5% 的噪声地板同量级**；`shm`/`mmap` 臂落在基准之上。
> 见 `probes/serve_ple_ab_session42.md`。
> **但下列仍然是未验证/不成立的**：
> ① 上面的注入是**随机投影**、输出是乱码 —— 只测存储代价，**不构成质量声明**；
> ② 该次运行的本地表是 **65/128 分片**（26 GB），rowid 对现有行数取了模；完整
> 128 分片（47.7 GiB）此后已到位，冷态自校验跑的就是全表，但**臂的 A/B 数字尚未在全表上重跑**
> （脚本用 `glob` 动态发现分片，全表下取模是空操作，重跑无需改代码）；
> ③ 绝对 tok/s 是 eager 数字，**不是 CUDA-graph 数字**（Python reader 进不了 graph，
> 要改成 splitting op + `PIECEWISE` capture，见 `docs/engine-integration.md` §4.1）；
> ④ 本节示例用的 `embed_tokens_per_layer` 路径**仍未测**（实测跑通的是 `embed_tokens` 注入点）；
> ⑤ **Python 级后台预取实测让性能更差 2.7×**（GIL 争用），要掩盖 I/O 必须把
> rowid 生成 + 取数 + 张量构造合并为一次 GIL-free 的 PyO3 调用 —— 见
> `docs/prefetch-lead-time.md` §6.2；
> ⑥ FP8 `weight_scale` 的反量化不在这一层。
> 另外，「加速」只能相对**同样从磁盘读**的方案或「跑不起来」成立 —— 表能装进 HBM 时用本库一定更慢。

```python
# ---- vLLM ----
from engramdb import Store
from engramdb.vllm_plugin import install_vllm_ple

store = Store("/path/to/engram-rows", shards=..., rows_per_shard=..., width=...)
install_vllm_ple(
    Qwen3_8FlashNextNGramEmbedding,      # 你实际跑的 vLLM 模型类
    store=store,
    attr_name="embed_tokens_per_layer",
    embedding_dim=hidden_size_per_layer_input,
)
from vllm import LLM
llm = LLM(model="...", ...)
```

```python
# ---- SGLang ----
from engramdb.sglang import install_sglang_ple, install_sglang_io_uring_reader

install_sglang_ple(Gemma4Model, store=store,
                   attr_name="embed_tokens_per_layer",
                   embedding_dim=hidden_size_per_layer_input)

install_sglang_io_uring_reader()          # 或者只替换低层 reader
```

面向 serving 的更通用方式是 `PleMemoryAdapter` + `TargetReaderHook`：

```python
from engramdb import PleMemoryAdapter, install_vllm_target_reader, install_sglang_target_reader

adapter = PleMemoryAdapter(memory)
hook = install_vllm_target_reader(model, reader, mode="post")     # 或 install_sglang_target_reader
```

> 旧的 `install_vllm_ple` / `install_sglang_ple` 仍保留，用于「只替换 PLE embedding 表」的兼容路径。

`DiskPleEmbedding` 支持后台预取、超时、共享 executor、错误回退与统计：

```python
from engramdb.vllm_plugin import DiskPleEmbedding

emb = DiskPleEmbedding(store, num_embeddings=..., embedding_dim=160,
                       dtype=torch.float8_e4m3fn, cache_size=4096, prefetch_timeout=0.5)
emb.prefetch([rowid1, rowid2, ...])
out   = emb(torch.tensor([...]))
stats = emb.get_stats()
wait  = emb.get_wait_distribution()       # p50/p90/p99/max
emb.close()
```

### 5.3 真实 PLE 磁盘 Adapter

不加载完整的大 PLE 表，直接用磁盘 Store 替换真实 PLE n-gram embedding：

```python
from engramdb import discover_ple, Store
from engramdb.ple_adapter import disk_ple_from_discovery, DiskPleNGramEmbedding

info  = discover_ple("/path/to/Qwen3.8-Flash-Next")
store = Store("/path/to/real-ple-rows", shards=128, rows_per_shard=2_500_012, width=160)

ple = disk_ple_from_discovery(store, info)        # 自动用 checkpoint 的 weight_scale 做 FP8 反量化
ple = DiskPleNGramEmbedding(store, embedding_dim=2560, num_heads=16, scale=info["weight_scale"])
```

### 5.4 engram-peft

```python
from engramdb.integrations import install_disk_multi_head_embedding
from engramdb.integrations import install_real_qwen_ple_embedding

install_disk_multi_head_embedding(store)                                     # float32 磁盘 MultiHeadEmbedding
install_real_qwen_ple_embedding(store, model_dir="/path/to/Qwen3.8-Flash-Next")   # 真实 FP8 注入
```

### 5.5 Rust / CLI

crates.io 已发布：`engramdb`（主库 + CLI）、`engramdb-core`（布局/定址/manifest）、
`engramdb-io`（视图/gather/IO 后端/线程池）、`engramdb-keygen`（确定性 rowid）。

```bash
cargo add engramdb engramdb-core engramdb-io engramdb-keygen
cargo install engramdb
engramdb --help
```

```rust
use engramdb_keygen::PleSpec;

let spec = PleSpec::real();
let rows = spec.rowids_for_seq(&[248044, 1000, 99999, 42]);
println!("{} rows, first = {:?}", rows.len(), rows[0]);
```

CLI 子命令：`build` / `index` / `gather` / `verify` / `bench-real` / `warm` / `view` /
`slot-index` / `prep` / `tables` / `serve` / `check`。

```bash
engramdb tables <root>
engramdb check <root>
engramdb view build data/real-rows 2000 /tmp/view.bin /tmp/keys.txt --slot 2560
engramdb view build data/real-rows 0 /tmp/full.view /tmp/full.keys.txt \
    --keys-stream /tmp/all-keys.txt --slot 2560 --slot-index /tmp/slot-idx
engramdb view bench data/real-rows /tmp/view.bin --keys /tmp/keys.txt --sub 2000
engramdb view lat /tmp/view.bin --warm
engramdb slot-index build  /tmp/keys.txt /tmp/slot-idx        --buckets 16384
engramdb slot-index build  /tmp/keys.txt /tmp/slot-idx-single --buckets 16384 --single-file
engramdb slot-index verify /tmp/keys.txt /tmp/slot-idx-single --cache 1024
engramdb serve <root> --port 8765 [--binary]
```

---

## 6. 当前状态

| 项目 | 状态 |
|---|---|
| 版本 | **v0.3.0**（crates.io + PyPI + GitHub Release 均已发布） |
| Python 桥 | **PyO3 是唯一后端** —— 扩展随 wheel 分发，**无纯 Python 回退**，导入失败即抛出带修复指引的 `ImportError`（roadmap §34） |
| C ABI | `crates/engramdb-cabi`（`libengramdb_c`）—— 面向 **C/C++ 的嵌入面**，Python 包不再加载它。⚠️ 只实现 `PLE_QWEN_V1`，V4.1/DeepSeek 规格未实现（技术债 V55） |
| PLE rowid | Python / C ABI / PyO3 / Rust 四路径一致，golden 对拍 |
| 真实 PLE | `discover_ple` + `load_ple_weight_scale` + `DiskPleNGramEmbedding` + FP8 磁盘适配 |
| 语义索引 | `SlotIndex` + `DiskSlotIndex`（v1/v2 多文件，v3 单文件 + offset table），原生 CLI 构建/校验 |
| Serving 层 | `PleMemory` / `PleSequence` / `PleSequenceStore` / `BundleManifest` / `TargetReaderRegistry` / `PleMemoryAdapter` |
| 引擎接入 | vLLM `PleDiskGather` + SGLang 低层 reader，均为类级 patch hook，不改上游源码 |
| 多表 / 服务 | `Database` + JSON / 二进制 Arrow IPC 最小服务 |
| CI | `cargo fmt` / `clippy -D warnings` / `test --workspace`（**32 passed**）+ Python wheel smoke + C ABI smoke + 基线门禁 |

### 6.1 验收目标

| 指标 | 目标 | 状态 |
|---|---|---|
| 视图字节放大 | ≤2× | ✅ **1.00×** |
| 视图路径吞吐 | ≥4M 等效行/s | ✅ 4.50M（200K 热态，8 线程） |
| **Engram 每 token 开销** | **≤500 μs/token** | ⚠️ **条件成立**：Store-I 需 batch ≥16 且 32 线程（V4.1 282 μs，56%）；**Store-P 任意 batch 都成立**（1.5–10%）。见 §3.2 |
| CPU 小模型 decode（**代理**） | 内存表 vs 磁盘表的相对开销固化并入门禁 | ✅ 代理闭环 |
| CPU 小模型 decode（**真机**） | ≥50 tok/s（配 MTP 冲 100） | ⏳ 待硬件 |
| **rowid 与引擎一致** | 与引擎自己的 PLE 代码**逐位相同** | ✅ **IDENTICAL** —— 37 用例 / 18,048 个 rowid，`probes/ple_rowid_exactness_session42.md` |
| GPU 端 vLLM A/B 差距 | ≤5% | ⚠️ **数值已达标，验收未闭合** —— 见下方六条子条件 |
| GPU 端 SGLang A/B 差距 | ≤5% | ❌ **当前不可做**：SGLang 0.5.19 无 `qwen4_exp`、无 `ple_layer_ids` 代码路径（见子条件 6） |
| 训练流有效吞吐 | ≥100K tok/s | ⏳ 未闭环 |

> 「待硬件」两项需要一台能加载 Qwen3.8-Flash-Next（FP8 ≈90 GB）或 DeepSeek-V4.1-Flash（≈510 GB）的机器。

#### 「GPU 端 A/B ≤5%」的六条子条件

2026-09-12 在 RTX 4090 + vLLM 0.29.0 + Qwen3.5-0.8B + 真实 PLE 表上跑通了真 serving，
batch=1 磁盘臂 45.7 tok/s vs 无 reader 45.2、batch=32 各臂均在噪声内 ⇒ **数值达标**。
但当时验的是下面这套配置，**逐条列出来才算验收**（`probes/serve_ple_ab_session42.md`）：

| # | 子条件 | 状态 |
|---|---|---|
| 1a | **输入嵌入**用模型自己的权重，不是随机投影 | ✅ 逐位忠实性已证（`probes/ple_disk_faithfulness_session42.json`，248,320 行 bf16 全等）；serving 侧已跑通，两臂 **greedy token id 逐位相同**（`probes/serve_faithful_embed_ab_session42.md`） |
| 1b | **PLE 路径**用模型自己的权重 | ❌ **本机不存在这样的 checkpoint**：带 `ple_layer_ids` 的 config 只有 `Qwen3.8-Flash-Next-FP8-tokenizer`（22 MB，无权重）；三个 Qwen3.5 的 `text_config` 一个 PLE 字段都没有。所以 16 行/token 的 PLE 臂**只能是**合成投影 |
| 2 | **完整分片** | ✅ 128 分片真数据全在（47.7 GiB）；`padded_vocab` 与磁盘行数相等（320,001,536） |
| 3 | **冷态**且自证驱逐生效 | ✅ 自校验已入脚本（实测 ratio 78.3×，判定 `cold`）；**待独占重跑** |
| 4 | **CUDA graph** 路径，或显式标注 eager | ❌ 目前是 `enforce_eager=True`。Python reader 进不了 graph，需改成 splitting op + `PIECEWISE` capture。（注：vLLM 自己也把 PLE 的 rowid 生成移出了 PIECEWISE graph） |
| 5 | 多种子/多轮 counterbalance 到噪声地板以下 | ⚠️ 忠实 A/B 已跑，但**性能栏判 VOID**：效应 0.54% ≪ 噪声 3.9%，`added_us_per_token` 甚至为负（−360.8 µs）。效应本身由 `disk_reader 114.2 µs/call` 直接测得，不需要 tok/s 反推 |
| 6 | **多引擎**（vLLM + SGLang） | ❌ **不是未做，是不可做**：vLLM 0.29.0 有 `Qwen4ExpForConditionalGeneration`（`vllm/models/qwen4_exp/`，含 `_validate_ple_layer_ids()`）；**SGLang 0.5.19 一处都没有** —— registry 无条目、无 `ple_layer_ids`、transformers 5.12.1 无 `qwen4_exp`。SGLang 缺的是 PLE 这个「缝」，不是我们的库不兼容 |

> **为什么必须逐条列**：eager 模式下引擎自身开销约 22 ms/token，5% 预算 = 1.1 ms，
> 而存储代价只有 0.6 ms —— **「达标」是廉价的**。见 roadmap §35.1。
> 「待硬件」已不成立：机器有，缺的是这六条。
> 在此之前，每 token 开销门禁是**可本地复现的替代证据**：它不测模型端到端，但 5% 预算在算术上由它保证。

#### 接入面在哪：vLLM 里的那一行

真 PLE 架构是 `qwen4_exp`（`Qwen3.8-Flash-Next`）。它在 vLLM 侧收束到
`vllm/models/qwen4_exp/nvidia/ple_layer.py` 的两行：

```python
ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
return self.ngram_embedding(ngram_ids).flatten(-2)      # ← 唯一需要改的行
```

三点结论（均为实测，见 `probes/ple_rowid_exactness_session42.md`）：

1. **上半行我们已经逐位相同。** 用引擎自己的 `compute_ngram_ids`（未修改的函数体）
   当裁判，对照 EngramDB 的生产 Rust 路径：37 用例、18,048 个 rowid、
   multiplier 由 `seed=1234` **独立推出**而非拷贝 ⇒ **IDENTICAL**。
2. **引擎自己已经把这个切口切好了。** 注释写着 *"Keep num_reqs-dependent ID
   generation outside PIECEWISE CUDA graphs"* —— 算 rowid 与取行在引擎里本来就是
   两件事。这正是预取需要的分工。
3. **下半行无路可走，这正是我们的位置。** `ngram_embedding` 是
   `PLEVocabParallelEmbedding`（GPU 常驻）。vLLM 0.29.0 的 offload 只有
   **整张量 / 整层** 两种粒度（`cpu_offload_gb` 按参数名段 + GiB 预算；
   `offload_group_size` 按 decoder layer 分组）。PLE 表是 layer 2 里的
   **单个 51 GB 张量**，所以任何 offload 配置都只能整表搬运，**没有「按 rowid 取
   16 行 = 2,560 B」这个粒度**。这不是配置能解决的问题。

---

## 7. 项目结构

```text
EngramDB/
├─ crates/
│  ├─ engramdb-core/      布局、badge、直接寻址、频率索引、manifest
│  ├─ engramdb-io/        Store-I/Store-P 读路径、批量 gather、IO 后端、常驻线程池
│  ├─ engramdb-keygen/    DeepSeek Engram / Qwen PLE 确定性 rowid
│  ├─ engramdb/           主 CLI + 最小服务
│  ├─ engramdb-bench/     探针与门禁（nvme_gate / view_gate / uring_gate / call_cost）
│  ├─ engramdb-cabi/      C ABI（C/C++ 嵌入面；**不是** Python 后端）
│  └─ engramdb-pyo3/      PyO3 原生扩展（Python 唯一后端）
├─ python/engramdb/       Python 包：Store/View/PageReader/PLE discovery/adapter/serving/引擎适配
├─ docs/                  设计、路线图、交接、规格、许可
├─ scripts/               构建、发布、探针、门禁
└─ probes/                实测数据（JSON/CSV/txt）
```

---

## 8. 文档导航

**先读这两个：**

- `docs/handoff.md` —— 空白上下文 agent 的交接：最新状态 / 资产 / 环境 / 待办
- `docs/design.md` —— 技术架构、负载模型、预算推导（§7.3）

**按需查：**

| 文档 | 内容 |
|---|---|
| `docs/roadmap.md` | 终极目标、技术债、借鉴矩阵、阶段计划、**每轮实测的完整记录与撤回声明** |
| `docs/engram-specs.md` | Engram / PLE 结构规格与证据链 |
| `docs/v41-engram-analysis.md` | DeepSeek-V4.1-Flash Engram 技术报告解读与规格闭合 |
| `docs/engine-integration.md` | vLLM / SGLang / llama.cpp 接入调研 |
| `docs/upstream-patches.md` | 不改上游源码的接入补丁草图 |
| `docs/licenses.md` | 许可与合规边界 |
| `python/README.md` | Python 包安装、引擎适配、多表 / Arrow / 服务客户端 |

**自己跑一遍：**

```bash
bash scripts/gate.sh          # cargo fmt / clippy -D warnings / test + 基线门禁
bash scripts/linux_verify.sh  # Linux 实机 wheel 冒烟
bash scripts/release_gate.sh  # 发布门禁（含真表 Arrow IPC 与 serving 阈值）
```

---

## 9. 一句话路线图

先证 **存储面**（已基本完成），再证 **端到端**（CPU/GPU 小模型 + PLE 的真实 tok/s），
最后把 **服务化 / 多表 / Arrow IPC** 与 **真实上游引擎接入** 做成稳定产品面。

当前三个缺口，按优先级：

1. **V4.1 支持落地**：keygen v2（4-gram、压缩 token map、DEAD mask）、存储参数化、
   V4.1 几何下的 Store-P 折叠 —— 输入已冻结，纯本地可做；
2. **顺序化视图的大表冷态复测**（930 MB/s 目前是 warm 顺序流）与按访问序重排；
3. **端到端真机验收**（vLLM/SGLang 的 PLE tok/s）—— 待硬件。
