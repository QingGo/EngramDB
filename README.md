# EngramDB

> **消歧声明**：GitHub 上另有多个同名 "EngramDB" 项目，多为通用 Agent 记忆/语义检索产品。
> 本项目与它们无关。
>
> **EngramDB = DeepSeek Engram / Qwen PLE（N-gram 嵌入记忆表）的磁盘优先存储引擎。**
> 它不做向量检索、不做 ANN、不做通用 KV 数据库；它把“确定性哈希寻址的 n-gram 嵌入表”
> 变成像 DuckDB 一样可嵌入、可构建、可预取、可服务的本地数据库。

---

## 1. 这个项目解决什么问题

Qwen3.8-Flash-Next 一类模型中的 **PLE / Engram 表**是：

- 超大、静态、只读的 n-gram 嵌入记忆表；
- 由 token 序列通过确定性哈希得到 rowid，因此 **查询地址在推理/训练开始前就已知**；
- 每个 token 需要读取固定 16~32 行、每个 payload 只有数 KB；
- 原始表规模可达 48GiB（FP8）~ 95GiB（BF16），不适合简单整表加载到 RAM/显存。

EngramDB 的目标是把这种表变成：

```text
build  →  index  →  warm  →  serve
```

一条命令链即可使用的磁盘优先存储基础设施，同时服务：

- **负载 A：训练/语料预处理**——高吞吐批量 e_t 生成；
- **负载 B：在线推理**——低延迟点查 + 与引擎计算重叠的预取。

---

## 2. 核心设计

### 2.1 两套存储视图

| 视图 | 内容 | 适用场景 | 特点 |
|---|---|---|---|
| **Store-I** | 原始行表，按 badge/分片存储 | 与引擎原生 gather 路径兼容、位级审计 | 原始 16 头 scatter 读放大高 |
| **Store-P** | 物化 e_t 视图，每个唯一 n-gram key 存一条 2560B 紧凑记录 | 推理点查、训练流主路径 | 16 次小读折叠为 1 次定长读 |

Store-P 的关键结论（真表实测）：

- 紧凑 2560B 槽（无 pad）是最终选型；
- 相比原始 scatter，IOPS 从 16:1 降到 1:1；
- 实际磁盘读放大可降到 **1.00×**；
- 代价是需要额外一份约等于原表大小的磁盘。

### 2.2 物理布局：badge

```text
rowid → badge_id = rowid / BPows
badge  = 连续 BPows 行
```

- 行按 badge 聚簇；
- badge 对齐到 4KB，并尽量对齐 2MB（Linux huge-page folio）；
- 直接寻址，无 B-Tree、无扫描页结构；
- 这是“布局即优化”的核心：把随机读变成可预测的页命中。

### 2.3 三级缓存与预取

```text
T1 RAM 热集    → 频率优先 + LRU，用户可配 --ram-budget
T2 OS 页缓存   → mmap / fadvise，主动批量预读
T3 NVMe       → preadv 默认；io_uring 作为可插拔语义实现
```

核心原则：

- **主动预取，不靠被动 page fault**；
- 预取计划在 token 生成时就可以产生，因为 rowid 是确定性的；
- 对 GPU 路径，预取起点应早于“到达 PLE 层”，而不是到了 PLE 层再同步读。

### 2.4 当前 IO 后端结论

| 后端 | 相对性能 | 结论 |
|---|---|---|
| `preadv`（默认） | 1.00× | 本地 NVMe/VHDX + 8 线程下已达到 IO 上限 |
| `UringBackend`（逐提交） | 0.97× | 无性能收益 |
| `UringBatchBackend`（批量） | 0.94× | 无性能收益 |

**结论：默认 preadv；保留 io_uring 语义实现，供网络盘 / cgroup 受限等未来环境激活。**

---

## 3. 当前实测性能

> 口径：真表 320M 行 × 160B FP8；外接 USB SSD 或桌面 NVMe/WSL；见 `docs/probes` 与 `probes/`。

### 3.1 关键数字

| 路径 | 环境 | 性能 | 备注 |
|---|---|---|---|
| A. 原始 16 行 scatter | USB SSD，8 线程，warm | 1.05M 行/s | 字节放大 20×，页命中极差 |
| B. Store-P 紧凑槽 | 200K 热态，8 线程 | 4.50M 行/s | 放大 1.00× |
| B. Store-P 全表冷随机 | USB 外盘 | 554K 行/s | 外盘 IOPS 上限 |
| B. Store-P 全表半冷随机 | WSL/NVMe，8 线程 | 19.2M 行/s | 桌面 NVMe 目标介质 |
| B. 全表顺序流 | NVMe | 930MB/s | 顺序化是最大未兑现杠杆 |
| 单记录延迟（warm） | 1 线程 | p50≈0.75–0.88μs，p99≈1.4–12μs | 比 10ms/token 低 3 个数量级 |
| 单记录延迟（Linux SSD 真冷） | 1 线程 | p50≈3.7μs，p99≈6.7μs | 冷热差仅约 1.85× |

### 3.2 验收目标

| 指标 | 目标 | 状态 |
|---|---|---|
| 视图路径吞吐 | ≥4M 等效行/s | ✅ 已达到 |
| 视图字节放大 | ≤2× | ✅ 1.00× |
| 端到端 CPU 小模型 decode | ≥50 tok/s（配 MTP 冲 100） | ⏳ 待实机 |
| GPU 端 vLLM/SGLang A/B 差距 | ≤5% | ⏳ 未做 |
| 训练流有效吞吐 | ≥100K tok/s | ⏳ 未闭环 |

---

## 4. 优化策略：哪些有用，哪些没用

### 4.1 已经被证明有用的

1. **Store-P 物化视图（2560B 紧凑槽）**
   - 16 路 scatter → 1 次定长读；
   - 相对原始 scatter 约 5× 以上吞吐，且磁盘读放大从 20× 降到 1×。
2. **并行 IO**
   - 8 线程才能兑现桌面 NVMe 带宽；
   - 单线程会被 IOPS 上限压住在 ~11K IOPS / 数十万行每秒。
3. **主动预取 + 访问序调度（方向）**
   - 全表随机序 88.7MB/s vs 顺序序 930MB/s；
   - 下一步应按实际访问序重排视图槽位，或按窗口顺序化读取。
4. **badge / 页对齐布局**
   - 保证页命中率，避免 llama.cpp 式“4.75M 次 gather 零同页”的反面路径。
5. **把“冷/热”交给现代 SSD**
   - NVMe 上真冷与热差异只有约 1.85×；
   - 真正影响性能的是介质类别（USB/HDD vs NVMe），不是页缓存态。

### 4.2 已经被证明没用/不值得投入的

1. **io_uring 追求性能**
   - 本地 NVMe/VHDX + 8t 下，逐提交 0.97×、批量 0.94×，均不如 preadv；
   - 已定案：不继续在 io_uring 性能上花时间。
2. **为大语料训练做热集 / 频率索引**
   - 30M token 真实语料中 top-1000 覆盖率 <6%，Zipf 假设不成立；
   - 频率索引只对 agent 型负载有效（top-100 覆盖 99%）。
3. **4KB pad 视图槽**
   - 初版 4KB 对齐槽放大 1.60×、吞吐 0.97M；
   - 紧凑 2560B 槽放大 1.00×、吞吐 4.50M，明显更优。
4. **USB/HDD/SD 介质上的性能采样**
   - 外盘性能是介质上限，不是引擎设计问题；
   - 树莓派 SD 性能采样已放弃，只做功能门禁。
5. **盲目“全量物化”**
   - 视图需要额外一份磁盘；如果磁盘受限，应做部分物化/FP8 视图，而不是默认全量。

---

## 5. 安装与使用

### 5.1 Python 包（推荐入口）

```bash
python3 -m pip install --upgrade engramdb-python
```

当前发布线包含 Linux x86_64/aarch64、macOS x86_64/arm64、Windows x86_64 wheel。

```python
import engramdb

# Store-I：打开原始行表
store = engramdb.Store(
    "/path/to/rows",
    shards=...,
    rows_per_shard=...,
    width=...,
)
data = store.fetch([rowid1, rowid2, rowid3])
store.close()

# Store-P：打开物化视图
view = engramdb.View("/path/to/view.bin")
rec = view.read_record(0)

# SGLang 兼容的低层页读取
reader = engramdb.PageReader(page_size=4096)
pages = reader.read_pages([fd0, fd1], [offset0, offset1])

# 如果是 Linux，还有 io_uring 版
if hasattr(engramdb, "IoUringPageReader"):
    io_reader = engramdb.IoUringPageReader(page_size=4096)
    pages = io_reader.read_pages([fd0, fd1], [offset0, offset1])
```

### 5.2 vLLM：不修改源码，启动前 patch PLE 表

```python
from engramdb import Store
from engramdb.vllm_plugin import install_vllm_ple

store = Store("/path/to/engram-rows", shards=..., rows_per_shard=..., width=...)

install_vllm_ple(
    Qwen3_8FlashNextNGramEmbedding,   # 你实际跑的 vLLM 模型类
    store=store,
    attr_name="embed_tokens_per_layer",
    embedding_dim=hidden_size_per_layer_input,
)

from vllm import LLM
llm = LLM(model="...", ...)
```

### 5.3 SGLang：不修改源码，启动前 patch PLE 表

```python
from engramdb.sglang import install_sglang_ple

install_sglang_ple(
    Gemma4Model,                     # 你实际跑的 SGLang 模型类
    store=store,
    attr_name="embed_tokens_per_layer",
    embedding_dim=hidden_size_per_layer_input,
)

# 然后正常启动 SGLang
```

也可以只替换低层 reader：

```python
from engramdb.sglang import install_sglang_io_uring_reader
install_sglang_io_uring_reader()
```

### 5.4 engram-peft

```python
from engramdb.integrations import install_disk_multi_head_embedding

install_disk_multi_head_embedding(store)
```

### 5.5 Rust / CLI

```bash
cargo test --workspace
cargo run --release -p engramdb -- view build data/real-rows 2000 /tmp/view.bin /tmp/keys.txt --slot 2560
cargo run --release -p engramdb -- view bench data/real-rows /tmp/view.bin --keys /tmp/keys.txt --sub 2000
cargo run --release -p engramdb -- view lat /tmp/view.bin --warm
```

---

## 6. 本项目当前状态

| 项目 | 状态 |
|---|---|
| crates.io | 四个核心 crate 已发布 |
| PyPI | `engramdb-python` 多平台 wheel 已发布 |
| Python 桥 | PyO3 原生扩展优先，ctypes 回退 |
| CI | cargo test + clippy + Python wheel smoke |
| SGLang 适配 | 低层 reader + 模型类 patch hook |
| vLLM 适配 | `PleDiskGather` + 模型类 patch hook |
| 性能契约 | 存储面已闭环，端到端待实机 |

---

## 7. 项目结构

```text
EngramDB/
├─ crates/
│  ├─ engramdb-core/      布局、badge、直接寻址、频率索引、manifest
│  ├─ engramdb-io/        View/ IO backend / 批量 gather / 预取计划
│  ├─ engramdb-keygen/    DeepSeek / Qwen PLE hash 与 rowid 生成
│  ├─ engramdb/           主 CLI
│  ├─ engramdb-bench/     探针
│  ├─ engramdb-python/    C ABI ctypes fallback
│  └─ engramdb-pyo3/      PyO3 原生扩展
├─ python/engramdb/       Python 包：Store/View/PageReader/适配层
├─ docs/                  设计、路线图、session-log、接入调研
├─ scripts/              构建、发布、探针、门禁
└─ probes/               实测数据与复现说明
```

---

## 8. 文档导航

- `docs/handoff.md` —— 空白上下文 agent 交接，最新状态/资产/环境/待办
- `docs/design.md` —— 技术架构、负载、性能基线、风险
- `docs/roadmap.md` —— 终极目标、技术债、借鉴矩阵、阶段计划
- `docs/engram-specs.md` —— Engram/PLE 结构规格与证据链
- `docs/engine-integration.md` —— vLLM / SGLang / llama.cpp 接入调研
- `docs/upstream-patches.md` —— SGLang/vLLM 不改源码的接入补丁草图
- `docs/session-log.md` —— 分 session 复盘
- `docs/licenses.md` —— 许可与合规边界
- `scripts/gate.sh` —— 本地门禁
- `scripts/linux_verify.sh` —— Linux/WSL/树莓派 wheel 实机冒烟
- `scripts/vllm_ple_smoke.py` —— 真实 vLLM 模型类 `install_vllm_ple` 验证
- `scripts/sglang_ple_smoke.py` —— 真实 SGLang 模型类 `install_sglang_ple` 验证

---

## 9. 路线图一句话

先证明 **存储面**（已基本完成），再证明 **端到端**（CPU/GPU 小模型 + PLE 的真实 tok/s），
最后把 **服务化 / 多表 / Arrow IPC** 与 **真实上游引擎接入** 做成稳定产品面。

当前最重要缺口：

1. 完成真实 vLLM/SGLang serving 中的 PLE 端到端 tok/s 验收（功能 hook 已在真实模型类上验证）；
2. 完成顺序化视图的冷盘 A/B 与调度，兑现全表随机 88.7MB/s → 数百 MB/s 的潜在提升（访问序构建/读取已在 WSL 验证）；
3. 服务化与多表形态。

> 已闭环：
> - 树莓派 aarch64 + WSL2 Ubuntu x86_64 均通过 v0.2.4 wheel 完整冒烟。
> - vLLM 0.28.0 与 SGLang 0.5.9 的真实 `Qwen3ForCausalLM` 均通过 `install_vllm_ple` / `install_sglang_ple` 类级 patch 及 `DiskPleEmbedding` 前向验证（Session 9）。
> - 访问序视图 `view build --keys` + 校验 + 顺序/随机读已在 WSL 跑通（Session 10）。
