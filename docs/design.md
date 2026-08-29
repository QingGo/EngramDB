# EngramDB 技术设计与开发方案

> 版本：v0.1（初稿，2026-08-29）
> 配套文档：`docs/engram-specs.md`（Engram/PLE 结构规格与证据链，本文只引用其结论）
> 状态：接受评审中；M0 探针优先执行，后续里程碑以数据为准滚动修订。

---

## 0. TL;DR（一句话方案）

EngramDB 是一个 **Rust 编写、磁盘优先、DuckDB 风格（嵌入式、无守护进程、目录即库）** 的静态嵌入表存储引擎，
专门为 "DeepSeek Engram / Qwen PLE" 这类**确定性哈希寻址的 N-gram 记忆表**提供：
**两套存储视图**（原始表 / 物化 e_t 视图）、**三种索引**（直接寻址、频率统计、分段列表）、
**三级缓存**（RAM 热集 → OS 页缓存 → NVMe）、**确定性预取计划**与**批量 gather 张量原语**，
并同时覆盖"训练语料预处理/预训练流"（高吞吐）与"推理点查"（低延迟）两种负载。

与 `engram-peft`（Python 研究库）保持**零耦合**，通过薄协议 + 适配例联动；
与 vLLM / SGLang / llama.cpp 通过**适配层**互操作（不做 fork，往上游贡献）；
CPU-only 是 EngramDB 的一等公民场景。

---

## 1. 背景与问题定义

### 1.1 场景（用户视角）

- 用户想要：把 **Qwen3.8-Flash-Next 的 51.2B PLE（n-gram）记忆表**冻结提取，嫁接到一个**更小模型**上；
  小模型主干随机初始化、gating+conv 与其余层可训练；表冻结、训练不更新。
- 设备：**单机、无 GPU**（开发机：macOS；生产目标：Linux / 消费级 GPU 可选）。
- 约束：显存/内存都可能不足（PLE BF16 全表约 95 GiB，FP8 约 48 GiB）；
  **磁盘可用但不宽裕**（开发期用**模拟数据**；最终用 ModelScope 单文件下载验证）。
- 需求：
  - 推理：CPU（或消费级 GPU）跑小模型主干，PLE 查询需要**低延迟**；
  - 训练/预处理：语料仓库先**离线查表**得 e_t（每 token 的 n-gram 向量），
    训练时把 e_t 序列与 packed token 序列一起流式喂入 → **高吞吐**；
  - EngramDB 被视为 "database"：**serve 不同 Engram 表、可建立不同索引**。

### 1.2 负载画像（两库两负载）

| | 负载 A：语料预处理/预训练流 | 负载 B：在线推理（单 token/批 token） |
|---|---|---|
| 查询键 | token 序列 → n-gram → rowid（双 hash：DeepSeek 式压缩版 / Qwen raw 版） | 同左（实时） |
| 访问模式 | 顺序语料 + 段内近随机；**Zipf 强**（短语重复） | 近似均匀（聊天/评估负载实测无热集，见 specs §3.4） |
| 每 token 字节 | 16×(160×1B)=2.5KB（FP8）~ 5KB（BF16） | 相同 |
| 关键指标 | **吞吐**（有效 tok/s、字节放大率） | **延迟**（p50/p95、每 token 额外毫秒） |
| 结果去向 | `[B,T,2560]` e_t 张量 → GPU/CPU 训练 | 行向量 → gating/conv（模型内） |

### 1.3 关键事实（决定设计的三个"只有 Engram 有"）

1. **rowid 预知性**：所有行地址在模型开始计算前全部由 token 确定 → 存在"预取计划"这种原语，
   通用 DB（RocksDB/LMDB/DuckDB）都没有对应操作。
2. **常数 IOPS、小 payload**：每 token 16~32 行、数 KB；与表大小无关（MoE 做不到）。
3. **行级零局部性**：llama.cpp 实测 4.75M 次 gather 无一命中同 4KB 页、16 个独立区域相距 ~20M 行
   → **布局优化（badge 聚集/物化视图）是唯一能改变"冷路径 IOPS"的杠杆**。

### 1.4 非目标（明确不做）

- 不做近邻/ANN 检索（查询语义是精确点查；不引入 HNSW/PQ 等）。类似"为什么不是向量库"，
  作为设计文档的边界说明保留。
- 不做 Engram 表的训练/更新（训练只写日志、更新表不在本项目范围；未来以段式增量设计预留）。
- 不做分布式多机（单机/小集群之后再说；FileStore 抽象保留 S3 远景）。
- M0 不实现任何引擎侧的 patch（只做适配契约与基准对照）。

---

## 2. 总体架构

### 2.1 形态

- **嵌入式模式（默认）**：`engramdb` 库/CLI 直接操作目录；`EngramStore::open(path)`；
  进程内 CPU 资源池（线程、内存预算均可控）。
- **服务模式（M4）**：`engramdb serve`，Unix domain socket + Arrow IPC；
  一进程多表、多索引、共享缓存，可被训练进程/引擎进程共享热点却互不加载两次。

### 2.2 逻辑分层

```
┌────────────────────────────────────────────────────────────┐
│  上层（不属于 EngramDB）                                     │
│  engram-peft (EngramLayer) │ Qwen PLE adapter │ 训练数据管线 │
└─────────────┬───────────────────────────┬──────────────────┘
              │ 协议：直接调用 (embedded) / Arrow IPC (server)
┌─────────────▼───────────────────────────▼──────────────────┐
│ EngramDB client API (EngramStoreClient)                     │
│  open/close(table_id) · prefetch(plan) · fetch(keys)        │
│  fetch_e(gram_keys) · build_index · stats                   │
├─────────────────────────────────────────────────────────────┤
│ PrefetchPlanner │ BatchGather │ TierManager │ Metrics        │
├─────────────────────────────────────────────────────────────┤
│ Store-I（原始表视图）    Store-P（物化 e_t 视图/段式训练库）    │
│  布局/索引/统计/清单/校验          同左                        │
├─────────────────────────────────────────────────────────────┤
│ IO backend trait：Linux io_uring / macOS preadv + kqueue     │
│ (页对齐 · 有界批次 · 双缓冲 · GIL-free)                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 两套存储视图

**Store-I（原始表视图 / 兼容视图）**
- 内容：checkpoint 原始张量（`[160, 320001536]` 列主或按头分区重排），
  与"一次 16 行 gather"语义完全一致。
- 作用：
  1. 与引擎本体（vLLM/SGLang llama.cpp）的本机原路径兼容（它们自己做 gather）；
  2. 未来做"按头分区 + badge"重排后的最优原始读取（引擎集成时输出该布局，行内数值不变，位级一致）。
- 注意：对**推理均匀分布**负载，Store-I 的 16 区域 gather 注定是 16 次页面访问 ——
  这正是 Store-P 视图出现的动机。

**Store-P（物化 e_t 视图 / 训练库）**
- 内容：对每个"唯一 n-gram key"存一条 **2560 维（16×160）e_t 记录**（16 行拼接结果）。
- 规模：20,000,000 条（Qwen 口径）× 2560 维 ≈ 51B 参数 ≈ 与原始表等大（BF16 ~95GiB / FP8 ~48GiB）。
  值 = 数据库中"物化视图"：把随机 16 路小读折叠成**一次定长读**，IOPS 16:1、页命中 16:1。
- Key 选择：
  - 训练流：n-gram 元组 → 稳定的 128 位 hash（tuple hash，与上层哈希独立）；
  - 查询时：任何引擎语义一致——**只要 n-gram 元组相同，e_t 相同**，该视图对推理/训练都有意义。
- **取舍声明（设计期）**：视图需要额外一份磁盘（约等于原表大小）。M0-P4 用数据回答"值得与否"；
  若磁盘受限，提供"部分物化"（仅热集/量化/FP8 视图）选项，且 Store-P 可以只存在训练侧。

### 2.4 三种索引（用户说的"数据库技术"落地）

| 索引 | 键 | 结构 | 用途 |
|---|---|---|---|
| I1 直接寻址 | rowid | 行 = 数组位移 + badge 块偏移（O(1)，零开销） | Store-I 主存取；无需任何 B-Tree 页结构 |
| I2 频率统计 | rowid → count | 与 rowid 对齐的平铺 u32 数组（构建期一次写） | 缓存保留优先级 + 指标；**仅 agent 型负载有效**（实测 top-100 99%；大语料无效见 §7.1） |
| I3 段式列表 | (segment_id, key) | 段内：排序索引 + 连续值块 + 统计头（min/max/count） | Store-P 训练流：阶段化顺序扫描 + 段缓存；跨段只读一次 |

I2 之所以有：**仅用于缓存保留优先级与指标**；绝不作为正确性依赖。
（修正：2026-08-29 P2 大语料实测证明"语料侧 Zipf 有效"仅对局部窗口/短语料成立——30M tokens 下唯一行占表空间 37-49%、top-1000 覆盖 <6%；agent 型负载实测 top-100=99%，才是热集策略的真实适用域。详见 §7.1）。

---

## 3. 存储与运行时设计（详细）

### 3.1 物理布局：badge

- 定义：`badge = 连续 BPows 行`，行内 row-major；
  badge 字节数 = `BPows × row_bytes`，**必须对齐到 4KB，并尽可能对齐到 2MB（Linux huge-page folio）**。

  | 行宽 (dtype) | row_bytes | 4KB 页内行数 | 推荐 badge |
  |---|---|---|---|
  | 160 × FP8 | 160B | 25.6 → 24 行（3.75KB） | 24 行/3.75KB 或 32 行/5KB |
  | 160 × BF16 | 320B | 12.8 → 12 行（3.75KB） | 24 行/7.5KB 或 32 行/10KB |
  | 64 × FP16（DeepSeek demo） | 128B | 32 | 32 行/4KB |

- 动机：
  1. 每次 IO 的"有效字节"/"命中字节"最大（字节放大率最小）；
  2. 规避 SGLang NVMe 路径踩过的 **2MB folio 分配**坑：badge 块按 folio 对齐 + `madvise(MADV_RANDOM|DONTNEED)` 按需释放；
  3. 直接寻址：`rowid → badge_id = rowid / BPows`，`in-badge offset = rowid % BPows`（全部整数除法一次完成）。

### 3.2 一级/二级/三级缓存策略

- **T1 RAM 缓存（用户配置 `--ram-budget`）**：
  - 采用"**频率优先保留 + LRU 最近使用**"：I2 已给出每行 count，缓存保留 = max(热集榜单, LRU)；
  - Store-P 视图：`sizeof(record)=5KB`，20M 条 ≈ 95GiB（BF16）/48GiB（FP8），
    通常放不进 RAM → T1 存"top ~X% by count"与近期窗口；其余交 T2。
- **T2 OS 页缓存（mmap/syscall 读）**：每页只被"正在使用"的 badge 接管；
  - Linux：`posix_fadvise(POSIX_FADV_RANDOM)` + `WILLNEED` 批量（先于 use 一个窗口）；
  - macOS：`fcntl(F_RDADVISE/ F_NOCACHE)` 平衡；
  - 重要：**不靠被动 fault**（llama.cpp 的 13.1 faults/token 是反面教材）；
- **T3 NVMe（io_uring / POSIX AIO）**：page-aligned、有界提交批次（默认 64）、双缓冲、GIL-free。

### 3.3 预取计划（PrefetchPlanner）——核心差异点

- 输入：**下一个窗口（1..N_ubatch）内全部 rowid / gram keys**（可以由任意引擎提前给出，
  EngramDB 不拥有 token 流——由上层提供 keys）。
- 处理：去重 → 按 badge 排序 → 合并连续区间（coalesce）→ 按 tier 判定 →
  T3 → 批量 IO 提交；T2 → `madvise(WILLNEED)`；T1 → 无操作。
- 窗口与外设规律：
  - 推理 decode：每 token 窗口 = 1 step；由于键在 token 化时就已知，**预取提前量 ≈ 一个 compute window**；
  - GPU 路径特别注意：PLE 在第 2 层，GPU 上前两层计算窗口 <1ms → **预取起点必须在"token 生成时刻"而非"到达 PLE 层"**（本方案与引擎内的 SGLang/vLLM 预取(只重叠 PLE 层)的区别点）；
  - 训练流：提前一个段（segment）的预算。
- 多线程/async：固定线程池（可配置 4-32），每线程持有独立 `uring`/`aiocb` 队列；
  结果进**锁-free 环形 staging 池**（复用预分配页），供 `gather` 使用。

### 3.4 批量 gather 原语

- `fetch(keys: &[u64]) -> TensorView(B, d)`：
  - 一次调用完成：分层查找 → 预取集合 → 异步等待 → 拼接 reshape → 输出 Arrow 数组视图 / numpy 视图 / 拷贝到 pinned buffer（供 H2D）。
  - 对应用方（torch/llama 等）：表现为"一次 request 一个 batch"，不感知 tier 组织。
- `fetch_e(gram_keys) -> [B, 2560]`：Store-P 版本（若启用视图）。
- 统计数据：hit-rate/IO 计数/读放大，全部上报 metrics（§7.4）。

### 3.5 构建流程（预处理，一次性）

```
token 语料 → [上层 hash 规范：Qwen ple / DeepSeek demo] → rowid 流
   → (a) 计数：精确外排 or Count-Min（可配）→ I2 频率数组
   → (b) 建表：按 badge 排序 → 顺序写 Store-I badged 文件 + manifest+ 校验和
   → (c) 可选：Store-P 视图：唯一 gram key → e_t 记录（顺序写）
   → (d) hot-set 面板写入 manifest（供 T1 使用）
```

- 吞吐设计：全部**顺序写**（NVMe 3GB/s+）；随机读仅发生在"从源表读取行"（必要时用源表 = checkpoint shards 直读）。
- `engramdb-index` 命令输出：`manifest.json`（表规格/布局/版本/校验），count 文件，hot-set。

### 3.6 服器协议（M4 草案）

- `unix://` socket；Arrow IPC **零拷贝张量流**（批量 fetch）与请求-响应（prefetch/stats）；
- 多表：`table_id` 前缀；每表独立 TableSpec（badge 大小、cache 预算、精度、索引开关）；
- 观察：open/warm/stats 遥测。

---

## 4. 负载设计（含预算估算）

### 4.1 负载 A：训练预处理/预训练流（高吞吐）

- 流水线每步：token 批 → hash → rowid 集 → （T1 命中 / T2 页 / T3 IO 并发批取）→
  `[B, T, 2560]` e_t 序列 → 与 packed token ids 一起送入训练（字节量与上下文无关）。
- 吞吐模型：
  - 每 token 独特行：训练流访问靠**页级并行 LRU**（实测 local 型 80% 页命中、12.8KB/tok）而非热集（大语料 Zipf 失效，见 §7.1）；
  - NVMe 有效吞吐目标：**≥ 1GB/s**（响应有效 tok/s ≥ 100K 量级，受 CPU hash/tokenize 限制，文档发布前实测）；
  - 指标：u03 有效 tok/s，u02 字节放大率。
- 工程注意：与训练数据加载器衔接 —— 提供 `engramdb-dataset`（PyO3 包）:
  与 HF datasets / 自研 loader 对标的 `IterableDataset`，输出 (token_ids, e_t) 流；`pack` 语义与上层一致（记录段边界）。

### 4.2 负载 B：推理（CPU 优先）

- 每 token 预算（以"50 tok/s（20ms/token）"为目标，`docs/design.md §7` 有验收）：

| 环节 | 预算 | 依据 |
|---|---|---|
| 小模型主干（CPU） | ~17ms（Q4~1B M 系列；0.5~1B x86 DDR5） | 内存带宽线性推算（specs §7 硬件表） |
| EngramDB | **≤1~2ms**，且与主干计算重叠，实际尾差 ~0.1ms（冷）/ ~0.05ms（热） | 16 行 = 2 页量级；io_uring 批读 +
  提前一个 compute-window 预取后，SSD 尾延迟被吸收 |
| 其他开销 | ~0.5ms | 采样/调度 |

- 做到 100 tok/s（10ms/token）的配方：≤0.5~1B Q4 + MTP/投机（自带头复用 2x）+ 表尽量进 RAM（或视角 Store-P + FP8）。
- **不因上下文变长而变化**：PLE 是固定 16 行/token，KV 缓存才是 CD 长尾巴（大模型侧与我们无关）。

### 4.3 负载 B：推理（消费级 GPU）

- 小主干窗口 2~15ms/token；EngramDB 主动预取成本 ≤0.2ms，占比 **≤3~5%**；
  - **必须钉死的前提**：预提取消"惰性"——引擎内实现（vLLM mmap、SGLang UVA）已经证明被动路径在快速
    GPU 上会形成 10-40% 级别的 stall（13 个 fault/token 就是一个真实用例）。本设计自始主动。
- 表在 SSD 时：单 token = 1-2 个 badge 读（Store-P）或 2-4 页（Store-I 原始布局）
  配 pinned staging + async H2D 流；对比 vLLM 磁盘路径实测 -8%（49-97 tok/s 基线），
  **我们的验收门槛：≤5%**（在 P 数据后复测）。
- 该场景属于 M1+（开发机上无 GPU），验收时在消费级卡上做 vLLM/SGLang 同机 A/B。

### 4.4 为什么"精确性必须是位级"：测量一致性

- Store-P 视图和 Store-I 的数值**必须完全一致**：e_t = concat(16 行) 是一个纯 reshape，
  gating 的 matmul (W_K/W_V) 在线执行。→ 校验位：**对同一 gram key 两种视图结果逐位一致（审计命令）**。

---

## 5. 与现有系统对比（借鉴与差异化）

### 5.1 可借鉴（已是社区验证过的路,我们照抄设计）

| 来源 | 机制 | 引用 |
|---|---|---|
| SGLang | 页对齐 + io_uring + 有界提交 + GIL-free + 双缓冲 + pinned staging + 异步 H2D + 与前置层重叠 | #36567 |
| vLLM | 去重/排序/合段 batch gather + READAHEAD (posix_fadvise WILLNEED) + CUDA graph 分段 | #54129 |
| llama.cpp | mmap + MADV_RANDOM + 每 ubatch 批量 WILLNEED + warm 顺序预读 | #27742 |
| 通用教训 | 无 cgroup 时页缓存被 checkpoint 冲走（部署指南）；FP8 表按 BF16 加载的 2x 膨胀 bug；TP1/uniproc 死锁 | #54070/#53899/#53960 |

### 5.2 我们的差异化（对以上都没做的事）

1. **布局即优化**：badge 块 + 2MB folio 对齐（SGLang 解决 folio 问题靠 madvise，我们是布局天然化解）；
2. **Store-P 物化视图（16:1 IOPS）**：没有任何引擎做（lm.cpp 实测证明原始布局 16 分区零局部性）；
3. **频率索引 + 分层**：引擎只做"放哪一层"，不做"为访问分布服务"（适用域已实测校准：agent 型负载有效 top-100=99%，大语料/chat 无效——见 §7.1；据此设计的三档分层仅作缓存级优化，不作正确性依赖）；
4. **双负载双库**：训练预计算库是引擎完全没覆盖的场景（引擎只做推理）；
5. **CPU-only 一等公民**：引擎路径大部分要求 CUDA/UVA/pinned（vLLM 需 N 卡、SGLang UVA 需统一内存模型）；
   我们无任何 CUDA 依赖，Mac/Linux 都能跑（且我们主动预取优于 llama.cpp 的被动 fault 路径）；
6. **天然规避已知 bug 面**：存储精度=存储属性（无 2x 膨胀风险）；无 worker/IPC 死锁拓扑；
7. **多表/多索引 + 服务化**（M4）：满足"serve 不同 En...表"的目标。

### 5.3 边界说明（为什么不"直接用"现有方案）

- 通用 DB（RocksDB/LMDB/DuckDB/SQLite）：点查（KV/B-Tree 页随机）或扫描（列式）都输给
  "直接寻址 + badge 布局"的 IOPS/字节放大；
  训练流段式列表 + 列式统计可以借鉴 DuckDB/Parquet 的思想，但整体形态仍需自研（见 §5.1 只抄实现）。
- 向量数据库：使用内存假设、图索引、ANN 语义——与"精确点查"完全不同，不做；类似
  "为什么不是向量库"写入文档作为对照（对用户有价值）。
- 现有引擎路径：已在 §5.1 列明，它们强于"引擎内 async/Fused kernel"，我们强于"存储体系与布局"。

---

## 6. 集成与互操作

### 6.1 与 engram-peft（用户自家库，~code/engram-peft）

- 现状：`EngramModel/EngramLayer/ContextAwareGating`；`nn.Embedding(sum(primes), d)` 在 GPU 上；
  hash 为 DeepSeek demo 式（压缩词表 + 全局素数）。
- 策略：**引擎不动**（冻结/训练接口一切不变）；
  EngramDB 提供：
  - `EngramStorage` 协议（Rust 侧 trait + Python 侧 typing.Protocol 对齐）：open/prefetch/fetch/fetch_e/stats；
  - `examples/interop_engram_peft.py`：将 `EngramLayer.forward` 的
    `self.embedding(shifted_indices)` 替换为 `storage.fetch(keys)` + 注册为 buffer 的 hook（示例演示）；
  - 将来（M2）通过 PyO3 的 `engramdb` 包让研究者一行 `storage="disk"` 切换；
  - **原则**：共享"表规格"且 hash 在层内按协议传入；EngramDB 不 import engram_peft，反过来也不引入。

### 6.2 与 Qwen3.8 PLE 的上层适配

- `examples/qwen_ple_adapter.py` + Rust `engramdb-keygen`：
  - hash spec：raw token、乘子 (splitmix64 派生, 64 位)、每头素数模、行偏移、EOS 段重置（详见 specs §3.3）；
  - 与 transformers/llama.cpp/vLLM 实现交叉对拍（M0-P0b 的"参考向量"）。

### 6.3 与推理引擎（vLLM / SGLang / llama.cpp）

- **接入契约**（引擎视角：一个"能 gather 的存储后端"）：
  ```
  EngramStoreClient {
    open(table, spec) -> handle
    prefetch(keys | gram_keys, window) -> None   // async fire & forget
    fetch(keys) -> TensorView[n,d]               // 阻塞，或 awaitable（async 引擎）
    set_scale(fp8) -> None
    stats() -> TierStats
  }
  ```
- 落地路径分级：
  1. **零改动**：训练侧 + engram-peft 研究流（无引擎依赖）；
  2. **轻适配**（引擎 plugin/CI 上游）：SGLang #36567 的 Rust NVMe reader 整体替换为 `engramdb-io`
     （同构且更全：布局 + 视图）；llama.cpp 的 PLE CPU gather 从 mmap 换成 badge 布局后端；
     vLLM `PleOffloadLayer` 的 offload worker 换为 client 调用（用我们做表加载/预取/分层）。
  3. **不进则不强求**：任何引擎若与我们的厂商深度绑定（UVA kernel、CUDA graph 分段），
     我们只贡献"数据面"（存表布局 + 索引 sidecar），不算引擎逻辑。
- **审计**：发布前在消费级卡跑 vLLM/SGLang A/B，验收：吞吐差 ≤5%（目标值，看 M0 数据修正）。

---

## 7. 性能实测基线（2026-08-29 更新）与指标

### 7.0 实测基线（真表 320M×160 FP8 @USB SSD；全部可复现，见 probes/ 与对应 bin）

| # | 探针 | 实测 | 结论 |
|---|---|---|---|
| P1 | 批式 gather（`gather_pp`：4KB 页对齐+分片 8 线程） | 冷 1.09M 行/s → 暖 1.44M 行/s | 0.78 页/行的随机行吞吐上限（页复用 1.3 行/页） |
| P4-A | 16 行 scatter（100K grams / 1.6M 行） | 1.054M 行/s（warm），1,252,642 唯一页，**字节放大 20.04×** | 16 头区域相距 ~20M 行 → 页复用不可修复；这是"必须视图化"的实证 |
| P4-B | 视图单记录（4KB 对齐槽，8 线程） | **5.23M 行/s，放大 1.60×**（带宽 12.5× 更省） | Store-P 是该负载下的唯一稳健赢家；单线程被 ~11K IOPS 底线压制（533K 行/s） |
| P3 | 页级 LRU 模型（1M tokens） | local(语料型) 命中 80%@12.8KB/tok；uniform(chat 型) 命中≈0%@**64KB/tok** | 训练流带宽 <0.15GB/s@10Ktok/s（无压力）；在线负载必须视图（64KB→4KB/tok） |
| P2 | 三域真实语料（官方分词器） | 见下方修正 | **修正 Zipf/热集假设**（§7.1） |

### 7.1 实测修正与重新校准的目标

**修正 A（最重要）——"训练语料侧 Zipf 强 / 热集收益大"在大语料下不成立**：
30M tokens fineweb 与 zh 的 PLE rowid 唯一行分别为 **1.19 亿（37% 表空间）/ 1.57 亿（49%）**，
flat 唯一率 24.8% / 32.6%；top-1000 行仅覆盖 **5.99% / 3.13%**（旧 174K-token 小样本给出过 13%，属严重高估）。
→ I2 频率索引的定位调整为：**仅用于"缓存保留优先级 + 指标"与 agent 型负载**
（agent 实测 top-100 覆盖 99.2%、unique 仅 52 万——该负载热集策略有效），**不再是训练吞吐的依赖项**。
修正后目标：训练流吞吐 = 页级并行读 12.8KB/tok（local 型）即可满足 ≥100K tok/s 的带宽预算，
无需热集；在线/评测负载以视图为默认路径。

**修正 B——目标表从"设计预期"改为"待 P4b 确认的合同"**（端到端 decode 尚未实测）：

| 指标 | 符号 | 目标 | 状态 |
|---|---|---|---|
| 查询面吞吐（视图路径） | — | ≥ 4M 等效行/s（已实测 5.23M） | ✅ 已达标 |
| 视图字节放大 | u02 | ≤ 2× | ✅ 1.60×（4KB 槽）；未 pad 2560B 版 2×页但跨页，待选型 |
| 冷点查 p50/p95（一次视图记录） | u01 | ≤ 0.5/1.6ms（对齐 4KB，8 线程吞吐外推） | ⏳ 单条延迟待测 |
| 训练流有效吞吐 | u03 | ≥ 100K tok/s | ⏳ 受 tokenize/hash CPU 上限，P4b 集成实测 |
| 端到端 decode（CPU 小模型） | tok/s | ≥ 50（配 MTP 冲 100） | ⏳ P4b 实机 |
| 端到端 decode（消费 GPU） | 对引擎差 | ≤ 5% | ⏳ 需 Linux GPU 环境（M1.5 后） |
| 内存 | — | 固定 `--ram-budget` | 架构性满足 |

### 7.2 指标规约（探针开始即收集，报告常包）

- u00 tier hit-rate（T1/T2/T3 各自命中比）；
- u01 单次 fetch 延迟分布（p50/p95/p99）；
- u02 **字节放大率** = (实际读字节)/(最小需要字节) —— 实测口径：P4-A 20.04×（16 行 scatter 真表）
  与 P4-B 1.60×（视图）为本项目基准；社区参照：llama.cpp/SGLang 报过 138KB 读/2.5KB 用；
- u03 有效 tok/s（查询路径单独统计）；
- u04 磁盘占用/预取窗口命中情况；
- 报表：`probes/p4_view_notes.md`、`probes/p2_report_v2.json`、`probes/agent_workload_stats.json` + CSV 源数据。

---

## 8. 工程结构（Rust Cargo workspace）

```
EngramDB/
├─ docs/                          # 本设计 + 规格类文档
├─ crates/
│  ├─ engramdb-core/              # 无 IO 依赖：布局/badge、索引(I1/I2/I3)、keygen trait、manifest
│  ├─ engramdb-io/                # tier 管理、PrefetchPlanner、IO backend trait
│  │   ├─ backend-io-uring/       # Linux: io_uring, O_DIRECT 可选
│  │   └─ backend-preadv/         # macOS: preadv + kqueue; F_NOCACHE 平衡
│  ├─ engramdb-keygen/            # DeepSeek-demo 与 Qwen-ple 两套 keygen 的 Rust 实现（+对拍 golden）
│  ├─ engramdb-bindings/          # PyO3 → 包名 `engramdb`（训练器/engram-peft 用）
│  ├─ engramdb-cli/               # engramdb init|build|index|prefetch|warm|serve|probe|inspect|stats
│  └─ engramdb-bench/             # criterion 探针（P1-P6）
├─ examples/                      # interop_engram_peft.py / qwen_ple_adapter.py
├─ scripts/                       # 参考实现 python 对拍（P0b）/ 数据制备
└─ probes/                        # 合成表生成器、报告输出目录
```

- 核心依赖保持克制：`memmap2`、`libc`/`rustix`、可选 `tokio`/`io-uring`（IO crate 内）；
  Arrow 依赖仅 server/bindings 侧引入。
- 错误处理与安全：`no_std` 不需要；但核心算法（keygen、索引）必须是**确定性纯函数**保证可对拍；
  IO 错误全部保留 errno 语义（同 SGLang 做法）。

### 8.1 CLI 草案

```bash
engramdb build  --table qwen38-ple --layout fp8 --badge 32 --source safetensors
engramdb index  --table qwen38-ple --mode exact|sketch          # (I2 频率索引 + hot-set)
engramdb view   --table qwen38-ple --gram-size 3 --heads 8       # 构建 Store-P 视图
engramdb probe  --table qwen38-ple --workload point|stream ...  # 探针套件
engramdb warm   --table qwen38-ple --tier ram --budget 8G       # 预热热集
engramdb serve  --root ./stores --table qwen38-ple --listen /tmp/engramdb.sock
engramdb stats  --server ... | --local ...
```

### 8.2 Python 侧（生成 `engramdb` 包）

```python
import engramdb
store = engramdb.EngramStore("tables/qwen38-ple")     # embedded
store.build_index(mode="sketch")
plan = store.plan_prefetch(keys_batch)                # async
rows = store.fetch(keys_batch)                        # numpy [B, 160*dtype]
et = store.fetch_e(gram_keys)                         # [B, 2560]
```

---

## 9. 开发计划与里程碑

### 9.1 里程碑总览

| 里程碑 | 内容 | 出口标准（gate） |
|---|---|---|
| **M0 探针（p0-p6）** | hash/keygen 对拍；布局微基准；语料统计；训练流模拟；16:1 视图 A/B；三对照库横向；IPC 冒烟；端到端 decode 模拟 | `probe-report.md` 全指标达标 or 数据修正设计；无真实 51GB 表依赖（用合成数据 + config.json 单文件元数据） |
| M1 存储核心 | badge/索引/分级/预取/CLI build+index+warm 出品；I1-I3 rust 全套 | 与 Python 参考实现位级一致；bench（P4/P6 重跑） |
| M2 互操作 | PyO3 包 + engram-peft 适配例 + Qwen PLE 适配例 | 两个 examples 可运行；e_t 位级一致审计 |
| M3 训练管线 | Store-P 段式 + DataLoader 集成；序列一致审计 | 端到端训练流跑通；吞吐数据 v2 |
| M4 服务化 | Arrow-IPC server + 多表 + stat 遥测 | P5 复测：embedded vs server 开销 <2% |
| M5（可选） | 引擎适配上游（SGLang reader 替换 / vLLM worker client / llama.cpp gather 后端） | 上游贡献 + 消费卡 A/B（≤5% 差距） |

### 9.2 M0 详细执行方案（本项目的第一个交付物）

**期限**：1~2 周（单机开发，Rust 工具链）。**磁盘需求（合成期）**：≤4GB。**真表验证**：后置（M1 交付验证前）。

#### P0：结构与 hash 规格（先行，阻断项）

- 锁死来源：`llama.cpp#27742`（qwen4exp 实现）+ `conversion/qwen4exp.py`（GGUF 元数据派生）
  + 我们已获的 `config.json`（modelscope API，已验证）→ 产出 `docs/engram-specs.md §3` 的**算法式描述**：
  `n-gram 元组 → (head_i, rowid_i) i=1..16`，含乘子谱、素数表、偏移、EOS/段重置；
- DeepSeek demo 同理（代码已全部拿到，specs §2 已写就）；
- **对拍方法**：Python 参考实现（`scripts/ref_hashes.py`）与 Rust `engramdb-keygen` 对同一输入产生
  golden vectors（0b×1024 行），相等即通过；再与 transformers 实现（qwen4_exp 模块，transformers v5.8 主线）
  交叉核对 1024 行 + 含 EOS/段边界的序列。

#### P1：布局微基准（合成表）

- 两种生成的规格：DeepSeek-demo（约 1/10 尺度）与 Qwen-ple（结构一致、行长 160、行数缩小）；
- 扫描：badge {4KB, 8KB, 20KB, 64KB} × QD {1, 8, 32, 128} × 并发 {1, 4, 16}；
- 指标：有效吞吐 GB/s、p50/p95、字节放大；输出曲线 CSV。

#### P2：语料统计

- 输入：本地抓取语料样本（几十 MB～几百 MB，文本域跨广度）；
- 输出：唯一 n-gram 计数（精确 mode 1 次 + sketch 对照）、拟合 Zipf 系数 s、命中率随热集大小曲线
  （用于：训练热集大小决策）；给出 20M 条全量聚类的墙钟估算。

#### P3：训练流模拟

- 构造 packed 语料 → 段大小参数化 {64K, 256K, 1M tokens} × 缓存预算 {4G, 16G, 48G}；
- 指标：T1 命中率、有效 tok/s、分字节放大；输出"吞吐 vs 段大小/缓存"热图。

#### P4：16:1 视图 A/B（核心差异化数据）

- 相同随机 gram keys 集合，分别：A) Store-I 原始行 16 路 gather（页面命中 16 区域）；
  B) Store-P 1 条 5KB 记录；
- 指标：IOPS、延迟分布、字节放大，给出"物化视图是否值得省一倍磁盘"的判定。

#### P5：IPC/服务化冒烟

- 嵌入式 vs 简单 Arrow IPC loop：对 128/1024/4096 keys 批量的开销差（决定 M4 是否做服务）。

#### P6：横向对照

- 同机同数据：LMDB（`heed` 绑定）/ RocksDB / DuckDB（批流）/ 裸 mmap 基线
  vs 我们的 badge 原型；点查（冷/热）与批流两维；
- 结论：自研 vs 打包进"组件化"（设计偏好**自研数据面**，DB 只做元数据工具的裁决由本数据支撑）。

#### P4b：端到端 decode 模拟（范围调整：用公开小模型走真机）

- 选型：Qwen/Qwen3-0.6B 或 1.7B 的 GGUF（纯 CPU，llama.cpp）；
- 打点：在 PLE 注入点插入我们的 gather（同一台机器 / 模拟 SSD 或可配置）；
- 输出：**同机 tok/s 与 p95**；得出"达成 50/100 tok/s 所需模型规模/量化档曲线"（文档 §7.1 表的实测来源）。

### 9.3 开发规范

- 语言：Rust 2024 edition（讨论确认：内存安全 + tokio/io_uring + Arrow/pyo3 生态；
  对拍工具仍用 Python）。
- 编码风格：fmt/clippy 严格（dedicated CI 从 M1 起）；测试：
  - 单元（keygen/索引/badge 布局）→ proptest；
  - 集成（对拍 golden、审计命令）；
  - 性能（criterion，P1-P6 全部并入 `engramdb-bench`）。
- 提交规范：遵循 repository 约定（如 commitizen），M0 阶段信息约定用 `feat/probe` 前缀。

---

## 10. 风险与缓解

| # | 风险 | 概率/影响 | 缓解 |
|---|---|---|---|
| 1 | **Qwen ple hash 规范提取不完整**（modeling 代码散布于 transformers/vLLM/llama.cpp，未全部拿到源码级） | 中/高 | P0 多源交叉对拍（transformers 主线 + NeMo + llama.cpp + GGUF 元数据）；用官方 config.json + checkpoint 单 shard 做 1024 行 golden（modelscope 单文件，几十 MB 可接受）；实在不行以 llama.cpp 为准并注明来源 |
| 2 | 合成数据与真实表结构不符（如 128-shard 栈式布局） | 低/高 | 隔离"结构元数据"（config + 128 分片清单小文件）与"行数据"（合成）；M0 独立用元数据构建与真表一致的布局表头，合成只填数值 |
| 3 | 推理无热集（聊天）→ 热集假设失效 | 高/中 | 已修正：推理不依赖热集；频率索引仅用于训练与缓存分级；P4/P4b 测退化边界 |
| 4 | 页面缓存干扰（checkpoint 读取冲掉 PLE 页） | 高/中 | 明确的 `madvise`/O_DIRECT/容量控制文档；storage 侧提供"独立设备/分区"建议（部署指南） |
| 5 | 2MB folio 导致的 RSS 膨胀 | 高/中 | badge 天然 2MB 对齐；`MADV_DONTNEED` 主动放冷页；M0 在 Linux 复测（macOS 参考） |
| 6 | 磁盘不足（真实表 ~48GiB + 视图 ~48GiB） | 中/中 | 分阶段：只做 Store-I 或 Store-P 其一；提供"视图部分物化/热集物化"选项；M0-P4 给出收益/成本曲线供决策 |
| 7 | 引擎互操作"浅层适配"可能长期不下游 | 中/低 | 分阶段验收：先内部数据集与 engram-peft；上游贡献列 M5 可选 |
| 8 | 工程范围蔓延（服务化/多索引全要做） | 中/中 | 里程碑 gate 严格；M4 未达 P5 数据指标则砍 |
| 9 | 许可证 | 低/中 | 权重提取/改发合规（qwen-community-1.0；论文 Apache-2.0）。文档内不打包权重；研发用需自行验证许可条款 |
| 10 | 开发者环境只有 mac 而无消费级 GPU（GPU 路径未实测） | 低/中 | M1+ 添加可执行验证项：在消费级卡复测；文档标注"开发机结果仅 CPU 口径" |

---

## 11. 决策记录（ADR 摘要）

| # | 决策 | 理由 | 状态 |
|---|---|---|---|
| D1 | 独立库 `EngramDB`（不与 engram-peft 合并） | 生命周期/依赖/定位不同；研究库保持轻 | 已定 |
| D2 | Rust（不用 Python 为主语言） | IO 密集、内存在安全并发、io_uring/Arrow/PyO3 生态、p6 对照绑定齐全 | 已定 |
| D3 | 双存储视图（Store-I 原始 / Store-P 物化 e_t） | 16:1 IOPS vs 一份磁盘的取舍 | 设计定；P4 数据后确认 |
| D4 | 无 DB 依赖作为核心；LMDB/RocksDB/DuckDB 仅作基准与可选工具 | 自研数据面更贴合；写路径（LSM）当前不需要（表冻结只读） | 已定 |
| D5 | 频率索引不用于推理正确性，仅用于训练与分层 | 社区实测：chat 负载热集不存在 | 已定（本轮最重要修正） |
| D6 | 预取提前量 = compute-window（GPU 从 token 生成时便开始） | GPU 上 PLE 层前后窗口 <1ms；需在更早期掩盖 | 已定 |
| D7 | 开发期用合成数据；真表元数据（config.json 单文件）先行 | 磁盘受限；结构元数据极小而完整 | 已定 |
| D8 | M0 出"P4b 端到端 decode 模拟"（真实小模型、无 GPU） | 50/100 tok/s 需要实机数据 | 已定（本设计增量） |

---

## 12. 附录

### 12.1 术语

- PLE：Per-Layer Embedding（Qwen 的 n-gram 记忆表）；Engram：DeepSeek 论文的模块名。
- e_t：单 token 的 n-gram 检索向量（16×160=2560 维, Qwen 口径）。
- badge：聚集基础存储块（固定行数、页/folio 对齐）。
- Store-I / Store-P：原始表视图 / 物化 e_t 视图。
- I1/I2/I3：直接寻址 / 频率统计 / 段式列表 三种索引。

### 12.2 待办开放问题（不阻塞 M0）

- Qwen PLE 表精确"每头行数"与 128-shard 栈式布局的确切划分（依赖 P0 提取）；
- 服务化是否值得实现的最终裁决（P5 数据）；
- 物化视图的”部分物化“比率（P4/P2 数据）；
- 是否要“预热 + 常驻热集”作为 OS 重启缓存（由 P3 数据定）。
