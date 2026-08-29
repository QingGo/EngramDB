# EngramDB 战略路线图（Roadmap）

> 作者视角：工程负责人复盘。基于截至 2026-08-29 的本轮 session（P0-P4 全部探针 + 真实语料/真实权重/真表基准）。
> 配套文档：`design.md`（设计）、`engram-specs.md`（结构与证据）、`probes/*`（原始数据）。

---

## 1. 终极目标（北极星定义重写）

**一句话**：*让"Engram/PLE 类确定性哈希记忆表"成为任何小模型与推理/训练框架都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。*

展开成三条可衡量的轴线：

| 轴 | 定义 | 验收（不可妥协口径） |
|---|---|---|
| A. 性能契约 | 单机 CPU+NVMe（或消费 GPU）下：嫁接小模型 decode ≥50 tok/s（配 MTP 冲 100）；EngramDB 在任何其参与的计算中开销 ≤5% 且字节放大 ≤2× | P4b 实机曲线（CPU 小模型）与 Linux GPU A/B（vLLM/SGLang 差 ≤5%）双门禁 |
| B. 形态契约 | "再数据库"：单目录库、`build/index/warm/serve` 一条命令链、一进程多表多索引、Arrow 零拷贝、嵌入式/服务化双形态、任意上层（训练器/引擎/engram-peft）薄层接入 | 三个 interop example 可复现；server 开销 P5 ≤2% |
| C. 科学契约 | 每个设计论断（尤其"假设"）必须有**真表/真语料/真负载**实测或严格对拍；断言与证据同址存放 | 论断 → probes/ 可复现命令；假设修正走"§7.1 修正节"流程 |

**不是目标**：替代推理引擎、做 ANN/向量检索、做分布式集群、训练/更新嵌入表（写路径）。

---

## 2. 本轮 session 技术债清单（诚实盘点）

### A. 数据与测量债（影响决策可信度，最高优先级）
1. **Zipf/热集假设被推翻** → 已修正（design §7.1）；但 P3 模拟器的 local 分布参数（文档窗 20K、80/20 比例）是**人为设定**，尚未用 P2 真实 rowid 流校准 → 债：P3 v2 校准输入。
2. **端到端目标（50/100 tok/s、≤5% GPU 差）仍是设计外推**，无实机数据 → 债：P4b（原 M0 项）未闭环。
3. **单条视图记录延迟 p50/p95 未测**（只测了吞吐）；GPU 路径未在 Linux 复测（io_uring/O_DIRECT/大页行为=N/A）。
4. P1 早期数字混乱（mock 内盘 warm 4.05ms/batch 等）与真表口径未统一 → 债：探针报告规范化（统一 CSV 格式 + `probes/run_registry.md`）。

### B. 工程债
5. **IO backend trait 只有 preadv（macOS）实现**；io_uring 后端（设计 §8 承诺）未写 → 阻塞 Linux 生产路径。
6. **PrefetchPlanner/双缓冲 ring/统一 gather 链未成型**：gather_pp 直接读；warm 独立；`PrefetchPlan` 只在单测里。
7. **`engramdb-cli` 命名与内容混乱**：真正的 CLI 在 `engramdb-cli`，但 p4view/p3sim/p2rowid 落在 `engramdb-bench`；
   探针与产品命令不分家 → 债：bin 布局重构（`engramdb` 主命令 + `engramdb-bench` 只含 probe）。
8. **视图（Store-P）真表构建器缺失**：P4 只在 100K grams 规模验证；320M 真表视图未建（也未定槽位选型：4KB pad 1.6× vs 2560B 跨页）。
9. **回归/CI 缺失**：bitwise 一致性只有一次性 CLI 验证；golden 只在 keygen 单测；无 bench 门禁。
10. **下载/工具链语义债**：corpus_build 探速 2MB 噪声大、无 sha256 校验、无 provenance 清单（manifest 只记大小）；wet 污染语料已弃用但脚本仍在（应标记废弃/删除）。
11. **许可合规未记录**：Qwen 权重（qwen-community-1.0）提取 PLE 嫁接/再分发边界未写成文（个人研究 vs 发布差异）。
12. **占名未执行**：crates.io/PyPI `engramdb` 审计当日空闲 → 注册窗口应尽快关闭。

### C. 过程债
13. 本轮多个"长任务半路发现慢路由/编码错误"（curl 代理参数、exfat 目录幻觉、np.save 而非 raw、单序列 encode 爆慢）——根因：**先跑再改 vs 先探剂量**。→ 制度化：每个新 I/O/大批次路径先 10MB 级剂量探针，任何 >60s 的任务必带进度条（已部分落实）。

---

## 3. 借鉴矩阵（分层、不相冲突）

原则：**取"方法与形态"，不取"实现与主键"**；凡与"精确为主键的静态内存表"语义冲突的（ANN/PQ、随机 KV 主键）一律不取。

| 层 | 借鉴对象 | 取什么 | 明确不取 | 为什么不相冲突 |
|---|---|---|---|---|
| 存储形态 | **DuckDB** | 嵌入式、无守护、目录即库、manifest 原子换签、Arrow IPC 零拷贝输出 | 列式/扫描查询引擎 | 我们=行定长点查；共享"被嵌入性"形态而非存储引擎实现 |
| 映射/寻址 | **Cassandra/Bigtable** | "热度编入键前缀"的有序分区思想（=我们的频率分层实现观） | LSM 写路径/副本/集群 | 我们用直接寻址数组为主键；热度仅作缓存分层与索引 sidecar |
| 段生命周期 | **Milvus** | 不可变段 + seal/flush/compact + 快照切换 | 向量图、过滤查询 | 只为"表构建+可选增量"借用生命周期模式 |
| SSD 优先读 | **DiskANN** | 冷数据顺序化、滑窗读取、"中心驻留 RAM"的等价物（热集合 + badge 滑窗） | Vamana/图/HNSW | 我们无近邻语义；只取"把随机 IO 收敛成窗口"的经验 |
| IO 工程 | **SGLang #36567** | 常驻 io_uring、页对齐、有界提交、GIL-free、双缓冲 | UVA kernel（引擎内） | 我们只做存储数据面；io_uring 属共需基础设施，双向不重叠 |
| 预取调度 | **vLLM #54129** | 去重/排序/合段 + posix_fadvise(WILLNEED) 批量预读 | CUDA graph 分段策略 | 他们绑定引擎；我们产出"计划"供任何消费方 |
| 启动/热策略 | **llama.cpp** | mmap+MADV_RANDOM、`warm_table` 顺序预热、"实测数字文化"（4.75M gathers 零同页） | 被动 fault 路径（13.1 faults/token 反面教材） | 我们主动预取替代被动 fault；只借鉴其"测量方法学" |
| 键生成 | **DeepSeek 论文 + transformers 官方** | 精确复刻 + golden 对拍 + 可验证性方法 | 官方自身不落地存储 | 键语义是"事实标准"，必须逐位一致（P0 已闭环） |
| 指标文化 | **Qdrant/向量库遥测** | 分档命中率、段驻留率、字节/内存放大率曝光 | — | 通用运维哲学 |
| 生态关系 | **社区路径** | 上游贡献策略：SGLang Rust reader / vLLM PLE-layer 后端 / llama.cpp gather 后端 | fork | 明确分工：他们强"引擎内融合"，我们强"存储/布局/视图/服务"——P4 已证明两者是互补而非竞争 |

**一句话收敛**："同域项目（SGLang/vLLM/llama.cpp）教我们**测量与工程纪律**；数据库项目（DuckDB/Milvus/Cassandra/DiskANN）教我们**形态与布局**；Qwen/DeepSeek 教我们**语义精确**。"三者拼成的正是"数据库化的 n-gram 记忆表"。

---

## 4. 开发计划（重排：Phase 0-6，每 Phase 一个可交付 + 门禁）

| Phase | 内容 | 门禁（gate） | 依赖 |
|---|---|---|---|
| **P0 矫正**（已完成，本文即产出） | design §7 实证化 + Zipf 修正；roadmap 落地 | 无（文档） | — |
| **P1 占位注册 + CI 骨架** | crates.io/PyPI `engramdb` 占名；`cargo test`+`bench-gate` 脚本 + GitHub Actions（macOS job）；bin 布局重构（`engramdb` 主 CLI 收编 build/index/warm/verify，probe 归 `engramdb-bench`） | 全部 benchmark 二进制收敛；gate 脚本绿 | P0 |
| **P2 存储真身（M1.5-A）** | IO backend trait 双实现（Linux io_uring 门禁用）；PrefetchPlanner→ring→ordered-gather 单链化；bitwise 测试入 cargo test；`warm`/`index` 命令端到端化 | P1 真表回归（gather ≥1M 行/s、bit-exact）在 Linux 复测通过 | P1 |
| **P3 视图工程（M1.5-B）** | Store-P 视图构建器（真表 51GB→视图；槽位选型：4KB pad vs 2.56KB 实测后定）；P4 自动化为固定参数 gate（`p4view bench` 固定 seeds → probes CSV 基线） | P4-B 视图路径 ≥4M 等效行/s 保持；回归 CSV 入库 | P2 |
| **P4 绑定+端到端（M2）** | PyO3 包 `engramdb`；engram-peft interop 例；qwen PLE adapter 例；**P4b 端到端 decode 模拟**（llama.cpp 小模型 + 视图 gather 注入） | 50/100 tok/s 实机曲线出炉；interop 例可复现（位级一致审计） | P3 |
| **P5 训练管线（M3）** | Store-P 段式 DataLoader（Python 侧）+ agent workload 注入仿真（semianalysis 时间轴/token 分布做负载频谱）；P3 模拟器用真分布校准 | 训练流 ≥100K tok/s 带宽口径实测；吞吐-缓存曲线 v2 | P4 |
| **P6 服务化（M4）** | Arrow IPC server、多表、stats 遥测 | P5 复测：embedded vs server 开销 ≤2%（≤32KB 批往返） | P5 |
| **P7（可选/上游）** | SGLang/vLLM/llama.cpp 存储后端贡献；生产 GPU 验收（≤5%） | 上游合入 + Linux GPU A/B | 按社区采纳度 |

**节奏纪律**：每 Phase 结束后 30 分钟内更新 roadmap/design §7（数据背书）；任何"假设"进文档必须自带可复现命令（`probes/` 引用）；超过 60s 的任务强制进度条（工具已就位）。

---

## 5. 稳定性与合规机制

- **回归基准包**：`engramdb-bench` 内部固定 seeds/尺寸（P1=65,536 keys；P4=100K grams/4K 槽）产出 CSV 基线，与 `probes/baseline_v{}.csv` 对比（吞吐、放大、延迟百分位、bit-exact）。
- **bit-exact 门禁**：`bitwise_check.py`（4096 rows default）进 CI；keygen golden 回归已入门禁。
- **数据可追溯**：语料 manifest（来源/大小/许可证）、raw sha256 记录；`corpus_build.py` 加校验与环境敏感路由参数记录（删除前先备份采集命令）。
- **合规文档**：单独 `docs/licenses.md` 简述：qwen-community-1.0（提取 PLE 权重与研究使用边界）、DeepSeek Apache-2.0 参照、`trace-commons` CC-BY-4.0（署名要求）、`semianalysis` Apache-2.0、语料三源许可一览。**发布任何产物前必须过此清单**。
- **异常处置准则**（本轮教训制度化）：批量任务 3 分钟无进度即可疑→栈采样确诊→预期外 >5×则停用换法（如 numpy bincount→Rust HashMap 案例）；路由/环境变量错误用剂量探针先验证再全量。
