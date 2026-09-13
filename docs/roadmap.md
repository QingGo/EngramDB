# EngramDB 战略路线图（Roadmap）

> 作者视角：工程负责人复盘。基于截至 2026-08-29 的本轮 session（P0-P4 全部探针 + 真实语料/真实权重/真表基准）。
> 配套文档：`design.md`（设计）、`engram-specs.md`（结构与证据）、`probes/*`（原始数据）。

---

## 0. 这本文件的读法（Session 44 加，先读这一节）

**这本 roadmap 是「复盘史」，不是活账。** 它有 4600 行、约 40 代「本轮复盘」，
每一代都带自己的计划与复选框。截至 Session 44，文件里共 **217 个未关 / 63 个已关**的复选框。

**那 217 个数字曾经被当成债务指标用，那是错的，Session 44 正式废止它。** 理由不是「数字难看」，
而是**这个指标在结构上只会往负走**：每一代复盘都会新增自己的计划，而旧一代的计划
在写下它的时候就已经被下一代取代 —— 于是「净关闭率」衡量的其实是
**「有多少代已经死掉的计划还没被删」**，而不是「有多少该做的事还没做」。
**一个只可能变负的指标不是指标。** 与此对照，§29.4 定的那条硬约束
（「净关闭率 > 0」）在 Sessions 40 → 44 之间从未达成过，而同期真实进展是：
子条件 4 从 ❌ 到 ✅、判据从 500 µs 换成可判定的 `L*`、真实 PLE 几何从「未验证」到索引级确证。

### 唯一的活账（两份，互为镜像）

| 账本 | 位置 | 管什么 |
|---|---|---|
| **验收目标** | `README.md` §6.1 + §6.1 的六条子条件表 | 「算不算达标」 |
| **优先级** | 本文 **§36.5**（P0 / P1 / P1.5 / P2 / P3） | 「下一步做什么」 |

**规则**：任何仍然必需的工作，必须在上面两处之一出现。**只出现在某代复盘复选框里的条目，
视为该代的历史记录，不计入债务。** 反过来，如果某件事仍然必需却没进活账，那是活账的缺陷，
应当**提升**它（见下方「本次提升」），而不是继续让它躺在旧章节里。

### 本次提升（Session 44 从旧复盘里捞出的、仍然必需的三项）

这三项在旧章节里躺着，但**都不在活账里**，而且都影响关键路径，所以提升进 §36.5：

1. **成本分解仪器**（原 Phase 0：「增加 `--profile-embedding`，分离 EngramDB fetch /
   Python convert / transformer compute」）。Session 43 只能用
   `read_us` vs `break_us` vs 整步差值来手工分解，噪声 ±0.2 ms —— 正是缺这个仪器。
   → §36.5 **P1.5** 的前置。
2. **LRU 命中率 / rowid 重复率**（原 Phase 0）。**这一项比看上去重要**：
   如果真实语料的 rowid 有显著重复，页缓存就会帮上忙，那么 Session 43 的
   **冷读最坏情况是过于悲观的**，`t_read` 与 `L*` 都要跟着改。
   → §36.5 **P2** 的前置（在把 `L*` 当结论之前必须先知道访问分布）。
3. **native rowid + gather + dequant 的边界**（原 Phase C）。README 只声称
   rowid 四路径一致；`gather + dequant` 是否已在 native 侧、Python convert 占多少，
   **从未测过** —— 而 Session 43 测得「H2D + add ≈ 118 µs」暗示 convert 这一侧不是零。
   → §36.5 **P2**。

### 已明确退役（不是删除，是冻结）

Track 1–5 / Phase 0–6 / Phase A-C / Phase R1-R5 / V0 这些世代里的**产品广度**条目
（Arrow IPC、Unix socket、认证限流、Dockerfile 固化、三仓库版本同点收编、
llama.cpp serving A/B、Arrow 自动消费等）**由 §36.5 P3 统一冻结**。
它们的复选框保留为「当时想做什么」的记录，**不再是待办**。
历史轮次的长文摘要已移入 `docs/archive/`。

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
7. **`engramdb-cli` 命名与内容混乱**：真正的 CLI 在 `engramdb`（原 engramdb-cli，Phase1 收敛），但 p4view/p3sim/p2rowid 落在 `engramdb-bench`；
   探针与产品命令不分家 → 债：bin 布局重构（`engramdb` 主命令 + `engramdb-bench` 只含 probe）。
8. **视图（Store-P）真表构建器缺失**：P4 只在 100K grams 规模验证；320M 真表视图未建（也未定槽位选型：4KB pad 1.6× vs 2560B 跨页）。
9. **回归/CI 缺失**：bitwise 一致性只有一次性 CLI 验证；golden 只在 keygen 单测；无 bench 门禁。
10. **下载/工具链语义债**：corpus_build 探速 2MB 噪声大、无 sha256 校验、无 provenance 清单（manifest 只记大小）；wet 污染语料已弃用但脚本仍在（应标记废弃/删除）。
11. **许可合规未记录**：Qwen 权重（qwen-community-1.0）提取 PLE 嫁接/再分发边界未写成文（个人研究 vs 发布差异）。
12. **占名待 token**：crates.io/PyPI `engramdb` 审计空闲；发布准备已完成（metadata/LICENSE/顺序=keygen→core→io→engramdb），待用户提供 crates.io/PyPI token 发布。

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
| **P1 占位注册 + CI 骨架** | 占名（crates.io/PyPI `engramdb`，需用户 token）；`gate.sh`+GitHub Actions（双 OS job）；bin 布局重构（已完成：`crates/engramdb` 主 CLI + `engramdb-bench` probe bins） | 全部 benchmark 二进制收敛；gate 绿 ✅ | P0 |
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

---

## 6. 第二轮复盘（2026-08-30：发布闭环 + Phase 1/2）

### 6.1 终极目标复核（不变，结构差异显化）

北极星仍是"确定性记忆表（Engram/PLE）的磁盘优先基础设施——DuckDB 之于分析数据库"。
本轮确认的**三层资产结构**（缺一不可）：
1. **性能层**：存储/索引/预取（P1/P3/P4 证据在手）——最硬
2. **绑定层**：crates.io 四 crate + PyPI `engramdb-python`(import=`engramdb`) + 四平台 Release 二进制 —— **本轮已闭环**（0.1.3 全链验证）
3. **科学层**：P0-P4 + P2 统计的"断言-证据"库 + roadmap/design/session-log 同址——成立

真正仍缺席的：**性能契约的端到端实机（CPU 50/100 tok/s、GPU ≤5%）** = 项目唯一的"概念验证缺口"。

### 6.2 本轮新技术债

| # | 债 | 处置 |
|---|---|---|
| N1 | **P4b 端到端 decode 未做**（性能契约悬空） | Phase 4 前置（当前最高优先级业务面） |
| N2 | **Linux 无门禁**：io_uring 只有 TODO、GPU 路径不可测、429 之外的 release 验证都在 mac | 建议租借小 Linux 云主机（~30-50 元/月）或用户已有机器——**N2 解锁 N1/N3** |
| N3 | **release 无 preflight 门禁**：bump.sh 提示"请跑 test"但流程靠自觉；应让发布前必须过 gate | release.yml 增加 `preflight` job（fmt/clippy/test）gate 发布 job（needs） |
| N4 | crates.io token 为全权限长存（本地 + CI secret 双份） | 待 crates.io trusted publishing（OIDC）正式可用后降级：生成"仅本仓库"token；CI 用 OIDC 同 PyPI（Phase 3.5 实验） |
| N5 | 文档一致性：design §9 里程碑未反映 Phase1/2 状态；probes/gate 规划搬迁 | 随 Phase 2b 收尾更新 |
| N6 | PyPI 相似名是"妥协名"（engramdb-python），长期需向 PyPI 提相似名豁免申请拿回 `engramdb` | 0.2.0 发布窗口期提交申请（材料：repo + 发布日期） |

### 6.3 借鉴增量（本轮）

| 来源 | 借鉴 | 状态 |
|---|---|---|
| **PyPA trusted publishing** | 零-token、事件绑定 release | ✅ 已落地（publish.yml OIDC） |
| **crates.io trusted publishing**（2025 后开放） | 与 PyPI 对称：cargo publish 走 OIDC 无需 token | ⏳ 确认细节（libs.rs/官方文档）并入 Phase 3.5 |
| **版本/发布工程**（cargo ecosystem 惯例） | semver 纪律 patch/minor/major 不越级；preflight 门禁与发布独立 job；tag 永远指向发布 commit | ✅ bump.sh + N3 规划 |
| 数据工程健康度 | **GHA 的 macOS 世代纪要**（runner 退役节奏）→ 产品发布矩阵也须设"平台生命周期" | ✅ macos-15-intel 矩阵已改；注释保留出处 |
| 测量文化小补 | P3 教训再确认：**模拟器参数必须用真实统计校准**（agent workload stats 已是真分布） | Phase 3 re-calibrate |

### 6.4 计划重排（v2.1，改动处加粗）

- **P2b（近期本机）**：CLI 端到端（warm/bench-real 接 agent_workload_stats 真指令序列）+ **CLI 集成测试入门禁**；design §9 状态同步（N5）
- **P2c（需决策）**：小 Linux 门禁环境（租/自有）→ 解锁 io_uring 后端实测（**N2 收敛点**）
- **P3（视图）**：P4 自动化 gate + Store-P 真表构建器 + 槽位选型——按 P4 已定结论推进
- **P4（M2 关键）**：PyO3 绑定 + engram-peft interop + **P4b 端到端 decode 实测（50/100 tok/s 曲线）**→ N1 收敛
- **P4a 发布增强**：crates.io OIDC 实验（N4）、PyPI 相似名申请（N6）
- P5/P6/P7 如前不变；每条出口 gate + 文档同步照旧

---

## 7. 第三轮复盘（2026-08-30 深夜：P4 深化 + 延迟首测 + 全表视图）

### 7.1 终极目标复核（不变 + 本轮把"差距"量化为三段）

北极星不变。本轮最大的价值是**把"端到端还差什么"从概念变为清单**（存面验收档案见 design §7.0 / probes/）：

| 场景 | 存储面（已有真值） | 应用面（缺口） |
|---|---|---|
| A 预训练批量 | 1.44M 行/s（warm）；带宽预算 12.8KB/tok ≪ 0.15GB/s@10Ktok/s（无压力） | Python 段式 DataLoader（P5/M3）、sweep 未接 |
| B 推理在线 | 视图 4.50M 行/s（warm）；**延迟 p99 ≈ 12μs**（比 10ms/token 低 3 个量级） | PyO3 + P4b decode 曲线 + 引擎接入（P4/M2）；单条冷延迟（Linux O_DIRECT） |

→ **两场景的"存储承诺"均已关闭；剩余的是绑定/接入面**（这正好是 roadmap 原设计的 M2 内容，未偏离）。

### 7.2 本轮新技术债（T 序列，沿用 N1-N6）

| # | 债 | 现状证据 | 处置 |
|---|---|---|---|
| T1 | **视图构建在探针层**：p4view build/bench/lat 都在 engramdb-bench bin，产品级 API/CLI 视图命令未成形（PyO3 绑定将无法复用） | `engramdb view` 不存在；构建器 130+ 行在 p4view.rs | **提升为 engramdb-io 公共 API**（ViewBuilder/ViewReader）+ CLI 子命令；探针减薄为 wrapper——P4 前端先行 |
| T2 | **绝对冷口径未闭环**：macOS 页缓存使"冷"半温；只有 warm 档真值 | lat 全表 100K 抽样 p50=0.88μs（实际半命中） | Linux O_DIRECT/io_uring（M2）复测；设计文档注明"B 部署前提 = warm/prefetch" |
| T3 | **SSD 资产脆弱性**：USB 掉载 2 次；构建 51.2GB 无 integrity 检查、无断点续传；keys 放 /tmp 被清 | build 22min 一次性输出，掉盘即重跑 | ① 构建产物 + sha256 manifest（synchronizable 清单）；② `--resume`/n 校验（已有 manifest 基础）；③ keys 默认写仓库 `probes/` 或 SSD（已改 --keys 可选 + manifest 带 n） |
| T4 | **全表 A/B 末证**：51.2GB 尺度上 scatter（A）vs 视图（B）的 5x 只在 200K/100K 验证；全表 A（1TB 读）未完成 | bench sub=2M 超时 | P4 v5 用抽样 A --sub 2M 小步推进；不追全表 A |
| T5 | **max 事件未归类**：2-4ms 簇（1/20K）来源未诊断（OS 换页/盘 sync/SSD GC） | lat 两次 max 2.2ms/4.35ms | 记录为"事件"暂不验；Linux 下对比 O_DIRECT 复测 |
| T6 | **多表/table_id 路径缺位**：实现仅单表；设计有多表 table_id 前缀 | CLI/io 均单表 | M1.5 收口：table_id 参数进入 CLI+Layout 复用（多表 = 目录粒度即可先支持目录分工） |
| T7 | **探针散布**：p4view/p2rowid/p3sim 各自为 bin，无统一基准 CLI | `cargo run -p engramdb-bench --bin xxx` | 合并方向：探针子命令随 P4 前端进 `engramdb probe`；bench bin 只在 M 阶段存在 |

### 7.3 借鉴增量（本轮，与前轮不重复）

| 来源 | 借鉴 | 落地 |
|---|---|---|
| **MLPerf 的"可复现验收"结构** | 每个基准 = 固定输入 + 固定命令 + 判定阈值 + 结果 CSV（可自动 diff） | ✅ 已成型（gate bench + baseline_view.csv + 固定 keys）；下一轮让 lat/CSV 也有判定行 |
| **SQLite integrity_check** | 产品级视图构建后自校验（全量/抽样行值与源表一致 + 修复重跑） | T3 处置②：视图构建加 `--verify`（抽样 1% 行值对拍源表） |
| **fio 的"每档都带直通开关"** | 口径切换显式化：`--direct 0|1` 之类（mac 无 O_DIRECT 时明确标注） | T2 处置：lat/bench 加 `--cache-mode`（warm/cold/auto）；os 支持 O_DIRECT 时自动冷测 |
| **管理产品运维**（发布链我们已借遍）结束——最后一笔 | 发布/构建产物生命周期（资产存在盘上 vs 可重建） | probes/ 存"如何重建"的命令（已逐步写入 notes；T3 全清单化） |

### 7.4 计划重排（v2.2，按"收敛于端到端"排序）

1. **P4 前端（本轮决策先行）**：ViewBuilder/ViewReader 提升进 engramdb-io + `engramdb view build|bench|lat` CLI；probes 减薄为测试——**同时关掉 T1/T7**（也是 PyO3 绑定的前置面）
2. **P4 v5 顺序化实验**：视图槽位按访问序重排（预期全表冷随机 88.7MB/s → >400MB/s 量级）——唯一未兑现的大杠杆（T4 顺带）
3. **P2b 收尾**（小额）：bench-real agent 数值入 baseline CSV + roadmap P2 状态同步
4. **P4b**（需 Linux/GPU 决策）：PyO3 + 50/100 tok/s 端到端曲线 —— 内容与性质同前，前置依赖 P4 前端
5. **P5 v0**（A 场景）：Python DataLoader + 100K tok/s 带宽口径
6. T3/T5/T6 作为平行工程债随以上插缝（T3 下一个视图构建即验）

### 7.5 稳定性原则（本轮重申，构成"稳健前进"的三条）
- **口径纪律**：吞吐/延迟/放大每个探针带环境注明（页缓存态、设备、并发）——基线 CSV 与 notes 已示范
- **可重建性**：一切不常驻仓库的资产（视图/行表/keys）都有明确重建命令；产物带 manifest（参数+耗时+校验）
- **门禁先行**：功能完成 + gate 绿 + 文档同步（每 milestone 三件套收口）

---

## 8. 第四轮复盘（2026-08-30：v0.2.0 发布 + Python/引擎接入面）

### 8.1 本轮目标与结果

- 目标：从“存储性能验证”转向“可发布、可集成的产品面”。
- 结果：
  - PyPI `engramdb-python 0.2.0` 发布成功（abi3 manylinux wheel + sdist）。
  - crates.io 四 crate `0.2.0` 发布成功。
  - engram-peft 磁盘集成进入 `engramdb.integrations`。
  - SGLang 兼容 `engramdb.PageReader.read_pages(fds, offsets)`。
  - vLLM 方向 `engramdb.vllm.PleDiskGather`。
  - ✅ 0.2.1 已发布：Linux `IoUringPageReader`、5 平台 PyPI wheel 矩阵、Python CI 冒烟。
  - ✅ 0.2.2 已发布：`engramdb.sglang.SGLangPageReader`、`engramdb.vllm_plugin.DiskPleEmbedding` 原型。

### 8.2 本轮新增技术债

| # | 债 | 现状 | 处置 |
|---|---|---|---|
| R1 | PyPI 只发布 Linux x86_64 wheel | 0.2.0 仅 manylinux x86_64 + sdist | ✅ 0.2.1 已增加 Linux aarch64 / macOS x86_64+arm64 / Windows wheel 矩阵 |
| R2 | `PageReader` 仍是 pread | 接口对，性能不是 io_uring | ✅ 0.2.1 已实现 Linux `IoUringPageReader`（io_uring batch）；已通过树莓派 + WSL2 实机 smoke（Session 8） |
| R3 | 新 Python API 未进 release | PageReader/PleDiskGather 在 0.2.0 之后 | ✅ 0.2.1 已包含 PageReader / PleDiskGather / IoUringPageReader |
| R4 | 无 Python CI smoke test | 仅在本地验证 | ✅ CI 新增 wheel 安装 + Store/PageReader/PleDiskGather 冒烟 |
| R5 | ~~没有真正接入 vLLM / SGLang 仓库~~ | ✅ 已在真实 vLLM/SGLang 模型类上验证类级 hook（Session 9）；完整 serving/性能仍待做 | 下一步为引擎内 serving 与性能 A/B |
| R6 | 没有目标硬件端到端性能数据 | Intel Mac PyTorch 测试已暂停 | 用用户自维护 wheel / Windows/WSL + 真实 PLE 表验证 |

### 8.3 借鉴增量（本轮）

| 来源 | 借鉴 | 落地 |
|---|---|---|
| SGLang PR #36567 | Rust + PyO3 io_uring reader API、页对齐、有界提交 | `PageReader` 接口对齐，下一步补 io_uring 实现 |
| vLLM blazux patch | dedup、pinned staging、async H2D、CUDA graph splitting、PREWARM | `PleDiskGather` 已落地 dedup/fetch，GPU 侧待接 |
| llama.cpp TENSOR_READ_LAZY | 大 tensor 才 lazy，小模型避免性能退化 | 运维策略：不要无脑磁盘化 |
| vLLM PR #54070 | file-backed mmap、cgroup 限容、MADV_RANDOM、first boot sidecar | 部署文档/缓存策略参考 |

### 8.4 v0.3 计划

1. ✅ 发 `0.2.1`：PageReader / PleDiskGather / 多平台 wheel。
2. ✅ 增加 Python CI smoke。
3. ✅ 实现 `IoUringPageReader`（Linux）。
4. 🔶 准备 SGLang 替换 patch：已有 `SGLangPageReader` 和 patch sketch，待上游源码/实机验证。
5. 🔶 准备 vLLM 插件原型：已有 `DiskPleEmbedding` / `patch_named_embedding`，待接入真实模型验证。
6. 在目标硬件跑真实 PLE 端到端。
7. 性能优化放在端到端验证之后。


---

## 9. 第五轮复盘（2026-08-30 后段：发布工程 + 无源码引擎适配 + README 重写）

### 9.1 本轮目标与结果

- 目标：把 `PageReader` / `PleDiskGather` / SGLang / vLLM 适配层变成可安装、可验证、文档化的产品面。
- 结果：
  - v0.2.1：5 平台 PyPI wheel、Python CI smoke、Linux `IoUringPageReader`。
  - v0.2.2：`engramdb.sglang` / `engramdb.vllm_plugin` 适配原型。
  - v0.2.3：修复 GitHub Release 资产重复上传问题。
  - v0.2.4：增加“不改源码”的类级 PLE patch hook（`install_vllm_ple` / `install_sglang_ple`）。
  - README 重写：使用方式、架构、性能指标、有用/无用优化策略、文档导航。

### 9.2 本轮发现的新技术债

| # | 债 | 现状 | 处置 |
|---|---|---|---|
| V1 | ~~没有在真实 Linux/WSL/树莓派上跑适配层~~ | ✅ 已关闭：树莓派 aarch64 + WSL2 Ubuntu x86_64 均通过 v0.2.4 wheel smoke | 已由 Session 8 验证，保留为发布前回归项 |
| V2 | ~~没有在真实 vLLM/SGLang 模型类上验证 hook~~ | ✅ 已关闭：vLLM 0.28.0 + SGLang 0.5.9 的真实 `Qwen3ForCausalLM` 类均通过类级/实例级 patch 与前向验证（Session 9） | 保留为发布前的引擎 smoke 回归项 |
| V3 | 模型类名/属性名需要用户手动传入 | 缺少自动发现或配置化 | 增加按模型名/配置映射表，或提供 entry-point 注册 |
| V4 | 端到端性能契约仍未闭环 | 🔶 已有真实 vLLM embedding A/B（Session 12/13）：raw disk 235-268μs/call，LRU 后 14-23μs/call；完整 decode 仍缺 | 做完整 serving decode 曲线与 GPU 路径 |
| V5 | 发布工程仍偏人工 | 已修 release-assets，但需要更完整的 preflight/回滚 | 后续接入自动 release 检查 + release notes 资产完整性断言 |
| V6 | 顺序化视图/访问序调度 | ✅ 核心已验证：冷顺序 785.8MB/s vs 冷随机 86.0MB/s ≈ 9.1×（Session 11） | 剩余为大表复测、冷多线程策略与调度器落地 |
| V7 | `DiskPleEmbedding` 无缓存，raw disk 路径延迟偏高 | 🔶 已实现行级 LRU 缓存（Session 13）：重复访问从 235-268μs/call 降到 14-23μs/call；首未命中仍走 raw disk | 后续做 Tier/预热/冷启动预取 |

### 9.3 借鉴增量

| 来源 | 借鉴 | 如何不冲突 |
|---|---|---|
| vLLM/SGLang 的“引擎内融合” | 引擎负责计算与 GPU/CUDA graph；我们只提供存储/PLE 数据面 | 分工：他们管融合，我们管布局/预取/视图 |
| 社区 runtime 插件的“启动前 hook”模式 | 通过类 patch / entry-point 实现零源码接入 | 不侵入上游代码，只在用户启动脚本中执行 |
| PyPA / GitHub Actions 发布工程 | trusted publishing、矩阵构建、glob 不重叠、资产完整性 | 对标常规发布纪律，不改变产品语义 |
| DuckDB 的“目录即库、嵌入式、manifest” | 拿来作为形态契约 | 我们不抄列式执行引擎，只抄被嵌入性和目录形态 |
| llama.cpp TENSOR_READ_LAZY | 大 tensor 才 lazy，小模型避免退化 | 我们由存储层主动预取，不依赖引擎懒加载 |
| SGLang PR #36567 / vLLM PR #54070 | io_uring、页对齐、cgroup、MADV_RANDOM | 已有适配层和调研，待真实环境 A/B |

### 9.4 下一阶段计划（v0.3 修正版）

1. ✅ **真实 Linux 验证**：已完成（Session 8），树莓派 aarch64 + WSL2 x86_64 均通过 v0.2.4 wheel smoke。
2. ✅ **真实引擎接入（功能面）**：已完成（Session 9），vLLM 0.28.0 与 SGLang 0.5.9 的真实 `Qwen3ForCausalLM` 均验证通过；剩余为完整 serving + 性能 A/B。
3. **端到端性能**：CPU 小模型 PLE decode ≥50 tok/s，或 GPU A/B ≤5%。
4. ✅ **顺序化视图 / 访问序调度（核心验证）**：已完成冷盘 A/B（Session 11），1 线程 786 vs 86 MB/s，约 9.1×；下一步做真实大表冷态复测和多线程冷读调度。
5. ✅ **存储产品化（原型）**：多表 `Database`、Arrow helpers、最小 TCP/JSON 服务已落地并 smoke 通过（Session 14）；下一步做 Arrow IPC wire、并发/认证、CLI 收敛。
6. **发布自动化加固**：release-assets 资产断言、全平台 wheel 自动验证、版本/文档同步检查。


---

## 10. 第六轮复盘（2026-08-30 后段：真实引擎验证 + 性能锚点 + 服务/多表/Arrow 原型）

### 10.1 本轮目标与结果

本轮不再停留在“存储面已达标”的结论上，而是把工作推进到：

- 真实 vLLM/SGLang 模型类验证；
- 访问序视图冷盘收益实测；
- PLE 数据面 A/B；
- 多表 / Arrow / 最小服务原型。

结果：

| 成果 | 状态 |
|---|---|
| vLLM 0.28.0 + SGLang 0.5.9 真实 `Qwen3ForCausalLM` 类级/实例级 hook | ✅ |
| `DiskPleEmbedding.forward` 在真实框架模型类上可运行 | ✅ |
| 访问序视图构建、校验、冷盘顺序/随机 A/B | ✅ 786 vs 86 MB/s ≈ 9.1× |
| vLLM 真实类 embedding A/B | ✅ raw disk 235-268μs，LRU 后 14-23μs |
| `DiskPleEmbedding` 行级 LRU 缓存 | ✅ |
| 多表 `Database` | ✅ |
| Arrow helper（Table / IPC bytes） | ✅ |
| 最小 TCP/JSON 服务（含 `fetch_arrow`） | ✅ |

### 10.2 本轮新发现的技术债

| # | 债 | 现状 | 处置 |
|---|---|---|---|
| V8 | PyO3 `Store` 是 `unsendable`，服务端不能跨线程共享 | 服务端目前每请求新开 Store；并发扩展受限 | Rust 侧提供线程安全 store 句柄 / 每线程连接池 |
| V9 | 服务原为 JSON + base64，不是真正二进制 Arrow IPC wire | 已新增 length-prefix binary protocol + `EngramDBClient`，`fetch_raw`/`fetch_arrow` 均可裸字节返回 | 继续做连接复用、认证、线程安全句柄与性能门禁 |
| V10 | GPU 路径被 torch/Pascal 兼容性卡住 | GTX1070 sm_61 与 vLLM/SGLang 当前 torch cu130/cu128 不兼容 | 换 cu121/cu126 老 torch 或走 llama.cpp/CPU 完成 E2E |
| V11 | 小文件冷读多线程反而更慢 | 8t 冷顺序 49MB/s < 1t 786MB/s | 冷读需要顺序流调度，不能盲目并行；大表/真实介质再定 |
| V12 | 多表/服务 Python 原型开始向 Rust 收敛；首批 `tables` + JSON `serve` 已落地 | Rust 仍缺 Arrow IPC、Unix socket、table_id 深度、manifest 完整性校验 | 继续在 Rust 侧补齐服务化与多表产品面 |
| V13 | v0.2.5 已发布 | ✅ PyPI/GitHub Release 已包含新功能 | 后续版本继续走 bump + preflight 流水线 |
| V14 | 首未命中仍走 raw disk，未做预热/Tier | LRU 只解决热重复访问 | 增加 Tier 缓存、PREFETCH/WARM、冷启动调度 |
| V15 | 模型类/属性名仍靠手填 | 只有 `Qwen3ForCausalLM` / `model.embed_tokens` 等已知样例 | 按模型 config 自动发现 PLE 属性，或提供配置映射/entry-point |

### 10.3 借鉴增量（本轮新增）

| 来源 | 借鉴 | 如何不冲突 |
|---|---|---|
| DuckDB / SQLite | 嵌入式、目录即库、manifest、Arrow 输出、每线程连接资源 | 我们只取“嵌入式数据库形态”，不做执行引擎/SQL |
| PyArrow | Arrow Table / IPC stream 作为批次数据契约 | 我们只把它当作存储读取的零拷贝输出协议 |
| Redis/Memcached | LRU/TTL、连接池、线程模型 | 用于 `DiskPleEmbedding` 缓存与服务端资源管理 |
| SGLang #36567 / vLLM #54070 | 页对齐、dedup、pinned staging、async H2D、PREWARM | 继续作为引擎侧参考，但我们保持引擎无关数据面 |
| llama.cpp | TENSOR_READ_LAZY、阈值、硬件 A/B | 只在部署与阈值层面借鉴，不复制推理内核 |
| PyPA / GHA | Trusted Publishing、矩阵构建、preflight、资产完整性 | 用于发布，不改变产品语义 |

### 10.4 下一阶段计划（v0.3→v0.4 修正版）

1. **发布 v0.2.5（已完成）**
   - ✅ PyPI 已发布，macOS/Windows/Linux wheel 构建与安装 smoke 通过；
   - Python wheel smoke 扩展已完成：Database / Arrow / server / LRU，并已加入 CI。

2. **真实 PLE 端到端性能闭环（V4/V10）**
   - ✅ 已获得 CPU 小模型首批端到端 decode 曲线（`scripts/cpu_tiny_decode_ab.py`）；
   - 继续：CPU 完整 serving（vLLM/SGLang）或更高保真大模型 A/B；
   - 其次尝试 GTX1070 可用 torch（cu121/cu126）下的 GPU A/B；
   - 若 GPU 不可行，以 llama.cpp CPU/GPU 路径作为替代验收。

3. **服务化/多表/Arrow 从原型变产品**
   - Rust 侧：首批 `tables` + JSON `serve` 已落地；继续补 table_id、manifest 完整性、Arrow IPC、Unix socket；
   - Python 侧：二进制 Arrow IPC wire 已落地，继续做连接复用、线程安全句柄、认证；
   - 性能门禁：embedded vs server ≤2%（≤32KB 批往返）。

4. **顺序化/冷读调度**
   - 真实大表冷态复测；
   - 自适应“顺序流优先”多线程策略；
   - 与 `StreamingPlanner` / Tier 预取打通。

5. **引擎接入深化**
   - vLLM/SGLang 完整 serving 中启用 PLE disk path；
   - 自动发现模型 PLE 属性；
   - 性能 A/B：功能一致 + 差距 ≤5%。

6. **长期**
   - 上游 patch / 贡献；
   - llama.cpp 文件格式/C ABI 接入；
   - 保持“不修改上游源码”的薄层适配哲学。

## 11. 第七轮复盘（2026-08-30 后段：发布验证 + Rust 首批服务化 + CPU E2E 首曲线）

### 11.1 终极目标再确认

一句话不变：

> 让 Engram/PLE 这类“确定性哈希 n-gram 记忆表”成为任何小模型、训练器、推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

本轮后需要把“接近目标”的判断标准更严格：

| 轴 | 验收口径 |
|---|---|
| 性能 | 真实小模型 + 真实/合成 PLE 表上的端到端 A/B，而不是 tiny toy model 或单独 embedding micro A/B；EngramDB 参与后总 tok/s 差距目标 ≤5% |
| 形态 | 单目录多表、manifest 可校验、嵌入式 + 服务化双形态、Arrow 零拷贝输出、引擎薄接入 |
| 科学 | 每个性能论断都有可复现脚本、冷热/介质/并发口径，并且拒绝“单次抖动即结论” |

### 11.2 本轮新认识与技术债（V16 起）

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V16 | CPU E2E 首曲线来自 tiny toy Qwen3，不是真实 PLE 表/真实服务引擎 | 数字只能做方向锚点，不能作为验收 | 下一轮用真实或大合成 PLE 表 + 更高保真 CPU/llama.cpp 路径 |
| V17 | v0.2.5 tag 之后 master 已有 Rust serve、CPU A/B 脚本等新代码 | 版本与代码开始分叉，容易混淆“哪个版本含什么” | 规划 v0.2.6，在 release 前把新功能纳入并重新 bump |
| V18 | Rust serve 只是 JSON + base64，没有 Arrow IPC/二进制协议 | 仍不是最终产品 wire，且与 Python 二进制服务能力不对齐 | 按“Rust 为核、Python 为薄壳”把二进制/Arrow 迁到 Rust |
| V19 | 服务仍无连接复用、认证、限流、线程安全句柄、性能门禁 | 不能作为生产服务使用 | 参考 Redis 连接模型 + DuckDB 每线程资源，先做每线程 store 池/句柄 |
| V20 | Manifest 只用于布局读取，没有完整性校验（文件大小、shard 数、checksum） | 数据损坏/部分复制时静默错误 | 增加 manifest schema、shard 文件尺寸校验、可选 checksum、`check` 子命令 |
| V21 | CPU A/B 噪声大，缺少固定 seed、固定序列、重复次数阈值、冷热状态 | 数字波动无法形成回归门槛 | 采用 MLPerf 式：固定输入、固定命令、判定阈值、结果 CSV 入库 |
| V22 | DiskPleEmbedding 仍为 Python 层逐 token 拼接，无原生零拷贝/异步 | 热路径虽接近内存，但持锁、GIL、Python 对象开销仍在 | 中期考虑 Rust/PyO3 原生 gather + Arrow 或内存池接口 |
| V23 | GPU 路径仍无可行 torch 构建 | 无法验证 GPU 端 ≤5% 门禁 | 尝试 cu121/cu126 老 torch，或先以 llama.cpp CPU/GPU 作为替代验收 |
| V24 | 模型 PLE 属性仍靠手填 | 换模型/换版本时接入成本高 | 增加 config 自动发现 + 注册表/映射文件 |

### 11.3 借鉴矩阵（本轮聚焦“如何不冲突地推进”）

| 来源 | 借鉴 | 不冲突的原因 |
|---|---|---|
| **DuckDB** | 单目录库、manifest、Arrow 输出、嵌入式+服务双形态 | 我们不做 SQL/执行引擎，只借用“数据库形态”和“可嵌入性” |
| **SQLite** | integrity_check、connection-per-thread、轻量服务化 | 我们不做关系模型；只取“可校验、可嵌入、每线程资源隔离” |
| **Redis/Memcached** | LRU/TTL、连接池、每连接状态、简单命令协议 | 我们不是通用 KV；只取缓存与服务端资源管理 |
| **Milvus** | 不可变段、seal/flush、快照发布 | 我们是静态只读表；只借生命周期与原子发布，不借向量检索 |
| **DiskANN** | 冷数据顺序化、滑窗、热集驻留 | 我们无近邻语义；只借“把随机 IO 收敛为顺序窗口”的经验 |
| **vLLM/SGLang** | dedup、页对齐、pinned staging、PREWARM、io_uring | 我们保持引擎无关数据面；薄 adapter 不修改上游 |
| **llama.cpp** | mmap/MADV_RANDOM、warm table、实测数字文化 | 我们只取部署阈值与测量纪律，不复制其推理内核 |
| **PyArrow** | Arrow Table/IPC 作为批次数据契约 | 我们只用它做存储读取输出，不做查询/执行 |
| **MLPerf** | 固定输入+固定命令+判定阈值+CSV 基线 | 用于把 A/B 从“跑一次”变成“可回归” |
| **fio** | `--cache-mode`、O_DIRECT、介质标注 | 用于把冷热/直通口径显式化，避免假冷/假热 |

### 11.4 下一步开发计划（v0.3 主线）

按“先建立可信性能基线，再产品化，再引擎深化”排序：

1. **v0.2.6 发布准备**
   - 把 Rust serve、CPU A/B 脚本、cache_size=0 修复纳入正式发布；
   - 增加 Rust `serve`/`tables` smoke 进 CI；
   - 保持 Python wheel smoke 全绿。

2. **可信性能门禁**
   - 把 CPU tiny decode A/B 改成固定 seed + 固定序列 + 多次取中位数；
   - 生成 `probes/cpu_decode_baseline.csv`；
   - 建立阈值：raw disk 不得比 memory 慢超过 X%，LRU 不得慢超过 Y%；未达标即回归失败。
   - 下一步尝试真实 PLE 表或大合成 PLE 表，而不是只 toy vocab。

3. **Rust 服务化收敛**
   - Rust `serve` 增加二进制 length-prefix 与 Arrow IPC；
   - 增加 manifest 校验/`engramdb check`；
   - 每线程 store 句柄/连接池，解决 `unsendable`；
   - 性能门禁：embedded vs server ≤2%（≤32KB 批往返）。

4. **冷读与调度**
   - 大表冷态顺序/随机复测；
   - 自适应单/少线程顺序流；
   - 与 `StreamingPlanner` / Tier/预取打通。

5. **引擎真实接入**
   - 优先 CPU 可用路径：llama.cpp 或可运行的 vLLM/SGLang CPU serving；
   - 自动发现 PLE 属性/配置映射；
   - 做端到端 A/B，目标差距 ≤5%。

6. **长期**
   - 上游 patch / C ABI / llama.cpp 文件格式；
   - GPU 路径待兼容 torch 或换硬件后补测；
   - 保持“不修改上游源码”的薄层哲学。

### 11.5 稳定前进的三条纪律

1. **先测量，后优化**：任何“快/慢”结论必须落在可复现脚本和 CSV，不接受单次 run 口述。
2. **收敛到 Rust 核心**：Python 只做薄 adapter/演示；服务、Arrow、manifest、多表逐步由 Rust 承担。
3. **版本和功能同源**：重大功能必须进入正式 tag，避免 master 与已发布版本长期分叉。

## 12. 第八轮复盘（2026-08-30 后段：真实 Qwen3.5-0.8B E2E + Rust 服务化深化 + v0.2.6）

### 12.1 终极目标再确认

终极目标不变：

> 让 Engram/PLE 这类“确定性哈希 n-gram 记忆表”成为任何小模型、训练器、推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

本轮的进展是：从 toy model 走向真实 0.8B 模型；从“功能 hook 可用”走向“真实端到端性能首锚点”；从 Python 服务原型走向 Rust 侧可校验、可二进制的服务雏形。

### 12.2 本轮实际完成

| 项 | 状态 |
|---|---|
| v0.2.5 真实 wheel 验证 | ✅ |
| v0.2.6 发布 | ✅ PyPI + GitHub Release |
| Rust `tables` / `serve` / `check` | ✅ |
| Rust 二进制 length-prefix 服务 | ✅ |
| Rust `view_read` | ✅ |
| CPU tiny decode A/B | ✅ |
| 真实 Qwen3.5-0.8B CPU decode A/B | ✅ 首批 |
| 真实模型软链与 WSL 复制 | ✅ `data/Qwen3.5-0.8B`（gitignore） |

### 12.3 本轮新技术债（V25 起）

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V25 | 真实 Qwen3.5 A/B 使用稀疏零值 store，不是真 PLE 表，也不是 bit-exact | 只能测“磁盘读取路径性能”，不能证明功能等价 | 用真实权重填充 store，或构建真实 PLE 表，增加 bit-exact 对照 |
| V26 | 真实模型 CPU A/B 噪声很大（几轮 memory 2.1–4.0、disk raw 2.4–3.9、LRU 1.5–3.7） | 不能形成可信回归阈值 | 固定 seed/输入、增加序列长度、多次取中位数、落 CSV 基线 |
| V27 | Rust serve 仍无 Arrow IPC、Unix socket、认证、限流、连接池、线程安全句柄、性能门禁 | 服务面只是雏形 | 按 v0.3 产品化继续收敛 |
| V28 | 真实模型在外部盘，WSL 副本在 `/mnt/c`，无自动化准备命令 | 换机器后复现成本高 | 写 `scripts/prep_real_model.sh`，自动软链/复制/校验文件完整 |
| V29 | 0.8B 真实模型仍未在 vLLM/SGLang serving 中验证 | 仍是 transformers 直跑，不是真实服务引擎 A/B | 尝试 vLLM/SGLang 加载 Qwen3.5-0.8B，或 llama.cpp 替代 |
| V30 | 未做真实 PLE 属性注入（当前 patch 的是普通 input embedding，不是 PLE 层） | 性能代表“磁盘 embedding 替换”，不是完整 PLE 语义 | 下一步从模型 config/权重中定位真实 PLE 表 |
| V31 | 真实模型太大，不能进 CI；CI 仍只有合成/极小模型 | 持续回归缺少真实负载 | 固定小规模真实性采样 + 独立 nightly/手动基准 job |
| V32 | v0.2.6 tag 后 master 又加入 view_read、真实模型脚本 | 版本再次领先发布线 | 定期把 master 收编进 v0.2.7，避免长期分叉 |

### 12.4 借鉴矩阵（第八轮增量）

| 来源 | 借鉴 | 不冲突原因 |
|---|---|---|
| **HuggingFace Transformers** | 模型目录/配置加载、架构注册、`from_pretrained` 复现 | 我们不做模型格式，只借“标准模型路径”做真实负载 |
| **vLLM / SGLang** | engine + adapter、PagedAttention、batching | 我们不复制推理内核；只提供存储后端/薄替换 |
| **llama.cpp** | 单二进制、GGUF、CPU/GPU 部署、基准文化 | 只取部署和测量方法论 |
| **DuckDB / SQLite** | 目录即库、manifest、integrity check、每线程资源 | 不取查询引擎/关系模型 |
| **Redis / Memcached** | LRU/TTL、连接管理、协议版本化 | 不取通用 KV 语义 |
| **DiskANN** | 冷数据顺序化、滑窗、热集分层 | 不取 ANN 图结构 |
| **MLPerf / fio** | 固定输入、判定阈值、CSV、cache-mode | 只用于性能门禁与实验口径 |
| **GitHub Actions / PyPA** | 版本 tag、preflight、artifact 管理、Trusted Publishing | 用于发布与回归，不改变产品设计 |

### 12.5 下一阶段开发计划

1. **可信性能基线（最高优先）**
   - 固定 seed、固定输入序列、固定 token 数；
   - `reps>=5`，输出中位数 + p90；
   - 生成 `probes/qwen35_cpu_baseline.csv`、`probes/cpu_tiny_baseline.csv`；
   - 设置门禁：raw 不得比 memory 慢超过 X%，LRU 不得慢超过 Y%。

2. **真实数据面**
   - 把稀疏 store 改为真实权重填充 store；
   - 增加 bit-exact 对照：memory output == disk output；
   - 从 Qwen3.5 权重中定位真实 PLE/Engram 表属性。

3. **Rust 服务产品化**
   - Unix socket；
   - Arrow IPC（或至少零拷贝 raw path）；
   - 每线程 store 句柄/连接池；
   - 认证/限流；
   - embedded vs server 性能门禁 ≤2%。

4. **真实服务引擎**
   - vLLM/SGLang/llama.cpp 加载 Qwen3.5-0.8B；
   - 用 EngramDB 替换 PLE 数据面；
   - 做 serving 级 A/B，目标 ≤5%。

5. **冷读与调度**
   - 大表冷态顺序/随机复测；
   - 自适应顺序流；
   - Tier / 预取打通。

6. **发布与维护**
   - v0.2.7 收编当前 master；
   - 真实模型准备脚本；
   - 保持“数据不进 git，只进代码/脚本/基线”。

### 12.6 稳定前进原则（第八轮强化）

1. **真实数据优先于玩偶数据**：能上真实模型/真实 PLE 就上真实，但必须同时保留可复现小规模 CI。
2. **性能数字必须可回归**：没有固定输入、中位数、CSV 和阈值的数字，只算“观察”，不算“结论”。
3. **Rust 为核，Python 为薄壳**：服务、协议、校验、存储 API 逐步下沉到 Rust。
4. **版本和功能同源**：每次真实功能合并后，尽快收编进下一个版本，避免 master 无限领先。
5. **薄接入，不修改上游**：所有引擎适配保持 plugin/patch 形式，避免 fork。

## 13. 第九轮增量（2026-08-30 后段：可信基线闭环 + 真实权重 bit-exact）

### 13.1 已完成

| 项 | 状态 |
|---|---|
| 固定 seed / eval / reps>=5 / median+p90 | ✅ |
| `probes/cpu_tiny_baseline.csv` | ✅ |
| `probes/qwen35_cpu_baseline.csv`（真实权重 store） | ✅ |
| `scripts/decode_baseline_check.py` 阈值门禁 | ✅ |
| 真实权重填充 store | ✅ |
| Qwen3.5 bit-exact（direct + generation） | ✅ |
| `scripts/prep_real_model.sh` | ✅ |
| `.gitignore` 放行 `probes/*baseline*.csv` | ✅ |

### 13.2 关键数字

- tiny：memory 394.76 tok/s，raw 282.39（39.8% 慢），LRU 244.96（61.2% 慢）。
- Qwen3.5-0.8B 真实权重：
  - memory 4.69 tok/s
  - raw 4.15 tok/s（13.0% 慢）
  - LRU 3.94 tok/s（19.2% 慢）
- bit-exact：`max_abs=0.0`，生成序列完全一致。

### 13.3 后续重点

1. 把 bit-exact 合入 A/B 主流程，让每次跑数同时验证功能。
2. 从 Qwen3.5 权重中定位真实 PLE/Engram 表属性；当前仍替换普通 `embed_tokens`。
3. v0.2.7 发布收编本轮所有内容。
4. Rust 服务产品化继续（Unix socket / Arrow / 连接池 / 认证）。

### 13.4 残留债务

- V25 部分闭合：真实权重 + bit-exact 已做；但仍不是真实 PLE 表语义。
- V26 闭合：可信 CPU 基线已建立；仍需在更多序列长度/输入上积累。
- V28 闭合：真实模型准备脚本已写。
- V29/V30/V31/V32 仍开放。

### 13.5 真实 PLE 自动发现（Session 19 增补）

- 新增 `python/engramdb/ple_discovery.py` 与 `scripts/inspect_ple_attributes.py`。
- 在真正的 Qwen3.8-Flash-Next / Qwen4Exp 模型中发现 PLE 表：
  `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_*.weight`。
- 确认 Qwen3.5-0.8B 不含 PLE，后续真实 PLE 性能验证应使用 Qwen4Exp/Qwen3.8 真模型，而不是 0.8B 玩具。
- V30 状态：已能自动发现真实 PLE 属性；下一步是使用该路径构造 disk-backed PLE adapter。

## 14. 第十轮系统性思考（Session 20：从“能跑”到“可信、可重叠、可服务”）

### 14.1 终极目标再锚定

一句话：

> **让 DeepSeek Engram / Qwen PLE 这类“确定性哈希 n-gram 记忆表”成为任何模型、训练器和推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。**

用户视角的终极价值：

```text
模型不需要把几十 GB 的 PLE 表塞进 RAM/VRAM
训练/推理引擎只需要一个薄 adapter
性能接近内存，且功能 bit-exact
可以嵌入式，也可以服务化
不修改上游源码
```

可验收的终点：

| 指标 | 目标 |
|---|---|
| 真实 PLE 表 bit-exact | ✅ 必须 |
| 真实服务引擎端到端差距 vs 内存 PLE | ≤5% |
| 嵌入式 vs 服务端 | ≤2% |
| CPU/GPU 双路径 | CPU 先行，GPU 等待兼容 |
| 可复现性 | 固定输入 + 中位数 + CSV + 阈值 |
| 形态 | Rust 核心 + Python/引擎薄 adapter + 可选服务 |

明确不做：

- 不做通用向量检索/ANN
- 不做 SQL 执行引擎
- 不做通用 KV 数据库
- 不 fork 修改 vLLM/SGLang/llama.cpp 上游
- 不让 Python 原型成为最终产品核心

### 14.2 本轮/近期会话发现的技术债（V33 起）

| # | 债务 | 影响 | 处置 |
|---|---|---|---|
| V33 | Qwen3.5-0.8B 不是真实 PLE 模型，只是普通 `embed_tokens` | 当前 bit-exact 证明的是“磁盘 embedding 替换”，不是真实 PLE 语义 | 已定位真实 PLE 在 Qwen4Exp/Qwen3.8：`layers.1.ple.ple_embedding.ngram_embedding.shard_*.weight`；下一步做真实 PLE adapter |
| V34 | LRU 没有命中率指标；单序列 decode 无复用，LRU 反而比 raw 慢 | 无法判断 cache 是否有价值；可能引入无谓 overhead | 增加 hit rate / 每 token rowid 重复率统计；只有命中率有证据时才启用 |
| V35 | DiskPleEmbedding 是同步 Python 薄层，无原生 gather、无异步预取 | memory vs disk 差距不全是磁盘 I/O，还包含 Python adapter 和关键路径同步开销 | 先做分阶段计时分离 fetch/adapter/compute；再下沉 Rust/PyO3 原生 gather + 预取 |
| V36 | 基线仍是单序列、短 token、固定顺序 memory→raw→LRU、5 reps | 有 order bias 和 WSL 噪声，不能形成稳定阈值 | 多序列、多 seed、随机化顺序、更长生成、多次中位数；CSV 带机器元数据 |
| V37 | bit-exact 是独立脚本，未合入 A/B 主流程 | 以后跑性能可能忘记验证功能 | 让 A/B 默认附带 bit-exact 检查 |
| V38 | 真实 PLE 表尚未接入任何真实模型 E2E | 最重要的目标路径还没有闭环 | 用 Qwen4Exp 的真实 PLE shard + keygen 做 store 级 bit-exact，再做 adapter 级 |
| V39 | v0.2.6 后 master 已累计可信基线、PLE discovery、bit-exact 等 | 版本再次领先，发布线分叉 | 尽快 v0.2.7 收编；之后小步发布 |
| V40 | Rust 服务仍无 Arrow IPC、Unix socket、认证、限流、连接池、线程安全句柄 | 服务面仍是原型 | 保持 v0.3 主线，但现在优先真实数据面与性能路径 |
| V41 | 基线只在 WSL 单机产生，没有跨平台和介质元数据 | 数字不能跨机器解释 | 在 CSV 中加入 `host/os/disk/store_file` 等列；可复跑 macOS/Linux |
| V42 | 没有精细 instrumentation，无法定位 memory/raw 差距来源 | 容易把 Python adapter 开销误判为磁盘慢 | 增加 `--profile-embedding` 输出 fetch/convert/compute 分段 |
| V43 | 真实模型不能进 CI，也没有 nightly real-model job | 真实回归只能手动 | 建独立 nightly/手动 job，CI 继续跑合成/小模型 |

### 14.3 借鉴矩阵（第十轮增量）

| 来源 | 借什么 | 不借什么 | 对应 EngramDB 目标 |
|---|---|---|---|
| **DuckDB / SQLite** | 目录即库、manifest、integrity check、每线程资源、嵌入式优先 | 不借 SQL/关系模型 | 存储库形态、可校验、可嵌入 |
| **vLLM / SGLang** | batching、PagedAttention 的确定性 rowid 提前量、engine adapter 模式 | 不借推理内核/调度实现 | 实现“预取/重叠”和“薄插入” |
| **llama.cpp** | CPU 优先、单二进制、GGUF/C ABI、严谨基准文化 | 不重写推理/量化 | 快速 CPU 验证、C ABI 接入 |
| **DiskANN** | 冷数据顺序化、滑窗、tier 分层、预取 | 不借近邻图/向量检索 | 解决 PLE 冷读与随机 IO 问题 |
| **Redis / Memcached** | LRU/TTL、连接协议、连接池、认证/限流 | 不借通用 KV 语义 | 服务化资源管理与缓存治理 |
| **MLPerf / fio** | 固定输入、固定命令、中位数、阈值、cache-mode、CSV | 不做 benchmark-only 产品 | 让性能结论可回归 |
| **PyArrow / Arrow IPC** | 数据契约、零拷贝批次、跨语言边界 | 不借查询/执行 | 服务与引擎之间的高效数据面 |
| **RocksDB / FoundationDB** | 不可变段、checksum、文件版本、原子发布 | 不借 LSM/事务复杂度 | 大表静态发布与完整性 |
| **HuggingFace Transformers** | 标准模型目录、config/权重发现、from_pretrained 复现 | 不借模型格式定义 | 自动发现真实 PLE 属性 |

关键不冲突原则：

- 我们只做“确定性 n-gram 表”的存储和访问，不越界到查询、检索、推理。
- 所有引擎适配都是薄 patch/adapter，不 copy 或 fork 上游。
- 所有借鉴都必须落到“可复现实验”或“可校验代码”，不能只停留在概念。

### 14.4 开发计划（分阶段、带验收）

#### Phase 0：测量硬化（立即，1–2 个迭代）
- [ ] 多序列、多 seed、随机化 memory/raw/lru 顺序
- [ ] `reps>=7`；输出 median/p90/CI
- [ ] 增加 `--profile-embedding`，分离 EngramDB fetch / Python convert / transformer compute
- [ ] LRU 增加 hit rate / rowid 重复率
- [ ] bit-exact 合入 A/B 主流程
- [ ] CSV 增加 host/os/disk/seed/seq 等元数据
- [ ] v0.2.7 发布，收编当前 master

**退出标准**：
- 同一个数字在两次独立 run 中不会因顺序或噪声颠倒结论。
- 能明确回答：memory vs raw 的差距里，多少是磁盘 I/O，多少是 Python adapter。

#### Phase 1：真实 PLE 数据面（核心）
- [ ] 用 `ple_discovery` + 真实 Qwen4Exp PLE shard 构造 EngramDB store
- [ ] Store fetch 与 safetensors 原始 shard 做 bit-exact
- [ ] 用 keygen rowid 抽样验证真实 PLE 行读取正确
- [ ] 实现 `patch_real_ple`：自动找到 `model.language_model.layers.*.ple` 并替换
- [ ] 在真实 PLE 小批量前向/生成上做功能 A/B

**退出标准**：
- 真实 PLE 行读 bit-exact。
- 至少一个真实 PLE 层能用 EngramDB 数据面完成前向，输出与内存一致。

#### Phase 2：性能架构（决定能否达到 ≤5%）
- [ ] Rust/PyO3 原生 `DiskPleEmbedding`，去掉 Python 热路径
- [ ] 根据 rowid 确定性实现“下一 token 预取”，与当前 transformer 计算重叠
- [ ] 批量 gather：一次 fetch 多 token/多请求所需行
- [ ] LRU/Tier 只在高命中率场景启用
- [ ] 冷态真实 PLE 表 A/B

**退出标准**：
- 端到端差距 ≤5%（真实 PLE 或真实模型场景）。
- 在无复用场景下，LRU 不劣于 raw。
- 预取确实把磁盘延迟从关键路径移走。

#### Phase 3：真实服务引擎 A/B
- [ ] vLLM / SGLang / llama.cpp 加载含真实 PLE 的模型
- [ ] EngramDB 替换 PLE 数据面
- [ ] serving 级 A/B，目标 ≤5%
- [ ] CPU 先行；GPU 等 torch/驱动兼容后补

#### Phase 4：Rust 服务产品化
- [ ] Unix socket
- [ ] Arrow IPC / 零拷贝 raw
- [ ] 每线程 store 句柄 / 连接池
- [ ] 认证 / 限流 / 协议版本
- [ ] embedded vs server ≤2%
- [ ] manifest checksum / 原子发布

#### Phase 5：长期维护
- [ ] 真实模型 nightly/manual job
- [ ] 跨机器基线 + 环境元数据
- [ ] 自动发现 + 注册表
- [ ] C ABI / GGUF 方向探索
- [ ] 保持“数据不进 git，代码/脚本/基线进 git”

### 14.5 稳定前进的五条纪律（第十轮强化）

1. **一个结论 = 一个可复现脚本 + 一个 CSV + 一个阈值**
   没有固定输入、中位数、CSV 的性能数字只是观察，不是结论。

2. **任何 cache 必须先有命中率证据**
   没有命中率，就没有资格谈 LRU/Tier 收益。

3. **真实 PLE 优先于 toy model**
   Qwen3.5-0.8B 只用于打通流程；真正的验收必须落在 Qwen4Exp/Qwen3.8 的真实 PLE 表上。

4. **Rust 为核，Python 只做薄 shell**
   Python 原型用来验证语义和快速实验，热路径最终必须下沉 Rust/PyO3。

5. **版本和功能同源，发布要小步**
   每完成一个真实闭环就尽快 bump/tag，避免 master 长期领先于发布版。

### 14.6 Session 20 增补：真实 PLE Store 位级验证

- 新增 `scripts/real_ple_bit_exact.py`，对真实 128-shard PLE 原始行做 Store 位级对照。
- 发现并修复 `gather_pp` 多分片偏移 bug：
  - 原实现用全局 rowid * row_bytes 作为文件内偏移；
  - 改为用 shard 内局部行偏移；
  - 修复 `gather_plan` 退化路径同类问题；
  - 增加回归测试。
- 验证：100 个跨 shard 随机 rowid，SHA-256 完全一致，`PLE_STORE_BIT_EXACT_PASS`。
- 这是真实 PLE 数据面闭环的第一步。

### 14.7 Session 20 增补：真实 Qwen4Exp PLE layer bit-exact

- 新增 `python/engramdb/ple_adapter.py`（`DiskPleNGramEmbedding`）：
  - 磁盘 PLE n-gram embedding，FP8 行 + weight_scale 反量化；
  - 支持顺序 decode 的最小历史状态。
- 新增 `scripts/ple_layer_bit_exact.py`：
  - 只加载 PLE 层小型权重，不加载完整大模型；
  - 真实 PLE 层 forward 与 EngramDB disk path 位级一致。
- 验证：
  ```text
  PLE_LAYER_BIT_EXACT_PASS
  max_abs=0.0
  ```
- 完整模型级 E2E：仍受整模型内存/资产限制，属于后续真实机器任务。

## 15. 第十一轮系统性思考（Session 21：真实 PLE 数据面第一里程碑）

### 15.1 终极目标再锚定

不变：

> **让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何模型、训练器、推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。**

本轮后的位置：

| 层 | 状态 |
|---|---|
| 真实 PLE Store 位级读取 | ✅ 已闭环 |
| PLE rowid 生成 | ✅ 与官方数学对齐 |
| PLE 层前向 bit-exact | ✅ 已闭环（自实现 PLE forward + 真实权重） |
| 磁盘 PLE adapter | ✅ 已可复用 |
| 完整模型加载时替换 PLE | ⚠️ 需要 custom loader |
| 完整模型 E2E A/B | ❌ 受环境/内存限制 |
| 服务引擎级 A/B | ❌ 未做 |

### 15.2 本轮完成与关键收获

- 发现并修复 `gather_pp` 多分片偏移 bug：
  - 此前只对单分片正确；
  - 真实 128-shard PLE 表会读到错误行；
  - 这是“看似能跑，实际错误”的典型数据面隐患。
- 新增 `DiskPleNGramEmbedding`：
  - 磁盘 PLE n-gram embedding；
  - 确定性 rowid；
  - FP8 + weight_scale 反量化；
  - 顺序 decode 最小历史。
- `scripts/ple_layer_bit_exact.py`：
  - 不加载完整大模型；
  - 只加载 PLE 层小权重；
  - 全 PLE 层 forward bit-exact。
- 结论：
  - 存储层、adaptor 层、PLE 数学层已经没有功能缺口；
  - 缺口转移到“完整模型加载/替换时机”和“真实算力/内存环境”。

### 15.3 本轮新技术债（V44 起）

| # | 债务 | 影响 | 处置 |
|---|---|---|---|
| V44 | 完整 Qwen4Exp 模型仍不能加载进内存或跳过 ngram_embedding 权重后替换 PLE | 无法做完整模型 E2E | 写 custom loader / from_pretrained 前置 patch；跳过 `ngram_embedding.shard_*` 权重 |
| V45 | `DiskPleNGramEmbedding` 只在自实现 PLE forward 中验证，未在官方 `Qwen4ExpTextPLELayer` 中验证 | 可能与官方 cache/量化/特殊路径有差异 | 在可加载完整模型的环境里用官方类实例替换并对比 |
| V46 | adapter 使用 Python 内部 token history，未接入 Transformers `Cache` | 流式 decode 与 MTP/多段输入可能不一致 | 接入官方 `past_key_values` conv_state 或提供等价引擎状态 |
| V47 | 尚未有真实 PLE 模型性能数据 | 无法判断磁盘 PLE 是否达到服务门槛 | 准备 big-memory 环境或引擎级替换后跑 A/B |
| V48 | 当前 PLE layer bit-exact 仅覆盖单段、冷路径 | 未覆盖跨段、EOS 重置、多 batch、MTP 等边界 | 扩展测试矩阵 |
| V49 | 完整模型资产不在可运行环境 | 开发和验证被环境卡住 | 寻找大内存机器/云主机，或走 llama.cpp/服务端路径 |
| V50 | Rust 核心尚无 PLE adapter 热路径 | Python 版只验证语义，不满足性能目标 | 后续把 rowid + gather + dequant 下沉 Rust/PyO3 |
| V51 | 服务化/发布仍然滞后 | 产品面未闭环 | 保持 v0.3 计划，但当前优先打通真实 E2E 路径 |

### 15.4 借鉴矩阵（第十一轮增量）

| 来源 | 借什么 | 不借什么 | 为什么对我们有用 |
|---|---|---|---|
| **HuggingFace Transformers** | `from_pretrained` 前置/后置 hook、state_dict 自定义加载、skip 大权重 | 不重写模型定义 | 解决“完整模型加载时不分配 ngram_embedding” |
| **vLLM / SGLang / llama.cpp** | 模型加载时替换 embedding 表、CPU offload、内存映射 | 不复制推理内核 | 把真实 PLE 接进可用推理路径 |
| **DuckDB / SQLite** | 嵌入式优先、manifest、integrity、连接模型 | 不借 SQL | 存储库产品形态 |
| **DiskANN / Memcached** | 冷热分层、LRU 命中率、预取窗口 | 不借 ANN/KV | 让磁盘 PLE 在真实推理中有性能意义 |
| **MLPerf / fio** | 固定输入、阈值、CSV、cache mode | 只做 benchmark | 所有性能结论可回归 |
| **Arrow / Rust** | 零拷贝批次、原生热路径 | 不借查询引擎 | 最终将 Python adapter 下沉 Rust |
| **RocksDB / FoundationDB** | checksum、原子发布、文件版本 | 不借 LSM | 大表可靠发布 |

关键不冲突：

- 我们不做模型训练/推理，只做 PLE 数据面。
- 我们不改上游源码，用 loader hook / adapter。
- 所有“快”的结论必须来自真实 PLE + 可复现基准。

### 15.5 下一步开发计划

#### Phase A：让 adapter 能被完整模型真正使用（最高优先）
- [ ] 写 `scripts/qwen4_ple_custom_loader.py`：
  - 加载完整模型所有非 PLE 权重；
  - 跳过 `ngram_embedding.shard_*.weight`；
  - 构造模型后用 `DiskPleNGramEmbedding` 替换真实 PLE 层。
- [ ] 把 `DiskPleNGramEmbedding` 接进官方 `Qwen4ExpTextPLELayer`，验证官方类 forward。
- [ ] 补跨段 / EOS / batch / 多段输入测试。
- [ ] 在能加载完整模型的机器上跑“memory vs disk PLE 层前向”对照。

**退出标准**：
- 能用官方 `Qwen4ExpForCausalLM` 或 `Qwen4ExpForConditionalGeneration` 加载模型且不把 200GB+ PLE 表载入内存。
- 官方 PLE layer forward 与 EngramDB disk adapter bit-exact。

#### Phase B：真实 E2E 算力/环境
- [ ] 找大内存 Linux / 工作站 / 云主机；
- [ ] 或使用 llama.cpp / vLLM / SGLang 的磁盘表替换路径；
- [ ] 完整模型 generate A/B。

**退出标准**：
- 真实 PLE 模型端到端跑通。
- 输出 bit-exact + tok/s + hit-rate + fetch/convert。

#### Phase C：性能架构
- [ ] Rust/PyO3 native PLE gather + rowid；
- [ ] 预取重叠，消除磁盘同步等待；
- [ ] 真实 PLE 冷/热基准。

#### Phase D：引擎服务化
- [ ] vLLM / SGLang / llama.cpp serving A/B；
- [ ] Unix socket / Arrow / 连接池 / 认证；
- [ ] 发布 v0.2.7+。

### 15.6 本轮纪律强化

1. **不能把“自实现数学验证”当成“官方模型验证”**
   还要在官方模型类中验证一次，才算真正闭环。

2. **大表不能因为“能跑”就认为正确**
   多分片、跨 shard、FP8 量化、EOS 边界都必须有 bit-exact 测试。

3. **环境限制不是技术债的终点，但要显式记录**
   完整模型 E2E 没做就是没做，不能假装闭环。

4. **继续坚持 Rust 为核**
   Python adapter 是语义验证和快速实验，不是最终性能产品。

5. **所有性能结论最终必须落在真实 PLE + 固定基准上**。

### 14.8 服务兄弟项目：qwen35-ple / engram-peft 契约对齐

- 新增 C ABI：
  - `engramdb_abi_version() -> u32`
  - `engramdb_rowids_for_seq(ids, len, out, out_cap, ple_spec) -> i32`
  - 已与 qwen35-ple `PleSpec.rowids_for_seq` 对拍通过。
- 增强 `DiskMultiHeadEmbedding`：
  - 支持 FP8 行 + `weight_scale` 反量化；
  - 支持 `output_dtype`；
  - 新增 `install_real_qwen_ple_embedding(store, scale, cache_size)`。
- 新增 `scripts/sibling_contract_smoke.py`：
  - C ABI rowids 对拍 qwen35-ple；
  - DiskMultiHeadEmbedding quick check；
  - 可选 engram-peft import 检查。
- 现状：qwen35-ple / engram-peft 依赖的存储与磁盘注入点已经可用；真实 PLE FP8 注入需要调用
  `install_real_qwen_ple_embedding`（带 scale），而不是默认 float32 注入。

## 16. 第十二轮系统性思考（Session 22：服务兄弟项目 qwen35-ple / engram-peft）

### 16.1 终极目标再锚定

不变：

> **让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何模型、训练器、推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。**

本轮之后，EngramDB 在四仓库协作中的位置更清晰：

```text
qwen35-ple        实验编排/评测
      ▲
engram-peft       模型/训练层
      ▲
EngramDB          PLE rowids + 存储 + C ABI + 磁盘注入
      ▲
LLM-CompileForge  推理 runtime（后续）
```

核心职责：

- 拥有 rowid 语义和 golden
- 拥有 Store-I / Store-P 数据面
- 提供 C ABI / Python API
- 提供 engram-peft 的磁盘注入点
- 不侵入模型/训练/推理逻辑

### 16.2 本轮完成

| 项 | 状态 |
|---|---|
| `engramdb_abi_version` | ✅ |
| `engramdb_rowids_for_seq` | ✅ 与 qwen35-ple 对拍一致 |
| `DiskMultiHeadEmbedding` FP8 反量化 | ✅ |
| `install_real_qwen_ple_embedding` | ✅ |
| `scripts/sibling_contract_smoke.py` | ✅ |
| qwen35-ple M0 quick | ✅ 通过 |

### 16.3 本轮新技术债（V52 起）

| # | 债务 | 影响 | 处置 |
|---|---|---|---|
| V52 | engram-peft 仍未真正消费 `table_source` 配置 | 用户仍需手动调用 `install_*`，不够方便 | 在 engram-peft 的 `get_engram_model` 中按 `table_source` 自动调用 EngramDB 注入 |
| V53 | qwen35-ple 真实 e2e 脚本仍用默认 float32 注入 | 直接跑真实 FP8 会读错行 | 更新兄弟项目脚本使用 `install_real_qwen_ple_embedding` |
| V54 | ~~`install_real_qwen_ple_embedding` 默认 scale 是硬编码~~ | ✅ 已解决 | `load_ple_weight_scale()` 自动从 checkpoint 读取，`install_real_qwen_ple_embedding(store, model_dir=...)` 可直接用 |
| V55 | C ABI 只实现 `PLE_QWEN_V1` | `ENG_DEEPSEEK_V1` 保留未实现 | 后续按需补 DeepSeek 表规格 |
| V56 | ~~C ABI rowids 没有 Python 便捷封装~~ | ✅ 已解决 | Python `engramdb.rowids_for_seq()`，优先 PyO3/C ABI，回退纯 Python |
| V57 | ~~兄弟契约 smoke 未进 CI~~ | ✅ 已解决 | 新增 `scripts/c_abi_smoke.py`，CI python-smoke 增加 C ABI 构建 + golden 对拍 |
| V58 | Python 磁盘热路径仍未下沉 Rust | 正确性已闭环，性能不达标 | 后续做 Rust/PyO3 native PLE gather + dequant |
| V59 | ~~版本落后于 master~~ | ✅ 已解决 | v0.2.7 已发布；本次修复 v0.2.7 CI 后发布 v0.2.8 |

### 16.4 借鉴矩阵（第十二轮增量）

| 来源 | 借什么 | 不借什么 | 目标 |
|---|---|---|---|
| **engram-peft** | config 驱动 `table_source`、引擎抽象、训练侧薄层 | 不借训练/模型实现 | 让 EngramDB 变得“配置即用” |
| **qwen35-ple** | 四仓库契约、golden 测试、编排层 | 不借实验逻辑 | 保证跨仓库正确性 |
| **HuggingFace** | model loading hook、skip 大权重、from_pretrained | 不重写模型 | 完整模型 E2E 加载路径 |
| **vLLM / SGLang / llama.cpp** | engine adapter、serving 替换 | 不复制推理 | 真实服务引擎接入 |
| **Rust / PyO3 / Arrow** | 原生热路径、零拷贝 | 不借查询引擎 | 性能目标 |
| **DuckDB / SQLite** | 嵌入式、manifest、cheksum | 不借 SQL | 产品形态 |
| **MLPerf / fio** | 固定基准、阈值、CSV | 只做测量 | 可回归性能结论 |

关键不冲突：

- EngramDB 不拥有模型/训练/推理逻辑
- 兄弟项目不拥有 rowid/存储/数据面
- 所有跨仓改动通过契约 + golden 守门
- 每个仓库只改自己职责内代码，调用方通过 API/config 组合

### 16.5 下一步计划

#### Phase A：让兄弟项目“配置即用”
- [ ] engram-peft：`table_source="engramdb:store"` 时自动调用 EngramDB 注入（兄弟侧）
- [ ] qwen35-ple：真实 e2e 改用 `install_real_qwen_ple_embedding`
- [x] EngramDB：自动读取 `weight_scale`
- [x] EngramDB：Python `rowids_for_seq()` 封装
- [x] EngramDB：C ABI 测试入 CI
- [x] v0.2.7 发布 / v0.2.8 修复 CI 后发布

**退出标准**：
- 在 engram-peft 中只配置 `table_source=engramdb:store`，不需要手动调用注入函数
- qwen35-ple 真实 e2e 脚本能正确读 FP8 PLE

#### Phase B：真实模型 E2E
- full-model custom loader + skip ngram_embedding
- 大内存/云环境
- 真实 PLE generate A/B

#### Phase C：性能
- Rust/PyO3 native rowid + gather + dequant
- 预取重叠
- LRU hit-rate 门禁

#### Phase D：服务/推理
- vLLM/SGLang/llama.cpp serving
- Unix socket / Arrow / 连接池
- C ABI / runtime 集成

### 16.6 本轮纪律

1. **跨仓库正确性必须以 golden/契约守门**，不能只靠本地自测。
2. **FP8/量化必须由存储层统一负责**，使用方只消费反量化后数值。
3. **配置驱动优先于手动调用**，方便使用才能成为基础设施。
4. **环境限制照实记录**，不能把“没跑”当成“能跑”。
5. **性能最终必须下沉 Rust**，Python 只负责语义和编排。

## 17. 第十三轮系统性思考（Session 24：v0.2.8 发布与工程稳定化）

### 17.1 终极目标再锚定

不变：

> **让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何模型、训练器、推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。**

我们不是“又一个 KV 存储”，也不是“向量数据库”。我们解决的是一个非常具体的开销问题：

- 确定性 n-gram 表非常大、只读、静态；
- 查询地址在推理/训练开始前就已知；
- 现在的痛点不是“有没有这张表”，而是“把这张表放进 RAM/显存太贵，放进磁盘又该如何做到低延迟、高吞吐、可编程、可服务”。

### 17.2 当前坐标

| 层 | 状态 |
|---|---|
| 存储面：Store-I / Store-P | ✅ 已闭环并有多平台基准 |
| 确定性 rowid：Rust / C ABI / PyO3 / Python | ✅ 四路径一致，golden 对拍 |
| 真实 Qwen PLE 数据面 | ✅ 128-shard Store bit-exact |
| 真实 PLE 层 forward bit-exact | ✅（自实现层） |
| Python 磁盘 Adapter / FP8 反量化 | ✅ |
| 兄弟项目契约 | ✅ C ABI + smoke + qwen35-ple M0 |
| 完整模型加载替换 PLE | ⚠️ 需要 official class / custom loader |
| 完整模型 E2E A/B | ❌ 受环境/内存限制 |
| 服务引擎级 A/B | ❌ 未做 |
| Rust native PLE 热路径 | ❌ 未做 |
| 发布/CI 稳定性 | ✅ v0.2.8 已修复，README 已刷新 |

### 17.3 本轮完成与发现

本轮（Session 23-24）主要做的是“把已经验证的正确性变成可发布、可安装、可文档化的产品面”：

- 修复 v0.2.7 CI 两个根因：
  - rustfmt import 顺序；
  - 无 torch 环境下 eager import `DiskPleNGramEmbedding` 导致 wheel smoke 失败。
- 补完 Phase A 的 EngramDB 侧：
  - `load_ple_weight_scale()` 自动读取 checkpoint；
  - `discover_ple()` 自动附带 `weight_scale`；
  - `disk_ple_from_discovery()` / `install_real_qwen_ple_embedding()` 自动 scale；
  - Python `rowids_for_seq()`；
  - PyO3 native `rowids_for_seq` / `abi_version`；
  - C ABI smoke 进入 CI。
- 发布 v0.2.8。
- 刷新 README / python README，补上 Rust/Python 安装与真实 PLE 用法。

关键发现：

1. **“功能已正确”不等于“可发布”**
   C ABI、bit-exact、真实 PLE 都已验证，但 CI 仍会因 import 顺序和可选依赖问题失败。
   说明发布工程和正确性工程必须同时管理。

2. **无 torch 环境是 Python 包的基本输入**
   不是所有用户都装 PyTorch；核心 Store/rowids/discovery 必须能在纯 Python 环境使用。
   这次修复建立了“核心轻依赖、PyTorch adapter 按需加载”的边界。

3. **文档与版本已经开始分叉**
   v0.2.8 tag 后 README 才更新，意味着 PyPI 上 v0.2.8 的长描述可能不是最新。
   需要把文档更新纳入版本收口，而不是 release 后补写。

4. **兄弟侧“配置即用”仍未完成**
   EngramDB 这一侧已经准备好了，但 engram-peft 消费 `table_source`、qwen35-ple 真实脚本切换仍是外部仓库动作。

### 17.4 本轮新技术债（V60 起）

| # | 债务 | 影响 | 处置 |
|---|---|---|---|
| V60 | 发布前没有强制跑“完整 release gate” | v0.2.7 的 CI 问题直到推送后才暴露 | 新增 `scripts/release_gate.sh`，bump/push 前本地强制跑 |
| V61 | README 更新晚于 v0.2.8 tag | PyPI/发布物长描述可能滞后 | 下个版本收编本文档更新 |
| V62 | `ple_adapter.py` 用 dummy nn 兼容无 torch | 类型/错误提示不够清晰 | 后续做懒加载 plugin 或 stub，避免 dummy module 进入公共面 |
| V63 | `install_real_qwen_ple_embedding` 无 model_dir 时仍静默回退硬编码 scale | 错误 checkpoint 可能用错 scale | 生产路径改为显式要求 `model_dir` 或 `scale`，避免静默错误 |
| V64 | Python `rowids_for_seq()` 纯 Python fallback 使用固定 multipliers | 非标准 checkpoint 或 DeepSeek 规格时需要调用方额外处理 | 支持从 `info` / `multipliers` 自动解析 |
| V65 | engram-peft 仍未真正消费 `table_source` | 用户仍需手动调用注入函数 | 兄弟侧按配置自动注入 |
| V66 | qwen35-ple 真实 e2e 仍未切到 FP8 wrapper | 真实 FP8 路径未在兄弟项目全链验证 | 更新兄弟侧脚本 |
| V67 | Rust native PLE gather + dequant 热路径未做 | Python 版只是语义验证 | Phase C 下沉 Rust/PyO3 |
| V68 | 完整模型 E2E 未做 | 无法证明“官方模型类 + 磁盘 PLE”真实可用 | 找大内存/云环境或 custom loader |
| V69 | vLLM/SGLang/llama.cpp serving A/B 未做 | 尚无服务场景性能结论 | Phase D |
| V70 | `ENG_DEEPSEEK_V1` C ABI 未实现 | DeepSeek 侧无法用 C ABI | 按需实现 |
| V71 | Python Store 是 unsendable，服务每请求开新 Store | 多线程/长连接下开销和安全隐患 | 后续 RUST 侧安全句柄 / 线程池 / 连接复用 |
| V72 | README 示例没有自动化测试 | 文档仍可能漂移 | 将关键示例做成 smoke 或 doctest |
| V73 | `discover_ple()` 重复读取大型 safetensors index | 大模型 discovery 有冗余 IO | 可缓存 index 或返回一个轻量 spec 对象 |

### 17.5 借鉴矩阵（第十三轮增量）

| 来源 | 借什么 | 不借什么 | 为什么对我们有用 |
|---|---|---|---|
| **DuckDB** | 嵌入式、文件即库、manifest、零拷贝、可发布生态 | 不借 SQL/OLAP 查询引擎 | 确立“磁盘优先基础设施”的产品形态 |
| **SQLite** | 单文件/便携、版本化格式、简单清晰 | 不借关系模型/事务语义 | 让 Store 易于迁移和校验 |
| **RocksDB / FoundationDB** | checksum、原子发布、文件版本、损坏检测 | 不借 LSM 或分布式事务 | 让大表发布可校验、可回滚 |
| **HuggingFace safetensors** | 分片 checkpoint、metadata index、lazy scalar 读取 | 不借模型定义/训练器 | `discover_ple` / `load_ple_weight_scale` 可复用该接口精神 |
| **vLLM / SGLang** | 模型加载 hook、权重替换、CPU offload、cache 管理 | 不借 serving 内核 | 不改上游源码接入真实引擎 |
| **llama.cpp** | mmap 大表、量化表、极简文件 | 不借 GGUF/推理 kernel | 验证“低配机器也能跑大 n-gram 表” |
| **Arrow** | IPC、零拷贝、列式传输 | 不借查询引擎 | 服务化时传输原始行/e_t |
| **DiskANN / Memcached** | LRU、冷热分层、预取 | 不借 ANN/通用 KV | 优化 PLE 在线读路径 |
| **MLPerf / fio** | 固定协议、阈值、CSV、可复现 | 不借其领域指标 | 所有性能结论可回归 |
| **engram-peft / qwen35-ple** | 配置驱动集成、四仓库 golden、契约测试 | 不借训练/评测逻辑 | 保证跨仓正确性 |
| **maturin / abi3 / PyPI** | 多平台 wheel、abi3、发布自动化 | 不借 Python 框架 | 降低安装门槛 |

### 17.6 下一阶段开发计划

#### Phase 0：发布与工程稳定性（先做，门槛）
- [x] 新增 `scripts/release_gate.sh`：
  - `cargo fmt --all --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --workspace`
  - `python_wheel_smoke.py`
  - `service_smoke.py`
  - `c_abi_smoke.py`
  - `decode_baseline_check.py`
  - 已接入 `scripts/bump.sh`（默认 bump 前先跑，可用 `--skip-gate` 跳过）。
- [ ] 将最新 README/python README 收编进下一个版本（本次已继续刷新，待 bump 收口）。
- [x] `install_real_qwen_ple_embedding` 去掉静默硬编码 fallback，或至少输出显式 warning。
- [x] `rowids_for_seq()` 支持 `multipliers`/`info` 来源。
- [ ] 把 README 核心示例抽成可执行 smoke，防止再次漂移（rowids/discovery/safetensors 示例已进入 `python_wheel_smoke.py`，其余待补）。

**退出标准**：
- 本地一条命令能完整预检所有发布门禁。
- 下一次 bump 前 README 与代码同一点提交。

#### Phase A：兄弟项目“配置即用”
- [x] engram-peft：支持 `table_source="engramdb:store"` 自动调用 EngramDB 注入（feature branch `feat/engramdb-table-source`）。
- [x] qwen35-ple：真实 e2e 改用配置驱动的 `install_real_qwen_ple_embedding` 路径（`run_m0_smoke.py --e2e --ple-model-dir ...`）。
- [ ] 跨仓契约 smoke 纳入兄弟项目 CI。

**退出标准**：
- 用户只需配置 `table_source=engramdb:store`，不需要手动 import 注入函数。
- qwen35-ple 真实 FP8 PLE 全链路跑通。

#### Phase B：真实模型 E2E
- [x] 写 custom loader / from_pretrained hook，跳过 `ngram_embedding.shard_*` 大权重（`engramdb.official_loader` + `qwen35-ple/scripts/qwen4_ple_custom_loader.py`）。
- [x] 真实 FP8 PLE e2e 已在本机跑通：Qwen3.5-0.8B + 真实 128-shard Store-I + 配置驱动 `table_source="engramdb:store"`，forward/generate 有限、无 NaN（`REAL_FP8_E2E_OK`）。
- [ ] 在官方 `Qwen4ExpForCausalLM` 或等价类中替换真实 PLE 并实机验证（代码路径已就绪，待包含 Qwen4Exp 的大内存/新 transformers 环境）。
- [ ] 找大内存 Linux / 云环境，跑 memory vs disk generate A/B。

**退出标准**：
- 完整模型加载不把 200GB+ PLE 表放进内存。
- 官方 PLE 层 forward 与磁盘 adapter bit-exact。
- 有真实 tok/s、hit-rate、fetch/convert 数据。

#### Phase C：性能架构
- [ ] Rust/PyO3 native rowid + gather + dequant。
- [ ] 预取重叠，隐藏磁盘同步等待。
- [ ] 真实 PLE 冷/热、批大小、并发矩阵基准。

**退出标准**：
- 磁盘 PLE 热路径不再依赖 Python 逐行转换。
- 性能结论可复现并接近“可服务”门槛。

#### Phase D：服务化 / 推理引擎
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] 安全的 Store 句柄 / 线程池 / 连接复用。
- [ ] Arrow IPC 服务化、认证、发布形态。

**退出标准**：
- 至少一个真实引擎能在不改上游源码的情况下使用 EngramDB PLE。
- 有 serving 场景的 tok/s 和延迟数据。

### 17.7 本轮纪律强化

1. **正确性、性能、发布工程三者同等重要**
   不能只验证 bit-exact 就发版；还要保证 CI、文档、安装路径都闭环。

2. **核心包必须轻依赖**
   Store、rowids、discovery、服务不应被迫导入 PyTorch；PyTorch adapter 必须按需加载。

3. **“配置即用”优先于“手动调用”**
   方便使用是基础设施的命门；兄弟侧自动消费配置比“提供更多函数”更重要。

4. **跨仓正确性继续靠 golden / C ABI 守门**
   不依赖各自仓库的偶然“能跑”。

5. **性能最终必须下沉 Rust**
   Python 只做语义验证和编排，不能作为性能终点。

6. **文档与版本必须同点收编**
   避免“代码已发布，README 还在旧版本”的分叉。


# 18. 第十四轮系统性思考（Session 26：配置即用 + 真实 FP8 e2e + Phase B 初步）

## 18.1 终极目标（不变）

> 让 DeepSeek Engram / Qwen PLE 这类“确定性哈希 n-gram 记忆表”成为任何小模型、训练器、
> 推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

三条不可妥协的轴线：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 模型端到端差距 ≤5%；CPU 小模型 ≥50 tok/s；EngramDB 参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录可嵌入 + 可服务；manifest 可校验；Arrow 零拷贝；engine 薄 adapter 不改上游 |
| C. 科学契约 | 每个性能/正确性结论有真实 PLE + 固定输入 + CSV/阈值；bit-exact 必须官方类或 golden 双保险 |

本轮后坐标：

- 配置即用已从“设计字段”变成“可执行闭环”
- 真实 FP8 Store-I 首次在真实小模型 e2e 中跑通
- 但“完整官方 Qwen4Exp 模型加载 + 性能 A/B”仍未闭环

## 18.2 当前坐标

| 层 | 状态 |
|---|---|
| EngramDB 存储/rowid/发现/发布门禁 | ✅ 已闭环 |
| engram-peft `table_source="engramdb:store"` 自动注入 | ✅ 已合入 master |
| qwen35-ple YAML → EngramConfig 桥接 | ✅ |
| 真实 FP8 Store-I e2e | ✅ 本机跑通（Qwen3.5-0.8B + 真实 128-shard） |
| 跨仓 golden/契约 CI | ✅ 已入 qwen35-ple CI |
| 官方 Qwen4Exp 完整模型加载 | ⚠️ 有 dry-run/代码路径，未实机 |
| 官方 PLE 层 + DiskPleNGramEmbedding bit-exact | ⚠️ 已有自实现层 bit-exact，官方类未验证 |
| memory vs disk 性能 A/B | ❌ |
| Rust/PyO3 原生热路径 | ❌ |
| vLLM/SGLang/llama.cpp serving A/B | ❌ |
| Store 线程安全/连接复用 | ❌ |

## 18.3 本轮新技术债（V74 起）

| # | 债务 | 影响 | 处置 |
|---|---|---|---|
| V74 | 真实 FP8 e2e 不是“官方 Qwen4Exp 完整模型” | 只证明“真实表 + engram-peft 配置驱动”可行，不能作为完整模型级验收 | 下一优先：官方 Qwen4Exp 模型类实机 |
| V75 | `qwen4_ple_custom_loader.py --load-model` 仍可能在 `from_config` 阶段分配巨大 ngram embedding | 未真正绕过 200GB+ PLE 内存 | ✅ 已在官方类构造前 patch `ngram_embedding` 为轻量占位，再加载非 shard 权重；待 Qwen4Exp 大内存实测 |
| V76 | `run_real_fp8_e2e.py` 依赖临时 PYTHONPATH/缓存库路径 | 不可复现，换机器/CI 不能直接跑 | 写可复现 venv/uv lock/安装脚本，或把必需轻量依赖正式化 |
| V77 | `DiskPleNGramEmbedding` 内部自管理 history，未接 Transformers `Cache` | 多段、streaming、MTP 等边界可能不一致 | 接入官方 cache/conv_state 语义，或提供严格单段验证 + 明确限制 |
| V78 | `table_source="engramdb:view"` 仍未实现 | 配置面只剩 store 一条路 | 后续实现 view reader 注入 |
| V79 | 跨仓 CI 只覆盖轻量 hash golden | 未覆盖 engram-peft 运行时和官方类 | 增加运行时/官方类 smoke（环境允许时） |
| V80 | 自动 store 注入在无 `model_dir`/`scale` 时静默选择 float32 路径 | 真实 FP8 行可能被错误解读 | 对 `PLE_QWEN_V1` 且未提供 scale/model_dir 时显式报错或强制要求 |
| V81 | engram-peft 全局类 patch 绑定单个 store | 多模型/多服务并发不安全 | 改为实例级注入或线程安全 store 注册表 |
| V82 | 尚无真实 memory vs disk A/B | 性能契约悬空 | 固定 seed/reps/token 数，输出 tok/s、hit-rate、fetch/convert 分段 |
| V83 | engram-peft/qwen35-ple 未 bump、README 未随版本收编 | 发布物与代码分叉 | 下一版本统一 bump + README 同 commit |
| V84 | Store 线程安全/连接复用仍未做 | 服务化/多请求不可靠 | Rust 侧安全句柄/每线程 store 池 |
| V85 | 官方模型类需要新版 transformers + 大内存 | 本机无法完成官方类 A/B | 找大内存 Linux/云环境，或拉取支持 Qwen4Exp 的 transformers 镜像 |

## 18.4 借鉴矩阵（本轮聚焦：如何不冲突地接近目标）

| 来源 | 借什么 | 明确不借 | 为什么对我们有用 |
|---|---|---|---|
| **DuckDB** | 嵌入式、目录即库、manifest、Arrow 输出、可发布生态 | 不借 SQL/OLAP | 确立“存储基础设施”形态 |
| **SQLite** | 单文件/便携、integrity check、每线程连接 | 不借关系模型/事务 | 让 Store 可校验、可嵌入 |
| **HuggingFace Transformers** | `from_pretrained` 前置 hook、state_dict 过滤、构造后替换大模块 | 不重写模型定义 | 解决“完整模型加载时跳过 PLE 大表” |
| **safetensors** | 分片 index、lazy scalar、metadata | 不借模型格式 | discovery 已有；后续可做 index 缓存 |
| **vLLM / SGLang** | 模型加载 hook、权重替换、cache/offload、engine adapter | 不借推理内核 | 不改上游源码接入真实引擎 |
| **llama.cpp** | CPU-first、顺序预热、单二进制、测量文化 | 不借 GGUF/推理 kernel | 用最低成本验证磁盘表可服务性 |
| **Arrow/IPC** | 零拷贝批次、跨语言边界 | 不借查询/执行引擎 | 服务化时传输原始行/e_t |
| **DiskANN / Memcached** | LRU、冷热分层、预取、命中率指标 | 不借 ANN/通用 KV | 优化在线读路径，并且必须带命中率证据 |
| **MLPerf / fio** | 固定输入、固定命令、阈值、CSV、cache-mode | 不借领域指标 | 所有性能结论可回归 |
| **RocksDB / FoundationDB** | 不可变段、checksum、原子发布 | 不借 LSM/分布式事务 | 大表发布可靠、可回滚 |
| **engram-peft / qwen35-ple** | 配置驱动、跨仓 golden、契约测试、编排 | 不借训练/评测/存储实现 | 让我们真正做到“配置即用” |
| **PyPA / maturin / abi3** | wheel 矩阵、abi3、Trusted Publishing | 不借 Python 框架 | 降低安装门槛 |

关键不冲突原则：

- 我们只做 PLE/Engram 的**存储、布局、读取、服务化**，不做模型/训练/推理内核。
- 所有引擎接入都是**薄 adapter / hook / patch**，不 fork 上游。
- 所有“快”的结论必须来自**真实 PLE + 固定基准**。
- 跨仓改动只走**契约 + golden**，避免四仓互相踩脚。

## 18.5 开发计划（按“先可信、再性能、再服务”排序）

### Phase B1：官方模型加载不分配大 PLE 表（最高优先）
- [x] 在 `AutoConfig.from_pretrained` / `from_config` 前 patch 官方 `Qwen4ExpTextNGramEmbedding` 构造，使用轻量占位。
- [x] 用 safetensors index 过滤 `ngram_embedding.shard_*` / `ngram_embedding.weight`，只加载非 PLE 权重。
- [x] 加载完模型后调用 `install_disk_ple_in_official_model` 替换所有 PLE 模块。
- [ ] 验证：峰值内存不包含 200GB+ PLE 表，且模型可 forward（代码路径已落地，待含 Qwen4Exp 的 Transformers/大内存环境实测）。

**退出标准**：官方 Qwen4Exp 模型在不加载 PLE 大表的情况下完成构造和加载。

### Phase B2：官方类 bit-exact
- [x] 在官方 `Qwen4ExpTextPLELayer` / `Qwen4ExpTextNGramEmbedding` 中替换 `DiskPleNGramEmbedding`（冻结官方快照结构 smoke）。
- [x] 小批量合成表与内存 PLE 层输出 max-abs=0（`scripts/qwen4_ple_bit_exact_small.py`，覆盖 batch + EOS + chunked streaming）。
- [ ] 覆盖 MTP、Transformers `Cache` streaming 边界及真实 PLE 行验证（内部 chunked streaming + batch + EOS 已通过小表 bit-exact）。

**退出标准**：官方类 + 磁盘 adapter 与官方内存路径 bit-exact。

### Phase B3：真实 A/B
- [ ] 固定输入、固定 seed、多次重复，输出 memory vs disk tok/s。
- [ ] 记录 hit-rate、fetch/convert 分段、LRU 开关。
- [ ] 落 CSV + 阈值门禁。

**退出标准**：有真实 PLE 模型的性能数据，能判断是否达到 ≤5% 目标。

### Phase C：Rust / PyO3 热路径
- [ ] native rowid + gather + dequant。
- [ ] 预取与 transformer 计算重叠。
- [ ] 冷/热、批大小、并发矩阵。

**退出标准**：磁盘 PLE 热路径不再依赖 Python 逐行转换；性能结论可复现。

### Phase D：服务化 / 推理引擎
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] Store 线程安全句柄/连接池。
- [ ] Arrow IPC 服务化、认证、发布形态。

**退出标准**：至少一个真实引擎不改源码使用 EngramDB PLE，且有 serving 性能数据。

### 工程稳定性
- [ ] 把 `run_real_fp8_e2e.py` 的临时依赖路径固化为可复现 env（uv lock / 安装脚本 / Dockerfile）。
- [ ] engram-peft、qwen35-ple、EngramDB 下版本统一 bump，README 同点收编。
- [ ] 将完整官方类/运行时 smoke 纳入可用环境 CI 或 nightly。

## 18.6 本轮纪律强化

1. **“能跑”不等于“验收通过”**
   真实 FP8 e2e 跑通只是功能里程碑；官方类和性能数据才是验收。

2. **先绕开内存分配，再谈加载**
   如果不能证明完整模型加载不分配 200GB+ PLE 表，就不能算 Phase B 完成。

3. **性能结论必须带 hit-rate 和分段计时**
   否则无法区分磁盘慢、Python adapter 慢还是引擎慢。

4. **可复现环境优先于临时 hack**
   使用 `/tmp/pylibs` 或手工 PYTHONPATH 只能在开发机上成立，不能成为交付形态。

5. **跨仓改动继续只走契约和 golden**
   功能可以快速迭代，但语义正确性必须由可验证契约守住。

6. **版本、文档、代码同点收编**
   避免“代码已经推进，发布物和 README 还停在上一版”。

# 19. 第十五轮系统性思考（Session 27：Phase B1/B2 代码落地 + 异步预取方向）

## 19.1 终极目标（不变）

> 让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何小模型、训练器、
> 推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

三条不可妥协的轴线：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 模型端到端差距 ≤5%；CPU 小模型 ≥50 tok/s；EngramDB 参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录可嵌入 + 可服务；manifest 可校验；Arrow 零拷贝；engine 薄 adapter 不改上游 |
| C. 科学契约 | 每个性能/正确性结论有真实 PLE + 固定输入 + CSV/阈值；bit-exact 必须官方类或 golden 双保险 |

本轮之后：

- Phase B1 从“设计”变成“可执行代码”：占位构造 + 非 PLE 分片加载 + 磁盘替换。
- Phase B2 小表 bit-exact 已通过：batch、EOS、chunked streaming 均 max-abs=0。
- 但完整官方 Qwen4Exp 实机、真实 PLE 行 A/B、异步预取仍未闭环。

## 19.2 当前坐标

| 层 | 状态 |
|---|---|
| 存储/rowid/发现/发布门禁 | ✅ 已闭环 |
| engram-peft 配置驱动自动注入 | ✅ |
| qwen35-ple 配置桥接 + 跨仓 CI | ✅ |
| 真实 FP8 Store-I e2e | ✅ |
| 官方加载占位 patch + 非 PLE 分片加载 | ✅ 代码落地 |
| 官方快照结构 smoke | ✅ |
| 官方类小表 bit-exact（batch/EOS/streaming） | ✅ |
| 完整官方 Qwen4Exp 实机验证 | ❌ 无 Qwen4Exp Transformers |
| 真实 PLE 行官方类 bit-exact | ❌ |
| memory vs disk A/B | ❌ |
| 异步预取 / 计算掩盖 I/O | ❌ |
| Rust/PyO3 原生热路径 | ❌ |
| serving A/B | ❌ |

## 19.3 本轮关键发现

1. **PLE 行只依赖 token ids，不依赖 hidden states**，因此异步预取在原理上完全可行。
2. **当前 PyO3 `Store` 不能直接做后台预取**：
   - `Store::fetch` 没有释放 GIL；
   - `Store` 是 `#[pyclass(unsendable)]`；
   - 没有 `prefetch()` / future 语义。
3. **DiskPleNGramEmbedding 曾不保留 batch 维度**：现在已修复，支持 `[B,S,E]`、每 batch 独立 n-gram context、chunked streaming。
4. **小表 bit-exact 已经证明 rowid、素数表、EOS、偏移和 batch/context 逻辑正确**，但还不能替代真实 PLE 行验证。
5. **可以用“稀疏真实行 oracle”绕开完整 48GB 表**：对固定 token 集合，只从真实 checkpoint 中读这些 token 踩到的行到内存，构造官方内存 embedding，与 DiskPle 的 Store 读取做位级对比。
6. **完整官方模型验证的核心阻塞不是 EngramDB，而是环境和资源**：需要包含 Qwen4Exp 的 Transformers + 足够内存/时间。

## 19.4 本轮新增技术债（V86 起）

| # | 债务 | 影响 | 处置 |
|---|---|---|---|
| V86 | PyO3 `Store.fetch` 持 GIL 且 `Store` 不可跨线程 | 异步预取/计算掩盖无法实现 | ✅ 已用 `py.allow_threads` + Store 去掉 `unsendable`；并发 fetch smoke 通过 |
| V87 | 没有 `DiskPle.prefetch()` / future / 模型级 pre-hook | 无法提前发起 PLE 行读取 | ✅ 已增加 `DiskPle.prefetch()`、future/wait、模型级 forward pre-hook |
| V88 | 完整官方 Qwen4Exp 未实机验证 | B1 退出标准未达到 | 找含 Qwen4Exp 的 Transformers/大内存环境 |
| V89 | 小表 bit-exact 是合成数据 | 不能证明真实 shard/dtype 路径 | 稀疏真实行 oracle + 固定 token 集合对拍 |
| V90 | DiskPle 自管理 context，未接 Transformers Cache/MTP | streaming/MTP 边界未完全可信 | 接 Cache，或先明确单段/内部流式限制 |
| V91 | 没有真实 memory vs disk A/B | 性能契约悬空 | 固定 seed/reps/tokens，输出 tok/s + hit-rate + fetch/convert |
| V92 | 热路径仍 Python rowid + gather + dequant | 性能上不去 | Rust/PyO3 native hot path + 预取重叠 |
| V93 | 官方类/torch 运行时测试只在本地，不在 CI | 回归保护弱 | 可用环境加 nightly runtime smoke |
| V94 | e2e 依赖临时 PYTHONPATH/手工包 | 换机器不可复现 | uv lock / Dockerfile / 可复现 env |
| V95 | 三仓库版本/README 未同点收编 | 发布物与代码分叉 | 下一版本统一 bump |
| V96 | Store 线程安全/连接复用未做 | 服务化/并发不可靠 | 每线程 Store 池或 Rust 安全句柄 |
| V97 | `engramdb:view` 自动消费未做 | 配置面只有 store | 后续实现 view reader 注入 |
| V98 | 没有可用的大内存/云验收路径 | 完整模型 A/B 长期卡住 | 明确远程/云执行方案 |

## 19.5 借鉴矩阵（本轮：如何不冲突地接近目标）

| 来源 | 借什么 | 明确不借 | 为什么对我们有用 |
|---|---|---|---|
| **DuckDB** | 嵌入式目录库、manifest、Arrow、可发布 | 不借 SQL/OLAP | 确立“存储基础设施”形态 |
| **SQLite** | 单文件、每线程连接、integrity 检查 | 不借关系模型/事务 | Store 可校验、可嵌入、可并发 |
| **HuggingFace Transformers** | 构造前 hook、state_dict 过滤、构造后替换 | 不重写模型定义 | 已解决“加载时跳过 PLE 大表” |
| **vLLM / SGLang** | 模型加载 hook、权重替换、调度/cache | 不借推理内核 | 不改源码接入真实引擎 |
| **llama.cpp** | CPU-first、单二进制、测量文化、低依赖 | 不借 GGUF/推理 kernel | 最低成本验证磁盘表可服务 |
| **CUDA/GPU 推理流水线** | 异步 stream / event、计算与传输重叠 | 不借 CUDA kernel | 同样思路可用于 CPU 的线程池 + future |
| **RocksDB / FoundationDB** | 不可变段、checksum、原子发布 | 不借 LSM/事务 | 大表发布可靠、可回滚 |
| **DiskANN / Memcached** | LRU、冷热分层、预取、命中率 | 不借 ANN/通用 KV | 在线读路径优化必须带命中率证据 |
| **Arrow/IPC** | 零拷贝、跨语言、批次 | 不借查询/执行引擎 | 服务化传输原始行/e_t |
| **MLPerf / fio** | 固定输入、阈值、CSV、cache-mode | 不借领域指标 | 性能结论可回归 |
| **PyPA / maturin / abi3** | wheel 矩阵、abi3、发布 | 不借 Python 框架 | 降低安装门槛 |
| **engram-peft / qwen35-ple** | 配置驱动、契约测试、编排 | 不借存储/训练内核 | 跨仓契约稳定 |
| **DeepSeek / Qwen 官方** | 精确定义、bit-exact 参考 | 不自创 hash/重训表 | 语义事实标准 |

关键不冲突原则：

- 只做 **存储、布局、读取、服务化**，不做模型/训练/推理内核。
- 所有引擎接入都是 **薄 adapter / hook / patch**，不 fork 上游。
- 所有“快”的结论必须来自 **真实 PLE + 固定基准**。
- 跨仓改动只走 **契约 + golden**。
- 先做 **可信正确性**，再做 **性能**，最后做 **服务化**。

## 19.6 后续开发计划（按“风险/不可信度”排序）

### Track 1：把 B1/B2 从“小表可信”推到“真实可信”
- [ ] 做微缩官方模型验证（不必须全模型）：
  - 用官方 modeling 代码/冻结快照构造 2 层或覆盖所有 `ple_layer_index` 的 mini 模型；
  - 小 hidden/vocab，可使用合成非 PLE 权重；
  - 跑占位 patch → filtered state dict → `install_disk_ple_in_official_model`；
  - 验证官方类 + DiskPle 的 forward / generate 与内存路径 bit-exact。
- [x] 稀疏真实行 oracle：
  - 固定 token 序列；
  - 只从真实 checkpoint 读取这些 token 命中的 PLE 行；
  - 与 DiskPle 的 Store 读取做 bit-exact；
  - 已跑通：144 个真实行 byte-identical，DiskPle real-Store maxdiff=0.0（`scripts/sparse_real_row_oracle.py`）。
- [ ] 将 mini 官方模型 + 真实行 oracle 纳入可复现 smoke。
- [ ] 完整模型验证（作为最终 memory/performance gate，不是 bit-exact 前置）：
  - 找 Qwen4Exp 版 Transformers + 大内存/云环境；
  - 跑 `qwen4_ple_custom_loader.py --load-model`；
  - 验证不加载 PLE 大表、非 PLE 权重完整加载、可 forward/generate。

**退出标准（bit-exact）**：mini 官方类 + 真实 PLE 行 bit-exact。
**退出标准（全模型）**：完整官方模型加载不分配 PLE 大表，并进入性能 A/B。

### Track 2：异步预取 + Rust 热路径（性能关键）
- [x] PyO3 `Store.fetch` 释放 GIL，Store 支持跨线程/每线程实例（并发 fetch smoke 已过）。
- [x] `DiskPle.prefetch(rowids)` + future/wait（支持有 cache 与无 cache）。
- [x] 模型级 forward pre-hook，提前对当前步所有 PLE 模块发起预取。
- [x] 真实 Store 上的 prefetch micro A/B（`prefetch_real_ab.py`）：模拟 30ms 计算窗口时，总耗时从 ~192ms 降到 ~34ms。
- [ ] Rust 原生 rowid + gather + dequant，或至少把 gather/dequant 移入热路径。
- [ ] 用 hit-rate、prefetch_wait、fetch_s、convert_s 做 A/B。

**退出标准**：能证明“前面层计算掩盖 PLE 通信”，且有分段数据。

### Track 3：真实 memory vs disk A/B
- [ ] 固定模型、固定输入、固定 seed、多次重复。
- [ ] memory 表 vs EngramDB 磁盘表。
- [ ] 输出 tok/s、延迟分布、hit-rate、fetch/convert、LRU 开关。
- [ ] 落 CSV + 阈值门禁。

**退出标准**：得到可判断“是否 ≤5% 性能差距”的数据。

### Track 4：服务化 / 推理引擎
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] Store 每线程连接池/线程安全句柄。
- [ ] Arrow IPC / 服务化发布形态。
- [ ] `engramdb:view` 自动消费。

**退出标准**：至少一个真实引擎不改源码使用 EngramDB PLE，并有 serving 数据。

### Track 5：工程稳定
- [ ] 固化 e2e 环境（uv lock / Dockerfile / 安装脚本）。
- [ ] 三仓库版本 bump + README 同点收编。
- [ ] 把官方类/runtime smoke 纳入可用 CI 或 nightly。

## 19.7 本轮纪律

1. **先可信，再性能，再服务**：小表 bit-exact 只是正确性第一步，不能替代真实行。
2. **性能结论必须带 hit-rate 和分段计时**。
3. **“异步”必须证明真的 overlap**：要看 GIL 是否释放、Store 是否可跨线程。
4. **完整模型验证尽早找环境**：不要让环境阻塞拖成隐形债务。
5. **跨仓改动继续只走契约 + golden**。
6. **版本、文档、代码同点收编**。

# 20. 第十六轮系统性思考（Session 28：预取管线落地 + 真实行低资源验证）

## 20.1 终极目标（不变）

> 让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何小模型、训练器、
> 推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

三条不可妥协的轴线：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 模型端到端差距 ≤5%；CPU 小模型 ≥50 tok/s；EngramDB 参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录可嵌入 + 可服务；manifest 可校验；Arrow 零拷贝；engine 薄 adapter 不改上游 |
| C. 科学契约 | 每个性能/正确性结论有真实 PLE + 固定输入 + CSV/阈值；bit-exact 必须官方类或 golden 双保险 |

## 20.2 本轮坐标更新

| 层 | 状态 |
|---|---|
| 存储/rowid/发现/发布门禁 | ✅ 已闭环 |
| engram-peft 配置驱动自动注入 | ✅ |
| qwen35-ple 配置桥接 + 跨仓 CI | ✅ |
| 真实 FP8 Store-I e2e | ✅ |
| 官方加载占位 + 非 PLE 分片加载 | ✅ 代码落地 |
| 官方类小表 bit-exact（batch/EOS/streaming） | ✅ |
| 稀疏真实行 oracle（checkpoint vs Store vs DiskPle） | ✅ maxdiff=0.0 |
| PyO3 Store GIL 释放 + 并发 fetch | ✅ |
| DiskPle prefetch / future / 模型级 pre-hook | ✅ |
| 真实 Store 预取 micro A/B | ✅ 模拟计算窗口已见显著收益 |
| 真实 full-model 性能 A/B | ❌ |
| Rust 原生 rowid/gather/dequant | ❌ |
| serving / vLLM / SGLang / llama.cpp | ❌ |
| Store 连接池 / 服务化生命周期 | ❌ |

## 20.3 本轮关键发现

1. **小资源也能验证真实行正确性**：9 个 token、144 个真实 PLE 行，即可证明 checkpoint ↔ Store-I byte-identical，且 DiskPle real-Store dequant maxdiff=0.0。
2. **异步预取已经不仅仅停留在设计**：
   - PyO3 `Store.fetch` 释放 GIL，Store 可并发；
   - `DiskPle.prefetch()` + future/wait 可用；
   - 模型级 pre-hook 可在 forward 早期发起预取。
3. **预取微基准显示有显著收益**：在真实 Store + 模拟 30ms 计算窗口下，同步 ~192ms → 预取 ~34ms。
4. **但这还不是最终验收**：当前是用 `time.sleep` 模拟前面层计算，且只测了单次 batch；不能替代真实模型端到端 tok/s。
5. **Python 热路径可能成为新的瓶颈**：rowid、列表转 tensor、FP8 dequant、flatten 仍在 Python 侧；预取隐藏了磁盘，不一定能隐藏 Python 开销。

## 20.4 本轮新增技术债（V99 起）

| # | 债务 | 影响 | 处置 |
|---|---|---|---|
| V99 | 预取收益只在模拟计算窗口下验证 | 不能证明真实模型端到端收益 | 在真实模型/真实 PLE 层做 sync vs prefetch A/B |
| V100 | `forward` 会等待所有 pending prefetch | 磁盘慢时可能阻塞，且无优先级/取消 | 增加 adaptive wait、超时、按需同步取 |
| V101 | Python 热路径仍做 rowid + convert | 预取藏住磁盘后 Python 成为瓶颈 | Rust/PyO3 native rowid + gather + dequant |
| V102 | prefetch executor 没有生命周期管理 | 长服务可能累积线程/资源 | 增加 close/shutdown/共享 executor |
| V103 | Store 并发只验证了单机 Python 线程 | 服务化连接池、每请求隔离未做 | Store 池/线程安全句柄 |
| V104 | 没有真实 full-model A/B | 性能契约仍悬空 | 大模型环境固定输入/seed/reps CSV |
| V105 | 没有 hit-rate / 真实等待分布 | 无法区分磁盘、Python、引擎开销 | 增加 hit-rate、p50/p95/p99、fetch/convert 分段 |
| V106 | MTP / Transformers Cache 未接 | streaming/MTP 边界风险 | 接 Cache 或明确限制 |
| V107 | 完整官方 Qwen4Exp 加载未实机 | B1 最终 gate 未过 | 云/大内存环境 |
| V108 | 可复现环境未固化 | 换机器无法跑 | uv lock / Dockerfile |
| V109 | 三仓库版本/README 未收编 | 发布物分叉 | 统一 bump |
| V110 | `engramdb:view` 自动消费未做 | 配置面不完整 | view reader 注入 |
| V111 | 还没有正式 benchmark harness/CSV 阈值 | 性能结论易漂移 | 固定输入 + CSV 门禁 |

## 20.5 借鉴矩阵（本轮：性能路径如何不冲突）

| 来源 | 借什么 | 明确不借 | 为什么有用 |
|---|---|---|---|
| **CUDA/GPU 推理** | async stream/event、计算与传输重叠 | 不借 CUDA kernel | 本轮已把相同思想落到 Rust thread + GIL release |
| **数据库系统 (PostgreSQL/Redis)** | 后台预取、连接池、future/等待、LRU | 不借 SQL/缓存语义 | 服务化时需要连接生命周期和池化 |
| **DuckDB** | 嵌入式目录库、manifest、Arrow | 不借 SQL/OLAP | 存储形态 |
| **SQLite** | 每线程连接、integrity | 不借关系模型 | Store 可并发、可校验 |
| **vLLM / SGLang** | prefill/decode 调度、continuous batching、engine hook | 不借推理内核 | 真实 serving 接入点 |
| **llama.cpp** | CPU-first、低依赖、测量文化 | 不借 GGUF/kernel | 最低成本验证 |
| **Memcached / DiskANN** | LRU、hit-rate、冷热分层 | 不借通用 KV/ANN | 预取必须带命中率证据 |
| **Arrow / IPC** | 零拷贝、跨语言批次 | 不借查询引擎 | 服务化传输 |
| **MLPerf / fio** | 固定输入、阈值、CSV、cache-mode | 不借领域指标 | 性能回归 |
| **RocksDB / FoundationDB** | 不可变段、checksum、原子发布 | 不借 LSM/事务 | 大表发布可靠 |
| **PyPA / maturin / abi3** | 发布矩阵、abi3 | 不借 Python 框架 | 安装门槛 |
| **engram-peft / qwen35-ple** | 配置驱动、契约测试、跨仓 CI | 不借存储/训练内核 | 跨仓稳定 |

不冲突原则不变：

- 只做存储、布局、读取、服务化，不做模型内核。
- 接入都是薄 adapter / hook / patch，不 fork 上游。
- 性能结论必须来自真实 PLE + 固定基准。
- 跨仓正确性只走契约 + golden。
- 先可信，再性能，再服务。

## 20.6 后续开发计划

### Track 1：真实模型预取 A/B（最高优先）
- [ ] 在真实 Qwen4Exp 或可运行 mini 官方模型中，接入模型级 prefetch hook。
- [ ] 记录实际 PLE 层到达时间、prefetch 完成时间、wait 时间。
- [ ] sync vs prefetch 的 end-to-end tok/s。
- [ ] 固定输入、固定 seed、多次重复，落 CSV。

**退出标准**：能证明真实模型里“前面层计算掩盖 PLE 通信”。

### Track 2：Prefetch 生产化
- [ ] prefetch executor 生命周期管理 / shutdown。
- [ ] 多 outstanding future、超时、错误回退到同步。
- [ ] hit-rate、wait 分布、fetch/convert 分段统计。
- [ ] 多 PLE 模块并发 prefetch 的合并去重。

**退出标准**：长服务稳定，统计完整。

### Track 3：Rust/PyO3 热路径
- [ ] native rowid 批量生成。
- [ ] native gather + FP8 dequant + flatten。
- [ ] 减少 Python 每 token 循环。
- [ ] 保持与 Python bit-exact 一致。

**退出标准**：热路径不再依赖 Python 逐行转换。

### Track 4：真实 memory vs disk A/B
- [ ] memory 表 vs EngramDB 磁盘表。
- [ ] tok/s + hit-rate + fetch/convert + prefetch_wait。
- [ ] CSV + 阈值门禁。

**退出标准**：判断是否达到 ≤5% 性能差距。

### Track 5：服务化 / 推理引擎
- [ ] Store 连接池 / 每线程句柄。
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] Arrow IPC 服务化。
- [ ] `engramdb:view` 自动消费。

**退出标准**：至少一个真实引擎不改源码可用。

### Track 6：工程稳定
- [ ] 可复现环境。
- [ ] 三仓库 bump + README。
- [ ] runtime/官方类 CI 或 nightly。

## 20.7 本轮纪律

1. **预取微基准不是最终结论**，必须上真实模型。
2. **“异步”必须测量真实 wait/hit-rate**，不能只看总时间。
3. **Python 热路径要同步评估**：藏住磁盘后，Python 可能成为新瓶颈。
4. **小资源验证优先**：稀疏真实行 + mini 官方模型足够验证正确性，全模型只做最终 gate。
5. **跨仓正确性只走契约 + golden。**
6. **版本、文档、代码同点收编。**

## 20.8 Session 29 增量（Prefetch 生产化起步）

本轮继续性能路径，落地了以下小步：

- `DiskPleEmbedding.close()` / `DiskPleNGramEmbedding.close()`：prefetch executor
  生命周期管理，幂等关闭，长服务/基准脚本不再泄漏后台线程。
- `DiskPleNGramEmbedding.prefetch()` 现在返回底层 future，并保存
  `_last_prefetch_future`，便于 A/B 脚本观测预取完成时机。
- `install_disk_ple_prefetch_hook()` 兼容 PyTorch 两种 pre-hook 调用约定：
  `hook(module, args)` 与 `hook(module, args, kwargs)`。
- 新增 `qwen35-ple/scripts/mini_official_prefetch_ab.py`：用冻结官方
  `Qwen4ExpTextPLELayer` + 真实 Store + 真实 dense 前后块做 mini 官方模型
  sync vs prefetch A/B，输出 CSV，并记录 PLE 层到达时 prefetch 是否已完成。
- 该脚本目前是低资源 smoke，不是完整模型 end-to-end tok/s；正式 A/B 仍需
  真实 Qwen4Exp 或足够大的 mini 官方模型 + 冷/热分离 + 固定阈值。
- 修复 20k 预计算慢路径：
  - `PleDiskGather.fetch` 改为直接返回 `Store.fetch` 连续缓冲区，去掉 Python 去重/切片/join；
  - 新增 `engramdb.fetch_e_t_tensor()` / `PleDiskGather.fetch_tensor()`，一次 fetch + torch 转 tensor；
  - qwen35 `real_ple.fetch_e_t`、`precompute_real_ple_features.py` 已切换；`run_phase0.py --live-store` 可直接读 Store。
- 新增 Rust `rowids_for_seq_with_history` + PyO3 导出，标准真实 PLE adapter 可走 native rowid。

状态：V101（Python 热路径）开始收敛，V102 已部分关闭；V99/V100/V104/V105 仍开放。



# 21. 第十七轮系统性思考（Session 29/30：快速读取路径落地 + v0.2.9 发布）

## 21.1 终极目标（不变）

> 让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何小模型、训练器、
> 推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

三条不可妥协的验收轴：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 模型端到端差距 ≤5%；CPU 小模型 ≥50 tok/s；EngramDB 参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录嵌入式 + 可服务化；manifest 可校验；Arrow 零拷贝；engine 薄 adapter 不改上游 |
| C. 科学契约 | 每个结论有真实 PLE + 固定输入 + 冷热标注 + CSV/阈值；bit-exact 以官方类或 golden 双保险 |

## 21.2 本轮坐标更新

| 层 | 状态 |
|---|---|
| 真实 Store ↔ checkpoint byte-exact | ✅ 144 行 oracle 已闭环 |
| 预取管线（future / hook / mini 官方 A/B） | ✅ 低资源 smoke 已跑通 |
| prefetch executor 生命周期 | ✅ close/shutdown 已落地 |
| Python 慢路径 `PleDiskGather` | ✅ 已改为直接 `Store.fetch`，新增 `fetch_e_t_tensor` |
| qwen35 live-store 直接读取 | ✅ 已落地（`--live-store`） |
| native rowid + history | ✅ PyO3 已导出，标准 adapter 可走 native |
| 完整 Qwen4Exp 官方模型 | ❌ 仍受环境限制 |
| 真实模型端到端 A/B | ❌ 未完成 |
| Rust/PyO3 原生 gather + FP8 dequant + flatten | ❌ 未完成 |
| serving / 连接池 / Arrow 生产化 | ❌ 未完成 |

## 21.3 本轮关键发现

1. **慢的不一定是 EngramDB 核心，而是 Python 适配层**：
   - 旧 `PleDiskGather.fetch` 在 20k token / 320k 行上是 16.857s；
   - 直接 `Store.fetch` 只做一次连续读取，才是真正的存储面路径。
2. **“直接 Store.fetch + torch.frombuffer”是正确的高层读取形态**：
   - 它同时适用于预计算 npy 和 live 训练读取；
   - 也避免了 10GB 中间文件的搬运成本。
3. **Python 热路径需要分阶段收敛**：
   - 已把 rowid 生成挪到 native；
   - 但 `DiskPleEmbedding.forward` 的 per-row bytes dict / join、`DiskPleNGramEmbedding` 的 Python batch 组装仍在；
   - 所以“磁盘被隐藏后，Python 可能成为新瓶颈”仍是真实风险。
4. **微基准必须区分冷/热和真实 I/O**：
   - 同一个 `Store.fetch` 在页缓存热时可能 0.03s，冷时可能数十秒；
   - 正式 A/B 必须记录介质、冷热、是否重复读取。
5. **完整模型不是正确性前置条件**：
   - 小表 bit-exact + 稀疏真实行 + mini 官方模型已经覆盖主要正确性风险；
   - 完整模型只应作为最终内存/性能 gate，不应阻塞开发。

## 21.4 本轮技术债（在 V99–V111 基础上新增/更新）

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V112 | `DiskPleEmbedding.forward` 仍走 Python per-row bytes dict/join | serving 热路径可能成为新瓶颈 | 用 `fetch_e_t_tensor`/native gather 替换 serving 内层 |
| V113 | 没有正式 live-store benchmark harness | 无法断言 20k/1M token 的真实收益 | 固定 tokens/rows/冷热/CSV/阈值 |
| V114 | 没有冷热分离的性能门禁 | 数字容易被页缓存欺骗 | 基准脚本加 `--cache-mode cold/warm`，记录介质 |
| V115 | 多 PLE 模块/多 outstanding prefetch 尚未合并去重 | 长服务可能重复读、线程失控 | prefetch 调度器 + 去重 + shared executor |
| V116 | full-model Qwen4Exp 仍未实机 | 最终 memory/performance gate 未过 | 云/大内存环境或缩小版官方模型继续逼近 |
| V117 | 三仓库版本/README/CI 未完全同点收编 | 发布物与实际代码可能漂移 | 每版本 gate 中固化跨仓 retest 指南 |

原有仍开放：V99（真实模型预取 A/B）、V100（超时/回退）、V104（memory vs disk）、V105（hit-rate/wait 分布）、V106（MTP/Cache）、V107（完整加载）、V108（可复现环境）、V109（三仓库收编）、V110（view 自动消费）、V111（benchmark harness）。

## 21.5 借鉴矩阵（本轮：如何不与已有设计冲突）

| 来源 | 借什么 | 不借 | 为什么不冲突 |
|---|---|---|---|
| **DuckDB** | 嵌入式单目录、manifest、Arrow/IPC 零拷贝 | 不借 SQL/OLAP/查询优化 | 我们只做定长点查的存储层 |
| **SQLite** | 每线程连接、integrity check、嵌入式产品形态 | 不借关系模型/事务 | 我们可复用它“稳定嵌入”的工程习惯 |
| **PostgreSQL/Redis/Memcached** | 连接池、后台预取、LRU、hit-rate、future/等待 | 不借 SQL/KV 语义 | 我们只需要服务化和冷热分层方法论 |
| **RocksDB/FoundationDB** | 不可变段、checksum、原子发布、manifest | 不借 LSM/分布式事务 | 我们的表是不可变静态大段 |
| **DiskANN/Milvus** | 冷热分层、顺序化读、滑窗、cache 统计 | 不借 ANN/向量检索 | 我们无近邻语义，只借 I/O 布局经验 |
| **vLLM/SGLang** | engine hook、continuous batching、去重、async overlap、统计 | 不借推理内核/CUDA kernel | 我们做存储数据面，引擎只做薄 adapter |
| **llama.cpp** | CPU-first、实测文化、顺序预热、低依赖 | 不借 GGUF/模型推理实现 | 我们借测量和启动策略 |
| **Arrow/IPC** | 跨语言零拷贝、批次结构 | 不借查询引擎/执行器 | 只做输出形态 |
| **MLPerf/fio** | 固定输入、固定命令、CSV、阈值、cache-mode | 不借各自领域指标 | 我们借可复现实验规范 |
| **engram-peft/qwen35-ple** | 配置驱动、契约测试、golden、跨仓 CI | 不借存储/训练内核 | 我们与它们是上层/下层关系，以契约为界 |

不冲突原则不变：

- 只做存储、布局、读取、服务化，不做模型内核。
- 接入都是薄 adapter / hook / patch，不 fork 上游。
- 性能结论必须来自真实 PLE + 固定基准 + 冷热标注。
- 跨仓正确性只走契约 + golden。
- 先可信，再性能，再服务。

## 21.6 后续开发计划（重排）

### Track 0：先把“测得准”解决（最高优先，不改大量架构）
- [x] 建正式 benchmark harness：固定 tokens、固定 rowids、重复次数、CSV、阈值（qwen35 `scripts/bench_live_store.py`）。
- [ ] 20k / 1M live-store 在目标机器复测并沉淀基线。
- [ ] 进一步区分：Python 组装、Store.fetch、torch 转换、实际磁盘 I/O 四段计时。
- [x] 写入 CSV + 阈值门禁（`--max-store-s / --max-tensor-s / --max-tensor-dedup-s`）。

**退出标准**：任何“快/慢”结论都能用一条命令复现且注明冷热。

### Track 1：消除剩余 Python 热路径
- [x] `DiskPleEmbedding` no-cache serving 路径改为直接 `Store.fetch` 连续读取，跳过 per-row dict/join。
- [ ] `DiskPleEmbedding` cache>0 路径仍可能受 Python bytes join 影响，需要继续评估。
- [ ] 评估 Rust/PyO3 native gather + FP8 dequant + flatten。
- [x] 保持与现有 bit-exact golden 一致（small bit-exact 仍 0.0 maxdiff）。

**退出标准**：磁盘隐藏后，Python 不再是主要开销；serving 路径与训练 live 路径共用同一 fast path。

### Track 2：Prefetch 生产化
- [x] 超时（可选 `prefetch_timeout`）、错误回退到同步。
- [x] 共享 executor 参数（`prefetch_executor`）已放到 `DiskPleEmbedding`/`DiskPleNGramEmbedding`/official loader。
- [ ] 多 PLE 模块之间的行合并去重尚未实现。
- [x] hit-rate、wait p50/p90/p99 统计已落地（`get_wait_distribution()`）。

**退出标准**：长服务稳定，统计完整。

### Track 3：真实模型性能验证
- [ ] full Qwen4Exp 或足够大的 mini 官方模型实机加载。
- [ ] memory vs disk A/B、sync vs prefetch A/B、tok/s。
- [ ] CSV + 阈值。

**退出标准**：能判断是否满足 ≤5% 性能契约。

### Track 4：服务化 / 推理引擎
- [ ] Store 连接池 / 每线程句柄。
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] Arrow IPC、`engramdb:view` 自动消费。

**退出标准**：至少一个真实引擎不改源码可用，并有 serving 数据。

### Track 5：工程稳定
- [ ] uv lock / Dockerfile 固化 e2e 环境。
- [ ] 三仓库版本 bump + README + retest 指南同点收编。
- [ ] runtime / 官方类 smoke 纳入 CI 或 nightly。

**退出标准**：换机器也能 10 分钟内复现当前最好结论。

## 21.7 本轮纪律

1. **Python 快不代表磁盘快，磁盘快不代表 Python 快**：必须分段计时。
2. **冷热是性能结论的一部分**，不标注冷热的数据不能当门禁。
3. **先测量再优化**：不再在未知瓶颈上继续堆 Python 逻辑。
4. **正确性继续用小资源闭环**，完整模型只做最终 gate。
5. **跨仓只走契约 + golden + retest 指南**，避免版本漂移。
6. **版本、文档、代码、retest 指南同点收编**。


# 22. 第十八轮系统性思考（Session 32：懒加载 live-store + WSL 实测 + 磁盘优先路径确认）

## 22.1 终极目标（不变）

> 让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何小模型、训练器、
> 推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

三条不变验收轴：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 模型端到端差距 ≤5%；CPU 小模型 ≥50 tok/s；EngramDB 参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录嵌入式 + 可服务化；manifest 可校验；Arrow 零拷贝；engine 薄 adapter 不改上游 |
| C. 科学契约 | 每个结论有真实 PLE + 固定输入 + 冷热标注 + CSV/阈值；bit-exact 有官方类或 golden 双保险 |

## 22.2 本轮坐标更新

| 层 | 状态 |
|---|---|
| `--live-store` 全量加载 1M e_t | ❌ 已确认不可行（10GB OOM） |
| 懒加载 `LiveETStore` / `LiveETView` | ✅ 已实现，按训练/评测窗口按需读取 |
| 1M token 内存占用模型 | ✅ 已明确：只保留 rowids，不再保留全量 e_t |
| WSL Store-I 随机读 | ❌ 100k token ≈ 56s，远低于 README 高性能 |
| Store-P / Rust 批量 / 多线程 WSL 实测 | ❌ 未做 |
| 磁盘优先使用方式 | ✅ 已从设计层确认并落地 |

## 22.3 本轮关键发现

1. **“全量加载 e_t 才能跑”是错误方向**：
   - 1M × 2560 × 4B ≈ 10GB，在 WSL 15GB 内存下必然 OOM；
   - 这违背了 EngramDB 的磁盘优先设计。
2. **按窗口懒加载是正确的产品形态**：
   - 只保留 `[T,16]` rowids，约 128MB / 1M token；
   - 每个训练 step 只读当前 `seq_len × 16` 行；
   - 内存占用从 10GB 降到 KB/MB 级。
3. **懒加载解决了内存问题，但没有解决 WSL 随机 IO 性能问题**：
   - `Store.fetch` 在 WSL 上 100k token / 1.6M 行约 56s；
   - 这说明 Store-I 原始随机 scatter 在虚拟磁盘上仍是瓶颈。
4. **性能问题要用 Store-P / Rust / 顺序化解决，而不是用“全量内存”绕开**：
   - 全量内存是把存储层问题转嫁给 RAM；
   - 正确路径是优化读放大和访问模式。
5. **后续需要把懒加载抽象成可复用 Dataset / DataLoader**：
   - 目前只在 `run_phase0.py` 内部实现；
   - 下一个目标是变成通用 live-store 数据流。

## 22.4 新增/更新技术债（在 V112–V117 基础上）

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V118 | WSL Store-I 随机读远低于 README 数字 | 大规模训练/全量扫描仍慢 | WSL 上建 Store-P + Rust/C 批量 + 多线程实测 |
| V119 | live-store 懒加载只在 run_phase0 内部 | 无法复用到其他实验/训练器 | 提炼为通用 `LiveETDataset` / `IterableDataset` |
| V120 | 没有验证 Store-P 在 WSL 上的真实收益 | 无法判断是否应切 Store-P | 构建 WSL Store-P 视图并做同口径 A/B |
| V121 | 没有 1M token 懒加载正式实验基准 | 不知道每 step/窗口真实 fetch 成本 | 固定 steps/seq_len/seed，记录每 step fetch 时间 |
| V122 | Store 连接生命周期仍非生产级 | 长服务/分布式训练有风险 | Store 池、线程安全句柄、显式 close/上下文 |

原 V112–V117 依然开放或部分开放。

## 22.5 借鉴矩阵（本轮增量）

| 来源 | 借什么 | 不借 | 为什么不冲突 |
|---|---|---|---|
| **PyTorch IterableDataset / DataLoader** | 流式、按 batch 拉取、worker 并发 | 不借训练循环 | 我们只做“按需从磁盘取特征”的数据源 |
| **HuggingFace Datasets（streaming）** | 不落全量内存、按需 map、可复现分片 | 不借 NLP 处理 | 我们借“大数据集不常驻内存”的形态 |
| **Arrow Dataset / IPC** | 分块、列式零拷贝、schema | 不借查询引擎 | 我们可把 Store 输出装成 Arrow 流 |
| **DiskANN / Milvus** | 冷热分层、顺序化、批量预取、缓存统计 | 不借 ANN/向量检索 | 继续只借 I/O 布局经验 |
| **vLLM/SGLang continuous batching** | 按 batch 调度、async prefetch、指标 | 不借推理内核 | 我们提供存储侧数据流 |
| **DuckDB** | 嵌入式、目录即库、不依赖大内存 | 不借 SQL | 形态对齐 |
| **RocksDB/FDB** | 不可变段、checksum、批量顺序读 | 不借 LSM/事务 | 保持静态大表语义 |

不冲突原则不变：

- 只做存储、布局、读取、服务化，不做模型内核。
- 接入都是薄 adapter / hook / patch，不 fork 上游。
- 性能结论必须来自真实 PLE + 固定基准 + 冷热标注。
- 跨仓正确性只走契约 + golden。
- 先可信，再性能，再服务。

## 22.6 后续开发计划

### Track A：把懒加载变成正式数据流（最高优先）
- [ ] 将 `LiveETStore` / `LiveETView` 从 `run_phase0.py` 提炼为 qwen35-ple 通用模块或 EngramDB Python API。
- [ ] 实现 `IterableDataset` / batch 级 reader：`__iter__` 每次只取一个窗口。
- [ ] 支持 control、分片、多 worker。
- [ ] 记录 per-batch fetch 时间、命中/未命中、总读取量。

**退出标准**：任意实验脚本可以用 3 行接入 live-store 数据流，不再全量加载 e_t。

### Track B：WSL/真实介质上的性能闭环
- [ ] 在 WSL 构建 Store-P 视图。
- [ ] Store-I vs Store-P vs 懒加载 vs 全量内存 同口径 A/B。
- [ ] 测试多线程 / C-Rust 批量读取。
- [ ] 输出 CSV + 阈值。

**退出标准**：能解释 WSL 慢在哪里，并给出可复现的最优路径。

### Track C：真实模型训练/推理验证
- [ ] 用懒加载 live 数据流跑 1M token real/control/seeds。
- [ ] 对比全量 npy / 懒加载 / Store-P 的 loss 与耗时。
- [ ] 如果可行，推进 100 tok/s CPU 推理闭环。

### Track D：服务化
- [ ] Store 连接池 / 每线程句柄。
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] Arrow IPC 数据流。

### Track E：工程稳定
- [ ] 固化 WSL 复现环境。
- [ ] 三仓库版本、README、retest 指南同点收编。
- [ ] 正式 CI / nightly 加入 live-store smoke。

## 22.7 本轮纪律

1. **内存不是用来替代磁盘的**：全量 e_t 是反模式。
2. **懒加载解决内存，不代表解决 IO**：必须继续优化 Store-P / 顺序化 / 多线程。
3. **所有大规模实验要有 per-batch / per-window fetch 时间**，否则无法区分训练慢和读取慢。
4. **真实介质上的 Store-P 必须实测**，不能拿 Mac/NVMe 数字外推到 WSL。
5. **继续小资源正确性闭环，完整模型/1M 只是最终性能 gate。**

---

# 23. 第十九轮系统性思考（Session 33：Track A 通用懒加载数据流落地）

## 23.1 本轮坐标更新

| 层 | 状态 |
|---|---|
| `LiveETStore` / `LiveETView` 从 `run_phase0.py` 提炼 | ✅ 已落地 `src/qwen35_ple/live_store.py` |
| `LiveETDataset` / `IterableDataset` 兼容流 | ✅ 已实现窗口级迭代 |
| control / shuffle / worker 分片 | ✅ 已实现 |
| DataLoader 多 worker | ✅ 每个 worker 自动重开 Store（pickle 支持） |
| per-batch fetch 时间、总读取量 | ✅ `LiveETBatch` + `FetchStats` |
| WSL Store-P / 1M 性能闭环 | ✅ 已跑 p4view + Python 懒加载；serving/完整模型仍待 |
| StorePool / 线程安全句柄 | ✅ `StorePool` / `ThreadLocalStore` |

## 23.2 本轮完成

- qwen35-ple `src/qwen35_ple/live_store.py`：
  - `FetchStats`（windows/tokens/rows/unique_rows/fetch_seconds/cache_hits）
  - `LiveETStore`（rowids-only、懒加载、reset_stats、context manager、pickle 重开 Store）
  - `LiveETView`（lazy slice/permuted/subset）
  - `LiveETViewStore`（Store-P 物化视图读取器，支持 padded slot）
  - `LiveETBatch`（tokens+e_t+start+fetch_seconds+rows）
  - `LiveETDataset`（IterableDataset 兼容、control/shuffle/worker_id/num_workers/max_windows）
- `run_phase0.py` 改为导入统一模块。
- 新增 `scripts/run_live_et_dataset_smoke.py`。
- 新增 `scripts/bench_store_vs_view.py`（Store-I vs Store-P A/B 骨架）。
- 新增 `scripts/bench_lazy_windows.py`（逐窗口懒加载，1M/CSV/percentile）。
- 新增 `engramdb.pool.StorePool` / `ThreadLocalStore`，`Database.fetch` 改用池。
- 新增 `tests/test_live_store.py`（9 tests），`python_wheel_smoke.py` 加 StorePool。
- WSL p4view：Store-P 20k/100k A/B，8t 约 22M rows/s。
- WSL Python 懒加载：1M Store-P 约 23.9s。
- qwen35 README 增加三行接入示例和本机/WSL 数据。

## 23.3 关键坑

1. **PyO3 `Store` 不可 pickle**：
   - DataLoader 多进程会失败；
   - 解决：`LiveETStore` 保存目录/分片/行数/宽度并在 worker 中重开。
2. **`__len__` 与迭代不一致**：
   - 修复为 `(n - seq_len)//step + 1`，保留 tiny-sequence 单窗口 fallback。
3. **macOS spawn 需要 `if __name__ == "__main__"`**。

## 23.4 Track A 退出标准检查

- [x] 任意实验脚本三行接入 live-store：`LiveETStore` + `LiveETDataset` + `for batch in dataset`
- [x] 不再全量加载 e_t
- [x] 支持 direct iteration / DataLoader / control / sharding / metrics
- [x] WSL Store-P A/B + WSL 1M 懒加载 CSV 已跑
- [ ] 仍缺：完整模型训练 real/control/3-seed、serving A/B、Store-P 访问序视图端到端

## 23.5 下一轮最高优先

1. **Track C 完整版**：用 `LiveETDataset` 在 WSL 跑真实模型 1M token real/control/3-seed，并把 fetch 时间与 loss 一起记录。
2. **Track B 纵深**：构建访问序 Store-P 视图、多线程批量预取，验证端到端训练/推理路径。
3. **Track D/E**：Store 连接池接入服务（已落地基础）、vLLM/SGLang/llama.cpp serving A/B、live-store smoke 入 CI/nightly。

---

# 24. 第二十轮系统性思考（Session 34：从中继试跑到真正端到端）

## 24.1 终极目标（不变）

> **让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何小模型、训练器、推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。**

三条不变验收轴：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 模型端到端差距 ≤5%；CPU 小模型 ≥50 tok/s（配 MTP 冲 100）；参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录嵌入式 + 可服务化；manifest 可校验；Arrow 零拷贝；引擎薄 adapter 不改上游 |
| C. 科学契约 | 真实 PLE + 固定输入 + 冷热标注 + CSV/阈值；bit-exact 有 golden/官方类双保险 |

## 24.2 本轮（v0.2.10 + WSL/Store-P/懒加载）坐标更新

| 层 | 状态 |
|---|---|
| Track A 通用懒加载数据流 | ✅ 完成：`LiveETStore` / `LiveETView` / `LiveETDataset` / `LiveETViewStore` |
| Track B 存储面 A/B | ✅ 初闭环：WSL p4view + Python 懒加载均跑通；仍缺语义访问序映射 |
| Track C 完整实验 | ⚠️ 只完成 I/O 基准，未完成真实模型 1M real/control/3-seed loss 对比 |
| Track D 服务化 | ⚠️ `StorePool` / `ThreadLocalStore` 已落地；vLLM/SGLang/llama.cpp A/B 未做 |
| Track E 工程稳定 | ⚠️ README/门禁/版本已推进；WSL 复现脚本、CI nightly、跨仓 golden 对齐未完成 |
| v0.2.10 | ✅ 已发布并推送 |

## 24.3 本轮关键发现

1. **Store-I 随机读是唯一真正的存储瓶颈**：
   - WSL 20k token Store-I 懒加载 22.4s；
   - 100k Store-P 懒加载 1.86s；
   - 1M Store-P 懒加载 23.9s。
   - 结论：磁盘优先不是问题，Store-I scatter 才是问题。
2. **Store-P 把 16 次散读折叠成 1 次定长读，收益约两个数量级**：
   - 本机 100k Store-I 60.5s vs Store-P 0.58s；
   - WSL p4view 8 线程 Store-P 22M rows/s vs Store-I 1.4M rows/s。
3. **访问序是关键**：
   - Store-P 顺序 1M 约 7.1s（本机）；
   - permuted/control 三 seed 17.2–17.9s（约 2.4× 惩罚）。
   - 说明“物化视图”只是第一步，“按访问序排布/调度”才是最终性能形态。
4. **多 worker 可行**：
   - `LiveETViewStore` pickle 重开 View 后，`DataLoader(num_workers=2)` 在 WSL 可跑。
5. **连接管理已初步产品化**：
   - `StorePool` / `ThreadLocalStore` 进入 Python API，`Database.fetch` 默认走池。
6. **完整模型实验仍然缺失**：
   - 所有数字都是“读取基准”，不是模型 loss / tok/s；
   - 不能把 I/O 快误认为科学结论。

## 24.4 本轮新增技术债

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V123 | 没有 token/rowid → Store-P slot 的语义映射 | 训练无法直接使用 Store-P，只能用 raw slot 基准 | ✅ 已实现 `qwen35_ple.slot_index.SlotIndex` + `--slot-index-out` + `run_phase0 --store-p-slot-index` |
| V124 | 访问序 Store-P 视图与调度未端到端 | 控制/随机访问惩罚约 2.4× | ✅ 已实现 access-order view 构建 + `LiveETViewStore(access_order=True)` / `LiveETDataset(access_order=True)` / `run_phase0 --access-order` |
| V125 | 没有完整模型 1M real/control/3-seed 实验 | 无法判断 PLE 嫁接是否真正有增益 | 用 LiveETDataset 驱动真实模型，输出 loss + fetch 时间 CSV |
| V126 | WSL 全量 pytest 存在 golden 漂移（1 个失败） | 跨仓正确性防线被削弱 | 固定 engram-peft 版本/重建 golden，或记录已知失败 |
| V127 | vLLM/SGLang/llama.cpp serving A/B 未做 | 产品形态未验收 | 实现薄 adapter + 真实小模型 A/B |
| V128 | 懒加载基准未进正式门禁/CI | 性能回归不可检测 | 将 20k/100k/1M 指标固化成 CSV 阈值脚本 |
| V129 | StorePool 尚未与 LiveET/训练 DataLoader 深度集成 | 连接生命周期仍偏基础 | 提供 per-thread pool + wait/borrow 统计 |
| V130 | Arrow IPC 路径未在本地/WSL 验证 | 零拷贝契约未闭环 | 安装 pyarrow 跑 `view_read_arrow` / `fetch_arrow` 端到端 |
| V131 | WSL 复现环境未脚本化 | 换机后需要手工装 engramdb/qwen35 | 写 `scripts/wsl_repro.sh` 固定版本与本仓代码 |
| V132 | 未规划 WSL 全表 Store-P 构建策略 | 320M gram 全表构建/校验可能耗时数小时 | 分批增量构建 + manifest/校验 + 可续跑 |

## 24.5 借鉴矩阵

| 来源 | 借什么 | 明确不借 | 为什么不冲突 |
|---|---|---|---|
| **DuckDB / SQLite** | 嵌入式、单目录、直接文件访问、零拷贝返回 | 不借 SQL/事务/查询优化器 | 我们只做“确定性记忆表存储”，不引入数据库语言 |
| **PyTorch IterableDataset / DataLoader** | 流式窗口、worker sharding、collate | 不借训练循环/sampler 策略 | 我们只提供存储侧数据源 |
| **HuggingFace Datasets streaming** | 大数据集不落内存、按需 map、分片、可复现 | 不借 NLP 处理 | 我们处理的是 PLE 特征流，不是文本 |
| **Arrow / IPC** | 列式零拷贝、schema、块传输 | 不借查询引擎/执行器 | 用于 Store-P 输出和跨进程边界 |
| **DiskANN / Milvus / FAISS** | 冷热分层、顺序化访问、批量预取、I/O 调度 | 不借 ANN/向量检索算法 | 只借 I/O 布局与批量策略 |
| **RocksDB / FDB** | 不可变段、checksum、批量顺序读 | 不借 LSM/事务/分布式 | 保持静态大表+视图语义 |
| **vLLM / SGLang** | continuous batching、异步 prefetch、指标 | 不借引擎调度/模型执行 | 我们只做存储侧 prefetch 与 reader |
| **llama.cpp / GGUF** | mmap、offset 直读、主机侧 tensor 组装 | 不借量化/图执行 | Store-P 可作为 mmap 数据源 |
| **XMemTransfer / Memory Grafting** | target-side reader、训练预算、实验方法 | 不搬它们的表结构 | 我们只借实验设计，不重复造记忆表 |
| **engram-peft / PEFT** | 训练接口、adapter、配置桥接 | 不借核心训练内核 | 我们以薄 adapter 接入 |
| **io_uring / Linux AIO** | 异步批量 I/O、多队列 | 不借具体引擎实现 | 用于 Rust/PyO3 底层 read path |

## 24.6 后续开发计划（按“先实证、再放大”）

### Phase 0：把“读取快”变成“实验能跑”
- [x] 实现 rowid-tuple → Store-P slot 语义映射与 manifest。
- [x] 实现 access-order Store-P 视图构建 + LiveETDataset 访问序调度。
- [ ] 用真实模型在 WSL 跑 1M token real/control/3-seed，同时记录：
  - 每窗口 fetch 时间、总读取量、cache/unique；
  - val loss / PPL / QA log-likelihood。
- [ ] 形成正式实验结果文档，并决定 PLE 嫁接是否继续放大。

### Phase 1：把基准变成门禁
- [ ] 固化 20k/100k/1M Store-P 懒加载 CSV + 阈值脚本。
- [ ] 固化 WSL 复现环境脚本（Python/engramdb/qwen35 版本、路径、命令）。
- [ ] 修复 WSL 全量 pytest 的 golden 漂移或明确记录已知失败。
- [ ] 将 live-store smoke 和 StorePool smoke 纳入 CI/nightly。

### Phase 2：把存储面推进到服务面
- [ ] `StorePool` 与 LiveET/DataLoader 深度融合，提供 borrow/wait 统计。
- [ ] Store-P mmap/PageReader 给 vLLM / SGLang / llama.cpp 的薄 adapter。
- [ ] 真实小模型 serving A/B：内存 vs Store-I vs Store-P，输出 tok/s + 尾差。
- [ ] Arrow IPC 端到端验证，替换 base64/JSON 大 payload。

### Phase 3：产品化收口
- [ ] WSL 全表 Store-P 分批构建与校验。
- [ ] 三仓库版本/README/retest/CI 完全同步。
- [ ] 根据完整模型结果决定是否进入 5M–20M token 阶段。

## 24.7 本轮纪律

1. **磁盘优先，不是“全量内存”的替代品而已**：Store-P/访问序才是磁盘优先的完全体。
2. **I/O基准不是实验结论**：只有真实模型 loss / tok/s 才能支持科学判断。
3. **所有性能结论必须包含**：介质、冷热、并发、访问序、CSV/阈值。
4. **跨仓正确性靠版本固定 + golden**，不能靠“本地能跑”。
5. **先打通端到端最小闭环，再谈放大**：先 1M 真实实验，再 5M/20M。
6. **每次发布前 release gate 必绿；版本只走 bump.sh。**

---

# 25. 第二十一轮系统性思考（v0.2.11：P0 语义索引落地 + 发布）

## 25.1 终极目标（不变）

> 让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何小模型、训练器、推理引擎都能廉价使用的**磁盘优先存储基础设施**——像 DuckDB 之于分析数据库。

三条验收轴不变：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 端到端差距 ≤5%；CPU 小模型 ≥50 tok/s（MTP 冲 100）；参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录嵌入式 + 可服务化；manifest 可校验；Arrow 零拷贝；引擎薄 adapter 不改上游 |
| C. 科学契约 | 真实 PLE + 固定输入 + 冷热标注 + CSV/阈值；bit-exact 有 golden/官方类双保险 |

## 25.2 本轮坐标更新

- ✅ **v0.2.11 已发布**：EngramDB Python 新增 `SlotIndex`（rowid-tuple → Store-P slot 语义索引）。
- ✅ **P0 代码部分完成**：V123 通用语义索引、V124 access-order 自动调度。
- ✅ **release gate 全绿**：fmt / clippy / workspace tests / 真表 bench / PyO3 / C ABI / smoke / decode baseline。
- ⚠️ **仍未完成科学闭环**：V125 真实模型 1M real/control/3-seed 实验仍是唯一阻隔“是否继续放大”的入口。

## 25.3 本轮发现/新增技术债

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V133 | `SlotIndex` 当前是纯内存二进制索引（`[N,16]×8B` + 排序副本），无法直接扩展到 320M 全表 | 真表全量语义索引会吃掉数十 GB 内存 | 改为 mmap/磁盘排序段/block index，或让 view manifest 直接携带 slot 解析所需的最小信息 |
| V134 | `SlotIndex` 在 EngramDB 与 qwen35-ple 各有一份实现 | 两仓行为可能漂移，维护成本翻倍 | 以 EngramDB 为 canonical，qwen35-ple 仅 re-export 或薄包装 |
| V135 | `engramdb view build` 原生 CLI 尚未输出/更新 slot index manifest | 用户从存储侧拿不到语义索引，只有 qwen 侧 builder 写了 | 将 slot index 生成/校验并入 `engramdb view build`，写 manifest 并在 `view verify` 验证 |
| V136 | access-order 自动调度只有机制，没有正式 A/B 基准与门禁 | 无法证明“按槽排序+窗口调度”在真表上的收益，也无法防回归 | 新增 `--access-order` 对照 CSV 阈值：naive vs sorted，固定 seeds/冷热/并发 |
| V137 | EngramDB Python 的 `SlotIndex` 依赖 numpy，但发布门禁环境未安装，只能以 `SlotIndex=None` 降级 | 功能在轻量环境不可见，且依赖声明与实际不一致 | 要么把 numpy 写入真正 runtime dependency 并在 release 环境预装，要么把 SlotIndex 做成显式可选子模块 |
| V138 | `LiveETDataset(access_order=True)` 会重排窗口顺序 | 对训练窗口顺序敏感的实验可能产生意外 | 明确文档/参数命名：`access_order` 只承诺 I/O 顺序，窗口重排用独立 `schedule_windows` 或在实验协议中声明 |
| V139 | 两仓 SlotIndex/access-order 无交叉 contract test | 无法自动发现语义或边界行为漂移 | 增加跨仓 smoke：同一 keys 文件 → EngramDB SlotIndex 与 qwen SlotIndex 输出一致 |

## 25.4 借鉴矩阵（本轮增量）

| 来源 | 借什么 | 明确不借 | 为什么可共存 |
|---|---|---|---|
| **DuckDB / SQLite** | 单目录、manifest、嵌入式、零拷贝 | 不借 SQL/查询优化器 | 我们只做“确定性点查 + 物化视图/索引” |
| **RocksDB / FDB** | 不可变 segment、checksum、manifest 原子切换 | 不借 LSM 写路径/事务/分布式 | 我们的表只读静态，只借“静态文件 + 可校验 index” |
| **Lucene / Roaring / FAISS IDMap** | 磁盘侧排序 key→offset 索引、block index、mmap 只读 | 不借 ANN/倒排/近邻 | 我们也是“静态 key→slot”查表，可以用类似磁盘索引思想 |
| **DiskANN / Milvus** | 冷热分层、顺序化访问、批量预取 | 不借向量图/过滤 | 只借 I/O 布局和缓存层次 |
| **PyTorch DataLoader / HF Datasets** | 流式窗口、worker 分片、可复现 seed | 不借训练循环 | 我们只提供数据源 |
| **vLLM / SGLang** | continuous batching、prefetch、指标暴露 | 不借引擎调度 | 我们提供存储 reader/adapter |
| **llama.cpp / GGUF** | mmap、offset table、warm_table | 不借量化图执行 | 只借冷启动/顺序预读 |
| **engram-peft / PEFT** | adapter、配置桥接 | 不借核心训练内核 | 我们只做薄接入 |
| **Arrow / IPC** | schema、列式零拷贝、块传输 | 不借查询执行器 | 用于 Store-P 输出和服务边界 |
| **Linux io_uring / AIO** | 批量异步 I/O、有界提交 | 不借具体引擎 | 底层 IO 优化，与上层语义正交 |

## 25.5 后续开发计划（重新排布）

### Phase A：把科学闭环补上（✅ 已完成）
- [x] WSL 真实模型 1M real/control/3-seed，输出 loss/PPL（fetch 时间另见 WSL lazy 基准，不在本次 JSON 中）。
- [x] 根据结果做 Go/No-Go：real 2.8167 < control 2.8738 < no-reader 2.9896，建议 Go 进入 5M–20M。
- [x] 结果已固化：`qwen35-ple/docs/phase-a-1m-result.md`。

### Phase B：把语义索引做成产品级
- [x] 以 EngramDB 为 canonical 整合 `SlotIndex`，qwen 只 re-export（已优先使用 EngramDB canonical，保留本地轻量 fallback）。
- [ ] `engramdb view build --slot-index` 原生生成 + manifest 更新；`view verify` 校验（已记录 `keys_out`，SlotIndex 可从 manifest 构建）。
- [x] 设计磁盘侧/block index，支持全表 320M 而不常驻内存（`DiskSlotIndex`：分桶 + 流式构建 + LRU + `build_from_keys_file`）。
- [x] 跨仓 contract test：同一 keys → 两个 SlotIndex 输出一致。

### Phase C：把性能变成门禁
- [x] access-order naive vs sorted 基准 + CSV 阈值（`bench_access_order.py --synthetic` 已入 CI；真实表阈值待 WSL）。
- [x] 20k/100k/1M 懒加载基准固化（`bench_lazy_windows.py --synthetic` 已入 CI；真实表 CSV 待 WSL）。
- [~] WSL 复现脚本 + golden 对齐 + live-store/StorePool smoke 入 CI（`wsl_repro.sh` 已加）。

### Phase D：服务化与全表
- [~] StorePool 与 LiveET/训练 DataLoader 深度融合 + wait/borrow 统计（`StorePool.stats()` 已加，LiveET 深度集成待做）。
- [ ] Arrow IPC 真实验证。
- [ ] vLLM/SGLang/llama.cpp serving A/B。
- [x] WSL 全表 Store-P 分批构建、断点续跑、校验（`build_full_store_p_batch.py` + `--keys-stream`）。

### Phase E：治理
- [ ] 明确 EngramDB Python runtime dependencies；SlotIndex 要么显式依赖 numpy，要么可选子模块。
- [ ] 三仓版本/README/CI 完全同步。
- [ ] 发布流程增加“新功能必须先有 contract/bench 门禁”的规则。

## 25.6 本轮纪律补充

1. **“能跑”不等于“可扩展”**：P0 语义索引在 1M 级可用，但全表必须走磁盘/block index。
2. **跨仓单一事实源**：同一概念不要在多个仓库各维护一份生产实现。
3. **性能结论必须有对照**：新增调度机制必须配 naive baseline + CSV/阈值。
4. **轻量环境可降级**：纯 Python 可选能力不得阻塞核心导入。
5. **科学实验仍是最终裁判**：在真实模型 loss/tok/s 出来之前，所有 I/O 优化都只是“候选基建”。

---

# 26. 第二十二轮系统性思考（Phase A 完成 + 磁盘索引落地 + 下一阶段）

## 26.1 终极目标（不变）

> 让 DeepSeek Engram / Qwen PLE 这类确定性哈希 n-gram 记忆表成为任何小模型、训练器、推理引擎都能廉价使用的**磁盘优先存储基础设施**——像 DuckDB 之于分析数据库。

三条验收轴不变：

| 轴 | 验收 |
|---|---|
| A. 性能契约 | 真实 PLE 端到端差距 ≤5%；CPU 小模型 ≥50 tok/s；参与开销 ≤5%；字节放大 ≤2× |
| B. 形态契约 | 单目录嵌入式 + 可服务化；manifest 可校验；Arrow 零拷贝；引擎薄 adapter 不改上游 |
| C. 科学契约 | 真实 PLE + 固定输入 + 冷热标注 + CSV/阈值；bit-exact 有 golden/官方类双保险 |

## 26.2 本轮坐标

- ✅ **Phase A 科学闭环完成**：WSL 1M real/control/3-seed → real < control < no-reader，Go。
- ✅ **v0.2.11 发布**，SlotIndex / access-order / manifest keys_out 已进入产品。
- ✅ **DiskSlotIndex 落地**：分桶磁盘索引、流式构建、LRU、`build_from_keys_file`。
- ✅ **全表批式构建工具**：`build_full_store_p_batch.py` + `--keys-stream` + 断点/校验。
- ✅ **合成性能门禁**：access-order / lazy-window 已入 CI。
- ✅ **原生 SlotIndex CLI**：`slot-index build|verify` + `view build --slot-index` / `view verify --slot-index`，并支持 Python v2 读取。
- ⚠️ 仍缺：Watch-level 真表全量验证、Arrow/serving、golden 修复。

## 26.3 本轮新增/更新技术债

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V140 | Phase A 结果使用 Store-I live-store，未用 Store-P/access-order 复跑 | 无法证明磁盘路径不改变科学结论 | 用 Store-P + slot-index + access-order 重跑 1M 三臂，输出 loss + fetch timing |
| V141 | DiskSlotIndex 尚无 320M 级真表构建/查找实测 | 只验证了小规模正确性，未验证规模性能 | WSL 上跑 10M/100M/320M 构建、磁盘放大、单查延迟、LRU 命中 |
| V142 | DiskSlotIndex 每个 bucket 一个文件 | 320M × 16k+ buckets 会产生大量小文件/目录项 | 评估单文件 + offset table，或原生 Rust 索引 |
| V143 | qwen35-ple 仍保留本地 SlotIndex fallback | 长期双实现漂移 | 把 fallback 也收敛为“仅测试用”，生产入口统一 EngramDB |
| V144 | ~~EngramDB CLI 仍未原生生成/校验 slot index~~ | ✅ 已闭环：`slot-index build|verify`、`view build --slot-index`、`view verify --slot-index`，Python 可读 v2 FNV 格式 | |
| V145 | Phase A 输出未记录 fetch timing | 科学结论与存储性能未同址 | Phase A2 直接记录 loss + fetch/wall 到同一 JSON |
| V146 | WSL 全量 pytest golden 漂移未修复 | 跨仓正确性防线弱 | 固定 engram-peft 版本或重建 golden，纳入 CI |
| V147 | CI 只有 synthetic 性能门禁 | 不能防真表性能回归 | 增加 nightly 真表 CSV 阈值 job |
| V148 | 大量新功能未发布 | 用户拿不到 DiskSlotIndex / batch builder | 发 v0.2.12 并同步三仓文档 |

## 26.4 借鉴矩阵（本轮增量）

| 来源 | 借什么 | 明确不借 | 为什么可共存 |
|---|---|---|---|
| **RocksDB / LevelDB SSTable** | 排序 key→offset 文件、block index、不可变段 | 不借 LSM 写放大/compaction/事务 | 我们的表只读，适合借用“静态排序索引文件”思想 |
| **Cassandra / Bigtable** | hash/range 分桶、局部性 | 不借分布式/副本 | 磁盘索引用分桶控制单查 IO |
| **SQLite / DuckDB B-tree** | 磁盘页索引、mmap 随机读 | 不借 SQL | 只借用“索引页+目录”的工程路径 |
| **LMDB / BoltDB** | 只读 mmap、B+tree、事务可选 | 不借 write transaction | 可作为原生磁盘索引实现参考 |
| **FAISS IDMap / DiskANN** | 静态 ID→offset、盘上顺序块 | 不借 ANN | 用于磁盘 slot 索引的块布局 |
| **HF Datasets / Parquet row-group** | 分块元数据、可流式构建 | 不借数据格式 | 用于大批量 batch builder 的元数据设计 |
| **ClickHouse / DuckDB columnar** | 不可变文件集、manifest、原子替换 | 不借列式查询 | 用于静态大表产物组织 |
| **vLLM / SGLang / llama.cpp** | 预取、指标、mmap/offset 直读 | 不借引擎 | 继续作为服务面接入对象 |

## 26.5 后续开发计划（重新排布）

### Phase A2：把科学结论钉在两层存储上
- [ ] 用 Store-P + slot-index + access-order 重跑 1M real/control/no-reader 3 seeds。
- [ ] 在同一 JSON 记录每窗口 fetch time、wall、rows、unique。
- [ ] 确认 Store-P 结果与 Store-I 一致，形成双路径科学结论。

### Phase B2：磁盘索引真表验证与产品化
- [ ] WSL 10M/100M/320M DiskSlotIndex 构建 + 查找基准。
- [ ] 评估单文件/offset table，或原生 Rust DiskSlotIndex。
- [x] `engramdb view build --slot-index` + `view verify --slot-index`。
- [ ] 移除 qwen 生产 fallback，统一 EngramDB canonical。

### Phase C2：真表门禁 + golden
- [ ] WSL 真表 access-order naive vs sorted CSV 阈值。
- [ ] WSL 真表 20k/100k/1M lazy CSV 阈值 + nightly CI。
- [ ] 修复 WSL golden 漂移并纳入 CI。

### Phase D2：服务化与 Arrow
- [ ] Arrow IPC 在 WSL/本地真实验证。
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] WSL 全表 Store-P 实际构建 + DiskSlotIndex + batch 全链路。
- [ ] StorePool 与 LiveET/DataLoader 深度集成。

### Phase E2：发布与治理
- [ ] v0.2.12 发布：DiskSlotIndex、`--keys-stream`、batch builder、StorePool stats。
- [ ] 三仓版本/README/CI 同步。
- [ ] 真表性能门禁纳入发布 gate。

## 26.6 本轮纪律补充

1. **科学结论必须双路径复现**：Store-I 和 Store-P 都要跑，不能只信一种存储路径。
2. **索引规模必须实测**：DiskSlotIndex 只有在 320M 级构建/查找数据出来后才算完成。
3. **所有新存储功能必须给盘放大/构建耗时/查找延迟**。
4. **生产入口单一事实源**：不允许两仓各维护一份“正式”实现。
5. **合成门禁只能防回归，不能替代真表门禁**。

---

# 27. 第二十三轮系统性思考（Session 37：原生 SlotIndex CLI + serving 架构规划）

## 27.1 本轮坐标

- ✅ 原生 SlotIndex CLI 闭环：`slot-index build|verify` + `view build --slot-index` / `view verify --slot-index`。
- ✅ Python DiskSlotIndex 支持 v1 / v2，并可直接生成 v2（FNV-1a 64）。
- ✅ 新增 `scripts/bench_disk_slot_index.py` 全表基准工具。
- ✅ 完成 vLLM / SGLang / PleMemory / TargetReader / Bundle 架构可行性分析。
- ⚠️ DiskSlotIndex 仍未跑 320M 全表实测。
- ✅ Serving 基础落地：`PleMemory` / `PleSequence` / `PleSequenceStore` / `BundleManifest` / `TargetReaderRegistry`。
- ✅ Serving 模块可选、不 torch 导入：`ple_math` 纯 Python，`ple_memory` / `bundle` / `target_reader` 可按需加载。
- ✅ 通用 Engine Adapter：`PleMemoryAdapter` / `TargetReaderHook` / vLLM-SGLang 注入别名已落地。

## 27.2 本轮新增/更新技术债

| # | 债 | 处置 |
|---|---|---|
| V149 | 无 `PleMemory` / `PleSequence` 统一抽象 | EngramDB 新增通用 serving 层 |
| V150 | 现有 vLLM/SGLang 插件只替换 embedding，不注入 target reader | 新增通用 reader 注入 adapter |
| V151 | 无 per-sequence 状态管理协议 | `PleSequence` + state store 协议 |
| V152 | 无通用 reader checkpoint / bundle 协议 | `TargetReaderRegistry` / `Bundle Manifest` |
| V153 | Arrow IPC / serving A/B 未验证 | 真表 Arrow + 引擎 A/B |
| V154 | v0.2.12 未发布 | 真表门禁后发布 |
| V155 | CI 只有 synthetic 门禁 | 真表 nightly CSV 阈值 |
| V156 | 高级 serving 模块与核心依赖未隔离 | serving 层可选子模块 |

## 27.3 后续开发计划

### Phase B2：磁盘索引真表验证与产品化
- [x] 本机 1M/10M DiskSlotIndex v3 构建/校验/lookup 基准（10M: build 135s, verify 87s）。
- [ ] WSL 100M/320M DiskSlotIndex 全表长跑。
- [x] 实现单文件/offset table：`data.bin` + `offsets.bin`（v3），Rust/Python 双向兼容。
- [x] `engramdb view build --slot-index` + `view verify --slot-index`。
- [ ] 补 `view build --slot-index` 真实表 e2e。

### Phase S1：PleMemory / PleSequence
- [x] `PleMemory`：统一 Store / View / SlotIndex 读取。
- [x] `PleSequence`：per-sequence history + `current_e_t()`。
- [x] `PleSequenceStore`：continuous batching 的 per-sequence 状态容器。
- [x] 纯 Python/torch 单元测试，不依赖 qwen。

### Phase S2：TargetReader Registry + Bundle
- [x] `engramdb.target_reader`：注册 + 加载协议。
- [x] `engramdb.bundle`：manifest + 路径解析 + schema version。
- [x] 不实现任何 qwen reader。

### Phase S3：通用 Engine Adapter
- [x] 通用 layer wrapper / forward hook（`PleMemoryAdapter` / `TargetReaderHook`）。
- [x] per-sequence state store（`PleSequenceStore`）。
- [x] 先纯 PyTorch，再 vLLM / SGLang（通用 adapter + `install_vllm_target_reader` / `install_sglang_target_reader`）。

### Phase S4：Arrow / 服务 / 真表门禁 / 发布
- [x] Arrow IPC 真表验证（`scripts/real_arrow_smoke.py`）。
- [x] serving A/B（`scripts/bench_serving_ab.py`，合成 + 真表）。
- [x] 真表 CSV/阈值门禁（`scripts/real_perf_gate.py` + release gate 集成）。
- [x] v0.2.12 发布。

## 27.4 本轮纪律

1. EngramDB 核心保持“确定性记忆表存储”，不做 SQL / ANN / 推理引擎。
2. Serving 层是可选高层模块，不得阻塞核心导入。
3. 所有磁盘索引/scale 结论必须有真表实测。
4. 新协议必须版本化。
5. qwen35-ple 由另一 agent 负责，EngramDB 只提供通用协议和存储能力。

> 完整计划/发现/尝试/踩坑/完成/未完成见 `docs/archive/round-37-full-summary.md`。

# 28. 第二十六轮系统性思考（Session 40：终极目标、新债与后续路线）

## 28.1 终极目标

让 EngramDB 成为 DeepSeek Engram / Qwen PLE n-gram 记忆表的**事实标准磁盘优先存储底座**：

- 存储正确：真实 320M 级表 Store-I / Store-P byte-identical。
- 性能达标：Store-P ≥4M 等效行/s；真实引擎 serving 差距 ≤5%；DiskSlotIndex 1M/10M/100M/320M 全表实测。
- 产品化：单目录、manifest、版本化、PyO3 为 Python 唯一下发路径。
- 生态一致：EngramDB 是 rowid→slot、PleMemory、Bundle、TargetReader 的唯一 canonical；qwen35-ple / engram-peft / vLLM / SGLang / llama.cpp 只做薄 adapter。
- 可演进：格式、索引、bundle 全部版本化，任何新能力必须有真表数据。

## 28.2 本轮新增技术债

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V157 | ~~`PleMemoryAdapter` 真表 torch 热路径仅约 1.6K–2K tok/s~~ **已重测：计算路径在免费介质上仅 11.6–198 µs/token（batch 1/4/64/1000），故 500–625 µs/token 是被 I/O 主导，不是 rowid** | 原处置方向误诊 | **改为：先测真表绝对值（需 128 分片），I/O 侧由 Store-P 折叠承接（§32.3, 7.69 µs/token）**；见 `probes/serve_ple_ab_session42.md` §2.3 |
| V158 | DiskSlotIndex v3 对 cache 敏感，verify/查询需高 cache | 大表内存和读放大不稳定 | 评估 block index / hash 均匀性 / Rust lookup API |
| V159 | 真实 20M keys 含重复 rowid tuple | 下游误用单槽 lookup 可能拿错记录 | 固化 `lookup` 代表槽与 `lookup_all` 全部槽契约 |
| V160 | Python 双桥：PyO3 + ctypes fallback | API 不完整、双维护、误导 | Python 发布只走 PyO3；C ABI 仅 C/C++ 外部用 |
| V161 | 发布流程出现“先 tag 后修 CI / 强推 tag” | release 可重复性差 | 先 CI 全绿再 tag，禁止无验证强推 |
| V162 | 通用 Engine Adapter 已落地，但无真实 vLLM/SGLang 模型级 A/B | 无法证明 serving 目标 | 真实引擎 A/B 出 CSV 阈值 |
| V163 | 真表门禁只在本地 release gate | 无法防远程回归 | self-hosted / WSL nightly 真表门禁 |
| V164 | `view build --slot-index` 缺真实表 e2e | 生产路径未闭环 | 补真实表 e2e |
| V165 | 100M/320M DiskSlotIndex 全表长跑未完成 | 规模结论不完整 | WSL 稳定环境跑全表 |

## 28.3 后续计划

### Phase R1：生产路径收敛
- [ ] Python 发布路径 PyO3-only；ctypes fallback 降级为源码/C/C++ 用途。
- [ ] `PleMemoryAdapter` / `PleMemory` 热路径下沉 Rust/PyO3。
- [ ] PleSequence/PleSequenceStore 与真实 PyTorch module 集成验证。

### Phase R2：磁盘索引产品化
- [ ] WSL 稳定环境 100M/320M DiskSlotIndex build/verify/lookup。
- [ ] 评估 block index、offset table、hash 均匀性、原生 Rust lookup。
- [ ] 补真实表 `view build --slot-index` e2e。
- [ ] 固化重复 rowid tuple 的 lookup 语义契约。

### Phase R3：真实引擎 serving
- [ ] 真实 vLLM 注入 PleMemoryAdapter/TargetReaderHook 并 A/B。
- [ ] 真实 SGLang 替换 reader/target-reader 并 A/B。
- [ ] 输出 tok/s、延迟、与原生路径差距 ≤5% 的 CSV。

### Phase R4：真表门禁与发布纪律
- [ ] 真表 nightly/self-hosted runner：Arrow + serving + DiskSlotIndex。
- [ ] release 流程：CI 全绿 → tag → release，禁止无验证强推。
- [ ] 三仓 README/版本/协议同步。

### Phase R5：生态 canonical 化
- [ ] qwen35-ple 移除本地 fallback，统一使用 EngramDB canonical。
- [ ] engram-peft 通过 Bundle / PleMemory 接入。
- [ ] vLLM / SGLang / llama.cpp 薄 adapter 统一协议。

## 28.4 借鉴矩阵（本轮）

| 来源 | 借什么 | 不借什么 | 怎么帮我们接近目标 |
|---|---|---|---|
| DuckDB | 嵌入式、目录即库、manifest、Arrow IPC 零拷贝 | SQL/查询引擎 | 确立单目录可嵌入产品形态 |
| SQLite | 文件格式版本化、schema 迁移、稳定性 | SQL/通用事务 | 用于 manifest/bundle 版本纪律 |
| RocksDB/LevelDB | 不可变排序段、block/offset index、bloom | LSM/compaction/写放大 | 用于静态 DiskSlotIndex block index |
| LMDB/MDBX | 只读 mmap、单文件、零拷贝 | 写事务/通用 KV | 用于 Store-P/索引单文件 mmap |
| Arrow/Parquet | Arrow IPC、chunk metadata、流式写 | 查询引擎 | 与训练器/引擎的数据契约 |
| Cassandra/Bigtable | hash/range 分桶、局部性 | 分布式/副本 | 用于 DiskSlotIndex 分桶 |
| vLLM | 自定义 op、CUDA graph splitting、pinned staging、async H2D | 不复制推理 | 指导 engine adapter 与真实验收 |
| SGLang | Rust reader、io_uring、页缓存、异步 H2D | 不复制引擎 | 指导低层 reader 替换与 Rust 集成 |
| llama.cpp | lazy mmap、tensor read、backend 接口 | 不复制 GGUF/推理 | 指导 Store-P 文件格式与 C ABI |
| Transformers/HF | module hooks、cache state、lazy loading | 训练内核/模型 | 用于 PleSequence/PleMemoryAdapter |
| engram-peft | adapter、patch、互操作契约 | 不重复训练逻辑 | 作为消费方与 contract test |
| qwen35-ple | reader、checkpoint、真实实验载体 | 不接管 reader 实现 | 作为消费方，EngramDB 只提供存储/协议 |
| Redis/Memcached | LRU、预取、统计遥测 | 不做通用 KV | 用于 cache 与 serving 指标 |

## 28.5 本轮纪律

1. Python 只以 PyO3 为发布路径；C ABI 只服务 C/C++。
2. 所有 serving/engine 结论必须有真实引擎 A/B 数字。
3. 所有 DiskSlotIndex 规模结论必须有大表实测。
4. 先 CI 全绿再 tag，禁止无验证强推 release tag。
5. 重复 rowid tuple 语义必须版本化、显示化。
6. 存储核心保持轻量，serving 层可选。
7. 跨仓单一事实源：EngramDB 是 canonical，其他仓只做消费方 adapter。

> 完整版见 `docs/archive/round-40-full-summary.md`。

---

# 29. 第二十七轮系统性思考（Session 41：终极目标再校准 + 收敛式开发计划）

> 触发：DeepSeek-V4.1-Flash 技术报告解读（见 `docs/v41-engram-analysis.md`）改变了外部地形的判断，
> 同时项目自身出现"计划面 >> 验证面"的失衡。本轮回答四个问题：终极目标是什么、计划怎么排、
> 借鉴什么且不冲突、如何更稳。

## 29.1 终极目标：愿景保留，但必须补一个**可证伪的内核**

现状（§1 / §28.1）的表述是**愿景**，不是**可证伪的论断**：

> "让 Engram/PLE 类确定性哈希记忆表成为任何小模型与推理/训练框架都能廉价使用的磁盘优先存储基础设施。"

它无法被任何单次测量推翻，因此**任何活动都能自称"在通往目标的路上"**——这正是下面 §29.2 失衡的根因。

**保留愿景，在其下补一条唯一的核心论断：**

> **核心论断（北极星内核）**
> 在一块消费级 NVMe 上，把 GB~百 GB 级的 n-gram 记忆表放在磁盘/宿主内存，
> 使**每 token 的 Engram 相关开销落进 5% 端到端预算**，
> 且该预算**单机、可复现、有门禁**。

### 29.1.1 把"≤5%"反推成可测预算（本轮最重要的产出）

> ⚠️ **Session 43 更正 —— 先读这一段再读下面的推导。**
> **500 µs 是「带假设的推导」，不是测量值**：它假设 100 tok/s = 10 ms/token，
> 而这个分母**从未被测量过**。Session 42 实测 CUDA graph 把分母改变了
> **7.3–7.8×**（0.8B 代理模型上 2.27–2.92 ms/token ⇒ 5% 只有 **113–146 µs**），
> **这个倍数比我们做过的任何存储优化都大**。
> 因此「195.9 µs 达标」**只在 eager 分母下成立**。
> 裁判已在 **§36.1 改写为调度可行性条件** `t_read ≤ τ(L_ple)`，
> 并化简为 **`L* = ⌈t_read / 单层时间⌉`（§36.2）** ——
> 用它可以当场判定：**Qwen 差一层、V4.1 余量 3.5×、
> 没有线程池则 V4.1 装不下**（`scripts/lead_layer_budget.py` 可复跑）。
> **本节自此仅作为历史推导与 `real_perf_gate.py` 的现行阈值保留。**

"GPU A/B 差距 ≤5%"之所以挂了十几轮 ⏳，是因为它**在现有硬件上不可测**。
但 5% 可以**反推**成一个本地可测的绝对量：

| 端到端 decode | 每 token 时间 | 5% 预算 | Qwen（16 行/token） | V4.1（48 行/token） |
|---|---|---|---|---|
| 50 tok/s | 20 ms | **1000 μs** | 62 μs/行 | 21 μs/行 |
| 100 tok/s | 10 ms | **500 μs** | 31 μs/行 | 10 μs/行 |

**保守门禁取 500 μs/token。** 于是"≤5%"被分解为一句本地可验证的话：

> **48 行/token（V4.1）或 16 行/token（Qwen）的全部读取 + 反量化 + adapter 开销，
> p99 ≤ 500 μs/token。**

这条论断不需要 125B/552B 模型即可测——它就是本轮之后一切工作的**唯一裁判**。

### 29.1.2 现有实测对照预算

| 路径 | 实测 | 折算 | 对照 500 μs |
|---|---|---|---|
| 真表 `Store.fetch` | 23.1K–45.8K tok/s | 22–43 μs/token | ✅ 11–23× 余量 |
| 真表 `PleMemory` | 51.9K–62.5K tok/s | 16–19 μs/token | ✅ 26–31× 余量 |
| 真表 `PleMemoryAdapter`（torch） | 1.6K–2.0K tok/s | **500–625 μs/token** | ❌ **正好压线/超出** |

> ⚠️ **上表第三行已在 §29.8 被推翻**：1.6K–2.0K tok/s 是冷缓存 + 10.5 MB 窗口的测量假象。
> 修复方法学后的真值是 **20.4 μs/token**。§29.8 记录了完整修正。

**结论：存储层已经在预算内 11–31 倍；唯一超出预算的是 Python/torch adapter 热路径。**
存储层继续做微优化，对北极星的边际贡献接近于零。

### 29.1.3 门禁的覆盖缺口（实测）

`scripts/real_perf_gate.py` 只读两个字段：

```
MIN_PLE_MEMORY_RPS = 5_000.0     # = 200 μs/token
MIN_STORE_FETCH_RPS = 5_000.0    # = 200 μs/token
```

而 `scripts/bench_serving_ab.py` **已经产出** `ple_memory_adapter.tokens_per_s`（:114），
门禁**却不读它**。

> **门禁阈值是对的（200 μs 比 5% 预算更严），门禁的覆盖是错的：
> 它守住了已经通过 11–31× 余量的两条路径，放过了唯一不达标的那条。**

这是本轮最可执行的一条发现，也是 Phase 1 的全部理由。

---

## 29.2 诚实盘点：三处失速（有数字）

| 指标 | 数值 | 说明 |
|---|---|---|
| session 数 | 40（约 2 周） | 迭代极快 |
| roadmap 行数 / 文档总量 | 2416 行 / 520 KB | README 仅 24 KB |
| 技术债 ID | **165 条**（V1–V165） | 平均每轮净增 ~6 条 |
| roadmap 复选框 | **已关 61 / 未关 195（76% 未关）**（Session 40 基线） | 净关闭率为负 |
| 验收表（README §3.2） | 2 ✅ / **3 ⏳** | 同 3 条挂了十几轮 |

**三处失速：**

1. **北极星不可证伪** → 见 §29.1，已给出修正。
2. **计划速度 > 验证速度**：76% 未关 + 每轮净增债务 → 债务表已从"待办"退化为"日记"，
   它记录的不是"要做什么"，而是"曾经想到过什么"。
3. **最高优先项依赖不具备的硬件**：125B / 552B 模型在现有机器上无法加载，
   于是"端到端 ≤5%"永久 ⏳。**把不可测的东西挂在验收表上，比不写更糟**——
   它让所有其他工作都可以自称"在为它做准备"。

> **纪律判断：这个项目的工程纪律（bit-exact、真表探针、release gate）是最大资产，
> 不需要改。需要改的是投资组合管理：把"记录"变成"关闭"。**

---

## 29.3 目标模型分叉：Qwen PLE vs V4.1 Engram

V4.1 报告（`docs/v41-engram-analysis.md`）带出一个必须显式决策的分叉：

| | Qwen3.8-Flash-Next PLE | DeepSeek-V4.1-Flash Engram |
|---|---|---|
| 规模 | 51.2B / 单层 / 16 行/token | **196.6B / 双层 / 48 行/token** |
| checkpoint 布局 | 列主序 `[160, 320M]` → 16 路 scatter | **行主序 `[rows,256]`，出厂即 Store-P 形状** |
| Store-P 视图的价值 | **高**（修布局） | **≈0**（布局已最优） |
| 介质 | 本地 NVMe | **宿主内存 + RDMA + 存储集群** |
| 可测性（现有硬件） | 表 48 GB，**可测** | 权重 510 GB，**不可测** |

**决策规则（唯一裁判 = 一个季度内能否拿到真实测量）：**

- **产品默认保持 Qwen PLE**：它是当前唯一能拿到真表 + 真模型 + 真测量的目标；
  Store-P / DiskSlotIndex 的投资继续有效。
- **接口必须 V4.1-ready，但不得改变产品默认**：
  1. `EngramSpec` 泛化（2/3/4-gram、多 layer、DEAD mask）——`engramdb-keygen` 层；
  2. `ByteSource` 抽象（为 RDMA/宿主内存留位）——`engramdb-io` 层；
  3. per-module 放置与 deadline 调度——`PrefetchPlan` 层；
  4. **不为 V4.1 做视图物化**（checkpoint 已是紧凑行主序，做一遍纯属浪费 196.6 GB）。
- **切换触发条件**（满足任一即重新评估）：拿到 ≥256 GB 内存的机器 / 存储集群合作 /
  上游出现 V4.1 Engram 的真实部署需求。

> 这样处理，两条路线**不冲突**：Qwen 决定"默认产品与门禁"，V4.1 决定"接口形状与未来分层"。

---

## 29.4 收敛式开发计划：净关闭率 > 0

**总原则：每轮结束时，关闭条目数 ≥ 新增条目数。** 这是本轮唯一的流程硬约束。

### Phase 0：把北极星写成预算（1 个 session，无新增债务）
- [ ] 把 §29.1.1 的推导写入 `design.md`（含假设与出处）。
- [ ] 在 `real_perf_gate.py` 增加 **adapter 阈值**（先记录、后强制），并让门禁读取
      `ple_memory_adapter.tokens_per_s`；缺失即 FAIL（不允许"没测=通过"）。
- **退出标准**：门禁能对三条路径同时给出 PASS/FAIL，且当前 adapter 路径必然 FAIL。

### Phase 1：杀死 adapter 热路径（最高杠杆，1–2 个 session）
- [ ] rowid 生成 / history 维护 / fetch / FP8 反量化 全部下沉 PyO3（V157）。
- [ ] 目标：`PleMemoryAdapter` 从 **500–625 μs/token 降到 ≤50 μs/token**（10× 以上）。
- [ ] 验收：真表 + 真 torch 路径，`real_perf_gate.py` adapter 阈值转强制并全绿。
- **为什么是它**：这是唯一超出 5% 预算的环节，约为存储层快路径的 **12–30×**，
  且**完全可在本机测量**。做完这一条，为核心论断贡献的是数量级级别的进展。

### Phase 2：把"端到端"拆成"代理已闭环 + 真机待硬件"（1–2 个 session）
- [ ] 用现有 `engram-peft + TinyLlama` / `qwen35_cpu_decode_ab.py` 路径，
      把「内存表 vs 磁盘表」的 decode 相对开销固化成门禁 CSV。
- [ ] 明确标注这是**相对开销**曲线，**不是**绝对 tok/s 承诺。
- **退出标准**：README §3.2 的"端到端 CPU 小模型 decode"由 ⏳ 改为
      "**代理闭环 ✅ + 真机待硬件 ⏳（前置条件已写明）**"。

### Phase 3：真机决策（管理动作，非工程动作）
- [ ] 二选一，**不允许继续以 ⏳ 悬挂**：
      (a) 落实一台能放下 Qwen3.8-Flash-Next（FP8 ≈90 GB）的机器，做一次性真实 A/B；
      (b) 正式把"≤5% GPU A/B"降级为"未来工作"，写明前置硬件条件。
- **退出标准**：验收表里不再有"无前置条件的永久 ⏳"。

### Phase 4：冻结与整理（稳定性，可与 Phase 0 并行）
- [ ] **冻结点**：DiskSlotIndex 停在 v3，不再加版本；adapter 面在 Phase 1/2 完成前不加新引擎。
- [ ] **文档预算**：`roadmap.md` 只保留 §1（目标）+ 最近一轮 + 借鉴矩阵 + **未关闭**债务；
      历史轮次归档到 `docs/archive/`。理由：520 KB 文档的维护成本已经开始挤占验证时间。
- [ ] **债务表改造**：只列未关闭项；新增一条必须同时关闭一条。

---

## 29.5 借鉴矩阵 v2：用"分层"机制化地防止冲突

用户问题"从类似项目分别借鉴什么、可以不互相冲突"，本质是**缺少一张层归属表**。
冲突从来不是"借鉴了坏东西"，而是**把只适用于某层的技术泄漏到了别的层**。

### 29.5.1 层模型（防冲突的机制）

| 层 | 内容 | 可变性 |
|---|---|---|
| **L0** | key / rowid 语义（哈希、素数、压缩词表、pad/DEAD） | **模型定义，不可变** |
| **L1** | 物理布局（Store-I / Store-P / 单文件 / 分片边界） | canonical，版本化 |
| **L2** | 索引（SlotIndex / DiskSlotIndex / block index） | 可替换 |
| **L3** | 缓存与分层（T1/T2/T3、lifetime class、量化分配） | 自由 |
| **L4** | 传输（preadv / io_uring / mmap / RDMA / 宿主内存） | 自由 |
| **L5** | 调度（预取计划、批合并、deadline、投机预取） | 自由 |
| **L6** | 形态 / API（嵌入式 / PyO3 / server / Arrow） | 自由 |
| **L7** | 适配器（vLLM / SGLang / llama.cpp / engram-peft / qwen35-ple） | 自由 |

**负载标签**：**A** 训练/预处理（Zipf、吞吐）｜**B** 推理 prefill（token 已知、批）｜
**C** 推理 decode（无 lookahead、延迟）｜**D** rollout（常驻）。

> **铁律：任何借鉴必须声明 `(层, 负载)`；L0 与 L1 的 canonical 语义不得被任何借鉴改变。**
> 热度、顺序、压缩、传输策略**只能作用在 L2–L7**，永远不能回写 L0/L1。

### 29.5.2 借鉴矩阵（标注层与负载）

| 来源 | 借什么 | 层 | 负载 | **明确不借 / 禁止泄漏到** | 理由（多为实测） |
|---|---|---|---|---|---|
| DuckDB | 嵌入式、目录即库、manifest 原子换签、Arrow IPC | L6 | 全 | 不借列式/扫描引擎；**不把 server 语义带进 L1** | 嵌入与服务在 L6 分叉，共享 L1/L2 |
| SQLite | 文件格式版本化、schema 迁移 | L1/L6 | 全 | 不借 SQL/事务 | 版本纪律 |
| LMDB | 只读 mmap 单文件零拷贝 | L1/L4 | A/B | **C 负载不用被动 fault** | llama.cpp 13.1 faults/token 反面 |
| DiskANN | 冷数据顺序化、滑窗读 | L5 + L1(视图构建) | **A/B 仅** | **禁止用于 C** | chat 无局部性（4.75M gather 零同页命中） |
| Cassandra / Bigtable | 桶化、局部性、"热度"概念 | L2/L3 | A | **禁止把热度编进 L0/L1 键** | rowid 由模型哈希决定，canonical 不可重排 |
| RocksDB | block / offset index、bloom | L2 | 全 | 不借 LSM / compaction | DiskSlotIndex block index（但见 Phase 4 冻结） |
| Milvus | 不可变段 + seal / compact + 快照 | L1 | A | 不借向量 / 过滤 | 视图生命周期 |
| SGLang | 常驻 io_uring、页对齐、有界提交、GIL-free | L4 | 网络盘 / cgroup | **不作为本地 NVMe 性能项** | 已定案 0.97× / 0.94× |
| vLLM（PLE / disk） | 去重 / 排序 / 合段 + `fadvise(WILLNEED)` | L5 | B/C | 不借 CUDA graph 分段 | 我们产出"计划"，引擎侧消费 |
| vLLM（prefix KV） | block 分配 + 前缀哈希 + 长 TTL LRU | L1/L3 | C | 不借"通用 KV"定位 | P3-12 统一存储面时启用 |
| llama.cpp | **测量方法学**、C ABI | L4/L7 | A | **不借被动 fault 路径** | 只借其"实测数字文化" |
| **DeepSeek V4.1 §3.1.3** | 全 batch 一次预取、per-rank 行分片过滤、FP8+scale 直送 GEMM | L5/L2/L1 | A/B | 不借训练 All2All / 优化器 | 新证据，直接可借 |
| **DeepSeek V4.1 §2.4.2** | 宿主内存 + RDMA 预取、**per-module 分层**、DEAD mask | L4/L5/L0 | B/C | — | 新证据；L0 部分进 keygen |
| **推荐系统 Hybrid Embedding** | 大表服务分层、量化分配、碰撞控制理论 | L3 | **A 仅** | **禁止把 hot/cold 频次假设用于 C** | 已实测 top-1000 覆盖 <6% |
| Transformers / HF | module hook、cache state、lazy loading | L7 | B/C | 不借训练内核 | adapter 面 |
| Redis / Memcached | LRU、分档命中率遥测 | L3 | C | 不做通用 KV | lifetime class + 分级命中率 |

**一句话收敛（v2）：**
> 同域引擎（SGLang/vLLM/llama.cpp）教我们 **L4/L5/L7 的测量与工程纪律**；
> 数据库（DuckDB/SQLite/LMDB/Milvus/RocksDB/Cassandra/DiskANN）教我们 **L1/L2/L6 的形态与布局**；
> DeepSeek/Qwen 教我们 **L0 的语义精确**；
> 推荐系统教我们 **L3 的大表服务经济学，且只在训练负载（A）成立**。
> **四者作用在不同层，所以天然不冲突；冲突只发生在有人越过自己的层去改 L0/L1。**

---

## 29.6 稳定前进的四条纪律

1. **净关闭率 > 0**：每轮关闭数 ≥ 新增数。这是唯一能扭转 76% 未关的手段。
2. **一个端到端论断**：同时最多有 **1 个**未闭环的端到端断言；闭环前不开启第二个。
3. **门禁前置**：没有阈值写进 `real_perf_gate.py` 的工作**不算完成**；"没测" = FAIL，不是 PASS。
4. **文档服从代码**：文档预算封顶，roadmap 只保留当前态 + 最近一轮；历史归档。

## 29.7 本轮明确"停止做"的事

- **停止**给 DiskSlotIndex 加版本（v1/v2/v3 已足够，冻结在 v3）。
- **停止**为 V4.1 做 Store-P 视图物化（其 checkpoint 已是紧凑行主序）。
- **停止**在存储层做微优化（已低于预算 11–31×）。
- **停止**新增引擎 adapter，直到 Phase 1/2 完成（现有 vLLM/SGLang/llama.cpp 面已超出验证能力）。
- **停止**把不可测的指标留在验收表上。

> **轮次长文摘要已退役（Session 44）**：原先这里挂着
> `docs/round-41-full-summary.md`（待补）—— 它从未写出，而轮次摘要这一形态本身也已退役
> （见本文件 §0）。第 41 轮以后的真实进展在 `docs/session-log.md` 与本文 §33–§36，
> 活账在 §36.5。**「待补」这类条目不再保留：它不是状态，是没有承诺**（§36.6 第八条）。

---

## 29.8 Phase 0–2 执行结果（Session 41 收尾）

**一句话：Phase 0 与 Phase 2 按计划完成；Phase 1 的立论被实测推翻，改为交付了更值钱的东西
（可信测量 + 真实浪费的修复）。**

### 29.8.1 Phase 1 立论被推翻（本轮最重要的诚实记录）

§29.1.2 断言"adapter 500–625 μs/token，是唯一超出预算的环节，比存储层贵 12–30×"。
实测该断言**错误**，根因是两处测量方法学缺陷：

| 缺陷 | 现象 | 证据 |
|---|---|---|
| 单次、无 warm-up | 同一份**未改动**的代码：冷测 1,710 tok/s，热测 25,865 tok/s | **15× 离散** |
| rowid 只覆盖 `0..65535` | "真表"路径实际只读 51.2 GB 表中的 **10.5 MB 窗口**，从不触盘 | `min=0 max=65535` |

修正后（真表、warm、全表随机 rowid、5 次中位数）：

| 路径 | 修正前（README 记录） | 修正后 | 对 500 μs 预算 |
|---|---|---|---|
| `Store.fetch` | 23.1K–45.8K tok/s | 90K–105K tok/s（9.5–10.9 μs/token） | 46–53× |
| `PleMemory` | 51.9K–62.5K tok/s | 73K–85K tok/s（11.8–13.2 μs/token） | 38–42× |
| `PleMemoryAdapter` | **1.6K–2.0K tok/s** | **47.7K–49.1K tok/s（20.4–21.0 μs/token）** | **24×** |

→ **没有任何一条路径超出预算；"10× 加速"的目标本身不成立。**

> ⚠️ **§30.1 进一步推翻了本节的"24× 余量"**：上表全是 **warm（page-cache 命中）** 读数。
> 用从未读过的行重测后，冷态 adapter = **476.8 μs/token = 95% 预算**（Qwen，USB 介质）。
> 正确表述是"**代码路径**已在预算内；**介质路径**几乎没有余量"。详见 §30.1。

### 29.8.2 仍然真实存在的浪费（已修）

代码审查发现 `PleSequence.feed` 先调 `fetch_raw(rows)` 再调 `fetch_tensor(rows)`，
而 `fetch_tensor` 内部**再次** `_coerce_rows` + `fetch_raw` ——**每 token 批读两次盘、做两次
rowid 转换**。修复为"读一次、由同一 buffer 派生 e_t"（新增 `PleMemory.tensor_from_raw`）。

- 效果：**35.8 → 20.6 μs/token（1.7×）**（warm、同口径对比）
- 正确性：`feed(as_tensor)` vs `fetch_tensor` 逐位相等，多 chunk + history、raw 路径、
  adapter 端到端、`current_e_t()` 惰性路径、scale 路径 —— 6 项断言全绿

### 29.8.3 交付物

| 项 | 文件 | 状态 |
|---|---|---|
| 预算推导（5% → 500 μs/token） | `docs/design.md` §7.3 | ✅ |
| 门禁读取 adapter，缺失/报错即 FAIL | `scripts/real_perf_gate.py` | ✅ |
| 门禁决策逻辑单测（7 个失败分支） | `scripts/real_perf_gate_test.py` | ✅ |
| 基准方法学修复（warm-up/reps/median/随机 rowid） | `scripts/bench_serving_ab.py` | ✅ |
| 真表 serving 探针 | `probes/serving_ab_v041.json` | ✅ |
| 相对开销推导 + 门禁 | `scripts/overhead_budget_check.py` | ✅ |
| decode 代理可复现（含造模型脚本） | `scripts/make_tiny_qwen3.py`、`probes/decode_ab_proxy_baseline.csv` | ✅ |
| 门禁集成 | `scripts/gate.sh` | ✅ |

### 29.8.4 Phase 2 的诚实边界

`cpu_tiny_decode_ab.py` 替换的是**输入 embedding**，解码期每 token 只读 **1 行**；
而 Qwen PLE 读 **16 行**、V4.1 Engram 读 **48 行**。因此该代理
**只能作为集成正确性 + 下界证据，不能作为 Engram 开销证据**（实测 disk-raw 慢 5.4%，
且 LRU 情形落入噪声地板——旧 CSV 全是 0.0%，那个门禁此前实际上无法失败）。

真正的相对开销由**组件测量 + 算术推导**给出（`overhead_budget_check.py`）：

| 模型 | @50 tok/s | @100 tok/s |
|---|---|---|
| Qwen PLE（16 行） | 0.102% | 0.204% |
| V4.1 Engram（48 行，×3 外推） | 0.305% | 0.611% |

均远在 5% 之内。**这是"代理闭环 ✅ + 真机待硬件 ⏳"的完整含义**：
本地可证的已经证完，剩下的只差一台能放下 125B/552B 主干的机器。

### 29.8.5 残留缺口（诚实列出）

1. **`cargo clippy` / `cargo test` 本次未运行**：沙箱禁止写 `~/.cargo`，且本地 registry
   已被裁剪（缺 `indoc` 等），`--offline` 无法解析。**本次未改动任何 Rust 代码**，
   `cargo fmt --check` 通过。需在正常环境跑一次完整 gate 确认。
2. **真机端到端仍未做**（前置条件已写明，见 README §3.2）。
3. 旧 CSV `probes/cpu_tiny_baseline.csv` / `qwen35_cpu_baseline.csv` 是 WSL 产物，
   其 `slowdown` 全为 0.0%（噪声地板）——**该门禁目前无法失败**，建议后续按同一
   方法学重测或明确标注为"仅功能性"。

---

## 29.9 V4.1 支持决策与实施计划（Session 41 增补）

### 29.9.1 决策：分三层，只做现在能做的那两层

| 层 | 内容 | 现在做吗 | 理由 |
|---|---|---|---|
| **L0 语义** | keygen（压缩词表 / 4-gram / DEAD mask / 双模块 / 双素数表） | ✅ **现在做** | 只需 6.4 MB `tokenizer.json`，**不需要 202.8 GB 权重**；可本地逐位对拍 |
| **L1/L2 存储** | 任意行宽、多表、双数组记录（payload｜scale） | ✅ **顺带做** | 现有 `width`/`Database` 已参数化；零风险 |
| **L4/L5 部署** | RDMA / 宿主内存 `ByteSource`、per-module 分层、投机预取 | ❌ **明确推迟** | 无可测硬件；现在加进验收表就是重演 §29.2 的病 |
| — | 下载 510 GB 权重 | ❌ **不做** | 外盘仅剩 ~155 GB 放不下 |
| — | 为 V4.1 建 Store-P 视图 | ⚠️ **§30.3 已纠正为"要做"** | 本节原判断"视图价值≈0"是错的：行主序只保证**一行**一次读，不保证**一个 token 的 24 行相邻**。冷读下 V4.1 为 286% 预算，行折叠是必需项 |

**核心洞察（本轮已实测）**：**验证 V4.1 keygen 不需要权重。**
压缩词表只依赖 `tokenizer.json`（6.4 MB）。实测结果：

```
tokenizer vocab = 129280
compressed vocab size = 99092  (1.0s)
config expects        = 99092
MATCH: True
pad_id(2) -> compressed 2
```

即 L0 这一整层**廉价且可关闭**；昂贵的只有 L4/L5，而那正是当前测不了的。

### 29.9.2 已就绪的资产（全部支持 `--check` 校验）

| 资产 | 文件 | 状态 |
|---|---|---|
| 冻结压缩词表（129,280 项 u32-LE） | `refs/v41_token_map.bin`（517,120 B）+ `.json`（含 tokenizer sha256） | ✅ 已生成并校验 |
| 冻结素数 / 乘子 / 行数闭合 | `refs/v41_engram_constants.json` | ✅ 48 素数 + 8 乘子 |
| 官方参考实现（MIT） | `refs/v41_engram.py` | ✅ |
| 生成 / 校验脚本 | `scripts/gen_v41_token_map.py`、`scripts/gen_v41_engram_constants.py` | ✅ 均支持 `--check` |

**→ V4.1 keygen 的全部输入都已冻结；剩下的只是写 Rust。**

### 29.9.3 实施计划

#### Phase V0：keygen v2（可本地闭环，1–2 session）
- [ ] `EngramSpec` 泛化：多 layer、多 order（2/3/4-gram）、任意 heads、`pad_id`、
      DEAD mask、可选 `token_map`。
- [ ] Rust 实现与官方 `inference/engram.py` 逐 id 对拍。
- [ ] 验收：随机序列 + 图像 mask + 序列首部 + 跨 DEAD 边界，**零差异**。
- [ ] golden 进 CI，与现有 Qwen golden 并列。
- **退出标准**：`engramdb-keygen` 同时通过 Qwen 与 V4.1 两套 golden。

#### Phase V1：存储层参数化（零风险）
- [ ] 记录格式支持"payload｜scale 双数组"（为 264 B/行 铺路）。
- [ ] 用**合成** V4.1 表（10M 行 × (256+8) B，双 layer）跑通 build / fetch / latency。
- [ ] 验收：合成表 fetch 吞吐与字节放大率进 `probes/`。
- **明确不做**：任何全量视图物化。

#### Phase V2：推迟项（写明启动条件，不进验收表）
- RDMA / 宿主内存 `ByteSource`；per-module 分层与 deadline 调度；投机解码感知预取。
- **启动条件（满足任一）**：≥256 GB 内存机器 / 存储集群合作 / 上游出现 V4.1 Engram 真实部署需求。

### 29.9.4 边界与风险

1. **不要把 V4.1 端到端加进验收表**：510 GB 权重在现有硬件上不可测。
2. **不要在 Rust 复刻** Unicode 归一化或 numpy PCG64：两者都已冻结为常量
   （`v41_token_map.bin` / `v41_engram_constants.json`）。
3. token map 依赖 `tokenizers` 版本；sidecar JSON 记录了 `tokenizer.json` 的 sha256，
   `--check` 可检测漂移。**Unicode 表版本变化会改变压缩词表大小 → 触发整表重哈希，
   这是本项目最需要防的一类静默错误。**

---

## 29.10 「engram 层可训练」支持决策（Session 41 增补）

> 问题：如果不冻结参数，我们需要支持 engram 层的高效训练吗？有办法支持吗？

### 29.10.1 结论：**不作为目标；但有三处 EngramDB 形状的缺口值得补**

**决定性证据（本机已 clone 的 DeepSeek 开源栈）**：写路径在集群尺度上**已经被 DeepEP 解决**，
存储集群层**已经被 3FS 解决**。我们没有必须打的仗。

| 层 | DeepSeek 的开源答案 | 证据 |
|---|---|---|
| 热层：可写 + RDMA 共享的表 | **DeepEP `ElasticBuffer`** | `engram_write(storage)`（`[num_entries,hidden]` bf16，带 barrier）、`engram_fetch(indices)`（异步 RDMA，返回 hook）、`get_engram_storage_size_hint`（entry 32B 对齐 + scale-factor pack） |
| 存储集群层 | **3FS** | checkpointing / dataloader 随机读 / KVCache for inference；180 存储节点实测 6.6 TiB/s；MIT |
| 训练时的表 + 优化器 + 梯度路由 | V4.1 报告 §3.1.3 | engram-parallel 进程组、All2All、梯度回传 owner rank、Sinkhorn |

**所以"给 EngramDB 加写路径"= 用一个磁盘优先引擎去重做 DeepEP 的 RDMA buffer。**
这是一个我们赢不了、也不需要参与的竞争；而且会**替换**我们的产品而不是扩展它
（badge 布局 / 直接寻址 / 无 B-tree / 不可变段 / 无 WAL —— 全部建立在只读之上）。

### 29.10.2 三处真实的缺口（都保持不可变语义）

**缺口 A：DeepEP 目前 GPU-only 且 FP8 是 TODO。**
- `ElasticBuffer` 原文："currently GPU-only, with **CPU and mixed (GPU+CPU) backends on the roadmap**"
- `engram_write` 原文："**# TODO: support FP8**"
- 但它的 size hint 已经预留了 `num_sf_packs = ceil_div(hidden, 32)`（itemsize ≤ 1 时）——
  **正是 V4.1 的 per-32 E8M0 block scale 布局。**
- → EngramDB 已有 FP8 + block scale + 264 B/行 的实测积累。
  **最高杠杆的动作不是做竞争性存储，而是提供"格式 + 转换器/装载器"**，
  把我们的 FP8+scale 布局喂给 `engram_write`。这是补他们的 TODO，成本低，且是让我们
  与集群场景产生关联的桥。

**缺口 B：单机 / 无 RDMA 的 PEFT 场景。**
- `engram-peft` **已经训练 engram 参数**（benchmark：LoRA+Engram 约 547.7M 可训练），
  且**已经通过 `table_source="engramdb:store"` 消费 EngramDB**。
- 547M 的表放得进显存 → 该规模**不需要磁盘**。但 51B（Qwen）/196B（V4.1）的表在 1–8 卡上微调就需要。
- → 正确形态是 **"基表只读留在磁盘（我们的强项）+ 可训练 delta 在内存"**。
  这正是 PEFT 语义，也是只读保证从"限制"变成"资产"的场景。

**缺口 C：确定性寻址 → 优化器状态也可预取。**
- `engram-peft` 原文："Engram employs sparse lookup; **only a tiny fraction of parameters
  (approx. 1%) are active and receive gradient updates per step**"。
- 由于寻址确定性，**这个 active set 在 step 0 就已知** —— 所以可预取的不只是嵌入行，
  还有 **momentum 优化器状态行**与梯度 scatter 路由。
- V4.1 正是为削减优化器状态才弃 Adam 改 momentum+Sinkhorn（Adam 对 196B 约为 1.5 TB，
  momentum 约 786 GB），并把优化器状态跨副本分片。
- → **复用 `PrefetchPlan`，无需任何新存储语义。这是本轮唯一"新"的技术点。**

### 29.10.3 如果将来真要做写路径，架构约束

**只能是"不可变基表 + 追加式 delta + 周期 compaction"，不能是可写表。**
沿用借鉴矩阵里已有的 Milvus 段模式：

- canonical 基表保持不可变、直接寻址（读路径全部保证不变）；
- 训练更新写入独立 delta（按 rowid 键控，**只追加**，因此崩溃安全、无需 WAL/MVCC）；
- 读 = `base ⊕ delta`；
- 周期把 delta 压实进新基表快照，原子换签。

**诚实的边界**：该设计只在"每步 active set × 每行优化器状态 ≪ 表"时成立。
粗算：768M 行、1% active/step、256 维 fp32 momentum → **每步约 7.9 GB 优化器状态**。
NVMe（~3 GB/s）追不上；**宿主内存 / RDMA 才可行** —— 又回到缺口 A 的地盘。
因此 §29.10.4 的探针是先决条件，不是可选项。

### 29.10.4 决策与下一步

| 项 | 决定 |
|---|---|
| 把写路径设为项目目标 | ❌ 不做 |
| 可写 canonical 表 / WAL / MVCC | ❌ 不做 |
| 与 DeepEP 竞争热层 | ❌ 不做 |
| 补 DeepEP 的 **FP8 + scale 布局 / 转换器** | ✅ 建议做（补他们的 TODO，成本低） |
| engram-peft **复合 checkpoint**（只读基表 + delta）契约 | ✅ 建议做（延伸 `BundleManifest`） |
| 优化器状态预取计划 | ⏳ **先探针后决定** |
| delta-log / 版本化快照存储 | ⏳ 仅在出现真实稀疏更新负载时启动 |

**唯一能定案的探针**：
> 取 V4.1/Qwen 真实行宽，模拟"每步 1% active set"的 momentum 行读取，
> 测**每步优化器状态加载耗时**与可接受的 step time 之比。
> 判据：若该比值 > 20%，则磁盘分层不成立，缺口 C 关闭，只保留缺口 A/B。

---

# 30. 第二十八轮系统性思考（Session 41 续：冷读真相与下一步计划）

## 30.1 本轮最重要的发现：§29.8 的结论被自己的探针推翻

§29.8 写的是"adapter 20.4 μs/token，**24× 余量**，存储层已不是瓶颈，应停止微优化"。
**这个结论是错的**——它是**全 page-cache 命中**的读数：4096 token × 16 行 = 65,536 行
= **10.5 MB**，在 32 GB RAM 上完全驻留，从未触盘。

修正方法学后重测（每次运行**随机 base offset** 指向从未读过的行；三条路径使用**互不重叠**
的 rowid 切片，否则先测的路径会把行拉进缓存、后面全测成内存）：

| 路径 | **COLD** | WARM | 冷/热 |
|---|---|---|---|
| `Store.fetch` | 394.1 | 16.94 | 23.3× |
| `PleMemory.fetch_raw` | 427.0 | 21.57 | 19.8× |
| `PleMemoryAdapter` | **476.8** | 36.69 | **13.0×** |

单位 μs/token；Qwen 口径 16 行/token；4096 tokens；**USB SSD**。
→ 冷读约 **29.8 μs/行**。

**对照 500 μs/token 预算**：

| 目标 | 冷读 | 占预算 | 判定 |
|---|---|---|---|
| Qwen PLE（16 行/token） | **477 μs/token** | **95%** | ⚠️ 几乎没有余量 |
| V4.1 Engram（48 行/token，×3 投影） | **1430 μs/token** | **286%** | ❌ 超预算 |

> **结论反转**：存储路径**正是**瓶颈，而且余量比想象的薄得多。
> 项目此前的全部工程（badge 布局、Store-P 行折叠、页对齐、批量预取、8 线程 gather、
> NVMe 目标介质）从"已经赢了、可以停"变回**不可或缺**。

## 30.2 一个已经咬到我们三次的系统性缺陷

| 出处 | 读数 | 真实情况 |
|---|---|---|
| README §3.3 | adapter "1.6K–2.0K tok/s" | 冷启动 + 10.5 MB 窗口 |
| §29.8（本轮我自己） | "24× 余量" | page-cache 全命中 |
| README §3.1 | "NVMe 19.2M 行/s"、"USB 554K 行/s" | 标注为"半冷"；且多次运行共享 RNG 种子 → **自热** |

**根因是同一个：没有把"工作集是否超过缓存"当作测量的一等元数据。**

→ **制度化（本轮新增纪律）**：任何 IO 探针必须记录
`(工作集字节数, 机器 RAM, 是否冷启动, base_offset)`；不满足
"工作集 ≫ 页缓存 **或** 只读从未读过的行"的读数，必须显式标注为 **warm-only**。

## 30.3 下一步开发计划

### Phase A（最高优先）：把冷读钉死，并换介质复测
- [x] `bench_serving_ab.py` 支持 `--base-offset`、三路径互不重叠切片、cold/warm 双报。
- [x] `overhead_budget_check.py` 增加冷读预算段（WARN <2× 余量 / FAIL 超预算）。
- [ ] **在 NVMe 上复测**（本机只有 USB；WSL/Windows 机器有 NVMe）。
- [ ] 探针增加工作集/缓存态元数据字段（§30.2 制度化）。
- **退出标准**：NVMe 冷读 μs/行有数，且 Qwen 冷读占预算 **< 50%**。

> **为什么这是最高优先**：这一个数字决定北极星真假。
> USB 上 95% = 没有余量；若 NVMe 真有 README 声称的 35× 优势则约 3%。
> 两种结论导向完全不同的路线，而目前**两种都只是猜测**。

### Phase B：行折叠（Store-P）从"可选"回到关键路径
V4.1 是 48 行/token，冷读下 286% 预算 → **必须把每 token 的读次数折下来**。

- [ ] **纠正 §29.9.1 的错误论断**："V4.1 checkpoint 已是行主序 → 视图价值≈0"。
      行主序只保证"**一行**一次读"；它**不保证**"一个 token 需要的 24 行相邻"。
      按观测到的 4-gram 物化 24 行（24 × 264 B ≈ 6336 B）→ 24 次散读折叠为 1 次。
      **Store-P 对 V4.1 与对 Qwen 同样重要。**
- [ ] 合成 V4.1 表上实测：24 次散读 vs 1 次 6336 B 读（冷态）。
- **退出标准**：折叠比 ≥ 4×（保守），并把冷态数字进 probes。

### Phase C：V4.1 keygen v2（与 A/B 无依赖，可并行）
沿用 §29.9.3 Phase V0：`EngramSpec` 泛化 + 与官方 `engram.py` 逐 id 对拍。
输入已全部冻结（`refs/v41_token_map.bin` / `v41_engram_constants.json`）。

### Phase D：收敛（延续 §29.6）
净关闭率 > 0；不新增 adapter；文档预算不变。

## 30.4 借鉴矩阵增量

本轮**无新增来源**，新增一条**测量纪律**：
DiskANN / llama.cpp 的"冷数据"概念必须落到**工作集字节数**，"冷"不是一个可以随便写的形容词。
llama.cpp 的"实测数字文化"在本轮以反例形式再次生效——**我们自己的数字也需要同样对待**。

## 30.5 纪律（在 §29.6 四条之上新增第五条）

1. 净关闭率 > 0。
2. 同时最多 1 个未闭环端到端断言。
3. 门禁前置；"没测" = FAIL。
4. 文档服从代码。
5. **任何 IO 读数必须声明工作集大小与缓存态；warm-only 的读数不得用于带宽/介质结论。**

---

## 30.6 Phase A/B 执行结果（Session 41 收尾）

### 30.6.1 Phase A：介质数字已拿到（经 Windows 主机进 WSL）

**访问路径（修正 handoff §4 的过时地址）**：
`docs/handoff.md` 记的是 `minam@192.168.31.108`，实测 **No route to host**。
可用地址是 **`minam@100.78.250.122`**（Tailscale），且链路是
**ssh → Windows cmd.exe → `wsl.exe -e bash -lc` → Ubuntu（用户 `zeng`）**。
注意：cmd 不认 `;` 也不认 `\"`，直接拼命令会被吞；本轮用
**base64 转发脚本**绕过引号地狱（本地 helper `/tmp/wslrun.sh`）。

机器：`DESKTOP-VI1IC4Q`，8 vCPU，15 GB RAM，Python 3.12.3，
根文件系统 `/dev/sdd` ext4 1 TB（VHDX）。

**介质冷读实测**（16 GB 测试文件，每个偏移只读一次 → 保证冷）：

| 读法 | **WSL 原生 fs** | USB 外盘 | 倍数 |
|---|---|---|---|
| 冷随机 pread 160 B | **18.6 μs/read** | 153.6 μs | 8.3× |
| 冷随机 pread 2560 B | **9.9 μs/read** | 165.3 μs | **16.7×** |

→ 这是**单线程**口径：此口径下 USB 慢 **8–17×**（"35×"偏高）。
   但**并行口径下差距扩大到约 40×**（见本节末尾），因为 NVMe 能吃队列深度而 USB 不能。
   另外**读长几乎不影响成本**（NVMe 上 160 B 与 2560 B 同量级），
   说明成本由随机 IO 延迟主导——这正是折叠有效的依据。

**换算到预算（500 μs/token）** —— ⚠️ 下表是**单线程**口径，**已被本节末尾的并行口径取代**，
保留在此仅为展示"口径差异有多大"：

| 形态 | WSL/NVMe（单线程） | USB |
|---|---|---|
| Store-I 16 次散读（Qwen） | 160–300 μs = 32–60% | 480 μs = 96% |
| Store-P 折叠 1 次读（Qwen） | 10–19 μs = 2–4% | 28.4 μs = 6% |
| Store-I 48 次散读（V4.1） | 480–900 μs = 96–180% | 超预算 |

→ 基于此我曾写下"V4.1 的行折叠不是优化而是必需项"。**该结论已在下方被并行实测推翻。**

**⚠️ 我在这里先犯了一次错，记录下来**：早先用 **Python 线程**测同一件事，得到
"8 线程 48.9 μs/read 反而比单线程 9.9 μs 更慢"，并据此写下了
"NVMe 上并行只会加剧竞争、并行度默认值可能是错的"。

**这个结论是错的。** 用 **原生 C + pthread**（无 GIL、无 Python 每次调用开销）复刻
`read_records_parallel` 的完全相同语义后：

| 线程数 | 2560 B 冷读 μs/read | 聚合 MB/s |
|---|---|---|
| 1 | 7.8 | 330 |
| 2 | 2.0 | 1271 |
| 4 | 1.0 | 2478 |
| **8** | **0.8** | **3282** |
| 16 | 0.9 | 2752 |

| 线程数 | 160 B 冷读 μs/read | 聚合 MB/s |
|---|---|---|
| 1 | 2.2 | 74 |
| 4 | 0.8 | 197 |
| **8** | **0.7** | **244** |
| 16 | 0.7 | 215 |

→ **NVMe 上并行收益巨大（8 线程 ≈ 10×），单线程才是瓶颈**（同步 pread 只能有 1 个 IO 在飞，
被延迟绑死；多线程才能压出 NVMe 的队列深度）。
之前那个"并行更慢"纯粹是 **Python 线程的测量假象**——这正是 §30.2 那类错误的第四次复发，
而且又是**我自己**犯的。

**结论：`available_parallelism()`（WSL 上 = 8）是正确默认值**，不需要改；
16 线程略差，说明 8 已接近饱和。

> 🛑 **本节上面的 C 线程表与下面的"修正后预算表"已在 §30.7 被撤回。**
> 那个 C benchmark 在后续线程轮次**复用了同一批偏移**，测到的是 page cache 命中而非介质；
> 用 EngramDB 真实代码路径在真实冷数据上复测得到的是 **902 μs/token（180% 预算）**，
> 而不是下面表中的 11.2 μs。**以 §30.7 为准。**
>
> 🛑🛑 **再修正（Session 42）**：§30.7 的 902 μs/token 是 **WSL/VHDX** 上的数字，
> 而 WSL 也不是生产介质。**最终以 §31 为准**：原生 NVMe + 真实 PLE 数据实测
> **Qwen 16 行 204 μs（41%）、V4.1 48 行 604 μs（121%）@8 线程**。
> 下面整张"修正后预算表"（含"NVMe 8t ≈ 0.7 μs/read、约 40×、V4.1 只占 6.7%"）
> **全部作废**；真实倍率是 **2.4×**（§31.5）。

**修正后的预算表（NVMe、冷、8 线程）**：

| 形态 | NVMe 8t | 占 500 μs 预算 | USB（实测） |
|---|---|---|---|
| Store-I 16 次散读（Qwen） | **11.2 μs** | **2.2%** | 480 μs（96%） |
| Store-P 折叠 1 次读（Qwen） | 0.8 μs | 0.2% | 28.4 μs（6%） |
| Store-I 48 次散读（V4.1） | **33.6 μs** | **6.7%** | ~1440 μs（288%） |
| Store-P 折叠 2 次读（V4.1） | 1.6 μs | 0.3% | — |

**三个结论（取代上一版）**：
1. **生产介质确实快得多，而且在并行下差距更大**：NVMe 8t ≈ 0.7–0.8 μs/read
   vs USB ≈ 30 μs/read，**约 40×** —— README 长期宣称的"35× 介质拖累"**基本准确**。
2. **在 NVMe + 并行下，连 V4.1 的 48 行散读也只有 6.7% 预算**，
   上一版"96–180% 超预算"是基于**单线程**数字的误判，**已推翻**。
3. **行折叠（Store-P）从"V4.1 的必需项"降级为"余量项"**（6.7% → 0.3%）：
   仍然值得做，但**不再是 V4.1 能否服务的前提**。
   §30.3 Phase B 的修复本身依然正确（USB 上 1.5×→16.9× 是真实收益）。

> 方法学边界：本轮用的是 WSL 原生 fs 上的**合成 16 GB 文件**，不是真实 EngramDB 表。
> 它回答的是"介质冷读成本"，不是"EngramDB 端到端"。WSL 上的仓库副本已过期
> （Aug 30，无 `.git`/`scripts`/`data`），要在真表上复测需先同步仓库并构建 Linux 版。

### 30.6.2 Phase B：Store-P 折叠**已兑现 16.9×**（先发现失效，再修复）

在全表视图 `p4view-full-2560.bin`（20,000,096 槽 × 2560 B = 51.2 GB）上与 Store-I 同口径冷测。

**第一轮（修复前）——折叠几乎无效：**

| 路径 | 冷读 | 相对 |
|---|---|---|
| Store-I：16 × 160 B 散读 / token | 397–480 μs/token（24.8–30.0 μs/行） | 1.00× |
| Store-P：1 × 2560 B 记录 / token | 282–310 μs/token | **仅 1.35–1.5×** |

**根因定位（三步排除）：**

1. 先怀疑"每次 Python 调用开销" → 给 C-ABI 补了缺失的 `read_records` 后重测，
   **仍然 282 μs/条**，排除调用开销。
2. 再用裸 `os.pread` 隔离介质与读长：

   | 读法（冷，单次一 syscall） | 成本 |
   |---|---|
   | view 2560 B（51.2 GB 文件） | 304.0 μs |
   | shard 2560 B（400 MB 文件） | 165.3 μs |
   | shard 160 B（400 MB 文件） | **153.6 μs** |

   → **读长几乎不影响成本（160 B vs 2560 B 只差 1.1×）**，成本由随机 IO 延迟主导。
   所以"16 次读折叠成 1 次"在原理上应当接近 16×。
3. 对照 `Store.fetch`：64,000 行一次调用只需 **26 μs/行**，比裸 pread 单发（153.6 μs）快 6×
   → **`Store.fetch` 是并行的，而 `ViewReader::read_records` 是串行 `for` 循环。**

**真正的根因（两处）：**
- `crates/engramdb-io/src/view.rs` 的 `read_records` 是**串行 for 循环**，无并行提交；
- 且并行阈值沿用了 `gather_pp` 的 `len <= 1024` 走串行——**而 32~512 正是真实 serving 批大小**，
  于是真实场景 100% 落在串行路径上。

**修复**：`read_records` 改为 `std::thread::scope` 并行（新增 `read_records_parallel` 显式控线程数），
阈值从 1024 降到 **32**。

**第二轮（修复后）——折叠收益兑现：**

| 批大小（记录） | 修复前 μs/条 | 修复后 μs/条 |
|---|---|---|
| 32 | 287.3 | 63.7 |
| 64 | 273.1 | 40.2 |
| 128 | 272.8 | 34.9 |
| 256 | 271.9 | 31.2 |
| 512 | 283.4 | 29.9 |
| 1024 | 286.2 | 30.5 |
| 4096 | 29.9（已并行） | **28.4** |

**端到端结论（冷态，USB）**：

| 路径 | 冷读 | 相对 Store-I |
|---|---|---|
| Store-I：16 × 160 B 散读 / token | 480.2 μs/token | 1.00× |
| Store-P：1 × 2560 B 记录 / token（并行） | **28.4 μs/token** | **16.9×** |

→ **§30.3 Phase B 的退出标准（折叠比 ≥ 4×）已达到（16.9×）。**
→ 同时把 §30.1 的"Qwen 冷读 95% 预算"改写为：
   **走 Store-P 时约 6% 预算**（28.4 / 500）。**存储面回到预算内，且这次是冷态读数。**

**正确性验证**（新增改动必须自证）：
`read_records` 与逐条 `read_record` 在 n ∈ {1, 5, 33, 100, 1000, 5000} 下**逐字节相同**；
重复索引、越界拒绝（rc=-3）、空输入均正确；`cargo test -p engramdb-io -p engramdb-keygen` 13 项全绿；
`cargo fmt --check` 与 `cargo clippy -D warnings` 干净。

**同时补齐的接口缺口**：`read_records` 此前**只存在于 PyO3**，C-ABI 桥没有
（新增 `engramdb_view_read_records`），ctypes 用户此前**根本无法使用折叠路径**。

### 30.6.3 一个方法论发现：本轮全部数字来自 **ctypes 回退路径**

```
_USING_PYO3 : False
_USING_CTYPES: True
engramdb._engramdb → ImportError: symbol not found: __Py_DecRef
```

`python/engramdb/_engramdb.so` 是在 **Python 3.12** 下构建的，本机 `python3` 是 **3.9**，
因此加载失败、回退到 ctypes C-ABI。后果：

- **本 session 所有性能数字（含 §30.1 的冷读）都是 ctypes 路径**，不是 PyO3。
- `read_records` 等 PyO3 专有方法在 ctypes 下不可见（这正是 §30.6.2 的直接原因）。
- 与 roadmap **V160**（"Python 发布只走 PyO3"）形成现实冲突：
  **声明的唯一路径在本机根本跑不起来。**

→ 待办：把"PyO3 能否在开发机默认 Python 上加载"加进门禁（一条 import 断言即可）；
   或在 `portable-dev.md` 明确记录构建 PyO3 所需的 Python 版本。

### 30.6.4 Phase B 修正后的结论

**Store-P 是正确方向，而且现在已被冷态实测证实——但兑现它需要一个此前缺失的并行读路径。**

| | 结论 |
|---|---|
| 设计是否正确 | ✅ 是。折叠把 16 次冷随机读变成 1 次，实测 **16.9×** |
| 数据面是否就绪 | ✅ 是。51.2 GB 全表视图已在盘（`p4view-full-2560.bin`） |
| 读取路径是否完整 | ❌ **此前不完整**：串行 `for` + 1024 阈值 = 真实批大小全部串行；C-ABI 无此接口 |
| 现状 | ✅ 已修：并行化 + 阈值降到 32 + C-ABI 补齐 + 13 项 Rust 测试与逐字节对拍全绿 |

**对交付目标的影响**：
- Qwen 冷读从 **480 μs/token（96% 预算）** 降到 **28.4 μs/token（6% 预算）**；
- 这把 §30.1「存储路径正是瓶颈」的结论**再次反转回"存储面在预算内"**——
  但这次是**冷态、含折叠、经并行读路径**的读数，而不是 §29.8 那种缓存命中的假象。
- §30.3 Phase B 退出标准（折叠比 ≥ 4×）**已达成**。

**仍未闭环**：并行度在 NVMe 上的正确取值（§30.6.1 末），以及真实 EngramDB 表在 WSL 上的复测
（需先同步仓库 + 构建 Linux 版）。

---

## 30.7 WSL 复测：**WSL/VHDX 不是生产介质的有效代理**（Session 41 收尾，重大修正）

### 30.7.1 做了什么

1. 源码打包（470 KB，仅 crates/python/scripts/refs）→ scp 到 Windows → 在 WSL 解包到 `~/engramdb-linux`；
2. 在 WSL 上 **成功构建** `libengramdb_c.so`（C ABI，6.7s）与 `lib_engramdb.so`（PyO3，12.1s）；
3. 造与真表**同几何**的合成表：128 shard × 2,500,012 行 × 160 B = **51.2 GB**；
4. 用 `Store.fetch`（真实代码路径）测冷读。

> 附带收获：WSL 的 Python 是 **3.12.3**，与本仓库 `.so` 的构建版本一致
> → **本 session 第一次跑通 PyO3 路径**（Mac 上因 3.9/3.12 ABI 不符一直回退 ctypes）。

### 30.7.2 结果

| 项 | 值 |
|---|---|
| PyO3 冷读 | 938.6 μs/token（58.66 μs/行） |
| ctypes 冷读 | 938.0 μs/token（58.63 μs/行） |
| **PyO3 vs ctypes** | **无实质差异**（冷读由 IO 主导，绑定层不是瓶颈） |
| 冷读 4 轮（每轮全新行） | 926 / 909 / 894 / 895 μs/token，**中位数 902** |
| 同批行重读（warm） | 13.33 μs/token（**68× 冷热比**） |

**中位数 902 μs/token = 500 μs 预算的 180%**（Qwen 16 行）；V4.1 外推 541%。

**与 Mac/USB 对照**：

| 介质 | 冷读 μs/行 | warm μs/token |
|---|---|---|
| Mac + USB 外盘（真表 51.2 GB） | **30.0** | 36.7 |
| WSL + VHDX（合成 48 GiB） | **55.9–57.9** | 13.3 |

→ **WSL/VHDX 的冷随机读比 Mac 上那台 USB 外盘还慢约 2×。**

### 30.7.3 必须撤回的上一轮结论

§30.6.1 里我写的"NVMe 8t ≈ 0.7 μs/read、比 USB 快约 40×、V4.1 只占 6.7% 预算、
行折叠降级为余量项"——**全部撤回**。原因：那个 C benchmark 在**后续线程轮次复用了同一批偏移**，
测到的是 page cache 命中，不是介质。改用"每轮新鲜偏移"重测后结果**自相矛盾**
（1 线程 2.2 μs/read，比首次全冷的 278 μs 还快 126×），说明 **Windows 宿主机缓存在起作用**，
该路径上的多线程读数**不可信**。

**成立且未撤回的部分**：
1. `gather_pp` 的并行是**真实有效**的：首次全冷单线程 278 μs/4 KB 页
   → EngramDB 8 线程 56 μs/行，约 **5× 并行收益**；
2. **PyO3 与 ctypes 冷读等价** → 此前所有基于 ctypes 的**冷读**结论仍然有效
   （这也解释了为什么 §30.1 的冷读发现不受后端影响）；
3. §30.6.2 对 `read_records` 的修复本身正确——**USB 上 1.5×→16.9× 是实测收益**。

### 30.7.4 Phase A 的真实状态：**仍未闭环**

`docs/handoff.md` 把 WSL 记为"Linux 语义测试 + GPU"用机；本轮证明它**不能**充当
"生产介质 = NVMe"的代理——**WSL2 的虚拟化存储栈（VHDX + virtio + Windows NTFS + 宿主缓存）
本身引入的延迟超过了介质差异**，再叠加动态扩展的 48 GB VHDX 与 2015 年的 i7-6700。

**要闭环 Phase A，需要以下之一**：
- 一台**裸机 Linux + 原生 NVMe**；
- 或直接在 Windows 上绕过 WSL，用原生 API 测同一块盘；
- 或接受"生产介质数字暂缺"，并把 V4.1 的行折叠**保持为关键路径**（保守默认）。

**保守默认的含义**：在拿到可信的原生 NVMe 冷读数字之前，
**不要**用任何"NVMe 很快"的假设去下调行折叠、预取、页对齐的优先级。

### 30.7.5 清理

远端 `~/store128`（48 GB 合成表）、`~/engramdb-linux`、测试二进制均已删除；
WSL 磁盘回到 7%。合成表约 1 分钟可重建（几何与命令已记录在 §30.7.1）。

---

## 31 Phase A 闭环：原生 NVMe 上的真实读数（Session 42）

§30.7.4 列出的三个出路线里，本轮走了第一条：**一台裸机容器 + 原生 NVMe**。
Phase A（"生产介质冷读"）**至此闭环**，且结论与 §30.7 的保守默认一致。

### 31.1 机器与口径

| 项 | 值 |
|---|---|
| 主机 | `ssh -p 28326 root@connect.nmb1.seetacloud.com`（AutoDL 容器，`autodl-container-63b64b9474`） |
| CPU | 2× Xeon Gold 6430，64C/128T，NUMA×2 |
| 内存 | 宿主 1007 GB；**本容器 cgroup `memory.max` = 120 GiB** |
| 存储 | `/root/autodl-tmp` = XFS on **`/dev/md0` = RAID1 over 2× Samsung MZQL27T6HBLA-00A07（PM9A3 7.68 TB 企业级 NVMe，PCIe 4.0 x4，16 GT/s）** |
| 数据 | **真实 Qwen3.8 PLE 行**：`qwen35-ple/qwen38-rows`，65 shard × 2,500,012 行 × 160 B = 25 GB |
| 内核/发行版 | Ubuntu 22.04.5 / glibc 2.35 / 5.15.0-94 |
| 工具链 | 现场装 rustup 1.98.1（tuna 镜像）+ rustfmt + clippy；gcc 11.4 |
| 宿主负载 | load average ≈ 11–13（128 核，来自其他租户；对 4 KiB 随机读延迟影响可忽略，见 §31.3 稳定性） |

**数据来源说明**：该目录的 `manifest.json` 仍写着 `num_shards: 128`，但**实际只剩 65 个**
（其余 63 个在本次之前已被删除腾空间）。这不影响本案：逐 shard 几何完全一致，
且 §31.4 证明**每行恰好占一个独立 4 KiB 页、页间零共享**，所以表总量不进入访问模式。

### 31.2 方法学升级：机械自校验（本轮两次拦住假读数）

§30.5 纪律 #5 说"必须声明工作集大小与缓存态"。本轮证明**光有条文不够**——
因为本机 cgroup 上限 120 GiB ≫ 表体积，"表比内存大所以是冷读"在这里**根本不成立**；
`drop_caches` 又被容器拒绝（`CapEff` 无 `cap_sys_admin`，实测 `Permission denied`）。

于是把纪律做成**机械约束**，写进工具本身：

1. **双射游走 + 轮间不相交切片**：`key(i) = (a·i + b) mod N`，`gcd(a,N)=1` ⇒
   `i` 不同则 `key` 必不同。第 r 轮取 `i = r, r+stride, r+2·stride, …`，
   **一次运行内绝不重读任何 key**。
2. **按文件 `fadvise(DONTNEED)`** 替代 `drop_caches`：每轮冷读前把 65 个 shard 的干净页逐出。
3. **结尾冷热自校验 + 自动裁决**：把最后一轮的 keys 立刻重读一遍。
   **比值 < 5× ⇒ 打印 `>>> VOID` 并以退出码 3 终止**，任何数字不得引用。

**这个闸门在本轮开火两次，两次都是真问题**：

| 次 | 现象 | 真相 |
|---|---|---|
| 1 | 真实 PLE 数据上 cold 1.14 μs/行，自校验比值 1.00× | 该数据 9-09 写入、9-11 被另一个项目读过，**26 GB 全在 page cache 里**。读数不是介质。 |
| 2 | io_uring 路径 cold 0.48 μs/行，`pages=0`，自校验 1.13× | `read_many` 每次都返回 Err，**一页都没读**（见 §31.7）。 |

> 第 1 次如果没有闸门，我会第三次报出"NVMe 比 USB 快 40×"这类数字。
> 第 2 次暴露了我自己 harness 的缺陷：**静默把"全失败"当成"很快"**——
> 已修成"0 页即 FATAL 退出码 4 + 打印首个错误"。

### 31.3 独立交叉验证：裸设备微基准（C，不经过 EngramDB）

先写一个**不依赖 EngramDB** 的 C 探针（`/root/nvme_probe.c`，24 GiB 真写文件、
20000 次读/配置），用两条互相独立的路径测同一件事：

| 配置 | mean μs/4 KiB 读 | p50 | p99 | 有效吞吐 |
|---|---|---|---|---|
| **O_DIRECT** c=1 | **77.28** | 72.66 | 97.76 | 50.5 MB/s |
| **buffered，全新偏移** c=1 | **78.35** | 75.50 | 93.01 | 49.8 MB/s |
| O_DIRECT c=8 | 80.18 | 76.16 | 119.84 | 387.6 MB/s |
| buffered 冷 c=8 | 84.14 | 81.68 | 99.94 | 370.6 MB/s |
| O_DIRECT c=32 | 86.26 | 78.55 | 187.74 | 1367.7 MB/s |
| buffered 冷 c=32 | 85.83 | 83.60 | 122.38 | 1428.6 MB/s |
| SELFCHECK-cold（c=1） | 78.01 | 75.51 | 91.59 | 50.0 MB/s |
| **SELFCHECK-warm（c=1）** | **1.52** | 1.34 | 3.74 | 2497.1 MB/s |

两点关键：

1. **O_DIRECT 与 buffered-全新偏移给出几乎相同的数字（77.28 vs 78.35）**——
   两条原理不同的路径互证，冷读数可信（冷热比 **51×**）。
2. **单线程 4 KiB 随机读延迟 ≈ 77 μs，且随并发上升几乎不变**（77→80→86），
   而吞吐线性放大到 1.43 GB/s。这是 NVMe 的典型形态：**QD1 延迟由盘内流水线决定，
   并发买到的是吞吐不是单次延迟。**

### 31.4 EngramDB 真实路径（`BadgeGather::gather_pp`）

门禁 `crates/engramdb-bench/src/bin/nvme_gate.rs`（`Store::fetch` 底层就是
`gather_pp(ids, out, 8)`）。512 token/轮 × 5 轮，fadvise 冷，两轮扫描：

**扫描 A：16 行/token（Qwen 几何，与 Mac/WSL 口径可比）**

| 线程 | μs/token | μs/行 | 相对 1 线程 | 占 500 μs 预算 |
|---|---|---|---|---|
| 1 | 1371.91 | 85.74 | 1.00× | 274% |
| 2 | 704.78 | 44.05 | 1.95× | 141% |
| 4 | 365.64 | 22.85 | 3.75× | 73% |
| 8 | 204.17 | 12.76 | 6.72× | 41% |
| 16 | 119.28 | 7.46 | 11.50× | 24% |
| 32 | **75.67** | 4.73 | 18.13× | **15%** |

**扫描 B：48 行/token（V4.1 几何）**

| 线程 | μs/token | μs/行 | 相对 1 线程 | 占 500 μs 预算 |
|---|---|---|---|---|
| 1 | 4078.41 | 84.97 | 1.00× | 816% |
| 4 | 1110.78 | 23.14 | 3.67× | 222% |
| **8（当前默认）** | **604.36** | 12.59 | 6.75× | **121%** |
| 16 | 353.44 | 7.36 | 11.54× | 71% |
| 32 | **222.58** | 4.64 | 18.32× | **45%** |

自校验：A `75.67 / 10.70 = 7.07×` → **VALID**；B `222.58 / 27.55 = 8.08×` → **VALID**。

**复现性**（同一 seed 0xA11CE1 复跑扫描 A，与上表逐格对比）：

| 线程 | 记录值 μs/token | 复跑 | 偏差 |
|---|---|---|---|
| 1 | 1371.91 | 1369.04 | −0.2% |
| 2 | 704.78 | 709.31 | +0.6% |
| 4 | 365.64 | 370.50 | +1.3% |
| 8 | 204.17 | 205.18 | +0.5% |
| 16 | 119.28 | 117.17 | −1.8% |
| 32 | 75.67 | 73.86 | −2.4% |

全部落在 **±2.4%** 内（其中一次复跑发生在把 `run_round` 的参数打包成 `RoundSpec`
的重构之后，即该重构**行为中性**）。宿主有其他租户的 load ≈ 11–13，
但 4 KiB 随机读延迟不受影响——这与 §31.3 "延迟由盘内流水线决定"一致。

**三个结论**：

1. **μs/行 与 rows_per_token 无关**：同样线程数下 A/B 两表几乎逐格相等
   （85.74/84.97、22.85/23.14、12.76/12.59、7.46/7.36、4.73/4.64）。
   且 8192 个 key 产生 **8192 个不同的 4 KiB 页 —— 恰好 1 行 1 页，零页共享**。
   ⇒ EngramDB 在此负载下**没有任何"顺带命中"**，成本 = 独立随机页读 × 行数。
2. **EngramDB 的额外开销很小**：单线程 85.74 μs/行 vs 裸 C 探针 77.28 μs/页 ⇒
   ≈ **+11%**（排序、分组、拷贝、回填）。引擎本身不是瓶颈。
3. **当前 8 线程默认值不够跑 V4.1**：604 μs/token = **121% 预算**。

### 31.5 最终判定：§30.6.1 的"40× / 6.7%"被实测彻底推翻

§30.6.1 曾写："NVMe 8t ≈ 0.7 μs/read，比 USB 快约 **40×**，V4.1 只占 **6.7%** 预算，
行折叠降级为余量项"。§30.7 已撤回，本轮给出**替代它的真实数字**：

| 介质 | 冷读 μs/行（8 线程） | 相对 |
|---|---|---|
| Mac + USB 外盘 | 30.0 | 1.00× |
| WSL2 + VHDX | 55.9–57.9 | 0.53× |
| **本机原生 NVMe（RAID1）** | **12.59–12.76** | **2.36×** |

**真实倍率是 2.4×，不是 40×。** 而且 V4.1 在 8 线程下是 **121% 超预算**，不是 6.7%。

### 31.6 修正后的预算表（取代 §30.6.1 全部版本）

以 500 μs/token 为 100%：

| 形态 | NVMe 8t（当前默认） | NVMe 16t | NVMe 32t | Mac+USB 8t | WSL 8t |
|---|---|---|---|---|---|
| Store-I 16 次散读（Qwen） | 204 μs（41%） | 119（24%） | 76（15%） | ~480（96%） | 902（180%） |
| **Store-I 48 次散读（V4.1）** | **604 μs（121%）** | 353（71%） | 223（45%） | ~1440（288%） | ~2707（541%） |

> Mac/WSL 的 48 行数字由实测 μs/行 线性外推——§31.4 结论 1 证明了该外推在本负载下成立。

**保守默认得到实测确认**：§30.7.4 说"在拿到可信 NVMe 数字前，不要把行折叠降级"。
现在有了可信数字，结论**仍然成立**——只是理由更精确了：
- **不是**"NVMe 慢"，而是"V4.1 的 48 次独立页读在默认并发下就是超预算"；
- 行折叠（Store-P）把 48 次散读压成 ~1 次紧凑读，仍是 V4.1 的**关键路径**。

### 31.7 真正的瓶颈是队列深度；io_uring 在本机被 seccomp 封禁

扫描 A/B 的加速曲线（1→32 线程：18.1×/18.3×）说明 **`pread` 路径的队列深度 = 线程数**。
`engramdb-io` 里已有 `UringBatchBackend`（一次 `read_many` 提交最多 256 个 SQE），
本应解耦这一点。于是写了 `crates/engramdb-bench/src/bin/uring_gate.rs` 去测——
**结论是在本机根本测不了**：

```
读失败：read_many on .../shard_000.bin (126 pages): IoUring::new: Operation not permitted (os error 1)
>>> FATAL: 0 页被读取 —— `read_many` 在本机完全不可用
```

根因已独立确证，不是猜测：

| 检查 | 结果 |
|---|---|
| `/proc/sys/kernel/io_uring_disabled` | `0`（内核允许） |
| `Seccomp` / `Seccomp_filters` | **`2` / `1`（filter 模式已启用）** |
| `CapEff` | `a80425fb` —— **无 `cap_sys_admin`**（同时解释了 `drop_caches` 被拒） |
| `syscall(425 /* io_uring_setup */, 8, &params)` | **`EPERM`** |

⇒ **Docker 默认 seccomp profile 拦掉了 io_uring。** 这对 EngramDB 是可操作信息：
"用 io_uring 提高 QD" 这条路**在容器化部署下未必可用**，需要显式 `--security-opt seccomp=...`。
README §2.4/§4.2 里"io_uring 无收益、已定案"的结论，其证据来自 **WSL/VHDX**——
而 §30.7 已判定 WSL 不是有效介质代理，故该结论应**降级为"未验证"**而非"已证否"。

**在本机可用的杠杆只有线程数**（8→32：12.6→4.6 μs/行）。这是一个**未做**的改动，
`Store::fetch` 目前硬编码 `gather_pp(..., 8)`。

### 31.8 在真实 Linux 上跑测试发现的缺陷

这台机器是本项目第一次拿到**正常 Linux + 完整 Rust 工具链**，于是补上了长期欠的
`cargo fmt` / `clippy` / `test --workspace`。**立刻抓到一个真实缺陷**：

`crates/engramdb-io/src/backend.rs` 的 `uring_roundtrip_and_semantics` 在 io_uring
不可用时**硬 panic**（`.unwrap()`），而不是跳过：

```
test backend::tests::uring_roundtrip_and_semantics ... FAILED
panicked at crates/engramdb-io/src/backend.rs:281:
  called `Result::unwrap()` on an `Err` value: "IoUring::new: Operation not permitted (os error 1)"
```

**任何在 Docker 默认 seccomp 下跑 CI 的人都会看到一个与被测代码无关的红灯。**
已修：新增 `uring_available()` 探测，不可用时打印 `SKIP ...（环境限制，不是缺陷）` 并返回。
这个缺陷在 macOS（测试被 `cfg` 掉）和 WSL（io_uring 可用）上**永远发现不了**——
它是"换一台真正不同的机器"的直接收益。

### 31.9 本轮结论与下一步

**Phase A：闭环。** 生产介质（原生 NVMe）上的冷读数字已拿到，且经两条独立路径互证 +
机械自校验裁决。

**净关闭**：Phase A（挂了两轮）✅；`cargo fmt`/`test` 全工作区 ✅（并修掉 1 个真实缺陷）；
V4.1 预算问题从"外推的 541%"变成"实测的 121%，且知道杠杆在哪" ✅。

**下一步（按价值排序，仍受 §29.6 五条纪律约束）**：

1. **把并发度做成可配置**（`Store::fetch` 的 8 是硬编码）。最小改动、最大收益：
   V4.1 从 121% → 71%（16t）。**但这会改变 `gather_pp` 的公共签名/默认值**，
   需要一次 A/B 门禁把默认值钉死。
2. **V4.1 Phase V0（keygen v2）**：纯本地、输入已冻结（`refs/v41_*`），不依赖任何硬件。
3. **行折叠（Store-P）在 V4.1 几何下的实测**：目前 48 行 = 48 页是**实测**，
   折叠后的 ~1 次读仍只有 §30.6 的旧数字支撑，**应在同一台机器上补测**。
4. `uring_gate` 保留在仓库里但**不做结论**——它现在的价值是"在有 io_uring 权限的机器上
   一次跑出答案"，以及记录 EPERM 这个部署事实。

**仍未闭环**：端到端 GPU/真机 decode（待硬件）、训练流有效吞吐、optimizer-state 预取探针。

---

## 32 并发度与行折叠：V4.1 预算问题的两个杠杆都实测了（Session 42 续）

§31.9 的下一步 #1（并发度）与 #3（V4.1 几何下的 Store-P 折叠）在本节闭环。
结论：**两者都成立，而且行折叠的收益比此前任何估计都大。**

### 32.1 先修判据本身：一次假 VOID

`nvme_gate` 的冷热自校验在小 batch 下会**误判**：那时每次调用的固定开销
（线程创建、分组、排序、回填）在 cold 与 warm 读数里**都要付**，比值必然趋近 1。
于是判据升级为**双条件**：`比值 ≥ 5x` **或** `边际 ≥ 2 μs/行`
（下界取自裸 C 探针 QD32 的 2.74 μs/页），并且**每个线程配置后都做一次自校验**。

升级后第一次跑就暴露了我自己的一个错误：边际误除以 `tokens × rows_per_token`
（正确应为 `rows_per_token`，因为两者都是 **μs/token**），**差了一个 `tokens` 因子**，
制造了 3 个假 VOID。修好后只剩 1 个真 VOID（见下）。

### 32.2 线程数 × batch 矩阵（Store-I，V4.1 几何 48 行/token）

冷读 μs/token（原生 NVMe，真实 PLE 行，`fadvise` 冷，每格 3 轮中位数）：

| tokens/次调用 | t=1 | t=4 | **t=8（旧默认）** | t=16 | **t=32** |
|---|---|---|---|---|---|
| 1 | 4951 | 1600 | 1064 | 894 ⚠️ | 953 |
| 4 | 4316 | 1277 | 789 | 506 | 436 |
| 16 | 4132 | 1142 | 655（131%） | 405（81%） | **282（56%）** |
| 64 | 4104 | 1111 | 612（122%） | 364（73%） | **239（48%）** |
| 512 | 4091 | 1095 | 593（119%） | 344（69%） | **220（44%）** |
| 4096 | 4063 | 1093 | 592（118%） | 338（68%） | **212（42%）** |

⚠️ = 唯一被判 VOID 的格子（batch=1/t=16，边际 1.71 < 2；单点噪声，且该区间本来就不在预算内）。

**同时量到的"纯开销"**（warm，同一批 keys 立刻重读，全部命中缓存）：

| tokens | t=1 | t=4 | t=8 | t=16 | t=32 |
|---|---|---|---|---|---|
| 1 | 459 | 330 | 322 | 812 | 515 |
| 4 | 295 | 114 | 115 | 146 | 202 |
| 16 | 237 | 94 | 66 | 72 | 66 |
| 512 | 121 | 48 | 31 | 28 | 31 |
| 4096 | 99 | 35 | 27 | 23 | 23 |

**边际成本**（cold − warm，即真正花在介质上的）：

| tokens | t=1 | t=8 | t=32 |
|---|---|---|---|
| 16（μs/行） | 81.16 | 12.26 | 4.50 |
| 512（μs/行） | 82.72 | 11.70 | 3.92 |
| 4096（μs/行） | 82.59 | 11.76 | 3.94 |

**边际跨 batch 高度稳定**（t=32 时 3.92–4.50 μs/行），并与裸 C 探针 QD32 的
2.74 μs/页一致（+43%，含引擎的排序/拷贝）。

**结论**：
1. **旧默认 8 让 V4.1 超预算（119–131%）；32 把它压到 42–56%。**
2. 这个选择对 batch 不敏感（边际稳定），所以一个常数就够，不需要自适应规则。
3. **batch=1 在任何线程数下都不可行**（最好 894 μs = 179%）——
   `gather_pp` **每次调用都 spawn 线程**，t=32 的固定开销在 1–4 token 下约 0.5 ms。
   这是"常驻线程池"该解决的，不是靠调参。

### 32.3 Store-P 折叠：V4.1 几何下的实测（`view_gate`）

§31.9 的下一步 #3 说"折叠后的 ~1 次读仍只有 §30.6 的旧数字支撑"。现在补上了。

**口径**：视图 `slot_bytes = 12,672`（= 48 行 × 256 B 载荷 + 8 缩放 × 48），
2,000,000 槽 × 12,672 B = **23 GiB**，同一台机器、同一 `fadvise` 冷读口径、同一双条件自校验。

| tokens/次调用 | t=1 | t=8 | t=32 | `default`(128) |
|---|---|---|---|---|
| 1 | 88.71 | 81.08 | 81.17 | 102.68 |
| 4 | 109.84 | 101.91 | 90.80 | 100.08 |
| 8 | 95.20 | 71.47 | 53.09 | — |
| 16 | 91.51 | **35.16** | 49.64 | 91.95 |
| 32 | 93.11 | 28.56 | 66.14 | 23.18 |
| 64 | 93.88 | 19.61 | 24.63 | 16.18 |
| 512 | 100.21 | 14.28 | **7.69** | 8.46 |
| 4096 | 96.35 | 13.28 | **5.53** | 7.34 |

**这是本轮最重要的数字。对照 Store-I（同机、同口径）：**

| 形态 | @512 tokens, t=8 | @512 tokens, t=32 | @16 tokens |
|---|---|---|---|
| Store-I（48 次散读） | 593 μs（119%） | 220 μs（44%） | 655 μs（131%） |
| **Store-P（1 次折叠读）** | **14.28 μs（2.9%）** | **7.69 μs（1.5%）** | **35.16 μs（7.0%）** |
| **折叠增益** | **41.5×** | **28.6×** | **18.6×** |

字节放大：Store-I 每 token 打 **48 × 4 KiB = 196,608 B** 的页流量，
Store-P 只打 **12,672 B** ⇒ **15.5×**。

**最关键的一点**：Store-P **在每一个 batch 下都进预算**，包括 batch=1
（81 μs = 16%）——而 Store-I 在 batch=1 下最好也只有 179%。
⇒ **行折叠不是"余量项"或"优化项"，它是把 V4.1 从"必须靠大 batch 才可行"
变成"任何 batch 都可行"的那个东西。** §30.6.1 曾把折叠降级为余量项，此处再次否定。

### 32.4 顺带发现：`read_records` 的串行阈值在 NVMe 上是净损失

`ViewReader::read_records_parallel` 有一条 `indices.len() < 32` 就走**串行**的路径。
实测（tokens=16，即 16 条记录）：**t=1 / t=8 / t=32 的读数完全相同（~90 μs/token）**
—— 证明线程数根本没被用到。而**刚好越过阈值的 32 条**走并行只有 28.56 μs/token（t=8）。
⇒ **阈值本身在 16–31 条区间净损失约 2.6×。**

已把阈值从 **32 降到 4**（n 条记录多花 (n−1) 次线程创建 ≈ 8–10 μs/个，
省下 (n−1) 次冷读延迟 ≈ 77 μs（NVMe）/ 272 μs（USB）；n=2 就已划得来）。
改后复测：16 条 **89.67 → 35.16 μs/token**。

### 32.5 代码改动

| 改动 | 位置 | 依据 |
|---|---|---|
| 新增 `DEFAULT_GATHER_THREADS = 32`（取代 10 处硬编码 `8`） | `crates/engramdb-io/src/batch.rs` + 4 个文件的调用点 | §32.2 矩阵 |
| PyO3 `Store::new` 增加可选 `threads`（默认取该常量） | `crates/engramdb-pyo3/src/lib.rs` | 让并发度的 A/B 不必改代码重编译 |
| `read_records` 串行阈值 32 → 4 | `crates/engramdb-io/src/view.rs` | §32.4 |
| `nvme_gate` 双条件判据 + 每配置自校验 + 修 marginal 除数 | `crates/engramdb-bench/src/bin/nvme_gate.rs` | §32.1 |
| 新增 `view_gate`（Store-P 折叠门禁） | `crates/engramdb-bench/src/bin/view_gate.rs` | §32.3 |

验证（原生 Linux，本机）：`cargo fmt --check` 干净、`cargo clippy --workspace --all-targets`
**0 warning / 0 error**、`cargo test --workspace` **27 passed / 0 failed**。

### 32.6 结论与下一步

**净关闭**：§31.9 的下一步 #1 ✅、#3 ✅；`read_records` 阈值缺陷 ✅；
判据自身的假 VOID 缺陷 ✅。

**V4.1 的预算问题现在有完整答案**：

- Store-I 48 次散读：**必须** batch ≥ 16 且 t=32（282 μs，56%）。batch=1 无解。
- Store-P 1 次折叠读：**任意 batch 都进预算**（7.7–35 μs，1.5–7%）。

**下一步（按价值排序）**：

1. **常驻线程池**取代 per-call spawn。实测依据：t=32 在 1–4 token 下的固定开销
   约 0.5 ms/call，在 512 token 下降到 31 μs/token。这是 Store-I 小 batch 无解的唯一原因，
   也是 Store-P 在 batch=1 时 81 μs 里的大部分。
2. **把 Store-P 的 12,672 B 记录做页对齐**（现在 12,672 = 3.09 个 4 KiB 页）。
   需要量，不要假设。
3. V4.1 Phase V0（keygen v2）——纯本地，输入已冻结，与本轮无依赖。
4. 端到端 GPU/真机 decode 仍待硬件（4090 24 GB 装不下 V4.1 的 202.8 GB engram 表，
   但足够跑 Qwen3.8-Flash-Next 的 FP8 ≈90 GB 的**切片**验证）。



---

## 33 常驻线程池：做成了，但中间踩了两个坑（Session 42 续）

§32.6 的下一步 #1。结论：**池子是净收益**（小 batch 快 1.9×），
但如果不是先量后做、且做了配对 A/B，我会**两次得出相反的结论**。

### 33.1 先量 spawn 到底多贵（决定要不要做）

`crates/engramdb-bench/src/bin/call_cost.rs` 把 `gather_pp` 的每次调用开销拆开，
先量最简单的那个：空 `thread::scope` + n 个 no-op spawn。

| n | 0 | 1 | 4 | 8 | 16 | **32** | 64 |
|---|---|---|---|---|---|---|---|
| μs/次调用 | 0.2 | 61.5 | 136.0 | 225.6 | 459.2 | **1025.1** | 2221.6 |
| 边际 μs/spawn | — | 61 | 34 | 28 | 29 | **32** | 35 |

**单次 spawn ≈ 30–35 μs，从 n=4 起严格线性。** 而 `gather_pp` 在 65 shard、t=32 时
每次调用要 spawn ≈22 个 ⇒ **≈0.7 ms/次调用**。这个数量级足以吃掉 500 μs 预算。

> 注意：我一开始是从 `nvme_gate` 的 warm 读数**反推**出"只差 56 μs、池子不值"的。
> 那是错的 —— 单次调用读数被缺页/冷 CPU 缓存抬高。**直接量 spawn 曲线才看到真相。**

### 33.2 坑一：池子被 `available_parallelism()` 卡在 16

第一版用 `available_parallelism()` 定池子大小。在没有自描述诊断之前，
A/B 显示池子让冷读**慢 1.7×**（边际 4.5 → 8.0 μs/行，开销却几乎没变）。
我先后猜过"惊群""线程放置"，都不对：换成 `notify_one` 精确唤醒**没有改善**。

加上自描述诊断后一次定位：

```
mode=pool workers=auto  →  pool_workers=16  →  415.71 μs/token
mode=pool workers=64    →  pool_workers=64  →  244.34 μs/token
```

**`available_parallelism()` 遵守 cgroup CPU 配额，在本机返回 16（而 `nproc` 是 128）。**
池子只有 16 个 worker，而 t=32 要提交 22 个任务 ⇒ 分两波 ⇒ **IO 并发度腰斩**。

**修复**（两处，都要）：
1. 池子下界取 `max(available_parallelism(), DEFAULT_GATHER_THREADS)` —— IO 密集负载的
   并发度不该被 CPU 配额限制（§32.2 已证明 t=32 显著优于 t=16）；
2. `scope_run` 里加一条：**任务数 > worker 数时主动回退到 `thread::scope`** ——
   宁可付 spawn 的钱，也不要悄悄降并发。

**同一个坑还有第二处**：`ViewReader::read_records` 的默认线程数原来也是
`available_parallelism()`（本机 16）。已改为 `DEFAULT_GATHER_THREADS`。

### 33.3 坑二：非配对 A/B 把结论搞反了（两次）

池子修好后又出现一次"tok=512 时 pool 417 vs nopool 248"。这次我没有改代码，而是做**配对交替**：

```
P N N P P N N P   (tok=512, t=32)
pool   242.41 245.77 247.22 248.93   marg 4.47 4.53 4.60 4.58
nopool 245.25 266.96 244.69 249.16   marg 4.43 4.90 4.44 4.51
```

**两者相同。** 之前那个 417 是**顺序伪影**：在顺序扫描里 `pool` 总是每组第一个跑，
吸收了前一轮（fadvise 批量驱逐 + 24576 次冷读）的余波。
同一台机器、同一二进制、同一 seed，只因为**先跑**就慢 1.7×。

> 这是同类错误的第三次（§30.2 记过两次）。前两次是"复用偏移"和"Python 线程冒充存储并行度"。
> **共同点：都是测量口径问题，而不是被测对象的问题。**

### 33.4 结果（配对交替，t=32，V4.1 几何，Store-I）

| batch | pool cold | nopool cold | pool 开销 | nopool 开销 |
|---|---|---|---|---|
| tokens=1 | **515 / 577** | 1008 / 1013 | **73 / 147** | 471 / 1839 |
| tokens=16 | **262 / 265** | 287 / 311 | 29 / 32 | 56 / 63 |
| tokens=512 | 246 / 254 | 246 / 249 | 29 / 31 | 32 / 34 |

- **小 batch 快 1.9×**（收益**全部**来自开销：471–1839 → 73–147 μs）；
- 中 batch 快约 15%；
- 大 batch 冷读持平，开销略降。
- **永远不再有负收益** —— 这正是 §33.2 那条"任务数 > worker 数就回退"的保证。

一个**仍然成立**的结论：`gather_pp` 每次调用固定开销的下限被压到约 **25–30 μs/token**
（t=32、大 batch）。batch=1 的 Store-I 即使有池子也只有 ~515 μs，仍在预算边缘；
**Store-P（§32.3，81 μs）仍是 batch=1 的唯一可行路径。**

### 33.5 代码改动

| 改动 | 位置 |
|---|---|
| 新增常驻池（`scope_run` / `workers` / `disabled`，含 5 个测试） | `crates/engramdb-io/src/pool.rs` |
| `gather_pp` 改走池子 | `crates/engramdb-io/src/batch.rs` |
| `read_records_parallel` 改走池子；`read_records` 默认线程数 16→32 | `crates/engramdb-io/src/view.rs` |
| 新增开销分解探针 | `crates/engramdb-bench/src/bin/call_cost.rs` |
| 门禁打印**实际 IO 路径**（自描述） | `crates/engramdb-bench/src/bin/nvme_gate.rs` |
| 逃生开关 `ENGRAMDB_NO_POOL=1`；A/B 用 `ENGRAMDB_POOL_WORKERS=n` | `crates/engramdb-io/src/pool.rs` |

安全性：`scope_run` 把 `Box<dyn FnOnce() + Send + 'a>` 擦除成 `'static` 后投递，
靠"计数到齐才返回、且计数在任务执行并释放 Box **之后**递增"保证借用有效
（rayon/crossbeam 同款）。任务 panic 被 worker 捕获、计数照常、在调用线程重放，
**池子不会因 panic 缩水或死锁**（有测试覆盖）。

验证：`cargo fmt --check` 干净、`clippy --workspace --all-targets` **0 finding**、
`cargo test --workspace` **32 passed / 0 failed**。

### 33.6 纪律增补（在 §29.6 四条 + §30.5 第五条之上）

6. **共享机器上的 A/B 必须配对交替**（`A B B A` 或更长），
   并在**同一次脚本运行内**完成；顺序扫描得出的差异不可信。
7. **每个基准运行必须自报它实际走的代码路径**（本次是 `pool_workers=/pool_disabled=`）。
   否则"A/B 无差异"可能只是"两组跑的是同一条路径"。

---

## 34 拆开「Python 后端」与「C 嵌入面」（Session 42 续）

### 34.1 决定

**删除 Python 的 ctypes 回退；保留 C ABI 但重新定位为 C/C++ 嵌入面。**

`crates/engramdb-python` 一直被当成一个东西，其实背了两个互不相关的产品：
一个 Python 开发回退、一个 C 嵌入面。它们的受众、生命周期、验收标准都不同，
混在一起的结果是两边都说不清。

### 34.2 证据：那份回退是负资产

| 事实 | 出处 |
|---|---|
| **wheel 里根本没有它** —— maturin 只打包 `engramdb-pyo3`，`module-name = "engramdb._engramdb"` | `python/pyproject.toml` |
| **只有「源码树 + 已 cargo 构建」才可达** —— 库搜索路径是 `Path(__file__).parents[2]/target/{release,debug}/libengramdb_c.*` | `python/engramdb/__init__.py`（旧） |
| **CI 从未覆盖** —— wheel smoke 装的是 wheel（=PyO3）；`c_abi_smoke.py` **不 import `engramdb` 包**，直接 dlopen cdylib | `.github/workflows/ci.yml` |
| **已经分叉** —— 给它加 `DEFAULT_GATHER_THREADS` 时，PyO3 `Store` 拿到 `threads=`，ctypes 没有 | §32.5 |
| **静默降级** —— 扩展导入失败不报错，只是悄悄换成子集实现 | Session 42 本人在 Mac 上整场跑在 ctypes 上而不自知（Python 3.9 vs 3.12 构建） |
| **它恰好是唯一不支持 V4.1 的那一面** | 技术债 V55 |

还有一条**在删的过程中才发现**的：`__init__.py` 无条件执行 `from .tables import Database`，
而 `tables.py` 又 `from . import Store` —— 所以旧 docstring 那句
「两者都没有时 `import engramdb` 仍可用」**本来就是假的**，它会在 `tables.py`
以一条 partially-initialized 的难懂错误炸掉。**所谓「软失败」从来没软过。**

⇒ 删掉它不是移除能力，而是把一条**不可达、未测试、会掩盖故障**的路径换成一个
**响亮且带修复指引的错误**。

### 34.3 顺带修掉的三个真实缺陷（都是同一类：**在 Linux 上假设 Linux 的一切都可用**）

1. **`IoUringPageReader` 在容器里硬崩。**
   类存在只说明平台是 Linux，**不代表内核允许 io_uring** —— Docker 默认 seccomp
   让 `io_uring_setup` 返回 `EPERM`。`SGLangPageReader` 又只要看到该类就优先用它，
   于是 Linux + 容器的用户一调用就崩。
   **已修**：新增 `uring_available()` 一次性探测（seccomp 按进程生效，不必每线程重试），
   不可用时 `read_pages` **退回 `pread`**，**短读/EOF 语义与 io_uring 路径严格一致**；
   新增 `backend` 只读属性（`"io_uring"` / `"pread"`）使实际路径可查。
   *这与 §31.8 修掉的 Rust 测试 panic 是同一个缺陷类，第三次出现。*

2. **`scripts/build_pyo3.sh` 在 Linux 上必然失败。**
   它硬编码 `cp target/release/lib_engramdb.dylib ...`，尽管上面就有 `uname -s = Darwin` 分支。
   **已修**：按平台选 `lib_engramdb.dylib` / `lib_engramdb.so` / `engramdb.dll`。

3. **`engramdb.__repr__()` 是死代码。**
   模块级 `__repr__` **不会**覆盖 `repr(module)`（PEP 562 只覆盖 `__getattr__` / `__dir__`），
   而它从未被任何地方显式调用过。**已删**；取而代之，wheel smoke 新增
   `test_native_backend_is_pyo3()`，断言 `_USING_PYO3`、`abi_version()`，以及
   **扩展实际从哪个文件加载**。

### 34.4 代码改动

| 改动 | 位置 |
|---|---|
| 删除 ctypes 回退（−201 行），导入失败改为带修复指引的 `ImportError` | `python/engramdb/__init__.py`（418 → 216 行） |
| 删除死 `__repr__`，简化三个 `_USING_*` 分支 | 同上 |
| crate 改名 `engramdb-python` → **`engramdb-cabi`**，重新定位为 C 嵌入面 | `crates/engramdb-cabi/`、根 `Cargo.toml` |
| 模块文档写明「不是 Python 后端」+ 能力边界（只有 `PLE_QWEN_V1`，V55） | `crates/engramdb-cabi/src/lib.rs` |
| `uring_available()` + `pread` 退化 + `backend` 属性 | `crates/engramdb-pyo3/src/lib.rs` |
| 跨平台 `.so` / `.dylib` / `.dll` 选择 | `scripts/build_pyo3.sh` |
| `test_native_backend_is_pyo3()`（替代被删的回退作为机械保障） | `scripts/python_wheel_smoke.py` |
| CI / release gate / binding smoke 改用 `-p engramdb-cabi` | `.github/workflows/ci.yml`、`scripts/release_gate.sh`、`scripts/python_binding_smoke.sh` |

**cdylib 名保持 `libengramdb_c`** —— C 消费者的链接名不该因为 crate 改名而断。

### 34.5 C ABI 的当前边界（写清楚，避免下次又混）

- 它是**给 C/C++ 的**：`engramdb_abi_version() == 1`，13 个导出符号，
  刻意不依赖绑定生成库以便离线构建。
- **只实现 `PLE_QWEN_V1`**（`ple_spec == 1`）；`ENG_DEEPSEEK_V1`（V4.1）传入即报错
  —— 技术债 **V55** 仍开着。
- **没有任何 C/C++ 消费者**（仓库内唯一 `.c` 是基准探针）。
- 与 llama.cpp 相关的是 roadmap 里**未勾选**的「C ABI / GGUF 方向探索」。

⇒ **它是一个「保留但未启用」的面。** 如果将来要接 C++ 引擎（llama.cpp / TRT-LLM），
需要先决定是否补 `EngramSpec` 泛化让它支持 V4.1；在那之前它只需要**别腐坏**
（`c_abi_smoke.py` 进 CI 就是为了这个）。

### 34.6 验证（原生 Linux，AutoDL）

- `cargo fmt --check` 干净、`clippy --workspace --all-targets` **0 finding**、
  `cargo test --workspace` **32 passed / 0 failed**；
- 改名后 `cargo build --workspace --release` 通过；
- **只在 PyO3 下**导入成功：`_USING_PYO3=True`、`abi_version=1`、
  `hasattr(engramdb, "_USING_CTYPES") == False`、golden rowid 正确；
- `scripts/python_wheel_smoke.py` **全绿**（含新的 native backend 守卫），
  三个 page reader 均报 `backend=pread` 且**真的读对**；
- `scripts/c_abi_smoke.py` 通过（`abi_version=1, 14 rowids match golden`）。

---

## 35 换裁判：从「每 token µs」到「存储代价 ÷ 引擎计算时间」（Session 42 尾）

### 35.1 北极星没变，裁判该改

北极星仍是「确定性记忆表（Engram/PLE）的磁盘优先基础设施 —— DuckDB 之于分析数据库」。

但本轮在 4090 + vLLM 0.29.0 上跑出真 serving 数后暴露一件事：
**§29.1.1 的 500 µs/token 裁判，本轮的每一个配置都满足它，而项目并没有更接近目标。**

| 配置 | 存储代价 | 500 µs 预算 | 判定 |
|---|---|---|---|
| batch=1（Qwen 几何，16 行） | 185 µs/次 | 内 | ✅ |
| batch=32 | ~19 µs/token | 内 | ✅ |
| adapter 计算路径（免费介质） | 11.6–198 µs/token | 内 | ✅ |

原因：**eager 模式下引擎自身开销 22 ms/token，5% 预算 = 1.1 ms。**
存储代价 0.6 ms 藏在引擎开销里，「达标」是廉价的。

⇒ **建议补一条平行裁判**（不是替换）：

> **把引擎自身开销 O 压到 ≤5% 之后，存储路径仍不成为新的瓶颈。**
> 即预算应相对于 **O 或引擎计算时间**，而不是相对于墙钟 token 时间。

理由：现行裁判是**引擎无关**的，而目标是**引擎相关**的。前者既容易「达标」又无法前进。

### 35.1b 兑现：把 graph 打开，裁判立刻咬人（Session 42 收盘实测定量）

上面是**论证**。本轮把 CUDA graph 打开后，它变成了**数字**
（`probes/engine_floor_session42.md`，同一模型、同一 GPU、同一负载）：

| 引擎 | 模式 | tok/s 中位 | ms/token | 500 µs 占单步 | graph 收益 |
|---|---|---|---|---|---|
| vLLM 0.29.0 | eager | 47.0 | 21.28 | 2.35% | — |
| vLLM 0.29.0 | **CUDA graph** | **342.0** | **2.92** | **17.10%** | **7.28×** |
| SGLang 0.5.19 | eager | 56.3 | 17.76 | 2.82% | — |
| SGLang 0.5.19 | **CUDA graph** | **440.4** | **2.27** | **22.02%** | **7.82×** |

把本轮直接测到的冷读代价（**195.9 µs/token**，16 行 × 160 B，NVMe 冷）放到四个分母上：

| 分母 | 占比 | 对 5% |
|---|---|---|
| vLLM eager 21.28 ms | 0.92% | ✅ |
| SGLang eager 17.76 ms | 1.10% | ✅ |
| vLLM graph 2.92 ms | **6.70%** | ❌ |
| SGLang graph 2.27 ms | **8.63%** | ❌ |

**⇒ 「eager 下落进噪声」从来不是达标，是分母被放大 7.3–7.8×。**
两条工程结论：

1. **两引擎 eager 可比（47.0 vs 56.3）**，graph 收益也相当（7.28× vs 7.82×）
   ⇒ 这不是某个引擎的实现质量问题，是**模式**问题。
2. **提前量检查（取上界）**：layer 2 在 SGLang graph 下只等到 `(2/24)·2270 = 189.2 µs`，
   而冷读要 195.9 µs ⇒ **按最有利假设也装不下**。
   这解释了 V4.1 为什么把 PLE 放在 **layer 14**（τ=2295 µs）：**不是随便选的层，
   是必须放得够深才装得下。** 出路是「藏」，不是「更快」。

**子条件 4 的路已有现成范例**（不再是推测）：vLLM 启动 dump 的
`compilation_config.splitting_ops` 里就有 `vllm::qwen4_exp_compute_ple_ngram_ids`
与 `vllm::qwen4_exp_ple_short_conv` —— 引擎把自己的 PLE rowid 计算注册成了
**splitting op**（`cudagraph_mode=FULL_AND_PIECEWISE`），可变形状部分留图外、其余进图。
我们要做的是照抄这个模式，不是发明机制。

### 35.1c 子条件 4 的首次实现：eager 全通、graph 全不通（8 次运行留档）

`probes/subcondition4_cuda_graph_session42.md`。**未闭合**，但失败点被压到一层：

| 环节 | 状态 |
|---|---|
| 类级补丁在 trace 前生效 | ✅ |
| 自定义 op 注册 + `mutates_args` 契约 | ✅（eager 下 `op_calls=256 / op_rows=510`） |
| 磁盘读取经 op 发生 | ✅（eager 下 reader 210–220 µs/次） |
| delta 进入模型（语义） | ✅（`identical_to_none=False`） |
| **`enforce_eager=False` 时 op 在 replay 中执行** | ❌ |

排掉的四个坑，每一个都值得单独记：

| # | 坑 | 后果 |
|---|---|---|
| W1 | 补丁打在 `LLM()` **之后** | `embed_tokens` 在编译区外所以照跑，但 `Qwen3_5Model.forward` 已 trace 完，`layer.forward` 被固化 ⇒ **静默无效** |
| W2 | **`~/.cache/vllm/torch_compile_cache`**（87 MB，由未打补丁的运行编译） | 后续每次运行复用旧图 ⇒ 补丁移到 `LLM()` 之前**仍然无效**。⇒ `VLLM_DISABLE_COMPILE_CACHE=1` 是这类实验的**必要条件** |
| W3 | 在被 trace 的 forward 里**改计数器** | torch 直接拒绝：*"Assigning / modifying buffers of nn.Module during forward pass is not allowed when using cudagraph inside the compiler"* ⇒ **计数器永远无法自证 replay**，必须用功能性判据（输出必须改变） |
| W4 | **静态 buffer + 普通加法** | 闭包变量是 constant、`nn.Module` buffer 是 `get_attr`，两者都被 Inductor 折进常量池，`hidden_states + 0` **被整个折叠掉** ⇒ 与 Inductor 根本不兼容 |

> W3 与 W4 合起来给出一条通用纪律：**在 CUDA graph 下，任何「图外写好、图内读」的方案
> 都必须让那个张量成为 `get_attr` 之外的东西** —— 即**算子的实参**。
> 这就是 `mutates_args` 存在的理由。

### 35.1d 子条件 4 的机制**在树上已经存在**，而且不是我们以为的那个（session 43 调研）

> 这一节把 §35.1c 结尾的四条假设全部结掉，并推翻其中的方向。
> 结论：**我们一直在错的那条路上找出口。** 出路不是让 `splitting_ops` 生效，
> 而是换一个**不依赖 torch.compile** 的机制 —— 那个机制 vLLM 0.29.0 里已经有。

#### 先结掉四条假设

**假设 4（`FULL_AND_PIECEWISE` 绕过切分点）成立，而且是最直接的原因。**
`vllm/config/compilation.py:53-63`：

```python
FULL_DECODE_ONLY = (FULL, NONE)
FULL_AND_PIECEWISE = (FULL, PIECEWISE)
def decode_mode(self): return CUDAGraphMode(self.value[0]) if self.separate_routine() else self
```

`value[0] == FULL` ⇒ **decode 阶段用 FULL**。而 `eager_break_during_capture`
（见下）里有一句 `if mode == CUDAGraphMode.FULL: return fn(*args, **kwargs)` ——
**FULL 下不打断**。我们的 op 因此在 capture 期被塞进一张 full graph，
replay 时不再执行。这与实测的 `reader_calls = 0` 完全一致。

**假设 1、2（`splitting_ops` 没传进去 / 匹配靠 tag）不需要再查。**
`vllm/compilation/partition_rules.py:14-38` 的 `should_split` 是唯一的消费点，
它按 `target._qualified_op_name` / `packet_name` 做**字符串相等**匹配，
不涉及 `tags`；`inductor_partition_rule_context` 则把同一个列表赋给
`torch._inductor.config.custom_should_partition_ops`。
即 `"vllm::engramdb_ple_read"` **本来就匹配得上** —— 问题从来不在匹配。

#### 真正的出路：`breakable_cudagraph`

vLLM 0.29.0（机器上那个版本）里有 `vllm/compilation/breakable_cudagraph.py`，
文件头第一句就是它的定位：

> *"This is an alternative to `CUDAGraphWrapper` that replaces vLLM's
> torch.compile-based FX graph splitting with **runtime stream-capture breaks**."*
>
> *"The idea (inspired by sgl-project/sglang#19102)"*

开关是环境变量（`vllm/envs.py:756-759`，默认 `0`）：

```
VLLM_USE_BREAKABLE_CUDAGRAPH=1     # "Experimental: breakable cudagraph does not rely on torch.compile"
```

接线在 `vllm/v1/worker/gpu_model_runner.py:5513-5518`：

```python
if (is_breakable_cudagraph_enabled()
        and cudagraph_mode != CUDAGraphMode.NONE
        and not self.parallel_config.use_ubatching):
    self.model = BreakableCUDAGraphWrapper(self.model, self.vllm_config)
```

对我们的意义，一句话：**W1–W4 全是 torch.compile / Inductor 的产物，而这条路径
根本不经过 torch.compile。** 编译缓存（W2）、Inductor 折常量（W4）、
trace 里禁止副作用（W3）在这一路径下**全部不适用**。

#### 接入方式：一个装饰器，而且签名要求和我们**已经写好的** op 一致

`breakable_cudagraph.py:59-91` 的 `eager_break_during_capture(fn)`：
把一个自定义算子的 **Python kernel** 变成图的断点。捕获期调用它时，
结束当前 segment → 在捕获流上 eager 执行 `fn` → 记录 `fn` 供 replay → 开新 segment。

它对我们 op 的要求（第 70-72 行，原文）：

> **"In-place output buffer required.** Decorated ops must write into a
> caller-provided output tensor; a fresh tensor returned by `fn` would change
> address each replay and break downstream graph segments."

**我们的 `_read(output, tag)` + `mutates_args=["output"]` 正好就是这个形状。**
所以改动量是：加一个装饰器 + 一个环境变量 + 把 `cudagraph_mode` 从
`FULL_AND_PIECEWISE` 改成 `PIECEWISE`（因为 FULL 不打断，见上）。
**`splitting_ops` 不需要了，`VLLM_DISABLE_COMPILE_CACHE=1` 也不再需要。**

#### 引擎自己的模板：`prefetch_ops.py`

更要紧的是，vLLM 树里已经有一个**和我们形状几乎相同**的现成实现，
位于 `vllm/model_executor/offloader/prefetch_ops.py`（94 行）：

```python
direct_register_custom_op(op_name="wait_prefetch",  op_func=_wait_prefetch_impl,
                          mutates_args=["input_tensor"],  fake_impl=_wait_prefetch_fake)
direct_register_custom_op(op_name="start_prefetch", op_func=_start_prefetch_impl,
                          mutates_args=["output_tensor"], fake_impl=_start_prefetch_fake)
```

文件头的注释就是 W3/W4 那条纪律的官方版本：

> *"These ops use mutates_args to create data dependencies that prevent the
> compiler from reordering prefetch/sync operations."*

而 `vllm/model_executor/offloader/prefetch.py` 的注释写着
*"Adapted from sglang/srt/utils/offloader.py"* —— **两个引擎在这里也是收敛的**。
它的做法值得直接照抄（`prefetch.py:155-156, 250-288, 512-547`）：

| 环节 | 做法 | 为什么对我们重要 |
|---|---|---|
| 独立 `copy_stream` | `torch.cuda.Stream()` | 取数与计算**真正并行** |
| fork | `current_stream().record_event(e); copy_stream.wait_event(e)` | 「事件 fork」可被 CUDA graph 捕获 |
| 完成信号 | `_copy_done_event.record(copy_stream)` | 等待变成**事件等待**，因此可进图 |
| wait | capture 中 `wait_event`，eager 下退化为 `wait_stream` | 两种模式都能用 |
| join | `join_after_forward()` 在最后一段闭合前 join | 否则 replay 报 unjoined stream |

**⇒ 关键洞察：「取数」和「等待」是两个算子，而且两者都能进图**
（fork 与 event-wait 都是可捕获的 CUDA 操作）。
**只有「磁盘读」这一段是宿主侧的、必须留在图外。** 于是最小设计是：

1. `start_ple_read(...)` —— eager 断点，提交 io_uring 后**立即返回**（不阻塞）；
2. 内核在后台完成 I/O，GPU 同时算 layer 0…L-1；
3. `wait_ple_read(...)` —— 取 pinned host buffer → **copy_stream 上的 async H2D**（进图）；
4. 消费端 event-wait（进图）。

#### 顺带解决「提前量」：断点天然给出重叠（有别于我们原先的假设）

我们原先以为需要一个自己造的调度器来把读藏进前面的层。**不需要 —— 断点机制自带。**
两个引擎的 replay 都是「launch 一段图 → 跑宿主函数 → launch 下一段」，**段间无同步**：

```python
# vLLM  vllm/compilation/breakable_cudagraph.py:212-214
def replay(self) -> None:
    for r in self.segments:
        r()

# SGLang python/sglang/srt/model_executor/runner_backend_utils/
#        breakable_cuda_graph/breakable_cuda_graph.py:281-290
def replay(self) -> None:
    for i, seg in enumerate(self._segments):
        seg.replay()                      # cudaGraphLaunch — 异步返回
        if i < len(self._break_fns):
            self._break_fns[i]()          # 宿主代码，与 GPU 并行
```

`cudaGraphLaunch` 立即返回，所以**断点函数在宿主上执行的同时，GPU 正在跑刚 launch 的那一段**。
把读放在 layer L 的断点上，它就与 **layer 0…L-1 的 GPU 时间**并行：

```
每步净增延迟 ≈ max(0, 读耗时 − τ(L))
```

| 断点位置 | τ(L) | 冷读 195.9 µs | 净增 |
|---|---|---|---|
| layer 2 | 189.2 µs（SGLang graph 实测） | 195.9 µs | **+6.7 µs**（装不下） |
| layer 14 | 2295 µs（V4.1 实测） | 195.9 µs | **0**（余量 ~11×） |

⇒ 这与 §35.1.1 的提前量模型**完全同形**，而且解释了 V4.1 为什么把 PLE 放在 layer 14：
**层深不是为了正确性，是为了让断点的重叠窗口盖住读延迟。**

#### 版本边界（必须记，否则会踩空）

| 引擎 | 版本 | 日期 | 有该机制 |
|---|---|---|---|
| SGLang | **v0.5.19** | 2026-09-05 | ✅（PR #19102，2026-04-11 merged） |
| vLLM | **v0.29.0** | 2026-09-09 | ✅（PR #42304，2026-05-16 merged） |
| vLLM | main（>0.29.0） | 2026-09-11 | ⚠️ PR #56312 把 breakable **限定为只接管 PIECEWISE**，FULL 交回标准 `CUDAGraphWrapper(FULL)` |

⇒ 我们机器上两个引擎**都有**这个机制。但**必须以 `cudagraph_mode=PIECEWISE` 运行**，
否则 decode 走 FULL、断点不生效 —— 这正是本轮实测到的失败。

#### 新的验收判据（替代 §35.1c 的隐式标准）

> **(a)** 确实运行在断点模式下：vLLM 侧 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
> **且 dump 后断言** `cudagraph_mode == PIECEWISE`（不信传进去的值，见下）；
> SGLang 侧 `--cuda-graph-backend-decode=breakable`
> **(b)** `reader_calls > 0`
> **(c)** `tokens_identical_to_none == False`
> **(d)** 断点真的发生了：`capture.num_eager_breaks > 0`

**(d) 是这一轮新加的。** 前三项在 vLLM 那八次失败运行里**至少有一次可能全绿**而仍然是假的 ——
因为「读发生了」和「断点发生了」是两件事：eager 模式下读也会发生。

#### ⚠️ 一个会伪装成「跑通」的静默降级

`vllm/config/compilation.py:1195-1217` 会在**注意力后端不支持 piecewise** 时
不报错地改掉我们的模式：

| 我们设的 | 静默变成 | 后果 |
|---|---|---|
| `PIECEWISE` | **`NONE`** | 变回 eager ⇒ 看着「有图」其实没有，`reader_calls` 会有值，**假阳性** |
| `FULL_AND_PIECEWISE` | **`FULL`** | 断点失效 ⇒ 就是本轮实测到的失败 |

`resolve_cudagraph_mode_and_sizes()`（同文件 1375-1435）还会按后端能力**自动挑**模式。
⇒ **上机必须 dump 一次启动后的 `compilation_config.cudagraph_mode` 并断言等于 `PIECEWISE`**，
不能假设传进去就是生效值。这与 §35.1c 的 W2（编译缓存复用旧产物）是同一类陷阱：
**注入类实验必须自证「我确实在我想测的那个模式下」**，而不只是自证「我的读发生了」。

#### 更强的信号：vLLM 对 PLE 家族**自动开启**这个机制

`vllm/config/vllm.py:75-94`：

```python
DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = frozenset({
    "DeepseekV32MTPModel", "DeepseekV32ForCausalLM",
    "DeepseekV4ForCausalLM",          # ← PLE / Engram 那一族
    "DeepSeekV4MTPModel", "Dots3NoteForCausalLM", ...
})
```

只要 `VLLM_USE_BREAKABLE_CUDAGRAPH` 未显式设置且架构在表里，
`_maybe_enable_breakable_cudagraph()`（`:710-728`）就自己设环境变量并打日志
*"Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1"*。

⇒ **引擎自己认定「带 PLE 的那一族模型就该用 breakable」。** 宿主侧取数不是我们
硬塞进去的异类，是引擎已经认下的模式。但我们本机没有 V4 的 checkpoint，
架构名对不上 ⇒ **自动开启不会触发，必须手动设环境变量**。

#### `mode` 被压成 NONE 不是问题 —— 引擎显式开了例外

同函数在开启时**主动**把编译模式压成 NONE：
`if enabled: self.compilation_config.mode = CompilationMode.NONE`。

而两处本来会把 `PIECEWISE` 打死的守卫**都豁免了 breakable**
（`vllm/config/vllm.py:1469-1480` 与 `:1714-1722`）：

```python
... and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH):     # 否则把 cudagraph_mode 打成 NONE
...
assert (self.compilation_config.mode == CompilationMode.VLLM_COMPILE
        or envs.VLLM_USE_BREAKABLE_CUDAGRAPH), (...)  # 否则直接崩
```

⇒ **`mode=NONE` + `cudagraph_mode=PIECEWISE` + breakable 是被承认的组合。**
「breakable 与 compile 互斥」指的是 **compile 被替换掉**，不是 **只能跑 eager**。

#### 两条路线的取舍

| | 路线 A：`splitting_ops` | 路线 B：`breakable_cudagraph` |
|---|---|---|
| 切图者 | Dynamo FX + Inductor | 运行时流捕获断点（不经过 compile） |
| 需要 | `mode=VLLM_COMPILE` + `splitting_ops` + `PIECEWISE` | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` + `PIECEWISE` |
| 我们已有的代码 | ✅ 已写好 | ⚠️ 加一个装饰器 |
| 受 Inductor 影响 | ✅ 会（W3/W4 两个坑依然在） | ❌ 不经过 |
| 重叠语义 | ⚠️ **未验证** | ✅ 代码里明摆着（§下） |
| vLLM 对 PLE 家族的默认 | ❌ | ✅ **是** |

**建议先试 B**（W3/W4 直接消失 + 引擎默认 + 重叠语义明确），A 作为后备。

#### SGLang 侧的对应物（⚠️ 官方文档是错的）

<https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph.md> 说
`SGLANG_USE_BREAKABLE_CUDA_GRAPH` *"Required for `@eager_on_graph` decorators
to take effect"* —— **在 v0.5.19 里这是假的。** 全树 `.py` 检索只有两处：
`environ.py:1290` 定义、`serving_hook.py:416` **只 `.set()` 从不读**。
**这个变量是只写的。**

真开关是 per-phase 的 CUDA graph 后端（`model_executor/cuda_graph_config.py`）：

```python
decode:  PhaseConfig(backend=Backend.FULL)                       # L147-149
prefill: PhaseConfig(backend=default_prefill_backend())          # L150-152
def default_prefill_backend(): return Backend.BREAKABLE if is_cuda() else Backend.TC_PIECEWISE  # L112-121
```

⇒ **CUDA 上 prefill 默认已是 breakable，decode 默认是 full。** 我们要：

```bash
--cuda-graph-backend-decode=breakable
```

`BreakableCUDAGraphCapture` 全 server 路径只有一处实例化
（`runner_backend/breakable_cuda_graph_backend.py:131`），
`_current_capture_var` 只在那里设置。否则 `eager_on_graph` 是**纯直通**
（`breakable_cuda_graph.py:221-224`）⇒ **只设环境变量 + 只加装饰器 = 重演 vLLM 那次失败
（注册得好好的，replay 一次都不执行）。这是最容易重复踩的一脚。**

另一个必须知道的边界：**内建断点在 v0.5.19 只覆盖 prefill**。
attention 的 `eager_on_graph` 站点门控在 `radix_attention.py:178` 的
`forward_batch.forward_mode.is_extend()` ⇒ `--cuda-graph-backend-decode=breakable` 下
**dense 模型的 decode 是「一整段、零断点」**，机制活着但断点得我们自己加
（正好我们只要一个）。顺带：`--debug-cuda-graph` 在 v0.5.19 本身是坏的 ——
它只设那个死变量、不设 decode 后端，而 `decode_cuda_graph_runner.py:1164-1168`
又 assert *"Breakable CUDA graph is required for --debug-cuda-graph"*。

> ⚠️ 本节全部为**源码阅读结论，尚未在 GPU 上验证**。上机第一件事是按 §6 的
> 验收判据跑一次；在拿到 `reader_calls > 0` 之前，§6.1 的子条件 4 保持 ❌。
> 完整的 API 契约、行号、三个 v0.5.19 特有陷阱（**别传 CPU 张量** /
> `_copy_output` 对未知类型静默不回写 ⇒ 原地写并 `return None` /
> `decode=tc_piecewise` 没实现）、以及「断点可放在层栈内部」的现成先例
> （`inkling.py:262-271`），见 **`docs/cuda-graph-injection.md`**。

**第一次上机建议用 SGLang**，理由不是性能而是**它不容易静默失败**：
SGLang 的开关是结构性的（decode runner 直接选 `BreakableCudaGraphBackend`），
而 vLLM 的开关要经过模式解析器 —— 那正是上一轮吃掉我们八次运行的东西。
但注意 **vLLM 才是 `qwen4_exp`/PLE 有真实支持的引擎**（子条件 6），
所以 SGLang 这一轮只验证**机制**（合成投影即可），真 PLE 仍须回 vLLM。

### 35.2 本轮技术债（V166–V177）

> 净关闭率纪律（§29.4）：12 条中只有 5 条是真正新增，
> 3 条是对既有条目的**更正**。更正比新增更值钱 —— 它们防止继续在错方向上投入。

**A. 真缺陷**

| # | 债 | 量级 | 状态 |
|---|---|---|---|
| V166 | `PleSpec::real()` 每次调用重建素数表（16 次 2000 万量级素数搜索） | 941 → 0.9 µs（**843×**） | ✅ 已修 `8d4b507` |

**B. 归因错误 —— 会误导投资方向，比缺陷更危险**

| # | 债 |
|---|---|
| V167 | **V157 误诊**：把 rowid 下沉 Rust 针对的不是瓶颈（adapter 计算路径在**免费介质**上仅 11.6–198 µs/token） |
| V168 | 「磁盘读是 serving 代价」—— 配对分离（`shm-keygen` vs `engram-i`）后 RAM 43.0 vs NVMe 43.4，噪声内无差别 |
| V169 | `docs/prefetch-lead-time.md` §4.3「批量为敌」量级错：`F` 随 batch 只涨 **3.2×**（非 32×），`gather_pp` 同页去重使其饱和 |

**C. 缺口**

| # | 债 |
|---|---|
| V170 | 真 serving A/B 的**六条子条件**未闭合：eager / 假 PLE（随机投影）/ 温态 / 半表（65/128）/ 单模型 / 单引擎 |
| V171 | SGLang **零验证**（V162 的另一半）；低层 reader 同形接口在 GPU 上从未跑过 |
| V172 | **V4.1 实现为零**；且 `crates/engramdb-keygen` 模块文档指向**不存在的 `demo.rs`** |
| V173 | **llama.cpp 零代码**；C ABI 12 符号是唯一入口，缺 Store-P 格式对外文档 |
| V174 | **无 HF 原生集成、无训练侧**（无 `PreTrainedModel` 子类、无梯度路径） |

**D. 方法论债 —— 最该记的一类**

| # | 债 |
|---|---|
| V175 | **注入自检缺失**：`inject=0` 产出过一份「全臂零开销」的漂亮假结果，靠事后补计数才发现。**凡 hook/注入类实验必须先自证触发过** |
| V176 | **单变量分离缺失**：没有配对臂，就把 keygen + 介质笼统归给磁盘 |
| V177 | **eager 下测 5% 预算的效度问题**（见 §35.1） |

另有两处**文档/工具**更正已随本轮提交：README §5.2 的「未验证 serving 配方」加边界；
`bench_serving_ab.py` 的真实表几何由硬编码 128 分片改为可配（否则指向半表会静默错映射）。

### 35.2b 收盘补记（V178–V181）：接入面只有一行，而 SGLang 没有那一行

**E. 结构性的（不是「还没做」，是「本机/本引擎做不到」）**

| # | 事实 |
|---|---|
| V178 | **本机不存在含 PLE 的 checkpoint。** 带 `ple_layer_ids` 的 config 只有 `Qwen3.8-Flash-Next-FP8-tokenizer`（22 MB，只有 config + tokenizer，**无权重**）；三个 Qwen3.5（0.8B/2B/4B）的 `text_config` **一个 PLE 字段都没有**。⇒ 子条件 1b（PLE 路径用模型自己的权重）在本机**不可达**；16 行/token 的 PLE 臂**只能**是合成投影。要闭合需 125B 主干 + 51B 表，或官方放出小号 PLE 模型 |
| V179 | **SGLang 0.5.19 没有 PLE 这个「缝」。** vLLM 0.29.0 有 `Qwen4ExpForConditionalGeneration`（`vllm/models/qwen4_exp/`，含 `_validate_ple_layer_ids()`）；SGLang registry **无条目**、全 `sglang/srt` **无 `ple_layer_ids`**、venv-sg 的 transformers 5.12.1 **无 `qwen4_exp`**。⇒ 子条件 6 从「❌ 未做」改判为「❌ **不可做**」。这不是我们库的兼容性问题 |
| V180 | **在测了 6 类代价之后，才第一次验证「读的是不是正确的行」。** 直到本轮才有 `scripts/ple_rowid_exactness.py`。此前所有 serving 数字都以「rowid 正确」为前提，而那个前提从未被检验过 |

**F. 本轮的正向结论（应写进对外表述）**

**接入面收束到 vLLM 里的一行**，而且**引擎自己已经把它切好了**：

```python
# vllm/models/qwen4_exp/nvidia/ple_layer.py
ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
return self.ngram_embedding(ngram_ids).flatten(-2)      # ← 唯一需要改的行
```

1. **上半行已证逐位相同。** 拿引擎自己的 `compute_ngram_ids`（未修改的函数体）当裁判，
   对照我们的生产 Rust 路径：37 用例 / 18,048 个 rowid / **IDENTICAL**，multiplier 由
   `seed=1234` 独立推出。见 `probes/ple_rowid_exactness_session42.md`。
2. **切口是引擎给的。** 源码注释：*"Keep num_reqs-dependent ID generation outside
   PIECEWISE CUDA graphs"* —— 算 rowid 与取行在引擎里本来就是两件事。
3. **下半行无路可走，那正是我们的位置。** `ngram_embedding` 是 GPU 常驻的
   `PLEVocabParallelEmbedding`；vLLM 0.29.0 的 offload 只有**整张量 / 整层**两种粒度
   （`cpu_offload_gb` 按参数名段 + GiB 预算；`offload_group_size` 按 decoder layer 分组）。
   PLE 表是 layer 2 里的**单个 51 GB 张量**，任何配置都只能整表搬运，
   **没有「按 rowid 取 16 行 = 2,560 B」这个粒度**。这是粒度问题，不是配置问题。

| V181 | `docs/prefetch-lead-time.md` §6 把 **Qwen3.5-0.8B** 与 `ple_layer_ids=[2]` 混为一谈（前者根本没有 PLE）。已改成一张来源表：`N` 取自实测模型，`L` 与 I/O 形状取自目标架构，注入是合成的。**凡跨模型引用的量都要标来源。** |
| V182 | **冷态自检对持有 mmap 的臂是瞎的。** `fadvise(DONTNEED)` 走 `invalidate_mapping_pages()`，**跳过被进程映射的页**。第一轮 `mmap` 臂一直持有 `np.memmap`，自检报 `cold` 而该臂实为温态（3814→850 µs 单调衰减）。加 `release()` 后平在 3801–3865 µs，**代价 −17.7% tok/s**。⇒ 「我 fadvise 了所以我是冷的」在有活映射时是错的；**冷态自检必须与工作负载读同一组页**（本轮的自检抽样 8 页 pread，看不见 mmap 的驻留） |

**G. 本轮唯一高于噪声地板的 tok/s 差异**

修正后 `mmap` 冷态 = **3801 µs/token（−17.7% tok/s，占单步 14.8%）**，
对比我们的批量 pread **195.9 µs/token（−0.2%，在 +2.6% 漂移内）= 19.4×**。
两者都对照同一个 500 µs 预算：**39% vs 760%**。

这条把「磁盘从来不是代价」的旧叙述**限定清楚了**：在 batch=1、表 47.7 GiB **且真冷**时，
用**持映射**的方式读要付 15% 的单步代价；用我们的批量 pread 付 0.9%。
早先「mmap ≈ none」是温态映射的假象。

### 35.3 计划：四个阶段，每阶段一个可证伪的退出标准

**Phase A（1 session，最高杠杆）—— 闭合真 serving 六条子条件**
不做新功能。只做：模型**自己的** ngram embedding（非随机投影）/ 冷态 `fadvise` /
完整分片或显式标注半表 / CUDA-graph 或显式标注 eager。
**退出标准**：README §6.1「GPU 端 vLLM/SGLang A/B 差距 ≤5%」从 ⏳ 变成一张六列 ✓/❌ 表。

> **已执行（Session 42 尾）**：表已闭合，且**多出一条原本没有的判据** ——
> 子条件表从 6 条变成 7 条（含 1a/1b 拆分），其中 **1b 与 6 是结构性不可达**（V178/V179），
> 不该继续挂在待办里当「还没做」。**Phase A 的真正剩余项只有 4（CUDA graph）与 5（counterbalance）**，
> 二者都需先把 reader 变成 splitting op。
> **新增的 7 号判据（rowid 逐位一致）已 ✅** —— 它比其余六条的优先级都高：
> 代价再低，读错行也是零分。

**Phase B（1–2 session）—— 换裁判**
测 `O_engine / F_storage` 随配置的变化。这是唯一能让「≤5%」在 CUDA graph 下仍可测的表述。

**Phase C —— 四条接入面按「谁能给出可证伪的数字」排序，不要四条都浅**
1. vLLM（最接近，只差 Phase A）→ 2. PyTorch 研究者（不需 GPU）→
3. ~~SGLang（补 V162 另一半）~~ → 4. llama.cpp（零代码，最后）

> **排序已改（V179）**：SGLang 从第 3 位**移出**。它缺的不是我们补的验证，
> 而是 `qwen4_exp` 这个架构本身 —— 没有 PLE 的缝，就不是「接入面」。
> 要恢复它只有两条路：等 SGLang 上游实现 `qwen4_exp`，或把我们的接入点
> 从「PLE 缝」上移到**输入嵌入缝**（后者每个引擎都有，且已证逐位相同 —— 见
> `probes/serve_faithful_embed_ab_session42.md`）。**后一条本身可能就是正确答案**：
> 它是唯一在所有引擎里都存在的缝。

**Phase D —— V4.1 只做接口形状，不做实现**（§29.3 已定）。
本轮补上 per-module deadline 的**量化依据**：layer 1 需 **293K IOPS**，layer 14 只需 **21K**，
差 **14×** ⇒ 「按模块分配存储层级」的倍数依据（`scripts/prefetch_lead_time.py` 可复算）。

### 35.4 借鉴什么，且为何不冲突

功能借鉴矩阵见 §3 / §11.3。本轮证明**方法论借鉴比功能借鉴更值钱**：

| 来源 | 借鉴 | 为何不冲突 |
|---|---|---|
| vLLM #54070 | 它**公开写了负面结果**（「不限容时反而更差」「冷 cache 是下界」）。本轮的 `prefetch-ub 慢 2.7×`、`批量为敌不成立` 是同类资产 | 他们绑引擎；我们产出任何消费方都能用的测量 |
| SGLang #36567 | reader 是 **Rust + PyO3 + GIL-free** —— 本轮证明这是**唯一可行**的预取路径（Python 级后台预取因 GIL 争用实测更慢 2.7×） | 同形接口，不重叠 |
| DuckDB | 「**先定验收，再写代码**」；benchmark 外部可复跑、不自证 | 我们缺的正是这个 |
| SQLite | 「**一个裁判**」（sqllogictest 是唯一真值） | 我们现有三个真值（500 µs / ≤5% / byte-identical）互不覆盖 |
| MLPerf | 「**规则先于结果**」：提交前锁定测量协议 | 本轮三次测量事故（假零开销、`comm` 假命中、冷启动混入中位）全是协议未锁定 |

**最该偷的一条**：MLPerf 式的**测量协议 checklist**。
`inject=0` 那次若有「hook 必须自证触发」一条，5 分钟即可发现，而非靠事后补计数。

### 35.5 一句话

**北极星没变，裁判该改了。** 本轮最有价值的产出不是那个 843× 的修复，
而是证明：**我们一直在用一个引擎无关的预算去验收一个引擎相关的目标，
于是既容易「达标」、又无法前进。**

---

## 36 目标重述与优先级重排（Session 43）

> §35 结尾说「北极星没变，裁判该改了」。本轮把这句话做完了：
> **裁判改成了一个可判定的不等式，而且它立刻回答了一个悬了很久的问题。**

### 36.1 北极星：愿景保留，内核改写

**愿景不变**（§1）：

> 让确定性哈希的 n-gram 记忆表（Qwen PLE / DeepSeek Engram）成为任何小模型与
> 推理/训练框架都能廉价使用的磁盘优先存储基础设施。

**内核（§29.1）需要改写。** 原文是：

> 在一块消费级 NVMe 上，使**每 token 的 Engram 相关开销落进 5% 端到端预算**。

它的操作化形式是 §29.1.1 反推出来的 **500 µs/token**（= 5% × 100 tok/s = 10 ms）。
**本轮第一次真的测了分母，而这个反推的前提被推翻了：**

| 引擎 | 模式 | ms/token | 该分母下的 5% |
|---|---|---|---|
| vLLM 0.29.0 | eager | 21.28 | 1064 µs |
| vLLM 0.29.0 | **CUDA graph** | **2.92** | **146 µs** |
| SGLang 0.5.19 | eager | 17.76 | 888 µs |
| SGLang 0.5.19 | **CUDA graph** | **2.27** | **113 µs** |

**CUDA graph 把分母改变了 7.3–7.8× —— 这个倍数比我们做过的任何存储优化都大。**
冷读 195.9 µs 在 500 µs 口径下占 39%（达标），在 graph 口径下占 134%（**超标 2.7×**）。

**但 500 µs 不是「错」，是「没有锚」**：它假设 100 tok/s（大模型的数字），
而我们只有 0.8B 代理模型的分母。**我们既不知道真分母，也不知道 PLE 层的位置。**

⇒ **改写后的内核（唯一裁判）：**

> 在目标架构的 **PLE 层**之前，该层所需的全部行**已经就绪**：
> `t_read(rows) ≤ τ(L_ple)`，且该条件**单机可测、有门禁、有安全余量**。

把「读 ≤ X µs」（绝对值）换成「读 ≤ τ(L)」（**调度可行性**）。
好处：它把「该优化存储，还是该找个更深的层」变成**一个算得出来的问题**，而不是一句口号。

### 36.2 判据化简：`L* = ⌈t_read / 单层时间⌉`

把 §29.1.1 的 `τ(L) = O_inter-step + (L/N)·C`（`prefetch-lead-time.md` §1.2）
代入 `t_read ≤ τ(L)`，并取保守的 `O = 0`：

```
L ≥ N · t_read / C
```

而 `C/N` **就是单层前向时间** ⇒ `N` 消失：

> ### `L* = ⌈ t_read / 单层前向时间 ⌉`
>
> **所需层号 = 读耗时 ÷ 单层 GPU 时间。** 没有别的参数。

**这个式子立刻被实测验证。** SGLang graph 2.27 ms / 24 层 = 94.6 µs/层，
冷读 195.9 µs ⇒ `L* = 3`。

> ### ⚠️ Session 44 更正：真实几何是 **48 层、0-based 第 1 层**
>
> 本节初稿写的是「Qwen PLE 在第 2 层（共 24 层）」——**那是替身模型的层数**
> （`Qwen3.5-0.8B`，我们唯一能真跑的那个），被错当成了 PLE 模型的几何。
>
> 真实几何现在取自 `Qwen3.8-Flash-Next-FP8` 的**权重索引本身**
> （`model.safetensors.index.json`，152,089 条权重、`total_size` 185.5 GB）：
>
> ```
> layer idx: min 0  max 47  count 48        ⇒ num_hidden_layers = 48
> PLE LAYER INDICES: [1]                    ⇒ 权重在 layers.1.ple.*，即 0-based 第 1 层
> ```
>
> `text_config.num_hidden_layers = 48`、`hidden_size = 2560` 也由此**从「未验证」升为已验证**
> （二者此前只是从机器上读来、未入库）。`ple_layer_ids = [2]` 是 **1-based**，
> 与 `layers.1` 并不矛盾。
>
> **裁决方向不变，幅度更差：需要 3 层，只有 1 层 ⇒ 差两层，不是差一层。**
> 原先那句「与 `τ(2) = 189.2 µs` 精确吻合」也随之作废 —— 该拿 `τ(1)` 比，而它从未被测过。
>
> **还有一条方法学保留**：94.6 µs/层是**整步平均**（2.27 ms ÷ 24），
> 里面**混着与层数无关的固定开销**（采样、attention metadata、prepare）。
> 若 `总时间 = 固定 + N×单层`，则 `94.6 = 固定/24 + 单层` ⇒ **单层 ≤ 94.6 µs**。
> 所以 `L* = 3` 是**下界**，真实需求只会更大 —— **下面的裁决全都是乐观界**。
> 要用斜率法（跑 24/12/6 层取差分）才能拿到真正的边际单层时间；这是待办。

可复跑：`scripts/lead_layer_budget.py`（所有输入都带出处，模型是纯算术）。

| 单层时间来源 | µs/层 | L*（16 行） | L*（48 行） |
|---|---|---|---|
| SGLang 0.5.19 graph，**Qwen3.5-0.8B 24 层**（实测） | 94.6 | 3 | 7 |
| vLLM 0.29.0 graph（实测） | 121.8 | 2 | 5 |
| V4.1 `τ(14)=2295 µs`（实测） | 163.9 | 2 | **4** |

**⇒ 两个真实几何的裁决：**

- **Qwen PLE 在 0-based 第 1 层（共 48 层）：差两层。** `L* = 3 > 1`。
  换成原来那个「第 2 层」的口径也只是差一层 —— **无论按哪种读法，它都装不下**，
  而真实几何比初稿更紧。**当前 Qwen 的层位置不在可行域边缘，是在可行域外面。**
- **V4.1 Engram 在第 14 层（共 48 层）：需要 4，有 14，余量 3.5×。**
  **V4.1 把 PLE 放那么深不是保守，是刚需的反面 —— 它有大量余量可以往下挪。**

#### 一个反直觉的推论：并发度是**可行性**要求，不是吞吐优化

冷态的代价由**页错误**主导（每行一个 4 KiB 页），所以它随**行数**而非字节数增长
（模型里显式写明这一点：换成按字节缩放，V4.1 的数会大 4.8×）。
单线程 85.8 µs/行：

| | L*（16 行） | L*（48 行） |
|---|---|---|
| 有线程池 | 3 | 4 |
| **无线程池** | 15 | **26** |

**⇒ 没有线程池，V4.1 在第 14 层是「装不下」的（需要 26 > 14）。**
`18.1×` 的并发收益不是「更快」，**它是让浅层 PLE 可调度的那件事**。
这一条应该写进任何对外材料：**并发度不是可选项。**

### 36.3 盘点：完成了什么

按**资产可靠性**分层（不是按时间）：

| 层 | 内容 | 证据 |
|---|---|---|
| **存储面** | Store-P 视图 **28.6–41.5×**；并发 **18.1×**；线程池再 **1.9×**；全表冷读 **195.9 µs/token** | `serve_ple_ab_fulltable_session42.md` |
| **一致性**（最强资产） | rowid 与引擎自己的 `compute_ngram_ids` **逐位相同**：37 用例 / 18,048 rowid / IDENTICAL；四路径一致 | `ple_rowid_exactness_session42.md` |
| **分母** | 引擎地板 2×2，两引擎都有 | `engine_floor_session42.md` |
| **机制**（本轮，⚠️ 未验证） | 两引擎装机版本都有一等公民断点机制；vLLM 对 PLE 家族**自动开启**；接入 = 一个开关 + 一个装饰器 | `docs/cuda-graph-injection.md` |

### 36.4 盘点：未完成什么

**关键路径上只有一项：子条件 4（CUDA graph）未闭合。**
它的性质已经变了 —— **不是「存储不够快」（早已 11–31× 富余），
而是「我们还没证明这个读不在关键路径上」。**

在它闭合之前，README 里**每一个 GPU 百分比都是合成的**，
包括现在最常引用的 6.70% / 8.63%（那是「eager 测的分子 ÷ 另一次运行测的分母」，
**不是一次测量**）。

| 项 | 状态 | 阻塞内核？ |
|---|---|---|
| 真分母锚定（目标模型 graph 步长） | ❌ 无 | **是** |
| 真 PLE checkpoint 上的 A/B | ❌ 本机无此 checkpoint | **是** |
| V4.1 真机上验证 `τ(L)` 与层位置 | ❌ 需 510 GB 机器 | 否 |
| V4.1 支持落地（keygen v2、存储参数化） | ⏳ 输入已冻结，纯本地可做 | 否 |
| 训练流 ≥100K tok/s | ⏳ | 否 |
| io_uring 收益 | ❌ **本机无法验证**（seccomp `EPERM`） | 否 |
| SGLang 侧真实 PLE | ❌ **结构上不可做**（无 `qwen4_exp`） | 否 |

#### 元债务：必须点名

```
roadmap 复选框：未关 217 / 已关 63   →   77.5% 未关
```

Session 40 基线 **76%**，§29.4 的硬约束是「**净关闭率 > 0**」。
**我们现在更差。** 而 Session 42–43 净增三份文档（roadmap 4047 → 4400+ 行，
另加 625 行新文档），**验收表关闭 0 项**。

这正是 §29.2 自己警告过的「债务表从『待办』退化为『日记』」。
**调研质量可以认可，但它不是关闭。**

### 36.5 优先级

排序原则：**先钉住裁判，再谈成绩。**

> **Session 44 的系统性思考在 §38** —— 它把目标收紧为「没有任何一层装得下、且无 HMM 时仍可服务」，
> 并给出四条 Track（A 非 HMM 分支落地 / B 上游对话 / C io_uring / D 冻结）与第十条纪律。
> **本节的 P0.5′ 与 P2.5 是 §38.4 Track A/B 的实现条目。**

**P0 — 锚定分母（前置，否则后面全是空谈）**
- 已完成（本轮，无 GPU）：把 500 µs 标注为带假设的推导（§29.1.1）；
  把判据化简为 `L* = ⌈t_read/单层时间⌉` 并复现实测（§36.2，`scripts/lead_layer_budget.py`）；
  得出「V4.1 余量 3.5×、Qwen 差两层（48 层、0-based 第 1 层）、无线程池则 V4.1 装不下」。
- 待做：在能放下 PLE 模型（或结构等价）的机器上测 **graph 模式的真实步长**，
  把 `L*` 从「代理模型推算」升级为「目标模型直测」。

**P0.5 — 换目标：把存储接到上游真实的缝上（Session 44 新增，最高优先）**
- **理由**：§37 确认上游已有 PLE offload（SGLang `main` 已发布 `backend="file"`；
  vLLM #54371 UVA 已合并 + #54070/#54129 两个磁盘 PR），**而主线的缝无断点** ——
  我们为替身缝建的 apparatus 不迁移。
- **动作（读代码 + 改接口，不是 GPU 工作）**：
  读 `Qwen4ExpNGramEmbedding` / `Qwen4ExpPinnedHostEmbedding` /
  `srt/models/qwen4_exp_ple_table.py`，确定我们的存储插在哪里。
- **最便宜的一处**：`vllm/v1/ple_offload/worker.py` 的 `PleOffloadRunner.__init__`
  权重发现循环，加 `_ple_disk_attach` 式钩子（契约 = `_gpu_output_buffer` + `_sem`
  + copy stream 上 signal `DONE_VALUE`）。**无需 graph break**，#54070 已证明可行。
- **判据**：能在**上游 main**（不是我们的探针）上跑出一个冷态数字。

**P1 — 关闭子条件 4（机制验证）—— ✅ 已完成（§36.8）**
- 四项判据**同时**成立：后端是 breakable（`backend_replay = 553/512`）/
  `break_calls_in_replay = 552/512` / 输出确实改变（三臂互异 sha1）/
  **replay 期确认在 BCG 内**（`BreakableCUDAGraph.replay` 的 flag，而**不是**
  `is_in_breakable_cuda_graph()` —— 后者在 capture 也为 True，会空洞通过）。
- 已顺带产出 graph 模式下的存储数字，但那笔是**串行上界**，不是存储代价（§36.8）。

**P1.5 — stage 2：把 1.30 ms 摘出关键路径 —— ⏸️ 建议冻结（Session 44 末尾降级）**
> **降级理由**：它优化的是**替身缝**。§37 确认主线集成**按构造无断点**
> （vLLM 图内 `cuStreamWaitValue32`；SGLang `file` 后端直接在 file-backed CPU 张量上
> gather），所以这套断点优化**不迁移到主线**。
> **例外**：SGLang #36567（io_uring，host 侧发起）**硬依赖 breakable backend 且从未被
> 验证** —— 若决定走那条线，本项立即恢复为 P1，且我们已有它缺的验证与 `τ(1)`。
> 下面的相位分解结论仍然有效（它是**成本模型**，与缝无关），只是不再驱动开发。

> **本条下面的归因在 Session 44 被实测全部推翻，原文保留作失败留档。**
> 它们是**臂间差值**的产物，从 ±0.2 ms 的噪声里读出了看起来很确定的答案。

- ~~`read_static` 判决实验把 1.30 ms 定死为「磁盘 0.68 + D2H 0.39 + rowid 0.24」~~
- ~~① 持久线程池 + 跨 token 批量化 ② 消灭 D2H ③ rowid 移出关键路径~~
- **实测（相位分解 + reader 对比 + 引擎内复测）**：
  | 项 | 差值法说 | 实测 |
  |---|---|---|
  | `read_us` | 445–468 µs（当作存储代价） | 那是**手写 Python reader**；原生 `Store.fetch` 引擎内 **101–108 µs**（同批冷行 276.7 vs Python 池 554.6 µs） |
  | D2H | ≈387 µs | 闲 **11.5 µs** / 忙 **2428 µs** —— 它等的是**整段在途流水线** |
  | rowid | ≈240 µs | **~50 µs 固定成本**，128 token 才 68.6 µs |
  | Python 池并发 | （隐含很高） | **3.3× 饱和**；批量化到 128 token 无改善 |
- **修正后的四步**：① **消灭 D2H**（最大项，等的是整段流水线）
  ② **改用原生 `Store.fetch`**（换实现，不是加并发）③ pinned 暂存 + `non_blocking=True`
  （18.8→11.2 µs，小但免费）④ rowid 移出关键路径（**收益最小，不值得优先做**）。
- **前置已兑现**：成本分解仪器（`scripts/sc4_phase_microbench.py` +
  `scripts/sc4_reader_compare.py` + in-situ 相位计时）已建成。
  **教训：差值法不能用来归因。** 本轮每一个被推翻的数都来自「两个噪声总量相减」。
- 判据：`read − break` 落到噪声地板以下，且 `break_calls_in_replay` 仍 > 0。
- **前置（Session 44 从旧复盘提升）**：先做**成本分解仪器**（原 Phase 0 的
  `--profile-embedding`，分离 fetch / convert / compute）。Session 43 只能靠
  `read_us` vs `break_us` vs 整步差值手工分解，噪声 ±0.2 ms，
  「磁盘与 D2H 各占一半」这个结论就是这么被噪声骗出来的（`probes/subcondition4_sglang_session43.md` §6.1）。
  **没有这个仪器，stage 2 的每一步都无法判断有没有真的回收。**

**P2 — 把 `L*` 落到本机**
- 机制通了之后，第一个真问题不是「多快」，而是：
  **在本机模型上，把读放在哪个 L 能让净增落到噪声地板以下？**
- 这是唯一能把「195.9 µs 超标 2.7×」变成「净增 ≈ 0」的动作，可测、可证伪、单机可复现。
- **前置一**：`L*` 的分母（边际单层时间）必须用斜率法测，不能用整步平均（§36.2 的保留）。
- **前置二（Session 44 从旧复盘提升）：先测 rowid 访问分布。**
  真实语料里 rowid 若有显著重复，页缓存就会帮上忙，Session 43 的**冷读最坏情况过于悲观**，
  `t_read` 与 `L*` 都要跟着改。旧复盘的 Phase 0 里写着「LRU 增加 hit rate / rowid 重复率」，
  但从未做过 —— **而它决定 `L*` 的分子，与前置一决定分母同等重要。**
- **前置三（同批提升）：native `rowid + gather + dequant` 的边界。**
  README 只声称 rowid 四路径一致；`gather + dequant` 是否已在 native 侧、
  Python convert 占多少，**从未测过** —— 而 Session 43 测得「H2D + add ≈ 118 µs」
  暗示 convert 这一侧不是零。旧复盘的 Phase C 有这一条，也没进过活账。

**P0.5′ — 楔子已被上游代码确证（§37.7 第一手核实）**
- 上游 `file` 后端是 **HMM 硬门控**（非 HMM 设备直接 raise），
  且预取在 **< 2048 行**与 **capture 期间**两处提前返回 ⇒ decode 路径上预取从不发出。
- **⇒ 在非 HMM 硬件上没有上游磁盘路径；在 HMM 硬件上最需要提前量的路径被关掉了预取。**
- 这使 P0.5 的目标具体化：**不是「接进上游」，而是「补上游拒绝跑的那一类硬件」**，
  并复用我们已实测的 `τ(L) ≥ ~325 µs` 作为设计约束。
- **待测（需要 HMM 机器）**：一次 decode gather 的 fault 驱动停顿有多长（§37.7 边界）。

**P2.5 — 把成本模型与 `τ(L)` 交给上游（Session 44 新增）**
- 这是**唯一上游没有、且对 #36567 直接有用**的东西：上游只报端到端差
  （disk −8% @c1 / −17% @c32），**从不问「这个读要提前多久发出」**。
- 我们的答案是：**≥ ~325 µs**，而 PLE 在 layer 1 恰好只给这么多。
- 形态：一条 issue / 一份可复现脚本，而不是又一个功能。
  `PleFilePrefetcher` 用了 WILLNEED，但**前置量**没人量过。

**P3 — 之后才谈广度**
- V4.1 接口、多表、Arrow IPC、engram-peft、C ABI 补齐 —— 产品面。
  在 P2 出结果之前投入，边际价值接近零。**建议正式冻结**（与 §29.4 Phase 4 一致）。

### 36.6 流程纪律（在 §29.6 五条之上新增三条）

**第六条（Session 44 重写）：债务只在活账上算，且指标必须可能变正。**
本条原文是「把调研也计入净关闭率，连续两轮为负就停下来做减法」。
**它的诊断是对的（Session 42–43 确实在净增文档而零关闭），但它选错了分母**：
它让 217 个散落在已死计划世代里的复选框当分母，于是这个比率**只可能变负**，
无论做多少实事。Session 44 已把这个指标废止并合并账本（见 §0）。

**改后的条文**：
1. 债务只在**活账**（README §6.1 + 本文件 §36.5）上计算，分母是活账的条目数。
2. **每一代新计划必须先把上一代活账结清或明确退役**，不许直接叠加 ——
   这正是 217 个复选框的成因。
3. 仍然必需却只躺在旧复盘里的条目，要**提升**进活账（Session 44 提升了三项），
   而不是继续让它躺着。
4. 若某指标在结构上不可能变正，**先修指标，再谈进度** —— 但修指标必须同时
   做一次真实审计（提升 + 退役），否则就是移动球门。

**第七条：区分「已确证」与「已读源码」。** 本轮 §35.1d 与
`docs/cuda-graph-injection.md` 全部是源码阅读结论，文档里都标了 ⚠️，
**但这要成为纪律而不是自觉** —— 上一轮正是因为把推论当结论，白跑了八次 GPU 运行。

**第九条（Session 44 新增）：关于上游能力的断言，必须落到 main/目标 branch 的具体 path 上。**

来源是一次真实的失败：本轮我**两次**断言「SGLang 没有 Engram/PLE」——
对已安装的 0.5.19 为真，**对 `main` 为假**（`main` 已有 `config.ple_offload_backend="file"`
的 file-backed PLE 表）。根因是**从 release + 一条关于 branch 的 bug 报告推理上游主线**。

规则：① 用 `api.github.com` 而不是 `github.com`（HTML 取回来几乎全是导航框架）；
② 断言必须带 **commit / tree 来源**；③ **「我装的版本里没有」不等于「上游没有」**；
④ 动手建 apparatus 之前先确认目标缝在目标 branch 上存在。
**同一类错误在同一 session 里出现了两次**（另一次是把 Python reader 的开销记到存储头上）——
共同点都是**验证了替身，没验证目标**。

**第八条：验收表上只保留有前置条件的项。** 「待硬件」不是状态，是**没有承诺**；
把它和「未验证」并列，会让所有其他工作都可以自称在为它做准备（§29.2 第 3 条失速）。

### 36.7 一句话

> 存储已经做完了，一致性是全项目最强的资产。
> **真正卡住的从来不是「读得够不够快」，而是「我们还没证明这个读不在关键路径上」。**
> 裁判已换成一个不等式，而且它当场给出了三个判决：
> **Qwen 差一层、V4.1 余量 3.5×、并发度是可行性的前提而不是优化。**
> 下一步只有一个动作：机械地闭合子条件 4，然后把 `L*` 落到本机。

（上句里的「Qwen 差一层」已在 §36.2 更正为 **差两层**：真实几何是 48 层、0-based 第 1 层。）

### 36.8 子条件 4 闭合（session 43，同日完成）

**§36.5 的 P1 已关。** 上一轮唯一未达成的那一项 —— 「`enforce_eager=False` 时 op 在 replay
中执行」—— 在 SGLang 上达成了。完整证据在 `probes/subcondition4_sglang_session43.md`。

**为什么是 SGLang 而不是 vLLM**：不是配置差异，是机制差异。vLLM 的 `FULL_AND_PIECEWISE`
让 decode 取 `FULL`，`splitting_ops` 切分点被绕过，replay 不跑任何 Python；SGLang 的
breakable graph 在**段间**执行真 host Python。**本轮最硬的旁证**：`forward_calls = 26`
而 `backend_replay = 553` —— replay 时模型的 Python `forward` 一次都没被调用，
**唯一会跑的 Python 就是断点函数**。

**数字**（`none 418.9 / break 393.5 / read 264.1 tok/s`）：
`read − break = +1.245 ms`、`read − none = +1.399 ms`。

**但这个数字不是存储代价，而且这一条比数字本身重要。** 它是首次尝试、完全串行的上界：
本轮 `read_us_median = 464.6 µs` vs 调优过的冷读 **195.9 µs**；`break` 臂**无磁盘**也有
504.5 µs。正面证据是 `break` 臂 504 µs host 工作只让整步慢 154 µs（≈70% 被前段 GPU kernel
掩盖）⇒ **重叠机制在工作，打断它的是 D2H 同步**。

#### 三个可上报的上游缺口（都在本轮实测撞到）

1. **BCG 后端不认识 dataclass 输出**：`_alloc_full_buffer` 只认
   `None/Tensor/PPProxyTensors/tuple/list`，遇到 `LogitsProcessorOutput` 直接
   `TypeError` ⇒ **几乎任何文本模型都无法用 `--cuda-graph-backend-decode=breakable`**
   （与断点无关：`qwen3_5.py` 里一个 `@eager_on_graph` 都没有也在 capture 阶段就炸）。
   修法有据：隔壁 `_copy_output` **已经支持** `__dict__` 对象，只是后端这四个没跟上。
2. **BCG 的「静止 buffer + 普通加法」在 vLLM 上失败的原因是 Inductor 常量折叠**
   （session 42 的 W4）。**BCG 不经过 Inductor**（它是 stream-capture 切段）
   ⇒ 同一个想法在 SGLang 上天然成立。这是选 SGLang 的第二个理由。
3. **跨进程注入**：`mp.set_start_method("spawn")` 让父进程 monkeypatch 失效
   （`serve_sglang_baseline.py` 曾把这条记为「SGLang 做不了磁盘臂」的原因）。
   出路是**挂到子进程本来就会 import 的模块末尾**，对 spawn/fork/exec 全有效。
   另外两个坑：子进程 PATH 缺 venv bin 会导致 `ninja` 找不到（看起来像 SGLang 崩了）；
   `engine.shutdown()` 用 SIGKILL ⇒ 子进程侧计数器必须由**后台线程**周期落盘。

#### §36.4 的「未完成」清单相应变化

- ~~子条件 4（CUDA graph 路径）~~ → **已闭合**。
- **新增 P1：stage 2（重叠）**。目标已被精确定量：**把 1.245 ms 从关键路径上摘掉**。
  手段按 Session 44 的**实测**重排：**消灭 D2H**（闲 11.5 µs / 忙 2428 µs，等的是整段
  在途流水线）→ **改用原生 `Store.fetch`**（引擎内 101 vs 手写 Python 445 µs）→
  pinned 暂存 + `non_blocking=True` → rowid 移出关键路径（收益最小）。
  这四条正是 `docs/cuda-graph-injection.md` §8.1 早已写下的前提，现在每条都有了实测支撑。
- **`L*` 的单层时间是整步平均**（含与层数无关的固定开销）⇒ **所有 `L*` 都是下界**。
  斜率法（24/12/6 层取差分）仍未做。
- **真实 PLE 模型跑不了**：`Qwen3.8-Flash-Next-FP8` 权重索引 `total_size = 185.5 GB`
  （48 层、512 专家 MoE、FP8），单张 4090 装不下 ⇒ 本轮用 `Qwen3.5-0.8B` + 合成投影，
  **只测代价，无任何质量声明**。

### 36.9 活账的首次测量（Session 44）

按改写后的 §36.6 第六条，债务只在**活账**上算。第一次测（`scripts/ledger_rate.py`）：

| | 行数 |
|---|---|
| README §6.1 验收目标 | 9 |
| README §6.1 六条子条件 | 8 |
| **合计** | **17** |
| ✅ done | 8 |
| ⚠️ partial | 5 |
| ❌ open | 4 |
| **闭合率** | **47.1%** |
| 非「完全未做」率 | 76.5% |

**4 个完全未开的行，以及各自为什么开着：**

| 行 | 为什么还开着 |
|---|---|
| CPU 小模型 decode（真机）≥50 tok/s | ⏳ **「待硬件」不是状态，是没有承诺**（§36.6 第八条）。要么给前置条件，要么退场 |
| GPU 端 vLLM A/B ≤5% | 6.70%/8.63% 仍是**两次测量相除的合成值**；单次测量已在 SGLang 上做出（子条件 4′），但那是串行上界 |
| 训练流有效吞吐 ≥100K tok/s | 未闭环，且**不在当前关键路径上** |
| 子条件 1b「PLE 路径用模型自己的权重」 | 本机**不存在**带权重的 PLE checkpoint（`Qwen3.8-Flash-Next-FP8` 是 185.5 GB / 48 层 MoE，单张 4090 装不下）。**这不是未做，是当前硬件下的不可做** —— 与子条件 6 同类，应当**明确改判**而不是一直挂 ❌ |

**注意 `4′` 被判为 partial 是正确的**：它虽然产出了数字，却自己标注了「这不是存储代价」。
分类规则让 ⚠️ 优先于 ✅，就是为了不让行内的交叉引用把「有前提的完成」洗成「完成」。

**已做成棘轮**（`scripts/gate.sh` + `.github/workflows/ci.yml`）：

```
python scripts/ledger_rate.py --max-open 4
```

`--max-open` 钉在今天这个值上，**所以这个数字只可能变好**。
用机器算而不是手数，是因为第一次手工统计就把介质表、文档导航表和几何表都算进去了，
还让行内的 `✅` 交叉引用把两行 open 洗成了 done。

> **诚实边界**：47.1% 比「217 未关 / 63 已关 = 22.5%」好看，但**这两者不可比** ——
> 分母完全不同，而且旧指标衡量的是「有多少代死掉的计划还没删」。
> 本节的价值不是数字变好，而是**第一次有了一个可能变好的数字**。
> 如果下一轮闭不上任何一个 open 行，那才是真信号。

### 36.10 `τ(L)` 被测出来了，判决随之改写（Session 44）

§36.2 的化简 `L* = ⌈t_read / 单层时间⌉` 一直没有被认真对待的一环是**分母**：
`94.6 µs/层` 是 `2.27 ms ÷ 24 层`，即**整步平均**，把与层数无关的固定开销
（采样、attention metadata、prepare）也摊进了每一层。我标过「只是下界」，但没有替代品。

**现在有了。** 在断点里加**可控忙等**（`ENGRAMDB_SC4_DELAY_US`）并扫描：

| delay µs | ms/tok | Δ vs `none`(2.410) | break body | 被藏住 |
|---|---|---|---|---|
| 0 | 2.468 | +58 | 345 | **287** |
| 100 | 2.575 | +165 | 510 | **345** |
| 200 | 2.798 | +388 | 569 | 181 |
| 400 | 2.863 | +453 | 799 | 346 |
| 800 | 3.548 | +1138 | 1043 | **−95** |
| 1600 | 4.311 | +1901 | 1966 | 65 |

`break_us` 精确跟着 delay 走（345→1966，+1620 对 1600）⇒ 忙等本身是准的。

> ### **τ(1) ≈ 300–350 µs**（±150 µs），而不是 94.6 µs
>
> 真实窗口是「层 0 的 compute + 步间空隙」。**平均值低估了 3 倍以上。**
> body ≲ 800 µs 时能藏 300 µs 上下；超过之后 **1:1 全暴露，一点也不藏**。

**两个输入都实测之后，判决改写：**

| `t_read` 来源 | `t_read` | `L*`（÷94.6） | 落进 τ(1)? | 对 Qwen（0-based 第 1 层） |
|---|---|---|---|---|
| 手写 Python reader（旧） | 445–468 µs | 5 | ❌ | 差四层 |
| **原生 `Store.fetch`（引擎内）** | **101–108 µs** | **2** | ✅ | **装得下** |
| 原生（独立脚本，强制狠冷） | 276.7 µs | 3 | ✅ | 装得下（边缘） |

**⇒ 「Qwen 差两层」是被 Python reader 的产物驱动的判决，不是被存储驱动的。**
`probes/subcondition4_sglang_session43.md` 的 §6 因此加了更正横幅。

**边界（不要把这条读成「问题解决了」）**：
1. 扫描点间散布 ±150 µs，「300–350」是**量级**不是精确值；
2. τ(1) 是在 **Qwen3.5-0.8B（24 层）**上测的，真实 PLE 模型是 48 层 MoE、单卡装不下
   ⇒ **这是替身模型上的窗口**；
3. `t_read` 的两个原生数（101 引擎内 / 277 狠冷）差 2.7×，差在**冷的定义**
   （独立脚本每次 rep 前 `fadvise` 掉 6.4 GB 页缓存）。
4. **§36.2 那条「所有 `L*` 都是下界」的保留依然成立**，但它的理由换了：
   不是「分母偏小」，而是**分母是在替身模型上测的**。

#### 一条方法论教训（第九条纪律的候选）

> **差值法不能用来归因。**

Session 43 的每一个归因数（D2H 387 µs、rowid 240 µs、磁盘占一半）都来自
「两个噪声总量相减」；Session 44 每一个**直接测量**的数都推翻了一个差值结论。
`§36.5` 把成本分解仪器列为前置是对的 —— **在它存在之前，任何归因都只是猜测。**

---

## 37. 上游图景（Session 44 外部调研）：我们不是第一个，而且主线的缝不需要我们的断点

> 证据：`research/PLE_INTEGRATION_SURFACE.md`（每条带 URL，标 [C] 已读 / [U] 未确认）。
> **本节的价值主要是负面的 —— 它缩小了我们以为自己拥有的空间。**

### 37.1 已确认的上游事实

| 引擎 | 状态 | 机制 | 要 graph break? |
|---|---|---|---|
| **SGLang `main`** | **已发布** `config.ple_offload_backend="file"` | file-backed PLE 表；`PleFilePrefetcher`（`posix_fadvise(WILLNEED)`）+ `PleFileRssTrimmer`（`MADV_DONTNEED`，8 GiB 预算）；门控在 `cudaDevAttrPageableMemoryAccessUsesHostPageTables`（GB10 / DGX-Spark HMM） | **否** |
| SGLang #36567 | open，未合，栈在 #36497 | Rust **io_uring** + `O_DIRECT` + 队列深度 512；host 侧发起 | **是（硬依赖）** |
| vLLM #54371 UVA | **已合并**（2026-09-09，`3116c5d`） | 整个本地分片 pin 在 RAM，GPU 经 UVA 直读 | 否 |
| vLLM #54070 disk | draft，栈在 #53899 | 最大 parameter 换成 file-backed mmap + `MADV_RANDOM`；**gather 路径一字未改** | **否**（图内 `cuStreamWaitValue32`） |
| vLLM #54129 mmap | open，**独立设计** | Model Runner V2 的 input-prep 阶段 gather | 否 |

⚠️ **有两个磁盘 PR，不是一个。** #54371 的「Not a duplicate」比的是 **#54129**，不是 #54070。

### 37.2 三条关键推论

**① UVA 不取代磁盘 —— 缝隙是真的。**
#54371 把**完整分片** pin 在 RAM。`csrc/libtorch_stable/cuda_view.cu` 走
`cudaHostGetDevicePointer`，而**非 pinned 张量会被 `cudaHostAlloc` 出新的全尺寸 buffer
并整表 memcpy** ⇒ 一个 pageable file mmap 会**悄悄付掉完整的 ~95 GB**。
要让 UVA 看见它就得 `cudaHostRegister`（pin ⇒ 放弃分页）或 HMM/ATS。
⇒ **「宿主内存 < 表」这一类部署，上游没有解。**

**② 主线的缝不需要我们的断点 —— 这是本轮最贵的负面结论。**
两个 offload 设计**按构造就是无断点的**：vLLM 用 fake-impl custom op 在图内做
`cuStreamWaitValue32`（等待是**图节点**，不是 host 断点）；SGLang 的 `file` 后端直接在
file-backed CPU 张量上 gather。
⇒ **我们为「让 host Python 在 replay 期间执行」建的那整套 apparatus（断点函数、分层自证、
`is_in_breakable_cuda_graph` 的判据陷阱），在主线集成里用不上。**
另注：`@eager_break_during_capture` **默认是空操作** —— 不设
`VLLM_USE_BREAKABLE_CUDAGRAPH` 时直接返回原函数，且在 `CUDAGraphMode.FULL` 下也返回原函数。

**③ 但 #36567 正好需要它，而且它从未被验证过 —— 这是我们唯一确定对得上的缝。**
#36567 是 host 侧发起（`.to("cpu").tolist()` + `ThreadPoolExecutor` + 阻塞 `future.result()`），
硬依赖 breakable backend —— **而它自己的 speed test 是 eager 跑的**，diff 里没有任何
breakable / `FULL_AND_PIECEWISE` 的 capture 验证。
**我们独立证明了那个机制可行（`backend_replay` / `break_calls_in_replay` 分层自证），
并量出了窗口 `τ(1) ≈ 325 µs`。这正是 #36567 缺的那块验证。**

### 37.3 我们真正独有、上游没有的四件东西

1. **成本模型**：页错误主导、每 token 16 页、原生 `Store.fetch` 101 µs vs Python 池 445 µs
   （上游报的是端到端 tok/s 差，没有任何相位分解）。
2. **可调度判据 `L*` 与实测窗口 `τ(L)`**：上游只报「disk −8% @c1 / −17% @c32」，
   **不问「这个读要提前多久发出」**。
3. **「读必须提前 ≥ ~325 µs 发出」这条约束** —— 而 PLE 在 layer 1 恰好只给这么多。
   `PleFilePrefetcher` 用了 WILLNEED，但**前置量**没人量过。
4. **确定性哈希感知的布局**（`Store-P` / `SlotIndex` / view 物化）——
   上游只是把原始表 mmap 掉，没有布局层。**这是唯一一条上游完全没有的能力。**

### 37.4 最小改动（已确认到函数级）

- **vLLM（最便宜）**：在 `vllm/v1/ple_offload/worker.py` 的 `PleOffloadRunner.__init__`
  权重发现循环里加一个 `_ple_disk_attach` 式钩子。契约 = `_gpu_output_buffer` + `_sem`
  + 在 copy stream 上 signal `DONE_VALUE`。**无需 graph break**，#54070 已证明可行。
- **SGLang**：`main` 已有 `backend="file"`。若要自己的 store，子类化
  `Qwen4ExpPinnedHostEmbedding`（`srt/models/qwen4_exp.py:768`）覆写 `gather`（:861）——
  其 `start_prefetch`(:1145) / `_consume_prefetched_embeddings`(:1183) /
  `_graph_prefetch_buffers` 已经是 capture-aware。**可能什么都不用写。**
- **DeepSeek-V4.1 Engram**：在 **`dsv4.1` branch**（**不在 main**）：
  `srt/layers/engram.py` 的 `EngramHasher`；embedding gather 是 graph-safe 的
  （`EngramEmbedding._owned_rows` → `engram_gather` Triton，纯设备侧），
  hasher 才需要 `eager_on_graph`。

### 37.5 对活账的影响（§36.5 重排）

- **P1.5（stage 2）降级**：它优化的是**替身缝**。主线缝无断点，我们的断点优化不迁移。
  **建议：冻结，除非走 #36567 那条线。**
- **新增 P0.5（最高优先）：换目标。** 读 `Qwen4ExpNGramEmbedding` /
  `Qwen4ExpPinnedHostEmbedding` / `qwen4_exp_ple_table.py`，把我们的存储接到**真实缝**上。
  这是**读代码 + 改接口**，不是 GPU 工作。
- **新增 P2.5：把成本模型与 `τ(L)` 交给上游。** 这是唯一上游没有、且对 #36567 直接有用的东西。
- **差异化叙事必须改**：不能说「我们能在 graph 断点里藏一个磁盘读」，
  要说「**表比宿主内存大时，我们不掉 17%**」，并拿 #54070 的 −8%/−17% 当基准。

### 37.6 ⚠️ 纪律失败留档（第九条纪律的来源）

我此前**两次**断言「SGLang 没有 Engram/PLE」：对 0.5.19 为真，**对 `main` 为假**。
根因是**从已安装的 release + 一条关于 branch 的 bug 报告去推理上游主线**，而不是读 `main`。

这与本 session 早些时候「把 Python reader 的开销记到存储头上」是**同一类错误：
验证了替身，没验证目标。**

⇒ **新增第九条纪律**：关于上游能力的断言，必须落到 **main / 目标 branch 的具体 path**，
并注明 commit 或 tree 来源。**「我装的版本里没有」不等于「上游没有」。**

### 37.7 P0.5 的第一手核实（Session 44，对着上游真代码）

克隆了 `main`：SGLang `14b647c`、vLLM `fa1b3b1`。**`vllm/v1/ple_offload/worker.py` 不存在
⇒ #53899 / #54070 未合入 main**（符合预期，它们是 open PR）。

#### 核实一：上游的 file 后端是 **HMM 硬门控**

`python/sglang/srt/models/qwen4_exp_ple_table.py:321 check_file_backend_supported()`：
设备不报告 `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 时**直接 raise**，
文案写明「unified-memory parts such as GB10」、「use `--ple-offload-backend pinned`」。
逃生口 `SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1` 的告警原文是
**「only if you know the device reads pageable host memory through the host page tables」**。

上游自己的动机（该文件 docstring，:10-30）：

> Meant for unified-memory parts (GB10 / DGX Spark and similar), where pinned host memory
> comes out of the *same* pool as the model weights and `pinned` therefore frees nothing:
> Qwen3.8-Flash-Next is **126.0 GiB of weights on a 121.63 GiB box** and does not boot with `pinned`.

⇒ **它不是为「表比宿主内存大」写的，是为「权重和表抢同一个物理池」写的。**

#### 核实二：预取在**恰好最需要提前量的地方**被关掉

`PleFilePrefetcher.enqueue`（:109-127）只有两个提前返回，就是全部逻辑：

```python
if flat_ids.numel() < self._min_rows:                      # PLE_FILE_PREFETCH_MIN_ROWS = 2048
    return False
if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
    return False
```

- **decode 每 token 是 16 行**（§36.9 已从 vLLM 的 `compute_ngram_ids` 结构确证）
  ⇒ 除非 batch ≥ 128，**预取根本不发**。
- **capture 期间显式关闭** ⇒ 图模式下这条 hint 永远不发。
- 调用点在 `qwen4_exp.py:881-882`，即 gather 路径内。

#### 我们的楔子（现在有上游代码作依据）

| | 上游 `file` 后端 | 我们 |
|---|---|---|
| 硬件要求 | **必须是 HMM 设备**（GB10/DGX-Spark 级）；4090/H100/A100 **直接拒绝运行** | 无要求（显式 staged read） |
| decode 预取 | **< 2048 行不发**（batch<128 全灭） | 按 `τ(L)` 提前发 |
| capture 期间 | **显式关闭** | 断点机制下正是它工作的时候 |
| 提前量 | 按需 page fault（**访问时才发生，提前量为 0**） | **实测要求 ≥ ~325 µs**（§36.10） |

**⇒ 结论：在非 HMM 硬件上，上游没有任何磁盘路径；在 HMM 硬件上，它在 decode+capture
这条最需要提前量的路径上把预取关掉了。这两点合起来是一个具体、可辩护的缝隙。**

#### ⚠️ 边界（这条是推理，不是测量）

「按需 page fault 的提前量为 0」是从上游代码结构推出的（fault 发生在访问点），
**我们没有 HMM 设备可以实测**（4090 无 HMM）。若将来能上 GB10 类机器，
这是第一个该测的数：**一次 decode gather 的 fault 驱动的停顿到底有多长。**
在该数出来之前，不要把上表最后一行当作已证结论。

---

## 38. 第三十轮系统性思考（Session 44：目标、债、借鉴、计划）

### 38.1 终极目标再锚定：从「让它成为可能」收紧到「没有一层装得下时仍然可用」

§1 的北极星（让确定性哈希 n-gram 表成为任何小模型/框架都能廉价使用的磁盘优先存储基础设施）
**方向不变**，但 §37 把「廉价可用」的**边界**改了 —— 因为上游已经把「可能」做掉了：

| | 上游解决 | 我们要解决 |
|---|---|---|
| 层次 | **别占显存**（表放宿主 RAM 或页缓存） | **宿主内存也不够，且硬件不支持内核直接解引用** |
| 硬件前提 | HMM/UVA（GB10 类）或 pin 满 RAM | **无 HMM 的普通 GPU**（4090 / H100 / A100） |
| 提前量 | 按需 page fault（访问时才发生） | **必须在消费前 ≥ τ 发起** |

> ### 收紧后的北极星
>
> **让确定性哈希 n-gram 表在「没有任何一层内存装得下它、且硬件不支持内核直接解引用」
> 的机器上仍然可服务** —— 上游解决的是「别占显存」，我们要解决的是「宿主内存也不够，
> 而 HMM 不在」。

**为什么这仍然可证伪**：一张 51.2 GB 的表 + 一个低于它的内存上限 + 一块没有 HMM 的 4090，
decode 时读**不得落在关键路径上**。这个命题可以被测错。

**内核判据（§36.1 的延续，加一条）**：

```
旧: t_read ≤ τ(L_ple)
新: t_read ≤ τ(L_ple)  且  发起时刻 ≤ 消费时刻 − τ(L_ple)
```

第二条是 §37.7 逼出来的：上游 `PleFilePrefetcher` 在 decode（<2048 行）与 capture 期间
**直接不预取** ⇒ **提前量恒为 0**。所以「读得够快」不再是充分条件，
**「读发得够早」是新增的必要条件。**

### 38.2 本轮技术债（V183–V194）

| # | 债 | 证据 | 处置 |
|---|---|---|---|
| **V183** | **差值归因**：用两个噪声总量相减来归因 | §36.5 初稿「磁盘0.68+D2H0.39+rowid0.24」；`probes/sc43.md` §6 | 工具已建（`sc4_phase_microbench.py`）。**但已发布的差值结论只是加了横幅，没有撤** —— 需逐个复查引用点 |
| **V184** | **探针与库分叉**：探针手写 Python reader 冒充存储代价 | 同批冷行 554.6 vs 原生 `Store.fetch` 276.7 µs | 已换。**但根因是探针不复用库**，需一条纪律（见 §38.5） |
| **V185** | `τ(L)` 的分母用整步平均（94.6 µs） | 实测 τ(1) ≈ 325 µs，低估 3 倍以上 | **仅替身模型上已修正**；目标模型上的边际单层时间仍未测（斜率法） |
| **V186** | **上游能力断言未落到 main** | 两次断言「SGLang 没有 Engram/PLE」，对 main 为假 | 已立第九条纪律。**已写进文档的错误断言需逐个复查**（§37 新增，风险在于旧章节） |
| **V187** | **SC4 apparatus 建在替身缝上** | §37.2②：主线缝按构造无断点 | 沉没成本。**剩余价值 = #36567 缺的那块验证**，不是主线集成 |
| **V188** | **无 HMM 设备 ⇒ 上游最关键路径无法验证** | §37.7 边界：按需 fault 的停顿多长只能推理 | 认下。列为「需要外部机器」而不是「待硬件」空条款 |
| **V189** | **真实 PLE 模型单卡装不下** | 185.5 GB / 48 层 MoE vs 24 GiB | 长期不可达。**所有「替身模型上的」结论必须带标签** |
| **V190** | 活账曾是只可能变负的指标 | 217 未关 / 63 已关 | 已修（§0 + `ledger_rate.py` 棘轮） |
| **V191** | `sc4_reader_compare.py` 的 `StorePool` harness bug | 每次调用新建 pool，测出 2454 µs 的假值 | 未修。**假值已在报告里标注，但脚本仍会产出它** |
| **V192** | `real_perf_gate.py` 的 500 µs 阈值已过期 | §36.10：判据已换成 `L*` | **刻意未动**（不动 CI 门禁是单独决策）。但要防止有人把它读成达标 |
| **V193** | SGLang 0.5.19 上的全部工作基于过期 release | 该版本零 engram/零 PLE | 认下。**P0.5 起一律对 main** |
| **V194** | `research/` 只有报告，无复现脚本 | API payload 已 gitignore（可重取） | 低优先。报告里每条带 URL，可手工重取 |

**这 12 条里，只有 V183/V190/V191 是纯工程；其余 9 条是同一件事的不同侧面：
我们在替身上做测量，然后把结论用在目标上。** ⇒ §38.5 第十条。

### 38.3 借鉴矩阵（本轮增量）—— 关键是**分层**，冲突只发生在同层

上游现在有 5 套相关实现（SGLang `main` file 后端、SGLang #36567 io_uring、
vLLM #54371 UVA、#54070 disk、#54129 mmap），还有 KV-cache offload 生态
（Mooncake / LMCache / NIXL）。**它们的机制不冲突，冲突只发生在「同一层选两个」时。**
所以先分层，再逐层指定借用对象：

| 层 | 借谁 | 借什么 | 为什么与别层不冲突 |
|---|---|---|---|
| **L0 硬件/OS 访问** | SGLang `check_file_backend_supported` | **能力探测 + fail-fast**（宁可 load 时报错，也不要内核里静默读脏数据） | L0 只决定「内核能否直接解引用」。我们要**两支都支持**：HMM 走直读，非 HMM 走 L3 |
| **L1 文件格式与布局** | **无人可借** | `Store-P` / `SlotIndex` / view 物化 | 上游只是把原始表 mmap 掉。**唯一独占层** |
| **L2 驻留与预取策略** | SGLang `PleFilePrefetcher.pages_for_rows` + `PleFileRssTrimmer` | ① 从 rowid 算**去重后的 4 KiB 页集**（而不是按字节区段）② `MADV_DONTNEED` 1 GiB 分块 + 显式预算 | 与他们**参数不同、机制相同**：我们要在 decode+capture 上也发。他们的两个提前返回是我们要绕过的，不是要抄的 |
| **L3 发起时机** | **无人可借** | `τ(L)` 与「发起时刻 ≤ 消费时刻 − τ」 | **唯一独占层之二**。§37.7 证明上游在这里是空白 |
| **L4 I/O 机制** | SGLang #36567 | Rust io_uring：`opcode::Read(...).offset()`、队列深度 512、`O_DIRECT`、4096 B 页、`submit_and_wait` 批提交 | 与 L2 互补：**L2 决定发什么，L4 决定怎么发得快**。我们的 Python 池在 3.3× 饱和（§36.10），这正是缺的机制 |
| **L5 首次写盘/二次启动** | vLLM #54070 | write-through 首次构建 + `.done.json` sidecar + COW 秒映射 + **跳过 checkpoint 分片读** | 与 L1 正交（**布局 vs 生命周期**）。这一条直接解决「51.2 GB 首次落盘」的启动代价 |
| **L6 引擎缝** | vLLM `PleOffloadLayer` 契约（`_gpu_output_buffer` + `_sem` + `DONE_VALUE`）；SGLang `Qwen4ExpPinnedHostEmbedding.gather` | **用他们的契约，不自创** | 与 L3 的冲突只有一种情形：**host-issued 读需要断点，device-side 不需要**。由 L0 探测结果选，不由偏好选 |
| **L7 多级 tier 抽象** | Mooncake / LMCache / NIXL | 传输引擎与 tier 元数据的抽象 | 与 L2 正交（**策略 vs 传输**）。V4.1 阶段的借鉴对象 |

> ### 不冲突的硬规则
>
> **每一层的选择必须对其他层透明。** 具体三条：
> ① **L0 的两支（HMM / 非 HMM）必须共用 L1–L3** —— 不允许出现两套布局或两套判据；
> ② **L3 必须同时容纳「无需断点的 device-side」与「需要断点的 host-issued」两种实现**，
>    由 L0 探测决定，不由偏好决定；
> ③ **L4/L5 是可选加速项，不能成为 L1–L3 的前提** —— 少了它们系统仍然正确，只是慢。
>
> 违反任一条，就会重演本轮的失败模式：**在两个不同的层上各做一次选择，
> 然后误以为它们是同一个决定。**

### 38.4 开发计划：四条 Track，每条都必须落到**目标**上

| Track | 内容 | 需要一个数 | 要 GPU? |
|---|---|---|---|
| **A（主）L0 非 HMM 分支落地** | 在 **SGLang `main`** 上子类化 `Qwen4ExpPinnedHostEmbedding`（`qwen4_exp.py:768`）覆写 `gather`（:861）；非 HMM 时走 host-issued + 断点，HMM 时走 device 直读。复用其 `start_prefetch`(:1145) / `_graph_prefetch_buffers`（已 capture-aware） | **在 main 上（不是我们的探针）跑出一个冷态数字** | 是 |
| **B（最便宜）上游对话** | 一条 issue：**你们的 `PleFilePrefetcher` 在 decode（<2048 行）与 capture 期间不预取**，附我们的 `τ(L) ≥ ~325 µs` 实测与可复现脚本 | 无（产出是对方的回应） | **否** |
| **C（按需）L4 I/O 机制** | 借 #36567 的 io_uring reader，把 101 µs 推向设备地板 | `read_us` 的下降额 | 是 |
| **D（冻结）** | 产品广度（§36.5 P3）；**替身缝上的 stage 2** | — | — |

**排序理由**：B **不需要 GPU，且杠杆最高**（它把我们唯一的独占结论送到能改上游的人手里）；
A 是主线的唯一实质进展；C **只在 A 显示出暴露之后才做** —— 否则就是又一次「优化一个还没证明存在的瓶颈」，
正是 §36.5 P1.5 被冻结的原因。

**每条 Track 的准入门槛（第十条纪律的机械化）**：产出必须带**目标标签**
（`target=sglang-main@14b647c` / `target=qwen3.5-0.8b` / …）。
**没有标签的数不算数。**

### 38.5 第十条纪律：替身与目标必须显式分离

本轮两个错误是**同一件事**：

| 错误 | 替身 | 目标 |
|---|---|---|
| 把 Python reader 的 445 µs 当成存储代价 | 手写 Python reader | `engramdb.Store.fetch` / 真引擎 |
| 断言「SGLang 没有 Engram/PLE」 | 已安装的 0.5.19 | `main` |

**⇒ 第十条：任何数字或能力断言都必须带「目标标签」，且关于目标的断言只能用目标本身支撑。**

推论三条：
1. **替身可以用于探索，不可用于结论。** 替身上得出的数，要引用必须先换到目标上复现。
2. **「我装的版本里没有」不等于「上游没有」**（第九条）；**「我的探针这样做」不等于「库这样做」**（本条）。
3. **冻结一个 Track 时要说清「它优化的是哪个目标」** —— P1.5 之所以该冻结，
   不是因为它做得差，是因为它的目标是替身缝。

### 38.6 一句话

> 上游已经把「别占显存」做完了，所以我们的目标必须收紧成
> **「没有任何一层装得下、且硬件不支持内核直接解引用时，仍然可服务」**。
> 本轮把两个唯一独占的层看清了：**布局（L1）**与**发起时机（L3）** ——
> 其余每一层都能从上游借到，只要**分层借、不跨层混同**。
> 而本轮的两次翻车也收敛成同一条纪律：
> **我们在替身上做测量，然后把结论用在目标上。**
