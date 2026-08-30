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
- [ ] 新增 `scripts/release_gate.sh`：
  - `cargo fmt --all --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --workspace`
  - `python_wheel_smoke.py`
  - `service_smoke.py`
  - `c_abi_smoke.py`
  - `decode_baseline_check.py`
- [ ] 将最新 README/python README 收编进下一个版本。
- [ ] `install_real_qwen_ple_embedding` 去掉静默硬编码 fallback，或至少输出显式 warning。
- [ ] `rowids_for_seq()` 支持 `multipliers`/`info` 来源。
- [ ] 把 README 核心示例抽成可执行 smoke，防止再次漂移。

**退出标准**：
- 本地一条命令能完整预检所有发布门禁。
- 下一次 bump 前 README 与代码同一点提交。

#### Phase A：兄弟项目“配置即用”
- [ ] engram-peft：支持 `table_source="engramdb:store"` 自动调用 EngramDB 注入。
- [ ] qwen35-ple：真实 e2e 改用 `install_real_qwen_ple_embedding(store, model_dir=...)`。
- [ ] 跨仓契约 smoke 纳入兄弟项目 CI。

**退出标准**：
- 用户只需配置 `table_source=engramdb:store`，不需要手动 import 注入函数。
- qwen35-ple 真实 FP8 PLE 全链路跑通。

#### Phase B：真实模型 E2E
- [ ] 写 custom loader / from_pretrained hook，跳过 `ngram_embedding.shard_*` 大权重。
- [ ] 在官方 `Qwen4ExpForCausalLM` 或等价类中替换真实 PLE。
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

