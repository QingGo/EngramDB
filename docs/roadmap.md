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
| V4 | 端到端性能契约仍未闭环 | 存储面已达标，应用面缺 | WSL+GPU/CPU 实机 PLE decode 曲线 |
| V5 | 发布工程仍偏人工 | 已修 release-assets，但需要更完整的 preflight/回滚 | 后续接入自动 release 检查 + release notes 资产完整性断言 |
| V6 | 顺序化视图/访问序调度 | ✅ 核心已验证：冷顺序 785.8MB/s vs 冷随机 86.0MB/s ≈ 9.1×（Session 11） | 剩余为大表复测、冷多线程策略与调度器落地 |

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
5. **存储产品化**：多表、manifest 完整性、服务化/Arrow IPC、CLI 收敛。
6. **发布自动化加固**：release-assets 资产断言、全平台 wheel 自动验证、版本/文档同步检查。

