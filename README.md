# EngramDB

> **消歧声明**：GitHub 上另有多个同名 "EngramDB" 项目，均为通用 Agent 记忆/语义检索类产品，
> 与"DeepSeek Engram"无关。已完成清点（2026-08-29）：
> - Agent 记忆类：`nkkko/nibzard`（MemoryNode 向量图时间库）、`ioteverythin`（SQL-native event-sourced 记忆，PyPI badge 未生效）、`andrewmooney`（MCP 服务器）、`egeapak`（coding agent 项目级备忘，磁盘文件）——语义相似搜索/备忘，非确定性查表
> - 检索类：`sxtj`（纯 Rust 嵌入向量库，HNSW+BM25+TUI）、`pyalwin`（法律 multi-hop 混合检索，DuckDB）
> - 其他类：`SohenDev/engramdb-visualization`（Grafana fork 改名）、`sn0wfree`（分析型单文件数据库 .hdb）
> 本项目与它们无关——**EngramDB = DeepSeek Engram / Qwen PLE（N-gram 嵌入记忆表）的磁盘优先存储引擎**：
> *为确定性哈希寻址的 n-gram 嵌入表提供数据库级存储、索引与负载优化*。
> 登记现状：crates.io / PyPI `engramdb` 均未被占用（审计当日）——应尽早发布占名。

磁盘优先的 Engram / PLE（n-gram 记忆表）存储引擎（Rust，DuckDB 风格：嵌入式 + 可选服务化）。

- 面向：DeepSeek **Engram**（arXiv 2601.07372）与 **Qwen3.8-Flash-Next PLE**（51.2B 参数 n-gram 表）的静态记忆表。
- 能力：确定性哈希寻址（rowid 预知）→ badge 布局 / 批量 gather / 确定性预取 / 三级缓存 / 频率索引；
  双视图：Store-I（原始表）与 Store-P（物化 e_t）；训练流（高吞吐）与推理点查（低延迟）。
- 形态：`engramdb` CLI / Rust lib / Python 绑定（PyO3，包名 `engramdb`）/ 服务化（Arrow IPC，M4）。

文档：
- `docs/linux-setup.md` —— **全新 Linux/WSL 环境零到一**（rustup/镜像/MTU/mock 验证/真表接入/探针复现/口径）
- `docs/design.md` —— 技术设计与开发方案（架构/负载/实测基线§7/里程碑/风险/ADR）
- `docs/engram-specs.md` —— Engram/PLE 结构规格与证据链
- `docs/roadmap.md` —— 战略路线图：终极目标 / 技术债 / 借鉴矩阵 / Phase 计划 / 稳定性机制
- `docs/session-log.md` —— 首个开发 session 完整复盘（尝试-坑-完成-新问题-计划）
- `docs/licenses.md` —— 许可与合规边界（模型权重/语料统计分发规则）
- `scripts/gate.sh` —— 本地门禁（fmt + clippy -D warnings + test）

权重数据（不入库）：真实 FP8 checkpoint 分片软链到 `data/qwen38-ple-fp8`
（外接 SSD：`/Volumes/My Passport/qwen38-ple`，约 53GB，ModelScope `Qwen/Qwen3.8-Flash-Next-FP8` 的 PLE 相关分片）。
开发多用 `scripts/mock_table_gen.py` 合成的结构等价表。

快速开始（开发期, 以 M0 探针为主）：
```bash
python3 scripts/prep_env.py quick       # 任意环境（Win/Linux/macOS）1 分钟就绪：自检 + 合成 PLE 表
python3 scripts/prep_env.py verify      # 校验产物
cargo run -p engramdb-bench               # 探针（占位，M0）
python3 scripts/extract_ple_spec.py       # 从真实分片提取规格 JSON（需真权重）
python3 scripts/mock_table_gen.py         # 生成结构等价合成表（prep_env quick 已含）
```
