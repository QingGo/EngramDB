# 交接 Handoff Prompt（空白上下文 Agent 开始工作前粘贴本文件全文）

> 你接手的是 **EngramDB** —— 一个由我（zengyingqing）持续构建的磁盘优先存储引擎项目。
> 以下信息令你拥有与前任会话等价的最新状态；开始工作前请先阅读仓库根目录与 `docs/`（本文件
> 为唯一权威快速入口，细节见 `docs/design.md`、`docs/roadmap.md`、`docs/session-log.md`、
> `docs/portable-dev.md`、`docs/linux-setup.md`、`probes/p4_view_notes.md`）。

## 0. 不要做的事（先看）
- **不要**改写已知结论/参数不提问：槽位 2560B、io_uring 性能面已定案（见 §5/§9），
  除非有新的可复现数据挑战它。
- **不要**把数据/表/视图文件加进 git（git 只有代码/文档/probes；`data/*` 是符号链接）。
- **不要**在未读取 `docs/session-log.md` 后复述"还没做过某实验"——本文件有全部状态。

## 1. 项目是什么
EngramDB = DeepSeek Engram / Qwen PLE（N-gram 嵌入记忆表）的**磁盘优先存储引擎**（Rust，
DuckDB 风格嵌入式）。两大负载：
- **A 训练预处理**（高吞吐批量读）：确定性哈希寻址 → badge 布局 + 页对齐批 gather
- **B 推理点查**（低延迟单记录读）：Store-P 物化视图（每 gram 16 头 2560B 紧凑槽，一次定长读）
北极星（2026-08 起三个决策）：**"用户电脑上跑大模型"也是产品面 → Windows 原生正式成为目标平台**；
生产介质 = 本地 NVMe（外盘/USB 仅研究/拷贝）；单机无 GPU（macOS 开发机 / Linux 生产）。

## 2. 仓库结构（~code/EngramDB）
```
crates/
  engramdb-core/      布局/Layout/ 计数索引/ fnv64 / ShardedStore(仅unix)
  engramdb-keygen/    PleSpec::real / rowids_for_seq（16 头 gram→rowid 映射）
  engramdb-io/        IoBackend trait（read_at/read_many）、PreadvBackend(跨平台 util
                      unix=pread / win=seek_read)、UringBackend + UringBatchBackend(Linux)、
                      BadgeGather(gather_pp 8t)、PrefetchPlan/StreamingPlanner、tiers、
                      view.rs（ViewBuilder/bench/lat 原语，CLI/P4 共用）
  engramdb/           CLI: build|index|gather|verify|warm|bench-real|view|prep
  engramdb-bench/     探针 bin：p4view(wrapper→io::view)、p2rowid、p3sim（仅unix）
scripts/  prep_env.py / corpus_build.py / mock_table_gen.py / gate.sh / release.sh / bump.sh
docs/     design.md（架构/实测基线§7）roadmap.md（战略/债/计划）session-log.md
          portable-dev.md（双机+外盘）linux-setup.md（新机器零到一）licenses.md
probes/   p4_view_notes.md（P4 v2-v9 全部结论）baseline_view.csv baseline_latency.csv
          view-keys-20k.txt（固定 20K keys，gate 用）agent_workload_stats.json
```

## 3. 当前完成度（截至最新 commit）
- **P2 数据面全部闭环**：三域语料（fineweb/zh/agent）稀疏统计、真实分布修正
  （大语料热集失效 → I2 索引仅作缓存优先级+agent 负载）；agent 负载 top100 覆盖 99.2%。
- **P4 存储面定案**（见 probes/p4_view_notes.md v2-v9）：
  - **2560B 紧凑槽**（4.50M vs 0.97M 行/s；放大 1.00 vs 1.60）
  - 全表视图构建（51.2G @ 22min，流式分块 RSS 395MB）可用命令重建
  - **性能主权表**：A 路径 ~1.09M 行/s（冷，平台无关）；B 视图 8t warm 4.5M~26M；
    全表冷随机 8t：NVMe 19.2M / USB 外盘 554K（**35× 介质拖累**）；真冷/热差仅 1.85×（SSD）
  - 延迟：视图单记录 p50=0.75~5μs、p99=1.4~12μs（存储层比 10ms/token 低 3 个量级）
  - **io_uring 性能面定案**：per-call 0.97×、batch 0.94× vs preadv —— 保留作语义实现，
    默认 = preadv；不要在性能面再花时间（除非介质=网络盘/cgroup 受限环境）
- **发布链**：✅ v0.2.1 已发布 crates.io 四 crate + PyPI `engramdb-python 0.2.1`
  （5 平台 wheel + sdist），包含
  `PageReader` / `PleDiskGather` / Linux `IoUringPageReader` 和多平台 PyPI wheel 矩阵；
  ✅ v0.2.2 已发布，额外包含 `engramdb.sglang` / `engramdb.vllm_plugin` 适配原型；
  ✅ v0.2.3 已发布，修复 release-assets 重复上传问题，GitHub Release 含 4 平台二进制 + Python 包；
  ✅ v0.2.4 已发布，包含“不改源码”的类级 PLE patch hook（`install_vllm_ple` / `install_sglang_ple`）；
  版本只走 scripts/bump.sh（现已同时更新依赖版本引用和 Python `__version__`）。
- **跨平台**：cargo check --target x86_64-pc-windows-msvc = 0 错误；Windows 原生=目标平台；
  WSL2 全链路验证过（x86_64 + aarch64 树莓派 17 tests 全绿）；
  **真实 Linux 实机验证已闭环**：树莓派 aarch64 + WSL2 Ubuntu x86_64 均安装 v0.2.4 wheel 并跑通完整 smoke（Session 8）。
- **Python 桥**：PyO3 原生扩展 `engramdb-pyo3` 已发布并优先使用，ctypes C-ABI 作回退；
  `DiskMultiHeadEmbedding` 和真实 `EngramLayer` 前向均通过；
  **TinyLlama + engram-peft + EngramDB 磁盘版完整文本生成已跑通**（Python 3.12 + torch 2.2.2）。
  已新增 `engramdb.PageReader`（SGLang 兼容）、`engramdb.vllm.PleDiskGather`（vLLM 方向）、
  Linux `IoUringPageReader`（io_uring batch）、`engramdb.sglang`、`engramdb.vllm_plugin`；
  支持 `install_vllm_ple` / `install_sglang_ple` 类级 patch，用户可不改引擎源码。
  多平台 PyPI wheel：Linux x86_64/aarch64、macOS x86_64/arm64、Windows x86_64；CI 含 Python 安装冒烟。
  README 已重写为完整用户入口（用法/架构/性能/优化策略）。
  **真实引擎模型类验证已闭环（Session 9）**：vLLM 0.28.0 与 SGLang 0.5.9 的真实
  `Qwen3ForCausalLM` 均通过类级/实例级 patch，`DiskPleEmbedding` 前向成功。
  **访问序视图已在 WSL 验证（Session 10/11）**：`view build --keys`、校验与冷盘 A/B 均跑通；冷顺序 785.8MB/s vs 冷随机 86.0MB/s（约 9.1×）。
  **多表/Arrow/最小服务原型已落地（Session 14）**：`Database`、`arrow_utils`、TCP/JSON server 均通过 smoke。
  **vLLM embedding A/B 已测（Session 12/13）**：raw disk 235-268μs/call；已实现行级 LRU 后降到 14-23μs/call。`DiskPleEmbedding` 首未命中仍走 raw disk，Tier/预热待做（V7）。

## 4. 机器与资产（重要）
| 机器 | 地址 | 用途 |
|---|---|---|
| 主开发机（Mac notebook，本章运行时所在） | 本机 | 主开发；外盘长期接 |
| 家庭机（Mac Intel） | zeng@100.73.212.21（免密） | 第二工作机，可带出门关机；**外盘可插** |
| 树莓派（aarch64, SD 卡） | zeng@192.168.31.110 | 仅功能/门禁验证（SD 不测性能），17 测试绿 |
| Windows（含 WSL2 / GTX1070） | minam@192.168.31.108 | Linux 语义测试 + GPU（P4b 用） |

外盘 `/Volumes/My Passport` 唯一物理数据地（剩余 ~155G）：
- `qwen38-rows`（真表 128 shard×2,500,012×160B = 48G）
- `p4view-full-2560.bin` + `.manifest.json`（51.2G 全表视图）
- `engramdb-data/`（corpus-build 6.0G / mock / p2-work）
- `qwen38-ple`（权重分片 53G）
- `data/*` 一律符号链接（换机/换挂载名用 `ln -sfn` 修复，见 portable-dev.md）

## 5. 环境
- 本机：macOS / Rust 1.98（cargo）。目标机：macOS 15.3.1 Intel / cargo 1.95（允许 rustup
  update）、python3 3.9.6、numpy 2.0.2（--user）
- mac 机 crates 目录默认走 TUNA? —— 本机无 config 亦可；**发布时用 scripts/release.sh**
  （自动绕开 TUNA 与 429；crates.io API 需 User-Agent）
- 长任务纪律：任何 >2min 的命令（远程跑、编译、scp、bench）都**后台+nohup 或投递 schtasks
  （仅 Windows 侧的 WSL；Windows→WSL 长任务必须 schtasks，ssh 会话子进程会被杀）**，
  并写日志文件轮询；不要同步长跑占死交互。

## 6. 直接可用的快速验证（10 秒起步）
```bash
cd ~/code/EngramDB && cargo test --workspace          # 17 tests 全绿
cargo build --release -p engramdb -p engramdb-bench --bin p4view
target/release/engramdb prep --dist agent --reqs 4 --cap-token 200 /tmp/keys.txt
target/release/engramdb view build data/real-rows 2000 /tmp/v.bin /tmp/k.txt --slot 2560
target/release/engramdb view bench data/real-rows /tmp/v.bin --keys /tmp/k.txt --sub 2000
target/release/engramdb view lat /tmp/v.bin --warm
bash scripts/gate.sh                                   # fmt+clippy -D warnings+test
```
探针复现/重建命令清单在 `probes/p4_view_notes.md` 顶部。

## 7. 待做（按优先级，Q3/Q4 已交）
1. **P4b GPU 首点**（唯一外部依赖决策）——WSL 与 Windows 的 CUDA 边界：nvcc=13.2 但
   driver=13.0（运行时须 ≥ 编译版本）。两条路：**升 NVIDIA 驱动 ≥13.2**（推荐）或
   **llama.cpp b10688 win-cuda-12.4 预编 zip**（`llama-b10688-bin-win-cuda-12.4-x64.zip`，
   下载/解压踩过坑：schtasks 投递 + 解压路径需探测）。拿到 Qwen3-0.6B-Q8_0（已下载
   639MB 在 WSL `~/qwen3-0.6b-q8.gguf` 与 Windows `C:\Users\minam\engramdb-transfer\`）
   的 **GPU tok/s 首点**（CPU 预填充基线已获 = 31.79 t/s；tg 段在 WSL+CPU 有 D-state 卡死谜题，GPU 为正解）。
2. **P3 主线延伸**（可选顺序）：视图"顺序化排布 + 访问序调度"（P4 v5；现只有测量脚
   `--order seq|rand`，真实验是"按访问序重排槽位"）；然后 P4b 端到端 decode 曲线
   （50/100 tok/s 对标目标）+ P5（训练侧 DataLoader/PyO3 绑定——engram-peft 联动零代码）。
3. **文档/复盘**：本 session（2026-08-30 深夜段：P4 v2-v9、跨平台 W0、prep、便携迁移、
   batchcmp 结论）尚未追加进 session-log Session 3/4——接手续时可以补。
4. 次要：树莓派性能采样**放弃**（tmpfs 飘忽 + SD 无代表性，门禁已达）；Windows 原生
   "VHDX vs NTFS 同盘对照"小实验未跑（W1 未尽，可选）。

## 8. 纪律（同前任）
- 每件**完成的活** = code + gate（fmt/clippy/test）+ 文档（notes/session-log）+ commit/push
- commit message 风格：`<type>(<scope>): <摘要>`；英文摘要，要点数行式
- **版本只 bump.sh**；不改 manifest 手打
- 复现/数据口径注介质、注冷热、注并发（吞吐+延迟分布，p50/p95/p99/max）
- 有疑问先读 `docs/session-log.md`（坑已记录：mac 假冷、掉盘、/tmp 清理、zsh `===`、
  目录前缀 tar 等）

## 9. 技术债速览（详细 roadmap §6-7）
T1 视图机制已入库（解决）；T2 真冷（已闭环 Linux fadvise/0 上 SSD 1.85×）；T3 资产可重建
（notes 顶部命令）；T4 全表 A/B 大样本未跑（抽样口径已闭环）；T5 max 罕见簇未归类（事件，不验收）；
T6 多表/table_id 缺位（目录粒度即可）；T7 探针统一（p4view 已薄壳化，p2rowid/p3sim 仍在 bench）。
N4 crates.io OIDC / N6 PyPI 相似名 留 0.2 窗口。

---
使用方式：将以上全文作为第一个 prompt 粘贴给新 agent（大模型上下文为空时）；工作途中
建议同时提供 `docs/session-log.md` 尾部与本文件，防止其基于猜测回溯。
