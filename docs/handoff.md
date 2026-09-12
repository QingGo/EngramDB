# 交接 Handoff Prompt（空白上下文 Agent 开始工作前粘贴本文件全文）

> 你接手的是 **EngramDB** —— 一个由我（zengyingqing）持续构建的磁盘优先存储引擎项目。
> 以下信息令你拥有与前任会话等价的最新状态；开始工作前请先阅读仓库根目录与 `docs/`（本文件
> 为唯一权威快速入口，细节见 `docs/design.md`、`docs/roadmap.md`、`docs/session-log.md`、
> `docs/session-summary.md`、`docs/portable-dev.md`、`docs/linux-setup.md`、
> `probes/p4_view_notes.md`）。

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
  ✅ v0.2.5 已发布并验证（包含 LRU、多表、Arrow helpers、JSON+二进制最小服务、扩展 wheel smoke）；
  ✅ v0.2.6 已发布（Rust serve/check/二进制协议、CPU A/B 脚本、cache_size=0 修复）；
  ✅ v0.2.7 已发布（真实 PLE Store/层 bit-exact、C ABI rowids、FP8 磁盘集成、兄弟项目契约）；
  ✅ v0.2.8 已发布（修复 v0.2.7 CI：rustfmt + 无 torch Python 导入；新增自动 weight_scale、Python rowids_for_seq、PyO3 native rowids、C ABI smoke 入 CI）；
    ✅ v0.2.9 与 v0.2.10 已发布（v0.2.10：`StorePool` / `ThreadLocalStore`、`Database` 池化读取、README/门禁更新）；v0.2.9 包含快速 `Store.fetch` / `fetch_e_t_tensor`、native rowid history、qwen35 live-store；
  版本只走 scripts/bump.sh（现已同时更新依赖版本引用和 Python `__version__`）。
- **跨平台**：cargo check --target x86_64-pc-windows-msvc = 0 错误；Windows 原生=目标平台；
  WSL2 全链路验证过（x86_64 + aarch64 树莓派 17 tests 全绿）；
  **真实 Linux 实机验证已闭环**：树莓派 aarch64 + WSL2 Ubuntu x86_64 均安装 v0.2.4 wheel 并跑通完整 smoke（Session 8）。
- **Python 桥**：PyO3 原生扩展 `engramdb-pyo3` 是**唯一**后端（ctypes 回退已于 Session 42 删除，见 roadmap §34）；C ABI 移至 `crates/engramdb-cabi`，定位为 C/C++ 嵌入面；
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
  **多表/Arrow/最小服务原型已落地（Session 14/15）**：`Database`、`arrow_utils`、TCP/JSON server（含 `fetch_arrow`）、二进制 length-prefix server + `EngramDBClient` 均通过 smoke。
  **vLLM embedding A/B 已测（Session 12/13）**：raw disk 235-268μs/call；已实现行级 LRU 后降到 14-23μs/call。`DiskPleEmbedding` 首未命中仍走 raw disk，Tier/预热待做（V7）。
  **Rust 侧多表/serve/check 已落地（Session 16/17）**：`engramdb tables <root>`、`engramdb serve <root> --port N [--binary]`、`engramdb check <root>`；支持 manifest 布局推断、完整性检查、JSON 与二进制 raw fetch。
  **CPU 小模型 E2E decode A/B 首曲线已获得（Session 16）**：`scripts/cpu_tiny_decode_ab.py`，memory vs disk raw vs disk LRU；原始磁盘约慢 16–24%，LRU 基本拉回。
  **真实 Qwen3.5-0.8B CPU E2E A/B 已跑通（Session 18）**：`scripts/qwen35_cpu_decode_ab.py`，模型软链于 `data/Qwen3.5-0.8B`；原始磁盘通常慢 20%+，LRU 在短序列下优势尚不明显。
    **可信 CPU decode 基线与真实权重 bit-exact 已闭环（Session 19）**：固定 seed/reps>=5/median+p90；`probes/cpu_tiny_baseline.csv`、`probes/qwen35_cpu_baseline.csv`、`scripts/decode_baseline_check.py` 已入库；`scripts/qwen35_bit_exact.py` 实机 `BIT_EXACT_PASS`；真实权重 store 下 Qwen3.5 raw 慢 13.0%、LRU 慢 19.2%。
  **真实 PLE 自动发现已定位（Session 19 增补）**：`python/engramdb/ple_discovery.py` + `scripts/inspect_ple_attributes.py`；真 Qwen3.8/Qwen4Exp 的 PLE 表路径为 `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_*.weight`，而 Qwen3.5-0.8B 没有 PLE。
  **真实 PLE 数据面与兄弟项目服务已闭环（Session 20-22）**：修复 `gather_pp` 多分片偏移；真实 128-shard Store bit-exact；`DiskPleNGramEmbedding` adapter；真实 PLE layer forward bit-exact；C ABI `rowids_for_seq` / `abi_version`；`install_real_qwen_ple_embedding`；`sibling_contract_smoke.py` 通过。完整模型 E2E 仍受环境限制。
  **v0.2.7 CI 修复 + Phase A EngramDB 侧补齐（Session 23）**：修复 rustfmt import 顺序与无 torch 环境导入失败；新增 `load_ple_weight_scale` 自动读取 checkpoint scale；新增 Python `rowids_for_seq()`；PyO3 native `rowids_for_seq`；C ABI smoke 进入 CI；本地 fmt/clippy/test/python smoke/C ABI smoke 全绿。
  **v0.2.8 发布 + README 刷新 + 系统性思考（Session 24）**：发布 v0.2.8；根 README 与 python README 补齐 Rust/Python 安装、真实 PLE、FP8 engram-peft 用法；roadmap 新增第 17 节系统性思考与技术债 V60-V73；后续优先做 release gate 和兄弟侧配置即用。
  **Session 25（当前开发，未发布）**：新增 `scripts/release_gate.sh` 并接入 `bump.sh` 默认前置；`discover_ple()` 自动读取 `layer_multipliers`（一次索引读取同时取 scale/multipliers）；`rowids_for_seq()` 支持 `info`/`multipliers`；`install_real_qwen_ple_embedding` 无来源时改为显式 warning；README/Python README 同步；`python_wheel_smoke.py` 增加 safetensors I64、discovery、自定义 multipliers 回归。
  **Phase A 兄弟侧推进（Session 25 后半）**：engram-peft 新增 `table_store_path` / `table_model_dir` / `table_shards` / `table_rows_per_shard` / `table_width` / `table_dtype` / `table_scale` / `table_cache_size` 配置，并在 `get_engram_model()` 中自动打开 Store 并注入 Disk MultiHeadEmbedding；已直接推 engram-peft master。qwen35-ple 新增 `to_engram_config()` YAML 桥接，`run_m0_smoke.py --e2e` 自动走真实 FP8 注入（`--ple-model-dir`），README 已补充配置即用示例；qwen35-ple main 已更新。跨仓契约 smoke 已加入 qwen35-ple CI（checkout EngramDB + engram-peft）。
  **Phase B 真实 PLE e2e（Session 25 末）**：新增 `engramdb.official_loader`（过滤 ngram shard + 磁盘 PLE 安装）和 `qwen4_ple_custom_loader.py` dry-run/完整加载入口；新增 `run_real_fp8_e2e.py` 轻量版真实 FP8 e2e，并已在本机跑通：Qwen3.5-0.8B + 真实 128-shard Store-I + 配置驱动自动注入，`REAL_FP8_E2E_OK`（约 9.6s，logits 有限、生成 shape 1x10）。官方 Qwen4Exp 模型类验证仍待大内存/新版 transformers 环境。
  **Session 26 系统性思考（第十四轮）**：复盘本轮“配置即用 + 真实 FP8 e2e + Phase B 初步”；新增 V74–V85；明确下一步最高优先级是官方 Qwen4Exp 加载前 patch ngram 占位、官方类 bit-exact、真实 memory/disk A/B，随后 Rust 热路径与服务化。完整版见 roadmap Section 18。




## 4. 机器与资产（重要）
| 机器 | 地址 | 用途 |
|---|---|---|
| 主开发机（Mac notebook，本章运行时所在） | 本机 | 主开发；外盘长期接 |
| 家庭机（Mac Intel） | zeng@100.73.212.21（免密） | 第二工作机，可带出门关机；**外盘可插** |
| 树莓派（aarch64, SD 卡） | zeng@192.168.31.110 | 仅功能/门禁验证（SD 不测性能），17 测试绿 |
| Windows（含 WSL2 / GTX1070） | **`minam@100.78.250.122`**（Tailscale；旧记的 `192.168.31.108` 已不可达） | Linux 语义测试 + GPU（P4b 用）。**⚠️ 不可用作"生产 NVMe"的性能代理**：WSL2 虚拟化存储栈的冷随机读比 Mac 的 USB 外盘还慢约 2×（见 roadmap §30.7）。进 WSL 需 `ssh → cmd.exe → wsl.exe -e bash -lc`，cmd 不认 `;` 与 `\"`，命令要 base64 转发 |
| **AutoDL 容器（原生 NVMe，Phase A 用机）** | **`ssh -p 28326 root@connect.nmb1.seetacloud.com`**（免密，已配好） | **生产介质口径的唯一有效机器**。2× Xeon Gold 6430 / 64C128T；宿主 1 TB RAM，**本容器 cgroup 120 GiB**；`/root/autodl-tmp` = XFS on **RAID1（2× Samsung PM9A3 7.68 TB NVMe，PCIe4 x4）**；RTX 4090 24 GB。见下方"使用要点" |

**AutoDL 使用要点（Session 42 实测，踩过的坑）**：

1. **`drop_caches` 被拒**（`CapEff` 无 `cap_sys_admin`，`echo 3 > /proc/sys/vm/drop_caches` → `Permission denied`）。
   容器里 cgroup 120 GiB ≫ 表体积，所以"表比内存大 ⇒ 冷读"**不成立**。
   **正确做法：按文件 `posix_fadvise(POSIX_FADV_DONTNEED)`**，且必须配冷热自校验（见 §31.2）。
2. **`io_uring_setup` 返回 `EPERM`** —— Docker 默认 seccomp 拦截（`Seccomp: 2`，`io_uring_disabled=0` 也没用）。
   任何走 io_uring 的测试/基准在这台机器上**不可用**，需 `--security-opt seccomp=...` 才可能放开。
3. **无网络**（github 超时、crates.io 403）。但 **`/etc/network_turbo` 提供学术代理**，
   且 tuna / ustc 镜像可用。装 Rust 的正确姿势：
   ```bash
   export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
   curl -sSfL $RUSTUP_DIST_SERVER/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o /root/rustup-init
   chmod +x /root/rustup-init && /root/rustup-init -y --profile minimal --default-toolchain stable --no-modify-path
   rustup component add rustfmt clippy
   # ~/.cargo/config.toml: source.crates-io replace-with = "tuna"（sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/）
   ```
4. **`python3` 不在 PATH**，但在 `/root/miniconda3/bin/python3`。`gcc`/`make`/`git` 已有。
5. **磁盘配额实际 86 GB**（`/root/autodl-tmp`，`df` 有时显示 70 G 可用），`/` 是 30 GB overlay。
   装 Rust + 源码 + 25 GB 数据够用，**128 shard 全表（51.2 GB）也放得下**，但要注意余量。
6. **已有他人/前任的项目数据**：`/root/autodl-tmp/qwen35-ple/`（49 GB，含
   `qwen38-rows` 真实 PLE 行 65 shard）。**不要删**；`qwen38-rows` 的 `manifest.json`
   声称 128 shard，实际只剩 **65**。
7. **宿主有其他租户**（load ≈ 11–13 / 128 核），但 4 KiB 随机读延迟不受影响
   （复跑偏差 ≤2.4%，见 §31.4）。

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

  **Session 27（第十五轮：Phase B1/B2 代码落地 + 异步预取方向）**：
  - Phase B1 已真正落地：`patch_official_ngram_embedding_for_disk_load()`、`load_official_checkpoint_without_ngram_shards()`、`qwen4_ple_custom_loader.py --load-model` 完整链路。
  - Phase B2 小表 bit-exact 通过：官方冻结快照 vs `DiskPleNGramEmbedding`，覆盖 batch/EOS/chunked streaming，max-abs=0。
  - `DiskPleNGramEmbedding` 支持自定义 prime table、batch 维度、每 batch context。
  - 新增 `qwen4_ple_official_loader_smoke.py`、`qwen4_ple_bit_exact_small.py`、`tests/test_phase_b_official_loader.py`。
  - 核心新债务：V86 PyO3 Store 持 GIL/unsendable、V87 无 prefetch API、V89 非真实行 bit-exact、V91 无真实 A/B、V92 Python 热路径。
  - 下一步：找 Qwen4Exp Transformers + 大内存环境，做完整模型验证和稀疏真实行 oracle；然后做异步预取 + Rust 热路径；再做真实 A/B 与服务化。
  - 系统思考全文见 `docs/roadmap.md` Section 19。

  **Session 27 低资源验证补充**：新增 `scripts/sparse_real_row_oracle.py`，用 9 个固定 token、144 个真实 PLE 行验证：
  - 原始 checkpoint FP8 行与 EngramDB Store-I byte-identical；
  - `DiskPleNGramEmbedding` 读真实 Store 与原始 checkpoint 行 dequant 后 maxdiff=0.0。
  - 结论：不需要完整大模型也能完成真实 PLE 行的位级验证；完整模型只留作内存/性能 gate。

  **性能关键路径进展**：PyO3 `Store.fetch` 已通过 `py.allow_threads` 释放 GIL，Store 去掉 `unsendable` 并验证并发 fetch；新增 `DiskPleEmbedding.prefetch()`、`DiskPleNGramEmbedding.prefetch()`、模型级 forward pre-hook 和 future/wait。下一步用真实 Store 做 hit-rate / prefetch_wait / tok/s A/B。

  **Session 28（第十六轮：性能关键路径）**：
  - 真实行低资源验证闭环：checkpoint ↔ Store-I byte-identical，DiskPle real-Store maxdiff=0.0。
  - PyO3 `Store.fetch` 释放 GIL，Store 支持并发；`DiskPle.prefetch()` + future/wait + 模型级 pre-hook 落地。
  - 真实 Store 微基准：模拟 30ms 计算窗口下 192ms → 34ms。
  - 新增技术债 V99–V111：真实模型预取 A/B、prefetch 生产化、Python 热路径 native、真实 memory vs disk A/B、serving。
  - 下一步：真实模型 sync vs prefetch A/B，然后 Rust 原生热路径，再做服务化。

  **Session 28 综合整理**：完整计划/发现/尝试/踩坑/完成/未完成/未来计划已写入 `docs/session-summary.md` 的 Session 28 综合整理章节；尝试与踩坑详细版见 `docs/session-log.md`。

   **Session 29（Prefetch 生产化起步 + Mini 官方模型 A/B）**：
   - `DiskPleEmbedding.close()` / `DiskPleNGramEmbedding.close()`：prefetch executor 生命周期管理。
   - `DiskPleNGramEmbedding.prefetch()` 返回底层 future，便于观测预取完成。
   - `install_disk_ple_prefetch_hook()` 兼容 `hook(module, args)` 与 `hook(module, args, kwargs)`。
   - 新增 `qwen35-ple/scripts/mini_official_prefetch_ab.py`：冻结官方 PLE layer + 真实 Store + dense 前后块的 mini A/B，输出 CSV。
   - 当前仍是低资源 smoke；完整模型、冷热分离、CSV 阈值、超时/回退/合并去重未完成。
    - 20k 预计算慢路径修复：`PleDiskGather.fetch` 改为直接 `Store.fetch`；新增 `fetch_e_t_tensor`；qwen35 `real_ple.fetch_e_t` / `precompute_real_ple_features.py` / `run_phase0 --live-store` 已切换。
    **Session 30 系统性思考**：详见 `docs/roadmap.md` Section 21（终极目标、V112–V117、Track 0–5、借鉴矩阵）。
    **Session 31/32（懒加载 live-store + WSL 实测）**：`run_phase0.py --live-store` 改为按窗口懒加载，1M token 不再因 10GB e_t OOM；确认磁盘优先是正确路径；新债 V118–V122 见 roadmap Section 22。
     **Session 32 综合整理**：完整计划/发现/尝试/踩坑/完成/未完成/未来计划见 `docs/session-summary.md` 的 Session 32 综合整理章节；详细尝试与踩坑见 `docs/session-log.md`。
  **Session 33（Track A 通用懒加载数据流 + B/C/D 起步）**：qwen35-ple 新增 `src/qwen35_ple/live_store.py`（`LiveETStore` / `LiveETView` / `LiveETViewStore` / `LiveETDataset` / `FetchStats` / `LiveETBatch`），支持 control、shuffle、DataLoader 多 worker（pickle 重开 Store）、每窗口 fetch 统计；`run_phase0.py` 已改为统一模块；新增 `scripts/run_live_et_dataset_smoke.py`、`scripts/bench_store_vs_view.py`、`scripts/bench_lazy_windows.py` 和 9 个测试；EngramDB 新增 `StorePool` / `ThreadLocalStore`。本机 Store-P 懒加载 1M 约 7.1s，Store-I 100k 约 60.5s；WSL p4view Store-P 20k/100k A/B 已跑（8t 约 22M rows/s）；WSL Python 懒加载也已跑（1M Store-P 约 23.9s）；继续推进 serving A/B 与完整模型训练。

  **Session 34（第二十轮系统性思考）**：终极目标不变；已确认“读取快”不等于“实验能跑”；新增技术债 V123–V132（语义 slot 映射、访问序、真实模型 1M、WSL golden、serving、门禁、Arrow、WSL 复现、全表 Store-P）；完整思考见 `docs/roadmap.md` Section 24、`docs/session-summary.md` Session 34。

  **P0 完成（V123/V124）**：通用 rowid→slot 语义索引（`qwen35_ple.slot_index.SlotIndex`、`--slot-index-out`、`run_phase0 --store-p-slot-index`）与自动访问序调度（`LiveETViewStore(access_order=True)`、`LiveETDataset(access_order=True)`、`run_phase0 --access-order`）已落地；真实模型 1M 实验仍由 qwen35-ple/WSL 侧继续。

  **Session 35（第二十一轮：v0.2.11 + 系统性思考）**：发布 v0.2.11；EngramDB Python 新增 `SlotIndex`；完成 P0 语义索引/访问序调度代码。新债 V133–V139（SlotIndex 扩展性/重复实现/CLI 原生索引/调度基准/numpy 依赖/窗口重排语义/跨仓契约），完整思考见 `docs/roadmap.md` Section 25、`docs/session-summary.md` Session 35。

  **Session 36（第二十二轮：Phase A + DiskSlotIndex + 全表工具）**：WSL 1M real/control/no-reader 3-seed 已核验并固化（real < control < no-reader，Go）；新增 `DiskSlotIndex`、`--keys-stream`、`build_full_store_p_batch.py`、合成 CI 门禁和 StorePool 遥测。新债 V140–V148（双路径复跑、全表索引实测、bucket 文件数、原生 CLI slot-index、fetch timing、golden、真表门禁、发布），完整思考见 `docs/roadmap.md` Section 26。

  **Session 37（第二十三轮：原生 SlotIndex CLI + serving 架构）**：EngramDB 新增原生 `slot-index build|verify`、`view build --slot-index` / `view verify --slot-index`；Python `DiskSlotIndex` 支持 v1/v2 并可直接生成 v2；新增 `scripts/bench_disk_slot_index.py`。完成 vLLM/SGLang/PleMemory/TargetReader/Bundle 架构可行性分析。新债 V149–V156（serving 层、engine adapter、per-sequence、bundle、Arrow、v0.2.12、真表门禁、依赖隔离）。完整版见 `docs/round-37-full-summary.md`。

  **Session 38（第二十四轮：Serving 层基础落地）**：
  - 新增纯 Python `ple_math.py`：Qwen PLE rowid 零第三方依赖。
  - 新增 `PleMemory` / `PleSequence` / `PleSequenceStore`：统一 Store-I/Store-P 读取、per-sequence history、continuous batching 状态容器。
  - 新增 `BundleManifest` / `TargetReaderRegistry` / `ReaderSpec`：bundle schema v1、路径解析、通用 reader 注册/加载协议。
  - Serving 层全部按需懒加载，不触发 torch/ple_adapter；`python_wheel_smoke.py` 增加对应测试。
  - 关闭 V149/V151/V152 基础部分，推进 V156；V150（通用 Engine Adapter）仍待做。

   **Session 39（第二十五轮：S3/B2/S4 + v0.2.12）**：
   - S3：`PleMemoryAdapter` / `TargetReaderHook` / vLLM-SGLang 注入别名完成。
   - B2：DiskSlotIndex v3 单文件 + offset table（Rust/Python），`slot-index build --single-file`，Rust e2e 通过。
   - B2：`gen_view_keys.py` 精确复现 view keys 流；`bench_disk_slot_index.py` 支持 `--single-file` / `--cache`。
   - S4：真表 Arrow IPC、serving A/B、真表性能阈值门禁、release gate 集成完成。
   - 本地 release gate `SKIP_BENCH=1` 通过；版本提升 v0.2.12。
   - 关闭 V142/V150/V153/V154/V155。

   **Session 40（第二十六轮：系统性思考）**：
   - 明确终极目标：EngramDB = 真实 PLE n-gram 记忆表的事实标准磁盘优先存储底座。
   - 新增技术债 V157–V165：serving Python 热路径、DiskSlotIndex v3 cache/规模、重复 tuple 语义、PyO3/ctypes 收敛、发布纪律、真实引擎 A/B、真表 nightly、真实表 e2e。
   - 后续计划重排为 Phase R1–R5：生产收敛、索引产品化、真实 serving、真表门禁/发布纪律、生态 canonical。
   - 借鉴矩阵更新：DuckDB/SQLite、RocksDB/LMDB、Arrow/Parquet、vLLM/SGLang/llama.cpp、Transformers/engram-peft/qwen35-ple。
   - 完整版见 `docs/round-40-full-summary.md` 与 Roadmap Section 28。

   **Session 42（第二十八轮：Phase A 闭环 + 并发度/行折叠实测）**：
   - **Phase A 闭环**：在 AutoDL 容器（原生 NVMe，见 §4 机器表）用**真实 Qwen3.8 PLE 行**测出可信冷读数字。
     **容器要点**：`drop_caches` 被拒（无 `cap_sys_admin`）→ 改用按文件 `fadvise(DONTNEED)`；
     `io_uring_setup` 被 seccomp 拦（`EPERM`）；镜像装 Rust 用 tuna；`python3` 在 `/root/miniconda3/bin`。
   - **方法学**：给基准加了**机械自校验**（双射 + 轮间不相交切片 + 每配置冷热裁决，
     `比值<5x` 且 `边际<2μs` 即 `>>> VOID` 并退出码 3）。它**当场拦下两次假读数**
     （真实数据其实全在 page cache；io_uring 路径一页没读却"很快"）。
   - **数字**：Store-I V4.1（48 行散读）@8t **604 μs（121%，超预算）**，@32t **220 μs（44%）**；
     Store-P（1 次折叠读）@512 tokens **14.28 μs（t=8）/ 7.69（t=32）** ⇒ **折叠增益 41.5×**。
     介质倍率 **2.4×**（NVMe vs Mac+USB），**不是** §30.6.1 曾称的 40×（已撤回）。
   - **代码**：新增 `DEFAULT_GATHER_THREADS = 32` 取代 10 处硬编码 `8`（旧值让 V4.1 超预算）；
     PyO3 `Store::new` 增加可选 `threads`；`read_records` 串行阈值 32→4（实测 16 条净损 2.6×）；
     新增门禁 `crates/engramdb-bench/src/bin/{nvme_gate,view_gate,uring_gate}.rs` 与 `scripts/nvme_raw_probe.c`。
   - **在真实 Linux 上跑测试抓到真实缺陷**：`uring_roundtrip_and_semantics` 在 io_uring 不可用时
     **硬 panic**（Docker 默认 seccomp 下必然红），已修为环境不支持时 SKIP。
   - 验证：`cargo fmt --check` 干净、`clippy --workspace --all-targets` **0 warn / 0 err**、
     `cargo test --workspace` **27 passed / 0 failed**。完整见 `docs/roadmap.md` §31–§32。

   **Session 42 续（常驻线程池 + 共享机器测量纪律）**：
   - 新增 `crates/engramdb-io/src/pool.rs`：常驻 fork-join 池取代 `gather_pp` /
     `read_records_parallel` 的每次调用 `std::thread::scope`。实测单次 spawn **30–35 μs**，
     t=32 时每次调用 ~22 个 ⇒ ~0.7 ms。换池后 batch=1 冷读 **1008 → 515 μs（1.9×）**。
     开关：`ENGRAMDB_NO_POOL=1`（退回旧行为）、`ENGRAMDB_POOL_WORKERS=n`（A/B）。
   - **坑（已记入 roadmap §33）**：池子大小**不能**用 `available_parallelism()` ——
     它遵守 cgroup CPU 配额，本机返回 **16**（`nproc`=128），会把 IO 并发度腰斩
     （冷读边际 4.5 → 8.0 μs/行）。现取 `max(available_parallelism(), 32)`，
     且 `scope_run` 在任务数 > worker 数时主动回退。
     `ViewReader::read_records` 的默认线程数同样从 `available_parallelism()` 改为 32。
   - **纪律增补（§33.6）**：① 共享机器上的 A/B 必须**配对交替**并在同一次脚本内完成 ——
     顺序扫描里"先跑"的那组会吸收上一轮余波，本轮因此**两次**把 1.9× 的收益读成 1.7× 的损失；
     ② 每个基准必须**自报实际走的代码路径**（`nvme_gate` 现在打印 `pool_workers=/pool_disabled=`），
     否则"A/B 无差异"可能只是两组跑的同一条路径。
   - 新增 `crates/engramdb-bench/src/bin/call_cost.rs`（调用开销分解探针）。
   - 验证：fmt 干净、clippy **0 finding**、`cargo test --workspace` **32 passed / 0 failed**。

   **Session 42 续（拆开 Python 后端与 C 嵌入面，roadmap §34）**：
   - **删除 Python 的 ctypes 回退**（`python/engramdb/__init__.py` 418 → 216 行）。
     它只对「源码树 + 未构建扩展」可达、**CI 从未覆盖**、会**静默降级**掩盖故障，
     且已经与 PyO3 分叉（`threads=` 只有 PyO3 有）。现导入失败直接抛带修复指引的
     `ImportError`（`cd python && maturin develop --release`）。
   - **crate 改名 `engramdb-python` → `engramdb-cabi`**，重新定位为 **C/C++ 嵌入面**。
     cdylib 名保持 `libengramdb_c`。⚠️ 它**只实现 `PLE_QWEN_V1`**，V4.1 规格未实现（V55 仍开着），
     且**当前没有任何 C/C++ 消费者** —— 是一个「保留但未启用」的面。
   - **顺带修掉三个「在 Linux 上假设 Linux 一切可用」的缺陷**：
     ① `IoUringPageReader` 在容器里硬崩 → 新增 `uring_available()` 探测 + `pread` 退化
     （短读/EOF 语义严格一致）+ `backend` 属性；`SGLangPageReader` 因此不再崩。
     ② `scripts/build_pyo3.sh` 硬编码 `.dylib`，在 Linux 必然失败 → 按平台选择。
     ③ `engramdb.__repr__()` 是死代码（模块级 `__repr__` 不覆盖 `repr(module)`）→ 已删，
     改由 `python_wheel_smoke.py` 的 `test_native_backend_is_pyo3()` 守卫
     （断言 `_USING_PYO3` + `abi_version` + **扩展实际加载路径**）。
   - 验证（AutoDL 原生 Linux）：fmt 干净、clippy **0**、`cargo test --workspace` **32 passed**、
     wheel smoke 全绿、`c_abi_smoke.py` 通过。
