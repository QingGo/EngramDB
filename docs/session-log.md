# Session Log & 历程复盘（2026-08-29）

> 本文件是项目第一个开发 session 的**完整叙事归档**：
> 尝试→结果→教训三要素配对；已完成状态以 git 历史为锚；假设修正与指标证据与 `roadmap.md`/`design.md §7` 互链。
> 静态快照：2026-08-29 深夜。

---

## 0. 起始问题

> DeepSeek Engram 能否落盘（显存/内存不足时），用数据库技术+索引，让训练预处理与推理在"高吞吐/低延迟"下都高效？
> 方案演化（用户澄清驱动）：
> 1. 单机无 GPU、冻结 Engram 表、嫁接小模型、只培训练上层 → 简化出"纯读路径"双库架构
> 2. 最终目标权：**Qwen3.8-Flash-Next 的 51B PLE 表**（已生产化的 Engram），"Engram as database"
> 3. 独立库 `EngramDB`（Rust），与 `engram-peft` 零耦合薄联动；服务化可后做
> 4. 开发硬件：Mac（无 GPU）+ 外接硬盘（exfat）；生产目标：Linux/消费级 GPU

---

## 1. 尝试过什么 → 结果

| # | 尝试 | 结果 | 教训（如果失败/低效） |
|---|---|---|---|
| 1 | 搜 DeepSeek Engram 论文/仓库 | ✅ 完整：arXiv 2601.07372 + demo 全码 | — |
| 2 | 搜 Qwen "3.8-Next" n-gram 参数 | ✅ 实锤：Qwen3.8-Flash-Next（2026-08-26），PLE=51.2B（`[160, 320,001,536]`） | 模型名"Qwen3.8×Next"实为 Flash-Next/Qwen4 预览；命名噪音多 |
| 3 | HF 直读 config.json | ❌ 超时 | 网络分层：先用 domestic/镜像做兜底 |
| 4 | **ModelScope API 单文件直读 `config.json`** | ✅ 秒级拿到全部文本配置（`ngram_size/heads/ngram_vocab_size_base/ple_embed_dim/...`） | 用户提示"modelscope 可单文件下载"价值极大 |
| 5 | 下载 128 个 ngram shard（33 个 safetensors，~53GB） | ✅ 完成；背景双 worker 并行 | 长任务一律后台+nohup；断点续传（modelscope 支持） |
| 6 | llama.cpp PR/transformers 提取**官方哈希实现** | ✅ `refs/qwen4_exp_modeling.py`（2781 行）；拿到 `_build_layer_multipliers`/`_find_nth_prime_after`/`_shift_right_ignore_eos` 全文 | 建模代码在 transformers 而非 vLLM（vLLM 路径 404 两次）→ 多源交叉验证 |
| 7 | Python 参考实现 + Rust keygen + golden 对拍 | ✅ 全绿（18 用例，含溢出回绕/EOS 段边界） | 中间修 3 个隐性 bug（i64 回绕、trigram 头偏移 off-by-8、前文切片） |
| 8 | 验证素数推算 | ❌ 第一次算错（`is_prime` 漏偶数检测 → 20000038"素数"）→ 修正后 **Σ=320,001,446 → ceil128 = 320,001,536 = 128×2,500,012 完美闭合** | 数值闭证必须与物理表大小交叉；素数序列值应与 GGUF 元数据一致 |
| 9 | 语料：Gutenberg 5 本 | ✅ 0.68MB/174K tokens | 代表性不足，仅作下界 |
| 10 | 语料：**raw Common Crawl WET** | ❌ 弃用——样本即"黄赌毒/SEO"内容（用户指出正确） | 高质量清洗语料才是 P2 统计的正确输入；raw web 不可用作统计源 |
| 11 | 语料：FineWeb-Edu / zh / agent / cc-traces（4 源，5.8GB） | ✅ 全部完成 + 解析三域文本 + agent 请求统计 | 探速窗口 2MB 误差大（同源 3.9~1.16MB/s 波动）；路由 pick 需长程实测 |
| 12 | P1 朴素路径（逐 badge read） | 3,586 行/s ❌ | 随机行+分片散读撞 IOPS 墙——不做"页对齐+并行"一切都是白搭 |
| 13 | **P1 gather_pp**（4KB 页对齐+8 线程） | ✅ 冷 1.09M → 暖 1.44M 行/s（**300-400×**） | 升序聚页 + 分片并行是 SSD 随机读的真正解法 |
| 14 | **P4 物化视图 A/B** | ✅ 视图 8 线程 **5.23M 行/s、放大 1.60×** vs scatter 1.05M 行/s、**20.04×** → 视图 5×且省 12.5× 带宽 | 第一版视图单线程 11K IOPS 压制；**并行+对齐**是前提 |
| 15 | P3 段级 LRU v1 | ❌ 段永不复用→命中 0（模型设计错） | 模型先对齐真实行为（页级 LRU v2 才是 OS 语义） |
| 16 | P3 页级 LRU v2 | ✅ local 80% 页命中/12.8KB/tok；uniform 0.1%/64KB/tok | "语料型无压力、chat 形必须视图"量化证据 |
| 17 | P2 大语料统计：numpy `bincount(minlength=320M)` | ❌ 单 3M 段 >350s（实测判死） | 全量位图法在稀疏 320M 空间不可行 → **Rust HashMap 稀疏计数** |
| 18 | P2 v3 Rust T3（`p2rowid`，进度条） | ✅ fineweb 36s；zh ~6min（unique 1.57 亿 → HashMap 大）；agent 快 | HashMap 扩容对中大计数集是主要成本；后续可按域分桶 |
| 19 | tokenize 提速 | 整文件单序列 encode ❌（encode_char_offsets+巨 HashMap）→ 1MB chunk + 8 进程 ✅ | 分批+并行永远先行；纯 python 单文件路径是死亡路线 |
| 20 | 工具链 | pyenv 裸跑 ❌ → **uv + tuna 索引 + pyproject** ✅ | 依赖管理交给 uv；脚本依赖显式化 |
| 21 | 端到端 bit-exact 校验（python vs Rust FNV） | ✅ PASS（4096 随机行） | 校验链成为回到 CI 的基石 |
| 22 | rustup 1.56→1.98 | 直连 196KB/s；代理 271；TUNA 294；rsproxy 失败(404)；**rls-preview 死锁**（旧 toolchain 组件）→ `rename 旧工具链 + toolchain add --profile minimal` ✅ | 镜像对比要有真实体量探测；组件固化是 rustup 老坑 |
| 23 | cargo 依赖 | TUNA git 索引超时 → 改 **sparse+** ✅ | sparse 协议优先 |
| 24 | 下载路由自动化 | `-x` 布尔穿帮 → curl 直连被墙挂死 → 修复后 route-auto 生效 | 小参数错误也会全流程停摆；fallback 要真测试 |

---

## 2. 踩过的坑（含"假象"）

**代码/逻辑**
- `enable_truncation(max_length=0)` 所有 token 被截成 0；去掉才正常。
- `np.save` 写出 `.npy`（带头）而非 raw 行流 → 与 Rust `Layout` 直接 `read_at` 不兼容 → `data.tofile()` 修正。
- Python `is_prime` 漏 `%2` 检查 → 素数表错误。
- Rust `u64 * u64` 未 wrapping → debug 溢出 panic（torch int64 是回绕语义，必须 `wrapping_mul`）。
- `gather_pp` 闭包共享 `&mut out` → 改为各线程独立 staging 缓冲后主线程回填。
- 自定义 `Layout::new` mock 与 `PleLayout` 不一致（一度 key 空间错位）→ 统一 `Layout::new(128, 2_500_012, 160, 1)` 常量。

**环境/工具**
- **数据写错盘**：`data/real-rows` 在仓库本地（只有 `data/qwen38-ple-fp8` 是软链）→ 44GB 写进内置盘直到 ENOSPC；真凶是"部分软链目录"。
- **exfat 目录幻觉**：`ls` 结果曾看似目录消失（实为 BSD ls `--time-style` 不支持 + `head` 截断 + 隐藏文件排序）→ "磁盘数据丢失"假警报 30 分钟排查。
- 外盘写入其实 ~740MB/s，下载瓶颈纯在网路（1-5MB/s）。
- `grep -E ... || echo OK` 在无匹配时因 head 管道退出码为 0 而**不显示** OK → "构建成功"误判多次（后来每轮 `cargo build | tail -1` 直读）。
- 长命令 user abort 会杀掉前台任务（rustup/下载）→ 立刻制度化 all bg + nohup + log 文件。
- pyenv `which sample` 指向 shim（无 sample 模块）→ 用 `/usr/bin/sample`。

**网络/数据**
- HF `datasets-server` 直连超时；`wikitext-2-raw` 需鉴权（401/403）。
- `hf-mirror` 对 `/api/...` 部分兼容但 `/tree` 递归需手动分页（2013-20 等根目录可用）。
- ModelScope 数据集 `repo/files` REST 405/参数错误（旧接口），SDK `HubApi.get_dataset_files` 可用（中文源结构才得以摸清：Skypile 780MB×N）。
- 探速 2MB 窗口不可靠 → pick_route 之后落地会话仍可能换源（fineweb 二次跑选了 modelscope 而非代理）。

**方法论**
- "先启动再说"导致三例返工（curl 代理、原始行格式、单序列 encode）；已沉淀为 roadmap §5 剂量探针+进度条准则。
- 测量置信度：任何"预期外的慢/快"必须 sample 栈定位（两次成功案例：encode_char_offsets、arr_bincount）。

---

## 3. 已完成（锚点：git + 产物）

| 里程碑 | 内容 | 证据 |
|---|---|---|
| Q1 P0 结构/哈希闭证 | 官方实现提取、素数/偏移/乘子数值闭证（320,001,536 闭合）、Rust keygen+golden | `refs/`、`crates/engramdb-keygen`、`tests/golden.json` |
| 真实权重资产化 | config.json（modelscope API）、33 safetensors、128×[2,500,012,160] F8 行提取（`data/real-rows` 软链外盘） | `data/qwen38-ple-fp8`、`scripts/extract_ple_*` |
| M1 存储核心 | badge 布局、I2 频率索引、`gather_pp`（页对齐+分片并行）、CLI build/index/warm/verify、真表基准 | `engramdb-core`/`engramdb-io`/`engramdb`、P1 基准 |
| Q2 P4 物化视图证据 | 16 行 scatter vs 视图的 5× 与 12.5× 带宽差；结论：视图路线 | `p4view`、`probes/p4_view_notes.md` |
| Q3 P3 页级 LRU 模型 | local/uniform 两档曲线 | `p3sim` |
| Q4 P2 v3 高质语料统计 | UV 依赖、阶段化/并行/续跑、Rust T3 进度条；三域 zipf/热集/唯一行实测+**Zipf 假设修正** | `pyproject.toml`、`p2rowid`、`probes/p2_report_v2.json`、`agent_workload_stats.json`（949 会话/136K 请求真实统计） |
| Q5 外部生态调研 | vLLM/SGLang/llama.cpp/NeMo 全部现状+数据；8 个同名项目清点；DB/向量库借鉴 | `design.md §5`、README 消歧 |
| Q6 文档体系 | design.md（§7 实证化+修正项）、engram-specs.md（结构证据链）、roadmap.md（战略+深度债），本 session-log | `docs/*` |

当前唯一硬性未达门禁：**P6（LMDB/RocksDB/DuckDB 横向对照）未执行（因优先完成真表/视图/语料证据链；已列后续）。

---

## 4. 新发现的问题（按严重度）

1. **⚠️ 大语料下"频率热集"失效**（P2 修正）：30M tokens 语料下 unique rows 占表空间 37-49%、top1000 仅 3-6%；原先"训练侧 Zipf 有效"仅 agent 型负载成立（top100=99%）。→ 设计定位已改；I2 只做缓存优先级。
2. **⚠️ 视图另需 ~1× 磁盘**：51GB 原表 + 视图（4KB 槽 1.6×，≈81GB）；全量视图对 320M 表=84GB，需部分物化策略（disk 预算先行者）。
3. **⚠️ 单条视图记录延迟未测**（吞吐已测），GPU 路径未 Linux 复测 → P4b/门禁依赖。
4. **io_uring 后端未实现**（Linux 生产路径阻断；mac preadv 仅为开发便利）。
5. **P3 模拟器 local 参数未校准**（人为 20K 文档窗/80-20）——应接 P2 真实 rowid 流重跑。
6. **同名生态竞争/占名窗口**：crates.io & PyPI `engramdb` 当前空闲（ioteverythin 的 badge 未实际发布）→ 尽快注册。
7. **许可证未成文**：qwen-community-1.0 提取/嫁接边界未写 → `docs/licenses.md` 待建。
8. **下载/解析工具债**：corpus_build（探速噪声、无 sha256）、wet 脚本废弃标识、manifest 缺 provenance。
9. **探针/产品命名混乱**：原 engramdb-cli 已改名 engramdb；`engramdb-bench` 含 4 个 probe bin（publish=false）；→ 统一布局。

---

## 5. 计划要完成的部分（见 roadmap.md §4 详表）

- **Phase 1（已完成 2026-08-29 深夜）**：crate 于 `engramdb`（原 engramdb-cli）；`engramdb-bench` publish=false；`git mv` + clippy/fmt 全清（含 1 处潜在 bug：multiplicand 乘子 zip） + 双 OS GitHub Actions + 本地 `scripts/gate.sh` 全绿；LICENSE(Apache-2.0) + docs/licenses.md（Qwen Community 1.0 边界成文）+ cargo config.toml 迁移；发布准备经 `cargo package --list` 确认，**待用户 crates.io/PyPI token 占名**（顺序 keygen→core→io→engramdb）。
- **Phase 2（M1.5-A）**：io_uring 后端 + Linux 门禁复测；PrefetchPlanner→ring→ordered-gather 单链化；bitwise 入 cargo test。
- **Phase 3（M1.5-B）**：Store-P 真表视图构建器（槽位选型 4KB-pad vs 2560B）；P4 自动化 gate。
- **Phase 4（M2）**：PyO3 `engramdb` 包 + engram-peft interop + qwen adapter + **P4b 端到端 decode 模拟**（50/100 tok/s 实测化）。
- **Phase 5（M3）**：Store-P 段式 DataLoader + agent workload 注入仿真；P3 用真分布重标定。
- **Phase 6（M4）**：Arrow IPC server（多表/统计）；P5 开销 gate。
- **Phase 7（可选）**：社区上游贡献（SGLang/vLLM/llama.cpp 存储后端）+ 消费级 GPU 验收（≤5%）。

每 Phase 出口=bench 门禁 + design/roadmap 同步更新（诊断-证据-修正闭环）。

---

## 6. 关键环境/数据备忘（复现用）

- 机器：macOS（darwin），16GB 级内存；外盘 `/Volumes/My Passport`（exfat，写入 ~740MB/s）。
- 代理：`127.0.0.1:7897`（http/https）；速度实测（20MB 探针）：代理→huggingface 5.5MB/s > modelscope API 4.3 > hf-mirror 3.9（8s 探针则波动大：0.7~1.2MB/s）。
- 工具链：rustup stable 1.98（minimal / TUNA 镜像）+ cargo sparse TUNA；uv 0.8.17 + tuna index；pyenv 3.13.2（仅作宿主）。
- 权重资产：PLE 相关 33 个 safetensors（53GB）→ `data/qwen38-ple-fp8`；128 行文件（51GB）→ `data/real-rows`（软链外盘）；tokenizer 4 件 12.8MB；语料 build 目录 `data/corpus-build/`（raw 5.8GB + text 450MB）。
- 仓库：`~/code/EngramDB`（git，首提交 `9a4c1c1` 起共 9 个 commit；远程未配置——占名与 GitHub 发布待办）。
