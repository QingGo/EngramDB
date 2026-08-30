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

---

# Session 2 复盘（2026-08-29 深夜 ~ 30 日：发布闭环 + Phase 1/2）

## 1. 尝试 → 结果（36 项快照）

### 文档与复盘体系（本次确立）
| # | 尝试 | 结果 |
|---|---|---|
| S2-1 | design.md §7 换成实测基线（P1/P4/P3/P2 数字）+ Zipf 断言 4 处修正 | ✅ commit `9a6898b`（含 roadmap/session-log/README 索引） |
| S2-2 | 终极目标重解 + 13 项技术债 + 借鉴矩阵 + Phase 0-7 门禁 | ✅ `docs/roadmap.md`（战略篇正式化） |
| S2-3 | 第二轮复盘（发布语境）→ roadmap §6 | ✅ commit `9077903`（本轮结束后） |

### Phase 1：工程收敛 + 发布通道
| # | 尝试 | 结果 | 坑 → 处理 |
|---|---|---|---|
| S2-4 | crate 改名 `engramdb-cli`→`engramdb`（git mv）+ bench publish=false | ✅ 全体构建绿灯 | — |
| S2-5 | clippy -D warnings 全清 | ✅ 顺手修掉 keygen 迭代器 zip（我优化时一度引入语义错→立即回退为 zip 映射，后经全量测试验证） | 教训：clippy 自动修会破坏语义；每次自动修复后必须审 diff + 跑测试 |
| S2-6 | LICENSE(Apache-2.0) + docs/licenses.md（权重/语料边界成文）+ config.toml 迁移 | ✅ | — |
| S2-7 | CI ci.yml（ubuntu+macos 矩阵）+ scripts/gate.sh（fmt+clippy+test） | ✅ 全绿；本地 `~/.cargo/config` 弃用警告也顺手清掉 | — |
| S2-8 | **crates.io 占名**（keygen→core→io→engramdb） | ✅ **4/4 成功**（429 限流后补发主名） | 坑 ①：tuna replace-with 让 `cargo login`/`publish` 拒绝（疑"非 remote 源"）：`--registry crates-io` + 临时 mv config.toml 绕开，后内嵌 release.sh（trap 自动恢复）；坑 ②：keywords 限制 5 个、categories 需已支持 slug（两次 400）；坑 ③：新 crate 速率限制 429（提示窗口时间）；坑 ④：crates.io API curl 需 User-Agent（403） |
| S2-9 | **PyPI 占名** | ⚠️ `engramdb` 被相似名规则拒（现有 `engram` 0.1.0a1 → 名字太像） | 决策：发布 **`engramdb-python`**（import 名保持 `engramdb`；候选 pyengramdb 作退路）；0.1.0 wheel+sdist 已上架；相似名豁免申请排 0.2 窗口（N6） |
| S2-10 | GitHub 仓库建立（用户 QingGo/EngramDB）+ SSH 推送 + repository 字段 | ✅ | sed `a` 语法 mac 坑 → python 修改 |
| S2-11 | **可信发布链**：publish.yml 先 token 版→用户配 Trusted Publisher→改 OIDC（environment `pypi` 绑定） | ✅ 全事件断言 | PyPI 表单字段 = Project `engramdb-python` / Workflow `publish.yml` / Env `pypi`（必须与 job environment 完全一致否则 401） |
| S2-12 | bump.sh 版本化（0.1.1 练习）→ 发现 release.yml 触发陷阱 | ⚠️→✅ 改 `on: release: published`（UI 才触发）→ 用户发现 push tag 期间 publish-pypi 先跑（当时它监听 push tags 而 release.yml 监听 release 事件 → 双语义不一致） | 统一：三个工作流全部 `push: tags: v*`（单一触发源=打 tag 即全发） |
| S2-13 | **YAML 解析失败**（release.yml line 21） | ✅ 根因：step `name:` 内含 `key: value` 样式（"ordered: keygen -> …"）被 GHA 解析当嵌套键 → 换成无冒号措辞 | 教训：workflow step name 永远别放 `x: y` 形态文本 |
| S2-14 | v0.1.2 全链验证 | ⚠️ release+publish-pypi 双绿；**assets 卡死**：macos-13 runner 已退役（用户提示） | 官方核实：macos-13 退役 2025-12-04；x86_64 唯一公共=macos-15-intel（到 2027-08）；arm=macos-15；改矩阵 + workflow_dispatch |
| S2-15 | 修正后想手动重跑 v0.1.2 | ❌ tag 快照无 dispatch → 改打 **0.1.3**（patch 修 CI 属标准语义） | 教训：**workflow 文件属于 tag 快照**——修完 CI 不重打 tag 不影响旧 tag |
| S2-16 | 0.1.3 全链（crates+PyPI+assets+自动 Release 创建+4 平台二进制） | ✅ 用户确认跑完 | —— |
| S2-17 | 发布 CI 处理本地镜像替换：release.sh 自动 trap 绕开 config.toml | ✅ | —— |

### Phase 2：存储真身（部分）
| # | 尝试 | 结果 | 教训 |
|---|---|---|---|
| S2-18 | `StreamingPlanner`（badge 粒度滑窗：token 流→增量 PrefetchPlan，窗口复用/弹出重发） | ✅ 3 单测 + `gather_plan` + 端到端回填测试（5 tests 全绿） | 三个测试语义错积累：①相邻行同 badge（不是"每行一 badge"）；②**同一个 plan 实例会累计**（每次 advance 应消费新计划）；③文件须填满 badge 尺寸（read_exact EOF） |
| S2-19 | 中途"测试假失败"排查（left=16 之谜） | ✅ 结论：**不是二进制缓存 bug，是断言对象看错**（fail 在第二条 `n_badges` 断言）+ 独立/全量差异假象 | 教训：测试失败先读**全部**断言行号再归因，别先怀疑工具链 |
| S2-20 | fnv64 提到 core（公共 API + 标准向量单测） | ✅ CLI 委托，verify 语义不变 | —— |
| S2-21 | **IoBackend trait**（Preadv 默认 + open_with_backend） | ✅ 7 tests 绿，gate 全绿 | 决策：**io_uring 不可验证不进主干**（无 Linux 机）→ 留 TODO（M2 门禁）；最初草写的 urring 后端 API 用法错误且无法本机测 → 直接删除，写文档化 TODO 而非保留雷 |

## 2. 坑清单（Session 2 新收）

1. tuna `[source] replace-with` 副作用：cargo login / publish 拒绝；tuna index 同步延迟致依赖解析失败 → 发布窗口内绕开（release.sh 已自动）
2. crates.io 发布校验：keywords ≤5；categories slug 有限；429 限流（含明确解锁时间）；API 403 需 UA
3. GHA YAML：step name 内 `<t: v>` 结构 = 解析错误
4. GHA runner 生命周期：macos-13 已退役（无限排队长），macos-14 2026-11 退役，x86_64 至 2027-08 → 发布矩阵反映现实
5. workflow 文件 = tag/ref 快照（dispatch 与修复都无法作用于旧 tag——要么 force tag 要么等下一版本）
6. PyPI：相似名（en gram）；classifier 未知；dist 残留 + zsh 通配符 "no matches found"（build 前 rm dist）
7. 资产同名覆盖（4 平台同名 engramdb）→ 平台后缀命名；merge-multiple 展平会影响 glob
8. PrefetchPlan 累计语义（测试层面的认知坑）
9. Badge row 语义：同分片相邻行共享 badge——planner/performance 设计均以"行簇"为单元，不是行

## 3. 已完成（Session 2 里程碑 → git）

| 内容 | commit |
|---|---|
| 文档三件套（design §7 / roadmap / session-log v1）+ README 索引 | 9a6898b |
| Phase 1 收敛（crates 改名/gate/CI/LICENSE/licenses.md/publish metadata） | 22a1819；bfc00ca |
| crates.io 4 名占名 | 7e4dbae 前序（发布批次） |
| PyPI engramdb-python 0.1.0（相似名规避 + publish.yml v1） | 7e4dbae |
| OIDC trusted publishing + 全链 single-tag（release/publish/assets） | 22a1819→后 5a1 行（`ci: single-tag` 等） |
| bump.sh + 发布流程（0.1.1/0.1.2/0.1.3 验证） | 0f93177 路线 |
| 0.1.3 全链成功（crates 4 + PyPI + Release 4 平台资产） | 用户确认 |
| Phase 2：StreamingPlanner/gather_plan/fnv64/IoBackend | 80b78aa；bfc00ca |
| 第二轮复盘 roadmap §6 + session-log §Session2 | 本文件 |

## 4. 新发现的问题（按严重度 → 处置）

1. N1 性能契约端到端悬空（P4b 不做 50/100 tok/s 只是估算）→ 需 P4
2. N2 无 Linux 门禁（io_uring 停摆、GPU 不可测）→ P2c 租机决策（~30-50 元/月）
3. N3 release 无 preflight（test 靠自觉）→ release 前加 preflight job/本地检查
4. N4 crates.io token 粗粒度+双存 → OIDC 化（crates.io trusted publishing）Phase 3.5
5. N5 design §9 里程碑与 probes 未同步 → 随 P2b 更新
6. N6 PyPI 相似名豁免 → 0.2 窗口提交
7. 数据面无新问题（P2 数据结论：hot-set 仅 agent 型负载成立）

## 5. 计划（v2.1，详见 roadmap §6.4）

P2b CLI 端到端（warm/bench-real 接 agent 真指令序列 + 集成测试入门禁 + N3 preflight）→ P2c Linux 门禁（决策点）→ P3 Store-P 视图构建 + P4 自动门 → P4 PyO3+en gram-peft interop + **P4b 端到端 decode 曲线** → P5/P6/P7 照旧。全部出口 = gate + 文档同步。

---

# Session 3 复盘（2026-08-30 全天：Phase2b 收锥 + P4 视图/延迟深化）

## 1. 尝试 → 结果（本轮要点）

| # | 尝试 | 结果 | 教训 |
|---|---|---|---|
| S3-1 | **prep_env.py** 跨平台环境准备（quick/verify/ckpt-check/full-eval） | ✅ README 第一屏；mock/corpus/real-rows 校验三态 | 实测 mock 表是 uint8 非 f16（规格重读）；verify 不该把不确定当失败 |
| S3-2 | **真表接入修复**（data/real-rows symlink → qwen38-rows SSD） | ✅ 128 分片 48G 直接可跑；gather/verify 真表 fnv 一致；bench-real agent 1.09M rows/s | **之前"断链"= 空目录 + 无脚本引导**——资产链每环要可验证 |
| S3-3 | bench-real 真实负载化（--dist agent + stats） | ✅ agent 3.94M vs uniform 2.67M rows/s（mock 表） | 真表 agent 1.09M（真分布下热集效应转真实） |
| S3-4 | CLI e2e 集成测试（进程级 build→gather→verify→warm→bench→index） | ✅ 1 集成测试入 gate；修 BadgeGather badge_ 命名 fallback + layout_for_dir | 测试数据"行≠值"边界断言错误 3 次（110 行 3 badge 才对准）；**"旧二进制"假失败教训**再次验证：先看全部断言断言行 |
| S3-5 | **N3 preflight**（三 workflow 发布前置 fmt/clippy/test） | ✅ release.yml / publish.yml / release-assets.yml needs 完成 | —— |
| S3-6 | **P4 v2 视图构建器**（--slot 4096/2560 选型 + manifest） | ✅ 槽位定案 **2560B 紧凑**（4.50M vs 0.97M 行/s，放大 1.00 vs 1.60） | 4KB 对齐槽 = 每 IO 62% 浪费 ⇒ "对齐"要按真实 payload 定，不是无脑 4KB |
| S3-7 | **全表视图 51.2GB 构建** | ✅ 22.0 分钟流式（500K/chunk，RSS 395MB）；**中途 SSD 掉载 → 挂载后完好** | 掉载二次教训：产物/keys 别放 /tmp（keys 被清）；长 IO 要有 md5/重建命令（T3） |
| S3-8 | **全表规模效应** | 冷随机 8t = 554K 行/s（88.7 MB/s）；**顺序 930MB/s**（dd） ⇒ **顺序序 = 10x 杠杆**（未兑现） | 页缓存下"冷"不可复现——macOS 语义内省（T2） |
| S3-9 | **延迟首测（lat 探针）** | warm p50=0.75-5.2μs / p99=1.4-12.3μs；max 有 2-4ms 罕见簇（事件不当验收） | **B 场景存储延迟兑现：p99 12μs ≪ 10ms/token (低 3 个量级)**；分位数四元组即够（p50/p95/p99/max） |
| S3-10 | gate bench 自动化 + 固定输入（view-keys-20k / baseline_view.csv） | ✅ 判据 ampl≤1.05 且 B8t≥2×A8t → PASS（17.9M vs 1.34M） | gate 内联解析 bug（AMP 从已过滤 OUT 取 → 空）修掉；教训：**断言变量要可溯源** |
| S3-11 | 第三轮复盘（roadmap §7 / 本节） | ✅ T1-T7 债档 + v2.2 计划 | —— |

## 2. 本轮坑清单（新增 6 条）

1. **macOS 页缓存假冷**：SSD 上"冷"档永远半温（已有读过的部分）→ 绝对冷要 O_DIRECT（Linux/M2）
2. **SSD 掉载**（USB 掉盘）：长 IO 期间掉载 → 重挂载数据完好；但构建会话中断可复现性差（T3）
3. **/tmp 被清**:macOS 定期清理 /tmp（长 session 后 keys 文件消失）→ 产物/中间文件落 SSD/仓库
4. **zsh `===` 展开陷阱**（`echo ===` 触发词命令展开）——shell 小坑，记录避免
5. **目标二进制过期**（改了 p4view 没重建 release → gate 用旧版 FAIL）→ gate.bin 前先确认二进制存在；建议 gate 提 `cargo build --release`前置或提示
6. **grep 自匹配**（`grep "[p]4view"` 在 zsh -c 命令串内部又匹配到） 见 ps 检查曾误报进程存活 —— 用 pgrep -f 注意自匹配

## 3. 已完成（Session 3 里程碑 → git）

核心：`prep_env`（quick 就绪）/ 真表修复 / bench-real agent / CLI e2e / N3 preflight / P4 v2+v3+v4（构建器+全表+延迟）/ gate 自动化 / 三复评（roadmap §7）
（commits：267d71e→580f03c→266040a（prep/readme）、5e0dc7e→7481eea（CLI e2e/preflight）、fcc68a8（p4 v2 gate）、eab8a31（全表 v3）、29ec400（lat v4）、s3rd-roadmap §7 段）

## 4. 新发现问题（按严重度）

- T1 视图构建仅在探针（产品面未成型）→ 第一优先级（P4 前端）
- T2 绝对冷无数据（Linux M2）
- T3 SSD 资产完整性/断点（下一次构建即验）
- T4 全表 A/B 未证（抽样即可）
- T5 max 事件待归因（Linux 再测）
- T6 多表缺位（表目录化）
- T7 探针统一（随 T1）

## 5. 计划（v2.2，详见 roadmap §7.4）

P4 前端（视图 API+CLI，关 T1/T7）→ P4 v5 顺序化（关 T4 大数据量确认）→ P2b 收尾 → P4b（Linux/GPU 决策门）→ P5 v0（DataLoader）→ T3-T6 插缝。

---

# Session 4 复盘（2026-08-30 深夜~次段：存储读取面 + Python C ABI 最小桥 + engram-peft 适配原型）

## 1. 本轮目标

根据 handoff 与讨论，从“继续调存储性能”转向“先端到端可用”：
- 给 Python 侧一个不需要完整 PyO3/maturin 环境的最小 Rust 桥；
- 准备 engram-peft `MultiHeadEmbedding` 的磁盘替代实现；
- 验证“磁盘读取 + 模型 embedding 查表”语义一致。

## 2. 尝试 → 结果

| # | 尝试 | 结果 | 备注 |
|---|---|---|---|
| S4-1 | `ViewReader` / `ViewBuilder` / `build_view_from_keys` 收口 | ✅ 已入库前工作区，加单元测试 | 为 PyO3/C ABI 提供底层读取面 |
| S4-2 | 尝试 PyO3 0.24 直接依赖 | ⚠️ 离线环境无法拉齐 transitive crates（autocfg/rustversion/syn 等版本不在本地 cache） | 决定先做 C ABI + ctypes，后续网络可用再升级 PyO3 |
| S4-3 | 新增 `engramdb-python` C ABI cdylib | ✅ 构建通过 | 只依赖 workspace 已有 crate，无新增外部依赖 |
| S4-4 | Python `engramdb.Store` / `View` ctypes 包装 | ✅ Store fetch 与 View read 冒烟通过 | 返回 bytes，Python 侧自行转 torch |
| S4-5 | `examples/interop_engram_peft.py` 磁盘版 `DiskMultiHeadEmbedding` | ✅ self_check 通过 | 输出与直接查表逐元素一致 |
| S4-6 | 真实 `EngramLayer` forward（Python 3.10 + torch 2.9 本地轮 + engram-peft 源码） | ✅ `engram_layer_check` 通过 | 磁盘版 MultiHeadEmbedding 已进入真实 Engram 层前向路径 |
| S4-7 | TinyLlama 全模型 E2E（Python 3.12 + torch 2.2.2 + RMSNorm fallback） | ✅ 磁盘版 Engram 层完成 forward，并生成短文本 | `examples/engram_tinyllama_e2e.py` 已跑通 |
| S4-8 | PyO3 原生扩展 `engramdb-pyo3` | ✅ 构建并 import 成功；Python 包优先加载 PyO3，ctypes 作回退 | 使用 `/tmp/cargo-home` + RUSTFLAGS dynamic_lookup 绕过本机 cargo 缓存写入限制 |
| S4-9 | `view verify` 抽样校验 | ✅ 新增 CLI 子命令和 `verify_view` API，单测覆盖 | 对齐 T3 的“构建后自校验”方向 |

## 3. 坑 / 环境注意

1. PyO3 最低依赖版本与本地 Cargo cache 不匹配，直接新增依赖会卡在下载/权限；
   **当前不强行引入 PyO3**，C ABI + ctypes 是离线可复现的最小桥。
2. 系统 Python 3.9 + torch 2.2.2 + numpy 2.0.2 有 numpy ABI 警告，但 `torch.frombuffer`
   仍可用；因此示例避免 `numpy`，用 `bytes + torch.frombuffer` 完成校验。
3. 大段 heredoc/长命令在这个环境容易触发超时，后续编辑改为小步文件替换。
4. 本地 Downloads 里的 torch 2.9.1a0 cp310 wheel 尽管能加载和跑小模块，
   跑 TinyLlama 完整 forward 会 segfault；全模型 E2E 需要换稳定 torch。

## 4. 产出

- `crates/engramdb-python/`：C ABI cdylib（Store/View open/fetch/read/close）。
- `python/engramdb/__init__.py`：ctypes 包装，`Store` / `View` / `read_keys`。
- `examples/interop_engram_peft.py`：DiskMultiHeadEmbedding 原型 + 自检。
- `python/README.md` 更新最小桥用法。

## 5. 下一步

- ✅ 已跑通 TinyLlama + engram-peft + EngramDB 磁盘版 `MultiHeadEmbedding` 的完整文本生成。
- ✅ PyO3 原生扩展已构建并成为 Python 包首选后端，ctypes 保留回退。
- 下一步：在用户自维护的 Intel Mac PyTorch wheel / 真实 PLE 表上做性能与规模验证；
  之后接 vLLM/SGLang/llama.cpp 的兼容层。


---

# Session 5 复盘（2026-08-30 后半段：v0.2.0 发布 + PyPI 上线 + vLLM/SGLang 接入面）

## 1. 本轮目标

- 把 PyO3 扩展用 maturin 正式接入 `engramdb-python`，完成可安装的 wheel/sdist。
- 发布 v0.2.0 到 PyPI 与 crates.io。
- 发布后继续做 vLLM / SGLang 接入面，而不是只停留在调研。

## 2. 尝试 → 结果

| # | 尝试 | 结果 | 备注 |
|---|---|---|---|
| S5-1 | maturin 打包 `engramdb-pyo3` | ✅ 构建出 abi3 wheel | `cp310-abi3` 支持 Python 3.10+ |
| S5-2 | Python 包优先 PyO3、ctypes 回退 | ✅ 修复 wheel 内无 C ABI 时 import 崩溃 | ctypes 回退仍不适合纯 wheel 场景，主路径是 PyO3 |
| S5-3 | `uv build` 发布 PyPI | ⚠️ 构建成功，上传失败 | `uv build` 默认 `--compatibility off`，Linux wheel 不是 manylinux |
| S5-4 | `auditwheel repair` 转 manylinux2014 | ❌ 报 too-recent versioned symbols | 新 glibc 编译产物不能修成老 manylinux |
| S5-5 | `maturin[zig]` + `--zig` 构建 manylinux2014 | ✅ 成功 | PyPI 上传成功，`0.2.0` 正式上线 |
| S5-6 | crates.io v0.2.0 发布 | ✅ 成功 | 四个 crate 均发布成功 |
| S5-7 | SGLang 兼容 `PageReader.read_pages(fds, offsets)` | ✅ 实现并测试 | 接口对齐 SGLang `IoUringReader.read_pages` |
| S5-8 | vLLM 方向 `PleDiskGather` | ✅ 实现并测试 | dedup + 批量 fetch + expansion |
| S5-9 | 引擎接入调研补充 | ✅ 更新 `docs/engine-integration.md` | 含 vLLM/SGLang/llama.cpp 实现细节和坑 |

## 3. 坑

1. **PyPI 拒绝普通 `linux_x86_64` wheel**
   - `uv build` 会以 `--compatibility off` 构建，产生非 manylinux wheel。
   - 修复：`maturin[zig]` + `--compatibility manylinux2014 --zig`。

2. **auditwheel 无法修复“too-recent versioned symbols”**
   - 在新 glibc runner 上编译的 wheel 不能降级修复到 manylinux2014。
   - 必须用 zig 交叉构建，或使用更老的 manylinux 构建环境。

3. **PyPI publish workflow 的临时触发策略**
   - 为了完成发布，临时加了 master push 触发。
   - 发布成功后已移除，只保留 `v*` tag 和 `workflow_dispatch`。

4. **ctypes 回退在 wheel 中不完整**
   - wheel 内没有 `libengramdb_c`，如果 PyO3 丢失则回退不可用。
   - 当前接受：发布形态只依赖 PyO3。

5. **Sandbox 不能写 engram-peft / vLLM / SGLang 仓库**
   - 因此只能把适配代码放在 EngramDB 包内，等待后续在上游仓库落地。

## 4. 已完成

- `engramdb-python` 0.2.0 发布到 PyPI：
  - `cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  - `sdist`
- crates.io 四 crate `0.2.0` 发布。
- `engramdb.PageReader`：
  - `read_pages(fds, offsets)`
  - 目前 pread 实现。
- `engramdb.vllm.PleDiskGather`：
  - dedup + batch fetch + expand。
- `docs/engine-integration.md` 扩展。

## 5. 新发现问题

- PyPI 目前只有 Linux x86_64 wheel，缺：
  - Linux aarch64
  - macOS x86_64 / arm64
  - Windows
- `PageReader` 是 pread，不是 io_uring / batch。
- `PageReader` / `PleDiskGather` 是在 `0.2.0` 之后新增，尚未发布。
- 没有 Python CI smoke test。
- 没有真实 vLLM / SGLang 仓库集成。
- 没有在目标硬件上验证真实 PLE 表性能。

## 6. 计划

1. 发 `0.2.1`，包含 `PageReader` 和 `PleDiskGather`。
2. 增加多平台 wheel 构建矩阵。
3. 增加 Python 安装后 smoke test 到 CI。
4. 实现 `IoUringPageReader`，复用 `engramdb-io::UringBatchBackend`。
5. 准备 SGLang 替换 patch。
6. 准备 vLLM 插件原型。
7. 在用户自维护 torch wheel / Windows/WSL + 真实 PLE 表上做端到端性能验证。
8. 视验证结果再优化访问序 / 预取 / 缓存 / 异步 H2D。


---

# Session 6 复盘（2026-08-30 后半段：v0.2.1 发布准备）

## 1. 目标

- 把 `PageReader` / `PleDiskGather` 正式装进 0.2.1。
- 补上多平台 PyPI wheel（Linux aarch64、macOS x86_64/arm64、Windows）。
- 加 Python wheel 安装冒烟 CI。
- 顺手实现 Linux `IoUringPageReader`。

## 2. 尝试 → 结果

| # | 尝试 | 结果 | 备注 |
|---|---|---|---|
| S6-1 | 扩展 `publish.yml` 为 5 平台 wheel 矩阵 + sdist + 统一发布 | ✅ 已写入 | Linux 用 maturin zig manylinux2014；macOS/Windows 原生 |
| S6-2 | 新增 `scripts/python_wheel_smoke.py` | ✅ 本地 mac wheel 冒烟通过 | 覆盖 Store、PleDiskGather、PageReader；torch 缺失时跳过 integrations |
| S6-3 | CI 增加 `python-smoke` job | ✅ 已写入 | Linux + macOS 构建安装后跑 smoke |
| S6-4 | 实现 Linux `IoUringPageReader` | ✅ 已写入 | thread-local io_uring，按批提交；需 Linux CI/实机验证 |
| S6-5 | 修复 bump.sh | ✅ 已写入 | 现在同时更新 workspace 版本、crate 依赖版本引用、Python `__version__` |
| S6-6 | 新增 `engramdb.sglang.SGLangPageReader` / `install_sglang_io_uring_reader` | ✅ 已写入 | SGLang `IoUringReader` 同形适配 |
| S6-7 | 新增 `engramdb.vllm_plugin.DiskPleEmbedding` / `patch_named_embedding` | ✅ 已写入 | vLLM PLE 磁盘替换原型，内部走 `PleDiskGather` |

## 3. 坑

1. `bump.sh` 原先只改每个 crate 的自身版本，**没有改依赖引用**（如 `engramdb-io` 仍依赖 core `0.2.0`），也没有改 workspace 根版本和 Python `__version__`；本次已修正脚本。
2. Python smoke 最初 `import engramdb.integrations` 直接要求 torch，导致普通 CI 环境失败；改为可选 import。
3. 本地 mac 构建时系统 Python 3.9 不满足 abi3-py310，需用 miniconda Python 3.13 构建。

## 4. 状态

- ✅ v0.2.1 已发布：crates.io 四 crate；PyPI `engramdb-python 0.2.1` 包含
  Linux x86_64/aarch64、macOS x86_64/arm64、Windows x86_64 共 5 个 wheel + sdist。
- ✅ v0.2.2 已发布，新增 `engramdb.sglang` / `engramdb.vllm_plugin` 适配原型到 PyPI。
- `IoUringPageReader` 已通过 GitHub Actions Linux 构建和 wheel 冒烟，但尚未在用户指定的
  WSL/树莓派上实机跑性能。
- GitHub CI（Ubuntu Linux）已验证 0.2.2 wheel 的 Python smoke 通过。

## 5. 下一步

1. 在 WSL / 树莓派上安装 `engramdb-python==0.2.2` 并跑 `scripts/python_wheel_smoke.py`。
2. 把 SGLang / vLLM 原型接到真实仓库做功能验证。
3. 目标硬件真实 PLE 端到端。

