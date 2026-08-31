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


---

# Session 7 复盘（2026-08-30 后段：发布修复 + 无源码引擎接入 + README/roadmap 重写）

## 1. 目标

- 修复 GitHub Release assets 重复上传问题。
- 把 SGLang/vLLM 接入变成“不改源码，启动前执行 hook”的形态。
- 重写 README：使用方式、架构、性能指标、优化策略。
- 系统性更新 roadmap：终极目标、技术债、借鉴矩阵、开发计划。

## 2. 尝试 → 结果

| # | 尝试 | 结果 |
|---|---|---|
| S7-1 | 定位 release-assets 失败根因 | ✅ 文件 glob 重叠，Python 包被上传两次 |
| S7-2 | 修复 glob 并改用 maturin manylinux 构建 | ✅ v0.2.3 release-assets 成功 |
| S7-3 | 增加 `patch_model_class_ple` / `install_vllm_ple` | ✅ 类级 hook，构造 LLM 前 patch |
| S7-4 | 增加 `install_sglang_ple` | ✅ 类级 hook，启动 SGLang 前 patch |
| S7-5 | 重写 README | ✅ 310 行，含性能/优化/用法/架构 |
| S7-6 | roadmap 新增第五轮复盘 | ✅ 记录 V1-V6 技术债和下一阶段计划 |

## 3. 当前状态

- v0.2.4 已打 tag 并推送，PyPI 发布中。
- 适配层不需要改 vLLM/SGLang 源码，只需要用户启动前执行 hook。
- 仍缺：真实 WSL/树莓派运行结果、真实模型类验证、端到端性能曲线。

## 4. 下一步

1. WSL / 树莓派运行 `scripts/linux_verify.sh`。
2. 选真实 vLLM/SGLang 模型类验证 `install_vllm_ple` / `install_sglang_ple`。
3. 完成 PLE 端到端性能验收。
4. 实现顺序化视图/访问序调度。


# Session 8 复盘（2026-08-30 后段：真实 Linux 实机验证闭环）

## 1. 目标

用户配置好免密 SSH 后，把 v0.2.4 的 PyPI wheel 放到真实的 Linux 环境运行完整 smoke：
树莓派（aarch64）和 Windows WSL2（x86_64）。

## 2. 尝试 → 结果

| # | 尝试 | 结果 |
|---|---|---|
| S8-1 | SSH 至树莓派 `makapi`（aarch64 / Debian / Python 3.13.5） | ✅ 免密登录成功 |
| S8-2 | 树莓派创建 venv 并安装 `engramdb-python==0.2.4` | ✅ 成功（来自 PyPI + piwheels） |
| S8-3 | 树莓派运行 `scripts/python_wheel_smoke.py` | ✅ 全绿：`vllm_plugin` OK、`PageReader` OK、`IoUringPageReader` OK、`SGLangPageReader` OK、`Store + PleDiskGather` OK |
| S8-4 | SSH 至 Windows（tailscale）并进入 WSL2 Ubuntu x86_64 | ✅ 免密登录成功，WSL 为 Linux 6.18 + Python 3.12 |
| S8-5 | WSL 使用 `uv` 创建 venv 并安装 `engramdb-python==0.2.4` | ✅ 成功（WSL 无 pip，改用 `/home/zeng/.local/bin/uv`） |
| S8-6 | WSL 运行 `scripts/python_wheel_smoke.py` | ✅ 全绿，与树莓派一致 |

## 3. 当前状态

- 技术债 **V1 已关闭**：真实 Linux 实机验证通过，不再只是 GitHub Actions CI。
- 已验证平台：
  - 树莓派 aarch64（Debian / Python 3.13）
  - WSL2 Ubuntu x86_64（Linux 6.18 / Python 3.12 via uv）
- 尚未覆盖：
  - 真实 vLLM / SGLang 模型类验证（V2）
  - 端到端 PLE decode 性能契约（V4）
  - GPU 路径

## 4. 下一步

1. 选一个真实模型类，在 WSL 中验证 `install_vllm_ple` / `install_sglang_ple`。
2. 在 WSL/GPU 上完成端到端 PLE 性能 A/B。
3. 顺序化视图/访问序调度。

# Session 9 复盘（2026-08-30 后段：真实 vLLM/SGLang 模型类验证闭环）

## 1. 目标

在 WSL2 真实 Linux 环境中安装 vLLM / SGLang，并验证：
- `install_vllm_ple` 能 patch 真实 vLLM 模型类；
- `install_sglang_ple` 能 patch 真实 SGLang 模型类；
- 替换后的 `DiskPleEmbedding` 能真实进行一次前向读取。

## 2. 环境

| 项 | vLLM | SGLang |
|---|---|---|
| WSL | Ubuntu x86_64 / Linux 6.18 | 同左 |
| Python | 3.12.3 | 3.12.3 |
| 框架 | vllm 0.28.0 | sglang 0.5.9 |
| torch | 2.13.0+cu130 | 2.9.1+cu128 |
| 模式 | CPU 单进程 + gloo | CPU 单进程 + gloo + `SGLANG_USE_CPU_ENGINE=1` |
| GPU | GTX1070 sm_61 与当前 torch 构建不兼容，未用于本次验证 | 同左 |

## 3. 验证方法

- 使用 `transformers.Qwen3Config` 构造一个极小的本地 Qwen3 模型目录（hidden=32、1 layer、vocab=128）。
- 为 `embed_tokens` 构造一个 128 宽（float32 32 维）EngramDB Store。
- 在构造模型前调用 `install_vllm_ple` / `install_sglang_ple`：
  ```python
  install_vllm_ple(
      Qwen3ForCausalLM,
      store=store,
      attr_name="model.embed_tokens",
      embedding_dim=32,
      dtype=torch.float32,
  )
  ```
- 构造真实 `Qwen3ForCausalLM` 实例。
- 断言 `model.model.embed_tokens` 已变成 `DiskPleEmbedding`。
- 调用该模块前向：`emb(torch.tensor([[1,2,3]]))`，输出 `(1,3,32)`。
- 再验证实例级 `patch_named_embedding` 同样有效。

## 4. 结果

| # | 尝试 | 结果 |
|---|---|---|
| S9-1 | WSL 安装 vLLM 0.28.0 + 本包 0.2.4 | ✅ |
| S9-2 | vLLM 真实 `Qwen3ForCausalLM` 类级 hook | ✅ `patched type: DiskPleEmbedding` |
| S9-3 | vLLM `DiskPleEmbedding.forward` | ✅ `embed out: (1,3,32) torch.float32` |
| S9-4 | vLLM 实例级 `patch_named_embedding` | ✅ |
| S9-5 | WSL 安装 SGLang 0.5.9 + 本包 0.2.4 | ✅ |
| S9-6 | SGLang 真实 `Qwen3ForCausalLM` 类级 hook | ✅ `patched type: DiskPleEmbedding` |
| S9-7 | SGLang `DiskPleEmbedding.forward` | ✅ `embed out: (1,3,32) torch.float32` |
| S9-8 | SGLang 实例级 `patch_named_embedding` | ✅ |

验证输出：
```text
# vLLM
VLLM_PLE_VERIFY_OK

# SGLang
SGLANG_PLE_VERIFY_OK
```

## 5. 当前状态

- 技术债 **V2 已关闭**：`install_vllm_ple` / `install_sglang_ple` 已在真实框架的
  真实模型类上完成功能验证。
- 仍未完成：
  - 完整 vLLM/SGLang serving 的端到端 PLE decode；
  - GPU 路径（GTX1070 与当前 torch cu130/cu128 的 sm_61 不兼容，需要 CUDA 12.6 或更老 torch 构建）；
  - 性能 A/B（V4）；
  - 顺序化视图 / 访问序调度（V6）；
  - 多表 / Arrow IPC / 服务化。

## 6. 下一步

1. 安装支持 GTX1070（sm_61）的 torch（例如 cu126 构建），重试 GPU 路径。
2. 用真实 PLE 表或大合成表完成端到端性能 A/B。
3. 顺序化视图基准与调度落地。
4. 存储产品化（多表 / Arrow IPC / 服务化）。

# Session 10 复盘（2026-08-30 后段：访问序视图 WSL 真机验证）

## 1. 目标

把当前仓库源码同步到 WSL2，用新编译的 `engramdb` 验证：
- `view build --keys`（访问序/调用方 keys 构建视图）在真实 Linux 可跑；
- 构建后的视图校验通过；
- 顺序读 vs 随机读在同一视图上的吞吐基线。

## 2. 结果

| 项 | 数值 |
|---|---|
| 输入 | `wsl-keys.txt` 前 319984 行 = 19999 grams × 16 heads |
| 构建 | `/tmp/access.view` 49MB，用时 0.4s |
| 校验 | 抽样 1000/19999 grams 全部与源表一致 |
| B 顺序 1t | 9.55M rows/s / 1529 MB/s |
| B 随机 1t | 9.25M rows/s / 1481 MB/s |
| B 顺序 8t | 29.57M rows/s / 4732 MB/s |
| B 随机 8t | 27.97M rows/s / 4475 MB/s |
| A scatter 8t | ~1.10–1.20M rows/s |

说明：该测试是构建后热页缓存下的吞吐，不是冷盘对比；访问序视图的构建/校验/读取路径已闭环，
冷盘顺序化收益仍需 O_DIRECT/fadvise 冷态 A/B。

## 3. 当前状态

- `build_view_from_keys` 和 `view bench --order seq|rand` 在真实 Linux WSL 上可用。
- V6 从“未实现”变为“构建/读取路径已验证，冷盘收益待测”。

# Session 11 复盘（2026-08-30 后段：冷盘顺序/随机 A/B）

## 1. 目标

在 WSL2 上用 `posix_fadvise(DONTNEED)` 近似冷缓存，验证访问序视图的顺序读相对随机读的真实收益。

## 2. 方法

- 使用 Session 10 构建的访问序视图 `/tmp/access.view`：
  - 19999 grams × 2560B = 49MB
- Python 脚本每次运行前对视图文件执行 `posix_fadvise(fd, 0, 0, DONTNEED)`
- 分别测量：
  - 顺序读（按物理槽位 0..N-1）
  - 随机读（LCG 固定 seed）
  - 1 线程与 8 线程

## 3. 结果

| 路径 | rows/s | MB/s |
|---|---|---|
| 冷顺序 1 线程 | 306,950 | 785.8 |
| 冷随机 1 线程 | 33,583 | 86.0 |
| 冷顺序 8 线程 | 19,306 | 49.4 |
| 冷随机 8 线程 | 20,736 | 53.1 |

关键结论：

- 单线程冷顺序 vs 冷随机：**785.8 MB/s vs 86.0 MB/s，约 9.1×**
- 这正式验证了访问序排布的核心杠杆：
  - 随机读 ≈ 86 MB/s（和此前全表随机 88.7MB/s 同一量级）
  - 顺序读 ≈ 786 MB/s（接近 SSD 顺序 930MB/s 的 84%）
- 8 线程冷读反而下降，说明小文件/冷缓存下多顺序流会互相干扰，此规模不适合多线程；真实大表需要再测。

## 4. 新增工具

- `scripts/wsl_cold_view_bench.py`：可复现的冷缓存视图 read benchmark。

## 5. 状态

- **V6 顺序化视图/访问序调度的核心收益已验证**：
  - 访问序构建 ✅
  - 校验 ✅
  - 冷盘顺序 vs 随机 ✅（1t：786 vs 86 MB/s）
- 待做：
  - 真实大表/大视图冷态复测
  - 多线程冷读策略（应避免多顺序流争抢）
  - 上游调度器接入

# Session 12 复盘（2026-08-30 后段：vLLM 真实模型类 embedding A/B）

## 1. 目标

在真实 vLLM `Qwen3ForCausalLM` 上，对：
- 内存版 `VocabParallelEmbedding`
- EngramDB 磁盘版 `DiskPleEmbedding`

做一次 CPU embedding 读取 A/B，为端到端性能提供第一个“PLE 数据面”实测锚点。

## 2. 结果

| 实现 | batch=1 | batch=4 | batch=16 |
|---|---|---|---|
| 内存 embedding | 10.2 μs/call | 10.4 μs/call | 10.9 μs/call |
| 磁盘 DiskPleEmbedding | 235.6 μs/call | 240.8 μs/call | 268.0 μs/call |
| 单 token 吞吐（batch=1） | — | — | 4,245 tok/s |
| 16 token/次吞吐 | — | — | 59,692 tok/s |

结论：

- 当前 `DiskPleEmbedding` 是“无缓存 raw disk 路径”，每次调用都走 Python dedup + `Store.fetch`。
- 小批量下约为内存 embedding 的 **23 倍延迟**，但绝对量级仍可达到数千到数万 token/s。
- 这说明下一步必须引入 **LRU/Tier 缓存**，否则 CPU 小模型 50 tok/s 的目标会被 PLE 读取拖累。

## 3. 新增工具

- `scripts/vllm_embedding_ab.py`：真实 vLLM 模型类上的 embedding A/B。

## 4. 遗留

- 还不是完整 decode tok/s 曲线，仅 PLE 数据面 micro A/B；
- GPU 路径仍不可用（GTX1070 sm_61 与当前 torch 不兼容）；
- 缓存层尚未实现。

# Session 13 复盘（2026-08-30 后段：DiskPleEmbedding LRU 缓存实现）

## 1. 目标

根据 Session 12 的 raw disk A/B 数据，给 `DiskPleEmbedding` 实现行级 LRU 缓存，降低重复 PLE 读取延迟。

## 2. 实现

- 在 `python/engramdb/vllm_plugin.py` 的 `DiskPleEmbedding` 中加入 `OrderedDict` LRU：
  - `cache_size` 默认 4096 行
  - `forward` 先批量查出未命中行，写入缓存，再从缓存拼接输出
  - 缓存满时淘汰最久未用行
- 保留原有 `PleDiskGather` 作为未命中路径。

## 3. 复测结果（真实 vLLM Qwen3ForCausalLM，CPU）

| 实现 | batch=1 | batch=4 | batch=16 |
|---|---|---|---|
| 内存 embedding | 18.2 μs/call | 10.7 μs/call | 12.1 μs/call |
| 磁盘 DiskPleEmbedding（带 LRU 缓存） | 14.0 μs/call | 21.7 μs/call | 22.9 μs/call |

对比 Session 12 raw disk：

- batch=1：235.6μs → 14.0μs（约 **17× 提升**）
- batch=4：240.8μs → 21.7μs（约 **11× 提升**）
- batch=16：268.0μs → 22.9μs（约 **12× 提升**）

现在磁盘 path 与内存 embedding 基本同量级，尤其在重复行访问场景下优势明显。

## 4. 新增工具

- `scripts/vllm_embedding_ab.py`：真实 vLLM 模型类上内存/磁盘 embedding A/B，可用于回归。

## 5. 遗留

- 首未命中仍走 raw disk，真实 PLE 冷启动仍需要预取/预热；
- 没有 Tier/TTL 策略，只有固定行数 LRU；
- 尚未做完整 decode tok/s；
- GPU 路径仍受 GTX1070 sm_61 兼容性限制。

# Session 14 复盘（2026-08-30 后段：多表 + Arrow + 最小服务原型）

## 1. 目标

把“服务化、多表、Arrow IPC”从纯路线图落到可运行原型。

## 2. 新增模块

| 模块 | 作用 |
|---|---|
| `python/engramdb/tables.py` | `Database`：多表目录注册、`list_tables`、按表 `fetch` |
| `python/engramdb/arrow_utils.py` | `store_fetch_arrow` / `view_read_arrow` / `table_to_ipc_bytes`（可选 pyarrow） |
| `python/engramdb/server.py` | 最小 TCP/JSON 服务：`ping` / `list_tables` / `fetch` / `fetch_arrow` / `view_read` |
| `scripts/service_smoke.py` | 多表 + Arrow IPC + 服务端到端 smoke |

## 3. 关键实现约束

- PyO3 的 `Store` 是 `unsendable`，不能跨线程共享。
- 因此服务端 `Database.fetch` 每次在当前线程新开 `Store`，用完即关。
- 这是“每请求独立 store”的原型权衡；后续如果要共享连接，需要 Rust 侧提供可跨线程的安全句柄或线程池。

## 4. 验证结果

```text
Database OK: ['alpha', 'beta'] b'\x01...'
Arrow OK: 2 ['rowid', 'row'] ipc_bytes 416
Server OK: ping / list_tables / fetch 全部通过
SERVICE_SMOKE_OK
```

## 5. 状态

- 多表：✅ 原型可用
- Arrow IPC：✅ 可生成 pyarrow Table 和 IPC bytes
- 服务化：✅ 最小 TCP/JSON 服务可跑
- 待做：
  - Arrow IPC bytes 已可通过服务返回，但仍是 base64 封装；后续可改二进制 length-prefix wire
  - 认证/并发/连接复用
  - 调度、预取、stats 遥测
  - 与 vLLM/SGLang 真正通过服务读取 PLE（当前仍是嵌入式直接调用）

# Session 15 复盘（2026-08-30 后段：扩展 wheel smoke + 二进制 Arrow IPC wire）

## 1. 目标

- 按 v0.2.5 发布准备要求，扩展 Python wheel smoke 覆盖新模块；
- 把服务从“JSON + base64”升级为真正的 length-prefix 二进制 wire，至少让 Arrow IPC / raw bytes 不再经过 base64 包装。

## 2. 完成内容

### 2.1 Python wheel smoke 扩展

`scripts/python_wheel_smoke.py` 新增：

- `Database` 多表读写；
- 可选 Arrow helper（有 pyarrow 时）；
- 最小 TCP/JSON 服务 ping / list_tables / fetch；
- `DiskPleEmbedding` LRU（有 torch 时）。

同时 CI 的 `python-smoke` job 增加 `scripts/service_smoke.py`，确保服务原型也进入回归。

### 2.2 二进制服务协议

新增：

- `python/engramdb/server.py`：
  - `EngramDBBinaryServer`：长度前缀 + 1-byte kind 的二进制响应；
  - 支持 `ping` / `list_tables` / `fetch_raw` / `fetch_arrow` / `view_read`；
  - 保留原 `EngramDBServer` JSON 模式作为兼容入口。
- `python/engramdb/service_client.py`：
  - `EngramDBClient`：同步二进制客户端；
  - `fetch_raw` 直接返回原始行字节；
  - `fetch_arrow` 直接返回 Arrow IPC stream 字节。

响应 kind：

- `0` = JSON
- `1` = raw bytes
- `2` = Arrow IPC stream

## 3. 验证

```text
Binary Server OK: ping/list_tables/fetch_raw
SERVICE_SMOKE_OK
```

二进制客户端在本地 Python 3.12 + PyO3 扩展下通过。

## 4. 状态与遗留

- 二进制 Arrow IPC wire：✅ 原型闭环
- JSON 兼容入口：保留
- 尚未做：
  - 连接复用 / 认证 / 限流
  - Rust 侧线程安全句柄（V8）
  - `EngramDBBinaryServer` 性能门禁（embedded vs server ≤2%）
  - Rust 多表 / manifest / serve（V12）

# Session 16 复盘（2026-08-30 后段：v0.2.5 实 wheel + Rust serve + CPU E2E A/B）

## 1. v0.2.5 真实 wheel 验证

- PyPI 已发布 `engramdb-python==0.2.5`（publish-pypi workflow 成功）。
- 在本机使用 Python 3.12 安装 macOS x86_64 wheel：
  - 版本 `0.2.5`
  - PyO3 原生绑定
  - `Database` / `EngramDBServer` / `EngramDBBinaryServer` / `EngramDBClient` 均可用
- 在安装 wheel 后运行：
  - `scripts/python_wheel_smoke.py`：通过（含 Database、JSON server、Arrow）
  - `scripts/service_smoke.py`：通过（含二进制 `fetch_raw` / `fetch_arrow`）
- 结论：v0.2.5 发布产物真实可安装、可运行。

## 2. Rust 侧多表 / manifest / serve 收敛（第一批）

新增 `crates/engramdb/src/serve.rs` 与 CLI 子命令：

- `engramdb tables <root>`：扫描多表根目录，列出含 manifest 或 shard 文件的表。
- `engramdb serve <root> --host 127.0.0.1 --port 8765`：
  - 最小 JSON TCP 服务；
  - `ping` / `list_tables` / `fetch`；
  - `fetch` 优先读取表目录中的 `manifest.json` 推断布局，缺失时接受请求参数。
- 单元测试：base64 编码、空目录表列举。
- 本地端到端验证：
  - `tables` 正确列出 `alpha`
  - `serve` 的 `ping` / `list_tables` / `fetch` 均返回正确数据
  - 带 manifest 的表可以只传 table+rowids，不需要手动传 layout 参数

这是 Rust 侧多表/服务化的第一批收敛；后续继续做 Arrow IPC、Unix socket、线程安全句柄、完整性校验。

## 3. CPU 端到端 decode A/B（第一版）

在 WSL + vLLM venv 中创建极小的 Qwen3 模型：

- hidden=32，vocab=128，1 层，max_position_embeddings=128
- 将输入 embedding 替换为 `DiskPleEmbedding`（磁盘 Store）
- 使用 `model.generate` 跑真实 prefill + decode

新增复现脚本：`scripts/cpu_tiny_decode_ab.py`

多次运行代表性结果（96/128 tokens，CPU，WSL）：

| 实现 | tok/s 范围 | 相对 memory |
|---|---|---|
| memory embedding | 253–603 | 1.00× |
| disk raw（cache=0） | 213–490 | 约 0.76–0.84× |
| disk LRU（cache=4096） | 249–488 | 约 0.79–1.03× |

结论：

- 首次拿到 CPU 小模型端到端 decode 曲线，不再是单纯 embedding micro A/B。
- 原始磁盘路径在无缓存时约慢 16–24%；
- LRU 热路径基本拉回内存水平，波动主要来自 WSL/CPU 噪声；
- 仍不是 vLLM/SGLang 完整 serving 基准，下一步继续朝真实服务引擎收敛。

## 4. 顺手修复

- `DiskPleEmbedding` 现在允许 `cache_size=0`：
  - 之前 `cache_size=0` 会先写入 LRU 后立即淘汰，导致 forward 取缓存时 KeyError；
  - 修复后 `cache_size=0` 显式走 raw no-cache 路径，可用于 A/B 和调试。

# Session 17 复盘（2026-08-30 后段：Rust manifest check + 二进制 serve）

## 1. 目标

在 Rust 侧继续收敛多表/服务能力：

- 增加 manifest 完整性校验；
- 将 Rust `serve` 从纯 JSON 升级到与 Python 客户端兼容的二进制 length-prefix 协议。

## 2. 完成内容

### 2.1 `engramdb check <root>`

新增 `crates/engramdb/src/serve.rs` 中的：

- `check_table(dir)`：
  - 解析并校验 manifest；
  - 检查每个预期 shard/badge 文件是否存在；
  - 检查文件非空且大小为 `row_bytes` 的整数倍；
  - 返回 `ok / shards_found / shards_expected / issues`。
- `check_root(root)`：扫描多表并汇总。

CLI：

```bash
engramdb check /path/to/tables-root
```

### 2.2 Rust 二进制 serve

`engramdb serve <root> --binary` 新增：

- 与 Python `EngramDBClient` 相同的 length-prefix 协议；
- `fetch_raw` 直接返回裸字节（不再 base64 包装）；
- `ping` / `list_tables` 返回 JSON kind；
- 保留原 JSON serve 为默认模式。

### 2.3 测试

- `serve` 单元测试新增 `check_table` 有效/缺文件路径；
- CLI e2e 新增 `tables_and_check_multi_table`；
- Rust 二进制协议已用 Python `EngramDBClient` 手动验证通过。

## 3. 验证结果

```text
ping True
tables ['alpha']
raw b'\x00...' 16
check: {"ok": true, "table_count": 1, ...}
```

## 4. 状态

- 多表目录发现：✅
- manifest 布局读取：✅
- manifest 完整性检查：✅ 首批
- Rust JSON serve：✅
- Rust 二进制 raw serve：✅
- 待做：
  - Rust Arrow IPC；
  - Unix socket；
  - 连接复用/认证/限流；
  - checksum 级完整性校验。

# Session 18 复盘（2026-08-30 后段：真实 Qwen3.5-0.8B CPU E2E A/B）

## 1. 模型准备

- 真实模型所在路径：
  ```text
  /Volumes/My Passport/model/Qwen3.5-0.8B
  ```
- 已在本仓库建立软链：
  ```text
  data/Qwen3.5-0.8B -> /Volumes/My Passport/model/Qwen3.5-0.8B
  ```
- `data/` 已在 `.gitignore` 中，软链不会进入 git。

## 2. 新增脚本

`scripts/qwen35_cpu_decode_ab.py`

- 加载真实 `Qwen3_5ForCausalLM`（transformers 5.16.1，WSL）
- 将 `model.embed_tokens` 替换为 `DiskPleEmbedding`
- 使用稀疏 store 文件避免实际写出 1GB embedding 行
- 支持 `memory / disk raw / disk LRU` 三路 A/B 与 CSV 输出

## 3. 实机结果（WSL + CPU，真实 0.8B）

因 WSL/CPU 负载波动较大，记录多轮代表值：

| 轮次 | memory | disk raw | disk LRU |
|---|---|---|---|
| 4 tokens, reps=3 | 3.66 tok/s | 2.84 tok/s | 2.87 tok/s |
| 4 tokens, reps=1 | 4.02 tok/s | 2.40 tok/s | 3.67 tok/s |
| 8 tokens, reps=1 | 2.12 tok/s | 3.93 tok/s | 1.53 tok/s |

初步结论：

- 已经跑通真实 0.8B 模型的端到端 CPU decode，不再是 tiny toy；
- 原始磁盘路径通常比内存慢约 20% 以上；
- LRU 在短序列/小工作集下优势不明显，因为每 token 大多是新行；
- WSL 当前噪声较大，需要更长序列 + 多次中位数才能形成稳定回归阈值。

## 4. 意义与后续

- 这是“真实模型 + 磁盘 PLE 数据面”的首个 E2E 性能锚点；
- 下一步：
  - 固定生成序列、固定 seed、增加序列长度；
  - 用真实权重填充 store（非稀疏零值）做 bit-exact 功能对照；
  - 尝试 vLLM/SGLang 对 Qwen3.5-0.8B 的真实 serving 路径；
  - 或者用 llama.cpp 替代 CPU 路径。

# Session 19 复盘（2026-08-30 后段：可信性能基线 + 真实权重 bit-exact）

## 1. 目标

把此前的“单次抖动观察”升级为可回归的 CPU decode 基线，并补上真实权重数据面：

- 固定 seed、固定输入、模型 eval、至少 5 次重复；
- 输出 median / p90 / mean / best / worst；
- 生成 `probes/cpu_tiny_baseline.csv`、`probes/qwen35_cpu_baseline.csv`；
- 增加阈值检查脚本；
- 用真实 Qwen3.5 embedding 填充 store，并验证 bit-exact。

## 2. 完成内容

### 2.1 脚本升级

- `scripts/cpu_tiny_decode_ab.py`
- `scripts/qwen35_cpu_decode_ab.py`

两脚本统一加入：

- `--seed`（默认 42）
- `--warmup`（默认 1）
- `--reps`（默认 7，强制 >=5）
- `model.eval()`
- 百分位统计：median / p90 / mean / min / max
- CSV 输出，默认写入 `probes/`
- 可选 `--max-raw-slowdown` / `--max-lru-slowdown` 回归门禁
- qwen 脚本增加 `--no-create-store`，可用已填充的真实 store 跑 A/B

### 2.2 阈值检查

新增 `scripts/decode_baseline_check.py`：

- 读取两份基线 CSV；
- 计算 memory vs disk raw / LRU 的 median 吞吐 slowdown；
- 默认阈值：
  - tiny：raw ≤50%，LRU ≤75%
  - qwen：raw ≤50%，LRU ≤50%
- 超阈值即非零退出。

### 2.3 真实权重与 bit-exact

新增 `scripts/qwen35_bit_exact.py`：

- 用 `embed_tokens.weight` 真实填充 store；
- 直接 embedding 比较：`max_abs=0.0`；
- 完整生成比较：memory 输出序列 == disk 输出序列；
- 实机结果：`BIT_EXACT_PASS`。

### 2.4 模型准备脚本

新增 `scripts/prep_real_model.sh`：

- `symlink`：默认建立 `data/Qwen3.5-0.8B` 软链；
- `copy`：可复制到 WSL 可见目录；
- `verify`：校验必需文件存在且非空。

## 3. 可信基线数据

### 3.1 tiny Qwen3（`probes/cpu_tiny_baseline.csv`）

| 路径 | median tok/s | p90 tok/s | 相对 memory 中位 slowdown |
|---|---|---|---|
| memory | 394.76 | 337.97 | — |
| disk raw | 282.39 | 259.89 | 39.8% |
| disk LRU | 244.96 | 208.02 | 61.2% |

- seed=42，seq=`5,6,7,8,9,10`，64 new tokens，7 reps。

### 3.2 真实 Qwen3.5-0.8B（`probes/qwen35_cpu_baseline.csv`，真实权重 store）

| 路径 | median tok/s | p90 tok/s | 相对 memory 中位 slowdown |
|---|---|---|---|
| memory | 4.69 | 4.64 | — |
| disk raw | 4.15 | 3.34 | 13.0% |
| disk LRU | 3.94 | 3.59 | 19.2% |

- seed=42，seq=`1,2,3,4,5`，8 new tokens，5 reps。
- 这次不再用稀疏零值 store，而是真实权重填充后的 store。

## 4. 验证结果

```text
PASS: cpu_tiny_baseline.csv disk-lru-cache4096 slowdown=61.2% (limit 75.0%)
PASS: cpu_tiny_baseline.csv disk-raw-cache0 slowdown=39.8% (limit 50.0%)
PASS: qwen35_cpu_baseline.csv disk-lru-cache4096 slowdown=19.2% (limit 50.0%)
PASS: qwen35_cpu_baseline.csv disk-raw-cache0 slowdown=13.0% (limit 50.0%)
GATE PASS: decode baseline thresholds satisfied
```

## 5. 状态与后续

- 可回归 CPU decode 基线：✅
- 真实权重 store + bit-exact：✅
- 待做：
  - 从 Qwen3.5 定位真实 PLE/Engram 表（当前仍是普通 `embed_tokens`）；
  - 把 bit-exact 直接合并进 A/B 脚本作为自动步骤；
  - vLLM/SGLang/llama.cpp serving 级 A/B；
  - v0.2.7 发布收编。

# Session 19 增补（2026-08-30 后段：自动发现真实 PLE 表属性）

在可信基线与 bit-exact 之后，补做了真实 PLE 元数据定位：

- `python/engramdb/ple_discovery.py`：纯元数据发现模块，不加载权重；
- `scripts/inspect_ple_attributes.py`：命令行快速检查真实模型是否含 PLE；
- 在 `Qwen3.8-Flash-Next / Qwen4Exp`（`/Volumes/My Passport/qwen38-ple`）中发现真实 PLE：
  - 属性路径：
    `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight`
  - `ple_layer_ids=[2]`（权重层实际为 1，因内部索引偏移）
  - `ple_embed_dim=2560`
  - `ngram_size=3`
  - `ngram_vocab_size_base=20000000`
  - `split_ngram_parts=128`
  - `heads_per_ngram=8`
  - 128 个 ngram embedding shard 权重
- `Qwen3.5-0.8B` 不含 PLE/Engram 表，因此之前真实模型 A/B 仍属于“普通 input embedding 磁盘替换”，不是真实 PLE 语义。

# Session 20 复盘（2026-08-30 后段：系统性思考与真实 PLE 路径）

## 1. 为什么做这次系统性思考

在完成可信基线和 bit-exact 后，需要回答：

- 我们距离终极目标还有多远？
- 哪些是真正的技术债，哪些只是“能跑”的噪音？
- 应该先做什么，才能稳而不乱地接近目标？

## 2. 关键结论

- 终极目标不变：磁盘优先的确定性 n-gram/PLE 存储基础设施，像 DuckDB 之于分析数据库。
- 当前最重要的缺口不是服务化，而是 **真实 PLE 数据面 + 可重叠的性能路径**。
- Qwen3.5-0.8B 没有 PLE；真实 PLE 在 Qwen4Exp/Qwen3.8 中，路径已定位。
- 当前 memory vs raw 差距中，很可能包含 Python adapter 同步开销，不能简单归因于磁盘慢。
- LRU 目前没有命中率证据，短无复用场景下反而更慢。

## 3. 已产出

- `docs/roadmap.md` 新增第 14 节：
  - 终极目标与验收指标
  - 新技术债 V33–V43
  - 第十轮借鉴矩阵
  - Phase 0–5 开发计划与退出标准
  - 五条稳定前进纪律


# Session 20 增补：真实 PLE Store 位级验证 + 修复多分片 gather 偏移 bug

## 1. 新脚本

- `scripts/real_ple_bit_exact.py`：
  - 直接对 `data/real-rows` 的 128 个真实 PLE 分片抽样 rowid；
  - 将原始文件逐行读取与 `engramdb.Store.fetch()` 返回字节做 SHA-256 对照。

## 2. 发现的严重 bug

在跑真实 PLE 验证时发现：

- `BadgeGather::gather_pp` 使用全局 `rowid * row_bytes` 作为 shard 文件内偏移；
- 对单分片正确；
- 对多分片（如真实 PLE 128 shards）会导致 `shard > 0` 的所有行读取错误；
- Python Store 之前读 shard 0 正确，跨 shard 全部 mismatch。

修复：

- `gather_pp` 改为 `rowid % rows_per_shard` 计算 shard 内局部偏移；
- 同步修复 `gather_plan` 的退化直接读路径同样的偏移问题；
- 新增回归测试 `gather_pp_multishard_uses_local_row_offsets`。

## 3. 验证结果

```text
checked=100 rowids shards=128 rows_per_shard=2500012 width=160
raw sha256:  44cfab18c4a051328d19a72d290fec9bb6680cbc0fd9fa53b6035467768eb547
store sha256: 44cfab18c4a051328d19a72d290fec9bb6680cbc0fd9fa53b6035467768eb547
mismatches: 0
PLE_STORE_BIT_EXACT_PASS
```

## 4. 意义

这是首个“真实 PLE 表 + EngramDB Store”的位级闭环验证。
修复前，多分片真实 PLE 数据面即使“能跑”，返回的也是错误行；修复后才有资格继续做真实 PLE adapter 和端到端性能。

# Session 20 增补 2：真实 Qwen4Exp PLE layer bit-exact + 磁盘适配器

## 1. 新增

- `python/engramdb/ple_adapter.py`：
  - `DiskPleNGramEmbedding`：磁盘版 Qwen PLE n-gram embedding；
  - 保持官方 rowid 生成逻辑；
  - 支持 FP8 行 * weight_scale 反量化；
  - 维护最小 token history，可用于顺序 decode 冷路径。
- `scripts/ple_layer_bit_exact.py`：
  - 不加载完整 50GB+ 模型；
  - 只加载 PLE 层的小型投影/卷积权重；
  - 使用真实 PLE 128-shard rows 作为 EngramDB Store；
  - 重新实现 PLE layer forward（projection + RMSNorm + grouped conv + gate）；
  - 对比 raw-file reference 与 EngramDB disk path。

## 2. 验证结果

```text
ref_out shape=(1, 5, 10240) dtype=torch.float32
max_abs=0.000000e+00 allclose=True
PLE_LAYER_BIT_EXACT_PASS
```

## 3. 意义

- 真实 Qwen4Exp PLE 层的前向 bit-exact 已闭环。
- 不需要把 200GB+ 的 ngram embedding 载入 RAM。
- 下一步只有完整模型级 E2E 仍受整模型内存/资产限制。

# Session 21 复盘（2026-08-30 后段：真实 PLE 数据面里程碑 + 系统性思考）

## 1. 本轮完成

- 真实 PLE 128-shard Store 位级读取闭环；
- 修复 `gather_pp` 多分片偏移 bug；
- 新增 `DiskPleNGramEmbedding` 磁盘 PLE adapter；
- 新增 `disk_ple_from_discovery` 自动工厂；
- 新增 `scripts/ple_layer_bit_exact.py`，真实 PLE 层 forward bit-exact 通过；
- 明确完整模型 E2E 仍受环境/内存限制。

## 2. 技术债

新增 V44–V51，最关键：

- 完整模型自定义加载 + 跳过 ngram_embedding 权重；
- adapter 与官方类/cache 的进一步验证；
- 完整模型性能数据缺失；
- Rust 热路径尚未实现。

## 3. 后续计划

- Phase A：让 adapter 能被完整模型真正使用；
- Phase B：准备真实 E2E 运行环境；
- Phase C：Rust 原生性能路径；
- Phase D：引擎服务化。

详细内容见 `docs/roadmap.md` 第 15 节。

# Session 21 增补：服务 qwen35-ple / engram-peft 兄弟项目

## 1. C ABI 契约

- 新增 `engramdb_abi_version`
- 新增 `engramdb_rowids_for_seq`
- 与 qwen35-ple `PleSpec` 样本对拍一致

## 2. 磁盘 MultiHeadEmbedding

- 支持 FP8 + weight_scale 反量化
- 新增 `install_real_qwen_ple_embedding`
- 修复/补强 qwen35-ple 真实 PLE FP8 注入路径

## 3. Smoke

`scripts/sibling_contract_smoke.py` 通过：

```text
[C ABI] rowids match qwen35-ple PleSpec
[C ABI] abi_version=1 rowids_for_seq OK
[DiskMultiHeadEmbedding] quick check OK
SIBLING_CONTRACT_SMOKE_OK
```

engram-peft 完整 import 在当前 Python 3.9 环境不可用，但代码路径已预留。

# Session 22 复盘（2026-08-30 后段：服务兄弟项目 + 系统性思考）

## 1. 目标

参考 qwen35-ple / engram-peft 两个兄弟项目，确认 EngramDB 能否在实际协作中：
- 保证正确性（rowids、FP8、bit-exact）
- 方便使用（配置/API）
- 为高性能热路径做准备

## 2. 完成

- C ABI：`engramdb_abi_version`、`engramdb_rowids_for_seq`
- `DiskMultiHeadEmbedding` 支持 FP8 + weight_scale
- `install_real_qwen_ple_embedding`
- `scripts/sibling_contract_smoke.py`
- qwen35-ple M0 quick 通过

## 3. 技术债

新增 V52–V59，核心：
- engram-peft 尚未真正消费 `table_source`
- qwen35-ple 真实 e2e 仍用默认 float32 注入
- scale 自动发现、C ABI Python 封装、CI、Rust 热路径

## 4. 后续

详见 roadmap 第 16 节。

# Session 23 复盘（2026-08-30 后段：修复 v0.2.7 CI + 补完 Phase A EngramDB 侧）

## 1. 发现与修复

- v0.2.7 CI 失败根因：
  - `crates/engramdb-python/src/lib.rs` 中 `engramdb_keygen` import 顺序不符合 rustfmt，导致所有 preflight 的 `cargo fmt --all --check` 失败。
  - `__init__.py` 顶层 eager import `DiskPleNGramEmbedding` 会强制加载 PyTorch，导致无 torch 的 python wheel smoke 失败。
- 修复：
  - 调整 import 顺序。
  - `DiskPleNGramEmbedding` 改为 lazy attribute import；`ple_adapter.py` 改为可选 PyTorch。
  - Python 包现在可在无 torch 环境导入并使用 Store / rowids / discovery。

## 2. 新增能力

- `engramdb.load_ple_weight_scale(model_dir)`：从真实 Qwen checkpoint 自动读取 FP8 `weight_scale`（支持 BF16/F16/F32 标量）。
- `discover_ple()` 返回中自动包含 `weight_scale`。
- `disk_ple_from_discovery(store, info)` 未传 `scale` 时自动使用 discovery 中的 weight_scale。
- `install_real_qwen_ple_embedding(store, model_dir=...)` 支持自动读取 scale。
- Python `engramdb.rowids_for_seq(tokens)` 公共 API：优先 PyO3 native，其次 C ABI，最后纯 Python。
- PyO3 新增 native `rowids_for_seq` / `abi_version`。
- C ABI smoke 独立脚本 `scripts/c_abi_smoke.py` 并接入 CI。

## 3. 验证

- `cargo fmt --all --check` ✅
- `cargo clippy --all-targets --all-features -- -D warnings` ✅
- `cargo test --workspace` ✅
- `python_wheel_smoke.py`（无 torch）✅
- `service_smoke.py` ✅
- `c_abi_smoke.py` ✅
- 真实 Qwen checkpoint `weight_scale` 读取 ✅（0.00019931793212890625）
- PyO3 / C ABI / 纯 Python rowids 三者一致 ✅

# Session 24 复盘（2026-08-30 后段：v0.2.8 发布后系统性思考 + README 刷新）

## 1. 完成

- 修复 v0.2.7 CI（rustfmt、无 torch 导入）。
- 补完 Phase A EngramDB 侧：
  - 自动读取 `weight_scale`
  - Python `rowids_for_seq()`
  - PyO3 native rowids / abi_version
  - C ABI smoke 入 CI
- 发布 v0.2.8。
- 刷新根 README 和 python/README：
  - Rust crate 安装与示例
  - Python PyPI 安装
  - 真实 PLE adapter
  - FP8 engram-peft 注入
- 新增 roadmap 第 17 节：第十三轮系统性思考。

## 2. 技术债

新增 V60–V73，重点：

- 发布前缺少完整 release gate
- README 更新晚于 v0.2.8 tag
- 兄弟侧仍未消费 `table_source`
- Rust native PLE 热路径未做
- 完整模型 E2E / serving A/B 未做

## 3. 后续

- Phase 0：发布工程稳定化
- Phase A：兄弟项目配置即用
- Phase B：真实模型 E2E
- Phase C：Rust 性能热路径
- Phase D：服务化 / 推理引擎

详见 `docs/roadmap.md` 第 17 节。

# Session 25 复盘（2026-08-30 后段：Phase 0 发布门禁 + rowid 元数据自动读取）

## 1. 完成

- 新增 `scripts/release_gate.sh`：
  - cargo fmt / clippy / test
  - 构建 PyO3 + C ABI
  - python wheel smoke / service smoke / C ABI smoke / decode baseline
  - 已接入 `scripts/bump.sh`，默认 bump 前先跑；可用 `--skip-gate` 或 `ENGRAMDB_SKIP_GATE=1` 跳过。
- `discover_ple()` 改为只读取一次 safetensors index，并自动读取：
  - `weight_scale`
  - `layer_multipliers`
- 新增 `load_ple_multipliers()` / `read_safetensors_i64()`。
- `rowids_for_seq()` 支持 `multipliers` 和 `info` 来源；非默认 multipliers 时自动走纯 Python 路径。
- `disk_ple_from_discovery()` 自动使用 discovery 返回的 `layer_multipliers`。
- `install_real_qwen_ple_embedding()` 无 scale 来源时不再静默，改为显式 warning。
- `python_wheel_smoke.py` 增加：
  - safetensors I64 读取
  - fake checkpoint discovery（scale + multipliers + shard 数）
  - 自定义 multipliers / info rowids 回归
- 根 README 与 python/README 同步补充 `load_ple_multipliers` 与 `rowids_for_seq(info=...)` 示例。

## 2. 验证

- `bash scripts/release_gate.sh`（SKIP_BENCH=1）✅
- `python_wheel_smoke.py` 全绿 ✅
- 真实 Qwen checkpoint：
  - `load_ple_multipliers` → `[23703573157769, 20109073645365, 8052911324071]` ✅
  - `discover_ple()` 返回 `weight_scale` + `layer_multipliers` ✅
  - native / custom multipliers 首 rowid 一致 ✅

## 3. Phase A 兄弟侧推进（继续开发）

- engram-peft：
  - `EngramConfig` 新增 `table_store_path` / `table_model_dir` / `table_shards` /
    `table_rows_per_shard` / `table_width` / `table_dtype` / `table_scale` /
    `table_cache_size`。
  - `get_engram_model()` 在构造 `EngramLayer` 前自动消费
    `table_source="engramdb:store"`：打开 EngramDB Store 并安装 Disk
    MultiHeadEmbedding；真实 PLE 检测到 `table_model_dir` / `table_scale` 时走
    `install_real_qwen_ple_embedding`。
  - 已直接推 engram-peft master。
- qwen35-ple：
  - `EngineConfig` 增加 `store_path` / `model_dir` / `shards` / `rows_per_shard` /
    `width` / `scale` / `cache_size` / `dtype` 字段。
  - 新增 `Qwen35PleConfig.to_engram_config()`，可直接从 YAML 桥接到
    engram-peft `EngramConfig`。
  - `run_m0_smoke.py --e2e` 改为配置驱动真实 FP8 注入，命令行增加
    `--ple-model-dir`。
  - 跨仓契约 smoke 已加入 qwen35-ple CI：checkout EngramDB + engram-peft，
    `test_cross_repo_hash_golden.py` 用轻量 torch stub 运行。
  - README 增加“配置即用”与真实 FP8 e2e 示例。
  - 已推 qwen35-ple main。

## 4. Phase B 推进

- 新增 `engramdb.official_loader`：
  - `filter_ngram_shard_state_dict`
  - `install_disk_ple_in_official_model`
  - `load_state_dict_without_ngram_shards`
- 新增 `qwen35-ple/scripts/qwen4_ple_custom_loader.py`：
  - dry-run 读取真实 PLE 元数据、shard 跳过计划。
  - 完整加载路径入口（待 Qwen4Exp transformers 大内存环境）。
- 新增 `qwen35-ple/scripts/run_real_fp8_e2e.py`：
  - 轻量加载 engram-peft 子模块，绕开 TRL/datasets。
  - 配置驱动 `table_source="engramdb:store"`。
- 已在本机跑通真实 FP8 e2e：
  - Qwen3.5-0.8B + 真实 128-shard Store-I + 自动 FP8 注入。
  - `REAL_FP8_E2E_OK`，logits 有限，生成 shape 1x10，单步约 9.6s。

## 5. 遗留

- README 最终收编需等下个版本 bump。
- README 核心示例全量化 smoke 仍需继续。
- 官方 Qwen4Exp 模型类实机验证与 memory/disk A/B 仍待大内存/新版 transformers 环境。


# Session 26 系统性思考（第十四轮）

## 1. 本轮定位

从“功能可发布”进入“真实数据面可运行、但尚未官方类验收”的阶段：

- Phase A：配置即用闭环 ✅
- Phase B：真实 FP8 e2e 首次跑通 ✅
- Phase B：官方 Qwen4Exp 完整模型 + 性能 A/B 未闭环 ❌

## 2. 核心认识

1. **“能跑”不是验收**：真实 FP8 e2e 证明了真实表 + 自动注入可运行，但还不是官方 Qwen4Exp 完整模型。
2. **加载绕过未真正完成**：`official_loader` 目前主要是过滤与替换，尚未在 `from_config` 前用轻量占位避免 200GB+ 分配。
3. **可复现性不足**：真实 e2e 是依靠本机已有库和临时 PYTHONPATH 跑通的，交付前必须固化环境。
4. **性能契约仍悬空**：没有 memory vs disk A/B，没有 hit-rate / fetch / convert 分段。

## 3. 技术债

V74–V85，详见 `docs/roadmap.md` Section 18.3。

## 4. 下一步

1. 官方 Qwen4Exp 加载前 patch ngram embedding（B1）
2. 官方类 bit-exact（B2）
3. 真实 memory vs disk A/B（B3）
4. Rust/PyO3 热路径（C）
5. 引擎 serving A/B + Store 线程安全（D）
6. 固化 e2e 环境 + 统一版本收编


# Session 26 尝试与踩坑记录

## 1. 尝试

- 直接向 engram-peft / qwen35-ple 的本地 `.git` 提交
- 用 `/tmp` 可写镜像 clone 提交并推送到 GitHub
- 在多个 Python 环境尝试真实 FP8 e2e
- 使用 `run_m0_smoke.py --e2e`，但完整 engram-peft 依赖过重
- 新写 `run_real_fp8_e2e.py` 用子模块加载绕过 TRL/datasets
- 手工从 uv cache / conda 环境复制纯 Python 依赖到 `/tmp/pylibs`
- 修改 qwen35-ple CI 加入跨仓 checkout
- 新增 `official_loader` / `qwen4_ple_custom_loader.py` dry-run
- 实际跑通真实 FP8 e2e：`REAL_FP8_E2E_OK`

## 2. 踩坑

| 坑 | 处理 |
|---|---|
| `.git` 不可写 | 使用可写镜像 clone 后 push |
| `import engram_peft` 需要 TRL/datasets | 只加载 config/model 子模块 |
| transformers 4.57 不认识 qwen3_5 | 改用 transformers 5.9 环境 |
| Python 3.10 缺 typing.override | 用 typing_extensions 补 |
| 很多 venv 没有 pip | 手动复制 uv cache 包 + dist-info |
| 混入不同 Python 版本 site-packages 导致 NumPy 崩 | 只复制纯 Python 包，不混 C 扩展 |
| 写 `outputs/` 被拒绝 | 输出到 `/tmp` |
| `--load-model` 仍未真正跳过 PLE 大分配 | 下一步在 from_config 前 patch 官方 ngram 类 |

## 3. 关键结果

```text
非 PLE tensor: 151960
PLE ngram tensor: 129
REAL_FP8_E2E_OK
elapsed ~9.6s
logits finite
generated shape [1, 10]
```

## 4. 交付物

- engram-peft master: `dc74c85`
- qwen35-ple main: `a5ca602`
- EngramDB master: `52a282c`
- 文档：
  - `docs/roadmap.md` Section 18
  - `docs/session-summary.md` Session 26
  - 本文件
  - `docs/handoff.md` 已同步

# Session 27 复盘（Phase B1/B2 代码落地 + 异步预取方向）

## 1. 完成
- `patch_official_ngram_embedding_for_disk_load()`：官方 ngram 构造前 1 行占位。
- `load_official_checkpoint_without_ngram_shards()`：safetensors 分片跳过 ngram，只加载非 PLE。
- `qwen4_ple_custom_loader.py --load-model`：占位 → 加载 → 替换。
- `qwen4_ple_official_loader_smoke.py`：冻结官方快照结构 smoke。
- `DiskPleNGramEmbedding` 支持自定义 prime_sizes/offsets、batch 维度、每 batch context、chunked streaming。
- `qwen4_ple_bit_exact_small.py`：小表官方 vs DiskPle bit-exact，batch + EOS + streaming 均 max-abs=0。
- `tests/test_phase_b_official_loader.py`：3 个 runtime 测试。
- `sparse_real_row_oracle.py`：低资源真实行 oracle。
- PyO3 `Store.fetch` 释放 GIL；Store 支持并发 fetch。
- `DiskPle.prefetch()` + future/wait；模型级 forward pre-hook。
- 新增 docs/phase-b1-b2-progress.md。

## 2. 关键结果
```text
OFFICIAL_SNAPSHOT_DISK_PLE_STRUCTURE_OK
OFFICIAL_DISK_PLE_BIT_EXACT_SMALL_OK
batch maxdiff 0.0
streaming maxdiff 0.0
SPARSE_REAL_ROW_ORACLE_OK
144 real rows byte-identical
DiskPle real-Store maxdiff vs checkpoint rows: 0.0
3 passed
```

## 3. 踩坑/发现
- 本地 Transformers 5.9.0 无 `qwen4_exp`，完整官方模型仍不能在本机跑。
- `DiskPleNGramEmbedding` 原先把 batch 展平，已修复。
- 小表 bit-exact 必须使用官方生成的 multipliers，不能直接用真实 Qwen 默认值。
- PyO3 `Store::fetch` 持 GIL 且 `unsendable`，是异步预取的核心阻塞。
- 异步预取理论可行：PLE rowid 只依赖 token ids，不依赖 hidden states。

## 4. 技术债
V86–V98，见 `docs/roadmap.md` Section 19.4。

# Session 28 复盘（性能关键路径：预取管线落地）

## 1. 完成
- PyO3 `Store.fetch` 释放 GIL，`Store` 去掉 `unsendable`。
- 并发 Store fetch smoke：同一 Store 多线程读取通过。
- `DiskPleEmbedding.prefetch()` + future/wait，支持有 cache / 无 cache。
- `DiskPleNGramEmbedding.prefetch()`。
- 模型级 `forward_pre_hook`：提前预取所有 DiskPle。
- 官方加载路径启用 prefetch。
- `sparse_real_row_oracle.py`：真实 checkpoint 行 ↔ Store ↔ DiskPle bit-exact。
- `prefetch_real_ab.py`：真实 Store 预取微基准。

## 2. 关键结果
```text
Store concurrent fetch OK
DiskPleEmbedding prefetch OK
SPARSE_REAL_ROW_ORACLE_OK
144 real rows byte-identical
DiskPle real-Store maxdiff vs checkpoint rows: 0.0
[sync]     total=192.390ms fetch_s=188.817ms
[prefetch] total=34.117ms  fetch_s=1.434ms wait_s=0.028ms issued=1024
PREFETCH_AB_SMOKE_OK
```

## 3. 踩坑/发现
- `py.allow_threads` 闭包不能 move `out`，需要借用后再使用。
- 无 cache 模式也必须消费 prefetch 结果，否则会重复同步 fetch。
- prefetch 微基准只是模拟计算窗口，不是真实模型端到端。
- Python 热路径在磁盘被隐藏后可能成为新瓶颈，需要 Rust native。

## 4. 技术债
V99–V111，见 `docs/roadmap.md` Section 20.4。

## Session 28 尝试与踩坑记录

### 1. 尝试

- 去掉 PyO3 `Store` 的 `unsendable`，让 Store 可跨线程。
- 在 `Store.fetch` 中用 `py.allow_threads` 包裹 `BadgeGather::gather_pp`。
- 给 `DiskPleEmbedding` 加后台 `ThreadPoolExecutor`、`prefetch()`、future/wait。
- 给 `DiskPleNGramEmbedding` 加 `prefetch(input_ids)`。
- 加模型级 `forward_pre_hook` 自动预取。
- 在真实 Store 上做 sync vs prefetch 微基准。
- 从原始 safetensors 按字节偏移只读真实 PLE 行，做稀疏真实行 oracle。
- 增加并发 Store fetch smoke、prefetch smoke、phase B 测试。

### 2. 踩坑

1. `py.allow_threads` 闭包不能 `move` 走 `out`，否则后面无法构造 `PyBytes`；改为捕获 `&mut out`。
2. PyO3 `Store` 必须确认内部 `BadgeGather` 是 Send/Sync 后才能去 `unsendable`。
3. 无 cache 模式下 prefetch 结果如果不单独保存，`forward` 会再次同步 fetch；增加 `_prefetched` 缓冲解决。
4. prefetch 微基准先跑 sync 会预热 OS 页缓存，导致 prefetch 的 `fetch_s` 偏小；正式 A/B 需要冷/热分离。
5. 本地 Transformers 没有 Qwen4Exp，完整官方模型无法实机验证；用小资源组合验证替代。
6. 磁盘 I/O 被隐藏后，Python 侧 rowid/转换/flatten 可能成为新的热点，不能只看 fetch 时间。

### 3. 关键结果

```text
Store concurrent fetch OK
DiskPleEmbedding prefetch OK
SPARSE_REAL_ROW_ORACLE_OK
144 real rows byte-identical
DiskPle real-Store maxdiff vs checkpoint rows: 0.0
[sync]     total=192.390ms fetch_s=188.817ms
[prefetch] total=34.117ms  fetch_s=1.434ms wait_s=0.028ms issued=1024
PREFETCH_AB_SMOKE_OK
3 passed
```

## Session 29 尝试与踩坑记录（Prefetch 生产化起步 + Mini 官方模型 A/B）

### 1. 尝试
- 给 `DiskPleEmbedding` 增加 `close()` / context manager，关闭后台 prefetch executor 并清理 pending。
- 给 `DiskPleNGramEmbedding` 增加 `close()`，并让 `prefetch()` 返回底层 future。
- 修正 `install_disk_ple_prefetch_hook()` 以兼容 PyTorch 两种 pre-hook 调用约定。
- 新建 qwen35-ple `mini_official_prefetch_ab.py`，用冻结官方 PLE layer + 真实 Store + dense 前后块做 mini A/B。
- 运行 `python_wheel_smoke.py` 和 `test_phase_b_official_loader.py` 验证。

### 2. 踩坑/发现
- 当前 PyTorch 版本调用 model forward pre-hook 时可能只传 `(module, args)`，原来的 `kwargs` 必填签名会 TypeError；改为 `kwargs=None` 默认并兼容两种形式。
- 本机小规模 A/B 受系统调度和 OS 页缓存影响很大，sync/prefetch 数字波动明显，不能作为最终性能结论。
- Mini A/B 需要至少一次未计时的 warmup 和冷/热分离后才能作为可靠基准。
- `DiskPleEmbedding.close()` 必须在关闭 Store 之前调用，否则后台 future 可能在 Store 关闭后仍在读取。

### 3. 关键结果
```text
python wheel smoke OK
3 passed
MINI_OFFICIAL_PREFETCH_AB_OK
```

### 4. 20k 预计算慢路径修复（追加）

- 尝试：把 `PleDiskGather.fetch` 从 Python 去重+切片+join 改成直接 `Store.fetch`；新增 `fetch_e_t_tensor` / `fetch_tensor`；qwen35 `real_ple.fetch_e_t` 和 precompute 切换到新路径；`run_phase0 --live-store`。
- 发现：FP8 tensor 不能直接用 `batch[idx]` 做 CPU 索引，需要先 `.to(float32)` 再索引；`fetch_e_t_tensor` 的 flat rowids 长度是 T×16，返回 shape 应为 `[T,16,160]`，不是 `[len(rowids),16,160]`。
- 结果：`python wheel smoke OK`、`cargo test -p engramdb-keygen` 4 passed、小规模真实 Store precompute 跑通。

## Session 30 系统性思考记录

- 用户要求做新一轮系统性思考：终极目标、本轮技术债、后续计划、借鉴矩阵。
- 写入 `docs/roadmap.md` Section 21：
  - 终极目标轴 A/B/C 不变；
  - 新增 V112–V117；
  - 重排 Track 0–5；
  - 借鉴 DuckDB/SQLite/Redis/RocksDB/DiskANN/vLLM/SGLang/llama.cpp/Arrow/MLPerf/engram-peft。
- 结论：先解决“测不准”，再消除 serving Python 热路径，再做真实模型 A/B 和服务化。

## Session 31 Track 0/1/2 推进记录

- 完成 qwen35 `bench_live_store.py` 阈值门禁。
- 完成 `DiskPleEmbedding` no-cache 直接 `Store.fetch` 快路径。
- 完成 prefetch 错误回退、可选超时、共享 executor 参数、wait 分布统计。
- 未完成：cache>0 路径优化、多 PLE 行级合并去重、native gather/dequant、1M 冷热基线。

## Session 32 懒加载 live-store 记录

- WSL 反馈：全量 `--live-store` 1M 会 OOM；100k token Store.fetch 约 56s，远低于 README。
- 实现 `LiveETStore` / `LiveETView`：
  - 不保留全量 e_t，只保留 rowids；
  - 训练/评测按窗口懒加载；
  - control 支持 lazy permutation。
- 结论：这是磁盘优先的正确路径，但 IO 性能仍需 Store-P / Rust / 多线程解决。
- 新增技术债 V118–V122，写入 roadmap Section 22。





