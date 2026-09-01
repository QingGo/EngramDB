# 第二十六轮完整汇总（Session 40：工程收尾、CI 修复、README 与系统性思考）

> 本轮同时完成了三件事：
> 1. 把 S3/B2/S4 的可交付部分落地并发布 v0.2.12；
> 2. 获取并修复 GitHub CI 的真实失败，确保 release 全绿；
> 3. 整理 README / 集成文档，并做系统性思考、沉淀新债与后续路线。

---

## 1. 本轮计划

1. 完成 `PleMemoryAdapter` / `TargetReaderHook` 等通用 Engine Adapter。
2. 完成 DiskSlotIndex v3 单文件 + offset table，并跑真实规模验证。
3. 完成真表 Arrow IPC、serving A/B、真表性能阈值门禁。
4. 发布 v0.2.12。
5. 获取 GitHub CI 失败原因并修复，确保 release 成功。
6. 更新 README 与引擎集成文档。
7. 系统性思考：终极目标、技术债、后续计划、借鉴矩阵。

---

## 2. 本轮发现

1. **Serving 层已经从“功能原型”走到“性能瓶颈识别”**
   - `PleMemory` 真表读取可达 52K–63K tok/s。
   - `PleMemoryAdapter` 真表 torch 热路径只有约 1.6K–2K tok/s。
   - 瓶颈在 Python 侧 rowid + history + batch fetch + dequant，不是磁盘。

2. **真实 20M keys 流存在重复 rowid tuple**
   - 旧 `slot-index verify` 用单槽精确匹配会误报。
   - 必须区分“代表槽”和“全部匹配槽”。

3. **PyO3 已经足够承担 Python 集成**
   - C ABI ctypes fallback 只提供基础 Store/View，缺少 PageReader、read_records 等。
   - Python 发布不需要保留双桥，C ABI 更适合 C/C++ 外部集成。

4. **DiskSlotIndex v3 单文件格式可行**
   - 10M build 约 135s，verify 约 87s，lookup 约 165μs。
   - 但 cache 敏感性明显，大表验证需要高 cache，下一步要做 block/offset 工程化。

5. **CI 真正失败点不是逻辑，而是新版 Clippy**
   - Ubuntu CI 启用了 `clippy::chunks_exact_to_as_chunks`，`-D warnings` 直接失败。
   - 本地 mac 较旧 clippy 没有暴露该问题。

---

## 3. 做的尝试

1. 实现 Rust/Python DiskSlotIndex v3 单文件 + offset table。
2. 用 LCG keys 生成器重建 20M 真实 view keys 流。
3. 跑 1M / 10M / 20M DiskSlotIndex 构建。
4. 修复真实 20M keys 重复 tuple 的 verify。
5. 增加 Rust e2e：单文件 roundtrip、重复 tuple verify。
6. 增加 `real_arrow_smoke.py`、`bench_serving_ab.py`、`real_perf_gate.py`。
7. 获取 GitHub API 的 CI job 信息，定位 clippy 错误。
8. 更新 README、python README、engine-integration 文档。
9. 系统性整理终极目标 / 技术债 / 路线 / 借鉴矩阵。

---

## 4. 踩过的坑

1. **GitHub CI 与新 Clippy 版本不一致**
   - 本地 `cargo clippy` 通过，但 CI 因 `chunks_exact_to_as_chunks` 失败。
   - 修复：用 `as_chunks::<N>()` 替代常量 `chunks_exact()`。

2. **真实 keys 重复 tuple 导致 slot-index verify 误报**
   - 普通唯一 keys 测试不能发现。
   - 修复：`contains_slot()` 检查当前 slot 是否属于所有匹配记录。

3. **DiskSlotIndex v3 低 cache 下 verify 很慢**
   - 因为 bucket 分布、cache 容量和重读共同造成。
   - 大表验证应使用更高 `--cache`，后续应做 block index。

4. **本地 PyO3 ABI 不匹配导致 PageReader 缺失**
   - 本地 Python 3.9 与 built `_engramdb.so` ABI 不匹配，ctypes fallback 又没有 PageReader。
   - 修复：`sglang.py` 改为可选导入，缺失时 `PageReader=None`，保证 smoke 可运行。

5. **发布流程曾出现“先 tag 后修 CI / 强推 tag”**
   - 教训：以后必须 CI 全绿再 tag，禁止无验证强推。

6. **真实 20M 全表验证耗时很长**
   - 20M 构建约 500s；完整 verify 仍需稳定环境长时间运行。
   - 这不是代码正确性问题，而是规模测试需要在稳定介质/调度下跑。

---

## 5. 完成的内容

- [x] `PleMemoryAdapter` / `TargetReaderHook` / vLLM-SGLang 注入别名。
- [x] DiskSlotIndex v3 单文件 + offset table（Rust/Python）。
- [x] `slot-index build --single-file` CLI。
- [x] `scripts/gen_view_keys.py`，精确复现 view build keys 流。
- [x] 1M / 10M DiskSlotIndex v3 实测。
- [x] 真实 20M keys 生成与 20M 构建。
- [x] 修复重复 rowid tuple verify，并增加回归测试。
- [x] 真表 Arrow IPC 验证。
- [x] serving A/B 脚本，真表 + 合成。
- [x] 真表性能阈值门禁 `real_perf_gate.py`。
- [x] release gate 集成真表验证。
- [x] 修复 GitHub CI clippy 失败。
- [x] CI、release、release-assets、publish-pypi 全部成功。
- [x] v0.2.12 发布。
- [x] README、python README、engine-integration 文档更新。
- [x] Roadmap Section 28、round-40 系统思考文档。

---

## 6. 未完成的内容

- [ ] WSL/稳定环境 100M/320M DiskSlotIndex 全表长跑。
- [ ] DiskSlotIndex block index / 更均匀 hash / 原生 Rust lookup API。
- [ ] `view build --slot-index` 真实表 e2e。
- [ ] `PleMemoryAdapter` 热路径下沉 Rust/PyO3。
- [ ] 真实 vLLM / SGLang 模型级 serving A/B。
- [ ] 真表门禁进入 self-hosted / WSL nightly。
- [ ] qwen35-ple 移除本地 fallback，统一使用 EngramDB canonical。
- [ ] engram-peft 通过 Bundle / PleMemory 接入。
- [ ] Python 发布路径正式切换为 PyO3-only（当前 ctypes 仍保留为开发回退）。

---

## 7. 本轮技术债

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V157 | `PleMemoryAdapter` 真表 torch 热路径仅约 1.6K–2K tok/s | 服务化还不是生产性能 | rowid/history/fetch/dequant 下沉 Rust/PyO3 |
| V158 | DiskSlotIndex v3 对 cache 敏感，verify/查询需高 cache | 大表读放大和内存不稳定 | 评估 block index / hash 均匀性 / Rust lookup |
| V159 | 真实 20M keys 含重复 rowid tuple | 下游若误用单槽 lookup 可能拿错记录 | 固化 `lookup` 代表槽与 `lookup_all` 全部槽契约 |
| V160 | Python 双桥：PyO3 + ctypes fallback | API 不完整、双维护、误导 | Python 发布只走 PyO3；C ABI 仅 C/C++ |
| V161 | 发布流程出现“先 tag 后修 CI / 强推 tag” | release 可重复性差 | 先 CI 全绿再 tag，禁止无验证强推 |
| V162 | 通用 Engine Adapter 已落地，但无真实 vLLM/SGLang 模型级 A/B | 无法证明 serving 目标 | 真实引擎 A/B 出 CSV 阈值 |
| V163 | 真表门禁只在本地 release gate | 无法防远程回归 | self-hosted / WSL nightly |
| V164 | `view build --slot-index` 缺真实表 e2e | 生产路径未闭环 | 补真实表 e2e |
| V165 | 100M/320M DiskSlotIndex 全表长跑未完成 | 规模结论不完整 | WSL 稳定环境跑全表 |

---

## 8. 终极目标


**让 EngramDB 成为 DeepSeek Engram / Qwen PLE n-gram 记忆表的事实标准磁盘优先存储底座。**

具体可验收的终极目标：

1. **存储正确性**
   - Store-I 与 Store-P 对真实 320M 级表 byte-identical。
   - rowid / e_t / FP8 dequant 与官方实现位级一致。

2. **性能目标**
   - 训练/预处理：Store-P 批量读取路径稳定 ≥4M 等效行/s。
   - 在线推理：vLLM / SGLang / llama.cpp 接入后与原生 memory/mmap 路径差距 ≤5%。
   - 磁盘索引：1M/10M/100M/320M 全表 build + verify + lookup 有实测数据。

3. **工程产品化**
   - 单目录库、manifest、版本化格式、可迁移。
   - PyO3 为唯一 Python 主路径；C ABI 只服务 C/C++ 外部集成。
   - CI / release gate / 真表门禁全部闭环。

4. **生态一致**
   - EngramDB 是 rowid → slot、PleMemory、Bundle、TargetReader 的唯一 canonical 实现。
   - qwen35-ple、engram-peft、vLLM、SGLang、llama.cpp 都通过薄 adapter 消费，不改上游。

5. **可演进**
   - 存储格式、索引格式、bundle schema 都带版本号。
   - 任何一个新索引/服务能力都必须给真表性能数据，不能只给 synthetic smoke。

---

## 9. 技术债详细表（系统思考版）

| # | 债 | 影响 | 处置方向 |
|---|---|---|---|
| V157 | `PleMemoryAdapter` 真表 torch 热路径仅约 1.6K–2K tok/s | 服务化原型可行，但还不是生产性能 | 把 rowid + history + batch fetch + dequant 下沉 Rust/PyO3 |
| V158 | DiskSlotIndex v3 对 cache 很敏感，verify/查询需要高 cache | 大表读放大和内存占用不稳定 | 评估 block index / 更均匀 hash / 原生 Rust lookup |
| V159 | 真实 20M keys 含重复 rowid tuple，语义必须区分“代表槽”与“全部槽” | 下游若误用单槽 lookup 可能拿到非预期记录 | 固化 `lookup`=representative、`lookup_all`=all 的接口契约 |
| V160 | Python 同时保留 PyO3 主路径与 ctypes fallback，API 不完整且双维护 | 增加漂移、smoke 脆弱、误导用户 | Python 只做 PyO3；C ABI 仅保留给 C/C++ |
| V161 | 发布流程出现多次“先打 tag 再修 CI / 再强推 tag” | release 可重复性差、审计混乱 | 先 CI 全绿再 tag；tag 后不允许无验证强推 |
| V162 | 通用 Engine Adapter 已落地，但没有真实 vLLM/SGLang 模型级 A/B | 无法证明 serving 性能目标 | 用真实小模型/真实引擎做 A/B，出 CSV 阈值 |
| V163 | 真表性能门禁只在本地 release gate，GitHub CI 没有真表数据 | 无法防远程回归 | 增加 self-hosted / WSL scheduled 真表 nightly |
| V164 | `view build --slot-index` 只有 synthetic e2e，缺真实表 e2e | 不能证明生产路径闭环 | 补真实表 `view build + slot-index + verify` e2e |
| V165 | 100M/320M DiskSlotIndex 全表长跑未完成 | 规模结论还不完整 | WSL/稳定环境跑 100M/320M，记录 build/verify/lookup |

---

## 10. 后续开发计划

### Phase R1：生产路径收敛
- [ ] Python 发布路径明确为 PyO3-only；ctypes fallback 仅保留源码开发或 C/C++ 外部调用。
- [ ] `PleMemoryAdapter` / `PleMemory` 热路径下沉 Rust/PyO3。
- [ ] `PleSequence` / `PleSequenceStore` 的 per-request state 与 PyTorch module 集成做真实验证。

### Phase R2：磁盘索引产品化
- [ ] WSL 稳定环境 100M/320M DiskSlotIndex build/verify/lookup。
- [ ] 评估 block index、offset table、更均匀哈希、原生 Rust lookup API。
- [ ] 补真实表 `view build --slot-index` e2e。
- [ ] 固化 `lookup` / `lookup_all` / 重复 tuple 语义契约。

### Phase R3：真实引擎 serving
- [ ] 在真实 vLLM 上注入 `PleMemoryAdapter` / `TargetReaderHook` 并 A/B。
- [ ] 在真实 SGLang 上替换 reader / target-reader 并 A/B。
- [ ] 输出 serving CSV 阈值：延迟、tok/s、差距百分比。

### Phase R4：真表门禁与发布纪律
- [ ] 真表 nightly/self-hosted runner：Arrow、serving、DiskSlotIndex。
- [ ] release 流程：CI 全绿 → tag → release，禁止无验证强推 tag。
- [ ] 三仓 README / 版本 / 协议文档同步。

### Phase R5：生态 canonical 化
- [ ] qwen35-ple 移除本地 fallback，统一使用 EngramDB canonical。
- [ ] engram-peft 通过 Bundle / PleMemory 接入。
- [ ] vLLM / SGLang / llama.cpp 薄 adapter 统一到同一协议。

---

## 11. 借鉴矩阵

| 来源 | 借什么 | 明确不借 | 怎样帮我们接近目标 |
|---|---|---|---|
| **DuckDB** | 嵌入式、目录即库、manifest、Arrow IPC、零拷贝 | SQL / 查询优化器 / 事务引擎 | 确立“单目录可嵌入存储”的产品形态 |
| **SQLite** | 文件格式版本化、schema 迁移、嵌入式稳定性 | SQL / 通用事务 | 用于 bundle / manifest 的版本与升级纪律 |
| **RocksDB / LevelDB** | 不可变排序段、block/offset index、bloom | LSM 写放大 / compaction / 写路径 | 用于静态 DiskSlotIndex 的 block index 设计 |
| **LMDB / MDBX** | 只读 mmap、单文件、零拷贝读 | 写事务 / 通用 KV | 用于 Store-P / 索引的单文件 mmap 读 |
| **Apache Arrow / Parquet** | Arrow IPC、列式批次、chunk metadata、流式写 | 查询引擎 / 执行器 | 作为与训练器/引擎之间的数据契约 |
| **Cassandra / Bigtable** | hash/range 分桶、局部性 | 分布式 / 副本 / 一致性协议 | 用于 DiskSlotIndex 分桶与局部性优化 |
| **vLLM** | 自定义 op、splitting CUDA graph、pinned staging、async H2D | 不复制推理调度 | 指导 engine adapter 和真实 serving A/B |
| **SGLang** | Rust reader、io_uring、页缓存、异步 H2D | 不复制引擎 | 指导低层 reader 替换与 Rust 集成 |
| **llama.cpp** | 文件 lazy tensor read、mmap、自定义 backend | 不复制 GGUF / 推理核心 | 指导 Store-P 文件格式和未来 C ABI 接入 |
| **Transformers / HF** | module hooks、cache state、lazy loading | 训练内核 / 模型实现 | 用于 PleSequence / PleMemoryAdapter 的 hook 与 state 设计 |
| **engram-peft** | adapter、patch、互操作契约 | 不重复训练逻辑 | 作为消费方和跨仓 contract test |
| **qwen35-ple** | reader、checkpoint、真实实验载体 | 不接管 reader 模型实现 | 作为另一个消费方，EngramDB 只提供存储/协议 |
| **Redis / Memcached** | LRU、预取、统计遥测 | 不做通用 KV / 分布式缓存 | 用于 cache 与 serving 指标设计 |

---

## 12. 本轮纪律

1. **Python 只以 PyO3 为发布路径**：C ABI ctypes 不再作为 Python 分发依赖。
2. **所有 serving / engine 结论必须有真实引擎 A/B 数字**。
3. **所有 DiskSlotIndex 规模结论必须有大表实测**，synthetic 只做门禁。
4. **先 CI 全绿再 tag**，禁止无验证强推 release tag。
5. **重复 tuple 语义必须版本化和显式化**：`lookup` 返回代表槽，`lookup_all` 返回全部槽。
6. **存储核心保持轻量**：serving 层可选，不得阻塞核心导入。
7. **跨仓单一事实源**：EngramDB 提供 canonical 存储/索引/serving 协议，其他仓只做消费方 adapter。
