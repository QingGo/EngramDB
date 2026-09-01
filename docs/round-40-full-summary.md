# 第二十六轮系统性思考（Session 40：终极目标、新债与后续路线）

> 本轮不做大功能开发，聚焦系统性复盘：明确终极目标、沉淀本轮技术债、
> 重新排布后续计划、从相似项目借鉴可复用工程思想并划清不借边界。

---

## 1. 终极目标

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

## 2. 本轮发现的技术债

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

## 3. 后续开发计划

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

## 4. 借鉴矩阵

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

## 5. 本轮纪律

1. **Python 只以 PyO3 为发布路径**：C ABI ctypes 不再作为 Python 分发依赖。
2. **所有 serving / engine 结论必须有真实引擎 A/B 数字**。
3. **所有 DiskSlotIndex 规模结论必须有大表实测**，synthetic 只做门禁。
4. **先 CI 全绿再 tag**，禁止无验证强推 release tag。
5. **重复 tuple 语义必须版本化和显式化**：`lookup` 返回代表槽，`lookup_all` 返回全部槽。
6. **存储核心保持轻量**：serving 层可选，不得阻塞核心导入。
7. **跨仓单一事实源**：EngramDB 提供 canonical 存储/索引/serving 协议，其他仓只做消费方 adapter。
