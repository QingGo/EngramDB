# 第二十三轮完整汇总（Session 37：原生 SlotIndex CLI + 服务化架构思考）

> 本轮聚焦 EngramDB 自身。qwen35-ple 相关实验、reader、模型适配由另一个 agent 独立负责，
> 本文只在“外部依赖/验收”层面提及，不把 qwen35-ple 改动列为 EngramDB 成果。

---

## 1. 本轮目标

1. 把 DiskSlotIndex 从“Python 侧可用”推进到“产品 CLI 原生可构建/可校验”。
2. 补齐大表磁盘索引的基准工具，为 320M 全表验证做准备。
3. 系统性评估 EngramDB 如何从“磁盘 embedding 替换器”升级为“可插拔 PLE memory + target reader 服务层”。
4. 明确 EngramDB 与 qwen35-ple 的边界：EngramDB 只负责存储、索引、通用 serving 协议。

---

## 2. 本轮计划

### 2.1 工程计划

- 实现 Rust 原生 `slot-index build|verify`。
- 在 `engramdb view build` / `engramdb view verify` 中接入 slot-index 生成与校验。
- 让 Python `DiskSlotIndex` 同时支持 v1（blake2b）和 v2（FNV-1a 64），并支持生成 v2。
- 增加全表 DiskSlotIndex 基准脚本，供 WSL 10M/100M/320M 验证。
- 更新 README、roadmap、session 文档。

### 2.2 架构研究计划

- 调查现有 vLLM / SGLang 插件能力边界。
- 评估 `PleMemory`、`PleSequence`、`TargetReaderRegistry`、`Bundle Manifest` 的可行性。
- 区分“EngramDB 核心存储职责”与“高层 serving 可选模块职责”。

---

## 3. 本轮发现

### 3.1 原生 SlotIndex CLI 已闭环（V144）

- `engramdb slot-index build <keys.txt> <out_dir> [--buckets N]`
- `engramdb slot-index verify <keys.txt> <out_dir>`
- `engramdb view build ... --slot-index DIR`
- `engramdb view verify ... --slot-index DIR`
- Rust 输出 Python 兼容的 `engramdb-disk-slot-index-v2`（FNV-1a 64 分桶）。
- Python `DiskSlotIndex` 现在可读 v1 / v2，也可直接生成 v2。

### 3.2 现有 vLLM / SGLang 插件仍是“embedding 替换”形态

当前：

```text
engramdb/vllm_plugin.py     DiskPleEmbedding + patch_model_class_ple
engramdb/vllm.py            fetch_e_t_tensor / PleDiskGather
engramdb/sglang.py          SGLangPageReader + install_sglang_ple
```

它们解决的是：

```text
把模型内部某个 PLE embedding 表替换为磁盘读取
```

但真正需要的是：

```text
固定 PLE 表
    +
可变的 target-side reader
    +
backbone / engine 解耦
```

因此 EngramDB 需要新增高层 serving 抽象。

### 3.3 已有代码可以作为 serving 层基础

- `DiskPleNGramEmbedding` 已包含：
  - rowids 生成
  - n-gram history 维护
  - 磁盘读取
  - 后台 prefetch
- `rowids_for_seq_with_history` 已支持增量 decode。
- `DiskPleEmbedding` 已包含 LRU、prefetch、统计、超时回退。
- `Store` / `View` / `SlotIndex` 已支持 Store-I / Store-P 两种访问方式。

### 3.4 主要缺口

- 没有 `PleMemory` 统一封装。
- 没有显式 `PleSequence` per-sequence 状态。
- 没有 generic `TargetReader` registry。
- 没有统一 `Bundle Manifest`。
- 没有真正把 target reader 注入 vLLM / SGLang forward 的 adapter。
- 没有真实 reader checkpoint 作为 serving 验证载体（属于 qwen 侧，但 EngramDB 需要定义加载协议）。

---

## 4. 做的尝试

### 4.1 Rust 原生磁盘索引

- 新增 `crates/engramdb/src/slot_index.rs`：
  - 流式两遍构建
  - FNV-1a 64 分桶
  - 每个 bucket 排序输出小文件
  - 有界 LRU 读取器
  - `build_from_keys_file` / `verify_from_keys_file`
- 接入 `engramdb` CLI。

### 4.2 Python v2 兼容

- `python/engramdb/disk_slot_index.py`：
  - 增加 FNV-1a 64 分桶函数
  - 支持 `engramdb-disk-slot-index-v2`
  - `build(..., hash_name="fnv1a-64")`
  - `build_from_keys_file(..., hash_name="fnv1a-64")`
  - 保持 v1 向后兼容

### 4.3 CLI 集成

- `view build --slot-index <dir>`
- `view verify --slot-index <dir>`
- `slot-index build|verify` 独立子命令

### 4.4 全表基准工具

- 新增 `scripts/bench_disk_slot_index.py`：
  - 可选真实 keys 或 synthetic keys
  - native build / verify 计时
  - Python DiskSlotIndex LRU lookup 吞吐
  - 输出 JSON

### 4.5 架构调研

- 通读现有 `vllm.py` / `vllm_plugin.py` / `sglang.py` / `ple_adapter.py`。
- 确认 `DiskPleNGramEmbedding` 可作为 `PleMemory` 的基础。
- 确认 `PleSequence` 所需的 `rowids_for_seq_with_history` 已经存在。
- 确认当前 engine adapter 不能直接注入 target reader，需要新的通用层。

---

## 5. 踩过的坑

### 5.1 Rust 编译/静态检查

1. `serde_json::to_vec_pretty` / `from_slice` 返回 `serde_json::Error`，不能复用 `io_err`。
   - 修复：`map_err(|e| e.to_string())`。
2. DiskSlotIndex LRU 缓存先 `get()` 再 `insert()` 触发 Rust 借用检查。
   - 修复：先用 `contains_key` 判断，再加载/淘汰/插入。
3. Clippy `while_let_on_iterator`。
   - 修复：`for line in lines`。

### 5.2 测试与验证

4. 最初 CLI 测试只覆盖独立 `slot-index` 命令，尚未覆盖 `view build --slot-index` 端到端。
   - 已记录为未完成，后续补 e2e。

### 5.3 外部运行环境（不影响 EngramDB 代码，但影响后续真表验证）

5. WSL 长时间任务曾被 tmux 启动后因 WSL 会话退出而被终止。
   - 后续跑 320M DiskSlotIndex 全表验证时，需要改用 Windows 计划任务/持久会话等机制。
6. 主机曾出现异常硬重启。
   - Windows Event 显示 Kernel-Power 41、无 bugcheck、无 WHEA，判断为硬件级断电/不稳定，非 EngramDB 代码导致。
   - 这会影响 WSL 真表基准的可靠性，后续跑规模实验前需确认环境稳定。

---

## 6. 完成的内容

- [x] `engramdb slot-index build|verify` 原生 CLI。
- [x] `view build --slot-index` / `view verify --slot-index` CLI 集成。
- [x] Python `DiskSlotIndex` 支持 v1 / v2，并支持生成 v2。
- [x] `DiskSlotIndex.build_from_keys_file(hash_name="fnv1a-64")`。
- [x] `scripts/bench_disk_slot_index.py` 全表基准工具。
- [x] Rust CLI e2e：`slot_index_build_verify_roundtrip` 通过。
- [x] cargo check / clippy / cargo test 通过。
- [x] Python v1 / v2 本地 lookup 验证通过。
- [x] README / python README / roadmap 更新。
- [x] V144 标记完成。
- [x] 完成 vLLM / SGLang / PleMemory / TargetReader / Bundle 架构可行性分析。

---

## 7. 未完成的内容

- [ ] DiskSlotIndex 320M 真表构建/查找基准。
- [ ] DiskSlotIndex 单文件 + offset table 或原生 Rust block index（V142）。
- [ ] `view build --slot-index` 的真实表 e2e 测试。
- [ ] `PleMemory` / `PleSequence` 实现。
- [ ] `TargetReaderRegistry` / `Bundle Manifest` 实现。
- [ ] 通用 vLLM / SGLang target reader 注入 adapter。
- [ ] per-sequence state 在 continuous batching 下的生命周期协议。
- [ ] Arrow IPC / serving A/B 真实验证。
- [ ] 真表性能阈值进 CI / nightly。
- [ ] v0.2.12 发布。

---

## 8. 技术债清单（本轮新增/更新）

| # | 债 | 影响 | 处置 |
|---|---|---|---|
| V141 | DiskSlotIndex 尚无 320M 级真表构建/查找实测 | 无法确认规模可用 | 用 `bench_disk_slot_index.py` 在 WSL 跑 10M/100M/320M |
| V142 | DiskSlotIndex 每 bucket 一个文件 | 16k+ 小文件 | 单文件 + offset table 或 Rust block index |
| V149 | 没有 `PleMemory` / `PleSequence` | 引擎适配重复实现增量 e_t 逻辑 | EngramDB 新增通用 serving 层 |
| V150 | 现有 vLLM/SGLang 插件只替换 embedding，不注入 target reader | 无法满足 qwen35-ple serving | 新增通用 reader 注入 adapter |
| V151 | 没有 per-sequence 状态管理协议 | continuous batching 下无法正确维护 history | `PleSequence` + state store 协议 |
| V152 | 没有通用 reader checkpoint / bundle 协议 | 外部 reader 无法统一加载 | `TargetReaderRegistry` / `Bundle Manifest` |
| V153 | Arrow IPC / serving A/B 未验证 | 零拷贝和真实性能契约未闭环 | 真表 Arrow + 引擎 A/B |
| V154 | v0.2.12 未发布 | 新功能用户不可用 | 真表门禁后发布 |
| V155 | CI 只有 synthetic 性能门禁 | 不能防真表性能回归 | 增加真表 nightly CSV 阈值 |
| V156 | 高级 serving 模块尚未与核心依赖隔离 | 轻量环境可能被拖重 | serving 层做成可选子模块 |

---

## 9. 借鉴矩阵（本轮）

| 来源 | 借什么 | 明确不借 | 为什么不冲突 |
|---|---|---|---|
| DuckDB / SQLite | 嵌入式、单目录、manifest、零拷贝 | SQL、查询优化器、事务 | 我们只做确定性点查 + 物化视图 |
| RocksDB / LevelDB | 不可变排序段、offset index、block index | LSM 写放大、compaction | 表只读，适合静态排序文件 |
| Cassandra / Bigtable | 分桶、局部性 | 分布式、副本 | 只借分桶控制随机 I/O |
| DiskANN / Milvus | ID→offset、冷热分层、顺序预读 | ANN、向量图 | 只借 I/O 布局 |
| vLLM / SGLang | continuous batching、per-request state、异步 prefetch | 引擎调度、attention | 我们只提供存储侧状态协议 |
| PyTorch DataLoader / HF Datasets | 流式窗口、worker 分片、可复现 | 训练循环 | 我们只做数据源 |
| Arrow / IPC | schema、零拷贝、块传输 | 查询执行器 | 用于 Store-P 输出和服务边界 |
| llama.cpp / GGUF | mmap、offset table、warm cache | 量化、图执行 | 只借冷启动/顺序读 |
| PEFT / engram-peft | adapter、checkpoint/config 版本化 | 训练内核 | 只定义外部 reader 加载协议 |
| Parquet / ClickHouse | 不可变文件集、row-group、manifest | 列式查询/压缩格式 | 用于静态大表构建产物 |
| Linux io_uring / AIO | 批量异步 I/O、有界提交 | 具体引擎逻辑 | 底层 I/O 优化 |

---

## 10. 未来计划

### Phase B2：磁盘索引真表验证与产品化

- WSL 10M / 100M / 320M DiskSlotIndex 构建 + verify + lookup 基准。
- 评估单文件 + offset table，或原生 Rust block index。
- 补 `view build --slot-index` 真实表 e2e。

### Phase S1：PleMemory / PleSequence

- `PleMemory`：支持 Store / View / SlotIndex。
- `PleSequence`：per-sequence n-gram history + `current_e_t()`。
- 纯 Python / torch 单元测试，不依赖 qwen。

### Phase S2：TargetReader Registry + Bundle

- `engramdb.target_reader`：注册 + 加载协议。
- `engramdb.bundle`：统一 manifest + 路径解析 + schema version。
- 不实现任何 qwen reader。

### Phase S3：通用 Engine Adapter

- 通用 layer wrapper / forward hook。
- per-sequence state store。
- 先纯 PyTorch 单序列验证，再接入 vLLM / SGLang。
- 外部 agent 提供 qwen 模型类和 reader 工厂。

### Phase S4：Arrow / 服务 / 真表门禁 / 发布

- Arrow IPC 真表验证。
- serving A/B。
- 真表 CSV 阈值入 nightly。
- v0.2.12 发布。

---

## 11. 本轮纪律

1. EngramDB 核心保持“确定性记忆表存储”，不做 SQL / ANN / 推理引擎。
2. Serving 层是可选高层 Python 模块，不得阻塞核心导入。
3. 所有磁盘索引/scale 结论必须有真表实测。
4. 新协议必须版本化。
5. qwen35-ple 由另一 agent 负责，EngramDB 只提供通用协议和存储能力。
