# DeepSeek-V4.1-Flash Engram 技术报告解读 —— 对 EngramDB 的新思路

> 整理日期：2026-09-12
> 一手证据：`DeepSeek_V41_Tech_Report.pdf`（51 页，经 hf-mirror 获取；huggingface.co 直连超时）
> 官方代码：`inference/engram.py` / `inference/model.py` / `inference/convert.py` / `config.json`
> 权重元数据：`model-00047-of-00048.safetensors` header（Range 请求实测）
>
> 本文分级标注：**【报告原文】**= 报告/官方代码直接写明；**【实测】**= 本次从官方权重/config 拉取并验证；
> **【推算】**= 由前两者推导，未在报告中逐字出现；**【建议】**= 对 EngramDB 的提案，尚未验证。

---

## 0. 结论先行

**一句话**：V4.1 用生产级证据确认了 EngramDB 的核心赌注（确定性寻址 → 可提前生成的预取计划），
但同时把问题从"**本地 NVMe 上的随机点查**"升级为"**跨机内存/RDMA + 多级生命周期 + 多模块截止时间**"。

三个最重要的事实变化：

1. **规模上了一个数量级，但布局变好了。** Engram 从 Qwen PLE 的 51.2B 参数 / 单层 / 16 行每 token，
   变成 **196.6B 参数 / 两层 / 48 行每 token**；每 token payload 从 2,560 B 涨到 **12,672 B（4.95×）**。
   但 HF checkpoint 的张量已经是 **`[rows, 256]` 行主序 + `[rows, 8]` scale 的 SoA 布局**【实测】——
   也就是我们花了半个项目去物化的 "Store-P 紧凑槽"，V4.1 出厂就是这个形状。
   **结论：对 V4.1，"视图物化"不再是 EngramDB 的主要价值；价值转移到内存层级、预取编排与分片路由。**

2. **介质从 NVMe 变成 RDMA/宿主内存。**【报告原文】§2.4.2：
   "deterministic addressing enables embeddings to be prefetched from host memory via **background RDMA transfers**"。
   发布公告也点名 "2,000 GPUs + a storage cluster"。我们现在的三级 T1 RAM/T2 page cache/T3 NVMe
   在 V4.1 的部署里是**不够用的**——需要第四级"远端内存"。

3. **两个 Engram 模块的预取截止时间差 13 层。**【推算】模块在 layer 1 与 layer 14，layer 1 的预取只能
   与第一个 Transformer block 重叠（几乎无余量），layer 14 有 13 层余量。
   **这直接给出一个新产品原语：按模块分配存储层级与副本策略。**

---

## 1. V4.1 Engram 精确规格（已闭合验证）

### 1.1 报告原文要点

**【报告原文】§2.4.2（模型侧）**

- 沿用原始 Engram 设计（tokenizer compression、multi-head hashing、context-aware gating、multi-branch
  integration），**两处修改**：
  1. **去掉 short causal convolution** —— 推理栈里收益不抵复杂度；
  2. 嵌入表优化器改为 **momentum + Sinkhorn balancing**。
- **196B Engram 参数**，均分到 **两个模块**。
- 每模块 N-gram 阶 **{2, 3, 4}**，**8 个 hash head**，**每阶总嵌入维度 2048**。
- 每个 head 索引约 **16M** 条目的表，表大小为**互不相同的素数**。
- **嵌入表与 key/value 投影都用 FP8**。
- 模块放在 **layer 1 和 layer 14**（zero-indexed）。
- 推理时确定性寻址 → **从宿主内存经后台 RDMA 预取**；第一个模块的预取与第一个 Transformer block 重叠。

**【报告原文】§3.1.3（训练/系统侧）**

- 表按行切分到 `engram parallel size` 的专用进程组；优化器状态再跨副本切分。
- **lookup index 只依赖输入 token 序列** → 每个 pipeline stage 处理 micro-batch **之前**，
  就对**整个 local batch** 发起预取。
- 反向时缓冲梯度，backbone 反向结束后回传 owner rank。
- 多模态：预取与梯度传输与 vision encoder 的前向/反向重叠。
- **嵌入以 FP8 存储与取回，检索到的值与 scaling factor 直接进入后续 GEMM。**
- Sinkhorn 归一化跨迭代维护行/列 scaling 向量，**避免反复写整个归一化矩阵**。
- **RL rollout 期间 Engram 表常驻 GPU 显存**，以降低宿主内存压力、避免宿主内存碎片导致的 OOM。

### 1.2 config + 官方代码闭合验证【实测】

`config.json`（text_config）：

| 键 | 值 |
|---|---|
| `engram_layer_ids` | `[1, 14]` |
| `engram_num_embeddings` | `[384006168, 384016682]` |
| `engram_max_ngram_size` | `4` |
| `engram_vocab_size` | `16000000`（每 (阶, head) 素数搜索起点） |
| `engram_n_heads` | `8` |
| `engram_head_dim` | `256` |
| `engram_pad_token_id` | `2` |
| `engram_compressed_vocab_size` | `99092` |

权重张量（shard 47 of 48，Range 实测 header）：

```
layers.1.engram.embed.weight   F8_E4M3  [384006168, 256]
layers.1.engram.embed.scale    F8_E8M0  [384006168, 8]
layers.1.engram.wkv.weight     F8_E4M3  [25600, 6144]
layers.1.engram.wkv.scale      F8_E8M0  [800, 192]
layers.1.engram.q_weight       BF16     [4, 5120]
layers.1.engram.k_weight       BF16     [4, 5120]
```

**素数分配闭合**（复刻 `EngramLayout.from_args` + `find_next_prime(seen)`）：
48 个素数互不重复，layer 1 的 24 个素数之和 = **384,006,168**，
layer 14 的 24 个素数之和 = **384,016,682** —— 与 `engram_num_embeddings` **逐位相等**。
总参数 = 768,022,850 行 × 256 = **196.6B**，与报告的 "196B Engram parameters" 吻合。

**乘子闭合**：`np.random.default_rng(10007 * layer_id)`，`bound = (i64::MAX // 99092) // 2 = 46,539,438,283,891`，
乘子 = `v*2+1`：

| layer | rng seed | multipliers（4 个，对应 2/3/4-gram 位移） |
|---|---|---|
| 1 | 10007 | `[76632096046245, 4839876093313, 35959672319349, 73987337458391]` |
| 14 | 140098 | `[67716810739261, 51510806800915, 30921347202721, 82619226485591]` |

> 注意：bound 用的是**压缩后**词表 99092，不是模型词表 129280。官方代码里有
> `assert vocab_size == args.engram_compressed_vocab_size`，注释明确写着
> "a mismatch there would silently rehash the whole table"。

### 1.3 存储预算【推算】

| 项 | 值 |
|---|---|
| 总行数 | 768,022,850（两层合计） |
| FP8 payload | 196.6 GB |
| E8M0 scale | 6.14 GB（占 3.1%） |
| 合计 | **202.8 GB** |
| 每行记录 | 256 B + 8 B = **264 B** |
| 每 token 行数 | 2 层 × 3 阶 × 8 head = **48 行** |
| 每 token payload | **12,288 B + 384 B scale = 12,672 B** |
| 对比 Qwen PLE | 16 × 160 = 2,560 B → **4.95×** |
| MP=8 每 rank | 约 48.0M 行 × 264 B × 2 层 ≈ **25.3 GB** |

**分片边界不落在桶边界上**【实测】：24 个素数桶（每桶约 16M 行）与 `part_num_embeddings = ceil(rows/mp)`
在 mp ∈ {2,4,8,16,32} 下**没有一个精确对齐**（0/23）。所以"rank 分片 = 整桶"不成立；
但 `rowid // part_num_embeddings` 仍是**纯算术的 rank 路由函数**。

### 1.4 与 EngramDB 现有基线的差异

| 维度 | 现有基线（Qwen PLE） | V4.1 Engram | 影响 |
|---|---|---|---|
| 阶数 | 2（2/3-gram） | **3（2/3/4-gram）** | keygen 需泛化到 4-gram |
| 层数 | 1（`ple_layer_ids=[2]`） | **2（layer 1, 14）** | 多表 + 分级截止时间 |
| head / token | 16 | 24 / 层，共 **48** | IOPS 3× |
| 行宽 | 160 B | **256 B** | 记录 264 B |
| 行数 | 320,001,536（单表） | 384,006,168 + 384,016,682 | 单文件 ~101 GB |
| 词表 | 原始 token id | **压缩词表 99,092** | 需要 tokenizer 归一化管线 |
| 边界语义 | EOS 分段回填 | **DEAD token（图像 span）阻断 n-gram** | 新增 mask 语义 |
| FP8 缩放 | checkpoint 级标量 `weight_scale` | **每行 8 个 E8M0（32 维一块）** | 尺度向量需进记录 |
| 物理布局 | 列主序 `[160, 320M]`，16 路 scatter | **行主序 `[rows,256]`，天然紧凑** | 视图物化动机消失 |
| 短卷积 | 有（kernel 4） | **已删除** | gating 更简单 |
| 优化器 | Adam | momentum + Sinkhorn | 表可能以未归一化形式存在 |

---

## 2. 从报告读出的五个设计意图

1. **存储与带宽已经取代计算成为第一瓶颈。**【报告原文】Abstract 开篇即说
   "large KV caches continue to strain HBM and SSD capacity and data-transfer bandwidth…
   constitute the primary bottleneck to further lowering deployment costs"。
   → EngramDB 所在的战场是报告钦定的主战场。

2. **确定性寻址的价值被推到"批次级编译期"。**【报告原文】
   "prefetch is therefore initiated for the entire local batch before each pipeline stage begins processing microbatches"。
   不是逐 token 预取，而是**整个 local batch 的行集合一次性规划**。这是质变，不是量变。

3. **分层生命周期，而非单一缓存。**【报告原文】持久 KV 保证 ≥72h；SWA KV 放"每机 10% 宿主 DRAM"、
   TTL 只有分钟级；Engram 表不可变、无 TTL。三种寿命被**显式分开管理**，不共用一个 LRU。

4. **优雅降级优于完美恢复。**【报告原文】SWA Bounded Replay：宁可重算 `n_win` 个 token 的近似状态，
   也不做 `L × n_win` 的全量重算——"turns a catastrophic miss into a graceful, inexpensive degradation"。
   这是一种**可接受的近似**换取延迟可预测性。

5. **放置策略是负载相关的。**【报告原文】训练 = 按行分片 + 全 batch 预取；RL rollout = **常驻 GPU 显存**；
   推理 = 宿主内存 + RDMA 预取。同一个表，三种放置。

---

## 3. 新思路

### P0-1 记录格式升级为"热元数据 | 冷载荷"，并分离 scale 数组

**【实测依据】** checkpoint 里 payload 与 scale 是**两个独立张量**（`[rows,256]` 与 `[rows,8]`）。

**【建议】** EngramDB 的记录模型从"定长槽"升级为**字段化记录（fielded record）**，支持按字段指定放置层级：

- `scale` 字段：768M 行 × 8 B = **6.14 GB**，占 3.1% 字节，但**每次取行都要读**。
  → 完全放进 RAM / 甚至 pin 在显存，是"小到可以全量常驻"的热元数据。
- `payload` 字段：196.6 GB，真正的冷载荷。

收益：取一行从"两次磁盘/网络往返"变成"一次冷读 + 一次内存读"；
并且 scale 全量常驻后，可以做**免反量化的预算**（e8m0 是 2 的幂，反量化就是指数平移）。

**【验收判据】** 与现有 Store-P 单块布局 A/B：单记录 p50/p99 延迟、每 token 系统调用数、IOPS。

> 这条同时把 README 里"紧凑槽 vs 4KB pad"的结论推广成：**同一记录内不同字段可以有不同介质策略**。

---

### P0-2 结构性寻址：用 config 直接算出物理偏移，替代 DiskSlotIndex

**【推算】** V4.1 的 rowid 空间是 `Σ primes`，桶结构 `(layer, order, head, prime, offset)` 完全由
`config.json` 决定。checkpoint 又是行主序稠密数组。

**【建议】** 对 V4.1 走"**无索引直接寻址**"路径：

```
bucket(layer, order, head) → (prime, offset)
row_in_bucket = mix % prime
rowid         = offset + row_in_bucket
file_offset   = header_len + rowid * 264        # payload|scale 交错时
```

不需要 `DiskSlotIndex` 的 16384 桶 + offset table（v3 实测 build 135 s / `data.bin` 1.36 GB / 10M grams）。

**【反面提醒】** 这不否定 `DiskSlotIndex`：它对 **Qwen 的列主序布局**和**未知布局的第三方权重**仍然必需。
建议把它降级为"布局未知时的兼容路径"，而不是主路径。

**【验收判据】** 10M 规模下 build 时间 ≈ 0（无索引可建）；lookup 与 `DiskSlotIndex v3` 的
164.7 μs/lookup 对比；元数据占用从 1.36 GB 降到 0。

---

### P0-3 keygen v2：压缩词表 + 4-gram + DEAD mask

**【实测依据】** `inference/engram.py` 定义了 V4.1 的精确语义，与现有 `engramdb-keygen` 差异很大：

1. **压缩 token map 是必需的**，且 `build_compressed_token_map` 是纯 Python + HF `tokenizers` 的正常化序列
   （NFKC → NFD → StripAccents → Lowercase → 合并空白 → 哨兵 → Strip → 哨兵还原），
   含 `\ufffd` 坏字节 token 的 raw-key 回退。**乘子由压缩词表大小派生**——算错就整表重哈希。

   > **【强烈建议】不要在 Rust 里重实现 Unicode 归一化。**
   > 该映射是"固定 tokenizer → 固定 129,280 项查表"，**把它冻结成一个小产物**
   > （129,280 × u32 ≈ 517 KB，`token_map.bin`），构建期用 Python + `tokenizers` 生成一次。
   > 之后 keygen 就退化成和现有 Qwen 路径同型的**纯整数运算**，可做 golden 对拍。

2. **阶数泛化到 4**：`rolling = XOR-k over shift 0..3`，输出 3 列（2/3/4-gram），不是 2 列。

3. **DEAD token 语义（新增，且是多模态刚需）**：官方 `NgramHashState.forward` 里

   ```python
   blocked = blocked | (positions < shift) | (source == self.DEAD)
   ```

   `blocked` 是**粘性**的——一旦某个位移被阻挡，更大的位移也全部阻挡；
   被阻挡的槽位填 `pad_id`（**压缩后**的 pad id），n-gram 永不跨越图像 span。

4. **恒定 IOPS 仍然成立**：无论是否被阻挡，每个位置**总是**产出 3 阶 × 8 head = 24 个 id。
   这对存储层是好消息——**每 token 行数是常数，预取计划大小可静态预测**。

5. **乘子也是"派生但不可廉价重放"的。**【实测】乘子来自 `np.random.default_rng(10007 * layer_id)`
   的 `integers(low=0, high=bound, size=(4,), dtype=int64)`。要逐位复现，就得在 Rust 里
   **同时**复刻 PCG64 状态推进**和** numpy `integers` 的 Lemire 有界采样算法——
   这是纯粹的移植风险，没有任何收益。

   > **【强烈建议】把 48 个素数与 8 个乘子一并冻结为常量**（只有 56 个整数，见附录 A），
   > 由构建期脚本从 config 生成/校验。这与 Qwen 路径"一律读权重、不派生"的既有结论同源
   > （`engram-specs.md` §3.3：config 无 seed 字段 → 派生态不可复现，一律读权重）。

**【建议】** `engramdb-keygen` 从"Qwen 专用常量 + 一个 spec"重构为**通用 `EngramSpec`**：
`{layer_ids, max_ngram_size, n_heads, head_dim, vocab_size, pad_id, compressed_token_map, multipliers, primes}`，
Qwen 与 V4.1 各是一个实例。V4.1 的 48 个素数表与 8 个乘子见附录 A，直接作为 golden 固化
（已落到 `refs/v41_engram_constants.json`，可用 `scripts/gen_v41_engram_constants.py` 复算校验）。

**【验收判据】** 与官方 `engram.py` 逐 id 对拍（随机序列 + 图像 mask + 序列首部 + 跨 DEAD 边界），
零差异；golden 进 CI，与现有 `golden.json` 并列。

---

### P0-4 分片路由原语（`rowid → (rank, local_row)`）

**【实测】** 官方 `ParallelEngramEmbedding` 每 rank 只持有 `ceil(rows/mp)` 个**连续行**，
非本 rank 的 index 被 mask 成 0，最后 `all_reduce` 求和。

**【建议】** EngramDB 暴露一个纯函数路由层：

- `route(rowid) → (rank, local_row)`，纯算术，无查表；
- **per-rank 预取计划生成器**：输入全局 hash-id 张量，输出每个 rank 需要的那部分
  （filter → sort → dedupe → 合并成顺序 IO）；
- 分片边界跨桶的告警：实测 0/23 对齐，即**同一个 (layer, order, head) 桶会被分片边界切穿、横跨两个 rank**
  （桶约 16M 行 < 每 rank 约 48M 行，所以最多跨 2 个 rank）。路由与副本放置**不能假设桶对齐**。

**【价值】** 这是"存储层 → 服务层"的接口契约。上游引擎只要给 hash id，EngramDB 就能给出
每台机器该读什么、按什么顺序读。**这正是报告里"整个 local batch 一次预取"所缺的那块拼图。**

---

### P1-5 传输层扩展：从 `preadv` 到"远端内存/RDMA"

**【报告原文】** 预取是 "background RDMA transfers" from host memory。

**【建议】** 把 IO backend 从"文件 IO 后端"抽象成"**字节源（byte source）**"，
`preadv` / `io_uring` / `mmap` 只是实现之一，新增：

- `RemoteMemorySource`（RDMA READ / 或退化为 TCP + 注册内存）；
- `HostMemorySource`（共享内存 / memfd），对应"表在宿主内存"这一层；
- 统一 `ByteSource` trait：`read_into(offset, len, dst)` + `batch_read(plan)` + `latency_class()`。

**这与 README §2.4 的既有结论不冲突**：本地 NVMe + 8 线程下 io_uring 没有收益，所以那条结论
继续成立；但"网络盘/受限环境"这一条现在有了**具体的、被官方部署验证过的**目标场景。
可以明确写进 roadmap：**"T3.5 远端内存层"**。

**【验收判据】** 在 `ByteSource` 抽象下跑同一份 `view bench`；本地 `preadv` 路径性能回归 ≤2%；
新增一个 loopback-RDMA（或 shmem）smoke，证明预取计划可跨源复用。

---

### P1-6 按模块分配存储层级（layer 1 vs layer 14 的截止时间差 13 层）

**【推算】** V4.1 forward 的真实顺序是：

```python
engram_hashes = engram_hash(input_ids, start_pos)      # 全程最早，纯 token 函数
for i, layer in enumerate(layers):
    if layer.engram is not None:
        h = layer.engram(h, engram_hashes[..., i, :])   # i == 1 和 i == 14
    h = layer(...)
```

- **layer 1 的 engram**：在 `i=1` 的循环开头调用，此前只跑完 embed + layer 0，**几乎零余量**；
- **layer 14 的 engram**：在 `i=14` 开头调用，此前已跑完 embed + layer 0..13，
  **比 layer 1 多 13 层**的可重叠计算量。

而两者**大小相同**（各 98.3 GB）。

**【建议】** 引入**per-module 放置策略**：

| 模块 | 建议层级 | 理由 |
|---|---|---|
| layer 1 | NVMe / 宿主内存 / 甚至部分常驻显存 | 关键路径，只有 1 层余量 |
| layer 14 | 远端存储集群 / 较慢介质 | 比 layer 1 多 13 层余量，可吸收延迟 |

这是"用时间换空间"的精确版本：**同一份数据的不同模块，可以按截止时间放到不同介质**。
报告虽然没有这么说，但它是报告给出的结构（layer 1 + 14）的直接推论，而且是 EngramDB 独有的机会——
引擎内实现只有一种介质（显存/宿主内存），存储引擎才有资格做异构分层。

**【验收判据】** 构造 layer 1 在 NVMe、layer 14 在慢源（限速/远端）的双模块模拟，
测 end-to-end tok/s 与 wait 分布 p99，证明"慢源只影响非关键路径"。

---

### P1-7 推测解码感知预取（DSpark 把零余量变成 k 步余量）

**【报告原文】** DSpark 一次前向并行产出 **5 个 draft 位置**，用轻量 Markov head 建模依赖，
confidence head 预测接受概率。

**【问题】** 解码时 position t 的 hash id 需要 token t-3..t，只有采样出 t 才知道 → **层内前向余量为 0**。
这就是"聊天负载热集不存在、必须靠并行预取"之外的第二个根本困难：**解码期没有 lookahead**。

**【建议】** 用 draft token 把 lookahead 造出来：

- 对 DSpark 的 5 个 draft 位置，预先算出**候选 hash id 并集**（纯整数运算，纳秒级）；
- 表是**只读**的，多取几行的代价只是带宽，不是正确性；
- 在 40 层前向执行期间，把这些候选行提前拉进 T1/T2；
- 被接受的 draft 命中缓存，被拒绝的浪费一次带宽 —— 但**关键路径上的查询变成命中**。

**【为什么这是新思路】** 通用 KV 库没有"未来 token 的确定性地址"这个概念；
即便是引擎内实现，也普遍把 engram 查询当成同步操作。**把投机解码的候选集当作预取输入，
是只有"确定性寻址 + 预取计划"这一架构才成立的优化。**

**【验收判据】** 合成 DSpark 接受率（如 0.6/0.8）下测 `wait_distribution()` 的 p50/p99，
对比无投机预取；统计"预取命中率"与"浪费带宽比"。

---

### P1-8 生命周期分级缓存 + 宿主 DRAM 仲裁

**【报告原文】** 部署里有三种寿命完全不同的对象争抢宿主内存：

| 对象 | 介质 | 寿命 | 可变性 |
|---|---|---|---|
| global KV（持久 KV 缓存） | SSD / 宿主内存 | **保证 ≥72 h** | 可变 |
| SWA KV | **每机 10% 宿主 DRAM** | **分钟级 TTL** | 可变 |
| Engram 表 | 宿主内存 → RDMA | **永久** | **只读** |

**【建议】** EngramDB 的 T1 RAM 热集从"频率 + LRU"升级为**寿命分级（lifetime class）**：

- `Immortal + ReadOnly`（Engram）：一旦驻留就**永不按 LRU 淘汰**，只受预算约束；可跨进程共享；
- `TTL(72h)`（global KV）：LRU + 保底寿命；
- `TTL(minutes)`（SWA KV）：到点即回收，**高周转**。

并为不同 class 设**独立预算与回收策略**，而不是让它们在一个 LRU 里互相驱逐。
报告明确说 SWA KV 的"high turnover suffices to serve the vast majority of concurrent active sessions"——
**寿命分级的收益是容量利用率，不是命中率**。

**【验收判据】** 三 class 混合负载下，Engram 常驻集合不被 KV 冲刷（对比统一 LRU 的命中率曲线）。

---

### P2-9 有界降级服务（bounded degradation）

**【报告原文】** Bounded Replay 的哲学：用**有界的近似**换取"miss 不再灾难"。

**【建议】** 给 `PleMemory` / `DiskPleEmbedding` 增加一个**近似返回**模式：

- 预取超时或后端失败时，不再"要么全有要么报错"，而是返回**部分 e_t** + 一个 `completeness` 标记；
- 最自然的降级路径：**丢掉最高阶（4-gram）的 8 行**，用 2/3-gram 近似；
  因为模型是对各阶求和后再过 WKV，缺一阶是有界扰动而非崩溃；
- 上层可以选择接受（继续解码）或等待（保质量）。

**【诚实标注】** 这是**提案，未验证**。缺一阶对最终 logits 的影响必须实测。
但从报告对 Bounded Replay 的取舍看，这一设计方向与 DeepSeek 的工程哲学一致。

**【验收判据】** 在受控限速下，测"部分 e_t"与"完整 e_t"的输出 KL / 首 token 一致率 / 任务指标，
画出"降级幅度 vs 延迟"曲线。**没有这条曲线就不要上线这个开关。**

---

### P2-10 训练侧：批次级行去重与合并

**【报告原文】** 训练时"对**整个 local batch**在 pipeline stage 开始前发起预取"。

**【建议】** EngramDB 的训练路径（`fetch_e_t_tensor`）增加**批次级计划器**：

1. 收集整个 local batch（含所有 micro-batch）的全部 rowid；
2. **按 rowid 去重**（4-gram 会高度重复）；
3. 按物理偏移排序，合并成顺序读；
4. 回填 scatter。

README §4.2 已经证明"**全局**频率索引/热集"对训练无用（top-1000 覆盖 <6%，Zipf 不成立）。
但**批内去重**是另一回事：它不依赖全局分布，只依赖"同一批里出现相同 n-gram"。
45T token 的多模态语料 + 3 阶 n-gram，批内重复率值得实测。

**【验收判据】** 在真实语料上测 `unique(rows)/total(rows)`（批内去重率）与合并后的 syscall 数、
有效吞吐（目标 ≥100K tok/s，README 中的未闭环项）。

---

### P2-11 放置 profile（train / rollout / serve）

**【报告原文】** 三种负载，三种放置：训练 = 分片 + 全批预取；**RL rollout = 常驻显存**；
推理 = 宿主内存 + RDMA。

**【建议】** 把"放置"提升为一等配置对象，例如 `PlacementProfile`：

```python
PlacementProfile.train()    # 分片并行 + batch 预取 + 无 T1 热集假设
PlacementProfile.rollout()  # 表常驻最快层，牺牲容量换零延迟
PlacementProfile.serve()    # 宿主内存 + RDMA + per-module 分层 + 投机预取
```

这与 README 的"负载 A / 负载 B"是一致的，只是多了 **rollout 这一档**，
并且明确 rollout 的取舍是"**占用最快介质以避免宿主内存碎片**"——这是 OOM 规避，不是性能优化。

---

### P3-12 战略扩张：Engram 表 + 持久 KV 缓存统一存储面

**【报告原文】** V4.1 的头号卖点是 KV cache 压缩：global KV 890 B/token，持久 KV 降到 1/8，
**且持久 KV "always on SSD or in host memory"**，配 72h LRU 保底。

**【观察】** 持久 KV 缓存的工程需求与 Engram 表**高度同构**：

- 都是"确定性的、可分页对齐的、批量预取的、磁盘/宿主内存优先的"大对象存储；
- 都需要 page 对齐、批量预取、多级缓存、淘汰策略；
- 区别只在**键的来源**（prefix hash vs n-gram hash）与**可变性**（可写 vs 只读）。

**【建议】** 把 EngramDB 的定位从"Engram/PLE 专用引擎"扩展为
"**条件记忆 + 持久 KV 的统一磁盘/宿主内存层**"，共享：

- `ByteSource` 抽象与 RDMA 传输；
- 预取计划器与批量合并器；
- 页对齐布局与 badge 聚簇；
- **寿命分级缓存与宿主 DRAM 仲裁**（P1-8）——这正是两者会互相争抢的资源。

**【收益】** 报告给出的部署画像是"2,000 GPU + 存储集群"，其中**KV 的容量需求远大于 Engram**
（长上下文 + 多会话 + 72h 保留）。只做 Engram 会把市场限制在 196 GB/模型；
把 KV 也纳入，才是报告描述的整个存储集群。

**【风险】** 这会稀释"不做通用 KV 数据库"的项目定位（README 开篇的消歧声明明确写了不做通用 KV）。
**建议保持边界**：只做"**确定性键 + 大对象 + 预取计划**"的 KV（即 prefix-cache 形态），
不做任意 KV/事务/ANN。是否推进需要一次单独的战略讨论。

---

## 4. 立即可执行的探针

| # | 探针 | 对应思路 | 方法 | 判据 |
|---|---|---|---|---|
| 1 | 冻结 token map | P0-3 | 用 `tokenizers` 生成 129,280 项映射，校验 `len(set) == 99092` | 与官方 assert 一致 |
| 2 | keygen v2 对拍 | P0-3 | Rust 复刻 vs 官方 `engram.py`，随机序列 + 图像 mask + 边界 | **零差异** |
| 3 | 无索引寻址 | P0-2 | 直接 `header + rowid*264` 读，与 `DiskSlotIndex v3` A/B | lookup 延迟、元数据 0 vs 1.36 GB |
| 4 | scale/payload 分离 | P0-1 | scale 数组全量常驻 vs 同块读取 | p50/p99、syscall 数 |
| 5 | 分片路由 | P0-4 | `route(rowid)` + per-rank 计划生成 | 计划覆盖率 100%，无重复读 |
| 6 | 批内去重率 | P2-10 | 真实语料 batch 的 `unique/total` | 给出曲线，判断是否值得做 |
| 7 | 降级质量曲线 | P2-9 | 丢 4-gram 的 e_t vs 完整 e_t | KL / 一致率曲线 |
| 8 | 双模块异构分层 | P1-6 | layer 1 快源 + layer 14 慢源 | end-to-end tok/s 不退化 |

前 3 条应在下一个 session 内完成；第 2 条是其余一切的前提。

---

## 5. 不建议投入 / 风险

1. **不要为 V4.1 建全量 Store-P 视图。** checkpoint 已是行主序紧凑布局（P0-2 实测），
   再物化一份只会浪费 196.6 GB。Store-P 应该退化为**可选缓存**（只物化观测到的高频 key 组合），
   或直接删除。这与 README §4.2 第 5 条"盲目全量物化"的结论一致，但理由更强了。

2. **不要在 Rust 里重实现 Unicode 归一化。** NFKC/NFD/StripAccents 的 Unicode 表版本差异
   会导致压缩词表大小不等于 99092 → 乘子全变 → 整表重哈希。**冻结查表**是唯一稳妥方案。

3. **不要假设 chat 负载有热集。** README 已证否（top-1000 <6%），V4.1 的 16M 桶 × 48 行只会更均匀。
   但**多模态的图像 token 是例外**（见下条）。

4. **不要忽略"图像 token 完全不需要 Engram"。**【官方代码】
   `gate = gate.masked_fill(~token_mask...)`，图像位置 gate 恒为 0。
   → 视觉密集的 agent 负载里，**预取计划可以直接跳过这些位置**，
   IOPS 降幅正比于图像 token 占比。这是纯计划层面的收益，值得做成显式优化。
   另外，图像边界之后的 `(max_ngram_size-1)` 个位置使用 pad 填充模板，
   **是有限的小集合，值得永久 pin 住**。

5. **警惕"rank 分片 = 整桶"的直觉。** 实测 0/23 对齐（P0-2），边界会切穿桶。
   路由必须用 `rowid // part_num_embeddings`，不能假设桶对齐。

6. **报告没有给出 Engram 的推理性能数字。** §2.4.2 只说"prefetch 与第一个 block 重叠"，
   没有 tok/s、没有惩罚百分比（对比原始 Engram 论文 §6.4 的 2.8% 峰值惩罚）。
   **不要把"报告没说"当成"没有问题"** —— 我们自己测。

---

## 附录 A：V4.1 Engram 常量（可作为 golden）

**压缩词表大小**：99092　**乘子上界**：46539438283891　**pad_id（压缩后）**：`token_map[2]`

**乘子**（`default_rng(10007*layer_id)`，`v*2+1`，索引 0..3 对应 shift 0..3）：

```
layer  1: [76632096046245, 4839876093313, 35959672319349, 73987337458391]
layer 14: [67716810739261, 51510806800915, 30921347202721, 82619226485591]
```

**素数表**（顺序分配，全局不重复；`search_start = 16000000 - 1`）：

| layer | order | head0 | head1 | head2 | head3 | head4 | head5 | head6 | head7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 16000057 | 16000079 | 16000081 | 16000097 | 16000121 | 16000129 | 16000133 | 16000183 |
| 1 | 3 | 16000189 | 16000207 | 16000211 | 16000253 | 16000277 | 16000289 | 16000307 | 16000321 |
| 1 | 4 | 16000339 | 16000381 | 16000393 | 16000399 | 16000403 | 16000409 | 16000447 | 16000463 |
| 14 | 2 | 16000477 | 16000487 | 16000499 | 16000507 | 16000511 | 16000573 | 16000609 | 16000627 |
| 14 | 3 | 16000667 | 16000669 | 16000693 | 16000697 | 16000711 | 16000729 | 16000759 | 16000769 |
| 14 | 4 | 16000781 | 16000799 | 16000813 | 16000819 | 16000841 | 16000877 | 16000879 | 16000889 |

行数闭合：Σ(layer 1) = 384,006,168；Σ(layer 14) = 384,016,682。

---

## 附录 B：证据来源

- 技术报告：`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf`
  （本次经 `https://hf-mirror.com/...` 获取，51 页，1.8 MB）
- 官方推理代码：`inference/engram.py`、`inference/model.py`、`inference/convert.py`、`inference/README.md`、`config.json`
- 权重张量形状：`model-00047-of-00048.safetensors` HTTP Range 读取 header
- 发布公告：<https://api-docs.deepseek.com/news/news260910/>
- 原文关键句（§2.4.2）：
  "196B Engram parameters evenly across two modules… N-gram orders {2, 3, 4}, with 8 hash heads
  and a total embedding dimension of 2048 per order… approximately 16M entries… distinct primes…
  Both the embedding tables and the key/value projections use FP8… placed at layers 1 and 14…
  prefetched from host memory via background RDMA transfers"
- 原文关键句（§3.1.3）：
  "Engram lookup indices depend solely on the input token sequence. Embedding prefetch is therefore
  initiated for the entire local batch before each pipeline stage begins processing microbatches…
  Embeddings are stored and fetched in FP8, with the retrieved values and scaling factors passed
  directly to the following GEMM… During RL rollouts, Engram embedding tables remain resident in
  GPU memory."
