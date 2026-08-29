# Engram / PLE 结构规格与证据链

> 本文是 EngramDB 设计的"规格基线"。所有实现必须与本文描述的结构一致（位级/数值级一致验证），
> 一致性检查由 `engramdb-keygen` 与 Python 参考实现对拍完成（M0-P0b）。
> 资讯截至 2026-08-29，按当时公开资料整理。

---

## 1. 背景与文献

### 1.1 DeepSeek Engram（论文）

- **论文**：*Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*，
  Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, et al. DeepSeek-AI + Peking University.
  arXiv: 2601.07372 (2026-01-12；v2 2026-07-12)。
- **仓库**：https://github.com/deepseek-ai/Engram （Apache-2.0；仅 demo 实现 `engram_demo_v1.py`，mock 掉 Attention/MoE/mHC）。
- **核心思想**：
  - Transformers 缺乏"知识查找"原语，只能靠计算模拟检索（例如解析 "Diana, Princess of Wales" 需要多层 Attention+FFN 组合特征）。
  - 引入 *conditional memory* 作为与 MoE (*conditional computation*) 互补的稀疏轴：静态 N-gram 嵌入表 + 确定性哈希 O(1) 查找。
  - 通过 *Sparsity Allocation* 问题发现 U 形标定律：最优分配在 ρ≈75%~80%（MoE 占稀疏预算），
    把约 20%~25% 分给 Engram。
- **关键结果**：Engram-27B（26.7B 总参，5.7B 嵌入参数；MoE 专家 72→55）在 iso-参数/iso-FLOPs 下全面优于 MoE-27B；
  长上下文（Multi-Query NIAH 84.2→97.0）；LongPPL 显著下降。
- **系统效率（论文 §2.5, §6.4）**：
  - 训练：嵌入表按 GPU 分片，All2All 收集激活行（前向）/分发梯度（反向）。
  - 推理：确定性寻址 → **主机内存异步预取 + 与前层计算重叠**；实测 100B 参数表 offload 到 host DRAM，
    吞吐惩罚峰值仅 **2.8%**（Dense-8B + nano-vLLM harness）。
  - 明确划分三级：GPU HBM → Host DRAM → **NVMe SSD**（长尾冷数据）；
    依据 N-gram 的 Zipf 分布（语料内成立；推理聊天负载近似均匀，见 §3.4 实测）。

### 1.2 Qwen3.8-Flash-Next 的 PLE（生产化）

- **发布**：2026-08-26，Alibaba Qwen 团队，开源权重（qwen-community-1.0），Qwen4 架构预览。
- **repo**：`QwenLM/Qwen3.8-Flash-Next`（仅 README + tech_report.pdf）；建模代码在 transformers/各类引擎内（`model_type: qwen4_exp`）。
- **模型构成**（与 EngramDB 相关的抽出）：
  - 125B 主干（MoE 512 专家 top-10 + 1 shared；6B 激活/token）+ 51B **N-gram Embedding（PLE）表** + 4B MTP 头。
  - 48 层 = 12 × [3 × (GatedDeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE)]；Gated Residual 4 流（hc_count=4, lowrank=320）。
  - **PLE 位于 decoder layer 2（`ple_layer_ids=[2]`，zero-based 索引 1）**，全模型唯一一处。
    该层先注入 PLE delta，再进入常规 GDN→MoE。
- 官方定位：PLE 是"无附加计算成本扩大容量的另一根轴"，表放宿主内存、异步预取，不占 GPU 显存。

---

## 2. DeepSeek Engram demo 精确结构（代码级解析，`engram_demo_v1.py`）

### 2.1 配置（demo 默认）

| 参数 | 值 |
|---|---|
| `tokenizer` | `deepseek-ai/DeepSeek-V3`（vocab 129,280） |
| `engram_vocab_size` | `[129280*5, 129280*5]`（2-gram、3-gram 各自容量基准） |
| `max_ngram_size` | 3 |
| `n_embed_per_ngram` | 512（每阶总维度） |
| `n_head_per_ngram` | 8 |
| `layer_ids` | `[1, 15]` |
| `pad_id` | 2 |
| `seed` | 0 |
| `kernel_size` | 4（conv） |
| backbone | hidden 1024, hc_mult 4, num_layers 30, vocab 129280 |

### 2.2 压缩分词（CompressedTokenizer）

- normalizer 序列：NFKC → NFD → StripAccents → Lowercase →
  把 `[ \t\r\n]+` 替换为空格；`^ $`（仅空格）替换为哨兵 `\uE000` → Strip → 哨兵 → 空格。
- 对每个 token id：decode 后规范化；若文本含 `�`（坏字/特殊状态），**使用 raw token 字符串作为 key**。
- 输出：`lookup_table`（旧 id→新 id，int64；长度=原词表）+ 新词表大小；有效压缩率约 23%（对 128k tokenizer）。
- `pad_id` 也做压缩映射，后续哈希移位用压缩后的 pad id。

### 2.3 素数（calculate_vocab_size_across_layers）

- 每 (layer, n-gram 阶) 有 `n_head_per_ngram` 个头，每个头的模 = 素数。
- 起始搜索点：`vocab_size_per_ngram[n-2] - 1`，然后向大找**下一个未用过的素数**（`isprime`，sympy）；
  全局 `seen_primes` 跨层共享（即跨层跨头不重复使用素数）。
- demo 规模估算：每阶 8 个素数 ≈ 每种约 646,401~646,417，16 头/层（2 阶 × 8 头），每层 sum ≈ 10.3M 行。

### 2.4 哈希（NgramHashMapping）

- 层独立乘子：`base_seed = seed + PRIME_1 * layer_id`（PRIME_1=10007）；
  `np.random.default_rng(base_seed).integers(0, half_bound, size=(max_ngram_size,), dtype=int64)`，
  half_bound = (i64::MAX // tokenizer_vocab_size) // 2 左右；**乘子 = r*2+1（奇数）**。
- 后缀 N-gram：输入 `[B,T]` 压缩 id；对 k=0..max_ngram_size-1 左移 pad（左侧 pad_id）。
- 混和：`mix = (t0 * m0); for k>0: mix ^= (t_k * m_k)`（按每阶取前 n 个 token）。
- 头哈希：`hash_head = mix % prime_{(n-2, k)}`，输出 `[B,T,total_heads]`（total_heads=(N-1)*K）。
- 补充注意：demo 的 n-gram 表大小按素数尺度（每头独立表）；**每头一个独立素数模 = 每头表的行数 ≈ 素数大小**。

### 2.5 嵌入与融合

- 单张统一 `nn.Embedding(sum(primes), D)`，`D = n_embed_per_ngram // n_head_per_ngram = 64`（demo）。
- 每头查得 `[B,T,k,d]`，按 head 偏移相加（offsets = primes 前缀和）取行，然后 `flatten(-2)` → `e_t [B,T,(N-1)*K*D]`（demo: 2*8*64=1024 维）。
- Gating（ContextAwareGating，每 hc 分支）：
  1. `v_t = W_V e_t`（共享）；`k_t = W_K^{(m)} e_t`（每分支）。
  2. RMSNorm(h)、RMSNorm(k)；`gate = (h_norm · k_norm)/sqrt(H)`。
  3. 稳定性：`gate = sign(gate) * sqrt(abs(gate).clamp_min(1e-6))` → sigmoid → `α_t∈(0,1)`。
  4. `ṽ = α * (W_V e_t)`（每分支不同 α，共享 W_V）。
  5. ShortConv：depthwise causal Conv1d（kernel=4, dilation=max_ngram_size，weights 初始化 = 0 → 训练起点是恒等映射）。
  6. `Y = SiLU(Conv(RMSNorm(ṽ))) + ṽ`；残差 `h ← h + Y`。
- 论文版本（Engram-27B 实测配置）：层 2、15；N=3；heads=8；嵌入维度 1280；嵌入参数 5.7B。

### 2.6 结构等价性结论（EngramDB 依赖的部分）

> **"查询面" = (层, n, head, token 元组) → 确定性 rowid → 固定宽度行向量**。
> 哈希算法（压缩、移位、乘子、素数、偏移）全部是确定性纯函数，与数值无关；
> 因此 EngramDB 只负责"rowid → 行数据"的存取，把 key 生成留在上层（`engramdb-keygen` 提供参考实现）。

---

## 3. Qwen3.8-Flash-Next PLE 精确规格（实测口径汇总）

### 3.1 config.json 直读（ModelScope API，2026-08-29）

文本配置（`text_config`）关键字段：

| 键 | 值 | 含义 |
|---|---|---|
| `model_type` | `qwen4_exp` | 架构名（transformers v5.8.0.dev0） |
| `ngram_size` | 3 | 2-gram + 3-gram |
| `ngram_vocab_size_base` | 20,000,000 | 每维度槽位基准 |
| `heads_per_ngram` | 8 | bigram 8 头 + trigram 8 头 = 16 头 |
| `ple_embed_dim` | 2560 | e_t 总维度 = 16 × 160 |
| `ple_layer_ids` | `[2]` | 注入层（decoder 第 2 层，零基 1） |
| `split_ngram_parts` | 128 | checkpoint 物理分片数 |
| `make_ngram_vocab_size_divisible_by` | 128 | 行数对齐粒度 |
| `ple_conv_kernel_size` | 4 | gating 后卷积核宽 |
| `hidden_size` | 2560 | 与 ple_embed_dim 相同 |
| `hc_count` / `hc_lowrank` | 4 / 320 | Gated Residual |
| `vocab_size` | 248320 | 模型词表 |
| `max_position_embeddings` | 262144 | 原生上下文 |

### 3.2 物理元组（NeMo AutoModel 文档 + GGUF 元数据分析）

- 张量名：`per_layer_token_embd.weight`，形状 **`[160, 320,001,536]`**（注意：列主序，宽 160 在首维）。
  = 51.2B 参数（`320,001,536 × 160`）。
- 16 个头（8 bigram + 8 trigram）**垂直堆叠**：每个头占一个连续的 160-行表段；
  行总数 = 16 × 每头词表（每头 ≈ 20,000,00x 素数，向上取整到 128 的倍数后加总 = 320,001,536）。
- 每头词表 = **20,000,000 以上的下一个素数**（GGUF 元数据实测：20000003, 20000023, 20000033, ...），
  并受 `make_ngram_vocab_size_divisible_by=128` 影响（pad 到 128 的倍数）。
- 每个 token：**16 行**（16 头各一行，行宽 160 维）。e_t = 拼接 → 2560 维（=ple_embed_dim=hidden_size）。
  每 token payload ≈ **5KB（BF16）/ 2.5KB（FP8 / 量化约 0.9-2.7KB）**。
- 128 个 checkpoint 分片（`split_ngram_parts=128`）；BE16 全表 ≈ 95.4 GiB，FP8 全表 ≈ 47.7 GiB。

### 3.3 哈希算法（官方实现： transformers `qwen4_exp/modeling_qwen4_exp.py`，2026-08-29 已提取到 refs/）

**P0 已经代码级闭合并数值验证**（与真实权重互证）：

- **乘子**：运行期以 checkpoint 存储的 `layer_multipliers`（I64[3]，`[23703573157769, 20109073645365, 8052911324071]`）为准；
  `_build_layer_multipliers`（splitmix64 派生）仅用于 fresh-init。config 无 `seed` 字段（校验 seed=0 不匹配）→ 派生态不可复现，**一律读权重**。
- **头素数**：`size_i = nth_prime(20_000_000-1, i+1)`，i=0..15 → `[20000003, 20000023, 20000033, 20000047, ...]`（与 GGUF 元数据一致）。
- **头偏移**：素数前缀和；`Σ = 320,001,446` → `padded = ceil(Σ/128)*128 = 320,001,536` = **checkpoint 总行数（128 分片 × 2,500,012）完美闭合**。
- **EOS**：`248044`；每位置段语义 `_shift_right_ignore_eos`（前文 EOS×2 冷启动；段内不足 shift 回填 EOS）。
- **混和**：`mix = ⊕_{k=0..n-1} (shifted_k × mult_k)`（i64 回绕语义 == u64 wrapping_mul），`ngram_ids[n,k] = mix % prime + offset`；16 行按 [8×bigram ‖ 8×trigram] 拼接。
- **rowid 空间**：ngram_ids ⊆ [0, 320,001,023]；表保存 320,001,536 行（尾部 512 行闲置）。
- **对拍**：`scripts/ref_ple_hash.py`（numpy 复刻）+ `scripts/gen_golden.py` → `crates/engramdb-keygen/tests/golden.json`；Rust 侧测试 `matches_python_golden` 全绿（含超词表 999999 的溢出回绕用例）。

- 主机侧计算（GGML/I32 图输入）：对每个 token，用**其前面 n-gram_size-1 个前置 token**（原 token id，**无词表压缩**）做混和：
  `mix = XOR-k over (tok_k × mult_k)`；**乘子来自 splitmix64 派生序列，量级 ~2^13~2^45（64 位整数）**；
  与 DeepSeek demo 结构同型（乘法-XOR 混和 + 素数取模），**但乘子/种子体系不同**。
- 行索引 `row = mix % head_vocab_size`（每头不同素数模；跨头碰撞几乎无关）。
- **序列边界**：位置不足以取全前置 token（如 EOS/换段）时用 EOS 回填并在分段处重置 —— 有严格的"segment reset"语义；
  llama.cpp 只在上下文连续时信任缓存，否则回退 EOS padding。**EngramDB 的 keygen 必须实现该语义**。
- 表值本身不做任何再编码；gating 侧的 conv（kernel=4）经深度卷积写成"移位缩放副本和"。

### 3.4 社区实测数据（设计基线，来源见 §5）

| 数据点 | 数值 | 意义 |
|---|---|---|
| llama.cpp M1 Max 64GB + 表在 SSD（全模型 mmap） | **18 tok/s**（大量分片）+ 表单精 26.8GiB IQ4_NL | 被动 mmap+fault 路径的天花板参考 |
| llamacpp 4.75M 次 gather 实测 | **0 次相邻 gather 命中同一 4KB 页**（16 区域相距约 20M 行） | 行级零局部性 → 顺序预读无效；页缓存仅服务约 4% | 
| 聊天会话实验（DGX Spark，0xBakeer） | 会话后表仅 **1.3% 驻留**，decode **13.1 major faults/token** | **推理热集不存在** ←设计反证 Zipf 不适用于 chat |
| `warm_table.py` 顺序预热 26.8GiB | ~26s | 全量预热在 NVMe 上可行但昂贵 |
| SGLang pinned+UVA（H200, TP4） | 权重 -23.5GB/卡, KV +78.5%, 吞吐 **-0.07%**, 输出逐位一致 | "GPU 服务 + 表驻 RAM"的黄金参考：吞吐几乎免费 |
| vLLM disk-backed（RTX PRO 6000, cgroup 48GB） | 79 vs 84 tok/s（**-8%**）；无 cgroup 时页缓存被 checkpoint 冲走 → 50-62 | "表在 NVMe"的现实税 ~8%，可优化 |
| vLLM mmap + READAHEAD | 冷 gather 81→4.1ms（96 行）；790→21ms（4096 行）；热 0.5~0.6µs/run | 批量预读是灵丹：应作为默认路径，禁懒惰 |
| SGLang NVMe Rust（#36567） | io_uring + 页对齐 + 有界批 + GIL-free；**踩到 2MB folio (hugepage) 分配问题** → RSS 增长 | 布局必须 2MB 对齐/显式 madvise，字节放大要压 |
| NeMo 训练 | DTensor Shard(0) + All2All + conv halo(9 token) | 训练侧仅作背景，EngramDB 不实现 |

---

## 4. 两套结构的共性（EngramDB 的设计依据）

1. **确定性寻址**：rowid 全部由 token 序列在计算开始前决定 →"预取计划"可在 step 0 产生——这是
   所有通用 KV 库（RocksDB/LMDB/DuckDB）都不会利用的自由。
2. **每 token 常数 IOPS**：16 行（Qwen 口径）或 32 行（Engram-27B 口径）→ 每 token payload
   只有 KB 级，与表总大小无关（对比 MoE：路由依赖 hidden state，无法预取）。
3. **行宽 160~3200 维**：point-read 粒度 320B~5KB → 4KB 页 = 12~25 行（BF16/FP8）→
   badge 聚集把 IOPS 再压 1~2 个量级是本项目核心。
4. **访问分布双形态**：
   - 训练语料：N-gram 呈 Zipf（常见短语重复命中）→ 频率索引/热集成立；
   - 聊天/评估负载：近似均匀 → 热集无意义，必须依赖"并行批量预取 + 时序重叠"而非 LRU。
   EngramDB 必须同时服务两种形态（这也是我们与引擎内实现的最大区别）。

---

## 5. 证据来源索引

- 论文：https://arxiv.org/abs/2601.07372 （v2 html: `https://arxiv.org/html/2601.07372v2`）
- DeepSeek demo：https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
- Qwen repo：https://github.com/QwenLM/Qwen3.8-Flash-Next
- 官方权重 config.json：ModelScope `Qwen/Qwen3.8-Flash-Next`（`/api/v1/models/.../repo?FilePath=config.json`，已验证可单文件获取）
- HF 模型卡：https://huggingface.co/Qwen/Qwen3.8-Flash-Next / `-FP8`
- NeMo 文档：https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/qwen/qwen3-8-flash-next
- vLLM recipe：https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- vLLM PR #53899（PLE CPU offload）、#54070（disk-backed）、#54129（mmap）、issue #53908（aux GPU）、#53960（TP1 deadlock）
- SGLang 博客：https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/ ；PR #36497（模型支持）、#36567（NVMe Rust reader）、issue #36514（aux GPU）
- llama.cpp PR #27742（qwen4exp 支持 + PLE 哈希/毫米级实现）
- 现场实测：https://lilting.ch/en/articles/qwen38-flash-next-llamacpp-m1max-test ；`0xBakeer/qwen38-flash-next-spark` how-it-works
