# Session 42 · rowid 逐位一致性：EngramDB keygen vs vLLM 自己的 PLE 代码

> 口径：**调用引擎自己的函数体**(未修改),对照 EngramDB 生产 Rust 路径。
> 产物：`scripts/ple_rowid_exactness.py`、`probes/ple_rowid_exactness_session42.json`。
> 纯 CPU,秒级,可复跑。

## 0. 为什么这个测试比任何吞吐数字都重要

这一轮 serving A/B 测的都是**代价** —— 读 N 行要多久。代价在「读错行」面前毫无意义:
一个 rowid 错了就静默污染每一个 token,而任何 tok/s 数字都不会显示出来。

权威算法在**引擎里**,不在我们仓库里。所以判据只能是:拿引擎自己的代码当裁判。

## 1. 方法【实测】

`vllm/models/qwen4_exp/nvidia/ple_layer.py` → `Qwen4ExpNGramEmbedding.compute_ngram_ids`。

该方法是普通方法,它对 `self` 的全部依赖只有 8 个属性。因此传入一个
`SimpleNamespace` 即可**原封不动**执行真实函数体 —— 不需要构造 GPU 模块、
TP group 或 quant method:

```python
shim = SimpleNamespace(
    ngram_size=3, heads_per_ngram=8, ngram_heads=16, eos_token_id=248044,
    positions_buffer=…, padded_buffer=…,
    ngram_heads_vocab_sizes=…, ngram_heads_offsets=…, layer_multipliers=…,
    _shift_precompute=PLE._shift_precompute, _shift_apply=PLE._shift_apply,
)
ids = Qwen4ExpNGramEmbedding.compute_ngram_ids(shim, input_ids, query_start_loc, ngram_context)
```

multiplier / 素数表 / offset 全部用**引擎自己的 classmethod** 现算
(`_make_layer_multipliers`、`_make_vocab_layout`),不是从我们 spec 里抄的 ——
否则就是自己考自己。

对照对象:`engramdb._engramdb.rowids_for_seq(tokens, PLE_QWEN_V1)`,即生产 Rust。

## 2. 结果【实测】VERDICT: **IDENTICAL**(37 用例,0 失败,1128 行 × 16 头 = 18,048 个 rowid)

```
engine  multipliers: [23703573157769, 20109073645365, 8052911324071]
engramdb multipliers: [23703573157769, 20109073645365, 8052911324071]   ← 独立推出，非拷贝
multipliers identical: True
```

| 组 | 覆盖 | 结果 |
|---|---|---|
| **A** 冷启动单请求 | n = 1, 2, 3, 4, 8, 63, 64, 257 | 8/8 OK |
| **B** 含 eos 的序列(段边界) | n=6/16/64 × 1–3 个 eos;eos 在首位/末位 | 11/11 OK |
| **C** 多请求 packed batch(引擎真实布局) | `[3,5]` `[1,1,1]` `[17,4,9,2]` `[2,2]` | 4/4 OK |
| **D** decode 步(2-token history) | new = 1, 2, 4;history 含 eos | 4/4 OK |
| **E** chunked prefill | cut = 1, 2, 5, 16, 63,各查两遍(对照引擎 + 自洽) | 10/10 OK |
| **F** 规格几何 | `padded_vocab`、前三个素数 | OK |

### 2.1 两个几何量对上了

```
engine padded_vocab = 320,001,536
on-disk table rows  = 320,001,536     ← 128 shard × 2,500,012（manifest 实测）
head_sizes[:3] = [20000003, 20000023, 20000033]
```

第一行不平凡:vLLM 把总行数向上取整到 `make_ngram_vocab_size_divisible_by=128` 的倍数,
而这个**取整后的数**必须等于磁盘上真实表的总行数。若不等,最后一个 shard 就短了,
所有高位 rowid 会读到表尾之外。它们相等。

### 2.2 §E 抓到的是**测试的** bug,不是库的 bug —— 但值得记

第一版 §E 我写的是 `toks[cut-2:cut]`。`cut=1` 时这是 `toks[-1:1]` ——
Python 负索引,结果是**空列表**。于是历史长度为 0,位置 1 起 16 个头全错。

真实语义是:**chunk 短于 `ngram_size-1=2` 时,上下文要用 eos 左填充**
(`[eos] * width + prefix)[-width:]`)。vLLM 侧 `ngram_context` 恒为 2 宽,
所以短请求就是 `[eos, t0]`。

这个边界在真实 serving 里是可达的:prompt 被 chunk 切在很靠前的位置时就会走到。
修好后 cut=1 / cut=2 都过。**留在这里是因为它是「边界要么测要么不知道」的实例。**

## 3. 结论

**EngramDB 的 rowid 推导与 vLLM 自己的 PLE 代码路径逐位相同**,可以插在
`ple_layer.py:456` 那一行上:

```python
# vllm/models/qwen4_exp/nvidia/ple_layer.py
ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)  # ← 与 EngramDB 一致
return self.ngram_embedding(ngram_ids).flatten(-2)                            # ← 唯一需要改的行
```

注意这一行两侧的分工:`compute_ngram_ids` **已经**在注释里被移出 CUDA graph
(「Keep num_reqs-dependent ID generation outside PIECEWISE CUDA graphs」),
而 `ngram_embedding(...)` 是一次 `F.embedding(ids, weight)`。也就是说引擎自己
已经把「算 rowid」和「取行」分开了 —— 这正是 prefetch 需要的切口。

`ngram_embedding` 的类型是 `PLEVocabParallelEmbedding`(继承 `VocabParallelEmbedding`),
即 **GPU 常驻**。§4 说明为什么 offload 配置救不了它。

## 4. vLLM 0.29.0 的 offload 为什么覆盖不了这张表【实测】

`vllm/config/offload.py` 提供两条路,都是**整张量 / 整层**粒度:

| 配置 | 粒度 | 语义 |
|---|---|---|
| `cpu_offload_gb` + `cpu_offload_params` | 参数**名段** + GiB 预算(UVA 零拷贝) | 按名字段凑够预算就整张量搬到 CPU,每次 forward 零拷贝读 |
| `offload_group_size` + `offload_num_in_group` | **decoder layer 分组** | 「每 N 层里 offload 最后 M 层」,异步 H2D 预取整层 |

PLE 表是 **layer 2 里的单个 51 GB 张量**。所以:

- 按层 offload ⇒ 每次 forward 搬 **51 GB**;
- 按名段 offload ⇒ 整张量搬,且要 51 GB pinned host 内存;

两条路都**没有「按 rowid 取 16 行 = 2,560 B」这个粒度**。这不是配置能解决的问题,
是粒度问题。EngramDB 补的正是这一格。

## 5. 边界(必须声明)

- 本测试证明 **rowid 正确**,不证明吞吐。吞吐见
  `probes/serve_ple_ab_session42.md`(合成投影)与
  `probes/serve_faithful_embed_ab_session42.md`(真实权重,1 行/token)。
- 对照的是 **vLLM 0.29.0** 的实现。SGLang 0.5.19 **没有** `qwen4_exp`
  支持(registry 无条目、无 `ple_layer_ids` 代码路径),因此**无从对照** ——
  这不是我们库的兼容性问题,是 SGLang 还没有这个缝。见 README §6.1 子条件 6。
- 测试用例是随机 token(种子 0),不是真实语料。rowid 是纯函数,
  覆盖度由「长度 + eos 位置 + 批次布局 + chunk 边界」四个维度保证。
