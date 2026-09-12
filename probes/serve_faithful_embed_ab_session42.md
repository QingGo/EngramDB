# Session 42 · 忠实注入 serving A/B：模型自己的权重表，从磁盘读

> 口径：**真引擎、真模型、真权重、真 GPU**。语义栏可引用,性能栏**无效**(见 §2)。
> 产物：`scripts/serve_faithful_embed_ab.py`、`probes/serve_faithful_embed_ab_session42.json`。
> 运行：守候脚本 20:30:27 自动触发(exit=0),`/root/engram-serve/faithful-ab.json`。

## 1. 与合成注入的区别

`probes/serve_ple_ab_session42.md` 注入的是**随机投影**,输出乱码,只能测代价。
本次把 `embed_tokens` 整体换成 `DiskEmbedding` —— 内容是该模型**自己训练出来的**
权重(已证逐位往返,`probes/ple_disk_faithfulness_session42.json`,sha256 `608f0ef5…`)。

一次运行同时出两半:

| 栏 | 判据 |
|---|---|
| **语义** | 两臂 greedy token id 必须**逐位相同** |
| 性能 | 两臂 tok/s |

第二栏只有在第一栏成立时才可引用。

## 2. 结果【实测】

```
model      /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B
engine     vLLM 0.29.0, enforce_eager=True
table      model.language_model.embed_tokens.weight  248320 × 1024 bf16
           width 2048 B, 2 shards
rows/token 1        ← 注意：不是 PLE 的 16 行
```

| 臂 | tok/s |
|---|---|
| baseline | 44.82 / 46.53 / 46.59 |
| disk-embed | 47.32 / 47.56 / 47.11 |

```
greedy_token_ids_identical : true          ← 本次唯一的强结论
delta_pct                  : +1.71%
added_us_per_token         : -360.8 µs     ← 负值，物理上不可能
disk_reader                : 192 calls, 114.2 µs/call
```

### 2.1 语义栏：**通过**

把模型自己的嵌入表换成只从磁盘表读的 reader,两臂 greedy token id 完全相同。
这是**正确性**证明,而且它便宜、无噪声、可复现。

### 2.2 性能栏：**无效(VOID)** —— 而且 `-360.8 µs` 是自证信号

真实新增代价是可直接测的:`64 calls/run × 114.2 µs = 7.31 ms`,占单轮 1353 ms 的

```
效应 0.54%
噪声 3.9%(baseline 自身 44.82 → 46.59)
```

**效应比噪声小一个数量级。** 所以「disk 快 1.7%」不是结论,是噪声恰好偏向了另一边 ——
负的 `added_us_per_token` 就是它的自证。

要在这个配置下压到 0.54% 以下,每臂约需 15 次迭代。**但没有必要**:
`disk_reader.us_per_call = 114.2 µs` 本身就是那个效应,直接测量得到,不需要 tok/s 去反推。

## 3. 边界(必须声明)

- **1 行/token,不是 16 行。** 测的是**输入嵌入**这一路,不是 PLE 的 I/O 形状。
  脚本自己的 JSON 里写了 `"note": "input embedding only (1 row/token); NOT the 16-row PLE I/O shape"`。
- **没有含 PLE 的 checkpoint 可跑。** 本机上带 `ple_layer_ids` 的 config 只有
  `Qwen3.8-Flash-Next-FP8-tokenizer`(22 MB,只有 config + tokenizer,**无权重**);
  三个 Qwen3.5(0.8B/2B/4B)的 `text_config` **一个 PLE 字段都没有**。
  所以「模型自己的 **PLE** 表」在这台机器上无法测 —— 这与
  `serve_ple_ab.py` 为什么只能用随机投影是同一个原因。
- 因此本文件证明的是:**磁盘读一个真实使用的表,不改变模型输出**。
  它不证明 PLE 路径的端到端语义。后者需要 `qwen4_exp` 权重(125B 主干 + 51B 表,
  本机 24 GiB 显存 / 12 GB 空闲盘都不够)。
- rowid 正确性另有独立证据,且**与引擎逐位相同**:
  `probes/ple_rowid_exactness_session42.md`(37 用例 / 18,048 个 rowid,IDENTICAL)。
