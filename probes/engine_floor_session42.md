# Session 42 · 引擎地板 2×2：eager 让「≤5%」变成一句空话

> 口径：**同一模型、同一 GPU、同一负载，只换引擎与 graph 开关**。
> 产物：`scripts/vllm_engine_floor.py`、`scripts/serve_sglang_baseline.py`、
> `probes/vllm_graph_session42.json`、`probes/sglang_baseline_session42.json`、
> `probes/sglang_eager_session42.json`。
> 负载：Qwen3.5-0.8B，batch=1，prompt 128，max_tokens 128，贪心，5 次迭代取中位。

## 1. 结果【实测】

| 引擎 | 模式 | tok/s 中位 | ms/token | **500 µs 占单步** | graph 收益 |
|---|---|---|---|---|---|
| vLLM 0.29.0 | `enforce_eager=True` | 47.0 | 21.28 | **2.35%** | — |
| vLLM 0.29.0 | **CUDA graph** | **342.0** | **2.92** | **17.10%** | **7.28×** |
| SGLang 0.5.19 | `disable_cuda_graph=True` | 56.3 | 17.76 | **2.82%** | — |
| SGLang 0.5.19 | **CUDA graph** | **440.4** | **2.27** | **22.02%** | **7.82×** |

三点：

1. **两个引擎的 eager 是可比的**（47.0 vs 56.3，1.20×）。所以差异不是「引擎好坏」，
   而是 graph 开关。
2. **graph 在两个引擎上都买 ~7.3–7.8×**。这不是某一个引擎的实现质量问题。
3. **分母变了 7–8 倍，同一笔存储代价的含义就变了 7–8 倍。**

## 2. 这一轮真正的结论：我们的账在 graph 模式下**是超的**

把 `probes/serve_ple_ab_fulltable_session42.md` 直接测到的冷读代价
（**195.9 µs/token**，16 行 × 160 B，NVMe 冷）放到四个分母上：

| 分母 | 195.9 µs 占单步 | 对 5% |
|---|---|---|
| vLLM eager 21.28 ms | 0.92% | ✅ 远低于 |
| SGLang eager 17.76 ms | 1.10% | ✅ 远低于 |
| vLLM graph 2.92 ms | **6.70%** | ❌ **超** |
| SGLang graph 2.27 ms | **8.63%** | ❌ **超** |

**⇒ 「eager 下磁盘臂落在噪声内」从来不是「我们达标」，而是「分母被放大了 7 倍」。**
graph 一开，同一笔代价从 1% 变成 6.7–8.6%，**越过 5% 线**。

这正是 `docs/roadmap.md` §35.1 的论点，本轮把它从论证变成了数字。

## 3. 更能说明问题的是提前量：layer 2 装不下这次读

用 `docs/prefetch-lead-time.md` 的模型 `τ(L) = O_inter-step + (L/N)·C`，
取 graph 模式的单步时间当 `C`（`O_inter-step` 记 0，即**上界**；N=24，L=2）：

| 引擎 | τ(2) 上界 | 我们的冷读 | 是否装得下 |
|---|---|---|---|
| SGLang graph | **189.2 µs** | 195.9 µs | ❌ **装不下**（上界都不够） |
| vLLM graph | 243.7 µs | 195.9 µs | ⚠️ 上界够，余量 48 µs |

SGLang 那一行是**上界**结论，所以是稳的：把整个单步时间都算作可用来提前的窗口，
layer 2 也只等到 189 µs，而冷读要 196 µs。**即使按最有利的假设也不够。**

这解释了 V4.1 为什么把 PLE 放在 **layer 14**（`τ(14) = 2295 µs`）而不是 layer 2：
**不是随便选的层，是必须放得够深才装得下。**

## 4. 顺带：子条件 4（CUDA graph）的路**不是推测**，引擎里有现成范例

vLLM 启动时 dump 的 `compilation_config.splitting_ops` 里有：

```
'vllm::qwen4_exp_compute_ple_ngram_ids',
'vllm::qwen4_exp_ple_short_conv',
```

也就是说 **vLLM 把 PLE 的 rowid 计算注册成了一个 splitting op** ——
这正是在 `ple_layer.py` 里那句注释（*"Keep num_reqs-dependent ID generation
outside PIECEWISE CUDA graphs"*）的实现方式：
`cudagraph_mode = FULL_AND_PIECEWISE`，splitting op 把图切开，
**可变形状的部分留在图外，其余全部进图**。

所以我们要做的不是发明机制，而是**照抄这个模式**：把 rowid+取数注册为
splitting op，让 reader 拿到一个静态输出缓冲，其余计算照旧进图。
这条路径有引擎自己的代码作为范例，不是猜想。

## 5. 边界（必须声明）

- **引擎地板与我们的存储代价来自两次不同的运行**（不同进程、不同时间）。
  两者都做了 5 次迭代取中位，但**没有交叉验证**；上表的百分比是相除得到的，
  不是一次同时测量的结果。要把它变成单次可证伪的测量，需要把 reader 装进
  graph 模式（即子条件 4）。
- **模型无 PLE**（Qwen3.5-0.8B），存储代价来自合成注入 ⇒ 只测代价，不测质量。
- **τ(2) 是模型预测，不是测量**，且只取了上界；真实窗口还应减去
  `O_inter-step` 与到达 layer 2 之前的非计算开销。
- SGLang graph 那一轮有一个 2.5× 的离群点（172.7 tok/s，另四次 374–442）；
  中位 440.4 对它是稳健的，但散布本身要如实记下。
- 两次引擎启动耗时差很大（vLLM graph 231.7 s，含 `torch.compile` + graph 捕获；
  SGLang graph 25.7–73.6 s）。这不影响稳态 tok/s。
