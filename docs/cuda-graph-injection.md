# 把宿主侧工作放进 CUDA graph 化的 decode 步

> **这份文档回答一个问题**：我们的磁盘读是宿主侧的阻塞 I/O，
> 而引擎把整个 decode 步捕获成 CUDA graph。**怎么让这个读进去、还被 GPU 时间盖住？**
>
> 口径：全部为 **vLLM 0.29.0 / SGLang 0.5.19 源码阅读结论**，
> 标注了文件与行号。**尚未在 GPU 上验证** —— 验证判据见 §6。
> 背景与实测数据见 `docs/roadmap.md` §35.1c/§35.1d、
> `probes/subcondition4_cuda_graph_session42.md`。

---

## 1. 一句话结论

**两个引擎都已经有一等公民的「可断 CUDA graph」机制**，
而且它**不经过 torch.compile** —— 我们踩的四个坑（编译缓存、
Inductor 折常量、trace 里禁止副作用、`splitting_ops` 不生效）**在这个模式下全部不适用**。

接入成本：**一个环境变量 + 一个装饰器 + 一个 `cudagraph_mode`**。

---

## 2. 我们原先走错的路（保留为反面教材）

我们原计划照抄 `vllm::qwen4_exp_compute_ple_ngram_ids`，把自己的算子注册进
`compilation_config.splitting_ops`。**这条路对我们无效**，原因在
`vllm/config/compilation.py:53-63`：

```python
FULL_DECODE_ONLY    = (FULL, NONE)
FULL_AND_PIECEWISE  = (FULL, PIECEWISE)

def decode_mode(self):
    return CUDAGraphMode(self.value[0]) if self.separate_routine() else self
```

`value[0] == FULL` ⇒ **decode 段走 FULL**。而断点机制在 FULL 下**明确不打断**
（见 §3.1 第 102-103 行）。所以我们的 op 在 capture 期被塞进一张 full graph，
replay 时不再执行 —— 与实测的 `reader_calls = 0` 一致。

顺带结掉两个猜测：`splitting_ops` 的**唯一**消费点是
`vllm/compilation/partition_rules.py:14-38` 的 `should_split`，
按 `target._qualified_op_name` / `packet_name` 做**字符串相等**匹配，
**不涉及 `tags`** —— `"vllm::engramdb_ple_read"` 本来就匹配得上。
问题从来不在匹配。

---

## 3. vLLM 0.29.0 的机制

### 3.1 `breakable_cudagraph.py`

`vllm/compilation/breakable_cudagraph.py`，文件头的自我定位：

> *"This is an alternative to `CUDAGraphWrapper` that replaces vLLM's
> torch.compile-based FX graph splitting with **runtime stream-capture breaks**."*
>
> *"The idea (inspired by sgl-project/sglang#19102)"*

开关（`vllm/envs.py:756-759`，默认 `0`）：

```bash
VLLM_USE_BREAKABLE_CUDAGRAPH=1
# 源码注释："Experimental: breakable cudagraph does not rely on torch.compile"
```

接线（`vllm/v1/worker/gpu_model_runner.py:5513-5518`）：

```python
if (is_breakable_cudagraph_enabled()
        and cudagraph_mode != CUDAGraphMode.NONE
        and not self.parallel_config.use_ubatching):
    self.model = BreakableCUDAGraphWrapper(self.model, self.vllm_config)
```

### 3.2 装饰器契约

`breakable_cudagraph.py:59-91` 的 `eager_break_during_capture(fn)`：
把一个自定义算子的 **Python kernel** 变成图断点。捕获期调用它时
→ 结束当前 segment → 在捕获流上 eager 执行 `fn` → 记录 `fn` 供 replay → 开新 segment。

它对我们 op 的要求（第 70-72 行，原文）：

> **"In-place output buffer required.** Decorated ops must write into a
> caller-provided output tensor; a fresh tensor returned by `fn` would change
> address each replay and break downstream graph segments."

**我们的 `_read(output, tag)` + `mutates_args=["output"]` 正好是这个形状。**

第 100-103 行是必须记住的例外：

```python
if is_forward_context_available():
    mode = get_forward_context().cudagraph_runtime_mode
    if mode == CUDAGraphMode.FULL:
        return fn(*args, **kwargs)      # ← FULL 下不打断
```

⇒ **必须用 `cudagraph_mode=PIECEWISE`**，不能用 `FULL_AND_PIECEWISE`。

### 3.3 引擎自己的模板：`prefetch_ops.py`

`vllm/model_executor/offloader/prefetch_ops.py`（94 行）和我们形状几乎相同：

```python
direct_register_custom_op(op_name="wait_prefetch",  op_func=_wait_prefetch_impl,
                          mutates_args=["input_tensor"],  fake_impl=_wait_prefetch_fake)
direct_register_custom_op(op_name="start_prefetch", op_func=_start_prefetch_impl,
                          mutates_args=["output_tensor"], fake_impl=_start_prefetch_fake)
```

文件头注释是那条纪律的官方版本：

> *"These ops use mutates_args to create data dependencies that prevent the
> compiler from reordering prefetch/sync operations."*

它的流/事件做法值得直接照抄（`vllm/model_executor/offloader/prefetch.py`）：

| 环节 | 做法 | 行号 | 为什么重要 |
|---|---|---|---|
| 独立 copy stream | `torch.cuda.Stream()` | 155-156 | 取数与计算**真正并行** |
| fork | `current_stream().record_event(e); copy_stream.wait_event(e)` | 525-529 | 事件 fork **可被 graph 捕获** |
| 完成信号 | `_copy_done_event.record(copy_stream)` | 546 | 等待变成事件等待 ⇒ 可进图 |
| wait | capture 中 `wait_event`，eager 下退化 `wait_stream` | 262-279 | 两种模式都能用 |
| join | `join_after_forward()` 在末段闭合前 join | 296-313 | 否则 replay 报 unjoined stream |

> `prefetch.py` 头部写着 *"Adapted from sglang/srt/utils/offloader.py"* ——
> **两个引擎在这里也是收敛的。**

---

## 4. SGLang 0.5.19 的机制

**v0.5.19 发布于 2026-09-05，机制由 PR #19102 引入（2026-04-11 merged）⇒ 已在装机版本里。**
路径：`python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`

| 项 | vLLM | SGLang |
|---|---|---|
| 开关 | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1`（`environ.py:1290`，`EnvBool(False)`） |
| 装饰器 | `@eager_break_during_capture`（无参） | `@eager_on_graph(enable=True, capture_stub=...)`（第 216 行，**带参**） |
| 纯断点 | —— | `break_graph()`（第 403 行，自身即 `@eager_on_graph(True)`） |
| 回写 | 要求 in-place 输出缓冲 | `_copy_output(dst, src)`（第 176 行）**额外支持** fresh tensor、tuple、dataclass、dict |
| 调试全 eager | —— | `--debug-cuda-graph` |

PR #19102 原文（明确覆盖 decode）：

> *"when enable `SGLANG_USE_BREAKABLE_CUDA_GRAPH`, the decode graph is breakable.
> The overhead is minimal if no graph break inserted."*

`capture_stub` 是一个有用的细节：捕获期用 stub 替代真实函数体
（*"contents are never consumed; warmup and replay run the real inner"*）——
**避免在建图时真读一次盘**。

文档：<https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph.md>

---

## 5. 为什么这个机制**自带重叠**（本轮最重要的发现）

我们原先以为需要一个自己造的调度器把读藏进前面的层。**不需要。**
两引擎的 replay 都是「launch 一段图 → 跑宿主函数 → launch 下一段」，**段间无同步**：

```python
# vLLM  vllm/compilation/breakable_cudagraph.py:212-214
def replay(self) -> None:
    for r in self.segments:
        r()

# SGLang  breakable_cuda_graph.py:281-290
def replay(self) -> None:
    for i, seg in enumerate(self._segments):
        seg.replay()                      # cudaGraphLaunch — 异步返回
        if i < len(self._break_fns):
            self._break_fns[i]()          # 宿主代码，与 GPU 并行
```

`cudaGraphLaunch` 立即返回 ⇒ **断点函数在宿主上执行的同时，GPU 正在跑刚 launch 的那一段。**
把读放在 layer L 的断点上，它与 **layer 0…L-1 的 GPU 时间**并行：

```
每步净增延迟 ≈ max(0, 读耗时 − τ(L))
```

| 断点位置 | τ(L) | 冷读 195.9 µs | 净增 | 依据 |
|---|---|---|---|---|
| layer 2 | 189.2 µs | 195.9 µs | **+6.7 µs**（装不下） | SGLang graph 实测 |
| layer 14 | 2295 µs | 195.9 µs | **0**（余量 ~11×） | V4.1 实测 |

⇒ 与 `docs/prefetch-lead-time.md` 的提前量模型**完全同形**。
**V4.1 把 PLE 放在 layer 14 不是为了正确性，是为了让重叠窗口盖住读延迟。**

---

## 6. 验收判据（在拿到这四项之前，子条件 4 保持 ❌）

> `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
> **且** `cudagraph_mode == PIECEWISE`（**dump 后断言，不要相信传进去的值**）
> **且** `reader_calls > 0`
> **且** `tokens_identical_to_none == False`

### ⚠️ 一个会伪装成「跑通」的静默降级

`vllm/config/compilation.py:1195-1217` 在**注意力后端不支持 piecewise** 时
不报错地改掉模式：

| 我们设的 | 静默变成 | 后果 |
|---|---|---|
| `PIECEWISE` | **`NONE`** | 变回 eager ⇒ `reader_calls` 有值但**没有图**，**假阳性** |
| `FULL_AND_PIECEWISE` | **`FULL`** | 断点失效 ⇒ 本轮实测到的失败 |

`resolve_cudagraph_mode_and_sizes()`（同文件 1375-1435）还会按后端能力**自动挑**模式。

⇒ 与 §35.1c 的 W2（编译缓存复用旧产物）是**同一类陷阱**：
**注入类实验必须自证「我确实运行在我想测的那个模式下」**，
而不只是自证「我的读发生了」。

---

## 7. 版本边界

| 引擎 | 版本 | 日期 | 有该机制 | 备注 |
|---|---|---|---|---|
| SGLang | **v0.5.19** | 2026-09-05 | ✅ | PR #19102，2026-04-11 merged |
| vLLM | **v0.29.0** | 2026-09-09 | ✅ | PR #42304，2026-05-16 merged |
| vLLM | main | 2026-09-11 | ⚠️ | PR #56312 把 breakable **限定为只接管 PIECEWISE**，FULL 交回标准 `CUDAGraphWrapper(FULL)` |

**我们机器上两个引擎都有这个机制。** #56312 晚于 v0.29.0，不影响我们，
但它印证了同一个结论：**必须以 `PIECEWISE` 运行**。

`FULL_AND_PIECEWISE` 下也能用的一条替代路径：让 PLE 读**只在 prefill/mixed 段**
（那就是 PIECEWISE）发生，decode 段走 FULL ——
但 PLE 是每 token 都要读的，所以不适用。

---

## 8. 推荐的最小设计

```
① start_ple_read(...)   eager 断点（layer 0 附近）
                        提交 io_uring 后立即返回，不阻塞
        ↓  内核在后台完成 I/O，GPU 同时算 layer 0…L-1
② wait_ple_read(...)    断点（layer L）
                        io_uring_wait_cqe → 通常已就绪，≈0 成本
                        取 pinned host buffer → copy_stream 上 async H2D（进图）
③ 消费端 event-wait（进图）
```

要点：

- **两个断点，不是一个。** 提交与等待分离，重叠才发生在①→②之间的 GPU 时间里。
  这正是 vLLM `start_prefetch` / `wait_prefetch` 的形状，照抄即可。
- **H2D 走独立 copy stream + event fork/join**，与 `prefetch.py:512-547` 一致。
- **输出必须写进调用方提供的静态缓冲**（vLLM 硬要求）；SGLang 更宽容，
  但**按硬要求写**才能在两个引擎上用同一份代码。
- **layer L 的选择由 τ(L) ≥ 读耗时 决定**，不是由模型结构决定。
  本机 128 分片全表冷读 195.9 µs ⇒ L 需满足 τ(L) ≥ 200 µs 且留余量。
