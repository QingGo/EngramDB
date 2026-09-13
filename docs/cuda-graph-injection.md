# 把宿主侧工作放进 CUDA graph 化的 decode 步

> **这份文档回答一个问题**：我们的磁盘读是宿主侧的阻塞 I/O，
> 而引擎把整个 decode 步捕获成 CUDA graph。**怎么让这个读进去、还被 GPU 时间盖住？**
>
> 口径：全部为 **vLLM 0.29.0 / SGLang 0.5.19 源码阅读结论**，标注文件与行号。
> **尚未在 GPU 上验证** —— 验证判据见 §6。
> 背景与实测数据见 `docs/roadmap.md` §35.1c/§35.1d、
> `probes/subcondition4_cuda_graph_session42.md`。

---

## 0. 结论

**两个引擎都有「可断 CUDA graph」机制,而且它可能就是我们该走的路。**
但两边的**开关都不是文档里写的那个** —— 这一节先把对的写下来。

### vLLM 0.29.0

```bash
VLLM_USE_BREAKABLE_CUDAGRAPH=1          # 环境变量
```
```python
compilation_config = {"cudagraph_mode": "PIECEWISE"}   # 不要用 FULL_AND_PIECEWISE
```
```python
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

@eager_break_during_capture             # 必须是【最外层】装饰器
def _read(output: torch.Tensor, tag: str) -> None:
    ...                                  # 阻塞磁盘读，原地写进 output
direct_register_custom_op(op_name="engramdb_ple_read", op_func=_read,
                          mutates_args=["output"], fake_impl=_fake)
```
**不需要 `splitting_ops`。编译模式会被引擎自己压成 `NONE`**（这是设计,见 §3.3）。

### SGLang 0.5.19

```bash
python -m sglang.launch_server ... --cuda-graph-backend-decode=breakable
```
**⚠️ `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1` 是死变量,不要用**（见 §4.1）。
```python
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph \
    import eager_on_graph

@eager_on_graph(enable=True)            # 注意是【带参】装饰器
def my_read(...): ...
```

### 更好的信号：vLLM 对 PLE 家族**自动开启**这个机制

`vllm/config/vllm.py:75-94`：

```python
DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = frozenset({
    "DeepseekV32MTPModel", "DeepseekV32ForCausalLM",
    "DeepseekV4ForCausalLM",          # ← PLE / Engram 那一族
    "DeepSeekV4MTPModel", "Dots3NoteForCausalLM", ...
})
```

`vllm/config/vllm.py:710-728`：只要 `VLLM_USE_BREAKABLE_CUDAGRAPH` 未显式设置
且架构在表里,就 `os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"` 并打日志
*"Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1"*。

⇒ **引擎自己认定:带 PLE 的那一族模型就该用 breakable。「宿主侧取数」不是我们要硬塞进去的
异类,是引擎已经认下的模式。** 但我们本机没有 V4 的 checkpoint,所以测的时候
架构名对不上、自动开启不会触发,**必须手动设环境变量**。

---

## 1. 我们原先走错的路（保留为反面教材）

我们原计划照抄 `vllm::qwen4_exp_compute_ple_ngram_ids`，把自己的算子注册进
`compilation_config.splitting_ops`。**这条路在 `FULL_AND_PIECEWISE` 下对我们无效**，
原因在 `vllm/config/compilation.py:53-63`：

```python
FULL_DECODE_ONLY    = (FULL, NONE)
FULL_AND_PIECEWISE  = (FULL, PIECEWISE)

def decode_mode(self):
    return CUDAGraphMode(self.value[0]) if self.separate_routine() else self
```

`value[0] == FULL` ⇒ **decode 段走 FULL**。而 `cudagraph_dispatcher.py:302-324` 的
`dispatch()` **先查 FULL、命中就返回**，PIECEWISE 根本轮不到：

```python
if CUDAGraphMode.FULL in allowed_modes:
    if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.FULL]:
        return CUDAGraphMode.FULL, batch_desc_to_check      # ← 先命中
if CUDAGraphMode.PIECEWISE in allowed_modes:
    ...
```

而 FULL 图是**包住整个模型**的（`gpu_model_runner.py:5524-5530`），
里面的 PIECEWISE wrapper 看到 mode 不匹配就直通（`cuda_graph.py:244-252`）。
⇒ 我们的 op 在 capture 期被塞进一张 full graph，**replay 时一行 Python 都不跑**。
与实测 `reader_calls = 0` 完全一致。

### 顺带结掉两个猜测（**都是错的**）

| 猜测 | 结论 |
|---|---|
| `splitting_ops` 匹配靠 `torch.Tag` | ❌ **refuted**。`partition_rules.py:14-38` 的 `should_split` 是**字符串相等**，从不看 tags。`"vllm::engramdb_ple_read"` 格式本来就对 |
| 编译缓存复用了旧产物（W2） | ⚠️ **W2 仍然成立**。`compute_hash`（`compilation.py:785-818`）确实包含 `splitting_ops`，**但不包含对模型 Python 代码的 monkeypatch**。stage 1 改的是类级补丁而非配置，hash 不变 ⇒ 87 MB 旧产物照用。这是我实测到的现象，不是推测 |

### 一个必须记住的坑

`set_splitting_ops_for_v1()`（`compilation.py:1140-1255`）里
**没有 `else` 分支** —— 你传非空列表时，它**既不合也并不替换**默认值。
⇒ 自己传 `splitting_ops` 会**丢掉所有 `_attention_ops`**（通常直接 assert 崩，
`cudagraph_dispatcher.py:49-61`）。我们上一轮的脚本里有
`default_splitting_ops()` 显式带上了 attention ops，这一点做对了。

---

## 2. 两条可行路线（都要 `PIECEWISE`）

`PIECEWISE` 是共同前提。**两条路的分界线是「谁来把图切开」**：

| | 路线 A：`splitting_ops` | 路线 B：`breakable_cudagraph` |
|---|---|---|
| 切图者 | Dynamo FX + Inductor（`split_graph`） | 运行时流捕获断点（不经过 compile） |
| 需要的配置 | `mode=VLLM_COMPILE` + `splitting_ops` + `cudagraph_mode=PIECEWISE` | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` + `cudagraph_mode=PIECEWISE`（mode 被压成 NONE） |
| 我们已有的代码 | ✅ 已写好（`serve_ple_ab_graph.py`） | ⚠️ 需加一个装饰器 |
| 受 Inductor 影响 | ✅ 会（W3/W4 那两个坑依然在） | ❌ 不经过 |
| vLLM 对 PLE 家族的默认 | ❌ 否 | ✅ **是**（§0） |

**建议先试 B**：它不经过 torch.compile，我们踩过的 W3/W4 两个坑直接消失；
而且它是引擎为 PLE 家族默认选的路。A 作为后备保留（改动更小，只需改一个模式）。

`use_inductor_graph_partition=True` 是**第三条**路（Inductor 分区 + 要求
`tags=(torch._C.Tag.cudagraph_unsafe,)`）。**不建议**：`vllm/env_override.py` 里有
针对它的 monkeypatch，注释写着 *"there exists operators inside of `splitting_ops`
that have an in-place mutation"* —— 而我们的 op 正好是 `mutates_args=["output"]`。

---

## 3. vLLM 路线 B 的机制细节

### 3.1 `breakable_cudagraph.py`

模块路径是 **`vllm/compilation/breakable_cudagraph.py`（文件，不是目录** ——
文档站上的目录形态是 Sphinx 产物，`breakable_cudagraph/__init__.py` 是 404）。
文件头的自我定位：

> *"This is an alternative to `CUDAGraphWrapper` that replaces vLLM's
> torch.compile-based FX graph splitting with **runtime stream-capture breaks**."*
>
> *"The idea (inspired by sgl-project/sglang#19102)"*

引入于 **PR #42304（2026-05-16 merged）**，v0.29.0（2026-09-08）之前 4 个月就在了。
PR 的动机原文：*"make breakable cudagraph and torch.compile fullgraph exclusive.
No dynamo and inductor involved when using breakable CG."*

### 3.2 装饰器契约

`breakable_cudagraph.py:59-91` 的 `eager_break_during_capture(fn)`：
把一个自定义算子的 **Python kernel** 变成图断点。捕获期调用它时
→ 结束当前 segment → 在捕获流上 eager 执行 `fn` → 记录 `fn` 供 replay → 开新 segment。

对我们 op 的要求（第 70-72 行，原文）：

> **"In-place output buffer required.** Decorated ops must write into a
> caller-provided output tensor; a fresh tensor returned by ``fn`` would change
> address each replay and break downstream graph segments."

以及第 74-88 行：**必须是「最外层」装饰器**（若有其他引入宿主副作用的装饰器）。

`add_eager`（`:195-208`）是它调用的底层原语：

```python
def add_eager(self, fn):
    self._end_segment()
    result = fn()
    self.segments.append(fn)
    self._num_eager_breaks += 1
    self._begin_segment()
    return result
```

**第 100-103 行是必须记住的例外：**

```python
if is_forward_context_available():
    mode = get_forward_context().cudagraph_runtime_mode
    if mode == CUDAGraphMode.FULL:
        return fn(*args, **kwargs)      # ← FULL 下不打断
```

⇒ 这就是为什么必须 `PIECEWISE`。

**我们的 `_read(output, tag)` + `mutates_args=["output"]` 正好是要求的形状。**

### 3.3 `mode=NONE` 不是问题 —— 引擎显式开了例外

`_maybe_enable_breakable_cudagraph()`（`vllm/config/vllm.py:710-728`）在开启时
**主动**把编译模式压成 NONE：

```python
enabled = is_breakable_cudagraph_enabled()
if enabled:
    self.compilation_config.mode = CompilationMode.NONE
return enabled
```

而两处本来会把 `PIECEWISE` 打死的守卫，**都显式豁免了 breakable**：

```python
# vllm/config/vllm.py:1469-1480
if (self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
        and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH):        # ← 豁免
    logger.info_once("Cudagraph mode %s is not compatible with compilation mode %s. ...")
    self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

# vllm/config/vllm.py:1714-1722
if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
    assert (self.compilation_config.mode == CompilationMode.VLLM_COMPILE
            or envs.VLLM_USE_BREAKABLE_CUDAGRAPH), (...)   # ← 豁免
```

⇒ **`mode=NONE` + `cudagraph_mode=PIECEWISE` + breakable 是被引擎承认的组合。**
「breakable 与 compile 互斥」指的是 *compile 被替换掉*，不是 *只能跑 eager*。

### 3.4 引擎自己的模板：`prefetch_ops.py`

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

**它是两个算子,不是一个** —— 提交与等待分离,重叠才发生在两者之间（§5）。

它的流/事件做法（`vllm/model_executor/offloader/prefetch.py`）：

| 环节 | 做法 | 行号 | 为什么重要 |
|---|---|---|---|
| 独立 copy stream | `torch.cuda.Stream()` | 155-156 | 取数与计算**真正并行** |
| fork | `current_stream().record_event(e); copy_stream.wait_event(e)` | 525-529 | 事件 fork **可被 graph 捕获** |
| 完成信号 | `_copy_done_event.record(copy_stream)` | 546 | 等待变成事件等待 ⇒ 可进图 |
| wait | capture 中 `wait_event`，eager 下退化 `wait_stream` | 262-279 | 两种模式都能用 |
| join | `join_after_forward()` 在末段闭合前 join | 296-313 | 否则 replay 报 unjoined stream |

> `prefetch.py` 头部写着 *"Adapted from sglang/srt/utils/offloader.py"*；
> vLLM 的 breakable 又是 *"inspired by sglang#19102"*。
> **两个引擎在这个问题上已经收敛到同一套接口。**

---

## 4. SGLang 0.5.19 的机制

### 4.1 ⚠️ 官方文档是**错的**

<https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph.md> 写着
*"`SGLANG_USE_BREAKABLE_CUDA_GRAPH` … Required for `@eager_on_graph` decorators to take effect"*。

**在 v0.5.19 里这是假的。** 全树 `.py` 检索（排除 `docs/`、排除 `multimodal_gen/`
的无关同名常量）只有两处：

```
python/sglang/srt/environ.py:1290:  SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)
python/sglang/srt/arg_groups/serving_hook.py:416:  envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.set("1")
```

**只有 `.set()`,没有任何 reader** —— 不是 `envs.X` 属性读、不是
`get_bool_env_var(...)`、也不是 `os.environ` 查表。**这个变量是只写的。**
（顺带：`--debug-cuda-graph`（`serving_hook.py:416`）只设这个死变量、
**不**设 decode 后端，而 `decode_cuda_graph_runner.py:1164-1168` 又 assert
*"Breakable CUDA graph is required for --debug-cuda-graph"* ⇒
**`--debug-cuda-graph` 在 v0.5.19 上本身是坏的。**）

### 4.2 真开关是 per-phase 的 CUDA graph 后端

`python/sglang/srt/model_executor/cuda_graph_config.py`：

```python
class Backend:                          # L40-47
    FULL = "full"; BREAKABLE = "breakable"
    TC_PIECEWISE = "tc_piecewise"; DISABLED = "disabled"

ALLOWED_BACKENDS_PER_PHASE = { Phase.DECODE: (FULL, BREAKABLE, TC_PIECEWISE, DISABLED), ... }  # L50-67

def default_prefill_backend():          # L112-121
    return Backend.BREAKABLE if is_cuda() else Backend.TC_PIECEWISE

@dataclass
class CudaGraphConfig:                  # L143-152
    decode:  PhaseConfig = field(default_factory=lambda: PhaseConfig(backend=Backend.FULL))
    prefill: PhaseConfig = field(default_factory=lambda: PhaseConfig(backend=default_prefill_backend()))
```

⇒ **在 CUDA 上,`breakable` 已经是 prefill 的默认后端;decode 默认是 `full`。**
所以我们要显式指定：

```bash
--cuda-graph-backend-decode=breakable
```
（或 `--cuda-graph-config '<json>'`；旧旗标 `--enable-breakable-cuda-graph` 是
`--cuda-graph-backend-prefill=breakable` 的废弃别名,`server_args.py:3954-3957`。）

`BreakableCUDAGraphCapture` 在整个 server 路径里**只有一处被实例化**：
`runner_backend/breakable_cuda_graph_backend.py:131`，而 `_current_capture_var`
只在那里被设置。其余环境下 `eager_on_graph` 的 wrapper 是纯直通
（`breakable_cuda_graph.py:221-224`：`capture = _current_capture_var.get();
if capture is None: return inner(*args, **kwargs)`）。

⇒ **只设环境变量、只加装饰器而不选后端,行为和 vLLM 那次失败一模一样：
注册得好好的,replay 时一次都不执行。** 这是最容易重复踩的一脚。

### 4.3 内建断点只覆盖 prefill（dense 模型）

全 `srt/` 树的 `eager_on_graph` 使用点只有两处：

- `breakable_cuda_graph.py:403` —— `break_graph()` 自身
- `radix_attention.py:577/580/655` —— attention wrapper（函数式调用，
  所以 `grep "@eager_on_graph"` 只捞得到 2 处）

而 attention 那几处门控在 `radix_attention.py:178`：

```python
forward_batch.forward_mode.is_extend()      # ← 只有 prefill
```

⇒ **`--cuda-graph-backend-decode=breakable` 下,dense 模型的 decode 步会被捕获成
「一整段、零个断点」。** 机制是活的,但我们**必须自己加断点**（这正是我们要的:
只要一个,加在选定层）。decode 可达的内建断点只存在于 mamba/线性注意力
（`radix_linear_attention.py:96`）、EP-MoE DeepEP（`moe/ep_moe/layer.py:225`）等处。

### 4.4 API 差异（vLLM vs SGLang）

| 项 | vLLM | SGLang |
|---|---|---|
| 开关 | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | `--cuda-graph-backend-decode=breakable` |
| 装饰器 | `@eager_break_during_capture`（**无参**） | `@eager_on_graph(enable=True, capture_stub=...)`（**带参**） |
| 纯断点 | —— | `break_graph()` |
| 回写 | **要求** in-place 输出缓冲 | `_copy_output(dst, src)`（`:176`）**额外支持** fresh tensor / tuple / dataclass / dict |
| 捕获期跳过真实工作 | —— | `capture_stub=` 参数 |

`capture_stub` 是个有用的细节：捕获期用 stub 替代真实函数体
（*"contents are never consumed; warmup and replay run the real inner"*）——
**避免建图时真读一次盘**。

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

⚠️ **路线 A（`splitting_ops`）是否同样自带重叠,本轮没有验证。**
`split_graph` 把切分出的子图留成普通 FX `GraphModule` 在 Python 里执行
（`backends.py:730-776, 1202-1224`）—— 从机制上看**应当**同样是「launch 图 → 跑 Python → launch 图」，
但需要实测确认它不引入额外同步。
**这是选路线 B 的又一个理由**：B 的重叠语义是代码里明摆着的。

---

## 6. 验收判据（在拿到这四项之前，子条件 4 保持 ❌）

> **(a)** 运行在断点模式下：vLLM 侧 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 且
> **dump 并断言** `cudagraph_mode == PIECEWISE`；SGLang 侧
> `--cuda-graph-backend-decode=breakable` 且断言 decode 用的是 `BreakableCudaGraphBackend`
> **(b)** `reader_calls > 0`
> **(c)** `tokens_identical_to_none == False`
> **(d)** 断点确实发生：`capture.num_eager_breaks > 0`（vLLM 有 `num_eager_breaks` 属性）

### ⚠️ 三个会伪装成「跑通」的陷阱

| # | 陷阱 | 症状 | 防 |
|---|---|---|---|
| 1 | **静默降级**：`compilation.py:1195-1217` 在注意力后端不支持 piecewise 时把 `PIECEWISE` 改成 **`NONE`** | 变回 eager ⇒ `reader_calls` 有值但**没有图**,**假阳性** | dump 后断言,不信传进去的值 |
| 2 | `FULL_AND_PIECEWISE` → decode 走 FULL（§1） | 断点不生效 ⇒ 本轮实测的失败 | 用 `PIECEWISE` |
| 3 | **SGLang 只设环境变量不选后端**（§4.1） | 装饰器纯直通,op 在图内执行 | 用 `--cuda-graph-backend-decode=breakable` |

⇒ 与 §35.1c 的 W2（编译缓存复用旧产物）是**同一类陷阱**：
**注入类实验必须自证「我确实运行在我想测的那个模式下」**，
而不只是自证「我的读发生了」。

---

## 7. 版本边界

| 引擎 | 版本 | 日期 | 有该机制 | 备注 |
|---|---|---|---|---|
| SGLang | **v0.5.19** | 2026-09-05 | ✅ | PR #19102，2026-04-11 merged |
| vLLM | **v0.29.0** | 2026-09-08/09 | ✅ | PR #42304，2026-05-16 merged |
| vLLM | main | 2026-09-11 | ⚠️ | PR #56312 把 breakable **限定为只接管 PIECEWISE**，FULL 交回标准 `CUDAGraphWrapper(FULL)` |

**机器上两个引擎都有。** #56312 晚于 v0.29.0，不影响我们，
但它**印证了同一个结论：必须以 `PIECEWISE` 运行。**

> 注意：本机 `~/code/vllm` 那份 clone 是 main@2026-05-15，**比 v0.29.0 旧**，
> 里面没有 `breakable_cudagraph.py`。**不要拿它当参考。**

---

## 8. 推荐的最小实现

```
① start_ple_read(...)   断点（layer 0 附近）
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
- **输出必须写进调用方提供的静态缓冲**（vLLM 硬要求）；SGLang 更宽容
  （`_copy_output` 支持 fresh tensor），但**按硬要求写**才能在两个引擎上用同一份代码。
- **layer L 由 `τ(L) ≥ 读耗时` 决定**，不是由模型结构决定。
  本机 128 分片全表冷读 195.9 µs ⇒ L 需满足 τ(L) ≥ 200 µs 且留余量。
