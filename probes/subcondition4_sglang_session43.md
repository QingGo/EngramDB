# Session 43 · 子条件 4（CUDA graph 路径）：**闭合**

> 口径：真引擎（SGLang 0.5.19）、真表（128/128 分片、47.7 GiB、320,001,536 行）、真 GPU（RTX 4090）。
> 产物：`scripts/sc4_sglang_break_read.py`、`scripts/engramdb_sc4_inject.py`、
> `scripts/sglang_bcg_output_patch.py`、`scripts/qwen3_5_sc4_hook.py`、
> `probes/sc4_sglang_session43.json`、`probes/sc4_sglang_d2h_isolation.json`。
> vLLM 那边的 8 次失败留档仍在 `probes/subcondition4_cuda_graph_session42.md`。

## 0. 结论先说

**子条件 4 闭合。** 上一轮唯一未达成的那一项 —— 「`enforce_eager=False` 时 op 在 replay 中执行」
—— 现在达成了，而且是用**同一把尺子**量的：

| 验收判据 | 本轮结果 |
|---|---|
| 图**真的**被 replay（不是静默退回 eager） | `backend_replay = 553 / 512` ✅ |
| **断点在 replay 期间执行**（上轮唯一未达成项） | `break_calls_in_replay = 552 / 512` ✅ |
| capture 与 replay 被干净区分 | `578 = 552 + 26` 精确分账 ✅ |
| 功能性自证：输出**不同于** baseline | `break_differs = read_differs = True`（三个互异 sha1）✅ |
| 真磁盘读取发生 | `rows = 17,552`，`read_us_median = 464.6 µs` ✅ |

## 1. 为什么换引擎：不是配置差异，是机制差异

vLLM 0.29.0 的失败不是「没调对」。`cudagraph_dispatcher.dispatch()` **先命中 `FULL`**，
而 `FULL_AND_PIECEWISE = (FULL, PIECEWISE)` ⇒ decode 取 `FULL` ⇒ 整模型被一个 full graph 包住
⇒ 里面的 PIECEWISE wrapper 见到 mode 不匹配就直通（`cuda_graph.py:244-252`）
⇒ **replay 时不跑任何 Python**，`splitting_ops` 的切分点永远轮不到。

SGLang 的 **breakable CUDA graph** 是另一种东西：

```python
# BreakableCUDAGraph.replay
for i, seg in enumerate(self._segments):
    seg.replay()                    # async cudaGraphLaunch
    if i < len(self._break_fns):
        self._break_fns[i]()        # <-- 真 host Python，段间无同步
```

`eager_on_graph` 在 capture 时结束当前段、跑一次 body 分配输出、记录 `replay_fn`，
然后开新段。**段与段之间真的有 Python 在跑**，I/O 就放在那里。这就是全部机制。

**本轮最硬的一个旁证**：`forward_calls = 26`，而 `backend_replay = 553`。

**replay 期间模型的 Python `forward` 一次都没被调用**（26 次是 capture 每桶一次 + eager prefill）。
⇒ replay 时唯一会执行的 Python **就是断点函数**。这既是 vLLM 失败的原因，
也是断点机制之所以是**唯一**出路的原因。

## 2. 路上撞到的真实缺口（都可上报）

### 2.1 BCG 后端不认识 dataclass 输出 —— 必须打补丁才能 capture

```
TypeError: Unsupported BCG output type:
           <class 'sglang.srt.layers.logits_processor.LogitsProcessorOutput'>
  at breakable_cuda_graph_backend.py:179 _alloc_full_buffer
```

`BreakableCudaGraphBackend` 用四个结构递归助手在段间搭桥
（`_output_rows` / `_alloc_full_buffer` / `_slice_output` / `_copy_output_to_buffer`），
它们只认 `None` / `Tensor` / `PPProxyTensors` / `tuple` / `list`。
而 `LogitsProcessorOutput` 是 dataclass ⇒ **几乎所有文本模型都无法用
`--cuda-graph-backend-decode=breakable`**（`qwen3_5.py` 里根本没有 `@eager_on_graph`
也在 capture 阶段就炸，可见与断点无关，是整段 forward 的桥接 buffer 分配不了）。

**修法有据可依**：同一个库隔壁 `breakable_cuda_graph.py` 的 `_copy_output`
**已经支持** `hasattr(dst, "__dict__")` 的对象并逐属性递归 —— 只是后端这四个没跟上。
`scripts/sglang_bcg_output_patch.py` 用同样规则补齐，且**用包装而非重写**：
原类型全部委派回原实现，未来 SGLang 自己补上也不冲突。

补上之后 0.8B 在 `--cuda-graph-backend-decode=breakable` 下正常起服务并 capture。

### 2.2 spawn 子进程拿不到父进程的 monkeypatch —— 换注入方式就解决

`serve_sglang_baseline.py` 曾把这条记为「SGLang 上做不了磁盘臂」的原因
（`mp.set_start_method("spawn")`）。

出路是**挂到子进程本来就会 import 的模块上**：往装好的
`sglang/srt/models/qwen3_5.py` 末尾追加几行加载器。它对 spawn / fork / exec 全都有效，
不依赖 `sitecustomize` 或 `PYTHONPATH`。**已被证明**：不打补丁时 capture 失败，
打上就成功 —— 而 capture 恰恰发生在 scheduler 子进程里。

### 2.3 两个会让人误判的坑

- **子进程 PATH 缺 venv bin**：以 `<venv>/bin/python -m ...` 启动但不 activate 时，
  子进程找不到 `ninja`，flashinfer JIT 失败，看起来像 SGLang 崩了。探针第 4 节就这样白跑一次。
- **`engine.shutdown()` 用 SIGKILL**：子进程的 `atexit` 和信号处理器**都不会跑**，
  子进程侧的计数器全部丢失。前两次冒烟测试因此读到全零 —— 而那份全零文件其实来自
  **driver 进程自己**（它 import 模型拿配置但不跑模型）。修法是把写盘放到**后台线程**
  每 500 ms 快照一次：关键路径只碰内存，记录行为不扰动被测对象。

## 3. 注入设计：两个契约

**契约一：断点必须原地改 buffer 并 `return None`。**
`_copy_output` 是**按张量**而非按元组回拷的，所以想写回就得改它拿到的 buffer。
`models/inkling.py:395-420` 就是这么做的，而且注释里明说了。
另一个选择是返回张量，但那样 `_copy_output(dst.copy_(src))` 会做一次无用的自拷贝。

**契约二：断点里只能用 stream-ordered 的 CUDA op，不能用 host 写内存。**
段间没有同步 ⇒ `seg.replay()` 是异步的，host 直接写它刚启动的段正在生产的显存会**race**。
inkling 用的是 `torch._foreach_copy_` —— 一个**排在同一条流上**的设备侧拷贝，
天然排在上一段 kernel 之后。本注入沿用同一规则：`copy_` + `add_` 都是入队 op。

**契约三（自证）：`is_in_breakable_cuda_graph()` 不能用来区分 replay。**
它在 capture 和 replay **都为 True**（`replay_session` 用的就是它）。
拿它当判据会**空洞地通过**。真正的判据是给 `BreakableCUDAGraph.replay` 装一个 flag ——
断点体只有在 `replay()` 调用栈内执行才算数。

## 4. 三个臂与数字

（第 6 节在此基础上加了第四个臂 `read_static`，用来把 D2H 单独隔离出来。）

```
none    418.9 tok/s   2.387 ms/tok   无注入
break   393.5         2.541          +0.154 ms   断点机制（合成 delta，无磁盘 I/O）
read    264.1         3.786          +1.245 ms   真磁盘读（16 行/ token）
```

`read − none = +1.399 ms = 基线的 58.6%`。三臂**各自内部确定性**（`det=True`），
三臂输出**互异** —— 功能性自证成立。（独立复跑一次得 `+1.399 / +1.488 ms`，
见第 6.3 节的噪声说明。）
与 440.4 tok/s 的既有 graph 基线同口径（同离线 `Engine` 路径、同 prompt 构造、同 median 口径），
`none` 的 418.9 说明本轮确实是 graph 态（eager 是 56.3）。

## 5. 这个数字**不是**存储代价 —— 边界必须声明

`+1.399 ms` 是**首次尝试、完全串行**的代价，而且是个很松的上界。拆开看：

| 项 | 值 | 说明 |
|---|---|---|
| `read_us_median` | **464.6 µs** | 本轮每 token 16 行的实际读 |
| 调优过的冷读 | **195.9 µs** | 既有测量（`serve_ple_ab_fulltable_session42`） |
| `break_us_median`（无磁盘） | **504.5 µs** | 光 D2H + numpy delta + 页锁定缺失的 H2D + add |

三点：

1. **磁盘读比调优值慢 2.4×**（464.6 vs 195.9）。差在**并发与暂存**：这里是每 token 一次
   `_POOL.map`（16 个小任务、chunksize 8），线程池唤醒延迟全摊在关键路径上，
   且 `n=1` 没有跨 token 批量化。
2. **断点体自己的 504 µs 与磁盘无关** —— 没有磁盘的 `break` 臂也是这个量级。
3. **但重叠确实在部分工作**：`break` 臂 504 µs 的 host 工作只让整步慢了 **154 µs**
   （≈70% 被前一段 GPU kernel 掩盖）。**真正打断重叠的是 D2H 同步**，不是磁盘。

⇒ **stage 1 完成的是「机制可行」的证明，不是「存储便宜」的证明。**
把 `+1.399 ms` 当成「Engram 每 token 开销」会严重高估。

## 6. 下一步被精确定量了（`read_static` 判决实验）

目标不是「让它跑起来」（已达成），而是**把 1.30 ms 从关键路径上摘掉**。
为了知道钱花在哪，加了一个 **`read_static`** 臂：行号在 **host 侧预先算好**，
断点里**完全不做 D2H**。于是 `read − read_static` 就是 D2H + rowid 计算的净代价。

| arm | tok/s | ms/tok | Δ vs `break` |
|---|---|---|---|
| `none` | 419.1 | 2.386 | — |
| `break` | 389.1 | 2.570 | +0.184（机制） |
| `read_static`（**无 D2H**） | 308.2 | 3.245 | **+0.675**（磁盘） |
| `read` | 258.1 | 3.874 | +1.304（磁盘 + D2H + rowid） |

```
readstatic − break      = +0.675 ms   ← 磁盘（无 D2H）
read − readstatic       = +0.629 ms   ← D2H + rowid 计算
```

### 6.1 两个假设被实测推翻，这里逐一记下

**假设一（错）：`chunksize=8` 让 16 行的读只跑 2 路并行。**
`_POOL.map(one, jobs, chunksize=8)` 确实只切出 2 个 chunk ⇒ 只有 2 个 worker。
按单线程 85.8 µs/行算，`16 × 85.8 / 2 ≈ 686 µs`，与实测的 461 µs 同量级，
看起来完美解释。改成 `chunksize=1`（16 路并行）后：

```
read_us_median   461 → 445 µs    只降 3.5%，不是腰斩
readstatic−break 0.675 → 0.446 ms
read−read_static 0.629 → 0.851 ms
read−break       1.304 → 1.297 ms   总量没动
```

⇒ **假设被推翻。** 瓶颈不是 16 行内部的并行度，而是**每次 16 行的固定派发延迟**
（线程池唤醒 + 16 次 pread 的系统调用开销）。16 行太少，摊不掉这个固定项；
跨 token 批量化才是出路。已把这条记在代码注释里，免得后人重犯。

**假设二（错）：磁盘与 D2H 各占一半。**
那条结论建立在「两个差值」上，而两个差值都会随噪声漂移。三次复跑：

| 量 | run A | run B | run C |
|---|---|---|---|
| `read − break` | 1.245 | 1.304 | **1.297** |
| `readstatic − break` | — | 0.675 | 0.446 |
| `read − read_static` | — | 0.629 | 0.851 |

⇒ **总量稳定在 1.30 ms，拆分项在 ±0.2 ms 内乱跳。**
所以「一半一半」不是结论，是噪声内的一个读数。

### 6.2 站得住的结论只有一条：host 侧是大头

**直接测量**（不是差值，不受拆分噪声影响）：

```
read_us_median（断点内直测的磁盘读）  = 445–468 µs
read − break（总量）                  = 1.30 ms（三次复跑 1.245/1.304/1.297）
⇒ host 侧残余                        ≈ 0.85 ms
```

再加一个独立证据：`read_static` 臂**完全不做 D2H**，整步仍比 `break` 贵 **0.446 ms**
—— 与 445 µs 的直测读几乎相等。**⇒ 磁盘读在这个形态下是 100% 暴露的，一点没被藏住。**

而 `break` 臂（无磁盘、有 D2H）的 body 是 504–515 µs，整步只贵 103–184 µs
⇒ **断点自己的 host 工作是能被 GPU 掩盖的**（约 65–80%）。
两条合起来说明：**掩盖机制在工作，但磁盘读和 D2H 把它挤掉了。**

### 6.3 修正后的结论与下一步

**结论（按证据强度排序）：**

1. **机制成立且可复跑**（第 0/4 节，四次运行全部 `backend_replay > 0` 且
   `break_calls_in_replay > 0`）。
2. **graph 模式下磁盘读的真实代价 ≈ 0.45 ms/token**（直测，16 行 × 160 B，本形态）。
3. **整步代价 ≈ 1.30 ms**（三次复跑稳定），其中约 **0.85 ms 在 host 侧**。
4. **磁盘那一半是完全暴露的**（`read_static − break ≈ read_us_median`）。
5. **调优过的 195.9 µs 与本形态 445–468 µs 的差距，来自跨 token 的批量化配置**，
   不是并行度、不是 chunksize。本 harness 尚未复现那个配置。

**下一步（stage 2）按实测重排：**

| # | 动作 | 预期回收 | 依据 |
|---|---|---|---|
| 1 | **跨 token 批量化 + 持久暂存**：不要每 token 单独发 16 个读 | 445 → 195.9 µs（目标值已存在，只是没复现） | §6.1 假设一被推翻后的唯一解释 |
| 2 | **消灭 D2H**：token 在步进前交给 host（采样本来就有一次 D2H，复用它） | D2H 那部分 | 假设二的噪声说明它至少有 0.4 ms 量级 |
| 3 | **rowid 移出关键路径**：step *t* 的行号在 step *t−1* 算 | 同上 | 同上 |
| 4 | 确认无分配器强同步；`non_blocking=True` | 残余 | `≈118 µs`（H2D+add，来自两次 body 之差） |

**2 与 3 可合并成一次「fill 步」**：前一步 tail 里拿 token、算行号、发起异步读；
断点里只剩「等 + H2D + add」。这就是 `docs/cuda-graph-injection.md` §8.1 四条前提的完整形态。

**⚠️ 纪律提醒**：本节的 2/3/4 三行都只有「≤ 一次读数」的证据强度，
而 §6.1 刚刚演示了这种强度会被噪声骗到。**下一轮动 stage 2 之前，必须先做到条件 5
要求的多轮 counterbalance**，否则会把噪声当进度。

## 7. 已知未做

- **stage 2（重叠）未做** —— 本轮是有意的串行变体。
- **真实 PLE 模型跑不了**：`Qwen3.8-Flash-Next-FP8` 权重索引 `total_size = 185.5 GB`
  （48 层、512 专家 MoE、FP8），单张 4090（24 GiB）装不下。
  所以本轮用的是 `Qwen3.5-0.8B` + **合成投影**，**只测代价，不构成任何质量声明**。
- **`L*` 的单层时间是整步平均**（含与层数无关的固定开销）⇒ 所有 `L*` 都是**下界**，
  真实需求更大。斜率法（24/12/6 层取差分）未做。
