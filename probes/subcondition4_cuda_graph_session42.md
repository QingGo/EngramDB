# Session 42 · 子条件 4（CUDA graph 路径）：未闭合，但失败点已精确定位

> 口径：**真引擎（vLLM 0.29.0）、真表（128/128 分片）、真 GPU**，8 次运行全部留档。
> 产物：`scripts/serve_ple_ab_graph.py`、
> `probes/sc4_stage1_instance_patch_void.json`、`sc4_stage1e_buffer_graph_void.json`、
> `sc4_stage2_op_graph_void.json`、`sc4_stage1_buffer_eager_valid.json`、
> `sc4_stage2_op_eager_valid.json`。

## 0. 结论先说

**子条件 4 未闭合。** 但这不是「试了一下没成」，而是把失败精确到了**一层**：

| 环节 | 状态 |
|---|---|
| 类级补丁在 trace 前生效 | ✅ 已验证（eager 下输出改变） |
| 自定义 op 注册 + `mutates_args` 契约 | ✅ 已验证（eager 下 `op_calls=256 / op_rows=510`） |
| 磁盘读取经 op 发生 | ✅ 已验证（eager 下 reader 210–220 µs/次） |
| delta 进入模型（语义） | ✅ 已验证（`identical_to_none=False`） |
| **`enforce_eager=False` 时 op 在 replay 中执行** | ❌ **未达成** |

也就是说：**eager 全通、graph 全不通**，分界线非常干净。

## 1. 为什么注入必须换形态

eager 的 A/B（`serve_ple_ab.py`）用的是「layer 2 的 forward hook 返回改过的输出」。
CUDA graph replay 时**不执行任何 Python**，hook 静默失效 ⇒ 全臂看起来一样快。
这正是 V175 记录的失败模式。

所以注入拆成两半：**填充**（eager，图外）+ **消费**（图内，读固定地址）。

## 2. 四次失败，每次一个不同的原因（都留了档）

### 2.1 实例级补丁：静默无效

```
stage1  counters={'fill_calls': 384, 'consume_calls': 0}  identical_to_none=True
```

`fill_calls=384` 但这个补丁是**同一个 `install()` 同时打的** ⇒ 说明
`embed_tokens` 在**编译区之外**（每步 eager 调用），而 `Qwen3_5Model.forward`
早已被 trace 并内联了 `layer.forward` 的原始版本。补丁打晚了。

### 2.2 类级补丁 + 编译缓存：仍然静默无效

把补丁移到 `LLM()` **之前**（类级）——还是无效。原因是
**`~/.cache/vllm/torch_compile_cache`**：一份 **87 MB、由未打补丁的运行编译出的产物**
被后续每次运行复用（时间戳 23:53 对得上）。

⇒ **`VLLM_DISABLE_COMPILE_CACHE=1` 是这类实验的必要条件**，不是优化。

### 2.3 在 trace 的函数里改计数器：torch 直接拒绝

```
RuntimeError: Assigning / modifying buffers of nn.Module during forward pass is not
allowed when using cudagraph inside the compiler because it will cause silent errors.
Please use eager mode or fix the code. ... (search for the usage of the function `update`)
```

⇒ 被 trace 的 forward 必须**无副作用**。这也意味着**计数器永远无法证明 replay 发生过**
（replay 不跑 Python，而 trace 里放计数器会被拒），
**唯一可用的自证是功能性的**：reader 臂的输出必须**不同于** baseline。

同时这条 traceback 暴露了第二个问题：`consume_skipped_non2d: 1` ——
layer 2 的 hidden_states **不一定是 2 维**。

### 2.4 注册成 module buffer：eager 通了，graph 仍不通

```
sc4_stage1_buffer_eager_valid.json   mode=eager  engram-i 47.2 tok/s  identical_to_none=False  ✅
sc4_stage1e_buffer_graph_void.json   mode=graph  engram-i 316.1 tok/s identical_to_none=True   ❌
```

eager 通过说明**补丁、注册、共享张量、填充、消费全部正确**。
graph 下无效的原因：**静态张量对 Inductor 是常量**。闭包变量会被当成 constant；
`nn.Module` buffer 会变成 `get_attr` 并被折进 Inductor 自己的常量池。
`hidden_states + 0` 被**折叠掉**，加法根本没进图。

⇒ **Stage 1 的「静态 buffer + 普通加法」路线与 Inductor 根本不兼容。**

## 3. Stage 2：照抄引擎自己的 splitting op

改成 vLLM 自己的机制（`ple_layer.py:1191` 的 `qwen4_exp_compute_ple_ngram_ids`）：

```python
def _read(output: torch.Tensor, tag: str) -> None:
    reader = _STATE["inj"].reader
    toks = _MODE["pending_tokens"]
    ...
    output[:n].copy_(delta)          # 写进图里创建的张量

direct_register_custom_op(op_name="engramdb_ple_read", op_func=_read,
                          mutates_args=["output"], fake_impl=_fake)
# 并把 "vllm::engramdb_ple_read" 加进 splitting_ops
```

图里的消费端改成用**图创建的**张量承接：

```python
delta = torch.empty(n, inj.hidden, dtype=flat.dtype, device=flat.device)
torch.ops.vllm.engramdb_ple_read(delta, "ple")
flat = flat + delta
```

**结果：**

```
sc4_stage2_op_eager_valid.json   mode=eager  op_calls=256 op_rows=510  identical_to_none=False  ✅
sc4_stage2_op_graph_void.json    mode=graph  op_calls 缺失（op 未执行）  identical_to_none=True   ❌
```

eager 下 op 跑通 ⇒ **注册、`mutates_args` 契约、磁盘读取、语义全部正确**。
graph 下 op **一次都没执行**。

一个有力的旁证：Stage 1 的 graph 运行里 reader 臂慢 **−22%**（填充做了 I/O），
Stage 2 只有 **−5%**（填充只存 token，I/O 本该在 op 里）——
这个差额本身证明 **I/O 确实没发生**，与 `op_calls` 缺失一致。

## 4. 剩下的假设（下一个 session 从这里开始）

分界线是「`enforce_eager=False` 时 splitting op 不被 eager 执行」。按可能性排序：

1. **`splitting_ops` 传入时机**：我们在构造 `LLM(compilation_config=...)` 时传入，
   但 `set_splitting_ops_for_v1()` 与 `compute_hash()` 的执行顺序可能使我们的项
   未进入实际使用的列表。**下一个动作**：启动后 dump 一遍
   `llm.llm_engine.vllm_config.compilation_config.splitting_ops`，确认它真的在里面。
2. **op 需要 tag**：`direct_register_custom_op` 有 `tags` 参数，引擎的 op 没传，
   但 vLLM 的 splitting pass 可能按 tag 识别（如 `torch.Tag`）。**下一个动作**：
   读 `vllm/compilation/` 里消费 `splitting_ops` 的那段，看匹配的是什么。
3. **`use_inductor_graph_partition=False` 下的切分实现**与我们假设的不同。
4. **`FULL_AND_PIECEWISE`** 可能把整段 forward 当作 full graph 捕获，
   从而绕过了 piecewise 的切分点。

## 5. 边界（必须声明）

- **子条件 4 仍未闭合**，README §6.1 里它保持 ❌。本文件只是把失败点从
  「不知道为什么不work」推进到「eager 全通、graph 的 op 不执行」。
- 本轮**没有**产出 graph 模式下的存储代价数字。当前引用的
  **6.70% / 8.63%** 仍是「两次测量相除」的合成值
  （eager 下测的 195.9 µs ÷ 另一次运行测的 graph 单步），
  不是单次测量 —— 这一点在 `probes/engine_floor_session42.md` §5 已声明。
- 所有 graph 运行都是 `VLLM_DISABLE_COMPILE_CACHE=1`（必要，见 §2.2）。
- 模型无 PLE，投影是合成的 ⇒ 只测代价，不构成质量声明。
- 8 次运行的**全部** JSON 都在 `probes/sc4_*.json`，
  包括四次 VOID —— **失败留档是这份文件的主要价值**。
