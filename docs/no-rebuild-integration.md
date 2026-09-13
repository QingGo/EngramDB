# 不改源码、不重编：接入方式与「让读被 GPU 盖住」的设计

Session 47 调研。两个问题：**(1)** 用户能不能用 pip 装好的 SGLang/vLLM 直接用上我们的库；
**(2)** PLE 的磁盘读怎么才能被 GPU 计算重叠。

结论：**(1) 能，而且两个引擎都有官方机制，我们不需要上游化补丁也能让用户用上；
(2) 能，而且不需要任何新机制 —— 把现在的一个断点拆成两个。**

---

## 第一部分：不改源码的接入

### SGLang：`SGLANG_EXTERNAL_MODEL_PACKAGE`

`python/sglang/srt/models/registry.py` 的最后两行就是全部答案：

```python
ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```

`overwrite=True` ⇒ 外部包里 `EntryClass` 指到的类**覆盖内建同名 arch**。
`import_model_classes` 扫描包内每个非子包模块，取模块级 `EntryClass`（可以是单个类或列表），
按 `cls.__name__` 注册。官方指南见
[PR #21050](https://github.com/sgl-project/sglang/pull/21050/files/) 加入的
`docs/supported_models/extending/support_new_models.md`。

**关键推论**：注册键是**类名**，而 Qwen3.8-Flash-Next 的 `config.json` 里
`architectures = ["Qwen4ExpForConditionalGeneration"]`。所以只要我们的类**就叫这个名字**，
就**连 `config.json` 都不用改**（改名的情况才需要动它，多模态还要额外设
`SGLANG_EXTERNAL_MM_MODEL_ARCH` / `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`）。

用户侧全流程：

```sh
pip install engramdb-sglang                 # 我们的包，声明对 sglang 的依赖
export SGLANG_EXTERNAL_MODEL_PACKAGE=engramdb_sglang.ext
export ENGRAMDB_PLE_STORE=/path/to/qwen38-rows
python -m sglang.launch_server \
  --model-path <Qwen3.8-Flash-Next-FP8> \
  --ple-offload-embedding --ple-offload-backend file \
  --cuda-graph-backend-decode breakable --cuda-graph-backend-prefill breakable
```

`--ple-offload-backend file` 只是为了让上游的配置管道把 `ple_offload_embedding` 打开并走到
那个 swap 点；我们**不**让 `allocate_ple_host_table` 被调用，所以不会建 51.2 GB 稀疏文件。
`check_file_backend_supported`（非 HMM 会 raise）由我们的包在导入时置空。

包内 `ext.py` 的形状：

```python
from sglang.srt.models import qwen4_exp as _q
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as _Base
from engramdb_sglang.store import install

install()          # 必须在 Qwen4ExpPLELayer 被构造之前

class Qwen4ExpForConditionalGeneration(_Base):
    pass

EntryClass = Qwen4ExpForConditionalGeneration
```

`install()` 做三件事，全部是运行期绑定替换，不碰任何文件：

1. `_q.Qwen4ExpPinnedHostEmbedding = StoreBackedEmbedding`
   —— `Qwen4ExpPLELayer.__init__` 在**调用时**查这个名字，所以替换模块全局就够；
2. `qwen4_exp_ple_table.check_file_backend_supported = lambda *a, **k: None`
   —— 搬掉非 HMM 的硬门控；
3. 把 store 根目录从 `ENGRAMDB_PLE_STORE` 读进来（`--ple-offload-dir` 会被上游填成
   SGLang 自己的 cache 目录，不能用来定位我们的表）。

**这个包不能 import 我们的补丁**：`Qwen4ExpStagedFileEmbedding` / `_adopt_embedding`
都只存在于 `0001`/`0002` 里。所以 `store.py` 要自带一份约 120 行的实现
（子类化**原版** `Qwen4ExpPinnedHostEmbedding`，自己复制那 16 个属性，
`gather` 用**原版** `eager_on_graph` 装饰 —— 这个原版 main 里就有）。
结果：**用户不需要我们的补丁，补丁 0001/0002 只作为「送回上游」的产物存在。**

版本前提：`SGLANG_EXTERNAL_MODEL_PACKAGE` 在 `environ.py` 里，
`envs.SGLANG_EXTERNAL_MODEL_PACKAGE` 出现在 registry 尾部 —— 用户在装之前
应当确认自己的版本有这两处（本机 0.5.19 是否有，未核，机器已关）。

### vLLM：entry point 插件

同一条路，机制不同：vLLM 有官方 [Plugin System](https://docs.vllm.ai/en/v0.24.0/design/plugin_system/)，
插件包在 `pyproject.toml` 里声明 entry point 即可，不需要 import 任何东西：

```toml
[project.entry-points."vllm.general_plugins"]
engramdb = "engramdb.vllm_plugin:register"
```

`register()` 里调 `ModelRegistry.register_model("Qwen4ExpForConditionalGeneration", ...)`。
我们已经有 `python/engramdb/vllm_plugin.py`（含 `patch_model_class_ple`），**缺的只是一个 entry point 声明**。

但要说清楚：**vLLM 这条只到 eager**。vLLM 的 graph 模式在 `FULL_AND_PIECEWISE` 下 decode 取
`FULL`，replay 不跑任何 Python，host 发起的读永远不会执行（路线 A 八次失败）；
路线 B（`VLLM_USE_BREAKABLE_CUDAGRAPH=1` + `cudagraph_mode=PIECEWISE`）从未跑过。
所以「不改源码 + graph 模式」目前只有 SGLang 一侧成立。

---

## 第二部分：让读被 GPU 盖住

### 现在为什么不重叠

一句话：**断点函数的第一句是 D2H，它同步整个 stream**。于是 replay 的顺序是
`seg0.replay()`（只是发射）→ 断点卡在 `.cpu()` 等 seg0 kernel 全部跑完（GPU 满载、host 空等）
→ host 读盘（**GPU 完全空闲**）→ 发布 H2D → `seg1.replay()`。

设计本意是要重叠的：`start_prefetch` 在第 *i* 层迭代开头为第 *i+1* 层的 PLE 发起，
提前量正好**一层**（τ(1) ≈ 325 µs）。但 host 发起的读必须先 D2H 才知道行号，这层提前量用不上。

### 主要手段：把一个断点拆成两个（不需要任何新机制）

`BreakableCUDAGraph.replay()` 本来就是
`for i, seg in enumerate(self._segments): seg.replay(); self._break_fns[i]()` ——
**一次 forward 里放多个断点是被支持的**，`_break_fns` 就是个列表。于是：

```
break A  (在 start_prefetch / gather)
    D2H 取行号（只等 seg0：embed + hash，很小）
    把 store 读 submit 到线程池 —— 不等待，立即返回
seg1.replay()      ← 前一个 decoder layer 的 GPU 工作，这就是 τ(1)
break B  (在 _consume_prefetched_embeddings)
    等 future → 发布 H2D
seg2.replay()
```

读盘与 seg1 的 GPU 计算在时间上真的并行了：`seg1.replay()` 发射完就返回，
host 立刻进 break B 等 future，而 GPU 正在跑前一层。

按我们已测的分量算（µs，warm）：

| tokens | rows | `read_rows` | 现在暴露（`gather_all`） | 拆分后暴露 ≈ |
|---|---|---|---|---|
| 1 | 16 | 98.7 | 142.2 | **~25**（读全被盖住） |
| 8 | 128 | 148.8 | 227.5 | ~30 |
| 32 | 512 | 429.4 | 527.9 | ~140 |
| 128 | 2048 | 792.5 | 919.0 | ~506 |

单 token warm 的单步会从 **196.5 → ~80 µs**。批量越大收益越小（读超过 τ(1) 的部分仍然暴露），
但方向是对的：`暴露 = max(0, read − τ(1)) + publish`。

**为什么这个做法能与 capture 共存**：它完全不用跨 stream 的 CUDA 机制。
上游在 [PR #29166](https://github.com/sgl-project/sglang/pull/29166) 里已经踩过这个坑 ——
CPU 权重 offload 用 `alt_stream` 预取 + event 同步，在 capture 期间会
`cudaErrorStreamCaptureIsolation`，上游的修法是**捕获时放弃重叠、改为 inline**，
「非捕获推理保持原有 alt_stream 重叠路径不变」。我们这边：读是纯 host I/O（无 CUDA），
H2D 是 replay stream 上的普通拷贝 —— 没有任何图外 event/stream 参与，所以没有那个问题。

（CUDA 本身有 `cudaGraphAddHostNode` 这个「图内 host 回调节点」原语，理论上是更正统的位置，
但两个引擎都没用它，我们也用不上。）

### 顺带便宜的一条：给 `fadvise` 真正的提前量

Session 45 量到：`posix_fadvise(WILLNEED)` 在**消费时刻**发出时，2048 行 warm 净亏 ~830 µs、
冷态**零收益**。诊断不是「WILLNEED 不好」，而是「**在消费时刻发 WILLNEED 没有用**」——
异步预读没有时间落地。

拆成两个断点后，hint 的自然位置就是 **break A**，它现在有 τ(1) ≈ 325 µs 的提前量。
这是直接冲冷态那条数字去的：冷读 2048 行现在 10.6 ms，其中相当一部分是串行缺页。

### 更深的一条：把行号搬到 host（去掉 D2H）

做到这一步，break A 连同步都不需要 —— 读可以在 seg0 还没排空前就发出，连 seg0 一起盖住。
可行性 Session 45 已证（host/device hash 逐位一致）。缺的只是把调度器 CPU 侧的 token 窗口
接过来，并复刻 `_shift_right_ignore_eos` 的窗口重置。

### 风险与未决

- store 读必须**释放 GIL**，否则 `seg1.replay()` 会被挡住。SC4 探针已经在线程池里跑过
  原生 `Store.fetch`（`read − break` 的差值就是在那条路上测的），可行；但要复测。
- 图模式下 prefetch buffer 是按 `lookup_tokens` 分桶的（`_graph_prefetch_buffers`），
  两个断点必须拿到**同一个** buffer。
- `_consume_prefetched_embeddings` 里有 `torch.cuda.current_stream().wait_stream(self._prefetch_stream)`，
  那是 capture 期 Python，replay 不会重入 `torch.cuda.stream(...)` 上下文 ——
  改成断点时要确认这个 wait 落在哪一段、会不会变成对图外 stream 的依赖（见上面 #29166 的教训）。
- **以上全部是设计，没有一行实机验证**（机器已关）。预期数字是从已测分量推算的，不是实测。
