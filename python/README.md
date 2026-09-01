## EngramDB 的 Python 分发包

Disk-first storage engine for **Engram / PLE n-gram memory tables** (Rust).

> **分发名 `engramdb-python`（PyPI 相似名规避）；import 名仍为 `engramdb`。**
>
> 当前 v0.2.11 同时包含两条 Python 接入路径：
> 1. **PyO3 原生扩展**（优先）：`crates/engramdb-pyo3`，构建后以
>    `python/engramdb/_engramdb.so` 提供 `Store` / `View` / `PageReader` / Linux `IoUringPageReader`。
> 2. **ctypes C-ABI 回退**：`crates/engramdb-python`，无 PyO3 构建产物时也能用。
>
> `python/engramdb/__init__.py` 会自动优先加载 PyO3，失败则回退 ctypes。

## 安装

已发布到 PyPI，包名 `engramdb-python`，import 名 `engramdb`：

```bash
# 直接安装发布版
python3 -m pip install --upgrade engramdb-python

# 或使用 uv
uv add engramdb-python
```

开发/本地构建也可以使用：

```bash
# 方式 1：maturin 构建 wheel（推荐，直接产生可安装的 native wheel）
cd python
CARGO_HOME=/tmp/cargo-home RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
  maturin build --release --interpreter python3

# 方式 2：开发期内直接构建并复制到包目录
CARGO_HOME=/tmp/cargo-home RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
  cargo build -p engramdb-pyo3 --release
cp target/release/lib_engramdb.dylib python/engramdb/_engramdb.so

# 在 engram-peft 等项目中用 uv 添加本地开发依赖
cd ~/code/engram-peft
uv add --editable ../EngramDB/python
```

## 快速使用

### 存储 / 视图 / 页读取

```python
import engramdb

# 打开 Store-I：目录内为 shard_000.bin 等定长行文件
store = engramdb.Store("path/to/rows", shards=1, rows_per_shard=100, width=256)
row_bytes = store.fetch([0, 1, 2])   # bytes, 每条 256B
store.close()

# 打开 Store-P 视图
view = engramdb.View("path/to/view.bin")
rec = view.read_record(0)            # 一条 e_t 记录

# rowid → Store-P slot 语义索引（纯 Python，需要 numpy）
index = engramdb.SlotIndex.from_keys_file("path/to/view.keys.txt", heads=16)
slot = index.lookup((r0, r1, ..., r15))       # 16 元 rowid tuple
slots = index.to_slots(rowids_matrix)          # [T,16] -> [T] physical slots

# 磁盘分桶版本：适合 320M 全表，Python 可读写 v1(blake2b)/v2(fnv1a-64)
disk_index = engramdb.DiskSlotIndex.build_from_keys_file(
    "path/to/view.keys.txt", "path/to/slot-idx", hash_name="fnv1a-64"
)
slot = disk_index.lookup((r0, r1, ..., r15))
slots = disk_index.to_slots(rowids_matrix)
disk_index.close()

# SGLang 兼容：从多个 fd/offset 读页（Unix 有 PageReader，Linux 另有 IoUringPageReader）
reader = engramdb.PageReader(page_size=4096)
pages = reader.read_pages([fd0, fd1], [offset0, offset1])
```

### 线程安全的 Store 连接池

```python
from engramdb import StorePool, ThreadLocalStore

pool = StorePool("path/to/rows", shards=128, rows_per_shard=2_500_012, width=160, pool_size=4)

with pool as store:          # 借一个句柄，用完自动归还
    data = store.fetch(rowids)

tls = ThreadLocalStore(pool) # 每线程一个句柄
handle = tls.get()
try:
    data = handle.fetch(rowids)
finally:
    tls.release_current()
```

### PLE rowid、自动发现、FP8 scale

```python
from engramdb import rowids_for_seq, discover_ple, load_ple_weight_scale, load_ple_multipliers

# Qwen PLE / Engram 确定性 rowid：[T, 16]
rows = rowids_for_seq([248044, 1000, 99999, 42])

# 从 checkpoint 自动发现 PLE 表元数据、weight_scale 与 rowid multipliers
info = discover_ple("/path/to/Qwen3.8-Flash-Next")
scale = load_ple_weight_scale("/path/to/Qwen3.8-Flash-Next")
mult = load_ple_multipliers("/path/to/Qwen3.8-Flash-Next")

# discovery 返回的 info 已包含 weight_scale 和 multipliers，可直接用于 rowids
rows = rowids_for_seq([248044, 1000, 99999, 42], info=info)
```

### 真实 PLE 磁盘 Adapter

```python
from engramdb import Store
from engramdb.ple_adapter import disk_ple_from_discovery

info = discover_ple("/path/to/Qwen3.8-Flash-Next")
store = Store("/path/to/real-ple-rows", shards=128, rows_per_shard=2_500_012, width=160)
ple = disk_ple_from_discovery(store, info)  # 自动使用 weight_scale
```

### 快速 e_t tensor 读取（v0.2.9+）

训练/预计算不要再走 Python 逐行 `bytes` 拼接，使用一次 `Store.fetch` + `torch.frombuffer`：

```python
from engramdb import Store, fetch_e_t_tensor

store = Store("/path/to/real-ple-rows", shards=128, rows_per_shard=2_500_012, width=160)

# flat_rowids 是 [T * 16] 的扁平行列表
e_t = fetch_e_t_tensor(
    store,
    flat_rowids,
    scale=0.00019931793212890625,
    num_heads=16,
    head_dim=160,
    dtype=torch.float8_e4m3fn,
    out_dtype=torch.float32,
)
# e_t.shape == (T, 16, 160)
```

也可以走 `PleDiskGather` 的 tensor 方法：

```python
from engramdb.vllm import PleDiskGather

gather = PleDiskGather(store, row_bytes=160)
e_t = gather.fetch_tensor(flat_rowids, scale=..., num_heads=16, head_dim=160)
```

`PleDiskGather.fetch` 也已改为直接返回 `Store.fetch` 的连续缓冲区，不再做 Python 去重/切片/join。

流式/带 n-gram history 的 rowid 可使用：

```python
from engramdb import rowids_for_seq_with_history

rows = rowids_for_seq_with_history([eos, eos], [10, 11, 12])
```

### DiskPleEmbedding 预取与运行统计

```python
from engramdb.vllm_plugin import DiskPleEmbedding
from concurrent.futures import ThreadPoolExecutor

shared_executor = ThreadPoolExecutor(max_workers=2)
emb = DiskPleEmbedding(
    store,
    num_embeddings=...,
    embedding_dim=160,
    dtype=torch.float8_e4m3fn,
    cache_size=4096,
    prefetch_executor=shared_executor,
    prefetch_timeout=0.5,
)

emb.prefetch([rowid1, rowid2, ...])
out = emb(torch.tensor([...]))

stats = emb.get_stats()
wait = emb.get_wait_distribution()   # p50 / p90 / p99 / max
emb.close()
```

后台预取失败会自动回退到同步读取；多个 PLE 模块可以共享同一个 `prefetch_executor`。



### Serving 层：PleMemory / PleSequence / Bundle（可选）

这些高层模块不会拖慢核心导入；访问 `engramdb.PleMemory` 时才按需加载。
它们不依赖 vLLM / SGLang，也不要求 PyTorch（除 `fetch_tensor` / `current_e_t` 外）。

```python
from engramdb import PleMemory, PleSequenceStore, BundleManifest, TargetReaderRegistry

# Store-I 或 Store-P 二选一
mem = PleMemory(
    store=store,
    head_dim=160,          # Store-I 单头行字节数
    num_heads=16,
    ngram_size=3,
    heads_per_ngram=8,
    scale=0.00019931793212890625,
)
# 或
# mem = PleMemory(view=view, slot_index=disk_index, num_heads=16)

# 单条请求：保存历史、流式取 e_t
seq = mem.new_sequence()
step = seq.feed([10, 11, 12])      # 返回 raw + rowids
current = seq.current_e_t()        # torch [T,16,160]（可选）

# continuous batching：按 seq id 管理 per-request 状态
states = PleSequenceStore(mem, max_sequences=4096)
states.feed("req-1", [10, 11])
states.feed("req-2", [999])
e_t_1 = states.current_e_t("req-1")

# Bundle Manifest：描述存储 + PLE 参数 + reader 入口
bundle = BundleManifest.load("bundle.json")
print(bundle.validate())
resolved = bundle.resolved()
memory = bundle.open_memory()

# TargetReader Registry：只定义加载协议，不实现具体 qwen reader
registry = TargetReaderRegistry()
@registry.register("my-reader", version="1")
def build_reader(path, **kwargs):
    return {"path": path, **kwargs}
reader = registry.create_from_manifest(bundle)
```

通用 Engine Adapter（S3）与真表验证（S4）还提供：

```python
from engramdb import PleMemoryAdapter, install_target_reader_hook

adapter = PleMemoryAdapter(memory)
e_t = adapter(input_ids, seq_ids=[0, 1])

hook = install_target_reader_hook(model, reader, mode="post")
```

支持真表验证的脚本：

```bash
# Store-P 单文件/offset 索引构建与基准
python scripts/bench_disk_slot_index.py --single-file --grams 10000000 --out /tmp/slot-idx-v3

# 重新生成与 view build 完全一致的 Store-P keys（不依赖 git 大文件）
python scripts/gen_view_keys.py --out /tmp/full.keys --grams 20000096

# 真表 Arrow IPC 校验 + serving A/B 阈值
ENGRAMDB_REAL_ROWS=/path/to/real-rows python scripts/real_arrow_smoke.py
ENGRAMDB_REAL_ROWS=/path/to/real-rows python scripts/real_perf_gate.py
```

## 引擎适配层

目标是 **不改 vLLM / SGLang 源码**，启动前执行一小段 hook 即可把 PLE 表切到 EngramDB。

### vLLM

```python
from engramdb import Store
from engramdb.vllm_plugin import install_vllm_ple

store = Store("/path/to/engram-rows", shards=..., rows_per_shard=..., width=...)

install_vllm_ple(
    Qwen3_8FlashNextNGramEmbedding,   # 实际运行的 vLLM 模型类
    store=store,
    attr_name="embed_tokens_per_layer",
    embedding_dim=hidden_size_per_layer_input,
)

from vllm import LLM
llm = LLM(model="...", ...)
```

如果已经构造好模型实例，可以用：

```python
from engramdb.vllm_plugin import patch_named_embedding
patch_named_embedding(model, "embed_tokens_per_layer", store=store, embedding_dim=...)
```

### SGLang

```python
from engramdb.sglang import install_sglang_ple

install_sglang_ple(
    Gemma4Model,                      # 实际运行的 SGLang 模型类
    store=store,
    attr_name="embed_tokens_per_layer",
    embedding_dim=hidden_size_per_layer_input,
)
```

低层 reader 替换：

```python
from engramdb.sglang import install_sglang_io_uring_reader
install_sglang_io_uring_reader()
```

## engram-peft 集成

安装 `engramdb-python` 后，可以直接使用内置的磁盘版 `MultiHeadEmbedding`：

```python
import engramdb
from engramdb.integrations import install_disk_multi_head_embedding

store = engramdb.Store("path/to/embedding-store", shards=1, rows_per_shard=100, width=256)
install_disk_multi_head_embedding(store)

# 之后再调用 engram-peft 的 get_engram_model(...) 即可让 Engram 层从磁盘读取 embedding
```

真实 Qwen PLE FP8 Store 使用专用注入：

```python
from engramdb.integrations import install_real_qwen_ple_embedding

# scale 会从 checkpoint 自动读取；也可以显式传 scale=0.0002
install_real_qwen_ple_embedding(
    store,
    model_dir="/path/to/Qwen3.8-Flash-Next",
)
```

## 多表 / Arrow / 最小服务原型

多表按目录组织：

```python
from engramdb import Database

db = Database("path/to/tables-root")
print(db.list_tables())  # ["alpha", "beta"]

raw = db.fetch("alpha", [1, 3], shards=1, rows_per_shard=100, width=256)
```

可选 Arrow 读取（需要 `pyarrow`）：

```python
from engramdb.arrow_utils import store_fetch_arrow, table_to_ipc_bytes

table = store_fetch_arrow(store, [0, 1, 2])
ipc = table_to_ipc_bytes(table)   # Arrow IPC stream bytes
```

最小服务（当前为原型，提供两种 wire 模式）：

JSON 模式：

```python
from engramdb import Database
from engramdb.server import EngramDBServer

server = EngramDBServer(Database("path/to/tables-root"), host="127.0.0.1", port=8765)
server.serve_forever()
```

二进制模式（长度前缀 + 1-byte kind，`fetch_raw` 直接返回原始字节，
`fetch_arrow` 直接返回 Arrow IPC stream，不需要 base64 包装）：

```python
from engramdb import Database, EngramDBBinaryServer, EngramDBClient

server = EngramDBBinaryServer(
    Database("path/to/tables-root"),
    host="127.0.0.1",
    port=8765,
)
server.serve_forever()

# 客户端
with EngramDBClient("127.0.0.1", 8765) as client:
    tables = client.list_tables()
    raw = client.fetch_raw("alpha", [0, 1, 2], shards=1, rows_per_shard=100, width=256)
    ipc = client.fetch_arrow("alpha", [0, 1, 2], shards=1, rows_per_shard=100, width=256)
```

服务命令包括：

- `ping`
- `list_tables`
- `fetch` / `fetch_raw`
- `fetch_arrow`（JSON 模式返回 base64 封装的 Arrow IPC；二进制模式返回裸 Arrow IPC stream）
- `view_read`

> 注意：PyO3 `Store` 已不再是 `unsendable`，`fetch` 会释放 GIL，因此同一个 Store
> 可以从多个 Python 线程并发读取（已有 `test_store_concurrent_fetch` 冒烟）。
> 服务端 `Database.fetch` 仍会在请求线程中按需打开/复用 Store；生产级连接池化还在后续计划中。

## 定位（一句话）

让"确定性哈希的 n-gram 记忆表"（Qwen PLE、DeepSeek Engram 等）像数据库一样落盘、建索引、预取、服务化——单机 CPU+NVMe 低延迟推理 / 高吞吐训练预处理。

详细文档见上游仓库 `docs/`（design.md / specifications / roadmap）以及 `examples/interop_engram_peft.py`。
