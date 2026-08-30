## EngramDB 的 Python 分发包

Disk-first storage engine for **Engram / PLE n-gram memory tables** (Rust).

> **分发名 `engramdb-python`（PyPI 相似名规避）；import 名仍为 `engramdb`。**
>
> 当前 v0.2.x 同时包含两条 Python 接入路径：
> 1. **PyO3 原生扩展**（优先）：`crates/engramdb-pyo3`，构建后以
>    `python/engramdb/_engramdb.so` 提供 `Store` / `View` / `PageReader` / Linux `IoUringPageReader`。
> 2. **ctypes C-ABI 回退**：`crates/engramdb-python`，无 PyO3 构建产物时也能用。
>
> `python/engramdb/__init__.py` 会自动优先加载 PyO3，失败则回退 ctypes。

## 快速使用

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

# 发布到 PyPI 后可直接：
# uv add engramdb-python
```

```python
import engramdb

# 打开 Store-I：目录内为 shard_000.bin 等定长行文件
store = engramdb.Store("path/to/rows", shards=1, rows_per_shard=100, width=256)
row_bytes = store.fetch([0, 1, 2])   # bytes, 每条 256B
store.close()

# 打开 Store-P 视图
view = engramdb.View("path/to/view.bin")
rec = view.read_record(0)            # 一条 e_t 记录

# SGLang 兼容：从多个 fd/offset 读页（Unix 有 PageReader，Linux 另有 IoUringPageReader）
reader = engramdb.PageReader(page_size=4096)
pages = reader.read_pages([fd0, fd1], [offset0, offset1])
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

最小 TCP/JSON 服务（当前为原型）：

```python
from engramdb import Database
from engramdb.server import EngramDBServer

server = EngramDBServer(Database("path/to/tables-root"), host="127.0.0.1", port=8765)
server.serve_forever()
```

服务命令包括：

- `ping`
- `list_tables`
- `fetch`
- `fetch_arrow`（返回 base64 封装的 Arrow IPC bytes）
- `view_read`

> 注意：PyO3 `Store` 是不可跨线程共享的 `unsendable` 对象，因此服务端 `Database.fetch`
> 会在每个请求所在线程新开 Store；多线程共享连接的后端需要 Rust 侧安全句柄或线程池。

## 定位（一句话）

让"确定性哈希的 n-gram 记忆表"（Qwen PLE、DeepSeek Engram 等）像数据库一样落盘、建索引、预取、服务化——单机 CPU+NVMe 低延迟推理 / 高吞吐训练预处理。

详细文档见上游仓库 `docs/`（design.md / specifications / roadmap）以及 `examples/interop_engram_peft.py`。
