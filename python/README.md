## EngramDB 的 Python 分发包

Disk-first storage engine for **Engram / PLE n-gram memory tables** (Rust).

> **分发名 `engramdb-python`（PyPI 相似名规避）；import 名仍为 `engramdb`。**
>
> 当前 v0.1.x 同时包含两条 Python 接入路径：
> 1. **PyO3 原生扩展**（优先）：`crates/engramdb-pyo3`，构建后以
>    `python/engramdb/_engramdb.so` 提供 `Store` / `View`。
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

## 定位（一句话）

让"确定性哈希的 n-gram 记忆表"（Qwen PLE、DeepSeek Engram 等）像数据库一样落盘、建索引、预取、服务化——单机 CPU+NVMe 低延迟推理 / 高吞吐训练预处理。

详细文档见上游仓库 `docs/`（design.md / specifications / roadmap）以及 `examples/interop_engram_peft.py`。
