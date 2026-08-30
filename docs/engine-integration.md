
# EngramDB 引擎接入调研与实施路径

> 状态：调研稿，用于指导后续 vLLM / SGLang / llama.cpp 接入实现。

## 1. 我们的对外接入面

| 层 | 接口 | 适合的引擎 |
|---|---|---|
| Python wheel | `engramdb-python` + `engramdb.integrations` | engram-peft、vLLM Python 插件 |
| Rust crate | `engramdb-io` / `engramdb-core` | SGLang Rust reader、自研后端 |
| C ABI | `crates/engramdb-python` | llama.cpp / C/C++ 插件、跨语言绑定 |
| 磁盘格式 | Store-I 分片、Store-P 视图、manifest | 任何能按文件格式读取的引擎 |

## 2. vLLM

上游相关：
- [vLLM PR #54070: Disk-backed PLE n-gram tables (`VLLM_PLE_DISK_OFFLOAD_DIR`)](https://github.com/vllm-project/vllm/pull/54070)

思路：
1. vLLM 已有“PLE 表放磁盘”的方向，但具体 backend 未绑定我们的格式。
2. 我们可在 Python 侧实现一个 **vLLM PLE disk backend 插件**：
   - 使用 `engramdb.Store` / `engramdb.View` 读取 rowid / e_t。
   - 对齐 vLLM 的 `PleOffloadLayer` 或 `VLLM_PLE_DISK_OFFLOAD_DIR` 目录语义。
   - 先做功能验证，后续再对比官方磁盘路径性能。
3. 不需要改 vLLM 核心，优先用插件/环境变量接入。

预期交付：
- `engramdb` Python 包提供 `vllm_ple_backend` 适配层。
- 在 vLLM 环境中可通过配置启用。

## 3. SGLang

上游相关：
- [SGLang PR #36567: stream PLE embeddings from NVMe](https://github.com/sgl-project/sglang/pull/36567)

思路：
1. SGLang 的 NVMe PLE reader 是 Rust 路径，和 `engramdb-io` 同构。
2. 最自然的接入方式是 **直接以 Rust crate 依赖** 替换其 reader：
   - 我们已有 `BadgeGather` / `ViewReader` / `UringBatchBackend`。
   - 需要根据 SGLang 当前 reader trait 写一个 adapter。
3. 若上游接口尚未稳定，可先通过 C ABI 暴露一个最小 `fetch_rows` / `fetch_view_record` 给 SGLang 的 C++/Python 侧调用。

预期交付：
- `engramdb-io` 增加面向 SGLang 的薄封装或文档化 C ABI。
- 后续在 SGLang 上游 PR 中替换其 NVMe reader。

## 4. llama.cpp

上游相关：
- [llama.cpp TENSOR_READ_LAZY](https://github.com/ggml-org/llama.cpp/pull/27794)
- [Qwen3.8-Flash-Next on DGX Spark：PLE 从磁盘流式读取，~25 tok/s](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-ud-q4-k-xl-gguf-on-dgx-spark-with-llama-cpp-gpu-experts-ple-n-gram-table-streamed-from-disk-25-tok-s-up-to-1m-context/381720)

思路：
1. llama.cpp 是 C/C++ 核心，接入成本最高。
2. 可行路径：
   - 提供 **Store-P 视图文件** 作为可直接 mmap/ld 的 tensor 源。
   - 或通过 C ABI 作为自定义 gather backend。
   - 或贡献到上游，替换其 PLE mmap/gather 路径。
3. 不建议第一阶段做。

预期交付：
- 至少完成 Store-P 文件格式说明，便于 llama.cpp 侧直接读取。
- C ABI 保持稳定，作为后续插件入口。

## 5. 推荐实施顺序

1. **vLLM Python 插件**（因为 Python 侧已闭环，改动小、验证快）。
2. **SGLang Rust adapter**（我们的 Rust 核心同构，价值最高）。
3. **llama.cpp 文件格式/上游贡献**（最后，成本最高）。

## 6. 验收口径

- vLLM：能通过配置启用 EngramDB 磁盘 PLE，输出与内存路径一致，性能差距 ≤5%（目标）。
- SGLang：能替换 NVMe reader 并复用 `engramdb-io`，bench 与现有 reader 同量级。
- llama.cpp：能通过我们的文件格式或 C ABI 完成 PLE gather，且不劣化 tok/s。
