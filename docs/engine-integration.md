
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

## 4.1 深度调研摘录（2026-08-30 多轮搜索）

### vLLM 官方磁盘 offload（PR #54070）

- 机制：`VLLM_PLE_DISK_OFFLOAD_DIR` + `VLLM_PLE_CPU_OFFLOAD=1`
  - 每个大参数（≥1GiB）替换为文件 backed mmap
  - 首次启动：读写映射写入文件，完成后写 sidecar
  - 后续启动：copy-on-write 映射，跳过 checkpoint shard 读取
  - `MADV_RANDOM` 防止 readahead 膨胀 RSS
- 性能：
  - 容器限 48GB 时：c1 -8%，c32 -17%，TTFT 持平
  - 不限容时反而更差：boot 阶段 checkpoint 流会冲掉 page cache
  - 冷 cache 是下界；连续 c32 会从 134 回升到 290 tok/s
- 坑：
  - 生产部署要给 serving container 设置 cgroup 内存上限
  - `ps` RSS 会包含可回收 file-backed pages，不代表匿名内存占用
  - 首次 boot 需要写约 95GB 文件

### vLLM 第三方 mmap 补丁（blazux/qwen3.8-Flash-DGX）

- 做法：直接 patch `Qwen3_8FlashNextNGramEmbedding`
  - `__init__` 用极小的 placeholder 替换 44GiB `VocabParallelEmbedding`
  - `load_weights` 丢弃大表，只保留 FP8 `weight_scale`
  - 自定义 op 做 `np.memmap` gather
- 关键优化：
  - 行 id 去重、排序、线程池让 page fault 重叠
  - 持久 pinned staging buffer + async H2D
  - decode 小批量（≤512 unique rows）跳过线程池
- 我们的对应实现：
  - `engramdb.vllm.PleDiskGather` 已提供 dedup + EngramDB 批量 fetch + expansion
  - 可作为 vLLM mmap patch 的 gather 层替代
- 坑：
  - CPU gather + H2D 不能直接进 CUDA graph；必须注册为 **splitting op** 并用 `PIECEWISE` capture
  - 自定义 op 必须在 `__init__` 注册，不能在 `forward_impl` 里注册
  - 冷区域首请求会支付 NVMe I/O；可用 `PREWARM=1` 预热
  - 每次 decode 有一次 host↔device sync，MTP 可摊薄

### SGLang NVMe PLE reader（PR #36567）

- 实现：
  - Rust PyO3 扩展 `sglang-storage`
  - `IoUringReader`：常驻 io_uring、页对齐持久 buffer、有界提交、`read_pages(fds, offsets)`
  - 另有 mmap row reader、页缓存、pinned staging、异步 H2D
  - 通过 `SGLANG_QWEN4_PLE_NVME_PATH` 开启
- 当前限制：
  - 初始只支持 TP1 + FP8 E4M3
  - 读取原始 sharded safetensors，不加载整表
- 实测：
  - Rust reader 16 随机行 p50 0.208ms / p95 0.627ms / p99 0.944ms
  - 端到端 ~24 tok/s（DGX Spark, MTP, 512 output）
- 对我们最有价值：
  - 它的 Rust reader API 与我们的 `engramdb-io` 同构，可直接作为对接接口
  - 我们已实现 `engramdb.PageReader.read_pages(fds, offsets)`，接口形状与它的 `IoUringReader.read_pages` 一致
  - 后续可以把 `engramdb.PageReader` 作为 SGLang Python 侧的可替换 reader

### llama.cpp TENSOR_READ_LAZY（PR #27794）

- 机制：
  - `--tensor-read-lazy on|off|auto`
  - `auto` 对大于 4GiB 的 tensor 使用 lazy mmap
  - 避免小模型被 lazy 读取拖慢（gemma4 实测 tg -8% ~ -10%）
- 对我们：
  - 验证了大表“不常驻”是主流方向
  - 阈值/大小级控制很重要，不能无脑 lazy

## 4.2 可借鉴的设计

1. **SGLang 的 Rust + PyO3 reader**：这是我们最好的对标模板。
2. **vLLM 自定义 op 的 splitting/捕获经验**：接入推理引擎时必须处理 CUDA graph。
3. **去重 + pinned staging + async H2D**：所有 gather 路径都应做。
4. **PREWARM / 页缓存管理**：冷启动和热态分离，和我们的 Warm/冷路径一致。
5. **容器内存上限/cgroup**：部署文档必须写。
6. **FP8 权重 scale 单独保留**：我们存储的是原始行，但接入引擎还需要处理反量化/scale。

## 5. 推荐实施顺序

1. **vLLM Python 插件**（因为 Python 侧已闭环，改动小、验证快）。
2. **SGLang Rust adapter**（我们的 Rust 核心同构，价值最高）。
3. **llama.cpp 文件格式/上游贡献**（最后，成本最高）。

## 6. 验收口径

- vLLM：能通过配置启用 EngramDB 磁盘 PLE，输出与内存路径一致，性能差距 ≤5%（目标）。
- SGLang：能替换 NVMe reader 并复用 `engramdb-io`，bench 与现有 reader 同量级。
- llama.cpp：能通过我们的文件格式或 C ABI 完成 PLE gather，且不劣化 tok/s。
