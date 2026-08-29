# EngramDB

Disk-first storage engine for **Engram / PLE n-gram memory tables** (Rust).

> **v0.1.0 占位包**：Rust 核心（crates.io: `engramdb-keygen`/`engramdb-core`/`engramdb-io`/`engramdb`）与 PyO3 绑定正在开发中；本包于 0.1.0 仅占名 + 版本声明。完整 API / 文档随 0.2.0 发布。

## 定位（一句话）

让"确定性哈希的 n-gram 记忆表"（Qwen PLE、DeepSeek Engram 等）像数据库一样落盘、建索引、预取、服务化——单机 CPU+NVMe 低延迟推理 / 高吞吐训练预处理。

详细文档见上游仓库 `docs/`（design.md / specifications / roadmap）。
