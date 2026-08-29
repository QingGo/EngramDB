# EngramDB

磁盘优先的 Engram / PLE（n-gram 记忆表）存储引擎（Rust，DuckDB 风格：嵌入式 + 可选服务化）。

- 面向：DeepSeek **Engram**（arXiv 2601.07372）与 **Qwen3.8-Flash-Next PLE**（51.2B 参数 n-gram 表）的静态记忆表。
- 能力：确定性哈希寻址（rowid 预知）→ badge 布局 / 批量 gather / 确定性预取 / 三级缓存 / 频率索引；
  双视图：Store-I（原始表）与 Store-P（物化 e_t）；训练流（高吞吐）与推理点查（低延迟）。
- 形态：`engramdb` CLI / Rust lib / Python 绑定（PyO3，包名 `engramdb`）/ 服务化（Arrow IPC，M4）。

文档：
- `docs/design.md` —— 技术设计与开发方案（架构/负载/指标/里程碑/风险/ADR）
- `docs/engram-specs.md` —— Engram/PLE 结构规格与证据链

权重数据（不入库）：真实 FP8 checkpoint 分片软链到 `data/qwen38-ple-fp8`
（外接 SSD：`/Volumes/My Passport/qwen38-ple`，约 53GB，ModelScope `Qwen/Qwen3.8-Flash-Next-FP8` 的 PLE 相关分片）。
开发多用 `scripts/mock_table_gen.py` 合成的结构等价表。

快速开始（开发期, 以 M0 探针为主）：
```bash
cargo run -p engramdb-bench               # 探针（占位，M0）
python3 scripts/extract_ple_spec.py       # 从真实分片提取规格 JSON
python3 scripts/mock_table_gen.py         # 生成结构等价合成表
```
