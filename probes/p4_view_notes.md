# P4 物化视图 A/B 探针结论（真表，2026-08-29）

设置：真表 128 分片 × [2,500,012, 160] F8（=320,001,536 行，外接 USB SSD）；
采样 100,000 个 gram key × 16 头 = 1,600,000 行；view = 每条 4KB 对齐槽位（2560B 数据 + pad）。

| 路径 | 实现 | 吞吐（等效行/s） | 字节放大 |
|---|---|---|---|
| A. 16 行 scatter | gather_pp（4KB 页对齐 + 分片 8 线程 + 页内聚读），warm | 1,053,656 行/s | **20.04x**（1,252,642 唯一页 / 1.6M 行 → 每页仅 1.3 行） |
| B. 视图单记录 | 4KB 对齐记录读，1 线程（cold） | 533,769 行/s | 1.60x |
| B. 视图单记录 | 4KB 对齐记录读，**8 线程**（warm） | **5,232,842 行/s** | 1.60x |

## 结论
1. **Store-P（物化视图）在同等硬件下比"原始 16 头 gather"快 ~5x（8 线程视图 523 万 vs 10 万簇群），且字节放大 12.5 倍更低**。
2. Scatter 路径的瓶颈是不可测的页复用：16 头区域天然分散（每头相距约 20M 行），每 4KB 页平均只命中 1.3 行，
   任何"排序/聚读"都无法修复（这正是 llama.cpp 实测"4.75M 次 gather 零同页"的直接证据）。
3. 视图路径的工程前提：**4KB 对齐记录 + 并行 IO**（单线程被约 11K IOPS 底线压制，8 线程才兑现带宽）。
4. 代价模型：视图 = 1 倍磁盘额外（20M 条 × 4KB ≈ 80GB @ FP8 口径；等价原表大小）；
   收益 = 推理/训练查询面 5x 吞吐 + 磁盘带宽 12x 节省 —— **P4 判定：物化视图值得做**（受磁盘预算约束时可选"部分物化/FP8+pad"）。

## 数值细节
- A 唯一页 1,252,642（1.6M 行, 0.78 页/行）；A warm 1.52s；B 8t warm 0.306s；B 1t cold 2.998s。
- 复现：`cargo run -q --release -p engramdb-bench --bin p4view -- gen data/real-rows 100000 data/real-rows/p4view.bin /tmp/p4_keys.txt`
  `cargo run -q --release -p engramdb-bench --bin p4view -- bench data/real-rows data/real-rows/p4view.bin /tmp/p4_keys.txt`
