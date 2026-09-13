# 第二十五轮完整汇总（Session 39：S3/B2/S4 落地与 v0.2.12）

> 本轮目标：把 Phase S3、B2、S4 的可交付部分全部在 EngramDB 内落地，
> 并用 release gate / 真表脚本验证。qwen35-ple 具体 reader 依旧由另一 agent 负责。

## 1. S3：通用 Engine Adapter

- [x] `python/engramdb/adapter.py`
  - `PleMemoryAdapter`：PyTorch 状态化 serving adapter，按 `seq_ids` 维护 per-request history。
  - `TargetReaderHook`：通用 forward hook，支持 `pre` / `post`。
  - `install_target_reader_hook()`、`install_bundle_adapter()`
  - `install_vllm_target_reader()` / `install_sglang_target_reader()` 薄别名。
- [x] `PleSequenceStore` 在上一轮已落地，本轮与其组合成完整 per-request 状态协议。
- [x] Python smoke 增加 `test_engine_adapter`，验证 tensor 输出和 hook 生命周期。

## 2. B2：DiskSlotIndex 单文件 / offset table + 规模验证

- [x] Rust 原生 v3 单文件格式：`data.bin` + `offsets.bin`，`slot-index build --single-file`。
- [x] Python `DiskSlotIndex` 支持 v3 读取与构建（`single_file=True`）。
- [x] `scripts/bench_disk_slot_index.py` 支持 `--single-file`、`--cache`。
- [x] `scripts/gen_view_keys.py`：精确复现 `view build` 的 LCG keys stream。
- [x] Rust e2e：`slot_index_single_file_roundtrip` 通过。
- [x] 本机实测 1M/10M 级单文件构建；320M 全表长跑由 WSL/真表环境继续。

## 3. S4：Arrow、serving A/B、真表门禁、发布

- [x] `scripts/real_arrow_smoke.py`：真表 Store-I → Arrow IPC 读写回验。
- [x] `scripts/bench_serving_ab.py`：Store.fetch / PleMemory / PleMemoryAdapter A/B。
- [x] `scripts/real_perf_gate.py`：真表 serving 吞吐阈值门禁。
- [x] release gate 集成：
  - 真表存在时自动跑 Arrow IPC + real perf threshold。
  - 无真表时跳过，不影响 CI。
- [x] `scripts/python_wheel_smoke.py`：S3/S4 相关 smoke 全部通过。
- [x] Rust fmt / clippy / test 通过。
- [x] `scripts/release_gate.sh SKIP_BENCH=1` 本地通过。
- [x] 版本提升至 v0.2.12（由 `scripts/bump.sh --skip-gate` 完成）。

## 4. 关键文件

```text
python/engramdb/adapter.py
python/engramdb/disk_slot_index.py
crates/engramdb/src/slot_index.rs
crates/engramdb/src/main.rs
crates/engramdb/tests/cli_e2e.rs
scripts/bench_disk_slot_index.py
scripts/gen_view_keys.py
scripts/bench_serving_ab.py
scripts/real_arrow_smoke.py
scripts/real_perf_gate.py
scripts/release_gate.sh
```

## 5. 未完成 / 后续

- DiskSlotIndex 320M 全表仍建议在 WSL/稳定真表环境跑完整 10M/100M/320M 三段，并记录 CSV。
- 真正的 vLLM/SGLang **模型级** A/B 仍需要外部引擎环境；EngramDB 侧已完成通用注入接口。
