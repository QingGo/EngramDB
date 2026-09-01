# 第二十四轮完整汇总（Session 38：Serving 层基础落地）

> 本轮只做 EngramDB 通用 serving 能力。qwen35-ple 具体 reader 仍由另一 agent 负责。

## 1. 目标

1. 把上一轮架构分析中的 `PleMemory` / `PleSequence` 从纸面变成可运行代码。
2. 补齐 `TargetReaderRegistry` / `Bundle Manifest` 通用加载协议。
3. 保证高级 serving 模块不阻塞核心导入、不强制依赖 torch/numpy。

## 2. 完成

- [x] 新增 `python/engramdb/ple_math.py`
  - Qwen PLE rowid 纯 Python 实现，零第三方依赖。
- [x] 新增 `python/engramdb/ple_memory.py`
  - `PleMemory`：统一 Store-I / Store-P + SlotIndex 读取。
  - `PleSequence`：per-request history、`feed()`、`current_e_t()`。
  - `PleSequenceStore`：continuous batching per-sequence 状态容器。
  - `ple_memory_from_discovery()`：由 `discover_ple()` 元数据构建。
- [x] 新增 `python/engramdb/bundle.py`
  - `BundleManifest`：schema v1、路径解析、校验、`open_memory()`。
- [x] 新增 `python/engramdb/target_reader.py`
  - `TargetReaderRegistry` / `ReaderSpec`：通用注册/加载协议，不实现 qwen reader。
- [x] `engramdb` 顶层按需懒加载 `PleMemory` / `PleSequenceStore` / `BundleManifest` / `TargetReaderRegistry`。
- [x] `scripts/python_wheel_smoke.py` 增加 `test_ple_memory` 和 `test_bundle_and_target_reader`。
- [x] README、roadmap、session 文档更新。

## 3. 技术债状态

| # | 状态 |
|---|---|
| V149 | ✅ 基础落地 |
| V151 | ✅ `PleSequenceStore` |
| V152 | ✅ `BundleManifest` + `TargetReaderRegistry` |
| V156 | ✅ 纯 Python `ple_math` + lazy loading |
| V150 | ⚠️ 仍待通用 Engine Adapter |
| V153 | ⚠️ Arrow / serving A/B 未验证 |

## 4. 下一阶段

1. Phase S3：通用 Engine Adapter / forward hook。
2. Phase B2：DiskSlotIndex 320M 真表实测。
3. Phase S4：Arrow / serving A/B / 真表门禁 / v0.2.12。
