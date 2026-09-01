# 本 Session 综合整理（2026-08-30 后段，至真实 Qwen3.5-0.8B）

> 本文件是对本轮主要工作的单一入口摘要。详细分 session 复盘见
> `docs/session-log.md` Session 8–18；第八轮系统性复盘与债务表见
> `docs/roadmap.md` Section 12。

## 1. 尝试过什么

| # | 尝试 | 结果 |
|---|---|---|
| 1 | 配置并测试 WSL / 树莓派免密 SSH | ✅ 成功 |
| 2 | 在树莓派 aarch64 + WSL2 x86_64 安装 `engramdb-python==0.2.4` 并跑 wheel smoke | ✅ 全绿 |
| 3 | 在 WSL 安装真实 vLLM 0.28.0 + torch 2.13.0+cu130 | ✅ |
| 4 | 在 WSL 安装真实 SGLang 0.5.9 + torch 2.9.1+cu128 | ✅ |
| 5 | 用真实 `Qwen3ForCausalLM` 验证 `install_vllm_ple` | ✅ |
| 6 | 用真实 `Qwen3ForCausalLM` 验证 `install_sglang_ple` | ✅ |
| 7 | 在 WSL 同步源码并编译最新 `engramdb` CLI | ✅ |
| 8 | 用 `view build --keys` 构建访问序视图 | ✅ 19999 grams 0.4s |
| 9 | 访问序视图校验 | ✅ 1000/19999 grams 全部一致 |
| 10 | 热页缓存下顺序/随机读吞吐 | ✅ |
| 11 | `posix_fadvise(DONTNEED)` 冷盘顺序/随机 A/B | ✅ 786 vs 86 MB/s |
| 12 | 真实 vLLM 模型类 embedding A/B（raw disk） | ✅ |
| 13 | 给 `DiskPleEmbedding` 实现行级 LRU 缓存并复测 | ✅ |
| 14 | 实现多表 `Database` | ✅ smoke 通过 |
| 15 | 实现 Arrow Table / IPC helper | ✅ |
| 16 | 实现最小 TCP/JSON 服务 + `fetch_arrow` | ✅ smoke 通过 |
| 17 | 实现二进制 length-prefix 服务 + `EngramDBClient` | ✅ smoke 通过 |
| 18 | 验证 PyPI 上真实的 `engramdb-python==0.2.5` macOS wheel | ✅ 安装+smoke 全绿 |
| 19 | Rust 侧首批多表 / manifest / `serve` 收敛 | ✅ `tables` + JSON `serve` 可用 |
| 20 | CPU 小模型端到端 decode A/B（memory vs disk raw vs LRU） | ✅ 首次拿到曲线 |
| 21 | 发布 v0.2.6（Rust serve/check/二进制、CPU A/B、cache_size=0 修复） | ✅ PyPI + GitHub Release |
| 22 | Rust `check`、二进制 serve、`view_read` | ✅ 单元/CLI e2e/手工验证 |
| 23 | 将真实 Qwen3.5-0.8B 软链到 `data/` 并复制到 WSL | ✅ 1.7GB 完整复制 |
| 24 | 真实 Qwen3.5-0.8B CPU 端到端 decode A/B | ✅ 首批真实模型曲线 |

## 2. 踩过的坑

| # | 坑 | 处理/教训 |
|---|---|---|
| 1 | WSL 默认 shell 是 zsh，`wsl -- bash -lc` 不总是按预期执行 | 使用 `wsl -e /usr/bin/bash -c` 或明确用 `-e /usr/bin/bash` |
| 2 | 通过 SSH 运行 WSL 长任务会超时/被杀 | 所有长任务改用 Windows `schtasks` + 日志文件后台跑 |
| 3 | 通过 PowerShell 传 WSL 命令时，双引号、管道、`||` 经常被错误解析 | 复杂命令写成独立 `.sh`/`.py` 文件，复制到 `/mnt/c/tmp` 再执行 |
| 4 | WSL 里没有系统 pip，且旧 venv 是从 macOS 同步来的坏软链 | 使用 `uv` 创建新 venv |
| 5 | PyTorch cu130/cu128 不支持 GTX1070 `sm_61` | GPU path 只能等待兼容 torch 或换 CPU/llama.cpp |
| 6 | 同一进程构造两个 vLLM 模型会报 `Duplicate layer name` | 第二个模型必须传独立 `prefix` |
| 7 | vLLM CPU 完整 forward 因 CUDA-only rotary op 失败 | 当前 vLLM+torch 不能直接 CPU 跑完整模型 forward |
| 8 | SGLang 模型构造需要初始化 parallel state + global server args + dp attention | 补齐 `init_distributed_environment` / `initialize_dp_attention` 后可用 |
| 9 | 服务端跨线程使用 PyO3 `Store` 触发 `unsendable` panic | 每个请求在当前线程新开 Store；长期要 Rust 安全句柄 |
| 10 | `view bench` / 构建脚本因 manifest 扩展名判断错误中断 | 注意 `view_out.with_extension("manifest.json")` 实际得到 `access.manifest.json` |
| 11 | 小文件冷读 8 线程反而比 1 线程慢 | 冷读不能无脑并行，需要顺序流调度 |
| 12 | WSL 执行 `/mnt/c` 下的 `.sh` 直接作为参数会挂起 | 用 `bash -c 'bash /mnt/c/tmp/xxx.sh'` 或复制到 WSL 本地 |
| 13 | 大文件复制用 `nohup scp` 会被工具环境误判结束 | 用 `screen` 或接受后台进程持续；需轮询目标文件大小确认 |
| 14 | WSL 直接读 `/mnt/c` 模型较慢但可用 | 大模型先完整复制到 WSL 可访问路径再跑 |
| 15 | 本地 transformers 4.57 不认识 Qwen3.5 | 使用 WSL `transformers 5.16.1` |
| 16 | `DiskPleEmbedding(cache_size=0)` 原先会 KeyError | 已修复为显式 raw no-cache 路径 |
| 17 | 真实 0.8B CPU A/B 噪声大 | 需要固定 seed、更多重复、中位数/CSV 才能形成回归 |
| 18 | Rust `serve --binary` 修改后忘记重新 build | 用旧二进制测试会误判“命令不存在/未实现” |

## 3. 已完成的部分

### 3.1 真机验证

- 树莓派 aarch64 + WSL2 Ubuntu x86_64：
  - `PageReader` / `IoUringPageReader` / `SGLangPageReader` / `Store` / `PleDiskGather` smoke 全绿。
- vLLM 0.28.0 真实 `Qwen3ForCausalLM`：
  - `install_vllm_ple` 类级 hook ✅
  - 实例级 `patch_named_embedding` ✅
  - `DiskPleEmbedding.forward` 输出 `(1,3,32)` ✅
- SGLang 0.5.9 真实 `Qwen3ForCausalLM`：
  - `install_sglang_ple` 类级 hook ✅
  - 实例级 `patch_named_embedding` ✅
  - `DiskPleEmbedding.forward` 输出 `(1,3,32)` ✅

### 3.2 访问序视图

- `view build --keys` 在 WSL 可用。
- 19999 grams（319984 rowids）构建 0.4s，校验通过。
- 热缓存：
  - 顺序 1t：9.55M rows/s
  - 随机 1t：9.25M rows/s
  - 顺序 8t：29.57M rows/s
  - 随机 8t：27.97M rows/s
- 冷缓存：
  - 顺序 1t：785.8 MB/s
  - 随机 1t：86.0 MB/s
  - **约 9.1× 收益**

### 3.3 PLE 数据面性能

- raw disk `DiskPleEmbedding`：
  - batch=1：235.6μs/call
  - batch=4：240.8μs/call
  - batch=16：268.0μs/call
- 加入行级 LRU 后：
  - batch=1：14.0μs/call
  - batch=4：21.7μs/call
  - batch=16：22.9μs/call

### 3.4 多表 / Arrow / 服务

新增 Python 模块：

- `python/engramdb/tables.py`：`Database` 多表注册与按表 fetch
- `python/engramdb/arrow_utils.py`：Arrow Table / IPC bytes
- `python/engramdb/server.py`：最小 TCP/JSON + 二进制 length-prefix 服务
- `python/engramdb/service_client.py`：二进制协议客户端
- `python/engramdb/vllm_plugin.py`：`DiskPleEmbedding` LRU 缓存

新增脚本：

- `scripts/wsl_cold_view_bench.py`
- `scripts/vllm_embedding_ab.py`
- `scripts/service_smoke.py`
- `scripts/vllm_ple_smoke.py`
- `scripts/sglang_ple_smoke.py`

smoke：

```text
Database OK: ['alpha', 'beta']
Arrow OK: 2 ['rowid', 'row'] ipc_bytes 416
Server Arrow OK: 2 ['rowid', 'row']
Binary Server OK: ping/list_tables/fetch_raw
SERVICE_SMOKE_OK
```

### 3.5 Rust 多表 / manifest / serve

新增 Rust 能力：

- `engramdb tables <root>`：扫描多表目录
- `engramdb serve <root> [--binary]`：
  - JSON：`ping` / `list_tables` / `fetch` / `view_read`
  - 二进制：length-prefix + kind，`fetch_raw` / `view_read` 直接裸字节
  - 与 Python `EngramDBClient` 协议兼容
- `engramdb check <root>`：
  - 校验 manifest 布局
  - 校验每个 shard/badge 文件存在、非空、大小为 row_bytes 整数倍
  - 返回汇总 JSON，存在问题则非零退出

验证：

```text
ping True
tables ['alpha']
check: {"ok": true, "table_count": 1, ...}
```

### 3.6 真实 Qwen3.5-0.8B 模型接入

- 模型源：
  ```text
  /Volumes/My Passport/model/Qwen3.5-0.8B
  ```
- 本地软链：
  ```text
  data/Qwen3.5-0.8B -> /Volumes/My Passport/model/Qwen3.5-0.8B
  ```
- `data/` 已在 `.gitignore`，不会进入 git。
- 已完整复制到 WSL：
  ```text
  /mnt/c/Users/minam/engramdb-transfer/Qwen3.5-0.8B
  ```
- WSL `transformers 5.16.1` 可加载：
  ```text
  Qwen3_5ForCausalLM
  embed_tokens: [248320, 1024]
  ```

新增脚本：

- `scripts/qwen35_cpu_decode_ab.py`
- `scripts/cpu_tiny_decode_ab.py`

### 3.7 真实模型 CPU E2E A/B 首次结果

使用稀疏 store 测读取路径性能，未做 bit-exact。

| 轮次 | memory | disk raw | disk LRU |
|---|---|---|---|
| 4 tokens, reps=3 | 3.66 tok/s | 2.84 tok/s | 2.87 tok/s |
| 4 tokens, reps=1 | 4.02 tok/s | 2.40 tok/s | 3.67 tok/s |
| 8 tokens, reps=1 | 2.12 tok/s | 3.93 tok/s | 1.53 tok/s |

初步结论：

- 已跑通真实 0.8B 模型端到端 CPU decode；
- 原始磁盘通常比内存慢 20% 以上；
- LRU 在短序列/小工作集下优势不明显；
- WSL CPU 噪声较大，尚不能作为最终回归结论。

## 4. 新发现的问题

| # | 问题 | 影响 |
|---|---|---|
| V8 | PyO3 `Store` 是 `unsendable` | 服务端不能跨线程共享，当前每请求新开 Store |
| V9 | 服务已有二进制 Arrow IPC wire（length-prefix），但尚无连接复用/认证/线程安全句柄 | 二进制数据传输面已闭环，生产级服务面仍待做 |
| V10 | GPU 路径被 torch/Pascal 兼容性卡住 | GTX1070 sm_61 无法使用当前 cu130/cu128 |
| V11 | 小文件冷读多线程反而更慢 | 冷读需要顺序流调度，不能盲目并行 |
| V12 | 多表/服务已开始向 Rust 收敛；Rust `tables`/`serve`/`check`/二进制 `view_read` 已落地 | 仍缺 Arrow IPC、Unix socket、table_id 深度、连接复用/认证、性能门禁 |
| V13 | v0.2.5、v0.2.6 已发布 | ✅ PyPI/GitHub Release |
| V14 | 首未命中仍走 raw disk | LRU 只解决热路径，冷启动/预热/Tier 未做 |
| V15 | 模型 PLE 属性仍靠手填 | 需要自动发现或配置映射 |
| V4 | 完整服务引擎 decode A/B 未闭环，但 CPU tiny + 真实 0.8B 两条 E2E 曲线已获得 | 继续向 vLLM/SGLang serving 收敛 |
| V25 | 真实 Qwen3.5 A/B 使用稀疏零值 store，不是真实 PLE 表，也不是 bit-exact | 需要真实权重填充 store 与 bit-exact 对照 |
| V26 | 真实 0.8B CPU A/B 噪声大（memory 2.1–4.0，raw 2.4–3.9，LRU 1.5–3.7） | 需要固定 seed、长序列、多次中位数、CSV 基线 |
| V27 | Rust serve 仍无 Arrow IPC、Unix socket、认证、限流、连接池、线程安全句柄、性能门禁 | 服务面仍是雏形 |
| V28 | 真实模型在外部盘 + 手动复制到 WSL，无自动化准备命令 | 需要 `scripts/prep_real_model.sh` |
| V29 | 真实模型未在 vLLM/SGLang serving 中验证 | 仍是 transformers 直跑 |
| V30 | 当前 patch 的是普通 input embedding，不是真实 PLE 层 | 性能代表“磁盘 embedding 替换”，不是完整 PLE 语义 |
| V31 | 真实模型太大，不能进 CI | 需要独立 nightly/手动基准 job |
| V32 | v0.2.6 tag 后 master 又加了 view_read、真实模型脚本 | 需要及时收编进 v0.2.7 |

## 5. 计划要完成的部分

### 已闭环 / 已完成

- ✅ v0.2.5、v0.2.6 发布
- ✅ Python wheel smoke 扩展并进入 CI
- ✅ Rust `tables` / `serve` / `check` / 二进制 / `view_read`
- ✅ CPU tiny + 真实 Qwen3.5-0.8B E2E A/B 脚本
- ✅ 真实模型软链（`data/` gitignore）

### 下一步（近期）

1. **可信性能基线**
   - 固定 seed、固定输入、固定 token 数；
   - `reps>=5`，输出中位数 + p90；
   - 生成 `probes/qwen35_cpu_baseline.csv`、`probes/cpu_tiny_baseline.csv`；
   - 设定 raw / LRU 相对 memory 的回归阈值。

2. **真实数据面**
   - 稀疏 store → 真实权重填充 store；
   - 增加 bit-exact：memory output == disk output；
   - 从 Qwen3.5 定位真实 PLE / Engram 表属性。

3. **Rust 服务产品化**
   - Unix socket；
   - Arrow IPC 或零拷贝 raw path；
   - 每线程 store 句柄 / 连接池；
   - 认证、限流；
   - embedded vs server ≤2%。

4. **真实服务引擎**
   - 尝试 vLLM / SGLang / llama.cpp 加载 Qwen3.5-0.8B；
   - EngramDB 替换 PLE 数据面；
   - serving 级 A/B，目标 ≤5%。

### 中期

5. 冷读顺序流调度与 `StreamingPlanner` / Tier 预取打通。
6. 大表冷态顺序/随机复测；自适应顺序流。
7. 自动发现模型 PLE 属性。
8. 真实模型准备脚本 `scripts/prep_real_model.sh`。

### 长期

9. vLLM/SGLang 完整 serving 集成与性能 A/B（≤5%）。
10. 上游 patch / llama.cpp 文件格式 / C ABI。
11. 保持“不改上游源码”的薄层接入哲学。
12. GPU 路径待兼容 torch 或换硬件后补测。

## 6. 相关提交

```text
a41263d docs(roadmap): add eighth-round systematic retrospect and v0.3 refinements
1ddf575 feat(bench): add real Qwen3.5-0.8B CPU decode A/B script
d796954 feat(rust): add view_read to JSON and binary serve
0c48771 release: bump v0.2.6
506f8a5 feat(bench): add CSV baseline output to CPU decode A/B script
4c76e90 feat(rust): add manifest check and binary-length serve protocol
c23017b feat(rust,engines): add Rust multi-table serve and CPU decode A/B
764831c ci(python): install pyarrow in smoke job to exercise Arrow paths
1fbe60a release: bump v0.2.5
5a07bf8 feat(python): add binary Arrow IPC service protocol and expand wheel smoke
0321ff7 docs(roadmap): add sixth-round systematic retrospect and updated debts/plan
6a59cbf docs(python): document multi-table, Arrow, and minimal service usage
b983677 feat(python): add fetch_arrow command to minimal service
c70ed5d feat(python): add multi-table Database, Arrow helpers, and minimal TCP service
3e49786 feat(engines): add LRU cache to DiskPleEmbedding and record vLLM A/B
2b56f9d docs(session): verify cold sequential vs random access-order view on WSL
1f23b75 docs(session): record WSL access-order view verification
c515d30 docs(engines): verify real vLLM/SGLang PLE hooks on WSL2
99cafd8 docs(session): record real Linux wheel verification on Pi and WSL2
```

## 7. Session 19 增量（可信基线 + 真实权重 bit-exact）

在 Session 18 之后继续完成：

### 已闭环

- ✅ `scripts/cpu_tiny_decode_ab.py` / `scripts/qwen35_cpu_decode_ab.py` 升级为可回归基线：
  - 固定 seed=42、模型 eval、reps>=5、median/p90/mean/min/max；
  - CSV 默认写入 `probes/`；
  - 可选 raw/LRU slowdown 门禁。
- ✅ `probes/cpu_tiny_baseline.csv`、`probes/qwen35_cpu_baseline.csv` 已入库
  （`.gitignore` 已放开 `probes/*baseline*.csv`）。
- ✅ `scripts/decode_baseline_check.py`：读取 CSV 并检查阈值，当前默认通过。
- ✅ `scripts/qwen35_bit_exact.py`：真实权重填充 store + bit-exact 验证。
  - 实机结果：直接 embedding `max_abs=0.0`，完整生成序列一致，`BIT_EXACT_PASS`。
- ✅ `scripts/prep_real_model.sh`：自动软链/复制/校验真实模型。
- ✅ 真实 Qwen3.5 A/B 改为真实权重 store 后重新出数：
  - memory 4.69 tok/s
  - disk raw 4.15 tok/s（慢 13.0%）
  - disk LRU 3.94 tok/s（慢 19.2%）

### 对应提交

```text
feat(bench): add reproducible CPU decode baseline stats and threshold check
feat(bench): add real-weight store fill and Qwen3.5 bit-exact check
feat(scripts): add real Qwen3.5 model prep helper
docs(session): record Session 19 trustworthy baseline and bit-exact progress
```

## 8. Session 19 增补：真实 PLE 自动发现

- 新增 `python/engramdb/ple_discovery.py`、`scripts/inspect_ple_attributes.py`。
- 在 Qwen3.8-Flash-Next / Qwen4Exp 中定位真实 PLE：
  - `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_*.weight`
  - `ple_embed_dim=2560`、`ngram_size=3`、`split_ngram_parts=128`、`heads_per_ngram=8`
- Qwen3.5-0.8B 无 PLE，所以当前 A/B 是普通 embedding 替换，不是真实 PLE 语义。

## 9. Session 20：系统性思考

- 在 roadmap 中新增第十轮系统性复盘（第 14 节）。
- 重新锚定终极目标与可验收指标。
- 列出本轮新发现的技术债 V33–V43。
- 制定分阶段计划：
  - Phase 0 测量硬化
  - Phase 1 真实 PLE 数据面
  - Phase 2 性能架构（原生 gather + 预取 + LRU 命中率）
  - Phase 3 真实服务引擎 A/B
  - Phase 4 Rust 服务产品化
  - Phase 5 长期维护
- 明确核心判断：先做真实 PLE + 可重叠性能，再做服务化；LRU 必须有命中率证据。

## 10. Session 20 增补：真实 PLE Store 位级闭环

- 新增 `scripts/real_ple_bit_exact.py`。
- 发现并修复 Rust `gather_pp` / `gather_plan` 在多分片 PLE 表上的偏移 bug：
  - 原来错误使用全局 rowid 计算文件偏移；
  - 现在使用 shard 内局部偏移。
- 新增回归测试 `gather_pp_multishard_uses_local_row_offsets`。
- 真实 PLE 128 shard 抽样 100 行 SHA-256 完全一致，`PLE_STORE_BIT_EXACT_PASS`。

## 11. Session 20 增补：真实 PLE Layer Bit-Exact + Adapter

- 新增 `python/engramdb/ple_adapter.py`：
  - `DiskPleNGramEmbedding` 磁盘 PLE adapter；
  - 自动生成 PLE rowid；
  - FP8 行反量化；
  - 顺序 decode 最小历史。
- 新增 `scripts/ple_layer_bit_exact.py`：
  - 不加载完整大模型；
  - 加载真实 PLE 层投影/卷积权重；
  - 真实 PLE 层 forward 对比 raw-file vs EngramDB；
  - 结果 `max_abs=0.0`，`PLE_LAYER_BIT_EXACT_PASS`。

## 12. Session 21：真实 PLE 数据面里程碑与系统性回顾

- 真实 PLE Store bit-exact ✅
- 修复多分片 gather 偏移 bug ✅
- DiskPleNGramEmbedding adapter ✅
- 真实 PLE 层 forward bit-exact ✅
- 完整模型 E2E ❌（环境/内存限制）
- 新增技术债 V44–V51
- 计划进入 Phase A：完整模型 custom loader + 官方类验证

## 13. Session 22：服务兄弟项目与系统性回顾

- 补齐 C ABI：rowids + abi version
- 增强 DiskMultiHeadEmbedding FP8 反量化
- 新增 install_real_qwen_ple_embedding
- 新增 sibling_contract_smoke.py
- qwen35-ple M0 quick 通过
- 新增技术债 V52–V59
- 计划聚焦：配置即用、真实 e2e、Rust 热路径

## 14. 第十二轮完整整理（Session 19–22：可信基线 → 真实 PLE → 兄弟项目服务）

### 14.1 本轮计划

1. 建立可信 CPU decode 性能基线
2. 真实权重 store + bit-exact
3. 自动发现并接入真实 Qwen PLE
4. 真实 PLE Store / PLE Layer bit-exact
5. 构造可复用磁盘 PLE adapter
6. 服务 qwen35-ple / engram-peft 兄弟项目
7. 补齐 C ABI 与 FP8 注入
8. 发布新版本

### 14.2 重要发现

- 真实 Qwen PLE 路径在 Qwen4Exp / Qwen3.8：
  `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_*.weight`
- Qwen3.5-0.8B 本身没有 PLE，只用于普通 embedding 替换验证。
- 发现 `gather_pp` 多分片偏移 bug：
  - 用全局 rowid 计算文件偏移；
  - 单分片正确；
  - 真实 128-shard 表全部错误。
- FP8 PLE 行必须乘 `weight_scale` 才等于真实数值。
- LRU 在无复用短生成中可能比 raw 慢；真实 PLE 场景必须在有命中率证据时使用。
- 兄弟项目契约要求：
  - `engramdb_rowids_for_seq`
  - `engramdb_abi_version`
  - 真实 FP8 磁盘注入需要支持 `float8_e4m3fn + scale`

### 14.3 做的尝试

- 固定 seed / eval / reps>=5 / median+p90
- 真实权重填充 Qwen3.5 store
- Qwen3.5 bit-exact
- Qwen4Exp PLE 元数据自动发现
- 真实 PLE 128-shard Store 位级验证
- 自实现 PLE layer forward bit-exact
- `DiskPleNGramEmbedding` adapter
- engram-peft 风格 `DiskMultiHeadEmbedding` FP8 反量化
- C ABI rowids 对拍 qwen35-ple
- qwen35-ple M0 quick smoke

### 14.4 踩过的坑

| 坑 | 处理 |
|---|---|
| 真实 PLE Store 读取跨 shard 全错 | 修复 `gather_pp` / `gather_plan` 局部偏移 |
| PyTorch/Numpy 本地环境 numpy 2 不兼容 | 用 WSL / 避免 numpy 路径 |
| 完整 Qwen4Exp 无法加载（200GB+ embedding） | 改为只加载 PLE 小权重 + 磁盘 Store 验证 |
| 自实现 PLE forward 容易 shape/groups 错 | 对照官方源码修复 flatten / conv groups |
| engram-peft 默认 float32 注入不能读 FP8 | 增加 scale / output_dtype / 专用 wrapper |
| 完整模型 E2E 环境不足 | 明确记录为环境限制 |

### 14.5 完成的内容

- ✅ 可信 CPU decode 基线 + CSV + 阈值检查
- ✅ 真实权重 Qwen3.5 bit-exact
- ✅ 真实 PLE Store bit-exact
- ✅ `gather_pp` 多分片 bug 修复 + 回归测试
- ✅ `ple_discovery` 自动发现
- ✅ `DiskPleNGramEmbedding` adapter
- ✅ 真实 PLE layer forward bit-exact
- ✅ C ABI `rowids_for_seq` / `abi_version`
- ✅ `DiskMultiHeadEmbedding` FP8 支持
- ✅ `install_real_qwen_ple_embedding`
- ✅ 兄弟项目契约 smoke
- ✅ qwen35-ple M0 quick 通过

### 14.6 未完成的内容

- ❌ 完整 Qwen4Exp 模型 E2E A/B（环境/内存限制）
- ❌ engram-peft 原生消费 `table_source`（兄弟侧待实现）
- ❌ qwen35-ple 真实 e2e 脚本改用新 FP8 wrapper（兄弟侧待更新）
- ❌ Rust native PLE 热路径
- ❌ vLLM/SGLang/llama.cpp serving A/B
- ❌ `ENG_DEEPSEEK_V1` C ABI 实现

### 14.7 未来计划

1. Phase A：配置即用
   - engram-peft 自动注入
   - qwen35-ple 真实路径更新
   - 自动读取 weight_scale
   - Python `rowids_for_seq`
   - C ABI 测试入 CI
2. Phase B：完整模型 E2E
3. Phase C：Rust 原生性能路径 + 预取
4. Phase D：服务化 / 推理引擎 A/B
5. 发布维护：v0.2.7

## 15. Session 23：v0.2.7 CI 修复与 Phase A EngramDB 侧补齐

### 15.1 修复

- `crates/engramdb-python/src/lib.rs` import 顺序不符合 rustfmt，导致所有 preflight 失败。
- `python/engramdb/__init__.py` 曾 eager import `DiskPleNGramEmbedding`，使无 torch 的 wheel smoke 无法导入。
- 修复后 Python 包在无 torch 环境下可正常使用 Store / discovery / rowids。

### 15.2 新增

- `load_ple_weight_scale(model_dir)`：从真实 checkpoint 自动读取 FP8 `weight_scale`。
- `discover_ple()` 自动附带 `weight_scale`。
- `disk_ple_from_discovery()` / `install_real_qwen_ple_embedding()` 支持自动 scale。
- Python `rowids_for_seq()`，底层优先 PyO3 → C ABI → 纯 Python。
- PyO3 native `rowids_for_seq` / `abi_version`。
- `scripts/c_abi_smoke.py` 并接入 CI。

### 15.3 状态

- fmt / clippy / cargo test 全绿。
- Python wheel smoke（无 torch）、service smoke、C ABI smoke 全绿。
- 真实 checkpoint `weight_scale=0.00019931793212890625` 自动读取验证通过。
- 下一步可发 v0.2.8，或继续 Phase A 的兄弟侧接入。

## 16. Session 24：系统性思考与文档刷新

### 16.1 目标

从“功能正确”走向“可发布、可安装、可文档化、可稳定前进”。

### 16.2 本轮发现

- CI 失败往往不是功能问题，而是发布工程问题。
- Python 核心包必须保持无 torch 可用。
- 文档更新不能晚于版本发布。
- 兄弟侧配置自动消费仍是最大“方便使用”缺口。

### 16.3 新增技术债

V60–V73，核心包括：release gate、README 收编、兄弟侧配置、Rust 热路径、完整模型 E2E、serving A/B、Store 线程安全。

### 16.4 计划

Phase 0 发布稳定 → Phase A 配置即用 → Phase B 完整模型 E2E → Phase C Rust 性能 → Phase D 服务化。

详见 `docs/roadmap.md` 第 17 节。

## 17. Session 24 综合整理（v0.2.8 发布 + 工程稳定化 + 文档刷新）

### 17.1 本轮计划

本轮的核心目标不是“再验证一个正确性点”，而是：

1. 修复 v0.2.7 暴露的 CI 失败；
2. 补齐 Phase A 中 EngramDB 自己负责的部分：
   - 自动读取真实 checkpoint 的 `weight_scale`
   - Python `rowids_for_seq()` 公共 API
   - PyO3 native rowids
   - C ABI smoke 进 CI
3. 发布一个干净的 v0.2.8；
4. 刷新 README，补上 Rust / Python 已发布库的安装与使用方法；
5. 做系统性思考，明确后续最稳的发展路径。

### 17.2 本轮发现

- **功能正确不等于可发布**
  C ABI、bit-exact、真实 PLE 都已验证，但 v0.2.7 仍因 rustfmt 和无 torch 环境导入失败。

- **Python 核心包必须轻依赖**
  不是所有用户都装 PyTorch；Store、rowids、discovery、服务必须能在纯 Python 环境使用。

- **文档与版本会分叉**
  v0.2.8 tag 后 README 才更新，说明文档应纳入版本收口，而不是发布后补写。

- **兄弟侧“配置即用”仍未闭环**
  EngramDB 侧已准备好，但 engram-peft 消费 `table_source`、qwen35-ple 真实 FP8 路径仍属于兄弟仓库动作。

- **发布工程需要本地一键 gate**
  CI 在远端失败才被发现，成本太高；应在 bump/push 前本地跑完整 release gate。

### 17.3 做的尝试

- 本地复现 CI：
  - `cargo fmt --all --check`
  - `cargo test --workspace`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `python_wheel_smoke.py`
  - `service_smoke.py`
  - `c_abi_smoke.py`
  - `decode_baseline_check.py`
- 验证无 torch 环境下的 Python 包导入。
- 用真实 Qwen checkpoint 验证：
  - `discover_ple()`
  - `load_ple_weight_scale()`
  - 实际读取到的 `weight_scale=0.00019931793212890625`
- 验证 rowids 三路一致：
  - PyO3 native
  - C ABI
  - 纯 Python fallback
- 刷新根 README 与 `python/README.md`，补安装与真实 PLE 用法。

### 17.4 踩过的坑

| 坑 | 处理 |
|---|---|
| `engramdb-keygen` import 顺序不符合 rustfmt | 调整 import 顺序，`cargo fmt` 通过 |
| `__init__.py` eager import `DiskPleNGramEmbedding` 强制加载 torch | 改为 lazy attribute import；`ple_adapter` 可选 torch |
| `ple_adapter` 无 torch 时 `nn.Module` 不存在 | 用 dummy `nn` 让模块可导入；后续应改为更干净 plugin/stub |
| 本地 PyO3 构建缺 `-undefined dynamic_lookup` | 使用 `RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"` |
| 发布后才发现 README 未更新 | 在 master 补文档，下版必须把 README 与代码同一 commit 收编 |
| 真实 checkpoint 的 `weight_scale` 是 BF16 而不是 F32 | 实现 BF16/F16/F32 标量读取 |
| `discover_ple()` 读取超大 safetensors index 两次 | 功能可用，后续可缓存 index 或输出轻量 spec |

### 17.5 完成的内容

- ✅ 修复 v0.2.7 CI：
  - rustfmt
  - 无 torch Python 导入
- ✅ `load_ple_weight_scale(model_dir)`
- ✅ `discover_ple()` 自动附带 `weight_scale`
- ✅ `disk_ple_from_discovery()` 自动使用 scale
- ✅ `install_real_qwen_ple_embedding(store, model_dir=...)` 自动 scale
- ✅ Python `engramdb.rowids_for_seq()`
- ✅ PyO3 native `rowids_for_seq()` / `abi_version()`
- ✅ C ABI smoke 脚本 + CI 接入
- ✅ `python_wheel_smoke.py` 增加 rowids 回归
- ✅ v0.2.8 发布并推送
- ✅ 根 README / python README 刷新
- ✅ roadmap 第 17 节系统性思考
- ✅ session-log / session-summary / handoff 更新

### 17.6 未完成的内容

- ✅ `scripts/release_gate.sh` 已创建，并已接入 `bump.sh` 默认流程
- ❌ 最新 README 尚未进入 v0.2.8 发布物（需下版收编）
- ✅ `install_real_qwen_ple_embedding` 无来源时改为显式 warning
- ✅ `rowids_for_seq()` 现支持 `multipliers`/`info`，`discover_ple()` 自动读取 `layer_multipliers`
- ✅ engram-peft 自动消费 `table_source`（已推 feature branch）
- ✅ qwen35-ple 真实 e2e 切 FP8 wrapper（`run_m0_smoke.py --e2e --ple-model-dir`）
- ❌ Rust native PLE gather + dequant 热路径
- ❌ 完整模型 E2E（官方类 + 磁盘 PLE）
- ❌ vLLM / SGLang / llama.cpp serving A/B
- ❌ `ENG_DEEPSEEK_V1` C ABI
- ❌ Python Store 线程安全/连接复用

### 17.7 当前状态

```text
v0.2.8 已发布
master 已包含 README 刷新 + 系统性思考
本地所有 gate 全绿
工作区干净
```

### 17.8 未来计划

#### Phase 0：发布与工程稳定（最高优先）
- ✅ 创建 `scripts/release_gate.sh`（bump 默认先跑）
- 🔶 把 README 收编进下一版本（内容已刷新，待 bump）
- ✅ 去掉静默硬编码 scale fallback（改为显式 warning）
- ✅ `rowids_for_seq()` 支持 `info` / `multipliers`；`discover_ple()` 附带 `layer_multipliers`
- 🔶 README 示例自动化（rowids/discovery/safetensors 已进 smoke，其余待补）

#### Phase A：兄弟项目配置即用
- ✅ engram-peft `table_source` 自动注入（已直接推 master）
- ✅ qwen35-ple 真实 FP8 路径（M0 e2e 配置驱动）
- ✅ 跨仓契约 smoke 已加入兄弟 CI（qwen35-ple checkout EngramDB + engram-peft）

#### Phase B：真实模型 E2E
- ✅ custom loader / skip ngram_embedding（`engramdb.official_loader` + `qwen4_ple_custom_loader.py`）
- ✅ 真实 FP8 PLE e2e 已跑通（Qwen3.5-0.8B + 真实 128-shard Store-I + 配置驱动自动注入）
- 🔶 官方 Qwen4Exp 模型类验证（代码路径就绪，待 Qwen4Exp transformers/大内存环境）
- ❌ 大内存/云环境 memory vs disk A/B

#### Phase C：Rust 性能路径
- native rowid + gather + dequant
- 预取重叠
- 真实 PLE 性能矩阵

#### Phase D：服务化 / 推理引擎
- vLLM / SGLang / llama.cpp serving A/B
- Store 线程安全 / 连接复用
- Arrow IPC 服务化 / 认证 / 发布形态

### 17.9 本轮纪律

1. 发布正确性与功能正确性同等重要。
2. 核心 Python 包必须保持轻依赖。
3. 文档与版本必须同点收编。
4. 跨仓正确性以 golden / C ABI 守门。
5. 性能最终必须下沉 Rust。
6. 配置驱动优先于手动调用。



# Session 26 系统性思考（第十四轮：配置即用 + 真实 FP8 e2e + Phase B 初步）

> 完整版见 `docs/roadmap.md` Section 18。

## 1. 终极目标

不变：

> 让 DeepSeek Engram / Qwen PLE 成为任何小模型、训练器、推理引擎都能廉价使用的磁盘优先存储基础设施——像 DuckDB 之于分析数据库。

本轮后最重要的进展是：**配置即用从设计字段变成可执行闭环**，并且**真实 FP8 Store-I 首次在真实小模型 e2e 中跑通**。

## 2. 本轮技术债（V74–V85）

- 真实 FP8 e2e 不是官方 Qwen4Exp 完整模型
- `--load-model` 仍可能分配巨大 ngram embedding
- e2e 依赖临时手工依赖路径，不可复现
- DiskPleNGramEmbedding 未接 Transformers Cache
- `engramdb:view` 未实现
- 跨仓 CI 仍只覆盖轻量 hash 契约
- 自动注入在无 scale/model_dir 时可能误读 FP8
- engram-peft 全局 patch 绑定单 store，多模型不安全
- 无真实 memory vs disk A/B
- 版本/README 未统一收编
- Store 线程安全/连接复用未做

## 3. 开发计划重点

1. **Phase B1**：官方 Qwen4Exp 加载前 patch ngram 占位，真正绕过 200GB+ PLE 内存
2. **Phase B2**：官方类 + DiskPleNGramEmbedding bit-exact
3. **Phase B3**：真实 memory vs disk A/B（固定 seed/reps/CSV）
4. **Phase C**：Rust/PyO3 native rowid + gather + dequant + 预取
5. **Phase D**：vLLM/SGLang/llama.cpp serving A/B + Store 线程安全 + Arrow
6. **工程**：固化 e2e 环境、统一 bump + README 收编、扩 runtime/官方类 CI

## 4. 本轮纪律

- “能跑”不等于“验收通过”
- 先证明不分配 PLE 大表，再谈完整模型
- 性能结论必须带 hit-rate 与分段计时
- 可复现环境优先于临时 hack
- 跨仓只走契约 + golden
- 版本、文档、代码同点收编

# Session 26 综合整理（Consolidated Round Review）

> 本段把本轮从“Phase 0 收尾”到“Phase A 配置即用”再到“Phase B 初步/真实 FP8 e2e”的完整过程做一页式归档。
> 详细战略与技术债见 `docs/roadmap.md` Section 18；详细会话日志见 `docs/session-log.md`。

## 1. 本轮计划

1. 完成 Phase 0：发布门禁、README/文档收口、rowid multipliers 自动读取。
2. 完成 Phase A：engram-peft 自动消费 `table_source="engramdb:store"`；qwen35-ple 配置驱动真实 FP8 e2e。
3. 加入跨仓契约 smoke 到兄弟项目 CI。
4. 推进 Phase B：官方模型加载时跳过 `ngram_embedding.shard_*`，并验证官方模型类。
5. 在真实 PLE 环境执行 FP8 e2e。
6. 做完后系统性思考并写入文档。

## 2. 发现

1. **engram-peft 远程 master 已经包含 `engine` / `table_spec` / `table_source` / `prime_sizes`**
   说明前一阶段的跨仓字段已经进入 master；我们只需在其上增量加“自动消费”和表路径字段。

2. **真实 FP8 e2e 可以跑通**
   使用 Qwen3.5-0.8B + 真实 128-shard Store-I + 配置驱动自动注入，成功得到：
   ```text
   REAL_FP8_E2E_OK
   elapsed ~9.6s
   logits finite
   generated shape [1, 10]
   ```

3. **真实 checkpoint 的 PLE 表规模非常明确**
   dry-run 显示：
   - 非 PLE tensor：151,960
   - PLE ngram tensor：129
   - 即完整加载时真正需要跳过/磁盘化的就是这 129 个 PLE 相关张量。

4. **transformers 版本是硬门槛**
   transformers 4.57 不识别 `qwen3_5`；必须使用支持 Qwen3.5/Qwen4Exp 的 5.x 版本。

5. **engram-peft 完整包导入依赖过重**
   直接 `import engram_peft` 会拉入 TRL/datasets 等；在仅做推理 e2e 时可以通过子模块加载绕过。

6. **跨仓 hash 契约其实不依赖 torch**
   `QwenPleHashMapping` 路径是纯 NumPy；测试中只需一个极轻 torch stub 即可运行，避免安装多 GB torch。

7. **本地 git 目录在部分仓库不可写**
   EngramDB 可写；engram-peft 与 qwen35-ple 的 `.git` 后来都变为“Operation not permitted”，无法直接 add/commit，需要用可写镜像 clone 后推送。

## 3. 做的尝试

1. 新增 `scripts/release_gate.sh`，并将 `bump.sh` 默认前置到该 gate。
2. 扩展 `discover_ple()` / `load_ple_multipliers()` / `rowids_for_seq(info=...)`。
3. 实现 engram-peft `table_*` 配置字段与 `get_engram_model()` 自动注入。
4. 实现 qwen35-ple `EngineConfig` 扩展和 `Qwen35PleConfig.to_engram_config()`。
5. 修改 `run_m0_smoke.py --e2e` 为配置驱动真实 FP8 路径。
6. 修改 qwen35-ple CI，checkout EngramDB + engram-peft，启用跨仓 golden/契约测试。
7. 新增 `engramdb.official_loader`，提供：
   - `filter_ngram_shard_state_dict`
   - `install_disk_ple_in_official_model`
   - `load_state_dict_without_ngram_shards`
8. 新增 `scripts/qwen4_ple_custom_loader.py`，可 dry-run 展示会跳过多少 PLE tensor。
9. 新增 `scripts/run_real_fp8_e2e.py`，用轻量子模块加载 engram-peft，避免 TRL/datasets。
10. 在多个 Python 环境中尝试运行真实 e2e；
11. 使用可写镜像 clone 将 engram-peft 提交直接推送到 master；
12. 将 qwen35-ple 的新增文件和文档推送到 main；
13. 完成第十四轮系统性思考并写入 roadmap/session 文档。

## 4. 踩过的坑

1. **engram-peft / qwen35-ple `.git` 不可写**
   - 现象：`git add`/`commit` 报 `Unable to create index.lock: Operation not permitted`。
   - 解决：`git clone --no-hardlinks` 到 `/tmp`，在 clone 中提交并推送；原工作区保留改动。

2. **直接 `import engram_peft` 需要 TRL/datasets**
   - 现象：`run_m0_smoke.py --e2e` 因缺 `peft`/`accelerate`/`trl`/`datasets` 失败。
   - 解决：借鉴 `run_qwen35_e2e.py` 的子模块加载法，写 `run_real_fp8_e2e.py`，只加载 `engram_peft.config` / `engram_peft.model`。

3. **transformers 4.57 不识别 Qwen3.5**
   - 现象：`AutoModelForCausalLM.from_pretrained` 报 `model type qwen3_5 not recognized`。
   - 解决：切换到 LLM-CompileForge 环境的 transformers 5.9。

4. **Python 3.10 缺 `typing.override`**
   - 现象：直接 import engram-peft 报 `cannot import name 'override' from 'typing'`。
   - 解决：在脚本里用 `typing_extensions` 补 `typing.override`。

5. **多个 venv 缺 pip，uv pip 又遇缓存权限**
   - 解决：从 uv cache / 其他 conda 环境手工复制纯 Python 包 + dist-info 到 `/tmp/pylibs`，用 PYTHONPATH 组合。

6. **混合不同 Python 版本的 site-packages 会崩**
   - 现象：把 Python 3.13 site-packages 直接加入 3.10 venv 导致 NumPy C 扩展不兼容。
   - 解决：只复制纯 Python 包（peft/jaxtyping/typeguard/accelerate）和 dist-info，不整目录混入。

7. **输出写到仓库 `outputs/` 被拒绝**
   - 现象：`PermissionError: outputs/real-fp8-e2e.json`。
   - 解决：`--output /tmp/real-fp8-e2e.json` 重跑成功。

8. **“完整加载”仍可能不是真正跳过分配**
   - 发现：`official_loader` 目前主要做过滤与替换，但完整 `--load-model` 还没有在官方类构造前使用轻量占位；这是下一轮必须补的实质缺口。

## 5. 完成的内容

- ✅ `scripts/release_gate.sh` + `bump.sh` 默认 gate
- ✅ `discover_ple()` 自动返回 `weight_scale` + `layer_multipliers`
- ✅ `load_ple_multipliers()` / `read_safetensors_i64()`
- ✅ `rowids_for_seq()` 支持 `multipliers` / `info`
- ✅ `install_real_qwen_ple_embedding` 无来源时显式 warning
- ✅ engram-peft `table_source="engramdb:store"` 自动注入，已推 master
- ✅ qwen35-ple `to_engram_config()` 配置桥接
- ✅ qwen35-ple CI 跨仓契约 smoke
- ✅ `engramdb.official_loader`
- ✅ `qwen4_ple_custom_loader.py` dry-run
- ✅ `run_real_fp8_e2e.py` 并实际跑通真实 FP8 e2e
- ✅ README / Python README / roadmap / session / handoff 更新
- ✅ 第十四轮系统性思考文档

## 6. 未完成的内容

- ❌ 官方 Qwen4Exp 完整模型实机加载验证
- ❌ `--load-model` 真正绕过 200GB+ PLE embedding 内存分配
- ❌ 官方 `Qwen4ExpTextPLELayer` + `DiskPleNGramEmbedding` bit-exact
- ❌ 真实 memory vs disk A/B
- ❌ Rust/PyO3 native rowid + gather + dequant
- ❌ 预取重叠
- ❌ vLLM / SGLang / llama.cpp serving A/B
- ❌ Store 线程安全 / 连接复用
- ❌ `engramdb:view` 自动消费
- ❌ engram-peft / qwen35-ple 版本 bump 与 README 同点发布收编
- ❌ 可复现 e2e 环境（当前依赖临时 `/tmp/pylibs`）

## 7. 未来计划

1. **Phase B1：官方模型加载不分配 PLE 大表**
   - 在 `from_config` / `from_pretrained` 前 patch 官方 `Qwen4ExpTextNGramEmbedding`
   - 过滤 ngram shard state dict
   - 构造后替换为磁盘 PLE
   - 验证峰值内存不含 200GB+ PLE 大表

2. **Phase B2：官方类 bit-exact**
   - 官方 PLE 层 + DiskPleNGramEmbedding 小批量真实 token 对拍
   - 覆盖 EOS / 多段 / batch / MTP / streaming

3. **Phase B3：真实 A/B**
   - memory vs disk
   - 固定 seed / reps / token 序列
   - tok/s + hit-rate + fetch/convert 分段 + CSV 阈值

4. **Phase C：Rust/PyO3 热路径**
   - native rowid + gather + dequant
   - 预取与计算重叠
   - 冷/热、并发、批大小矩阵

5. **Phase D：服务化 / 推理引擎**
   - vLLM / SGLang / llama.cpp serving A/B
   - Store 线程安全 / 连接池
   - Arrow IPC 服务化 / 认证

6. **工程稳定**
   - 固化真实 e2e 依赖环境
   - 三仓库统一 bump + README 同点收编
   - 扩展 runtime/官方类 CI 或 nightly

## 8. 当前状态

```text
EngramDB   master 52a282c（docs session26）
engram-peft master dc74c85（auto table_source）
qwen35-ple main  a5ca602（config bridge + real FP8 e2e）
真实 FP8 e2e 已跑通：REAL_FP8_E2E_OK
官方 Qwen4Exp 完整模型：未验证
性能 A/B：未做
```

## 9. 本轮纪律

1. 功能“能跑”不是验收。
2. 先证明不分配 PLE 大表，再谈完整模型。
3. 所有性能结论必须带 hit-rate 与分段计时。
4. 可复现环境优先于临时 hack。
5. 跨仓正确性只走契约 + golden。
6. 版本、文档、代码必须同点收编。

# Session 27 系统性思考（第十五轮：B1/B2 代码落地 + 异步预取方向）

## 1. 本轮定位
从“真实 FP8 e2e 能跑”推进到：
- 官方加载占位已真正落地；
- 官方类小表 bit-exact 已通过；
- 真实 PLE 行稀疏 oracle 已通过；
- 异步预取基础已落地（GIL 释放、prefetch/future、模型级 hook）。

## 2. 本轮发现
1. PLE 行只依赖 token ids，理论上可以提前预取。
2. 当前 PyO3 Store 持 GIL、不可跨线程，不能直接异步。
3. DiskPleNGramEmbedding 已修复 batch 维度，并支持每 batch context 和 chunked streaming。
4. 小表 bit-exact 证明 rowid/素数表/EOS/context 逻辑正确，但仍是合成数据。
5. 已经用“稀疏真实行 oracle”绕开完整表加载：144 个真实行与 checkpoint byte-identical，DiskPle real-Store maxdiff=0.0。
6. 完整官方模型验证的主阻塞是 Qwen4Exp Transformers + 大内存环境。

## 3. 新增技术债
V86–V98，详见 `docs/roadmap.md` Section 19.4。核心是：
- V86 PyO3 Store 持 GIL / unsendable
- V87 无 prefetch API / 模型级 pre-hook
- V89 小表 bit-exact 非真实行
- V91 无 memory vs disk A/B
- V92 Python 热路径未 native 化

## 4. 下一步
- Track 1：找 Qwen4Exp 环境 + 稀疏真实行 oracle，把 B1/B2 推到真实可信。
- Track 2：异步预取 + Rust 热路径。
- Track 3：真实 memory vs disk A/B。
- Track 4：serving / 推理引擎。
- Track 5：工程稳定 + 版本收编。

## 5. 本轮纪律
1. 先可信，再性能，再服务。
2. 性能结论必须带 hit-rate 和分段计时。
3. “异步”必须证明真的 overlap。
4. 跨仓正确性只走契约 + golden。
5. 版本、文档、代码同点收编。

# Session 28 系统性思考（第十六轮：预取管线落地 + 真实行低资源验证）

## 1. 本轮定位
- 低资源真实行验证完成：checkpoint ↔ Store ↔ DiskPle bit-exact。
- 性能关键路径开始落地：GIL 释放、prefetch/future、模型级 pre-hook、真实 Store 微基准。
- 仍未到最终性能验收：真实 full-model A/B、Rust 热路径、serving。

## 2. 本轮发现
1. 小资源也能验证真实行：9 token / 144 行即可 maxdiff=0.0。
2. PyO3 `Store.fetch` 释放 GIL 后，同一 Store 可被多线程并发读。
3. `DiskPle.prefetch()` 已能掩盖模拟计算窗口：192ms → 34ms。
4. 但微基准不能替代真实模型；Python 热路径和 prefetch 生命周期仍是下一步风险。
5. 完整 full-model 只应作为最终内存/性能 gate。

## 3. 新增技术债
V99–V111，详见 `docs/roadmap.md` Section 20.4。核心：
- V99 未在真实模型验证预取收益
- V100 prefetch 等待策略未生产化
- V101 Python 热路径可能成新瓶颈
- V104 无真实 memory vs disk A/B
- V105 无 hit-rate/等待分布

## 4. 下一步
- Track 1：真实模型预取 A/B
- Track 2：prefetch 生产化
- Track 3：Rust/PyO3 热路径
- Track 4：真实 memory vs disk A/B
- Track 5：服务化 / 推理引擎
- Track 6：工程稳定

## 5. 本轮纪律
1. 微基准不是最终结论。
2. 异步必须测量真实 wait/hit-rate。
3. Python 热路径要同步评估。
4. 小资源验证优先，全模型只做最终 gate。
5. 跨仓正确性只走契约 + golden。
6. 版本、文档、代码同点收编。

# Session 28 综合整理（Consolidated Round Review）

## 1. 本轮计划

1. 用小资源完成真实 PLE 行验证：
   - 稀疏真实行 oracle
   - 真实 checkpoint ↔ Store-I → DiskPle 位级对比
2. 进入性能关键路径：
   - PyO3 `Store.fetch` 释放 GIL
   - Store 支持并发读取
   - `DiskPle.prefetch()` + future/wait
   - 模型级 forward pre-hook
   - 真实 Store 上做 prefetch micro A/B
3. 把发现、技术债、后续计划整理进 roadmap / session 文档。

## 2. 发现

1. **PLE 行只依赖 token ids**，不依赖 hidden states，因此预取理论可行。
2. **当前 PyO3 `Store` 原本 `unsendable`，`fetch` 持 GIL**，这是异步化的第一个阻塞。
3. **去掉 `unsendable` + `allow_threads` 后，同一个 Store 可被多线程并发 fetch**，且性能正确。
4. **稀疏真实行 oracle 可以绕过全表加载**：
   - 9 个 token
   - 144 个真实 PLE 行
   - checkpoint ↔ Store-I byte-identical
   - DiskPle real-Store dequant maxdiff=0.0
5. **预取 micro A/B 能看到明显收益**：
   - sync ~192ms
   - prefetch ~34ms
   - 在模拟 30ms 计算窗口下，磁盘 fetch 基本被掩盖
6. **但 micro 不等于真实模型**：
   - 当前用 `time.sleep` 模拟前面层计算
   - 没有真实 PLE 层到达时间 / prefetch wait 分布
7. **磁盘被掩盖后，Python 热路径可能成为新瓶颈**：
   - rowid、列表转 tensor、FP8 dequant、flatten 都在 Python。
8. **官方 Qwen4Exp 完整模型仍受环境限制**，但正确性不需要它作为前置。

## 3. 做的尝试

1. 修改 PyO3 `Store`：
   - 去掉 `unsendable`
   - `fetch` 用 `py.allow_threads` 包裹 `BadgeGather::gather_pp`
2. 重构 `DiskPleEmbedding`：
   - 增加后台 `ThreadPoolExecutor`
   - 增加 `_pending` / `_pending_rows`
   - 增加 `prefetch()`
   - 增加 `_settle_prefetches()` 消费后台结果
   - 支持 cache 和 no-cache 两种模式
3. 给 `DiskPleNGramEmbedding` 增加 `prefetch(input_ids)`：
   - 不修改内部 context
   - 复用当前每 batch history 计算 rowid
4. 在 `official_loader` 增加 `install_disk_ple_prefetch_hook()`：
   - 模型级 pre-hook
   - `install_disk_ple_in_official_model(..., prefetch=True)` 自动启用
5. 新写 `sparse_real_row_oracle.py`：
   - 从原始 safetensors 按字节偏移只读命中的真实 FP8 行
   - 与 Store fetch 对比
   - 与 DiskPle dequant 输出对比
6. 新写 `prefetch_real_ab.py`：
   - 真实 Store
   - sync vs prefetch + 模拟 compute window
7. 增加 smoke/tests：
   - Store 并发 fetch
   - DiskPle prefetch
   - 原有 python wheel smoke
   - qwen35 phase B tests

## 4. 踩过的坑

1. **`py.allow_threads` 闭包不能 move `out`**：
   - 第一次直接把 `out` mov进闭包，编译错误；
   - 改为闭包捕获 `&mut out`，闭包结束后仍可使用 `out`。
2. **`Store` 原本 `unsendable`**：
   - 直接放后台线程会受限；
   - 需要确认 `BadgeGather` 是 Send/Sync 后去掉 `unsendable`。
3. **no-cache 模式下 prefetch 结果会被重复 fetch**：
   - 如果 `cache_size=0`，`_settle_prefetches` 不落 cache；
   - 后来增加 `_prefetched` 缓冲区，专门消费后台预取结果。
4. **微基准可能受 OS 页缓存影响**：
   - 先跑 sync 会预热，后续 prefetch 的 `fetch_s` 可能偏小；
   - 正式 A/B 需要冷/热分离并记录介质。
5. **预取微基准不是真实模型**：
   - `time.sleep` 不是前面层计算；
   - 不能据此宣布满足性能契约。
6. **本地 Transformers 没有 Qwen4Exp**：
   - 完整官方模型仍不能在本机跑；
   - 因此用冻结官方快照 + mini 模型 + 稀疏真实行组合验证。
7. **Python 热路径容易在优化后浮现**：
   - 磁盘 I/O 被隐藏后，不能忽略 Python 侧转换成本。

## 5. 完成的内容

### 正确性/真实行
- [x] 稀疏真实行 oracle：144 行 byte-identical
- [x] DiskPle 读真实 Store 与 checkpoint 行 dequant maxdiff=0.0

### 性能基础
- [x] PyO3 `Store.fetch` 释放 GIL
- [x] Store 并发 fetch smoke
- [x] `DiskPleEmbedding.prefetch()` + future/wait
- [x] 无 cache prefetch
- [x] `DiskPleNGramEmbedding.prefetch()`
- [x] 模型级 forward pre-hook
- [x] 官方完整加载路径默认启用 prefetch
- [x] 真实 Store prefetch micro A/B

### 测试/文档
- [x] python wheel smoke 通过
- [x] qwen35 phase B tests 3 passed
- [x] docs 更新：roadmap Section 20、session-summary、session-log、handoff

## 6. 未完成的内容

- [ ] 真实官方 Qwen4Exp 完整模型加载/forward
- [ ] 真实模型 sync vs prefetch A/B
- [ ] 记录真实 PLE 层到达时间、prefetch wait 分布、hit-rate
- [ ] Rust 原生 rowid + gather + dequant 热路径
- [ ] prefetch executor 生命周期 / shutdown / 超时
- [ ] Store 连接池 / 服务化
- [ ] MTP / Transformers Cache 集成
- [ ] memory vs disk 真实性能 A/B
- [ ] vLLM / SGLang / llama.cpp serving A/B
- [ ] `engramdb:view` 自动消费
- [ ] 可复现环境固化
- [ ] 三仓库版本 bump + README 同点收编
- [ ] runtime/官方类 CI 或 nightly

## 7. 未来计划

按风险/可信度排序：

1. **真实模型预取 A/B**
   - 真实 PLE 层
   - 记录 wait/hit-rate
   - sync vs prefetch tok/s
2. **Prefetch 生产化**
   - executor 生命周期
   - 超时 / 回退 / 去重
   - 统计增强
3. **Rust/PyO3 原生热路径**
   - native rowid + gather + dequant
   - 减少 Python 开销
4. **真实 memory vs disk A/B**
   - CSV + 阈值
5. **服务化 / 推理引擎**
   - Store 连接池
   - vLLM / SGLang / llama.cpp
6. **工程稳定**
   - 可复现环境
   - 版本收编
   - CI/nightly

## 8. 当前状态

```text
EngramDB   master 03cbf41（Session 28 系统性思考）
qwen35-ple main  5250582（sparse oracle + prefetch AB + docs）
真实行验证：SPARSE_REAL_ROW_ORACLE_OK
预取微基准：PREFETCH_AB_SMOKE_OK
官方完整模型：仍未实机
```

## 9. 本轮纪律

1. 微基准不是最终结论，必须上真实模型。
2. 异步必须测量真实 wait/hit-rate。
3. 磁盘被隐藏后，Python 热路径要同步评估。
4. 小资源验证优先，全模型只做最终 gate。
5. 跨仓正确性只走契约 + golden。
6. 版本、文档、代码同点收编。

# Session 29 增量（Prefetch 生产化起步 + Mini 官方模型 A/B）

## 1. 完成
- `DiskPleEmbedding.close()`：幂等关闭 prefecth executor，清洗 pending 状态。
- `DiskPleNGramEmbedding.close()`：转发关闭底层 disk table。
- `DiskPleNGramEmbedding.prefetch()` 返回底层 future 并保存 `_last_prefetch_future`，供 A/B 观测。
- `install_disk_ple_prefetch_hook()` 兼容 PyTorch 两种 pre-hook 签名（`(module, args)` 与 `(module, args, kwargs)`）。
- 新增 `/tmp/qwen35-ple-dev/scripts/mini_official_prefetch_ab.py`：
  - 冻结官方 `Qwen4ExpTextPLELayer` + 真实 Store + dense 前后块；
  - mini 官方模型级 prefetch hook；
  - 记录 wall / earlier / ple / post / prefetch_wait / fetch / ready-at-PLE；
  - 可输出 CSV。
- qwen35-ple docs 记录 mini A/B 入口。

## 2. 结果
- `scripts/python_wheel_smoke.py` 全绿（含 `DiskPleEmbedding.close()` 断言）。
- `tests/test_phase_b_official_loader.py` 3 passed。
- Mini A/B 脚本可跑通：`MINI_OFFICIAL_PREFETCH_AB_OK`。
- 本机小规模运行受系统调度/页缓存影响波动较大，尚未作为正式性能结论。

## 3. 未完成 / 下一步
- 仍需要完整 Qwen4Exp 或足够大的可运行 mini 官方模型做稳定 A/B。
- 需要冷/热分离、固定 seed、多重复、CSV 阈值。
- Prefetch 超时/错误回退/多模块合并去重仍未做。
- Python 热路径仍未 native 化。

## 4. 20k 预计算慢路径修复

- 旧 `PleDiskGather.fetch` 的 Python 去重/字节切片/join 是 20k token 预计算瓶颈。
- 修复：
  - `PleDiskGather.fetch` 改为直接返回 `Store.fetch` 的连续缓冲区；
  - 新增 `engramdb.fetch_e_t_tensor()`：一次 `Store.fetch` + `torch.frombuffer` 返回 `[T,16,160]` tensor；
  - `PleDiskGather.fetch_tensor()` 提供同类快速路径；
  - `qwen35_ple.real_ple.fetch_e_t` 与 `precompute_real_ple_features.py` 切换到该路径；
  - `run_phase0.py --live-store` 支持直接从 Store 读取 PLE 行，不必先写 `e_t.npy`。
- 同时新增 Rust `rowids_for_seq_with_history`（含 PyO3 导出）并让标准真实 PLE adapter 在可用时走 native rowid。
- Smoke：`python_wheel_smoke.py` 全绿；`cargo test -p engramdb-keygen` 4 passed；qwen35 小规模 precompute 跑通。

## 5. Session 30 系统性思考（第三轮系统复盘）

- **终极目标不变**：磁盘优先的确定性 n-gram 记忆表基础设施，像 DuckDB 之于分析数据库。
- **本轮最重要的位置判断**：慢的不是存储核心，而是 Python 适配层；修复后，下一个风险是“磁盘被隐藏后 Python serving 热路径再次成为瓶颈”。
- **新增技术债 V112–V117**：
  - V112 serving 仍走 Python bytes dict/join；
  - V113 没有正式 live-store bench harness；
  - V114 没有冷热分离门禁；
  - V115 多 PLE/多 outstanding 未合并去重；
  - V116 full-model 未实机；
  - V117 三仓库版本/README/retest 未完全同点收编。
- **借鉴矩阵**：DuckDB/SQLite/Redis/RocksDB/DiskANN/vLLM/SGLang/llama.cpp/Arrow/MLPerf/engram-peft，均只借“方法、形态、工程纪律”，不借模型内核或查询/KV语义。
- **后续计划重排**：
  1. Track 0 先做“测得准”：固定输入、冷/热、CSV、阈值。
  2. Track 1 消除剩余 Python serving 热路径。
  3. Track 2 prefetch 生产化。
  4. Track 3 真实模型性能验证。
  5. Track 4 服务化/引擎接入。
  6. Track 5 工程稳定性。
- 详见 `docs/roadmap.md` Section 21。

## 6. Session 31：Track 0/1/2 推进记录

- **Track 0**：
  - qwen35 新增 `scripts/bench_live_store.py`，支持固定 tokens、warmup、reps、CSV、中位数；
  - 新增阈值门禁：`--max-store-s` / `--max-tensor-s` / `--max-tensor-dedup-s`，可返回 `LIVE_STORE_BENCH_OK/FAIL`。
- **Track 1**：
  - `DiskPleEmbedding` no-cache forward 改用直接 `Store.fetch` 连续读取；
  - 跳过 Python per-row dict/join；
  - bit-exact 小表仍 0.0；
  - cache>0 路径仍待进一步优化。
- **Track 2**：
  - prefetch 错误会回退到同步；
  - 支持可选 `prefetch_timeout`；
  - `prefetch_executor` 可跨模块共享；
  - 新增 `get_wait_distribution()` 返回 p50/p90/p99/max；
  - 多 PLE 行级合并去重仍未完成。
- Smoke：`python_wheel_smoke.py` 全绿；qwen35 phase B 3 passed；bit-exact small 通过。

## 7. Session 32：懒加载 live-store 与磁盘优先路径确认

- WSL 实测确认：全量 `--live-store` 1M e_t 会 OOM（约 10GB）。
- 实现 `LiveETStore` / `LiveETView`：
  - 只保留 `[T,16]` rowids；
  - 每个训练/评测窗口按需从 Store 读取；
  - control 模式也支持懒加载置换；
  - no-reader / real / control 均可跑。
- 结论：**推荐用法是磁盘优先 + 按窗口 lazy fetch，不是全量加载 e_t。**
- 新债：V118 WSL 随机 IO 慢、V119 懒加载需抽象为通用 Dataset、V120 Store-P WSL 未验证、V121 无 1M lazy 基准、V122 Store 连接生命周期。
- 下一步：将懒加载提炼为正式数据流，并在 WSL 做 Store-P / 多线程 / Rust 批量同口径 A/B。
- 详见 `docs/roadmap.md` Section 22。

# Session 32 综合整理（Consolidated Round Review）

## 1. 本轮计划

1. 完成 Track 0：可复现 live-store benchmark + CSV + 阈值。
2. 完成 Track 1：减少 serving/training Python 热路径。
3. 完成 Track 2：prefetch 错误回退、超时、共享 executor、wait 统计。
4. 根据 WSL 实测，把 `--live-store` 从“全量加载 e_t”改为“按窗口懒加载”。
5. 更新 README/pyREADME/retest 指南，反映 v0.2.9+ 新 API 和推荐使用方式。
6. 再次系统性思考，把终极目标、技术债、借鉴矩阵、后续计划写入 roadmap。

## 2. 本轮发现

1. **全量加载 e_t 是反模式**：
   - 1M × 2560 × 4B ≈ 10GB；
   - WSL 15GB 内存会 OOM；
   - 违背 EngramDB 磁盘优先设计本质。
2. **懒加载是正确形态**：
   - 只保留 `[T,16]` rowids（约 128MB / 1M token）；
   - 每个训练/评测窗口按需读取；
   - 内存从 10GB 降到 KB/MB 级。
3. **懒加载解决内存，不解决 WSL IO**：
   - 100k token / 1.6M 行 ≈ 56s；
   - 原始 Store-I 随机 scatter 在 WSL 虚拟盘上仍是瓶颈；
   - 必须靠 Store-P / Rust/C 批量 / 多线程 / 顺序化解决。
4. **Python 适配层不再是主要慢路径**：
   - `PleDiskGather.fetch` 已改为直接 `Store.fetch`；
   - `fetch_e_t_tensor` 提供一次 fetch + torch 转换；
   - no-cache `DiskPleEmbedding.forward` 也走连续 buffer 快路径。
5. **prefetch 可以更健壮**：
   - 错误回退到同步；
   - 可选超时；
   - 共享 executor；
   - wait 分布可测量。

## 3. 做的尝试

1. 发布 v0.2.9，包含：
   - `PleDiskGather.fetch` 直连化；
   - `fetch_e_t_tensor` / `PleDiskGather.fetch_tensor`；
   - native rowid + history + PyO3 导出；
   - README/retest 指南同步。
2. qwen35 `bench_live_store.py`：
   - 固定 token/rowids；
   - warmup/reps；
   - CSV/中位数/阈值；
   - `LIVE_STORE_BENCH_OK/FAIL`。
3. `DiskPleEmbedding` 生产化：
   - no-cache fast path；
   - prefetch 错误回退；
   - `prefetch_timeout`；
   - `prefetch_executor` 共享；
   - `get_wait_distribution()`。
4. `run_phase0.py --live-store` 懒加载改造：
   - 新增 `LiveETStore` / `LiveETView`；
   - 不预加载完整 e_t；
   - 训练/评测按窗口读取；
   - control 支持 lazy permutation。
5. README 更新：
   - 根 README 增加快速 e_t / prefetch / 状态行；
   - python README 修复“Store 是 unsendable”的过时说明；
   - qwen35 README 更新 live-store 推荐方式。

## 4. 踩过的坑

1. **FP8 tensor 不能直接 CPU 索引**：
   - `fetch_e_t_tensor(dedup=True)` 直接 `batch[idx]` 报 `index_cpu not implemented for Float8_e4m3fn`；
   - 修复：先 `.to(float32)` 再索引。
2. **`fetch_e_t_tensor` 的 flat rowids 语义是 T×16**：
   - 返回 shape 应为 `[T,16,160]`，不是 `[len(rowids),16,160]`；
   - 否则 reshape 会报错。
3. **PyTorch pre-hook 签名差异**：
   - 有些版本只传 `(module, args)`，不传 `kwargs`；
   - 修复为 `kwargs=None` 默认兼容。
4. **全量 live-store OOM**：
   - 1M e_t 在 WSL 触发系统杀进程；
   - 改为懒加载后解决。
5. **WSL Store.fetch 慢不是 Python 层**：
   - 100k token ≈ 56s；
   - 是随机 IO/介质/Store-I 路径问题，不能靠全量内存绕开。
6. **关闭顺序**：
   - `DiskPleEmbedding.close()` 应在 Store close 之前调用；
   - 否则后台 prefetch future 可能在 Store 关闭后读取。

## 5. 完成的内容

### 发布
- [x] EngramDB v0.2.9 发布并推送 tag。
- [x] Mac/WSL 验收通过。

### 快速读取路径
- [x] `PleDiskGather.fetch` 直接 `Store.fetch`。
- [x] `engramdb.fetch_e_t_tensor()`。
- [x] `PleDiskGather.fetch_tensor()`。
- [x] qwen35 `real_ple.fetch_e_t` / precompute 切换到快路径。
- [x] `run_phase0.py --live-store` 懒加载。

### 热路径/生产化
- [x] `DiskPleEmbedding` no-cache fast path。
- [x] prefetch 错误回退。
- [x] 可选超时。
- [x] 共享 executor 参数。
- [x] wait 分布统计。
- [x] native rowid + history（Rust + PyO3）。

### 基准/文档
- [x] `bench_live_store.py` + 阈值。
- [x] README / pyREADME / qwen README 更新。
- [x] retest 指南 + 验收记录。
- [x] roadmap Section 22 系统性思考。

## 6. 未完成的内容

- [ ] 多 PLE 模块行级合并去重。
- [ ] `cache_size > 0` 下 `DiskPleEmbedding.forward` 的 Python bytes join 优化。
- [ ] Rust/PyO3 native gather + FP8 dequant + flatten。
- [ ] WSL Store-P 视图构建与 A/B。
- [ ] WSL 多线程 / C-Rust 批量读取实测。
- [ ] 1M token 懒加载正式基准（每 step fetch 时间/CSV）。
- [ ] 通用 `LiveETDataset` / `IterableDataset` 抽象。
- [ ] 完整 Qwen4Exp 官方模型加载/性能验证。
- [ ] Store 连接池 / 服务化。
- [ ] vLLM / SGLang / llama.cpp serving A/B。
- [ ] MTP / Transformers Cache 集成。
- [ ] `engramdb:view` 自动消费。
- [ ] 三仓库版本/README/CI 完全同点收编。

## 7. 未来的计划

### Track A：正式懒加载数据流
- 把 `LiveETStore` / `LiveETView` 提炼为通用模块或 EngramDB Python API。
- 实现 `IterableDataset` / 多 worker / 分片。
- 记录每窗口 fetch 时间。

### Track B：WSL/真实介质性能闭环
- WSL 构建 Store-P。
- Store-I vs Store-P vs 懒加载 vs 全量内存同口径 A/B。
- Rust/C 批量 + 多线程实测。
- CSV + 阈值。

### Track C：真实模型训练/推理验证
- 1M token real/control/seeds 懒加载实验。
- 对比 loss/耗时。
- 推进 CPU 100 tok/s 推理闭环。

### Track D：服务化
- Store 连接池。
- vLLM / SGLang / llama.cpp serving A/B。
- Arrow IPC 数据流。

### Track E：工程稳定
- 可复现 WSL 环境。
- 三仓库版本/README/retest 同步。
- live-store smoke 入 CI/nightly。

## 8. 当前状态

```text
EngramDB master 23b059f
qwen35-ple main 4295584
v0.2.9 已发布
快速读取路径 ✅
懒加载 live-store ✅
WSL Store-P / 大规模性能 ❌
完整模型端到端 ❌
```

## 9. 本轮纪律

1. 内存不是用来替代磁盘的。
2. 懒加载解决内存，不代表解决 IO。
3. 所有大规模实验要记录 per-window fetch 时间。
4. 真实介质上的 Store-P 必须实测，不能外推。
5. 小资源正确性继续闭环，完整模型只做最终 gate。
6. 版本、文档、代码、retest 指南同点收编。


## Session 33 综合整理（Track A 通用懒加载数据流）

### 1. 本轮计划

- 把 `LiveETStore` / `LiveETView` 从 `run_phase0.py` 提炼为 qwen35-ple 通用模块。
- 实现 `IterableDataset` / 窗口级 reader，支持 control、分片、多 worker。
- 记录 per-batch fetch 时间、总读取量。
- 让任意实验脚本三行接入，不再全量加载 e_t。

### 2. 本轮发现

1. PyTorch DataLoader 多进程不能直接 pickle PyO3 `Store`，必须保存重开参数。
2. `LiveETDataset` 应同时支持 numpy 预计算数组（兼容旧路径）和 live Store。
3. 多 worker 分片可通过 `get_worker_info()` 自动完成，无需调用方手动传 worker id。
4. 懒加载抽象完成后，Track A 退出标准基本满足；性能瓶颈仍在 Storage I/O 层。

### 3. 做的尝试

- 新建 `qwen35_ple/live_store.py`，包含 `FetchStats` / `LiveETStore` / `LiveETView` / `LiveETViewStore` / `LiveETBatch` / `LiveETDataset`。
- `LiveETStore` 支持 pickle，worker 中重开 Store。
- `run_phase0.py` 改用统一模块。
- 新增冒烟脚本、Store-I vs Store-P A/B 脚本、懒加载逐窗口基准脚本和 9 个单元测试。
- 新增 EngramDB `StorePool` / `ThreadLocalStore`。
- 验证 `DataLoader(num_workers=2)` 在 tiny Store 上可跑。
- 更新 qwen35 README / roadmap / session-log。

### 4. 踩过的坑

1. **PyO3 Store 不可 pickle** → `LiveETStore.__getstate__/__setstate__` 保存并重开。
2. **`__len__` 与 `_window_starts()` 不一致** → 统一为 `(n-seq_len)//step+1`。
3. **macOS spawn 必须 `if __name__ == "__main__"`** → 冒烟脚本加 main guard。

### 5. 完成的内容

- [x] `src/qwen35_ple/live_store.py`（含 Store-P `LiveETViewStore`）
- [x] `run_phase0.py` 去除内置 LiveET 类
- [x] `scripts/run_live_et_dataset_smoke.py`
- [x] `scripts/bench_store_vs_view.py`（Store-I vs Store-P A/B 骨架）
- [x] `scripts/bench_lazy_windows.py`（懒加载逐窗口基准）
- [x] `engramdb.pool.StorePool` / `ThreadLocalStore`
- [x] `tests/test_live_store.py`（9 tests）
- [x] README 三行接入示例
- [x] qwen35 与 EngramDB roadmap / session-log 更新

### 6. 未完成的内容

- [ ] Track B：WSL Store-P 构建与 A/B。
- [ ] Track C：1M token 懒加载正式实验与每窗口 CSV。
- [ ] Track D/E：连接池、服务化、CI nightly live-store smoke。

### 7. 当前状态

```text
EngramDB master 2b83abe + StorePool/Database pool
qwen35-ple main 80cb9f7 + Track A/B/C bench
v0.2.9 已发布
通用 LiveETDataset ✅
DataLoader 多 worker ✅
StorePool / ThreadLocalStore ✅
Store-P 懒加载 ✅（本机 1M 约 7.1s）
WSL Store-P A/B ✅（p4view 20k/100k, 8t ~22M rows/s）
WSL Python 懒加载 ✅（20k Store-I 22.4s / 100k Store-P 1.9s / 1M Store-P 23.9s）
WSL serving A/B / 完整模型训练 ❌
```

## Session 34 综合整理（系统性思考：从 I/O 快走向端到端实验）

### 1. 本轮计划

1. 做第二十轮系统性思考：明确终极目标、当前坐标、技术债、借鉴矩阵、开发计划。
2. 继续推进 Track B/C/D/E：
   - WSL Store-P 构建与 A/B；
   - WSL Python 懒加载实测；
   - StorePool / 多 worker；
   - 发布 v0.2.10。
3. 开始 P0：把 Store-P 从“raw slot 基准”推进到“语料语义访问路径”：
   - 构建 access-order Store-P 视图；
   - 让 `run_phase0.py` 可直接使用 Store-P；
   - 为完整模型 1M 实验铺路。

### 2. 本轮发现

1. **Store-I 随机读是唯一真正的存储瓶颈**：
   - WSL 20k Store-I lazy 22.4s；
   - 100k Store-P lazy 1.86s；
   - 1M Store-P lazy 23.9s。
2. **Store-P 比 Store-I 快约两个数量级**：
   - 本机 100k Store-I 60.5s vs Store-P 0.58s；
   - WSL p4view Store-P 8t 约 22M rows/s vs Store-I 约 1.4M rows/s。
3. **访问序是关键**：
   - 本机 1M Store-P 顺序 7.1s；
   - permuted/control 三 seed 17.2–17.9s，随机惩罚约 2.4×。
4. **多 worker 可行**：
   - `LiveETViewStore` pickle 重开 View 后，WSL `DataLoader(num_workers=2)` 可跑。
5. **v0.2.10 可发布**：
   - StorePool / ThreadLocalStore / Database 池化 / README / gate 修复全部通过。
6. **access-order Store-P 语义映射可以做到精确**：
   - 用 `engramdb view build --keys` 按语料 rowid 顺序建视图；
   - 验证 `LiveETViewStore` 与 `LiveETStore` 同一批 token 的 e_t `maxdiff=0.0`。

### 3. 做的尝试

- 在 WSL 用 `p4view` 构建 20k / 100k / 1M Store-P 视图并跑 A/B。
- 在 WSL qwen35 venv 安装 `engramdb-python==0.2.9`，同步 `live_store.py` 与 benchmark 脚本。
- 跑 WSL 20k Store-I / 100k Store-P / 1M Store-P 懒加载。
- 跑 WSL Store-P 2 worker DataLoader。
- 新增 `engramdb.pool.StorePool` / `ThreadLocalStore`，`Database.fetch` 改用池。
- 新增 `scripts/build_corpus_store_p_view.py`：
  - 根据 tokens 生成 rowids；
  - 写 flat keys；
  - 调 `engramdb view build --keys` 构建 access-order Store-P；
  - 输出 `slot_indices.npy`（slot i = token i）。
- 为 `LiveETViewStore` 增加 `view()` 切分，并让 `run_phase0.py --store-p-view` 可直接走 Store-P 训练路径。
- 发布 v0.2.10，tag 已推送。

### 4. 踩过的坑

1. **WSL 原有 `engramdb` 二进制不支持 `--keys`**：
   - `build_corpus_store_p_view.py` 在 WSL 首次运行失败；
   - 需要先用新源码构建 `engramdb view build --keys` 版本。
2. **`LiveETViewStore` 实例属性 `self.view` 遮蔽了同名方法 `view()`**：
   - `view.view(...)` 实际调用 View 对象，报 `'FakeView' object is not callable`；
   - 修复：内部属性改为 `self._view`，保留 `view()` 方法供 `_split` / 训练分隔使用。
3. **`p4view bench` 旧命令缺 `--keys`**：
   - `scripts/gate.sh` 传入的 keys 文件被当成未知位置参数；
   - 修复为 `--keys probes/view-keys-20k.txt`。
4. **WSL qwen35 venv 原本是损坏的 Mac 符号链接**：
   - 使用 `uv pip install --python ... engramdb-python==0.2.9` 重建可用的 Python 环境。
5. **WSL 全量 pytest 存在 golden 漂移**：
   - 非本仓新增代码导致 1 个官方 golden 失败；
   - V126 记录为待修，不可当作“当前全绿”。

### 5. 完成的内容

- [x] v0.2.10 发布并推送 tag。
- [x] `StorePool` / `ThreadLocalStore` / `Database` 池化读取。
- [x] WSL Store-P p4view A/B（20k / 100k）。
- [x] WSL Python 懒加载（20k Store-I / 100k Store-P / 1M Store-P）。
- [x] Store-P 多 worker DataLoader 验证。
- [x] `LiveETViewStore` pickle 与 `view()` 切片。
- [x] access-order Store-P 构建脚本 `build_corpus_store_p_view.py`。
- [x] 本机验证 access-order Store-P 与 Store-I e_t `maxdiff=0.0`。
- [x] `run_phase0.py --store-p-view / --store-p-slot-indices` 接入训练入口。
- [x] V123：通用 rowid-tuple → full Store-P slot 语义索引（`SlotIndex` / `--slot-index-out` / `--store-p-slot-index`）。
- [x] V124：access-order 视图 + LiveETDataset 自动访问序调度（`access_order=True` / `--access-order`）。
- [x] README / roadmap / session-log / handoff 更新。
- [x] release gate 全绿。

### 6. 未完成的内容

- [ ] V125：真实模型 1M real/control/3-seed loss 实验。
- [ ] V126：WSL golden 漂移修复。
- [ ] V127：vLLM / SGLang / llama.cpp serving A/B。
- [ ] V128：懒加载基准固化为正式门禁 / CI 阈值。
- [ ] V129：StorePool 与 LiveET/DataLoader 深度集成与 wait 统计。
- [ ] V130：Arrow IPC 在 Mac/WSL 实际验证。
- [ ] V131：WSL 复现环境脚本化。
- [ ] V132：WSL 全表 Store-P 分批构建策略。

### 7. 未来的计划

#### Phase 0：把“读取快”变成“实验能跑”

- [x] 通用 rowid-tuple → Store-P slot 索引/manifest。
- [x] access-order Store-P 视图 + 访问序调度。
- [ ] WSL 真实模型 1M real/control/3-seed，同时记录 loss + fetch 时间。

#### Phase 1：把基准变成门禁

- 固化 20k/100k/1M 懒加载 CSV + 阈值。
- WSL 复现脚本。
- golden 漂移修复。
- live-store smoke / StorePool smoke 入 CI / nightly。

#### Phase 2：服务化

- vLLM / SGLang / llama.cpp 薄 adapter + A/B。
- Arrow IPC 端到端。
- StorePool 与训练/推理深度集成。

#### Phase 3：产品化收口

- WSL 全表 Store-P 分批构建与校验。
- 三仓库版本/README/retest/CI 完全同步。
- 根据 1M 结果决定是否进入 5M–20M token。

### 8. 当前状态

```text
EngramDB v0.2.10 (tag pushed)
qwen35-ple P0 代码已完成（语义索引 + 自动访问序调度）
StorePool / ThreadLocalStore ✅
WSL Store-P A/B ✅
WSL 1M Store-P lazy 23.9s ✅
access-order Store-P 语义视图 ✅
通用 rowid→slot 语义索引 ✅
自动访问序调度 ✅
完整模型 1M 三线实验 ⏳（由 qwen35-ple/WSL 侧继续）
serving / Arrow / 全表 Store-P ❌
```

### 9. 本轮纪律

1. 磁盘优先不是“全量内存的替代品”，Store-P + 访问序才是完全体。
2. I/O 基准不是实验结论，只有真实模型 loss / tok/s 能支持科学判断。
3. 性能结论必须包含介质、冷热、并发、访问序、CSV/阈值。
4. 跨仓正确性靠版本固定 + golden，不靠“本地能跑”。
5. 先打通端到端最小闭环，再谈 5M/20M 放大。
6. 发布前 release gate 必绿，版本只走 bump.sh。

---

## Session 35（第二十一轮：P0 语义索引 + v0.2.11 发布）

### 1. 本轮完成

- [x] 通用 rowid→slot 语义索引：`qwen35_ple.slot_index.SlotIndex` + `engramdb.SlotIndex`。
- [x] access-order 自动调度：`LiveETViewStore(access_order=True)` + `LiveETDataset(access_order=True)`。
- [x] `build_corpus_store_p_view.py` 自动输出 `*.slot_index.npz` 并写入 view manifest。
- [x] `run_phase0.py --store-p-slot-index` / `--access-order` 接入。
- [x] 新增 6 个 SlotIndex/access-order 测试；qwen35-ple 全量 25 passed。
- [x] 发布 **v0.2.11**，release gate 全绿，tag 已推送。

### 2. 本轮发现的技术债

| # | 债 |
|---|---|
| V133 | SlotIndex 全表 320M 无法纯内存承载 |
| V134 | SlotIndex 在两仓重复实现，需统一 canonical |
| V135 | `engramdb view build` 未原生生成 slot index |
| V136 | access-order 调度缺正式 A/B 与门禁 |
| V137 | numpy 依赖/降级语义未完全理清 |
| V138 | `access_order` 窗口重排对训练顺序敏感实验的语义未单独建模 |
| V139 | 两仓缺交叉 contract test |

### 3. 下一阶段

1. Phase A：真实模型 1M 三线实验（最高优先科学门禁）。
2. Phase B：SlotIndex 产品化/磁盘化 + 统一实现 + CLI manifest。
3. Phase C：access-order 基准/门禁 + WSL 复现/golden。
4. Phase D：serving/Arrow/全表 Store-P。
5. Phase E：依赖/CI/跨仓治理。

详见 `docs/roadmap.md` Section 25。

---

## Session 36（第二十二轮：Phase A 科学闭环 + DiskSlotIndex + 全表工具）

### 1. 本轮完成

- [x] WSL 1M real/control/no-reader 3-seed 已核验并固化：real < control < no-reader，Go。
- [x] `DiskSlotIndex`：分桶磁盘索引、流式构建、LRU、`build_from_keys_file`。
- [x] `engramdb view build --keys-stream` + `build_full_store_p_batch.py` 全表批式构建/断点/校验。
- [x] `bench_access_order.py` / `bench_lazy_windows.py` 合成门禁入 CI。
- [x] `StorePool.stats()` 遥测。
- [x] qwen35-ple 跨仓 contract test：DiskSlotIndex 与内存 SlotIndex 一致。
- [x] v0.2.11 已发布；Phase A 结果文档已加入 qwen 仓库。

### 2. 本轮新增技术债

| # | 债 |
|---|---|
| V140 | Phase A 未用 Store-P/access-order 复跑 |
| V141 | DiskSlotIndex 尚无 320M 级真表实测 |
| V142 | DiskSlotIndex 每 bucket 一个文件 |
| V143 | qwen 保留本地 SlotIndex fallback |
| V144 | EngramDB CLI 未原生生成/校验 slot index |
| V145 | Phase A JSON 无 fetch timing |
| V146 | WSL golden 漂移未修复 |
| V147 | CI 只有合成性能门禁 |
| V148 | 新功能尚未发布到下一版本 |

### 3. 下一阶段

- Phase A2：Store-P/access-order 复跑 1M，记录 loss + fetch timing。
- Phase B2：DiskSlotIndex 全表实测 + 产品化 + CLI 原生索引。
- Phase C2：真表性能门禁 + golden 修复。
- Phase D2：Arrow / serving / 全表实际构建。
- Phase E2：v0.2.12 发布与三仓同步。

详见 `docs/roadmap.md` Section 26。






