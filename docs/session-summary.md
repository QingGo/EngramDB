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
