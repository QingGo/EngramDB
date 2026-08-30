# 本 Session 综合整理（2026-08-30 后段）

> 本文件是对本轮主要工作的单一入口摘要。详细分 session 复盘见
> `docs/session-log.md` Session 8–14；战略复盘与债务表见
> `docs/roadmap.md` Section 10。

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
- `python/engramdb/server.py`：最小 TCP/JSON 服务
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
SERVICE_SMOKE_OK
```

## 4. 新发现的问题

| # | 问题 | 影响 |
|---|---|---|
| V8 | PyO3 `Store` 是 `unsendable` | 服务端不能跨线程共享，当前每请求新开 Store |
| V9 | 服务仍是 JSON + base64 | 不是真正二进制 Arrow IPC wire |
| V10 | GPU 路径被 torch/Pascal 兼容性卡住 | GTX1070 sm_61 无法使用当前 cu130/cu128 |
| V11 | 小文件冷读多线程反而更慢 | 冷读需要顺序流调度，不能盲目并行 |
| V12 | 多表/服务只是 Python 原型 | Rust CLI、manifest、table_id、serve 未收敛 |
| V13 | 自 v0.2.4 后有实质代码变更但未发布 | 新功能未进入 PyPI |
| V14 | 首未命中仍走 raw disk | LRU 只解决热路径，冷启动/预热/Tier 未做 |
| V15 | 模型 PLE 属性仍靠手填 | 需要自动发现或配置映射 |
| V4 | 完整 decode tok/s A/B 未闭环 | 最大悬空性能门禁 |

## 5. 计划要完成的部分

### 近期

1. 发布 v0.2.5，包含 LRU、多表、Arrow helpers、最小服务。
2. 扩展 Python wheel smoke，覆盖 Database / Arrow / server / LRU。
3. CPU 完整 serving 做 PLE decode A/B。
4. 尝试兼容 GTX1070 的 torch 构建做 GPU A/B。

### 中期

5. Rust 侧多表 table_id、manifest 完整性、`serve`。
6. 服务升级为二进制 Arrow IPC wire，解决线程安全句柄。
7. 冷读顺序流调度与 `StreamingPlanner` / Tier 预取打通。
8. 自动发现模型 PLE 属性。

### 长期

9. vLLM/SGLang 完整 serving 集成与性能 A/B（≤5%）。
10. 上游 patch / llama.cpp 文件格式 / C ABI。
11. 保持“不改上游源码”的薄层接入哲学。

## 6. 相关提交

```text
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
