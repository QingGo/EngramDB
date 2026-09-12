# Linux 开发环境（零到一：30 分钟指南）

本文面向**全新 Linux 机器**（含 WSL2）：从装 Rust 到第一帧可复现基准。

## 0. 前置要求
- 硬件：纯开发无需 GPU；跑大基准建议桌面 NVMe（外盘/HDD 的吞吐口径差异见 design §7 介质分层）
- 网络：能访问 github.com（国内：TUNA 镜像 + 代理见 §1）

## 1. 工具链与网络（一次搞定）

```bash
# Rust（rustup 官方脚本）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

# crates.io 走 TUNA sparse 镜像（本地网络快；发布时用 release.sh 自动绕过）
cat > ~/.cargo/config.toml << 'EOF'
[source.tuona]
replace-with = "rsproxy-sparse"
[registries.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
EOF
# 离线/防火墙内另见 docs — 或直接保留官方源（macOS 开发亦可）
```

### WSL2 专项坑（重要）
- **TLS/MTU**：WSL2 内 github/crates.io 握手失败时，修 MTU（root 一次）：

  ```bash
  # 在 Windows 侧（管理员 PowerShell 或 wsl 内）：
  wsl.exe -d Ubuntu -u root ip link set eth0 mtu 1400
  ```
- **长任务保活**：ssh 会话/后台会被 WSL 终止 → **必须 schtasks**（Windows 侧）：
  ```bat
  schtasks /create /tn <name> /tr "<C:\path\run.bat>" /sc once /st 23:59 /f
  schtasks /run /tn <name>
  ```
  结果从 WSL 内文件（`/tmp/xxx.log`）轮询。
- **Windows 原生 side**（仅当验证 Windows 目标）：`cargo check --target x86_64-pc-windows-msvc` 已在 CI 级通过；原生性能基准为可选（WSL/Linux 为主口径）。

## 2. 克隆 + 构建 + 验证

```bash
git clone https://github.com/QingGo/EngramDB.git && cd EngramDB
cargo test --workspace          # 全绿 = 17 tests（1+5+8+3）
cargo build --release -p engramdb
```

## 3. Mock 数据 + 全链 smoke（无需任何真实模型）

```bash
python3 scripts/prep_env.py quick        # 生成 mock 表（结构等价，uint8）
./target/release/engramdb build data/mock-qwen38-ple /tmp/edb-badged
./target/release/engramdb index /tmp/edb-badged
./target/release/engramdb warm /tmp/edb-badged
./target/release/engramdb bench-real /tmp/edb-badged --dist agent --iters 4
./target/release/engramdb prep --dist agent --reqs 4 --cap-token 200 /tmp/keys.txt
```

## 4. 真实数据接入（私有资产，源在 SA/ModelScope）

| 资产 | 路径（本机约定） | 来源 |
|---|---|---|
| Qwen PLE FP8 权重分片 | `data/qwen38-ple-fp8 -> /Volumes/My Passport/qwen38-ple`（53GB） | `Qwen/Qwen3.8-Flash-Next-FP8`（ModelScope） |
| 真表行存储 | `data/real-rows -> /Volumes/My Passport/qwen38-rows`（48GB, 128 shard） | 由权重生成（`engramdb view build` 所需） |
| 全表视图（可选） | `/Volumes/My Passport/p4view-full-2560.bin`（51.2GB + manifest） | `p4view build data/real-rows 20000096 <dst> <keys> --slot 2560` |

> 真表/视图重建命令清单：`probes/p4_view_notes.md` 顶部（T3 可重建性）。
> 语料（FineWeb-Edu/zh/agent 三域）：`data/corpus-build/` + `scripts/corpus_build.py`。

## 5. 复现基线（探针）

```bash
cargo build --release -p engramdb-bench --bin p4view
# 吞吐/延迟（真表 + 20K keys 固定输入）
target/release/p4view build data/real-rows 20000 /tmp/v.bin probes/view-keys-20k.txt --slot 2560
target/release/p4view bench data/real-rows /tmp/v.bin --sub 20000
target/release/p4view lat /tmp/v.bin --warm
# 门禁（结构 + 基准判据）
bash scripts/gate.sh
```

基线存档：`probes/baseline_view.csv` / `probes/baseline_latency.csv`（跨机型对照见其注释）。

## 6. 已知数据口径（写明，避免误读）
- **介质**：外盘/USB 吞吐与桌面 NVMe 差 35×（全表冷随机 8t：554K vs 19.2M rows/s）——性能口径必须注明介质
- **温/冷**：Linux 可用 `--cold`（fadvise drop）真冷；SSD 上冷/热差仅 1.85×
- **配置路径**：每次 `cargo build` 仅在 workspace 根；`target/release/engramdb` 为 CLI

## 7. 常见问题速查
| 现象 | 处置 |
|---|---|
| `gather_pp: index out of bounds` | 表分片与布局不匹配（真表=128 shard；mock 用 `--dir` 检查）|
| crates.io 429（发布时）| `release.sh` 已自动规避 TUNA；本地开发换官方源 |
| bench 数字漂移 | 固定 seed + keys 文件 + `--warm` 口径 + 介质标注 |

## PyPI 镜像选择：先量持续吞吐，别量索引页

**结论（2026-09-12，AutoDL 容器，100 MB 区段实测）**：

| 镜像 | 持续吞吐 |
|---|---|
| `mirrors.aliyun.com/pypi/simple` | **16.98 MB/s** |
| `pypi.tuna.tsinghua.edu.cn/simple` | **16.33 MB/s** |
| `mirrors.ustc.edu.cn/pypi/simple` | **0.56 MB/s** |
| `files.pythonhosted.org`（直连 PyPI CDN） | 0.09 MB/s |

**差 30 倍。** 配合 `uv` 的 8 路并发，实测 aliyun 达到 **42.7 MB/s**。

### 教训：第一次测量用的是错的样本

最初用「下载 1.7 MB 的 simple 索引页」比较镜像，TLS 握手与建连占了大部分时间，
于是选出了 **ustc（实际最慢）**，装 vLLM 花了 40 分钟。
**索引页不能代表持续吞吐。**

### 正确的测速方法

拿一个**真实的大 wheel**，下 100 MB 区段：

```bash
HREF=$(curl -s https://pypi.tuna.tsinghua.edu.cn/simple/nvidia-cublas/ \
  | grep -oE 'href="[^"]*manylinux[^"]*x86_64\.whl[^"]*"' | tail -1 | sed 's/href="//;s/"$//;s/#.*//')
REL=$(echo "$HREF" | sed 's#^\.\./\.\./##')        # 注意：tuna 的 href 是相对路径
curl -sL -o /dev/null -w '%{speed_download}\n' --max-time 50 -r 0-104857599 \
  "https://mirrors.aliyun.com/pypi/$REL"
```

### 附带发现

- `github.com` 从 AutoDL 容器**不可达**；`pypi.org` 与 `files.pythonhosted.org` 可达但极慢。
- **中断 `uv pip install` 会作废该轮的全部下载**：半成品留在 `$UV_CACHE_DIR/.tmp*`，
  不进入可用缓存。实测累积了 **5.4 GB 孤儿**。重装前应先 `rm -rf $UV_CACHE_DIR/.tmp*`。
- venv 与 `UV_CACHE_DIR` 应放**同一文件系统**（uv 文档：否则无法 link，退回拷贝）。
  *注意*：这条只影响链接性能，**不影响是否会重新下载** —— 曾误判为「跨盘导致重下」，
  同盘（xfs→xfs）实测缓存照样增长，该假设已被否证。
