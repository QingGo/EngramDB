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
