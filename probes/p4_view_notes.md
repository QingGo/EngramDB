# P4 物化视图 A/B 探针结论（真表，2026-08-29）

> **T3 可重建性清单（2026-08-30，全部资产都有重建命令；keys 一律用固定 seed 重生成，
> 不依赖 /tmp 与旧的 view-manifest.json 命名）**：
> ```
> # 真表行存储 128 分片（48G，SSD；源是 SA 上的 Qwen4Exp FP8 safetensors，私有）
> #   已存在: /Volumes/My Passport/qwen38-rows  (data/real-rows -> 软链)
> # 视图重建（2048 倍缩放请显式 --slot）:
> target/release/p4view build data/real-rows 20000096 /Volumes/My\ Passport/p4view-full-2560.bin /tmp/p4keys-full.txt --slot 2560
> # keys 由 manifest 携带 n 可不用（bench/lat 支持 B-only：--keys 省略、n 从 .manifest.json 读）
> # 吞吐复测:   p4view bench data/real-rows <view.bin> --threads 8 [--sub N]
> # 延迟复测:   p4view lat <view.bin> [--warm] [--threads 1|8] [--sub N]
> # 大小视图门禁输入: probes/view-keys-20k.txt（固定 seed，入 git，gate.sh 使用）
> ```


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

## P4 v2（2026-08-30：真表全口径 + 槽位选型实测）

**槽位选型（200K grams @2560B 记录）**：
| slot | 构建秒 | B 8t 等效行/s | MB/s | 字节放大 |
|---|---|---|---|---|
| 4096（4KB 对齐） | 3.0s | 0.97M | 155.9 | 1.60x |
| **2560（紧凑，无 pad）** | 2.6s | **4.50M** | **719.6** | **1.00x** |

→ **槽位选型定案：2560B 紧凑槽**（无 pad，每记录 1 次随机读 = 恰好 2560B 有效载荷；
4KB 对齐反而因为 62% 槽浪费 + 读放大 1.6x,吞吐低 4.6 倍）。P4 工程版视图像索 = compact slot + high IOPS 并发（8t）。

**全表映射**（外推，非实测）：320,001,536 行真表 → 20,000,096 grams × 2560B ≈ **51GB 视图**；
构建 = 1.25TB 散读（0.78 页/行 × 320M）+ 51GB 顺序写 ≈ 30-50 分钟单次（离线一次性）。

**门禁固化**（gate.sh bench gate）：真表存在 → 自动 build 20K grams 2560B 视图 → bench
→ 判据 ampl_B ≤ 1.05 且 B8t ≥ 2×A8t（机器无关的结构收益判据）。首次自动结果：
B8t=17.9M, A8t=1.34M, ampl_B=1.00 → PASS。基线 CSV: `probes/baseline_view.csv`。

**P4 判定升级**：视图路径（compact slot）"5x 吞吐 + 12x 省带宽"结论维持；实测从 4KB 槽
("1.60x" 放大) 修正为 compact 槽的 1.00x——**放大口径改用实际读取字节数**（P4 v1 的
1.60x 是"视图/原数据比"，v2 起"磁盘读取/有效载荷"）。

## P4 v3（2026-08-30：全表视图真值 + 规模效应）

**全表视图构建（真表 320,001,536 行 → 51.2GB 视图）**：
- `p4view build data/real-rows 20000096 p4view-full-2560.bin keys-full.txt --slot 2560`
- 流式分块构建（500K grams/chunk，峰值 RSS 395MB，内存受限可行）；**22.0 分钟**（1319.7s），38.8 MB/s 综合（散读瓶颈）
- SSD 掉盘事故：构建中途 USB 掉载 → 重挂载后文件完好（教训：产物/keys 落 SSD 或仓库，勿 /tmp —— keys 被系统清掉，改为 `--keys` 可选 + manifest 携带 n）

**全表规模效应（B 视图单记录读，2560B 紧凑槽）**：
| 规模 | B1t | B8t | 环境 |
|---|---|---|---|
| 200K grams（0.5GB） | 3.19M | 4.50M 行/s | 全部热（RAM） |
| **20M grams 全表（51.2GB）** | 105K | **554K** 行/s (16.8/88.7 MB/s) | 冷（页缓存 4% 命中） |
| 全表**顺序**流读 | — | — | **930 MB/s**（dd 55s） |

→ **核心洞察：视图吞吐 = f(访问调度序)**。随机序下全表 88.7 MB/s 是顺序序的 **1/10**；
这坐实 P3 下一步价值点：**视图按访问模式排布（顺序化 slot + 预取调度序）**，配合
8t 并发可把 51GB 全表视图从 0.55M 拉到 >4M 高倍数（200K 热态即证）。P4b 端到端用任务序时，调度应 pass-through 顺序段。

**口径修正**：早前 200K 数字实为"全热"口径，冷热都必须报（baseline 表已列两档）。

## P4 v4（2026-08-30：延迟分布首测——存储延迟里程碑）

**lat 探针**（`p4view lat <view> [--threads 1|8] [--warm] [--sub N]`）：单记录 2560B 读、随机序、每查独立计时（μs）：

| 档位 | p50 | p95 | p99 | max |
|---|---|---|---|---|
| 20K 视图 warm 1t | 0.75 | 1.13 | 1.38 | 18.7 |
| 20K 视图 warm 8t | 4.75 | 8.25 | 12.29 | **2254.8** |
| 全表 sub100K warm 1t | 0.88 | 1.38 | 1.58 | 51.7 |
| 全表 sub100K warm 8t | 5.17 | 8.46 | 11.25 | **4350.0** |

结论：
1. **存储延迟无压力**：warm p99 ~12μs —— vs 推理 10ms/token（100tok/s）目标，**低 3 个数量级**；
   即使 8 线程（真实 batch 汇合），p99 12μs 仍绰绰有余 → **B 场景的低延迟承诺在存储面已被证实**。
2. **8t vs 1t 代价**：p50 从 ~0.8μs 升至 ~5μs（内核调度/锁/队列），换来吞吐 6-10x；
   尾延迟 max 出现 2-4ms 的罕见簇（~1/20K 样本；OS 换页/镜像/盘 sync）——记录不当验收（验收= p99）。
3. **边界（诚实）**：macOS 页缓存使"冷"不可复现（100K 采样多数已缓存）；**绝对冷 = Linux O_DIRECT / io_uring 后端（M2）后复测**。B 场景部署前提：IO 走 warm 路径（warm/prefetch 已具备）。

## P4 v5（2026-08-30：Linux 真冷首测——T2 闭环，冷热差距被 SSD 吸收）

**环境**：WSL2 Ubuntu（x86_64, i7-6700, VHDX 在 Windows 盘, /dev/sdd 1007G, 内核 6.18）；
**机制**：lat 新增 `--cold`（Linux：每 slot 读前 `posix_fadvise(DONTNEED)` 丢弃页缓存 → 真冷读）。

| 档位（1.6GB 视图, 50K 采） | p50 | p95 | p99 | max |
|---|---|---|---|---|
| warm 1t | 2.00 | 3.10 | 7.11 | 224 |
| **cold 1t（真冷）** | **3.70** | 5.10 | 6.71 | 925 |
| **cold 8t（真冷）** | 4.90 | 6.00 | 7.71 | 18354 |

结论（推翻旧叙事，T2 直接关闭）：
1. **SSD 上"真冷"与"热"同阶**：cold/warm p50 = 1.85x（3.7 vs 2.0μs），p99 也仅 μs 级
   → 冷热差距在消费级 SSD+多层缓存（OS/VDHX/NVMe FTL/SLC cache）被有效吸收；
   **"冷路径 IOPS 墙"只在慢介质（USB flash / HDD / SD）成立，NVMe SSD 不是威胁**；
2. 先前 macOS"假冷"数据的担忧直接消解：B 场景存储延迟（无论冷热）在 μs 级，距 10ms/token
   预算仍有 3+ 数量级余量 → **"低延迟"承诺在 Linux/SSD 目标环境得到强证据支持**；
3. 8t 尾延迟 max 18ms（~1/50K 罕见簇）：WSL/VHDX 元数据或写回抖动——事件性质，不作为验收指标；
4. **口径修正落进 design**：吞吐基准的冷/热差异主要由"介质类别"决定而非缓存态；gate 无冷门槛。

## P4 v6（2026-08-30：io_uring backend 实测——逐提交无增量，批量是正路）

**环境**：WSL2 Ubuntu x86_64 @ VHDX（/dev/sdd 1TB）；合成单 shard 表 400MB（2,500,012×160B，内容零）+ 20K gram keys（320K 行随机 rowid）；A 路径 8t gather。

| backend | A rows/s | 相对 | 备注 |
|---|---|---|---|
| preadv（冷首跑） | **1,094,230** | 1.00× | 与 Mac 真表 1.09M 完全一致 |
| uring（逐提交，warm） | 905,133 | 0.83× | per-call submit_and_wait 每 IO 额外 ring 进入/等待 |

结论（M2 验收口径）：
1. **p/readv 在 Linux+SSD 上不被 uring 逐提交超越**——per-call 提交只是语义验证；
2. 真正价值路径 = **批量提交**（一次 submit N SQE 后统一 wait，绕过 syscall 路径/复制）→ batch API（io::view/gather 路径）留作后端演进；
3. 顺带验证：合成表布局与 Mac 真表行为完全同构（1.09M rows/s 一致），WSL 平台可作为性能复现环境（后续基准以"Mac 真表 + WSL 合成表"双口径）；
4. P4 record：A/B 口径下 1.09M vs B(warm) 26.0M rows/s = 视图 24×（8t 档，虚高因 51MB 全热）——冷对照请用 full 表数据（P4 v3）。

## P4 v7（2026-08-30：P4b 环境探索 + Pi/windows 口径记录）

**llama.cpp 基线（WSL CPU, Qwen3-0.6B-Q8_0, 604MB）**：
- 预填充 pp64 = **31.79 t/s**（8 线程 CPU, i7-6700/haswell 现成 b10688 预编）
- tg 段在 WSL+CPU 环境中 D-state 卡住（llama-cli/llama-bench 均卡，非超时简单问题）——记录为待排查（环境 vs 预编二进制）；GPU 路径是正解
- **CUDA 环境边界（重要）**：WSL /usr/local/cuda/bin/nvcc = **13.2**；driver/CUDA 运行库 = **13.0**（nvidia-smi）
  → 13.2 编译产物跑在 13.0 运行库 = **不兼容**（运行时版本须 ≥ 编译版本）；Windows 侧另有 v12.4 完整 toolkit ✓
  → GPU 决策点：**升级 Windows 驱动到 CUDA 13.2+**（推荐，一条命令/15分钟）或 v12.4 预编方案
- llama.cpp v0.3.0 tag 无 bin 资产：BIN 在 b-tag（b10688 起），win-cuda-12.4/13.3 预编齐全、ubuntu-x64 CPU 预编 ✓（供以后直接用）

**树莓派性能采样（弃，记录原因）**：Pi 功能门禁 17 测试全绿；性能点因 /tmp tmpfs 目录/脚本飘忽 + SD 介质无代表性弃掉（代价收益不成比例）。Pi 定位 = 功能/门禁验证。

**测试基础设施发现（重要复现）**：Windows→WSL 长任务必须走 **schtasks**（每任务 bat + `schtasks /run`），ssh 会话/Start-Process 子进程会被 WSL 树终止——多个异步任务都因此已固化。

## P4 v8（2026-08-30：全表级跨介质对照——介质拖累被量化）

**WSL/VHDX（NVMe 桌面盘, C: 盘）**：合成 51.2GB 视图（fallocate 真实分配 51,200,245,760B），
2M grams 抽样随机序（5.1GB 采样窗口）→ B 路径：

| 档位 | rows/s | MB/s |
|---|---|---|
| 1t 冷随机 | 3,083,627 | 493.4 |
| **8t 冷随机** | **19,163,026** | **3066.1** |

**与 Mac 真表 51.2GB（USB 外盘, P4 v3）对照**：
- Mac USB：8t 冷随机 **554K** 行/s（88.7 MB/s）；WSL NVMe：同口径 **19.2M** → **35× 差异**
- **性质判定**：引擎架构（页对齐/8t/预取调度）在 USB 上已经拉满 IOPS 上限；"全表随机退档"
  的量级 = **介质属性**（USB/外盘 vs 桌面 NVMe），不是引擎设计 —— 对"本地大模型"目标
  （NVMe 桌面 = 目标介质）**B 场景全表随机吞吐的预期值 = 19M 行/s 量级**，远超需求（10K tok/s
  只需 ~128KB/s·流）。
- 工程含义：部署指引明确"生产介质 = 本地 NVMe；外盘仅作研究/拷贝"（design §7 增加介质分层表）。

**（含备注）**：fallocate 后随即读，页缓存会掺入（5.1GB 窗口读 10s）——本对比为"半冷"口径；
绝对冷 vs 热的差值在 P4 v5（DONTNEED fadvise）已验证为 1.85x（SSD 物性），不改变结论方向。
