# 双机开发与便携外盘（2026-08-30 起）

## 设备布局
| 机器 | 角色 | 数据 |
|---|---|---|
| 本机（Mac notebook, 主开发） | 日常热键开发 | 移动硬盘（同外盘） |
| 家庭机（zeng@100.73.212.21, Mac Intel） | 第二工作机（关机可能）| 无内置数据——外盘插入即全套 |

**数据分层**：
- 仓库（git, origin/master 单一源）：代码+文档+probes（全部可重构）
- 外盘 `/Volumes/My Passport`（仅一处物理存在）：
  - `engramdb-data/` — corpus-build（语料 6.0G）/ mock-qwen38-ple / p2-work
  - `qwen38-rows/`（真表 128 分片 48G）
  - `qwen38-ple/`（权重分片 53G）
  - `p4view-full-2560.bin` + `.manifest.json`（51.2G 视图）
- 仓库内 `data/*` **一律 symlink** 指向外盘（`data/corpus-build` → `/Volumes/My Passport/engramdb-data/corpus-build` 等）

## 外盘插入步骤（换机必读）
1. 插入 → 检查挂载点名：`ls /Volumes/`（常见 `My Passport`；不同名则：
   `ln -sfn "/Volumes/<新名>/engramdb-data/corpus-build" ~/code/EngramDB/data/corpus-build`（其余 4 个同法）
2. 校验清单（一项差则重建/回退）：
   ```bash
   ls /Volumes/My\ Passport/qwen38-rows | wc -l   # 128
   ls -la "/Volumes/My Passport/p4view-full-2560.bin"  # ~51G
   du -sh /Volumes/My\ Passport/engramdb-data          # 6.7G
   ```
3. `cd ~/code/EngramDB && bash scripts/gate.sh  # 或最小冒烟：cargo test --workspace`

## 双机同步纪律
- **代码单源** = origin/master；每台机器：`git pull` 起步、commit+push 收尾
- **关机前 checklist**：`git status` 干净 → `git push` → 进行中结论/数据写入 `docs/session-log.md` 或 `probes/p4_view_notes.md`
- **不要**用 scp/rsync 覆盖仓库（绕过单源易漂移）；仅外盘数据可物理搬
- 探针/关键数据文件（probes/*.csv、view-keys-20k.txt）在库；**大资产**（视图/真表/权重）只存在于外盘，重建命令见 `probes/p4_view_notes.md` 顶部清单

## 第二台机环境备忘（zeng@100.73.212.21）
- macOS 15.3.1 / Intel；cargo 1.95（允许 rustup update 至最新）；python3 3.9.6
- numpy 已 `pip3 install --user`（scripts 中 mock_table_gen/p2_ngram_stats/extract_ple_spec/gen_golden/prep_env 需要）
- 全链冒烟：`cargo test --workspace`（17 tests）+ `bash scripts/gate.sh`
- 网络：crates.io 走默认源（非 TUNA 时本地亦可）；发布时用 `scripts/release.sh`（自动规避）

## 快速链路（两台通用）
```bash
engramdb prep --dist agent --reqs 4 --cap-token 200 /tmp/keys.txt
engramdb view build data/real-rows 20000 /tmp/v.bin /tmp/k.txt --slot 2560
engramdb view bench data/real-rows /tmp/v.bin --keys /tmp/k.txt --sub 20000
engramdb view lat /tmp/v.bin --warm
```
