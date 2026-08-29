//! EngramDB CLI（M1 版）。
//!
//! 子命令（最小可用集）：
//! - build     从真实/模拟分片目录构建 Store-I 徽标存储（badge 布局 + manifest）
//! - index     从 rowid 流文件构建 I2 频率索引（counts.bin / counts.txt dump）
//! - gather    取一批 rowid（stdin 行分隔），输出校验和（验证用）
//! - verify    用 Python 参考脚本给同一批 rowid 求 fnv，与本端对拍（bit-exact）
//! - bench-real 真表 P1 微基准（keygen rowid 流）
//!
//! 用法见 `engramdb --help`。

use std::fs::File;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use engramdb_core::count_index::CountIndex;
use engramdb_core::layout::Layout;
use engramdb_keygen::PleSpec;

use engramdb_core::fnv64;

mod workload;

use workload::{gen_tokens, AgentStats, Mode};

fn fnv(bytes: &[u8]) -> u64 {
    fnv64(bytes)
}

fn ple_layout() -> Layout {
    Layout::new(128, 2_500_012, 160, 1)
}

/// 按目录实际分片（shard_/badge_ 命名皆可）自适应布局；
/// 非满 128 分片（mock 表 / 部分数据）也可运行。
fn layout_for_dir(dir: &Path) -> Result<Layout, String> {
    let base = ple_layout();
    let mut n = 0u32;
    for i in 0..base.shards {
        if dir.join(format!("shard_{:03}.bin", i)).exists()
            || dir.join(format!("badge_{:03}.bin", i)).exists()
        {
            n += 1;
        } else {
            break;
        }
    }
    if n == 0 {
        return Err(format!("目录 {dir:?} 中没有分片文件"));
    }
    Ok(Layout::new(n as u64, base.rows_per_shard, base.width, 1))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        println!("usage: engramdb <build|index|gather|verify|bench-real|warm> [args...]");
        return;
    };
    let rest = args;
    let out = match cmd.as_str() {
        "build" => cmd_build(rest),
        "index" => cmd_index(rest),
        "gather" => cmd_gather(rest),
        "verify" => cmd_verify(rest),
        "bench-real" => cmd_bench_real(rest),
        "warm" => cmd_warm(rest),
        _ => Err(format!("unknown command: {cmd}")),
    };
    if let Err(e) = out {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// build <shard_dir> <out_dir>：把 `shard_NNN.bin`（raw fps8 行，宽 160）转成 badge 布局存储。
fn cmd_build(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let src = PathBuf::from(args.next().ok_or("需要 <shard_dir>")?);
    let dst = PathBuf::from(args.next().ok_or("需要 <out_dir>")?);
    let layout = ple_layout();
    std::fs::create_dir_all(&dst).map_err(|e| e.to_string())?;

    // 逐分片：行连续 → 切 badge → 每分片写一个 badged 文件
    for shard in 0..layout.shards {
        let sp = src.join(format!("shard_{:03}.bin", shard));
        if !sp.exists() {
            return Err(format!("缺分片 {sp:?}（{}/{}）", shard, layout.shards));
        }
        let mut f = File::open(&sp).map_err(io_err)?;
        let mut out = File::create(dst.join(format!("badge_{:03}.bin", shard))).map_err(io_err)?;
        let badge_bytes = layout.badge_bytes() as usize;
        let mut badge = vec![0u8; badge_bytes];
        loop {
            let mut got = 0usize;
            while got < badge_bytes {
                let n = std::io::Read::read(&mut f, &mut badge[got..]).map_err(io_err)?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            if got == 0 {
                break;
            }
            // 尾块（不满一 badge）填零（pad），值域不受影响（死行）
            if got < badge_bytes {
                badge[got..].fill(0);
            }
            std::io::Write::write_all(&mut out, &badge).map_err(io_err)?;
        }
        drop(out);
    }

    let manifest = serde_json::json!({
        "layout": { "shards": layout.shards, "rows_per_shard": layout.rows_per_shard,
                    "width": layout.width, "elem_bytes": 1, "badge_rows": layout.badge_rows,
                    "badge_bytes": layout.badge_bytes(), "total_rows": layout.total_rows() },
        "source": src.to_string_lossy(),
    });
    std::fs::write(
        dst.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(io_err)?;
    println!(
        "built {dst:?}  ({} shards × {} rows × {})",
        layout.shards, layout.rows_per_shard, layout.width
    );
    Ok(())
}

/// index <rowids_bin> <out_dir[:index/>]：I2 频率索引。
fn cmd_index(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let src = PathBuf::from(args.next().ok_or("需要 <rowids.bin>")?);
    let dst = args.next().unwrap_or_else(|| "index".to_string());
    let dst = PathBuf::from(dst);
    std::fs::create_dir_all(&dst).map_err(io_err)?;
    let f = File::open(&src).map_err(io_err)?;
    let idx = CountIndex::build_from_bin_stream(std::io::BufReader::new(f)).map_err(io_err)?;
    idx.write_bin(&dst.join("counts.bin")).map_err(io_err)?;
    idx.write_dump(&dst.join("counts.dump.txt"))
        .map_err(io_err)?;
    println!("indexed {} unique rows -> {dst:?}", idx.iter().count());
    Ok(())
}

/// warm <dir> [--tier t3]：顺序读（丢弃）预热 OS 页缓存（T2/T3 之间的冷启动优化）。
fn cmd_warm(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let dir = PathBuf::from(rest.next().ok_or("需要 <rows_dir|badge_dir>")?);
    let mut budget_gb = 1.0f64;
    let mut rest2 = rest;
    while let Some(arg) = rest2.next() {
        if arg == "--budget" {
            budget_gb = rest2
                .next()
                .ok_or("budget 值")?
                .parse()
                .map_err(|e: std::num::ParseFloatError| e.to_string())?;
        }
    }
    let layout = ple_layout();
    let mut warmed: u64 = 0;
    let budget = (budget_gb * 1e9) as u64;
    for shard in 0..layout.shards {
        let p = dir.join(format!("shard_{:03}.bin", shard));
        if !p.exists() {
            // badge 布局目录按 badge_ 前缀
            let p2 = dir.join(format!("badge_{:03}.bin", shard));
            if !p2.exists() {
                continue;
            }
            let mut f = File::open(&p2).map_err(io_err)?;
            let mut probe = 0u8;
            let _gone = 0u64;
            while warmed < budget {
                let mut buf = [0u8; 1 << 20];
                let n = std::io::Read::read(&mut f, &mut buf).map_err(io_err)?;
                if n == 0 {
                    break;
                }
                warmed += n as u64;
                let gone = warmed;
                let _ = gone;
                probe = buf[0];
            }
            let _ = probe;
        } else {
            let mut f = File::open(&p).map_err(io_err)?;
            let mut probe = 0u8;
            while warmed < budget {
                let mut buf = [0u8; 1 << 20];
                let n = std::io::Read::read(&mut f, &mut buf).map_err(io_err)?;
                if n == 0 {
                    break;
                }
                warmed += n as u64;
                probe = buf[0];
            }
            let _ = probe;
        }
        if warmed >= budget {
            break;
        }
    }
    println!("warmed {:.2} GB of OS page cache", warmed as f64 / 1e9);
    Ok(())
}

/// gather <badge_dir>：stdin 每行一个 rowid（升序更好），输出 fnv64 校验和（供对拍）。
fn cmd_gather(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let dir = PathBuf::from(args.next().ok_or("需要 <badge_dir>")?);
    let layout = layout_for_dir(&dir)?;
    let batch = engramdb_io::batch::BadgeGather::open(&dir, &layout).map_err(io_err)?;
    let stdin = std::io::stdin();
    let mut keys = Vec::new();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        keys.push(line.parse::<u64>().map_err(|e| e.to_string())?);
    }
    let w = layout.width as usize;
    let mut out = vec![0u8; keys.len() * w];
    batch.gather_planned(&keys, &mut out).map_err(io_err)?;
    println!("{}", fnv(&out));
    Ok(())
}

/// verify <badge_dir> <input_rowids>：打印每行校验与总计（配合 python 对拍）。
fn cmd_verify(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let dir = PathBuf::from(args.next().ok_or("需要 <badge_dir>")?);
    let input = PathBuf::from(args.next().ok_or("需要 <rowids.txt>")?);
    let layout = layout_for_dir(&dir)?;
    let batch = engramdb_io::batch::BadgeGather::open(&dir, &layout).map_err(io_err)?;
    let mut keys = Vec::new();
    let text = std::fs::read_to_string(&input).map_err(io_err)?;
    for line in text.lines() {
        let line = line.trim();
        if !line.is_empty() {
            keys.push(line.parse::<u64>().map_err(|e| e.to_string())?);
        }
    }
    let w = layout.width as usize;
    if keys.is_empty() {
        println!("fnv=0");
        return Ok(());
    }
    let mut out = vec![0u8; keys.len() * w];
    batch.gather_planned(&keys, &mut out).map_err(io_err)?;
    println!("fnv={} keys={}", fnv(&out), keys.len());
    Ok(())
}

/// bench-real <badge_dir> [--dist uniform|agent] [--stats probes/agent_workload_stats.json]
/// [--reqs 64] [--cap-token 10000] [--iters 8] [--hot-hit 0.35] [--seed 1]
/// 真表口径 16 行/token 批取。uniform = 全词表随机（对照）；agent =
/// 真实流量分布（P2：热集 + 会话记忆 + 短尾），统计来自 probes/agent_workload_stats.json。
fn cmd_bench_real(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let dir = PathBuf::from(args.next().ok_or("需要 <badge_dir>")?);
    let mut dist = "uniform".to_string();
    let mut stats_path = PathBuf::from("probes/agent_workload_stats.json");
    let mut reqs = 64usize;
    let mut cap_token = 10_000u32;
    let mut iters = 8usize;
    let mut hot_hit = 0.35f64;
    let mut seed = 1u64;
    let mut rest = args;
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--dist" => dist = rest.next().ok_or("dist 值")?,
            "--stats" => stats_path = PathBuf::from(rest.next().ok_or("stats 路径")?),
            "--reqs" => reqs = rest.next().ok_or("reqs")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--cap-token" => cap_token = rest.next().ok_or("cap")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--iters" => iters = rest.next().ok_or("iters")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--hot-hit" => hot_hit = rest.next().ok_or("hot")?.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?,
            "--seed" => seed = rest.next().ok_or("seed")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            _ => return Err(format!("未知参数: {a}")),
        }
    }
    let stats = if dist == "agent" {
        Some(AgentStats::load(&stats_path)?)
    } else {
        None
    };
    let mode = if dist == "agent" {
        Mode::Agent(stats_path.clone(), hot_hit)
    } else {
        Mode::Uniform
    };
    let tokens = gen_tokens(&mode, stats.as_ref(), reqs, cap_token, seed)?;
    if tokens.is_empty() {
        return Err("空 token 序列".into());
    }
    let spec = PleSpec::real();
    let mut rowids: Vec<u32> = Vec::with_capacity(tokens.len() * 16);
    for t in 0..tokens.len() {
        let c = tokens[t];
        let triple = [tokens[t.saturating_sub(2)], tokens[t.saturating_sub(1)], c];
        let ids = spec.rowids_for_seq(&triple);
        rowids.extend_from_slice(&ids[0]);
    }
    let keys: Vec<u64> = rowids.iter().map(|&x| x as u64).collect();
    let threads: usize = 8;
    let layout = layout_for_dir(&dir)?;
    let batch = engramdb_io::batch::BadgeGather::open(&dir, &layout).map_err(io_err)?;
    let w = layout.width as usize;
    let mut out = vec![0u8; keys.len() * w];
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        batch.gather_pp(&keys, &mut out, threads).map_err(io_err)?;
        black_box(&out);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let rows_per_s = keys.len() as f64 / dt;
    println!(
        "rows/s={:.0}  tokens={}  keys/batch={}  payload=KB/tok={:.1}",
        rows_per_s,
        tokens.len(),
        keys.len(),
        keys.len() * 160 / 1024 / (keys.len() / 16)
    );
    Ok(())
}

fn black_box<T>(x: &T) {
    unsafe {
        std::ptr::read_volatile(&(x as *const T as usize));
    }
}

fn io_err(e: std::io::Error) -> String {
    e.to_string()
}

/// 让外部可直接调用 fnv（供 python 对拍脚本）——纯函数保持公开。
pub fn exported_fnv(bytes: &[u8]) -> u64 {
    fnv(bytes)
}

#[allow(dead_code)]
fn _keep_serde(_p: &Path) {
    let _ = serde_json::Value::Null;
}
