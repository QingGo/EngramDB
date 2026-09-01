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
use engramdb_io::backend::{default_backend, IoBackend};
use engramdb_io::batch::BadgeGather;
use engramdb_io::view;

mod serve;
mod slot_index;
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
        println!("usage: engramdb <build|index|gather|verify|bench-real|warm|view|slot-index|prep|tables|serve> [args...]");
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
        "view" => cmd_view(rest),
        "slot-index" => cmd_slot_index(rest),
        "prep" => cmd_prep(rest),
        "tables" => cmd_tables(rest),
        "serve" => cmd_serve(rest),
        "check" => cmd_check(rest),
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

/// slot-index build <keys.txt> <out_dir> [--buckets N] [--cache N]
/// slot-index verify <keys.txt> <out_dir> [--cache N]
/// 原生磁盘 SlotIndex：行/产品侧无需 Python，直接生成 Python v2 兼容索引。
fn cmd_slot_index(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let Some(sub) = rest.next() else {
        return Err("slot-index 需要子命令 build|verify".into());
    };
    let it = rest;
    match sub.as_str() {
        "build" => cmd_slot_index_build(it),
        "verify" => cmd_slot_index_verify(it),
        _ => Err(format!("unknown slot-index subcommand: {sub}")),
    }
}

fn cmd_slot_index_build(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let keys = PathBuf::from(rest.next().ok_or("需要 <keys.txt>")?);
    let out_dir = PathBuf::from(rest.next().ok_or("需要 <out_dir>")?);
    let mut buckets = slot_index::DEFAULT_BUCKETS;
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--buckets" => {
                buckets = rest
                    .next()
                    .ok_or("buckets 值")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    let stats = slot_index::build_from_keys_file(&keys, &out_dir, buckets)?;
    println!(
        "slot-index built: count={} buckets={} bytes={} took={:.2}s -> {}",
        stats.count,
        buckets,
        stats.bytes,
        stats.seconds,
        out_dir.display()
    );
    Ok(())
}

fn cmd_slot_index_verify(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let keys = PathBuf::from(rest.next().ok_or("需要 <keys.txt>")?);
    let out_dir = PathBuf::from(rest.next().ok_or("需要 <out_dir>")?);
    let mut cache = slot_index::DEFAULT_CACHE_BUCKETS;
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--cache" => {
                cache = rest
                    .next()
                    .ok_or("cache 值")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    let verified = slot_index::verify_from_keys_file(&keys, &out_dir, cache)?;
    println!("slot-index verified: {verified} grams OK");
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
            "--reqs" => {
                reqs = rest
                    .next()
                    .ok_or("reqs")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--cap-token" => {
                cap_token = rest
                    .next()
                    .ok_or("cap")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--iters" => {
                iters = rest
                    .next()
                    .ok_or("iters")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--hot-hit" => {
                hot_hit = rest
                    .next()
                    .ok_or("hot")?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--seed" => {
                seed = rest
                    .next()
                    .ok_or("seed")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
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

/// prep [<tokens.u32.npy> | --dist uniform|agent] <out_keys.txt>
/// 把 token 流（P2 语料导出 npy，或 --dist 真实流量模拟）映射为 gram rowid keys
/// （每 token 16 头平铺），供后续 build/gather/verify/view 链消费（训练数据准备入口）。
/// npy 格式与 p2rowid 一致（np.save 原生 u32 数组）。
fn cmd_prep(rest: impl Iterator<Item = String>) -> Result<(), String> {
    let mut args: Vec<String> = rest.collect();
    if args.is_empty() {
        return Err("usage: engramdb prep <tokens.u32.npy | --dist agent> <out_keys.txt>".into());
    }
    let mut dist = false;
    let mut dist_name = "uniform".to_string();
    let mut stats_path = PathBuf::from("probes/agent_workload_stats.json");
    let mut reqs = 64usize;
    let mut cap_token = 10_000u32;
    let mut hot_hit = 0.35f64;
    let mut seed = 1u64;
    let mut tokens_src: Option<PathBuf> = None;
    let mut pos: Vec<PathBuf> = Vec::new();
    let mut it = args.drain(..);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dist" => {
                dist = true;
                dist_name = it.next().ok_or("dist 值")?;
            }
            "--stats" => stats_path = PathBuf::from(it.next().ok_or("stats 路径")?),
            "--reqs" => {
                reqs = it
                    .next()
                    .ok_or("reqs")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--cap-token" => {
                cap_token = it
                    .next()
                    .ok_or("cap")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--hot-hit" => {
                hot_hit = it
                    .next()
                    .ok_or("hot")?
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?
            }
            "--seed" => {
                seed = it
                    .next()
                    .ok_or("seed")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            _ => pos.push(PathBuf::from(a)),
        }
    }
    if dist {
        if pos.len() != 1 {
            return Err("--dist 模式下只需要 <out_keys.txt>".into());
        }
    } else if pos.len() == 2 {
        tokens_src = Some(pos[0].clone());
    } else if pos.len() != 1 {
        return Err("需要 <tokens.u32.npy> <out_keys.txt>".into());
    }
    if pos.len() > 1 {
        let _ = pos.pop();
    }
    let out_keys = pos.pop().ok_or("缺少 <out_keys.txt>")?;
    tokens_src = tokens_src.take();
    let tokens: Vec<u32> = if dist {
        let stats = if dist_name == "agent" {
            Some(AgentStats::load(&stats_path)?)
        } else {
            None
        };
        let mode = if dist_name == "agent" {
            Mode::Agent(stats_path.clone(), hot_hit)
        } else {
            Mode::Uniform
        };
        let t = gen_tokens(&mode, stats.as_ref(), reqs, cap_token, seed)?;
        if t.is_empty() {
            return Err("空 token 序列".into());
        }
        t
    } else {
        let src = tokens_src.ok_or("缺少 token 源或 --dist")?;
        read_tokens_npy(&src)?
    };
    let spec = PleSpec::real();
    let mut w =
        std::io::BufWriter::new(std::fs::File::create(&out_keys).map_err(|e| e.to_string())?);
    use std::io::Write as _;
    for t in 0..tokens.len() {
        let triple = [
            tokens[t.saturating_sub(2)],
            tokens[t.saturating_sub(1)],
            tokens[t],
        ];
        let ids = spec.rowids_for_seq(&triple);
        for &r in &ids[0] {
            writeln!(w, "{}", r as u64).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    println!(
        "prep: tokens={} rows={} keys_file={}",
        tokens.len(),
        tokens.len() * 16,
        out_keys.display()
    );
    Ok(())
}

/// tables <root>：列出多表根目录下的所有 EngramDB 表。
fn cmd_tables(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let root = PathBuf::from(rest.next().ok_or("需要 <root>")?);
    let tables = serve::list_tables(&root)?;
    let out = serde_json::json!({ "tables": tables });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}

/// check <root>：校验多表根目录下每张表的 manifest 与分片文件完整性。
fn cmd_check(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let root = PathBuf::from(rest.next().ok_or("需要 <root>")?);
    let summary = serve::check_root(&root)?;
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    if summary["ok"] != true {
        return Err("one or more tables failed integrity check".into());
    }
    Ok(())
}

/// serve <root> [--host 127.0.0.1] [--port 8765] [--binary]
/// 支持 ping / list_tables / fetch / fetch_raw（多表按目录解析，优先读 manifest）。
fn cmd_serve(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let root = PathBuf::from(rest.next().ok_or("需要 <root>")?);
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8765;
    let mut binary = false;
    let mut it = rest;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--host" => host = it.next().ok_or("host 值")?,
            "--port" => {
                port = it
                    .next()
                    .ok_or("port 值")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--binary" => binary = true,
            other => return Err(format!("未知参数 {other}")),
        }
    }
    if binary {
        serve::run_binary(&root, &host, port)
    } else {
        serve::run(&root, &host, port)
    }
}

/// 读取 np.save 原生 u32 数组（P2 语料导出格式，与 p2rowid 同源）。
fn read_tokens_npy(path: &Path) -> Result<Vec<u32>, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 8 || &data[0..6] != b"\x93NUMPY" {
        return Err(format!("{path:?} 非 npy（np.save 原生 u32）"));
    }
    let mut idx = 6usize;
    let major = data[idx];
    idx += 1;
    idx += 1; // minor
    let hlen = if major == 1 {
        u16::from_le_bytes([data[idx], data[idx + 1]]) as usize
    } else {
        u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]) as usize
    };
    idx += if major == 1 { 2 } else { 4 };
    let raw = &data[idx + hlen..];
    if raw.len() % 4 != 0 {
        return Err("npy 数据非 u32 对齐".into());
    }
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn black_box<T>(x: &T) {
    unsafe {
        std::ptr::read_volatile(&(x as *const T as usize));
    }
}

fn io_err(e: std::io::Error) -> String {
    e.to_string()
}

/// 在视图 manifest 中记录原生生成的 slot-index 路径。
fn update_view_manifest_slot_index(view_path: &Path, idx_dir: &Path) -> Result<(), String> {
    let mp = view_path.with_extension("manifest.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&mp).map_err(io_err)?).map_err(|e| e.to_string())?;
    value["slot_index"] = serde_json::json!(idx_dir.display().to_string());
    std::fs::write(
        &mp,
        serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
    )
    .map_err(io_err)?;
    Ok(())
}

/// 从视图 manifest 读取 keys_out 路径（供 verify --slot-index 使用）。
fn manifest_keys_path(view_path: &Path) -> Result<PathBuf, String> {
    let mp = view_path.with_extension("manifest.json");
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&mp).map_err(io_err)?).map_err(|e| e.to_string())?;
    let keys = value
        .get("keys_out")
        .and_then(|v| v.as_str())
        .ok_or("view manifest missing keys_out")?;
    Ok(PathBuf::from(keys))
}

/// 让外部可直接调用 fnv（供 python 对拍脚本）——纯函数保持公开。
pub fn exported_fnv(bytes: &[u8]) -> u64 {
    fnv(bytes)
}

#[allow(dead_code)]
fn _keep_serde(_p: &Path) {
    let _ = serde_json::Value::Null;
}

/// view <build|bench|lat|verify> ...：Store-P 物化视图（P4 产品面；与探针 p4view 同构）。
/// 用法：
///   engramdb view build <rows_dir> <n_grams> <view.bin> <keys.txt> [--slot 2560|4096] [--keys IN_KEYS] [--keys-stream KEYS] [--slot-index DIR] [--verify]
///   engramdb view bench <rows_dir> <view.bin> [--keys K] [--sub N] [--threads 8] [--slot B] [--backend preadv|uring]
///   engramdb view lat <view.bin> [--threads 1|8] [--warm] [--cold] [--sub N] [--slot B]
///   engramdb view verify <rows_dir> <view.bin> [--keys K] [--slot-index DIR] [--sub N]
fn cmd_view(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let Some(sub) = rest.next() else {
        return Err("view 需要子命令 build|bench|lat|verify".into());
    };
    let it = rest;
    match sub.as_str() {
        "build" => cmd_view_build(it),
        "bench" => cmd_view_bench(it),
        "lat" => cmd_view_lat(it),
        "verify" => cmd_view_verify(it),
        _ => Err(format!("unknown view subcommand: {sub}")),
    }
}

fn backend_for(name: Option<String>) -> Result<Box<dyn IoBackend>, String> {
    match name.as_deref() {
        None | Some("preadv") => Ok(default_backend()),
        #[cfg(target_os = "linux")]
        Some("uring") => Ok(Box::new(engramdb_io::backend::UringBackend)),
        #[cfg(target_os = "linux")]
        Some("uring-batch") => Ok(Box::new(engramdb_io::backend::UringBatchBackend)),
        #[cfg(not(target_os = "linux"))]
        Some("uring") => Err("uring 后端仅 Linux 可用（当前平台不可用）".into()),
        Some(other) => Err(format!("unknown backend: {other}")),
    }
}

fn cmd_view_build(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let n: usize = rest
        .next()
        .ok_or("n_grams")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let view_out = PathBuf::from(rest.next().ok_or("view.bin")?);
    let keys_out = PathBuf::from(rest.next().ok_or("keys.txt")?);
    let mut slot_bytes: u64 = 2560;
    let mut backend_name: Option<String> = None;
    let mut keys_in: Option<PathBuf> = None;
    let mut keys_stream: Option<PathBuf> = None;
    let mut slot_index_out: Option<PathBuf> = None;
    let mut verify = false;
    let mut it = rest;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--slot" => {
                slot_bytes = it
                    .next()
                    .ok_or("slot")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--backend" => backend_name = Some(it.next().ok_or("backend")?),
            "--keys" => keys_in = Some(PathBuf::from(it.next().ok_or("keys 路径")?)),
            "--keys-stream" | "--keys-file" => {
                keys_stream = Some(PathBuf::from(it.next().ok_or("keys 路径")?))
            }
            "--slot-index" => {
                slot_index_out = Some(PathBuf::from(it.next().ok_or("slot-index 路径")?))
            }
            "--verify" => verify = true,
            "--seed" => {
                let _ = it.next();
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    let layout = layout_for_dir(&rows_dir)?;
    let batch = BadgeGather::open_with_backend(&rows_dir, &layout, backend_for(backend_name)?)
        .map_err(|e| e.to_string())?;
    if let Some(ks) = keys_stream {
        if keys_in.is_some() {
            return Err("不能同时指定 --keys 与 --keys-stream".into());
        }
        // 流式构建：不把整个 keys 文件读入内存，适合全表/超大访问序视图。
        let _ =
            view::build_view_from_keys_file(&batch, &ks, slot_bytes, &view_out, Some(&keys_out))
                .map_err(|e| e.to_string())?;
    } else if let Some(kf) = keys_in {
        // 使用调用方提供的访问序/rowid 列表构建视图。keys 文件为 16 头平铺：
        // 每 gram 连续 16 行，物理槽位顺序 = 文件顺序。
        let all = view::read_keys(&kf).map_err(|e| e.to_string())?;
        let need = n * view::HEAD_W as usize;
        if all.len() < need {
            return Err(format!(
                "--keys 文件只有 {} 行，需要至少 {need} 行（{n} grams × {} heads）",
                all.len(),
                view::HEAD_W
            ));
        }
        let _ = view::build_view_from_keys(
            &batch,
            &all[..need],
            slot_bytes,
            &view_out,
            Some(&keys_out),
        )
        .map_err(|e| e.to_string())?;
    } else {
        let _ = view::build_view(&batch, n, slot_bytes, &view_out, Some(&keys_out))
            .map_err(|e| e.to_string())?;
    }
    if let Some(idx_dir) = slot_index_out {
        let stats =
            slot_index::build_from_keys_file(&keys_out, &idx_dir, slot_index::DEFAULT_BUCKETS)?;
        update_view_manifest_slot_index(&view_out, &idx_dir)?;
        println!(
            "view slot-index built: count={} bytes={} took={:.2}s -> {}",
            stats.count,
            stats.bytes,
            stats.seconds,
            idx_dir.display()
        );
    }
    if verify {
        let keys = view::read_keys(&keys_out).map_err(|e| e.to_string())?;
        view::verify_view(&batch, &view_out, Some(&keys), 0).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn cmd_view_bench(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let mut keys_path: Option<PathBuf> = None;
    let mut sub_grams = 0usize;
    let mut threads = 8usize;
    let mut slot_bytes: u64 = 0;
    let mut backend_name: Option<String> = None;
    let mut order = String::from("rand");
    let mut it = rest;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--order" => order = it.next().ok_or("order 值")?,
            "--keys" => keys_path = Some(PathBuf::from(it.next().ok_or("keys 路径")?)),
            "--sub" => {
                sub_grams = it
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--threads" => {
                threads = it
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--slot" => {
                slot_bytes = it
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--backend" => backend_name = Some(it.next().ok_or("backend")?),
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    let layout = layout_for_dir(&rows_dir)?;
    let batch = BadgeGather::open_with_backend(&rows_dir, &layout, backend_for(backend_name)?)
        .map_err(|e| e.to_string())?;
    let keys = match &keys_path {
        Some(kf) => Some(view::read_keys(kf).map_err(|e| e.to_string())?),
        None => None,
    };
    view::bench_view(
        &batch,
        &view_file,
        keys.as_deref(),
        sub_grams,
        threads,
        slot_bytes,
        &order,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_view_lat(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let mut threads = 8usize;
    let mut warm = false;
    let mut cold = false;
    let mut sub_grams = 0usize;
    let mut slot_bytes: u64 = 0;
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--threads" => {
                threads = rest
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--warm" => warm = true,
            "--cold" => cold = true,
            "--sub" => {
                sub_grams = rest
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--slot" => {
                slot_bytes = rest
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    view::lat_view(&view_file, threads, warm, cold, sub_grams, slot_bytes)
        .map_err(|e| e.to_string())
}

fn cmd_view_verify(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let mut keys_path: Option<PathBuf> = None;
    let mut slot_index_dir: Option<PathBuf> = None;
    let mut sub_grams = 0usize;
    let mut backend_name: Option<String> = None;
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--keys" => keys_path = Some(PathBuf::from(rest.next().ok_or("keys 路径")?)),
            "--slot-index" => {
                slot_index_dir = Some(PathBuf::from(rest.next().ok_or("slot-index 路径")?))
            }
            "--sub" => {
                sub_grams = rest
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--backend" => backend_name = Some(rest.next().ok_or("backend")?),
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    if let Some(idx_dir) = slot_index_dir {
        let keys_file = match &keys_path {
            Some(kf) => kf.clone(),
            None => manifest_keys_path(&view_file)?,
        };
        let verified = slot_index::verify_from_keys_file(
            &keys_file,
            &idx_dir,
            slot_index::DEFAULT_CACHE_BUCKETS,
        )?;
        println!("view slot-index verified: {verified} grams OK");
    }
    let layout = layout_for_dir(&rows_dir)?;
    let batch = BadgeGather::open_with_backend(&rows_dir, &layout, backend_for(backend_name)?)
        .map_err(|e| e.to_string())?;
    let keys = match &keys_path {
        Some(kf) => Some(view::read_keys(kf).map_err(|e| e.to_string())?),
        None => None,
    };
    view::verify_view(&batch, &view_file, keys.as_deref(), sub_grams).map_err(|e| e.to_string())
}
