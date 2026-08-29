//! P4 v2：Store-P 视图构建器 + 视图全表吞吐基准（真表，M1.5-B）。
//!
//! build <rows_dir> <n_grams> <view.bin> <keys.txt> [--slot 4096|2560]
//!   真表行存储 → 物化视图：每 gram 16 头行（2560B）连续存入 slot（4096 = 4KB 对齐槽，
//!   2560 = 紧凑槽）。输出 manifest.json（row 数/slot 类型/构建秒数/构建 MB/s）。
//! bench <rows_dir> <view.bin> <keys.txt> [--threads 8 --iters 3]
//!   路径 A（原始 16 行 scatter）+ 路径 B（视图单记录读）时间/吞吐/字节放大，
//!   并按 P4 口径输出 CSV 行（name,rows,rows_per_s,mb_per_s,ampl）。
//!
//! 复现（P4 判定口径）：`p4view build data/real-rows <N> <view.bin> <keys.txt>`
//! `p4view bench data/real-rows <view.bin> <keys.txt>`

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;

const HEAD_W: u64 = 16;
const ROW_BYTES: u64 = 160;
const RECORD_BYTES: u64 = HEAD_W * ROW_BYTES; // 2560B（16 头拼接）

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        println!("usage: p4view <build|bench> <...> [--slot 4096|2560]");
        return;
    };
    let out = match cmd.as_str() {
        "build" => cmd_build(args),
        "bench" => cmd_bench(args),
        _ => Err(format!("unknown {cmd}")),
    };
    match out {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn lay() -> Layout {
    Layout::new(128, 2_500_012, 160, 1)
}

fn cmd_build(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let n: usize = rest
        .next()
        .ok_or("n_grams")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let view_out = PathBuf::from(rest.next().ok_or("view.bin")?);
    let keys_out = PathBuf::from(rest.next().ok_or("keys.txt")?);
    let mut slot_bytes: u64 = 4096;
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
            "--seed" => {
                let _ = it.next();
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    if slot_bytes != 4096 && slot_bytes != RECORD_BYTES {
        return Err(format!(
            "slot 只能 4096 或 {RECORD_BYTES}（收到 {slot_bytes}）"
        ));
    }

    let layout = lay();
    let batch = BadgeGather::open(&rows_dir, &layout).map_err(|e| e.to_string())?;

    let spec = engramdb_keygen::PleSpec::real();
    let mut rng_state: u64 = 0xDEAD_BEEF_1234_5678;
    let mut keys_file = File::create(&keys_out).map_err(|e| e.to_string())?;
    let mut view = File::create(&view_out).map_err(|e| e.to_string())?;
    let mut rowids = Vec::with_capacity(n * 16);
    for _ in 0..n {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let a = (rng_state % 248_320) as u32;
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let b = (rng_state % 248_320) as u32;
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let c = (rng_state % 248_320) as u32;
        let ids = spec.rowids_for_seq(&[a, b, c]);
        for &r in &ids[0] {
            rowids.push(r as u64);
        }
    }
    let mut out = vec![0u8; rowids.len() * ROW_BYTES as usize];
    let t0 = std::time::Instant::now();
    batch
        .gather_pp(&rowids, &mut out, 8)
        .map_err(|e| e.to_string())?;
    let mut slot = vec![0u8; slot_bytes as usize];
    for i in 0..n {
        let rec = &out[i * 16 * ROW_BYTES as usize..(i + 1) * 16 * ROW_BYTES as usize];
        slot[..rec.len()].copy_from_slice(rec);
        view.write_all(&slot).map_err(|e| e.to_string())?;
    }
    let build_s = t0.elapsed().as_secs_f64();
    for &r in &rowids {
        writeln!(keys_file, "{r}").map_err(|e| e.to_string())?;
    }
    view.sync_all().map_err(|e| e.to_string())?;
    let manifest = serde_json::json!({
        "grans": n,
        "heads": HEAD_W,
        "slot_bytes": slot_bytes,
        "record_bytes": RECORD_BYTES,
        "build_seconds": build_s,
        "build_mb_s": (n as f64 * slot_bytes as f64 / 1e6) / build_s,
        "rows": rowids.len(),
        "source": rows_dir.to_string_lossy(),
    });
    let mp = view_out.with_file_name("view-manifest.json");
    std::fs::write(&mp, serde_json::to_vec_pretty(&manifest).unwrap())
        .map_err(|e| e.to_string())?;
    println!(
        "view built: n={n} slot={slot_bytes}B view={} took={build_s:.1}s",
        view_out.display()
    );
    Ok(())
}

fn cmd_bench(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let keys_file = PathBuf::from(rest.next().ok_or("keys.txt")?);
    let mut threads = 8usize;
    let mut slot_bytes: u64 = 0; // 0 = 从 view-manifest.json 读（默认）
    let mut it = rest;
    while let Some(a) = it.next() {
        match a.as_str() {
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
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    if slot_bytes == 0 {
        let mp = view_file.with_file_name("view-manifest.json");
        if mp.exists() {
            let m: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&mp).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            slot_bytes = m["slot_bytes"].as_u64().unwrap_or(4096);
        } else {
            slot_bytes = 4096;
        }
    }

    let layout = lay();
    let batch = BadgeGather::open(&rows_dir, &layout).map_err(|e| e.to_string())?;

    let mut keys = Vec::new();
    for l in BufReader::new(std::fs::File::open(&keys_file).map_err(|e| e.to_string())?)
        .lines()
        .map_while(Result::ok)
    {
        let l = l.trim();
        if !l.is_empty() {
            keys.push(
                l.parse::<u64>()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?,
            );
        }
    }
    let n_grams = keys.len() / HEAD_W as usize;
    let w = layout.width as usize;

    // ---- A: 16 行 scatter ----
    let mut out_a = vec![0u8; keys.len() * w];
    let t0 = std::time::Instant::now();
    batch
        .gather_pp(&keys, &mut out_a, 8)
        .map_err(|e| e.to_string())?;
    let _dt_a = t0.elapsed();
    let pages_a = unique_pages(&keys, &layout);
    println!("A unique 4KiB pages: {} (rows {})", pages_a, keys.len());

    // ---- B: view 单记录读 ----
    let vf = std::fs::File::open(&view_file).map_err(|e| e.to_string())?;
    let mut g_state: u64 = 0xCAFE_BEEF_0F1E_2D3C;
    let mut order: Vec<u64> = Vec::with_capacity(n_grams);
    for _ in 0..n_grams {
        g_state = g_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        order.push(g_state % n_grams as u64);
    }
    let view_ref = &vf;
    let run_par = |threads: usize| -> std::time::Duration {
        let t = std::time::Instant::now();
        std::thread::scope(|sc| {
            let chunk = n_grams.div_ceil(threads);
            for th in 0..threads {
                let lo = th * chunk;
                if lo >= n_grams {
                    break;
                }
                let hi = (lo + chunk).min(n_grams);
                let slice = &order[lo..hi];
                sc.spawn(move || {
                    let mut buf = vec![0u8; slot_bytes as usize];
                    for &rec in slice {
                        let off = rec * slot_bytes;
                        let _ = view_ref.read_exact_at(&mut buf, off);
                    }
                });
            }
        });
        t.elapsed()
    };
    let mut csv = String::new();
    let dt_b = run_par(1);
    csv.push_str(&report("B", n_grams as u64 * HEAD_W, dt_b));
    let dt_b2 = run_par(threads);
    csv.push_str(&report("B", n_grams as u64 * HEAD_W, dt_b2));
    let t3 = std::time::Instant::now();
    batch
        .gather_pp(&keys, &mut out_a, 8)
        .map_err(|e| e.to_string())?;
    let dt_a2 = t3.elapsed();
    csv.push_str(&report("A", keys.len() as u64, dt_a2));

    let ba = keys.len() as u64 * ROW_BYTES;
    let ra_a = pages_a as u64 * 4096;
    let ra_b = n_grams as u64 * slot_bytes;
    csv.push_str(&format!(
        "amplification,A,{},B,{}\n",
        ra_a as f64 / ba as f64,
        ra_b as f64 / ba as f64
    ));
    println!("{csv}");
    Ok(())
}

fn unique_pages(keys: &[u64], layout: &Layout) -> usize {
    let mut set = std::collections::HashSet::with_capacity(keys.len() / 8);
    for &k in keys {
        let byte_off = k * layout.row_bytes;
        set.insert(byte_off / 4096);
    }
    set.len()
}

fn report(name: &str, rows: u64, dt: std::time::Duration) -> String {
    let s = dt.as_secs_f64();
    let rps = rows as f64 / s.max(1e-9);
    let mbps = (rows as f64 * ROW_BYTES as f64) / 1e6 / s.max(1e-9);
    println!("{name}: rows={rows} time={s:.3}s  rows/s={rps:.0}  MB/s={mbps:.1}");
    format!("{name},{rows},{:.0},{:.1}\n", rps, mbps)
}
