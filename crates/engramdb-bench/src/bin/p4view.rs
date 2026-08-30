//! P4 v2/v4：Store-P 视图构建器 + 吞吐/延迟双基准（真表，M1.5-B）。
//!
//! build <rows_dir> <n_grams> <view.bin> <keys.txt> [--slot 4096|2560]
//!   真表行存储 → 物化视图：每 gram 16 头行（2560B）连续存入 slot（4096 = 4KB 对齐槽，
//!   2560 = 紧凑槽）。输出 manifest.json（row 数/slot 类型/构建秒数/构建 MB/s）。
//! bench <rows_dir> <view.bin> <keys.txt> [--threads 8 --iters 3]
//!   路径 A（原始 16 行 scatter）+ 路径 B（视图单记录读）时间/吞吐/字节放大，
//!   并按 P4 口径输出 CSV 行（name,rows,rows_per_s,mb_per_s,ampl）。
//!
//! lat <view.bin> [--sub N] [--threads 1|8] [--warm] [--slot B]
//!   延迟分布：随机序单记录读，逐次独立计时 → p50/p95/p99/max/mean（μs）。
//!   --warm 先全文件顺序预读（热档）；缺省即冷档（新进程 first-access）。
//!
//! 复现（P4 判定口径）：`p4view build data/real-rows <N> <view.bin> <keys.txt>`
//! `p4view bench data/real-rows <view.bin> <keys.txt>`

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
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
        "lat" => cmd_lat(args),
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

    // 流式分块构建（内存受限）：每 chunk 500K grams；rng 状态跨 chunk 连续
    const CHUNK_G: usize = 500_000;
    let spec = engramdb_keygen::PleSpec::real();
    let mut rng_state: u64 = 0xDEAD_BEEF_1234_5678;
    let mut keys_file = File::create(&keys_out).map_err(|e| e.to_string())?;
    let mut view = File::create(&view_out).map_err(|e| e.to_string())?;
    let t0 = std::time::Instant::now();
    let mut slot = vec![0u8; slot_bytes as usize];
    let mut done = 0usize;
    while done < n {
        let m = CHUNK_G.min(n - done);
        let mut rowids = Vec::with_capacity(m * HEAD_W as usize);
        for _ in 0..m {
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
        batch
            .gather_pp(&rowids, &mut out, 8)
            .map_err(|e| e.to_string())?;
        for i in 0..m {
            let rec = &out[i * 16 * ROW_BYTES as usize..(i + 1) * 16 * ROW_BYTES as usize];
            slot[..rec.len()].copy_from_slice(rec);
            view.write_all(&slot).map_err(|e| e.to_string())?;
        }
        for &r in &rowids {
            writeln!(keys_file, "{r}").map_err(|e| e.to_string())?;
        }
        done += m;
    }
    let build_s = t0.elapsed().as_secs_f64();
    view.sync_all().map_err(|e| e.to_string())?;
    let manifest = serde_json::json!({
        "grans": n,
        "heads": HEAD_W,
        "slot_bytes": slot_bytes,
        "record_bytes": RECORD_BYTES,
        "build_seconds": build_s,
        "build_mb_s": (n as f64 * slot_bytes as f64 / 1e6) / build_s,
        "rows": n as u64 * HEAD_W,
        "source": rows_dir.to_string_lossy(),
    });
    let mp = view_out.with_extension("manifest.json");
    std::fs::write(&mp, serde_json::to_vec_pretty(&manifest).unwrap())
        .map_err(|e| e.to_string())?;
    println!(
        "view built: n={n} slot={slot_bytes}B view={} took={build_s:.1}s",
        view_out.display()
    );
    Ok(())
}

/// 延迟分布探针（P4 验收关键项）。
/// 随机序单记录读 × N（每查独立计时）→ p50/p95/p99/max/mean（μs）。
fn cmd_lat(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let mut threads: usize = 8;
    let mut warm = false;
    let mut cold_parse = false;
    let mut sub_grams: usize = 0; // 0 = 全量
    let mut slot_bytes: u64 = 0; // 0 = manifest
    let mut it = rest;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" => threads = it.next().ok_or("v")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--warm" => warm = true,
            "--cold" => cold_parse = true,
            "--sub" => sub_grams = it.next().ok_or("v")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            "--slot" => slot_bytes = it.next().ok_or("v")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "linux")]
    let cold = cold_parse;
    #[cfg(not(target_os = "linux"))]
    let _cold = cold_parse;
    #[cfg(not(target_os = "linux"))]
    if cold_parse {
        eprintln!("warn: --cold 仅 Linux 有效（需 posix_fadvise DONTNEED）");
    }
    if slot_bytes == 0 {
        let mp = view_file.with_extension("manifest.json");
        if mp.exists() {
            let m: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&mp).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            slot_bytes = m["slot_bytes"].as_u64().unwrap_or(RECORD_BYTES);
        } else {
            slot_bytes = RECORD_BYTES;
        }
    }
    let vf = std::fs::File::open(&view_file).map_err(|e| e.to_string())?;
    let meta = vf.metadata().map_err(|e| e.to_string())?;
    let n = (meta.len() / slot_bytes) as usize;
    let n_grams = if sub_grams > 0 { n.min(sub_grams) } else { n };
    if n_grams == 0 {
        return Err("视图为空".into());
    }
    if warm {
        // 全文件顺序预读（把文件全拉进页缓存）
        let mut buf = vec![0u8; 8 << 20];
        let mut off: u64 = 0;
        let f = vf.try_clone().map_err(|e| e.to_string())?;
        while off < meta.len() {
            let want = (buf.len() as u64).min(meta.len() - off) as usize;
            let rd = f.read_at(&mut buf[..want], off).map_err(|e| e.to_string())?;
            if rd == 0 {
                break;
            }
            off += rd as u64;
        }
    }
    // 随机访问序（固定 seed，跨线程共享序列由各线程错位切片保持覆盖）
    let order = rand_order(n_grams, 0xFEED_BEEF_0D0F_1E2C);
    let view_ref = &vf;
    #[cfg(target_os = "linux")]
    let cold = cold_parse;
    #[cfg(not(target_os = "linux"))]
    let _cold = cold_parse;
    let per_thread = |tid: usize| -> Vec<u32> {
        let t = std::time::Instant::now();
        let mut buf = vec![0u8; slot_bytes as usize];
        let mut out = Vec::new();
        let stride = order.len() / threads;
        let lo = tid * stride;
        let hi = if tid + 1 == threads { order.len() } else { lo + stride };
        for &rec in &order[lo..hi] {
            let t0 = std::time::Instant::now();
#[cfg(target_os = "linux")]
            if cold {
                // Linux 真冷：读前对该 slot 丢弃页缓存（DONTNEED），绕过 warm 语义
                unsafe {
                    libc::posix_fadvise(
                        view_ref.as_raw_fd(),
                        (rec * slot_bytes) as i64,
                        slot_bytes as i64,
                        libc::POSIX_FADV_DONTNEED,
                    );
                }
            }
            let _ = view_ref.read_exact_at(&mut buf, rec * slot_bytes);
            let ns = t0.elapsed().as_nanos();
            out.push(ns.min(u32::MAX as u128) as u32);
        }
        let _ = t;
        out
    };
    let mut times: Vec<u32> = std::thread::scope(|sc| {
        let mut h = Vec::new();
        for tid in 0..threads {
            h.push(sc.spawn(move || per_thread(tid)));
        }
        let mut all = Vec::with_capacity(n_grams);
        for x in h {
            all.extend(x.join().unwrap());
        }
        all
    });
    times.sort_unstable();
    let p = |q: f64| -> f64 {
        let idx = ((times.len() - 1) as f64 * q).round() as usize;
        times[idx] as f64
    };
    let mean = times.iter().map(|&x| x as f64).sum::<f64>() / times.len() as f64;
    let (p50, p95, p99, mx) = (p(0.50), p(0.95), p(0.99), times.last().copied().unwrap_or(0) as f64);
    let us = |ns: f64| ns / 1000.0;
    println!(
        "lat: n={n_grams} slot={slot_bytes} threads={threads} {} [μs] p50={:.2} p95={:.2} p99={:.2} max={:.2} mean={:.2}",
        if warm { "warm" } else { "cold" },
        us(p50), us(p95), us(p99), us(mx), us(mean)
    );
    println!("latency_us,p50,p95,p99,max,mean,{},{},{},{},{}
", us(p50), us(p95), us(p99), us(mx), us(mean));
    Ok(())
}

fn rand_order(n: usize, seed: u64) -> Vec<u64> {
    let mut st = seed;
    let mut v: Vec<u64> = (0..n as u64).collect();
    // Fisher-Yates（xorshift64）
    for i in (1..n).rev() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let j = (st % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn cmd_bench(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let mut threads = 8usize;
    let mut keys_file: Option<PathBuf> = None;
    let mut sub_grams: usize = 0; // 0 = 全量
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
            "--keys" => {
                keys_file = Some(PathBuf::from(it.next().ok_or("keys 路径")?))
            }
            "--sub" => {
                sub_grams = it
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
    let mut grans_from_manifest: Option<u64> = None;
    if slot_bytes == 0 {
        let mp = view_file.with_extension("manifest.json");
        if mp.exists() {
            let m: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&mp).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            slot_bytes = m["slot_bytes"].as_u64().unwrap_or(4096);
            grans_from_manifest = m["grans"].as_u64();
        } else {
            slot_bytes = 4096;
        }
    }

    let layout = lay();
    let batch = BadgeGather::open(&rows_dir, &layout).map_err(|e| e.to_string())?;

    let mut keys: Vec<u64> = Vec::new();
    if let Some(kf) = &keys_file {
        for l in BufReader::new(std::fs::File::open(kf).map_err(|e| e.to_string())?)
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
    }
    let mut n_grams = if !keys.is_empty() {
        keys.len() / HEAD_W as usize
    } else {
        grans_from_manifest.unwrap_or_default() as usize
    };
    if sub_grams > 0 {
        n_grams = n_grams.min(sub_grams);
    }
    keys.truncate(n_grams * HEAD_W as usize);
    let w = layout.width as usize;
    if n_grams == 0 {
        return Err("无 keys 且 manifest 缺 grans".into());
    }

    // ---- A: 16 行 scatter（无 keys 时跳过：纯视图口径） ----
    let mut out_a = vec![0u8; keys.len() * w];
    if !keys.is_empty() {
        let t0 = std::time::Instant::now();
        batch
            .gather_pp(&keys, &mut out_a, 8)
            .map_err(|e| e.to_string())?;
        let _dt_a = t0.elapsed();
        let p = unique_pages(&keys, &layout);
        println!("A unique 4KiB pages: {} (rows {})", p, keys.len());
    }

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
    if !keys.is_empty() {
        let t3 = std::time::Instant::now();
        batch
            .gather_pp(&keys, &mut out_a, 8)
            .map_err(|e| e.to_string())?;
        let dt_a2 = t3.elapsed();
        csv.push_str(&report("A", keys.len() as u64, dt_a2));
    }

    csv.push_str(&format!(
        "amplification,B,{}\n",
        slot_bytes as f64 / RECORD_BYTES as f64
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
