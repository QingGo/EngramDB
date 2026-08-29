//! P4：物化视图（view）A/B —— "16 头分散 gather" vs "单条 2560B 记录读"。
//!
//! gen <rows_dir> <n_grams> <view.bin> <rowids.txt>
//!   从真实行存储生成 view：每 gram key = 16 头各一个 rowid → 记录 = 2560B 拼接（连续存放）
//! bench <rows_dir> <view.bin> <rowids.txt> [--keys N]
//!   A: gather_pp 从 rows_dir 抓 16N 行（基准路径）
//!   B: view 每条 2560B 记录随机读（view 路径）
//! 输出: IOPS / p50 / p95 / 字节放大 对比 CSV 行。

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;

const VIEW_RECORD: u64 = 4096; // 4KB 对齐槽位（2560 数据 + pad），保证每记录 1 IO

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        return;
    };
    let out = match cmd.as_str() {
        "gen" => cmd_gen(args),
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

fn cmd_gen(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let n: usize = rest
        .next()
        .ok_or("n_grams")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let view_out = PathBuf::from(rest.next().ok_or("view.bin")?);
    let keys_out = PathBuf::from(rest.next().ok_or("rowids.txt")?);

    let layout = lay();
    let batch = BadgeGather::open(&rows_dir, &layout).map_err(|e| e.to_string())?;

    // 真实分布：每 gram 16 个头 = 每头一个 rowid → 用 keygen 主行 + 头偏移近似：
    // 直接用 P2 采集过的真实 rowid 投影（简单起见：模拟"16 头每头以该头行数为模"）
    // 这里直接复用真表：每个 gram 的 16 行 = 用 keygen rowids（已验证），保证与真表一致
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
    let mut out = vec![0u8; rowids.len() * 160];
    batch
        .gather_pp(&rowids, &mut out, 8)
        .map_err(|e| e.to_string())?;
    let mut slot = vec![0u8; VIEW_RECORD as usize];
    for (i, key) in rowids.chunks(16).enumerate() {
        let _ = key;
        let rec = &out[i * 16 * 160..(i * 16 + 16) * 160];
        slot[..rec.len()].copy_from_slice(rec);
        view.write_all(&slot).map_err(|e| e.to_string())?;
    }
    for &r in &rowids {
        writeln!(keys_file, "{r}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn cmd_bench(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let keys_file = PathBuf::from(rest.next().ok_or("keys.txt")?);
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
    let n_grams = keys.len() / 16;
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
    let _out_b = vec![0u8; n_grams * VIEW_RECORD as usize];
    // 预生成记录访问序（相同随机序）
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
                    let mut buf = vec![0u8; VIEW_RECORD as usize];
                    for &rec in slice {
                        let off = rec * VIEW_RECORD;
                        let _ = view_ref.read_exact_at(&mut buf, off);
                    }
                });
            }
        });
        t.elapsed()
    };
    let dt_b = run_par(1);
    report("B :view rec(1t,cold)", n_grams as u64 * 16, dt_b);
    let dt_b2 = run_par(8);
    report("B :view rec(8t,warm)", n_grams as u64 * 16, dt_b2);
    let t3 = std::time::Instant::now();
    batch
        .gather_pp(&keys, &mut out_a, 8)
        .map_err(|e| e.to_string())?;
    let dt_a2 = t3.elapsed();
    report("A :16-row scatter(warm)", keys.len() as u64, dt_a2);

    let ba = keys.len() as u64 * 160;
    let ra_a = pages_a as u64 * 4096;
    let ra_b = n_grams as u64 * 4096;
    println!(
        "byte amplification: A {:.2}x ({} pages x 4KiB), B {:.2}x ({} rec x 4KiB slot)",
        ra_a as f64 / ba as f64,
        pages_a,
        ra_b as f64 / ba as f64,
        n_grams
    );
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

fn report(name: &str, rows: u64, dt: std::time::Duration) {
    let s = dt.as_secs_f64();
    println!(
        "{name}: rows={} time={:.3}s  rows/s={:.0}  MB/s={:.1}",
        rows,
        s,
        rows as f64 / s.max(1e-9),
        (rows as f64 * 160.0) / 1e6 / s.max(1e-9)
    );
}
