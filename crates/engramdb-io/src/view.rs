//! Store-P 物化视图（P4）：构建器 + 视图读写/基准/延迟原语。
//!
//! 视图 = 每 gram 16 头行（160B×16=2560B 紧凑槽，可选 4096 对齐槽）连续物化，
//! 1 次定长读替代 16 路随机行读（IOPS 16:1、字节放大 1.00× vs scatter 20×）。
//! build/bench/lat 原语与探针（p4view）及 CLI（`engramdb view`）共用；
//! 输出格式保持探测基线（gate.sh + probes/*.csv 解析）不变。
//!
//! 规模：流式分块（500K grams/chunk）→ 16GB 内存机可构建全表；manifest.json
//! 记录 grans/slot/耗时（lat/bench 以 manifest 为 n 的缺省来源）。

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::batch::BadgeGather;
use engramdb_core::layout::Layout;

pub const HEAD_W: u64 = 16;
pub const ROW_BYTES: u64 = 160;
pub const RECORD_BYTES: u64 = HEAD_W * ROW_BYTES; // 2560B

fn slot_of(view_file: &Path) -> (u64, u64) {
    let mp = view_file.with_extension("manifest.json");
    if mp.exists() {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap_or_default())
            .unwrap_or(serde_json::Value::Null);
        let slot = m["slot_bytes"].as_u64().unwrap_or(RECORD_BYTES);
        let grans = m["grans"].as_u64().unwrap_or(0);
        (slot, grans)
    } else {
        (RECORD_BYTES, 0)
    }
}

/// 构建视图：固定 LCG seed 按 gram 序列生成 16 头 rowid，流式分块 gather 后写入
/// 定长槽；manifest 写 `<view>.manifest.json`；keys（rowid 列表）写 `keys_out`
/// （None = 不写；n 已在 manifest，读取方可 B-only）。
pub fn build_view(
    batch: &BadgeGather,
    n: usize,
    slot_bytes: u64,
    view_out: &Path,
    keys_out: Option<&Path>,
) -> std::io::Result<f64> {
    const CHUNK_G: usize = 500_000;
    let spec = engramdb_keygen::PleSpec::real();
    let mut rng_state: u64 = 0xDEAD_BEEF_1234_5678;
    let mut keys = keys_out.map(|p| BufWriter::new(File::create(p).unwrap()));
    let mut view = BufWriter::with_capacity(
        64 << 20,
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(view_out)?,
    );
    let t0 = std::time::Instant::now();
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
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut s = vec![0u8; slot_bytes as usize];
        let rec_len = (HEAD_W * ROW_BYTES) as usize;
        for i in 0..m {
            let rec = &out[i * rec_len..(i + 1) * rec_len];
            s[..rec.len()].copy_from_slice(rec);
            view.write_all(&s)?;
        }
        if let Some(w) = keys.as_mut() {
            for &r in &rowids {
                writeln!(w, "{r}")?;
            }
        }
        done += m;
    }
    view.flush()?;
    let build_s = t0.elapsed().as_secs_f64();
    let m = serde_json::json!({
        "grans": n,
        "heads": HEAD_W,
        "slot_bytes": slot_bytes,
        "record_bytes": RECORD_BYTES,
        "build_seconds": build_s,
        "build_mb_s": (n as f64 * slot_bytes as f64 / 1e6) / build_s,
        "rows": n as u64 * HEAD_W,
        "source": format!("shards={}", batch.layout.shards),
    });
    std::fs::write(
        view_out.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&m).unwrap(),
    )?;
    println!(
        "view built: n={n} slot={slot_bytes}B view={} took={build_s:.1}s",
        view_out.display()
    );
    Ok(build_s)
}

/// A 路径（16 行 scatter）唯一 4KiB 页数（字节放大的分母）。
pub fn unique_pages(keys: &[u64], layout: &Layout) -> usize {
    let mut set = std::collections::HashSet::with_capacity(keys.len() / 8);
    for &k in keys {
        set.insert(k * layout.row_bytes / 4096);
    }
    set.len()
}

/// 从 keys 文件读 rowid（trim + 空行跳过）。
pub fn read_keys(p: &Path) -> std::io::Result<Vec<u64>> {
    let mut k = Vec::new();
    for l in BufReader::new(File::open(p)?).lines().map_while(Result::ok) {
        let l = l.trim();
        if !l.is_empty() {
            k.push(
                l.parse::<u64>()
                    .map_err(|e: std::num::ParseIntError| std::io::Error::other(e.to_string()))?,
            );
        }
    }
    Ok(k)
}

fn report(name: &str, rows: u64, dt: std::time::Duration) -> String {
    let s = dt.as_secs_f64();
    let rps = rows as f64 / s.max(1e-9);
    let mbps = rows as f64 * ROW_BYTES as f64 / 1e6 / s.max(1e-9);
    println!("{name}: rows={rows} time={s:.3}s  rows/s={rps:.0}  MB/s={mbps:.1}");
    format!("{name},{rows},{:.0},{:.1}\n", rps, mbps)
}

/// 吞吐基准：A（scatter 对照, 有 keys 时）+ B（视图单记录读, 1t & 8t 两档）。
/// 返回 (A_rows_per_s, B8t_rows_per_s, ampl)。CSV 打印保持探测格式。
pub fn bench_view(
    batch: &BadgeGather,
    view_file: &Path,
    keys: Option<&[u64]>,
    sub_grams: usize,
    threads: usize,
    req_slot: u64,
) -> std::io::Result<(f64, f64, f64)> {
    let (slot_bytes, grans) = if req_slot > 0 {
        (req_slot, 0)
    } else {
        slot_of(view_file)
    };
    let keys = keys.unwrap_or(&[]);
    let mut n_grams = if !keys.is_empty() {
        keys.len() / HEAD_W as usize
    } else {
        grans as usize
    };
    if sub_grams > 0 {
        n_grams = n_grams.min(sub_grams);
    }
    if n_grams == 0 {
        return Err(std::io::Error::other("无 keys 且 manifest 缺 grans"));
    }
    let w = batch.layout.width as usize;

    // A: 16 行 scatter（unique 页统计 + 8t 计时）
    let out_a_len = keys.len() * w;
    let mut out_a = vec![0u8; out_a_len];
    if !keys.is_empty() {
        let t0 = std::time::Instant::now();
        batch
            .gather_pp(keys, &mut out_a, 8)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let dt_a = t0.elapsed();
        let p = unique_pages(&keys[..n_grams * HEAD_W as usize], batch.layout);
        println!("A unique 4KiB pages: {} (rows {})", p, keys.len());
        let _ = dt_a;
    }

    // B: view 单记录读（固定 LCG 随机序）
    let vf = File::open(view_file)?;
    let meta = vf.metadata()?;
    let grans_actual = (meta.len() / slot_bytes) as usize;
    let n_grams = n_grams.min(grans_actual);
    let mut g_state: u64 = 0xCAFE_BEEF_0F1E_2D3C;
    let mut order: Vec<u64> = Vec::with_capacity(n_grams);
    for _ in 0..n_grams {
        g_state = g_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        order.push(g_state % n_grams as u64);
    }
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
                let fb = &vf;
                sc.spawn(move || {
                    let mut buf = vec![0u8; slot_bytes as usize];
                    for &rec in slice {
                        let _ = fb.read_exact_at(&mut buf, rec * slot_bytes);
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
    let a_rps = if !keys.is_empty() {
        let t3 = std::time::Instant::now();
        batch
            .gather_pp(&keys[..n_grams * HEAD_W as usize], &mut out_a, 8)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let dt_a2 = t3.elapsed();
        let a = report("A", (n_grams * HEAD_W as usize) as u64, dt_a2);
        csv.push_str(&a);
        (n_grams * HEAD_W as usize) as f64 / dt_a2.as_secs_f64().max(1e-9)
    } else {
        0.0
    };
    csv.push_str(&format!(
        "amplification,B,{}\n",
        slot_bytes as f64 / RECORD_BYTES as f64
    ));
    let b_rps = (n_grams * HEAD_W as usize) as f64 / dt_b2.as_secs_f64().max(1e-9);
    println!("{csv}");
    Ok((a_rps, b_rps, slot_bytes as f64 / RECORD_BYTES as f64))
}

/// Fisher-Yates 随机访问序（xorshift64；固定 seed 可复现）。
pub fn rand_order(n: usize, seed: u64) -> Vec<u64> {
    let mut st = seed;
    let mut v: Vec<u64> = (0..n as u64).collect();
    for i in (1..n).rev() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let j = (st % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// 延迟分布：随机序单记录读 × N（每查独立计时）→ p50/p95/p99/max/mean（μs）。
/// warm=全文件顺序预读；cold=Linux fadvise(DONTNEED)（真冷，需 Linux）。
pub fn lat_view(
    view_file: &Path,
    threads: usize,
    warm: bool,
    cold: bool,
    sub_grams: usize,
    req_slot: u64,
) -> std::io::Result<()> {
    let (slot_bytes, manifest_n) = if req_slot > 0 {
        (req_slot, 0usize)
    } else {
        let (s, g) = slot_of(view_file);
        (s, g as usize)
    };
    let vf = File::open(view_file)?;
    let meta = vf.metadata()?;
    let n_all = (meta.len() / slot_bytes) as usize;
    let n = if sub_grams > 0 {
        n_all.min(sub_grams)
    } else if manifest_n > 0 {
        n_all.min(manifest_n)
    } else {
        n_all
    };
    if n == 0 {
        return Err(std::io::Error::other("视图为空"));
    }
    if warm {
        let mut buf = vec![0u8; 8 << 20];
        let mut off = 0u64;
        let f = vf.try_clone()?;
        while off < meta.len() {
            let want = (buf.len() as u64).min(meta.len() - off) as usize;
            let rd = f.read_at(&mut buf[..want], off)?;
            if rd == 0 {
                break;
            }
            off += rd as u64;
        }
    }
    let order = rand_order(n, 0xFEED_BEEF_0D0F_1E2C);
    let view_ref = &vf;
    #[cfg(not(target_os = "linux"))]
    let _ = cold;
    let per_thread = |tid: usize| -> Vec<u32> {
        let mut buf = vec![0u8; slot_bytes as usize];
        let mut out = Vec::new();
        let stride = order.len() / threads;
        let lo = tid * stride;
        let hi = if tid + 1 == threads {
            order.len()
        } else {
            lo + stride
        };
        for &rec in &order[lo..hi] {
            let t0 = std::time::Instant::now();
            #[cfg(target_os = "linux")]
            if cold {
                use std::os::fd::AsRawFd;
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
            out.push(t0.elapsed().as_nanos().min(u32::MAX as u128) as u32);
        }
        out
    };
    let mut times: Vec<u32> = std::thread::scope(|sc| {
        let mut h = Vec::new();
        for tid in 0..threads {
            h.push(sc.spawn(move || per_thread(tid)));
        }
        let mut all = Vec::with_capacity(n);
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
    let (p50, p95, p99, mx) = (
        p(0.50),
        p(0.95),
        p(0.99),
        *times.last().unwrap_or(&0) as f64,
    );
    let us = |ns: f64| ns / 1000.0;
    println!(
        "lat: n={n} slot={slot_bytes} threads={threads} {} [μs] p50={:.2} p95={:.2} p99={:.2} max={:.2} mean={:.2}",
        if warm { "warm" } else { "cold" },
        us(p50), us(p95), us(p99), us(mx), us(mean)
    );
    println!(
        "latency_us,p50,p95,p99,max,mean,{},{},{},{},{}\n",
        us(p50),
        us(p95),
        us(p99),
        us(mx),
        us(mean)
    );
    Ok(())
}
