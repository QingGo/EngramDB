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
use std::path::Path;

use crate::backend::{platform_read_at, platform_read_exact_at};
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

/// 视图读取器：打开已经构建好的 Store-P 视图，按物理槽位读取记录。
/// 这是面向产品面/PyO3 的读取 API（`build_view` 负责写；本类型负责读）。
pub struct ViewReader {
    file: File,
    slot_bytes: u64,
    count: usize,
}

impl ViewReader {
    /// 打开视图文件；优先从 `.manifest.json` 读取槽宽与记录数，缺失时按文件大小推断。
    pub fn open(view_file: &Path) -> std::io::Result<Self> {
        let (slot_bytes, manifest_grans) = slot_of(view_file);
        let file = File::open(view_file)?;
        let meta = file.metadata()?;
        let count_file = meta
            .len()
            .checked_div(slot_bytes)
            .map(|n| n as usize)
            .unwrap_or(0);
        let count = if manifest_grans > 0 {
            count_file.min(manifest_grans as usize)
        } else {
            count_file
        };
        Ok(Self {
            file,
            slot_bytes,
            count,
        })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// 读取一个 gram 的完整 e_t 记录到 `buf`。返回实际读到的槽宽（通常 2560B）。
    pub fn read_record(&self, index: usize, buf: &mut [u8]) -> std::io::Result<usize> {
        if index >= self.count {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "view record index {index} out of range (count {})",
                    self.count
                ),
            ));
        }
        let want = (self.slot_bytes as usize).min(buf.len());
        if want == 0 {
            return Ok(0);
        }
        platform_read_exact_at(&self.file, &mut buf[..want], index as u64 * self.slot_bytes)?;
        Ok(want)
    }

    /// 按物理槽位读取多条记录；`out` 长度必须 >= indices.len() * slot_bytes。
    pub fn read_records(&self, indices: &[usize], out: &mut [u8]) -> std::io::Result<()> {
        let want = self.slot_bytes as usize;
        if out.len() < indices.len() * want {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read_records: output buffer too small",
            ));
        }
        for (j, &idx) in indices.iter().enumerate() {
            self.read_record(idx, &mut out[j * want..(j + 1) * want])?;
        }
        Ok(())
    }
}

/// 视图构建器：持有源表 gather 句柄与槽宽，提供随机采样构建和调用方访问序构建。
pub struct ViewBuilder<'a> {
    batch: &'a BadgeGather<'a>,
    slot_bytes: u64,
}

impl<'a> ViewBuilder<'a> {
    pub fn new(batch: &'a BadgeGather<'a>, slot_bytes: u64) -> Self {
        Self { batch, slot_bytes }
    }

    /// 随机采样构建（原有 LCG 路径）。
    pub fn build_random(
        &self,
        n: usize,
        view_out: &Path,
        keys_out: Option<&Path>,
    ) -> std::io::Result<f64> {
        build_view(self.batch, n, self.slot_bytes, view_out, keys_out)
    }

    /// 按调用方提供的 rowid 访问序构建。
    pub fn build_from_keys(
        &self,
        keys: &[u64],
        view_out: &Path,
        keys_out: Option<&Path>,
    ) -> std::io::Result<f64> {
        build_view_from_keys(self.batch, keys, self.slot_bytes, view_out, keys_out)
    }
}

/// 从调用方提供的 rowid 列表构建视图（`keys` 为 16 头平铺：每 gram 连续 16 行）。
/// 槽位物理顺序 = 调用方给的顺序；因此可用来生成“按访问序排布”的视图：
/// 把实际推理/训练访问的 gram 顺序直接作为 keys 顺序写入，顺序读取即连续 IO。
pub fn build_view_from_keys(
    batch: &BadgeGather,
    keys: &[u64],
    slot_bytes: u64,
    view_out: &Path,
    keys_out: Option<&Path>,
) -> std::io::Result<f64> {
    const CHUNK_ROWS: usize = 500_000 * HEAD_W as usize;
    if !keys.len().is_multiple_of(HEAD_W as usize) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "keys length {} is not a multiple of heads {HEAD_W}",
                keys.len()
            ),
        ));
    }
    let n = keys.len() / HEAD_W as usize;
    let rec_len = (HEAD_W * ROW_BYTES) as usize;
    let mut view = BufWriter::with_capacity(
        64 << 20,
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(view_out)?,
    );
    let mut keys_w = keys_out.map(|p| BufWriter::new(File::create(p).unwrap()));
    let t0 = std::time::Instant::now();
    for chunk in keys.chunks(CHUNK_ROWS) {
        let m = chunk.len() / HEAD_W as usize;
        let mut out = vec![0u8; chunk.len() * ROW_BYTES as usize];
        batch
            .gather_pp(chunk, &mut out, 8)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut slot = vec![0u8; slot_bytes as usize];
        for i in 0..m {
            let rec = &out[i * rec_len..(i + 1) * rec_len];
            slot[..rec.len()].copy_from_slice(rec);
            view.write_all(&slot)?;
        }
        if let Some(w) = keys_w.as_mut() {
            for &r in chunk {
                writeln!(w, "{r}")?;
            }
        }
    }
    view.flush()?;
    let build_s = t0.elapsed().as_secs_f64();
    let manifest = serde_json::json!({
        "grans": n,
        "heads": HEAD_W,
        "slot_bytes": slot_bytes,
        "record_bytes": RECORD_BYTES,
        "build_seconds": build_s,
        "build_mb_s": (n as f64 * slot_bytes as f64 / 1e6) / build_s.max(1e-9),
        "rows": n as u64 * HEAD_W,
        "source": format!("provided-keys:{} shards={}", keys.len(), batch.layout.shards),
        "layout": "access-order",
    });
    std::fs::write(
        view_out.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )?;
    println!(
        "view built from keys: n={n} slot={slot_bytes}B view={} took={build_s:.1}s",
        view_out.display()
    );
    Ok(build_s)
}

/// 流式文件版：从 rowid 文本文件（每行一个 u64）构建访问序视图，适合 keys 文件很大时使用。
pub fn build_view_from_keys_file(
    batch: &BadgeGather,
    keys_path: &Path,
    slot_bytes: u64,
    view_out: &Path,
    keys_out: Option<&Path>,
) -> std::io::Result<f64> {
    const CHUNK_ROWS: usize = 500_000 * HEAD_W as usize;
    let mut reader = BufReader::new(File::open(keys_path)?);
    let mut view = BufWriter::with_capacity(
        64 << 20,
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(view_out)?,
    );
    let mut keys_w = keys_out.map(|p| BufWriter::new(File::create(p).unwrap()));
    let rec_len = (HEAD_W * ROW_BYTES) as usize;
    let t0 = std::time::Instant::now();
    let mut total = 0usize;
    let mut chunk: Vec<u64> = Vec::with_capacity(CHUNK_ROWS);
    let mut line = String::new();
    loop {
        line.clear();
        let r = reader.read_line(&mut line)?;
        if r == 0 {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        chunk.push(
            t.parse::<u64>()
                .map_err(|e: std::num::ParseIntError| std::io::Error::other(e.to_string()))?,
        );
        if chunk.len() == CHUNK_ROWS {
            let m = chunk.len() / HEAD_W as usize;
            let mut out = vec![0u8; chunk.len() * ROW_BYTES as usize];
            batch
                .gather_pp(&chunk, &mut out, 8)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let mut slot = vec![0u8; slot_bytes as usize];
            for i in 0..m {
                let rec = &out[i * rec_len..(i + 1) * rec_len];
                slot[..rec.len()].copy_from_slice(rec);
                view.write_all(&slot)?;
            }
            if let Some(w) = keys_w.as_mut() {
                for &r in &chunk {
                    writeln!(w, "{r}")?;
                }
            }
            total += chunk.len();
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        if !chunk.len().is_multiple_of(HEAD_W as usize) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "keys file has {} rowids, not a multiple of heads {HEAD_W}",
                    chunk.len() + total
                ),
            ));
        }
        let m = chunk.len() / HEAD_W as usize;
        let mut out = vec![0u8; chunk.len() * ROW_BYTES as usize];
        batch
            .gather_pp(&chunk, &mut out, 8)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut slot = vec![0u8; slot_bytes as usize];
        for i in 0..m {
            let rec = &out[i * rec_len..(i + 1) * rec_len];
            slot[..rec.len()].copy_from_slice(rec);
            view.write_all(&slot)?;
        }
        if let Some(w) = keys_w.as_mut() {
            for &r in &chunk {
                writeln!(w, "{r}")?;
            }
        }
        total += chunk.len();
    }
    if total == 0 || !total.is_multiple_of(HEAD_W as usize) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("keys file total {total} rowids not multiple of heads"),
        ));
    }
    view.flush()?;
    let build_s = t0.elapsed().as_secs_f64();
    let n = total / HEAD_W as usize;
    let manifest = serde_json::json!({
        "grans": n,
        "heads": HEAD_W,
        "slot_bytes": slot_bytes,
        "record_bytes": RECORD_BYTES,
        "build_seconds": build_s,
        "build_mb_s": (n as f64 * slot_bytes as f64 / 1e6) / build_s.max(1e-9),
        "rows": n as u64 * HEAD_W,
        "source": format!("provided-keys-file:{} shards={}", total, batch.layout.shards),
        "layout": "access-order",
    });
    std::fs::write(
        view_out.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )?;
    println!(
        "view built from keys file: n={n} slot={slot_bytes}B view={} took={build_s:.1}s",
        view_out.display()
    );
    Ok(build_s)
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
    order_mode: &str,
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
    let mut order: Vec<u64>;
    if order_mode == "seq" {
        order = (0..n_grams as u64).collect();
    } else {
        let mut g_state: u64 = 0xCAFE_BEEF_0F1E_2D3C;
        order = Vec::with_capacity(n_grams);
        for _ in 0..n_grams {
            g_state = g_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.push(g_state % n_grams as u64);
        }
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
                        let _ = platform_read_exact_at(fb, &mut buf, rec * slot_bytes);
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
            let rd = platform_read_at(&f, &mut buf[..want], off)?;
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
            let _ = platform_read_exact_at(view_ref, &mut buf, rec * slot_bytes);
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

/// 校验视图：用 keys（每 gram 连续 16 个 rowid）从源表重新 gather，和视图记录逐字节比对。
/// 默认抽样 1000 个 gram；`sub_grams` 可显式指定检查数量。
pub fn verify_view(
    batch: &BadgeGather,
    view_file: &Path,
    keys: Option<&[u64]>,
    sub_grams: usize,
) -> std::io::Result<()> {
    let reader = ViewReader::open(view_file)?;
    let n = reader.len();
    if n == 0 {
        return Err(std::io::Error::other("视图为空"));
    }
    let keys = keys.unwrap_or(&[]);
    if keys.is_empty() {
        return Err(std::io::Error::other(
            "verify 需要 keys 文件（每 gram 连续 16 个 rowid）",
        ));
    }
    let key_grams = keys.len() / HEAD_W as usize;
    if key_grams < n {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("keys 文件只有 {key_grams} grams，不足以覆盖视图 {n} grams"),
        ));
    }
    let total = n.min(key_grams);
    let check = if sub_grams > 0 {
        sub_grams.min(total)
    } else {
        total.min(1000)
    };
    let slot = reader.slot_bytes() as usize;
    let mut view_buf = vec![0u8; slot];
    let mut src_buf = vec![0u8; RECORD_BYTES as usize];
    for gi in 0..check {
        let start = gi * HEAD_W as usize;
        let rowids = &keys[start..start + HEAD_W as usize];
        batch
            .gather_pp(rowids, &mut src_buf, 8)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let got = reader.read_record(gi, &mut view_buf)?;
        if got < RECORD_BYTES as usize || view_buf[..RECORD_BYTES as usize] != src_buf[..] {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("view record {gi} does not match source rows"),
            ));
        }
    }
    println!("view verified: checked {check}/{total} grams, slot={slot}B, all match");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn build_from_keys_and_view_reader_roundtrip() {
        let dir = std::env::temp_dir().join("engramdb-view-reader-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let layout = Layout::new(1, 100, ROW_BYTES, 1); // 100 rows × 160B
        let shard = dir.join("shard_000.bin");
        let mut f = File::create(&shard).unwrap();
        let mut row = vec![0u8; ROW_BYTES as usize];
        for r in 0..100u64 {
            row.fill((r % 251) as u8);
            f.write_all(&row).unwrap();
        }
        drop(f);
        let bg = BadgeGather::open(&dir, &layout).unwrap();
        let keys: Vec<u64> = (0..32).collect(); // 2 grams × 16 heads
        let view_path = dir.join("ordered.bin");
        let keys_path = dir.join("ordered.keys.txt");
        build_view_from_keys(&bg, &keys, RECORD_BYTES, &view_path, Some(&keys_path)).unwrap();

        let vr = ViewReader::open(&view_path).unwrap();
        assert_eq!(vr.len(), 2);
        assert_eq!(vr.slot_bytes(), RECORD_BYTES);
        let mut buf = vec![0u8; RECORD_BYTES as usize];
        let got = vr.read_record(0, &mut buf).unwrap();
        assert_eq!(got, RECORD_BYTES as usize);
        for i in 0..16usize {
            let r = i as u64;
            let expect = (r % 251) as u8;
            assert_eq!(buf[i * ROW_BYTES as usize], expect, "row {r} first byte");
        }

        let mut two = vec![0u8; 2 * RECORD_BYTES as usize];
        vr.read_records(&[1, 0], &mut two).unwrap();
        // record 1 begins with row 16 value 16
        assert_eq!(two[0], 16);

        verify_view(&bg, &view_path, Some(&keys), 0).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }
}
