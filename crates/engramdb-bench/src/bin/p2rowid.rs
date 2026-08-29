//! T3(Rust)：PLE rowid 稀疏统计（替代 numpy 320M 全量 bincount）。
//! 输入：token 目录（*.u32.npy，np.save 原生格式）→ 输出 stats json。
//! 用法: p2rowid <tokens_dir> <out.json>
//! 进度条：\r 行内刷新（nohup 场景用 tr '\r' '\n' 查看）。

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use engramdb_keygen::PleSpec;

const CHUNK_TOKENS: usize = 1_000_000;

fn read_tokens(path: &PathBuf) -> Vec<u32> {
    let data = std::fs::read(path).expect("npy");
    let mut idx = 6usize;
    let major = data[idx];
    idx += 1;
    let _minor = data[idx];
    idx += 1;
    let hlen = if major == 1 {
        u16::from_le_bytes([data[idx], data[idx + 1]]) as usize
    } else {
        u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]) as usize
    };
    idx += if major == 1 { 2 } else { 4 };
    let _header = std::str::from_utf8(&data[idx..idx + hlen]).unwrap();
    let raw = data[idx + hlen..].to_vec();
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn header_n(path: &PathBuf) -> usize {
    let data = std::fs::read(path).expect("npy");
    let mut idx = 6usize;
    let major = data[idx];
    idx += 1;
    idx += 1;
    if major == 1 {
        let l = u16::from_le_bytes([data[idx], data[idx + 1]]) as usize;
        idx += 2;
        let header = std::str::from_utf8(&data[idx..idx + l]).unwrap();
        shape_n(header)
    } else {
        let l =
            u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]) as usize;
        idx += 4;
        let header = std::str::from_utf8(&data[idx..idx + l]).unwrap();
        shape_n(header)
    }
}

fn shape_n(header: &str) -> usize {
    let s = header
        .split(',')
        .find(|p| p.contains("shape"))
        .and_then(|p| p.split(':').nth(1))
        .unwrap_or("(0,)");
    let inner: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    inner.parse().unwrap_or(0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "data/p2-work/tokens/fineweb".into()),
    );
    let out = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "data/p2-work/stats/fineweb_rowid.json".into()),
    );

    let spec = PleSpec::real();
    let files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "npy").unwrap_or(false))
        .collect();
    let total_tokens: usize = files.iter().map(header_n).sum();

    let mut counts: HashMap<u32, u32> = HashMap::new();
    let mut total_gets: u64 = 0;
    let t0 = Instant::now();
    let mut done_tokens = 0usize;

    for f in &files {
        let tokens = read_tokens(f);
        for chunk in tokens.chunks(CHUNK_TOKENS) {
            let ids = spec.rowids_for_seq(chunk);
            for row in &ids {
                for &r in row {
                    *counts.entry(r).or_insert(0) += 1;
                    total_gets += 1;
                }
            }
            done_tokens += chunk.len();
            let pct = done_tokens as f64 / total_tokens.max(1) as f64 * 100.0;
            let elapsed = t0.elapsed().as_secs_f64();
            let eta = if pct > 0.0 {
                elapsed / pct * (100.0 - pct)
            } else {
                0.0
            };
            eprint!(
                "\r[p2rowid] {:5.1}%  tokens {}/{}  {:7.2}s  ETA {:6.1}s  unique_rows {}  gets {}",
                pct,
                done_tokens,
                total_tokens,
                elapsed,
                eta,
                counts.len(),
                total_gets
            );
        }
        eprintln!();
    }
    eprintln!("[p2rowid] finalizing counts...");
    let mut vals: Vec<u32> = counts.values().copied().collect();
    vals.sort_unstable_by(|a, b| b.cmp(a));
    let mut tier = serde_json::Map::new();
    for (k, limit) in [
        ("100", 100usize),
        ("1000", 1000),
        ("10000", 10000),
        ("100000", 100000),
        ("1000000", 1000000),
    ] {
        let lim = limit.min(vals.len());
        let c: u64 = vals[..lim].iter().map(|&v| v as u64).sum();
        tier.insert(
            k.to_string(),
            serde_json::json!(round3(c as f64 / total_gets as f64 * 100.0)),
        );
    }
    let res = serde_json::json!({
        "unique_rows": counts.len(),
        "total_gets": total_gets,
        "flat_unique_pct": round3(counts.len() as f64 / total_gets as f64 * 100.0),
        "tier_curve_top_rows": serde_json::Value::Object(tier),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&res).unwrap()).ok();
    println!(
        "\n[p2rowid] done in {:.1}s -> {}",
        t0.elapsed().as_secs_f64(),
        out.display()
    );
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}
