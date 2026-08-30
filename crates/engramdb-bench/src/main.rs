//! M0 探针：P1 布局/批取微基准（mock 表，真实指标口径）。
//!
//! 输出：有效吞吐（行/s、MB/s）、p50/p95 延迟（批粒度）、字节放大率。

use std::path::Path;
use std::time::Instant;

#[cfg(unix)]
use engramdb_core::layout::Layout;
#[cfg(unix)]
use engramdb_core::store::ShardedStore;
#[cfg(unix)]
use engramdb_keygen::PleSpec;

#[cfg(windows)]
fn main() {
    eprintln!("engramdb-bench: M0 探针仅 unix 平台（需要 ShardedStore）；请用 engramdb view/p4view 探针组");
    std::process::exit(1);
}

#[cfg(unix)]
fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = String::from("data/mock-qwen38-ple");
    let mut batch = 4096usize;
    let mut iters = 16usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().unwrap(),
            "--batch" => batch = args.next().unwrap().parse().unwrap(),
            "--iters" => iters = args.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }

    // P1 使用 keygen 生成的 rowid（真实寻址语义）
    let spec = PleSpec::real();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let mock_rows = 8 * 156_251u64;
    let keys: Vec<u64> = (0..batch)
        .map(|_| {
            if next() & 1 == 0 {
                next() % mock_rows
            } else {
                next() % spec.padded % mock_rows
            }
        })
        .collect();

    let layout = Layout::new(8, 156_251, 160, 1);
    let store = ShardedStore::open(Path::new(&dir), layout)
        .unwrap_or_else(|e| panic!("open {}: {}", dir, e));

    let w = store.layout.width as usize;
    let mut out = vec![0u8; batch * w];

    store.gather(&keys, &mut out).unwrap();

    let mut times = Vec::with_capacity(iters);
    let payload_bytes = (keys.len() as u64) * store.layout.row_bytes;
    for _ in 0..iters {
        let t = Instant::now();
        store.gather(&keys, &mut out).unwrap();
        times.push(t.elapsed());
        black_box(&out);
    }

    let p50 = percentile(&mut times, 0.5);
    let p95 = percentile(&mut times, 0.95);
    let total_time: std::time::Duration = times.iter().sum();
    let rows_per_s = keys.len() as f64 * iters as f64 / total_time.as_secs_f64();
    let mb_per_s = payload_bytes as f64 * iters as f64 / 1e6 / total_time.as_secs_f64();
    let badge_bytes = store.layout.badge_bytes();
    let amplification = badge_bytes as f64 / payload_bytes as f64;
    println!("P1 gather: batch={} iters={}", batch, iters);
    println!("  rows/s        = {:.0}", rows_per_s);
    println!("  payload MB/s  = {:.1}", mb_per_s);
    println!("  batch p50     = {:?}  p95 = {:?}", p50, p95);
    println!(
        "  badge overhead = {:.3}x  ({} bytes/badge vs {:.0}B payload/batch)",
        amplification, badge_bytes, payload_bytes as f64
    );
}

fn percentile(xs: &mut [std::time::Duration], p: f64) -> std::time::Duration {
    xs.sort();
    let idx = (((xs.len() - 1) as f64) * p).round() as usize;
    xs[idx]
}

fn black_box<T>(x: &T) {
    // rustc<1.66 无 std::hint::black_box；volatile 读兜底
    unsafe {
        std::ptr::read_volatile(&(x as *const T as usize));
    }
}
