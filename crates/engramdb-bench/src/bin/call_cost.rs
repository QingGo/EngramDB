//! call_cost — 把 `gather_pp` 的**每次调用固定开销**拆开计时。
//!
//! 动机（roadmap §32.2 的实测）：V4.1 几何、tokens=1（48 行）时，
//! warm 读数在 t=1 是 **459 μs/call**、t=32 是 **515 μs/call** ——
//! **两者只差 56 μs**。如果这个差值是线程创建的全部代价，那么
//! "建常驻线程池"就只值 11%，不是主要项。所以先把各组成部分单独量出来，
//! 再决定优化什么。
//!
//! 拆的项（都在 warm/已缓存数据上做，所以测的是**调用开销**而不是介质）：
//!   A. `gather_pp` 完整调用（t=1 / t=32）        —— 目标总和
//!   B. 空 `thread::scope` + N 个 no-op spawn     —— 线程创建
//!   C. 按 shard 分组 + 排序（HashMap + sort）    —— 计划开销
//!   D. 每 shard 一个 `vec![0u8; PAGE+2*row]`     —— 缓冲分配 + 清零
//!   E. 结果搬运（`extend_from_slice` + 回填）    —— 组装开销
//!
//! 用法：
//!   call_cost <dir> <shards> <rows_per_shard> <width> <nkeys_csv> [reps]

use std::collections::HashMap;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;

const PAGE: u64 = 4096;

/// 空 scope + n 个 no-op spawn，一次多少 μs。
fn cost_spawn(nspawn: usize, reps: usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..reps {
        std::thread::scope(|s| {
            let mut hs = Vec::with_capacity(nspawn);
            for _ in 0..nspawn {
                hs.push(s.spawn(|| {
                    black_box(0u64);
                }));
            }
            for h in hs {
                let _ = h.join();
            }
        });
    }
    black_box(t0.elapsed());
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
}

/// 按 shard 分组（与 `gather_pp` 同构）+ 每组排序。
fn cost_group_sort(keys: &[u64], layout: &Layout, reps: usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..reps {
        let mut groups: HashMap<u64, Vec<(u64, usize)>> = HashMap::new();
        for (i, &k) in keys.iter().enumerate() {
            let (shard, _, _) = layout.locate(k);
            groups.entry(shard).or_default().push((k, i));
        }
        let mut tasks: Vec<(u64, Vec<(u64, usize)>)> = groups.into_iter().collect();
        tasks.sort_unstable_by_key(|&(s, _)| s);
        for (_, v) in tasks.iter_mut() {
            v.sort_unstable();
        }
        black_box(&tasks);
    }
    black_box(t0.elapsed());
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
}

/// 每个 shard 一个 `vec![0u8; PAGE + 2*row_bytes]`（零初始化），并触碰它。
fn cost_alloc_buffers(nshards: usize, row_bytes: u64, reps: usize) -> f64 {
    let sz = (PAGE + 2 * row_bytes) as usize;
    let t0 = Instant::now();
    for _ in 0..reps {
        let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(nshards);
        for _ in 0..nshards {
            let mut b = vec![0u8; sz];
            b[0] = 1;
            b[sz - 1] = 2;
            bufs.push(b);
        }
        black_box(&bufs);
    }
    black_box(t0.elapsed());
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
}

/// 结果搬运：每组 `extend_from_slice(row)` 再回填到 out。
fn cost_assemble(keys: &[u64], width: usize, row_bytes: usize, reps: usize) -> f64 {
    let src = vec![7u8; row_bytes];
    let t0 = Instant::now();
    for _ in 0..reps {
        let mut out_rows: Vec<u8> = Vec::new();
        let mut out_idxs: Vec<usize> = Vec::new();
        for (i, _) in keys.iter().enumerate() {
            out_rows.extend_from_slice(&src);
            out_idxs.push(i);
        }
        let mut out = vec![0u8; keys.len() * width];
        for (j, &oi) in out_idxs.iter().enumerate() {
            out[oi * width..(oi + 1) * width]
                .copy_from_slice(&out_rows[j * width..(j + 1) * width]);
        }
        black_box(&out);
    }
    black_box(t0.elapsed());
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
}

fn cost_gather_pp(
    batch: &BadgeGather<'_>,
    keys: &[u64],
    out: &mut [u8],
    threads: usize,
    reps: usize,
) -> f64 {
    // 预热一次
    batch.gather_pp(keys, out, threads).expect("warmup");
    let t0 = Instant::now();
    for _ in 0..reps {
        batch
            .gather_pp(black_box(keys), out, threads)
            .expect("gather");
    }
    black_box(t0.elapsed());
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        eprintln!("usage: call_cost <dir> <shards> <rows_per_shard> <width> <nkeys_csv> [reps]");
        std::process::exit(2);
    }
    let dir = Path::new(&args[0]);
    let shards: u64 = args[1].parse().unwrap();
    let rows_per_shard: u64 = args[2].parse().unwrap();
    let width: u64 = args[3].parse().unwrap();
    let nkeys_list: Vec<usize> = args[4].split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(200);

    let layout = Layout::new(shards, rows_per_shard, width, 1);
    let batch = BadgeGather::open(dir, &layout).expect("open");
    let total = layout.total_rows();

    // ---- 先量 spawn 曲线：这是"要不要建常驻线程池"的唯一依据 ----
    println!("== spawn 曲线（空 scope + n 个 no-op spawn，{reps} 次均值）==");
    println!("{:>6} {:>12} {:>14}", "n", "μs/call", "μs/spawn");
    let mut base = 0.0;
    for n in [0usize, 1, 2, 4, 8, 16, 32, 64] {
        let us = if n == 0 {
            // n=0：只创建 scope 本身（不 spawn）
            let t0 = Instant::now();
            for _ in 0..reps {
                std::thread::scope(|_s| {
                    black_box(0u64);
                });
            }
            black_box(t0.elapsed());
            t0.elapsed().as_secs_f64() * 1e6 / reps as f64
        } else {
            cost_spawn(n, reps)
        };
        if n == 0 {
            base = us;
        }
        let per = if n > 0 { (us - base) / n as f64 } else { 0.0 };
        println!("{:>6} {:>12.1} {:>14.1}", n, us, per);
    }
    println!("   （`μs/spawn` = 相对 n=0 的增量；n=0 本身 {base:.1} μs 是 scope 的固定成本）\n");

    println!(
        "== call_cost  dir={}  shards={}  width={}  reps={} ==",
        dir.display(),
        shards,
        width,
        reps
    );
    println!("   全部在 warm（已缓存）数据上做 ⇒ 测的是**调用开销**，不是介质\n");
    println!(
        "{:>7} {:>7} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "nkeys",
        "shards#",
        "gather_t1",
        "gather_t32",
        "scope_N",
        "group+sort",
        "alloc_buf",
        "assemble",
        "unaccounted"
    );

    // 确定性的 key 生成（与 nvme_gate 的双射同族，但这里只要分散即可）
    for &nkeys in &nkeys_list {
        let mut s: u64 = 0xC0FFEE;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        };
        let keys: Vec<u64> = (0..nkeys).map(|_| next() % total).collect();
        let nshards = {
            let mut set = std::collections::HashSet::new();
            for &k in &keys {
                set.insert(k / rows_per_shard);
            }
            set.len()
        };

        let mut out = vec![0u8; nkeys * width as usize];
        let gt1 = cost_gather_pp(&batch, &keys, &mut out, 1, reps);
        let gt32 = cost_gather_pp(&batch, &keys, &mut out, 32, reps);
        let sc = cost_spawn(nshards.min(32), reps);
        let gs = cost_group_sort(&keys, &layout, reps);
        let ab = cost_alloc_buffers(nshards, layout.row_bytes, reps);
        let asm = cost_assemble(&keys, width as usize, layout.row_bytes as usize, reps);
        let unacc = gt1 - sc - gs - ab - asm;

        println!(
            "{:>7} {:>7} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>9.1} {:>9.1} {:>9.1}",
            nkeys, nshards, gt1, gt32, sc, gs, ab, asm, unacc
        );
    }
    println!("\n   单位 μs/次调用。`unaccounted` = gather_t1 减去其余四项（可为负 = 估值偏高）。");
}
