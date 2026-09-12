//! uring_gate — io_uring 批量页读（`UringBatchBackend::read_many`）的冷读能力门禁。
//!
//! 动机：`BadgeGather::gather_pp` 走同步 `pread`，其**队列深度 = 线程数**。
//! §30.8 实测：8 线程 12.76 μs/行、16 线程 7.46、32 线程 4.73 ——
//! 提升完全来自并发度，说明瓶颈是 QD 而不是介质。
//! 本门禁把同一批 keys 换成「按 shard 分组 → 一次 `read_many` 提交 N 个 SQE」，
//! 看**单线程**能否用深队列拿到同等或更好的吞吐。
//!
//! 口径与 `nvme_gate` 完全一致：双射 + 轮间不相交切片 + `fadvise(DONTNEED)`
//! + 结尾冷热自校验（比值 < 5x ⇒ VOID，退出码 3）。
//!
//! 用法：
//!   uring_gate <dir> <shards> <rows_per_shard> <width> \
//!              <rows_per_token> <tokens_per_round> <rounds> [threads_csv] [seed]

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("uring_gate: 仅 Linux（需要 io_uring）");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    imp::main()
}

#[cfg(target_os = "linux")]
mod imp {
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::Path;
    use std::time::Instant;

    use engramdb_core::layout::Layout;
    use engramdb_io::backend::{IoBackend, UringBatchBackend};

    const PAGE: u64 = 4096;

    struct Bijection {
        a: u128,
        b: u128,
        n: u128,
    }

    fn gcd(mut x: u128, mut y: u128) -> u128 {
        while y != 0 {
            let t = x % y;
            x = y;
            y = t;
        }
        x
    }

    impl Bijection {
        fn new(n: u64, seed: u64) -> Self {
            let n = n as u128;
            let mut s = seed;
            let mut next = move || {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };
            // 置低位强制为奇（`% 2 == 0` 的写法会触发 clippy::manual_is_multiple_of）
            let mut a = ((next() as u128) % n) | 1;
            while gcd(a, n) != 1 {
                a += 2;
            }
            let b = (next() as u128) % n;
            Self { a, b, n }
        }
        #[inline]
        fn key(&self, i: u64) -> u64 {
            ((self.a * i as u128 + self.b) % self.n) as u64
        }
    }

    fn drop_page_cache(dir: &Path, shards: u64) {
        use std::os::unix::io::AsRawFd;
        for i in 0..shards {
            let p = dir.join(format!("shard_{:03}.bin", i));
            if let Ok(f) = File::open(&p) {
                unsafe {
                    libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
                }
            }
        }
    }

    /// 把 keys 按 shard 分组 → 去重页 → 每 shard 一次 `read_many`。
    /// 返回 (读到的页数, 首个错误)。**绝不允许静默返回 0 页**：读失败必须现形。
    fn batched_read(
        dir: &Path,
        layout: &Layout,
        keys: &[u64],
        nthreads: usize,
        backend: &UringBatchBackend,
    ) -> (usize, Option<String>) {
        let rb = layout.row_bytes;
        let rps = layout.rows_per_shard;

        let mut by_shard: HashMap<u64, Vec<u64>> = HashMap::new();
        for &k in keys {
            let shard = k / rps;
            let local_row = k % rps;
            by_shard
                .entry(shard)
                .or_default()
                .push((local_row * rb) & !(PAGE - 1));
        }

        let mut tasks: Vec<(u64, Vec<u64>)> = by_shard.into_iter().collect();
        tasks.sort_unstable_by_key(|&(s, _)| s);
        let nt = nthreads.max(1).min(tasks.len());
        let chunk = tasks.len().div_ceil(nt);
        let mut total_pages = 0usize;
        let mut first_err: Option<String> = None;

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in tasks.chunks(chunk) {
                handles.push(s.spawn(move || {
                    let mut n = 0usize;
                    let mut err: Option<String> = None;
                    for (shard, mut pages) in t.iter().cloned() {
                        pages.sort_unstable();
                        pages.dedup();
                        let p = dir.join(format!("shard_{:03}.bin", shard));
                        let f = match File::open(&p) {
                            Ok(f) => f,
                            Err(e) => {
                                err.get_or_insert_with(|| format!("open {}: {e}", p.display()));
                                continue;
                            }
                        };
                        let fsize = f.metadata().map(|m| m.len()).unwrap_or(0);
                        let mut bufs: Vec<Vec<u8>> = pages
                            .iter()
                            .map(|&off| vec![0u8; (fsize.saturating_sub(off)).min(PAGE) as usize])
                            .collect();
                        let mut reqs: Vec<(u64, &mut [u8])> = pages
                            .iter()
                            .cloned()
                            .zip(bufs.iter_mut().map(|b| b.as_mut_slice()))
                            .collect();
                        match backend.read_many(&f, &mut reqs) {
                            Ok(()) => n += reqs.len(),
                            Err(e) => {
                                err.get_or_insert_with(|| {
                                    format!(
                                        "read_many on {} ({} pages): {e}",
                                        p.display(),
                                        reqs.len()
                                    )
                                });
                            }
                        }
                    }
                    (n, err)
                }));
            }
            for h in handles {
                if let Ok((v, e)) = h.join() {
                    total_pages += v;
                    if first_err.is_none() {
                        first_err = e;
                    }
                }
            }
        });
        (total_pages, first_err)
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.len() < 7 {
            eprintln!(
                "usage: uring_gate <dir> <shards> <rows_per_shard> <width> \
                 <rows_per_token> <tokens_per_round> <rounds> [threads_csv] [seed]"
            );
            std::process::exit(2);
        }
        let dir = Path::new(&args[0]);
        let shards: u64 = args[1].parse().unwrap();
        let rows_per_shard: u64 = args[2].parse().unwrap();
        let width: u64 = args[3].parse().unwrap();
        let rows_per_token: usize = args[4].parse().unwrap();
        let tokens: usize = args[5].parse().unwrap();
        let rounds: usize = args[6].parse().unwrap();
        let thread_list: Vec<usize> = args
            .get(7)
            .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or_else(|| vec![1]);
        let seed: u64 = args
            .get(8)
            .map(|s| {
                let t = s.trim();
                match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    Some(h) => u64::from_str_radix(h, 16).expect("bad hex seed"),
                    None => t.parse().expect("bad seed"),
                }
            })
            .unwrap_or(0x0D_1D_1D);

        let layout = Layout::new(shards, rows_per_shard, width, 1);
        let n = layout.total_rows();
        let stride = rounds + 1;
        let bij = Bijection::new(n, seed);
        let nkeys = tokens * rows_per_token;
        let backend = UringBatchBackend;

        println!(
            "== uring_gate  dir={}  shards={}  rows_per_shard={}  width={} ==",
            dir.display(),
            shards,
            rows_per_shard,
            width
        );
        println!(
            "   key space N={}  rows_per_token={}  tokens/round={}  rounds={}  seed={:#x}",
            n, rows_per_token, tokens, rounds, seed
        );
        println!(
            "   路径：keys → 按 shard 分组 → 去重 4 KiB 页 → 每 shard 一次 read_many（内部 256/次提交）"
        );
        println!("   冷读：每轮前 fadvise(DONTNEED)；结尾冷热自校验\n");

        let mut keys: Vec<u64> = Vec::with_capacity(nkeys);
        let mut medians: Vec<(usize, f64)> = Vec::new();
        let mut last_keys: Vec<u64> = Vec::new();
        let mut last_threads = 1usize;

        for &th in &thread_list {
            let mut v = Vec::new();
            for r in 0..rounds {
                drop_page_cache(dir, shards);
                keys.clear();
                let mut i = r as u64;
                for _ in 0..nkeys {
                    keys.push(bij.key(i));
                    i += stride as u64;
                }
                let t0 = Instant::now();
                let (pages, err) = batched_read(dir, &layout, &keys, th, &backend);
                let dt = t0.elapsed().as_secs_f64();
                if let Some(e) = &err {
                    println!("     读失败：{e}");
                }
                if pages == 0 {
                    println!(
                        "   >>> FATAL: 0 页被读取 —— `read_many` 在本机完全不可用（本次作废，退出码 4）。\n\
                         >>>        常见原因：容器 seccomp 拦截 io_uring_setup（EPERM）。"
                    );
                    std::process::exit(4);
                }
                let us_tok = dt * 1e6 / tokens as f64;
                println!(
                    "  threads={:<3} round {}: {:>9.2} us/token  {:>8.2} us/row  wall={:>9.1} ms  pages={}",
                    th,
                    r,
                    us_tok,
                    us_tok / rows_per_token as f64,
                    dt * 1e3,
                    pages
                );
                v.push(us_tok);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = v[v.len() / 2];
            println!(
                "  threads={:<3} MEDIAN over {} rounds: {:.2} us/token  ({:.2} us/row)\n",
                th,
                rounds,
                med,
                med / rows_per_token as f64
            );
            medians.push((th, med));
            last_keys = keys.clone();
            last_threads = th;
        }

        println!(
            "-- 机械自校验：重读上一轮（threads={}）的 keys（**不** drop 缓存） --",
            last_threads
        );
        let t0 = Instant::now();
        let (_wp, werr) = batched_read(dir, &layout, &last_keys, last_threads, &backend);
        if let Some(e) = &werr {
            println!("   warm 读失败：{e}");
        }
        let dt = t0.elapsed().as_secs_f64();
        let warm_us = dt * 1e6 / tokens as f64;
        let (_, cold_us) = *medians.last().unwrap();
        let ratio = cold_us / warm_us;
        println!(
            "   cold(median)={:.2} us/token  warm={:.2} us/token  ratio={:.2}x",
            cold_us, warm_us, ratio
        );
        if ratio >= 5.0 {
            println!("   >>> VALID: 冷热比 {:.2}x ≥ 5x。", ratio);
        } else {
            println!(
                "   >>> VOID: 冷热比仅 {:.2}x (<5x)，读数不冷，本次作废。",
                ratio
            );
            std::process::exit(3);
        }
    }
}
