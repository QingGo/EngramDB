//! view_gate — Store-P 折叠路径（`ViewReader::read_records_parallel`）的冷读门禁。
//!
//! 动机：§32 实测 Store-I 在 V4.1 几何下**每行占一个独立 4 KiB 页、零页共享**，
//! 48 行/token 就是 48 次散读。Store-P 把 token 的整个 e_t 物化成**一条紧凑记录**，
//! 于是每次调用只需要 **1 次读**。本门禁把这笔账在同一台机器、同一冷读口径下量出来，
//! 而不是引用 §30.6 在别的介质上的旧数字。
//!
//! 口径与 `nvme_gate` 完全一致：
//!   * 双射游走 `key(i)=(a·i+b) mod N`（gcd(a,N)=1）+ 轮间不相交切片 ⇒ 运行内绝不重读；
//!   * 每轮冷读前对视图文件 `fadvise(DONTNEED)`（容器里 `drop_caches` 被拒）；
//!   * **每个线程配置后**都做 warm 自校验，**双条件**裁决（比值 ≥5x 或 边际 ≥2 μs/token）。
//!
//! 用法：
//!   view_gate <view_file> <records_per_token> <tokens_per_round> <rounds> \
//!             [threads_csv] [seed]
//!
//! `threads_csv` 里可以写 `default` 表示用 `ViewReader::read_records`（它会取
//! `available_parallelism()`）——用来暴露"默认值在多核机器上取到 128"这个问题。

#[cfg(not(unix))]
fn main() {
    eprintln!("view_gate: 仅 unix");
    std::process::exit(1);
}

#[cfg(unix)]
fn main() {
    imp::main()
}

#[cfg(unix)]
mod imp {
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use engramdb_io::view::ViewReader;

    /// 冷热比判据（IO 主导区间）。
    const MIN_RATIO: f64 = 5.0;
    /// 边际成本判据，μs/token（开销主导区间）。理由同 `nvme_gate`。
    const MIN_MARGINAL: f64 = 2.0;

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
            let mut a = ((next() as u128) % n) | 1;
            while gcd(a, n) != 1 {
                a += 2;
            }
            let b = (next() as u128) % n;
            Self { a, b, n }
        }
        #[inline]
        fn idx(&self, i: u64) -> usize {
            ((self.a * i as u128 + self.b) % self.n) as usize
        }
    }

    fn drop_page_cache(view: &Path) {
        use std::os::unix::io::AsRawFd;
        if let Ok(f) = File::open(view) {
            unsafe {
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.len() < 4 {
            eprintln!(
                "usage: view_gate <view_file> <records_per_token> <tokens_per_round> <rounds> \
                 [threads_csv] [seed]   (threads_csv 里可用 'default')"
            );
            std::process::exit(2);
        }
        let view_path = PathBuf::from(&args[0]);
        let rpt: usize = args[1].parse().unwrap();
        let tokens: usize = args[2].parse().unwrap();
        let rounds: usize = args[3].parse().unwrap();
        let thread_specs: Vec<String> = args
            .get(4)
            .map(|s| s.split(',').map(|x| x.to_string()).collect())
            .unwrap_or_else(|| vec!["1".into(), "8".into(), "32".into()]);
        let seed: u64 = args
            .get(5)
            .map(|s| {
                let t = s.trim();
                match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    Some(h) => u64::from_str_radix(h, 16).expect("bad hex seed"),
                    None => t.parse().expect("bad seed"),
                }
            })
            .unwrap_or(0x7E_11);

        let vr = ViewReader::open(&view_path).expect("open view");
        let n = vr.len();
        let slot = vr.slot_bytes();
        let nkeys = tokens * rpt;
        let stride = rounds + 1;
        let bij = Bijection::new(n as u64, seed);

        println!(
            "== view_gate  view={}  slot_bytes={}  records={} ({:.1} GiB) ==",
            view_path.display(),
            slot,
            n,
            n as f64 * slot as f64 / 1073741824.0
        );
        println!(
            "   records_per_token={}  tokens/round={}  rounds={}  keys/round={}  seed={:#x}",
            rpt, tokens, rounds, nkeys, seed
        );
        println!(
            "   每次调用读取 {} 条紧凑记录 × {} B = {:.1} KiB（Store-I 的等价物是 {} 次散读）",
            nkeys,
            slot,
            nkeys as f64 * slot as f64 / 1024.0,
            nkeys * 48
        );
        println!("   冷读：每轮前 fadvise(DONTNEED)；每个配置后做冷热自校验\n");

        let mut out = vec![0u8; nkeys * slot as usize];
        let mut idxs: Vec<usize> = Vec::with_capacity(nkeys);
        let mut any_void = false;

        for spec in &thread_specs {
            let use_default = spec == "default";
            let th: usize = if use_default {
                0
            } else {
                spec.parse().expect("bad threads")
            };
            let label = if use_default {
                "default".to_string()
            } else {
                th.to_string()
            };

            let mut meds = Vec::with_capacity(rounds);
            for r in 0..rounds {
                drop_page_cache(&view_path);
                idxs.clear();
                let mut i = r as u64;
                for _ in 0..nkeys {
                    idxs.push(bij.idx(i));
                    i += stride as u64;
                }
                let t0 = Instant::now();
                if use_default {
                    vr.read_records(&idxs, &mut out).expect("read_records");
                } else {
                    vr.read_records_parallel(&idxs, &mut out, th)
                        .expect("read_records_parallel");
                }
                let dt = t0.elapsed().as_secs_f64();
                let us_tok = dt * 1e6 / tokens as f64;
                println!(
                    "  threads={:<8} round {}: {:>9.2} us/token  wall={:>8.2} ms  {:.0} MB/s",
                    label,
                    r,
                    us_tok,
                    dt * 1e3,
                    nkeys as f64 * slot as f64 / 1e6 / dt
                );
                meds.push(us_tok);
            }
            meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = meds[meds.len() / 2];

            // ---- 自校验：立刻重读同一批（不 drop）----
            let t0 = Instant::now();
            if use_default {
                vr.read_records(&idxs, &mut out).expect("warm read_records");
            } else {
                vr.read_records_parallel(&idxs, &mut out, th)
                    .expect("warm read_records_parallel");
            }
            let dt = t0.elapsed().as_secs_f64();
            let warm = dt * 1e6 / tokens as f64;
            let ratio = med / warm;
            let marginal = (med - warm).max(0.0) / rpt as f64;

            println!(
                "  threads={:<8} MEDIAN cold: {:>9.2} us/token | warm: {:>9.2} us/token | \
                 ratio {:>6.2}x | marginal {:>8.2} us/token",
                label, med, warm, ratio, marginal
            );
            if ratio >= MIN_RATIO || marginal >= MIN_MARGINAL {
                let why = if ratio >= MIN_RATIO {
                    format!("比值 {ratio:.2}x")
                } else {
                    format!("边际 {marginal:.2} us/token")
                };
                println!("  threads={:<8} >>> VALID（{why}）\n", label);
            } else {
                println!(
                    "  threads={:<8} >>> VOID：比值 {ratio:.2}x 且边际 {marginal:.2} —— 不冷\n",
                    label
                );
                any_void = true;
            }
        }

        if any_void {
            println!("   >>> 有配置被判 VOID：只引用 VALID 的配置。");
            std::process::exit(3);
        }
        println!("   >>> 全部配置 VALID。");
    }
}
