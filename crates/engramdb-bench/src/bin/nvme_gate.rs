//! nvme_gate — EngramDB 真实读路径（`BadgeGather::gather_pp`）在真实块设备上的冷读门禁。
//!
//! 口径（见 `docs/roadmap.md` §30.5 纪律 #5、§30.7）：
//!
//!   * 每一轮从一个**双射游走**上取彼此**不相交**的索引切片，因此一次运行内
//!     **没有任何字节范围被读第二次** —— 冷读来自"从未读过"，而不是来自"表比内存大"。
//!     （这一点在 1 TiB RAM 的机器上尤其重要：按体积论证冷读在这里根本不成立。）
//!   * 结尾做**机械自校验**：取最新一轮的 keys 立刻重读一遍。若 `warm` 没有比
//!     `cold` 快很多，说明本案的冷读口径是**坏的**，本次运行作废（VOID）。
//!   * 打印工作集大小与实际不同的 4 KiB 页数，不假设 page cache 状态。
//!
//! 用法：
//!   nvme_gate <dir> <shards> <rows_per_shard> <width> \
//!             <rows_per_token> <tokens_per_round> <rounds> <threads,threads,...>

use std::path::Path;
use std::time::Instant;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;

/// 冷热自校验的**比值**判据 —— 适用于 IO 主导区间（大 batch）。
const MIN_RATIO: f64 = 5.0;

/// 冷热自校验的**边际成本**判据，单位 μs/行 —— 适用于**开销主导区间**（小 batch）。
///
/// 小 batch 下每次调用的固定开销（线程创建、分组、排序、回填）在 cold 与 warm
/// 读数里**都要付**，所以比值必然趋近 1，比值判据在这里不适用。
/// 此时改用"冷读比热读多花了多少"来判断：真正的冷读必须有可观的边际成本。
///
/// 下界来自裸设备 C 探针（`scripts/nvme_raw_probe.c`）：QD32 时 4 KiB 随机读
/// **2.74 μs/页**，QD1 时 77.3 μs/页。取 2.0 μs/行 作保守下界——低于它就说明
/// 读到的是缓存而不是介质。
const MIN_MARGINAL: f64 = 2.0;

/// 在冷读轮之前把该文件的干净页从 page cache 里丢掉。
///
/// 必要性：本机 cgroup 内存上限 120 GiB，而表只有 26–51 GB，
/// **"表比内存大" 在这里根本不成立** —— 一份被读过的数据会永久驻留缓存。
/// `drop_caches` 在容器内被拒（无 CAP_SYS_ADMIN），所以只能按文件 `fadvise`。
fn drop_page_cache(dir: &Path, shards: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        for i in 0..shards {
            let p = dir.join(format!("shard_{:03}.bin", i));
            if let Ok(f) = std::fs::File::open(&p) {
                // 返回值忽略：advisory，失败不致命（自校验会兜住）
                unsafe {
                    libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
                }
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (dir, shards);
    }
}

/// 双射游走：key(i) = (a*i + b) mod N，其中 gcd(a, N) == 1。
/// 保证 i 不同 ⇒ key 不同，所以"每轮用不相交的 i 切片"⇔"每轮用不相交的 keys"。
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
        // splitmix64 派生初始 a/b
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

/// 这一批 keys 实际会打到多少个不同的 (shard, 4 KiB page)。
fn distinct_pages(keys: &[u64], layout: &Layout) -> usize {
    let rb = layout.row_bytes;
    let mut set = std::collections::HashSet::with_capacity(keys.len());
    for &k in keys {
        let shard = k / layout.rows_per_shard;
        let local_row = k % layout.rows_per_shard;
        set.insert((shard, (local_row * rb) & !4095u64));
    }
    set.len()
}

struct RoundResult {
    us_per_token: f64,
    us_per_row: f64,
    wall_ms: f64,
    pages: usize,
}

/// 一轮的参数（打包成一个结构体：散开的 10 个参数会触发 clippy::too_many_arguments）。
struct RoundSpec {
    round: usize,
    stride: usize,
    rows_per_token: usize,
    tokens: usize,
    threads: usize,
}

fn run_round(
    batch: &BadgeGather<'_>,
    layout: &Layout,
    bij: &Bijection,
    spec: &RoundSpec,
    out: &mut [u8],
    keys: &mut Vec<u64>,
) -> RoundResult {
    let RoundSpec {
        round,
        stride,
        rows_per_token,
        tokens,
        threads,
    } = *spec;
    let nkeys = tokens * rows_per_token;
    keys.clear();
    // 第 round 轮取 i = round, round+stride, round+2*stride, ...
    let mut i = round as u64;
    for _ in 0..nkeys {
        keys.push(bij.key(i));
        i += stride as u64;
    }
    let pages = distinct_pages(keys, layout);

    let t0 = Instant::now();
    batch
        .gather_pp(keys, out, threads)
        .expect("gather_pp failed");
    let dt = t0.elapsed().as_secs_f64();

    RoundResult {
        us_per_token: dt * 1e6 / tokens as f64,
        us_per_row: dt * 1e6 / nkeys as f64,
        wall_ms: dt * 1e3,
        pages,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 7 {
        eprintln!(
            "usage: nvme_gate <dir> <shards> <rows_per_shard> <width> \
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
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);
    let seed: u64 = args
        .get(8)
        .map(|s| {
            let t = s.trim();
            match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                Some(h) => u64::from_str_radix(h, 16).expect("bad hex seed"),
                None => t.parse().expect("bad seed"),
            }
        })
        .unwrap_or(0xE6_9A_11);
    let drop_mode: String = args.get(9).cloned().unwrap_or_else(|| "none".to_string());

    let layout = Layout::new(shards, rows_per_shard, width, 1);
    let n = layout.total_rows();
    let batch = BadgeGather::open(dir, &layout).expect("open store dir");

    // stride：轮与轮之间留出槽位，保证索引切片互不相交
    let stride = rounds + 1;
    let bij = Bijection::new(n, seed);

    let nkeys = tokens * rows_per_token;
    println!(
        "== nvme_gate  dir={}  shards={}  rows_per_shard={}  width={} (row_bytes={}) ==",
        dir.display(),
        shards,
        rows_per_shard,
        width,
        layout.row_bytes
    );
    println!(
        "   key space N={}  rows_per_token={}  tokens/round={}  rounds={}  stride={}  seed={:#x}",
        n, rows_per_token, tokens, rounds, stride, seed
    );
    println!(
        "   per-round rows={} ({:.2} MiB payload);  total distinct indices consumed={} of {}",
        nkeys,
        nkeys as f64 * layout.row_bytes as f64 / 1048576.0,
        nkeys * stride,
        n
    );
    println!(
        "   冷读来源：双射 + 轮间不相交切片 ⇒ 本次运行内**从不重读**任何 key。\n\
         但**这不足以**保证冷读：若数据此前已被别人读过，它会一直留在 page cache 里\n\
         （本机 cgroup 上限 120 GiB ≫ 表体积，\"表比内存大\" 不成立，drop_caches 又被拒）。\n\
         故 drop_mode={} 决定是否在每轮冷读前按文件 fadvise(DONTNEED)。",
        drop_mode
    );
    // 自描述：每次运行报告**实际走的 IO 路径**。做 A/B 时若这一行没变，
    // 就说明两组读数差别不来自该机制 —— 这是本轮踩过的坑（roadmap §33.2）。
    println!(
        "   io path: pool_workers={} pool_disabled={}   (env: ENGRAMDB_NO_POOL / ENGRAMDB_POOL_WORKERS)",
        engramdb_io::pool::workers(),
        engramdb_io::pool::disabled()
    );
    println!();

    let mut out = vec![0u8; nkeys * width as usize];
    let mut keys: Vec<u64> = Vec::with_capacity(nkeys);
    let mut last_keys: Vec<u64> = Vec::new();
    let mut last_threads = 8usize;
    let mut any_void = false;

    println!(
        "-- 每个线程配置后都做一次冷热自校验。判据是**双条件**：\n\
         \x20  比值 ≥ {MIN_RATIO}x（IO 主导区间），**或** 边际成本 ≥ {MIN_MARGINAL} μs/行\n\
         \x20  （开销主导区间——小 batch 下每次调用的固定开销 cold/warm 都要付，\n\
         \x20   比值必然趋近 1，此时比值判据不适用）。\n\
         \x20  {MIN_MARGINAL} μs/行 的下界来自裸设备 C 探针：QD32 时每 4 KiB 页 2.74 μs。\n"
    );

    for &th in &thread_list {
        let mut res = Vec::with_capacity(rounds);
        for r in 0..rounds {
            if drop_mode == "fadvise" {
                drop_page_cache(dir, shards);
            }
            let rr = run_round(
                &batch,
                &layout,
                &bij,
                &RoundSpec {
                    round: r,
                    stride,
                    rows_per_token,
                    tokens,
                    threads: th,
                },
                &mut out,
                &mut keys,
            );
            println!(
                "  threads={:<3} round {}: {:>9.2} us/token  {:>8.2} us/row  wall={:>9.2} ms  distinct_4KiB_pages={}",
                th, r, rr.us_per_token, rr.us_per_row, rr.wall_ms, rr.pages
            );
            res.push(rr);
        }
        let mut v: Vec<f64> = res.iter().map(|r| r.us_per_token).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];

        // ---- 本配置的机械自校验：立刻重读同一批 keys（不 drop 缓存）----
        let t0 = Instant::now();
        batch
            .gather_pp(&keys, &mut out, th)
            .expect("warm gather_pp failed");
        let dt = t0.elapsed().as_secs_f64();
        let warm_us = dt * 1e6 / tokens as f64;
        let ratio = med / warm_us;
        // med 与 warm_us 都是 **μs/token**，所以每个 token 的行数就是 rows_per_token。
        // （曾经误除以 tokens*rows_per_token，差了一个 tokens 因子，制造过假 VOID。）
        let marginal_us_per_row = (med - warm_us).max(0.0) / rows_per_token as f64;

        println!(
            "  threads={:<3} MEDIAN cold: {:>9.2} us/token ({:>7.2} us/row) | \
             warm(overhead): {:>9.2} us/token ({:>7.2} us/row) | ratio {:>5.2}x | marginal {:>6.2} us/row",
            th,
            med,
            med / rows_per_token as f64,
            warm_us,
            warm_us / rows_per_token as f64,
            ratio,
            marginal_us_per_row
        );
        let ok = ratio >= MIN_RATIO || marginal_us_per_row >= MIN_MARGINAL;
        if ok {
            let why = if ratio >= MIN_RATIO {
                format!("比值 {ratio:.2}x ≥ {MIN_RATIO}x")
            } else {
                format!("边际 {marginal_us_per_row:.2} μs/行 ≥ {MIN_MARGINAL}")
            };
            println!("  threads={:<3} >>> VALID（{why}）\n", th);
        } else {
            println!(
                "  threads={:<3} >>> VOID：比值 {ratio:.2}x < {MIN_RATIO}x 且边际 \
                 {marginal_us_per_row:.2} μs/行 < {MIN_MARGINAL} —— 读数不冷，本配置作废\n",
                th
            );
            any_void = true;
        }
        last_keys = keys.clone();
        last_threads = th;
    }

    // ---- 全局收尾：最后一次自校验（复用上一轮的 keys）----
    println!(
        "-- 全局收尾自校验（round {}，threads={}） --",
        rounds - 1,
        last_threads
    );
    let t0 = Instant::now();
    batch
        .gather_pp(&last_keys, &mut out, last_threads)
        .expect("warm gather_pp failed");
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "   SELFCHECK-warm: {:.2} us/token  wall={:.2} ms",
        dt * 1e6 / tokens as f64,
        dt * 1e3
    );
    if any_void {
        println!("   >>> 至少一个线程配置被判 VOID：**只引用被判 VALID 的配置**，其余不得引用。");
        std::process::exit(3);
    }
    println!("   >>> 全部线程配置 VALID。");
}
