//! P3：训练流读取模拟 v2 —— 页级 LRU 缓存模型（贴近 OS 页缓存 / badge 页行为）。
//!
//! 每 token 16 行（160B/行, 0.78 页/行 由行密度决定——这里按真实分布逐行映射页）。
//! 页 miss → 4096B 读 + LRU 入缓存（FIFO 近似）。
//! 模式：
//!   uniform : 行完全随机（chat/评估负载）
//!   local   : 文档窗口模型（窗口 20K token 内 80% 行来自 2 万文档行集，20% 随机；跨文档漂移）
//!
//! 用法: --mode local|uniform --tokens 1000000 --seg N --cache-mb M | --grid

use std::collections::{HashMap, HashSet, VecDeque};

const ROWS_PER_TOKEN: usize = 16;
const ROW_BYTES: u64 = 160;
const PAGE: u64 = 4096;
const TABLE_ROWS: u64 = 320_001_536;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut mode = String::from("local");
    let mut tokens = 1_000_000usize;
    let mut cache_mb = 512u64;
    let mut grid = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--mode" => mode = args.next().unwrap(),
            "--tokens" => tokens = parse_usize(args.next().unwrap()),
            "--cache-mb" => cache_mb = parse_usize(args.next().unwrap()) as u64,
            "--grid" => grid = true,
            _ => {}
        }
    }

    if grid {
        println!("mode\tcacheMB\thit%\treadKB/tok\tGBps@10ktok/s");
        for &c in &[32u64, 128, 512, 2048] {
            let r = simulate(&mode, tokens, c);
            println!(
                "{}\t{}\t{:.2}\t{:.2}\t{:.2}",
                mode,
                c,
                r.hit * 100.0,
                r.read_bytes as f64 / tokens as f64 / 1024.0,
                (r.read_bytes as f64 / tokens as f64) * 10_000.0 / 1e9
            );
        }
        return;
    }

    let r = simulate(&mode, tokens, cache_mb);
    println!(
        "mode={mode} tokens={tokens} cache={cache_mb}MB -> page-hit={:.2}% readKB/tok={:.2}",
        r.hit * 100.0,
        r.read_bytes as f64 / tokens as f64 / 1024.0
    );
}

fn parse_usize(s: String) -> usize {
    s.replace('_', "").parse::<usize>().unwrap_or(0)
}

struct SimResult {
    hit: f64,
    read_bytes: u64,
}

fn simulate(mode: &str, tokens: usize, cache_mb: u64) -> SimResult {
    let cache_pages = ((cache_mb * 1024 * 1024) / PAGE).max(1) as usize;
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut rng = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let doc_lines = 20_000u64;
    let mut doc_start: u64 = 0;
    let mut cache: HashMap<u64, bool> = HashMap::new(); // page -> resident
    let mut lru: VecDeque<u64> = VecDeque::new(); // FIFO 近似 LRU
    let mut seen: HashSet<u64> = HashSet::new(); // 区分第一次看到（必 miss）
    let mut hits: u64 = 0;
    let mut read_bytes: u64 = 0;

    for t in 0..tokens {
        if mode == "local" && t % doc_lines as usize == 0 {
            doc_start = rng() % (TABLE_ROWS - doc_lines);
        }
        for _ in 0..ROWS_PER_TOKEN {
            let rowid = if mode == "local" {
                if rng() % 10 < 8 {
                    doc_start + rng() % doc_lines
                } else {
                    rng() % TABLE_ROWS
                }
            } else {
                rng() % TABLE_ROWS
            };
            let page = (rowid * ROW_BYTES) / PAGE;
            let first_time = seen.insert(page);
            let resident = matches!(cache.get(&page), Some(&true));
            if first_time {
                // 首次必 miss（读取）
                read_bytes += PAGE;
            } else if resident {
                hits += 1;
                continue;
            } else {
                // 出现过但已驱逐 → miss
                read_bytes += PAGE;
            }
            // 入缓存
            cache.insert(page, true);
            lru.push_back(page);
            while lru.len() > cache_pages {
                if let Some(ev) = lru.pop_front() {
                    cache.insert(ev, false);
                }
            }
        }
    }

    SimResult {
        hit: hits as f64 / (tokens * ROWS_PER_TOKEN) as f64,
        read_bytes,
    }
}
