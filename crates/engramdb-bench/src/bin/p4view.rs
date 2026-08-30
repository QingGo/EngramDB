//! P4 探针（wrapper）：视图构建/基准/延迟原语已提升到 engramdb-io（`io::view`），
//! 本 bin 保留命令行兼容（probes 复现命令与 gate.sh 依赖不变），执行即调用库原语。
//!
//! 复现（P4 判定口径）：`p4view build data/real-rows <N> <view.bin> <keys.txt>`
//! `p4view bench data/real-rows <view.bin> <keys.txt>` / `p4view lat <view.bin> [--warm|--cold]`
//! 产品面（同构）：`engramdb view build|bench|lat`。

use std::path::PathBuf;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;
use engramdb_io::view;

fn lay() -> Layout {
    Layout::new(128, 2_500_012, 160, 1)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        println!("usage: p4view <build|bench|lat> <...> [--slot 2560|4096]");
        return;
    };
    let out = match cmd.as_str() {
        "build" => cmd_build(args),
        "bench" => cmd_bench(args),
        "lat" => cmd_lat(args),
        _ => Err(format!("unknown {cmd}")),
    };
    if let Err(e) = out {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn cmd_build(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let layout = lay();
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let n: usize = rest
        .next()
        .ok_or("n_grams")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let view_out = PathBuf::from(rest.next().ok_or("view.bin")?);
    let keys_out = PathBuf::from(rest.next().ok_or("keys.txt")?);
    let mut slot_bytes: u64 = 4096;
    let mut it = rest;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--slot" => {
                slot_bytes = it
                    .next()
                    .ok_or("slot")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--seed" => {
                let _ = it.next();
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    let batch = BadgeGather::open(&rows_dir, &layout).map_err(|e| e.to_string())?;
    let _ = view::build_view(&batch, n, slot_bytes, &view_out, Some(&keys_out))
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_bench(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let layout = lay();
    let rows_dir = PathBuf::from(rest.next().ok_or("rows_dir")?);
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let mut keys_path: Option<PathBuf> = None;
    let mut sub_grams = 0usize;
    let mut threads = 8usize;
    let mut slot_bytes: u64 = 0;
    let mut it = rest;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--keys" => keys_path = Some(PathBuf::from(it.next().ok_or("keys 路径")?)),
            "--sub" => {
                sub_grams = it
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--threads" => {
                threads = it
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--slot" => {
                slot_bytes = it
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    let batch = BadgeGather::open(&rows_dir, &layout).map_err(|e| e.to_string())?;
    let keys = match &keys_path {
        Some(kf) => Some(view::read_keys(kf).map_err(|e| e.to_string())?),
        None => None,
    };
    view::bench_view(
        &batch,
        &view_file,
        keys.as_deref(),
        sub_grams,
        threads,
        slot_bytes,
        "rand",
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_lat(mut rest: impl Iterator<Item = String>) -> Result<(), String> {
    let view_file = PathBuf::from(rest.next().ok_or("view.bin")?);
    let mut threads = 8usize;
    let mut warm = false;
    let mut cold = false;
    let mut sub_grams = 0usize;
    let mut slot_bytes: u64 = 0;
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--threads" => {
                threads = rest
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--warm" => warm = true,
            "--cold" => cold = true,
            "--sub" => {
                sub_grams = rest
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            "--slot" => {
                slot_bytes = rest
                    .next()
                    .ok_or("v")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?
            }
            _ => return Err(format!("未知参数 {a}")),
        }
    }
    view::lat_view(&view_file, threads, warm, cold, sub_grams, slot_bytes)
        .map_err(|e| e.to_string())
}
