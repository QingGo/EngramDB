//! Minimal Rust multi-table / manifest / TCP serve prototype.
//!
//! This is the first Rust-side convergence for the Python service prototype:
//! a single process can enumerate tables under a root directory, read each
//! table's manifest, and serve simple JSON store fetches over TCP.
//!
//! The wire is intentionally minimal (newline-delimited JSON, like the Python
//! JSON server). Binary/Arrow IPC can be layered on later.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;
use engramdb_io::view::ViewReader;
use serde_json::Value;

const KIND_JSON: u8 = 0;
const KIND_RAW: u8 = 1;

/// Return table ids under `root` that contain either a manifest or shard data.
pub fn list_tables(root: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let rd = std::fs::read_dir(root).map_err(|e| e.to_string())?;
    for entry in rd {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if table_is_data_dir(&path) {
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                out.push(name.to_string());
            }
        }
    }
    out.sort();
    Ok(out)
}

fn table_is_data_dir(dir: &Path) -> bool {
    if dir.join("manifest.json").exists() {
        return true;
    }
    if let Ok(mut rd) = std::fs::read_dir(dir) {
        while let Some(Ok(entry)) = rd.next() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("shard_") && name.ends_with(".bin") {
                return true;
            }
            if name.starts_with("badge_") && name.ends_with(".bin") {
                return true;
            }
        }
    }
    false
}

fn load_manifest(dir: &Path) -> Option<Value> {
    let p = dir.join("manifest.json");
    let data = std::fs::read(p).ok()?;
    serde_json::from_slice(&data).ok()
}

pub fn layout_from_manifest(manifest: &Value) -> Option<Layout> {
    let l = manifest.get("layout")?;
    let shards = l.get("shards")?.as_u64()?;
    let rows_per_shard = l.get("rows_per_shard")?.as_u64()?;
    let width = l.get("width")?.as_u64()?;
    let elem_bytes = l.get("elem_bytes").and_then(Value::as_u64).unwrap_or(1);
    Some(Layout::new(shards, rows_per_shard, width, elem_bytes))
}

/// Validate one table directory against its manifest.
///
/// Checks that the manifest parses, every expected shard file exists, and each
/// shard file has a size that is a non-zero multiple of the row width.  This is
/// a first integrity gate; checksums and exact row counts can be layered on.
pub fn check_table(dir: &Path) -> Result<Value, String> {
    let manifest = load_manifest(dir).ok_or("missing or invalid manifest.json")?;
    let layout = layout_from_manifest(&manifest).ok_or("manifest has no valid layout")?;
    let mut issues: Vec<String> = Vec::new();
    let mut shards_found = 0u64;
    for i in 0..layout.shards {
        let shard = dir.join(format!("shard_{:03}.bin", i));
        let badge = dir.join(format!("badge_{:03}.bin", i));
        let path = if shard.exists() {
            shard
        } else if badge.exists() {
            badge
        } else {
            issues.push(format!("missing shard file {}", i));
            continue;
        };
        shards_found += 1;
        let len = std::fs::metadata(&path).map_err(|e| e.to_string())?.len();
        if len == 0 {
            issues.push(format!("shard {} is empty", i));
        } else if len % layout.row_bytes != 0 {
            issues.push(format!(
                "shard {} size {} is not a multiple of row_bytes {}",
                i, len, layout.row_bytes
            ));
        }
    }
    let ok = shards_found == layout.shards && issues.is_empty();
    Ok(serde_json::json!({
        "ok": ok,
        "shards_found": shards_found,
        "shards_expected": layout.shards,
        "row_bytes": layout.row_bytes,
        "issues": issues,
    }))
}

/// Check every table under a multi-table root and return a summary value.
pub fn check_root(root: &Path) -> Result<Value, String> {
    let tables = list_tables(root)?;
    let mut results = Vec::new();
    let mut total_ok = true;
    for table in &tables {
        let dir = root.join(table);
        match check_table(&dir) {
            Ok(mut v) => {
                if v["ok"] != true {
                    total_ok = false;
                }
                v["table"] = serde_json::json!(table);
                results.push(v);
            }
            Err(e) => {
                total_ok = false;
                results.push(serde_json::json!({
                    "table": table,
                    "ok": false,
                    "error": e,
                }));
            }
        }
    }
    Ok(serde_json::json!({
        "ok": total_ok,
        "table_count": tables.len(),
        "tables": results,
    }))
}

fn table_dir(root: &Path, table: &str) -> PathBuf {
    root.join(table)
}

fn resolve_layout(dir: &Path, req: &Value) -> Result<Layout, String> {
    if let Some(m) = load_manifest(dir) {
        return layout_from_manifest(&m).ok_or_else(|| "manifest missing layout?".to_string());
    }
    let shards = req
        .get("shards")
        .and_then(Value::as_u64)
        .ok_or("no manifest; fetch requires shards")?;
    let rows_per_shard = req
        .get("rows_per_shard")
        .and_then(Value::as_u64)
        .ok_or("no manifest; fetch requires rows_per_shard")?;
    let width = req
        .get("width")
        .and_then(Value::as_u64)
        .ok_or("no manifest; fetch requires width")?;
    let elem_bytes = req.get("elem_bytes").and_then(Value::as_u64).unwrap_or(1);
    Ok(Layout::new(shards, rows_per_shard, width, elem_bytes))
}

fn fetch_raw_table(root: &Path, req: &Value) -> Result<(Vec<u8>, u64), String> {
    let table = req
        .get("table")
        .and_then(Value::as_str)
        .ok_or("fetch requires table")?;
    let dir = table_dir(root, table);
    if !dir.is_dir() {
        return Err(format!("table not found: {table}"));
    }
    let layout = resolve_layout(&dir, req)?;
    let rowids: Vec<u64> = req
        .get("rowids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .ok_or("fetch requires rowids array")?;

    let batch = BadgeGather::open(&dir, &layout).map_err(|e| e.to_string())?;
    let w = layout.width as usize;
    let mut out = vec![0u8; rowids.len() * w];
    batch
        .gather_pp(&rowids, &mut out, 8)
        .map_err(|e| e.to_string())?;
    Ok((out, layout.width))
}

fn read_view_record(path: &Path, index: usize) -> Result<Vec<u8>, String> {
    let reader = ViewReader::open(path).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.slot_bytes() as usize];
    let n = reader
        .read_record(index, &mut buf)
        .map_err(|e| e.to_string())?;
    buf.truncate(n);
    Ok(buf)
}

fn fetch_table(root: &Path, req: &Value) -> Result<Value, String> {
    let (raw, width) = fetch_raw_table(root, req)?;
    let table = req
        .get("table")
        .and_then(Value::as_str)
        .ok_or("fetch requires table")?;
    let rowids: Vec<u64> = req
        .get("rowids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .ok_or("fetch requires rowids array")?;
    Ok(serde_json::json!({
        "ok": true,
        "table": table,
        "num_rows": rowids.len(),
        "width": width,
        "raw_base64": base64_encode(&raw),
    }))
}

fn handle_request(root: &Path, req: &Value) -> Value {
    let cmd = req.get("cmd").and_then(Value::as_str).unwrap_or("");
    match cmd {
        "ping" => serde_json::json!({"ok": true, "pong": true}),
        "list_tables" => match list_tables(root) {
            Ok(tables) => serde_json::json!({"ok": true, "tables": tables}),
            Err(e) => serde_json::json!({"ok": false, "error": e}),
        },
        "fetch" => match fetch_table(root, req) {
            Ok(v) => v,
            Err(e) => serde_json::json!({"ok": false, "error": e}),
        },
        "view_read" => match req.get("path").and_then(Value::as_str) {
            Some(path) => {
                let index = req.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                match read_view_record(Path::new(path), index) {
                    Ok(data) => serde_json::json!({
                        "ok": true,
                        "index": index,
                        "slot_base64": base64_encode(&data),
                    }),
                    Err(e) => serde_json::json!({"ok": false, "error": e}),
                }
            }
            None => serde_json::json!({"ok": false, "error": "view_read requires path"}),
        },
        other => serde_json::json!({"ok": false, "error": format!("unknown command: {other}")}),
    }
}

fn binary_dispatch(root: &Path, req: &Value) -> (u8, Vec<u8>) {
    let cmd = req.get("cmd").and_then(Value::as_str).unwrap_or("");
    match cmd {
        "ping" => (
            KIND_JSON,
            serde_json::to_vec(&serde_json::json!({"ok": true, "pong": true})).unwrap_or_default(),
        ),
        "list_tables" => match list_tables(root) {
            Ok(tables) => (
                KIND_JSON,
                serde_json::to_vec(&serde_json::json!({"ok": true, "tables": tables}))
                    .unwrap_or_default(),
            ),
            Err(e) => (
                KIND_JSON,
                serde_json::to_vec(&serde_json::json!({"ok": false, "error": e}))
                    .unwrap_or_default(),
            ),
        },
        "fetch_raw" => match fetch_raw_table(root, req) {
            Ok((raw, _width)) => (KIND_RAW, raw),
            Err(e) => (
                KIND_JSON,
                serde_json::to_vec(&serde_json::json!({"ok": false, "error": e}))
                    .unwrap_or_default(),
            ),
        },
        "view_read" => match req.get("path").and_then(Value::as_str) {
            Some(path) => {
                let index = req.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                match read_view_record(Path::new(path), index) {
                    Ok(data) => (KIND_RAW, data),
                    Err(e) => (
                        KIND_JSON,
                        serde_json::to_vec(&serde_json::json!({"ok": false, "error": e}))
                            .unwrap_or_default(),
                    ),
                }
            }
            None => (
                KIND_JSON,
                serde_json::to_vec(&serde_json::json!({
                    "ok": false,
                    "error": "view_read requires path"
                }))
                .unwrap_or_default(),
            ),
        },
        other => (
            KIND_JSON,
            serde_json::to_vec(&serde_json::json!({
                "ok": false,
                "error": format!("unknown command: {other}")
            }))
            .unwrap_or_default(),
        ),
    }
}

fn read_exact_stream(stream: &mut TcpStream, buf: &mut [u8]) -> Result<usize, String> {
    let mut off = 0usize;
    while off < buf.len() {
        let n = stream.read(&mut buf[off..]).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        off += n;
    }
    Ok(off)
}

fn handle_binary_connection(root: PathBuf, mut stream: TcpStream) -> Result<(), String> {
    loop {
        let mut header = [0u8; 4];
        let n = read_exact_stream(&mut stream, &mut header)?;
        if n < 4 {
            return Ok(());
        }
        let body_len = u32::from_be_bytes(header) as usize;
        let mut body = vec![0u8; body_len];
        let got = read_exact_stream(&mut stream, &mut body)?;
        if got < body_len {
            return Err("truncated binary request".into());
        }
        let (kind, payload) = match serde_json::from_slice::<Value>(&body) {
            Ok(req) => binary_dispatch(&root, &req),
            Err(e) => (
                KIND_JSON,
                serde_json::to_vec(&serde_json::json!({
                    "ok": false,
                    "error": format!("bad json: {e}")
                }))
                .unwrap_or_default(),
            ),
        };
        let frame_len = 1u32 + payload.len() as u32;
        stream
            .write_all(&frame_len.to_be_bytes())
            .map_err(|e| e.to_string())?;
        stream.write_all(&[kind]).map_err(|e| e.to_string())?;
        stream.write_all(&payload).map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())?;
    }
}

pub fn run_binary(root: &Path, host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| e.to_string())?;
    println!(
        "engramdb binary serve listening on {}:{} (root {:?})",
        host, port, root
    );
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let root = root.to_path_buf();
                std::thread::spawn(move || {
                    let _ = handle_binary_connection(root, stream);
                });
            }
            Err(e) => eprintln!("serve accept error: {e}"),
        }
    }
    Ok(())
}

fn handle_connection(root: PathBuf, mut stream: TcpStream) -> Result<(), String> {
    let reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Value>(line) {
            Ok(req) => handle_request(&root, &req),
            Err(e) => serde_json::json!({"ok": false, "error": format!("bad json: {e}")}),
        };
        let mut data = serde_json::to_vec(&resp).map_err(|e| e.to_string())?;
        data.push(b'\n');
        stream.write_all(&data).map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn run(root: &Path, host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| e.to_string())?;
    println!(
        "engramdb serve listening on {}:{} (root {:?})",
        host, port, root
    );
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let root = root.to_path_buf();
                std::thread::spawn(move || {
                    let _ = handle_connection(root, stream);
                });
            }
            Err(e) => eprintln!("serve accept error: {e}"),
        }
    }
    Ok(())
}

fn base64_encode(data: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHA[(n >> 18) as usize & 63] as char);
        out.push(ALPHA[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHA[(n >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHA[n as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn list_empty_root() {
        let dir = std::env::temp_dir().join("engramdb-serve-list-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let names = list_tables(&dir).unwrap();
        assert!(names.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_table_valid_and_missing() {
        let dir = std::env::temp_dir().join("engramdb-serve-check-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "layout": {
                    "shards": 2,
                    "rows_per_shard": 4,
                    "width": 8,
                    "elem_bytes": 1,
                }
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("shard_000.bin"), vec![0u8; 32]).unwrap();
        std::fs::write(dir.join("shard_001.bin"), vec![0u8; 32]).unwrap();
        let v = check_table(&dir).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["shards_found"], 2);

        std::fs::remove_file(dir.join("shard_001.bin")).unwrap();
        let v = check_table(&dir).unwrap();
        assert_eq!(v["ok"], false);
        assert!(!v["issues"].as_array().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
