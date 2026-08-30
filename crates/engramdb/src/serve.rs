//! Minimal Rust multi-table / manifest / TCP serve prototype.
//!
//! This is the first Rust-side convergence for the Python service prototype:
//! a single process can enumerate tables under a root directory, read each
//! table's manifest, and serve simple JSON store fetches over TCP.
//!
//! The wire is intentionally minimal (newline-delimited JSON, like the Python
//! JSON server). Binary/Arrow IPC can be layered on later.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;
use serde_json::Value;

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

fn table_dir(root: &Path, table: &str) -> PathBuf {
    root.join(table)
}

fn fetch_table(root: &Path, req: &Value) -> Result<Value, String> {
    let table = req
        .get("table")
        .and_then(Value::as_str)
        .ok_or("fetch requires table")?;
    let dir = table_dir(root, table);
    if !dir.is_dir() {
        return Err(format!("table not found: {table}"));
    }

    let layout = if let Some(m) = load_manifest(&dir) {
        layout_from_manifest(&m).ok_or("manifest missing layout?")?
    } else {
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
        let elem_bytes = req
            .get("elem_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        Layout::new(shards, rows_per_shard, width, elem_bytes)
    };

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

    Ok(serde_json::json!({
        "ok": true,
        "table": table,
        "num_rows": rowids.len(),
        "width": layout.width,
        "raw_base64": base64_encode(&out),
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
        other => serde_json::json!({"ok": false, "error": format!("unknown command: {other}")}),
    }
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
    println!("engramdb serve listening on {}:{} (root {:?})", host, port, root);
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
}
