//! Native disk-backed Store-P slot index.
//!
//! This is the Rust counterpart of :mod:`engramdb.disk_slot_index`.  It writes
//! the same on-disk layout as the Python v2 format (``engramdb-disk-slot-index-v2``)
//! so a view built by the CLI can be consumed directly by Python/qwen35-ple.
//!
//! A record is 136 bytes:
//! * 16 little-endian ``u64`` rowids (the PLE 16-head tuple), then
//! * 1 little-endian ``u64`` physical slot id.
//!
//! Bucketing uses FNV-1a 64 over the 128-byte key, matching
//! :func:`engramdb.disk_slot_index._fnv1a_bucket`.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use engramdb_core::fnv64;

pub const FORMAT_V2: &str = "engramdb-disk-slot-index-v2";
pub const HEADS: usize = 16;
pub const RECORD_BYTES: usize = HEADS * 8 + 8;
pub const DEFAULT_BUCKETS: usize = 16384;
pub const DEFAULT_CACHE_BUCKETS: usize = 64;

type Record = [u8; RECORD_BYTES];

pub struct BuildStats {
    pub count: u64,
    pub bytes: u64,
    pub seconds: f64,
}

fn key_from_gram(parts: &[u64; HEADS]) -> [u8; HEADS * 8] {
    let mut key = [0u8; HEADS * 8];
    for (i, v) in parts.iter().enumerate() {
        key[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    key
}

fn bucket_of(key: &[u8], num_buckets: usize) -> usize {
    (fnv64(key) % num_buckets as u64) as usize
}

fn slot_of_record(rec: &Record) -> u64 {
    u64::from_le_bytes(rec[HEADS * 8..].try_into().unwrap())
}

/// Build a disk slot index from a flat EngramDB keys file.
///
/// The keys file contains one rowid per line, 16 rowids per gram, in physical
/// slot order.  The builder streams the file twice; peak memory is bounded by
/// one bucket at a time during the sort phase.
pub fn build_from_keys_file(
    keys_path: &Path,
    out_dir: &Path,
    num_buckets: usize,
) -> Result<BuildStats, String> {
    let t0 = std::time::Instant::now();
    if num_buckets == 0 {
        return Err("num_buckets must be > 0".into());
    }
    std::fs::create_dir_all(out_dir).map_err(io_err)?;
    let buckets_dir = out_dir.join("buckets");
    std::fs::create_dir_all(&buckets_dir).map_err(io_err)?;
    let tmp_dir = out_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(io_err)?;

    let raw_path = tmp_dir.join("raw.bin");
    let grouped_path = tmp_dir.join("grouped.bin");

    // Pass 1: stream keys, write raw ordered records, count bucket occupancy.
    let counts = pass1(keys_path, &raw_path, num_buckets)?;
    let count: u64 = counts.iter().sum();

    // Pass 2: scatter raw records into a single grouped file by bucket.
    let mut offsets = vec![0u64; num_buckets + 1];
    for i in 0..num_buckets {
        offsets[i + 1] = offsets[i] + counts[i] * RECORD_BYTES as u64;
    }
    let grouped_size = offsets[num_buckets];
    let mut grouped = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&grouped_path)
        .map_err(io_err)?;
    grouped.set_len(grouped_size).map_err(io_err)?;
    let mut cursors = vec![0u64; num_buckets];
    pass2(&raw_path, &mut grouped, &offsets, &mut cursors)?;
    drop(grouped);

    // Pass 3: sort each bucket region and write one small file per bucket.
    let mut grouped_file = File::open(&grouped_path).map_err(io_err)?;
    for bucket in 0..num_buckets {
        let start = offsets[bucket];
        let end = offsets[bucket + 1];
        if start == end {
            continue;
        }
        grouped_file.seek(SeekFrom::Start(start)).map_err(io_err)?;
        let len = (end - start) as usize;
        let mut data = vec![0u8; len];
        grouped_file.read_exact(&mut data).map_err(io_err)?;
        let mut records: Vec<Record> = data
            .chunks_exact(RECORD_BYTES)
            .map(|c| c.try_into().expect("record size"))
            .collect();
        records.sort();
        let mut buf = Vec::with_capacity(len);
        for r in &records {
            buf.extend_from_slice(r);
        }
        let path = buckets_dir.join(format!("{bucket:04}.bin"));
        std::fs::write(&path, &buf).map_err(io_err)?;
    }

    let meta = serde_json::json!({
        "format": FORMAT_V2,
        "hash": "fnv1a-64",
        "heads": HEADS,
        "num_buckets": num_buckets,
        "count": count,
        "record_bytes": RECORD_BYTES,
        "cache_buckets": DEFAULT_CACHE_BUCKETS,
    });
    std::fs::write(
        out_dir.join("index.json"),
        serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(io_err)?;

    let _ = std::fs::remove_dir_all(&tmp_dir);
    let bytes = grouped_size;
    Ok(BuildStats {
        count,
        bytes,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

fn pass1(keys_path: &Path, raw_path: &Path, num_buckets: usize) -> Result<Vec<u64>, String> {
    let mut counts = vec![0u64; num_buckets];
    let mut raw = BufWriter::new(File::create(raw_path).map_err(io_err)?);
    let file = File::open(keys_path).map_err(io_err)?;
    let mut reader = BufReader::new(file);
    let mut parts = [0u64; HEADS];
    let mut pos = 0usize;
    let mut slot = 0u64;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).map_err(io_err)?;
        if n == 0 {
            break;
        }
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let v: u64 = s.parse().map_err(|e: std::num::ParseIntError| {
            format!(
                "invalid rowid at line {}: {}",
                slot * HEADS as u64 + pos as u64 + 1,
                e
            )
        })?;
        parts[pos] = v;
        pos += 1;
        if pos == HEADS {
            let key = key_from_gram(&parts);
            let bucket = bucket_of(&key, num_buckets);
            counts[bucket] += 1;
            raw.write_all(&key).map_err(io_err)?;
            raw.write_all(&slot.to_le_bytes()).map_err(io_err)?;
            slot += 1;
            pos = 0;
        }
    }
    if pos != 0 {
        return Err(format!(
            "keys file ended with {} rowids in an incomplete {HEADS}-head tuple",
            pos
        ));
    }
    raw.flush().map_err(io_err)?;
    Ok(counts)
}

fn pass2(
    raw_path: &Path,
    grouped: &mut File,
    offsets: &[u64],
    cursors: &mut [u64],
) -> Result<(), String> {
    let mut raw = BufReader::new(File::open(raw_path).map_err(io_err)?);
    let mut rec = [0u8; RECORD_BYTES];
    loop {
        let mut filled = 0;
        while filled < RECORD_BYTES {
            let n = raw.read(&mut rec[filled..]).map_err(io_err)?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break;
        }
        if filled != RECORD_BYTES {
            return Err("truncated raw record during scatter".into());
        }
        let key: [u8; HEADS * 8] = rec[..HEADS * 8].try_into().unwrap();
        let bucket = bucket_of(&key, offsets.len() - 1);
        let pos = offsets[bucket] + cursors[bucket] * RECORD_BYTES as u64;
        grouped.seek(SeekFrom::Start(pos)).map_err(io_err)?;
        grouped.write_all(&rec).map_err(io_err)?;
        cursors[bucket] += 1;
    }
    Ok(())
}

/// Read-only disk slot index with a small bounded bucket LRU.
pub struct DiskSlotIndexReader {
    dir: PathBuf,
    num_buckets: usize,
    count: u64,
    cache: HashMap<usize, Vec<Record>>,
    order: VecDeque<usize>,
    cache_capacity: usize,
}

impl DiskSlotIndexReader {
    pub fn open(dir: &Path, cache_capacity: usize) -> Result<Self, String> {
        let meta_path = dir.join("index.json");
        if !meta_path.exists() {
            return Err(format!("slot index not found: {}", meta_path.display()));
        }
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).map_err(io_err)?)
                .map_err(|e| e.to_string())?;
        let format = meta
            .get("format")
            .and_then(|v| v.as_str())
            .ok_or("index.json missing format")?;
        if format != FORMAT_V2 {
            return Err(format!(
                "unsupported slot index format: {format} (expected {FORMAT_V2})"
            ));
        }
        let num_buckets = meta
            .get("num_buckets")
            .and_then(|v| v.as_u64())
            .ok_or("index.json missing num_buckets")? as usize;
        let count = meta
            .get("count")
            .and_then(|v| v.as_u64())
            .ok_or("index.json missing count")?;
        Ok(Self {
            dir: dir.to_path_buf(),
            num_buckets,
            count,
            cache: HashMap::new(),
            order: VecDeque::new(),
            cache_capacity: cache_capacity.max(1),
        })
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    fn records(&mut self, bucket: usize) -> Result<&[Record], String> {
        if !self.cache.contains_key(&bucket) {
            let path = self.dir.join("buckets").join(format!("{bucket:04}.bin"));
            let records: Vec<Record> = if path.exists() {
                let data = std::fs::read(&path).map_err(io_err)?;
                data.chunks_exact(RECORD_BYTES)
                    .map(|c| c.try_into().expect("record size"))
                    .collect()
            } else {
                Vec::new()
            };
            if self.order.len() >= self.cache_capacity {
                if let Some(old) = self.order.pop_front() {
                    self.cache.remove(&old);
                }
            }
            self.cache.insert(bucket, records);
            self.order.push_back(bucket);
        }
        Ok(self.cache.get(&bucket).unwrap())
    }

    pub fn lookup(&mut self, parts: &[u64; HEADS]) -> Result<u64, String> {
        let key = key_from_gram(parts);
        let bucket = bucket_of(&key, self.num_buckets);
        let records = self.records(bucket)?;
        let idx = records
            .binary_search_by(|probe| probe[..HEADS * 8].cmp(&key[..]))
            .map_err(|_| format!("rowid tuple not found: {parts:?}"))?;
        Ok(slot_of_record(&records[idx]))
    }
}

/// Verify a disk slot index against the source flat keys file.
pub fn verify_from_keys_file(
    keys_path: &Path,
    idx_dir: &Path,
    cache_capacity: usize,
) -> Result<u64, String> {
    let mut reader = DiskSlotIndexReader::open(idx_dir, cache_capacity)?;
    let file = File::open(keys_path).map_err(io_err)?;
    let lines = BufReader::new(file).lines();
    let mut parts = [0u64; HEADS];
    let mut pos = 0usize;
    let mut slot = 0u64;
    let mut verified = 0u64;
    for line in lines {
        let line = line.map_err(io_err)?;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let v: u64 = s
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        parts[pos] = v;
        pos += 1;
        if pos == HEADS {
            let got = reader.lookup(&parts)?;
            if got != slot {
                return Err(format!(
                    "slot mismatch at gram {slot}: expected {slot}, index returned {got}"
                ));
            }
            verified += 1;
            slot += 1;
            pos = 0;
        }
    }
    if pos != 0 {
        return Err("keys file has incomplete rowid tuple".into());
    }
    if verified != reader.count() {
        return Err(format!(
            "slot index count {} != keys file count {verified}",
            reader.count()
        ));
    }
    Ok(verified)
}

fn io_err(e: io::Error) -> String {
    e.to_string()
}
