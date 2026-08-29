//! I2 频率索引：rowid → count（排序键值对文件）与热集选择。
//!
//! 构建：流式 sum（计数原子性由调用方保证，去重由输入保证），内存 map → 排序 → 落盘。
//! 400M 级表全量应用时换外部排序（M1.5）；当前语料规模（千万级唯一）内存版足够。

use std::collections::HashMap;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Default)]
pub struct CountIndex {
    counts: HashMap<u64, u32>,
}

impl CountIndex {
    /// 从二进制流构建：每行 u64 (nat endian) rowid 序列（重复即累加）。
    pub fn build_from_bin_stream(mut src: impl Read) -> std::io::Result<Self> {
        let mut counts = HashMap::<u64, u32>::new();
        let mut buf = [0u8; 8];
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            if n != 8 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "partial rowid",
                ));
            }
            let rowid = u64::from_le_bytes(buf);
            *counts.entry(rowid).or_insert(0) += 1;
        }
        Ok(Self { counts })
    }

    pub fn count(&self, rowid: u64) -> u64 {
        self.counts.get(&rowid).copied().unwrap_or(0) as u64
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, u32)> {
        let mut v: Vec<_> = self.counts.iter().map(|(&a, &b)| (a, b)).collect();
        v.sort_unstable_by_key(|&(a, _)| a);
        v.into_iter()
    }

    /// 落盘：binary (u64 rowid, u32 count) 排序流。
    pub fn write_bin(&self, path: &Path) -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        for (rowid, count) in self.iter() {
            w.write_all(&rowid.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
        }
        w.flush()
    }

    pub fn write_dump(&self, path: &Path) -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        for (rowid, count) in self.iter() {
            writeln!(w, "{} {}", rowid, count)?;
        }
        w.flush()
    }

    pub fn from_bin(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let mut counts = HashMap::new();
        for chunk in data.chunks(12) {
            if chunk.len() != 12 {
                break;
            }
            let rowid = u64::from_le_bytes(chunk[..8].try_into().unwrap());
            let c = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
            counts.insert(rowid, c);
        }
        Ok(Self { counts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_count_write_read() {
        let data: Vec<u8> = [5u64, 3, 5, 7, 5]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let idx = CountIndex::build_from_bin_stream(&data[..]).unwrap();
        assert_eq!(idx.count(5), 3);
        assert_eq!(idx.count(3), 1);
        let p = std::env::temp_dir().join("ct_test.bin");
        idx.write_bin(&p).unwrap();
        let idx2 = CountIndex::from_bin(&p).unwrap();
        assert_eq!(idx2.count(7), 1);
    }
}
