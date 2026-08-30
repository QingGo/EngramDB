#![cfg(unix)]
//! Store-I 的批量 gather：分片文件 + 行号 → 组装 `[n, width]` 缓冲区。
//!
//! 最小实现：每个 shard 一个文件，`read_at` 按 badge 读取并拷贝行。
//! 大数据量生产版：io_uring / preadv 多队列、页缓存命中判定、双缓冲（M1 实现）。
//! 本文件为 M0-P1 探针服务：真实调用路径（行号寻址）不变。

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::layout::Layout;

pub struct ShardedStore {
    pub layout: Layout,
    files: Vec<File>,
}

impl ShardedStore {
    pub fn open(dir: &Path, layout: Layout) -> std::io::Result<Self> {
        let mut files = Vec::with_capacity(layout.shards as usize);
        for i in 0..layout.shards {
            let p = dir.join(format!("shard_{:03}.bin", i));
            files.push(File::open(p)?);
        }
        Ok(Self { layout, files })
    }

    /// 批取 `keys` 行到 `out[n*width]`（按 rowid 顺序；顺序无关，因为 gather 是按 key 的）。
    pub fn gather(&self, keys: &[u64], out: &mut [u8]) -> std::io::Result<()> {
        let w = self.layout.width as usize;
        let rb = self.layout.row_bytes as usize;
        let mut badge_buf = vec![0u8; self.layout.badge_bytes() as usize];
        let mut current_badge = u64::MAX;
        for (i, &k) in keys.iter().enumerate() {
            let (shard, badge, in_badge) = self.layout.locate(k);
            let off = badge * self.layout.badge_bytes();
            if (shard, badge) != (current_badge, current_badge >> 32) {
                // 一次读取 badge 块（4KB 对齐粒度）
                self.files[shard as usize].read_exact_at(&mut badge_buf, off)?;
                current_badge = shard << 32 | badge;
            }
            let src = in_badge as usize * rb;
            let dst = i * w;
            out[dst..dst + w].copy_from_slice(&badge_buf[src..src + w]);
        }
        Ok(())
    }
}
