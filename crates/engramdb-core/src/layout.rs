//! 布局：行宽、badge、分片划分（与真实 checkpoint 结构一致的参数化小实现）。

/// 物理布局描述：分片化的定长行存储（如 PLE: 128 shards × [2_500_012, 160] FP8）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    pub shards: u64,
    pub rows_per_shard: u64,
    pub width: u64,
    pub row_bytes: u64,     // width * elem_bytes
    pub badge_rows: u64,    // 每个 badge 的行数（用户可调，对齐 4KB 优先）
}

impl Layout {
    pub fn new(shards: u64, rows_per_shard: u64, width: u64, elem_bytes: u64) -> Self {
        let row_bytes = width * elem_bytes;
        let badge_rows = aligned_badge_rows(row_bytes, 4096);
        Self { shards, rows_per_shard, width, row_bytes, badge_rows }
    }

    pub fn total_rows(&self) -> u64 { self.shards * self.rows_per_shard }

    /// badge 字节数（4KB 对齐推荐）
    pub fn badge_bytes(&self) -> u64 { self.badge_rows * self.row_bytes }

    /// rowid → (shard_id, badge_id, in_badge_row)
    pub fn locate(&self, rowid: u64) -> (u64, u64, u64) {
        let shard = rowid / self.rows_per_shard;
        let in_shard = rowid % self.rows_per_shard;
        let badge = in_shard / self.badge_rows;
        let in_badge = in_shard % self.badge_rows;
        (shard, badge, in_badge)
    }

    /// 一维张量模式：总表 = 单列大数组（当 shards == 1 时）
    pub fn byte_offset(&self, rowid: u64) -> u64 {
        rowid * self.row_bytes
    }
}

/// 行宽 -> 尽量使 badge ≥ 4KB 且行数整除性佳的 badge_rows。
pub fn aligned_badge_rows(row_bytes: u64, min_bytes: u64) -> u64 {
    let per = (min_bytes / row_bytes.max(1)).max(1);
    per
}

/// 行范围（为批量连续读服务）
#[derive(Debug, Clone, Copy)]
pub struct RowExtent {
    pub rows: u64,
    pub row_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_matches_struct() {
        // 真实 PLE 规格的 1/16 mock：8 shards × 156_251 行 × 160 宽
        let l = Layout::new(8, 156_251, 160, 1);
        assert_eq!(l.total_rows(), 8 * 156_251);
        assert_eq!(l.row_bytes, 160);
        // 4KB / 160B = 25.6 → 25 行/badge
        assert_eq!(l.badge_rows, 25);
        assert_eq!(l.badge_bytes(), 25 * 160);
        let (s, b, r) = l.locate(156_253);
        assert_eq!((s, b, r), (1, 0, 2));
    }

    #[test]
    fn row_bytes_bf16() {
        let l = Layout::new(8, 156_251, 160, 2);
        assert_eq!(l.row_bytes, 320);
        // 4096/320=12.8 → 12 行
        assert_eq!(l.badge_rows, 12);
    }
}
