//! 分层管理（M1 简单版）：频率驱动热集 + 预算；T1=RAM 显式驻留，T2=OS 页缓存，T3=显式 IO。
//!
//! M1 引入 TierManager 的决策面：给定 count 索引与 ram budget，
//! 输出"热集行集合"（T1）与"需要显式预取的分片-块"（T3 提示）。
//! M1.5 后接 io_uring 后端与共享内存缓存池（服务化）。

use std::collections::BinaryHeap;
use std::path::Path;

/// 热集：count 排序的 top-N 行（T1 驻留候选）。
#[derive(Debug, Default)]
pub struct TierManager {
    pub hot_rows: Vec<u64>,
    pub ram_budget_bytes: u64,
}

impl TierManager {
    /// 从 count 索引构建热集（仅取 top `$budget/row_bytes` 行）。
    /// 输入语义：`row_costs` = (rowid, count)。为保持 M1 小而正确，
    /// 这里接 "count-dump" 文件：行 {rowid} {count}（步进）。
    pub fn from_counts_dump(
        path: &Path,
        budget_bytes: u64,
        row_bytes: u64,
    ) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut heap: BinaryHeap<(u64, u64)> = BinaryHeap::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let (rowid, count) = match (it.next(), it.next()) {
                (Some(a), Some(b)) => (a.parse().unwrap_or(0), b.parse().unwrap_or(0)),
                _ => (0, 0),
            };
            if rowid == 0 {
                continue;
            }
            heap.push((count, rowid));
        }
        let max_rows = (budget_bytes / row_bytes.max(1)) as usize;
        let mut hot_rows = Vec::with_capacity(max_rows.min(heap.len()));
        while hot_rows.len() < max_rows {
            match heap.pop() {
                Some((_, rowid)) => hot_rows.push(rowid),
                None => break,
            }
        }
        hot_rows.sort_unstable();
        Ok(Self {
            hot_rows,
            ram_budget_bytes: budget_bytes,
        })
    }

    pub fn is_hot(&self, rowid: u64) -> bool {
        // M1：线性（热集调用点在 gather 之前，后续优化为 bitset）
        self.hot_rows.binary_search(&rowid).is_ok()
    }
}
