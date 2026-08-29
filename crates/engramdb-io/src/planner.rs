//! 流式预取规划器（M1.5 首交付）：连续的 token rowid 流 → 增量 badge 读取计划。
//!
//! 证据与设计（docs/design.md §3.4、§7.0）：
//! - 预取窗口 = 计算窗口：窗口随 token 推进滚动，IO 与计算重叠；
//! - 窗口内 badge/BLOCK 只派发一次（下一 token 重复访问 = 已驻留，不再读盘）；
//! - 不做语料级热集：P2 证明大语料下唯一行占表 37-49%、top1K 覆盖 <6%。
//!
//! 粒度说明：本版本按 "badge"（行簇）为读取单元，等价于 Store-I 路径；
//! Store-P（4KB 视图记录）接同一结构（badge 语义换成 view_page）。

use std::collections::{HashSet, VecDeque};

use crate::batch::PrefetchPlan;
use engramdb_core::layout::Layout;

pub struct StreamingPlanner {
    window_badges: usize,
    queue: VecDeque<(u64, u64)>,
    in_window: HashSet<(u64, u64)>,
}

impl StreamingPlanner {
    pub fn new(window_badges: usize) -> Self {
        Self {
            window_badges,
            queue: VecDeque::new(),
            in_window: HashSet::new(),
        }
    }

    /// 推进一批 rowid（一个 token 的 16 个 head 行）。
    /// 返回"新 badge"数量；`plan` 只追加本次需要读的 badge（升序语义由消费方 settle）。
    pub fn advance(&mut self, rows: &[u64], layout: &Layout, plan: &mut PrefetchPlan) -> usize {
        let mut n_new = 0;
        for &r in rows {
            let (shard, badge, _) = layout.locate(r);
            let key = (shard, badge);
            if !self.in_window.contains(&key) {
                self.queue.push_back(key);
                self.in_window.insert(key);
                plan.entry(shard, badge);
                n_new += 1;
            }
        }
        while self.in_window.len() > self.window_badges {
            if let Some(old) = self.queue.pop_front() {
                self.in_window.remove(&old);
            }
        }
        n_new
    }

    pub fn window_len(&self) -> usize {
        self.in_window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay() -> Layout {
        Layout::new(2, 10_000, 160, 1)
    }

    #[test]
    fn first_token_all_new() {
        let l = lay();
        let mut sp = StreamingPlanner::new(100);
        let mut plan = PrefetchPlan::default();
        let rows: Vec<u64> = (0..16).map(|i| i * 1000).collect();
        let n = sp.advance(&rows, &l, &mut plan);
        assert_eq!(n, 16);
        assert_eq!(plan.n_badges, 16);
        assert_eq!(sp.window_len(), 16);
    }

    #[test]
    fn repeated_token_reuses_window() {
        let l = lay();
        let mut sp = StreamingPlanner::new(100);
        let mut plan = PrefetchPlan::default();
        let rows: Vec<u64> = (0..16).map(|i| i * 1000).collect();
        let _ = sp.advance(&rows, &l, &mut plan);
        let mut plan2 = PrefetchPlan::default();
        let n = sp.advance(&rows, &l, &mut plan2);
        assert_eq!(n, 0, "same token again -> all badges resident");
        assert_eq!(plan2.n_badges, 0);
    }

    #[test]
    fn window_eviction_republish() {
        let l = lay();
        let mut sp = StreamingPlanner::new(4);
        // 推进 6 个互不相同的 token（各 2 行且不同 badge，窗口预算 4 badge）
        for t in 0..6 {
            let mut plan = PrefetchPlan::default();
            let rows: Vec<u64> = vec![t * 1000, (t + 30) * 1000];
            let n = sp.advance(&rows, &l, &mut plan);
            // 窗口=4：前两 token 会被挤出 -> 后续每 token 仍 2 新 badge
            assert_eq!(n, 2, "t={t} window len={}", sp.window_len());
        }
        assert!(sp.window_len() <= 4);
    }
}
