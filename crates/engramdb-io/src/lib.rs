//! engramdb-io：分层读取、批量 badge gather、预取计划。
//!
//! M1 范围：Warm+冷路径（OS 页缓存 + 分片文件 pread），带排序去重的批式 badge 读取，
//! 多线程池化；T3 NVMe 专用 io_uring 后端留 M1.5（Linux）替换 `pread` 后端，接口不变。

pub mod backend;
pub mod batch;
pub mod planner;
pub mod tiers;

pub use backend::{default_backend, IoBackend, PreadvBackend};
pub use batch::{BadgeGather, PrefetchPlan};
pub use planner::StreamingPlanner;
pub use tiers::TierManager;
