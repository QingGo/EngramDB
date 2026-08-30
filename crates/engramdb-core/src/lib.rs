//! EngramDB core：与 IO 后端无关的布局、索引与批量 gather 原语。
//!
//! 设计基线（`docs/design.md` §3）：
//! - Store-I 原始表视图：行 = rowid → 分片/偏移；与真实 PLE checkpoint 一致的分片组织
//! - badge 聚集：`badge = BPows` 行一组，按 4KB/2MB 对齐，行宽 row_bytes 固定
//! - 直接寻址：rowid → (shard, badge_id, in_badge_row) 全部为整数除法一次

pub mod count_index;
pub mod fnv;
pub mod layout;
#[cfg(unix)]
pub mod store;

pub use count_index::CountIndex;
pub use fnv::fnv64;
pub use layout::{Layout, RowExtent};
