//! 批式 badge 读取：排序去重 + 预取计划 + 多线程 gather。

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use engramdb_core::layout::Layout;

/// 预取计划：按分片分组的 badge 块列表（每 shard 内部升序，供顺序预读/合并）。
#[derive(Debug, Default, Clone)]
pub struct PrefetchPlan {
    /// shard_id -> 去重升序的 badge 块号
    pub shard_badges: HashMap<u64, Vec<u64>>,
    pub n_badges: usize,
}

impl PrefetchPlan {
    pub fn build(keys: &[u64], layout: &Layout) -> Self {
        let mut shard_badges: HashMap<u64, Vec<u64>> = HashMap::new();
        for &k in keys {
            let (shard, badge, _) = layout.locate(k);
            shard_badges.entry(shard).or_default().push(badge);
        }
        for v in shard_badges.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        let n_badges = shard_badges.values().map(|v| v.len()).sum();
        Self {
            shard_badges,
            n_badges,
        }
    }

    pub fn badges(&self, shard: u64) -> &[u64] {
        self.shard_badges
            .get(&shard)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// 读路径（pread 池化）：一次 `gather_plan` = 按 plan 拉取全部 badge 并组装 `[n,width]`。
pub struct BadgeGather<'a> {
    pub layout: &'a Layout,
    files: Vec<File>,
}

impl<'a> BadgeGather<'a> {
    pub fn open(dir: &Path, layout: &'a Layout) -> std::io::Result<Self> {
        let n = layout.shards as usize;
        let mut files = Vec::with_capacity(n);
        for i in 0..n {
            files.push(File::open(dir.join(format!("shard_{:03}.bin", i)))?);
        }
        Ok(Self { layout, files })
    }

    pub fn into_files(self) -> Vec<File> {
        self.files
    }

    /// 多线程 gather：`keys`/`out` 按 chunk 分片并行（各线程独立 badge 缓冲区）。
    pub fn gather_parallel(
        &self,
        keys: &[u64],
        out: &mut [u8],
        threads: usize,
    ) -> std::io::Result<()> {
        if threads <= 1 || keys.len() <= 1024 {
            return self.gather_naive(keys, out);
        }
        let w = self.layout.width as usize;
        let chunk_keys = keys.len().div_ceil(threads);
        std::thread::scope(|s| {
            for (kc, oc) in keys.chunks(chunk_keys).zip(out.chunks_mut(chunk_keys * w)) {
                s.spawn(|| {
                    let _ = self.gather_naive(kc, oc);
                });
            }
        });
        Ok(())
    }

    /// 朴素单线程：逐 key 定位 badge 读取（每个线程独立缓冲区）。
    pub fn gather_naive(&self, keys: &[u64], out: &mut [u8]) -> std::io::Result<()> {
        let w = self.layout.width as usize;
        let rb = self.layout.row_bytes as usize;
        let mut badge_buf = vec![0u8; self.layout.badge_bytes() as usize];
        let mut last: Option<(u64, u64)> = None;
        for (i, &k) in keys.iter().enumerate() {
            let (shard, badge, in_badge) = self.layout.locate(k);
            if last != Some((shard, badge)) {
                let off = badge * self.layout.badge_bytes();
                self.files[shard as usize].read_exact_at(&mut badge_buf, off)?;
                last = Some((shard, badge));
            }
            let src = in_badge as usize * rb;
            out[i * w..(i + 1) * w].copy_from_slice(&badge_buf[src..src + w]);
        }
        Ok(())
    }

    /// 页对齐专用读路径：按 shard 分线程，shard 内按键升序聚页（4KiB 对齐，每页只读一次），
    /// 行跨页边界时补读该行尾部。各线程只写自己的采集缓冲，主线程回填 out（无共享 &mut）。
    pub fn gather_pp(&self, keys: &[u64], out: &mut [u8], threads: usize) -> std::io::Result<()> {
        const PAGE: u64 = 4096;
        let w = self.layout.width as usize;
        let rb = self.layout.row_bytes as usize;
        let mut groups: HashMap<u64, Vec<(u64, usize)>> = HashMap::new();
        for (i, &k) in keys.iter().enumerate() {
            let (shard, _, _) = self.layout.locate(k);
            groups.entry(shard).or_default().push((k, i));
        }
        let mut tasks: Vec<(u64, Vec<(u64, usize)>)> = groups.into_iter().collect();
        tasks.sort_unstable_by_key(|&(s, _)| s);

        // 各任务独立产出 (idxs 升序, rows 扁平)
        let nt = threads.max(1).min(tasks.len());
        let chunk = tasks.len().div_ceil(nt);
        let mut results: Vec<(Vec<usize>, Vec<u8>)> = Vec::new();

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let mut task_iter = tasks.into_iter();
            while task_iter.len() > 0 {
                let t: Vec<(u64, Vec<(u64, usize)>)> = task_iter.by_ref().take(chunk).collect();
                handles.push(s.spawn(move || {
                    let mut out_rows: Vec<u8> = Vec::new();
                    let mut out_idxs: Vec<usize> = Vec::new();
                    for (shard, mut pairs) in t {
                        pairs.sort_unstable();
                        let f = &self.files[shard as usize];
                        let mut last_page: Option<u64> = None;
                        let mut page = vec![0u8; (PAGE + 2 * (rb as u64)) as usize];
                        let mut prev_key: Option<u64> = None;
                        for (k, oi) in pairs {
                            let (_, _, in_b) = self.layout.locate(k);
                            let byte_off = k * rb as u64;
                            let page_id = byte_off & !(PAGE - 1);
                            if last_page != Some(page_id) {
                                let want = (PAGE + rb as u64) as usize;
                                let n = f.read_at(&mut page[..want], page_id).unwrap_or(0);
                                let _ = n;
                                last_page = Some(page_id);
                            }
                            let in_page = (byte_off - page_id) as usize;
                            if in_page + rb <= PAGE as usize {
                                out_rows.extend_from_slice(&page[in_page..in_page + rb]);
                            } else {
                                let mut tmp = vec![0u8; rb];
                                let _ = f.read_exact_at(&mut tmp, byte_off);
                                out_rows.extend_from_slice(&tmp);
                            }
                            out_idxs.push(oi);
                            let _ = (in_b, prev_key);
                            prev_key = Some(k);
                        }
                    }
                    (out_idxs, out_rows)
                }));
            }
            for h in handles {
                if let Ok(r) = h.join() {
                    results.push(r);
                }
            }
        });

        for (idxs, rows) in results {
            for (j, &oi) in idxs.iter().enumerate() {
                let slice = &rows[j * w..(j + 1) * w];
                out[oi * w..(oi + 1) * w].copy_from_slice(slice);
            }
        }
        Ok(())
    }

    /// 有序批式读：对同一 badge 的 keys 组内合并读（按 plan 的排序），
    /// 期望预取服务器将 plan 先落地 —— 本实现直接做"计划->读取"。
    pub fn gather_planned(&self, keys: &[u64], out: &mut [u8]) -> std::io::Result<()> {
        // 简单实现：按 (shard,badge) 分组，逐组读一次，再按 key 顺序回填
        let plan = PrefetchPlan::build(keys, self.layout);
        let rb = self.layout.row_bytes as usize;
        let w = self.layout.width as usize;
        let mut cache: HashMap<(u64, u64), Vec<u8>> = HashMap::new();
        let mut groups: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
        for (i, &k) in keys.iter().enumerate() {
            let (s, b, _) = self.layout.locate(k);
            groups.entry((s, b)).or_default().push(i);
        }
        for (&(s, b), idxs) in &groups {
            let buf = cache.entry((s, b)).or_insert_with(|| {
                let mut buf = vec![0u8; self.layout.badge_bytes() as usize];
                let off = b * self.layout.badge_bytes();
                let _ = self.files[s as usize].read_exact_at(&mut buf, off);
                buf
            });
            for &i in idxs {
                let (_, _, in_badge) = self.layout.locate(keys[i]);
                let src = in_badge as usize * rb;
                out[i * w..(i + 1) * w].copy_from_slice(&buf[src..src + w]);
            }
        }
        let _ = plan;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engramdb_core::layout::Layout;

    #[test]
    fn plan_dedup_sort() {
        let layout = Layout::new(1, 10_000, 160, 1); // 单分片
        let keys = vec![9999, 0, 5, 9999, 250, 250];
        let p = PrefetchPlan::build(&keys, &layout);
        assert_eq!(p.badges(0), &[0, 10, 399]);
    }
}
