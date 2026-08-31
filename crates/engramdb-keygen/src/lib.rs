//! EngramDB keygen: 从 token 序列推导 PLE/Engram 确定性 rowid。
//!
//! ① Qwen PLE（精确实现，参照 transformers `qwen4_exp/modeling_qwen4_exp.py`）：
//!    - 乘子来自 checkpoint（[23703573157769, 20109073645365, 8052911324071]）
//!    - 每头素数 = nth_prime_after(20_000_000 - 1, global_head_idx + 1)，16 头
//!    - rowid = (Σ ⊗（移位后 token × 乘子）) % 素数 + 头偏移（offsets = 素数前缀和）
//!    - 段语义：`_shift_right_ignore_eos`（越界/跨 EOS 回填 EOS=248044）
//!    - 表：padded_vocab = ceil(Σ素数 /128)*128 = 320,001,536 行 × 160 维（= 128 分片 × 2,500,012）
//!
//! ② DeepSeek Engram（demo 口径）在 `demo.rs`：压缩词表 + 各层独立乘子（占位，M0 后续对拍）。
//!
//! 本 crate 无 IO 依赖，纯函数，`golden.json`（scripts/ref_ple_hash.py 生成）做 P0 对拍。

const PLE_MULTIPLIERS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

pub const PLE_EOS: u32 = 248_044;
pub const PLE_NGRAM_SIZE: usize = 3;
pub const PLE_HEADS_PER_NGRAM: usize = 8;
pub const PLE_HEADS: usize = 16;
pub const PLE_BASE: u64 = 20_000_000;
pub const PLE_DIVISOR: u64 = 128;
pub const PLE_SHARDS: u64 = 128;
pub const PLE_ROWS_PER_SHARD: u64 = 2_500_012;

/// PLE 表规格（真实 checkpoint 口径）。
#[derive(Debug, Clone)]
pub struct PleSpec {
    pub multipliers: [u64; 3],
    pub prime_sizes: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub divisors: Vec<u64>, // 每头素数（与 prime_sizes 相同，语义分离用）
    pub total: u64,         // Σ素数
    pub padded: u64,        // ceil(total/128)*128 = 320,001,536
    pub rows_per_shard: u64,
    pub shards: u64,
    pub eos: u32,
}

impl Default for PleSpec {
    fn default() -> Self {
        Self::real()
    }
}

impl PleSpec {
    pub fn real() -> Self {
        let mut prime_sizes = Vec::with_capacity(PLE_HEADS);
        for i in 0..PLE_HEADS as u64 {
            prime_sizes.push(nth_prime_after(PLE_BASE - 1, i + 1));
        }
        let mut head_offsets = Vec::with_capacity(PLE_HEADS);
        let mut acc = 0u64;
        for s in &prime_sizes {
            head_offsets.push(acc);
            acc += s;
        }
        let total = acc;
        let padded = round_up(total, PLE_DIVISOR);
        let rows_per_shard = PLE_ROWS_PER_SHARD;
        let shards = PLE_SHARDS;
        assert_eq!(
            rows_per_shard * shards,
            padded,
            "P0: 分片网格与 padded 表必须闭合"
        );
        assert_eq!(padded, 320_001_536);
        Self {
            multipliers: PLE_MULTIPLIERS,
            divisors: prime_sizes.clone(),
            prime_sizes,
            head_offsets,
            total,
            padded,
            rows_per_shard,
            shards,
            eos: PLE_EOS,
        }
    }

    /// 17-math 闭包验证（P0）：rowids 索引空间 ⊂ padded
    pub fn total_vocab(&self) -> u64 {
        self.total
    }

    /// 逐位置生成 16 个 rowid（含头偏移）。语义同官方 forward（cold start：前文 = EOS×2）。
    pub fn rowids_for_seq(&self, tokens: &[u32]) -> Vec<[u32; PLE_HEADS]> {
        let context = [self.eos; PLE_NGRAM_SIZE - 1];
        self.rowids_for_seq_with_history(&context, tokens)
    }

    /// 带显式 n-gram history 的逐位置 rowid 生成。
    ///
    /// `history` 是已知的前文（通常长度 = ngram_size - 1），`tokens` 是当前步输入；
    /// 返回只对应 `tokens` 的 rowid。语义与
    /// `engramdb.ple_adapter.ple_rowids(..., history=history)` 一致。
    pub fn rowids_for_seq_with_history(
        &self,
        history: &[u32],
        tokens: &[u32],
    ) -> Vec<[u32; PLE_HEADS]> {
        let t = tokens.len();
        let mut hist = Vec::with_capacity(history.len() + t);
        hist.extend_from_slice(history);
        hist.extend_from_slice(tokens);

        // _shift_right_ignore_eos for shift 0..3
        let mut shifted: Vec<Vec<u32>> = Vec::with_capacity(PLE_NGRAM_SIZE);
        for shift in 0..PLE_NGRAM_SIZE {
            shifted.push(shift_right_ignore_eos(&hist, shift, self.eos));
        }

        // ngram_ids 计算覆盖 hist 全部位置，最终取最后 t 个
        let hlen = hist.len();
        let mut ids_all: Vec<[u32; PLE_HEADS]> = Vec::with_capacity(hlen);
        for pos in 0..hlen {
            let mut row = [0u32; PLE_HEADS];
            for (ngram_order, shift_range) in [(2, 0usize), (3, PLE_HEADS_PER_NGRAM)] {
                let mut mixed = (shifted[0][pos] as u64).wrapping_mul(self.multipliers[0]);
                for (shifted_row, m) in shifted
                    .iter()
                    .take(ngram_order)
                    .skip(1)
                    .zip(self.multipliers.iter().skip(1))
                {
                    mixed ^= (shifted_row[pos] as u64).wrapping_mul(*m);
                }
                for h in 0..PLE_HEADS_PER_NGRAM {
                    let gi = (ngram_order - 2) * PLE_HEADS_PER_NGRAM + h; // 全局头序号 0..16
                    let idx = shift_range + h;
                    let (size, off) = (self.divisors[gi], self.head_offsets[gi]);
                    let rid = (mixed % size) + off;
                    row[idx] = rid as u32;
                }
            }
            ids_all.push(row);
        }
        let skip = hist.len() - t;
        ids_all[skip..].to_vec()
    }
}

fn round_up(v: u64, d: u64) -> u64 {
    v.div_ceil(d) * d
}

/// 官方 `_shift_right_ignore_eos`：移 shift 位，跨段（EOS 后不足）回填 eos。
fn shift_right_ignore_eos(hist: &[u32], shift: usize, eos: u32) -> Vec<u32> {
    if shift == 0 {
        return hist.to_vec();
    }
    let n = hist.len();
    // position_in_segment：最后一个 EOS（严格在 pos 之前）+1 为段起点
    let mut prev_incl = vec![-1i64; n];
    let mut last = -1i64;
    for (i, &x) in hist.iter().enumerate() {
        if x == eos {
            last = i as i64;
        }
        prev_incl[i] = last;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let seg_start = if i == 0 { 0 } else { prev_incl[i - 1] + 1 };
        let pos_in_seg = i as i64 - seg_start;
        let src = i as i64 - shift as i64;
        let valid = pos_in_seg >= shift as i64 && src >= 0;
        out.push(if valid { hist[src as usize] } else { eos });
    }
    out
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3u64;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

pub fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    #[test]
    fn primes_match_gguf() {
        let s = PleSpec::real();
        assert_eq!(&s.prime_sizes[..3], &[20_000_003, 20_000_023, 20_000_033]);
        assert_eq!(s.padded, 320_001_536);
        assert_eq!(s.rows_per_shard * s.shards, s.padded);
    }

    #[test]
    fn matches_python_golden() {
        let s = PleSpec::real();
        let gen = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden.json");
        let data = std::fs::read_to_string(&gen)
            .expect("golden.json：由 scripts/ref_ple_hash.py 生成并纳入版本库");
        let g: serde_json::Value = serde_json::from_str(&data).expect("golden json");
        let tokens: Vec<u32> = g["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expect: Vec<Vec<u32>> = g["rowids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect();
        let got = s.rowids_for_seq(&tokens);
        assert_eq!(got.len(), expect.len());
        for (i, (gv, ev)) in got.iter().zip(expect.iter()).enumerate() {
            let gv: Vec<u32> = gv.to_vec();
            if gv != *ev && i < 4 {
                eprintln!(
                    "[debug mismatch] pos={i} hashes that differ: {:?}",
                    gv.iter()
                        .zip(ev.iter())
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .collect::<Vec<_>>()
                );
                eprintln!("[debug mismatch] got {:?}", &gv[..8]);
                eprintln!("[debug mismatch] exp {:?}", &ev[..8]);
            }
            assert_eq!(gv, *ev, "position {i} mismatch");
        }
    }

    #[test]
    fn rowids_in_padded_space() {
        let s = PleSpec::real();
        let ids = s.rowids_for_seq(&[1000, 99_999, 42]);
        for r in ids {
            assert!(r.iter().all(|&x| (x as u64) < s.padded));
        }
    }

    #[test]
    fn history_matches_streaming_concatenation() {
        let s = PleSpec::real();
        let tokens = [10u32, 11, 12, 13];
        let full = s.rowids_for_seq(&tokens);
        let part1 = s.rowids_for_seq_with_history(&[s.eos, s.eos], &[10, 11]);
        let part2 = s.rowids_for_seq_with_history(&[10, 11], &[12, 13]);
        assert_eq!(&full[..2], &part1[..]);
        assert_eq!(&full[2..], &part2[..]);
    }
}
