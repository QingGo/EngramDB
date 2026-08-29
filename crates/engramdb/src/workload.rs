//! 真实负载生成：把 `probes/agent_workload_stats.json`（真实 agent 流量分布）转为
//! CLI 可用的行键序列。P2 结论：agent 型负载 = 高频热集 + 短尾新 token，
//! 与 uniform 随机（旧 bench-real）的天壤之别——这里把两种模式都做成一等公民对照。
//!
//! 模型（近真）：
//! - 每请求 token 数 ~ in_tokens 分布（p50/p95/max 三分位插值 + 抖动）
//! - 每 token 生成 16 行（与 P1 口径一致）：行 = rowids_for_seq([t-2, t-1, t])[0]
//! - 热 token：全程共享的热集（模拟 agent 域高频重复）；每请求续用上一请求尾部
//!   （会话记忆）+ 少量新 token 注入（`hot_hit` 概率可调）

use std::path::Path;

pub struct Dist {
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl Dist {
    fn from_json(v: &serde_json::Value) -> Option<Self> {
        Some(Self {
            p50: v.get("p50")?.as_f64()?,
            p95: v.get("p95")?.as_f64()?,
            max: v.get("max")?.as_f64()?,
        })
    }
}

pub struct AgentStats {
    pub in_tokens: Dist,
    #[allow(dead_code)]
    pub blocks: Dist,
}

impl AgentStats {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let in_b = v
            .get("in_tokens")
            .and_then(Dist::from_json)
            .ok_or("stats: 缺 in_tokens")?;
        let bl = v
            .get("blocks_per_request")
            .and_then(Dist::from_json)
            .ok_or("stats: 缺 blocks_per_request")?;
        Ok(Self {
            in_tokens: in_b,
            blocks: bl,
        })
    }

    /// 二分位插值抽样（r ∈ [0,1)）。
    fn draw(d: &Dist, r: f64, jitter: f64) -> u32 {
        let (lo, hi) = if r < 0.5 {
            (d.p50, d.p95)
        } else {
            (d.p95, d.max)
        };
        let t = if r < 0.5 { r / 0.5 } else { (r - 0.5) / 0.5 };
        let v = lo + (hi - lo) * t;
        let v = v * (1.0 + jitter * (r - 0.5));
        v.max(16.0) as u32
    }
}

/// token 词表大小（真实 PLE）。
pub const VOCAB: u32 = 248_320;

/// xorshift64（无 rand 依赖，确定性种子）。
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn frac(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub enum Mode {
    /// 旧口径：全词表 LCG 均匀随机（对照基准）。
    Uniform,
    /// agent 真实分布：热集 + 短尾；`hot_hit` 是新 token 注入率反面（默认 0.35）。
    Agent(std::path::PathBuf, f64),
}

/// 生成 token 序列（已按 token 划分；调用方按 P1 口径展开行）。
pub fn gen_tokens(mode: &Mode, stats: Option<&AgentStats>, n_reqs: usize, cap_token: u32, seed: u64) -> Result<Vec<u32>, String> {
    let mut rng = Rng(seed);
    Ok(match mode {
        Mode::Uniform => {
            let n = n_reqs * 24; // uniform 模式：固定小批（对照）
            (0..n).map(|_| (rng.next() % VOCAB as u64) as u32).collect()
        }
        Mode::Agent(path, hot_hit) => {
            let st = match stats {
                Some(s) => s,
                None => &AgentStats::load(path)?,
            };
            let mut out = Vec::new();
            let mut hot: Vec<u32> = Vec::with_capacity(1 << 16);
            for _ in 0..n_reqs {
                let n_tok = AgentStats::draw(&st.in_tokens, rng.frac(), 0.10)
                    .min(cap_token);
                for i in 0..n_tok {
                    if !hot.is_empty() && rng.frac() < *hot_hit {
                        let j = (rng.next() % hot.len().min(4096) as u64) as usize;
                        out.push(hot[hot.len() - 1 - j]);
                    } else {
                        let t = (rng.next() % VOCAB as u64) as u32;
                        hot.push(t);
                        out.push(t);
                    }
                    let _ = i;
                }
                // 会话续接：保留 hot 尾部窗口，模拟记忆延续
            }
            out
        }
    })
}
