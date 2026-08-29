//! CLI 端到端集成测试（真实进程调用，bit-exact 门禁）。
//!
//! 链路：mock raw 表（1 shard，完整 badge 尺寸）→ `build` → `gather`/`verify`
//! 与本地 fnv64 对拍（bit-exact）→ `warm` → `bench-real`（agent 真实分布）
//! → `index`（I2 索引产物）。所有命令对临时表执行，无外部数据依赖。

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bins() -> &'static str {
    env!("CARGO_BIN_EXE_engramdb")
}

fn run(args: &[&str], cwd: &Path, stdin_data: Option<&[u8]>) -> Output {
    use std::io::Write;
    let mut c = Command::new(bins());
    c.args(args).current_dir(cwd);
    let mut child = c
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn engramdb");
    if let Some(data) = stdin_data {
        child.stdin.as_mut().unwrap().write_all(data).unwrap();
    }
    drop(child.stdin.take());
    child.wait_with_output().expect("wait")
}

struct Temp(PathBuf);
impl Temp {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "engramdb-cli-e2e-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        Temp(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn row_value(i: u64) -> u8 {
    ((i % 110) + 1) as u8
}

fn write_shard_dir(dir: &Path) {
    // 1 分片 × 110 行（3 badge：两满 + 尾长 10 行 pad 0）；行 i 值 = row_value(i)
    let mut data = vec![0u8; 110 * 160];
    for i in 0..110u64 {
        for c in 0..160 {
            data[i as usize * 160 + c] = row_value(i);
        }
    }
    std::fs::write(dir.join("shard_000.bin"), &data).unwrap();
}

fn stdout(s: &Output) -> String {
    String::from_utf8_lossy(&s.stdout).into_owned()
}

fn expected_fnv(rowids: &[u64]) -> u64 {
    // 与 gather 同口径：行值 = row_value(r)；r>=110（超出 badge 文件）→ 全 0 行
    let mut buf = vec![0u8; rowids.len() * 160];
    for (i, &r) in rowids.iter().enumerate() {
        if r < 110 {
            let v = row_value(r);
            for c in 0..160 {
                buf[i * 160 + c] = v;
            }
        }
    }
    engramdb_core::fnv64(&buf)
}

#[test]
fn build_gather_verify_bench_chain() {
    let tmp = Temp::new("chain");
    let raw = tmp.0.join("raw");
    let out = tmp.0.join("tbl");
    std::fs::create_dir_all(&raw).unwrap();
    write_shard_dir(&raw);
    // build: 需要 128 分片目录（CLI 按 128 shards 扫描）→ 生成 128 个同构分片文件
    for i in 0..128 {
        let d = raw.join(format!("shard_{:03}.bin", i));
        if i > 0 {
            std::fs::copy(raw.join("shard_000.bin"), d).unwrap();
        }
    }

    let o = run(&["build", raw.to_str().unwrap(), out.to_str().unwrap()], &tmp.0, None);
    assert!(o.status.success(), "build: {}", stdout(&o));
    assert!(stdout(&o).contains("built"), "build stdout");
    let b = out.join("badge_000.bin");
    assert!(b.exists(), "badge_000.bin 缺失");
    assert!(out.join("manifest.json").exists(), "manifest.json 缺失");

    // gather：0..10 + 跨 badge 行（行号=rowid，25 行/badge 满表）
    let mut rowids: Vec<u64> = (0..10).collect();
    rowids.extend([10, 11, 24, 30, 77]); // 24=边界, 30,77>25（超 badge 无行 → 视为下一 badge）
    let stdin: String = rowids.iter().map(|r| format!("{r}\n")).collect();
    let o = run(&["gather", out.to_str().unwrap()], &tmp.0, Some(stdin.as_bytes()));
    assert!(o.status.success(), "gather: {}", stdout(&o));
    let got: u64 = stdout(&o).trim().parse().unwrap();
    let exp = expected_fnv(&rowids);
    assert_eq!(got, exp, "gather fnv 与本地 fnv64 对拍失败");

    // verify（文件输入）
    let rowid_file = tmp.0.join("rowids.txt");
    std::fs::write(&rowid_file, stdin.clone()).unwrap();
    let o = run(&["verify", out.to_str().unwrap(), rowid_file.to_str().unwrap()], &tmp.0, None);
    assert!(o.status.success(), "verify: {}", stdout(&o));
    assert!(stdout(&o).contains(&format!("fnv={exp}")), "verify fnv 不匹配");

    // warm（tiny budget 冒烟）
    let o = run(&["warm", out.to_str().unwrap(), "--budget", "0.000000001"], &tmp.0, None);
    assert!(o.status.success(), "warm: {}", stdout(&o));
    assert!(stdout(&o).contains("warmed"), "warm stdout");

    // bench-real agent 分布（真实 stats 文件路径从仓库 root 相对——集成测试 cwd=tmp，
    // 需要把 probes 路径绝对化：取仓库根。这里用 CARGO_MANIFEST_DIR 上一层/../.. 解出）
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let stats = repo.join("probes/agent_workload_stats.json");
    if stats.exists() {
        let o = run(
            &[
                "bench-real",
                out.to_str().unwrap(),
                "--dist",
                "agent",
                "--stats",
                stats.to_str().unwrap(),
                "--reqs",
                "4",
                "--cap-token",
                "500",
                "--iters",
                "1",
            ],
            &tmp.0,
            None,
        );
        assert!(o.status.success(), "bench-real: {}", stdout(&o));
        assert!(stdout(&o).contains("rows/s="), "bench-real stdout");
    }

    // index：u64 LE 流
    let rb = tmp.0.join("rowids.bin");
    let mut buf = Vec::new();
    for r in &rowids {
        buf.extend_from_slice(&r.to_le_bytes());
    }
    std::fs::write(&rb, &buf).unwrap();
    let idx = tmp.0.join("index");
    let o = run(&["index", rb.to_str().unwrap(), idx.to_str().unwrap()], &tmp.0, None);
    assert!(o.status.success(), "index: {}", stdout(&o));
    assert!(idx.join("counts.bin").exists(), "counts.bin 缺失");
    let dump = std::fs::read_to_string(idx.join("counts.dump.txt")).unwrap();
    assert!(dump.lines().count() >= rowids.len() - 1, "dump 行数异常");
}
