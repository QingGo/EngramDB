//! IO 后端抽象：同一 badge 读路径可插拔为「同步 preadv（mac/通用）」或
//! 「io_uring（Linux，M1.5）」。上层（BadgeGather）只依赖 `IoBackend`。
//!
//! 后端语义承诺（与设计 §8 一致）：
//! - `read_exact_at`：定位读并填满 buf（EOF 为 Err）；
//! - `read_at`：尽力读（返回实际字节数）。

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;

pub struct PreadvBackend;

impl IoBackend for PreadvBackend {
    fn read_exact_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
        f.read_exact_at(buf, off)
    }

    fn read_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
        f.read_at(buf, off)
    }
}

pub trait IoBackend: Send + Sync + 'static {
    fn read_exact_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<()>;
    fn read_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<usize>;
}

/// 平台默认后端。Linux 的 io_uring 真实现（常驻 ring + 有界提交）计划在
/// 具备 Linux 门禁环境的 M2 落地：本机（macOS）只能验证 preadv 语义，
/// 未经 Linux 实测的 uring 提交路径不进入主干（roadmap Phase 2 拆分记录）。
/// 实现 API 参照 io-uring 0.8：Ring::submit_and_wait + opcode::Read(offset)（TODO）。
pub fn default_backend() -> Box<dyn IoBackend> {
    Box::new(PreadvBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preadv_roundtrip() {
        let dir = std::env::temp_dir().join("engramdb-backend-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.bin");
        std::fs::write(&p, (0u8..255).collect::<Vec<u8>>()).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let b = PreadvBackend;
        let mut buf = [0u8; 4];
        b.read_exact_at(&f, &mut buf, 10).unwrap();
        assert_eq!(&buf, &[10, 11, 12, 13]);
        let mut p2 = [0u8; 2];
        let n = b.read_at(&f, &mut p2, 253).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&p2, &[253, 254]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_backend_constructs() {
        let _ = default_backend();
    }
}
