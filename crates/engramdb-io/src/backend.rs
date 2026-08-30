//! IO 后端抽象：同一 badge 读路径可插拔为「同步 preadv（mac/通用）」或
//! 「io_uring（Linux，M1.5）」。上层（BadgeGather）只依赖 `IoBackend`。
//!
//! 后端语义承诺（与设计 §8 一致）：
//! - `read_exact_at`：定位读并填满 buf（EOF 为 Err）；
//! - `read_at`：尽力读（返回实际字节数）。

use std::fs::File;
use std::io;

/// 跨平台定位读：unix=pread（`read_at`），windows=OVERLAPPED-free 的 `seek_read`
/// （逻辑等价 pread：设置文件指针后读回塞回原指针语义由 OS 处理）。
pub fn platform_read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    platform_read_at_impl(f, buf, off)
}

#[cfg(unix)]
fn platform_read_at_impl(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, off)
}

#[cfg(windows)]
fn platform_read_at_impl(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, off)
}

#[cfg(not(any(unix, windows)))]
fn platform_read_at_impl(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    let mut f2 = f.try_clone()?;
    use std::io::{Read, Seek, SeekFrom};
    f2.seek(SeekFrom::Start(off))?;
    f2.read(buf)
}

/// 跨平台"填满读"（EOF 报错），供非 trait 调用方复用。
pub fn platform_read_exact_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = platform_read_at(f, &mut buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read_exact_at: eof inside buffer",
            ));
        }
        done += n;
    }
    Ok(())
}

/// 标准库定位读后端（unix=pread / windows=seek_read，语义等同；名随历史保留 preadv）。
pub struct PreadvBackend;

impl IoBackend for PreadvBackend {
    fn read_exact_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
        platform_read_exact_at(f, buf, off)
    }

    fn read_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
        platform_read_at(f, buf, off)
    }
}

pub trait IoBackend: Send + Sync + 'static {
    fn read_exact_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<()>;
    fn read_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<usize>;

    /// 批量定位读（默认逐条回退——逐条语义等价，覆盖者可批式提交）。
    /// 每个 req 必须以 `read_exact_at` 语义填满（长度不足报错）。
    fn read_many(&self, f: &File, reqs: &mut [(u64, &mut [u8])]) -> io::Result<()> {
        for (off, buf) in reqs.iter_mut() {
            self.read_exact_at(f, buf, *off)?;
        }
        Ok(())
    }
}

/// 平台默认后端。**保持 preadv**（可插拔实验：显式传 UringBackend）。
/// Linux 的 io_uring 真实现（常驻 ring + 有界提交）M2 落地：UringBackend 提供
/// per-call 提交语义（正确性/平台能力验证）；批量提交面（batch API）留后端演进。
pub fn default_backend() -> Box<dyn IoBackend> {
    Box::new(PreadvBackend)
}

// ---------- Linux: io_uring（M2）----------

/// io_uring 后端：每线程一个常驻 ring（IoUring 非 Send），每次 read 提交并等待 1 个
/// 完成。语义与 preadv 一致（尽力读/填满读），复用 trait 无需上层改动。
/// 注意：per-call 提交在并行度足够（8t 各自 ring）时仍有优势（并法输入复制、多路竞争
/// 展开），完全批量化（一次 submit N 个 SQE）留 batch API 演进（M1.5+）。
#[cfg(target_os = "linux")]
pub struct UringBackend;

/// 批量 io_uring 后端：read_many = 一次 submit N SQE + 整体 wait（逐请求 user_data 索回）。
/// M2 的"正路"实现——per-call UringBackend 已被 benchmark 判为无增益（-17%）。
#[cfg(target_os = "linux")]
pub struct UringBatchBackend;

#[cfg(target_os = "linux")]
impl IoBackend for UringBatchBackend {
    fn read_exact_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
        platform_read_exact_at(f, buf, off)
    }
    fn read_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
        platform_read_at(f, buf, off)
    }
    fn read_many(&self, f: &File, reqs: &mut [(u64, &mut [u8])]) -> io::Result<()> {
        let _ = uring_batch_read(f, reqs)?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn uring_batch_read(f: &File, reqs: &mut [(u64, &mut [u8])]) -> io::Result<usize> {
    const DEPTH: u32 = 256;
    use std::os::fd::AsRawFd;
    URING.with(|sl| {
        let mut r = sl.borrow_mut();
        if r.is_none() {
            let ring = io_uring::IoUring::new(DEPTH)
                .map_err(|e| io::Error::other(format!("IoUring::new: {e}")))?;
            *r = Some(ring);
        }
        let ring = r.as_mut().unwrap();
        let fd = f.as_raw_fd();
        let mut total = 0usize;
        for chunk in reqs.chunks_mut(DEPTH as usize) {
            let expect: Vec<(usize, usize)> = chunk
                .iter()
                .enumerate()
                .map(|(i, (_, b))| (i, b.len()))
                .collect();
            for (i, (off, buf)) in chunk.iter_mut().enumerate() {
                let sqe = io_uring::opcode::Read::new(
                    io_uring::types::Fd(fd),
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                )
                .offset(*off);
                let sqe = sqe.build().user_data(i as u64);
                unsafe {
                    ring.submission()
                        .push(&sqe)
                        .map_err(|e| io::Error::other(format!("SQ push: {e}")))?;
                }
            }
            let n = chunk.len() as u32;
            ring.submit_and_wait(n)
                .map_err(|e| io::Error::other(format!("submit_and_wait: {e}")))?;
            for _ in 0..n {
                let cqe = ring
                    .completion()
                    .next()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "no cqe"))?;
                let res = cqe.result();
                let idx = cqe.user_data() as usize;
                let elen = expect.get(idx).map(|x| x.1).unwrap_or(0);
                if res < 0 {
                    return Err(io::Error::from_raw_os_error(-res));
                }
                if (res as usize) != elen {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("read_many: short read {res} < {elen}"),
                    ));
                }
                total += res as usize;
            }
        }
        Ok(total)
    })
}

#[cfg(target_os = "linux")]
thread_local! {
    static URING: std::cell::RefCell<Option<io_uring::IoUring>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(target_os = "linux")]
fn submit_uring_read(f: &std::fs::File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    use std::os::fd::AsRawFd;
    URING.with(|sl| {
        let mut r = sl.borrow_mut();
        if r.is_none() {
            let ring = io_uring::IoUring::new(256)
                .map_err(|e| io::Error::other(format!("IoUring::new: {e}")))?;
            *r = Some(ring);
        }
        let ring = r.as_mut().unwrap();
        let fd = f.as_raw_fd();
        let sqe = io_uring::opcode::Read::new(
            io_uring::types::Fd(fd),
            buf.as_mut_ptr(),
            buf.len() as u32,
        )
        .offset(off);
        let sqe = sqe.build().user_data(0);
        unsafe {
            ring.submission()
                .push(&sqe)
                .map_err(|e| io::Error::other(format!("SQ push: {e}")))?;
        }
        ring.submit_and_wait(1)
            .map_err(|e| io::Error::other(format!("submit_and_wait: {e}")))?;
        let cqe = ring
            .completion()
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "no cqe"))?;
        let res = cqe.result();
        if res < 0 {
            return Err(io::Error::from_raw_os_error(-res));
        }
        Ok(res as usize)
    })
}

#[cfg(target_os = "linux")]
impl IoBackend for UringBackend {
    fn read_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
        submit_uring_read(f, buf, off)
    }

    fn read_exact_at(&self, f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
        let mut done = 0usize;
        while done < buf.len() {
            let n = self.read_at(f, &mut buf[done..], off + done as u64)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "read_exact_at: eof inside buffer",
                ));
            }
            done += n;
        }
        Ok(())
    }
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

    #[cfg(target_os = "linux")]
    #[test]
    fn uring_roundtrip_and_semantics() {
        let dir = std::env::temp_dir().join("engramdb-uring-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.bin");
        std::fs::write(&p, (0usize..512).map(|i| i as u8).collect::<Vec<u8>>()).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let b = UringBackend;
        let mut buf = [0u8; 4];
        b.read_exact_at(&f, &mut buf, 10).unwrap();
        assert_eq!(&buf, &[10, 11, 12, 13]);
        // 跨两次提交偏移正确
        let mut p2 = [0u8; 2];
        let n = b.read_at(&f, &mut p2, 253).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&p2, &[253, 254]);
        // EOF 语义：read_at 返回 0？（尝试超出文件尾）
        let mut p3 = [0u8; 4];
        let n3 = b.read_at(&f, &mut p3, 512).unwrap();
        assert_eq!(n3, 0);
        // read_exact_at EOF -> Err
        let r = b.read_exact_at(&f, &mut p3, 511);
        assert!(r.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
