//! PyO3 bindings for EngramDB.
//!
//! This is the native extension face of the Python package.  It exposes the
//! same minimal Store/View operations as the ctypes C-ABI bridge, but as a
//! real Python extension module (the long-term replacement for ctypes).

use std::path::Path;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;
use engramdb_io::view::{self, ViewReader};
use engramdb_keygen::PleSpec;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(unsendable)]
struct Store {
    batch: BadgeGather<'static>,
    rows_per_shard: u64,
    shards: u64,
    width: u64,
}

#[pymethods]
impl Store {
    #[new]
    #[pyo3(signature = (dir, shards, rows_per_shard, width))]
    fn new(dir: &str, shards: u64, rows_per_shard: u64, width: u64) -> PyResult<Self> {
        let layout = Box::leak(Box::new(Layout::new(shards, rows_per_shard, width, 1)));
        let batch = BadgeGather::open(Path::new(dir), layout)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self {
            batch,
            rows_per_shard,
            shards,
            width,
        })
    }

    fn fetch<'py>(&self, py: Python<'py>, rowids: Vec<u64>) -> PyResult<Bound<'py, PyBytes>> {
        if rowids.is_empty() {
            return Ok(PyBytes::new(py, &[]));
        }
        let mut out = vec![0u8; rowids.len() * self.width as usize];
        self.batch
            .gather_pp(&rowids, &mut out, 8)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &out))
    }

    fn fetch_one<'py>(&self, py: Python<'py>, rowid: u64) -> PyResult<Bound<'py, PyBytes>> {
        self.fetch(py, vec![rowid])
    }

    /// Compatibility no-op for the ctypes API (resources are dropped with the object).
    fn close(&self) {}

    #[getter]
    fn width(&self) -> u64 {
        self.width
    }

    #[getter]
    fn total_rows(&self) -> u64 {
        self.shards * self.rows_per_shard
    }
}

#[pyclass(unsendable)]
struct View {
    reader: ViewReader,
}

#[pymethods]
impl View {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let reader = ViewReader::open(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { reader })
    }

    fn len(&self) -> usize {
        self.reader.len()
    }

    fn slot_bytes(&self) -> u64 {
        self.reader.slot_bytes()
    }

    /// Compatibility no-op for the ctypes API (resources are dropped with the object).
    fn close(&self) {}

    fn read_record<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyBytes>> {
        let mut buf = vec![0u8; self.reader.slot_bytes() as usize];
        self.reader
            .read_record(index, &mut buf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &buf))
    }

    fn read_records<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let slot = self.reader.slot_bytes() as usize;
        let mut out = vec![0u8; indices.len() * slot];
        self.reader
            .read_records(&indices, &mut out)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &out))
    }
}

/// Page reader compatible with the shape of SGLang's `IoUringReader.read_pages`.
///
/// This is a pread-based implementation (Unix); it is a useful integration point
/// for engines that already have raw file descriptors and offset lists.
#[cfg(unix)]
#[pyclass(unsendable)]
struct PageReader {
    page_size: usize,
}

#[cfg(unix)]
#[pymethods]
impl PageReader {
    #[new]
    #[pyo3(signature = (page_size=4096))]
    fn new(page_size: usize) -> PyResult<Self> {
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "page_size must be a positive power of two",
            ));
        }
        Ok(Self { page_size })
    }

    fn read_pages<'py>(
        &self,
        py: Python<'py>,
        file_descriptors: Vec<i32>,
        offsets: Vec<u64>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        if file_descriptors.len() != offsets.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "file_descriptors and offsets must have the same length",
            ));
        }
        let page_size = self.page_size;
        let mut pages = Vec::with_capacity(file_descriptors.len());
        for (fd, offset) in file_descriptors.into_iter().zip(offsets) {
            let mut buf = vec![0u8; page_size];
            // SAFETY: buf is valid for writes of page_size bytes; pread does not
            // modify the file offset.
            let n = unsafe {
                libc::pread(
                    fd,
                    buf.as_mut_ptr().cast(),
                    buf.len(),
                    offset as libc::off_t,
                )
            };
            if n < 0 {
                return Err(pyo3::exceptions::PyOSError::new_err(
                    std::io::Error::last_os_error().to_string(),
                ));
            }
            let n = n as usize;
            if n == 0 {
                return Err(pyo3::exceptions::PyOSError::new_err(
                    "EOF while reading page",
                ));
            }
            buf.truncate(n);
            pages.push(PyBytes::new(py, &buf).unbind());
        }
        Ok(pages)
    }
}

/// Linux io_uring-backed page reader with the same Python API shape as
/// SGLang's `IoUringReader.read_pages(fds, offsets)`.
///
/// This batches all requests in one ring submission per call (up to 256 at a
/// time), which is the intended path for Linux inference hosts.  The ring is
/// kept thread-local so repeated calls reuse the same io_uring instance.
#[cfg(target_os = "linux")]
#[pyclass]
struct IoUringPageReader {
    page_size: usize,
}

#[cfg(target_os = "linux")]
#[pymethods]
impl IoUringPageReader {
    #[new]
    #[pyo3(signature = (page_size=4096))]
    fn new(page_size: usize) -> PyResult<Self> {
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "page_size must be a positive power of two",
            ));
        }
        Ok(Self { page_size })
    }

    fn read_pages<'py>(
        &self,
        py: Python<'py>,
        file_descriptors: Vec<i32>,
        offsets: Vec<u64>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        if file_descriptors.len() != offsets.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "file_descriptors and offsets must have the same length",
            ));
        }

        const DEPTH: u32 = 256;
        let page_size = self.page_size;
        let mut pages: Vec<Vec<u8>> = (0..file_descriptors.len())
            .map(|_| vec![0u8; page_size])
            .collect();

        IO_URING_PAGE_READER.with(|sl| -> PyResult<()> {
            let mut r = sl.borrow_mut();
            if r.is_none() {
                let ring = io_uring::IoUring::new(DEPTH).map_err(|e| {
                    pyo3::exceptions::PyOSError::new_err(format!("IoUring::new: {e}"))
                })?;
                *r = Some(ring);
            }
            let ring = r.as_mut().unwrap();

            let n = file_descriptors.len();
            let mut start = 0usize;
            while start < n {
                let end = (start + DEPTH as usize).min(n);
                let count = end - start;

                for i in start..end {
                    let sqe = io_uring::opcode::Read::new(
                        io_uring::types::Fd(file_descriptors[i]),
                        pages[i].as_mut_ptr(),
                        pages[i].len() as u32,
                    )
                    .offset(offsets[i])
                    .build()
                    .user_data((i - start) as u64);
                    unsafe {
                        ring.submission().push(&sqe).map_err(|e| {
                            pyo3::exceptions::PyOSError::new_err(format!("SQ push: {e}"))
                        })?;
                    }
                }

                ring.submit_and_wait(count).map_err(|e| {
                    pyo3::exceptions::PyOSError::new_err(format!("submit_and_wait: {e}"))
                })?;

                for _ in 0..count {
                    let cqe = ring.completion().next().ok_or_else(|| {
                        pyo3::exceptions::PyOSError::new_err("no cqe for io_uring read")
                    })?;
                    let res = cqe.result();
                    let idx = cqe.user_data() as usize;
                    let page_idx = start + idx;
                    if res < 0 {
                        return Err(pyo3::exceptions::PyOSError::new_err(
                            std::io::Error::from_raw_os_error(-res).to_string(),
                        ));
                    }
                    if res == 0 {
                        return Err(pyo3::exceptions::PyOSError::new_err(
                            "EOF while reading page",
                        ));
                    }
                    pages[page_idx].truncate(res as usize);
                }

                start = end;
            }
            Ok(())
        })?;

        Ok(pages
            .into_iter()
            .map(|p| PyBytes::new(py, &p).unbind())
            .collect())
    }
}

#[cfg(target_os = "linux")]
thread_local! {
    static IO_URING_PAGE_READER: std::cell::RefCell<Option<io_uring::IoUring>> =
        const { std::cell::RefCell::new(None) };
}

#[pyfunction]
fn read_keys(path: &str) -> PyResult<Vec<u64>> {
    view::read_keys(Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

#[pyfunction]
fn abi_version() -> u32 {
    1
}

#[pyfunction]
fn rowids_for_seq(tokens: Vec<u32>, ple_spec: u32) -> PyResult<Vec<Vec<u32>>> {
    if ple_spec != 1 {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "only PLE_QWEN_V1=1 is implemented",
        ));
    }
    let spec = PleSpec::real();
    Ok(spec
        .rowids_for_seq(&tokens)
        .into_iter()
        .map(|row| row.to_vec())
        .collect())
}

#[pymodule]
fn _engramdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Store>()?;
    m.add_class::<View>()?;
    #[cfg(unix)]
    m.add_class::<PageReader>()?;
    #[cfg(target_os = "linux")]
    m.add_class::<IoUringPageReader>()?;
    m.add_function(wrap_pyfunction!(read_keys, m)?)?;
    m.add_function(wrap_pyfunction!(abi_version, m)?)?;
    m.add_function(wrap_pyfunction!(rowids_for_seq, m)?)?;
    Ok(())
}
