//! PyO3 bindings for EngramDB.
//!
//! This is the native extension face of the Python package.  It exposes the
//! same minimal Store/View operations as the ctypes C-ABI bridge, but as a
//! real Python extension module (the long-term replacement for ctypes).

use std::path::Path;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;
use engramdb_io::view::{self, ViewReader};
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

#[pyfunction]
fn read_keys(path: &str) -> PyResult<Vec<u64>> {
    view::read_keys(Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

#[pymodule]
fn _engramdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Store>()?;
    m.add_class::<View>()?;
    m.add_function(wrap_pyfunction!(read_keys, m)?)?;
    Ok(())
}
