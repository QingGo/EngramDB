//! EngramDB 的 **C ABI** —— 面向 C/C++ 的嵌入面。
//!
//! # 这个 crate 不是 Python 后端
//!
//! Python 包只使用 PyO3 扩展（`crates/engramdb-pyo3`）。曾经存在一个由
//! `python/engramdb/__init__.py` 通过 `ctypes` 加载本 cdylib 的**回退实现**，
//! 已在 roadmap §34 删除 —— 它是同一套 API 的第二份实现，**CI 从未覆盖过**，
//! 并且会在扩展导入失败时**静默降级**成子集功能而不是报错。
//!
//! 因此本 crate 现在的定位是：给 C/C++ 消费者一个**小而稳定、带版本号**的表面
//! （`engramdb_abi_version() == 1`）。它刻意不依赖任何绑定生成库，
//! 以便在离线沙箱里也能构建。
//!
//! # ⚠️ 能力边界
//!
//! 本面**只实现 `PLE_QWEN_V1`**（`ple_spec == 1`）；`ENG_DEEPSEEK_V1`
//! （DeepSeek-V4.1 Engram 规格）**尚未实现**，传入会直接报错。
//! 见 roadmap 技术债 **V55**。要让 C/C++ 面支持 V4.1，需要先做 `EngramSpec` 泛化。

#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]

use std::ffi::{c_char, CStr};
use std::path::Path;
use std::ptr;

use engramdb_core::layout::Layout;
use engramdb_io::batch::{BadgeGather, DEFAULT_GATHER_THREADS};
use engramdb_io::view::ViewReader;

pub struct StoreHandle {
    batch: BadgeGather<'static>,
    width: u64,
}

pub struct ViewHandle {
    reader: ViewReader,
}

#[no_mangle]
pub extern "C" fn engramdb_store_open(
    dir: *const c_char,
    shards: u64,
    rows_per_shard: u64,
    width: u64,
) -> *mut StoreHandle {
    if dir.is_null() {
        return ptr::null_mut();
    }
    let dir = unsafe { CStr::from_ptr(dir) }
        .to_string_lossy()
        .into_owned();
    let layout = Box::leak(Box::new(Layout::new(shards, rows_per_shard, width, 1)));
    let batch = match BadgeGather::open(Path::new(&dir), layout) {
        Ok(b) => b,
        Err(_) => return ptr::null_mut(),
    };
    Box::into_raw(Box::new(StoreHandle { batch, width }))
}

#[no_mangle]
pub unsafe extern "C" fn engramdb_store_fetch(
    handle: *mut StoreHandle,
    rowids: *const u64,
    n: usize,
    out: *mut u8,
    out_cap: usize,
) -> i32 {
    if handle.is_null() || rowids.is_null() || out.is_null() {
        return -1;
    }
    let h = &mut *handle;
    let Some(need) = n.checked_mul(h.width as usize) else {
        return -2;
    };
    if need == 0 || out_cap < need {
        return -2;
    }
    let ids = std::slice::from_raw_parts(rowids, n);
    let out_slice = std::slice::from_raw_parts_mut(out, need);
    match h.batch.gather_pp(ids, out_slice, DEFAULT_GATHER_THREADS) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub extern "C" fn engramdb_store_width(handle: *mut StoreHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).width }
}

#[no_mangle]
pub extern "C" fn engramdb_store_close(handle: *mut StoreHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[no_mangle]
pub extern "C" fn engramdb_abi_version() -> u32 {
    1
}

/// Compute PLE/Engram rowids for a token sequence.
///
/// Returns ``[len, 16]`` u64 rowids in head-major order.
/// `ple_spec`: 1 = PLE_QWEN_V1 (Qwen Flash-Next), 2 = ENG_DEEPSEEK_V1 (reserved).
#[no_mangle]
pub unsafe extern "C" fn engramdb_rowids_for_seq(
    ids: *const u32,
    len: usize,
    out: *mut u64,
    out_cap: usize,
    ple_spec: u32,
) -> i32 {
    if ids.is_null() || out.is_null() {
        return -1;
    }
    if ple_spec != 1 {
        // ENG_DEEPSEEK_V1 is not implemented in the C ABI yet.
        return -3;
    }
    let Some(need) = len.checked_mul(16) else {
        return -2;
    };
    if need == 0 || out_cap < need {
        return -2;
    }
    let tokens = std::slice::from_raw_parts(ids, len);
    let out_slice = std::slice::from_raw_parts_mut(out, need);
    // cached: see engramdb_keygen::real_spec
    let spec = engramdb_keygen::real_spec();
    let rows = spec.rowids_for_seq(tokens);
    for (i, row) in rows.iter().enumerate() {
        for (j, rid) in row.iter().enumerate() {
            out_slice[i * 16 + j] = u64::from(*rid);
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn engramdb_view_open(path: *const c_char) -> *mut ViewHandle {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .into_owned();
    let reader = match ViewReader::open(Path::new(&path)) {
        Ok(r) => r,
        Err(_) => return ptr::null_mut(),
    };
    Box::into_raw(Box::new(ViewHandle { reader }))
}

#[no_mangle]
pub unsafe extern "C" fn engramdb_view_read_record(
    handle: *mut ViewHandle,
    index: usize,
    buf: *mut u8,
    buf_cap: usize,
) -> i32 {
    if handle.is_null() || buf.is_null() {
        return -1;
    }
    let h = &*handle;
    let need = h.reader.slot_bytes() as usize;
    if buf_cap < need {
        return -2;
    }
    let out = std::slice::from_raw_parts_mut(buf, need);
    match h.reader.read_record(index, out) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn engramdb_view_read_records(
    handle: *mut ViewHandle,
    indices: *const usize,
    count: usize,
    buf: *mut u8,
    buf_cap: usize,
) -> i32 {
    if handle.is_null() || indices.is_null() || buf.is_null() {
        return -1;
    }
    let h = &*handle;
    let slot = h.reader.slot_bytes() as usize;
    let need = match count.checked_mul(slot) {
        Some(n) => n,
        None => return -2,
    };
    if buf_cap < need {
        return -2;
    }
    let idx = std::slice::from_raw_parts(indices, count);
    let out = std::slice::from_raw_parts_mut(buf, need);
    match h.reader.read_records(idx, out) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn engramdb_view_len(handle: *mut ViewHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    (*handle).reader.len()
}

#[no_mangle]
pub unsafe extern "C" fn engramdb_view_slot_bytes(handle: *mut ViewHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }
    (*handle).reader.slot_bytes()
}

#[no_mangle]
pub extern "C" fn engramdb_view_close(handle: *mut ViewHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
