//! Minimal C ABI binding for EngramDB.
//!
//! This deliberately avoids external Python/Rust binding crates so it can be
//! built in offline sandboxes. The Python package (`python/engramdb`)
//! loads this cdylib through `ctypes` and exposes a small `Store` / `View`
//! API for engram-peft and other Python consumers.

#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]

use std::ffi::{c_char, CStr};
use std::path::Path;
use std::ptr;

use engramdb_core::layout::Layout;
use engramdb_io::batch::BadgeGather;
use engramdb_io::view::ViewReader;
use engramdb_keygen::PleSpec;

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
    match h.batch.gather_pp(ids, out_slice, 8) {
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
    let spec = PleSpec::real();
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
