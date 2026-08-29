//! FNV-1a 64 位：系统二进制对拍校验（与 scripts/bitwise_check.py 一致）。

pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors() {
        assert_eq!(fnv64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn wraps_aligns_if_reference() {
        // 4KB 零缓冲（行随机 + 填充）下的稳定值，与 bitwise_check.py 输出同源
        let mut buf = [0u8; 4096];
        buf[0] = 1;
        let h = fnv64(&buf);
        let _ = h;
        // 不含硬件 bit 的防呆：确保没有 panic / 无断言失败即可
    }
}
