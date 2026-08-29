#!/usr/bin/env python3
"""生成 keygen 对拍 golden（P0b）到 crates/engramdb-keygen/tests/golden.json。

与 Rust 侧 `matches_python_golden` 测试配套；每次变更哈希语义后重新生成并提交。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from ref_ple_hash import ple_rowids, spec_from_weights, head_vocab_sizes

TOKENS = [248044, 5, 1000, 99999, 42, 7, 248044, 1234, 8, 999999, 2048, 332, 248044, 77]


def main() -> int:
    mult = spec_from_weights("data/qwen38-ple-fp8")
    sizes = head_vocab_sizes()
    rowids = ple_rowids(np.array(TOKENS, dtype=np.int64), mult, sizes)
    out = {
        "tokens": TOKENS,
        "multipliers": mult,
        "prime_sizes": sizes,
        "rowids": [[int(x) for x in r] for r in rowids],
    }
    dst = Path(__file__).resolve().parents[1] / "crates" / "engramdb-keygen" / "tests" / "golden.json"
    dst.write_text(json.dumps(out, indent=1))
    print("golden written:", dst)


if __name__ == "__main__":
    raise SystemExit(main())
