#!/usr/bin/env python3
"""Install (or remove) the SGLang-side hook that makes the SC4 probe reachable.

Why a hook and not an env var
-----------------------------
SGLang starts its scheduler with ``mp.set_start_method("spawn")``, so a
monkeypatch applied in the parent process is **not** inherited by the process
that actually runs the model.  ``scripts/serve_sglang_baseline.py`` recorded this
as the reason a disk arm was out of reach on SGLang.

The way through is to hang the patch off a module the child imports anyway.
Appending a few lines to the installed ``sglang/srt/models/qwen3_5.py`` runs in
every process that imports the model, on every start method, with no
``sitecustomize``/``PYTHONPATH`` dependency.  Verified end to end: the capture
path for ``--cuda-graph-backend-decode=breakable`` fails without a patch applied
this way and succeeds with one -- and capture happens in the scheduler child.

Usage::

    python scripts/qwen3_5_sc4_hook.py install     # append the loader (idempotent)
    python scripts/qwen3_5_sc4_hook.py uninstall   # restore from .orig
    python scripts/qwen3_5_sc4_hook.py status

The appended text is inert unless ``ENGRAMDB_SC4_ARM`` is set, so leaving it
installed is harmless.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MARKER = "# --- engramdb sc4 hook (see scripts/engramdb_sc4_inject.py) ---"

LOADER = f'''


{MARKER}
def _engramdb_install_sc4_inject():
    import os as _os
    import sys as _sys

    if not _os.environ.get("ENGRAMDB_SC4_ARM"):
        return
    _here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for _cand in ("/root/engramdb/scripts", _here):
        if _os.path.isdir(_cand) and _cand not in _sys.path:
            _sys.path.insert(0, _cand)
    import engramdb_sc4_inject as _inj

    _inj.install()


_engramdb_install_sc4_inject()
'''


def target_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    import sglang

    return Path(sglang.__file__).parent / "srt" / "models" / "qwen3_5.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("install", "uninstall", "status"))
    ap.add_argument("--path", default=None, help="override qwen3_5.py location")
    args = ap.parse_args()

    p = target_path(args.path)
    if not p.exists():
        print(f"not found: {p}")
        return 2
    orig = p.with_suffix(p.suffix + ".orig")
    text = p.read_text()

    if args.action == "status":
        print(f"path       : {p}")
        print(f"backup     : {orig} {'(present)' if orig.exists() else '(absent)'}")
        print(f"hook       : {'INSTALLED' if MARKER in text else 'absent'}")
        return 0

    if args.action == "install":
        if MARKER in text:
            print("hook already installed")
            return 0
        if not orig.exists():
            shutil.copy2(p, orig)
            print(f"backup written: {orig}")
        p.write_text(text + LOADER)
        print(f"hook appended to {p}")
        r = subprocess.run([sys.executable, "-c", f"import ast;ast.parse(open({str(p)!r}).read())"])
        print("syntax:", "OK" if r.returncode == 0 else "BROKEN")
        return r.returncode

    if args.action == "uninstall":
        if not orig.exists():
            print(f"no backup at {orig}; nothing to restore")
            return 1
        shutil.copy2(orig, p)
        print(f"restored {p} from {orig}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
