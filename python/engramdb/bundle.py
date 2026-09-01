"""EngramDB bundle manifest helpers.

A bundle is a small JSON document that describes one deployed PLE artifact:

* which storage to use (Store-I rows or Store-P view + slot index),
* the PLE rowid/scale metadata,
* optional target-reader entries.

This module intentionally does not open a reader or load any model.  It only
parses/validates/resolves paths; the optional serving layer can consume the
resolved manifest without importing torch/vLLM/SGLang.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "engramdb-bundle-v1"


@dataclass
class BundleManifest:
    """A parsed EngramDB bundle manifest.

    ``base_dir`` is used to resolve relative paths in :meth:`resolve` and
    :meth:`open_memory`.
    """

    data: dict[str, Any]
    base_dir: str | Path | None = None
    path: str | Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "BundleManifest":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data, base_dir=path.parent, path=path)

    @property
    def schema(self) -> str | None:
        return self.data.get("schema")

    @property
    def id(self) -> str | None:
        return self.data.get("id")

    @property
    def memory(self) -> dict[str, Any]:
        return self.data.get("memory", {})

    @property
    def ple(self) -> dict[str, Any]:
        return self.data.get("ple", {})

    @property
    def readers(self) -> list[dict[str, Any]]:
        return self.data.get("readers", [])

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def resolve_path(self, rel: str | Path | None) -> Path | None:
        """Resolve a manifest-relative path against ``base_dir``."""
        if rel is None:
            return None
        p = Path(rel)
        if p.is_absolute() or self.base_dir is None:
            return p
        return Path(self.base_dir) / p

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty means valid)."""
        errors: list[str] = []
        if not self.schema:
            errors.append("missing schema")
        elif not str(self.schema).startswith("engramdb-bundle-"):
            errors.append(f"unsupported schema: {self.schema}")
        mem = self.memory
        mtype = mem.get("type")
        if mtype not in ("store", "view"):
            errors.append("memory.type must be 'store' or 'view'")
        elif mtype == "store":
            store = mem.get("store", {})
            for key in ("path", "shards", "rows_per_shard", "width"):
                if key not in store:
                    errors.append(f"memory.store missing {key}")
        elif mtype == "view":
            view = mem.get("view", {})
            if "path" not in view:
                errors.append("memory.view missing path")
            slot = mem.get("slot_index")
            if slot is not None and "path" not in slot:
                errors.append("memory.slot_index missing path")
            if slot is None and not (
                mem.get("sequential_view") or mem.get("sequential")
            ):
                errors.append(
                    "memory.view requires slot_index or sequential_view=true"
                )
        ple = self.ple
        if "ple_embed_dim" not in ple:
            errors.append("ple missing ple_embed_dim")
        if "num_heads" not in ple and "ngram_size" not in ple:
            errors.append("ple missing num_heads or ngram_size")
        return errors

    def resolved(self) -> dict[str, Any]:
        """Return a copy with paths resolved to absolute :class:`Path` objects."""
        out = json.loads(json.dumps(self.data))
        mem = out.get("memory", {})
        if mem.get("type") == "store" and "store" in mem:
            store_path = self.resolve_path(mem["store"].get("path"))
            if store_path is not None:
                mem["store"]["path"] = str(store_path)
        elif mem.get("type") == "view" and "view" in mem:
            view_path = self.resolve_path(mem["view"].get("path"))
            if view_path is not None:
                mem["view"]["path"] = str(view_path)
            slot = mem.get("slot_index")
            if slot is not None:
                slot_path = self.resolve_path(slot.get("path"))
                if slot_path is not None:
                    slot["path"] = str(slot_path)
        for reader in out.get("readers", []):
            rp = self.resolve_path(reader.get("path"))
            if rp is not None:
                reader["path"] = str(rp)
        return out

    def open_memory(self) -> Any:
        """Open the storage described by this manifest as a :class:`PleMemory`.

        The returned ``PleMemory`` does not own the underlying Store/View, so
        callers should keep references if they need to close them explicitly.
        """
        from . import Store, View
        from .ple_memory import PleMemory

        errors = self.validate()
        if errors:
            raise ValueError("invalid bundle manifest: " + "; ".join(errors))

        mem_cfg = self.memory
        ple_cfg = self.ple
        ngram_size = int(ple_cfg.get("ngram_size") or 3)
        heads_per_ngram = int(ple_cfg.get("heads_per_ngram") or 8)
        num_heads = int(
            ple_cfg.get("num_heads")
            or (ngram_size - 1) * heads_per_ngram
        )
        head_dim = int(
            ple_cfg.get("head_dim")
            or int(ple_cfg.get("ple_embed_dim")) // num_heads
        )
        kwargs = {
            "head_dim": head_dim,
            "num_heads": num_heads,
            "ngram_size": ngram_size,
            "heads_per_ngram": heads_per_ngram,
            "scale": float(ple_cfg.get("scale") or 1.0),
            "eos": int(ple_cfg.get("eos") or 248044),
            "multipliers": ple_cfg.get("multipliers"),
            "prime_sizes": ple_cfg.get("prime_sizes"),
            "offsets": ple_cfg.get("offsets"),
            "sequential_view": bool(
                mem_cfg.get("sequential_view") or mem_cfg.get("sequential")
            ),
            "start_slot": int(mem_cfg.get("start_slot") or 0),
        }

        if mem_cfg.get("type") == "store":
            store_cfg = mem_cfg["store"]
            store = Store(
                str(self.resolve_path(store_cfg["path"])),
                int(store_cfg["shards"]),
                int(store_cfg["rows_per_shard"]),
                int(store_cfg["width"]),
            )
            return PleMemory(store=store, **kwargs)

        view_cfg = mem_cfg["view"]
        view = View(str(self.resolve_path(view_cfg["path"])))
        slot_index = None
        slot_cfg = mem_cfg.get("slot_index")
        if slot_cfg:
            slot_path = self.resolve_path(slot_cfg["path"])
            slot_type = str(slot_cfg.get("type") or "disk")
            if slot_type == "memory" or (slot_path is not None and str(slot_path).endswith(".npz")):
                from .slot_index import SlotIndex
                slot_index = SlotIndex.load(str(slot_path))
            else:
                from .disk_slot_index import DiskSlotIndex

                slot_index = DiskSlotIndex(str(slot_path))
        return PleMemory(view=view, slot_index=slot_index, **kwargs)


def bundle_manifest_from_path(path: str | Path) -> BundleManifest:
    """Alias for :meth:`BundleManifest.load`."""
    return BundleManifest.load(path)


__all__ = ["BundleManifest", "bundle_manifest_from_path", "BUNDLE_SCHEMA"]
