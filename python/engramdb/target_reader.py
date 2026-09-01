"""Generic target-reader registry for EngramDB bundles.

This is intentionally not a specific reader implementation.  EngramDB defines
only the *loading protocol*:

* a target reader is referenced by a stable name and version,
* a registry maps that name to a factory,
* a bundle manifest entry supplies a path/options for deployment.

Actual Qwen PLE reader/model code belongs to the qwen35-ple project; this
module only gives engines and bundles a uniform way to discover and construct
readers without importing any specific framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .bundle import BundleManifest

ReaderFactory = Callable[..., Any]


@dataclass
class ReaderSpec:
    """Resolved description of one target reader in a bundle."""

    name: str
    version: str = "1"
    type: str = "generic"
    path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str | Path | None = None) -> "ReaderSpec":
        path = data.get("path")
        if path is not None and base_dir is not None:
            p = Path(path)
            if not p.is_absolute():
                path = str(Path(base_dir) / p)
        return cls(
            name=str(data.get("name") or ""),
            version=str(data.get("version") or "1"),
            type=str(data.get("type") or "generic"),
            path=path,
            options=dict(data.get("options") or {}),
        )


class TargetReaderRegistry:
    """Registry for named target-reader factories.

    Example::

        registry = TargetReaderRegistry()
        registry.register("my-reader", factory)
        reader = registry.create("my-reader", path="/checkpoint", device="cpu")
    """

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], ReaderFactory] = {}
        self._latest: dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: ReaderFactory | None = None,
        *,
        version: str = "1",
        override: bool = False,
    ) -> ReaderFactory:
        """Register a factory under ``name``.

        Can be used as a decorator::

            @registry.register("demo")
            def build_demo(path, **kw):
                return ...
        """
        name = str(name)
        version = str(version)
        key = (name, version)

        def _register(fn: ReaderFactory) -> ReaderFactory:
            if key in self._factories and not override:
                raise ValueError(
                    f"reader {name!r} version {version!r} is already registered"
                )
            self._factories[key] = fn
            self._latest[name] = version
            return fn

        if factory is not None:
            return _register(factory)
        return _register  # type: ignore[return-value]

    def unregister(self, name: str, version: str | None = None) -> None:
        """Remove one or all versions of a reader name."""
        if version is None:
            for key in [k for k in self._factories if k[0] == str(name)]:
                del self._factories[key]
            if self._latest.get(str(name)) is not None:
                del self._latest[str(name)]
            return
        key = (str(name), str(version))
        self._factories.pop(key, None)
        if self._latest.get(str(name)) == str(version):
            remaining = [k[1] for k in self._factories if k[0] == str(name)]
            if remaining:
                self._latest[str(name)] = remaining[-1]
            else:
                self._latest.pop(str(name), None)

    def available(self) -> list[dict[str, str]]:
        """Return registered (name, latest version) pairs."""
        return [
            {"name": name, "version": version}
            for name, version in sorted(self._latest.items())
        ]

    def has(self, name: str, version: str | None = None) -> bool:
        name = str(name)
        if version is None:
            return name in self._latest
        return (name, str(version)) in self._factories

    def create(
        self,
        name: str,
        *,
        version: str | None = None,
        path: str | Path | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Construct a reader from the registry.

        ``options`` and keyword arguments are forwarded to the factory.
        """
        name = str(name)
        if version is None:
            version = self._latest.get(name)
        if version is None:
            raise KeyError(f"no target reader registered: {name!r}")
        factory = self._factories.get((name, str(version)))
        if factory is None:
            raise KeyError(f"no target reader registered: {name!r} v{version}")
        merged = dict(options or {})
        merged.update(kwargs)
        if path is not None:
            merged.setdefault("path", str(path))
        return factory(**merged)

    def create_from_spec(self, spec: ReaderSpec) -> Any:
        """Construct a reader from a :class:`ReaderSpec`."""
        return self.create(
            spec.name,
            version=spec.version,
            path=spec.path,
            options=spec.options,
        )

    def create_from_manifest(
        self,
        manifest: BundleManifest | dict[str, Any],
        *,
        index: int = 0,
    ) -> Any:
        """Construct the ``index``-th reader declared in a bundle manifest."""
        if isinstance(manifest, BundleManifest):
            base_dir = manifest.base_dir
            readers = manifest.readers
        else:
            base_dir = None
            readers = manifest.get("readers", []) if isinstance(manifest, dict) else []
        if not readers:
            raise LookupError("bundle manifest has no readers")
        if index < 0 or index >= len(readers):
            raise IndexError(f"reader index {index} out of range")
        spec = ReaderSpec.from_dict(readers[index], base_dir=base_dir)
        return self.create_from_spec(spec)

    def load_all_from_manifest(self, manifest: BundleManifest | dict[str, Any]) -> list[Any]:
        """Construct every reader declared in a bundle manifest."""
        if isinstance(manifest, BundleManifest):
            base_dir = manifest.base_dir
            readers = manifest.readers
        else:
            base_dir = None
            readers = manifest.get("readers", []) if isinstance(manifest, dict) else []
        out = []
        for entry in readers:
            spec = ReaderSpec.from_dict(entry, base_dir=base_dir)
            out.append(self.create_from_spec(spec))
        return out


__all__ = ["TargetReaderRegistry", "ReaderSpec", "ReaderFactory"]
