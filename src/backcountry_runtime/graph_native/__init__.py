"""MLX graph rewriting, compiled once during setup and reusable in exported bundles.

``rewrite(roots, pattern_outputs, parameters, replacement)`` returns new roots
and the number of replacements. The callback receives ordered boundary inputs
and returns ordered boundary outputs, or ``None`` to retain that occurrence.
All calls are graph construction work; cache/compile their result for inference.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile

_module = None
_name = "_graph_native"


REBUILDABLE = {"source": None, "mlx_library": None}  # a binary differing only here is built again


def _library_tag() -> str | None:
    """The wheel tag of the MLX library (mlx-metal: headers, libmlx), which
    MLX publishes once per macOS release (14, 15, 26)."""
    try:
        wheel = importlib.metadata.distribution("mlx-metal").read_text("WHEEL") or ""
    except importlib.metadata.PackageNotFoundError:
        return None
    return next((line.split(":", 1)[1].strip() for line in wheel.splitlines() if line.startswith("Tag:")), None)


def _identity(library: str | None = None) -> dict:
    root = Path(__file__).parent
    source = b"".join((root / name).read_bytes() for name in ("graph_native.cpp", "CMakeLists.txt"))
    # any MLX: the package ships a binary for the newest release it was built
    # with, and another version's is built on first use and checked on load
    version = importlib.metadata.version("mlx")
    return {
        "mlx": version,
        "python": sys.implementation.cache_tag,
        "machine": platform.machine(),
        # the MLX library it is built against: its wheel's tag, not the Python's build target
        "mlx_library": library or _library_tag(),
        "source": hashlib.sha256(source).hexdigest(),
    }


def _build(directory: Path, mlx_dir: Path | None = None, identity: dict | None = None) -> Path:
    """Build into directory against the installed MLX library, or the one whose
    CMake config is at mlx_dir (stamped with identity)."""
    import mlx.core as mx

    try:
        import cmake
        import nanobind
    except ImportError as exc:
        raise RuntimeError("Graph installation needs the package build dependencies: nanobind==3.1.0 and cmake>=3.27") from exc
    if importlib.metadata.version("nanobind") != "3.1.0":
        raise RuntimeError("Graph installation requires nanobind==3.1.0 to match MLX's Python array bindings")
    root = Path(__file__).parent.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = str(Path(cmake.CMAKE_BIN_DIR) / "cmake")
    with tempfile.TemporaryDirectory(prefix="graph-build-", dir=directory) as temporary:
        build = Path(temporary)
        commands = [
            [executable, "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
             f"-DPython_EXECUTABLE={sys.executable}", f"-Dnanobind_DIR={nanobind.cmake_dir()}",
             f"-DMLX_DIR={mlx_dir or Path(mx.__file__).parent / 'share/cmake/MLX'}"],
            [executable, "--build", str(build), "-j", "2"],
        ]
        for command in commands:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(
                    f"Could not build the MLX graph extension for MLX {importlib.metadata.version('mlx')} "
                    "(the package ships one prebuilt for the newest MLX it was released with; another "
                    "version is built here, which needs Apple's Command Line Tools: xcode-select --install):\n"
                    + result.stdout[-8000:])
        built = build / (_name + sysconfig.get_config_var("EXT_SUFFIX"))
        output = directory / built.name
        # Atomic publication also allows independent worker processes to prepare it.
        os.replace(built, output)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stamp:
        json.dump(identity or _identity(), stamp, sort_keys=True)
    os.replace(stamp.name, directory / "build.json")
    return output


def _stamped(stamp: Path) -> dict | None:
    try:
        return json.loads(stamp.read_text())
    except (OSError, ValueError):
        return None


def _compatible(stamp: Path, identity: dict) -> bool:
    return _stamped(stamp) == identity


def prepare(destination: str | Path | None = None) -> Path:
    """Prepare the extension, optionally copying its binary and ABI stamp for export."""
    identity = _identity()
    filename = _name + sysconfig.get_config_var("EXT_SUFFIX")
    root = Path(__file__).parent
    # A bundle's binary sits at the root; the package ships one per MLX library
    # and Python in prebuilt/<library>/<python>/ (tools/build_graph_native.py),
    # so no compiler is needed. One whose source was edited since, or another
    # library's or Python's, is built anew.
    prebuilt = root / "prebuilt" / str(identity["mlx_library"]) / identity["python"]
    path = next((folder / filename for folder in (root, prebuilt)
                 if (folder / filename).is_file() and _compatible(folder / "build.json", identity)), None)
    stamped = _stamped(root / "build.json") if (root / filename).is_file() else None
    if path is None and stamped is not None and {**stamped, **REBUILDABLE} != {**identity, **REBUILDABLE}:
        raise RuntimeError("The bundled graph extension does not match this Python/MLX/platform version; rebuild the artifact runtime")
    if path is None:
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "backcountry" / "graph" / digest
        path = cache / filename
        stamp = cache / "build.json"
        if not path.is_file() or not _compatible(stamp, identity):
            path = _build(cache)
    if destination is not None:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        output = target / filename
        if path.resolve() != output.resolve():
            shutil.copy2(path, output)
            shutil.copy2(path.with_name("build.json"), target / "build.json")
        return output
    return path


def _load():
    global _module
    if _module is None:
        import mlx.core  # Register MLX's array type and load its shared library first.

        path = prepare()
        spec = importlib.util.spec_from_file_location(_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load graph extension at {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            module.array_id(mlx.core.array(1))   # it takes MLX's arrays: built against this MLX's bindings
        except Exception as error:
            raise RuntimeError(
                f"The MLX graph extension does not work with MLX {importlib.metadata.version('mlx')} "
                f"({type(error).__name__}: {error}): this MLX changed what the extension builds against; "
                "a backcountry release for it is needed (or an MLX backcountry was released with)") from error
        _module = module
    return _module


def rewrite(roots, pattern_outputs, parameters, replacement, *, anchors=()):
    """Known boundary arrays constrain matching before choosing an occurrence."""
    return _load().rewrite(roots, pattern_outputs, parameters, replacement, anchors)


def same_structure(original_roots, rewritten_roots):
    return _load().same_structure(original_roots, rewritten_roots)


def same_computation(original_roots, rewritten_roots):
    """same_structure, with constants remade on each call compared by value."""
    return _load().same_computation(original_roots, rewritten_roots)


def array_id(array):
    return _load().array_id(array)
