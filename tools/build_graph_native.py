"""Build the graph extension for every macOS release MLX publishes a library
for, and every Python the package supports.

The extension (src/backcountry_runtime/graph_native) is C++ built against MLX's
library, which comes in one wheel per macOS release (mlx-metal for macOS 14, 15
and 26), and against one Python's headers. The package ships one built binary
per wheel and Python in
src/backcountry_runtime/graph_native/prebuilt/<wheel tag>/<python>/, so
installing needs no compiler. Run this after editing graph_native.cpp or
CMakeLists.txt, after changing the MLX version in uv.lock, or after widening
requires-python: it downloads each mlx-metal wheel the lock pins (checking its
hash), builds against it under each Python (fetched by uv) and replaces
prebuilt/.

Needs Apple's Command Line Tools (the C++ compiler).

Usage: uv run python tools/build_graph_native.py
"""
import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from backcountry_runtime import graph_native

ROOT = Path(__file__).resolve().parents[1]
PREBUILT = Path(graph_native.__file__).parent / "prebuilt"
WHEELS = Path.home() / ".cache" / "backcountry" / "wheels"


def wheel(entry: dict) -> Path:
    """The wheel at entry's url, downloaded once and checked against the lock's hash."""
    path = WHEELS / entry["url"].rsplit("/", 1)[1]
    if not path.is_file():
        WHEELS.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(entry["url"]) as response, open(path.with_suffix(".part"), "wb") as out:
            shutil.copyfileobj(response, out)
        path.with_suffix(".part").rename(path)
    algorithm, digest = entry["hash"].split(":", 1)
    if hashlib.new(algorithm, path.read_bytes()).hexdigest() != digest:
        path.unlink()
        raise SystemExit(f"{path.name}: hash does not match uv.lock")
    return path


# requires-python in pyproject.toml; not 3.15: MLX's build for it makes its array
# type immutable, and the tracer records a model by patching its methods
PYTHONS = ("3.12", "3.13", "3.14")


def build_all(out: Path) -> None:
    """Under this interpreter, build against every mlx-metal wheel the lock pins
    into out/<wheel tag>/<python>/."""
    version = importlib.metadata.version("mlx")
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    metal = next(p for p in lock["package"] if p["name"] == "mlx-metal" and p["version"] == version)
    with tempfile.TemporaryDirectory() as temporary:
        for entry in metal["wheels"]:
            tag = "-".join(entry["url"].rsplit("/", 1)[1][:-len(".whl")].split("-")[2:])
            unpacked = Path(temporary) / tag
            with zipfile.ZipFile(wheel(entry)) as archive:
                archive.extractall(unpacked)
            identity = graph_native._identity(tag)
            binary = graph_native._build(unpacked / "out", mlx_dir=unpacked / "mlx" / "share" / "cmake" / "MLX",
                                         identity=identity)
            target = out / tag / identity["python"]
            target.mkdir(parents=True)
            shutil.copy2(binary, target / binary.name)
            shutil.copy2(binary.with_name("build.json"), target / "build.json")
            (target / "build.json").chmod(0o644)
            print(f"built against {tag} for {identity['python']}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--into":  # one Python, run by the loop below
        build_all(Path(sys.argv[2]))
        return
    pins = [f"{name}=={importlib.metadata.version(name)}" for name in ("mlx", "nanobind", "cmake")]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with tempfile.TemporaryDirectory() as temporary:
        for python in PYTHONS:
            command = ["uv", "run", "--no-project", "--isolated", "--python", python,
                       *(arg for pin in pins for arg in ("--with", pin)),
                       "python", __file__, "--into", temporary]
            subprocess.run(command, env=env, cwd=ROOT, check=True)
        shutil.rmtree(PREBUILT, ignore_errors=True)
        shutil.copytree(temporary, PREBUILT)
    count = len(list(PREBUILT.glob("*/*/build.json")))
    print(f"wrote {count} binaries to {PREBUILT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
