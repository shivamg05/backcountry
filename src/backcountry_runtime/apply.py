"""apply(): install a job's artifact onto a freshly loaded build() model.

Loads the swap table, the generated patch module, and the kernel specs, then
swaps wrapper instances in at their scope addresses. Works in a fresh process
with no harness import; a root-scope entry returns the wrapper in place of the
model, so callers use the return value.

Each layer is replaced only when it is the layer the kernels were made for:
the same class, weight shapes and dtypes, and quantization (the swap table's
"module", written at export), under the MLX version the artifact was made
with. Any other layer keeps its own code, and a warning says which and
why, so the call is safe on a model the artifact does not fit. An engine that
holds the same layers at other addresses (MLX-VLM's language_model) gets them
replaced where they are; a chip other than the one the kernels were timed on
gets them with a warning that their speed there is unmeasured. Called without
an artifact, apply() picks the newest one in the local store that fits.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

from . import kernels as _kernels
from .swap import install


def apply(model: object, artifact_dir: str | Path | None = None, *, baseline=False, strict=False,
          _installed: list | None = None) -> object:
    """artifact_dir: the artifact's folder; left out, the newest artifact in the
    local store (artifacts_home(), where every finished job puts its own)
    whose layers this model has. strict: raise on a layer that does not
    match (the artifact's own checks). _installed: gets (path, original,
    wrapper) for each layer replaced, so compare() can switch them."""
    if artifact_dir is None:
        artifact_dir = find(model)
        if artifact_dir is None:
            from .generation import stored
            if stored(model) is None:  # else its artifact is a generation loop: backcountry.generate
                warnings.warn(f"no artifact in {artifacts_home()} is for this model: it runs as it is",
                              stacklevel=2)
            return model
    artifact_dir = Path(artifact_dir)
    prefix = "baseline_" if baseline else ""
    table, metadata = _read(artifact_dir, prefix)

    spec_mod = importlib.util.spec_from_file_location(
        "autotune_patch_wrappers", artifact_dir / "patch" / f"{prefix}wrappers.py"
    )
    wrappers = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(wrappers)

    specs = {}
    for metal in ([] if baseline else sorted((artifact_dir / "kernels").glob("*.metal"))):
        spec = _kernels.load_spec(metal)
        specs[spec.kernel_id] = spec

    version = _version_problem(metadata.get("mlx"))
    _warn_other_chip(artifact_dir.name, metadata.get("chip"))
    if not version and (libraries := changed_libraries(artifact_dir)):
        warnings.warn(f"{artifact_dir.name}: checked with other library versions ({'; '.join(libraries)}); "
                      "its kernels are installed, but outputs and speed here are unverified", stacklevel=2)
    paths = _scope_paths(model, table, metadata)
    moved = {} if version else _moves(model, table, paths)
    skipped = {}
    patched = model
    for entry, path in zip(table, paths):
        cls = getattr(wrappers, entry["wrapper_class"])
        kernel_specs = {kid: specs[kid] for kid in entry["kernel_ids"]}
        path = moved.get(path, path)
        try:
            original = patched if path == "" else _resolve_child(patched, path)
        except (AttributeError, KeyError, IndexError, ValueError):
            original, why = None, "the model has no layer there"
        else:
            why = version or _mismatch(entry.get("module"), original)
        if why:
            skipped[path or "(the model)"] = why
            continue
        wrapper = cls(patched if path == "" else original, kernel_specs)
        if path == "":
            patched = wrapper
        else:
            install(patched, path, wrapper)
        if _installed is not None:
            _installed.append((path, original, wrapper))
    if skipped:
        reasons = sorted(set(skipped.values()))
        message = (f"{artifact_dir.name}: {len(skipped)} of {len(table)} layers left as they are, "
                   f"since they are not the layers its kernels were made for: {'; '.join(reasons)} "
                   f"(first: {next(iter(skipped))})")
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    return patched


def backcountry_home() -> Path:
    """backcountry's own folder on this Mac: $BACKCOUNTRY_HOME, by default ~/.backcountry."""
    import os
    return Path(os.environ.get("BACKCOUNTRY_HOME") or "~/.backcountry").expanduser()


def artifacts_home() -> Path:
    """The local store of finished artifacts apply(model) looks in:
    $BACKCOUNTRY_HOME/artifacts, by default ~/.backcountry/artifacts."""
    return backcountry_home() / "artifacts"


def register(artifact_dir: str | Path, name: str) -> Path:
    """Copy a finished, validated artifact into the local store as `name`
    (replacing one of that name), where apply(model) finds it."""
    import shutil
    home = artifacts_home()
    home.mkdir(parents=True, exist_ok=True)
    incoming, dest = home / f".{name}.incoming", home / name
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.copytree(artifact_dir, incoming, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.rmtree(dest, ignore_errors=True)
    incoming.rename(dest)
    return dest


def find(model) -> Path | None:
    """The newest artifact in the local store whose every layer this model
    has, each the layer its kernels were made for, under this MLX version. Artifacts exported before layers were recorded are never
    picked this way: nothing says what they fit."""
    home = artifacts_home()
    stored = [d for d in home.iterdir() if not d.name.startswith(".") and (d / "swap_table.json").is_file()] \
        if home.is_dir() else []
    for artifact_dir in sorted(stored, key=lambda d: (d / "swap_table.json").stat().st_mtime, reverse=True):
        if _fits(model, artifact_dir):
            return artifact_dir
    return None


def _fits(model, artifact_dir: Path) -> bool:
    try:
        table, metadata = _read(artifact_dir, "")
        if not table or _version_problem(metadata.get("mlx")) or any(e.get("module") is None for e in table):
            return False
        paths = _scope_paths(model, table, metadata)
    except (OSError, ValueError):
        return False
    moved = _moves(model, table, paths)
    for entry, path in zip(table, paths):
        path = moved.get(path, path)
        try:
            original = model if path == "" else _resolve_child(model, path)
        except (AttributeError, KeyError, IndexError, ValueError):
            return False
        if _mismatch(entry["module"], original) is not None:
            return False
    return True


def _read(artifact_dir: Path, prefix: str) -> tuple[list, dict]:
    table = json.loads((artifact_dir / f"{prefix}swap_table.json").read_text())
    metadata_path = artifact_dir / "bundle.json"
    return table, json.loads(metadata_path.read_text()) if metadata_path.exists() else {}


def _scope_paths(model, table, metadata) -> list[str]:
    """Each entry's address in the model a caller hands over. Recorded paths
    of a cached job include the harness's step prefix (ContextStep.model);
    normal inference callers supply their own cache and pass the model."""
    from .state import ContextStep
    from .inference import AppRun, LibraryInference
    bare_context_model = (metadata.get("context") is not None or metadata.get("use_library_inference")
                          or metadata.get("app_run")) \
        and not isinstance(model, (ContextStep, LibraryInference, AppRun))
    paths = []
    for entry in table:
        path = entry["scope_path"]
        if bare_context_model:
            if path == "model":
                path = ""
            elif path.startswith("model."):
                path = path[len("model."):]
            else:
                raise ValueError(f"cached artifact contains a non-model patch path: {path!r}")
        paths.append(path)
    return paths


def _moves(model, table, paths) -> dict:
    nested = [(entry, path) for entry, path in zip(table, paths) if path]
    return _relocation(model, [e for e, _ in nested], [p for _, p in nested])


def _relocation(model, table, paths) -> dict:
    """Where each recorded layer is in a model that holds the same layers at
    other addresses: one level down (MLX-VLM keeps the language model at
    language_model), or the model's inner body handed over alone. The one
    move of the recorded prefix that finds every layer, each the layer its
    kernels were made for; {} when the recorded addresses hold, or when no
    move, or more than one, fits (each layer is then looked for where it was
    recorded, and a missing one is reported)."""
    named = getattr(model, "named_modules", None)
    if not callable(named) or not paths:
        return {}
    live = dict(named())
    if all(p in live for p in paths):
        return {}
    first = paths[0].split(".")
    for k in range(len(first)):  # keep as much of the recorded address as fits: layers.0 never becomes layers.1
        rest, old = ".".join(first[k:]), ".".join(first[:k])
        fits = set()
        for name in live:
            if name != rest and not name.endswith("." + rest):
                continue
            new = name[:len(name) - len(rest)].rstrip(".")
            moved = tuple(_moved(p, old, new) for p in paths)
            if all(m in live and _mismatch(entry.get("module"), live[m]) is None
                   for m, entry in zip(moved, table)):
                fits.add(moved)
        if fits:
            return dict(zip(paths, next(iter(fits)))) if len(fits) == 1 else {}
    return {}


def _moved(path: str, old: str, new: str) -> str | None:
    """path with its leading `old` components replaced by `new`."""
    if old and path != old and not path.startswith(old + "."):
        return None
    rest = path[len(old):].lstrip(".")
    return ".".join(part for part in (new, rest) if part)


def _warn_other_chip(name: str, made_on: str | None) -> None:
    """Kernels are timed on one chip: on another they still compute the same
    values, but whether they are faster there was never measured."""
    import mlx.core as mx
    here = mx.device_info().get("device_name")
    if made_on and here and made_on != here:
        warnings.warn(f"{name}: its kernels were timed on an {made_on}; this is an {here}, where their speed "
                      "is unmeasured (run the artifact's benchmark.py to measure it)", stacklevel=3)


def layer_signature(module) -> dict:
    """What a layer's kernels were made for: its class, every weight's shape
    and dtype, and each quantized part's bits, group size and mode."""
    from mlx.utils import tree_flatten
    parameters = module.parameters() if callable(getattr(module, "parameters", None)) else {}
    quantized = {}
    for name, part in (module.named_modules() if callable(getattr(module, "named_modules", None)) else []):
        if hasattr(part, "bits") and hasattr(part, "group_size"):
            quantized[name] = [part.bits, part.group_size, getattr(part, "mode", None)]
    return {"class": type(module).__qualname__,
            "weights": {name: [list(a.shape), str(a.dtype).removeprefix("mlx.core.")]
                        for name, a in tree_flatten(parameters)},
            "quantization": quantized}


def _mismatch(recorded: dict | None, module) -> str | None:
    """Why this layer is not the recorded one, or None. An artifact exported
    before layers were recorded installs as before."""
    if recorded is None:
        return None
    found = json.loads(json.dumps(layer_signature(module)))
    if found["class"] != recorded["class"]:
        return f"a {found['class']} where the kernels were made for a {recorded['class']}"
    if found["quantization"] != recorded["quantization"]:
        return "quantized differently"
    if found["weights"] != recorded["weights"]:
        name = next((n for n in sorted(set(found["weights"]) | set(recorded["weights"]))
                     if found["weights"].get(n) != recorded["weights"].get(n)), None)
        return f"weight {name} is {found['weights'].get(name)}, not {recorded['weights'].get(name)}"
    return None


def changed_libraries(artifact_dir) -> list[str]:
    """The libraries the model code ran on when this result was checked
    (requirements.txt: mlx-lm, mlx-vlm, mflux, an engine's, ...) that are
    installed at another version here, each as "name made_with -> here". A
    library's code changing between versions (MLX-LM 0.31.3 and LM Studio's
    0.31.4 build differ in generation and in the Qwen model) voids an exact
    result made against it."""
    from importlib import metadata
    from pathlib import Path
    path = Path(artifact_dir) / "requirements.txt"
    changed = []
    for line in path.read_text().splitlines() if path.is_file() else ():
        name, _, made_with = line.strip().partition("==")
        if line.startswith("#") or not made_with:
            continue
        try:
            here = metadata.version(name)
        except metadata.PackageNotFoundError:
            here = "not installed"
        if here != made_with:
            changed.append(f"{name} {made_with} -> {here}")
    return changed


def _version_problem(made_with: str | None) -> str | None:
    """Any other MLX release, a patch one included, can compute the replaced
    operations differently: under 0.32.4 MLX's own expert matmul sums in another
    order than 0.32.2's, so Qwen3-30B-A3B kernels exact under 0.32.2 changed its
    reply from the second token and were no faster."""
    import mlx.core as mx
    if made_with and made_with != mx.__version__:
        return f"made with MLX {made_with}, this is MLX {mx.__version__}"
    return None


def _resolve_child(model: object, path: str):
    from .swap import resolve

    _, _, child = resolve(model, path)
    return child
