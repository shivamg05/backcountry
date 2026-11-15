"""Manifest: the job contract. Parse, validate, default; frozen once loaded.

Every defaulted field is listed in Manifest.defaulted so the report can
record what was actually used.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import mlx.core as mx
import yaml

DEFAULT_SWEEP_SIZES = (1, 13, 50, 4096)
DEFAULT_BUDGET_PER_REGION = 25
DEFAULT_BUDGET_TOTAL = 250
DEFAULT_BUDGET_MODEL = 20  # identical: output: inference-loop attempts
IDENTICAL = ("computation", "output")  # what a job keeps the same
DEFAULT_INT_RANGE = (0, 100)
BASELINES = ("compiled", "plain")
DEFAULT_JOB_SEED = 0x4D455441  # fixed by the harness, recorded, never supplied
BOUNDARY_INPUT_SETS = 3

# Floating-evaluation changes use these defaults unless the manifest overrides them.
from backcountry_runtime.numeric import DEFAULT_TOLERANCES, validate_tolerances

_DTYPE_NAMES: Mapping[str, mx.Dtype] = MappingProxyType({
    name: getattr(mx, attr)
    for name, attr in {
        "bool": "bool_", "uint8": "uint8", "uint16": "uint16", "uint32": "uint32",
        "uint64": "uint64", "int8": "int8", "int16": "int16", "int32": "int32",
        "int64": "int64", "float16": "float16", "bfloat16": "bfloat16",
        "float32": "float32", "complex64": "complex64",
    }.items()
})

FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
INT_DTYPES = frozenset({"uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"})


# What kind of model this is, for the knowledge file: later jobs on the same
# kind see each other's attempts first.
MODEL_TYPES = ("llm", "moe-llm", "vlm", "vla", "diffusion", "other")


class ManifestError(ValueError):
    """A manifest problem the user can fix; the message says how."""


def resolve_dtype(name: str) -> mx.Dtype:
    if name not in _DTYPE_NAMES:
        raise ManifestError(
            f"unknown dtype {name!r}; use one of {sorted(_DTYPE_NAMES)}"
        )
    return _DTYPE_NAMES[name]


@dataclass(frozen=True)
class InputSpec:
    """One positional argument of the model: a shape plus a dtype.

    A str dim is a named, sweepable dim; int dims are model constants and never
    move. Integer dtypes sample from [low, high) so token ids stay in range.
    """

    shape: tuple[int | str, ...]
    dtype: str
    low: int = DEFAULT_INT_RANGE[0]
    high: int = DEFAULT_INT_RANGE[1]

    def named_dims(self) -> frozenset[str]:
        return frozenset(d for d in self.shape if isinstance(d, str))


@dataclass(frozen=True)
class Workload:
    """The real input shapes to optimize for. Name is a report label only.

    A context is how many tokens of conversation are already in place when
    the call runs: the job runs the call as a step over the model's own KV
    cache, filled once and restored after every call. A decode step is
    `shape: [1, 1]` plus `context: 512`. context: 0 supplies an empty cache;
    omitting context leaves the model's ordinary forward call unchanged.

    generated: the tokens are generated tokens a loop reads several at a time
    (checking guesses), so every call computes each token with one-token
    arithmetic (backcountry_runtime.exact_rows) and kernels must keep its bits:
    the calls an identical: output job's speculative loop makes. Needs a context.
    """

    inputs: tuple[InputSpec, ...]
    name: str
    context: int | None = None
    generated: bool = False

    def named_dims(self) -> frozenset[str]:
        return frozenset().union(*(i.named_dims() for i in self.inputs)) if self.inputs else frozenset()


@dataclass(frozen=True)
class FinalBenchmark:
    steps: int = 10
    pairs: int = 4
    warmup_steps: int = 3  # minimum; the clock ramp can require more


@dataclass(frozen=True)
class SizeSet:
    """The sizes a manifest's range covers: every size from lo to hi,
    or exactly the listed ones (a model whose sizes come from a few settings,
    such as image resolutions)."""

    lo: int
    hi: int
    listed: tuple[int, ...] = ()

    def sizes(self) -> tuple[int, ...]:
        return self.listed or tuple(range(self.lo, self.hi + 1))

    def recordings(self, primary: int) -> list[int]:
        """Where the model is recorded besides the primary size: every listed
        size, or both ends and the middle of a range."""
        points = self.listed or (self.lo, (self.lo + self.hi) // 2, self.hi)
        return [n for n in dict.fromkeys(points) if n != primary]


SEARCH_SIZES = 10  # sizes of a range the search itself runs at


def search_sizes(sizes: SizeSet, primary: int | None, seed: int, count: int = SEARCH_SIZES) -> tuple[int, ...]:
    """The sizes the search runs at: every listed size, or over a range one
    size per bin of `count` bins growing by the same factor from lo to hi,
    the primary size (if any) in its own bin and a seeded random size in each other.
    Speed changes with the regime (launch-bound at small sizes, compute-bound
    at large), which moves over multiples, so the bins grow by a factor;
    tile alignment (128 vs 129 rows) is not smooth, so the size in a bin is
    random rather than a grid point."""
    import random
    fixed = set() if primary is None else {primary}
    if sizes.listed:
        return tuple(sorted(set(sizes.listed) | fixed))
    lo, hi, rng = sizes.lo, sizes.hi, random.Random(seed)
    edges = [round(lo * (hi / lo) ** (i / count)) for i in range(count)] + [hi + 1]
    chosen = set()
    for a, b in zip(edges, edges[1:]):
        if a < b:
            chosen.add(primary if primary is not None and a <= primary < b else rng.randrange(a, b))
    return tuple(sorted(chosen | fixed))


@dataclass(frozen=True)
class Manifest:
    model_path: Path
    workloads: tuple[Workload, ...]
    sweep: Mapping[str, tuple[int, ...]]
    primary: Mapping[str, int]
    tolerances: tuple[float, float] | None
    budget_per_region: int
    budget_total: int
    seed: int
    defaulted: tuple[str, ...]
    baseline: str = "compiled"     # "compiled" | "plain": what a win is measured against
    tolerance_allowed: bool = False  # kernels may differ within tolerance (opt-in: tolerances set)
    final_benchmark: FinalBenchmark = field(default_factory=FinalBenchmark)
    use_library_inference: bool | None = None  # None resolves once from model/workload support
    model_type: str = "other"      # one of MODEL_TYPES: labels this job's knowledge-file lines
    # range, one named dim -> SizeSet: after the search,
    # shipped kernels also run at every size of the set they are checked
    # correct and faster at. Unset, an MLX-LM prompt workload gets it
    # automatically (serve_auto); false turns that off.
    serve: Mapping[str, "SizeSet"] = field(default_factory=dict)
    serve_auto: bool = True
    # what must stay the same: "computation" (today's kernels: the same math)
    # or "output" (the same outputs for the same inputs; the inference loop,
    # run(), may change too). budget_model counts the inference-loop attempts.
    identical: str = "computation"
    function: str | None = None  # "module:qualname" the app calls; the loop search rewrites it
    budget_model: int = 0
    # Hugging Face checkpoints the loop search may use beside the model (a
    # multi-token head, a drafter): downloaded before it starts
    assets: tuple[str, ...] = ()


    def named_dims(self) -> frozenset[str]:
        return frozenset().union(*(w.named_dims() for w in self.workloads))

    def tolerance_for(self, dtype: str) -> tuple[float, float]:
        if self.tolerances is not None:
            return self.tolerances
        if dtype in DEFAULT_TOLERANCES:
            return DEFAULT_TOLERANCES[dtype]
        raise ManifestError(f"no tolerance default for dtype {dtype!r}; set tolerances in the manifest")


def _parse_shape(raw: Any, where: str) -> tuple[int | str, ...]:
    if not isinstance(raw, list):
        raise ManifestError(f"{where}: shape must be a list ([] for a single number), got {raw!r}")
    dims: list[int | str] = []
    for d in raw:
        if isinstance(d, bool) or not isinstance(d, (int, str)):
            raise ManifestError(f"{where}: shape dims are ints or named strings, got {d!r}")
        if isinstance(d, int) and d < 1:
            raise ManifestError(f"{where}: integer dims must be >= 1, got {d}")
        if isinstance(d, str) and not d.isidentifier():
            raise ManifestError(f"{where}: named dim {d!r} must be an identifier")
        dims.append(d)
    return tuple(dims)


def _parse_input(raw: Any, where: str) -> InputSpec:
    if not isinstance(raw, dict) or "shape" not in raw or "dtype" not in raw:
        raise ManifestError(f"{where}: each input needs {{shape, dtype}}, got {raw!r}")
    unknown = set(raw) - {"shape", "dtype", "low", "high"}
    if unknown:
        raise ManifestError(f"{where}: unknown input keys {sorted(unknown)}")
    dtype = raw["dtype"]
    if not isinstance(dtype, str):
        raise ManifestError(f"{where}: dtype must be a string, got {dtype!r}")
    resolve_dtype(dtype)
    low = raw.get("low", DEFAULT_INT_RANGE[0])
    high = raw.get("high", DEFAULT_INT_RANGE[1])
    if ("low" in raw or "high" in raw) and dtype not in INT_DTYPES:
        raise ManifestError(f"{where}: low/high apply to integer dtypes only")
    if not (isinstance(low, int) and isinstance(high, int) and low < high):
        raise ManifestError(f"{where}: need integer low < high, got low={low!r} high={high!r}")
    return InputSpec(shape=_parse_shape(raw["shape"], where), dtype=dtype, low=low, high=high)


def _parse_workload(raw: Any, index: int) -> Workload:
    where = f"workloads[{index}]"
    if not isinstance(raw, dict) or "inputs" not in raw:
        raise ManifestError(f"{where}: a workload needs an inputs list, got {raw!r}")
    unknown = set(raw) - {"inputs", "name", "context", "generated"}
    if unknown:
        raise ManifestError(f"{where}: unknown workload keys {sorted(unknown)}")
    raw_inputs = raw["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ManifestError(f"{where}: inputs must be a non-empty list")
    name = raw.get("name", f"workload{index}")
    if not isinstance(name, str) or not name:
        raise ManifestError(f"{where}: name must be a non-empty string")
    if "@" in name:
        raise ManifestError(f"{where}: name cannot contain @, which is reserved for generated shape labels")
    inputs = tuple(_parse_input(r, f"{where}.inputs[{i}]") for i, r in enumerate(raw_inputs))
    context = raw.get("context")
    if context is not None:
        if isinstance(context, bool) or not isinstance(context, int) or context < 0:
            raise ManifestError(f"{where}: context must be a whole number of tokens, got {context!r}")
        if len(inputs) != 1 or inputs[0].dtype not in INT_DTYPES:
            raise ManifestError(f"{where}: a workload with a context has one input, the token ids")
    generated = raw.get("generated", False)
    if not isinstance(generated, bool) or (generated and context is None):
        raise ManifestError(f"{where}: generated must be true or false, and needs a context")
    return Workload(inputs=inputs, name=name, context=context, generated=generated)


def _parse_positive_int(raw: Any, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ManifestError(f"{where}: must be a positive integer, got {raw!r}")
    return raw


SERVE_KEY = "range"  # the sizes the job optimizes for: searched, checked, served
SERVE_ALIAS = "check_optimizations_for"  # its earlier name, still read


def load(path: str | Path) -> Manifest:
    """Parse and validate a manifest file. Model path resolves relative to it."""
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ManifestError(f"manifest file not found: {path}")
    except yaml.YAMLError as e:
        raise ManifestError(f"manifest is not valid YAML: {e}")
    if not isinstance(raw, dict):
        raise ManifestError(f"manifest must be a mapping, got {type(raw).__name__}")
    if SERVE_ALIAS in raw:
        if SERVE_KEY in raw:
            raise ManifestError(f"give {SERVE_KEY} or its earlier name {SERVE_ALIAS}, not both")
        raw[SERVE_KEY] = raw.pop(SERVE_ALIAS)

    known = {"model", "workloads", "sweep", "primary", "tolerances", "budget",
             "baseline", "final_benchmark", "use_library_inference", SERVE_KEY,
             "model_type", "identical", "function", "assets"}
    unknown = set(raw) - known
    if unknown:
        raise ManifestError(
            f"unknown manifest keys {sorted(unknown)}; dtypes and quantization are not "
            f"knobs and the chip is auto-detected (allowed: {sorted(known)})"
        )
    if "model" not in raw:
        raise ManifestError("manifest needs a model: path to a .py file defining build()")
    model_path = (path.parent / str(raw["model"])).resolve()
    if not model_path.is_file():
        raise ManifestError(f"model file does not exist: {model_path}")
    if model_path.suffix != ".py":
        raise ManifestError(f"model must be a .py file defining build(), got {model_path.name}")

    raw_workloads = raw.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ManifestError("manifest needs a non-empty workloads list")
    workloads = tuple(_parse_workload(w, i) for i, w in enumerate(raw_workloads))
    names = [w.name for w in workloads]
    if len(set(names)) != len(names):
        raise ManifestError(f"workload names must be unique, got {names}")
    if sum(w.context is not None for w in workloads) > 1:
        raise ManifestError("a manifest can have one workload with a context (one conversation, "
                            "the model's one cache); the others run without one")

    named = frozenset().union(*(w.named_dims() for w in workloads))
    defaulted: list[str] = []
    use_library_inference = raw.get("use_library_inference")
    if "use_library_inference" in raw:
        if not isinstance(use_library_inference, bool):
            raise ManifestError("use_library_inference must be true or false")
    else:
        defaulted.append("use_library_inference")

    raw_sweep = raw.get("sweep", {})
    if not isinstance(raw_sweep, dict):
        raise ManifestError("sweep must be a mapping of named dim to size list")
    sweep: dict[str, tuple[int, ...]] = {}
    for dim, sizes in raw_sweep.items():
        if dim not in named:
            raise ManifestError(
                f"sweep names dim {dim!r} which appears in no workload shape; "
                f"integer dims are model constants and are never swept (named dims: {sorted(named) or 'none'})"
            )
        if not isinstance(sizes, list) or not sizes:
            raise ManifestError(f"sweep.{dim}: must be a non-empty list of sizes")
        sweep[dim] = tuple(_parse_positive_int(s, f"sweep.{dim}") for s in sizes)
    served = set(raw.get(SERVE_KEY) or {}) if isinstance(raw.get(SERVE_KEY), dict) else set()
    raw_primary_dims = raw.get("primary") if isinstance(raw.get("primary"), dict) else {}
    for dim in sorted(named - set(sweep)):
        if dim in served:
            # serving checks its own sizes after the search; no correctness
            # sweep reaches the search unless one is asked for
            if dim not in raw_primary_dims:
                raise ManifestError(f"{SERVE_KEY}.{dim}: set primary.{dim}, the size to optimize for")
            sweep[dim] = (_parse_positive_int(raw_primary_dims[dim], f"primary.{dim}"),)
            continue
        sweep[dim] = DEFAULT_SWEEP_SIZES
        defaulted.append(f"sweep.{dim}")

    # A named dim needs one concrete size for tracing and pricing; the sweep sizes
    # are correctness-only. Default: the largest sweep size. Overridable via primary.
    raw_primary = raw.get("primary", {})
    if not isinstance(raw_primary, dict):
        raise ManifestError("primary must be a mapping of named dim to one size")
    primary: dict[str, int] = {}
    for dim, size in raw_primary.items():
        if dim not in named:
            raise ManifestError(f"primary names dim {dim!r} which appears in no workload shape")
        primary[dim] = _parse_positive_int(size, f"primary.{dim}")
    for dim in sorted(named - set(primary)):
        primary[dim] = max(sweep[dim])
        defaulted.append(f"primary.{dim}")

    raw_serve = raw.get(SERVE_KEY)
    serve_auto = raw_serve is None
    if raw_serve is False:
        raw_serve = {}
    elif raw_serve is None:
        raw_serve = {}
        defaulted.append(SERVE_KEY)
    if not isinstance(raw_serve, dict) or len(raw_serve) > 1:
        raise ManifestError(f"{SERVE_KEY} must map one named dim to [smallest, largest] or "
                            "{sizes: [...]}, or be false")
    serve: dict[str, SizeSet] = {}
    for dim, span in raw_serve.items():
        if dim not in named:
            raise ManifestError(f"{SERVE_KEY} names dim {dim!r} which appears in no workload shape")
        if isinstance(span, dict):
            if set(span) != {"sizes"} or not isinstance(span["sizes"], list):
                raise ManifestError(f"{SERVE_KEY}.{dim}: exact sizes are {{sizes: [a, b, c, ...]}}")
            listed = tuple(sorted({_parse_positive_int(v, f"{SERVE_KEY}.{dim}.sizes") for v in span["sizes"]}))
            primary_size = raw_primary_dims.get(dim)
            if len(set(listed) | ({primary_size} - {None})) < 3:
                raise ManifestError(f"{SERVE_KEY}.{dim}.sizes: give at least three sizes, the optimized one "
                                    "included, so the code for other sizes can be derived and checked")
            serve[dim] = SizeSet(listed[0], listed[-1], listed)
            continue
        if not isinstance(span, list) or len(span) != 2:
            raise ManifestError(f"{SERVE_KEY}.{dim}: must be [smallest, largest] or {{sizes: [...]}}")
        lo, hi = (_parse_positive_int(v, f"{SERVE_KEY}.{dim}") for v in span)
        if hi - lo < 2:
            raise ManifestError(f"{SERVE_KEY}.{dim}: the largest size must exceed the smallest by at least 2")
        serve[dim] = SizeSet(lo, hi)

    tolerances: tuple[float, float] | None = None
    # tolerance is opt-in: "default" allows each output dtype's default error,
    # {rtol, atol} sets one; without either, every kernel must be bit-identical
    allow_tolerance = "tolerances" in raw
    if raw.get("tolerances") == "default":
        pass
    elif "tolerances" in raw:
        t = raw["tolerances"]
        if not isinstance(t, dict) or set(t) != {"rtol", "atol"}:
            raise ManifestError('tolerances must be "default" or {rtol, atol}')
        if any(isinstance(v, bool) for v in t.values()):
            raise ManifestError("tolerances must be numbers, not booleans")
        try:
            tolerances = (float(t["rtol"]), float(t["atol"]))
        except (TypeError, ValueError):
            raise ManifestError(f"tolerances must be numbers, got {t!r}")
        if any(not math.isfinite(v) or v < 0 for v in tolerances):
            raise ManifestError("tolerances must be finite and non-negative")
        try:
            validate_tolerances(tolerances)
        except ValueError as error:
            raise ManifestError(str(error)) from error
    else:
        defaulted.append("tolerances")

    # What "faster" is measured against: the model under mx.compile (the
    # faster way to run it, so the honest bar) or exactly as build() returns
    # it. Choosing by measurement is not built; the manifest decides.
    baseline = raw.get("baseline", "compiled")
    if baseline not in BASELINES:
        raise ManifestError(f"baseline must be one of {list(BASELINES)}, got {baseline!r}")
    if "baseline" not in raw:
        defaulted.append("baseline")

    raw_final = raw.get("final_benchmark", {})
    if not isinstance(raw_final, dict) or set(raw_final) - {"steps", "pairs", "warmup_steps"}:
        raise ManifestError("final_benchmark must be {steps?, pairs?, warmup_steps?}")
    final_values = {}
    for key in ("steps", "pairs", "warmup_steps"):
        if key in raw_final:
            final_values[key] = _parse_positive_int(raw_final[key], f"final_benchmark.{key}")
        else:
            defaulted.append(f"final_benchmark.{key}")
    final_benchmark = FinalBenchmark(**final_values)
    if final_benchmark.pairs < 4 or final_benchmark.pairs % 2:
        raise ManifestError("final_benchmark.pairs must be even and >= 4 to balance run order")

    identical = raw.get("identical", "computation")
    if identical not in IDENTICAL:
        raise ManifestError(f"identical must be one of {list(IDENTICAL)}, got {identical!r}")
    if "identical" not in raw:
        defaulted.append("identical")
    # the function the application calls the model through, which the loop
    # search hands back faster (the model file's run() calls it as the app does)
    function = raw.get("function")
    if function is not None:
        if not isinstance(function, str) or ":" not in function or not all(function.partition(":")[::2]):
            raise ManifestError(f'function must be "module:qualname", got {function!r}')
        if identical != "output":
            raise ManifestError("function names what the loop search rewrites: it needs identical: output")
        if final_benchmark.steps != 1:
            raise ManifestError("with function, run() is one use of the app's function (it makes its whole reply or "
                                "image itself): final_benchmark.steps counts run() calls per use and must be 1")
    assets = raw.get("assets") or []
    if not isinstance(assets, list) or not all(isinstance(a, str) and a.count("/") == 1 and all(a.split("/"))
                                               for a in assets):
        raise ManifestError(f'assets must be a list of Hugging Face ids ("author/name"), got {assets!r}')
    if assets and identical != "output":
        raise ManifestError("assets are for the generation loop search: they need identical: output")
    raw_budget = raw.get("budget", {})
    if isinstance(raw_budget, dict) and "hours" in raw_budget:
        raise ManifestError("budget.hours is no longer supported: attempts bound a run "
                            "(budget.total, budget.per_region, budget.model)")
    if not isinstance(raw_budget, dict) or set(raw_budget) - {"per_region", "total", "model"}:
        raise ManifestError("budget must be {per_region?, total?, model?}")
    if "model" in raw_budget and identical != "output":
        raise ManifestError("budget.model counts run() attempts, which only identical: output makes")
    budget_model = 0
    if identical == "output":
        budget_model = (_parse_positive_int(raw_budget["model"], "budget.model") if "model" in raw_budget
                        else DEFAULT_BUDGET_MODEL)
    if "per_region" in raw_budget:
        per_region = _parse_positive_int(raw_budget["per_region"], "budget.per_region")
    else:
        per_region = DEFAULT_BUDGET_PER_REGION
        defaulted.append("budget.per_region")
    if identical == "output" and raw_budget.get("total") == 0:
        total = 0  # the inference-loop search alone: no kernel search
    elif "total" in raw_budget:
        total = _parse_positive_int(raw_budget["total"], "budget.total")
    else:
        total = DEFAULT_BUDGET_TOTAL
        defaulted.append("budget.total")

    defaulted.append("seed")
    model_type = raw.get("model_type", "other")  # only labels the knowledge file's lines
    if model_type not in MODEL_TYPES:
        raise ManifestError(f"model_type must be one of {list(MODEL_TYPES)}; got {model_type!r}")
    return Manifest(
        model_path=model_path,
        workloads=workloads,
        sweep=MappingProxyType(sweep),
        primary=MappingProxyType(primary),
        tolerances=tolerances,
        tolerance_allowed=allow_tolerance,
        budget_per_region=per_region,
        budget_total=total,
        seed=DEFAULT_JOB_SEED,
        baseline=baseline,
        final_benchmark=final_benchmark,
        use_library_inference=use_library_inference,
        model_type=model_type,
        defaulted=tuple(defaulted),
        serve=MappingProxyType(serve),
        serve_auto=serve_auto,
        identical=identical,
        function=function,
        budget_model=budget_model,
        assets=tuple(assets),
    )


_BUILD_PROBE = textwrap.dedent("""
    import importlib.util, inspect, sys
    path = sys.argv[1]
    spec = importlib.util.spec_from_file_location("autotune_model", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"import failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    build = getattr(mod, "build", None)
    if build is None:
        print("no build() found", file=sys.stderr)
        sys.exit(4)
    if not callable(build):
        print(f"build is not callable (it is {type(build).__name__})", file=sys.stderr)
        sys.exit(4)
    try:
        params = inspect.signature(build).parameters.values()
    except (TypeError, ValueError):
        params = ()
    required = [p.name for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if required:
        print(f"build() must take no arguments, but requires {required}", file=sys.stderr)
        sys.exit(5)
    print("ok")
""")


def check_build(manifest: Manifest, timeout_s: float = 60.0) -> None:
    """Probe the build() contract in a fresh subprocess with a foreign CWD.

    Catches: a model file that fails to import cleanly (including import-time
    CWD-relative reads, since the probe runs elsewhere), a missing or
    non-callable build, and a build() with required arguments. Raises
    ManifestError with the fix spelled out.
    """
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            [sys.executable, "-c", _BUILD_PROBE, str(manifest.model_path)],
            capture_output=True, text=True, cwd=cwd, timeout=timeout_s,
        )
    if proc.returncode == 0:
        return
    detail = proc.stderr.strip() or f"exit code {proc.returncode}"
    fix = {
        3: "the model file must import cleanly in a fresh process with a different "
           "working directory; resolve paths relative to __file__, never the CWD",
        4: f"define build() at module level in {manifest.model_path.name}; it must "
           "return the model as a self-contained repeatable callable, not a live object",
        5: "build() takes no arguments; bake configuration into the model file",
    }.get(proc.returncode, "the model file must be importable in a fresh subprocess")
    raise ManifestError(f"model probe failed ({detail}). Fix: {fix}")
