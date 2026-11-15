"""Artifact emit (spec "Leaving a region, and the artifact"): kernels, the
generated patch module, the swap table, the vendored runtime, apply(), report,
and with a ModelBundle the model source, workload recipes and the load,
validate and benchmark scripts.

The harness installs the same generated code it measured; emitting is writing
that code down, never regenerating it differently. Wrapper classes whose
bodies are identical are written once, under one name, and every module path
they served points at that class. The artifact works in a fresh process with
no backcountry.harness import: apply.py puts the vendored runtime on sys.path under its
real package name.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Sequence

import mlx.core as mx

import backcountry_runtime

from ..bind.emit import MODULE_HEADER, EmittedWrapper
from ..report import Report
from .bundle import ModelBundle, write_bundle
from backcountry_runtime.kernels import KernelSpec

_CLASS_LINE = re.compile(r"^class (\w+)\(", re.M)

_APPLY_SHIM = '''\
"""Load this artifact onto a freshly loaded build() model:

    from artifact.apply import apply
    model = apply(build())
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
sys.path.insert(0, str(_here / "runtime"))

from backcountry_runtime.apply import apply as _apply


def apply(model, strict=False):
    return _apply(model, _here, strict=strict)
'''


def write_kernel(directory: str | Path, spec: KernelSpec) -> None:
    """Keep editable bodies in .metal files, including each ordered stage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{spec.kernel_id}.metal").write_text(spec.source)
    launch = json.loads(spec.to_json())
    del launch["source"]
    if spec.stages:
        stage_dir = directory / f"{spec.kernel_id}.stages"
        stage_dir.mkdir(exist_ok=True)
        for i, stage in enumerate(launch["stages"]):
            (stage_dir / f"{i}.metal").write_text(stage["kernel"].pop("source"))
    (directory / f"{spec.kernel_id}.launch.json").write_text(json.dumps(launch, indent=1) + "\n")


def dedupe_wrappers(wrappers: Sequence[EmittedWrapper]) -> tuple[str, list[dict]]:
    """The exported patch module and swap table: one class per distinct body.

    Two wrappers share a class when their sources are identical once the class
    name is replaced, so the body shipped is byte for byte the body measured.
    The shared class is named from the kernel ids it carries, and a comment
    above it lists the module paths it serves.
    """
    groups: dict[str, list[EmittedWrapper]] = {}
    for w in wrappers:
        groups.setdefault(_CLASS_LINE.sub("class __CLASS__(", w.source, count=1), []).append(w)
    classes, table, names = [], [], set()
    for key, members in groups.items():
        if "class __CLASS__(" not in key:
            name, source = members[0].class_name, key
        else:
            ids = list(dict.fromkeys(k for w in members for k in w.kernel_ids))
            base = "W_" + "__".join(re.sub(r"\W", "_", k) for k in ids) if ids else members[0].class_name
            name, n = base, 2
            while name in names:
                name, n = f"{base}_{n}", n + 1
            paths = ", ".join(w.scope_path or "(root)" for w in members)
            source = key.replace("class __CLASS__(", f"# serves {paths}\nclass {name}(", 1)
        names.add(name)
        classes.append(source)
        # the span map says which recorded op each replayed line came from;
        # it is per module path, so it rides in the table, not the class
        table += [{"scope_path": w.scope_path, "wrapper_class": name, "kernel_ids": w.kernel_ids,
                   "span_map": [list(entry) for entry in w.span_map]}
                  for w in members]
    return MODULE_HEADER + "".join(classes), table


def emit_artifact(
    out_dir: str | Path,
    kernels: list[KernelSpec],
    wrappers: list[EmittedWrapper],
    report: Report,
    validate=None,
    bundle: ModelBundle | None = None,
    layers: dict | None = None,
) -> Path:
    """layers: scope path -> the replaced layer's apply.layer_signature, which
    apply() checks a user's model against before replacing it."""
    final = Path(out_dir)
    if final.exists() and not final.is_dir():
        raise FileExistsError(f"artifact destination is not a directory: {final}")
    # build beside the target and swap at the end, so a mid-emit failure can
    # never destroy an existing good artifact
    final.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=final.name + ".building-", dir=final.parent))
    try:
        (out / "kernels").mkdir(parents=True)
        (out / "patch").mkdir()
        (out / "buffers").mkdir()  # reserved for future precompute; empty in v1

        for spec in kernels:
            write_kernel(out / "kernels", spec)

        module, table = dedupe_wrappers(wrappers)
        for entry in table:
            if layers and entry["scope_path"] in layers:
                entry["module"] = layers[entry["scope_path"]]
        (out / "patch" / "wrappers.py").write_text(module)
        (out / "swap_table.json").write_text(json.dumps(table, indent=1) + "\n")

        baseline_wrappers = bundle.baseline_wrappers if bundle is not None else []
        if baseline_wrappers:
            baseline_module, baseline_table = dedupe_wrappers(baseline_wrappers)
            (out / "patch" / "baseline_wrappers.py").write_text(baseline_module)
            (out / "baseline_swap_table.json").write_text(json.dumps(baseline_table, indent=1) + "\n")

        runtime_src = Path(backcountry_runtime.__file__).parent
        runtime_dst = out / "runtime" / "backcountry_runtime"
        runtime_dst.mkdir(parents=True)
        for py in [*runtime_src.glob("*.py"), *runtime_src.glob("LICENSE-*")]:  # with third-party notices
            shutil.copy(py, runtime_dst / py.name)
        if any("GraphWrapper" in w.source for w in [*wrappers, *baseline_wrappers]):
            from backcountry_runtime.graph_native import prepare
            native_dst = runtime_dst / "graph_native"
            shutil.copytree(runtime_src / "graph_native", native_dst,
                            ignore=shutil.ignore_patterns("__pycache__", "*.so", "build.json", "prebuilt"))
            prepare(native_dst)  # ship the tested binary; inference needs no compiler

        (out / "apply.py").write_text(_APPLY_SHIM)
        (out / "__init__.py").write_text("from .apply import apply\n")
        report.write(out / "report.json")
        if report.manifest_path and Path(report.manifest_path).is_file():
            # the request as written: sweep, budget and baseline, not just the shapes timed
            shutil.copy(report.manifest_path, out / "manifest.yaml")

        if bundle is not None:
            write_bundle(out, bundle, table, report.to_dict(), package=final.name)
            (out / "__init__.py").write_text("from .apply import apply\nfrom .load import load\n")
        if report.model_level.get("source") and report.model_level.get("final", {}).get("ok"):
            write_model_run(out, report.model_level, package=final.name)
        if validate is not None:
            validate(out)

        previous = final.with_name(final.name + ".previous-" + uuid.uuid4().hex)
        had_previous = final.exists()
        if had_previous:
            final.rename(previous)
        try:
            out.rename(final)
        except BaseException:
            if had_previous:
                previous.rename(final)
            raise
        if had_previous:
            shutil.rmtree(previous)
        return final
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


_CHECK = '''
import importlib.util, json, sys
from pathlib import Path
import mlx.core as mx

mode = sys.argv[4]
apply_spec = importlib.util.spec_from_file_location("artifact_apply", sys.argv[2])
apply_mod = importlib.util.module_from_spec(apply_spec)
apply_spec.loader.exec_module(apply_mod)
spec_file = json.loads(Path(sys.argv[3]).read_text())
cases, context = spec_file["cases"], spec_file["context"]

def as_step(model):
    """The job's workload ran as a step over a filled KV cache: rebuild that step."""
    if metadata.get("app_run"):
        from backcountry_runtime.inference import AppRun
        return AppRun(model, mod.run, prefix_tokens=mx.load(context["tokens"])["tokens"] if context else None)
    if metadata.get("use_library_inference"):
        from backcountry_runtime.inference import LibraryInference
        tokens = mx.load(context["tokens"])["tokens"] if context else None
        return LibraryInference(model, steps=metadata["final_benchmark"]["steps"], prefix_tokens=tokens)
    if context is None:
        return model
    from backcountry_runtime.state import context_step  # apply.py put the vendored runtime on sys.path
    warm = mx.load(context["warm"])
    return context_step(model, context["context"], mx.load(context["tokens"])["tokens"],
                        [warm[f"i{k}"] for k in range(len(warm))], generated=context.get("generated", False))

def for_case(step, name):
    """The step a case's workload called: beside a conversation, a workload
    without one calls the same model from no cache."""
    if not context or not context.get("workload"):
        return step
    from backcountry_runtime.state import step_for
    return step_for(step, name.split("@")[0] == context["workload"].split("@")[0])

def check(model, how):
    for case in cases:
        inputs = mx.load(case["inputs"])
        outs = for_case(model, case["name"])(*[inputs[f"i{k}"] for k in range(len(inputs))])
        flat = []
        def walk(o):
            if isinstance(o, mx.array): flat.append(o)
            elif isinstance(o, (list, tuple)): [walk(v) for v in o]
            elif isinstance(o, dict): [walk(v) for v in o.values()]
        walk(outs)
        expected = mx.load(case["expected"])
        label = f"{case['name']} via {how}"
        assert len(flat) == len(expected), f"{label}: {len(flat)} outputs, expected {len(expected)}"
        for k, got in enumerate(flat):
            want = expected[f"o{k}"]
            assert got.shape == want.shape, f"{label}: output {k} shape changed"
            assert got.dtype == want.dtype, f"{label}: output {k} dtype changed"
            from backcountry_runtime.exact import bitwise_equal
            assert bitwise_equal(got, want), f"{label}: output {k} differs from the patched model"

load_path = Path(sys.argv[2]).with_name("load.py")
metadata_path = load_path.with_name("bundle.json")
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
source_weights = metadata.get("weight_policy") == "model_source"
if mode == "apply":
    from backcountry_runtime.checkpoints import bundle_checkpoint_pins, use_checkpoint_pins
    metadata_path = load_path.with_name("bundle.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    with use_checkpoint_pins(bundle_checkpoint_pins(metadata, load_path.parent)):
        model_path = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(model_path.parent))
        spec = importlib.util.spec_from_file_location("autotune_model", model_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        if source_weights:
            from backcountry_runtime.swap import copy_model, require_independent_models
            original = mod.build()
            try:  # a copy holds no second set of weights, even while it is made
                model = copy_model(original)
            except Exception:
                model = mod.build()
            require_independent_models(original, model)
            if hasattr(model, "update"):
                model.update(original.parameters())
            original = as_step(original)
            model = as_step(apply_mod.apply(model, strict=True))
        else:
            model = as_step(apply_mod.apply(mod.build(), strict=True))
    if metadata_path.exists() and json.loads(metadata_path.read_text())["baseline"] == "compiled":
        plain = model
        model = mx.compile(lambda *args: plain(*args))
    if source_weights:
        validate_spec = importlib.util.spec_from_file_location("artifact_validate", load_path.with_name("validate.py"))
        validate_mod = importlib.util.module_from_spec(validate_spec)
        validate_spec.loader.exec_module(validate_mod)
        from backcountry_runtime.state import correctness_call
        # Use plain execution here so the cache observation includes state.
        candidate = plain if metadata["baseline"] == "compiled" else model
        for case in cases:
            data = mx.load(case["inputs"])
            inputs = [data[f"i{k}"] for k in range(len(data))]
            golden_file = metadata.get("goldens", {}).get(case["name"])
            if golden_file:
                golden = mx.load(str(load_path.parent / golden_file))
                result = validate_mod.check_changing_outputs(
                    lambda: correctness_call(for_case(original, case["name"]), inputs),
                    lambda: correctness_call(for_case(candidate, case["name"]), inputs),
                    [golden[f"o{k}"] for k in range(len(golden))], case["name"])
            else:
                result = validate_mod.check_outputs(
                    lambda: correctness_call(for_case(original, case["name"]), inputs),
                    lambda: correctness_call(for_case(candidate, case["name"]), inputs), case["name"],
                    **validate_mod.policy_arguments(metadata))
            assert result["passed"], f"apply() differs from fresh original: {result}"
    else:
        check(model, "apply()")
else:
    # A separate isolated interpreter has never imported the original model
    # or its helper modules. Only the bundle supplies the definition.
    load_spec = importlib.util.spec_from_file_location("artifact_load", load_path)
    load_mod = importlib.util.module_from_spec(load_spec)
    load_spec.loader.exec_module(load_mod)
    if not source_weights:
        check(load_mod.load(), "load()")
    validate_spec = importlib.util.spec_from_file_location("artifact_validate", load_path.with_name("validate.py"))
    validate_mod = importlib.util.module_from_spec(validate_spec)
    validate_spec.loader.exec_module(validate_mod)
    results = validate_mod.validate(include_sequences=True)
    failed = [result for result in results if not result["passed"]]
    assert not failed, f"bundle violates its recorded correctness policy: {failed}"
print("__AUTOTUNE_ARTIFACT_OK__")
'''


def check_apply(artifact_dir: str | Path, model_path: str | Path, inputs: list[mx.array],
                expected: list[mx.array], timeout_s: float = 900.0) -> None:
    """Compatibility wrapper for one workload; see check_apply_many."""
    check_apply_many(artifact_dir, model_path, [("workload", inputs, expected)], timeout_s)


def bundle_check_timeout(artifact_dir, names, timeout_s: float) -> float:
    """Seconds the bundle's own validation may take: four observed arms, 3x
    cooling, plus 25% slack and startup. A call's time comes from the job-start
    clock, else from the final check's paired timings: over a range most
    workloads have only the latter, and without it LFM2.5's check got the
    900 s floor and timed out."""
    metadata = json.loads((Path(artifact_dir) / "bundle.json").read_text())
    report = json.loads((Path(artifact_dir) / "report.json").read_text())
    steps = metadata.get("final_benchmark", {}).get("steps", 10)
    clocks = report.get("step_ms", {})
    final = report.get("final", {})
    timed = {**final.get("fresh_timings", {}), **final.get("timings", {})}

    def call_ms(name):
        row = clocks.get(name, {})
        return max(float(row.get("before", 0)), float(row.get("after", 0))) or \
            float(timed.get(name, {}).get("median_baseline_ms", 0))
    known = [call_ms(name) for name in names]
    # a loop-only job clocks no steps: its final's calls stand in (Qwen3.8-27B's got the floor and timed out)
    level = report.get("model_level") or {}
    loop_ms = [float((part or {}).get(key) or 0) for part in (level.get("final") or {}, (level.get("final") or {}).get("prefill"))
               for key in ("default_ms", "best_ms")]
    longest = max(known + loop_ms, default=0.0)  # a workload timed nowhere (a serve size) counts as the longest
    work_s = sum(ms or longest for ms in known) / 1000
    return max(timeout_s, 120 + 20 * (steps + 1) * work_s)


def check_apply_many(
    artifact_dir: str | Path,
    model_path: str | Path,
    workloads: Sequence[tuple[str, list[mx.array], list[mx.array]]],
    timeout_s: float = 900.0,
    context: tuple[int, mx.array, list[mx.array]] | None = None,
) -> None:
    """Rebuild in a fresh process and check every workload and sweep, through
    apply() on the repo's model file and, when the artifact carries a bundle,
    through load() on the bundle's own copy of the model source.

    Source-based bundles compare freshly built original and patched models
    sharing identical weights under the recorded correctness rule. Legacy
    bundles also reproduce the saved job outputs on their fixed weights.
    context is (context, tokens, warm inputs) when the workload ran as a step
    over a filled KV cache; the child rebuilds that step around build().
    """
    if not workloads:
        raise ValueError("artifact validation requires at least one workload")
    metadata_path = Path(artifact_dir) / "bundle.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    source_weights = metadata.get("weight_policy") == "model_source"
    with tempfile.TemporaryDirectory() as tmp:
        cases = []
        for index, (name, inputs, expected) in enumerate(workloads):
            in_path = f"{tmp}/{index}_inputs.safetensors"
            out_path = f"{tmp}/{index}_expected.safetensors"
            mx.save_safetensors(in_path, {f"i{k}": t for k, t in enumerate(inputs)})
            if not source_weights:
                mx.save_safetensors(out_path, {f"o{k}": t for k, t in enumerate(expected)})
            cases.append({"name": name, "inputs": in_path, "expected": out_path})
        spec = {"cases": cases, "context": None}
        if context is not None:
            mx.save_safetensors(f"{tmp}/context.safetensors", {"tokens": context[1]})
            mx.save_safetensors(f"{tmp}/warm.safetensors", {f"i{k}": t for k, t in enumerate(context[2])})
            spec["context"] = {"context": context[0], "tokens": f"{tmp}/context.safetensors",
                               "warm": f"{tmp}/warm.safetensors",
                               "generated": bool(context[3]) if len(context) > 3 else False,
                               "workload": context[4] if len(context) > 4 else None}
        case_path = Path(tmp) / "cases.json"
        case_path.write_text(json.dumps(spec))
        modes = ["apply", "bundle"] if (Path(artifact_dir) / "load.py").exists() else ["apply"]
        for mode in modes:
            effective_timeout = timeout_s
            if mode == "bundle":
                effective_timeout = bundle_check_timeout(artifact_dir, [name for name, _, _ in workloads], timeout_s)
            import os
            environment = dict(os.environ)
            if mode == "bundle":
                environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                if metadata.get("weight_policy") != "model_source":
                    environment.update(HF_HOME=f"{tmp}/empty-hf", HF_HUB_CACHE=f"{tmp}/empty-hf/hub",
                                       TRANSFORMERS_CACHE=f"{tmp}/empty-hf/transformers")
            proc = subprocess.run(
                [sys.executable, "-I", "-c", textwrap.dedent(_CHECK), str(Path(model_path).resolve()),
                 str(Path(artifact_dir).resolve() / "apply.py"), str(case_path), mode],
                capture_output=True, text=True, timeout=effective_timeout, cwd=tmp, env=environment,
            )
            if proc.returncode != 0 or proc.stdout.strip().splitlines()[-1:] != ["__AUTOTUNE_ARTIFACT_OK__"]:
                raise RuntimeError(f"the artifact did not reproduce the patched model via {mode} "
                                   f"in a fresh process (exit code {proc.returncode}): "
                                   f"{proc.stderr.strip()[-2000:]}\n{proc.stdout.strip()[-1000:]}")


def time_in_fresh_process(
    artifact_dir: str | Path,
    model_path: str | Path,
    cases: Sequence[tuple[str, list[mx.array]]],
    runs: int = 10,
    timeout_s: float = 3600.0,
) -> dict:
    """Time workloads the job could not (too close to the GPU's memory limit
    with the job around them) the way a user would: a fresh process builds the
    model once and runtime compare() switches the artifact's layers in and out
    on that one copy of the weights, `runs` alternated calls each. A prompt
    workload reads its prompt to the first token through MLX-LM when the job
    used library inference. Returns {name: timing or {"error": why}}."""
    with tempfile.TemporaryDirectory() as tmp:
        spec = {"runs": runs, "cases": []}
        for index, (name, inputs) in enumerate(cases):
            path = f"{tmp}/{index}_inputs.safetensors"
            mx.save_safetensors(path, {f"i{k}": t for k, t in enumerate(inputs)})
            spec["cases"].append({"name": name, "inputs": path})
        (Path(tmp) / "timing.json").write_text(json.dumps(spec))
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", textwrap.dedent(_TIME), str(Path(model_path).resolve()),
                 str(Path(artifact_dir).resolve() / "apply.py"), str(Path(tmp) / "timing.json")],
                capture_output=True, text=True, timeout=timeout_s, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {name: {"error": f"the fresh process took over {timeout_s:g} s"} for name, _ in cases}
        marker = "__AUTOTUNE_TIMED__"
        line = next((line for line in reversed(proc.stdout.splitlines()) if line.startswith(marker)), None)
        if line is None:
            why = f"the fresh process failed (exit code {proc.returncode}): {proc.stderr.strip()[-500:]}"
            return {name: {"error": why} for name, _ in cases}
        return json.loads(line[len(marker):])


_TIME = '''
import importlib.util, json, statistics, sys
from pathlib import Path
import mlx.core as mx

apply_path = Path(sys.argv[2])
apply_spec = importlib.util.spec_from_file_location("artifact_apply", apply_path)
apply_mod = importlib.util.module_from_spec(apply_spec)
apply_spec.loader.exec_module(apply_mod)  # puts the vendored runtime on sys.path
from backcountry_runtime.checkpoints import bundle_checkpoint_pins, use_checkpoint_pins
from backcountry_runtime.compare import compare

spec = json.loads(Path(sys.argv[3]).read_text())
metadata_path = apply_path.with_name("bundle.json")
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
with use_checkpoint_pins(bundle_checkpoint_pins(metadata, apply_path.parent)):
    model_path = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(model_path.parent))
    mspec = importlib.util.spec_from_file_location("autotune_model", model_path)
    mod = importlib.util.module_from_spec(mspec)
    sys.modules[mspec.name] = mod
    mspec.loader.exec_module(mod)
    model = mod.build()
results = {}
for case in spec["cases"]:
    data = mx.load(case["inputs"])
    inputs = [data[f"i{k}"] for k in range(len(data))]
    if metadata.get("use_library_inference"):
        from backcountry_runtime.inference import LibraryInference
        fn = lambda m: LibraryInference(m, steps=1)(*inputs)  # the prompt read to its first token
    else:
        fn = lambda m: m(*inputs)
    try:
        c = compare(fn, model, apply_path.parent, runs=spec["runs"])
        results[case["name"]] = {
            "identical": c.identical, "replaced_layers": c.replaced, "runs": len(c.stock_s),
            "stock_ms": round(statistics.median(c.stock_s) * 1e3, 3) if c.stock_s else None,
            "optimized_ms": round(statistics.median(c.optimized_s) * 1e3, 3) if c.optimized_s else None,
            "speedup": round(c.speedup, 4) if c.speedup else None}
    except Exception as error:  # one size failing (out of memory) still reports the others
        results[case["name"]] = {"error": f"{type(error).__name__}: {str(error)[:400]}"}
    mx.clear_cache()
print("__AUTOTUNE_TIMED__" + json.dumps(results))
'''


_MODEL_LOOP = '''"""The inference loop the job found.{usage}
model is the model after apply() (or as built, when nothing else was
patched). The outputs match the original's for the same inputs: see README.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE / "runtime") not in sys.path:
    sys.path.insert(0, str(_HERE / "runtime"))

from backcountry_runtime.model_run import ModelRun, load_loop  # noqa: E402

_loop = load_loop((_HERE / "run.py").read_text(), "artifact_run")
ASSETS = {assets!r}  # the models beside it the loop reads, pinned to the revisions the job checked


def assets():
    """ASSETS with each one's local path, found in the Hugging Face cache by
    its pinned revision (nothing is downloaded)."""
    from backcountry_runtime.model_run import asset_path
    return [{{**a, "path": asset_path(a["repo"], a["revision"])}} for a in ASSETS]


TOKENIZER = {tokenizer!r}  # the model's tokenizer, pinned like ASSETS (a generate_step's setup may read it)


def pinned_tokenizer():
    """The pinned tokenizer, loaded from the Hugging Face cache, or None."""
    if TOKENIZER is None:
        return None
    from mlx_lm.utils import load_tokenizer
    from backcountry_runtime.model_run import asset_path
    return load_tokenizer(Path(asset_path(TOKENIZER["repo"], TOKENIZER["revision"])))


def model_file_run():
    """The model file's own run(), the loop the search started from, when the
    found loop still calls it (helpers["app_run"]); else None."""
    if "app_run" not in (_HERE / "run.py").read_text():
        return None
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("_artifact_load", _HERE / "load.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    return getattr(loader._definition(json.loads((_HERE / "bundle.json").read_text())), "run", None)


def model_run(model, *, steps={steps}, tokenizer=None, app_run=None):
    """The model called through the found loop; call it like the job's model.{history_doc}"""
    return ModelRun(model, _loop, {{"steps": steps, "tokenizer": tokenizer or pinned_tokenizer(),
                                   "app_run": app_run or model_file_run(),
                                   "conversation": {history}, "text": {text}, "assets": assets()}})
'''

# A model file's run(): the found one, called as that run() is.
_RUN = '''

def run(model, *inputs):
    """The model file's run(model, *inputs), found faster: the same arguments and outputs."""
    return model_run(model)(*inputs)
'''

# MLX-LM's generation: the found generate_step, and MLX-LM's own
# stream_generate and generate calling it.
_GENERATION = '''

def _drop_ins():
    from backcountry_runtime.generation import drop_ins
    return drop_ins(_loop.generate_step, _loop.setup, made_for={made_for!r},
                    helpers={{"assets": assets(), "tokenizer": pinned_tokenizer()}})


generate_step, stream_generate, generate = _drop_ins()
'''

# A named function the app calls (an engine's generation): the found one,
# answering for the original on the model it was made for.
_FUNCTION = '''

TARGET = {target!r}  # the function this loop answers for, "module:qualname"


def _answer():
    from backcountry_runtime.functions import drop_in, find_for, resolve
    return drop_in(resolve(TARGET)[2], _loop.function, _loop.setup, made_for={made_for!r},
                   helpers={{"assets": assets(), "tokenizer": pinned_tokenizer()}}, find=find_for(TARGET))


answer = _answer()  # call it as the original is called; backcountry.faster() finds it
'''

_FUNCTION_USAGE = """ The application's own function, found faster:

    import backcountry
    {name} = backcountry.faster({name})   # where the app imports {target}

It takes the same arguments and returns the same outputs. On the model it was
made for, with the libraries it was checked with, the found function runs;
anything else runs the original.
"""

_GENERATION_USAGE = """ MLX-LM's generation with it, called as MLX-LM's:

    from backcountry import apply, stream_generate   # or generate, generate_step
    model = apply(model)                                # the kernels, when the bundle has any
    for response in stream_generate(model, tokenizer, prompt, max_tokens=256, sampler=sampler):
        print(response.text, end="")

The same arguments, outputs and behavior as mlx_lm's own (its stream_generate
and generate, calling the found generate_step). A call that sets an argument
the job did not check (a quantized or rotating KV cache, input embeddings), or
a model other than the one it was made for, runs mlx_lm's own generate_step.
backcountry's functions find this loop in the local store; this module's
own generate_step, stream_generate and generate are the same functions, and
model_run() calls the loop as the job did.
"""

_RUN_USAGE = """ Called as the model file's run() is:

    from {pkg}.model_loop import run
    outputs = run(model, *inputs)

model_run(model, steps={steps}) is the same loop as the job called it.
"""


def _pins(model_level: dict) -> list[dict]:
    """The models beside this one the loop read, as the job described them to
    it (a loop picks one by its type), less the local path a bundle finds."""
    return [{k: v for k, v in a.items() if k != "path"} for a in model_level.get("assets") or []]


def write_model_run(out: Path, model_level: dict, package: str) -> None:
    """Ship the inference-loop search's run() (run.py), a loader (model_loop.py) and a
    README section saying what it is and how it was checked."""
    from backcountry_runtime.generation import is_generation
    source = model_level["source"]
    if model_level.get("written_as"):   # a marked function's rewrite: the app runs it only for that code
        from backcountry_runtime.tune import WRITTEN_AS
        head, _, rest = source.partition("\n")
        source = f"{head}\n{WRITTEN_AS}{model_level['written_as']}\n{rest}"
    (out / "run.py").write_text(source)
    history = bool(model_level.get("conversation"))
    pins = _pins(model_level)
    final = model_level["final"]
    from backcountry_runtime.functions import target_of
    target = target_of(model_level["source"])
    generation = target is None and is_generation(model_level["source"])
    pkg = package if package.isidentifier() else "artifact"
    usage = (_FUNCTION_USAGE.format(name=target.partition(":")[2].split(".")[-1], target=target) if target else
             (_GENERATION_USAGE if generation else _RUN_USAGE).format(pkg=pkg, steps=model_level.get("steps", 1)))
    (out / "model_loop.py").write_text(_MODEL_LOOP.format(
        usage=usage, steps=model_level.get("steps", 1),
        history=history, text=bool(model_level.get("text")), assets=pins, tokenizer=model_level.get("tokenizer"),
        history_doc=(" The input is the conversation so far (token ids,\n    [1, n]); it is read into "
                     "the model's memory, then `steps` tokens are generated." if history else ""))
        + (_FUNCTION.format(target=target, made_for=final.get("made_for")) if target else
           _GENERATION.format(made_for=final.get("made_for")) if generation else _RUN))
    cases = ", ".join(f"{c['context'] or 'the prompt'}/{c['steps']}" for c in final.get("cases", []))
    found = ("a new MLX-LM generate_step the job found: for every call, the same tokens and log-probabilities "
             "as mlx_lm's, produced differently. `model_loop.py` loads it and gives generate_step, "
             "stream_generate and generate, called as mlx_lm's (usage in its docstring)" if generation else
             "a new inference loop the job found: the same outputs for the same inputs, produced differently. "
             "`model_loop.py` loads it (usage in its docstring)")
    lines = ["", "## Inference loop", "",
             f"`run.py` is {found}.", "",
             f"- Measured against the job's default loop on {final.get('inputs')} inputs the search never saw"
             + (f", each generating {final['steps']} tokens after a {final['context']:,}-token conversation "
                "(the only conversation length timed)" if final.get("context") and final.get("steps") else "")
             + f": {final.get('speedup', 0):.3f}x ({final.get('default_ms', 0):.1f} ms vs "
             f"{final.get('best_ms', 0):.1f} ms, {final.get('n')} paired blocks).",
             f"- Outputs checked against the untouched model on fresh inputs at these conversation/reply "
             f"lengths ({cases}): "
             + ("every step's token and log-probabilities identical to mlx_lm's generate_step under greedy "
                "decoding, a seeded temperature/top-p sampler, a repetition penalty and a small "
                "prefill_step_size, with the same progress calls and cache afterwards. "
                if generation else "every token identical to plain greedy decoding. ")
             + "Calls that read several "
             "generated tokens at once are computed with one-token arithmetic per token (exact_rows in the "
             "runtime), which is what keeps them identical.",
             f"- Models used beside it: {', '.join(p['repo'] for p in pins) or 'none'}."]
    if final.get("prefill"):
        read = final["prefill"]
        size = f"a {read['tokens']:,}-token prompt" if read.get("tokens") else "a prompt"
        lines.append(f"- Reading {size} from no cache to its first token, the same generate_step: "
                     f"{read.get('speedup') or 0:.3f}x "
                     f"({read.get('default_ms', 0):.1f} ms vs {read.get('best_ms', 0):.1f} ms, {read.get('n')} paired "
                     "blocks); it may not be slower than mlx_lm's.")
    under = final.get("kernels_under_loop")
    if under:
        lines.append(f"- The kernels under this loop: {under.get('speedup') or 0:.3f}x (the same loop on the untouched "
                     f"model {under.get('untouched_ms', 0):.1f} ms vs the patched one {under.get('patched_ms', 0):.1f} ms, "
                     f"{under.get('n')} paired blocks).")
    write_readme(out, {"speedup": final.get("speedup", 0), "generation": generation,
                       "section": "\n".join(lines) + "\n"})


def write_readme(out: Path, loop: dict | None = None) -> None:
    """README.md rendered from the bundle's own record (bundle.json and
    report.json) and the loop it ships, if any: written again, never edited,
    when the loop is added or taken out."""
    from .bundle import render_readme
    readme = out / "README.md"
    if not (out / "bundle.json").is_file():  # no bundle: the loop's section alone
        readme.write_text(loop["section"].lstrip("\n") if loop else "")
        return
    metadata = json.loads((out / "bundle.json").read_text())
    report = json.loads((out / "report.json").read_text()) if (out / "report.json").is_file() else {}
    readme.write_text(render_readme(metadata, report, metadata.get("package", "artifact"), loop))


def drop_model_run(out: Path) -> None:
    """Take the inference loop out of a bundle (run.py, model_loop.py and its
    README section), leaving whatever else it ships."""
    for name in ("run.py", "model_loop.py"):
        (out / name).unlink(missing_ok=True)
    write_readme(out)


_RUN_CHECK = '''
import importlib.util, json, sys
from pathlib import Path
import mlx.core as mx
art, model_path, case_path, settings = Path(sys.argv[1]), sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
sys.path.insert(0, str(art / "runtime"))
from backcountry_runtime.model_run import ModelRun, load_loop
caught = [None]
try:
    import mlx_lm.utils as lm_utils
    original = lm_utils.load_tokenizer
    def load_tokenizer(*a, **k):
        caught[0] = original(*a, **k)
        return caught[0]
    lm_utils.load_tokenizer = load_tokenizer
except ImportError:
    pass
spec = importlib.util.spec_from_file_location("autotune_model", model_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.build()
apply_spec = importlib.util.spec_from_file_location("artifact_apply", art / "apply.py")
apply_mod = importlib.util.module_from_spec(apply_spec)
apply_spec.loader.exec_module(apply_mod)
model = apply_mod.apply(model, strict=True)
from backcountry_runtime.model_run import asset_path
assets = [{**a, "path": asset_path(a["repo"], a["revision"])} for a in settings.get("assets", [])]
# the tokenizer as the job gave it: only one the bundle pins
helpers = {"steps": settings["steps"], "tokenizer": caught[0] if settings.get("tokenizer") else None,
           "app_run": getattr(mod, "run", None),
           "conversation": bool(settings.get("conversation")), "text": bool(settings.get("text")), "assets": assets}
run = load_loop((art / "run.py").read_text(), "artifact_run")
from mlx.utils import tree_flatten
data = mx.load(case_path)
calls = settings.get("calls") or []
shipped = None
if run.__dict__.get("generate_step") is not None:
    # what a user calls: the bundle's own generate_step, as mlx_lm's callers call it
    loop_spec = importlib.util.spec_from_file_location("artifact_model_loop", art / "model_loop.py")
    loop_mod = importlib.util.module_from_spec(loop_spec)
    loop_spec.loader.exec_module(loop_mod)
    shipped = loop_mod.generate_step
    if not shipped.made_for(model):  # it would run mlx_lm's own: identical, and not the loop
        print(json.dumps({"ok": False, "where": "the bundle's generate_step does not take this model for the one "
                                                "it was made for"}))
        sys.exit(0)
    import warnings
    with warnings.catch_warnings(record=True) as failed:
        warnings.simplefilter("always")
        if not shipped.prepare(model):  # its setup, as on a user's first call
            print(json.dumps({"ok": False, "where": "the bundle's generate_step could not set up: "
                              + "; ".join(str(w.message) for w in failed)}))
            sys.exit(0)
shipped_run = None
if run.__dict__.get("target") is not None:
    # what a user calls: the bundle's answer for the app's function (faster()),
    # called through the model file's run() exactly as the job called the loop
    from backcountry_runtime.functions import answering
    loop_spec = importlib.util.spec_from_file_location("artifact_model_loop", art / "model_loop.py")
    loop_mod = importlib.util.module_from_spec(loop_spec)
    loop_spec.loader.exec_module(loop_mod)
    import warnings
    with warnings.catch_warnings(record=True) as failed:
        warnings.simplefilter("always")
        from backcountry_runtime.tune import is_marked
        # a marked function's loop is made for the model it reaches, not the replay holding it
        if not loop_mod.answer.prepare(model.parts[0] if is_marked(run.__dict__["target"]) else model):
            print(json.dumps({"ok": False, "where": "the bundle's function does not take this model or could "
                              "not set up: " + "; ".join(str(w.message) for w in failed)}))
            sys.exit(0)

    def shipped_run(model, *inputs, helpers):
        with answering(run.__dict__["target"], loop_mod.answer, helpers["app_run"]):
            return helpers["app_run"](model, *inputs)
# which parts of the model ran, each call's input shape: the bundle's loop must
# run as the job's did, or it quietly ran another way (a setup that failed where
# it ships, or the loop's own fallback, runs MLX-LM's: identical, not faster).
# Every layer is recorded, not just the model, since a loop may call the parts
# alone (a Qwen3.8 MTP loop did); a module outside the model of the same kind
# (a helper model's layer) is recorded by its kind.
def _first_difference(got, want):
    i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    return f"first difference at call {i}: {got[i] if i < len(got) else 'none'} vs {want[i] if i < len(want) else 'none'}"
# A layer apply() replaced is recorded, not what runs inside it: its kernel's
# own first-call work is the kernel's, checked by its own gates.
seen, _inside = [], []
_paths = {id(m): name or "model" for name, m in model.named_modules()}
_table = art / "swap_table.json"
_replaced = {e["wrapper_class"] for e in json.loads(_table.read_text())} if _table.is_file() else set()
for _cls in {c for m in model.modules() for c in type(m).__mro__ if "__call__" in c.__dict__}:
    def _counted(self, *a, _base=_cls.__dict__["__call__"], **k):
        if _inside:
            return _base(self, *a, **k)
        path = _paths.get(id(self), type(self).__name__)
        seen.append((path, tuple(a[0].shape) if a and hasattr(a[0], "shape") else None))
        if type(self).__name__ not in _replaced:
            return _base(self, *a, **k)
        _inside.append(path)
        try:
            return _base(self, *a, **k)
        finally:
            _inside.pop()
    _cls.__call__ = _counted
k = 0
while f"in{k}_0" in data:
    inputs = [data[f"in{k}_{j}"] for j in range(len([n for n in data if n.startswith(f"in{k}_")]))]
    call = calls[k] if k < len(calls) else {}
    step = ModelRun(model, run, {**helpers, "call": call})
    seen.clear()
    out = [v for _, v in tree_flatten(step(*inputs))]
    job_calls = list(seen)
    saved = [data[f"out{k}_{i}"] for i in range(len([n for n in data if n.startswith(f"out{k}_")]))]
    if len(out) != len(saved) or not all(mx.array_equal(a, b).item() for a, b in zip(out, saved)):
        print(json.dumps({"ok": False, "case": k}))
        sys.exit(0)
    if shipped_run is not None:
        seen.clear()
        got = [v for _, v in tree_flatten(ModelRun(model, shipped_run, {**helpers, "call": call})(*inputs))]
        if len(got) != len(saved) or not all(mx.array_equal(a, b).item() for a, b in zip(got, saved)):
            print(json.dumps({"ok": False, "case": k, "where": "the bundle's function"}))
            sys.exit(0)
        if seen != job_calls:
            print(json.dumps({"ok": False, "case": k, "where": f"the bundle's function made {len(seen)} layer "
                              f"calls, not as the job's loop did ({len(job_calls)}; {_first_difference(seen, job_calls)}): "
                              "it did not run as the job checked it"}))
            sys.exit(0)
    if shipped is not None:
        from backcountry_runtime.generation import call_arguments
        from backcountry_runtime.model_run import _read
        from mlx_lm.models.cache import make_prompt_cache
        tokens = inputs[0].reshape(1, -1)
        seen.clear()  # the read counted too, as in the job's calls
        if helpers["conversation"]:
            cache, prompt = _read(model, tokens), tokens[0, -1:]
        else:
            cache, prompt = make_prompt_cache(model), tokens[0]
        if call.get("seed") is not None:
            mx.random.seed(call["seed"])
        yields = list(shipped(prompt, model, max_tokens=settings["steps"], prompt_cache=cache,
                              **call_arguments(call)))
        got = [mx.array([int(t) for t, _ in yields], dtype=mx.int32), mx.stack([lp for _, lp in yields])]
        # the job's outputs are tokens, then logprobs
        if not all(mx.array_equal(a, b).item() for a, b in zip(got, saved[:2])):
            print(json.dumps({"ok": False, "case": k, "where": "the bundle's generate_step"}))
            sys.exit(0)
        if seen != job_calls:
            print(json.dumps({"ok": False, "case": k, "where": f"the bundle's generate_step made {len(seen)} layer "
                              f"calls, not as the job's loop did ({len(job_calls)}, read included; "
                              f"{_first_difference(seen, job_calls)}): it did not run as the job checked it"}))
            sys.exit(0)
    k += 1
# identical outputs alone pass a loop that quietly ran without its drafter
unused = [a["repo"] for a in assets if a["repo"] not in step._run.__dict__.get("assets_read", ())]
print(json.dumps({"ok": k > 0 and not unused, "cases": k, "unused_assets": unused}))
'''


def check_model_run(artifact_dir, model_path, case_path, model_level, timeout_s: float = 1800.0) -> None:
    """Load the bundle's run.py on a fresh build() (after apply()) in a fresh
    process and require it to reproduce the job's outputs exactly."""
    case_path = Path(case_path)
    if not case_path.is_file():
        raise RuntimeError(f"no saved outputs to check the artifact's run.py against: {case_path}")
    settings = {"steps": model_level.get("steps", 1), "conversation": bool(model_level.get("conversation")),
                "text": bool(model_level.get("text")),
                "assets": _pins(model_level), "calls": (model_level.get("final") or {}).get("check_calls"),
                "tokenizer": model_level.get("tokenizer")}
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}  # a run.py that downloads anything fails here
    proc = subprocess.run([sys.executable, "-c", _RUN_CHECK, str(Path(artifact_dir).resolve()), str(model_path),
                           str(case_path), json.dumps(settings)],
                          capture_output=True, text=True, timeout=timeout_s, env=env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    result = json.loads(lines[-1]) if lines else {}
    if proc.returncode != 0 or not result.get("ok"):
        raise RuntimeError(f"the artifact's run.py did not reproduce the job's outputs in a fresh process: "
                           f"{result or proc.stderr[-2000:]}")
