"""A kernel is measured against what its scope will actually run.

Graph delivery compiles the scope it installs into, and mx.compile fuses on
its own, so compiling alone can move the step. That effect is clocked and
reported by itself; the kernel's whole-model number is taken against the
scope compiled with no kernel, and the final speedup is split the same way.
"""

from pathlib import Path

import pytest

from backcountry_runtime.stats import comparison_from_samples

from backcountry.harness.loop import JobRunner, RegionRun
from backcountry.harness.measure.session import Session
from backcountry_runtime.graph import GraphWrapper
from backcountry_runtime.swap import resolve_value
from tests.test_install import WIN, elementwise

FIXTURE = Path(__file__).parent / "fixtures" / "planted_win.py"


def _runner(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {FIXTURE}
model_type: other
baseline: plain
workloads:
  - name: main
    inputs: [{{shape: [4, 1024], dtype: float32}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       clock_pairs=4, session=Session(sleep=lambda seconds: None))
    monkeypatch.setattr(runner, "_model_win", lambda e2e: all(c.passed for c in e2e.checks))
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    return runner


def test_delivery_is_planned_from_the_record_and_sets_every_library_arm(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    regions = {r.ops: r for r in runner.build_regions() if not r.rejected}
    whole = regions[("array.__mul__", "array.__add__", "mx.maximum", "array.__mul__",
                     "array.__add__", "mx.minimum", "array.__sub__", "array.__mul__")]
    part = regions[("mx.maximum",)]
    assert whole.delivery == {"main": "direct"}   # the kernel call alone, nothing to compile
    # A plain baseline runs the model uncompiled, and so does the scope that
    # delivers the kernel: the library arm is the plain ops users run.
    assert part.delivery == {"main": "replay"} and "plain baseline" in part.delivery_reasons["main"]
    assert part.library_arm("main", "plain") == "plain" and whole.library_arm("main", "plain") == "plain"
    assert whole.library_arm("main", "compiled") == "compiled"
    for region in (part, whole):
        region.p["main"] = region.t_orig_ms["main"] = region.t_rep_ms["main"] = 1.0
    runner.store.set_count = lambda *a, **k: 0
    kernel = elementwise("max_probe", "out0[i] = metal::max(in0[i], 0.0f);")
    assert runner._ladder_job(part, kernel, "preserving", True).baseline == "plain"
    assert runner._ladder_job(whole, kernel, "preserving", True).baseline == "plain"
    # a compiled baseline keeps graph insertion, and its library arm compiles
    runner.baseline = "compiled"
    compiled_part = next(r for r in runner.build_regions() if r.ops == ("mx.maximum",) and not r.rejected)
    assert compiled_part.delivery == {"main": "graph"} and not compiled_part.delivery_reasons
    assert compiled_part.library_arm("main", "plain") == "compiled"


def test_a_graph_kernel_is_clocked_against_its_compiled_scope(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    regions = runner.build_regions()
    region = next(r for r in regions if r.ops == ("mx.maximum",))
    kernel = elementwise("acct_max", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
    try:
        assert runner._bind_and_promote(RegionRun(region), kernel, WIN), runner.log.rows()[-1]
        assert isinstance(resolve_value(runner.model, "chain"), GraphWrapper)
        accepted = runner.report.accepted[-1]
        assert accepted["incumbent_compiled_scopes"] == ["chain"]
        assert accepted["timing_baseline"] == "incumbent_with_compiled_scopes"
        runner._final_check()
        split = runner.report.final["delivery"]
        assert split["compiled_scopes"] == ["chain"]
        assert set(split["timings"]["main"]) == {"plain_vs_compiled_identity", "compiled_identity_vs_patched"}
    finally:
        runner.tracer.uninstall()


def test_a_direct_kernel_is_clocked_against_the_plain_scope(tmp_path, monkeypatch):
    """A kernel that is a module's whole calculation deploys as the bare kernel
    call, so it is measured against the plain module it replaces."""
    from tests.test_sibling_bind import _runner as sibling_runner
    runner, definitions = sibling_runner(tmp_path, monkeypatch)
    region, kernel = definitions["front"]
    assert region.delivery == {"main": "direct"}
    try:
        assert runner._bind_and_promote(RegionRun(region), kernel, WIN), runner.log.rows()[-1]
        assert not isinstance(resolve_value(runner.model, "block.lin_in"), GraphWrapper)
        assert runner.report.accepted[-1]["incumbent_compiled_scopes"] == []
        # a bare kernel call in place of one multiply is a call-site tax on a
        # 20 us step, not a win, and the chained step clock resolves that; this
        # test is about the accounting split, so the final veto is stubbed
        monkeypatch.setattr("backcountry.harness.e2e.step_veto",
                            lambda *args, **kwargs: (comparison_from_samples([1.0] * 4, [0.9] * 4), True))
        runner._final_check()
        assert runner.report.final["delivery"] == {"compiled_scopes": [], "timings": {}}
    finally:
        runner.tracer.uninstall()


def test_a_plain_baseline_outside_a_cache_settles_nothing_before_the_search(tmp_path, monkeypatch):
    """Against a plain baseline, outside a filled cache, every scope delivers
    by replay, as it runs and as it ships: no scope is compiled, so there is
    no compiled identity to certify and no compiled clock before the search.
    Replay copies are certified when a kernel first goes into them."""
    runner = _runner(tmp_path, monkeypatch)
    regions = runner.build_regions()
    runner._clock_steps()
    runner._settle_delivery(regions)
    assert not runner.certified_scopes
    clocks = runner.report.baseline["clocks_ms"]["main"]
    assert clocks["plain"] > 0 and clocks["compiled"] is None and "compiled_vs_plain" not in clocks
    assert runner.report.baseline["choice"] == "plain" and not runner.report.baseline["compiled_available"]
    assert not runner.installed and runner.model is not runner.baseline_model


def test_a_scope_whose_compiled_identity_differs_settles_on_replay(tmp_path, monkeypatch):
    """A scope whose compiled copy is not bitwise the original moves its
    regions to replay before any clock runs. Compiling the qwen3.5 fixture's
    block changed its output under MLX 0.32.2 and no longer does under 0.32.4,
    so certification is made to report the change."""
    import backcountry.harness.loop as loop_mod
    from backcountry.harness.bind.certify import CertificationResult
    real = loop_mod.certify_identities

    def changed_block(model, wrappers, runs):
        if "model.model.layers.0" in wrappers:
            return CertificationResult(False, "output changed (compiled copy)", "model.model.layers.0")
        return real(model=model, wrappers=wrappers, runs=runs)

    monkeypatch.setattr(loop_mod, "certify_identities", changed_block)
    fixture = Path(__file__).parent / "fixtures" / "qwen_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
model_type: other
baseline: plain
use_library_inference: false
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 4], dtype: int32, high: 64}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       session=Session(sleep=lambda seconds: None))
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    regions = runner.build_regions()
    block = next(r for r in regions if r.ops == ("array.__add__",) and not r.rejected)
    assert block.delivery == {"prompt": "graph"}
    runner._settle_delivery(regions)
    assert "model.model.layers.0" in runner.replay_scopes
    assert block.delivery == {"prompt": "replay"} and "output changed" in block.delivery_reasons["prompt"]
    assert block.library_arm("prompt", runner.baseline) == "plain"


def test_a_parent_absorbing_an_installed_child_is_measured_against_the_child(tmp_path, monkeypatch):
    """Composition: the parent's incumbent is the parent compiled with the
    child's cut it absorbs and nothing else, so the new cut alone is measured."""
    from tests.test_sibling_bind import _runner as sibling_runner
    runner, definitions = sibling_runner(tmp_path, monkeypatch)
    try:
        for key in ("front", "fused"):
            assert runner._bind_and_promote(RegionRun(definitions[key][0]), definitions[key][1], WIN), \
                runner.log.rows()[-1]
        first, second = runner.report.accepted[-2:]
        assert first["incumbent_compiled_scopes"] == []   # a bare kernel call against the plain module
        assert second["incumbent_compiled_scopes"] == ["block"]
        assert isinstance(resolve_value(runner.model, "block"), GraphWrapper)
    finally:
        runner.tracer.uninstall()


def test_a_region_at_the_top_level_is_rejected_before_any_budget(tmp_path, monkeypatch):
    """Installation swaps a module in for its parent; ops the model runs in
    its own top-level call have no parent, so such a region is named as
    unsupported at discovery instead of failing every bind."""
    (tmp_path / "model.py").write_text(
        "import mlx.core as mx\nimport mlx.nn as nn\n\n"
        "class Inner(nn.Module):\n"
        "    def __call__(self, x):\n        return mx.maximum(x, 0.0)\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.inner = Inner()\n"
        "    def __call__(self, x):\n        return self.inner(x) * 2.0 + 1.0\n\n"
        "def build():\n    return Model()\n")
    (tmp_path / "manifest.yaml").write_text(
        f"model: {tmp_path / 'model.py'}\nmodel_type: other\nbaseline: plain\n"
        "workloads:\n  - name: main\n    inputs: [{shape: [8, 16], dtype: float32}]\n")
    runner = JobRunner(tmp_path / "manifest.yaml", tmp_path / "work", judge_factory=lambda _: None,
                       session=Session(sleep=lambda _: None))
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    viable = runner.build_regions()
    assert [r.ops for r in viable] == [("mx.maximum",)]
    stranded = [row for row in runner.report.stranded if "top level" in row["reason"]]
    assert sorted(tuple(row["ops"]) for row in stranded) == sorted([
        ("array.__mul__",), ("array.__add__",), ("array.__mul__", "array.__add__"),
        ("mx.maximum", "array.__mul__"), ("mx.maximum", "array.__mul__", "array.__add__")])


def test_library_inference_takes_the_compiled_baseline_and_kernels_compose_into_it(tmp_path, monkeypatch):
    """A compiled baseline under mlx-lm generation (asked for: the default
    is plain, as MLX-LM runs) is the model with its outermost compilable
    scopes compiled and empty: it is installed as the
    starting state, every kernel composes into it, every whole-model number
    is taken against it and nothing else, and the artifact carries it."""
    from backcountry.harness.ladder.gates import LadderResult
    fixture = Path(__file__).parent / "fixtures" / "llama_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
model_type: other
use_library_inference: true
baseline: compiled
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
final_benchmark: {{steps: 1, pairs: 4, warmup_steps: 1}}
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4,
                       session=Session(sleep=lambda seconds: None))
    monkeypatch.setattr(runner, "_model_win", lambda e2e: all(c.passed for c in e2e.checks))
    monkeypatch.setattr(runner, "_holdout", lambda *a: None)  # the region is never priced here
    # The wiring is under test, not this fixture's microsecond clocks: every
    # paired comparison reads as a resolved improvement.
    import backcountry.harness.e2e as e2e
    import backcountry.harness.loop as loop
    from backcountry.harness.measure.clocks import comparison_from_samples
    resolved = lambda *a, **k: comparison_from_samples([2.0] * 8, [1.0] * 8)
    monkeypatch.setattr(loop, "compare", resolved)
    monkeypatch.setattr(e2e, "compare", resolved)  # the whole-model checks' own clock too
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        regions = runner.build_regions()
        runner._clock_steps()
        runner._settle_delivery(regions)
        baseline = runner.report.baseline
        assert baseline["choice"] == "compiled" and baseline["requested"] == "compiled"
        clocks = baseline["clocks_ms"]["prompt"]
        assert clocks["plain"] > 0 and clocks["compiled"] > 0
        assert runner.step_ms["prompt"] == clocks["compiled"]
        settled = [r for r in runner.log.rows() if r["kind"] == "delivery_settled"][-1]
        scopes = settled["baseline_scopes"]
        assert scopes and set(runner.installed) == set(scopes) and not runner._kernels_installed()
        assert all(isinstance(resolve_value(runner.model, p), GraphWrapper) for p in scopes)
        region = next(r for r in regions if r.ops == ("array.__add__",) and not r.rejected)
        kernel = runner._build_scaffold(region)
        region.p["prompt"] = region.t_orig_ms["prompt"] = region.t_rep_ms["prompt"] = 1.0
        assert runner._bind_and_promote(RegionRun(region), kernel,
                                        LadderResult("tentative_ship", None, {}, .1, 1., .1, 0., [])), \
            runner.log.rows()[-1]
        assert runner._kernels_installed()
        # A replay parent absorbing a baseline scope bypasses it; the rest stay.
        carrying = [p for p in runner.installed if p in scopes
                    and kernel.kernel_id in getattr(resolve_value(runner.model, p), "KERNEL_IDS", ())]
        assert carrying, "the kernel composes into the compiled baseline's own wrapper"
        accepted = runner.report.accepted[-1]
        assert accepted["timing_baseline"] == "incumbent_with_compiled_scopes"
        assert set(accepted["incumbent_compiled_scopes"]) >= set(carrying)
        runner._final_check()
        assert "against_plain" not in runner.report.final and "delivery" not in runner.report.final
        artifact = runner._emit_artifact(tmp_path / "artifact")
        import json
        table = json.loads((artifact / "swap_table.json").read_text())
        assert {row["scope_path"] for row in table} >= set(carrying)
        metadata = json.loads((artifact / "bundle.json").read_text())
        assert set(metadata["baseline_scopes"]) == set(scopes)
        baseline_table = json.loads((artifact / "baseline_swap_table.json").read_text())
        assert all(not row["kernel_ids"] for row in baseline_table)
        assert set(runner.report.step_ms["prompt"]) >= {
            "baseline_tokens_per_second", "candidate_tokens_per_second"}
        # Reproduce the compiled baseline in a fresh process using only the bundle.
        import subprocess, sys
        code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import load
from backcountry_runtime.graph import GraphWrapper
from backcountry_runtime.swap import resolve_value
import json
root = Path(sys.argv[1])
metadata = json.loads((root / 'bundle.json').read_text())
original = load.load(patched=False)
baseline = load.load(patched=False, measurement_baseline=True, share_weights_with=original.model)
patched = load.load(share_weights_with=original.model)
for path in metadata['baseline_scopes']:
    assert not isinstance(resolve_value(original.model, path), GraphWrapper)
    scope = resolve_value(baseline.model, path)
    assert isinstance(scope, GraphWrapper) and not scope._specs
import validate
assert all(row['passed'] for row in validate.validate(original, baseline))
for path in metadata['baseline_scopes']:
    scope = resolve_value(baseline.model, path)
    assert not scope._graph_fallback_reason and not scope._graph_fallbacks
assert all(row['passed'] for row in validate.validate(original, patched))
# Old compiled-inference bundles must fail explicitly instead of timing eager.
metadata.pop('baseline_scopes')
(root / 'bundle.json').write_text(json.dumps(metadata))
try:
    load.load(patched=False, measurement_baseline=True)
except ValueError as error:
    assert 'compiled baseline scopes' in str(error)
else:
    raise AssertionError('missing compiled baseline was silently accepted')
"""
        child = subprocess.run([sys.executable, "-c", code, str(artifact)],
                               capture_output=True, text=True, timeout=120)
        assert child.returncode == 0, child.stdout + child.stderr

    finally:
        runner.tracer.uninstall()


def test_a_library_inference_job_saves_and_restores_its_compiled_baseline(tmp_path, monkeypatch):
    """The state a job saves before every attempt, on an mlx-lm job over a
    cache: it pickles with the compiled baseline's scopes installed, and a
    new runner restores it to the same installed scopes and the same
    outputs, without clocking anything again."""
    import backcountry.harness.loop as loop
    from backcountry.harness import resume as resume_mod
    from backcountry.harness.measure.clocks import comparison_from_samples
    fixture = Path(__file__).parent / "fixtures" / "llama_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
model_type: other
use_library_inference: true
baseline: compiled
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
final_benchmark: {{steps: 1, pairs: 4, warmup_steps: 1}}
""")
    monkeypatch.setattr(loop, "compare", lambda *a, **k: comparison_from_samples([2.0] * 8, [1.0] * 8))
    quiet = Session(sleep=lambda seconds: None)
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4, session=quiet)
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        regions = runner.build_regions()
        runner._clock_steps()
        runner._settle_delivery(regions)
        assert runner.baseline == "compiled" and runner.installed
        runner.pending_regions = [r for r in regions if not r.rejected]
        runner._search_state = {"ranked": [], "shipped": [], "run": None, "accepted_before": 0}
        runner._safe_point("after setup")
    finally:
        runner.tracer.uninstall()
    state = resume_mod.load(tmp_path / "work")
    assert state["where"] == "after setup" and state["compiled_baseline"]
    assert set(state["installed"]) == set(runner.installed)

    resumed = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4,
                        session=quiet, resume=True)
    try:
        resumed._restore()
    finally:
        resumed.tracer.uninstall()
    assert resumed.baseline == "compiled" and set(resumed.installed) == set(runner.installed)
    assert all(isinstance(resolve_value(resumed.model, p), GraphWrapper) for p in resumed.installed)
    assert resumed._compiled_baseline is not None and resumed.step_ms == runner.step_ms
    assert len(resumed.pending_regions) == len(runner.pending_regions)
    assert [r["kind"] for r in resumed.log.rows()][-1] == "resumed"


def test_a_decode_job_over_a_filled_cache_saves_and_restores(tmp_path, monkeypatch):
    """The same save and restore for one decode step over a filled KV cache:
    the step keeps state, so the baseline is plain, and the resumed runner
    rebuilds the cache from the same tokens and passes its output check."""
    import backcountry.harness.loop as loop
    from backcountry.harness import resume as resume_mod
    from backcountry.harness.measure.clocks import comparison_from_samples
    fixture = Path(__file__).parent / "fixtures" / "llama_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
model_type: other
workloads:
  - name: decode
    context: 5
    inputs: [{{shape: [1, 1], dtype: int32, high: 64}}]
""")
    monkeypatch.setattr(loop, "compare", lambda *a, **k: comparison_from_samples([2.0] * 8, [1.0] * 8))
    quiet = Session(sleep=lambda seconds: None)
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4, session=quiet)
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        regions = runner.build_regions()
        runner._clock_steps()
        runner._settle_delivery(regions)
        runner.pending_regions = [r for r in regions if not r.rejected]
        runner._search_state = {"ranked": [], "shipped": [], "run": None, "accepted_before": 0}
        runner._safe_point("after setup")
    finally:
        runner.tracer.uninstall()
    assert resume_mod.load(tmp_path / "work")["where"] == "after setup"
    resumed = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4,
                        session=quiet, resume=True)
    try:
        resumed._restore()
    finally:
        resumed.tracer.uninstall()
    assert resumed.baseline == runner.baseline and resumed.step_ms == runner.step_ms
    assert set(resumed.installed) == set(runner.installed)
    assert [r["kind"] for r in resumed.log.rows()][-1] == "resumed"


def test_consecutive_steps_are_checked_uncompiled_under_a_compiled_baseline(tmp_path, monkeypatch):
    """Compiling a chain of small ops changes its rounding by itself: on a
    RoPE step a third of the outputs, so a kernel exact against the model
    failed an exact check against the compiled model, and the job failed.
    The consecutive-step check reads the uncompiled models, as every other
    correctness check does; the compiled steps are only timed."""
    from dataclasses import asdict
    import backcountry.harness.loop as loop
    from backcountry.harness.e2e import WorkloadCheck
    fixture = Path(__file__).parent / "fixtures" / "planted_win.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
model_type: other
baseline: compiled
workloads:
  - name: main
    inputs: [{{shape: [64, 1024], dtype: float32}}]
final_benchmark: {{steps: 3, pairs: 4, warmup_steps: 1}}
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       session=Session(sleep=lambda seconds: None))
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    runner.baseline = "compiled"
    observed = []
    import backcountry_runtime.state as state
    original_observation = state.sequence_observation

    def observe(model, inputs, steps):
        observed.append(model)
        return original_observation(model, inputs, steps)
    monkeypatch.setattr(state, "sequence_observation", observe)
    monkeypatch.setattr(loop, "compare_sequences", lambda *a, **k: {
        "timing": asdict(comparison_from_samples([100.0] * 4, [90.0] * 4)), "win_confirmed": True, "speedup": 100 / 90})
    rows, result = runner._final_sequences([WorkloadCheck("main", 0, 0, 0, 1, True)])
    assert result.passed and result.checks[-1].name == "main:sequence"
    # each model's run is repeated, and every one of them is the plain module
    assert {id(m) for m in observed} == {id(runner.baseline_model), id(runner.model)} and len(observed) == 4
