"""Failures observed during the FLUX preparation and delivery review."""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from backcountry.harness.e2e import preserving_check, run_e2e
from backcountry.harness.measure.clocks import loop_iterations, sample_group
from backcountry.harness.measure.session import Session, memory_state, page_in


@pytest.mark.parametrize("bad", [
    lambda x: x * float("nan"),
    lambda x: x[:, :1],
    lambda x: x.astype(mx.float16),
    lambda x: [],
    lambda x: [x],
])
def test_model_check_rejects_corrupt_output_contract(bad):
    x = mx.ones((2, 3))
    assert not preserving_check(lambda: x, lambda: bad(x), "w").passed


def test_model_check_keeps_small_outputs_independent_of_large_ones():
    reference = [mx.array([1e6]), mx.array([1.0])]
    corrupted = [reference[0], mx.array([2.0])]
    assert not preserving_check(lambda: reference, lambda: corrupted, "w").passed


def test_model_check_accepts_matching_nonfinite_patterns():
    x = mx.array([float("nan"), float("inf"), -float("inf"), 1.0])
    assert preserving_check(lambda: x, lambda: mx.array(x), "w").passed


def test_slow_region_does_not_force_twenty_repeats():
    passes = []

    def loop_for(n):
        def run():
            passes.append(n)
            return 0.2 * n + 0.001
        return run

    assert loop_iterations(lambda fn: fn(), loop_for) == 1
    assert sum(passes) < 10


def test_loop_sizing_subtracts_fixed_submission_cost():
    def size(overhead):
        return loop_iterations(lambda fn: fn(), lambda n: lambda: overhead + n * 0.00001)
    assert size(0.0001) == pytest.approx(size(0.010), abs=1)


def test_group_shares_model_arm_and_reverses_order():
    order = []

    class FakeSession:
        def fresh_chunk(self, fn): pass
        def warm_until_stable(self, fn): pass
        def timed(self, fn): return fn()
        def settle(self): pass
        def log(self, *args, **kwargs): pass
        def memory(self, where, **detail): return {"where": where, **detail, **memory_state()}

    def arm(name, cost):
        def run():
            order.append(name)
            return cost
        return run

    session = FakeSession()
    logged = []
    session.log = lambda kind, **row: logged.append((kind, row))
    samples = sample_group(session, {
        "model": arm("model", 1.5), "a": arm("a", 0.01), "b": arm("b", 0.02)}, pairs=4)
    assert order == ["model", "a", "b", "b", "a", "model"] * 2
    assert samples["model"] == (1500.0,) * 4
    # every raw sample and the memory state are kept, so a slow block can be diagnosed later
    (kind, row), = logged
    assert kind == "sample_group" and row["rows"]["a"] == [10.0] * 4
    state = {"mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb", "gpu_limit_gb", "footprint_gb", "compressed_gb",
             "free_gb", "swap_used_gb"}
    assert set(row["memory_after"]) == state
    assert set(row["memory_before"]) == state | {"where", "arms"}


def test_a_model_step_is_timed_warm_after_other_arms():
    """Pricing timed Gemma 4's decode step straight after region loops, where
    it read 140-185 ms against 35 ms right after another step, so every share
    came out 2.5x small. A model step (an arm with steps) now gets one
    untimed call when another arm ran just before it; loops do not."""
    order = []

    class FakeSession:
        def fresh_chunk(self, fn): pass
        def warm_until_stable(self, fn): pass
        def timed(self, fn): fn(); return 0.001
        def settle(self): pass
        def log(self, *args, **kwargs): pass
        def memory(self, where, **detail): return {}

    def model():
        order.append("model")
    model.steps = 1
    sample_group(FakeSession(), {"model": model, "a": lambda: order.append("a")}, pairs=4)
    # forward: warm + timed (after a); reverse: warm + timed (after a);
    # the next forward follows the model itself, so no warm call
    assert order == ["model", "model", "a", "a", "model", "model", "model", "a", "a", "model", "model"]


def test_a_group_holds_one_regions_sets_at_a_time_and_none_while_the_model_runs():
    """Every region of a pricing wave held its sets through the whole group:
    48 regions at 772 tokens put Qwen3.8-27B at 19.6 GB, past the GPU's 19.07
    GB, and the timed model pass timed out. Each region's sets now load for
    its own turn and are gone before the model or another region runs."""
    from backcountry.harness.measure.clocks import HeldSets
    events, loads = [], []

    class FakeSession:
        def fresh_chunk(self, fn): fn()
        def warm_until_stable(self, fn): fn()
        def timed(self, fn): fn(); return 0.001
        def settle(self): pass
        def log(self, *args, **kwargs): pass
        def memory(self, where, **detail): return {}

    regions = {name: HeldSets(lambda name=name: loads.append(name) or [{0: name}]) for name in ("a", "b")}

    def model():
        events.append(("model", tuple(n for n, h in regions.items() if h.sets is not None)))

    def make(name):
        return lambda sets: (lambda: events.append((name, sets[0][0])))

    arms = {"model": model}
    for name, held in regions.items():
        arms[f"{name}:library"] = held.arm(make(name))
        arms[f"{name}:link"] = held.arm(make(name))
    sample_group(FakeSession(), arms, pairs=4)
    assert all(open_ == () for kind, open_ in events if kind == "model")
    assert all(kind == got for kind, got in events if kind != "model")  # each arm ran on its own sets
    assert all(h.sets is None for h in regions.values())               # nothing left open
    # one load per turn, both arms of a region in one turn: warm-up, then each
    # forward pass (a, b) and reverse pass (b still open from its turn, then a)
    assert loads == ["a", "b"] + ["a", "b", "a"] * 2


def test_cooling_is_logged_before_sleep(tmp_path):
    path = tmp_path / "session.jsonl"
    seen = []
    def sleep(_seconds):
        seen.append(json.loads(path.read_text().splitlines()[-1])["kind"])
    session = Session(log_path=path, sleep=sleep, duty_idle_factor=3.0)
    session._debt_s = 2.0
    session.settle()
    assert seen == ["cooling"]
    assert session.idled_s == 6.0


def test_cooling_returns_mlx_cached_memory(tmp_path):
    """Freed buffers held for reuse pushed a 24 GB Mac into swap mid-pricing."""
    import mlx.core as mx
    x = mx.random.normal((64, 1024, 1024))
    y = x * 2
    mx.eval(y)
    del x, y
    assert mx.get_cache_memory() > 0
    cached_while_idle = []
    session = Session(log_path=tmp_path / "session.jsonl",
                      sleep=lambda _s: cached_while_idle.append(mx.get_cache_memory()))
    session._debt_s = 1.0
    session.settle()
    assert cached_while_idle == [0]
    cooling = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()
               if '"cooling"' in line][0]
    assert cooling["cache_cleared_gb"] > 0


def test_scheduled_cooling_returns_cached_memory_even_when_no_sleep_follows(tmp_path, monkeypatch):
    """Search comparisons schedule their cooldown; judge thinking can outlast
    it. The pool is a stand-in so other tests' GPU work cannot race it; the
    slept path above uses MLX's real pool."""
    pool = {"bytes": 3 * 10**9}
    monkeypatch.setattr(mx, "get_cache_memory", lambda: pool["bytes"])
    monkeypatch.setattr(mx, "clear_cache", lambda: pool.update(bytes=0))
    clock = [100.0]
    slept = []
    session = Session(log_path=tmp_path / "session.jsonl", sleep=slept.append, now=lambda: clock[0])
    session._debt_s = 1.0
    session.defer_settle()
    assert pool["bytes"] == 0
    clock[0] += 60.0  # the judge thought for longer than the 3 s cooldown
    session.wait_ready()
    assert slept == []
    scheduled = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()
                 if '"cooling_scheduled"' in line][0]
    assert scheduled["cache_cleared_gb"] == 3.0


def test_failed_artifact_validation_preserves_existing_artifact(tmp_path):
    from backcountry.harness.artifact.emit import emit_artifact
    from backcountry.harness.report import Report
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "keep.txt").write_text("previous verified result")
    def fail(_staged):
        raise RuntimeError("fresh-process output differs")
    with pytest.raises(RuntimeError, match="output differs"):
        emit_artifact(target, [], [], Report(manifest_path="fixture"), validate=fail)
    assert (target / "keep.txt").read_text() == "previous verified result"
    assert not target.with_name("artifact.building").exists()


def test_artifact_keeps_the_manifest_the_job_ran(tmp_path):
    from backcountry.harness.artifact.emit import emit_artifact
    from backcountry.harness.report import Report
    manifest = tmp_path / "job.yaml"
    manifest.write_text("model: m.py\nmodel_type: other\nsweep: {I: [64, 256]}\n")
    out = emit_artifact(tmp_path / "artifact", [], [], Report(manifest_path=str(manifest)))
    assert (out / "manifest.yaml").read_text() == manifest.read_text()


def test_mixed_dtype_compute_estimate_uses_each_ops_throughput():
    from backcountry.harness.measure.peaks import Peaks
    from backcountry.harness.regions.roofline import compute_time_ms
    from backcountry.harness.trace.types import TraceNode
    def op(seq, dtype, count):
        return TraceNode(seq, "mx.add", (0, 1), (2,), (),
                         (((count,), dtype),), {}, "@0", seq)
    nodes = [op(0, "float32", 1), op(1, "bfloat16", 1000000)]
    peaks = Peaks(100, {"float32": 1000, "bfloat16": 2000})
    assert compute_time_ms(nodes, peaks) == pytest.approx(1e-9 + 0.0005)


def test_e2e_times_every_declared_workload(monkeypatch):
    from backcountry.harness import e2e
    from types import SimpleNamespace
    timed_sizes = []
    def clock(session, baseline, candidate, pairs, **kwargs):
        timed_sizes.append(baseline().size)
        return SimpleNamespace(), True
    monkeypatch.setattr(e2e, "step_veto", clock)
    result = run_e2e(Session(), lambda x: x, lambda x: mx.array(x),
                     [("small", [mx.ones(2)]), ("large", [mx.ones(5)])])
    assert result.passed
    assert timed_sizes == [2, 5]
    assert set(result.workload_vetos) == {"small", "large"}


def test_failed_preparation_leaves_report_and_restores_tracer(tmp_path, monkeypatch):
    from backcountry.harness.loop import JobRunner
    manifest = tmp_path / "manifest.yaml"
    fixture = Path(__file__).parent / "fixtures" / "planted_win.py"
    manifest.write_text(f"model: {fixture}\nmodel_type: other\nworkloads:\n  - inputs: [{{shape: [4, 1024], dtype: float32}}]\n")
    runner = JobRunner(manifest, tmp_path / "work", lambda r: None)
    def broken():
        runner.tracer.install()
        raise RuntimeError("preparation failed")
    monkeypatch.setattr(runner, "load_model", broken)
    with pytest.raises(RuntimeError, match="preparation failed"):
        runner.run()
    report = json.loads((tmp_path / "work" / "report.json").read_text())
    assert report["session"]["status"] == "failed"
    assert report["session"]["stage"] == "loading model"
    assert runner.tracer.verify_restored() == []


def test_ship_compares_with_current_incumbent_on_every_workload():
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.e2e import E2EResult, WorkloadCheck
    from backcountry.harness.measure.clocks import comparison_from_samples
    runner = object.__new__(JobRunner)
    runner.model_ratio = 0.50  # an old ratio from another measurement window
    win = comparison_from_samples([100.0] * 8, [98.0] * 8)
    loss = comparison_from_samples([100.0] * 8, [101.0] * 8)
    check = WorkloadCheck("w", 0, 0, 0, 1, True)
    result = E2EResult([check], win, True, {"first": win, "second": win})
    assert runner._model_win(result)
    result.workload_vetos["second"] = loss
    assert not runner._model_win(result)


def test_region_clock_must_nominate_before_model_is_built(monkeypatch):
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner
    runner = object.__new__(JobRunner)
    def forbidden():
        raise AssertionError("a slower region should not pay for model validation")
    monkeypatch.setattr(runner, "_copy_incumbent", forbidden)
    assert not runner._bind_and_promote(None, None, SimpleNamespace(outcome="correct_slower"))


def test_symmetric_clock_subtracts_link_in_the_same_window():
    from backcountry.harness.measure.clocks import paired_means, comparison_from_samples
    # Four arms move together under linear drift. Their centered differences
    # must retain the true 2 ms win after removing the shared chain cost.
    rows = {name: [] for name in ("library", "candidate", "probe", "link")}
    costs = {"library": 12.0, "candidate": 10.0, "probe": 6.0, "link": 2.0}
    order = list(rows)
    for _ in range(8):
        for name in order:
            rows[name].append(costs[name] + sum(map(len, rows.values())) * 0.05)
        order.reverse()
    centered = {k: paired_means(v) for k, v in rows.items()}
    net = {k: [v - l for v, l in zip(centered[k], centered["link"])]
           for k in ("library", "candidate")}
    clock = comparison_from_samples(net["library"], net["candidate"])
    assert clock.median_baseline_ms == pytest.approx(10)
    assert clock.median_delta_ms == pytest.approx(2)
    assert clock.wins_by(0.1)


def test_buffer_limit_counts_scratch_and_mlx_metadata():
    from backcountry.harness.ladder.static_checks import buffer_count
    from backcountry_runtime.kernels import KernelSpec
    spec = KernelSpec("many", "many", ("in0", "in1"), tuple(f"tmp{i}" for i in range(28)),
                      "// in0_shape in0_strides in1_ndim")
    # MLX scans the source text, including comments. Scalar arrays receive no
    # shape/stride/ndim buffers even when those strings appear.
    assert buffer_count(spec, (2, 0)) == 32


def test_scaffold_repair_counts_toward_the_job_budget(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner, RegionRun
    runner = object.__new__(JobRunner)
    runner.work_dir = tmp_path  # where an operator's skip request would be
    runner.manifest = SimpleNamespace(budget_per_region=2, budget_total=1)
    runner._finish_requested = lambda: False
    runner.total_hypotheses = 0
    runner.log = SimpleNamespace(append=lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(runner, "_meta", lambda *args: {})
    monkeypatch.setattr(runner, "_note_lesson", lambda *args: None)
    monkeypatch.setattr(runner, "_ask_judge", lambda *args: (calls.append(1) or SimpleNamespace(kernel=None), None))
    run = RegionRun(SimpleNamespace(fingerprint="region", roofline=None))
    kernel = SimpleNamespace(kernel_id="scaffold")
    result = SimpleNamespace(failed_gate="compile", detail={}, region_ms=None, library_ms=None,
                             win_ms=None, sigma_ms=None, floor_ms=None)
    assert runner._judge_fix(run, None, kernel, result, "preserving") is None
    assert (runner.total_hypotheses, run.hypotheses) == (1, 1)
    assert runner._judge_fix(run, None, kernel, result, "preserving") is None
    assert calls == [1]


def test_a_replay_guard_treats_a_recorded_none_with_is(tmp_path):
    """work-2026-09-16-1332 crashed on its first region: a recurrent block
    takes cache=None on the prompt call and an array on the next, and the
    wrapper's guard compared with ==, which mlx refuses for array vs None.
    The guard must use `is None` and hand the array call to the original."""
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner, _load_class, _resolve
    from backcountry_runtime.state import correctness_call
    from backcountry_runtime.swap import install as swap_install

    model_file = tmp_path / "tiny_recurrent_gemma.py"
    model_file.write_text(
        "def build():\n"
        "    from mlx_lm.models.recurrent_gemma import Model, ModelArgs\n"
        "    return Model(ModelArgs(model_type='recurrent_gemma', attention_bias=False, conv1d_width=4,\n"
        "                           hidden_size=64, intermediate_size=128, logits_soft_cap=30.,\n"
        "                           num_attention_heads=4, num_hidden_layers=2, num_key_value_heads=1,\n"
        "                           rms_norm_eps=1e-6, rope_theta=10000., attention_window_size=64,\n"
        "                           vocab_size=64, block_types=['recurrent', 'attention']))\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"model: {model_file}\nmodel_type: other\nworkloads:\n  - name: prefill\n    inputs:\n"
                        "      - shape: [1, 12]\n        dtype: int32\n        low: 0\n        high: 64\n"
                        "budget: {per_region: 1, total: 1}\nfinal_benchmark: {steps: 1, pairs: 4, warmup_steps: 1}\n")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: None, session=Session())
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    trace = runner.traces["prefill"]
    scope = next(sc for sc in trace.scope_calls if sc.address == "model.model.layers.0.temporal_block.rg_lru@0")
    emitted = runner._emit_installation([(trace, scope, [])], "Id_rg_lru", replay=True)
    assert "cache is None" in emitted.source and "cache == None" not in emitted.source
    path = "model.model.layers.0.temporal_block.rg_lru"
    swap_install(runner.baseline_model, path, _load_class(emitted)(_resolve(runner.baseline_model, path), {}))
    # the prompt call replays; the generated token's call carries an array cache and runs the original
    correctness_call(runner.baseline_model, runner.tensors["prefill"])


def test_certification_blames_the_scope_that_ran_first():
    """work-2026-09-16-1332 spent 47 minutes settling delivery: one scope whose
    compiled output differed changed every scope after it, and the batch check
    blamed whichever mismatch sorted first by name, one per full-model pass,
    143 times. The earliest-executed mismatch is the one to blame."""
    import mlx.nn as nn
    from backcountry.harness.bind.certify import certify_identities

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_first = nn.Linear(4, 4)
            self.a_second = nn.Linear(4, 4)

        def __call__(self, x):
            return self.a_second(self.z_first(x))

    class Exact(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def __call__(self, x):
            return self.wrapped(x)

    class Off(Exact):
        def __call__(self, x):
            return self.wrapped(x) + 1

    model = Two()
    x = mx.ones((2, 4))
    wrappers = {"a_second": Exact(model.a_second), "z_first": Off(model.z_first)}  # name order blames a_second
    check = certify_identities(model, wrappers, [lambda: model(x)], repeated_calls=1)
    assert not check.ok and check.scope == "z_first", (check.scope, check.reason)


def test_a_trace_pins_no_array_versions():
    """work-2026-09-16-1332 traced a 500-token RecurrentGemma prefill in 109 s
    at a 59 GB peak: the recorder kept a value snapshot of every version of
    the scan's 5 MB output, 9,000 of them, and the library's mid-pass eval
    made them all resident. A trace needs shapes and ids only; a capture
    keeps snapshots for the boundary it came for and nothing else."""
    import mlx.nn as nn
    from backcountry.harness.trace import Tracer
    from backcountry.harness.regions.price import capture_boundaries

    class Scan(nn.Module):
        def __call__(self, x):
            y = mx.zeros_like(x)                 # 4 MB, rewritten in place per step
            h = mx.zeros((x.shape[2],))
            for t in range(x.shape[1]):
                h = x[:, t] * 0.5 + h
                y[:, t] = h
            mx.eval(y)                           # the library evaluates mid-pass
            return y * 2

    model = Scan()
    x = mx.ones((1, 200, 5000))
    tracer = Tracer()
    tracer.install()
    try:
        mx.eval(model(x))
        mx.reset_peak_memory()
        trace, _ = tracer.trace(model, [x])
        peak = mx.get_peak_memory()
        assert not tracer.recorder._by_aid and tracer.recorder.snapshot_seqs is None
        # a handful of 4 MB versions in flight, never two hundred
        assert peak < 20 * x.nbytes, f"trace peak {peak/1e6:.0f} MB for a {x.nbytes/1e6:.0f} MB array"
        setitems = [n for n in trace.nodes if n.op.endswith("__setitem__")]
        assert len(setitems) == 200
        # a capture keeps exactly the boundary it came for
        wanted = set(trace.nodes[-1].out_arrays) | set(setitems[3].out_arrays)
        got = capture_boundaries(tracer, model, [x], trace, wanted)
        assert set(got) == wanted and tracer.recorder.snapshot_seqs is None
    finally:
        tracer.uninstall()


def test_minimum_win_scales_with_the_step_up_to_the_fixed_floor():
    """A region win must save at least 1% of the step, never more than the
    30 us floor: a 44 us step asks for 0.44 us and leaves the three-sigma rule
    to decide, a 1.2 s step still asks for the full 30 us."""
    from backcountry.harness.loop import JobRunner, MIN_WIN_MS
    runner = JobRunner.__new__(JobRunner)
    runner.step_ms = {"tiny": 0.044, "flux": 1200.0, "edge": 3.0}
    assert runner._min_win_ms("tiny") == pytest.approx(0.00044)
    assert runner._min_win_ms("flux") == MIN_WIN_MS
    assert runner._min_win_ms("edge") == MIN_WIN_MS
    assert runner._min_win_ms("unclocked") == MIN_WIN_MS


def test_untimed_model_work_leaves_a_memory_line_in_both_logs(tmp_path):
    """A job that runs out of GPU memory dies as a Metal GPU timeout, which
    names no cause; the line before the work says where it stood."""
    session = Session(log_path=tmp_path / "session.jsonl")
    job_log = []
    session.on_memory = job_log.append
    mx.eval(mx.ones((256, 256)) @ mx.ones((256, 256)))
    session.off_clock(lambda: mx.eval(mx.ones((64,)) * 2), paced=False, where="capture prompt set 0")
    (row,) = job_log
    assert row["where"] == "capture prompt set 0"
    assert row["gpu_limit_gb"] > 0 and row["mlx_peak_gb"] >= row["mlx_active_gb"] >= 0
    lines = [json.loads(l) for l in (tmp_path / "session.jsonl").read_text().splitlines()]
    assert [l["where"] for l in lines if l["kind"] == "memory"] == ["capture prompt set 0"]


def test_a_memory_line_counts_the_whole_process_not_only_mlx_arrays():
    """Qwen3.8-27B's GPU timeouts came with MLX's own count flat at 15.42 GB;
    the process footprint and the machine's free memory are what can show a creep."""
    import mlx.core as mx
    before = memory_state()
    held = mx.ones((1 << 26,))  # 256 MB
    mx.eval(held)
    after = memory_state()
    assert after["footprint_gb"] - before["footprint_gb"] > 0.2
    assert after["free_gb"] is not None and after["free_gb"] > 0


def test_gpu_work_waits_for_compressed_weights_to_come_back_on_the_cpu(tmp_path, monkeypatch):
    """Qwen3.8-27B's job died four times on its first read after a candidate
    check: macOS had compressed 13 GB of its idle weights while the check's
    process ran, and restoring them from the GPU's side ran past Metal's timeout."""
    from backcountry.harness.measure import session as session_mod
    weights = mx.ones((1 << 20,))
    mx.eval(weights)
    shared = [weights, weights]  # two model copies sharing one buffer
    assert page_in(shared) == weights.nbytes

    session = Session(log_path=tmp_path / "session.jsonl")
    touched = []
    session.resident = lambda: touched.append(1) or shared
    compressed = {"gb": 0.1}
    monkeypatch.setattr(session_mod, "_compressed_gb", lambda: compressed["gb"])
    session.wait_ready()
    assert touched == []  # little compressed: the GPU restores it well inside its timeout
    compressed["gb"] = 13.0
    session.timed(lambda: mx.eval(weights * 2))  # every GPU call passes the gate
    assert touched == [1]
    (row,) = [json.loads(l) for l in (tmp_path / "session.jsonl").read_text().splitlines()
              if json.loads(l)["kind"] == "paged_in"]
    assert row["compressed_gb"] == 13.0 and row["touched_gb"] == round(weights.nbytes / 1e9, 3)


def test_a_process_reports_its_own_compressed_memory():
    from backcountry.harness.measure.session import _compressed_gb
    assert 0 <= _compressed_gb() < 64
