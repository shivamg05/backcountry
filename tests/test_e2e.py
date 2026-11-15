"""The e2e floor, patched-vs-original, weight sharing, and the step veto."""

import importlib.util
from pathlib import Path

import mlx.core as mx

from backcountry.harness.bind.emit import MODULE_HEADER, Splice, emit_wrapper
from backcountry.harness.e2e import E2EResult, preserving_check, run_e2e, share_weights
from backcountry.harness.measure.session import Session
from backcountry.harness.trace import Tracer
from backcountry_runtime.kernels import KernelSpec
from backcountry_runtime.swap import install

import pytest

# whole jobs and live models: minutes, not seconds
pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"


def load_model():
    spec = importlib.util.spec_from_file_location("fixture_e2e", FIXTURES / "repeated_layers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def test_share_weights_by_identity():
    a, b = load_model(), load_model()
    shared = share_weights(a, b)
    assert shared >= 8  # 4 layers x (g, w)
    x = mx.random.normal((4, 16), key=mx.random.key(0))
    ya, yb = a(x), b(x)
    mx.eval(ya, yb)
    assert mx.array_equal(ya, yb).item()


def test_preserving_check_passes_identical_and_fails_sabotage():
    model = load_model()
    x = mx.random.normal((4, 16), key=mx.random.key(1))
    good = preserving_check(lambda: model(x), lambda: model(x), "same")
    assert good.passed and good.floor_max_abs == 0.0

    bad = preserving_check(lambda: model(x), lambda: model(x) + 0.05, "sabotaged")
    assert not bad.passed
    assert bad.max_abs > bad.allowance


def test_e2e_on_patched_model_sits_on_floor():
    """The real thing: a kernel-spliced model against a fresh baseline, weights
    shared, outputs on the floor, step veto under the paired discipline."""
    tracer = Tracer()
    tracer.install()
    try:
        baseline_model = load_model()
        patched_model = load_model()
        share_weights(baseline_model, patched_model)

        x = mx.random.normal((4, 16), key=mx.random.key(2))
        trace, _ = tracer.trace(patched_model, [x])
        add_node = next(
            n for n in trace.nodes
            if n.op == "array.__add__" and n.module_address == "layers.0@0"
        )
        kernel = KernelSpec(
            kernel_id="k_e2e_add", name="e2e_test_add",
            input_names=("a", "b"), output_names=("out",),
            source="uint i = thread_position_in_grid.x;\nout[i] = a[i] + b[i];",
            grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
            threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
            output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
            output_dtypes=("float32",),
        )
        splice = Splice(
            kernel=kernel, start_seq=add_node.seq, end_seq=add_node.seq,
            input_ids=tuple(add_node.in_arrays), output_ids=tuple(add_node.out_arrays),
        )
        scope = next(sc for sc in trace.scope_calls if sc.address == "layers.0@0")
        emitted = emit_wrapper(trace, scope, [splice], "E2eLayer0")
        ns = {}
        exec(compile(MODULE_HEADER + emitted.source, "<generated>", "exec"), ns)
        install(patched_model, "layers.0", ns["E2eLayer0"](
            patched_model.layers[0], {kernel.kernel_id: kernel}
        ))
    finally:
        tracer.uninstall()
        assert tracer.verify_restored() == []

    session = Session()
    result = run_e2e(
        session, baseline_model, patched_model,
        workloads=[("w", [x])], veto_pairs=8,
    )
    assert result.checks[0].passed, result.checks[0]
    assert result.checks[0].max_abs == 0.0  # elementwise add is order-preserving
    # the veto ran on the paired discipline; whether a lone custom add beats
    # the library's own add by the margin is a measurement, not a fixture
    # property, and a quiet machine can resolve it either way
    assert result.veto is not None and result.veto.median_baseline_ms > 0


def test_over_a_range_the_model_check_times_the_ends_and_the_weakest_size():
    """Ten search sizes would make every whole-model check ten timings; it
    times the shortest, the longest and the size the region clock found the
    kernel weakest at, and ships with no resolved loss and two resolved wins."""
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.manifest import InputSpec, Workload
    from backcountry_runtime.stats import comparison_from_samples
    base = Workload(inputs=(InputSpec(shape=(1, "T"), dtype="int32"),), name="p")
    fixed = [Workload(inputs=(InputSpec(shape=(1, n), dtype="int32"),), name=f"p@T={n}") for n in (42, 425, 2028)]
    runner = SimpleNamespace(report=SimpleNamespace(constants={"search_sizes": {"T": [42, 128, 425, 2028]}}),
                             manifest=SimpleNamespace(workloads=(base, *fixed), primary={"T": 128}))
    runner._sized_workloads = lambda: JobRunner._sized_workloads(runner)
    region = SimpleNamespace(workloads={"p", "p@T=42", "p@T=425", "p@T=2028"})
    row = lambda lib, k: {"library_ms": lib, "kernel_ms": k}
    detail = {"sizes": {"p": row(3.0, 2.0), "p@T=425": row(6.0, 5.9), "p@T=42": row(1.0, 0.5), "p@T=2028": row(9, 8)}}
    assert JobRunner._model_check_sizes(runner, region, detail) == ["p@T=42", "p@T=425", "p@T=2028"]
    # regions are priced at the shortest, the primary and the longest size
    assert JobRunner._priced_workloads(runner) == {"p@T=42", "p", "p@T=2028"}
    # the compiled clock's spread: both ends and the primary among them
    assert JobRunner._spread_sizes(runner, 3) == {"p@T=42", "p", "p@T=2028"}
    assert JobRunner._spread_sizes(runner, 4) is None  # no more than four sizes: all of them
    runner.report.constants = {}
    assert JobRunner._model_check_sizes(runner, region, detail) is None  # no range: every workload, as before
    assert JobRunner._priced_workloads(runner) == {"p", "p@T=42", "p@T=425", "p@T=2028"}

    faster = comparison_from_samples([10.0, 10.1, 10.0, 10.1], [9.0, 9.1, 9.0, 9.1])
    tie = comparison_from_samples([10.0, 10.2, 9.8, 10.1], [10.1, 9.9, 10.0, 10.2])
    slower = comparison_from_samples([9.0, 9.1, 9.0, 9.1], [10.0, 10.1, 10.0, 10.1])
    check = lambda vetos: E2EResult(checks=[], veto=next(iter(vetos.values())), workload_vetos=vetos,
                                    required=frozenset(vetos), min_wins=2)
    win = JobRunner._model_win.__get__(runner)
    assert win(check({"a": faster, "b": faster, "c": tie}))
    assert not win(check({"a": faster, "b": tie, "c": tie}))       # one win of three
    assert not win(check({"a": faster, "b": faster, "c": slower}))  # a resolved loss


def test_over_a_range_the_final_decides_with_eight_pairs():
    """Ten sizes, each timed with 8 pairs (more if the manifest asks), for
    direct calls too; a single size keeps its own count."""
    from types import SimpleNamespace
    from backcountry.harness.loop import FINAL_RANGE_PAIRS, SHIP_PAIRS, JobRunner
    runner = SimpleNamespace(use_library_inference=True,
                             manifest=SimpleNamespace(final_benchmark=SimpleNamespace(pairs=4)),
                             report=SimpleNamespace(constants={"search_sizes": {"T": [42, 128]}}))
    assert FINAL_RANGE_PAIRS == 8 and JobRunner._final_pairs(runner) == 8
    runner.use_library_inference = False
    assert JobRunner._final_pairs(runner) == 8
    runner.manifest.final_benchmark.pairs = 16
    assert JobRunner._final_pairs(runner) == 16
    runner.report.constants = {}
    assert JobRunner._final_pairs(runner) == SHIP_PAIRS
    runner.use_library_inference, runner.manifest.final_benchmark.pairs = True, 4
    assert JobRunner._final_pairs(runner) == 4  # one size: unchanged


def test_the_final_counts_only_differences_of_one_percent_and_needs_two_winning_sizes():
    """A total gain under 1% of the step is not worth shipping and a loss
    under 1% does not block; over a range two sizes must win."""
    from backcountry.harness.e2e import E2EResult
    from backcountry.harness.loop import JobRunner
    from backcountry_runtime.stats import comparison_from_samples
    from types import SimpleNamespace
    by = lambda share: comparison_from_samples([100.0] * 8, [100.0 * (1 - share) + d for d in (0.01, -0.01) * 4])
    runner = SimpleNamespace(manifest=SimpleNamespace(workloads=[SimpleNamespace(name=n) for n in "abc"]))
    def ships(**shares):
        final = E2EResult(veto=by(0.0), workload_vetos={n: by(s) for n, s in shares.items()},
                          min_wins=2, margin_share=0.01)
        return JobRunner._model_win(runner, final)
    assert ships(a=0.02, b=0.015, c=-0.005)       # two clear wins; a resolved 0.5% loss does not block
    assert not ships(a=0.02, b=0.005, c=0.0)      # a resolved 0.5% gain is not a win
    assert not ships(a=0.02, b=0.015, c=-0.02)    # a 2% loss blocks



def test_the_consecutive_steps_ship_under_the_same_rule_as_one_step():
    """Bonsai 2 9B's 16 consecutive steps at 1, 2 and 3 requests, as two runs
    measured them: a 1.5% win at three requests alone shipped kernels that ran
    LM Studio at 1.00x, so over a range two sizes must win by more than 1%, as
    in the one-step final."""
    from backcountry.harness.loop import JobRunner
    from backcountry_runtime.stats import comparison_from_samples
    from types import SimpleNamespace
    names = ["d", "d@B=2", "d@B=3"]
    runner = SimpleNamespace(report=SimpleNamespace(constants={"search_sizes": {"B": [1, 2, 3]}}),
                             manifest=SimpleNamespace(workloads=[SimpleNamespace(name=n) for n in names]))
    ships = lambda *runs: JobRunner._model_win(runner, JobRunner._sequence_result(
        runner, [], {n: comparison_from_samples(*r) for n, r in zip(names, runs)}))
    shipped_flat = (([672.7, 674.6, 672.2, 667.3], [658.1, 647.6, 657.5, 671.0]),
                    ([1337.6, 1323.6, 1321.7, 1328.0], [1306.0, 1318.8, 1310.8, 1313.3]),
                    ([1855.7, 1848.3, 1859.6, 1844.4], [1819.2, 1825.0, 1834.4, 1822.6]))
    faster = (([664.5, 664.7, 653.9, 655.1], [555.3, 551.6, 558.8, 554.7]),
              ([1315.0, 1314.8, 1319.8, 1313.5], [962.0, 986.4, 988.9, 983.5]),
              ([1822.5, 1820.9, 1825.9, 1823.4], [1254.0, 1244.8, 1255.0, 1239.5]))
    assert not ships(*shipped_flat)  # LM Studio 1.00x
    assert ships(*faster)            # LM Studio 1.20x at one request, 1.37x at two
    runner.report.constants, runner.manifest.workloads = {}, runner.manifest.workloads[2:]
    names[:] = names[2:]
    assert ships(shipped_flat[2])    # one workload, no range: its own win

def test_the_model_check_times_sizes_the_kernel_runs_not_ones_it_leaves_to_the_library():
    """A size the kernel declines (the size gate's stretch rule) would time
    the library against itself: the check's ends and weakest size come from
    the sizes the kernel runs."""
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.manifest import InputSpec, Workload
    base = Workload(inputs=(InputSpec(shape=(1, "T"), dtype="int32"),), name="p")
    fixed = [Workload(inputs=(InputSpec(shape=(1, n), dtype="int32"),), name=f"p@T={n}") for n in (42, 425, 900, 2028)]
    runner = SimpleNamespace(report=SimpleNamespace(constants={"search_sizes": {"T": [42, 128, 425, 900, 2028]}}),
                             manifest=SimpleNamespace(workloads=(base, *fixed), primary={"T": 128}))
    runner._sized_workloads = lambda: JobRunner._sized_workloads(runner)
    region = SimpleNamespace(workloads={"p", *(w.name for w in fixed)})
    row = lambda lib, k, declined=False: {"library_ms": lib, "kernel_ms": k, "declined": declined}
    detail = {"sizes": {"p@T=42": row(1.0, 1.0, True), "p": row(3.0, 2.0), "p@T=425": row(6.0, 5.9),
                        "p@T=900": row(8.0, 7.0), "p@T=2028": row(9, 8)}}
    assert JobRunner._model_check_sizes(runner, region, detail) == ["p", "p@T=425", "p@T=2028"]


def test_sizes_too_close_to_the_gpu_limit_are_checked_but_never_timed():
    """A workload whose run peaks within FIT_HEADROOM_GB of the GPU's limit
    (and every longer size of its workload) is left out of every whole-model
    timing: pricing, the spread, the model check and what a win requires.
    Qwen3.8-27B ran its job out of GPU memory six times timing 1,833 tokens
    on a 24 GB Mac."""
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner, _fits
    from backcountry.harness.manifest import InputSpec, Workload
    base = Workload(inputs=(InputSpec(shape=(1, "T"), dtype="int32"),), name="p")
    fixed = [Workload(inputs=(InputSpec(shape=(1, n), dtype="int32"),), name=f"p@T={n}")
             for n in (42, 425, 900, 1833, 2028)]
    runner = SimpleNamespace(report=SimpleNamespace(constants={"search_sizes": {"T": [42, 128, 425, 900, 1833, 2028]}}),
                             manifest=SimpleNamespace(workloads=(base, *fixed), primary={"T": 128}),
                             _untimed={"p@T=1833"})
    runner._sized_workloads = lambda: JobRunner._sized_workloads(runner)
    assert not _fits(runner, "p@T=1833") and not _fits(runner, "p@T=2028")  # longer: also too big
    assert not _fits(runner, "p@T=1900:fresh") and _fits(runner, "p@T=1500:fresh")
    assert _fits(runner, "p") and _fits(runner, "p@T=900") and _fits(runner, "reply")
    assert JobRunner._priced_workloads(runner) == {"p@T=42", "p", "p@T=900"}
    assert JobRunner._spread_sizes(runner, 3) == {"p@T=42", "p", "p@T=900"}
    # a job priced before the fit was measured keeps its priced sizes, less what does not fit
    runner.report.constants["priced_workloads"] = ["p", "p@T=42", "p@T=2028"]
    assert JobRunner._priced_workloads(runner) == {"p@T=42", "p"}
    del runner.report.constants["priced_workloads"]
    region = SimpleNamespace(workloads={"p", *(w.name for w in fixed)})
    row = lambda lib, k: {"library_ms": lib, "kernel_ms": k}
    detail = {"sizes": {"p@T=42": row(1.0, 0.9), "p": row(3.0, 2.0), "p@T=425": row(6.0, 5.9),
                        "p@T=900": row(8.0, 7.0), "p@T=1833": row(9, 8), "p@T=2028": row(9, 8)}}
    assert JobRunner._model_check_sizes(runner, region, detail) == ["p@T=42", "p@T=425", "p@T=900"]
    runner.tensors = {w.name: [] for w in (base, *fixed)}
    runner._step_fn = lambda model, tensors: model
    runner.model = "patched"
    assert set(JobRunner._timed_arms(runner, "untouched")) == {"p", "p@T=42", "p@T=425", "p@T=900"}


def test_memory_fit_measures_each_workload_once_and_caps_the_cache(monkeypatch):
    from types import SimpleNamespace
    import mlx.core as mx
    from backcountry.harness.loop import FIT_HEADROOM_GB, JobRunner
    peaks = {"p": 16.0, "p@T=1118": 17.6, "p@T=1833": 18.4}
    ran, capped = [], []
    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": 19.07e9})
    monkeypatch.setattr(mx, "get_peak_memory", lambda: peaks[ran[-1]] * 1e9)
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: capped.append(n / 1e9))
    monkeypatch.setattr(mx, "eval", lambda *a: None)
    workloads = [SimpleNamespace(name=n) for n in peaks]
    runner = SimpleNamespace(report=SimpleNamespace(constants={}), manifest=SimpleNamespace(workloads=workloads),
                             tensors={n: n for n in peaks}, model=None, log=SimpleNamespace(append=lambda *a, **k: None),
                             session=SimpleNamespace(off_clock=lambda fn, **kw: fn()), use_library_inference=False)
    runner._choose_baseline = lambda: ran.append("baseline chosen")  # before a step is sized or run
    runner._step_fn = lambda model, tensors: (lambda: ran.append(tensors))
    runner._sized_workloads = lambda: [(512, "p"), (1118, "p@T=1118"), (1833, "p@T=1833")]
    JobRunner._memory_fit(runner)
    fit = runner.report.constants["memory_fit"]
    assert ran.pop(0) == "baseline chosen"
    assert ran == list(peaks) and fit["untimed"] == ["p@T=1833"] and runner._untimed == {"p@T=1833"}
    assert capped[-1] == pytest.approx(19.07 - FIT_HEADROOM_GB - 17.6)  # the largest timed one keeps its line
    JobRunner._memory_fit(runner)  # a resumed process keeps the first measurement
    assert len(ran) == 3 and runner._untimed == {"p@T=1833"}
    runner.report.constants = {}
    peaks["p"] = 18.5  # the primary itself does not fit: say so before any timing
    ran.clear()
    runner._choose_baseline = lambda: None
    with pytest.raises(RuntimeError, match="cannot time the model's primary size"):
        JobRunner._memory_fit(runner)


def test_untimed_workloads_are_timed_after_export_in_a_fresh_process(monkeypatch, tmp_path):
    """What the job could not time is timed the way a user would, in a fresh
    process after export; a failure there is recorded, never raised."""
    from types import SimpleNamespace
    from backcountry.harness.artifact import emit
    from backcountry.harness.loop import JobRunner
    asked = []

    def fake(artifact_dir, model_path, cases, **kw):
        asked.append([name for name, _ in cases])
        return {"p@T=1833": {"speedup": 1.01, "identical": True}}

    monkeypatch.setattr(emit, "time_in_fresh_process", fake)
    written = []
    runner = SimpleNamespace(
        manifest=SimpleNamespace(workloads=[SimpleNamespace(name=n) for n in ("p", "p@T=900", "p@T=1833")],
                                 model_path="model.py"),
        _untimed={"p@T=1833"}, tensors={"p": [], "p@T=900": [], "p@T=1833": ["tokens"]},
        report=SimpleNamespace(final={"passed": True}, write=lambda path: written.append(path)),
        session=SimpleNamespace(wait_ready=lambda: None), work_dir=tmp_path,
        log=SimpleNamespace(append=lambda *a, **k: None))
    JobRunner._time_untimed_fresh(runner, tmp_path / "artifact")
    assert asked == [["p@T=1833"]] and runner.report.final["fresh_process_timings"]["p@T=1833"]["speedup"] == 1.01
    monkeypatch.setattr(emit, "time_in_fresh_process", lambda *a, **k: 1 / 0)
    JobRunner._time_untimed_fresh(runner, tmp_path / "artifact")
    assert "ZeroDivisionError" in runner.report.final["fresh_process_timings"]["p@T=1833"]["error"]
    runner._untimed = set()  # everything fit: nothing to do
    asked.clear()
    monkeypatch.setattr(emit, "time_in_fresh_process", fake)
    JobRunner._time_untimed_fresh(runner, tmp_path / "artifact")
    assert asked == []


def test_fresh_process_timing_reads_the_childs_result_or_says_why_not(monkeypatch, tmp_path):
    import subprocess
    from types import SimpleNamespace
    from backcountry.harness.artifact import emit
    out = SimpleNamespace(returncode=0, stdout='loading\n__AUTOTUNE_TIMED__{"p": {"speedup": 1.02}}\n', stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: out)
    assert emit.time_in_fresh_process(tmp_path, "model.py", [("p", [mx.zeros((1, 4), dtype=mx.int32)])]) \
        == {"p": {"speedup": 1.02}}
    out.stdout, out.returncode, out.stderr = "", 1, "RuntimeError: Insufficient Memory"
    result = emit.time_in_fresh_process(tmp_path, "model.py", [("p", [mx.zeros((1, 4), dtype=mx.int32)])])
    assert "Insufficient Memory" in result["p"]["error"]

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("python", 1)

    monkeypatch.setattr(subprocess, "run", slow)
    assert "took over" in emit.time_in_fresh_process(tmp_path, "model.py", [("p", [])], timeout_s=1)["p"]["error"]


def test_a_comparison_that_nears_the_gpu_limit_stops_its_size_being_timed():
    """A comparison holds more than one run: Qwen3.8-27B's single 1,118-token
    read peaked at 17.46 GB, its comparisons there at 19.02 GB, and the next
    ran out of GPU memory. The comparison's own peak decides."""
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner, _fits
    logged = []
    runner = SimpleNamespace(
        report=SimpleNamespace(constants={"memory_fit": {"gpu_limit_gb": 19.07, "headroom_gb": 1.0,
                                                         "untimed": ["p@T=1833"]}}),
        _untimed={"p@T=1833"}, session=SimpleNamespace(last_compare_peak_gb=17.9),
        log=SimpleNamespace(append=lambda kind, **row: logged.append((kind, row))))
    JobRunner._timing_memory(runner, "p@T=1118")  # under the 18.07 GB line: still timed
    assert _fits(runner, "p@T=1118") and not logged
    runner.session.last_compare_peak_gb = 19.02
    JobRunner._timing_memory(runner, "p@T=1118")
    assert not _fits(runner, "p@T=1118") and not _fits(runner, "p@T=1500:fresh") and _fits(runner, "p@T=662")
    fit = runner.report.constants["memory_fit"]
    assert fit["untimed"] == ["p@T=1118", "p@T=1833"] and fit["untimed_by_comparison"] == {"p@T=1118": 19.02}
    assert logged[0][0] == "memory_untimed"
    JobRunner._timing_memory(runner, "p@T=1118")  # already untimed: nothing more
    assert len(logged) == 1


def test_a_win_needs_every_required_size_timed_and_may_count_one_timed_before_it_stopped_fitting():
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner
    from backcountry_runtime.stats import comparison_from_samples
    faster = comparison_from_samples([10.0, 10.1, 10.0, 10.1], [9.0, 9.1, 9.0, 9.1])
    check = SimpleNamespace(passed=True)
    e2e = lambda timed, required: SimpleNamespace(checks=[check], veto=faster, workload_vetos=dict.fromkeys(timed, faster),
                                                  required=frozenset(required), min_wins=2, target_workload=None)
    runner = SimpleNamespace()
    assert JobRunner._model_win(runner, e2e(["a", "b", "c"], ["a", "b"]))   # c measured, then stopped fitting
    assert not JobRunner._model_win(runner, e2e(["a", "b"], ["a", "b", "c"]))  # c never timed


def test_a_comparisons_peak_is_its_own():
    import mlx.core as mx
    from backcountry.harness.measure.clocks import compare
    big = mx.zeros((64, 1024, 1024))  # 256 MB before the comparison
    mx.eval(big)
    del big
    s = Session(sleep=lambda seconds: None)
    compare(s, lambda: mx.zeros((4,)) + 1, lambda: mx.zeros((4,)) + 2, pairs=2)
    assert 0 <= s.last_compare_peak_gb < 0.25
