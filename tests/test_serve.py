"""manifest serve: after the search, shipped kernels run at every size in a
range they were checked right at, through generated code that reads the size
from its argument; everything else runs the original module."""

import mlx.core as mx
import pytest

from backcountry.harness.loop import JobRunner, RegionRun
from backcountry.harness.bind.sized import NotSizeGeneric, generalize
from backcountry.harness.manifest import SizeSet
from backcountry.harness.serve import served_range
from backcountry_runtime import kernels
from backcountry_runtime.kernels import KernelSpec
from tests.test_install import WIN, _runner, elementwise


@pytest.fixture(autouse=True)
def no_holdout(request, monkeypatch):
    """These runners install kernels into regions they never priced; the
    ship-time holdout runs the ladder, which needs a price, so it has its own
    test."""
    if "holdout" not in request.node.name:
        monkeypatch.setattr(JobRunner, "_holdout", lambda self, *a: None)


REAL_SERVE_GRID = JobRunner._serve_grid


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Row count L optimized at 64, served over 4..96. Elementwise stand-ins
    never beat the library on a clock, so the grid is told they win and the
    whole-model decision is forced, as the install tests do."""
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": True} for n in grid})
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}")
    yield r
    r.tracer.uninstall()


def _ship(runner, kernel):
    region = runner.region("mx.maximum")
    runner._capture([region])
    assert runner._bind_and_promote(RegionRun(region=region), kernel, WIN)


def _counting(monkeypatch):
    calls = []
    real = kernels.try_sized
    monkeypatch.setattr(kernels, "try_sized",
                        lambda spec, ins, **kw: calls.append(tuple(ins[0].shape)) or real(spec, ins, **kw))
    return calls


def test_a_kernel_runs_at_sizes_the_job_never_recorded(served, monkeypatch):
    _ship(served, elementwise("rmax_s1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    report = served.report.serve
    assert report["scopes"]["chain"]["sizes"] == [[4, 96]]
    assert report["kernels"]["rmax_s1"]["sizes"] == [[4, 96]]
    source = served.emitted["chain"].source
    assert "(ReplayWrapper)" in source
    assert "_kernels.try_sized(_s, _ins) if 4 <= _n <= 96 else None" in source

    calls = _counting(monkeypatch)
    for rows in (37, 5, 96):
        x = mx.random.normal((rows, 1024), key=mx.random.key(rows))
        assert mx.array_equal(served.model(x), served.baseline_model(x)).item()
    assert calls == [(37, 1024), (5, 1024), (96, 1024)]
    calls.clear()
    x = mx.random.normal((100, 1024))  # outside the range: the original module
    assert mx.array_equal(served.model(x), served.baseline_model(x)).item() and calls == []


def test_a_kernel_whose_operations_reshape_to_the_size_is_served_at_every_size(tmp_path, monkeypatch):
    """A region's recorded reshape to 2 x the row count differs at each size;
    that setting follows the size, so the sandbox check replays the operations
    with it set for each size, and the kernel serves the whole range."""
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": True} for n in grid})
    runner = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}",
                     fixture="sized_reshape.py")
    try:
        region = runner.region("mx.maximum", "array.reshape", "array.__mul__")
        runner._capture([region])
        kernel = KernelSpec(
            kernel_id="rmax_reshape", name="at_rmax_reshape", input_names=("in0",), output_names=("out0",),
            source="uint i = thread_position_in_grid.x;\nout0[i] = metal::max(in0[i], 0.0f) * 0.5f;",
            grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("256", "1", "1"),
            output_shapes=(("in0.shape[0] * 2", "512"),), output_dtypes=("float32",))
        assert runner._bind_and_promote(RegionRun(region=region), kernel, WIN)
        assert runner._serve_sizes()
        assert runner.report.serve["kernels"]["rmax_reshape"]["sizes"] == [[4, 96]]
        calls = _counting(monkeypatch)
        for rows in (37, 5, 96):
            x = mx.random.normal((rows, 1024), key=mx.random.key(rows))
            assert mx.array_equal(runner.model(x), runner.baseline_model(x)).item()
        assert calls == [(37, 1024), (5, 1024), (96, 1024)]
    finally:
        runner.tracer.uninstall()


def test_sizes_where_the_kernel_is_wrong_run_the_original(served, monkeypatch):
    # right from 16 rows up, wrong below: the sandbox check finds each size
    _ship(served, elementwise("rmax_s2", "out0[i] = (in0_shape[0] < 16) ? 1.0f : "
                                         "(in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    kernel = served.report.serve["kernels"]["rmax_s2"]
    assert kernel["sizes"] == [[16, 96]] and kernel["wrong_count"] == 12
    # the generated code serves only where the kernel runs: below 16 the original block runs
    assert served.report.serve["scopes"]["chain"]["sizes"] == [[16, 96]]

    calls = _counting(monkeypatch)
    for rows in (7, 16):
        x = mx.random.normal((rows, 1024), key=mx.random.key(rows))
        assert mx.array_equal(served.model(x), served.baseline_model(x)).item()
    assert calls == [(16, 1024)]


def test_a_kernel_wrong_only_on_a_later_copy_of_the_region_is_not_served_there(tmp_path, monkeypatch):
    """Three copies of one layer: the first sees inputs near zero, the others
    8 and up. A kernel wrong on large inputs at one unrecorded size matches
    the first copy there; each pass records the next copies too, so the check
    finds it and that size runs the original. (gpt-oss's expert kernel was off
    MLX's rounding at 124 rows on 9 copies in 10; the first copy matched.)"""
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": True} for n in grid})
    runner = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}",
                     fixture="repeated_chain.py")
    try:
        _ship(runner, elementwise("rmax_copies", "out0[i] = (in0_shape[0] == 37 && in0[i] >= 7.5f) ? 0.0f : "
                                                 "(in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
        assert runner._serve_sizes()
        kernel = runner.report.serve["kernels"]["rmax_copies"]
        assert kernel["wrong_count"] == 1 and "copy 2 of 3" in kernel["wrong"][37]
        assert all(37 not in range(a, b + 1) for a, b in kernel["sizes"])
        x = mx.random.normal((37, 1024), key=mx.random.key(37))
        assert mx.array_equal(runner.model(x), runner.baseline_model(x)).item()
    finally:
        runner.tracer.uninstall()


def test_a_final_lost_at_a_few_lengths_serves_every_other_one(served):
    """A loss at up to MAX_CUT_LENGTHS timed lengths cuts each, out to the
    nearest lengths that did not lose, and serves the rest without checking
    again; more losses leave nothing to cut, so the searched sizes ship."""
    _ship(served, elementwise("rmax_s4", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    restore = served._serve_sizes()
    timed = {"main": {}, "main@L=16": {}, "main@L=32": {}, "main@L=80": {}}
    served.report.final = {"timings": timed, "lost": ["main@L=32"]}
    cut = served._losing_spans()
    assert cut == [(17, 63)]  # between 16 and the primary 64, which did not lose
    served._unserve(restore, "lost at 32")
    assert served._install_served(*served._serve_plan, excluded=cut)
    assert served.report.serve["scopes"]["chain"]["sizes"] == [[4, 16], [64, 96]]
    assert "withdrawn" not in served.report.serve and served.report.serve["excluded"] == [[17, 63]]
    served.report.final = {"timings": {**timed, "main@L=48": {}}, "lost": ["main", "main@L=16", "main@L=32", "main@L=48"]}
    assert served._losing_spans() == []


def test_a_final_whose_outputs_changed_at_one_length_serves_every_other_one(served, monkeypatch):
    """Outputs changed at one checked length (gpt-oss at 32 tokens, where MLX's
    matmul rounds another way): that length, out to the nearest checked ones
    that held, goes back to the original and the whole final runs again. If
    it fails again, only the searched sizes are served."""
    _ship(served, elementwise("rmax_s7", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    names = ["main", "main@L=16", "main@serve:L=32", "main@L=80:fresh"]

    def finals(*outcomes):
        calls = []

        def final_check():
            wrong = outcomes[len(calls)]
            calls.append(served.report.serve.get("excluded"))
            served.report.final = {"checks": [{"name": n, "passed": n not in wrong} for n in names]}
            if wrong:
                raise RuntimeError("final model validation failed: the patched model's outputs changed")
            served.final_ok = True
        monkeypatch.setattr(served, "_final_check", final_check)
        return calls

    searched = served.emitted["chain"]
    calls = finals({"main@serve:L=32"}, {"main@L=16"}, set())
    served._final_with_serving(served._serve_sizes())
    assert len(calls) == 3 and served.final_ok
    assert "failed again" in served.report.serve["withdrawn"] and served.emitted["chain"] is searched

    calls = finals({"main@serve:L=32"}, set())
    served._final_with_serving(served._serve_sizes())
    assert calls == [None, [[17, 63]]] and served.final_ok
    assert served.report.serve["scopes"]["chain"]["sizes"] == [[4, 16], [64, 96]]
    assert "withdrawn" not in served.report.serve


def test_withdrawing_restores_the_searched_wrapper(served):
    _ship(served, elementwise("rmax_s3", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    searched = served.emitted["chain"]
    restore = served._serve_sizes()
    assert served.emitted["chain"] is not searched
    served._unserve(restore, "test")
    assert served.emitted["chain"] is searched and served.report.serve["withdrawn"] == "test"
    assert type(served.model.chain).__name__ == searched.class_name


def test_integers_that_follow_the_size_become_expressions():
    sources = [f"x.reshape(1, {n - 1}, 8)\ny = {n}\nz = 7\n" for n in (10, 20, 35)]
    source, varied = generalize(sources, [10, 20, 35])
    assert source == "x.reshape(1, (_n - 1), 8)\ny = _n\nz = 7\n" and varied == 2
    with pytest.raises(NotSizeGeneric, match="not one integer linear function"):
        generalize(["y = 1\n", "y = 4\n", "y = 9\n"], [1, 2, 3])
    with pytest.raises(NotSizeGeneric, match="different operations"):
        generalize(["y = f(1)\n", "y = f(2)\n", "y = g(3)(4)\n"], [1, 2, 3])


def test_the_served_range_follows_the_timed_wins_and_stops_at_a_loss():
    win, tie, loss = {"win": True}, {}, {"loss": True}
    grid = {100: tie, 400: win, 1000: win, 2048: win}
    assert served_range(list(range(50, 2049)), grid, 2047) == [(400, 2048)]
    assert served_range(list(range(500, 2049)), grid, 2047) == [(1000, 2048)]   # 400 is not right
    assert served_range(list(range(1, 2000)), grid, 2047) == []                 # not right where it shipped
    # a random sample slower than the library breaks the range; one that ties does not
    samples = {1500: loss, 1600: win, 1200: tie}
    assert served_range(list(range(50, 2049)), {**grid, **samples}, 2047, grid) == [(400, 1000), (1600, 2048)]
    assert served_range(list(range(50, 2049)), {**grid, 1200: tie}, 2047, grid) == [(400, 2048)]


def test_the_primary_size_earns_its_place_like_any_other():
    """The search no longer ships on the primary size alone, so a kernel the
    grid finds slower there is not served there."""
    win, loss = {"win": True}, {"loss": True}
    grid = {32: win, 128: loss, 512: win, 2048: win}
    assert served_range(list(range(32, 2049)), grid, 128, grid) == [(32, 32), (512, 2048)]
    assert served_range(list(range(32, 2049)), {**grid, 128: {}}, 128, grid) == [(32, 32), (512, 2048)]
    assert served_range(list(range(32, 2049)), {**grid, 128: win}, 128, grid) == [(32, 2048)]


def test_a_kernel_that_only_ties_at_the_primary_size_still_serves_where_it_wins(served, monkeypatch):
    """Qwen3.8-27B's second kernel tied at the job's own 511 rows and won at 21
    of 22 timed sizes; the walk checked only where it won, never the primary,
    and a kernel not right at the primary serves nowhere: it served nothing.
    The primary is always checked; the tie still keeps it off there."""
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": n != 64} for n in grid})
    _ship(served, elementwise("rmax_tie", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    sizes = served.report.serve["kernels"]["rmax_tie"]["sizes"]
    assert sizes and sizes[0][0] == 4 and sizes[-1][1] == 96
    assert all(not a <= 64 <= b for a, b in sizes)


def test_every_stretch_the_kernel_wins_is_served_not_only_the_one_around_the_primary():
    """gpt-oss: faster from 32 to 128 tokens, slower around 200-320, faster again above."""
    win, loss = {"win": True}, {"loss": True}
    grid = {128: win, 323: win, 512: win, 813: loss, 1290: loss, 2048: win, 5161: win, 8192: win}
    assert served_range(list(range(128, 8193)), grid, 512, grid) == [(128, 512), (2048, 8192)]


def test_generated_code_that_computes_something_else_is_never_served(served, monkeypatch):
    """The every-size check compares the generated code's calculation with the
    original module's; one changed constant fails it at every size."""
    import backcountry.harness.serve as serve
    _ship(served, elementwise("rmax_s4", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    searched = served.emitted["chain"]
    real = serve.plan_scopes

    def tampered(*args, **kwargs):
        plans, refused = real(*args, **kwargs)
        for plan in plans:
            assert "* 2.0)" in plan.sized.template
            plan.sized.template = plan.sized.template.replace("* 2.0)", "* 2.5)", 1)
        return plans, refused

    monkeypatch.setattr(serve, "plan_scopes", tampered)
    assert served._serve_sizes() is None
    scope = served.report.serve["scopes"]["chain"]
    assert scope["sizes"] == [] and scope["first_difference"][1] == "different leaf"  # the 2.5
    assert served.emitted["chain"] is searched


def _prompt_runner(tmp_path, extra=""):
    from pathlib import Path
    from backcountry.harness.measure.session import Session
    model = Path(__file__).parent / "fixtures" / "llama_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {model}
model_type: other
baseline: plain
use_library_inference: true
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 12], dtype: int32, high: 64}}]
final_benchmark: {{steps: 1, pairs: 4, warmup_steps: 1}}
{extra}""")
    return JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: None,
                     session=Session(sleep=lambda _: None))


def test_an_mlx_lm_prompt_run_checks_every_prompt_length_by_default(tmp_path):
    runner = _prompt_runner(tmp_path)
    try:
        runner.load_model()
        assert dict(runner.manifest.serve) == {"prompt_tokens": SizeSet(3, 2049)}
        assert runner.manifest.primary["prompt_tokens"] == 12
        runner.trace_workloads()
        # the optimized prompt is unchanged; the ends and middle of the range are recorded too
        assert runner.tensors["prompt"][0].shape == (1, 12)
        assert sorted(t[0].shape[1] for t in runner.serve_tensors.values()) == [3, 1026, 2049]
    finally:
        runner.tracer.uninstall()


def test_check_optimizations_for_false_keeps_the_single_size(tmp_path):
    runner = _prompt_runner(tmp_path, "check_optimizations_for: false\n")
    try:
        runner.load_model()
        runner.trace_workloads()
        assert not runner.manifest.serve and not runner.serve_tensors
    finally:
        runner.tracer.uninstall()


def test_exact_sizes_serve_only_those_sizes(tmp_path, monkeypatch):
    """A model whose sizes come from a few settings (image resolutions) lists
    them: each is recorded, checked and timed, and only those run the kernel."""
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": n != 32} for n in grid})
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: {sizes: [8, 32, 96]}}")
    try:
        _ship(r, elementwise("rmax_s5", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
        assert sorted(t[0].shape[0] for t in r.serve_tensors.values()) == [8, 32, 96]
        assert r._serve_sizes()
        report = r.report.serve
        # 32 was timed slower: the kernel leaves it, and so does the generated code (the original block runs)
        assert report["scopes"]["chain"]["sizes"] == [[8, 8], [64, 64], [96, 96]]
        assert report["kernels"]["rmax_s5"]["sizes"] == [[8, 8], [64, 64], [96, 96]]
        assert "_n in (8, 64, 96)" in r.emitted["chain"].source
        calls = _counting(monkeypatch)
        for rows in (8, 32, 37, 96):
            x = mx.random.normal((rows, 1024), key=mx.random.key(rows))
            assert mx.array_equal(r.model(x), r.baseline_model(x)).item()
        assert calls == [(8, 1024), (96, 1024)]  # 32 slower, 37 not listed: the original ops
    finally:
        r.tracer.uninstall()


def _tolerance_runner(tmp_path, monkeypatch):
    import textwrap
    from backcountry.harness.measure.session import Session
    from tests.test_install import FIXTURES
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    timed = []

    def grid(self, job, sizes):  # every timed size runs on inputs from a real pass at that size
        assert all(str(n) in job["recordings"] for n in sizes)
        timed.extend(sizes)
        return {n: {"win": True} for n in sizes}
    monkeypatch.setattr(JobRunner, "_serve_grid", grid)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {FIXTURES / "planted_win.py"}
        model_type: other
        workloads:
          - {{name: main, inputs: [{{shape: [L, 1024], dtype: float32}}]}}
          - {{name: short, inputs: [{{shape: [16, 1024], dtype: float32}}]}}
        primary: {{L: 64}}
        check_optimizations_for: {{L: [4, 96]}}
        baseline: plain
    """))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4,
                       session=Session(sleep=lambda s: None))
    runner.timed = timed
    return runner


@pytest.mark.parametrize("extra, served", [
    ("", [[4, 96]]),  # bit-exact though tagged within tolerance: no error anywhere
    # a small error everywhere, as at every searched size (the largest is 72 rows);
    # a larger one past row 80, which no whole-model check saw
    (" + 1e-7f + (i / 1024 >= 80 ? 1e-5f : 0.0f)", [[4, 80]]),
])
def test_a_kernel_within_tolerance_serves_where_its_error_stays_within_the_searched_sizes(
        tmp_path, monkeypatch, extra, served):
    """The search's whole-model checks ran at its sizes across the range (up to
    72 rows) and at 16. Elsewhere the kernel serves where its region error is
    no larger than the largest it had there, so a kernel exact everywhere
    serves the whole range."""
    runner = _tolerance_runner(tmp_path, monkeypatch)
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(g for g in runner.build_regions() if tuple(g.ops) == ("mx.maximum",))
        runner._capture([region])
        kernel = elementwise("rmax_t1", "out0[i] = ((in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f))"
                             + extra + ";")
        assert runner._bind_and_promote(RegionRun(region=region), kernel, WIN, assoc_tag="changing")
        assert runner._serve_sizes()
        report = runner.report.serve["kernels"]["rmax_t1"]
        assert report["sizes"] == served
        assert len(runner.timed) > 10 and min(runner.timed) == 4 and max(runner.timed) == 96
        recorded = runner.store._path("x", "serve:96", 0, "inputs").parent.parent.parent
        assert len(list(recorded.glob("*/serve%3A*/set0.inputs.safetensors"))) >= len(set(runner.timed)) - 1
        assert (report["allowed_error"] == 0.0) if not extra else 0 < report["allowed_error"] < 1e-6
        calls = _counting(monkeypatch)
        rows = (16, 37, 64, 96) if served == [[4, 96]] else (16, 37, 64, 80)
        for n in rows:
            runner.model(mx.random.normal((n, 1024), key=mx.random.key(n)))
        assert calls == [(n, 1024) for n in rows if n <= served[0][1]]
    finally:
        runner.tracer.uninstall()


def test_a_scope_whose_size_steps_by_more_than_one_serves_its_whole_checked_run():
    """A scope called with 4 rows per token (4 experts) is checked at every
    4th size, one per token count; the sizes between never occur, so the run
    around the primary steps from one checked size to the next."""
    from backcountry.harness.bind.sized import SizedWrapper
    from backcountry.harness.serve import ScopePlan
    sized = SizedWrapper("experts", "", [], "a0", 0, [512, 128, 4160, 8192], {})
    plan = ScopePlan("experts", "main", "experts@0", sized, [], {},
                     checked={n: None for n in range(128, 8193, 4)})
    assert plan.served(False) == tuple(range(128, 8193))
    plan.checked[1000] = "the replay computes something else"
    assert plan.served(False) == tuple(range(128, 997))


def test_a_float_region_is_checked_at_every_size_on_the_first_rows_of_its_longest_real_pass(served, monkeypatch):
    """A region whose inputs are all floating point is checked at every size
    without running the model there: each size not recorded is checked on the
    first rows of the longest recording (a causal model's first n rows are
    what n tokens compute). Qwen3.8-27B's real pass per length took 13 hours
    per kernel. Only the timing points run the model, and none is left on disk
    beyond them."""
    checked, real = [], JobRunner._serve_correct
    walked = []
    monkeypatch.setattr(JobRunner, "_serve_recordings", lambda self, *a, real=JobRunner._serve_recordings:
                        walked.append(a[-1]) or real(self, *a))

    def check(self, job, sizes, errors=None):
        assert all(str(n) in job["recordings"] or n <= job["cut_from"][0] for n in sizes)
        assert all("weight" not in p for p in job["recordings"].values())
        checked.extend(sizes)
        return real(self, job, sizes, errors)
    monkeypatch.setattr(JobRunner, "_serve_correct", check)
    _ship(served, elementwise("rmax_s6", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    assert sorted(set(checked)) == list(range(4, 97))
    report = served.report.serve["kernels"]["rmax_s6"]
    assert report["checked"] == 93 and report["checked_range"] == [4, 96] and report["sizes"] == [[4, 96]]
    assert report["cut_from"] == 96
    assert len(walked) == 2  # the timing grid and its samples, nothing per size
    left = {int(p.parent.name.split("%3A")[1]) for p in served.store.root.glob("*/serve%3A*/set0.inputs.safetensors")}
    assert left == {n for row in report["recorded"] for n in range(row[0], row[1] + 1)}  # timing recordings only


def test_a_region_with_an_integer_input_is_never_cut(tmp_path):
    """A cut keeps an index list's values, which can point past it: on
    Qwen3-30B-A3B the unsort order kept an index of 16,254 among 1,032 rows and
    a correct kernel wrote out of bounds. A region with any integer input but
    a weight (a 4-bit matrix is uint32) gets a real pass at every size."""
    from types import SimpleNamespace
    floats = {"recordings": {"8": "a", "96": "b"}, "weight_inputs": (False, True),
              "shapes": [[[[1, 0], 16], "float16"], [[16, 16], "uint32"]]}
    assert JobRunner._cut_source(floats) == [96, "b"]
    indices = {**floats, "shapes": [[[[1, 0], 16], "float16"], [[[1, 0]], "mlx.core.int32"]],
               "weight_inputs": (False, False)}
    offsets = {**indices, "shapes": [[[[1, 0], 16], "float16"], [[9], "uint32"]]}  # fixed shape, row values
    assert JobRunner._cut_source(indices) is None and JobRunner._cut_source(offsets) is None

    walked = []

    def recordings(region, fingerprint, member, template, values):
        walked.extend(values)
        for v in values:
            (tmp_path / f"r{v}").write_bytes(b"x")
        return {v: (v, str(tmp_path / f"r{v}")) for v in values}
    runner = SimpleNamespace(SERVE_BATCH=4, SERVE_RECORD_BYTES=1 << 20, _serve_recordings=recordings,
                             _cut_source=JobRunner._cut_source,
                             _serve_correct=lambda job, sizes, errors=None: (set(sizes), {}))
    correct, wrong, cut = JobRunner._serve_every_size(runner, indices, None, "f", None, None,
                                                      list(range(1, 11)), [1, 10], None, {})
    assert correct == set(range(1, 11)) and cut is None and set(walked) == set(range(1, 11))
    walked.clear()
    correct, wrong, cut = JobRunner._serve_every_size(runner, floats, None, "f", None, None,
                                                      list(range(1, 11)), [1, 10], None, {})
    assert correct == set(range(1, 11)) and cut == 96 and walked == []


def test_shader_validation_runs_the_kernel_once_for_its_reports_and_outputs_are_judged_without_it(served, monkeypatch):
    """What runs under shader validation is not what ships: that pass runs the
    kernel once per size, on the first copy where no input is an index, for its
    out-of-bounds reports alone. Without it, every copy at every size is
    compared with the recorded operations, and three runs at SERVE_REPEATS
    sizes. Checked in full both ways, Qwen3.8-27B's down projection took 2.3 h."""
    import dataclasses
    from backcountry.harness.ladder.sizes import evaluate_sizes
    from backcountry.harness.sandbox import protocol
    specs, real = [], protocol.run_job
    monkeypatch.setattr(protocol, "run_job", lambda spec, mode, **kw: specs.append((mode, spec)) or real(spec, mode, **kw))
    _ship(served, elementwise("rmax_s9", "out0[i] = (in0_shape[0] < 16) ? 1.0f : "
                                         "(in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    assert served.report.serve["kernels"]["rmax_s9"]["sizes"] == [[16, 96]]
    checks = [(mode, spec) for mode, spec in specs if spec.phase == "validate"]
    assert {mode for mode, _ in checks} == {"validate", "score"}
    for mode, spec in checks:
        if mode == "validate":
            assert spec.addresses_only and spec.copies == 1
        else:
            assert not spec.addresses_only and spec.copies == 0
            assert len(spec.repeat_at) <= served.SERVE_REPEATS + 1
    # the validation pass compares nothing: a size the kernel is wrong at passes there
    spec = next(spec for mode, spec in checks if mode == "score" and 8 in spec.sizes)
    assert evaluate_sizes(dataclasses.replace(spec, sizes=(8,), addresses_only=True)).detail["results"] == {"8": None}
    assert "differs" in evaluate_sizes(dataclasses.replace(spec, sizes=(8,))).detail["results"]["8"]


def test_a_validation_report_names_the_size_it_came_from():
    """A worker checking many sizes marks each on its stderr, so one bad size
    does not take the whole batch with it; a report before any marker does."""
    from backcountry.harness.sandbox.protocol import SIZE_MARKER, Verdict, _merge_validation
    line = "Invalid device store at offset 65248, executing kernel function"
    verdict = Verdict(True, None, ("sizes",), {})
    _merge_validation(verdict, "\n".join([f"{SIZE_MARKER}1032", f"{SIZE_MARKER}1040", line, line,
                                          f"{SIZE_MARKER}2000"]))
    assert verdict.detail["validation"]["by_size"] == {"1040": 2}
    assert verdict.detail["validation"]["unattributed"] == 0
    early = Verdict(True, None, ("sizes",), {})
    _merge_validation(early, "\n".join([line, f"{SIZE_MARKER}8"]))
    assert "by_size" not in early.detail["validation"]


def test_a_stretch_steps_over_sizes_no_pass_produces_and_stops_at_a_wrong_one():
    """A layer seeing 8 rows per token is checked at every 8th size; with the
    wrong sizes given, the sizes between need not be checked, but a wrong one
    still breaks the stretch."""
    grid = {256: {"win": True}, 1024: {"win": True}, 4096: {"win": True}}
    correct = list(range(256, 4097, 8))
    # the old rule wants every size between checked: only the timed points serve
    assert served_range(correct, grid, 1024, grid) == [(256, 256), (1024, 1024), (4096, 4096)]
    assert served_range(correct, grid, 1024, grid, wrong={}) == [(256, 4096)]
    assert served_range([n for n in correct if n != 2000], grid, 1024, grid,
                        wrong={2000: "differs"}) == [(256, 1024), (4096, 4096)]


def test_a_scope_with_several_rows_per_token_still_maps_to_tokens():
    """gpt-oss's expert layers see 4 rows per token: tokens = rows / 4. Without
    the map the serve stage cannot run the model at a size, so times nothing."""
    from backcountry.harness.serve import _dim_fit, in_dim
    fit = _dim_fit([128, 32, 1040, 2048], [512, 128, 4160, 8192])
    assert fit == (0.25, 0)
    assert in_dim(fit, range(128, 8193)) == [[32, 2048]]  # only whole token counts exist
    assert _dim_fit([12, 3, 2049], [12, 3, 2049]) == (1, 0)
    assert _dim_fit([1, 2, 5], [1, 2, 3]) is None


def test_a_layer_whose_size_is_no_straight_line_in_the_input_still_serves_a_range(tmp_path, monkeypatch):
    """The serve step runs the model at input lengths and reads the layer's
    size off each recording, so a layer seeing L * L rows still gets real
    timings, and the kernel serves between them."""
    import textwrap
    from pathlib import Path
    from backcountry.harness.measure.session import Session
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    timed = []

    def grid(self, job, sizes):
        assert all(str(n) in job["recordings"] for n in sizes)
        timed.extend(sizes)
        return {n: {"win": True} for n in sizes}
    monkeypatch.setattr(JobRunner, "_serve_grid", grid)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {Path(__file__).parent / "fixtures" / "square_rows.py"}
        model_type: other
        workloads:
          - {{name: main, inputs: [{{shape: [L, 64], dtype: float32}}]}}
        primary: {{L: 12}}
        check_optimizations_for: {{L: [4, 24]}}
        baseline: plain
    """))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4,
                       session=Session(sleep=lambda s: None))
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(g for g in runner.build_regions() if tuple(g.ops) == ("mx.maximum",))
        runner._capture([region])
        kernel = elementwise("rmax_q1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
        assert runner._bind_and_promote(RegionRun(region=region), kernel, WIN)
        assert runner._serve_sizes()
        report = runner.report.serve["kernels"]["rmax_q1"]
        assert report["sizes"] == [[16, 576]]  # 4 * 4 to 24 * 24 rows
        assert set(timed) <= {n * n for n in range(4, 25)} and {16, 576} <= set(timed)
        assert report["L_recorded"][0] == 4 and report["L_recorded"][-1] == 24
    finally:
        runner.tracer.uninstall()


def test_the_final_check_times_fresh_sizes_and_reads_the_headline_off_them(served):
    """Over a range the final check runs the model at ten more sizes the
    search never ran at and times five of them; the headline is their
    geometric mean and the worst size, so a kernel tuned to the search's sizes
    cannot lift it."""
    _ship(served, elementwise("rmax_s7", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    served._final_check()
    headline = served.report.final["headline"]
    searched = set(served.report.constants["search_sizes"]["L"])
    fresh = [row["L"] for row in headline["curve"] if row["fresh"]]
    assert fresh and not set(fresh) & searched and all(4 <= n <= 96 for n in fresh)
    assert {row["L"] for row in headline["curve"] if not row["fresh"]} == searched
    assert [row["L"] for row in headline["curve"]] == sorted(row["L"] for row in headline["curve"])
    assert headline["geomean_fresh"] > 0 and headline["worst"]["L"] in fresh + sorted(searched)
    assert all(c["passed"] for c in served.report.final["checks"])
    checked = [c for c in served.report.final["checks"] if ":fresh" in str(c)]
    assert len(fresh) == 5 and len(checked) > len(fresh)  # all checked, five timed


def test_the_holdout_runs_a_kernel_bound_to_its_recorded_shapes_at_the_fresh_sizes(tmp_path, monkeypatch):
    """A region seen at several shapes pins its kernel to them (input_signatures);
    anywhere else the library ran. The holdout ran the kernel through that
    path, so at every fresh size it stepped aside, the static gate failed the
    whole check and the failure was ignored: Qwen3.8-27B's region 1 kernel
    passed a holdout that never ran it. Now the fresh shapes are added for
    the check (serving ignores the recorded list the same way), a failure
    anywhere fails the holdout, and a size the kernel's own rule declines is
    left to the library."""
    from dataclasses import replace
    from backcountry.harness.ladder.gates import LadderResult
    from backcountry.harness.measure.peaks import Peaks
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}")
    r.peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 1000.0}, launch_us=5.0)
    region = r.region("mx.maximum")
    r._capture([region])
    r._price_and_rank([region])
    searched = r.report.constants["search_sizes"]["L"]
    listed = [[[[n, 1024], "float32"]] for n in sorted(set(searched))]
    contract = replace(r._contract(region), input_signature=None, input_signatures=listed)
    monkeypatch.setattr(r, "_contract", lambda _region: contract)
    bound = lambda kernel: replace(kernel, input_signatures=listed)
    relu = "(in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f)"
    seen = "bool seen = " + " || ".join(f"in0_shape[0] == {n}" for n in searched) + ";\n"
    nominated = lambda: LadderResult("tentative_ship", None, {"target_workload": "main"}, 1.0, 2.0, 1.0, 0.0, [])

    wrong = r._holdout(region, bound(elementwise("rmax_fit_b", seen + f"out0[i] = seen ? {relu} : in0[i];")),
                       nominated(), "preserving")
    assert wrong is not None and wrong.status == "correctness_failed" and "drawn fresh" in wrong.reason, wrong

    result = nominated()  # right everywhere; small elementwise shapes may still lose to the library
    ok = r._holdout(region, bound(elementwise("rmax_ok_b", f"out0[i] = {relu};")), result, "preserving")
    assert ok is None or ok.status in ("holdout_slower", "holdout_unmeasured"), ok  # tiny shapes may not time
    fresh = [label for label in result.detail["sizes"] if label.endswith(":holdout")]
    assert fresh and all(result.detail["sizes"][label]["kernel_ms"] for label in fresh)  # it ran there

    declines = replace(bound(elementwise("rmax_decl_b", f"out0[i] = {relu};")), fallback_predicate=" and ".join(
        f"in0.shape[0] != {n}" for n in searched))
    assert r._holdout(region, declines, nominated(), "preserving") is None
    assert [row for row in r.log.rows() if row["kind"] == "holdout_declined"]

    broken = bound(elementwise("rmax_bad_b", "out0[i] = undefined_name;"))
    failed = r._holdout(region, broken, nominated(), "preserving")
    assert failed is not None and failed.status == "correctness_failed", failed
    r.tracer.uninstall()


def test_over_a_range_regions_are_priced_at_the_ends_and_the_primary(served):
    """Pricing at ten sizes cost 22 of a 0.6B model's 40-minute setup; a
    price ranks regions and picks the judge's target, so three sizes do. The
    kernel is still checked and timed at every size."""
    from backcountry.harness.measure.peaks import Peaks
    served.peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 1000.0}, launch_us=5.0)
    region = served.region("mx.maximum")
    served._capture([region])
    served._price_and_rank([region])
    sized = served._sized_workloads()
    assert len(sized) > 3
    assert set(region.p) == {sized[0][1], served.manifest.workloads[0].name, sized[-1][1]}
    sets = served._eval_sets(region)
    assert sets[0].t_library_ms is not None and len(sets) >= len(sized)
    assert {s.label for s in sets if s.t_library_ms is None} == {name for _n, name in sized} - set(region.p)


def test_a_kernel_fitted_to_the_search_sizes_fails_the_holdout_at_sizes_drawn_fresh(tmp_path):
    """The search runs at sizes fixed for the job, so a kernel can be right,
    or fast, at exactly those. Before the whole-model check the kernel is
    checked and clocked at sizes drawn fresh, recorded from real passes."""
    from backcountry.harness.ladder.gates import LadderResult
    from backcountry.harness.measure.peaks import Peaks
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}")
    r.peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 1000.0}, launch_us=5.0)
    region = r.region("mx.maximum")
    r._capture([region])
    r._price_and_rank([region])
    searched = r.report.constants["search_sizes"]["L"]
    seen = "bool seen = " + " || ".join(f"in0_shape[0] == {n}" for n in searched) + ";\n"
    relu = "(in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f)"
    nominated = lambda: LadderResult("tentative_ship", None, {"target_workload": "main"}, 1.0, 2.0, 1.0, 0.0, [])

    wrong = r._holdout(region, elementwise("rmax_fit", seen + f"out0[i] = seen ? {relu} : in0[i];"),
                       nominated(), "preserving")
    assert wrong.status == "correctness_failed" and "drawn fresh" in wrong.reason, wrong

    stall = ("float v = " + relu + ";\nif (!seen) { float w = in0[i]; for (uint j = 0; j < 500u; ++j) "
             "{ w = metal::sin(w) + 1.0f; } if (w > 1.0e30f) { v = w; } }\nout0[i] = v;")
    result = nominated()
    slow = r._holdout(region, elementwise("rmax_stall", seen + stall), result, "preserving")
    assert slow.status == "holdout_slower" and slow.outcome == "correct_slower", slow
    fresh = [label for label in result.detail["sizes"] if label.endswith(":holdout")]
    assert fresh and not {int(label.split("=")[1].split(":")[0]) for label in fresh} & set(searched)
    assert all(result.detail["sizes"][label]["loss"] for label in fresh)
    r.tracer.uninstall()


def test_a_kernel_pinned_to_recorded_shapes_still_runs_where_the_serve_step_checked_it():
    """A multi-shape region's kernel lists the shapes it was searched at;
    a recorded-shape wrapper (try_call) hands any other shape to the library,
    the serve step's size-generic wrapper (try_sized) runs it there, bounded
    only by the kernel's own rule."""
    from dataclasses import replace
    double = replace(elementwise("double_pinned", "out0[i] = in0[i] * 2.0f;"),
                     input_signatures=[[[[4, 8], "float32"]]])
    recorded, unrecorded = mx.ones((4, 8)), mx.ones((6, 8))
    assert kernels.try_call(double, [recorded]) is not None
    assert kernels.try_call(double, [unrecorded]) is None
    served = kernels.try_sized(double, [unrecorded])
    assert served is not None and mx.array_equal(served[0], unrecorded * 2).item()
    declined = replace(double, kernel_id="double_declines", fallback_predicate="in0.shape[0] > 5")
    assert kernels.try_sized(declined, [unrecorded]) is None


def test_over_a_range_the_job_start_clock_times_only_the_priced_sizes(tmp_path):
    """The step clock at all ten sizes was an hour of Qwen3.8-27B's setup, and
    only the priced sizes' clocks are read."""
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}")
    r.tracer.uninstall()
    r._clock_steps()
    assert set(r.step_ms) == r._priced_workloads() and len(r.step_ms) == 3
    assert len(r.manifest.workloads) > 3


def test_a_finish_request_ends_a_selection_wave_at_its_next_pacing_idle(tmp_path, monkeypatch):
    """On Qwen3.8-27B a finish asked for 2 minutes into a pricing wave waited
    45 for it; the wave's pacing idles now look for the request."""
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}")
    from backcountry.harness.measure.peaks import Peaks
    r.tracer.uninstall()
    r.peaks = Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 1000.0}, launch_us=5.0)
    region = r.region("mx.maximum")
    r.pending_regions = [region]
    r.report.coverage["selection"] = {"candidates": 1, "waves": []}
    priced = []

    def capture(regions):
        (r.work_dir / "finish-search.request").touch()  # the operator asks mid-wave
        r.session._idle(60.0)                            # the wave's next cooling pause

    monkeypatch.setattr(r, "_capture", capture)
    monkeypatch.setattr(r, "_price_and_rank", lambda regions: priced.append(regions) or regions)
    assert r._next_regions() == [] and not priced
    assert [row for row in r.log.rows() if row["kind"] == "selection_abandoned"]
    assert r.session.stop_when is None and not r.tracer.patcher.installed
    # the rest the stop cut short is still owed by the next GPU call (the final check)
    assert r.session._ready_at - r.session._now() > 55.0
    r.session._ready_at = 0.0
    assert r._next_regions() == []  # and no new wave starts while it stands


def test_a_finish_request_ends_re_pricing_after_a_ship(tmp_path, monkeypatch):
    """Re-pricing what is left after a region ships is pricing too: a finish
    asked for during it ends it at its next pacing idle, and the search stops."""
    from types import SimpleNamespace
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}")
    r.tracer.uninstall()
    region = r.region("mx.maximum")
    r._search_state = {"ranked": [], "shipped": [], "run": None, "accepted_before": 0}
    r.report.accepted.append({"fingerprint": region.fingerprint})
    monkeypatch.setattr(r, "_record_region_closed", lambda run: None)
    refreshed = []

    def refresh(ranked, shipped):
        (r.work_dir / "finish-search.request").touch()  # the operator asks mid-refresh
        r.session._idle(60.0)                            # its next cooling pause
        refreshed.append(ranked)

    monkeypatch.setattr(r, "_refresh_after_ship", refresh)
    assert r._close_region(SimpleNamespace(region=region, shipped=object())) is False and not refreshed
    assert [row for row in r.log.rows() if row["kind"] == "selection_abandoned"]
    assert r.session.stop_when is None
    r.session._ready_at = 0.0


def test_a_kernel_that_leaves_the_shortest_sizes_to_the_library_installs_and_runs_the_original_there(
        served, monkeypatch):
    """The size gate's stretch rule: a kernel may decline the ends of the
    range. Installing it must not expect it where it declines (counted as
    expected there, the install is refused), and there the model runs the
    original ops, so its outputs are the library's."""
    from dataclasses import replace
    import backcountry.harness.loop as loop
    kernel = replace(elementwise("rmax_end", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"),
                     fallback_predicate="in0.shape[0] < 8")
    region = served.region("mx.maximum")
    served._capture([region])
    with monkeypatch.context() as m:
        m.setattr(loop, "_declines", lambda trace, cut: False)
        assert not served._bind_and_promote(RegionRun(region=region), kernel, WIN)
    assert served._bind_and_promote(RegionRun(region=region), kernel, WIN)
    named = sorted(served.report.constants["search_sizes"]["L"])
    short = [w for w in served.manifest.workloads if any(w.name.endswith(f"@L={n}") for n in named if n < 8)]
    long = [w for w in served.manifest.workloads if any(w.name.endswith(f"@L={n}") for n in named if n >= 8)]
    assert short and long
    for w in short + long[:1]:
        got, want = served.model(*served.tensors[w.name]), served.baseline_model(*served.tensors[w.name])
        assert mx.array_equal(got, want).item()
    # the serve step's clock skips the sizes it declines (timing them crashed the job)
    monkeypatch.setattr(JobRunner, "_serve_grid", REAL_SERVE_GRID)
    assert served._serve_sizes()
    grid = next(iter(served.report.serve["kernels"].values()))["grid"]
    assert all(row == {"declined": True, "win": False, "loss": False} for n, row in grid.items() if n < 8)
    assert any("library_ms" in row for n, row in grid.items() if n >= 8)


def test_the_serve_walk_stops_once_the_layer_size_passes_the_stretch(tmp_path):
    """A layer whose size stops growing (a prompt read in fixed chunks) is not
    run at every longer length: those passes only repeat checked sizes."""
    from types import SimpleNamespace
    walked = []

    def recordings(region, fingerprint, member, template, values):
        walked.extend(values)
        found = {}
        for v in values:
            path = tmp_path / f"r{v}"
            path.write_bytes(b"x" * 10)
            found[v] = (min(v, 10), str(path))  # the layer sees at most 10 rows
        return found
    runner = SimpleNamespace(SERVE_BATCH=4, SERVE_RECORD_BYTES=1 << 20, _serve_recordings=recordings,
                             _cut_source=JobRunner._cut_source,
                             _serve_correct=lambda job, sizes, errors=None: (set(sizes), {}))
    correct, wrong, _cut = JobRunner._serve_every_size(runner, {"recordings": {}}, None, "f", None, None,
                                                       list(range(1, 11)), [1, 100], None, {})
    assert correct == set(range(1, 11)) and not wrong
    assert max(walked) < 20  # not every length up to 100


def test_over_a_range_no_region_opens_in_a_scope_that_cannot_follow_the_size(tmp_path):
    """A region whose scope runs other operations at another recorded size is
    set aside before any attempt: its kernel could run only at the recorded
    sizes, and its replay would take every kernel inside it along (gpt-oss-20b
    spent 25 attempts on such a region, and its kernel left the expert kernels
    serving 13 prompt lengths of 2,018). Regions in a scope that follows the
    size stay; without a range nothing changes."""
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}",
                fixture="size_branch.py")
    try:
        assert r.region("mx.maximum").rejected is None
        stranded = {tuple(s["ops"]): s["reason"] for s in r.report.stranded}
        why = next(reason for ops, reason in stranded.items() if "mx.exp" in ops)
        assert "cannot follow the size" in why and "branchy" in why, stranded
    finally:
        r.tracer.uninstall()
    (tmp_path / "plain").mkdir()
    plain = _runner(tmp_path / "plain", "[64, 1024]", fixture="size_branch.py")
    try:
        assert not any("follow the size" in s["reason"] for s in plain.report.stranded)
        assert any("mx.exp" in reg.ops for reg in plain.regions)
    finally:
        plain.tracer.uninstall()


def test_a_scope_that_cannot_be_replayed_at_another_size_is_a_reason_not_a_crash(monkeypatch):
    # Kokoro (mlx-audio): at another size a convolution's ops interleave with
    # others; reading that scope raised NotReplayable through region screening
    # and check went down with a traceback
    from types import SimpleNamespace
    from backcountry.harness import serve
    from backcountry.harness.bind import certify
    from backcountry.harness.bind.emit import NotReplayable

    def interleaved(*_args):
        raise NotReplayable("scope '?ConvWeighted@0' ops are not contiguous in the record")

    monkeypatch.setattr(certify, "scope_call_by_address", lambda trace, address: SimpleNamespace(address=address))
    monkeypatch.setattr(serve, "_corresponding", interleaved)
    reason = serve.follows_size(None, SimpleNamespace(address="conv"), [object()])
    assert reason == "at another size, scope '?ConvWeighted@0' ops are not contiguous in the record"
