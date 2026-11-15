"""The measurement kit. Timing tests use real GPU work and modest sample
counts; assertions are on paired quantities so machine load cannot flip them.
"""

import mlx.core as mx
import pytest

from backcountry.harness.measure.clocks import compare, step_clock
from backcountry.harness.measure.controls import aa_null, synthetic_workload
from backcountry.harness.measure.peaks import measure_bandwidth, measure_flops, measure_launch_us
from backcountry.harness.measure.session import (WARM_CAP, WARM_PATIENCE, Session,
                                       time_once)


@pytest.fixture(scope="module")
def session():
    return Session()


def require_quiet_machine():
    """Some assertions here are about chip physics and the session floor; a
    machine busy with other work cannot measure either. The
    paired design survives load, but a 3% planted signal does not clear a
    contention-inflated floor, and a starved bandwidth run is not a peak.
    Thermal throttle is the same problem at zero load, so both gates apply."""
    from tests.conftest import require_healthy_gpu, require_quiet_load

    require_quiet_load()
    require_healthy_gpu()


def quiet_session():
    """No real idling, for tests that check mechanics rather than numbers."""
    return Session(sleep=lambda s: None)


def test_time_once_scales_with_work():
    # depth ratios only hold while per-launch overhead is small next to the
    # work; on a throttled chip the overhead dominates and 8x depth reads 2.8x
    require_quiet_machine()
    small = synthetic_workload(depth=2)
    big = synthetic_workload(depth=16)
    s = quiet_session()
    s.warm_until_stable(small)
    s.warm_until_stable(big)
    t_small = min(time_once(small) for _ in range(3))
    t_big = min(time_once(big) for _ in range(3))
    assert t_big > 3 * t_small


def test_session_settle_pays_debt():
    slept = []
    s = Session(sleep=slept.append, duty_idle_factor=3.0)
    fn = synthetic_workload(depth=2)
    t = s.timed(fn)
    s.settle()
    assert len(slept) == 1
    assert slept[0] == pytest.approx(3.0 * t, rel=1e-6)
    s.settle()
    assert len(slept) == 1  # debt already paid


def test_fresh_chunk_paces_and_ramps_until_flat():
    slept = []
    s = Session(sleep=slept.append, max_chunk_work_s=0.0)
    fn = synthetic_workload(depth=2)
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return fn()

    s.timed(counted)
    s.fresh_chunk(counted)
    assert len(slept) == 1  # debt past a zero threshold pays immediately
    # the ramp is self-sizing: at least patience samples, never past the cap
    assert 1 + WARM_PATIENCE <= calls["n"] <= 1 + WARM_CAP


def test_fresh_chunk_noop_below_threshold():
    slept = []
    s = Session(sleep=slept.append, max_chunk_work_s=1e9)
    fn = synthetic_workload(depth=2)
    s.timed(fn)
    s.fresh_chunk(fn)
    assert slept == []


def test_warm_until_stable_converges(session):
    times = session.warm_until_stable(synthetic_workload(depth=4))
    assert WARM_PATIENCE <= len(times) <= WARM_CAP
    # it stops because the reading stopped falling, so the tail is not a new low
    assert min(times[-WARM_PATIENCE:]) >= min(times) or len(times) == WARM_CAP


def test_step_clock_reports_median(session):
    clock = step_clock(session, synthetic_workload(depth=4), reps=5)
    assert clock.median_ms > 0
    assert len(clock.samples_ms) == 5
    assert min(clock.samples_ms) <= clock.median_ms <= max(clock.samples_ms)


def test_aa_null_shows_no_significant_win(session):
    """Law 12: the A/A control must not ship. Same fn both arms."""
    result = aa_null(session, pairs=12)
    assert not result.wins_by(margin_ms=0.0)
    assert not result.loses_by(margin_ms=0.0)


def test_injected_slowdown_is_bit_identical():
    base = synthetic_workload(depth=8, seed=3)
    injected = synthetic_workload(depth=8, injected_links=2, seed=3)
    a, b = base(), injected()
    mx.eval(a, b)
    assert mx.array_equal(a, b).item()


def test_injected_slowdown_is_detected(session):
    """A known ~3% planted slowdown is caught by the paired
    comparison while outputs stay bit-identical."""
    require_quiet_machine()
    base = synthetic_workload(depth=33, seed=5)
    injected = synthetic_workload(depth=33, injected_links=1, seed=5)
    result = compare(session, base, injected, pairs=16)
    assert result.loses_by(margin_ms=0.0), (
        f"planted 3% slowdown not detected: delta={result.median_delta_ms:.4f}ms "
        f"sigma={result.sigma_ms:.4f}ms base={result.median_baseline_ms:.2f}ms "
        f"ratio={result.median_ratio:.5f} blocks={result.n}; "
        f"baseline_ms={[round(t, 4) for t in result.baseline_ms]}; "
        f"candidate_ms={[round(t, 4) for t in result.candidate_ms]}"
    )
    slow_pct = -result.median_delta_ms / result.median_baseline_ms * 100
    assert 0.5 < slow_pct < 8.0, f"implausible slowdown size {slow_pct:.2f}%"


def test_compare_rejects_odd_pairs(session):
    with pytest.raises(ValueError):
        compare(session, lambda: mx.zeros(1), lambda: mx.zeros(1), pairs=3)


def test_peaks_are_physically_sane(session):
    require_quiet_machine()
    bw = measure_bandwidth(session, samples=3)
    assert 30 < bw < 1000, f"bandwidth {bw:.0f} GB/s out of any plausible M-series range"
    fp32 = measure_flops(session, mx.float32, samples=3)
    fp16 = measure_flops(session, mx.float16, samples=3)
    assert 500 < fp32 < 50_000, f"fp32 {fp32:.0f} GFLOPs implausible"
    assert fp16 > 0.5 * fp32, "fp16 should not be far below fp32 on Apple GPUs"
    launch = measure_launch_us(session, samples=3)
    assert 0.1 < launch < 100, f"launch cost {launch:.2f}us implausible"


def test_implausible_peaks_are_named():
    """The degraded-machine guard (seen live: a GPU at a tenth of its known
    speed within one boot session)."""
    from backcountry.harness.measure.peaks import Peaks, implausible

    healthy = Peaks(bandwidth_gbps=92.8, flops_gflops={"float32": 2800.0}, launch_us=6.0)
    assert implausible(healthy) is None
    slow_bw = Peaks(bandwidth_gbps=9.4, flops_gflops={"float32": 2800.0})
    assert "bandwidth" in implausible(slow_bw)
    slow_fp = Peaks(bandwidth_gbps=92.8, flops_gflops={"float32": 180.0})
    assert "fp32" in implausible(slow_fp)
    no_fp32 = Peaks(bandwidth_gbps=92.8, flops_gflops={})
    assert implausible(no_fp32) is None


def test_ramp_keeps_going_while_the_reading_still_falls():
    """The old rule stopped once two readings agreed, which a ramping GPU does
    while both are still far above steady. Feed a ramp that plateaus twice:
    the ramp must see through the first plateau and stop on the real floor."""
    from backcountry.harness.measure.session import _until_flat

    # the shape tools/ramp_after_idle.py measured after a 0.75s idle: a long
    # plateau, then a step down, then the floor
    curve = [21.0, 19.1, 18.8, 18.8, 18.3, 9.4, 8.1, 8.2, 8.8, 8.1, 8.8] + [7.8] * 30  # ms
    readings = iter(t / 1e3 for t in curve)
    seen = []

    def fake_timed(_fn):
        t = next(readings)
        seen.append(round(t * 1e3, 1))
        return t

    _until_flat(fake_timed, None, rtol=0.01, abs_s=0.0, cap=60, patience=WARM_PATIENCE)
    assert seen[-1] == 7.8, "stopped on the 18.8 plateau instead of the real floor"
    assert 9.4 in seen, "never saw past the first plateau"


def test_best_of_peaks_keeps_the_higher_reading():
    from backcountry.harness.measure.peaks import Peaks, best_of

    a = Peaks(90.0, {"float32": 2000.0, "float16": 3000.0}, 7.0)
    b = Peaks(95.0, {"float32": 1800.0}, 6.0)
    m = best_of(a, b)
    assert (m.bandwidth_gbps, m.launch_us) == (95.0, 6.0)
    assert m.flops_gflops == {"float32": 2000.0, "float16": 3000.0}


def test_gpu_utilization_parses_what_macos_reports():
    from backcountry.harness.measure.peaks import gpu_utilization

    text = '"PerformanceStatistics" = {"Tiler Utilization %"=98,"Device Utilization %"=37,"x"=1}'
    assert gpu_utilization(text) == 37.0
    assert gpu_utilization("nothing about the GPU here") is None


def test_gpu_core_count_parses_what_macos_reports():
    from backcountry.harness.measure.peaks import gpu_core_count

    text = 'some other line\n      "gpu-core-count" = 10\n      "IOClass" = "AGXAcceleratorG16X"\n'
    assert gpu_core_count(text) == 10
    assert gpu_core_count("nothing here") is None


def test_gpu_facts_follow_the_chip_generation():
    from backcountry.harness.measure.peaks import gpu_facts

    m2, m4, m5 = gpu_facts("Apple M2 Max"), gpu_facts("Apple M4"), gpu_facts("Apple M5 Pro")
    assert m4["threadgroup_memory_bytes"] == 32768 and m4["simd_width"] == 32
    assert "208 KB" in m2["on_chip_memory"] and "dynamically" in m4["on_chip_memory"]
    assert "no dedicated" in m4["matrix_unit"] and "neural accelerator" in m5["matrix_unit"]


def test_chained_loop_computes_the_same_values_and_prices_its_link():
    """The chain adds a zero from the last pass to the next pass's linked
    input, so every pass computes exactly what the unchained loop computes,
    the link rides on the smallest non-weight float input, and the link
    loop alone hands every input back unchanged."""
    from backcountry.harness.measure.clocks import chained_loop, link_input, link_loop, timing_sets
    from backcountry.harness.measure.session import time_once

    w = mx.random.normal((256, 64), key=mx.random.key(1))
    x = mx.random.normal((4, 64), key=mx.random.key(2))
    mx.eval(w, x)
    sets = timing_sets([{7: w, 3: x}])
    assert len(sets) > 1 and all(set(s) == {3, 7} for s in sets)
    assert link_input(sets[0], weight_ids={7}) == 3
    assert link_input({7: w, 3: x}) == 3  # the smallest float input even with no weight ids
    assert link_input({1: mx.zeros((4,), dtype=mx.int32)}) is None

    passes = lambda b: [b[3] @ b[7].T]
    chained = chained_loop(passes, sets, 12, 3)()
    plain = [passes(sets[i % len(sets)]) for i in range(12)]
    mx.eval(chained, plain)
    assert all(mx.array_equal(c[0], p[0]).item() for c, p in zip(chained, plain))
    linked = link_loop(sets, 12, 3)()
    mx.eval(linked)
    assert all(mx.array_equal(l[0], sets[i % len(sets)][3]).item() for i, l in enumerate(linked))


def test_chained_launches_do_not_overlap():
    """Protects: the region clock's chain law. Metal runs independent launches
    side by side, so a one-threadgroup kernel timed in an unchained loop reads
    several times faster than it runs in a model, where each layer waits for
    the last (seen live on the 2026-09-02 Qwen run: 0.03 ms unchained, 0.33
    chained, 0.36 in the model)."""
    from tests.conftest import require_healthy_gpu, require_quiet_load
    from backcountry.harness.measure.clocks import chained_loop
    from backcountry.harness.measure.session import time_once

    require_healthy_gpu()
    require_quiet_load()
    kernel = mx.fast.metal_kernel(
        name="pin_one_threadgroup_matvec", input_names=["w", "x"], output_names=["out"],
        source="""
            uint r = thread_position_in_grid.x;
            float acc = 0.0f;
            for (uint c = 0; c < 2048u; ++c) { acc += (float)w[r * 2048u + c] * (float)x[c]; }
            out[r] = acc;
        """)
    ws = [mx.random.normal((128, 2048), key=mx.random.key(i)).astype(mx.float16) for i in range(8)]
    x = mx.random.normal((2048,), key=mx.random.key(99)).astype(mx.float16)
    mx.eval(ws, x)
    sets = [{0: w, 1: x} for w in ws]
    one = lambda b: kernel(inputs=[b[0], b[1]], grid=(128, 1, 1), threadgroup=(128, 1, 1),
                           output_shapes=[(128,)], output_dtypes=[mx.float32])
    n = 40
    unchained = lambda: [one(sets[i % len(sets)]) for i in range(n)]
    chained = chained_loop(one, sets, n, 1)
    for _ in range(3):
        time_once(unchained)
        time_once(chained)
    t_un = min(time_once(unchained) for _ in range(5))
    t_ch = min(time_once(chained) for _ in range(5))
    assert t_ch > 2.0 * t_un, (t_un, t_ch)


# -- the floor probe ------------------------------------------------------------

def test_stream_probe_runs_on_odd_shapes_and_small_inputs():
    """The probe must build and run for whatever a boundary holds: odd byte
    counts, integers, bools, empty arrays, and inputs so small that
    mx.fast.metal_kernel binds them in constant memory (under 8 elements; the
    8-element input pins that boundary, since it is read through the vector
    cast that only device memory allows)."""
    from backcountry.harness.measure.probe import MLX_CONSTANT_BELOW, stream_probe

    ins = [((7, 3), "float16"), ((1,), "int32"), ((5,), "bool"), ((2, 3, 4), "bfloat16"),
           ((0,), "float32"), ((MLX_CONSTANT_BELOW,), "float32")]
    outs = [((3, 5), "float32"), ((1, 1, 1), "bfloat16"), ((9,), "uint8")]
    arrays = [mx.zeros(s, dtype=getattr(mx, "bool_" if d == "bool" else d)) for s, d in ins]
    res = stream_probe(ins, outs)(arrays)
    mx.eval(res)
    assert [tuple(r.shape) for r in res] == [s for s, _ in outs]
    assert [str(r.dtype).removeprefix("mlx.core.") for r in res] == [d for _, d in outs]


def test_stream_probe_reads_only_what_the_region_reads():
    """An expert matmul handed 128 experts reads the 8 its index picks; a
    probe streaming all 128 put its floor 16x above the library, so Gemma 4's
    two expert regions ranked at zero headroom and never opened."""
    from backcountry.harness.measure.probe import probe_source

    experts = ((128, 32, 8), "uint32")  # 131072 bytes, 8192 packed_uint4
    assert "8192u" in probe_source([experts], [16])
    eight = probe_source([experts], [16], [131072 * 8 // 128])
    assert "512u" in eight and "8192u" not in eight


def test_mlx_matvec_sits_near_the_stream_floor():
    """A floor is only a floor if the library cannot beat it. Paired in one
    window, chained and cache-cold, MLX's bf16 matvec over a 4 MB matrix reads
    between 0.9x and 1.5x the probe on this chip (spike 12: 1.0x to 1.2x)."""
    from tests.conftest import require_healthy_gpu, require_quiet_load
    from backcountry.harness.measure.clocks import (CLOCK_TARGET_MS, chained_loop, compare, link_input,
                                          link_loop, loop_iterations, timing_sets)
    from backcountry.harness.measure.probe import floor_from, stream_probe
    from backcountry.harness.measure.session import Session, time_once

    require_healthy_gpu()
    require_quiet_load()
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    w = mx.random.normal((2048, 1024)).astype(mx.bfloat16)
    mx.eval(x, w)
    sets = timing_sets([{0: x, 1: w}])
    link_id = link_input(sets[0], {1})
    lib = lambda b: [b[0] @ b[1].T]
    iters = loop_iterations(time_once, lambda n: chained_loop(lib, sets, n, link_id), CLOCK_TARGET_MS)
    lib_loop = chained_loop(lib, sets, iters, link_id)
    probe = stream_probe([((1, 1, 1024), "bfloat16"), ((2048, 1024), "bfloat16")], [((1, 1, 2048), "bfloat16")])
    probe_loop = chained_loop(lambda b: probe([b[0], b[1]]), sets, iters, link_id)
    session = Session()
    net = compare(session, link_loop(sets, iters, link_id), lib_loop, pairs=8)
    library_ms = -net.median_delta_ms / iters
    floor = compare(session, probe_loop, lib_loop, pairs=8)
    floor_ms = floor_from(floor, net.median_baseline_ms, iters, library_ms)
    assert 0.9 <= library_ms / floor_ms <= 1.5



def test_a_chained_closure_reads_per_step_and_owes_the_whole_sample():
    from backcountry.harness.measure.clocks import chained_steps
    from backcountry.harness.measure.session import Session, time_once
    x = mx.random.normal((256, 256))
    w = mx.random.normal((256, 256))
    mx.eval(x, w)
    step = lambda a, b: a @ b
    one = chained_steps(step, [x, w], 1)
    eight = chained_steps(step, [x, w], 8)
    assert eight.steps == 8
    outs = eight()
    mx.eval(outs)
    assert len(outs) == 8 and all(mx.array_equal(o, x @ w).item() for o in outs)
    for fn in (one, eight):
        for _ in range(3):
            time_once(fn)
    per_step = min(time_once(eight) for _ in range(5))
    single = min(time_once(one) for _ in range(5))
    assert per_step < single  # the per-step reading, with the sample's fixed cost spread over 8
    session = Session(sleep=lambda s: None)
    t = session.timed(eight)
    assert session.work_s == pytest.approx(t * 8)


def test_a_pass_that_fills_a_sample_is_sized_without_a_loop():
    from backcountry.harness.measure.clocks import loop_iterations
    timed = []
    def loop_for(n):
        def run():
            timed.append(n)
            return 0.025 * n
        return run
    assert loop_iterations(lambda fn: fn(), loop_for) == 1
    assert timed == [1, 1]  # the throwaway and one reading


@pytest.mark.parametrize("ramped", [True, False])
def test_step_clock_reuses_ramp_and_reports_before_cooling(ramped):
    events = []

    class FakeSession:
        def fresh_chunk(self, fn):
            events.append("ramp")
            return ramped

        def warm_until_stable(self, fn):
            events.append("warm")
            return [0.01]

        def timed(self, fn):
            events.append("sample")
            return 0.01

        def log(self, kind, **row):
            events.append(kind)

        def settle(self):
            events.append("cool")

    clock = step_clock(FakeSession(), lambda: None, reps=9)
    assert events == ["ramp"] + ([] if ramped else ["warm"]) + ["sample"] * 9 + ["step_clock", "cool"]
    assert clock.median_ms == 10


def test_unpaced_work_owes_no_cooling_and_rest_cools_once():
    """The serve step's thousands of untimed recordings owed 3x their work (then)
    as idle (30 of its 71 minutes on Qwen3-30B-A3B); unpaced, they owe
    nothing, and the timed grid after them rests once."""
    slept = []
    session = Session(sleep=slept.append)
    assert session.off_clock(lambda: mx.eval(mx.ones((256, 256)) @ mx.ones((256, 256))) or 7, paced=False) == 7
    assert slept == [] and session._debt_s == 0.0 and session.work_s > 0
    session.rest(60.0)
    assert slept == [60.0]
    session.off_clock(lambda: mx.eval(mx.ones((256, 256)) @ mx.ones((256, 256))))
    assert len(slept) == 1  # paced work schedules its rest instead of taking it now
    session.wait_ready()
    assert len(slept) == 2  # and the next GPU call takes it


def test_a_long_call_warms_with_two_flat_readings_a_short_one_with_ten():
    """A call of a second or more outlasts the GPU's clock ramp, and more warm
    calls only heat the chip (Qwen3-VL-8B, 11 s calls rose 12%); a short call
    keeps the full patience, since the ramp spans several of them."""
    from backcountry.harness.measure.session import _until_flat, WARM_PATIENCE
    flat = lambda seconds: _until_flat(lambda fn: seconds, None, 0.01, 1e-4, 60, WARM_PATIENCE)
    assert len(flat(11.0)) == 3        # the first reading, then two that do not improve
    assert len(flat(0.38)) == WARM_PATIENCE + 1


def test_one_spiked_pair_does_not_hide_a_win():
    """A 2% win over ten pairs, one of which hit a paging spike: the spiked
    difference counts only as far out as an ordinary large one, so the win
    still resolves. Four pairs say too little for that, and keep the plain
    standard deviation."""
    from backcountry_runtime.stats import comparison_from_samples
    base = [254.0 + 0.5 * (i % 3) for i in range(10)]
    cand = [b - 5.0 + 0.4 * ((i * 7) % 5 - 2) for i, b in enumerate(base)]
    cand[3] += 300.0  # the spike
    comp = comparison_from_samples(base, cand)
    assert comp.spread_ms > 90 and comp.sigma_ms < 1.5 and comp.wins_by(0.0)
    few = comparison_from_samples(base[:4], cand[:4])
    assert few.sigma_ms > 50 and not few.wins_by(0.0)
