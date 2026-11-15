"""A kernel may leave the ends of a range to the library: its own code runs one
unbroken stretch of at least 7 in 10 of the sizes, settled from shapes before
anything runs."""
from dataclasses import asdict, replace

import mlx.core as mx
import pytest

from backcountry.harness.ladder import child
from backcountry.harness.ladder.child import stretch_problem
from backcountry.harness.measure.session import Session
from backcountry.harness.sandbox.protocol import EvalSetSpec, LadderSpec
from backcountry_runtime.kernels import KernelSpec

# Qwen3-VL-8B's gate/up region: its ten prompt lengths, the chunks of the long
# ones, and the 1-token call every prompt ends with
ROWS = {"prompt@T=85": 84, "prompt": 1024, "prompt@copy:1126": 1, "prompt@T=128": 127, "prompt@T=205": 204,
        "prompt@T=434": 433, "prompt@T=497": 496, "prompt@T=1325": 1324, "prompt@T=2816": 2048,
        "prompt@T=2816@copy:1126": 767, "prompt@T=4066@copy:1126": 2017, "prompt@T=7749@copy:3450": 1604}
LENGTHS = ["prompt", "prompt@T=85", "prompt@T=128", "prompt@T=205", "prompt@T=434", "prompt@T=497",
           "prompt@T=1325", "prompt@T=2816", "prompt@T=4066", "prompt@T=7749"]


def _table(declined):
    table = {label: {"shape": [1, rows, 4096], "declined": label in declined} for label, rows in ROWS.items()}
    uses = {label: {child._workload(label)} for label in ROWS}
    uses["prompt@copy:1126"] = set(LENGTHS)  # the 1-token call: one shape at every length
    uses["prompt@T=2816"] = {"prompt@T=2816", "prompt@T=4066", "prompt@T=7749"}  # their 2,048-row chunks
    return table, uses


def test_a_kernel_may_leave_the_shortest_sizes_and_the_one_token_call_to_the_library():
    """The region's best kernel so far won from 433 rows up: 7 of the 10 lengths."""
    assert stretch_problem(*_table({"prompt@copy:1126", "prompt@T=85", "prompt@T=128", "prompt@T=205"})) is None


def test_its_own_stretch_must_cover_seven_in_ten():
    problem = stretch_problem(*_table({"prompt@copy:1126", "prompt@T=85", "prompt@T=128", "prompt@T=205",
                                       "prompt@T=434"}))
    assert "6 of 10" in problem and "at least 7" in problem


def test_declined_sizes_sit_at_the_ends_never_inside_the_stretch():
    assert "at the ends" in stretch_problem(*_table({"prompt@T=1325"}))


def test_the_longest_end_may_be_left_too():
    """The 2,048-row chunks are the most rows: leaving them drops the three
    longest prompts, which still keeps 7."""
    assert stretch_problem(*_table({"prompt@T=2816"})) is None


def _kernel(predicate):
    return KernelSpec(kernel_id="stretch", name="at_stretch", input_names=("in0",), output_names=("out0",),
                      source="out0[0] = in0[0];", grid=("in0.shape[0]", "1", "1"),
                      output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",),
                      fallback_predicate=predicate)


@pytest.mark.parametrize("predicate,passes,why", [
    ("in0.shape[0] < 12", True, None),                  # leaves the 8-row size: 4 of 5 run
    ("in0.shape[0] == 32", False, "at the ends"),        # a hole inside the stretch
    ("in0.shape[0] < 40", False, "at least 4"),          # leaves 3 of 5
])
def test_the_worker_settles_the_stretch_before_it_runs_anything(monkeypatch, predicate, passes, why):
    """The real worker path, on the CPU: a stretch that holds gets past the
    static stage to its first launch; one that breaks the rule is refused
    there, before any kernel call."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rows = {"prompt": 64, "prompt@T=9": 8, "prompt@T=17": 16, "prompt@T=33": 32, "prompt@T=129": 128}
        tensors = {}
        for label, n in rows.items():
            tensors[f"in:{label}"] = {0: mx.zeros((n, 4))}
            tensors[f"out:{label}"] = {1: mx.zeros((n, 4))}
        monkeypatch.setattr(child, "load_set", lambda path: tensors[path])
        monkeypatch.setattr(child, "saturate_pool", lambda *_: None)
        monkeypatch.setattr(child, "Session", lambda **kw: Session(sleep=lambda _: None))
        attempted = []

        class Launched(Exception):
            pass

        def call(*args, **kwargs):
            attempted.append(True)
            raise Launched()  # past the static stage: nothing more to see here

        monkeypatch.setattr(child, "call", call)
        sets = tuple(EvalSetSpec(label, (f"in:{label}",), (f"out:{label}",), 1.0, False, None) for label in rows)
        job = LadderSpec(kernel=asdict(_kernel(predicate)), assoc_tag="changing", nodes_json="[]",
                         input_ids=(0,), output_ids=(1,), eval_sets=sets, tolerances={}, kappa=1.25,
                         changing_floor=None, min_win_ms=0.0, phase="validate")
        if passes:
            with pytest.raises(Launched):
                child.evaluate_ladder(job)
            assert attempted
        else:
            verdict = child.evaluate_ladder(job)
            assert not verdict.passed and verdict.failed_gate == "static"
            assert "fallback_on_workload" in verdict.detail["reason"] and why in verdict.detail["reason"]
            assert attempted == []
    finally:
        mx.set_default_device(previous)


def test_the_judge_sees_a_declined_size_as_declined():
    from types import SimpleNamespace
    from backcountry.harness.loop import JobRunner
    runner = SimpleNamespace(report=SimpleNamespace(constants={"search_sizes": {"T": [9, 64]}}),
                             manifest=SimpleNamespace(workloads=[], primary={"T": 64}))
    rows = {"prompt@T=9": {"library_ms": 1.0, "kernel_ms": 1.0, "win": False, "loss": False, "declined": True},
            "prompt@T=64": {"library_ms": 2.0, "kernel_ms": 1.5, "win": True, "loss": False}}
    table = JobRunner._size_table(runner, {"sizes": rows})
    assert [row["result"] for row in table] == ["declined", "faster"]
    assert "left to MLX" in table[0]["size"]
