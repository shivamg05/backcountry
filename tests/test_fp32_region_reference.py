"""A changing kernel is graded against its region recomputed in fp32.

Qwen3.8-27B's fastest region-1 kernel was 1.04-1.36x the library at every
prompt length and was rolled back: the gated-delta state is fp32, held to an
fp32 tolerance (1e-5), and any kernel that is not bitwise the library moves it
about 1% through the bf16 math that feeds it. The region recomputed in fp32
moves it as far, so the reference is the model with the changed regions
recomputed so, and a kernel may be no further from it than 1.25x the library.
"""
from dataclasses import replace

import mlx.core as mx
import pytest

from backcountry.harness.e2e import preserving_check
from backcountry.harness.loop import JobRunner, RegionRun
from backcountry_runtime.kernels import KernelSpec
from backcountry_runtime.state import correctness_call
from tests.test_install import WIN, _runner

REGION = ("array.astype", "array.__mul__", "array.__add__")


def mix_kernel(kernel_id: str, value: str) -> KernelSpec:
    """out = value over (h, a, b) for h = bf16(x): the fixture's multiply-add."""
    return KernelSpec(
        kernel_id=kernel_id, name=f"at_{kernel_id}", input_names=("in0", "in1", "in2"),
        output_names=("out0",),
        source=("uint i = thread_position_in_grid.x;\nuint c = i % in0_shape[1];\n"
                "float h = float(bfloat(in0[i]));\nfloat a = float(in1[c]);\nfloat b = float(in2[c]);\n"
                f"out0[i] = bfloat({value});"),
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("256", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("bfloat16",),
    )


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    # correctness is what this tests; a lone elementwise kernel is no faster
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: all(c.passed for c in e2e.checks))
    monkeypatch.setattr(JobRunner, "_holdout", lambda self, *a: None)
    r = _runner(tmp_path, "[8, 1024]", "tolerances: default", fixture="bf16_state.py")  # it grades changing kernels
    region = r.region(*REGION)
    r._capture([region])
    yield r, region
    r.tracer.uninstall()


def test_a_kernel_rounding_once_in_fp32_ships_and_a_sloppy_one_does_not(mixed, tmp_path):
    r, region = mixed
    t = r.tensors["main"]

    sloppy = r._bind_and_promote(RegionRun(region=region), mix_kernel("mix_sloppy", "(h * a + b) * 0.98f"),
                                 WIN, "changing")
    assert sloppy.status == "correctness_failed", sloppy

    once = r._bind_and_promote(RegionRun(region=region), mix_kernel("mix_once", "h * a + b"), WIN, "changing")
    assert once, once
    # under the fixed tolerances it was held to before, the fp32 state fails
    old = preserving_check(lambda: correctness_call(r.baseline_model, t), lambda: correctness_call(r.model, t),
                           "main", exact=False, tolerances=r.manifest.tolerances)
    assert not old.passed and old.failure["output"] == 1, old
    r._check_unchanged("after the ship")  # the resume check grades it the same way

    # the bundle carries the reference's outputs and validates against them
    import json
    r.final_ok = True
    out = r._emit_artifact(tmp_path / "artifact")
    metadata = json.loads((out / "bundle.json").read_text())
    assert metadata["correctness_rule"] == "fp32_relative" and set(metadata["goldens"]) == {"main"}


def test_the_reference_with_promotion_off_is_the_library_bitwise(mixed, monkeypatch):
    """The stand-in plumbing alone changes nothing: replaying the region at
    its own precision through the reference install reproduces the library."""
    import backcountry.harness.bind.fp32_region as fp32_region
    from backcountry.harness.trace.replay import replay
    r, region = mixed
    assert r._bind_and_promote(RegionRun(region=region), mix_kernel("mix_once_b", "h * a + b"), WIN, "changing")
    monkeypatch.setattr(fp32_region, "golden_outputs", lambda nodes, binds, outs: replay(nodes, binds, outs))
    reference = r._reference_model()
    t = r.tensors["main"]
    want, got = correctness_call(r.baseline_model, t), correctness_call(reference, t)
    assert all(mx.array_equal(a, b).item() for a, b in zip(want, got))
    monkeypatch.undo()
    promoted = correctness_call(r._reference_model(), t)
    assert not mx.array_equal(promoted[0], want[0]).item()  # with promotion it is a different reference
