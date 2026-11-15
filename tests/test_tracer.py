"""The tracer against the fixture zoo. Each fixture exists to exercise one
mechanism.

The patch surface is process-global state, so one module-scoped Tracer serves
all tests here, and the final test uninstalls and checks exact restoration.
"""

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from backcountry.harness.trace import Tracer
from tests.conftest import current_tracer, tracer_for_module
from backcountry.harness.trace.recorder import KERNEL_PREFIX, OPAQUE_OP, UNNAMED_KERNEL_OP, ArrayRef
from backcountry.harness.trace.replay import replay
from backcountry.harness.trace.types import Retention, TraceIncomplete

FIXTURES = Path(__file__).parent / "fixtures"

_module_tracer = tracer_for_module()


def tracer() -> Tracer:
    return current_tracer()


def load_fixture(name: str):
    tracer()  # patches must be up before the model file imports
    spec = importlib.util.spec_from_file_location(f"fixture_{name}", FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # a compiled function in it is named by this import path
    spec.loader.exec_module(mod)
    return mod.build()


def trace_of(name: str, shape, dtype=mx.float32):
    model = load_fixture(name)
    x = mx.random.normal(shape, key=mx.random.key(0)).astype(dtype)
    return tracer().trace(model, [x])


def test_operator_soup_records_completely():
    trace, outs = trace_of("operator_soup", (4, 8))
    ops = [n.op for n in trace.nodes]
    # dunders, reflected, in-place, matmul, slicing, slice write, comparison all present
    assert "array.__add__" in ops
    assert "array.__rmul__" in ops
    assert "array.__rsub__" in ops
    assert "array.__iadd__" in ops
    assert "array.__matmul__" in ops
    assert "array.__getitem__" in ops
    assert "array.__setitem__" in ops
    assert "array.__gt__" in ops
    assert "array.__neg__" in ops
    assert "mx.zeros_like" in ops
    # every node inside the module carries a non-empty module address
    assert all(n.module_address for n in trace.nodes)
    # the step output is produced by a recorded call
    assert trace.step_outputs
    assert any(trace.step_outputs[0] in n.out_arrays for n in trace.nodes)


def test_a_finished_trace_keeps_no_hold_on_the_model():
    """The job frees its models before exporting; a trace must not pin one."""
    import gc
    import weakref

    model = load_fixture("repeated_layers")
    x = mx.random.normal((64, 16), key=mx.random.key(0))
    tracer().trace(model, [x])
    gone = weakref.ref(model)
    del model
    gc.collect()
    assert gone() is None


def test_a_model_that_evaluates_mid_pass_frees_its_intermediates_while_recorded():
    """mlx-lm evaluates between 2,048-token prompt chunks. Recording must not
    keep what the model itself let go of: holding every intermediate made a
    4,800-token gpt-oss prompt need ~35 GB to record."""
    import gc
    import weakref

    seen = []

    def model(x):
        h = x * 2.0
        mx.eval(h)  # as between prompt chunks
        first = weakref.ref(h)
        for _ in range(3):
            h = h * 2.0 + 1.0
            mx.eval(h)
        gc.collect()
        seen.append(first() is None)
        return h

    tracer().trace(model, [mx.ones((64, 64))])
    assert seen == [True]


def test_reading_a_result_as_numpy_is_an_evaluation():
    """An app hands its result over as numpy (np.array(model(x))): that read
    goes through __buffer__, and the work behind it must count as run. Missed,
    a scoring function's whole model recorded as never run and gave no region."""
    import numpy as np

    def model(x):
        y = x * 2.0 + 1.0
        return mx.array(np.array(y))

    trace, _ = tracer().trace(model, [mx.ones((4, 4))])
    work = {n.seq for n in trace.nodes if n.op in ("array.__mul__", "array.__add__")}
    assert trace.in_pass_evaluation and work and not work & trace.dead


def test_recording_is_lazy():
    before = mx.get_peak_memory()
    trace, outs = trace_of("repeated_layers", (64, 16))
    assert not trace.in_pass_evaluation
    assert len(trace.nodes) > 0


def test_plain_function_model_has_top_level_address():
    trace, _ = trace_of("plain_function", (4, 16))
    assert [n.op for n in trace.nodes] == ["mx.maximum", "array.__matmul__", "array.__add__"]
    assert all(n.module_address == "@0" for n in trace.nodes)


def test_module_addresses_distinguish_layers():
    trace, _ = trace_of("repeated_layers", (4, 16))
    addresses = {n.module_address for n in trace.nodes}
    assert addresses == {f"layers.{i}@0" for i in range(4)}
    per_layer = [sorted(n.op for n in trace.nodes if n.module_address == f"layers.{i}@0") for i in range(4)]
    assert all(p == per_layer[0] for p in per_layer)


def test_named_compiled_function_records_as_one_call_by_import_path():
    """A compiled section records as one call, never as the ops inside it,
    and one the tracer can name by import path replays through that path."""
    from tests.conftest import arrays_by_path

    model = load_fixture("compiled_submodule")
    x = mx.random.normal((4, 8), key=mx.random.key(0))
    trace, outs = tracer().trace(model, [x])
    ops = [n.op for n in trace.nodes]
    assert "compiled:fixture_compiled_submodule.fast_tanh" in ops
    assert "mx.tanh" not in ops and OPAQUE_OP not in ops
    by_path = arrays_by_path(model)
    bindings = {aid: x for aid in trace.inputs}
    bindings.update({aid: by_path[trace.weight_paths[aid]] for aid in trace.weights
                     if trace.weight_paths.get(aid) in by_path})
    replayed = replay(trace.nodes, bindings, trace.step_outputs)
    mx.eval(list(replayed.values()))
    assert mx.array_equal(replayed[trace.step_outputs[0]], outs).item()


def test_activation_module_records_its_compiled_function():
    """mlx's activation modules (nn.SiLU, nn.ReLU, nn.Sigmoid, ...) hold their
    compiled function inside __call__ rather than looking it up by name; it
    records and replays like any named compiled function."""
    import mlx.nn as nn
    from tests.conftest import arrays_by_path

    class Gated(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.SiLU()
            self.lin = nn.Linear(8, 8)

        def __call__(self, x):
            return self.lin(self.act(x))

    model = Gated()
    x = mx.random.normal((4, 8), key=mx.random.key(0))
    trace, outs = tracer().trace(model, [x])
    assert "compiled:mlx.nn.layers.activations.silu" in [n.op for n in trace.nodes]
    by_path = arrays_by_path(model)
    bindings = {aid: x for aid in trace.inputs}
    bindings.update({aid: by_path[trace.weight_paths[aid]] for aid in trace.weights})
    replayed = replay(trace.nodes, bindings, trace.step_outputs)
    mx.eval(list(replayed.values()))
    assert mx.array_equal(replayed[trace.step_outputs[0]], outs).item()


def test_a_compiled_helper_held_as_an_attribute_records_the_same_every_pass():
    """mlx's TransformerEncoderLayer keeps its activation=relu default as an
    attribute: the compiled function itself. Its first call used to record
    the ops inside it and later, cached calls nothing, so a job's second pass
    disagreed with its first (CaptureMismatch on any such model)."""
    import mlx.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(16, 4)

        def __call__(self, x):
            return self.layer(x, None)

    model = Encoder()
    x = mx.random.normal((2, 8, 16), key=mx.random.key(0))
    passes = [[n.op for n in tracer().trace(model, [x])[0].nodes] for _ in range(2)]
    assert passes[0] == passes[1] and "compiled:mlx.nn.layers.activations.relu" in passes[0]


def test_compiled_lambda_is_one_opaque_call():
    trace, _ = trace_of("opaque_submodule", (4, 8))
    ops = [n.op for n in trace.nodes]
    assert ops.count(OPAQUE_OP) == 1
    # the ops inside the compiled section (tanh, mul) must NOT be recorded
    assert "mx.tanh" not in ops
    # the opaque node anchors completeness: its output feeds the next op
    opaque = next(n for n in trace.nodes if n.op == OPAQUE_OP)
    assert trace.edges.get(opaque.seq)
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        replay([opaque], {aid: mx.zeros((4, 8)) for aid in opaque.in_arrays}, opaque.out_arrays)


def test_named_custom_kernel_records_definition_and_launch():
    """Definition and launch replay bitwise without importing the model."""
    from tests.conftest import arrays_by_path

    model = load_fixture("kernel_submodule")
    x = mx.random.normal((4, 8), key=mx.random.key(0))
    trace, outs = tracer().trace(model, [x])
    ops = [n.op for n in trace.nodes]
    assert len(ops) == 3 and ops[1] == UNNAMED_KERNEL_OP
    node = trace.nodes[1]
    launch = node.scalar_args["kwargs"]
    assert node.scalar_args["args"] == ()
    assert launch["inputs"] == [ArrayRef(0), 4] and launch["grid"] == (32, 1, 1)
    assert launch["template"] == [("T", mx.float32)] and launch["output_shapes"] == [(4, 8)]
    assert node.in_arrays == (trace.nodes[0].out_arrays[0],)
    assert node.out_specs == (((4, 8), "float32"),)
    by_path = arrays_by_path(model)
    bindings = {aid: x for aid in trace.inputs}
    bindings.update({aid: by_path[trace.weight_paths[aid]] for aid in trace.weights
                     if trace.weight_paths.get(aid) in by_path})
    replayed = replay(trace.nodes, bindings, trace.step_outputs)
    mx.eval(list(replayed.values()))
    assert mx.array_equal(replayed[trace.step_outputs[0]], outs).item()


def test_custom_kernel_stand_in_is_transparent_outside_a_pass():
    """Disarmed, the stand-in the factory hands the model forwards the call
    and records nothing; the module teardown checks the factory is restored."""
    load_fixture("kernel_submodule")
    fixture = sys.modules["fixture_kernel_submodule"]
    assert type(fixture.scale_kernel).__name__ == "CapturedKernel"
    before = len(tracer().recorder.nodes)
    out = fixture.scale(mx.arange(8, dtype=mx.float32), 3)
    mx.eval(out)
    assert out.tolist() == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    assert len(tracer().recorder.nodes) == before


def test_custom_kernel_held_on_the_model_is_one_unnamed_opaque_call():
    trace, _ = trace_of("kernel_closure", (4, 8))
    ops = [n.op for n in trace.nodes]
    assert ops.count(UNNAMED_KERNEL_OP) == 1
    # the opaque node anchors completeness: its output feeds the next op
    node = next(n for n in trace.nodes if n.op == UNNAMED_KERNEL_OP)
    assert trace.edges.get(node.seq)
    assert node.kernel_definition["kwargs"]["source"]
    result = replay([node], {aid: mx.zeros((4, 8)) for aid in node.in_arrays}, node.out_arrays)
    mx.eval(list(result.values()))


def test_cache_retention_marks_python_retained():
    trace, _ = trace_of("cache_retention", (4, 16))
    k_producer = trace.nodes[0]
    assert k_producer.op == "array.__matmul__"
    k_id = k_producer.out_arrays[0]
    assert trace.liveness[k_id].kind is Retention.PYTHON_RETAINED
    assert trace.liveness[k_id].consumed_by  # retained AND consumed, the dangerous case


def test_eager_item_records_completely_with_warning_flag():
    trace, _ = trace_of("eager_item", (4, 8))
    assert trace.in_pass_evaluation
    ops = [n.op for n in trace.nodes]
    assert "array.__matmul__" in ops and "mx.abs" in ops
    assert "array.__truediv__" in ops or "mx.divide" in ops


def test_data_branch_records_only_taken_path():
    trace, _ = trace_of("data_branch", (4, 8))
    ops = [n.op for n in trace.nodes]
    assert trace.in_pass_evaluation  # .item() decided the branch
    assert ("mx.tanh" in ops) != ("mx.sigmoid" in ops)


def test_an_unwrapped_entry_points_result_is_an_unknown_source():
    """A call the recorder never sees leaves an array it cannot explain. The
    record keeps it as an unknown source (no value frozen, its real value
    captured at every pass) instead of aborting the job; its consumers are
    recorded as usual."""
    from backcountry.harness.trace.recorder import UNKNOWN_SOURCE_OP, is_opaque
    tr = tracer()

    from backcountry.harness.trace.optable import resolve
    raw_exp = resolve("mx.exp")  # the pre-patch original: a call the recorder never sees

    class Sneaky:
        def __call__(self, x):
            return x[:1] + raw_exp(x[:1])

    trace, _ = tr.trace(Sneaky(), [mx.random.normal((4, 4), key=mx.random.key(0))])
    source = next(n for n in trace.nodes if n.op == UNKNOWN_SOURCE_OP)
    add = next(n for n in trace.nodes if n.op == "array.__add__")
    assert source.out_arrays[0] in add.in_arrays and not source.in_arrays and is_opaque(source.op)


def test_a_small_constant_built_in_forward_records_as_a_creation_call():
    """mx.array(...) is the array class and cannot be wrapped. A small finite
    array made from host data records as that creation call, so the trace has
    its producer, replay rebuilds it with its dtype, and nothing is frozen
    that was computed from the inputs."""
    from backcountry.harness.trace.replay import replay
    tr = tracer()

    class Scaled:
        def __call__(self, x):
            d = x.shape[-1]
            return x / mx.sqrt(mx.array(float(d), dtype=mx.float16))

    x = mx.random.normal((4, 16), key=mx.random.key(0))
    trace, out = tr.trace(Scaled(), [x])
    made = [n for n in trace.nodes if n.op == "mx.array"]
    assert len(made) == 1 and made[0].in_arrays == ()
    assert made[0].scalar_args["args"] == (16.0,) and made[0].scalar_args["kwargs"] == {"dtype": mx.float16}
    sqrt = next(n for n in trace.nodes if n.op == "mx.sqrt")
    assert sqrt.in_arrays == made[0].out_arrays
    (inp,) = trace.inputs
    again = replay(trace.nodes, {inp: x}, trace.step_outputs)[trace.step_outputs[0]]
    assert again.dtype == out.dtype and mx.array_equal(again, out).item()


@pytest.mark.parametrize("constant", [
    lambda x: mx.array([float("nan")] * 4),          # not finite
    lambda x: mx.array([1.0] * 2048).reshape(-1, 4)[:1],  # too large to carry as a literal
])
def test_a_constant_that_cannot_be_carried_is_an_unknown_source(constant):
    from backcountry.harness.trace.recorder import CONSTANT_OP, UNKNOWN_SOURCE_OP
    tr = tracer()

    class Model:
        def __call__(self, x):
            return x[:1] + constant(x)

    trace, _ = tr.trace(Model(), [mx.random.normal((4, 4), key=mx.random.key(0))])
    assert [n.op for n in trace.nodes].count(UNKNOWN_SOURCE_OP) == 1
    assert not any(n.op == CONSTANT_OP for n in trace.nodes)  # nothing carried as a literal


def test_reading_a_constant_is_not_a_mid_step_evaluation():
    """mflux's attention scale, 1 / mx.sqrt(head_dim), is read back as a float
    every call. It comes from Python numbers alone, so mx.compile evaluates it
    too: the step still compiles. A read of anything computed from the input
    or a weight is still flagged."""
    tr = tracer()

    class Scaled:
        def __init__(self, read):
            self.weight = mx.ones((16,))
            self.read = read

        def __call__(self, x):
            return x * float(self.read(self, x))

    constant = Scaled(lambda self, x: 1 / mx.sqrt(x.shape[-1]))
    x = mx.random.normal((4, 16), key=mx.random.key(0))
    trace, _ = tr.trace(constant, [x])
    assert not trace.in_pass_evaluation and not trace.eval_sites
    step = mx.compile(lambda x: constant(x))
    assert mx.allclose(step(x), constant(x)).item()
    for read in (lambda self, x: x.sum(), lambda self, x: self.weight.sum() / x.shape[-1]):
        trace, _ = tr.trace(Scaled(read), [x])
        assert trace.in_pass_evaluation and trace.eval_sites


def test_a_constant_a_call_reads_as_a_number_is_recorded_as_that_number():
    """mflux passes scale=1 / mx.sqrt(head_dim), an array, where attention
    takes a float; the call reads it as a number. Recorded as that number,
    a span cut around the call replays compiled on its own. As an array it
    became a compiled input the call could not read, and pricing crashed;
    counted as step work, its calls ended regions that a compiled step does
    not contain, and installing a kernel there found nothing to replace."""
    from backcountry.harness.trace.replay import compile_replay
    tr = tracer()

    class Attention:
        def __call__(self, q, k, v):
            scale = 1 / mx.sqrt(q.shape[-1])
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    q, k, v = (mx.random.normal((1, 2, 8, 16), key=mx.random.key(i)) for i in range(3))
    trace, out = tr.trace(Attention(), [q, k, v])
    (sdpa,) = [n for n in trace.nodes if n.op == "mx.fast.scaled_dot_product_attention"]
    assert sdpa.scalar_args["kwargs"]["scale"] == 0.25 and len(sdpa.in_arrays) == 3
    assert not trace.in_pass_evaluation
    # a compiled step computes the scale while it is built; its graph holds
    # no such calls, so no region may end on them
    made_scale = {n.seq for n in trace.nodes if n.op != "mx.fast.scaled_dot_product_attention"}
    assert made_scale and made_scale <= trace.dead
    run = compile_replay([sdpa], sdpa.in_arrays, sdpa.out_arrays)
    (again,) = run(dict(zip(sdpa.in_arrays, (q, k, v))))
    assert mx.array_equal(again, out).item()


def test_after_an_in_pass_evaluation_a_second_pass_exposes_data_constants():
    """Once the model has evaluated arrays, a value Python read from the data
    looks like host data, so it is recorded as a constant all the same; a
    second pass on other inputs tells them apart. A value read from the data
    changes and marks every scope around it; one set by the shapes (Gemma 4's
    cache offset, wrapped in mx.array after mlx-lm evaluates the cache) does
    not."""
    import mlx.nn as nn
    from backcountry.harness.trace.recorder import CONSTANT_OP, varying_constant_scopes
    tr = tracer()

    class Scales(nn.Module):
        def __call__(self, x, n):
            scale = 2.0 if x.sum().item() > 0 else 3.0  # read from the data
            return x * mx.array(scale) + mx.array(n)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scales = Scales()

        def __call__(self, x):
            return self.scales(x, x.shape[0]) + 1.0

    pos, neg = mx.ones((4, 4)), -mx.ones((4, 4))
    trace, _ = tr.trace(Model(), [pos])
    assert [n.scalar_args["args"] for n in trace.nodes if n.op == CONSTANT_OP] == [(2.0,), (4,)]
    assert varying_constant_scopes(trace, tr.trace(Model(), [pos * 2])[0]) == set()
    assert varying_constant_scopes(trace, tr.trace(Model(), [neg])[0]) == {"", "scales"}


def test_an_unknown_source_passed_into_a_module_belongs_to_its_caller():
    """The caller made the array and the module only reads it, so the module
    can still host generated code; the caller cannot."""
    import mlx.nn as nn
    from backcountry.harness.bind.certify import screen_scope
    from backcountry.harness.trace.recorder import UNKNOWN_SOURCE_OP
    tr = tracer()

    class Rope(nn.Module):
        def __call__(self, x, offsets):
            return x * offsets[:4]

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = Rope()

        def __call__(self, x):
            return self.rope(x, offsets=mx.array([1.0] * 2048))  # too large to carry

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

        def __call__(self, x):
            return self.attn(x) + 1.0

    trace, _ = tr.trace(Model(), [mx.random.normal((4, 4), key=mx.random.key(0))])
    source = next(n for n in trace.nodes if n.op == UNKNOWN_SOURCE_OP)
    mul = next(n for n in trace.nodes if n.op == "array.__mul__")
    assert source.module_address.startswith("attn@") and mul.module_address.startswith("attn.rope@")
    assert screen_scope(trace, mul.module_stack) is None
    assert "unknown source" in screen_scope(trace, mul.module_stack[:-1])


def test_step_output_also_intermediate():
    trace, _ = trace_of("step_output_region", (4, 16))
    h_node = next(n for n in trace.nodes if n.op == "mx.maximum")
    h_id = h_node.out_arrays[0]
    assert h_id in trace.step_outputs
    assert trace.liveness[h_id].kind is Retention.STEP_OUTPUT
    assert trace.liveness[h_id].consumed_by


def test_norm_three_projections_share_input():
    trace, _ = trace_of("norm_three_proj", (4, 32))
    norm = next(n for n in trace.nodes if n.op == "mx.fast.rms_norm")
    consumers = trace.edges[norm.seq]
    assert len(consumers) == 3


def test_replay_reproduces_library_outputs_bitwise():
    from tests.conftest import arrays_by_path

    model = load_fixture("norm_three_proj")
    x = mx.random.normal((4, 32), key=mx.random.key(7))
    trace, outs = tracer().trace(model, [x])
    by_path = arrays_by_path(model)
    bindings = {aid: x for aid in trace.inputs}
    for aid in trace.weights:
        if aid in trace.weight_paths and trace.weight_paths[aid] in by_path:
            bindings[aid] = by_path[trace.weight_paths[aid]]
    replayed = replay(trace.nodes, bindings, trace.step_outputs)
    mx.eval(list(replayed.values()))
    for got, want in zip([replayed[a] for a in trace.step_outputs], list(outs)):
        assert mx.array_equal(got, want).item()


def test_setitem_renames_ssa():
    trace, _ = trace_of("operator_soup", (4, 8))
    setitem = next(n for n in trace.nodes if n.op == "array.__setitem__")
    buf_before = setitem.in_arrays[0]
    buf_after = setitem.out_arrays[0]
    assert buf_before != buf_after
    # consumers after the write read the renamed id, not the old one
    later_consumers = trace.liveness[buf_after].consumed_by
    assert later_consumers
    assert all(s > setitem.seq for s in later_consumers)
