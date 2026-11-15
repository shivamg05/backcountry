"""Captured boundaries must retain the value at their recorded operation."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from backcountry.harness.trace import Tracer
from backcountry.harness.regions.price import capture_boundaries
from backcountry_runtime.exact import bitwise_equal


@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
@pytest.mark.parametrize('mutate_input', [False, True])
def test_capture_preserves_each_version_before_later_inplace_updates(dtype, mutate_input):
    class Model(nn.Module):
        def __call__(self, x):
            y = x if mutate_input else x * 2
            y += 3
            y *= 4
            y[0] = 0
            return y

    model = Model()
    original = mx.arange(8).astype(dtype)
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [mx.array(original)])
        ids = set(trace.inputs) | {a for node in trace.nodes for a in node.out_arrays}
        captured = capture_boundaries(tracer, model, [mx.array(original)], trace, ids)
    finally:
        tracer.uninstall()
    assert not trace.in_pass_evaluation
    assert bitwise_equal(captured[next(iter(trace.inputs))], original)
    value = mx.array(original)
    for node in trace.nodes:
        assert len(node.out_arrays) == 1, node.op
        if node.op == 'array.__mul__': value = value * 2
        elif node.op == 'array.__iadd__': value = value + 3
        elif node.op == 'array.__imul__': value = value * 4
        elif node.op == 'array.__setitem__':
            value = mx.array(value)
            value[0] = 0
        else: raise AssertionError(node.op)
        assert bitwise_equal(captured[node.out_arrays[0]], value), node.op


def test_inplace_identity_wrapper_preserves_values_and_recorded_operations():
    from backcountry.harness.bind.emit import MODULE_HEADER, emit_wrapper
    from backcountry.harness.bind.verify import verify_retrace

    class Step(nn.Module):
        def __call__(self, x):
            y = x * 2
            y += 3
            y *= 4
            y[0] = 0
            return y

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.step = Step()

        def __call__(self, x):
            return self.step(x)

    model = Model()
    x = mx.arange(8).astype(mx.float16)
    tracer = Tracer()
    tracer.install()
    try:
        trace, expected = tracer.trace(model, [x])
        scope = next(s for s in trace.scope_calls if s.address == 'step@0')
        emitted = emit_wrapper(trace, scope, [], 'IdentityStep')
        namespace = {}
        exec(compile(MODULE_HEADER + emitted.source, '<identity>', 'exec'), namespace)
        model.step = namespace[emitted.class_name](model.step, {})
        retrace, actual = tracer.trace(model, [x])
        assert bitwise_equal(actual, expected)
        report = verify_retrace(trace, retrace, [], [])
        assert report.ok, report.reasons
    finally:
        tracer.uninstall()


def _weight_id(trace, path):
    return next(aid for aid, p in trace.weight_paths.items() if p == path)


def test_capture_hands_back_an_unchanged_weight_itself_not_a_copy():
    # A copy of every weight a region reads tipped Qwen3.8-27B's capture pass
    # over the GPU's limit: its two vocabulary tables are 0.76 GB each.
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = mx.random.normal((2048, 4096))  # 32 MB

        def __call__(self, x):
            return x @ self.table

    model = Model()
    mx.eval(model.parameters())
    x = mx.ones((4, 2048))
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [x])
        aid = _weight_id(trace, 'table')
        mx.clear_cache()
        before = mx.get_active_memory()
        captured = capture_boundaries(tracer, model, [x], trace, {aid})
        grew = mx.get_active_memory() - before
    finally:
        tracer.uninstall()
    assert captured[aid] is model.table
    assert grew < model.table.nbytes // 2


def test_capture_keeps_a_weights_value_from_before_the_pass_changed_it():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.buf = mx.zeros((4,))

        def __call__(self, x):
            self.buf[0] = x[0]   # in place: the object the model holds changes
            return x + self.buf

    model = Model()
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [mx.array([1.0, 2.0, 3.0, 4.0])])
        aid = _weight_id(trace, "buf")
        before = mx.array(model.buf)
        mx.eval(before)
        captured = capture_boundaries(tracer, model, [mx.array([7.0, 2.0, 3.0, 4.0])], trace, {aid})
    finally:
        tracer.uninstall()
    assert bitwise_equal(captured[aid], before)
    assert model.buf[0].item() == 7.0
