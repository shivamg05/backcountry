"""A call on an object that holds model state (a KV cache) is one recorded
state call: a barrier for regions, and replayed by the generated wrapper as
the same call on the same object, so the chains on either side of a cache
write can be delivered. Before this every fusion inside a decoder's
attention block stranded on that write."""

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from backcountry.harness.bind.certify import certify_identity, screen_scope
from backcountry.harness.bind.emit import MODULE_HEADER, Splice, emit_wrapper
from backcountry.harness.bind.verify import verify_retrace
from backcountry.harness.regions.build import build_stretches
from backcountry.harness.regions.roofline import step_floor
from backcountry.harness.measure.peaks import Peaks
from backcountry.harness.trace.replay import replay
from backcountry.harness.trace.types import Retention
from backcountry.harness.trace.walk import state_holders
from backcountry_runtime.kernels import KernelSpec
from backcountry_runtime.swap import flatten_arrays, install, uninstall
from tests.conftest import current_tracer, tracer_for_module

_module_tracer = tracer_for_module()

STATE_OP = "state:Cache.update_and_fetch"

# (x + g) * b, then tanh: an add feeding a multiply cannot be contracted into
# a fused multiply-add, and precise:: tanh reproduces the library's bits
FUSED = """\
uint i = thread_position_in_grid.x;
uint c = i % (uint)in0_shape[1];
out0[i] = metal::precise::tanh((in0[i] + in1[c]) * in2[c]);
"""
FUSED_KERNEL = KernelSpec(
    kernel_id="k_pre_cache", name="state_test_pre_cache",
    input_names=("in0", "in1", "in2"), output_names=("out0",),
    source=FUSED, grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
    threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
    output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",),
)


def build_model():
    current_tracer()
    path = Path(__file__).parent / "fixtures" / "state_call.py"
    spec = importlib.util.spec_from_file_location("fixture_state_call", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build()


def traced(model=None, shape=(4, 16)):
    model = model or build_model()
    x = mx.random.normal(shape, key=mx.random.key(5))
    trace, _ = current_tracer().trace(model, [x])
    return model, x, trace


def wrapper_class(emitted):
    ns = {}
    exec(compile(MODULE_HEADER + emitted.source, "<generated>", "exec"), ns)
    return ns[emitted.class_name]


def outputs(model, x, calls=3):
    outs = [model(x) for _ in range(calls)]
    mx.eval(outs)
    return outs


def layer_scope(trace):
    return next(sc for sc in trace.scope_calls if sc.address == "layer@0")


def certify_layer(model, x, trace):
    layer = layer_scope(trace)
    assert screen_scope(trace, layer.stack) is None
    cls = wrapper_class(emit_wrapper(trace, layer, [], "IdLayer"))

    def install_cb(wrapper):
        occupant = install(model, "layer", wrapper)
        return lambda: uninstall(model, "layer", occupant)

    return certify_identity(build_wrapper=lambda: cls(model.layer, {}),
                            install=install_cb, runs=[lambda: model(x)])


def test_the_cache_method_records_as_one_state_call():
    model, x, trace = traced()
    ops = [n.op for n in trace.nodes]
    assert ops == ["array.__add__", "array.__mul__", "mx.tanh", STATE_OP, "array.sum", "array.__add__"]
    state = trace.nodes[3]
    assert state.in_arrays == trace.nodes[2].out_arrays and len(state.out_arrays) == 1
    receiver = state.scalar_args["receiver"]
    assert (receiver["id"], receiver["path"]) == (id(model.cache), "cache")
    # the ops the method ran, collapsed because one of them wrote the cache
    assert receiver["inner_ops"] == ["array.__getitem__", "array.__setitem__", "array.__getitem__"]
    # the write inside the method is the method's business: nothing the
    # model keeps is a recorded production, and the buffer is a weight
    assert not trace.python_retained() and trace.state_calls() == [3]
    assert "cache.keys" in trace.weight_paths.values()
    assert trace.liveness[state.out_arrays[0]].kind is Retention.CONSUMED
    # the layer scope received the cache as an object argument, after a literal
    layer = layer_scope(trace)
    assert layer.obj_ids == (id(model.cache),)
    assert layer.args_template[1] is None


def test_a_state_call_is_a_barrier_and_never_replays_in_process():
    _, _, trace = traced()
    spans = {(s.start_seq, s.end_seq) for s in build_stretches(trace, "w")}
    assert (0, 2) in spans and (4, 5) in spans
    assert not any(a <= 3 <= b for a, b in spans)
    state = trace.nodes[3]
    try:
        replay([state], {state.in_arrays[0]: mx.zeros((4, 16))}, state.out_arrays)
    except RuntimeError as e:
        assert "model's own state" in str(e)
    else:
        raise AssertionError("a state call must not replay in process")


def test_the_scout_line_counts_the_state_a_step_reads_and_the_launches_it_fires():
    """Hiding the cache write must not hide its physics: the bytes the state
    call hands back are state read from memory, and its inner ops are
    launches."""
    _, _, trace = traced()
    f = step_floor(trace, Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 3000.0}), step_ms=1.0)
    x_bytes, g_bytes, kv_bytes = 4 * 16 * 4, 16 * 4, 5 * 16 * 4
    assert f["bytes_mb"] * 1e6 == x_bytes + 2 * g_bytes + kv_bytes + x_bytes  # + the output
    assert f["launches"] == 3 + 1 + 2  # the chain, the setitem inside the state call, sum and add


def test_the_scout_line_reads_the_weights_once_per_call_and_a_lookup_reads_its_rows():
    """A step that calls the model twice (two generated tokens, each needing
    the one before it) reads every weight twice; an embedding lookup reads
    the rows it returns, not the table."""
    class TwoTokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 16)
            self.proj = nn.Linear(16, 16, bias=False)

        def __call__(self, ids):
            first = self.proj(self.embed(ids))
            return self.proj(first)

    model = TwoTokens()
    ids = mx.array([3], dtype=mx.int32)
    trace, _ = current_tracer().trace(model, [ids])
    f = step_floor(trace, Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 3000.0}), step_ms=1.0)
    row, weight, out = 16 * 4, 16 * 16 * 4, 16 * 4
    assert f["bytes_mb"] * 1e6 == 4 + row + 2 * weight + out  # ids, one row, the weight twice, the output


def test_setup_warns_when_even_perfect_kernels_cannot_reach_a_noticeable_gain():
    """Llama 8B decode's floor said -7.8% room before the first kernel and a
    7.4 h run confirmed it: say so at setup, not after the run."""
    from backcountry.harness.loop import little_room
    assert little_room({"decode": {"room": 0.05}, "prefill": {"room": 0.08}}) is not None
    assert "1.05x" in little_room({"decode": {"room": 0.05}})
    assert little_room({"decode": {"room": 0.05}, "prefill": {"room": 0.30}}) is None  # one workload has room
    assert little_room({"decode": {"room": -0.08}}) is not None
    assert little_room({}) is None and little_room({"decode": {"room": None}}) is None


def test_the_layer_scope_certifies_with_the_state_call_replayed():
    """The identity wrapper calls cache.update_and_fetch on the object the
    layer was handed, after a literal argument, so the write and the offset
    bump happen for real and three repeated calls match the original bit
    for bit. Its signature must be valid Python: a positional parameter
    after a literal one takes no default."""
    model, x, trace = traced()
    emitted = emit_wrapper(trace, layer_scope(trace), [], "IdLayer")
    assert "def __call__(self, a0=_UNSET, a1=_UNSET, a2=_UNSET, **_kw):" in emitted.source
    assert "a2.update_and_fetch(" in emitted.source
    result = certify_layer(model, x, trace)
    assert result.ok, result.reason


def test_a_replay_takes_the_recorded_arguments_by_position_or_by_name():
    """A loop may call a module with keywords where the job's recording
    passed them by position (mlx-lm's model passes the cache positionally,
    a loop calling the layers directly passes cache=...). The wrapper puts
    either form in the recorded order by the module's own parameter names;
    an argument the recording never saw runs the module itself."""
    from backcountry_runtime.swap import recorded_form
    layer = build_model().layer  # Layer.__call__(self, x, mask, cache)
    assert recorded_form(layer, (1, None, "c"), {}, 3, {}) == [1, None, "c"]
    assert recorded_form(layer, (1, None), {"cache": "c"}, 3, {}) == [1, None, "c"]
    assert recorded_form(layer, (1,), {"mask": None, "cache": "c"}, 3, {}) == [1, None, "c"]
    assert recorded_form(layer, (1, None, "c"), {"other": 2}, 3, {}) is None
    assert recorded_form(layer, (1, None), {"cache": "c", "x": 0}, 3, {}) is None


def test_a_kernel_ships_before_the_cache_write_and_the_state_advances():
    """The chain before the state call becomes one kernel; the retrace shows
    the cut gone, the state call in place, and the cache holding what the
    kernel wrote."""
    model, x, trace = traced()
    before = outputs(model, x)
    chain = trace.nodes[0:3]
    splice = Splice(kernel=FUSED_KERNEL, start_seq=0, end_seq=2,
                    input_ids=(chain[0].in_arrays[0], chain[0].in_arrays[1], chain[1].in_arrays[1]),
                    output_ids=chain[2].out_arrays, fingerprint="pre_cache")
    emitted = emit_wrapper(trace, layer_scope(trace), [splice], "FusedLayer")
    assert "_kernels.try_call(_s, _ins)" in emitted.source
    cls = wrapper_class(emitted)
    occupant = install(model, "layer", cls(model.layer, {FUSED_KERNEL.kernel_id: FUSED_KERNEL}))
    try:
        after = outputs(model, x)
        assert all(mx.array_equal(a, b).item() for a, b in zip(after, before))
        retrace, _ = current_tracer().trace(model, [x])
        assert [n.op for n in retrace.nodes] == ["custom_kernel", STATE_OP, "array.sum", "array.__add__"]
        report = verify_retrace(trace, retrace, [(0, 2)], [FUSED_KERNEL.kernel_id])
        assert report.ok, report.reasons
        written = model.cache.keys[4]
        want = mx.tanh((x + model.layer.g) * model.layer.b)[0]
        assert mx.array_equal(written, want).item()
    finally:
        uninstall(model, "layer", occupant)
    assert all(mx.array_equal(a, b).item() for a, b in zip(outputs(model, x), before))


class Step(nn.Module):
    """A root that hands its layer a cache object and rewinds it: the shape
    of every decode step, with the cache class chosen per test."""

    def __init__(self, layer, cache):
        super().__init__()
        self.layer = layer
        self.cache = cache

    def __call__(self, x):
        y = self.layer(x, None, self.cache)
        self.cache.offset = 4
        return y


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = mx.ones((16,))

    def __call__(self, x, mask, cache):
        return sum(k.sum(axis=0) for k in flatten_arrays(cache.update_and_fetch(mx.tanh(x * self.g)))) + x


class OwnBuffer:
    """A cache that hands back its buffer itself, as mlx_lm's rotating and
    concatenating caches do."""

    def __init__(self):
        self.keys = mx.zeros((8, 16))
        self.offset = 4

    def update_and_fetch(self, k):
        self.keys[self.offset:self.offset + 1] = k[0:1]
        self.offset += 1
        return self.keys


class Nested(OwnBuffer):
    """A cache that returns a nested structure, as mlx_lm's quantized cache
    does with two triples."""

    def update_and_fetch(self, k):
        keys = super().update_and_fetch(k)
        return (keys[:self.offset], keys[:2]), (keys[:1],)


class Rebinding(OwnBuffer):
    """A cache whose write is a rebinding, not a write in place."""

    def update_and_fetch(self, k):
        self.keys = mx.concatenate([self.keys[:self.offset], k[0:1], self.keys[self.offset + 1:]])
        self.offset += 1
        return self.keys[:self.offset]


class Callable(OwnBuffer):
    """State on an object that is itself callable."""

    def __call__(self, x):
        return x


def test_caches_that_return_their_buffer_or_a_structure_or_rebind_all_deliver():
    for cache_cls in (OwnBuffer, Nested, Rebinding, Callable):
        model = Step(Layer(), cache_cls())
        model, x, trace = traced(model)
        state = [n for n in trace.nodes if n.op.startswith("state:")]
        assert len(state) == 1 and state[0].op == f"state:{cache_cls.__name__}.update_and_fetch", cache_cls
        assert not trace.python_retained(), cache_cls
        result = certify_layer(model, x, trace)
        assert result.ok, (cache_cls, result.reason)
        assert model.cache.offset == 4


class ArrayOffset(OwnBuffer):
    """A cache whose position is an array the model reads (mlx-lm's
    BatchKVCache: one offset per sequence, fed to rope)."""

    def __init__(self):
        super().__init__()
        self.offset = mx.array([4])

    def update_and_fetch(self, k):
        self.offset = self.offset + 1
        return self.keys


class Reads(nn.Module):
    def __call__(self, x, cache):
        return x * cache.offset.astype(mx.float32) + cache.update_and_fetch(x).sum(axis=0)


class TwoSteps(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer, self.cache = Reads(), ArrayOffset()

    def __call__(self, x):
        return self.layer(self.layer(x, self.cache), self.cache)


def test_an_array_a_cache_call_keeps_on_its_object_is_read_later_as_an_unexplained_array():
    # the offset made inside the first call lost its producer when the call
    # collapsed into one state call, kept its id, and the second step's read
    # aborted the pass: "not a workload input, a weight, or an earlier output"
    from backcountry.harness.trace.recorder import CONSTANT_OP, UNKNOWN_SOURCE_OP
    model, x, trace = traced(TwoSteps())
    reads = [n for n in trace.nodes if n.op == "array.astype"]
    sources = {aid: n.op for n in trace.nodes for aid in n.out_arrays}
    assert len(reads) == 2 and sources[reads[1].in_arrays[0]] in (UNKNOWN_SOURCE_OP, CONSTANT_OP)


class Helper:
    """A stateless helper that holds a table: its method is pure, so its
    ops stay in the record and can be regions, and the step keeps no state."""

    def __init__(self):
        self.table = mx.arange(16, dtype=mx.float32)

    def scale(self, x):
        return x * self.table


class PureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.helper = Helper()

    def __call__(self, x, mask, cache):
        return mx.tanh(self.helper.scale(x)) + x


def test_a_pure_helper_stays_visible_and_keeps_no_state():
    model = Step(PureLayer(), OwnBuffer())
    model.layer.helper.scale(mx.ones((4, 16)))  # a call before tracing changes nothing below
    model, x, trace = traced(model)
    assert [n.op for n in trace.nodes] == ["array.__mul__", "mx.tanh", "array.__add__"]
    assert not trace.state_calls() and not trace.python_retained()
    assert "layer.helper.table" in trace.weight_paths.values()
    assert dict(state_holders(model))["layer.helper"] is model.layer.helper
    result = certify_layer(model, x, trace)
    assert result.ok, result.reason


def test_an_object_on_a_dead_holders_id_is_not_taken_for_a_holder():
    # mlx-lm's batch_generate makes and drops caches every step; a new object
    # came back on a dead holder's id and was recorded as one on some passes
    # only, so the same request recorded different work each time (35,091,
    # 35,078 and 35,161 calls) and check stopped a batch app as if it kept a cache
    import weakref
    from backcountry.harness.trace.recorder import Recorder
    recorder, holder, other = Recorder(), OwnBuffer(), OwnBuffer()
    recorder.add_holder(holder, "cache")
    assert recorder.is_holder(holder) and not recorder.is_holder(other)
    recorder._holder_refs[id(other)] = weakref.ref(holder)   # other now sits on that id
    assert not recorder.is_holder(other)
