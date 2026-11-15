"""A generated workload (generated: true): the calls a speculative loop makes,
reading several generated tokens at once over a conversation. Its step runs
under exact_rows, its recording shows the row-exact kernels as recorded
kernel calls (nothing unrecorded), and its wrappers serve only such calls."""

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from backcountry.harness.bind.emit import _variant_checks
from backcountry.harness.trace.recorder import UNKNOWN_SOURCE_OP
from backcountry_runtime.state import context_step
from tests.conftest import current_tracer, tracer_for_module
from tests.test_exact_rows import tiny

_module_tracer = tracer_for_module()


def test_a_generated_step_records_its_exact_kernels_and_guards_its_wrappers():
    model = tiny(dtype=mx.bfloat16)
    history, new = mx.arange(24)[None] % 512, mx.array([[5, 9]])
    step = context_step(model, 24, history, [new], generated=True)
    trace, outs = current_tracer().trace(step, [new])
    mx.eval(outs)
    ops = [n.op for n in trace.nodes]
    assert trace.generated and UNKNOWN_SOURCE_OP not in ops
    assert ops.count("metal_kernel") >= 2 * 7  # every linear of both layers, as a recorded kernel
    # the recorded step computes what one-token calls compute
    cache = make_prompt_cache(model)
    mx.eval(model(history, cache=cache))
    one = mx.concatenate([model(new[:, i:i + 1], cache=cache) for i in range(2)], axis=1)
    assert mx.array_equal(step(new), one).item()
    scope = next(s for s in trace.scope_calls if s.address.endswith("self_attn@0"))
    assert "_rows_exact(a0, a2) is True" in _variant_checks(trace, scope)
    plain, _ = current_tracer().trace(context_step(model, 24, history, [new]), [new])
    assert not plain.generated and "metal_kernel" not in [n.op for n in plain.nodes]


def test_kernels_run_only_on_calls_of_the_kind_they_were_found_on():
    """A kernel found on calls that read generated tokens declines every other
    call, and any other kernel declines those calls: a 2-token user message and
    a 2-token check of guesses have the same shape but not the same reference."""
    from backcountry_runtime import kernels
    from backcountry_runtime.exact_rows import generating, on
    model = tiny()
    cache = make_prompt_cache(model)
    mx.eval(model(mx.arange(12)[None], cache=cache))
    with generating(model, start=12):
        model(mx.array([[3]]), cache=cache)  # one token: plain decoding
        assert not on() and kernels.try_call(None, [], generated=True) is None
        model(mx.array([[4, 5]]), cache=cache)  # two generated tokens at once
        assert on() and kernels.try_call(None, [], generated=False) is None
        assert kernels.try_sized(None, [], generated=False) is None
    assert not on()


def test_a_replay_around_the_layers_serves_the_generated_call_it_recorded():
    """A cut that crosses a layer boundary (a residual add, then the next
    layer's norm) installs around the module holding the layers, whose own
    call is where exact_rows decides a call reads generated tokens. The
    wrapper runs first, so it decides from its own input and cache: it serves
    the call it recorded, with one-token arithmetic, and the output layer
    after it runs row-exact too."""
    from backcountry.harness.bind.emit import MODULE_HEADER, emit_wrapper
    from backcountry_runtime.swap import install
    model = tiny(dtype=mx.bfloat16)
    history, new = mx.arange(24)[None] % 512, mx.array([[5, 9]])
    step = context_step(model, 24, history, [new], generated=True)
    trace, _ = current_tracer().trace(step, [new])
    current_tracer().uninstall()
    scope = next(s for s in trace.scope_calls if s.address == "model.model@0")
    emitted = emit_wrapper(trace, scope, [], "LayersReplay")
    ns = {}
    exec(compile(MODULE_HEADER + emitted.source, "<generated>", "exec"), ns)
    inner = type(model.model)
    install(step, "model.model", ns["LayersReplay"](model.model, {}))
    calls, original = [0], inner.__call__

    def counted(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    inner.__call__ = counted
    try:
        out = step(new)
        mx.eval(out)
    finally:
        inner.__call__ = original
        current_tracer().install()  # the module's later tests record with it
    assert calls[0] == 0  # the replay served: the layers' own code never ran
    cache = make_prompt_cache(model)
    mx.eval(model(history, cache=cache))
    one = mx.concatenate([model(new[:, i:i + 1], cache=cache) for i in range(2)], axis=1)
    assert mx.array_equal(out, one).item()


def test_a_job_on_check_calls_keeps_only_regions_that_serve_every_position():
    """A loop checks guesses at every position, and a recorded replay of a
    scope that holds the cache serves only the position it recorded. On
    Qwen3.8-27B such a kernel served one of 64 calls and a later kernel
    composed into it did too. A job on check calls keeps the regions whose
    scope holds no cache (the MLP) and leaves the rest (attention)."""
    from backcountry.harness.loop import JobRunner, build_stretches, group_copies
    model = tiny(dtype=mx.bfloat16)
    history, new = mx.arange(24)[None] % 512, mx.array([[5, 9]])
    trace, _ = current_tracer().trace(context_step(model, 24, history, [new], generated=True), [new])
    runner = object.__new__(JobRunner)
    runner.traces = {"w": trace}
    regions = group_copies(runner.traces, {"w": build_stretches(trace, "w")})
    inner = lambda r: r.members[0].scope_stack[-1]
    mlp = [r for r in regions if inner(r).endswith(".mlp@0")]
    attention = [r for r in regions if inner(r).endswith(".self_attn@0")]
    assert mlp and not any(runner._holds_cache(r) for r in mlp)
    assert attention and all(runner._holds_cache(r) for r in attention)
    # its calls read generated tokens, so delivery replays: no scope is certified
    # or clocked compiled (on Qwen3.8-27B that certification ran out of GPU memory)
    import types
    rows = []
    runner.baseline, runner.replay_scopes = "plain", set()
    runner.context = types.SimpleNamespace(generated=True)
    runner.log = types.SimpleNamespace(append=lambda kind, **row: rows.append((kind, row)))
    runner._identity = lambda *a: (_ for _ in ()).throw(AssertionError("certified a scope"))
    for r in mlp:
        r.delivery["w"] = "graph"
    runner._settle_delivery(mlp)
    assert all(r.delivery["w"] == "replay" for r in mlp) and rows[-1][0] == "delivery_settled"
    assert rows[-1][1]["graph"] == [] and runner.replay_scopes


def test_a_loop_that_leaves_a_guess_in_the_cache_when_stopped_early_is_seen():
    """A caller stops reading at an end-of-reply token, and a chat's next turn
    continues from the cache it leaves: Qwen3.8-27B's loop left one guessed
    token more there, and every later turn differed. The checked call that
    stops part way sees it; a run to max_tokens does not."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step as stock

    from backcountry_runtime.generation import generation_run
    from tests.test_exact_rows import tiny

    def leaves_a_guess(prompt, model, **kwargs):
        try:
            yield from stock(prompt, model, **kwargs)
        except GeneratorExit:  # stopped early: one more token fed, as a guess left behind
            model(mx.array([[1]]), cache=kwargs["prompt_cache"])
            raise

    model, prompt = tiny(), mx.arange(20) % 512
    outputs = {}
    for name, step in (("stock", stock), ("candidate", leaves_a_guess)):
        run = generation_run(step)
        outputs[name] = [run(model, prompt, helpers={"steps": 16, "call": call})
                         for call in ({"greedy": True}, {"greedy": True, "stop_after": 7})]
    (full, stopped), (full_c, stopped_c) = outputs["stock"], outputs["candidate"]
    for key in ("tokens", "cache_offsets", "cache_digest"):
        assert mx.array_equal(full[key], full_c[key]).item()
    assert stopped["tokens"].size == 7 and mx.array_equal(stopped["tokens"], stopped_c["tokens"]).item()
    assert not mx.array_equal(stopped["cache_offsets"], stopped_c["cache_offsets"]).item()
    assert not mx.array_equal(stopped["cache_digest"], stopped_c["cache_digest"]).item()
