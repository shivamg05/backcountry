"""A kernel inside a block the model hands its cache to: the judge's method
for the block calls the kernel from the block's own code, so it runs at every
decode position instead of only where the block was recorded (plus a few)."""

import json
import re
import subprocess
import sys
import textwrap
from collections import Counter

import mlx.core as mx
import pytest

from backcountry.harness.ladder.gates import LadderResult
from backcountry.harness.loop import LIVE_ATTEMPTS, RegionRun
from backcountry_runtime import kernels as runtime_kernels
from backcountry_runtime.graph import UNRECORDED_STATE_TRACES, GraphWrapper
from tests.test_context_workload import MUL_KERNEL, _resolve, _runner

METHOD = '''class {name}(ReplayWrapper):
    def __call__(self, x, cache=None):
        out = _kernels.try_call(self._specs["ctx_mul"], [x, self.wk])
        keys = out[0] if out is not None else x * self.wk
        if cache is not None:
            keys = cache.update_and_fetch(keys)
        h = x + mx.sum(keys, axis=1, keepdims=True)
        out = _kernels.try_call(self._specs["ctx_mul"], [h, self.wo])
        return out[0] if out is not None else h * self.wo
'''
WRONG = METHOD.replace("return out[0] if", "return 1.0 + out[0] if")
NO_KERNEL = '''class {name}(ReplayWrapper):
    def __call__(self, x, cache=None):
        keys = x * self.wk
        if cache is not None:
            keys = cache.update_and_fetch(keys)
        return (x + mx.sum(keys, axis=1, keepdims=True)) * self.wo
'''
PATHS = ("model.layers.0", "model.layers.1")


class MethodJudge:
    """Writes one method body for every block and accepts every review."""

    def __init__(self, body):
        self.body, self.briefings = body, []

    def _ask(self, system, messages):
        brief = json.loads(messages[0]["content"])
        if "class_name" not in brief:
            return json.dumps({"verdict": "accept", "reason": "the block's own work"})
        self.briefings.append(brief)
        return json.dumps({"hypothesis": "the kernel from the block's code",
                           "source": self.body.format(name=brief["class_name"])})


def _bound(tmp_path, monkeypatch, judge, kernel=MUL_KERNEL):
    runner = _runner(tmp_path, context=4, monkeypatch=monkeypatch)
    runner.judge_factory = lambda region: judge
    try:
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if r.ops == ("array.__mul__",))
        assert runner._bind_and_promote(
            RegionRun(region=region), kernel,
            LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])), runner.log.rows()[-1]
        assert all(isinstance(_resolve(runner.model, p), GraphWrapper) for p in PATHS)
        runner._live_memory_scopes()
    finally:
        runner.tracer.uninstall()
    return runner


def _decode(model, steps, monkeypatch):
    """Outputs and kernel launches of a prompt then `steps` one-token calls
    from an empty cache, far past the positions the job recorded."""
    launched = Counter()
    real = runtime_kernels.try_call

    def counting(spec, inputs):
        out = real(spec, inputs)
        launched[spec.kernel_id] += out is not None
        return out

    monkeypatch.setattr(runtime_kernels, "try_call", counting)
    cache, outs = model.make_cache(), []
    outs.append(model(mx.array([[1, 2, 3]]), cache=cache))
    launched.clear()
    for t in range(steps):
        outs.append(model(mx.array([[t % 32]]), cache=cache))
    mx.eval(outs)
    monkeypatch.setattr(runtime_kernels, "try_call", real)
    return outs, launched[MUL_KERNEL.kernel_id]


def test_a_block_method_serves_every_decode_position(tmp_path, monkeypatch):
    judge = MethodJudge(METHOD)
    runner = _bound(tmp_path, monkeypatch, judge)
    assert runner.live_scopes == set(PATHS)
    # the second layer is the same block class with the same kernel: the first
    # layer's method is tried there first, checked there, and the judge is not asked
    assert [b["class_name"] for b in judge.briefings] == ["L_model_layers_0"]
    assert "x * self.wk" in judge.briefings[0]["module_source"]
    assert "ctx_mul" in judge.briefings[0]["reference"]
    steps = 3 * UNRECORDED_STATE_TRACES
    patched, launches = _decode(runner.model.model, steps, monkeypatch)
    original, _ = _decode(runner.baseline_model.model, steps, monkeypatch)
    assert all(mx.array_equal(a, b).item() for a, b in zip(patched, original))
    assert launches == 2 * len(PATHS) * steps  # two cuts per layer, every token
    assert all(runner.emitted[p].class_name == f"L_{p.replace('.', '_')}" for p in PATHS)


def test_a_wrong_method_is_refused_and_the_recording_stays(tmp_path, monkeypatch):
    judge = MethodJudge(WRONG)
    runner = _bound(tmp_path, monkeypatch, judge)
    assert runner.live_scopes == set()
    # the second layer is the same block class: it is not asked about again
    assert len(judge.briefings) == LIVE_ATTEMPTS
    assert "outputs differ" in judge.briefings[1]["previous_attempt"]["problem"]
    assert all(isinstance(_resolve(runner.model, p), GraphWrapper) for p in PATHS)
    rows = [r for r in runner.log.rows() if r.get("kind") == "live_method"]
    assert [r["installed"] for r in rows] == [False, False]


def test_a_method_that_skips_the_kernel_is_refused(tmp_path, monkeypatch):
    runner = _bound(tmp_path, monkeypatch, MethodJudge(NO_KERNEL))
    assert runner.live_scopes == set()
    rows = [r for r in runner.log.rows() if r.get("kind") == "live_method"]
    assert "on 0 of" in rows[0]["reason"] and "same block class" in rows[1]["reason"]


def test_the_bundle_carries_the_method_into_a_fresh_process(tmp_path, monkeypatch):
    runner = _bound(tmp_path, monkeypatch, MethodJudge(METHOD))
    runner.final_ok = True
    out = runner._emit_artifact(tmp_path / "artifact")
    shipped = (out / "patch" / "wrappers.py").read_text()  # one class per distinct body, named for its kernel
    assert '_kernels.try_call(self._specs["ctx_mul"], [x, self.wk])' in shipped and "_graph_call" not in shipped
    child = textwrap.dedent(f"""
        import importlib.util
        from pathlib import Path
        import mlx.core as mx
        art = Path({str(out)!r})
        spec = importlib.util.spec_from_file_location("load", art / "load.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        original = loader.load(patched=False).model.model
        patched = loader.load().model.model
        import backcountry_runtime.kernels as k
        real, launched = k.try_call, [0]
        def counting(spec, inputs):
            out = real(spec, inputs)
            launched[0] += out is not None
            return out
        k.try_call = counting
        ca, cb = original.make_cache(), patched.make_cache()
        same = mx.array_equal(original(mx.array([[1, 2, 3]]), cache=ca), patched(mx.array([[1, 2, 3]]), cache=cb)).item()
        launched[0] = 0
        for t in range(30):
            same &= mx.array_equal(original(mx.array([[t]]), cache=ca), patched(mx.array([[t]]), cache=cb)).item()
        print("RESULT", same, launched[0])
    """)
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    same, launched = proc.stdout.split("RESULT ")[1].split()
    assert same == "True" and int(launched) == 2 * len(PATHS) * 30


BLOCKS = {
    "llama_cache_model.py": '''class {name}(ReplayWrapper):
    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        o = _kernels.try_call(self._specs["{kid}"], [x, r])
        h = o[0] if o is not None else x + r
        r = self.mlp(self.post_attention_layernorm(h))
        o = _kernels.try_call(self._specs["{kid}"], [h, r])
        return o[0] if o is not None else h + r
''',
    "qwen_cache_model.py": '''class {name}(ReplayWrapper):
    def __call__(self, x, mask=None, cache=None):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        o = _kernels.try_call(self._specs["{kid}"], [x, r])
        h = o[0] if o is not None else x + r
        m = self.mlp(self.post_attention_layernorm(h))
        o = _kernels.try_call(self._specs["{kid}"], [h, m])
        return o[0] if o is not None else h + m
''',
}


class BlockJudge:
    """Writes the given block method (or hands back the job's own recording
    as the 'method'), and accepts every review."""

    def __init__(self, template=None):
        self.template, self.briefings = template, []

    def _ask(self, system, messages):
        brief = json.loads(messages[0]["content"])
        if "class_name" not in brief:
            return json.dumps({"verdict": "accept", "reason": "the block's own work"})
        self.briefings.append(brief)
        if self.template is None:  # the recording, renamed: it cannot serve new positions
            source = re.sub(r"^class \w+\(", f"class {brief['class_name']}(", brief["reference"], count=1, flags=re.M)
        else:
            source = self.template.format(name=brief["class_name"], kid=next(iter(brief["kernels"])))
        return json.dumps({"hypothesis": "the kernel from the block's code", "source": source})


@pytest.mark.parametrize("fixture", sorted(BLOCKS))
@pytest.mark.parametrize("method", ["written", "recording"])
def test_real_blocks_take_a_method_and_refuse_a_recording(tmp_path, monkeypatch, fixture, method):
    """Real mlx-lm decoder blocks (Llama; Qwen3.5 with recurrent layers),
    which receive the cache: the residual-add kernel called from the block's
    own method runs at every later position, bit for bit; the job's own
    recording offered as the method is refused, since it fits only the
    positions it was recorded at."""
    from pathlib import Path
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.measure.session import Session
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {Path(__file__).parent / 'fixtures' / fixture}
model_type: other
baseline: plain
use_library_inference: false
workloads:
  - name: prompt
    context: 5
    inputs: [{{shape: [1, 4], dtype: int32, high: 64}}]
""")
    judge = BlockJudge(BLOCKS[fixture] if method == "written" else None)
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: judge, clock_pairs=4,
                       session=Session(sleep=lambda _: None))
    monkeypatch.setattr(runner, "_model_win", lambda result: all(c.passed for c in result.checks))
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if r.ops == ("array.__add__",) and not r.rejected)
        kernel = runner._build_scaffold(region)
        region.p["prompt"] = 1.0
        region.t_orig_ms["prompt"] = region.t_rep_ms["prompt"] = 1.0
        runner.tracer.uninstall()
        assert runner._bind_and_promote(RegionRun(region), kernel,
                                        LadderResult("tentative_ship", None, {}, .1, 1., .1, 0., [])), \
            runner.log.rows()[-1]
        runner._live_memory_scopes()
    finally:
        runner.tracer.uninstall()
    rows = [r for r in runner.log.rows() if r.get("kind") == "live_method"]
    assert rows, "no block that receives the cache carried the kernel"
    if method == "recording":
        # a block whose state keeps one size (Qwen3.5's recurrent layer) looks
        # the same at every position, so even its recording serves them all;
        # a block whose cache grows (any attention block) is refused
        attention = {p for p in (r["scope"] for r in rows)
                     if "self_attn" in _resolve(runner.model, p).wrapped}
        assert attention and all(not r["installed"] for r in rows if r["scope"] in attention), \
            [r["reason"] for r in rows]
        return
    assert runner.live_scopes == {r["scope"] for r in rows} and all(r["installed"] for r in rows), [r["reason"] for r in rows]
    # far past the recorded position, from an empty cache: same outputs, kernel every call
    launched = Counter()
    real = runtime_kernels.try_call

    def counting(spec, inputs):
        out = real(spec, inputs)
        launched[spec.kernel_id] += out is not None
        return out

    patched, original = runner.model.model, runner.baseline_model.model
    ca, cb = patched.make_cache(), original.make_cache()
    x = runner.tensors["prompt"][0]
    for call in range(12):
        monkeypatch.setattr(runtime_kernels, "try_call", counting)
        a = patched(x, cache=ca)
        monkeypatch.setattr(runtime_kernels, "try_call", real)
        b = original(x, cache=cb)
        assert mx.array_equal(a, b).item(), call
    assert launched[kernel.kernel_id] == 2 * len(runner.live_scopes) * 12



def test_the_per_size_step_widens_a_block_methods_kernels(tmp_path, monkeypatch):
    """Over a range of new-token counts on top of a context, a kernel that a
    block's own method calls runs at every count the per-size step checked it
    at, not only the recorded one; withdrawing the step narrows it again."""
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.measure.session import Session
    from tests.test_context_workload import MODEL
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: all(c.passed for c in result.checks))
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": True} for n in grid})
    monkeypatch.setattr(JobRunner, "_holdout", lambda self, *a: None)  # needs a priced region, tested elsewhere
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {MODEL}
model_type: other
baseline: plain
workloads:
  - name: decode
    inputs: [{{shape: [1, L], dtype: int32, high: 32}}]
    context: 4
primary: {{L: 2}}
range: {{L: [1, 48]}}
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: MethodJudge(METHOD),
                       clock_pairs=4, session=Session(sleep=lambda seconds: None))
    runner.load_model()
    try:
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if r.ops == ("array.__mul__",))
        runner._capture([region])
        assert runner._bind_and_promote(
            RegionRun(region=region), MUL_KERNEL,
            LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])), runner.log.rows()[-1]
        runner._live_memory_scopes()
    finally:
        runner.tracer.uninstall()
    assert runner.live_scopes == set(PATHS)

    def at(count):
        """Kernel launches and whether the outputs match, for `count` new tokens."""
        launched = Counter()
        real = runtime_kernels.try_call
        def counting(spec, ins):
            out = real(spec, ins)
            launched[out is not None] += 1
            return out
        monkeypatch.setattr(runtime_kernels, "try_call", counting)
        tokens = mx.array([[t % 32 for t in range(count)]])
        same = mx.array_equal(runner.model(tokens), runner.baseline_model(tokens)).item()
        monkeypatch.setattr(runtime_kernels, "try_call", real)
        return launched[True], same

    recorded = {int(label.rsplit("=", 1)[1]) for label in runner.traces if "=" in label} | {
        2, *runner.manifest.serve["L"].recordings(2)}
    unrecorded = next(n for n in range(3, 48) if n not in recorded)
    assert at(unrecorded) == (0, True)  # before the step: only the recorded shapes
    restore = runner._serve_sizes()
    assert restore
    assert at(unrecorded) == (2 * len(PATHS), True)  # two cuts per layer, at a count never recorded
    assert at(2) == (2 * len(PATHS), True)
    runner._unserve(restore, "test")
    assert at(unrecorded) == (0, True) and runner.live_scopes == set(PATHS)


HISTORY_METHOD = '''class {name}(ReplayWrapper):
    def __call__(self, x, cache=None):
        out = _kernels.try_call(self._specs["ctx_mul"], [x, self.wk])
        keys = out[0] if out is not None else x * self.wk
        if cache is not None:
            keys = cache.update_and_fetch(keys)
        out = _kernels.try_call(self._specs["ctx_mul"], [keys, self.wk])
        keys = out[0] if out is not None else keys * self.wk
        h = x + mx.sum(keys, axis=1, keepdims=True)
        out = _kernels.try_call(self._specs["ctx_mul"], [h, self.wo])
        return out[0] if out is not None else h * self.wo
'''


def test_a_kernel_on_the_conversation_runs_across_a_span_of_its_lengths(tmp_path, monkeypatch):
    """A kernel a block's method calls on the whole conversation (its input
    grows with it, as attention's does) is recorded at one length. The
    history step checks it at lengths spread over a span, bit for bit over
    calls in a row, and it then runs at a length the job never recorded;
    withdrawing the step holds it at the recorded length again."""
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.measure.session import Session
    from backcountry_runtime.stats import comparison_from_samples
    from pathlib import Path
    import backcountry.harness.loop as loop_mod
    model_file = Path(__file__).parent / "fixtures" / "history_model.py"
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: all(c.passed for c in result.checks))
    monkeypatch.setattr(JobRunner, "HISTORY_SPAN", (4, 64))
    monkeypatch.setattr(JobRunner, "HISTORY_POINTS", 4)
    monkeypatch.setattr(JobRunner, "HISTORY_STEPS", 3)
    # an elementwise stand-in never beats the library on a clock: the timing is told it ties
    monkeypatch.setattr(loop_mod, "compare", lambda *a, **k: comparison_from_samples([1.0] * 4, [1.0] * 4))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {model_file}
model_type: other
baseline: plain
workloads:
  - name: decode
    inputs: [{{shape: [1, 1], dtype: int32, high: 32}}]
    context: 8
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: MethodJudge(HISTORY_METHOD),
                       clock_pairs=4, session=Session(sleep=lambda seconds: None))
    runner.load_model()
    try:
        runner.trace_workloads()
        region = max((r for r in runner.build_regions() if r.ops == ("array.__mul__",)), key=lambda r: len(r.members))
        assert runner._bind_and_promote(
            RegionRun(region=region), MUL_KERNEL,
            LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])), runner.log.rows()[-1]
        runner._live_memory_scopes()
    finally:
        runner.tracer.uninstall()
    assert runner.live_scopes == set(PATHS)

    def history_calls(length):
        """Kernel launches on the conversation, and whether the outputs match, one call after `length` tokens."""
        ran = []
        real = runtime_kernels.try_call

        def counting(spec, ins):
            out = real(spec, ins)
            if out is not None and ins[0].shape[1] > 1:
                ran.append(ins[0].shape[1])
            return out
        caches = runner._history_cache(runner.model.model, length), \
            runner._history_cache(runner.baseline_model.model, length)
        monkeypatch.setattr(runtime_kernels, "try_call", counting)  # the decode call alone
        a = runner.model.model(runner.tensors["decode"][0], cache=caches[0])
        monkeypatch.setattr(runtime_kernels, "try_call", real)
        b = runner.baseline_model.model(runner.tensors["decode"][0], cache=caches[1])
        return len(ran), mx.array_equal(a, b).item()

    assert history_calls(20) == (0, True)  # before the step: only the recorded length
    restore = runner._serve_history_lengths()
    assert restore
    span = next(r for r in runner.log.rows() if r.get("kind") == "history_span")["span"]
    assert span[0] <= 8 <= span[1] and span[1] > 8
    assert history_calls(20) == (len(PATHS), True)  # a length never recorded
    runner._unserve(restore, "test")
    assert history_calls(20) == (0, True)


def _prompt_job(tmp_path, monkeypatch, judge):
    """An MLX-LM prompt-read job on a real Llama block that receives the
    cache, with a residual-add kernel installed and the method step run."""
    from pathlib import Path
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.measure.session import Session
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {Path(__file__).parent / 'fixtures' / 'llama_cache_model.py'}
model_type: llm
baseline: plain
use_library_inference: true
workloads:
  - name: prompt
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: judge, clock_pairs=4,
                       session=Session(sleep=lambda _: None))
    monkeypatch.setattr(runner, "_model_win", lambda result: all(c.passed for c in result.checks))
    monkeypatch.setattr(runner, "_holdout", lambda *a: None)  # a stand-in kernel never beats the library
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if r.ops == ("array.__add__",) and not r.rejected)
        kernel = runner._build_scaffold(region)
        runner._capture([region])
        for w in region.workloads:  # MLX-LM prompt jobs also record other prompt sizes
            region.p[w] = 1.0
            region.t_orig_ms[w] = region.t_rep_ms[w] = 1.0
        runner.tracer.uninstall()
        assert runner._bind_and_promote(RegionRun(region), kernel,
                                        LadderResult("tentative_ship", None, {}, .1, 1., .1, 0., [])), \
            runner.log.rows()[-1]
        runner._live_memory_scopes()
    finally:
        runner.tracer.uninstall()
    return runner, kernel


def _two_messages(runner, kernel, monkeypatch):
    """Kernel launches while a second message is read on top of a first, as
    MLX-LM reads them (all but the last token, then the last), and whether
    every output matches the untouched model."""
    patched = runner.model.model
    original = runner.baseline_model.model
    x = runner.tensors["prompt"][0].reshape(1, -1)
    ca, cb = patched.make_cache(), original.make_cache()
    launched = Counter()
    real = runtime_kernels.try_call

    def counting(spec, inputs):
        out = real(spec, inputs)
        launched[spec.kernel_id] += out is not None
        return out

    same = True
    for message in range(2):
        for part in (x[:, :-1], x[:, -1:]):
            if message == 1:
                monkeypatch.setattr(runtime_kernels, "try_call", counting)
            a = patched(part, cache=ca)
            monkeypatch.setattr(runtime_kernels, "try_call", real)
            same &= mx.array_equal(a, original(part, cache=cb)).item()
    return launched[kernel.kernel_id], same


def test_a_prompt_jobs_block_method_reads_a_second_message_with_its_kernel(tmp_path, monkeypatch):
    """A kernel found reading a prompt from an empty cache, in a block that
    receives the cache: the block's own method calls it again when a second
    message is read on top of the first (the recording declines there),
    bit for bit."""
    runner, kernel = _prompt_job(tmp_path, monkeypatch, BlockJudge(BLOCKS["llama_cache_model.py"]))
    rows = [r for r in runner.log.rows() if r.get("kind") == "live_method"]
    assert rows and all(r["installed"] for r in rows), [r.get("reason") for r in rows]
    launched, same = _two_messages(runner, kernel, monkeypatch)
    # two adds per layer, on both reads (the 5-token body and the last token, sizes the job recorded)
    assert same and launched == 2 * 2 * len(runner.live_scopes)


def test_a_prompt_jobs_recording_is_refused_as_a_method(tmp_path, monkeypatch):
    """The job's own recording offered as the method fits only the empty
    cache it was recorded on, so the check refuses it on the second message
    and the recording stays as it was."""
    runner, kernel = _prompt_job(tmp_path, monkeypatch, BlockJudge(None))
    rows = [r for r in runner.log.rows() if r.get("kind") == "live_method"]
    assert rows and not any(r["installed"] for r in rows)
    assert any("second messages" in (r["reason"] or "") for r in rows), [r["reason"] for r in rows]
    assert runner.live_scopes == set()
    _launched, same = _two_messages(runner, kernel, monkeypatch)
    assert same  # the job's own delivery stays (a graph re-traces a few new cache states, then stops)


def test_a_long_recording_is_sent_as_each_kernels_first_calls():
    """A whole-model recording (gpt-oss's was 119,802 lines) is sent as its
    __init__, each kernel's first calls with the lines around them, and its
    returns, so it fits in one request; a short one is sent whole."""
    from backcountry.harness.loop import REFERENCE_CHARS, REFERENCE_SITES, _reference_excerpt
    body = []
    for layer in range(400):
        body += [f"        v{layer}_{i} = mx.add(v{layer}_{i - 1}, v{layer}_{i - 1})" for i in range(40)]
        body.append(f"        _outs = _kernels.try_call(self._specs['kern_a'], [v{layer}_39])")
    source = "\n".join(["class R(ReplayWrapper):", "    def __init__(self, wrapped, specs=None):",
                        "        object.__setattr__(self, '_weights', (1, 2))",
                        "    def __call__(self, a0):"] + body + ["        return v399_39"])
    assert len(source) > REFERENCE_CHARS
    short = _reference_excerpt(source, ["kern_a"])
    assert len(short) < REFERENCE_CHARS / 4
    assert "_weights" in short and short.count("_specs['kern_a']") == REFERENCE_SITES
    assert "lines left out: the same pattern continues" in short and short.endswith("return v399_39")
    small = "class R(ReplayWrapper):\n    def __call__(self, a0):\n        return a0"
    assert _reference_excerpt(small, ["kern_a"]) == small


def test_the_judge_gets_the_code_of_the_parts_a_block_calls():
    """A kernel that spans two of a block's parts needs their code: the
    briefing carries the source of each class the block's code calls, and
    none of MLX's own layers."""
    from pathlib import Path
    from backcountry.harness.loop import _parts_source
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "llama_cache_model", Path(__file__).parent / "fixtures" / "llama_cache_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build()
    parts = _parts_source(model.model)
    assert {"TransformerBlock", "Attention", "MLP"} <= set(parts), sorted(parts)
    assert "def __call__" in parts["Attention"]
    assert not any(name in parts for name in ("QuantizedLinear", "RMSNorm", "RoPE", "LlamaModel"))


def test_a_kernel_whose_own_rule_declines_a_decode_call_still_gets_its_method(tmp_path, monkeypatch):
    """gpt-oss's attention-input kernel was found on both its prompt and its
    decode workload, but its own shape rule declines one-token inputs, so it
    never runs on a decode call: the check asks it to run where it takes the
    inputs (reading a message), and the block's method is installed. One
    that runs nowhere is still refused."""
    from dataclasses import replace
    from pathlib import Path
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.measure.session import Session
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {Path(__file__).parent / 'fixtures' / 'llama_cache_model.py'}
model_type: other
baseline: plain
use_library_inference: false
workloads:
  - name: prompt
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
  - name: reply
    context: 5
    inputs: [{{shape: [1, 1], dtype: int32, high: 64}}]
""")
    judge = BlockJudge(BLOCKS["llama_cache_model.py"])
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: judge, clock_pairs=4,
                       session=Session(sleep=lambda _: None))
    monkeypatch.setattr(runner, "_model_win", lambda result: all(c.passed for c in result.checks))
    try:
        runner.load_model()
        runner.trace_workloads()
        regions = runner.build_regions()
        region = next(r for r in regions if r.ops == ("array.__add__",) and not r.rejected)
        kernel = replace(runner._build_scaffold(region), fallback_predicate="in0.shape[1] == 1")
        runner._capture([region])
        for w in region.workloads:
            region.p[w] = 1.0
            region.t_orig_ms[w] = region.t_rep_ms[w] = 1.0
        runner.tracer.uninstall()
        assert {"prompt", "reply"} <= set(region.workloads)
        assert runner._bind_and_promote(RegionRun(region), kernel,
                                        LadderResult("tentative_ship", None, {}, .1, 1., .1, 0., [])), \
            runner.log.rows()[-1]
        runner._live_memory_scopes()
    finally:
        runner.tracer.uninstall()
    rows = [r for r in runner.log.rows() if r.get("kind") == "live_method"]
    assert rows and runner.live_scopes == {r["scope"] for r in rows}, [r.get("reason") for r in rows]


WINDOW_METHOD = '''class {name}(ReplayWrapper):
    def __call__(self, x, cache=None):
        out = _kernels.try_call(self._specs["ctx_mul"], [x, self.wk])
        keys = out[0] if out is not None else x * self.wk
        if cache is not None:
            keys = cache.update_and_fetch(keys)
        h = x + mx.sum(keys, axis=1, keepdims=True)
        if x.shape[1] > 8:
            h = mx.maximum(h, -1e9)
        out = _kernels.try_call(self._specs["ctx_mul"], [h, self.wo])
        return out[0] if out is not None else h * self.wo
'''


def test_the_per_size_step_widens_a_methods_kernels_when_the_block_branches_on_size(tmp_path, monkeypatch):
    """gpt-oss's whole model runs other operations past its sliding window,
    so no one replay follows its size and the per-size step refused it. A
    block whose own method calls its kernels needs no replay: each kernel's
    own operations recur at every recorded size, so its kernels are checked
    and run at sizes never recorded, past the window too."""
    from pathlib import Path
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.measure.session import Session
    model = Path(__file__).parent / "fixtures" / "window_context_model.py"
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: all(c.passed for c in result.checks))
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": True} for n in grid})
    monkeypatch.setattr(JobRunner, "_holdout", lambda self, *a: None)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {model}
model_type: other
baseline: plain
workloads:
  - name: decode
    inputs: [{{shape: [1, L], dtype: int32, high: 32}}]
    context: 4
primary: {{L: 2}}
range: {{L: [1, 48]}}
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: MethodJudge(WINDOW_METHOD),
                       clock_pairs=4, session=Session(sleep=lambda seconds: None))
    runner.load_model()
    try:
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if r.ops == ("array.__mul__",))
        runner._capture([region])
        assert runner._bind_and_promote(
            RegionRun(region=region), MUL_KERNEL,
            LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])), runner.log.rows()[-1]
        runner._live_memory_scopes()
    finally:
        runner.tracer.uninstall()
    assert runner.live_scopes == set(PATHS)

    def at(count):
        launched = Counter()
        real = runtime_kernels.try_call
        def counting(spec, ins):
            out = real(spec, ins)
            launched[out is not None] += 1
            return out
        monkeypatch.setattr(runtime_kernels, "try_call", counting)
        tokens = mx.array([[t % 32 for t in range(count)]])
        same = mx.array_equal(runner.model(tokens), runner.baseline_model(tokens)).item()
        monkeypatch.setattr(runtime_kernels, "try_call", real)
        return launched[True], same

    recorded = {int(label.rsplit("=", 1)[1]) for label in runner.traces if "=" in label} | {
        2, *runner.manifest.serve["L"].recordings(2)}
    past = next(n for n in range(9, 48) if n not in recorded)  # past the window, never recorded
    assert at(past) == (0, True)
    restore = runner._serve_sizes()
    assert restore
    assert not [r for r in runner.log.rows() if r.get("kind") == "serve_refused"]
    assert at(past) == (2 * len(PATHS), True)
    assert at(5) == (2 * len(PATHS), True)  # and under it


def _batch_tokens(model, prompts, steps=6):
    """Each prompt's generated tokens when MLX-LM's BatchGenerator serves
    them together (left padding, BatchKVCache), as mlx_lm.batch_generate and
    batching servers call the model."""
    from mlx_lm.generate import BatchGenerator
    gen = BatchGenerator(model, max_tokens=steps)
    uids = gen.insert(prompts, [steps] * len(prompts))
    out = {uid: [] for uid in uids}
    while responses := gen.next_generated():
        for r in responses:
            out[r.uid].append(r.token)
    gen.close()
    return [out[uid] for uid in uids]


def test_several_requests_at_once_get_the_original_outputs(tmp_path, monkeypatch):
    """Several requests at once (MLX-LM's BatchGenerator: a batch of prompts of
    different lengths, left-padded, then decoded together): kernels checked at
    one request decline the batched shapes, and every block's method, a
    rewritten copy of the model's own code, handles the batch cache and mask
    as the model does, so the tokens are the untouched model's."""
    runner, kernel = _prompt_job(tmp_path, monkeypatch, BlockJudge(BLOCKS["llama_cache_model.py"]))
    assert runner.live_scopes
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13]]
    assert _batch_tokens(runner.model.model, prompts) == _batch_tokens(runner.baseline_model.model, prompts)
