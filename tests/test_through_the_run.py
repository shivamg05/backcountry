"""Whatever the job searches (prefill, decode after a conversation, the
inference loop, kernels), it runs the model the way it is used and ships
what that use can run: MLX-LM's own generation when the model file has no
run(), the model file's run() when it has one. Here a job with both
workloads and the loop search, then kernels, on the tiny Llama: a stand-in
kernel goes into the layer that writes the KV cache (speed decisions forced;
correctness, install and the fresh-process check are real), and apply() puts
the artifact on a bare model that, run as before and through the shipped
loop, gives the untouched model's tokens."""

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import pytest

pytestmark = pytest.mark.integration  # whole jobs and fresh-process exports

FIX = Path(__file__).parent / "fixtures"
STEPS, CONTEXT = 4, 5


def _module(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), FIX / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fixture", ["llama_cache_model.py", "llama_engine.py"])  # MLX-LM's generation; an engine's run()
def test_prefill_decode_and_the_loop_run_and_ship_through_the_run(tmp_path, monkeypatch, fixture):
    import backcountry.harness.loop as loop_mod
    from backcountry.harness import model_search
    from backcountry.harness.ladder.gates import LadderResult
    from backcountry.harness.loop import JobRunner, RegionRun
    from backcountry.harness.measure.session import Session
    from backcountry_runtime.model_run import APP_RUN
    from backcountry import apply
    from tests.test_identical_outputs import SAME_STEP, Scripted, _win
    engine = fixture == "llama_engine.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {FIX / fixture}
model_type: llm
baseline: plain
{'' if engine else 'use_library_inference: true'}
identical: output
workloads:
  - name: prefill
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
  - name: decode
    context: {CONTEXT}
    inputs: [{{shape: [1, 1], dtype: int32, high: 64}}]
budget: {{per_region: 1, total: 1, model: 1}}
final_benchmark: {{steps: {STEPS}}}
""")
    # A stand-in is no speedup: only the speed decisions are controlled.
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: all(c.passed for c in result.checks))
    monkeypatch.setattr(JobRunner, "_holdout", lambda self, *a: None)
    monkeypatch.setattr(JobRunner, "_decode_check", lambda self: None)
    monkeypatch.setattr(JobRunner, "_second_message_check", lambda self: None)
    monkeypatch.setattr(loop_mod, "FINAL_MARGIN", float("inf"))
    monkeypatch.setattr(loop_mod, "apply_floor", lambda regions, **k: [r for r in regions if r.ops == ("array.__add__",)])

    def search(self):  # the kernel search: the add inside the layer that writes the KV cache
        ranked = self._search_state["ranked"]
        region = next(r for r in ranked + self.pending_regions if not r.rejected)
        kernel = self._build_scaffold(region)
        if region not in ranked:
            if not self.tracer.patcher.installed:
                self.tracer.install()
            self._capture([region])
        for w in region.workloads:
            region.p[w] = 1.0
            region.t_orig_ms[w] = region.t_rep_ms[w] = 1.0
        self.tracer.uninstall()
        promoted = self._bind_and_promote(RegionRun(region), kernel,
                                          LadderResult("tentative_ship", None, {}, .1, 1., .1, 0., []))
        assert promoted.status == "shipped", promoted
    monkeypatch.setattr(JobRunner, "_search", search)

    class Judge(Scripted):  # past the loop's replies, a judge that declines (a block's live method)
        def _ask(self, system, messages):
            if not self.replies:
                raise RuntimeError("no reply")
            return super()._ask(system, messages)
    judge = Judge([{"hypothesis": "the same loop", "kind": "same", "source": APP_RUN if engine else SAME_STEP},
                   {"verdict": "accept", "reason": "same work"}])
    runner = JobRunner(manifest, tmp_path / "work", session=Session(sleep=lambda _: None),
                       judge_factory=lambda region: judge)
    report = runner.run()
    assert runner._kernel_job.final_ok and report.model_level["final"]["ok"], report.model_level.get("final")
    artifact = runner.emit_artifact(tmp_path / "artifact")  # strict fresh-process check of both workloads and run.py

    model_file = _module(fixture)
    conversation = mx.random.randint(0, 64, (1, CONTEXT + 6), key=mx.random.key(8)).astype(mx.int32)

    def as_used(model):
        if engine:
            return [int(t) for t in model_file.run(model, conversation).tolist()]
        from mlx_lm.generate import generate_step
        return [int(t) for t, _ in generate_step(conversation.reshape(-1), model, max_tokens=STEPS)]

    def through_the_loop(model):
        spec = importlib.util.spec_from_file_location("shipped_loop", artifact / "model_loop.py")
        loop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loop)
        try:
            if engine:  # the engine's own run(), found faster, called as it is
                return [int(t) for t in loop.run(model, conversation).reshape(-1).tolist()]
            return [int(t) for t in loop.model_run(model, steps=STEPS)(conversation)["tokens"].reshape(-1).tolist()]
        finally:  # the bundle's loader registers the model file as autotune_model, for a user's process
            sys.modules.pop("autotune_model", None)

    want = as_used(model_file.build())
    patched = apply(model_file.build(), artifact)
    assert any(type(m).__module__ == "autotune_patch_wrappers" for _, m in patched.named_modules())
    assert as_used(patched) == want  # the user's own way of running it, unchanged
    assert through_the_loop(patched) == want  # the shipped loop, called with the same conversation
    if not engine:
        # MLX-LM's generation, as a caller (mlx_lm.server) calls it: a sampler, a
        # repetition penalty, a cache holding the conversation and a progress
        # callback; the bundle's generate_step against mlx_lm's own on the same model
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        spec = importlib.util.spec_from_file_location("shipped_loop", artifact / "model_loop.py")
        loop = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loop)
        sys.modules.pop("autotune_model", None)
        assert loop.generate_step.made_for(patched)

        def generated(step, model):
            cache, seen = make_prompt_cache(model), []
            model(conversation[:, :-1], cache=cache)
            mx.random.seed(3)
            out = list(step(conversation[0, -1:], model, max_tokens=STEPS, prompt_cache=cache,
                            sampler=make_sampler(temp=0.8, top_p=0.95),
                            logits_processors=make_logits_processors(repetition_penalty=1.1),
                            prompt_progress_callback=lambda *a: seen.append(a)))
            return [t for t, _ in out], mx.stack([lp for _, lp in out]), seen, cache[0].offset
        mine, theirs = generated(loop.generate_step, patched), generated(generate_step, patched)
        assert mine[0] == theirs[0] and mx.array_equal(mine[1], theirs[1]).item() and mine[2:] == theirs[2:]
