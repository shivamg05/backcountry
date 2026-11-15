"""Any function an application calls the model through comes back faster:
an engine's own generation function (here one shaped as LM Studio's
mlx-engine is: a kit, a worker thread, greedy passed as a function) is named
in the manifest, the loop search rewrites it, the bundle checks the shipped
answer the way the app calls it, and backcountry.faster() finds it."""
import textwrap
from pathlib import Path

import mlx.core as mx

from backcountry.harness import model_search
from backcountry.harness.loop import JobRunner
from backcountry.harness.measure.session import Session
from tests.test_identical_outputs import Scripted, _win

FIXTURES = Path(__file__).parent / "fixtures"
TARGET = "tests.fixtures.tiny_engine:create_generator"
SAME = f'''# target: {TARGET}
from tests.fixtures.tiny_engine import create_generator as _original


def create_generator(kit, prompt_tokens, **kwargs):
    """The engine's own, called through: the same tokens."""
    yield from _original(kit, prompt_tokens, **kwargs)
'''


def manifest(tmp_path, steps=1, model="engine_kit_model.py"):
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(f"""
        model: {FIXTURES / model}
        model_type: llm
        baseline: plain
        identical: output
        function: {TARGET}
        workloads:
          - name: prompt
            inputs: [{{shape: [1, 12], dtype: int32, low: 0, high: 64}}]
        budget: {{total: 0, model: 1}}
        final_benchmark: {{steps: {steps}, pairs: 4}}
    """))
    return p


def test_an_engines_own_function_is_searched_shipped_and_found(tmp_path, monkeypatch):
    import backcountry
    from backcountry_runtime import functions
    from tests.fixtures import tiny_engine
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    monkeypatch.setenv("BACKCOUNTRY_HOME", str(tmp_path / "home"))
    seen = {}

    def judge_for(region):
        judge = Scripted([{"hypothesis": "the engine's own", "kind": "same", "source": SAME.split("\n", 1)[1]},
                          {"verdict": "accept", "reason": "same work"}])
        ask = judge._ask

        def remember(system, messages):
            seen.setdefault("system", system)
            seen.setdefault("briefing", messages[0]["content"])
            return ask(system, messages)
        judge._ask = remember
        return judge
    runner = JobRunner(manifest(tmp_path), tmp_path / "work", session=Session(sleep=lambda s: None),
                       judge_factory=judge_for)
    report = runner.run()
    level = report.model_level
    assert level["final"]["ok"], level["history"]
    # the judge saw the engine's function and the loop it hides (mlx-lm's generate_step), and
    # its reply, missing the target line, got it back
    assert TARGET in seen["briefing"] and "def generate_step" in seen["briefing"]
    assert "create_generator" in seen["briefing"] and "generate_step(prompt, model, *, max_tokens" not in seen["system"]
    assert level["source"].startswith(f"# target: {TARGET}") and level["final"]["made_for"]
    artifact = runner.emit_artifact(tmp_path / "artifact")  # its fresh-process check calls the shipped answer
    assert f"TARGET = {TARGET!r}" in (artifact / "model_loop.py").read_text()

    from tests.fixtures.engine_kit_model import build
    model = build()
    kit = tiny_engine.Kit(model)
    prompt = list(range(3, 15))
    plain = list(tiny_engine.create_generator(kit, prompt))
    fast = backcountry.faster(tiny_engine.create_generator)
    ran = functions._ran
    assert list(fast(kit, prompt)) == plain and functions._ran == ran + 1  # the stored answer ran
    other = tiny_engine.Kit(build().update({}) or build())
    other.model.layers.pop()  # not the model it was made for: the original
    assert list(fast(other, prompt)) == list(tiny_engine.create_generator(other, prompt))
    assert functions._ran == ran + 1
    result = backcountry.compare(lambda m: list(fast(tiny_engine.Kit(m), prompt)), model, runs=2)
    assert result.loop and result.identical


def test_faster_installs_the_stored_kernels_once_into_the_model_it_finds(monkeypatch):
    """An engine that loads its own model (LM Studio's kit) gets the stored
    kernels through the function it calls: the first call on a model applies
    them in place, later calls do not."""
    import backcountry
    from backcountry_runtime import apply as apply_mod
    from backcountry_runtime import functions
    from tests.fixtures import tiny_engine
    from tests.fixtures.engine_kit_model import build
    applied = []
    monkeypatch.setattr(apply_mod, "apply", lambda model, *a, **k: applied.append(id(model)) or model)
    monkeypatch.setattr(functions, "_kernels_done", set())
    monkeypatch.setattr(functions, "stored_for", lambda target, model: None)
    kit = tiny_engine.Kit(build())
    fast = backcountry.faster(tiny_engine.create_generator)
    first, second = list(fast(kit, [3, 4, 5])), list(fast(kit, [3, 4, 5]))
    assert first == second and applied == [id(kit.model)]


def test_a_function_job_is_one_use_per_run(tmp_path):
    """final_benchmark.steps counts run() calls per use; with function, run()
    makes the whole reply itself (steps: 64 meant as tokens made the bundle
    check run 64 generations per case and time out)."""
    import pytest
    from backcountry.harness import manifest as manifest_mod
    assert manifest_mod.load(manifest(tmp_path)).final_benchmark.steps == 1
    with pytest.raises(manifest_mod.ManifestError, match="must be 1"):
        manifest_mod.load(manifest(tmp_path, steps=64))


def test_a_model_held_by_its_engine_is_released_before_the_bundle_check(tmp_path, monkeypatch):
    """An engine that keeps the model behind its own running thread (LM Studio's
    batched kit) does not free it when the job drops it, and the export refused
    to load a second copy; the model file's release() lets it go first."""
    import pytest
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    monkeypatch.setenv("BACKCOUNTRY_HOME", str(tmp_path / "home"))

    def job(path, model_file):
        path.mkdir()
        judge = Scripted([{"hypothesis": "the engine's own", "kind": "same", "source": SAME},
                          {"verdict": "accept", "reason": "same work"}])
        runner = JobRunner(manifest(path, model=model_file), path / "work",
                           session=Session(sleep=lambda s: None), judge_factory=lambda region: judge)
        assert runner.run().model_level["final"]["ok"]
        return runner
    released = job(tmp_path / "a", "engine_held_model.py")
    released.emit_artifact(tmp_path / "a" / "artifact")  # its fresh-process check passes
    assert not released.model_module.HELD
    held = job(tmp_path / "b", "engine_held_no_release.py")
    with pytest.raises(RuntimeError, match="still holds"):
        held.emit_artifact(tmp_path / "b" / "artifact")
