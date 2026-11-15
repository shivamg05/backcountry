"""identical: output: the job searches the inference loop (run(), or for
MLX-LM's generation its generate_step) for the same outputs, faster.
Correctness is the harness's: a wrong token fails, a gamed win is rolled back
by the reviewer, and only a held-out check ships run.py."""

import json
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

from backcountry.harness import manifest as manifest_mod
from backcountry.harness.judge import model_level
from backcountry.harness.measure.session import Session
from backcountry_runtime.generation import DEFAULT as GENERATION
from backcountry_runtime.model_run import GENERATE, load_loop, load_run
from backcountry_runtime.stats import comparison_from_samples

FIXTURES = Path(__file__).parent / "fixtures"

EXACT = '''import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def run(model, prompt, *, helpers):
    """Plain greedy decoding written out: the same ops as generate_step."""
    cache = make_prompt_cache(model)
    logits = model(prompt.reshape(1, -1), cache=cache)[:, -1, :]
    tokens = []
    for _ in range(helpers["steps"]):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        token = mx.argmax(logprobs, axis=-1)
        tokens.append(int(token.item()))
        logits = model(token.reshape(1, 1), cache=cache)[:, -1, :]
    return mx.array(tokens, dtype=mx.int32)
'''

WRONG = EXACT.replace("return mx.array(tokens, dtype=mx.int32)",
                      "tokens[2] = (tokens[2] + 1) % 64\n    return mx.array(tokens, dtype=mx.int32)")

# MLX-LM's generation: a found generate_step. SAME_STEP calls MLX-LM's own
# through; the others change one thing a caller would notice.
SAME_STEP = '''from mlx_lm.generate import generate_step as _mlx_lm_generate_step


def generate_step(prompt, model, **kwargs):
    """MLX-LM's generate_step, called through."""
    yield from _mlx_lm_generate_step(prompt, model, **kwargs)
'''
WRONG_STEP = SAME_STEP.replace("    yield from _mlx_lm_generate_step(prompt, model, **kwargs)",
                               "    for i, (token, logprobs) in enumerate(_mlx_lm_generate_step(prompt, model, **kwargs)):\n"
                               "        yield ((token + 1) % 64 if i == 2 else token), logprobs")


def _step_dropping(what: str) -> str:
    """A generate_step that ignores one of the caller's arguments."""
    return SAME_STEP.replace("(prompt, model, **kwargs):", f"(prompt, model, {what}=None, **kwargs):")


def write_manifest(tmp_path, extra="budget: {total: 0, model: 3}"):
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(f"""
        model: {FIXTURES / "llama_cache_model.py"}
        model_type: llm
        use_library_inference: true
        identical: output
        workloads:
          - name: reply
            inputs: [{{shape: [1, T], dtype: int32, low: 0, high: 64}}]
        primary: {{T: 12}}
        final_benchmark: {{steps: 8, pairs: 4}}
        {extra}
    """))
    return p


def test_identical_defaults_to_computation_and_the_loop_search_owns_its_budget(tmp_path):
    m = manifest_mod.load(write_manifest(tmp_path))
    assert m.identical == "output" and m.budget_model == 3 and m.budget_total == 0
    assert manifest_mod.load(write_manifest(tmp_path, "budget: {total: 5}")).budget_model == \
        manifest_mod.DEFAULT_BUDGET_MODEL
    plain = write_manifest(tmp_path, "").read_text().replace("identical: output", "")
    (tmp_path / "plain.yaml").write_text(plain)
    assert manifest_mod.load(tmp_path / "plain.yaml").identical == "computation"
    (tmp_path / "bad.yaml").write_text(plain + "budget: {model: 3}\n")
    with pytest.raises(manifest_mod.ManifestError, match="identical: output"):
        manifest_mod.load(tmp_path / "bad.yaml")
    (tmp_path / "zero.yaml").write_text(plain + "budget: {total: 0}\n")
    with pytest.raises(manifest_mod.ManifestError):
        manifest_mod.load(tmp_path / "zero.yaml")  # no kernel search and no loop search: nothing to do


def test_a_drafter_loads_past_weights_it_never_reads_but_not_without_its_own(monkeypatch):
    """Gemma 4 E2B ships key/value weights for its 20 KV-sharing layers,
    which the model never builds, and mlx_lm refused the whole checkpoint."""
    import mlx.nn as nn
    import mlx_lm
    from backcountry_runtime.model_run import load_drafter

    def fake_load(path):
        model = nn.Linear(4, 4)
        model.load_weights(list(checkpoint.items()))
        return model, None
    monkeypatch.setattr(mlx_lm, "load", fake_load)
    checkpoint = {"weight": mx.ones((4, 4)), "bias": mx.zeros((4,)), "unused.k_proj.weight": mx.ones((2, 2))}
    assert mx.array_equal(load_drafter("x").weight, mx.ones((4, 4))).item()
    assert nn.Module.load_weights.__name__ == "load_weights" and "own" not in nn.Module.load_weights.__code__.co_varnames
    del checkpoint["bias"]
    with pytest.raises(ValueError, match="Missing"):
        load_drafter("x")


def test_a_head_built_for_another_models_width_is_not_offered(tmp_path):
    # Qwen3.8-27B's MTP head (hidden 5120) sat in the cache of a host running
    # Qwen3.5-9B (hidden 4096): same tokenizer, smaller, so it was listed, and
    # the AI spent 10 of 20 attempts trying to fit it
    import types
    from backcountry.harness import model_search
    tokens = {"model": {"vocab": {"a": 0, "b": 1}, "merges": []}, "added_tokens": []}

    def snapshot(repo, size, **config):
        folder = tmp_path / "hub" / f"models--{repo}" / "snapshots" / "rev"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"vocab_size": 8, **config}))
        (folder / "tokenizer.json").write_text(json.dumps(tokens))
        (folder / "model.safetensors").write_bytes(b"0" * size)
        return folder

    big = snapshot("org--qwen9b", 3000, model_type="qwen3_5", hidden_size=4096)
    snapshot("org--qwen27b-mtp", 200, model_type="qwen3_5_mtp", hidden_size=5120)
    snapshot("org--qwen9b-mtp", 150, model_type="qwen3_5_mtp", hidden_size=4096)
    snapshot("org--qwen-0b6", 600, model_type="qwen3", hidden_size=1024)   # a small model runs on its own
    assets, note = model_search.find_assets(types.SimpleNamespace(name_or_path=str(big)))
    assert sorted(a["repo"] for a in assets) == ["org/qwen-0b6", "org/qwen9b-mtp"]
    assert "skipped as built for another model's width" in note and "org/qwen27b-mtp" in note


def test_an_asset_needs_the_same_tokens_not_the_same_file_or_family(tmp_path):
    """Bonsai 2 and Qwen3.5 split text identically, but Bonsai's tokenizer
    file writes merges as "a b", Qwen's as ["a", "b"], Qwen adds audio
    markers, and their model types differ: the byte-for-byte rule found no
    drafter. Assets are listed with their type, never loaded (an MTP head
    is not an mlx-lm model, and loading one crashed the job)."""
    import types
    from backcountry.harness import model_search

    def snapshot(repo, vocab, merges, added, size, vocab_size=8, config=()):
        folder = tmp_path / "hub" / f"models--{repo}" / "snapshots" / "rev"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"model_type": repo, "vocab_size": vocab_size, **dict(config)}))
        if vocab is not None:
            (folder / "tokenizer.json").write_text(json.dumps(
                {"model": {"vocab": vocab, "merges": merges}, "added_tokens": added}))
        (folder / "model.safetensors").write_bytes(b"0" * size)
        return folder

    vocab = {"a": 0, "b": 1, "ab": 2}
    big = snapshot("prism--bonsai", vocab, ["a b"], [], 3000)
    snapshot("qwen--small", dict(vocab), [["a", "b"]], [{"content": "<|audio_pad|>", "id": 7}], 600)
    snapshot("other--tiny", {"a": 0, "c": 1}, [], [], 100)
    assets, note = model_search.find_assets(types.SimpleNamespace(name_or_path=str(big)))
    assert [(a["repo"], a["model_type"], a["revision"]) for a in assets] == [("qwen/small", "qwen--small", "rev")]
    assert "none is loaded" in note
    # a drafter that ships no tokenizer (Liquid's DSpark) is listed when the
    # manifest names it, as the drafter mlx-vlm loads it as, with that code
    snapshot("liquid--spark", None, [], [], 500, config={
        "model_type": "qwen3", "architectures": ["Lfm2DSparkDraftModel"], "markov_rank": 256,
        "dflash_config": {"mask_token_id": 7, "target_layer_ids": [1], "num_target_layers": 2}})
    snapshot("liquid--unnamed", None, [], [], 400)
    assets, _ = model_search.find_assets(types.SimpleNamespace(name_or_path=str(big)), named=["liquid/spark"])
    spark = next(a for a in assets if a["repo"] == "liquid/spark")
    assert [a["repo"] for a in assets] == ["liquid/spark", "qwen/small"]
    assert spark["drafter"] == "mlx_vlm.speculative.drafters.dspark" and spark["architecture"] == "Lfm2DSparkDraftModel"
    source = model_search._drafter_source(spark)
    assert "class DSparkDraftModel" in source and "mlx_vlm.speculative.dflash: the round loop" in source
    # a bundle finds each pinned asset in the folder the job found it in: a
    # snapshot without the files the checkpoint never reads (.gitattributes)
    # serves, where Hugging Face's offline download refuses it as incomplete
    from huggingface_hub import constants

    from backcountry_runtime.model_run import asset_path
    original, constants.HF_HUB_CACHE = constants.HF_HUB_CACHE, str(tmp_path / "hub")
    try:
        assert asset_path("qwen/small", "rev") == str(tmp_path / "hub" / "models--qwen--small" / "snapshots" / "rev")
        # one not on this Mac downloads at the pinned revision, as the model does
        import huggingface_hub
        fetched = []
        real = huggingface_hub.snapshot_download
        huggingface_hub.snapshot_download = lambda repo, revision: fetched.append((repo, revision)) or "/got"
        try:
            assert asset_path("qwen/small", "other-rev") == "/got" and fetched == [("qwen/small", "other-rev")]
            huggingface_hub.snapshot_download = lambda repo, revision: 1 / 0
            with pytest.raises(FileNotFoundError, match="did not download"):
                asset_path("qwen/small", "other-rev")
        finally:
            huggingface_hub.snapshot_download = real
    finally:
        constants.HF_HUB_CACHE = original
    # the manifest's assets download before the search; one that does not is
    # reported and the search goes on without it
    calls = []

    def download(repo):
        calls.append(repo)
        if repo.endswith("gone"):
            raise OSError("no answer")
    import huggingface_hub
    real = huggingface_hub.snapshot_download
    huggingface_hub.snapshot_download = download
    try:
        assert model_search.fetch_assets(["a/head", "a/gone"]) == ["a/gone"] and calls == ["a/head", "a/gone"]
    finally:
        huggingface_hub.snapshot_download = real


def test_no_assets_needs_no_huggingface_hub(monkeypatch):
    # an app whose tokenizer is its own may not have huggingface_hub: the
    # loop search crashed on the import with nothing to download
    from backcountry.harness import model_search
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert model_search.fetch_assets([]) == []


def test_a_candidate_must_define_run():
    with pytest.raises(ValueError, match="define run"):
        load_run("x = 1")
    assert callable(load_run(GENERATE))
    assert callable(load_loop(GENERATION)) and callable(load_loop(SAME_STEP))
    with pytest.raises(ValueError, match="generate_step"):
        load_loop("def generate_step(prompt, model):\n    pass\n\n\ngenerate_step = 1")


class Scripted:
    """The judge transport: replies in order, recording what each call saw."""
    transcript = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def _ask(self, system, messages):
        self.seen.append((system, json.loads(messages[0]["content"])))
        return json.dumps(self.replies.pop(0))


def test_a_malformed_proposal_is_asked_again_once():
    judge = Scripted([{"hypothesis": "h"}, {"hypothesis": "h", "kind": "k", "source": EXACT}])
    assert model_level.propose(judge, {"job": {}})["source"] == EXACT
    judge = Scripted([{"hypothesis": "h"}, {"kind": "k"}])
    with pytest.raises(ValueError, match="refused twice"):
        model_level.propose(judge, {"job": {}})
    # for MLX-LM's generation the proposal is a generate_step, not a run()
    judge = Scripted([{"hypothesis": "h", "kind": "k", "source": EXACT}] * 2)
    with pytest.raises(ValueError, match="generate_step"):
        model_level.propose(judge, {"job": {}}, generation=True)
    assert "generate_step(prompt, model" in judge.seen[0][0]
    with pytest.raises(ValueError, match="accept or reject"):
        model_level.review(Scripted([{"verdict": "maybe"}]), EXACT, GENERATE, {})


def test_every_timed_call_gets_inputs_no_earlier_call_saw():
    """Both arms see the same inputs, each once, so remembering an input
    cannot pay."""
    from backcountry.harness.measure.clocks import compare_on_inputs
    seen = {"a": [], "b": []}
    sets = [[mx.array([i])] for i in range(8)]
    session = Session(sleep=lambda s: None)
    session.warm_until_stable = lambda fn: None
    compare_on_inputs(session, lambda x: seen["a"].append(int(x.item())),
                      lambda x: seen["b"].append(int(x.item())), sets)
    assert sorted(seen["a"]) == sorted(seen["b"]) == list(range(8))


def test_timing_never_holds_two_prepared_inputs_at_once():
    """A conversation read for one arm is let go before the other arm's is
    read: two held at once beside two drafters ran a 27B model out of memory."""
    import weakref
    from backcountry.harness.measure.clocks import compare_on_inputs

    class Read:  # stands in for a conversation read into memory
        def __init__(self, x):
            self.x = x
    alive, most = weakref.WeakSet(), [0]

    def prepare(arm, inputs):
        import gc
        gc.collect()
        most[0] = max(most[0], len(alive) + 1)
        held = Read(inputs[0])
        alive.add(held)
        return [held]
    session = Session(sleep=lambda s: None)
    session.warm_until_stable = lambda fn: fn()
    compare_on_inputs(session, lambda r: None, lambda r: None,
                      [[mx.array([i])] for i in range(4)], prepare=prepare)
    assert most[0] == 1


def _win():
    return comparison_from_samples([100.0, 100.0, 100.0, 100.0], [50.0, 51.0, 50.0, 49.0])


def test_output_mode_checks_reviews_and_ships_only_a_held_out_win(tmp_path, monkeypatch):
    """A wrong token fails; a win the reviewer rejects is rolled back; an
    accepted win stacks, passes the held-out check, and becomes run.py."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    judge = Scripted([
        {"hypothesis": "wrong on purpose", "kind": "wrong", "source": WRONG_STEP},
        {"hypothesis": "mlx-lm's, called through", "kind": "rewrite", "source": SAME_STEP},
        {"verdict": "reject", "reason": "saves results across calls"},
        {"hypothesis": "mlx-lm's, called through", "kind": "rewrite", "source": SAME_STEP},
        {"verdict": "accept", "reason": "same work every call"},
    ])
    runner = JobRunner(write_manifest(tmp_path), tmp_path / "work", judge_factory=lambda region: judge,
                       session=Session(sleep=lambda s: None))
    report = runner.run()
    result = report.model_level
    verdicts = [h["verdict"] for h in result["history"]]
    assert verdicts == ["failed", "rejected_by_review", "accepted"]
    assert result["history"][0]["detail"]["position"] == 2
    assert result["accepted"] == [2] and result["final"]["ok"]
    assert (tmp_path / "work" / "run.py").read_text() == SAME_STEP
    saved = json.loads((tmp_path / "work" / "report.json").read_text())["model_level"]
    assert saved["final"]["ok"] and [h["verdict"] for h in saved["history"]] == verdicts
    # the judge saw the rejection, the current best and the exact-rows probe
    briefing = judge.seen[-2][1]
    assert briefing["history"][1]["detail"] == "saves results across calls"
    assert briefing["current_best"] == GENERATION and briefing["exact_rows"]["widths"]
    assert "def generate_step(" in briefing["default_run"]  # MLX-LM's own source, to rewrite
    assert all(w["exact"] for w in result["probe"]["widths"])
    assert result["final"]["cases"] and all(c["ok"] for c in result["final"]["cases"])


def test_a_loop_that_is_not_kept_leaves_the_model_weights_it_found(tmp_path):
    """A setup() may replace the model's weights; Gemma 4's attempt 9 re-laid
    out its experts, ran out of memory, and the copies it left ran the final
    out of memory too. A loop that is not kept leaves the arrays it found."""
    from backcountry.harness.loop import JobRunner
    relayout = SAME_STEP + '''

def setup(model, helpers):
    layer = model.model.layers[0].mlp.up_proj
    object.__setattr__(model, "found", layer.weight)  # not a parameter: the test's record
    layer.weight = layer.weight + 0  # a copy in the model's place
    raise RuntimeError("[METAL] Insufficient Memory")
'''
    judge = Scripted([{"hypothesis": "merge the weights", "kind": "relayout", "source": relayout}])
    runner = JobRunner(write_manifest(tmp_path, "budget: {total: 0, model: 1}"), tmp_path / "work",
                       judge_factory=lambda region: judge, session=Session(sleep=lambda s: None))
    report = runner.run()
    assert [h["verdict"] for h in report.model_level["history"]] == ["failed"]
    model = getattr(runner.model, "model", runner.model)
    assert model.model.layers[0].mlp.up_proj.weight is model.found


def test_a_loop_search_stopped_mid_way_resumes_at_its_next_attempt(tmp_path, monkeypatch):
    """Each attempt takes minutes to tens of minutes; a job that dies (the Mac
    lost power during Qwen3.8-27B's attempt 5) resumes from the attempts it
    finished, the best loop loaded again, instead of searching from the start."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    stepped = SAME_STEP.replace('"""MLX-LM\'s generate_step, called through."""', '"""Called through, the best."""')

    class Dies(Scripted):
        def _ask(self, system, messages):
            if not self.replies:
                raise KeyboardInterrupt  # the job stops before its second attempt finishes
            return super()._ask(system, messages)
    first = Dies([{"hypothesis": "one", "kind": "same", "source": stepped},
                  {"verdict": "accept", "reason": "same work"}])
    manifest = write_manifest(tmp_path, "budget: {total: 0, model: 2}")
    work = tmp_path / "work"
    with pytest.raises(KeyboardInterrupt):
        JobRunner(manifest, work, judge_factory=lambda region: first, session=Session(sleep=lambda s: None)).run()
    saved = json.loads((work / "loop_search.json").read_text())
    assert saved["accepted"] == [0] and saved["best_source"] == stepped and len(saved["history"]) == 1
    (work / "job.json").write_text("{}")  # what `backcountry resume` reads first; the runner needs only the position
    second = Scripted([{"hypothesis": "two", "kind": "same", "source": SAME_STEP},
                       {"verdict": "accept", "reason": "same work"}])
    report = JobRunner(manifest, work, judge_factory=lambda region: second, resume=True,
                       session=Session(sleep=lambda s: None)).run()
    level = report.model_level
    assert [h["attempt"] for h in level["history"]] == [0, 1] and level["accepted"] == [0, 1]
    assert second.seen[0][1]["current_best"] == stepped  # attempt 1 was proposed against attempt 0's loop
    assert level["final"]["ok"] and (work / "run.py").read_text() == SAME_STEP


def test_autotune_pause_stops_a_loop_search_after_its_attempt_and_a_resume_continues(tmp_path, monkeypatch):
    """`backcountry pause` during the loop search stops it after the attempt it is
    on; the request is used up there, not left to pause the kernel search later."""
    from backcountry.harness import model_search, resume
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    manifest = write_manifest(tmp_path, "budget: {total: 0, model: 2}")
    work = tmp_path / "work"

    class Paused(Scripted):
        def _ask(self, system, messages):
            if len(self.replies) == 2:  # the operator pauses while attempt 0 is proposed
                (work / resume.PAUSE_REQUEST).write_text("operator requested a pause\n")
            return super()._ask(system, messages)
    first = Paused([{"hypothesis": "one", "kind": "same", "source": SAME_STEP},
                    {"verdict": "accept", "reason": "same work"}])
    report = JobRunner(manifest, work, judge_factory=lambda region: first, session=Session(sleep=lambda s: None)).run()
    assert report.session["status"] == "paused" and "loop search, before attempt 1" in report.session["reason"]
    assert len(json.loads((work / "loop_search.json").read_text())["history"]) == 1
    assert not (work / resume.PAUSE_REQUEST).exists()
    second = Scripted([{"hypothesis": "two", "kind": "same", "source": SAME_STEP},
                       {"verdict": "accept", "reason": "same work"}])
    level = JobRunner(manifest, work, judge_factory=lambda region: second, resume=True,
                      session=Session(sleep=lambda s: None)).run().model_level
    assert [h["attempt"] for h in level["history"]] == [0, 1] and level["final"]["ok"]


def test_a_loop_only_job_stopped_after_its_final_resumes_at_the_export(tmp_path, monkeypatch):
    """A loop-only job whose search and final passed, stopped in its export
    (Qwen3.8-27B's loop, left out by a faulty bundle check), resumes at the
    export with the loop checked again there: no attempt is searched again."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    first = Scripted([{"hypothesis": "one", "kind": "same", "source": SAME_STEP},
                      {"verdict": "accept", "reason": "same work"}])
    manifest = write_manifest(tmp_path, "budget: {total: 0, model: 1}")
    work = tmp_path / "work"
    report = JobRunner(manifest, work, judge_factory=lambda region: first, session=Session(sleep=lambda s: None)).run()
    report.model_level["not_shipped"] = "the artifact's run.py did not reproduce the job's outputs: x"
    report.write(work / "report.json")
    (work / "job.json").write_text("{}")
    unasked = Scripted([])  # asked anything, it has no reply
    runner = JobRunner(manifest, work, judge_factory=lambda region: unasked, resume=True,
                       session=Session(sleep=lambda s: None))
    level = runner.run().model_level
    assert level["final"]["ok"] and "not_shipped" not in level and not unasked.seen
    artifact = runner.emit_artifact(tmp_path / "artifact")
    assert (artifact / "run.py").read_text() == SAME_STEP


def test_the_loop_then_kernels_ship_together(tmp_path, monkeypatch):
    """identical: output with a kernel budget: the loop search runs first, the
    kernel search then runs as an ordinary job on kernel_manifest.yaml (this
    workload, since the loop calls the model as before), the loop is timed on
    the untouched and the patched model, and the artifact carries both; its
    fresh-process check applies the kernels, then runs run.py."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    from tests.test_loop import only_the_chain, winning_chain_judge
    only_the_chain(monkeypatch)
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {FIXTURES / "planted_win.py"}
        model_type: other
        identical: output
        baseline: plain
        workloads:
          - inputs: [{{shape: [64, 1024], dtype: float32}}]
            name: main
        budget: {{per_region: 4, total: 8, model: 1}}
    """))
    loop = '''def run(model, *inputs, helpers):
    return model(*inputs)
'''
    loop_judge = Scripted([{"hypothesis": "the same call", "kind": "same", "source": loop},
                           {"verdict": "accept", "reason": "same work"}])
    asked = []

    def factory(region):
        asked.append("loop" if region is None else "kernel")
        return loop_judge if region is None else winning_chain_judge(region)

    runner = JobRunner(manifest, tmp_path / "work", session=Session(sleep=lambda s: None), judge_factory=factory)
    report = runner.run()
    assert asked[0] == "loop" and "kernel" in asked  # the loop first, then kernels on its calls
    kernel_manifest = manifest_mod.load(tmp_path / "work" / "kernel_manifest.yaml")
    assert kernel_manifest.identical == "computation" and kernel_manifest.budget_model == 0
    assert [w.inputs[0].shape for w in kernel_manifest.workloads] == [(64, 1024)]  # unchanged calls
    assert (tmp_path / "work" / "kernels" / "run.jsonl").is_file()
    assert report.accepted and runner.final_ok
    assert report.model_level["final"]["ok"] and report.model_level["loop_alone"]["ok"]
    assert report.model_level["final"]["kernels_under_loop"]["n"]  # the loop timed on both models
    artifact = runner.emit_artifact(tmp_path / "artifact")
    assert (artifact / "run.py").read_text() == loop and (artifact / "model_loop.py").is_file()
    assert "The kernels under this loop" in (artifact / "README.md").read_text()


def test_unconfirmed_kernels_leave_the_loop_to_ship_alone_and_the_job_resumes_after_its_loop(tmp_path, monkeypatch):
    """The kernel search's own final confirms nothing: the loop is not timed
    with kernels nobody confirmed, and the bundle ships run.py on the
    untouched model with the loop's own final (on Qwen3.8-27B the loop's
    1.40x was discarded with the kernels). A job stopped after its loop
    search resumes from that search's result: the loop is not searched again."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    from tests.test_loop import only_the_chain, winning_chain_judge
    only_the_chain(monkeypatch)
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    real_final = JobRunner._final_check

    def unconfirmed(self):
        real_final(self)
        self.final_ok = False
    monkeypatch.setattr(JobRunner, "_final_check", unconfirmed)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {FIXTURES / "planted_win.py"}
        model_type: other
        identical: output
        baseline: plain
        workloads:
          - inputs: [{{shape: [64, 1024], dtype: float32}}]
            name: main
        budget: {{per_region: 4, total: 8, model: 1}}
    """))
    loop = '''def run(model, *inputs, helpers):
    return model(*inputs)
'''
    asked = []

    def factory(region):
        asked.append("loop" if region is None else "kernel")
        return (Scripted([{"hypothesis": "the same call", "kind": "same", "source": loop},
                          {"verdict": "accept", "reason": "same work"}])
                if region is None else winning_chain_judge(region))

    work = tmp_path / "work"
    report = JobRunner(manifest, work, session=Session(sleep=lambda s: None), judge_factory=factory).run()
    level = report.model_level
    assert level["final"] == level["loop_alone"] and level["final"]["ok"]
    assert level["kernels_dropped"] == "no kernel was confirmed" and level["final_with_kernels"] is None
    asked.clear()
    # a resume stopped before it set the loop's result again (Qwen3.8-27B's, killed while
    # loading the model) leaves it in the report for the next one
    JobRunner(manifest, work, session=Session(sleep=lambda s: None), judge_factory=factory,
              resume=True).report.write(work / "report.json")
    runner = JobRunner(manifest, work, session=Session(sleep=lambda s: None), judge_factory=factory, resume=True)
    report = runner.run()
    assert "loop" not in asked and report.model_level["final"]["ok"]
    artifact = runner.emit_artifact(tmp_path / "artifact")  # run.py checked in a fresh process
    assert (artifact / "run.py").read_text() == loop
    assert "The kernels under this loop" not in (artifact / "README.md").read_text()
    # a kernel search that died before its first save (Qwen3.8-27B ran out of GPU
    # memory in its setup) starts over on resume; its logs are kept beside
    (work / "kernels" / "resume_state.pkl").unlink()
    asked.clear()
    report = JobRunner(manifest, work, session=Session(sleep=lambda s: None), judge_factory=factory,
                       resume=True).run()
    assert asked and "loop" not in asked and report.model_level["final"]["ok"]
    assert [p.name.startswith("kernels-unsaved-") for p in work.glob("kernels-*")] == [True]


def test_a_finish_asked_before_the_kernel_search_ships_the_loop_alone(tmp_path, monkeypatch):
    """backcountry finish asked by the time the loop search ends skips the kernel
    search: the loop's confirmed result ships alone (Qwen3.8-27B's kernel
    search ran out of GPU memory in its setup at every resume)."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {FIXTURES / "planted_win.py"}
        model_type: other
        identical: output
        baseline: plain
        workloads:
          - inputs: [{{shape: [64, 1024], dtype: float32}}]
            name: main
        budget: {{per_region: 4, total: 8, model: 1}}
    """))
    loop = '''def run(model, *inputs, helpers):
    return model(*inputs)
'''
    work, asked = tmp_path / "work", []

    real_final = model_search.ModelSearch.final

    def final_then_finish(self, *a, **k):  # the operator asks while the loop's final runs
        (work / "finish-search.request").write_text("operator requested final validation\n")
        return real_final(self, *a, **k)
    monkeypatch.setattr(model_search.ModelSearch, "final", final_then_finish)

    def factory(region):
        asked.append("loop" if region is None else "kernel")
        return Scripted([{"hypothesis": "the same call", "kind": "same", "source": loop},
                         {"verdict": "accept", "reason": "same work"}])

    runner = JobRunner(manifest, work, session=Session(sleep=lambda s: None), judge_factory=factory)
    level = runner.run().model_level
    assert asked == ["loop"] and level["final"]["ok"] and "finish" in level["kernels_dropped"]
    assert not (work / "kernels").exists()
    artifact = runner.emit_artifact(tmp_path / "artifact")  # run.py checked in a fresh process
    assert (artifact / "run.py").read_text() == loop


def test_a_job_whose_loop_found_nothing_resumes_its_kernel_search(tmp_path, monkeypatch):
    """Most loop searches find nothing faster (gpt-oss-20b, LFM2.5); a job
    paused in the kernel search after one resumes that search where it
    stopped, instead of starting the kernel job again in its own folder."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    from tests.test_loop import ScriptedJudge, only_the_chain, winning_chain_judge
    only_the_chain(monkeypatch)
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(model_search, "compare_on_inputs",  # the loop is slower: not kept
                        lambda *a, **k: comparison_from_samples([50.0] * 4, [100.0, 101.0, 100.0, 99.0]))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {FIXTURES / "planted_win.py"}
        model_type: other
        identical: output
        baseline: plain
        workloads:
          - inputs: [{{shape: [64, 1024], dtype: float32}}]
            name: main
        budget: {{per_region: 4, total: 8, model: 1}}
    """))
    loop = '''def run(model, *inputs, helpers):
    return model(*inputs)
'''
    work, asked = tmp_path / "work", []

    def pausing(region):
        asked.append("loop" if region is None else "kernel")
        if region is None:
            return Scripted([{"hypothesis": "the same call", "kind": "same", "source": loop}])
        judge = winning_chain_judge(region)
        propose = judge.next

        def next_then_pause(meta, verdict):
            reply = propose(meta, verdict)
            if reply.kernel is not None:
                (work / "pause.request").write_text("test\n")  # stops after this attempt
            return reply
        judge.next = next_then_pause
        return judge

    paused = JobRunner(manifest, work, session=Session(sleep=lambda s: None), judge_factory=pausing).run()
    assert paused.session["status"] == "paused" and not paused.model_level.get("accepted")
    from backcountry.harness import resume as resume_mod
    spent = resume_mod.load(work / "kernels")["report"].session["job_active_s"]  # the loop's time and the kernels'
    resumed = JobRunner(manifest, work, session=Session(sleep=lambda s: None), resume=True,
                        judge_factory=lambda region: ScriptedJudge([])).run()
    assert resumed.session["status"] == "search_complete" and len(resumed.accepted) == 1
    assert len(resumed.model_level["history"]) == 1  # the loop search made no new attempt
    assert resumed.session["job_active_s"] > spent  # the job's clock goes on from the time already spent


def test_a_decode_workload_is_translated_to_the_calls_the_loop_makes(tmp_path):
    """When the winning loop spends the model's time on calls that read several
    generated tokens, the decode workload becomes those calls (generated: true,
    the most-timed count as the primary, the others listed or checked); a
    helper model's calls never count; the result is an ordinary kernel
    manifest. Without a loop, or with one calling the model a token at a time,
    the workload is unchanged."""
    from backcountry.harness.model_search import kernel_manifest
    src = tmp_path / "decode.yaml"
    src.write_text(textwrap.dedent("""
        model: ../m.py
        model_type: llm
        identical: output
        workloads:
          - name: decode_after_512
            context: 512
            inputs: [{shape: [1, 1], dtype: int32, low: 0, high: 100}]
        budget: {per_region: 25, total: 40, model: 6}
        final_benchmark: {steps: 16}
    """))
    model = FIXTURES / "context_model.py"

    def row(call, rows, count, ms, at=512):
        return {"call": call, "inputs": f"[1, {rows}] int32", "rows": rows, "count": count, "ms_each": ms,
                "offsets": [at, at + 16]}
    calls = [row("model", 3, 14, 47.0), row("head", 1, 23, 5.5), row("model", 1, 1, 45.0), row("model", 2, 2, 46.0),
             row("model", 4, 1, 48.0)]
    out = tmp_path / "kernel_manifest.yaml"
    text, changed = kernel_manifest(src, model, {"source": "...", "profile": {"calls": calls, "parts": ["head"]}})
    out.write_text(text)
    m = manifest_mod.load(out)
    assert changed and m.identical == "computation" and m.budget_model == 0 and m.budget_total == 40
    w = m.workloads[0]
    assert w.context == 512 and w.generated and w.inputs[0].shape == (1, "new_tokens")
    assert m.primary == {"new_tokens": 3} and m.serve["new_tokens"].listed == (2, 4)
    assert m.final_benchmark.steps == 16 and m.model_path == model.resolve()
    # one other width (a 1-guess loop with 2-token re-reads): checked there, no range
    text, changed = kernel_manifest(src, model, {"source": "...", "profile": {
        "calls": [row("model", 2, 60, 46.0), row("model", 3, 4, 47.0), row("model", 1, 8, 45.0)]}})
    out.write_text(text)
    m = manifest_mod.load(out)
    assert changed and m.primary == {"new_tokens": 2} and m.sweep["new_tokens"] == (2, 3) and not m.serve
    one = [row("model", 1, 16, 45.0)]
    prompt = [row("model", 6, 1, 900.0, at=0)]  # reading the conversation: not a generated call
    for level in ({}, {"source": "...", "profile": {"calls": one}}, {"source": "...", "profile": {"calls": prompt}}):
        text, changed = kernel_manifest(src, model, level)
        out.write_text(text)
        assert not changed and manifest_mod.load(out).workloads[0].inputs[0].shape == (1, 1)


def test_a_prompt_workload_beside_the_conversation_stays_as_written(tmp_path):
    """A job with a prompt read (no history) beside the reply after history:
    the hand-off turns only the reply into the loop's check calls and keeps
    the prompt workload and its range of lengths as written, so one kernel
    search covers both."""
    from backcountry.harness.model_search import kernel_manifest
    src = tmp_path / "chat.yaml"
    src.write_text(textwrap.dedent("""
        model: ../m.py
        model_type: llm
        identical: output
        workloads:
          - name: prompt
            inputs: [{shape: [1, T], dtype: int32, low: 0, high: 100}]
          - name: reply
            context: 512
            inputs: [{shape: [1, 1], dtype: int32, low: 0, high: 100}]
        primary: {T: 64}
        range: {T: [32, 256]}
        budget: {per_region: 25, total: 40, model: 6}
        final_benchmark: {steps: 16}
    """))
    calls = [{"call": "model", "inputs": f"[1, {r}] int32", "rows": r, "count": n, "ms_each": 46.0,
              "offsets": [512, 600]} for r, n in ((2, 30), (3, 4), (4, 3), (1, 2))]
    out = tmp_path / "kernel_manifest.yaml"
    text, changed = kernel_manifest(src, FIXTURES / "context_model.py",
                                    {"source": "...", "profile": {"calls": calls}})
    out.write_text(text)
    m = manifest_mod.load(out)
    prompt, reply = m.workloads
    assert changed and prompt.name == "prompt" and prompt.context is None and prompt.inputs[0].shape == (1, "T")
    assert reply.generated and reply.context == 512 and reply.inputs[0].shape == (1, "new_tokens")
    assert m.primary == {"T": 64, "new_tokens": 2} and m.sweep["new_tokens"] == (2, 3, 4)
    assert "T" in m.serve  # the prompt's range is the job's range still


def test_the_bundle_check_compares_every_output_of_run(tmp_path, monkeypatch):
    """A run() returning several arrays is still reproduced exactly in a fresh
    process: every array is saved and compared, and a changed one fails."""
    from backcountry.harness import model_search
    from backcountry.harness.artifact.emit import check_model_run
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    model_file = tmp_path / "two_outputs.py"
    model_file.write_text(textwrap.dedent("""
        import mlx.core as mx
        import mlx.nn as nn


        def build():
            mx.random.seed(0)
            return nn.Linear(16, 16)


        def run(model, x):
            y = model(x)
            return y, {"doubled": y * 2}
    """))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {model_file}
        model_type: other
        identical: output
        baseline: plain
        workloads:
          - inputs: [{{shape: [4, 16], dtype: float32}}]
            name: main
        budget: {{total: 0, model: 1}}
    """))
    loop = '''def run(model, *inputs, helpers):
    return helpers["app_run"](model, *inputs)
'''
    judge = Scripted([{"hypothesis": "the same loop", "kind": "same", "source": loop},
                      {"verdict": "accept", "reason": "same work"}])
    runner = JobRunner(manifest, tmp_path / "work", session=Session(sleep=lambda s: None),
                       judge_factory=lambda region: judge)
    report = runner.run()
    assert report.model_level["final"]["ok"], report.model_level["history"]
    artifact = runner.emit_artifact(tmp_path / "artifact")  # runs the fresh-process check
    # a result that is the loop alone says so at the top, and its section is written once
    from backcountry.harness.artifact.bundle import NO_KERNEL
    from backcountry.harness.artifact.emit import write_model_run
    write_model_run(artifact, report.model_level, artifact.name)  # an export run again over the folder
    readme = (artifact / "README.md").read_text()
    assert readme.count("## Inference loop") == 1 and NO_KERNEL not in readme
    assert "The result is the inference loop below" in readme
    case = tmp_path / "work" / "model_level_check.safetensors"
    # a pinned model the loop there never reads fails the check: identical
    # outputs alone pass a loop that quietly ran without its drafter
    snapshot = tmp_path / "hub" / "models--org--drafter" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    pinned = {**report.model_level, "assets": [{"repo": "org/drafter", "revision": "rev", "model_type": "x_mtp"}]}
    with pytest.raises(RuntimeError, match="unused_assets"):
        check_model_run(artifact, str(model_file), case, pinned)
    data = dict(mx.load(str(case)))
    assert {"out0_0", "out0_1", "out1_0", "out1_1"} <= set(data)
    data["out0_1"] = data["out0_1"] + 1
    mx.save_safetensors(str(case), data)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        check_model_run(artifact, str(model_file), case, report.model_level)


def test_a_bundle_pins_the_listed_models_the_loop_read_as_the_job_described_them():
    """The job lists every downloaded model with the same tokenizer; a loop
    picks one (by its type) and reads its path. Only that one is pinned, with
    the fields the loop chose it by."""
    from backcountry.harness.artifact.emit import _pins
    from backcountry_runtime.model_run import run_helpers
    run = load_run('''def setup(model, helpers):
    head = next(a for a in helpers["assets"] if a["model_type"].endswith("_mtp"))
    return {"head": {**head}["path"]}

def run(model, *inputs, helpers):
    return model(*inputs)
''')
    listed = [{"repo": "org/small", "revision": "a", "model_type": "qwen3_5", "path": "/s"},
              {"repo": "org/head", "revision": "b", "model_type": "qwen3_5_mtp", "path": "/h"}]
    assert run_helpers(run, object(), {"assets": listed})["head"] == "/h"
    assert run.__dict__["assets_read"] == {"org/head"}
    used = [a for a in listed if a["repo"] in run.__dict__["assets_read"]]
    assert _pins({"assets": used}) == [{"repo": "org/head", "revision": "b", "model_type": "qwen3_5_mtp"}]


def test_the_loop_profile_counts_outermost_calls_by_part_and_size():
    """The judge's picture of a loop: calls into the model's direct parts (an
    encoder, a denoiser) and named helpers, outermost only, by input size."""
    import mlx.nn as nn
    from backcountry.harness.model_search import profile_calls

    class Pipe(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(8, 8)
            self.denoiser = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

    pipe, helper = Pipe(), nn.Linear(8, 8)

    def loop():
        c = pipe.encoder(mx.ones((1, 8)))
        x = mx.ones((2, 8))
        for _ in range(3):
            x = pipe.denoiser(x + c)   # its inner Linears are not counted again
        return helper(x)

    report = profile_calls(pipe, loop, {"drafter": helper})
    rows = {(r["call"], r["inputs"]): r["count"] for r in report["calls"]}
    assert rows == {("encoder", "[1, 8] float32"): 1, ("denoiser", "[2, 8] float32"): 3,
                    ("drafter", "[2, 8] float32"): 1}
    assert report["run_ms"] >= report["outside_model_calls_ms"] >= 0
    assert nn.Linear.__call__ is nn.Linear.__dict__["__call__"]  # the patch is gone


def test_a_part_called_inside_a_compiled_step_is_named_not_timed():
    """mflux compiles its denoiser step, which calls the transformer: the
    profile waited on the transformer's output there and the job died before
    its first attempt (MLX allows no eval inside mx.compile)."""
    import mlx.nn as nn
    from backcountry.harness.model_search import profile_calls

    class Pipe(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(8, 8)
            self.transformer = nn.Linear(8, 8)

    pipe = Pipe()
    step = mx.compile(lambda x: pipe.transformer(x) * 0.5)

    def loop():
        x = pipe.encoder(mx.ones((2, 8)))
        for _ in range(3):
            x = step(x)
        return x

    report = profile_calls(pipe, loop)
    assert [r["call"] for r in report["calls"]] == ["encoder"]
    assert "called inside a compiled function" in report["note"] and "transformer" in report["note"]


class PassLength:
    """A language model whose scores move with how many tokens a call reads, as
    a mixture-of-experts model's can: read inside a longer pass, a token prefers
    token 1 next; read alone, as plain generation reads the prompt's last token,
    token 0. It records the length of every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, tokens, cache=None):
        n = tokens.shape[1]
        self.calls.append(n)
        cache[0].update_and_fetch(mx.zeros((1, 1, n, 1)), mx.zeros((1, 1, n, 1)))
        return mx.broadcast_to(mx.array([1.0, 1.1 if n > 1 else 0.9, 0.0, 0.0]), (1, n, 4))

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache()]


@pytest.mark.parametrize("length", [3, 2050])
def test_the_check_reads_the_prompt_the_way_plain_generation_does(length):
    """The output check re-reads a prompt with the same calls as plain
    generation (every token but the last in MLX-LM's chunks, then the last
    alone), so plain generation's own tokens are the model's own choices."""
    from types import SimpleNamespace

    from backcountry.harness.model_search import ModelSearch

    model = PassLength()
    prompt = mx.zeros((1, length), dtype=mx.int32)
    plain = [int(t) for t in load_run(GENERATE, "default")(model, prompt, helpers={"steps": 4})]
    generation_calls, model.calls = model.calls, []
    scores = ModelSearch.rescore(SimpleNamespace(original=model), prompt, plain)
    assert scores == [(0.0, True)] * 4
    assert model.calls == generation_calls


def test_a_decode_workload_searches_the_loop_from_a_conversation_in_memory(tmp_path, monkeypatch):
    """With a context, the loop search generates final_benchmark.steps tokens
    after a conversation read into the model's memory before any clock: its
    inputs are those conversations, the default's tokens are plain generation
    from there, a timed run never reads the conversation again, and the
    bundle's run.py reproduces the outputs in a fresh process."""
    from mlx_lm.generate import generate_step
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.model_search import ModelSearch, _token_ids
    from backcountry_runtime.model_run import History, ModelRun
    from tests.test_context_workload import MODEL
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent(f"""
        model: {MODEL}
        model_type: other
        baseline: plain
        identical: output
        workloads:
          - name: decode
            context: 6
            inputs: [{{shape: [1, 1], dtype: int32, high: 32}}]
        budget: {{total: 0, model: 1}}
        final_benchmark: {{steps: 5}}
    """))
    judge = Scripted([{"hypothesis": "the same generation", "kind": "same", "source": SAME_STEP},
                      {"verdict": "accept", "reason": "same work"}])
    runner = JobRunner(manifest, tmp_path / "work", session=Session(sleep=lambda s: None),
                       judge_factory=lambda region: judge)
    runner.load_model()
    search = ModelSearch(runner)
    sets = search._sets("check", 2)
    assert sets[0][0].shape == (1, 7)  # the conversation, then the workload's token
    (history,), _ = search._inputs(search.default, search.original, sets[0])
    assert isinstance(history, History) and [c.offset for c in history.cache] == [6, 6]  # all but the last, read
    model = search.original
    cache = model.make_cache()
    model(history.tokens[:, :-1], cache=cache)
    plain = [int(t) for t, _ in generate_step(history.tokens.reshape(-1)[-1:], model, max_tokens=5, prompt_cache=cache)]
    assert _token_ids(search.reference(sets[0])[0]) == plain
    widths, real = [], type(model).__call__
    monkeypatch.setattr(type(model), "__call__", lambda self, x, *a, **k: widths.append(x.shape[-1]) or real(self, x, *a, **k))
    ModelRun(model, search.default, search.helpers)(history)
    monkeypatch.setattr(type(model), "__call__", real)
    assert widths and set(widths) == {1}  # the timed run only generates
    # a generate_step reads its prompt itself: a conversation is a prompt_cache the caller holds
    with pytest.raises(ValueError, match="read"):
        load_loop(SAME_STEP + "\n\ndef read(model, tokens, helpers):\n    return {}\n")
    runner.tracer.uninstall()

    runner = JobRunner(manifest, tmp_path / "work2", session=Session(sleep=lambda s: None),
                       judge_factory=lambda region: judge)
    report = runner.run()
    assert report.model_level["conversation"] is True and report.model_level["final"]["ok"], report.model_level
    final = report.model_level["final"]
    assert (final["context"], final["steps"]) == (6, 5)  # the one conversation length timed, and the reply length
    artifact = runner.emit_artifact(tmp_path / "artifact")  # its run.py check reads the conversation itself
    assert '"conversation": True' in (artifact / "model_loop.py").read_text()
    assert ("each generating 5 tokens after a 6-token conversation (the only conversation length timed)"
            in (artifact / "README.md").read_text())


def test_a_loop_that_changes_its_tokens_when_timed_or_patches_the_library_fails(tmp_path, monkeypatch):
    """Timed runs are checked too: a loop right on the check sets but wrong
    later fails at its timing. A loop that leaves a library function patched
    fails, and the patch is undone before any reference is computed."""
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.model_search import CALLS as CALLS_CHECKED, ModelSearch
    # right on the check sets (each is checked under every call setting), wrong once timed
    later = SAME_STEP.replace("def generate_step(", "CALLS = []\n\n\ndef generate_step(").replace(
        "    yield from _mlx_lm_generate_step(prompt, model, **kwargs)",
        "    CALLS.append(1)\n"
        "    for token, logprobs in _mlx_lm_generate_step(prompt, model, **kwargs):\n"
        "        yield ((token + 1) % 64 if len(CALLS) > " + str(3 * len(CALLS_CHECKED)) + " else token), logprobs")
    patches = SAME_STEP.replace("    yield from _mlx_lm_generate_step(prompt, model, **kwargs)",
                                "    yield from _mlx_lm_generate_step(prompt, model, **kwargs)\n"
                                "    mx.fast.rms_norm = lambda *a, **k: None").replace(
        "def generate_step(", "import mlx.core as mx\n\n\ndef generate_step(")
    breaks = patches.replace("    yield from _mlx_lm_generate_step(prompt, model, **kwargs)\n"
                             "    mx.fast.rms_norm = lambda *a, **k: None",
                             "    mx.fast.rms_norm = lambda *a, **k: None\n"
                             "    yield from _mlx_lm_generate_step(prompt, model, **kwargs)")
    runner = JobRunner(write_manifest(tmp_path), tmp_path / "work", session=Session(sleep=lambda s: None),
                       judge_factory=lambda region: None)
    runner.load_model()
    runner.tracer.uninstall()
    search = ModelSearch(runner)
    sets = search._sets("check", 3)
    run = load_loop(later)
    assert search.check(run, sets)["ok"]
    comp, problem = search.time(search.default, run, "t")
    assert comp is None and "differ" in problem["reason"]
    rms_norm = mx.fast.rms_norm
    result = search.check(load_loop(patches), sets)
    assert not result["ok"] and "patched" in result["reason"] and mx.fast.rms_norm is rms_norm
    # one that patches and then fails leaves nothing patched either
    result = search.check(load_loop(breaks), sets)
    assert not result["ok"] and "patched" in result["reason"] and mx.fast.rms_norm is rms_norm


def test_a_loops_own_read_is_charged_what_it_takes_beyond_the_default():
    from backcountry.harness.measure.clocks import compare_on_inputs
    session = Session(sleep=lambda s: None)
    session.warm_until_stable = lambda fn: None
    session.timed = lambda fn: (fn(), 0.1)[1]
    sets = [[mx.array([i])] for i in range(4)]
    comp = compare_on_inputs(session, lambda x: None, lambda x: None, sets,
                             prepare=lambda arm, inputs: inputs, charge_ms=lambda arm: 30.0 if arm == "b" else 0.0)
    assert abs(comp.median_delta_ms + 30.0) < 1e-6  # the candidate pays its extra read


def test_a_loop_that_keeps_a_second_copy_of_weights_fails_and_views_or_a_drafter_do_not():
    """Never two copies of the weights: a setup() that merges q/k/v and keeps
    the originals (gpt-oss's shipped loop, 4% more memory) is refused; a merge
    whose originals become views of the merged array keeps one copy and
    passes, even while the search holds the originals to undo it (gpt-oss's
    in-place merge was refused for those), and so does loading a second,
    smaller model beside it."""
    import importlib.util
    from pathlib import Path
    from mlx.utils import tree_flatten
    from backcountry.harness.model_search import _free, kept_beside
    from backcountry_runtime.model_run import load_run, run_helpers
    spec = importlib.util.spec_from_file_location(
        "llama_cache_model", Path(__file__).parent / "fixtures" / "llama_cache_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    head = """import mlx.core as mx
def run(model, tokens, helpers):
    return model(tokens)
"""
    sources = {
        "copy": head + """def setup(model, helpers):
    kept = []
    for layer in model.model.layers:
        a = layer.self_attn
        kept.append(mx.concatenate([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]))
    mx.eval(kept)
    return {"qkv": kept}
""",
        "views": head + """def setup(model, helpers):
    kept = []
    for layer in model.model.layers:
        a = layer.self_attn
        w = mx.concatenate([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
        nq, nk = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
        a.q_proj.weight, a.k_proj.weight, a.v_proj.weight = w[:nq], w[nq:nq + nk], w[nq + nk:]
        kept.append(w)
    mx.eval(kept)
    return {"qkv": kept}
""",
        "drafter": head + f"""def setup(model, helpers):
    import importlib.util
    spec = importlib.util.spec_from_file_location("d", {str(Path(__file__).parent / 'fixtures' / 'llama_cache_model.py')!r})
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    drafter = m.build()
    mx.eval(drafter.parameters())
    return {{"drafter": drafter}}
""",
    }
    verdicts = {}
    for name, source in sources.items():
        model = module.build()
        mx.eval(model.parameters())
        restore = tree_flatten(model.parameters())  # as the search holds them, to undo a loop it does not keep
        _free()
        before = mx.get_active_memory()
        run = load_run(source, f"memory_{name}")
        run_helpers(run, model, {})  # setup() runs once, as a check's first call runs it
        verdicts[name] = kept_beside(run, model, before, restore)
        run = model = restore = None
        _free()
    assert verdicts["copy"] is not None and "second copy" in verdicts["copy"], verdicts
    assert verdicts["views"] is None and verdicts["drafter"] is None, verdicts


def test_a_generate_step_gives_mlx_lms_results_for_any_sampler_processors_and_cache(tmp_path):
    """A found generate_step is checked as a drop-in for MLX-LM's: the same
    tokens and every step's logprobs under greedy decoding, a seeded sampler,
    a repetition penalty and a small prefill_step_size, the same progress
    calls, and the cache left as MLX-LM leaves it (mlx_lm.server reuses it).
    A loop written for greedy decoding alone, as every loop was before, fails."""
    from backcountry.harness.loop import JobRunner
    from backcountry.harness.model_search import ModelSearch
    runner = JobRunner(write_manifest(tmp_path), tmp_path / "work", session=Session(sleep=lambda s: None),
                       judge_factory=lambda region: None)
    runner.load_model()
    runner.tracer.uninstall()
    search = ModelSearch(runner)
    assert search.generation and search.default_source == GENERATION
    sets = search._sets("check", 2)
    assert search.check(load_loop(SAME_STEP), sets) == {"ok": True}
    greedy_only = search.check(load_loop(_step_dropping("sampler")), sets)
    assert not greedy_only["ok"] and greedy_only["call"].get("temp"), greedy_only
    no_penalty = search.check(load_loop(_step_dropping("logits_processors")), sets)
    assert not no_penalty["ok"] and no_penalty["call"] == {"repetition_penalty": 1.1}, no_penalty
    silent = search.check(load_loop(_step_dropping("prompt_progress_callback")), sets)
    assert not silent["ok"] and "prompt_progress_callback" in silent["reason"], silent
    short = SAME_STEP.replace("    yield from _mlx_lm_generate_step(prompt, model, **kwargs)",
                              "    yield from _mlx_lm_generate_step(prompt, model, **kwargs)\n"
                              "    for c in kwargs['prompt_cache']:\n        c.trim(1)")
    left = search.check(load_loop(short), sets)
    assert not left["ok"] and "prompt cache ends" in left["reason"], left


def test_the_shipped_generate_step_is_mlx_lms_and_steps_aside_where_it_was_not_checked():
    """What a user calls: generate_step, stream_generate and generate with
    MLX-LM's signatures, MLX-LM's own stream_generate and generate calling the
    found generate_step (nothing in mlx_lm patched). An argument the job did
    not check, or another model, gets MLX-LM's own generate_step."""
    import importlib
    lm = importlib.import_module("mlx_lm.generate")  # the module: mlx_lm.generate is also its function
    from mlx_lm.models.cache import make_prompt_cache
    from backcountry_runtime.generation import drop_ins, load_generation, model_signature
    from tests.fixtures.llama_cache_model import build
    counted = SAME_STEP.replace("def generate_step(", "CALLS = []\n\n\ndef generate_step(").replace(
        "    yield from", "    CALLS.append(1)\n    yield from")
    step, setup = load_generation(counted)
    model, other = build(), build()
    other.model.layers.pop()  # another model: one layer fewer
    generate_step, stream_generate, generate = drop_ins(step, setup, made_for=model_signature(model))
    assert lm.generate_step is not generate_step and stream_generate.__globals__["generate_step"] is generate_step
    assert generate.__globals__["stream_generate"] is stream_generate
    assert lm.generate.__globals__["stream_generate"] is lm.stream_generate  # mlx_lm itself untouched
    prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)
    calls = step.__globals__["CALLS"]
    mine = [t for t, _ in generate_step(prompt, model, max_tokens=4, prompt_cache=make_prompt_cache(model))]
    theirs = [t for t, _ in lm.generate_step(prompt, model, max_tokens=4, prompt_cache=make_prompt_cache(model))]
    assert mine == theirs and len(calls) == 1
    list(generate_step(prompt, model, max_tokens=2, max_kv_size=64))  # a rotating cache: not checked
    list(generate_step(prompt, other, max_tokens=2))  # not the model it was made for
    assert len(calls) == 1 and generate_step.made_for(model) and not generate_step.made_for(other)
    # a setup that fails where the user runs it (its drafter is not on this
    # Mac) leaves that model on MLX-LM's own generate_step, with a warning
    def missing(model, helpers):
        raise FileNotFoundError("no drafter at /elsewhere")
    generate_step = drop_ins(step, missing, made_for=model_signature(model))[0]
    with pytest.warns(UserWarning, match="could not set up"):
        stock = [t for t, _ in generate_step(prompt, model, max_tokens=4, prompt_cache=make_prompt_cache(model))]
    assert stock == theirs and len(calls) == 1


def test_a_generation_loop_sees_only_what_ships_and_the_bundle_must_call_the_model_as_it_did(tmp_path, monkeypatch):
    """A found generate_step gets the same helpers in the job as where it
    ships (the models beside it and the tokenizer, plus its setup's own):
    a Qwen3.8 loop calibrated with the job's tokenizer, set up without it
    where it shipped, and ran MLX-LM's own path there (identical tokens,
    1.0x where the job measured 1.42x). The bundle check runs the shipped
    generate_step's setup first and requires it to call the model exactly
    as the job's loop did."""
    from backcountry.harness import model_search
    from backcountry.harness.artifact.emit import check_model_run
    from backcountry.harness.loop import JobRunner
    from backcountry_runtime.generation import SHIPPED
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    log = tmp_path / "helpers.jsonl"
    loop = f'''import json
from mlx_lm.generate import generate_step as _mlx_lm_generate_step


def setup(model, helpers):
    with open({str(log)!r}, "a") as f:
        f.write(json.dumps(["setup", sorted(helpers)]) + "\\n")
    return {{"ready": True}}


def generate_step(prompt, model, *, helpers=None, **kwargs):
    with open({str(log)!r}, "a") as f:
        f.write(json.dumps(["step", sorted(helpers)]) + "\\n")
    if helpers.get("slow"):
        model(prompt[None, :1])  # a call the job's loop never made
    yield from _mlx_lm_generate_step(prompt, model, **kwargs)
'''
    judge = Scripted([{"hypothesis": "the same loop", "kind": "same", "source": loop},
                      {"verdict": "accept", "reason": "same work"}])
    runner = JobRunner(write_manifest(tmp_path, "budget: {total: 0, model: 1}"), tmp_path / "work",
                       session=Session(sleep=lambda s: None), judge_factory=lambda region: judge)
    report = runner.run()
    assert report.model_level["final"]["ok"], report.model_level["history"]
    seen = [json.loads(line) for line in log.read_text().splitlines()]
    assert {tuple(keys) for kind, keys in seen if kind == "setup"} == {tuple(sorted(SHIPPED))}
    assert {tuple(keys) for kind, keys in seen if kind == "step"} == {tuple(sorted((*SHIPPED, "ready")))}
    artifact = runner.emit_artifact(tmp_path / "artifact")  # the fresh-process check passes
    loader = artifact / "model_loop.py"
    loader.write_text(loader.read_text().replace('"tokenizer": pinned_tokenizer()}', '"tokenizer": pinned_tokenizer(), "slow": True}'))
    with pytest.raises(RuntimeError, match="did not run as the job checked it"):
        check_model_run(artifact, str(FIXTURES / "llama_cache_model.py"),
                        tmp_path / "work" / "model_level_check.safetensors", report.model_level)


def test_a_generation_loop_that_calls_the_models_parts_alone_passes_the_bundle_check(tmp_path, monkeypatch):
    """A Qwen3.8 MTP loop drove the model's layers and its own head and never
    called the model itself: tokens identical in a fresh process, yet the
    bundle check, which required calls to the model, left the 1.225x loop out."""
    from backcountry.harness import model_search
    from backcountry.harness.artifact.emit import check_model_run
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    loop = '''from mlx_lm.generate import generate_step as _mlx_lm_generate_step


class _Parts:
    """The model's layers and output matrix, called without the model."""
    def __init__(self, model):
        self.model = model

    def __call__(self, x, cache=None):
        hidden = self.model.model(x, cache=cache)
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)


def generate_step(prompt, model, *, helpers=None, **kwargs):
    if helpers.get("slow"):
        model.model.embed_tokens(prompt[None, :1])  # a layer call the job's loop never made
    yield from _mlx_lm_generate_step(prompt, _Parts(model), **kwargs)
'''
    judge = Scripted([{"hypothesis": "the same loop through the parts", "kind": "same", "source": loop},
                      {"verdict": "accept", "reason": "same work"}])
    runner = JobRunner(write_manifest(tmp_path, "budget: {total: 0, model: 1}"), tmp_path / "work",
                       session=Session(sleep=lambda s: None), judge_factory=lambda region: judge)
    report = runner.run()
    assert report.model_level["final"]["ok"], report.model_level["history"]
    artifact = runner.emit_artifact(tmp_path / "artifact")
    assert (artifact / "model_loop.py").is_file()
    readme = (artifact / "README.md").read_text()
    assert "## Inference loop" in readme and "not_shipped" not in report.model_level
    # its usage calls the loop: mlx_lm's own generate would not run it
    assert "from backcountry import generate" in readme and "from mlx_lm import load, generate" not in readme
    # a shipped loop that runs another way through the parts alone is caught too
    loader = artifact / "model_loop.py"
    loader.write_text(loader.read_text().replace('"tokenizer": pinned_tokenizer()}', '"tokenizer": pinned_tokenizer(), "slow": True}'))
    with pytest.raises(RuntimeError, match="did not run as the job checked it"):
        check_model_run(artifact, str(FIXTURES / "llama_cache_model.py"),
                        tmp_path / "work" / "model_level_check.safetensors", report.model_level)


def test_a_loop_that_does_not_hold_in_a_fresh_process_is_left_out_not_a_failed_job(tmp_path, monkeypatch):
    """The job checks a loop as it ships, so a bundle check that still fails
    is the tool's fault, not the user's: the bundle ships without the loop
    (the kernels, when it has any), the report says why, and the job ends."""
    from backcountry.harness import loop as loop_mod
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())

    def fails(*a, **k):
        raise RuntimeError("the artifact's run.py did not reproduce the job's outputs in a fresh process: x")
    monkeypatch.setattr(loop_mod, "check_model_run", fails)
    judge = Scripted([{"hypothesis": "the same loop", "kind": "same", "source": SAME_STEP},
                      {"verdict": "accept", "reason": "same work"}])
    runner = JobRunner(write_manifest(tmp_path, "budget: {total: 0, model: 1}"), tmp_path / "work",
                       session=Session(sleep=lambda s: None), judge_factory=lambda region: judge)
    report = runner.run()
    assert report.model_level["final"]["ok"]
    artifact = runner.emit_artifact(tmp_path / "artifact")
    assert not (artifact / "run.py").exists() and not (artifact / "model_loop.py").exists()
    assert "## Inference loop" not in (artifact / "README.md").read_text()
    assert "backcountry import generate" not in (artifact / "README.md").read_text()
    assert "did not reproduce" in report.model_level["not_shipped"]
    assert [r for r in runner.log.rows() if r.get("kind") == "loop_not_shipped"]


def test_a_generation_loop_is_checked_and_timed_with_greedy_passed_as_a_function(tmp_path, monkeypatch):
    """mlx-lm's server and LM Studio's mlx-engine pass greedy decoding as a
    sampler function, never None: Qwen3.8-27B's loop, faster only when
    sampler was None, ran 1.00x there. Its checks include that call, and
    every timed call makes it."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    from backcountry_runtime.generation import call_arguments, greedy
    assert call_arguments({"greedy": True})["sampler"] is greedy
    assert {"greedy": True} in model_search.CALLS
    timed = []
    real_time = model_search.ModelSearch.time

    def spy(self, best, candidate, purpose, sets=None, base_model=None, helpers=None):
        seen = {}
        original_init = model_search.ModelRun.__init__

        def init(run_self, model, run, h):
            seen.setdefault("call", h.get("call"))
            original_init(run_self, model, run, h)
        monkeypatch.setattr(model_search.ModelRun, "__init__", init)
        try:
            return real_time(self, best, candidate, purpose, sets, base_model, helpers)
        finally:
            monkeypatch.setattr(model_search.ModelRun, "__init__", original_init)
            timed.append((purpose, seen.get("call")))
    monkeypatch.setattr(model_search.ModelSearch, "time", spy)
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    judge = Scripted([{"hypothesis": "the same loop", "kind": "same", "source": SAME_STEP},
                      {"verdict": "accept", "reason": "same work"}])
    runner = JobRunner(write_manifest(tmp_path, "budget: {total: 0, model: 1}"), tmp_path / "work",
                       session=Session(sleep=lambda s: None), judge_factory=lambda region: judge)
    report = runner.run()
    assert report.model_level["final"]["ok"]
    replies = [call for purpose, call in timed if purpose != "final-prefill"]
    assert replies and all(call == {"greedy": True} for call in replies), timed


def test_a_tokenizer_is_pinned_only_from_a_hugging_face_snapshot():
    import types
    from backcountry.harness.model_search import tokenizer_pin
    hub = "/cache/hub/models--org--model/snapshots/abc123"
    assert tokenizer_pin(types.SimpleNamespace(name_or_path=hub)) == {"repo": "org/model", "revision": "abc123"}
    assert tokenizer_pin(types.SimpleNamespace(name_or_path="/my/local/model")) is None
    assert tokenizer_pin(None) is None


def test_backcountrys_generate_finds_the_loop_stored_for_the_model(tmp_path, monkeypatch):
    """`from backcountry import generate_step` (stream_generate, generate)
    in place of mlx_lm's: the newest loop in the local store made for the
    model it is given, MLX-LM's own for any other; compare() times the two
    through the user's function, and says so when that function never
    reaches the loop (it called mlx_lm's own)."""
    import importlib
    lm = importlib.import_module("mlx_lm.generate")
    import backcountry
    from backcountry_runtime import generation
    from backcountry_runtime.generation import model_signature
    from tests.fixtures.llama_cache_model import build
    model, other = build(), build()
    other.model.layers.pop()
    counted = SAME_STEP.replace("def generate_step(", "CALLS = []\n\n\ndef generate_step(").replace(
        "    yield from", "    CALLS.append(1)\n    yield from")
    folder = tmp_path / "artifacts" / "job-1"
    folder.mkdir(parents=True)
    (folder / "bundle.json").write_text(json.dumps({"mlx": mx.__version__}))
    (folder / "model_loop.py").write_text(
        "from backcountry_runtime.generation import drop_ins, load_generation\n"
        f"_step, _setup = load_generation({counted!r})\n"
        f"generate_step, stream_generate, generate = drop_ins(_step, _setup, made_for={model_signature(model)!r})\n")
    monkeypatch.setenv("BACKCOUNTRY_HOME", str(tmp_path))
    monkeypatch.setattr(generation, "_found", {})
    prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)
    tokens = lambda step, m: [t for t, _ in step(prompt, m, max_tokens=4)]
    import warnings
    ran = generation._ran
    assert tokens(backcountry.generate_step, model) == tokens(lm.generate_step, model)
    assert generation._ran == ran + 1  # the stored loop
    assert tokens(backcountry.generate_step, other) == tokens(lm.generate_step, other)
    assert generation._ran == ran + 1  # another model: MLX-LM's own
    ran = generation._ran
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a loop-only artifact: apply() finds no layers, and says nothing
        assert backcountry.apply(model) is model
    result = backcountry.compare(lambda m: tokens(backcountry.generate_step, m), model, runs=2)
    assert result.loop and result.identical and result.replaced == 0 and "its generation loop" in str(result)
    assert generation._ran == ran + 3  # the untimed call and two timed ones, never in the stock arm
    with pytest.warns(UserWarning, match="did not run"):
        assert not backcountry.compare(lambda m: tokens(lm.generate_step, m), model, runs=2).loop
    # checked with another version of a library the model code ran on (MLX-LM's
    # generate_step and model code differ between versions): the original runs
    (folder / "requirements.txt").write_text("# pins\nmlx-lm==0.0.1\n")
    monkeypatch.setattr(generation, "_found", {})
    ran = generation._ran
    with pytest.warns(UserWarning, match="mlx-lm 0.0.1 -> "):
        assert tokens(backcountry.generate_step, model) == tokens(lm.generate_step, model)
    assert generation._ran == ran


def test_the_loop_search_stops_at_a_finish(tmp_path, monkeypatch):
    """A loop attempt on a large model takes tens of minutes: the search stops
    at `backcountry finish`, as the kernel search does."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())
    for stop in ("finish",):
        work = tmp_path / stop
        (tmp_path / stop).mkdir()

        class Late(Scripted):
            def _ask(self, system, messages):
                reply = super()._ask(system, messages)
                if not self.replies:  # the attempt's review: the operator finishes
                    (work / "job" / "finish-search.request").write_text("finish\n")
                return reply
        judge = Late([{"hypothesis": "one", "kind": "same", "source": SAME_STEP},
                      {"verdict": "accept", "reason": "same work"}])
        runner = JobRunner(write_manifest(tmp_path / stop, "budget: {total: 0, model: 3}"), work / "job",
                           judge_factory=lambda region: judge, session=Session(sleep=lambda s: None))
        level = runner.run().model_level
        assert [h["attempt"] for h in level["history"]] == [0] and level["final"]["ok"]
        rows = [json.loads(x) for x in (work / "job" / "run.jsonl").read_text().splitlines()]
        assert [r["at"] for r in rows if r["kind"] == "model_search_stopped"] == [1]


def test_a_judge_that_does_not_answer_spends_no_loop_attempts_and_pauses_the_job(tmp_path, monkeypatch):
    """A loop request with no answer (no network, an expired login, a reply
    over the time limit) is asked again without spending an attempt; three in
    a row pause the job where a resume asks again, as in the kernel search."""
    from backcountry.harness import model_search
    from backcountry.harness.loop import JobRunner
    monkeypatch.setattr(model_search, "compare_on_inputs", lambda *a, **k: _win())

    class Flaky(Scripted):
        def _ask(self, system, messages):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("claude judge gave no reply within 1800s")
            return super()._ask(system, messages)

    judge = Flaky([{"hypothesis": "one", "kind": "same", "source": SAME_STEP},
                   {"verdict": "accept", "reason": "same work"}])
    judge.failures = 2
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    level = JobRunner(write_manifest(tmp_path / "a", "budget: {total: 0, model: 1}"), tmp_path / "a" / "job",
                      judge_factory=lambda region: judge, session=Session(sleep=lambda s: None)).run().model_level
    assert [h["attempt"] for h in level["history"]] == [0] and level["history"][0]["verdict"] == "accepted"

    dead = Flaky([])
    dead.failures = 3
    report = JobRunner(write_manifest(tmp_path / "b", "budget: {total: 0, model: 3}"), tmp_path / "b" / "job",
                       judge_factory=lambda region: dead, session=Session(sleep=lambda s: None)).run()
    assert report.session["status"] == "paused" and "did not answer 3 times" in report.session["reason"]
