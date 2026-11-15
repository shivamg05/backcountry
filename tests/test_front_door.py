"""A model id and how it is used, in place of a model file and a manifest:
`backcountry run <model id> --use chat|prompt|reply [--attempts N]` writes both into
the run folder, every inferred value commented, as a manifest that loads."""
import json
from pathlib import Path

import pytest

from backcountry.harness import manifest as manifest_mod
from backcountry.harness.front_door import is_model_file, is_model_id, workloads_from, write, write_from_examples


def model_folder(tmp_path, name, **config):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3", "vocab_size": 151936, **config}))
    return folder


@pytest.mark.parametrize("use,names", [("chat", ["prompt", "reply"]), ("prompt", ["prompt"]), ("reply", ["reply"])])
def test_a_model_id_becomes_a_model_file_and_a_manifest_that_loads(tmp_path, use, names):
    folder = model_folder(tmp_path, "dense")
    written = write(str(folder), use, tmp_path / "run", attempts=250)
    m = manifest_mod.load(written)
    assert [w.name for w in m.workloads] == names and m.model_type == "llm"
    assert m.identical == "output" and m.use_library_inference
    assert (m.budget_per_region, m.budget_total, m.budget_model) == (25, 250, 20)
    assert f'load("{folder}")' in (tmp_path / "run" / "model.py").read_text()
    assert ("range" in written.read_text()) == (use != "reply")


def test_experts_and_a_vision_checkpoints_text_config_are_read(tmp_path):
    folder = model_folder(tmp_path, "moe", text_config={"vocab_size": 248320, "num_experts": 64})
    text = write(str(folder), "reply", tmp_path / "run").read_text()
    assert "model_type: moe-llm" in text and "high: 248320" in text
    assert "budget: {per_region: 12, total: 36, model: 6}" in text  # no --attempts: a short run


def test_what_counts_as_a_model_id(tmp_path):
    assert is_model_id("mlx-community/Qwen3-8B-4bit")
    assert is_model_id(str(model_folder(tmp_path, "local")))
    (tmp_path / "m.yaml").write_text("x")
    assert not is_model_id(str(tmp_path / "m.yaml")) and not is_model_id("models/workloads/x.yaml")
    with pytest.raises(ValueError, match="--use"):
        write(str(tmp_path / "local"), "image", None, tmp_path / "run2")


def test_helpers_published_beside_the_model_are_found_on_the_hub_and_named_in_the_manifest(tmp_path):
    """A user names the model; its multi-token head or drafter, published beside
    it, is downloaded for the loop without them knowing it exists (the
    Qwen3.8-27B loop is 1.21x with its MTP head and has nothing without it)."""
    from types import SimpleNamespace

    from backcountry.harness.front_door import helpers_for
    hub = ["Qwen3.8-27B-4bit", "Qwen3.8-27B-MTP-bf16", "Qwen3.8-27B-MTP-4bit", "Qwen3.8-27B-MTP-8bit",
           "Qwen3.8-27B-TWIN-TURBO-Uncensored-oQ4-mtp", "Swift-Qwen3.8-27b-oQ6e-mtp",
           "gemma-4-26B-A4B-it-assistant-bf16", "gemma-4-26B-A4B-it-qat-assistant-4bit",
           "LFM2.5-1.2B-Instruct-MLX-4bit", "LFM2.5-1.2B-Instruct-DSpark"]
    asked = []

    def list_models(author, search, limit):
        asked.append((author, search))
        return [SimpleNamespace(id=f"{author}/{n}") for n in hub if search in n.lower()]
    # the model's own precision when published, else another; fine-tunes and
    # other models whose names contain it are not the model's helpers
    assert helpers_for("mlx-community/Qwen3.8-27B-4bit", list_models) == ["mlx-community/Qwen3.8-27B-MTP-4bit"]
    assert asked == [("mlx-community", "qwen3.8-27b")]
    assert helpers_for("mlx-community/gemma-4-26B-A4B-it-4bit", list_models) == \
        ["mlx-community/gemma-4-26B-A4B-it-assistant-bf16"]
    # Liquid's drafters: a DSpark beside the MLX conversion of the model
    assert helpers_for("LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit", list_models) == ["LiquidAI/LFM2.5-1.2B-Instruct-DSpark"]
    assert helpers_for("mlx-community/Qwen3-8B-4bit", list_models) == []
    assert helpers_for(str(model_folder(tmp_path, "local")), list_models) == []  # a local folder
    assert helpers_for("mlx-community/Qwen3.8-27B-4bit", lambda **k: 1 / 0) == []  # no answer from the Hub

    folder = model_folder(tmp_path, "dense")
    m = manifest_mod.load(write(str(folder), "reply", tmp_path / "run", helpers=["mlx-community/Qwen3.8-27B-MTP-4bit"]))
    assert m.assets == ("mlx-community/Qwen3.8-27B-MTP-4bit",)
    assert manifest_mod.load(write(str(folder), "reply", tmp_path / "run2")).assets == ()


def test_a_loop_is_offered_the_head_then_trained_drafters_then_the_smallest_sibling():
    """With no assets named, the loop search downloads what is published for
    the model, in this order: its own multi-token head, drafters trained for
    it (the author's, then z-lab's DFlash), then the author's smallest model of
    the family at the same precision. A head for another model is never one."""
    from types import SimpleNamespace

    from backcountry.harness.front_door import published_helpers
    hub = {"mlx-community": ["Qwen3-8B-4bit", "Qwen3-8B-MTP-4bit", "Qwen3-8B-eagle3-4bit", "Qwen3-0.6B-4bit",
                             "Qwen3-1.7B-4bit", "Qwen3-0.6B-bf16", "Qwen3-14B-4bit", "Qwen3.8-27B-MTP-4bit",
                             "Qwen3-30B-A3B-4bit"],
           "z-lab": ["Qwen3-8B-DFlash-b16", "Qwen3-8B-DFlash-b16-GGUF", "Qwen3-4B-DFlash-b16"]}

    def list_models(author, search, limit):
        return [SimpleNamespace(id=f"{author}/{n}") for n in hub.get(author, []) if search in n.lower()]
    assert published_helpers("mlx-community/Qwen3-8B-4bit", list_models) == [
        "mlx-community/Qwen3-8B-MTP-4bit", "z-lab/Qwen3-8B-DFlash-b16", "mlx-community/Qwen3-8B-eagle3-4bit",
        "mlx-community/Qwen3-0.6B-4bit"]
    # a mixture-of-experts model has no smaller model of its shape; the smallest has none smaller
    assert published_helpers("mlx-community/Qwen3-30B-A3B-4bit", list_models) == []
    assert published_helpers("mlx-community/Qwen3-0.6B-4bit", list_models) == []
    assert published_helpers("mlx-community/Qwen3-8B-4bit", lambda **k: 1 / 0) == []  # no answer from the Hub
    assert published_helpers("", list_models) == []                                     # not a Hub model


def test_the_model_repo_is_read_from_its_hugging_face_cache_folder(tmp_path):
    from types import SimpleNamespace

    from backcountry.harness.model_search import repo_of
    snapshot = tmp_path / "hub" / "models--mlx-community--Qwen3-8B-4bit" / "snapshots" / "abc123"
    assert repo_of(SimpleNamespace(name_or_path=str(snapshot))) == "mlx-community/Qwen3-8B-4bit"
    assert repo_of(SimpleNamespace(name_or_path=str(tmp_path / "my-model"))) == ""
    assert repo_of(None) == ""


def test_assets_are_hugging_face_ids_for_the_loop_search(tmp_path):
    folder = model_folder(tmp_path, "dense")
    text = write(str(folder), "reply", tmp_path / "run", helpers=["a/head"]).read_text()
    for bad, match in [(text.replace("  - a/head", "  - head"), "Hugging Face ids"),
                       (text.replace("identical: output", "identical: computation"), "identical: output")]:
        (tmp_path / "bad.yaml").write_text(bad.replace("model: model.py", f"model: {tmp_path / 'run' / 'model.py'}"))
        with pytest.raises(manifest_mod.ManifestError, match=match):
            manifest_mod.load(tmp_path / "bad.yaml")
    # the kernel search after the loop runs the model alone: no assets
    import yaml

    from backcountry.harness.model_search import kernel_manifest
    kernels, _ = kernel_manifest(tmp_path / "run" / "manifest.yaml", tmp_path / "run" / "model.py", {})
    assert "assets" not in yaml.safe_load(kernels)


def test_attempts_bound_the_search_and_the_result_says_how_much_it_rests_on(tmp_path):
    """--attempts sets the search's depth (no time limit: a run's length
    follows from its attempts), and the report says how much searching the
    result rests on and how long the checks after it took."""
    from backcountry.harness.cli import search_text
    from backcountry.harness.report import Report
    folder = model_folder(tmp_path, "dense")
    m = manifest_mod.load(write(str(folder), "prompt", tmp_path / "a", attempts=100))
    assert (m.budget_per_region, m.budget_total) == (25, 100)
    m = manifest_mod.load(write(str(folder), "prompt", tmp_path / "b", attempts=10))
    assert (m.budget_per_region, m.budget_total) == (10, 10)
    with pytest.raises(ValueError, match="at least 1"):
        write(str(folder), "prompt", tmp_path / "c", attempts=0)
    report = Report(regions=[{"hypotheses": 12}, {"hypotheses": 3}, {"hypotheses": 0}],
                    model_level={"history": [{}] * 6}, session={"job_active_s": 3 * 3600, "finish_requested": True})
    assert search_text(report) == "search: 6 loop attempts, 15 kernel attempts in 2 regions, 3.0 h, stopped by backcountry finish"
    # the checks after the search (every prompt length, the final timings, the export) are not search time
    report.session.update(search_s=8 * 3600, job_active_s=12.6 * 3600, finish_requested=False)
    assert search_text(report) == ("search: 6 loop attempts, 15 kernel attempts in 2 regions, 8.0 h; "
                                   "checks and export after it: 4.6 h")


EXAMPLES = Path(__file__).parent / "fixtures" / "examples_model.py"


def call(*shapes, dtype="float32", ints=None):
    return [{"shape": list(s), "dtype": dtype, **(ints or {})} for s in shapes]


def test_a_model_file_with_examples_becomes_a_manifest_that_loads(tmp_path):
    m = manifest_mod.load(write_from_examples(EXAMPLES, tmp_path / "run", attempts=60))
    assert m.model_path == EXAMPLES.resolve() and (m.budget_per_region, m.budget_total) == (25, 60)
    (w,) = m.workloads
    assert [(i.shape, i.dtype) for i in w.inputs] == [((1, "T", 64), "bfloat16"), ((1, "T"), "int32")]
    assert (w.inputs[1].low, w.inputs[1].high) == (3, 40)
    assert m.primary == {"T": 16} and m.serve["T"].sizes() == (8, 16, 32)


def test_the_sizes_examples_change_become_the_named_size():
    one = workloads_from([call((1, 4, 8))], "m")
    assert one == ([{"name": "m", "inputs": call((1, 4, 8))}], {}, [])
    two = workloads_from([call((1, 4, 8)), call((1, 9, 8))], "m")
    assert [w["name"] for w in two[0]] == ["m_4", "m_9"] and two[1] == {}  # a range needs three sizes
    many = workloads_from([call((1, 4, 8), (4,)), call((1, 2, 8), (2,)), call((1, 6, 8), (6,))], "m")
    assert many[0][0]["inputs"] == call((1, "T", 8), ("T",)) and many[1:] == ({"T": 4}, [4, 2, 6])


def test_examples_that_are_not_one_kind_of_call_are_refused():
    with pytest.raises(ValueError, match="more than one independent way"):
        workloads_from([call((1, 4, 8)), call((1, 6, 9)), call((1, 6, 8))], "m")
    with pytest.raises(ValueError, match="same kind of call"):
        workloads_from([call((1, 4, 8)), call((1, 4))], "m")
    with pytest.raises(ValueError, match="same kind of call"):
        workloads_from([call((1, 4, 8)), call((1, 4, 8), dtype="float16")], "m")


def test_a_model_file_without_examples_says_what_is_missing(tmp_path):
    bare = tmp_path / "bare.py"
    bare.write_text("def build():\n    return None\n")
    with pytest.raises(ValueError, match="no examples"):
        write_from_examples(bare, tmp_path / "run")
    assert is_model_file(str(bare)) and not is_model_file(str(tmp_path / "missing.py"))


def test_a_model_file_with_run_takes_its_examples_as_run_inputs(tmp_path):
    model = tmp_path / "loop_model.py"
    model.write_text("import mlx.core as mx\n\n\ndef build():\n    return lambda x: x * 2\n\n\n"
                     "def run(model, latents):\n    for _ in range(4):\n        latents = model(latents)\n    return latents\n\n\n"
                     "def examples():\n    return [mx.zeros((1, 8, 4))]\n")
    m = manifest_mod.load(write_from_examples(model, tmp_path / "run"))
    assert m.final_benchmark.steps == 1 and [i.shape for i in m.workloads[0].inputs] == [(1, 8, 4)]
    assert "the inputs of run()" in (tmp_path / "run" / "manifest.yaml").read_text()


def test_rounding_differences_are_allowed_only_when_asked_for(tmp_path):
    assert not manifest_mod.load(write_from_examples(EXAMPLES, tmp_path / "exact")).tolerance_allowed
    m = manifest_mod.load(write_from_examples(EXAMPLES, tmp_path / "loose", tolerances="default"))
    assert m.tolerance_allowed and m.tolerances is None  # each output type's default limit
