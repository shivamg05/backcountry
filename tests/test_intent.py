"""backcountry init and backcountry.yaml: what the argument is, the questions
only the user can answer, the plan said back, and the full manifest a run
writes from it."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backcountry.harness import intent
from backcountry.harness.intent import IntentError, Target

FIXTURES = Path(__file__).parent / "fixtures"
QWEN_CONFIG = {"model_type": "qwen3", "vocab_size": 151936}


class Hub:
    """A stand-in for huggingface_hub.HfApi."""

    def __init__(self, error=None, tags=("mlx",), names=()):
        self.error, self.tags, self.names = error, tags, names

    def model_info(self, model_id):
        if self.error:
            raise self.error
        return SimpleNamespace(tags=list(self.tags))

    def list_models(self, search, author, limit):
        return [SimpleNamespace(id=f"{author}/{n}") for n in self.names if search.lower() in n.lower()]


def _model_folder(tmp_path, config=QWEN_CONFIG):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def test_init_tells_files_folders_manifests_and_hub_ids_apart(tmp_path):
    no_examples = tmp_path / "m.py"
    no_examples.write_text("def build():\n    return None\n")
    with pytest.raises(IntentError, match="has no examples"):
        intent.classify(str(no_examples))
    no_build = tmp_path / "n.py"
    no_build.write_text("def examples():\n    return []\n")
    with pytest.raises(IntentError, match="has no build"):
        intent.classify(str(no_build))
    assert intent.classify(str(FIXTURES / "examples_model.py")).kind == "model_file"
    manifest = tmp_path / "full.yaml"
    manifest.write_text("model: m.py\nworkloads: []\n")
    assert intent.classify(str(manifest)).kind == "manifest"
    short = tmp_path / "backcountry.yaml"
    short.write_text("model: org/name\nattempts: 3\n")
    assert intent.classify(str(short)).kind == "intent"
    assert intent.classify(str(_model_folder(tmp_path))).kind == "local_model"
    (tmp_path / "empty").mkdir()
    with pytest.raises(IntentError, match="without a config.json"):
        intent.classify(str(tmp_path / "empty"))
    with pytest.raises(IntentError, match="not a file here and not a Hugging Face id"):
        intent.classify("qwen3")


def test_a_hub_id_is_checked_without_downloading_weights(monkeypatch):
    import backcountry.harness.front_door as front_door
    monkeypatch.setattr(front_door, "config_of", lambda model_id: QWEN_CONFIG)
    target = intent.classify("https://huggingface.co/mlx-community/Qwen3-8B-4bit", hub=Hub())
    assert (target.kind, target.value, target.note) == ("hub_model", "mlx-community/Qwen3-8B-4bit", None)

    class RepositoryNotFoundError(Exception):
        pass

    class GatedRepoError(Exception):
        pass

    names = ("Qwen3-8B-4bit", "Qwen3-8B-6bit", "Qwen3.6-35B-A3B-8bit")
    with pytest.raises(IntentError, match="did you mean mlx-community/Qwen3-8B-4bit"):
        intent.classify("mlx-community/Qwen3-8B-4bitt", hub=Hub(RepositoryNotFoundError(), names=names))
    with pytest.raises(IntentError, match="is gated"):
        intent.classify("meta-llama/Llama-3-8B", hub=Hub(GatedRepoError()))
    with pytest.raises(IntentError, match="could not reach Hugging Face"):
        intent.classify("org/name", hub=Hub(ConnectionError("offline")))
    unconverted = intent.classify("Qwen/Qwen3-8B", hub=Hub(tags=("transformers",), names=names))
    assert "not an MLX conversion" in unconverted.note


def test_questions_are_only_what_the_model_cannot_say():
    llm = Target("hub_model", "org/llm", QWEN_CONFIG)
    replies = intent.answer(llm, {"optimize": "replies", "history_tokens": "4096"})
    assert set(replies) == {"optimize", "history_tokens", "reply_tokens", "keep", "attempts"}  # no prompt sizes
    assert replies["history_tokens"] == 4096 and replies["keep"] == "same-output"
    asked = []
    intent.answer(llm, {}, ask=lambda q: asked.append(q["id"]) or None)
    assert asked == ["optimize", "prompt_tokens", "optimize_at", "history_tokens", "reply_tokens", "keep", "attempts"]
    model_file = Target("model_file", "m.py")
    assert set(intent.answer(model_file, {})) == {"tolerance", "attempts"}
    with pytest.raises(IntentError, match="one of prompts, replies, chat"):
        intent.answer(llm, {"optimize": "images"})
    with pytest.raises(IntentError, match="a range like 32-2049"):
        intent.answer(llm, {"prompt_tokens": "2049-32"})
    with pytest.raises(IntentError, match="must be within prompt_tokens"):
        intent.answer(llm, {"prompt_tokens": "32-256", "optimize_at": "512"})


def test_the_plan_is_said_back_in_the_users_terms():
    llm = Target("hub_model", "org/llm", QWEN_CONFIG)
    text = intent.summary(llm, intent.answer(llm, {"optimize": "chat", "keep": "same-math"}))
    assert "reading prompts of 32-2,049 tokens (optimized at 512)" in text
    assert "writing 16-token replies after 2,048 tokens of conversation" in text
    assert "the exact same math" in text


def test_a_language_model_intent_becomes_the_manifest_it_describes(tmp_path):
    folder = _model_folder(tmp_path)
    target = intent.classify(str(folder))
    answers = intent.answer(target, {"optimize": "chat", "prompt_tokens": "64-4096", "optimize_at": "1024",
                                     "history_tokens": "1000", "reply_tokens": "32", "keep": "same-math",
                                     "attempts": "40"})
    short = intent.write_intent(target, answers, tmp_path / "backcountry.yaml")
    assert intent.classify(str(short)).kind == "intent"
    manifest = yaml.safe_load(intent.resolve(short, tmp_path / "run").read_text())
    assert manifest["identical"] == "computation" and "assets" not in manifest
    assert manifest["primary"] == {"T": 1024} and manifest["range"] == {"T": [64, 4096]}
    reply = next(w for w in manifest["workloads"] if w["name"] == "reply")
    assert reply["context"] == 1000 and manifest["final_benchmark"]["steps"] == 32
    assert manifest["budget"] == {"per_region": 25, "total": 40}
    assert (tmp_path / "run" / "model.py").is_file()


def test_a_model_file_intent_takes_only_attempts_and_tolerance(tmp_path):
    target = intent.classify(str(FIXTURES / "examples_model.py"))
    short = intent.write_intent(target, intent.answer(target, {"tolerance": "default", "attempts": "3"}),
                                tmp_path / "backcountry.yaml")
    manifest = yaml.safe_load(intent.resolve(short, tmp_path / "run").read_text())
    assert manifest["tolerances"] == "default" and manifest["budget"]["total"] == 3
    short.write_text(short.read_text() + "optimize: chat\n")
    with pytest.raises(IntentError, match="describe a language model's use"):
        intent.resolve(short, tmp_path / "run2")
    short.write_text("model: m.py\nlengths: 3\n")
    with pytest.raises(IntentError, match="unknown field"):
        intent.resolve(short, tmp_path / "run3")


def test_init_writes_backcountry_yaml_from_flags_and_never_over_a_manifest(tmp_path, monkeypatch, capsys):
    from backcountry.harness.cli import main
    monkeypatch.chdir(tmp_path)
    folder = _model_folder(tmp_path)
    assert main(["init", str(folder), "--yes", "--optimize", "prompts", "--attempts", "5"]) == 0
    written = yaml.safe_load((tmp_path / "backcountry.yaml").read_text())
    assert written["optimize"] == "prompts" and written["attempts"] == 5
    assert "Will make" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["init", "backcountry.yaml", "--yes"])  # already a backcountry.yaml: check and start it
    assert main(["init", str(folder), "--questions"]) == 0
    assert json.loads(capsys.readouterr().out)["questions"][0]["id"] == "optimize"


def test_check_text_says_the_verdict_in_the_users_terms():
    from types import SimpleNamespace as NS
    from backcountry.harness.cli import check_text
    report = NS(constants={"memory_fit": {"gpu_limit_gb": 19.07, "peaks_gb": {"prompt": 18.2, "reply": 16.0},
                                          "untimed": ["prompt@T=1833"]},
                           "pacing": {"full_speed_ms": [104.3, 104.6]}},
                coverage={"step_floor": {"prompt": {"room": 0.03}, "reply": {"room": 0.04}}, "regions_found": 12},
                step_ms={"prompt": {"before": 27.6}, "reply": {"before": 9.7}})
    args = NS(manifest="runs/x/manifest.yaml", target_shown="backcountry.yaml")
    text = check_text(report, "claude-cli", args)
    assert "may not be worth a run: up to 4% headroom" in text and "prompt@T=1833" in text and "3% headroom" in text
    assert text.splitlines()[-1] == "next: backcountry start backcountry.yaml"
    report.coverage["step_floor"]["prompt"]["room"] = 0.30
    shown = check_text(report, "claude-cli", args)
    assert "up to 30% headroom" in shown and "worth a run" not in shown and "nothing to work on" not in shown
    # Whisper via mlx-whisper: every region unsupported; check said "next: start" and nothing more
    report.coverage.update(regions_found=0, discovery={"unknown_sources": ["model@0"]})
    none = check_text(report, "claude-cli", args)
    assert "none: the kernel search has nothing to work on here (the model reads arrays the recorder " \
           "could not place: model@0); only a faster version of the function itself" in none


def test_a_size_named_inside_an_argument_reads_back_from_the_yaml():
    size = {"messages": 4, "messages[0]['content']": 256, "max_tokens": 64}
    assert yaml.safe_load(intent._flow(size)) == size
    assert intent._flow({"question": 16}) == "{question: 16}"      # plain names stay plain


def test_a_function_with_no_size_questions_checks_as_the_app_called_it(tmp_path):
    # init writes `sizes: - {}` for a function with nothing to size (parakeet-mlx
    # handed an audio file's path); check rejected its own yaml
    assert intent._sizes_from([{}], tmp_path / "backcountry.yaml") == [{}]
    with pytest.raises(intent.IntentError, match="one entry per size"):
        intent._sizes_from([], tmp_path / "backcountry.yaml")
