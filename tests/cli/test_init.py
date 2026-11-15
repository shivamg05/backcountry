"""`backcountry init` on an app with its function marked, as a person sees it
in a terminal and as an agent runs it with flags: the calls listed by their
sizes, the questions in the units the model sees, Enter taking the sizes the
app used, wrong answers explained and asked again, and failures said plainly."""

import json

import yaml

from .apps import BROKEN, CHAT, CLASSIFY, FORECAST, IMAGE, NEVER_CALLED, SERVER, SHORT_CONTEXT, TORCH
from .conftest import backcountry

ENTER = [""] * 10


def plain(out: str) -> None:
    """What a person reads: no traces of the model's insides, no crash dumps,
    every question with its default written out."""
    for trace in ("ndarray", "float32", "mlx.core", "array(", "#0", "Traceback"):
        assert trace not in out, f"{trace!r} in the output:\n{out}"
    for line in out.splitlines():
        if line.rstrip().endswith("]:") and "[Y/n]" not in line:
            assert "[default: " in line, line


def written(folder) -> dict:
    """backcountry.yaml beside the app, init's own files hidden in .backcountry."""
    assert sorted(p.name for p in folder.iterdir()) == [".backcountry", "app.py", "backcountry.yaml"]
    assert "*" in (folder / ".backcountry" / ".gitignore").read_text().split()
    return yaml.safe_load((folder / "backcountry.yaml").read_text())


def test_a_chat_reply_is_asked_in_tokens_and_enter_takes_the_questions_the_app_asked(app):
    folder = app(CHAT)
    done = backcountry("init", "app.py", cwd=folder, answers=ENTER)
    assert done.code == 0, done
    plain(done.out)
    assert "reply (marked with @backcountry.tune) was called 2 times" in done.out
    assert "question=3 tokens" in done.out and "question=11 tokens" in done.out
    assert "These change how much work the model does:\n  1. question (tokens)" in done.out
    assert "These run the model more or fewer times (one value, used for every size):\n  2. max_tokens" in done.out
    assert "[default: 3, 11]:" in done.out and "max_tokens: the value to use for every size [default: 4]:" in done.out
    assert "checking 2 sizes ...\n  all 2 ran" in done.out
    assert "Will make reply faster at 2 sizes (question: 3, 11 tokens; max_tokens: 4)" in done.out
    assert written(folder)["attempts"] == 100


def test_an_images_width_and_height_are_one_question_answered_in_pairs(app):
    folder = app(IMAGE)
    done = backcountry("init", "app.py", cwd=folder, answers=["32x32, 64x48"] + ENTER)
    assert done.code == 0, done
    plain(done.out)
    assert "1. width and height (together)" in done.out and "2. steps" in done.out
    assert "These don't change the work:\n  prompt, seed" in done.out
    assert "Sizes for width and height (together): a list like 512x512, 1024x768 [default: 64x64]:" in done.out
    assert "width x height: 32x32, 64x48; steps: 3" in done.out
    written(folder)


def test_a_batch_of_images_is_asked_as_a_count(app):
    folder = app(CLASSIFY)
    done = backcountry("init", "app.py", cwd=folder, answers=ENTER)
    assert done.code == 0, done
    plain(done.out)
    assert "images=8" in done.out and "images=2" in done.out
    assert "1. images (count)" in done.out and "[default: 2, 8]:" in done.out
    written(folder)


def test_an_agent_reads_the_questions_as_json_and_answers_them_by_number(app):
    folder = app(FORECAST)
    asked = backcountry("init", "app.py", "--questions", cwd=folder)
    assert asked.code == 0, asked
    report = json.loads(asked.out[asked.out.index("\n{") + 1:])
    assert report["calls"]["forecast"] == [c for c in report["calls"]["forecast"] if "history axis 0=" in c]
    questions = {q["knobs"][0]: q for q in report["sizes"]["forecast"]}
    series = questions["history axis 0"]
    assert series["default"] == "4, 16"                       # the app's own calls: 16 series, then 4
    assert not (folder / "backcountry.yaml").exists()         # --questions writes nothing
    done = backcountry("init", "app.py", "--yes", "--sizes", f"{series['id']}=2, 8", "--attempts", "12", cwd=folder)
    assert done.code == 0, done
    plain(done.out)
    assert "history axis 0: 2, 8" in done.out and "checking 2 sizes" in done.out
    assert written(folder)["attempts"] == 12


def test_a_wrong_answer_is_explained_and_asked_again(app):
    folder = app(CHAT)
    done = backcountry("init", "app.py", cwd=folder, answers=["lots", "64-8", "4, 8"] + ENTER)
    assert done.code == 0, done
    assert "'lots' is not a size; give a list like 1, 8, 32 or a range like 32-4096" in done.out
    assert "a range goes from small to large; got 64-8" in done.out
    assert "question: 4, 8 tokens" in done.out


def test_more_sizes_than_a_run_can_check_is_asked_again(app):
    folder = app(IMAGE)
    many = ", ".join(f"{8 * i}x{8 * i}" for i in range(1, 10))
    done = backcountry("init", "app.py", cwd=folder, answers=[many, "16x16, 32x32"] + ENTER)
    assert done.code == 0, done
    assert "that is 9 sizes" in done.out and "give at most 8" in done.out
    assert "width x height: 16x16, 32x32" in done.out


def test_a_size_the_model_cannot_take_is_said_before_anything_is_written(app):
    folder = app(SHORT_CONTEXT)
    done = backcountry("init", "app.py", cwd=folder, answers=["4, 64"] + ENTER)
    assert done.code != 0
    assert "1 of 2 sizes did not run; first: question=64 tokens" in done.out
    assert "live parts of the running app" not in done.out   # a size it cannot take, not a function it cannot replay
    assert not (folder / "backcountry.yaml").exists()
    assert not (folder / ".backcountry" / "backcountry_model.py").exists()
    # with a project already set up, a failed init leaves its pair as it was: the
    # new model file beside the old yaml was a project check then rejected
    assert backcountry("init", "app.py", cwd=folder, answers=["4"] + ENTER).code == 0
    pair = [(folder / name).read_bytes() for name in ("backcountry.yaml", ".backcountry/backcountry_model.py")]
    assert backcountry("init", "app.py", cwd=folder, answers=["4, 64"] + ENTER).code != 0
    assert [(folder / name).read_bytes() for name in ("backcountry.yaml", ".backcountry/backcountry_model.py")] == pair


def test_a_marked_function_the_app_never_calls_is_named(app):
    done = backcountry("init", "app.py", cwd=app(NEVER_CALLED))
    assert done.code != 0 and "reply needs question: give it with --request \"question=...\"" in done.out
    plain(done.out)
    given = backcountry("init", "app.py", "--yes", "--request", "question=what is mlx", cwd=app(NEVER_CALLED))
    assert "never called by app.py: it runs on the request given with --request" in given.out


def test_an_app_that_makes_no_request_is_given_one_typed_at_the_prompt(app):
    folder = app(NEVER_CALLED)
    done = backcountry("init", "app.py", cwd=folder, answers=["what is mlx"] + ENTER)
    assert "reply (marked with @backcountry.tune) is never called by app.py: Give it one request" in done.out
    assert "question [default: none: required]: what is mlx" in done.out
    assert "1. question=text (11 characters)" in done.out    # made: this app's reply runs no model
    assert "reply ran no MLX model backcountry can reach" in done.out


def test_a_server_is_loaded_without_its_loop_and_asked_for_a_request(app):
    folder = app(SERVER)
    done = backcountry("init", "app.py", "--ask", cwd=folder, answers=["what is mlx and why is it fast"] + ENTER)
    assert done.code == 0, done
    plain(done.out)
    assert 'never called by app.py outside its `if __name__ == "__main__":` block' in done.out
    assert "1. question=8 tokens, max_tokens=4" in done.out          # the typed request, counted in tokens
    assert "Sizes for question (tokens): a list like 1, 8, 32 or a range like 32-4096 [default: 8]:" in done.out
    model_file = (folder / ".backcountry" / "backcountry_model.py").read_text()
    assert "REQUEST = {'question': 'what is mlx and why is it fast'}" in model_file and "MAIN = False" in model_file
    assert "SCRIPT = 'app.py'" in model_file                   # from the project's folder: it can move
    intent = yaml.safe_load((folder / "backcountry.yaml").read_text())
    assert intent["model"] == ".backcountry/backcountry_model.py"
    assert intent["sizes"] == [{"question": 8, "max_tokens": 4}]
    assert intent["request"] == {"question": "what is mlx and why is it fast"}
    assert "!backcountry_model.py" in (folder / ".backcountry" / ".gitignore").read_text().split()


def test_an_agent_reads_the_request_question_and_answers_it_with_a_flag(app):
    folder = app(SERVER)
    asked = backcountry("init", "app.py", "--ask", "--questions", cwd=folder)
    assert asked.code == 0, asked
    questions = json.loads(asked.out[asked.out.index("{"):])
    assert [a["name"] for a in questions["request"]["arguments"]] == ["question", "max_tokens"]
    given = backcountry("init", "app.py", "--ask", "--questions", "--request", "question=what is mlx", cwd=folder)
    assert given.code == 0, given                    # the request given: the questions after it
    after = json.loads(given.out[given.out.index("{"):])
    assert "request" not in after and after["sizes"]["reply"], after
    done = backcountry("init", "app.py", "--ask", "--yes", "--request", "question=what is mlx; max_tokens=6",
                       "--sizes", "1=8-64", cwd=folder)
    assert done.code == 0, done
    assert "1. question=3 tokens, max_tokens=6" in done.out
    assert "at 5 sizes (question: 8, 13, 23, 38, 64 tokens; max_tokens: 6)" in done.out


def test_a_script_that_does_not_run_says_where(app):
    done = backcountry("init", "app.py", cwd=app(BROKEN))
    assert done.code != 0 and "app.py" in done.out and "SyntaxError" in done.out
    assert "backcountry.harness" not in done.out          # the user's error, not a dump of backcountry's insides


def test_a_pytorch_script_is_told_backcountry_runs_mlx(app):
    done = backcountry("init", "app.py", cwd=app(TORCH))
    assert done.code != 0 and "runs its model with PyTorch" in done.out
    plain(done.out)
