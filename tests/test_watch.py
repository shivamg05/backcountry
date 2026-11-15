"""init watches the user's own script: the parts of the model it calls (any
library or the user's own code), their inputs and share of the time, and a
model file that takes the chosen part from the script itself."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from backcountry.harness import watch
from backcountry.harness.intent import IntentError, classify
from backcountry_runtime.watch import Recorder, capture, positional, run_script

APP = '''
import mlx.core as mx
import mlx.nn as nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def __call__(self, x):
        return nn.relu(self.linear(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [Block(), Block()]

    def __call__(self, x, scale=None):
        for block in self.blocks:
            x = block(x)
        return x if scale is None else x * scale


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(16, 2)

    def __call__(self, x, training):
        return self.out(x)


class Cached(nn.Module):
    def __call__(self, x, cache):
        return x + cache[0]


step = mx.compile(lambda x: x + 1)   # compiled by the app: watched eagerly
mx.random.seed(0)
net, head, cached = Net(), Head(), Cached()
for rows in (4, 8, 8):
    h = net(step(mx.random.normal((rows, 16))))
    head(h, rows == 4)
    cached(h, [mx.zeros((1,))])
'''


def _app(tmp_path):
    script = tmp_path / "app.py"
    script.write_text(APP)
    return script


def test_the_recorder_sees_outermost_calls_their_inputs_and_puts_everything_back(tmp_path):
    compile_before, init_before = mx.compile, nn.Module.__init__
    recorder = Recorder()
    run_script(str(_app(tmp_path)), [], recorder)
    assert mx.compile is compile_before and nn.Module.__init__ is init_before
    parts = {p.qualname: p for p in recorder.parts.values()}
    assert set(parts) == {"Net", "Head", "Cached"}  # the blocks inside Net are not parts of their own
    net = parts["Net"]
    assert net.calls == 3 and sorted(net.shapes.values()) == [1, 2] and net.unsupported is None
    assert net.inputs[((("x", (8, 16), "float32"),), "{}")][0].shape == (8, 16)
    assert parts["Head"].unsupported is None and sorted(key[1] for key in parts["Head"].shapes) == [
        "{'training': False}", "{'training': True}"]
    assert "'cache' is a list" in parts["Cached"].unsupported


def test_positional_binds_keywords_in_the_signatures_order():
    class M(nn.Module):
        def __call__(self, x, y=None, flag=False):
            return x

    m, x = M(), mx.zeros((2,))
    assert list(positional(M.__call__, m, (x,), {"y": x})[0]) == ["x", "y"]
    assert positional(M.__call__, m, (x, None), {"flag": (1, "a")})[1] == {"y": None, "flag": (1, "a")}
    assert "'y' is a dict" in positional(M.__call__, m, (x, {}), {})[2]


def test_capture_runs_the_script_until_the_part_is_called_and_returns_it(tmp_path):
    script = str(_app(tmp_path))
    net = capture(script, [], ("__main__", "Net", 0))
    assert type(net).__name__ == "Net" and len(net.blocks) == 2
    with pytest.raises(RuntimeError, match="finished without calling"):
        capture(script, [], ("__main__", "Net", 1))


def test_init_on_a_script_writes_a_model_file_from_what_it_saw(tmp_path, monkeypatch):
    script = _app(tmp_path)
    assert classify(str(script)).kind == "app_script"
    recording = watch.record(script, tmp_path / ".backcountry")
    net = next(p for p in recording["parts"] if p["qualname"] == "Net")
    assert watch.choosable(net) and not watch.choosable(next(p for p in recording["parts"] if p["qualname"] == "Cached"))
    assert "2 input sizes" in watch.describe(net)
    model_file = watch.write_model_file(recording, net, 1, tmp_path, tmp_path / ".backcountry")
    namespace = {}
    exec(model_file.read_text(), namespace)
    examples = namespace["examples"]()
    assert examples[0][0].shape[0] == net["sizes"][1]["inputs"][0]["shape"][0]  # the chosen size goes first
    assert type(namespace["build"]()).__name__ == "Net" and "run" not in namespace


def test_a_part_called_with_plain_options_gets_a_run_that_passes_them(tmp_path):
    recording = watch.record(_app(tmp_path), tmp_path / ".backcountry")
    head = next(p for p in recording["parts"] if p["qualname"] == "Head")
    assert watch.choosable(head) and "called with {'training': False} or {'training': True}" in watch.describe(head)
    false = next(i for i, size in enumerate(head["sizes"]) if size["options"] == "{'training': False}")
    model_file = watch.write_model_file(recording, head, false, tmp_path, tmp_path / ".backcountry")
    namespace = {}
    exec(model_file.read_text(), namespace)
    assert "1 recorded size called with other options is left out" in model_file.read_text()
    (example,) = namespace["examples"]()  # only the calls made with the chosen options
    assert example[0].shape == (8, 16) and namespace["OPTIONS"] == {"training": False}
    assert namespace["run"](namespace["build"](), *example).shape == (8, 2)


def test_init_on_a_script_end_to_end(tmp_path, monkeypatch, capsys):
    from backcountry.harness.cli import main
    import yaml
    script = _app(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(script), "--yes", "--attempts", "3"]) == 0
    written = yaml.safe_load((tmp_path / "backcountry.yaml").read_text())
    assert written["model"].endswith(watch.MODEL_FILE) and written["attempts"] == 3
    out = capsys.readouterr().out
    assert "Net (__main__)" in out and "cannot be timed alone" in out


def test_a_script_that_runs_no_model_says_what_to_do(tmp_path):
    from backcountry.harness.intent import init
    script = tmp_path / "app.py"
    script.write_text("print('no model here')\n")
    with pytest.raises(IntentError, match="called no part of an MLX model"):
        init(str(script), {}, tmp_path / "backcountry.yaml", interactive=False)


def test_a_language_model_seen_in_a_script_gets_its_lengths_as_defaults():
    part = {"sizes": [{"inputs": [{"shape": [1, 148]}], "calls": 1}, {"inputs": [{"shape": [1, 248]}], "calls": 1},
                      {"inputs": [{"shape": [1, 1]}], "calls": 50}]}
    assert watch.language_model_defaults(part) == {"optimize": "chat", "optimize_at": 248, "prompt_tokens": "148-248",
                                                   "history_tokens": 248, "reply_tokens": 25}
    one = {"sizes": [{"inputs": [{"shape": [1, 18]}], "calls": 1}, {"inputs": [{"shape": [1, 1]}], "calls": 511}]}
    defaults = watch.language_model_defaults(one)   # one short prompt: the range reaches down to it
    assert defaults["optimize_at"] == 18 and defaults["prompt_tokens"] == "18-2049"


def test_a_pytorch_script_is_told_before_it_runs(tmp_path):
    script = tmp_path / "run_qwen.py"
    script.write_text("from transformers import AutoModelForCausalLM\nraise SystemExit('downloading 16 GB')\n")
    with pytest.raises(ValueError, match="PyTorch \\(transformers\\).*mlx-community"):
        watch.record(script, tmp_path / "rec")
    script.write_text("import torch\nfrom mflux.models import Flux\n")
    assert watch.not_mlx(script) is None   # MLX's family alongside: run it


def test_a_script_that_hangs_waits_for_input_or_crashes_is_told_what_to_change(tmp_path):
    script = tmp_path / "app.py"
    script.write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(ValueError, match="without ending.*a server.*backcountry init app.py --ask"):
        watch.record(script, tmp_path / "rec", timeout_s=1)
    script.write_text("input('image? ')\n")
    with pytest.raises(ValueError, match="asked for typed input.*after --"):
        watch.record(script, tmp_path / "rec")
    script.write_text("raise SystemExit('no weights at weights/')\n")
    with pytest.raises(ValueError, match="(?s)stopped with an error.*no weights at weights/"):
        watch.record(script, tmp_path / "rec")


def test_a_script_that_ends_with_sys_exit_zero_ran_to_its_end(tmp_path):
    # `sys.exit(main())` with main() returning 0 is a clean end: it was
    # reported as "stopped with an error (exit code 0)" and nothing was recorded
    script = tmp_path / "app.py"
    script.write_text(APP + "\nimport sys\n\n\ndef main():\n    return 0\n\n\nsys.exit(main())\n")
    recording = watch.record(script, tmp_path / "rec")
    assert {part["qualname"] for part in recording["parts"]} >= {"Net", "Head"}
    script.write_text(APP + "\nraise SystemExit(3)\n")
    with pytest.raises(ValueError, match="stopped with an error \\(exit code 3\\)"):
        watch.record(script, tmp_path / "rec")


def test_a_function_that_fails_at_every_size_is_told_to_mark_one_that_takes_plain_data(monkeypatch):
    # a server's model-thread function handed a callback into its event loop
    # failed at the app's own size too: no size helps, the choice of function does
    from backcountry.harness.intent import IntentError, _check_sizes
    monkeypatch.setattr(watch, "run_sizes", lambda _file: [(False, "RuntimeError: Event loop is closed")])
    with pytest.raises(IntentError, match="(?s)Event loop is closed.*takes plain data.*a callback"):
        _check_sizes(Path("model.py"), {"sizes": [{"command": 55}], "knobs": []})


def test_a_scripts_own_arguments_come_after_a_double_dash(tmp_path, monkeypatch):
    from backcountry.harness.cli import main
    script = tmp_path / "app.py"
    script.write_text(APP.replace("for rows in (4, 8, 8):", "import sys\nfor rows in [int(sys.argv[1])]:"))
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(script), "--yes", "--part", "1", "--", "6"]) == 0
    namespace = {}
    exec((tmp_path / ".backcountry" / watch.MODEL_FILE).read_text(), namespace)
    assert namespace["ARGS"] == ["6"] and namespace["examples"]()[0][0].shape == (6, 16)
    with pytest.raises(SystemExit):
        main(["init", "org/name", "--yes", "--", "6"])
