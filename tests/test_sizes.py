"""The sizes of a marked function, one way for every model: knobs found in a
call's arguments, probed for whether they change the model's work, asked as
lists, ranges or tuples, and the call remade at each chosen size from the
app's own values."""

import mlx.core as mx
import numpy as np
import pytest

from backcountry.harness import size_questions as q
from backcountry.harness import watch
from backcountry_runtime import sizes


class Tokenizer:
    """A stand-in: one token per word."""

    def encode(self, text):
        return [len(word) for word in text.split()]

    def decode(self, ids):
        return " ".join("w" * max(i, 1) for i in ids)


def test_knobs_are_found_the_same_way_in_any_call():
    def forecast(model, history, mask, horizon=24, prompt="a b c", messages=None, image=None, **options):
        pass

    args = (None, np.zeros((16, 336, 8)), mx.ones((16, 336)))
    kwargs = {"messages": [{"role": "user", "content": "a long question"}], "max_tokens": 150}
    found = {k["name"]: k for k in sizes.find(forecast, args, kwargs)}
    assert found["horizon"]["kind"] == "int" and found["max_tokens"]["kind"] == "int"   # **options flattened
    assert found["prompt"]["kind"] == "text" and found["messages"]["kind"] == "length"
    assert found["messages[0]['content']"]["kind"] == "text"                           # the text inside a chat
    tied = found["history axis 1 and mask axis 1"]                                     # arrays move together
    assert tied["kind"] == "axis" and tied["value"] == 336


def test_an_image_passed_as_a_file_path_is_sized_by_its_width_and_height(tmp_path):
    # vision apps pass a path (mlx-vlm's API takes one); a Path was skipped
    # and a str path was sized as text, by its characters
    from pathlib import Path
    from PIL import Image
    Image.new("RGB", (64, 48), (200, 10, 10)).save(tmp_path / "receipt.png")

    def read(image, model=None, note=None):
        pass

    for given in (tmp_path / "receipt.png", str(tmp_path / "receipt.png")):
        knobs = sizes.find(read, (given,), {"note": str(tmp_path / "notes.png")})   # no such file: text
        found = {k["name"]: k for k in knobs}
        assert found["image"]["kind"] == "image" and found["image"]["value"] == [64, 48]
        assert found["note"]["kind"] == "text"
        image = sizes.apply(read, (given,), {}, knobs, {"image": [32, 24]})[0][0]
        assert type(image) is type(given) and Image.open(image).size == (32, 24)
        assert Path(image).parent != tmp_path and Image.open(given).size == (64, 48)   # a copy; the app's file untouched


def test_a_model_passed_to_the_function_is_never_a_size():
    # MLX's Module is a dict: gpt-oss keeps layer_types (strings) among its
    # attributes, and init asked "model['model']['layer_types'][0] (tokens)"
    import mlx.nn as nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_types = ["sliding_attention", "full_attention"]

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    def extract(model, notes, retries=1):
        pass

    names = [k["name"] for k in sizes.find(extract, (Outer(), "some notes here"), {})]
    assert names == ["notes", "retries"]


def test_a_call_is_remade_at_a_size_from_the_apps_own_values():
    def run(history, mask, horizon=24, prompt="one two three", rows=(1, 2, 3), **options):
        pass

    args = (np.arange(16 * 4).reshape(16, 4), mx.ones((16, 4)))
    knobs = sizes.find(run, args, {"steps": 4})
    settings = {"history axis 0 and mask axis 0": 40, "horizon": 48, "prompt": 7, "rows": 5, "steps": 8}
    new_args, new_kwargs = sizes.apply(run, args, {"steps": 4}, knobs, settings, Tokenizer())
    history, mask, horizon, prompt, rows = new_args
    assert history.shape == (40, 4) and mask.shape == (40, 4) and (history[16] == history[0]).all()  # repeated
    assert horizon == 48 and rows == (1, 2, 3, 1, 2) and new_kwargs == {"steps": 8}
    assert len(Tokenizer().encode(prompt)) == 7                                        # tokens, with a tokenizer
    assert sizes.tokenizer_for(run, args, {"t": Tokenizer()}) is not None


def test_answers_are_lists_ranges_or_tuples_and_combine_into_a_handful_of_sizes():
    width = {"name": "width", "kind": "int", "value": 1024}
    height = {"name": "height", "kind": "int", "value": 1024}
    rows = {"name": "rows", "kind": "length", "value": 32}
    assert q.spread(32, 4096) == [32, 108, 362, 1218, 4096]
    assert q.parse([rows], "1, 8, 32") == [{"rows": 1}, {"rows": 8}, {"rows": 32}]
    assert len(q.parse([rows], "32-4096")) == 5
    assert q.parse([width, height], "512x512, 768x1344") == [{"width": 512, "height": 512},
                                                              {"width": 768, "height": 1344}]
    image = {"name": "photo", "kind": "image", "value": [1280, 960]}
    assert q.parse([image], "640x480") == [{"photo": [640, 480]}] and q.default([image]) == "1280x960"
    with pytest.raises(q.SizeError, match="2 sizes joined by x"):
        q.parse([width, height], "512")
    with pytest.raises(q.SizeError, match="small to large"):
        q.parse([rows], "64-8")
    steps = {"name": "steps", "kind": "int", "value": 4}
    chosen = q.sizes([[width, height], [rows]], [steps], {"1": "512x512, 1024x1024", "2": "1, 8", "3": "8"})
    assert len(chosen) == 4 and chosen[0] == {"width": 512, "height": 512, "rows": 1, "steps": 8}
    with pytest.raises(q.SizeError, match="that is 15 sizes"):
        q.sizes([[rows], [width]], [], {"1": "32-4096", "2": "1, 2, 3"})
    assert q.plan(chosen) == "width: 512, 1024; height: 512, 1024; rows: 1, 8; steps: 8"
    for knob, parts in ((width, ["DiT#0"]), (height, ["DiT#0"]), (rows, ["DiT#0"])):
        knob.update(effect="size", parts=parts, signature=[[knob["name"], 1]])
    assert q.plan(chosen, [width, height, rows, steps]) == "width x height: 512x512, 1024x1024; rows: 1, 8; steps: 8"


def test_each_question_defaults_to_the_sizes_the_app_called_with():
    rows = {"name": "rows", "kind": "length", "value": 32}
    width = {"name": "width", "kind": "int", "value": 512}
    height = {"name": "height", "kind": "int", "value": 512}
    calls = [{"rows": 32, "width": 512, "height": 512}, {"rows": 4, "width": 1024, "height": 768},
             {"rows": 32, "width": 512, "height": 512}]
    assert q.defaults([[rows], [width, height]], calls) == ["4, 32", "512x512, 1024x768"]
    assert q.defaults([[rows]]) == ["32"]                                  # no record: the probed call's
    many = [{"rows": n, "width": n} for n in (1, 2, 3, 4)]
    trimmed = q.defaults([[rows], [width]], many)                          # 16 combinations: trimmed to 8 or fewer
    assert trimmed == ["1, 2, 3, 4", "1, 4"]
    assert len(q.sizes([[rows], [width]], [], {"1": trimmed[0], "2": trimmed[1]})) == 8
    assert q.shown([512, 768]) == "512x768" and q.shown(5, "tokens") == "5 tokens"


def test_knobs_that_change_the_work_the_same_way_are_asked_together():
    knobs = [{"name": "width", "kind": "int", "value": 1024, "effect": "size", "parts": ["DiT#0", "VAE#0"],
              "signature": [["VAE#0 ((1, 64, 32, 3),)", 1]]},   # a wide image and a tall one: other shapes,
             {"name": "height", "kind": "int", "value": 1024, "effect": "size", "parts": ["DiT#0", "VAE#0"],
              "signature": [["VAE#0 ((1, 32, 64, 3),)", 1]]},   # the same parts: asked together
             {"name": "rows", "kind": "length", "value": 8, "effect": "size", "parts": ["DiT#0"],
              "signature": [["DiT#0 ((4, 64),)", 1]]},
             {"name": "steps", "kind": "int", "value": 4, "effect": "more"},
             {"name": "seed", "kind": "int", "value": 0, "effect": "same"},
             {"name": "x axis 2", "kind": "axis", "value": 8, "effect": "fails"}]
    groups, more, unchanged = q.sort(knobs)
    assert [[k["name"] for k in g] for g in groups] == [["width", "height"], ["rows"]]
    assert q.label(groups[0]) == "width and height (together)"
    text = q.describe(groups, more, unchanged)
    assert "1. width and height (together)" in text and "2. rows (count)" in text and "3. steps" in text
    assert "don't change the work:\n  seed" in text and "can't change" in text and "x axis 2" in text


APP = '''
import mlx.core as mx
import mlx.nn as nn
import backcountry


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.out = nn.Linear(16, 64)

    def __call__(self, tokens):
        return self.out(self.embed(tokens))


@backcountry.tune
def reply(model, prompt, steps=4, seed=0):
    tokens = mx.array([[ord(c) % 64 for c in prompt]])
    out = []
    for _ in range(steps):
        logits = model(tokens)
        out.append(mx.argmax(logits[:, -1], axis=-1))
    return mx.stack(out)


mx.random.seed(0)
model = Net()
print(reply(model, "hello there"))
'''


def test_init_probes_each_knob_and_replays_the_chosen_sizes(tmp_path):
    script = tmp_path / "app.py"
    script.write_text(APP)
    recording = watch.record(script, tmp_path / ".backcountry")
    knobs = {k["name"]: k for k in recording["marked"][0]["knobs"]}
    assert (knobs["prompt"]["effect"], knobs["prompt"]["unit"]) == ("size", "characters")  # no tokenizer here
    assert knobs["steps"]["effect"] == "more" and knobs["seed"]["effect"] == "same"
    assert ["prompt", 11, "characters"] in recording["marked"][0]["sizes"]       # each call's sizes, for defaults
    assert "prompt=11 characters" in watch.describe_marked(recording["marked"][0])
    groups, more, _ = q.sort(list(knobs.values()))
    points = {"knobs": [k for g in groups for k in g] + more,
              "sizes": q.sizes(groups, more, {"1": "4, 40", "2": "2"})}
    model_file = watch.write_marked_model_file(recording, ("__main__", "reply"), 0, tmp_path, points)
    assert [ran for ran, _ in watch.run_sizes(model_file)] == [True, True]
    namespace = {"__file__": str(model_file)}   # the app is found from the model file's folder
    exec(model_file.read_text(), namespace)
    request = namespace["build"]()
    assert [len(call[0][1]) for call in request._calls] == [4, 40]   # the app's prompt cut and repeated
    assert request(mx.zeros((2,)))[0].shape == (2, 1)                 # two steps, as chosen
