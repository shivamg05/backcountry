"""@backcountry.tune: the app's own request function, marked where it is
defined, recorded by init and replayed by the search with the arguments the
app gave it."""

import copy
import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from backcountry.harness import watch
from backcountry_runtime import tune as tune_mod
from backcountry_runtime.swap import copy_model

APP = '''
import mlx.core as mx
import mlx.nn as nn
import backcountry


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.out = nn.Linear(16, 32)

    def __call__(self, tokens, cache=None):
        h = self.embed(tokens)
        if cache is not None:
            h = h + cache[-1]
        return self.out(h), [*(cache or []), h[:, -1:]]


class Kit:
    def __init__(self, model):
        self.model = model


@backcountry.tune
def reply(kit, prompt, history, steps=4):
    """A user's loop: text in, a cache it keeps itself, an early stop, text out."""
    tokens = mx.array([[ord(c) % 32 for c in prompt]])
    logits, cache = kit.model(tokens)
    out = []
    for _ in range(steps):
        token = mx.argmax(logits[:, -1], axis=-1)
        if token.item() == 31:
            break
        out.append(chr(65 + token.item()))
        logits, cache = kit.model(token[:, None], cache=cache)
    history.append(prompt)
    print("reply:", "".join(out))
    return "".join(out)


mx.random.seed(0)
kit = Kit(Net())
history = []
for prompt in ("hi", "a longer prompt", "mid size"):
    reply(kit, prompt, history)
'''


def _app(tmp_path):
    script = tmp_path / "app.py"
    script.write_text(APP)
    return script


def test_the_mark_changes_nothing_when_backcountry_is_not_running():
    @tune_mod.tune
    def double(x):
        return x * 2

    assert double(3) == 6 and double.__backcountry_tune__(4) == 8


def test_locate_finds_the_model_in_arguments_objects_or_globals():
    model = nn.Linear(2, 2)

    class Kit:
        pass

    kit = Kit()
    kit.model = model

    def uses_argument(kit, prompt):
        return kit

    assert tune_mod.locate(model, uses_argument, (kit, "p"), {}) == (("arg", 0), [("attr", "model")])
    assert tune_mod.resolve((("arg", 0), [("attr", "model")]), uses_argument, (kit, "p"), {}) is model
    namespace = {"MODEL": {"net": [model]}}
    exec("def uses_global(x):\n    return MODEL['net'][0](x)\n", namespace)
    location = tune_mod.locate(model, namespace["uses_global"], (mx.zeros((2,)),), {})
    assert location == (("global", "MODEL"), [("item", "net"), ("item", 0)])


def test_a_result_becomes_arrays_text_as_bytes():
    out = tune_mod.arrays_of({"text": "hé", "n": 3, "logits": mx.ones((2,)), "skip": None})
    assert out[0].tolist() == list("hé".encode()) and out[1].item() == 3 and out[2].shape == (2,)

    def tokens():
        yield mx.array(1)
        yield "x"
    assert len(tune_mod.arrays_of(tokens())) == 2
    import numpy as np
    assert tune_mod.describe_value(np.zeros((4, 3), np.float32)) == "ndarray [4, 3] float32"
    from PIL import Image
    picture = Image.fromarray(np.zeros((4, 6, 3), np.uint8))   # PIL exposes __array_interface__, not __array__
    assert tune_mod.arrays_of(picture)[0].shape == (4, 6, 3)

    class Generated:   # a library's result object (mflux's GeneratedImage): its image, not its other fields
        def __init__(self):
            self.image, self.seconds, self.prompt, self.config = picture, 41.7, "a fox", object()
    assert [a.shape for a in tune_mod.arrays_of(Generated())] == [(4, 6, 3)]


def test_init_records_the_marked_calls_and_the_search_replays_them(tmp_path):
    recording = watch.record(_app(tmp_path), tmp_path / ".backcountry")
    calls = watch.marked_functions(recording)[("__main__", "reply")]
    assert len(calls) == 3 and "prompt=text (15 characters)" in calls[1]["arguments"]
    assert calls[1]["parts"] == [{"class": "__main__.Net", "location": [["arg", 0], [["attr", "model"]]]}]
    model_file = watch.write_marked_model_file(recording, ("__main__", "reply"), 1, tmp_path)
    namespace = {"__file__": str(model_file)}   # the app is found from the model file's folder
    exec(model_file.read_text(), namespace)
    assert namespace["CALLS"] == [1, 0, 2]   # the chosen call first
    request = namespace["build"]()
    examples = namespace["examples"]()
    first, again = request(examples[0]), request(examples[0])
    assert bytes(first[0].tolist()).decode() == bytes(again[0].tolist()).decode()  # the same reply every replay
    assert request._calls[0][0][2] == ["hi"]   # the history the app had passed, not what replays appended
    assert isinstance(request.parts[0], nn.Module)


def test_release_frees_the_apps_model_once_the_job_is_done(tmp_path):
    """The job's fresh-process check loads the model again: the app's globals
    (its model) must not stay alive through the marked function backcountry ran."""
    import gc
    import weakref
    recording = watch.record(_app(tmp_path), tmp_path / ".backcountry")
    model_file = watch.write_marked_model_file(recording, ("__main__", "reply"), 0, tmp_path)
    namespace = {"__file__": str(model_file)}
    exec(model_file.read_text(), namespace)
    request = namespace["build"]()
    model = weakref.ref(request.parts[0])
    del request
    gc.collect()
    assert model() is not None          # still held: the app's function, and its globals
    namespace["release"]()
    gc.collect()
    assert model() is None


def test_a_copy_of_the_request_is_independent_when_the_model_is_an_argument(tmp_path):
    recording = watch.record(_app(tmp_path), tmp_path / ".backcountry")
    model_file = watch.write_marked_model_file(recording, ("__main__", "reply"), 0, tmp_path)
    namespace = {"__file__": str(model_file)}   # the app is found from the model file's folder
    exec(model_file.read_text(), namespace)
    request = namespace["build"]()
    copied = copy_model(request)
    assert copied.parts[0] is not request.parts[0]
    assert copied._calls[0][0][0].model is copied.parts[0]   # its own kit holds its own model
    assert copied.parts[0].out.weight is request.parts[0].out.weight   # weights shared, not duplicated
    which = mx.zeros((1,))
    assert copied(which)[0].tolist() == request(which)[0].tolist()


def test_a_copy_reaching_its_model_through_globals_runs_its_own_and_puts_the_global_back():
    namespace = {"mx": mx}
    exec("def predict(x):\n    return MODEL(x)\n", namespace)
    original = nn.Linear(2, 2)
    namespace["MODEL"] = original
    request = tune_mod.Request(namespace["predict"], [((mx.ones((1, 2)),), {})], [original],
                               [(("global", "MODEL"), [])])
    copied = copy_model(request)
    seen = []
    copied.parts[0].__class__ = type("Seen", (nn.Linear,), {"__call__": lambda self, x: seen.append(1) or
                                                            nn.Linear.__call__(self, x)})
    copied(mx.zeros((1,)))
    assert seen == [1] and namespace["MODEL"] is original   # its own model ran; the app's is back


def test_a_model_a_library_keeps_itself_is_found_and_swapped(tmp_path, monkeypatch):
    import sys
    import types
    library = types.ModuleType("fakewhisper")
    exec("class Holder:\n    model = None\n\ndef transcribe(x):\n    if Holder.model is None:\n"
         "        import mlx.nn as nn\n        Holder.model = nn.Linear(2, 2)\n    return Holder.model(x)\n",
         library.__dict__)
    monkeypatch.setitem(sys.modules, "fakewhisper", library)
    namespace = {"fakewhisper": library}
    exec("def run(x):\n    return fakewhisper.transcribe(x)\n", namespace)
    namespace["run"](mx.ones((1, 2)))
    model = library.Holder.model
    assert tune_mod.locate(model, namespace["run"], (mx.ones((1, 2)),), {}) is None   # not in the app's reach
    location = tune_mod.locate_in_libraries(model, namespace["run"])
    assert location == (("module", "fakewhisper:Holder"), [("attr", "model")])
    request = tune_mod.Request(namespace["run"], [((mx.ones((1, 2)),), {})], [model], [location])
    copied = copy_model(request)
    assert copied.parts[0] is not model
    copied(mx.zeros((1,)))
    assert library.Holder.model is model


LAZY_LIBRARY = '''
import mlx.nn as nn
loads = []


class Holder:   # as mlx-whisper's ModelHolder: the model is made on first use
    model = None

    @classmethod
    def get(cls):
        if cls.model is None:
            loads.append(1)
            cls.model = nn.Linear(2, 2)
        return cls.model


def transcribe(x):
    return Holder.get()(x)
'''


def test_a_model_a_library_loads_on_first_use_is_the_one_the_replays_run(tmp_path, monkeypatch):
    # the rerun stops before the first call, when the library holds no model yet:
    # the job held None, put it back for every replay, and each replay loaded
    # the model again, where no kernel could reach it (Whisper: 0 regions)
    import sys
    (tmp_path / "fakelib.py").write_text(LAZY_LIBRARY)
    (tmp_path / "app.py").write_text("import mlx.core as mx\nimport backcountry\nimport fakelib\n\n\n"
                                     "@backcountry.tune\ndef run(x):\n    return fakelib.transcribe(x)\n\n\n"
                                     "run(mx.ones((1, 2)))\n")
    monkeypatch.delitem(sys.modules, "fakelib", raising=False)
    recording = watch.record(tmp_path / "app.py", tmp_path / ".backcountry")
    model_file = watch.write_marked_model_file(recording, ("__main__", "run"), 0, tmp_path)
    namespace = {"__file__": str(model_file)}
    exec(model_file.read_text(), namespace)
    request = namespace["build"]()
    library = sys.modules["fakelib"]
    assert isinstance(request.parts[0], nn.Module) and request.parts[0] is library.Holder.model
    request(namespace["examples"]()[0])
    request(namespace["examples"]()[0])
    assert len(library.loads) == 1   # loaded once, then the replays run that model


def test_init_on_a_marked_script_end_to_end(tmp_path, monkeypatch, capsys):
    from backcountry.harness.cli import main
    import yaml
    script = _app(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["init", str(script), "--yes", "--size", "2", "--attempts", "3"]) == 0
    out = capsys.readouterr().out
    assert "reply (marked with @backcountry.tune) was called 3 times" in out
    assert yaml.safe_load((tmp_path / "backcountry.yaml").read_text())["model"].endswith(watch.MODEL_FILE)
    assert main(["init", str(script), "--questions"]) == 0          # what an agent asks: the size questions,
    asked = json.loads((lambda o: o[o.index("\n{") + 1:])(capsys.readouterr().out))["sizes"]["reply"]
    prompt = next(q for q in asked if "prompt" in q["ask"])
    assert prompt["default"] == "2, 8, 15"                           # defaulting to the app's own calls


STREAM_APP = """
import mlx.core as mx
import mlx.nn as nn
import backcountry


@backcountry.tune
def stream(model, n):
    x = mx.ones((1, 4))
    for _ in range(n):
        x = model(x)
        yield x.sum().item()


mx.random.seed(0)
model = nn.Linear(4, 4)
for n in (2, 5):
    print(list(stream(model, n)))
"""


def test_a_streaming_function_is_recorded_while_the_app_reads_it_and_replayed_whole(tmp_path):
    script = tmp_path / "app.py"
    script.write_text(STREAM_APP)
    recording = watch.record(script, tmp_path / ".backcountry")
    calls = watch.marked_functions(recording)[("__main__", "stream")]
    assert [c["parts"][0]["location"] for c in calls] == [[["arg", 0], []]] * 2
    model_file = watch.write_marked_model_file(recording, ("__main__", "stream"), 1, tmp_path)
    namespace = {"__file__": str(model_file)}   # the app is found from the model file's folder
    exec(model_file.read_text(), namespace)
    request = namespace["build"]()
    assert len(request(mx.zeros((1,)))) == 5   # the five items it streamed


SERVED = APP.replace('''kit = Kit(Net())
history = []
for prompt in ("hi", "a longer prompt", "mid size"):
    reply(kit, prompt, history)''', '''import concurrent.futures, sys
worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
kit = worker.submit(lambda: Kit(Net())).result()   # loaded on the thread that runs it, as servers do
history = []


def handle(prompt):   # a server: an error in the request fails it, not the app
    try:
        return worker.submit(reply, kit, prompt, history).result()
    except Exception:
        return None


if any(handle(prompt) is None for prompt in ("hi", "a longer prompt", "mid size")):
    sys.exit("a request failed")''')


def test_a_function_a_server_runs_on_its_model_thread_is_replayed(tmp_path):
    # the stop before the last wanted call was an exception the worker thread
    # kept: the server failed the request and the app with it (a FastAPI app
    # running mlx-lm on one thread). There the call runs and the app ends
    script = tmp_path / "app.py"
    script.write_text(SERVED)
    recording = watch.record(script, tmp_path / ".backcountry")
    model_file = watch.write_marked_model_file(recording, ("__main__", "reply"), 2, tmp_path)
    namespace = {"__file__": str(model_file)}
    exec(model_file.read_text(), namespace)
    request = namespace["build"]()
    first, again = request(namespace["examples"]()[0]), request(namespace["examples"]()[0])
    assert bytes(first[0].tolist()) == bytes(again[0].tolist())


VISION = '''
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
import backcountry

model = nn.Linear(3, 4)


@backcountry.tune
def read(image, model):   # a vision app passes its image as a file
    pixels = mx.array(np.asarray(Image.open(image).convert("RGB")), dtype=mx.float32)
    return model(pixels.reshape(-1, 3)).sum(axis=0), mx.array(Image.open(image).size)


Image.new("RGB", (64, 48), (200, 10, 10)).save(Path(__file__).parent / "receipt.png")
read(Path(__file__).parent / "receipt.png", model)
'''


def test_an_image_file_the_function_takes_is_replayed_at_each_chosen_size(tmp_path):
    # the size reached the model file without saying the image was a file, and
    # each replay called .resize() on a path
    script = tmp_path / "app.py"
    script.write_text(VISION)
    recording = watch.record(script, tmp_path / ".backcountry")
    knobs = watch.marked_functions(recording)[("__main__", "read")][0]["knobs"]
    image = next(k for k in knobs if k["name"] == "image")
    points = {"knobs": [image], "sizes": [{"image": [64, 48]}, {"image": [32, 24]}]}
    model_file = watch.write_marked_model_file(recording, ("__main__", "read"), 0, tmp_path, points)
    namespace = {"__file__": str(model_file)}
    exec(model_file.read_text(), namespace)
    request = namespace["build"]()
    assert [request(which)[1].tolist() for which in namespace["examples"]()] == [[64, 48], [32, 24]]


def test_a_rerun_that_calls_with_other_arguments_says_so(tmp_path):
    script = _app(tmp_path)
    recording = watch.record(script, tmp_path / ".backcountry")
    model_file = watch.write_marked_model_file(recording, ("__main__", "reply"), 0, tmp_path)
    script.write_text(APP.replace('("hi", "a longer prompt", "mid size")', '("hey", "a longer prompt", "mid size")'))
    namespace = {"__file__": str(model_file)}   # the app is found from the model file's folder
    exec(model_file.read_text(), namespace)
    with pytest.raises(RuntimeError, match="other arguments than when init watched it.*seed any randomness"):
        namespace["build"]()


def test_a_stored_result_is_for_the_function_where_it_is_defined(tmp_path):
    library = tmp_path / "myapp.py"
    library.write_text("import backcountry\n\n@backcountry.tune\ndef score(x):\n    return x\n")
    server = tmp_path / "server.py"
    server.write_text("from myapp import score\nprint(score(2))\n")
    recording = watch.record(server, tmp_path / ".backcountry")
    call = recording["marked"][0]
    model_file = watch.write_marked_model_file(recording, ("myapp", "score"), 0, tmp_path)
    namespace = {"__file__": str(model_file)}
    exec(model_file.read_text(), namespace)
    import sys
    try:  # a job's build, from another folder, finds the app's own module
        assert (tmp_path.parent / namespace["FILE"]).resolve() == library.resolve()   # from the project's folder
        assert namespace["build"]()(mx.zeros((1,)))[0].item() == 2
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("myapp", None)
    assert call["function"] == ["myapp", "score"] and call["file"] == str(library)
    import importlib.util
    spec = importlib.util.spec_from_file_location("myapp", library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    made_for = {"SCRIPT": str(server), "FUNCTION": ("myapp", "score"), "PARTS": [], "FILE": str(library)}
    assert tune_mod.made_for_function(made_for, module.score.__backcountry_tune__)
    other = tmp_path / "bench.py"
    other.write_text("def score(x):\n    return x\n")
    namespace = {}
    exec(compile(other.read_text(), str(other), "exec"), namespace)
    assert not tune_mod.made_for_function(made_for, namespace["score"])   # same name, another file


def test_a_marked_function_the_script_never_calls_is_asked_a_request_for(tmp_path):
    from backcountry.harness.intent import IntentError, init
    script = tmp_path / "app.py"
    script.write_text("import backcountry\n\n@backcountry.tune\ndef answer(q):\n    return q\n")
    with pytest.raises(IntentError, match='answer needs q: give it with --request "q=..."'):
        init(str(script), {}, tmp_path / "backcountry.yaml", interactive=False)


def test_only_the_apps_model_side_globals_fill_an_argument():
    # a Gradio app's `style = gr.Radio(...)` shares its name with the handler's
    # `style` argument: init passed the radio button where a style name goes
    class Tokenizer:
        def encode(self, text):
            return [1]

        def decode(self, ids):
            return ""

    class Radio:   # a UI component: no model, no tokenizer
        value = "Pirate"

    class Engine:
        def __init__(self):
            self.model = nn.Linear(2, 2)

    namespace = {"model": nn.Linear(2, 2), "tokenizer": Tokenizer(), "style": Radio(), "engine": Engine()}
    exec("def shift_tone(model, tokenizer, engine, text, style):\n    return text\n", namespace)
    filled = {p["name"] for p in tune_mod.parameters(namespace["shift_tone"]) if p["global"]}
    assert filled == {"model", "tokenizer", "engine"}


def test_layers_called_one_by_one_become_the_model_that_holds_them():
    """mflux decodes with its VAE's layers called one at a time: the part is
    the VAE, found through its own tree at any depth."""
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [[nn.Linear(2, 2), nn.Linear(2, 2)]]

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

    class Pipe:
        pass

    pipe = Pipe()
    pipe.vae = VAE()
    deep = pipe.vae.decoder.blocks[0][1]
    namespace = {"pipe": pipe}
    exec("def decode(x):\n    return pipe.vae.decoder.blocks[0][1](x)\n", namespace)
    location = tune_mod.locate(deep, namespace["decode"], (mx.ones((1, 2)),), {})
    assert location == (("global", "pipe"), [("attr", "vae")])


def test_an_engines_iterator_is_a_stream_and_its_objects_are_never_opened():
    """LM Studio's create_generator hands back an iterator object, not a
    generator; and a result must never be walked into the model kit it
    holds (one array per tokenizer string ran Metal out of buffers)."""
    class Kit:
        def __init__(self):
            self.vocab = [f"t{i}" for i in range(100000)]

    class Results:
        def __init__(self):
            self.kit, self.items = Kit(), iter([("hi", 1), ("there", 2)])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.items)

    stream = Results()
    assert tune_mod.streams(stream) and not tune_mod.streams("text")
    assert len(tune_mod.arrays_of(stream)) == 4          # two items of text and a number
    assert tune_mod.arrays_of(Kit()) == []               # an arbitrary object is left out


def test_a_result_is_published_beside_the_function_and_found_there_first(tmp_path, monkeypatch):
    app = tmp_path / "app.py"
    app.write_text("def score(model, rows):\n    return rows\n")
    model_file = tmp_path / ".backcountry" / "backcountry_model.py"
    model_file.parent.mkdir()
    model_file.write_text(f"SCRIPT = {str(app)!r}\nFUNCTION = ('__main__', 'score')\nFILE = {str(app)!r}\n"
                          "PARTS = [[['arg', 0], []]]\n")
    artifact = tmp_path / "runs" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "swap_table.json").write_text("[]")
    published = tune_mod.publish(artifact, model_file)
    assert published == tmp_path / "backcountry-results" / "score" and (published / "swap_table.json").is_file()
    assert tune_mod.publish(artifact, tmp_path / "app.py") is None        # not a marked function's model file

    import backcountry_runtime.apply as apply_mod
    seen = []
    monkeypatch.setattr(apply_mod, "apply", lambda model, path, **kw: seen.append(path) or model)
    monkeypatch.setattr(apply_mod, "artifacts_home", lambda: tmp_path / "empty-store")
    monkeypatch.setattr(tune_mod, "tuned_for", lambda d: {"SCRIPT": str(app), "FUNCTION": ("__main__", "score"),
                                                          "FILE": str(app), "PARTS": [[["arg", 0], []]]})
    namespace = {}
    exec(compile(app.read_text(), str(app), "exec"), namespace)
    assert tune_mod.install_stored(namespace["score"], (nn.Linear(2, 2), 1), {}) == "score"
    assert seen == [published]

    calls = []
    monkeypatch.setattr(tune_mod, "install_stored", lambda *a, **k: calls.append(a))
    off = tune_mod.tune(apply=False)(namespace["score"])
    on = tune_mod.tune(namespace["score"])
    off(None, 1), on(None, 1)
    assert len(calls) == 1   # apply=False keeps the mark and runs it untouched


def test_a_streaming_handler_given_a_typed_request_runs_as_the_app_would_read_it(tmp_path):
    # a Gradio handler yields its text; init made the typed request and never
    # read the stream, so the model never ran ("called no part of an MLX model")
    script = tmp_path / "app.py"
    script.write_text("import mlx.core as mx\nimport mlx.nn as nn\nimport backcountry\n\nmodel = nn.Linear(4, 4)\n\n\n"
                      "@backcountry.tune\ndef shift_tone(text, style):\n"
                      "    for i in range(2):\n        yield model(mx.ones((1, 4)) * len(text)).sum().item()\n\n\n"
                      "if __name__ == '__main__':\n    raise SystemExit('a server loop')\n")
    recording = watch.record(script, tmp_path / ".backcountry", main=False,
                             request={"function": ["__backcountry_app__", "shift_tone"], "inputs": {"text": "hello", "style": "Pirate"}})
    assert recording["marked"] and any(part["qualname"] == "Linear" for part in recording["parts"])


def test_an_empty_closure_cell_is_skipped_when_looking_for_the_model():
    # Gradio's objects hold functions with a closure name never set; reading it
    # raised ValueError and check died
    from backcountry.harness.trace.walk import reachable

    def make():
        def inner():
            return later
        return inner
        later = 1   # never reached: inner's cell stays empty

    holder = {"fn": make(), "w": mx.ones((2,))}
    assert [path for path, value in reachable(holder) if isinstance(value, mx.array)] == ["w"]
