"""`backcountry init my_app.py`: run the user's own script once, in its own
process, and list the parts of the model it called (backcountry_runtime.watch):
how often, at which input sizes, and their share of the time. The chosen part
becomes a model file whose build() runs the script until it first calls that
part and whose examples() are the inputs the script really passed.

    python -m backcountry.harness.watch SCRIPT OUT_DIR [ARGS...]   # the recording process
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RECORDING = "backcountry_recording.json"
INPUTS = "backcountry_inputs.safetensors"
MODEL_FILE = "backcountry_model.py"
REQUEST = "backcountry_request.json"   # a request the user typed at init, for an app that makes none


def not_mlx(script: Path) -> str | None:
    """A note when the script uses PyTorch and nothing from MLX's family (read
    from its imports, never run): backcountry speeds up MLX models only."""
    import ast
    try:
        tree = ast.parse(Path(script).read_text())
    except (OSError, SyntaxError):
        return None
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            tops.add(node.module.split(".")[0])
    if not tops & {"torch", "transformers", "diffusers"} or any("mlx" in t or t == "mflux" for t in tops):
        return None
    return (f"{Path(script).name} runs its model with PyTorch ({', '.join(sorted(tops & {'torch', 'transformers', 'diffusers'}))}), "
            "and backcountry speeds up MLX models on Apple silicon. Use the model's MLX version (for a Hugging Face "
            "model, its mlx-community conversion: `backcountry init mlx-community/<name>`), or run it with MLX")


def record(script: Path, out_dir: Path, argv: list[str] | None = None, timeout_s: float = 1200,
           request: dict | None = None, main: bool = True) -> dict:
    """Run the script once in a new process with the recorder on; returns the
    recording ({"parts": [...], "loads": [...]}) and saves the inputs.
    request: {"function", "inputs"}, a call the user typed for an app that
    makes none, made once the script has run; main=False runs the script
    without its __main__ block (a server's loop)."""
    if (note := not_mlx(script)) is not None:
        raise ValueError(note)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text("# written by backcountry: this machine's recording, never committed;\n"
                                        "# the model file init writes from it goes with backcountry.yaml\n"
                                        f"*\n!{MODEL_FILE}\n")
    name = Path(script).name
    env = dict(os.environ)
    env.pop("BACKCOUNTRY_REQUEST", None)
    env.pop("BACKCOUNTRY_NO_MAIN", None)
    if request is not None:
        (out_dir / REQUEST).write_text(json.dumps(request, indent=1))
        env["BACKCOUNTRY_REQUEST"] = str((out_dir / REQUEST).resolve())
    if not main:
        env["BACKCOUNTRY_NO_MAIN"] = "1"
    try:
        proc = subprocess.run([sys.executable, "-m", "backcountry.harness.watch", str(Path(script).resolve()),
                               str(out_dir.resolve()), *(argv or [])], capture_output=True, text=True,
                              timeout=timeout_s, cwd=Path(script).parent, stdin=subprocess.DEVNULL, env=env)
    except subprocess.TimeoutExpired:
        raise ValueError(f"{name} ran for {timeout_s / 60:.0f} minutes without ending. If it waits for requests (a "
                         f"server), put its serving loop under `if __name__ == \"__main__\":` and run "
                         f"`backcountry init {name} --ask`: init loads it without that block and asks you for a "
                         "request to make") from None
    recording = out_dir / RECORDING
    if proc.returncode != 0 or not recording.is_file():
        if "EOFError" in proc.stderr:
            raise ValueError(f"{name} asked for typed input, which init cannot give: put the inputs in the script, "
                             f"or pass its arguments after -- (backcountry init {name} -- --image cat.png)")
        raise ValueError(f"{name} stopped with an error (exit code {proc.returncode}), so init did not see it run to "
                         f"its end. Its last lines:\n{proc.stderr.strip()[-2000:]}")
    return json.loads(recording.read_text())


def _their_traceback(error: BaseException) -> str:
    """The traceback of an error in the user's script without the frames of
    backcountry and runpy that ran it."""
    import traceback
    src = Path(__file__).resolve().parents[2]   # holds backcountry and backcountry_runtime
    ours = tuple(str(src / package) for package in ("backcountry", "backcountry_runtime"))
    shown = traceback.TracebackException.from_exception(error)
    shown.stack = traceback.StackSummary.from_list(
        [frame for frame in shown.stack if not frame.filename.startswith(ours + ("<frozen",))])
    return "".join(shown.format())


def _main(script: str, out_dir: str, argv: list[str]) -> None:
    import os
    import mlx.core as mx
    from backcountry_runtime import tune
    from backcountry_runtime.tune import Calls, given_call, handling, streams
    from backcountry_runtime.watch import Recorder, run_script
    recorder = Recorder()
    marked = Calls(recorder)
    # a request the user typed at init, for an app that makes none (BACKCOUNTRY_REQUEST),
    # made once the app has run; BACKCOUNTRY_NO_MAIN leaves out its __main__ block (a server)
    request = json.loads(Path(os.environ["BACKCOUNTRY_REQUEST"]).read_text()) \
        if os.environ.get("BACKCOUNTRY_REQUEST") else None
    with handling(marked):
        try:
            def make_request():
                fn = tune.defined[tuple(request["function"])]
                result = marked(fn, *given_call(fn, request["inputs"], str(Path(script).resolve().parent)))
                if streams(result):   # a streaming handler (Gradio's) runs only as it is read, as the app reads it
                    for _ in result:
                        pass
            run_script(script, argv, recorder, main=not os.environ.get("BACKCOUNTRY_NO_MAIN"),
                       then=make_request if request is not None else None)
        except Exception as error:  # the app's own error: its lines, not backcountry's
            print(_their_traceback(error), file=sys.stderr, end="")
            sys.exit(1)
    parts, arrays = [], {}
    total = sum(p.seconds for p in recorder.parts.values()) or 1.0
    for index, part in enumerate(sorted(recorder.parts.values(), key=lambda p: -p.seconds)):
        sizes = []
        for s, (key, count) in enumerate(sorted(part.shapes.items(), key=lambda kv: -kv[1])):
            keys = []
            for k, array in enumerate(part.inputs[key]):
                keys.append(f"p{index}_s{s}_{k}")
                arrays[keys[-1]] = array
            sizes.append({"inputs": [{"name": name, "shape": list(shape), "dtype": dtype} for name, shape, dtype in key[0]],
                          "options": key[1], "calls": count, "keys": keys})
        parts.append({"module": part.module, "qualname": part.qualname, "instance": part.instance,
                      "calls": part.calls, "seconds": round(part.seconds, 4), "share": part.seconds / total,
                      "sizes": sizes, "parameters": part.parameters, "unsupported": part.unsupported,
                      "language_model": part.language_model})
    if arrays and not marked.calls:  # a marked function replays its own arguments: no part inputs
        mx.save_safetensors(str(Path(out_dir) / INPUTS), arrays)
    (Path(out_dir) / RECORDING).write_text(json.dumps({"script": script, "argv": argv, "parts": parts,
                                                       "loads": recorder.loads, "marked": marked.calls,
                                                       "defined": [list(n) for n in tune.marked],
                                                       # what init asks for a request when the app makes none
                                                       "parameters": {f"{m}:{q}": tune.parameters(fn) for (m, q), fn
                                                                      in tune.defined.items()},
                                                       "request": request,
                                                       "main": not os.environ.get("BACKCOUNTRY_NO_MAIN")}, indent=1))


def describe(part: dict) -> str:
    sizes = len(part["sizes"])
    what = f"{part['qualname']} ({part['module']})"
    if part["language_model"]:
        how = "a language model called through MLX-LM"
    elif part["unsupported"]:
        how = f"cannot be timed alone: {part['unsupported']}"
    else:
        how = f"{sizes} input size{'s' if sizes != 1 else ''}"
        options = sorted({size["options"] for size in part["sizes"]} - {"{}"})
        if options:
            how += f", called with {' or '.join(options)}"
    return f"{what}: {part['calls']} call{'s' if part['calls'] != 1 else ''}, {part['share']:.0%} of the time, {how}"


def choosable(part: dict) -> bool:
    return part["language_model"] or (not part["unsupported"] and bool(part["sizes"]))


def write_model_file(recording: dict, part: dict, first_size: int, out_dir: Path, recorded_in: Path) -> Path:
    """The model file for a watched part: build() runs the script until it
    first calls the part; examples() pass the inputs it really passed, the size
    to optimize first."""
    chosen = part["sizes"][first_size]
    form = lambda size: ([a["name"] for a in size["inputs"]], size["options"])
    # one way of calling it: the chosen size's inputs and options; sizes it was
    # called with other options are left out
    order = [first_size] + [i for i, size in enumerate(part["sizes"])
                            if i != first_size and form(size) == form(chosen)]
    calls = [part["sizes"][i]["keys"] for i in order]
    names, options = form(chosen)
    by_name = options != "{}" or names != part["parameters"][:len(names)]
    run = f'''

NAMES = {names!r}
OPTIONS = {options}


def run(model, *inputs):
    return model(**dict(zip(NAMES, inputs)), **OPTIONS)
''' if by_name else ""
    left_out = len(part["sizes"]) - len(order)
    note = (f"\n{left_out} recorded size{'s' if left_out != 1 else ''} called with other options "
            f"{'are' if left_out != 1 else 'is'} left out.") if left_out else ""
    path = out_dir / MODEL_FILE
    path.write_text(f'''"""{part['qualname']} as {Path(recording['script']).name} calls it (written by `backcountry init`).

build() runs {Path(recording['script']).name} until it first calls this part, and returns it: your own
loading code builds it. examples() pass the inputs it really passed, the one to
optimize first{"; run() passes them by name with the options the script used" if by_name else ""}.
Re-run `backcountry init` after changing how the script uses it.{note}
"""

SCRIPT = {recording['script']!r}
ARGS = {recording['argv']!r}
PART = ({part['module']!r}, {part['qualname']!r}, {part['instance']})
INPUTS = {str((recorded_in / INPUTS).resolve())!r}
CALLS = {calls!r}


def build():
    from backcountry_runtime.watch import capture
    return capture(SCRIPT, ARGS, PART)


def examples():
    import mlx.core as mx
    saved = mx.load(INPUTS)
    return [tuple(saved[key] for key in keys) for keys in CALLS]
{run}''')
    return path


def marked_functions(recording: dict) -> dict:
    """{(module, qualname): [its calls, in order]} for functions marked with
    @backcountry.tune that the script called."""
    functions = {}
    for call in recording.get("marked", []):
        functions.setdefault(tuple(call["function"]), []).append(call)
    return functions


def describe_marked(call: dict) -> str:
    """A call as init lists it: its sizes in the units the questions use
    (question=5 tokens, width=512), else its arguments, and its time."""
    if call.get("sizes"):
        from .size_questions import shown
        return f"{', '.join(f'{name}={shown(value, unit)}' for name, value, unit in call['sizes'])} " \
               f"({call['seconds']:.2f} s)"
    return f"{call['arguments']}: {call['seconds']:.2f} s"


MAX_CALLS = 5   # recorded calls the search replays: each is checked and timed


def _from(path: str, project: Path) -> str:
    """path as written in a model file init leaves in the project: from the
    project's folder when inside it (the project can be copied or cloned)."""
    import os
    resolved = Path(path).resolve()
    return os.path.relpath(resolved, project.resolve()) if resolved.is_relative_to(project.resolve()) else str(resolved)


def write_marked_model_file(recording: dict, function: tuple, first: int, out_dir: Path,
                            points: dict | None = None, loop: bool = False) -> Path:
    """The model file for a marked function: build() reruns the app until it
    makes the chosen call and replays the user's own function with its
    arguments; with points ({"knobs", "sizes"}: init's size answers), at each
    size, the app's own values repeated or cut to it; without, at each
    recorded call. examples() pick which (a length-n array is the n-th).
    loop: also run(), one call as the app makes it, the recipe the loop search
    calls a rewrite of the function through."""
    calls = marked_functions(recording)[tuple(function)]
    if points is None:
        order = [first] + [i for i in range(len(calls)) if i != first][:MAX_CALLS - 1]
    else:
        order = [first]
    chosen = calls[first]
    locations = [part["location"] for part in chosen["parts"]]
    listed = "\n".join(f"    {calls[i]['index']},   # {describe_marked(calls[i])}" for i in order)
    script = Path(recording["script"]).name
    keep = ("name", "kind", "param", "path", "members", "unit", "file")   # file: an image passed as a path
    knobs = [{k: v for k, v in knob.items() if k in keep} for knob in (points or {}).get("knobs", [])]
    sized = "" if points is None else f"""
KNOBS = {knobs!r}   # what sets a size in the call's arguments
SIZES = {points["sizes"]!r}   # the sizes to search: the call made at each
"""
    request = recording.get("request")
    given = "" if request is None and recording.get("main", True) else f"""REQUEST = {(request or {}).get("inputs")!r}   # the request you gave init, made once {script} has run
MAIN = {recording.get("main", True)!r}   # False: {script} runs without its `if __name__ == "__main__":` block
"""
    path = out_dir / MODEL_FILE
    path.write_text(f'''"""{function[1]} as {script} calls it, marked with @backcountry.tune (written by `backcountry init`).

build() reruns {script} until it has made the call below, and returns your own
function with the arguments {script} gave it; the search replays it at each size,
as your app ran it. Re-run `backcountry init` after changing how the app calls it.
"""

SCRIPT = {_from(recording['script'], out_dir.parent)!r}
ARGS = {recording['argv']!r}
FUNCTION = {tuple(function)!r}
FILE = {_from(chosen.get("file", recording["script"]), out_dir.parent)!r}   # where it is defined: the app uses these kernels there
CALLS = [   # which of its calls, in the order the app made them
{listed}
]
PARTS = {locations!r}   # where it finds its model
ARGUMENTS = {[calls[i]["arguments"] for i in order]!r}   # what each was given: a rerun must match
{sized}{given}

def build():
    from pathlib import Path
    from backcountry_runtime.tune import request
    project = Path(__file__).resolve().parent.parent   # a path above is read from the project's folder
    return request(str(project / SCRIPT), ARGS, FUNCTION, CALLS, PARTS, ARGUMENTS{", KNOBS, SIZES" if points is not None else ""}{", given=REQUEST, main=MAIN" if given else ""})


def examples():
    import mlx.core as mx
    return [mx.zeros((n,)) for n in range(1, {"len(SIZES)" if points is not None else "len(CALLS)"} + 1)]


def release():
    """The job is done with {script}: drop what backcountry still holds of it (its
    function, and through it the app's globals and model), so the model is freed."""
    from backcountry_runtime.tune import forget
    forget()
''' + ("" if not loop else '''

def run(model, which):
    """One call of the function as the app makes it: what the loop search
    calls a faster version of it through, and checks against the original."""
    return model(which)
'''))
    return path


_RUN_SIZES = r"""
import importlib.util, json, sys, time, traceback
import mlx.core as mx
spec = importlib.util.spec_from_file_location("backcountry_sizes", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
request = module.build()
results = []
for example in module.examples():
    try:
        start = time.perf_counter()
        mx.eval(request(example))
        results.append([True, round(time.perf_counter() - start, 3)])
    except Exception as error:
        results.append([False, (traceback.format_exception_only(error)[-1].strip())[:300]])
print("\nSIZES " + json.dumps(results))
"""


def run_sizes(model_file: Path, timeout_s: float = 3600) -> list:
    """Each size of a marked function's model file made once, in a fresh
    process: [(ran, seconds or why it did not)]."""
    proc = subprocess.run([sys.executable, "-c", _RUN_SIZES, str(model_file)], capture_output=True, text=True,
                          timeout=timeout_s, cwd=Path(model_file).parent, stdin=subprocess.DEVNULL)
    if "\nSIZES " not in proc.stdout:
        raise ValueError(f"the app did not run to the marked call again (exit code {proc.returncode}):\n"
                         f"{proc.stderr.strip()[-2000:]}")
    return [tuple(r) for r in json.loads(proc.stdout.rsplit("\nSIZES ", 1)[1])]


def language_model_defaults(part: dict) -> dict:
    """What the script did with a language model, as init's defaults: the
    prompt lengths it read and the tokens it wrote after each."""
    lengths = [size["inputs"][0]["shape"][-1] for size in part["sizes"] if size["inputs"]]
    prompts = sorted({n for n in lengths if n > 1})
    decode = sum(size["calls"] for size in part["sizes"] if size["inputs"] and size["inputs"][0]["shape"][-1] == 1)
    defaults = {"optimize": "chat" if prompts and decode else "prompts" if prompts else "replies"}
    if prompts:
        defaults["optimize_at"] = prompts[0] if len(prompts) == 1 else sorted(prompts)[len(prompts) // 2]
        # the range covers what the script read: one 18-token prompt widens
        # the usual 32-2049 down to 18, not an optimize_at outside its range
        defaults["prompt_tokens"] = f"{min(prompts[0], 32)}-{max(prompts[-1], 2049)}" if len(prompts) == 1 \
            else f"{prompts[0]}-{prompts[-1]}"
        defaults["history_tokens"] = max(prompts)
    if decode and prompts:
        defaults["reply_tokens"] = max(1, decode // len(prompts))
    return defaults


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2], sys.argv[3:])
