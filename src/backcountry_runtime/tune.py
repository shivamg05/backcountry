"""`@backcountry.tune` marks the function an app calls for one request (write
a reply, make an image, classify a batch), where the app defines it:

    @backcountry.tune
    def generate_text(model, tokenizer, prompt): ...

The app runs as it always does. Once a backcountry job for the function has
finished, its first call installs the kernels the job found into the model
the function uses (install_stored), so the app gets them with no other change;
BACKCOUNTRY_OFF=1 runs it untouched. Under
`backcountry init app.py` each call is recorded (Calls): its arguments, how
long it took and the model parts it called. The search then replays one
recorded call at a time (Request): the user's own function, given the very
arguments the app passed, is the "model" the job traces, checks and times.
Its loop, cache, sampling and post-processing stay the user's own code.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
import io
import random
import threading
import time
import types

import mlx.core as mx
import mlx.nn as nn

_handler = None   # set while backcountry records or replays the app
marked: list[tuple[str, str]] = []   # every function marked in this process, as it was defined
# a marked function as a loop target, "<file>:<qualname>": the function this
# process replays or runs there, and where it finds its model (init's PARTS)
_live: dict[str, tuple] = {}
_answers: dict = {}   # while the loop search checks a rewrite: original function -> what answers for it
defined: dict = {}    # every marked function in this process, by name_of: the one init calls when the app does not


def forget():
    """Drop every marked function this process ran, and with them the app
    globals they read (its model): a job's model file calls it as it ends."""
    _live.clear()
    _answers.clear()
    defined.clear()


# -- a request the user gives init, for an app that makes none -------------------

def parameters(fn) -> list[dict]:
    """What init asks for a request to fn: each parameter's name, its default
    (shown) and its annotation, and whether the app has a global of that name
    (its model, its tokenizer), which is then passed and not asked."""
    out = []
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.append({"name": name, "default": None if p.default is p.empty else repr(p.default),
                    "annotation": None if p.annotation is p.empty else
                    getattr(p.annotation, "__name__", None) or str(p.annotation),
                    "global": p.default is p.empty and name in fn.__globals__
                    and _app_object(fn.__globals__[name])})
    return out


def _app_object(value) -> bool:
    """A global the app loaded for the model's side (the model, its tokenizer or
    processor, an engine holding the model), passed rather than asked. A Gradio
    app's `style = gr.Radio(...)` shares its name with the handler's `style`
    argument, and the radio button was passed where a style name goes."""
    if isinstance(value, nn.Module):
        return True
    if callable(getattr(value, "encode", None)) and callable(getattr(value, "decode", None)):
        return True   # a tokenizer
    held = getattr(value, "__dict__", None)
    return isinstance(held, dict) and any(
        isinstance(v, nn.Module) or (callable(getattr(v, "encode", None)) and callable(getattr(v, "decode", None)))
        for v in held.values())   # an engine, a processor


def given_call(fn, answers: dict, folder: str) -> tuple[tuple, dict]:
    """(args, kwargs) for a call of fn from the user's answers (text as typed):
    a global of a parameter's name when it was not asked, its default when
    left empty, else the answer read as a value (parse_value)."""
    kwargs = {}
    for p in parameters(fn):
        text = answers.get(p["name"])
        if text not in (None, ""):
            kwargs[p["name"]] = parse_value(text, p["annotation"], folder)
        elif p["global"]:
            kwargs[p["name"]] = fn.__globals__[p["name"]]
        elif p["default"] is None:
            raise ValueError(f"{fn.__qualname__} needs {p['name']}: give it a value")
    return (), kwargs


def parse_value(text: str, annotation: str | None, folder: str):
    """A typed answer as the value a function is given: a file (beside the app,
    or a full path) read by its kind (an image opened unless the parameter is
    annotated as a str or Path, a .json loaded, a .npy loaded, a .txt read); a
    Python literal (a number, a list, a dict, a quoted string) as itself;
    anything else as the text."""
    import ast
    import json
    import os
    path = text if os.path.isabs(text) else os.path.join(folder, text)
    if os.path.isfile(path) and (annotation or "") not in ("str", "Path"):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            from PIL import Image
            with Image.open(path) as image:
                return image.convert("RGB")
        if ext == ".json":
            with open(path) as f:
                return json.load(f)
        if ext == ".npy":
            import numpy as np
            return np.load(path)
        if ext == ".txt":
            with open(path) as f:
                return f.read()
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def marked_target(file: str, qualname: str) -> str:
    """The loop search's name for a marked function: the file defining it and
    its name (a script's functions all live in __main__, so the module says
    nothing)."""
    from pathlib import Path
    return f"{Path(file).resolve()}:{qualname}"


def is_marked(target: str) -> bool:
    return target.partition(":")[0].endswith(".py")


def written_as(fn) -> str:
    """A marked function's own source, the mark left off: what the loop search rewrites."""
    import textwrap
    lines = textwrap.dedent(inspect.getsource(fn)).split("\n")
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    return "\n".join(lines)


WRITTEN_AS = "# written as: "   # run.py's second line: the digest of the code it was made from


def digest(source: str) -> str:
    import hashlib
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def live(target: str) -> tuple:
    """(function, locations) for a marked target in this process: registered
    by a model file's build() (request) or by the app's first call."""
    try:
        return _live[target]
    except KeyError:
        raise LookupError(f"{target}: this process has not run that marked function "
                          "(its model file's build() or the app registers it)") from None


def model_finder(target: str):
    """(args, kwargs) -> the model a marked function's call works on, found
    where init saw it (an argument, a global, a library's own holder)."""
    fn, locations = live(target)
    return lambda args, kwargs: resolve(locations[0], fn, args, kwargs) if locations else None


def tune(fn=None, *, apply: bool = True):
    """Mark the function an app calls for one request (see the module doc).
    @backcountry.tune(apply=False) keeps the mark but runs the function untouched."""
    if fn is None:
        return lambda f: tune(f, apply=apply)

    @functools.wraps(fn)
    def tuned(*args, **kwargs):
        handler = _handler
        if handler is None:
            if not tuned.__backcountry_checked__:
                tuned.__backcountry_checked__ = True
                if apply:
                    tuned.__backcountry_found__ = install_stored(fn, args, kwargs, loop=True)
            return (tuned.__backcountry_found__ or fn)(*args, **kwargs)
        return handler(fn, args, kwargs)

    tuned.__backcountry_checked__ = False
    tuned.__backcountry_found__ = None   # a faster version of fn a finished job found, once installed
    tuned.__backcountry_tune__ = fn
    marked.append(name_of(fn))
    defined[name_of(fn)] = fn
    return tuned


RESULTS = "backcountry-results"   # beside the file that defines a marked function: its result, to commit or ship


def results_dir(file: str, qualname: str):
    from pathlib import Path
    return Path(file).resolve().parent / RESULTS / qualname


def publish(artifact_dir, model_file):
    """A finished artifact made for a marked function, copied beside the file
    that defines it (backcountry-results/<function>), replacing the last one:
    the app finds it there, and it ships with the code. None otherwise."""
    import shutil
    from pathlib import Path
    made_for = marked_model_file(model_file)
    if made_for is None:
        return None
    dest = results_dir(made_for.get("FILE", made_for["SCRIPT"]), made_for["FUNCTION"][1])
    incoming = dest.with_name(f".{dest.name}.incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.copytree(artifact_dir, incoming, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.rmtree(dest, ignore_errors=True)
    incoming.rename(dest)
    return dest


def name_of(fn) -> tuple[str, str]:
    return fn.__module__, fn.__qualname__


@contextlib.contextmanager
def handling(handler):
    global _handler
    saved, _handler = _handler, handler
    try:
        yield handler
    finally:
        _handler = saved


# -- a finished job's kernels, in the app -----------------------------------------

def tuned_for(artifact_dir) -> dict | None:
    """What a stored artifact was made for, when it was made for a marked
    function: {"SCRIPT", "FUNCTION", "PARTS"} read from its model file."""
    import json
    from pathlib import Path
    try:
        entry = json.loads((Path(artifact_dir) / "bundle.json").read_text())["entry"]
    except (OSError, ValueError, KeyError):
        return None
    return marked_model_file(Path(artifact_dir) / "model" / entry)


def marked_model_file(path) -> dict | None:
    """{"SCRIPT", "FUNCTION", "PARTS"} of a model file init wrote for a marked
    function, read from its literals (never run); None for any other."""
    import ast
    from pathlib import Path
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, ValueError, SyntaxError):
        return None
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("SCRIPT", "FUNCTION", "PARTS", "FILE", "ARGUMENTS", "LEVEL", "SIZES", "KNOBS"):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                return None
    return found if {"SCRIPT", "FUNCTION", "PARTS"} <= set(found) else None


def call_label(made_for: dict, workload: str) -> str:
    """A workload of a marked function's job in the user's terms: which
    recorded call it replays and what that call was given."""
    import re
    found = re.search(r"(?:_|@T=)(\d+)$", workload)
    n = int(found.group(1)) if found else 1
    if made_for.get("SIZES") is not None:
        from .sizes import label
        setting = made_for["SIZES"][n - 1:n]
        return label(setting[0] if setting else {}, made_for.get("KNOBS") or []) or "as your app called it"
    given = (made_for.get("ARGUMENTS") or [])[n - 1:n]
    return f"call {n}" + (f" ({given[0]})" if given else "")


def made_for_function(made_for: dict | None, fn, beside: bool = False) -> bool:
    """Whether an artifact was made for this function as it is defined: its
    name and the file it is written in (any script may import it), never
    another function of the same name elsewhere. beside: the artifact sits
    in that file's own backcountry-results folder, so the file's name is
    enough (the project was copied or cloned since: another full path)."""
    from pathlib import Path
    if made_for is None or tuple(made_for["FUNCTION"])[1] != fn.__qualname__:
        return False
    made = Path(made_for.get("FILE", made_for["SCRIPT"]))
    here = Path(fn.__code__.co_filename)
    return made.name == here.name if beside else made.resolve() == here.resolve()


class _Parts(nn.Module):
    """The parts a call reaches, at the addresses a Request held them."""

    def __init__(self, parts):
        super().__init__()
        self.parts = list(parts)


def install_stored(fn, args, kwargs, loop: bool = False):
    """On a marked function's first call: the newest stored artifact made for
    this function in this file, installed into the model this call reaches
    (in place: the app keeps its own objects). Returns the artifact's name;
    with loop, the faster version of fn it carries (run.py), or None."""
    import os
    import sys
    import warnings
    from pathlib import Path
    if os.environ.get("BACKCOUNTRY_OFF"):
        return None
    from .apply import apply, artifacts_home
    beside = results_dir(fn.__code__.co_filename, fn.__qualname__)
    home = artifacts_home()
    stored = [d for d in home.iterdir() if (d / "swap_table.json").is_file()] if home.is_dir() else []
    stored = sorted(stored, key=lambda d: (d / "swap_table.json").stat().st_mtime, reverse=True)
    for artifact in ([beside] if (beside / "swap_table.json").is_file() else []) + stored:
        made_for = tuned_for(artifact)
        if not made_for_function(made_for, fn, beside=artifact == beside):
            continue
        try:
            parts = [resolve(location, fn, args, kwargs) for location in made_for["PARTS"]]
            installed = []
            if made_for.get("LEVEL") == "model":  # the model itself was searched (an engine's): its own layers
                apply(parts[0], artifact, _installed=installed)
                installed = [(f"parts.0.{path}" if path else "parts.0", original, wrapper) for path, original, wrapper in installed]
            else:
                apply(_Parts(parts), artifact, _installed=installed)
        except Exception as error:  # the app runs as it is, and says why
            warnings.warn(f"backcountry: {artifact.name} was not installed into {fn.__qualname__}: "
                          f"{type(error).__name__}: {error}", stacklevel=3)
            return None
        # a layer inside a part is replaced where the app holds it; a whole
        # part replaced would change only the holder, so it is left out
        whole = [path for path, _, _ in installed if path.count(".") == 1]
        reached = len(installed) - len(whole)
        left = f"; {len(whole)} that replace a whole part are left out" if whole else ""
        found = _stored_loop(artifact, fn, made_for) if loop else None
        layers = f"{reached} optimized layer{'s' if reached != 1 else ''}"
        print(f"backcountry: {fn.__qualname__} runs "
              + ("a faster version of itself (the same outputs)" + (f" with {layers}" if reached else "")
                 if found else f"with {layers}")
              + f" from {artifact}{left}. To run it untouched: BACKCOUNTRY_OFF=1, or @backcountry.tune(apply=False); "
              f"to remove it, delete that folder", file=sys.stderr, flush=True)
        return found if loop else artifact.name
    return None


def _stored_loop(artifact, fn, made_for: dict):
    """The faster version of fn an artifact carries (its run.py names the
    marked function), set up to answer for fn on the model it was made for;
    None when it carries none, or it does not load here (the app runs fn)."""
    import importlib.util
    import warnings
    from .functions import target_of
    run = artifact / "run.py"
    text = run.read_text() if run.is_file() else ""
    target = target_of(text)
    if target is None or not is_marked(target):
        return None
    made_from = next((line[len(WRITTEN_AS):].strip() for line in text.split("\n")[1:2] if line.startswith(WRITTEN_AS)), None)
    if made_from is not None and made_from != digest(written_as(fn)):   # its rewrite would answer for older code
        warnings.warn(f"backcountry: {fn.__qualname__} has changed since its faster version in {artifact} was made; "
                      "it runs as written (run `backcountry start` again for one of the new code)", stacklevel=4)
        return None
    _live[target] = (fn, [tuple(location) for location in made_for["PARTS"]])  # its name where the job ran
    try:
        spec = importlib.util.spec_from_file_location(f"backcountry_loop_{abs(hash(str(artifact)))}",
                                                      artifact / "model_loop.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.answer
    except Exception as error:
        warnings.warn(f"backcountry: the faster {fn.__qualname__} in {artifact.name} did not load "
                      f"({type(error).__name__}: {error}); {fn.__qualname__} runs as written", stacklevel=4)
        return None


# -- the arguments a call was given ---------------------------------------------

def snapshot(value, memo=None):
    """The value with its lists, tuples and dicts copied (a function may append
    to the history it is given) and everything else shared; with memo, the
    model parts in it are memo's copies."""
    if memo is not None and isinstance(value, nn.Module):
        return copy.deepcopy(value, memo)
    if isinstance(value, list):
        return [snapshot(v, memo) for v in value]
    if isinstance(value, tuple):
        return tuple(snapshot(v, memo) for v in value)
    if isinstance(value, dict) and not isinstance(value, nn.Module):
        return {k: snapshot(v, memo) for k, v in value.items()}
    return value


def describe_value(value) -> str:
    if isinstance(value, mx.array):
        return f"array {list(value.shape)} {str(value.dtype).removeprefix('mlx.core.')}"
    if isinstance(getattr(value, "shape", None), tuple) and hasattr(value, "dtype"):  # numpy and its kin
        return f"{type(value).__name__} {list(value.shape)} {value.dtype}"
    if isinstance(value, str):
        return f"text ({len(value)} characters)"
    if isinstance(value, (bool, int, float)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} of {len(value)}"
    return type(value).__name__


def describe_call(fn, args, kwargs) -> str:
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        items = bound.arguments.items()
    except (TypeError, ValueError):
        items = [*((str(i), a) for i, a in enumerate(args)), *kwargs.items()]
    return ", ".join(f"{name}={describe_value(value)}" for name, value in items)


# -- where the function finds its model -----------------------------------------

_LEAVES = (str, bytes, int, float, bool, type(None), types.ModuleType, type, types.FunctionType,
           types.BuiltinFunctionType, types.MethodType, mx.array)


def _steps(value):
    """(step, child) for each thing a search for a model part looks inside."""
    if isinstance(value, _LEAVES):
        return []
    if isinstance(value, dict):  # an nn.Module is a dict of its children
        return [(("item", k), v) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [(("item", i), v) for i, v in enumerate(value)]
    if type(value).__module__.split(".")[0] == "mlx":
        return []
    try:
        return [(("attr", k), v) for k, v in vars(value).items()]
    except TypeError:
        return []


def roots(fn, args, kwargs):
    """Where a call's model can be: its arguments, then the globals and
    closure the function reads, then the rest of its module's globals."""
    for i, value in enumerate(args):
        yield ("arg", i), value
    for key, value in kwargs.items():
        yield ("kwarg", key), value
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            yield ("closure", name), cell.cell_contents
        except ValueError:  # an empty cell: nothing there to find
            continue
    names = fn.__globals__
    read = [n for n in fn.__code__.co_names if n in names]
    for name in [*read, *(n for n in names if n not in read)]:
        if not name.startswith("__"):
            yield ("global", name), names[name]


def _holds(value, module, trees: dict) -> bool:
    """Whether value is the module or an nn.Module whose tree holds it (at
    any depth: a VAE's decoder.up_blocks[0].resnets[1].norm1)."""
    if value is module:
        return True
    if not isinstance(value, nn.Module):
        return False
    if id(value) not in trees:
        trees[id(value)] = {id(m) for _, m in value.named_modules()}
    return id(module) in trees[id(value)]


def locate(module, fn, args, kwargs, depth: int = 5, budget: int = 50000, trees: dict | None = None):
    """The shortest path from a call's roots to the outermost model that holds
    a part it called, e.g. (("arg", 0), [("attr", "model")]), or None: an app
    that calls a VAE's layers one by one gets the VAE, whose whole tree the
    search then sees."""
    trees = {} if trees is None else trees
    seen = set()
    frontier = []
    for root, value in roots(fn, args, kwargs):
        if _holds(value, module, trees):
            return root, []
        frontier.append((root, [], value))
    for _ in range(depth):
        following = []
        for root, path, value in frontier:
            if id(value) in seen or isinstance(value, nn.Module):  # a model that does not hold it: not inside
                continue
            seen.add(id(value))
            budget -= 1
            if budget < 0:
                return None
            for step, child in _steps(value):
                if _holds(child, module, trees):
                    return root, path + [step]
                following.append((root, path + [step], child))
        frontier = following
    return None


_NOT_LIBRARIES = {"mlx", "numpy", "backcountry", "backcountry.harness", "backcountry_runtime", "backcountry", "builtins"}


def held_at(part, fn, args, kwargs, depth: int = 5, budget: int = 50000) -> list:
    """Every place a call's roots hold this very object, as locate's
    locations: an engine keeps the model the app loaded as engine.model too."""
    found, seen, frontier = [], set(), []
    for root, value in roots(fn, args, kwargs):
        if value is part:
            found.append((root, []))
        elif not isinstance(value, nn.Module):
            frontier.append((root, [], value))
    for _ in range(depth):
        following = []
        for root, path, value in frontier:
            if id(value) in seen:
                continue
            seen.add(id(value))
            budget -= 1
            if budget < 0:
                return found
            for step, child in _steps(value):
                if child is part:
                    found.append((root, path + [step]))
                elif not isinstance(child, nn.Module):
                    following.append((root, path + [step], child))
        frontier = following
    return found


def locate_in_libraries(module, fn, depth: int = 4, budget: int = 50000):
    """Where a library keeps the model it loaded itself (mlx-whisper's
    ModelHolder.model): searched from the attributes of the modules of the
    libraries the function's globals use, through their classes, as
    (("module", "pkg.mod:attribute"), path), or None."""
    import sys
    tops = set()
    for value in fn.__globals__.values():
        name = getattr(value, "__module__", None) if not isinstance(value, types.ModuleType) else value.__name__
        if isinstance(name, str):
            tops.add(name.split(".")[0])
    tops -= _NOT_LIBRARIES | set(sys.stdlib_module_names) | {(fn.__module__ or "__main__").split(".")[0]}

    def inside(value):
        owner = value.__name__ if isinstance(value, types.ModuleType) else getattr(value, "__module__", "")
        return isinstance(owner, str) and owner.split(".")[0] in tops

    frontier, trees = [], {}
    for name, loaded in list(sys.modules.items()):
        if loaded is None or name.split(".")[0] not in tops:
            continue
        for attribute, value in list(vars(loaded).items()):
            if _holds(value, module, trees):
                return ("module", f"{name}:{attribute}"), []
            if isinstance(value, type) and inside(value):
                frontier.append((("module", f"{name}:{attribute}"), [], value))
    seen = set()
    for _ in range(depth):
        following = []
        for root, path, value in frontier:
            if id(value) in seen:
                continue
            seen.add(id(value))
            budget -= 1
            if budget < 0:
                return None
            if isinstance(value, nn.Module):
                continue
            children = [(("attr", k), v) for k, v in vars(value).items() if not k.startswith("__")] \
                if isinstance(value, type) else _steps(value)
            for step, child in children:
                if _holds(child, module, trees):
                    return root, path + [step]
                if isinstance(child, type) and not inside(child):
                    continue
                following.append((root, path + [step], child))
        frontier = following
    return None


def resolve(location, fn, args, kwargs):
    (kind, key), path = location
    if kind == "arg":
        value = args[key]
    elif kind == "kwarg":
        value = kwargs[key]
    elif kind == "closure":
        value = fn.__closure__[fn.__code__.co_freevars.index(key)].cell_contents
    elif kind == "module":
        import sys
        name, _, attribute = key.rpartition(":")
        value = getattr(sys.modules[name], attribute)
    else:
        value = fn.__globals__[key]
    for how, step in path:
        value = value[step] if how == "item" else getattr(value, step)
    return value


# -- what a call returned, as arrays the job compares ---------------------------

def streams(value) -> bool:
    """A result whose work runs while the app reads it: a generator, or any
    iterator an engine hands back (LM Studio's create_generator)."""
    import collections.abc
    return isinstance(value, collections.abc.Iterator) and not isinstance(value, (str, bytes, mx.array))


def arrays_of(value, out=None, depth=0, names=None, path="result"):
    """A call's result as arrays: arrays as they are, text as its bytes,
    numbers as scalars; lists, dicts, dataclasses, named tuples and streams
    opened up. Any other object gives only the arrays and images it holds
    directly (mflux's GeneratedImage: its image), nothing else of it: opening
    an arbitrary object reached a model kit's tokenizer, one array per
    vocabulary string, and that image's other fields include its generation
    time, which differs on every run. names: gets where in the result each
    array came from (result[1].generation_tps), to say which part differs."""
    import dataclasses
    out = [] if out is None else out
    if depth > 6 or value is None:
        return out
    add = lambda array: (out.append(array), names is not None and names.append(path))
    if isinstance(value, mx.array):
        add(value)
    elif isinstance(value, (str, bytes)):
        data = value.encode() if isinstance(value, str) else value
        add(mx.array(list(data), dtype=mx.uint8))
    elif isinstance(value, (bool, int, float)):
        add(mx.array(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            arrays_of(item, out, depth + 1, names, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)) or streams(value):
        for i, item in enumerate(value):
            arrays_of(item, out, depth + 1, names, f"{path}[{i}]")
    elif (hasattr(value, "__array__") or hasattr(value, "__array_interface__")) \
            and not isinstance(value, nn.Module):  # numpy, a PIL image
        import numpy as np
        add(mx.array(np.asarray(value)))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            arrays_of(getattr(value, f.name), out, depth + 1, names, f"{path}.{f.name}")
    elif not isinstance(value, (nn.Module, type, types.ModuleType, types.FunctionType)):
        for key, item in getattr(value, "__dict__", {}).items():
            if isinstance(item, mx.array) or ((hasattr(item, "__array__") or hasattr(item, "__array_interface__"))
                                              and not isinstance(item, (nn.Module, type))):
                arrays_of(item, out, depth + 1, names, f"{path}.{key}")
    return out


@contextlib.contextmanager
def _replaying():
    """One replayed call: what it prints (stdout and stderr) goes nowhere, and its random draws
    (mx.random, numpy, random) start from the same seed every time, so a
    sampling app gives the same output on every replay."""
    import numpy as np
    np_state, py_state = np.random.get_state(), random.getstate()
    mx.random.seed(0)
    np.random.seed(0)
    random.seed(0)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield  # an error still raises; prints and progress bars (mflux's) go nowhere
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


class Request(nn.Module):
    """The user's own function and the arguments the app gave it, as the job's
    model: called with an array whose length picks the recorded call (1 is
    the first chosen), it runs that call as the app did and returns its
    result as arrays. Its parts (the model the function reaches) are its
    children, so the job traces and replaces work inside them."""

    def __init__(self, fn, calls, parts, locations, also=None):
        super().__init__()
        self.parts = list(parts)
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_calls", calls)
        object.__setattr__(self, "_locations", [tuple(location) for location in locations])
        # every other place the app holds the same part (an engine's own engine.model beside the
        # global model): a copy is put there too, or the engine would run the original
        object.__setattr__(self, "_also", [list(more) for more in (also or [[] for _ in self._locations])])

    def _held(self, parts):
        """(location, part) for each place a part is held: init's, then the others."""
        return [(tuple(location), part) for main, more, part in zip(self._locations, self._also, parts)
                for location in [main, *more]]

    def __call__(self, which):
        args, kwargs = snapshot(self._calls[which.shape[0] - 1])
        with self._holding(), _replaying():
            result = _answers.get(self._fn, self._fn)(*args, **kwargs)
            names = []
            out = arrays_of(result, names=names)
            object.__setattr__(self, "_names", names)   # where each output came from (check names a differing one)
        if not out:
            raise ValueError(f"{self._fn.__qualname__} returned nothing to compare (it returned "
                             f"{type(result).__name__}): return what the request produces")
        return tuple(out)

    @contextlib.contextmanager
    def _holding(self):
        """A model the function reaches through globals (its module's, or a
        library's that keeps its model itself) is this Request's own part for
        the call, and what was there is put back after: a copy runs its own
        model, and the original's call changes nothing."""
        saved = []
        try:
            for location, part in self._held(self.parts):
                if location[0][0] in ("arg", "kwarg"):
                    continue
                owner, how, step = _owner(location, self._fn)
                current = _get(owner, how, step)
                if current is not part:
                    saved.append((owner, how, step, current))
                    _set(owner, how, step, part)
            yield
        finally:
            for owner, how, step, value in reversed(saved):
                _set(owner, how, step, value)

    def __deepcopy__(self, memo):
        parts = [copy.deepcopy(part, memo) for part in self.parts]
        calls = []
        for args, kwargs in self._calls:
            args, kwargs = list(snapshot(args)), snapshot(kwargs)
            for location, part in self._held(parts):
                if location[0][0] in ("arg", "kwarg"):
                    _place(location, args, kwargs, part)
            calls.append((tuple(args), kwargs))
        return Request(self._fn, calls, parts, self._locations, self._also)


def _place(location, args, kwargs, part):
    """Put a copy's own part where its arguments hold the model: every object
    on the way is copied shallowly (a kit holding the model gets a copy whose
    .model is the part), everything else stays shared."""
    (kind, key), path = location
    holder = args if kind == "arg" else kwargs
    if not path:
        holder[key] = part
        return
    value = holder[key] = _own(holder[key])
    for how, step in path[:-1]:
        child = _own(value[step] if how == "item" else getattr(value, step))
        if how == "item":
            value[step] = child
        else:
            setattr(value, step, child)
        value = child
    how, step = path[-1]
    if how == "item":
        value[step] = part
    else:
        setattr(value, step, part)


def _owner(location, fn):
    """(object, "item" | "attr" | "cell", key) holding a part reached
    through globals, a closure or a library module."""
    (kind, key), path = location
    if path:
        return resolve(((kind, key), path[:-1]), fn, (), {}), *path[-1]
    if kind == "closure":
        return fn.__closure__[fn.__code__.co_freevars.index(key)], "cell", None
    if kind == "module":
        import sys
        name, _, attribute = key.rpartition(":")
        return sys.modules[name], "attr", attribute
    return fn.__globals__, "item", key


def _get(owner, how, step):
    if how == "cell":
        return owner.cell_contents
    return owner[step] if how == "item" else getattr(owner, step)


def _set(owner, how, step, value):
    if how == "cell":
        owner.cell_contents = value
    elif how == "item":
        owner[step] = value
    else:
        setattr(owner, step, value)


def _own(value):
    if isinstance(value, tuple):
        raise TypeError("the model sits inside a tuple, which a copy cannot change")
    return copy.copy(value)


# -- the two ways backcountry runs the app ----------------------------------------

class Calls:
    """While init runs the app: every call of a marked function, with its
    arguments described, its time and the model parts it called (seen by the
    watch Recorder, which must be installed). The first call of each function
    is then probed: each knob in its arguments (backcountry_runtime.sizes) set
    to another size, the function run once more, and what the model saw
    compared, so init asks only about the knobs that change the work."""

    def __init__(self, recorder, probe: bool = True):
        self.recorder = recorder
        self.calls: list[dict] = []
        self.counts: dict = {}
        self.depth = 0
        self.probe = probe
        self.probed: set = set()

    def __call__(self, fn, args, kwargs):
        if self.depth:  # a marked function inside another: part of the outer call
            return fn(*args, **kwargs)
        name = name_of(fn)
        index = self.counts.get(name, 0)
        self.counts[name] = index + 1
        call = {"fn": fn, "args": args, "kwargs": kwargs, "snapshot": snapshot((args, kwargs)), "name": name,
                "index": index, "described": describe_call(fn, args, kwargs), "before": self._calls_now(),
                "shapes": self._shapes_now(), "caller": threading.get_ident()}
        self.depth = 1
        call["start"] = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            if streams(result):  # a stream: its work runs while the app reads it
                return self._following(result, call)
            mx.eval(arrays_of(result))
        finally:
            self.depth = 0
        self._record(call, time.perf_counter() - call["start"])
        return result

    def _calls_now(self):
        return {key: part.calls for key, part in self.recorder.parts.items()}

    def _shapes_now(self):
        return {key: (part.calls, dict(part.shapes)) for key, part in self.recorder.parts.items()}

    def _signature(self, before) -> dict:
        """What the model saw since before: {part and input shapes: calls},
        and each part's call count (a call with a cache records no shapes:
        mlx-vlm's decode steps), so fewer replies show as fewer calls."""
        seen = {}
        for key, part in self.recorder.parts.items():
            calls, earlier = before.get(key, (0, {}))
            name = f"{part.module}.{part.qualname}#{part.instance}"
            if part.calls - calls:
                seen[f"{name} calls"] = part.calls - calls
            for shape, count in part.shapes.items():
                more = count - earlier.get(shape, 0)
                if more:
                    seen[f"{name} {shape!r}"] = more
        return seen

    def _following(self, stream, call):
        """The stream as the app reads it, recorded as one call when it ends:
        the time from the call to its last item, and the parts it ran."""
        try:
            for item in stream:
                self.depth = 1
                yield item
        finally:
            self.depth = 0
            self._record(call, time.perf_counter() - call["start"])

    def _record(self, call, seconds):
        fn, (args, kwargs) = call["fn"], (call["args"], call["kwargs"])
        before = call["before"]
        called = [key for key, part in self.recorder.parts.items() if part.calls > before.get(key, 0)]
        # an engine runs the model on its own threads: what the call made ran elsewhere
        elsewhere = any(t != call["caller"] for key in called for t in self.recorder.parts[key].threads)
        parts, missing, trees = [], [], {}
        for key in called:
            module = self.recorder.objects[key]
            location = locate(module, fn, args, kwargs, trees=trees) or locate_in_libraries(module, fn)
            part = self.recorder.parts[key]
            if location is None:
                missing.append({"class": f"{part.module}.{part.qualname}", "location": None})
            elif all(p["location"] != location for p in parts):  # its layers called one by one: one part
                held = resolve(location, fn, args, kwargs)
                parts.append({"class": f"{type(held).__module__}.{type(held).__qualname__}", "location": location})
        record = {"function": list(call["name"]), "file": fn.__code__.co_filename, "index": call["index"],
                  "arguments": call["described"], "seconds": round(seconds, 4), "parts": parts,
                  "unlocated": missing, "other_threads": elsewhere,
                  "language_model": any(self.recorder.parts[k].language_model for k in called),
                  "loads": list(self.recorder.loads)}
        self.calls.append(record)
        if self.probe and parts:
            knobs, tokenizer = self._knobs(fn, *call["snapshot"])
            # each call's sizes, in the units init asks in: the sizes the app used
            record["sizes"] = [[k["name"], k["value"], k.get("unit")] for k in knobs]
            if call["name"] not in self.probed:
                self.probed.add(call["name"])
                record["knobs"] = self._probe(fn, *call["snapshot"], self._signature(call["shapes"]), knobs, tokenizer)

    @staticmethod
    def _knobs(fn, args, kwargs):
        """(the call's knobs with their values, text in tokens when a
        tokenizer is reachable, the tokenizer)."""
        from . import sizes
        tokenizer = sizes.tokenizer_for(fn, args, kwargs)
        knobs = sizes.find(fn, args, kwargs)
        for knob in knobs:
            if knob["kind"] == "text":
                knob["value"] = sizes.text_size(knob, fn, args, kwargs, tokenizer)
                knob["unit"] = "tokens" if tokenizer is not None else "characters"
        return knobs, tokenizer

    def _probe(self, fn, args, kwargs, base: dict, knobs: list, tokenizer) -> list[dict]:
        """Each knob of the call set to its probe size, the function run once,
        and its effect: size (the model's shapes change), more (the same
        shapes, run more or fewer times), same (no change) or fails."""
        from . import sizes
        for knob in knobs:
            try:
                a, k = sizes.apply(fn, *snapshot((args, kwargs)), [knob], {knob["name"]: sizes.probe_value(knob)},
                                   tokenizer)
                seen = self._measured(fn, a, k)
            except Exception as error:  # the model cannot take it at another size
                knob["effect"], knob["why"] = "fails", f"{type(error).__name__}: {error}"[:200]
                continue
            knob["effect"] = "same" if seen == base else "more" if set(seen) == set(base) else "size"
            knob["signature"] = sorted(seen.items())
            # the parts of the model whose work it changed: width and height
            # change the same ones (their shapes differ: a wide image, a tall one)
            changed = {key for key in set(seen) | set(base) if seen.get(key) != base.get(key)}
            knob["parts"] = sorted({key.rsplit(" calls", 1)[0].split(" ", 1)[0] for key in changed})
        return knobs

    def _measured(self, fn, args, kwargs) -> dict:
        shapes = self._shapes_now()
        self.depth = 1
        try:
            with _replaying():
                mx.eval(arrays_of(fn(*args, **kwargs)))
        finally:
            self.depth = 0
        return self._signature(shapes)


class Captured(Exception):
    pass


class Replay:
    """While a model file's build() reruns the app: the arguments of the
    wanted calls of one marked function, then a stop just before the last of
    them runs (the calls before it run, so the app's state is as it was)."""

    def __init__(self, function: tuple[str, str], wanted: list[int]):
        self.function, self.wanted, self.last = tuple(function), set(wanted), max(wanted)
        self.count, self.fn, self.calls, self.depth = 0, None, {}, 0

    def __call__(self, fn, args, kwargs):
        if self.depth or name_of(fn) != self.function:
            return fn(*args, **kwargs)
        index, self.count = self.count, self.count + 1
        if index in self.wanted:
            self.fn, self.calls[index] = fn, snapshot((args, kwargs))
        if index == self.last and threading.current_thread() is threading.main_thread():
            raise Captured()
        # on a worker thread (a server's model thread) the stop never reached
        # the app's main code: the server failed the request, and the app with
        # it. There the call runs and the app goes on to its end
        self.depth = 1
        try:
            return fn(*args, **kwargs)
        finally:
            self.depth = 0


def request(script: str, argv: list[str], function, calls: list[int], locations: list,
            arguments: list[str] | None = None, knobs: list | None = None, sizes: list | None = None,
            given: dict | None = None, main: bool = True) -> Request:
    """A model file's build() for a marked function: rerun the app until it
    makes the last wanted call, and return a Request over the wanted calls
    (in the order given: the first is the one to optimize). given: the
    request the user typed at init for an app that makes none, made once the
    app has run (main=False: without its __main__ block, a server's loop)."""
    import os
    from .watch import run_script
    replay = Replay(function, calls)
    try:
        with handling(replay):
            run_script(script, argv, None, main=main)
            if given is not None:  # the request the user gave init, made as init made it
                fn = defined[tuple(function)]
                replay(fn, *given_call(fn, given, os.path.dirname(os.path.abspath(script))))
    except Captured:
        pass
    else:
        if replay.count <= replay.last:   # a call reached on a worker thread runs on (Replay)
            raise RuntimeError(f"{script} made {replay.count} calls of {function[1]}, fewer than recorded "
                               f"({max(calls) + 1}): it must run the way it did when init watched it")
    ordered = [replay.calls[i] for i in calls]
    for index, (call_args, call_kwargs), recorded in zip(calls, ordered, arguments or []):
        now = describe_call(replay.fn, call_args, call_kwargs)
        if now != recorded:
            raise RuntimeError(f"{script} called {function[1]} with other arguments than when init watched it "
                               f"(call {index + 1}: then {recorded}; now {now}). It must make the same calls every "
                               "run: seed any randomness that picks its inputs, then rerun `backcountry init`")
    args, kwargs = ordered[0]
    _live[marked_target(replay.fn.__code__.co_filename, replay.fn.__qualname__)] = (replay.fn, list(locations))
    parts = [resolve(location, replay.fn, args, kwargs) for location in locations]
    if any(part is None for part in parts):
        # a library that loads its model on first use (mlx-whisper's ModelHolder)
        # has none before the call the rerun stopped at: the job held None, put it
        # back for every replay, and each replay loaded the model from disk with
        # no kernel able to reach it. The call is made once, and the model looked for again
        once_args, once_kwargs = snapshot((args, kwargs))
        with _replaying():
            replay.fn(*once_args, **once_kwargs)
        parts = [resolve(location, replay.fn, args, kwargs) for location in locations]
        if any(part is None for part in parts):
            raise RuntimeError(f"{function[1]}'s model is not where init found it ({locations}), even after a call")
    if sizes is not None:  # the chosen sizes: the first call's own values repeated or cut to each
        from . import sizes as sizing
        texts = any(knob["kind"] == "text" for knob in knobs)
        tokenizer = sizing.tokenizer_for(replay.fn, args, kwargs) if texts else None
        ordered = [sizing.apply(replay.fn, *snapshot((args, kwargs)), knobs, setting, tokenizer) for setting in sizes]
    same = lambda a, b: repr(_plain(a)) == repr(_plain(b))
    also = [[loc for loc in held_at(part, replay.fn, args, kwargs) if not same(loc, location)]
            for location, part in zip(locations, parts)]
    return Request(replay.fn, ordered, parts, locations, also)


def _plain(location):
    """A location as nested lists (JSON gives lists, a search gives tuples)."""
    return [list(location[0]), [list(step) for step in location[1]]]
