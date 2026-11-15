"""Any function an application calls the model through, found faster: the
loop search's general drop-in.

An engine's own generation function (LM Studio's mlx-engine create_generator,
mlx-lm's stream_generate, mlx-vlm's, an image pipeline's) is named as its
target, "module:qualname". The model file's run(model, *inputs) is the recipe
that calls it the way the application does; the job runs that recipe with the
target answered by a candidate with the same signature, checks every output
against the original's and times it. A source for a target starts with the
line "# target: module:qualname" and defines a function of that name (and
optionally setup(model, helpers)).

A user gets the found function as `fn = backcountry.faster(fn)`: the newest
stored result for that target and the model it is handed (found inside its
arguments, as an engine's kit holds it), checked with the libraries installed
here; anything else runs the original. The first call on a model also installs
the stored kernels made for its layers, so an engine that loads its own model
needs no other line.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import types
import warnings
from contextlib import contextmanager

import mlx.nn as nn

from .tune import is_marked, live, model_finder, written_as

TARGET = "# target: "
NEVER = 1 << 62  # no position: the exact-rows switch waits for the first output


def target_of(source: str) -> str | None:
    """The target a loop's source names on its first line, or None."""
    first = source.lstrip().split("\n", 1)[0]
    return first[len(TARGET):].strip() if first.startswith(TARGET) else None


def resolve(target: str):
    """(owner, attribute, function) for "module:qualname": the object the name
    lives on (a module or a class) and the function there. A marked function
    ("<file>:<qualname>", tune.marked_target) is the one this process runs
    from that file; its owner is the namespace it reads its globals from."""
    module, _, qualname = target.partition(":")
    if is_marked(target):
        fn, _ = live(target)
        owner = _Globals()
        owner.__dict__ = fn.__globals__
        return owner, qualname.split(".")[-1], fn
    owner = importlib.import_module(module)
    *path, name = qualname.split(".")
    for part in path:
        owner = getattr(owner, part)
    return owner, name, getattr(owner, name)


class _Globals:
    """A function's globals as an object (vars() of it is that dict)."""


def find_for(target: str):
    """How a call of target finds its model: a marked function where init saw
    it; any other among its arguments (model_in)."""
    return model_finder(target) if is_marked(target) else model_in


def default_source(target: str) -> str:
    """The search's starting point: the original, unchanged."""
    if is_marked(target):  # a script's function cannot be imported: its own source, the mark left off
        return f"{TARGET}{target}\n" + written_as(live(target)[0])
    module, _, qualname = target.partition(":")
    head = qualname.split(".")[0]
    return (f"{TARGET}{target}\nfrom {module} import {head}  # the original; its source is in the briefing\n"
            + ("" if "." not in qualname else f"{qualname.split('.')[-1]} = {qualname}\n"))


def load_function(source: str, name: str = "candidate_function"):
    """(function, setup or None) a target's source defines, in a module of its own."""
    target = target_of(source)
    attribute = target.partition(":")[2].split(".")[-1]
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    base = live(target)[0].__globals__ if is_marked(target) else None
    if base is not None:  # a marked function reads the app's globals (its model, tokenizer, imports)
        module.__dict__.update({k: v for k, v in base.items() if k not in ("__name__", "__file__")})
    before = dict(module.__dict__)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    fn = module.__dict__.get(attribute)
    if not callable(fn):
        raise ValueError(f"the source must define {attribute}, called as {target} is")
    setup = module.__dict__.get("setup")
    if setup is not None and not callable(setup):
        raise ValueError("setup must be a function")
    if base is not None:
        own = {k for k, v in module.__dict__.items() if k not in before or before[k] is not v}
        fn = _reading(fn, module.__dict__, base, own)
        setup = setup if setup is None or "setup" not in own else _reading(setup, module.__dict__, base, own)
    return fn, setup


def _reading(fn, space: dict, base: dict, own: set):
    """fn, reading the app's globals as they are at each call (the app may
    rebind one between calls, and a replay swaps in its own copy of the
    model), its source's own names kept."""
    @functools.wraps(fn)
    def call(*args, **kwargs):
        space.update({k: v for k, v in base.items() if k not in own and k not in ("__name__", "__file__")})
        return fn(*args, **kwargs)
    return call


def model_in(args, kwargs) -> nn.Module | None:
    """The model a call works on: the first nn.Module among its arguments, or
    held one attribute deep (an engine's kit: kit.model)."""
    values = [*args, *kwargs.values()]
    for value in values:
        if isinstance(value, nn.Module):
            return value
    for value in values:
        for attribute in vars(value).values() if hasattr(value, "__dict__") else ():
            if isinstance(attribute, nn.Module):
                return attribute
    return None


def call_found(fn, args, kwargs, helpers: dict | None, find=None):
    """A found function's call, the one way the job checks and times it and a
    user's call runs it: helpers (SHIPPED and its setup's own) when it takes
    them, with the exact-rows switch installed for the model in its arguments,
    so the several-token calls it makes inside exact_rows.checking() (its guess
    checks, on whatever thread the engine runs the model) get one-token
    arithmetic per token and equal plain decoding. Nothing else changes: the
    engine reads its prompt its own way."""
    from .exact_rows import generating
    model = (find or model_in)(args, kwargs)
    if helpers is not None:
        kwargs = {**kwargs, "helpers": helpers}
    if model is None:
        return fn(*args, **kwargs)
    with generating(model, NEVER):
        result = fn(*args, **kwargs)
    if not inspect.isgenerator(result):
        return result

    def outputs():
        with generating(model, NEVER):
            yield from result
    return outputs()


@contextmanager
def answering(target: str, answer, *callers):
    """While active, `target` is answered by `answer`: on the object it lives
    on, and in the globals of each caller (a model file's run() that imported
    it by name), wherever they hold the original."""
    owner, attribute, original = resolve(target)
    if is_marked(target):  # its replay calls the function it holds (tune.Request): answered there
        from .tune import _answers
        _answers[original] = answer
        try:
            yield original
        finally:
            _answers.pop(original, None)
        return
    swapped = [(owner.__dict__ if isinstance(owner, type) else vars(owner), attribute)]
    for caller in callers:
        names = getattr(caller, "__globals__", {})
        swapped += [(names, k) for k, v in list(names.items()) if v is original]
    saved = [(space, key, space[key]) for space, key in swapped]
    try:
        for space, key in swapped:
            if isinstance(owner, type) and space is owner.__dict__:
                setattr(owner, key, answer)
            else:
                space[key] = answer
        yield original
    finally:
        for space, key, value in saved:
            if isinstance(owner, type) and space is owner.__dict__:
                setattr(owner, key, value)
            else:
                space[key] = value


def function_run(target: str, fn, setup=None):
    """The job's run(model, *inputs, helpers) for a found function: the model
    file's run() (helpers["app_run"]) with target answered by fn. The function
    sees what it will have where it ships: SHIPPED and its setup()'s entries."""
    from .generation import SHIPPED, _takes_helpers
    takes = _takes_helpers(fn)

    def run(model, *inputs, helpers):
        own = {k: helpers[k] for k in SHIPPED if k in helpers}
        own.update(run.__dict__.get("_setup_done", {}).get(id(model), {}))

        def answer(*args, **kwargs):
            return call_found(fn, args, kwargs, own if takes else None, find_for(target))
        with answering(target, answer, helpers["app_run"]):
            return helpers["app_run"](model, *inputs)

    shipped_setup = None if setup is None else \
        (lambda model, helpers: setup(model, {k: helpers[k] for k in SHIPPED if k in helpers}))
    run.__dict__.update(setup=shipped_setup, read=None, target=target, function=fn)
    return run


def drop_in(original, fn, setup=None, *, made_for: str | None = None, helpers: dict | None = None, find=None):
    """original, found faster: fn for a call on the model it was made for
    (a model_signature), set up once per model; any other call, or a setup
    that fails here (a drafter not on this Mac), runs the original. find:
    how a call finds its model (find_for), else among its arguments."""
    find = find or model_in
    from .generation import _takes_helpers, model_signature
    takes = _takes_helpers(fn)
    prepared: dict[int, dict | None] = {}

    def ready(model) -> bool:
        if id(model) not in prepared:
            if made_for is not None and model_signature(model) != made_for:
                prepared[id(model)] = None
                return False
            try:
                extra = setup(model, dict(helpers or {})) if setup is not None else {}
                if not isinstance(extra, dict):
                    raise ValueError("setup() must return a dict of helpers")
                prepared[id(model)] = {**(helpers or {}), **extra}
            except Exception as error:
                warnings.warn(f"the faster {original.__qualname__} could not set up ({type(error).__name__}: "
                              f"{error}); the original runs", stacklevel=3)
                prepared[id(model)] = None
        return prepared[id(model)] is not None

    @functools.wraps(original)
    def answer(*args, **kwargs):
        model = find(args, kwargs)
        if model is None or not ready(model):
            return original(*args, **kwargs)
        return call_found(fn, args, kwargs, prepared[id(model)] if takes else None, find)

    answer.prepare = lambda model: model is not None and ready(model)
    return answer


_stored: dict[tuple[str, int], object] = {}
_kernels_done: set[int] = set()  # models faster() has given their stored kernels (compare() adds its own)
_stock = False  # compare() times the original beside a found function
_ran = 0        # calls a found function took (compare() checks its function reached one)


def faster(fn):
    """fn, or the faster function a finished job found for it: for each call,
    the newest result in the local store made for fn ("module:qualname") and
    the model among the call's arguments, checked with the libraries
    installed here. Anything else, or a result that cannot set up here, runs
    fn itself, unchanged."""
    target = f"{fn.__module__}:{fn.__qualname__}"

    @functools.wraps(fn)
    def answer(*args, **kwargs):
        global _ran
        model = model_in(args, kwargs)
        if model is not None and not _stock and id(model) not in _kernels_done:
            _install_kernels(model)
        found = None if model is None or _stock else stored_for(target, model)
        if found is None:
            return fn(*args, **kwargs)
        _ran += 1
        return found(*args, **kwargs)

    answer.target = target
    return answer


def _install_kernels(model) -> None:
    """The stored kernels made for this model's layers, installed in place
    once: an engine that loads its own model (LM Studio's kit) gets them
    through the function it calls, with no apply() of its own. A model no
    stored kernels fit is left as it is, silently."""
    from .apply import apply
    _kernels_done.add(id(model))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="no artifact in")
        apply(model)  # in place for layers; a whole-model wrapper it would return cannot replace the caller's


def stored_for(target: str, model):
    """The stored drop-in for target made for this model, or None."""
    key = (target, id(model))
    if key not in _stored:
        _stored[key] = _load_stored(target, model)
    return _stored[key]


def _load_stored(target: str, model):
    import importlib.util
    import re
    from .apply import artifacts_home, changed_libraries
    from .generation import model_signature
    home = artifacts_home()
    loops = [d / "model_loop.py" for d in home.iterdir() if not d.name.startswith(".")] if home.is_dir() else []
    signature = None
    for path in sorted((p for p in loops if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True):
        text = path.read_text()
        if f"TARGET = {target!r}" not in text:
            continue
        made_for = re.search(r"made_for='([0-9a-f]{64})'", text)
        signature = signature or model_signature(model)
        if made_for is None or made_for.group(1) != signature:
            continue
        if changed := changed_libraries(path.parent):
            warnings.warn(f"{path.parent.name}: its faster {target} was checked with other library versions "
                          f"({'; '.join(changed)}): the original runs", stacklevel=4)
            continue
        spec = importlib.util.spec_from_file_location(f"stored_function_{path.parent.name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.answer
    return None
