"""Watch an app use its model: run the user's own script and record the model
calls it makes, or run it until it first calls one part of the model and take
that part. Any MLX model, from any library or the user's own code: a part is
any mlx.nn.Module the script calls.

`backcountry init my_app.py` records (record()); the model file it writes
returns the chosen part with capture() and passes the recorded inputs.
"""

from __future__ import annotations

import functools
import inspect
import runpy
import sys
import threading
import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


class Captured(Exception):
    def __init__(self, module):
        super().__init__("captured")
        self.module = module


@dataclass
class Part:
    """One part of the model the script called (outermost calls only: a
    transformer, not each of its blocks)."""
    module: str                 # the class's module, e.g. mflux.models.flux2...
    qualname: str               # the class, e.g. Flux2Transformer
    instance: int               # which instance of the class, in the order they were first called
    calls: int = 0
    seconds: float = 0.0
    shapes: dict = field(default_factory=dict)    # (inputs' names, shapes, dtypes; options) -> times called
    inputs: dict = field(default_factory=dict)    # the same key -> the first inputs, in call order
    parameters: list = field(default_factory=list)  # the call's parameter names, in order
    unsupported: str | None = None                # why the engine cannot pass these inputs
    language_model: bool = False                  # an MLX-LM model called with a cache
    threads: dict = field(default_factory=dict)   # thread id -> calls made on it (an engine's worker)


def _flatten(value, out):
    if isinstance(value, mx.array):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _flatten(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _flatten(v, out)
    return out


def _plain(value) -> bool:
    """A plain Python option: a flag, a count, a mode name, or a tuple of them."""
    if isinstance(value, (list, tuple)):
        return all(_plain(v) for v in value)
    return value is None or isinstance(value, (bool, int, float, str))


def positional(function, module, args, kwargs):
    """The call's array inputs by parameter name, in the signature's order, and
    its plain options ({"training": False}), or (None, None, why) when an input
    is neither (a cache, a config object)."""
    try:
        signature = inspect.signature(function)
        bound = signature.bind(module, *args, **kwargs)
    except TypeError as error:
        return None, None, f"its call does not bind to its signature ({error})"
    arrays, options = {}, {}
    for name, value in list(bound.arguments.items())[1:]:  # drop self
        if isinstance(value, mx.array):
            arrays[name] = value
        elif _plain(value):
            options[name] = value
        else:
            return None, None, f"its input {name!r} is a {type(value).__name__}, not an array or a plain option"
    if any(p.kind is p.POSITIONAL_ONLY for p in list(signature.parameters.values())[1:]) and options:
        return None, None, "its inputs are positional-only, so its options cannot be passed by name"
    return arrays, options, None


class Recorder:
    """Patches each mlx.nn.Module class the first time one of its instances is
    made, so calls to it are seen; uninstall() puts every class back."""

    def __init__(self, capture: tuple[str, str, int] | None = None):
        self.capture = capture
        self.parts: dict[int, Part] = {}
        self.order: dict[tuple[str, str], list[int]] = {}
        self.loads: list[str] = []     # mlx_lm.load() paths, to name a language model
        self.objects: dict[int, object] = {}  # id -> the part itself, to find where a marked function holds it
        self._local = threading.local()
        self._patched: dict[type, object] = {}
        self._init = None

    def install(self):
        self._init = nn.Module.__init__
        recorder = self
        # a compiled step cannot be timed or have its inputs kept from inside
        # (mflux compiles its denoising step): while watching, mx.compile
        # returns the function as it is, so every call runs eagerly
        self._compile = mx.compile

        def compile(fun=None, *args, **kwargs):
            return fun if fun is not None else (lambda f: f)

        mx.compile = compile

        def init(module, *args, **kwargs):
            recorder._init(module, *args, **kwargs)
            recorder._patch(type(module))

        nn.Module.__init__ = init
        try:  # name a language model: MLX-LM's load() is given its id or folder
            import mlx_lm
            import mlx_lm.utils
            for owner in (mlx_lm, mlx_lm.utils):
                original = getattr(owner, "load", None)
                if original is not None and not getattr(original, "_watched", False):
                    owner.load = self._watch_load(original)
                    self._patched[owner] = ("load", original)
        except ImportError:
            pass
        return self

    def _watch_load(self, original):
        @functools.wraps(original)
        def load(path, *args, **kwargs):
            self.loads.append(str(path))
            return original(path, *args, **kwargs)
        load._watched = True
        return load

    def uninstall(self):
        if self._init is not None:
            nn.Module.__init__ = self._init
            mx.compile = self._compile
        for owner, saved in self._patched.items():
            if isinstance(saved, tuple):
                setattr(owner, saved[0], saved[1])
            else:
                owner.__call__ = saved

    def _patch(self, cls):
        for klass in cls.__mro__:
            if klass is nn.Module or klass is object or klass in self._patched or "__call__" not in klass.__dict__:
                continue
            original = klass.__dict__["__call__"]
            self._patched[klass] = original
            klass.__call__ = self._wrap(original)

    def _wrap(self, original):
        recorder = self

        @functools.wraps(original)
        def call(module, *args, **kwargs):
            if getattr(recorder._local, "depth", 0):
                return original(module, *args, **kwargs)
            part = recorder._part(module)
            if recorder.capture and (part.module, part.qualname, part.instance) == tuple(recorder.capture):
                raise Captured(module)
            recorder._local.depth = 1
            try:
                mx.eval(_flatten((args, kwargs), []))
                start = time.perf_counter()
                out = original(module, *args, **kwargs)
                mx.eval(_flatten(out, []))
                elapsed = time.perf_counter() - start
            finally:
                recorder._local.depth = 0
            recorder._record(part, original, module, args, kwargs, elapsed)
            return out

        return call

    def _part(self, module) -> Part:
        if id(module) not in self.parts:
            key = (type(module).__module__, type(module).__qualname__)
            seen = self.order.setdefault(key, [])
            seen.append(id(module))
            self.parts[id(module)] = Part(key[0], key[1], len(seen) - 1)
            self.objects[id(module)] = module
        return self.parts[id(module)]

    def _record(self, part, original, module, args, kwargs, elapsed):
        part.calls += 1
        part.seconds += elapsed
        thread = threading.get_ident()
        part.threads[thread] = part.threads.get(thread, 0) + 1
        if part.module.startswith("mlx_lm.models.") and "cache" in kwargs:
            part.language_model = True
        if part.language_model:  # its tokens' shape is what init needs: the prompt lengths and replies
            tokens = args[0] if args and isinstance(args[0], mx.array) else kwargs.get("inputs")
            if isinstance(tokens, mx.array):
                arrays, options = {"inputs": tokens}, {}
            else:
                return
        else:
            arrays, options, why = positional(original, module, args, kwargs)
            if arrays is None:
                part.unsupported = part.unsupported or why
                return
            part.parameters = list(inspect.signature(original).parameters)[1:]
        key = (tuple((name, tuple(a.shape), str(a.dtype).removeprefix("mlx.core.")) for name, a in arrays.items()),
               repr(options))
        part.shapes[key] = part.shapes.get(key, 0) + 1
        if key not in part.inputs:
            # evaluated here, on the calling thread's stream: an engine that runs
            # its model on a worker thread (LM Studio) leaves arrays the main
            # thread cannot evaluate
            part.inputs[key] = [mx.array(a) for a in arrays.values()]
            mx.eval(part.inputs[key])


def run_script(script: str, argv: list[str], recorder: Recorder | None, main: bool = True, then=None):
    """Run the script as `python script argv...` would, with the recorder on
    (if one is given). main=False runs it as an import does: its
    `if __name__ == "__main__":` block (a server's loop) is left out. then():
    run after the script, the recorder still on (a request the user gave)."""
    saved = sys.argv
    sys.argv = [script, *argv]
    # `python app.py` imports from the app's folder (its own modules); it stays
    # on the path, since a replayed function may import lazily
    import os
    folder = os.path.dirname(os.path.abspath(script))
    if folder not in sys.path:
        sys.path.insert(0, folder)
    if recorder is not None:
        recorder.install()
    try:
        try:
            runpy.run_path(script, run_name="__main__" if main else "__backcountry_app__")
        except SystemExit as end:
            # `sys.exit(main())` with main() returning 0 is how many scripts end:
            # a clean end, not an error (a nonzero code is one, and goes on up)
            if end.code not in (None, 0):
                raise
        if then is not None:
            then()
    finally:
        if recorder is not None:
            recorder.uninstall()
        sys.argv = saved


def capture(script: str, argv: list[str], part: tuple[str, str, int]):
    """Run the script until it first calls `part` (class module, class name,
    instance), and return that module: the script's own loading code builds it.
    A model file's build() for a part watched in the user's app."""
    recorder = Recorder(capture=part)
    try:
        run_script(script, argv, recorder)
    except Captured as caught:
        return caught.module
    raise RuntimeError(f"{script} finished without calling {part[1]} (instance {part[2]}): "
                       "it must run the model the way the recording saw")
