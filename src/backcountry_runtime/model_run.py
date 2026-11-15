"""A model used through a replaceable inference loop: run(model, *inputs, helpers).

With identical: output, the job searches over this function. The default reproduces how the job
already uses the model (MLX-LM's greedy generation, the model file's own
run(), or one forward call); a candidate changes how the outputs are produced,
never what they are. A shipped bundle carries the winning run() as source and
loads it with load_loop. For MLX-LM's generation the loop is a generate_step
instead (generation.py), which load_loop calls as a run().
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import types

import mlx.core as mx
import mlx.nn as nn

# Greedy generation exactly as MLX-LM's generate_step runs it.
GENERATE = '''import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache


def run(model, prompt, *, helpers):
    """Read the prompt, then generate helpers["steps"] tokens greedily, as
    MLX-LM's generate_step does. Returns the token ids."""
    cache = make_prompt_cache(model)
    steps = generate_step(prompt.reshape(-1), model, max_tokens=helpers["steps"], prompt_cache=cache)
    return mx.array([int(token) for token, _ in steps], dtype=mx.int32)
'''

# Greedy generation after a conversation already in the model's memory: a
# decode workload's context, then helpers["steps"] tokens.
GENERATE_AFTER_HISTORY = '''import mlx.core as mx
from mlx_lm.generate import generate_step


def run(model, tokens, *, helpers):
    """helpers["cache"] already holds the model's memory of every token of
    `tokens` but the last: generate helpers["steps"] tokens greedily from the
    last one, as MLX-LM's generate_step does. Returns the token ids."""
    steps = generate_step(tokens.reshape(-1)[-1:], model, max_tokens=helpers["steps"],
                          prompt_cache=helpers["cache"])
    return mx.array([int(token) for token, _ in steps], dtype=mx.int32)
'''

# The model file's own run(model, *inputs), as the job times it.
APP_RUN = '''def run(model, *inputs, helpers):
    """The model file's run(), unchanged; its source is in the briefing."""
    return helpers["app_run"](model, *inputs)
'''

# One forward call.
FORWARD = '''def run(model, *inputs, helpers):
    """One call of the model, as the job times it."""
    return model(*inputs)
'''


def load_drafter(path: str):
    """A drafter loaded with mlx_lm, keeping only the weights the model has.
    Some checkpoints ship weights the model never reads (Gemma 4 E2B's key
    and value projections in the 20 layers that share earlier layers' KV),
    and mlx_lm refuses them; a weight the model needs and lacks still fails."""
    from mlx.utils import tree_flatten
    from mlx_lm import load

    original = nn.Module.load_weights

    def load_weights(self, weights, strict=True):
        if isinstance(weights, list):
            own = dict(tree_flatten(self.parameters()))
            weights = [(k, v) for k, v in weights if k in own]
        return original(self, weights, strict=strict)

    nn.Module.load_weights = load_weights
    try:
        return load(path)[0]
    finally:
        nn.Module.load_weights = original


def load_run(source: str, name: str = "candidate_run"):
    """The run function a source defines, in a module of its own. A source may
    also define setup(model, helpers) -> dict, run once before the first call
    (load a drafter or a multi-token head the loop uses; its entries join
    helpers), and read(model, tokens, helpers) -> dict, which reads a
    conversation into memory before the clock in place of the default read
    (its dict holds "cache", the model's own, plus anything the loop keeps,
    such as a head's memory of the conversation); both are attached to run."""
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    run = module.__dict__.get("run")
    if not callable(run):
        raise ValueError("the source must define run(model, *inputs, helpers)")
    for hook in ("setup", "read"):
        fn = module.__dict__.get(hook)
        if fn is not None and not callable(fn):
            raise ValueError(f"{hook} must be a function")
        run.__dict__[hook] = fn
    return run


def load_loop(source: str, name: str = "candidate_run"):
    """The job's run() for a loop's source: a found replacement for a named
    function (its first line "# target: module:qualname", functions.function_run),
    a generate_step (MLX-LM's generation, generation.generation_run), or a run()
    as written."""
    from .functions import function_run, load_function, target_of
    from .generation import generation_run, is_generation, load_generation
    if (target := target_of(source)) is not None:
        return function_run(target, *load_function(source, name))
    if is_generation(source):
        return generation_run(*load_generation(source, name))
    return load_run(source, name)


def asset_path(repo: str, revision: str) -> str:
    """Where a pinned model beside this one sits in the Hugging Face cache:
    the snapshot folder the job found it in, downloaded at that revision when
    it is not there (as mlx_lm.load downloads the model). A snapshot that
    lacks a file the checkpoint does not need (.gitattributes) still serves."""
    from huggingface_hub import constants, snapshot_download
    path = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not (path / "config.json").is_file():
        try:
            path = Path(snapshot_download(repo, revision=revision))
        except Exception as e:
            raise FileNotFoundError(f"{repo} at revision {revision} is not in the Hugging Face cache ({path}) "
                                    f"and did not download: {e}") from e
    return str(path)


class _Listed(dict):
    """A listed asset that notes, in its run's assets_read, when its path is
    read: what an exported bundle pins. Reading it whole (a copy, its items
    or values) counts as reading its path."""

    def __init__(self, entry, read: set):
        super().__init__(entry)
        self._read = read

    def _note(self):
        self._read.add(dict.get(self, "repo"))

    def __getitem__(self, key):
        if key == "path":
            self._note()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key == "path":
            self._note()
        return dict.get(self, key, default)

    def __iter__(self):  # a copy ({**a}, dict(a)) then reads through __getitem__
        return iter(dict.keys(self))

    def items(self):
        self._note()
        return dict.items(self)

    def values(self):
        self._note()
        return dict.values(self)


def run_helpers(run, model, helpers: dict) -> dict:
    """helpers with run's setup() entries, set up once per run and model.
    The listed assets note which ones run reads (run.assets_read)."""
    if helpers.get("assets"):
        read = run.__dict__.setdefault("assets_read", set())
        helpers = {**helpers, "assets": [_Listed(a, read) for a in helpers["assets"]]}
    setup = run.__dict__.get("setup")
    if setup is None:
        return helpers
    done = run.__dict__.setdefault("_setup_done", {})
    if id(model) not in done:
        extra = setup(model, helpers)
        if not isinstance(extra, dict):
            raise ValueError("setup() must return a dict of helpers")
        done[id(model)] = extra
    return {**helpers, **done[id(model)]}


class ModelRun(nn.Module):
    """The model called through run(). The model stays an ordinary child, so
    installed kernels and tracing see the exact model run() calls. Each call
    evaluates its outputs and waits for the GPU, so a timed call contains
    all of its work. For text, calls that read several generated tokens at
    once are computed with one-token arithmetic per token (exact_rows), so a
    loop that checks guesses that way emits exactly plain decoding's tokens."""

    def __init__(self, model, run, helpers):
        super().__init__()
        self.model = model
        object.__setattr__(self, "_run", run)
        object.__setattr__(self, "_helpers", run_helpers(run, model, helpers))

    def __call__(self, *inputs):
        from .exact_rows import generating
        helpers = self._helpers
        first = inputs[0] if inputs else None
        if helpers.get("conversation") and isinstance(first, mx.array):
            # outside a job: read the conversation first
            first = History.read(self.model, first, helpers, self._run.__dict__.get("read"))
        if isinstance(first, History):
            # run() continues a conversation already in memory
            helpers = {**helpers, **first.fresh()}
            inputs = (first.tokens, *inputs[1:])
        text = isinstance(first, History) or helpers.get("text")
        start = inputs[0].shape[-1] - 1 if text and isinstance(inputs[0], mx.array) else None
        try:
            from mlx_lm.generate import generation_stream, wired_limit
        except ImportError:
            limit = None
        else:
            limit = wired_limit(self.model, [generation_stream]) if hasattr(self.model, "layers") else None
        with contextlib.ExitStack() as stack:
            if limit is not None:
                stack.enter_context(limit)
            if start is not None:
                stack.enter_context(generating(self.model, start))
            out = self._run(self.model, *inputs, helpers=helpers)
            mx.eval(out)
        mx.synchronize()
        return out


class History:
    """A conversation already in memory: its tokens, every one but the last
    read into the model's cache (the default read, or run's own read(), which
    can also fill a drafter's or head's memory of it). A job reads it before
    any clock starts, as a decode workload's context is read, so a timed run
    is only the generating; each run gets a fresh copy of what was read."""

    def __init__(self, tokens: mx.array, state: dict):
        self.tokens = tokens
        self.state = state

    @property
    def cache(self):
        return self.state["cache"]

    @classmethod
    def read(cls, model, tokens: mx.array, helpers: dict | None = None, read=None) -> "History":
        """read: run's own read(model, tokens, helpers) -> dict holding "cache"."""
        tokens = tokens.reshape(1, -1)
        if read is None:
            return cls(tokens, {"cache": _read(model, tokens)})
        state = read(model, tokens, helpers or {})
        if not isinstance(state, dict) or "cache" not in state:
            raise ValueError('read() must return a dict holding "cache", the model\'s cache')
        mx.eval([v for v in state.values() if isinstance(v, mx.array)])
        return cls(tokens, state)

    def fresh(self) -> dict:
        """helpers entries for one run: a copy of what was read (models and
        other modules shared, not copied)."""
        from .state import _copy_cache
        return {k: v if isinstance(v, nn.Module) else _copy_cache(v) for k, v in self.state.items()}


PREFILL_STEP = 2048  # mlx-lm generate_step's prefill_step_size


def _read(model, tokens: mx.array) -> list:
    """model's cache holding every token of tokens but the last, read the way
    mlx-lm's generate_step reads a prompt before its last token: in chunks of
    PREFILL_STEP tokens."""
    from .state import _make_cache, _with_cache_work
    cache = _make_cache(model)
    for start in range(0, tokens.shape[1] - 1, PREFILL_STEP):
        chunk = tokens[:, start:min(start + PREFILL_STEP, tokens.shape[1] - 1)]
        mx.eval(_with_cache_work(model(chunk, cache=cache), cache))
    return cache
