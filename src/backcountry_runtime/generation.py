"""MLX-LM's generation as the loop the job searches: a source that defines
generate_step with MLX-LM's own signature and yields, faster.

The search starts from MLX-LM's generate_step itself and keeps its contract:
the same arguments (sampler, logits processors, prompt cache, prefill step
size, progress callback), the same (token, logprobs) per step, the same cache
afterwards. A source may also define setup(model, helpers) -> dict, run once
per model before its first call; when its generate_step takes a helpers
argument, setup's entries reach it there. A user gets the found function
three ways, like MLX-LM's own (drop_ins): generate_step, and stream_generate
and generate, which are MLX-LM's own functions calling it.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import types
import warnings

import mlx.core as mx
import mlx.nn as nn

# The default: MLX-LM's generate_step, unchanged.
DEFAULT = '''from mlx_lm.generate import generate_step  # MLX-LM's own; its source is in the briefing
'''

# The arguments a found generate_step is checked with; a call that sets any
# other (a quantized or rotating KV cache, input embeddings) runs MLX-LM's own.
COVERED = ("max_tokens", "sampler", "logits_processors", "prompt_cache", "prefill_step_size",
           "prompt_progress_callback")

# The helpers a found generate_step and its setup() get, in the job and
# wherever it is used (drop_ins): the models beside it and the model's
# tokenizer, both pinned in the bundle. Nothing the job alone has.
SHIPPED = ("assets", "tokenizer")


def is_generation(source: str) -> bool:
    """Whether a loop's source is a generate_step (else a run())."""
    return "def run(" not in source and ("def generate_step(" in source or "import generate_step" in source)


def load_generation(source: str, name: str = "candidate_generation"):
    """(generate_step, setup or None) a source defines, in a module of its own."""
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    step = module.__dict__.get("generate_step")
    if not callable(step):
        raise ValueError("the source must define generate_step(prompt, model, **kwargs) as MLX-LM's is")
    setup = module.__dict__.get("setup")
    if setup is not None and not callable(setup):
        raise ValueError("setup must be a function")
    if module.__dict__.get("read") is not None:
        raise ValueError("a generate_step reads its prompt itself (prompt_cache holds any earlier one); "
                         "read() is for run() loops")
    return step, setup


def _takes_helpers(step) -> bool:
    try:
        return "helpers" in inspect.signature(step).parameters
    except (TypeError, ValueError):
        return False


def greedy(logprobs):
    """Greedy decoding passed as a sampler function, as mlx-lm's server and
    engines (LM Studio's mlx-engine) pass it at temperature 0."""
    return mx.argmax(logprobs, axis=-1)


def call_arguments(settings: dict) -> dict:
    """generate_step's keyword arguments for one checked call, from plain
    settings (what a bundle's check stores): greedy as a sampler function,
    temp/top_p (with seed, set before the call), repetition_penalty,
    prefill_step_size."""
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    kwargs = {}
    if settings.get("greedy"):
        kwargs["sampler"] = greedy
    if settings.get("temp"):
        kwargs["sampler"] = make_sampler(temp=settings["temp"], top_p=settings.get("top_p", 0.0))
    if settings.get("repetition_penalty"):
        kwargs["logits_processors"] = make_logits_processors(repetition_penalty=settings["repetition_penalty"])
    if settings.get("prefill_step_size"):
        kwargs["prefill_step_size"] = settings["prefill_step_size"]
    return kwargs


def _offsets(cache) -> list[int]:
    return [c.offset for c in cache if isinstance(getattr(c, "offset", None), int)]


def _cache_digest(cache) -> mx.array:
    """Every cache entry's contents (a KV cache up to its offset, a recurrent
    state whole) as two sums over its bits: what the next turn of a chat
    reads, compared without holding a copy of the cache."""
    rows = []
    for c in cache:
        state = getattr(c, "state", None)  # a cache with no state (a test's stand-in) adds nothing
        state = state if isinstance(state, (list, tuple)) else [state]
        for a in state:
            if not isinstance(a, mx.array) or a.size == 0:
                continue
            flat = mx.contiguous(a).reshape(-1)
            bits = flat.view(mx.uint32) if flat.nbytes % 4 == 0 else flat.view(mx.uint8).astype(mx.uint32)
            weights = mx.arange(bits.size, dtype=mx.uint32) % 65521 + 1
            rows.append(mx.stack([bits.sum(), (bits * weights).sum()]))
            mx.eval(rows[-1])
    return mx.stack(rows) if rows else mx.zeros((0, 2), dtype=mx.uint32)


def generation_run(step, setup=None):
    """The job's run(model, tokens, *, helpers) for a generate_step: the
    prompt (or, with helpers["cache"] holding a conversation, its last token)
    then helpers["steps"] tokens, with helpers["call"]'s settings. Returns
    the tokens, every step's logprobs, the progress calls and the cache's
    positions and contents after the call, all compared with MLX-LM's own.
    A call with stop_after reads that many tokens and closes the generator, as
    a caller does at an end-of-reply token: the cache it leaves is what the
    next turn of a chat continues from."""
    takes = _takes_helpers(step)

    def run(model, tokens, *, helpers):
        from mlx_lm.models.cache import make_prompt_cache
        settings = helpers.get("call") or {}
        # the loop sees what it will have where it ships: SHIPPED and its setup()'s entries
        own = {k: helpers[k] for k in SHIPPED if k in helpers}
        own.update(run.__dict__.get("_setup_done", {}).get(id(model), {}))
        cache = helpers.get("cache")
        prompt = tokens.reshape(-1)
        if cache is None:
            cache = make_prompt_cache(model)
        else:
            prompt = prompt[-1:]
        progress = []
        kwargs = {**call_arguments(settings), "max_tokens": helpers["steps"], "prompt_cache": cache,
                  "prompt_progress_callback": lambda done, total: progress.append((done, total))}
        if settings.get("seed") is not None:
            mx.random.seed(settings["seed"])
        found = call_found(step, prompt, model, kwargs, own if takes else None)
        stop = settings.get("stop_after")
        steps = list(itertools.islice(found, stop) if stop else found)
        found.close()
        return {"tokens": mx.array([int(t) for t, _ in steps], dtype=mx.int32),
                "logprobs": mx.stack([lp for _, lp in steps]),
                "progress": mx.array(progress, dtype=mx.int32).reshape(-1, 2),
                "cache_offsets": mx.array(_offsets(cache), dtype=mx.int32),
                "cache_digest": _cache_digest(cache)}
    shipped_setup = None if setup is None else \
        (lambda model, helpers: setup(model, {k: helpers[k] for k in SHIPPED if k in helpers}))
    run.__dict__.update(setup=shipped_setup, read=None, generate_step=step)
    return run


def call_found(step, prompt, model, kwargs: dict, helpers: dict | None):
    """A found generate_step's call, the one way the job checks and times it
    and a user calls it (drop_ins): helpers (SHIPPED and its setup's own)
    when it takes them, and calls that read several generated tokens
    computed with one-token arithmetic per token."""
    from .exact_rows import _offset, generating
    if helpers is not None:
        kwargs = {**kwargs, "helpers": helpers}
    before = _offset(kwargs.get("prompt_cache") or []) or 0
    with generating(model, before + len(prompt) - 1):
        yield from step(prompt, model, **kwargs)


def model_signature(model) -> str:
    """What a found loop was made for, as a digest: the model's class and every
    weight's name, shape and dtype, an installed kernel wrapper's level left
    out of the names (a model after apply() is the model it was made for)."""
    inner = model
    while isinstance(inner, nn.Module) and isinstance(inner.get("wrapped") if hasattr(inner, "get") else None,
                                                     nn.Module):
        inner = inner["wrapped"]  # apply() wrapped the whole model
    rows = [type(inner).__qualname__]
    if isinstance(model, nn.Module):
        from mlx.utils import tree_flatten
        for name, a in sorted(tree_flatten(model.parameters())):
            name = f".{name}."
            while ".wrapped." in name:
                name = name.replace(".wrapped.", ".")
            rows.append(f"{name.strip('.')}:{list(a.shape)}:{a.dtype}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def rebound(fn, **names):
    """fn with some of the globals it reads replaced: MLX-LM's own code
    calling another function under the same name, nothing else changed and
    nothing patched in MLX-LM."""
    new = types.FunctionType(fn.__code__, {**fn.__globals__, **names}, fn.__name__, fn.__defaults__,
                             fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__doc__, new.__qualname__, new.__module__ = fn.__doc__, fn.__qualname__, fn.__module__
    return new


def drop_ins(step, setup=None, *, made_for: str | None = None, helpers: dict | None = None):
    """(generate_step, stream_generate, generate) for a user: MLX-LM's
    signatures and behavior, the found generate_step inside. A call that sets
    an argument outside COVERED, or a model other than the one the loop was
    made for (made_for, a model_signature), runs MLX-LM's own generate_step.
    setup runs once per model, before its first call."""
    import importlib
    lm = importlib.import_module("mlx_lm.generate")  # the module: mlx_lm.generate is also its function
    original = lm.generate_step
    params = inspect.signature(original).parameters
    takes = _takes_helpers(step)
    fits: dict[int, bool] = {}
    prepared: dict[int, dict] = {}

    def ready(model) -> bool:
        """Whether setup ran for this model; one that fails (a drafter that
        is not on this Mac) leaves the model on MLX-LM's own, with a warning."""
        if id(model) not in prepared:
            try:
                extra = setup(model, dict(helpers or {})) if setup is not None else {}
                if not isinstance(extra, dict):
                    raise ValueError("setup() must return a dict of helpers")
                prepared[id(model)] = {**(helpers or {}), **extra}
            except Exception as error:
                warnings.warn(f"the faster generate_step could not set up ({type(error).__name__}: {error}); "
                              "this model runs MLX-LM's own", stacklevel=3)
                prepared[id(model)] = None
        return prepared[id(model)] is not None

    def found(prompt, model, kwargs):
        return call_found(step, prompt, model, kwargs, prepared[id(model)] if takes else None)

    def made_for_this(model) -> bool:
        if id(model) not in fits:
            fits[id(model)] = made_for is None or model_signature(model) == made_for
        return fits[id(model)]

    def generate_step(prompt, model, **kwargs):
        made_for_this(model)
        # an unknown argument goes to MLX-LM's own too, which refuses it as it would
        unchecked = [k for k, v in kwargs.items() if k not in COVERED
                     and (k not in params or (v is not params[k].default and v != params[k].default))]
        if unchecked or not fits[id(model)] or not ready(model):
            return original(prompt, model, **kwargs)
        return found(prompt, model, kwargs)

    generate_step.__doc__ = original.__doc__
    generate_step.made_for = made_for_this  # whether a model gets the found loop
    generate_step.prepare = ready  # run setup for a model now; False when it failed
    stream_generate = rebound(lm.stream_generate, generate_step=generate_step)
    generate = rebound(lm.generate, stream_generate=stream_generate)
    return generate_step, stream_generate, generate


# MLX-LM's generation for a user, with any loop a job found for the model:
# `from backcountry import generate` in place of mlx_lm's, the same
# arguments and outputs. A model no stored loop was made for, or a call
# outside what the loop was checked with, runs MLX-LM's own.
_found: dict[int, object] = {}
_stock = False  # compare() sets it to time MLX-LM's own beside the loop
_ran = 0        # calls the stored loop took (compare() checks its function used them)


def stored(model):
    """The newest generate_step in the local store made for this model
    (artifacts_home()), or None."""
    if id(model) not in _found:
        _found[id(model)] = _stored(model)
    return _found[id(model)]


def _stored(model):
    import importlib.util
    import json
    import re
    from .apply import _version_problem, artifacts_home, changed_libraries
    home = artifacts_home()
    loops = [d / "model_loop.py" for d in home.iterdir() if not d.name.startswith(".")] if home.is_dir() else []
    signature = None
    for path in sorted((p for p in loops if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True):
        text = path.read_text()
        if "generate_step, stream_generate, generate = " not in text:
            continue  # a loop for another function (functions.faster finds those)
        made_for = re.search(r"made_for='([0-9a-f]{64})'", text)
        bundle = path.parent / "bundle.json"
        if made_for is None or _version_problem(json.loads(bundle.read_text()).get("mlx") if bundle.is_file()
                                                else None):
            continue
        signature = signature or model_signature(model)
        if made_for.group(1) == signature:
            if changed := changed_libraries(path.parent):
                warnings.warn(f"{path.parent.name}: its generation loop was checked with other library "
                              f"versions ({'; '.join(changed)}): the original runs", stacklevel=4)
                continue
            spec = importlib.util.spec_from_file_location(f"stored_loop_{path.parent.name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.generate_step
    return None


def generate_step(prompt, model, **kwargs):
    """MLX-LM's generate_step, or the faster one a job found for this model."""
    global _ran
    found = None if _stock else stored(model)
    if found is None:
        import importlib
        return importlib.import_module("mlx_lm.generate").generate_step(prompt, model, **kwargs)
    _ran += 1
    return found(prompt, model, **kwargs)


def stream_generate(model, tokenizer, prompt, **kwargs):
    """MLX-LM's stream_generate, calling generate_step above."""
    return _mlx_lm_calling("stream_generate")(model, tokenizer, prompt, **kwargs)


def generate(model, tokenizer, prompt, **kwargs):
    """MLX-LM's generate, calling generate_step above."""
    return _mlx_lm_calling("generate")(model, tokenizer, prompt, **kwargs)


def _mlx_lm_calling(name: str):
    """MLX-LM's own stream_generate or generate, calling generate_step above."""
    if name not in _calling:
        import importlib
        lm = importlib.import_module("mlx_lm.generate")
        _calling["stream_generate"] = rebound(lm.stream_generate, generate_step=generate_step)
        _calling["generate"] = rebound(lm.generate, stream_generate=_calling["stream_generate"])
    return _calling[name]


_calling: dict[str, object] = {}
