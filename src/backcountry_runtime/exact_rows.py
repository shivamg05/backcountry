"""Exact rows: a call over several new tokens gives each token the numbers its
own one-token call gives.

Plain greedy decoding calls the model once per generated token. A loop that
reads several new tokens in one call instead (checking guesses, re-reading
accepted tokens after a rollback) must reproduce those one-token calls bit for
bit, or tokens flip wherever the model's top two choices are nearly tied. MLX
picks other kernels for several rows (quantized matmul: qmv for one row,
qmv_wide or tiled qmm for more; attention: other plans as the query count
grows), so a stock several-token call is close but not identical.

While generating(model, start) is active, every call into the model's layers
that reads several tokens at or after position start (a position the original
loop reads one token at a time) computes its linear layers with one-row
arithmetic per row (oMLX's multi-row kernel for affine quantized weights,
otherwise one call per row) and its attention one query row at a time over
its causal prefix. Calls that start before start (the prompt, read in chunks
as the original reads it) and one-token calls are untouched, so plain decoding
is unchanged. Whether this covers a given model is measured, not assumed: the
job compares several-token calls with one-token calls under it at the start.
"""

from __future__ import annotations

import threading
import types
import weakref
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

from . import omlx_row_exact

# One switch for the process, not per thread: engines (LM Studio's mlx-engine,
# oMLX) call the model on a worker thread while the generation function that
# turned it on runs on another. They run one generation on a model at a time.
_shared = types.SimpleNamespace(owners=set(), targets=set(), start=0, on=False, depth=0, checking=0,
                                inside=0,  # owner calls running now (attention happens only inside them)
                                saved={})
_entering = threading.Lock()  # turning the switch on or off, from any thread


def _state():
    return _shared


def _rows(x) -> int:
    return x.size // x.shape[-1] if isinstance(x, mx.array) and x.ndim >= 2 else 1


def _offset(cache):
    """The position the call starts at: the first cache entry with an offset."""
    for c in (cache if isinstance(cache, (list, tuple)) else [cache]):
        offset = getattr(c, "offset", None)
        if isinstance(offset, int):
            return offset
    return None


def _decide(s, x, cache) -> None:
    """Whether this call into the layers reads several generated tokens. It
    stays set after the layers return, so the output layer that reads their
    result is covered too; the next call decides anew."""
    offset = _offset(cache)
    new = x.shape[1] if isinstance(x, mx.array) and x.ndim >= 2 else 1
    s.on = new > 1 and (s.checking > 0 or (offset is not None and offset >= s.start))


def _owner_call(original):
    def __call__(self, *args, **kwargs):
        s = _state()
        if id(self) in s.owners:
            _decide(s, args[0] if args else next(iter(kwargs.values()), None),
                    kwargs.get("cache", args[1] if len(args) > 1 else None))
            s.inside += 1
            try:
                return original(self, *args, **kwargs)
            finally:
                s.inside -= 1
        return original(self, *args, **kwargs)
    return __call__


def _linear_call(original, quantized):
    def __call__(self, x):
        s = _state()
        if s.on and id(self) in s.targets and _rows(x) > 1:
            if quantized:
                return omlx_row_exact.quantized_linear(self, x)
            flat = x.reshape(-1, 1, 1, x.shape[-1])
            out = mx.concatenate([original(self, flat[i]) for i in range(flat.shape[0])], axis=0)
            return out.reshape(*x.shape[:-1], out.shape[-1])
        return original(self, x)
    return __call__


class _TiedOutput:
    """A tied embedding's output projection (embed_tokens.as_linear: LFM2,
    Gemma, Llama 3.2 1B), seen as the linear layer it acts as."""

    def __init__(self, embedding, as_linear):
        self.embedding, self.as_linear = embedding, as_linear
        self.weight, self.scales, self.biases = embedding["weight"], embedding.get("scales"), embedding.get("biases")
        self.bits, self.group_size = getattr(embedding, "bits", None), getattr(embedding, "group_size", None)
        self.mode = getattr(embedding, "mode", "affine")

    def __contains__(self, key):  # no bias
        return False

    def __call__(self, x):
        return self.as_linear(self.embedding, x)


def _as_linear_call(original, quantized):
    def as_linear(self, x):
        s = _state()
        if s.on and id(self) in s.targets and _rows(x) > 1:
            if quantized:
                return omlx_row_exact.quantized_linear(_TiedOutput(self, original), x)
            flat = x.reshape(-1, 1, 1, x.shape[-1])
            out = mx.concatenate([original(self, flat[i]) for i in range(flat.shape[0])], axis=0)
            return out.reshape(*x.shape[:-1], out.shape[-1])
        return original(self, x)
    return as_linear


def _sdpa_call(original):
    def scaled_dot_product_attention(q, k, v, *args, scale, mask=None, **kwargs):
        L = q.shape[2]
        s = _state()
        if not (s.on and s.inside and L > 1 and (mask is None or (isinstance(mask, str) and mask == "causal"))):
            return original(q, k, v, *args, scale=scale, mask=mask, **kwargs)
        n = k.shape[2]
        return mx.concatenate([original(q[:, :, i:i + 1], k[:, :, :n - L + i + 1], v[:, :, :n - L + i + 1],
                                        *args, scale=scale, mask=None, **kwargs) for i in range(L)], axis=2)
    return scaled_dot_product_attention


def on(x=None, cache=None) -> bool:
    """Whether the call running now reads several generated tokens (and so
    runs row-exact): what kernel wrappers check before serving it. A wrapper
    with state passes its call's first input and cache: one installed at or
    above the module holding the layers runs before that module decides, so
    outside the layers the call decides here, as that module would."""
    s = _state()
    if cache is not None and s.depth and not s.inside:
        _decide(s, x, cache)
    return s.on


def layer_owners(model) -> list:
    """Modules that hold the model's layers as a list child (mlx-lm's inner
    model); every forward pass goes through one."""
    return [m for _, m in model.named_modules() if isinstance(m.children().get("layers"), list)]


# id(model) -> (weakref to it, ids and classes of its layer owners, ids of its
# linear layers): ids, not modules, so the cache never keeps a model's weights alive
_FOUND: dict = {}


def _found(model):
    """model's layer owners and linear layers, walked once per model (a walk
    of a large model's modules costs milliseconds, and every call enters here)."""
    hit = _FOUND.get(id(model))
    if hit is None or hit[0]() is not model:
        owners = layer_owners(model)
        targets = {id(m) for _, m in model.named_modules()
                   if isinstance(m, (nn.QuantizedLinear, nn.Linear, nn.QuantizedEmbedding, nn.Embedding))}
        hit = _FOUND[id(model)] = (weakref.ref(model), {id(m) for m in owners}, {type(m) for m in owners}, targets)
    return hit[1:]


@contextmanager
def checking():
    """Within generating(): every several-token call made inside this block,
    on any thread, checks generated tokens and gets one-token arithmetic per
    token. A loop that rewrites an engine's own function marks its guess
    checks this way; the engine reads its prompt outside it, its own way."""
    s = _state()
    with _entering:
        s.checking += 1
    try:
        yield
    finally:
        with _entering:
            s.checking -= 1


@contextmanager
def generating(model, start: int):
    """Row-exact several-token calls into model's layers from position start."""
    s = _state()
    with _entering:
        if s.depth == 0:
            s.owners, classes, s.targets = _found(model)
            s.saved = {cls: cls.__dict__.get("__call__") for cls in classes}
            s.saved[nn.QuantizedLinear] = nn.QuantizedLinear.__dict__["__call__"]
            s.saved[nn.Linear] = nn.Linear.__dict__["__call__"]
            s.saved_sdpa = mx.fast.scaled_dot_product_attention
            # a tied output layer reads the input embedding as a linear one
            s.saved_as_linear = {cls: cls.__dict__["as_linear"] for cls in (nn.QuantizedEmbedding, nn.Embedding)}
            for cls in classes:
                cls.__call__ = _owner_call(cls.__call__)
            nn.QuantizedLinear.__call__ = _linear_call(s.saved[nn.QuantizedLinear], True)
            nn.Linear.__call__ = _linear_call(s.saved[nn.Linear], False)
            mx.fast.scaled_dot_product_attention = _sdpa_call(s.saved_sdpa)
            for cls, original in s.saved_as_linear.items():
                cls.as_linear = _as_linear_call(original, cls is nn.QuantizedEmbedding)
        outer = (s.start, s.on)
        s.start, s.on, s.depth = start, False, s.depth + 1
    try:
        yield
    finally:
        with _entering:
            s.depth -= 1
            s.start, s.on = outer
            if s.depth == 0:
                for cls, original in s.saved.items():
                    if original is None:
                        del cls.__call__
                    else:
                        cls.__call__ = original
                mx.fast.scaled_dot_product_attention = s.saved_sdpa
                for cls, original in s.saved_as_linear.items():
                    cls.as_linear = original
                s.owners, s.targets, s.saved = set(), set(), {}
