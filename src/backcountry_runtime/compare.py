"""compare(): see the speedup yourself, through your own code and inputs.

    from backcountry import compare
    model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
    print(compare(lambda m: generate(m, tokenizer, prompt, max_tokens=100), model))

The function is called with the model as loaded and with the artifact's layers
switched in (and its generation loop, when fn generates through backcountry's
generate, stream_generate or generate_step), alternately, on one copy of the weights: first once each untimed
(their outputs are compared), then `runs` timed calls each. Before every call
the random seed is set the same, so sampled outputs compare too. The model is
left as it was given; apply() it afterwards to keep the optimization.
"""

from __future__ import annotations

import statistics
import time
import warnings
from dataclasses import dataclass, field

import mlx.core as mx

from .apply import apply
from .swap import install


@dataclass
class Comparison:
    replaced: int                       # layers the artifact replaced in this model
    difference: str | None              # where the outputs first differ; None when identical
    loop: bool = False                  # a found generation loop ran in place of MLX-LM's own
    stock_s: list = field(default_factory=list)
    optimized_s: list = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return self.difference is None

    @property
    def speedup(self) -> float | None:
        if not self.stock_s or not self.optimized_s:
            return None
        return statistics.median(self.stock_s) / statistics.median(self.optimized_s)

    def __str__(self) -> str:
        if not self.replaced and not self.loop:
            return "no layer of this model is one the artifact was made for: nothing to compare"
        what = " and ".join(([f"{self.replaced} layers replaced"] if self.replaced else [])
                            + (["its generation loop"] if self.loop else []))
        outputs = "outputs identical" if self.identical else f"outputs differ: {self.difference}"
        stock, optimized = statistics.median(self.stock_s), statistics.median(self.optimized_s)
        return (f"{what}; {outputs}; {self.speedup:.3f}x faster "
                f"(as loaded {stock:.3f} s, optimized {optimized:.3f} s, median of "
                f"{len(self.stock_s)} runs each, alternated)")


def compare(fn, model, artifact_dir=None, *, runs: int = 10, seed: int = 0) -> Comparison:
    """fn(model) -> outputs: how your application uses the model. artifact_dir
    as for apply(): left out, the newest stored artifact that fits. runs: timed
    calls of each (on gpt-oss-20b, 4 read 0.97x where 10 read 0.999x)."""
    from . import functions, generation
    functions._kernels_done.add(id(model))  # compare() switches the kernels itself
    installed = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        patched = apply(model, artifact_dir, _installed=installed)
    looped = generation.stored(model) is not None
    nested = [(path, original, wrapper) for path, original, wrapper in installed if path]

    def arm(optimized: bool):
        generation._stock = functions._stock = not optimized
        for path, original, wrapper in nested:
            install(model, path, wrapper if optimized else original)
        return patched if optimized else model

    def call(optimized: bool):
        target = arm(optimized)
        mx.random.seed(seed)
        mx.synchronize()
        start = time.perf_counter()
        out = _materialized(fn(target))
        mx.synchronize()
        return out, time.perf_counter() - start

    result = Comparison(len(installed), None)
    try:
        stock, _ = call(False)
        ran = generation._ran + functions._ran
        optimized, _ = call(True)
        result.loop = generation._ran + functions._ran > ran  # a found generate_step or faster() function ran
        if not installed and not looped and not result.loop:
            return Comparison(0, None)
        if looped and not result.loop:
            warnings.warn("the faster generation loop stored for this model did not run: fn must generate "
                          "through backcountry's generate, stream_generate or generate_step", stacklevel=2)
            if not installed:
                return Comparison(0, None)
        result.difference = _difference(stock, optimized, "output")
        for i in range(runs):  # as loaded, optimized, optimized, as loaded, ...
            for optimized in ((False, True) if i % 2 == 0 else (True, False)):
                (result.optimized_s if optimized else result.stock_s).append(call(optimized)[1])
    finally:
        arm(False)
        generation._stock = functions._stock = False
    return result


def _materialized(out):
    """The outputs, evaluated: a generator is run to its end (stream_generate)."""
    if hasattr(out, "__next__"):
        out = list(out)
    arrays = []

    def visit(value):
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, (list, tuple)):
            for v in value:
                visit(v)
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif hasattr(value, "__dict__") and not callable(value):
            for v in vars(value).values():
                visit(v)

    visit(out)
    mx.eval(arrays)
    return out


def _difference(a, b, where: str) -> str | None:
    """Where two outputs first differ, or None when they are the same: arrays
    bit for bit, everything else by equality, except Python floats, which in
    an engine's results are measurements (mlx-lm's tokens per second, peak
    memory), not outputs."""
    if isinstance(a, float) and isinstance(b, float):
        return None
    if isinstance(a, mx.array) or isinstance(b, mx.array):
        if not (isinstance(a, mx.array) and isinstance(b, mx.array)):
            return f"{where} is an array in one run only"
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{where}: {a.dtype}{list(a.shape)} as loaded, {b.dtype}{list(b.shape)} optimized"
        if mx.array_equal(a, b, equal_nan=True).item():
            return None
        if mx.issubdtype(a.dtype, mx.floating):
            gap = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
            return f"{where} differs by up to {gap:.3g}"
        return f"{where} differs"
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{where} has {len(a)} items as loaded, {len(b)} optimized"
        return next((d for i, (x, y) in enumerate(zip(a, b)) if (d := _difference(x, y, f"{where}[{i}]"))), None)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{where} has other keys"
        return next((d for k in a if (d := _difference(a[k], b[k], f"{where}[{k!r}]"))), None)
    if hasattr(a, "__dict__") and type(a) is type(b) and not callable(a):
        return _difference(vars(a), vars(b), f"{where}")
    return None if a == b else f"{where}: {a!r} as loaded, {b!r} optimized"
