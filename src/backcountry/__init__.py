"""Use a model backcountry optimized: load it the way you normally do, then

    from backcountry import apply
    model = apply(model)   # keep the returned model

Everything after that (generation, batching, caches) stays as it was. With no
artifact named, apply() uses the newest one a finished job left in the local
store (~/.backcountry/artifacts) that fits this model; apply(model,
"path/to/artifact") names one. A layer the artifact's kernels were not made
for keeps its own code, with a warning.

When a job found a faster generation loop for an MLX-LM model, it comes back as
MLX-LM's own functions, the same arguments and outputs: import them from here
in place of mlx_lm's,

    from backcountry import generate   # or stream_generate, generate_step

and they use the stored loop made for the model they are given (any other
model, or an argument the loop was not checked with, runs MLX-LM's own).

Any other function an application calls the model through (an engine's own
generation, an image pipeline's method), searched with `function:` in the
manifest, comes back the same way, wherever the application imports it:

    create_generator = backcountry.faster(create_generator)
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # what editors and type checkers read; at run time each name loads on first use
    from backcountry_runtime.apply import apply
    from backcountry_runtime.compare import compare
    from backcountry_runtime.functions import faster
    from backcountry_runtime.generation import generate, generate_step, stream_generate
    from backcountry_runtime.tune import tune

# each name from the runtime module that defines it, imported on first use: the
# command line and the job's own processes (backcountry.harness) need none of them
_FROM = {"apply": "apply", "compare": "compare", "faster": "functions", "generate": "generation",
         "generate_step": "generation", "stream_generate": "generation", "tune": "tune"}
__all__ = list(_FROM)


def __getattr__(name):
    if name not in _FROM:
        raise AttributeError(f"module 'backcountry' has no attribute {name!r}")
    value = getattr(import_module(f"backcountry_runtime.{_FROM[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *__all__])
