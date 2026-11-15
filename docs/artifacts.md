# Using an artifact

When a run succeeds, it prints `verified artifact` and a folder path. That
folder is the artifact: a self-contained code bundle that makes your model use
the faster kernels. You don't need this optimizer installed to use it.

(The `checkpoints/` folder is not the artifact. Checkpoints are saved during the
search in case the run dies, and haven't passed the final checks.)

## Apply it to your model

Copy the folder into your project as `artifact/` and install its pinned
dependencies:

```sh
python -m pip install -r artifact/requirements.txt
```

Then load your model the way you normally do and patch it. For an MLX-LM model
whose `build()` returns the model directly:

```python
from mlx_lm import load, generate
from backcountry import apply

model, tokenizer = load("/path/to/my-checkpoint")  # or a compatible Hub model ID
model = apply(model, "path/to/artifact")
text = generate(model, tokenizer, prompt="Explain gravity simply.")
```

On the Mac that ran the job you can leave the folder out: every finished job
copies its artifact to `~/.backcountry/artifacts/<job folder>/` (or
`$BACKCOUNTRY_HOME/artifacts/`), and `apply(model)` uses the newest one there
whose layers your model has, checked layer by layer as below. A model none
fits runs as it is, with a warning. Artifacts exported before layers were
recorded are only used when named.

`from artifact import apply` and `apply(model)` work too, without installing
backcountry. For a custom model, build and load it the same way your
original `build()` does, then call `apply()`. **Always use the returned
model.** If your `build()` wraps the model in something, recreate that wrapper
too. Everything after `apply()` (your generation loop, batching, caches) stays
as it was.

**Each layer is checked before it is replaced.** `apply()` replaces a layer
only if it is the one the kernels were made for: the same class, weight shapes
and dtypes, and quantization, under the same MLX version, patch number
included. Any other layer keeps its own code
and a warning says which and why, so on a model the artifact doesn't fit
nothing changes. Artifacts exported before this check install as before.

**Wherever your engine keeps the layers.** An engine that holds the same model
at other addresses, such as MLX-VLM, which keeps the language model under
`language_model`, or one that hands over only the model's inner body, gets
the same layers replaced where they are, each still checked as above.

**On another chip** the kernels compute the same values, but their speed was
measured only on the chip the job ran on (recorded in `bundle.json`).
`apply()` installs them and warns; run `benchmark.py` to measure them there.

**What has to match.** The kernels were built for one exact model structure, so
the operations, module layout, weight shapes, dtypes and quantization all have
to match the model that was optimized:

- **Different weight values, same everything else** (e.g. a fine-tune that
  keeps the same layers and quantization): fine as is.
- **Different architecture or quantization** (e.g. 4-bit → 8-bit): run the
  optimizer again.

Either way, check correctness and speed on the weights and workload you'll
actually ship.

**Or let the bundle build the model.** `from artifact import load`, then use
`load().inference_model`. This runs the bundled copy of your `build()` and
gives you the patched model. It loads whatever weights `build()` picks; there's
no argument to point it at a different checkpoint. Each artifact's own
`README.md` shows both routes, with the example that fits the model (MLX-LM or
your own builder) and imports named after the bundle's folder.

### Decode (`context`) runs

If the run optimized a decode step (a workload with `context`), `load()` rebuilds
exactly the step that was measured. It's what the benchmark uses. For real
generation, where your code owns the cache and it keeps growing, use
`load().inference_model`.

A kernel only ever runs on the input shapes it was verified on, such as a
prompt chunk of exactly 2,047 tokens, unless the run used
[other sizes](manifest.md#other-sizes-where-else-the-kernels-run) (automatic for
MLX-LM prompts). Then the layers it
changed carry generated code that reads the size (`if 1000 <= _n <= 2048:`
before each kernel call, in `patch/wrappers.py`), and each kernel runs at every
size in its checked range, from an empty cache: a prompt, or the first chunk of
a longer one. That code runs as plain MLX calls, with nothing traced or
compiled at run time, and any other call goes straight to the original layer.
For runs without it, where the cache stands when a verified shape arrives
matters:

- **Kernels whose inputs come only from the layer's own input and weights**
  (e.g. a projection of the layer's input) run at any cache position,
  including the later chunks of a long prompt, as long as those inputs have a
  verified shape. A layer checks each new cache state by re-tracing it once. A
  cache that keeps a fixed size (a convolution or state-space layer's) looks
  the same at every token and is traced once. One that grows (a KV cache)
  looks new at every token, so each layer traces at most 8 new positions and
  then runs the original, rather than re-trace every token.
- **Kernels fed by the cache or by the position** (e.g. anything after
  attention reads the cache, or an input built from the cache offset) run
  only at the positions the run recorded: such an input could hold a value
  that never changed while the kernel was being checked. Elsewhere the
  original operations run in their place.
- **Scopes the tool delivered by replay** (the report names them), and
  bundles exported before this rule, serve only the recorded positions.

A block that receives the cache (an attention block, a decoder layer, a
whole model: in a prompt job, a decode run, or in generation) and carries
kernels does not keep those limits when the judge's own method for it passes:
after the search the judge rewrites that block's `__call__` to call the
kernels from the block's own code, leaving the position and cache code live.
Without it, a chat's second message, read on top of the first, gets none of
the block's kernels. It is installed only if the model then matches the
untouched one bit for bit, three times over, on a first message, 40 decode
calls and a second message read on top of them; calls a kernel found on a
prompt while every second message is read, and one found on decode on every
decode call, unless its own shape rule declines those inputs (a kernel for many
rows never takes a one-token call); and passes review. The report's `live_method` events say which
blocks got one. Its class is in
`patch/wrappers.py` like any other, and it runs at every position, with the
original operations wherever a kernel does not take the shapes. Over a range
(`range:`), the per-size step checks such a block's kernels at every size the
way it checks any other scope's, even when the block itself runs other
operations at other sizes (a whole model past its sliding window), as long as
each kernel's own operations recur there; they also run at each size where they
pass, so a call of a few new tokens on top of a conversation (a speculative
loop's check) runs them too. A kernel whose input grows with the conversation
itself (attention over the cache) is recorded at the job's context length; in a
decode job it also runs across a span of lengths, out from the job's own, where
it is bit-exact over 8 calls in a row and not slower at 10 lengths spread from 64
to 8,192 tokens, and the block check then passes with it there (the report's
`history_length` and `history_span` events).

In a job that generates over a conversation (a workload with `context:`), a
region that runs from one layer into the next (say, a layer's last matmul plus
the next layer's norm) is not searched: its only place in the model is the
model body, which is handed every layer's cache, so its kernel, and any kernel
the replay takes in, would run only at the recorded conversation length. The
same operations inside one layer are searched as usual. The report lists the
region with that reason. Within one layer, a region that sits in a block holding
the cache but never reads it (a layer's matmul plus the residual add after it)
is searched after every other region: its kernel would reach new positions only
through a rewrite of the whole block, which carries every kernel inside it and
cost one request 13% on Bonsai 2 9B, while the same matmul without the add
installs at its own scope.

The final multi-step measurement already includes those fallbacks.

## Weights and other files

**Weights are never copied into the bundle automatically**, and that includes
the recovery checkpoints. The bundled `build()` loads weights exactly as your
original did: a pretrained checkpoint comes from the usual Hugging Face cache,
download or local path, and random weights stay random.

So whatever `build()` needs has to exist on the machine where you run the
bundle: checkpoints, local files, environment settings, extra libraries.

- **Want a local data file included?** Add `ARTIFACT_FILES = [...]` to your
  model file, as a plain list of relative paths. Your model's local Python
  imports are copied automatically; other files are only bundled if they're
  listed there.
- **Checkpoint paths and adapter options** that `build()` works out at runtime
  stay as they are. Export doesn't rewrite them.

**Code dependencies.** Before loading the model, the tool checks that
everything your model file imports can be packaged. Anything outside the
standard setup must be either an installed package or source code inside your
model project.

**Checkpoint versions.** During a run, the tool records which MLX-LM checkpoint
snapshot it loaded and uses that same snapshot for every comparison copy. The
bundle notes the source and revision for reference, but it doesn't force that
revision or point at the original machine's cache. If future builds must get
exactly that revision, pin it in your model file.

## Check it yourself

Through your own code, on your own inputs: `compare(fn, model)` calls
`fn(model)` with the model as loaded and with the artifact's layers switched
in, alternately, on one copy of the weights: once each untimed, whose outputs
it compares (arrays bit for bit, other values by equality, Python floats such
as mlx-lm's tokens per second skipped), then ten timed calls each (`runs=`). The random
seed is set the same before every call, so sampled outputs compare too. It
returns the result (`identical`, `difference`, `speedup`, the times) and
leaves the model as it was given.

```python
from backcountry import compare
result = compare(lambda m: generate(m, tokenizer, prompt=my_prompt, max_tokens=100), model)
print(result)
```

The bundle also carries the exact inputs the run measured, so you can re-check
it on your own machine. From inside the artifact folder:

```sh
python validate.py     # patched vs original outputs on the saved inputs, the job's own rule
python benchmark.py    # repeat the saved final measurement; exits 1 unless a win is confirmed
```

- **`validate.py`** compares the patched model with the bundled original, using
  the run's own rule: exact match, or within the manifest's `rtol`/`atol`.
  It doesn't need any fp32 reference files.
- **`benchmark.py`** reruns the final measurement the way it was done: same
  execution mode (including compiled parts of the model) and, by default, the
  same number of repetitions the run actually used. That can be more than the
  manifest asked for, because short direct-call workloads get extra repetitions.
  Passing `--steps` changes that count, which makes it a different experiment.
  Older bundles that didn't record a count use the manifest's.
  Compiled library-inference bundles also carry the original compiled scopes
  separately from the optimized code. For manual timing, use
  `load(patched=False, measurement_baseline=True)`; `load(patched=False)`
  remains the untouched correctness reference. Older bundles missing these
  scopes need re-exporting to reproduce that compiled baseline.

Timing noise, or a different machine, can still move the number.

Keep the pinned dependencies in `requirements.txt`.

## How the bundle is checked before you get it

Export builds the bundle in a temporary folder next to the destination and only
moves it into place once it passes. If writing or checking fails, your accepted
checkpoints and any earlier artifact are left untouched. If the final move
fails, the previous artifact is restored.

The check runs in a fresh process:

- It builds the original and the patched model with the **same freshly loaded
  weights**. (Weights are shared before any decode cache is filled.) That way
  randomly initialized models compare fairly, without freezing their weights
  or comparing two unrelated random models.
- It checks every workload, every sweep size and the recorded other sizes
  in the manifest, through both
  `apply()` and the bundle's own `load()`, including the cache state for
  decode runs.
- It runs **offline**, using your existing Hugging Face cache. If something the
  model needs is missing, it reports that instead of downloading it mid-export.

If any check fails, nothing is published.

**Where it goes.** By default, `<work-dir>/artifact/`. You can pick another
destination, but it has to be new, and it can't overlap the run's logs,
kernels or checkpoints. The tool rejects a bad path before any GPU work starts.

## What's in the bundle

- Your model's source code, copied unchanged.
- `manifest.yaml`: the manifest the job ran, copied as written. Its paths point
  to where the job ran.
- The exact inputs the run measured. For decode runs, this includes the tokens
  that filled the cache, so `load()` rebuilds the same step.
- The kernels and the generated code that plugs them into your model.
- A pinned `requirements.txt`.
- `validate.py` and `benchmark.py`, so the bundle can check and re-time itself.
- A `README.md` written for that run. It opens with each workload's speedup
  and what one timed call did. In an MLX-LM chat job, a conversation workload
  generates its reply tokens after the saved conversation, and every other
  workload reads one prompt of its length to the first reply token, with no
  conversation before it. A last line names the largest confirmed speedup and
  the workload it was timed on. A found inference loop's section names the one
  conversation length its speedup was timed at. After that come three commands to install, validate and re-time the bundle, then how to
  use it in your code. A reference section below lists which modules got which
  kernel, how it was checked, and what each file is.

What each number in `report.json` means is covered in the
[usage guide](usage.md#whats-in-reportjson).
