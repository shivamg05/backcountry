# Write a manifest

A manifest is a small YAML file that tells the tool three things: **which
model**, **what inputs you want to get faster**, and **how many attempts it
gets**. One manifest = one optimization run.

The quickest way in is to copy the example below, save it as `my_run.yaml` beside
your model file, and change the model and prompt length. Everything else can stay
as is.

Jump to: [decode](#measure-one-decode-step) ·
[other models](#measure-a-model-call) ·
[several input sizes](#optimize-several-input-sizes) ·
[field reference](#field-reference)


**Most runs don't need a manifest written by hand.** `backcountry init`
writes a short `backcountry.yaml` (model, what to make faster, sizes, what to
keep identical, attempts), and each run turns it into the full manifest
described here, in its own folder. Write a full manifest when you need what the
short file can't say: several kinds of call, your own limits, extra sizes, an
engine's own function.

## Start here: a language model

This one asks: *how fast can Llama read a 512-token prompt and produce its
first token?* You don't supply a prompt: the tool cuts one from real text
(chat in several languages, web pages, books, math problems and code) with the
model's own tokenizer, which it picks up when the model file loads the model
through MLX-LM or MLX-VLM. If the tokenizer has a chat template, the prompt is
a chat request in that format, ending where the model's reply starts: reply to
a conversation's last message, solve a math problem, explain some code, or
summarize a page. The context in front of the request is cut from its start
so the prompt has exactly the length the workload asks for.

```yaml
model: llama8b.py               # Python file that loads the model
model_type: llm                 # what kind of model: llm, moe-llm, vlm, vla, diffusion or other
use_library_inference: true     # time MLX-LM's own generate path, uncompiled as MLX-LM runs it
workloads:
  - name: first_token           # just a label for the report
    inputs:
      - shape: [1, 512]         # one prompt, 512 tokens
        dtype: int32            # token IDs are integers
        low: 0
        high: 128000            # random IDs are drawn from 0..127999
budget:
  per_region: 8                 # max attempts on any one part of the model
  total: 24                     # max attempts for the whole run
final_benchmark:
  steps: 1                      # generate one token per request
```

Run it:

```sh
backcountry run my_run.yaml --judge claude-cli
```

Results land in `runs/`. **That's enough to start a run.** The rest of this
page is for other goals and details.

A few things that trip people up:

- **`model` is relative to the YAML file**, not to where you run the command.
  See [write a model file](usage.md#1-provide-a-model).
- **`dtype: int32` is the input type, not the model's precision.** Token IDs
  are always integers. This Llama file loads 4-bit weights; precision and
  quantization live in the model file.
- **Keep `high` at or below the model's vocabulary size** when you switch
  models, or the random IDs will be out of range.
- **A "region" is a chunk of the model** the tool tries to replace with a
  faster GPU kernel. The budget counts attempts, not minutes or tokens.

**What gets timed:** the full request from prompt to output token, including
sampling and any extra work MLX-LM queues up. Loading weights and tokenizing
text are not timed.

Set `steps: 32` to time a whole 32-token request instead. That includes reading
the prompt, so the tokens/sec it reports isn't a pure decode speed. For that,
see the next section.

`use_library_inference: true` only works for complete MLX-LM language models
with one integer input shaped `[T]` or `[1, T]`. If your model doesn't fit, the
tool says so before the search starts. Use a
[direct model call](#measure-a-model-call) instead.

## Measure one decode step

Use this to time how long it takes to process *one new token* after the model
has already read 512. Those 512 tokens live in the **cache**, the model's saved
work from earlier tokens. The tool fills the cache before timing and resets it
between every comparison, so both versions start from the same place.

```yaml
model: llama8b.py
model_type: llm
baseline: compiled
use_library_inference: false    # call the model directly, no sampling
workloads:
  - name: decode
    context: 512                # tokens already in the cache
    inputs:
      - shape: [1, 1]           # one new token
        dtype: int32
        low: 0
        high: 128000
budget: {per_region: 8, total: 24}
final_benchmark: {steps: 1, pairs: 4, warmup_steps: 3}
```

With `steps` above 1, the final benchmark calls the model that many times in a
row, feeding the same token each time and letting the cache grow. Copying the
cache into place is part of that timing.

A workload with a `context` has one integer input, and the model has to know
how to create and update a cache; the bundled MLX-LM models do. A manifest can
have **one** such workload (one conversation, the model's one cache) beside any
number of workloads without one, so prefill and decode are searched, checked
and shipped together. Each runs the way the model is used, on the same model:

- **`use_library_inference: true`** (the default for MLX-LM models): MLX-LM's
  own generation. A prefill workload is a prompt read to its first token, as
  MLX-LM reads one (2,048 tokens at a time, then the last token alone); the
  decode workload is `steps` tokens generated after the conversation.
- **A `run()` in the model file:** `run()` itself, handed the conversation
  ahead of the workload's own tokens (it reads it, as an engine does).
- **`use_library_inference: false`:** direct calls: the decode workload one
  call over the cached conversation, a prefill workload one call from no cache.

```yaml
use_library_inference: true
workloads:
  - name: prefill         # reading a first message, 64 to 2,049 tokens
    inputs: [{shape: [1, T], dtype: int32, low: 0, high: 151643}]
  - name: decode          # 64 tokens after 512 tokens of conversation
    context: 512
    inputs: [{shape: [1, 1], dtype: int32, low: 0, high: 151643}]
primary: {T: 1024}
range: {T: [64, 2049]}
final_benchmark: {steps: 64}   # the decode workload's tokens; a prefill is one read per use
```

A kernel is kept only if it is right on every workload, faster on the one it
was found for, and slower on none.

## Measure a model call

For anything that isn't MLX-LM text generation, set
`use_library_inference: false` and the tool times a plain call to the model
(a "forward pass"). This example uses the small bundled model. It needs no
download and takes 128 rows of 256 numbers:

```yaml
model: tiny_mlp.py
model_type: other
baseline: compiled
use_library_inference: false
workloads:
  - name: forward
    inputs:
      - shape: [128, 256]
        dtype: float32
budget: {per_region: 4, total: 8}
final_benchmark: {steps: 10, pairs: 4, warmup_steps: 3}
```

This times the model call and nothing around it. For an image denoiser, that's
denoiser calls, not the whole image pipeline. For a speech encoder, it's encoder
calls, not audio preprocessing or transcription. For a model with several
inputs, see [the FLUX manifest](../manifest_flux.yaml).

## Describe your inputs

Each item under `inputs` is one positional argument, in order. Three items
means the tool calls `model(a, b, c)`, so shapes and dtypes have to match what
your model expects. What each dimension *means* (batch, sequence length, …) is
up to the model; the tool doesn't interpret it.

The tool makes up the input values, except a language model's prompt: when the
model file loads its tokenizer through MLX-LM or MLX-VLM and the first input is
token ids (a 32- or 64-bit integer array), that input is real text. For integer
inputs, `low` and `high` set
the range: `low` is included, `high` isn't. They don't apply to float inputs.

## Optimize several input sizes

Add one workload per input case you care about. This one targets a short and a
long prompt:

```yaml
model: llama8b.py
model_type: llm
baseline: compiled
use_library_inference: true
workloads:
  - name: short_prompt
    inputs: [{shape: [1, 128], dtype: int32, low: 0, high: 128000}]
  - name: long_prompt
    inputs: [{shape: [1, 512], dtype: int32, low: 0, high: 128000}]
budget: {per_region: 8, total: 24}
final_benchmark: {steps: 1, pairs: 4, warmup_steps: 3}
```

How this plays out:

- Each attempt aims at one workload. It's kept only if it's clearly faster there
  **and** not clearly slower on any other.
- The final result has to speed up at least one workload and slow down none,
  after allowing for timing noise. Gains aren't averaged, so a big win on one
  can't hide a loss on another.
- More workloads means more checking and timing per attempt, and they all share
  the one `total` budget.

Workload names are only labels. Calling one `decode` doesn't make it a decode
measurement; `context` and `use_library_inference` do that.

## Optional: check extra sizes for correctness

Skip this unless you want the result checked at sizes you aren't optimizing.
Give a dimension a name, like `B`, then say which size to optimize and which
sizes to test:

```yaml
model: tiny_mlp.py
model_type: other
baseline: compiled
use_library_inference: false
workloads:
  - name: forward
    inputs: [{shape: [B, 256], dtype: float32}]
primary: {B: 128}                 # optimize for batch 128
sweep: {B: [1, 32, 128, 256]}     # also check correctness at these sizes
budget: {per_region: 4, total: 8}
```

The sweep only checks correctness. It doesn't make those sizes faster, and it
doesn't promise the new kernel runs there. At any size where a kernel hasn't
been verified, the model just uses its original code. To make a range of sizes
faster, see the next section.

## Other sizes: where else the kernels run

A range of sizes changes the search itself. Besides the optimized size it
runs at nine more spread over the range: the range is cut into ten bins, each
about the same factor wider than the last, and a random size is drawn in each
bin (the optimized size keeps its own). Every kernel is then checked correct
and timed at all ten while it is written, and the judge sees a table of how it
did at each. A kernel goes on to the whole-model check only when it is faster
at half of them or more and slower at none. It may leave the ends of the range
to the library (its fallback predicate true on the smallest sizes, the
largest, or both) as long as its own code runs one unbroken stretch of at
least 7 of the 10: a tile built for long prompts need not beat MLX on the
shortest. Those sizes count toward the half without being wins, run MLX's own
code in the shipped model, and are still checked for correctness; the
optimized size always runs the kernel. The whole-model check runs the whole model at
all ten for correctness and times three of the sizes the kernel runs (the
shortest, the longest, and the one where the kernel was weakest), and it ships only with no clear slowdown
and clear wins at two of them; the timing stops at the first clear slowdown,
which no later size can undo. A size too close to the Mac's GPU memory limit
to time with the job around it is checked but never timed
([measurement.md](measurement.md)). Speed changes with the size, so a kernel fitted
to one size (Laya's tiles sized to 93 rows lost 10-20% at 64, 128 and 512)
shows up while it is being written. The sizes are the same in every job; they
are in the report as `search_sizes`. The final check runs and times the
model at those ten and at ten more drawn the same way, none of which the
search saw (outputs are checked there too), and the result to quote is the
geometric mean over the ten new ones plus the worst size (`final.headline`). Before that final check, the tool checks the kernels at
every other size in the range too, and uses each one wherever it is correct
and faster there.

**MLX-LM prompts: automatic.** A prompt workload read from an empty cache
(`use_library_inference: true`, `context: 0`, `shape: [1, 2048]`) is checked at
every prompt length from 3 tokens to 2,049, one past MLX-LM's 2,048-token
chunk. You write nothing extra. To keep the kernels at the one optimized size,
add `range: false`.

**Other models: name the size.** The tool can't tell which dimension is the
size for a custom model, so give it a name, say which size to optimize, and the
range to check:

```yaml
workloads:
  - name: denoise
    inputs:
      - {shape: [1, T, 128], dtype: bfloat16}   # T = image tokens
primary: {T: 1024}                               # the size the search optimizes
range: {T: [256, 4096]}                          # optimize for 256 to 4,096
```

When only a few sizes ever occur, list them instead of a range. An image
model's image-token count comes from the resolution, so 512, 768, 768x1024 and
1,024 pixels are all it ever sees:

```yaml
primary: {T: 2304}                                       # 768 x 768
range: {T: {sizes: [1024, 3072, 4096]}}                  # 512 x 512, 768 x 1024, 1024 x 1024
```

Listed sizes are each recorded, checked and timed; nothing between them is
used. Give at least three sizes, the optimized one included.

**Several requests at once.** An engine that batches requests (LM Studio's,
a batching server) calls the model with one row per request. Name that
dimension and give it a range, and the kernels are searched and served for
each request count:

```yaml
workloads:
  - name: decode_after_512
    context: 512
    inputs:
      - {shape: [B, 1], dtype: int32, low: 0, high: 248320}   # B requests at once
primary: {B: 1}
range: {B: [1, 4]}
```

Each request count gets its own conversation in memory, the same history in
every request, filled before anything is recorded. On Bonsai 2 9B in LM
Studio's engine this made two requests at once 1.37x faster (tokens identical
per request), where a one-request job's kernels stepped aside at two.

After the search the tool records the model at both ends and the middle of the
range and writes new code for each layer it changed, with sizes read from the
layer's input instead of written in. Then it checks, and uses only what
passed:

- The new layer code builds the same calculation as the original layer, at
  every size in the range.
- Each kernel gives bit-identical results to the operations it replaces, at
  every size it will run at, with the GPU's out-of-bounds checking on and
  again off, each time on inputs from a real run of the model at that size
  (only as far as the layer's first four copies, so it is cheap). Those
  operations are replayed at each size with any whole number that follows the
  size set for it, such as a reshape to the sequence length; a kernel whose
  operations change with the size in any other way stays at the optimized
  size.
- Each kernel is faster than those operations, timed on inputs from a real
  run of the model at each size. Over a range it is timed at about ten sizes
  spread over it and used over every stretch between two of them it wins at.
  Speed zig-zags between sizes (kernels work in tiles of rows), so it is also
  timed at twelve random sizes inside those stretches, and it is never used at
  a size where it was clearly slower. Listed sizes are each timed.

A kernel that is wrong or slower at some sizes simply isn't used there; the
original operations run instead. Sizes outside the range, and layers whose code
changes shape in any other way with the size (a Python loop over tokens, say),
run as the search left them. Kernels accepted within a tolerance rather than
bit-exactly serve at the sizes the search checked them at (the optimized size
and the sizes of any other workloads in the manifest) unless the every-size
check finds them wrong there, and at other sizes in the range only where their output differs from the original operations by no
more than it did at those checked sizes. The report's `check_optimizations_for`
section (named for the key's earlier name) lists every layer and kernel with its range and, if left out, why, plus
whole-model timings at the recorded sizes.

Choose a range where the model runs the same code path (an MLX-LM prompt of 1
or 2 tokens takes another one). For MLX-LM, only the first 2,048-token chunk of
a longer prompt gets the kernels, and generating the answer runs as before.
This stage added about 25 minutes to an LFM2.5-1.2B run at the automatic range,
mostly the every-size kernel check; it grows with the number of kernels
shipped. The named dimension needs `primary`; no correctness sweep is added to
the search unless you also set `sweep`.

## Identical outputs: the model may be used differently

```yaml
identical: output
budget: {total: 0, model: 20}     # 0 kernel attempts: only the inference loop
final_benchmark: {steps: 128}     # a language model generates 128 tokens per use
```

By default a job keeps the model's computation and makes its kernels faster.
`identical: output` also lets it change how the model is used: the function
you call the model through, which the AI rewrites whole and the artifact hands
back with the same signature. For an MLX-LM model that is MLX-LM's own
`generate_step` (what `mlx_lm.generate`, `stream_generate` and `mlx_lm.server`
call): the search starts from its source, and `from backcountry import
generate` (or `stream_generate`, `generate_step`) gives it back, called exactly
as MLX-LM's, for the model the loop was made for (the artifact's
`model_loop.py` has the same functions). With a `run()` in the model file it is that `run()`; otherwise one
call of the model. Examples: guessing tokens with a smaller model of the
same family and checking them with the model (speculative decoding), or
computing step-independent values once in a denoising loop. Each candidate is
checked on fresh inputs and timed against the current best on more fresh
inputs; winners stack, and a reviewer reads every win for anything that games
the check. A loop keeps one copy of the weights: one that still holds more
than 1% of the model's weights in memory after its runs, beyond any model it
loads (a drafter), fails its check (a merge must replace the originals with
views of the merged array, not sit beside them). At the end the best loop is checked and timed against the default on
inputs the search never saw, and ships as `run.py` in the artifact.

"The same outputs": for MLX-LM's generation, the same results as MLX-LM's own
`generate_step` for any call: every step's token and its log-probabilities bit
for bit, checked under greedy decoding, a seeded temperature/top-p sampler, a
repetition penalty and a small `prefill_step_size`, with a prompt cache already
holding a conversation and on a prompt read from no cache, plus the same
progress-callback calls and the cache left with the same positions and contents,
also when the caller stops reading part way, as it does at an end-of-reply token
(a chat's next turn continues from that cache: `mlx_lm.chat`, `mlx_lm.server`). It must also read a prompt no slower than
MLX-LM's (`model_level.final.prefill`). A call that sets an argument the job
did not check (`kv_bits`, `max_kv_size`, `input_embeddings`), or a model other
than the one it was made for, runs MLX-LM's own `generate_step`. For a
model file's `run()`, every token must equal the untouched model's. A loop that reads
several generated tokens in one model call (checking guessed tokens, re-reading
accepted ones) would otherwise differ in the last bits, because MLX runs other
kernels for several tokens than for one, and flip tokens where the model's top
two choices are nearly tied. So while a loop generates, the job computes every
such call with one-token arithmetic per token: the model's linear layers row by
row in one multi-row kernel (from oMLX, `src/backcountry_runtime/omlx_row_exact.py`),
attention one query at a time (`src/backcountry_runtime/exact_rows.py`). Plain
decoding's one-token calls and the prompt read are untouched. At the start the
job measures, on this model and Mac, whether such calls match one-token calls
bit for bit and what they cost (`model_level.probe`); the AI sees that. The
bundle's `run.py` carries the same rule. Timed runs are checked too, and the
final check covers half, the same and twice the conversation length, each with
the reply length and twice it (`model_level.final.cases`). Everything else must
match within `tolerances`. No approximate methods: no lossy caches, skipped
steps or lower precision.

With a kernel budget too, the loop search runs first, then the kernel search
runs as an ordinary job (in `kernels/`) on `kernel_manifest.yaml`, which the job
writes: this manifest aimed at the calls the winning loop spends the model's
time on (read off `model_level.profile`). When that is calls reading several
generated tokens at once (a speculative loop's checks), a decode workload
becomes those calls: `[1, new_tokens]` over the same conversation, `generated:
true`, the token count it spends the most time on as the primary and the others
as listed or checked sizes (checked sizes when another workload's sizes are
the job's range). Workloads without a conversation, such as a prompt read,
stay as written, and the loop search runs on the conversation workload. Such a
workload runs every call with one-token
arithmetic per token, keeps its kernels bit-exact (they stand in for one-token
calls), and its kernels run only on such calls, never on a prompt read or a
one-token call of the same shape. A workload the loop calls as before is
unchanged. The final numbers time the winning loop on the patched model
against the untouched model's default loop (`model_level.final`; the loop alone
is `model_level.loop_alone`), and `model_level.final.kernels_under_loop` times the
winning loop on the untouched and on the patched model: what the kernels add
under it. The bundle carries the kernels and `run.py`. When the kernel search
confirms no kernel, or the loop with the kernels fails that final, the bundle
carries `run.py` on the untouched model with the loop's own final, and
`model_level.kernels_dropped` says why. Models a loop can use beside the
model (a smaller model of the family, a multi-token-prediction head, a drafter)
are listed, not loaded, from the models already downloaded with the same
tokenizer (`helpers["assets"]`, `model_level.assets_listed`); a loop loads what it uses
in its `setup()`. The bundle pins only the ones the winning loop read
(`model_level.assets`), each to the revision the job checked, and its
fresh-process check fails if the loop there does not read every one of them.
With a `context` (a decode workload), the loop generates `final_benchmark.steps`
tokens after that many tokens of conversation, which are already in the model's
memory (`helpers["cache"]`) when its clock starts, so only the generating is
timed. A loop that needs its own memory of the conversation (a head's) reads it
itself in `read()`, also before the clock; whatever that read takes beyond the
default read is added to its time.
Output mode can't resume.

## Field reference

Only `model` and `workloads` are required. Each workload needs `inputs`, and
each input needs `shape` and `dtype`. Anything you leave out gets its default.
Misspelled or unknown fields are rejected. A dotted name like `budget.total`
means a nested field.

| Field | Default | What it does |
| --- | --- | --- |
| `model` | required | Path to a Python file with a `build()` function that takes no required arguments. Relative to the YAML file. |
| `baseline` | `compiled`; `plain` for MLX-LM generation | What to beat. `compiled` = the model under `mx.compile`; `plain` = the model as it runs today. MLX-LM runs models uncompiled, so its generation defaults to `plain`. See [compilation](#notes). |
| `model_type` | `other` | What kind of model this is: `llm`, `moe-llm`, `vlm`, `vla`, `diffusion` or `other`. Labels this job's lines in the [knowledge file](usage.md); a later job of the same type sees them first. |
| `use_library_inference` | automatic | `true` times MLX-LM generation; `false` times direct model calls. Left out, the tool picks based on what the model supports. |
| `workloads` | required | List of input cases to optimize. At least one. |
| `workloads[].name` | `workload0`, `workload1`, … | Report label. Must be unique and non-empty, with no `@`. |
| `workloads[].context` | none | Tokens already in the cache. `0` = empty cache. One workload per manifest may have one (with one integer input); the others run without a cache. |
| `workloads[].generated` | `false` | With a `context`: the tokens are generated tokens a loop reads several at a time (checking guesses), so every call computes each token with one-token arithmetic and kernels must keep every bit. Written by `identical: output` jobs for the kernel search; see [identical outputs](#identical-outputs-the-model-may-be-used-differently). |
| `workloads[].inputs` | required | Positional arguments to the model, in order. At least one. |
| `workloads[].inputs[].shape` | required | List of positive sizes or dimension names, e.g. `[1, L]`; `[]` for a single number, such as a diffusion timestep. |
| `workloads[].inputs[].dtype` | required | Input type; see the list below. |
| `workloads[].inputs[].low` / `high` | `0` / `100` | Range for random integers, `low` included, `high` excluded. `low` must be less than `high`. Integer dtypes only. |
| `primary` | largest sweep size | Size to optimize for, per named dimension. |
| `sweep` | `[1, 13, 50, 4096]` | Sizes to check for correctness, per named dimension. Set this yourself when you use named dimensions. None by default for a `range` dimension. |
| `range` | automatic for MLX-LM prompts | One named dimension and `[smallest, largest]`, or `{sizes: [...]}` for exact sizes: the sizes the job optimizes for. The search runs at ten sizes across it (every listed size), and afterwards the kernels run wherever each is checked correct and faster; `false` keeps everything at the optimized size. Its earlier name, `check_optimizations_for`, is still read. See [other sizes](#other-sizes-where-else-the-kernels-run). |
| `identical` | `computation` | What the job keeps the same. `computation`: the same math, faster kernels. `output`: the same outputs for the same inputs, and the job may also change how the model is used. See [identical outputs](#identical-outputs-the-model-may-be-used-differently). |
| `budget.per_region` | `25` | Max attempts on one region. |
| `budget.total` | `250` | Max kernel attempts for the whole run. With `identical: output`, `0` skips the kernel search. |
| `budget.model` | `20` | `identical: output` only: max attempts at a new inference loop. |
| `assets` | none | With `identical: output`: Hugging Face ids of checkpoints the loop may use beside the model (a multi-token head, a drafter), downloaded before the loop search. With none listed, the loop search downloads the ones published for a Hugging Face model, in this order: its own multi-token head, drafters trained for it (its author's, then z-lab's DFlash drafters), then its author's smallest model of the same family at the same precision (`Qwen3.5-4B-4bit` gets `Qwen3.5-4B-MTP-4bit`, `z-lab/Qwen3.5-4B-DFlash` and `Qwen3.5-0.8B-4bit`). The loop is offered every smaller checkpoint already on your Mac that splits text into the same tokens; a listed one that does not, or that is not smaller than the model, is reported and left out. A listed drafter that ships no tokenizer of its own (Liquid's DSpark drafters) uses the model's when its vocabulary is the model's size, and the judge is shown the mlx-vlm code that loads and runs it. `backcountry run <model id>` lists the ones its author published beside it (`Qwen3.8-27B-4bit` gets `Qwen3.8-27B-MTP-4bit`, `LFM2.5-1.2B-Instruct-MLX-4bit` gets `LFM2.5-1.2B-Instruct-DSpark`). An artifact pins each one it uses to the revision the job checked and downloads it there when it is missing. |
| `function` | none | With `identical: output`: the function the application calls the model through, as `"module:qualname"` (`mlx_engine.generate:create_generator`, `mlx_lm.generate:stream_generate`, an image pipeline's method). The model file's `run()` calls it the way the app does; the loop search rewrites it, and the app gets it back with `fn = backcountry.faster(fn)`. For an engine that keeps the model in a kit, return the model from `build()` with the kit attached to it, and use `baseline: plain`. `final_benchmark.steps` must be 1: `run()` is one whole use. An engine that keeps the model behind its own running thread is not freed by dropping it: define `release()` in the model file to shut down every one it built (LM Studio's `kit.shutdown()`); the job calls it before the export's fresh-process check. |
| `tolerances` | none: every kernel bit-identical | Opt in to kernels whose outputs differ slightly from the original: `default` (each output dtype's limit below) or `{rtol: …, atol: …}`. See [correctness](#notes). |
| `final_benchmark.steps` | `10` | Library mode: tokens generated per request. Direct calls: how many calls in a row one use of the model makes (an image's denoising steps); the final times whole runs of that many. `1` = one call per use (a chat prompt read once per turn): the final's paired check already times that, so no runs are timed. |
| `final_benchmark.pairs` | `4` | Number of original-vs-optimized timing pairs. Even, and at least 4. |
| `final_benchmark.warmup_steps` | `3` | Minimum warmup calls. The tool may warm up longer if timings haven't settled. |

All counts in `budget` and `final_benchmark` must be positive integers, except `budget.total: 0` under `identical: output`.

Input dtypes: `bool`, `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`,
`int32`, `int64`, `float16`, `bfloat16`, `float32`, `complex64`. A dtype being
accepted here doesn't mean every operation has a starter kernel for it.

## Notes

**Compilation.** The bar is the model as its library runs it. MLX-LM runs
models uncompiled, so its generation defaults to `baseline: plain`
(compiling every expert block of Qwen3-30B-A3B changed its time to the first
token by 0.1% or less). With `baseline: compiled`, plain model calls run under
`mx.compile`, and library generation compiles the parts of the model it
safely can. Models that keep internal state between calls sometimes can't be
compiled from outside; those fall back to plain, and the report says so and
why. Very short final benchmarks may be repeated more times to get a usable
timing, and the result records the counts actually used.

**Cache when `context` is left out.** Library generation starts with an empty
cache. Direct calls get no cache at all.

**Correctness.** A change that does the exact same math must give the exact same
output bits. By default that is the only kind allowed: every kernel must be
bit-identical to the original. With `tolerances` set, a change that reorders
floating-point math (and so rounds slightly differently) is allowed an error of
`atol + rtol * abs(original)`, and the job's summary counts it as "within
tolerance", never as identical. Integer outputs always have to match exactly.
`tolerances: default` uses these limits:

| Output dtype | `rtol` | `atol` |
| --- | ---: | ---: |
| `float32` | `0.00001` | `0.000001` |
| `float16` | `0.01` | `0.02` |
| `bfloat16` | `0.02` | `0.04` |

Setting `tolerances: {rtol, atol}` overrides these. Both values must be finite, non-negative
and fit in a float32. On top of the error limit, the checker also compares
shape and dtype, catches NaNs and infinities, and where it applies, allows for
how much the original model's own output varies from run to run.

These limits judge a kernel against the library at the region it replaces.
For the whole model, a kernel that reorders math is graded against a
reference instead: the original model with the replaced regions recomputed in
`float32` and rounded back. Its worst error against that reference, each
output's error taken relative to its largest value, may be at most 1.25 times
the original model's own. The output dtype alone doesn't say how precise a
value is: Qwen3.5's attention state is `float32` but computed from `bfloat16`
math, so any reordering upstream moves it about 1%, as recomputing the region
in `float32` does, far past the `float32` limit above. A region whose code
can't be recomputed in `float32` (a captured custom Metal kernel, say) falls
back to the limits above. Exported bundles carry the reference's outputs and
validate against them the same way.

## What doesn't go in the manifest

- **Checkpoint, weights, quantization:** in the model's Python file.
- **Which judge, judge model, output folder:** command-line flags, e.g.
  `--judge claude-cli --model YOUR_JUDGE_MODEL --work-dir runs/my-run`.
- **Hardware, random seed, cooling:** the tool handles these.

For setup, stopping a run cleanly and reading results, head back to the
[usage guide](usage.md).
