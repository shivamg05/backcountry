# Running an optimization

This is the full path from "I have a model" to "I have a faster model", writing
the model file yourself. Most apps don't need to: mark the function your app
calls for one request with `@backcountry.tune` and run `backcountry init app.py`
([quickstart](../README.md#quickstart)); init writes the model
file. Write one by hand when you want full control over how the model is built
and called. If you haven't installed anything yet, do the
[quickstart](../README.md#quickstart) first. Run every command here from the folder that holds your manifest.

## 1. Provide a model

Write a Python file with a `build()` function that returns your MLX model,
ready to call:

```python
def build():
    from mlx_lm import load
    model, _ = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    return model
```

The tool needs separate "original" and "optimized" copies it can compare. It
calls `build()` once and copies the model, sharing the weights, so a large
model fits. If something in your model can't be copied, it calls `build()`
again instead. That's where these rules come from:

- **Load inside `build()`, not at import time.** That includes downloads and
  weight loading.
- **Return a new model every call.** Don't keep one model in a global and hand
  it back each time. (The tool shares weights between the copies itself.)
- **Resolve local file paths relative to `__file__`**, so the file still works
  when it's run from another directory.
- **Using random weights?** Seed them inside `build()` so separate runs get the
  same model.

By default the tool times one call of that model. MLX-LM models are run
through MLX-LM's own generation call instead. If your application uses the
model some other way, say an engine's own loop around the model, add an
optional `run(model, *inputs)` next to `build()` that does exactly that, and the
tool traces, checks and times `run()` in its place (the manifest's inputs are
`run()`'s inputs; a workload with a `context` hands `run()` that conversation
ahead of its own tokens). An engine's `run()` can make its own cache: the tool
finds it. With `identical: output`, the loop search starts from `run()` and
ships its rewritten version. A loop that only repeats the same call (an
image's denoising steps) is better left out: one call shows the same work, and
each timing sample of a 4-step FLUX image took 34 s, four steps' worth.

```python
def run(model, prompt):                       # the engine's own loop: a reply of 16 tokens
    cache, tokens, out = model.make_cache(), prompt, []
    for _ in range(16):
        tokens = mx.argmax(model(tokens, cache=cache)[:, -1:], axis=-1)
        out.append(tokens)
    return mx.concatenate(out, axis=1)
```

**Show the inputs with `examples()`** and you can skip the manifest: it
returns a list of real inputs, one entry per call (a tuple of arrays, or one
array), at the sizes your application sends. `backcountry run my_model.py` reads
each input's shape and type from them (and, for integer inputs, the range of
values), names the one size that changes between examples `T`, optimizes at the
first example's size, and checks and serves the others: with two sizes it times
each, with three or more it lists them as the job's sizes. It writes that as
`manifest.yaml` in the run folder. The examples are read in a separate process
and never timed; random values of the same shape and type stand in for them,
as in every manifest. Examples that change size in more than one independent
way need a manifest written by hand.

```python
def examples():
    return [(mx.zeros((1, t, 128), dtype=mx.bfloat16), mx.arange(t)[None]) for t in (2304, 1024, 4096)]
```

Don't call `mx.compile` inside `run()`: a compiled call is one opaque step to
the tool, so nothing inside it could be optimized. If your application
compiles, set `baseline: compiled` and the tool compiles `run()` as a whole.

You don't have to change your model code. The tool finds the MLX operations,
and any custom Metal kernels you already use, on its own; no hooks needed. It
never changes your model's precision or quantization. Parts of the model it
can't handle are skipped and listed in the report.

## 2. Write a manifest

Skip this if your model file has `examples()` (or for a model id): the job writes it.

The manifest is a short YAML file that says which inputs to make faster and how
many attempts to allow. [Write a manifest](manifest.md) has examples to copy,
including time to first token, a single decode step and a plain model call, plus
every available field. Save one as `my_run.yaml` beside your model file and change its
model path and inputs. Paths inside the manifest are relative to the manifest file.

## 3. Run it

```sh
backcountry run my_run.yaml --judge claude-cli
```

What to expect:

- **It takes a while.** Depending on the model and budget, a run can take hours.
  Long pauses are normal: the tool rests the GPU between timings so heat
  doesn't skew them, and it waits on the AI for each new kernel.
- **Keep the terminal open and other GPU work quiet.**
- **One job at a time.** The tool refuses to start a second one.
- **Output goes to a new folder under `runs/`**, and the tool prints the path.
  If you pick your own with `--work-dir`, it has to be empty.
- With a CLI judge or `--judge-cmd`, the tool sends one small test request
  before loading the model to make sure the AI is reachable. That doesn't count
  as an attempt.

**Choosing the AI (the "judge").** The judge is the AI that writes kernels.

| `--judge` | What you need |
|---|---|
| `claude-cli`, `codex`, `gemini` | That CLI installed and signed in. `--model` picks its model. `--judge-effort` sets effort for `claude-cli`. |
| `api` (the default if you leave it out) | `ANTHROPIC_API_KEY` set. `--model` picks the Anthropic model. |
| `--judge-cmd "<command>"` | Any headless command you like. See the [judge protocol](judge-protocol.md). |
| `agent` | Requests are written to files and someone (or some agent) answers them by hand. See the [judge protocol](judge-protocol.md). |

To have a coding agent run and watch the job for you, point it at
[RUNNING.md](../RUNNING.md). The [operator guide](operating.md) lists the
updates it should send you.

**Stopping early.** To stop new attempts but still get the final checks and the
packaged result, run this from another terminal:

```sh
backcountry finish
```

An attempt in progress finishes first; region pricing (between regions) stops
at its next cooling pause, within seconds. Don't just kill the process: you'd lose
the final checks and be left with only the recovery checkpoints.

**Skipping a region.** To give up on the region being searched and move on to
the next one:

```sh
backcountry skip
```

The attempt in progress finishes first. The region closes, and the job opens
the next region if any attempts are left. A skip asked between regions does
nothing: it only ever applies to the region that was open when you asked.

**Pausing.** To stop now and continue later, say before the Mac sleeps:

```sh
backcountry pause    # stops after the current attempt
backcountry resume   # later
```

The job saves its state after setup, between regions and before each judge
call. `pause` stops it at the next of those points; it exits with status
`paused`. That waits for the attempt in progress, which can take a while (on
FLUX, half take over 10 minutes and the longest took an hour). In a hurry,
press Ctrl-C and resume later: you lose only that attempt, and the resumed job
tries it again. `resume` starts a new process with the same manifest and judge
(`--judge`, `--model`, `--judge-effort` and `--judge-cmd` override them),
records the model again (a few seconds), reinstalls the accepted kernels and
checks the outputs still match, then carries on in the same region with the
same budget. Setup isn't measured again. It refuses if the manifest file
changed since the job started.

The job also pauses itself if the judge doesn't answer three times in a row
(no network, an expired login, a reply over 30 minutes); those calls don't use up attempts. Fix the
cause and resume. After Ctrl-C, a crash or a killed process, `resume` continues
from the last saved point in the same way. Sleeping the Mac without pausing is not safe: the job
may lose its network or be stopped.

Every job also adds what it measured to a knowledge file shared by all your
jobs, `~/.backcountry/knowledge.jsonl` (set `BACKCOUNTRY_KNOWLEDGE`
to use another file): one line per kernel tried, with the model and its
`model_type` from the manifest, the region's operations and shapes, the measured
outcome, the judge's one-line change and the kernel's code, plus the judge's
reflection on it. Once a region's opening designs are in, the judge sees 12
earlier attempts drawn at random: up to 4 that shipped, right and faster in the
whole model (most attempts don't), the rest from anything, the same model type first, so a
lesson can carry across models and kernel types. It reads
the code of any entry on request. It's only a starting point: any kernel taken
from it is checked from scratch. An outcome is recorded when the attempt is
judged, so a kernel that was pending then and shipped later still reads
pending. Delete the file to start over.

## 4. Read the result

A run ends in one of three ways (or stops as `paused`, to be resumed):

- **Verified artifact.** You have a faster model. See [Using an artifact](artifacts.md).
- **No confirmed improvement.** The run worked, but nothing it found beat the
  baseline by a margin it could confirm.
- **Failure.** Something broke. See [troubleshooting](#troubleshooting).

**How to read the speedup.** Speedup = original time ÷ optimized time, for that
workload against that baseline. So `1.25x` means the work takes 20% less time,
not 25% less.

For MLX-LM generation runs, you'll also see generated tokens/sec before and
after. That's output tokens divided by the *whole* request time, prompt
processing included, so it isn't pure decode speed. Direct model-call runs
report time per call instead.

**Only the final result counts.** During the search you'll see kernels that
are fast on their own, and changes that get accepted along the way. Neither is
the result. The result is the final whole-workload comparison.

What's in the output folder:

- `report.json`: every measured workload, the baseline, all attempts, the final
  checks and the outcome.
- `run.jsonl`: progress events, one per line. `candidates.log`: a readable
  record of each attempt.
- `checkpoints/`: changes accepted during the search, saved in case the run
  dies. They haven't passed the final checks.
- `artifact/`: the bundle you actually use. It's only written after the final
  comparison confirms a win and the bundle passes validation.

### What's in `report.json`

At the end, the tool prints how many regions got a confirmed speedup, and two
before/after timings: one step of the model, and a longer sequence of
consecutive steps. Both are measured against the baseline in your manifest
(compiled unless you changed it). The final comparison is what decides whether
anything ships: a decode job ships only if its consecutive steps are faster by
more than 1%, at two of the timed sizes when it covers a range, and slower at
none. It measures the task your manifest describes, which may be
narrower than your whole application.

`report.json` has the full story: every region and why work on it stopped,
every attempt and its verdict, and the step timings for both plain and compiled
MLX. The fields you'll most likely look at:

| Field | What it tells you |
| --- | --- |
| `step_ms[workload].win_confirmed` | Whether the final measurement confirmed a speedup from an installed kernel. **This is the one to trust.** A `speedup` ratio on its own can be noise, especially when nothing was installed. |
| `final.decode` | MLX-LM prompt jobs: the untouched and the optimized model generating 128 tokens after a 512-token prompt, paired (`speedup`, `slower`). Only a clear slowdown fails the final. |
| `final.second_message` | MLX-LM prompt jobs: the untouched and the optimized model reading the job's prompt with a first message of the same length already in their history (a chat's later turns), paired (`speedup`, `slower`, `history_tokens`, `prompt_tokens`). Only a clear slowdown fails the final. |
| `final.headline` | With a `range` in the manifest: the whole model at every size the final check timed (`curve`, smallest first), the geometric mean of the speedup over the timed sizes the search never ran at (`geomean_fresh`), and the worst size (`worst`). Quote this rather than one size's speedup. |
| `step_ms[workload].speedup_vs_plain` | With the compiled baseline: the finished model against plain, uncompiled MLX. This includes what compiling alone buys you. |
| `step_ms[workload].speedup_vs_compiled` | With the plain baseline: plain MLX plus the new kernels, against the untouched model under `mx.compile`. Below 1 means compiling alone would have been faster. Absent for MLX-LM generation and for a step over a cache, which cannot be compiled from outside. |
| `plain_win_confirmed` (and its siblings) | Whether that cross-baseline comparison was confirmed. It's measured in pairs at the end of the job, at the primary size only. |
| `final.passed`, `reason`, `outcome` | If outputs were right but the win couldn't be confirmed, the job still ends normally: `final.passed` is false with a `reason`, `outcome` is `unconfirmed`, all measurements are kept, and no artifact is written. Wrong outputs end the job with an error instead. |
| `step_ms[workload].min_win_ms` | The smallest per-step saving a region's change needed to count as a win there: 1% of the step, capped at 30 µs. |
| `step_ms[workload].steps_per_sample` | How many back-to-back steps went into each timed sample. More than 1 means a single step was too short to bring the GPU up to full speed on its own. Every whole-model number is still reported per step. |
| `floor_ms` (on each attempt) | A physical floor timed right next to that kernel: an estimate of how fast the hardware could do that work. It's used for ranking, not a hard limit. |

The final timing comes from the same comparison as the final correctness check.

For timing details and what each log field means, see the
[measurement reference](measurement.md).

## Troubleshooting

| What happened | What to do |
|---|---|
| Judge missing or login expired | Follow the printed message, sign in again, and run the same command again (`backcountry resume` for a paused job). |
| Extension build failed | The tool ships its small C++ extension built for each macOS release MLX supports (14, 15, 26) and each supported Python (3.12, 3.13, 3.14), against the newest MLX it was released with; it builds one on first use for any other MLX version, or after its source (`src/backcountry_runtime/graph_native/`) was edited, which needs Apple's Command Line Tools (`xcode-select --install`); then `uv run python tools/build_graph_native.py` rebuilds the shipped ones. Keep the compiler error if it still fails. |
| A region is unsupported | Nothing to do. The tool skips it, keeps searching the others, and notes it in the report. |
| No confirmed improvement | Not an error: nothing passed both the correctness check and the final timing within the budget. It also doesn't prove there's no speedup to find. The timing may just have been too noisy to call. |
| Judge stops responding mid-run | After three requests in a row fail to get through (connection or provider errors, not bad kernels), the search stops and keeps any accepted checkpoints. Fix the provider problem before running again. |
| The GPU fails mid-run | A kernel that hangs or fails the GPU is rejected alone once a trusted GPU check passes; the search goes on. If the GPU fails under the job itself (a macOS GPU timeout while the Mac is busy) or the check fails, the job resumes from its last save in a fresh process after a two-minute rest, up to three times, then stops with the error. A failure that came while a kernel was being checked is charged to that kernel: the resumed job checks the same kernel again, without asking the judge, and if the GPU fails under it a second time the kernel is recorded as failed (`job_died`, with the steps it died in) and the search moves on. Dropping one kernel uses two of the three resumes. |
| Crash, worker failure or GPU timeout | Keep the console output, `report.json` and `run.jsonl`, and report which stage failed with its traceback. Don't immediately retry a kernel that timed out on the GPU. For a GPU timeout, read the last `memory` line in `run.jsonl`: memory in use or its peak near `gpu_limit_gb` means the job ran out of GPU memory (the model plus what the job held), not that a kernel hung. |

Background load and heat can hide small speedups. The tool accounts for timing
noise, but it can't promise a win on every model. See [limitations](limitations.md).
