# Limits and interpreting results

The short version: **only the final, confirmed whole-workload measurement
counts as a speedup.** Everything you see during the search, such as a region's
estimated gain or a kernel that's fast on its own, is a clue the search uses,
not the result.

## What the tool covers

- **MLX on Apple Silicon only.** Other GPU backends aren't supported.
- **Not every part of every model.** Some operations can't be captured,
  rewritten or safely replaced yet. Those regions are skipped, and the report
  lists them with the reason. When starter-code generation is the only obstacle,
  the tool keeps the original operations as a checked reference and lets the AI
  write a replacement from scratch. That replacement must pass the same checks;
  this is sampled verification, not a mathematical proof for every input.
- **Cache handling and MLX-LM generation depend on the model's API.** If you
  explicitly ask for something the model doesn't support, the run should refuse
  to start rather than quietly time a different task.

- **A function whose output doesn't repeat bit for bit needs `tolerance:
  default`.** Each kernel is confirmed against the untouched function's own
  output. Some libraries' GPU math doesn't repeat exactly (Kokoro in mlx-audio
  differs by about 1e-7 between two identical calls), so identical outputs
  can't be confirmed there: under `tolerance: exact` `check` says so in
  minutes, and under `tolerance: default` every kernel is graded within each
  output type's rounding limit.

## What a result does and doesn't prove

- **Kernels are verified only for the workloads you declared** (plus any
  runtime checks that apply). At an untested input size or cache layout, the
  model falls back to its original code. One result doesn't prove a speedup at
  every size or on another machine.
- **A result belongs to the MLX it was made with.** Kernels match the original
  bit for bit only against that library: under MLX 0.32.4 the expert matmul
  sums in another order than 0.32.2's, and Qwen3-30B-A3B kernels made under
  0.32.2 changed its reply from the second token. So a result installs only
  under the exact MLX version it was made with (patch number included); after
  an upgrade the model runs its original code, with a warning, until you run
  the job again.
- **Several requests at once need a request-count range.** Kernels are
  checked at the input shapes the manifest names. A job over one request
  leaves batches (MLX-LM's `batch_generate`, a batching server) to the
  original code, so outputs match but nothing is faster there; name the
  request dimension and give it a range
  ([several requests at once](manifest.md#other-sizes-where-else-the-kernels-run))
  to search and serve each count. A model file's `run()` that batches through
  MLX-LM's `BatchGenerator` can't be searched yet: the job stops while
  recording it.
- **Where the library is already near memory speed, there's little to win.**
  Generating a token reads every weight once, so a matmul can't beat the time
  it takes to read its weights. Qwen3-8B 4-bit in LM Studio's engine at one to
  three requests: the three matmul regions searched (77% of the step) already
  ran at 80-90% of the measured memory speed, and none of 36 exact kernels beat
  the library by more than the 1% margin at the sizes timed, so the job made
  nothing (three hours). The rest of the step is many small operations (835 GPU
  launches per token), each under 3% of it, which the budget didn't reach.
  Bonsai 2 9B's gain in the same engine came from MLX's 2-bit matmul, which was
  slow at two to five requests.
- **Small gains can stay unconfirmed.** Timing noise, background GPU work, heat
  and power state all affect measurements. A gain ships only when it's more than
  twice the uncertainty of its measurement; on a 57 ms decode step that's about
  0.4 ms, so a 0.5% gain is right at the edge. More attempts don't guarantee a
  confirmed win.

## Using the result

- **Artifacts don't include weights by default.** They need a compatible
  architecture, dtypes and quantization, the same dependencies, and whatever
  external files your `build()` loads. See
  [artifact compatibility](artifacts.md#apply-it-to-your-model).

## Running a job

- **Some models have little room for kernels.** At setup the job measures each
  workload against the fastest the chip could run it (reading its weights or
  doing its arithmetic, whichever takes longer); when even perfect
  kernels could not make any workload 1.10x faster, it warns before the search.
  The search still runs, since a loop search (`identical: output`) changes the
  work itself, but kernel gains there stay small.
- **The budget counts attempts**, not time. A run's length depends on the
  model, and can be hours; the final checks and export run after the budget is
  spent.
- **Jobs can stop early** on a worker failure or a timeout. A macOS GPU
  failure resumes the job from its last save, up to three times; anything else
  stops it. If that happens, keep the report, the console output and the
  checkpoints.
