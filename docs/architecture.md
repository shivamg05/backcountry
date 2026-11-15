# Architecture

The tool optimizes a declared workload on a particular machine. The AI proposes
code; the harness independently decides whether that code is correct and faster.
The invariants below describe the boundaries that implementation changes must preserve.

1. **Load and trace.** `manifest.py` defines the model, inputs, budget and measurement
   objective. `trace/` records the operations and values crossing their boundaries.
2. **Select regions.** `regions/` groups legal replacement regions and estimates
   where replacing work could save whole-model time. Estimates guide search;
   they never establish a speedup.
3. **Build a starter.** `scaffold/` uses supported lowering, MLX source, or captured
   custom-kernel source to provide an editable implementation. If lowering cannot
   produce a starter, a replayable region keeps its original calls as the reference;
   the agent can write replacement Metal from scratch, in one or multiple stages.
   The reference itself cannot ship. Unreplayable or unbindable regions stay unsupported.
   If a region appears at several shapes, its reference retains the original wiring
   for each shape. One proposed kernel must pass all recorded cases; other shapes
   still use the original model. A failed generated starter takes this same path
   before spending a judge attempt on repair.
4. **Search.** `judge/` supplies bounded metadata and code context. The agent plans
   four opening designs, then may revisit earlier candidates, combine ideas, or
   start another design. Budgets limit actual attempts. Full earlier code remains
   readable; earlier successes and failures inform later proposals. A CLI agent
   may test candidates while it writes (`./try`, `LadderJob.quick`: outputs at
   the first recorded case and a short clock there); a try is never an attempt
   or a verdict.
5. **Verify and measure.** `ladder/`, `sandbox/`, and `measure/` check candidates in
   workers. `bind/` and `loop.py` verify installation and compare the actual model
   with its baseline. Bad outputs, ineffective replacements and unresolved gains
   do not become confirmed wins. A block that receives the model's cache and
   carries kernels gets the judge's own method calling them (live delivery),
   installed only if exact over first messages, second messages read on top of
   them and whole decode runs; otherwise its recording stays.
6. **Inference-loop search** (`identical: output`). `model_search.py` asks the judge for a
   whole inference loop: for MLX-LM's generation a replacement `generate_step` with
   MLX-LM's signature (`src/backcountry_runtime/generation.py`, called as a run() by the
   harness), else `run(model, *inputs, helpers)` (`src/backcountry_runtime/model_run.py`). It
   checks its outputs against the untouched model (MLX-LM's generation: every step's
   token and logprobs bit for bit under several call settings, the progress calls
   and the cache left; else the same tokens; calls that read several generated tokens are computed with
   one-token arithmetic per token, `src/backcountry_runtime/exact_rows.py`), times it on
   fresh inputs, and has a reviewer read every win before it stacks.
7. **Export.** `artifact/` writes a bundle, validates it in a fresh process, then
   publishes it. An inference-loop win ships as `run.py` and `model_loop.py` (for
   MLX-LM's generation, drop-in `generate_step`, `stream_generate` and `generate`),
   checked in a fresh process to reproduce the job's outputs exactly, through the
   drop-in too. `src/backcountry_runtime/` runs the installed kernels without depending
   on the search harness. See [artifact usage](artifacts.md).

## Installation and exported baselines

Graph matching constrains known boundary inputs (module weights and arguments)
before selecting branches. Distinct recorded operations must match distinct live
operations; boundary inputs may still alias. Before searching a region, an
original-computation replacement checks its graph cuts across the declared
workloads, then restores the incumbent. A failed check closes the region without
spending judge attempts. Candidate correctness and performance gates still apply.

Library-inference bundles retain the exact empty compiled wrappers used by the
job's baseline. Artifact benchmarks reinstall those wrappers on the original arm;
correctness validation continues to use the untouched model. Older bundles lacking
that baseline cannot claim a compiled comparison and must be re-exported.

## Boundaries to preserve

- Keep model precision, quantization and intended computation unchanged.
- The judge receives approved metadata and source, not tensors, weights,
  activations or numeric acceptance thresholds. It cannot override a gate.
- Arithmetic-preserving candidates must match exactly. Arithmetic-reordering
  candidates use the manifest's tolerance policy. Tolerance is not permission to
  change the model or its number format.
- Compare like with like: the same inputs, cache state, execution mode and objective.
  The recorded baseline is part of the result. Both compilation and replacement
  overhead must be represented in the deployed comparison.
- Compilation-time graph substitution is preferred where safe. Other supported
  scopes use certified replay. The implementation exported must be the one tested.
- Pair measurements, warm the GPU, pace GPU work, and account for uncertainty.
  Force lazy work to finish inside the measured interval. A timeout is an execution
  failure, not a performance verdict. Workers share the physical GPU.
- Cache state must be reset or advanced consistently with the declared task.
  A forward pass and a full generation request are different measurement objectives.
- Validate every live output and the whole model, then validate the exported bundle
  in a fresh process. Preserve accepted checkpoints if finalization fails.

Behavioral platform assumptions are pinned by the `test_platform_*` tests and
other regression tests. See the [measurement reference](measurement.md) for
log fields and timing details. Old
machine-specific measurements are not universal constants or product guarantees.

## Development

Run `uv sync`, then tests relevant to the change, for example
`uv run pytest tests/test_judge.py tests/test_widening.py` (`uv sync` installs the package
from `src/` with the test dependencies). The full suite includes
GPU jobs and can take substantial time. Run GPU tests serially, without another
optimization active. Live-provider tests are opt-in.

`tests/cli/` runs the real `backcountry` command the way a person (answers
typed at each prompt in a terminal) or an agent (flags, `--questions`) does,
and checks what it prints, one file per command: `uv run pytest tests/cli`, or
`tests/cli/test_init.py` alone. Its whole jobs use a judge that proposes
nothing (`tests/cli/fake_judge.py`), so they need no AI or network; they are
marked `integration` (minutes; `-m 'not integration'` leaves the seconds-long
ones). A change to what a command prints or asks comes with a test there.

Pull requests are welcome. Keep changes focused, preserve the invariants above,
and include the relevant validation. Keep generated runs and weights out of Git.

### Known gaps

**A constant the model reads back as a number.** mflux computes its attention
scale as `1 / mx.sqrt(head_dim)` and passes that array as `scale=`, which the
call reads as a float. The recording stores the number in the call and treats
the calls that computed it as work the step never runs
([measurement](measurement.md)). Two things follow for any module that
contains such calls:

- During the search its kernels are held by replay delivery, not graph
  delivery: `graph_scope_reason` declines any scope that mixes never-run calls
  with live ones. That rule exists for a module output the step drops; here
  the never-run calls are inside the module and a compiled step would not run
  them, so graph delivery would be safe. Under a whole-step compiled baseline
  (a directly called model such as mflux's FLUX.2) nothing is lost, because
  the replayed module is compiled with the rest of the step. Under library
  inference, where modules are compiled one by one, such a module runs
  uncompiled, which costs the final number only where compiling a module is
  itself a large gain: recurrent models (Mamba, RecurrentGemma), not
  transformers. No model run so far is both.
- No region crosses those calls, since a never-run call ends a region. On
  FLUX.2 a region cannot run from the start of an attention layer into the
  attention call itself; each side is still a candidate on its own.

Fixing either means telling these calls apart from work a module computes and
the step drops.
