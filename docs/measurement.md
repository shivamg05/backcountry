# Measurement and log reference

Start with [the usage guide](usage.md) to configure a run. This page explains
internal measurements, installation methods, and diagnostic events.

Preparation checks that a region has a delivery scope and a buildable
starter. A CPU estimate selects regions whose recorded spans do not overlap.
The others stay queued and become available as selected regions finish.
Repeated regions are grouped only when their required inputs and outputs have
the same roles and order. A final layer that returns fewer live values is a
separate target. Saved input/reference file headers are checked before each
wave is priced or searched; a mismatch stops with the exact file and reason.
Measured ranking uses Amdahl's law: the region's share of the step times the
fraction of its cost that its estimated limit could remove. Both parts of that
limit are clocked beside the spot in the same window: a kernel that only
streams its bytes, and a large plain matmul at its dtype for the arithmetic
(`measured_compute_gflops` in the pricing table). That score says what an
ideal kernel could remove, not how hard the kernel is to write, so it is
weighted by length: a region keeps 1% less of it for each operation past the
first (`length_weight` in the pricing table; a 3-op region keeps 98%, a 10-op
one 91%). Reshapes, transposes and slices are free for a kernel and are not
counted. When a region closes without a win, every queued region that shares
more than 60% of its operations (operations in common, of the operations
either covers) waits behind every region not yet tried (`similar_demoted`);
without that the next region opened is the same one give or take an
operation, and the budget goes to one spot.
A wave's regions are priced together against one shared model arm, but each
region's saved inputs are loaded only for its own turn and dropped before the
model or another region runs: held all at once, 48 regions of Qwen3.8-27B put
the job past the GPU's working-set limit and the model pass timed out.
A region whose estimated share of the step (its ops' ideal costs over the
step clock, no GPU time) is under a quarter of the floor is not priced
(`unpriced` in `run.jsonl`): on Qwen3.8-27B pricing measured 178 regions for
2.6 hours to keep 9, those 9 were the top 9 by this estimate, and the cut would
have priced 31.
`price_stability` in the pricing report shows agreement between observations:
closer to 1 means more consistent, while small values mean the ranking is noisy.
It is a diagnostic, not a confidence interval or proof of no headroom.
The report's `pricing` table retains measured regions, largest share first,
including their rejection reasons. `coverage.discovery` distinguishes legal
regions from those the tool can currently build and install. Unsupported
does not mean unprofitable. The roofline is a scheduling estimate, not a
guaranteed upper bound on a real kernel's performance.

`report.json` exists from startup and records failures as well as results.
Its `accepted` list records each completed full-model promotion immediately,
even if the region has not finished. Each acceptance also saves the exact
kernels, wrappers, runtime, and measurement report under
`<work-dir>/checkpoints/accepted-0001/` (then `0002`, etc.). These are recovery
packages: final comparison and fresh-process artifact validation are still
pending, and the checkpoint report says so. A later crash does not erase them.
A job that stops can usually be resumed instead (`backcountry resume`, see
[pause and resume](usage.md#3-run-it)); keep the checkpoints and logs either way.
`session.jsonl` records each cooling pause before sleeping, with its duration,
then records completion. Long pauses also appear on stdout. The cooldown is
twice the GPU work since the last rest, timed or not, and is taken before the
next GPU call rather than straight after the work, so Python and judge time
in between count toward it. On a Mac whose heat it can read, the job takes no
such cooldown while macOS reports no thermal pressure. The first pressure
switches it to the 2x rest for the rest of the job, after resting until the
pressure has cleared and a short GPU speed probe (about 0.1 s of fp16
matmuls) runs back inside the range the same Mac read cool, measured before
the job's first GPU work. The switch waits for the end of the measurement it
came in, never splitting one, and that measurement is taken again
(`heat_calibrated`, `heat_tripped`, `heat_cooldown`, `heat_cooldown_done` in
`session.jsonl`). No temperature decides anything: dies run at different
temperatures on different chips, and the hottest reading is only recorded.
`report.json`'s `constants.pacing` gives the policy, the cool range, the trip
and how long its cooldown took. Measured: the Mac mini M4 ran Qwen3.8-27B an
hour at 84 C with no pressure and no rests, each 1,833-token read within 0.1%;
the fanless MacBook Air M4 reports pressure within minutes. On the Air, the
probe slowed smoothly as the chip warmed (+8.7% at 83 C) but stayed steady,
which paired comparisons cancel; under pressure it turned erratic (runs 2-21
ms apart instead of under 1). The probe recovers within a minute of the load
stopping, but the Mac's body does not: resting only that long let pressure
return 40-60 s into every comparison (Qwen3-8B timed against itself,
0.988-1.014), while cooling once and then resting 2x read 0.9995-1.0016. A Mac without the readings (or whose readings stop working) rests 2x. Every pause, whether slept or scheduled,
also returns the memory MLX keeps for reuse (`cache_cleared_gb`); left alone it
grew to 14.5 GB on a 24 GB Mac, and under that memory pressure timed passes ran
up to 7x slow. Each paired block (`sample_group`) logs every raw sample plus MLX
memory and swap in use before and after, so a block that ran under memory
pressure is visible. A `memory` line, in `run.jsonl` too, comes at the start of
every stage and before every paired block and every untimed model pass (a
recording, a whole-model check): MLX memory in use (`mlx_active_gb`), its peak
since the previous line (`mlx_peak_gb`), the GPU's working-set limit
(`gpu_limit_gb`), the process's whole footprint (`footprint_gb`: MLX's arrays
plus what MLX does not count, such as compiled kernels and Python's own memory),
how much of it macOS holds compressed (`compressed_gb`), the machine's free
memory (`free_gb`) and swap. Running out of GPU memory shows up in Metal as a
GPU timeout, which names no cause; the last `memory` line says where the job
stood, and `job_failed` keeps the stack the error came through. Each paired
comparison's `compare` line also gives the most MLX held between its samples
(`max_held_gb`, in use plus cached), the most it kept cached (`max_cache_gb`)
and the peak (`peak_gb`), so memory that creeps across one comparison's samples
toward the GPU's limit shows up there.

Before any timing (and again when a job resumes, unless it already has one)
the job runs each workload once and keeps its MLX peak in
`constants.memory_fit`. A workload that peaks within 1 GB of the GPU's
working-set limit, and over a range every longer size of its workload, is
still checked for correct outputs everywhere but never timed as a whole model:
not when regions are priced, not in a kernel's whole-model check, not in the
final check, whose report lists it under `final.not_timed`. The 1 GB is what
MLX's count leaves out: the GPU holds 0.5-0.8 GB more for the job, and Qwen3.8-27B on
a 24 GB Mac mini (19.07 GB limit), which reads 1,833 tokens at an 18.15 GB
peak alone, ran its job out of GPU memory at 19.2 GB in every timing there.
MLX's cache of freed buffers is capped at what the largest timed workload
leaves under the same line (`memory_fit.cache_limit_gb`): uncapped, the GPU's
allocation crept 2 GB between the samples of one 1,118-token comparison, a
model that peaks at 17.2 GB alone. A job whose primary size does not fit stops
before timing anything and says so. One run is not the whole story: a
comparison holds more of the model's memory between its samples, so each
comparison's own peak (its `peak_gb`, counted from the comparison's start) is
read too, and a workload whose comparison came within the same 1 GB of the
limit stops being timed for the rest of the job, with every longer size of it
(`memory_untimed` in `run.jsonl`, `memory_fit.untimed_by_comparison`). A
check measured before that still counts. Qwen3.8-27B read 1,118 tokens at a
17.46 GB peak alone, while its comparisons there reached 19.02 GB and the next
ran out of GPU memory. After the artifact is exported (the job
has dropped its own copy of the model by then), each workload left untimed is
timed the way a user would time it: a fresh process builds the model once and
the runtime's `compare()` switches the artifact's layers in and out on that one
copy, ten alternated calls each, a prompt read to its first token through
MLX-LM. `final.fresh_process_timings` gives the median times, the speedup and
whether outputs matched, or why it could not run there (out of memory too,
say). These numbers are reported, not used to decide anything. While the job
waits on its candidate checks and the judge's tries, each its own process,
macOS may compress the job's idle model weights; restored by the GPU's first
command, 13 GB of Qwen3.8-27B's took past Metal's timeout and killed the job.
So before any GPU work, a job with more than 0.5 GB compressed reads every
page of its weights back from the CPU first (`paged_in`: how much was
compressed, what was touched, and the seconds it took: 1 to 49 on Qwen3.8-27B). `cooling_scheduled` means a completed
comparison or correctness check has returned its verdict and CPU work may use
that cooldown. Candidate workers also return their remaining cooling deadline
to the parent (`cooling_adopted`), so process cleanup, preparation and judge
thinking can use the same interval. The parent waits before the next worker or
model evaluation; `cooling_reused` reports that remainder.
A pending cooldown does not mean the verdict is still being measured.
Step timing reuses a successful post-cooling warm-up instead of warming twice.
A warm-up samples until 10 readings in a row fail to improve, since the
GPU's clocks ramp over several short calls; once a call takes a second or
more, two are enough. Longer warm-ups of multi-second calls only heated the
chip: on Qwen3-VL-8B at 11 s a call, ten more readings rose 12%, and the
model timed against itself read the same either way.
Its median is printed and logged as `step_clock` in `session.jsonl` before
cooling; the workload-level summary follows when the clock returns.
`ladder_result.result.detail.pacing` records accounted work and time spent
sleeping inside each successful worker phase. Parent waits appear in
`session.jsonl`. These are wall-clock accounting figures, not direct GPU-active
time: compilation and mixed CPU/GPU checks are still charged conservatively.
Warmup entries include their first-call time and total work; `off_clock_work`
records capture/correctness phase time. 
The judge normally returns its initial plan and first kernel together. Once a
region has an installed kernel, `incumbent_screen` reports a fresh local
comparison against it. A resolved local regression skips whole-model timing;
this screening result is not a measured whole-model regression. Identity wrapper
checks share model passes while comparing each scope separately.

The `trace` row's `never_evaluated` count is work the model builds but never
runs (MLX is lazy; a result nothing returns, keeps, evaluates or reads never
executes), such as the logits of a prompt-processing call the library drops.
It is left out of regions, prices and the step floor, and
`coverage.discovery.never_evaluated_ops` totals it. The count includes the
calls that compute a constant the model only reads back as a number (an
attention scale `1 / mx.sqrt(head_dim)` passed as `scale=`): the call is
recorded with the number, and a compiled step holds none of those calls. No
region crosses one.

How a win gets installed: the `installation` row names the method for each
module it touches. `direct` means the replacement covers the whole module
call and is one guarded kernel call. Against a plain baseline (the model as
its library runs it, uncompiled), outside a filled KV cache, every other
module delivers by `replay`: uncompiled, as it runs and as the length-generic
code after the search ships, so the region clock and every whole-model check
compare against the plain operations users run. Replaying the whole of
Qwen3-30B-A3B this way read 1.000x, 0.999x and 0.9995x of the untouched model
at 42, 128 and 2,028 tokens, bitwise equal. `graph` delivery serves a
compiled baseline and a step over a filled cache, whose new positions only it
can follow. `graph` means the original module builds
its calculation as usual, the replacement is substituted into that graph, and
the result is compiled once and reused for calls of the same shapes. `replay`
means generated Python re-runs the module's recorded operations with the
replacement spliced in; it is kept for scopes graph substitution cannot
preserve, such as one whose compiled identity is not bitwise the original
or one that reaches state it was not handed. Replay serves only the cache
positions the job recorded. Graph delivery also serves other positions and
cache sizes of the same kind of cache: it re-traces the scope once per new
cache state (at most 8 per scope) and there substitutes only kernels marked
`position_free` when bound, those whose inputs are computed from the scope's
arguments and own weights alone, and only where their verified operations
appear with their verified input shapes; if one does not, the call runs the
original. Timing still covers only what the job measured.
Before the search, when a plain-baseline job delivers by graph (a step over
a filled cache), the model with its outermost certified scopes compiled and
empty is timed against the plain model (paired,
20 pairs at each workload; over a range, 12 pairs at five sizes from the
shortest to the longest, the primary among them; `baseline.clocks_ms`): if
compiling alone is clearly slower anywhere, every scope moves to replay.
A `graph_fallback` row names why a scope moved to replay, `delivery_settled`
lists the split before the search, and `graph_verified`
records that both the inspected and the compiled calculation contained exactly
the expected substitutions. Every method passes the same correctness and
whole-model timing checks, and the artifact carries whichever was measured.

With a `range` in the manifest (automatic for an MLX-LM prompt
read from an empty cache: every prompt length from 3 to one past MLX-LM's
prefill chunk, logged as `other_sizes_default`), a stage after the search
replaces each changed
scope's delivery with generated code that reads its size from an argument
(`S_<scope>` classes), when the scope's replay at the primary size and at the
recorded range ends and middle differs only in integers that are one exact
linear function of that size. `serve_refused` names each scope that does not
generalize; it keeps its searched delivery. The generated code runs the scope's
recorded operations and each kernel inside an `if lo <= _n <= hi` of its own,
as plain MLX calls: nothing is traced, rewritten or compiled at run time, so a
new size costs nothing extra. It is not compiled because `mx.compile` builds a
new graph for every input shape; on seven models its gain was nil for the
transformers and large only for recurrent ones, while each new size paid for a
compile. Every other call goes to the original module. Each size is served only after
these checks:

- **Every size, no GPU work:** the model runs once per size in the range with
  each scope checked in place. The generated code (kernels off) and the
  original module build MLX graphs from the same arguments, and the graphs,
  outputs and state writes alike, must be the same operations, attributes,
  constants and sharing. A scope serves the contiguous sizes around the
  primary size that pass (`scopes` in the report's `check_optimizations_for` section, with the
  first difference found).
- **Every size, in the sandbox:** each bit-exact kernel (the `preserving` rule)
  is checked against its recorded operations with outputs poisoned, with
  shader validation off, as the model runs, on every recorded copy at every
  size; at 32 sizes spread over the range it also runs three times and must
  match itself. With Metal's shader validation on it runs once more at every
  size for validation's out-of-bounds reports alone (on the first copy, where
  no input is an index, since shapes alone then set its addresses): a kernel
  compiled for validation can compute differently, and it is not what ships.
  Checked in full both ways, Qwen3.8-27B's down projection took 2.3 h over
  its 2,000 sizes, measured 2.5 h; this way 27 min. Inputs at a size
  come from a real pass of the model. Where the region's inputs are all
  floating point, a size is checked on the first rows of the longest real pass
  (a causal model's first n rows of a long prompt are what an n-token prompt
  computes), every recorded copy of the region in it: one copy is not enough,
  since gpt-oss-20b's expert kernel was off MLX's rounding at 124 rows (where
  MLX's own matmul switches method) on about nine copies in ten, and the one
  copy then checked matched. Running the model at each size instead took
  Qwen3.8-27B 13 hours per kernel. A region with any integer input but a
  weight is never cut: an index list keeps its values, which can point past
  the cut (on Qwen3-30B-A3B the unsort order kept an index of 16,254 among
  1,032 rows, and a correct kernel wrote out of bounds). For those the model
  is run at every value of the manifest's dimension whose layer size lands
  where the kernel is to serve, a batch at a time, and only up to the region's
  first four copies (MLX computes nothing past them). Those recordings hold
  only the inputs that change with the size and are deleted once checked.
  Inputs are never built or repeated up (gpt-oss groups each expert's
  tokens). A size with no real pass of its own (or, for a floating-point
  region, a longer one) is not served. Shader validation reports are matched to the size that produced
  them (the worker marks each size on its stderr); a report it cannot place
  fails its whole batch. This sweep takes no cooling pauses, its recordings
  included (`off_clock_work` with `paced: false`); nothing in it is timed.
  Each timing grid rests 60 s first instead.
- **Listed sizes (`{sizes: [...]}`) are each timed** and served where the
  kernel wins; over a range:
- **A grid of about ten sizes, then twelve random sizes, in the sandbox:** the
  model is run at each, in the manifest's own dimension (prompt tokens, image
  side), and the kernel is timed on the region's inputs from that pass, the
  layer's size read off them; it need not follow the dimension in a straight
  line. A cut of a longer recording can mislead: a short gpt-oss prompt sends
  its tokens to ~20 experts, a cut of a long one to one or two, and the kernel
  reads each expert's weights. A size no pass lines up with (the model runs
  other operations there) is not timed. The region
  clock's paired loop and relative margin (1% or three standard deviations)
  decide where the kernel wins or clearly loses. It is served over every
  stretch between two timed sizes it wins at, broken by a grid point it does
  not win at or any size it lost at; the random sizes, drawn inside those
  stretches, catch losses between grid points. `serve_kernel` logs each
  kernel's ranges, grid, samples and wrong sizes.

Kernels accepted within tolerance serve at the sizes the search checked them
at: the optimized size and every other workload's size that ran them, where
the whole-model tolerance checks and timings already covered them, unless the
per-size check below finds them wrong there. Elsewhere in
the range they go through the same per-size check and timing, with one
difference: instead of matching the recorded operations bit for bit, the
kernel's largest output difference from them (max_abs) at a size may be no
larger than the largest it had at the searched sizes it is right at
(`allowed_error` in `serve_kernel`). A kernel that measured exact there must
be exact everywhere it serves. This assumes an error the whole model tolerated at the searched
sizes is tolerated at the others; the final check at the serve sizes is the
only whole-model test of that. Starters
and kernels whose operations take size-dependent settings keep their recorded
size. The final check then runs
on the served model, also at the recorded serve sizes. If outputs change, or
the patched model is slower, at up to three of the lengths it checks, each of
those lengths goes back to the original out to the nearest checked lengths
that held, and the whole final check runs again (`excluded` in the report).
Otherwise, or if that fails too, `serve_withdrawn` records why, the searched
wrappers go back, and the final check runs again on them.

The default requested baseline is the compiled model. Under library
inference it is realized as the model with its outermost compilable scopes
compiled and nothing inserted (`delivery_settled` names the scopes); that
model is installed empty as the job's starting state, every kernel composes
into it, and every whole-model number is measured against it. The artifact
carries it. `baseline: plain` in the manifest keeps the plain model as the
baseline instead; kernels are then measured against their scope compiled with
the cuts it carried before (`incumbent_compiled_scopes` on the accepted row).
A directly called model is compiled whole unless the recording shows state
that `mx.compile` cannot see: arrays the model keeps after the step, a cache
write, or reading back a value computed from the step's inputs or weights
(`.item()` or `float(x)` on one, as a branch does). The `baseline` row names
which. Reading a constant computed from Python numbers alone, such as an
attention scale `1 / mx.sqrt(head_dim)`, does not count: compile evaluates it
too. The same rule keeps a scope that reads such a value off graph delivery.
Each region's report row says its `delivery` and which `library_arm` its
clocks ran (a graph scope's library ops run as one compiled graph, the way the
deployed scope will).

Candidate workers have a separate ten-second limit for each GPU evaluation,
including the first compilation and launch. Cooling does not consume that
limit; it still counts toward the overall worker budget. If either deadline
expires, the worker is killed. Once it exits, a fresh worker must complete a
small checked GPU operation, then wait until a fixed matmul runs within 1.5x of
its time recorded before the first candidate: killing a worker does not stop
GPU work it already submitted. Success rejects only the candidate and schedules
cooling before search continues. A wrong answer, or a GPU still busy after two
minutes, stops the job.

Keep other GPU work quiet. Interleaved measurements reduce drift, but cannot
guarantee that arbitrary background load affects both arms equally. The macOS
GPU utilization counter does not measure remaining throughput; compare it
with measured bandwidth, compute throughput, and timing noise.
Bandwidth is measured as read speed (1 GB read four times per sample, best of
six), because a model step's memory traffic is almost all reads. On an M4 it
reads about 106 GB/s when quiet. Other GPU work lowers that reading and, with
it, every memory-bound room estimate: a decode run started while the GPU was
79% busy measured 75 GB/s and reported 0% room.
The CLI prevents a second CLI job from running at the same time. If the A/A
control finds a significant difference between identical code twice, the
run stops before search because those measurements could create false wins.

Every whole-model decision times the two versions in alternating pairs and
reads the median of the per-pair differences. A win counts once that median
gain is more than twice its standard error (`model_confidence_sigma` in the
report's constants): shipping a kernel, its independent repeat, the final
check that permits export, and every `win_confirmed` in the report. A
slowdown counts at three times (`model_regression_sigma`), and the A/A
control calls any difference at three times either way. Two was chosen over
three for decode steps, where single-kernel gains of 0.5 to 1% landed at 2 to
2.9 standard errors and none shipped. Region clocks keep their own margin:
1% of the library time or three standard deviations.

Over a manifest's `range` the search runs at ten sizes
(`search_sizes` in the report's constants), each an ordinary workload.
Regions are priced (ranked, and the judge's target picked) at three of them:
the shortest, the primary and the longest, and the job-start step clock and
room line are taken at those three too. The
region clock times the kernel at every one of them (`sizes` and `size_gate` in
the ladder result), each distinct input shape once: MLX-LM ends every prompt
with a 1-token call, the same shape at every length, so it is one more case,
not ten. A kernel is nominated only when it wins at half of
them or more and clearly loses at none; the judge sees each size's library and
kernel time. The ten sizes are fixed for the job, so a kernel can be fitted
to them; before the whole-model check, a nominated kernel is checked and
clocked at three more sizes drawn fresh for that check (one per third of the
range on a log scale, recorded from real passes of the untouched model). Wrong
there, or clearly slower than the library, it stops (`holdout` in the run
log); it need not be faster there. Against an installed kernel, a candidate clearly slower at more
sizes than it is clearly faster is a resolved regression. The whole-model
check then checks outputs at every size and times three (the shortest, the
longest, and the one where the region clock found the kernel weakest) with 8
pairs each; it needs no resolved slowdown and resolved wins at two of the
three, and its independent repeat times the same three the same way. The
final check adds ten fresh sizes, drawn from the same bins with another seed
and none of them a search size: it checks outputs at all ten and times the
whole model at five of them, spread smallest to largest, beside the ten search
sizes (`final.headline.curve`), and reports the geometric mean over the timed
fresh sizes and the worst size. A kernel tuned to the search's
sizes cannot lift that number. The final is decided at the ten search
sizes, with 8 pairs each (`final_benchmark.pairs` if higher). It needs a
clear win of at least 1% at two of them and no size more than 1% slower:
every final counts a difference under 1% of the step as neither, since a
smaller total gain is not worth shipping and, without that margin, an
identical model read clearly slower at 3 of 57 sizes with 4 pairs and 1 of 38
with 8. Simulated on the paired noise measured on Qwen3.8-27B and Qwen3-VL-8B
(about 0.3% of a step per reading, 1% on a bad stretch), a model with no real
change passes 0% of the time (1% when noisy), a real 1% gain ships 99% (91%),
and 1.5% always. The
fresh sizes are timed with 8 pairs for the headline and do not decide the
final; a veto at more sizes would fail good kernels by chance.

A model called once per use (`final_benchmark.steps: 1` with direct calls,
such as reading a chat prompt) is decided by that paired check alone: it
already times exactly one call, so there is no run of calls in a row to time.
With more steps (a denoising loop, a reply's tokens) the final also times
whole runs of that many calls, back to back.

An MLX-LM prompt job's final check also times generation: both models write
128 tokens after the same 512-token prompt (all but its last token prepared
untimed), paired, 8 pairs (`final.decode`). It is reported, and fails the
final only when the optimized model is clearly slower, so a kernel built for
reading prompts cannot slow down writing.

It also times a second message (`final.second_message`): both models read the
job's prompt with a first message of the same length already in their
history (prepared untimed), paired, 8 pairs. That is what every later turn of
a chat waits for, and where a kernel delivered through a recording of an empty
cache gains nothing. It too is reported, and fails the final only when the
optimized model is clearly slower.

A kernel whose whole-model gain is too small to resolve (neither a win nor a
slowdown) is not rolled back: it stays installed as `pending`, and later
regions are searched on the model that includes it. Every whole-model check
is timed against the confirmed installation, never against pending kernels:
a pending kernel's scope keeps its delivery in the comparison arm, carrying
only its confirmed cuts. When a check resolves a win, the new kernel and
every pending one ship together, each with its own row in `accepted`
(`confirmed_with` names the set). A resolved slowdown or changed outputs
roll back only the new kernel, and a win that does not repeat in its
independent check is rolled back as before. Kernels still pending when the
search ends are dropped (`pending_dropped`), so only a resolved win ships.
On GLM-4 9B decode, three kernels that each stayed unresolved alone resolved
together twice, at 1.98 and 2.22 ms of a 62 ms step. A region whose kernel
ships later this way still shows nothing shipped in its own region row;
`accepted` is the record.

Correctness is always judged on the uncompiled model, including the final run
of consecutive steps; under a compiled baseline those steps are compiled only
to be timed. Compiling a chain of small operations changes its rounding by
itself (on a RoPE step, a third of the outputs), so a kernel that matches the
model bit for bit cannot match the compiled model.
