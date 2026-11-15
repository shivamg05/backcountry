"""The job driver and region loop (spec "The loop" and "Closing a region").

One hypothesis at a time. The harness owns measurement, correctness, bind, and
the artifact; the judge proposes. Verdicts: failed, correct-but-slower (climb),
tentative ship, then bind and e2e decide whether a ship is real.
"""

from __future__ import annotations

from backcountry_runtime.numeric import DEFAULT_TOLERANCES

import importlib.util
import json
import math
import copy
import gc
import random
import shutil
import statistics
import sys
import re
import time
import traceback
import weakref
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

import mlx.core as mx
from mlx.utils import tree_flatten

from . import manifest as manifest_mod
from . import resume as resume_mod
from .bind.certify import certify_identities, find_scope_call, scope_calls_at, screen_scope
from .bind.emit import (EmittedWrapper, NotReplayable, Splice, compose_scope_variants,
                        emit_wrapper, emit_wrapper_variants)
from .bind.swap import install as swap_install, uninstall as swap_uninstall
from .bind.verify import verify_retrace
from .e2e import E2EResult, _flatten_params, preserving_check, run_e2e, share_weights
from .judge.directions import OPENERS, RUNNER_UP_FOLLOW_UPS, RUNNER_UP_MARGIN
from .knowledge import Knowledge, knowledge_file, outcome as knowledge_outcome
from .judge.prompts import ENVELOPE_KEYS, render_region_state
from .judge.queue import Queue, QueueError
from .judge.schema import JudgeBabble, MalformedResponse, _proposal
from .ladder.gates import EvalSet, LadderJob, LadderResult, run_ladder
from .ladder.static_checks import RegionContract, SINGLE_GROUP_ELEMENT_LIMIT, check as static_check
from .artifact.bundle import ModelBundle
from .artifact.emit import check_apply_many, check_model_run, drop_model_run, emit_artifact, write_kernel
from .log import RunLog, TextLog, wall_now
from .measure.clocks import chained_steps, compare, comparison_from_samples, loop_iterations, step_clock
from backcountry_runtime.graph import GraphWrapper
from .measure.controls import aa_null
from .measure.peaks import (BUSY_GPU_PERCENT, gpu_core_count, gpu_facts, gpu_utilization,
                            implausible as peaks_implausible, measure_peaks)
from .measure.heat import Heat
from .measure.session import Session, Stopped
from .measure.sequences import compare_sequences
from .regions.build import build_stretches, is_view, weight_like_ids
from .regions.fingerprint import group_copies
from .regions.price import PRICE_PAIRS, _boundary_shapes, capture_boundaries, capture_instances, op_times, price_group
from .regions.rank import (REGION_FLOOR_P, SIMILAR_OPS, apply_floor, demote_similar, estimate_regions,
                           free_members, length_weight, rank, select_frontier, Taken)
from .regions.roofline import observed_peaks, step_floor, stretch_roofline
from .regions.store import BoundaryStore
from .regions.types import Region, Stretch
from .report import Report
from .sandbox.watchdog import GPU_WINDOW_S
from .scaffold.lower import SINGLE_GROUP_WORK_LIMIT
from .trace import Tracer
from .trace.recorder import UNKNOWN_SOURCE_OP, ArrayRef, varying_constant_scopes
from .trace.walk import flatten_arrays
from .regions.sweep import SweepDivergence, locate_span
from .trace.serialize import nodes_to_json
from .trace.types import Trace
from .workload import bind_shape, catch_tokenizer, context_tokens, materialize, text_tokens, workload_seeds
from backcountry_runtime.kernels import KernelSpec, _loaded as _loaded_kernel
from backcountry_runtime.exact import bitwise_equal
from backcountry_runtime.swap import copy_model, require_independent_models
from backcountry_runtime.sequence import make_sequence, generation_throughput
from backcountry_runtime.state import ContextSequence, ContextStep, context_step
from backcountry_runtime.stats import LOSS_SIGMAS, WIN_SIGMAS, workload_win

MIN_WIN_MS = 0.030   # the most a win must save per step across copies: a few tens of microseconds
MIN_WIN_SHARE = 0.01  # or 1% of the step when that is less (2026-09-18: 18 resolved wins of 5-16% of a
                      # 20-300 us MetalBench step sat under the fixed floor; unchanged for steps over 3 ms)
CLOCK_PAIRS = 32    # ABBA pairs behind the ship clock and the headline
# ABBA pairs behind a judge's ./try clock: enough to rank designs and see a
# large loss; the full ladder's CLOCK_PAIRS decide what ships
QUICK_TRY_PAIRS = 4
TRIES_SHOWN = 24  # the latest tries a later reply sees
# A kernel inside a block handed the cache serves, through a recording of that
# block, only the cache positions it saw and UNRECORDED_STATE_TRACES more. The
# judge's own method for the block is checked over this many decode calls after
# each of three prompts, well past those positions, and gets this many tries.
LIVE_STEPS = 40
LIVE_ATTEMPTS = 2
# Each comparison has ten ABBA blocks (forty forward calls). Resolved wins
# receive one fresh comparison before installation; inconclusive results stop.
SHIP_PAIRS = 20
# Over a range of sizes the whole-model check times three of them (the
# shortest, the longest, and the one the region clock found the kernel weakest
# at) with fewer pairs: through MLX-LM's own call, 4 blocks resolved 0.3 ms at
# 128 tokens against 55-96 ms wins on Qwen3-30B-A3B. It needs no resolved loss
# and resolved wins at two of the three.
MODEL_CHECK_SIZES = 3
MODEL_CHECK_PAIRS = 8
MODEL_CHECK_WINS = 2
# The final check over a range decides at all ten search sizes with 8 pairs
# each, needing resolved wins at FINAL_RANGE_WINS of them. Every final counts
# a difference under FINAL_MARGIN of the step as neither a win nor a loss: a
# smaller total gain is not worth shipping, and with a zero margin an
# identical model read clearly slower at 3 of 57 sizes with 4 pairs, 1 of 38
# with 8 (Qwen3-0.6B, 2026-10-07). Simulated on the paired noise measured on
# Qwen3.8-27B and Qwen3-VL-8B (0.3% of a step per reading, 1% on a bad
# stretch), a model with no real change passes 0% (1%) of the time, a real 1%
# gain ships 99% (91%) and 1.5% always; 10 pairs changed those by under 5 points.
FINAL_RANGE_PAIRS = 8
FINAL_RANGE_WINS = 2
FINAL_MARGIN = 0.01
# A workload is timed only when one run of it peaks at least FIT_HEADROOM_GB
# of MLX memory under the GPU's working-set limit, and MLX's cache is capped so
# the largest timed one plus the cache stays there too. MLX's count leaves out
# 0.5-0.8 GB the GPU holds for the job: Qwen3.8-27B (24 GB Mac mini, 19.07 GB
# limit) read 1,833 tokens at an 18.15 GB MLX peak alone and ran the job out
# of GPU memory at 19.2 GB allocated in every timing there, and at 1,118 tokens
# (17.2 GB alone) as the allocation crept 2 GB between one comparison's samples.
FIT_HEADROOM_GB = 1.0
# Share of the model's weights the job may still hold when its artifact's
# fresh process loads them again: past it, the check would hold two copies.
HELD_WEIGHTS_SHARE = 0.1


class CheckError(RuntimeError):
    """What `backcountry check` found that would stop the job: said, not raised."""
MAX_CUT_LENGTHS = 3  # lengths a final may lose and still serve every other one; more falls back to the searched sizes
# The final's fresh sizes feed only the headline (its mean and worst size):
# the plan's 4 alternating blocks, so the worst size is not a 2-sample reading.
FINAL_FRESH_PAIRS = 8
FINAL_FRESH_TIMED = 5  # of the ten fresh sizes, spread smallest to largest
# Before that, a nominated kernel is checked and clocked at three sizes drawn
# fresh for each check (the search's ten are fixed for the job, so a kernel
# can be fitted to them): wrong there or clearly slower, it stops.
HOLDOUT_SIZES = 3
UNPRICED_FLOOR_FRACTION = 0.25  # estimated share under this fraction of the floor: not priced
# Over a range, the compiled clock (plain baseline: is compiling the delivery
# scopes, empty, slower than the plain model?) times five sizes with 12 pairs.
# At all ten with 20 it took 23 min of GPU work on Qwen3-30B-A3B before
# cooling; one false "slower" sends the whole job to replay, and 5 x 12 keeps
# about today's chance of that (~7%, one-sided t at 3 SE).
COMPILED_CLOCK_SIZES, COMPILED_CLOCK_PAIRS = 5, 12
# MLX-LM jobs: the final check also times generation, 128 tokens after a
# 512-token prompt (the common TG128 point), so a kernel built for reading
# prompts cannot slow down writing. Reported; it fails the final only when
# clearly slower.
DECODE_PROMPT, DECODE_TOKENS = 512, 128
@dataclass
class PromotionResult:
    status: str
    reason: str = ""
    checks: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def __bool__(self):
        return self.status == "shipped"

    @property
    def outcome(self):
        if self:
            return "shipped"
        return "rolled_back" if self.status in {"binding_failed", "correctness_failed"} else "correct_slower"


def _promotion_feedback(result, promotion):
    """Keep local nomination separate from the whole-model decision."""
    if "ship" in result.detail:
        result.detail["region_nominated"] = result.detail.pop("ship")
    result.detail["model_check"] = asdict(promotion)


@dataclass
class RegionRun:
    region: Region
    scaffold: KernelSpec | None = None
    head: KernelSpec | None = None
    shipped: KernelSpec | None = None
    head_ms: float | None = None
    shipped_ms: float | None = None
    head_ratio: float | None = None   # head_ms over the library beside it: the drift-free number
    shipped_ratio: float | None = None
    head_floor_ms: float | None = None     # the floor probe clocked beside head, same child
    shipped_floor_ms: float | None = None
    head_sigma_ms: float | None = None     # uncertainty of head's win, same clock
    head_age: int = 0                      # evaluated kernels since head last moved
    tries: int = 0                         # the judge's ./try checks (advisory, not attempts)
    try_log: list = field(default_factory=list)  # their results, shown in later replies
    best_try: tuple | None = None          # (kernel_ms / library_ms, kernel id) of the fastest correct try
    floor_streak: int = 0                  # consecutive kernels within one sigma of the floor beside them
    openers: list[str] = field(default_factory=list)      # kinds the widening round opened
    opener_kernels: tuple = ()        # the openers' kernel ids, in order
    runner_up: str | None = None      # the second-best opener, when it was close to the best
    runner_up_left: int = 0           # follow-ups its branch still takes
    past_runs: list | None = None     # the index of earlier jobs' attempts, fixed once it has any
    op_ms: dict | None = None         # where the library's time goes, per op, read once
    head_tag: str = "preserving"      # assoc tag of the edit that produced head
    head_workload: str | None = None
    shipped_workload: str | None = None
    last_kernel: str | None = None    # the kernel the latest verdict was about
    attempts: dict[str, dict] = field(default_factory=dict)  # kernel id -> its verdict
    hypotheses: int = 0
    errors: int = 0         # consecutive transport failures; three pause the job
    refused: int = 0        # consecutive replies with nothing to evaluate
    empty_replies: int = 0  # all such replies, to name each one once
    close_rule: str | None = None
    kernels: dict[str, KernelSpec] = field(default_factory=dict)
    queue: Queue | None = None
    last_verdict: dict | None = None


class JobRunner:
    use_library_inference: bool | None = None  # resolved against build() before tracing
    text_tokenizer = None  # set when prompts are real text: an MLX-LM job whose build() loaded a tokenizer
    knowledge: Knowledge | None = None  # what earlier jobs found; set for every real job
    pending: dict = MappingProxyType({})  # none until __init__; read-only for partial runners

    def __init__(self, manifest_path: str | Path, work_dir: str | Path,
                 judge_factory, session: Session | None = None,
                 clock_pairs: int = CLOCK_PAIRS, resume: bool = False):
        self.manifest = manifest_mod.load(manifest_path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # a resumed job continues this directory's logs from its saved state
        self._resumed = resume
        self._resume_state = resume_mod.load(self.work_dir) if resume else None
        # an identical: output job stopped after its loop search resumes from that search's result
        self._finished_loop = resume_mod.finished_loop(self.work_dir) if resume and not self._resume_state else None
        # one stopped during its loop search continues that search from its last attempt
        self._loop_search = resume_mod.loop_search(self.work_dir) \
            if resume and not self._resume_state and not self._finished_loop else None
        if resume and self._resume_state is None and self._finished_loop is None and self._loop_search is None:
            raise RuntimeError(f"{self.work_dir} has no saved state ({resume_mod.STATE}) to resume from")
        if (self.work_dir / "run.jsonl").exists() and not resume:
            # run.jsonl appends; a reused dir would interleave two jobs' rows
            raise RuntimeError(
                f"{self.work_dir} holds a previous run (run.jsonl exists); "
                "pass a fresh --work-dir or move the old one aside")
        self._search_state: dict | None = None  # the search's position, saved at every safe point
        self._resumable = True  # this job saves its state at safe points
        self.judge_factory = judge_factory
        self.clock_pairs = clock_pairs
        self.baseline = self.manifest.baseline  # settled by _clock_steps once the traces are in
        # the workload whose call is a step over a filled KV cache, set by load_model
        self.context = next((w for w in self.manifest.workloads if w.context is not None), None)
        self.context_seed = self.context_tokens = None
        self.session = session or Session(log_path=self.work_dir / "session.jsonl", heat=Heat.for_machine())
        self.log = RunLog(self.work_dir / "run.jsonl")
        self.session.on_memory = lambda row: self.log.append("memory", **row)
        self.session.on_heat = self._heat_tripped
        self.candidates = TextLog(self.work_dir / "candidates.log")
        self.kernel_dir = self.work_dir / "kernels"
        # hours run before this process, and its start: a loop search resumed part way goes on from its own
        self._clock_base = ((self._loop_search or {}).get("active_s", 0.0), time.monotonic())
        self.report = Report(manifest_path=str(manifest_path))
        if self._finished_loop:  # kept in every report this process writes: one stopped before it
            self.report.model_level = dict(self._finished_loop)  # set it again would lose it to the next resume
        self.knowledge = Knowledge(knowledge_file(), job=str(self.work_dir.resolve()), setup={
            "chip": mx.device_info().get("device_name"), "mlx": mx.__version__,
            "model": Path(self.manifest.model_path).name, "model_type": self.manifest.model_type})
        self.report.constants = {
            "min_win_ms": MIN_WIN_MS, "min_win_share": MIN_WIN_SHARE,
            "budget_per_region": self.manifest.budget_per_region,
            "budget_total": self.manifest.budget_total, "seed": self.manifest.seed,
            "openers_per_region": OPENERS,
            "clock_pairs": self.clock_pairs, "price_pairs": PRICE_PAIRS,
            "ship_pairs": SHIP_PAIRS, "model_min_gain_pct": 0.0,
            "model_confidence_sigma": WIN_SIGMAS, "model_regression_sigma": LOSS_SIGMAS,
            "model_confirmation_comparisons": 1,
            "final_benchmark": asdict(self.manifest.final_benchmark),
            "numeric_policy": "exact_preserving_baseline_tolerance_changing",
            "tolerances": self.manifest.tolerances,
            "tolerance_defaults": dict(DEFAULT_TOLERANCES),
            "duty_idle_factor": self._rest_factor(),
            "pacing": self._pacing(),
            "gpu_evaluation_timeout_s": GPU_WINDOW_S,
            "single_group_element_limit": SINGLE_GROUP_ELEMENT_LIMIT,
            "single_group_scaffold_work_limit": SINGLE_GROUP_WORK_LIMIT,
            "defaulted": list(self.manifest.defaulted),
        }
        self.store = BoundaryStore(self.work_dir / "boundaries")
        self.tracer = Tracer()
        self.traces: dict[str, Trace] = {}
        self.tensors: dict[str, list[mx.array]] = {}
        # the sweep: one trace and one input set per named dim size, keyed
        # "workload@dim=size", and each region's span located in that trace
        self.sweep_traces: dict[str, Trace] = {}
        self.sweep_tensors: dict[str, list[mx.array]] = {}
        self.sweep_spans: dict[tuple[str, str], Stretch] = {}
        # manifest serve: the model recorded at more sizes, for generating
        # size-generic delivery after the search; never used by the search
        self.serve_traces: dict[str, list[Trace]] = {}
        self.serve_tensors: dict[str, list[mx.array]] = {}
        # scopes whose constants changed with the inputs at some size (a value
        # read from the data after a mid-pass evaluation): they host no kernel
        self.data_constants: set[str] = set()
        self.step_ms: dict[str, float] = {}
        self.step_chain: dict[str, int] = {}  # dependent steps per timed sample, per workload
        self.peaks = None  # measured by measure_machine, after the step clock
        self.gpu_busy_at_start = gpu_utilization()  # the counter trails: read before any GPU work of ours
        self.total_hypotheses = 0
        self.lessons: list[dict] = []  # what the judge wrote down for later regions
        # scope -> (original module, {(workload, call address): splices}, kernels)
        self.installed: dict[str, tuple] = {}
        self.scaffold_overrides: dict[str, KernelSpec] = {}
        self.cuts: dict[str, dict[tuple[int, int], str]] = {}  # workload -> {span: kernel id}
        self.certified_scopes: set[str] = set()
        self._compiled_baseline = None  # under a compiled baseline: the untouched model with its baseline scopes compiled
        self.replay_scopes: set[str] = set()
        self.emitted: dict[str, EmittedWrapper] = {}
        self.live_scopes: set[str] = set()  # blocks that call their kernels from their own method
        self.final_ok = False  # export requires a completed final check
        # Historical product of accepted step ratios, for diagnostics only.
        # Ship decisions remeasure both candidate and installed incumbent.
        self.model_ratio = 1.0
        self.model_ratios = {w.name: 1.0 for w in self.manifest.workloads}
        self.pending_regions: list[Region] = []
        self.selection_wave = 0
        self.shipped_tags: dict[str, str] = {}
        self.shipped_regions: dict[str, Region] = {}  # fingerprint -> the region shipped
        # Kernels kept installed while the whole model cannot tell them from
        # noise; they ship together once a whole-model check resolves a win.
        self.pending: dict[str, dict] = {}  # fingerprint -> {region, kernel, assoc_tag}
        # The installation as last confirmed: every whole-model check is timed against it.
        self.confirmed = {"emitted": {}, "kernels": {}, "variants": {}, "cuts": {}}

    # -- stage 1: model and traces -------------------------------------------

    def _weights(self):
        """The weights of every model copy the job holds (shared buffers repeat)."""
        for model in (self.model, self._pristine, self.baseline_model):
            if hasattr(model, "parameters"):
                yield from _flatten_params(model.parameters()).values()

    def load_model(self):
        from backcountry.harness.artifact.bundle import (check_model_dependencies, capture_checkpoint_loads,
                                               resolve_model_checkpoints)
        check_model_dependencies(self.manifest.model_path)
        from backcountry_runtime.graph_native import prepare
        prepare()  # build/ABI failures must happen before model search
        manifest_mod.check_build(self.manifest)
        self.tracer.install(model_module_name="autotune_model")
        try:
            spec = importlib.util.spec_from_file_location("autotune_model", self.manifest.model_path)
            module = importlib.util.module_from_spec(spec)
            with capture_checkpoint_loads() as captured:
                spec.loader.exec_module(module)
                self.model_module = module
                self.build_model = self._build_model
                self.model = self.build_model()
                self._pristine = self._try_copy(self.model)
                self.baseline_model = self._fresh_model()
                require_independent_models(self.model, self.baseline_model)
            self.checkpoint_pins = resolve_model_checkpoints(self.manifest.model_path, captured=captured)
            self.report.constants["checkpoint_pins"] = self.checkpoint_pins
        except BaseException:
            self.tracer.uninstall()  # a failed load leaves no patches behind
            raise
        shared = share_weights(self.model, self.baseline_model)
        mx.clear_cache()  # return the freed second weight copy to the OS
        self.session.resident = self._weights
        total = len(_flatten_params(self.baseline_model.parameters())) \
            if hasattr(self.baseline_model, "parameters") else 0
        context = None if self.context is None else {
            "workload": self.context.name, "tokens": self.context.context, "seed": self.context_seed}
        self.log.append("model", shared_weights=shared, parameters=total, context=context)
        self._default_other_sizes()
        self._search_at_sizes()
        if shared < total:
            self._env_warning(f"only {shared} of {total} parameters could be shared between "
                              "the two model copies; both stay resident, which adds noise "
                              "to the whole-model checks")

    AUTO_SIZE_DIM = "prompt_tokens"

    def _search_at_sizes(self) -> None:
        """check_optimizations_for names the sizes the user cares about, so the
        search runs across them, not at one: each workload with that dim gets
        a sibling at every size of manifest.search_sizes but the primary (its
        own), with that size fixed. Each sibling is an ordinary workload: every
        region is captured, checked and timed there, and the judge sees how its
        kernel does at each. A kernel tuned to one size (Laya's tiles fitted to
        93 rows lost 10-20% at 64, 128 and 512) shows up while it is written,
        not after the search."""
        from dataclasses import replace
        m = self.manifest
        if not getattr(m, "serve", None):
            return
        state = getattr(self, "_resume_state", None)
        if state is not None and "search_sizes" not in state["report"].constants:
            return  # a job started before the search ran across sizes resumes as it was
        dim, span = next(iter(m.serve.items()))
        sizes = manifest_mod.search_sizes(span, m.primary[dim], m.seed)
        have = {tuple(tuple(i.shape) for i in w.inputs) for w in m.workloads}
        added = []
        for w in m.workloads:
            if dim not in w.named_dims():
                continue
            for n in sizes:
                fixed = replace(w, name=f"{w.name}@{dim}={n}", context=None, inputs=tuple(
                    replace(i, shape=tuple(n if d == dim else d for d in i.shape)) for i in w.inputs))
                shapes = tuple(tuple(i.shape) for i in fixed.inputs)
                if n != m.primary[dim] and shapes not in have:
                    added.append(fixed)
                    have.add(shapes)
        if added:
            self.manifest = replace(m, workloads=m.workloads + tuple(added))
        self.report.constants["search_sizes"] = {dim: list(sizes)}
        self.log.append("search_sizes", dim=dim, sizes=list(sizes), workloads=[w.name for w in added])

    def _default_other_sizes(self) -> None:
        """An MLX-LM prompt read from an empty cache, with no
        check_optimizations_for: check the kernels at every prompt length
        from 3 tokens (a shorter prompt takes another code path) to one past
        MLX-LM's prefill chunk, the most rows one call reads. Nothing else is
        guessed: for any other workload the manifest names the dimension."""
        m = self.manifest
        if m.serve or not m.serve_auto or not self.use_library_inference or len(m.workloads) != 1:
            return
        w = m.workloads[0]
        spec = w.inputs[0]
        if w.context not in (None, 0) or len(w.inputs) != 1 or not all(isinstance(d, int) for d in spec.shape):
            return
        chunk = _prefill_chunk()
        tokens, lo, hi = spec.shape[-1], 3, chunk + 1
        if not lo < tokens <= hi:
            return
        from dataclasses import replace
        from types import MappingProxyType
        dim = self.AUTO_SIZE_DIM
        named = replace(w, inputs=(replace(spec, shape=spec.shape[:-1] + (dim,)),))
        if self.context is w:
            self.context = named
        self.manifest = replace(m, workloads=(named,), primary=MappingProxyType({**m.primary, dim: tokens}),
                                sweep=MappingProxyType({**m.sweep, dim: (tokens,)}),
                                serve=MappingProxyType({dim: manifest_mod.SizeSet(lo, hi)}))
        self.report.constants["check_optimizations_for"] = {dim: [lo, hi], "set": "automatically"}
        self.log.append("other_sizes_default", dim=dim, range=[lo, hi], optimized=tokens)

    def _try_copy(self, model):
        """A copy of the model as built, kept untouched as the source of every
        later copy; None, logged, when the model cannot be copied, and every
        copy is then a fresh build()."""
        try:
            return copy_model(model)
        except Exception as error:
            self.log.append("model_copy_unavailable", reason=f"{type(error).__name__}: {error}")
            return None

    def _fresh_model(self):
        """An independent model as build() returns it. A copy shares the
        weights from the start; a second build() holds two sets until they are
        shared, which a large model cannot fit."""
        return self.build_model() if self._pristine is None else copy_model(self._pristine)

    def _materialize(self, w, dims, seed):
        """A workload's inputs at these sizes: the prompt as real text when
        the job has the model's tokenizer, everything else seeded random."""
        return materialize(w, dims, seed, self.text_tokenizer)

    def _build_model(self):
        """Every build retains custom definitions, even between trace passes."""
        from backcountry_runtime.captured_kernels import capture_construction
        from .artifact.bundle import use_checkpoint_pins
        with capture_construction(), use_checkpoint_pins(getattr(self, "checkpoint_pins", None)), \
                catch_tokenizer() as tokenizer:
            model = self.model_module.build()
            self._built_tokenizer = tokenizer[0]
            # the step's first calls (a context fill) can define kernels lazily
            return self._as_step(model)

    def _as_step(self, model):
        """The model as the job runs it: as built, or, when the workload
        declares a context, as one repeatable step over the model's own KV
        cache filled with that many seeded tokens."""
        from backcountry_runtime.inference import AppRun, LibraryInference, resolve_library_inference
        state = getattr(self, "_resume_state", None)
        # A first input of token ids is the prompt: real text when build()
        # loaded the model's tokenizer. A job resumes with the inputs it started with.
        first = self.manifest.workloads[0].inputs[0]
        if self.text_tokenizer is None and first.dtype in ("int32", "int64", "uint32", "uint64") and first.shape \
                and (state is None or state["report"].constants.get("inputs") == "real text"):
            self.text_tokenizer = getattr(self, "_built_tokenizer", None)
        self.report.constants["inputs"] = "real text" if self.text_tokenizer is not None else "random"
        run = getattr(self.model_module, "run", None)
        if run is not None:
            # The model file says how its application uses the model: that call
            # is what is traced, checked and timed, in place of one model call;
            # a workload's conversation is handed to it ahead of its own tokens.
            if self.manifest.use_library_inference:
                raise manifest_mod.ManifestError(
                    "the model file defines run(), which is how the model is used: drop use_library_inference")
            self.use_library_inference = False
            self.report.constants.update(use_library_inference=False, app_run=True, measurement={
                "kind": "app_run", "generated_tokens": None, "baseline": self.manifest.baseline})
            w = self.context
            if w is not None:
                self.context_seed = workload_seeds(self.manifest.seed, f"{w.name}:context", 1)[0]
                self.context_tokens = context_tokens(w, self.manifest.primary, self.context_seed,
                                                     self.text_tokenizer)
            return AppRun(model, run, prefix_tokens=self.context_tokens if w is not None else None)
        selected = resolve_library_inference(
            model, self.manifest.use_library_inference, self.manifest.workloads,
            dims=self.manifest.primary, sweep=self.manifest.sweep)
        if self.use_library_inference is not None and selected != self.use_library_inference:
            raise manifest_mod.ManifestError("build() changed its inference compatibility between builds")
        self.use_library_inference = selected
        started_compiled = state is not None and state["report"].baseline.get("requested") == "compiled"
        if selected and "baseline" in self.manifest.defaulted and self.manifest.baseline != "plain" \
                and not started_compiled:  # a job resumes with the baseline it started with
            # The bar is the model as its library runs it, and MLX-LM runs it
            # uncompiled (compiling every MoE block of Qwen3-30B-A3B changed its
            # time to first token by 1.002x, 1.000x, 0.999x at 129, 513, 1,041
            # tokens). A manifest that names a baseline keeps it.
            from dataclasses import replace
            self.manifest = replace(self.manifest, baseline="plain")
            self.baseline = "plain"
        self.report.constants["use_library_inference"] = selected
        self.report.constants["measurement"] = {
            "kind": "library_generation" if selected else "forward",
            "generated_tokens": self.manifest.final_benchmark.steps if selected else None,
            "baseline": "unmodified library inference" if selected else self.manifest.baseline,
        }
        w = self.context
        if selected:
            if w is not None:
                self.context_seed = workload_seeds(self.manifest.seed, f"{w.name}:context", 1)[0]
                self.context_tokens = context_tokens(w, self.manifest.primary, self.context_seed,
                                                     self.text_tokenizer)
            return LibraryInference(model, steps=self.manifest.final_benchmark.steps,
                                    prefix_tokens=self.context_tokens)
        if w is None:
            return model
        self.context_seed = workload_seeds(self.manifest.seed, f"{w.name}:context", 1)[0]
        self.context_tokens = context_tokens(w, self.manifest.primary, self.context_seed, self.text_tokenizer)
        warm = self._materialize(w, self.manifest.primary, self.context_seed)
        try:
            step = context_step(model, w.context, self.context_tokens, warm, generated=w.generated)
        except TypeError as e:
            raise manifest_mod.ManifestError(f"workload {w.name!r} declares a context, but {e}") from None
        # several requests at once (a batch dim swept or ranged): each batch size's
        # conversation is filled now, before anything is recorded (filled during a
        # recording, the fill itself was recorded and its arrays held: 15 GB on Bonsai)
        batches = {bind_shape(w.inputs[0].shape, dims)[0] for _, dims in self._sweep_points(w)}
        batch_dim = w.inputs[0].shape[0]
        span = self.manifest.serve.get(batch_dim) if isinstance(batch_dim, str) else None
        if span is not None:
            batches |= set(span.listed or range(span.lo, span.hi + 1))
        step.prepare(batches)
        return step

    def trace_workloads(self):
        for w in self.manifest.workloads:
            seeds = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)
            self.tensors[w.name] = self._materialize(w, self.manifest.primary, seeds[0])
            trace, _ = self.tracer.trace(self._for(self.model, w.name), self.tensors[w.name])
            self.traces[w.name] = trace
            self._confirm_constants(trace, w, self.manifest.primary, seeds[1])
            if trace.in_pass_evaluation:
                self.log.append("memory_warning", workload=w.name,
                                detail="model evaluated mid-record; intermediates stayed resident")
            self.log.append("trace", workload=w.name, nodes=len(trace.nodes), never_evaluated=len(trace.dead))
            for label, dims in self._sweep_points(w):
                self.sweep_tensors[label] = self._materialize(w, dims, seeds[0])
                self.sweep_traces[label], _ = self.tracer.trace(self._for(self.model, w.name),
                                                                self.sweep_tensors[label])
                self._confirm_constants(self.sweep_traces[label], w, dims, seeds[1])
                self.log.append("trace", workload=label, nodes=len(self.sweep_traces[label].nodes))
            for dim, sizes in getattr(self.manifest, "serve", {}).items():
                if dim not in w.named_dims():
                    continue
                for size in sizes.recordings(self.manifest.primary[dim]):
                    label = f"{w.name}@serve:{dim}={size}"
                    self.serve_tensors[label] = self._materialize(w, {**self.manifest.primary, dim: size}, seeds[0])
                    trace, _ = self.tracer.trace(self._for(self.model, w.name), self.serve_tensors[label])
                    self.serve_traces.setdefault(w.name, []).append(trace)
                    self._confirm_constants(trace, w, {**self.manifest.primary, dim: size}, seeds[1])
                    self.log.append("trace", workload=label, nodes=len(trace.nodes))

    def _confirm_constants(self, trace: Trace, w, dims: dict[str, int], seed: int) -> None:
        """After a mid-pass evaluation a value read from the data is recorded
        like any constant, so trace the pass again on other inputs: a scope
        whose constants differ there hosts no kernel."""
        if not trace.in_pass_evaluation:
            return
        other, _ = self.tracer.trace(self._for(self.model, w.name), self._materialize(w, dims, seed))
        if varying := varying_constant_scopes(trace, other) - self.data_constants:
            self.data_constants |= varying
            self.log.append("data_constants", workload=w.name, scopes=sorted(varying))

    def _sweep_points(self, w) -> list[tuple[str, dict[str, int]]]:
        """Each named dim of the workload at each sweep size but the primary,
        the other dims held at their primary sizes."""
        points = []
        for dim in sorted(w.named_dims()):
            for size in self.manifest.sweep[dim]:
                if size != self.manifest.primary[dim]:
                    points.append((f"{w.name}@{dim}={size}", {**self.manifest.primary, dim: size}))
        return points

    # -- stage 2: regions -----------------------------------------------------

    def _serve_blocker(self, region: Region, follows: dict) -> str | None:
        """Over a range of sizes, why a kernel in this region could run only at
        the recorded sizes: its scope is not the same operations at the job's
        other recorded sizes, so the serve step cannot give it the sizes
        between, nor any kernel its scope would take into its replay (on
        gpt-oss-20b an attention-input region sat in the whole model body; its
        kernel took the five expert kernels into one whole-model replay, and
        none of them ran past the 13 recorded prompt lengths). follows caches
        each scope's answer."""
        from .serve import follows_size
        for m in region.members:
            others = self.serve_traces.get(m.workload)
            trace = self.traces[m.workload]
            scope = find_scope_call(trace, m.scope_stack) if others else None
            if scope is None:
                continue
            key = (m.workload, scope.address)
            if key not in follows:
                follows[key] = follows_size(trace, scope, others)
            if follows[key]:
                return (f"its scope ({scope.address}) cannot follow the size over the range ({follows[key]}): "
                        "a kernel there would run only at the recorded sizes, and so would any kernel its "
                        "replay took in")
        return None

    def _replays_marked_function(self) -> bool:
        from backcountry_runtime.tune import Request
        return isinstance(getattr(self, "baseline_model", None), Request)

    def build_regions(self) -> list[Region]:
        stretches = {name: build_stretches(t, name) for name, t in self.traces.items()}
        regions = group_copies(self.traces, stretches)
        # one generated copy serves every call of a scope, so a scope that makes
        # an unknown source, or a constant that changed with the inputs, in any
        # recorded call hosts none
        unknown = {address.rsplit("@", 1)[0] for t in self.traces.values() for n in t.nodes
                   if n.op == UNKNOWN_SOURCE_OP for address in n.module_stack}
        viable, follows = [], {}
        for r in regions:
            reason = screen_scope(self.traces[r.members[0].workload], r.members[0].scope_stack)
            scope_path = r.members[0].scope_stack[-1].rsplit("@", 1)[0]
            if reason is None and scope_path in unknown:
                reason = "a call of the scope makes an array the trace cannot explain (an unknown source)"
            if reason is None and scope_path in self.data_constants:
                reason = "a call of the scope makes a constant that changed with the inputs"
            if any(m.scope_stack[-1].rsplit("@", 1)[0] == "" for m in r.members):
                # Installation swaps a module; the model's own top-level call
                # has no parent to swap under, so no attempt could ever land.
                reason = "the region runs at the model's top level, where no module can be swapped"
            if reason is None and self._replays_marked_function() and any(
                    re.fullmatch(r"parts\.\d+", m.scope_stack[-1].rsplit("@", 1)[0]) for m in r.members):
                # a marked function holds its model itself (in its arguments or
                # globals): swapping the job's handle on a whole part would never
                # reach the object the function calls; layers inside it do
                reason = "the region is a whole part the marked function holds itself; only layers inside it swap"
            if reason is None and self.serve_traces and (self._generated() or not self._holds_cache(r)):
                # a scope that holds the cache gets a method of its own after the
                # search (live delivery), which runs at every size: only a
                # replay must follow the size
                reason = self._serve_blocker(r, follows)
            if reason is None and self._generated() and self._holds_cache(r):
                # a loop checks its guesses at every position, and a recorded
                # replay of a cache-holding scope serves only the recorded one
                reason = "its scope holds the cache, so its replay would serve one position of the loop's calls"
            if reason is None and self._decodes() and self._takes_layer_caches(r):
                # Generation reaches a new position every token. A recording of
                # a scope handed several layers' caches (the model body, for a
                # region crossing from one layer into the next) serves only the
                # recorded one, and a method of its own means rewriting the whole
                # model's forward: on gpt-oss-20b and Bonsai 2 9B that method
                # failed its check, the recording shipped, its kernel (and every
                # kernel it took in) never ran past the recorded position, and
                # replies were no faster. The same ops inside one layer remain.
                reason = ("its scope takes several layers' caches, so a kernel there would run only at the "
                          "recorded conversation length while generating")
            if reason is not None:
                r.rejected = f"no certified delivery scope: {reason}"
            else:
                self._plan_delivery(r)
                try:
                    self._build_scaffold(r)
                except Exception as error:
                    # Source generation is CPU-only. Discover an unsupported
                    # shape or argument before capturing and timing the cut.
                    r.rejected = f"cannot build scaffold: {type(error).__name__}: {error}"
            if r.rejected:
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
            else:
                if self._decodes() and self._holds_cache(r) and not self._reads_cache(r):
                    # While generating, a kernel in a block that holds the cache
                    # runs at new positions only through a rewrite of the whole
                    # block, which carries every other kernel inside it. A region
                    # that never touches the cache is there only because its ops
                    # reach the block's own code (Bonsai 2 9B: a matmul and the
                    # residual add after it); its twin without that code reaches
                    # the matmul at the matmul's own scope, so it goes first. On
                    # Bonsai the block rewrites made one request 13% slower and
                    # failed on 8 of 32 layers.
                    r.demoted = "a block that holds the cache, which its ops never read"
                viable.append(r)
        self.log.append("regions", candidates=len(regions), screened=len(viable))
        self.report.coverage["discovery"] = {
            "legal_regions": len(regions), "supported_regions": len(viable),
            "unsupported_regions": len(regions) - len(viable),
            "never_evaluated_ops": sum(len(t.dead) for t in self.traces.values()),
            # arrays the record could not explain: their scopes host no kernel
            "unknown_sources": sorted({n.module_address for t in self.traces.values()
                                       for n in t.nodes if n.op == UNKNOWN_SOURCE_OP}),
            # scopes whose constants changed with the inputs: they host no kernel
            "data_constants": sorted(self.data_constants)}
        return viable

    def _holds_cache(self, region: Region) -> bool:
        """Whether the region's scope reads or writes a cache (a state call)."""
        from .bind.emit import scope_nodes
        from .trace.recorder import state_method
        trace = self.traces[region.members[0].workload]
        scope = find_scope_call(trace, region.members[0].scope_stack)
        return scope is not None and any(state_method(n.op) for n in scope_nodes(trace, scope))

    def _takes_layer_caches(self, region: Region) -> bool:
        """Whether the region's scope is handed more than one cache (one per
        layer: the model body rather than one block)."""
        trace = self.traces[region.members[0].workload]
        scope = find_scope_call(trace, region.members[0].scope_stack)
        return scope is not None and len(set(scope.obj_ids)) > 1

    def _reads_cache(self, region: Region) -> bool:
        """Whether the region calls the cache or takes in what a cache call
        returned (attention over the history)."""
        from .trace.recorder import state_method
        trace = self.traces[region.members[0].workload]
        m = region.members[0]
        fetched = {a for n in trace.nodes if state_method(n.op) for a in n.out_arrays}
        return any(state_method(n.op) for n in trace.nodes[m.start_seq:m.end_seq + 1]) \
            or bool(fetched.intersection(m.input_ids))

    def _decodes(self) -> bool:
        """The job generates over a conversation: a workload decodes on a filled cache."""
        return any(w.context for w in self.manifest.workloads)

    def _plan_delivery(self, region: Region) -> None:
        """How a kernel will install at the region's scope, read from the
        record before any clock: the kernel call alone when the region is the
        whole scope; against a plain baseline outside a filled cache, replay;
        else graph insertion where compiling the scope keeps its Python
        behavior, and replay where it does not. Every clock's library arm follows this choice,
        so a kernel is measured against what its scope will actually run."""
        from .bind.emit import covers_scope, scope_nodes
        from .bind.graph import graph_scope_reason
        for member in region.members:
            if member.workload in region.delivery:
                continue
            trace = self.traces[member.workload]
            scope = find_scope_call(trace, member.scope_stack)
            path = scope.address.rsplit("@", 1)[0] if scope else ""
            if path == "":
                region.delivery[member.workload] = "replay"
                continue
            # One wrapper serves every recorded call of the scope, so every
            # call decides: a cut must be the whole calculation in each, and
            # a compiled scope must keep its Python in each.
            calls = scope_calls_at(trace, path)
            spans = {(m.start_seq, m.end_seq) for m in region.members if m.workload == member.workload}
            if all(covers_scope(nodes, {span for span in spans
                                        if nodes[0].seq <= span[0] and span[1] <= nodes[-1].seq})
                   for sc in calls if (nodes := scope_nodes(trace, sc))):
                region.delivery[member.workload] = "direct"
                continue
            if self.baseline != "compiled" and self.context is None:
                # The bar is the model as its library runs it, uncompiled, and
                # what ships over a range is uncompiled replay: a compiled scope
                # would time kernels against compiled ops users never run. A
                # replay copy costs nothing measurable (Qwen3-30B-A3B, the whole
                # model replayed: 1.000x, 0.999x, 0.9995x at 42, 128 and 2,028
                # tokens). Graph insertion stays for a compiled baseline and for
                # a step over a filled cache, whose new positions only it serves.
                region.delivery[member.workload] = "replay"
                region.delivery_reasons[member.workload] = "plain baseline: delivered uncompiled, as it runs"
                continue
            reason = next((r for sc in calls if (r := graph_scope_reason(trace, sc))), None)
            region.delivery[member.workload] = "replay" if reason else "graph"
            if reason:
                region.delivery_reasons[member.workload] = reason

    def capture_and_price(self, regions: list[Region]) -> list[Region]:
        """Clock the step and chip, then capture and price distinct regions.

        The step is clocked before the peaks, not after. measure_machine is
        the heaviest burst of GPU work in the job and pays it back as the
        longest pacing idle in the job; a step clocked right after that idle
        read 8-16 ms for a model that runs at 5.7, because idling slows the
        CPU side and this step spends ~40% of its time building ops in Python.
        Clocking first costs nothing: the room line needs both numbers and is
        computed after either way."""
        # the recorder must be fully removed before any timing
        self.tracer.uninstall()
        mx.clear_cache()  # capture's activation buffers; warm-up refills what clocks need
        self._memory_fit()
        self._clock_steps()
        self._settle_delivery(regions)
        self.measure_machine()
        self._record_step_floor()
        self.pending_regions = list(regions)
        self.report.coverage["selection"] = {"candidates": len(regions), "waves": []}
        return self._next_regions()

    def _next_regions(self, blockers: list[Region] = ()) -> list[Region]:
        """Price one set of alternatives. Overlapping cuts keep their turn."""
        while (self.pending_regions and self.total_hypotheses < self.manifest.budget_total
               and not self._finish_requested()):
            taken = Taken(blockers)
            eligible = [r for r in self.pending_regions
                        if len(free_members(r, taken)) == r.copies]
            estimates = estimate_regions(eligible, self.traces, self.peaks, self.step_ms)
            unpriced = self._unpriceable(eligible, estimates)
            if unpriced:
                dropped = {id(r) for r in unpriced}
                eligible = [r for r in eligible if id(r) not in dropped]
                self.pending_regions = [r for r in self.pending_regions if id(r) not in dropped]
            ready, deferred = select_frontier(eligible, estimates)
            if not ready:
                return []
            chosen = {r.fingerprint for r in ready}
            self.pending_regions = [r for r in self.pending_regions if r.fingerprint not in chosen]
            self.selection_wave += 1
            row = {"wave": self.selection_wave, "selected": len(ready),
                   "deferred": len(self.pending_regions), "conflicts": deferred}
            self.report.coverage["selection"]["waves"].append(row)
            self.log.append("selection", **row)
            print(f"selection {self.selection_wave}: measuring {len(ready)} regions; "
                  f"{len(self.pending_regions)} overlapping alternatives queued", flush=True)
            self.report.write(self.work_dir / "report.json")

            def wave():
                self.tracer.install()
                try:
                    self._capture(ready)
                finally:
                    self.tracer.uninstall()
                return self._price_and_rank(ready)

            ranked = self._until_finish(wave, "selection wave")
            if ranked is None:
                return []
            self.report.write(self.work_dir / "report.json")
            # Already-priced candidates should get their search turn after
            # one alternative wave, even if that wave falls below the floor.
            # Without this boundary a long chain of rejected alternatives
            # sends us back through exhaustive pricing while useful work waits.
            if ranked or blockers:
                return ranked
        return []

    def _until_finish(self, work, where: str):
        """Run capture and pricing so a finish request ends them at their next
        pacing idle; None when it did. On Qwen3.8-27B a finish asked for 2
        minutes into a pricing wave waited 45 for it."""
        self.session.stop_when = self._finish_requested
        try:
            return work()
        except Stopped:
            self.log.append("selection_abandoned", wave=self.selection_wave, where=where,
                            reason="operator requested final validation")
            return None
        finally:
            self.session.stop_when = None

    def _unpriceable(self, regions: list[Region], estimates) -> list[Region]:
        """Regions whose estimated share of the step is under a quarter of the
        floor, rejected without a price. On Qwen3.8-27B pricing measured 178
        regions for 2.6 hours to keep 9; ranked by this estimate those 9 were
        the top 9, each estimated within 0.74-1.15x of its measured share, and
        this cut would have priced 31. A region whose sizes the step clock did
        not time has no estimate and is priced as before."""
        out = []
        for r in regions:
            if not any(m.workload in self.step_ms for m in r.members):
                continue
            share = estimates[r.fingerprint].combined_p
            if share < UNPRICED_FLOOR_FRACTION * REGION_FLOOR_P:
                r.rejected = (f"estimated share {share:.4f} is under {UNPRICED_FLOOR_FRACTION:g} of the "
                              f"{REGION_FLOOR_P} floor; not priced")
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
                out.append(r)
        if out:
            self.log.append("unpriced", regions=len(out), under=UNPRICED_FLOOR_FRACTION * REGION_FLOOR_P)
        return out

    def _refresh_after_ship(self, ranked: list[Region], shipped: list[Region]) -> list[Region]:
        """Remove occupied copies and refresh any remaining prices safely."""
        recapture = []
        taken = Taken(shipped)
        for pool in (ranked, self.pending_regions):
            for candidate in list(pool):
                remaining = free_members(candidate, taken)
                if not remaining:
                    pool.remove(candidate)
                    self.log.append("region_covered", fingerprint=candidate.fingerprint,
                                    reason="every copy touches a shipped cut")
                elif len(remaining) != candidate.copies:
                    candidate.members = remaining
                    if pool is ranked:
                        # The old representative may have been removed. The
                        # same workload label can now name different array IDs
                        # or shapes; old boundary files cannot price that copy.
                        recapture.append(candidate)
        if recapture:
            self.tracer.install()
            try:
                self._capture(recapture)
            finally:
                self.tracer.uninstall()
        # These clocks predate the installed change. Pending regions are
        # captured and priced by _next_regions when their turn arrives.
        return self._price_and_rank(ranked) if ranked else []

    def _capture(self, regions: list[Region]) -> None:
        """k input sets per workload, the boundary arrays of every viable
        region saved from a recorded pass at each."""
        self.session.wait_ready()
        for w in self.manifest.workloads:
            wanted: set[int] = set()
            for r in regions:
                for _label, m, _copies in capture_instances(r, self.traces):
                    if m.workload == w.name:
                        wanted |= set(m.input_ids) | set(m.output_ids)
            if not wanted:
                continue
            seeds = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)
            for si, seed in enumerate(seeds):
                tensors = self._materialize(w, self.manifest.primary, seed)
                arrays = self.session.off_clock(lambda: capture_boundaries(
                    self.tracer, self._for(self.baseline_model, w.name), tensors, self.traces[w.name], wanted),
                    where=f"capture {w.name} set {si}")
                for r in regions:
                    for label, m, _copies in capture_instances(r, self.traces):
                        if m.workload != w.name:
                            continue
                        try:
                            ins = {a: arrays[a] for a in m.input_ids}
                            outs = {a: arrays[a] for a in m.output_ids}
                        except KeyError as e:
                            # a capture gap costs this region, never the job
                            r.rejected = f"capture missed boundary array {e}"
                            break
                        self.store.save(r.fingerprint, label, si, "inputs", ins, self.traces[w.name].weight_paths)
                        self.store.save(r.fingerprint, label, si, "outputs", outs)
                # saved; held into the next pass they add a set to its peak
                arrays = ins = outs = None
            self._capture_sweep(w, [r for r in regions if not r.rejected])
        self._validate_boundaries(regions)

    def _validate_boundaries(self, regions: list[Region]) -> None:
        """Verify each wave's contracts and saved sets before pricing or search."""
        from .regions.fingerprint import boundary_roles
        from .regions.store import BoundaryMismatch
        for region in regions:
            if region.rejected:
                continue
            rep = region.members[0]
            expected = boundary_roles(self.traces[rep.workload], rep)
            for member in region.members:
                if boundary_roles(self.traces[member.workload], member) != expected:
                    raise BoundaryMismatch(f"Region {region.fingerprint} groups incompatible "
                                           f"input/output roles at {member.workload}:{member.start_seq}")
            instances = [(label, member, manifest_mod.BOUNDARY_INPUT_SETS)
                         for label, member, _ in capture_instances(region, self.traces)]
            for label, span, _ in self._sweep_instances(region):
                if boundary_roles(self.sweep_traces[label], span) != expected:
                    raise BoundaryMismatch(f"Region {region.fingerprint} has incompatible "
                                           f"input/output roles at sweep {label}")
                instances.append((label, span, 1))
            for label, member, count in instances:
                for si in range(count):
                    self.store.validate(region.fingerprint, label, si, "inputs", member.input_ids)
                    self.store.validate(region.fingerprint, label, si, "outputs", member.output_ids)

    def _capture_sweep(self, w, regions: list[Region]) -> None:
        """One correctness set per sweep size for every region the workload
        fires in: the region's span located in the trace at that size, its
        boundary arrays saved from one recorded pass. A region whose ops the
        model no longer runs in one run at some size strands here."""
        for label, _dims in self._sweep_points(w):
            retrace = self.sweep_traces[label]
            wanted: set[int] = set()
            for r in regions:
                rep = next((m for m in r.members if m.workload == w.name), None)
                if rep is None:
                    continue
                try:
                    span = locate_span(self.traces[w.name], rep, retrace, w.name)
                except SweepDivergence as e:
                    r.rejected = f"the op stream diverges at {label}: {e}"
                    continue
                self.sweep_spans[(r.fingerprint, label)] = span
                wanted |= set(span.input_ids) | set(span.output_ids)
            if not wanted:
                continue
            arrays = self.session.off_clock(lambda: capture_boundaries(
                self.tracer, self._for(self.baseline_model, label), self.sweep_tensors[label], retrace, wanted),
                where=f"capture {label}")
            for r in regions:
                span = self.sweep_spans.get((r.fingerprint, label))
                if span is None:
                    continue
                try:
                    ins = {a: arrays[a] for a in span.input_ids}
                    outs = {a: arrays[a] for a in span.output_ids}
                except KeyError as e:
                    r.rejected = f"capture missed boundary array {e} at {label}"
                    continue
                self.store.save(r.fingerprint, label, 0, "inputs", ins, retrace.weight_paths)
                self.store.save(r.fingerprint, label, 0, "outputs", outs)
            arrays = ins = outs = None  # saved; not held into the next pass

    def _workload_of(self, tensors) -> str | None:
        """Which workload these inputs are: the job's own input sets, by identity."""
        for table in (getattr(self, "tensors", {}), getattr(self, "sweep_tensors", {}),
                      getattr(self, "serve_tensors", {})):
            for name, t in table.items():
                if t is tensors:
                    return name
        return None

    def _for(self, model, workload):
        """The step a workload calls on this model (a name or one of the job's
        input sets): the model itself, except in a job that has a conversation
        workload beside workloads without one, where each of those calls a
        PlainStep on the same model (state.step_for)."""
        from backcountry_runtime.state import step_for
        context = getattr(self, "context", None)
        if context is None or all(w.name.split("@")[0] == context.name.split("@")[0]
                                  for w in self.manifest.workloads):
            return model
        name = workload if isinstance(workload, str) else self._workload_of(workload)
        if name is None:
            raise RuntimeError("these inputs belong to no workload of the job, so which step they call is unknown")
        return step_for(model, name.split("@")[0] == context.name.split("@")[0])

    def _step_fn(self, model, tensors: list[mx.array], baseline: str | None = None, workload: str | None = None):
        """One step of the model as the baseline runs it, chained into one
        timed sample when the step is small (_chain). Compiled means a fresh
        mx.compile closure, built here and never reused across a swap:
        compile caches on the callable, so an old closure keeps the old graph.
        workload: whose step, for inputs that are not one of the job's own sets."""
        model = self._for(model, workload or tensors)
        if self.use_library_inference:
            return lambda: model(*tensors)
        call = lambda *t: model(*t)
        if (baseline or self.baseline) == "compiled":
            call = mx.compile(call)
            self.session.timed(lambda: call(*tensors))  # compile now; account for cooling debt
        return chained_steps(call, tensors, self._chain(tensors))

    def _chain(self, tensors: list[mx.array]) -> int:
        """How many dependent steps one timed sample of this workload holds:
        one for a step of CLOCK_TARGET_MS or more, enough to fill that for a
        small model, whose lone call leaves the GPU idle between samples and
        never brings its clock up (see tools/ramp_after_idle.py). Sized once per
        workload on the arm the baseline runs; a step over the model's own
        cache or a library generation is never chained."""
        name = next((n for n, t in self.tensors.items() if t is tensors), None)
        if name is None or self.use_library_inference or (
                getattr(self, "context", None) is not None
                and self._for(self.baseline_model, name) is self.baseline_model):
            return 1
        if name not in self.step_chain:
            step = self._for(self.baseline_model, name)
            call = lambda *t: step(*t)
            if self.baseline == "compiled":
                call = mx.compile(call)
            self.step_chain[name] = loop_iterations(
                lambda fn: self.session.timed(fn) * fn.steps,
                lambda n: chained_steps(call, tensors, n))
            self.log.append("step_chain", workload=name, steps=self.step_chain[name])
        return self.step_chain[name]

    def _memory_fit(self) -> None:
        """Which workloads fit this Mac's GPU with the job around them: each
        runs once on the live model and its MLX peak is kept (in a resumed
        job, the first process's). One past the GPU's working-set limit less
        FIT_HEADROOM_GB is still checked for correctness everywhere, but no
        whole-model clock times it, and over a range neither is any longer
        size of its workload. MLX's cache is then capped at what the largest
        timed workload leaves under the same line."""
        fit = self.report.constants.get("memory_fit")
        if fit is None:
            if not self.use_library_inference:
                self._choose_baseline()
            limit = mx.device_info()["max_recommended_working_set_size"] / 1e9
            peaks = {}
            for w in self.manifest.workloads:
                step = self._step_fn(self.model, self.tensors[w.name])
                self.session.off_clock(lambda: mx.eval(step()), where=f"memory fit {w.name}")
                peaks[w.name] = round(mx.get_peak_memory() / 1e9, 3)
            fit = {"gpu_limit_gb": round(limit, 3), "headroom_gb": FIT_HEADROOM_GB, "peaks_gb": peaks,
                   "untimed": sorted(n for n, p in peaks.items() if p > limit - FIT_HEADROOM_GB)}
            self.report.constants["memory_fit"] = fit
            self.log.append("memory_fit", **fit)
        self._untimed = set(fit["untimed"])
        # over a range every sized choice keeps the primary; without one, any workload will do
        primary = next((name for _n, name in self._sized_workloads() if "@" not in name), None)
        stuck = [primary] if primary in self._untimed else \
            [] if len(self._untimed) < len(fit["peaks_gb"]) else sorted(self._untimed)
        if stuck:
            raise RuntimeError(
                f"{', '.join(stuck)} peaks at {max(fit['peaks_gb'][n] for n in stuck):.1f} GB of GPU memory, within "
                f"{fit['headroom_gb']:g} GB of this Mac's {fit['gpu_limit_gb']:.1f} GB limit: the job cannot time "
                "the model's primary size here; choose a smaller primary or a Mac with more memory")
        if self._untimed:
            print("memory: not timed (too close to the GPU's limit): " + ", ".join(sorted(self._untimed)), flush=True)
        timed = [p for n, p in fit["peaks_gb"].items() if _fits(self, n)]
        cap = max(0.0, fit["gpu_limit_gb"] - fit["headroom_gb"] - max(timed, default=0.0))
        mx.set_cache_limit(int(cap * 1e9))
        fit["cache_limit_gb"] = round(cap, 3)

    def _timing_memory(self, name: str) -> None:
        """After a whole-model comparison: one whose own peak came within
        FIT_HEADROOM_GB of the GPU's limit leaves that workload (and over a
        range every longer size of it) untimed for the rest of the job. A
        comparison holds more than one run of the model: Qwen3.8-27B read 1,118
        tokens at a 17.46 GB peak alone, its comparisons there reached 19.02 GB
        and the next one ran out of GPU memory."""
        report = getattr(self, "report", None)  # a runner without a memory check has nothing to untime
        fit = report.constants.get("memory_fit") if report is not None else None
        peak = getattr(getattr(self, "session", None), "last_compare_peak_gb", None)
        if fit is None or peak is None or not _fits(self, name) \
                or peak <= fit["gpu_limit_gb"] - fit["headroom_gb"]:
            return
        self._untimed = set(getattr(self, "_untimed", ())) | {name}
        fit["untimed"] = sorted(self._untimed)
        fit.setdefault("untimed_by_comparison", {})[name] = peak
        self.log.append("memory_untimed", workload=name, comparison_peak_gb=peak,
                        line_gb=round(fit["gpu_limit_gb"] - fit["headroom_gb"], 3))
        print(f"memory: no longer timing {name} (a comparison there peaked at {peak:.2f} GB)", flush=True)

    def _baseline_arm(self):
        """What every whole-model number is measured against: the untouched
        model, or under a compiled baseline the same model with its baseline
        scopes compiled and empty. Correctness always uses the untouched one."""
        return self.baseline_model if self._compiled_baseline is None else self._compiled_baseline

    def _kernels_installed(self) -> bool:
        """Whether any installed wrapper carries a cut; a compiled baseline's
        empty wrappers alone are not an improvement to check or ship."""
        return any(not isinstance(state, tuple) or any(state[1].values()) for state in self.installed.values())

    def _pacing(self) -> dict:
        """How this job rests between measurements, for the report."""
        heat = getattr(self.session, "heat", None)
        if heat is None:
            return {"policy": f"rest {self._rest_factor():g}x work",
                    "reason": "no heat readings on this machine, or a session given without a heat watcher"}
        return heat.record()

    def _rest_factor(self) -> float:
        """The rest after GPU work, as a multiple of it, once one is due."""
        return getattr(self.session, "_rest_factor", getattr(self.session, "duty_idle_factor", 0.0))

    def _heat_tripped(self, record: dict) -> None:
        self.log.append("heat_tripped", **record)
        if getattr(self.report, "constants", None):
            self.report.constants["pacing"] = record
            self.report.write(self.work_dir / "report.json")

    def _timed_arms(self, incumbent=None, names=None):
        """Fresh timing closures for every declared workload, or those named."""
        incumbent = self._baseline_arm() if incumbent is None else incumbent
        return {w.name: (self._step_fn(incumbent, self.tensors[w.name]),
                         self._step_fn(self.model, self.tensors[w.name]))
                for w in self.manifest.workloads if (names is None or w.name in names) and _fits(self, w.name)}

    def _sized_workloads(self) -> list[tuple[int, str]]:
        """Over a range of sizes: (size, workload) for the primary and every
        search size, smallest first; empty without a range."""
        named = self.report.constants.get("search_sizes") or {}
        if not named:
            return []
        dim = next(iter(named))

        def size(w):
            if dim in w.named_dims():
                return self.manifest.primary[dim]
            tail = w.name.rsplit("=", 1)[-1] if f"@{dim}=" in w.name else ""
            return int(tail) if tail.isdigit() else None

        return sorted((size(w), w.name) for w in self.manifest.workloads if size(w) is not None)

    def _spread_sizes(self, count: int) -> set[str] | None:
        """Over a range: count of the sized workloads, spread evenly from the
        shortest to the longest, the primary in place of the nearest inner
        one; None without a range or when there are no more than count."""
        names = [name for _n, name in self._sized_workloads() if _fits(self, name)]
        if len(names) <= count:
            return None
        picks = _spread(names, count)
        primary = next(name for name in names if "@" not in name)
        if primary not in picks:
            at = names.index(primary)
            inner = min(range(1, count - 1), key=lambda j: abs(names.index(picks[j]) - at))
            picks[inner] = primary
        return set(picks)

    def _priced_workloads(self) -> set[str]:
        """The workloads regions are priced at: every one, or over a range the
        shortest, the primary and the longest size. A price ranks regions and
        sets the judge's target; ten sizes cost 22 of a 0.6B setup's 40 min.
        Once regions are priced the set is the one they were priced at, less
        what does not fit (a job priced before _memory_fit measured it)."""
        if (priced := self.report.constants.get("priced_workloads")) is not None:
            return {name for name in priced if _fits(self, name)}
        every = {w.name for w in self.manifest.workloads if _fits(self, w.name)}
        sized = [(n, name) for n, name in self._sized_workloads() if _fits(self, name)]
        if len(sized) <= MODEL_CHECK_SIZES:
            return every
        primary = next(name for n, name in sized if "@" not in name)
        return every - {name for _n, name in sized} | {sized[0][1], primary, sized[-1][1]}

    def _model_check_sizes(self, region: Region, detail: dict) -> list[str] | None:
        """Over a range of sizes: the workloads the whole-model check times,
        the shortest and longest sizes the region runs at and the one its
        region clock was weakest at (lowest library / kernel ratio). None
        times every workload, as without a range."""
        rows = (detail or {}).get("sizes") or {}
        # sizes the kernel leaves to the library (the size gate's stretch rule)
        # would time the library against itself
        sized = [(n, name) for n, name in self._sized_workloads()
                 if name in region.workloads and not rows.get(name, {}).get("declined") and _fits(self, name)]
        if len(sized) <= MODEL_CHECK_SIZES:
            return None
        chosen = [sized[0][1], sized[-1][1]]
        ratios = sorted((row["library_ms"] / row["kernel_ms"], label) for label, row in rows.items()
                        if label not in chosen and label in dict((n, s) for s, n in sized) and row["kernel_ms"] > 0
                        and not row.get("declined"))
        middle = ratios[0][1] if ratios else sized[len(sized) // 2][1]
        return [chosen[0], middle, chosen[1]]

    def _copy_incumbent(self, emitted=None, overrides=None, kernels=None):
        """A fresh model carrying the given installed wrappers (with kernels,
        scope -> kernel specs, default the installed ones), with overrides
        (scope -> (wrapper, kernel specs)) in place of whatever those scopes
        and their descendants would have carried."""
        emitted = self.emitted if emitted is None else emitted
        kernels = kernels or {path: state[2] for path, state in self.installed.items()}
        overrides = overrides or {}
        if not emitted and not overrides:
            return self.baseline_model
        model = self._fresh_model()
        require_independent_models(self.model, model)
        require_independent_models(self.baseline_model, model)
        share_weights(self.baseline_model, model)
        for path, wrapper in emitted.items():
            if any(path == root or path.startswith(root + ".") for root in overrides):
                continue
            swap_install(model, path, _load_class(wrapper)(_resolve(model, path), kernels[path]))
        for path, (wrapper, specs) in overrides.items():
            swap_install(model, path, _load_class(wrapper)(_resolve(model, path), specs))
        mx.clear_cache()
        return model

    def _identity(self, path: str, name: str):
        """The scope compiled with no cut, over every recorded call of it;
        NotReplayable names why the scope cannot be compiled."""
        from .bind.graph import emit_graph_wrapper_variants
        definitions = [(trace, sc, []) for trace in self.traces.values()
                       for sc in trace.scope_calls if sc.address.rsplit("@", 1)[0] == path]
        return emit_graph_wrapper_variants(definitions, name)

    def _settle_delivery(self, regions: list[Region]) -> None:
        """Before any clock: certify the identity of every scope planned for
        graph delivery, so the plan every clock follows is the delivery that
        will ship (a scope whose compiled identity is not bitwise the original
        moves its regions to replay), then clock the untouched model against
        itself with the outermost certified scopes compiled and empty. That
        is the compiled clock: a resolved slowdown sends those scopes to
        replay too; otherwise, when the manifest asks for the compiled
        baseline under library inference, that model becomes the baseline,
        installed empty so every kernel composes into it."""
        if self.baseline == "compiled":
            return
        planned: dict[str, list] = {}
        for region in regions:
            for member in region.members:
                if region.rejected or region.delivery.get(member.workload) != "graph":
                    continue
                scope = find_scope_call(self.traces[member.workload], member.scope_stack)
                planned.setdefault(scope.address.rsplit("@", 1)[0], []).append((region, member.workload))
        if self._generated():
            # its calls read generated tokens, so delivery replays (_emit_installation): nothing is
            # certified or clocked compiled (on Qwen3.8-27B that certification ran out of GPU memory)
            for path in sorted(planned):
                self._fall_back(planned[path], path, "the job's calls read generated tokens: a compiled "
                                "graph would not run them under exact rows")
            self.log.append("delivery_settled", graph=[], replay=sorted(self.replay_scopes),
                            baseline=self.baseline, baseline_scopes=[])
            return
        identity = {}
        for path in sorted(planned):
            try:
                identity[path] = self._identity(path, f"Id_{_safe(path)}")
            except NotReplayable as error:
                self._fall_back(planned[path], path, str(error))
        from backcountry_runtime.state import correctness_call
        runs = [lambda t=self.tensors[w.name], m=self._for(self.baseline_model, w.name): correctness_call(m, t)
                for w in self.manifest.workloads]
        while identity:
            wrappers = {path: _load_class(e)(_resolve(self.baseline_model, path), {})
                        for path, e in identity.items()}
            # nothing in certification is timed: it runs unpaced
            check = self.session.off_clock(lambda: certify_identities(
                model=self.baseline_model, wrappers=wrappers, runs=runs), paced=False)
            if check.ok:
                break
            failed = check.scope if check.scope in identity else next(iter(identity))
            identity.pop(failed)
            self._fall_back(planned[failed], failed, check.reason)
        # Two compiled scopes cannot nest over one captured state, so the
        # outermost certified scopes stand in for all beneath them.
        outermost = {p: (e, {}) for p, e in identity.items()
                     if not any(q != p and p.startswith(q + ".") for q in identity)}
        slower = False
        if outermost:
            self.session.wait_ready()
            compiled = self._copy_incumbent({}, outermost)
            # a compiled baseline adopted below takes its step times from this
            # clock, so it times every workload
            adopting = self.use_library_inference and self.manifest.baseline == "compiled"
            spread = None if adopting else self._spread_sizes(COMPILED_CLOCK_SIZES)
            for w in self.manifest.workloads:
                if spread is not None and w.name not in spread or not _fits(self, w.name):
                    continue
                comparison = compare(self.session, self._step_fn(self.baseline_model, self.tensors[w.name]),
                                     self._step_fn(compiled, self.tensors[w.name]),
                                     pairs=SHIP_PAIRS if spread is None else COMPILED_CLOCK_PAIRS,
                                     defer_cooling=True)
                slower |= comparison.loses_by(0.0)
                clocks = self.report.baseline.setdefault("clocks_ms", {}).setdefault(
                    w.name, {"plain": self.step_ms.get(w.name)})
                clocks["compiled"] = statistics.median(comparison.candidate_ms)
                clocks["compiled_vs_plain"] = {"ratio": comparison.median_ratio, "stability": comparison.stability,
                                               "verdict": ("resolved_improvement" if comparison.wins_by(0.0) else
                                                           "resolved_regression" if comparison.loses_by(0.0)
                                                           else "inconclusive")}
                fallbacks = [p for p in outermost if (wrapper := _resolve(compiled, p))
                             and (wrapper._graph_fallbacks or wrapper._graph_fallback_reason)]
                if fallbacks:
                    raise RuntimeError(f"the compiled clock ran the original module at {fallbacks}")
        if slower:
            # Compiling these scopes costs the step more than fusion saves
            # (each compiled call has a host price): replay delivers here.
            for path in list(identity):
                self._fall_back(planned[path], path, "compiling the scope makes the step slower")
            identity.clear()
            outermost.clear()
        self.certified_scopes.update(identity)
        adopt = bool(outermost) and self.use_library_inference and self.manifest.baseline == "compiled"
        self.report.baseline.update(compiled_available=bool(outermost), choice="compiled" if adopt else "plain",
                                    reason=None if adopt or not outermost else self.report.baseline.get("reason"))
        if adopt:
            # The baseline every win is measured against is this model:
            # compiled where it can be, empty. Installed as the starting
            # state, every kernel composes into it, and the artifact ships it.
            self.baseline = "compiled"
            self._compiled_baseline = compiled
            self.baseline_wrappers = [emitted for emitted, _ in outermost.values()]
            self.report.constants["measurement"]["baseline"] = "library inference with compiled model scopes"
            for w in self.manifest.workloads:
                self.step_ms[w.name] = self.report.baseline["clocks_ms"][w.name]["compiled"]
                self.report.step_ms[w.name] = {"before": self.step_ms[w.name]}
            for path, (emitted, _specs) in outermost.items():
                original = _resolve(self.model, path)
                swap_install(self.model, path, _load_class(emitted)(original, {}))
                variants = {(name, sc.address): [] for name, trace in self.traces.items()
                            for sc in trace.scope_calls if sc.address.rsplit("@", 1)[0] == path}
                self.installed[path] = (original, variants, {})
                self.emitted[path] = emitted
        self.log.append("delivery_settled", graph=sorted(identity), replay=sorted(self.replay_scopes),
                        baseline=self.baseline, baseline_scopes=sorted(outermost) if adopt else [])

    def _fall_back(self, regions, path: str, reason: str) -> None:
        """Graph delivery declined for a scope: its regions take replay there."""
        self.replay_scopes.add(path)
        self.log.append("graph_fallback", scope=path, reason=reason)
        for region, workload in regions:
            region.delivery[workload] = "replay"
            region.delivery_reasons[workload] = reason

    def _choose_baseline(self) -> None:
        """Plain for a step that keeps Python state, else as requested; once,
        before anything times or sizes a step: a step chain sized on a compiled
        step that is then timed plain is far too long (a whole reply replayed
        through a compiled call skips its Python loop and read 40x fast)."""
        if self.report.baseline.get("choice") is not None:
            self.baseline = self.report.baseline["choice"]
            return
        kept = {w: marks for w, t in self.traces.items() if (marks := _state_marks(t))}
        requested = self.manifest.baseline
        marked = self._replays_marked_function()
        self.baseline = "plain" if kept or marked else requested
        reason = ("a marked function runs as the app calls it, with no compile around the whole call (what it "
                  "compiles inside stays compiled)") if marked else None if not kept else (
            "the step keeps Python state (" +
            "; ".join(f"{w}: {marks}" for w, marks in kept.items()) +
            "); a compiled call would leave the model holding tracers")
        self.report.baseline = {"requested": requested, "choice": self.baseline,
                                "compiled_available": not kept and not marked, "reason": reason, "clocks_ms": {}}
        self.log.append("baseline", requested=requested, choice=self.baseline,
                        compiled_available=not kept and not marked, reason=reason)

    def _clock_steps(self) -> None:
        """The step clocks on every workload, and the baseline every share and
        win is measured against. A step that keeps Python state (a KV cache)
        cannot be compiled from outside the model: mx.compile swaps only state
        handed to it in a dict or list, and one compiled call would leave the
        model holding tracers. Such a step gets the plain baseline, and its
        compiled clock is never taken. Where the compiled clock is taken, the
        model is checked alive and unchanged afterwards, because a compiled
        call on a state-keeping step returns a plausible number first and
        breaks the model only on the next call."""
        if self.use_library_inference:
            # The library runs the plain model. The compiled baseline, when
            # requested, is the same model with its outermost compilable
            # scopes compiled and empty, settled with delivery below.
            self.baseline = "plain"
            self.report.baseline = {
                "requested": self.manifest.baseline, "choice": "plain",
                "execution": "library_generation", "compiled_available": False,
                "reason": "the compiled baseline is settled once delivery is",
                "clocks_ms": {},
            }
            self.log.append("baseline", **self.report.baseline)
            for w in self._clocked_workloads():
                clock = step_clock(self.session, self._step_fn(self.baseline_model, self.tensors[w.name]))
                self.step_ms[w.name] = clock.median_ms
                self.report.step_ms[w.name] = {"before": clock.median_ms, "steps_per_sample": 1,
                                               "min_win_ms": self._min_win_ms(w.name)}
                self.report.baseline["clocks_ms"][w.name] = {"plain": clock.median_ms, "compiled": None}
                self.log.append("step_clock", workload=w.name, phase="before", baseline="library_generation",
                                median_ms=clock.median_ms, generated_tokens=self.manifest.final_benchmark.steps)
            return
        self._choose_baseline()
        for w in self._clocked_workloads():
            print(f"baseline: timing {w.name} ({self.baseline})", flush=True)
            tensors = self.tensors[w.name]
            clocks = {"plain": step_clock(self.session,
                                          self._step_fn(self.model, tensors, "plain")).median_ms,
                      "compiled": None}
            if self.baseline == "compiled":
                # mx.array copies: the step may return a buffer it writes
                before = [mx.array(a) for a in flatten_arrays(self._for(self.model, tensors)(*tensors))]
                mx.eval(before)
                clocks["compiled"] = step_clock(
                    self.session, self._step_fn(self.model, tensors, "compiled")).median_ms
                self._assert_survived_compile(w.name, tensors, before)
            self.report.baseline["clocks_ms"][w.name] = clocks
            self.step_ms[w.name] = clocks[self.baseline]
            self.report.step_ms[w.name] = {"before": clocks[self.baseline],
                                           "steps_per_sample": self.step_chain.get(w.name, 1),
                                           "min_win_ms": self._min_win_ms(w.name)}
            self.log.append("step_clock", workload=w.name, phase="before", baseline=self.baseline,
                            median_ms=clocks[self.baseline], plain_ms=clocks["plain"],
                            compiled_ms=clocks["compiled"])
    def _clocked_workloads(self) -> list:
        """The workloads the job-start step clock times: the ones regions are
        priced at. Over a range the step clock at all ten sizes took an hour of
        Qwen3.8-27B's setup, and only the priced sizes' clocks are read."""
        priced = self._priced_workloads()
        return [w for w in self.manifest.workloads if w.name in priced]

    def _record_step_floor(self) -> None:
        """The scout line: how much of each step is physics no kernel can
        touch, and how much is room. Needs both the step clock and the chip's
        peaks, so it runs after both and constrains the order of neither."""
        if self.peaks is None:  # a job measures the chip; a test may not
            return
        for w in self._clocked_workloads():
            trace = self.traces[w.name]
            floor = step_floor(trace, self.peaks, self.step_ms[w.name])
            self._raise_peaks((n for n in trace.nodes if n.seq not in trace.dead),
                              self.step_ms[w.name], f"the {w.name} step", floor["bytes_mb"] * 1e6)
            floor = step_floor(trace, self.peaks, self.step_ms[w.name])
            self.report.coverage.setdefault("step_floor", {})[w.name] = floor
            self.log.append("step_floor", workload=w.name, **floor)
        if warning := little_room(self.report.coverage.get("step_floor", {})):
            self._env_warning(warning)

    def _assert_survived_compile(self, workload: str, tensors, before) -> None:
        """A compiled call on a step that keeps state the detector missed does
        not raise: it returns a plausible number and leaves the model holding
        tracers, so every later win would be measured against garbage. Take
        the same step again and demand the same bits."""
        advice = ("the step keeps state the trace did not show; rerun with baseline: plain "
                  "in the manifest and report the model to the maintainer")
        try:
            after = flatten_arrays(self._for(self.model, tensors)(*tensors))
            mx.eval(after)
        except Exception as e:
            raise RuntimeError(f"the compiled baseline broke the model on {workload}: {e}; {advice}") from e
        repeats = getattr(getattr(self, "report", None), "constants", {}).get("output_repeats", True)
        if repeats is False and not self._requires_exact():  # repeats only to rounding: within the tolerance
            same = self._outputs_check(f"{workload}:compiled", lambda: before, lambda: after, exact=False).passed
        else:
            same = len(after) == len(before) and all(bitwise_equal(a, b) for a, b in zip(after, before))
        if not same:
            raise RuntimeError(f"the compiled baseline changed what {workload} returns; {advice}")

    def _price_and_rank(self, regions: list[Region]) -> list[Region]:
        """Each region's share of the step, its physical limit, the share
        floor, and the ranking."""
        self.session.wait_ready()
        priced = self._priced_workloads()
        self.report.constants["priced_workloads"] = sorted(priced)
        step_fns = {w.name: self._step_fn(self.model, self.tensors[w.name])
                    for w in self.manifest.workloads if w.name in priced}
        survivors = [r for r in regions if not r.rejected]
        self.log.append("pricing", regions=len(survivors), wave=self.selection_wave)
        print(f"pricing: {len(survivors)} regions sharing paired model measurements", flush=True)
        price_group(survivors, self.session, self.traces, self.store, step_fns,
                    baseline=self.baseline, pairs=PRICE_PAIRS)
        for r in survivors:
            self.log.append("priced", fingerprint=r.fingerprint, p=r.p, region_ms=r.t_rep_ms,
                            price_stability=dict(r.stability))
        for r in survivors:
            if r.rejected:
                continue
            rep = r.members[0]
            # one copy's cost against one copy's floor, both from one paired
            # window, so nothing the machine did between pricing and the peak
            # readings can open headroom that is not there
            by_workload = {}
            for label, member, copies in capture_instances(r, self.traces):
                if label not in r.prices:
                    continue  # a size the regions are not priced at
                price = r.prices[label]
                trace = self.traces[member.workload]
                if self.peaks is not None:  # a job measures the chip; a test may not
                    self._raise_peaks(trace.nodes[member.start_seq:member.end_seq + 1], price.ms,
                                      f"region {r.fingerprint[:8]} at {label}")
                roof = stretch_roofline(trace, member, self.peaks, t_orig_ms=price.ms,
                                        floor_ms=price.floor_ms, rates=price.gflops)
                by_workload.setdefault(member.workload, []).append((roof, copies))
                if member is rep:
                    r.roofline = roof
            for workload, roofs in by_workload.items():
                total_floor = sum(roof.t_roofline_ms * copies for roof, copies in roofs)
                r.rooflines[workload] = replace(
                    roofs[0][0], s_max=r.t_orig_ms[workload] / max(total_floor, 1e-9))
        kept = rank(apply_floor(survivors))
        latest = {row["fingerprint"]: row for row in self.report.pricing}
        for r in survivors:
            latest[r.fingerprint] = {
                "fingerprint": r.fingerprint, "ops": list(r.ops), "copies": r.copies,
                "p": dict(r.p), "region_ms": dict(r.t_rep_ms),
                "ranking_basis": "estimated_removable_share",
                "length_weight": length_weight(r),
                "price_stability": dict(r.stability),
                "estimate_basis": "boundary_probe_and_known_compute",
                "compute_model_complete": "metal_kernel" not in r.ops,
                "headroom_is_estimate": True,
                "estimated_s_max": {w: roof.s_max for w, roof in r.rooflines.items()},
                "measured_compute_gflops": {label: price.gflops for label, price in r.prices.items()},
                "potential_model_latency_reduction_pct": {
                    w: 100 * share for w, share in r.removable_p.items()},
                "status": "rejected" if r.rejected else "eligible",
                "reason": r.rejected,
            }
        self.report.pricing = sorted(latest.values(),
                                     key=lambda row: -sum(row["p"].values()))
        # every compute op belongs to exactly one atomic region, so their
        # shares sum to the fraction of the step that any candidate can reach;
        # the rest runs inside stranded scopes or ops no kernel can replace
        self.report.coverage.update({
            "measured_atomic_share_this_wave": {
                w.name: sum(r.p.get(w.name, 0.0) for r in regions if self._atomic(r))
                for w in self.manifest.workloads
            },
            "step_ms": dict(self.step_ms),
        })
        for r in regions:
            if r.rejected:  # share-floor cuts belong in the report
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
        self.log.append("ranked", kept=len(kept))
        self.report.write(self.work_dir / "report.json")
        return kept

    def _atomic(self, region: Region) -> bool:
        """One compute op, with any views in front of it taken of weights: a
        projection through its transpose, or a lone op. Each compute op is in
        exactly one such region."""
        rep = region.members[0]
        trace = self.traces[rep.workload]
        nodes = trace.nodes[rep.start_seq:rep.end_seq + 1]
        if sum(not is_view(n) for n in nodes) != 1:
            return False
        weight_like = weight_like_ids(trace)
        return all(all(a in weight_like for a in n.in_arrays) for n in nodes if is_view(n))

    def measure_machine(self) -> None:
        """Record chip throughput and require the timing control to pass.

        Utilization and peak estimates are informational. A significant
        difference between identical arms means the clock cannot currently
        distinguish optimization from bias, so it gets one retry, then stops.
        """
        print("calibration: measuring bandwidth, compute throughput and timing noise", flush=True)
        busy = self.gpu_busy_at_start
        self.report.session["gpu_utilization_at_start_pct"] = busy
        if busy is not None and busy > BUSY_GPU_PERCENT:
            self._env_warning(f"macOS reports {busy:.0f}% GPU utilization before the job; "
                              "this counter does not measure available throughput. "
                              "Measured throughput and paired timings determine the results")
        self._record_peaks(measure_peaks(self.session), "job_start")
        reason = peaks_implausible(self.peaks)
        if reason:
            self._env_warning(f"{reason}; the chip is throttled or shared, so small wins may go "
                              "unnoticed and the room line will read high")
        for attempt in (1, 2):
            floor = aa_null(self.session, pairs=8)
            passed = not (floor.wins_by(0.0, LOSS_SIGMAS) or floor.loses_by(0.0))  # any difference, either way
            self.report.session["aa_floor_sigma_ms"] = floor.sigma_ms
            self.report.session["aa_control"] = {"passed": passed, "attempts": attempt}
            self.log.append("aa_floor", attempt=attempt, passed=passed,
                            sigma_ms=floor.sigma_ms, median_delta_ms=floor.median_delta_ms,
                            stability=round(floor.stability, 3))
            if passed:
                break
            if attempt == 1:
                self._env_warning(f"the A/A control found a {floor.median_delta_ms:+.4f} ms "
                                  "difference between identical code; this can create false wins. "
                                  "Retrying the timing control once")
            else:
                raise RuntimeError("A/A timing control failed twice: identical code showed a "
                                   "significant timing difference. Measurements are not trustworthy; "
                                   "the run stopped before searching kernels")

    def _raise_peaks(self, nodes, measured_ms: float, source: str, bytes_moved: float = 0.0) -> None:
        """The chip seen doing these ops, or moving these bytes, faster than
        the peaks allow raises them: a ceiling never sits under a measurement."""
        raised = observed_peaks(self.peaks, nodes, measured_ms, bytes_moved)
        if raised is not self.peaks:
            self._record_peaks(raised, f"raised by {source}")

    def _record_peaks(self, peaks, when: str) -> None:
        self.peaks = peaks
        self.report.peaks = {"bandwidth_gbps": peaks.bandwidth_gbps,
                             "flops_gflops": peaks.flops_gflops, "launch_us": peaks.launch_us}
        self.log.append("peaks", when=when, **self.report.peaks)

    def _env_warning(self, detail: str) -> None:
        self.log.append("env_warning", detail=detail)
        print(f"WARNING: {detail}")

    # -- stage 3: one region --------------------------------------------------

    def _contract(self, region: Region) -> RegionContract:
        rep = region.members[0]
        specs = self.traces[rep.workload].span_specs(rep.start_seq, rep.end_seq)
        scaffold = self._build_scaffold(region)
        return RegionContract(
            input_names=tuple(f"in{i}" for i in range(len(rep.input_ids))),
            input_ranks=tuple(len(specs[a][0]) for a in rep.input_ids),
            input_dtypes=tuple(specs[a][1] for a in rep.input_ids),
            output_names=tuple(f"out{i}" for i in range(len(rep.output_ids))),
            output_ranks=tuple(len(specs[a][0]) for a in rep.output_ids),
            output_dtypes=tuple(specs[a][1] for a in rep.output_ids),
            live_outputs=tuple(f"out{i}" for i in range(len(rep.output_ids))),
            input_shapes=tuple(tuple(specs[a][0]) for a in rep.input_ids),
            output_shapes=tuple(tuple(specs[a][0]) for a in rep.output_ids),
            native_call=scaffold.native_call,
            input_signature=scaffold.input_signature,
            input_signatures=scaffold.input_signatures,
        )

    def _eval_sets(self, region: Region) -> list[EvalSet]:
        sets = []
        for label, m, _copies in capture_instances(region, self.traces):
            k = self.store.set_count(region.fingerprint, label)
            if region.t_orig_ms.get(m.workload) is None and m.workload in self._priced_workloads():
                # open_region screens this; backstop so it can never reach
                # ladder validation as a bare None
                raise RuntimeError(f"region eval set unpriced for workload {m.workload!r}")
            sets.append(EvalSet(
                label=label,
                inputs_paths=[str(self.store._path(region.fingerprint, label, i, "inputs"))
                              for i in range(k)],
                reference_paths=[str(self.store._path(region.fingerprint, label, i, "outputs"))
                                 for i in range(k)],
                t_library_ms=region.prices[label].ms if label in region.prices
                    else region.t_rep_ms.get(m.workload),
                correctness_only=False,
                nodes_json=nodes_to_json(self.traces[m.workload].nodes[m.start_seq:m.end_seq + 1]),
            ))
        for label, span, _specs in self._sweep_instances(region):
            nodes = self.sweep_traces[label].nodes[span.start_seq:span.end_seq + 1]
            sets.append(EvalSet(
                label=label,
                inputs_paths=[str(self.store._path(region.fingerprint, label, 0, "inputs"))],
                reference_paths=[str(self.store._path(region.fingerprint, label, 0, "outputs"))],
                correctness_only=True,
                nodes_json=nodes_to_json(nodes),
            ))
        return sets

    def _sweep_instances(self, region: Region) -> list[tuple[str, Stretch, dict]]:
        """(label, span, array specs) for each sweep size the region was
        located and captured at."""
        out = []
        for (fingerprint, label), span in sorted(self.sweep_spans.items()):
            if fingerprint == region.fingerprint:
                specs = self.sweep_traces[label].span_specs(span.start_seq, span.end_seq)
                out.append((label, span, specs))
        return out

    def _target_workload(self, region: Region, requested: str | None = None) -> str:
        """Choose before timing, independently of manifest order. Only a
        workload the region was priced at has a library time to race: over a
        range, the shortest, the primary and the longest size."""
        if requested is not None:
            if requested not in region.t_rep_ms:
                raise ValueError(f"target_workload {requested!r} must be one of the region's timing "
                                 f"workloads {sorted(region.t_rep_ms)}")
            return requested
        return min(region.workloads, key=lambda w: (
            -region.removable_p.get(w, 0.0), -region.p.get(w, 0.0), w))

    def _min_win_ms(self, workload: str) -> float:
        """The least a region win may save per step: 1% of the workload's step,
        never more than MIN_WIN_MS. The three-sigma rule guards noise; this
        floor only keeps trivial wins from an install and a final check."""
        step = getattr(self, "step_ms", {}).get(workload)
        return min(MIN_WIN_MS, MIN_WIN_SHARE * step) if step else MIN_WIN_MS

    def _ladder_job(self, region: Region, kernel: KernelSpec, assoc_tag: str,
                    run_clock: bool, target_workload: str | None = None) -> LadderJob:
        target = self._target_workload(region, target_workload)
        if self._exact_only():
            # a call that stands in for one-token calls must keep their bits, and
            # a job whose manifest sets no tolerances allows no other: a kernel
            # that adds in its own order is checked bit for bit here too
            assoc_tag = "preserving"
        label, rep, copies = next(instance for instance in capture_instances(region, self.traces)
                                   if instance[1].workload == target)
        trace = self.traces[rep.workload]
        nodes = trace.nodes[rep.start_seq:rep.end_seq + 1]
        contract = self._contract(region)
        eval_sets = self._eval_sets(region)
        eval_sets.sort(key=lambda es: es.label != label)
        roof = region.rooflines.get(target, region.roofline)
        return LadderJob(
            baseline=region.library_arm(target, self.baseline),
            kernel=kernel,
            contract=contract,
            assoc_tag=assoc_tag,
            nodes_json=nodes_to_json(nodes),
            input_ids=rep.input_ids,
            output_ids=rep.output_ids,
            eval_sets=eval_sets,
            tolerances=self.manifest.tolerances,
            min_win_ms=self._min_win_ms(target) / max(copies, 1),
            run_clock=run_clock,
            clock_pairs=self.clock_pairs,
            # the timing child runs a few hundred passes under pacing at every
            # timed size; a region whose one pass takes hundreds of ms needs
            # minutes there, not a fixed cap (an unpriced size as the costliest
            # priced one: the longest size is always priced)
            timeout_s=120.0 + sum(30.0 + 0.8 * (es.t_library_ms or max(e.t_library_ms or 0.0 for e in eval_sets))
                                  for es in eval_sets if not es.correctness_only),
            weight_inputs=tuple(a in trace.weights for a in rep.input_ids),
            compute_floor_ms=roof.t_compute_ms if roof else 0.0,
        )

    def _kernel_from_proposal(self, run: RegionRun, region: Region, proposal, hyp_id: str) -> KernelSpec:
        kid = _kernel_id(region, hyp_id)
        parent = resolve_parent(run, proposal.parent_kernel_id)
        return kernel_from_proposal(self._contract(region), parent, proposal, kid)

    def _evaluate_kernel(self, region, kernel, assoc_tag, run_clock=True, target_workload=None):
        try:
            target = self._target_workload(region, target_workload)
        except ValueError as error:
            return LadderResult("failed", "static", {"reason": str(error)}, None, None, None, None, [])
        job = self._ladder_job(region, kernel, assoc_tag, run_clock, target)
        job.timing_incumbent = next((cut.kernel
            for _original, variants, _kernels in self.installed.values()
            for cuts in variants.values() for cut in cuts
            if cut.fingerprint == region.fingerprint), None)
        result = run_ladder(job, session=self.session)
        result.detail.update(target_workload=target, timing_case=job.eval_sets[0].label)
        # Preserve completed gates before whole-model checks can be interrupted.
        self.log.append("ladder_result", fingerprint=region.fingerprint,
                        kernel=kernel.kernel_id, assoc_tag=assoc_tag, result=asdict(result))
        return result

    def open_region(self, region: Region, judge) -> RegionRun:
        from .scaffold import NoScaffold

        run = RegionRun(region=region)
        for m in region.members:
            if region.t_orig_ms.get(m.workload) is None and m.workload in self._priced_workloads():
                run.close_rule = f"workload {m.workload!r} was never priced for this region"
                self.log.append("region_skip", fingerprint=region.fingerprint,
                                reason=run.close_rule)
                return run
        rep = region.members[0]
        trace = self.traces[rep.workload]
        try:
            scaffold = self._build_scaffold(region)
            original = self._rename(self._reference_scaffold(region), region, "binding_check")
        except NoScaffold as e:
            run.close_rule = f"no scaffold: {e}"
            self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
            return run
        except Exception as e:
            # a generator bug costs the region with a named reason, not the job
            run.close_rule = f"scaffold build error: {type(e).__name__}: {e}"
            self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
            return run
        binding = self._bind_and_promote(run, original, None, preflight=True)
        self.log.append("binding_preflight", fingerprint=region.fingerprint,
                        passed=binding.status == "preflight_passed", reason=binding.reason)
        if binding.status != "preflight_passed":
            run.close_rule = f"installation preflight failed: {binding.reason}"
            self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
            return run
        run.op_ms = self._op_times(region)
        scaffold = self._rename(scaffold, region, "scaffold")
        run.kernels[scaffold.kernel_id] = scaffold
        write_kernel(self.kernel_dir, scaffold)
        if scaffold.reference_sequence is not None:
            self.log.append("reference_starter", fingerprint=region.fingerprint,
                            ops=list(region.ops),
                            detail="Original operations retained; the judge writes replacement Metal from scratch")
        tag = _tag(scaffold)
        result = self._evaluate_kernel(region, scaffold, tag, run_clock=scaffold.reference_sequence is None)
        if result.outcome == "failed" and scaffold.reference_sequence is None:
            # Generated code can fail to compile, cover every shape, or preserve
            # rounding. Validate the original at every recorded shape before
            # spending an attempt on repairing our starter.
            try:
                original = self._rename(self._reference_scaffold(region), region, "scaffold")
            except NoScaffold:
                original = None
            if original is not None:
                self.scaffold_overrides[region.fingerprint] = original
                original_result = self._evaluate_kernel(region, original, "preserving", run_clock=False)
                if original_result.outcome != "failed":
                    self.log.append("scaffold_reference_fallback", fingerprint=region.fingerprint,
                                    failed_gate=result.failed_gate, detail=result.detail)
                    scaffold, result, tag = original, original_result, _tag(original)
                    run.kernels[scaffold.kernel_id] = scaffold
                    write_kernel(self.kernel_dir, scaffold)
                else:
                    self.scaffold_overrides.pop(region.fingerprint, None)
        if result.outcome == "failed":
            self._record_attempt(run, "scaffold", "scaffold", "the harness's starting kernel",
                                 tag, scaffold, None, result)
            self.log.append("scaffold_failed", fingerprint=region.fingerprint,
                            gate=result.failed_gate, detail=result.detail)
            if scaffold.reference_sequence is not None:
                # A replacement cannot repair an untrustworthy reference. Keep
                # this region closed rather than ask the judge to guess its math.
                run.close_rule = f"original reference failed validation at {result.failed_gate}"
                return run
            # one judge fix attempt, per the spec
            fixed = self._judge_fix(run, judge, scaffold, result, tag)
            if fixed is None:
                run.close_rule = f"scaffold failed {result.failed_gate} (the one judge fix attempt did not produce a passing kernel)"
                return run
            scaffold, result = fixed
        run.scaffold = scaffold
        self._set_head(run, scaffold, result, tag)
        self.log.append("scaffold_ok", fingerprint=region.fingerprint,
                        kernel=scaffold.kernel_id, region_ms=result.region_ms,
                        library_ms=result.library_ms)
        # the starting kernel is judged on the whole model like any edit; a
        # naive scaffold rarely wins, but a stitched one at library parity can
        promotion = self._bind_and_promote(run, scaffold, result, assoc_tag=tag)
        _promotion_feedback(result, promotion)
        if promotion:
            run.shipped, run.shipped_ms, run.shipped_ratio = scaffold, result.region_ms, _ratio(result)
            run.shipped_floor_ms = result.floor_ms
            run.shipped_workload = result.detail.get("target_workload")
            outcome = "shipped"
        else:
            outcome = promotion.outcome
            if outcome == "rolled_back":
                # Keep the rejected source for repair, but make the default
                # parent an original implementation with a matching contract.
                from .scaffold.native import native_seed
                prior_override = self.scaffold_overrides.get(region.fingerprint)
                try:
                    node = trace.nodes[rep.start_seq]
                    if rep.start_seq == rep.end_seq and node.kernel_definition is not None:
                        original = native_seed(trace, rep)
                    else:
                        original = self._reference_scaffold(region)
                    original = self._rename(original, region, "original")
                except (NoScaffold, ValueError, KeyError, AttributeError) as error:
                    original = None
                    run.close_rule = f"whole-model scaffold rejection; original starter unavailable: {error}"
                if original is not None:
                    self.scaffold_overrides[region.fingerprint] = original
                    original_result = self._evaluate_kernel(region, original, "preserving", run_clock=False)
                    if original_result.outcome == "failed":
                        run.close_rule = ("whole-model scaffold rejection; original starter failed "
                                          f"{original_result.failed_gate}: {_short_reason(original_result.detail)}")
                    else:
                        run.scaffold = original
                        run.kernels[original.kernel_id] = original
                        write_kernel(self.kernel_dir, original)
                        self._set_head(run, original, original_result, "preserving")
                        self.log.append("scaffold_model_fallback", fingerprint=region.fingerprint,
                                        rejected_kernel=scaffold.kernel_id, original_kernel=original.kernel_id)
                if run.close_rule is not None:
                    if prior_override is None:
                        self.scaffold_overrides.pop(region.fingerprint, None)
                    else:
                        self.scaffold_overrides[region.fingerprint] = prior_override
                    run.scaffold = run.head = None
                    run.head_ms = run.head_ratio = run.head_floor_ms = None
                    self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
        repaired = scaffold is not run.kernels.get(_kernel_id(region, "scaffold"))
        self._record_attempt(
            run, "scafix" if repaired else "scaffold", "fix" if repaired else "scaffold",
            "the judge's one fix of the starting kernel" if repaired else "the harness's starting kernel",
            tag, scaffold, None, result, outcome=outcome)
        return run

    def _reference_scaffold(self, region: Region):
        from .scaffold.native import reference_sequence_seed
        rep = region.members[0]
        instances = [(self.traces[m.workload], m)
                     for _, m, _ in capture_instances(region, self.traces)]
        return reference_sequence_seed(self.traces[rep.workload], rep, instances)

    def _build_scaffold(self, region: Region):
        if region.fingerprint in getattr(self, "scaffold_overrides", {}):
            return self.scaffold_overrides[region.fingerprint]
        from .scaffold import NoScaffold, build_scaffold
        from .ladder.static_checks import METAL_BUFFER_LIMIT, buffer_count
        instances = [(self.traces[m.workload], m)
                     for _label, m, _copies in capture_instances(region, self.traces)]
        for workload in self.manifest.workloads:
            rep = next((m for m in region.members if m.workload == workload.name), None)
            if rep is None:
                continue
            for label, _dims in self._sweep_points(workload):
                trace = self.sweep_traces[label]
                instances.append((trace, locate_span(self.traces[workload.name], rep,
                                                     trace, workload.name)))
        shapes = [[trace.span_specs(m.start_seq, m.end_seq)[a][0] for a in m.input_ids]
                  for trace, m in instances]
        rep = region.members[0]
        scaffold = build_scaffold(self.traces[rep.workload], rep, shapes)
        if scaffold.reference_sequence is not None:
            scaffold = self._reference_scaffold(region)
        buffers = buffer_count(scaffold, [len(shape) for shape in shapes[0]])
        if buffers > METAL_BUFFER_LIMIT:
            raise NoScaffold("buffer-limit", f"starter needs {buffers} Metal buffer arguments; "
                             f"the limit is {METAL_BUFFER_LIMIT}")
        return scaffold

    def _rename(self, spec: KernelSpec, region: Region, tag: str) -> KernelSpec:
        kid = _kernel_id(region, tag)
        d = {k: getattr(spec, k) for k in spec.__dataclass_fields__}
        d["kernel_id"] = kid
        d["name"] = f"at_{kid}"
        return KernelSpec(**d)

    def _judge_fix(self, run: RegionRun, judge, scaffold, result, tag: str):
        """The spec's one repair attempt on a starting kernel that failed its
        own checks, held to the same rule (tag) as the starter. Returns
        (kernel, ladder result) or None."""
        if self._close_rule(run) is not None:
            return None
        self._spend(run)
        region = run.region
        run.head, run.last_kernel = scaffold, scaffold.kernel_id
        verdict = _verdict_payload("scaffold", scaffold.kernel_id, "failed", result)
        writing_for = {"id": "scafix", "kind": "fix", "assoc_tag": tag,
                       "hypothesis": "repair the starting kernel so it passes the checks"}
        meta = self._meta(run, Queue(), writing_for)
        with self._judge_tries(judge, run, Queue()):
            resp, error = self._ask_judge(run, "scaffold_fix", lambda: judge.next(meta, verdict))
        if error is not None:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint,
                            outcome="judge_error", reason=error)
            return None
        self._note_lesson(run, resp)
        if resp.kernel is None:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint, outcome="yield")
            return None
        fixed = self._kernel_from_proposal(run, region, resp.kernel, "scafix")
        run.kernels[fixed.kernel_id] = fixed
        write_kernel(self.kernel_dir, fixed)
        target = {"target_workload": resp.kernel.target_workload} if resp.kernel.target_workload else {}
        check = self._evaluate_kernel(region, fixed, tag, **target)
        if check.outcome == "failed":
            self._record_attempt(run, "scafix", "fix", "the judge's one fix of the starting kernel",
                                 tag, fixed, resp.kernel.parent_kernel_id, check)
            self.log.append("scaffold_fix", fingerprint=region.fingerprint,
                            outcome="fix_failed_ladder", gate=check.failed_gate,
                            detail=check.detail)
            return None
        self.log.append("scaffold_fix", fingerprint=region.fingerprint, outcome="fixed")
        return fixed, check

    def hypothesis_cycle(self, run: RegionRun, judge) -> None:
        """One hypothesis at a time until the budget is spent. Every call to
        the judge carries the last verdict and the queue; the judge edits its
        plan first, then writes Metal for the front ready item; the ladder
        decides; repeat. A reply with nothing to evaluate is asked again once
        with the reason, then each further one costs an attempt, so the
        budget is spent by the judge and never handed back."""
        rule = self._close_rule(run)
        if rule:
            run.close_rule = rule
            return
        region = run.region
        if run.queue is None and getattr(judge, "combined_start", False):
            run.queue = Queue()  # the normal mutation/proposal protocol also starts a search
        if run.queue is None:
            run.queue = Queue()
            queue = run.queue
            verdict = run.last_verdict
            meta = self._meta(run, queue, None)
            resp, failure = self._ask_judge(run, "seed", lambda: judge.seed(meta))
            problem = None
            if resp is not None:
                self._note_lesson(run, resp)
                try:
                    queue.seed(resp.queue)
                except QueueError as e:
                    problem = f"your seed queue was refused: {e}"
            if resp is None or problem:
                verdict, close = self._empty_reply(run, None, failure, problem, verdict)
                run.last_verdict = verdict
                if close:
                    run.close_rule = close
                    return
        queue, verdict = run.queue, run.last_verdict

        while True:
            rule = self._close_rule(run)
            if rule:
                run.close_rule = rule
                return
            # before each judge call: the verdict the call carries is the resume point
            run.last_verdict = verdict
            self._safe_point(f"region {region.fingerprint}, attempt {run.hypotheses + 1}")
            front = queue.peek_ready()
            meta = self._meta(run, queue, _item_view(front) if front else None)
            replay, self._replay = getattr(self, "_replay", None), None
            if replay and replay["region"] == region.fingerprint:
                # the GPU failed under the job while it ran this reply's kernel: the
                # same kernel again, from the same saved point, without the judge
                resp, failure = replay["response"], None
                self.log.append("attempt_replayed", fingerprint=region.fingerprint,
                                kernel=replay["kernel"], deaths=replay["deaths"])
            else:
                replay = None
                with self._judge_tries(judge, run, queue):
                    resp, failure = self._ask_judge(run, "next", lambda: judge.next(meta, verdict))
            item = problem = None
            if resp is not None:
                run.errors = 0
                self._note_lesson(run, resp)
                try:
                    queue.apply_mutations(resp.mutations)
                except QueueError as e:
                    problem = f"your plan edits were refused and none applied: {e}"
                else:
                    if resp.kernel is None:
                        problem = "a yield is refused while the budget lasts"
                    else:
                        item = queue.ready_item(resp.kernel.item_id)
                        if item is None:
                            named = resp.kernel.item_id
                            problem = (f"your kernel is for {named!r}, which is not a ready queued item"
                                       if named else "no queued item is ready for your kernel") + \
                                      f" (queued: {', '.join(queue.ids()) or 'nothing'})"
                        else:
                            problem = (self._opener_rule(run, item, resp.kernel.parent_kernel_id)
                                       or self._static_refusal(run, resp.kernel, item.id))
                            if problem is None:
                                queue.pop_ready(item.id)
                            else:
                                item = None
            if item is None:
                verdict, close = self._empty_reply(run, front, failure, problem, verdict)
                if close:
                    run.close_rule = close
                    return
                continue
            run.refused = 0
            self._spend(run)
            parent_spec = resolve_parent(run, resp.kernel.parent_kernel_id)
            parent = parent_spec.kernel_id if parent_spec else resp.kernel.parent_kernel_id
            opener = parent_spec is run.scaffold and len(run.openers) < self._openers(run)
            if opener:
                run.openers.append(item.kind)
            elif run.runner_up_left and parent_spec is not None \
                    and _in_branch(run, parent_spec.kernel_id, run.runner_up):
                run.runner_up_left -= 1
            kernel = self._kernel_from_proposal(run, region, resp.kernel, item.id)
            if opener:
                run.opener_kernels += (kernel.kernel_id,)
            run.kernels[kernel.kernel_id] = kernel
            write_kernel(self.kernel_dir, kernel)
            deaths = replay["deaths"] if replay else []
            if getattr(self, "_resumable", False):
                resume_mod.start_attempt(self.work_dir, region.fingerprint, kernel.kernel_id, resp, deaths)
            if len(deaths) >= resume_mod.DEATHS_TO_DROP:
                result = LadderResult("failed", "job_died", {"failures": [{
                    "check": "job_died",
                    "detail": f"the GPU failed under the job {len(deaths)} times while it ran this kernel "
                              f"(in {', '.join(d['step'] for d in deaths)}; last: {deaths[-1]['error']}); "
                              "it was dropped without its outputs being judged. The failure may come from "
                              "the Mac (memory, other processes) rather than the kernel"}]},
                    None, None, None, None, [])
            elif parent_spec is not None:
                target = {"target_workload": resp.kernel.target_workload} if resp.kernel.target_workload else {}
                result = self._evaluate_kernel(region, kernel, item.assoc_tag, **target)
            else:
                # the parent is the judge's own memory of what it edited; an
                # unknown one is a mistake to name, not a kernel to measure
                result = LadderResult("failed", "static", {"failures": [{
                    "check": "parent_kernel_id",
                    "detail": f"{parent!r} names no kernel of this region; say head, "
                              f"scaffold, shipped, a hypothesis id, or one of {sorted(run.kernels)}"}]},
                    None, None, None, None, [])
            outcome = self._apply_verdict(run, item, kernel, result)
            queue.record_verdict(item.id, outcome)
            verdict = _verdict_payload(item.id, kernel.kernel_id, outcome, result, self._size_table(result.detail))
            self._record_attempt(run, item.id, item.kind, item.hypothesis, item.assoc_tag,
                                 kernel, parent, result, outcome=outcome)
            if getattr(self, "_resumable", False):
                resume_mod.end_attempt(self.work_dir)
            self._settle_runner_up(run)

    def _settle_runner_up(self, run: RegionRun) -> None:
        """Once every opener has a verdict: when the second-best opener's
        time over the library clocked beside it is within RUNNER_UP_MARGIN of
        the best's, or was clocked at another workload so the two cannot be
        ranked, its branch takes the next RUNNER_UP_FOLLOW_UPS kernels."""
        if run.runner_up is not None or len(run.opener_kernels) < self._openers(run) \
                or any(k not in run.attempts for k in run.opener_kernels):
            return
        ratios = sorted((a["region_ms"] / a["library_ms"], k, a.get("target_workload"))
                        for k in run.opener_kernels for a in [run.attempts[k]]
                        if a.get("region_ms") and a.get("library_ms"))
        if len(ratios) < 2 or (ratios[1][2] == ratios[0][2]
                               and ratios[1][0] > ratios[0][0] * (1 + RUNNER_UP_MARGIN)):
            return
        run.runner_up, run.runner_up_left = ratios[1][1], RUNNER_UP_FOLLOW_UPS
        self.log.append("runner_up", region=run.region.fingerprint, kernel=run.runner_up,
                        ratio=round(ratios[1][0], 4), best=ratios[0][1],
                        best_ratio=round(ratios[0][0], 4), follow_ups=RUNNER_UP_FOLLOW_UPS,
                        comparable=ratios[1][2] == ratios[0][2])

    def _note_lesson(self, run: RegionRun, resp) -> None:
        """Link a note to the evaluated candidate and its parent, never a future proposal."""
        result = run.attempts.get(run.last_kernel)
        if not resp.lesson or result is None:
            return
        evidence = [run.last_kernel]
        parent = result.get("parent")
        if parent in run.attempts and parent not in evidence:
            evidence.append(parent)
        note = {"region": run.region.fingerprint, "ops": list(run.region.ops),
                "lesson": resp.lesson, "evidence": evidence,
                "results": {kid: run.attempts[kid] for kid in evidence}}
        if note in self.lessons:
            return
        self.lessons.append(note)
        self.log.append("lesson", fingerprint=run.region.fingerprint, **note)
        if self.knowledge is not None:
            self.knowledge.add_reflection(run.last_kernel, resp.lesson)

    def _budget(self, run: RegionRun) -> dict:
        """Attempts left for the region and for the job: what the judge is
        told and what the close rule reads."""
        return {"attempts_left_region": self.manifest.budget_per_region - run.hypotheses,
                "attempts_left_job": self.manifest.budget_total - self.total_hypotheses}

    def _static_refusal(self, run: RegionRun, proposal, hyp_id: str) -> str | None:
        """The ladder's first gate, asked before the attempt is charged. A kernel
        the static checks reject never reaches the GPU, so the judge hears why
        under the refusal rule (asked again once for free, then charged)
        instead of losing an attempt to a contract slip; 14 of 140 MetalBench
        attempts went to one such slip. The ladder still runs the same checks.
        Returns the refusal, or None."""
        if resolve_parent(run, proposal.parent_kernel_id) is None:
            return None  # an unknown parent is the verdict's to report
        failures = static_check(self._kernel_from_proposal(run, run.region, proposal, hyp_id),
                                self._contract(run.region))
        if not failures:
            return None
        return "the static checks refuse your kernel: " + "; ".join(f"{f.check}: {f.detail}" for f in failures)

    def _spend(self, run: RegionRun) -> None:
        run.hypotheses += 1
        self.total_hypotheses += 1

    def _openers(self, run: RegionRun) -> int:
        """How many openers the region's widening round holds."""
        return OPENERS

    def _opener_rule(self, run: RegionRun, item, parent_name: str) -> str | None:
        """While the widening round lasts, a kernel is an opener, written
        against the scaffold under a kind no earlier opener used, or a repair
        of a kernel that failed. Returns the refusal, or None. A parent that
        names nothing is left to the verdict that says so."""
        left = self._openers(run) - len(run.openers)
        if left <= 0:
            return self._runner_up_rule(run, parent_name)
        parent = resolve_parent(run, parent_name)
        if parent is None:
            return None
        if parent is run.scaffold:
            if item.kind in run.openers:
                return (f"the widening round already opened {item.kind!r}; an opener needs a kind "
                        f"no earlier opener used (opened: {', '.join(run.openers)})")
            return None
        if run.attempts.get(parent.kernel_id, {}).get("verdict") in ("failed", "rolled_back"):
            return None
        return (f"the widening round has {left} opener(s) left, so {parent_name!r} is refused as a "
                f"parent: write against the scaffold under a kind no earlier opener used, "
                f"or repair a kernel that failed")

    def _runner_up_rule(self, run: RegionRun, parent_name: str) -> str | None:
        """While the runner-up's follow-ups last, a kernel builds on the
        runner-up or a kernel derived from it, or repairs a kernel that failed."""
        if not run.runner_up_left:
            return None
        parent = resolve_parent(run, parent_name)
        if parent is None or _in_branch(run, parent.kernel_id, run.runner_up) \
                or run.attempts.get(parent.kernel_id, {}).get("verdict") in ("failed", "rolled_back"):
            return None
        return (f"the second-best opener {run.runner_up!r} trails the best by under "
                f"{RUNNER_UP_MARGIN:.0%} or was timed at another workload, too close for one attempt "
                f"to rank, so its branch takes the "
                f"next {run.runner_up_left} kernel(s): build on {run.runner_up!r} or a kernel derived "
                f"from it, or repair a kernel that failed")

    def _empty_reply(self, run: RegionRun, front, failure: str | None, problem: str | None,
                     last: dict | None) -> tuple[dict, str | None]:
        """Bookkeeping for a reply that left nothing to evaluate: babble, a
        transport failure, or a refusal (a yield, refused plan edits, a
        kernel for an item that is not ready, a refused seed). Returns the
        verdict the next call carries; three transport failures in a row pause
        the job, resumably. A babble costs an attempt, since the transport already asked
        once more; a refusal is asked again once for free, then charged."""
        if failure == "babble":
            self._spend(run)
            hyp = self._record_empty(run, front, "judge_babble", None)
            return {"hypothesis_id": hyp, "outcome": "failed", "failed_gate": "judge_babble"}, None
        if failure is not None:
            run.errors += 1
            hyp = self._record_empty(run, front, "judge_error", failure)
            verdict = {"hypothesis_id": hyp, "outcome": "failed", "failed_gate": "judge_error",
                       "detail": {"reason": failure}}
            if run.errors >= 3:
                # the network or the judge's service is down (a Mac waking from sleep):
                # stop where a resumed job can ask again, rather than fail
                raise resume_mod.Paused(f"the judge did not answer 3 times in a row; last error: {failure}")
            return verdict, None
        run.refused += 1
        if run.refused > 1:
            self._spend(run)
            self._record_empty(run, front, "plan_refused", problem)
        problem = f"{problem}; {self._budget(run)['attempts_left_region']} attempts remain in this region"
        self.log.append("plan_refused", fingerprint=run.region.fingerprint, reason=problem)
        return {**(last or {}), "plan_refused": problem}, None

    def _record_empty(self, run: RegionRun, front, gate: str, reason: str | None) -> str:
        """An attempt row for a reply with nothing to evaluate, under its own
        id: the queued item it was asked for stays queued and keeps its name."""
        run.empty_replies += 1
        hyp = f"{front.id if front else 'none'}_{gate}{run.empty_replies}"
        self._record_attempt(run, hyp, front.kind if front else "none",
                             front.hypothesis if front else "", None, None, None, None,
                             gate=gate, reason=reason)
        return hyp

    @contextmanager
    def _judge_tries(self, judge, run: RegionRun, queue):
        """While the judge writes the region's next kernel, a judge that can
        run ./try (a CLI agent) gets the quick check for its candidates."""
        if not hasattr(judge, "try_handler"):
            yield
            return
        judge.try_handler = lambda payload: self._quick_try(run, queue, payload)
        try:
            yield
        finally:
            judge.try_handler = None

    def _quick_try(self, run: RegionRun, queue, payload) -> dict:
        """One ./try: the candidate's outputs checked against the region's
        recordings and a short clock against the library at every timed size,
        without shader validation or pacing idles (LadderJob.quick). Advisory:
        never a verdict or an attempt."""
        region = run.region
        kernel_obj = payload.get("kernel") if isinstance(payload, dict) else None
        try:
            proposal = _proposal(kernel_obj)
        except MalformedResponse as error:
            return {"outcome": "failed", "failed_gate": "schema", "reason": str(error)}
        if resolve_parent(run, proposal.parent_kernel_id) is None:
            return {"outcome": "failed", "failed_gate": "parent_kernel_id",
                    "reason": f"{proposal.parent_kernel_id!r} names no kernel of this region; say head, "
                              f"scaffold, shipped, a hypothesis id, or one of {sorted(run.kernels)}"}
        item_id = payload.get("item_id")
        item = queue.ready_item(item_id) if isinstance(item_id, str) else None
        run.tries += 1
        kernel = self._kernel_from_proposal(run, region, proposal, f"try{run.tries}")
        started = time.perf_counter()
        try:
            job = self._ladder_job(region, kernel, item.assoc_tag if item else "preserving", True,
                                   proposal.target_workload)
            job = replace(job, quick=True, clock_pairs=QUICK_TRY_PAIRS)
            result = run_ladder(job, session=self.session)
        except Exception as error:  # a try never stops the job; the submitted kernel gets the real checks
            self.log.append("judge_try", fingerprint=region.fingerprint, kernel=kernel.kernel_id,
                            outcome="error", reason=f"{type(error).__name__}: {error}")
            return {"outcome": "failed", "failed_gate": "harness", "reason": f"{type(error).__name__}: {error}"[:2000]}
        outcome = {"tentative_ship": "faster"}.get(result.outcome, result.outcome)
        answer = {"outcome": outcome, "failed_gate": result.failed_gate,
                  "kernel_ms": result.region_ms, "library_ms": result.library_ms,
                  "seconds": round(time.perf_counter() - started, 1)}
        sizes = (result.detail or {}).get("sizes") or {}
        if sizes:  # the full check's rule: faster at half the sizes or more, slower at none
            answer["sizes"] = {label: {"kernel_ms": row["kernel_ms"], "library_ms": row["library_ms"],
                                       "result": "win" if row["win"] else "loss" if row["loss"] else "tie"}
                               for label, row in sizes.items()}
        if result.outcome == "failed":
            # as a verdict shows it: acceptance settings hidden
            answer["reason"] = json.dumps(_safe_detail(_without_raw_sizes(result.detail)), default=str)[:3000]
        per_size = {label: round(row["kernel_ms"] / row["library_ms"], 3)
                    for label, row in sizes.items() if row["library_ms"]}
        run.try_log = (run.try_log + [{"try": run.tries, "kernel": kernel.kernel_id,
                                       "parent": proposal.parent_kernel_id, "outcome": outcome,
                                       "failed_gate": result.failed_gate,
                                       "kernel_over_library": per_size}])[-TRIES_SHOWN:]
        if result.outcome != "failed" and per_size:
            ratio = max(per_size.values())  # ranked by its worst size: a loss anywhere stops it shipping
            if run.best_try is None or ratio < run.best_try[0]:
                if run.best_try is not None:
                    run.kernels.pop(run.best_try[1], None)
                run.best_try = (ratio, kernel.kernel_id)
                run.kernels[kernel.kernel_id] = kernel
                answer["kept_as"] = kernel.kernel_id
        self.log.append("judge_try", fingerprint=region.fingerprint, kernel=kernel.kernel_id,
                        outcome=outcome, failed_gate=result.failed_gate, kernel_over_library=per_size,
                        seconds=answer["seconds"])
        return answer

    def _ask_judge(self, run: RegionRun, phase: str, call):
        """One judge call with its log row: (response, None), or (None, why)."""
        t0 = time.perf_counter()
        print(f"judge: {phase} for region {run.region.fingerprint[:8]}", flush=True)
        self.log.append("judge_pending", fingerprint=run.region.fingerprint, phase=phase)
        try:
            resp = call()
        except JudgeBabble:
            run.errors = 0  # a malformed answer still proves the transport answered
            self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                            latency_s=round(time.perf_counter() - t0, 1), action="babble")
            return None, "babble"
        except Exception as e:
            # Preserve the diagnostic; repeated transport failures pause the job
            # in _empty_reply without consuming optimization attempts.
            self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                            latency_s=round(time.perf_counter() - t0, 1),
                            action="error", reason=str(e))
            return None, f"{type(e).__name__}: {e}"
        summary = ({"queue": len(resp.queue)} if phase == "seed" else
                   {"mutations": len(resp.mutations),
                    "action": "yield" if resp.kernel is None else "proposal"})
        run.errors = 0
        self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                        latency_s=round(time.perf_counter() - t0, 1), **summary)
        return resp, None

    def _set_head(self, run: RegionRun, kernel: KernelSpec, clock, tag: str) -> None:
        """clock is a ladder result or a recorded attempt: anything with the
        kernel's region_ms and library_ms."""
        get = clock.get if isinstance(clock, dict) else lambda k: getattr(clock, k)
        run.head, run.head_tag = kernel, tag
        run.head_ms = get("region_ms")
        run.head_floor_ms = get("floor_ms")
        run.head_sigma_ms = get("sigma_ms")
        run.head_age = 0
        run.head_ratio = _ratio(clock)
        run.head_workload = (clock.get("target_workload") if isinstance(clock, dict)
                             else clock.detail.get("target_workload"))

    def _apply_verdict(self, run: RegionRun, item, kernel: KernelSpec, result) -> str:
        """Fail, climb, or ship. A correct kernel is installed and judged on
        the whole model, never on the region clock: the region clock cannot
        see the call site or the wrapper, so it stays informational and only
        picks head, the kernel the judge keeps editing. Its ratio to the
        library it was clocked beside compares across clocks taken minutes
        apart, which the running-time milliseconds do not."""
        run.head_age += 1
        at_floor = (None not in (result.region_ms, result.floor_ms, result.sigma_ms)
                    and result.region_ms - result.floor_ms <= result.sigma_ms)
        run.floor_streak = run.floor_streak + 1 if at_floor else 0
        if result.outcome == "failed":
            return "failed"
        promotion = self._bind_and_promote(run, kernel, result, assoc_tag=item.assoc_tag)
        _promotion_feedback(result, promotion)
        if promotion:
            run.shipped, run.shipped_ms, run.shipped_ratio = kernel, result.region_ms, _ratio(result)
            run.shipped_floor_ms = result.floor_ms
            run.shipped_workload = result.detail.get("target_workload")
            self._set_head(run, kernel, result, item.assoc_tag)
            return "shipped"
        if promotion.outcome == "rolled_back":
            return "rolled_back"  # retain the source for repair, keep the previous valid head
        # Correctness passed (or the region was not nominated for model testing).
        # Ratios from different workloads are not comparable. Keep this
        # attempt editable without pretending it beat a head timed elsewhere.
        if (run.head_ratio is None or run.head_workload != result.detail.get("target_workload")
                or _ratio(result) < run.head_ratio):
            self._set_head(run, kernel, result, item.assoc_tag)
        return "correct_slower"

    def _record_attempt(self, run: RegionRun, hyp_id: str, kind: str, text: str,
                        assoc_tag: str | None, kernel, parent: str | None, result,
                        outcome: str | None = None, gate: str | None = None,
                        reason: str | None = None) -> None:
        """One attempt, written three ways: the report row, the run.jsonl
        verdict row, and one line of candidates.log."""
        region = run.region
        if result is None:  # the judge produced nothing to evaluate
            outcome, detail = "failed", {"reason": reason} if reason else {}
            region_ms = library_ms = win_ms = sigma_ms = floor_ms = None
        else:
            outcome = outcome or result.outcome
            gate, detail = result.failed_gate, result.detail
            region_ms, library_ms = result.region_ms, result.library_ms
            win_ms, sigma_ms, floor_ms = result.win_ms, result.sigma_ms, result.floor_ms
        sizes = self._size_table(detail)
        if kernel is not None:
            run.last_kernel = kernel.kernel_id
            run.attempts[kernel.kernel_id] = {
                "hypothesis_id": hyp_id, "kernel_id": kernel.kernel_id, "parent": parent,
                "hypothesis": text, "kind": kind, "assoc_tag": assoc_tag,
                "verdict": outcome, "failed_gate": gate, "detail": _safe_detail(_without_raw_sizes(detail)),
                "region_ms": region_ms, "library_ms": library_ms, "win_ms": win_ms,
                "floor_ms": floor_ms, "sigma_ms": sigma_ms,
                "target_workload": detail.get("target_workload"),
                "timing_case": detail.get("timing_case"),
                "model_check": _safe_detail(detail.get("model_check", {})),
                "incumbent_screen": _safe_detail(detail.get("incumbent_screen", {})),
                **({"sizes": sizes} if sizes else {}),
            }
            if self.knowledge is not None:
                view = _kernel_view(kernel)
                self.knowledge.add_attempt(
                    region.ops, self._io_specs(region), kernel.kernel_id,
                    knowledge_outcome(outcome, gate, region_ms, library_ms,
                                      detail.get("model_check", {}).get("status"), sizes),
                    outcome == "shipped",  # right and faster in the whole model, not just its region
                    text, {k: view[k] for k in ("header", "source", "stages", "grid", "threadgroup")
                           if view.get(k)})
        if detail.get("incumbent_screen", {}).get("resolved_regression"):
            how = "correct_slower: slower than the installed kernel in a fresh region comparison; model timing skipped"
        elif detail.get("model_check", {}).get("reason"):
            how = f"{outcome}: {detail['model_check']['reason']}"
        elif gate:
            how = f"{outcome} at {gate}: {_short_reason(detail)}"
        elif sizes and len(sizes) > 1:
            how = (f"{outcome}: faster at {sum(r['result'] == 'faster' for r in sizes)} of {len(sizes)} sizes, "
                   f"slower at {', '.join(r['size'] for r in sizes if r['result'] == 'slower') or 'none'} "
                   f"({min(r['ratio'] for r in sizes):.2f}x to {max(r['ratio'] for r in sizes):.2f}x the library)")
        elif region_ms is not None and library_ms is not None:
            how = (f"{outcome}: {region_ms:.4f} ms vs library {library_ms:.4f} ms per copy "
                   f"(win {win_ms:+.4f}, sigma {sigma_ms:.4f})")
        elif outcome == "correct_slower":
            how = "correct, not clocked"
        else:
            how = outcome
        self.report.add_hypothesis(
            hypothesis_id=hyp_id, region=region.fingerprint, kind=kind,
            hypothesis_text=text, assoc_tag=assoc_tag, parent=parent,
            kernel=kernel.kernel_id if kernel else None, verdict=outcome,
            failed_gate=gate, region_ms=region_ms, library_ms=library_ms,
            win_ms=win_ms, sigma_ms=sigma_ms, floor_ms=floor_ms, summary=how,
            target_workload=detail.get("target_workload"), timing_case=detail.get("timing_case"),
            sizes=sizes,
        )
        self.log.append("verdict", fingerprint=region.fingerprint, hypothesis=hyp_id,
                        hypothesis_kind=kind, hypothesis_text=text, assoc_tag=assoc_tag,
                        kernel=kernel.kernel_id if kernel else None, parent=parent,
                        outcome=outcome, gate=gate, region_ms=region_ms,
                        library_ms=library_ms, win_ms=win_ms, sigma_ms=sigma_ms,
                        floor_ms=floor_ms, detail=detail)
        self.candidates.append(
            f"{wall_now()}\t{region.fingerprint[:8]}\t{hyp_id}\t{kind}\t{how}\t{text}")

    def _size_table(self, detail: dict | None) -> list[dict]:
        """The region clock at every size, smallest first, as the judge and the
        knowledge file see it: the size, the library's and the kernel's time
        per copy, their ratio and whether the kernel was faster or slower
        beyond the timing noise. Empty when the kernel was not clocked."""
        rows = (detail or {}).get("sizes") or {}
        if not rows:
            return []
        named = self.report.constants.get("search_sizes") or {}
        dim = next(iter(named), None)

        def size(label):
            # labels: a workload ("prompt", "prompt@T=42"), maybe with a second
            # boundary shape of the region in it ("...@copy:796"; under MLX-LM,
            # the prompt's last token, run alone)
            workload, _, copy = label.removesuffix(":holdout").partition("@copy:")
            tail = workload.split("@", 1)[1] if "@" in workload else ""
            if "=" not in tail:
                w = next((w for w in self.manifest.workloads if w.name == workload), None)
                tail = f"{dim}={self.manifest.primary[dim]}" if dim and w and dim in w.named_dims() else workload
            return tail

        def note(label, row):
            notes = ([f"another shape: input {row['shape']}"] if "@copy:" in label and row.get("shape") else []) \
                + ([f"the same shape at {row['same_at']} sizes, timed once"] if row.get("same_at", 1) > 1 else []) \
                + (["drawn fresh for the check before the whole model"] if label.endswith(":holdout") else []) \
                + (["left to MLX by the kernel's fallback; kernel_ms is MLX"] if row.get("declined") else [])
            return f" ({'; '.join(notes)})" if notes else ""

        def order(label):
            value = size(label).split(" ", 1)[0].rsplit("=", 1)[-1]
            return (0, int(value), label) if value.isdigit() else (1, 0, label)

        return [{"size": size(label) + note(label, row), "mlx_ms": round(row["library_ms"], 4), "kernel_ms": round(row["kernel_ms"], 4),
                 "ratio": round(row["library_ms"] / row["kernel_ms"], 3),
                 "result": "declined" if row.get("declined") else
                           "faster" if row["win"] else "slower" if row["loss"] else "tie"}
                for label, row in sorted(rows.items(), key=lambda item: order(item[0]))]

    def _io_specs(self, region: Region) -> dict:
        """Each recorded instance's input and output specs, by label."""
        io_specs = {}
        for label, m, _copies in capture_instances(region, self.traces):
            specs = self.traces[m.workload].span_specs(m.start_seq, m.end_seq)
            io_specs[label] = {
                "inputs": [specs[a] for a in m.input_ids],
                "outputs": [specs[a] for a in m.output_ids],
            }
        for label, span, specs in self._sweep_instances(region):
            io_specs[label] = {
                "inputs": [specs[a] for a in span.input_ids],
                "outputs": [specs[a] for a in span.output_ids],
            }
        return io_specs

    def _meta(self, run: RegionRun, queue: Queue, writing_for: dict | None) -> dict:
        """Approved metadata; the transport shows a compact view with exact lookups."""
        region = run.region
        io_specs = self._io_specs(region)
        if not run.past_runs:  # nothing read yet, or nothing there when it was: earlier jobs' work can arrive mid-job
            run.past_runs = self.knowledge.index(region.fingerprint) if self.knowledge else []
        rep = region.members[0]
        kernels = {kid: {**_kernel_view(spec), **run.attempts.get(kid, {})}
                   for kid, spec in run.kernels.items()}
        chip = {"gpu_cores": gpu_core_count(), **gpu_facts()}
        if self.peaks is not None:
            chip.update(bandwidth_gbps=self.peaks.bandwidth_gbps, launch_us=self.peaks.launch_us,
                        flops_gflops=self.peaks.flops_gflops)
        history = [{**h, "detail": run.attempts.get(h.get("kernel"), {}).get("detail", {})}
                   for h in self.report.hypotheses if h["region"] == region.fingerprint]
        done = [{"ops": r["ops"], "copies": r["copies"], "attempts": r["hypotheses"],
                 "shipped": f"{r['s']:.2f}x the library" if r.get("s") else "nothing",
                 "close": r["close_rule"]} for r in self.report.regions]
        return render_region_state(
            region=region, io_specs=io_specs, chip=chip,
            ops=_ops_view(self.traces[rep.workload], rep),
            kernels=kernels,
            scaffold=run.scaffold.kernel_id if run.scaffold else None,
            head=run.head.kernel_id if run.head else None,
            shipped=run.shipped.kernel_id if run.shipped else None,
            head_ms=run.head_ms, shipped_ms=run.shipped_ms, assoc_tag=run.head_tag,
            head_floor_ms=run.head_floor_ms, shipped_floor_ms=run.shipped_floor_ms,
            queue=queue, history=history, lessons=list(self.lessons), regions_done=done,
            budget=self._budget(run),
            last_verdict=run.attempts.get(run.last_kernel) if run.last_kernel else None,
            writing_for=writing_for,
            default_target_workload=self._target_workload(region),
            head_workload=run.head_workload, shipped_workload=run.shipped_workload,
            widening={"openers": self._openers(run), "opened": list(run.openers),
                      "left": max(self._openers(run) - len(run.openers), 0),
                      **({"runner_up": run.runner_up, "runner_up_left": run.runner_up_left}
                         if run.runner_up_left else {})},
            # the openers start from the region alone, so past work cannot narrow them
            past_runs=run.past_runs if len(run.openers) >= self._openers(run) else [],
            op_ms=run.op_ms,
            exact_only=self._exact_only(),
            tries=run.try_log,
        )

    def _op_times(self, region: Region) -> dict:
        """Each op of one copy timed alone on the default timing workload's
        captured inputs: what an engineer looks at before writing a kernel."""
        target = self._target_workload(region)
        label, rep, _copies = next(i for i in capture_instances(region, self.traces) if i[1].workload == target)
        times = op_times(self.session, self.traces[rep.workload], rep,
                         self.store.load(region.fingerprint, label, 0, "inputs"))
        ops = [{"op": op.split(".")[-1], "ms": round(ms, 4)} for op, ms in times]
        self.log.append("op_times", fingerprint=region.fingerprint, workload=target, ops=ops)
        return {"workload": target, "ops": ops}

    def _request_dirs(self) -> tuple:
        """Where an operator's finish, pause and skip requests are read: the
        job's folder, and for the kernel search a loop search started, its
        parent's, where the CLI writes them."""
        parent = getattr(self, "parent_dir", None)
        return (self.work_dir,) if parent is None else (self.work_dir, parent)

    def _active_s(self) -> float:
        """Seconds this job has run, across pause and resume."""
        before, start = getattr(self, "_clock_base", (0.0, time.monotonic()))
        return before + time.monotonic() - start

    def _finish_requested(self) -> bool:
        if not any((d / "finish-search.request").is_file() for d in self._request_dirs()):
            return False
        if not self.report.session.get("finish_requested"):
            self.report.session["finish_requested"] = True
            self.log.append("search_finish_requested", reason="operator requested final validation")
            self.report.write(self.work_dir / "report.json")
        return True

    def _close_rule(self, run: RegionRun) -> str | None:
        """Spend the search budget, unless the operator requests finalization
        or nothing inside a kernel can pay any more: a launch-bound region
        whose head the last two attempts did not move, the last three kernels
        each within one sigma of the launch floor clocked beside them. One
        window is weak evidence: the floor probe swings about 30% between
        windows with the GPU's state, so three windows must agree."""
        if self._finish_requested():
            return "operator requested final validation"
        if any((d / resume_mod.SKIP_REQUEST).is_file() for d in self._request_dirs()):
            return "operator skipped the region"
        left = self._budget(run)
        if left["attempts_left_region"] <= 0:
            return "the region's hypothesis budget is spent"
        if left["attempts_left_job"] <= 0:
            return "the job budget is spent"
        roof = run.region.roofline
        if (len(run.openers) >= self._openers(run) and roof is not None
                and roof.bound == "launch" and run.head_age >= 2 and run.floor_streak >= 3):
            return ("no discernible headroom: the region is launch-bound, the last two attempts did "
                    "not move head, and the last three kernels each sat within one sigma of the "
                    "launch floor clocked beside them")
        return None

    # -- bind and promote -----------------------------------------------------

    def _emit_installation(self, definitions, name, *, replay=False):
        """Prefer graph insertion; retain certified replay for unsafe state boundaries.
        A job whose calls read generated tokens replays: a compiled graph would
        not run them under exact_rows."""
        from .bind.emit import covers_scope, scope_nodes
        replay = replay or self._generated()
        if all(covers_scope(scope_nodes(trace, scope), [(c.start_seq, c.end_seq) for c in cuts])
               for trace, scope, cuts in definitions):
            # Replacing a complete calculation needs only a guarded kernel
            # call. There is no surrounding graph to rebuild or compile.
            self.log.append("installation", scope=definitions[0][1].address, method="direct")
            return emit_wrapper_variants(definitions, name)
        if not replay:
            from .bind.graph import emit_graph_wrapper_variants
            try:
                emitted = emit_graph_wrapper_variants(definitions, name)
                self.log.append("installation", scope=definitions[0][1].address, method="graph")
                return emitted
            except NotReplayable as error:
                self.log.append("graph_fallback", scope=definitions[0][1].address,
                                reason=str(error))
        self.log.append("installation", scope=definitions[0][1].address, method="replay")
        return emit_wrapper_variants(definitions, name)

    def _live_memory_scopes(self) -> None:
        """Kernels inside a block the model hands its cache to (an attention
        block, a decoder layer, a whole model) are delivered through a
        recording of that block, which fits only the history and cache
        positions it was recorded at and a few more: a second message, read
        on top of the first, gets none of them, and generation reaches a new
        position every token, so they would stop after a handful of tokens.
        For each such block the judge writes the block's own method calling
        the kernels, with position and cache code left live. It is installed
        only if the model then matches the untouched one bit for bit over
        first messages, second messages and whole decode runs, calls each
        kernel where its workload needs it (_live_runs_match), and passes
        review; otherwise the recording stays. Blocks of one class carrying the same
        kernels share the method the first of them got: they are checked
        together, and one at a time if that fails."""
        if self._generated():
            return  # its calls read generated tokens under exact_rows; a live method is checked on decoding
        if self.baseline_model is None:
            return
        groups: dict[tuple, list] = {}
        for path in sorted(self.installed):
            original, variants, kernels = self.installed[path]
            calls = {key: next(sc for sc in self.traces[key[0]].scope_calls if sc.address == key[1])
                     for key in variants}
            if not kernels or not any(sc.obj_ids for sc in calls.values()):
                continue
            # the judge needs to see which values feed each kernel, once: the
            # fewest recorded calls that hold every kernel, not every size (all
            # of gpt-oss's sizes were 119,802 lines, too long to send)
            chosen, left = [], {c.kernel.kernel_id for cuts in variants.values() for c in cuts}
            for key in sorted(variants, key=lambda k: -len({c.kernel.kernel_id for c in variants[k]})):
                if left & {c.kernel.kernel_id for c in variants[key]}:
                    chosen.append(key)
                    left -= {c.kernel.kernel_id for c in variants[key]}
            reference = emit_wrapper_variants([(self.traces[w], calls[(w, a)], variants[(w, a)])
                                               for w, a in chosen], f"R_{_safe(path)}")
            guarded = self._shape_guarded(kernels)
            groups.setdefault((type(original), tuple(sorted(guarded))), []).append(
                (path, original, guarded, _reference_excerpt(reference.source, guarded)))
        for members in groups.values():
            source = first_problem = None
            for i, (path, original, kernels, reference) in enumerate(members):
                if i > 0 and source is None:
                    # the same class code with the same kernels: asking again won't differ
                    self.log.append("live_method", scope=path, installed=False,
                                    reason=f"no method for {members[0][0]}, the same block class: {first_problem}")
                    continue
                if source is not None and i == 1 and len(members) > 1:
                    rest = [(p, o, k, self._live_emitted(p, k, source)) for p, o, k, _r in members[1:]]
                    if self._check_live(rest)[0] is None:
                        for p, _o, _k, emitted in rest:
                            self._live_installed(p, emitted)
                            self.log.append("live_method", scope=p, installed=True, reason=None)
                        break
                if source is not None and self._check_live(
                        [(path, original, kernels, self._live_emitted(path, kernels, source))])[0] is None:
                    self._live_installed(path, self._live_emitted(path, kernels, source))
                    self.log.append("live_method", scope=path, installed=True, reason=None)
                    continue
                reason, written = self._live_method(path, original, kernels, reference)
                source = written or source
                first_problem = first_problem or reason
                self.log.append("live_method", scope=path, installed=reason is None, reason=reason)

    def _live_emitted(self, path, kernels, source: str) -> EmittedWrapper:
        name = f"L_{_safe(path)}"
        return EmittedWrapper(name, path, re.sub(r"^class \w+\(", f"class {name}(", source, count=1, flags=re.M),
                              list(self.emitted[path].span_map), sorted(kernels))

    def _live_installed(self, path, emitted) -> None:
        self.emitted[path] = emitted
        self.live_scopes.add(path)
        self.log.append("installation", scope=path, method="live")

    def _shape_guarded(self, kernels: dict) -> dict:
        """The kernels, each declining by itself every input shape it was not
        verified at in an installed cut. Recorded wrappers guard shapes in
        their own code before calling a kernel; a method written for a block
        may not, so the kernel carries the guard."""
        verified: dict[str, list] = {}
        for _original, variants, _kernels in self.installed.values():
            for (w, _a), cuts in variants.items():
                for cut in cuts:
                    specs = self.traces[w].span_specs(cut.start_seq, cut.end_seq)
                    signature = [[list(specs[i][0]), str(specs[i][1]).removeprefix("mlx.core.")]
                                 for i in cut.input_ids]
                    if signature not in verified.setdefault(cut.kernel.kernel_id, []):
                        verified[cut.kernel.kernel_id].append(signature)
        return {kid: replace(k, input_signature=None, input_signatures=verified.get(kid, []))
                for kid, k in kernels.items()}

    def _live_method(self, path, original, kernels, reference: str):
        """Ask the judge for one block's method, check and install it:
        (None, its source) when installed, else (why not, None)."""
        from .judge import method_level
        module_source = _class_source(type(original), getattr(self, "model_module", None))
        if module_source is None:
            return "the block's source is not available", None
        judge = self.judge_factory(None)
        if not callable(getattr(judge, "_ask", None)):
            return "the judge cannot write a block's method", None
        name = f"L_{_safe(path)}"
        briefing = {"class_name": name, "module_source": module_source, "parts_source": _parts_source(original),
                    "reference": reference,
                    "kernels": {kid: {"inputs": list(k.input_names), "outputs": list(k.output_names)}
                                for kid, k in kernels.items()}}
        problem = None
        for _ in range(LIVE_ATTEMPTS):
            try:
                reply = method_level.propose(judge, briefing)
            except (ValueError, RuntimeError, OSError) as error:
                return f"no method: {error}"[:400], None
            emitted = EmittedWrapper(name, path, reply["source"], list(self.emitted[path].span_map),
                                     sorted(kernels))
            problem, previous = self._check_live([(path, original, kernels, emitted)])
            if problem is None:
                try:
                    verdict = method_level.review(judge, reply["source"], module_source)
                except (ValueError, RuntimeError, OSError) as error:
                    verdict = {"verdict": "reject", "reason": f"the review could not be completed: {error}"}
                if verdict["verdict"] == "accept":
                    self._live_installed(path, emitted)
                    return None, reply["source"]
                swap_install(self.model, path, previous[path])
                problem = f"rejected by review: {verdict['reason']}"[:400]
            briefing["previous_attempt"] = {"source": reply["source"], "problem": problem}
        return problem, None

    def _check_live(self, entries):
        """entries: (path, original block, kernels, emitted method). (problem,
        None) with every block restored, or (None, {path: previous occupant})
        with every method installed."""
        built, inside = [], []
        for path, original, kernels, emitted in entries:
            try:
                cls = _load_class(emitted)
                wrapper = cls(original, kernels)
            except Exception as error:  # the judge's source is checked here, never trusted
                return f"does not load: {type(error).__name__}: {error}"[:400], None

            def counted(cls=cls, path=path):
                class Counted(cls):
                    def __call__(self, *args, **kwargs):
                        inside.append(path)
                        try:
                            return super().__call__(*args, **kwargs)
                        finally:
                            inside.pop()
                return Counted
            built.append((path, original, kernels, wrapper, counted()(original, kernels)))
        previous = {path: swap_install(self.model, path, probe) for path, _o, _k, _w, probe in built}
        try:
            problem = self._live_runs_match({path: kernels for path, _o, kernels, _w, _p in built}, inside)
        except Exception as error:
            problem = f"raised {type(error).__name__}: {error}"[:400]
        for path, _o, _k, wrapper, _p in built:
            swap_install(self.model, path, previous[path] if problem is not None else wrapper)
        return (problem, None) if problem is not None else (None, previous)

    def _live_runs_match(self, kernels_by_path: dict, inside: list) -> str | None:
        """The patched model against the untouched one the way a model is
        used, three times from an empty cache: a first message, LIVE_STEPS
        decode calls, a second message read on top of that history, and one
        more decode call. Every output and the final caches match bit for
        bit. Each kernel found on a decode workload runs from its block on
        every decode call, unless its own shape rule declines those inputs.
        Each kernel found on a prompt workload is called from its block while
        every second message is read (a recording of the block declines
        there: it fits only the history it was recorded on). Every kernel
        runs somewhere in the check."""
        from backcountry_runtime import kernels as runtime_kernels
        from backcountry_runtime.state import _cache_observation, _make_cache
        patched = getattr(self.model, "model", self.model)
        untouched = getattr(self.baseline_model, "model", self.baseline_model)
        decode = self.context if self.context is not None and self.context.context else None
        prompt_workload = next((w for w in self.manifest.workloads if not w.context), None)
        if prompt_workload is not None:
            # the prompt the job searched, at its size, and two more like it
            seeds = workload_seeds(self.manifest.seed, f"{prompt_workload.name}:live", 2)
            messages = [self.tensors[prompt_workload.name][0]] + [
                self._materialize(prompt_workload, self.manifest.primary, s)[0] for s in seeds]
            messages = [m.reshape(1, -1) for m in messages]
        if decode is not None:  # the conversation the job used, and parts of it
            tokens, step = self.context_tokens.reshape(1, -1), self.tensors[decode.name][0]
            n = tokens.shape[1]
            firsts = [tokens[:, :max(1, n // 3)], tokens, tokens[:, n // 2:]]
            seconds = messages if prompt_workload is not None else firsts
        else:
            firsts = seconds = messages
            step = messages[0][:, -1:]
        found = {}  # kernel id -> the workloads the search installed it on, per block
        for path in kernels_by_path:
            for (w, _a), cuts in self.installed[path][1].items():
                for cut in cuts:
                    found.setdefault((path, cut.kernel.kernel_id), set()).add(w)
        called, ran, served = Counter(), Counter(), Counter()
        phase = ["first"]
        real = runtime_kernels.try_call

        def counting(spec, inputs, **kwargs):
            out = real(spec, inputs, **kwargs)
            if inside:
                key = (phase[0], inside[-1], spec.kernel_id)
                called[key] += 1
                ran[key] += out is not None
                # served: it ran, or its own shape rule turned these inputs down
                # (gpt-oss's attention-input kernel declines one-token reads)
                served[key] += out is not None or runtime_kernels._loaded(spec).launch(
                    inputs, checked_elsewhere=True).fallback
            return out

        def read(model, x, cache):
            """A message read as the job's step reads it: MLX-LM's generation
            reads all but its last token in chunks, then the last token."""
            if not self.use_library_inference or x.shape[1] < 2:
                return [model(x, cache=cache)]
            body, chunk = x[:, :-1], _prefill_chunk()
            return [model(body[:, i:i + chunk], cache=cache) for i in range(0, body.shape[1], chunk)] + \
                [model(x[:, -1:], cache=cache)]

        runtime_kernels.try_call = counting
        try:
            for first, second in zip(firsts, seconds):
                caches = _make_cache(untouched), _make_cache(patched)
                calls = [("first", first)] + [("decode", step)] * LIVE_STEPS + [("second", second), ("after", step)]
                for i, (phase[0], x) in enumerate(calls):
                    a = flatten_arrays(read(untouched, x, caches[0]))
                    b = flatten_arrays(read(patched, x, caches[1]))
                    mx.eval(a, b)
                    if len(a) != len(b) or not all(bitwise_equal(p, q) for p, q in zip(a, b)):
                        where = {"first": "reading a first message", "second": "reading a second message on top",
                                 "decode": f"decode call {i}", "after": "the decode call after a second message"}
                        return (f"outputs differ from the untouched model at {where[phase[0]]} "
                                f"({first.shape[1]}-token first message)")
                states = [flatten_arrays(_cache_observation(cache)) for cache in caches]
                if len(states[0]) != len(states[1]) or not all(
                        bitwise_equal(p, q) for p, q in zip(*states)):
                    return f"the cache differs from the untouched model's after a {first.shape[1]}-token first message"
        finally:
            runtime_kernels.try_call = real
        rounds = len(firsts)
        for path, kernels in kernels_by_path.items():
            for kid in sorted(kernels):
                workloads = found.get((path, kid), set())
                if decode is not None and decode.name in workloads:
                    need = LIVE_STEPS * rounds
                    if served[("decode", path, kid)] < need:
                        return f"kernel {kid} ran from {path} on {ran[('decode', path, kid)]} of {need} decode calls"
                if workloads - {decode.name if decode is not None else None}:
                    if called[("second", path, kid)] < rounds:
                        return (f"kernel {kid} was called from {path} on {called[('second', path, kid)]} of "
                                f"{rounds} second messages, read on top of the history")
                phases = ("first", "decode", "second", "after")
                if not sum(ran[(p, path, kid)] for p in phases):
                    calls = sum(called[(p, path, kid)] for p in phases)
                    return f"kernel {kid} ran from {path} on 0 of {calls} calls in the check"
        return None

    def _verify_installations(self, workload, tensors, states, cuts, baseline, *, graph_only=False):
        """Inspect rewritten graphs directly; legacy replay keeps its literal retrace."""
        from backcountry_runtime.graph import GraphBindingError
        graph = {p: _resolve(self.model, p) for p in states
                 if isinstance(_resolve(self.model, p), GraphWrapper)}
        # a kernel that leaves this workload's shapes to the library (the size
        # gate's stretch rule) is expected nowhere in it: the original ops run
        declined = {(s.start_seq, s.end_seq) for state in states.values()
                    for (w, _), ss in state[1].items() if w == workload for s in ss
                    if _declines(self.traces.get(workload), s)}
        graph_spans = set()
        expected = {}
        for path, wrapper in graph.items():
            splices = [s for (w, _), ss in states[path][1].items() if w == workload for s in ss]
            graph_spans.update((s.start_seq, s.end_seq) for s in splices)
            expected[path] = Counter(s.kernel.kernel_id for s in splices
                                     if (s.start_seq, s.end_seq) not in declined)
        if graph:
            self.tracer.uninstall()

            def hits(calls):
                actual = Counter()
                for call in calls:
                    for row in call:
                        if row["hits"]:
                            actual[row["kernel_id"]] += row["hits"]
                return actual

            with ExitStack() as stack:
                observations = {p: stack.enter_context(wrapper.validate_graph()) for p, wrapper in graph.items()}
                try:
                    self.session.off_clock(lambda: self._for(self.model, workload)(*tensors))
                except GraphBindingError as error:
                    raise NotReplayable(str(error)) from error
                for path, calls in observations.items():
                    if hits(calls) != expected[path]:
                        raise NotReplayable(f"graph scope {path}: replaced {dict(hits(calls))}, expected {dict(expected[path])}")
            # The deployed path: a compiled scope's first call traces the same
            # substitutions, or runs the original and says why. Calls with one
            # signature share one trace, so evidence here is per signature and
            # each entry must have found its own rule's cuts.
            for wrapper in graph.values():
                wrapper._graph_evidence.clear()
            self.session.off_clock(lambda: self._for(self.model, workload)(*tensors))
            for path, wrapper in graph.items():
                reasons = wrapper._graph_fallbacks + [wrapper._graph_fallback_reason]
                if any(reasons):
                    raise NotReplayable(f"graph scope {path} ran the original when compiled: "
                                        + "; ".join(r for r in reasons if r))
                short = [row for call in wrapper._graph_evidence for row in call if row["hits"] != row["expected"]]
                if short:
                    raise NotReplayable(f"graph scope {path} compiled: {short}")
            self.log.append("graph_verified", workload=workload,
                            scopes={p: dict(v) for p, v in expected.items()})
        remaining = {span: kid for span, kid in cuts.items() if span not in graph_spans and span not in declined}
        if not graph_only and (not graph or remaining):
            # Graph scopes are restored only for this check: the legacy log
            # describes Python calls, whereas graph insertion happens later.
            removed = []
            try:
                for path, wrapper in graph.items():
                    removed.append((path, swap_install(self.model, path, wrapper.wrapped)))
                self.tracer.uninstall()
                self.tracer.install()
                retrace, _ = self.tracer.trace(self._for(self.model, workload), tensors)
                spans = sorted(remaining)
                result = verify_retrace(baseline, retrace, spans, [remaining[s] for s in spans])
                if not result.ok:
                    raise NotReplayable("; ".join(result.reasons))
            finally:
                self.tracer.uninstall()
                for path, wrapper in reversed(removed):
                    swap_install(self.model, path, wrapper)

    def _holdout(self, region: Region, kernel: KernelSpec, result, assoc_tag: str) -> PromotionResult | None:
        """Over a range: the nominated kernel at HOLDOUT_SIZES sizes drawn
        fresh for this check (one per bin of the range, none a search size),
        each recorded from a real pass of the untouched model and checked and
        clocked as at the search sizes. Wrong there, or clearly slower than
        the library, it stops; it need not be faster. None: on to the
        whole-model check. A size the region does not run at is skipped."""
        self._attempt_step("the check at new sizes")
        named = self.report.constants.get("search_sizes") or {}
        if not named or not getattr(self.manifest, "serve", None):
            return None
        dim, span = next(iter(self.manifest.serve.items()))
        w = next((w for w in self.manifest.workloads if dim in w.named_dims()), None)
        rep = next((m for _label, m, _copies in capture_instances(region, self.traces)
                    if w is not None and m.workload == w.name), None)
        if span.listed or rep is None:
            return None  # a list is every size the model sees: nothing between to draw from
        # counted across resumes (saved): unsaved, each resumed Qwen3.8-27B search drew
        # 84, 329 and 1812 tokens again, sizes an earlier kernel had already been checked at
        self._holdout_draws = getattr(self, "_holdout_draws", 0) + 1
        sizes = [n for n in manifest_mod.search_sizes(span, None, self.manifest.seed + 1000 + self._holdout_draws,
                                                      count=HOLDOUT_SIZES) if n not in set(named[dim])]
        seed = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)[0]
        sets, signatures, was_installed = [], [], self.tracer.patcher.installed
        if not was_installed:
            self.tracer.install()
        try:
            for n in sizes:
                label = f"{w.name}@{dim}={n}:holdout"
                tensors = self._materialize(w, {**self.manifest.primary, dim: n}, seed)
                retrace, _ = self.tracer.trace(self._for(self.baseline_model, w.name), tensors)
                try:
                    at = locate_span(self.traces[w.name], rep, retrace, w.name)
                except SweepDivergence:
                    continue
                arrays = self.session.off_clock(lambda: capture_boundaries(
                    self.tracer, self._for(self.baseline_model, w.name), tensors, retrace,
                    set(at.input_ids) | set(at.output_ids)))
                self.store.save(region.fingerprint, label, 0, "inputs", {a: arrays[a] for a in at.input_ids},
                                retrace.weight_paths)
                self.store.save(region.fingerprint, label, 0, "outputs", {a: arrays[a] for a in at.output_ids})
                specs = retrace.span_specs(at.start_seq, at.end_seq)
                signatures.append([[list(specs[a][0]), specs[a][1]] for a in at.input_ids])
                sets.append(EvalSet(
                    label=label, correctness_only=False,
                    inputs_paths=[str(self.store._path(region.fingerprint, label, 0, "inputs"))],
                    reference_paths=[str(self.store._path(region.fingerprint, label, 0, "outputs"))],
                    nodes_json=nodes_to_json(retrace.nodes[at.start_seq:at.end_seq + 1])))
        finally:
            if not was_installed:
                self.tracer.uninstall()
        if not sets:
            return None
        job = self._ladder_job(region, kernel, assoc_tag, True, result.detail.get("target_workload"))
        job.eval_sets = [job.eval_sets[0], *sets]
        listed = kernel.input_signatures or ([kernel.input_signature] if kernel.input_signature else None)
        own = _loaded_kernel(kernel)._fallback
        declined = [label for label, sig in zip([es.label for es in sets], signatures)
                    if own is not None and own.evaluate([tuple(shape) for shape, _ in sig])]
        if declined:
            # where the kernel's own rule takes the library, served there it
            # is the library: correct and no slower, nothing to check
            keep = [i for i, es in enumerate(sets) if es.label not in declined]
            sets, signatures = [sets[i] for i in keep], [signatures[i] for i in keep]
            job.eval_sets = [job.eval_sets[0], *sets]
            self.log.append("holdout_declined", fingerprint=region.fingerprint, kernel=kernel.kernel_id,
                            sizes=[label.removesuffix(":holdout") for label in declined])
            if not sets:
                return None
        if listed is not None:
            # The recorded shapes bound where a kernel runs only until the
            # serve step checks it at a size (try_sized); left as recorded,
            # every fresh size fell back to the library and the static gate
            # failed the whole check. The kernel's own fallback rule still holds.
            wider = listed + [sig for sig in signatures if sig not in listed]
            job.kernel = replace(kernel, input_signature=None, input_signatures=wider)
            job.contract = replace(job.contract, input_signature=None, input_signatures=wider)
        check = run_ladder(job, session=self.session)
        rows = {label: row for label, row in check.detail.get("sizes", {}).items() if label.endswith(":holdout")}
        result.detail.setdefault("sizes", {}).update(rows)
        # a clock failure is a size too small to time, not a wrong answer
        wrong = (check.outcome == "failed" and check.failed_gate != "clock"
                 and str(check.detail.get("eval_set", "")).endswith(":holdout"))
        slower = sorted(label for label, row in rows.items() if row["loss"])
        self.log.append("holdout", fingerprint=region.fingerprint, kernel=kernel.kernel_id, sizes=sizes,
                        outcome=check.outcome, failed_gate=check.failed_gate, wrong=wrong, slower=slower)
        if wrong:
            return PromotionResult("correctness_failed", f"wrong at {check.detail['eval_set'].removesuffix(':holdout')}, "
                                   f"a size drawn fresh for this check ({check.failed_gate})")
        if check.outcome == "failed":
            # a failure anywhere in the check is a failure: a static or
            # compile failure says nothing passed at the fresh sizes
            reason = check.detail.get("reason") or check.detail.get("eval_set") or "no detail"
            status = "holdout_unmeasured" if check.failed_gate == "clock" else "correctness_failed"
            return PromotionResult(status, f"the check at sizes drawn fresh ({sizes}) failed "
                                   f"at its {check.failed_gate} gate: {reason}")
        if slower:
            return PromotionResult("holdout_slower", "clearly slower than the library at "
                                   + ", ".join(label.removesuffix(":holdout") for label in slower)
                                   + ", drawn fresh for this check")
        return None

    def _bind_and_promote(self, run: RegionRun, kernel: KernelSpec, result,
                          assoc_tag: str = "preserving", *, preflight=False) -> PromotionResult:
        """Bind the kernel into the live model and keep it only if the whole
        patched model is faster than what is already installed.

        The region clock nominates. A paired forward-pass comparison against
        the installed incumbent decides. Failed candidates are unwound.
        Preflight installs the original computation only to check graph cuts,
        then always unwinds without timing, promotion, or a checkpoint."""
        if not preflight and (result.outcome != "tentative_ship" or kernel.reference_sequence is not None):
            return PromotionResult("not_tested", "region did not nominate an installable replacement")
        self.session.wait_ready()
        region = run.region
        if not preflight and (held := self._holdout(region, kernel, result, assoc_tag)) is not None:
            return held
        self._attempt_step("the whole-model check")
        installed_now: list[tuple[str, object]] = []
        emitted_before: dict[str, object] = {}   # scope -> prior entry (None = absent)
        pending_installed: dict[str, tuple] = {}
        pending_removed: set[str] = set()
        try:
            self.tracer.patcher.install()
        except RuntimeError:
            pass  # already installed

        def unwind():
            """Every failure path must leave the model, the artifact record,
            and the patch surface exactly as before this attempt."""
            failures = []
            for path, occupant in reversed(installed_now):
                try:
                    swap_uninstall(self.model, path, occupant)
                except Exception as e:
                    self.log.append("rollback_error", scope=path, reason=str(e))
                    failures.append(f"{path}: {e}")
            for path, prior in emitted_before.items():
                if prior is None:
                    self.emitted.pop(path, None)
                else:
                    self.emitted[path] = prior
            try:
                self.tracer.uninstall()
            except Exception as e:
                failures.append(f"tracer cleanup: {e}")
            if failures:
                raise RuntimeError("rollback failed; stopping before further GPU work: " + "; ".join(failures))

        baseline_traces = dict(self.traces)
        try:
            changes = {}
            for m in region.members:
                trace = self.traces[m.workload]
                scope = find_scope_call(trace, m.scope_stack)
                if scope is None:
                    raise NotReplayable(f"no scope call for {m.scope_stack!r}")
                scope_path = scope.address.rsplit("@", 1)[0]
                if scope_path == "":
                    raise NotReplayable("root-scope delivery is not supported yet")

                splice = Splice(
                    kernel=kernel, start_seq=m.start_seq, end_seq=m.end_seq,
                    input_ids=m.input_ids, output_ids=m.output_ids,
                    fingerprint=region.fingerprint,
                )
                changes.setdefault(scope_path, []).append((m.workload, scope, splice))

            # A parent replay bypasses child calls. Compose all their cuts at
            # the outermost scope, keeping each workload's trace-local ids.
            updates = compose_scope_variants(
                self.traces, {path: state[1] for path, state in self.installed.items()}, changes)
            identities = {}
            use_replay = set(self.replay_scopes)
            for scope_path, variants in updates.items():
                if scope_path in self.certified_scopes:
                    continue
                definitions = [(self.traces[w], next(sc for sc in self.traces[w].scope_calls
                                if sc.address == address), []) for w, address in variants]
                emitted_id = self._emit_installation(definitions, f"Id_{_safe(scope_path)}",
                                                     replay=scope_path in use_replay)
                identities[scope_path] = _load_class(emitted_id)(_resolve(self.baseline_model, scope_path), {})
            if identities:
                from backcountry_runtime.state import correctness_call
                from backcountry_runtime.graph import GraphBindingError
                from .bind.certify import CertificationResult
                self.log.append("identity_certification", phase="start", scopes=len(identities))
                try:
                    check = self.session.off_clock(lambda: certify_identities(
                        model=self.baseline_model, wrappers=identities,
                        runs=[lambda t=self.tensors[w.name], m=self._for(self.baseline_model, w.name):
                              correctness_call(m, t) for w in self.manifest.workloads]), paced=False)
                except GraphBindingError as error:
                    check = CertificationResult(False, str(error))
                if check.ok:
                    # An identity that quietly ran the original proves nothing.
                    fell_back = [p for p, w in identities.items() if isinstance(w, GraphWrapper)
                                 and (w._graph_fallback_reason or w._graph_fallbacks)]
                    if fell_back:
                        check = CertificationResult(False, "graph identity ran the original module at "
                                                    + ", ".join(fell_back))
                self.log.append("identity_certification", phase="done", scopes=len(identities), passed=check.ok)
                if not check.ok:
                    # A scope whose identity changes under compilation can
                    # still use the previously validated replay mechanism.
                    failed_graph = {p for p, wrapper in identities.items() if isinstance(wrapper, GraphWrapper)}
                    if failed_graph:
                        use_replay.update(failed_graph)
                        self.replay_scopes.update(failed_graph)
                        self.log.append("graph_fallback", scopes=sorted(failed_graph), reason=check.reason)
                        for path in failed_graph:
                            definitions = [(self.traces[w], next(sc for sc in self.traces[w].scope_calls
                                            if sc.address == address), []) for w, address in updates[path]]
                            emitted_id = self._emit_installation(definitions, f"Id_{_safe(path)}", replay=True)
                            identities[path] = _load_class(emitted_id)(_resolve(self.baseline_model, path), {})
                        check = self.session.off_clock(lambda: certify_identities(
                            model=self.baseline_model, wrappers=identities,
                            runs=[lambda t=self.tensors[w.name], m=self._for(self.baseline_model, w.name):
                                  correctness_call(m, t) for w in self.manifest.workloads]), paced=False)
                    if not check.ok:
                        self.log.append("certification_failed", scopes=list(identities), reason=check.reason)
                        unwind()
                        return PromotionResult("binding_failed", check.reason)
                self.certified_scopes.update(identities)
                for path in identities:
                    if not isinstance(identities[path], GraphWrapper):
                        for workload, _address in updates[path]:
                            region.delivery[workload] = "replay"
            # The incumbent at every scope this attempt compiles is that
            # scope compiled with the cuts it carried before, none on a first
            # install: compilation's own effect on the step never counts for
            # or against the kernel, whatever the scope absorbs.
            # The arm every candidate is timed against is the confirmed
            # installation: a pending kernel's scope carries only its
            # confirmed cuts there, like a scope this attempt changes.
            pending_scopes = {path for path, state in self.installed.items()
                              if any(s.fingerprint in self.pending for cuts in state[1].values() for s in cuts)}
            carried = compose_scope_variants(
                self.traces, self.confirmed["variants"],
                {path: [] for path in {*changes, *updates, *pending_scopes}})
            whole_model_compiled = self.baseline == "compiled" and not self.use_library_inference

            def confirmed_at(scope_path, keys):
                """The scope's delivery carrying only the confirmed cuts it held."""
                kept = {key: list(carried.get(scope_path, {}).get(key, [])) for key in keys}
                prior_definitions = [
                    (self.traces[workload],
                     next(sc for sc in self.traces[workload].scope_calls if sc.address == address), cuts)
                    for (workload, address), cuts in kept.items()]
                return (self._emit_installation(prior_definitions, f"Inc_{_safe(scope_path)}",
                                                replay=scope_path in use_replay),
                        {s.kernel.kernel_id: s.kernel for cuts in kept.values() for s in cuts})
            candidates = {}
            for scope_path, variants in updates.items():
                prior = self.installed.get(scope_path)
                original = prior[0] if prior else _resolve(self.model, scope_path)
                # The combined replay owns these kernels now. Restore the
                # underlying children so live fallback, clones and exported
                # wrappers all see the same original module tree.
                descendants = [path for path in self.installed
                               if path.startswith(scope_path + ".")]
                for path in sorted(descendants, key=lambda p: p.count("."), reverse=True):
                    occupant = swap_install(self.model, path, self.installed[path][0])
                    installed_now.append((path, occupant))
                    emitted_before.setdefault(path, self.emitted.get(path))
                    self.emitted.pop(path, None)
                    pending_removed.add(path)
                definitions = [
                    (self.traces[workload],
                     next(sc for sc in self.traces[workload].scope_calls if sc.address == address), cuts)
                    for (workload, address), cuts in variants.items()]
                kernels = {s.kernel.kernel_id: s.kernel for cuts in variants.values() for s in cuts}
                emitted = self._emit_installation(definitions, f"W_{_safe(scope_path)}",
                                                  replay=scope_path in use_replay)
                cls = _load_class(emitted)
                occupant = swap_install(self.model, scope_path, cls(original, kernels))
                installed_now.append((scope_path, occupant))
                emitted_before.setdefault(scope_path, self.emitted.get(scope_path))
                self.emitted[scope_path] = emitted
                pending_installed[scope_path] = (original, variants, kernels)
                # The incumbent at this scope: the same delivery carrying only
                # the cuts it carried before, so what the wrapper itself does
                # to the step (compiling it; a replay parent bypassing its
                # compiled children) never counts for or against the kernel.
                # A bare kernel call replaces a plain module and is measured
                # against it; a whole-model compiled baseline needs nothing.
                if not whole_model_compiled and "direct" not in region.delivery.values():
                    candidates[scope_path] = confirmed_at(scope_path, variants)
            for scope_path in pending_scopes:
                holders = {s.fingerprint for cuts in self.installed[scope_path][1].values() for s in cuts}
                if (whole_model_compiled or any(scope_path == root or scope_path.startswith(root + ".")
                                                for root in updates)
                        or any("direct" in self.pending[fp]["region"].delivery.values()
                               for fp in holders & set(self.pending))):
                    continue
                candidates[scope_path] = confirmed_at(scope_path, self.installed[scope_path][1])
            incumbent = None if preflight else self._copy_incumbent(
                self.confirmed["emitted"], candidates, kernels=self.confirmed["kernels"])
            compiled_scopes = sorted(p for p, (e, _) in candidates.items() if issubclass(_load_class(e), GraphWrapper))

            # retrace: every installed cut, this one included, must be exactly
            # one custom dispatch per copy against the job-start recording
            pending_cuts: dict[str, dict[tuple[int, int], str]] = {}
            affected_workloads = {workload for variants in updates.values()
                                  for workload, _address in variants}
            self.session.wait_ready()  # wrapper construction used the identity check's cooling time
            for w in self.manifest.workloads:
                new = {(m.start_seq, m.end_seq): kernel.kernel_id
                       for m in region.members if m.workload == w.name}
                if w.name not in affected_workloads:
                    continue
                cuts = {**self.cuts.get(w.name, {}), **new}
                states = {p: state for p, state in self.installed.items() if p not in pending_removed}
                states.update(pending_installed)
                self._verify_installations(w.name, self.tensors[w.name], states, cuts, baseline_traces[w.name],
                                           graph_only=preflight)
                pending_cuts[w.name] = cuts

            if preflight:
                unwind()
                return PromotionResult("preflight_passed", "original graph cuts installed successfully")

            self.tracer.uninstall()
            sizes = self._model_check_sizes(region, result.detail)
            timed_arms = self._timed_arms(incumbent) if sizes is None else self._timed_arms(incumbent, names=sizes)
            exact = self._requires_exact(assoc_tag, region.fingerprint)
            reference = None if exact else self._reference_model(
                {**{p: state for p, state in self.installed.items() if p not in pending_removed},
                 **pending_installed}, pending=(region.fingerprint, assoc_tag))
            e2e = run_e2e(
                self.session, self.baseline_model, self.model, step_of=self._for,
                workloads=[(w.name, self.tensors[w.name]) for w in self.manifest.workloads],
                timed=timed_arms,
                veto_pairs=MODEL_CHECK_PAIRS if sizes else SHIP_PAIRS,
                defer_cooling=True,
                exact=exact,
                tolerances=self.manifest.tolerances,
                target_workload=None if sizes else result.detail.get("target_workload"),
                reference_model=reference,
                stop_at_loss=True,
                fits=lambda name: _fits(self, name),
                after_timing=self._timing_memory,
            )
            reference = None
            if sizes:
                e2e.required, e2e.min_wins = frozenset(sizes), MODEL_CHECK_WINS
            def commit():
                for path in pending_removed:
                    self.installed.pop(path)
                self.installed.update(pending_installed)
                self.cuts.update(pending_cuts)
                self.shipped_tags[region.fingerprint] = assoc_tag
                self.shipped_regions[region.fingerprint] = region

            if not self._model_win(e2e):
                veto = e2e.veto
                self.log.append("not_shipped", fingerprint=region.fingerprint,
                                reason="outputs changed" if not all(c.passed for c in e2e.checks)
                                else "not faster",
                                model_verdicts={name: (
                                    "resolved_regression" if value.loses_by(0.0) else
                                    "resolved_improvement" if value.wins_by(0.0) else "inconclusive")
                                    for name, value in e2e.workload_vetos.items()},
                                checks=[c.__dict__ for c in e2e.checks],
                                veto={"baseline_ms": veto.median_baseline_ms,
                                      "delta_ms": veto.median_delta_ms, "sigma_ms": veto.sigma_ms,
                                      "ratio": veto.median_ratio, "stability": veto.stability}
                                if veto else None)
                status = ("correctness_failed" if not all(c.passed for c in e2e.checks) else
                          "regression" if any(v.loses_by(0.0) for v in e2e.workload_vetos.values()) else "inconclusive")
                if status == "inconclusive":
                    # Too small to tell from noise alone: it stays installed and
                    # ships once a whole-model check of the set resolves a win.
                    commit()
                    self.pending[region.fingerprint] = {"region": region, "kernel": kernel.kernel_id,
                                                        "assoc_tag": assoc_tag}
                    self.log.append("pending", fingerprint=region.fingerprint, kernel=kernel.kernel_id,
                                    pending=sorted(p["kernel"] for p in self.pending.values()))
                    return PromotionResult(
                        "pending", "whole-model speedup unresolved: kept installed with the other pending "
                        "kernels, which ship together once a whole-model check resolves a win",
                        [asdict(c) for c in e2e.checks], {name: asdict(v) for name, v in e2e.workload_vetos.items()})
                unwind()
                return PromotionResult(status, "whole-model outputs changed" if status == "correctness_failed" else
                                       "whole-model regression" if status == "regression" else "whole-model speedup unresolved",
                                       [asdict(c) for c in e2e.checks],
                                       {name: asdict(v) for name, v in e2e.workload_vetos.items()})
            nomination_timings = {name: asdict(value) for name, value in e2e.workload_vetos.items()}
            e2e = self._confirm_model_win(e2e, incumbent, timed=timed_arms)
            if not self._model_win(e2e):
                self.log.append("not_shipped", fingerprint=region.fingerprint,
                                reason="whole-model win did not repeat in independent confirmation",
                                nomination_timings=nomination_timings,
                                confirmation_timings={name: asdict(value)
                                                      for name, value in e2e.workload_vetos.items()})
                unwind()
                return PromotionResult("confirmation_failed", "whole-model win did not repeat in independent confirmation",
                                       [asdict(c) for c in e2e.checks],
                                       {name: asdict(v) for name, v in e2e.workload_vetos.items()})
            model_ratios = {name: self.model_ratios.get(name, 1.0) * value.median_ratio
                            for name, value in e2e.workload_vetos.items()}
            accepted = dict(
                fingerprint=region.fingerprint, kernel=kernel.kernel_id, assoc_tag=assoc_tag,
                target_workload=e2e.target_workload,
                model_ratios=model_ratios,
                model_ratio=next(iter(model_ratios.values())) if len(model_ratios) == 1 else None,
                model_ratio_kind="product_of_incremental_measurements",
                timing_baseline=("incumbent_with_compiled_scopes" if compiled_scopes else
                                 "previously_installed_model" if self.installed else "original_model"),
                incumbent_compiled_scopes=compiled_scopes,
                model_latency_reduction_pct={
                    name: 100 * (1 - value.median_ratio)
                    for name, value in e2e.workload_vetos.items()},
                checks=[asdict(c) for c in e2e.checks],
                timings={name: asdict(value) for name, value in e2e.workload_vetos.items()},
                nomination_timings=nomination_timings,
                confirmation="fresh_fixed_size_comparison",
            )
            # Write the exact accepted wrappers before committing the live
            # state. No GPU work: even a later device failure cannot lose them.
            accepted["checkpoint"] = str(self._save_checkpoint(accepted))
        except BaseException as e:
            unwind()
            from backcountry_runtime.graph import GraphBindingError
            if not isinstance(e, (NotReplayable, GraphBindingError)):
                raise  # interrupts, device errors and harness bugs must stop the job
            self.log.append("bind_failed", fingerprint=region.fingerprint,
                            reason=f"{type(e).__name__}: {e}")
            return PromotionResult("binding_failed", str(e))
        self.model_ratio = accepted["model_ratio"]
        self.model_ratios = {**self.model_ratios, **accepted["model_ratios"]}  # sizes not timed keep theirs
        commit()
        # the win was timed with every pending kernel installed: they ship with it
        riders = [p for fp, p in self.pending.items() if fp != region.fingerprint]
        for p in riders:
            self.report.accepted.append({
                "fingerprint": p["region"].fingerprint, "kernel": p["kernel"], "assoc_tag": p["assoc_tag"],
                "confirmed_with": kernel.kernel_id, "model_ratios": accepted["model_ratios"],
                "checkpoint": accepted["checkpoint"]})
        accepted["confirmed_with"] = [p["kernel"] for p in riders]
        self.report.accepted.append(accepted)
        self.pending = {}
        self._snapshot_confirmed()
        self.log.append("shipped", **accepted)
        self.report.write(self.work_dir / "report.json")
        return PromotionResult("shipped", "whole-model correctness and independent timing confirmation passed",
                               accepted["checks"], accepted["timings"])

    def _snapshot_confirmed(self) -> None:
        """The installation as it stands, as the one every later check is timed against."""
        self.confirmed = {"emitted": dict(self.emitted),
                          "kernels": {path: state[2] for path, state in self.installed.items()},
                          "variants": {path: state[1] for path, state in self.installed.items()},
                          "cuts": {w: dict(spans) for w, spans in self.cuts.items()}}

    def _drop_pending(self) -> None:
        """When the search ends, kernels still pending go: only a resolved
        whole-model win ships. The model returns to the confirmed installation."""
        if not self.pending:
            return
        self.log.append("pending_dropped", kernels=sorted(p["kernel"] for p in self.pending.values()))
        for path in sorted(self.installed, key=lambda p: p.count("."), reverse=True):
            swap_install(self.model, path, self.installed[path][0])
        # a pending kernel can follow a shipped one in the same region (LFM2.5's
        # a483a790: h10 shipped, h14 pending); the shipped one stays, with its tag
        confirmed = {s.fingerprint for variants in self.confirmed["variants"].values()
                     for cuts in variants.values() for s in cuts}
        tags = {a["fingerprint"]: a["assoc_tag"] for a in self.report.accepted}  # the last ship per region
        for fp, p in self.pending.items():
            if fp in confirmed:
                self.shipped_tags[fp], self.shipped_regions[fp] = tags[fp], p["region"]
            else:
                self.shipped_tags.pop(fp, None)
                self.shipped_regions.pop(fp, None)
        self.pending = {}
        self.emitted = dict(self.confirmed["emitted"])
        self.cuts = {w: dict(spans) for w, spans in self.confirmed["cuts"].items()}
        self.installed = {}
        self._reinstall({path: (self.confirmed["variants"][path], self.confirmed["kernels"][path])
                         for path in self.emitted})
        self._check_unchanged("after dropping the pending kernels")

    def _save_checkpoint(self, accepted: dict) -> Path:
        """Durable recovery package; final timing and fresh-process export are pending."""
        # after the highest existing one: a job resumed after a crash may not know a
        # checkpoint saved after its last safe point, and must not overwrite it
        taken = [int(p.name.split("-")[1]) for p in (self.work_dir / "checkpoints").glob("accepted-[0-9][0-9][0-9][0-9]")]
        path = self.work_dir / "checkpoints" / f"accepted-{max([len(self.report.accepted), *taken]) + 1:04d}"
        snapshot = copy.deepcopy(self.report)
        snapshot.accepted.append({**accepted, "checkpoint": str(path)})
        snapshot.session.update(status="accepted_checkpoint", final_validation="pending",
                                artifact_validation="pending")
        specs = {}
        for scope in self.emitted:
            specs.update(getattr(_resolve(self.model, scope), "_specs", {}))
        bundle = self._model_bundle()
        # The live accepted-tag map is committed only AFTER this durable save.
        bundle.exact = self._requires_exact(accepted.get("assoc_tag"), accepted.get("fingerprint"))
        bundle.recovery = True  # final validation remains pending
        return emit_artifact(path, list(specs.values()), list(self.emitted.values()), snapshot,
                             bundle=bundle, layers=self._layer_signatures())

    def _model_win(self, e2e) -> bool:
        """Win on the target (or any final workload), with no resolved losses."""
        if not all(c.passed for c in e2e.checks) or e2e.veto is None:
            return False
        comparisons = e2e.workload_vetos
        required = getattr(e2e, "required", None)
        if required is None and hasattr(self, "manifest"):
            required = {w.name for w in self.manifest.workloads if _fits(self, w.name)}
        if required is not None and not set(required) <= set(comparisons):
            return False  # early target rejection or incomplete timing is never a ship
        # sizes timed beyond the required ones (one measured before it stopped
        # fitting, a size of the range) still count
        share = getattr(e2e, "margin_share", 0.0)
        wins = sum(c.wins_by(share * c.median_baseline_ms) for c in comparisons.values())
        return workload_win(comparisons or {"main": e2e.veto}, getattr(e2e, "target_workload", None), share) \
            and wins >= getattr(e2e, "min_wins", 1)

    def _confirm_model_win(self, nomination, incumbent, *, timed=None):
        """One independent, fixed-size confirmation, never retry-until-pass."""
        target = getattr(nomination, "target_workload", None)
        if target is None:
            target = next((name for name, value in sorted(getattr(nomination, "workload_vetos", {}).items())
                           if value.wins_by(0.0)), None)
        required = getattr(nomination, "required", None)
        if required is not None:  # over a range: the same sizes and rule, no single target
            target = None
            required = frozenset(name for name in required if _fits(self, name))
        pairs = MODEL_CHECK_PAIRS if required is not None else SHIP_PAIRS
        self.log.append("model_confirmation", phase="start", pairs=pairs, target_workload=target)
        arms = timed if timed is not None else \
            self._timed_arms(incumbent) if required is None else self._timed_arms(incumbent, names=required)
        timings = {}
        for name in sorted(arms, key=lambda name: (name != target, name)):
            if not _fits(self, name):
                continue
            timings[name] = compare(self.session, *arms[name], pairs=pairs, defer_cooling=True)
            self._timing_memory(name)
            if name == target and not timings[name].wins_by(0.0):
                break
        result = E2EResult(checks=nomination.checks,
                          veto=next(iter(timings.values()), None), workload_vetos=timings,
                          target_workload=target, required=required,
                          min_wins=getattr(nomination, "min_wins", 1))
        self.log.append("model_confirmation", phase="done", passed=self._model_win(result),
                        timings={name: asdict(value) for name, value in timings.items()})
        return result

    # -- the whole job --------------------------------------------------------

    def check(self) -> Report:
        """`backcountry check`: the job's own setup up to its room estimate, then
        stop. No judge call, no attempt: load and record the model, find its
        regions, measure what fits in memory, clock each step, measure the chip
        and say how much of each step no kernel can touch. The same functions
        the job runs, so check and job agree."""
        self.report.session["status"] = "checking"
        self._phase("loading model")
        self.load_model()
        self._phase("recording the model")
        self.trace_workloads()
        self._same_work_again()
        regions = self.build_regions()
        self.tracer.uninstall()
        mx.clear_cache()
        self._measure_repeatability()
        self._same_output_again()
        self._phase("measuring memory, speed and the chip")
        self._memory_fit()
        self._clock_steps()
        self.measure_machine()
        self._record_step_floor()
        self.report.coverage["regions_found"] = len(regions)
        self.report.constants["pacing"] = self._pacing()
        self.report.session["status"] = "checked"
        self.report.write(self.work_dir / "report.json")
        return self.report

    def _same_work_again(self) -> None:
        """The search replays each size many times and needs the same work every
        time; something the app keeps between calls (an engine's prompt cache)
        changes it, and the kernel search stopped on it mid-job. Record the
        first size again and say so here, in minutes."""
        w = self.manifest.workloads[0]
        again, _ = self.tracer.trace(self._for(self.model, w.name), self.tensors[w.name])
        first = len(self.traces[w.name].nodes)
        if len(again.nodes) != first:
            raise CheckError(
                f"the model did different work the second time it got the same request ({first} recorded "
                f"calls, then {len(again.nodes)}): the app keeps something between calls that changes the work, "
                "such as a serving engine's prompt cache (a request it has just read is not read again) or a "
                "conversation the marked function continues (a method on an object holding the chat's cache). "
                "The search replays each request many times and needs the same work each time: run the app with "
                "that cache off (each request read in full), or mark a function that starts from fresh state "
                "on every call (one that takes the whole conversation), then check again")

    def _measure_repeatability(self) -> None:
        """Two calls of the first size, before anything is installed: whether the
        untouched model gives its outputs bit for bit (report.constants
        output_repeats). Kokoro (mlx-audio) gives the same audio to within 1e-7,
        not bit for bit, even run plain: under 'tolerance: default' such a job
        grades every edit within its tolerance (_requires_exact; the user's call,
        2026-11-15), since there are no exact bits to preserve. Run once a job,
        after the model loads."""
        from backcountry_runtime.state import correctness_call
        if not self.manifest.workloads or getattr(self, "model", None) is None:
            return  # nothing to call (a job assembled in parts, as tests do)
        w = self.manifest.workloads[0]
        if w.name not in self.tensors:
            self.tensors[w.name] = self._materialize(
                w, self.manifest.primary, workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)[0])
        first = correctness_call(self._for(self.model, w.name), self.tensors[w.name])
        names = list(getattr(self._request(), "_names", []))
        second = correctness_call(self._for(self.model, w.name), self.tensors[w.name])
        repeats = self._outputs_check(w.name + ":again", lambda: first, lambda: second, exact=True).passed
        self.report.constants["output_repeats"] = repeats
        self._repeat_pair = (first, second, names)
        if not repeats:
            self.log.append("output_does_not_repeat", tolerance_allowed=self.manifest.tolerance_allowed)

    def _request(self):
        """The marked function's Request: the model itself, or held by the run
        that may rewrite it (keep: same-output). None for a model file."""
        from backcountry_runtime.tune import Request
        return next((m for m in (self.model, getattr(self.model, "model", None)) if isinstance(m, Request)), None)

    def _same_output_again(self) -> None:
        """A marked function's result must be the same for the same request:
        one that returns its speed with its answer (mlx-vlm's GenerationResult,
        a tokens-per-second figure in a dict) never matches itself, and every
        candidate would fail hours into the run. The two calls of
        _measure_repeatability, compared as candidates are; the parts that
        differ are named."""
        request = self._request()
        if request is None or self.report.constants.get("output_repeats", True):
            return
        first, second, names = self._repeat_pair
        pairs = [(name, a, b) for name, a, b in zip(names, first, second)
                 if a.shape != b.shape or not mx.array_equal(a, b).item()]
        differ = [name for name, _a, _b in pairs] or ["its result"]
        shown = ", ".join(differ[:5]) + (f" and {len(differ) - 5} more" if len(differ) > 5 else "")
        rounding = pairs and all(a.ndim and a.shape == b.shape and mx.issubdtype(a.dtype, mx.floating)
                                 for _n, a, b in pairs)
        if rounding:
            largest = max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() for _n, a, b in pairs)
            w = self.manifest.workloads[0]
            within = self.manifest.tolerance_allowed and self._outputs_check(
                w.name + ":again", lambda: first, lambda: second, exact=False).passed
            if within:
                print(f"note: {request._fn.__qualname__} gives the same result to within {largest:.2g} but not bit "
                      "for bit; every kernel is graded within the tolerance (tolerance: default)", flush=True)
                return
            if not self.manifest.tolerance_allowed:
                raise CheckError(
                    f"{request._fn.__qualname__} gives the same result to within {largest:.2g} but not bit for bit "
                    f"when given the same request twice ({shown}): the library's GPU math does not repeat exactly, "
                    "so 'tolerance: exact' (identical outputs) can never be met. Set 'tolerance: default' in "
                    "backcountry.yaml (each kernel graded within each output type's rounding limit), then check again")
        raise CheckError(
            f"{request._fn.__qualname__} returned something different the second time it got the same "
            f"request ({shown}): outputs are compared to decide what ships, so it must return only what the "
            "request produces. A speed or memory figure in its result (a library's result object, a stats "
            "dict) changes every run: return the answer itself (its text, arrays or image), and keep the "
            "figures out of what it returns")

    def run(self) -> Report:
        """Keep a readable partial result even if preparation or search fails."""
        self.report.session["status"] = "running"
        self.report.write(self.work_dir / "report.json")
        try:
            report = self._run()
            report.session["status"] = "search_complete"
            report.constants["pacing"] = self._pacing()  # the peak temperature the whole job reached
            return report
        except resume_mod.Paused as pause:
            # state was saved at the safe point this stopped at
            self.report.session.update(status="paused", reason=str(pause), error=None)
            self.log.append("paused", reason=str(pause), stage=self.report.session.get("stage"))
            print(f"paused: {pause}. Resume with: backcountry resume --work-dir {self.work_dir}", flush=True)
            return self.report
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            detail = f"{type(error).__name__}: {error}"
            self.report.session.update(status=status, error=detail)
            # where it died: a Metal error names no call, and Qwen3.8-27B's two
            # GPU timeouts left only this message to go on
            stack = "".join(traceback.format_exception(error))[-6000:]
            self.log.append("job_failed", stage=self.report.session.get("stage"), reason=detail, stack=stack)
            raise
        finally:
            self.tracer.uninstall()
            self.report.write(self.work_dir / "report.json")

    def _phase(self, stage: str) -> None:
        self.report.session["stage"] = stage
        self.log.append("stage", name=stage)
        self.session.memory(f"stage: {stage}")
        self.report.write(self.work_dir / "report.json")
        print(stage, flush=True)

    def _run(self) -> Report:
        self.log.append(
            "job", manifest=self.report.manifest_path,
            workloads={w.name: [[str(d) for d in i.shape] for i in w.inputs]
                       for w in self.manifest.workloads},
            budget_per_region=self.manifest.budget_per_region,
            budget_total=self.manifest.budget_total,
            defaulted=list(self.manifest.defaulted),
            gpu_utilization_pct=self.gpu_busy_at_start,
        )
        if getattr(self, "_resume_state", None) is not None:
            self._restore()
            if self.final_ok:  # saved after its final passed: only the export is left
                return self.report
        elif not self.manifest.budget_total:  # the inference-loop search alone: no kernel search
            self._phase("loading model")
            self.load_model()
            for w in self.manifest.workloads:  # what the artifact's load check runs
                self.tensors[w.name] = self._materialize(w, self.manifest.primary,
                                                         workload_seeds(self.manifest.seed, w.name, 1)[0])
            self.tracer.uninstall()  # nothing is traced here, and every timing runs without the recorder
            self._measure_repeatability()
            if self._finished_loop:  # its final passed: only the export is left, checked again there
                self.report.model_level = {k: v for k, v in self._finished_loop.items() if k != "not_shipped"}
                self.log.append("resumed", at="after the loop search", final=self._finished_loop["final"])
                return self.report
            return self._model_search()
        else:
            self._phase("loading model")
            self.load_model()
            self._measure_repeatability()
            if getattr(self.manifest, "identical", "computation") == "output":
                return self._loop_then_kernels()
            self._phase("tracing workloads and finding legal regions")
            self.trace_workloads()
            regions = self.build_regions()
            self._phase("measuring baseline and selecting regions")
            ranked = self.capture_and_price(regions)
            self._search_state = {"ranked": ranked, "shipped": [], "run": None, "accepted_before": 0}
        self._phase("searching kernels")
        self.report.write(self.work_dir / "report.json")  # partial: peaks, coverage, strands
        self._search()
        self._drop_pending()
        self._live_memory_scopes()
        ranked = self._search_state["ranked"]

        selection = self.report.coverage.setdefault("selection", {})
        selection["unsearched"] = len(ranked) + len(self.pending_regions)

        # The headline is a ratio of two clocks taken together: the
        # untouched model is re-measured beside the patched one (_final_check),
        # never subtracted from the job-start clock taken on a different machine
        # state. "before" stays in the report as the job-start observation it is.
        # With nothing installed there is nothing to time: the model against
        # itself took 4 hours on Qwen3.8-27B to read 1.00x at ten sizes.
        # the search's time on the job's clock; the checks and export come after
        self.report.session["search_s"] = round((getattr(self, "parent_clock", None) or self._active_s)(), 1)
        self._phase("checking final model timings")
        self._final_with_serving((self._serve_sizes() or []) + self._serve_history_lengths())
        self._clock_against_other_baseline()
        self.report.session["idled_s"] = round(self.session.idled_s, 1)
        self.report.write(self.work_dir / "report.json")
        if getattr(self, "final_ok", False):  # a passed final is never run again: a failed export resumes at the export
            self._safe_point("after the final check")
        return self.report

    def _final_with_serving(self, served: list) -> None:
        """The final check with the kernels served at every size checked. Where
        it finds changed outputs, or a loss, at up to MAX_CUT_LENGTHS lengths,
        those lengths and out to the nearest checked ones that held go back to
        the original, and the whole check runs again; past that, or if it
        fails again, only the searched sizes are served."""
        excluded = []
        try:
            self._final_check()
        except RuntimeError as error:
            if not served:
                raise
            excluded = self._wrong_spans()
            self._unserve(served, f"the final check failed with size-generic delivery: {error}"
                          + (f"; served again without {excluded}" if excluded else ""))
            served = self._install_served(*self._serve_plan, excluded=excluded) if excluded else None
            if served:
                try:
                    self._final_check()
                except RuntimeError as again:
                    self._unserve(served, f"the final check failed again without {excluded}: {again}")
                    served = None
            if not served:
                self._final_check()
        if served and not self.final_ok and (cut := self._losing_spans()):
            # a few lengths lost: serve everywhere else (the checks already ran), and check again
            excluded = excluded + cut
            self._unserve(served, f"lost at {self.report.final['lost']}: served again without {excluded}")
            served = self._install_served(*self._serve_plan, excluded=excluded)
            if served:
                self._final_check()
        if served and not self.final_ok:
            self._unserve(served, self.report.final.get("reason", "the final check did not confirm a win"))
            self._final_check()

    def _loop_then_kernels(self) -> Report:
        """identical: output with a kernel budget. The loop search runs first,
        on the workload as written. The kernel search then runs as an ordinary
        job (kernels/) on kernel_manifest.yaml: this manifest with each
        workload set to the calls the winning loop makes (the same workload
        when it calls the model as before). Last, on the kernel job's models,
        the winning loop on the patched model is timed against the default on
        the untouched one, and against itself on the untouched one (what the
        kernels add under it); its bundle carries the kernels and run.py.
        When no kernel is confirmed, or that final fails, the bundle carries
        run.py on the untouched model with the loop's own final
        (model_level.kernels_dropped says why)."""
        from backcountry_runtime.model_run import load_loop
        from .model_search import ModelSearch, kernel_manifest
        for w in self.manifest.workloads:  # what the artifact's load check runs
            self.tensors[w.name] = self._materialize(w, self.manifest.primary,
                                                     workload_seeds(self.manifest.seed, w.name, 1)[0])
        self.tracer.uninstall()  # the loop search times the model: the recorder must be gone
        if getattr(self, "_finished_loop", None):
            self.report.model_level = self._finished_loop
            self.log.append("resumed", at="after the loop search", final=self._finished_loop["final"])
        else:
            self._model_search()
        level = self.report.model_level
        text, changed = kernel_manifest(self.report.manifest_path, self.manifest.model_path, level)
        path = self.work_dir / "kernel_manifest.yaml"
        path.write_text(text)
        self.log.append("kernel_manifest", path=str(path), changed=changed)
        if level.get("source") and (level.get("final") or {}).get("ok") and self._finish_requested():
            # finish asked before the kernel search began: the loop ships alone, exported as a loop-only
            # job's (Qwen3.8-27B's kernel search ran out of GPU memory in its setup at every resume)
            self.report.model_level = {**level, "kernels_dropped": "backcountry finish was requested before the kernel search"}
            return self.report
        # the kernel job builds its own two copies of the model: drop ours first
        self.tracer.uninstall()
        self.model = self._pristine = self.baseline_model = self._compiled_baseline = None
        self.tensors = {}
        gc.collect()
        mx.clear_cache()
        self._phase("searching kernels on the loop's calls")
        # a resumed job continues a kernel search it saved, whether or not its loop search kept a loop
        kernel_dir = self.work_dir / "kernels"
        resume_kernels = getattr(self, "_resumed", False) and (kernel_dir / resume_mod.STATE).is_file()
        if not resume_kernels and (kernel_dir / "run.jsonl").exists():
            # one that died before its first save (Qwen3.8-27B ran out of GPU memory in its
            # setup) starts over; its logs are kept beside
            kernel_dir.rename(kernel_dir.with_name(f"kernels-unsaved-{time.strftime('%Y%m%d-%H%M%S')}"))
        kernels = JobRunner(path, kernel_dir, judge_factory=self.judge_factory,
                            session=self.session, clock_pairs=self.clock_pairs, resume=resume_kernels)
        kernels.parent_dir = self.work_dir  # the operator's requests are written here
        saved = (kernels._resume_state or {}).get("report")
        if saved is not None and saved.session.get("job_active_s", 0.0) > self._active_s():
            # resumed: the job's clock goes on from where its kernel search saved it, not from zero
            self._clock_base = (saved.session["job_active_s"], time.monotonic())
        kernels.parent_clock = self._active_s  # and its time is counted on it
        report = kernels.run()
        if report.session.get("status") == "paused":  # its state is saved in kernels/: the job resumes there
            raise resume_mod.Paused(report.session.get("reason") or "the kernel search paused")
        if level.get("source"):
            alone, final = level.get("final"), None
            if kernels.final_ok and kernels.emitted:
                self._phase("timing the loop with the kernels")
                final = ModelSearch(kernels, self.manifest, check_against=kernels.model).final(
                    load_loop(level["source"], "winning_run"))
            level = {**level, "loop_alone": alone, "final": final}
            if not (final or {}).get("ok"):
                # the loop's own confirmed result stands; no unconfirmed kernel ships under it
                why = ("no kernel was confirmed" if final is None
                       else f"the loop with the kernels failed its final: {final.get('reason')}")
                level.update(final=alone, kernels_dropped=why, final_with_kernels=final)
                kernels.emitted, kernels.installed, kernels.live_scopes = {}, {}, set()
                # the outputs run.py must reproduce are the loop search's own
                shutil.copy(self.work_dir / "model_level_check.safetensors", kernels.work_dir)
        report.model_level = level
        report.session["job_active_s"] = round(self._active_s(), 1)  # the loop search's time and the kernels'
        report.write(kernels.work_dir / "report.json")
        report.write(self.work_dir / "report.json")
        self._kernel_job, self.final_ok, self.report = kernels, kernels.final_ok, report
        return report

    def _model_search(self) -> Report:
        """identical: output: search the inference loop on the model as the kernel
        search left it (see model_search.py)."""
        from .model_search import ModelSearch
        self._phase("searching the inference loop")
        ModelSearch(self).run()
        self.report.session["idled_s"] = round(self.session.idled_s, 1)
        self.report.write(self.work_dir / "report.json")
        return self.report

    def _search(self) -> None:
        """Open ranked regions until the budget or the queue runs out, from the
        saved position: a region the job stopped inside is finished first."""
        state = self._search_state
        if state["run"] is not None:
            run = state["run"]
            run.errors = 0  # a fresh process asks a reachable judge again
            self.log.append("region_resumed", fingerprint=run.region.fingerprint,
                            attempts=run.hypotheses, attempts_left_job=self.manifest.budget_total - self.total_hypotheses)
            self.hypothesis_cycle(run, self.judge_factory(run.region))
            if not self._close_region(run):
                return
        while state["ranked"] or self.pending_regions:
            self._safe_point("between regions")
            if self._finish_requested():
                break
            if self.total_hypotheses >= self.manifest.budget_total:
                break
            if not state["ranked"]:
                state["ranked"] = self._next_regions()
                if not state["ranked"]:
                    break
            region = state["ranked"].pop(0)
            free = free_members(region, [r for r in state["shipped"] if r is not region])
            if not free:
                self.log.append("region_covered", fingerprint=region.fingerprint,
                                reason="every copy touches a cut already shipped")
                continue
            if len(free) < region.copies:
                # a shipped bigger cut owns the other copies; this region goes
                # on at the copies still free, priced at that count
                self.log.append("region_trimmed", fingerprint=region.fingerprint,
                                copies_before=region.copies, copies_free=len(free))
                region.members = free
                for w in region.workloads:
                    n = sum(m.workload == w for m in free)
                    region.t_orig_ms[w] = region.t_rep_ms.get(w, 0.0) * n
                    region.p[w] = region.p_rep.get(w, 0.0) * n
            state["accepted_before"] = len(self.report.accepted)
            # a skip names the region that was open when it was asked for
            for d in self._request_dirs():
                (d / resume_mod.SKIP_REQUEST).unlink(missing_ok=True)
            roof = region.roofline
            self.log.append(
                "region_open", fingerprint=region.fingerprint, ops=list(region.ops),
                copies=region.copies, p=dict(region.p),
                bound=roof.bound if roof else None, s_max=roof.s_max if roof else None,
                roofline_ms=roof.t_roofline_ms if roof else None,
                t_orig_ms=dict(region.t_orig_ms), t_rep_ms=dict(region.t_rep_ms),
            )
            self.candidates.append(_region_line(region, "open"))
            judge = self.judge_factory(region)
            run = self.open_region(region, judge)
            state["run"] = run
            if run.scaffold is not None and run.close_rule is None:
                self.hypothesis_cycle(run, judge)
            if not self._close_region(run):
                break

    def _close_region(self, run) -> bool:
        """Record a finished region and re-rank what is left. False when the
        job should stop searching (finish requested, budget spent)."""
        state = self._search_state
        region = run.region
        # a pending kernel occupies its cuts like a shipped one
        won = run.shipped is not None or region.fingerprint in self.pending
        if won and region not in state["shipped"]:
            state["shipped"].append(region)
        self._record_region_closed(run)
        state["run"] = None
        if not won:
            # no win here: its near-copies wait behind everything not yet tried
            moved = demote_similar(region, state["ranked"] + self.pending_regions)
            if moved:
                self.log.append("similar_demoted", fingerprint=region.fingerprint, regions=len(moved),
                                shared_ops_over=SIMILAR_OPS)
        if self._finish_requested() or self.total_hypotheses >= self.manifest.budget_total:
            return False
        if len(self.report.accepted) > state["accepted_before"]:
            refreshed = self._until_finish(lambda: self._refresh_after_ship(state["ranked"], state["shipped"]),
                                           "re-pricing after a ship")
            if refreshed is None:
                return False
            state["ranked"] = refreshed
        # The completed region releases unshipped overlapping alternatives.
        # Remaining ranked targets block overlaps until their own search ends.
        state["ranked"] = rank(state["ranked"] + self._next_regions(blockers=state["ranked"]))
        return True

    def _safe_point(self, where: str) -> None:
        """No GPU work in flight and every verdict recorded: save the state a
        resumed job continues from, and stop here if a pause was asked for."""
        if not getattr(self, "_resumable", False):
            return  # a runner assembled without the constructor (a test's partial runner)
        self.report.session["active_s"] = round(self._active_s(), 1)  # a resumed job's clock goes on from here
        if parent := getattr(self, "parent_clock", None):  # and the job's
            self.report.session["job_active_s"] = round(parent(), 1)
        resume_mod.save(self, where)
        if any(resume_mod.pause_requested(d) for d in self._request_dirs()):
            raise resume_mod.Paused(f"pause requested ({where})")

    def _attempt_step(self, step: str) -> None:
        """The step the running kernel attempt is on, should the GPU fail under it."""
        if getattr(self, "_resumable", False):
            resume_mod.attempt_step(self.work_dir, step)

    def _restore(self) -> None:
        """Rebuild what a new process cannot load (the model, its recording,
        the installed wrappers) and restore the rest from the saved state."""
        state = self._resume_state
        self._phase("resuming: loading model")
        self.load_model()
        self._phase("resuming: recording the model")
        self.trace_workloads()
        recorded = {r.fingerprint: r for r in self.build_regions()}
        self.tracer.uninstall()
        self._measure_repeatability()
        search = state["search"]
        saved = list(state["pending_regions"]) + list(search["ranked"]) + list(search["shipped"]) \
            + list(state["shipped_regions"].values()) + ([search["run"].region] if search["run"] else [])
        for region in saved:
            fresh = recorded.get(region.fingerprint)
            spans = {(m.workload, m.start_seq, m.end_seq) for m in fresh.members} if fresh else set()
            if not spans >= {(m.workload, m.start_seq, m.end_seq) for m in region.members}:
                raise RuntimeError(f"cannot resume: the model no longer records region {region.fingerprint} "
                                   "as it was saved; the model, manifest or tool changed")
        for name in resume_mod.SAVED:
            if name in state:
                setattr(self, name, state[name])
        self._clock_base = (self.report.session.get("active_s", 0.0), time.monotonic())
        self._search_state = search
        self._replay = resume_mod.failed_attempt(self.work_dir)  # run again, or dropped, at its attempt
        if state["compiled_baseline"]:
            outermost = {e.scope_path: (e, {}) for e in self.baseline_wrappers}
            self._compiled_baseline = self._copy_incumbent({}, outermost)
        self._reinstall(state["installed"])
        if "confirmed" not in state:  # saved before pending kernels existed: all installed was confirmed
            self._snapshot_confirmed()
        self._check_unchanged("cannot resume")
        # a job priced before the fit was measured keeps the sizes its regions were priced at
        self.report.constants.setdefault("priced_workloads", sorted(self._priced_workloads()))
        self._memory_fit()
        # this process rests by its own heat readings; the saved report has the last one's
        before = self.report.constants.get("pacing")
        self.report.constants["pacing"] = self._pacing()
        self.report.session.setdefault("resumed", []).append(
            {"at": state["where"], "attempts_used": self.total_hypotheses, "installed": sorted(self.emitted),
             "pacing_before": before})
        self.log.append("resumed", at=state["where"], attempts_used=self.total_hypotheses,
                        installed=sorted(self.emitted), accepted=len(self.report.accepted))

    def _reinstall(self, saved: dict) -> None:
        """Install every emitted wrapper on the live model with its saved
        (variants, kernels)."""
        for path, emitted in self.emitted.items():
            variants, kernels = saved[path]
            original = _resolve(self.model, path)
            swap_install(self.model, path, _load_class(emitted)(original, kernels))
            self.installed[path] = (original, variants, kernels)
        mx.clear_cache()

    def _check_unchanged(self, when: str) -> None:
        """The live model's outputs still match the untouched model's."""
        from backcountry_runtime.state import correctness_call
        reference = self._reference_model()
        for w in self.manifest.workloads:
            tensors = self.tensors[w.name]
            check = self.session.off_clock(lambda: self._outputs_check(
                w.name + ":" + when,
                lambda: correctness_call(self._for(self.baseline_model, w.name), tensors),
                lambda: correctness_call(self._for(self.model, w.name), tensors),
                None if reference is None else lambda: correctness_call(self._for(reference, w.name), tensors)))
            if not check.passed:
                raise RuntimeError(f"{when}: the model's outputs differ from the "
                                   f"untouched model's on {w.name}: {check.reason}")

    def _record_region_closed(self, run):
        self.report.add_region(
            fingerprint=run.region.fingerprint, ops=list(run.region.ops),
            copies=run.region.copies, workloads=list(run.region.workloads),
            p=dict(run.region.p), t_orig_ms=dict(run.region.t_orig_ms),
            t_rep_ms=dict(run.region.t_rep_ms),
            roofline_ms=run.region.roofline.t_roofline_ms if run.region.roofline else None,
            bound=run.region.roofline.bound if run.region.roofline else None, s_max=run.region.roofline.s_max if run.region.roofline else None,
            t_shipped_ms={run.shipped_workload: run.shipped_ms}
                if run.shipped_ms is not None and run.shipped_workload is not None else None,
            speedup=(1.0 / run.shipped_ratio) if run.shipped_ratio else None,
            close_rule=run.close_rule, hypotheses=run.hypotheses, head_ms=run.head_ms,
            timing_workload=run.shipped_workload,
            delivery=dict(run.region.delivery),
            library_arm={w: run.region.library_arm(w, self.baseline) for w in run.region.delivery},
        )
        self.log.append("region_closed", fingerprint=run.region.fingerprint,
                        rule=run.close_rule, shipped=run.shipped is not None,
                        hypotheses=run.hypotheses, head_ms=run.head_ms,
                        shipped_ms=run.shipped_ms,
                        head_workload=run.head_workload, shipped_workload=run.shipped_workload,
                        outcomes=self.report.regions[-1]["outcomes"])
        self.candidates.append(_region_line(run.region, "closed", run))
        self.report.write(self.work_dir / "report.json")  # a crash still leaves the story so far


    def _clock_against_other_baseline(self) -> None:
        """The finished model against the baseline the job did not ship against,
        paired and interleaved at the primary size, so every report says where
        the model stands against both eager MLX and mx.compile. Shipping was decided against
        self.baseline; this only records, measured rather than derived from
        step clocks taken minutes apart. A model that keeps Python state cannot
        be compiled from outside, so it gets no compiled number."""
        other = "plain" if self.baseline == "compiled" else "compiled"
        if other == "compiled" and (self.use_library_inference or self.context is not None
                                    or any(_state_marks(t) for t in getattr(self, "traces", {}).values())):
            return  # library generation, a cache step and a step that writes state or reads values mid-call
            # (an engine's run()) cannot be compiled from outside
        if not self._kernels_installed():
            return  # the untouched model against both baselines is in the job-start clocks
        # only recorded, so at the primary size: at all ten sizes of a range it
        # was about 7.5 of a 31-hour end timing on Qwen3-VL-8B (estimate)
        for w in self.manifest.workloads[:1]:
            tensors = self.tensors[w.name]
            comp = compare(self.session, self._step_fn(self.baseline_model, tensors, other),
                           self._step_fn(self.model, tensors), pairs=FINAL_RANGE_PAIRS)
            clocks = {f"{other}_at_end": comp.median_baseline_ms,
                      f"speedup_vs_{other}": (1.0 / comp.median_ratio) if comp.median_ratio else None,
                      f"{other}_win_confirmed": comp.wins_by(0.0), f"{other}_loss_confirmed": comp.loses_by(0.0)}
            self.report.step_ms.setdefault(w.name, {}).update(clocks)
            self.log.append("baseline_comparison", workload=w.name, against=other,
                            stability=round(comp.stability, 3), **clocks)

    def _fresh_sizes(self) -> dict[str, list[mx.array]]:
        """Over a range: inputs at ten more sizes drawn as the search's were,
        with another seed, none of them a size the search ran at. The final
        check runs and times the model there too, and the headline is read off
        them, so a kernel tuned to the search's sizes cannot lift it."""
        named = self.report.constants.get("search_sizes") or {}
        if not named or not getattr(self.manifest, "serve", None):
            return {}
        dim, span = next(iter(self.manifest.serve.items()))
        if span.listed:
            return {}  # a list is every size the model sees: nothing between to draw from
        taken = set(named[dim])
        sizes = [n for n in manifest_mod.search_sizes(span, self.manifest.primary[dim], self.manifest.seed + 1)
                 if n not in taken]
        out = {}
        for w in self.manifest.workloads:
            if dim not in w.named_dims():
                continue
            seed = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)[0]
            for n in sizes:
                out[f"{w.name}@{dim}={n}:fresh"] = self._materialize(w, {**self.manifest.primary, dim: n}, seed)
        return out

    def _final_pairs(self) -> int:
        """Pairs behind the final check's decision: over a range at least
        FINAL_RANGE_PAIRS."""
        if self.report.constants.get("search_sizes"):
            return max(self.manifest.final_benchmark.pairs, FINAL_RANGE_PAIRS)
        return self.manifest.final_benchmark.pairs if self.use_library_inference else SHIP_PAIRS

    def _decode_check(self) -> dict | None:
        """MLX-LM prompt jobs: the untouched and the optimized model each
        generating DECODE_TOKENS tokens after a DECODE_PROMPT-token prompt
        (all but its last token prepared untimed, as a conversation's are),
        paired. None for other jobs, or for a job that already times
        generation after a context."""
        from backcountry_runtime.inference import LibraryInference
        from .manifest import resolve_dtype
        from .workload import _materialize_input
        if not self.use_library_inference or self.context is not None:
            return None
        spec = self.manifest.workloads[0].inputs[0]
        if self.text_tokenizer is not None:
            tokens = text_tokens(self.text_tokenizer, (1, DECODE_PROMPT), resolve_dtype(spec.dtype),
                                 self.manifest.seed + 3)
            prefix, last = tokens[:, :-1], [tokens[:, -1:]]
        else:
            prefix = _materialize_input(spec, (1, DECODE_PROMPT - 1), mx.random.key(self.manifest.seed + 3))
            last = [_materialize_input(spec, (1, 1), mx.random.key(self.manifest.seed + 4))]
        arms = [LibraryInference(arm.model, steps=DECODE_TOKENS, prefix_tokens=prefix)
                for arm in (self._baseline_arm(), self.model)]
        comp = compare(self.session, self._step_fn(arms[0], last), self._step_fn(arms[1], last),
                       pairs=MODEL_CHECK_PAIRS)
        self.log.append("decode_check", prompt_tokens=DECODE_PROMPT, generated_tokens=DECODE_TOKENS, **asdict(comp))
        return {"prompt_tokens": DECODE_PROMPT, "generated_tokens": DECODE_TOKENS, **asdict(comp),
                "speedup": 1.0 / comp.median_ratio if comp.median_ratio > 0 else None,
                "slower": comp.loses_by(0.0)}

    def _second_message_check(self) -> dict | None:
        """MLX-LM prompt jobs: the untouched and the optimized model each
        reading the job's prompt as a second message, on top of a first of
        the same length (prepared untimed, as a conversation's history is),
        paired. This is what every later turn of a chat waits for; a kernel
        delivered through a recording of an empty cache gains nothing here.
        None for jobs without an MLX-LM prompt workload."""
        from backcountry_runtime.inference import LibraryInference
        w = next((w for w in self.manifest.workloads if not w.context), None)
        if not self.use_library_inference or w is None:
            return None
        seed = workload_seeds(self.manifest.seed, f"{w.name}:second_message", 1)[0]
        first = self._materialize(w, self.manifest.primary, seed)[0].reshape(1, -1)
        tensors = self.tensors[w.name]
        arms = [LibraryInference(arm.model, steps=1, prefix_tokens=first)
                for arm in (self._baseline_arm(), self.model)]
        comp = compare(self.session, self._step_fn(arms[0], tensors), self._step_fn(arms[1], tensors),
                       pairs=MODEL_CHECK_PAIRS)
        history, tokens = first.shape[1], tensors[0].size
        self.log.append("second_message_check", history_tokens=history, prompt_tokens=tokens, **asdict(comp))
        return {"history_tokens": history, "prompt_tokens": tokens, **asdict(comp),
                "speedup": 1.0 / comp.median_ratio if comp.median_ratio > 0 else None,
                "slower": comp.loses_by(0.0)}

    def _headline(self, comparisons, fresh) -> dict | None:
        """Over a range: the whole model at every size the final check timed,
        smallest first, and the result in two numbers: the geometric mean of
        the speedup over the sizes the search never saw, and the worst size."""
        named = self.report.constants.get("search_sizes") or {}
        if not named:
            return None
        dim = next(iter(named))

        def size(name):
            if name in fresh or f"@{dim}=" in name:
                return int(name.split(f"@{dim}=", 1)[1].split(":", 1)[0])
            w = next((w for w in self.manifest.workloads if w.name == name), None)
            return self.manifest.primary[dim] if w and dim in w.named_dims() else None

        curve = sorted(({dim: size(name), "baseline_ms": c.median_baseline_ms,
                         "patched_ms": statistics.median(c.candidate_ms),
                         "speedup": 1.0 / c.median_ratio if c.median_ratio else None,
                         "result": "faster" if c.wins_by(0.0) else "slower" if c.loses_by(0.0) else "tie",
                         "fresh": name in fresh}
                        for name, c in comparisons.items() if size(name) is not None),
                       key=lambda row: row[dim])
        if not curve or any(row["speedup"] is None for row in curve):
            return None
        geomean = lambda rows: math.exp(statistics.fmean(math.log(r["speedup"]) for r in rows)) if rows else None
        worst = min(curve, key=lambda row: row["speedup"])
        return {"dim": dim, "curve": curve, "geomean_fresh": geomean([r for r in curve if r["fresh"]]),
                "geomean_all": geomean(curve), "worst": {dim: worst[dim], "speedup": worst["speedup"]}}

    def _record_final_clock(self, workload, comp) -> None:
        after = statistics.median(comp.candidate_ms)
        confirmed = bool(self.installed) and comp.wins_by(0.0)
        clocks = {
            "after": after, "baseline_at_end": comp.median_baseline_ms,
            "speedup": (1.0 / comp.median_ratio) if comp.median_ratio else None,
            "stability": comp.stability, "win_confirmed": confirmed,
        }
        if self.use_library_inference:
            clocks.update(generation_throughput({
                "workload_kind": "library_generation", "steps": self.manifest.final_benchmark.steps,
                "baseline_sequence_ms": comp.median_baseline_ms, "candidate_sequence_ms": after}))
        self.report.step_ms.setdefault(workload, {}).update(clocks)
        self.log.append("step_clock", workload=workload, phase="after", median_ms=after,
                        baseline_at_end_ms=comp.median_baseline_ms, speedup=clocks["speedup"],
                        stability=round(comp.stability, 3), win_confirmed=confirmed,
                        **{k: v for k, v in clocks.items() if k.endswith("tokens_per_second")})

    def _final_check(self) -> None:
        """The last end-to-end check once the regions are done: the patched
        model as a whole against the untouched model, on every workload and
        then at every sweep size, where the wrappers must hand the unrecorded
        shapes back to the original modules."""
        self.session.wait_ready()
        if not self._kernels_installed():
            self.final_ok = True
            self.report.final = {"passed": True, "reason": "no replacements installed"}
            return
        fresh = self._fresh_sizes()
        timed = self._timed_arms()
        final = run_e2e(
            self.session, self.baseline_model, self.model, step_of=self._for,
            workloads=[(w.name, self.tensors[w.name]) for w in self.manifest.workloads]
            + sorted(self.sweep_tensors.items()) + sorted(getattr(self, 'serve_tensors', {}).items())
            + sorted(fresh.items()),
            timed=timed,
            veto_pairs=self._final_pairs(),
            exact=self._requires_exact(),
            tolerances=self.manifest.tolerances,
            reference_model=None if self._requires_exact() else self._reference_model(untraced=fresh),
            fits=lambda name: _fits(self, name),
            after_timing=self._timing_memory,
        )
        final.margin_share = FINAL_MARGIN
        if final.workload_vetos:
            final.veto_passed = not any(c.loses_by(FINAL_MARGIN * c.median_baseline_ms)
                                        for c in final.workload_vetos.values())
        if self.report.constants.get("search_sizes"):
            final.min_wins = FINAL_RANGE_WINS
        paired_passed = final.passed and self._model_win(final)
        # The fresh sizes are checked above, all of them, and five are timed
        # here for the headline only: with 4 pairs an identical model read
        # clearly slower at 3 of 57 sizes, so a veto at each of ten more would
        # fail most good kernels.
        timed_fresh = _spread(sorted((label for label in fresh if _fits(self, label)),
                                     key=lambda label: int(label.split("=", 1)[1].split(":", 1)[0])),
                              FINAL_FRESH_TIMED)
        fresh_timings = {}
        for label in timed_fresh if all(c.passed for c in final.checks) else ():
            if not _fits(self, label):
                continue
            fresh_timings[label] = compare(self.session, self._step_fn(self._baseline_arm(), fresh[label], workload=label),
                                           self._step_fn(self.model, fresh[label], workload=label),
                                           pairs=FINAL_FRESH_PAIRS)
            self._timing_memory(label)
        for name, comp in final.workload_vetos.items():
            self._record_final_clock(name, comp)
        self.report.final = {"passed": paired_passed, "veto_passed": final.veto_passed,
                             "paired_win_confirmed": paired_passed,
                             "timings": {name: asdict(value) for name, value in final.workload_vetos.items()},
                             "checks": [c.__dict__ for c in final.checks]}
        if fresh_timings:
            self.report.final["fresh_timings"] = {name: asdict(value) for name, value in fresh_timings.items()}
        if untimed := sorted(n for n in [*(w.name for w in self.manifest.workloads), *fresh] if not _fits(self, n)):
            fit = self.report.constants["memory_fit"]
            self.report.final["not_timed"] = {
                "workloads": untimed, "reason": f"checked for correctness only: a run peaks within "
                f"{fit['headroom_gb']:g} GB of this Mac's {fit['gpu_limit_gb']:.1f} GB GPU limit"}
        if headline := self._headline({**final.workload_vetos, **fresh_timings}, fresh):
            self.report.final["headline"] = headline
        if final.passed and self._compiled_baseline is None:
            self.report.final["delivery"] = self._final_delivery_split()
        self.log.append("final_e2e", passed=final.passed, veto_passed=final.veto_passed,
                        checks=self.report.final["checks"])
        if not all(c.passed for c in final.checks):
            raise RuntimeError("final model validation failed: the patched model's outputs changed; "
                               "accepted checkpoints are preserved")
        if not final.veto_passed:
            self.report.final["lost"] = sorted(name for name, c in final.workload_vetos.items()
                                               if c.loses_by(FINAL_MARGIN * c.median_baseline_ms))
            return self._unconfirmed("the patched step measured slower than the baseline")
        if self.use_library_inference:
            # Each comparison already runs the full generation task, including
            # evolving state. Repeating it N times would change the objective.
            self.final_ok = paired_passed
            if (decode := self._decode_check()) is not None:
                self.report.final["decode"] = decode
                if decode["slower"]:
                    return self._unconfirmed(f"generating {DECODE_TOKENS} tokens after a {DECODE_PROMPT}-token "
                                             "prompt measured slower than the baseline")
            if (second := self._second_message_check()) is not None:
                self.report.final["second_message"] = second
                if second["slower"]:
                    return self._unconfirmed(f"reading a {second['prompt_tokens']}-token second message after "
                                             f"{second['history_tokens']} tokens of conversation measured slower "
                                             "than the baseline")
            self.report.final["measurement"] = dict(self.report.constants["measurement"])
            self.report.write(self.work_dir / "report.json")
            if not self.final_ok:
                self._unconfirmed("library inference timing did not confirm a speedup")
            return
        if self.manifest.final_benchmark.steps == 1:
            # The model is called once per use (one prompt read per chat
            # turn): the paired check above already timed exactly that.
            self.final_ok = paired_passed
            self.report.write(self.work_dir / "report.json")
            if not self.final_ok:
                self._unconfirmed("the paired timing did not confirm a speedup")
            return
        # An inconclusive single-step clock cannot settle deployment performance.
        # Correct, non-regressing candidates still reach the consecutive-step test.
        # Only a confirmed sequence win permits export.
        self._phase("checking consecutive model steps")
        sequences, deployed = self._final_sequences(final.checks)
        self.report.final["sequences"] = sequences
        # the same decision rule as every install: faster with confidence,
        # now over whole uninterrupted runs of the model
        self.final_ok = self._model_win(deployed)
        self.report.final["passed"] = self.final_ok
        self.report.final["lost"] = sorted(name for name, row in sequences.items() if row.get("regression_confirmed"))
        for name, row in sequences.items():
            self.report.step_ms[name]["sequence_win_confirmed"] = row["win_confirmed"]
            self.report.step_ms[name]["sequence_speedup"] = row["speedup"]
            self.report.step_ms[name]["win_confirmed"] &= self.final_ok
        self.report.write(self.work_dir / "report.json")
        if not self.final_ok:
            self._unconfirmed("the runs of consecutive steps did not confirm a speedup")

    # -- after the search: kernels at the served sizes -------------------------

    def _losing_spans(self) -> list[tuple[int, int]]:
        """What serving may leave out after a final it lost: each length the
        final found slower, as the span out to the nearest lengths it timed
        that did not lose. Empty when more than MAX_CUT_LENGTHS lost, or a
        loss is at no length of the served range (a workload without the dim)."""
        return self._spans_around(self.report.final.get("lost") or [], self.report.final.get("timings", {}))

    def _wrong_spans(self) -> list[tuple[int, int]]:
        """What serving may leave out after a final whose outputs changed: each
        length they changed at, as the span out to the nearest lengths checked
        that held (gpt-oss's expert kernel at 32 tokens, the one length where
        MLX's matmul rounds another way). The final then runs again in full."""
        checks = self.report.final.get("checks") or []
        return self._spans_around([c["name"] for c in checks if not c["passed"]], [c["name"] for c in checks])

    def _spans_around(self, lost, names) -> list[tuple[int, int]]:
        """Each of the lengths lost, as the span out to the nearest of names
        not lost; empty when more than MAX_CUT_LENGTHS lost or one is at no
        length of the served range."""
        dim = (self.report.serve or {}).get("dim")
        if not dim or not lost or len(lost) > MAX_CUT_LENGTHS or not getattr(self, "_serve_plan", None):
            return []
        sizes = {name: self._label_size(name, dim) for name in names}
        if any(sizes.get(name) is None for name in lost):
            return []
        bad = {sizes[name] for name in lost}
        good = sorted({n for n in sizes.values() if n is not None} - bad)
        served = self.manifest.serve[dim]
        if served.listed:  # exact sizes: each is its own
            return [(n, n) for n in sorted(bad)]
        return [(max([g for g in good if g < n], default=served.lo - 1) + 1,
                 min([g for g in good if g > n], default=served.hi + 1) - 1) for n in sorted(bad)]

    def _label_size(self, label: str, dim: str) -> int | None:
        """The dim's value in a timed workload: its label's (prompt@T=42,
        prompt@serve:T=32) or its manifest primary; None without the dim."""
        name, _, at = label.partition("@")
        if at:
            key, _, value = at.removeprefix("serve:").split(":")[0].partition("=")
            return int(value) if key == dim else None
        w = next((w for w in self.manifest.workloads if w.name == name), None)
        return self.manifest.primary.get(dim) if w is not None and dim in w.named_dims() else None

    def _serve_sizes(self) -> list | None:
        """check_optimizations_for: give each scope that generalizes generated
        code that reads its size, checked against the original at every size
        it serves, in place of its searched wrapper. Returns what _unserve
        restores, or None."""
        from .serve import check_replays, plan_scopes
        if not getattr(self.manifest, "serve", None) or not self._kernels_installed():
            return None
        dim, sizes = next(iter(self.manifest.serve.items()))
        exact = bool(sizes.listed)
        self._phase("checking kernels at the other sizes")
        started = time.monotonic()
        workloads = [w for w in self.manifest.workloads if dim in w.named_dims()]
        primary = self.manifest.primary[dim]
        plans, refused = plan_scopes(self.installed, self.traces, self.serve_traces, {w.name for w in workloads},
                                     [primary] + sizes.recordings(primary), live=self.live_scopes)
        report = self.report.serve = {"dim": dim, "sizes": list(sizes.listed) or [sizes.lo, sizes.hi],
                                      "exact": exact, "scopes": {}, "kernels": {}, "not_generalized": refused}
        for path, reason in refused.items():
            self.log.append("serve_refused", scope=path, reason=reason)
        for plan in plans:
            if plan.path in self.live_scopes:
                # its own method runs at every size; only its kernels' sizes are checked
                plan.checked = {n: None for n in plan.sized.sizes}
        replays = [plan for plan in plans if plan.path not in self.live_scopes]
        if replays:
            workload = workloads[0]
            seed = workload_seeds(self.manifest.seed, workload.name, manifest_mod.BOUNDARY_INPUT_SETS)[0]
            candidates = {}
            for plan in replays:
                original, _variants, specs = self.installed[plan.path]
                candidate = plan.sized.finalize(f"C_{_safe(plan.path)}", plan.sized.sizes, {})
                candidates[plan.path] = _load_class(candidate)(original, specs)
            self.session.off_clock(lambda: check_replays(
                self.model, replays, candidates,
                lambda size: self._for(self.model, workload.name)(
                    *self._materialize(workload, {**self.manifest.primary, dim: size}, seed)),
                sorted(set(sizes.sizes()) | {primary}), swap_install, swap_uninstall))
        kernel_sizes = self._serve_kernels([plan for plan in plans if plan.served(exact)], exact)
        self._serve_plan = (plans, kernel_sizes, exact)
        report["check_s"] = round(time.monotonic() - started, 1)
        return self._install_served(plans, kernel_sizes, exact)

    def _install_served(self, plans, kernel_sizes, exact, excluded=()) -> list | None:
        """Each scope's size-reading code, serving only the sizes where one of
        its kernels runs (elsewhere the original block runs, untouched: Gemma
        4's generated code cost 1% at 42 tokens, where no kernel ran), less any
        excluded spans of the manifest dim; returns what _unserve restores."""
        from .serve import in_dim
        from .bind.sized import runs
        from .ladder.sizes import shape_at
        report, dim = self.report.serve, self.report.serve["dim"]
        restore = []
        for plan in plans:
            served = [n for n in plan.served(exact) if not _in_spans(plan.to_dim, n, excluded)]
            kernels = {kid: tuple(sorted(set(kernel_sizes.get(kid) or (plan.sized.sizes[0],)) & set(served)))
                       for kid in plan.sized.kernel_ids}
            served = sorted(set().union(*kernels.values()))
            report["scopes"][plan.path] = {
                "call": plan.address, "size": f"{plan.sized.size_arg}.shape[{plan.sized.size_dim}]",
                "recorded_sizes": plan.sized.sizes, "generalized_integers": plan.sized.generalized,
                "sizes": runs(served), f"{dim}_sizes": in_dim(plan, served),
                "first_difference": next(([n, why] for n, why in sorted(plan.checked.items()) if why), None)}
            if not served:
                continue
            original, _variants, specs = self.installed[plan.path]
            if plan.path in self.live_scopes:
                # the block's own method already runs at every size: its kernels
                # may now also run at the shapes of the sizes checked here
                live = _resolve(self.model, plan.path)
                wider = {}
                for kid, spec in live._specs.items():
                    template, allowed = plan.kernel_shapes.get(kid), list(spec.input_signatures or [])
                    for n in (kernels.get(kid, ()) if template else ()):
                        signature = [[list(shape), str(dtype).removeprefix("mlx.core.")]
                                     for shape, dtype in shape_at(template, n)]
                        if signature not in allowed:
                            allowed.append(signature)
                    wider[kid] = replace(spec, input_signatures=allowed)
                occupant = swap_install(self.model, plan.path, type(live)(original, wider))
                restore.append((plan.path, occupant, self.emitted.get(plan.path)))
                continue
            emitted = plan.sized.finalize(f"S_{_safe(plan.path)}", served, kernels)
            occupant = swap_install(self.model, plan.path, _load_class(emitted)(original, specs))
            restore.append((plan.path, occupant, self.emitted.get(plan.path)))
            self.emitted[plan.path] = emitted
        if excluded:
            report.pop("withdrawn", None)
            report["excluded"] = [list(span) for span in excluded]
        self.log.append("serve_installed", scopes=[path for path, _, _ in restore],
                        kernels=report["kernels"], check_s=report["check_s"], excluded=list(excluded))
        self.report.write(self.work_dir / "report.json")
        return restore or None

    HISTORY_SPAN = (64, 8192)  # conversation lengths a kernel that reads it is checked across
    HISTORY_POINTS = 10        # lengths it is checked and timed at, spread by the same factor
    HISTORY_STEPS = 8          # decode calls in a row at each, every output compared
    HISTORY_PAIRS = 4

    def _serve_history_lengths(self) -> list:
        """A kernel a block's own method calls on the conversation itself
        (attention over the cache) has inputs that grow with it, and its shape
        guard holds it at the length the job recorded. It may run across a
        span of lengths: the span of HISTORY_POINTS lengths over HISTORY_SPAN,
        out from the job's own, at which it is bit-exact over HISTORY_STEPS
        calls in a row and not slower; then the block check, which drives
        every position of three conversations, must pass with it there too.
        Returns what _unserve restores."""
        if self.context is None or not self.live_scopes:
            return []
        from backcountry_runtime import kernels as runtime_kernels
        from backcountry_runtime.exact import bitwise_equal
        from backcountry_runtime.state import _cache_observation, _copy_cache
        patched, untouched = self.model.model, self.baseline_model.model
        own, step = self.context.context, self.tensors[self.context.name][0]
        seen, real = {}, runtime_kernels.try_call
        verified = {kid: spec.input_signatures or [] for p in self.live_scopes
                    for kid, spec in _resolve(self.model, p)._specs.items()}
        grown: dict = {}  # kid -> calls that ran it at a shape the job did not record

        def recording(spec, inputs, **kwargs):
            signature = [[list(a.shape), str(a.dtype).removeprefix("mlx.core.")] for a in inputs]
            seen.setdefault(spec.kernel_id, []).append(signature)
            out = real(spec, inputs, **kwargs)
            if out is not None and signature not in verified.get(spec.kernel_id, []):
                grown[spec.kernel_id] = grown.get(spec.kernel_id, 0) + 1
            return out

        calls = {}
        runtime_kernels.try_call = recording
        try:
            for length in (own, 2 * own):  # what grows with the conversation: each call's shapes at two lengths
                seen.clear()
                mx.eval(patched(step, cache=self._history_cache(patched, length)))
                for kid, signatures in seen.items():
                    calls.setdefault(kid, []).append(signatures)
        finally:
            runtime_kernels.try_call = real
        follows: dict = {}  # kid -> templates, per input [[dim or (a, b)], dtype]: the dim is a * length + b
        for kid, pair in calls.items():
            if len(pair) != 2 or len(pair[0]) != len(pair[1]):
                continue
            for first, second in zip(*pair):
                if first == second:
                    continue
                template = []
                for (s1, dtype), (s2, _) in zip(first, second):
                    dims = []
                    for v1, v2 in zip(s1, s2):
                        a, rest = divmod(v2 - v1, own)
                        dims.append(v1 if v1 == v2 else (a, v1 - a * own) if not rest else None)
                    template.append([dims, dtype])
                if all(d is not None for dims, _ in template for d in dims) and \
                        template not in follows.setdefault(kid, []):
                    follows[kid].append(template)
        follows = {kid: templates for kid, templates in follows.items() if templates}
        paths = sorted(p for p in self.live_scopes if set(_resolve(self.model, p)._specs) & set(follows))
        if not paths:
            return []
        lo_span, hi_span = self.HISTORY_SPAN
        points = sorted({round(lo_span * (hi_span / lo_span) ** (i / (self.HISTORY_POINTS - 1)))
                         for i in range(self.HISTORY_POINTS)})

        def ranged(template, lo, hi):
            return [[[d if isinstance(d, int) else [d[0] * lo + d[1], d[0] * hi + d[1]] for d in dims], dtype]
                    for dims, dtype in template]

        def widened(path, lo, hi):
            wrapper = _resolve(self.model, path)
            return {kid: (replace(spec, input_signatures=list(spec.input_signatures or [])
                                  + [ranged(t, lo, hi) for t in follows[kid]])
                          if kid in follows else spec) for kid, spec in wrapper._specs.items()}

        def sequence(model, cache):
            def run():
                c = _copy_cache(cache)
                outs = [model(step, cache=c) for _ in range(self.HISTORY_STEPS)]
                return outs + [_cache_observation(c)]
            return run

        self._phase("checking kernels at other conversation lengths")
        passed = {}
        installed = {p: swap_install(self.model, p, type(_resolve(self.model, p))(
            self.installed[p][0], widened(p, lo_span, hi_span))) for p in paths}
        try:
            for length in points:
                grown.clear()
                caches = self._history_cache(untouched, length), self._history_cache(patched, length)
                a = sequence(untouched, caches[0])()
                runtime_kernels.try_call = recording
                try:
                    b = sequence(patched, caches[1])()
                finally:
                    runtime_kernels.try_call = real
                same = all(bitwise_equal(x, y) for x, y in zip(flatten_arrays(a), flatten_arrays(b)))
                ran = all(grown.get(kid, 0) >= self.HISTORY_STEPS for kid in follows)
                slower = same and ran and compare(self.session, sequence(untouched, caches[0]),
                                                  sequence(patched, caches[1]),
                                                  pairs=self.HISTORY_PAIRS).loses_by(0.0)
                passed[length] = same and ran and not slower
                self.log.append("history_length", length=length, exact=same, kernels_ran=ran, slower=bool(slower))
                del caches, a, b
                mx.clear_cache()
        finally:
            for p, previous in installed.items():
                swap_install(self.model, p, previous)
        below = [n for n in points if n <= own][::-1]
        above = [n for n in points if n >= own]
        lo = hi = own
        for n in below:
            if not passed[n]:
                break
            lo = n
        for n in above:
            if not passed[n]:
                break
            hi = n
        if lo == hi:
            self.log.append("history_span", kernels=sorted(follows), span=None)
            return []
        entries = [(p, self.installed[p][0], widened(p, lo, hi), self.emitted[p]) for p in paths]
        problem, previous = self._check_live(entries)
        self.log.append("history_span", kernels=sorted(follows), span=[lo, hi], block_check=problem)
        if problem is not None:
            return []
        return [(p, previous[p], self.emitted[p]) for p in paths]

    def _history_cache(self, model, length: int) -> list:
        """The model's cache holding `length` tokens of a conversation drawn
        from the workload's token range, read 2,048 at a time."""
        from backcountry_runtime.state import _make_cache, _with_cache_work
        spec = self.context.inputs[0]
        tokens = mx.random.randint(int(spec.low or 0), int(spec.high), (1, length),
                                   key=mx.random.key(self.manifest.seed + length)).astype(mx.int32)
        cache = _make_cache(model)
        for start in range(0, length, 2048):
            mx.eval(_with_cache_work(model(tokens[:, start:start + 2048], cache=cache), cache))
        return cache

    SERVE_GRID = 10     # timing points per kernel over a range, both ends included
    SERVE_SAMPLES = 12  # random sizes timed inside the range the grid picks: a clear loss cuts it
    SERVE_BATCH = 256   # sizes per correctness worker: a hang loses one batch
    # The serve step's recordings are untimed and number thousands (4,038 on
    # Qwen3-30B-A3B, 10 min of work): paced like timed work they cost 30 min
    # of cooling. They run unpaced; each timing grid rests this long first,
    # about as long as the OS heat flag took to clear once idle (2026-09-16).
    SERVE_REST_S = 60.0
    SERVE_RECORD_BYTES = 2 << 30  # recordings on disk at once, waiting to be checked: Qwen3-30B-A3B's
    # expert input is 64 MB at 2,048 tokens, so a batch of 256 there would be 16 GB
    # copies of the region each pass records, all checked: gpt-oss's expert kernel was off
    # MLX's rounding at 124 rows on 9 copies in 10, and the one copy checked then matched
    SERVE_COPIES = 4
    # sizes per check run three times and compared (a race shows at some of
    # them); every other size runs once against the recorded operations
    SERVE_REPEATS = 32

    def _serve_kernels(self, plans, exact: bool) -> dict:
        """Each bit-exact kernel's served sizes, in its scopes' size: the sizes
        it is right at (every one checked in the sandbox) and wins at. Over a
        range the wins come from a timing grid plus random samples; exact sizes
        are each timed. Kernels left out keep their recorded size only."""
        from .ladder.sizes import shape_at
        from .serve import in_dim, served_range
        from .bind.sized import call_size, runs
        report, dim = self.report.serve["kernels"], self.report.serve["dim"]
        found: dict[str, dict] = {}
        for plan in plans:
            served = set(plan.served(exact))
            for splice in plan.splices:
                kid = splice.kernel.kernel_id
                # the scope sizes of every workload the search ran this kernel in
                searched = {call_size(self.traces[w], next(sc for sc in self.traces[w].scope_calls
                                                           if sc.address == address),
                                      plan.sized.size_arg, plan.sized.size_dim)
                            for (w, address), cuts in self.installed[plan.path][1].items()
                            if any(c.kernel.kernel_id == kid for c in cuts)}
                info = found.setdefault(kid, {
                    "spec": splice.kernel, "fingerprint": splice.fingerprint, "primary": plan.sized.sizes[0],
                    "to_dim": plan.to_dim, "template": plan.kernel_shapes.get(kid),
                    "settings": plan.kernel_settings.get(kid), "sizes": served, "searched": searched})
                if (info["template"] != plan.kernel_shapes.get(kid)
                        or info["settings"] != plan.kernel_settings.get(kid)
                        or info["primary"] != plan.sized.sizes[0]):
                    info["template"] = None
                info["sizes"] &= served
                info["searched"] &= searched
        chosen = {}
        for kid, info in found.items():
            spec, fingerprint, template, primary = info["spec"], info["fingerprint"], info["template"], info["primary"]
            why = None
            # a kernel within tolerance serves where the search's whole-model
            # checks ran, and elsewhere where its region error is no larger
            # than the largest it had there
            checked = None if self.shipped_tags.get(fingerprint) == "preserving" else \
                tuple(sorted((info["searched"] & info["sizes"]) | {primary}))
            searched_only = {"sizes": runs(checked or ()), f"{dim}_sizes": in_dim(info["to_dim"], checked or ()),
                             "reason": "checked within tolerance, so served at the sizes the search checked it at"}
            if spec.native_call is not None or spec.reference_sequence is not None:
                why = "not a searched kernel"
            elif template is None:
                why = "its operations or input shapes do not follow the size"
            instance = None
            if why is None:
                region = self.shipped_regions[fingerprint]
                want = shape_at(template, primary)
                for label, member, _copies in capture_instances(region, self.traces):
                    specs = self.traces[member.workload].span_specs(member.start_seq, member.end_seq)
                    if [[list(specs[a][0]), specs[a][1]] for a in member.input_ids] == want:
                        instance = (label, member)
                        break
                if instance is None:
                    why = "no saved inputs at its recorded size"
            if why is not None and checked is not None:
                chosen[kid], report[kid] = checked, searched_only
                continue
            if why is not None:
                report[kid] = {"sizes": [[primary, primary]], f"{dim}_sizes": in_dim(info["to_dim"], (primary,)),
                               "reason": why}
                continue
            label, member = instance
            trace = self.traces[member.workload]
            job = dict(kernel=json.loads(spec.to_json()),
                       nodes_json=nodes_to_json(trace.nodes[member.start_seq:member.end_seq + 1]),
                       input_ids=tuple(member.input_ids), output_ids=tuple(member.output_ids),
                       inputs_path=str(self.store._path(fingerprint, label, 0, "inputs")), shapes=template,
                       settings=info["settings"] or [],
                       weight_inputs=tuple(a in trace.weights for a in member.input_ids))
            # Timings use only inputs from real passes of the model, run at
            # sizes in the manifest's own dim (prompt tokens, image sides):
            # the layer's size is read off each recording, so it need not
            # follow the dim in any particular way. Correctness uses a real
            # recording where there is one, else a cut of the longest.
            job["recordings"], dims = {}, {}  # layer size -> inputs file, -> dim value

            def record(values):
                found = self._serve_recordings(self.shipped_regions[fingerprint], fingerprint, member,
                                               template, values)
                for value, (n, path) in found.items():
                    if n in info["sizes"]:
                        job["recordings"][str(n)], dims[n] = path, value
                return {n for n, _ in found.values() if n in info["sizes"]}

            def timed(values):
                sizes = sorted(record(values))
                self.session.rest(self.SERVE_REST_S)  # the recordings are unpaced
                return sizes, self._serve_grid(job, sizes)

            self._searched_recordings(fingerprint, member, template, job["recordings"])
            if str(primary) not in job["recordings"]:
                job["recordings"][str(primary)] = job["inputs_path"]
            searched_ok = set()
            if checked is not None:
                # the error bar: the largest error at the searched sizes it is
                # right at; a size flagged there is neither a bar nor served
                errors = {}
                searched_ok, searched_wrong = self._serve_correct(
                    {**job, "allowed_error": float("inf")},
                    [n for n in checked if str(n) in job["recordings"]], errors)
                job["allowed_error"] = max((e for n, e in errors.items() if n in searched_ok), default=float("nan"))
                if not job["allowed_error"] < float("inf"):  # no finite error to bound the others by
                    chosen[kid] = tuple(sorted(searched_ok))
                    report[kid] = {**searched_only, "sizes": runs(chosen[kid]),
                                   f"{dim}_sizes": in_dim(info["to_dim"], chosen[kid]),
                                   "wrong": dict(sorted(searched_wrong.items())[:20])}
                    continue
            span = self.report.serve["sizes"]
            if exact:
                # a few sizes: time every one, serve those it is right and faster at
                grid, clock = timed(sorted(set(span) | {self.manifest.primary[dim]}))
                wins = sorted(n for n in grid if clock.get(n, {}).get("win"))
            else:
                lo, hi, steps = span[0], span[-1], self.SERVE_GRID - 1
                grid, clock = timed(sorted({lo, hi, self.manifest.primary[dim]}
                                           | {round(lo * (hi / lo) ** (i / steps)) for i in range(1, steps)}))
                # speed between grid points zig-zags with the kernels' tiles:
                # time random sizes inside the stretches too, and cut at any loss
                stretches = served_range(list(range(min(info["sizes"]), max(info["sizes"]) + 1)), clock, primary, grid)
                inside = sorted({v for a, b in stretches if a in dims and b in dims
                                 for v in range(dims[a] + 1, dims[b]) if v not in dims.values()})
                picks = random.Random(kid).sample(inside, min(self.SERVE_SAMPLES, len(inside)))
                clock.update(timed(picks)[1])
                wins = [n for a, b in served_range(list(range(min(info["sizes"]), max(info["sizes"]) + 1)),
                                                   clock, primary, grid) for n in range(a, b + 1)]
            # the primary size is checked even where it only ties: served_range
            # serves nothing for a kernel not right there (Qwen3.8-27B's second
            # kernel tied at 511 rows, won at 21 of 22 timed sizes, served none)
            correct, wrong, cut_from = self._serve_every_size(job, region, fingerprint, member, template,
                                                              sorted(set(wins) | {primary}),
                                                              [] if exact else span, info["to_dim"], dims)
            served = tuple(n for n in wins if n in correct) if exact else \
                tuple(n for a, b in served_range(sorted(correct), clock, primary, grid, wrong=wrong)
                      for n in range(a, b + 1))
            if checked is not None:
                served = tuple(sorted(set(served) | searched_ok))
            chosen[kid] = served
            report[kid] = {"sizes": runs(served), f"{dim}_sizes": in_dim(info["to_dim"], served),
                           "grid": {n: row for n, row in sorted(clock.items()) if n in grid},
                           "samples": {n: row for n, row in sorted(clock.items()) if n not in grid},
                           "checked": len(correct) + len(wrong),  # sizes checked, real or cut
                           "cut_from": cut_from,  # their inputs: first rows of this size's real pass, where none of their own
                           "checked_range": [min(correct | set(wrong)), max(correct | set(wrong))]
                           if correct or wrong else None,
                           "recorded": runs(int(n) for n in job["recordings"]),  # sizes with a real pass
                           f"{dim}_recorded": sorted(dims.values()),  # where the model was run for them
                           "wrong": dict(sorted(wrong.items())[:20]), "wrong_count": len(wrong)}
            if checked is not None:
                report[kid].update(allowed_error=job["allowed_error"], reason=(
                    "within tolerance: served at the sizes the search checked it at, and where its region "
                    "error is no larger than the largest it had there"))
            self.log.append("serve_kernel", kernel=kid, **report[kid])
        return chosen

    def _serve_every_size(self, job, region, fingerprint, member, template, wins, span, to_dim, dims):
        """Check the kernel at every size in wins that a pass of the model
        produces: (correct, wrong, the size cut from or None). Sizes already
        recorded are checked as they are. A region whose inputs are all
        floating point is checked at the others on the first rows of the
        longest recording, every copy of the region in it (_cut_source).
        Otherwise, over a range (span, in the dim's units), the model is run
        at every other dim value whose size lands in wins, a batch at a time,
        and each batch's recordings are deleted once checked. Only the first
        SERVE_COPIES copies of the region are recorded, so a pass evaluates
        the model up to them and no further. A size whose pass records
        nothing (the model runs other operations there) is wrong."""
        from .ladder.sizes import recording_sets
        wanted = set(wins)
        correct, wrong = self._serve_correct(job, sorted(n for n in wanted if str(n) in job["recordings"]))
        cut = self._cut_source(job)
        if cut is not None:
            # Qwen3.8-27B: a real pass per prompt length was 2,000 whole-model
            # passes per kernel, 13 hours; the cuts take minutes
            shorter = sorted(n for n in wanted if n <= cut[0] and n not in correct and n not in wrong)
            ok, bad = self._serve_correct({**job, "cut_from": cut}, shorter)
            correct |= ok
            wrong.update(bad)
            wanted -= set(shorter)
            if not wanted - correct - set(wrong):
                return correct, wrong, cut[0]

        def size_of(value):
            if to_dim is None:
                return None
            n = (value - to_dim[1]) / to_dim[0]
            return int(n) if n == int(n) else None

        values = [v for v in range(span[0], span[-1] + 1) if v not in dims.values()
                  and (to_dim is None or size_of(v) in wanted)] if span else []
        start, largest = 0, None  # values ascend, so the last recording is the largest so far
        reached, rising = 0, True  # the largest size a pass produced; whether sizes only rose
        while start < len(values) and not (rising and wanted and reached >= max(wanted)):
            # past the stretch's top, later passes only repeat or exceed it: LFM2.5
            # reads prompts in 2,048-token chunks, so every longer prompt repeats
            # the 2,048-row size (6,000 passes per kernel, all already checked)
            count = 1 if largest is None else max(1, min(self.SERVE_BATCH, self.SERVE_RECORD_BYTES // largest))
            chunk = values[start:start + count]
            start += count
            found = self._serve_recordings(region, fingerprint, member, template, chunk)
            for value in chunk:
                if value in found:
                    rising &= found[value][0] >= reached
                    reached = max(reached, found[value][0])
            largest = max([sum(f.stat().st_size for f in recording_sets(p)) for _n, p in found.values()]
                          + [largest or 1])
            fresh = {n: path for n, path in found.values()
                     if n in wanted and str(n) not in job["recordings"] and n not in correct and n not in wrong}
            for value in chunk:
                if value not in found and size_of(value) in wanted:
                    wrong[size_of(value)] = "no real pass of the model recorded at this size"
            ok, bad = self._serve_correct({**job, "recordings": {**job["recordings"],
                                                                 **{str(n): p for n, p in fresh.items()}}},
                                          sorted(fresh))
            correct |= ok
            wrong.update(bad)
            kept = set(job["recordings"].values())
            for _n, path in found.values():
                if path not in kept:
                    for f in recording_sets(path):
                        f.unlink(missing_ok=True)
        return correct, wrong, None if cut is None else cut[0]

    @staticmethod
    def _cut_source(job) -> list | None:
        """[size, recording]: the longest real recording, whose first rows
        stand in for a pass at each smaller size; None when any input but a
        weight is not floating point. A cut keeps an index list's values,
        which can point past it (Qwen3-30B-A3B's unsort order), and an offset
        list (an expert's first row) can too, whatever its shape."""
        if not job["recordings"] or not JobRunner._floats_only(job):
            return None
        longest = max(job["recordings"], key=int)
        return [int(longest), job["recordings"][longest]]

    def _serve_recordings(self, region, fingerprint, member, template, values) -> dict[int, tuple[int, str]]:
        """Run the model at each value of the manifest's dim and save the
        region's inputs from that pass, keyed like member's: value -> (the
        layer's size there, the file). Built inputs can mislead: a short
        gpt-oss prompt sends its tokens to ~20 experts, a cut of a long one to
        one or two. A value no recorded pass lines up with (the model runs
        other operations there) gets nothing."""
        from .regions.price import CaptureMismatch
        dim = self.report.serve["dim"]
        w = next(w for w in self.manifest.workloads if dim in w.named_dims())
        seed = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)[0]
        found = {}
        self.tracer.install()
        try:
            for value in values:
                tensors = self._materialize(w, {**self.manifest.primary, dim: value}, seed)
                for _label, other, _copies in capture_instances(region, self.traces):
                    copies = self._serve_copies(region, other)
                    got = None
                    for tried in (copies, copies[:1]) if len(copies) > 1 else (copies,):
                        try:
                            got = self.session.off_clock(lambda: capture_boundaries(
                                self.tracer, self._for(self.baseline_model, w.name), tensors,
                                self.traces[other.workload], {a for c in tried for a in c.input_ids}), paced=False)
                        except CaptureMismatch:
                            continue
                        copies = tried
                        break
                    if got is None:
                        continue
                    sets = [{mine: got[theirs] for mine, theirs in zip(member.input_ids, c.input_ids)}
                            for c in copies]
                    n = _layer_size(template, sets[0])
                    if n is not None:
                        # each copy its own set beside the first, all checked
                        paths = [self.store.save(fingerprint, f"serve:{n}", i, "inputs", self._sized_only(member, a))
                                 for i, a in enumerate(a for a in sets if _layer_size(template, a) == n)]
                        found[value] = (n, str(paths[0]))
                        break
                del tensors
                mx.clear_cache()
        finally:
            self.tracer.uninstall()
        return found

    def _serve_copies(self, region, other) -> list:
        """other and the next copies of the region in its pass, with its shapes,
        up to SERVE_COPIES: the copies one recording pass saves."""
        trace = self.traces[other.workload]
        shapes = _boundary_shapes(trace, other)
        later = [m for m in sorted(region.members, key=lambda m: m.start_seq)
                 if m is not other and m.workload == other.workload and _boundary_shapes(trace, m) == shapes]
        return [other] + later[:self.SERVE_COPIES - 1]

    def _searched_recordings(self, fingerprint, member, template, recordings) -> None:
        """The inputs the search saved at each of its workloads, keyed like
        member's, by the layer's size there: what its whole-model checks saw."""
        for label, other, _copies in capture_instances(self.shipped_regions[fingerprint], self.traces):
            path = self.store._path(fingerprint, label, 0, "inputs")
            if not path.is_file():
                continue
            saved = self.store.load(fingerprint, label, 0, "inputs")
            arrays = {mine: saved[theirs] for mine, theirs in zip(member.input_ids, other.input_ids)}
            n = _layer_size(template, arrays)
            if n is not None and str(n) not in recordings:
                recordings[str(n)] = str(self.store.save(fingerprint, f"serve:{n}", 0, "inputs",
                                                         self._sized_only(member, arrays)))

    def _sized_only(self, member, arrays) -> dict:
        """A recording's inputs without the weights, which are the same at
        every size: the sandbox takes them from the search's own recording.
        Saved with them, Qwen3-30B-A3B's expert weights were 88 of every 94 MB."""
        weights = self.traces[member.workload].weights
        return {aid: a for aid, a in arrays.items() if aid not in weights}

    @staticmethod
    def _floats_only(job) -> bool:
        """Whether every input but a weight is floating point: no value of
        one can be an index or an offset into the others."""
        dtypes = [str(dtype).removeprefix("mlx.core.") for _dims, dtype in job["shapes"]]
        weights = job.get("weight_inputs") or (False,) * len(dtypes)
        return all(d.startswith(("float", "bfloat")) for d, weight in zip(dtypes, weights) if not weight)

    def _serve_correct(self, job: dict, sizes: list[int], errors: dict | None = None) -> tuple[set, dict]:
        """Which sizes the kernel is right at, checked in the sandbox in batches
        (a hang loses one batch); and why the others are not. With shader
        validation off every recorded copy is compared with the recorded
        operations at every size, and runs three times at SERVE_REPEATS of
        them; with it on the kernel runs once per copy for its out-of-bounds
        reports alone (what validation computes is not what ships), on the
        first copy only where no input is an index (shapes alone then set the
        addresses). Qwen3.8-27B's down projection, checked in full both ways
        at its 2,000 sizes, measured 2.5 h; this way 27 min. errors, when given,
        collects each size's largest region error (with job["allowed_error"])."""
        from .ladder.sizes import SizesSpec
        from .sandbox.protocol import require_responsive, run_job as run_worker
        wrong = {}
        repeat = tuple(sorted(set(sizes[::-(-len(sizes) // self.SERVE_REPEATS) or 1]) | set(sizes[-1:])))
        how = {"validate": dict(addresses_only=True, copies=1 if self._floats_only(job) else 0),
               "score": dict(repeat_at=repeat)}
        for start in range(0, len(sizes), self.SERVE_BATCH):
            batch = tuple(sizes[start:start + self.SERVE_BATCH])
            for mode in ("validate", "score"):
                checked = run_worker(SizesSpec(sizes=batch, phase="validate", **job, **how[mode]), mode,
                                     timeout_s=120.0 + 10.0 * len(batch))
                require_responsive(checked)
                results = checked.detail.get("results", {})
                if errors is not None:
                    for n, e in checked.detail.get("errors", {}).items():
                        errors[int(n)] = max(e, errors.get(int(n), e))
                validation = checked.detail.get("validation")
                if validation:
                    by_size = validation.get("by_size")
                    if not by_size or validation.get("unattributed"):  # cannot tell which size: all of them
                        wrong.update({n: "shader validation reported an invalid device access" for n in batch})
                        break
                    wrong.update({n: "shader validation reported an invalid device access"
                                  for n in batch if str(n) in by_size})
                for n in batch:
                    reason = results.get(str(n), checked.detail.get("reason", "not checked"))
                    if reason is not None and n not in wrong:
                        wrong[n] = reason
        return set(sizes) - set(wrong), wrong

    def _serve_grid(self, job: dict, grid: list[int]) -> dict[int, dict]:
        """Each grid size's clock, the kernel against its operations, in the sandbox."""
        from .ladder.sizes import SizesSpec
        from .sandbox.protocol import require_responsive, run_job as run_worker
        timed = run_worker(SizesSpec(sizes=tuple(grid), phase="score", **job), "score",
                           timeout_s=120.0 + 60.0 * len(grid))
        require_responsive(timed)
        return {int(n): row for n, row in timed.detail.get("results", {}).items()}

    def _unserve(self, restore: list, reason: str) -> None:
        """Put the searched wrappers back; the final check runs again on them."""
        for path, occupant, emitted in reversed(restore):
            swap_install(self.model, path, occupant)
            self.emitted[path] = emitted
        self.report.serve["withdrawn"] = reason
        self.log.append("serve_withdrawn", reason=reason)
        self.report.write(self.work_dir / "report.json")

    def _unconfirmed(self, reason: str) -> None:
        """The outputs are right and the win is not confirmed: a measurement
        result, not a broken job. The report keeps every number measured and
        the checkpoints stay; nothing is exported, since only a confirmed win
        ships. Wrong outputs still raise."""
        self.final_ok = False
        self.report.final.update(passed=False, reason=reason)
        self.log.append("final_unconfirmed", reason=reason)
        self.report.write(self.work_dir / "report.json")

    def _final_delivery_split(self) -> dict:
        """Split the final speedup: the untouched model against the installed
        scopes compiled with no kernel, and that against the patched model.
        Only a plain baseline with graph-delivered scopes has anything to split."""
        if self.baseline != "plain":
            return {"compiled_scopes": [], "timings": {}}
        compiled = {}
        for path in self.emitted:
            if isinstance(_resolve(self.model, path), GraphWrapper):
                try:
                    compiled[path] = (self._identity(path, f"Id_{_safe(path)}"), {})
                except NotReplayable as error:  # a report: a scope that cannot be compiled alone has no split
                    return {"compiled_scopes": [], "timings": {}, "not_split": f"{path}: {error}"}
        if not compiled:
            return {"compiled_scopes": [], "timings": {}}
        identity = self._copy_incumbent({}, compiled)
        pairs = self.manifest.final_benchmark.pairs if self.use_library_inference else SHIP_PAIRS
        timings, arms = {}, self._timed_arms()
        for w in self.manifest.workloads:
            if w.name not in arms:
                continue
            plain, patched = arms[w.name]
            identity_step = self._step_fn(identity, self.tensors[w.name])
            timings[w.name] = {
                "plain_vs_compiled_identity": asdict(compare(self.session, plain, identity_step, pairs=pairs)),
                "compiled_identity_vs_patched": asdict(compare(self.session, identity_step, patched, pairs=pairs)),
            }
        split = {"compiled_scopes": sorted(compiled), "timings": timings}
        self.log.append("final_delivery_split", **split)
        return split

    def _final_sequences(self, checks) -> tuple[dict, E2EResult]:
        """Time the model the way it is deployed: whole runs of consecutive
        steps, original and patched, each run uninterrupted, the two
        alternated in both orders with cooling between runs. Returns the
        report rows and the comparison in the form every install decision
        reads."""
        config = self.manifest.final_benchmark
        results, timings = {}, {}
        sequence_checks = list(checks)
        for workload in self.manifest.workloads:
            name, tensors = workload.name, self.tensors[workload.name]
            arms = [self._for(model, name) for model in (self.baseline_model, self.model)]
            # beside a conversation, a workload without one is one call per use
            # (a prompt read once), and the steps are the conversation's
            conversation = self.context is not None and arms[0] is self.baseline_model \
                and isinstance(arms[0], ContextStep)  # a run() handed the conversation is one use per call
            per_use = config.steps if self.context is None or conversation else 1
            steps = []
            for model in arms:
                # compile the forward call only, as a caller would
                call = lambda *args, m=model: m(*args)
                steps.append(mx.compile(call) if self.baseline == "compiled" else call)
            # a run is never shorter than one timed sample: ten steps of a 20 us
            # model is mostly the sample's fixed submit-and-sync cost, and could not
            # confirm a known 1.5x gap
            length = max(per_use, self._chain(tensors))
            runs = [make_sequence(step, tensors, length) for step in steps]
            kind = "repeated_forward"
            if self.use_library_inference:
                kind = "library_generation"
                runs = [lambda m=model: m(*tensors) for model in arms]
            elif conversation:
                kind = "advancing_cache_fixed_tokens"
                models = [ContextSequence(model, config.steps) for model in arms]
                runs = [lambda m=model: m(*tensors) for model in models]
            from backcountry_runtime.state import sequence_observation
            # Correctness is judged as in every other check, uncompiled; the
            # compiled steps above are timed only. Compiling a chain of small
            # ops changes its rounding by itself (a RoPE step: a third of its
            # outputs), so a kernel exact against the model could never match
            # the compiled model bit for bit.
            correctness_runs = [lambda model=model: sequence_observation(model, tensors, per_use)
                                for model in arms]
            reference = self._reference_model()
            check = self.session.off_clock(lambda: self._outputs_check(
                name + ":sequence", *correctness_runs,
                None if reference is None
                else lambda: sequence_observation(self._for(reference, name), tensors, per_use)))
            reference = None
            sequence_checks.append(check)
            if not check.passed:
                raise RuntimeError(f"consecutive model outputs failed for {name}: {check.reason}")
            warm = [chained_steps(step, tensors, self._chain(tensors)) for step in steps]
            row = compare_sequences(self.session, *runs, *warm, pairs=config.pairs,
                                    warmup_steps=config.warmup_steps, label=name)
            timings[name] = comparison_from_samples(row["timing"]["baseline_ms"],
                                                    row["timing"]["candidate_ms"])
            row["steps"] = length if kind == "repeated_forward" else config.steps
            row["workload_kind"] = kind
            row.update(generation_throughput(row))
            row["prefix_copy_included"] = conversation
            results[name] = row
            self.log.append("sequence_comparison", workload=name, **row)
            self.report.final.setdefault("sequences", {})[name] = row
            self.report.write(self.work_dir / "report.json")
        return results, self._sequence_result(sequence_checks, timings)

    def _sequence_result(self, checks, timings) -> E2EResult:
        """The consecutive-step comparison under the single-step final's rule:
        a difference under FINAL_MARGIN of the step is neither a win nor a loss,
        and over a range FINAL_RANGE_WINS sizes must win. Without it a 1.5% win
        at one of three request counts shipped Bonsai 2 9B kernels that ran
        LM Studio at 1.00x at one and two requests."""
        wins = min(FINAL_RANGE_WINS, len(timings)) if self.report.constants.get("search_sizes") else 1
        return E2EResult(checks=checks, veto=next(iter(timings.values()), None), workload_vetos=timings,
                         min_wins=wins, margin_share=FINAL_MARGIN)

    def emit_artifact(self, out_dir: str | Path) -> Path:
        if getattr(self, "_kernel_job", None) is not None:
            return self._kernel_job.emit_artifact(out_dir)  # the kernels, searched on the loop's calls
        self._phase("exporting and checking artifact")
        try:
            out = self._emit_artifact(out_dir)
            self.report.session.update(status="complete", artifact=str(out))
            self.report.write(self.work_dir / "report.json")
            self.report.write(out / "report.json")
            if not (out / "run.py").is_file() and not json.loads((out / "swap_table.json").read_text() or "[]"):
                # its loop dropped at the fresh-process check and no kernel: nothing for an app to use
                self.log.append("artifact_empty", path=str(out))
                return out
            # where apply(model) finds it, named for its job (work-.../artifact)
            from backcountry_runtime.apply import register
            stored = register(out, out.parent.name if out.name == "artifact" else out.name)
            self.log.append("artifact_stored", path=str(stored))
            # a marked function's result, beside the file that defines it
            from backcountry_runtime.tune import publish
            if (published := publish(out, self.manifest.model_path)) is not None:
                self.log.append("artifact_published", path=str(published))
                print(f"result: {published} (the marked function uses it from there; commit or ship it with "
                      "the code, or delete it to remove it)", flush=True)
            return out
        except BaseException as error:
            self.report.session.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=f"{type(error).__name__}: {error}")
            self.log.append("job_failed", stage="exporting and checking artifact",
                            reason=self.report.session["error"])
            self.report.write(self.work_dir / "report.json")
            raise

    def _reference_model(self, states=None, pending: tuple[str, str] | None = None, untraced=None):
        """The untouched model with every changing cut recomputed in fp32
        (bind.fp32_region), at every recorded size: what an inexact patched
        model is graded against. states: the installation to mirror (default
        the installed one); pending: (fingerprint, assoc tag) of a cut being
        promoted; untraced: label -> inputs at sizes nothing recorded (the
        final check's fresh sizes), recorded here so the reference covers
        them. None when every cut is exact."""
        from .bind.fp32_region import NoReference, reference_definitions
        changing = {fp for fp, tag in self.shipped_tags.items() if tag == "changing"}
        if pending is not None:
            changing.discard(pending[0])
            if pending[1] == "changing":
                changing.add(pending[0])
        if not changing:
            return None
        extra = dict(self.sweep_traces)
        for w in self.manifest.workloads:
            labels = [label for label in getattr(self, "serve_tensors", {}) if label.startswith(f"{w.name}@serve:")]
            extra.update(zip(labels, self.serve_traces.get(w.name, [])))
        if untraced:
            was_installed = self.tracer.patcher.installed
            if not was_installed:
                self.tracer.install()
            try:
                for label, tensors in untraced.items():
                    extra[label], _ = self.tracer.trace(self._for(self.baseline_model, label), tensors)
            finally:
                if not was_installed:
                    self.tracer.uninstall()
        try:
            definitions = reference_definitions(self.traces, extra, self.installed if states is None else states,
                                                changing)
        except NoReference as error:
            self.log.append("no_fp32_reference", reason=str(error))
            return None
        emitted = {path: self._emit_installation(defs, f"Ref_{_safe(path)}", replay=True)
                   for path, defs in definitions.items()}
        kernels = {path: {c.kernel.kernel_id: c.kernel for _t, _s, cuts in defs for c in cuts}
                   for path, defs in definitions.items()}
        return self._copy_incumbent(emitted, kernels=kernels)

    def _outputs_check(self, name, baseline_run, patched_run, reference_run=None, exact=None):
        """One whole-model output comparison under the job's numeric policy:
        bitwise while every cut is exact; with a changing cut, graded against
        the fp32 reference when there is one, else the manifest tolerances."""
        from .e2e import reference_check
        exact = self._requires_exact() if exact is None else exact
        if not exact and reference_run is not None:
            return reference_check(baseline_run, patched_run, reference_run, name)
        return preserving_check(baseline_run, patched_run, name, exact=exact,
                                tolerances=self.manifest.tolerances)

    def _generated(self) -> bool:
        """The job's calls read generated tokens (a generated workload, under exact_rows)."""
        return bool(getattr(getattr(self, "context", None), "generated", False))

    def _exact_only(self) -> bool:
        """Every edit must be bit-identical: the job's calls read generated
        tokens, or its manifest sets no tolerances (tolerance is opt-in)."""
        return self._generated() or not self.manifest.tolerance_allowed

    def _requires_exact(self, pending_tag: str | None = None, pending_region: str | None = None):
        """Numerical policy follows installed edits, including a pending replacement;
        a job that allows no tolerance always requires exact bits."""
        if self._exact_only():
            return True
        if self.report.constants.get("output_repeats") is False:
            # the untouched model is not bit-repeatable (Kokoro, 1e-7 apart):
            # no exact bits to preserve, so every edit is graded within the
            # tolerance the user chose (_measure_repeatability)
            return False
        return pending_tag != "changing" and not any(
            tag == "changing" for region, tag in self.shipped_tags.items()
            if pending_region is None or region != pending_region)

    def _model_bundle(self) -> ModelBundle:
        """The model source, traced inputs and final-check settings the artifact
        carries so it loads, validates and re-times without the harness."""
        context = None if self.context is None else {
            "workload": self.context.name, "context": self.context.context,
            "seed": self.context_seed, "tokens": self.context_tokens, "generated": self.context.generated}
        return ModelBundle(self.manifest.model_path, self.baseline,
                           {**self.tensors, **self.sweep_tensors, **getattr(self, 'serve_tensors', {})}, self.manifest.workloads,
                           asdict(self.manifest.final_benchmark), context=context,
                           exact=self._requires_exact(), tolerances=self.manifest.tolerances,
                           baseline_wrappers=getattr(self, "baseline_wrappers", []),
                           use_library_inference=bool(self.use_library_inference),
                           app_run=bool(self.report.constants.get("app_run")),
                           checkpoint_pins=getattr(self, "checkpoint_pins", None))

    def _layer_signatures(self) -> dict:
        """Each replaced layer as it was before its wrapper: what apply() checks
        a user's model against, layer by layer, before replacing it."""
        from backcountry_runtime.apply import layer_signature
        return {path: layer_signature(state[0]) for path, state in self.installed.items()
                if isinstance(state, tuple)}

    def _write_goldens(self, bundle, cases) -> None:
        """A changing cut ships graded against its fp32 reference: the bundle
        carries the reference's outputs at every saved size, each written as
        it is computed. A method of its own, so the reference model (sharing
        the job's weights) and everything made from it die when it returns: a
        closure here kept Gemma 4's whole model alive through the fresh check."""
        reference = self._reference_model()
        if reference is None:
            return
        from backcountry_runtime.state import correctness_call
        from .artifact.validate import _observe
        # each written as it is computed: every size's logits held at once
        # took Gemma 4's export to 23 GB on a 19 GB GPU
        spill = self.work_dir / "export_goldens"
        spill.mkdir(exist_ok=True)

        def written(outputs, path):
            mx.save_safetensors(str(path), {f"o{k}": t for k, t in enumerate(outputs)})
            return path
        for i, (name, tensors, _) in enumerate(cases):
            bundle.goldens[name] = self.session.off_clock(
                lambda: written(_observe(lambda: correctness_call(self._for(reference, name), tensors))[0],
                                spill / f"{i}.safetensors"),
                where=f"fp32 reference {name}")
        if self.context is not None and isinstance(reference, ContextStep):
            # a decode bundle replays its consecutive steps against the
            # reference's own trajectory (benchmark.py), at the measured length
            name = self.context.name
            steps = (self.report.final.get("sequences", {}).get(name, {}).get("steps")
                     or self.manifest.final_benchmark.steps)
            step = reference
            bundle.sequence_goldens[name] = self.session.off_clock(
                lambda: written(_observe(lambda: step.sequence(self.tensors[name], steps))[0],
                                spill / "sequence.safetensors"),
                where=f"fp32 reference {name} over {steps} steps")

    def _emit_artifact(self, out_dir: str | Path) -> Path:
        """Write the artifact, then load it onto a fresh build() in a fresh
        process and check it reproduces the patched model."""
        model_level = self.report.model_level.get("final", {})
        if not self.final_ok and not (model_level.get("ok") and not self.emitted):
            raise RuntimeError("the final whole-model check failed; no artifact is written "
                               "for a patched model that does not match the original")
        specs: dict[str, KernelSpec] = {}
        # a block's own method calls kernels that carry their shape guard:
        # where blocks share a kernel, the guarded spec is the one shipped
        for path in sorted(self.emitted, key=lambda p: p in self.live_scopes):
            # no name may keep a wrapper past this loop: it holds its scope's
            # weights (LFM2.5's served model.model held all 13.6 GB) through
            # the fresh-process check below
            specs.update(getattr(_resolve(self.model, path), "_specs", {}))
        # Fresh export comparisons build their own original with shared weights.
        cases = [(name, tensors, []) for name, tensors in
                 list(self.tensors.items()) + sorted(self.sweep_tensors.items())
                 + sorted(getattr(self, 'serve_tensors', {}).items())]
        context = None if self.context is None else (
            self.context.context, self.context_tokens, self.tensors[self.context.name], self.context.generated,
            self.context.name)
        bundle = self._model_bundle()
        self._write_goldens(bundle, cases)
        layers = self._layer_signatures()

        def check(staged):
            # written into the bundle: about 2 GB on Qwen3.8-27B, not to be
            # held while the fresh process builds the model again
            bundle.goldens.clear()
            bundle.sequence_goldens.clear()
            gc.collect()
            mx.clear_cache()
            check_apply_many(staged, self.manifest.model_path, cases, context=context)
            if (staged / "run.py").is_file():
                try:
                    check_model_run(staged, self.manifest.model_path,
                                    self.work_dir / "model_level_check.safetensors", self.report.model_level)
                except RuntimeError as error:
                    # the loop did not hold outside the job (a tool fault: the job checks it as
                    # it ships): the bundle ships without it, and says why, not a failed job
                    drop_model_run(staged)
                    self.report.model_level["not_shipped"] = str(error)[:1000]
                    self.log.append("loop_not_shipped", reason=str(error)[:1000])
        # The check below builds the model again in a fresh process. The job is
        # done with its own copy; a model half this Mac's memory cannot be held
        # twice (gpt-oss-20b's check failed while the job still held 11 GB).
        # The installed scopes' original modules hold their weights too:
        # LFM2.5's expert blocks kept 12.9 of its 13.6 GB with the model dropped.
        # Whatever still holds a weight is found here, before a second copy loads.
        weights = [weakref.ref(a) for _, a in tree_flatten(getattr(self.baseline_model, "parameters", dict)())]
        total = sum(r().nbytes for r in weights)
        # an engine that keeps the model behind a running thread (LM Studio's kit)
        # is not freed by dropping it: the model file's release() shuts down every
        # one it built (the job builds the model more than once)
        release = getattr(self.model_module, "release", None)
        if callable(release):
            release()
        self.model = self._pristine = self.baseline_model = self._compiled_baseline = None
        self.installed = {}
        gc.collect()
        mx.clear_cache()
        self.session.memory("before the fresh-process artifact check")
        held = sum(a.nbytes for a in (r() for r in weights) if a is not None)
        if held and held >= HELD_WEIGHTS_SHARE * total:
            raise RuntimeError(f"the job still holds {held / 1e9:.2f} of its model's {total / 1e9:.2f} GB of "
                               "weights, and the fresh-process check would load a second copy")
        self.session.wait_ready()
        out = emit_artifact(
            out_dir, list(specs.values()), list(self.emitted.values()), self.report,
            validate=check,
            bundle=bundle, layers=layers)
        shutil.rmtree(self.work_dir / "export_goldens", ignore_errors=True)  # copied into the artifact
        self.log.append("artifact_checked", artifact=str(out), workloads=[c[0] for c in cases])
        self._time_untimed_fresh(out)
        return out

    def _time_untimed_fresh(self, artifact_dir) -> None:
        """The workloads too close to the GPU's limit to time in the job
        (_memory_fit), timed after export the way a user would: a fresh process
        holding only the model, runtime compare() switching the artifact's
        layers on one copy of the weights. For the report only: it decides
        nothing, and a failure is recorded, never raised."""
        untimed = [w.name for w in self.manifest.workloads if not _fits(self, w.name)]
        if not untimed or self.report.final is None:
            return
        from .artifact.emit import time_in_fresh_process
        self.session.wait_ready()
        try:
            timings = time_in_fresh_process(artifact_dir, self.manifest.model_path,
                                            [(name, self.tensors[name]) for name in untimed])
        except Exception as error:  # the bundle is already made and checked
            timings = {name: {"error": f"{type(error).__name__}: {str(error)[:400]}"} for name in untimed}
        self.report.final["fresh_process_timings"] = timings
        self.log.append("fresh_process_timings", timings=timings)
        self.report.write(self.work_dir / "report.json")


# The smallest gain worth a long search: under it, a user would not notice.
WORTHWHILE_SPEEDUP = 1.10


def little_room(floors: dict) -> str | None:
    """A warning when even perfect kernels could not make any workload
    WORTHWHILE_SPEEDUP faster (each step against its physics floor), else
    None. A warning, not a stop: the floor is an upper bound for kernels,
    and a loop search (identical: output) changes the work itself."""
    bounds = {name: 1 / (1 - f["room"]) for name, f in floors.items() if f.get("room") is not None and f["room"] < 1}
    if not bounds or max(bounds.values()) >= WORTHWHILE_SPEEDUP:
        return None
    each = ", ".join(f"{name} {bound:.2f}x" for name, bound in sorted(bounds.items()))
    return (f"even perfect kernels could make this model at most this much faster ({each}), under "
            f"{WORTHWHILE_SPEEDUP:.2f}x: each step already runs close to what the chip can do (reading its "
            "weights or doing its arithmetic), so a kernel search alone cannot reach a noticeable gain here")


def _in_spans(to_dim, size: int, spans) -> bool:
    """Whether a scope size falls in any (lo, hi) span of the manifest dim."""
    if not spans or to_dim is None:
        return False
    value = to_dim[0] * size + to_dim[1]
    return any(lo <= value <= hi for lo, hi in spans)


def _layer_size(template, arrays) -> int | None:
    """The size n at which these recorded inputs have the template's shapes
    (an input dim a * n + b), or None when they fit none."""
    from .ladder.sizes import shape_at
    shapes = [[list(a.shape), str(a.dtype).removeprefix("mlx.core.")] for a in arrays.values()]
    for (dims, _dtype), (shape, _) in zip(template, shapes):
        for d, have in zip(dims, shape):
            if not isinstance(d, int) and d[0] and (have - d[1]) % d[0] == 0:
                n = (have - d[1]) // d[0]
                return n if [[list(sh), dt] for sh, dt in shape_at(template, n)] == shapes else None
    return None


def _state_marks(trace: Trace) -> str:
    """Why this step cannot be compiled from outside the model, or "" if it
    can: every recorded sign that the step is not a pure function of its
    inputs. An array the model kept after the pass, a call that changed an
    object's state (a KV cache write), or an evaluation mid-step (a branch on
    a value): mx.compile breaks on each."""
    marks = []
    if arrays := len(trace.python_retained()):
        marks.append(f"{arrays} arrays kept")
    if calls := len(trace.state_calls()):
        marks.append(f"{calls} state calls")
    if trace.in_pass_evaluation:
        marks.append("evaluates mid-step")
    return ", ".join(marks)


def _safe(path: str) -> str:
    return path.replace(".", "_") or "root"


def _ops_view(trace: Trace, stretch: Stretch) -> list[dict]:
    """The recorded calls of one copy, with arrays named by role (in0, out0,
    t<seq> for intermediates) and every non-tensor argument spelled out."""
    names = {aid: f"in{i}" for i, aid in enumerate(stretch.input_ids)}
    names.update({aid: f"out{i}" for i, aid in enumerate(stretch.output_ids)})
    view = []
    for node in trace.nodes[stretch.start_seq:stretch.end_seq + 1]:
        for output, aid in enumerate(node.out_arrays):
            names.setdefault(aid, f"t{node.seq}_{output}")

        def name(obj, node=node):
            if isinstance(obj, ArrayRef):
                return names[node.in_arrays[obj.index]]
            if isinstance(obj, (list, tuple)):
                return [name(v) for v in obj]
            if isinstance(obj, dict):
                return {k: name(v) for k, v in obj.items()}
            if isinstance(obj, slice):
                return f"slice({obj.start}, {obj.stop}, {obj.step})"
            return obj if isinstance(obj, (int, float, bool, str)) or obj is None else str(obj)

        view.append({
            "op": node.op,
            "args": name(list(node.scalar_args["args"])),
            "kwargs": name(dict(node.scalar_args["kwargs"])),
            "outputs": [names[aid] for aid in node.out_arrays],
        })
    from .judge.prompts import diagnostic_metadata
    return diagnostic_metadata(view)


def _ratio(clock) -> float | None:
    """A kernel's time over the library's, both from the same clock."""
    get = clock.get if isinstance(clock, dict) else lambda k: getattr(clock, k)
    region_ms, library_ms = get("region_ms"), get("library_ms")
    return (region_ms / library_ms) if region_ms and library_ms else None


def _item_view(item) -> dict:
    return {"id": item.id, "kind": item.kind, "assoc_tag": item.assoc_tag,
            "hypothesis": item.hypothesis}


def _declines(trace, cut) -> bool:
    """Whether the cut's kernel hands these recorded shapes to the original
    ops: its fallback predicate on the cut's input shapes in the trace."""
    if trace is None or cut.kernel.fallback_predicate is None:
        return False
    from backcountry_runtime.grammar import Expr
    specs = trace.span_specs(cut.start_seq, cut.end_seq)
    return bool(Expr(cut.kernel.fallback_predicate).evaluate([tuple(specs[a][0]) for a in cut.input_ids]))


def _fits(job, name: str) -> bool:
    """Whether a whole-model clock may time this workload (_memory_fit);
    over a range a size at or past an untimed size of its workload does not."""
    untimed = getattr(job, "_untimed", ())
    if name in untimed:
        return False
    base, at, rest = name.partition("@")
    if not at or "=" not in rest:
        return True
    size = int(rest.split("=", 1)[1].split(":", 1)[0])
    return not any(u.partition("@")[0] == base and "=" in u and int(u.split("=", 1)[1]) <= size
                   for u in untimed)


def _spread(items: list, count: int) -> list:
    """count of items evenly from the first to the last; all of them when
    there are no more."""
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def _verdict_payload(hyp_id: str, kernel_id: str, outcome: str, result, sizes=None) -> dict:
    return {
        "hypothesis_id": hyp_id, "kernel_id": kernel_id, "outcome": outcome,
        "failed_gate": result.failed_gate, "detail": _safe_detail(_without_raw_sizes(result.detail)),
        "region_ms": result.region_ms, "library_ms": result.library_ms,
        "win_ms": result.win_ms, "sigma_ms": result.sigma_ms, "floor_ms": result.floor_ms,
        **({"sizes": sizes} if sizes else {}),
    }


def _without_raw_sizes(detail: dict | None) -> dict | None:
    """The ladder's per-size clock rows are shown as the sizes table; the raw
    rows (keyed by recording label) would show the same numbers twice."""
    return {k: v for k, v in detail.items() if k != "sizes"} if detail else detail


def _short_reason(detail: dict | None) -> str:
    """The first thing worth reading in a failed gate's detail, on one line."""
    if not detail:
        return ""
    if detail.get("failures"):
        f = detail["failures"][0]
        return f"{f.get('check')}: {f.get('detail')}"
    if detail.get("diagnostics"):
        d = detail["diagnostics"][0]
        return f"line {d.get('body_line')}: {d.get('message')}"
    parts = [str(detail[k]) for k in ("regime", "kind", "reason", "note") if detail.get(k)]
    return "; ".join(parts)[:160]


def _region_line(region: Region, event: str, run: "RegionRun | None" = None) -> str:
    roof = region.roofline
    rep = ", ".join(f"{w}={ms:.4f}" for w, ms in region.t_rep_ms.items())
    if event == "open":
        how = f"copies={region.copies}\tlibrary ms per copy: {rep}"
        if roof:
            how += f"\troofline ms per copy: {roof.t_roofline_ms:.4f}"
    else:
        shipped = run.shipped.kernel_id if run and run.shipped else "none"
        how = f"{run.close_rule}\thypotheses={run.hypotheses}\tshipped={shipped}"
    return f"{wall_now()}\t{region.fingerprint[:8]}\tregion_{event}\t{' > '.join(region.ops)}\t{how}"



def _tag(kernel: KernelSpec) -> str:
    """The rule a harness-built kernel is checked under: bitwise, unless its
    lowering adds in its own order, which is a changing kernel like any other."""
    return "changing" if kernel.reassociates else "preserving"


def _kernel_id(region: Region, tag: str) -> str:
    """Full region identity keeps globally stored kernels distinct."""
    return f"r{region.fingerprint}_{tag}"


def _in_branch(run: RegionRun, kernel_id: str | None, root: str | None) -> bool:
    """Whether kernel_id is root or descends from it through recorded parents."""
    seen = set()
    while kernel_id is not None and kernel_id not in seen:
        if kernel_id == root:
            return True
        seen.add(kernel_id)
        kernel_id = run.attempts.get(kernel_id, {}).get("parent")
    return False


def resolve_parent(run: RegionRun, name: str) -> KernelSpec | None:
    """The kernel a proposal edits: head, scaffold, shipped, a hypothesis id,
    or a kernel id; None when the name matches nothing in this region."""
    by_role = {"head": run.head, "scaffold": run.scaffold, "shipped": run.shipped}
    if name in by_role:
        return by_role[name]
    if name in run.kernels:
        return run.kernels[name]
    return run.kernels.get(_kernel_id(run.region, name))


def kernel_from_proposal(contract: RegionContract, parent: KernelSpec | None, proposal,
                         kernel_id: str) -> KernelSpec:
    """The harness owns names, dtypes, and the call site; the judge supplies
    the body and the launch. A header or template left out means the parent's."""
    scratch = tuple(proposal.scratch)
    header = proposal.header or (parent.header if parent else "")
    native = parent.native_call if parent else None
    stages = tuple(replace(stage, kernel=replace(stage.kernel,
                   kernel_id=f"{kernel_id}_stage{i}", name=f"at_{kernel_id}_stage{i}",
                   header=stage.kernel.header or header))
                   for i, stage in enumerate(proposal.stages))
    atomic_outputs = False
    if parent is not None and not stages:
        atomic_outputs = native["factory"].get("atomic_outputs", False) if native else parent.atomic_outputs
    return KernelSpec(
        kernel_id=kernel_id,
        name=f"at_{kernel_id}",
        input_names=contract.input_names,
        output_names=contract.output_names + tuple(s[0] for s in scratch),
        source=proposal.source,
        header=header,
        grid=tuple(proposal.grid),
        threadgroup=tuple(proposal.threadgroup),
        output_shapes=(tuple(tuple(s) for s in proposal.output_shapes)
                       + tuple(tuple(s[2]) for s in scratch)),
        output_dtypes=contract.output_dtypes + tuple(s[1] for s in scratch),
        template=() if stages else tuple(proposal.template) or (parent.template if parent else ()),
        fallback_predicate=proposal.fallback_predicate,
        native_call=native,
        input_signature=contract.input_signature,
        input_signatures=contract.input_signatures,
        ensure_row_contiguous=parent.ensure_row_contiguous if parent else True,
        atomic_outputs=atomic_outputs,
        stages=stages,
    )


def _kernel_view(spec: KernelSpec) -> dict:
    """What the judge may see of a kernel: its source and launch story.
    Native call-site settings are visible as read-only context, never proposal fields."""
    from .judge.prompts import diagnostic_metadata
    return {
        "fallback_predicate": spec.fallback_predicate,
        "native_call": diagnostic_metadata(spec.native_call),
        "input_signature": spec.input_signature,
        "input_signatures": spec.input_signatures,
        "reference_sequence": diagnostic_metadata(spec.reference_sequence),
        "source": spec.source,
        "header": spec.header,
        "grid": list(spec.grid),
        "threadgroup": list(spec.threadgroup),
        "output_shapes": [list(s) for s in spec.output_shapes],
        "output_dtypes": list(spec.output_dtypes),
        "template": [list(t) for t in spec.template],
        "stages": [{"inputs": list(stage.inputs), "outputs": list(stage.outputs),
                    **_kernel_view(stage.kernel)} for stage in spec.stages],
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
    }


# the acceptance envelope: values the judge could tune an edit to sit under
def _safe_detail(detail: dict | None) -> dict:
    """Judge-visible gate detail: distances like max_excess only, never the
    acceptance envelope. The judge may learn how far a check missed, never
    where the line is."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items()
                    if not isinstance(k, str) or k.lower() not in
                    ENVELOPE_KEYS | {"reference", "candidate"}}
        if isinstance(value, (list, tuple)):
            return [strip(v) for v in value]
        return value
    from .judge.prompts import diagnostic_metadata
    return diagnostic_metadata(strip(detail)) if detail else {}


REFERENCE_CHARS = 60_000  # a longer recording is sent as the lines around each kernel call
REFERENCE_WINDOW = 30     # lines kept on each side of a kernel call
REFERENCE_SITES = 2       # calls shown per kernel; the rest repeat the pattern


def _reference_excerpt(source: str, kernel_ids) -> str:
    """A recording short enough to send: whole when it is, else its
    __init__ (which names every weight), each kernel's first calls with the
    lines around them, and its returns. A block whose layers repeat (a whole
    model) shows each kernel's pattern without every layer."""
    if len(source) <= REFERENCE_CHARS:
        return source
    lines = source.split("\n")
    call = next((i for i, line in enumerate(lines) if line.lstrip().startswith("def __call__")), 0)
    keep = set(range(call + 1))
    seen = Counter()
    for i, line in enumerate(lines):
        kid = next((k for k in kernel_ids if f"_specs['{k}']" in line or f'_specs["{k}"]' in line), None)
        if kid is not None and seen[kid] < REFERENCE_SITES:
            seen[kid] += 1
            keep |= set(range(max(call, i - REFERENCE_WINDOW), min(len(lines), i + REFERENCE_WINDOW + 1)))
        elif line.lstrip().startswith("return "):
            keep.add(i)
    out, gap = [], 0
    for i, line in enumerate(lines):
        if i in keep:
            if gap:
                out.append(f"        # ... {gap} lines left out: the same pattern continues")
            out.append(line)
            gap = 0
        else:
            gap += 1
    return "\n".join(out)


def _parts_source(block) -> dict:
    """The source of each class a block's own code calls (its submodules,
    outside mlx.nn): a kernel that spans two of them needs both rewritten
    inline, so the judge needs their code beside the block's."""
    found = {}
    for _name, part in (block.named_modules() if callable(getattr(block, "named_modules", None)) else []):
        cls = type(part)
        if part is block or cls.__module__.startswith("mlx.nn") or cls.__qualname__ in found:
            continue
        source = _class_source(cls, sys.modules.get(cls.__module__))
        if source is not None:
            found[cls.__qualname__] = source
    return found


def _prefill_chunk() -> int:
    """The most prompt tokens one MLX-LM generation call reads."""
    import inspect
    from mlx_lm.generate import generate_step
    return inspect.signature(generate_step).parameters["prefill_step_size"].default


def _class_source(cls, model_module=None) -> str | None:
    """A class's source: inspect's, or read from the file its module came from
    (a model file loaded by path is not importable by name, so inspect cannot
    find it)."""
    import ast
    import inspect
    import sys
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError):
        pass
    for module in (sys.modules.get(cls.__module__), model_module):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                return ast.get_source_segment(text, node)
    return None


def _load_class(emitted: EmittedWrapper):
    from .bind.emit import MODULE_HEADER

    ns: dict = {}
    exec(compile(MODULE_HEADER + emitted.source, f"<wrapper {emitted.class_name}>", "exec"), ns)
    return ns[emitted.class_name]


def _resolve(model, path: str):
    from backcountry_runtime.swap import resolve

    _, _, child = resolve(model, path)
    return child


def run_job(manifest_path, work_dir, judge_factory, session=None) -> Report:
    return JobRunner(manifest_path, work_dir, judge_factory, session=session).run()
