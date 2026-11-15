"""One kernel at many sizes of one dimension, inside one sandbox child.

The parent serves a shipped kernel at other sizes only where this job found
it right (phase "validate": with shader validation off, outputs poisoned and
bitwise equal to the recorded operations replayed at that size, or within
allowed_error of those operations for a kernel within tolerance, and three
runs bitwise equal at the sizes in repeat_at; with it on, run for its
out-of-bounds reports alone, addresses_only: what validation computes is not
what ships, and a kernel can be right only with it on) and faster (phase
"score": the region clock's paired loop and margin at each size of a grid).

Inputs at a size come from a real pass of the model at that size
(recordings), or, for a region whose inputs are all floating point, from the
first rows of a longer real pass (cut_from): a causal model's first n rows of
a long prompt are what an n-token prompt computes. A size with neither is not
checked, so not served. Integer inputs are never cut: an index list keeps its
values, which can point past the cut (Qwen3-30B-A3B's unsort order: an index
of 16,254 among 1,032 rows, which made a correct kernel write out of bounds);
and nothing is built or repeated up (gpt-oss groups each expert's tokens).
Timings use real passes only. A recording holds only the inputs that change with the
size; the weights come from the search's own recording (inputs_path). A pass
can record several copies of the region (gpt-oss's expert matmul runs 72
times a pass), each saved as its own set beside the first: every set is
checked, since a kernel off the library's rounding at a size can still match
it on one copy (gpt-oss at 124 rows: about one copy in ten).

With shader validation on, the worker writes SIZE_MARKER and the size to its
stderr before each size and waits for the GPU after it, so the parent can tell
which size a validation report belongs to.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from backcountry.harness.ladder.numeric import max_abs_diff
from backcountry.harness.measure.clocks import (CLOCK_TARGET_MS, chained_loop, comparison_from_samples,
                                      link_input, loop_iterations, paired_means, sample_group,
                                      timing_sets)
from backcountry.harness.measure.heat import Heat
from backcountry.harness.measure.session import Session
from backcountry.harness.regions.store import load_set
from backcountry.harness.sandbox.poison import saturate_pool
from backcountry.harness.sandbox.protocol import SIZE_MARKER, Verdict
from backcountry.harness.sandbox.watchdog import gpu_window
from backcountry.harness.trace.replay import prepare_replay, replay
from backcountry.harness.trace.serialize import node_from_dict
from backcountry_runtime.exact import bitwise_equal
from backcountry_runtime.kernels import KernelSpec, try_sized

RUNS = 3  # as the determinism gate


@dataclass(frozen=True)
class SizesSpec:
    kernel: dict                      # KernelSpec fields
    nodes_json: str                   # the region's recorded operations
    input_ids: tuple[int, ...]        # kernel input order
    output_ids: tuple[int, ...]
    inputs_path: str                  # the search's recorded inputs at the primary size, weights included
    shapes: list                      # per input [[dim or [a, b]], dtype]: a * n + b
    sizes: tuple[int, ...]
    phase: str                        # "validate" | "score"
    weight_inputs: tuple[bool, ...] = ()
    settings: list = ()               # [[path, a, b]]: integers in the recorded ops that follow the size
    allowed_error: float | None = None  # None: bitwise; else the largest max_abs from the recorded ops
    recordings: dict = dataclasses.field(default_factory=dict)  # str(size) -> size-dependent inputs from a real pass there
    cut_from: list = ()               # [size, recording]: a longer real pass whose first rows serve smaller sizes
    addresses_only: bool = False      # validate: the kernel once per copy, for shader validation's reports alone
    copies: int = 0                   # recorded copies checked per size, first ones first; 0: every one
    repeat_at: tuple | None = None    # sizes run RUNS times and compared; elsewhere once; None: every size
    clock_pairs: int = 8
    defer_cooling: bool = False
    kind: str = "sizes"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(text: str) -> "SizesSpec":
        d = json.loads(text)
        if d.get("kind") != "sizes":
            raise ValueError(f"not a sizes spec: kind={d.get('kind')!r}")
        for key in ("input_ids", "output_ids", "sizes", "weight_inputs"):
            d[key] = tuple(d.get(key, ()))
        return SizesSpec(**d)


def shape_at(template, n: int) -> list:
    return [[[d if isinstance(d, int) else d[0] * n + d[1] for d in dims], dtype] for dims, dtype in template]


def nodes_at(nodes_json: str, settings, n: int) -> list:
    """The recorded operations with each integer that follows the size set
    for size n."""
    nodes = json.loads(nodes_json)
    for path, a, b in settings:
        *parents, last = path
        holder = nodes
        for key in parents:
            holder = holder[key]
        holder[last] = a * n + b
    return [node_from_dict(d) for d in nodes]


def evaluate_sizes(spec: SizesSpec) -> Verdict:
    """The correctness sweep takes no cooling pauses: those keep the chip's
    temperature steady for timing, and nothing here is timed. The clock phase
    paces like every other clock."""
    session = Session(heat=Heat.for_machine())
    started = time.perf_counter()
    try:
        return _evaluate(spec, session)
    finally:
        mx.synchronize()
        if spec.phase == "score":
            session._debt_s += max(0.0, time.perf_counter() - started - session.idled_s - session.work_s)
            session.settle()


def _evaluate(spec: SizesSpec, session: Session) -> Verdict:
    kspec = KernelSpec.from_json(json.dumps(spec.kernel))
    recorded = load_set(spec.inputs_path)
    n_out = len(spec.output_ids)
    poison = 0.0 if kspec.atomic_outputs else float("nan")
    results: dict[str, object] = {}
    errors: dict[str, float] = {}

    cut = tuple(spec.cut_from)
    cut_sets = []  # the longest recording, read once: Qwen3.8-27B's is 4 copies of 71 MB

    def has(n):
        return str(n) in spec.recordings or bool(cut) and n <= cut[0]

    def binds_at(n):
        """The inputs from the real pass at size n, one per set recorded
        there, else the first rows of each set of cut_from's, or None."""
        if not has(n):
            return None
        found, own = [], str(n) in spec.recordings
        if not own and not cut_sets:
            cut_sets.extend({**recorded, **load_set(path)} for path in recording_sets(cut[1]))
            mx.eval([a for source in cut_sets for a in source.values()])
        sources = [{**recorded, **load_set(path)}
                   for path in recording_sets(spec.recordings[str(n)])[:spec.copies or None]] if own else cut_sets
        for source in sources[:spec.copies or None]:
            binds = {aid: source[aid] for aid in spec.input_ids}
            if not own:
                binds = {aid: _first_rows(binds[aid], dims, cut[0], n)
                         for aid, (dims, _dtype) in zip(spec.input_ids, spec.shapes)}
            for aid, (shape, _dtype) in zip(spec.input_ids, shape_at(spec.shapes, n)):
                if list(binds[aid].shape) != shape:
                    raise ValueError(f"the recording at size {n} has input {aid} shaped "
                                     f"{list(binds[aid].shape)}, not {shape}")
            mx.eval(list(binds.values()))
            found.append(binds)
        return found

    if spec.phase == "validate":
        recorded_sizes = [n for n in spec.sizes if has(n)]
        poisoned = []
        if recorded_sizes and not spec.addresses_only:
            largest = max(recorded_sizes)
            poisoned = [a.nbytes for a in _reference(nodes_at(spec.nodes_json, spec.settings, largest),
                                                     binds_at(largest)[0], spec.output_ids)]
            saturate_pool(poisoned)
        for n in spec.sizes:
            sets = binds_at(n)
            if sets is None:
                results[str(n)] = "no real pass of the model recorded at this size"
                continue
            os.write(2, f"{SIZE_MARKER}{n}\n".encode())
            results[str(n)] = None
            for copy, binds in enumerate(sets):
                inputs = [binds[i] for i in spec.input_ids]
                if spec.addresses_only:
                    with gpu_window():
                        outs = try_sized(kspec, inputs)
                        if outs is not None:
                            mx.eval(outs[:n_out])
                        mx.synchronize()  # its validation reports land before the next size's marker
                    del outs, inputs
                    mx.clear_cache()
                    continue
                with gpu_window():
                    reference = _reference(nodes_at(spec.nodes_json, spec.settings, n), binds, spec.output_ids)
                    runs = []
                    for _ in range(RUNS if spec.repeat_at is None or n in spec.repeat_at else 1):
                        outs = try_sized(kspec, inputs, init_value=poison)
                        if outs is None:
                            break
                        outs = outs[:n_out]
                        mx.eval(outs)
                        runs.append(outs)
                    mx.synchronize()  # its validation reports land before the next size's marker
                if runs and spec.allowed_error is not None:
                    error = max(max_abs_diff(a, b) for a, b in zip(runs[0], reference, strict=True))
                    errors[str(n)] = max(error, errors.get(str(n), error))
                of = f" (recorded copy {copy + 1} of {len(sets)})" if len(sets) > 1 else ""
                if not runs:
                    results[str(n)] = "the kernel's own fallback covers this size"
                elif spec.allowed_error is not None and not errors[str(n)] <= spec.allowed_error:
                    results[str(n)] = (f"differs from the recorded operations by {errors[str(n)]:.3g}, "
                                       f"more than {spec.allowed_error:.3g}{of}")
                elif spec.allowed_error is None and not all(bitwise_equal(a, b) for a, b
                                                            in zip(runs[0], reference, strict=True)):
                    results[str(n)] = f"differs from the recorded operations{of}"
                elif not all(bitwise_equal(a, b) for run in runs[1:] for a, b in zip(runs[0], run)):
                    results[str(n)] = f"differs between repeated runs{of}"
                # MLX keeps a buffer of every size it has freed: over a batch of 256
                # sizes that grew past 10 GB beside LFM2.5's 13.6 GB model, and the
                # job hit a GPU timeout. Each size starts from the poisoned pool alone.
                del reference, runs, inputs
                mx.clear_cache()
                saturate_pool(poisoned)
                if results[str(n)] is not None:
                    break
        return Verdict(True, None, ("sizes",), {"results": results, "errors": errors})

    weight_ids = {i for i, w in zip(spec.input_ids, spec.weight_inputs) if w}
    for n in spec.sizes:
        sets = binds_at(n)
        if sets is None:  # the parent times only sizes it recorded
            continue
        binds = sets[0]
        if try_sized(kspec, [binds[i] for i in spec.input_ids]) is None:
            # its own fallback rule leaves this size to the library: nothing to race
            results[str(n)] = {"declined": True, "win": False, "loss": False}
            continue
        sets = timing_sets([binds])
        link = link_input(binds, weight_ids)
        library = prepare_replay(nodes_at(spec.nodes_json, spec.settings, n), binds, spec.output_ids)
        kernel = lambda b: try_sized(kspec, [b[i] for i in spec.input_ids])
        with gpu_window():
            iters = loop_iterations(session.timed, lambda k: chained_loop(library, sets, k, link), CLOCK_TARGET_MS)
        arms = {"library": chained_loop(library, sets, iters, link),
                "candidate": chained_loop(kernel, sets, iters, link)}
        rows = {k: paired_means(v) for k, v in sample_group(session, arms, pairs=spec.clock_pairs).items()}
        comp = comparison_from_samples([v / iters for v in rows["library"]],
                                       [v / iters for v in rows["candidate"]])
        library_ms = comp.median_baseline_ms
        margin = max(0.01 * library_ms, 3.0 * comp.sigma_ms)
        results[str(n)] = {"library_ms": library_ms, "kernel_ms": statistics.median(rows["candidate"]) / iters,
                           "win": bool(comp.median_delta_ms > margin),
                           "loss": bool(comp.median_delta_ms < -margin)}
    return Verdict(True, None, ("sizes",), {"results": results})


def _first_rows(array, dims, recorded: int, n: int):
    """array, recorded at size `recorded`, cut to size n along each axis that
    follows the size (a * size + b: history at the front stays whole)."""
    index = []
    for axis, d in enumerate(dims):
        if isinstance(d, int):
            index.append(slice(None))
            continue
        a, b = d
        if array.shape[axis] != a * recorded + b:
            raise ValueError(f"the recording cut from has axis {axis} of {array.shape[axis]}, "
                             f"not {a * recorded + b}")
        index.append(slice(0, a * n + b))
    return array[tuple(index)]


def recording_sets(path: str | Path) -> list[Path]:
    """A recording and the sets saved beside it (set1, set2, ...: other copies
    of the region from the same pass), first one first."""
    first = Path(path)
    kind = first.name.partition(".")[2]
    if not first.name.startswith("set") or not kind:  # not a store set
        return [first]
    return [first] + sorted((p for p in first.parent.glob(f"set*.{kind}") if p != first), key=lambda p: p.name)


def _reference(nodes, binds, output_ids) -> list[mx.array]:
    res = replay(nodes, binds, output_ids)
    outs = [res[i] for i in output_ids]
    mx.eval(outs)
    return outs
