"""Roofline (spec "Roofline"): a region's physical speed limit on this chip.

Bytes counts only what crosses the region's boundary, never the intermediate
round trips: those are the cost fusion hopes to delete, so they cannot be part
of the ideal. Flops are estimates from op and shape; they steer pricing and
the skip decision, nothing else.

The bytes-and-launch part of the limit is measured, not computed, when the
caller has a probe clock: one launch that streams the boundary, timed beside
the region in the same paired window (measure/probe.py). A dependent kernel
pays its launch and its stream in series, and the peak from a 512 MB pass is
out of reach for a 2 MB weight, so the arithmetic overstated headroom by a
third at the sizes a decode step is made of. The flops term stays arithmetic
against the matmul peak.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Mapping

from ..measure.peaks import Peaks
from ..trace.recorder import ArrayRef
from ..trace.types import STATE_PREFIX, Trace, TraceNode
from .build import VIEW_OPS, is_view
from .types import Region, Roofline, Stretch

_DTYPE_BYTES = {
    "bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "float16": 2,
    "bfloat16": 2, "uint32": 4, "int32": 4, "float32": 4, "uint64": 8,
    "int64": 8, "complex64": 8,
}


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _bytes_of(spec: tuple[tuple[int, ...], str]) -> int:
    shape, dtype = spec
    return _numel(shape) * _DTYPE_BYTES.get(dtype, 4)


# expert matmuls: the positions of the operands indexed by expert, and of the
# index that picks them (rhs_indices)
_GATHERS = {"mx.gather_qmm": ((1, 2, 3), 5), "mx.gather_mm": ((1,), 3)}


def read_bytes(node: TraceNode) -> dict[int, int]:
    """array id -> the bytes this call reads of it. All of every input, except
    an expert matmul's experts: a call touches at most as many as its index
    has entries, 8 of 128 for one decode token, all of them for a long prompt;
    and a slice or lookup (one token's row of an embedding table), which reads
    at most what it returns."""
    read = {aid: _bytes_of(spec) for aid, spec in zip(node.in_arrays, node.in_specs)}
    if node.op in ("array.__getitem__", "mx.take") and node.in_arrays and node.out_specs:
        read[node.in_arrays[0]] = min(read[node.in_arrays[0]], _bytes_of(node.out_specs[0]))
    if node.op in _GATHERS:
        operands, at = _GATHERS[node.op]
        args, kwargs = node.scalar_args["args"], node.scalar_args["kwargs"]
        index = kwargs.get("rhs_indices", args[at] if len(args) > at else None)
        if isinstance(index, ArrayRef):
            picked = _numel(node.in_specs[index.index][0])
            for ref in (args[p] for p in operands if p < len(args)):
                if isinstance(ref, ArrayRef):
                    shape = node.in_specs[ref.index][0]
                    aid = node.in_arrays[ref.index]
                    read[aid] = read[aid] * min(picked, shape[0]) // shape[0]
    return read


def nodes_read_bytes(nodes: Iterable[TraceNode]) -> dict[int, int]:
    """array id -> the bytes these calls read of it (read_bytes)."""
    read: dict[int, int] = {}
    for node in nodes:
        read.update({aid: max(b, read.get(aid, 0)) for aid, b in read_bytes(node).items()})
    return read


def node_flops(node: TraceNode) -> float:
    """Estimated flops from op and recorded shapes. A matmul is 2*M*N*K; a
    norm a few passes over N; elementwise one per output element."""
    if node.kernel_definition is not None:
        return 0.0  # unknown arithmetic is omitted from the model lower bound, never invented
    out_elems = sum(_numel(s) for s, _ in node.out_specs)
    op = node.op
    if op in ("mx.matmul", "array.__matmul__", "mx.addmm", "mx.quantized_matmul",
              "mx.gather_qmm", "mx.block_masked_mm", "mx.gather_mm"):
        # K is the last dim of the first operand; output already holds M*N
        k = node.in_specs[0][0][-1] if node.in_specs[0][0] else 1
        return 2.0 * out_elems * k
    if op in ("mx.fast.rms_norm", "mx.fast.layer_norm"):
        return 4.0 * out_elems
    if op in ("mx.softmax", "mx.logsumexp"):
        return 5.0 * out_elems
    if op == "mx.fast.scaled_dot_product_attention":
        # q @ k^T and attn @ v: 2 * 2 * B*H*Lq*D*Lk
        q_shape = node.in_specs[0][0]
        k_shape = node.in_specs[1][0]
        lk = k_shape[-2] if len(k_shape) >= 2 else 1
        return 4.0 * _numel(q_shape) * lk
    if op in ("mx.fast.rope",):
        return 6.0 * out_elems
    if is_view(node):
        return 0.0
    if op in ("mx.sum", "mx.mean", "mx.max", "mx.min", "mx.prod", "mx.var", "mx.std"):
        return float(sum(_numel(s) for s, _ in node.in_specs))
    return float(out_elems)


def stretch_roofline(
    trace: Trace, stretch: Stretch, peaks: Peaks, t_orig_ms: float,
    floor_ms: float | None = None, rates: Mapping[str, float] | None = None,
) -> Roofline:
    """rates: the arithmetic ceiling per dtype measured in the region's own
    pricing window; the job-start peaks stand in for any dtype it lacks."""
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    spec_of = trace.span_specs(stretch.start_seq, stretch.end_seq)
    read = nodes_read_bytes(nodes)

    boundary_bytes = sum(read[aid] for aid in stretch.input_ids)
    boundary_bytes += sum(_bytes_of(spec_of[aid]) for aid in stretch.output_ids)
    launches = 1  # the ideal kernel for a region is one launch, whatever the library fires

    t_mem = boundary_bytes / (peaks.bandwidth_gbps * 1e9) * 1e3
    t_compute = compute_time_ms(nodes, peaks, rates)
    t_launch = launches * peaks.launch_us / 1e3
    if floor_ms is None:
        t_roof = max(t_mem, t_compute, t_launch)
        bound = {t_mem: "memory", t_compute: "compute", t_launch: "launch"}[t_roof]
    else:
        t_roof = max(floor_ms, t_compute)
        bound = "compute" if t_compute > floor_ms else ("memory" if t_mem >= t_launch else "launch")
    return Roofline(
        t_mem_ms=t_mem,
        t_compute_ms=t_compute,
        t_launch_ms=t_launch,
        t_roofline_ms=t_roof,
        bound=bound,
        s_max=(t_orig_ms / t_roof) if t_roof > 0 else 1.0,
        t_floor_ms=floor_ms,
    )


def step_floor(trace: Trace, peaks: Peaks, step_ms: float) -> dict:
    """The whole step against its own physics: the bytes that must come from
    outside it (weights, state the model keeps, its inputs), counted once per
    module call that reads them (a step of 16 tokens reads the weights 16
    times), plus its outputs,
    the flops of every recorded op, and the launches the library fires. Room
    is the part of the step that is not physics, the most any kernel work on
    this workload could ever take."""
    produced: set[int] = set()
    outside: dict[tuple[int, str | None], int] = {}  # (array id, the module call reading it) -> bytes
    flops, launches = 0.0, 0
    for node in trace.nodes:
        if node.seq in trace.dead:
            continue
        for aid, b in read_bytes(node).items():
            if aid not in produced:
                key = (aid, node.module_address)
                outside[key] = max(b, outside.get(key, 0))
        if node.op.startswith(STATE_PREFIX):
            # state the step reads back from memory, and the launches the
            # object's own method fired
            for aid, spec in zip(node.out_arrays, node.out_specs):
                outside.setdefault((aid, node.module_address), _bytes_of(spec))
            launches += sum(op not in VIEW_OPS and op != "array.__getitem__"
                            for op in node.scalar_args["receiver"]["inner_ops"])
        else:
            launches += not is_view(node)
        produced.update(node.out_arrays)
        flops += node_flops(node)
    specs = trace.span_specs(0, len(trace.nodes) - 1)
    total = sum(outside.values()) + sum(_bytes_of(specs[a]) for a in trace.step_outputs if a in specs)
    t_mem = total / (peaks.bandwidth_gbps * 1e9) * 1e3
    t_compute = compute_time_ms((n for n in trace.nodes if n.seq not in trace.dead), peaks)
    floor = max(t_mem, t_compute)
    return {"bytes_mb": total / 1e6, "gflop": flops / 1e9, "launches": launches,
            "t_mem_ms": t_mem, "t_compute_ms": t_compute, "floor_ms": floor,
            "step_ms": step_ms, "room": (1.0 - floor / step_ms) if step_ms > 0 else None}


def observed_peaks(peaks: Peaks, nodes: Iterable[TraceNode], measured_ms: float,
                   bytes_moved: float = 0.0) -> Peaks:
    """A peak is a ceiling. The chip measured doing these ops, or moving
    these bytes, in less time than the peaks allow proves the peaks low by
    that factor, so they rise to it; the probe at job start is one reading
    from another minute and sat 10 to 25% under the library's own rate on
    FLUX and RecurrentGemma. Returns the same object when nothing was proved."""
    if measured_ms <= 0:
        return peaks
    compute = compute_time_ms(list(nodes), peaks) / measured_ms
    memory = bytes_moved / (peaks.bandwidth_gbps * 1e9) * 1e3 / measured_ms
    if compute <= 1.0 and memory <= 1.0:
        return peaks
    return replace(peaks,
                   flops_gflops={d: v * max(compute, 1.0) for d, v in peaks.flops_gflops.items()},
                   bandwidth_gbps=peaks.bandwidth_gbps * max(memory, 1.0))


def compute_dtype(node: TraceNode) -> str:
    """The dtype an op's arithmetic is priced at: its first floating output."""
    return next((d for _, d in node.out_specs if d in ("float16", "bfloat16", "float32")), "float32")


def compute_time_ms(nodes: Iterable[TraceNode], peaks: Peaks,
                    rates: Mapping[str, float] | None = None) -> float:
    """Price each op at its own dtype's throughput, including mixed models:
    the rate measured beside the region when there is one, else the peak.

    This is a scheduling estimate: GEMM throughput does not model every op's
    instruction mix. A small fp32 prelude must not price a bf16 transformer.
    """
    fallback = max(peaks.flops_gflops.values(), default=1e3)
    milliseconds = 0.0
    for node in nodes:
        dtype = compute_dtype(node)
        peak = (rates or {}).get(dtype) or peaks.flops_gflops.get(dtype, fallback)
        milliseconds += node_flops(node) / (peak * 1e6)
    return milliseconds
