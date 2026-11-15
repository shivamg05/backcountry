"""A region recomputed in fp32, installed where its kernel would go.

A kernel that changes rounding (assoc tag "changing") cannot be bitwise the
library, and the whole model has no fp32 reference on a large model: an fp32
copy does not fit, and a custom kernel inside (Qwen3.5's gated delta) hides
its arithmetic from promotion. Its region can be promoted, though: the same
recorded ops, float bindings lifted to fp32 and quantized matmuls dequantized
in fp32 (ladder.golden). The model with every changing region recomputed so,
and rounded back to the recorded dtypes, is the reference the kernel is graded
against: no further from it than the library is (validate.check_changing_outputs).

Measured on Qwen3.8-27B, gate and up matmuls of all 64 layers: that reference
moves the gated-delta state 1-4% (max-relative) away from the library, as far
as a correct kernel does; the fixed fp32 tolerance (1e-5) the state was held
to before rejected every kernel that was not bitwise the library.

Job side only: the stand-in never ships; a bundle carries the reference's
outputs (goldens), not this code.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

import mlx.core as mx

from backcountry_runtime import kernels as kernels_mod
from backcountry_runtime.kernels import KernelSpec, Launch

from ..ladder.golden import golden_outputs, substitution_table
from ..measure.probe import dtype_of
from ..regions.sweep import SweepDivergence, locate_span
from ..regions.types import Stretch
from ..trace.types import Trace
from .emit import Splice

SUFFIX = ":fp32_reference"


class NoReference(ValueError):
    """The changed regions cannot all be recomputed in fp32; the job keeps
    the manifest tolerances."""


def _signature(arrays) -> tuple:
    return tuple((tuple(a.shape), str(a.dtype).removeprefix("mlx.core.")) for a in arrays)


class Fp32Region:
    """Stands in for a loaded kernel (kernels._loaded): one recorded span per
    input signature, replayed in fp32 and rounded back."""

    def __init__(self, spec: KernelSpec, spans: Iterable[tuple[Trace, Splice]]):
        self.spec = spec
        self._spans: dict[tuple, tuple] = {}
        for trace, cut in spans:
            specs = trace.span_specs(cut.start_seq, cut.end_seq)
            signature = tuple((tuple(specs[a][0]), specs[a][1]) for a in cut.input_ids)
            self._spans.setdefault(signature, (
                trace.nodes[cut.start_seq:cut.end_seq + 1], cut.input_ids, cut.output_ids,
                [dtype_of(specs[a][1]) for a in cut.output_ids]))

    def launch(self, inputs, checked_elsewhere: bool = False) -> Launch:
        return Launch(fallback=_signature(inputs) not in self._spans)

    def fallback_fires(self, inputs) -> bool:
        return self.launch(inputs).fallback

    def __call__(self, inputs, init_value=None, launch=None) -> list[mx.array]:
        nodes, input_ids, output_ids, dtypes = self._spans[_signature(inputs)]
        values = golden_outputs(nodes, dict(zip(input_ids, inputs)), output_ids)
        return [values[a].astype(dtype) for a, dtype in zip(output_ids, dtypes)]


def reference_definitions(
    traces: Mapping[str, Trace],
    extra: Mapping[str, Trace],
    states: Mapping[str, tuple],
    changing: set[str],
) -> dict[str, list[tuple]]:
    """Per installed scope, the wrapper definitions [(trace, scope call,
    cuts)] of a reference model: each changing region's cuts handed to its
    fp32 stand-in, every other cut left out (the library's own ops). extra
    (label -> trace: sweep and serve recordings, "<workload>@...") gets the
    same cuts located in it, so the reference runs at every saved size.
    Registers the stand-ins with the kernel runtime."""
    cuts_at: dict[str, dict[tuple[str, str], list[Splice]]] = {}
    for path, (_original, variants, _kernels) in states.items():
        for (workload, address), cuts in variants.items():
            kept = [cut for cut in cuts if cut.fingerprint in changing]
            if kept:
                cuts_at.setdefault(path, {})[(workload, address)] = kept
            for label, retrace in extra.items():
                if label.split("@", 1)[0] != workload or not kept:
                    continue
                located = [_relocate(traces[workload], cut, retrace, label) for cut in kept]
                if all(c is not None for c in located) and any(
                        sc.address == address for sc in retrace.scope_calls):
                    cuts_at.setdefault(path, {})[(label, address)] = located
    every = {**traces, **extra}
    present = {cut.fingerprint for variants in cuts_at.values() for cuts in variants.values() for cut in cuts}
    if present != set(changing):
        raise NoReference(f"no installed cuts for {sorted(set(changing) - present)}")
    table = substitution_table()
    for variants in cuts_at.values():
        for (label, _address), cuts in variants.items():
            for cut in cuts:
                for node in every[label].nodes[cut.start_seq:cut.end_seq + 1]:
                    if node.kernel_definition is not None and node.op not in table:
                        raise NoReference(f"region {cut.fingerprint} runs captured Metal source "
                                          f"({node.op}) with no audited fp32 form")
    spans: dict[str, list[tuple[Trace, Splice]]] = {}
    for variants in cuts_at.values():
        for (label, _address), cuts in variants.items():
            for cut in cuts:
                spans.setdefault(cut.fingerprint, []).append((every[label], cut))
    specs = {}
    for fingerprint, entries in spans.items():
        spec = replace(entries[0][1].kernel, kernel_id=fingerprint + SUFFIX, fallback_predicate=None,
                       input_signature=None, input_signatures=None)
        kernels_mod._cache[spec.kernel_id] = Fp32Region(spec, entries)
        specs[fingerprint] = spec
    return {path: [(every[label], next(sc for sc in every[label].scope_calls if sc.address == address),
                    [replace(cut, kernel=specs[cut.fingerprint]) for cut in cuts])
                   for (label, address), cuts in variants.items()]
            for path, variants in cuts_at.items()}


def _relocate(priced: Trace, cut: Splice, retrace: Trace, label: str) -> Splice | None:
    """The cut's ops in another recording, its inputs and outputs matched by
    role (first node and slot that touches each), in the cut's order; None
    when the op stream diverged there."""
    roles = _positions(priced, cut.start_seq, cut.end_seq)
    natural = lambda ids: tuple(sorted(ids, key=lambda a: roles[a]))
    try:
        span = locate_span(priced, Stretch(label, cut.start_seq, cut.end_seq, natural(cut.input_ids),
                                           natural(cut.output_ids), ()), retrace, label)
    except SweepDivergence:
        return None
    by_role = {pos: aid for aid, pos in _positions(retrace, span.start_seq, span.end_seq).items()}
    try:
        return replace(cut, start_seq=span.start_seq, end_seq=span.end_seq,
                       input_ids=tuple(by_role[roles[a]] for a in cut.input_ids),
                       output_ids=tuple(by_role[roles[a]] for a in cut.output_ids))
    except KeyError:
        return None


def _positions(trace: Trace, start: int, end: int) -> dict[int, tuple]:
    """array id -> (in/out, node offset, slot) where the span first touches it
    (regions.fingerprint.boundary_roles)."""
    positions = {}
    for n, node in enumerate(trace.nodes[start:end + 1]):
        for kind, ids in (("in", node.in_arrays), ("out", node.out_arrays)):
            for slot, aid in enumerate(ids):
                positions.setdefault(aid, (kind, n, slot))
    return positions
