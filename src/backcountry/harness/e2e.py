"""Whole-model correctness against the untouched model and the step-time veto.

Preserving edits are bit-identical. Changed floating-point evaluation follows
fixed manifest tolerances, using the same implementation as exported bundles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import mlx.core as mx

from .measure.clocks import PairedComparison, compare
from .measure.session import Session

@dataclass
class WorkloadCheck:
    name: str
    floor_max_abs: float
    max_abs: float
    allowance: float
    cosine: float
    passed: bool
    reason: str = ""
    floor_cosine: float = 1.0
    cosine_allowance: float = 0.0
    rule: str = "exact"
    failure: dict | None = None


@dataclass
class E2EResult:
    checks: list[WorkloadCheck] = field(default_factory=list)
    veto: PairedComparison | None = None
    veto_passed: bool = True
    workload_vetos: dict[str, PairedComparison] = field(default_factory=dict)
    target_workload: str | None = None
    required: frozenset[str] | None = None  # the workloads it must time; None: every one
    min_wins: int = 1                      # resolved wins it needs among them
    margin_share: float = 0.0              # a difference under this share of the baseline is neither

    @property
    def passed(self) -> bool:
        return self.veto_passed and all(c.passed for c in self.checks)


def preserving_check(
    baseline_run: Callable[[], object],
    patched_run: Callable[[], object],
    name: str,
    *, exact: bool = True,
    tolerances: tuple[float, float] | None = None,
) -> WorkloadCheck:
    """Use the same output policy as the independently loaded artifact.

    Baseline is always the untouched model. Tolerances are fixed by the job,
    never enlarged by accumulated edits or observed baseline wobble.
    """
    from .artifact.validate import check_outputs
    result = check_outputs(baseline_run, patched_run, name,
                           exact=exact, tolerances=tolerances)
    fields = WorkloadCheck.__dataclass_fields__
    return WorkloadCheck(**{key: value for key, value in result.items() if key in fields})


def reference_check(baseline_run, patched_run, reference_run, name: str) -> WorkloadCheck:
    """A changing edit graded against a reference that recomputes the changed
    regions in fp32 (bind.fp32_region): its worst error against it, relative
    to each output's largest value, may be 1.25x the library's
    (validate.check_changing_outputs, the rule bundles validate with)."""
    from .artifact.validate import _observe, check_changing_outputs
    golden, _ = _observe(reference_run)
    result = check_changing_outputs(baseline_run, patched_run, golden, name)
    fields = WorkloadCheck.__dataclass_fields__
    return WorkloadCheck(**{key: value for key, value in result.items() if key in fields})


def changing_check(baseline_run, patched_run, name: str, *, tolerances=None) -> WorkloadCheck:
    """Changed floating-point evaluation must stay within manifest tolerances."""
    return preserving_check(baseline_run, patched_run, name,
                            exact=False, tolerances=tolerances)


def step_veto(
    session: Session,
    baseline_run: Callable[[], object],
    patched_run: Callable[[], object],
    pairs: int = 16,
    defer_cooling: bool = False,
) -> tuple[PairedComparison, bool]:
    """The patched step must not be slower with confidence: a non-regression
    veto under the interleaved paired discipline. The decision to keep an
    install is the caller's, and it asks for a resolved win, not merely no
    loss; there is no fixed percentage anywhere in either rule."""
    result = compare(session, baseline_run, patched_run, pairs=pairs, defer_cooling=defer_cooling)
    return result, not result.loses_by(0.0)


def run_e2e(
    session: Session,
    baseline_model: Callable,
    patched_model: Callable,
    workloads: Sequence[tuple[str, list[mx.array]]],
    veto_pairs: int = 16,
    timed: tuple[Callable[[], object], Callable[[], object]]
        | Mapping[str, tuple[Callable[[], object], Callable[[], object]]] | None = None,
    exact: bool = True,
    tolerances: tuple[float, float] | None = None,
    defer_cooling: bool = False,
    target_workload: str | None = None,
    reference_model: Callable | None = None,
    step_of: Callable | None = None,
    stop_at_loss: bool = False,
    fits: Callable[[str], bool] | None = None,
    after_timing: Callable[[str], None] | None = None,
) -> E2EResult:
    """Check outputs on plain models and compare every declared workload.

    A timing mapping supplies compiled arms and explicitly selects the shapes
    to time; other shapes receive correctness checks only (e.g. sweep shapes).
    A legacy tuple supplies the first workload's arms. A nominated target is
    timed first; if it does not win, skip the remaining timings. Correctness
    still covers every input, and callers must require complete timings before
    accepting an install. A reference model (the changed regions recomputed in
    fp32) grades an inexact patch in place of fixed tolerances. step_of(model,
    name) is the step a workload calls on a model (each model by default).
    stop_at_loss skips the remaining timings once one workload resolves a loss,
    which no later timing can turn into a ship. fits(name) says whether a
    workload may still be timed (one a comparison found too close to the GPU's
    memory limit may not); after_timing(name) is told after each comparison.
    """
    step_of = step_of or (lambda model, name: model)
    if not workloads:
        raise ValueError("end-to-end validation requires at least one workload")
    names = {name for name, _ in workloads}
    if isinstance(timed, Mapping) and (not timed or set(timed) - names):
        raise ValueError("timed workloads must be a nonempty subset of checked workloads")
    timed_names = set(timed) if isinstance(timed, Mapping) else names
    if target_workload is not None and target_workload not in timed_names:
        raise ValueError("target workload must be one of the timed workloads")
    result = E2EResult(target_workload=target_workload)
    from backcountry_runtime.state import correctness_call
    for name, tensors in workloads:
        baseline_run = lambda t=tensors, m=step_of(baseline_model, name): correctness_call(m, t)
        patched_run = lambda t=tensors, m=step_of(patched_model, name): correctness_call(m, t)
        if reference_model is not None and not exact:
            reference_run = lambda t=tensors, m=step_of(reference_model, name): correctness_call(m, t)
            check = lambda: reference_check(baseline_run, patched_run, reference_run, name)
        else:
            check = lambda: preserving_check(baseline_run, patched_run, name,
                                              exact=exact, tolerances=tolerances)
        result.checks.append(session.off_clock(check, defer_cooling=defer_cooling,
                                               where=f"whole-model check {name}"))
    if not all(check.passed for check in result.checks):
        result.veto_passed = False
        return result
    ordered = list(enumerate(workloads))
    if target_workload is not None:
        ordered.sort(key=lambda item: item[1][0] != target_workload)
    for i, (name, tensors) in ordered:
        if fits is not None and not fits(name):
            continue
        if isinstance(timed, Mapping):
            if name not in timed:
                continue
            arms = timed[name]
        elif i == 0 and timed is not None:
            arms = timed
        else:
            arms = (lambda t=tensors, m=step_of(baseline_model, name): m(*t),
                    lambda t=tensors, m=step_of(patched_model, name): m(*t))
        comparison, passed = step_veto(session, *arms, pairs=veto_pairs, defer_cooling=defer_cooling)
        if after_timing is not None:
            after_timing(name)
        result.workload_vetos[name] = comparison
        result.veto_passed &= passed
        if result.veto is None:
            result.veto = comparison
        if name == target_workload and not comparison.wins_by(0.0):
            break
        if stop_at_loss and comparison.loses_by(0.0):
            break
    return result


def share_weights(donor, receiver) -> int:
    """Point receiver's parameters at donor's arrays and return how many
    were shared; the caller verifies the count is every parameter."""
    import mlx.nn as nn

    if not isinstance(receiver, nn.Module) or not isinstance(donor, nn.Module):
        return 0
    params = donor.parameters()
    receiver.update(params)
    from backcountry_runtime.state import ContextStep, _copy_cache
    from backcountry_runtime.inference import LibraryInference
    if isinstance(donor, ContextStep) and isinstance(receiver, ContextStep):
        # The saved prefix was computed with the donor's weights too.
        receiver._cache = _copy_cache(donor._cache)
        receiver._caches = {batch: _copy_cache(cache) for batch, cache in donor._caches.items()}
    if isinstance(donor, LibraryInference) and isinstance(receiver, LibraryInference):
        # Prefix state must correspond to the shared weights, not the second
        # build's independently initialized parameters.
        receiver._start = _copy_cache(donor._start)
    shared = 0
    flat_d = _flatten_params(params)
    flat_r = _flatten_params(receiver.parameters())
    for k, v in flat_d.items():
        if k in flat_r and flat_r[k] is v:
            shared += 1
    return shared


def _flatten_params(tree, prefix="") -> dict:
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten_params(v, f"{prefix}.{k}"))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(_flatten_params(v, f"{prefix}.{i}"))
    else:
        out[prefix] = tree
    return out
