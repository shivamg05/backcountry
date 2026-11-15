"""The paired comparison and its summary statistics, shared by the harness
clocks and an exported bundle's benchmark. Sign convention: positive delta
means the candidate is faster."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

# SE(median) for normal noise is ~1.2533 * sigma / sqrt(n). The margins below
# read on the uncertainty of the measured win, so sigma_ms is that standard
# error, not the raw per-sample spread.
_MEDIAN_SE_FACTOR = 1.2533
# A win counts once the median gain clears WIN_SIGMAS standard errors; a
# slowdown counts at LOSS_SIGMAS. Wins on a memory-bound step are often under
# 1%, about 2.5 standard errors of a whole-model comparison: at 3 a stream of
# them went unconfirmed. A slowdown keeps the stricter bar, and the A/A
# control, which looks for any difference at all, uses LOSS_SIGMAS both ways.
WIN_SIGMAS = 2.0
LOSS_SIGMAS = 3.0
# From this many pairs, a paired difference further than CLIP_SPREADS from
# the median (in 1.4826 x their median absolute deviation, the standard
# deviation for normal noise) counts as that far before the standard
# deviation is taken, so one pair that hit a paging spike cannot hide a win:
# on Qwen3.8-27B one such comparison read 38 ms of noise where that step's
# comparisons read 1.6-4.2 ms. Simulated at 10 pairs, a false win at 2
# sigmas goes from 2.7% to 3.3% (the median absolute deviation alone: 5.3%).
# Fewer pairs keep the plain standard deviation: four values say too little.
ROBUST_PAIRS = 8
CLIP_SPREADS = 3.0


@dataclass(frozen=True)
class PairedComparison:
    baseline_ms: tuple[float, ...]
    candidate_ms: tuple[float, ...]
    deltas_ms: tuple[float, ...]  # per pair, baseline - candidate: positive = faster
    median_baseline_ms: float
    median_delta_ms: float
    spread_ms: float  # raw stdev of the paired deltas
    sigma_ms: float   # standard error of the median delta (see ROBUST_PAIRS); margins are multiples of it
    n: int
    ratios: tuple[float, ...] = ()   # per pair, candidate / baseline
    median_ratio: float = 0.0        # the drift-immune quantity: report this
    stability: float = 0.0           # 0..1, how far the pairs agreed

    def wins_by(self, margin_ms: float, sigmas: float = WIN_SIGMAS) -> bool:
        return self.median_delta_ms > max(margin_ms, sigmas * self.sigma_ms)

    def loses_by(self, margin_ms: float, sigmas: float = LOSS_SIGMAS) -> bool:
        return -self.median_delta_ms > max(margin_ms, sigmas * self.sigma_ms)


def workload_win(
    comparisons: Mapping[str, PairedComparison], target_workload: str | None = None,
    share: float = 0.0,
) -> bool:
    """Require a resolved win and no resolved regression among measured targets;
    a difference under share of the baseline counts as neither.

    A nominated target must win itself, so independent confirmation cannot
    switch to another workload. Callers must separately ensure every required
    workload was measured. An unresolved result is not proof of zero slowdown.
    """
    if not comparisons:
        return False
    for c in comparisons.values():
        if not c.n or not (len(c.baseline_ms) == len(c.candidate_ms) == len(c.deltas_ms) == c.n):
            return False
        values = (*c.baseline_ms, *c.candidate_ms, *c.deltas_ms, *c.ratios,
                  c.median_baseline_ms, c.median_delta_ms, c.spread_ms,
                  c.sigma_ms, c.median_ratio, c.stability)
        # NaN comparisons can look like neither a win nor a loss. Another
        # workload's win must never turn invalid measurements into acceptance.
        if not all(isfinite(value) for value in values) or c.loses_by(share * c.median_baseline_ms):
            return False
    if target_workload is not None:
        target = comparisons.get(target_workload)
        return target is not None and target.wins_by(share * target.median_baseline_ms)
    return any(c.wins_by(share * c.median_baseline_ms) for c in comparisons.values())


def _ratio_stats(base: list[float], cand: list[float]) -> tuple[tuple[float, ...], float, float]:
    """Per-pair ratios, their median, and a 0..1 agreement score. The score
    reads the ratios rather than the absolute times: on a drifting machine the
    absolutes are meant to move and the ratios are not, so their spread is the
    honest confidence in the comparison."""
    ratios = tuple(c / b for b, c in zip(base, cand) if b > 0)
    if not ratios:
        return (), 0.0, 0.0
    median = statistics.median(ratios)
    if len(ratios) < 4 or median <= 0:
        return ratios, median, 0.0
    q1, _, q3 = statistics.quantiles(ratios, n=4)
    # median / (median + IQR): 1 when the pairs agreed exactly, 0.5 when the
    # spread equals the value, never saturating, so a hopeless measurement and
    # a merely noisy one stay distinguishable.
    return ratios, median, median / (median + (q3 - q1))


def paired_means(values: Sequence[float]) -> list[float]:
    """One observation per forward/reverse block, centered on the same time."""
    if len(values) < 2 or len(values) % 2:
        raise ValueError("samples must contain complete forward/reverse blocks")
    return [(values[i] + values[i + 1]) / 2 for i in range(0, len(values), 2)]


def comparison_from_samples(base: Sequence[float], cand: Sequence[float]) -> PairedComparison:
    """Summarize aligned observations in milliseconds."""
    if not base or len(base) != len(cand):
        raise ValueError("comparison needs equally sized, nonempty sample sets")
    deltas = [a - b for a, b in zip(base, cand)]
    spread = statistics.stdev(deltas) if len(deltas) > 1 else float("inf")
    noise = spread
    if len(deltas) >= ROBUST_PAIRS:
        middle = statistics.median(deltas)
        reach = CLIP_SPREADS * 1.4826 * statistics.median(abs(d - middle) for d in deltas)
        if reach > 0:
            noise = statistics.stdev([min(max(d, middle - reach), middle + reach) for d in deltas])
    ratios, median_ratio, stability = _ratio_stats(base, cand)
    return PairedComparison(
        baseline_ms=tuple(base), candidate_ms=tuple(cand), deltas_ms=tuple(deltas),
        median_baseline_ms=statistics.median(base), median_delta_ms=statistics.median(deltas),
        spread_ms=spread, sigma_ms=_MEDIAN_SE_FACTOR * noise / len(deltas) ** 0.5,
        n=len(deltas), ratios=ratios, median_ratio=median_ratio, stability=stability)
