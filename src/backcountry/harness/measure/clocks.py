"""Clocks: the step clock and the paired interleaved comparison.

Law 4: never subtract two separately timed quantities. Every A/B number here
comes from one session, alternating ABBA so thermal drift is common-mode.
Pacing idles land only at block boundaries, followed by ramp-warm samples,
so no timed sample ever starts on ramped-down clocks. Sign convention:
positive delta means the candidate is faster.
"""

from __future__ import annotations

import functools
import statistics

import mlx.core as mx
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from backcountry_runtime.sequence import make_sequence
from backcountry_runtime.stats import (_MEDIAN_SE_FACTOR, PairedComparison, _ratio_stats,
                                     comparison_from_samples, paired_means)
from backcountry_runtime.swap import flatten_arrays

from .session import Session, memory_state

# One timed sample of a looped region holds this much work, so the fixed
# submit-and-sync cost of an evaluation is a rounding error in the per-pass
# figure. The pricing clock and the ship clock size their loops with the
# same rule, or every headroom figure would carry the gap between them.
CLOCK_TARGET_MS = 20.0
CLOCK_EST_ITERS = 10   # passes in the estimate that sizes the loop
CLOCK_MIN_ITERS = 1    # a slow pass already amortizes submission cost
CLOCK_MAX_ITERS = 2000
# Outputs one timed sample may hold: a sample keeps every pass's outputs until
# it ends, and a noisy estimate can ask for CLOCK_MAX_ITERS passes of a cheap
# region with large outputs (2000 x 64 MB swapped a 24 GB Mac). A correct
# estimate never needs more: ~20 ms at ~100 GB/s moves about 2 GB.
CLOCK_MAX_BYTES = 1 << 30


@dataclass(frozen=True)
class StepClock:
    median_ms: float
    samples_ms: tuple[float, ...]
    warm_ms: tuple[float, ...]


# A timed loop must read its bytes from memory and run its passes one after
# another, the way a model step does. Rotating a working set this large keeps
# the data out of the GPU's cache; chaining the passes keeps Metal from running
# independent launches side by side, which makes a one-threadgroup kernel look
# many times faster than it is in a model, where each layer waits for the last.
CACHE_DEFEAT_BYTES = 128 * 1024 * 1024
MAX_TIMING_SETS = 16


def synthesize_like(arrays: Mapping[int, mx.array], seed: int) -> dict[int, mx.array]:
    """Timing-only inputs: the same shapes and dtypes, random values. Never
    used for a correctness comparison."""
    keys = mx.random.split(mx.random.key(seed), max(len(arrays), 1))
    out = {}
    for i, (aid, arr) in enumerate(sorted(arrays.items())):
        if arr.dtype in (mx.float16, mx.bfloat16, mx.float32):
            out[aid] = mx.random.normal(arr.shape, dtype=arr.dtype, key=keys[i])
        else:
            out[aid] = arr  # integer inputs (indices) keep real values
    mx.eval(list(out.values()))
    return out


def timing_sets(sets: Sequence[Mapping[int, mx.array]]) -> list[dict[int, mx.array]]:
    """The sets a timed loop rotates over: the given ones plus synthesized
    look-alikes until the working set is too big for the cache."""
    out = [dict(s) for s in sets]
    set_bytes = sum(a.nbytes for a in out[0].values())
    seed = 0
    while set_bytes * len(out) < CACHE_DEFEAT_BYTES and len(out) < MAX_TIMING_SETS:
        out.append({**out[0], **synthesize_like(out[0], 7000 + seed)})
        seed += 1
    return out


class HeldSets:
    """A region's timing sets, in memory only while its arms are being timed.

    Pricing times every region of a wave against one shared model arm. Built
    up front, all of their sets sat in memory while the model ran: 48 regions
    at 772 tokens put Qwen3.8-27B's job at 19.6 GB, past the GPU's 19.07 GB,
    and the model pass timed out. sample_group opens one HeldSets at a time:
    open() loads the sets and builds its arms' loops before the first of them
    is timed, and close() drops them before any other arm runs, so the model
    is timed with none resident. load() must return the same sets every time
    (timing_sets seeds its look-alikes)."""

    def __init__(self, load: Callable[[], list[dict[int, mx.array]]]):
        self._load = load
        self._arms: list[HeldArm] = []
        self.sets: list[dict[int, mx.array]] | None = None

    def arm(self, make: Callable[[Sequence[Mapping[int, mx.array]]], Callable[[], list]]) -> "HeldArm":
        """An arm timing the loop make(sets) builds over these sets."""
        arm = HeldArm(self, make)
        self._arms.append(arm)
        return arm

    def open(self) -> None:
        if self.sets is None:
            self.sets = self._load()
            for arm in self._arms:
                arm.loop = arm.make(self.sets)

    def close(self) -> None:
        for arm in self._arms:
            arm.loop = None
        self.sets = None


class HeldArm:
    """One timed loop over a HeldSets' sets; it runs only while they are open."""

    def __init__(self, held: HeldSets, make):
        self.held, self.make = held, make
        self.loop: Callable[[], list] | None = None

    def __call__(self) -> list:
        if self.loop is None:
            raise RuntimeError("a held arm ran with its sets closed")
        return self.loop()


def _switch_held(arm, current: HeldSets | None) -> HeldSets | None:
    """Open arm's sets, closing whichever are open; a plain arm closes them all."""
    held = getattr(arm, "held", None)
    if held is not current:
        if current is not None:
            current.close()
        if held is not None:
            held.open()
    return held


def link_input(binds: Mapping[int, mx.array], weight_ids: Iterable[int] = ()) -> int | None:
    """The input the chain link rides on: the smallest float input that is not
    a weight (a weight would cost a copy per pass), else the smallest float
    input; None when nothing float enters, which leaves the loop unchained."""
    floats = {a: arr for a, arr in binds.items() if arr.dtype in (mx.float16, mx.bfloat16, mx.float32)}
    pool = {a: arr for a, arr in floats.items() if a not in set(weight_ids)} or floats
    return min(pool, key=lambda a: pool[a].nbytes) if pool else None


def chained_loop(pass_fn: Callable[[Mapping[int, mx.array]], object],
                 sets: Sequence[Mapping[int, mx.array]], iters: int,
                 link_id: int | None) -> Callable[[], list]:
    """iters passes over the rotating sets, each pass's linked input carrying
    a zero taken from the previous pass's first output, so no pass can start
    before the last one ends. The link costs the same on every arm; paired
    against link_loop, it drops out of a per-pass figure."""
    def loop() -> list:
        outs, link = [], None
        for i in range(iters):
            binds = sets[i % len(sets)]
            if link is not None:
                binds = {**binds, link_id: binds[link_id] + link}
            out = pass_fn(binds)
            outs.append(out)
            if link_id is not None:
                if isinstance(out, (list, tuple)) and not out:
                    raise ValueError("a timed pass returned no outputs; the chain has nothing to link")
                first = out[0] if isinstance(out, (list, tuple)) else out
                link = (first.reshape(-1)[0] * 0).astype(binds[link_id].dtype)
        return outs
    return loop


def link_loop(sets: Sequence[Mapping[int, mx.array]], iters: int, link_id: int) -> Callable[[], list]:
    """The chain alone: the same loop around a pass that only hands its
    linked input back. Paired against a real loop, it takes the link's cost
    and a sample's fixed submit-and-sync cost out of the per-pass figure."""
    return chained_loop(lambda b: [b[link_id]], sets, iters, link_id)


def chained_steps(step: Callable[..., object], tensors: Sequence[mx.array], k: int) -> Callable[[], list]:
    """k dependent calls of a model step as one timed sample (make_sequence),
    read per step: time_once divides by the steps count the closure carries.
    A step under CLOCK_TARGET_MS called on its own leaves the GPU idle
    between samples and never brings its clock up; a chain this long does
    (see tools/ramp_after_idle.py). A chain of one is the plain call."""
    run = make_sequence(step, tensors, k) if k > 1 else (lambda: step(*tensors))
    run.steps = k
    return run


def loop_iterations(timer: Callable[[Callable[[], object]], float],
                    loop_for: Callable[[int], Callable[[], object]],
                    target_ms: float = CLOCK_TARGET_MS) -> int:
    """How many passes one timed sample should hold. loop_for(n) is the loop
    of n passes as it will be timed. The per-pass estimate is the difference
    between a longer and a shorter warm loop, so a sample's fixed
    submit-and-sync cost, which a busy GPU can push to several milliseconds,
    is not mistaken for pass time. The first loop is thrown away (Metal
    compile); a pass that fills the sample on its own is timed once more and
    needs no loop. A GPU at idle clocks makes every pass look several times
    slow and the loop too short, and only a burst as long as the loop itself
    brings the clock up, so the sized loop is timed until its reading stops
    falling and re-sized from that reading while it still grows."""
    loop, outputs = loop_for(1), []

    def first():
        outputs.append(loop())
        return outputs[-1]
    first.__dict__.update(getattr(loop, "__dict__", {}))  # a timer may read what the loop carries
    timer(first)
    pass_bytes = sum(a.nbytes for a in flatten_arrays(outputs))
    most = max(CLOCK_MIN_ITERS, CLOCK_MAX_BYTES // pass_bytes) if pass_bytes else CLOCK_MAX_ITERS
    outputs.clear()
    short = 1
    t_short = timer(loop_for(short))
    if t_short * 1e3 >= target_ms:
        return CLOCK_MIN_ITERS
    t_long = timer(loop_for(4 * short))
    if (t_long - t_short) * 1e3 < target_ms / 4 and 4 * CLOCK_EST_ITERS <= most:
        short = CLOCK_EST_ITERS
        t_short = timer(loop_for(short))
        t_long = timer(loop_for(4 * short))
    long = 4 * short
    t_est_ms = (t_long - t_short) / (long - short) * 1e3
    if t_est_ms <= 0:
        # noise (a clock still ramping) made the longer loop no slower: the
        # longer loop's own time per pass bounds a pass instead of "free"
        t_est_ms = t_long / long * 1e3
    iters = _clamp(target_ms / max(t_est_ms, 1e-3), most)
    for _ in range(3):
        best = timer(loop_for(iters))
        while (t := timer(loop_for(iters))) < best * 0.99:
            best = t
        grown = _clamp(target_ms / max(best * 1e3 / iters, 1e-3), most)
        if grown <= iters * 1.25:
            break
        iters = grown
    return iters


def _clamp(iters: float, most: int = CLOCK_MAX_ITERS) -> int:
    return max(CLOCK_MIN_ITERS, min(int(iters), most, CLOCK_MAX_ITERS))


def _again_if_hot(measure):
    """A measurement the chip went hot in is taken again, once, after the
    cooldown: the heat watcher trips at most once per session, and a fanless
    Air timed against itself read 1.017 in the comparison it tripped in (it
    throttled from there on) and 0.999-1.002 in the three after."""
    @functools.wraps(measure)
    def again(session, *args, **kwargs):
        trips = getattr(session, "heat_trips", 0)
        result = measure(session, *args, **kwargs)
        if getattr(session, "heat_trips", 0) == trips:
            return result
        session.wait_ready()  # pays the cooldown the trip left owed
        session.log("measured_again", what=measure.__name__, why="the chip went hot during it")
        return measure(session, *args, **kwargs)
    return again


@_again_if_hot
def sample_group(session: Session, arms: dict[str, Callable[[], object]],
                 pairs: int = 8, *, defer_cooling: bool = False) -> dict[str, tuple[float, ...]]:
    """Measure several arms together, reversing their order every pass.

    The model arm is shared by all region prices in this group. A forward
    pass followed by its reverse brackets every other arm with that model,
    without rerunning a slow model separately for every candidate. Returns
    aligned sample rows in milliseconds; no old baseline enters a row.
    Deferred cooling requires the caller to gate its next GPU phase, or to
    hand the deadline to a parent that will do so.
    """
    if not arms or pairs < 2 or pairs % 2:
        raise ValueError("a group needs arms and an even number of pairs >= 2")
    names = list(arms)
    memory_before = session.memory("sample_group", arms=len(arms))
    held = None    # the one HeldSets open; held arms load their sets only for their turn
    touched = set()  # held arms run once since their sets were loaded

    def ready(arm):
        """Open arm's sets for its turn. Memory just loaded is slow on its
        first GPU read: timed straight after loading, Qwen3-4B's regions
        priced 15% above the same regions held throughout. One untimed pass
        per arm pays that before its sample."""
        nonlocal held
        if getattr(arm, "held", None) is not held:
            held = _switch_held(arm, held)
            touched.clear()
        if held is not None and arm not in touched:
            session.timed(arm)
            touched.add(arm)

    samples: dict[str, list[float]] = {name: [] for name in names}
    try:
        ready(arms[names[0]])
        session.fresh_chunk(arms[names[0]])
        for name in names:
            ready(arms[name])
            session.warm_until_stable(arms[name])
        last = arms[names[-1]]
        for index in range(pairs):
            for name in names if index % 2 == 0 else reversed(names):
                arm = arms[name]
                ready(arm)
                if hasattr(arm, "steps") and arm is not last:
                    # a model step straight after other GPU work is not how
                    # it runs: Gemma 4's decode step read 140-185 ms there and
                    # 35 ms after another step. One untimed step first.
                    session.timed(arm)
                samples[name].append(session.timed(arm) * 1e3)
                last = arm
    finally:
        _switch_held(None, held)
        if defer_cooling:
            session.defer_settle()
        else:
            session.settle()
    session.log("sample_group", arms=len(arms), pairs=pairs, rows=samples,
                memory_before=memory_before, memory_after=memory_state())
    return {name: tuple(values) for name, values in samples.items()}


@_again_if_hot
def step_clock(session: Session, fn: Callable[[], object], reps: int = 9) -> StepClock:
    """The honest cost of one call of fn: warm until stable, then the median."""
    warmed = session.fresh_chunk(fn)
    # A successful post-cooling ramp already warmed this same function.
    warm = [] if warmed else session.warm_until_stable(fn)
    samples = [session.timed(fn) for _ in range(reps)]
    clock = StepClock(
        median_ms=statistics.median(samples) * 1e3,
        samples_ms=tuple(t * 1e3 for t in samples),
        warm_ms=tuple(t * 1e3 for t in warm),
    )
    session.log("step_clock", median_ms=clock.median_ms, n=reps)
    print(f"step clock: {clock.median_ms:.3f} ms (median of {reps} samples)", flush=True)
    session.settle()
    return clock


@_again_if_hot
def compare(
    session: Session,
    baseline_fn: Callable[[], object],
    candidate_fn: Callable[[], object],
    pairs: int = 16,
    warm_baseline: bool = True,
    defer_cooling: bool = False,
) -> PairedComparison:
    """Paired interleaved A/B. Alternate ABBA and BAAB blocks, averaging
    each arm's two samples before comparing them. Both surround the same midpoint,
    so linear drift cancels within each observation. A block is one
    observation, not two independent pairs. Both arms are warmed first (a
    caller that just warmed the baseline may say so); the baseline is
    measured here, now, never reused.

    Pacing happens once, before the warms, so every sample of the comparison
    is taken in the state the model actually runs in. Idling between samples
    of one comparison measured a machine nobody deploys on: the same step read
    5.7 ms back to back and 8-16 ms paced."""
    if pairs < 2 or pairs % 2:
        raise ValueError(f"pairs must be even and >= 2, got {pairs}")
    mx.reset_peak_memory()  # this comparison's own peak (session.last_compare_peak_gb), not the check before it
    warmed = session.fresh_chunk(baseline_fn)
    if warm_baseline and not warmed:
        session.warm_until_stable(baseline_fn)
    session.warm_until_stable(candidate_fn)
    base: list[float] = []
    cand: list[float] = []
    held = cached = 0  # MLX's memory between samples: a creep toward the GPU's limit shows here
    for block in range(pairs // 2):
        # Balance which arm occupies the outside slots. Fixed ABBA cancels
        # linear drift but gives a recurring slot/cache bias to the same arm.
        order = ("a", "b", "b", "a") if block % 2 == 0 else ("b", "a", "a", "b")
        for arm in order:
            if arm == "a":
                base.append(session.timed(baseline_fn))
            else:
                cand.append(session.timed(candidate_fn))
            cached = max(cached, mx.get_cache_memory())
            held = max(held, mx.get_active_memory() + mx.get_cache_memory())
    if defer_cooling:
        session.defer_settle()
    else:
        session.settle()
    result = comparison_from_samples(
        paired_means([t * 1e3 for t in base]),
        paired_means([t * 1e3 for t in cand]),
    )
    session.log(
        "compare",
        median_baseline_ms=result.median_baseline_ms,
        median_delta_ms=result.median_delta_ms,
        median_ratio=result.median_ratio,
        stability=round(result.stability, 3),
        sigma_ms=result.sigma_ms,
        n=result.n,
        max_held_gb=round(held / 1e9, 3),
        max_cache_gb=round(cached / 1e9, 3),
        peak_gb=round(mx.get_peak_memory() / 1e9, 3),
    )
    session.last_compare_peak_gb = round(mx.get_peak_memory() / 1e9, 3)
    return result


@_again_if_hot
def compare_on_inputs(
    session: Session,
    baseline: Callable[..., object],
    candidate: Callable[..., object],
    input_sets: Sequence[Sequence[object]],
    *,
    warm: Sequence[object] | None = None,
    prepare: Callable[[str, Sequence[object]], Sequence[object]] | None = None,
    charge_ms: Callable[[str], float] | None = None,
) -> PairedComparison:
    """compare(), but every timed call gets inputs no call before it saw:
    each block times both arms on two fresh input sets, a(p) b(p) b(q) a(q)
    or the mirror, so neither arm can gain from repeating an input and the
    arms see the same inputs. Needs an even number of sets, at least four.
    warm: the set warming uses, never timed (by default the first set).
    prepare(arm, inputs): an arm's own inputs for a set ("a" or "b"), made
    before its call and off the clock (a conversation read into its memory).
    charge_ms(arm): time added to every one of that arm's calls, asked after
    the last call (work prepare did that the other arm's prepare did not)."""
    if len(input_sets) < 4 or len(input_sets) % 2:
        raise ValueError(f"need an even number of input sets, at least 4; got {len(input_sets)}")
    own = prepare or (lambda arm, inputs: inputs)
    first = input_sets[0] if warm is None else warm
    # one arm's inputs at a time: on a model that nearly fills the GPU two
    # conversations held at once ran Qwen3.8-27B's timing out of memory
    warm_a = own("a", first)
    session.warm_until_stable(lambda: baseline(*warm_a))
    del warm_a
    warm_b = own("b", first)
    session.warm_until_stable(lambda: candidate(*warm_b))
    del warm_b
    base: list[float] = []
    cand: list[float] = []
    for block in range(len(input_sets) // 2):
        p, q = input_sets[2 * block], input_sets[2 * block + 1]
        if block % 2 == 0:
            calls = (("a", p), ("b", p), ("b", q), ("a", q))
        else:
            calls = (("b", p), ("a", p), ("a", q), ("b", q))
        for arm, inputs in calls:
            fn = baseline if arm == "a" else candidate
            args = own(arm, inputs)
            (base if arm == "a" else cand).append(session.timed(lambda: fn(*args)))
            del args
    session.settle()
    extra_a, extra_b = (charge_ms("a"), charge_ms("b")) if charge_ms else (0.0, 0.0)
    result = comparison_from_samples(paired_means([t * 1e3 + extra_a for t in base]),
                                     paired_means([t * 1e3 + extra_b for t in cand]))
    session.log("compare_on_inputs", median_baseline_ms=result.median_baseline_ms,
                median_delta_ms=result.median_delta_ms, median_ratio=result.median_ratio,
                sigma_ms=result.sigma_ms, n=result.n)
    return result
