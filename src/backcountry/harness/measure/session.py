"""Measurement session: the timing recipe, duty-cycle pacing, warm-until-stable.

After a pacing idle the GPU runs at ramped-down clocks, so a block takes
unmeasured ramp samples until the reading stops falling before anything is
timed. Timing a sample right after an idle reads 1.5-1.6x slow and would bias
whichever arm ran first. Callers pace once per measurement, not between the
samples of one, so every sample is taken in the state the model runs in.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

import mlx.core as mx
import numpy as np

from backcountry.harness.log import json_safe, wall_now
from backcountry.harness.measure.heat import MAX_COOLDOWN_S, Heat
from backcountry.harness.sandbox.watchdog import gpu_window

# Idle after GPU work, as a multiple of it. 2026-10-07, Qwen3-30B-A3B time to
# first token at 512 tokens, identical model in both arms, 20 min each: 3x
# ranged 7% (from a warm start), 2x 1.7% and flat, 1.5x 3.5% with a possible
# +1% creep; paired ratios within 0.990-1.003 at all three. A session given a
# Heat watcher rests none of this until macOS first reports thermal pressure
# (measure/heat.py).
DUTY_IDLE_FACTOR = 2.0
STOP_CHECK_S = 5.0  # how often a pacing idle looks at its stop condition
MAX_CHUNK_WORK_S = 0.25
WARM_STABLE_RTOL = 0.01
WARM_STABLE_ABS_S = 100e-6  # 1% of a ~1ms kernel is under dispatch jitter
WARM_CAP = 60
# Readings that must fail to improve before the ramp is called done. The GPU
# ramps in steps, not smoothly: after a 0.75s idle the decode step sat at
# ~19 ms for five samples before dropping to ~9 (tools/ramp_after_idle.py).
# A smaller patience stops on a plateau and calls a cold machine warm.
WARM_PATIENCE = 10
# Compressed memory this process brings back on the CPU before GPU work
# (Session.resident); the GPU pages in less than this well inside its timeout.
PAGE_IN_GB = 0.5
PAGE_BYTES = 16384  # Apple Silicon's page
# A call of a second or more outlasts that ramp, and more warm calls only heat
# the chip: on Qwen3-VL-8B at 2,049 tokens (11 s a call) ten more readings
# rose 10,955 -> 12,330 ms and the timed samples started 12% slow. Two that
# fail to improve are enough; a paired comparison of the model against itself
# read 0.998-1.002 either way at 2.7 s and 11 s (2026-10-12).
WARM_LONG_CALL_S = 1.0
WARM_LONG_PATIENCE = 2


def time_once(fn: Callable[[], object]) -> float:
    """The one timing recipe: synchronize, run, eval outputs, synchronize.

    fn returns its outputs (an array or a tree of arrays); evaluating them here
    is what defeats laziness for whatever fn computed. A closure carrying a
    steps count (clocks.chained_steps) holds that many dependent model steps,
    and the reading is per step.
    """
    with gpu_window():
        mx.synchronize()
        t0 = time.perf_counter()
        outs = fn()
        mx.eval(outs)
        mx.synchronize()
        return (time.perf_counter() - t0) / getattr(fn, "steps", 1)


def _until_flat(timed, fn, rtol: float, abs_s: float, cap: int, patience: int) -> list[float]:
    """Sample until the reading stops falling, and return what was timed.

    Not "until two readings agree": while the GPU clocks are still ramping,
    consecutive readings agree with each other and both sit far above the
    steady state. That is how a paced step clock read 8-16 ms for a model
    that runs at 5.7.
    """
    times: list[float] = []
    best: float | None = None
    stale = 0
    for _ in range(cap):
        t = timed(fn)
        times.append(t)
        stale = 0 if best is None or best - t > max(rtol * best, abs_s) else stale + 1
        best = t if best is None else min(best, t)
        if stale >= (WARM_LONG_PATIENCE if best >= WARM_LONG_CALL_S else patience):
            break
    return times


class Stopped(Exception):
    """A pacing idle saw its stop condition (an operator's finish request)."""


class Session:
    """Owns pacing debt and the append-only health log for one measuring process.

    Timed samples accrue GPU-work debt. fresh_chunk() is the one gate every
    timed loop calls before a sample block: once debt passes the chunk size it
    pays the idle (3x work) and then ramps the clocks back up unmeasured.
    settle() pays whatever debt a measurement leaves behind.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        duty_idle_factor: float = DUTY_IDLE_FACTOR,
        max_chunk_work_s: float = MAX_CHUNK_WORK_S,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        heat: Heat | None = None,
    ):
        # with a heat watcher the 2x rest starts only once the chip is hot
        self.heat = heat
        self.duty_idle_factor = 0.0 if heat is not None else duty_idle_factor
        self._rest_factor = duty_idle_factor
        self._cooldown_owed = False
        self.heat_trips = 0  # a measurement this changes during is taken again (clocks._again_if_hot)
        # told when the heat watcher trips (the job's report records it)
        self.on_heat: Callable[[dict], None] | None = None
        self.max_chunk_work_s = max_chunk_work_s
        self._sleep = sleep
        self._now = now
        self._ready_at = 0.0
        self._debt_s = 0.0
        self._t0 = time.perf_counter()
        self._log_path = Path(log_path) if log_path else None
        self.idled_s = 0.0
        self.work_s = 0.0
        # the job's own log, which gets every memory line too
        self.on_memory: Callable[[dict], None] | None = None
        # checked through every pacing idle while set; true raises Stopped
        self.stop_when: Callable[[], bool] | None = None
        # the arrays GPU work here reads (the model's weights), brought back
        # into memory before it when macOS has compressed this process
        self.resident: Callable[[], Iterable[mx.array]] | None = None

    def memory(self, where: str, **detail: object) -> dict:
        """One memory line, written before heavy GPU work: what is in use now,
        the peak since the previous line, the GPU's working-set limit and swap.
        A job that runs out of memory dies in Metal as a GPU timeout, which
        names no cause; the last line says where it stood."""
        row = {"where": where, **detail, **memory_state()}
        mx.reset_peak_memory()
        self.log("memory", **row)
        if self.on_memory is not None:
            self.on_memory(row)
        return row

    def log(self, kind: str, **row: object) -> None:
        if self._log_path is None:
            return
        row = {"t": round(time.perf_counter() - self._t0, 3), "wall": wall_now(),
               "kind": kind, **row}
        with self._log_path.open("a") as f:
            f.write(json.dumps(json_safe(row), allow_nan=False) + "\n")

    def timed(self, fn: Callable[[], object]) -> float:
        self.wait_ready(between_samples=True)
        t = time_once(fn)
        work = t * getattr(fn, "steps", 1)  # the debt is the whole sample's GPU work
        self._debt_s += work
        self.work_s += work
        self._watch_heat()
        return t

    def _watch_heat(self, force: bool = False) -> None:
        """A trip switches this session to the 2x rest for good. The cooldown a
        pressure trip owes waits for the next measurement boundary (wait_ready),
        never between the samples of one comparison."""
        if self.heat is None or (reading := self.heat.check(force)) is None:
            return
        self.duty_idle_factor = self._rest_factor
        self._cooldown_owed = not self.heat.failed
        self.heat_trips += 1
        self.log("heat_tripped", **reading, peak_die_c=self.heat.peak_c, rest_factor=self._rest_factor)
        print(f"heat: {reading['reason']}; resting {self._rest_factor:g}x work from now on", flush=True)
        if self.on_heat is not None:
            self.on_heat(self.heat.record())

    def _cool_down(self) -> None:
        """After a trip: rest until macOS reports no pressure and the speed
        probe is back in this Mac's cool range (about three minutes on the
        fanless Air), at most MAX_COOLDOWN_S. A stop request still interrupts it."""
        started = self._now()
        self.log("heat_cooldown", peak_die_c=self.heat.peak_c)
        back = None
        while back is None and self._now() - started < MAX_COOLDOWN_S:
            self._idle(STOP_CHECK_S * 3, quiet=True)
            back = self.heat.recovered()
        seconds = self._now() - started
        self.heat.cooled(seconds)
        self._cooldown_owed = False
        self.log("heat_cooldown_done", seconds=round(seconds, 1), back_to_full_speed=back is not None,
                 **(back or {}), full_speed_ms=self.heat.full_speed_ms)

    def off_clock(self, fn: Callable[[], object], *, defer_cooling: bool = True, paced: bool = True,
                  where: str = "untimed model work"):
        """Run GPU work that evaluates its own results, outside any clock.

        Untimed work (capture, correctness checks) still heats the chip, so it
        owes 2x its time as rest like timed work, but the rest is not taken
        here: it is scheduled and the next GPU call waits out what is left, so
        Python and judge time in between count toward it. On Qwen3.8-27B rests
        taken straight after untimed work were 80 of 480 minutes of cooling.
        Unpaced work owes nothing (a run of recordings whose caller rests once
        before its timed phase, rest()). A memory line names where it ran.
        """
        self.wait_ready()
        mx.synchronize()
        self.memory(where)
        start = time.perf_counter()
        try:
            return fn()
        finally:
            mx.synchronize()
            elapsed = time.perf_counter() - start
            self.work_s += elapsed
            self._watch_heat()
            self.log("off_clock_work", seconds=round(elapsed, 3), **({} if paced else {"paced": False}))
            if paced:
                self._debt_s += elapsed
                if defer_cooling:
                    self.defer_settle()
                else:
                    self.settle()

    def fresh_chunk(self, fn: Callable[[], object]) -> bool:
        """Call once before a block of timed samples. Pays pacing debt when a
        chunk's worth has accrued, then ramps the clocks back up with
        unmeasured samples until the reading stops falling. Ramping is a
        property of the hardware, so one function brings the clocks up for
        every arm about to be measured."""
        cooled = self.wait_ready()
        if self._debt_s < self.max_chunk_work_s and not cooled:
            return False
        idle = self.duty_idle_factor * self._debt_s
        self._debt_s = 0.0
        if idle > 0:
            self._idle(idle)
        elif not cooled:
            return False  # no rest due (a cool chip under a heat watcher): nothing to ramp back from
        ramp = _until_flat(self.timed, fn, WARM_STABLE_RTOL, WARM_STABLE_ABS_S,
                           WARM_CAP, WARM_PATIENCE)
        # an idle past ~0.25s is not fully recoverable, so a ramp that never
        # flattens means the block below it is timing a cold machine: say so
        self.log("ramp", idle_s=round(idle, 3), n=len(ramp), flat=len(ramp) < WARM_CAP,
                 first_ms=ramp[0] * 1e3, last_ms=ramp[-1] * 1e3)
        return len(ramp) < WARM_CAP

    def rest(self, seconds: float) -> None:
        """One fixed cooldown before a timed phase that follows unpaced work."""
        self.wait_ready()
        self._idle(seconds)

    def settle(self) -> None:
        """Pay any remaining debt at the end of a measured phase."""
        self.wait_ready()
        if self._debt_s <= 0:
            return
        idle = self.duty_idle_factor * self._debt_s
        self._debt_s = 0.0
        self._idle(idle)

    def defer_settle(self) -> float:
        """Let CPU work use the cooldown; the next GPU phase pays the remainder.

        Judge thinking often outlasts the cooldown, so no sleep may follow;
        the cache is released here, where the timed phase ends."""
        cleared_gb = self._release_cache()
        if self._debt_s > 0:
            seconds = self.duty_idle_factor * self._debt_s
            self._debt_s = 0.0
            if seconds > 0:
                self._ready_at = max(self._now(), self._ready_at) + seconds
                self.log("cooling_scheduled", seconds=round(seconds, 3), cache_cleared_gb=cleared_gb)
        return self._ready_at

    def adopt_cooling(self, ready_at: float) -> None:
        """Accept a worker's deadline on this machine's monotonic clock.

        The worker has synchronized its GPU work before handing it back.
        Process teardown, parsing and judge thinking consume the same interval;
        transferring a duration instead would restart cooling in the parent.
        """
        if (isinstance(ready_at, bool) or not isinstance(ready_at, (int, float))
                or not math.isfinite(ready_at) or ready_at < 0):
            raise ValueError("worker cooling deadline must be finite and nonnegative")
        self._ready_at = max(self._ready_at, ready_at)
        self.log("cooling_adopted", remaining_s=round(max(0.0, ready_at - self._now()), 3))

    def wait_ready(self, between_samples: bool = False) -> bool:
        """Gate GPU work, including subprocess launches. Returns whether cooled.
        At a measurement boundary it also reads the heat, and pays a cooldown
        a trip left owed; between the samples of one comparison it does not."""
        cooled = False
        if not between_samples and self.heat is not None:
            if self.heat.full_speed_ms is None and not self.heat.failed and self.heat.calibrate():
                # this Mac's cool speed, before the job's first GPU work heats it
                self.log("heat_calibrated", full_speed_ms=list(self.heat.full_speed_ms))
            self._watch_heat(force=True)
            if self._cooldown_owed:
                self._cool_down()
                cooled = True
        self._page_in()
        if not self._ready_at:
            return cooled
        seconds = max(0.0, self._ready_at - self._now())
        if seconds > 0:
            self._idle(seconds)
        # If sleep is interrupted, a caller that resumes still owes the remainder.
        self._ready_at = 0.0
        self.log("cooling_reused", remaining_s=round(seconds, 3))
        return True

    def _page_in(self) -> None:
        """Before GPU work: memory macOS compressed while this process waited
        (on its candidate checks and the judge's tries, each its own process)
        comes back on the CPU. Qwen3.8-27B's job died four times on its first
        read after one: 13 GB of weights to restore from the GPU's side."""
        if self.resident is None or (compressed := _compressed_gb()) is None or compressed < PAGE_IN_GB:
            return
        start = time.perf_counter()
        touched = page_in(self.resident())
        self.log("paged_in", compressed_gb=round(compressed, 3), touched_gb=round(touched / 1e9, 3),
                 seconds=round(time.perf_counter() - start, 3), compressed_after_gb=_compressed_gb())

    @staticmethod
    def _release_cache() -> float:
        """Return MLX's reusable freed buffers to the system; the GB released.

        Across capture and many timed arms they grew to 14.5 GB on a 24 GB Mac,
        and under that memory pressure timed passes ran up to 7x slow. This
        runs only where a timed phase has ended: nothing is being measured, and
        every arm warms again before its first sample."""
        cached_gb = mx.get_cache_memory() / 1e9
        mx.clear_cache()
        return round(cached_gb, 3)

    def _idle(self, seconds: float, quiet: bool = False) -> None:
        self.log("cooling", seconds=round(seconds, 3), cache_cleared_gb=self._release_cache())
        if seconds >= 5.0 and not quiet:
            print(f"cooling: {seconds:.1f}s after GPU work", flush=True)
        started = self._now()
        try:
            if self.stop_when is None:
                self._sleep(seconds)
            else:
                left = seconds
                while left > 0:
                    if self.stop_when():
                        # what is left is still owed: the next GPU call waits it out
                        self._ready_at = max(self._ready_at, started + seconds)
                        raise Stopped("stop requested during a pacing idle")
                    self._sleep(min(left, STOP_CHECK_S))
                    left -= STOP_CHECK_S
        except BaseException:
            self.idled_s += min(seconds, max(0.0, self._now() - started))
            raise
        self.idled_s += seconds
        self.log("cooling_done", seconds=round(seconds, 3))

    def warm_until_stable(
        self,
        fn: Callable[[], object],
        rtol: float = WARM_STABLE_RTOL,
        abs_s: float = WARM_STABLE_ABS_S,
        cap: int = WARM_CAP,
    ) -> list[float]:
        """Law 3: first call thrown away (Metal compile), then sample until the
        reading stops falling. The absolute floor exists because 1% of a small
        kernel is under dispatch jitter. Returns the kept warmup timings."""
        first = self.timed(fn)
        times = _until_flat(self.timed, fn, rtol, abs_s, cap, WARM_PATIENCE)
        self.log("warm", n=len(times) + 1, stable=len(times) < cap,
                 first_ms=first * 1e3, work_s=round(first + sum(times), 3), last_ms=times[-1] * 1e3)
        return times


def _footprint_gb() -> float | None:
    """What macOS charges this process (proc_pid_rusage's phys_footprint):
    MLX's arrays plus what it does not count, compiled kernels and Python's own."""
    try:
        usage = (ctypes.c_uint64 * 64)()
        if ctypes.CDLL("/usr/lib/libSystem.dylib").proc_pid_rusage(os.getpid(), 0, ctypes.byref(usage)):
            return None
        return usage[9] / 1e9  # rusage_info_v0: a 16-byte uuid, then ri_phys_footprint 8th
    except (OSError, AttributeError):
        return None


def _compressed_gb() -> float | None:
    """How much of this process's memory macOS holds compressed (task_info's
    TASK_VM_INFO: a uint64 virtual size, two int32s, then uint64s, compressed 14th)."""
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
        info, count = (ctypes.c_uint32 * 200)(), ctypes.c_uint32(200)
        if lib.task_info(ctypes.c_uint32.in_dll(lib, "mach_task_self_"), 22, info, ctypes.byref(count)):
            return None
        return ctypes.cast(info, ctypes.POINTER(ctypes.c_uint64))[2 + 13] / 1e9
    except (OSError, AttributeError, ValueError):
        return None


def page_in(arrays) -> int:
    """Touch one byte in every page of these arrays from the CPU; the bytes
    touched, each buffer once. A page macOS compressed comes back on a CPU
    fault, with no watchdog; brought back by the GPU's first command instead,
    13 GB of Qwen3.8-27B's weights took past the GPU's timeout and killed the job."""
    seen, total = set(), 0
    for a in arrays:
        try:
            data = np.frombuffer(memoryview(a), dtype=np.uint8)
        except (TypeError, ValueError, BufferError):  # not a plain buffer: the GPU brings it
            continue
        address = data.__array_interface__["data"][0]
        if address in seen or not data.size:
            continue
        seen.add(address)
        int(data[::PAGE_BYTES].sum())
        total += data.nbytes
    return total


def memory_state() -> dict[str, float | None]:
    """MLX's GPU memory, its peak since the last reset, the GPU's working-set
    limit, the process's whole footprint, and the machine's free memory and
    swap in use, in GB, so a slow block can be told apart from a machine that
    was paging, and a creeping process from a model at its limit."""
    try:
        text = subprocess.run(["sysctl", "-n", "vm.swapusage", "vm.page_free_count", "hw.pagesize"],
                              capture_output=True, text=True, timeout=5).stdout
        used = re.search(r"used = ([0-9.]+)M", text)
        swap_gb = float(used.group(1)) / 1024 if used else None
        pages, size = (int(n) for n in text.split("\n")[1:3])
        free_gb = pages * size / 1e9
    except (OSError, subprocess.SubprocessError, ValueError):
        swap_gb = free_gb = None
    return {"mlx_active_gb": mx.get_active_memory() / 1e9, "mlx_cache_gb": mx.get_cache_memory() / 1e9,
            "mlx_peak_gb": mx.get_peak_memory() / 1e9,
            "gpu_limit_gb": mx.device_info()["max_recommended_working_set_size"] / 1e9,
            "footprint_gb": _footprint_gb(), "compressed_gb": _compressed_gb(),
            "free_gb": free_gb, "swap_used_gb": swap_gb}
