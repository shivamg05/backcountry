"""The chip's heat, read without root: macOS thermal pressure, the hottest die
sensor, and a short GPU speed probe. A job rests none of its work while macOS
reports no thermal pressure. The first pressure trips it, for the rest of the
job: it rests until pressure has cleared and the probe runs back inside the
range the same Mac read when cool, then rests 2x its work like a Mac without
readings (session.DUTY_IDLE_FACTOR). No temperature decides anything (die
temperatures differ by chip); the hottest is only recorded.

Measured 2026-11-11 on the MacBook Air M4 (fanless), probe = 20 chained fp16
2048^3 matmuls, median of 5: cool, 104.3-104.6 ms with 0.1-0.8 ms between its
5 runs. Under nonstop load it slowed smoothly with the die (113.5 ms, +8.7%, at
83 C) while pressure stayed nominal and each probe stayed steady (0.1-0.5 ms);
paired comparisons cancel a smooth drift like that, and the Mac mini, steady at
84 C with no pressure, timed the model against itself at 0.9991-1.0003. At
193 s macOS reported pressure and the chip turned erratic: 114-125 ms, 2-21 ms
between runs, the jumps that spoil a measurement. Idle, pressure cleared in
about 30 s with the probe still 2% slow (resuming there trips again at once);
the probe was back in its cool range about three minutes after the load.
Why 2x after the trip, not just a rest back to full speed: the probe recovers
in under a minute once the load stops, but a fanless Mac's body stays warm. On
the Air (Qwen3-8B at 1,833 tokens, the model timed against itself, 4 pairs),
resting only until the probe recovered (46-125 s) let pressure return 40-60 s
into every comparison: 0.988-1.014, reads 10.3-13.8 s. Cooling once and then
resting 2x read 1.017 in the comparison it tripped in (taken again) and
0.9995-1.0016 after it, reads 9.4 s; 2x from the start, 0.995-1.008.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import statistics
import time
from typing import Callable

CHECK_EVERY_S = 5.0  # a die read costs ~50 ms, so a job reads at most this often
MAX_COOLDOWN_S = 1800.0  # a Mac that never gets back to full speed (another load on the GPU) still goes on
CALIBRATION_PROBES = 6
FULL_SPEED_ENV = "BACKCOUNTRY_FULL_SPEED_MS"  # the parent's cool range, for its worker processes

_TEMPERATURE_EVENT = 15


def thermal_pressure() -> int | None:
    """NSProcessInfo.thermalState: 0 nominal, 1 fair, 2 serious, 3 critical."""
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        info = send(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        return int(send_long(info, objc.sel_registerName(b"thermalState"))) if info else None
    except (OSError, AttributeError, TypeError):
        return None


def die_c() -> float | None:
    """The hottest of the chip's die sensors ("PMU tdie*"), through IOHID."""
    try:
        return _hottest_die()
    except (OSError, AttributeError, TypeError, ValueError):
        return None


def _hottest_die() -> float | None:
    v = ctypes.c_void_p
    iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
    cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    for fn, res, args in (
            (cf.CFStringCreateWithCString, v, [v, ctypes.c_char_p, ctypes.c_uint32]),
            (cf.CFNumberCreate, v, [v, ctypes.c_int, v]),
            (cf.CFDictionaryCreate, v, [v, v, v, ctypes.c_long, v, v]),
            (cf.CFArrayGetCount, ctypes.c_long, [v]),
            (cf.CFArrayGetValueAtIndex, v, [v, ctypes.c_long]),
            (cf.CFStringGetCString, ctypes.c_bool, [v, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
            (cf.CFRelease, None, [v]),
            (iokit.IOHIDEventSystemClientCreate, v, [v]),
            (iokit.IOHIDEventSystemClientSetMatching, None, [v, v]),
            (iokit.IOHIDEventSystemClientCopyServices, v, [v]),
            (iokit.IOHIDServiceClientCopyProperty, v, [v, v]),
            (iokit.IOHIDServiceClientCopyEvent, v, [v, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]),
            (iokit.IOHIDEventGetFloatValue, ctypes.c_double, [v, ctypes.c_int32])):
        fn.restype, fn.argtypes = res, args
    text = lambda s: cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)
    number = lambda n: cf.CFNumberCreate(None, 3, ctypes.byref(ctypes.c_int32(n)))
    keys = (v * 2)(text("PrimaryUsagePage"), text("PrimaryUsage"))
    values = (v * 2)(number(0xFF00), number(5))  # Apple's vendor page, temperature sensors
    client = iokit.IOHIDEventSystemClientCreate(None)
    if not client:
        return None
    iokit.IOHIDEventSystemClientSetMatching(client, cf.CFDictionaryCreate(None, keys, values, 2, None, None))
    services = iokit.IOHIDEventSystemClientCopyServices(client)
    if not services:
        return None
    product, name, hottest = text("Product"), ctypes.create_string_buffer(256), None
    for i in range(cf.CFArrayGetCount(services)):
        service = cf.CFArrayGetValueAtIndex(services, i)
        label = iokit.IOHIDServiceClientCopyProperty(service, product)
        if not label:
            continue
        cf.CFStringGetCString(label, name, 256, 0x08000100)
        cf.CFRelease(label)
        if b"tdie" not in name.value:
            continue
        event = iokit.IOHIDServiceClientCopyEvent(service, _TEMPERATURE_EVENT, 0, 0)
        if not event:
            continue
        c = iokit.IOHIDEventGetFloatValue(event, _TEMPERATURE_EVENT << 16)
        cf.CFRelease(event)
        if 0 < c < 150:
            hottest = c if hottest is None else max(hottest, c)
    return hottest


def probe_ms() -> float:
    """The median of 5 timed runs (after one untimed) of 20 chained fp16
    2048x2048 matmuls: about 0.1 s each on an M4, compute-bound, so it slows
    with the GPU's clock."""
    import mlx.core as mx
    global _PROBE
    if _PROBE is None:
        a, b = (mx.random.normal((2048, 2048), key=mx.random.key(i)).astype(mx.float16) for i in (1, 2))
        mx.eval(a, b)
        _PROBE = a, b

    def chain():
        x, b = _PROBE
        for _ in range(20):
            x = (x @ b) * 0.01
        return x

    mx.eval(chain())
    mx.synchronize()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        mx.eval(chain())
        mx.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


_PROBE = None


class Heat:
    """Tells a job when to rest. The first macOS thermal pressure, or readings
    that stop working, trips it for good; the cooldown that follows a pressure
    trip lasts until pressure has cleared and the speed probe reads inside the
    range this Mac showed cool, measured before the job's first GPU work (or
    handed down by the parent process). The session rests 2x from then on."""

    def __init__(self, pressure: Callable[[], int | None] | None = None,
                 die: Callable[[], float | None] | None = None, probe: Callable[[], float] | None = None,
                 now: Callable[[], float] = time.monotonic, full_speed_ms: tuple[float, float] | None = None):
        self._pressure, self._die, self._probe = pressure or thermal_pressure, die or die_c, probe or probe_ms
        self._now = now
        self._last = None
        self.full_speed_ms = full_speed_ms  # (fastest, slowest) cool probe
        self.tripped: dict | None = None
        self.failed = False
        self.cooldown_s: float | None = None
        self.peak_c: float | None = None

    @classmethod
    def for_machine(cls) -> Heat | None:
        """A watcher when pressure and the die read here, else None (the 2x
        rest). A worker process takes its parent's cool range."""
        given = os.environ.get(FULL_SPEED_ENV)
        try:
            full = tuple(float(x) for x in given.split(",")) if given else None
        except ValueError:
            full = None
        heat = cls(full_speed_ms=full if full and len(full) == 2 else None)
        return heat if heat.read() is not None else None

    def calibrate(self) -> tuple[float, float] | None:
        """The cool range: CALIBRATION_PROBES probes on a Mac under no pressure,
        shared with worker processes through the environment. None while there
        is pressure (the first cooldown calibrates once it clears)."""
        if self.full_speed_ms is None and self._pressure() == 0:
            runs = [self._probe() for _ in range(CALIBRATION_PROBES)]
            self.full_speed_ms = (round(min(runs), 3), round(max(runs), 3))
            os.environ[FULL_SPEED_ENV] = ",".join(str(x) for x in self.full_speed_ms)
        return self.full_speed_ms

    def read(self) -> dict | None:
        pressure, die = self._pressure(), self._die()
        if pressure is None or die is None:
            return None
        self.peak_c = die if self.peak_c is None else max(self.peak_c, die)
        return {"pressure": pressure, "die_c": round(die, 1)}

    def check(self, force: bool = False) -> dict | None:
        """The reading that trips the watcher, once; None otherwise. Reads at
        most every CHECK_EVERY_S unless forced. A reading that fails counts as
        hot: without it nothing says the chip is cool."""
        if self.tripped is not None or (not force and self._last is not None
                                        and self._now() - self._last < CHECK_EVERY_S):
            return None
        self._last = self._now()
        reading = self.read()
        if reading is None:
            self.failed = True
            self.tripped = {"reason": "the heat readings stopped working"}
        elif reading["pressure"] > 0:
            self.tripped = {"reason": "macOS reported thermal pressure", **reading}
        return self.tripped

    def recovered(self) -> dict | None:
        """The reading that ends a cooldown: no pressure, and the probe inside
        the cool range (its slowest run plus the range's own width). None while
        the chip is still under pressure or slow."""
        reading = self.read()
        if reading is None or reading["pressure"] > 0 or self.calibrate() is None:
            return None
        fast, slow = self.full_speed_ms
        ms = self._probe()
        if ms > slow + (slow - fast):
            return None
        if ms < fast:  # a faster cool reading than calibration saw: the range follows it
            self.full_speed_ms = (round(ms, 3), slow)
        return {**reading, "probe_ms": round(ms, 3)}

    def cooled(self, seconds: float) -> None:
        self.cooldown_s = round(seconds, 1)

    def record(self) -> dict:
        """What the report says about how this job rested."""
        return {"policy": "rest 2x after macOS thermal pressure" if self.tripped else "no rest without pressure",
                "full_speed_ms": None if self.full_speed_ms is None else list(self.full_speed_ms),
                "peak_die_c": None if self.peak_c is None else round(self.peak_c, 1),
                "tripped": self.tripped, "cooldown_s": self.cooldown_s}
