"""Rest only once the chip is hot: a session with a heat watcher rests none of
its work while macOS reports no thermal pressure. The first pressure owes a
rest until it has cleared and a speed probe runs back inside the range the
same Mac read cool, then the session rests 2x its work for good."""

import json

import pytest

from backcountry.harness.measure import heat as heat_mod
from backcountry.harness.measure.heat import CALIBRATION_PROBES, FULL_SPEED_ENV, Heat
from backcountry.harness.measure.session import Session, Stopped


class Chip:
    """Scripted readings: each read takes the next (pressure, die) pair and
    each probe the next time in ms, the last of either repeating."""

    def __init__(self, *readings, probes=(100.0,)):
        self.readings, self.reads = list(readings), 0
        self.probes, self.probed = list(probes), 0

    def _next(self):
        return self.readings[min(self.reads, len(self.readings) - 1)]

    def pressure(self):
        return self._next()[0]

    def die(self):
        value = self._next()[1]
        self.reads += 1
        return value

    def probe(self):
        value = self.probes[min(self.probed, len(self.probes) - 1)]
        self.probed += 1
        return value


@pytest.fixture(autouse=True)
def no_inherited_range(monkeypatch):
    monkeypatch.delenv(FULL_SPEED_ENV, raising=False)


def session_on(chip, clock=None, tmp_path=None):
    clock = clock or [0.0]

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    slept = []
    heat = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: clock[0])
    s = Session(log_path=None if tmp_path is None else tmp_path / "session.jsonl",
                sleep=sleep, now=lambda: clock[0], heat=heat)
    return s, slept, clock


def kinds(s):
    return [json.loads(line)["kind"] for line in s._log_path.read_text().splitlines()]


def test_a_cool_chip_rests_none_of_its_work_and_measures_its_speed_once(tmp_path):
    chip = Chip((0, 70.0), probes=(100.0, 100.4, 100.2))
    s, slept, _ = session_on(chip, tmp_path=tmp_path)
    assert s.wait_ready() is False
    assert s.heat.full_speed_ms == (100.0, 100.4) and chip.probed == CALIBRATION_PROBES
    for _ in range(5):
        s.timed(lambda: None)
    s._debt_s = 600.0  # ten minutes of GPU work since the last rest
    s.settle()
    s.defer_settle()
    assert s.wait_ready() is False
    assert sum(slept) == 0 and s.heat_trips == 0 and chip.probed == CALIBRATION_PROBES
    assert kinds(s).count("heat_calibrated") == 1


def test_the_first_pressure_trips_for_good():
    chip = Chip((1, 60.0), (0, 50.0))
    h = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: 0.0)
    assert h.check(force=True)["reason"] == "macOS reported thermal pressure"
    assert h.check(force=True) is None and h.tripped["reason"] == "macOS reported thermal pressure"
    assert h.record()["policy"] == "rest 2x after macOS thermal pressure" and not h.failed


@pytest.mark.parametrize("reading", [(None, 60.0), (0, None)])
def test_readings_that_stop_working_switch_to_the_2x_rest_with_no_cooldown(reading):
    chip = Chip((0, 60.0), reading)
    s, slept, _ = session_on(chip)
    s.wait_ready()
    s._watch_heat(force=True)
    assert s.heat.failed and s.duty_idle_factor == 2.0 and not s._cooldown_owed
    assert s.heat.tripped["reason"] == "the heat readings stopped working"
    s._debt_s, slept[:] = 10.0, []
    s.settle()
    assert slept == [20.0]


def test_reads_at_most_every_few_seconds_unless_forced():
    chip, now = Chip((0, 60.0)), [0.0]
    h = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: now[0])
    h.check()
    h.check()
    now[0] += heat_mod.CHECK_EVERY_S - 1
    h.check()
    assert chip.reads == 1
    h.check(force=True)
    now[0] += heat_mod.CHECK_EVERY_S
    h.check()
    assert chip.reads == 3


def test_pressure_mid_comparison_waits_for_the_boundary_then_rests_until_full_speed(tmp_path):
    # read at the boundary, then at samples 2 and 3; pressure at the third,
    # clearing while the chip is still slow
    chip = Chip((0, 70.0), (0, 70.0), (1, 83.0), (1, 80.0), (0, 66.0),
                probes=[100.0, 100.2] * 3 + [103.0, 101.5, 100.3])
    s, slept, clock = session_on(chip, tmp_path=tmp_path)
    s.wait_ready()
    assert s.heat.full_speed_ms == (100.0, 100.2)
    for _ in range(3):
        s.timed(lambda: None)
        clock[0] += heat_mod.CHECK_EVERY_S  # samples a few seconds apart
    assert s.heat_trips == 1 and slept == []  # nothing rested between the samples of the comparison
    assert s.wait_ready() is True  # the boundary pays the cooldown, and says the clocks are cold
    # one rest under pressure (no probe), then back to full speed is the cool
    # range's slowest plus its width, 100.4 ms: 103 and 101.5 rest on, 100.3 ends it
    assert chip.probed == CALIBRATION_PROBES + 3 and slept == [15.0] * 4
    assert s.heat.cooldown_s == 60.0 and s.duty_idle_factor == 2.0
    s._debt_s, slept[:] = 10.0, []
    s.settle()
    assert slept == [20.0]  # from now on the 2x rest
    done = [json.loads(line) for line in s._log_path.read_text().splitlines()
            if json.loads(line)["kind"] == "heat_cooldown_done"][-1]
    assert done["back_to_full_speed"] and done["probe_ms"] == 100.3


def test_a_cooldown_never_ends_on_pressure_alone():
    chip = Chip((0, 60.0), probes=[100.0, 100.2] * 3 + [102.0, 102.0, 102.0, 100.1])
    h = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: 0.0)
    h.calibrate()
    assert h.recovered() is None and h.recovered() is None and h.recovered() is None
    assert h.recovered()["probe_ms"] == 100.1


def test_the_cool_range_follows_a_faster_cool_reading():
    chip = Chip((0, 50.0), probes=[100.0, 100.3] * 3 + [99.5])
    h = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: 0.0)
    assert h.calibrate() == (100.0, 100.3)
    assert h.recovered() and h.full_speed_ms == (99.5, 100.3)


def test_no_calibration_under_pressure():
    chip = Chip((1, 85.0))
    h = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: 0.0)
    assert h.calibrate() is None and chip.probed == 0


def test_workers_take_the_parents_cool_range(monkeypatch):
    chip = Chip((0, 50.0), probes=(100.0, 100.4))
    h = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: 0.0)
    h.calibrate()
    import os
    assert os.environ[FULL_SPEED_ENV] == "100.0,100.4"
    monkeypatch.setattr(heat_mod, "thermal_pressure", lambda: 0)
    monkeypatch.setattr(heat_mod, "die_c", lambda: 50.0)
    worker = Heat.for_machine()
    assert worker.full_speed_ms == (100.0, 100.4)


def test_a_stop_request_interrupts_the_cooldown():
    s, _, _ = session_on(Chip((1, 90.0)))
    s.stop_when = lambda: True
    s._watch_heat(force=True)
    with pytest.raises(Stopped):
        s.wait_ready()


def test_the_cooldown_gives_up_after_its_cap_and_keeps_resting_2x():
    s, slept, _ = session_on(Chip((1, 95.0)))
    s.wait_ready()
    assert heat_mod.MAX_COOLDOWN_S <= sum(slept) < heat_mod.MAX_COOLDOWN_S + 60
    assert s.duty_idle_factor == 2.0


def test_no_rest_due_means_no_ramp_back_up():
    s, _, _ = session_on(Chip((0, 60.0)))
    s._debt_s = 30.0
    calls = []
    assert s.fresh_chunk(lambda: calls.append(1)) is False
    assert calls == []


def test_without_readings_the_machine_rests_2x(monkeypatch):
    monkeypatch.setattr(heat_mod, "die_c", lambda: None)
    assert Heat.for_machine() is None
    assert Session().duty_idle_factor == 2.0  # a session given no watcher is unchanged


def test_the_report_records_the_range_the_trip_and_the_rest():
    chip = Chip((0, 61.0), (1, 84.2), (0, 70.0), probes=(100.0, 100.2))
    h = Heat(pressure=chip.pressure, die=chip.die, probe=chip.probe, now=lambda: 0.0)
    assert h.record()["policy"] == "no rest without pressure"
    h.calibrate()
    for _ in range(3):
        h.check(force=True)
    h.cooled(42.0)
    assert h.record() == {"policy": "rest 2x after macOS thermal pressure", "full_speed_ms": [100.0, 100.2],
                          "peak_die_c": 84.2, "cooldown_s": 42.0,
                          "tripped": {"reason": "macOS reported thermal pressure", "pressure": 1, "die_c": 84.2}}


def test_this_mac_reads_its_heat_and_its_speed():
    if heat_mod.thermal_pressure() is None or heat_mod.die_c() is None:
        pytest.skip("no heat readings on this machine")
    assert 0 <= heat_mod.thermal_pressure() <= 3
    assert 10 < heat_mod.die_c() < 120
    assert 1 < heat_mod.probe_ms() < 5000


def test_a_comparison_the_chip_went_hot_in_is_taken_again_after_the_cooldown(tmp_path):
    from backcountry.harness.measure.clocks import compare
    # cool for the first readings, pressure mid-way, then cool and back to full speed
    chip = Chip(*[(0, 70.0)] * 4, (1, 83.0), (0, 60.0))
    clock = [0.0]
    s, slept, _ = session_on(chip, clock, tmp_path=tmp_path)
    s.wait_ready()

    def step():
        clock[0] += heat_mod.CHECK_EVERY_S  # each call a few seconds of GPU work

    compare(s, step, step, pairs=2)
    seen = kinds(s)
    assert seen.count("heat_tripped") == 1 and seen.count("measured_again") == 1
    assert seen.index("heat_cooldown_done") < seen.index("measured_again")
    assert s.heat_trips == 1 and s.duty_idle_factor == 2.0
    first = seen.index("measured_again")
    assert "cooling" in seen[first:]  # the retake rests 2x like any later measurement


def test_a_cool_comparison_is_taken_once():
    from backcountry.harness.measure.clocks import compare
    s, _, _ = session_on(Chip((0, 60.0)))
    calls = []
    compare(s, lambda: calls.append("a"), lambda: calls.append("b"), pairs=2)
    assert calls and s.heat_trips == 0
