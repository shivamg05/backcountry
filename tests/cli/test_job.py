"""A job from the command line, end to end, with a judge that proposes
nothing (no AI, no network): `run` in the foreground, and `start` in the
background controlled by `status`, `pause`, `resume` and `finish`. Both end
the way a run with no win does: said plainly, nothing written beside the
app."""

import json
import re
import time

import pytest

from .apps import FORECAST
from .conftest import FAKE_JUDGE, backcountry

pytestmark = pytest.mark.integration


def status(work, cwd) -> dict:
    done = backcountry("status", "--work-dir", work, "--json", cwd=cwd)
    assert done.code == 0, done
    return json.loads(done.out)


def wait_until(work, cwd, done, minutes=20) -> dict:
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        facts = status(work, cwd)
        if done(facts):
            return facts
        time.sleep(5)
    raise AssertionError(f"still {facts['state']} after {minutes} min: {facts}")


@pytest.fixture
def initialized(app):
    folder = app(FORECAST)
    assert backcountry("init", "app.py", "--yes", "--attempts", "2", cwd=folder).code == 0
    return folder


def test_run_in_the_foreground_ends_with_its_result_said_plainly(initialized):
    done = backcountry("run", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, "--work-dir", "runs/fg",
                       cwd=initialized, timeout=1800)
    assert done.code == 0, done
    assert "no confirmed improvement" in done.out
    assert not (initialized / "backcountry-results").exists()


def test_start_then_pause_resume_and_finish(initialized):
    work = initialized / "runs" / "bg"
    started = backcountry("start", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, "--work-dir", work,
                          cwd=initialized)
    assert started.code == 0 and f"started in the background: {work}" in started.out, started
    assert "backcountry status" in started.out
    assert backcountry("pause", "--work-dir", work, cwd=initialized).code == 0
    paused = wait_until(work, initialized, lambda f: not f["running"])
    assert paused["state"] == "paused", paused
    text = backcountry("status", "--work-dir", work, cwd=initialized).out
    assert "paused" in text.splitlines()[0]
    resumed = backcountry("resume", "--work-dir", work, "--background", cwd=initialized)
    assert resumed.code == 0, resumed
    wait_until(work, initialized, lambda f: f["running"], minutes=2)
    assert backcountry("finish", "--work-dir", work, cwd=initialized).code == 0
    ended = wait_until(work, initialized, lambda f: not f["running"])
    assert ended["state"] not in ("stopped", "paused") and ended["problem"] is None, ended
    assert ended["shipped"] == [] and ended["outcome"] == "no_improvement", ended
    said = backcountry("status", "--work-dir", work, cwd=initialized).out
    assert re.search(r" · complete · \d+ of 2 attempts$", said.splitlines()[0]), said    # no stale stage
    assert "outcome: no confirmed improvement; no artifact produced" in said
    assert not (initialized / "backcountry-results").exists()
