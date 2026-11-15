"""A marked function's own code made faster: init asks whether the search may
rewrite it (same outputs), the loop search finds the rewrite, checks it
against the original and ships it beside the app, and the app, run as it
always is, uses it from its first call. The judge is a script that proposes
one known rewrite (no AI, no network)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from .apps import ENGINE, WASTEFUL
from .conftest import backcountry

pytestmark = pytest.mark.integration

LOOP_JUDGE = f"{sys.executable} {Path(__file__).parent / 'fake_loop_judge.py'}"


def app_run(folder, **env) -> tuple[str, float, str]:
    """python app.py as the user runs it: (checksum, ms a call, what it said)."""
    import os
    done = subprocess.run([sys.executable, "app.py"], cwd=folder, capture_output=True, text=True, timeout=600,
                          env={**os.environ, **env})
    assert done.returncode == 0, done.stderr
    checksum, ms = re.search(r"checksum (\S+) ms (\S+)", done.stdout).groups()
    return checksum, float(ms), done.stderr


def test_the_marked_function_itself_is_rewritten_checked_and_used_by_the_app(app):
    folder = app(WASTEFUL)
    asked = backcountry("init", "app.py", cwd=folder, answers=[""] * 10)
    assert asked.code == 0, asked
    assert "May the search also rewrite how score uses the model" in asked.out
    assert "a faster version of score itself first (the same outputs)" in asked.out
    assert yaml.safe_load((folder / "backcountry.yaml").read_text())["keep"] == "same-output"
    assert backcountry("init", "app.py", "--yes", "--attempts", "1", cwd=folder).code == 0
    done = backcountry("run", "backcountry.yaml", "--judge-cmd", LOOP_JUDGE, "--work-dir", "runs/loop",
                       cwd=folder, timeout=1800)
    assert done.code == 0, done
    assert re.search(r"inference loop: [\d.]+x against the job's default loop", done.out), done.out
    import json
    events = [json.loads(line) for line in (folder / "runs/loop/run.jsonl").read_text().splitlines()]
    profile = next(e["profile"] for e in events if e["kind"] == "model_search_profile")
    assert profile["calls"][0]["count"] == 8, profile   # the judge sees the model's calls inside score
    result = folder / "backcountry-results" / "score"
    assert (result / "run.py").is_file()
    faster, fast_ms, said = app_run(folder)
    assert "score runs a faster version of itself (the same outputs)" in said
    plain, plain_ms, _ = app_run(folder, BACKCOUNTRY_OFF="1")
    assert faster == plain               # the same outputs
    assert fast_ms < plain_ms / 2        # one product where there were eight
    # the project, result included, cloned to another folder: the result still applies there
    clone = folder.parent / "clone"
    shutil.copytree(folder, clone, ignore=shutil.ignore_patterns("runs", ".backcountry"))
    _, _, said = app_run(clone)
    assert "score runs a faster version of itself (the same outputs)" in said
    # score edited after the run: the rewrite of the old code no longer answers for it
    app_file = clone / "app.py"
    app_file.write_text(app_file.read_text().replace("    return np.array(y)\n", "    return np.array(y) * 2\n"))
    edited, _, said = app_run(clone)
    assert "score has changed since its faster version in" in said and "was made; it runs as written" in said
    assert float(edited) == pytest.approx(2 * float(plain), rel=1e-4)


def test_same_math_keeps_the_function_as_written(app):
    folder = app(WASTEFUL)
    assert backcountry("init", "app.py", "--yes", "--keep", "same-math", cwd=folder).code == 0
    assert yaml.safe_load((folder / "backcountry.yaml").read_text())["keep"] == "same-math"
    model_file = (folder / ".backcountry" / "backcountry_model.py").read_text()
    assert "def run(" not in model_file


def test_an_engine_that_runs_its_model_on_its_own_thread_gets_its_handler_rewritten_and_used(app):
    """Any serving engine, not one product: the handler hands the request to a
    worker thread that runs the model on a stream of its own, and the engine
    holds the model as engine.model beside the app's global."""
    folder = app(ENGINE)
    assert backcountry("init", "app.py", "--yes", "--attempts", "1", cwd=folder).code == 0
    done = backcountry("run", "backcountry.yaml", "--judge-cmd", LOOP_JUDGE, "--work-dir", "runs/loop",
                       cwd=folder, timeout=1800)
    assert done.code == 0, done
    assert re.search(r"inference loop: [\d.]+x against the job's default loop", done.out), done.out
    faster, fast_ms, said = app_run(folder)
    assert "handle runs a faster version of itself (the same outputs)" in said
    plain, plain_ms, _ = app_run(folder, BACKCOUNTRY_OFF="1")
    assert faster == plain               # the same reply
    assert fast_ms < plain_ms * 0.75     # one reply asked for where there were two
