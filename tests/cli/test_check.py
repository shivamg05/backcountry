"""`backcountry check` on a marked function: the AI answers, then each chosen
size's time and headroom, named as init listed it, and nothing is changed."""

import re
import shutil

import pytest

from .apps import CACHED, FORECAST, JITTER, SERVER, TIMED
from .conftest import FAKE_JUDGE, backcountry


def test_an_ai_that_does_not_answer_is_said_before_any_model_loads(app):
    folder = app(FORECAST)
    assert backcountry("init", "app.py", "--yes", cwd=folder).code == 0
    done = backcountry("check", "backcountry.yaml", "--judge-cmd", "false", cwd=folder)
    assert done.code != 0 and "Judge readiness check failed" in done.out
    assert "loading model" not in done.out



def test_a_missing_backcountry_yaml_is_said_plainly_by_check_start_and_run(tmp_path):
    for command in ("check", "start", "run"):
        done = backcountry(command, "backcountry.yaml", cwd=tmp_path)
        assert done.code == 2 and "backcountry.yaml not found: write it with `backcountry init" in done.out, done
        assert "Traceback" not in done.out

@pytest.mark.integration
def test_check_names_each_size_as_init_listed_it_with_its_headroom(app):
    folder = app(FORECAST)
    assert backcountry("init", "app.py", "--yes", cwd=folder).code == 0
    done = backcountry("check", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, cwd=folder)
    assert done.code == 0, done
    summary = done.out[done.out.index("check: backcountry.yaml"):]
    assert "signed in and answering" in summary
    for series in (4, 16):
        line = next(l for l in summary.splitlines() if f"history axis 0={series}," in l)
        assert " ms a call" in line and "% headroom" in line, line
    assert "headroom (how much faster this Mac could run it at best; an estimate, not a promise)" in summary
    assert summary.rstrip().endswith("next: backcountry start backcountry.yaml")
    assert not (folder / "backcountry-results").exists()        # a check changes nothing


@pytest.mark.integration
def test_a_request_typed_at_init_is_what_the_server_app_is_measured_on(app):
    folder = app(SERVER)
    assert backcountry("init", "app.py", "--ask", "--yes", "--request", "question=what is mlx and why is it fast",
                       cwd=folder).code == 0
    done = backcountry("check", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, cwd=folder)
    assert done.code == 0, done
    assert "question=8 tokens, max_tokens=4: " in done.out and " ms a call" in done.out


@pytest.mark.integration
def test_a_function_that_does_less_work_for_a_request_it_has_seen_is_stopped_at_check(app):
    folder = app(CACHED)
    assert backcountry("init", "app.py", "--ask", "--yes", "--request", "question=what is mlx and why is it fast",
                       cwd=folder).code == 0
    done = backcountry("check", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, cwd=folder)
    assert done.code != 0
    assert "did different work the second time it got the same request" in done.out, done.out
    assert "prompt cache" in done.out
    assert "Traceback" not in done.out


def test_a_function_that_returns_its_speed_with_its_answer_is_stopped_at_check(app):
    folder = app(TIMED)
    assert backcountry("init", "app.py", "--yes", cwd=folder).code == 0
    done = backcountry("check", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, cwd=folder)
    assert done.code != 0
    assert "returned something different the second time it got the same request (result['tokens_per_sec'])" in done.out, done.out
    assert "result['text']" not in done.out and "Traceback" not in done.out


def test_a_result_that_repeats_only_to_rounding_is_graded_within_the_default_tolerance(app):
    # Kokoro (mlx-audio) gives the same audio to within 1e-7, not bit for bit.
    # Under 'exact' check points at 'default'; under 'default' every kernel is
    # graded within it (the user's call, 2026-11-15) and check goes on
    folder = app(JITTER)
    assert backcountry("init", "app.py", "--yes", "--tolerance", "exact", cwd=folder).code == 0
    done = backcountry("check", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, cwd=folder)
    assert done.code != 0 and "Traceback" not in done.out
    assert "but not bit for bit" in done.out and "Set 'tolerance: default' in backcountry.yaml" in done.out, done.out
    assert backcountry("init", "app.py", "--yes", "--tolerance", "default", cwd=folder).code == 0
    done = backcountry("check", "backcountry.yaml", "--judge-cmd", FAKE_JUDGE, cwd=folder)
    assert done.code == 0, done
    assert "every kernel is graded within the tolerance" in done.out


@pytest.mark.integration
def test_a_cloned_project_runs_with_the_sizes_its_yaml_says_now(app, tmp_path_factory):
    folder = app(SERVER)
    assert backcountry("init", "app.py", "--ask", "--yes", "--request", "question=what is mlx and why is it fast",
                       "--sizes", "1=8-64", cwd=folder).code == 0
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    clone = elsewhere / "project"
    shutil.copytree(folder, clone, ignore=shutil.ignore_patterns("backcountry_recording.json"))
    intent = clone / "backcountry.yaml"
    intent.write_text(re.sub(r"  - \{question: (13|23|38), max_tokens: 4\}\n", "", intent.read_text()))
    done = backcountry("check", str(intent), "--judge-cmd", FAKE_JUDGE, cwd=elsewhere)
    assert done.code == 0, done
    timed = re.findall(r"question=(\d+) tokens, max_tokens=4: ", done.out)
    assert timed == ["8", "64"], done.out
