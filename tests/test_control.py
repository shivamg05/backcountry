"""Run control: the newest run, status as one screen or JSON, waiting for the
next milestone, a background start, and which AI writes kernels by default."""

import json
import os
import time

import pytest

from backcountry.harness import control


def _run(root, name, events=(), report=None, kernels=None, budget=12):
    work = root / name
    work.mkdir(parents=True)
    (work / "manifest.yaml").write_text(f"model: model.py\nbudget: {{per_region: 6, total: {budget}}}\n")
    (work / "job.json").write_text(json.dumps({"manifest": str(work / "manifest.yaml"), "artifact": str(work / "artifact")}))
    (work / "run.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (work / "report.json").write_text(json.dumps(report or {"session": {"status": "running", "stage": "searching kernels"}}))
    if kernels is not None:
        (work / "kernels").mkdir()
        (work / "kernels" / "run.jsonl").write_text("".join(json.dumps(e) + "\n" for e in kernels.pop("events", ())))
        (work / "kernels" / "report.json").write_text(json.dumps(kernels))
    return work


def test_control_commands_act_on_the_newest_run_unless_one_is_named(tmp_path):
    with pytest.raises(ValueError, match="no runs under"):
        control.resolve(None, tmp_path)
    old = _run(tmp_path, "a")
    new = _run(tmp_path, "b")
    os.utime(old / "run.jsonl", (time.time() - 100, time.time() - 100))
    os.utime(old / "job.json", (time.time() - 100, time.time() - 100))
    assert control.resolve(None, tmp_path) == new
    assert control.resolve(str(old), tmp_path) == old
    with pytest.raises(ValueError, match="not a run folder"):
        control.resolve(str(tmp_path), tmp_path)


def test_status_reads_the_run_and_its_kernel_search(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "running", lambda work: True)
    work = _run(tmp_path, "r", report={"session": {"status": "running", "stage": "searching kernels on the loop's calls"},
                                       "model_level": {"final": {"ok": True, "speedup": 1.527}}},
                kernels={"hypotheses": [{"id": "scaffold"}, {"id": "h1"}, {"id": "h2"}],
                         "regions": [{"fingerprint": "9a124b29aaaa", "hypotheses": 2}],
                         "events": [{"wall": "2026-11-12T01:00:00", "kind": "shipped", "kernel": "h1"},
                                    {"wall": "2026-11-12T01:31:00", "kind": "job_failed", "reason": "Insufficient Memory"}]})
    facts = control.summary(work)
    assert facts["attempts"] == 2 and facts["attempts_budget"] == 12 and facts["shipped"] == ["h1"]
    assert facts["loop_speedup"] == 1.527 and facts["current_region"] == "9a124b29"
    assert "Insufficient Memory" in facts["problem"]["error"]
    screen = control.render(facts)
    assert "2 of 12 attempts" in screen and "1.527x" in screen and "last problem" in screen
    with (work / "run.jsonl").open("a") as log:  # a restart after the crash clears the problem
        log.write(json.dumps({"wall": "2026-11-12T01:40:00", "kind": "resumed"}) + "\n")
    assert control.summary(work)["problem"] is None
    monkeypatch.setattr(control, "running", lambda work: False)
    assert control.summary(work)["state"] == "stopped" and "backcountry resume" in control.render(control.summary(work))


def test_wait_returns_at_the_next_milestone_or_when_the_job_ends(tmp_path, monkeypatch):
    work = _run(tmp_path, "w", events=[{"wall": "2026-11-12T00:00:00", "kind": "region_open"}])
    monkeypatch.setattr(control, "running", lambda work: True)

    def sleep(seconds):  # the job ships a kernel while we wait
        with (work / "run.jsonl").open("a") as log:
            log.write(json.dumps({"wall": "2026-11-12T00:05:00", "kind": "lesson"}) + "\n")
            log.write(json.dumps({"wall": "2026-11-12T00:06:00", "kind": "shipped", "kernel": "h3"}) + "\n")

    assert control.wait(work, sleep=sleep)["milestone"]["kind"] == "shipped"
    monkeypatch.setattr(control, "running", lambda work: False)
    assert control.wait(work, sleep=lambda s: None)["milestone"]["kind"] == "process_ended"


def test_the_default_ai_is_a_signed_in_cli_else_the_api(monkeypatch):
    found = {"codex"}
    monkeypatch.setattr(control.shutil, "which", lambda name: f"/bin/{name}" if name in found else None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert control.default_judge() == "codex"
    found.add("claude")
    assert control.default_judge() == "claude-cli"
    found.clear()
    with pytest.raises(ValueError, match="no AI to write kernels"):
        control.default_judge()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert control.default_judge() == "api"


def test_start_runs_the_job_in_the_background_and_says_why_one_stopped(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "fake_python"
    fake.write_text("#!/bin/sh\necho loading model\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setattr(control.sys, "executable", str(fake))
    work = tmp_path / "run"
    assert control.start(["m.yaml"], work, ready_s=20) == 0
    pid = int((work / control.PID).read_text())
    assert control.running(work) and "started in the background" in capsys.readouterr().out
    os.killpg(pid, 15)
    fake.write_text("#!/bin/sh\necho 'no such manifest' >&2\nexit 2\n")
    assert control.start(["m.yaml"], tmp_path / "run2", ready_s=20) == 1
    assert "no such manifest" in capsys.readouterr().err


def test_pause_without_a_folder_reaches_the_newest_run(tmp_path, monkeypatch):
    from backcountry.harness.cli import main
    from backcountry.harness.resume import PAUSE_REQUEST
    work = _run(tmp_path / "runs", "only")
    monkeypatch.chdir(tmp_path)
    assert main(["pause"]) == 0
    assert (work / PAUSE_REQUEST).is_file()


def test_resume_in_the_background_starts_the_resume_command(tmp_path, monkeypatch):
    from backcountry.harness.cli import main
    work = _run(tmp_path / "runs", "only")
    started = []
    monkeypatch.setattr(control, "start", lambda argv, w, command="run", **kw: started.append((argv, w, command)) or 0)
    monkeypatch.chdir(tmp_path)
    assert main(["resume", "--background"]) == 0
    argv, where, command = started[0]
    assert command == "resume" and "--background" not in argv and argv[-2] == "--work-dir"
    assert (tmp_path / argv[-1]).resolve() == work.resolve()


def test_guide_prints_the_steps(capsys):
    from backcountry.harness.cli import main
    assert main(["guide"]) == 0
    text = capsys.readouterr().out
    for command in ("init", "--questions", "check", "start", "status --wait", "finish"):
        assert command in text
