"""Run control for people and agents alike: the newest run, a run started in
the background, and one screen (or JSON) on how a run is going.

    backcountry start mlx-community/Qwen3-8B-4bit --attempts 100
    backcountry status --wait      # returns at the next milestone
    backcountry pause | resume | finish | skip   # the newest run unless --work-dir names one
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

RUNS = Path("runs")
# what `status --wait` returns at: a kernel shipped, a region opened or closed,
# a pause, a resume or a crash, the loop search's end and the final result
MILESTONES = ("shipped", "region_open", "region_closed", "paused", "resumed", "job_failed",
              "model_search_done", "final_e2e", "artifact_checked", "memory_untimed")
CONSOLE = "console.log"
PID = "job.pid"


def newest_run(root: Path = RUNS) -> Path | None:
    """The run folder whose log changed last; None when there is none."""
    runs = [d for d in Path(root).glob("*") if (d / "job.json").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda d: max((f.stat().st_mtime for f in (d / "run.jsonl", d / "kernels" / "run.jsonl",
                                                                    d / "job.json") if f.is_file()), default=0))


def resolve(work_dir: str | None, root: Path = RUNS) -> Path:
    """The run a control command acts on: the one named, else the newest."""
    if work_dir:
        work = Path(work_dir)
        if not (work / "job.json").is_file() and not (work / "run.jsonl").is_file():
            raise ValueError(f"{work} is not a run folder (no job.json or run.jsonl)")
        return work
    work = newest_run(root)
    if work is None:
        raise ValueError(f"no runs under {root}/: start one with `backcountry start` (or name one with --work-dir)")
    return work


def _jobs(work: Path) -> list[Path]:
    """The job folders of a run: the run, and the kernel search a loop search started."""
    return [work] + ([work / "kernels"] if (work / "kernels" / "run.jsonl").is_file() else [])


def events(work: Path) -> list[dict]:
    rows = []
    for job in _jobs(work):
        log = job / "run.jsonl"
        if log.is_file():
            for line in log.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # a line being written
    return sorted(rows, key=lambda row: row.get("wall", ""))


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def running(work: Path) -> bool:
    """Whether the run's process is alive: its pid file (a started run), else a
    process named with its folder."""
    try:
        os.kill(int((work / PID).read_text()), 0)
        return True
    except (OSError, ValueError):
        pass
    found = subprocess.run(["pgrep", "-f", f"--work-dir {work}"], capture_output=True, text=True).stdout
    return bool(found.strip())


def _budget(work: Path) -> int | None:
    job = _read(work / "job.json")
    for path in (work / "kernel_manifest.yaml", Path(job.get("manifest", "")), work / "manifest.yaml"):
        try:
            budget = (yaml.safe_load(path.read_text()) or {}).get("budget") or {}
        except (OSError, yaml.YAMLError):
            continue
        if budget.get("total") is not None:
            return int(budget["total"])
    return None


def summary(work: Path) -> dict:
    """How the run is going, as facts: state, stage, attempts, regions, what
    shipped, the loop's speedup, the last problem, the artifact."""
    top = _read(work / "report.json")
    kernels = _read(work / "kernels" / "report.json") if (work / "kernels" / "report.json").is_file() else top
    session = top.get("session", {})
    state = session.get("status") or ("running" if running(work) else "starting")
    alive = running(work)
    if state == "running" and not alive:
        state = "stopped"  # killed, crashed, or the Mac restarted: `backcountry resume`
    elif alive and state != "running":
        state = "running"  # a resume reloading the model: the report still says how the last process ended
    rows = events(work)
    failures = [r for r in rows if r.get("kind") == "job_failed"]
    restarts = [r for r in rows if r.get("kind") in ("resumed", "job")]
    problem = failures[-1] if failures and (not restarts or failures[-1]["wall"] >= restarts[-1]["wall"]) else None
    shipped = [r for r in rows if r.get("kind") == "shipped"]
    regions = [{"region": r.get("fingerprint", "")[:8], "attempts": r.get("hypotheses", 0)}
               for r in kernels.get("regions", []) if r.get("hypotheses")]
    opened = [r for r in rows if r.get("kind") in ("region_open", "region_resumed", "region_closed")]
    if opened and opened[-1]["kind"] != "region_closed" and \
            not any(r["region"] == opened[-1].get("fingerprint", "")[:8] for r in regions):
        regions.append({"region": opened[-1].get("fingerprint", "")[:8], "attempts": 0})
    loop = top.get("model_level", {}).get("final", {})
    artifact = Path(_read(work / "job.json").get("artifact", work / "artifact"))
    return {
        "run": str(work), "state": state, "running": alive, "stage": session.get("stage"),
        # each region's starter kernel is listed but never charged to the budget
        "attempts": sum(h.get("id") != "scaffold" for h in kernels.get("hypotheses", [])),
        "attempts_budget": _budget(work),
        "regions": regions, "current_region": regions[-1]["region"] if regions and state == "running" else None,
        "shipped": [r.get("kernel") for r in shipped],
        "loop_speedup": loop.get("speedup") if loop.get("ok") else None,
        "problem": None if problem is None else {"at": problem.get("wall"), "error": str(problem.get("reason"))[:300]},
        "last_event": None if not rows else {"kind": rows[-1].get("kind"), "at": rows[-1].get("wall")},
        "artifact": str(artifact) if artifact.is_dir() and any(artifact.iterdir()) else None,
        "outcome": session.get("outcome"),
    }


OUTCOMES = {   # how a run with no artifact ended, as `backcountry run` says it
    "no_improvement": "no confirmed improvement; no artifact produced (nothing beat the original, a normal result)",
    "unconfirmed": "no confirmed improvement; no artifact produced (kernels won in the search, but the final "
                   "whole-model check did not confirm the speedup)",
}


def render(facts: dict) -> str:
    """One screen."""
    budget = f" of {facts['attempts_budget']}" if facts["attempts_budget"] else ""
    stage = facts["stage"] if facts["running"] else None   # an ended run's last stage is history
    lines = [f"{facts['run']} · {facts['state']}" + (f" · {stage}" if stage else "")
             + f" · {facts['attempts']}{budget} attempts"]
    if facts["loop_speedup"]:
        lines.append(f"faster generation loop: {facts['loop_speedup']:.3f}x (confirmed)")
    lines.append(f"kernels shipped: {len(facts['shipped'])}" + (f" ({', '.join(facts['shipped'])})" if facts["shipped"] else ""))
    if facts["current_region"]:
        now = facts["regions"][-1]
        lines.append(f"now: region {now['region']}, attempt {now['attempts']}")
    if facts["problem"]:
        lines.append(f"last problem ({facts['problem']['at']}): {facts['problem']['error']}")
    if facts["state"] == "stopped":
        lines.append("the job is not running: `backcountry resume` continues it from its last save")
    if facts["artifact"]:
        lines.append(f"artifact: {facts['artifact']}")
    elif facts["outcome"]:
        lines.append(f"outcome: {OUTCOMES.get(facts['outcome'], facts['outcome'])}")
    return "\n".join(lines)


def wait(work: Path, poll_s: float = 10.0, sleep=time.sleep) -> dict:
    """Block until the next milestone (or the job's process ends), then return
    the summary with the milestone that ended the wait."""
    seen = sum(r.get("kind") in MILESTONES for r in events(work))
    while True:
        rows = [r for r in events(work) if r.get("kind") in MILESTONES]
        if len(rows) > seen:
            return {**summary(work), "milestone": rows[seen]}
        if not running(work):
            return {**summary(work), "milestone": {"kind": "process_ended"}}
        sleep(poll_s)


def default_judge() -> str:
    """The AI that writes kernels when none is named: a signed-in agent CLI
    if one is installed, else the Anthropic API with a key."""
    for name, command in (("claude-cli", "claude"), ("codex", "codex"), ("gemini", "gemini")):
        if shutil.which(command):
            return name
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    raise ValueError("no AI to write kernels: install Claude Code (`claude`), Codex or the Gemini CLI and sign in, "
                     "or set ANTHROPIC_API_KEY")


def start(run_argv: list[str], work: Path, ready_s: float = 120.0, command: str = "run") -> int:
    """`backcountry run` in the background: a new session (closing the terminal
    or the agent does not stop it) under caffeinate -i (the Mac does not idle-
    sleep mid-run), its output in <run>/console.log. Waits until the job is past
    its judge check and loading the model, or prints why it stopped."""
    work.mkdir(parents=True, exist_ok=True)
    console = (work / CONSOLE).open("ab")
    argv = [sys.executable, "-m", "backcountry.harness.cli", command, *run_argv]
    if shutil.which("caffeinate"):
        argv = ["caffeinate", "-i", *argv]
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                            start_new_session=True)
    (work / PID).write_text(str(proc.pid))
    deadline = time.monotonic() + ready_s
    while time.monotonic() < deadline:
        text = (work / CONSOLE).read_text(errors="replace")
        if proc.poll() is not None:
            print(text[-3000:], file=sys.stderr)
            print(f"the job stopped before it began (exit code {proc.returncode}); its output is in {work / CONSOLE}",
                  file=sys.stderr)
            return 1
        if "loading model" in text or "searching" in text or "resuming" in text:
            break
        time.sleep(1)
    print(f"started in the background: {work}\n"
          f"  backcountry status           how it is going (--wait returns at the next milestone)\n"
          f"  backcountry pause | finish   stop it resumably | wrap up and export\n"
          f"  output: {work / CONSOLE}")
    return 0
