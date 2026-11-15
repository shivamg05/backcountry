"""Pause and resume: a job's state saved at safe points, restored in a new process.

A job saves its state after setup, before every judge call and between
regions: points where no GPU work is in flight and every verdict so far is
recorded. `backcountry pause` asks the job to stop at the next such point;
three unreachable judge calls in a row stop it there too. `backcountry resume`
loads the model and records it again (deterministic, seconds), checks the
recording still yields every region the saved state refers to, restores the
state, reinstalls the saved wrappers, checks the model's outputs against the
untouched model, and continues from the saved point, the open region's queue
and history included. Nothing measured during setup is measured again.

The live model, the recorder, logs and the saved boundary files are rebuilt
or reopened; everything in SAVED is restored as saved.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import uuid
from pathlib import Path

STATE = "resume_state.pkl"
PAUSE_REQUEST = "pause.request"
SKIP_REQUEST = "skip-region.request"  # stands until the next region opens
JOB = "job.json"
LOOP_SEARCH = "loop_search.json"  # the inference-loop search's position, after every attempt
ATTEMPT = "attempt.pkl"  # the kernel attempt running: its judge reply, its step, the GPU failures it ran into
DEATHS_TO_DROP = 2  # GPU failures under one kernel attempt before it is recorded failed
_PROCESS = uuid.uuid4().hex  # a GPU failure counts against an attempt only in the process running it

# runner attributes a resumed job restores as saved
SAVED = ("report", "baseline", "step_ms", "step_chain", "peaks", "total_hypotheses", "lessons",
         "scaffold_overrides", "cuts", "certified_scopes", "replay_scopes", "emitted", "model_ratio",
         "model_ratios", "pending_regions", "selection_wave", "shipped_tags", "shipped_regions",
         "sweep_spans", "gpu_busy_at_start", "baseline_wrappers", "pending", "confirmed", "live_scopes",
         "final_ok", "_holdout_draws")


class Paused(Exception):
    """The job stopped at a safe point and can be resumed; the message says why."""


def save(runner, where: str) -> None:
    """Write the runner's state and search position, atomically."""
    state = {name: getattr(runner, name) for name in SAVED if hasattr(runner, name)}
    # the modules a wrapper replaced are live objects: a resumed job finds them again
    state["installed"] = {path: (variants, kernels) for path, (_original, variants, kernels)
                          in getattr(runner, "installed", {}).items()}
    state["compiled_baseline"] = getattr(runner, "_compiled_baseline", None) is not None
    state["search"] = getattr(runner, "_search_state", None)
    state["where"] = where
    _dump(Path(runner.work_dir) / STATE, state)


def _dump(path: Path, state) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load(work_dir, name: str = STATE) -> dict | None:
    path = Path(work_dir) / name
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def start_attempt(work_dir, region: str, kernel: str, response, deaths: list) -> None:
    """Before a kernel attempt's GPU work: the judge's reply that made it and
    the GPU failures it already ran into, so a job the GPU fails under can
    run the same kernel again after it resumes, without asking the judge."""
    _dump(Path(work_dir) / ATTEMPT, {"region": region, "kernel": kernel, "response": response,
                                     "deaths": deaths, "step": "the region checks", "process": _PROCESS})


def attempt_step(work_dir, step: str) -> None:
    """The step this process's running attempt is on, for its failure record."""
    attempt = load(work_dir, ATTEMPT)
    if attempt is not None and attempt["process"] == _PROCESS:
        _dump(Path(work_dir) / ATTEMPT, {**attempt, "step": step})


def end_attempt(work_dir) -> None:
    (Path(work_dir) / ATTEMPT).unlink(missing_ok=True)


def note_gpu_failure(work_dir, error) -> None:
    """A GPU failure under the job: charged to the kernel attempt this process
    was running, in the job's directory or its kernel search's (kernels/).
    One that came while the job was resuming (re-recording the model, say)
    finds no attempt of this process and counts against none."""
    for directory in (Path(work_dir), Path(work_dir) / "kernels"):
        attempt = load(directory, ATTEMPT)
        if attempt is not None and attempt["process"] == _PROCESS:
            deaths = attempt["deaths"] + [{"step": attempt["step"], "error": str(error)[:300]}]
            _dump(directory / ATTEMPT, {**attempt, "deaths": deaths, "process": None})


def failed_attempt(work_dir) -> dict | None:
    """The attempt a GPU failure stopped the job in, for the resumed job; an
    attempt stopped any other way (a kill) is asked of the judge again."""
    attempt = load(work_dir, ATTEMPT)
    return attempt if attempt is not None and attempt["deaths"] else None


def finished_loop(work_dir) -> dict | None:
    """The loop search's result an identical: output job saved in its report
    once that search passed its final (run.py and the outputs the bundle's
    check compares beside it), or None: a job stopped after it resumes from
    there instead of searching the loop again."""
    work = Path(work_dir)
    try:
        level = json.loads((work / "report.json").read_text()).get("model_level") or {}
    except (OSError, ValueError):
        return None
    final = level.get("loop_alone") or level.get("final") or {}
    if not (level.get("source") and final.get("ok") and (work / "run.py").is_file()
            and (work / "model_level_check.safetensors").is_file()):
        return None
    level = {k: v for k, v in level.items() if k not in ("loop_alone", "kernels_dropped", "final_with_kernels")}
    return {**level, "final": final}


def save_loop_search(work_dir, position: dict) -> None:
    """The loop search's position after an attempt (its history, the accepted
    attempts and the best loop's source), atomically: a job that dies mid-search
    (power lost) resumes at the next attempt instead of starting over."""
    path = Path(work_dir) / LOOP_SEARCH
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(position, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    os.replace(temporary, path)


def loop_search(work_dir) -> dict | None:
    """The loop search's saved position, or None."""
    try:
        return json.loads((Path(work_dir) / LOOP_SEARCH).read_text())
    except (OSError, ValueError):
        return None


def resumable(work_dir) -> bool:
    """Whether a work directory saved anything to resume from: a job's state,
    the kernel search's under it (an identical: output job's kernels/), a
    finished loop search or one stopped part way."""
    work = Path(work_dir)
    return ((work / STATE).is_file() or (work / "kernels" / STATE).is_file()
            or finished_loop(work) is not None or loop_search(work) is not None)


def pause_requested(work_dir) -> bool:
    """A pending pause request, consumed so the resumed job does not stop again."""
    request = Path(work_dir) / PAUSE_REQUEST
    if not request.is_file():
        return False
    request.unlink(missing_ok=True)
    return True


def code_identity() -> str:
    """The checked-out commit plus a digest of uncommitted changes, when git can say."""
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}+{hashlib.sha256(diff).hexdigest()[:12]}" if diff else head


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_job(work_dir, manifest, judge: dict, artifact) -> None:
    """What `backcountry resume` needs to start the same job again."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    digest = file_digest(manifest) if Path(manifest).is_file() else None
    (Path(work_dir) / JOB).write_text(json.dumps({
        "manifest": str(Path(manifest).resolve()), "manifest_sha256": digest,
        "code": code_identity(), "judge": judge, "artifact": str(artifact)}, indent=1) + "\n")


def read_job(work_dir) -> dict | None:
    path = Path(work_dir) / JOB
    return json.loads(path.read_text()) if path.is_file() else None
