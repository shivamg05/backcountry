"""backcountry init | check | start | run | status | pause | resume | finish | skip"""

import argparse
import json
from datetime import datetime
from contextlib import contextmanager
import fcntl
import os
import sys
import tempfile
from pathlib import Path


@contextmanager
def _run_lock():
    """Only one CLI job may measure the shared GPU; crashes release the lock."""
    path = Path(tempfile.gettempdir()) / f"backcountry-{os.getuid()}.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another backcountry job is running on this Mac; wait for it to finish") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _output_paths(work_dir, artifact):
    from .control import CONSOLE, PID
    work = Path(work_dir).resolve()
    # `backcountry start` leaves its console log and pid file there first
    if work.exists() and (not work.is_dir() or any(p.name not in (CONSOLE, PID) for p in work.iterdir())):
        raise ValueError(f"work directory is not empty: {work}; choose a fresh directory")
    out = Path(artifact).resolve() if artifact else work / "artifact"
    if out == work or out in work.parents:
        raise ValueError("--artifact must not be the work directory or one of its parents")
    for name in ("kernels", "boundaries", "checkpoints", "judge_io", "run.jsonl",
                 "session.jsonl", "report.json", "judge.jsonl", "candidates.log"):
        reserved = work / name
        if out == reserved or reserved in out.parents:
            raise ValueError(f"--artifact overlaps run files: {reserved}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"artifact destination already exists: {out}; choose a fresh destination")
    return out


def _drop_boundaries(work_dir) -> None:
    """A finished job's captured region inputs and outputs: only a search, or
    a resumed one, reads them, and they ran to tens of GB per job (108 of the
    115 GB under runs/). A paused, failed or killed job keeps them."""
    import shutil
    for job in (Path(work_dir), Path(work_dir) / "kernels"):
        shutil.rmtree(job / "boundaries", ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="backcountry")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="say what to make faster: writes backcountry.yaml from a script that uses "
                                       "the model (any library), a model id, a local model folder or a model file, "
                                       "asking only what the model cannot say")
    init.add_argument("target", help="a script that runs your model the way your app does (init watches it and asks "
                                     "which part to make faster); a Hugging Face id or local folder MLX-LM loads; or "
                                     "a model file with build() and examples()")
    for name, help_text in (("--optimize", "prompts, replies or chat (both)"), ("--prompt-tokens", "e.g. 32-2049"),
                            ("--optimize-at", "the prompt length the search optimizes at"),
                            ("--history-tokens", "conversation tokens before a reply"),
                            ("--reply-tokens", "tokens in a reply"), ("--keep", "same-output or same-math"),
                            ("--tolerance", "model files: exact or default"), ("--attempts", "kernel attempts"),
                            ("--part", "a watched script: which part (its number in the list)"),
                            ("--size", "a watched script: which recorded input size to optimize (its number)"),
                            ("--sizes", "a marked function: the sizes to optimize, by question number "
                                        "(\"1=32-4096; 2=512x512, 1024x1024\")"),
                            ("--request", "a marked function the app never calls: one request to make, "
                                          "by argument name (\"question=Write a haiku; history=chat.json\")")):
        init.add_argument(name, default=None, help=help_text)
    init.add_argument("--ask", action="store_true", help="run the script without its `if __name__ == "
                      "\"__main__\":` block (a server's loop) and ask for a request to its marked function")
    init.add_argument("script_args", nargs="*", metavar="-- SCRIPT_ARGS",
                      help="a watched script's own arguments, after -- (init app.py -- --image cat.png)")
    init.add_argument("--out", default="backcountry.yaml", help="where to write it (default: ./backcountry.yaml)")
    init.add_argument("--yes", action="store_true", help="ask nothing: flags, else the defaults")
    init.add_argument("--questions", action="store_true",
                      help="print the questions as JSON (for an agent to ask the user), write nothing")
    sub.add_parser("guide", help="how to run backcountry for someone, step by step (for agents, and people)")
    check = sub.add_parser("check", help="no attempts (minutes; up to an hour for requests that generate thousands of tokens): will it run, does it fit, how much room is there, "
                                         "is the AI signed in")
    _run_arguments(check)
    check.set_defaults(work_dir=None)
    run = sub.add_parser("run", help="run one optimization job here, in the foreground")
    _run_arguments(run)
    start = sub.add_parser("start", help="run one optimization job in the background (survives closing the "
                                         "terminal or the agent); then `backcountry status`")
    _run_arguments(start)
    status = sub.add_parser("status", help="how a run is going: the newest unless --work-dir names one")
    status.add_argument("--work-dir", default=None)
    status.add_argument("--json", action="store_true", help="the same facts as JSON, for agents and scripts")
    status.add_argument("--wait", action="store_true", help="wait for the next milestone (a kernel shipped, a "
                        "region opened or closed, a pause, resume or crash, the final result), then print")
    newest = "the newest run unless --work-dir names one"
    finish = sub.add_parser("finish", help=f"stop new search attempts and finish validation/export ({newest})")
    finish.add_argument("--work-dir", default=None)
    pause = sub.add_parser("pause", help=f"stop a running job at its next safe point, resumable ({newest})")
    pause.add_argument("--work-dir", default=None)
    skip = sub.add_parser("skip", help=f"close the region being searched and move on to the next one ({newest})")
    skip.add_argument("--work-dir", default=None)
    resume = sub.add_parser("resume", help=f"continue a paused or stopped job from its saved state ({newest})")
    resume.add_argument("--work-dir", default=None)
    for name in ("--judge", "--model", "--judge-effort", "--judge-cmd"):
        resume.add_argument(name, default=None, help="override the setting the job was started with")
    resume.add_argument("--background", action="store_true",
                        help="resume in the background, as `start` runs (then `backcountry status`)")
    args = parser.parse_args(argv)
    if args.command == "guide":
        from .guide import main as guide
        return guide()
    from . import control
    if args.command == "init":
        from .intent import IntentError, init
        given = {key: getattr(args, key) for key in ("optimize", "prompt_tokens", "optimize_at", "history_tokens",
                                                     "reply_tokens", "keep", "tolerance", "attempts", "part", "size",
                                                     "sizes", "script_args", "request", "ask")}
        try:
            return init(args.target, given, Path(args.out), interactive=sys.stdin.isatty() and not args.yes,
                        show_questions=args.questions)
        except IntentError as error:  # what to change, not how to call init: no usage lines
            print(f"backcountry init: {error}", file=sys.stderr)
            raise SystemExit(2) from None
    if args.command in ("finish", "pause", "skip", "resume", "status"):
        try:
            args.work_dir = str(control.resolve(args.work_dir))
        except ValueError as error:
            parser.error(str(error))
    if args.command == "status":
        facts = control.wait(Path(args.work_dir)) if args.wait else control.summary(Path(args.work_dir))
        print(json.dumps(facts, indent=1) if args.json else control.render(facts)
              + (f"\nmilestone: {facts['milestone']['kind']}" if args.wait else ""))
        return 0
    if args.command in ("finish", "pause", "skip"):
        work = Path(args.work_dir).resolve()
        if not (work / "run.jsonl").is_file():
            parser.error(f"no run.jsonl in {work}")
        if args.command == "finish":
            (work / "finish-search.request").write_text("operator requested final validation\n")
            print("Finish requested: the running job will finish its current work, then validate and export if confirmed.")
        elif args.command == "skip":
            from .resume import SKIP_REQUEST
            (work / SKIP_REQUEST).write_text("operator skipped the region\n")
            print("Skip requested: the region being searched closes after the attempt it is on, and the job "
                  "moves on to the next region if attempts are left. Between regions it does nothing.")
        else:
            from .resume import PAUSE_REQUEST
            (work / PAUSE_REQUEST).write_text("operator requested a pause\n")
            print("Pause requested: the job stops at its next safe point (after the attempt it is on) and "
                  f"exits as paused. Resume with: backcountry resume --work-dir {args.work_dir}")
        return 0
    if args.command == "resume":
        if args.background:
            rest = [a for a in (argv if argv is not None else sys.argv[1:])[1:] if a != "--background"]
            if "--work-dir" not in rest:
                rest += ["--work-dir", args.work_dir]
            return control.start(rest, Path(args.work_dir), command="resume")
        return _resume(args, parser)

    if args.judge is None and not args.judge_cmd:
        try:
            args.judge = control.default_judge()
        except ValueError as error:
            parser.error(str(error))
    if args.command == "start":
        _manifest_exists(args, parser)
        try:
            _output_paths(args.work_dir, args.artifact)
        except ValueError as error:
            parser.error(str(error))
        run_argv = [a for a in (argv if argv is not None else sys.argv[1:])][1:]
        if "--work-dir" not in run_argv:
            run_argv += ["--work-dir", args.work_dir]
        if "--judge" not in run_argv and not args.judge_cmd:
            run_argv += ["--judge", args.judge]
        return control.start(run_argv, Path(args.work_dir))
    if args.command == "check":
        if args.work_dir is None:
            args.work_dir = str(Path("runs") / ("check-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        args.target_shown = args.manifest
        _manifest_exists(args, parser)
        _prepare(args, parser)
        return _locked(parser, lambda: _check(args, parser))
    _manifest_exists(args, parser)
    _prepare(args, parser)
    return _locked(parser, lambda: _execute(args, parser))


def _locked(parser, work):
    """work() as the one job on this Mac; another running says so plainly
    (`backcountry status` shows it), never as a traceback."""
    try:
        lock = _run_lock()
        lock.__enter__()
    except RuntimeError as error:
        parser.error(f"{error} (backcountry status shows it)")
    try:
        return work()
    finally:
        lock.__exit__(None, None, None)


def _manifest_exists(args, parser) -> None:
    """A missing backcountry.yaml said plainly: a failed init leaves none, and
    check, start or run went on and died in a traceback."""
    if args.manifest.endswith((".yaml", ".yml")) and not Path(args.manifest).is_file():
        parser.error(f"{args.manifest} not found: write it with `backcountry init <your app>.py` first")


def _prepare(args, parser) -> None:
    """The run folder checked, and args.manifest the full manifest: a
    backcountry.yaml, model file or model id is written out into the folder."""
    from .front_door import is_model_file, is_model_id, write, write_from_examples
    from .intent import classify, resolve
    try:
        args.artifact = _output_paths(args.work_dir, args.artifact)
        is_intent = args.manifest.endswith((".yaml", ".yml")) and Path(args.manifest).is_file() \
            and classify(args.manifest).kind == "intent"
        if is_intent:
            if args.use != "chat" or args.attempts is not None or args.tolerances is not None:
                parser.error("--use, --attempts and --tolerances go in backcountry.yaml (optimize, attempts, tolerance)")
            args.manifest = str(resolve(Path(args.manifest), Path(args.work_dir)))
            print(f"wrote {args.manifest} from the backcountry.yaml: the full manifest this run uses")
        elif is_model_file(args.manifest):
            if args.use != "chat":
                parser.error("--use goes with a model id; a model file's examples() say how it is used")
            args.manifest = str(write_from_examples(Path(args.manifest), Path(args.work_dir), args.attempts,
                                                   tolerances=args.tolerances))
            print(f"wrote {args.manifest} from the model file's examples(): edit it and run it to change anything")
        elif args.tolerances is not None:
            parser.error("--tolerances goes with a model file; a manifest sets its own tolerances")
        elif is_model_id(args.manifest):
            # a model id, not a manifest: write model.py and manifest.yaml into the run folder
            args.manifest = str(write(args.manifest, args.use, Path(args.work_dir), attempts=args.attempts))
            print(f"wrote {args.manifest} and model.py: edit them and run the manifest to change anything")
        elif args.attempts is not None or args.use != "chat":
            parser.error("--use and --attempts go with a model id or model file; a manifest sets its own workloads and budget")
    except ValueError as error:  # IntentError is one
        parser.error(str(error))


def _run_arguments(run) -> None:
    """What `run` and `start` both take."""
    run.add_argument("manifest", help="a manifest file; a model id as MLX-LM loads it "
                                      "(mlx-community/Qwen3-8B-4bit): the job writes its own model file and manifest; "
                                      "or a model file with build() and examples() (real inputs): the job writes the manifest")
    run.add_argument("--use", default="chat", choices=("chat", "prompt", "reply"),
                     help="with a model id: what to make faster, reading prompts, writing replies, or both (chat)")
    run.add_argument("--attempts", type=int, default=None,
                     help="with a model id or model file: kernel attempts the search may make (default: a short run of 36)")
    run.add_argument("--tolerances", choices=("default",), default=None,
                     help="with a model file: also keep kernels whose outputs differ from the original by "
                          "rounding, within each output type's default limit (docs/manifest.md); "
                          "without it every kernel must be bit-identical")
    run.add_argument("--work-dir", default=str(Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")),
                     help="fresh run directory; defaults to runs/<timestamp>")
    run.add_argument("--artifact", default=None, help="fresh output directory; defaults to <work-dir>/artifact")
    run.add_argument("--model", default=None, help="judge model id; defaults to the selected provider's model")
    run.add_argument("--judge", default=None,
                     help="the AI that writes kernels; default: a signed-in Claude Code, Codex or Gemini CLI, "
                          "else the Anthropic API. api: Anthropic SDK (needs ANTHROPIC_API_KEY); "
                          "claude-cli, codex, gemini: that agent's CLI, headless, in an empty dir; "
                          "agent: answer <work-dir>/judge_io by hand (see docs/judge-protocol.md)")
    run.add_argument("--judge-effort", default=None,
                     help="thinking effort for the claude-cli judge (low, medium, high, xhigh, max); "
                          "default low, the level the CLI ran at before its 2026-09-17 update")
    run.add_argument("--judge-cmd", default=None,
                     help="run this exact command as the judge, any headless agent CLI; a "
                          "{system}/{prompt} token is filled in, else the prompt goes on stdin. "
                          "Overrides --judge.")


def _resume(args, parser) -> int:
    """Start the job saved in a work directory again, with the settings it
    was started with unless overridden."""
    from .resume import STATE, file_digest, code_identity, read_job, resumable
    work = Path(args.work_dir)
    job = read_job(work)
    if job is None or not resumable(work):
        parser.error(f"{work} has no saved job state to resume ({STATE}, job.json); only jobs started "
                     "with pause and resume support can be resumed")
    if file_digest(job["manifest"]) != job["manifest_sha256"]:
        parser.error(f"the manifest {job['manifest']} changed since the job started; restore it to resume")
    if code_identity() != job["code"]:
        print(f"warning: the tool's code changed since the job started ({job['code']} -> {code_identity()}); "
              "resuming with the current code", flush=True)
    judge = job["judge"]
    for key in ("judge", "model", "judge_effort", "judge_cmd"):
        if getattr(args, key) is not None:
            judge[key] = getattr(args, key)
    resumed = argparse.Namespace(manifest=job["manifest"], work_dir=str(work), artifact=Path(job["artifact"]),
                                 resume=True, **judge)
    with _run_lock():
        return _execute(resumed, parser)


def _make_judge(args, parser):
    """The AI that writes kernels, checked signed in before any model loads;
    returns (judge, label)."""
    from .judge.agent import CLI_PRESETS, AgentFileJudge, CliJudge

    if args.judge_cmd:
        import shlex
        judge = CliJudge(shlex.split(args.judge_cmd))
        label = args.judge_cmd
    elif args.judge == "api":
        from .judge.client import AnthropicJudge, DEFAULT_MODEL
        judge = AnthropicJudge(model=args.model or DEFAULT_MODEL)
        label = f"api ({args.model or DEFAULT_MODEL})"
    elif args.judge in CLI_PRESETS:
        from .judge.agent import DEFAULT_EFFORT, claude_argv
        preset = CLI_PRESETS[args.judge]
        effort = args.judge_effort or DEFAULT_EFFORT
        judge = (CliJudge(preset(args.model, effort), try_command=preset(args.model, effort, tries=True))
                 if preset is claude_argv else CliJudge(preset(args.model)))
        label = f"{args.judge} ({args.model or 'provider default'}"
        label += f", effort {effort})" if preset is claude_argv else ")"
    elif args.judge == "agent":
        mailbox = Path(args.work_dir) / "judge_io"
        judge = AgentFileJudge(mailbox)
        print(f"agent judge: answer requests in {mailbox}/ (protocol in docs/judge-protocol.md)")
        label = "agent"
    else:
        parser.error(f"--judge must be api, agent, or one of {sorted(CLI_PRESETS)}, "
                     f"or use --judge-cmd; got {args.judge!r}")

    if isinstance(judge, CliJudge):
        print("Checking judge connection and model access before loading the model...", flush=True)
        try:
            judge.check_available()
        except ValueError as error:
            parser.error(str(error))
        print("Judge readiness check passed.", flush=True)
    return judge, label


def _check(args, parser) -> int:
    """`backcountry check`: the AI answers, then the job's setup up to its room
    estimate (JobRunner.check), then one screen of what it found."""
    from .loop import CheckError, JobRunner
    judge, label = _make_judge(args, parser)
    runner = JobRunner(args.manifest, args.work_dir, judge_factory=lambda region: judge)
    try:
        report = runner.check()
    except CheckError as error:
        print(f"backcountry check: {error}")
        return 1
    print(check_text(report, label, args))
    return 0


def _marked_model(manifest) -> dict | None:
    """What init recorded for a marked function, when the manifest's model file
    is one it wrote; else None."""
    import yaml
    from backcountry_runtime.tune import marked_model_file
    try:
        model = yaml.safe_load(Path(manifest).read_text()).get("model")
        made_for = marked_model_file(Path(manifest).parent / model) if model else None
        # a model searched for an engine keeps its sizes' own names (prompt, reply)
        return None if made_for is None or made_for.get("LEVEL") == "model" else made_for
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def check_text(report, label, args) -> str:
    """What check found, in the user's terms."""
    fit = report.constants.get("memory_fit") or {}
    floors = report.coverage.get("step_floor") or {}
    pacing = report.constants.get("pacing") or {}
    shown = getattr(args, "target_shown", args.manifest)
    lines = [f"check: {shown} (full manifest: {args.manifest})", f"AI: {label}, signed in and answering"]
    heat = pacing.get("full_speed_ms")
    lines.append("heat: " + (f"readable; this Mac's full-speed probe {heat[0]:.1f}-{heat[1]:.1f} ms"
                             if heat else pacing.get("policy", "no readings: the job rests 2x its work")))
    if fit:
        lines.append(f"memory: GPU limit {fit['gpu_limit_gb']:.1f} GB; largest workload peak "
                     f"{max(fit['peaks_gb'].values()):.1f} GB")
        if fit.get("untimed"):
            lines.append(f"  too close to the limit to time (checked, timed after export in a fresh process): "
                         f"{', '.join(fit['untimed'])}")
    rooms = {name: floor["room"] for name, floor in floors.items() if floor.get("room") is not None}
    marked = _marked_model(args.manifest)
    for name, clocks in report.step_ms.items():
        before = clocks.get("before")
        room = rooms.get(name)
        if marked is not None:  # the user's own function: name the call, not the workload
            from backcountry_runtime.tune import call_label
            name = call_label(marked, name)
        lines.append((f"  {name}: {before:.2f} ms {'a call' if marked else 'a step'}" if before else f"  {name}")
                     + (f", up to {room:.0%} headroom" if room is not None else ""))
    if rooms:
        best = max(rooms.values())
        lines.append(("may not be worth a run: " if best < 0.05 else "") + f"up to {best:.0%} headroom "
                     "(how much faster this Mac could run it at best; an estimate, not a promise)")
    found = report.coverage.get("regions_found", 0)
    lines.append(f"regions a kernel could replace: {found}")
    if not found:
        # Whisper through mlx-whisper: 69,580 regions, all unsupported; check went
        # straight on to "next: start" as if a kernel search had something to do
        unknown = (report.coverage.get("discovery") or {}).get("unknown_sources") or []
        lines.append("  none: the kernel search has nothing to work on here"
                     + (f" (the model reads arrays the recorder could not place: {', '.join(unknown[:3])})"
                        if unknown else "")
                     + "; only a faster version of the function itself (keep: same-output) can make it faster")
    lines.append(f"next: backcountry start {shown}")
    return "\n".join(lines)


def _execute(args, parser):

    from .loop import JobRunner
    judge, label = _make_judge(args, parser)
    judge.transcript = Path(args.work_dir) / "judge.jsonl"
    print(f"judge: {label}")
    print(f"run log: {Path(args.work_dir) / 'run.jsonl'} (one JSON line per event; tail it)")
    print(f"candidates: {Path(args.work_dir) / 'candidates.log'} (one line per attempt)")
    resuming = getattr(args, "resume", False)
    runner = JobRunner(
        args.manifest,
        args.work_dir,
        judge_factory=lambda region: judge,
        **({"resume": True} if resuming else {}),
    )
    if not resuming:
        from .resume import write_job
        write_job(args.work_dir, args.manifest, {"judge": args.judge, "model": args.model,
                                                 "judge_effort": args.judge_effort, "judge_cmd": args.judge_cmd},
                  args.artifact)
    try:
        report = runner.run()
        if report.session.get("status") == "paused":
            print(f"job paused: {report.session.get('reason')}\n"
                  f"resume with: backcountry resume --work-dir {args.work_dir}")
            return 0
        artifact = None
        if (report.accepted and runner.final_ok) or report.model_level.get("final", {}).get("ok"):
            artifact = runner.emit_artifact(args.artifact)
        else:
            report.session.update(status="complete", artifact=None,
                                  outcome="unconfirmed" if report.accepted else "no_improvement")
            report.write(Path(args.work_dir) / "report.json")
        _drop_boundaries(args.work_dir)
    except Exception as error:
        print(f"job failed: see {args.work_dir}/report.json and the error below", file=sys.stderr)
        _auto_resume(error, args.work_dir)  # returns only when the job should stop here
        raise
    _print_result(report, artifact, _marked_model(args.manifest))
    return 0


AUTO_RESUMES = 3          # fresh-process resumes after GPU failures, per job start
AUTO_RESUME_WAIT_S = 120  # the GPU's rest before each
_GPU_FAILURES = ("Command buffer execution failed", "GPU Timeout", "GPU recovery check failed")


def _auto_resume(error, work_dir) -> None:
    """A job the GPU failed under (a macOS GPU timeout while the Mac was busy,
    a failed GPU check) resumes from its last save in a fresh process, whose
    GPU state is new, after a rest: up to AUTO_RESUMES times. The failure is
    charged to the kernel attempt it came in: the resumed job runs that kernel
    again, and drops it at the second. Any other error, a job with no save,
    or the last try, returns and the error stands."""
    from .resume import note_gpu_failure, resumable
    work = Path(work_dir)
    done = int(os.environ.get("AUTOTUNE_AUTO_RESUMES", "0"))
    if not any(text in str(error) for text in _GPU_FAILURES):
        return
    note_gpu_failure(work, error)
    if done >= AUTO_RESUMES or not resumable(work):
        return
    print(f"the GPU failed under the job; resuming it from its last save in a fresh process in "
          f"{AUTO_RESUME_WAIT_S} s ({done + 1} of {AUTO_RESUMES})", file=sys.stderr, flush=True)
    import time
    time.sleep(AUTO_RESUME_WAIT_S)
    os.environ["AUTOTUNE_AUTO_RESUMES"] = str(done + 1)
    os.execv(sys.executable, [sys.executable, "-m", "backcountry.harness.cli", "resume", "--work-dir", str(work)])


def exactness(report) -> str:
    """How close the shipped kernels' outputs are to the original's: each
    shipped region's last accepted kernel, bit-identical or within tolerance."""
    shipped = {r.get("fingerprint") for r in report.regions if r.get("s")}
    last = {a.get("fingerprint"): a.get("assoc_tag") for a in report.accepted if a.get("fingerprint") in shipped}
    exact = sum(tag == "preserving" for tag in last.values())
    near = len(last) - exact
    return ", ".join(([f"{exact} bit-identical"] if exact else [])
                     + ([f"{near} within tolerance: outputs differ slightly from the original's"] if near else []))


def search_text(report) -> str:
    """How much searching the result rests on, and how long the run took:
    "search: 6 loop attempts, 36 kernel attempts in 3 regions, 3.0 h, stopped
    by backcountry finish; checks and export after it: 1.2 h"."""
    opened = [r["hypotheses"] for r in report.regions if r.get("hypotheses")]
    parts = ([f"{len(report.model_level.get('history', []))} loop attempts"] if report.model_level else []) + [
        f"{sum(opened)} kernel attempts in {len(opened)} regions"]
    total_s = report.session.get("job_active_s") or report.session.get("active_s")
    search_s = report.session.get("search_s") or total_s
    if search_s:
        parts.append(f"{search_s / 3600:.1f} h")
    why = "backcountry finish" if report.session.get("finish_requested") else None
    after = (f"; checks and export after it: {(total_s - search_s) / 3600:.1f} h"
             if total_s and search_s and total_s - search_s >= 360 else "")
    return "search: " + ", ".join(parts) + (f", stopped by {why}" if why else "") + after


def _print_result(report, artifact, marked: dict | None = None):
    """marked: a marked function's model file (its calls are named as the
    user's calls, in place of workloads)."""
    from backcountry_runtime.sequence import throughput_text
    from backcountry_runtime.tune import call_label

    loop = report.model_level.get("final", {})
    if artifact:
        shipped = sum(bool(r.get("s")) for r in report.regions)
        if report.regions:
            print(f"job complete: verified artifact with {shipped}/{len(report.regions)} regions optimized"
                  + (f" ({exact})" if (exact := exactness(report)) else ""))
        else:
            print("job complete: verified artifact")
    else:
        print("job complete: no confirmed improvement; no artifact produced")
        if report.accepted:
            print("  Search accepted candidates, but final validation did not confirm the improvement.")
    print("  " + search_text(report))
    if loop.get("ok") and report.model_level.get("not_shipped"):
        print(f"  inference loop: {loop['speedup']:.3f}x in the job, NOT shipped: it did not hold in a fresh "
              f"process ({report.model_level['not_shipped'][:300]})")
    elif loop.get("ok"):
        print(f"  inference loop: {loop['speedup']:.3f}x against the job's default loop on "
              f"{loop['inputs']} held-out inputs ({loop['best_ms']:.1f} ms vs {loop['default_ms']:.1f} ms), "
              f"outputs identical at {len(loop.get('cases', []))} conversation and reply lengths; run.py in the artifact")
    measurement = report.final.get("measurement", report.constants.get("measurement", {}))
    objective = "a whole call" if marked else "forward pass"
    if measurement.get("kind") == "library_generation":
        objective = f"library generation ({measurement['generated_tokens']} generated tokens)"
    for w, clocks in report.step_ms.items():
        # the untouched model is re-measured beside the patched one at the end;
        # the job-start clock is a different window and never enters this line
        name = call_label(marked, w) if marked else w
        if clocks.get("speedup"):
            result = (f"{clocks['speedup']:.3f}x confirmed speedup" if clocks.get("win_confirmed")
                      else "no confirmed speedup")
            print(f"  {name}: {objective}: patched {clocks['after']:.3f} ms vs baseline "
                  f"{clocks['baseline_at_end']:.3f} ms ({report.baseline.get('choice')}), "
                  f"measured together: {result} (pair agreement {clocks['stability']:.2f})")
        sequence = report.final.get("sequences", {}).get(w)
        if sequence and clocks.get("sequence_speedup"):
            result = (f"{clocks['sequence_speedup']:.3f}x confirmed speedup"
                      if clocks.get("sequence_win_confirmed") else "no confirmed speedup")
            kind = sequence.get("workload_kind", "repeated_forward")
            description = {
                "repeated_forward": "repeated calls" if marked else "repeated forward passes",
                "advancing_cache_fixed_tokens": "steps with advancing cache and fixed input tokens",
                "library_generation": "generated tokens via library inference",
            }.get(kind, kind)
            print(f"  {name}: {sequence['steps']} {description}: patched "
                  f"{sequence['candidate_sequence_ms']:.1f} ms vs untouched "
                  f"{sequence['baseline_sequence_ms']:.1f} ms, run whole and alternated: {result}")
            if rates := throughput_text(sequence):
                print(f"    {rates}")
    if headline := report.final.get("headline"):
        dim, worst = headline["dim"], headline["worst"]
        fresh = sum(row["fresh"] for row in headline["curve"])
        if headline.get("geomean_fresh"):
            unconfirmed = "" if report.final.get("passed") else " (unconfirmed: the final check did not confirm a win)"
            print(f"  across {fresh} sizes the search never saw{unconfirmed}: {headline['geomean_fresh']:.3f}x "
                  f"(geometric mean); worst {worst['speedup']:.3f}x at {dim}={worst[dim]} "
                  f"of the {len(headline['curve'])} timed")
    if artifact:
        print(f"artifact: {artifact}")


if __name__ == "__main__":
    sys.exit(main())
