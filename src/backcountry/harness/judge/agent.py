"""Judge transports that need no API key.

CliJudge shells out to any headless agent CLI (Claude Code, Codex, Gemini,
...), one fresh process per ask in an empty working directory so nothing about
the repo leaks in. Every agent is driven the same way: a {system}/{prompt}
token in the command is filled in, otherwise the system prompt is folded onto
the front of the messages and the whole prompt goes on stdin. The reply is the
process's stdout. Presets build the argv for the common agents; a custom
command (--judge-cmd) drives any other.

AgentFileJudge turns judge calls into a file handshake so a live agent (a
Claude Code session opened on this repo) can be the judge: the harness
writes NNNN.request.json into a mailbox directory and blocks until the
agent writes NNNN.response.json holding one JSON object that matches the
schema carried inside the request. docs/judge-protocol.md is the operator guide.

Both feed the same exchange loop as the API client, so malformed replies
get the one re-ask and then JudgeBabble, identically across transports.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .client import DEFAULT_MODEL, JsonJudge

# replies at high effort on a large region have taken up to 27 minutes; at 10
# minutes FLUX's first matmul region timed out three times and paused the job
CLI_TIMEOUT_S = 1800.0
MAILBOX_TIMEOUT_S = 1800.0
GRACE_S = 10.0
_POLL_S = 0.5


SYSTEM_TOKEN = "{system}"
PROMPT_TOKEN = "{prompt}"
RESPONSE_TOKEN = "{response}"


def _document(messages: list[dict]) -> str:
    """The message list as one text block; the re-ask needs the judge to see
    its rejected reply and the rejection reason in order."""
    parts = []
    for m in messages:
        tag = "your previous response" if m["role"] == "assistant" else "user"
        parts.append(f"[{tag}]\n{m['content']}")
    return "\n\n".join(parts)


JUDGE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "low"


def claude_argv(model: str | None = None, effort: str = DEFAULT_EFFORT, tries: bool = False) -> list[str]:
    """Claude Code in print mode with tools, MCP, settings, and slash commands
    off and no session persisted, so the judge sees only what the harness
    sends. System via --system-prompt; the messages come on stdin. The
    thinking effort is pinned: left to the CLI, its default moved with the
    2026-09-17 update and one reply went from 18 s (low) to 98 s. tries: its
    one tool is the shell, allowed to run ./try alone (it cannot read outside
    its empty working directory)."""
    if effort not in JUDGE_EFFORTS:
        raise ValueError(f"judge effort must be one of {JUDGE_EFFORTS}, got {effort!r}")
    tools = ["--tools", "Bash", "--allowedTools", "Bash(./try)"] if tries else ["--tools", ""]
    return ["claude", "-p", "--output-format", "text", *tools,
            "--strict-mcp-config", "--setting-sources", "",
            "--disable-slash-commands", "--no-session-persistence",
            "--model", model or DEFAULT_MODEL, "--effort", effort, "--system-prompt", SYSTEM_TOKEN]


def codex_argv(model: str | None = None) -> list[str]:
    """Codex exec reads the briefing from stdin and saves only its final reply.

    A fresh judge directory is deliberately not a Git checkout. Disable local
    instructions and tools that could read outside the metadata briefing.
    Omitting a model lets Codex resolve its own default, never an Anthropic id.
    """
    argv = ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
            "--ignore-user-config", "--sandbox", "read-only", "--color", "never",
            "-c", "project_doc_max_bytes=0", "-c", 'web_search="disabled"']
    for feature in ("shell_tool", "plugins", "memories", "apps", "multi_agent",
                    "browser_use", "computer_use", "image_generation", "view_image",
                    "hooks", "workspace_dependencies"):
        argv.extend(["--disable", feature])
    if model:
        argv.extend(["--model", model])
    return argv + ["--output-last-message", RESPONSE_TOKEN, "-"]


def gemini_argv(model: str | None = None) -> list[str]:
    """Google Gemini CLI, non-interactive. System folded into the prompt. Flags
    vary by version, so --judge-cmd overrides."""
    return ["gemini"] + (["--model", model] if model else []) + ["--prompt", PROMPT_TOKEN]


CLI_PRESETS = {"claude-cli": claude_argv, "claude": claude_argv,
               "codex": codex_argv, "gemini": gemini_argv}


class CliJudge(JsonJudge):
    """One CLI process per ask, agent-agnostic. The command may carry a
    {system} token (the harness system prompt is substituted there) and a
    {prompt} token (the messages are substituted there); with neither, the
    system is folded onto the front of the messages and the whole prompt goes
    on stdin. A {response} token names a fresh output file whose content is
    the reply instead of stdout. The working directory starts empty; preset
    flags control whether the agent can load other local context."""

    def __init__(self, command: list[str], timeout_s: float = CLI_TIMEOUT_S,
                 try_command: list[str] | None = None):
        """try_command: the same agent with ./try allowed, used while a
        try_handler is set (a kernel search's region)."""
        if not command:
            raise ValueError("a CLI judge needs a command to run")
        self._command = list(command)
        self._try_command = list(try_command) if try_command else None
        self._timeout_s = timeout_s
        self._cwd: str | None = None
        # set by the kernel search while it asks for a region's next kernel:
        # payload dict -> result dict, run by the job while the agent works
        self.try_handler = None

    def check_available(self) -> None:
        """Exercise the configured model and transport before any GPU work."""
        if shutil.which(self._command[0]) is None:
            # claude installs to ~/.local/bin, which a shell over ssh often leaves off PATH
            local = Path.home() / ".local" / "bin" / self._command[0]
            hint = (f"it is at {local}, which is not on PATH: add {local.parent} to PATH"
                    if os.sep not in self._command[0] and local.exists() else "install it or choose --judge-cmd")
            raise ValueError(f"judge executable not found: {self._command[0]}; {hint}")
        try:
            reply = self._ask(
                'This is a connection check. Do not use tools. Reply only with {"ready":true}.',
                [{"role": "user", "content": 'Return {"ready":true}.'}],
                timeout_s=min(self._timeout_s, 60.0),
            )
            try:
                ready = json.loads(reply)
            except ValueError:
                raise ValueError(f"expected readiness JSON; received {reply[-1000:]!r}") from None
            if ready != {"ready": True}:
                raise ValueError(f"expected the readiness JSON response; received {reply[-1000:]!r}")
        except (RuntimeError, OSError, ValueError) as error:
            raise ValueError(f"Judge readiness check failed before model loading: {error}") from error

    def _ask(self, system: str, messages: list[dict], *, timeout_s: float | None = None) -> str:
        if self._cwd is None:
            self._cwd = tempfile.mkdtemp(prefix="judge_cli_")
            atexit.register(shutil.rmtree, self._cwd, ignore_errors=True)
        document = _document(messages)
        prompt = document if SYSTEM_TOKEN in self._command else f"{system}\n\n{document}"
        # A late reply from an earlier process must never answer the next ask.
        with tempfile.TemporaryDirectory(prefix="request_", dir=self._cwd) as request_dir:
            return self._run_request(system, prompt, request_dir, timeout_s=timeout_s)

    def _run_request(self, system: str, prompt: str, request_dir: str,
                     *, timeout_s: float | None = None) -> str:
        response_path = Path(request_dir) / "response.json"
        tries = self.try_handler is not None and self._try_command is not None
        command = self._try_command if tries else self._command
        if tries:
            _write_try_script(Path(request_dir))
            if SYSTEM_TOKEN in command:
                system = system + "\n\n" + TRY_GUIDE
            else:
                prompt = TRY_GUIDE + "\n\n" + prompt
        argv = [system if t == SYSTEM_TOKEN else prompt if t == PROMPT_TOKEN
                else str(response_path) if t == RESPONSE_TOKEN else t
                for t in command]
        # empty (not inherited) stdin when the prompt rides argv, so the child
        # never blocks reading a stdin nobody is writing
        stdin = "" if PROMPT_TOKEN in self._command else prompt
        agent = self._command[0]
        timeout_s = self._timeout_s if timeout_s is None else timeout_s
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, cwd=request_dir,
                              start_new_session=True) as proc:
            try:
                if tries:
                    stdout, stderr = self._communicate_serving_tries(proc, stdin, timeout_s, Path(request_dir))
                else:
                    stdout, stderr = proc.communicate(input=stdin, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{agent} judge gave no reply within {timeout_s:.0f}s")
            finally:
                # Agent CLIs may launch children. Clean up the entire group on
                # success, timeout, failure, or interruption before the next ask.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if proc.returncode != 0:
            detail = "\n".join(f"{name}: {value.strip()[-1000:]}"
                               for name, value in (("stderr", stderr), ("stdout", stdout))
                               if value and value.strip()) or "no diagnostic output"
            if Path(agent).name == "claude" and any(
                word in detail.lower() for word in ("oauth", "login", "log in", "sign in", "authentication", "unauthorized")
            ):
                detail += "\nSign in in your terminal with `claude auth login`, then run the same command again."
            raise RuntimeError(f"{agent} judge exited {proc.returncode}: {detail}")
        if RESPONSE_TOKEN in self._command:
            if not response_path.exists():
                raise RuntimeError(f"{agent} judge exited without writing its final response")
            return response_path.read_text().strip()
        return stdout.strip()


    def _communicate_serving_tries(self, proc, stdin: str, timeout_s: float, request_dir: Path):
        """proc.communicate, while answering the agent's ./try requests: each
        is a file tries/<n>.req the script waits on until tries/<n>.res
        appears. Time the job spends on a try is not the agent's, so it
        extends the reply's time limit."""
        done: dict = {}

        def talk():
            try:
                done["io"] = proc.communicate(input=stdin)
            except BaseException as error:  # raised again on the calling thread
                done["error"] = error

        thread = threading.Thread(target=talk, daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout_s
        served = 0
        while thread.is_alive():
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(proc.args, timeout_s)
            for request in sorted((request_dir / "tries").glob("*.req")):
                started = time.monotonic()
                if served >= MAX_TRIES:
                    answer = {"error": f"the {MAX_TRIES} tries for this reply are used: submit your kernel"}
                else:
                    served += 1
                    try:
                        payload = json.loads(request.read_text())
                    except (OSError, ValueError) as error:
                        answer = {"error": f"./try reads one JSON object on stdin: {error}"}
                    else:
                        answer = self.try_handler(payload)
                result = request.with_suffix(".res")
                result.with_suffix(".tmp").write_text(json.dumps(answer, default=str))
                result.with_suffix(".tmp").rename(result)
                request.unlink(missing_ok=True)
                deadline += time.monotonic() - started
            thread.join(0.25)
        if "error" in done:
            raise done["error"]
        return done["io"]


# A judge's tries per reply: each costs the job seconds to a minute of GPU.
MAX_TRIES = 12
TRY_GUIDE = f"""TESTING BEFORE YOU SUBMIT: you may test kernels first with `./try`, the one
command you can run. Give it one JSON object on stdin: {{"kernel": <exactly what
you would put in your reply's "kernel" field>, "item_id": <optional queued item,
for its correctness rule>}}, for example

./try <<'END'
{{"kernel": {{"parent_kernel_id": "head", "source": "...", "grid": [...], ...}}}}
END

The running job compiles it, checks its outputs against the region's recorded
ones (bit for bit unless the item allows rounding changes) and times it against
the library at every recorded size, in seconds to a minute, and prints JSON:
outcome (failed, correct_slower or faster: faster at half the sizes or more and
slower at none, as the full check requires), the failed gate and why, and per
size kernel_ms, library_ms and win, loss or tie. Test every kernel before you submit it: fix it
until it compiles and is correct, then try variants of what limits it (tile
sizes, how many outputs a thread owns, vector widths, unrolling) and submit the
fastest you measured; an untested kernel usually wastes an attempt. Up to
{MAX_TRIES} tries per reply; they do not use the search budget, and the kernel
you submit still gets the full checks (every recorded size, shader validation,
the whole model). Your earlier replies' tries are in region_state.tries; the
fastest correct one is kept in kernels, so you can name it as parent_kernel_id."""

_TRY_SCRIPT = """#!/bin/sh
# One candidate kernel (JSON on stdin) to the running job; prints its quick check.
mkdir -p tries
f=tries/$$
cat > "$f.tmp" && mv "$f.tmp" "$f.req"
i=0
while [ ! -f "$f.res" ]; do
  sleep 0.5
  i=$((i + 1))
  if [ "$i" -gt 3600 ]; then echo '{"error": "the job did not answer within 30 minutes"}'; exit 1; fi
done
cat "$f.res"
rm -f "$f.res"
"""


def _write_try_script(directory: Path) -> None:
    path = directory / "try"
    path.write_text(_TRY_SCRIPT)
    path.chmod(0o755)


class AgentFileJudge(JsonJudge):
    """Blocking file mailbox: a live agent answers each request by hand."""

    def __init__(self, mailbox: str | Path, timeout_s: float = MAILBOX_TIMEOUT_S,
                 grace_s: float = GRACE_S):
        self._dir = Path(mailbox)
        self._dir.mkdir(parents=True, exist_ok=True)
        leftover = sorted(p.name for p in self._dir.glob("*.json"))
        if leftover:
            # a reused mailbox would answer new requests with last run's mail
            raise RuntimeError(
                f"judge mailbox {self._dir} holds {len(leftover)} file(s) from a "
                f"previous run ({leftover[0]} ...); move or delete them first")
        self._seq = 0
        self._timeout_s = timeout_s
        self._grace_s = grace_s

    def _ask(self, system: str, messages: list[dict]) -> str:
        self._seq += 1
        name = f"{self._seq:04d}"
        request = {"seq": self._seq, "respond_in": f"{name}.response.json",
                   "system": system, "messages": messages}
        tmp = self._dir / f".{name}.request.tmp"
        tmp.write_text(json.dumps(request, indent=2))
        tmp.rename(self._dir / f"{name}.request.json")
        return self._await_response(self._dir / f"{name}.response.json")

    def _await_response(self, path: Path) -> str:
        """Accept the file once it parses as one JSON document; a non-JSON
        file that stops changing for the grace window is handed over as-is so
        the re-ask can say why. Reads tolerate the agent still writing or
        deleting to rewrite."""
        deadline = time.monotonic() + self._timeout_s
        last_text = None
        stable_since = 0.0
        while time.monotonic() <= deadline:
            try:
                text = path.read_text()
            except OSError:
                text = ""
            if text:
                try:
                    json.loads(text)
                    return text
                except (json.JSONDecodeError, RecursionError):
                    now = time.monotonic()
                    if text != last_text:
                        last_text, stable_since = text, now
                    elif now - stable_since >= self._grace_s:
                        return text
            time.sleep(_POLL_S)
        raise TimeoutError(
            f"no complete {path.name} after {self._timeout_s:.0f}s; "
            f"is a judge agent watching {self._dir}? (see docs/judge-protocol.md)")
