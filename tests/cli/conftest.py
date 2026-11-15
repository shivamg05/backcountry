"""The CLI tests run the real `backcountry` command, as a person or an agent
does, and check what it prints: in a terminal (answers typed at each prompt)
or with flags. Run them alone with `pytest tests/cli`, or one command's with
`pytest tests/cli/test_init.py`."""

import os
import pty
import select
import subprocess
import sys
from pathlib import Path

import pytest

BACKCOUNTRY = str(Path(sys.executable).parent / "backcountry")
FAKE_JUDGE = f"{sys.executable} {Path(__file__).parent / 'fake_judge.py'}"


class Result:
    def __init__(self, code: int, out: str):
        self.code, self.out = code, out

    def __repr__(self):
        return f"exit {self.code}:\n{self.out}"


def backcountry(*args, cwd, answers: list[str] | None = None, timeout: float = 900) -> Result:
    """Run `backcountry args` in cwd. With answers: in a terminal, typing the
    next answer (Enter for "") at each prompt; the output then shows them, as
    the person saw it. Without: plain pipes, as an agent runs it."""
    command = [BACKCOUNTRY, *map(str, args)]
    if answers is None:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        return Result(done.returncode, done.stdout + done.stderr)
    leader, follower = pty.openpty()
    process = subprocess.Popen(command, cwd=cwd, stdin=follower, stdout=follower, stderr=follower,
                               start_new_session=True)
    os.close(follower)
    answers, out, tail = list(answers), [], b""
    while True:
        ready, _, _ = select.select([leader], [], [], timeout)
        if not ready:
            process.kill()
            raise TimeoutError("".join(out))
        try:
            data = os.read(leader, 4096)
        except OSError:   # the command exited
            break
        if not data:
            break
        out.append(data.decode(errors="replace"))
        tail = (tail + data)[-200:]
        if tail.rstrip(b" ").endswith(b"]:"):   # a prompt: "... [default: x]:" or "[Y/n]:"
            if not answers:
                process.kill()
                raise AssertionError("asked more questions than the test answers:\n" + "".join(out))
            os.write(leader, (answers.pop(0) + "\n").encode())
            tail = b""
    os.close(leader)
    code = process.wait(timeout=60)
    return Result(code, "".join(out).replace("\r\n", "\n"))


@pytest.fixture
def app(tmp_path):
    """app(source) -> the folder holding app.py with that source."""
    def write(source: str, name: str = "app.py") -> Path:
        (tmp_path / name).write_text(source)
        return tmp_path
    return write
