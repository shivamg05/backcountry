"""A CLI judge may test kernels with ./try while it writes a reply: the job
answers each try with a quick check, and the agent's reply comes after."""
import json

from backcountry.harness.judge.agent import MAX_TRIES, SYSTEM_TOKEN, TRY_GUIDE, CliJudge, claude_argv

# a stand-in agent: tries twice, then replies with what the second try printed
AGENT = """
./try <<'END' > first.json
{"kernel": {"n": 1}}
END
printf 'not json' | ./try > second.json
cat first.json
"""


def test_the_agent_tests_kernels_and_the_job_answers_each_try():
    seen = []

    def handler(payload):
        seen.append(payload)
        return {"outcome": "faster", "kernel_ms": 1.0, "library_ms": 2.0}

    judge = CliJudge(["sh", "-c", "exit 3"], try_command=["sh", "-c", AGENT])
    judge.try_handler = handler
    reply = json.loads(judge._ask("system", [{"role": "user", "content": "hi"}]))
    assert reply == {"outcome": "faster", "kernel_ms": 1.0, "library_ms": 2.0}
    assert seen == [{"kernel": {"n": 1}}]  # the second payload was not JSON: answered with an error, not handled


def test_without_a_handler_the_agent_runs_without_tools():
    judge = CliJudge(["sh", "-c", "echo plain"], try_command=["sh", "-c", "echo with-tries"])
    assert judge._ask("system", [{"role": "user", "content": "hi"}]) == "plain"


def test_the_guide_reaches_the_agent_and_tries_are_capped(tmp_path):
    loop = f"for i in $(seq 1 {MAX_TRIES + 1}); do echo '{{}}' | ./try; echo; done; echo; echo SYSTEM: \"$1\""
    judge = CliJudge(["sh", "-c", "exit 3"], try_command=["sh", "-c", loop, "agent", SYSTEM_TOKEN])
    judge.try_handler = lambda payload: {"outcome": "correct_slower"}
    text = judge._ask("the system", [{"role": "user", "content": "hi"}])
    answers = [json.loads(line) for line in text.split("SYSTEM:")[0].splitlines() if line.startswith("{")]
    assert sum(a.get("outcome") == "correct_slower" for a in answers) == MAX_TRIES
    assert "tries for this reply are used" in answers[-1]["error"]
    assert "the system" in text and TRY_GUIDE.splitlines()[0] in text


def test_claude_is_allowed_one_command_when_it_may_try():
    argv = claude_argv("m", "low", tries=True)
    assert argv[argv.index("--tools") + 1] == "Bash" and argv[argv.index("--allowedTools") + 1] == "Bash(./try)"
    plain = claude_argv("m", "low")
    assert plain[plain.index("--tools") + 1] == "" and "--allowedTools" not in plain
