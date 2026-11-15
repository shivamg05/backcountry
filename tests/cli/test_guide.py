"""`backcountry guide` and every command's --help: what an agent reads to run
backcountry must name only commands and flags that exist."""

import re

from .conftest import backcountry

COMMANDS = ["init", "guide", "check", "run", "start", "status", "finish", "pause", "skip", "resume"]


def test_every_command_has_help(tmp_path):
    top = backcountry("--help", cwd=tmp_path)
    assert top.code == 0 and all(command in top.out for command in COMMANDS)
    for command in COMMANDS:
        done = backcountry(command, "--help", cwd=tmp_path)
        assert done.code == 0 and f"usage: backcountry {command}" in done.out, done


def test_the_guide_names_only_commands_and_flags_that_exist(tmp_path):
    guide = backcountry("guide", cwd=tmp_path)
    assert guide.code == 0
    # the commands it gives: indented lines and `quoted` ones, not prose ("run backcountry for someone")
    named = re.findall(r"(?:^\s+|`)backcountry ([a-z]+[^\n`|(]*)", guide.out, re.MULTILINE)
    assert {line.split()[0] for line in named} >= {"init", "check", "start", "status"}
    helps = {command: backcountry(command, "--help", cwd=tmp_path).out for command in COMMANDS}
    for line in named:
        command = line.split()[0]
        assert command in helps, f"the guide names `backcountry {command}`, which does not exist"
        for flag in re.findall(r"--[a-z-]+", line):
            assert flag in helps[command], f"the guide gives `backcountry {command} {flag}`, which it does not take"
    for alternative in re.findall(r"backcountry pause \| ([^\n(]*)", guide.out):   # pause | resume --background | ...
        for part in filter(str.strip, alternative.split("|")):
            command, *flags = part.split()
            assert command in helps and all(f in helps[command] for f in flags), part


def test_the_guide_says_where_a_marked_functions_result_goes_and_how_to_turn_it_off(tmp_path):
    out = backcountry("guide", cwd=tmp_path).out
    assert "backcountry-results/<function>/" in out
    assert "BACKCOUNTRY_OFF=1" in out and "@backcountry.tune(apply=False)" in out
