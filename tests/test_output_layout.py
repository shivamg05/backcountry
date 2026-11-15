"""Output defaults stay out of source, and reruns preserve earlier evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.usefixtures("unused_manifest")
def test_cli_default_is_fresh_and_explicit_paths_still_work(tmp_path, monkeypatch):
    from backcountry.harness import cli
    monkeypatch.chdir(tmp_path)
    paths = []
    monkeypatch.setattr(cli, '_execute', lambda args, parser: paths.append(Path(args.work_dir)) or 0)
    assert cli.main(['run', 'unused.yaml']) == 0
    assert cli.main(['run', 'unused.yaml']) == 0
    assert paths[0].parent == paths[1].parent == Path('runs')
    assert paths[0] != paths[1]
    assert not Path('runs').exists()  # parsing/help must not create run directories
    explicit = tmp_path / 'elsewhere'
    assert cli.main(['run', 'unused.yaml', '--work-dir', str(explicit)]) == 0
    assert paths[-1] == explicit


@pytest.mark.usefixtures("unused_manifest")
@pytest.mark.parametrize('status', ['search_complete', 'paused'])
def test_a_finished_job_drops_its_boundaries_and_a_paused_one_keeps_them(tmp_path, monkeypatch, status):
    """Captured region inputs serve only a search: tens of GB a job, freed
    when it ends, kept while it can resume."""
    from backcountry.harness import cli, loop
    from backcountry.harness.judge.agent import CliJudge
    from backcountry.harness.report import Report
    work = tmp_path / 'work'

    class FinishedRunner:
        final_ok = False

        def __init__(self, manifest, work_dir, judge_factory):
            for job in (work, work / 'kernels'):
                (job / 'boundaries').mkdir(parents=True)

        def run(self):
            report = Report(manifest_path='unused.yaml')
            report.session['status'] = status
            return report

    monkeypatch.setattr(loop, 'JobRunner', FinishedRunner)
    monkeypatch.setattr('backcountry.harness.judge.agent.shutil.which', lambda command: command)
    monkeypatch.setattr(CliJudge, 'check_available', lambda self: None)
    assert cli.main(['run', 'unused.yaml', '--judge', 'claude-cli', '--work-dir', str(work)]) == 0
    kept = status == 'paused'
    assert (work / 'boundaries').exists() == kept and (work / 'kernels' / 'boundaries').exists() == kept


def test_the_summary_says_which_shipped_kernels_are_bit_identical():
    """A user reading "1.09x confirmed" needs to know whether the outputs are
    the original's bit for bit or close to them: each shipped region's last
    accepted kernel counts, a region rolled back or superseded does not."""
    import types
    from backcountry.harness.cli import exactness
    report = types.SimpleNamespace(
        regions=[{"fingerprint": "a", "s": 1.2}, {"fingerprint": "b", "s": 1.1}, {"fingerprint": "c", "s": None}],
        accepted=[{"fingerprint": "a", "assoc_tag": "changing"}, {"fingerprint": "a", "assoc_tag": "preserving"},
                  {"fingerprint": "b", "assoc_tag": "changing"}, {"fingerprint": "c", "assoc_tag": "changing"}])
    assert exactness(report) == "1 bit-identical, 1 within tolerance: outputs differ slightly from the original's"
    assert exactness(types.SimpleNamespace(regions=[], accepted=[])) == ""
