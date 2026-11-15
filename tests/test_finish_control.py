from types import SimpleNamespace
import json
import pytest
from backcountry.harness.cli import main, _run_lock
from backcountry.harness.loop import JobRunner
from backcountry.harness.e2e import E2EResult
from backcountry_runtime.stats import comparison_from_samples
from tests.test_run_recovery import bare_runner


def test_finish_command_does_not_take_gpu_lock(tmp_path):
    r = bare_runner(tmp_path)
    (tmp_path / 'run.jsonl').touch()
    with _run_lock():
        assert main(['finish', '--work-dir', str(tmp_path)]) == 0
    assert r._close_rule(None) == 'operator requested final validation'
    r._close_rule(None)
    rows = [json.loads(x) for x in (tmp_path / 'run.jsonl').read_text().splitlines()]
    assert sum(x['kind'] == 'search_finish_requested' for x in rows) == 1
    r.hypothesis_cycle(SimpleNamespace(close_rule=None), None)  # no judge call


def test_finish_requires_existing_run(tmp_path):
    with pytest.raises(SystemExit):main(['finish', '--work-dir', str(tmp_path)])
    assert not (tmp_path / 'finish-search.request').exists()


@pytest.mark.parametrize('sequence_wins', [True, False])
def test_inconclusive_single_step_reaches_sequences(tmp_path, monkeypatch, sequence_wins):
    import backcountry.harness.loop as loop
    r=bare_runner(tmp_path)
    r.installed={'layer': object()};r.shipped_tags={};r.baseline_model=r.model=None
    r.session=SimpleNamespace(wait_ready=lambda:None,memory=lambda where,**detail:{});r.tensors={'main':[]};r.manifest.workloads=[SimpleNamespace(name='main')]
    r._timed_arms=lambda:{}
    tie=comparison_from_samples([10]*10,[9,11]*5)
    monkeypatch.setattr(loop,'run_e2e',lambda *a,**k:E2EResult(veto=tie,workload_vetos={'main':tie}))
    seq=comparison_from_samples([200]*10, [180]*10 if sequence_wins else [199,201]*5)
    seen=[]
    def sequences(checks):
        seen.append(True)
        return {'main':{'win_confirmed':sequence_wins,'speedup':1/seq.median_ratio}}, E2EResult(veto=seq,workload_vetos={'main':seq})
    r._final_sequences=sequences
    r._final_check()  # right outputs and no confirmed win ends the job normally, with the reason recorded
    if not sequence_wins: assert 'consecutive steps' in r.report.final['reason']
    assert seen == [True]
    assert r.final_ok == sequence_wins
    assert not r.report.final['paired_win_confirmed']


def test_regression_still_blocks_sequences(tmp_path,monkeypatch):
    import backcountry.harness.loop as loop
    r=bare_runner(tmp_path);r.installed={'layer':object()};r.shipped_tags={}
    r.baseline_model=r.model=None;r.session=SimpleNamespace(wait_ready=lambda:None,memory=lambda where,**detail:{});r.tensors={};r._timed_arms=lambda:{}
    monkeypatch.setattr(loop,'run_e2e',lambda *a,**k:E2EResult(veto_passed=False))
    r._final_sequences=lambda checks:pytest.fail('regression reached sequences')
    r._final_check()  # a measured slowdown is a result, not a crash; it still never reaches the sequences
    assert r.final_ok is False and 'slower' in r.report.final['reason']


def test_finish_goes_to_final_checks_without_opening_region(tmp_path):
    r=bare_runner(tmp_path)
    r.manifest.budget_per_region=40;r.manifest.budget_total=200;r.manifest.defaulted=[]
    r.gpu_busy_at_start=None;r.pending_regions=[];r.installed={'saved':(object(),{},{})}
    r.session=SimpleNamespace(idled_s=0,memory=lambda where,**detail:{})
    r.load_model=r.trace_workloads=lambda:None
    r.build_regions=lambda:[]
    r.capture_and_price=lambda regions:[object()]
    r.judge_factory=lambda region:pytest.fail('opened a new region')
    seen=[];r._final_check=lambda:seen.append(True)
    (tmp_path/'finish-search.request').touch()
    r._run()
    assert seen == [True]
    assert r.report.coverage['selection']['unsearched'] == 1


def test_skip_closes_the_open_region_and_the_next_one_opens_clean(tmp_path):
    """`backcountry skip` closes the region being searched at its next check
    between attempts; the job moves on, and the request ends when the next
    region opens, so it never skips that one too."""
    r = bare_runner(tmp_path)
    (tmp_path / 'run.jsonl').touch()
    region = lambda name: SimpleNamespace(fingerprint=name, ops=['mx.add'], copies=1, p={},
                                          members=[SimpleNamespace(workload='w', start_seq=0, end_seq=0)],
                                          t_orig_ms={}, t_rep_ms={}, roofline=None, workloads=[])
    r._search_state = {'ranked': [region('a'), region('b')], 'shipped': [], 'run': None, 'accepted_before': 0}
    r.pending_regions, r.total_hypotheses, r.candidates = [], 0, []
    r.manifest.budget_total = 10
    r._safe_point = lambda where: None
    r._budget = lambda run: {'attempts_left_region': 1, 'attempts_left_job': 1}
    r.judge_factory = lambda reg: None
    r._openers = lambda run: 1
    r.open_region = lambda reg, judge: SimpleNamespace(region=reg, scaffold=object(), close_rule=None, openers=[])
    rules = {}

    def cycle(run, judge):
        if run.region.fingerprint == 'a':
            assert main(['skip', '--work-dir', str(tmp_path)]) == 0  # asked mid-region
        rules[run.region.fingerprint] = r._close_rule(run)

    r.hypothesis_cycle = cycle
    r._close_region = lambda run: True
    r._search()
    assert rules == {'a': 'operator skipped the region', 'b': None}
    assert not (tmp_path / 'skip-region.request').exists()


def test_one_call_per_use_is_decided_by_the_paired_check(tmp_path, monkeypatch):
    """final_benchmark.steps: 1 with direct calls (a prompt read once per chat
    turn): the paired check already timed one call, so no runs of calls in a
    row are timed and its result decides."""
    import backcountry.harness.loop as loop
    from backcountry.harness.manifest import FinalBenchmark
    for wins, ok in ((True, True), (False, False)):
        r = bare_runner(tmp_path)
        r.installed = {'layer': object()}; r.baseline_model = r.model = None
        r.session = SimpleNamespace(wait_ready=lambda: None, memory=lambda where, **detail: {})
        r.tensors = {'main': []}; r.manifest.workloads = [SimpleNamespace(name='main')]
        r.manifest.final_benchmark = FinalBenchmark(steps=1)
        r._timed_arms = lambda: {}
        comp = comparison_from_samples([10] * 10, [9] * 10 if wins else [9, 11] * 5)
        monkeypatch.setattr(loop, 'run_e2e', lambda *a, **k: E2EResult(veto=comp, workload_vetos={'main': comp}))
        r._final_sequences = lambda checks: pytest.fail('a single call per use reached the runs in a row')
        r._final_check()
        assert r.final_ok is ok and r.report.final['paired_win_confirmed'] is ok


def test_a_kernel_search_a_loop_search_started_takes_the_operators_requests(tmp_path):
    """`backcountry finish`, `skip` and `pause` write into the job's folder; the
    kernel search a loop search starts works in its kernels/ folder and reads
    them there too (it never saw them before: a finish went unheard)."""
    from backcountry.harness import resume as resume_mod
    (tmp_path / 'kernels').mkdir()
    r = bare_runner(tmp_path / 'kernels')
    (tmp_path / 'run.jsonl').touch()
    assert main(['finish', '--work-dir', str(tmp_path)]) == 0
    assert not r._finish_requested()  # as before: the request was in the job's folder only
    r.parent_dir = tmp_path
    assert r._close_rule(None) == 'operator requested final validation'
    (tmp_path / 'finish-search.request').unlink()
    (tmp_path / resume_mod.SKIP_REQUEST).write_text('skip\n')
    assert r._close_rule(None) == 'operator skipped the region'
    (tmp_path / resume_mod.PAUSE_REQUEST).write_text('pause\n')
    assert any(resume_mod.pause_requested(d) for d in r._request_dirs())
    assert not (tmp_path / resume_mod.PAUSE_REQUEST).exists()  # used once


def test_a_time_budget_is_refused_attempts_bound_a_run(tmp_path):
    from backcountry.harness.manifest import ManifestError, load
    from tests.test_run_recovery import FIXTURES
    base = f"model: {FIXTURES / 'planted_win.py'}\nmodel_type: other\nworkloads:\n  - name: main\n    inputs: [{{shape: [4], dtype: float32}}]\n"
    path = tmp_path / "hours.yaml"
    path.write_text(base + "budget: {total: 8, hours: 2}\n")
    with pytest.raises(ManifestError, match="attempts bound a run"):
        load(path)


def test_a_job_the_gpu_failed_under_resumes_in_a_fresh_process(tmp_path, monkeypatch):
    """A GPU failure mid-job (a macOS GPU timeout while the Mac was busy)
    re-runs the job from its last save in a new process after a rest, up to
    three times; any other error, or a job with no save, stops as before."""
    from backcountry.harness import cli
    from backcountry.harness.resume import STATE
    execs, sleeps = [], []
    monkeypatch.setattr(cli.os, 'execv', lambda exe, argv: execs.append(argv))
    monkeypatch.setattr('time.sleep', lambda s: sleeps.append(s))
    monkeypatch.delenv('AUTOTUNE_AUTO_RESUMES', raising=False)
    gpu = RuntimeError('[METAL] Command buffer execution failed: Caused GPU Timeout Error (00000002)')
    cli._auto_resume(gpu, tmp_path)
    assert not execs  # nothing saved to resume from
    (tmp_path / STATE).write_bytes(b'saved')
    cli._auto_resume(ValueError('a bug in the tool'), tmp_path)
    assert not execs
    for _ in range(cli.AUTO_RESUMES + 1):
        cli._auto_resume(gpu, tmp_path)
    assert len(execs) == cli.AUTO_RESUMES and sleeps == [cli.AUTO_RESUME_WAIT_S] * cli.AUTO_RESUMES
    assert execs[0][-4:] == ['backcountry.harness.cli', 'resume', '--work-dir', str(tmp_path)]
    # an identical: output job saves its kernel search under kernels/; with no loop kept, nothing above it
    execs.clear()
    monkeypatch.delenv('AUTOTUNE_AUTO_RESUMES')
    job = tmp_path / 'loop_then_kernels'
    (job / 'kernels').mkdir(parents=True)
    (job / 'kernels' / STATE).write_bytes(b'saved')
    cli._auto_resume(gpu, job)
    assert execs and execs[0][-1] == str(job)
