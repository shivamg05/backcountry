"""A prompt read and replies after a conversation in one job: each workload
calls its own step on one model (the reply its conversation, the prompt the
same model from no cache), a kernel installed for one workload leaves the
other exact, and the artifact reproduces both in a fresh process."""

from pathlib import Path

import pytest

from backcountry.harness.ladder.gates import LadderResult
from backcountry.harness.loop import JobRunner, RegionRun
from backcountry.harness.measure.session import Session
from backcountry_runtime.state import ContextStep, PlainStep

pytestmark = pytest.mark.integration  # whole jobs and fresh-process exports


@pytest.mark.parametrize('fixture,tag', [('llama_cache_model.py', 'preserving'), ('qwen_cache_model.py', 'preserving'),
                                         ('llama_cache_model.py', 'changing')])
def test_a_prompt_and_a_reply_share_one_model_one_search_and_one_artifact(tmp_path, monkeypatch, fixture, tag):
    model = Path(__file__).parent / 'fixtures' / fixture
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(f'''model: {model}
model_type: other
baseline: plain
use_library_inference: false
workloads:
  - name: prompt
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
  - name: reply
    context: 5
    inputs: [{{shape: [1, 1], dtype: int32, high: 64}}]
''')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda _: None, clock_pairs=4,
                       session=Session(sleep=lambda _: None))
    # An identity seed is not a speedup. Only the speed decision is controlled;
    # execution, both workloads' checks, installation and fresh export are real.
    monkeypatch.setattr(runner, '_model_win', lambda result: all(c.passed for c in result.checks))
    try:
        runner.load_model()
        assert isinstance(runner.model, ContextStep)
        assert runner._for(runner.model, 'reply') is runner.model
        prompt = runner._for(runner.model, 'prompt')
        assert isinstance(prompt, PlainStep) and prompt.model is runner.model.model  # one model
        runner.trace_workloads()
        assert {'prompt', 'reply'} <= set(runner.traces)
        # the prompt is read from no history: the model's own call; the reply follows its conversation
        import mlx.core as mx
        x = runner.tensors['prompt'][0]
        bare = runner.baseline_model.model(x)
        assert mx.array_equal(runner._for(runner.baseline_model, 'prompt')(x), bare).item()
        y = runner.tensors['reply'][0]
        assert not mx.array_equal(runner._for(runner.baseline_model, 'reply')(y), runner.baseline_model.model(y)).item()
        regions = runner.build_regions()
        region = next(r for r in regions if r.ops == ('array.__add__',) and not r.rejected)
        kernel = runner._build_scaffold(region)
        runner._capture([r for r in regions if not r.rejected])  # each workload's boundaries, by its own step
        assert not region.rejected, region.rejected
        for w in region.workloads:
            region.p[w] = 1.0
            region.t_orig_ms[w] = region.t_rep_ms[w] = 1.0
        runner.tracer.uninstall()
        assert runner._bind_and_promote(RegionRun(region), kernel,
                                        LadderResult('tentative_ship', None, {}, .1, 1., .1, 0., [])), \
            runner.log.rows()[-1]
        checked = [r for r in runner.log.rows() if r['kind'] == 'verdict' or r['kind'] == 'shipped']
        assert runner.installed, checked
        if tag == 'changing':  # graded against the fp32 reference: each workload's golden from its own step
            runner.shipped_tags[region.fingerprint] = 'changing'
        runner.final_ok = True
        artifact = runner._emit_artifact(tmp_path / 'artifact')  # both workloads, fresh process
    finally:
        runner.tracer.uninstall()
    import json
    bundle = json.loads((artifact / 'bundle.json').read_text())
    assert set(bundle['workloads']) >= {'prompt', 'reply'} and bundle['context']['workload'] == 'reply'


def test_a_loop_and_a_changing_kernel_on_a_prompt_and_a_reply_ship_one_artifact(tmp_path, monkeypatch):
    """The whole chat job, end to end: the loop search on the reply, then the
    kernel job on both workloads with a kernel graded against its fp32
    reference, the loop's final with the kernels, and one artifact whose
    fresh-process check reproduces both workloads and runs run.py. Every part
    passed alone before Gemma 4's export failed on their combination."""
    import backcountry.harness.loop as loop_mod
    from backcountry.harness import model_search
    from tests.test_identical_outputs import SAME_STEP, Scripted, _win
    model = Path(__file__).parent / 'fixtures' / 'llama_cache_model.py'
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(f'''model: {model}
model_type: other
baseline: plain
use_library_inference: false
identical: output
workloads:
  - name: prompt
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
  - name: reply
    context: 5
    inputs: [{{shape: [1, 1], dtype: int32, high: 64}}]
budget: {{per_region: 1, total: 1, model: 1}}
final_benchmark: {{steps: 4}}
tolerances: default
''')
    # A scaffold is not a speedup, nor a tiny model's noise a loss: only the speed decisions are controlled.
    monkeypatch.setattr(model_search, 'compare_on_inputs', lambda *a, **k: _win())
    monkeypatch.setattr(JobRunner, '_model_win', lambda self, result: all(c.passed for c in result.checks))
    monkeypatch.setattr(loop_mod, 'FINAL_MARGIN', float('inf'))

    def search(self):  # the kernel job's search: one MLP region's scaffold, shipped as a changing cut
        ranked = self._search_state['ranked']
        # in a block that holds no cache, as Gemma 4's expert kernels were
        mlp = [r for r in ranked + self.pending_regions if not r.rejected
               and all(m.scope_stack[-1].split('@')[0].endswith('.mlp') for m in r.members)]
        region = min(mlp, key=lambda r: len(r.ops))
        kernel = self._build_scaffold(region)
        if region not in ranked:  # captured the job's way, under the recorder
            self.tracer.install()
            self._capture([region])
        for w in region.workloads:
            region.p[w] = 1.0
            region.t_orig_ms[w] = region.t_rep_ms[w] = 1.0
        self.tracer.uninstall()
        promoted = self._bind_and_promote(RegionRun(region), kernel,
                                          LadderResult('tentative_ship', None, {}, .1, 1., .1, 0., []), 'changing')
        assert promoted.status == 'shipped', [c for c in promoted.checks if not c.get('passed')] or promoted
    monkeypatch.setattr(JobRunner, '_search', search)
    judge = Scripted([{'hypothesis': 'the same generation', 'kind': 'same', 'source': SAME_STEP},
                      {'verdict': 'accept', 'reason': 'same work'}])
    runner = JobRunner(manifest, tmp_path / 'work', session=Session(sleep=lambda _: None),
                       judge_factory=lambda region: judge)
    report = runner.run()
    kernels = runner._kernel_job
    assert kernels.final_ok and 'changing' in kernels.shipped_tags.values(), (kernels.report.final, kernels.shipped_tags)
    assert report.model_level['final']['ok'] and 'kernels_dropped' not in report.model_level, report.model_level
    artifact = runner.emit_artifact(tmp_path / 'artifact')  # both workloads and run.py, fresh process
    import json
    bundle = json.loads((artifact / 'bundle.json').read_text())
    assert set(bundle['workloads']) >= {'prompt', 'reply'} and (artifact / 'run.py').is_file(), report.model_level.get('not_shipped')
    assert bundle['goldens'] and not (tmp_path / 'work' / 'kernels' / 'export_goldens').exists()  # written, then cleared
