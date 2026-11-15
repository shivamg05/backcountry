"""A model file's optional run(model, *inputs): one use of the model the way
its application uses it (a denoising loop, a batch of requests), traced,
checked, timed and shipped as one call in place of one model call."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import yaml

from backcountry.harness.artifact.bundle import ModelBundle
from backcountry.harness.artifact.emit import emit_artifact
from backcountry.harness.bind.emit import Splice
from backcountry.harness.bind.graph import emit_graph_wrapper_variants
from backcountry.harness.loop import JobRunner
from backcountry.harness.measure.session import Session
from backcountry.harness.report import Report
from backcountry_runtime.inference import AppRun
from backcountry_runtime.kernels import KernelSpec

FIXTURE = Path(__file__).parent / "fixtures" / "app_run_model.py"


def _job(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "model": str(FIXTURE), "model_type": "diffusion", "baseline": "plain",
        "workloads": [{"name": "denoise", "inputs": [{"shape": [8, 64], "dtype": "float32"}]}]}))
    return JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: None,
                     session=Session(sleep=lambda _: None))


def test_the_job_traces_and_times_run_not_one_model_call(tmp_path):
    runner = _job(tmp_path)
    try:
        runner.load_model()
        runner.trace_workloads()
        assert isinstance(runner.model, AppRun) and runner.report.constants["measurement"]["kind"] == "app_run"
        trace = runner.traces["denoise"]
        # three steps of the loop: every op of the model three times
        assert sum(n.op == "mx.multiply" for n in trace.nodes if "probe" in (n.module_address or "")) == 3
        x = runner.tensors["denoise"][0]
        out = runner.model(x)
        expected = x
        for t in (1.0, 0.5, 0.25):
            expected = expected - t * runner.model.model(expected)
        assert mx.array_equal(out, expected).item()
        regions = runner.build_regions()
        assert any(r.copies == 3 for r in regions)  # the same region at each step
    finally:
        runner.tracer.uninstall()


def test_a_bundle_runs_and_validates_through_run(tmp_path):
    runner = _job(tmp_path)
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        trace = runner.traces["denoise"]
        kernel = KernelSpec(kernel_id="copy", name="app_run_copy", input_names=("x",), output_names=("out",),
                            source="uint i = thread_position_in_grid.x; out[i] = x[i];",
                            grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("64", "1", "1"),
                            output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",))
        variants = []
        for scope in trace.scope_calls:
            if scope.address.startswith("model.probe@"):
                node = next(n for n in trace.nodes if n.module_address == scope.address and n.op == "mx.multiply")
                variants.append((trace, scope, [Splice(kernel=kernel, start_seq=node.seq, end_seq=node.seq,
                                                       input_ids=node.in_arrays, output_ids=node.out_arrays)]))
        assert len(variants) == 3
        wrapper = emit_graph_wrapper_variants(variants, "AppRunProbe")
        bundle = ModelBundle(model_path=FIXTURE, project_root=FIXTURE.parent, baseline="plain",
                             workloads=runner.tensors, workload_config=runner.manifest.workloads,
                             final_benchmark={"steps": 1, "pairs": 4, "warmup_steps": 1}, app_run=True)
        out = emit_artifact(tmp_path / "artifact", [kernel], [wrapper], Report(), bundle=bundle)
    finally:
        runner.tracer.uninstall()
    metadata = json.loads((out / "bundle.json").read_text())
    assert metadata["app_run"] and metadata["measurement"]["kind"] == "app_run"
    child = textwrap.dedent("""
        import importlib.util, sys
        from pathlib import Path
        import mlx.core as mx
        art = Path(sys.argv[1])
        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            value = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(value)
            return value
        loader = module('bundle_load', art / 'load.py')
        original = loader.load(patched=False)
        patched = loader.load(share_weights_with=original.model)
        rows = module('bundle_validate', art / 'validate.py').validate(original, patched)
        assert all(row['passed'] for row in rows), rows
        x = mx.load(str(art / 'workloads/denoise.safetensors'))['i0']
        assert mx.array_equal(patched(x), original(x)).item()
        # an application passes its bare model to apply() and calls it its own way
        bare = original.definition.build()
        bare.update(original.inference_model.parameters())
        bare = module('bundle_apply', art / 'apply.py').apply(bare)
        assert type(bare.probe).__name__ != 'Probe'
        assert mx.array_equal(original.definition.run(bare, x), original(x)).item()
        print('ok')
    """)
    result = subprocess.run([sys.executable, "-c", child, str(out)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-3000:]
