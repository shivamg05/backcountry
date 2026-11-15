"""Users supply a plain build(); the runner owns kernel capture ordering."""
import textwrap

import mlx.core as mx
import pytest

from backcountry.harness.loop import JobRunner
from backcountry.harness.measure.session import Session
from backcountry.harness.trace.replay import replay
from backcountry.harness.trace.serialize import nodes_from_json, nodes_to_json


@pytest.mark.parametrize('at_import', [True, False])
def test_runner_captures_initial_and_rebuilt_model(tmp_path, at_import):
    factory = '''mx.fast.metal_kernel(name="load_scale", input_names=["x"],
        output_names=["y"], source="uint i = thread_position_in_grid.x; y[i] = x[i] * 2;")'''
    source = 'import mlx.core as mx\nimport mlx.nn as nn\n'
    if at_import:
        source += 'kernel = ' + factory + '\n'
    source += 'class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n'
    source += '        self.kernel = ' + ('kernel' if at_import else factory) + '\n'
    source += '''    def __call__(self, x):
        return self.kernel(inputs=[x], grid=(4,1,1), threadgroup=(4,1,1),
                           output_shapes=[(4,)], output_dtypes=[x.dtype])[0]
def build():
    return Model()
'''
    (tmp_path / 'model.py').write_text(source)
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(textwrap.dedent('''
        model: model.py
        model_type: other
        workloads:
          - name: main
            inputs: [{shape: [4], dtype: float32}]
        budget: {per_region: 1, total: 1}
    '''))
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda region: None,
                       session=Session(sleep=lambda s: None))
    original_factory = mx.fast.metal_kernel
    try:
        runner.load_model()
        runner.trace_workloads()
        initial = runner.traces['main']
        assert len(initial.nodes) == 1 and initial.nodes[0].kernel_definition is not None
        runner.tracer.uninstall()
        rebuilt = runner.build_model()
        assert mx.fast.metal_kernel is original_factory
        runner.tracer.install()
        inputs = runner.tensors['main']
        trace, out = runner.tracer.trace(rebuilt, inputs)
        nodes = nodes_from_json(nodes_to_json(trace.nodes))
        assert nodes[0].kernel_definition == initial.nodes[0].kernel_definition
        got = replay(nodes, {next(iter(trace.inputs)): inputs[0]}, trace.step_outputs)
        assert mx.array_equal(got[trace.step_outputs[0]], out).item()
        runner.tracer.uninstall()
        build = runner.model_module.build
        def failed_build():
            build()
            raise RuntimeError("injected build failure")
        runner.model_module.build = failed_build
        with pytest.raises(RuntimeError, match="injected build failure"):
            runner.build_model()
        assert mx.fast.metal_kernel is original_factory
    finally:
        runner.tracer.uninstall()
    assert mx.fast.metal_kernel is original_factory


def test_a_model_copy_shares_weights_and_compiled_functions_but_not_state():
    import mlx.nn as nn
    from backcountry_runtime.swap import copy_model
    from backcountry_runtime.swap import require_independent_models

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)
            self.act = mx.compile(lambda x: x * mx.sigmoid(x))
            self._state = [mx.zeros((4,))]  # not a parameter: written in place by a call

        def __call__(self, x):
            self._state[0][:1] = mx.array([1.0])
            return self.act(self.proj(x))

    block = Block()
    mx.eval(block.parameters())
    copied = copy_model(block)
    require_independent_models(block, copied)
    assert copied.proj.weight is block.proj.weight and copied.act is block.act
    assert copied._state is not block._state and copied._state[0] is not block._state[0]
    x = mx.ones((2, 8))
    assert mx.array_equal(copied(x), block(x)).item()
    copied._state[0][1:2] = mx.array([5.0])
    assert block._state[0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_a_model_copy_shares_the_tracers_compiled_proxies():
    """Bonsai 2 9B (MLX-VLM) keeps a compiled rotary helper in a module's own
    dict, so while the tracer is installed the module holds the tracer's proxy.
    A copy that copied the proxy copied its recorder too, and the baseline
    model's calls went unrecorded."""
    import mlx.nn as nn
    from backcountry.harness.trace.patches import _CompiledProxy
    from backcountry.harness.trace.recorder import Recorder
    from backcountry_runtime.swap import copy_model

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self._helpers = {2: _CompiledProxy(mx.compile(lambda x: x + 1), Recorder())}

    held = Holder()
    copied = copy_model(held)
    assert copied._helpers is not held._helpers and copied._helpers[2] is held._helpers[2]


@pytest.mark.parametrize('copyable', [True, False])
def test_the_runner_copies_the_built_model_instead_of_building_it_again(tmp_path, copyable):
    (tmp_path / 'model.py').write_text(textwrap.dedent(f'''
        import threading
        import mlx.core as mx
        import mlx.nn as nn
        BUILDS = []
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                {'' if copyable else 'self._lock = threading.Lock()  # cannot be copied'}
            def __call__(self, x):
                return self.proj(x)
        def build():
            BUILDS.append(1)
            mx.random.seed(0)
            return Model()
    '''))
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(textwrap.dedent('''
        model: model.py
        model_type: other
        workloads:
          - name: main
            inputs: [{shape: [2, 4], dtype: float32}]
        budget: {per_region: 1, total: 1}
    '''))
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda region: None,
                       session=Session(sleep=lambda s: None))
    try:
        runner.load_model()
        fresh = runner._fresh_model()
    finally:
        runner.tracer.uninstall()
    builds = len(runner.model_module.BUILDS)
    assert builds == (1 if copyable else 3)
    assert runner.baseline_model.proj.weight is runner.model.proj.weight
    assert fresh.proj is not runner.model.proj and fresh.proj is not runner.baseline_model.proj
    unavailable = [row for row in runner.log.rows() if row["kind"] == "model_copy_unavailable"]
    assert bool(unavailable) == (not copyable)
