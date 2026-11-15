"""apply() replaces a layer only when it is the layer the kernels were made
for: same class, weight shapes and dtypes, quantization and MLX version.
Any other layer keeps its own code, with a warning, so one line is
safe in an engine whose model the artifact does not fit."""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from backcountry_runtime.apply import layer_signature
from backcountry import apply

FIXTURES = Path(__file__).parent / 'fixtures'
PATH = 'model.layers.0.mlp'
WRAPPER = '''import mlx.nn as nn


class W(nn.Module):
    def __init__(self, wrapped, specs):
        super().__init__()
        self.wrapped = wrapped

    def __call__(self, *args):
        return self.wrapped(*args)
'''


def _llama(bits=4):
    spec = importlib.util.spec_from_file_location('llama_fixture', FIXTURES / 'llama_cache_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build()
    if bits != 4:  # the same architecture, quantized otherwise
        from mlx_lm.models.llama import Model, ModelArgs
        model = Model(model.args)
        nn.quantize(model, group_size=64, bits=bits)
    return model


def _artifact(tmp_path, recorded, mlx_version=mx.__version__):
    out = tmp_path / 'artifact'
    (out / 'patch').mkdir(parents=True)
    (out / 'kernels').mkdir()
    (out / 'patch' / 'wrappers.py').write_text(WRAPPER)
    entry = {'scope_path': PATH, 'wrapper_class': 'W', 'kernel_ids': [], 'span_map': []}
    if recorded is not None:
        entry['module'] = recorded
    (out / 'swap_table.json').write_text(json.dumps([entry]))
    (out / 'bundle.json').write_text(json.dumps({'mlx': mlx_version}))
    return out


def _recorded():
    return json.loads(json.dumps(layer_signature(_llama().model.layers[0].mlp)))


def test_the_layer_the_kernels_were_made_for_is_replaced(tmp_path):
    art = _artifact(tmp_path, _recorded())
    model = apply(_llama(), art)
    assert type(model.model.layers[0].mlp).__name__ == 'W'
    # applied twice, the layer is already replaced: left as it is
    with pytest.warns(UserWarning, match='a W where the kernels were made for a MLP'):
        again = apply(model, art)
    assert type(again.model.layers[0].mlp.wrapped).__name__ == 'MLP'


def test_any_other_layer_keeps_its_own_code_and_says_why(tmp_path):
    art = _artifact(tmp_path, _recorded())
    with pytest.warns(UserWarning, match='1 of 1 layers left as they are.*quantized differently'):
        model = apply(_llama(bits=8), art)
    assert type(model.model.layers[0].mlp).__name__ == 'MLP'
    with pytest.raises(ValueError, match='quantized differently'):
        apply(_llama(bits=8), art, strict=True)  # the artifact's own checks
    other = _llama()
    other.model.layers.pop()
    other.model.layers.pop()  # a smaller model: no layer 0
    with pytest.warns(UserWarning, match='the model has no layer there'):
        apply(other, art)


def test_another_mlx_version_replaces_nothing_a_patch_release_included(tmp_path):
    """Qwen3-30B-A3B kernels exact under MLX 0.32.2 changed the reply from the
    second token under 0.32.4, whose expert matmul sums in another order."""
    major, minor, patch = mx.__version__.split('.')[:3]
    for i, other in enumerate((f'{major}.{int(minor) + 1}.0', f'{major}.{minor}.{int(patch) + 1}')):
        with pytest.warns(UserWarning, match=f'made with MLX {other}'):
            model = apply(_llama(), _artifact(tmp_path / str(i), _recorded(), other))
        assert type(model.model.layers[0].mlp).__name__ == 'MLP'
    model = apply(_llama(), _artifact(tmp_path / 'same', _recorded(), mx.__version__))
    assert type(model.model.layers[0].mlp).__name__ == 'W'


def test_an_artifact_exported_before_layers_were_recorded_installs_as_before(tmp_path):
    model = apply(_llama(), _artifact(tmp_path, None))
    assert type(model.model.layers[0].mlp).__name__ == 'W'


def test_an_engine_that_nests_the_model_elsewhere_gets_the_same_layers(tmp_path):
    """An engine that holds the same layers at other addresses (MLX-VLM keeps
    the language model at language_model; another hands over the model's
    inner body alone) gets them replaced where they are, each still checked
    against the layer its kernels were made for."""
    art = _artifact(tmp_path, _recorded())

    class Engine(nn.Module):
        def __init__(self, language_model):
            super().__init__()
            self.language_model = language_model

    engine = apply(Engine(_llama()), art)
    assert type(engine.language_model.model.layers[0].mlp).__name__ == 'W'
    body = apply(_llama().model, art)
    assert type(body.layers[0].mlp).__name__ == 'W'
    # nested elsewhere and quantized otherwise: no layer is the one, nothing changes
    with pytest.warns(UserWarning, match='the model has no layer there'):
        other = apply(Engine(_llama(bits=8)), art)
    assert type(other.language_model.model.layers[0].mlp).__name__ == 'MLP'


def test_another_chip_gets_the_kernels_with_a_warning_that_their_speed_is_unmeasured(tmp_path):
    import warnings
    art = _artifact(tmp_path, _recorded())
    here = mx.device_info()['device_name']
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        (art / 'bundle.json').write_text(json.dumps({'mlx': mx.__version__, 'chip': here}))
        assert type(apply(_llama(), art).model.layers[0].mlp).__name__ == 'W'
    (art / 'bundle.json').write_text(json.dumps({'mlx': mx.__version__, 'chip': 'Apple M0 Imaginary'}))
    with pytest.warns(UserWarning, match='timed on an Apple M0 Imaginary.*unmeasured'):
        model = apply(_llama(), art)
    assert type(model.model.layers[0].mlp).__name__ == 'W'  # the same values on any chip


def test_apply_without_an_artifact_uses_the_newest_stored_one_that_fits(tmp_path, monkeypatch):
    """apply(model) alone: a finished job left its artifact in the local
    store; the newest one whose layers this model has is used, and a model
    none fits runs as it is, with a warning."""
    import os
    import time
    from backcountry_runtime.apply import artifacts_home, find, register
    monkeypatch.setenv('BACKCOUNTRY_HOME', str(tmp_path / 'home'))
    with pytest.warns(UserWarning, match='no artifact in .* is for this model'):
        assert type(apply(_llama()).model.layers[0].mlp).__name__ == 'MLP'
    other = _recorded()
    other['quantization'] = {k: [8, 64, 'affine'] for k in other['quantization']}
    old = register(_artifact(tmp_path / 'a', _recorded()), 'old-job')
    time.sleep(0.01)
    register(_artifact(tmp_path / 'b', other), 'newer-job-for-8-bit')
    assert find(_llama()) == old  # the newer one is for another quantization
    assert type(apply(_llama()).model.layers[0].mlp).__name__ == 'W'
    time.sleep(0.01)
    newest = register(_artifact(tmp_path / 'c', _recorded()), 'newest-job')
    os.utime(newest / 'swap_table.json')
    assert find(_llama()) == newest and newest.parent == artifacts_home()
    # an artifact exported before layers were recorded says nothing of what it fits
    register(_artifact(tmp_path / 'd', None), 'unrecorded-job')
    assert find(_llama()) == newest


def test_a_result_names_the_libraries_installed_at_other_versions(tmp_path):
    """requirements.txt pins every library the model code ran on (mlx-lm,
    mlx-vlm, mflux, an engine's): one at another version here voids an exact
    result made against it, MLX's own included."""
    from importlib import metadata
    import mlx.core as mx
    from backcountry_runtime.apply import changed_libraries
    major, minor, *_ = mx.__version__.split(".")
    (tmp_path / "requirements.txt").write_text(
        f"# pins\nmlx=={major}.{minor}.999\nmlx-lm=={metadata.version('mlx-lm')}\n"
        "mlx-lm-not-a-real-package==1.0\nmlx-vlm==0.0.1\n")
    changed = changed_libraries(tmp_path)
    assert changed[0] == f"mlx {major}.{minor}.999 -> {mx.__version__}"  # a patch release counts
    assert changed[1] == "mlx-lm-not-a-real-package 1.0 -> not installed"
    assert changed[2].startswith("mlx-vlm 0.0.1 -> ") and len(changed) == 3  # mlx-lm: same
    assert changed_libraries(tmp_path / "nothing-here") == []
