"""compare(): the user's own function on the model as loaded and with the
artifact's layers switched in, alternately, on one copy of the weights."""

import mlx.core as mx
import pytest

from backcountry import compare
from tests.test_apply_check import WRAPPER, _artifact, _llama, _recorded

WRONG = WRAPPER.replace("return self.wrapped(*args)", "return self.wrapped(*args) + 1")


def test_identical_outputs_and_a_speedup_through_the_users_function(tmp_path):
    model = _llama()
    original_mlp = model.model.layers[0].mlp
    tokens = mx.array([[1, 2, 3, 4]])
    calls = []

    def generate(m):  # how an application might use the model: logits and a sampled token
        calls.append(type(m.model.layers[0].mlp).__name__)
        logits = m(tokens)
        return {"logits": logits, "token": mx.random.categorical(logits[:, -1]).item(), "tps": 123.4 * len(calls)}

    result = compare(generate, model, _artifact(tmp_path, _recorded()), runs=3)
    assert result.replaced == 1 and result.identical, result.difference
    assert len(result.stock_s) == len(result.optimized_s) == 3 and result.speedup > 0
    assert calls[:2] == ["MLP", "W"] and calls.count("W") == calls.count("MLP") == 4
    assert model.model.layers[0].mlp is original_mlp  # left as it was given
    assert "outputs identical" in str(result) and "median of 3 runs each" in str(result)


def test_a_difference_is_named_and_nothing_fits_says_so(tmp_path):
    art = _artifact(tmp_path, _recorded())
    (art / "patch" / "wrappers.py").write_text(WRONG)
    result = compare(lambda m: m(mx.array([[1, 2, 3]])), _llama(), art, runs=1)
    assert not result.identical and "differs by up to" in result.difference
    with pytest.warns(UserWarning, match="quantized differently"):
        other = compare(lambda m: m(mx.array([[1]])), _llama(bits=8), _artifact(tmp_path / "b", _recorded()))
    assert other.replaced == 0 and "nothing to compare" in str(other)


def test_the_artifact_store_is_in_backcountrys_folder(tmp_path, monkeypatch):
    from backcountry_runtime.apply import artifacts_home
    monkeypatch.delenv("BACKCOUNTRY_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert artifacts_home() == tmp_path / ".backcountry" / "artifacts"
    monkeypatch.setenv("BACKCOUNTRY_HOME", str(tmp_path / "new"))
    assert artifacts_home() == tmp_path / "new" / "artifacts"
