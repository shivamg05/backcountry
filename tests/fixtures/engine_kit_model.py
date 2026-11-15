"""The tiny Llama behind tests/fixtures/tiny_engine.py, as an application uses
that engine: build() loads the model the engine keeps in its kit, and run()
calls the engine's create_generator the way the app does."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo, where `tests` is: a model file finds its own imports
import mlx.core as mx
import mlx.nn as nn

from tests.fixtures.tiny_engine import Kit, create_generator


def build():
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(19)
    model = Model(ModelArgs(
        model_type='llama', hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, rms_norm_eps=1e-6))
    model.eval()
    nn.quantize(model, group_size=64, bits=4)
    return model


def run(model, tokens):
    """The engine's kit lives on the model it holds, so the two are freed together."""
    kit = model.__dict__.get("_kit")
    if kit is None:
        kit = Kit(model)
        object.__setattr__(model, "_kit", kit)
    return mx.array(list(create_generator(kit, tokens.reshape(-1).tolist(), max_tokens=8)), dtype=mx.int32)
