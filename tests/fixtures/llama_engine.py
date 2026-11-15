"""The tiny Llama behind an engine's own run(): read the prompt (and any
conversation before it) and write four tokens greedily, the way an inference
engine built on mlx-lm does. The tool times, checks and optimizes this call."""
import mlx.core as mx
import mlx.nn as nn


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
    from mlx_lm.generate import generate_step
    return mx.array([token for token, _ in generate_step(tokens.reshape(-1), model, max_tokens=4)], dtype=mx.int32)
