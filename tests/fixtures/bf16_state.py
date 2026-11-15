"""Fixture: a bf16 calculation feeding an fp32 running state, Qwen3.5's
gated-delta memory in miniature. The library rounds the multiply-add twice in
bf16; a kernel rounding once in fp32 is more accurate, and moves the fp32
state far more than an fp32 tolerance (1e-5) allows."""

import mlx.core as mx
import mlx.nn as nn


class Mix(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = mx.random.normal((1024,)).astype(mx.bfloat16)
        self.b = mx.random.normal((1024,)).astype(mx.bfloat16)

    def __call__(self, x):
        h = x.astype(mx.bfloat16)
        return h * self.a + self.b


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(5)
        self.mix = Mix()

    def __call__(self, x):
        y = self.mix(x)
        return y, mx.cumsum(y.astype(mx.float32), axis=-1)


def build():
    return Model()
