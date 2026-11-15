"""Fixture: a region whose operations take a setting that follows the size, a
reshape to twice the row count, as attention reshapes to the sequence length."""

import mlx.core as mx
import mlx.nn as nn


class Chain(nn.Module):
    def __call__(self, x):
        y = mx.maximum(x, 0.0)
        y = y.reshape(2 * x.shape[0], 512)
        return y * 0.5


class Sized(nn.Module):
    def __init__(self):
        super().__init__()
        self.chain = Chain()

    def __call__(self, x):
        return self.chain(x)


def build():
    return Sized()
