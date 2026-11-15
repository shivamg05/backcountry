"""Fixture: a layer whose rows grow with the square of the input length (every
pair of positions), so its size is no straight-line function of that length."""

import mlx.core as mx
import mlx.nn as nn


class Chain(nn.Module):
    def __call__(self, x):
        return mx.maximum(x, 0.0) * 2.0


class SquareRows(nn.Module):
    def __init__(self):
        super().__init__()
        self.chain = Chain()

    def __call__(self, x):  # x: [L, 64] -> the chain sees [L * L, 64]
        return self.chain((x[:, None, :] * x[None, :, :]).reshape(-1, 64))


def build():
    return SquareRows()
