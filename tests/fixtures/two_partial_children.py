"""Two child modules of two elementwise ops each, different ops so each is its
own region: a one-op kernel covers only part of its scope, so it installs
through a wrapper rather than as a bare call."""

import mlx.core as mx
import mlx.nn as nn


class Clip(nn.Module):
    def __init__(self, low: bool):
        super().__init__()
        self.low = low

    def __call__(self, x):
        y = mx.maximum(x, 0.0) if self.low else mx.minimum(x, 8.0)
        return y * 0.5


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Clip(low=True)
        self.second = Clip(low=False)

    def __call__(self, x):
        return self.second(self.first(x))


def build():
    return Model()
