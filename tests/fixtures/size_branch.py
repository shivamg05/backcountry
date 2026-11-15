"""Fixture: two child modules over rows of x. `chain` runs the same
operations at every row count; `branchy` adds one when there are more than
32 rows, as a model body does when a prompt passes its sliding window (a
different attention mask). Over a range of row counts a kernel can follow
the size in chain, never in branchy."""

import mlx.core as mx
import mlx.nn as nn


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = mx.random.normal((1024,))

    def __call__(self, x):
        y = x * 2.0
        y = y + self.a
        return mx.maximum(y, 0.0)


class Branchy(nn.Module):
    def __call__(self, x):
        y = mx.minimum(x, 8.0)
        if x.shape[0] > 32:
            y = y * 0.5
        return mx.exp(y)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(23)
        self.chain = Chain()
        self.branchy = Branchy()

    def __call__(self, x):
        return self.branchy(self.chain(x))


def build():
    return Model()
