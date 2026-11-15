"""Fixture: three identical elementwise layers. The first sees the raw input
(within a few units of zero); every later one sees values of 8 and up, so a
kernel can be right on the first copy and wrong on the others."""

import mlx.core as mx
import mlx.nn as nn


class Chain(nn.Module):
    def __call__(self, x):
        y = mx.maximum(x, 0.0)
        return y * 0.5 + 8.0


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Chain() for _ in range(3)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build():
    return Stack()
