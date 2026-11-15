"""A toy denoiser used the way a sampling application uses it: run() calls
the model once per step in a Python loop, as an image model's denoising
loop does. The probe is an identity a test can swap a kernel into."""

import mlx.core as mx
import mlx.nn as nn

STEPS = (1.0, 0.5, 0.25)


class Probe(nn.Module):
    def __call__(self, value):
        return mx.multiply(value, 1.0)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64)
        self.b = nn.Linear(64, 64)
        self.probe = Probe()

    def __call__(self, x):
        return self.probe(self.b(nn.gelu(self.a(x))))


def build():
    mx.random.seed(0)
    model = Denoiser()
    mx.eval(model.parameters())
    return model


def run(model, x):
    for t in STEPS:
        x = x - t * model(x)
    return x
