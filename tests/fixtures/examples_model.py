"""A small model whose file shows its use: examples() returns real inputs,
one tuple per call, at the sizes its application sends (16, 8 and 32 tokens)."""
import mlx.core as mx
import mlx.nn as nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 64)
        self.proj = nn.Linear(64, 64)

    def __call__(self, x, ids):
        return nn.gelu(self.proj(x + self.embed(ids)))


def build():
    mx.random.seed(0)
    model = Block()
    mx.eval(model.parameters())
    return model


def examples():
    mx.random.seed(1)
    return [(mx.random.normal((1, t, 64)).astype(mx.bfloat16), mx.random.randint(3, 40, (1, t)))
            for t in (16, 8, 32)]
