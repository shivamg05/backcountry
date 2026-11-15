"""Small apps the CLI tests run, each shaped like a real one of its kind,
with its function marked: a chat reply (text in tokens, a reply length), an
image generator (width and height, steps, a prompt and seed), a forecaster
(numpy arrays) and a classifier (a batch of images). Tiny models with random
weights: nothing is downloaded."""

CHAT = '''
import mlx.core as mx
import mlx.nn as nn
import backcountry


class Tokenizer:
    """One token per word."""

    def encode(self, text):
        return [sum(map(ord, word)) % 64 for word in text.split()]

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 32)
        self.block = nn.TransformerEncoderLayer(32, 4)
        self.out = nn.Linear(32, 64)

    def __call__(self, tokens):
        return self.out(self.block(self.embed(tokens), mask=None))


tokenizer = Tokenizer()
mx.random.seed(0)
model = LM()


@backcountry.tune
def reply(question, max_tokens=4):
    tokens = mx.array([tokenizer.encode(question)])
    last = mx.argmax(model(tokens)[:, -1], axis=-1)          # read the question
    out = [last]
    for _ in range(max_tokens - 1):                           # then one token at a time
        last = mx.argmax(model(last[:, None])[:, -1], axis=-1)
        out.append(last)
    return tokenizer.decode(mx.concatenate(out).tolist())


print(reply("what is mlx"))
print(reply("write a short poem about the sea and then explain it"))
'''


# A reply returned with its speed, as library result objects carry one: the
# same request gives a different result every time.
TIMED = CHAT.replace("def reply(question, max_tokens=4):\n", "def reply(question, max_tokens=4):\n"
                     "    import time\n    start = time.perf_counter()\n").replace(
    "    return tokenizer.decode(mx.concatenate(out).tolist())",
    "    text = tokenizer.decode(mx.concatenate(out).tolist())\n"
    "    return {\"text\": text, \"tokens_per_sec\": max_tokens / (time.perf_counter() - start)}")


# A result that repeats only to rounding, as Kokoro's audio does (its GPU math
# is not bit-repeatable): here a draw no replay seeds.
JITTER = CHAT.replace(
    "    return tokenizer.decode(mx.concatenate(out).tolist())",
    "    import numpy as np\n"
    "    wave = mx.concatenate(out).astype(mx.float32) / 64\n"
    "    return wave + mx.array(np.random.default_rng().normal(size=wave.shape) * 1e-7, dtype=mx.float32)")

IMAGE = '''
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import backcountry


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.text = nn.Linear(16, 4)

    def __call__(self, latent, text):
        return latent - 0.1 * self.conv(latent) + self.text(text)[:, None, None, :]


mx.random.seed(0)
net = Denoiser()


def encode(prompt):
    codes = [ord(c) % 16 for c in prompt[:16]] + [0] * (16 - len(prompt[:16]))   # padded to 16
    return mx.array([codes], dtype=mx.float32) / 16


@backcountry.tune
def make_image(prompt, width=64, height=64, steps=3, seed=0):
    mx.random.seed(seed)
    latent = mx.random.normal((1, height // 8, width // 8, 4))
    text = encode(prompt)
    for _ in range(steps):
        latent = net(latent, text)
    return np.array(latent)


print(make_image("a lighthouse at dusk").shape)
'''

FORECAST = '''
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import backcountry


class Forecaster(nn.Module):
    def __init__(self, features=8, dim=32, horizon_max=48):
        super().__init__()
        self.inp = nn.Linear(features, dim)
        self.enc = nn.TransformerEncoder(num_layers=1, dims=dim, num_heads=4)
        self.out = nn.Linear(dim, horizon_max)

    def __call__(self, x):
        return self.out(self.enc(self.inp(x), mask=None)[:, -1])


@backcountry.tune
def forecast(model, history):
    return np.array(model(mx.array(history)))


mx.random.seed(0)
model = Forecaster()
rng = np.random.default_rng(0)
for series in (16, 4):
    print(forecast(model, rng.standard_normal((series, 48, 8)).astype(np.float32)).shape)
'''

CLASSIFY = '''
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
import backcountry


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=2)
        self.head = nn.Linear(8, 10)

    def __call__(self, x):
        return self.head(self.conv(x).mean(axis=(1, 2)))


mx.random.seed(0)
net = Net()


@backcountry.tune
def classify(images):
    batch = np.stack([np.asarray(im.resize((32, 32)), dtype=np.float32) / 255 for im in images])
    return np.array(mx.argmax(net(mx.array(batch)), axis=-1))


pictures = [Image.fromarray((np.random.default_rng(i).random((48, 64, 3)) * 255).astype("uint8")) for i in range(8)]
print(classify(pictures))
print(classify(pictures[:2]))
'''

# a model that only takes questions up to 16 tokens (a fixed position table)
SHORT_CONTEXT = '''
import mlx.core as mx
import mlx.nn as nn
import backcountry


class Tokenizer:
    def encode(self, text):
        return [len(word) for word in text.split()]

    def decode(self, ids):
        return " ".join("w" * i for i in ids)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.positions = mx.zeros((16, 16))

    def __call__(self, tokens):
        return self.embed(tokens) + self.positions[: tokens.shape[1]]


tokenizer = Tokenizer()
model = LM()


@backcountry.tune
def reply(question):
    return model(mx.array([tokenizer.encode(question)])).sum()


print(reply("what is mlx"))
'''

NEVER_CALLED = '''
import backcountry


@backcountry.tune
def reply(question):
    return question


print("this app never calls reply")
'''

BROKEN = '''
import backcountry

def reply(question)
    return question
'''

TORCH = '''
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
print(model(torch.tensor([[1, 2, 3]])))
'''

# A request function with waste in its own loop, not in the model: a rewrite
# of the function (the loop search) is what makes it faster.
WASTEFUL = '''
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import backcountry

mx.random.seed(0)
model = nn.Linear(512, 512)


@backcountry.tune
def score(rows):
    x = mx.array(rows)
    for _ in range(8):   # the same product, computed eight times over
        y = model(x)
        mx.eval(y)
    return np.array(y)


rng = np.random.default_rng(0)
rows = rng.standard_normal((64, 512)).astype(np.float32)
out = score(rows)
start = time.perf_counter()
for _ in range(20):
    score(rows)
print(f"checksum {float(np.abs(out).sum()):.4f} ms {1000 * (time.perf_counter() - start) / 20:.3f}")
'''

# The chat app as a server: the model loads at the top, nothing calls reply,
# and its loop waits for requests under `if __name__ == "__main__":`.
SERVER = CHAT.split("\n\nprint(reply(")[0] + '''


def serve():
    import time
    while True:   # waits for requests, as a server does
        time.sleep(1)


if __name__ == "__main__":
    serve()
'''

# A serving engine of the app's own: requests go to one worker thread that runs
# the model on a stream of its own (as engines do), and the handler waits for
# the text. Its handler asks the engine twice for one reply: waste a rewrite of
# the handler (the loop search) removes.
ENGINE = CHAT.split("\n\n@backcountry.tune")[0].replace(
    "model = LM()\n", "model = LM()\nmx.eval(model.parameters())   # loaded, as a real loader leaves it\n") + '''


class Engine:
    def __init__(self, model, tokenizer):
        import concurrent.futures
        self.model, self.tokenizer = model, tokenizer
        self.pool = concurrent.futures.ThreadPoolExecutor(  # its worker runs the model on a stream of its own
            max_workers=1, initializer=lambda: mx.set_default_stream(mx.new_stream(mx.gpu)))

    def _generate(self, question, max_tokens):
        tokens = mx.array([self.tokenizer.encode(question)])
        last = mx.argmax(self.model(tokens)[:, -1], axis=-1)
        out = [last]
        for _ in range(max_tokens - 1):
            last = mx.argmax(self.model(last[:, None])[:, -1], axis=-1)
            mx.eval(last)
            out.append(last)
        return self.tokenizer.decode(mx.concatenate(out).tolist())

    def submit(self, question, max_tokens):
        return self.pool.submit(self._generate, question, max_tokens)


engine = Engine(model, tokenizer)


@backcountry.tune
def handle(question, max_tokens=8):
    engine.submit(question, max_tokens).result()   # asked twice: the first reply is thrown away
    return engine.submit(question, max_tokens).result()


import time
print(handle("what is mlx"))
print(handle("write a short poem about the sea and then explain it"))
start = time.perf_counter()
for _ in range(20):
    reply_text = handle("what is mlx")
print(f"checksum {reply_text.replace(' ', '_')} ms {1000 * (time.perf_counter() - start) / 20:.3f}")
'''

# A server with a prompt cache, as engines keep one: the last question read is
# kept, and asked again it is not read again, so the same request does less
# work the second time.
CACHED = SERVER.replace(
    "@backcountry.tune\ndef reply(question, max_tokens=4):\n    tokens = mx.array([tokenizer.encode(question)])\n"
    "    last = mx.argmax(model(tokens)[:, -1], axis=-1)          # read the question\n",
    "kept = {}\n\n\n@backcountry.tune\ndef reply(question, max_tokens=4):\n    if kept.get('question') != question:\n"
    "        tokens = mx.array([tokenizer.encode(question)])\n"
    "        kept.update(question=question, last=mx.argmax(model(tokens)[:, -1], axis=-1))\n"
    "    last = kept['last']\n")
assert CACHED != SERVER
