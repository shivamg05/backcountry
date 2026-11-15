"""An engine of its own, shaped as LM Studio's mlx-engine is: it keeps the
model in a kit, its generation function takes the kit and token ids and
yields tokens, and the model runs on the engine's worker thread, with greedy
decoding passed as a sampler function."""
import queue
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx


class Kit:
    def __init__(self, model):
        self.model = model
        self.worker = ThreadPoolExecutor(max_workers=1)


def greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


def create_generator(kit, prompt_tokens, *, max_tokens=8, temp=0.0):
    """Token ids, one at a time, generated on the kit's worker thread."""
    from mlx_lm.generate import generate_step
    out = queue.Queue()

    def work():
        try:
            for token, _ in generate_step(mx.array(prompt_tokens), kit.model, max_tokens=max_tokens, sampler=greedy):
                out.put(token)
        finally:
            out.put(None)
    kit.worker.submit(work)
    while (token := out.get()) is not None:
        yield token
