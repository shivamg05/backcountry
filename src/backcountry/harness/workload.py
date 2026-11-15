"""Workload tensors: bind named dims and materialize seeded inputs.

The harness fixes the synthesis seeds (derived from the job seed, recorded);
the manifest never supplies them. A language model's prompt is real text cut
from a shipped pool when the harness has the model's tokenizer, as a chat
request in the model's own chat format when it has one. The same manifest and seed must materialize
byte-identical tensors in any process, which a pinned test proves.
"""

from __future__ import annotations

import contextlib
import functools
import gzip
import hashlib
import importlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Mapping

import mlx.core as mx

from .manifest import FLOAT_DTYPES, INT_DTYPES, InputSpec, Manifest, Workload, resolve_dtype


def bind_shape(shape: tuple[int | str, ...], dims: Mapping[str, int]) -> tuple[int, ...]:
    """Replace named dims with concrete sizes; integer dims pass through."""
    bound: list[int] = []
    for d in shape:
        if isinstance(d, str):
            if d not in dims:
                raise ValueError(f"named dim {d!r} has no binding in {dict(dims)}")
            bound.append(dims[d])
        else:
            bound.append(d)
    return tuple(bound)


def workload_seeds(job_seed: int, workload_name: str, k: int) -> list[int]:
    """k stable per-workload seeds. Hash-derived so workload order never matters."""
    seeds = []
    for i in range(k):
        digest = hashlib.sha256(f"{job_seed}:{workload_name}:{i}".encode()).digest()
        seeds.append(int.from_bytes(digest[:8], "big"))
    return seeds


def _materialize_input(spec: InputSpec, shape: tuple[int, ...], key: mx.array) -> mx.array:
    dtype = resolve_dtype(spec.dtype)
    if spec.dtype in FLOAT_DTYPES:
        return mx.random.normal(shape, dtype=dtype, key=key)
    if spec.dtype in INT_DTYPES:
        return mx.random.randint(spec.low, spec.high, shape, dtype=dtype, key=key)
    if spec.dtype == "bool":
        return mx.random.randint(0, 2, shape, key=key).astype(mx.bool_)
    raise ValueError(f"cannot synthesize dtype {spec.dtype!r}")


TEXT_POOL = Path(__file__).parent / "inputs" / "text.jsonl.gz"  # sources in inputs/SOURCES.md


@contextlib.contextmanager
def catch_tokenizer():
    """The tokenizer a model file's build() loads through mlx-lm or mlx-vlm,
    caught as the library makes it: caught[0] after the block, None when
    there was none."""
    caught = [None]
    patched = []
    for module, name in (("mlx_lm.utils", "load_tokenizer"), ("mlx_vlm.utils", "load_processor")):
        try:
            utils = importlib.import_module(module)
        except ImportError:
            continue
        original = getattr(utils, name, None)
        if original is None:
            continue

        def load(*args, _original=original, **kwargs):
            made = _original(*args, **kwargs)
            caught[0] = getattr(made, "tokenizer", made)  # mlx-vlm's processor holds one
            return made

        setattr(utils, name, load)
        patched.append((utils, name, original))
    try:
        yield caught
    finally:
        for utils, name, original in patched:
            setattr(utils, name, original)


@functools.lru_cache(maxsize=1)
def _text_pool() -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    with gzip.open(TEXT_POOL, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pool.setdefault(item["kind"], []).append(item["text"])
    return pool


@functools.lru_cache(maxsize=2)
def _encoded_pool(tokenizer) -> dict[str, list[list[int]]]:
    return {kind: [tokenizer.encode(text, add_special_tokens=False) for text in texts]
            for kind, texts in _text_pool().items()}


# what a user asks of each kind of text; chats and math problems carry their own
ASKS = {"code": "Explain what this code does.", "web": "Summarize the text above.",
        "book": "Summarize the passage above."}


def _ask(kind: str, text: str) -> tuple[str, str]:
    """(context, request): one pool text as a chat message asks it."""
    if kind == "chat":  # "User: ...\n\nAssistant: ..." turns: reply to the last user message
        context, _, last = text.rpartition("User: ")
        return context, last.split("\n\nAssistant: ")[0]
    if kind == "math":  # a problem, then its worked answer
        return "", text.split("\n\n")[0]
    return text, ASKS.get(kind, "Summarize the text above.")


@functools.lru_cache(maxsize=2)
def _chat_frame(tokenizer) -> tuple[list[int], list[int]] | None:
    """The tokens before and after one user message in the tokenizer's chat
    format, ending where the model's reply starts; None without a template."""
    if not getattr(tokenizer, "chat_template", None):
        return None
    marker = "\x00message\x00"
    text = tokenizer.apply_chat_template([{"role": "user", "content": marker}],
                                         add_generation_prompt=True, tokenize=False)
    before, after = text.split(marker)
    return (tokenizer.encode(before, add_special_tokens=False),
            tokenizer.encode(after, add_special_tokens=False))


def text_tokens(tokenizer, shape: tuple[int, ...], dtype, seed: int) -> mx.array:
    """Real text as token ids of exactly this shape. Each row is one kind of
    text (conversations, web pages, books, math problems or code) chosen by
    seed. With a chat template, a row is a user message in that format that
    ends where the model's reply starts: a request (reply to the chat's last
    message, solve the problem, explain the code, summarize the page) after
    its context, cut from the front so the row is exactly full, the way a
    chat app drops the oldest history. Without one, or when a row is too
    short to hold a request, the kind's texts are joined in a seeded order
    until the row is full; a kind shorter than the row continues with the
    others, and the whole pool repeats."""
    pool = _encoded_pool(tokenizer)
    kinds = sorted(pool)
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    frame = _chat_frame(tokenizer)
    n = shape[-1]
    rng = random.Random(seed)
    rows = []
    for _ in range(math.prod(shape[:-1])):
        kind = rng.choice(kinds)
        others = [t for k in kinds if k != kind for t in pool[k]]
        row: list[int] = []
        order = rng.sample(pool[kind], len(pool[kind])) + rng.sample(others, len(others))
        if frame is not None and n > len(frame[0]) + len(frame[1]) + 8:
            before, after = frame
            context, request = _ask(kind, rng.choice(_text_pool()[kind]))
            ask = tokenizer.encode(request, add_special_tokens=False)[-(n - len(before) - len(after)):]
            room = n - len(before) - len(after) - len(ask)
            filler = tokenizer.encode(context, add_special_tokens=False) + separator if context else []
            for text in itertools.cycle(order):  # earlier texts in front, as long history
                if len(filler) >= room:
                    break
                filler = text + separator + filler
            rows.append(before + (filler[len(filler) - room:] if room else []) + ask + after)
            continue
        for text in itertools.cycle(order):  # past the whole pool, it starts over
            row += text + separator
            if len(row) >= n:
                break
        rows.append(row[:n])
    return mx.array(rows, dtype=dtype).reshape(shape)


def context_tokens(workload: Workload, dims: Mapping[str, int], seed: int, tokenizer=None) -> mx.array:
    """The tokens already in the conversation for a workload with a context:
    its token input with the sequence (last) dim set to the context."""
    spec = workload.inputs[0]
    shape = bind_shape(spec.shape, dims)[:-1] + (workload.context,)
    tokens = (text_tokens(tokenizer, shape, resolve_dtype(spec.dtype), seed) if tokenizer is not None
              else _materialize_input(spec, shape, mx.random.key(seed)))
    mx.eval(tokens)
    return tokens


def materialize(workload: Workload, dims: Mapping[str, int], seed: int, tokenizer=None) -> list[mx.array]:
    """One tensor per positional argument of the model, deterministically from
    seed. Given the model's tokenizer, the first input is the prompt: real text."""
    keys = mx.random.split(mx.random.key(seed), len(workload.inputs))
    tensors = []
    for i, spec in enumerate(workload.inputs):
        shape = bind_shape(spec.shape, dims)
        if i == 0 and tokenizer is not None:
            tensors.append(text_tokens(tokenizer, shape, resolve_dtype(spec.dtype), seed))
        else:
            tensors.append(_materialize_input(spec, shape, keys[i]))
    mx.eval(tensors)
    return tensors
