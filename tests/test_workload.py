"""Seeded workload materialization. Determinism is the load-bearing property."""

import hashlib
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest

from backcountry.harness.manifest import BOUNDARY_INPUT_SETS, InputSpec, Workload
from backcountry.harness.workload import (TEXT_POOL, bind_shape, catch_tokenizer, materialize, text_tokens,
                                workload_seeds)


def spec(shape, dtype="float32", **kw):
    return InputSpec(shape=tuple(shape), dtype=dtype, **kw)


def test_bind_shape():
    assert bind_shape((8, "L"), {"L": 512}) == (8, 512)
    assert bind_shape((4, 4), {}) == (4, 4)
    with pytest.raises(ValueError, match="no binding"):
        bind_shape((8, "L"), {"M": 2})


def test_workload_seeds_stable_and_distinct():
    a = workload_seeds(1, "decode", 3)
    assert a == workload_seeds(1, "decode", 3)
    assert len(set(a)) == 3
    assert a != workload_seeds(1, "midbatch", 3)
    assert a != workload_seeds(2, "decode", 3)


def test_materialize_shapes_and_dtypes():
    w = Workload(
        inputs=(
            spec([2, "L"], "float16"),
            spec([3], "bfloat16"),
            spec(["L", 4], "int32", low=5, high=9),
            spec([2, 2], "bool"),
        ),
        name="w",
    )
    t = materialize(w, {"L": 7}, seed=42)
    assert [x.shape for x in t] == [(2, 7), (3,), (7, 4), (2, 2)]
    assert [x.dtype for x in t] == [mx.float16, mx.bfloat16, mx.int32, mx.bool_]
    assert mx.all(t[2] >= 5).item() and mx.all(t[2] < 9).item()


def test_materialize_deterministic_in_process():
    w = Workload(inputs=(spec([16, 16]), spec([8], "int32")), name="w")
    a = materialize(w, {}, seed=7)
    b = materialize(w, {}, seed=7)
    c = materialize(w, {}, seed=8)
    assert all(mx.array_equal(x, y).item() for x, y in zip(a, b))
    assert not mx.array_equal(a[0], c[0]).item()


def _digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(bytes(memoryview(t)))
    return h.hexdigest()


def test_materialize_byte_identical_across_processes():
    """The same manifest and seed materialize byte-identical tensors in two
    processes."""
    w = Workload(inputs=(spec([32, 64]), spec([4, "L"], "int32")), name="w")
    here = _digest(materialize(w, {"L": 13}, seed=1234))
    child_src = textwrap.dedent("""
        import hashlib
        import mlx.core as mx
        from backcountry.harness.manifest import InputSpec, Workload
        from backcountry.harness.workload import materialize
        w = Workload(inputs=(
            InputSpec(shape=(32, 64), dtype="float32"),
            InputSpec(shape=(4, "L"), dtype="int32"),
        ), name="w")
        h = hashlib.sha256()
        for t in materialize(w, {"L": 13}, seed=1234):
            h.update(bytes(memoryview(t)))
        print(h.hexdigest())
    """)
    proc = subprocess.run([sys.executable, "-c", child_src], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == here


def test_k_seed_sets_differ():
    w = Workload(inputs=(spec([16, 16]),), name="w")
    seeds = workload_seeds(0, "w", BOUNDARY_INPUT_SETS)
    sets = [materialize(w, {}, s) for s in seeds]
    digests = {_digest(t) for t in sets}
    assert len(digests) == BOUNDARY_INPUT_SETS


class CharTokenizer:
    """One token per character: enough to see which text a prompt came from."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False  # prompts are text, not chat markers
        return [ord(c) for c in text]


def _pool_texts():
    import gzip, json
    with gzip.open(TEXT_POOL, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_the_text_pool_mixes_kinds_each_long_enough_for_the_longest_prompts():
    items = _pool_texts()
    chars = {}
    for item in items:
        chars[item["kind"]] = chars.get(item["kind"], 0) + len(item["text"])
    assert set(chars) == {"chat", "web", "book", "math", "code"}
    assert min(chars.values()) > 4 * 8193  # a token is several characters; jobs go to 8,193 tokens


def test_text_prompts_are_real_text_of_exact_shape_and_seeded():
    tok = CharTokenizer()
    starts = {item["text"][:40] for item in _pool_texts()}
    a = text_tokens(tok, (3, 300), mx.int32, seed=5)
    assert a.shape == (3, 300) and a.dtype == mx.int32
    assert mx.array_equal(a, text_tokens(tok, (3, 300), mx.int32, seed=5)).item()
    assert not mx.array_equal(a, text_tokens(tok, (3, 300), mx.int32, seed=6)).item()
    for row in a.tolist():
        assert "".join(map(chr, row[:40])) in starts  # each row opens with a text from the pool
    # longer than any one kind: the row continues with the other kinds
    assert text_tokens(tok, (1, 2_000_000), mx.int32, seed=1).shape == (1, 2_000_000)


class ChatTokenizer(CharTokenizer):
    """A character tokenizer with a chat template, in Gemma's shape."""
    chat_template = "gemma-like"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return f"<bos><start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n<start_of_turn>model\n"


ASKED = ("Explain what this code does.", "Summarize the text above.", "Summarize the passage above.")


def test_with_a_chat_template_every_length_is_a_request_ending_where_the_reply_starts():
    """A prompt cut mid-sentence from a web page is nothing a chat model is
    sent, and the hardest text for a drafter to guess. With a chat template,
    each row asks for a reply, and its context is cut from the front so any
    length comes out exact."""
    tok = ChatTokenizer()
    head, tail = "<bos><start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"
    pool = "\n".join(item["text"] for item in _pool_texts())
    rows = set()
    for n in (69, 70, 128, 129, 234, 3000):
        for seed in range(12):
            row = text_tokens(tok, (1, n), mx.int32, seed=seed)
            assert row.shape == (1, n)
            text = "".join(map(chr, row[0].tolist()))
            assert text.startswith(head) and text.endswith(tail)
            body = text[len(head):-len(tail)]
            # a request closes the message: an ask, or a chat's last message or a problem from the pool
            assert body[-8:] in pool + " ".join(ASKED)
            rows.add(text)
    assert len(rows) > 60  # different prompts across seeds and lengths
    a = text_tokens(tok, (2, 300), mx.int32, seed=4)
    assert mx.array_equal(a, text_tokens(tok, (2, 300), mx.int32, seed=4)).item()
    kinds = {"".join(map(chr, text_tokens(tok, (1, 400), mx.int32, seed=s)[0].tolist()))[-80:] for s in range(40)}
    assert any(k.endswith(ASKED[0] + tail) for k in kinds) and any(k.endswith(ASKED[1] + tail) for k in kinds)
    # too short to hold a request: plain text as before
    assert not "".join(map(chr, text_tokens(tok, (1, 40), mx.int32, seed=1)[0].tolist())).startswith(head)


def test_with_a_tokenizer_only_the_prompt_input_becomes_text():
    w = Workload(inputs=(spec([1, "T"], "int32", low=0, high=10), spec([4, 4])), name="w")
    plain = materialize(w, {"T": 64}, seed=3)
    text = materialize(w, {"T": 64}, seed=3, tokenizer=CharTokenizer())
    assert mx.array_equal(text[0], text_tokens(CharTokenizer(), (1, 64), mx.int32, seed=3)).item()
    assert mx.array_equal(text[1], plain[1]).item()  # every other input is as before


def test_catch_tokenizer_sees_what_mlx_lm_loads_and_restores_it(monkeypatch):
    import mlx_lm.utils as utils
    sentinel = object()
    monkeypatch.setattr(utils, "load_tokenizer", lambda *a, **k: sentinel)
    original = utils.load_tokenizer
    with catch_tokenizer() as caught:
        assert caught[0] is None
        assert utils.load_tokenizer("some/path") is sentinel
    assert caught[0] is sentinel and utils.load_tokenizer is original


def test_catch_tokenizer_takes_the_tokenizer_out_of_an_mlx_vlm_processor(monkeypatch):
    mlx_vlm_utils = pytest.importorskip("mlx_vlm.utils")
    tokenizer = object()
    monkeypatch.setattr(mlx_vlm_utils, "load_processor", lambda *a, **k: type("P", (), {"tokenizer": tokenizer})())
    with catch_tokenizer() as caught:
        mlx_vlm_utils.load_processor("some/path")
    assert caught[0] is tokenizer
