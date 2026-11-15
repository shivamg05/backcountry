"""Exact rows: inside generating(), a model call over several new tokens gives
every token the numbers its own one-token call gives, bit for bit, logits and
cache; calls that start before the generated part, and one-token calls, are
left as they are."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import llama
from mlx_lm.models.cache import make_prompt_cache

from backcountry_runtime.exact_rows import generating


def tiny(quantize=True, dtype=None, tied=False):
    """dtype=mx.bfloat16: as real 4-bit checkpoints run, where the multi-row
    kernel serves; in float32 the switch runs one call per row."""
    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs(model_type="llama", hidden_size=512, num_hidden_layers=2,
                                        intermediate_size=1024, num_attention_heads=8, num_key_value_heads=2,
                                        rms_norm_eps=1e-5, vocab_size=512, tie_word_embeddings=tied))
    if dtype is not None:
        model.set_dtype(dtype)
    if quantize:
        nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def cache_arrays(cache):
    return [a for c in cache for a in c.state]


def read(model, prompt):
    cache = make_prompt_cache(model)
    mx.eval(model(prompt[None], cache=cache), [c.state for c in cache])
    return cache


@pytest.mark.parametrize("quantize,dtype", [(True, None), (True, mx.bfloat16), (False, None)])
@pytest.mark.parametrize("w", [2, 3, 5, 9])
@pytest.mark.parametrize("tied", [False, True])  # an output layer that is the input embedding (LFM2, Gemma)
def test_several_new_tokens_match_one_token_calls_bit_for_bit(quantize, dtype, w, tied):
    model = tiny(quantize, dtype, tied)
    prompt, new = mx.arange(40) % 512, (mx.arange(w) * 7 + 3) % 512
    one = read(model, prompt)
    expected = mx.concatenate([model(new[None, i:i + 1], cache=one) for i in range(w)], axis=1)
    stock = model(new[None], cache=read(model, prompt))
    if dtype is None:  # the control: here stock MLX differs at every width (bf16 4-bit can coincide)
        assert not mx.array_equal(stock, expected).item()
    several = read(model, prompt)
    with generating(model, start=len(prompt)):
        got = model(new[None], cache=several)
    mx.eval(expected, got)
    assert mx.array_equal(got, expected).item()
    assert all(mx.array_equal(a, b).item() for a, b in zip(cache_arrays(several), cache_arrays(one)))


def test_calls_before_the_generated_part_and_outside_it_are_left_alone():
    model = tiny()
    prompt = mx.arange(40) % 512
    ref = model(prompt[None, :6], cache=make_prompt_cache(model))
    with generating(model, start=len(prompt)):
        got = model(prompt[None, :6], cache=make_prompt_cache(model))  # a prompt read: starts at 0
    assert mx.array_equal(got, ref).item()
    linear, sdpa = nn.QuantizedLinear.__dict__["__call__"], mx.fast.scaled_dot_product_attention
    with generating(model, start=0):
        assert nn.QuantizedLinear.__dict__["__call__"] is not linear
    assert nn.QuantizedLinear.__dict__["__call__"] is linear  # nothing left patched
    assert mx.fast.scaled_dot_product_attention is sdpa


def test_a_model_used_under_the_switch_can_still_be_freed():
    """The switch remembers each model's layers by id: a job that drops a
    model (the loop search's, before the kernel job loads its own) frees it."""
    import gc
    import weakref
    model = tiny()
    cache = make_prompt_cache(model)
    with generating(model, start=0):
        mx.eval(model(mx.array([[1, 2]]), cache=cache))
    gone = weakref.ref(model.model)
    del model, cache
    gc.collect()
    assert gone() is None


def test_the_switch_holds_on_the_thread_that_calls_the_model():
    """Engines (LM Studio's mlx-engine, oMLX) call the model on a worker thread
    while the generation function that turned the switch on runs elsewhere:
    a several-token call there must still match one-token calls."""
    import threading
    model = tiny(True, mx.bfloat16)
    prompt, new = mx.arange(40) % 512, (mx.arange(3) * 7 + 3) % 512
    one = read(model, prompt)
    expected = mx.concatenate([model(new[None, i:i + 1], cache=one) for i in range(3)], axis=1)
    mx.eval(expected)
    several, got = read(model, prompt), {}

    def worker():
        got["logits"] = model(new[None], cache=several)
        mx.eval(got["logits"])
    with generating(model, start=len(prompt)):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert mx.array_equal(got["logits"], expected).item()


def test_a_loop_marks_its_guess_checks_and_only_those_are_exact():
    """A loop rewriting an engine's function marks its guess checks with
    checking(): there, on the engine's worker thread, a several-token call
    equals one-token calls; the same call outside it (the engine's own prompt
    read) is left as MLX computes it."""
    import threading
    from backcountry_runtime.exact_rows import checking
    model = tiny(True, None)
    prompt, new = mx.arange(40) % 512, (mx.arange(3) * 7 + 3) % 512
    one = read(model, prompt)
    expected = mx.concatenate([model(new[None, i:i + 1], cache=one) for i in range(3)], axis=1)
    stock = model(new[None], cache=read(model, prompt))
    mx.eval(expected, stock)
    got = {}

    def worker(name, marked):
        cache = read(model, prompt)
        if marked:
            with checking():
                got[name] = model(new[None], cache=cache)
        else:
            got[name] = model(new[None], cache=cache)
        mx.eval(got[name])
    with generating(model, start=1 << 62):  # installed, the automatic rule off
        for name, marked in (("checked", True), ("unmarked", False)):
            thread = threading.Thread(target=worker, args=(name, marked))
            thread.start()
            thread.join()
    assert mx.array_equal(got["checked"], expected).item()
    assert mx.array_equal(got["unmarked"], stock).item()
