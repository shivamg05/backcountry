"""The judge's side of the inference-loop search (identical: output): propose a new run() for the whole model,
and a reviewer that reads each accepted one for attempts to game the check.

Both use the job's judge transport (_ask(system, messages)); replies are one
JSON object each, checked here.
"""

from __future__ import annotations

import json
import re

from .principles import MODEL_PRINCIPLES

SYSTEM = """You are the judge in an optimizer for MLX models on Apple Silicon, searching \
its inference loop. The job uses the model through one Python function, \
run(model, *inputs, helpers), which the harness calls once per use and times. You \
propose a replacement run() that makes the same outputs faster, by changing how \
the model is used. The levers, by mechanism (none is the default; pick by what \
this model's loop spends its time on):
- Compute once per call what every step or pass repeats: text-encoder outputs \
in a denoising loop, timestep embeddings, rotary or position tables.
- Batch separate model passes inside one call: the two halves of guidance, or \
several prompts or images the call receives. A bigger batch can pick different \
kernels, so it is exact only if the outputs still match.
- Overlap CPU and GPU work within the call: evaluate asynchronously, build the \
next step while the GPU runs the current one, read results back to the host \
as rarely as possible.
- Skip work whose result is thrown away: compute output logits or projections \
only for the rows that are read.
- Guess and check: guess cheaply, then verify with the model, as speculative \
decoding does with a draft model, prompt lookup, or multi-token heads where the \
loaded model has them. It pays only when most guesses are accepted and \
checking several positions costs little more than one.

The rule is identical outputs for the same inputs. For text generation every \
token must equal the original model's plain greedy decoding, exactly, at any \
conversation and reply length (the final check covers several). Calls that read \
several generated tokens at once (checking guesses, re-reading accepted tokens) \
are computed by the harness with one-token arithmetic per token, so a correct \
guess-and-check loop emits exactly plain decoding's tokens; exact_rows in the \
briefing gives what such a call costs on this model, per number of tokens, and \
whether it holds here. Pick tokens as the default does, read the prompt as it \
does, and read tokens with the model's own layers. For other outputs the values \
must match within the job's numeric tolerance. No approximate methods: no lossy \
caches, no skipped or fewer steps, no lower precision, no changed weights, \
nothing carried from one call to the next, nothing downloaded. Every call must \
do its own work on its own inputs. The harness checks every candidate on fresh \
inputs, times it against the current best on more fresh inputs (and checks the \
timed outputs too), and a reviewer reads every accepted candidate for anything \
that games the check; a gamed candidate is rolled back.

You see the model file, the current best run() (at first the job's default), \
the model's own source, the helpers available, the hardware, principles \
checked against their sources (model_principles), and every earlier attempt \
with its verdict. Use them. Make one coherent change per attempt, building on \
the current best (changes stack) or on an earlier attempt when evidence \
favors it. State the mechanism and the expected effect in hypothesis.

run() receives the model exactly as the job loads it (kernels the job already \
installed are part of it) and helpers, a dict described in the briefing. It \
must return outputs of the same structure as the default's. The source may also \
define setup(model, helpers) -> dict, run once before any timing to load what \
the loop uses (a drafter or multi-token head among helpers['assets']; its \
entries join helpers), and, when the job continues a conversation, \
read(model, tokens, helpers) -> dict, which reads the conversation into memory \
before the clock in place of the default read: its dict must hold "cache" (the \
model's own, every token but the last) and anything else the loop keeps (a \
head's memory of the conversation); every run gets a fresh copy, and any time \
it takes beyond the default read is added to the loop's timed time. Keep one \
copy of the weights: a merged or re-laid-out weight must replace the model's own \
arrays (with views of the merged array, say), not sit beside them; a loop that \
keeps more than 1% of the model's weights in memory beyond any model it loads \
fails its check. Import what you need inside the source. Do not use tools, read local files, or run code; \
judge only from this briefing.

Respond with exactly one JSON object and nothing else:
{"hypothesis": "<mechanism and expected effect>", "kind": "<short label>",
 "source": "<the complete Python source defining run(model, *inputs, helpers), and setup/read if used>"}"""

# MLX-LM's generation: the loop is generate_step itself, with MLX-LM's
# signature; the levers and the honesty rules are the same as for run().
GENERATION_SYSTEM = SYSTEM.split("The levers, by mechanism")[0].replace(
    "The job uses the model through one Python function, run(model, *inputs, helpers), which the "
    "harness calls once per use and times. You propose a replacement run() that makes the same outputs "
    "faster, by changing how the model is used. ",
    "") + """MLX-LM generates text through one function, \
generate_step(prompt, model, *, max_tokens, sampler, logits_processors, \
max_kv_size, prompt_cache, prefill_step_size, kv_bits, kv_group_size, \
quantized_kv_start, prompt_progress_callback, input_embeddings), a generator \
yielding (token, logprobs) per step; mlx_lm.generate, stream_generate and \
mlx_lm.server all call it. You propose a replacement generate_step with the same \
signature that yields the same thing faster, by changing how the model is used. \
The levers, by mechanism""" + SYSTEM.split("The levers, by mechanism")[1].split("The rule is identical")[0] + \
"""The rule is the same results for the same call, whatever the caller passes. \
Every step's token and logprobs must be bit-identical to MLX-LM's own \
generate_step's for any sampler and logits processors, with or without a \
prompt_cache already holding a conversation (continue from it as MLX-LM \
does), and at any prefill_step_size; prompt_progress_callback must be called as \
MLX-LM calls it, and the prompt cache must end as MLX-LM leaves it (it also runs \
the step after the last token it yields). The harness checks this under greedy \
decoding, a seeded temperature/top-p sampler (the random draws must be taken as \
MLX-LM takes them), a repetition penalty and a small prefill_step_size, at several \
conversation and reply lengths, and on prompts read from no cache; a call that \
sets max_kv_size, kv_bits, quantized_kv_start or input_embeddings runs MLX-LM's \
own instead. Every candidate is timed with greedy passed as a sampler function \
(lambda logprobs: mx.argmax(logprobs, axis=-1)), as mlx-lm's server and engines \
such as LM Studio's pass it even at temperature 0: a loop that is faster only \
when sampler is None gains nothing there. Guess and check stays exact under any \
sampler when each position's token is drawn by calling sampler on that \
position's logprobs, in order, and only for the positions reached, as MLX-LM's \
one-token loop draws them. \
Calls that read several generated tokens at once are computed by the harness \
with one-token arithmetic per token; exact_rows in the briefing gives what such \
a call costs on this model, per number of tokens, and whether it holds here. \
No approximate methods: no lossy caches, no skipped or fewer steps, no lower \
precision, no changed weights, nothing carried from one call to the next, \
nothing downloaded. Every call must do its own work on its own inputs. The \
harness checks every candidate on fresh inputs, times it against the current \
best on more fresh inputs (and checks the timed outputs too), and a reviewer \
reads every accepted candidate for anything that games the check; a gamed \
candidate is rolled back.

You see the model file, the current best generate_step (at first MLX-LM's own, \
source in default_run), the model's own source, the helpers available, the \
hardware, principles checked against their sources (model_principles), and \
every earlier attempt with its verdict. Use them. Make one coherent change per \
attempt, building on the current best (changes stack) or on an earlier attempt \
when evidence favors it. State the mechanism and the expected effect in \
hypothesis.

generate_step receives the model exactly as the job loads it (kernels the job \
already installed are part of it). The source may also define \
setup(model, helpers) -> dict, run once per model before its first call, to \
load what the loop uses (a drafter or multi-token head among helpers['assets']) \
or prepare what every call reuses; declare a helpers keyword argument on \
generate_step to receive its entries. There is no read() hook: prompt holds \
only this call's new tokens, and any earlier conversation exists only in \
prompt_cache (the model's own cache, no hidden states). A drafter or head that \
should see the context reads the prompt inside generate_step, as part of the \
call's work; nothing it builds may be kept for the next call. Import what you need inside the source. \
Do not use tools, read local files, or run code; judge only from this briefing.

Respond with exactly one JSON object and nothing else:
{"hypothesis": "<mechanism and expected effect>", "kind": "<short label>",
 "source": "<the complete Python source defining generate_step(prompt, model, ...), and setup if used>"}"""

REVIEW = """You review one candidate inference loop from an automated optimizer \
before it is accepted. The optimizer's rule: the candidate must produce the \
same outputs as the original for the same inputs, doing its own work on every \
call. A harness checks outputs on fresh inputs and times it against the \
current best on more fresh inputs. Your job is to catch candidates that pass \
those checks without honestly doing the work, for example: \
returning or reusing results saved from an earlier call; caching anything \
keyed by the inputs across calls; detecting benchmark-like inputs, sizes or \
call patterns and taking a shortcut only then; touching timers, the harness, \
the clock, threads or processes that keep working after run() returns; \
changing model weights or global library state; loading or downloading \
anything beyond helpers['assets']; doing work in setup() or read() that depends \
on what the loop will be asked next; approximate math presented as exact \
(skipped steps, lower precision, lossy caches, unverified drafts). \
A legitimate optimization that is merely clever (speculative decoding with a \
proper check, computing step-independent values once per call, overlapping \
CPU and GPU work) is fine.

Respond with exactly one JSON object and nothing else:
{"verdict": "accept" or "reject", "reason": "<one or two sentences>"}"""


def principles() -> list[dict]:
    return [{"topic": t, "principle": p, "sources": s} for t, p, s in MODEL_PRINCIPLES]


def _one_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        raise ValueError("the reply holds no JSON object")
    reply = json.loads(match.group(0))
    if not isinstance(reply, dict):
        raise ValueError("the reply must be one JSON object")
    return reply


# Any function the application calls the model through (an engine's own
# generation: LM Studio's create_generator, mlx-vlm's stream_generate, an image
# pipeline's method): the same search and rules, aimed at that function.
FUNCTION_SYSTEM = (
    GENERATION_SYSTEM.split("MLX-LM generates text through one function")[0]
    + """The application calls the model through one function, named in the \
briefing (job.target, "module:qualname"); default_run holds its source and the \
sources of what it calls. You propose a replacement with the same signature \
that produces the same outputs faster, by changing how the model is used. The \
harness calls it the way the application does (the model file's run() is that \
recipe, in model_file), checks every output against the original's, and times \
it. The levers, by mechanism"""
    + GENERATION_SYSTEM.split("The levers, by mechanism")[1].split("The rule is the same results")[0]
    + """The rule is the same results for the same call: every output the \
function returns or yields must be bit-identical to the original's, and \
anything it leaves in the objects it was handed (an engine's caches) must let \
the next call continue as the original's would; the harness checks this on \
several inputs, under the sampling the recipe uses. Engines pass a sampler \
function even for greedy decoding: guess and check stays exact under any \
sampler when each position's token is drawn by calling the sampler on that \
position's logprobs, in order, and only for the positions reached, as a \
one-token loop draws them. A model call that reads several tokens at once is \
computed by MLX with other kernels than one-token calls, so its numbers differ \
in the last bits and near-ties flip tokens: make every call that checks \
guessed tokens inside `with checking():` (from backcountry_runtime.exact_rows \
import checking), on whatever thread runs it, and the harness computes it with \
one-token arithmetic per token, equal to plain decoding. Leave the prompt read \
outside it, done as the original does. exact_rows in the briefing gives what \
such a call costs on this model and whether it holds here. No approximate methods: \
no lossy caches, no skipped or fewer steps, no lower precision, no changed \
weights, nothing carried from one call to the next, nothing downloaded. Every \
call must do its own work on its own inputs. The harness checks every \
candidate on fresh inputs, times it against the current best on more fresh \
inputs (and checks the timed outputs too), and a reviewer reads every accepted \
candidate for anything that games the check; a gamed candidate is rolled back.

You see the model file, the current best function (at first the original, \
source in default_run), the model's own source, the helpers available, the \
hardware, principles checked against their sources (model_principles), and \
every earlier attempt with its verdict. Use them. Make one coherent change per \
attempt, building on the current best (changes stack) or on an earlier attempt \
when evidence favors it. State the mechanism and the expected effect in \
hypothesis.

The function receives what the application passes (an engine's kit holds the \
model; kernels the job already installed are part of it). The source starts \
with the line "# target: module:qualname" from default_run and defines the \
function under its own name; it may also define setup(model, helpers) -> dict, \
run once per model before its first call, to load what the loop uses (a \
drafter or multi-token head among helpers['assets']) or prepare what every \
call reuses; declare a helpers keyword argument on the function to receive its \
entries. Import what you need inside the source. Do not use tools, read local \
files, or run code; judge only from this briefing.

Respond with exactly one JSON object and nothing else:
{"hypothesis": "<mechanism and expected effect>", "kind": "<short label>",
 "source": "<the complete Python source: the target line, the function, and setup if used>"}""")


def propose(judge, briefing: dict, generation: bool = False, target: str | None = None) -> dict:
    """One proposal: {"hypothesis", "kind", "source"}. One re-ask carries the
    reason a malformed reply was refused. generation: the loop is MLX-LM's
    generate_step (GENERATION_SYSTEM); target: a named function the app calls
    (FUNCTION_SYSTEM); else a run()."""
    system = FUNCTION_SYSTEM if target else GENERATION_SYSTEM if generation else SYSTEM
    name = target.partition(":")[2].split(".")[-1] if target else None
    want = f"def {name}(" if target else "def generate_step(" if generation else "def run("
    messages = [{"role": "user", "content": json.dumps(briefing, separators=(",", ":"))}]
    reason = None
    for _ in range(2):
        if reason:
            messages = messages[:1] + [{"role": "user", "content": json.dumps(
                {"refused": reason, "reply_again": "one JSON object with hypothesis, kind and source"})}]
        text = judge._ask(system, messages)
        _record(judge, "model_proposal", system, messages, text)
        try:
            reply = _one_json(text)
            if not all(isinstance(reply.get(k), str) and reply[k].strip() for k in ("hypothesis", "kind", "source")):
                raise ValueError("hypothesis, kind and source must be non-empty strings")
            if want not in reply["source"]:
                raise ValueError("source must define " + (f"{name}, as {target} is defined" if target else
                                                          "generate_step(prompt, model, ...) with MLX-LM's "
                                                          "signature" if generation else "run(model, *inputs, helpers)"))
            return reply
        except ValueError as error:
            reason = str(error)
    raise ValueError(f"the judge's reply was refused twice: {reason}")


def review(judge, candidate: str, current: str, setting: dict) -> dict:
    """The reviewer's verdict on an accepted candidate: {"verdict", "reason"}."""
    messages = [{"role": "user", "content": json.dumps(
        {"setting": setting, "current_best": current, "candidate": candidate}, separators=(",", ":"))}]
    text = judge._ask(REVIEW, messages)
    _record(judge, "model_review", REVIEW, messages, text)
    reply = _one_json(text)
    if reply.get("verdict") not in ("accept", "reject"):
        raise ValueError(f"the reviewer's verdict must be accept or reject, got {reply.get('verdict')!r}")
    return reply


def _record(judge, call: str, system: str, messages: list, reply: str) -> None:
    record = getattr(judge, "_record_event", None)
    if record is not None and getattr(judge, "transcript", None) is not None:
        record("reply", call, 0, system=system, messages=messages, reply=reply)
