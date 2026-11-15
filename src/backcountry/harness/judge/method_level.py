"""The judge writes a block's own method around shipped kernels (live delivery).

A kernel inside a block the model hands its cache to (an attention block, a
decoder layer, a whole model) is otherwise delivered through a recording of
that block, which fits only the history, cache positions and sizes it was
recorded at. Here the judge rewrites the block's __call__ so the kernels are
called from the block's own code and the position and cache code stays live;
the harness checks the result over first and second messages and whole decode
runs, and a reviewer reads it. Replies are one JSON object each.
"""

from __future__ import annotations

import json
import re

from .model_level import _one_json, _record

SYSTEM = """You write one Python class for an optimizer for MLX models on Apple \
Silicon. Some of a block's operations were replaced by verified Metal kernels. \
Today they are delivered through a recording of the block made at one call: \
one history, one cache position, one size, so they stop applying on a second \
message, once generation moves on, or at another size. Your class makes the \
block call the kernels from its own code, so all of those work.

You get the block's own source (module_source), the source of the classes its \
code calls (parts_source), the recording with the kernels spliced in \
(reference: it shows exactly which values feed each kernel and which values \
its outputs replace, at the recorded call; a long one shows each kernel's \
first calls, and the rest repeat the pattern), and each kernel's inputs and \
outputs. Write the block's __call__ again, line for line from module_source, \
except where the reference calls a kernel: there, call the same kernel on the \
same values. A kernel that spans several of the block's parts (the end of one \
layer and the start of the next) needs those parts' code written inline, line \
for line from parts_source. Everything else, above all positions, offsets, \
masks, sizes and cache reads and writes, must stay the code's own, computed \
live on every call.

Rules:
- Define exactly one class, named class_name, subclassing ReplayWrapper, with \
only a __call__ method taking the same parameters as the block's __call__.
- The block's attributes and submodules are available as self.<name> (they \
forward to the original block); the original block itself is self.wrapped.
- Call a kernel as _kernels.try_call(self._specs["<kernel id>"], [inputs in the \
reference's order]). It returns a list of output arrays, or None when the kernel \
does not take those shapes; then compute the replaced operations exactly as \
module_source does. Never call a kernel where the reference does not.
- Names available: mx, _kernels, ReplayWrapper. Import anything else inside \
__call__ (for example the helpers module_source imports).
- Return exactly what the block returns. Keep nothing between calls, read no \
files, and change no weights or library state.

Respond with exactly one JSON object and nothing else:
{"hypothesis": "<what you changed and why it is the same computation>",
 "source": "<the complete class definition>"}"""

REVIEW = """You review one rewritten model block from an automated optimizer \
before it is installed. Verified kernels must be called from the block's own \
code in place of the operations they replace, and everything else must be the \
original block's code, computed live on every call. A harness already checked \
that its outputs match the original over first messages, second messages read \
on top of them, and whole decode runs. Reject anything \
that passes those checks without honestly doing the block's work: results kept \
between calls, shortcuts taken only for some inputs, sizes or positions, \
changed weights or global state, code that differs from the original beyond \
the kernel calls and their fallbacks.

Respond with exactly one JSON object and nothing else:
{"verdict": "accept" or "reject", "reason": "<one or two sentences>"}"""

_CLASS = re.compile(r"^class\s+(\w+)\s*\(", re.M)


def propose(judge, briefing: dict) -> dict:
    """{"hypothesis", "source"}; one re-ask carries the reason a reply was refused."""
    messages = [{"role": "user", "content": json.dumps(briefing, separators=(",", ":"))}]
    reason = None
    for _ in range(2):
        if reason:
            messages = messages[:1] + [{"role": "user", "content": json.dumps(
                {"refused": reason, "reply_again": "one JSON object with hypothesis and source"})}]
        text = judge._ask(SYSTEM, messages)
        _record(judge, "method_proposal", SYSTEM, messages, text)
        try:
            reply = _one_json(text)
            source = reply.get("source")
            if not isinstance(source, str) or not isinstance(reply.get("hypothesis"), str):
                raise ValueError("hypothesis and source must be strings")
            names = _CLASS.findall(source)
            if names != [briefing["class_name"]]:
                raise ValueError(f"source must define exactly one class named {briefing['class_name']}, "
                                 f"found {names}")
            return reply
        except ValueError as error:
            reason = str(error)
    raise ValueError(f"the judge's reply was refused twice: {reason}")


def review(judge, source: str, module_source: str) -> dict:
    messages = [{"role": "user", "content": json.dumps(
        {"original_block": module_source, "rewritten_block": source}, separators=(",", ":"))}]
    text = judge._ask(REVIEW, messages)
    _record(judge, "method_review", REVIEW, messages, text)
    reply = _one_json(text)
    if reply.get("verdict") not in ("accept", "reject"):
        raise ValueError(f"the reviewer's verdict must be accept or reject, got {reply.get('verdict')!r}")
    return reply
