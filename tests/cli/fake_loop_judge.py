"""A judge command for the loop CLI tests (--judge-cmd): rewrites the WASTEFUL
app's score() to compute its product once and the ENGINE app's handle() to ask
its engine once (the same outputs), accepts at review, and proposes no
kernels, like fake_judge.py."""

import json
import sys

prompt = sys.stdin.read()
if "connection check" in prompt:
    print(json.dumps({"ready": True}))
elif '"candidate"' in prompt and '"current_best"' in prompt and '"setting"' in prompt:
    print(json.dumps({"verdict": "accept", "reason": "the same product, computed once"}))
elif '"default_run"' in prompt and "def handle(" in prompt:   # the ENGINE app's request handler
    print(json.dumps({"hypothesis": "the handler asks the engine twice for one reply: ask once", "kind": "loop",
                      "source": "def handle(question, max_tokens=8):\n"
                                "    return engine.submit(question, max_tokens).result()\n"}))
elif '"default_run"' in prompt:
    print(json.dumps({"hypothesis": "the loop recomputes one product: compute it once", "kind": "loop",
                      "source": "def score(rows):\n    x = mx.array(rows)\n    return np.array(model(x))\n"}))
elif '"verdict"' in prompt:
    print(json.dumps({"mutations": [], "kernel": None}))
else:
    print(json.dumps({"queue": [{"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                                 "hypothesis": "keep the chain's intermediates in registers"}]}))
