"""A judge command for CLI tests (--judge-cmd): answers the connection check,
plans one idea for a region, then proposes nothing. A job runs every step a
real one does, with no AI and no network, and ends with no improvement."""

import json
import sys

prompt = sys.stdin.read()
if "connection check" in prompt:
    print(json.dumps({"ready": True}))
elif '"verdict"' in prompt:
    print(json.dumps({"mutations": [], "kernel": None}))
else:
    print(json.dumps({"queue": [{"id": "h1", "kind": "on-chip", "assoc_tag": "preserving",
                                 "hypothesis": "keep the chain's intermediates in registers"}]}))
