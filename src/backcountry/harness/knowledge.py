"""What earlier jobs tried, for the next one.

One JSON line per evaluated kernel, appended to ~/.backcountry/knowledge.jsonl and
shared by every job: the model and its type, the region
and its shapes, the harness's measured outcome, the judge's one-line change
and the kernel's code. The judge's reflection on an attempt, written a turn
later, is its own line, joined by kernel. Nothing is edited afterwards.

Once a region's opening designs are in (they start from the region alone, so
past work does not narrow them), the judge sees an index of earlier jobs'
attempts, one short entry each, drawn at random: up to four that shipped
(right and faster in the whole model; most attempts don't, and a random draw
alone would rarely show one),
the rest from anything, the same model type first, so a lesson can cross
models and kernel types. Code is read on
request. Nothing from it is trusted: a reused kernel goes through every check
like any other.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

# about a tenth of a judge call (~80K characters): small retrieval budgets do best
INDEX_ROWS, WIN_ROWS = 12, 4
REGION_CHARS, SHAPES_CHARS, CHANGE_CHARS, REFLECTION_CHARS = 100, 120, 200, 240


def knowledge_file() -> Path:
    """Shared by every job on this Mac: $BACKCOUNTRY_KNOWLEDGE, by default
    knowledge.jsonl in backcountry's folder (~/.backcountry)."""
    from backcountry_runtime.apply import backcountry_home
    return Path(os.environ.get("BACKCOUNTRY_KNOWLEDGE") or backcountry_home() / "knowledge.jsonl")


def region_label(ops) -> str:
    """A region by its operations, in order, repeats dropped (mx.fast.rms_norm -> rms_norm)."""
    return " + ".join(dict.fromkeys(op.split(".")[-1].strip("_") for op in ops))


def shapes(io_specs: dict) -> str:
    """The first recorded instance's input and output shapes."""
    spec = next(iter(io_specs.values()), {"inputs": [], "outputs": []})
    show = lambda side: ", ".join(f"{list(shape)} {dtype}" for shape, dtype in spec[side])
    return f"in {show('inputs')}; out {show('outputs')}"


def outcome(verdict, gate, region_ms, library_ms, whole_model, sizes=None) -> str:
    """The harness's verdict in one line; nothing the judge wrote. With
    sizes (the per-size table), how it did across them; rows written before
    the search ran across sizes name one size only."""
    text = verdict + (f" at {gate}" if gate else "")
    if sizes:
        ratios = [row["ratio"] for row in sizes]
        text += (f": faster at {sum(row['result'] == 'faster' for row in sizes)} of {len(sizes)} sizes, "
                 f"slower at {sum(row['result'] == 'slower' for row in sizes)} "
                 f"({min(ratios):.2f}x to {max(ratios):.2f}x the library, "
                 f"{sizes[0]['size']} to {sizes[-1]['size']})")
    elif region_ms and library_ms:
        text += f": region {region_ms:.4g} ms vs library {library_ms:.4g} ms ({library_ms / region_ms:.2f}x)"
    if whole_model and whole_model != "not_tested":
        text += f"; whole model: {whole_model}"
    return text


def _excerpt(text, limit):
    text = text or ""
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


class Knowledge:
    def __init__(self, file: Path, job: str, setup: dict):
        self.file, self.job, self.setup = Path(file), job, setup

    def _append(self, record: dict) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.file.open("a") as f:
            f.write(json.dumps({"job": self.job, "when": when, **self.setup, **record},
                               default=str, allow_nan=False) + "\n")

    def add_attempt(self, ops, io_specs: dict, kernel: str, result: str, shipped: bool,
                    change: str | None, code: dict) -> None:
        self._append({"type": "attempt", "region": region_label(ops), "shapes": shapes(io_specs),
                      "kernel": kernel, "outcome": result, "shipped": shipped,
                      "change": change, "code": code})

    def add_reflection(self, kernel: str, text: str) -> None:
        self._append({"type": "reflection", "kernel": kernel, "text": text})

    def index(self, seed: str) -> list[dict]:
        """Earlier jobs' attempts, one short entry each; empty when there are none."""
        if not self.file.is_file():
            return []
        rows = [json.loads(line) for line in self.file.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r.get("job") != self.job]  # this job's own work is in the briefing
        reflections = {(r["job"], r["kernel"]): r["text"] for r in rows if r.get("type") == "reflection"}
        attempts = [r for r in rows if r.get("type") == "attempt" and "outcome" in r]
        rng = random.Random(seed)
        rng.shuffle(attempts)
        kind = self.setup.get("model_type")
        attempts.sort(key=lambda r: r.get("model_type") != kind)  # stable: shuffled within
        wins = [r for r in attempts if r.get("shipped")][:WIN_ROWS]
        chosen = wins + [r for r in attempts if r not in wins][:INDEX_ROWS - len(wins)]
        rng.shuffle(chosen)  # wins are not listed first
        return [{"model": r["model"], "model_type": r.get("model_type"), "chip": r.get("chip"),
                 "region": _excerpt(r["region"], REGION_CHARS), "shapes": _excerpt(r["shapes"], SHAPES_CHARS),
                 "outcome": r["outcome"],
                 "change": _excerpt(r.get("change"), CHANGE_CHARS),
                 "reflection": _excerpt(reflections.get((r["job"], r["kernel"])), REFLECTION_CHARS),
                 "code": r["code"]}
                for r in chosen]
