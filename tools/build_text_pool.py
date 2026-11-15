"""Build src/backcountry/harness/inputs/text.jsonl.gz, the real text prompts are cut from.

Five kinds of text so no single corpus sets the token mix: chat in many
languages (OpenAssistant oasst2), web pages (C4), public-domain books (PG-19),
math word problems (GSM8K) and code (MLX's C++ headers and mlx-lm's Python,
from the installed packages). Licenses are in src/backcountry/harness/inputs/SOURCES.md.

The four datasets come as rows from Hugging Face's dataset viewer, saved as
JSON pages in one folder:

  rows?dataset=OpenAssistant/oasst2&config=default&split=validation  (offsets 0..700) -> oasst2_<offset>.json
  rows?dataset=allenai/c4&config=en&split=validation&length=100                     -> c4.json
  rows?dataset=openai/gsm8k&config=main&split=test&length=100                       -> gsm8k.json
  rows?dataset=emozilla/pg19&config=default&split=validation (offsets 0, 4, 8)       -> pg19_<offset>.json

Usage: python tools/build_text_pool.py <folder of those pages>
"""

import glob
import gzip
import json
import random
import sys
from pathlib import Path

import mlx
import mlx_lm

ITEM_CHARS = 30_000  # one book excerpt or code file; long enough for many tokens, short enough to vary
OUT = Path(__file__).resolve().parent.parent / "src" / "backcountry" / "harness" / "inputs" / "text.jsonl.gz"


def rows(folder, pattern):
    for path in sorted(glob.glob(str(Path(folder) / pattern))):
        yield from (r["row"] for r in json.load(open(path))["rows"])


def chats(folder):
    """Each conversation as it went: the opening message, then the best-ranked
    reply at every turn."""
    messages = [m for m in rows(folder, "oasst2_*.json") if not m["deleted"]]
    children = {}
    for m in messages:
        children.setdefault(m["parent_id"], []).append(m)
    for root in children.get(None, []):
        turns, m = [], root
        while m is not None:
            turns.append(f"{'User' if m['role'] == 'prompter' else 'Assistant'}: {m['text'].strip()}")
            replies = children.get(m["message_id"], [])
            m = min(replies, key=lambda r: 99 if r["rank"] is None else r["rank"], default=None)
        yield "\n\n".join(turns)


def books(folder):
    """Two excerpts per book from past its first tenth, where the
    transcriber's front matter has ended."""
    for book in rows(folder, "pg19_*.json"):
        text = book["text"]
        for start in (len(text) // 10, len(text) // 2):
            yield text[start:start + ITEM_CHARS].split("\n", 1)[-1]  # from a line start


def code():
    roots = [Path(mlx.__path__[0]) / "include" / "mlx", Path(mlx_lm.__file__).parent]
    files = sorted(p for root in roots for p in root.rglob("*") if p.suffix in (".h", ".py"))
    for p in random.Random(0).sample(files, 60):
        text = p.read_text(errors="replace")
        if len(text) > 2000:
            yield text[:ITEM_CHARS]


def main(folder):
    items = (
        [("chat", t) for t in chats(folder)]
        + [("web", r["text"]) for r in rows(folder, "c4.json")]
        + [("book", t) for t in books(folder)]
        + [("math", f"{r['question']}\n\n{r['answer']}") for r in rows(folder, "gsm8k.json")]
        + [("code", t) for t in code()]
    )
    OUT.parent.mkdir(exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for kind, text in items:
            f.write(json.dumps({"kind": kind, "text": text}, ensure_ascii=False) + "\n")
    counts = {k: sum(1 for kind, _ in items if kind == k) for k in dict.fromkeys(k for k, _ in items)}
    chars = {k: sum(len(t) for kind, t in items if kind == k) for k in counts}
    print(OUT, OUT.stat().st_size, "bytes;", {k: (counts[k], chars[k]) for k in counts})


if __name__ == "__main__":
    main(sys.argv[1])
