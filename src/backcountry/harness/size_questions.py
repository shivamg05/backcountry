"""The size questions for a marked function, one way for every model.

init probes each knob of the function's first call (backcountry_runtime.sizes,
tune.Calls) and gets its effect: size (the model's shapes change), more (the
model runs more or fewer times at the same shapes), same or fails. Knobs whose
probes changed the model's work the same way move together (an image's width
and height) and are asked as one question. The answers are lists (1, 8, 32),
ranges (32-4096, spread into a few sizes) or tuples (512x512, 1024x768); the
sizes to search are every combination, a handful at most, since each one is
checked and timed.
"""

from __future__ import annotations

import itertools
import json
import math

MAX_SIZES = 8      # every size is replayed, checked and timed in the search
RANGE_SIZES = 5    # a range is spread into this many sizes, its ends included

UNITS = {"int": "", "length": " (count)", "axis": " (length)", "image": " (width x height)"}


class SizeError(ValueError):
    pass


def shown(value, unit=None) -> str:
    """A size as the user writes it: 512x512, 5 tokens, 16."""
    text = "x".join(map(str, value)) if isinstance(value, list) else str(value)
    return f"{text} {unit}" if unit else text


def unit(knob) -> str:
    return f" ({knob['unit']})" if knob["kind"] == "text" else UNITS[knob["kind"]]


def sort(knobs):
    """(groups of knobs that change the size, knobs that change how many
    times the model runs, knobs that change nothing)."""
    groups = {}
    for knob in knobs:
        if knob.get("effect") == "size":
            # numbers that change the same parts of the model go together
            # (an image's width and height); any knob with the very same
            # effect too; lists, text and axes otherwise stay their own question
            key = ("int", json.dumps(knob.get("parts"))) if knob["kind"] == "int" and knob.get("parts") \
                else ("same", json.dumps(knob["signature"]))
            groups.setdefault(key, []).append(knob)
    more = [k for k in knobs if k.get("effect") == "more"]
    unchanged = [k for k in knobs if k.get("effect") in ("same", "fails")]
    return list(groups.values()), more, unchanged


def plan(sizes: list[dict], knobs: list | None = None) -> str:
    """The sizes as one line, each question's values once: rows: 256, 512;
    width x height: 512x512, 768x768."""
    together = [g for g in sort(knobs)[0] if len(g) > 1] if knobs else []
    names = [[k["name"] for k in g] for g in together]
    grouped = {n for g in names for n in g}
    values = {}
    for setting in sizes:
        for group in names:
            text = "x".join(str(setting[n]) for n in group if n in setting)
            if text and text not in values.setdefault(" x ".join(group), []):
                values[" x ".join(group)].append(text)
        for name, value in setting.items():
            if name in grouped:
                continue
            text = "x".join(map(str, value)) if isinstance(value, list) else str(value)
            if text not in values.setdefault(name, []):
                values[name].append(text)
    units = {k["name"]: k["unit"] for k in knobs or [] if k.get("unit")}
    return "; ".join(f"{name}: {', '.join(text)}{' ' + units[name] if name in units else ''}"
                     for name, text in values.items())


def _width(group) -> int:
    """Values per answer item: one per knob, two for an image."""
    return sum(2 if k["kind"] == "image" else 1 for k in group)


def label(group) -> str:
    names = " and ".join(k["name"] for k in group)
    return names + (" (together)" if len(group) > 1 else unit(group[0]))


def default(group, recorded: list[dict] | None = None) -> str:
    """The sizes the app called with: each recorded call's values for the
    group ({knob name: value}, one dict per call), smallest first; the
    probed call's own without a record."""
    def one(values):
        flat = [v for value in values for v in (value if isinstance(value, list) else [value])]
        return "x".join(str(v) for v in flat)
    names = [k["name"] for k in group]
    seen = [tuple(map(json.dumps, (call[n] for n in names))) for call in recorded or [] if all(n in call for n in names)]
    if not seen:
        return one(k["value"] for k in group)
    distinct = sorted(set(seen), key=lambda t: [json.loads(v) for v in t])
    return ", ".join(one(json.loads(v) for v in t) for t in distinct)


def defaults(groups, recorded: list[dict] | None = None) -> list[str]:
    """Each group's default, trimmed to the smallest and largest, then the
    first call's, while their combinations are more than MAX_SIZES."""
    answers = [default(g, recorded) for g in groups]
    for i in reversed(range(len(answers))):
        if math.prod(len(a.split(",")) for a in answers) <= MAX_SIZES:
            break
        items = answers[i].split(", ")
        answers[i] = ", ".join([items[0], items[-1]]) if len(items) > 2 else answers[i]
    if math.prod(len(a.split(",")) for a in answers) > MAX_SIZES:
        answers = [default(g) for g in groups]
    return answers


def example(group) -> str:
    if _width(group) > 1:
        return "a list like 512x512, 1024x768"
    return "a list like 1, 8, 32 or a range like 32-4096"


def spread(low: int, high: int, count: int = RANGE_SIZES) -> list[int]:
    """count sizes from low to high, ends included, evenly on a log scale."""
    if low >= high:
        raise SizeError(f"a range goes from small to large; got {low}-{high}")
    ratio = (high / low) ** (1 / (count - 1)) if low > 0 else None
    sizes = [low] + [round(low * ratio ** i) if ratio else round(low + (high - low) * i / (count - 1))
                     for i in range(1, count - 1)] + [high]
    return sorted(set(sizes))


def parse(group, text: str) -> list[dict]:
    """An answer for a group: the settings it names, each {knob name: value}."""
    width = _width(group)
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        raise SizeError(f"{label(group)}: give {example(group)}")
    settings = []
    for item in items:
        if width == 1 and "-" in item.strip("-"):
            low, _, high = item.partition("-")
            try:
                values = spread(int(low), int(high))
            except ValueError as error:
                raise SizeError(f"{label(group)}: {item!r} is not a range like 32-4096 ({error})") from None
            settings += [{group[0]["name"]: v} for v in values]
            continue
        parts = [p.strip() for p in item.lower().split("x")]
        if len(parts) != width or not all(p.isdigit() and int(p) > 0 for p in parts):
            raise SizeError(f"{label(group)}: {item!r} is not {'a size' if width == 1 else f'{width} sizes joined by x'}; "
                            f"give {example(group)}")
        values, setting = [int(p) for p in parts], {}
        for knob in group:
            take = 2 if knob["kind"] == "image" else 1
            setting[knob["name"]] = values[:take] if take == 2 else values[0]
            values = values[take:]
        settings.append(setting)
    return settings


def parse_once(knob, text: str) -> int:
    """A knob that runs the model more or fewer times takes one value."""
    text = str(text).strip()
    if not text.isdigit():
        raise SizeError(f"{knob['name']}: one number, the value your app uses; got {text!r}")
    return int(text)


def questions(groups, more, recorded: list[dict] | None = None) -> list[dict]:
    """The questions, each defaulting to the sizes the app called with
    (recorded: each call's {knob name: value})."""
    asked = [{"id": str(i + 1), "ask": f"Sizes for {label(group)}: {example(group)}", "default": answer,
              "knobs": [k["name"] for k in group]} for i, (group, answer) in enumerate(zip(groups, defaults(groups, recorded)))]
    asked += [{"id": str(len(groups) + i + 1), "ask": f"{knob['name']}: the value to use for every size",
               "default": str(knob["value"]), "knobs": [knob["name"]]} for i, knob in enumerate(more)]
    return asked


def describe(groups, more, unchanged) -> str:
    lines = []
    if groups:
        lines.append("These change how much work the model does:")
        lines += [f"  {i + 1}. {label(g)}" for i, g in enumerate(groups)]
    if more:
        lines.append("These run the model more or fewer times (one value, used for every size):")
        lines += [f"  {len(groups) + i + 1}. {k['name']}" for i, k in enumerate(more)]
    same = [k["name"] for k in unchanged if k.get("effect") == "same"]
    fixed = [k["name"] for k in unchanged if k.get("effect") == "fails"]
    if same:
        lines.append("These don't change the work:")
        lines.append("  " + ", ".join(same))
    if fixed:
        lines.append("These can't change (the call fails at another size):")
        lines.append("  " + ", ".join(fixed))
    return "\n".join(lines)


def sizes(groups, more, answers: dict) -> list[dict]:
    """Every combination of the answers: the sizes to search, each
    {knob name: value}. answers: {question id: text}."""
    per_group = [parse(group, answers[str(i + 1)]) for i, group in enumerate(groups)]
    once = {knob["name"]: parse_once(knob, answers[str(len(groups) + i + 1)]) for i, knob in enumerate(more)}
    count = math.prod(len(g) for g in per_group) if per_group else 1
    if count > MAX_SIZES:
        raise SizeError(f"that is {count} sizes ({' x '.join(str(len(g)) for g in per_group)}); each is checked and "
                        f"timed, so give at most {MAX_SIZES}: fewer values, or a shorter range")
    combined = [dict(itertools.chain.from_iterable(s.items() for s in combo)) for combo in itertools.product(*per_group)]
    return [{**setting, **once} for setting in (combined or [{}])]


def size_label(setting: dict, knobs: list = ()) -> str:
    """A size as init lists it: question=16 tokens, width=512, height=512."""
    from backcountry_runtime.sizes import label
    return label(setting, knobs)
