"""The sizes of a marked function's call, one way for every kind of model:
knobs found in its arguments, and the same call made at other settings.

A knob is something in a call's arguments that can be set:
  int     a number argument (width=1024, max_tokens=150, horizon=24)
  text    a text argument, or the longest text inside a list or dict argument
          (a chat's last message); measured in tokens when a tokenizer is
          reachable from the function, else in characters
  length  a list or tuple argument's length (a batch of images, token ids)
  axis    an array argument's axis; axes of the same length in other array
          arguments move with it (x [16, 336, 8] and mask [16, 336])
  image   a PIL image argument's width and height, or an image file's (a
          path to a .png or .jpg, as mlx-vlm apps pass): resized copies
          are written to a temporary folder and their paths passed
A setting is the knob's value in its own units, made by repeating or cutting
the real value the app passed (never invented data). init finds which knobs
change the model's work by probing; the search builds each chosen size's call
with apply().
"""

from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

MAX_TEXT_DEPTH = 3
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def bind(fn, args, kwargs):
    """The call's arguments by name, defaults filled in; a **kwargs dict's
    entries are their own names."""
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _var_keyword(fn) -> str | None:
    return next((p.name for p in inspect.signature(fn).parameters.values() if p.kind is p.VAR_KEYWORD), None)


def named(fn, args, kwargs) -> dict:
    """{name: value} for every argument, a **kwargs dict's entries flattened."""
    bound = bind(fn, args, kwargs)
    values = dict(bound.arguments)
    extra = _var_keyword(fn)
    if extra:
        values.update(values.pop(extra, {}) or {})
    return values


def _is_array(value) -> bool:
    return isinstance(value, mx.array) or (type(value).__module__ == "numpy" and hasattr(value, "shape"))


def _is_image(value) -> bool:
    return type(value).__module__.startswith("PIL.") and hasattr(value, "resize") and hasattr(value, "size")


def _image_file_size(value) -> list[int] | None:
    """[width, height] of an image file a path (str or Path) names, else None."""
    if not isinstance(value, (str, os.PathLike)) or len(os.fspath(value)) > 4096:
        return None
    path = Path(value)
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
        return None
    try:
        from PIL import Image
        with Image.open(path) as image:
            return list(image.size)
    except Exception:  # no Pillow, or not an image after all: not a knob
        return None


def _resized_image_file(value, size):
    """A copy of the image file at value resized to size, in a temporary
    folder; its path, as the same type the app passed (str or Path)."""
    from PIL import Image
    path = Path(value).resolve()
    width, height = (int(v) for v in size)
    stamp = hashlib.sha256(f"{path}:{path.stat().st_mtime_ns}".encode()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / "backcountry-images" / f"{path.stem}-{stamp}-{width}x{height}{path.suffix}"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            image.resize((width, height)).save(out)
    return type(value)(str(out))


def _longest_text(value, path=(), depth=0):
    """(path, text) of the longest text inside a list or dict, or None."""
    if isinstance(value, str):
        return path, value
    if depth >= MAX_TEXT_DEPTH:
        return None
    items = (() if isinstance(value, nn.Module) else value.items() if isinstance(value, dict)
             else enumerate(value) if isinstance(value, (list, tuple)) else ())
    best = None
    for key, item in items:
        found = _longest_text(item, path + (key,), depth + 1)
        if found and (best is None or len(found[1]) > len(best[1])):
            best = found
    return best


def find(fn, args, kwargs) -> list[dict]:
    """Every knob in the call, each {"name", "kind", "param", ...}."""
    knobs = []
    arrays = []
    for name, value in named(fn, args, kwargs).items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            knobs.append({"name": name, "kind": "int", "param": name, "value": value})
        elif (size := _image_file_size(value)) is not None:
            knobs.append({"name": name, "kind": "image", "param": name, "value": size, "file": True})
        elif isinstance(value, str):
            knobs.append({"name": name, "kind": "text", "param": name, "path": [], "value": len(value)})
        elif _is_array(value):
            arrays.append((name, value))
        elif _is_image(value):
            knobs.append({"name": name, "kind": "image", "param": name, "value": list(value.size)})
        elif isinstance(value, (list, tuple, dict)) and not isinstance(value, nn.Module):
            # a model is a dict too (MLX's Module): gpt-oss keeps layer_types, a list
            # of strings, among its attributes, and init asked a size of the model
            if isinstance(value, (list, tuple)) and value:
                knobs.append({"name": name, "kind": "length", "param": name, "value": len(value)})
            found = _longest_text(value)
            if found and found[0]:
                where = "".join(f"[{k!r}]" for k in found[0])
                knobs.append({"name": f"{name}{where}", "kind": "text", "param": name, "path": list(found[0]),
                              "value": len(found[1])})
    # array axes: one knob per distinct length among non-trivial axes, moving
    # every axis of that length in every array argument together
    lengths = {}
    for name, value in arrays:
        for axis, n in enumerate(value.shape):
            if n > 1:
                lengths.setdefault(n, []).append([name, axis])
    for n, members in lengths.items():
        label = " and ".join(f"{name} axis {axis}" for name, axis in members)
        knobs.append({"name": label, "kind": "axis", "members": members, "value": n})
    return knobs


def _resized_list(items, n):
    items = list(items)
    if not items:
        raise ValueError("an empty list has nothing to repeat")
    return (items * (n // len(items) + 1))[:n]


def _resized_array(value, axis, n):
    length = value.shape[axis]
    reps = n // length + 1
    if isinstance(value, mx.array):
        tiled = mx.concatenate([value] * reps, axis=axis)
        return mx.take(tiled, mx.arange(n), axis=axis)
    import numpy as np
    tiled = np.concatenate([value] * reps, axis=axis)
    return np.take(tiled, np.arange(n), axis=axis)


def _resized_text(text: str, n: int, tokenizer=None) -> str:
    """text at n characters, or at n tokens with a tokenizer (the text's own
    tokens repeated or cut, decoded back)."""
    if tokenizer is None:
        return (text * (n // max(len(text), 1) + 1))[:n]
    ids = list(tokenizer.encode(text))
    return tokenizer.decode(_resized_list(ids, n))


def tokenizer_for(fn, args, kwargs):
    """A tokenizer the function reaches (an argument, an attribute of one, or
    a global): anything with encode() and decode() that turns text into ints."""
    candidates = list(named(fn, args, kwargs).values()) + list(fn.__globals__.values())
    seen = set()
    for value in candidates:
        for item in (value, getattr(value, "tokenizer", None)):
            if item is None or id(item) in seen or isinstance(item, type):
                continue
            seen.add(id(item))
            encode, decode = getattr(item, "encode", None), getattr(item, "decode", None)
            if callable(encode) and callable(decode) and not isinstance(item, (str, bytes)):
                try:
                    ids = list(encode("probe"))
                except Exception:
                    continue
                if ids and all(isinstance(i, int) for i in ids):
                    return item
    return None


def text_size(knob, fn, args, kwargs, tokenizer) -> int:
    """A text knob's current size in its units (tokens with a tokenizer)."""
    value = named(fn, args, kwargs)[knob["param"]]
    for key in knob.get("path", []):
        value = value[key]
    return len(tokenizer.encode(value)) if tokenizer is not None else len(value)


def apply(fn, args, kwargs, knobs: list[dict], settings: dict, tokenizer=None):
    """(args, kwargs) of the call with each knob set (settings: {name: value});
    the value the app passed is repeated or cut, never invented."""
    bound = bind(fn, args, kwargs)
    extra = _var_keyword(fn)
    holder = dict(bound.arguments)
    flat = dict(holder.pop(extra, {}) or {}) if extra else {}

    def get(name):
        return holder[name] if name in holder else flat[name]

    def put(name, value):
        if name in holder:
            holder[name] = value
        else:
            flat[name] = value

    for knob in knobs:
        if knob["name"] not in settings:
            continue
        value = settings[knob["name"]]
        kind = knob["kind"]
        if kind == "int":
            put(knob["param"], int(value))
        elif kind == "length":
            original = get(knob["param"])
            put(knob["param"], type(original)(_resized_list(original, int(value))))
        elif kind == "image" and knob.get("file"):
            put(knob["param"], _resized_image_file(get(knob["param"]), value))
        elif kind == "image":
            put(knob["param"], get(knob["param"]).resize(tuple(int(v) for v in value)))
        elif kind == "axis":
            for name, axis in knob["members"]:
                put(name, _resized_array(get(name), axis, int(value)))
        elif kind == "text":
            path = knob.get("path", [])
            if not path:
                put(knob["param"], _resized_text(get(knob["param"]), int(value), tokenizer))
            else:
                import copy
                container = copy.deepcopy(get(knob["param"]))
                inner = container
                for key in path[:-1]:
                    inner = inner[key]
                inner[path[-1]] = _resized_text(inner[path[-1]], int(value), tokenizer)
                put(knob["param"], container)
    if extra:
        holder[extra] = flat
    bound.arguments.update(holder)
    return bound.args, bound.kwargs


def label(setting: dict, knobs: list = ()) -> str:
    """A size as init lists it: question=16 tokens, width=512, height=512."""
    units = {k["name"]: k.get("unit") for k in knobs}
    return ", ".join(f"{name}={'x'.join(map(str, v)) if isinstance(v, list) else v}"
                     f"{' ' + units[name] if units.get(name) else ''}" for name, v in setting.items())


def probe_value(knob):
    """The setting a probe tries: half the size (cheap: a probe at twice a
    FLUX image's size would take minutes), or twice a size too small to halve."""
    def one(v):
        v = int(v)
        return v // 2 if v >= 4 else 2 * v + 1
    value = knob["value"]
    return [one(v) for v in value] if isinstance(value, list) else one(value)
