"""What the user wants made faster, in their own terms: `backcountry init`
writes it as a short `backcountry.yaml`, and `run`, `start` and `check` turn it
into the full manifest in the run folder (the record of what ran):

    model: mlx-community/Qwen3-8B-4bit   # or ./my_model.py, or a local model folder
    optimize: chat                       # prompts | replies | chat (both)
    prompt_tokens: 32-2049
    optimize_at: 512
    history_tokens: 2048
    reply_tokens: 16
    keep: same-output                    # or same-math
    attempts: 100

A model file (build() and examples()) needs only `model` and `attempts` (and
`tolerance: default` to keep kernels that round differently). A full manifest
(with `workloads:`) is never an intent file: it runs as it is.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

INTENT_FILE = "backcountry.yaml"
OPTIMIZE = {"prompts": "prompt", "replies": "reply", "chat": "chat"}
FIELDS = ("model", "optimize", "prompt_tokens", "optimize_at", "history_tokens", "reply_tokens",
          "keep", "tolerance", "attempts", "judge", "serves", "sizes", "request")


class IntentError(ValueError):
    """What is wrong, said so the user (or agent) knows what to change."""


# -- what the argument is --------------------------------------------------------------

@dataclass
class Target:
    kind: str           # model_file | app_script | manifest | intent | local_model | hub_model
    value: str          # the path or the model id
    config: dict | None = None
    note: str | None = None


def classify(argument: str, hub=None) -> Target:
    """A local .py (a model file), a .yaml (a full manifest, or an intent file
    when it has no workloads), a local folder with config.json (a model on
    disk), or a Hugging Face id (asked whether it exists, and its config read:
    no weights download here). hub: a stand-in for huggingface_hub.HfApi."""
    text = argument.strip()
    match = re.match(r"^(?:https?://)?(?:huggingface\.co|hf\.co)/([^/]+/[^/?#]+)", text)
    if match:
        text = match.group(1)
    path = Path(text).expanduser()
    if path.suffix == ".py":
        if not path.is_file():
            raise IntentError(f"no file {path}")
        source = path.read_text()
        has_build, has_examples = (bool(re.search(rf"^def {name}\(", source, re.M)) for name in ("build", "examples"))
        if not has_build and not has_examples:
            return Target("app_script", str(path))  # a script that uses the model: init watches it run
        if not has_build:
            raise IntentError(f"{path} has no build(): a model file defines build(), which returns the model, "
                              "and examples(), which returns a few real inputs (one per call)")
        if not has_examples:
            raise IntentError(f"{path} has no examples(): return a few real inputs, one per call, at the sizes your "
                              "app sends (three or more sizes make a range)")
        return Target("model_file", str(path))
    if path.suffix in (".yaml", ".yml"):
        if not path.is_file():
            raise IntentError(f"no file {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict) or "model" not in raw:
            raise IntentError(f"{path} is neither a manifest nor a backcountry.yaml: it has no `model:`")
        return Target("manifest" if "workloads" in raw else "intent", str(path))
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise IntentError(f"{path} is a folder without a config.json: give a model folder MLX-LM can load, "
                              "a model file (build() and examples()), or a Hugging Face id")
        return Target("local_model", str(path), json.loads((path / "config.json").read_text()))
    if text.count("/") != 1 or path.exists():
        raise IntentError(f"{argument!r} is not a file here and not a Hugging Face id (owner/name)")
    return _hub_model(text, hub)


def _hub_model(model_id: str, hub=None) -> Target:
    try:
        if hub is None:
            from huggingface_hub import HfApi
            hub = HfApi()
        info = hub.model_info(model_id)
    except Exception as error:  # not found, gated, private, offline
        name = type(error).__name__
        if "Gated" in name:
            raise IntentError(f"{model_id} is gated: accept its terms on huggingface.co and log in "
                              "(`hf auth login`), then try again") from None
        if "NotFound" in name or "404" in str(error):
            raise IntentError(f"no model {model_id} on Hugging Face{_suggest(model_id, hub)}") from None
        raise IntentError(f"could not reach Hugging Face for {model_id} ({name}); "
                          "check the network, or give a local model folder") from None
    tags = set(getattr(info, "tags", None) or [])
    note = None if "mlx" in tags else (
        f"{model_id} is not an MLX conversion: MLX-LM may still load it (larger and slower than a quantized MLX copy)"
        + _suggest(model_id, hub))
    from .front_door import config_of
    return Target("hub_model", model_id, config_of(model_id), note)


def _suggest(model_id: str, hub) -> str:
    """MLX copies by name: the name searched as given, then without its last
    part (a typo is usually at the end: Qwen3-8B-4bitt)."""
    name = model_id.split("/")[-1]
    for search in dict.fromkeys((name, name.rsplit("-", 1)[0])):
        try:
            found = [m.id for m in hub.list_models(search=search, author="mlx-community", limit=50)]
        except Exception:
            return ""
        if found:
            import difflib
            close = sorted(found, key=lambda other: -difflib.SequenceMatcher(
                None, name.lower(), other.split("/")[-1].lower()).ratio())[:3]
            return f"; did you mean {' or '.join(close)}?"
    return ""


def is_language_model(config: dict | None) -> bool:
    """Whether MLX-LM can run it as text generation: it names a vocabulary and
    an architecture MLX-LM has."""
    if not config:
        return False
    text = config.get("text_config", config)
    if not (text.get("vocab_size") or config.get("vocab_size")):
        return False
    import importlib.util
    if importlib.util.find_spec("mlx_lm") is None:
        raise IntentError("this is a language model, and running one needs mlx-lm: `uv add mlx-lm` "
                          "(or `pip install mlx-lm`)")
    kind = config.get("model_type") or text.get("model_type")
    return bool(kind) and importlib.util.find_spec(f"mlx_lm.models.{kind}") is not None


# -- the questions -----------------------------------------------------------------

def questions(target: Target) -> list[dict]:
    """Only what the model cannot say: what the app does with it, the sizes it
    sees, how identical the outputs must be, and how long to search."""
    attempts = {"id": "attempts", "ask": "How many kernel attempts may the search make? (each takes minutes on a "
                "small model, up to half an hour on a large one)", "default": 100, "type": "int"}
    if target.kind == "model_file":
        return [{"id": "tolerance", "ask": "Must every kernel give bit-identical outputs (exact), or may outputs "
                 "differ by rounding within each type's limit (default)?", "default": "exact",
                 "choices": ["exact", "default"]}, attempts]
    return [
        {"id": "optimize", "ask": "What should get faster: reading prompts, writing replies, or both (chat)?",
         "default": "chat", "choices": list(OPTIMIZE)},
        {"id": "prompt_tokens", "ask": "How long are the prompts your app sends, in tokens (a range)?",
         "default": "32-2049", "type": "range", "when": ("prompts", "chat")},
        {"id": "optimize_at", "ask": "Which prompt length matters most (the search optimizes there)?",
         "default": 512, "type": "int", "when": ("prompts", "chat")},
        {"id": "history_tokens", "ask": "How long is the conversation before a reply, in tokens?",
         "default": 2048, "type": "int", "when": ("replies", "chat")},
        {"id": "reply_tokens", "ask": "How long is a typical reply, in tokens?", "default": 16, "type": "int",
         "when": ("replies", "chat")},
        {"id": "keep", "ask": "Keep the same output tokens and allow a faster generation loop (same-output), or "
         "keep the exact same math with faster kernels only (same-math)?", "default": "same-output",
         "choices": ["same-output", "same-math"]},
        attempts]


def _parse(question: dict, value):
    if value is None or value == "":
        return question["default"]
    if question.get("choices") and str(value) not in question["choices"]:
        raise IntentError(f"{question['id']}: one of {', '.join(question['choices'])}; got {value!r}")
    if question.get("type") == "int":
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise IntentError(f"{question['id']}: a whole number; got {value!r}") from None
        if number < 1:
            raise IntentError(f"{question['id']}: at least 1; got {number}")
        return number
    if question.get("type") == "range":
        low, high = _range(value)
        return f"{low}-{high}"
    return str(value)


def _range(value) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(value))
    if not match or int(match.group(1)) >= int(match.group(2)):
        raise IntentError(f"a range like 32-2049 (smallest-largest); got {value!r}")
    return int(match.group(1)), int(match.group(2))


def answer(target: Target, given: dict, ask=None, defaults: dict | None = None) -> dict:
    """Every question answered: from `given` (flags), else from ask(question)
    (a person at a terminal), else its default (`defaults` replace the
    standard ones: what a watched script actually did)."""
    answers = {}
    for question in questions(target):
        if defaults and question["id"] in defaults:
            question = {**question, "default": defaults[question["id"]]}
        if question.get("when") and answers.get("optimize") not in question["when"]:
            continue
        value = given.get(question["id"])
        if value is None and ask is not None:
            value = ask(question)
        answers[question["id"]] = _parse(question, value)
    if "optimize_at" in answers:
        low, high = _range(answers["prompt_tokens"])
        if not low <= answers["optimize_at"] <= high:
            raise IntentError(f"optimize_at ({answers['optimize_at']}) must be within prompt_tokens ({low}-{high})")
    return answers


def terminal_ask(question: dict):
    hint = f" ({'/'.join(question['choices'])})" if question.get("choices") else ""
    reply = input(f"{question['ask']}{hint} [default: {question['default']}]: ").strip()
    return reply or None


def summary(target: Target, answers: dict) -> str:
    """The plan in one or two plain sentences, for the user to confirm."""
    if target.kind == "model_file":
        outputs = "bit-identical outputs" if answers["tolerance"] == "exact" else \
            "outputs within each type's rounding limit"
        return (f"Will make {target.value} faster at the sizes its examples() show, keeping {outputs}, "
                f"in up to {answers['attempts']} kernel attempts.")
    parts = []
    if answers["optimize"] in ("prompts", "chat"):
        low, high = _range(answers["prompt_tokens"])
        parts.append(f"reading prompts of {low:,}-{high:,} tokens (optimized at {answers['optimize_at']:,})")
    if answers["optimize"] in ("replies", "chat"):
        parts.append(f"writing {answers['reply_tokens']:,}-token replies after {answers['history_tokens']:,} tokens "
                     "of conversation")
    keep = ("the same output tokens (a faster generation loop may be found first)" if answers["keep"] == "same-output"
            else "the exact same math (faster kernels only)")
    return f"Will make {target.value} faster at {' and '.join(parts)}, keeping {keep}, in up to " \
           f"{answers['attempts']} kernel attempts."


def write_intent(target: Target, answers: dict, out: Path) -> Path:
    """backcountry.yaml: each answer on its own line, commented with what it means."""
    comments = {
        "optimize": "what gets faster: prompts | replies | chat (both)",
        "prompt_tokens": "the prompt lengths your app sends",
        "optimize_at": "the prompt length the search optimizes at",
        "history_tokens": "tokens of conversation before a reply",
        "reply_tokens": "tokens in a typical reply",
        "keep": "same-output (a faster loop allowed, the same outputs) | same-math (kernels only)",
        "tolerance": "exact (bit-identical) | default (rounding within each type's limit)",
        "attempts": "kernel attempts the search may make",
        "serves": "the marked function whose first call installs the result (written by init)",
        "sizes": "one call made at each (re-run init after changing how the app calls it)",
        "request": "the request init made for an app that makes none",
    }
    model = target.value
    if target.kind in ("model_file", "local_model"):
        model = Path(model).resolve()
        try:  # inside the project: written from backcountry.yaml's own folder, so the project can move
            model = str(model.relative_to(out.resolve().parent))
        except ValueError:
            model = str(model)
    lines = [f"# Written by `backcountry init`. Edit any line, then: backcountry check {out.name}",
             f"model: {model}"]
    for key, value in answers.items():
        if isinstance(value, list):  # one entry a line, to edit
            lines += [f"{key + ':':<32}# {comments[key]}"] + [f"  - {_flow(item)}" for item in value]
            continue
        shown = _flow(value) if isinstance(value, dict) else str(value)
        lines.append(f"{key + ':':<16}{shown:<16}  # {comments[key]}")
    out.write_text("\n".join(lines) + "\n")
    return out


def _flow(value) -> str:
    """A dict as YAML on one line: {question: 16, max_tokens: 4}; a name YAML
    would misread (messages[0]['content']) quoted."""
    if isinstance(value, dict):
        def key(k):
            return k if re.fullmatch(r"[A-Za-z_][\w.-]*", k) else json.dumps(k)
        return "{" + ", ".join(f"{key(k)}: {_flow(v)}" for k, v in value.items()) + "}"
    return json.dumps(value) if isinstance(value, str) else str(value)


# -- from intent to manifest ----------------------------------------------------------

def load(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = sorted(set(raw) - set(FIELDS))
    if unknown:
        raise IntentError(f"{path}: unknown field(s) {', '.join(unknown)}; the fields are {', '.join(FIELDS)} "
                          "(a full manifest, with workloads:, takes the rest)")
    if "model" not in raw:
        raise IntentError(f"{path}: needs model:")
    return raw


def resolve(path: Path, work_dir: Path) -> Path:
    """The full manifest (and for a model id, model.py) for an intent file,
    written into work_dir; returns the manifest's path."""
    from .front_door import write, write_from_examples
    raw = load(path)
    beside = Path(path).resolve().parent / str(raw["model"])   # a path in it is read from its own folder
    target = classify(str(beside) if beside.exists() else str(raw["model"]))
    attempts = raw.get("attempts")
    if target.kind == "model_file":
        from backcountry_runtime.tune import marked_model_file, marked_target
        if marked_model_file(target.value):
            target = Target("model_file", str(_marked_for_this_run(Path(target.value), raw, Path(path), work_dir)))
        made_for = marked_model_file(target.value)
        extra = sorted(set(raw) & {"optimize", "prompt_tokens", "optimize_at", "history_tokens", "reply_tokens", "keep",
                                   "serves", "sizes", "request"} - ({"keep", "sizes", "request"} if made_for else set()))
        if extra:
            raise IntentError(f"{path}: {', '.join(extra)} describe a language model's use; a model file's "
                              "examples() say how it is used (only attempts and tolerance apply)")
        tolerance = raw.get("tolerance", "exact")
        if tolerance not in ("exact", "default"):
            raise IntentError(f"{path}: tolerance is exact or default; got {tolerance!r}")
        keep = raw.get("keep", "same-math")
        if keep not in ("same-output", "same-math"):
            raise IntentError(f"{path}: keep is same-output or same-math; got {keep!r}")
        function = marked_target(made_for.get("FILE", made_for["SCRIPT"]), made_for["FUNCTION"][1]) \
            if made_for and keep == "same-output" else None
        return write_from_examples(Path(target.value), work_dir, attempts,
                                   tolerances=None if tolerance == "exact" else "default", function=function)
    if target.kind in ("manifest", "intent"):
        raise IntentError(f"{path}: model: names a {target.kind}; it should name the model itself")
    if not is_language_model(target.config):
        raise IntentError(f"{target.value} is not a text model MLX-LM runs; write a model file (build() and "
                          "examples()) for it and give that as model:")
    optimize = raw.get("optimize", "chat")
    if optimize not in OPTIMIZE:
        raise IntentError(f"{path}: optimize is one of {', '.join(OPTIMIZE)}; got {optimize!r}")
    if "tolerance" in raw:
        raise IntentError(f"{path}: tolerance applies to model files; a language model keeps same-output or same-math")
    low, high = _range(raw.get("prompt_tokens", "32-2049"))
    return write(target.value, OPTIMIZE[optimize], work_dir, attempts=attempts, prompt_tokens=(low, high),
                 optimize_at=int(raw.get("optimize_at", 512)), history_tokens=int(raw.get("history_tokens", 2048)),
                 reply_tokens=int(raw.get("reply_tokens", 16)), keep=raw.get("keep", "same-output"),
                 written_by=f"backcountry run {Path(path).name}", serves=raw.get("serves"))


def _marked_for_this_run(template: Path, raw: dict, intent: Path, work_dir: Path) -> Path:
    """The marked function's model file this run uses, in its work folder: the
    one init wrote (its paths relative to the project, so the project can be
    copied or cloned) with the app's paths for this machine and the sizes and
    request backcountry.yaml says now (edited or not)."""
    project = intent.resolve().parent
    lines = []
    for line in template.read_text().splitlines():
        name = line.split(" = ", 1)[0] if " = " in line and not line.startswith(" ") else None
        if name in ("SCRIPT", "FILE"):
            value = ast.literal_eval(line.split(" = ", 1)[1].split("   #", 1)[0])
            line = f"{name} = {str((project / value).resolve())!r}"
        elif name == "SIZES" and raw.get("sizes") is not None:
            line = f"SIZES = {_sizes_from(raw['sizes'], intent)!r}   # the sizes to search: the call made at each"
        elif name == "REQUEST" and raw.get("request") is not None:
            line = f"REQUEST = {dict(raw['request'])!r}   # the request made once the app has run"
        lines.append(line)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / template.name
    out.write_text("\n".join(lines) + "\n")
    return out


def _sizes_from(sizes, intent: Path) -> list[dict]:
    # {} is the call as the app made it: what init writes for a function with no
    # size questions (an audio file's path), which check rejected
    if not isinstance(sizes, list) or not sizes or not all(isinstance(s, dict) for s in sizes):
        raise IntentError(f"{intent}: sizes is a list with one entry per size, each naming its values "
                          "({question: 16} or {width: 512, height: 512})")
    return [dict(s) for s in sizes]


def _confirm_and_write(target: Target, answers: dict, out: Path, interactive: bool, plan: str | None = None) -> int:
    print(plan or summary(target, answers))
    if interactive and input("Write it? [Y/n]: ").strip().lower() in ("n", "no"):
        print("nothing written")
        return 1
    if out.exists() and interactive and input(f"{out} exists: replace it? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("nothing written")
        return 1
    write_intent(target, answers, out)
    print(f"wrote {out}\nnext: backcountry check {out}   (no attempts; minutes, or up to an hour when a request generates thousands of tokens; then `backcountry start {out}`)")
    return 0


def _init_watched(target: Target, given: dict, out: Path, interactive: bool, show_questions: bool) -> int:
    """A script that uses its model, any library or the user's own code: run it
    once, list the parts it called, and ask which to make faster."""
    from .watch import choosable, describe, language_model_defaults, record, write_model_file
    print(f"running {target.value} once to see how it uses its model ...", flush=True)
    recorded_in = out.parent.resolve() / ".backcountry"   # the recording and the inputs it saw
    main = not given.get("ask")
    try:
        recording = record(Path(target.value), recorded_in, given.get("script_args"), main=main)
        if not recording.get("marked") and recording.get("defined"):
            # the app makes no request (a server waits for one): the user gives one, made once it has run
            request = _ask_request(Path(target.value).name, recording, given, interactive, show_questions)
            if request is None:
                return 0
            recording = record(Path(target.value), recorded_in, given.get("script_args"), request=request, main=main)
    except ValueError as error:
        raise IntentError(str(error)) from None
    if recording.get("marked"):
        return _init_marked(target, recording, given, out, interactive, show_questions)
    parts = recording["parts"]
    if not parts:
        raise IntentError(f"{target.value} called no part of an MLX model (an mlx.nn.Module): run the model "
                          "in it the way your app does, or write a model file with build() and examples()")
    usable = [i for i, part in enumerate(parts) if choosable(part)]
    if not usable:
        raise IntentError("no part it called can be timed alone:\n  " + "\n  ".join(describe(p) for p in parts)
                          + "\nwrite a model file with build() and examples() for the part to make faster")
    part_question = {"id": "part", "ask": "Which part should get faster?", "default": str(usable[0] + 1),
                     "choices": [str(i + 1) for i in usable], "options": {str(i + 1): describe(p) for i, p in enumerate(parts)}}
    if show_questions:
        print(json.dumps({"target": {"kind": target.kind, "value": target.value}, "questions": [part_question],
                          "then": "for a language model, the questions of `init <its id>`; otherwise which "
                                  "recorded input size matters most (--size), tolerance and attempts"}, indent=1))
        return 0
    print("\n".join(f"  {i + 1}. {describe(p)}" for i, p in enumerate(parts)))
    choice = given.get("part") or (terminal_ask(part_question) if interactive else None) or part_question["default"]
    if str(choice) not in part_question["choices"]:
        raise IntentError(f"part: one of {', '.join(part_question['choices'])}; got {choice!r}")
    part = parts[int(choice) - 1]
    if part["language_model"]:
        if not recording["loads"]:
            raise IntentError("the script loads its language model without MLX-LM's load(): run "
                              "`backcountry init <its Hugging Face id or folder>` instead")
        model = classify(recording["loads"][-1])
        answers = answer(model, given, terminal_ask if interactive else None, defaults=language_model_defaults(part))
        return _confirm_and_write(model, answers, out, interactive)
    sizes = part["sizes"]
    if len(sizes) > 1:
        print("  input sizes it was called with:")
        for i, size in enumerate(sizes):
            shapes = ", ".join(f"{a['shape']} {a['dtype']}" for a in size["inputs"])
            options = f" with {size['options']}" if size["options"] != "{}" else ""
            print(f"    {i + 1}. {shapes}{options} ({size['calls']} calls)")
    size_question = {"id": "size", "ask": "Which input size matters most (the search optimizes there; the others "
                     "are checked and served where faster)?", "default": "1",
                     "choices": [str(i + 1) for i in range(len(sizes))]}
    pick = given.get("size") or (terminal_ask(size_question) if interactive and len(sizes) > 1 else None) or "1"
    if str(pick) not in size_question["choices"]:
        raise IntentError(f"size: one of {', '.join(size_question['choices'])}; got {pick!r}")
    model_file = write_model_file(recording, part, int(pick) - 1, recorded_in, recorded_in)
    print(f"wrote {model_file}: build() runs your script until it calls {part['qualname']}; examples() are its inputs")
    target = Target("model_file", str(model_file))
    answers = answer(target, given, terminal_ask if interactive else None)
    return _confirm_and_write(target, answers, out, interactive)


def _ask_request(script: str, recording: dict, given: dict, interactive: bool, show_questions: bool) -> dict | None:
    """One request to the marked function, from the user, for an app that
    makes none: each argument init cannot fill from the app's globals of the
    same name, typed at the prompt or given as --request "name=value; ...".
    None after printing the questions as JSON (show_questions)."""
    module, qualname = recording["defined"][0]
    params = recording["parameters"][f"{module}:{qualname}"]
    asked = [p for p in params if not p["global"]]
    where = "" if recording.get("main", True) else " outside its `if __name__ == \"__main__\":` block"
    hint = "text, a number, a list, or a file beside the app: an image, a .json, a .txt"
    if show_questions and not given.get("request"):     # once it is given, the questions after it
        print(json.dumps({"target": {"kind": "app_script", "value": script}, "request": {
            "function": qualname, "why": f"{script} never calls {qualname}{where}",
            "arguments": [{"name": p["name"], "default": p["default"], "annotation": p["annotation"]} for p in asked],
            "from_the_app": [p["name"] for p in params if p["global"]], "values": hint},
            "then": "give it with --request \"name=value; name=value\", then answer the rest"}, indent=1))
        return None
    typed = _request_flag(given.get("request"))
    unknown = sorted(set(typed) - {p["name"] for p in asked})
    if unknown:
        raise IntentError(f"--request names {', '.join(unknown)}, which {qualname} does not take "
                          f"(it takes {', '.join(p['name'] for p in asked) or 'nothing init can ask'})")
    if not show_questions:
        ask = ("it runs on the request given with --request" if typed and not interactive
               else f"Give it one request, as your app would ({hint}):")
        print(f"{qualname} (marked with @backcountry.tune) is never called by {script}{where}: {ask}")
    filled = [p["name"] for p in params if p["global"]]
    if filled and not show_questions:
        print(f"  {', '.join(filled)}: {script}'s own ({'its globals' if len(filled) > 1 else 'its global'} of that name)")
    inputs = {}
    for p in asked:
        value = typed.get(p["name"])
        if value is None and interactive:
            kind = f" ({p['annotation']})" if p["annotation"] else ""
            shown = p["default"] if p["default"] is not None else "none: required"
            value = input(f"  {p['name']}{kind} [default: {shown}]: ").strip()
            while not value and p["default"] is None:
                value = input(f"  {p['name']} is required [default: none: required]: ").strip()
        if not value and p["default"] is None:
            raise IntentError(f"{qualname} needs {p['name']}: give it with --request \"{p['name']}=...\"")
        if value:
            inputs[p["name"]] = value
    return {"function": [module, qualname], "inputs": inputs}


def _request_flag(text: str | None) -> dict:
    """--request "question=Write a haiku; history=chat.json" as {name: text}."""
    out = {}
    for item in (text or "").split(";"):
        if item.strip():
            name, sep, value = item.partition("=")
            if not sep:
                raise IntentError(f"--request takes name=value pairs separated by ';', got {item.strip()!r}")
            out[name.strip()] = value.strip()
    return out


def watch_max_calls() -> int:
    from .watch import MAX_CALLS
    return MAX_CALLS


def _init_marked(target: Target, recording: dict, given: dict, out: Path, interactive: bool,
                 show_questions: bool) -> int:
    """A script whose request function is marked with @backcountry.tune: list
    its calls and ask which matters most; the search replays the function."""
    from .watch import describe_marked, marked_functions
    functions = marked_functions(recording)
    names = list(functions)
    if len(names) > 1:
        function_question = {"id": "part", "ask": "Which marked function should get faster?", "default": "1",
                             "choices": [str(i + 1) for i in range(len(names))],
                             "options": {str(i + 1): f"{n[1]} ({len(functions[n])} calls)" for i, n in enumerate(names)}}
    else:
        function_question = None
    if show_questions:
        from .size_questions import questions as size_questions, sort
        sized = {}
        for name in names:
            calls = functions[name]
            base = next((c for c in calls if c.get("knobs") is not None), None)
            if base is not None:
                groups, more, _ = sort(base["knobs"])
                sized[name[1]] = size_questions(groups, more, _recorded(calls))
        print(json.dumps({"target": {"kind": target.kind, "value": target.value},
                          "questions": [q for q in (function_question,) if q],
                          "calls": {n[1]: [describe_marked(c) for c in functions[n]] for n in names},
                          "sizes": sized,
                          "then": "answer the sizes with --sizes, by question id (--sizes \"1=32-4096; 2=256\"), "
                                  "then --keep, --tolerance and --attempts",
                          "keep": {**KEEP_MARKED, "ask": KEEP_MARKED["ask"].format(name="the function")}}, indent=1))
        return 0
    choice = "1"
    if function_question:
        print("\n".join(f"  {k}. {v}" for k, v in function_question["options"].items()))
        choice = given.get("part") or (terminal_ask(function_question) if interactive else None) or "1"
        if str(choice) not in function_question["choices"]:
            raise IntentError(f"part: one of {', '.join(function_question['choices'])}; got {choice!r}")
    function = names[int(choice) - 1]
    calls = functions[function]
    print(f"{function[1]} (marked with @backcountry.tune) was called {len(calls)} time{'s' if len(calls) != 1 else ''}:")
    for i, call in enumerate(calls):
        print(f"    {i + 1}. {describe_marked(call)}")
    unlocated = sorted({p["class"] for call in calls for p in call["unlocated"]})
    if unlocated:
        many = len(unlocated) > 1
        print(f"  note: it calls {', '.join(unlocated)}, which {'are' if many else 'is'} not reachable from its "
              f"arguments, its module's globals or a library it uses; {'their' if many else 'that'} work is left as it is")
    base = next((i for i, call in enumerate(calls) if call.get("knobs") is not None), 0)
    if not calls[base]["parts"]:
        raise IntentError(f"{function[1]} ran no MLX model backcountry can reach (an mlx.nn.Module in its arguments, "
                          "its module's globals or a library it uses)")
    points = _ask_sizes(calls[base].get("knobs") or [], given, interactive, _recorded(calls))
    if calls[base]["seconds"] > 10:
        print(f"  note: a call takes about {calls[base]['seconds']:.0f} s, and the search replays each size many "
              "times (each kernel's whole-model check and timing), so a run will take hours. Marking a smaller "
              "function (one step of a loop), or smaller sizes, makes it faster")
    from .watch import MODEL_FILE
    # the model file and backcountry.yaml go together: an init that fails or
    # is declined puts back the model file it found (a failed size check left
    # a new one beside the old yaml, which check then rejected)
    kept = out.parent.resolve() / ".backcountry" / MODEL_FILE
    before = kept.read_bytes() if kept.is_file() else None
    code = 1
    try:
        code = _write_marked(recording, function, base, out, points, given, interactive)
        return code
    finally:
        if code != 0:
            if before is None:
                kept.unlink(missing_ok=True)
            else:
                kept.write_bytes(before)


def _write_marked(recording, function, base, out, points, given, interactive) -> int:
    from .watch import write_marked_model_file
    model_file = write_marked_model_file(recording, function, base, out.parent.resolve() / ".backcountry", points)
    _check_sizes(model_file, points)
    model = Target("model_file", str(model_file))
    keep = {**KEEP_MARKED, "ask": KEEP_MARKED["ask"].format(name=function[1])}
    value = given.get("keep")
    if value is None and interactive:
        value = terminal_ask(keep)
    answers = {"keep": _parse(keep, value), **answer(model, given, terminal_ask if interactive else None)}
    if answers["keep"] == "same-output":  # the loop search calls the function through run()
        write_marked_model_file(recording, function, base, model_file.parent, points, loop=True)
    answers["sizes"] = points["sizes"]   # in backcountry.yaml, where they can be edited
    if recording.get("request"):
        answers["request"] = recording["request"]["inputs"]
    from .size_questions import plan as sizes_plan
    count = len(points["sizes"])
    shown = sizes_plan(points["sizes"], points["knobs"]) or "as your app called it"
    outputs = "identical outputs" if answers["tolerance"] == "exact" else "outputs within each type's rounding limit"
    how = (f"a faster version of {function[1]} itself first (the same outputs), then " if answers["keep"] == "same-output"
           else "")
    plan = (f"Will make {function[1]} faster at {count} size{'s' if count != 1 else ''} ({shown}), keeping {outputs}: "
            f"{how}up to {answers['attempts']} kernel attempts.")
    return _confirm_and_write(model, answers, out, interactive, plan)


# a marked function: its own code may change too, its outputs kept
KEEP_MARKED = {"id": "keep", "ask": "May the search also rewrite how {name} uses the model (a faster loop, the same "
               "outputs), or only make the model's own math faster (kernels)? (same-output adds hours)",
               "default": "same-output", "choices": ["same-output", "same-math"]}


def _recorded(calls: list) -> list[dict]:
    """Each recorded call's sizes, {knob name: value}."""
    return [{name: value for name, value, _ in call.get("sizes") or []} for call in calls]


def _ask_sizes(knobs: list, given: dict, interactive: bool, recorded: list[dict] | None = None) -> dict:
    """The sizes to search, asked in the user's terms (the knobs init's probe
    found change the work), each defaulting to the sizes the app called with;
    --sizes "1=32-4096; 2=256" answers them by number."""
    from .size_questions import SizeError, describe, parse, parse_once, questions, sizes, sort
    groups, more, unchanged = sort(knobs)
    print(describe(groups, more, unchanged))
    asked = questions(groups, more, recorded)
    flags = {}
    for item in filter(None, (part.strip() for part in str(given.get("sizes") or "").split(";"))):
        key, _, value = item.partition("=")
        flags[key.strip()] = value.strip()
    answers = {}
    for question in asked:
        i = int(question["id"]) - 1
        check = (lambda text, g=groups[i]: parse(g, text)) if i < len(groups) else \
            (lambda text, k=more[i - len(groups)]: parse_once(k, text))
        while True:
            text = flags.get(question["id"]) or (terminal_ask(question) if interactive else None) or question["default"]
            try:
                check(text)
                sizes(groups[:i + 1], [], {**answers, question["id"]: text}) if i < len(groups) else None
            except SizeError as error:
                if not interactive or question["id"] in flags:
                    raise IntentError(str(error)) from None
                print(f"  {error}")
                continue
            answers[question["id"]] = text
            break
    try:
        points = sizes(groups, more, answers)
    except SizeError as error:
        raise IntentError(str(error)) from None
    return {"knobs": [k for g in groups for k in g] + more, "sizes": points}


def _check_sizes(model_file: Path, points: dict) -> None:
    """Every chosen size made once, before any search: a size the function
    cannot take is said now, not hours into a run."""
    from .size_questions import size_label
    from .watch import run_sizes
    count = len(points["sizes"])
    print(f"checking {count} size{'s' if count != 1 else ''} ...", flush=True)
    results = run_sizes(model_file)
    failed = [(points["sizes"][i], why) for i, (ok, why) in enumerate(results) if not ok]
    if failed:
        setting, why = failed[0]
        # every size failing, the app's own call among them, is a function that
        # cannot be replayed once the app has run (a server's callback into its
        # closed event loop), not a size it cannot take
        hint = ("\nThe search replays the marked function from the arguments the app passed, after the app "
                "has run. Mark a function that takes plain data (text, numbers, arrays, files, the model) and "
                "returns the request's result, not one handed live parts of the running app (a callback, "
                "a queue, an event loop); if there is none, move the request's work into one and mark that"
                if len(failed) == count else "")
        raise IntentError(f"{len(failed)} of {count} sizes did not run; first: "
                          f"{size_label(setting, points['knobs']) or 'as recorded'}: "
                          f"{why}{hint}")
    print(f"  all {count} ran")


def init(argument: str, given: dict, out: Path, interactive: bool, show_questions: bool = False) -> int:
    """`backcountry init`: work out what the argument is, ask what only the
    user knows (or take it from flags), say the plan, write backcountry.yaml."""
    target = classify(argument)
    if given.get("script_args") and target.kind != "app_script":
        raise IntentError(f"arguments after -- are a script's own, and {argument} is not a script init watches")
    if target.kind == "app_script":
        return _init_watched(target, given, out, interactive, show_questions)
    if target.kind in ("manifest", "intent"):
        raise IntentError(f"{argument} is already a {'manifest' if target.kind == 'manifest' else 'backcountry.yaml'}: "
                          f"run it with `backcountry check {argument}` then `backcountry start {argument}`")
    if target.kind in ("hub_model", "local_model") and not is_language_model(target.config):
        raise IntentError(f"{target.value} is not a text model MLX-LM runs. For any other model (image, audio, "
                          "vision, your own): give init a script that runs it the way your app does "
                          "(`backcountry init my_app.py`); init watches it and asks which part to make faster")
    if show_questions:
        print(json.dumps({"target": {"kind": target.kind, "value": target.value, "note": target.note},
                          "questions": questions(target)}, indent=1))
        return 0
    if target.note:
        print(f"note: {target.note}")
    answers = answer(target, given, terminal_ask if interactive else None)
    return _confirm_and_write(target, answers, out, interactive)
