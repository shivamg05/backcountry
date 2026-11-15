"""The front door: a model id and how it is used, in place of a model file and
a manifest. `backcountry run mlx-community/Qwen3-8B-4bit --use chat` writes both
into the run folder, every inferred value commented, and runs them: they are
the record of what ran and the escape hatch (edit them, run the manifest).

Covers language models as MLX-LM loads them: the uses are what a chat app does
with one (read a prompt, write a reply, or both, in one job). Any other model:
`backcountry run my_model.py`, a model file whose examples() returns real inputs
(write_from_examples): the manifest's shapes and sizes are read from them.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

USES = ("chat", "prompt", "reply")
# a checkpoint published to run beside a model: its name is the model's plus
# one of these (Qwen3.8-27B-4bit -> Qwen3.8-27B-MTP-4bit, gemma-4-26B-A4B-it-4bit
# -> gemma-4-26B-A4B-it-assistant-4bit, LFM2.5-1.2B-Instruct-MLX-4bit -> LFM2.5-1.2B-Instruct-DSpark)
HELPER_WORDS = ("mtp", "draft", "drafter", "assistant", "dflash", "dspark", "eagle", "eagle3")
# the order a loop is offered them in: the model's own multi-token head first,
# then trained drafters, then plain draft models
HELPER_ORDER = ("mtp", "dflash", "dspark", "eagle3", "eagle", "assistant", "drafter", "draft")
DFLASH_AUTHOR = "z-lab"   # publishes DFlash drafters for other authors' models (Qwen3-8B-DFlash-b16)
_SIZE = re.compile(r"(?i)(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])")
# format suffixes after the name: a precision, or MLX for a conversion
_PRECISION = re.compile(r"(?i)-(\d+(\.\d+)?bit|bf16|fp16|fp32|mxfp\d+|nvfp\d+|q\d+|dwq|mlx)$")

MODEL_FILE = '''"""{model_id} as MLX-LM loads it (written by `backcountry run {model_id}`)."""


def build():
    from mlx_lm import load

    model, _ = load("{model_id}")
    return model
'''


def config_of(model_id: str) -> dict:
    """The model's config.json: a local folder's, or the Hugging Face repo's
    (only that file is fetched here; the weights download when the job loads)."""
    local = Path(model_id).expanduser()
    if (local / "config.json").is_file():
        return json.loads((local / "config.json").read_text())
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())


def _split(name: str) -> tuple[str, str]:
    """A checkpoint name without its precision suffixes, and the suffixes."""
    base = name
    while (m := _PRECISION.search(base)):
        base = base[:m.start()]
    return base.lower(), name[len(base):].lower()


def helpers_for(model_id: str, list_models=None) -> list[str]:
    """Helper checkpoints published beside a Hugging Face model, by its author,
    for a generation loop to use (a multi-token head, a drafter): the name is
    the model's plus a helper word, at the model's precision when one is
    published. A local folder, or no answer from the Hub, has none."""
    if Path(model_id).expanduser().exists() or model_id.count("/") != 1:
        return []
    author, name = model_id.split("/")
    base, precision = _split(name)
    try:
        if list_models is None:
            from huggingface_hub import HfApi
            list_models = HfApi().list_models
        names = [m.id.split("/", 1)[1] for m in list_models(author=author, search=base, limit=500)]
    except Exception:
        return []
    found = {}
    for other in names:
        other_base, other_precision = _split(other)
        word = other_base[len(base) + 1:] if other_base.startswith(base + "-") else ""
        if word in HELPER_WORDS:
            found.setdefault(word, []).append((other_precision != precision, other))
    return [f"{author}/{min(options)[1]}" for _, options in sorted(found.items(), key=lambda kv: HELPER_ORDER.index(kv[0]))]


def published_helpers(model_id: str, list_models=None) -> list[str]:
    """Checkpoints a generation loop may use beside a Hugging Face model, in
    the order it is offered them: the author's multi-token head, the drafters
    trained for the model (the author's, then z-lab's DFlash drafters), then
    the author's smallest model of the same family at the same precision, a
    plain draft model (Qwen3-8B-4bit -> Qwen3-0.6B-4bit). None is downloaded
    here; no answer from the Hub gives fewer."""
    if Path(model_id).expanduser().exists() or model_id.count("/") != 1:
        return []
    if list_models is None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return []
        list_models = HfApi().list_models
    author, name = model_id.split("/")
    base, precision = _split(name)
    ranked = [(HELPER_ORDER.index(_split(h.split("/")[1])[0][len(base) + 1:]), h)
              for h in helpers_for(model_id, list_models)]
    try:
        trained = [m.id for m in list_models(author=DFLASH_AUTHOR, search=base, limit=100)]
    except Exception:
        trained = []
    ranked += [(HELPER_ORDER.index("dflash"), repo) for repo in trained
               if repo.split("/")[1].lower().startswith(base + "-") and "dflash" in repo.lower()
               and "gguf" not in repo.lower()]   # llama.cpp's format, which MLX does not load
    helpers = [repo for _, repo in sorted(ranked, key=lambda r: r[0])]
    size = _SIZE.search(base)
    if size:
        prefix, suffix = base[:size.start()], base[size.end():]
        try:
            names = [m.id.split("/", 1)[1] for m in list_models(author=author, search=prefix.rstrip("-_ "), limit=500)]
        except Exception:
            names = []
        smaller = []
        for other in names:
            other_base, other_precision = _split(other)
            match = _SIZE.search(other_base)
            if (match and other_precision == precision and other_base[:match.start()] == prefix
                    and other_base[match.end():] == suffix and float(match.group(1)) < float(size.group(1))):
                smaller.append((float(match.group(1)), f"{author}/{other}"))
        helpers += [min(smaller)[1]] if smaller else []
    return helpers


def write(model_id: str, use: str, work_dir: Path, helpers: list[str] | None = None,
          attempts: int | None = None, prompt_tokens: tuple[int, int] = (32, 2049), optimize_at: int = 512,
          history_tokens: int = 2048, reply_tokens: int = 16, keep: str = "same-output",
          written_by: str | None = None, serves: dict | None = None) -> Path:
    """model.py and manifest.yaml for model_id in work_dir; returns the manifest.
    helpers: checkpoints for the loop beside the model (default: found on the Hub).
    attempts: kernel attempts the search may make (unset: a short run).
    prompt_tokens, optimize_at: the prompt lengths to make faster and the one
    the search optimizes at; history_tokens, reply_tokens: a reply's
    conversation so far and its length. keep: same-output (a faster generation
    loop first, same tokens, then kernels) or same-math (kernels only)."""
    if use not in USES:
        raise ValueError(f"--use must be one of {list(USES)}, got {use!r}")
    if attempts is not None and attempts < 1:
        raise ValueError(f"--attempts must be at least 1, got {attempts}")
    if keep not in ("same-output", "same-math"):
        raise ValueError(f"keep must be same-output or same-math, got {keep!r}")
    low, high = prompt_tokens
    if not 1 <= low < high or not low <= optimize_at <= high:
        raise ValueError(f"prompt lengths must be a range with the optimized one inside it; "
                         f"got {low}-{high}, optimized at {optimize_at}")
    if history_tokens < 1 or reply_tokens < 1:
        raise ValueError("history_tokens and reply_tokens must be at least 1")
    loop = keep == "same-output"
    config = config_of(model_id)
    text = config.get("text_config", config)
    vocab = int(text.get("vocab_size") or config.get("vocab_size"))
    experts = any(text.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    work_dir.mkdir(parents=True, exist_ok=True)
    model_file = MODEL_FILE.format(model_id=model_id)
    if serves:  # a marked function in an engine: its first call installs the result into this model
        model_file += (f"\n\n# The marked function the result is for (backcountry init): its first call installs it\n"
                       f"SCRIPT = {serves['script']!r}\nFUNCTION = {tuple(serves['function'])!r}\n"
                       f"FILE = {serves['file']!r}\nPARTS = {serves['parts']!r}\nLEVEL = 'model'\n")
    (work_dir / "model.py").write_text(model_file)
    prompt = f"""  - name: prompt                # reading a prompt to its first token, {low:,} to {high:,} tokens
    inputs: [{{shape: [1, T], dtype: int32, low: 0, high: {vocab}}}]
"""
    reply = f"""  - name: reply                 # {reply_tokens:,} tokens of a reply, after {history_tokens:,} tokens of conversation
    context: {history_tokens}
    inputs: [{{shape: [1, 1], dtype: int32, low: 0, high: {vocab}}}]
"""
    helpers = (helpers_for(model_id) if helpers is None else helpers) if loop else []
    assets = ("assets:                          # published beside the model to run with it; downloaded for the loop\n"
              + "".join(f"  - {h}\n" for h in helpers)) if helpers else ""
    sizes = f"primary: {{T: {optimize_at}}}\nrange: {{T: [{low}, {high}]}}\n" if use in ("chat", "prompt") else ""
    budget = _budget(attempts, loop=loop)
    identical = ("identical: output                # a faster generation loop first (same tokens), then kernels\n"
                 if loop else "identical: computation           # the same math: faster kernels only\n")
    steps = reply_tokens if use in ("chat", "reply") else 16
    (work_dir / "manifest.yaml").write_text(f"""# Written by `{written_by or f'backcountry run {model_id} --use {use}'}`: edit it and run this file to change anything.
model: model.py
model_type: {"moe-llm" if experts else "llm"}            # from config.json{" (it has experts)" if experts else ""}
baseline: plain
use_library_inference: true      # every workload through MLX-LM's own generation, as an app runs it
{identical}workloads:
{prompt if use in ("chat", "prompt") else ""}{reply if use in ("chat", "reply") else ""}{sizes}{assets}budget: {budget}   # attempts per region, in all{", and at the loop" if loop else ""}
final_benchmark: {{steps: {steps}, pairs: 4, warmup_steps: 3}}
""")
    return work_dir / "manifest.yaml"


def _budget(attempts: int | None, loop: bool) -> str:
    """attempts bound the search; unset: a short run (36 kernel attempts,
    which stranded most regions)."""
    if attempts is not None and attempts < 1:
        raise ValueError(f"--attempts must be at least 1, got {attempts}")
    if attempts:
        return f"{{per_region: {min(25, attempts)}, total: {attempts}" + (", model: 20" if loop else "") + "}"
    return "{per_region: 12, total: 36" + (", model: 6" if loop else "") + "}"


_READ_EXAMPLES = r"""
import importlib.util, json, sys
import mlx.core as mx
spec = importlib.util.spec_from_file_location("autotune_examples", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not callable(getattr(module, "examples", None)):
    sys.exit("the model file has no examples(): a function returning a list of real inputs, one per call")
rows = []
for example in module.examples():
    row = []
    for a in (example if isinstance(example, (list, tuple)) else (example,)):
        a = a if isinstance(a, mx.array) else mx.array(a)
        item = {"shape": list(a.shape), "dtype": str(a.dtype).rsplit(".", 1)[-1]}
        if mx.issubdtype(a.dtype, mx.integer) and a.size:
            item["low"], item["high"] = int(a.min().item()), int(a.max().item()) + 1
        row.append(item)
    rows.append(row)
print("\nEXAMPLES " + json.dumps({"examples": rows, "run": callable(getattr(module, "run", None))}))
"""


def read_examples(model_file: Path) -> dict:
    """The shapes, dtypes and integer ranges of the model file's examples(), read
    in a separate process so the file's imports stay out of the job's."""
    result = subprocess.run([sys.executable, "-c", _READ_EXAMPLES, str(model_file)],
                            text=True, capture_output=True, cwd=model_file.parent)
    if result.returncode:
        raise ValueError(f"reading examples() from {model_file} failed:\n{(result.stderr or result.stdout)[-3000:]}")
    return json.loads(result.stdout.rsplit("\nEXAMPLES ", 1)[1])


def workloads_from(examples: list[list[dict]], name: str) -> tuple[list[dict], dict, list[int]]:
    """Manifest workloads from example calls: one input list, with the one size
    that changes between examples named T; returns (workloads, primary, sizes).
    One size: fixed shapes. Two: a workload at each (a range needs three).
    Three or more: T at the first example's size, the others listed."""
    if not examples or not examples[0]:
        raise ValueError("examples() returned no calls (or a call with no inputs)")
    first = examples[0]
    for i, call in enumerate(examples):
        if [(len(a["shape"]), a["dtype"]) for a in call] != [(len(a["shape"]), a["dtype"]) for a in first]:
            raise ValueError(f"example {i + 1} has other inputs than example 1 (count, rank or dtype): "
                             "every example must be the same kind of call, at other sizes")
    varying = {(i, axis): [call[i]["shape"][axis] for call in examples]
               for i, a in enumerate(first) for axis in range(len(a["shape"]))
               if len({call[i]["shape"][axis] for call in examples}) > 1}
    if len({tuple(v) for v in varying.values()}) > 1:
        raise ValueError("the examples' sizes change in more than one independent way "
                         f"({sorted(varying)}); write a manifest instead (docs/manifest.md)")
    sizes = list(dict.fromkeys(next(iter(varying.values())))) if varying else []

    def inputs(size):
        out = []
        for i, a in enumerate(first):
            shape = [size if (i, axis) in varying else n for axis, n in enumerate(a["shape"])]
            item = {"shape": shape, "dtype": a["dtype"]}
            lows = [call[i]["low"] for call in examples if "low" in call[i]]
            if lows:
                item["low"], item["high"] = min(lows), max(call[i]["high"] for call in examples if "high" in call[i])
            out.append(item)
        return out

    if len(sizes) == 2:
        return [{"name": f"{name}_{n}", "inputs": inputs(n)} for n in sizes], {}, sizes
    if len(sizes) > 2:
        return [{"name": name, "inputs": inputs("T")}], {"T": sizes[0]}, sizes
    return [{"name": name, "inputs": inputs(None)}], {}, []


def write_from_examples(model_file: Path, work_dir: Path, attempts: int | None = None,
                        tolerances: str | None = None, function: str | None = None) -> Path:
    """manifest.yaml in work_dir for a model file with build(), examples() and
    optionally run(); returns the manifest. tolerances: "default" also keeps
    kernels that round differently (within each output type's limit).
    function: the app's function run() calls, which the loop search may
    rewrite (same outputs) before the kernel search."""
    model_file = Path(model_file).resolve()
    read = read_examples(model_file)
    workloads, primary, sizes = workloads_from(read["examples"], re.sub(r"\W", "_", model_file.stem))
    budget = _budget(attempts, loop=function is not None)
    lines = [f"# Written by `backcountry run {model_file.name}` from its examples(): edit it and run this file to change anything.",
             f"model: {model_file}",
             "model_type: other",
             *([f"identical: output                # a faster loop in {function.rpartition(':')[2]} first "
                "(same outputs), then kernels", f"function: {json.dumps(function)}"] if function else []),
             "workloads:" + ("                       # the inputs of run(), as its examples pass them" if read["run"]
                             else "                       # the model's inputs, as its examples pass them")]
    for w in workloads:
        lines.append(f"  - name: {w['name']}")
        lines.append("    inputs:")
        for a in w["inputs"]:
            ints = f", low: {a['low']}, high: {a['high']}" if "low" in a else ""
            lines.append(f"      - {{shape: [{', '.join(str(n) for n in a['shape'])}], dtype: {a['dtype']}{ints}}}")
    if primary:
        lines.append(f"primary: {{T: {primary['T']}}}            # the first example's size: the search optimizes for it")
        lines.append(f"range: {{T: {{sizes: [{', '.join(str(n) for n in sorted(sizes))}]}}}}   # every example's size: "
                     "checked, and served where faster")
    lines.append(f"budget: {budget}   # attempts per region and in all")
    if read["run"]:
        lines.append("final_benchmark: {steps: 1, pairs: 4, warmup_steps: 3}   # run() is one whole use")
    if tolerances:
        lines.append(f"tolerances: {tolerances}             # kernels may round differently, within each output type's limit")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "manifest.yaml").write_text("\n".join(lines) + "\n")
    return work_dir / "manifest.yaml"


def is_model_file(argument: str) -> bool:
    """Whether `backcountry run`'s argument is a model file (build() and examples())."""
    return argument.endswith(".py") and Path(argument).expanduser().is_file()


def is_model_id(argument: str) -> bool:
    """Whether `backcountry run`'s argument names a model rather than a manifest file."""
    path = Path(argument).expanduser()
    if path.suffix in (".yaml", ".yml") or path.is_file():
        return False
    return (path / "config.json").is_file() or (argument.count("/") == 1 and not path.exists())
