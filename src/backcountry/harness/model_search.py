"""The inference-loop search (identical: output): search the model's inference loop, run(model, *inputs, helpers),
for the same outputs, faster.

The judge proposes a whole run(); the harness checks its outputs against the
untouched model's on fresh inputs, times it against the current best on more
fresh inputs, and a reviewer reads every candidate that wins. Winners stack:
each next proposal starts from the current best. At the end the best run() is
checked against the job's default at several conversation and reply lengths,
timed on inputs the search never saw, and written to run.py beside the report.

For text generation, outputs are token ids and they must equal the untouched
model's plain greedy decoding exactly. A loop that reads several generated
tokens in one call (checking guesses) can meet that because the harness
computes such calls with one-token arithmetic per token (exact_rows); a probe
at the start measures whether that holds on this model and what it costs.
Other outputs must match within the job's tolerance.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from backcountry_runtime.generation import DEFAULT as GENERATION, model_signature
from backcountry_runtime.functions import TARGET, default_source, resolve, target_of
from backcountry_runtime.tune import digest, is_marked
from backcountry_runtime.model_run import APP_RUN, FORWARD, History, ModelRun, _read, load_loop, run_helpers

from .judge import model_level
from . import resume as resume_mod
from .front_door import published_helpers
from .resume import save_loop_search
from .judge.principles import PRINCIPLES
from .measure.clocks import compare_on_inputs
from .workload import context_tokens, workload_seeds

CHECK_SETS = 3      # inputs every candidate's outputs are checked on
TIMING_SETS = 8     # fresh inputs per timing, two per block (plus one to warm on)
HOLDOUT_SETS = 6    # the final timing's inputs, never seen during the search
SPREAD_SETS = 2     # the final check's inputs per conversation and reply length
PROBE_WIDTHS = (2, 3, 4, 5, 6, 8)  # tokens per call a loop might read at once
PROBE_STEPS, PROBE_ROUNDS = 5, 3   # calls per timed loop (after one not counted), paired rounds per width
MAX_HISTORY = 8192
# The calls a generate_step is checked under (MLX-LM's generation): greedy,
# a seeded sampler, a repetition penalty, a prompt read in small chunks.
# Every step's token and logprobs must match MLX-LM's own under each, so
# whatever sampler or processors a user passes gets the same output.
# the last two stop reading part way, as a caller does at an end-of-reply token (the
# Qwen3.8-27B loop left one guessed token more in the cache there, and a chat's next turn differed)
CALLS = ({}, {"greedy": True}, {"temp": 0.8, "top_p": 0.95, "seed": 11}, {"repetition_penalty": 1.1},
         {"prefill_step_size": 128}, {"greedy": True, "stop_after": 7},
         {"temp": 0.8, "top_p": 0.95, "seed": 11, "stop_after": 12})
# how a generate_step is timed: greedy passed as a sampler function, as mlx-lm's
# server and engines pass it (a loop that only sped up sampler=None ran 1.00x in
# LM Studio's mlx-engine, where Qwen3.8-27B's measured 1.42x)
TIMED_CALL = {"greedy": True}
PREFILL_MARGIN = 0.01  # a generate_step may not read a prompt slower by more than this share (the final's margin)
SOURCE_CHARS = 60_000  # model source shown to the judge
ASSET_SOURCE_CHARS = 80_000  # per drafter: the mlx-vlm code that loads and runs it
TRANSPORT_FAILURES = 3
# what a loop may keep in memory beside the model's weights and any model it
# loads (a drafter), as a share of those weights: tables and fused biases fit;
# a second copy of a projection does not (gpt-oss's merged q/k/v kept 4%)
KEPT_SHARE = 0.01


class ModelSearch:
    def __init__(self, runner, manifest=None, check_against=None):
        """manifest: the job's own, when the runner is the kernel job that ran
        on its translation: the loop's inputs come from the job's workload.
        check_against: the model the default loop's reference outputs come
        from, when not the untouched one: the loop with the kernel job's
        kernels is checked against the default loop with them (a kernel that
        changes values within the job's tolerance was certified by that job),
        and timed against the untouched model."""
        self.runner = runner
        self.log = runner.log
        self.session = runner.session
        self.manifest = manifest or runner.manifest
        step = runner.model
        self.model = getattr(step, "model", step)          # with any kernels the job installed
        base = runner.baseline_model
        self.original = getattr(base, "model", base)        # untouched
        self.checked_against = getattr(check_against, "model", check_against) or self.original
        # the workload the loop runs on: the conversation, when the job has one
        self.workload = next((w for w in self.manifest.workloads if w.context is not None),
                             self.manifest.workloads[0])
        # A workload with a context is one more token after a conversation:
        # the loop generates final_benchmark.steps tokens from there, with the
        # conversation already in memory when its clock starts. The model
        # file's run() is the loop it starts from when there is one; it is
        # handed the whole conversation and reads it itself.
        app_run = getattr(runner.model_module, "run", None)
        # a named function the app calls (an engine's generation): what the search
        # rewrites, called through the model file's run(), the app's recipe
        self.target = getattr(self.manifest, "function", None)
        if self.target is not None and app_run is None:
            raise ValueError(f"function {self.target} needs a run() in the model file that calls it as the app does")
        self.conversation = self.workload.context is not None
        self.history = self.conversation and app_run is None
        self.text = bool(runner.use_library_inference) or self.history or \
            (self.target is not None and runner.text_tokenizer is not None)
        # MLX-LM's generation: the loop is its generate_step, called as MLX-LM's
        # callers call it (a prompt, or a conversation's last token after it)
        self.generation = self.text and app_run is None
        self.default_source = default_source(self.target) if self.target else \
            APP_RUN if app_run else GENERATION if self.generation else FORWARD
        self.calls = CALLS if self.generation else ({},)
        # the same generate_step reads prompts: checked (and timed at the end) on
        # the job's prompt workload, read from no cache
        self.prefill = next((w for w in self.manifest.workloads if w.context is None), None) \
            if self.generation and self.conversation else None
        # the tokenizer the models beside this one must share: the job's, or for the app's own
        # function the one it reaches (its text was counted in it), its outputs still its own
        tokenizer = runner.text_tokenizer
        marked = self.target is not None and is_marked(self.target)
        if tokenizer is None and marked:
            from backcountry_runtime.sizes import tokenizer_for
            fn, (args, kwargs) = self.model._fn, self.model._calls[0]
            tokenizer = tokenizer_for(fn, args, kwargs)
        self.helpers = {"steps": self.manifest.final_benchmark.steps, "tokenizer": tokenizer,
                        "app_run": app_run, "conversation": self.history, "text": self.text}
        self.assets, self.assets_note = [], "none: assets are listed for text generation only"
        if self.text or (marked and tokenizer is not None):
            # none named: the ones published for the model, its own multi-token
            # head first, then drafters trained for it, then a smaller sibling
            named = self.manifest.assets or tuple(published_helpers(repo_of(tokenizer)))
            missing = fetch_assets(named)
            self.assets, self.assets_note = find_assets(tokenizer, named=named)
            unusable = [r for r in named if r not in missing and r not in {a["repo"] for a in self.assets}]
            whose = "named in the manifest" if self.manifest.assets else "published for the model"
            self.assets_note += "".join([
                f"; not downloaded (the Hub did not answer): {', '.join(missing)}" if missing else "",
                f"; {whose} but not listed (another tokenizer, or not smaller than the model): "
                f"{', '.join(unusable)}" if unusable else ""])
        self.helpers["assets"] = self.assets
        # a loop gets the tokenizer where it ships only when the bundle can pin it
        self.tokenizer_pin = tokenizer_pin(tokenizer)
        if self.tokenizer_pin is None:
            self.helpers["tokenizer"] = None
        self.default = load_loop(self.default_source, "default_run")
        self._references: dict[str, object] = {}
        self._stock = _stock_ops(self.original, self.model)
        self._read_s = None   # the default's conversation read, seconds (what a loop's own read is charged against)
        self.probe = None
        self._profile = None

    # -- inputs and references ------------------------------------------------

    def _sets(self, purpose: str, n: int, context: int | None = None) -> list[list[mx.array]]:
        """n input sets; for a workload with a context, each set's first input
        is the whole conversation (its context, then the workload's token),
        read into memory per loop by _inputs."""
        seeds = workload_seeds(self.manifest.seed, f"{self.workload.name}:output:{purpose}", n)
        sets = [self.runner._materialize(self.workload, self.manifest.primary, s) for s in seeds]
        if self.conversation:
            from dataclasses import replace
            workload = self.workload if context is None else replace(self.workload, context=context)
            sets = [[mx.concatenate([
                context_tokens(workload, self.manifest.primary, seed, self.runner.text_tokenizer).reshape(1, -1),
                inputs[0].reshape(1, -1).astype(mx.int32)], axis=1)] for seed, inputs in zip(seeds, sets)]
        return sets

    def _inputs(self, run, model, inputs, helpers=None):
        """The inputs run() is called with: for a conversation, it read into
        memory (run's own read() when it defines one), off any clock.
        Returns (inputs, seconds the read took)."""
        if not (helpers or self.helpers).get("conversation"):
            return inputs, 0.0
        mx.synchronize()
        start = time.perf_counter()
        history = History.read(model, inputs[0], run_helpers(run, model, helpers or self.helpers),
                               run.__dict__.get("read"))
        mx.synchronize()
        return [history, *inputs[1:]], time.perf_counter() - start

    def _key(self, inputs, steps) -> str:
        h = hashlib.sha256(str(steps).encode())
        for x in inputs:
            h.update(np.asarray(x).tobytes())
        return h.hexdigest()

    def _helpers(self, steps=None) -> dict:
        return self.helpers if steps is None else {**self.helpers, "steps": steps}

    def _prefill_sets(self, purpose: str, n: int) -> list[list[mx.array]]:
        """n prompts of the job's prompt workload, for a generate_step to read from no cache."""
        seeds = workload_seeds(self.manifest.seed, f"{self.prefill.name}:output:{purpose}", n)
        return [self.runner._materialize(self.prefill, self.manifest.primary, s) for s in seeds]

    def _prefill_helpers(self) -> dict:
        """A prompt read to its first token, as MLX-LM's callers read one."""
        return {**self.helpers, "conversation": False, "steps": 1}

    def _outputs(self, run, model, inputs, helpers) -> list:
        """run's outputs on one input set under each checked call: a
        conversation read once, each call on a fresh copy of it."""
        own, _ = self._inputs(run, model, inputs, helpers)
        outs = [_host(ModelRun(model, run, {**helpers, "call": call})(*own)) for call in self.calls]
        del own
        _free()
        return outs

    def reference(self, inputs, steps=None, helpers=None) -> list:
        """The untouched model's outputs through the job's default loop, one per checked call."""
        helpers = helpers or self._helpers(steps)
        key = self._key(inputs, (helpers["steps"], helpers.get("conversation")))
        if key not in self._references:
            self._assert_stock()
            self._references[key] = self._outputs(self.default, self.checked_against, inputs, helpers)
        return self._references[key]

    def _assert_stock(self) -> None:
        """References and checks run on the library as installed: a candidate
        that left a patch behind (a matmul, attention, a layer's call) would
        otherwise change the reference the same way it changed itself. A
        patch found is undone, and the check that found it fails."""
        changed = [name for name, fn in _stock_ops(self.original, self.model, names=self._stock).items()
                   if fn is not self._stock[name]]
        if changed:
            for name in changed:
                _set_op(name, self._stock[name], self.original, self.model)
            raise LibraryPatched(f"a loop left the library or the model's classes patched ({sorted(changed)[:5]}); "
                                 "a loop must not patch them")

    def profile(self, run) -> dict:
        """Where one use of the loop spends its time, for the judge: see
        profile_calls. One extra run on an input of its own, never a check or
        timing input."""
        own, _ = self._inputs(run, self.model, self._sets("profile", 1)[0])
        step = ModelRun(self.model, run, self.helpers)
        parts = {k: v for k, v in step._helpers.items() if isinstance(v, nn.Module)}
        model = self.model
        if self.target is not None and is_marked(self.target):   # the model the function calls, not the function
            model = next((p for p in model.parts if isinstance(p, nn.Module)), model)
        return profile_calls(model, lambda: step(*own), parts)

    # -- the probe --------------------------------------------------------------

    def measure_probe(self) -> dict:
        """Whether a call that reads w generated tokens at once gives each token
        exactly its one-token call's numbers (logits and every cache field)
        under the harness's exact rows, and what it costs against one token,
        stock and exact, at the job's own conversation length."""
        from backcountry_runtime.exact_rows import generating
        from backcountry_runtime.state import _copy_cache, _make_cache, _with_cache_work
        inputs = self._sets("probe", 1)[0]
        tokens = inputs[0].reshape(1, -1)
        new = mx.array([_token_ids(self.reference(inputs)[0])])
        if new.shape[1] < max(PROBE_WIDTHS):
            new = mx.concatenate([new, tokens[:, :max(PROBE_WIDTHS)]], axis=1)
        model = self.model
        if self.history:
            base = _read(model, tokens)
            mx.eval(_with_cache_work(model(tokens[:, -1:], cache=base), base))
        else:
            base = _make_cache(model)
            mx.eval(_with_cache_work(model(tokens, cache=base), base))
        from backcountry_runtime.exact_rows import _offset
        start = _offset(base)
        if start is None:
            return {"widths": [], "note": "the model's cache has no position (no attention layer with a "
                                          "KV cache), so several-token calls cannot be made exact here"}

        def arrays(cache):
            """Every array the cache holds: its state where it defines one
            (up to its offset), else every array among its fields."""
            from backcountry_runtime.swap import flatten_arrays
            return flatten_arrays([getattr(c, "state", None) if hasattr(type(c), "state") else vars(c)
                                   for c in cache])

        def call(w, exact):
            c = _copy_cache(base)
            if exact:
                with generating(model, start):
                    out = model(new[:, :w], cache=c)
            else:
                out = model(new[:, :w], cache=c)
            mx.eval(out, arrays(c))
            return out, c

        steps = PROBE_STEPS + 1
        stream = mx.concatenate([new] * -(-steps * max(PROBE_WIDTHS) // new.shape[1]), axis=1)

        def loop_ms(w, exact):
            """What one w-token call costs as a loop makes it: consecutive
            calls on one copy of the conversation, each waiting for its
            result; the median call, the first one not counted. A call timed
            alone, from a fresh copy, read 1.3-1.7x one token for 2 tokens
            where these read 1.05x (Qwen3.8-27B)."""
            c = _copy_cache(base)
            mx.eval(arrays(c))
            times = []
            for i in range(steps):
                x = stream[:, i * w:(i + 1) * w]
                mx.synchronize()
                t = time.perf_counter()
                if exact:
                    with generating(model, start):
                        out = model(x, cache=c)
                else:
                    out = model(x, cache=c)
                mx.eval(out)
                times.append(time.perf_counter() - t)
            return statistics.median(times[1:]) * 1e3

        def paired(w):
            """w tokens against one token, stock and exact, alternated so a
            paging spike or a drifting clock lands on both: the median of
            PROBE_ROUNDS paired ratios each."""
            ratios, ones = {False: [], True: []}, []
            for _ in range(PROBE_ROUNDS):
                ones.append(loop_ms(1, False))
                for exact in (False, True):
                    ratios[exact].append(loop_ms(w, exact) / ones[-1])
            return statistics.median(ratios[False]), statistics.median(ratios[True]), statistics.median(ones)

        rows, ones = [], []
        for w in PROBE_WIDTHS:
            got, c = call(w, True)
            ref_cache = _copy_cache(base)
            ref = mx.concatenate([model(new[:, i:i + 1], cache=ref_cache) for i in range(w)], axis=1)
            mx.eval(ref, arrays(ref_cache))
            exact = bool(mx.array_equal(got, ref).item()) and all(
                a.shape == b.shape and bool(mx.array_equal(a, b).item())
                for a, b in zip(arrays(c), arrays(ref_cache)))
            del got, c, ref, ref_cache
            stock_x, exact_x, one_ms = paired(w)
            ones.append(one_ms)
            rows.append({"tokens": w, "exact": exact, "stock_x": round(stock_x, 2), "exact_x": round(exact_x, 2)})
            _free()
        one = statistics.median(ones)
        return {"one_token_ms": round(one, 2), "widths": rows,
                "note": ("cost of one call that reads w generated tokens, in one-token calls, each timed "
                         "as a loop makes them (consecutive calls on one cache): stock MLX (not exact) and "
                         "with the harness's exact rows, which every such call gets")}

    # -- the output check -----------------------------------------------------

    def rescore(self, prompt: mx.array, tokens: list[int]) -> list[tuple[float, bool]]:
        """Per token: how far below the untouched model's own top choice it
        is, and whether it is that choice, with the model reading the prompt
        the way plain generation does (every token but the last, then the last
        alone) and then the tokens one at a time. A diagnostic only."""
        prompt = prompt.reshape(1, -1)
        c = _read(self.original, prompt)
        logits = self.original(prompt[:, -1:], cache=c)[:, -1]
        scores = []
        for t in tokens:
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            scores.append((float((lp.max() - lp[0, t]).item()), int(mx.argmax(lp, axis=-1).item()) == t))
            logits = self.original(mx.array([[t]]), cache=c)[:, -1]
        return scores

    def check(self, run, sets, steps=None, helpers=None) -> dict:
        """Outputs of run() against the untouched model's on each input set,
        under each checked call: text must be the same tokens (for a
        generate_step, also every step's logprobs, its progress calls and
        the cache it leaves), other outputs within tolerance."""
        helpers = helpers or self._helpers(steps)
        name = "generate_step" if self.generation else "run()"
        for i, inputs in enumerate(sets):
            refs = self.reference(inputs, helpers=helpers)
            try:
                outs = self._outputs(run, self.model, inputs, helpers)
                self._assert_stock()
            except Exception as error:  # a candidate's own failure is its verdict
                try:  # a patch it left before failing is undone too, before any later check
                    self._assert_stock()
                except LibraryPatched as patched:
                    error = patched
                return {"ok": False, "input": i, "reason": f"{name} raised {type(error).__name__}: {error}"[:600]}
            for call, out, ref in zip(self.calls, outs, refs):
                problem = self._differs(inputs, out, ref, call)
                if problem:
                    return {"ok": False, "input": i, **({"call": call} if call else {}), **problem}
        return {"ok": True}

    def _check_all(self, run, sets) -> dict:
        """check() on the loop's input sets, then, for a generate_step beside a
        prompt workload, on prompts read from no cache."""
        result = self.check(run, sets)
        if result["ok"] and self.prefill is not None:
            result = self.check(run, self._prefill_sets("check", CHECK_SETS), helpers=self._prefill_helpers())
            if not result["ok"]:
                result["case"] = "a prompt read from no cache, to its first token"
        return result

    def _differs(self, inputs, out, ref, call=None) -> dict | None:
        if not self.text:
            problem = _values_differ(out, ref, self.manifest)
            return {"reason": problem} if problem else None
        under = f" (with {call})" if call else ""
        ref_tokens, tokens = _token_ids(ref), _token_ids(out)
        if len(tokens) != len(ref_tokens):
            return {"reason": f"returned {len(tokens)} tokens{under}; the default returns {len(ref_tokens)}"}
        first = next((k for k, (a, b) in enumerate(zip(tokens, ref_tokens)) if a != b), None)
        if first is not None:
            gap, _ = self.rescore(inputs[0], tokens[:first + 1])[first]
            return {"position": first, "reason": (
                f"token {first} differs from the default's{under} ({tokens[first]} vs {ref_tokens[first]}); the "
                f"untouched model rates it {gap:.3f} below its own choice there. Every token must equal the "
                "default's: pick each token as mlx-lm does, read the prompt as the default does, and read "
                "generated tokens with the model's own call")}
        if not isinstance(ref, dict):
            return None
        if not isinstance(out, dict):
            return {"reason": "returned no logprobs; generate_step yields (token, logprobs) per step"}
        lp, ref_lp = out.get("logprobs"), ref["logprobs"]
        if lp is None or lp.shape != ref_lp.shape or not np.array_equal(lp, ref_lp):
            if lp is None or lp.shape != ref_lp.shape:
                return {"reason": f"yielded logprobs of shape {None if lp is None else list(lp.shape)}{under}; "
                                  f"the default yields {list(ref_lp.shape)}"}
            diff = np.abs(lp.astype(np.float64) - ref_lp.astype(np.float64))
            step = int(np.argmax(diff.reshape(diff.shape[0], -1).max(axis=1)))
            return {"position": step, "reason": (
                f"the tokens match, but step {step}'s logprobs differ from the default's{under} (largest "
                f"difference {float(diff.max()):.3g}). Every step's logprobs must be bit-identical, so any "
                "sampler a caller passes picks the same tokens: compute them as the default does")}
        if not np.array_equal(out.get("progress"), ref["progress"]):
            return {"reason": f"prompt_progress_callback was called with {np.asarray(out.get('progress')).tolist()}"
                              f"{under}; the default calls it with {ref['progress'].tolist()}"}
        if not np.array_equal(out.get("cache_offsets"), ref["cache_offsets"]):
            return {"reason": (
                f"the prompt cache ends at positions {np.asarray(out.get('cache_offsets')).tolist()[:4]}{under}; the "
                f"default leaves {ref['cache_offsets'].tolist()[:4]}. A caller reuses the cache (mlx_lm.server does): "
                "leave it as the default does, which also runs the step after the last token it yields"
                + (f"; a caller may stop reading after any token (here after {call['stop_after']}, as at an "
                   "end-of-reply token): the cache must then hold exactly the tokens it read"
                   if call and call.get("stop_after") else ""))}
        if not np.array_equal(out.get("cache_digest"), ref.get("cache_digest")):
            return {"reason": (
                f"the prompt cache ends at the default's positions{under}, but its contents differ from the "
                "default's. A chat continues the next turn from that cache (mlx_lm.chat, mlx_lm.server): leave "
                "every entry, recurrent states included, exactly as the default does, also when the caller "
                "stops reading early")}
        return None

    # -- the search -----------------------------------------------------------

    def _briefing(self, history, best_source, left) -> dict:
        from .measure.peaks import gpu_core_count, gpu_facts
        chip = {"gpu_cores": gpu_core_count(), **gpu_facts()}
        if self.runner.peaks is not None:
            chip.update(bandwidth_gbps=self.runner.peaks.bandwidth_gbps, launch_us=self.runner.peaks.launch_us)
        return {
            "job": {
                "model_type": self.manifest.model_type,
                "target": self.target,
                "inputs": [[list(map(str, s.shape)), s.dtype] for s in self.workload.inputs],
                "sizes": dict(self.manifest.primary),
                "outputs": (GENERATION_OUTPUTS if self.generation else
                            "token ids: helpers['steps'] tokens of greedy generation after the conversation "
                            "(every token of the input but the last), from its last token" if self.history else
                            "token ids: helpers['steps'] tokens of greedy generation after the prompt"
                            if self.text else "the default run()'s outputs, same structure"),
                "check": (GENERATION_CHECK if self.generation else
                          "every token must equal the untouched model's plain greedy decoding, exactly"
                          if self.text else "every output value within the job's numeric tolerance"),
                "calls_checked": list(self.calls) if self.generation else None,
            },
            "helpers": {
                "steps": self.helpers["steps"],
                "note": GENERATION_HELPERS if self.generation else None,
                "tokenizer": "the model's tokenizer" if self.helpers.get("tokenizer") is not None else None,
                # helpers['assets'] is this list itself (a judge given {"listed": ...} indexed it so)
                "assets": self.assets if self.text or self.assets else None,
                "assets_note": self.assets_note if self.text or self.assets else None,
                "asset_sources": {a["repo"]: _drafter_source(a) for a in self.assets if a.get("drafter")}
                if self.text or self.assets else None,
                "app_run": "the model file's run(model, *inputs)" if self.helpers.get("app_run") else None,
                "cache": ("the model's own cache (mlx-lm's make_prompt_cache), already holding every token of "
                          "the input but the last, read before the clock starts; continue from it. A source "
                          "that defines read(model, tokens, helpers) -> dict reads the conversation itself "
                          "instead (its dict must hold 'cache'; its other entries reach run() through "
                          "helpers); any time it takes beyond the default read is added to its timed time"
                          if self.history else None),
            },
            "exact_rows": self.probe,
            "model_file": Path(self.manifest.model_path).read_text()[:SOURCE_CHARS],
            "model_source": _source_of(self.original),
            "default_run": (_generate_step_source() if self.generation else
                            _target_sources(self.target) if self.target else self.default_source),
            "current_best": best_source,
            "current_best_profile": self._profile,
            "history": history,
            "budget": {"attempts_left": left},
            "chip": chip,
            "memory": self._memory(),
            "model_principles": model_level.principles(),
            "hardware_principles": [{"topic": t, "principle": p} for t, p, _ in PRINCIPLES
                                    if t in ("hardware", "performance")],
        }

    def resident(self):
        """The model's weights held in memory for the whole search, as mlx-lm
        holds them while it generates (wired_limit): conversation reads and the
        probe run outside any timed call, and a model this Mac can only just
        hold is otherwise paged out between calls."""
        import contextlib
        try:
            from mlx_lm.generate import generation_stream, wired_limit
        except ImportError:
            return contextlib.nullcontext()
        return wired_limit(self.model, [generation_stream])

    def _memory(self) -> dict:
        """What the GPU can hold beside the model: anything a loop loads
        (a drafter, a head) and its caches must fit, or its runs fail."""
        from .measure.session import memory_state
        state = memory_state()
        limit, used = state["gpu_limit_gb"], state["mlx_active_gb"]
        return {"gpu_limit_gb": round(limit, 2), "in_use_gb": round(used, 2),
                "room_gb": round(limit - used, 2),
                "note": "the GPU's working-set limit and what the job holds now (the model, its caches); a "
                        "helper's weights plus its caches must fit in room_gb with margin for the model's "
                        "own working memory"}

    def _stop(self) -> str | None:
        """Why the search stops before its next attempt: `backcountry finish`
        was run; else None (its attempts bound it)."""
        if self.runner._finish_requested():
            return "operator requested final validation"
        return None

    def run(self) -> dict:
        with self.resident():
            return self._run()

    def _run(self) -> dict:
        budget = self.manifest.budget_model
        self.log.append("model_search", budget=budget, assets=self.assets_note,
                        outputs="text" if self.text else "values")
        check_sets = self._sets("check", CHECK_SETS)
        for inputs in check_sets:  # every reference before any candidate runs
            self.reference(inputs)
        if self.prefill is not None:
            for inputs in self._prefill_sets("check", CHECK_SETS):
                self.reference(inputs, helpers=self._prefill_helpers())
        if self.history:  # what a loop's own conversation read is charged against
            self._read_s = statistics.median(self._inputs(self.default, self.model, s)[1] for s in check_sets)
        if self.text:
            self.probe = self.measure_probe()
            self.log.append("model_search_probe", probe=self.probe)
        judge = self.runner.judge_factory(None)
        best, best_source = self.default, self.default_source
        self._profile = self._default_profile = self.profile(best)
        self.log.append("model_search_profile", profile=self._profile)
        history, failures, accepted = [], 0, []
        saved = getattr(self.runner, "_loop_search", None)
        if saved:  # a job stopped mid-search: its attempts stand, the best loop is loaded again
            history, accepted = saved["history"], saved["accepted"]
            if saved["best_source"] != best_source:
                best, best_source = load_loop(saved["best_source"], "resumed_best"), saved["best_source"]
                self._profile = self.profile(best)
            self.log.append("resumed", at=f"loop search attempt {len(history)}", accepted=accepted)
        attempt = len(history)
        while attempt < budget:
            # `backcountry pause` stops here, its finished attempts saved; unread, the
            # request paused Qwen3.8-27B's kernel search hours after a killed loop search
            if any(resume_mod.pause_requested(d) for d in self.runner._request_dirs()):
                # its position saved, also before the first attempt: a resume continues here
                save_loop_search(self.runner.work_dir, {"history": history, "accepted": accepted,
                                                        "best_source": best_source,
                                                        "active_s": round(self.runner._active_s(), 1)})
                raise resume_mod.Paused(f"pause requested (loop search, before attempt {attempt})")
            stop = self._stop()
            if stop:
                self.log.append("model_search_stopped", at=attempt, reason=stop)
                break
            try:
                reply = model_level.propose(judge, self._briefing(history, best_source, budget - attempt),
                                            generation=self.generation, target=self.target)
                failures = 0
            except ValueError as error:  # an answer the harness refused twice: an attempt spent
                history.append({"attempt": attempt, "verdict": "no_proposal", "detail": str(error)[:400]})
                self.log.append("model_attempt", attempt=attempt, verdict="no_proposal", detail=str(error)[:400])
                attempt += 1
                continue
            except (RuntimeError, OSError) as error:
                # no answer (no network, an expired login, a reply over the time
                # limit): no attempt spent, and as in the kernel search three in
                # a row pause the job where a resume asks again
                failures += 1
                self.log.append("model_judge_error", attempt=attempt, detail=str(error)[:400])
                if failures >= TRANSPORT_FAILURES:
                    raise resume_mod.Paused(f"the judge did not answer {TRANSPORT_FAILURES} times in a row; "
                                            f"last error: {error}")
                continue
            row = {"attempt": attempt, "change": reply["kind"], "hypothesis": reply["hypothesis"][:1200]}
            weights = tree_flatten(self.model.parameters())  # references: what an unkept loop's setup must leave
            _free()
            held = mx.get_active_memory()
            try:
                if self.target and target_of(reply["source"]) is None:  # the target line, as the briefing asks
                    reply["source"] = f"{TARGET}{self.target}\n{reply['source']}"
                run = load_loop(reply["source"], f"candidate_{attempt}")
            except Exception as error:
                row.update(verdict="failed", detail=f"does not load: {type(error).__name__}: {error}"[:600])
            else:
                result = self._check_all(run, check_sets)
                if result["ok"] and (kept := kept_beside(run, self.model, held, weights)) is not None:
                    result = {"ok": False, "reason": kept}
                if not result["ok"]:
                    row.update(verdict="failed", detail=result)
                else:
                    try:
                        comp, timed = self.time(best, run, f"attempt-{attempt}")
                    except Exception as error:  # a candidate's own failure (out of memory too) is its verdict
                        comp, timed = None, {"reason": f"timing raised {type(error).__name__}: {error}"[:600]}
                    if timed is not None:
                        row.update(verdict="failed", detail=timed)
                    else:
                        speedup = 1.0 / comp.median_ratio if comp.median_ratio > 0 else None
                        row.update(speedup_vs_best=speedup, best_ms=comp.median_baseline_ms,
                                   candidate_ms=comp.median_baseline_ms - comp.median_delta_ms)
                        if comp.wins_by(0.0):
                            verdict = self.review(judge, reply["source"], best_source)
                            if verdict["verdict"] == "accept":
                                best, best_source = run, reply["source"]
                                accepted.append(attempt)
                                self._profile = self.profile(best)
                                self.log.append("model_search_profile", profile=self._profile)
                                row.update(verdict="accepted", review=verdict["reason"])
                            else:
                                row.update(verdict="rejected_by_review", detail=verdict["reason"])
                        else:
                            row.update(verdict="slower" if comp.loses_by(0.0) else "not_faster")
            row["source"] = reply["source"] if row["verdict"] in ("failed", "accepted") else None
            history.append(row)
            if row["verdict"] != "accepted":  # free what it loaded (a drafter) before the next one
                run = None
                # and the model's own weights: a setup() may replace them (Gemma 4's
                # attempt 9 left 2.66 GB of re-laid-out experts, so the final ran out of memory)
                self.model.update(tree_unflatten(weights), strict=False)
                _free()
            weights = None
            self.log.append("model_attempt", **{k: v for k, v in row.items() if k != "source"})
            save_loop_search(self.runner.work_dir, {"history": history, "accepted": accepted,
                                                    "best_source": best_source,
                                                    "active_s": round(self.runner._active_s(), 1)})
            attempt += 1
        result = {"budget": budget, "attempts": len(history), "accepted": accepted, "assets_listed": self.assets,
                  "steps": self.helpers["steps"], "probe": self.probe, "profile": self._profile,
                  "default_profile": self._default_profile, "conversation": self.history, "text": self.text,
                  "tokenizer": self.tokenizer_pin,
                  "history": [{k: v for k, v in h.items() if k != "source"} for h in history]}
        if accepted:
            result["final"] = self.final(best)
            if result["final"]["ok"]:
                (self.runner.work_dir / "run.py").write_text(best_source)
                result["run_py"] = str(self.runner.work_dir / "run.py")
                result["source"] = best_source
                if self.target and is_marked(self.target):   # the marked function's code it was made from
                    result["written_as"] = digest(self.default_source.split("\n", 1)[1])
        # what the bundle pins: the listed models the winning loop read
        result["assets"] = [a for a in self.assets if a["repo"] in best.__dict__.get("assets_read", ())]
        self.runner.report.model_level = result
        self.log.append("model_search_done", **{k: v for k, v in result.items() if k not in ("history", "source")})
        return result

    def _save_check(self, best, sets) -> None:
        """Inputs and the best run()'s outputs on them, for the artifact's
        fresh-process check that its run.py reproduces them exactly."""
        arrays, calls = {}, []
        cases = [(inputs, call) for inputs in sets for call in self.calls[:2]]  # greedy, and sampled
        for k, (inputs, call) in enumerate(cases):
            for j, x in enumerate(inputs):
                arrays[f"in{k}_{j}"] = x
            own, _ = self._inputs(best, self.model, inputs)
            leaves = [v for _, v in tree_flatten(ModelRun(self.model, best, {**self.helpers, "call": call})(*own))]
            if not all(isinstance(v, mx.array) for v in leaves):
                raise ValueError("run() returns values other than arrays, which the bundle's fresh-process "
                                 "check cannot compare")
            arrays.update({f"out{k}_{i}": v for i, v in enumerate(leaves)})
            calls.append(call)
        mx.save_safetensors(str(self.runner.work_dir / "model_level_check.safetensors"), arrays)
        self.check_calls = calls

    def time(self, best, candidate, purpose: str, sets=None, base_model=None, helpers=None):
        """best (on base_model, by default the job's model) against candidate
        (on the job's model), each with its own conversation reads off the
        clock; a loop's own read is charged what it takes beyond the default
        read. Returns (comparison, None) or (None, why its timed outputs
        differed from the other arm's)."""
        sets = sets or self._sets(f"time:{purpose}", TIMING_SETS + 1)
        helpers = helpers or self.helpers
        if self.generation and "call" not in helpers:
            helpers = {**helpers, "call": TIMED_CALL}
        warm, sets = sets[0], sets[1:]
        base_model = self.model if base_model is None else base_model
        arms = {"a": (best, base_model), "b": (candidate, self.model)}
        reads: dict[str, dict] = {"a": {}, "b": {}}  # arm -> input key -> seconds its read took
        outs: dict[str, dict] = {"a": {}, "b": {}}
        current = {}

        def prepare(arm, inputs):
            run, model = arms[arm]
            _free()  # off the clock: the last call's caches must not crowd this one's
            own, seconds = self._inputs(run, model, inputs, helpers)
            current[arm] = self._key(inputs, None)
            reads[arm][current[arm]] = seconds
            return own

        def recorded(arm):
            run, model = arms[arm]
            step = ModelRun(model, run, helpers)

            def call(*inputs):
                out = step(*inputs)
                if self.text:
                    outs[arm][current[arm]] = _token_ids(out)
                return out
            return call

        charged = {}

        def charge_ms(arm):
            """A loop's own read is charged what it takes beyond the default
            read: per input, against the other arm's read of the same input
            moments apart when that arm reads the default way (paired, so a
            machine's drift cancels), else against the default read measured
            at the start; the median over the inputs."""
            run, _ = arms[arm]
            if run.__dict__.get("read") is None or not reads[arm]:
                return 0.0
            other = "b" if arm == "a" else "a"
            if arms[other][0].__dict__.get("read") is None:
                extra = [reads[arm][k] - reads[other][k] for k in reads[arm] if k in reads[other]]
            else:
                extra = [] if self._read_s is None else [t - self._read_s for t in reads[arm].values()]
            # only extra time is charged: a read no slower than the default's costs nothing
            charged[arm] = max(0.0, round(1e3 * statistics.median(extra), 1)) if extra else 0.0
            return charged[arm]

        comp = compare_on_inputs(self.session, recorded("a"), recorded("b"), sets, warm=warm,
                                 prepare=prepare, charge_ms=charge_ms)
        if charged:
            self.log.append("model_read_charge", purpose=purpose, charged_ms=charged)
        try:
            self._assert_stock()
        except LibraryPatched as error:
            return None, {"reason": str(error)}
        for key, tokens in outs["b"].items():
            if outs["a"].get(key) not in (None, tokens):
                return None, {"reason": "a timed run's tokens differ from the other arm's on the same input"}
        return comp, None

    def review(self, judge, candidate: str, current: str) -> dict:
        try:
            return model_level.review(judge, candidate, current, {
                "contract": "generate_step" if self.generation else "run",
                "outputs": (GENERATION_OUTPUTS if self.generation else
                            "token ids of greedy generation" if self.text else "the default run()'s outputs"),
                "check": "outputs compared with the untouched model's on fresh inputs, then timed on more "
                         "fresh inputs; every call gets inputs no earlier call saw"})
        except (ValueError, RuntimeError, OSError) as error:
            return {"verdict": "reject", "reason": f"the review could not be completed: {error}"[:400]}

    def _spread(self) -> list[tuple[int | None, int, list]]:
        """The final check's cases: (conversation length, reply length, input
        sets) around the job's own, so a loop is checked beyond the one
        length it was searched at: half, the same and twice the conversation
        (up to MAX_HISTORY), each with the job's reply length and twice it."""
        steps = self.helpers["steps"]
        contexts = ([None] if not self.conversation else
                    sorted({max(8, self.workload.context // 2), self.workload.context,
                            min(MAX_HISTORY, 2 * self.workload.context)}))
        return [(c, s, self._sets(f"spread:{c}:{s}", SPREAD_SETS, context=c))
                for c in contexts for s in (steps, 2 * steps)]

    def final(self, best, base_model=None) -> dict:
        """The best run() against the job's default: outputs checked at
        several conversation and reply lengths on inputs the search never saw,
        then timed (the default on the untouched model, the best on the
        model as the job built it)."""
        cases = []
        for context, steps, sets in self._spread():
            result = self.check(best, sets, steps=None if steps == self.helpers["steps"] else steps)
            cases.append({"context": context, "steps": steps, "sets": len(sets), "ok": result["ok"]})
            if not result["ok"]:
                return {"ok": False, "reason": "the final output check failed", "detail": result, "cases": cases}
        if self.prefill is not None:
            result = self.check(best, self._prefill_sets("final-check", SPREAD_SETS), helpers=self._prefill_helpers())
            cases.append({"context": None, "steps": 1, "prompt": self.prefill.name, "sets": SPREAD_SETS,
                          "ok": result["ok"]})
            if not result["ok"]:
                return {"ok": False, "reason": "the final output check failed reading a prompt", "detail": result,
                        "cases": cases}
        pairs = max(self.manifest.final_benchmark.pairs, 4)
        holdout = self._sets("final-time", 2 * pairs + 1)
        comp, timed = self.time(self.default, best, "final", sets=holdout,
                                base_model=self.original if base_model is None else base_model)
        if timed is not None:
            return {"ok": False, "reason": "the final timing's outputs differed", "detail": timed, "cases": cases}
        speedup = 1.0 / comp.median_ratio if comp.median_ratio > 0 else None
        self._save_check(best, holdout[:2])
        final = {"ok": comp.wins_by(0.0), "speedup": speedup, "default_ms": comp.median_baseline_ms,
                 "best_ms": comp.median_baseline_ms - comp.median_delta_ms, "sigma_ms": comp.sigma_ms,
                 "n": comp.n, "cases": cases, "inputs": len(holdout) - 1,
                 # what each timed call did: the job's conversation length (the only one timed) and reply length
                 "context": self.workload.context if self.conversation else None,
                 "steps": self.helpers["steps"]}
        final["check_calls"] = self.check_calls  # the settings of the saved cases the bundle's check replays
        if self.generation or self.target:
            # a user's model gets the loop only if it is this one (a marked function's: the model it reaches)
            final["made_for"] = model_signature(self.model.parts[0] if self.target and is_marked(self.target)
                                                else self.model)
        if self.prefill is not None and final["ok"]:
            # the same generate_step reads prompts: no slower there than MLX-LM's
            prompts = self._prefill_sets("final-time", 2 * pairs + 1)
            read, timed = self.time(self.default, best, "final-prefill", sets=prompts,
                                    base_model=self.original if base_model is None else base_model,
                                    helpers=self._prefill_helpers())
            if timed is not None:
                return {**final, "ok": False, "reason": "the final prompt timing's outputs differed", "detail": timed}
            final["prefill"] = {"tokens": prompts[0][0].shape[-1],
                                "speedup": 1.0 / read.median_ratio if read.median_ratio > 0 else None,
                                "default_ms": read.median_baseline_ms,
                                "best_ms": read.median_baseline_ms - read.median_delta_ms,
                                "sigma_ms": read.sigma_ms, "n": read.n}
            if read.loses_by(PREFILL_MARGIN * read.median_baseline_ms):
                return {**final, "ok": False, "reason": f"it reads a prompt slower than the default "
                        f"({final['prefill']['best_ms']:.1f} ms vs {read.median_baseline_ms:.1f} ms)"}
        if getattr(self.runner, "_kernels_installed", lambda: False)():
            # what the kernel search's kernels still add once the loop changed how
            # the model is called: the same loop on the untouched and the patched model
            under, timed = self.time(best, best, "final-kernels", sets=self._sets("final-kernels", 2 * pairs + 1),
                                     base_model=self.original)
            if under is not None:
                final["kernels_under_loop"] = {
                    "speedup": 1.0 / under.median_ratio if under.median_ratio > 0 else None,
                    "untouched_ms": under.median_baseline_ms,
                    "patched_ms": under.median_baseline_ms - under.median_delta_ms,
                    "sigma_ms": under.sigma_ms, "n": under.n, "resolved": under.wins_by(0.0)}
        return final


def kept_beside(run, model, before: int, restore=()) -> str | None:
    """Why a loop holds too much memory once its runs are done, or None: what
    it keeps beyond the active memory `before` it loaded, less any model it
    loaded beside this one (a drafter's weights), may be at most KEPT_SHARE
    of the model's weights. More is a second copy of weights the model still
    holds: a merge that keeps the originals. restore: the (name, array) list
    the search holds to put back the weights of a loop it does not keep; an
    original it holds that the model no longer does is the search's memory,
    not the loop's (an in-place merge read as a copy on gpt-oss-20b)."""
    _free()
    current = {id(a) for _, a in tree_flatten(model.parameters())}
    released = sum(a.nbytes for _, a in restore if id(a) not in current)
    grown = mx.get_active_memory() - before - released
    inside = {id(m) for _, m in model.named_modules()} | {id(model)}
    loaded, seen = 0, set()

    def visit(value, depth):
        if id(value) in seen or depth > 4:
            return
        seen.add(id(value))
        if isinstance(value, nn.Module):
            if id(value) not in inside:
                nonlocal loaded
                loaded += sum(a.nbytes for _, a in tree_flatten(value.parameters()))
            return
        if isinstance(value, dict):
            items = value.values()
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            return
        for item in items:
            visit(item, depth + 1)

    visit(dict(run.__dict__), 0)
    visit({k: v for k, v in run.__globals__.items() if not k.startswith("__")}, 0)
    size = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    extra = grown - loaded
    if extra <= KEPT_SHARE * size:
        return None
    return (f"the loop keeps {extra / 1e9:.2f} GB in memory after its runs, beside the model's {size / 1e9:.2f} GB "
            f"of weights (and {loaded / 1e9:.2f} GB of models it loads), more than {KEPT_SHARE:.0%} of them. "
            "Either a second copy of weights the model still holds (merge in place: replace the model's own "
            "arrays with views of the merged one, so one copy stays), or state left behind after a call (a "
            "drafter's or head's cache, arrays held on the model or in module globals): free it when the "
            "call ends")


GENERATION_OUTPUTS = ("what generate_step yields, step by step: each token and its logprobs vector (log-softmax "
                      "of the last position's logits after the logits processors), the prompt_progress_callback "
                      "calls, and the prompt cache it leaves. Called as MLX-LM's callers call it: a conversation's "
                      "last token with prompt_cache already holding the rest (as mlx_lm.server continues one), "
                      "then max_tokens tokens; and a fresh prompt read to its first token")
GENERATION_CHECK = ("every step's token and logprobs bit-identical to MLX-LM's own generate_step's under each "
                    "call in calls_checked (greedy, a seeded temperature/top-p sampler set by mx.random.seed "
                    "before the call, a repetition penalty, a smaller prefill_step_size), the same "
                    "prompt_progress_callback calls and the same cache positions and contents afterwards, also "
                    "when the caller stops reading part way (stop_after, as at an end-of-reply token); at several "
                    "conversation and reply lengths at the end, and on prompts read from no cache")
GENERATION_HELPERS = ("generate_step is called with MLX-LM's arguments (max_tokens is helpers['steps'] in the "
                      "job's calls; prompt_cache holds the conversation). It receives helpers only if it declares "
                      "a helpers argument: these entries plus setup()'s. A call setting an argument outside "
                      "max_tokens, sampler, logits_processors, prompt_cache, prefill_step_size and "
                      "prompt_progress_callback runs MLX-LM's own generate_step instead")


def _target_sources(target: str) -> str:
    """A named function as the judge rewrites it: its source, then the sources
    of the functions it calls by name from outside the standard library, two
    calls deep (LM Studio's create_generator hides its decode loop inside
    mlx-lm's stream_generate and generate_step), within SOURCE_CHARS."""
    import sys
    _, _, original = resolve(target)
    parts, seen, frontier = [], set(), [original]
    for depth in range(3):
        following = []
        for fn in frontier:
            fn = inspect.unwrap(fn)
            if id(fn) in seen or not (inspect.isfunction(fn) or inspect.isclass(fn)):
                continue
            seen.add(id(fn))
            module = getattr(fn, "__module__", "") or ""
            if module.split(".")[0] in sys.stdlib_module_names:
                continue
            try:
                parts.append(f"# {module}.{fn.__qualname__}\n{inspect.getsource(fn)}")
            except (OSError, TypeError):
                continue
            following += _callees(fn)
        frontier = following
    text = (f"# The default: the original {target}, unchanged. A replacement keeps the first line "
            f"\"{TARGET}{target}\" and defines {target.partition(':')[2].split('.')[-1]} with this signature; "
            "the sources of what it calls follow it.\n"
            + ("# It is the app's own function, marked with @backcountry.tune in that file: a replacement runs "
               "with that file's globals (the model, tokenizer and imports the original reads by name); define "
               "any helper it needs in the replacement itself.\n" if is_marked(target) else "")
            + "\n\n".join(parts))
    return text[:SOURCE_CHARS]


def _callees(fn) -> list:
    """What fn calls by name: its module's globals it names, in its own code
    and in functions nested in it, and what it imports inside its body (an
    engine's create_generator imports mlx-lm's generate_step there)."""
    import ast
    import importlib
    import textwrap
    codes, names = [getattr(fn, "__code__", None)], set()
    while codes:
        code = codes.pop()
        if code is not None:
            names.update(code.co_names)
            codes += [c for c in code.co_consts if inspect.iscode(c)]
    found = [fn.__globals__[n] for n in sorted(names) if n in getattr(fn, "__globals__", {})]
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            try:
                module = importlib.import_module(node.module)
            except Exception:
                continue
            found += [getattr(module, a.name) for a in node.names if hasattr(module, a.name)]
    return found


def _generate_step_source() -> str:
    """MLX-LM's generate_step as the judge rewrites it, with what its module provides."""
    import importlib
    lm = importlib.import_module("mlx_lm.generate")  # the module: mlx_lm.generate is also its function
    return ("# The default: `from mlx_lm.generate import generate_step`, MLX-LM's own, unchanged. Its source "
            "follows; mlx_lm.generate also defines generation_stream and maybe_quantize_kv_cache and imports "
            "mlx_lm.models.cache as cache. A replacement defines generate_step with this signature.\n"
            + inspect.getsource(lm.generate_step))


def _token_ids(out) -> list[int]:
    """The token ids of a text loop's outputs (a generate_step's are under "tokens")."""
    tokens = out["tokens"] if isinstance(out, dict) else out
    return [int(t) for t in np.asarray(tokens).reshape(-1)]


def _free() -> None:
    """Return a finished run's memory to the GPU: its caches sit in reference
    cycles that Python's collector frees only when enough Python objects pile
    up (540 KV caches after two timings on Qwen3-8B), and MLX keeps freed
    buffers for reuse. On a model that nearly fills the GPU (Qwen3.8-27B) that
    garbage alone ran timing out of memory."""
    import gc
    gc.collect()
    mx.clear_cache()


class LibraryPatched(RuntimeError):
    """A loop left a library function or a model class's call replaced."""


def _set_op(name: str, fn, *models) -> None:
    """Put back the function _stock_ops found under name."""
    if name.startswith("mx.fast."):
        setattr(mx.fast, name.removeprefix("mx.fast."), fn)
    elif name.startswith("mx."):
        setattr(mx, name.removeprefix("mx."), fn)
    else:
        for model in models:
            for _, m in model.named_modules():
                cls = type(m)
                if f"{cls.__module__}.{cls.__qualname__}.__call__" == name:
                    if fn is None:
                        cls.__dict__.get("__call__") is not None and delattr(cls, "__call__")
                    else:
                        cls.__call__ = fn
                    return


def _stock_ops(*models, names=None) -> dict:
    """The library functions and model-class calls a loop could patch, by name."""
    ops = {f"mx.{n}": getattr(mx, n) for n in ("quantized_matmul", "matmul", "gather_qmm", "addmm", "softmax",
                                                "logsumexp", "argmax")}
    ops.update({f"mx.fast.{n}": getattr(mx.fast, n) for n in ("scaled_dot_product_attention", "rms_norm",
                                                              "layer_norm", "rope")})
    for model in models:
        for _, m in (model.named_modules() if isinstance(model, nn.Module) else ()):
            cls = type(m)
            ops[f"{cls.__module__}.{cls.__qualname__}.__call__"] = cls.__dict__.get("__call__")
    if names is not None:
        ops = {k: v for k, v in ops.items() if k in names}
    return ops


PROFILE_ROWS = 12


def profile_calls(model, call, parts=None) -> dict:
    """One run of call(): every call into model, one of its direct parts (a
    pipeline's encoder, denoiser or decoder), the module holding its layers
    and that module's siblings (a loop may call the layers and the output
    layer directly), or a named helper model, counting only outermost calls so
    nothing is counted twice, grouped by part, input sizes and the position
    the call starts at (its cache's offset). Each call is timed with a wait for
    the GPU after it, so the times add up but overlap between calls is not
    shown; the rest of the run is time outside model calls (Python, sampling,
    small ops the loop runs itself). rows: tokens per call (the second dim of
    its first input), used to hand the loop's calls to the kernel search."""
    import time
    from backcountry_runtime.exact_rows import _offset, layer_owners
    from backcountry_runtime.swap import flatten_arrays
    targets = {id(model): "model"}
    objects = [model]

    def add(name, m):
        if isinstance(m, nn.Module) and id(m) not in targets:
            targets[id(m)] = name
            objects.append(m)

    if isinstance(model, nn.Module):
        for name, child in model.children().items():
            add(name, child)
        paths = {id(m): p for p, m in model.named_modules()}
        for owner in layer_owners(model):
            path = paths.get(id(owner), "")
            add(path or "layers", owner)
            parent = _module_at(model, path.rsplit(".", 1)[0]) if "." in path else model
            for name, sibling in parent.children().items():
                add((path.rsplit(".", 1)[0] + "." if "." in path else "") + name, sibling)
    for name, helper in (parts or {}).items():
        add(name, helper)
    records, depth, saved, compiled = [], [0], {}, set()

    def timed(original):
        def __call__(self, *args, **kwargs):
            if id(self) not in targets or depth[0]:
                return original(self, *args, **kwargs)
            depth[0] += 1
            cache = kwargs.get("cache", args[1] if len(args) > 1 else None)
            offset = _offset(cache) if cache is not None else None  # where the call starts
            mx.synchronize()
            start = time.perf_counter()
            try:
                out = original(self, *args, **kwargs)
                try:
                    mx.eval(flatten_arrays(out))
                except ValueError as error:
                    # called inside mx.compile (mflux compiles its denoiser
                    # step): nothing can be waited on there, so it is not timed
                    if "function transformations" not in str(error):
                        raise
                    compiled.add(targets[id(self)])
                    return out
                mx.synchronize()
            finally:
                depth[0] -= 1
            arrays = flatten_arrays((args, kwargs))
            sizes = ", ".join(f"{list(a.shape)} {str(a.dtype).removeprefix('mlx.core.')}" for a in arrays)
            first = arrays[0] if arrays else None
            rows = int(first.shape[1]) if first is not None and first.ndim >= 2 else None
            records.append((targets[id(self)], sizes, rows, offset, time.perf_counter() - start))
            return out
        return __call__

    for cls in {type(o) for o in objects}:
        saved[cls] = cls.__dict__.get("__call__")
        cls.__call__ = timed(cls.__call__)
    try:
        mx.synchronize()
        start = time.perf_counter()
        mx.eval(flatten_arrays(call()))
        mx.synchronize()
        total = time.perf_counter() - start
    finally:
        for cls, original in saved.items():
            if original is None:
                del cls.__call__
            else:
                cls.__call__ = original
    groups: dict[tuple, list] = {}
    for name, sizes, rows, offset, seconds in records:
        groups.setdefault((name, sizes, rows), []).append((seconds, offset))
    rows = sorted(({"call": name, "inputs": sizes, "rows": r, "count": len(t),
                    "offsets": [min(o for _, o in t if o is not None), max(o for _, o in t if o is not None)]
                    if any(o is not None for _, o in t) else None,
                    "ms_each": round(1e3 * sum(x for x, _ in t) / len(t), 3),
                    "share_of_run": round(sum(x for x, _ in t) / total, 3)} for (name, sizes, r), t in groups.items()),
                  key=lambda r: -r["share_of_run"])
    inside = sum(r[-1] for r in records)
    return {"run_ms": round(1e3 * total, 2), "outside_model_calls_ms": round(1e3 * (total - inside), 2),
            "calls": rows[:PROFILE_ROWS], "parts": sorted(parts or {}),
            "note": "each call timed with a wait after it; overlap is not shown" + (
                f"; not timed, called inside a compiled function (its time is in outside_model_calls_ms): "
                f"{', '.join(sorted(compiled))}" if compiled else "")}


def _module_at(model, path: str):
    m = model
    for part in path.split("."):
        m = m[int(part)] if isinstance(m, list) else getattr(m, part)
    return m


def repo_of(tokenizer) -> str:
    """The Hugging Face id of the model's checkpoint, read from its folder in
    the Hugging Face cache (models--author--name/snapshots/rev); "" for a
    model loaded from anywhere else."""
    path = Path(getattr(tokenizer, "name_or_path", "") or "")
    folder = path.parents[1].name if len(path.parents) > 1 and path.parent.name == "snapshots" else ""
    parts = folder.split("--")
    return "/".join(parts[1:]) if parts[0] == "models" and len(parts) == 3 else ""


def fetch_assets(repos) -> list[str]:
    """Download the manifest's assets into the Hugging Face cache, where
    find_assets lists them beside the model's own. Returns the ones that did
    not download (the search goes on without them). huggingface_hub is
    imported only when there is one: an app with none may not have it."""
    if not repos:
        return []
    from huggingface_hub import snapshot_download
    missing = []
    for repo in repos:
        try:
            snapshot_download(repo)
        except Exception:
            missing.append(repo)
    return missing


def find_assets(tokenizer, named=()) -> tuple[list[dict], str]:
    """Models already on this Mac that a loop could use beside the model: every
    smaller checkpoint in the Hugging Face cache that splits text into the same
    tokens (same vocabulary and merge rules), whatever its layout or type (a
    small model of the family, a multi-token-prediction head, a block-diffusion
    drafter). A drafter that ships no tokenizer (Liquid's DSpark drafters) uses
    the model's: listed when named (the manifest's assets) and its vocabulary
    is the model's size. Listed with its type and size, never loaded: a loop
    that wants one loads it in setup(). Nothing is downloaded. Returns (assets, note)."""
    path = Path(getattr(tokenizer, "name_or_path", "") or "")
    if tokenizer is None or not (path / "config.json").is_file():
        return [], "none: the model's checkpoint folder is unknown"
    mine = _checkpoint(path)
    if mine is None or not mine["tokenizer"]:
        return [], "none: the model's tokenizer files were not found"
    hub = path.parents[2] if len(path.parents) > 2 else None
    found, misfits = [], []
    width = _hidden_size(json.loads((path / "config.json").read_text()))
    for snapshot in sorted(hub.glob("*/snapshots/*") if hub is not None else ()):
        if snapshot.resolve() == path.resolve():
            continue
        other = _checkpoint(snapshot)
        repo = snapshot.parent.parent.name.removeprefix("models--").replace("--", "/")
        if (other and other["vocab"] == mine["vocab"] and 0 < other["bytes"] < mine["bytes"]
                and (_tokens(snapshot) == _tokens(path) if other["tokenizer"] else repo in named)):
            config = json.loads((snapshot / "config.json").read_text())
            drafter = _drafter_module(snapshot, config)
            if _is_head(config, drafter) and None not in (width, _hidden_size(config)) and _hidden_size(config) != width:
                # a head reads the model's hidden states: Qwen3.8-27B's MTP head (5120 wide)
                # was offered for Qwen3.5-9B (4096), and the AI spent 10 of 20 attempts on it
                misfits.append(repo)
                continue
            found.append({"repo": repo, "revision": snapshot.name, "path": str(snapshot),
                          "model_type": config.get("model_type"),
                          "architecture": (config.get("architectures") or [None])[0],
                          "drafter": drafter,
                          "gb": round(other["bytes"] / 1e9, 2),
                          **({} if other["tokenizer"] else {"tokenizer": "none of its own: the model's"})})
    skipped = (f"; skipped as built for another model's width (a head reads the model's hidden states, "
               f"{width} wide here): {', '.join(misfits)}") if misfits else ""
    if not found:
        return [], "none: no smaller model with the same tokenizer is downloaded" + skipped
    return found, (f"{len(found)} smaller checkpoint(s) with the model's tokenizer are downloaded (the model "
                   f"itself is {mine['bytes'] / 1e9:.2f} GB); helpers['assets'] lists them, none is loaded. "
                   "A loop that uses one loads it from its path in setup(model, helpers): mlx_lm.load for "
                   "mlx-lm model types, mlx_vlm.speculative.drafters.load_drafter for heads and drafters "
                   "mlx-vlm knows (an asset's drafter names the mlx-vlm module that loads it; asset_sources "
                   "holds that code and the round loop mlx-vlm runs it with)" + skipped)


def _hidden_size(config: dict) -> int | None:
    """A checkpoint's hidden width (a multimodal one keeps it in text_config)."""
    width = config.get("hidden_size") or (config.get("text_config") or {}).get("hidden_size")
    return width if isinstance(width, int) else None


def _is_head(config: dict, drafter: str | None) -> bool:
    """A checkpoint that reads the model's hidden states rather than running on
    its own: a multi-token-prediction head, or a drafter mlx-vlm loads as one."""
    names = " ".join([str(config.get("model_type") or ""), *map(str, config.get("architectures") or [])]).lower()
    return drafter is not None or "mtp" in names or "draft" in names


def _drafter_module(snapshot: Path, config: dict) -> str | None:
    """The mlx-vlm module that loads this checkpoint as a speculative drafter, or None."""
    try:
        from mlx_vlm.utils import get_model_and_args
        module, _ = get_model_and_args(config, snapshot)
    except Exception:
        return None
    return module.__name__ if module.__name__.startswith("mlx_vlm.speculative.drafters.") else None


def _drafter_source(asset: dict) -> str:
    """What a loop needs to use a drafter mlx-vlm knows: the package that
    loads it, the drafter modules it builds on, and the round loop mlx-vlm
    runs that kind of drafter with (a reference: the loop is the judge's)."""
    import importlib
    import re
    from mlx_vlm.speculative.drafters import resolve_drafter_kind
    parts, seen, queue = [], set(), [asset["drafter"]]
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        module = importlib.import_module(name)
        files = (sorted(Path(module.__file__).parent.glob("*.py")) if hasattr(module, "__path__")
                 else [Path(module.__file__)])
        for file in files:
            text = file.read_text()
            parts.append(f"# {name}: {file.name}\n{text}")
            queue += [f"mlx_vlm.speculative.drafters.{m}" for m in re.findall(r"^from \.\.(\w+)", text, re.M)]
    try:
        kind = resolve_drafter_kind(Path(asset["path"]))
        loop = Path(importlib.import_module(f"mlx_vlm.speculative.{kind}").__file__)
        parts.append(f"# mlx_vlm.speculative.{kind}: the round loop mlx-vlm runs it with\n{loop.read_text()}")
    except Exception:
        pass
    return "\n\n".join(parts)[:ASSET_SOURCE_CHARS]


def tokenizer_pin(tokenizer) -> dict | None:
    """The tokenizer's Hugging Face repo and revision, when it was loaded from
    a snapshot in the cache (a bundle finds it there again, as it finds its
    assets); else None."""
    path = Path(getattr(tokenizer, "name_or_path", "") or "")
    if tokenizer is None or path.parent.name != "snapshots" or not path.parent.parent.name.startswith("models--"):
        return None
    return {"repo": path.parent.parent.name.removeprefix("models--").replace("--", "/"), "revision": path.name}


def _checkpoint(folder: Path):
    try:
        config = json.loads((folder / "config.json").read_text())
        weights = sum(p.stat().st_size for p in folder.glob("*.safetensors"))
        return {"vocab": config.get("vocab_size") or config.get("text_config", {}).get("vocab_size"),
                "bytes": weights, "tokenizer": (folder / "tokenizer.json").is_file()}
    except (OSError, ValueError):
        return None


@functools.lru_cache(maxsize=None)
def _tokens(folder: Path):
    """What decides how a tokenizer splits text: its vocabulary and merge
    rules, merges written either as "a b" or ["a", "b"]. Special tokens the
    other model adds (Qwen3.5's audio markers, absent from Bonsai 2) never
    come out of text."""
    model = json.loads((folder / "tokenizer.json").read_text())["model"]
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model.get("merges", [])]
    return hashlib.sha256(json.dumps([sorted(model.get("vocab", {}).items()), merges]).encode()).hexdigest()


def _source_of(model) -> str:
    try:
        return inspect.getsource(inspect.getmodule(type(model)))[:SOURCE_CHARS]
    except (OSError, TypeError):
        return ""


class _Bfloat16(np.ndarray):
    """A bfloat16 output held as float32 (numpy has no bfloat16), so its
    check still uses bfloat16's tolerance."""


def _as_array(x):
    """An input as an array: a conversation in memory by its tokens."""
    return x.tokens if isinstance(x, History) else x


def _host(out):
    """Outputs as host values, so later work cannot change them."""
    if isinstance(out, mx.array):
        if out.dtype == mx.bfloat16:
            return np.array(out.astype(mx.float32)).view(_Bfloat16)
        return np.array(out)
    if isinstance(out, (list, tuple)):
        return type(out)(_host(o) for o in out)
    if isinstance(out, dict):
        return {k: _host(v) for k, v in out.items()}
    return out


def _values_differ(out, ref, manifest) -> str | None:
    if type(out) is not type(ref) and not (isinstance(out, np.ndarray) and isinstance(ref, np.ndarray)):
        return f"returned {type(out).__name__}; the default returns {type(ref).__name__}"
    if isinstance(ref, dict):
        if set(out) != set(ref):
            return f"returned keys {sorted(out)}; the default returns {sorted(ref)}"
        return next((p for k in ref if (p := _values_differ(out[k], ref[k], manifest))), None)
    if isinstance(ref, (list, tuple)):
        if len(out) != len(ref):
            return f"returned {len(out)} outputs; the default returns {len(ref)}"
        return next((p for a, b in zip(out, ref) if (p := _values_differ(a, b, manifest))), None)
    if isinstance(ref, np.ndarray):
        if out.shape != ref.shape or out.dtype != ref.dtype:
            return f"returned {out.dtype}{list(out.shape)}; the default returns {ref.dtype}{list(ref.shape)}"
        if ref.dtype.kind in "biu":  # token ids, counts, masks: equal or not
            return None if np.array_equal(out, ref) else "an output differs from the untouched model's"
        rtol, atol = manifest.tolerance_for(_dtype_name(ref))
        a, b = out.astype(np.float64), ref.astype(np.float64)
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            return "an output differs from the untouched model's beyond the job's tolerance"
        return None
    return None if out == ref else "an output differs from the untouched model's"


def _dtype_name(a: np.ndarray) -> str:
    return "bfloat16" if isinstance(a, _Bfloat16) else str(a.dtype)


NEW_TOKENS = "new_tokens"  # the dim a decode workload's token count becomes


def kernel_manifest(manifest_path, model_path, level: dict) -> tuple[str, bool]:
    """The kernel search's manifest after the loop search: this one as an
    ordinary kernel job (identical: computation, the loop's budget dropped, the
    model path absolute), aimed at the calls the winning loop spends the model's
    time on (model_level.profile). When most of it goes to calls that read
    several generated tokens at once (a speculative loop's checks), a decode
    workload becomes those calls: [1, new_tokens] over the same conversation,
    generated: true (computed with one-token arithmetic per token, kernels kept
    bit-exact), the token count it spends the most time on as the primary and
    any other as a checked size. Otherwise every workload stays as written.
    Returns (YAML text, whether a workload changed)."""
    import yaml
    raw = yaml.safe_load(Path(manifest_path).read_text())
    raw["model"] = str(Path(model_path).resolve())
    raw["identical"] = "computation"
    raw.pop("assets", None)  # the loop's: the kernels run the model alone
    raw.pop("function", None)  # what the loop search rewrote: the kernels run under the app's own calls
    # the loop's attempts stay the loop's
    raw["budget"] = {k: v for k, v in (raw.get("budget") or {}).items() if k != "model"}
    profile = (level.get("profile") or {}) if level.get("source") else {}
    helpers = set(profile.get("parts", []))
    # the conversation: the workload the loop generates in (any others stay as written)
    spec = next((w for w in raw["workloads"] if w.get("context") is not None), raw["workloads"][0])
    context = spec.get("context")
    widths: dict[int, float] = {}
    one_token = 0.0
    for row in profile.get("calls", []):
        if row["call"] in helpers or not row.get("rows") or context is None:
            continue
        spent = row["count"] * row["ms_each"]
        if row["rows"] == 1:
            one_token += spent
        elif row.get("offsets") and row["offsets"][0] >= context:
            widths[row["rows"]] = widths.get(row["rows"], 0.0) + spent
    changed = bool(widths) and sum(widths.values()) > one_token
    if changed:
        main = max(widths, key=widths.get)
        spec["generated"] = True
        spec["inputs"][0]["shape"] = list(spec["inputs"][0]["shape"][:-1]) + [NEW_TOKENS]
        raw["primary"] = {**(raw.get("primary") or {}), NEW_TOKENS: main}
        for key in ("range", "check_optimizations_for"):
            if isinstance(raw.get(key), dict):
                raw[key].pop(NEW_TOKENS, None)
            if not raw.get(key):
                raw.pop(key, None)
        others = sorted(set(widths) - {main})
        # a listed range: the primary and at least two more, unless another
        # workload's sizes already are the job's range (one dim is served)
        if len(others) >= 2 and not any(raw.get(k) for k in ("range", "check_optimizations_for")):
            raw["range"] = {NEW_TOKENS: {"sizes": others}}
        else:  # checked there too, served where it was recorded
            raw["sweep"] = {**(raw.get("sweep") or {}), NEW_TOKENS: sorted(widths)}
    header = ("# Written by the job after the loop search: the kernel search's manifest, the job's own\n"
              "# aimed at the calls the winning loop makes (model_level.profile).\n")
    return header + yaml.safe_dump(raw, sort_keys=False), changed
