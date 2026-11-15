"""Serve shipped kernels at every size of one dimension, after the search.

The search optimizes one recorded size. For `serve: {dim: [lo, hi]}` the job
also records the model at the ends and middle of that range, and then, for
each scope carrying shipped kernels:

1. Emits the scope's replay at every recorded size and turns the integers
   that change with the size into expressions of it (bind/sized.py). A scope
   whose replays differ in any other way keeps its searched delivery.
2. Checks the generalized replay against the original module at every size in
   the range: both are built as MLX graphs from the same arguments, and the
   graphs must be the same operations, attributes, constants and sharing,
   outputs and state alike (no GPU work; kernels off).
3. Checks each kernel against its recorded operations at every size a pass
   of the model produces, on inputs recorded from a real pass there, in the
   sandbox (shader validation, poisoned outputs, repeated runs).
4. Times each kernel against its operations on a grid of sizes, then at
   random sizes inside the range the grid picked; a kernel is served over its
   correct sizes around the optimized one, out to the last grid point on each
   side before one it does not win at, and never past a size it lost at.

The scope's generated class then replaces its searched wrapper: sizes in the
scope's range run the replay, and each kernel runs only inside its own range.
Everything else passes to the original module.
"""

from __future__ import annotations

import contextlib
import re
import textwrap
from types import SimpleNamespace
from dataclasses import dataclass, field

import mlx.core as mx

from backcountry_runtime import graph_native, kernels as runtime_kernels
from backcountry_runtime.graph import _is_holder, _python_snapshot, _rewind_python, _state_arrays
from backcountry_runtime.swap import UNSET, ReplayWrapper, flatten_arrays, recorded_form

from .bind.emit import NotReplayable, Splice, scope_nodes
from .bind.sized import NotSizeGeneric, SizedWrapper, _fit, _size_argument, call_size, sized_wrapper
from .trace.serialize import node_to_dict


@dataclass
class ScopePlan:
    path: str
    workload: str
    address: str
    sized: SizedWrapper
    splices: list[Splice]              # at the primary size
    kernel_shapes: dict                # kernel id -> per input [[dim or [a, b]], dtype], a * n + b
    kernel_settings: dict = field(default_factory=dict)  # kernel id -> [[path, a, b]] in its recorded ops
    checked: dict = field(default_factory=dict)   # size -> None (same calculation) or why not
    to_dim: tuple | None = None        # (a, b): the manifest dim is a * size + b

    def served(self, exact: bool) -> tuple[int, ...]:
        """The checked sizes the scope serves: for a range, the run around the
        primary size out to the last passing checked size on each side (a
        scope's size can step by more than one per step of the manifest dim,
        4 rows per token for 4 experts, and the sizes between never occur);
        for exact sizes, every one that passed. Nothing when the primary size
        itself did not pass."""
        primary = self.sized.sizes[0]
        if self.checked.get(primary, "unchecked") is not None:
            return ()
        if exact:
            return tuple(sorted(n for n, why in self.checked.items() if why is None))
        sizes = sorted(self.checked)
        lo = hi = sizes.index(primary)
        while lo > 0 and self.checked[sizes[lo - 1]] is None:
            lo -= 1
        while hi + 1 < len(sizes) and self.checked[sizes[hi + 1]] is None:
            hi += 1
        return tuple(range(sizes[lo], sizes[hi] + 1))


def _corresponding(primary_trace, primary_scope, trace, scope):
    """Array ids and seqs of one scope call in another recording, matched
    node by node; the op sequence must be the same."""
    a, b = scope_nodes(primary_trace, primary_scope), scope_nodes(trace, scope)
    if len(a) != len(b) or any(x.op != y.op for x, y in zip(a, b)):
        raise NotSizeGeneric("the scope runs different operations at another size")
    ids, seqs = {}, {}
    for x, y in zip(a, b):
        seqs[x.seq] = y.seq
        for p, q in zip(x.in_arrays + x.out_arrays, y.in_arrays + y.out_arrays):
            if ids.setdefault(p, q) != q:
                raise NotSizeGeneric("the scope's dataflow differs at another size")
    return ids, seqs


def _corresponding_cuts(primary_trace, primary_scope, trace, scope, splices) -> list[Splice]:
    """Each splice in another recording of a scope whose operations differ
    there: the same occurrence of the cut's own operation sequence, matched
    node by node. Raises when a cut's operations occur a different number of
    times, so no occurrence is the same one."""
    a, b = scope_nodes(primary_trace, primary_scope), scope_nodes(trace, scope)
    ops_a, ops_b = [x.op for x in a], [y.op for y in b]
    out = []
    for s in splices:
        first = next(i for i, x in enumerate(a) if x.seq == s.start_seq)
        last = next(i for i, x in enumerate(a) if x.seq == s.end_seq)
        pattern = ops_a[first:last + 1]
        starts_a, starts_b = _occurrences(ops_a, pattern), _occurrences(ops_b, pattern)
        if len(starts_a) != len(starts_b):
            raise NotSizeGeneric("a kernel's operations occur a different number of times at another size")
        start = starts_b[starts_a.index(first)]
        ids, seqs = {}, {}
        for x, y in zip(a[first:last + 1], b[start:start + len(pattern)]):
            seqs[x.seq] = y.seq
            for p, q in zip(x.in_arrays + x.out_arrays, y.in_arrays + y.out_arrays):
                if ids.setdefault(p, q) != q:
                    raise NotSizeGeneric("a kernel's dataflow differs at another size")
        out.append(Splice(s.kernel, seqs[s.start_seq], seqs[s.end_seq], tuple(ids[i] for i in s.input_ids),
                          tuple(ids[i] for i in s.output_ids), s.fingerprint))
    return out


def _occurrences(ops: list, pattern: list) -> list[int]:
    n = len(pattern)
    return [i for i in range(len(ops) - n + 1) if ops[i] == pattern[0] and ops[i:i + n] == pattern]


def _live_sized(recorded, path: str) -> SizedWrapper:
    """A live scope's size per recording, read off its arguments as a replay's
    would be; no replay is generated (its own method serves every size)."""
    expr, _containers, dim = _size_argument(recorded)
    sizes = [call_size(trace, scope, expr, dim) for trace, scope, _ in recorded]
    if len(set(sizes)) < 3:
        raise NotSizeGeneric("generalizing needs recordings at three or more distinct sizes")
    kernel_ids = list(dict.fromkeys(s.kernel.kernel_id for s in recorded[0][2]))
    return SizedWrapper(path, "", kernel_ids, expr, dim, sizes, 0)


def follows_size(trace, scope, others) -> str | None:
    """Why a scope's replay cannot follow the size over the other recordings
    (None when it can): it must be called at every one, with the same
    operations and dataflow. What the serve step needs before anything else."""
    from .bind.certify import scope_call_by_address
    for other in others:
        other_scope = scope_call_by_address(other, scope.address)
        if other_scope is None:
            return "the scope is not called at every recorded size"
        try:
            _corresponding(trace, scope, other, other_scope)
        except NotSizeGeneric as error:
            return str(error)
        except NotReplayable as error:
            # at another size the scope's calls can interleave with others
            # (Kokoro's weighted convolutions, ops not contiguous there): that
            # region cannot follow the size; check went down on the raise
            return f"at another size, {error}"
    return None


def _calls(node_dicts) -> list:
    return [{k: d[k] for k in ("op", "args", "kwargs", "kernel_definition")} for d in node_dicts]


def _settings(calls: list, sizes) -> list | None:
    """Where one kernel's recorded operations differ between sizes only in
    integers that follow the size (a reshape to the sequence length, say):
    [[path, a, b]], the integer at path being a * n + b. None when anything
    else differs."""
    found = []

    def walk(values, path):
        first = values[0]
        if all(v == first for v in values):
            return True
        if all(type(v) is int for v in values):
            fit = _fit(values, sizes)
            if fit is not None:
                found.append([path, *fit])
            return fit is not None
        if all(isinstance(v, list) and len(v) == len(first) for v in values):
            return all(walk([v[i] for v in values], path + [i]) for i in range(len(first)))
        if all(isinstance(v, dict) and v.keys() == first.keys() for v in values):
            return all(walk([v[k] for v in values], path + [k]) for k in first)
        return False

    return found if walk(calls, []) else None


def _dim_fit(values, sizes):
    """a, b with the manifest dim's value = a * size + b at every recorded
    size. a may be a fraction: gpt-oss's expert layers see 4 rows per token."""
    from fractions import Fraction
    (n0, v0), (n1, v1) = (sizes[0], values[0]), (sizes[1], values[1])
    a = Fraction(v1 - v0, n1 - n0)
    b = v0 - a * n0
    return (a, b) if all(a * n + b == v for n, v in zip(sizes, values)) else None


def plan_scopes(installed: dict, traces: dict, serve_traces: dict, workloads, dims,
                live=frozenset()) -> tuple[list, dict]:
    """A plan per scope that generalizes; why the others do not. dims: the
    manifest dim's value at the primary recording and at each serve trace.
    live: scopes whose own method calls their kernels (live delivery). Such
    a scope runs its own code at every size, so it needs no replay that
    follows the size, and its operations may differ between sizes (a mask
    past a sliding window): only each kernel's own operations must recur."""
    plans, refused = [], {}
    for path, (_original, variants, _kernels) in installed.items():
        reasons = []
        for (workload, address), splices in variants.items():
            if not splices or workload not in workloads:
                continue
            trace = traces[workload]
            scope = next(sc for sc in trace.scope_calls if sc.address == address)
            recorded = [(trace, scope, splices)]
            try:
                for other in serve_traces[workload]:
                    other_scope = next((sc for sc in other.scope_calls if sc.address == address), None)
                    if other_scope is None:
                        raise NotSizeGeneric("the scope is not called at every recorded size")
                    try:
                        ids, seqs = _corresponding(trace, scope, other, other_scope)
                    except NotSizeGeneric:
                        if path not in live:
                            raise
                        recorded.append((other, other_scope, _corresponding_cuts(trace, scope, other,
                                                                                other_scope, splices)))
                        continue
                    recorded.append((other, other_scope, [
                        Splice(s.kernel, seqs[s.start_seq], seqs[s.end_seq],
                               tuple(ids[a] for a in s.input_ids), tuple(ids[a] for a in s.output_ids),
                               s.fingerprint) for s in splices]))
                sized = _live_sized(recorded, path) if path in live else sized_wrapper(recorded, path)
            except (NotSizeGeneric, NotReplayable) as error:
                reasons.append(f"{address}: {error}")
                continue
            shapes, settings = _kernel_shapes(recorded, sized.sizes)
            plans.append(ScopePlan(path, workload, address, sized, list(splices), shapes, settings,
                                   to_dim=_dim_fit(dims, sized.sizes)))
            break
        else:
            refused[path] = "; ".join(reasons) or "no shipped kernel at a served workload"
    return plans, refused


def _kernel_shapes(recorded, sizes) -> tuple[dict, dict]:
    """Each kernel's input shapes as exact integer functions of the scope's
    size, and the integer settings of its recorded operations that follow the
    size (the sandbox check replays those operations at every size with them).
    A kernel whose operations differ between sizes in any other way gets None."""
    shapes, settings = {}, {}
    for i, splice in enumerate(recorded[0][2]):
        per_size, calls = [], []
        for trace, _scope, splices in recorded:
            s = splices[i]
            specs = trace.span_specs(s.start_seq, s.end_seq)
            per_size.append([specs[a] for a in s.input_ids])
            calls.append(_calls(node_to_dict(n) for n in trace.nodes[s.start_seq:s.end_seq + 1]))
        kid = splice.kernel.kernel_id
        found = _settings(calls, sizes)
        if found is None or settings.get(kid, found) != found:
            shapes[kid] = None  # its operations take a setting that does not follow the size
            continue
        settings[kid] = found
        template = []
        for j, (shape, dtype) in enumerate(per_size[0]):
            dims = []
            for d in range(len(shape)):
                values = [p[j][0][d] for p in per_size]
                fit = values[0] if len(set(values)) == 1 else _fit(values, sizes)
                dims.append(list(fit) if isinstance(fit, tuple) else fit)
            if None in dims or any(len(p[j][0]) != len(shape) or p[j][1] != dtype for p in per_size):
                template = None
                break
            template.append([dims, dtype])
        if shapes.get(kid, template) != template:
            template = None  # two scopes size one kernel differently
        shapes[kid] = template
    return shapes, settings


# -- 2: the replay against the original at every size ------------------------

class _Enough(Exception):
    """Every scope was checked on this pass; the rest of the model need not run."""


@contextlib.contextmanager
def _kernels_off():
    saved = runtime_kernels.try_sized, runtime_kernels.try_call
    runtime_kernels.try_sized = runtime_kernels.try_call = lambda spec, inputs, **kwargs: None
    try:
        yield
    finally:
        runtime_kernels.try_sized, runtime_kernels.try_call = saved


def _holders(value, out):
    if isinstance(value, (tuple, list)):
        for v in value:
            _holders(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _holders(v, out)
    elif _is_holder(value) and value not in out:
        out.append(value)
    return out


def size_reader(sized: SizedWrapper, module):
    """The generated wrapper's own size read, as a function of its arguments
    (None for a call the wrapper would hand to the module itself)."""
    match = re.search(r"^    def __call__\(self(?:, )?(?P<sig>[^\n]*)\):\n"
                      r"(?P<bind>(?:        if _kw[^\n]*\n(?:            [^\n]*\n)*)?)"
                      r"        (?P<read>_n = [^\n]*)\n", sized.template, re.M)
    bind = re.sub(r"return self\.wrapped\([^\n]*\)", "return None", match["bind"])
    namespace = {"mx": mx, "_recorded_form": recorded_form, "_UNSET": UNSET,
                 "self": SimpleNamespace(wrapped=module)}
    body = textwrap.indent(textwrap.dedent(bind + f"        {match['read']}\n"), "    ")
    exec(f"def size({match['sig']}):\n{body}    return _n\n", namespace)
    return namespace["size"]


class _Pass:
    """One run of the model: which scopes were checked, and a stop once all were."""

    def __init__(self, paths):
        self.waiting = set(paths)

    def reached(self, path):
        self.waiting.discard(path)
        if not self.waiting:
            raise _Enough


class _Check(ReplayWrapper):
    """Stands in for one scope during the check: builds the candidate's graph
    and the original's from the same arguments, compares them, and returns
    the original's result so the model runs on as before."""

    def __init__(self, original, candidate, plan: ScopePlan, bounds):
        super().__init__(original)
        object.__setattr__(self, "_check", (candidate, plan, size_reader(plan.sized, original), bounds))
        object.__setattr__(self, "run", None)

    def __call__(self, *args, **kwargs):
        candidate, plan, size, bounds = self._check
        n = size(*args, **kwargs)
        if n is None or not bounds[0] <= n <= bounds[1]:
            return self.wrapped(*args, **kwargs)
        if n not in plan.checked:
            holders = _holders((args, kwargs), [])
            saved = _python_snapshot((args, kwargs))
            with _kernels_off():
                ours = flatten_arrays(candidate(*args, **kwargs)) + _state_arrays([vars(h) for h in holders])
            _rewind_python(saved)
            result = self.wrapped(*args, **kwargs)
            theirs = flatten_arrays(result) + _state_arrays([vars(h) for h in holders])
            same, why = graph_native.same_computation(theirs, ours)
            plan.checked[n] = None if same else why
        else:
            result = self.wrapped(*args, **kwargs)
        self.run.reached(plan.path)
        return result


def check_replays(model, plans: list[ScopePlan], candidates: dict, run_at, sizes, install, uninstall):
    """Drive the model once per size with every planned scope checked in
    place, stopping each run once every scope was reached (nothing is
    evaluated by then). run_at(size) runs the model on inputs of that size;
    install and uninstall swap modules at a path."""
    checks, occupants = [], []
    for plan in plans:
        check = _Check(candidates[plan.path].wrapped, candidates[plan.path], plan,
                       (min(plan.sized.sizes), max(plan.sized.sizes)))
        occupants.append((plan.path, install(model, plan.path, check)))
        checks.append(check)
    try:
        for size in sizes:
            run = _Pass(plan.path for plan in plans)
            for check in checks:
                object.__setattr__(check, "run", run)
            try:
                run_at(size)
            except _Enough:
                pass
    finally:
        for path, occupant in reversed(occupants):
            uninstall(model, path, occupant)


# -- 4: where a kernel is served -----------------------------------------------

def served_range(correct: list[int], clock: dict[int, dict], primary: int,
                 grid=None, wrong=None) -> list[tuple[int, int]]:
    """Every stretch of correct sizes the kernel serves: each run from a timed
    size it wins at to the next one it wins at, with every size between
    correct, broken by a size it is clearly slower at or a grid point it does
    not win at. clock: size -> {"win", "loss"} for the grid and the random
    samples; a sample that only ties is ignored. The primary size earns its
    place like any other (the search no longer ships on it alone); a kernel
    not correct there serves nowhere.
    With wrong (the sizes checked and found wrong), the sizes between need
    only include no wrong one: every size a pass of the model produces was
    checked, and the others never occur (a layer with 8 rows per token sees
    only every 8th size)."""
    from .bind.sized import runs
    grid = set(clock) if grid is None else set(grid)
    ok = set(correct)
    if primary not in ok:
        return []
    wins = {n for n, row in clock.items() if row.get("win") and n in ok}
    stops = {n for n, row in clock.items() if row.get("loss") or (n in grid and n not in wins)}
    marks = sorted(wins | stops)
    served = set(wins)
    bad = sorted(wrong) if wrong is not None else None
    for a, b in zip(marks, marks[1:]):
        if not (a in wins and b in wins):
            continue
        if bad is None and all(n in ok for n in range(a, b + 1)) or \
                bad is not None and not any(a <= n <= b for n in bad):
            served.update(range(a, b + 1))
    return [tuple(run) for run in runs(served)]


def in_dim(plan_or_map, sizes):
    """Served sizes of a scope as runs in the manifest dim's units."""
    from .bind.sized import runs
    fit = plan_or_map.to_dim if isinstance(plan_or_map, ScopePlan) else plan_or_map
    if not sizes or fit is None:
        return None
    a, b = fit
    return runs(int(v) for v in (a * n + b for n in sizes) if v == int(v))
