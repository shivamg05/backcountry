"""The static replayability screen and identity certification.

The screen is content-monotone: anything disqualifying inside the smallest
enclosing scope is inside every ancestor too, so one check at the region's own
scope decides "no certified delivery scope".

Identity certification is the measured fact that a scope's replay is
invisible: the identity wrapper (same replay, no kernel change) must produce
bitwise-identical outputs everywhere the scope fires, including across
repeated calls of a stateful step, and its retrace must match the baseline
under the span-map projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn

from ..trace.recorder import OPAQUE_OP, UNKNOWN_SOURCE_OP, UNNAMED_KERNEL_OP
from ..trace.types import Retention, ScopeCall, Trace
from ..trace.walk import flatten_arrays
from ..artifact.validate import _back, _held, _structure
from .emit import MODULE_HEADER, NotReplayable, emit_wrapper, scope_nodes
from backcountry_runtime.exact import bitwise_equal
from backcountry_runtime.swap import ReplayWrapper, install as swap_install, uninstall as swap_uninstall, resolve


_INDEX: dict[int, tuple] = {}   # id(trace) -> (trace, {stack: first call}, {path: calls}, count, {address: first call})


def _scope_index(trace: Trace) -> tuple:
    """A trace's scope calls by stack and by module path, built once: a
    256-token reply recorded 244k calls, and a scan per region took hours."""
    held = _INDEX.get(id(trace))
    if held is None or held[0] is not trace or held[3] != len(trace.scope_calls):
        while len(_INDEX) >= 16:
            _INDEX.pop(next(iter(_INDEX)))
        by_stack, by_path, by_address = {}, {}, {}
        for sc in trace.scope_calls:
            by_stack.setdefault(sc.stack, sc)
            by_path.setdefault(sc.address.rsplit("@", 1)[0], []).append(sc)
            by_address.setdefault(sc.address, sc)
        held = _INDEX[id(trace)] = (trace, by_stack, by_path, len(trace.scope_calls), by_address)
    return held


def find_scope_call(trace: Trace, stack: tuple[str, ...]) -> ScopeCall | None:
    return _scope_index(trace)[1].get(stack)


def scope_call_by_address(trace: Trace, address: str) -> ScopeCall | None:
    return _scope_index(trace)[4].get(address)


def scope_calls_at(trace: Trace, path: str) -> list[ScopeCall]:
    """Every recorded call of the module at path (an address without its @n)."""
    return _scope_index(trace)[2].get(path, [])


def screen_scope(trace: Trace, stack: tuple[str, ...]) -> str | None:
    """None if the scope qualifies, else the named reason it cannot host a
    generated replay wrapper."""
    scope = find_scope_call(trace, stack)
    if scope is None:
        return f"no recorded call for scope {stack!r}"
    depth = len(stack)
    for site in trace.eval_sites:
        if site[:depth] == stack:
            return "the scope evaluates mid-call (data-dependent control flow)"
    try:
        nodes = scope_nodes(trace, scope)
    except NotReplayable as e:
        return str(e)
    for node in nodes:
        if node.op == OPAQUE_OP:
            return "the scope contains a compiled call the harness cannot name"
        if node.op == UNKNOWN_SOURCE_OP:
            return "the scope makes an array the trace cannot explain (an unknown source)"
        if node.op == UNNAMED_KERNEL_OP and node.kernel_definition is None:
            return "the scope contains a custom kernel the harness cannot name"
        for out in node.out_arrays:
            if trace.liveness[out].kind is Retention.PYTHON_RETAINED:
                return (
                    f"the scope produces a python-retained value "
                    f"({node.op!r} at seq {node.seq}); the wrapper can never "
                    f"reach the reference the model kept"
                )
    try:
        emitted = emit_wrapper(trace, scope, [], "_ScreenProbe")
        compile(MODULE_HEADER + emitted.source, "<screen>", "exec")
    except NotReplayable as e:
        return str(e)
    except SyntaxError as e:
        return f"the generated wrapper is not valid Python: {e.msg} at line {e.lineno}"
    return None


@dataclass
class CertificationResult:
    ok: bool
    reason: str = ""
    scope: str | None = None   # the scope whose identity failed, when one is known


def certify_identity(
    build_wrapper: Callable[[], object],
    install: Callable[[object], Callable[[], None]],
    runs: Sequence[Callable[[], object]],
    repeated_calls: int = 3,
) -> CertificationResult:
    """Generic identity certification: capture baseline outputs of every run,
    install the identity wrapper, re-run, compare bitwise, uninstall.

    runs are zero-arg callables that each execute the model once on one
    workload and return the output tree; they are called repeated_calls times
    per arm so a stateful step must match call by call.
    """
    baseline: list[list] = []
    for run in runs:
        baseline.append([_flatten(run()) for _ in range(repeated_calls)])

    wrapper = build_wrapper()
    uninstall = install(wrapper)
    try:
        for i, run in enumerate(runs):
            for j in range(repeated_calls):
                got = _flatten(run())
                want = baseline[i][j]
                if len(got) != len(want):
                    return CertificationResult(
                        False, f"run {i} call {j}: output arity changed"
                    )
                for k, (g, w) in enumerate(zip(got, want)):
                    if not mx.array_equal(g, w).item():
                        return CertificationResult(
                            False,
                            f"run {i} call {j} output {k}: identity replay is not bitwise "
                            f"invisible (a step whose state moves between calls, such as a "
                            f"cache offset, must be rewound by the model file so every call "
                            f"is the same step)",
                        )
    finally:
        uninstall()
    return CertificationResult(True)


def _flatten(tree: object) -> list[mx.array]:
    out = flatten_arrays(tree)
    mx.eval(out)
    return out


KEEP_BYTES = 2 << 30  # outputs one certification pass keeps for comparison


def certify_identities(model, wrappers: dict[str, object], runs: Sequence[Callable],
                       repeated_calls: int = 3) -> CertificationResult:
    """Check a batch of independent scopes in six model passes per workload.

    Observe every scope's outputs in both arms, so two wrong wrappers cannot
    pass by cancelling at the model output. Nested targets run in separate
    batches: a parent's replay may bypass its children. All temporary swaps
    unwind even when a call or comparison raises. Runs must rewind model state,
    as they do for the other whole-model correctness checks.
    """
    pending = dict(wrappers)
    if repeated_calls < 1:
        raise ValueError("identity certification requires at least one call per arm")
    while pending:
        paths = [p for p in pending if not any(p.startswith(q + ".") for q in pending if q != p)]
        batch = {p: pending.pop(p) for p in paths}
        result = _certify_batch(model, batch, runs, repeated_calls)
        if not result.ok:
            return result
    return CertificationResult(True)


def _certify_batch(model, wrappers, runs, repeated_calls):
    def snapshot(out):
        # Preserve scalar state and container structure as well as array values,
        # copied to host memory as they are made: off the GPU's working set, and
        # before the model can update an array in place.
        return _structure(out), _held(flatten_arrays(out))

    class Observed(ReplayWrapper):
        def __init__(self, wrapped, path, observe):
            nn.Module.__init__(self)
            self.wrapped = wrapped
            object.__setattr__(self, "_path", path)
            object.__setattr__(self, "_observe", observe)

        def __call__(self, *args, **kwargs):
            out = self.wrapped(*args, **kwargs)
            self._observe(self._path, out)
            return out

    def observed(run, paths, replacement, observe):
        installed = []
        try:
            for path in paths:
                original = resolve(model, path)[2]
                target = wrappers[path] if replacement else original
                previous = swap_install(model, path, Observed(target, path, observe))
                installed.append((path, previous))
            out = run()
            observe("", out)
            return out
        finally:
            for path, original in reversed(installed):
                swap_uninstall(model, path, original)

    class Changed(BaseException):  # not an Exception: no model code may swallow it
        pass

    seen = set()
    for index, run in enumerate(runs):
        for repetition in range(repeated_calls):
            left = list(wrappers)
            while left:
                # The original's outputs are kept in host memory, in the order
                # the scopes run, up to KEEP_BYTES; scopes past it wait for
                # another pass. The replacement's are compared as they are made
                # and dropped. Qwen3-30B-A3B at 2,028 tokens: one pass of all 144
                # expert projections kept 5.6 GB. Kept on the GPU, 2 GB of
                # Qwen3.8-27B's 64 MLP outputs at 1,833 tokens ran it out of
                # memory beside 15.4 GB of weights and the prompt's own work.
                expected, kept = {"": []}, [0]

                def keep(path, out):
                    if path not in expected:
                        if path and len(expected) > 1 and kept[0] >= KEEP_BYTES:
                            return  # the next pass
                        expected[path] = []
                    values = snapshot(out)
                    kept[0] += sum(x.nbytes for x, _ in values[1])
                    expected[path].append(values)

                observed(run, left, False, keep)
                group = [path for path in left if path in expected]
                if not group:
                    break  # the rest never ran at this workload
                seen.update(group)
                counts = dict.fromkeys(expected, 0)

                def check(path, out):
                    # A scope whose output differs changes every scope after it,
                    # so the first mismatch to execute is the one to blame.
                    call = counts[path]
                    counts[path] += 1
                    if call >= len(expected[path]):
                        raise Changed(CertificationResult(
                            False, f"identity scope {path!r}: call count changed at workload {index}", path or None))
                    want_structure, want_values = expected[path][call]
                    want_values = _back(want_values)
                    got_values = flatten_arrays(out)
                    mx.eval(got_values)
                    if (_structure(out) != want_structure or len(want_values) != len(got_values)
                            or any(not bitwise_equal(a, b) for a, b in zip(want_values, got_values))):
                        raise Changed(CertificationResult(
                            False, f"identity scope {path or '<model>'!r}: output changed at workload {index}, "
                            f"repetition {repetition}, call {call}", path or None))
                try:
                    observed(run, group, True, check)
                except Changed as changed:
                    return changed.args[0]
                for path, calls in expected.items():
                    if counts[path] != len(calls):
                        return CertificationResult(
                            False, f"identity scope {path!r}: call count changed at workload {index}", path or None)
                left = [path for path in left if path not in expected]
                del expected
    missing = set(wrappers) - seen
    if missing:
        return CertificationResult(False, f"identity scopes never executed: {sorted(missing)}", sorted(missing)[0])
    return CertificationResult(True)
