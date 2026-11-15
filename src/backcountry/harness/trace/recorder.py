"""Record mode: node capture, array identity, module addressing.

The recorder holds a weak reference to every array it sees and checks it on
each lookup, so id() is a sound identity for the whole pass without keeping
any array alive: a model that evaluates mid-pass (mlx-lm between 2,048-token
prompt chunks) frees its intermediates as it would unrecorded. Mutation (x[i] = v, an in-place dunder
returning its own receiver, an op returning an input object) is handled as SSA
renaming: the mutated object gets a fresh array_id going forward, produced by
the mutating node; earlier consumers keep the old id.
"""

from __future__ import annotations

import io
import math
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import mlx.core as mx

from .freeze import freeze
from .optable import MUTATING_METHODS
from .types import STATE_PREFIX, ScopeCall, Trace, TraceNode
from .walk import flatten_arrays, reachable
from backcountry_runtime.swap import state_signature

OPAQUE_OP = "compiled_fn"       # a compiled call the harness cannot name
COMPILED_PREFIX = "compiled:"   # a compiled call named by its import path
UNNAMED_KERNEL_OP = "metal_kernel"   # source-backed call (or legacy uncaptured call)
KERNEL_PREFIX = "metal_kernel:"      # a custom kernel call named by its import path
# An array the record cannot explain: made outside the patch surface by
# something that is not provably a constant (Gemma 4's mx.array(cache.offset)
# after mlx-lm evaluated the cache mid-call). Its real value is captured at
# every pass; no generated code ever recreates it.
UNKNOWN_SOURCE_OP = "unknown_source"


def compiled_op(path: str | None) -> str:
    return f"{COMPILED_PREFIX}{path}" if path else OPAQUE_OP


def kernel_op(path: str | None) -> str:
    return f"{KERNEL_PREFIX}{path}" if path else UNNAMED_KERNEL_OP


def is_opaque(op: str) -> bool:
    """A compiled section, a custom kernel call, or a state call: never inside
    a region, always a chain barrier."""
    return (op in (OPAQUE_OP, UNNAMED_KERNEL_OP, UNKNOWN_SOURCE_OP) or op.startswith(COMPILED_PREFIX)
            or op.startswith(KERNEL_PREFIX) or op.startswith(STATE_PREFIX))


def compiled_path(op: str) -> str | None:
    return op[len(COMPILED_PREFIX):] if op.startswith(COMPILED_PREFIX) else None


def kernel_path(op: str) -> str | None:
    return op[len(KERNEL_PREFIX):] if op.startswith(KERNEL_PREFIX) else None


def state_method(op: str) -> str | None:
    """The method name of a state call, e.g. "update_and_fetch" for
    "state:KVCache.update_and_fetch"; None for any other op."""
    return op.rsplit(".", 1)[1] if op.startswith(STATE_PREFIX) else None


# a constant this small lives as a literal in the trace and in generated wrappers
CONSTANT_OP = "mx.array"
CONSTANT_MAX_ELEMENTS = 1024


@dataclass(frozen=True)
class ArrayRef:
    """Placeholder for an array argument inside a node's args template."""
    index: int


@dataclass(frozen=True)
class ObjectRef:
    """Placeholder for a non-array, non-literal argument of a module call
    (a cache object handed down the tree); index into the scope's obj_ids."""
    index: int


def _is_literal(obj: object) -> bool:
    return obj is None or obj is Ellipsis or isinstance(obj, (bool, int, float, str, mx.Dtype))


def dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


def _spec(arr: mx.array) -> tuple[tuple[int, ...], str]:
    return (tuple(arr.shape), dtype_name(arr.dtype))


class Recorder:
    # Snapshot walks skip anything flagged _trace_internal, so a model holding
    # a compiled proxy (which references this recorder) cannot leak the
    # recorder's own held arrays into weights or retention.
    _trace_internal = True

    def __init__(self) -> None:
        self.armed = False
        self._armed_thread = threading.get_ident()
        self._suppress = 0
        self._model_for_walk: object | None = None
        # id(object) -> path for the objects holding model state; the patcher
        # sets it when it wraps the model, before any pass (add_holder, is_holder)
        self.state_holders: dict[int, str] = {}
        self._holder_refs: dict[int, Callable[[], object]] = {}
        # Node seqs whose outputs keep a value snapshot; None keeps every one.
        # A snapshot pins its array version until the pass ends, so a trace
        # keeps none and a capture keeps only the boundary it came for.
        self.snapshot_seqs: set[int] | None = None
        self._reset_pass()

    def _reset_pass(self) -> None:
        self.nodes: list[TraceNode] = []
        self._refs: dict[int, weakref.ref] = {}  # id(array) -> the array that id belongs to
        self._ids: dict[int, int] = {}
        self._by_aid: dict[int, mx.array] = {}
        self._next_id = 0
        self.inputs: set[int] = set()
        self.weights: set[int] = set()
        self.weight_paths: dict[int, str] = {}
        self._weight_refs: dict[int, weakref.ref] = {}  # weight array id -> the model's own array
        self.step_outputs: list[int] = []
        self.in_pass_evaluation = False
        self.eval_sites: list[tuple[str, ...]] = []
        self.evaluated: set[int] = set()
        self._call_numbers: list[dict[int, object]] = []  # per open op call: id(array) -> number read
        self._addr_stack: list[str] = []
        self._pending_scopes: list[tuple] = []
        self._reading_module_args = False  # module_enter is templatizing the call's arguments
        self.scope_calls: list[ScopeCall] = []
        self._instance_paths: dict[int, str] = {}
        self._call_counts: dict[int, int] = {}
        self._pos_counts: dict[tuple[str, str], int] = {}
        self._root_id: int | None = None
        self._frames: list[list] = []  # open state-holder calls: [start, arrays before, mutated]

    @property
    def recording(self) -> bool:
        return self.armed and self._suppress == 0

    @contextmanager
    def suppressed(self) -> Iterator[None]:
        """Silence recording inside an opaque call (a compiled section's trace
        re-executes its python; those inner ops are not part of the record)."""
        self._suppress += 1
        try:
            yield
        finally:
            self._suppress -= 1

    def _register(self, arr: mx.array, snapshot: bool = True) -> int:
        # Keep an independent lazy handle: later in-place updates to arr must
        # not rewrite the value captured under this version of its array id.
        oid = id(arr)
        if self._known(arr) is not None:
            return self._ids[oid]
        self._refs[oid] = weakref.ref(arr)
        self._ids[oid] = self._next_id
        for frame in self._frames:   # made inside a state call (state_exit)
            frame[5].append(oid)
        if snapshot:
            self._by_aid[self._next_id] = self._snapshot(arr)
        self._next_id += 1
        return self._ids[oid]

    def _snapshot(self, arr: mx.array) -> mx.array:
        """An independent handle on arr's value now. Made on another thread than
        the one that armed the pass (an engine's worker runs the model on a
        stream of its own), it is evaluated there: the arming thread cannot
        evaluate work queued on a stream it does not have."""
        copy = mx.array(arr)
        if threading.get_ident() != self._armed_thread:
            with self.suppressed():
                mx.eval(copy)
        return copy

    def add_holder(self, obj: object, path: str | None) -> None:
        self.state_holders[id(obj)] = path
        try:
            self._holder_refs[id(obj)] = weakref.ref(obj)
        except TypeError:  # no weak references: held, so its id is never reused
            self._holder_refs[id(obj)] = lambda held=obj: held

    def is_holder(self, obj: object) -> bool:
        """obj is a state holder. Its id alone was not enough: mlx-lm's
        batch_generate makes and drops caches every step, a new object came
        back on a dead holder's id and was taken for one on some passes only,
        and the same request recorded different work each time."""
        ref = self._holder_refs.get(id(obj))
        return ref is not None and ref() is obj

    def _known(self, arr: mx.array) -> int | None:
        """arr's array id, or None when unseen: a freed array's id() can come
        back on a new one."""
        ref = self._refs.get(id(arr))
        return self._ids[id(arr)] if ref is not None and ref() is arr else None

    def _register_output(self, arr: mx.array, seq: int) -> int:
        """An output object already known under an id was mutated (or returned
        unchanged): rename so the node is its producer from here on."""
        snapshot = self.snapshot_seqs is None or seq in self.snapshot_seqs
        oid = id(arr)
        if self._known(arr) is not None:
            self._ids[oid] = self._next_id
            for frame in self._frames:   # made inside a state call (state_exit)
                frame[5].append(oid)
            if snapshot:
                self._by_aid[self._next_id] = self._snapshot(arr)
            self._next_id += 1
            return self._ids[oid]
        return self._register(arr, snapshot)

    def arrays_for(self, ids: Iterable[int]) -> dict[int, mx.array]:
        """Lazy snapshots of the recorded array versions, valid until freeze.
        Boundary capture evaluates these without observing later mutations.
        A weight the pass never changed is the model's own array, not a
        snapshot: evaluating a snapshot copies it, and on a model near the
        GPU's limit (Qwen3.8-27B's two 0.76 GB vocabulary tables) the copy
        is what tipped a capture pass over."""
        return {aid: self._version(aid) for aid in ids}

    def _version(self, aid: int) -> mx.array:
        ref = self._weight_refs.get(aid)
        arr = ref() if ref is not None else None
        # a mutation renames the object, so it no longer answers to aid
        return arr if arr is not None and self._known(arr) == aid else self._by_aid[aid]

    # -- pass lifecycle ------------------------------------------------------

    def arm(self, model: object, workload_arrays: list[mx.array], module_paths: dict[int, str]) -> None:
        """Start a recording pass: register weights (snapshot walk) and inputs,
        reset all per-pass state. module_paths maps id(instance) -> tree path."""
        if self.armed:
            raise RuntimeError("recorder already armed")
        self._reset_pass()
        self._armed_thread = threading.get_ident()
        self._register_reachable(model)
        self._instance_paths = dict(module_paths)
        self._root_id = id(model)
        self._model_for_walk = model
        for arr in workload_arrays:
            self.inputs.add(self._register(arr))
        self.armed = True

    def _register_reachable(self, model: object) -> None:
        for path, obj in reachable(model):
            if isinstance(obj, mx.array):
                aid = self._register(obj)
                self.weights.add(aid)
                self._weight_refs[aid] = weakref.ref(obj)
                self.weight_paths.setdefault(aid, path)

    def step(self, model: Callable, args: tuple) -> object:
        """One traced call of the model: roots the address space, resets the
        per-step counters, captures the returned tree."""
        if not self.armed:
            raise RuntimeError("recorder is not armed")
        self._call_counts.clear()
        self._pos_counts.clear()
        k = self._call_counts.get(id(model), 0)
        self._call_counts[id(model)] = k + 1
        self._addr_stack.append(f"@{k}")
        args_t, kwargs_t, arg_ids, _, obj_ids = self._templatize(args, {}, objects=True)
        try:
            outs = model(*args)
        except BaseException:
            self._addr_stack.pop()
            raise
        out_t, _, out_ids, _, _ = self._templatize((outs,), {})
        self.scope_calls.append(ScopeCall(
            address=self._addr_stack[-1], stack=tuple(self._addr_stack),
            args_template=args_t, kwargs_template=kwargs_t, arg_ids=tuple(arg_ids),
            out_template=out_t[0], out_ids=tuple(out_ids), obj_ids=tuple(obj_ids),
        ))
        self._addr_stack.pop()
        self.step_outputs = [self._register(a) for a in flatten_arrays(outs)]
        return outs

    def disarm(self) -> None:
        self.armed = False

    def freeze_pass(self) -> Trace:
        """Freeze after a step: retention via the positive snapshot walk (any
        produced array reachable from the model), then drop everything."""
        # what a state call returns is the method's own business, not a kept value
        produced = {aid for n in self.nodes if not n.op.startswith(STATE_PREFIX) for aid in n.out_arrays}
        post = [a for _, a in reachable(self._model_for_walk) if isinstance(a, mx.array)] \
            if self._model_for_walk else []
        retained = {self._known(a) for a in post} & produced
        trace = freeze(
            self.nodes,
            inputs=self.inputs,
            weights=self.weights,
            step_outputs=self.step_outputs,
            retained=retained,
            weight_paths=self.weight_paths,
            in_pass_evaluation=self.in_pass_evaluation,
            eval_sites=tuple(self.eval_sites),
            scope_calls=tuple(self.scope_calls),
            evaluated=self.evaluated,
        )
        self._refs.clear()
        self._by_aid.clear()
        self._weight_refs.clear()
        self._model_for_walk = None
        return trace

    # -- hooks called by the patch surface -----------------------------------

    def note_evaluation(self, *values, number=None) -> None:
        """The model evaluated these arrays itself: that work is live even
        when nothing recorded reads the result. Reading a constant, a value
        computed from Python numbers alone (mflux's 1 / mx.sqrt(head_dim)),
        is not a mid-step evaluation: its value never changes and mx.compile
        evaluates it too. Any other read is. A constant an MLX call reads as
        a number while it runs (that scale passed as scale=) is recorded as
        the number the call received, so a replayed or compiled span passes
        the number too, never an array it would have to read. The calls that
        made a constant are not step work: a compiled step computes them once
        while it is built and its graph holds none of them, so nothing marks
        them live and no region, price or floor counts them."""
        if self.recording:
            ids = [self._known(arr) for arr in flatten_arrays(values)]
            if None in ids or self._computed_from_state(ids):
                self.evaluated.update(aid for aid in ids if aid is not None)
                self.in_pass_evaluation = True
                self.eval_sites.append(tuple(self._addr_stack))
            elif number is not None and self._call_numbers:
                self._call_numbers[-1][id(values[0])] = number

    def call_enter(self) -> None:
        self._call_numbers.append({})

    def call_exit(self) -> dict[int, object]:
        return self._call_numbers.pop()

    def _computed_from_state(self, ids) -> bool:
        """Whether any of these arrays traces back to an input, a weight, a
        state call or an array with no recorded producer."""
        producer = {aid: n for n in self.nodes for aid in n.out_arrays}
        roots = self.inputs | self.weights
        pending, seen = list(ids), set()
        while pending:
            aid = pending.pop()
            if aid in seen:
                continue
            seen.add(aid)
            node = producer.get(aid)
            if aid in roots or node is None or node.op.startswith(STATE_PREFIX):
                return True
            pending.extend(node.in_arrays)
        return False

    def module_enter(self, instance: object, args: tuple = (), kwargs: dict | None = None) -> None:
        path = self._instance_paths.get(id(instance), f"?{type(instance).__name__}")
        k = self._call_counts.get(id(instance), 0)
        self._call_counts[id(instance)] = k + 1
        self._addr_stack.append(f"{path}@{k}")
        self._reading_module_args = True
        try:
            args_t, kwargs_t, ids, _, obj_ids = self._templatize(args, kwargs or {}, objects=True)
        finally:
            self._reading_module_args = False
        self._pending_scopes.append((self._addr_stack[-1], tuple(self._addr_stack), args_t, kwargs_t,
                                     tuple(ids), tuple(obj_ids)))

    def module_exit(self, result: object = None) -> None:
        address, stack, args_t, kwargs_t, arg_ids, obj_ids = self._pending_scopes.pop()
        out_t, _, out_ids, _, _ = self._templatize((result,), {})
        self.scope_calls.append(ScopeCall(
            address=address, stack=stack, args_template=args_t,
            kwargs_template=kwargs_t, arg_ids=arg_ids,
            out_template=out_t[0], out_ids=tuple(out_ids), obj_ids=obj_ids,
        ))
        self._addr_stack.pop()

    def module_abort(self) -> None:
        self._pending_scopes.pop()
        self._addr_stack.pop()

    def is_root(self, instance: object) -> bool:
        return id(instance) == self._root_id

    def _record_constant(self, arr: mx.array) -> None:
        """An array first met as an argument was made outside the patch surface.
        mx.array(...) over host data is the one legal way, since the class
        cannot be wrapped, so a small finite array with nothing computed behind
        it is recorded as that creation call, the way mx.arange is. Once a pass
        has evaluated arrays, a value Python read from the data looks like host
        data too (Gemma 4 wraps its cache offset in mx.array after mlx-lm
        evaluates the cache mid-call; a token count would look the same), so
        the job traces such a pass again on other inputs and a scope whose
        constants differ there hosts no generated code (varying_constant_scopes).
        Any other array the record cannot explain is recorded as an unknown
        source. Every pass captures its real value, so work downstream of it is
        searched as usual, but nothing that would have to recreate it (a stretch
        holding it, a scope that makes it) is ever generated."""
        if not self.recording:
            return
        if not (arr.size > CONSTANT_MAX_ELEMENTS
                or arr.dtype == mx.complex64 or not _built_from_data(arr)):
            with self.suppressed():  # reading the value is not the model evaluating
                value = arr.tolist()
            if all(math.isfinite(v) for v in _flat(value)):
                self._append_node(CONSTANT_OP, (value,), {"dtype": arr.dtype}, [arr])
                return
        if not self._reading_module_args:
            self._append_node(UNKNOWN_SOURCE_OP, (), {}, [arr])
            return
        # a module's argument came from its caller: the caller's scope made it
        # (Gemma 4's attention wraps the cache offset, its rope only reads it)
        scope = self._addr_stack.pop()
        try:
            self._append_node(UNKNOWN_SOURCE_OP, (), {}, [arr])
        finally:
            self._addr_stack.append(scope)

    def _templatize(self, args: tuple, kwargs: dict, objects: bool = False,
                    numbers: dict[int, object] | None = None):
        """Arrays become ArrayRefs. With objects=True (module calls), any
        other non-literal value becomes an ObjectRef, so a wrapper can pass
        a cache object through and call its methods where the record did."""
        in_ids: list[int] = []
        in_specs: list = []
        obj_ids: list[int] = []

        def template(obj: Any) -> Any:
            if numbers and isinstance(obj, mx.array) and id(obj) in numbers:
                return numbers[id(obj)]
            if isinstance(obj, mx.array):
                if self._known(obj) is None:
                    self._record_constant(obj)
                in_ids.append(self._register(obj))
                in_specs.append(_spec(obj))
                return ArrayRef(len(in_ids) - 1)
            if isinstance(obj, tuple):
                return tuple(template(v) for v in obj)
            if isinstance(obj, list):
                return [template(v) for v in obj]
            if isinstance(obj, dict):
                return {k: template(v) for k, v in obj.items()}
            if isinstance(obj, slice):
                if any(isinstance(v, mx.array) for v in (obj.start, obj.stop, obj.step)):
                    return slice(template(obj.start), template(obj.stop), template(obj.step))
                return obj
            if objects and not _is_literal(obj):
                obj_ids.append(id(obj))
                return ObjectRef(len(obj_ids) - 1)
            return obj

        args_t = tuple(template(a) for a in args)
        kwargs_t = {k: template(v) for k, v in kwargs.items()}
        return args_t, kwargs_t, in_ids, in_specs, obj_ids

    def maybe_record(
        self,
        op_name: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        mutates_first: bool = False,
        numbers: dict[int, object] | None = None,
    ) -> None:
        if not self.recording:
            return
        out_objs = flatten_arrays(result)
        if mutates_first and all(arr is not args[0] for arr in out_objs):
            out_objs.insert(0, args[0])
        if not out_objs:
            return
        self._append_node(op_name, args, kwargs, out_objs, numbers=numbers)

    def state_enter(self, obj: object) -> None:
        """A call on a state holder begins: where the record stands, which
        arrays the object holds, and its position counter. Replay guards the
        captured position rather than guessing the provenance of scalars
        from equal integer values."""
        try:
            signature = state_signature(obj)
        except TypeError:
            signature = None  # a field no wrapper can describe; graph delivery declines the scope
        self._frames.append([len(self.nodes), _array_ids(obj), False,
                             getattr(obj, "offset", None), signature, []])

    def state_abort(self) -> None:
        self._frames.pop()

    def state_exit(self, obj: object, method: str, args: tuple, kwargs: dict, result: Any) -> None:
        """A method that left the object's state alone keeps its ops in the
        record like any others. One that changed it (a write in place, a
        rebinding) collapses into one opaque state call: the wrapper replays
        it by calling the same method on the same object, so what it does to
        its state happens for real, where the recorder cannot see it."""
        start, before, mutated, offset_before, signature, made = self._frames.pop()
        inner = self.nodes[start:]
        if not (method in {"__getitem__", "__setitem__", "__getattribute__", "state"} or mutated
                or _array_ids(obj) != before or any(_mutates(n.op) for n in inner)):
            return
        if self._frames:
            self._frames[-1][2] = True  # a method that calls a mutating one mutated state too
        ops = []
        for n in inner:
            self._pos_counts[(n.module_address, n.op)] -= 1
            ops += n.scalar_args["receiver"]["inner_ops"] if n.op.startswith(STATE_PREFIX) else [n.op]
        # the arrays those ops made lose their producer with them: one the call
        # keeps on its object and the model reads later (mlx-lm's BatchKVCache
        # offset, fed to rope the next step) answered to an id no node made, and
        # the pass aborted. Forgotten, it is met again as an array the record
        # cannot explain; what the call returns is registered as its output below
        inner_out = {aid for n in inner for aid in n.out_arrays}
        for oid in made:
            if self._ids.get(oid) in inner_out:
                del self._ids[oid], self._refs[oid]
        del self.nodes[start:]
        receiver = {"id": id(obj), "path": self.state_holders.get(id(obj)),
                    "inner_ops": ops, "offset_before": offset_before}
        if signature is not None:
            receiver["state_signature"] = signature
        self._append_node(f"{STATE_PREFIX}{type(obj).__name__}.{method}", args, kwargs,
                          flatten_arrays(result), receiver=receiver)

    def _append_node(self, op_name: str, args: tuple, kwargs: dict, out_objs: list,
                     receiver: dict | None = None, numbers: dict[int, object] | None = None) -> None:
        args_t, kwargs_t, in_ids, in_specs, _ = self._templatize(args, kwargs, numbers=numbers)
        seq = len(self.nodes)
        out_ids = [self._register_output(a, seq) for a in out_objs]
        scalar_args: dict = {"args": args_t, "kwargs": kwargs_t}
        if receiver is not None:
            scalar_args["receiver"] = receiver

        stack = tuple(self._addr_stack)
        address = stack[-1] if stack else ""
        pos_key = (address, op_name)
        position = self._pos_counts.get(pos_key, 0)
        self._pos_counts[pos_key] = position + 1

        self.nodes.append(TraceNode(
            seq=len(self.nodes),
            op=op_name,
            in_arrays=tuple(in_ids),
            out_arrays=tuple(out_ids),
            in_specs=tuple(in_specs),
            out_specs=tuple(_spec(a) for a in out_objs),
            scalar_args=scalar_args,
            module_address=address,
            position_in_module=position,
            module_stack=stack,
        ))


def varying_constant_scopes(trace: Trace, other: Trace) -> set[str]:
    """Scope paths (addresses without their call count) whose constants differ
    between two traces of the same pass on different inputs: a value that
    changes with the data, not the shapes, so no generated code may freeze it.
    Every scope holding a constant if the two passes ran different calls."""
    mine = [n for n in trace.nodes if n.op == CONSTANT_OP]
    theirs = [n for n in other.nodes if n.op == CONSTANT_OP]
    if [n.module_stack for n in mine] != [n.module_stack for n in theirs]:
        differing = mine
    else:
        differing = [a for a, b in zip(mine, theirs) if a.scalar_args != b.scalar_args]
    return {address.rsplit("@", 1)[0] for n in differing for address in n.module_stack}


def _built_from_data(arr: mx.array) -> bool:
    """Whether nothing was computed to make arr: its graph export has no edge."""
    graph = io.StringIO()
    mx.export_to_dot(graph, arr)
    return "->" not in graph.getvalue()


def _flat(value: object):
    if isinstance(value, list):
        for v in value:
            yield from _flat(v)
    else:
        yield value


def _array_ids(obj: object) -> frozenset[int]:
    return frozenset(id(a) for _, a in reachable(obj) if isinstance(a, mx.array))


def _mutates(op: str) -> bool:
    return op.removeprefix("array.") in MUTATING_METHODS
