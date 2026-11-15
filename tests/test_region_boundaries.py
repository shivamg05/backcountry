"""Copies must agree on their public boundary, not just their operations."""
from dataclasses import replace

import mlx.core as mx
import pytest

from backcountry.harness.loop import JobRunner
from backcountry.harness.manifest import BOUNDARY_INPUT_SETS
from backcountry.harness.regions.build import _boundary
from backcountry.harness.regions.fingerprint import fingerprint, group_copies
from backcountry.harness.regions.store import BoundaryMismatch, BoundaryStore
from backcountry.harness.regions.sweep import SweepDivergence, locate_span
from backcountry.harness.regions.types import Region
from backcountry.harness.trace.types import Trace, TraceNode, Liveness, Retention


def pair(n=4, offset=0, native=False):
    spec = ((n,), 'float32')
    nodes = tuple(TraceNode(i, op, (offset + i,), (offset + i + 1,),
                           (spec,), (spec,), {'args': (), 'kwargs': {}}, 'layer', i,
                           ('root', 'layer'), {'source': 'example'} if native and i == 0 else None)
                  for i, op in enumerate(('mx.abs', 'mx.square')))
    trace = Trace(nodes, {0: (1,)}, (offset + 1, offset + 2), frozenset(),
                  frozenset((offset,)), {
                      offset + 1: Liveness(Retention.STEP_OUTPUT, (1,)),
                      offset + 2: Liveness(Retention.STEP_OUTPUT, ()),
                  })
    return trace, _boundary(trace, 'w', 0, 1)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('change', ['drop', 'different_value', 'reorder'])
def test_output_roles_distinguish_otherwise_identical_regions(native, change):
    trace, full = pair(native=native)
    if change == 'drop':
        a, b = full, replace(full, output_ids=(2,))
    elif change == 'different_value':
        a, b = replace(full, output_ids=(1,)), replace(full, output_ids=(2,))
    else:
        a, b = full, replace(full, output_ids=(2, 1))
    assert fingerprint(trace, a) != fingerprint(trace, b)


def test_input_order_is_part_of_contract():
    trace, full = pair()
    node = replace(trace.nodes[0], in_arrays=(0, 10), in_specs=trace.nodes[0].in_specs * 2)
    trace = replace(trace, nodes=(node, trace.nodes[1]))
    a = replace(full, input_ids=(0, 10))
    assert fingerprint(trace, a) != fingerprint(trace, replace(a, input_ids=(10, 0)))


def test_compatible_sizes_and_array_ids_still_group():
    a, sa = pair(4)
    b, sb = pair(16, offset=100)
    regions = group_copies({'small': a, 'large': b},
                          {'small': [replace(sa, workload='small')],
                           'large': [replace(sb, workload='large')]})
    assert len(regions) == 1 and regions[0].copies == 2


def test_sweep_refuses_changed_live_outputs():
    trace, full = pair()
    retrace = replace(trace, step_outputs=(2,))
    with pytest.raises(SweepDivergence, match='input/output roles'):
        locate_span(trace, full, retrace, 'w')
    assert locate_span(trace, full, pair(16)[0], 'w').output_ids == (1, 2)


def prepared(tmp_path):
    trace, span = pair()
    region = Region('target', ('mx.abs', 'mx.square'), [span])
    runner = object.__new__(JobRunner)
    runner.traces = {'w': trace}
    runner.store = BoundaryStore(tmp_path)
    runner._sweep_instances = lambda r: []
    for si in range(BOUNDARY_INPUT_SETS):
        for kind, ids in (('inputs', span.input_ids), ('outputs', span.output_ids)):
            runner.store.save(region.fingerprint, 'w', si, kind,
                              {i: mx.ones((4,), stream=mx.cpu) for i in ids})
    return runner, region


@pytest.mark.parametrize('damage', ['missing_file', 'missing_output', 'extra_output', 'truncated', 'missing_set'])
def test_preflight_rejects_incomplete_saved_data(tmp_path, damage):
    runner, region = prepared(tmp_path)
    runner._validate_boundaries([region])
    path = runner.store._path('target', 'w', BOUNDARY_INPUT_SETS - 1, 'outputs')
    if damage == 'missing_file':
        path.unlink()
    elif damage == 'missing_set':
        runner.store._path('target', 'w', 1, 'inputs').unlink()
    elif damage == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    else:
        ids = (2,) if damage == 'missing_output' else (1, 2, 3)
        runner.store.save('target', 'w', BOUNDARY_INPUT_SETS - 1, 'outputs',
                          {i: mx.ones((4,), stream=mx.cpu) for i in ids})
    with pytest.raises(BoundaryMismatch, match='Saved boundary data mismatch'):
        runner._validate_boundaries([region])


def test_preflight_rejects_bad_group_before_reading_files(tmp_path):
    runner, region = prepared(tmp_path)
    region.members.append(replace(region.members[0], output_ids=(2,)))
    with pytest.raises(BoundaryMismatch, match='incompatible input/output roles'):
        runner._validate_boundaries([region])


def test_preflight_checks_sweep_files_too(tmp_path):
    runner, region = prepared(tmp_path)
    runner.sweep_traces = {'w@L=16': runner.traces['w']}
    runner._sweep_instances = lambda r: [('w@L=16', region.members[0], {})]
    with pytest.raises(BoundaryMismatch, match='w%40L%3D16'):
        runner._validate_boundaries([region])


def test_preflight_rejects_sweep_with_changed_output_roles(tmp_path):
    runner, region = prepared(tmp_path)
    runner.sweep_traces = {'w@L=16': runner.traces['w']}
    span = replace(region.members[0], output_ids=(2,))
    runner._sweep_instances = lambda r: [('w@L=16', span, {})]
    with pytest.raises(BoundaryMismatch, match='input/output roles at sweep'):
        runner._validate_boundaries([region])


def test_actual_trace_separates_final_copy_and_replays_saved_outputs(tmp_path):
    from backcountry.harness.trace import Tracer
    from backcountry.harness.regions.price import capture_boundaries, capture_instances
    from backcountry.harness.trace.replay import replay
    from backcountry.harness.ladder.child import _project_ids, _ordered
    from backcountry.harness.regions.store import load_set
    def model(x):
        a = x + x
        b = mx.abs(a)
        c = b + b
        d = mx.abs(c)
        return a, d
    tr = Tracer()
    tr.install()
    try:
        x = mx.array([-1., 2., -3., 4.])
        trace, _ = tr.trace(model, [x])
        spans = [_boundary(trace, 'w', 0, 1), _boundary(trace, 'w', 2, 3)]
        assert [len(s.output_ids) for s in spans] == [2, 1]
        regions = group_copies({'w': trace}, {'w': spans})
        assert len(regions) == 2
        wanted = {i for s in spans for i in s.input_ids + s.output_ids}
        arrays = capture_boundaries(tr, model, [x], trace, wanted)
    finally:
        tr.uninstall()
    store = BoundaryStore(tmp_path)
    for r in regions:
        label, s, _ = capture_instances(r, {'w': trace})[0]
        nodes = trace.nodes[s.start_seq:s.end_seq + 1]
        path = store.save(r.fingerprint, label, 0, 'outputs', {i: arrays[i] for i in s.output_ids})
        store.validate(r.fingerprint, label, 0, 'outputs', s.output_ids)
        projected = _project_ids(nodes, nodes, s.output_ids)
        refs = _ordered(load_set(path), projected, str(path))
        result = replay(nodes, {i: arrays[i] for i in s.input_ids}, projected)
        for aid, ref in zip(projected, refs, strict=True):
            assert mx.array_equal(result[aid], ref).item()


def test_a_regions_weights_are_saved_once_and_loaded_back_with_every_set(tmp_path):
    """Weights are the same in every set and at every size: saved with each,
    Qwen3-30B-A3B's expert weights were 88 of every 94 MB."""
    store = BoundaryStore(tmp_path)
    weight = mx.arange(1000, dtype=mx.float32)
    for si in range(3):
        store.save("r", "main", si, "inputs", {1: mx.full((4,), float(si)), 2: weight}, {2: "layer.weight"})
    store.save("r", "other", 0, "inputs", {7: mx.zeros((2,)), 9: weight}, {9: "layer.weight"})
    assert [p.name for p in (tmp_path / "r").glob("*.safetensors")] == ["weights.safetensors"]
    for si in range(3):
        got = store.load("r", "main", si, "inputs")
        assert got[1].tolist() == [float(si)] * 4 and mx.array_equal(got[2], weight).item()
        store.validate("r", "main", si, "inputs", (1, 2))
        assert store._path("r", "main", si, "inputs").stat().st_size < 1000  # the weight is not in the set
    assert mx.array_equal(store.load("r", "other", 0, "inputs")[9], weight).item()
    store.save("r", "plain", 0, "inputs", {1: mx.ones((2,))})  # no weights: a set as before
    assert store.load("r", "plain", 0, "inputs")[1].tolist() == [1.0, 1.0]


def test_large_weights_are_one_copy_shared_by_every_set_small_ones_a_copy_each(tmp_path, monkeypatch):
    """The cache-cold clocks rotate each set's own copy of a small region's
    weights; a region whose weights alone outgrow that working set shares one
    copy (Qwen3.8-27B's 0.76 GB vocabulary region held 4.57 GB as six copies
    and the GPU timed out on swap). The sets alone keep that copy alive: once
    they are dropped nothing cached behind them holds it."""
    import backcountry.harness.measure.clocks as clocks
    from backcountry.harness.regions.store import _shared_weights
    store = BoundaryStore(tmp_path)
    for name, size in (("small", 100), ("large", 5000)):
        weight = mx.arange(size, dtype=mx.float32)
        for si in range(3):
            store.save(name, "main", si, "inputs", {1: mx.full((4,), float(si)), 2: weight}, {2: "w"})
    monkeypatch.setattr(clocks, "CACHE_DEFEAT_BYTES", 4 * 1000)  # between the two weights' sizes
    small = [s[2] for s in store.load_sets("small", "main", 3)]
    large = [s[2] for s in store.load_sets("large", "main", 3)]
    assert small[0] is not small[1] and large[0] is large[1] is large[2]
    assert all(mx.array_equal(w, mx.arange(5000, dtype=mx.float32)).item() for w in large)
    assert _shared_weights.cache_info().currsize == 0
    store.load("large", "main", 0, "inputs")
    assert _shared_weights.cache_info().currsize == 0


def test_a_copy_that_brings_new_weights_adds_them_to_the_regions_file(tmp_path):
    """A region's copy at another layer reads other weights. Adding them
    rewrote the weights file while its old contents were still being read
    lazily out of it, which truncated the read ([read] Unable to read)."""
    store = BoundaryStore(tmp_path)
    first, second = mx.arange(100000, dtype=mx.float32), mx.ones((7,))
    store.save("r", "a", 0, "inputs", {1: mx.zeros((3,)), 2: first}, {2: "layers.0.w"})
    store.save("r", "b", 0, "inputs", {1: mx.zeros((3,)), 2: second}, {2: "layers.1.w"})
    assert mx.array_equal(store.load("r", "a", 0, "inputs")[2], first).item()
    assert mx.array_equal(store.load("r", "b", 0, "inputs")[2], second).item()
    assert sorted(f.name for f in (tmp_path / "r").iterdir() if f.is_file()) == ["weights.safetensors"]


def test_a_state_array_that_differs_between_passes_stays_with_its_set(tmp_path):
    """Anything reachable from the model counts as a weight, a cache included;
    a value that differs from the one already shared under its path is kept in
    its own set rather than handed to another."""
    store = BoundaryStore(tmp_path)
    store.save("r", "a", 0, "inputs", {1: mx.zeros((3,))}, {1: "cache.keys"})
    store.save("r", "b", 0, "inputs", {1: mx.ones((3,))}, {1: "cache.keys"})
    assert store.load("r", "a", 0, "inputs")[1].tolist() == [0.0] * 3
    assert store.load("r", "b", 0, "inputs")[1].tolist() == [1.0] * 3


def test_a_set_whose_weights_file_is_gone_fails_the_preflight(tmp_path):
    store = BoundaryStore(tmp_path)
    store.save("r", "main", 0, "inputs", {1: mx.zeros((2,)), 2: mx.ones((2,))}, {2: "layer.weight"})
    (tmp_path / "r" / "weights.safetensors").unlink()
    with pytest.raises(BoundaryMismatch, match="weights file"):
        store.validate("r", "main", 0, "inputs", (1, 2))
