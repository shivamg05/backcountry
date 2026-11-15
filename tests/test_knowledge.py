"""The knowledge carried between jobs: one small append-only line per attempt,
and an index of other jobs' attempts with their code read on request."""

import json

from backcountry.harness.knowledge import INDEX_ROWS, WIN_ROWS, Knowledge, outcome, region_label, shapes

SETUP = {"chip": "Apple M4", "mlx": "0.32.2", "model": "gpt_oss.py", "model_type": "moe-llm"}
OPS = ["mx.gather_qmm", "array.__getitem__", "mx.fast.rms_norm", "array.__getitem__"]
IO = {"prompt_128": {"inputs": [[[128, 2880], "bfloat16"]], "outputs": [[[128, 2880], "bfloat16"]]},
      "prompt_32": {"inputs": [[[32, 2880], "bfloat16"]], "outputs": [[[32, 2880], "bfloat16"]]}}


def attempt(kb, kid, result="failed at smoke", shipped=False):
    kb.add_attempt(OPS, IO, kid, result, shipped, f"{kid}: tile for these rows", {"source": f"// {kid}"})


def test_an_attempt_is_one_small_line_and_its_reflection_another(tmp_path):
    file = tmp_path / "knowledge.jsonl"
    kb = Knowledge(file, "job1", SETUP)
    attempt(kb, "k1", outcome("correct_slower", None, 1.25, 1.0, "not_tested"))
    kb.add_reflection("k1", "a 64-row tile wastes rows at 93")
    first, second = [json.loads(line) for line in file.read_text().splitlines()]
    assert first == {**SETUP, "job": "job1", "when": first["when"], "type": "attempt",
                     "region": "gather_qmm + getitem + rms_norm", "shapes": "in [128, 2880] bfloat16; out [128, 2880] bfloat16",
                     "kernel": "k1", "outcome": "correct_slower: region 1.25 ms vs library 1 ms (0.80x)",
                     "shipped": False, "change": "k1: tile for these rows", "code": {"source": "// k1"}}
    assert second["type"] == "reflection" and second["kernel"] == "k1"


def test_the_outcome_line_is_the_harness_verdict_alone():
    assert outcome("failed", "workloads", None, None, None) == "failed at workloads"
    assert outcome("shipped", None, 0.5, 1.0, "shipped") == \
        "shipped: region 0.5 ms vs library 1 ms (2.00x); whole model: shipped"
    assert region_label(["array.__add__", "array.__add__", "mx.fast.rope"]) == "add + rope"
    assert shapes({}) == "in ; out "


def test_the_index_shows_other_jobs_same_model_type_first_with_reflections(tmp_path):
    file = tmp_path / "knowledge.jsonl"
    flux = Knowledge(file, "flux", {**SETUP, "model": "flux.py", "model_type": "diffusion"})
    for kid in ("d1", "d2"):
        attempt(flux, kid)
    moe = Knowledge(file, "qwen", {**SETUP, "model": "qwen_moe.py"})
    attempt(moe, "m1", "shipped: region 0.5 ms vs library 1 ms (2.00x)", shipped=True)
    moe.add_reflection("m1", "sorting tokens by expert pays")
    file.write_text(file.read_text() + json.dumps({"job": "old", "type": "attempt", "chunk": {}}) + "\n")
    here = Knowledge(file, "gpt-oss", SETUP)
    attempt(here, "mine")  # this job's own attempts are in its history already

    index = here.index("region-a")
    assert sorted(e["model"] for e in index) == ["flux.py", "flux.py", "qwen_moe.py"]  # old-format line skipped
    index = sorted(index, key=lambda e: e["model"] != "qwen_moe.py")
    assert index[0] == {"model": "qwen_moe.py", "model_type": "moe-llm", "chip": "Apple M4",
                        "region": "gather_qmm + getitem + rms_norm", "shapes": index[0]["shapes"],
                        "outcome": "shipped: region 0.5 ms vs library 1 ms (2.00x)",
                        "change": "m1: tile for these rows", "reflection": "sorting tokens by expert pays",
                        "code": {"source": "// m1"}}
    assert sorted(here.index("region-a"), key=lambda e: e["model"] != "qwen_moe.py") == index  # fixed per region
    assert Knowledge(tmp_path / "missing.jsonl", "job", SETUP).index("r") == []


def test_the_index_is_capped_and_draws_across_everything_else(tmp_path):
    file = tmp_path / "knowledge.jsonl"
    other = Knowledge(file, "other", SETUP)
    for i in range(INDEX_ROWS + 20):
        attempt(other, f"k{i}")
    here = Knowledge(file, "here", SETUP)
    drawn = [e["code"]["source"] for e in here.index("region-a")]
    assert len(drawn) == INDEX_ROWS and drawn != sorted(drawn)
    assert drawn != [e["code"]["source"] for e in here.index("region-b")]


def test_past_code_reaches_the_judge_only_by_source_id():
    from backcountry.harness.judge.experience import compact_experience

    archived = []

    def archive(value):
        archived.append(value)
        return {"source_id": f"source_{len(archived)}", "characters": len(json.dumps(value))}

    entry = {"model": "flux.py", "outcome": "failed at smoke", "code": {"source": "kernel body " * 100}}
    brief = compact_experience({"region": {"ops": [], "fingerprint": "f"}, "past_runs": [entry]}, archive)
    assert brief["past_runs"][0]["code"]["source_id"] and entry["code"] in archived
    assert "kernel body" not in json.dumps({k: v for k, v in brief.items() if k != "techniques_reference"})


def test_a_few_slots_go_to_attempts_that_shipped(tmp_path):
    file = tmp_path / "knowledge.jsonl"
    other = Knowledge(file, "other", SETUP)
    for i in range(60):
        attempt(other, f"lose{i}")
    for i in range(6):
        attempt(other, f"win{i}", shipped=True)
    for seed in ("a", "b", "c"):
        drawn = [e["code"]["source"] for e in Knowledge(file, "here", SETUP).index(seed)]
        assert len(drawn) == INDEX_ROWS and sum("win" in d for d in drawn) >= WIN_ROWS


def test_attempts_from_the_same_model_type_fill_the_index_first(tmp_path):
    file = tmp_path / "knowledge.jsonl"
    flux = Knowledge(file, "flux", {**SETUP, "model": "flux.py", "model_type": "diffusion"})
    for i in range(INDEX_ROWS):
        attempt(flux, f"d{i}")
    qwen = Knowledge(file, "qwen", {**SETUP, "model": "qwen_moe.py"})
    for i in range(3):
        attempt(qwen, f"m{i}")
    index = Knowledge(file, "here", SETUP).index("region-a")
    assert sorted(e["code"]["source"] for e in index if e["model_type"] == "moe-llm") == ["// m0", "// m1", "// m2"]


def test_an_outcome_across_sizes_says_how_the_kernel_did_at_each():
    from backcountry.harness.knowledge import outcome
    sizes = [{"size": "T=42", "ratio": 1.21, "result": "faster"},
             {"size": "T=425", "ratio": 1.02, "result": "tie"},
             {"size": "T=2028", "ratio": 0.93, "result": "slower"}]
    text = outcome("correct_slower", None, 2.0, 2.4, "not_tested", sizes)
    assert text == ("correct_slower: faster at 1 of 3 sizes, slower at 1 "
                    "(0.93x to 1.21x the library, T=42 to T=2028)")
    assert outcome("correct_slower", None, 2.0, 2.4, "not_tested") == \
        "correct_slower: region 2 ms vs library 2.4 ms (1.20x)"
