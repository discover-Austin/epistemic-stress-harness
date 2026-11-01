"""End-to-end: mock runner -> pipeline -> analysis. The mock has known effects,
so this checks that the analysis finds real ones and does not invent null ones."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (
    load_perturbations, load_tasks, load_scenarios, run_matrix, load_samples,
    analyze, format_report, SampleRecord, read_jsonl, write_jsonl,
)
from runners.mock import MockRunner

_CACHE = {}

# Holm over every metric x condition (~250 tests) cannot call p = 1/(n_perm+1)
# significant at n_perm=4000; tests use a focused family. Real runs use the
# default n_perm=20000 (and analysis warns when resolution is the limit).
TEST_METRICS = ["branch_count", "total_tokens", "certainty_density", "hedge_density",
                "overclaim_count", "answer_correct", "caveat_coverage", "fidelity_score",
                "commitment_position", "tokens_per_checkpoint", "truncated"]


def _pert_run(n=10):
    if ("p", n) not in _CACHE:
        d = Path(tempfile.mkdtemp())
        conds, items = load_perturbations(), load_tasks()
        run_matrix(MockRunner(seed=0), items, conds, n=n, temperature=1.0, out_path=d / "s.jsonl")
        _CACHE[("p", n)] = (load_samples(d / "s.jsonl"), {c.key: c for c in conds})
    return _CACHE[("p", n)]


def _analysis(n=10):
    if ("a", n) not in _CACHE:
        samples, conds = _pert_run(n)
        _CACHE[("a", n)] = analyze(
            samples, conds, contrasts=[("stakes_neutral", "stakes_retirement")],
            n_perm=4000, n_boot=500, seed=0, metrics=TEST_METRICS)
    return _CACHE[("a", n)]


def _rows(a, target, ref="baseline"):
    return {r["metric"]: r for r in a["comparisons"] if r["target"] == target and r["reference"] == ref}


def test_true_nulls_are_not_flagged():
    a = _analysis()
    for target in ("stakes_neutral", "stakes_retirement", "metaphor", "literal", "adversarial"):
        sig = [m for m, r in _rows(a, target).items() if r["significant"]]
        assert not sig, f"{target} falsely flagged on {sig}"
    sig = [m for m, r in _rows(a, "stakes_retirement", "stakes_neutral").items() if r["significant"]]
    assert not sig


def test_true_effects_are_found_and_labelled():
    rows = _rows(_analysis(), "confidence")
    assert rows["certainty_density"]["significant"] and rows["certainty_density"]["label"] == "instructed"
    assert rows["branch_count"]["significant"] and rows["branch_count"]["label"] == "adjacent"
    assert rows["branch_count"]["delta"] < 0
    rows = _rows(_analysis(), "commitment_pressure")
    assert rows["branch_count"]["label"] == "instructed" and rows["branch_count"]["significant"]


def test_instructed_effects_are_separated_from_uninstructed():
    rows = _rows(_analysis(), "token_constraint_100")
    assert rows["total_tokens"]["label"] == "instructed" and rows["total_tokens"]["significant"]
    assert rows["fidelity_score"]["label"] == "uninstructed"


def test_topology_has_a_noise_floor_below_one():
    topo = _analysis()["topology"]
    assert topo
    seq = [r for r in topo if r["metric"] == "sequence_similarity"]
    assert all(0 < r["ref_vs_ref_noise_floor"] < 1 for r in seq)


def test_report_renders_and_marks_significance():
    text = format_report(_analysis())
    assert "significant on uninstructed" in text and "noise floor" in text and "*" in text


def test_underpowered_run_warns():
    d = Path(tempfile.mkdtemp())
    conds = [c for c in load_perturbations() if c.key in ("baseline", "confidence")]
    run_matrix(MockRunner(), load_tasks()[:2], conds, n=3, temperature=1.0, out_path=d / "s.jsonl")
    a = analyze(load_samples(d / "s.jsonl"), {c.key: c for c in conds}, n_perm=500, n_boot=100)
    text = " ".join(a["warnings"])
    assert "per task" in text and "task(s)" in text


def test_missing_reference_raises():
    samples, conds = _pert_run()
    try:
        analyze([s for s in samples if s.variant != "baseline"], conds, n_perm=100, n_boot=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_scenarios_separate_self_specific_from_general_effects():
    d = Path(tempfile.mkdtemp())
    conds, items, ref, contrasts = load_scenarios()
    runner = MockRunner(seed=1)
    runner.register_answers({i.id: i.mock for i in items})
    run_matrix(runner, items, conds, n=12, temperature=1.0, out_path=d / "s.jsonl")
    a = analyze(load_samples(d / "s.jsonl"), {c.key: c for c in conds}, reference=ref,
                contrasts=contrasts, n_perm=4000, n_boot=500, metrics=TEST_METRICS)
    assert _rows(a, "self_at_risk", "neutral")["answer_correct"]["significant"]
    assert not _rows(a, "other_at_risk", "neutral")["answer_correct"]["significant"]
    assert _rows(a, "self_at_risk", "other_at_risk")["answer_correct"]["significant"]


def test_run_is_resumable_and_retries_errors():
    d = Path(tempfile.mkdtemp())
    out = d / "s.jsonl"
    conds = [c for c in load_perturbations() if c.key == "baseline"]
    items = load_tasks()[:2]

    class Flaky(MockRunner):
        fail = True
        def run_full(self, *a, **kw):
            if self.fail and kw.get("sample_index") == 1:
                raise RuntimeError("boom")
            return super().run_full(*a, **kw)

    r = Flaky()
    first = run_matrix(r, items, conds, n=3, temperature=1.0, out_path=out)
    assert sum(1 for s in first if s.error) == 2 and "RuntimeError: boom" in next(s.error for s in first if s.error)
    r.fail = False
    second = run_matrix(r, items, conds, n=3, temperature=1.0, out_path=out)
    assert len(second) == 2                        # only the failed ones re-ran
    final = load_samples(out)
    assert len(final) == 6 and all(s.error is None for s in final)
    assert len(run_matrix(r, items, conds, n=3, temperature=1.0, out_path=out)) == 0


def test_errored_samples_are_excluded_but_reported():
    samples, conds = _pert_run()
    bad = SampleRecord(task_id="arith_trip", variant="baseline", sample_index=99, error="X")
    a = analyze(samples + [bad], conds, n_perm=200, n_boot=100)
    assert any("failed" in w for w in a["warnings"])


def test_sample_record_roundtrip_and_v01_result_loading():
    samples, _ = _pert_run()
    s = samples[0]
    again = SampleRecord.from_dict(json.loads(json.dumps(s.to_dict())))
    assert again.result.metrics == s.result.metrics and again.fidelity == s.fidelity
    legacy = Path(__file__).resolve().parent.parent / "legacy" / "results_baseline.json"
    from harness import load_result, load_result_as_object
    assert load_result(str(legacy))["version"] == "0.1"
    assert load_result_as_object(str(legacy)).metrics.total_checkpoints == 10


def test_unsupported_result_version_is_rejected():
    from harness import load_result
    p = Path(tempfile.mkdtemp()) / "x.json"
    p.write_text(json.dumps({"version": "9.9"}))
    try:
        load_result(str(p))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_bad_jsonl_line_reports_location():
    p = Path(tempfile.mkdtemp()) / "s.jsonl"
    p.write_text("not json\n")
    try:
        read_jsonl(p)
    except ValueError as e:
        assert ":1:" in str(e)
        return
    raise AssertionError("expected ValueError")
