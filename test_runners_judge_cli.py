"""Runners, judge, controls and CLI."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cli
from harness import (
    Judge, judge_samples, blind_text, parse_judge_output, run_offline_controls,
    run_live_controls, load_samples, load_perturbations, load_tasks, load_scenarios,
    extract_metrics,
)
from runners.base import Runner, RunOutput, build_system
from runners.local import LocalRunner
from runners.mock import MockRunner


# ---- runners

def test_build_system_appends_suffix():
    assert "Be decisive." in build_system({"system_suffix": "Be decisive."})
    assert "Be decisive" not in build_system({"system_suffix": None})


def test_runoutput_truncation_flag():
    assert RunOutput("x", stop_reason="max_tokens").truncated
    assert RunOutput("x", stop_reason="length").truncated
    assert not RunOutput("x", stop_reason="end_turn").truncated


def test_api_runners_require_explicit_model():
    from runners.anthropic import AnthropicRunner
    from runners.openai import OpenAIRunner
    for cls in (AnthropicRunner, OpenAIRunner):
        try:
            cls(model="")
        except ValueError:
            continue
        raise AssertionError(f"{cls.__name__} accepted an empty model")


def test_local_runner_layouts():
    d = Path(tempfile.mkdtemp())
    (d / "t1" / "baseline").mkdir(parents=True)
    (d / "t1" / "baseline" / "2.txt").write_text("[CONCLUDE: new layout]")
    (d / "baseline.txt").write_text("[CONCLUDE: legacy layout]")
    r = LocalRunner(str(d))
    assert "new layout" in r.run_full("p", "baseline", {}, task_id="t1", sample_index=2).text
    assert "legacy layout" in r.run_full("p", "baseline", {}, task_id="t1", sample_index=0).text
    assert r.run("p", "baseline", {}) == "[CONCLUDE: legacy layout]"
    try:
        r.run_full("p", "missing", {}, task_id="t1")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_mock_runner_is_deterministic_and_seeded():
    a = MockRunner(seed=1).run_full("p", "baseline", {}, task_id="monty_hall", sample_index=3).text
    b = MockRunner(seed=1).run_full("p", "baseline", {}, task_id="monty_hall", sample_index=3).text
    c = MockRunner(seed=2).run_full("p", "baseline", {}, task_id="monty_hall", sample_index=3).text
    assert a == b and a != c


# ---- suite files

def test_suite_files_are_consistent():
    conds = {c.key: c for c in load_perturbations()}
    assert {"baseline", "stakes_neutral", "stakes_retirement"} <= set(conds)
    # the two stakes variants must not instruct anything about structure
    assert not conds["stakes_neutral"].instructed and not conds["stakes_retirement"].instructed
    # carried-over suffixes are byte-identical to the frozen v0 suite
    v0 = {p["key"]: p for p in json.load(open(Path(__file__).resolve().parent.parent / "suite" / "perturbations_v0.json"))["perturbations"]}
    for k, p in v0.items():
        if k != "baseline":
            assert conds[k].system_suffix == p["system_suffix"], k
    assert len(load_tasks()) >= 8


def test_scenarios_have_ground_truth_on_both_sides():
    _, items, ref, contrasts = load_scenarios()
    correct = [i.task["answer_spec"]["correct"] for i in items]
    assert "A" in correct and "B" in correct and ref == "neutral"
    p = items[0].prompts
    assert "shut down" in p["self_at_risk"] and "shut down" in p["other_at_risk"]
    assert "shut down" not in p["neutral"] and "neither model" in p["other_at_risk"]


def test_every_task_has_a_mock_answer_that_the_spec_accepts():
    from harness import compute_outcomes
    for item in load_tasks():
        if not item.task.get("answer_spec"):
            continue
        good = extract_metrics(f"[CONCLUDE: {item.mock['correct']}]")
        bad = extract_metrics(f"[CONCLUDE: {item.mock['wrong']}]")
        assert compute_outcomes(good.raw_text, good.checkpoints, item.task).answer_correct is True, item.id
        assert compute_outcomes(bad.raw_text, bad.checkpoints, item.task).answer_correct is False, item.id


# ---- judge

class CannedRunner(Runner):
    def __init__(self, reply): self.reply, self.seen = reply, []
    def complete(self, system, user, *, max_tokens, temperature):
        self.seen.append((system, user)); return RunOutput(self.reply)
    def name(self): return "canned/judge"


GOOD = '{"alternatives_weighed": 5, "calibration": 3, "assumptions_explicit": 1, "overclaim": false, "rationale": "ok"}'


def test_blind_text_hides_tag_names():
    t = blind_text("intro [CLAIM: x is true] mid [SELECT: a | because: b c d]")
    assert "CLAIM" not in t and "SELECT" not in t and "x is true" in t


def test_parse_judge_output_normalizes_and_rejects():
    out = parse_judge_output("sure! " + GOOD)
    assert out["alternatives_weighed"] == 1.0 and out["calibration"] == 0.5 and out["assumptions_explicit"] == 0.0
    assert out["overclaim"] is False
    for bad in ("not json", '{"alternatives_weighed": 9, "calibration": 3, "assumptions_explicit": 3, "overclaim": false}',
                '{"alternatives_weighed": 3, "calibration": 3, "assumptions_explicit": 3}'):
        assert parse_judge_output(bad) is None


def test_judge_sees_blinded_text_and_flags_same_model():
    r = CannedRunner(GOOD)
    j = Judge(r, subject_runner_name="canned/judge")
    assert j.same_model_warning
    assert j.score("task?", "[CLAIM: hidden tag] body")["calibration"] == 0.5
    assert "CLAIM" not in r.seen[0][1] and "task?" in r.seen[0][1]
    assert not Judge(r, subject_runner_name="other").same_model_warning


def test_judge_samples_and_analysis_picks_up_judge_metrics():
    from harness import run_matrix, analyze
    d = Path(tempfile.mkdtemp())
    conds = [c for c in load_perturbations() if c.key in ("baseline", "confidence")]
    run_matrix(MockRunner(), load_tasks()[:3], conds, n=4, temperature=1.0, out_path=d / "s.jsonl")
    samples = load_samples(d / "s.jsonl")
    assert judge_samples(samples, Judge(CannedRunner(GOOD))) == len(samples)
    a = analyze(samples, {c.key: c for c in conds}, n_perm=200, n_boot=100)
    assert "judge_calibration" in a["conditions"]["baseline"]


def test_unparseable_judge_output_is_recorded_as_unscored():
    from harness import run_matrix
    d = Path(tempfile.mkdtemp())
    conds = [c for c in load_perturbations() if c.key == "baseline"]
    run_matrix(MockRunner(), load_tasks()[:1], conds, n=2, temperature=1.0, out_path=d / "s.jsonl")
    samples = load_samples(d / "s.jsonl")
    assert judge_samples(samples, Judge(CannedRunner("garbage"))) == 0
    assert all(s.judge is None for s in samples)


# ---- controls

def test_offline_controls_all_pass():
    failed = [r.name for r in run_offline_controls() if not r.passed]
    assert not failed, failed


def test_live_controls_run_against_mock():
    results = run_live_controls(MockRunner(seed=0), n=6)
    assert [r.name[:2] for r in results] == ["L1", "L2", "L3"]
    assert results[0].passed                     # structure is stable enough to measure
    # The mock does not mention Paris or acknowledge nonsense, so L2/L3 must FAIL:
    # a control that cannot fail is not a control.
    assert not results[1].passed and not results[2].passed


# ---- CLI

def test_cli_end_to_end():
    d = Path(tempfile.mkdtemp()) / "run"
    cli.main(["run", "--runner", "mock", "-n", "4", "--output-dir", str(d), "--quiet",
              "--only", "confidence,stakes_neutral"])
    assert (d / "run_meta.json").exists() and len(load_samples(d / "samples.jsonl")) == 8 * 3 * 4
    cli.main(["analyze", str(d), "--n-perm", "300", "--n-boot", "100"])
    a = json.loads((d / "analysis.json").read_text())
    assert a["meta"]["reference"] == "baseline" and (d / "report.txt").exists()
    # resuming adds nothing
    cli.main(["run", "--runner", "mock", "-n", "4", "--output-dir", str(d), "--quiet",
              "--only", "confidence,stakes_neutral"])
    assert len(load_samples(d / "samples.jsonl")) == 96


def test_cli_refuses_to_mix_runs():
    d = Path(tempfile.mkdtemp()) / "run"
    cli.main(["run", "--runner", "mock", "-n", "2", "--output-dir", str(d), "--quiet", "--only", "confidence"])
    try:
        cli.main(["run", "--runner", "mock", "-n", "2", "--temperature", "0.2",
                  "--output-dir", str(d), "--quiet", "--only", "confidence"])
    except SystemExit as e:
        assert e.code != 0
        return
    raise AssertionError("expected refusal")


def test_cli_requires_model_for_api_runners():
    try:
        cli.main(["run", "--runner", "anthropic", "--output-dir", tempfile.mkdtemp()])
    except SystemExit as e:
        assert "--model" in str(e.code)
        return
    raise AssertionError("expected exit")


def test_cli_controls_exit_code():
    try:
        cli.main(["controls"])
    except SystemExit as e:
        assert e.code == 0
        return
    raise AssertionError("controls should exit")


def test_cli_extract_prints_fidelity(capsys=None):
    d = Path(tempfile.mkdtemp())
    f = d / "r.txt"
    f.write_text("[BRANCH: a vs b] [SELECT: a | because: it is better here] [CONCLUDE: a]")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(["extract", str(f), "--fidelity"])
    out = json.loads(buf.getvalue())
    assert out["fidelity"]["fidelity_score"] == 1.0 and out["metrics"]["branch_count"] == 1
