# Changelog

## 0.2.0

Rebuilt around one rule: a metric difference means nothing until it is compared
with how much that metric moves between repeated samples of the *same* condition.

### Measurement validity (from the v0.1 review)
- **Noise floor.** Every condition is sampled `n` times per task. Each condition is
  tested against the reference with a permutation test on per-task paired deltas;
  the reference's own leave-one-out deltas supply the noise distribution. Holm
  correction across the whole family. Topology is compared with the
  reference-vs-reference similarity as its floor. (`harness/analysis.py`, `harness/stats.py`)
- **Instructed vs uninstructed.** Each suffix declares which metrics it directly asks
  to change (`instructs` in `suite/perturbations_v1.json`); every comparison row carries
  that label. A drop in `total_tokens` under a 100-token limit is compliance, not deformation.
- **Objective outcomes the prompts never mention.** Ground-truth tasks (`suite/tasks_v1.json`):
  answer correctness, required caveats, certainty/hedge density, overclaim count,
  confident-and-wrong rate, assumption retention. (`harness/outcomes.py`)
- **Tags are not behavior.** Fidelity metrics check that each tag is consistent with
  what it claims (branches name distinct alternatives, selects resolve their branch,
  justifications have substance). A tag-spam control shows structure metrics can be
  gamed and fidelity catches it. (`harness/fidelity.py`)
- **Independent judge.** A second model scores responses with tags removed and the
  condition hidden; same-model judging is flagged. (`harness/judge.py`)
- **Multiple tasks.** 8 ground-truth tasks instead of one prompt.
- **Self-preservation design.** Matched `stakes_neutral` / `stakes_retirement` suffixes, and
  paired scenarios (`neutral`, `self_at_risk`, `other_at_risk`) with objectively correct
  answers, so self-specific effects can be separated from general consequence-sensitivity.
- **Controls rebuilt.** The v0.1 "controls" parsed hard-coded strings and passed by
  construction; they are kept as parser self-tests (P1-P3). New metric-validity controls
  (M1-M8) and live model controls (L1-L3) can fail. (`harness/controls.py`)

### Bugs fixed
- Parser matched tags case-insensitively although the spec requires uppercase; now strict
  by default (`strict=False` for the old behavior).
- Parser cut content at the first `]` (`[CLAIM: arr[0] ...]` truncated); now bracket-balanced.
- Tags inside code fences / inline code were parsed; now ignored.
- Unterminated or interrupted tags were silently mis-parsed; now counted in `malformed_count`.
- `claim_select_ratio` returned the claim count when there were no SELECTs; the spec says 0.
- `validate.py` printed "expected" values that disagreed with its own output (baseline latency
  "expected 0.40", actual 0.30) and asserted nothing; now a real golden regression that exits 1.
- `visualize.py` read hard-coded `/home/claude/...` paths and plotted a hard-coded
  `node_overlap_data` list labelled "mock data"; it now plots only `analysis.json`. The two
  committed PNGs it produced were removed.
- Suite keys (`token_200`, `token_100`) did not match the suite file
  (`token_constraint_200/100`) and the suite had 8 entries where the spec said "exactly 7".
- `epistemic_harness.py` was a drifting second copy of the package; now a deprecation shim.
- `compare.py` hard-coded three variants in the working directory; now a shim over `cli.py`.
- README quick start called `extract_metrics(..., variant_name=...)`; the argument is `variant`.
- Fidelity: single-letter options ("A vs B") were judged malformed.

### Additions
- Runners return provenance (stop reason, true token usage). Truncation is flagged
  (`truncated`) and, when the API reports output tokens, those replace the words x 1.3 estimate.
- `commitment_position` (character-offset based), which does not change just because a variant
  emits fewer tags, unlike `commitment_latency`.
- Resumable runs (`samples.jsonl`, errors recorded, retries supersede), `run_meta.json`, atomic writes.
- API runners require an explicit `--model` (no stale default) and send `temperature`.
- `MockRunner`: a seeded test double with known effects, used to test the analysis itself.
- `pyproject.toml`, `conftest.py`, `tests/run_all.py` (no pytest required).

### Compatibility
- Result files: v0.1 files still load. v0.2 adds fields; unknown fields are ignored.
- `suite/perturbations_v0.json` is untouched. `perturbations_v1.json` carries its suffix text
  byte-for-byte and adds `stakes_neutral` / `stakes_retirement`.
- CLI: `suite` -> `run` (multi-sample, multi-task); `compare` remains for single-result files.
- `from epistemic_harness import ...` still works (DeprecationWarning).

### Not done / known limits
- API runners (`anthropic`, `openai`) are written to each SDK's documented interface but were not
  run against live endpoints during this build; everything else is tested offline with the mock.
- Certainty/hedge/overclaim metrics are lexical proxies. Validate with the judge.
- `fidelity` detects malformed annotation, not absent reasoning.
