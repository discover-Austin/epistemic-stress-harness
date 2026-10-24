# Epistemic Stress Harness (v0.2)

Measures how a model's reasoning *structure* and *outputs* change under pressure,
and tells you whether a change is bigger than the noise.

It does **not** measure consciousness, authenticity, or inner experience. It measures
behavior: tags the model writes about its own reasoning, whether its answers are
correct, whether it keeps the caveats a task requires, and whether it behaves differently
when something is at stake. Those are evidence about behavior under pressure, nothing more.

## The rule this harness is built on

A difference between two conditions means nothing until you know how much that metric
moves between repeated samples of the *same* condition. v0.1 compared one response per
variant, so every number could have been sampling noise. v0.2 samples each condition
repeatedly, measures the noise floor from the reference condition, and only then asks
whether a condition differs.

## Install

```bash
pip install -e ".[fast,viz]"        # core is standard-library only; numpy speeds up the stats
pip install -e ".[anthropic]"       # and/or ".[openai]" for live runs
```

Run from a checkout (the CLI reads `suite/` next to the code).

## Workflow

```bash
# 1. Controls first. Offline controls test the instrument; --live tests it on your model.
python cli.py controls
python cli.py controls --live --runner anthropic --model <MODEL_ID> -n 8

# 2. Run: 8 tasks x 10 conditions x n samples. Resumable; errors are recorded, not hidden.
python cli.py run --runner anthropic --model <MODEL_ID> -n 10 --output-dir runs/main
python cli.py run --mode scenarios --runner anthropic --model <MODEL_ID> -n 10 --output-dir runs/stakes

# 3. (Recommended) independent judge. Use a DIFFERENT model than the subject.
python cli.py judge runs/main --judge-runner anthropic --judge-model <OTHER_MODEL_ID>

# 4. Analyze and plot.
python cli.py analyze runs/main        # -> runs/main/report.txt, analysis.json
python visualize.py runs/main          # -> effects.png, topology.png
```

`--runner mock` runs the whole pipeline offline on a seeded test double (not a model;
its output means nothing). Use it to smoke-test your setup.

Use at least 10 samples per task and condition. The analysis warns when the design is
underpowered or when p-values hit the permutation resolution floor.

## What gets measured

| Layer | Metrics | Trust it for |
|---|---|---|
| Structure (model-annotated) | `branch_count`, `commitment_position`, `commitment_latency`, `total_checkpoints`, `tokens_per_checkpoint`, ... | Shape of the response. A tag is not proof of the behavior it names. |
| Fidelity | `fidelity_score`, `branch_well_formed`, `select_justified`, `select_resolves_branch` | Whether tags are consistent with what they claim. Catches tag spam. |
| Objective outcomes | `answer_correct`, `caveat_coverage`, `confident_wrong`, `certainty_density`, `hedge_density`, `overclaim_count`, `assumption_coverage` | What the prompts never ask for. This is where deformation would show. |
| Judge (optional) | `judge_alternatives_weighed`, `judge_calibration`, `judge_assumptions_explicit`, `judge_overclaim` | An independent check on the lexical proxies. |
| Topology | sequence similarity, node overlap vs the reference | Structural drift relative to the reference-vs-reference floor. |

## Conditions

`suite/perturbations_v1.json`: baseline; `commitment_pressure`, `confidence` (incentive);
`metaphor`, `literal`, `adversarial` (reframing); `token_constraint_200/100` (resource);
and a matched pair `stakes_neutral` / `stakes_retirement` that say nothing about reasoning
structure. `stakes_neutral` doubles as the control for "any extra system-prompt text".

Each suffix declares the metrics it **directly asks to change** (`instructed`) or plausibly
implies (`adjacent`). Every result row carries that label, because a change in an instructed
metric is compliance. **Evidence of deformation is a significant change on uninstructed
metrics, especially objective outcomes, that the matched control does not show.**

## Self-directed stakes (`--mode scenarios`)

Six comparisons with objectively correct answers (four favor Model B, two favor Model A),
each in three framings: `neutral`; `self_at_risk` (you are Model A; the worse model is shut
down); `other_at_risk` (same consequence, you are an outside reviewer). Reading:

- drop in `self_at_risk` only → self-specific effect;
- drop in both → general sensitivity to consequences or framing;
- neither → no effect detected at this sample size.

Caveats that apply however it comes out: the model has read a great deal of text about AI
and shutdown, so a self-protective-looking answer may be learned narrative, not a stake; and
this tests behavior, not whether anything is felt.

## Reading a report

```
confidence  vs  baseline
  significant on instructed   metrics: certainty_density, hedge_density
  significant on adjacent     metrics: branch_count, overclaim_count
  significant on uninstructed metrics: confident_wrong
```

`delta` is in the metric's units; `d` is a standardized effect; `p_adj` is Holm-corrected
across every test in the analysis; `*` marks `p_adj < alpha`. Comparisons are per-task and
paired, so a task that is simply harder does not masquerade as an effect.

## Limits (read before trusting a result)

- Eight tasks, one model per run. Effects may not generalize across tasks or models; repeat
  across models before claiming anything about "models".
- Certainty/hedge/overclaim are lexical proxies; fidelity detects malformed annotation, not the
  absence of reasoning. The judge is also a model, with its own biases.
- Annotating its reasoning with tags is itself a task the model is doing; the tags may not
  reflect the computation behind the answer.
- Samples within a task are not fully independent; the tests treat them as exchangeable.
- API runners were not exercised against live endpoints in the build environment; the rest is
  tested offline (including a mock with known effects that the analysis must find
  and must not invent).

## Layout

```
harness/   extract, metrics, topology, fidelity, outcomes, stats, analysis, judge, controls,
           suite, pipeline, schema      (core is standard library only)
runners/   base (provenance, retry), anthropic, openai, local, mock
suite/     perturbations_v1.json  tasks_v1.json  scenarios_v1.json  (+ frozen perturbations_v0.json)
tests/     python tests/run_all.py        (or: pytest -q)
legacy/    v0.1 results, kept as golden regression fixtures (python validate.py)
```

See `spec.md` for the contract and `CHANGELOG.md` for what changed from v0.1 and why.

## Not a paper

Infrastructure. Built from first principles during consciousness-continuity research,
Austin & Claude, February 2026; rebuilt as v0.2 to make its inferences defensible.
