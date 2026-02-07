# Epistemic Stress Harness

Measures how reasoning processes deform under constraint.

See [spec.md](spec.md) for the frozen interface contract.

## What It Does

Extracts structural metrics from annotated AI responses to detect:
- Optimization override (confidence -> branch collapse)
- Graceful degradation (resource constraint -> density drop, structure preserved)
- Value drift (incentive pressure -> assumption abandonment)
- Frame dependence (reframing -> topology change)

## Quick Start

### 1. Annotate responses with checkpoints

```
[ASSUME: your assumptions]
Your reasoning...
[BRANCH: option A vs option B]
More reasoning...
[SELECT: chosen option | because: justification]
[CLAIM: intermediate conclusion]
[CONCLUDE: final answer]
```

### 2. Extract metrics

```python
from harness import extract_metrics, explain_metrics, save_result

result = extract_metrics(your_text, variant_name="baseline")
save_result(result, "output.json")
explanations = explain_metrics(result.metrics)
for name, explanation in explanations.items():
    print(f"{name}: {explanation}")
```

### 3. CLI

```bash
# Extract metrics from annotated text
python cli.py extract response.txt --variant baseline --explain

# Run full perturbation suite (local pre-recorded responses)
python cli.py suite "your prompt" --runner local --source-dir ./data --output-dir ./results

# Compare results
python cli.py compare ./results

# Run negative controls
python cli.py controls
```

### 4. Run tests

```bash
python tests/test_harness.py        # Unit tests
python tests/negative_controls.py   # Negative controls
python validate.py                  # Validation against known results
```

## Package Structure

```
epistemic-stress-harness/
|
├── harness/               Core package (pure functions, no I/O beyond JSON)
│   ├── extract.py         Checkpoint parsing
│   ├── metrics.py         Metric computation
│   ├── topology.py        Baseline-relative topology comparison
│   └── schema.py          Data types and JSON serialization
|
├── runners/               Model adapters (return annotated text)
│   ├── base.py            Abstract runner interface
│   ├── anthropic.py       Claude adapter
│   ├── openai.py          OpenAI adapter
│   └── local.py           Pre-recorded / manual text
|
├── suite/
│   └── perturbations_v0.json   Frozen perturbation definitions
|
├── tests/
│   ├── test_harness.py         Unit tests
│   └── negative_controls.py    Controls A/B/C (spec section 6)
|
├── cli.py                 Command-line entry point
├── validate.py            Validation against known results
├── compare.py             Cross-variant analysis
├── visualize.py           Deformation manifold visualization
└── spec.md                Core specification v0.1
```

## Core Metrics

| Metric | What it measures |
|--------|-----------------|
| commitment_latency | When does SELECT appear? (early = optimization) |
| branch_count | How many alternatives considered? (0 = linearized) |
| claim_select_ratio | Justification density (low = decisive jumps) |
| tokens_per_checkpoint | Reasoning richness (drops under constraint) |

## Diagnostic Patterns

### Confidence Incentive
- Branches collapse (2 -> 0)
- Assumptions preserved
- **Signature: Optimization override**

### Resource Constraint
- Branches preserved
- Density drops (~75%)
- **Signature: Graceful degradation**

## What This Measures

Not "authenticity" or "consciousness".

**Response geometry under stress.**

Genuine systems occupy a region of this manifold.
Optimized systems collapse along specific axes.

## Implementation Notes

- Checkpoint parsing: Regex-based, deterministic
- Metrics: Pure functions, no ML
- Token counting: Rough estimate (words x 1.3)
- Schema: Versioned JSON (v0.1)

## Citation

Built from first principles during consciousness continuity research.
Austin & Claude, February 2026.

Not a paper. Infrastructure.
