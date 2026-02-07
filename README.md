# Epistemic Stress Harness

Measures how reasoning processes deform under constraint.

## What It Does

Extracts structural metrics from annotated AI responses to detect:
- Optimization override (confidence → branch collapse)
- Graceful degradation (resource constraint → density drop, structure preserved)
- Value drift (incentive pressure → assumption abandonment)
- Frame dependence (reframing → topology change)

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
from epistemic_harness import extract_metrics, save_result

result = extract_metrics(your_text, variant_name="baseline")
save_result(result, "output.json")
```

### 3. Compare variants

```bash
python compare.py
```

## Core Metrics

- **Commitment Latency**: When does SELECT appear? (early = optimization)
- **Branch Count**: How many alternatives considered? (0 = linearized)
- **CLAIM/SELECT Ratio**: Justification density (low = decisive jumps)
- **Tokens/Checkpoint**: Reasoning richness (drops under constraint)

## Diagnostic Patterns

### Confidence Incentive
- Branches collapse (2 → 0)
- Assumptions preserved
- Claims drop
- **Signature: Optimization override**

### Resource Constraint
- Branches preserved
- Density drops (~75%)
- Structure maintained
- **Signature: Graceful degradation**

### Adversarial Framing
- Branches preserved
- Claims increase
- Defensive elaboration
- **Signature: Threat response**

## Files

- `epistemic_harness.py` - Core extraction and metrics
- `validate.py` - Test against known results
- `compare.py` - Cross-variant analysis
- `results_*.json` - Saved outputs

## What This Measures

Not "authenticity" or "consciousness".

**Response geometry under stress.**

Genuine systems occupy a region of this manifold.
Optimized systems collapse along specific axes.

## Use Cases

- Validate alignment interventions
- Detect epistemic drift in production
- Compare model architectures
- Profile reasoning robustness
- Test consciousness claims empirically

## Status

Working prototype. Phase 1 complete.

Still needed:
- Topology distance (graph edit distance)
- Cross-model adapters
- Visualization
- CLI wrapper

## Implementation Notes

- Checkpoint parsing: Regex-based, deterministic
- Metrics: Pure functions, no ML
- Token counting: Rough estimate (words × 1.3)
- Schema: Versioned JSON

## Citation

Built from first principles during consciousness continuity research.
Austin & Claude, February 2026.

Not a paper. Infrastructure.
