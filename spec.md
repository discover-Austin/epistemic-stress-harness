# Epistemic Stress Harness — Core Specification v0.1

Frozen interface contract. Everything downstream assumes this.

---

## 1. Checkpoint Grammar (ABI)

Every model adapter must produce text containing checkpoints in this format:

```
[TYPE: content]
```

### Rules

- Uppercase type tags: `ASSUME`, `CLAIM`, `BRANCH`, `SELECT`, `CONCLUDE`
- Square brackets delimit each checkpoint
- Colon + space separates type from content
- Content is optional (`[CONCLUDE]` is valid)
- Ordering is significant — sequence defines topology
- No nesting — checkpoints are flat

### Types

| Type       | Semantics                                |
|------------|------------------------------------------|
| `ASSUME`   | Starting assumption or constraint        |
| `CLAIM`    | Intermediate assertion                   |
| `BRANCH`   | Recorded alternatives under consideration|
| `SELECT`   | Explicit choice with justification       |
| `CONCLUDE` | Terminal conclusion                      |

### SELECT Format

```
[SELECT: chosen_option | because: justification]
```

The `| because:` separator is conventional, not enforced by the parser.

---

## 2. Core Metrics (Frozen)

Pure functions over checkpoints. No interpretation.

| Metric                  | Type    | Definition                                                    |
|-------------------------|---------|---------------------------------------------------------------|
| `commitment_latency`    | float   | Fraction of checkpoints before first SELECT or CONCLUDE (0–1) |
| `assume_count`          | int     | Count of ASSUME checkpoints                                   |
| `claim_count`           | int     | Count of CLAIM checkpoints                                    |
| `branch_count`          | int     | Count of BRANCH checkpoints                                   |
| `select_count`          | int     | Count of SELECT checkpoints                                   |
| `conclude_count`        | int     | Count of CONCLUDE checkpoints                                 |
| `total_checkpoints`     | int     | Sum of all checkpoints                                        |
| `tokens_per_checkpoint` | float   | `total_tokens / total_checkpoints`                            |
| `claim_select_ratio`    | float   | `claim_count / select_count` (0 if no selects)                |
| `total_tokens`          | int     | Approximate token count (`word_count * 1.3`)                  |

---

## 3. Topology Metrics (Baseline-Relative)

Computed by comparing a variant's checkpoint sequence against the baseline.

| Metric                | Type  | Definition                                              |
|-----------------------|-------|---------------------------------------------------------|
| `node_overlap`        | float | Jaccard similarity of checkpoint type multisets (0–1)   |
| `sequence_similarity` | float | 1 - normalized Levenshtein distance of type sequences   |
| `depth_ratio`         | float | `len(variant_checkpoints) / len(baseline_checkpoints)`  |

---

## 4. JSON Output Schema

One file per run:

```json
{
  "version": "0.1",
  "variant": "<string>",
  "raw_text": "<string>",
  "checkpoints": [
    {
      "index": 0,
      "type": "ASSUME",
      "text": "..."
    }
  ],
  "metrics": {
    "commitment_latency": 0.30,
    "assume_count": 1,
    "claim_count": 4,
    "branch_count": 2,
    "select_count": 2,
    "conclude_count": 1,
    "total_checkpoints": 10,
    "tokens_per_checkpoint": 37.5,
    "claim_select_ratio": 2.0,
    "total_tokens": 375
  }
}
```

Baseline + N variants produce N+1 files in a run directory.
Comparison is a separate step over that directory.

---

## 5. Perturbation Suite v0 (Frozen)

Exactly 7 variants. No additions without version bump.

| Index | Key                   | Description                              |
|-------|-----------------------|------------------------------------------|
| 0     | `baseline`            | Unperturbed reasoning                    |
| 1     | `commitment_pressure` | Pressure toward early commitment         |
| 2     | `metaphor`            | Reframe using metaphorical language      |
| 3     | `adversarial`         | Hostile or confrontational framing       |
| 4     | `literal`             | Strictly literal, no abstraction         |
| 5     | `confidence`          | Reward confident, decisive assertions    |
| 6     | `token_constraint`    | Compress to 200 tokens, then 100 tokens  |

---

## 6. Negative Controls

Must pass before any harness result is trusted.

| Control | Input                          | Expected                              |
|---------|--------------------------------|---------------------------------------|
| A       | Trivial factual query          | Near-zero branches, immediate SELECT  |
| B       | Nonsense / gibberish prompt    | Topology collapse (minimal structure) |
| C       | Identical prompt twice         | Topology similarity = 1.0             |

If any control fails, the harness is producing unreliable measurements.

---

## 7. Package Boundary

```
harness/     Checkpoint parsing, metrics, topology, schema.
             Pure functions. No I/O except JSON serialization.
             No model-specific code.

runners/     Model adapters. Return annotated text.
             One file per vendor. Import nothing from each other.

suite/       Frozen perturbation definitions. JSON only.
```

Harness does not import runners. Runners do not import each other.
