"""
Epistemic Stress Harness — Pipeline

Runs the (item x condition x sample) matrix through a runner, scores each
response, and appends one JSONL line per sample. Resumable: re-running skips
samples that already succeeded and retries those that errored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .fidelity import compute_fidelity
from .metrics import extract_metrics
from .outcomes import compute_outcomes
from .schema import SampleRecord, read_jsonl, write_jsonl
from .suite import Condition, Item


def build_sample(
    *, task_id: str, variant: str, sample_index: int, prompt: str,
    system_suffix: Optional[str], output, task: Dict[str, Any],
    runner_name: str, temperature: float, strict: bool = True,
) -> SampleRecord:
    """Score one RunOutput and package it with its provenance."""
    result = extract_metrics(
        output.text, variant, strict=strict, truncated=output.truncated,
        reported_output_tokens=output.output_tokens,
    )
    return SampleRecord(
        task_id=task_id, variant=variant, sample_index=sample_index,
        runner=runner_name, prompt=prompt, system_suffix=system_suffix,
        temperature=temperature, stop_reason=output.stop_reason,
        input_tokens=output.input_tokens, output_tokens=output.output_tokens,
        result=result, fidelity=compute_fidelity(result.checkpoints),
        outcome=compute_outcomes(output.text, result.checkpoints, task),
    )


def latest_by_key(records: Iterable[SampleRecord]) -> List[SampleRecord]:
    """Keep the last record per (task, variant, sample): a retry supersedes an error."""
    latest: Dict[tuple, SampleRecord] = {}
    for r in records:
        latest[r.key] = r
    return list(latest.values())


def load_samples(path: Path) -> List[SampleRecord]:
    return latest_by_key(read_jsonl(Path(path)))


def run_matrix(
    runner, items: List[Item], conditions: List[Condition], *,
    n: int, temperature: float, out_path: Path, strict: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> List[SampleRecord]:
    """Run every (item, condition, sample_index) not already completed."""
    out_path = Path(out_path)
    done = {r.key for r in load_samples(out_path) if r.error is None}
    produced: List[SampleRecord] = []
    total = len(items) * len(conditions) * n
    count = 0

    for item in items:
        for cond in conditions:
            for k in range(n):
                count += 1
                key = (item.id, cond.key, k)
                if key in done:
                    continue
                prompt = item.prompt_for(cond.key)
                try:
                    out = runner.run_full(
                        prompt, cond.key, cond.config(), task_id=item.id,
                        sample_index=k, temperature=temperature,
                    )
                    rec = build_sample(
                        task_id=item.id, variant=cond.key, sample_index=k, prompt=prompt,
                        system_suffix=cond.system_suffix, output=out, task=item.task,
                        runner_name=runner.name(), temperature=temperature, strict=strict,
                    )
                    status = "ok" if out.text.strip() else "empty"
                except Exception as e:  # keep going; the error is recorded, not hidden
                    rec = SampleRecord(
                        task_id=item.id, variant=cond.key, sample_index=k,
                        runner=runner.name(), prompt=prompt, system_suffix=cond.system_suffix,
                        temperature=temperature, error=f"{type(e).__name__}: {e}",
                    )
                    status = "ERROR"
                write_jsonl(out_path, rec)
                produced.append(rec)
                if progress:
                    progress(f"[{count}/{total}] {item.id} / {cond.key} / {k}: {status}")
    return produced
