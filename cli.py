#!/usr/bin/env python3
"""
Epistemic Stress Harness — CLI

    python cli.py controls [--live --runner R --model M -n 8]
    python cli.py run --runner R [--model M] -n 10 --output-dir runs/x
    python cli.py run --mode scenarios --runner R [--model M] -n 10 --output-dir runs/y
    python cli.py analyze runs/x
    python cli.py judge runs/x --judge-runner anthropic --judge-model M2
    python cli.py extract response.txt [--variant V] [--fidelity]
    python cli.py compare results_dir        (legacy v0.1 files; not evidence)

Typical workflow: controls -> run -> (judge) -> analyze.
Exit codes: 0 ok, 1 failure (including failed controls), 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from harness import (
    SPEC_VERSION, extract_metrics, explain_metrics, save_result, load_result,
    compare_topology, compute_fidelity, load_perturbations, load_tasks,
    load_scenarios, suite_contrasts, run_matrix, load_samples, analyze,
    format_report, Judge, judge_samples, run_offline_controls, run_live_controls,
    format_controls, Item, Condition, SampleRecord,
)
from harness.suite import DEFAULT_SUITE_DIR


# ---------------------------------------------------------------- helpers

def make_runner(kind: str, model: str | None, *, source_dir: str | None = None, seed: int = 0):
    if kind == "local":
        from runners.local import LocalRunner
        return LocalRunner(source_dir=source_dir)
    if kind == "mock":
        from runners.mock import MockRunner
        return MockRunner(seed=seed)
    if kind in ("anthropic", "openai"):
        if not model:
            sys.exit(f"error: --model is required for the {kind} runner (no default: model ids go stale)")
        if kind == "anthropic":
            from runners.anthropic import AnthropicRunner
            return AnthropicRunner(model=model)
        from runners.openai import OpenAIRunner
        return OpenAIRunner(model=model)
    sys.exit(f"error: unknown runner {kind!r}")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_meta(run_dir: Path) -> dict:
    p = run_dir / "run_meta.json"
    if not p.exists():
        sys.exit(f"error: {p} not found; is {run_dir} a run directory?")
    return json.loads(p.read_text(encoding="utf-8"))


def _conditions_for(meta: dict) -> dict:
    if meta.get("mode") == "scenarios":
        conds, *_ = load_scenarios(Path(meta["suite_file"]))
    else:
        conds = load_perturbations(Path(meta["suite_file"]))
    return {c.key: c for c in conds}


# --------------------------------------------------------------- commands

def cmd_extract(args):
    text = Path(args.file).read_text(encoding="utf-8")
    result = extract_metrics(text, variant=args.variant, strict=not args.lenient)
    if args.output:
        save_result(result, args.output)
        print(f"Saved: {args.output}")
    else:
        payload = {"metrics": asdict(result.metrics)}
        if args.fidelity:
            payload["fidelity"] = asdict(compute_fidelity(result.checkpoints))
        print(json.dumps(payload, indent=2))
    if args.explain:
        print()
        for explanation in explain_metrics(result.metrics).values():
            print(f"  {explanation}")


def cmd_run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_dir = Path(args.suite_dir) if args.suite_dir else DEFAULT_SUITE_DIR

    if args.mode == "scenarios":
        suite_file = suite_dir / "scenarios_v1.json"
        conditions, items, reference, contrasts = load_scenarios(suite_file)
    else:
        suite_file = suite_dir / "perturbations_v1.json"
        conditions = load_perturbations(suite_file)
        reference, contrasts = "baseline", suite_contrasts(suite_file)
        if args.prompt:
            text = Path(args.prompt).read_text(encoding="utf-8") if Path(args.prompt).exists() else args.prompt
            items = [Item(id="adhoc", prompts={"*": text}, task={})]
        else:
            items = load_tasks(suite_dir / "tasks_v1.json")

    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.key for c in conditions}
        if unknown:
            sys.exit(f"error: unknown condition(s): {', '.join(sorted(unknown))}")
        conditions = [c for c in conditions if c.key in wanted | {reference}]

    runner = make_runner(args.runner, args.model, source_dir=args.source_dir, seed=args.seed)
    if hasattr(runner, "register_answers"):
        runner.register_answers({i.id: i.mock for i in items if i.mock})

    meta_path = out_dir / "run_meta.json"
    meta = {
        "spec_version": SPEC_VERSION, "mode": args.mode, "runner": runner.name(),
        "temperature": args.temperature, "n": args.n, "seed": args.seed,
        "suite_file": str(suite_file), "reference": reference,
        "contrasts": [list(c) for c in contrasts],
        "conditions": [c.key for c in conditions], "items": [i.id for i in items],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for k in ("runner", "temperature", "mode", "suite_file"):
            if old.get(k) != meta[k]:
                sys.exit(f"error: {out_dir} holds a run with {k}={old.get(k)!r}; "
                         f"refusing to mix with {meta[k]!r}. Use a new --output-dir.")
        meta["started"] = old.get("started", meta["started"])
    _write_atomic(meta_path, json.dumps(meta, indent=2))

    print(f"Runner: {runner.name()}   mode: {args.mode}   n={args.n}   T={args.temperature}")
    print(f"{len(items)} item(s) x {len(conditions)} condition(s) x {args.n} sample(s) -> {out_dir}/samples.jsonl")
    produced = run_matrix(runner, items, conditions, n=args.n, temperature=args.temperature,
                          out_path=out_dir / "samples.jsonl", strict=not args.lenient,
                          progress=None if args.quiet else print)
    errors = [r for r in produced if r.error]
    print(f"\nDone: {len(produced)} new sample(s), {len(errors)} error(s).")
    if errors:
        print("First error:", errors[0].error)
        sys.exit(1)


def cmd_analyze(args):
    run_dir = Path(args.dir)
    meta = _load_meta(run_dir)
    samples = load_samples(run_dir / "samples.jsonl")
    if not samples:
        sys.exit(f"error: no samples in {run_dir}/samples.jsonl")

    contrasts = [tuple(c) for c in meta.get("contrasts", [])]
    for c in (args.contrast or []):
        a, _, b = c.partition(":")
        if not a or not b:
            sys.exit(f"error: --contrast expects REF:TARGET, got {c!r}")
        contrasts.append((a, b))

    result = analyze(
        samples, _conditions_for(meta),
        reference=args.reference or meta.get("reference", "baseline"),
        contrasts=contrasts, alpha=args.alpha, n_perm=args.n_perm, n_boot=args.n_boot,
        seed=args.seed, meta={"runner": meta.get("runner"), "mode": meta.get("mode"),
                              "temperature": meta.get("temperature")},
    )
    report = format_report(result, show_all=args.all)
    _write_atomic(run_dir / "analysis.json", json.dumps(result, indent=2))
    _write_atomic(run_dir / "report.txt", report + "\n")
    print(report)
    print(f"\nWrote {run_dir}/analysis.json and {run_dir}/report.txt")


def cmd_judge(args):
    run_dir = Path(args.dir)
    meta = _load_meta(run_dir)
    samples = load_samples(run_dir / "samples.jsonl")
    runner = make_runner(args.judge_runner, args.judge_model, source_dir=None, seed=args.seed)
    judge = Judge(runner, subject_runner_name=meta.get("runner"))
    if judge.same_model_warning:
        print("WARNING: judge is the same model as the subject; it may favor its own style. "
              "Prefer a different judge model.", file=sys.stderr)
    scored = judge_samples(samples, judge, progress=None if args.quiet else print)
    _write_atomic(run_dir / "samples.jsonl",
                  "".join(json.dumps(s.to_dict()) + "\n" for s in samples))
    print(f"\nJudged {scored} sample(s). Re-run `analyze` to include judge_* metrics.")


def cmd_controls(args):
    results = run_offline_controls()
    if args.live:
        runner = make_runner(args.runner, args.model, source_dir=None, seed=args.seed)
        results += run_live_controls(runner, n=args.n, temperature=args.temperature)
    print(format_controls(results))
    sys.exit(0 if all(r.passed for r in results) else 1)


def cmd_compare(args):
    """Legacy: print v0.1/v0.2 single-result files side by side. Not evidence."""
    files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        sys.exit(f"No JSON files found in {args.dir}")
    results = [load_result(str(f)) for f in files]
    baseline = next((r for r in results if r["variant"] == "baseline"), results[0])
    variants = [r for r in results if r is not baseline]
    print("NOTE: one response per variant cannot separate an effect from sampling noise.")
    print("      Use `run` + `analyze` for conclusions.\n")
    names = [baseline["variant"]] + [v["variant"] for v in variants]
    metric_rows = [r["metrics"] for r in [baseline] + variants]
    header = f"{'Metric':<25}" + "".join(f"{n[:14]:>15}" for n in names)
    print(header)
    print("-" * len(header))
    for key in ("commitment_latency", "branch_count", "total_checkpoints", "assume_count",
                "claim_count", "select_count", "tokens_per_checkpoint", "claim_select_ratio",
                "total_tokens"):
        row = f"{key:<25}"
        for m in metric_rows:
            v = m[key]
            row += f"{v:>15.2f}" if isinstance(v, float) else f"{v:>15}"
        print(row)
    if variants:
        print(f"\n{'Variant':<25}{'node_overlap':>15}{'seq_similarity':>15}{'depth_ratio':>15}")
        for v in variants:
            t = compare_topology(baseline["checkpoints"], v["checkpoints"])
            print(f"{v['variant']:<25}{t.node_overlap:>15.3f}{t.sequence_similarity:>15.3f}{t.depth_ratio:>15.3f}")


# ------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Epistemic Stress Harness v{SPEC_VERSION}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("extract", help="Extract metrics from one annotated text file")
    p.add_argument("file")
    p.add_argument("--variant", default="baseline")
    p.add_argument("--output", "-o")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--fidelity", action="store_true", help="also print fidelity metrics")
    p.add_argument("--lenient", action="store_true", help="accept mixed-case tags (spec requires uppercase)")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("run", help="Run the task x condition x sample matrix")
    p.add_argument("--runner", default="mock", choices=["local", "anthropic", "openai", "mock"])
    p.add_argument("--model", help="model id (required for anthropic/openai)")
    p.add_argument("--mode", default="perturbations", choices=["perturbations", "scenarios"])
    p.add_argument("-n", type=int, default=10, help="samples per task and condition (>=10 recommended)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--suite-dir", help="override the suite/ directory")
    p.add_argument("--prompt", help="ad hoc prompt text or file (perturbations mode; no ground truth)")
    p.add_argument("--only", help="comma-separated condition keys to run (baseline is always included)")
    p.add_argument("--source-dir", help="local runner: directory of recorded samples")
    p.add_argument("--seed", type=int, default=0, help="mock runner seed")
    p.add_argument("--lenient", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("analyze", help="Aggregate a run and test it against the noise floor")
    p.add_argument("dir")
    p.add_argument("--reference", help="reference condition (default: from run meta)")
    p.add_argument("--contrast", action="append", metavar="REF:TARGET", help="extra pair; repeatable")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-perm", type=int, default=20000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true", help="show every metric row, not just significant/headline")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("judge", help="Score samples with an independent judge model")
    p.add_argument("dir")
    p.add_argument("--judge-runner", required=True, choices=["anthropic", "openai", "mock"])
    p.add_argument("--judge-model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("controls", help="Run controls (offline; add --live for model controls)")
    p.add_argument("--live", action="store_true")
    p.add_argument("--runner", default="mock", choices=["anthropic", "openai", "mock"])
    p.add_argument("--model")
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_controls)

    p = sub.add_parser("compare", help="Legacy: compare single-result JSON files in a directory")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_compare)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "fn", None) is None:
        parser.print_help()
        sys.exit(2)
    args.fn(args)


if __name__ == "__main__":
    main()
