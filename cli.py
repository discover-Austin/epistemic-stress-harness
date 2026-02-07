#!/usr/bin/env python3
"""
Epistemic Stress Harness — CLI

Usage:
    python cli.py extract <file>                     Extract metrics from annotated text
    python cli.py extract <file> --variant confidence Extract with variant label
    python cli.py compare <dir>                      Compare all results in directory
    python cli.py suite <prompt> --runner local --source-dir ./data
                                                     Run full perturbation suite
    python cli.py controls                           Run negative controls
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from harness import (
    extract_metrics,
    explain_metrics,
    save_result,
    load_result,
    compare_topology,
    SPEC_VERSION,
)


def cmd_extract(args):
    """Extract metrics from a single annotated text file."""
    text = Path(args.file).read_text()
    result = extract_metrics(text, variant=args.variant)

    if args.output:
        save_result(result, args.output)
        print(f"Saved: {args.output}")
    else:
        print(json.dumps(asdict(result.metrics), indent=2))

    if args.explain:
        print()
        for name, explanation in explain_metrics(result.metrics).items():
            print(f"  {explanation}")


def cmd_compare(args):
    """Compare all result JSON files in a directory."""
    result_dir = Path(args.dir)
    files = sorted(result_dir.glob("*.json"))

    if not files:
        print(f"No JSON files found in {result_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for f in files:
        data = load_result(str(f))
        results.append(data)

    # Find baseline
    baseline = None
    variants = []
    for r in results:
        if r["variant"] == "baseline":
            baseline = r
        else:
            variants.append(r)

    if baseline is None:
        print("No baseline found. First result used as baseline.", file=sys.stderr)
        baseline = results[0]
        variants = results[1:]

    b_m = baseline["metrics"]

    # Table header
    names = [baseline["variant"]] + [v["variant"] for v in variants]
    all_metrics = [b_m] + [v["metrics"] for v in variants]

    header = f"{'Metric':<25}" + "".join(f"{n:>15}" for n in names)
    print(header)
    print("-" * len(header))

    metric_keys = [
        "commitment_latency", "branch_count", "total_checkpoints",
        "assume_count", "claim_count", "select_count",
        "tokens_per_checkpoint", "claim_select_ratio", "total_tokens",
    ]

    for key in metric_keys:
        row = f"{key:<25}"
        for m in all_metrics:
            val = m[key]
            if isinstance(val, float):
                row += f"{val:>15.2f}"
            else:
                row += f"{val:>15}"
        print(row)

    # Topology comparison
    if variants:
        print()
        print("Topology (vs baseline):")
        print(f"{'Variant':<25} {'node_overlap':>15} {'seq_similarity':>15} {'depth_ratio':>15}")
        print("-" * 70)

        from harness import compare_topology as _compare
        for v in variants:
            topo = _compare(baseline["checkpoints"], v["checkpoints"])
            print(
                f"{v['variant']:<25} "
                f"{topo.node_overlap:>15.3f} "
                f"{topo.sequence_similarity:>15.3f} "
                f"{topo.depth_ratio:>15.3f}"
            )


def cmd_suite(args):
    """Run the full perturbation suite against a prompt."""
    suite_path = Path(__file__).parent / "suite" / "perturbations_v0.json"
    with open(suite_path) as f:
        suite = json.load(f)

    # Resolve runner
    if args.runner == "local":
        from runners.local import LocalRunner
        runner = LocalRunner(source_dir=args.source_dir)
    elif args.runner == "anthropic":
        from runners.anthropic import AnthropicRunner
        runner = AnthropicRunner(model=args.model or "claude-sonnet-4-20250514")
    elif args.runner == "openai":
        from runners.openai import OpenAIRunner
        runner = OpenAIRunner(model=args.model or "gpt-4o")
    else:
        print(f"Unknown runner: {args.runner}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt
    if Path(prompt).exists():
        prompt = Path(prompt).read_text()

    print(f"Runner: {runner.name()}")
    print(f"Suite: v{suite['version']} ({len(suite['perturbations'])} perturbations)")
    print(f"Output: {output_dir}")
    print()

    for p in suite["perturbations"]:
        key = p["key"]
        config = {k: v for k, v in p.items() if k not in ("key", "index", "description")}

        print(f"  [{p['index']}] {key}...", end=" ", flush=True)

        try:
            text = runner.run(prompt, key, config)
            result = extract_metrics(text, variant=key)
            outfile = output_dir / f"{key}.json"
            save_result(result, str(outfile))
            print(f"OK ({result.metrics.total_checkpoints} checkpoints)")
        except Exception as e:
            print(f"FAIL ({e})")

    print()
    print(f"Done. Results in {output_dir}/")


def cmd_controls(args):
    """Run negative controls to validate harness integrity."""
    from tests.negative_controls import run_controls
    run_controls()


def main():
    parser = argparse.ArgumentParser(
        description=f"Epistemic Stress Harness v{SPEC_VERSION}",
    )
    sub = parser.add_subparsers(dest="command")

    # extract
    p_extract = sub.add_parser("extract", help="Extract metrics from annotated text")
    p_extract.add_argument("file", help="Path to annotated text file")
    p_extract.add_argument("--variant", default="baseline", help="Variant label")
    p_extract.add_argument("--output", "-o", help="Output JSON path")
    p_extract.add_argument("--explain", action="store_true", help="Show metric explanations")

    # compare
    p_compare = sub.add_parser("compare", help="Compare results in a directory")
    p_compare.add_argument("dir", help="Directory containing result JSON files")

    # suite
    p_suite = sub.add_parser("suite", help="Run full perturbation suite")
    p_suite.add_argument("prompt", help="Prompt text or path to prompt file")
    p_suite.add_argument("--runner", default="local", choices=["local", "anthropic", "openai"])
    p_suite.add_argument("--model", help="Model identifier (runner-specific)")
    p_suite.add_argument("--source-dir", help="Source directory for local runner")
    p_suite.add_argument("--output-dir", default="./results", help="Output directory")

    # controls
    sub.add_parser("controls", help="Run negative controls")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "extract": cmd_extract,
        "compare": cmd_compare,
        "suite": cmd_suite,
        "controls": cmd_controls,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
