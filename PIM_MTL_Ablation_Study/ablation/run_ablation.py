"""
Command-Line Interface for PIM-MTL Systematic Ablation Studies.

Usage examples:
    python -m ablation.run_ablation --category all
    python -m ablation.run_ablation --category physics
    python -m ablation.run_ablation --experiment baseline
    python -m ablation.run_ablation --resume
    python -m ablation.run_ablation --quick
"""

import os
import sys
import argparse
from typing import List

# Ensure utf-8 stdout/stderr for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from ablation.config import (
    EXPERIMENT_REGISTRY,
    ALL_CATEGORIES,
    ExperimentConfig,
)
from ablation.orchestrator import ExperimentOrchestrator
from ablation.visualization import generate_all_plots
from ablation.reporting import generate_ablation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PIM-MTL Systematic Ablation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all"] + ALL_CATEGORIES,
        help="Experiment category to execute.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment name to execute (e.g. 'baseline', 'phys_linear_hinge').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previously interrupted experiments (skip already completed ones).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick pipeline test mode (5 epochs, single seed).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save checkpoints, logs, plots, and reports.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training epochs across experiments.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated random seeds (e.g. '42,123,2025').",
    )
    return parser.parse_args()


def select_experiments(
    category: str,
    experiment_name: str = None,
) -> List[ExperimentConfig]:
    """Filter experiment registry based on CLI arguments."""
    if experiment_name:
        if experiment_name not in EXPERIMENT_REGISTRY:
            print(f"[ERROR] Unknown experiment '{experiment_name}'.")
            print(f"Available experiments: {list(EXPERIMENT_REGISTRY.keys())}")
            sys.exit(1)
        return [EXPERIMENT_REGISTRY[experiment_name]]

    if category == "all":
        # Always run baseline first, then remaining experiments
        experiments = [EXPERIMENT_REGISTRY["baseline"]]
        for k, v in EXPERIMENT_REGISTRY.items():
            if k != "baseline":
                experiments.append(v)
        return experiments

    # Category filter
    experiments = [v for v in EXPERIMENT_REGISTRY.values() if v.category == category]
    if len(experiments) == 0:
        print(f"[ERROR] No experiments found for category '{category}'.")
        sys.exit(1)
    return experiments


def main() -> None:
    args = parse_args()

    # Always ensure results dir exists
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    experiments = select_experiments(args.category, args.experiment)

    # Optional overrides
    if args.epochs is not None:
        for exp in experiments:
            exp.train.epochs = args.epochs

    if args.seeds is not None:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
        for exp in experiments:
            exp.train.seeds = seed_list

    print("\n" + "=" * 80)
    print("      PIM-MTL ABLATION STUDY ORCHESTRATION PIPELINE")
    print("=" * 80)
    print(f"Total experiments selected: {len(experiments)}")
    print(f"Execution Mode: {'QUICK SMOKE TEST (5 epochs, 1 seed)' if args.quick else 'FULL PRODUCTION STUDY'}")
    print(f"Resume Mode: {args.resume}")
    print(f"Results Directory: {results_dir}")
    print("=" * 80 + "\n")

    orchestrator = ExperimentOrchestrator(
        results_dir=results_dir,
        quick_mode=args.quick,
        resume=args.resume,
    )

    for i, exp in enumerate(experiments, start=1):
        print(f"\n>>> Running Experiment [{i}/{len(experiments)}]: {exp.name}")
        orchestrator.run_experiment(exp)

    # Post-run reporting and visualization
    summary_csv = os.path.join(results_dir, "ablation_results.csv")
    figures_dir = os.path.join(results_dir, "figures")
    report_md = os.path.join(results_dir, "ablation_report.md")

    if os.path.exists(summary_csv):
        print("\n" + "=" * 80)
        print("GENERATING ABLATION ARTIFACTS (PLOTS & RESEARCH REPORT)")
        print("=" * 80)
        generate_all_plots(summary_csv, figures_dir)
        generate_ablation_report(summary_csv, report_md)
        print(f"\n[DONE] All ablation artifacts successfully compiled in: {results_dir}")
    else:
        print(f"[WARN] No results found at {summary_csv}; skipping report and figure compilation.")


if __name__ == "__main__":
    main()
