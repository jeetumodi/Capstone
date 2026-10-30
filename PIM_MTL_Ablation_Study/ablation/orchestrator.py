"""
Experiment Orchestration and Life-Cycle Management.

Manages:
- Multi-seed execution and variance estimation (seeds 42, 123, 2025).
- Resumption support for interrupted runs.
- Checkpoint persistence and logging.
- System hardware/software profiling.
- Incremental CSV result writing.
"""

import os
import sys
import json
import platform
import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from ablation.config import ExperimentConfig, EXPERIMENT_REGISTRY
from ablation.data import prepare_data
from ablation.trainer import train_single_run
from ablation.evaluator import evaluate_model_on_test


def get_system_environment_info() -> Dict[str, Any]:
    """Capture hardware, OS, and framework versions for experimental reproducibility."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "os": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_runtime_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "N/A (CPU execution)"
    return info


class ExperimentOrchestrator:
    """
    Coordinates execution of ablation studies across configurations and random seeds.
    """

    def __init__(
        self,
        results_dir: str = "results",
        quick_mode: bool = False,
        resume: bool = False,
    ) -> None:
        self.results_dir = os.path.abspath(results_dir)
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.figures_dir = os.path.join(self.results_dir, "figures")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.summary_csv_path = os.path.join(self.results_dir, "ablation_results.csv")
        self.detailed_csv_path = os.path.join(self.results_dir, "ablation_runs_detailed.csv")
        self.env_json_path = os.path.join(self.results_dir, "environment_info.json")

        self.quick_mode = quick_mode
        self.resume = resume
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save system environment metadata
        env_info = get_system_environment_info()
        with open(self.env_json_path, "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)

    def is_experiment_completed(self, exp_name: str) -> bool:
        """Check if an experiment has already been completed in summary CSV."""
        if not os.path.exists(self.summary_csv_path):
            return False
        try:
            df = pd.read_csv(self.summary_csv_path)
            return exp_name in df["experiment_name"].values
        except Exception:
            return False

    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Execute an experiment across all configured random seeds.
        """
        if self.resume and self.is_experiment_completed(config.name):
            print(f"[SKIP] Experiment '{config.name}' already completed. Skipping (resume mode).")
            df = pd.read_csv(self.summary_csv_path)
            return df[df["experiment_name"] == config.name].iloc[0].to_dict()

        seeds = [42] if self.quick_mode else config.train.seeds
        epochs = 5 if self.quick_mode else config.train.epochs

        # Make copy of config with adjusted epochs if quick mode
        active_config = config
        active_config.train.epochs = epochs

        print("\n" + "=" * 80)
        print(f"RUNNING EXPERIMENT: {config.name.upper()} (Category: {config.category})")
        print(f"Description: {config.description}")
        print(f"Seeds: {seeds} | Epochs: {epochs} | Device: {self.device}")
        print("=" * 80)

        # Prepare dataset once per experiment (fixed 70:30 split, train-only preprocessor fitting)
        train_data, test_data, preprocessor = prepare_data(active_config.data)
        X_test, y_test_scaled, mask_test, y_test_raw = test_data

        seed_results: List[Dict[str, float]] = []
        seed_durations: List[float] = []

        for seed in seeds:
            print(f"\n--- Starting Seed {seed} ---")
            model, history, duration_sec = train_single_run(
                config=active_config,
                train_data=train_data,
                preprocessor=preprocessor,
                seed=seed,
                device=self.device,
                verbose=True,
            )
            seed_durations.append(duration_sec)

            # Evaluate model on unseen test set
            metrics, preds_raw = evaluate_model_on_test(
                model=model,
                X_test=X_test,
                y_test_raw=y_test_raw,
                mask_test=mask_test,
                preprocessor=preprocessor,
                device=self.device,
            )
            seed_results.append(metrics)

            # Save checkpoint
            ckpt_path = os.path.join(self.checkpoints_dir, f"{config.name}_seed_{seed}.pt")
            model.save(ckpt_path, extra={"seed": seed, "metrics": metrics, "config": active_config.to_dict()})

            # Save training history and predictions
            history_path = os.path.join(self.logs_dir, f"{config.name}_seed_{seed}_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f)

            preds_df = pd.DataFrame({
                "Test_Index": preprocessor["test_indices"],
                "True_Cs_Mpa": y_test_raw[:, 0],
                "Pred_Cs_Mpa": preds_raw[:, 0],
                "True_Ts_Mpa": y_test_raw[:, 1],
                "Pred_Ts_Mpa": preds_raw[:, 1],
                "True_Fs_Mpa": y_test_raw[:, 2],
                "Pred_Fs_Mpa": preds_raw[:, 2],
            })
            preds_csv_path = os.path.join(self.logs_dir, f"{config.name}_seed_{seed}_test_preds.csv")
            preds_df.to_csv(preds_csv_path, index=False)

            # Log detailed per-seed row
            detailed_row = {
                "experiment_name": config.name,
                "category": config.category,
                "seed": seed,
                "duration_sec": duration_sec,
                **metrics,
            }
            self._append_detailed_row(detailed_row)

            print(
                f"[Seed {seed} Result] R^2(Cs/Ts/Fs): {metrics['Cs_r2']:.4f} / {metrics['Ts_r2']:.4f} / {metrics['Fs_r2']:.4f} | "
                f"Mean R^2: {metrics['mean_r2']:.4f} | Impossible(Ts>=Cs): {metrics['pvr_impossible_rate']:.2f}% | "
                f"Inversion(Ts>=Fs): {metrics['pvr_hierarchy_inversion_rate']:.2f}% | Overall PCR: {metrics['overall_pcr']:.2f}%"
            )

        # Aggregate across seeds (Mean and Std)
        agg_summary = self._aggregate_seed_results(config, seed_results, seed_durations, seeds)
        self._append_summary_row(agg_summary)

        print("\n" + "-" * 80)
        print(f"COMPLETED EXPERIMENT: {config.name}")
        print(
            f"Mean R^2: {agg_summary['mean_r2_mean']:.4f} +/- {agg_summary['mean_r2_std']:.4f} | "
            f"Overall PCR: {agg_summary['overall_pcr_mean']:.2f}% +/- {agg_summary['overall_pcr_std']:.2f}%"
        )
        print("-" * 80)
        return agg_summary

    def _aggregate_seed_results(
        self,
        config: ExperimentConfig,
        seed_results: List[Dict[str, float]],
        seed_durations: List[float],
        seeds: List[int],
    ) -> Dict[str, Any]:
        """Compute statistical aggregation across seeds."""
        res_df = pd.DataFrame(seed_results)

        summary: Dict[str, Any] = {
            "experiment_name": config.name,
            "category": config.category,
            "description": config.description,
            "seeds_tested": str(seeds),
            "num_seeds": len(seeds),
            "epochs": config.train.epochs,
            "input_dim": 19 if config.data.include_ndt else 17,
            "activation": config.model.activation,
            "normalization": config.model.normalization,
            "lambda_phys": config.loss.lambda_phys,
            "hinge_type": config.loss.hinge_type,
            "task_weights": str(config.loss.task_weights),
            "missing_strategy": config.data.missing_strategy,
            "duration_sec_mean": float(np.mean(seed_durations)),
            "duration_sec_total": float(np.sum(seed_durations)),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Calculate mean and std for all numeric metrics
        for col in res_df.columns:
            if np.issubdtype(res_df[col].dtype, np.number):
                summary[f"{col}_mean"] = float(res_df[col].mean())
                summary[f"{col}_std"] = float(res_df[col].std(ddof=1)) if len(seeds) > 1 else 0.0

        return summary

    def _append_detailed_row(self, row: Dict[str, Any]) -> None:
        """Append a single run record to ablation_runs_detailed.csv."""
        df_new = pd.DataFrame([row])
        if os.path.exists(self.detailed_csv_path):
            df_new.to_csv(self.detailed_csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(self.detailed_csv_path, mode="w", header=True, index=False)

    def _append_summary_row(self, row: Dict[str, Any]) -> None:
        """Append or update a summary record in ablation_results.csv."""
        df_new = pd.DataFrame([row])
        if os.path.exists(self.summary_csv_path):
            try:
                existing_df = pd.read_csv(self.summary_csv_path)
                # If experiment already exists, replace row
                if row["experiment_name"] in existing_df["experiment_name"].values:
                    existing_df = existing_df[existing_df["experiment_name"] != row["experiment_name"]]
                combined = pd.concat([existing_df, df_new], ignore_index=True)
                combined.to_csv(self.summary_csv_path, index=False)
            except Exception:
                df_new.to_csv(self.summary_csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(self.summary_csv_path, mode="w", header=True, index=False)
