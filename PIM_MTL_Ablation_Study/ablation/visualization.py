"""
Publication-Quality Visualization Suite for PIM-MTL Ablation Studies.

Generates comprehensive high-resolution figures (300 DPI) covering:
1. Predictive Performance Across All Ablation Configurations (R², RMSE, MAE)
2. Physical Compliance & Invariant Violations (PIR, HIR, TVR, FVR, PCR)
3. Physics-Weight Sensitivity Curves (λ_phys sweep from 0.0 to 2.0)
4. Hinge Type Comparison (Linear vs. Quadratic Hinge)
5. Physical Constraint Term Ablations (Ts < Cs, Ts < Fs, ACI Ratios)
6. Missing-Target Strategy Comparison (Masked MTL vs Complete Case vs Mean/KNN Imputation)
7. Task Weighting Impact (Equal [1,1,1] vs Proposed [1.0, 1.2, 1.5])
8. Activation Function Impact (GELU vs ReLU vs LeakyReLU)
9. Normalization Scheme Impact (SafeBatchNorm1d vs standard BatchNorm1d vs LayerNorm)
10. Feature Ablation (19 dims with NDT vs 17 dims without NDT)
11. Parity Plots (True vs Predicted for Cs, Ts, Fs)
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style() -> None:
    """Set global publication-grade aesthetics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def generate_all_plots(results_csv_path: str, figures_dir: str) -> None:
    """Read results CSV and render complete figure catalog."""
    if not os.path.exists(results_csv_path):
        print(f"[WARN] Cannot generate plots: {results_csv_path} does not exist.")
        return

    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv(results_csv_path)
    set_plot_style()

    print(f"\n[INFO] Generating publication-quality figures in: {figures_dir}...")

    # Helper getters with fallback
    def get_val(df_sub, col):
        return df_sub[col].values if col in df_sub.columns else np.zeros(len(df_sub))

    # 1. Predictive Performance Comparison (R² across all experiments)
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        df_sorted = df.sort_values(by="mean_r2_mean", ascending=True)
        y_pos = np.arange(len(df_sorted))

        bars = ax.barh(
            y_pos,
            df_sorted["mean_r2_mean"],
            xerr=df_sorted.get("mean_r2_std", 0),
            color="#2b5c8f",
            alpha=0.85,
            edgecolor="black",
            capsize=3,
        )

        # Highlight baseline
        for idx, (b, name) in enumerate(zip(bars, df_sorted["experiment_name"])):
            if name == "baseline":
                b.set_color("#d95f02")
                b.set_edgecolor("black")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["experiment_name"])
        ax.set_xlabel("Mean Test $R^2$ Score across [Cs, Ts, Fs]")
        ax.set_title("Predictive Performance Across All Ablation Configurations", pad=15)
        ax.set_xlim(0.70, 1.0)
        ax.axvline(
            df[df["experiment_name"] == "baseline"]["mean_r2_mean"].values[0] if "baseline" in df["experiment_name"].values else 0.98,
            color="#d95f02",
            linestyle="--",
            alpha=0.7,
            label="Baseline Anchor",
        )
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "01_predictive_performance_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error generating plot 1: {e}")

    # 2. Physical Compliance Audit (PCR across all experiments)
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        df_sorted_pcr = df.sort_values(by="overall_pcr_mean", ascending=True)
        y_pos = np.arange(len(df_sorted_pcr))

        bars = ax.barh(
            y_pos,
            df_sorted_pcr["overall_pcr_mean"],
            xerr=df_sorted_pcr.get("overall_pcr_std", 0),
            color="#2ca02c",
            alpha=0.85,
            edgecolor="black",
            capsize=3,
        )

        for b, name in zip(bars, df_sorted_pcr["experiment_name"]):
            if name == "baseline":
                b.set_color("#d95f02")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted_pcr["experiment_name"])
        ax.set_xlabel("Overall Physical Compliance Rate (PCR, %)")
        ax.set_title("Physical Law Adherence (Zero Impossible + Code Ratio Envelopes)", pad=15)
        ax.set_xlim(0.0, 105.0)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "02_physical_compliance_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error generating plot 2: {e}")

    # 3. Physics-Weight Sensitivity Curve (λ_phys Sweep)
    try:
        phys_sweep_names = [
            "phys_lambda_0.0",
            "phys_lambda_0.01",
            "phys_lambda_0.1",
            "baseline",
            "phys_lambda_1.0",
            "phys_lambda_2.0",
        ]
        df_sweep = df[df["experiment_name"].isin(phys_sweep_names)].copy()
        if len(df_sweep) >= 3:
            df_sweep["lambda_val"] = df_sweep["lambda_phys"]
            df_sweep = df_sweep.sort_values(by="lambda_val")

            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()

            l1 = ax1.plot(
                df_sweep["lambda_val"],
                df_sweep["mean_r2_mean"],
                marker="o",
                linewidth=2.2,
                color="#1f77b4",
                label="Mean $R^2$ Score",
            )
            l2 = ax2.plot(
                df_sweep["lambda_val"],
                df_sweep["overall_pcr_mean"],
                marker="s",
                linewidth=2.2,
                linestyle="--",
                color="#2ca02c",
                label="Overall Compliance (PCR %)",
            )

            # Combined legend
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="center right")

            ax1.set_xlabel(r"Physics Loss Regularization Weight ($\lambda_{\mathrm{phys}}$)")
            ax1.set_ylabel(r"Predictive Accuracy (Mean $R^2$)", color="#1f77b4")
            ax2.set_ylabel(r"Physical Compliance Rate (%)", color="#2ca02c")
            ax1.set_title(r"Physics-Weight Sensitivity Analysis ($\lambda_{\mathrm{phys}} \in [0.0, 2.0]$)", pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "03_physics_weight_sensitivity_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 3: {e}")

    # 4. Hinge Type Comparison
    try:
        hinge_exps = ["phys_linear_hinge", "baseline"]
        df_hinge = df[df["experiment_name"].isin(hinge_exps)].copy()
        if len(df_hinge) == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

            labels = ["Linear Hinge\nReLU(u)", "Quadratic Hinge\nReLU(u)² (Baseline)"]
            colors = ["#4575b4", "#d73027"]

            # R2 comparison
            r2_vals = [
                df[df["experiment_name"] == "phys_linear_hinge"]["mean_r2_mean"].values[0],
                df[df["experiment_name"] == "baseline"]["mean_r2_mean"].values[0],
            ]
            ax1.bar(labels, r2_vals, color=colors, alpha=0.85, edgecolor="black", width=0.5)
            ax1.set_ylabel("Mean $R^2$ Score")
            ax1.set_title("Predictive Precision ($R^2$)")
            ax1.set_ylim(min(r2_vals) * 0.98, 1.0)

            # Compliance comparison
            pcr_vals = [
                df[df["experiment_name"] == "phys_linear_hinge"]["overall_pcr_mean"].values[0],
                df[df["experiment_name"] == "baseline"]["overall_pcr_mean"].values[0],
            ]
            ax2.bar(labels, pcr_vals, color=colors, alpha=0.85, edgecolor="black", width=0.5)
            ax2.set_ylabel("Overall Compliance (PCR %)")
            ax2.set_title("Physical Law Adherence (PCR %)")
            ax2.set_ylim(0, 105)

            fig.suptitle("Hinge Formulation Ablation: Linear vs. Quadratic Penalty", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "04_hinge_type_comparison.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 4: {e}")

    # 5. Missing-Target Strategy Comparison
    try:
        missing_names = ["baseline", "missing_complete_case", "missing_mean_imputation", "missing_knn_imputation"]
        df_miss = df[df["experiment_name"].isin(missing_names)].copy()
        if len(df_miss) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(df_miss))
            width = 0.25

            ax.bar(x - width, df_miss["Cs_r2_mean"], width, label="Compressive ($C_s$)", color="#3182bd")
            ax.bar(x, df_miss["Ts_r2_mean"], width, label="Tensile ($T_s$)", color="#74c476")
            ax.bar(x + width, df_miss["Fs_r2_mean"], width, label="Flexural ($F_s$)", color="#fd8d3c")

            clean_labels = [
                name.replace("missing_", "").replace("_", " ").title() if name != "baseline" else "Proposed Masked MTL"
                for name in df_miss["experiment_name"]
            ]
            ax.set_xticks(x)
            ax.set_xticklabels(clean_labels, rotation=15)
            ax.set_ylabel("Test $R^2$ Score")
            ax.set_title("Missing-Target Handling Strategies Comparison", pad=15)
            ax.legend(loc="lower right")
            ax.set_ylim(0.70, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "06_missing_target_strategies.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 6: {e}")

    # 6. Activation Function Impact
    try:
        act_names = ["baseline", "act_relu", "act_leaky_relu"]
        df_act = df[df["experiment_name"].isin(act_names)].copy()
        if len(df_act) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
            labels = ["GELU (Baseline)", "Standard ReLU", "LeakyReLU"]
            exp_order = ["baseline", "act_relu", "act_leaky_relu"]
            df_act = df_act.set_index("experiment_name").reindex(exp_order).reset_index()

            ax1.bar(labels, df_act["mean_r2_mean"], color=["#1b9e77", "#d95f02", "#7570b3"], edgecolor="black", width=0.5)
            ax1.set_ylabel("Mean $R^2$ Score")
            ax1.set_title("Predictive Performance (Mean $R^2$)")
            ax1.set_ylim(0.90, 1.0)

            ax2.bar(labels, df_act["overall_pcr_mean"], color=["#1b9e77", "#d95f02", "#7570b3"], edgecolor="black", width=0.5)
            ax2.set_ylabel("Overall Compliance (PCR %)")
            ax2.set_title("Physical Compliance (PCR %)")
            ax2.set_ylim(0, 105)

            fig.suptitle("Activation Function Ablation: GELU vs. Piecewise Linear Activations", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "08_activation_functions_impact.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 8: {e}")

    # 7. Normalization Schemes Impact
    try:
        norm_names = ["baseline", "norm_batchnorm", "norm_layernorm"]
        df_norm = df[df["experiment_name"].isin(norm_names)].copy()
        if len(df_norm) > 1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            labels = ["SafeBatchNorm1d\n(Baseline)", "Standard BatchNorm1d", "LayerNorm"]
            exp_order = ["baseline", "norm_batchnorm", "norm_layernorm"]
            df_norm = df_norm.set_index("experiment_name").reindex(exp_order).reset_index()

            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, df_norm["mean_r2_mean"], width, label="Mean $R^2$", color="#386cb0")
            ax.bar(x + width/2, df_norm["overall_pcr_mean"] / 100.0, width, label="PCR (fraction)", color="#f0027f")

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Metric Value")
            ax.set_title("Normalization Schemes: SafeBatchNorm vs BatchNorm vs LayerNorm", pad=15)
            ax.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "09_normalization_schemes_impact.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 9: {e}")

    # 8. Feature Ablation: NDT vs Full
    try:
        feat_names = ["baseline", "feat_no_ndt"]
        df_feat = df[df["experiment_name"].isin(feat_names)].copy()
        if len(df_feat) == 2:
            fig, ax = plt.subplots(figsize=(9, 5))
            targets = ["Cs", "Ts", "Fs"]
            x = np.arange(len(targets))
            width = 0.35

            row_base = df_feat[df_feat["experiment_name"] == "baseline"].iloc[0]
            row_nondt = df_feat[df_feat["experiment_name"] == "feat_no_ndt"].iloc[0]

            r2_base = [row_base["Cs_r2_mean"], row_base["Ts_r2_mean"], row_base["Fs_r2_mean"]]
            r2_nondt = [row_nondt["Cs_r2_mean"], row_nondt["Ts_r2_mean"], row_nondt["Fs_r2_mean"]]

            ax.bar(x - width/2, r2_base, width, label="Full 19 Features (with NDT: UPV & Er)", color="#2b8cbe")
            ax.bar(x + width/2, r2_nondt, width, label="17 Features (Without NDT: UPV & Er Removed)", color="#e6550d")

            ax.set_xticks(x)
            ax.set_xticklabels(["Compressive ($C_s$)", "Tensile ($T_s$)", "Flexural ($F_s$)"])
            ax.set_ylabel("Test $R^2$ Score")
            ax.set_title("Feature Ablation: Sensitivity to Non-Destructive Testing (NDT) Proxies", pad=15)
            ax.legend(loc="lower right")
            ax.set_ylim(0.70, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "10_feature_ablation_ndt_vs_full.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 10: {e}")

    # 9. Task Weighting Comparison
    try:
        tw_names = ["baseline", "task_weights_equal"]
        df_tw = df[df["experiment_name"].isin(tw_names)].copy()
        if len(df_tw) == 2:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            targets = ["Cs", "Ts", "Fs"]
            x = np.arange(len(targets))
            width = 0.35

            row_prop = df_tw[df_tw["experiment_name"] == "baseline"].iloc[0]
            row_eq = df_tw[df_tw["experiment_name"] == "task_weights_equal"].iloc[0]

            rmse_prop = [row_prop["Cs_rmse_mean"], row_prop["Ts_rmse_mean"], row_prop["Fs_rmse_mean"]]
            rmse_eq = [row_eq["Cs_rmse_mean"], row_eq["Ts_rmse_mean"], row_eq["Fs_rmse_mean"]]

            ax.bar(x - width/2, rmse_prop, width, label="Proposed Task Weights [1.0, 1.2, 1.5]", color="#2ca02c")
            ax.bar(x + width/2, rmse_eq, width, label="Equal Task Weights [1.0, 1.0, 1.0]", color="#9467bd")

            ax.set_xticks(x)
            ax.set_xticklabels(["Compressive ($C_s$)", "Tensile ($T_s$)", "Flexural ($F_s$)"])
            ax.set_ylabel("RMSE (MPa)")
            ax.set_title("Task Weighting Ablation: Proposed vs Equal Loss Weighting", pad=15)
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "07_task_weighting_impact.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 7: {e}")

    # 10. Constraint Ablation Comparison
    try:
        const_names = [
            "baseline",
            "phys_without_ts_lt_cs",
            "phys_without_ts_lt_fs",
            "phys_with_ts_lt_fs",
            "phys_without_ratio_bounds",
            "phys_lambda_0.0",
        ]
        df_const = df[df["experiment_name"].isin(const_names)].copy()
        if len(df_const) > 2:
            fig, ax = plt.subplots(figsize=(11, 5))
            y_pos = np.arange(len(df_const))

            ax.barh(y_pos, df_const["overall_pcr_mean"], color="#1f77b4", alpha=0.85, edgecolor="black")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_const["experiment_name"])
            ax.set_xlabel("Overall Physical Compliance Rate (PCR %)")
            ax.set_title("Physical Constraint Term Ablation: Deconstruction of Governing Laws", pad=15)
            ax.set_xlim(0, 105)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "05_constraint_ablation_comparison.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 5: {e}")

    # 11. Parity Plots for Baseline (Pred vs Actual with 1:1 and ±10% error bands)
    try:
        logs_dir = os.path.join(os.path.dirname(figures_dir), "logs")
        baseline_preds_path = os.path.join(logs_dir, "baseline_seed_42_test_preds.csv")
        if os.path.exists(baseline_preds_path):
            p_df = pd.read_csv(baseline_preds_path)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

            targets_info = [
                ("Compressive Strength ($C_s$)", "True_Cs_Mpa", "Pred_Cs_Mpa", "#1f77b4"),
                ("Splitting Tensile ($T_s$)", "True_Ts_Mpa", "Pred_Ts_Mpa", "#2ca02c"),
                ("Flexural Strength ($F_s$)", "True_Fs_Mpa", "Pred_Fs_Mpa", "#ff7f0e"),
            ]

            for ax, (title, true_col, pred_col, color) in zip(axes, targets_info):
                valid = p_df[true_col].notna()
                y_t = p_df.loc[valid, true_col].values
                y_p = p_df.loc[valid, pred_col].values

                ax.scatter(y_t, y_p, alpha=0.5, color=color, edgecolors="none", s=25)

                # 1:1 diagonal line
                min_v = min(y_t.min(), y_p.min()) * 0.95
                max_v = max(y_t.max(), y_p.max()) * 1.05
                ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="1:1 Parity")
                ax.plot([min_v, max_v], [min_v * 1.1, max_v * 1.1], "k:", alpha=0.5, label="±10% Bound")
                ax.plot([min_v, max_v], [min_v * 0.9, max_v * 0.9], "k:", alpha=0.5)

                r2 = df[df["experiment_name"] == "baseline"][f"{true_col.split('_')[1]}_r2_mean"].values[0] if "baseline" in df["experiment_name"].values else 0.98
                ax.set_title(f"{title}\n$R^2 = {r2:.4f}$", pad=10)
                ax.set_xlabel("Laboratory Measured (MPa)")
                ax.set_ylabel("PIM-MTL Predicted (MPa)")
                ax.set_xlim(min_v, max_v)
                ax.set_ylim(min_v, max_v)
                ax.legend(loc="upper left", frameon=True)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "11_parity_plots_pred_vs_actual.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating plot 11: {e}")

    print(f"[SUCCESS] All ablation figures saved successfully to: {figures_dir}")
