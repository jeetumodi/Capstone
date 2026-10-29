"""
Visualization Suite for Model Comparison & ACI Physical Feasibility Auditing.
Generates publication-quality, perfectly spaced figures:
1. cs_performance_comparison.png: 3 subplots (MSE, RMSE, R2 for Cs)
2. ts_performance_comparison.png: 3 subplots (MSE, RMSE, R2 for Ts)
3. fs_performance_comparison.png: 3 subplots (MSE, RMSE, R2 for Fs)
4. physical_violation_rates.png: ACI & physical violation rate comparisons
5. aci_constraint_envelopes.png: Scatter distributions vs shaded ACI admissible zones
6. actual_vs_predicted_parity.png: Parity plots (Experimental vs Predicted)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR, MODEL_COLORS, ACI_BOUNDS

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300

# Consistent, clean 2-line wrapped model labels to ensure zero overlap
CLEAN_MODEL_LABELS = {
    "Ridge Regression": "Ridge\nRegression",
    "Naive Bayes Regressor": "Naive Bayes\nRegressor",
    "XGBoost Regressor": "XGBoost\nRegressor",
    "LightGBM Regressor": "LightGBM\nRegressor",
    "Single-Task ANN (PyTorch)": "Single-Task\nANN",
    "PIM-MTL": "PIM-MTL",
}

MODEL_ORDER = list(CLEAN_MODEL_LABELS.keys())


def plot_single_target_metrics(
    accuracy_df: pd.DataFrame,
    target_key: str,
    target_title: str,
    unit_label: str,
    output_filename: str,
    output_dir: Path = PLOTS_DIR,
):
    """
    Generates a dedicated figure for a single target (Cs, Ts, or Fs)
    containing exactly 3 subplots:
    - Subplot 1: MSE comparison for each model
    - Subplot 2: RMSE comparison for each model
    - Subplot 3: R2 comparison for each model
    Engineered with generous spacing, distinct colors, and zero label overlap.
    """
    sub = accuracy_df[accuracy_df["Target"] == target_key].copy()
    sub["Model"] = pd.Categorical(sub["Model"], categories=MODEL_ORDER, ordered=True)
    sub = sub.sort_values("Model").reset_index(drop=True)

    labels = [CLEAN_MODEL_LABELS[m] for m in sub["Model"]]
    bar_colors = [MODEL_COLORS.get(m, "#4a90e2") for m in sub["Model"]]
    x_pos = np.arange(len(sub))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8), constrained_layout=True)

    # -------------------------------------------------------------
    # Subplot 1: Mean Squared Error (MSE)
    # -------------------------------------------------------------
    ax_mse = axes[0]
    bars_mse = ax_mse.bar(x_pos, sub["MSE"], color=bar_colors, edgecolor="black", width=0.58, alpha=0.9)
    ax_mse.set_title(f"A. Mean Squared Error (MSE)", fontsize=12, fontweight="bold", pad=12)
    ax_mse.set_ylabel(f"MSE ({unit_label}²)", fontsize=11, fontweight="semibold")
    ax_mse.set_xticks(x_pos)
    ax_mse.set_xticklabels(labels, fontsize=9.0)
    max_mse = sub["MSE"].max()
    ax_mse.set_ylim(0, max_mse * 1.22)
    ax_mse.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars_mse, sub["MSE"]):
        y_text = bar.get_height()
        fmt = f"{val:.4f}" if max_mse < 1.0 else f"{val:.2f}"
        ax_mse.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text + max_mse * 0.02,
            fmt,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # Subplot 2: Root Mean Squared Error (RMSE)
    # -------------------------------------------------------------
    ax_rmse = axes[1]
    bars_rmse = ax_rmse.bar(x_pos, sub["RMSE"], color=bar_colors, edgecolor="black", width=0.58, alpha=0.9)
    ax_rmse.set_title(f"B. Root Mean Squared Error (RMSE)", fontsize=12, fontweight="bold", pad=12)
    ax_rmse.set_ylabel(f"RMSE ({unit_label})", fontsize=11, fontweight="semibold")
    ax_rmse.set_xticks(x_pos)
    ax_rmse.set_xticklabels(labels, fontsize=9.0)
    max_rmse = sub["RMSE"].max()
    ax_rmse.set_ylim(0, max_rmse * 1.20)
    ax_rmse.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars_rmse, sub["RMSE"]):
        y_text = bar.get_height()
        fmt = f"{val:.4f}" if max_rmse < 0.2 else f"{val:.3f}"
        ax_rmse.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text + max_rmse * 0.02,
            fmt,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # Subplot 3: Coefficient of Determination (R2)
    # -------------------------------------------------------------
    ax_r2 = axes[2]
    bars_r2 = ax_r2.bar(x_pos, sub["R2"], color=bar_colors, edgecolor="black", width=0.58, alpha=0.9)
    ax_r2.set_title(f"C. Coefficient of Determination ($R^2$)", fontsize=12, fontweight="bold", pad=12)
    ax_r2.set_ylabel("Coefficient of Determination ($R^2$)", fontsize=11, fontweight="semibold")
    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels(labels, fontsize=9.0)
    min_r2 = max(0.0, sub["R2"].min() - 0.1)
    ax_r2.set_ylim(min_r2, 1.05)
    ax_r2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars_r2, sub["R2"]):
        y_text = bar.get_height()
        ax_r2.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text + 0.008,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.suptitle(f"{target_title} Performance Comparison: MSE | RMSE | $R^2$", fontsize=15, fontweight="bold", y=1.04)
    out_path = output_dir / output_filename
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved Plot]: {out_path}")


def plot_violation_rates(constraint_df: pd.DataFrame, output_dir: Path = PLOTS_DIR):
    """Physical & ACI 318 Violation Rates Comparison Bar Chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.2), constrained_layout=True)

    sub_df = constraint_df.copy()
    sub_df["Model"] = pd.Categorical(sub_df["Model"], categories=MODEL_ORDER, ordered=True)
    sub_df = sub_df.sort_values("Model", ascending=False).reset_index(drop=True)

    models = sub_df["Model"].tolist()
    y_pos = np.arange(len(models))

    # Panel 1: Severe Inversion Violations (Impossible Ts >= Cs & Hierarchy Ts >= Fs)
    width = 0.35
    b1 = ax1.barh(y_pos - width / 2, sub_df["Impossible_Rate_Pct"], width, label="Physically Impossible ($T_s \\geq C_s$)", color="#d9534f", edgecolor="black")
    b2 = ax1.barh(y_pos + width / 2, sub_df["Hierarchy_Viol_Rate_Pct"], width, label="Hierarchy Violation ($T_s \\geq F_s$)", color="#f0ad4e", edgecolor="black")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontweight="semibold", fontsize=9.5)
    ax1.set_xlabel("Violation Rate (%)", fontweight="bold")
    ax1.set_title("A. Physically Impossible & Hierarchy Inversion Violations", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    for b in list(b1) + list(b2):
        w = b.get_width()
        if w > 0.05:
            ax1.text(w + 0.4, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, fontweight="bold")
        else:
            ax1.text(0.2, b.get_y() + b.get_height() / 2, "0.0%", va="center", fontsize=8.5, color="green", fontweight="bold")

    # Panel 2: ACI 318 Ratio Envelope Violations (Rt & Rf)
    b3 = ax2.barh(y_pos - width / 2, sub_df["ACI_Rt_Viol_Rate_Pct"], width, label="ACI Tensile Ratio Violations ($R_t \\notin [0.06, 0.18]$)", color="#5bc0de", edgecolor="black")
    b4 = ax2.barh(y_pos + width / 2, sub_df["ACI_Rf_Viol_Rate_Pct"], width, label="ACI Flexural Ratio Violations ($R_f \\notin [0.08, 0.25]$)", color="#5cb85c", edgecolor="black")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontweight="semibold", fontsize=9.5)
    ax2.set_xlabel("Violation Rate (%)", fontweight="bold")
    ax2.set_title("B. ACI 318-19 Continuum Mechanics Envelope Violations", fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", linestyle="--", alpha=0.7)

    for b in list(b3) + list(b4):
        w = b.get_width()
        if w > 0.05:
            ax2.text(w + 0.5, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5, fontweight="bold")
        else:
            ax2.text(0.2, b.get_y() + b.get_height() / 2, "0.0%", va="center", fontsize=8.5, color="green", fontweight="bold")

    fig.suptitle("Physical & Building Code Feasibility Audit Across Candidate Architectures", fontsize=15, fontweight="bold", y=1.03)
    out_path = output_dir / "physical_violation_rates.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved Plot]: {out_path}")


def plot_aci_scatter_envelopes(test_export_df: pd.DataFrame, output_dir: Path = PLOTS_DIR):
    """Scatter distributions vs shaded ACI 318-19 envelopes comparing XGBoost, Single-Task ANN, and PIM-MTL."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), sharex=True, sharey="row", constrained_layout=True)
    models_to_plot = [
        ("xgboost", "XGBoost Regressor"),
        ("ann", "Single-Task ANN"),
        ("pim_mtl", "PIM-MTL"),
    ]
    cs_range = np.linspace(0, 60, 200)

    # Row 1: Tensile Strength (Ts) vs Compressive Strength (Cs)
    for col_idx, (m_key, m_name) in enumerate(models_to_plot):
        ax = axes[0, col_idx]
        cs_vals = test_export_df[f"Pred_Cs_{m_key}"].values
        ts_vals = test_export_df[f"Pred_Ts_{m_key}"].values

        lower_bound = ACI_BOUNDS["Rt_min"] * cs_range
        upper_bound = ACI_BOUNDS["Rt_max"] * cs_range

        ax.fill_between(cs_range, lower_bound, upper_bound, color="green", alpha=0.15, label="ACI 318 Envelope [0.06, 0.18]")
        ax.plot(cs_range, upper_bound, "g--", linewidth=1.2)
        ax.plot(cs_range, lower_bound, "g--", linewidth=1.2)

        rt = ts_vals / np.maximum(cs_vals, 1e-4)
        viol_mask = (rt < ACI_BOUNDS["Rt_min"] - 1e-5) | (rt > ACI_BOUNDS["Rt_max"] + 1e-5) | (ts_vals >= cs_vals)

        ax.scatter(cs_vals[~viol_mask], ts_vals[~viol_mask], s=12, color="#007bff", alpha=0.5, label="Compliant")
        if np.sum(viol_mask) > 0:
            ax.scatter(cs_vals[viol_mask], ts_vals[viol_mask], s=20, color="#d9534f", marker="x", alpha=0.8, label=f"Violations (N={np.sum(viol_mask)})")

        viol_pct = (np.sum(viol_mask) / len(cs_vals)) * 100.0
        ax.set_title(f"{m_name}\n(Tensile Violation Rate: {viol_pct:.2f}%)", fontweight="bold")
        ax.set_ylabel("Predicted Tensile Strength $T_s$ (MPa)")
        ax.set_ylim(-0.2, 7.0)
        ax.set_xlim(0, 60)
        ax.legend(loc="upper left", fontsize=8)

    # Row 2: Flexural Strength (Fs) vs Compressive Strength (Cs)
    for col_idx, (m_key, m_name) in enumerate(models_to_plot):
        ax = axes[1, col_idx]
        cs_vals = test_export_df[f"Pred_Cs_{m_key}"].values
        fs_vals = test_export_df[f"Pred_Fs_{m_key}"].values

        lower_bound = ACI_BOUNDS["Rf_min"] * cs_range
        upper_bound = ACI_BOUNDS["Rf_max"] * cs_range

        ax.fill_between(cs_range, lower_bound, upper_bound, color="green", alpha=0.15, label="ACI 318 Envelope [0.08, 0.25]")
        ax.plot(cs_range, upper_bound, "g--", linewidth=1.2)
        ax.plot(cs_range, lower_bound, "g--", linewidth=1.2)

        rf = fs_vals / np.maximum(cs_vals, 1e-4)
        viol_mask = (rf < ACI_BOUNDS["Rf_min"] - 1e-5) | (rf > ACI_BOUNDS["Rf_max"] + 1e-5)

        ax.scatter(cs_vals[~viol_mask], fs_vals[~viol_mask], s=12, color="#007bff", alpha=0.5, label="Compliant")
        if np.sum(viol_mask) > 0:
            ax.scatter(cs_vals[viol_mask], fs_vals[viol_mask], s=20, color="#d9534f", marker="x", alpha=0.8, label=f"Violations (N={np.sum(viol_mask)})")

        viol_pct = (np.sum(viol_mask) / len(cs_vals)) * 100.0
        ax.set_title(f"{m_name}\n(Flexural Violation Rate: {viol_pct:.2f}%)", fontweight="bold")
        ax.set_xlabel("Predicted Compressive Strength $C_s$ (MPa)", fontweight="bold")
        ax.set_ylabel("Predicted Flexural Strength $F_s$ (MPa)")
        ax.set_ylim(-0.2, 10.0)
        ax.set_xlim(0, 60)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Specimen Distributions vs ACI 318-19 Admissible Physical Envelopes", fontsize=15, fontweight="bold", y=1.02)
    out_path = output_dir / "aci_constraint_envelopes.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved Plot]: {out_path}")


def plot_parity_actual_vs_predicted(test_export_df: pd.DataFrame, output_dir: Path = PLOTS_DIR):
    """Parity Plots (y = x) comparing Ground Truth vs Predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
    targets_info = [
        ("True_Cs_Mpa", "Pred_Cs_pim_mtl", "Pred_Cs_xgboost", "Compressive Strength (Cs, MPa)"),
        ("True_Ts_Mpa", "Pred_Ts_pim_mtl", "Pred_Ts_xgboost", "Tensile Strength (Ts, MPa)"),
        ("True_Fs_Mpa", "Pred_Fs_pim_mtl", "Pred_Fs_xgboost", "Flexural Strength (Fs, MPa)"),
    ]

    for idx, (true_col, pim_col, xgb_col, title) in enumerate(targets_info):
        ax = axes[idx]
        mask = ~test_export_df[true_col].isna()
        y_t = test_export_df.loc[mask, true_col].values
        y_pim = test_export_df.loc[mask, pim_col].values
        y_xgb = test_export_df.loc[mask, xgb_col].values

        min_val = min(np.min(y_t), np.min(y_pim), np.min(y_xgb))
        max_val = max(np.max(y_t), np.max(y_pim), np.max(y_xgb))

        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.2, label="Parity Line ($y = x$)")
        ax.scatter(y_t, y_xgb, s=14, color="#17a2b8", alpha=0.5, label="XGBoost (Unconstrained)")
        ax.scatter(y_t, y_pim, s=14, color="#28a745", alpha=0.6, label="PIM-MTL")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Experimental Ground Truth (MPa)", fontweight="bold")
        ax.set_ylabel("Predicted Strength (MPa)")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Parity Comparison (Experimental vs Predicted) on Held-Out Test Set", fontsize=15, fontweight="bold", y=1.03)
    out_path = output_dir / "actual_vs_predicted_parity.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [Saved Plot]: {out_path}")


def generate_all_plots(accuracy_df, constraint_df, summary_df, test_export_df):
    """Generate all figures as requested."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dedicated CS Plot: 3 Subplots (MSE, RMSE, R2)
    plot_single_target_metrics(
        accuracy_df=accuracy_df,
        target_key="Cs",
        target_title="Compressive Strength ($C_s$)",
        unit_label="MPa",
        output_filename="cs_performance_comparison.png",
    )

    # 2. Dedicated TS Plot: 3 Subplots (MSE, RMSE, R2)
    plot_single_target_metrics(
        accuracy_df=accuracy_df,
        target_key="Ts",
        target_title="Tensile Strength ($T_s$)",
        unit_label="MPa",
        output_filename="ts_performance_comparison.png",
    )

    # 3. Dedicated FS Plot: 3 Subplots (MSE, RMSE, R2)
    plot_single_target_metrics(
        accuracy_df=accuracy_df,
        target_key="Fs",
        target_title="Flexural Strength ($F_s$)",
        unit_label="MPa",
        output_filename="fs_performance_comparison.png",
    )

    # 4. Supplementary Physical Constraint Visualizations
    plot_violation_rates(constraint_df)
    plot_aci_scatter_envelopes(test_export_df)
    plot_parity_actual_vs_predicted(test_export_df)

    print(f"[All Visualization Figures Successfully Saved to]: {PLOTS_DIR}")
