# Concrete 3DCP: Model Benchmark & ACI 318-19 Constraint Auditing Suite

This module provides a unified benchmark comparing **all baseline models** (from [`4thsem`](../4thsem)) against the **Physics-Informed Masked Multi-Task Learning (PIM-MTL)** network (from [`3rdYearsem5`](../3rdYearsem5)) on an identical, standardized **30% held-out test split (1,326 specimens)**.

---

## 1. Executive Findings: Accuracy vs. Physical Feasibility Trade-Off

| Model Architecture | Compressive $C_s$ ($R^2$ / RMSE) | Tensile $T_s$ ($R^2$ / RMSE) | Flexural $F_s$ ($R^2$ / RMSE) | ACI Tensile Violations ($R_t \notin [0.06, 0.18]$) | ACI Flexural Violations ($R_f \notin [0.08, 0.25]$) | Hierarchy Violations ($T_s \ge F_s$) | Overall Physical Compliance Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 0.9027 / 3.04 MPa | 0.9178 / 0.26 MPa | 0.9866 / 0.28 MPa | 0.15% | 26.17% | 26.17% | 73.68% |
| **Naive Bayes Regressor** | 0.7819 / 4.55 MPa | 0.5361 / 0.62 MPa | 0.9719 / 0.40 MPa | 4.90% | 16.21% | 18.55% | 72.47% |
| **XGBoost Regressor** | **0.9954 / 0.66 MPa** | **0.9983 / 0.04 MPa** | **0.9998 / 0.03 MPa** | 12.59% | 39.82% | 40.95% | **45.55%** (54.45% Violations) |
| **LightGBM Regressor** | **0.9955 / 0.65 MPa** | **0.9982 / 0.04 MPa** | **0.9998 / 0.04 MPa** | 12.37% | 39.59% | 38.91% | **48.04%** (51.96% Violations) |
| **Single-Task ANN (PyTorch)** | 0.9921 / 0.87 MPa | 0.9949 / 0.06 MPa | 0.9992 / 0.07 MPa | 12.37% | 32.96% | 36.43% | **54.60%** (45.40% Violations) |
| **PIM-MTL (Physics-Informed)** | 0.9807 / 1.35 MPa | 0.9837 / 0.12 MPa | 0.9955 / 0.16 MPa | **3.24%** (75% reduction) | **17.27%** (56% reduction) | **18.10%** (56% reduction) | **78.66% (Highest Compliance)** |

---

## 2. Directory Structure & Generated Figures

```text
c:\Users\jeetu\projects\Capstone\model_comparsion\
├── config.py                 # Core paths, ACI 318 bounds, and color palettes
├── evaluate_constraints.py   # Statistical metric engine & ACI physical constraint auditor
├── run_comparison.py         # Main execution pipeline (evaluates all models on 30% test set)
├── plot_comparisons.py       # High-resolution publication-quality figure generator
├── README.md                 # Executive documentation and reproduction guide
├── results/
│   ├── accuracy_metrics.csv       # MSE, RMSE, R2, Adj_R2, MAE per target for all models
│   ├── constraint_violations.csv  # Violation counts, rates (%), mean & max violation deltas
│   ├── comprehensive_summary.csv  # Combined matrix of accuracy and physical feasibility
│   ├── full_test_predictions.csv  # Specimen-level predictions [Cs, Ts, Fs] and ratios
│   └── comparison_report.md       # Formatted Markdown report
└── plots/
    ├── cs_performance_comparison.png      # [Plot 1] CS specific: 3 subplots (MSE, RMSE, R2) with zero overlap
    ├── ts_performance_comparison.png      # [Plot 2] TS specific: 3 subplots (MSE, RMSE, R2) with zero overlap
    ├── fs_performance_comparison.png      # [Plot 3] FS specific: 3 subplots (MSE, RMSE, R2) with zero overlap
    ├── physical_violation_rates.png       # ACI & physical violation rate comparisons across all models
    ├── aci_constraint_envelopes.png       # Scatter distributions vs shaded ACI admissible zones
    └── actual_vs_predicted_parity.png     # Parity plots (Experimental vs Predicted)
```

---

## 3. How to Reproduce and Re-Run

From the `model_comparsion` directory, run:

```bash
# Execute complete benchmark and regenerate all tables, CSVs, and plots
python run_comparison.py
```
