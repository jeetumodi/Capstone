# Physics-Informed Masked Multi-Task Learning (PIM-MTL) Ablation Study

Comprehensive, modular, and reproducible ablation study suite for the research project:
**"Physics-Informed Masked Multi-Task Learning (PIM-MTL) for 3D Concrete Printing Mechanical Strength Prediction"**.

This repository conducts a systematic empirical deconstruction of inductive biases across six foundational categories:
1. **Physics Loss & Constraint Formulations** ($\lambda_{\mathrm{phys}}$ sweeps, quadratic vs linear hinge, $T_s < C_s$, $T_s < F_s$, ACI/Eurocode ratio envelopes)
2. **Missing-Target Handling Strategies** (Masked Multi-Task Learning vs Complete-Case Deletion vs Mean Imputation vs KNN Imputation)
3. **Task Weighting Dynamics** (Proposed $[1.0, 1.2, 1.5]$ vs Equal $[1.0, 1.0, 1.0]$)
4. **Activation Functions** (GELU vs ReLU vs LeakyReLU)
5. **Normalization Schemes** (SafeBatchNorm1d vs Standard BatchNorm1d vs LayerNorm)
6. **Feature Sensitivity** (Full 19-dimensional input vs 17-dimensional input without NDT features $UPV$ and $Er$)

---

## 1. Directory Structure

```text
PIM_MTL_Ablation_Study/
├── data/
│   └── Data.csv                     # 4,420 concrete mix specimens (read-only benchmark copy)
├── ablation/
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Strongly-typed dataclasses & 20 ablation experiment registry
│   ├── data.py                      # 70:30 split, train-only RobustScaler/OHE, missing-target strategies
│   ├── models.py                    # Modular PIM_MTL_Net (dynamic in_dim, GELU/ReLU/LeakyReLU, SafeBN/BN/LN)
│   ├── losses.py                    # Modular PIM_MTL_Loss (quadratic/linear hinge, ratio bounds, hierarchy)
│   ├── trainer.py                   # Adam + CosineAnnealingLR, gradient clipping, history logging
│   ├── evaluator.py                 # Predictive (R², RMSE, MAE) & Physical Compliance (PIR, HIR, TVR, FVR, PCR)
│   ├── orchestrator.py              # Multi-seed management (42, 123, 2025), resumption, logging
│   ├── visualization.py             # Publication-quality figure generator (300 DPI)
│   ├── reporting.py                 # Automated research report compiler (markdown with delta tables)
│   └── run_ablation.py              # CLI entry point
├── results/
│   ├── ablation_results.csv         # Master results table (mean ± std across seeds)
│   ├── ablation_runs_detailed.csv   # Per-seed granular execution records
│   ├── ablation_report.md           # Comprehensive research report
│   ├── environment_info.json        # Hardware, PyTorch, CUDA, and OS execution metadata
│   ├── figures/                     # High-resolution publication plots (PNG)
│   ├── checkpoints/                 # PyTorch model state weights per experiment and seed
│   └── logs/                        # Epoch loss histories and test predictions CSVs
├── notebooks/
│   └── analysis_notebook.ipynb      # Interactive Jupyter notebook for results exploration
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation and execution guide
```

---

## 2. Experimental Registry & Ablation Matrix

| Category | Experiment Name | Key Specification |
|---|---|---|
| **Physics** | `baseline` | Full PIM-MTL ($\lambda=0.5$, quadratic hinge, SafeBN, GELU, task weights $[1.0, 1.2, 1.5]$, masked MTL, 19 features) |
| **Physics** | `phys_lambda_0.0` | No physics regularizer ($\lambda=0.0$) |
| **Physics** | `phys_linear_hinge` | First-order linear hinge penalty $\mathrm{ReLU}(u)$ instead of quadratic hinge $\mathrm{ReLU}(u)^2$ |
| **Physics** | `phys_lambda_0.01` | Sensitivity sweep $\lambda_{\mathrm{phys}} = 0.01$ |
| **Physics** | `phys_lambda_0.1` | Sensitivity sweep $\lambda_{\mathrm{phys}} = 0.1$ |
| **Physics** | `phys_lambda_1.0` | Sensitivity sweep $\lambda_{\mathrm{phys}} = 1.0$ |
| **Physics** | `phys_lambda_2.0` | Sensitivity sweep $\lambda_{\mathrm{phys}} = 2.0$ |
| **Physics** | `phys_without_ts_lt_cs` | Relaxed tensile upper bound ($R_t$ unconstrained), allowing $T_s \ge C_s$ |
| **Physics** | `phys_without_ts_lt_fs` | Standard unconstrained flexural-tensile hierarchy |
| **Physics** | `phys_with_ts_lt_fs` | Explicit quadratic penalty enforcing hierarchy $T_s < F_s$ |
| **Physics** | `phys_without_ratio_bounds` | Omits ACI/Eurocode envelopes; enforces ordering invariants only ($T_s < F_s < C_s$) |
| **Missing Data** | `missing_complete_case` | Listwise deletion of incomplete samples (drops 60.2% of dataset) |
| **Missing Data** | `missing_mean_imputation` | Target column mean imputation (fit strictly on training observed labels) |
| **Missing Data** | `missing_knn_imputation` | KNN target imputation ($k=5$, fit strictly on training features and labels) |
| **Task Weights** | `task_weights_equal` | Equal task loss weights $[1.0, 1.0, 1.0]$ |
| **Activation** | `act_relu` | Piecewise-linear ReLU activation |
| **Activation** | `act_leaky_relu` | Piecewise-linear LeakyReLU ($\alpha = 0.01$) |
| **Normalization**| `norm_batchnorm` | Standard PyTorch `nn.BatchNorm1d` |
| **Normalization**| `norm_layernorm` | PyTorch `nn.LayerNorm` |
| **Features** | `feat_no_ndt` | Drop NDT features (`UPV_m/s`, `Er_ohm-cm`) before scaling (17-dim input) |

---

## 3. Installation & Setup

Ensure Python 3.10+ (tested on Python 3.14.3 with PyTorch CUDA) is available:

```bash
cd C:\Users\jeetu\projects\Capstone\PIM_MTL_Ablation_Study
pip install -r requirements.txt
```

---

## 4. Execution Commands

### Quick Pipeline Verification (Smoke Test)
Test the end-to-end pipeline in 5 epochs on a single seed:
```bash
python -m ablation.run_ablation --experiment baseline --quick
```

### Run All 20 Ablation Experiments (Full Multi-Seed Protocol)
Runs 60 epochs per run across seeds `[42, 123, 2025]`:
```bash
python -m ablation.run_ablation --category all
```

### Run Specific Category
```bash
# Physics loss & regularizer ablations
python -m ablation.run_ablation --category physics

# Missing-target handling strategies
python -m ablation.run_ablation --category missing_data

# Task loss weighting
python -m ablation.run_ablation --category task_weighting

# Activation functions
python -m ablation.run_ablation --category activation

# Normalization schemes
python -m ablation.run_ablation --category normalization

# Feature ablation (NDT vs Non-NDT)
python -m ablation.run_ablation --category feature_ablation
```

### Run a Single Experiment
```bash
python -m ablation.run_ablation --experiment baseline
python -m ablation.run_ablation --experiment phys_linear_hinge
```

### Resume Interrupted Study
Automatically skips already completed experiments recorded in `results/ablation_results.csv`:
```bash
python -m ablation.run_ablation --category all --resume
```

---

## 5. Evaluation Metrics

### Predictive Metrics (Observed Targets Only)
- **Coefficient of Determination ($R^2$):** Evaluated separately for $C_s, T_s, F_s$ and as Mean $R^2$.
- **Root Mean Squared Error (RMSE, MPa):** $\sqrt{\frac{1}{N_{\mathrm{obs}}} \sum (y - \hat{y})^2}$.
- **Mean Absolute Error (MAE, MPa):** $\frac{1}{N_{\mathrm{obs}}} \sum |y - \hat{y}|$.

### Physical Feasibility & Invariant Compliance Metrics (All 1,326 Test Specimens)
- **Physically Impossible Rate (PIR, %):** Percentage of test predictions where splitting tensile capacity exceeds compressive capacity ($\hat{T}_s \ge \hat{C}_s$).
- **Hierarchy Inversion Rate (HIR, %):** Percentage of test predictions where splitting tensile capacity exceeds flexural capacity ($\hat{T}_s \ge \hat{F}_s$).
- **Tensile Violation Rate (TVR, %):** Percentage outside ACI 318 envelope $R_t \in [0.06, 0.18]$.
- **Flexural Violation Rate (FVR, %):** Percentage outside ACI 318 envelope $R_f \in [0.08, 0.25]$.
- **Overall Physical Compliance Rate (PCR, %):** Percentage of specimens satisfying ALL 4 physical criteria simultaneously.

---

## 6. Scientific Rigor Safeguards

1. **Immutable Source:** Original project `3rdYearsem5` is completely untouched.
2. **Strict Isolation:** Preprocessing transformers (`OneHotEncoder`, `RobustScaler`) and target imputers are fitted strictly on the 70% training split.
3. **Genuine Test Evaluation:** Imputation models are evaluated strictly against real, un-imputed ground truth test targets.
4. **Safe Numerical Handling:** Compressive denominator is clamped ($\hat{C}_s \ge 1.0\ \mathrm{MPa}$) preventing divide-by-zero singularities.

