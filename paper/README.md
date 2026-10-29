# Intelligent 3DCP: Mechanical Strength Prediction Using Physics-Informed Masked Multi-Task Learning

**IEEE Conference Research Paper**  
**Authors:** Jeetu Kumar, Sagar Hiremath, Dr. Sowmya Kamath S  
**Affiliation:** Department of Information Technology, National Institute of Technology Karnataka (NITK), Surathkal, Mangalore, India

---

## 1. Directory Structure

```text
paper/
├── main.tex              # Complete IEEE conference format LaTeX paper source
├── references.bib        # Curated, validated BibTeX bibliography
├── IEEEtran.cls          # Official IEEE conference LaTeX document class
├── IEEEtran.bst          # Official IEEE BibTeX bibliography style
├── README.md             # Compilation and submission guide
└── images/               # High-resolution figures referenced in the paper
    ├── 01_target_missingness_profile.png
    ├── 02_continuous_features_distribution.png
    ├── 03_categorical_features_distribution.png
    ├── 04_target_distributions.png
    ├── 05_ground_truth_mechanical_ratios.png
    ├── 06_correlation_matrix.png
    ├── 07_feature_target_relationships.png
    ├── 08_training_convergence_curves.png
    ├── 09_parity_plots_pred_vs_actual.png
    ├── 10_residual_error_distributions.png
    ├── 11_ratio_comparison_gt_vs_pred.png
    ├── 12_model_performance_comparison.png
    ├── 13_physical_compliance_audit.png
    ├── 3dcp_workflow.png
    └── 3dcp_printer.jpg
```

---

## 2. Compilation Instructions

### Option A: Local Compilation (Command Line)

Ensure you have TeX Live, MacTeX, or MiKTeX installed. Run the standard sequence:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using `latexmk`:

```bash
latexmk -pdf main.tex
```

### Option B: Compiling on Overleaf

1. Download the `paper.zip` archive (or compress this `paper/` folder into a ZIP file).
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project**.
3. Select `paper.zip`. Overleaf will automatically detect `main.tex`, `IEEEtran.cls`, and `references.bib` and compile the document into a high-fidelity two-column IEEE PDF.

---

## 3. Paper Highlights & Key Results

- **Dataset:** 4,420 multivariate concrete specimens from the ConcreteXAI benchmark.
- **Problem Solved:** Extreme asymmetric missingness ($C_s$: 100.0\% complete; $T_s$: 21.7\% missing; $F_s$: 60.2\% missing). Complete triplet ground truth on only 39.8\% of specimens.
- **Methodology (PIM-MTL):**
  - Shared 3-layer deep latent trunk with `SafeBatchNorm1d` (safe for single-mix edge inference $N=1$).
  - Exact **Masked Gradient Gating**: missing targets yield strictly zero backpropagation gradient (0\% data dropped, 0\% synthetic imputation).
  - **Differentiable Physics Regularizer**: enforces ACI 318 / Eurocode 2 ratio bounds ($0.06 \le T_s/C_s \le 0.18$ and $0.08 \le F_s/C_s \le 0.25$) on all 4,420 specimens.
- **Held-Out Test Results ($N=884$):**
  - **Compressive ($C_s$):** $R^2 = 0.9818$, $\text{RMSE} = 1.3143$~MPa, $\text{MAE} = 0.9931$~MPa
  - **Tensile ($T_s$):** $R^2 = 0.9857$, $\text{RMSE} = 0.1097$~MPa, $\text{MAE} = 0.0820$~MPa
  - **Flexural ($F_s$):** $R^2 = 0.9972$, $\text{RMSE} = 0.1262$~MPa, $\text{MAE} = 0.0990$~MPa
  - **Physically Impossible Predictions ($T_s \ge C_s$):** **0.00\%** (100\% compliant across all 884 test mixes).
