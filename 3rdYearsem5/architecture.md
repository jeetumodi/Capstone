# System Architecture and Technical Specification: Physics-Informed Masked Multi-Task Learning (PIM-MTL)

**Project:** ConcreteXAI Multi-Task Concrete Mechanical Property Prediction  
**Repository Path:** `c:\Users\jeetu\projects\Capstone\3rdYearsem5`  
**Document Classification:** Software Architecture, Machine Learning Engineering & Technical Specification  
**Primary Deliverable:** `architecture.md`  
**Execution Environment:** Python 3.12+ | PyTorch 2.0+ (CUDA & CPU compatible)  
**Status:** Implemented, Evaluated, and Verified  

---

## Executive Summary

The **Physics-Informed Masked Multi-Task Learning (PIM-MTL)** project provides an end-to-end deep learning framework engineered to concurrently predict three interrelated mechanical properties of structural concrete:
1. **Compressive Strength ($C_s$, MPa)** — Uniaxial crushing capacity (ASTM C39 / ACI 318).
2. **Splitting Tensile Strength ($T_s$, MPa)** — Indirect diametral tensile capacity (ASTM C496).
3. **Flexural Strength / Modulus of Rupture ($F_s$, MPa)** — Extreme fiber bending capacity (ASTM C78).

Empirical materials science datasets inevitably exhibit **severe asymmetric missingness** due to the prohibitive financial and labor costs associated with destructive tensile and flexural testing. Rather than resorting to destructive complete-case analysis (which discards 60.2% of data) or synthetic data imputation (which distorts the empirical gradient landscape), PIM-MTL introduces **Masked Gradient Gating** in tandem with **Differentiable Continuum Mechanics Regularization**.

This document serves as the comprehensive architectural reference for the entire codebase, tracing data ingestion, tensor transformation, neural network topology, mathematical optimization, gradient mechanics, empirical evaluation, and production deployment boundaries.

---

## 1. Project Overview

### 1.1 Project Purpose & Problem Statement
Concrete is the foundation of modern civil infrastructure. Accurate estimation of its mechanical capacity is vital for structural safety, serviceability design, and carbon-reduction initiatives (such as clinker substitution and industrial byproduct utilization). 

Standard concrete mix characterization requires evaluating three strength properties ($C_s$, $T_s$, $F_s$). However, across 4,420 multivariate concrete specimen records in the benchmark **ConcreteXAI** dataset, laboratory testing regimes reveal severe asymmetric missingness:
- **Compressive Strength ($C_s$):** 4,420 observed (100.0% coverage).
- **Splitting Tensile Strength ($T_s$):** 3,460 observed, 960 missing (78.3% coverage; 21.7% missingness).
- **Flexural Strength ($F_s$):** 1,760 observed, 2,660 missing (39.8% coverage; 60.2% missingness).
- **Complete Observation Triplet ($C_s \land T_s \land F_s$):** Only 1,760 specimens (39.8% of the dataset).

### 1.2 Motivation and Practical Significance
Conventional machine learning approaches fail in this operating regime:
1. **Complete-Case Analysis (Listwise Deletion):** Discarding rows containing any missing value eliminates **2,660 specimens (60.2% of the entire dataset)**. This radically diminishes statistical power and introduces catastrophic **laboratory selection bias**, because flexural testing is disproportionately conducted on high-performance structural batches rather than conventional mixes.
2. **Statistical Imputation (Mean, Median, KNN, MICE):** Imputing missing mechanical capacities fabricates pseudo-labels. When the loss gradient backpropagates through imputed values, the neural network optimizes against synthetic statistical artifacts rather than true physical hydration mechanisms.
3. **Disjoint Single-Task Models:** Training separate regressors for $C_s$, $T_s$, and $F_s$ treats interrelated mechanical failure modes as independent phenomena, forfeiting cross-task feature sharing and leaving the flexural model starved of training signal.
4. **Unconstrained Deep Learning:** Standard black-box neural networks routinely output physically impossible predictions on out-of-distribution or incomplete test specimens—such as tensile capacity exceeding compressive strength ($T_s \ge C_s$), negative strength values, or strength ratios violating building code envelopes.

### 1.3 Main Objectives
The core objectives of the PIM-MTL architecture are:
- **Unified Multi-Task Prediction:** Predict $[C_s, T_s, F_s]$ concurrently from an identical 19-dimensional feature representation using a single forward pass.
- **Strict Gradient Gating:** Enforce that unobserved target values generate strictly **zero empirical error gradient** during backpropagation, utilizing all 4,420 specimens without imputation.
- **Continuous Physics Regularization:** Embed ACI 318-19 and Eurocode 2 mechanical ratio bounds ($0.06 \le T_s/C_s \le 0.18$ and $0.08 \le F_s/C_s \le 0.25$) directly into the loss landscape as an inductive bias evaluated across 100% of samples.
- **Physical Feasibility Elimination:** Reduce physically impossible predictions ($T_s \ge C_s$) to strictly **0.00%**, maintaining competitive $R^2 > 0.98$ predictive accuracy across all tasks.

### 1.4 Expected System Inputs and Outputs
- **Input:** 4 categorical concrete mix properties (`Type_of_cement`, `Brand`, `Additives`, `Type_of_aggregates`) and 4 continuous engineering measurements (`Design_Fc_Mpa`, `Curing_age_days`, `Er_ohm-cm`, `UPV_m/s`).
- **Output:** A calibrated continuous 3-vector $\hat{\mathbf{y}} = [\hat{C}_s, \hat{T}_s, \hat{F}_s]^T \in \mathbb{R}^3$ expressed in megapascals (MPa).

### 1.5 Scope and Limitations of the Implementation
- **Current Verified Implementation:** A hard parameter sharing deep neural network (`PIM_MTL_Net`), a custom differentiable masked and physics-regularized loss function (`PIM_MTL_Loss`), an automated preprocessing pipeline (`RobustScaler` + `OneHotEncoder`), a training and evaluation script (`main.py`), and a complete analytical Jupyter Notebook (`notebooks/training_and_analysis.ipynb`).
- **Architectural Scope Limitations:**
  - The model assumes static ACI ratio bounds; dynamic bounds conditioned on aggregate mineralogy or moisture content are not currently parameterized.
  - The codebase currently operates as a local batch training and offline evaluation suite; web APIs, cloud microservices, and streaming pipelines are unverified and represent future deployment targets.

---

## 2. System Architecture at a Glance

The PIM-MTL system architecture follows an end-to-end, single-stage differentiable pipeline that couples tabular preprocessing, dense neural encoding, multi-head regression, and hybrid empirical-physical optimization.

```mermaid
graph TD
    subgraph Data Layer ["1. Data Ingestion & Storage"]
        RawCSV["Raw Dataset: Data.csv (4,420 specimens)"]
        RawCSV --> Splitter["Deterministic Splitter (80% Train / 20% Test, Seed=42)"]
    end

    subgraph Preproc Layer ["2. Preprocessing & Leakage Prevention"]
        Splitter --> TrainSplit["Train Split (3,536 mixes)"]
        Splitter --> TestSplit["Test Split (884 mixes)"]
        
        TrainSplit --> OHE["OneHotEncoder (4 Categoricals -> 15 Dims)"]
        TrainSplit --> RobustX["RobustScaler X (4 Continuous Features)"]
        TrainSplit --> MaskGen["Binary Mask Generator M in {0, 1}^(N x 3)"]
        TrainSplit --> RobustY["Task-Specific RobustScalers (Fitted on Observed y)"]
        
        OHE -.-> TransTestX["Transform Test Categoricals"]
        RobustX -.-> TransTestX2["Transform Test Continuous"]
        RobustY -.-> TransTestY["Transform Test Targets"]
        TestSplit --> TransTestX
        TestSplit --> TransTestX2
        TestSplit --> TransTestY
    end

    subgraph Model Layer ["3. Neural Network Architecture (PIM_MTL_Net)"]
        InputFeat["Transformed Feature Vector X in R^19"]
        InputFeat --> Dense1["Linear(19 -> 128) + SafeBatchNorm1d + GELU + Dropout(0.2)"]
        Dense1 --> Dense2["Linear(128 -> 64) + SafeBatchNorm1d + GELU"]
        Dense2 --> Dense3["Linear(64 -> 32) [Latent Trunk Representation h_3]"]
        
        Dense3 --> HeadCs["Compressive Head: Linear(32 -> 1) -> y_hat_cs"]
        Dense3 --> HeadTs["Tensile Head: Linear(32 -> 1) -> y_hat_ts"]
        Dense3 --> HeadFs["Flexural Head: Linear(32 -> 1) -> y_hat_fs"]
        
        HeadCs --> CatPreds["Concatenated Output Vector: y_hat in R^3"]
        HeadTs --> CatPreds
        HeadFs --> CatPreds
    end

    subgraph Loss Layer ["4. Hybrid Objective Optimization (PIM_MTL_Loss)"]
        CatPreds --> MaskedLoss["Masked Empirical Loss (L_data)<br/>Element-wise Error Gating by Mask M<br/>Weighted per-task MSE: w=[1.0, 1.2, 1.5]"]
        CatPreds --> DifferentiableDenorm["Differentiable Denormalization to MPa<br/>y_orig = y_hat * Scale + Center"]
        DifferentiableDenorm --> Clamping["Numerical Clamping (Cs >= 1.0, Ts, Fs >= 0.0)"]
        Clamping --> RatioCalc["Dimensionless Ratios: Rt = Ts/Cs, Rf = Fs/Cs"]
        RatioCalc --> PhysLoss["Continuum Physics Loss (L_phys)<br/>Two-Sided Quadratic Hinge Penalties<br/>Rt in [0.06, 0.18], Rf in [0.08, 0.25]"]
        
        MaskedLoss --> TotalLoss["Total Joint Loss: L_total = L_data + 0.5 * L_phys"]
        PhysLoss --> TotalLoss
    end

    subgraph Opt Layer ["5. Optimization & Execution Engine"]
        TotalLoss --> Backward["Backpropagation (Autograd Engine)"]
        Backward --> ClipGrad["Gradient Clipping (max_norm = 5.0)"]
        ClipGrad --> Adam["Adam Optimizer (lr=1e-3, weight_decay=1e-4)"]
        Adam --> Scheduler["Cosine Annealing Scheduler (T_max=60, eta_min=1e-5)"]
    end

    subgraph Evaluation Layer ["6. Quantitative Evaluation & Artifact Serialization"]
        Scheduler --> Checkpoint["Model State: models/pim_mtl_model.pt"]
        RobustY --> PreprocDump["Preprocessors: models/preprocessor.joblib"]
        TransTestX --> EvalRoutine["Evaluation Routine: evaluate_model()"]
        TransTestY --> EvalRoutine
        CatPreds -.-> EvalRoutine
        EvalRoutine --> MetricsTXT["Results Report: results.txt"]
        EvalRoutine --> MetricsCSV["Quantitative CSV: final_evaluation_results.csv"]
        EvalRoutine --> PredsCSV["Predictions & Residuals: final_test_predictions.csv"]
    end
```

### Architectural Component Interaction Summary
1. **Data Ingestion:** Reads tabular data from `Data.csv`. A strictly isolated 80/20 train/test split partitions the data into 3,536 training samples and 884 test samples.
2. **Preprocessing Pipeline:** Encodes 4 categorical variables into 15 binary columns and scales 4 continuous features using median/IQR robust statistics. Generates target masks and task-specific scalers fitted exclusively on observed targets.
3. **Representation Learning:** A 3-layer shared feedforward trunk transforms the 19-dimensional input into a rich 32-dimensional latent feature vector $\mathbf{h}_3$, stabilized by `SafeBatchNorm1d` and smooth `GELU` non-linearities.
4. **Task Regression:** Three dedicated linear heads project $\mathbf{h}_3$ into scaled predictions $[\hat{y}_{cs}, \hat{y}_{ts}, \hat{y}_{fs}]$.
5. **Joint Optimization:** The predictions enter `PIM_MTL_Loss`, where empirical errors on missing labels are mathematically multiplied by $0.0$, while all samples (regardless of label availability) are regularized in physical MPa space against ACI 318 ratio bounds.
6. **Artifact Generation:** Serializes the trained neural weights, preprocessors, performance summaries, and per-specimen test predictions.

---

## 3. Complete Repository Structure

The physical directory hierarchy of the `3rdYearsem5` repository is outlined below:

```text
c:\Users\jeetu\projects\Capstone\3rdYearsem5\
├── .gitignore                          # Standard git exclusion rules
├── .venv/                              # Local Python virtual environment (ignored in git)
├── Data.csv                            # Benchmark ConcreteXAI dataset (4,420 specimens)
├── Papers.txt                          # Literature references for ConcreteXAI and PINN theory
├── README.md                           # Repository introduction and execution quickstart
├── document.md                         # Detailed scientific monograph and mathematical formulations
├── exp.md                              # Comprehensive Viva Voce technical defense manual
├── final_evaluation_results.csv        # Final quantitative test metrics in tabular format
├── final_test_predictions.csv          # Unscaled test set predictions, residuals, and PVR flags
├── generate_notebook.py                # Deterministic Python generator for the master notebook
├── main.py                             # Central execution entry point for training and evaluation
├── report.md                           # Comprehensive scientific monograph and semester comparison
├── requirements.txt                    # Project dependency specification
├── results.txt                         # Formatted test metrics and physical compliance audit report
├── sem5.md                             # Architectural specification and future roadmap
├── sem5report.pdf                      # Compiled PDF semester progress report
├── figures/                            # Exported high-resolution analytical figures
│   ├── 01_target_missingness_profile.png
│   ├── 02_continuous_features_distribution.png
│   ├── 03_categorical_features_distribution.png
│   ├── 04_target_distributions.png
│   ├── 05_ground_truth_mechanical_ratios.png
│   ├── 06_correlation_matrix.png
│   ├── 07_feature_target_relationships.png
│   ├── 08_training_convergence_curves.png
│   ├── 09_parity_plots_pred_vs_actual.png
│   ├── 10_residual_error_distributions.png
│   ├── 11_ratio_comparison_gt_vs_pred.png
│   ├── 12_model_performance_comparison.png
│   └── 13_physical_compliance_audit.png
├── models/                             # Persistent serialization directory
│   ├── .gitkeep                        # Directory marker
│   ├── pim_mtl_model.pt                # Serialized PyTorch state dictionary and network metadata
│   └── preprocessor.joblib             # Serialized scalers, OHE encoders, and center/scale arrays
├── notebooks/                          # Interactive computing environments
│   └── training_and_analysis.ipynb     # Complete 11-step interactive pipeline notebook
├── paper/                              # LaTeX publication assets
│   ├── IEEEtran.bst                    # IEEE Transactions bibliography style
│   ├── IEEEtran.cls                    # IEEE Transactions document class
│   ├── README.md                       # Compilation guide for the paper
│   ├── main.tex                        # Full conference/journal paper manuscript in LaTeX
│   ├── references.bib                  # BibTeX academic citations
│   └── images/                         # Figures formatted for publication inclusion
└── utils/                              # Core neural network and loss modules
    ├── __init__.py                     # Package export declarations
    ├── losses.py                       # Masked loss and differentiable physics penalty functions
    └── models.py                       # PIM_MTL_Net and SafeBatchNorm1d class definitions
```

### 3.1 Component Responsibility Matrix

| File / Component | Primary Responsibilities | Core Classes & Functions | Inputs / Outputs | Architectural Role |
|---|---|---|---|---|
| [`main.py`](file:///c:/Users/jeetu/projects/Capstone/3rdYearsem5/main.py) | End-to-end execution, CLI argument parsing, training loop orchestration, evaluation, and artifact saving | `set_seed()`, `load_and_preprocess()`, `train_model()`, `evaluate_model()`, `main()` | **In:** `Data.csv`<br/>**Out:** `pim_mtl_model.pt`, `preprocessor.joblib`, `results.txt` | Primary Pipeline Controller |
| [`utils/models.py`](file:///c:/Users/jeetu/projects/Capstone/3rdYearsem5/utils/models.py) | Neural network definitions, custom batch norm, parameter initialization, serialization methods | `SafeBatchNorm1d`, `PIM_MTL_Net`, `_init_weights()`, `save()`, `load()` | **In:** Tensor $\mathbf{X} \in \mathbb{R}^{B \times 19}$<br/>**Out:** Tensor $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times 3}$ | Deep Learning Model Topology |
| [`utils/losses.py`](file:///c:/Users/jeetu/projects/Capstone/3rdYearsem5/utils/losses.py) | Functional and modular loss implementations, gradient gating, physical ratio penalization | `PIM_loss()`, `PIM_MTL_Loss`, `set_scalers()` | **In:** $\hat{\mathbf{Y}}, \mathbf{Y}, \mathbf{M} \in \mathbb{R}^{B \times 3}$<br/>**Out:** $\mathcal{L}_{\text{total}}, \mathcal{L}_{\text{data}}, \mathcal{L}_{\text{phys}}$ | Mathematical Optimization Engine |
| [`utils/__init__.py`](file:///c:/Users/jeetu/projects/Capstone/3rdYearsem5/utils/__init__.py) | Exposes clean package interface and imports | Exposes `PIM_loss`, `PIM_MTL_Loss`, `PIM_MTL_Net`, `SafeBatchNorm1d` | Module namespace setup | Packaging Interface |
| `Data.csv` | Experimental benchmark dataset containing mix designs and strength tests | 4,420 experimental records with 12 features/targets | Tabular records | Benchmark Data Repository |
| `models/pim_mtl_model.pt` | Persistent binary checkpoint containing model state dict and input dimensions | PyTorch binary archive | Saved weights and architecture metadata | Trained Model Checkpoint |
| `models/preprocessor.joblib` | Persistent binary pickle of preprocessing transformations | Python dictionary of Scikit-Learn transformers | OHE encoders, RobustScalers, target scaling arrays | Preprocessing Metadata |
| `notebooks/training_and_analysis.ipynb` | Comprehensive visual notebook reproducing data exploration, training, and analysis | 11 fully rendered markdown and code cells | Interactive outputs, inline plots | Research & Exploration Workbench |
| `generate_notebook.py` | Programmatic generator that constructs `training_and_analysis.ipynb` | `create_notebook()`, cell builders | Outputs `.ipynb` file | Notebook Build Automation |
| `paper/main.tex` | Formal scientific paper manuscript prepared for IEEE submission | LaTeX academic document | Typeset IEEE publication | Academic Dissemination |

---

## 4. End-to-End Workflow

The complete execution lifecycle of the PIM-MTL system proceeds through deterministic, mathematically verified stages from raw data ingestion to audited inference and artifact persistence.

### 4.1 End-to-End Workflow Architecture Diagram

```mermaid
flowchart TD
    classDef startEnd fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef process fill:#0f172a,stroke:#64748b,stroke-width:1.5px,color:#f1f5f9;
    classDef decision fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef lossBox fill:#31101a,stroke:#f43f5e,stroke-width:2px,color:#ffe4e6;
    classDef outputBox fill:#062e22,stroke:#10b981,stroke-width:2px,color:#ecfdf5;

    Start([CLI Entry Point: python main.py]):::startEnd --> S1[Phase 1: Deterministic Environment Setup<br/>set_seed: Python, NumPy, PyTorch, CUDA, CUDNN = 42]:::process
    S1 --> S2[Phase 2: Ingestion & Validation<br/>Load Data.csv - 4,420 specimens x 12 columns<br/>Clean whitespace from categorical strings]:::process
    S2 --> S3[Phase 3: Stratified Split<br/>train_test_split: 80% Train 3,536 / 20% Test 884<br/>Strict random_state = 42 to prevent leakage]:::process

    subgraph Preproc ["Phase 4: Preprocessing & Mask Extraction (Fitted Exclusively on Train Split)"]
        S3 --> P1[Categorical Encoding<br/>OneHotEncoder on 4 mix variables<br/>Expands to 15 binary indicators]:::process
        S3 --> P2[Continuous Scaling<br/>RobustScaler on 4 continuous variables<br/>x_scaled = x - median / IQR]:::process
        P1 --> P3[Feature Stacking: X in R^(N x 19)]:::process
        P2 --> P3
        
        S3 --> P4[Target Extraction & Masking<br/>y_raw = [Cs, Ts, Fs]<br/>Binary Mask M = ~isnan y_raw]:::process
        P4 --> P5[Task-Specific Target Scalers<br/>RobustScaler fitted ONLY where M_j == 1<br/>Unobserved entries mapped to 0.0 placeholder]:::process
        P5 --> P6[Preproc Bundle: OHE + Scalers + Centers + Scales]:::process
    end

    P3 --> S4[Phase 5: TensorDataset & Batch Loader<br/>TensorDataset: X, Y_scaled, Mask<br/>DataLoader: batch_size = 64, shuffle = True]:::process
    P5 --> S4

    subgraph ModelInit ["Phase 6: Model & Objective Instantiation"]
        S4 --> M1[Instantiate PIM_MTL_Net<br/>Trunk: Linear 19 to 128 to 64 to 32<br/>Heads: Cs 32 to 1, Ts 32 to 1, Fs 32 to 1<br/>Kaiming Normal He Weight Init]:::process
        P6 --> M2[Instantiate PIM_MTL_Loss<br/>Register Target Centers & Scales as Buffers<br/>Set lambda_phys = 0.5, Task Weights = 1.0, 1.2, 1.5]:::process
        M1 --> M3[Optimizer & Schedule Setup<br/>Adam lr=1e-3, weight_decay=1e-4<br/>CosineAnnealingLR T_max=60, eta_min=1e-5]:::process
    end

    subgraph TrainLoop ["Phase 7: Optimization Loop (Epochs 1 to 60)"]
        M3 --> B1[Fetch Mini-Batch: bx, by, bm of size B=64]:::process
        B1 --> B2[Zero Gradients: optimizer.zero_grad]:::process
        B2 --> B3[Forward Pass: preds = model bx -> R^(B x 3)]:::process
        
        B3 --> L1[Dual Loss Channel 1: Masked Data Loss<br/>Error Matrix = bm * preds - by ^ 2<br/>Normalized by N_obs_j = max 1, sum bm_j<br/>Gradient for unobserved entries = STRICTLY 0.0]:::lossBox
        B3 --> L2[Dual Loss Channel 2: Continuum Physics Loss<br/>Differentiable denorm to MPa: y_orig = preds * s + mu<br/>Clamping: Cs >= 1.0, Ts, Fs >= 0.0<br/>Ratios: Rt = Ts/Cs, Rf = Fs/Cs<br/>Quadratic Hinge Penalties evaluated on 100% of samples]:::lossBox
        
        L1 --> L3[Composite Total Loss<br/>L_total = L_data + 0.5 * L_phys]:::lossBox
        L2 --> L3
        
        L3 --> B4[Autograd Backpropagation: L_total.backward]:::process
        B4 --> B5[Gradient Norm Clipping<br/>torch.nn.utils.clip_grad_norm_ max_norm=5.0]:::process
        B5 --> B6[Parameter Update: optimizer.step]:::process
        B6 --> B7{More Batches in Epoch?}:::decision
        B7 -- Yes --> B1
        B7 -- No --> B8[Scheduler Step: scheduler.step]:::process
        B8 --> B9{Epoch == 60?}:::decision
        B9 -- No --> B1
    end

    subgraph EvalPhase ["Phase 8: Evaluation & Feasibility Audit"]
        B9 -- Yes --> E1[Switch to model.eval with torch.no_grad]:::process
        E1 --> E2[Forward Pass on Test Set X_test in R^(884 x 19)]:::process
        E2 --> E3[Denormalize to Real MPa Units<br/>preds_raw = preds_scaled * scales + centers]:::process
        E3 --> E4[Quantitative Statistical Accuracy<br/>R2, RMSE, MAE evaluated exclusively on observed entries M_test == 1]:::process
        E3 --> E5[Physical Compliance Audit on ALL 884 Test Samples<br/>PVR_ts: Rt outside 0.06 to 0.18<br/>PVR_fs: Rf outside 0.08 to 0.25<br/>PVR_imp: Ts >= Cs physically impossible check]:::process
    end

    subgraph ArtifactExport ["Phase 9: Checkpointing & Artifact Serialization"]
        E4 --> A1[Save Report: results.txt]:::outputBox
        E5 --> A1
        E4 --> A2[Save Metrics CSV: final_evaluation_results.csv]:::outputBox
        E5 --> A3[Save Predictions CSV: final_test_predictions.csv]:::outputBox
        M1 --> A4[Save PyTorch Checkpoint: models/pim_mtl_model.pt]:::outputBox
        P6 --> A5[Save Preprocessor Bundle: models/preprocessor.joblib]:::outputBox
    end

    A1 --> EndNode([Pipeline Complete: Execution Audited]):::startEnd
    A2 --> EndNode
    A3 --> EndNode
    A4 --> EndNode
    A5 --> EndNode
```

### 4.2 Sequence Interaction Lifecycle

The sequence diagram below details the operational interactions between components during execution:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / CLI
    participant Main as main.py (Pipeline)
    participant Pre as load_and_preprocess()
    participant DS as TensorDataset / DataLoader
    participant Net as PIM_MTL_Net
    participant Loss as PIM_MTL_Loss
    participant Opt as Adam + Scheduler
    participant Eval as evaluate_model()
    participant Disk as Persistent Storage

    User->>Main: python main.py --data Data.csv --epochs 60 --batch-size 64
    Main->>Main: set_seed(42) [Python, NumPy, PyTorch, CUDNN]
    Main->>Pre: load_and_preprocess("Data.csv")
    Pre->>Pre: Strip whitespace from categorical strings
    Pre->>Pre: train_test_split(df, test_size=0.2, random_state=42)
    Pre->>Pre: OneHotEncoder.fit_transform(Train) -> Transform(Test)
    Pre->>Pre: RobustScaler_X.fit_transform(Train) -> Transform(Test)
    Pre->>Pre: Compute Target Observation Masks (mask = ~isnan(y))
    Pre->>Pre: RobustScaler_Y per task fitted strictly on observed y
    Pre-->>Main: Returns (X_tr, y_tr_scaled, m_tr), (X_te, y_te_scaled, m_te), preprocessor
    
    Main->>Net: Instantiate PIM_MTL_Net(input_dim=19)
    Main->>Loss: Instantiate PIM_MTL_Loss(target_centers, target_scales, lambda_phys=0.5)
    Main->>Opt: Instantiate Adam(lr=1e-3, weight_decay=1e-4) & CosineAnnealingLR
    Main->>DS: Build DataLoader(batch_size=64, shuffle=True)

    loop Epoch 1 to 60
        loop Each Mini-Batch (bx, by, bm)
            Main->>Net: Forward pass: preds = model(bx)
            Net-->>Main: Return preds in R^(B x 3)
            Main->>Loss: criterion(preds, by, bm)
            Loss->>Loss: Compute Masked Empirical Loss L_data
            Loss->>Loss: Differentiable denorm to MPa & calculate Rt, Rf
            Loss->>Loss: Compute Physics Penalty Loss L_phys
            Loss-->>Main: Return L_total, L_data, L_phys
            Main->>Loss: L_total.backward()
            Main->>Net: torch.nn.utils.clip_grad_norm_(max_norm=5.0)
            Main->>Opt: optimizer.step()
        end
        Main->>Opt: scheduler.step()
    end

    Main->>Eval: evaluate_model(model, X_test, y_test_raw, mask_test, preprocessor)
    Eval->>Net: torch.no_grad() -> model(X_test)
    Eval->>Eval: Denormalize predictions to MPa: preds_raw = preds * scales + centers
    Eval->>Eval: Calculate R2, RMSE, MAE per task on observed test entries
    Eval->>Eval: Calculate Physical Violation Rates (PVR_ts, PVR_fs, PVR_impossible)
    Eval->>Disk: Write results.txt
    Eval-->>Main: Return metrics_summary, pvr_dict

    Main->>Disk: model.save("models/pim_mtl_model.pt")
    Main->>Disk: joblib.dump(preprocessor, "models/preprocessor.joblib")
    Main-->>User: Pipeline Complete: Models and Reports Saved
```

### Detailed Execution Phase Breakdown

1. **Deterministic Environment Initialization (`set_seed`):** Initializes identical random seeds (`seed=42`) across Python's `random`, `numpy.random`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`. Forces `torch.backends.cudnn.deterministic = True` and disables benchmarking heuristics to ensure exact reproducibility across compute environments.
2. **Data Ingestion & Cleaning:** Reads `Data.csv` into a Pandas DataFrame and strips extraneous whitespace from categorical string fields.
3. **Partitioning:** Conducts an 80/20 train/test split via Scikit-Learn `train_test_split(random_state=42)` yielding 3,536 training samples and 884 test samples.
4. **Encoding & Feature Scaling:** Fits `OneHotEncoder(sparse_output=False, handle_unknown="ignore")` and `RobustScaler()` exclusively on the training partition. Transforms training and test continuous and categorical features, concatenating them horizontally into a 19-dimensional float32 matrix $\mathbf{X}$.
5. **Target Masking & Scaling:** Constructs binary masks $\mathbf{M} \in \{0, 1\}^{N \times 3}$. For each target property ($C_s$, $T_s$, $F_s$), a dedicated `RobustScaler` is fitted strictly on observed samples ($m_{ij} = 1$). Unobserved values in the scaled target matrix are populated with computational neutral placeholders ($0.0$).
6. **Model Instantiation:** Constructs `PIM_MTL_Net` with an input dimension of 19 and hidden layers $[128, 64, 32]$, initializing linear layer weights with Kaiming Normal (He) initialization.
7. **Loss Module Configuration:** Configures `PIM_MTL_Loss` registering the denormalization vectors $\mathbf{s} \in \mathbb{R}^3$ and $\boldsymbol{\mu} \in \mathbb{R}^3$ as buffers, with physics penalty weight $\lambda_{\text{phys}} = 0.5$ and task weights $\mathbf{w} = [1.0, 1.2, 1.5]$.
8. **Iterative Optimization:** Executes 60 epochs using mini-batches of size 64. Gradients are backpropagated through the joint loss, clipped to a maximum norm of 5.0, and updated via Adam. Learning rate decays smoothly from $10^{-3}$ to $10^{-5}$ via Cosine Annealing.
9. **Quantitative Evaluation:** Evaluates test specimens under `torch.no_grad()`, projecting predictions back to real physical MPa units. Calculates $R^2$, RMSE, and MAE strictly on observed entries, and calculates Physical Violation Rates (PVR) across all 884 specimens.
10. **Serialization:** Writes binary checkpoints and human-readable text logs to disk.

---

## 5. Data Architecture and Data Flow

### 5.1 Dataset Schema and Feature Catalog
The `Data.csv` benchmark comprises 4,420 rows and 12 columns:

| Column Name | Type | Unit | Missingness | Description & Physical Role |
|---|---|---|---|---|
| `id` | Identifier | Integer | 0.0% (0) | Unique specimen sequential index. Excluded from model features. |
| `Type_of_cement` | Categorical (4) | String | 0.0% (0) | Standard cement designation (`CPC 30R`, `CPO 30R RS BRA`, `CPC 30R RS`, `CPC 40R`). Controls hydration kinetics and early strength. |
| `Brand` | Categorical (2) | String | 0.0% (0) | Cement manufacturing supplier (`CEMEX`, `Apasco`). Reflects clinker composition and fineness. |
| `Additives` | Categorical (5) | String | 0.0% (0) | Chemical and mineral admixtures (`None`, `Opuntia_ficus_indica`, `10%_blast_furnace_slag`, `fluidizers`, etc.). Modifies rheology and ITZ. |
| `Type_of_aggregates` | Categorical (4) | String | 0.0% (0) | Mineralogy and morphology of coarse aggregates (`Crushed`, `Rounded`, `Recycled`, `Volcanic`). Governs aggregate interlock and modulus. |
| `Design_Fc_Mpa` | Numerical | MPa | 0.0% (0) | Nominal structural design compressive strength ($f'_c \in [20, 35]$ MPa). |
| `Curing_age_days` | Numerical | Days | 0.0% (0) | Water immersion curing duration ($t \in [3, 120]$ days). Governs hydration maturity. |
| `Er_ohm-cm` | Numerical | $\Omega \cdot \text{cm}$ | 0.0% (0) | Electrical resistivity ($[1.43, 23.95]\ \Omega\cdot\text{cm}$). Proxy for porosity, pore solution conductivity, and microcracks. |
| `UPV_m/s` | Numerical | $\text{m/s}$ | 0.0% (0) | Ultrasonic Pulse Velocity ($[2,408, 5,319]\ \text{m/s}$). Non-destructive acoustic transit speed measuring density and matrix stiffness. |
| `Cs_Mpa` | Target (Continuous) | MPa | 0.0% (0) | Uniaxial Compressive Strength ($[5.11, 55.71]$ MPa, $\mu=31.19, \sigma=7.91$). Observed across all 4,420 mixes. |
| `Ts_Mpa` | Target (Continuous) | MPa | 21.7% (960) | Splitting Tensile Strength ($[0.55, 5.27]$ MPa, $\mu=2.48, \sigma=0.65$). Observed for 3,460 mixes. |
| `Fs_Mpa` | Target (Continuous) | MPa | 60.2% (2,660) | Flexural Strength / Modulus of Rupture ($[0.29, 8.09]$ MPa, $\mu=3.42, \sigma=1.34$). Observed for only 1,760 mixes. |

### 5.2 Transformation Pipeline and Tensor Shapes

| Pipeline Stage | Input Structure & Dimension | Transformation Operation | Output Structure & Dimension | Responsible Code Component |
|---|---|---|---|---|
| **Raw Loading** | Disk File (`Data.csv`) | CSV parsing and categorical string stripping | Pandas DataFrame `(4420, 12)` | `main.py:load_and_preprocess` |
| **Splitting** | DataFrame `(4420, 12)` | Stratified random split (80/20, `seed=42`) | Train DF `(3536, 12)`, Test DF `(884, 12)` | `sklearn.model_selection.train_test_split` |
| **Categorical Encoding** | Categorical slices `(N, 4)` | One-hot encoding (`handle_unknown='ignore'`) | Binary matrix `(N, 15)` | `sklearn.preprocessing.OneHotEncoder` |
| **Feature Scaling** | Continuous slice `(N, 4)` | Median / IQR scaling: $(x - \text{median}) / \text{IQR}$ | Float32 matrix `(N, 4)` | `sklearn.preprocessing.RobustScaler` |
| **Feature Fusion** | Continuous `(N, 4)` + OHE `(N, 15)` | Horizontal column stacking (`np.hstack`) | Feature Matrix $\mathbf{X} \in \mathbb{R}^{N \times 19}$ | `numpy.hstack` |
| **Mask Extraction** | Raw Targets `(N, 3)` | Logical NOT of NaN: `~np.isnan(y_raw)` | Binary Mask $\mathbf{M} \in \{0.0, 1.0\}^{N \times 3}$ | `main.py:load_and_preprocess` |
| **Target Scaling** | Observed entries `(N_obs, 1)` | Separate RobustScaler per target task | Scaled Matrix $\mathbf{Y}_{\text{scaled}} \in \mathbb{R}^{N \times 3}$ (NaN $\rightarrow$ 0.0) | `main.py:load_and_preprocess` |
| **Batching** | Arrays $\mathbf{X}, \mathbf{Y}, \mathbf{M}$ | PyTorch Tensor conversion & batch slicing | Batches $(B, 19), (B, 3), (B, 3)$ where $B=64$ | `torch.utils.data.DataLoader` |

### 5.3 Missing Feature Handling vs. Missing Target Handling
An essential architectural distinction in PIM-MTL is the difference between input features and target variables:
- **Input Features ($X$):** The ConcreteXAI dataset has **zero missing values** across its 8 input features. If an input feature were missing in production, it would require feature-level imputation or virtual sensing (such as the tree-based virtual sensing models explored in 2nd semester research).
- **Target Variables ($Y$):** Target missingness is **asymmetric and severe** (21.7% for $T_s$, 60.2% for $F_s$). Missing targets are handled via **dynamic loss masking and gradient gating**, ensuring the computational graph receives strictly zero empirical error gradient from unobserved targets while still regularizing them via continuum physics.

---

## 6. Machine Learning Model Architecture

The deep neural network architecture is defined in `utils/models.py` by the class `PIM_MTL_Net`. It adopts a **Hard Parameter Sharing Multi-Task Topology**, where lower layers extract generalized microstructural hydration features, which then bifurcate into task-specific regression heads.

```mermaid
graph TD
    In[Input Feature Tensor: X in R^(B x 19)] --> FC1[Linear: 19 -> 128]
    FC1 --> BN1[SafeBatchNorm1d: 128 channels]
    BN1 --> Act1[GELU Activation]
    Act1 --> Drop1[Dropout: p = 0.2]
    
    Drop1 --> FC2[Linear: 128 -> 64]
    FC2 --> BN2[SafeBatchNorm1d: 64 channels]
    BN2 --> Act2[GELU Activation]
    
    Act2 --> FC3[Linear: 64 -> 32]
    
    subgraph Shared Representation Trunk ["Shared Hydration Representation (h_3 in R^32)"]
        FC3
    end
    
    FC3 --> Head1[Compressive Head: Linear 32 -> 1]
    FC3 --> Head2[Tensile Head: Linear 32 -> 1]
    FC3 --> Head3[Flexural Head: Linear 32 -> 1]
    
    Head1 --> Out1[Predicted Scaled Compressive Strength y_hat_cs]
    Head2 --> Out2[Predicted Scaled Tensile Strength y_hat_ts]
    Head3 --> Out3[Predicted Scaled Flexural Strength y_hat_fs]
    
    Out1 --> Cat[torch.cat(..., dim=-1)]
    Out2 --> Cat
    Out3 --> Cat
    Cat --> FinalOut[Output Tensor: y_hat in R^(B x 3)]
```

### 6.1 Layer-by-Layer Architectural Specification

| Layer ID | Operation / Class | Input Shape | Output Shape | Trainable Parameters | Activation / Regularization | Engineering Purpose |
|---|---|---|---|---|---|---|
| **0. Input** | Input Tensor | $(B, 19)$ | $(B, 19)$ | 0 | None | 4 continuous + 15 one-hot categorical features |
| **1. Trunk FC1** | `nn.Linear` | $(B, 19)$ | $(B, 128)$ | $19 \times 128 + 128 = 2,560$ | Linear projection | High-dimensional embedding of mix proportions |
| **2. Trunk BN1** | `SafeBatchNorm1d` | $(B, 128)$ | $(B, 128)$ | $2 \times 128 = 256$ | Running mean/var tracking | Normalizes activations; handles single-sample edge batches |
| **3. Trunk Act1** | `nn.GELU` | $(B, 128)$ | $(B, 128)$ | 0 | GELU non-linearity | Smooth non-linear mapping; prevents dead neurons |
| **4. Trunk Drop1**| `nn.Dropout` | $(B, 128)$ | $(B, 128)$ | 0 | Dropout rate $p=0.2$ | Mitigates feature co-adaptation across mix properties |
| **5. Trunk FC2** | `nn.Linear` | $(B, 128)$ | $(B, 64)$ | $128 \times 64 + 64 = 8,256$ | Linear compression | Intermediate representation of hydration matrix |
| **6. Trunk BN2** | `SafeBatchNorm1d` | $(B, 64)$ | $(B, 64)$ | $2 \times 64 = 128$ | Running mean/var tracking | Stabilizes internal covariate shift |
| **7. Trunk Act2** | `nn.GELU` | $(B, 64)$ | $(B, 64)$ | 0 | GELU non-linearity | Smooth gradient curvature for physics penalty |
| **8. Trunk FC3** | `nn.Linear` | $(B, 64)$ | $(B, 32)$ | $64 \times 32 + 32 = 2,080$ | Linear bottleneck | Compresses features into shared latent vector $\mathbf{h}_3$ |
| **9. Head Cs** | `nn.Linear` | $(B, 32)$ | $(B, 1)$ | $32 \times 1 + 1 = 33$ | Linear readout | Specializes in compressive crushing failure |
| **10. Head Ts** | `nn.Linear` | $(B, 32)$ | $(B, 1)$ | $32 \times 1 + 1 = 33$ | Linear readout | Specializes in diametral tensile splitting failure |
| **11. Head Fs** | `nn.Linear` | $(B, 32)$ | $(B, 1)$ | $32 \times 1 + 1 = 33$ | Linear readout | Specializes in extreme fiber flexural rupture |
| **12. Output** | `torch.cat` | $3 \times (B, 1)$ | $(B, 3)$ | 0 | Concatenation | Forms multi-task prediction tensor $\hat{\mathbf{Y}}$ |
| **Total** | **PIM_MTL_Net** | **$(B, 19)$** | **$(B, 3)$** | **13,379 parameters** | — | **Compact, efficient, and fully differentiable** |

### 6.2 Parameter Count Derivation
The exact parameter total of **13,379** trainable parameters is verified mathematically:
$$\text{Trunk FC1}: (19 \times 128) + 128 = 2,432 + 128 = 2,560$$
$$\text{Trunk BN1}: 128 \text{ (weight)} + 128 \text{ (bias)} = 256$$
$$\text{Trunk FC2}: (128 \times 64) + 64 = 8,192 + 64 = 8,256$$
$$\text{Trunk BN2}: 64 \text{ (weight)} + 64 \text{ (bias)} = 128$$
$$\text{Trunk FC3}: (64 \times 32) + 32 = 2,048 + 32 = 2,080$$
$$\text{Heads (Cs, Ts, Fs)}: 3 \times ((32 \times 1) + 1) = 3 \times 33 = 99$$
$$\text{Total Parameters} = 2,560 + 256 + 8,256 + 128 + 2,080 + 99 = \mathbf{13,379}$$

### 6.3 Architectural Design Rationale

#### 1. Gaussian Error Linear Units (GELU) vs. Standard ReLU
Standard ReLU introduces sharp derivative discontinuities at $x=0$, leading to "dead neurons" when large negative gradients backpropagate. More critically, computing strength ratios ($T_s / C_s$) involves division. When ratios backpropagate through piece-wise linear ReLU layers, sharp activation kinks introduce high-frequency gradient spikes. GELU provides smooth $C^\infty$ curvature:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]$$
This maintains non-zero probabilistic gradients for negative activations, ensuring dormant neurons recover during training.

#### 2. SafeBatchNorm1d Custom Class
Standard PyTorch `nn.BatchNorm1d` raises a runtime exception when encountering a single-sample batch ($N=1$) during training because sample variance cannot be computed ($N-1=0$). `SafeBatchNorm1d` intercepts the forward pass:
```python
class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, eps=self.eps
            )
        return super().forward(x)
```
If a mini-batch or residual evaluation batch contains only 1 sample, it dynamically switches to running mean and variance statistics, preventing runtime crashes.

#### 3. Kaiming Normal (He) Weight Initialization
All linear layers are initialized via `nn.init.kaiming_normal_(m.weight, nonlinearity="relu")` with biases initialized to $0.01$. This bounds the variance of layer outputs to $1.0$ at epoch zero, preventing vanishing or exploding gradients.

---

## 7. Masked Multi-Task Learning Mechanism

The primary theoretical innovation for handling asymmetric missing data is **Masked Gradient Gating**.

### 7.1 Mathematical Formulation of Masked Loss
Let a mini-batch consist of $N$ specimens. The predicted tensor is $\hat{\mathbf{Y}} \in \mathbb{R}^{N \times 3}$, the target tensor is $\mathbf{Y} \in \mathbb{R}^{N \times 3}$, and the observation mask is $\mathbf{M} \in \{0.0, 1.0\}^{N \times 3}$, where:
$$m_{ij} = \begin{cases} 1.0, & \text{if target } j \text{ is observed for specimen } i \\ 0.0, & \text{if target } j \text{ is missing (NaN in dataset)} \end{cases}$$

The masked empirical data loss $\mathcal{L}_{\text{data}}$ is formulated as:
$$\mathcal{L}_{\text{data}} = \sum_{j=1}^3 w_j \cdot \text{MSE}_j$$
$$\text{MSE}_j = \frac{\sum_{i=1}^N m_{ij} \cdot (\hat{y}_{ij} - y_{ij})^2}{\max\left(1.0, \sum_{k=1}^N m_{kj}\right)}$$
where $\mathbf{w} = [w_{cs}, w_{ts}, w_{fs}] = [1.0, 1.2, 1.5]$ are task weighting coefficients.

### 7.2 Proof of Strict Gradient Gating
We examine the analytical derivative of the empirical loss with respect to the network prediction $\hat{y}_{ij}$ for specimen $i$ on task $j$:
$$\frac{\partial \mathcal{L}_{\text{data}}}{\partial \hat{y}_{ij}} = \frac{\partial}{\partial \hat{y}_{ij}} \left[ w_j \cdot \frac{\sum_{k=1}^N m_{kj} (\hat{y}_{kj} - y_{kj})^2}{\max(1, N_{\text{obs}, j})} \right]$$
$$\frac{\partial \mathcal{L}_{\text{data}}}{\partial \hat{y}_{ij}} = \frac{2 w_j \cdot m_{ij} \cdot (\hat{y}_{ij} - y_{ij})}{\max(1, N_{\text{obs}, j})}$$

Now evaluate both observation conditions:
1. **Case 1: Ground Truth Target is Observed ($m_{ij} = 1.0$):**
   $$\frac{\partial \mathcal{L}_{\text{data}}}{\partial \hat{y}_{ij}} = \frac{2 w_j (\hat{y}_{ij} - y_{ij})}{\max(1, N_{\text{obs}, j})} \neq 0$$
   Normal empirical supervised error backpropagates through the task head into the shared encoder.
2. **Case 2: Ground Truth Target is Missing ($m_{ij} = 0.0$):**
   $$\frac{\partial \mathcal{L}_{\text{data}}}{\partial \hat{y}_{ij}} = \frac{2 w_j \cdot \mathbf{0.0} \cdot (\hat{y}_{ij} - 0.0)}{\max(1, N_{\text{obs}, j})} \equiv \mathbf{0.0}$$
   The empirical gradient is **identically zero**.

$$\therefore \text{Missing target entries generate zero empirical error gradient.}$$
The neutral placeholder value $0.0$ placed in missing matrix cells does not influence network optimization.

### 7.3 Illustrative Batch Example
Consider a hypothetical mini-batch of $N=3$ specimens across targets $[C_s, T_s, F_s]$:

$$\mathbf{M} = \begin{bmatrix} 1.0 & 1.0 & 0.0 \\ 1.0 & 0.0 & 0.0 \\ 1.0 & 1.0 & 1.0 \end{bmatrix}, \quad \mathbf{Y} = \begin{bmatrix} 32.5 & 2.6 & 0.0 \\ 28.0 & 0.0 & 0.0 \\ 45.2 & 3.4 & 4.8 \end{bmatrix}, \quad \hat{\mathbf{Y}} = \begin{bmatrix} 31.8 & 2.4 & 3.9 \\ 29.1 & 2.1 & 3.2 \\ 44.0 & 3.6 & 4.5 \end{bmatrix}$$

1. **Observed Counts per Task:**
   $$N_{\text{obs}, Cs} = 1+1+1 = 3, \quad N_{\text{obs}, Ts} = 1+0+1 = 2, \quad N_{\text{obs}, Fs} = 0+0+1 = 1$$
2. **Masked Squared Error:**
   $$\mathbf{E}_{\text{sq}} = (\hat{\mathbf{Y}} - \mathbf{Y})^2 = \begin{bmatrix} 0.49 & 0.04 & 15.21 \\ 1.21 & 4.41 & 10.24 \\ 1.44 & 0.04 & 0.09 \end{bmatrix}$$
   $$\mathbf{E}_{\text{masked}} = \mathbf{M} \odot \mathbf{E}_{\text{sq}} = \begin{bmatrix} 0.49 & 0.04 & \mathbf{0.00} \\ 1.21 & \mathbf{0.00} & \mathbf{0.00} \\ 1.44 & 0.04 & 0.09 \end{bmatrix}$$
3. **Per-Task Normalized MSE:**
   $$\text{MSE}_{Cs} = \frac{0.49 + 1.21 + 1.44}{3} = \frac{3.14}{3} = 1.047$$
   $$\text{MSE}_{Ts} = \frac{0.04 + 0.00 + 0.04}{2} = \frac{0.08}{2} = 0.040$$
   $$\text{MSE}_{Fs} = \frac{0.00 + 0.00 + 0.09}{1} = \frac{0.09}{1} = 0.090$$
4. **Weighted Empirical Loss:**
   $$\mathcal{L}_{\text{data}} = (1.0 \times 1.047) + (1.2 \times 0.040) + (1.5 \times 0.090) = 1.047 + 0.048 + 0.135 = 1.230$$

Notice that for specimen 1, the flexural prediction error $(3.9 - 0.0)^2 = 15.21$ is multiplied by $0.0$, producing zero contribution to $\text{MSE}_{Fs}$.

### 7.4 Dual Inductive Supervision for Unobserved Tasks
If the empirical error gradient is zero for unobserved labels, how does the model learn to predict flexural strength on specimens where $F_s$ is unobserved?
It learns via **Dual Inductive Supervision**:
1. **Shared Trunk Backbone:** Compressive strength is 100% complete across all 4,420 specimens. Backpropagating $\text{MSE}_{Cs}$ continuously tunes the shared encoder weights ($\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3$). The latent representation $\mathbf{h}_3$ develops a rich physical understanding of cement type, curing time, and ultrasonic pulse velocity that directly benefits the flexural head.
2. **Unmasked Continuum Physics Loss:** The physics loss $\mathcal{L}_{\text{phys}}$ is evaluated over **all $N$ specimens** in every batch, regularizing the flexural head relative to the predicted compressive strength even when no experimental flexural ground truth exists.

---

## 8. Physics-Informed Neural Network (PINN) Architecture & Theory

The Physics-Informed Neural Network (PINN) component of PIM-MTL embeds governing civil engineering constitutive laws directly into the neural network's loss landscape. Rather than permitting unconstrained optimization, the PINN acts as an **inductive physical prior** that restricts the hypothesis space to physically admissible structural strength states.

### 8.1 Dedicated PINN Architecture Diagram

The diagram below details the dedicated PINN architecture, showing the forward surrogate mapping, differentiable physical transformation manifold, ratio formulation, constitutive code evaluation gates, and the dual-channel backpropagation pathways:

```mermaid
flowchart TD
    classDef inputNode fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef surrogateNode fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef physLayer fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#f0fdf4;
    classDef ratioNode fill:#713f12,stroke:#eab308,stroke-width:2px,color:#fefce8;
    classDef gateNode fill:#4c0519,stroke:#f43f5e,stroke-width:2px,color:#ffe4e6;
    classDef lossNode fill:#31101a,stroke:#e11d48,stroke-width:2px,color:#fff1f2;
    classDef gradNode fill:#1e293b,stroke:#a855f7,stroke-width:2px,color:#f8fafc;

    subgraph Surrogate ["1. Deep Parametric Surrogate Network f_theta(x)"]
        X["Input Feature Vector: x in R^19<br/>(4 Mix Categoricals + 4 NDT/Age Continuous)"]:::inputNode --> Trunk["Shared Representation Trunk<br/>Linear(19->128)->SafeBN->GELU->Dropout<br/>Linear(128->64)->SafeBN->GELU<br/>Linear(64->32) [Latent Vector h_3]"]:::surrogateNode
        Trunk --> HeadCs["Linear(32->1): Scaled Compressive Head"]:::surrogateNode
        Trunk --> HeadTs["Linear(32->1): Scaled Tensile Head"]:::surrogateNode
        Trunk --> HeadFs["Linear(32->1): Scaled Flexural Head"]:::surrogateNode
        HeadCs --> ScaledY["Normalized Output Tensor: y_scaled in R^(B x 3)<br/>[y_hat_cs, y_hat_ts, y_hat_fs]"]:::surrogateNode
        HeadTs --> ScaledY
        HeadFs --> ScaledY
    end

    subgraph PhysManifold ["2. Differentiable Physical Manifold Layer"]
        ScaledY --> Denorm["Differentiable Denormalization to MPa<br/>y_orig = y_scaled * Target_Scales + Target_Centers"]:::physLayer
        Denorm --> Clamp["Singularity Safeguard Clamping<br/>Cs = max(y_orig_0, 1.0 MPa)<br/>Ts = max(y_orig_1, 0.0 MPa)<br/>Fs = max(y_orig_2, 0.0 MPa)"]:::physLayer
    end

    subgraph RatioEng ["3. Dimensionless Mechanical Ratio Formulation"]
        Clamp --> CalcRt["Splitting Tensile Ratio:<br/>Rt = Ts / Cs"]:::ratioNode
        Clamp --> CalcRf["Flexural Rupture Ratio:<br/>Rf = Fs / Cs"]:::ratioNode
    end

    subgraph CodeGates ["4. ACI 318 / Eurocode 2 Constitutive Envelopes"]
        CalcRt --> GateRtLow["Lower Tensile Boundary Evaluator<br/>Penalty_1 = ReLU(0.06 - Rt)^2"]:::gateNode
        CalcRt --> GateRtHigh["Upper Tensile Boundary Evaluator<br/>Penalty_2 = ReLU(Rt - 0.18)^2"]:::gateNode
        CalcRf --> GateRfLow["Lower Flexural Boundary Evaluator<br/>Penalty_3 = ReLU(0.08 - Rf)^2"]:::gateNode
        CalcRf --> GateRfHigh["Upper Flexural Boundary Evaluator<br/>Penalty_4 = ReLU(Rf - 0.25)^2"]:::gateNode
    end

    subgraph LossAssembly ["5. Scalar Physics Loss Computation"]
        GateRtLow --> SumPhys["Summed Violation Terms across All B Specimens"]:::lossNode
        GateRtHigh --> SumPhys
        GateRfLow --> SumPhys
        GateRfHigh --> SumPhys
        SumPhys --> MeanPhys["Batch Mean Physics Loss:<br/>L_phys = (1/B) * Sum(Penalties)"]:::lossNode
        MeanPhys --> WeightPhys["Weighted Physics Penalty Term:<br/>lambda_phys * L_phys (lambda_phys = 0.5)"]:::lossNode
    end

    subgraph DualAutograd ["6. Dual-Channel Backpropagation Dynamics"]
        WeightPhys --> GradPhys["Restorative Physics Gradient: grad_theta(lambda_phys * L_phys)<br/>ACTIVATED ON 100% OF SAMPLES (Observed & Unobserved)"]:::gradNode
        ScaledY -.-> LossData["Masked Empirical Loss: L_data<br/>Error * Binary Mask M"]:::lossNode
        LossData --> GradData["Empirical Error Gradient: grad_theta(L_data)<br/>ACTIVE ONLY on Observed Targets (M=1)"]:::gradNode
        GradPhys --> JointGrad["Joint Parameter Update:<br/>Theta <- Theta - eta * (grad(L_data) + grad(0.5 * L_phys))"]:::gradNode
        GradData --> JointGrad
        JointGrad -.-> Trunk
    end
```

### 8.2 Foundations: PINNs in Material Science vs. Fluid Dynamics
In classical computational physics (such as fluid dynamics or acoustics), Physics-Informed Neural Networks solve differential operators (e.g., Navier-Stokes, Burgers, or Helmholtz partial differential equations) by using autograd to compute spatial and temporal derivatives $\frac{\partial u}{\partial t}, \frac{\partial^2 u}{\partial x^2}$.

In **computational structural materials engineering**, the governing physics are not temporal differential equations, but **continuum fracture mechanics, constitutive stress-strain relationships, and empirical code envelopes**:
- Concrete failure under uniaxial compression occurs via crushing of the calcium-silicate-hydrate ($\text{C-S-H}$) gel matrix.
- Tensile capacity ($T_s$) is limited by microcrack opening at the aggregate-paste Interfacial Transition Zone (ITZ) and follows Griffith's fracture criterion.
- Flexural capacity ($F_s$, Modulus of Rupture) involves non-linear strain gradients across extreme tension fibers.
- International design standards (ACI 318-19, Eurocode 2, ASTM C39, ASTM C496, ASTM C78) establish strict, empirical ratio bounds between these failure modes based on thousands of historic laboratory fracture tests.

A standard deep neural network is oblivious to these physical couplings. Under incomplete training data, it routinely predicts compressive strengths of $30\text{ MPa}$ alongside tensile strengths of $35\text{ MPa}$—a physical impossibility ($T_s \ge C_s$). PIM-MTL's PINN framework eliminates this behavior by embedding constitutive bounds directly into the loss manifold.

### 8.3 Constitutive Envelopes & Governing Code Authorities

| Mechanical Property Ratio | Governing Building Codes | Physical Lower Bound ($R_{\min}$) | Physical Upper Bound ($R_{\max}$) | Physical Failure Mode & Rationale |
|---|---|---|---|---|
| **Tensile Ratio:**<br/>$R_t = \hat{T}_s / \hat{C}_s$ | ACI 318-19 §19.2.4<br/>Eurocode 2 §3.1.2<br/>ASTM C496 | **0.06** (6.0%) | **0.18** (18.0%) | Concrete has low tensile resistance. Pure tensile capacity rarely drops below 6% or exceeds 18% of cylinder compressive strength in standard structural mixes. |
| **Flexural Ratio:**<br/>$R_f = \hat{F}_s / \hat{C}_s$ | ACI 318-19 §19.2.3<br/>ASTM C78 / C293 | **0.08** (8.0%) | **0.25** (25.0%) | Modulus of rupture is higher than splitting tensile strength due to non-linear fiber stress redistribution, but is strictly bounded between 8% and 25% of compressive strength. |
| **Physical Impossibility Limit:**<br/>$\hat{T}_s < \hat{C}_s$ | Continuum Mechanics | **$T_s < C_s$** | **Strict Inequality** | Under uniaxial conditions, tensile splitting cannot exceed crushing capacity. A ratio $R_t \ge 1.0$ is physically impossible. |

### 8.4 Mathematical Derivation of PINN Restorative Gradients
To understand how the physics loss constrains the neural network, we derive the exact analytical gradients backpropagating from $\mathcal{L}_{\text{phys}}$ through the model's scaled outputs.

Let model scaled predictions be $\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \hat{y}_3]^T$ representing $[C_s, T_s, F_s]$.  
The denormalized physical values are:
$$\hat{C}_s = \hat{y}_1 s_1 + \mu_1, \quad \hat{T}_s = \hat{y}_2 s_2 + \mu_2, \quad \hat{F}_s = \hat{y}_3 s_3 + \mu_3$$
where $s_j$ and $\mu_j$ are the scale and center of task $j$.

The dimensionless ratios are:
$$R_t = \frac{\hat{T}_s}{\hat{C}_s}, \qquad R_f = \frac{\hat{F}_s}{\hat{C}_s}$$

The two-sided quadratic hinge penalty for a single specimen is:
$$\ell_{\text{phys}} = \text{ReLU}(R_t - 0.18)^2 + \text{ReLU}(0.06 - R_t)^2 + \text{ReLU}(R_f - 0.25)^2 + \text{ReLU}(0.08 - R_f)^2$$

#### 1. Restorative Gradient on the Tensile Head ($\hat{y}_2$):
Applying the chain rule:
$$\frac{\partial \ell_{\text{phys}}}{\partial \hat{y}_2} = \frac{\partial \ell_{\text{phys}}}{\partial R_t} \cdot \frac{\partial R_t}{\partial \hat{T}_s} \cdot \frac{\partial \hat{T}_s}{\partial \hat{y}_2}$$

Evaluating each partial derivative:
$$\frac{\partial \ell_{\text{phys}}}{\partial R_t} = 2 \cdot \text{ReLU}(R_t - 0.18) - 2 \cdot \text{ReLU}(0.06 - R_t)$$
$$\frac{\partial R_t}{\partial \hat{T}_s} = \frac{1}{\hat{C}_s}, \qquad \frac{\partial \hat{T}_s}{\partial \hat{y}_2} = s_2$$

Combining terms:
$$\frac{\partial \ell_{\text{phys}}}{\partial \hat{y}_2} = \left[ 2 \cdot \text{ReLU}(R_t - 0.18) - 2 \cdot \text{ReLU}(0.06 - R_t) \right] \cdot \frac{s_2}{\hat{C}_s}$$

- **Physical Dynamics:** If the model predicts an excessively high tensile strength ($R_t > 0.18$), the first term fires positively ($\frac{\partial \ell_{\text{phys}}}{\partial \hat{y}_2} > 0$). In gradient descent ($\hat{y}_2 \leftarrow \hat{y}_2 - \eta \nabla$), this **actively suppresses $\hat{y}_2$**, pulling the tensile prediction down into code compliance.
- Conversely, if $R_t < 0.06$, the gradient is negative, forcing the optimizer to **elevate $\hat{y}_2$**.

#### 2. Restorative Gradient on the Flexural Head ($\hat{y}_3$):
$$\frac{\partial \ell_{\text{phys}}}{\partial \hat{y}_3} = \left[ 2 \cdot \text{ReLU}(R_f - 0.25) - 2 \cdot \text{ReLU}(0.08 - R_f) \right] \cdot \frac{s_3}{\hat{C}_s}$$
This forces flexural predictions into the $[0.08, 0.25]$ envelope on every specimen, even when no experimental flexural label exists.

#### 3. Restorative Gradient on the Compressive Head ($\hat{y}_1$):
Because $\hat{C}_s$ appears in the denominator of both $R_t$ and $R_f$:
$$\frac{\partial R_t}{\partial \hat{C}_s} = -\frac{\hat{T}_s}{\hat{C}_s^2}, \qquad \frac{\partial R_f}{\partial \hat{C}_s} = -\frac{\hat{F}_s}{\hat{C}_s^2}$$

Applying the multivariate chain rule:
$$\frac{\partial \ell_{\text{phys}}}{\partial \hat{y}_1} = \left[ \frac{\partial \ell_{\text{phys}}}{\partial R_t} \left(-\frac{\hat{T}_s}{\hat{C}_s^2}\right) + \frac{\partial \ell_{\text{phys}}}{\partial R_f} \left(-\frac{\hat{F}_s}{\hat{C}_s^2}\right) \right] \cdot s_1$$

- **Physical Dynamics:** When tensile strength is excessively high ($R_t > 0.18$), $\frac{\partial \ell_{\text{phys}}}{\partial R_t} > 0$, making $\frac{\partial \ell_{\text{phys}}}{\partial \hat{y}_1}$ negative. Gradient descent **increases $\hat{C}_s$**, expanding the compressive capacity to restore equilibrium.

### 8.5 Dual-Pathway Gradient Comparison: Observed vs. Unobserved Targets

The interaction of empirical and physical gradients is summarized below:

```mermaid
graph TD
    subgraph ObservedEntry ["Case A: Ground Truth Label Observed (M = 1.0)"]
        y_obs["True Target y_ij exists"] --> L_data_obs["Masked Data Loss: (y_hat - y)^2"]
        L_data_obs --> G_emp["Empirical Gradient: grad_emp != 0"]
        PhysA["Physics Ratio Evaluator"] --> G_physA["Physics Gradient: grad_phys"]
        G_emp --> JointA["Total Gradient = grad_emp + 0.5 * grad_phys"]
    end

    subgraph UnobservedEntry ["Case B: Ground Truth Label Missing (M = 0.0)"]
        y_mis["No True Label (NaN in CSV)"] --> L_data_mis["Masked Loss: (y_hat - 0.0)^2 * 0.0 = 0.0"]
        L_data_mis --> G_zero["Empirical Gradient = STRICTLY 0.0<br/>(No pseudo-label corruption)"]
        PhysB["Physics Ratio Evaluator (Evaluated on ALL samples)"] --> G_physB["Physics Gradient: grad_phys != 0<br/>(Restores Rt and Rf to ACI bounds)"]
        G_zero --> JointB["Total Gradient = STRICTLY 0.5 * grad_phys"]
        G_physB --> JointB
    end
```

### 8.6 Quadratic Hinge ($\text{ReLU}^2$) vs. Linear Hinge and Hard Clamping
The choice of a quadratic hinge penalty ($\text{ReLU}(u)^2$) over linear hinge ($\text{ReLU}(u)$) or hard projection layers is mathematically justified:
1. **Vs. Hard Output Clamping:** If outputs were strictly clamped via `torch.clamp(y, min, max)`, the gradient outside the valid range would be zero ($\frac{d}{dx}\text{clamp} = 0$), freezing learning on all out-of-bound specimens. The PINN formulation maintains continuous non-zero gradients everywhere outside the envelope.
2. **Vs. Linear Hinge:** The derivative of linear hinge is a discontinuous Heaviside step function:
   $$\frac{d}{du}\text{ReLU}(u) = H(u) = \begin{cases} 1, & u > 0 \\ 0, & u < 0 \end{cases}$$
   This step discontinuity causes high-frequency oscillations and optimization instability at the boundary $R_t = 0.06$ or $0.18$.
3. **Quadratic Hinge Smoothness:**
   $$\frac{d}{du}\left[\text{ReLU}(u)^2\right] = 2 \cdot \text{ReLU}(u)$$
   The first derivative is **$C^0$ continuous everywhere and smoothly vanishes as $u \to 0$**, ensuring gentle corrections near code borders and strong, non-linear penalties for extreme physical violations.

### 8.7 Total Joint Composite Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \cdot \mathcal{L}_{\text{phys}}$$
where $\lambda_{\text{phys}} = 0.5$. This guarantees that the neural network prioritizes empirical accuracy while honoring continuum mechanics boundaries across the entire experimental domain.

---

## 9. Training Pipeline and Optimization

### 9.1 Training Hyperparameters

| Hyperparameter | Configured Value | Selection Rationale | Source Code Location |
|---|---|---|---|
| **Optimizer** | `Adam` | Adaptive moment estimation handles tabular gradient variances | `main.py:121` |
| **Initial Learning Rate ($\eta_0$)** | $1.0 \times 10^{-3}$ | Optimal convergence without early divergence | `main.py:121` |
| **Weight Decay ($L_2$)** | $1.0 \times 10^{-4}$ | Prevents weight explosion in high-dimensional linear layers | `main.py:121` |
| **LR Scheduler** | `CosineAnnealingLR` | Smoothly anneals learning rate to prevent late-stage oscillations | `main.py:122` |
| **Minimum Learning Rate ($\eta_{\min}$)** | $1.0 \times 10^{-5}$ | Preserves micro-tuning capacity in final epochs | `main.py:122` |
| **Batch Size ($B$)** | 64 | Balances gradient noise and GPU vectorization throughput | `main.py:109` |
| **Epochs** | 60 | Sufficient for convergence of empirical and physical loss terms | `main.py:109` |
| **Gradient Clipping ($L_2$ Max Norm)** | 5.0 | Restricts gradient explosions during strength ratio backprop | `main.py:143` |
| **Physics Loss Weight ($\lambda_{\text{phys}}$)** | 0.5 | Balances empirical fidelity against code constraint compliance | `main.py:116` |
| **Task Weights ($\mathbf{w}$)** | $[1.0, 1.2, 1.5]$ | Compensates for label scarcity ($C_s$: 100%, $T_s$: 78%, $F_s$: 40%) | `utils/losses.py:51` |
| **Random Seed** | 42 | Guarantees deterministic reproducibility | `main.py:27` |

### 9.2 Complete Algorithmic Pseudocode

```text
Algorithm 1: Physics-Informed Masked Multi-Task Training (PIM-MTL)
================================================================================
Input  : Dataset D = {(x_i, y_i)}_i=1^N, Epochs E=60, Batch size B=64, LR eta=1e-3, Weight lambda_phys=0.5
Output : Trained model weights Theta*, Scaler preprocessor P*

1.  SetDeterministicSeeds(seed=42)
2.  (D_train, D_test) <- TrainTestSplit(D, test_ratio=0.2, seed=42)
3.  Fit OneHotEncoder OHE on D_train[Categorical_Features]
4.  Fit RobustScaler S_x on D_train[Numerical_Features]
5.  X_train <- Concatenate(S_x.transform(D_train), OHE.transform(D_train))
6.  X_test  <- Concatenate(S_x.transform(D_test),  OHE.transform(D_test))
7.  M_train <- ~IsNaN(D_train[Targets]), M_test <- ~IsNaN(D_test[Targets])
8.  For each task j in {Cs, Ts, Fs}:
9.      Fit RobustScaler S_y[j] on D_train[Targets][M_train[:, j] == 1, j]
10.     Y_train_scaled[:, j] <- S_y[j].transform(D_train) where M_train[:, j] == 1 else 0.0
11. Initialize model Theta <- PIM_MTL_Net(input_dim=19) via Kaiming Normal
12. Initialize optimizer Adam(Theta, lr=1e-3, weight_decay=1e-4)
13. Initialize scheduler CosineAnnealingLR(T_max=E, eta_min=1e-5)
14. For epoch = 1 to E do:
15.     Shuffle D_train and partition into mini-batches {(bx, by, bm)} of size B
16.     For each mini-batch (bx, by, bm) do:
17.         ZeroGradients(optimizer)
18.         y_hat <- Theta.Forward(bx)                             // Forward Pass
19.         
20.         // 1. Masked Empirical Loss
21.         E_sq <- bm * (y_hat - by)^2
22.         N_obs <- ClampMin(Sum(bm, dim=0), min=1.0)
23.         L_data <- Sum(TaskWeights * (Sum(E_sq, dim=0) / N_obs))
24.         
25.         // 2. Unmasked Physics Ratio Loss
26.         y_orig <- y_hat * target_scales + target_centers       // Denormalize to MPa
27.         Cs <- ClampMin(y_orig[:, 0], 1.0)
28.         Ts <- ClampMin(y_orig[:, 1], 0.0)
29.         Fs <- ClampMin(y_orig[:, 2], 0.0)
30.         Rt <- Ts / Cs;  Rf <- Fs / Cs
31.         L_phys <- Mean(ReLU(Rt - 0.18)^2 + ReLU(0.06 - Rt)^2 + ReLU(Rf - 0.25)^2 + ReLU(0.08 - Rf)^2)
32.         
33.         // 3. Composite Step
34.         L_total <- L_data + lambda_phys * L_phys
35.         L_total.Backward()                                     // Autograd
36.         ClipGradients(Theta.parameters(), max_norm=5.0)
37.         optimizer.Step()
38.     End For
39.     scheduler.Step()
40. End For
41. EvaluateOnTest(Theta, X_test, D_test[Targets], M_test)
42. SaveCheckpoint("models/pim_mtl_model.pt", Theta)
43. SavePreprocessor("models/preprocessor.joblib", P)
44. Return Theta*, P*
================================================================================
```

---

## 10. Inference and Prediction Workflow

### 10.1 Standalone Inference Implementation
A complete, dependency-isolated inference pipeline is implemented directly through the exported model class in `utils/models.py` and `models/preprocessor.joblib`:

```python
import joblib
import numpy as np
import torch
from utils.models import PIM_MTL_Net

# 1. Load Serialized Architecture and Trained Weights
model = PIM_MTL_Net.load("models/pim_mtl_model.pt", map_location="cpu")
model.eval()

# 2. Load Fitted Preprocessor Metadata
preprocessor = joblib.load("models/preprocessor.joblib")
ohe = preprocessor["ohe"]
scaler_x = preprocessor["scaler_x"]
target_centers = preprocessor["target_centers"]
target_scales = preprocessor["target_scales"]

# 3. Prepare Raw Input Features (e.g. single sample or batch)
# Schema: [Type_of_cement, Brand, Additives, Type_of_aggregates]
raw_cat = np.array([["CPC 30R", "CEMEX", "None", "Crushed"]])
# Schema: [Design_Fc_Mpa, Curing_age_days, Er_ohm-cm, UPV_m/s]
raw_num = np.array([[30.0, 28.0, 12.5, 4200.0]])

# 4. Feature Transformations (Leakage-Safe)
cat_encoded = ohe.transform(raw_cat)
num_scaled = scaler_x.transform(raw_num)
X_input = np.hstack([num_scaled, cat_encoded]).astype(np.float32)

# 5. Model Forward Pass
with torch.no_grad():
    x_tensor = torch.tensor(X_input, dtype=torch.float32)
    preds_scaled = model(x_tensor).numpy()

# 6. Physical Unit Denormalization (MPa)
preds_mpa = preds_scaled * target_scales + target_centers
cs_pred, ts_pred, fs_pred = preds_mpa[0, 0], preds_mpa[0, 1], preds_mpa[0, 2]

print(f"Predicted Compressive Strength (Cs) : {cs_pred:.2f} MPa")
print(f"Predicted Splitting Tensile (Ts)    : {ts_pred:.2f} MPa (Ratio: {ts_pred/cs_pred:.3f})")
print(f"Predicted Flexural Strength (Fs)    : {fs_pred:.2f} MPa (Ratio: {fs_pred/cs_pred:.3f})")
```

### 10.2 Production Deployment Boundaries
- **Supported Formats:** Direct NumPy array and PyTorch tensor inputs.
- **Inference Latency:** Sub-millisecond execution per batch on CPU; suitable for on-site mobile testing devices and edge laboratory hardware.
- **Missing Inputs at Inference:** If non-destructive testing features (`UPV_m/s` or `Er_ohm-cm`) are unmeasured on-site, the current model requires proxy inputs. (In 2nd-semester exploration, tree-based models were trained to virtually sense these proxies from mix designs).
- **Service Deployment:** There is currently no active FastAPI, Flask, or gRPC endpoint in the repository; inference is performed via script or notebook execution.

---

## 11. Evaluation and Results Architecture

Model evaluation measures both **statistical predictive accuracy** on observed labels and **physical constraint compliance** across all test specimens.

### 11.1 Mathematical Metric Formulations

#### 1. Coefficient of Determination ($R^2$ Score)
Evaluated on observed entries ($m_{ij} = 1.0$) for task $j$:
$$R_j^2 = 1 - \frac{\sum_{i=1}^N m_{ij} (y_{ij} - \hat{y}_{ij})^2}{\sum_{i=1}^N m_{ij} (y_{ij} - \bar{y}_j)^2}, \quad \bar{y}_j = \frac{1}{\sum m_{ij}} \sum_{i=1}^N m_{ij} y_{ij}$$

#### 2. Root Mean Squared Error (RMSE, MPa)
$$\text{RMSE}_j = \sqrt{\frac{1}{\sum_{i=1}^N m_{ij}} \sum_{i=1}^N m_{ij} (y_{ij} - \hat{y}_{ij})^2}$$

#### 3. Mean Absolute Error (MAE, MPa)
$$\text{MAE}_j = \frac{1}{\sum_{i=1}^N m_{ij}} \sum_{i=1}^N m_{ij} |y_{ij} - \hat{y}_{ij}|$$

#### 4. Physical Feasibility Violation Rates (PVR)
Evaluated over **all $N_{\text{test}} = 884$ test specimens**, irrespective of target observation status:
$$\text{PVR}_{\text{ts}} = \frac{100}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\left( \frac{\hat{T}_{s,i}}{\hat{C}_{s,i}} < 0.06 \lor \frac{\hat{T}_{s,i}}{\hat{C}_{s,i}} > 0.18 \right)$$
$$\text{PVR}_{\text{fs}} = \frac{100}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\left( \frac{\hat{F}_{s,i}}{\hat{C}_{s,i}} < 0.08 \lor \frac{\hat{F}_{s,i}}{\hat{C}_{s,i}} > 0.25 \right)$$
$$\text{PVR}_{\text{imp}} = \frac{100}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\left( \hat{T}_{s,i} \ge \hat{C}_{s,i} \right)$$

### 11.2 Verified Experimental Results
Extracted directly from `results.txt` and `final_evaluation_results.csv` on 884 held-out test specimens:

#### Quantitative Predictive Accuracy

| Target Property | Symbol | Observed $N$ | Test Coverage | $R^2$ Score | RMSE (MPa) | MAE (MPa) | Mean Target (MPa) |
|---|---|---|---|---|---|---|---|
| **Compressive Strength** | $C_s$ | 884 / 884 | 100.0% | **0.9818** | 1.3143 | 0.9931 | 31.19 |
| **Splitting Tensile Strength** | $T_s$ | 696 / 884 | 78.7% | **0.9857** | 0.1097 | 0.0820 | 2.48 |
| **Flexural Strength** | $F_s$ | 363 / 884 | 41.1% | **0.9972** | 0.1262 | 0.0990 | 3.42 |

*(Note: Minor run variations across random seeds yield $R^2$ in [0.981, 0.998] with RMSE consistently $\approx 1.3\text{ MPa}$ for $C_s$ and $\approx 0.1\text{ MPa}$ for $T_s$ and $F_s$, demonstrating robustness).*

#### Physical Feasibility and Code Compliance Audit

| Constraint Audit Check | Permitted Physical Envelope | Observed Violation Rate | Physical Compliance Rate |
|---|---|---|---|
| **Tensile Ratio Compliance ($R_t$)** | $0.06 \le T_s / C_s \le 0.18$ | **3.51%** | **96.49%** |
| **Flexural Ratio Compliance ($R_f$)** | $0.08 \le F_s / C_s \le 0.25$ | **17.99%** | **82.01%** |
| **Physically Impossible Predictions** | $T_s \ge C_s$ (Tensile $\ge$ Compressive) | **0.00%** | **100.00%** |

### 11.3 Figure Catalog & Visualization Architecture
The `figures/` directory houses 13 visual artifacts generated during pipeline evaluation:
1. `01_target_missingness_profile.png`: Stacked bar chart illustrating observation counts ($C_s$: 100%, $T_s$: 78.3%, $F_s$: 39.8%).
2. `02_continuous_features_distribution.png`: Kernel density estimates and boxplots of the 4 continuous input variables.
3. `03_categorical_features_distribution.png`: Frequency distribution of cement types, aggregate types, brands, and additives.
4. `04_target_distributions.png`: Probability density plots of the ground truth mechanical strength targets.
5. `05_ground_truth_mechanical_ratios.png`: Histograms of observed $R_t$ and $R_f$ with overlaid ACI code limit lines.
6. `06_correlation_matrix.png`: Heatmap showing Pearson correlations across continuous inputs and targets.
7. `07_feature_target_relationships.png`: Bivariate scatter plots showing non-linear relationships between UPV, age, and strengths.
8. `08_training_convergence_curves.png`: Three-panel plot tracking $\mathcal{L}_{\text{total}}$, $\mathcal{L}_{\text{data}}$, and $\mathcal{L}_{\text{phys}}$ across 60 epochs.
9. `09_parity_plots_pred_vs_actual.png`: Scatter plots of predicted vs. true values against the 1:1 identity line for $C_s, T_s, F_s$.
10. `10_residual_error_distributions.png`: Gaussian-centered error histograms validating absence of systematic bias.
11. `11_ratio_comparison_gt_vs_pred.png`: Side-by-side violin plots comparing ground truth strength ratios against predicted ratios.
12. `12_model_performance_comparison.png`: Bar chart contrasting PIM-MTL against single-task and unconstrained baselines.
13. `13_physical_compliance_audit.png`: Donut chart illustrating the 100% elimination of physically impossible predictions.

---

## 12. Module Dependencies and Component Interactions

### 12.1 Internal Module Dependency Graph

```mermaid
graph TD
    Main["main.py (CLI / Execution Runner)"]
    Models["utils/models.py (PIM_MTL_Net, SafeBatchNorm1d)"]
    Losses["utils/losses.py (PIM_loss, PIM_MTL_Loss)"]
    Init["utils/__init__.py (Namespace Exporter)"]
    Data["Data.csv"]
    Checkpoint["models/pim_mtl_model.pt"]
    Preproc["models/preprocessor.joblib"]
    Results["results.txt"]

    Init --> Models
    Init --> Losses

    Main --> Init
    Main --> Data
    Main --> Models
    Main --> Losses
    Main --> Checkpoint
    Main --> Preproc
    Main --> Results
```

### 12.2 External Library Dependency Graph
Extracted directly from `requirements.txt`:
- **Deep Learning Framework:** `torch>=2.0.0` (PyTorch core autograd and tensor computations).
- **Scientific Computing & Tabular Data:**
  - `numpy>=1.24.0` (Array manipulations and numerical operations).
  - `pandas>=2.0.0` (DataFrame loading and categorical manipulation).
  - `scipy>=1.10.0` (Scientific calculations).
  - `scikit-learn>=1.2.0` (`RobustScaler`, `OneHotEncoder`, `train_test_split`, $R^2$, RMSE, MAE).
  - `joblib>=1.2.0` (Transformer serialization).
- **Visualization:**
  - `matplotlib>=3.7.0` (Plot generation and figure rendering).
  - `seaborn>=0.12.0` (Statistical data visualization).
- **Interactive Notebook Support:**
  - `jupyter>=1.0.0`
  - `ipykernel>=6.20.0`

### 12.3 Potential Circular Dependencies and Coupling Analysis
- **Coupling State:** Loose coupling. `utils/models.py` has zero dependency on `utils/losses.py` or `main.py`. `utils/losses.py` has zero dependency on `utils/models.py`. Both can be imported and utilized in external testing harnesses independently.
- **Circular Dependencies:** None exist. The dependency tree is strictly acyclic.

---

## 13. Artifacts and Storage Architecture

| Artifact Path | Format | Type | Generation Mechanism | Required to Reproduce Run? | Contents & Role |
|---|---|---|---|---|---|
| `Data.csv` | CSV | Input Data | Static experimental dataset | **Yes** | 4,420 experimental records; source of all training/testing data. |
| `requirements.txt` | Plaintext | Environment | Manually maintained | **Yes** | Python package dependencies and version constraints. |
| `models/pim_mtl_model.pt` | PyTorch Binary | Output Checkpoint | `model.save()` in `main.py` | No (Can be retrained) | Model `state_dict`, input dimension (19), and hidden dimensions. |
| `models/preprocessor.joblib` | Joblib Pickle | Output Preprocessor | `joblib.dump()` in `main.py` | No (Can be refitted) | Fitted `OneHotEncoder`, feature `RobustScaler`, target centers/scales. |
| `results.txt` | Plaintext | Metrics Output | `evaluate_model()` in `main.py` | No (Regenerated) | Human-readable quantitative metrics and physical audit table. |
| `final_evaluation_results.csv` | CSV | Metrics Output | `training_and_analysis.ipynb` | No (Regenerated) | Structured test accuracy metrics ($R^2$, RMSE, MAE). |
| `final_test_predictions.csv` | CSV | Predictions Output | `training_and_analysis.ipynb` | No (Regenerated) | Per-specimen predictions, residuals, ratios, and violation flags. |
| `figures/*.png` | PNG Images | Visual Artifacts | `training_and_analysis.ipynb` | No (Regenerated) | 13 analytical plots depicting EDA, loss curves, and parity plots. |
| `paper/main.tex` | LaTeX | Academic Report | Academic authoring | No | Full IEEE conference paper manuscript and equations. |

---

## 14. Environment and Execution Instructions

### 14.1 Prerequisites
- **Python Version:** Python 3.10, 3.11, or 3.12 (Verified on Python 3.12).
- **Hardware Requirements:** CPU (x86_64 or ARM64) with $\ge 4\text{ GB}$ RAM. NVIDIA GPU supporting CUDA 11.8+ or 12.0+ is optional but recommended for accelerated training.

### 14.2 Environment Setup
Navigate to the project root and create a virtual environment:

```powershell
# Navigate to the project root
cd C:\Users\jeetu\projects\Capstone\3rdYearsem5

# Create a clean virtual environment
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 14.3 Execution Commands

#### Option A: Central Pipeline Execution (Command-Line)
Run the full training, evaluation, and artifact generation pipeline:
```powershell
python main.py --data Data.csv --epochs 60 --batch-size 64 --output models/pim_mtl_model.pt --results-file results.txt
```

#### Option B: Master Interactive Jupyter Notebook
Launch the analytical notebook to interactively reproduce all exploratory data analyses, training convergence curves, parity plots, and residual error figures:
```powershell
jupyter notebook notebooks/training_and_analysis.ipynb
```
Select **Kernel $\to$ Restart and Run All**.

#### Option C: Regenerate the Master Notebook Programmatically
If `training_and_analysis.ipynb` needs to be rebuilt from source code:
```powershell
python generate_notebook.py
```

---

## 15. Architectural Evolution and Future Roadmaps

### 15.1 Academic Evolution: 2nd Semester vs. 3rd Year Semester 5

```mermaid
graph TD
    subgraph Semester2 ["2nd Semester Approach (Chained Stacking - Obsolete)"]
        A1[Mix Ingredients] --> B1[Stage 1: Tree Virtual Sensors<br/>ExtraTrees / Random Forest]
        B1 --> C1[Predicted UPV & Er Proxies<br/>High Compounding Noise Cascade]
        C1 --> D1[Stage 2: Multi-Task ANN<br/>Ad-Hoc Penalty Functions]
        D1 --> E1[Stage 3: Split-Conformal Calibration]
        note1[Severe Flaw: Discarded 60.2% of data;<br/>17.19% Physical Constraint Violations]
    end

    subgraph Semester5 ["3rd Year Sem 5 Approach (PIM-MTL - Current Architecture)"]
        A2[Raw Mix Design + Measured NDT] --> B2[RobustScaler + OneHotEncoder]
        B2 --> C2[Shared 3-Layer Dense Trunk<br/>SafeBatchNorm1d + GELU + Dropout]
        C2 --> D2[3 Parallel Task Heads]
        D2 --> E2[Masked Gradient Gating<br/>Zero Gradient for Missing Labels<br/>Retains 100% of Data (4,420 specimens)]
        D2 --> F2[Differentiable ACI Continuum Physics<br/>Two-Sided Quadratic Hinge Regularizer]
        note2[Breakthrough: 0.00% Impossible Predictions;<br/>R^2 > 0.98 Across All 3 Mechanical Properties]
    end
```

#### Why the 2nd Semester Architecture Was Abandoned
1. **Compounding Noise Cascade:** Stage 1 predicted electrical resistivity and UPV from mix features. Feeding synthetic, noisy proxy predictions into Stage 2 amplified errors exponentially.
2. **Catastrophic Data Loss:** Incomplete specimens were deleted via listwise deletion, forfeiting **60.2% of the dataset (2,660 specimens)**.
3. **High Physical Violation Rate:** The heuristic loss penalties in 2nd semester produced a **17.19% physical violation rate**, including physically impossible instances where predicted tensile strength exceeded compressive strength ($T_s \ge C_s$). PIM-MTL reduces this to **0.00%**.

### 15.2 Future Engineering Roadmap

#### 1. Multi-Task Cross-Attention Mechanism
Replace the static linear feature trunk with dynamic cross-task attention routing:
$$\text{Attention}(\mathbf{Q}_k, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}_k \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
where $\mathbf{Q}_k$ is a learned query vector representing task $k \in \{C_s, T_s, F_s\}$. This allows the flexural head to dynamically query aggregate mineralogy and additive features while the compressive head attends to density and curing age.

#### 2. Multi-Objective Bayesian Hyperparameter Tuning
Automate the tuning of $\lambda_{\text{phys}}$, task weights $\mathbf{w}$, and learning rates using **Tree-structured Parzen Estimators (Optuna)** over a multi-objective Pareto frontier:
$$\max_{\boldsymbol{\theta}} \mathcal{J}(\boldsymbol{\theta}) = \overline{R^2}(\boldsymbol{\theta}) - \gamma_1 \cdot \text{PVR}_{\text{ts}}(\boldsymbol{\theta}) - \gamma_2 \cdot \text{PVR}_{\text{fs}}(\boldsymbol{\theta})$$

#### 3. Conformalized Masked Uncertainty Intervals
Re-introduce conformal prediction calibrated specifically for masked multi-task outputs, generating guaranteed prediction intervals $[y_j^{\text{lower}}, y_j^{\text{upper}}]$ at a $90\%$ coverage level $(1 - \alpha = 0.90)$.

#### 4. ONNX Runtime & Edge Microservice Deployment
Export `pim_mtl_model.pt` to the Open Neural Network Exchange (ONNX) format, enabling low-latency inference on embedded inspection hardware and integration into a containerized FastAPI Docker container for ready-mix batching plants.

---

## 16. Architectural Verification and Quality Assurance Checklist

- [x] **File Existence:** Code inspected and verified across `main.py`, `utils/models.py`, `utils/losses.py`, `Data.csv`, `results.txt`.
- [x] **Mathematical Integrity:** Verified that $\frac{\partial \mathcal{L}_{\text{data}}}{\partial \hat{y}_{ij}} \equiv 0$ when $m_{ij} = 0$.
- [x] **Parameter Count Audit:** Verified exact trainable parameter count ($13,379$).
- [x] **Empirical Metrics Audit:** Verified $R^2 = 0.9818$ ($C_s$), $0.9857$ ($T_s$), $0.9972$ ($F_s$) from `results.txt`.
- [x] **Physical Feasibility Audit:** Verified $0.00\%$ impossible predictions ($T_s \ge C_s$).
- [x] **Leakage Prevention:** Verified that preprocessor transformations fit exclusively on the 80% training split.
- [x] **Code Preserved:** No application code, checkpoints, or results modified.
