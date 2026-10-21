# Physics-Informed Masked Multi-Task Learning (PIM-MTL)

## Technical Monograph, Architecture Specification & Future Roadmaps

**Scope:** Exclusively covers the **3rdYearsem5** implementation  
**Repository Path:** `c:\Users\jeetu\projects\Capstone\3rdYearsem5`  
**Dataset:** ConcreteXAI (4,420 Multivariate Experimental Mixes)  
**Primary Artifacts:** `Data.csv`, `main.py`, `results.txt`, `utils/models.py`, `utils/losses.py`, `models/pim_mtl_model.pt`

---

## 1. Project Overview & Problem Definition

In structural concrete design (ASTM C39, ASTM C496, ASTM C78; ACI 318, Eurocode 2), concrete mechanical capacity is governed by three primary strength parameters:

1. **Compressive Strength ($C_s$, MPa):** Uniaxial resistance to compressive crushing.
2. **Splitting Tensile Strength ($T_s$, MPa):** Indirect diametral tensile capacity governing crack initiation and shear.
3. **Flexural Strength / Modulus of Rupture ($F_s$, MPa):** Extreme fiber bending resistance governing deflection and slab capacity.

### The Asymmetric Missingness Problem

While compressive testing is universal, tensile and flexural testing are destructive, expensive, and require specialized testing apparatuses. Consequently, the **ConcreteXAI** dataset exhibits severe asymmetric missingness:

| Target Property | Column Name | Total Samples | Observed | Missing | Missingness (%) | Mean $\pm$ Std (MPa) |
|---|---|---:|---:|---:|---:|---|
| **Compressive Strength ($C_s$)** | `Cs_Mpa` | 4,420 | 4,420 | 0 | **0.0%** | $31.19 \pm 7.91$ |
| **Splitting Tensile Strength ($T_s$)** | `Ts_Mpa` | 4,420 | 3,460 | 960 | **21.7%** | $2.48 \pm 0.65$ |
| **Flexural Strength ($F_s$)** | `Fs_Mpa` | 4,420 | 1,760 | 2,660 | **60.2%** | $3.42 \pm 1.34$ |
| **Complete Triplet (All 3 Present)** | - | 4,420 | 1,760 | 2,660 | **60.2%** | - |

**PIM-MTL** solves this by uniting **Masked Gradient Gating** (training on all 4,420 specimens with strictly zero gradient for missing targets) with **Differentiable Physics Regularization** (enforcing ACI 318 continuum mechanics bounds on every specimen).

---

## 2. Missing Data Handling: Before Training vs. During Training

A critical question in multi-task learning under asymmetric missingness is: **Is the missing data added before training or during training?**

> [!IMPORTANT]
> **Definitive Answer:**  
> Missing values are **NEVER imputed or synthesized with fake values** at any stage.
> - **Before Training (Data Preparation Phase):** A binary observation mask tensor is constructed, and missing cells are assigned a neutral computational placeholder ($0.0$) purely to satisfy PyTorch tensor dimension requirements.
> - **During Training (Batch Execution Phase):** The mask is actively multiplied into the empirical loss at every batch step, dynamically gating gradients so that missing values produce **strictly zero backpropagation gradient**.

```text
[ Raw CSV: contains NaNs for Ts & Fs ]
                   │
                   ▼ (BEFORE TRAINING: In load_and_preprocess)
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Extract Binary Mask Tensor:                                         │
│    M[i, j] = 1.0 if y[i, j] is valid, 0.0 if y[i, j] is NaN            │
│ 2. Create Scaled Target Tensor:                                        │
│    y_scaled[i, j] = RobustScaler(y[i, j]) if observed, else 0.0        │
│    (0.0 is ONLY a matrix placeholder; it is NOT an imputed value)      │
└────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼ (DURING TRAINING: Inside PIM_loss at every mini-batch)
┌────────────────────────────────────────────────────────────────────────┐
│ Forward Pass -> Preds y_hat (N, 3)                                     │
│                                                                        │
│ Empirical Loss:                                                        │
│ Masked_Errors = M * (y_hat - y_scaled)^2                               │
│ For missing target: M[i, j] = 0.0  ===>  Error * 0.0 = 0.0             │
│ Gradient: d(L_data) / d(y_hat[i, j]) = STRICTLY 0.0                    │
│                                                                        │
│ Physics Loss (UNMASKED on ALL samples):                                │
│ Differentiably converts y_hat to MPa and evaluates ACI ratio bounds.  │
│ Regularizes unobserved Ts and Fs heads using continuum mechanics.      │
└────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Proof of Gradient Gating

For target $j \in \{1, 2, 3\}$ and specimen $i$:

$$
\mathcal{L}_{\text{data}} = \sum_{j=1}^3 w_j \left[ \frac{\sum_{i=1}^N m_{ij} (y_{ij} - \hat{y}_{ij})^2}{\max\left(1, \sum_{i=1}^N m_{ij}\right)} \right]
$$

Differentiating with respect to the prediction $\hat{y}_{ij}$:

$$
\frac{\partial \mathcal{L}_{\text{data}}}{\partial \hat{y}_{ij}} = \frac{-2 w_j m_{ij} (y_{ij} - \hat{y}_{ij})}{\max(1, \sum_{k=1}^N m_{kj})}
$$

- **If observed ($m_{ij} = 1.0$):** Normal backpropagation gradient.
- **If missing ($m_{ij} = 0.0$):**

$$
\frac{\partial \mathcal{L}_{\text{data}}}{\partial \hat{y}_{ij}} = \frac{-2 w_j \cdot \mathbf{0} \cdot (0.0 - \hat{y}_{ij})}{\max(1, \sum_{k=1}^N m_{kj})} \equiv \mathbf{0}
$$

The neutral placeholder ($0.0$) has zero influence on the computational graph.

---

## 3. Preprocessing & Input Transformation Pipeline

All preprocessing is fitted strictly on the 80% training split (3,536 samples) to prevent data leakage.

### 3.1 Feature Space (19 Dimensions Total)

1. **Categorical Features (4 inputs $\rightarrow$ 15 one-hot binary dimensions):**
   - ``Type_of_cement` (4 levels: CPC 30R, CPO 30R RS BRA, CPC 30R RS, CPC 40R)
   - `Brand` (2 levels: CEMEX, Apasco)
   - `Additives` (5 levels: None, Opuntia biopolymer, Slag, Fluidizers)
   - ``Type_of_aggregates` (4 levels: Crushed, Rounded gravel, Recycled, Volcanic)

$$
\mathbf{x}_{\text{cat}} = \text{OneHotEncoder}(\text{sparse}=\text{False}, \text{handle\_unknown}=\text{"ignore"})
$$

2. **Continuous Features (4 inputs $\rightarrow$ 4 robust scaled dimensions):**
   - `Design_Fc_Mpa` ($20 - 35\text{ MPa}$)
   - `Curing_age_days` ($7 - 120\text{ days}$)
   - `Er_ohm-cm` ($1.43 - 23.95\ \Omega\cdot\text{cm}$)
   - `UPV_m/s` ($2,431 - 5,319\text{ m/s}$)

Scaled using median and Interquartile Range (IQR):

$$
\mathbf{x}_{\text{num}, i}^{(k)} = \frac{\mathbf{x}_{\text{num}, i}^{(k)} - \text{median}(\mathbf{X}_{\text{train}}^{(k)})}{\text{IQR}(\mathbf{X}_{\text{train}}^{(k)})}, \quad \text{IQR} = Q_3 - Q_1
$$

*Reasoning:* Bulk electrical resistivity (`Er_ohm-cm`) exhibits heavy right-tail skewness from non-linear pore water percolation. RobustScaler eliminates outlier distortion without squashing variance.

---

## 4. Deep Multi-Task Architecture Specification

The network architecture is implemented in `utils/models.py`:

```text
Input Features x in R^19
        │
        ▼
Linear(19, 128) -> SafeBatchNorm1d(128) -> GELU -> Dropout(p=0.2)
        │
        ▼
Linear(128, 64) -> SafeBatchNorm1d(64)  -> GELU
        │
        ▼
Linear(64, 32)   === [ Shared Latent Representation h_3 in R^32 ]
        │
  ┌─────┴─────────────────────┬─────────────────────┐
  │                           │                     │
  ▼                           ▼                     ▼
Linear(32, 1)               Linear(32, 1)         Linear(32, 1)
Cs Head (Compressive)       Ts Head (Tensile)     Fs Head (Flexural)
  │                           │                     │
  ▼                           ▼                     ▼
 y_hat_1                     y_hat_2               y_hat_3
```

### 4.1 Layer-by-Layer Mathematical Operations

1. **Layer 1 (Shared Encoder):**

$$
\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 19}, \mathbf{b}_1 \in \mathbb{R}^{128}
$$

$$
\mathbf{h}_1 = \text{Dropout}_{0.2}\Big( \text{GELU}\big( \text{SafeBatchNorm1d}(\mathbf{z}_1) \big) \Big)
$$

2. **Layer 2 (Shared Encoder):**

$$
\mathbf{z}_2 = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}, \mathbf{b}_2 \in \mathbb{R}^{64}
$$

$$
\mathbf{h}_2 = \text{GELU}\big( \text{SafeBatchNorm1d}(\mathbf{z}_2) \big)
$$

3. **Layer 3 (Trunk Compression):**

$$
\mathbf{h}_3 = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3, \quad \mathbf{W}_3 \in \mathbb{R}^{32 \times 64}, \mathbf{b}_3 \in \mathbb{R}^{32}
$$

4. **Task-Specific Prediction Heads:**

$$
\hat{y}_1 = \mathbf{w}_{cs}^T \mathbf{h}_3 + b_{cs}, \quad \mathbf{w}_{cs} \in \mathbb{R}^{32}, b_{cs} \in \mathbb{R} \quad (\text{Compressive})
$$

$$
\hat{y}_2 = \mathbf{w}_{ts}^T \mathbf{h}_3 + b_{ts}, \quad \mathbf{w}_{ts} \in \mathbb{R}^{32}, b_{ts} \in \mathbb{R} \quad (\text{Tensile})
$$

$$
\hat{y}_3 = \mathbf{w}_{fs}^T \mathbf{h}_3 + b_{fs}, \quad \mathbf{w}_{fs} \in \mathbb{R}^{32}, b_{fs} \in \mathbb{R} \quad (\text{Flexural})
$$

$$
\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \hat{y}_3]^T \in \mathbb{R}^3
$$

### 4.2 Parameter Count Audit

- **Layer 1:** $(19 \times 128) + 128 + (2 \times 128 \text{ BN}) = 2,432 + 128 + 256 = 2,816$
- **Layer 2:** $(128 \times 64) + 64 + (2 \times 64 \text{ BN}) = 8,192 + 64 + 128 = 8,384$
- **Layer 3:** $(64 \times 32) + 32 = 2,048 + 32 = 2,080$
- **Task Heads:** $3 \times (32 \times 1 + 1) = 99$
- **Total Trainable Parameters:** **13,379 parameters**

### 4.3 Key Components

- **GELU Non-Linearity:** $\text{GELU}(x) = x \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$. Smooth curvature with non-zero gradients for negative values avoids dying neurons.
- **SafeBatchNorm1d:** Fallback to running statistics when batch size $N=1$, preventing division by zero during single-mix on-site inference.
- **Kaiming Normal Initialization:** $\mathbf{W} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right)$, stabilizing variance propagation across layers.

---

## 5. Loss Formulation & Physics Integration

### 5.1 Differentiable Denormalization & Clamping

Inside the PyTorch graph, predictions are mapped back to real MPa space:

$$
\hat{\mathbf{y}}_{\text{MPa}} = \hat{\mathbf{y}}_{\text{scaled}} \odot \mathbf{s} + \boldsymbol{\mu}
$$

$$
\hat{C}_s = \max(\hat{y}_{1, \text{MPa}}, 1.0), \quad \hat{T}_s = \max(\hat{y}_{2, \text{MPa}}, 0.0), \quad \hat{F}_s = \max(\hat{y}_{3, \text{MPa}}, 0.0)
$$

- Clamping $\hat{C}_s \ge 1.0\text{ MPa}$ avoids explosive division-by-zero singularities during early epochs.

### 5.2 Civil Engineering Bounds & Physics Regularization ($\mathcal{L}_{\text{phys}}$)

Dimensionless strength ratios are defined as:

$$
R_t = \frac{\hat{T}_s}{\hat{C}_s}, \qquad R_f = \frac{\hat{F}_s}{\hat{C}_s}
$$

Engineering boundaries derived from ACI 318 and ASTM C496/C78:

- **Tensile Ratio Bound:** $R_t \in [0.06, 0.18]$ (Splitting tensile $f_{ct} \approx 0.56\sqrt{f'_c}$)
- **Flexural Ratio Bound:** $R_f \in [0.08, 0.25]$ (Modulus of rupture $f_r \approx 0.70\sqrt{f'_c}$; $F_s \approx 1.4 T_s$)

Enforced via two-sided quadratic hinge penalties on **all $N$ specimens in the batch**:

$$
\mathcal{L}_{\text{phys}} = \frac{1}{N} \sum_{i=1}^N \Big[ \text{ReLU}(R_{t,i} - 0.18)^2 + \text{ReLU}(0.06 - R_{t,i})^2 + \text{ReLU}(R_{f,i} - 0.25)^2 + \text{ReLU}(0.08 - R_{f,i})^2 \Big]
$$

### 5.3 Joint Multi-Objective Function

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}}, \quad \lambda_{\text{phys}} = 0.5
$$

---

## 6. Training Dynamics & Results Summary (`results.txt`)

### 6.1 Training Hyperparameters

- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$)
- **Batch Size:** $N = 64$
- **Epochs:** $60$
- **Scheduler:** `CosineAnnealingLR` ($\eta_{\max} = 10^{-3} \rightarrow \eta_{\min} = 10^{-5}$)
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(max_norm=5.0)`

### 6.2 Test Set Results (884 Unseen Specimens)

```text
======================================================================
      PIM-MTL CONCRETE STRENGTH PREDICTION PIPELINE RESULTS
======================================================================

Total Test Specimens: 884
Compute Device      : cuda:0

--- Quantitative Accuracy Metrics ---
Target     | Observed N | Coverage   | R2 Score   | RMSE (MPa)   | MAE (MPa)
----------------------------------------------------------------------
Cs_Mpa     | 884        |    100.0% |    0.9812  |     1.3353   |   1.0183
Ts_Mpa     | 696        |     78.7% |    0.9867  |     0.1060   |   0.0761
Fs_Mpa     | 363        |     41.1% |    0.9982  |     0.1024   |   0.0755
----------------------------------------------------------------------

--- Physical Feasibility Audit ---
Tensile Ratio Violations (Rt outside [0.06, 0.18])  : 3.17% (96.83% Compliant)
Flexural Ratio Violations (Rf outside [0.08, 0.25]) : 17.76% (82.24% Compliant)
Physically Impossible Predictions (Ts >= Cs)        : 0.00% (100.00% Compliant)
======================================================================
```

---

## 7. Comprehensive Future Works & Advanced Engineering Roadmap

While PIM-MTL demonstrates state-of-the-art predictive accuracy and zero physically impossible predictions, several high-impact research extensions can be pursued.

---

### Future Work 1: Automated Bayesian Hyperparameter Tuning (Optuna / BoTorch)

Currently, hyperparameters ($\lambda_{\text{phys}} = 0.5$, hidden dimensions $[128, 64, 32]$, task weights $w=[1.0, 1.2, 1.5]$) are manually tuned. Future iterations will implement **Bayesian Optimization via Gaussian Process Regression and Tree-structured Parzen Estimators (TPE)**.

#### 1. Optimization Formulation

Let $\boldsymbol{\theta} \in \Theta$ be the hyperparameter configuration vector:

$$
\boldsymbol{\theta} = \{\lambda_{\text{phys}}, \text{lr}, \text{weight\_decay}, \text{dropout}, w_2, w_3, d_1, d_2, d_3\}
$$

We seek to maximize a composite multi-objective acquisition objective balancing statistical accuracy and physical compliance:

$$
\mathcal{J}(\boldsymbol{\theta}) = \overline{R^2}(\boldsymbol{\theta}) - \gamma_1 \cdot \text{PVR}_t(\boldsymbol{\theta}) - \gamma_2 \cdot \text{PVR}_f(\boldsymbol{\theta})
$$

#### 2. Acquisition Function (Expected Improvement, EI)

Given observed evaluations $\mathcal{D}_t = \{(\boldsymbol{\theta}_k, \mathcal{J}_k)\}_{k=1}^t$ modeled by a Gaussian Process $\mathcal{GP}\left(\mu(\boldsymbol{\theta}), \sigma^2(\boldsymbol{\theta})\right)$:

$$
\text{EI}(\boldsymbol{\theta}) = \mathbb{E}\left[ \max\left(0, \mathcal{J}(\boldsymbol{\theta}) - \mathcal{J}^+ \right) \right] = (\mu(\boldsymbol{\theta}) - \mathcal{J}^+)\Phi\left(\frac{\mu(\boldsymbol{\theta}) - \mathcal{J}^+}{\sigma(\boldsymbol{\theta})}\right) + \sigma(\boldsymbol{\theta})\phi\left(\frac{\mu(\boldsymbol{\theta}) - \mathcal{J}^+}{\sigma(\boldsymbol{\theta})}\right)
$$

where $\mathcal{J}^+ = \max_k \mathcal{J}_k$, $\Phi(\cdot)$ is the standard normal CDF, and $\phi(\cdot)$ is the PDF.

#### 3. Hyperparameter Search Space Specification

| Hyperparameter | Type | Search Range | Sampling Distribution |
|---|---|---|---|
| $\lambda_{\text{phys}}$ | Continuous | $[0.05, 2.0]$ | Log-Uniform |
| Learning Rate ($\text{lr}$) | Continuous | $[10^{-4}, 5 \times 10^{-3}]$ | Log-Uniform |
| Weight Decay | Continuous | $[10^{-6}, 10^{-2}]$ | Log-Uniform |
| Dropout Rate | Continuous | $[0.0, 0.4]$ | Uniform |
| Task Weight $w_{Ts}$ | Continuous | $[0.8, 2.5]$ | Uniform |
| Task Weight $w_{Fs}$ | Continuous | $[1.0, 3.5]$ | Uniform |
| Hidden Layer 1 ($d_1$) | Integer | $[64, 256]$ (step 32) | Discrete Grid |
| Hidden Layer 2 ($d_2$) | Integer | $[32, 128]$ (step 16) | Discrete Grid |
| Hidden Layer 3 ($d_3$) | Integer | $[16, 64]$ (step 8) | Discrete Grid |

---

## 8. Summary & Verification Commands

All core scripts and artifacts are maintained in `3rdYearsem5/`:

- **CLI Run:**

  ```powershell
  python main.py --data Data.csv --epochs 60 --batch-size 64 --results-file results.txt
  ```

- **Interactive Jupyter Notebook:**

  ```powershell
  jupyter notebook notebooks/training_and_analysis.ipynb
  ```

- **Inspect Saved Results:**

  ```powershell
  Get-Content results.txt
  ```
