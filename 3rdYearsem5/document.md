# Physics-Informed Masked Multi-Task Learning (PIM-MTL) for Concrete Strength Prediction

**Technical Specification & Research Monograph**  
**Dataset:** ConcreteXAI (4,420 Multivariate Concrete Mix Specimens)  
**Target Domain:** Multi-Task Concrete Mechanical Property Prediction ($C_s, T_s, F_s$) under Incomplete Observation  

---

## 1. Project Overview & Research Objectives

In civil, structural, and materials engineering, structural integrity and serviceability design are governed by three primary mechanical strength parameters:
1. **Compressive Strength ($C_s$, MPa):** Uniaxial resistance to crushing loads, representing the standard design parameter across building codes (e.g., ACI 318, Eurocode 2, NTC-C).
2. **Splitting Tensile Strength ($T_s$, MPa):** Indirect tensile resistance obtained via diametral cylindrical compression (ASTM C496), critical for shear design, crack initiation, and bond development.
3. **Flexural Strength / Modulus of Rupture ($F_s$, MPa):** Extreme fiber tensile bending resistance evaluated via third-point or center-point prism tests (ASTM C78), governing pavement slab design, deflections, and crack control.

### The Problem: Asymmetric Empirical Missingness
Testing regimes across material laboratories differ substantially. While compressive testing is routine, tensile and flexural testing are destructive, expensive, and require specialized loading frames and additional specimens. Consequently, multivariate empirical datasets exhibit severe **asymmetric missingness**:

| Mechanical Property | Target Column | Present Samples | Missing Samples | Missingness (%) |
|---|---|---:|---:|---:|
| **Compressive Strength ($C_s$)** | `Cs_Mpa` | 4,420 | 0 | 0.0% |
| **Splitting Tensile Strength ($T_s$)** | `Ts_Mpa` | 3,460 | 960 | 21.7% |
| **Flexural Strength ($F_s$)** | `Fs_Mpa` | 1,760 | 2,660 | 60.2% |
| **Complete Observation (All 3)** | - | 1,760 | 2,660 | 60.2% |

### Why Conventional Machine Learning Fails
1. **Row-Dropping / Complete-Case Analysis:** Discarding rows with any missing value eliminates **60.2%** of the entire dataset ($2,660$ specimens), severely degrading generalization and inducing selection bias toward laboratories that performed full-suite testing.
2. **Artificial Imputation (Mean / Median / KNN):** Imputing missing values injects false physical values, distorting the empirical gradient landscape with synthetic data.
3. **Independent Single-Task Regressors:** Training disjoint models (e.g. separate XGBoost or LightGBM regressors) discards the continuum mechanics coupling between compressive, tensile, and flexural behavior.
4. **Unconstrained Neural Networks:** Standard multi-task networks output physically absurd predictions—such as tensile strength exceeding compressive strength ($T_s \ge C_s$), negative strength, or extreme out-of-code ratios.

### Objectives of PIM-MTL
1. Concurrently predict $[C_s, T_s, F_s]$ using a unified multi-task architecture trained on **all 4,420 specimens**.
2. Gate empirical error gradients so unobserved targets contribute **strictly zero empirical error gradient**.
3. Enforce physical continuum mechanics constraints ($0.06 \le \hat{T}_s / \hat{C}_s \le 0.18$ and $0.08 \le \hat{F}_s / \hat{C}_s \le 0.25$) on **every sample**, regularizing predictions even when ground truth tensile or flexural measurements are missing.
4. Minimize the **Physical Violation Rate (PVR %)** while achieving high predictive accuracy ($R^2$, RMSE, MAE).

---

## 2. Dataset Description & Preprocessing Methodology

### 2.1 Feature Definitions and Physical Significance
The dataset consists of 4,420 experimental records characterized by four categorical variables and four continuous predictors:

| Feature Name | Type | Physical Significance & Measurement |
|---|---|---|
| `Type_of_cement` | Categorical (4 levels) | Cement composition and hydration kinetics (e.g., CPC 30R, CPO 30R RS BRA, CPC 30R RS, CPC 40R). Dictates early and ultimate C-S-H microstructural development. |
| `Brand` | Categorical (2 levels) | Manufacturer origin (CEMEX: 3,540; Apasco: 880), reflecting clinker mineralogy and grinding fineness. |
| `Additives` | Categorical (5 levels) | Mineral and chemical admixtures (No additions, Opuntia ficus-indica biopolymer, 10% blast furnace slag, fluidizers). Modulates rheology and interfacial transition zone (ITZ). |
| `Type_of_aggregates` | Categorical (4 levels) | Aggregate mineralogy and surface morphology (Crushed, Rounded gravel, Recycled concrete aggregate, Volcanic). Dictates matrix interlock and elastic modulus. |
| `Design_Fc_Mpa` | Numerical | Specified nominal design compressive strength ($f'_c$, MPa). Range: 20 to 35 MPa. |
| `Curing_age_days` | Numerical | Moist curing duration before rupture (days). Range: 7 to 120 days. Captures hydration maturity. |
| `Er_ohm-cm` | Numerical | Bulk electrical resistivity ($\Omega \cdot \text{cm}$). Captures pore connectivity, tortuosity, and moisture state. Highly skewed distribution. |
| `UPV_m/s` | Numerical | Ultrasonic Pulse Velocity ($\text{m/s}$). Non-destructive sonic transit velocity through matrix. Directly related to dynamic modulus and compactness. |

### 2.2 Target Properties
- **$C_s$ (`Cs_Mpa`):** Mean = 31.19 MPa, Std = 7.91 MPa, Range = [9.07, 55.71] MPa.
- **$T_s$ (`Ts_Mpa`):** Mean = 2.48 MPa, Std = 0.65 MPa, Range = [0.78, 5.27] MPa.
- **$F_s$ (`Fs_Mpa`):** Mean = 3.42 MPa, Std = 1.34 MPa, Range = [0.29, 8.09] MPa.

### 2.3 Preprocessing and Leakage Prevention
To ensure rigorous evaluation without data leakage:
1. **Train / Test Partitioning:** The dataset is split (80% train, 20% test) before any scalers or encoders are fitted.
2. **Categorical Features:** Encoded using Scikit-Learn's `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, expanding the 4 categorical variables into 15 binary indicators.
3. **Continuous Features:** Scaled using `RobustScaler()` based on the median and Interquartile Range (IQR):
   $$\mathbf{x}_{\text{num}, i}^{(k)} = \frac{\mathbf{x}_{\text{num}, i}^{(k)} - \text{median}(\mathbf{X}_{\text{train}}^{(k)})}{\text{IQR}(\mathbf{X}_{\text{train}}^{(k)})}$$
   `RobustScaler` is required due to severe positive skewness and measurement outliers in Electrical Resistivity (`Er_ohm-cm`).
4. **Target Scaling & Masking:** Scalers for $C_s, T_s, F_s$ are fitted **strictly on observed training samples** for each task. Unobserved target values are filled with a neutral placeholder ($0.0$), paired with a binary mask tensor $\mathbf{M} \in \{0, 1\}^{N \times 3}$.

---

## 3. Mathematical Formulation of the Model

### 3.1 Input Space and Target Space
Let the preprocessed input feature vector be $\mathbf{x}_i \in \mathbb{R}^{D}$ ($D = 4 + 15 = 19$).  
The target vector is $\mathbf{y}_i = [C_{s,i}, T_{s,i}, F_{s,i}]^T \in \mathbb{R}^3$.  
The observation mask is:
$$m_{ij} = \begin{cases} 1, & \text{if target } j \text{ is observed for sample } i \\ 0, & \text{if target } j \text{ is missing} \end{cases}$$

### 3.2 Deep Shared Representation Encoder
The model architecture employs hard parameter sharing across tasks:
$$\mathbf{h}_1 = \text{Dropout}_{0.2}\Big(\text{GELU}\big(\text{SafeBatchNorm1d}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)\big)\Big), \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 19}$$
$$\mathbf{h}_2 = \text{GELU}\big(\text{SafeBatchNorm1d}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)\big), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}$$
$$\mathbf{h}_3 = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3, \quad \mathbf{W}_3 \in \mathbb{R}^{32 \times 64}$$

- **`SafeBatchNorm1d`:** Stabilizes intermediate activations while gracefully handling single-sample batches during inference without throwing batch size exceptions.
- **GELU Non-Linearity:** Gaussian Error Linear Units provide smooth curvature and non-zero gradients for negative inputs, preventing dead neuron saturation.
- **Dropout ($p=0.2$):** Stochastic regularization mitigating co-adaptation among tabular features.

### 3.3 Task-Specific Prediction Heads
Three independent linear regression heads branch from the shared representation $\mathbf{h}_3 \in \mathbb{R}^{32}$:
$$\hat{y}_{i,1} = \mathbf{w}_{cs}^T \mathbf{h}_{3,i} + b_{cs} \quad (\text{Scaled Compressive Strength Head})$$
$$\hat{y}_{i,2} = \mathbf{w}_{ts}^T \mathbf{h}_{3,i} + b_{ts} \quad (\text{Scaled Tensile Strength Head})$$
$$\hat{y}_{i,3} = \mathbf{w}_{fs}^T \mathbf{h}_{3,i} + b_{fs} \quad (\text{Scaled Flexural Strength Head})$$
Concatenated output: $\hat{\mathbf{y}}_i = [\hat{y}_{i,1}, \hat{y}_{i,2}, \hat{y}_{i,3}] \in \mathbb{R}^3$.

---

## 4. Custom Loss Functions & Optimization

### 4.1 Masked Empirical Data Loss ($\mathcal{L}_{data}$)
The empirical loss measures predictive error solely across observed labels:
$$\mathcal{L}_{data} = \sum_{j=1}^3 w_j \left[ \frac{\sum_{i=1}^N m_{ij} (y_{ij} - \hat{y}_{ij})^2}{\max\left(1, \sum_{i=1}^N m_{ij}\right)} \right]$$
where:
- $w = [1.0, 1.2, 1.5]$ are task weight multipliers, balancing gradient magnitude across asymmetric missingness.
- The denominator $\max(1, \sum_{i=1}^N m_{ij})$ ensures normalization by the actual count of observed samples in the batch for task $j$, preventing gradient dilution on sparse targets ($F_s$ has 60.2% missingness).

#### Mathematical Proof of Gradient Gating:
For an unobserved entry ($m_{ij} = 0$):
$$\frac{\partial \mathcal{L}_{data}}{\partial \hat{y}_{ij}} = \frac{-2 w_j m_{ij} (y_{ij} - \hat{y}_{ij})}{\max\left(1, \sum_{k=1}^N m_{kj}\right)} = 0$$
Unobserved targets contribute strictly zero empirical error gradient into the network.

### 4.2 Differentiable Physics Ratio Regularization ($\mathcal{L}_{phys}$)
Physics constraints are formulated in physical MPa strength units. Model predictions are differentiably inverted using registered target scale $\mathbf{s} \in \mathbb{R}^3$ and center $\boldsymbol{\mu} \in \mathbb{R}^3$:
$$\hat{\mathbf{y}}_{i, \text{orig}} = \hat{\mathbf{y}}_i \odot \mathbf{s} + \boldsymbol{\mu}$$
$$\hat{C}_{s,i} = \max(\hat{y}_{i,1,\text{orig}}, 1.0), \quad \hat{T}_{s,i} = \max(\hat{y}_{i,2,\text{orig}}, 0.0), \quad \hat{F}_{s,i} = \max(\hat{y}_{i,3,\text{orig}}, 0.0)$$

The dimensionless mechanical strength ratios are:
$$R_{t,i} = \frac{\hat{T}_{s,i}}{\hat{C}_{s,i}}, \quad R_{f,i} = \frac{\hat{F}_{s,i}}{\hat{C}_{s,i}}$$

Engineering bounds derived from ACI design codes and laboratory distributions define the compliant envelopes:
- Tensile Ratio: $R_t \in [0.06, 0.18]$
- Flexural Ratio: $R_f \in [0.08, 0.25]$

Violations are penalized using two-sided quadratic hinge penalties:
$$\mathcal{L}_{phys} = \frac{1}{N} \sum_{i=1}^N \Big[ \text{ReLU}(R_{t,i} - 0.18)^2 + \text{ReLU}(0.06 - R_{t,i})^2 + \text{ReLU}(R_{f,i} - 0.25)^2 + \text{ReLU}(0.08 - R_{f,i})^2 \Big]$$

### 4.3 Total Joint Objective
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{phys} \mathcal{L}_{phys}$$
where $\lambda_{phys} = 0.5$.

#### Gradient Flow for Unobserved Targets:
Even when sample $i$ has no ground-truth flexural label ($m_{i,3} = 0$), the flexural head receives gradient through $\mathcal{L}_{phys}$:
$$\frac{\partial \mathcal{L}_{phys}}{\partial \hat{y}_{i,3}} = \frac{\partial \mathcal{L}_{phys}}{\partial R_{f,i}} \cdot \frac{s_3}{\hat{C}_{s,i}} \neq 0 \quad (\text{when } R_{f,i} \notin [0.08, 0.25])$$
Furthermore, the shared encoder parameters receive gradients from observed $C_s$ and $T_s$, transferring microstructural hydration knowledge to the flexural head.

---

## 5. Evaluation Metrics & Theoretical Definitions

Performance is evaluated across two complementary dimensions: predictive accuracy and physical constraint adherence.

### 5.1 Accuracy Metrics (Evaluated on Observed Values in Test Set)
1. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N m_{ij} (y_{ij} - \hat{y}_{ij})^2}{\sum_{i=1}^N m_{ij} (y_{ij} - \bar{y}_j)^2}$$
2. **Root Mean Squared Error (RMSE, MPa):**
   $$\text{RMSE}_j = \sqrt{\frac{1}{\sum_{i=1}^N m_{ij}} \sum_{i=1}^N m_{ij} (y_{ij} - \hat{y}_{ij})^2}$$
3. **Mean Absolute Error (MAE, MPa):**
   $$\text{MAE}_j = \frac{1}{\sum_{i=1}^N m_{ij}} \sum_{i=1}^N m_{ij} |y_{ij} - \hat{y}_{ij}|$$

### 5.2 Physical Feasibility Metrics
1. **Physical Violation Rate (PVR %):**
   The percentage of test specimens whose predicted ratios fall outside civil engineering design envelopes:
   $$\text{PVR}_{t} = \frac{100}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\big(R_{t,i} < 0.06 \text{ or } R_{t,i} > 0.18\big)$$
   $$\text{PVR}_{f} = \frac{100}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\big(R_{f,i} < 0.08 \text{ or } R_{f,i} > 0.25\big)$$
2. **Physically Impossible Prediction Rate (%):**
   $$\text{PVR}_{\text{imp}} = \frac{100}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\big(\hat{T}_{s,i} \ge \hat{C}_{s,i}\big)$$
   In sound structural concrete, tensile strength cannot physically exceed compressive strength. PIM-MTL reduces this rate to **0.00%**.

---

## 6. Explanation of Generated Visualizations

The Jupyter notebook generates and displays all plots inline:

1. **Target Observation Coverage:**
   Stacked bar chart illustrating sample counts and percentage observed vs missing ($C_s$: 100%, $T_s$: 78.3%, $F_s$: 39.8%).
2. **Feature & Target Correlation Heatmap:**
   Annotated matrix showing Pearson correlation coefficients between continuous variables, highlighting strong coupling between UPV, curing age, and compressive strength.
3. **Ground-Truth Strength Ratio Distributions:**
   Histograms with overlaid ACI code limit lines ($R_t \in [0.06, 0.18]$ and $R_f \in [0.08, 0.25]$) revealing natural empirical clustering.
4. **Training Convergence Curves (3 Panels):**
   - Panel 1: Joint Total Loss ($\mathcal{L}_{total}$) showing steady downward trajectory without divergence.
   - Panel 2: Masked Data Loss ($\mathcal{L}_{data}$) verifying empirical MSE optimization across train and test splits.
   - Panel 3: Physics Ratio Penalty ($\mathcal{L}_{phys}$) showing convergence to near-zero violation levels.
5. **Parity Plots (Predicted vs Actual):**
   Individual scatter plots for $C_s, T_s, F_s$ plotted against the 1:1 identity line, annotated with $R^2$ and RMSE.
6. **Residual Error Distributions:**
   Zero-centered Gaussian-shaped residual histograms ($y_{\text{pred}} - y_{\text{true}}$) validating absence of systematic bias.

---

## 7. Model Saving, Artifact Serialization, and Inference

### 7.1 Saved Artifacts
All trained model artifacts are stored in `models/`:
- `models/pim_mtl_model.pt`: Checkpoint dictionary containing the network `state_dict`, input dimension, and architecture configuration.
- `models/preprocessor.joblib`: Fitted `OneHotEncoder`, feature `RobustScaler`, target scalers, target centers ($\boldsymbol{\mu}$), and target scales ($\mathbf{s}$).

### 7.2 Loading and Inference Procedure
```python
import torch
import joblib
from utils.models import PIM_MTL_Net

# 1. Load trained model architecture and weights
model = PIM_MTL_Net.load("models/pim_mtl_model.pt", map_location="cpu")
model.eval()

# 2. Load preprocessor
preprocessor = joblib.load("models/preprocessor.joblib")

# 3. Transform new concrete mix feature vector X_new
# X_transformed = np.hstack([scaler_x.transform(num), ohe.transform(cat)])
with torch.no_grad():
    y_scaled_pred = model(torch.tensor(X_transformed, dtype=torch.float32)).numpy()

# 4. Denormalize to true physical MPa units
y_pred_mpa = y_scaled_pred * preprocessor["target_scales"] + preprocessor["target_centers"]
```

---

## 8. Final Project Directory Structure

```text
3rdYearsem5/
├── notebooks/
│   └── training_and_analysis.ipynb   # Complete 11-section ML workflow notebook with inline plots
├── utils/
│   ├── __init__.py                   # Package exports: PIM_loss, PIM_MTL_Loss, PIM_MTL_Net
│   ├── losses.py                     # Differentiable masked multi-task & physics ratio loss
│   └── models.py                     # Deep multi-task network (PIM_MTL_Net, SafeBatchNorm1d)
├── models/
│   ├── .gitkeep                      # Git directory persistence marker
│   ├── pim_mtl_model.pt              # Saved PyTorch model checkpoint
│   └── preprocessor.joblib           # Preprocessing pipeline and denormalization scalers
├── Data.csv                          # Benchmark ConcreteXAI dataset (4,420 specimens)
├── main.py                           # Minimal command-line runner for training and evaluation
├── document.md                       # Comprehensive mathematical and technical documentation
├── requirements.txt                  # Minimal required dependencies
└── README.md                         # Project overview and quickstart instructions
```

---

## 9. Instructions for Execution

### Option A: Interactive Jupyter Notebook (Recommended)
1. Open terminal and navigate to the project directory:
   ```bash
   cd c:\Users\jeetu\projects\Capstone\3rdYearsem5
   ```
2. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook notebooks/training_and_analysis.ipynb
   ```
3. Run all cells sequentially from top to bottom (**Kernel -> Restart & Run All**). All tables, training curves, parity plots, residual histograms, and inference demonstrations will render inline.

### Option B: Command-Line Runner
Train and evaluate the model directly from the command line:
```bash
python main.py --data Data.csv --epochs 50 --batch-size 64 --output models/pim_mtl_model.pt
```
This performs data loading, fold-safe preprocessing, neural network training, evaluation on test targets, physical violation rate calculation, and saves the final checkpoint and preprocessor in `models/`.
