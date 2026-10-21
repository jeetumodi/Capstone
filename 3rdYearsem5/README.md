# Physics-Informed Masked Multi-Task Learning (PIM-MTL)

A clean, minimal, and reproducible machine learning codebase for multi-task concrete mechanical strength prediction under asymmetric missingness on the **ConcreteXAI** dataset.

Concurrently predicts:
- **$C_s$**: Compressive Strength (MPa) [100% observed]
- **$T_s$**: Splitting Tensile Strength (MPa) [78.3% observed, 21.7% missing]
- **$F_s$**: Flexural Strength / Modulus of Rupture (MPa) [39.8% observed, 60.2% missing]

---

## Repository Structure

```text
3rdYearsem5/
├── notebooks/
│   └── training_and_analysis.ipynb   # Complete 11-step ML workflow notebook with all plots
├── utils/
│   ├── __init__.py                   # Clean exports (PIM_loss, PIM_MTL_Loss, PIM_MTL_Net)
│   ├── losses.py                     # Masked multi-task loss and physics ratio penalty
│   └── models.py                     # PIM_MTL_Net architecture and SafeBatchNorm1d
├── models/
│   ├── .gitkeep                      # Git placeholder
│   ├── pim_mtl_model.pt              # Trained model checkpoint
│   └── preprocessor.joblib           # Preprocessor scalers and target medians/IQRs
├── Data.csv                          # Benchmark dataset (4,420 specimens)
├── main.py                           # Minimal entry point for training and inference
├── document.md                       # Comprehensive scientific documentation and formulas
├── requirements.txt                  # Minimal dependencies
└── README.md                         # Quickstart guide
```

---

## Quickstart

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Complete ML Workflow in Jupyter Notebook
Open and run the notebook:
```bash
jupyter notebook notebooks/training_and_analysis.ipynb
```
The notebook contains the complete 11-step workflow:
1. Imports and Environment Setup
2. Data Loading
3. Data Inspection and Analysis
4. Data Preprocessing
5. Exploratory Data Analysis (inline plots)
6. Model and Loss Function Setup
7. Training Loop (with Adam and Cosine Annealing)
8. Training Monitoring (convergence curves)
9. Visualization of Results (parity plots & residual analysis)
10. Evaluation (quantitative $R^2$, RMSE, MAE, and Physical Violation Rate PVR)
11. Model Saving and Inference Demonstration

### 3. Command-Line Execution
To run training and evaluation directly from the terminal:
```bash
python main.py --data Data.csv --epochs 50 --batch-size 64
```

---

## Technical Documentation
For full mathematical formulations, gradient gating proofs, civil engineering constraint rationale, and detailed ablation analyses, refer to [document.md](document.md).
