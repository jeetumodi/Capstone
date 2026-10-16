"""
Intelligent Concrete Strength Prediction Pipeline
Two-Stage Virtual Sensing + Physics-Informed Multi-Task ANN + Conformal Calibration
Targeting: ConcreteXAI Benchmark (Data.csv)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib

# ==============================================================================
# 0. DETERMINISTIC SEED & DEVICE SETUP
# ==============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. DATA INGESTION & COMPOSITE HASH IDENTIFIERS
# ==============================================================================
def load_and_structure_data(filepath: str = "ConcreteXAI\Data.csv") -> pd.DataFrame:
    """
    Loads ConcreteXAI CSV. Constructs composite Mix_ID to guarantee
    GroupKFold partitions strictly across concrete formulations.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        # Programmatic synthetic fallback conforming exactly to ConcreteXAI schema
        print(f"[WARN] '{filepath}' not found. Initializing synthetic benchmark...")
        n_samples = 4420
        cements = ['CPO 30R RS BRA', 'CPC 40R', 'CPC 30R RS', 'CPC 30R']
        brands = ['CEMEX', 'Apasco']
        additives = ['Opuntia_ficus_indica', 'No_additions', 'Starch_fluidizer', 
                     'fluidizer', 'Blast_furnace_slag_10%']
        aggregates = ['Rounded', 'Crushed', 'Volcanic', 'Recycled']
        
        mix_combinations = []
        for _ in range(12):
            mix_combinations.append({
                'Type_of_cement': random.choice(cements),
                'Brand': random.choice(brands),
                'Additives': random.choice(additives),
                'Type_of_aggregates': random.choice(aggregates),
                'Design_Fc_Mpa': random.choice([25.0, 28.0, 30.0, 35.0])
            })
            
        records = []
        curing_schedule = [3, 7, 14, 28, 40, 60, 90, 120]
        for i in range(n_samples):
            mix = random.choice(mix_combinations)
            age = random.choice(curing_schedule)
            cs = float(np.clip(mix['Design_Fc_Mpa'] * (0.3 + 0.7 * np.log10(age)) + np.random.normal(0, 3), 5.0, 56.0))
            ts = float(np.clip(0.35 * (cs ** 0.67) + np.random.normal(0, 0.3), 0.5, 5.3)) if random.random() > 0.217 else np.nan
            fs = float(np.clip(0.65 * (cs ** 0.60) + np.random.normal(0, 0.4), 0.3, 8.1)) if random.random() > 0.602 else np.nan
            er = float(np.clip(1.5 + 0.18 * cs + 0.04 * age + np.random.normal(0, 0.8), 1.8, 16.2))
            upv = float(np.clip(2200 + 35 * cs + 5 * age + np.random.normal(0, 80), 2345.0, 4666.0))
            
            row = dict(mix)
            row.update({
                'Curing_age_days': age,
                'Cs_Mpa': cs,
                'Ts_Mpa': ts,
                'Fs_Mpa': fs,
                'Er_ohm-cm': er,
                'UPV_m/s': upv
            })
            records.append(row)
        df = pd.DataFrame(records)

    # Construct unique composite Mix_ID string
    df['Mix_ID'] = (
        df['Type_of_cement'].astype(str) + "_" +
        df['Brand'].astype(str) + "_" +
        df['Additives'].astype(str) + "_" +
        df['Type_of_aggregates'].astype(str) + "_" +
        df['Design_Fc_Mpa'].astype(str)
    )
    return df

# ==============================================================================
# 2. FEATURE PIPELINE & TRANSFORMER DEFINITIONS
# ==============================================================================
CATEGORICAL_FEATURES = ['Type_of_cement', 'Brand', 'Additives', 'Type_of_aggregates']
NUMERICAL_INPUTS = ['Design_Fc_Mpa', 'Curing_age_days']
NDT_TARGETS = ['Er_ohm-cm', 'UPV_m/s']
STRENGTH_TARGETS = ['Cs_Mpa', 'Ts_Mpa', 'Fs_Mpa']

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('num', StandardScaler(), NUMERICAL_INPUTS)
        ]
    )

# ==============================================================================
# 3. PYTORCH MULTI-TASK DATASET WITH MASKING & DERIVATIVES
# ==============================================================================
class ConcreteMTLDataset(Dataset):
    """
    PyTorch Dataset outputting preprocessed features, observed targets,
    and boolean valid masks to natively handle target missingness.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, age_indices: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        
        # Mask missing targets: 1 if present, 0 if NaN
        masks = (~np.isnan(y)).astype(np.float32)
        cleaned_y = np.nan_to_num(y, nan=0.0)
        
        self.y = torch.tensor(cleaned_y, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.age_indices = age_indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]

# ==============================================================================
# 4. STAGE 2 ARCHITECTURE: PHYSICS-INFORMED MULTI-TASK NETWORK
# ==============================================================================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class PhysicsInformedMTNN(nn.Module):
    """
    Multi-Task Neural Network with Shared Representation Trunk
    and Independent Regression Heads for Cs, Ts, and Fs.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.2):
        super(PhysicsInformedMTNN, self).__init__()
        
        # Shared Latent Trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            Mish()
        )
        
        # Dedicated Task-Specific Heads
        trunk_out = hidden_dim // 4
        self.head_cs = nn.Linear(trunk_out, 1)  # Compressive
        self.head_ts = nn.Linear(trunk_out, 1)  # Tensile
        self.head_fs = nn.Linear(trunk_out, 1)  # Flexural

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_trunk(x)
        out_cs = self.head_cs(features)
        out_ts = self.head_ts(features)
        out_fs = self.head_fs(features)
        return torch.cat([out_cs, out_ts, out_fs], dim=1)

def compute_physics_loss(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor,
                         inputs: torch.Tensor, age_col_idx: int,
                         lambda_ratio: float = 0.5, lambda_mono: float = 0.1) -> torch.Tensor:
    """
    Computes masked multi-task MSE combined with structural domain penalties:
    1. Flexural Strength <= 35% Compressive Strength
    2. Tensile Strength <= Flexural Strength
    3. Monotonic strength gain with curing age
    """
    # 1. Masked Empirical Loss
    diff_sq = (preds - targets) ** 2
    masked_diff = diff_sq * masks
    loss_mse = masked_diff.sum() / (masks.sum() + 1e-8)
    
    cs = preds[:, 0]
    ts = preds[:, 1]
    fs = preds[:, 2]
    
    # 2. Mechanics Ratio Penalties
    # Violation 1: Fs should not exceed 0.35 * Cs
    penalty_ratio1 = torch.relu(fs - 0.35 * cs).pow(2).mean()
    # Violation 2: Ts should not exceed Fs
    penalty_ratio2 = torch.relu(ts - fs).pow(2).mean()
    loss_ratio = penalty_ratio1 + penalty_ratio2
    
    # 3. Monotonic Curing Age Gradient Penalty
    # Approximate numerical derivative: inputs must have grad enabled
    if inputs.requires_grad:
        grad_cs = torch.autograd.grad(
            outputs=cs.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0]
        # Slope with respect to standardized curing age column
        d_cs_d_age = grad_cs[:, age_col_idx]
        loss_mono = torch.relu(-d_cs_d_age).pow(2).mean()
    else:
        loss_mono = torch.tensor(0.0, device=preds.device)
        
    return loss_mse + (lambda_ratio * loss_ratio) + (lambda_mono * loss_mono)

# ==============================================================================
# 5. INDUCTIVE CONFORMAL PREDICTION CALIBRATOR
# ==============================================================================
class MultiTaskConformalCalibrator:
    """
    Split Conformal Prediction Engine computing valid non-conformity quantiles
    per target task to guarantee nominal coverage (1 - alpha).
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = {}

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates non-conformity threshold q_hat per strength target.
        """
        for i, target_name in enumerate(STRENGTH_TARGETS):
            valid_mask = ~np.isnan(y_true[:, i])
            residuals = np.abs(y_true[valid_mask, i] - y_pred[valid_mask, i])
            n_cal = len(residuals)
            
            p = int(np.ceil((n_cal + 1) * (1.0 - self.alpha))) / n_cal
            p = min(1.0, max(0.0, p))
            self.q_hat[target_name] = float(np.quantile(residuals, p))

    def predict_intervals(self, y_pred: np.ndarray) -> dict:
        """
        Returns point predictions, lower bounds (characteristic design strength),
        and upper bounds for all targets.
        """
        intervals = {}
        for i, target_name in enumerate(STRENGTH_TARGETS):
            margin = self.q_hat.get(target_name, 0.0)
            intervals[target_name] = {
                'point': y_pred[:, i],
                'lower': np.clip(y_pred[:, i] - margin, 0.0, None), # Physical bound >= 0
                'upper': y_pred[:, i] + margin,
                'margin': margin
            }
        return intervals

# ==============================================================================
# 6. COMPLETE END-TO-END EXECUTION PIPELINE
# ==============================================================================
def run_intelligent_concrete_pipeline():
    print("=" * 80)
    print("RUNNING TWO-STAGE VIRTUAL SENSING + MULTI-TASK CONFORMAL PIPELINE")
    print("=" * 80)
    
    # 6.1 Load Data & Define Leakage-Free Splits
    df = load_and_structure_data("Data.csv")
    mix_ids = df['Mix_ID'].values
    unique_mixes = np.unique(mix_ids)
    print(f"[INFO] Dataset Shape: {df.shape} across {len(unique_mixes)} distinct mix designs.")
    
    # Stratify by Mix_ID: 8 mixes Train, 2 mixes Calibration, 2 mixes Test
    np.random.shuffle(unique_mixes)
    train_mixes = unique_mixes[:8]
    cal_mixes = unique_mixes[8:10]
    test_mixes = unique_mixes[10:]
    
    train_df = df[df['Mix_ID'].isin(train_mixes)].copy()
    cal_df = df[df['Mix_ID'].isin(cal_mixes)].copy()
    test_df = df[df['Mix_ID'].isin(test_mixes)].copy()
    
    print(f"[SPLIT] Train: {len(train_df)} rows | Cal: {len(cal_df)} rows | Test: {len(test_df)} rows")
    
    # 6.2 Stage 1: Train Virtual Sensing Models for NDT Variables
    preprocessor = build_preprocessor()
    X_train_raw = preprocessor.fit_transform(train_df[CATEGORICAL_FEATURES + NUMERICAL_INPUTS])
    X_cal_raw = preprocessor.transform(cal_df[CATEGORICAL_FEATURES + NUMERICAL_INPUTS])
    X_test_raw = preprocessor.transform(test_df[CATEGORICAL_FEATURES + NUMERICAL_INPUTS])
    
    stage1_models = {}
    virtual_train_ndt = np.zeros((len(train_df), 2))
    virtual_cal_ndt = np.zeros((len(cal_df), 2))
    virtual_test_ndt = np.zeros((len(test_df), 2))
    
    print("\n--- Training Stage 1: Virtual Sensing (LightGBM) ---")
    for idx, ndt_col in enumerate(NDT_TARGETS):
        reg = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1)
        reg.fit(X_train_raw, train_df[ndt_col].values)
        stage1_models[ndt_col] = reg
        
        virtual_train_ndt[:, idx] = reg.predict(X_train_raw)
        virtual_cal_ndt[:, idx] = reg.predict(X_cal_raw)
        virtual_test_ndt[:, idx] = reg.predict(X_test_raw)
        
        test_r2 = r2_score(test_df[ndt_col].values, virtual_test_ndt[:, idx])
        print(f"[{ndt_col}] Stage 1 Virtual Sensing Generalization R^2: {test_r2:.4f}")
        
    # Scale virtual NDT proxies before feeding into Stage 2
    ndt_scaler = StandardScaler()
    scaled_vtrain_ndt = ndt_scaler.fit_transform(virtual_train_ndt)
    scaled_vcal_ndt = ndt_scaler.transform(virtual_cal_ndt)
    scaled_vtest_ndt = ndt_scaler.transform(virtual_test_ndt)
    
    # Construct Stage 2 Input Matrices
    X_train_stage2 = np.hstack([X_train_raw, scaled_vtrain_ndt])
    X_cal_stage2 = np.hstack([X_cal_raw, scaled_vcal_ndt])
    X_test_stage2 = np.hstack([X_test_raw, scaled_vtest_ndt])
    
    y_train = train_df[STRENGTH_TARGETS].values
    y_cal = cal_df[STRENGTH_TARGETS].values
    y_test = test_df[STRENGTH_TARGETS].values
    
    # 6.3 Stage 2: Train Physics-Informed Multi-Task Neural Network
    train_dataset = ConcreteMTLDataset(X_train_stage2, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    input_dim = X_train_stage2.shape[1]
    # Locate the standardized 'Curing_age_days' column index for monotonicity gradient
    age_col_idx = X_train_raw.shape[1] - 1
    
    model = PhysicsInformedMTNN(input_dim=input_dim, hidden_dim=128, dropout_rate=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    print("\n--- Training Stage 2: Physics-Informed Multi-Task ANN ---")
    model.train()
    epochs = 150
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            batch_x.requires_grad_(True) # Needed for physical monotonicity derivative
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = compute_physics_loss(preds, batch_y, batch_mask, batch_x, age_col_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            
        scheduler.step()
        if epoch % 30 == 0 or epoch == epochs:
            avg_loss = total_loss / len(train_dataset)
            print(f"Epoch {epoch:03d}/{epochs} | Physics Composite Loss: {avg_loss:.4f}")
            
    # 6.4 Inference on Calibration and Test Partitions
    model.eval()
    with torch.no_grad():
        cal_preds = model(torch.tensor(X_cal_stage2, dtype=torch.float32).to(device)).cpu().numpy()
        test_preds = model(torch.tensor(X_test_stage2, dtype=torch.float32).to(device)).cpu().numpy()
        
    # 6.5 Conformal Calibration (Stage 3)
    conformal = MultiTaskConformalCalibrator(alpha=0.05)
    conformal.calibrate(y_cal, cal_preds)
    test_intervals = conformal.predict_intervals(test_preds)
    
    # 6.6 Performance Benchmarking & Metric Reporting
    print("\n" + "=" * 80)
    print("FINAL TEST-SET EVALUATION ON UNSEEN CONCRETE FORMULATIONS")
    print("=" * 80)
    
    for i, target_name in enumerate(STRENGTH_TARGETS):
        actuals = y_test[:, i]
        valid_idx = ~np.isnan(actuals)
        
        y_t = actuals[valid_idx]
        y_p = test_preds[valid_idx, i]
        lower_b = test_intervals[target_name]['lower'][valid_idx]
        upper_b = test_intervals[target_name]['upper'][valid_idx]
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        
        # Empirical Coverage Probability (PICP)
        covered = (y_t >= lower_b) & (y_t <= upper_b)
        picp = np.mean(covered) * 100.0
        mpiw = np.mean(upper_b - lower_b)
        
        print(f"Target: {target_name.upper():<8} | R^2: {r2:.4f} | RMSE: {rmse:.3f} MPa | MAE: {mae:.3f} MPa")
        print(f"         95% Conformal Coverage: {picp:.2f}% | Mean Interval Width: {mpiw:.3f} MPa | Margin: +/-{conformal.q_hat[target_name]:.3f} MPa")
        print("-" * 80)

if __name__ == "__main__":
    run_intelligent_concrete_pipeline()