"""
Configuration Management for PIM-MTL Ablation Studies.

Defines strongly-typed dataclasses for dataset, model, loss, and training parameters,
along with the complete registry of ablation experiment configurations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import copy


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    data_path: str = "data/Data.csv"
    test_size: float = 0.3  # Fixed 70:30 train/test split
    split_seed: int = 42    # Fixed seed for data splitting
    include_ndt: bool = True  # If False, drop UPV_m/s and Er_ohm-cm before scaling
    missing_strategy: str = "masked"  # 'masked', 'complete_case', 'mean_imputation', 'knn_imputation'
    knn_neighbors: int = 5


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    activation: str = "gelu"  # 'gelu', 'relu', 'leaky_relu'
    normalization: str = "safebatchnorm"  # 'safebatchnorm', 'batchnorm', 'layernorm', 'none'


@dataclass
class LossConfig:
    """Loss function and physics regularization parameters."""
    lambda_phys: float = 0.5
    hinge_type: str = "quadratic"  # 'quadratic', 'linear'
    task_weights: Tuple[float, float, float] = (1.0, 1.2, 1.5)  # Weights for [Cs, Ts, Fs]
    use_ratio_bounds: bool = True
    rt_bounds: Tuple[float, float] = (0.06, 0.18)  # Splitting tensile ratio ACI/Eurocode envelope
    rf_bounds: Tuple[float, float] = (0.08, 0.25)  # Flexural ratio ACI/Eurocode envelope
    use_ts_le_cs: bool = False  # Explicit Ts < Cs penalty
    use_ts_le_fs: bool = False  # Explicit Ts < Fs hierarchy penalty
    normalization: str = "per_task"  # 'per_task' or 'batch_size'
    epsilon: float = 1e-6


@dataclass
class TrainConfig:
    """Training optimization parameters."""
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2025])


@dataclass
class ExperimentConfig:
    """Full experiment specification."""
    name: str
    category: str  # 'physics', 'missing_data', 'task_weighting', 'activation', 'normalization', 'feature_ablation'
    description: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_baseline_config() -> ExperimentConfig:
    """Return the canonical PIM-MTL baseline experiment configuration."""
    return ExperimentConfig(
        name="baseline",
        category="physics",
        description="Full PIM-MTL baseline (λ=0.5, quadratic hinge, SafeBN, GELU, task weights [1.0, 1.2, 1.5], masked MTL, 19 features)",
        data=DataConfig(),
        model=ModelConfig(),
        loss=LossConfig(),
        train=TrainConfig(),
    )


def build_experiment_registry() -> Dict[str, ExperimentConfig]:
    """
    Construct the complete registry of 20 ablation experiments
    covering all six research ablation categories.
    """
    registry: Dict[str, ExperimentConfig] = {}

    # 1. Baseline
    base = get_baseline_config()
    registry["baseline"] = base

    # ----------------------------------------------------
    # Category A: Physics Loss
    # ----------------------------------------------------
    # A1: No physics loss (λ = 0)
    cfg = copy.deepcopy(base)
    cfg.name = "phys_lambda_0.0"
    cfg.category = "physics"
    cfg.description = "Ablation: No physics loss (λ_phys = 0.0)"
    cfg.loss.lambda_phys = 0.0
    registry[cfg.name] = cfg

    # A2: Linear hinge instead of quadratic hinge
    cfg = copy.deepcopy(base)
    cfg.name = "phys_linear_hinge"
    cfg.category = "physics"
    cfg.description = "Ablation: Linear hinge penalty instead of quadratic hinge"
    cfg.loss.hinge_type = "linear"
    registry[cfg.name] = cfg

    # A3: Physics weight sweeps [0.01, 0.1, 1.0, 2.0] (0.5 is baseline)
    for l_val in [0.01, 0.1, 1.0, 2.0]:
        cfg = copy.deepcopy(base)
        cfg.name = f"phys_lambda_{l_val}"
        cfg.category = "physics"
        cfg.description = f"Ablation: Physics weight sensitivity sweep (λ_phys = {l_val})"
        cfg.loss.lambda_phys = l_val
        registry[cfg.name] = cfg

    # A4: Without Ts < Cs constraint (remove upper bound on Rt so Ts can exceed Cs)
    cfg = copy.deepcopy(base)
    cfg.name = "phys_without_ts_lt_cs"
    cfg.category = "physics"
    cfg.description = "Ablation: Without Ts < Cs constraint (unbounded upper Rt ratio)"
    cfg.loss.rt_bounds = (0.06, 100.0)
    cfg.loss.use_ts_le_cs = False
    registry[cfg.name] = cfg

    # A5: Without Ts < Fs constraint (unconstrained hierarchy)
    cfg = copy.deepcopy(base)
    cfg.name = "phys_without_ts_lt_fs"
    cfg.category = "physics"
    cfg.description = "Ablation: Without Ts < Fs hierarchy constraint (standard unconstrained hierarchy)"
    cfg.loss.use_ts_le_fs = False
    registry[cfg.name] = cfg

    # A6: With explicit Ts < Fs hierarchy constraint
    cfg = copy.deepcopy(base)
    cfg.name = "phys_with_ts_lt_fs"
    cfg.category = "physics"
    cfg.description = "Ablation: With explicit Ts < Fs hierarchy penalty enabled"
    cfg.loss.use_ts_le_fs = True
    registry[cfg.name] = cfg

    # A7: Without ACI/Eurocode ratio constraints (retaining only basic hierarchy Ts < Fs < Cs)
    cfg = copy.deepcopy(base)
    cfg.name = "phys_without_ratio_bounds"
    cfg.category = "physics"
    cfg.description = "Ablation: Without ACI/Eurocode ratio constraints (ordering Ts < Fs < Cs only)"
    cfg.loss.use_ratio_bounds = False
    cfg.loss.use_ts_le_cs = True
    cfg.loss.use_ts_le_fs = True
    registry[cfg.name] = cfg

    # ----------------------------------------------------
    # Category B: Missing-Target Handling
    # ----------------------------------------------------
    # B1: Complete-case training (listwise deletion of incomplete samples)
    cfg = copy.deepcopy(base)
    cfg.name = "missing_complete_case"
    cfg.category = "missing_data"
    cfg.description = "Ablation: Complete-case training (drop mixes with missing targets)"
    cfg.data.missing_strategy = "complete_case"
    registry[cfg.name] = cfg

    # B2: Mean target imputation on training split
    cfg = copy.deepcopy(base)
    cfg.name = "missing_mean_imputation"
    cfg.category = "missing_data"
    cfg.description = "Ablation: Mean target imputation on training data only"
    cfg.data.missing_strategy = "mean_imputation"
    registry[cfg.name] = cfg

    # B3: KNN target imputation on training split
    cfg = copy.deepcopy(base)
    cfg.name = "missing_knn_imputation"
    cfg.category = "missing_data"
    cfg.description = "Ablation: KNN target imputation (k=5) on training data only"
    cfg.data.missing_strategy = "knn_imputation"
    cfg.data.knn_neighbors = 5
    registry[cfg.name] = cfg

    # ----------------------------------------------------
    # Category C: Task Weighting
    # ----------------------------------------------------
    # C1: Equal task weighting [1.0, 1.0, 1.0]
    cfg = copy.deepcopy(base)
    cfg.name = "task_weights_equal"
    cfg.category = "task_weighting"
    cfg.description = "Ablation: Equal task weights [1.0, 1.0, 1.0] vs proposed [1.0, 1.2, 1.5]"
    cfg.loss.task_weights = (1.0, 1.0, 1.0)
    registry[cfg.name] = cfg

    # ----------------------------------------------------
    # Category D: Activation Functions
    # ----------------------------------------------------
    # D1: Standard ReLU
    cfg = copy.deepcopy(base)
    cfg.name = "act_relu"
    cfg.category = "activation"
    cfg.description = "Ablation: Standard ReLU activation instead of GELU"
    cfg.model.activation = "relu"
    registry[cfg.name] = cfg

    # D2: LeakyReLU
    cfg = copy.deepcopy(base)
    cfg.name = "act_leaky_relu"
    cfg.category = "activation"
    cfg.description = "Ablation: LeakyReLU (slope=0.01) activation instead of GELU"
    cfg.model.activation = "leaky_relu"
    registry[cfg.name] = cfg

    # ----------------------------------------------------
    # Category E: Normalization
    # ----------------------------------------------------
    # E1: Standard BatchNorm1d
    cfg = copy.deepcopy(base)
    cfg.name = "norm_batchnorm"
    cfg.category = "normalization"
    cfg.description = "Ablation: Standard PyTorch nn.BatchNorm1d instead of SafeBatchNorm1d"
    cfg.model.normalization = "batchnorm"
    registry[cfg.name] = cfg

    # E2: LayerNorm
    cfg = copy.deepcopy(base)
    cfg.name = "norm_layernorm"
    cfg.category = "normalization"
    cfg.description = "Ablation: nn.LayerNorm instead of SafeBatchNorm1d"
    cfg.model.normalization = "layernorm"
    registry[cfg.name] = cfg

    # ----------------------------------------------------
    # Category F: Feature Ablation
    # ----------------------------------------------------
    # F1: Remove NDT features (UPV_m/s and Er_ohm-cm) before preprocessing
    cfg = copy.deepcopy(base)
    cfg.name = "feat_no_ndt"
    cfg.category = "feature_ablation"
    cfg.description = "Ablation: Remove NDT features (UPV_m/s and Er_ohm-cm) before preprocessing (dim=17)"
    cfg.data.include_ndt = False
    registry[cfg.name] = cfg

    return registry


EXPERIMENT_REGISTRY = build_experiment_registry()
ALL_CATEGORIES = [
    "physics",
    "missing_data",
    "task_weighting",
    "activation",
    "normalization",
    "feature_ablation",
]

