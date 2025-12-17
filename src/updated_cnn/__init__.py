"""Updated CNN module for regression-only training."""

from .model import CNNModel
from .train import (
    Trainer,
    TrainingConfig,
    TrainingResult,
    build_config_from_mapping,
    run_training,
)

__all__ = [
    "CNNModel",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "build_config_from_mapping",
    "run_training",
]
