"""KAN module for training Kolmogorov-Arnold Networks on phylogenetic datasets."""

from .trainer import KANTrainer, TrainingResult, run_training

__all__ = [
    "KANTrainer",
    "TrainingResult",
    "run_training",
]
