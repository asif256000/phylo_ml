"""Configuration entrypoint for the cnn package.

This module re-exports the shared training configuration so that consumers
can import everything they need from `cnn` without reaching into other
packages.
"""

from src.configuration.training import (
    ConfigurationError,
    DataSettings,
    LabelTransformSettings,
    ModelSettings,
    OutputSettings,
    TrainerSettings,
    TrainingConfig,
    load_training_config,
)

__all__ = [
    "ConfigurationError",
    "DataSettings",
    "LabelTransformSettings",
    "ModelSettings",
    "OutputSettings",
    "TrainerSettings",
    "TrainingConfig",
    "load_training_config",
]
