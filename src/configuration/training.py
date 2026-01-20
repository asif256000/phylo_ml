from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(RuntimeError):
    """Raised when a training configuration cannot be parsed or validated."""


_ALLOWED_LABEL_TRANSFORMS = {"sqrt", "log"}


@dataclass(frozen=True)
class LabelTransformSettings:
    """Configuration for label pre/post-processing."""

    strategy: str = "sqrt"

    def __post_init__(self) -> None:
        if self.strategy not in _ALLOWED_LABEL_TRANSFORMS:
            raise ConfigurationError(
                f"'label_transform.strategy' must be one of {_ALLOWED_LABEL_TRANSFORMS}, received '{self.strategy}'"
            )


@dataclass(frozen=True)
class KANSettings:
    """Configuration for Kolmogorov-Arnold Network specific hyperparameters."""

    hidden_layers: tuple[int, ...] = field(default_factory=tuple)
    grid: int = 5
    spline_order: int = 3
    mult_arity: int = 2
    noise_scale: float = 0.3
    base_function: str = "silu"
    symbolic_enabled: bool = True
    affine_trainable: bool = False
    grid_eps: float = 0.02
    grid_range: tuple[float, float] = (-1.0, 1.0)
    sp_trainable: bool = True
    sb_trainable: bool = True
    sparse_init: bool = False
    auto_save: bool = False


@dataclass(frozen=True)
class ModelSettings:
    """Definition of architecture hyperparameters."""

    in_channels: int
    num_outputs: int | None = None
    num_taxa: int | None = None
    rooted: bool = True
    topology_classification: bool = False
    topology_weight: float = 1.0
    kan_settings: KANSettings | None = None


@dataclass(frozen=True)
class DataSettings:
    """Definition of dataset loading and DataLoader hyperparameters."""

    dataset_file: Path
    batch_size: int
    num_workers: int
    train_ratio: float
    val_ratio: float
    seed: int


@dataclass(frozen=True)
class TrainerSettings:
    """Training loop hyperparameters."""

    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class OutputSettings:
    """Configuration for artefacts produced during training."""

    results_dir: Path
    zoomed_plots: bool = False
    individual_branch_plots: bool = False
    branch_sum_plots: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Aggregated training configuration for phylogenetic models."""

    seed: int
    data: DataSettings
    trainer: TrainerSettings
    model: ModelSettings
    outputs: OutputSettings
    label_transform: LabelTransformSettings

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        base_path: Path | None = None,
    ) -> "TrainingConfig":
        seed = int(payload.get("seed", 42))

        data_settings = _parse_data_settings(payload, base_path)
        trainer_settings = _parse_trainer_settings(payload.get("trainer", {}))
        model_settings = _parse_model_settings(payload.get("model", {}))
        outputs_settings = _parse_output_settings(payload.get("outputs", {}), base_path)
        label_transform_settings = _parse_label_transform(payload.get("label_transform", "sqrt"))

        return cls(
            seed=seed,
            data=data_settings,
            trainer=trainer_settings,
            model=model_settings,
            outputs=outputs_settings,
            label_transform=label_transform_settings,
        )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a training configuration from a YAML or JSON file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Training configuration file not found: {config_path}")

    try:
        text = config_path.read_text()
    except OSError as exc:  # pragma: no cover - filesystem guard
        raise ConfigurationError(f"Failed to read configuration file: {config_path}") from exc

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigurationError("Training configuration must be in YAML or JSON format")

    if not isinstance(data, Mapping):
        raise ConfigurationError("Top-level training configuration must be a mapping")

    base_path = config_path.parent.resolve()
    return TrainingConfig.from_mapping(data, base_path=base_path)


def _parse_data_settings(payload: Mapping[str, Any], base_path: Path | None) -> DataSettings:
    data_payload = payload.get("data", {})
    if not isinstance(data_payload, Mapping):
        raise ConfigurationError("'data' section must be a mapping")

    dataset_value = data_payload.get("dataset_file") or payload.get("dataset_file")
    if dataset_value is None:
        raise ConfigurationError("'data.dataset_file' must be provided")
    dataset_path = Path(str(dataset_value)).expanduser()

    batch_size = int(data_payload.get("batch_size", 32))
    if batch_size <= 0:
        raise ConfigurationError("'data.batch_size' must be positive")

    num_workers = int(data_payload.get("num_workers", 0))
    if num_workers < 0:
        raise ConfigurationError("'data.num_workers' cannot be negative")

    train_ratio = float(data_payload.get("train_ratio", 0.70))
    val_ratio = float(data_payload.get("val_ratio", 0.15))
    if not 0 < train_ratio < 1:
        raise ConfigurationError("'data.train_ratio' must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ConfigurationError("'data.val_ratio' must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ConfigurationError("'data.train_ratio' + 'data.val_ratio' must be less than 1")

    seed = int(data_payload.get("seed", payload.get("seed", 42)))

    return DataSettings(
        dataset_file=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )


def _parse_trainer_settings(trainer_payload: Mapping[str, Any]) -> TrainerSettings:
    if not isinstance(trainer_payload, Mapping):
        raise ConfigurationError("'trainer' section must be a mapping")

    epochs = int(trainer_payload.get("epochs", 20))
    patience = int(trainer_payload.get("patience", 20))
    if epochs <= 0:
        raise ConfigurationError("'trainer.epochs' must be positive")
    if patience <= 0:
        raise ConfigurationError("'trainer.patience' must be positive")

    learning_rate = float(trainer_payload.get("learning_rate", 1e-3))
    if learning_rate <= 0:
        raise ConfigurationError("'trainer.learning_rate' must be positive")

    weight_decay = float(trainer_payload.get("weight_decay", 0.0))
    if weight_decay < 0:
        raise ConfigurationError("'trainer.weight_decay' cannot be negative")

    return TrainerSettings(
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def _parse_model_settings(model_payload: Mapping[str, Any]) -> ModelSettings:
    if not isinstance(model_payload, Mapping):
        raise ConfigurationError("'model' section must be a mapping")

    if "cnn" in model_payload:
        raise ConfigurationError("'model.cnn' is no longer supported; remove CNN-specific configuration")

    unsupported_keys = {"conv_layers", "linear_layers", "global_pool"}
    for key in unsupported_keys:
        if key in model_payload:
            raise ConfigurationError(f"'model.{key}' is no longer supported; remove CNN-specific configuration")

    kan_payload = model_payload.get("kan")
    if kan_payload is not None and not isinstance(kan_payload, Mapping):
        raise ConfigurationError("'model.kan' section must be a mapping when provided")

    in_channels_candidate = model_payload.get("in_channels")
    in_channels = int(in_channels_candidate if in_channels_candidate is not None else 4)
    if in_channels <= 0:
        raise ConfigurationError("'model.in_channels' must be positive")

    num_outputs = model_payload.get("num_outputs")
    if num_outputs is None:
        num_outputs_value: int | None = None
    else:
        num_outputs_value = int(num_outputs)
        if num_outputs_value <= 0:
            raise ConfigurationError("'model.num_outputs' must be positive when provided")

    num_taxa = model_payload.get("num_taxa")
    if num_taxa is None:
        num_taxa_value: int | None = None
    else:
        num_taxa_value = int(num_taxa)
        if num_taxa_value <= 0:
            raise ConfigurationError("'model.num_taxa' must be positive when provided")

    rooted = bool(model_payload.get("rooted", True))

    topology_classification = bool(model_payload.get("topology_classification", False))
    topology_weight = float(model_payload.get("topology_weight", 1.0))
    if topology_weight < 0:
        raise ConfigurationError("'model.topology_weight' cannot be negative")

    kan_settings = _parse_kan_settings(kan_payload) if kan_payload is not None else None

    return ModelSettings(
        in_channels=in_channels,
        num_outputs=num_outputs_value,
        num_taxa=num_taxa_value,
        rooted=rooted,
        topology_classification=topology_classification,
        topology_weight=topology_weight,
        kan_settings=kan_settings,
    )


def _parse_output_settings(
    output_payload: Mapping[str, Any],
    base_path: Path | None,
) -> OutputSettings:
    if not isinstance(output_payload, Mapping):
        raise ConfigurationError("'outputs' section must be a mapping")

    results_dir_value = output_payload.get("results_dir")
    if results_dir_value is None:
        results_dir = _default_results_dir(base_path)
    else:
        results_dir = Path(str(results_dir_value)).expanduser()
    zoomed_plots = bool(output_payload.get("zoomed_plots", False))
    individual_branch_plots = bool(output_payload.get("individual_branch_plots", False))
    branch_sum_plots = bool(output_payload.get("branch_sum_plots", False))

    return OutputSettings(
        results_dir=results_dir,
        zoomed_plots=zoomed_plots,
        individual_branch_plots=individual_branch_plots,
        branch_sum_plots=branch_sum_plots,
    )


def _parse_label_transform(value: Any) -> LabelTransformSettings:
    if isinstance(value, Mapping):
        strategy = str(value.get("strategy", "sqrt")).lower()
        return LabelTransformSettings(strategy=strategy)

    if isinstance(value, str):
        return LabelTransformSettings(strategy=value.lower())

    raise ConfigurationError("'label_transform' must be a string or mapping with a 'strategy' key")


def _parse_kan_settings(payload: Mapping[str, Any]) -> KANSettings:
    if not isinstance(payload, Mapping):
        raise ConfigurationError("'model.kan' must be a mapping when provided")

    hidden_layers_value = payload.get("hidden_layers", (128, 64))
    if isinstance(hidden_layers_value, Sequence) and not isinstance(hidden_layers_value, (str, bytes)):
        hidden_layers = tuple(int(v) for v in hidden_layers_value)
    else:
        raise ConfigurationError("'model.kan.hidden_layers' must be a sequence of integers")
    if not hidden_layers:
        raise ConfigurationError("'model.kan.hidden_layers' must contain at least one hidden layer size")
    if any(size <= 0 for size in hidden_layers):
        raise ConfigurationError("'model.kan.hidden_layers' values must be positive")

    grid = int(payload.get("grid", 5))
    if grid <= 0:
        raise ConfigurationError("'model.kan.grid' must be positive")

    spline_order = int(payload.get("spline_order", 3))
    if spline_order <= 0:
        raise ConfigurationError("'model.kan.spline_order' must be positive")

    mult_arity = int(payload.get("mult_arity", 2))
    if mult_arity <= 0:
        raise ConfigurationError("'model.kan.mult_arity' must be positive")

    noise_scale = float(payload.get("noise_scale", 0.3))
    if noise_scale < 0:
        raise ConfigurationError("'model.kan.noise_scale' cannot be negative")

    base_function = str(payload.get("base_function", "silu"))

    symbolic_enabled = bool(payload.get("symbolic_enabled", True))
    affine_trainable = bool(payload.get("affine_trainable", False))
    sparse_init = bool(payload.get("sparse_init", False))
    auto_save = bool(payload.get("auto_save", False))

    grid_eps = float(payload.get("grid_eps", 0.02))
    if not 0 <= grid_eps <= 1:
        raise ConfigurationError("'model.kan.grid_eps' must be between 0 and 1")

    grid_range_value = payload.get("grid_range", (-1.0, 1.0))
    try:
        grid_range = tuple(float(v) for v in grid_range_value)
    except (TypeError, ValueError):
        raise ConfigurationError("'model.kan.grid_range' must contain numeric values")
    if len(grid_range) != 2:
        raise ConfigurationError("'model.kan.grid_range' must contain exactly two values")
    if grid_range[0] >= grid_range[1]:
        raise ConfigurationError("'model.kan.grid_range' lower bound must be less than upper bound")

    sp_trainable = bool(payload.get("sp_trainable", True))
    sb_trainable = bool(payload.get("sb_trainable", True))

    return KANSettings(
        hidden_layers=hidden_layers,
        grid=grid,
        spline_order=spline_order,
        mult_arity=mult_arity,
        noise_scale=noise_scale,
        base_function=base_function,
        symbolic_enabled=symbolic_enabled,
        affine_trainable=affine_trainable,
        grid_eps=grid_eps,
        grid_range=grid_range,
        sp_trainable=sp_trainable,
        sb_trainable=sb_trainable,
        sparse_init=sparse_init,
        auto_save=auto_save,
    )


def _resolve_path(value: Any, base_path: Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path

    root_dir = _discover_workspace_root(base_path)
    return (root_dir / path).resolve()


def _discover_workspace_root(base_path: Path | None) -> Path:
    if base_path is None:
        return Path.cwd()

    base = base_path.resolve()
    if base.is_file():
        base = base.parent

    candidates = [base, *base.parents]
    markers = ("src", "npy_data", "xml_data", "branch_plots", "config")

    for candidate in candidates:
        try:
            if any((candidate / marker).exists() for marker in markers):
                return candidate
        except PermissionError:  # pragma: no cover - filesystem guard
            continue

    return base


def _default_results_dir(base_path: Path | None) -> Path:
    root_dir = _discover_workspace_root(base_path)
    base_results = root_dir / "latest_results"
    return _next_available_results_dir(base_results)


def _next_available_results_dir(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    parent = base_path.parent
    base_name = base_path.name
    existing_suffixes = [0]

    if parent.exists():
        for item in parent.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if name == base_name:
                existing_suffixes.append(0)
            elif name.startswith(base_name + "_"):
                suffix_part = name[len(base_name) + 1 :]
                if suffix_part.isdigit():
                    existing_suffixes.append(int(suffix_part))

    next_suffix = max(existing_suffixes) + 1
    return parent / f"{base_name}_{next_suffix}"
