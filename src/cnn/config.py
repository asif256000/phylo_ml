from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(RuntimeError):
    """Raised when a CNN training configuration cannot be parsed or validated."""


_ALLOWED_ACTIVATIONS = {"relu", "gelu", "elu", "identity"}
_ALLOWED_GLOBAL_POOLS = {"adaptive_avg", "adaptive_max", "identity"}
_ALLOWED_POOL_KINDS = {"identity", "max", "avg"}
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
class PoolingSettings:
    """Configuration for an optional pooling layer following a convolution."""

    kind: str = "identity"
    kernel_size: tuple[int, int] | None = None
    stride: tuple[int, int] | None = None


@dataclass(frozen=True)
class ConvLayerSettings:
    """Settings describing a single convolutional block."""

    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    activation: str = "relu"
    pool: PoolingSettings = field(default_factory=PoolingSettings)


@dataclass(frozen=True)
class LinearLayerSettings:
    """Settings describing a fully connected block before the output layer."""

    out_features: int
    activation: str = "relu"
    dropout: float = 0.0


@dataclass(frozen=True)
class ModelSettings:
    """Definition of the CNN architecture hyperparameters."""

    in_channels: int
    conv_layers: tuple[ConvLayerSettings, ...]
    linear_layers: tuple[LinearLayerSettings, ...]
    global_pool: str
    num_outputs: int | None = None
    num_taxa: int | None = None
    rooted: bool = True
    topology_classification: bool = False
    topology_weight: float = 1.0


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

    branch_plot_dir: Path
    zoomed_plots: bool = False
    individual_branch_plots: bool = False
    branch_sum_plots: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Aggregated training configuration for the CNN pipeline."""

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
    """Load the CNN training configuration from a YAML or JSON file."""

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
    # Read dataset path as provided, without resolving relative to workspace
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

    in_channels = int(model_payload.get("in_channels", 4))
    if in_channels <= 0:
        raise ConfigurationError("'model.in_channels' must be positive")

    conv_layers_payload = model_payload.get("conv_layers")
    if conv_layers_payload is None:
        conv_layers = _default_conv_layers()
    else:
        conv_layers = _parse_conv_layers(conv_layers_payload)

    linear_layers_payload = model_payload.get("linear_layers")
    if linear_layers_payload is None:
        linear_layers = _default_linear_layers()
    else:
        linear_layers = _parse_linear_layers(linear_layers_payload)

    global_pool = str(model_payload.get("global_pool", "adaptive_avg")).lower()
    if global_pool not in _ALLOWED_GLOBAL_POOLS:
        raise ConfigurationError(
            f"'model.global_pool' must be one of {_ALLOWED_GLOBAL_POOLS}, received '{global_pool}'"
        )

    num_outputs = model_payload.get("num_outputs")
    num_outputs_value: int | None
    if num_outputs is None:
        num_outputs_value = None
    else:
        num_outputs_value = int(num_outputs)
        if num_outputs_value <= 0:
            raise ConfigurationError("'model.num_outputs' must be positive when provided")

    num_taxa = model_payload.get("num_taxa")
    num_taxa_value: int | None
    if num_taxa is None:
        num_taxa_value = None
    else:
        num_taxa_value = int(num_taxa)
        if num_taxa_value <= 0:
            raise ConfigurationError("'model.num_taxa' must be positive when provided")

    rooted = bool(model_payload.get("rooted", True))

    topology_classification = bool(model_payload.get("topology_classification", False))
    topology_weight = float(model_payload.get("topology_weight", 1.0))
    if topology_weight < 0:
        raise ConfigurationError("'model.topology_weight' cannot be negative")

    return ModelSettings(
        in_channels=in_channels,
        conv_layers=conv_layers,
        linear_layers=linear_layers,
        global_pool=global_pool,
        num_outputs=num_outputs_value,
        num_taxa=num_taxa_value,
        rooted=rooted,
        topology_classification=topology_classification,
        topology_weight=topology_weight,
    )


def _parse_output_settings(
    output_payload: Mapping[str, Any],
    base_path: Path | None,
) -> OutputSettings:
    if not isinstance(output_payload, Mapping):
        raise ConfigurationError("'outputs' section must be a mapping")

    branch_plot_dir_value = output_payload.get("branch_plot_dir", "branch_plots")
    # Read branch plot directory as provided, without resolving relative to workspace
    branch_plot_dir = Path(str(branch_plot_dir_value)).expanduser()
    zoomed_plots = bool(output_payload.get("zoomed_plots", False))
    individual_branch_plots = bool(output_payload.get("individual_branch_plots", False))
    branch_sum_plots = bool(output_payload.get("branch_sum_plots", False))

    return OutputSettings(
        branch_plot_dir=branch_plot_dir,
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


def _parse_conv_layers(data: Sequence[Any]) -> tuple[ConvLayerSettings, ...]:
    if not isinstance(data, Sequence):
        raise ConfigurationError("'model.conv_layers' must be a sequence")

    layers: list[ConvLayerSettings] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, Mapping):
            raise ConfigurationError("Each convolution layer definition must be a mapping")

        try:
            out_channels = int(entry["out_channels"])
        except KeyError as exc:
            raise ConfigurationError("Convolution layer missing 'out_channels'") from exc
        if out_channels <= 0:
            raise ConfigurationError("'out_channels' must be positive for every convolution layer")

        kernel_size = _parse_pair(entry.get("kernel_size"), "kernel_size", required=True)
        if kernel_size is None:
            raise ConfigurationError("'kernel_size' must be provided for every convolution layer")
        stride = _parse_pair(entry.get("stride"), "stride", default=(1, 1), required=False) or (1, 1)
        padding = _parse_pair(entry.get("padding"), "padding", default=(0, 0), required=False) or (0, 0)

        activation = str(entry.get("activation", "relu")).lower()
        if activation not in _ALLOWED_ACTIVATIONS:
            raise ConfigurationError(
                f"Unsupported activation '{activation}' in convolution layer {index}; allowed values: {_ALLOWED_ACTIVATIONS}"
            )

        pool = _parse_pooling(entry.get("pool"))

        layers.append(
            ConvLayerSettings(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                pool=pool,
            )
        )

    if not layers:
        raise ConfigurationError("'model.conv_layers' must define at least one layer")

    return tuple(layers)


def _parse_linear_layers(data: Sequence[Any]) -> tuple[LinearLayerSettings, ...]:
    if not isinstance(data, Sequence):
        raise ConfigurationError("'model.linear_layers' must be a sequence")

    layers: list[LinearLayerSettings] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, Mapping):
            raise ConfigurationError("Each linear layer definition must be a mapping")

        try:
            out_features = int(entry["out_features"])
        except KeyError as exc:
            raise ConfigurationError("Linear layer missing 'out_features'") from exc
        if out_features <= 0:
            raise ConfigurationError("'out_features' must be positive for every linear layer")

        activation = str(entry.get("activation", "relu")).lower()
        if activation not in _ALLOWED_ACTIVATIONS:
            raise ConfigurationError(
                f"Unsupported activation '{activation}' in linear layer {index}; allowed values: {_ALLOWED_ACTIVATIONS}"
            )

        dropout = float(entry.get("dropout", 0.0))
        if dropout < 0 or dropout >= 1:
            raise ConfigurationError("'dropout' must be in the interval [0, 1)")

        layers.append(
            LinearLayerSettings(
                out_features=out_features,
                activation=activation,
                dropout=dropout,
            )
        )

    if not layers:
        raise ConfigurationError("'model.linear_layers' must define at least one layer")

    return tuple(layers)


def _parse_pooling(pool_payload: Any) -> PoolingSettings:
    if pool_payload is None:
        return PoolingSettings()
    if isinstance(pool_payload, str):
        kind = pool_payload.lower()
        if kind not in _ALLOWED_POOL_KINDS:
            raise ConfigurationError(
                f"Pooling kind '{kind}' is not supported; allowed values: {_ALLOWED_POOL_KINDS}"
            )
        return PoolingSettings(kind=kind)
    if not isinstance(pool_payload, Mapping):
        raise ConfigurationError("'pool' definition must be a string or mapping")

    kind = str(pool_payload.get("type", pool_payload.get("kind", "identity"))).lower()
    if kind not in _ALLOWED_POOL_KINDS:
        raise ConfigurationError(
            f"Pooling kind '{kind}' is not supported; allowed values: {_ALLOWED_POOL_KINDS}"
        )

    kernel_size = _parse_pair(pool_payload.get("kernel_size"), "pool.kernel_size", required=False)
    stride = _parse_pair(pool_payload.get("stride"), "pool.stride", required=False)

    return PoolingSettings(kind=kind, kernel_size=kernel_size, stride=stride)


def _parse_pair(
    value: Any,
    field_name: str,
    default: tuple[int, int] | None = None,
    *,
    required: bool,
) -> tuple[int, int] | None:
    if value is None:
        if required:
            raise ConfigurationError(f"'{field_name}' must be provided")
        return default

    if isinstance(value, (int, float)):
        int_value = int(value)
        return (int_value, int_value)

    if not isinstance(value, Sequence):
        raise ConfigurationError(f"'{field_name}' must be a sequence of two integers")

    if len(value) != 2:
        raise ConfigurationError(f"'{field_name}' must contain exactly two integers")

    first = int(value[0])
    second = int(value[1])
    return (first, second)


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


def _default_conv_layers() -> tuple[ConvLayerSettings, ...]:
    return (
        ConvLayerSettings(
            out_channels=64,
            kernel_size=(-1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation="relu",
            pool=PoolingSettings(kind="identity"),
        ),
        ConvLayerSettings(
            out_channels=128,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            activation="relu",
            pool=PoolingSettings(kind="max", kernel_size=(1, 2), stride=(1, 2)),
        ),
    )


def _default_linear_layers() -> tuple[LinearLayerSettings, ...]:
    return (
        LinearLayerSettings(out_features=256, activation="relu", dropout=0.0),
        LinearLayerSettings(out_features=256, activation="relu", dropout=0.2),
    )
