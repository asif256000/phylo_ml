from __future__ import annotations

import torch
import torch.nn as nn

from src.utils import format_tuple, infer_num_outputs

from .config import ModelSettings, PoolingSettings


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "identity": nn.Identity,
}


def _build_activation(name: str) -> nn.Module:
    try:
        activation_cls = _ACTIVATIONS[name]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported activation '{name}'") from exc
    return activation_cls()


def _build_pooling(settings: PoolingSettings) -> nn.Module:
    if settings.kind == "identity":
        return nn.Identity()

    kernel_size = settings.kernel_size or (1, 1)
    stride = settings.stride or kernel_size

    if settings.kind == "max":
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    if settings.kind == "avg":
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    raise ValueError(f"Unsupported pooling kind '{settings.kind}'")  # pragma: no cover


def _build_global_pool(kind: str) -> nn.Module:
    if kind == "identity":
        return nn.Identity()
    if kind == "adaptive_avg":
        return nn.AdaptiveAvgPool2d(1)
    if kind == "adaptive_max":
        return nn.AdaptiveMaxPool2d(1)
    raise ValueError(f"Unsupported global pooling '{kind}'")  # pragma: no cover


def _resolve_kernel_size(candidate: tuple[int, int], num_taxa: int) -> tuple[int, int]:
    height, width = candidate
    if height == -1:
        height = num_taxa
    if width <= 0:
        raise ValueError("Kernel width must be positive")
    if height <= 0:
        raise ValueError("Kernel height must be positive")
    return height, width


def _resolve_padding(candidate: tuple[int, int]) -> tuple[int, int]:
    return int(candidate[0]), int(candidate[1])


def _resolve_stride(candidate: tuple[int, int]) -> tuple[int, int]:
    return int(candidate[0]), int(candidate[1])


class CNNModel(nn.Module):
    """Configurable CNN that predicts phylogenetic branch lengths."""

    def __init__(
        self,
        model_settings: ModelSettings,
        *,
        num_taxa: int,
        num_outputs: int,
        num_topology_classes: int | None = None,
        label_transform: str | None = None,
        tree_rooted: bool = True,
    ) -> None:
        super().__init__()
        if num_taxa <= 0:
            raise ValueError("num_taxa must be positive")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")

        self.model_settings = model_settings
        self.num_taxa = num_taxa
        self.num_outputs = num_outputs
        self.num_topology_classes = num_topology_classes
        self.label_transform_strategy = label_transform or "none"
        self.tree_rooted = tree_rooted
        self.topology_classification = model_settings.topology_classification

        self.conv_layers = nn.ModuleList()
        in_channels = model_settings.in_channels
        for layer_cfg in model_settings.conv_layers:
            kernel_size = _resolve_kernel_size(layer_cfg.kernel_size, num_taxa)
            stride = _resolve_stride(layer_cfg.stride)
            padding = _resolve_padding(layer_cfg.padding)

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_cfg.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            activation = _build_activation(layer_cfg.activation)
            pooling = _build_pooling(layer_cfg.pool)

            block = nn.ModuleDict({
                "conv": conv,
                "activation": activation,
                "pool": pooling,
            })
            self.conv_layers.append(block)
            in_channels = layer_cfg.out_channels

        self.global_pool = _build_global_pool(model_settings.global_pool)
        self.flatten = nn.Flatten()

        self.linear_layers = nn.ModuleList()
        in_features = in_channels
        for layer_cfg in model_settings.linear_layers:
            linear = nn.Linear(in_features, layer_cfg.out_features)
            activation = _build_activation(layer_cfg.activation)
            dropout_module: nn.Module = nn.Dropout(layer_cfg.dropout) if layer_cfg.dropout > 0 else nn.Identity()

            block = nn.ModuleDict({
                "linear": linear,
                "activation": activation,
                "dropout": dropout_module,
            })
            self.linear_layers.append(block)
            in_features = layer_cfg.out_features

        self.output_layer = nn.Linear(in_features, num_outputs)
        
        if self.topology_classification:
            if num_topology_classes is None or num_topology_classes <= 0:
                raise ValueError("num_topology_classes must be positive when topology_classification is enabled")
            self.topology_head = nn.Linear(in_features, num_topology_classes)
        else:
            self.topology_head = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        for block in self.conv_layers:
            if not isinstance(block, nn.ModuleDict):  # pragma: no cover - defensive
                raise RuntimeError("Unexpected convolutional block type")
            conv = block["conv"]
            activation = block["activation"]
            pool = block["pool"]
            if not isinstance(conv, nn.Conv2d) or not isinstance(activation, nn.Module) or not isinstance(pool, nn.Module):
                raise RuntimeError("Malformed convolutional block configuration")
            x = conv(x)
            x = activation(x)
            x = pool(x)

        x = self.global_pool(x)
        x = self.flatten(x)

        for block in self.linear_layers:
            if not isinstance(block, nn.ModuleDict):  # pragma: no cover - defensive
                raise RuntimeError("Unexpected linear block type")
            linear = block["linear"]
            activation = block["activation"]
            dropout = block["dropout"]
            if not isinstance(linear, nn.Linear) or not isinstance(activation, nn.Module) or not isinstance(dropout, nn.Module):
                raise RuntimeError("Malformed linear block configuration")
            x = linear(x)
            x = activation(x)
            x = dropout(x)

        y_br = self.output_layer(x)
        y_top = self.topology_head(x) if self.topology_head is not None else None
        return y_br, y_top

    @classmethod
    def from_config(
        cls,
        model_settings: ModelSettings,
        *,
        num_taxa: int,
        num_outputs: int | None = None,
        num_topology_classes: int | None = None,
        rooted: bool | None = None,
        label_transform: str | None = None,
    ) -> "CNNModel":
        resolved_rooted = model_settings.rooted if rooted is None else rooted
        resolved_outputs = num_outputs
        if resolved_outputs is None:
            resolved_outputs = model_settings.num_outputs or infer_num_outputs(num_taxa, rooted=resolved_rooted)
        return cls(
            model_settings,
            num_taxa=num_taxa,
            num_outputs=resolved_outputs,
            num_topology_classes=num_topology_classes,
            label_transform=label_transform,
            tree_rooted=resolved_rooted,
        )

    def __str__(self) -> str:
        lines: list[str] = []
        lines.append("CNNModel architecture:")
        lines.append(
            f"  Inputs: in_channels={self.model_settings.in_channels}, num_taxa={self.num_taxa}, rooted={self.tree_rooted}"
        )
        lines.append(f"  Label transform: {self.label_transform_strategy}")
        lines.append(f"  Outputs: num_outputs={self.num_outputs}")
        lines.append("  Convolutional Blocks:")
        for idx, block in enumerate(self.conv_layers, start=1):
            if not isinstance(block, nn.ModuleDict):  # pragma: no cover - defensive
                raise RuntimeError("Unexpected convolutional block type")
            conv = block["conv"]
            pool = block["pool"]
            activation = block["activation"]
            if not isinstance(conv, nn.Conv2d) or not isinstance(pool, nn.Module) or not isinstance(activation, nn.Module):
                raise RuntimeError("Malformed convolutional block configuration")
            lines.append(
                "    conv{idx}: Conv2d(in_channels={in_c}, out_channels={out_c}, kernel_size={kernel}, stride={stride}, padding={padding})".format(
                    idx=idx,
                    in_c=conv.in_channels,
                    out_c=conv.out_channels,
                    kernel=format_tuple(conv.kernel_size),
                    stride=format_tuple(conv.stride),
                    padding=format_tuple(conv.padding),
                )
            )
            lines.append(f"      activation: {activation.__class__.__name__}")
            if not isinstance(pool, nn.Identity):
                if hasattr(pool, "kernel_size") and hasattr(pool, "stride"):
                    kernel = getattr(pool, "kernel_size", None)
                    stride = getattr(pool, "stride", None)
                    lines.append(
                        "      pool: {name}(kernel_size={kernel}, stride={stride})".format(
                            name=pool.__class__.__name__,
                            kernel=format_tuple(kernel),
                            stride=format_tuple(stride),
                        )
                    )
                else:
                    lines.append(f"      pool: {pool.__class__.__name__}")
            else:
                lines.append("      pool: Identity")

        lines.append(
            f"  Global pooling: {self.global_pool.__class__.__name__}"
        )

        lines.append("  Linear Blocks:")
        for idx, block in enumerate(self.linear_layers, start=1):
            if not isinstance(block, nn.ModuleDict):  # pragma: no cover - defensive
                raise RuntimeError("Unexpected linear block type")
            linear = block["linear"]
            activation = block["activation"]
            dropout = block["dropout"]
            if not isinstance(linear, nn.Linear) or not isinstance(activation, nn.Module) or not isinstance(dropout, nn.Module):
                raise RuntimeError("Malformed linear block configuration")
            lines.append(
                "    linear{idx}: Linear(in_features={in_f}, out_features={out_f})".format(
                    idx=idx,
                    in_f=linear.in_features,
                    out_f=linear.out_features,
                )
            )
            lines.append(f"      activation: {activation.__class__.__name__}")
            if isinstance(dropout, nn.Dropout):
                lines.append(f"      dropout: Dropout(p={dropout.p})")
            else:
                lines.append(f"      dropout: {dropout.__class__.__name__}")

        lines.append(
            "  Output layer: Linear(in_features={in_f}, out_features={out_f})".format(
                in_f=self.output_layer.in_features,
                out_f=self.output_layer.out_features,
            )
        )

        return "\n".join(lines)
