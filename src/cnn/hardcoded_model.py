from __future__ import annotations

import torch
import torch.nn as nn

from src.utils import format_tuple, infer_num_outputs

from .config import ModelSettings


class CNNModel(nn.Module):
    """Hardcoded CNN that mirrors the reference training architecture."""

    def __init__(
        self,
        *,
        num_taxa: int,
        num_outputs: int | None = None,
        in_channels: int = 4,
        label_transform: str | None = None,
        tree_rooted: bool = True,
        num_topology_classes: int | None = None,
        topology_classification: bool = False,
    ) -> None:
        super().__init__()
        if num_taxa <= 0:
            raise ValueError("num_taxa must be positive")

        resolved_outputs = num_outputs or infer_num_outputs(num_taxa, rooted=tree_rooted)
        if resolved_outputs <= 0:
            raise ValueError("num_outputs must be positive")

        self.num_taxa = num_taxa
        self.num_outputs = resolved_outputs
        self.in_channels = in_channels
        self.label_transform_strategy = label_transform or "none"
        self.tree_rooted = tree_rooted
        self.topology_classification = topology_classification
        self.num_topology_classes = num_topology_classes

        conv_specs = [
            {
                "out_channels": 64,
                "kernel_size": (num_taxa, 1),
                "stride": (1, 1),
                "padding": (0, 0),
                "activation": nn.ReLU(),
                "pool": nn.Identity(),
            },
            {
                "out_channels": 128,
                "kernel_size": (1, 3),
                "stride": (1, 1),
                "padding": (0, 1),
                "activation": nn.ReLU(),
                "pool": nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            },
        ]

        self.conv_layers = nn.ModuleList()
        in_ch = in_channels
        for spec in conv_specs:
            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=spec["out_channels"],
                kernel_size=spec["kernel_size"],
                stride=spec["stride"],
                padding=spec["padding"],
            )
            block = nn.ModuleDict({
                "conv": conv,
                "activation": spec["activation"],
                "pool": spec["pool"],
            })
            self.conv_layers.append(block)
            in_ch = spec["out_channels"]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        linear_specs = [
            {"out_features": 256, "activation": nn.ReLU(), "dropout": 0.0},
            {"out_features": 256, "activation": nn.ReLU(), "dropout": 0.2},
        ]

        self.linear_layers = nn.ModuleList()
        in_features = in_ch
        for spec in linear_specs:
            linear = nn.Linear(in_features, spec["out_features"])
            activation = spec["activation"]
            dropout_module: nn.Module = nn.Dropout(spec["dropout"]) if spec["dropout"] > 0 else nn.Identity()
            block = nn.ModuleDict({
                "linear": linear,
                "activation": activation,
                "dropout": dropout_module,
            })
            self.linear_layers.append(block)
            in_features = spec["out_features"]

        self.output_layer = nn.Linear(in_features, resolved_outputs)

        self.topology_head: nn.Module | None = None
        if self.topology_classification:
            if self.num_topology_classes is None or self.num_topology_classes <= 0:
                raise ValueError("num_topology_classes must be positive when topology_classification is True")
            self.topology_head = nn.Linear(in_features, self.num_topology_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        for block in self.conv_layers:
            if not isinstance(block, nn.ModuleDict):  # defensive guard for type checkers
                raise RuntimeError("Unexpected conv block type")
            conv = block["conv"]
            activation = block["activation"]
            pool = block["pool"]
            if not isinstance(conv, nn.Conv2d):
                raise RuntimeError("Unexpected conv layer type")
            if not isinstance(activation, nn.Module):
                raise RuntimeError("Unexpected activation layer type")
            if not isinstance(pool, nn.Module):
                raise RuntimeError("Unexpected pooling layer type")
            x = conv(x)
            x = activation(x)
            x = pool(x)

        x = self.global_pool(x)
        x = self.flatten(x)

        for block in self.linear_layers:
            if not isinstance(block, nn.ModuleDict):  # defensive guard for type checkers
                raise RuntimeError("Unexpected linear block type")
            linear = block["linear"]
            activation = block["activation"]
            dropout = block["dropout"]
            if not isinstance(linear, nn.Linear):
                raise RuntimeError("Unexpected linear layer type")
            if not isinstance(activation, nn.Module):
                raise RuntimeError("Unexpected activation layer type")
            if not isinstance(dropout, nn.Module):
                raise RuntimeError("Unexpected dropout layer type")
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
        rooted: bool | None = None,
        label_transform: str | None = None,
    ) -> "CNNModel":
        resolved_rooted = model_settings.rooted if rooted is None else rooted
        resolved_outputs = num_outputs
        if resolved_outputs is None:
            resolved_outputs = model_settings.num_outputs or infer_num_outputs(num_taxa, rooted=resolved_rooted)
        return cls(
            num_taxa=num_taxa,
            num_outputs=resolved_outputs,
            in_channels=model_settings.in_channels,
            label_transform=label_transform,
            tree_rooted=resolved_rooted,
            num_topology_classes=model_settings.num_topology_classes,
            topology_classification=model_settings.topology_classification,
        )

    def __str__(self) -> str:
        lines: list[str] = []
        lines.append("Hardcoded CNNModel architecture:")
        lines.append(
            f"  Inputs: in_channels={self.in_channels}, num_taxa={self.num_taxa}, rooted={self.tree_rooted}"
        )
        lines.append(f"  Label transform: {self.label_transform_strategy}")
        lines.append(f"  Outputs: num_outputs={self.num_outputs}")
        lines.append("  Convolutional Layers:")
        for idx, block in enumerate(self.conv_layers, start=1):
            if not isinstance(block, nn.ModuleDict):
                raise RuntimeError("Unexpected conv block type")
            conv = block["conv"]
            activation = block["activation"]
            pool = block["pool"]
            if not isinstance(conv, nn.Conv2d):
                raise RuntimeError("Unexpected conv layer type")
            if not isinstance(activation, nn.Module):
                raise RuntimeError("Unexpected activation layer type")
            if not isinstance(pool, nn.Module):
                raise RuntimeError("Unexpected pooling layer type")
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
            if isinstance(pool, nn.Identity):
                lines.append("      pool: Identity")
            elif hasattr(pool, "kernel_size") and hasattr(pool, "stride"):
                lines.append(
                    "      pool: {name}(kernel_size={kernel}, stride={stride})".format(
                        name=pool.__class__.__name__,
                        kernel=format_tuple(getattr(pool, "kernel_size", None)),
                        stride=format_tuple(getattr(pool, "stride", None)),
                    )
                )
            else:
                lines.append(f"      pool: {pool.__class__.__name__}")

        lines.append(f"  Global pooling: {self.global_pool.__class__.__name__}")

        lines.append("  Linear Layers:")
        for idx, block in enumerate(self.linear_layers, start=1):
            if not isinstance(block, nn.ModuleDict):
                raise RuntimeError("Unexpected linear block type")
            linear = block["linear"]
            activation = block["activation"]
            dropout = block["dropout"]
            if not isinstance(linear, nn.Linear):
                raise RuntimeError("Unexpected linear layer type")
            if not isinstance(activation, nn.Module):
                raise RuntimeError("Unexpected activation layer type")
            if not isinstance(dropout, nn.Module):
                raise RuntimeError("Unexpected dropout layer type")
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
        if self.topology_head is not None and isinstance(self.topology_head, nn.Linear):
            lines.append(
                "  Topology head: Linear(in_features={in_f}, out_features={out_f})".format(
                    in_f=self.topology_head.in_features,
                    out_f=self.topology_head.out_features,
                )
            )
        return "\n".join(lines)
