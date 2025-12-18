from __future__ import annotations

import torch
import torch.nn as nn

from src.utils import format_tuple


class CNNModel(nn.Module):
    """Hardcoded CNN for branch-length regression only (no classification head).

    Architecture is fixed (two conv blocks, two linear blocks) and configured only by
    input channel count, taxa count, and output size. This keeps YAML model settings
    simple: only `in_channels`, `rooted`, and `num_outputs` are consumed.
    """

    def __init__(
        self,
        *,
        num_taxa: int,
        num_outputs: int,
        in_channels: int = 4,
        label_transform: str | None = None,
        tree_rooted: bool = True,
    ) -> None:
        super().__init__()
        if num_taxa <= 0:
            raise ValueError("num_taxa must be positive")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")

        self.num_taxa = num_taxa
        self.num_outputs = num_outputs
        self.in_channels = in_channels
        self.label_transform_strategy = label_transform or "none"
        self.tree_rooted = tree_rooted

        # Convolutional stack (explicit modules, no ModuleList)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=(num_taxa, 1),
            stride=(num_taxa, 1),
            # padding=(0, 0),
        )
        self.act1 = nn.ReLU()
        self.pool1 = nn.Identity()

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(1, 3),
            stride=(1, 1),
            # padding=(0, 1),
        )
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Fully connected stack
        self.fc1 = nn.Linear(64, 256)
        self.fc1_act = nn.ReLU()
        self.fc1_drop = nn.Identity()

        self.fc2 = nn.Linear(256, 256)
        self.fc2_act = nn.ReLU()
        self.fc2_drop = nn.Dropout(0.2)

        self.output_layer = nn.Linear(256, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1_drop(self.fc1_act(self.fc1(x)))
        x = self.fc2_drop(self.fc2_act(self.fc2(x)))
        return self.output_layer(x)

    @classmethod
    def from_data_shapes(
        cls,
        *,
        encoded_shape: tuple[int, int, int],
        y_br_shape: tuple[int, ...],
        label_transform: str | None = None,
        tree_rooted: bool = True,
        in_channels: int | None = None,
        num_outputs: int | None = None,
    ) -> "CNNModel":
        if len(encoded_shape) != 3:
            raise ValueError("encoded_shape must be (taxa, seq_length, channels)")
        num_taxa, _, num_channels = encoded_shape
        outputs = num_outputs or y_br_shape[-1]
        channels = in_channels or num_channels
        return cls(
            num_taxa=num_taxa,
            num_outputs=outputs,
            in_channels=channels,
            label_transform=label_transform,
            tree_rooted=tree_rooted,
        )

    def __str__(self) -> str:
        return "\n".join(
            [
                "CNNModel architecture (regression only):",
                f"  Inputs: in_channels={self.in_channels}, num_taxa={self.num_taxa}, rooted={self.tree_rooted}",
                f"  Label transform: {self.label_transform_strategy}",
                f"  Outputs: num_outputs={self.num_outputs}",
                "  Convolutional Layers:",
                "    conv1: Conv2d(in_channels={in_c}, out_channels=64, kernel_size={kernel}, stride={stride}, padding={padding})".format(
                    in_c=self.conv1.in_channels,
                    kernel=format_tuple(self.conv1.kernel_size),
                    stride=format_tuple(self.conv1.stride),
                    padding=format_tuple(self.conv1.padding),
                ),
                f"      activation: {self.act1.__class__.__name__}",
                f"      pool: {self.pool1.__class__.__name__}",
                "    conv2: Conv2d(in_channels=64, out_channels=128, kernel_size={kernel}, stride={stride}, padding={padding})".format(
                    kernel=format_tuple(self.conv2.kernel_size),
                    stride=format_tuple(self.conv2.stride),
                    padding=format_tuple(self.conv2.padding),
                ),
                f"      activation: {self.act2.__class__.__name__}",
                "      pool: {name}(kernel_size={kernel}, stride={stride})".format(
                    name=self.pool2.__class__.__name__,
                    kernel=format_tuple(getattr(self.pool2, "kernel_size", ())),
                    stride=format_tuple(getattr(self.pool2, "stride", ())),
                ),
                f"  Global pooling: {self.global_pool.__class__.__name__}",
                "  Linear Layers:",
                "    fc1: Linear(in_features={in_f}, out_features={out_f})".format(
                    in_f=self.fc1.in_features,
                    out_f=self.fc1.out_features,
                ),
                f"      activation: {self.fc1_act.__class__.__name__}",
                f"      dropout: {self.fc1_drop.__class__.__name__}",
                "    fc2: Linear(in_features={in_f}, out_features={out_f})".format(
                    in_f=self.fc2.in_features,
                    out_f=self.fc2.out_features,
                ),
                f"      activation: {self.fc2_act.__class__.__name__}",
                f"      dropout: {self.fc2_drop.__class__.__name__}",
                "  Output layer: Linear(in_features={in_f}, out_features={out_f})".format(
                    in_f=self.output_layer.in_features,
                    out_f=self.output_layer.out_features,
                ),
            ]
        )
