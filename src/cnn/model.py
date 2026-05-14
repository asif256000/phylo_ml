from __future__ import annotations

import torch
import torch.nn as nn

from src.utils import format_tuple


class _CNNBase(nn.Module):
    """Shared CNN trunk for regression and topology classification."""

    def __init__(
        self,
        *,
        num_taxa: int,
        num_outputs: int,
        in_channels: int = 4,
        label_transform: str | None = None,
        tree_rooted: bool = True,
        topology_classification: bool = False,
        num_topology_classes: int | None = None,
    ) -> None:
        super().__init__()
        if num_taxa <= 0:
            raise ValueError("num_taxa must be positive")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")
        if topology_classification and (num_topology_classes is None or num_topology_classes <= 0):
            raise ValueError("num_topology_classes must be positive when topology classification is enabled")

        self.num_taxa = num_taxa
        self.num_outputs = num_outputs
        self.in_channels = in_channels
        self.label_transform_strategy = label_transform or "none"
        self.tree_rooted = tree_rooted
        self.topology_classification = topology_classification
        self.num_topology_classes = num_topology_classes

        # Convolutional stack (deeper + batch norm to improve topology signals)
        #TF: Conv2D(filters=64, kernel_size=(3, 1), activation="relu", padding="same")
        #trying kernel_size=(num_taxa, 2), and stride=(num_taxa, 1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(num_taxa, 1),
            # kernel_size=(num_taxa, 2),
            #stride=(num_taxa, 1),
             stride=(1, 1),
            #padding=(0, 1),
            padding="same",
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        #self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        #TF: x = Conv2D(filters=128, kernel_size=(3, 4), activation="relu", padding="same")(x)
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(num_taxa, 4),
            stride=(1, 1),
            #padding=(0, 1),
            padding="same",
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        #TF: x = MaxPooling2D(pool_size=(1, 2))(x)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) #stride=(1, 2) is equivalent to default when kernel_size=(1, 2)

        #x = Conv2D(filters=128, kernel_size=(3, 4), activation="relu", padding="same")(x)
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(num_taxa, 4),
            stride=(1, 1),
            #padding=(0, 1),
            padding="same",
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        #Tf: x = MaxPooling2D(pool_size=(1, 2))(x)
        #self.pool3 = nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2),stride=(1, 2))
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.feature_dim = 128

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        return self.flatten(x)


class ParallelCNNModel(_CNNBase):
    """Parallel regression + topology heads sharing a trunk."""
    def __init__(
        self,
        *,
        num_taxa: int,
        num_outputs: int,
        in_channels: int = 4,
        label_transform: str | None = None,
        tree_rooted: bool = True,
        topology_classification: bool = False,
        num_topology_classes: int | None = None,
    ) -> None:
        super().__init__(
            num_taxa=num_taxa,
            num_outputs=num_outputs,
            in_channels=in_channels,
            label_transform=label_transform,
            tree_rooted=tree_rooted,
            topology_classification=topology_classification,
            num_topology_classes=num_topology_classes,
        )
        self.architecture = "parallel"
        #TF: added the dropout_rate=0.1 
        self.dropout_rate = 0.1

        # Add shared dense layer (equivalent to TF's Dense(128, activation="relu"))
        self.shared_fc = nn.Linear(self.feature_dim, 128)  # self.feature_dim is 128 from convs
        self.shared_act = nn.ReLU()
        self.shared_drop = nn.Dropout(self.dropout_rate)  # Equivalent to TF's Dropout(dropout_rate)
        

        # Regression head (MLP)
        #self.reg_fc1 = nn.Linear(self.feature_dim, 256)
        self.reg_fc1 = nn.Linear(self.feature_dim, 128)
        self.reg_act1 = nn.ReLU()
        self.reg_drop1 = nn.Dropout(0.1)

        #self.reg_fc2 = nn.Linear(256, 128)
        #self.reg_act2 = nn.ReLU()
        #self.reg_drop2 = nn.Dropout(0.1)

        self.output_layer = nn.Linear(128, num_outputs)

        # Topology classification head (MLP)
        if topology_classification and num_topology_classes is not None:
            #self.top_fc1 = nn.Linear(self.feature_dim, 256)
            self.top_fc1 = nn.Linear(self.feature_dim, 128)
            self.top_act1 = nn.ReLU()
            self.top_drop1 = nn.Dropout(0.1)
            #self.top_fc2 = nn.Linear(256, 128)
            #self.top_act2 = nn.ReLU()
            #self.top_drop2 = nn.Dropout(0.1)
            self.topology_head = nn.Linear(128, num_topology_classes)
        else:
            self.top_fc1 = None
            self.top_act1 = None
            self.top_drop1 = None
            self.top_fc2 = None
            self.top_act2 = None
            self.top_drop2 = None
            self.topology_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(x)
        #Apply shared dense layer before branching to regression and topology heads
        x = self.shared_drop(self.shared_act(self.shared_fc(x)))
        reg = self.reg_drop1(self.reg_act1(self.reg_fc1(x)))
        #reg = self.reg_drop2(self.reg_act2(self.reg_fc2(reg)))
        y_br = self.output_layer(reg)
        if self.topology_head is None:
            return y_br
        top = self.top_drop1(self.top_act1(self.top_fc1(x)))
        #top = self.top_drop2(self.top_act2(self.top_fc2(top)))
        y_top = self.topology_head(top)
        #TF: add the softmax activation to the topology head output for classification
        y_top = torch.softmax(y_top, dim=1)
        return y_br, y_top

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


class SerialCNNModel(_CNNBase):
    """Serial regression -> topology classification (classification sees regression output)."""

    def __init__(
        self,
        *,
        num_taxa: int,
        num_outputs: int,
        in_channels: int = 4,
        label_transform: str | None = None,
        tree_rooted: bool = True,
        topology_classification: bool = False,
        num_topology_classes: int | None = None,
    ) -> None:
        super().__init__(
            num_taxa=num_taxa,
            num_outputs=num_outputs,
            in_channels=in_channels,
            label_transform=label_transform,
            tree_rooted=tree_rooted,
            topology_classification=topology_classification,
            num_topology_classes=num_topology_classes,
        )
        self.architecture = "serial"

        # Regression head (MLP)
        self.reg_fc1 = nn.Linear(self.feature_dim, 256)
        self.reg_act1 = nn.ReLU()
        self.reg_drop1 = nn.Dropout(0.1)

        self.reg_fc2 = nn.Linear(256, 128)
        self.reg_act2 = nn.ReLU()
        self.reg_drop2 = nn.Dropout(0.1)

        self.output_layer = nn.Linear(128, num_outputs)

        # Topology classification head (MLP), fed by trunk + regression output
        if topology_classification and num_topology_classes is not None:
            top_input_dim = self.feature_dim + num_outputs
            self.top_fc1 = nn.Linear(top_input_dim, 256)
            self.top_act1 = nn.ReLU()
            self.top_drop1 = nn.Dropout(0.1)
            self.top_fc2 = nn.Linear(256, 128)
            self.top_act2 = nn.ReLU()
            self.top_drop2 = nn.Dropout(0.1)
            self.topology_head = nn.Linear(128, num_topology_classes)
        else:
            self.top_fc1 = None
            self.top_act1 = None
            self.top_drop1 = None
            self.top_fc2 = None
            self.top_act2 = None
            self.top_drop2 = None
            self.topology_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(x)
        reg = self.reg_drop1(self.reg_act1(self.reg_fc1(x)))
        reg = self.reg_drop2(self.reg_act2(self.reg_fc2(reg)))
        y_br = self.output_layer(reg)
        if self.topology_head is None:
            return y_br
        top_input = torch.cat([x, y_br], dim=1)
        top = self.top_drop1(self.top_act1(self.top_fc1(top_input)))
        top = self.top_drop2(self.top_act2(self.top_fc2(top)))
        y_top = self.topology_head(top)
        return y_br, y_top

    def __str__(self) -> str:
        architecture = getattr(self, "architecture", "parallel")
        top_in = (
            self.top_fc1.in_features
            if getattr(self, "top_fc1", None) is not None
            else "n/a"
        )
        return "\n".join(
            [
                "CNNModel architecture:",
                f"  Regression/Classification: {architecture}",
                f"  Inputs: in_channels={self.in_channels}, num_taxa={self.num_taxa}, rooted={self.tree_rooted}",
                f"  Label transform: {self.label_transform_strategy}",
                f"  Outputs: num_outputs={self.num_outputs}",
                "  Convolutional Layers:",
                "    conv1: Conv2d(in_channels={in_c}, out_channels=128, kernel_size={kernel}, stride={stride}, padding={padding})".format(
                    in_c=self.conv1.in_channels,
                    kernel=format_tuple(self.conv1.kernel_size),
                    stride=format_tuple(self.conv1.stride),
                    padding=format_tuple(self.conv1.padding),
                ),
                "      norm: BatchNorm2d(128)",
                f"      activation: {self.act1.__class__.__name__}",
                f"      pool: {self.pool1.__class__.__name__}",
                "    conv2: Conv2d(in_channels=128, out_channels=128, kernel_size={kernel}, stride={stride}, padding={padding})".format(
                    kernel=format_tuple(self.conv2.kernel_size),
                    stride=format_tuple(self.conv2.stride),
                    padding=format_tuple(self.conv2.padding),
                ),
                "      norm: BatchNorm2d(128)",
                f"      activation: {self.act2.__class__.__name__}",
                "      pool: {name}(kernel_size={kernel}, stride={stride})".format(
                    name=self.pool2.__class__.__name__,
                    kernel=format_tuple(getattr(self.pool2, "kernel_size", ())),
                    stride=format_tuple(getattr(self.pool2, "stride", ())),
                ),
                "    conv3: Conv2d(in_channels=128, out_channels=128, kernel_size={kernel}, stride={stride}, padding={padding})".format(
                    kernel=format_tuple(self.conv3.kernel_size),
                    stride=format_tuple(self.conv3.stride),
                    padding=format_tuple(self.conv3.padding),
                ),
                "      norm: BatchNorm2d(128)",
                f"      activation: {self.act3.__class__.__name__}",
                f"  Global pooling: {self.global_pool.__class__.__name__}",
                "  Regression head:",
                "    reg_fc1: Linear(in_features={in_f}, out_features={out_f})".format(
                    in_f=self.reg_fc1.in_features,
                    out_f=self.reg_fc1.out_features,
                ),
                f"      activation: {self.reg_act1.__class__.__name__}",
                f"      dropout: {self.reg_drop1.__class__.__name__}",
                "    reg_fc2: Linear(in_features={in_f}, out_features={out_f})".format(
                    in_f=self.reg_fc2.in_features,
                    out_f=self.reg_fc2.out_features,
                ),
                f"      activation: {self.reg_act2.__class__.__name__}",
                f"      dropout: {self.reg_drop2.__class__.__name__}",
                "  Output layer: Linear(in_features={in_f}, out_features={out_f})".format(
                    in_f=self.output_layer.in_features,
                    out_f=self.output_layer.out_features,
                ),
                f"  Topology classification: {self.topology_classification}",
                (
                    "  Topology head: Linear(in_features={in_f}, out_features={out_f})".format(
                        in_f=top_in,
                        out_f=self.topology_head.out_features,
                    )
                    if self.topology_head is not None
                    else "  Topology head: None"
                ),
            ]
        )


# Backwards compatibility: CNNModel points to the parallel architecture.
CNNModel = ParallelCNNModel
