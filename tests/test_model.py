import torch
import torch.nn as nn

from src.cnn.model import CNNModel


def test_model_forward_pass_output_shape():
    batch_size = 4
    num_taxa = 2
    seq_length = 1000
    num_outputs = 3
    in_channels = 4

    dummy_input = torch.randn(batch_size, in_channels, num_taxa, seq_length)
    model = CNNModel(
        num_taxa=num_taxa,
        num_outputs=num_outputs,
        in_channels=in_channels,
        label_transform="sqrt",
        tree_rooted=True,
    )

    output = model(dummy_input)

    assert output.shape == (batch_size, num_outputs)


def test_model_layer_configuration():
    num_taxa = 2
    seq_length = 1000
    model = CNNModel(
        num_taxa=num_taxa,
        num_outputs=3,
        in_channels=4,
        label_transform="sqrt",
        tree_rooted=True,
    )

    conv1 = model.conv1
    assert conv1.kernel_size == (num_taxa, 1)
    assert conv1.stride == (num_taxa, 1)
    assert conv1.padding == (0, 0)
    assert isinstance(model.pool1, nn.Identity)

    conv2 = model.conv2
    assert conv2.kernel_size == (1, 3)
    assert conv2.stride == (1, 1)
    assert conv2.padding == (0, 0)
    assert isinstance(model.pool2, nn.AvgPool2d)

    assert model.reg_fc1.in_features == 64
    assert model.reg_fc2.out_features == 128

    # Forward a single example to ensure pooling and fully connected layers are wired correctly
    dummy_input = torch.randn(1, 4, num_taxa, seq_length)
    output = model(dummy_input)
    assert output.shape == (1, 3)


def test_model_string_representation_includes_layers():
    num_taxa = 2
    model = CNNModel(
        num_taxa=num_taxa,
        num_outputs=3,
        in_channels=4,
        label_transform="sqrt",
        tree_rooted=True,
    )

    description = str(model)

    assert "CNNModel architecture:" in description
    assert "Label transform:" in description
    assert "Conv2d" in description
    assert "Linear" in description
    assert "Global pooling" in description


def test_model_with_topology_head_returns_tuple():
    num_taxa = 4
    num_outputs = 10
    num_topology_classes = 5
    model = CNNModel(
        num_taxa=num_taxa,
        num_outputs=num_outputs,
        in_channels=4,
        label_transform="sqrt",
        tree_rooted=True,
        topology_classification=True,
        num_topology_classes=num_topology_classes,
    )

    dummy_input = torch.randn(2, 4, num_taxa, 1000)
    output = model(dummy_input)
    assert isinstance(output, tuple)
    y_br, y_top = output
    assert y_br.shape == (2, num_outputs)
    assert y_top.shape == (2, num_topology_classes)
