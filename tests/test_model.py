from pathlib import Path

import torch
import torch.nn as nn

from src.cnn.model import CNNModel
from src.configuration.training import TrainingConfig


def test_model_forward_pass_output_shape():
    batch_size = 4
    num_taxa = 2
    seq_length = 1000
    num_outputs = 3

    dummy_input = torch.randn(batch_size, 4, num_taxa, seq_length)
    training_config = TrainingConfig.from_mapping(
        {"data": {"dataset_file": "dummy.npy"}},
        base_path=Path.cwd(),
    )
    model = CNNModel.from_config(
        training_config.model,
        num_taxa=num_taxa,
        num_outputs=num_outputs,
        label_transform=training_config.label_transform.strategy,
    )

    output, _ = model(dummy_input)

    assert output.shape == (batch_size, num_outputs)


def test_model_layer_configuration():
    num_taxa = 2
    seq_length = 1000
    training_config = TrainingConfig.from_mapping(
        {"data": {"dataset_file": "dummy.npy"}},
        base_path=Path.cwd(),
    )
    model = CNNModel.from_config(
        training_config.model,
        num_taxa=num_taxa,
        num_outputs=3,
        label_transform=training_config.label_transform.strategy,
    )

    first_block = model.conv_layers[0]
    conv1 = first_block["conv"]
    assert conv1.kernel_size == (2, 1)
    assert conv1.stride == (1, 1)
    assert conv1.padding == (0, 0)
    assert isinstance(first_block["pool"], nn.Identity)

    second_block = model.conv_layers[1]
    conv2 = second_block["conv"]
    assert conv2.kernel_size == (1, 3)
    assert conv2.stride == (1, 1)
    assert conv2.padding == (0, 1)

    # Forward a single example to ensure pooling and fully connected layers are wired correctly
    dummy_input = torch.randn(1, 4, num_taxa, seq_length)
    output, _ = model(dummy_input)
    assert output.shape == (1, 3)


def test_model_string_representation_includes_layers():
    num_taxa = 2
    training_config = TrainingConfig.from_mapping(
        {"data": {"dataset_file": "dummy.npy"}},
        base_path=Path.cwd(),
    )
    model = CNNModel.from_config(
        training_config.model,
        num_taxa=num_taxa,
        num_outputs=3,
        label_transform=training_config.label_transform.strategy,
    )

    description = str(model)

    assert "CNNModel architecture:" in description
    assert "Label transform:" in description
    assert "Conv2d" in description
    assert "Linear" in description
    assert "Global pooling" in description
