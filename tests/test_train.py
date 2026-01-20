import numpy as np
import pytest
import torch

from src.configuration.training import TrainingConfig
from src.updated_cnn.train import LabelTransformer, Trainer, split_indices


def test_split_indices_small_dataset_has_all_splits():
    train_idx, val_idx, test_idx = split_indices(total_size=5, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert len(train_idx) == 3
    assert len(val_idx) == 1
    assert len(test_idx) == 1


def test_split_indices_no_validation_ratio():
    train_idx, val_idx, test_idx = split_indices(total_size=3, train_ratio=0.7, val_ratio=0.0, seed=0)
    assert len(train_idx) == 2
    assert len(val_idx) == 0
    assert len(test_idx) == 1


def test_split_indices_raises_when_dataset_too_small():
    with pytest.raises(ValueError):
        split_indices(total_size=2, train_ratio=0.7, val_ratio=0.2, seed=0)


def test_training_config_uses_absolute_paths_as_is(tmp_path):
    project_root = tmp_path
    config_dir = project_root / "config"
    config_dir.mkdir()
    data_dir = project_root / "npy_data"
    data_dir.mkdir()
    (data_dir / "sample.npy").touch()

    payload = {
        "seed": 7,
        "data": {"dataset_file": str((data_dir / "sample.npy").resolve())},
        "trainer": {},
        "model": {},
        "outputs": {"results_dir": str((project_root / "branch_plots/run").resolve())},
    }

    config = TrainingConfig.from_mapping(payload, base_path=config_dir)

    expected_dataset = (data_dir / "sample.npy").resolve()
    expected_plots = (project_root / "branch_plots/run").resolve()

    assert config.data.dataset_file == expected_dataset
    assert config.outputs.results_dir == expected_plots


def test_label_transform_roundtrip_sqrt():
    transformer = LabelTransformer("sqrt")
    values = np.array([[0.1, 0.5, 1.0], [2.0, 3.5, 4.0]], dtype=np.float32)
    transformed = transformer.transform_numpy(values)
    recovered = transformer.inverse_tensor(torch.from_numpy(transformed)).numpy()
    assert np.allclose(recovered, values, atol=1e-6)


def test_label_transform_roundtrip_log():
    transformer = LabelTransformer("log")
    values = np.array([[0.1, 0.5, 1.0], [2.0, 3.5, 4.0]], dtype=np.float32)
    transformed = transformer.transform_numpy(values)
    recovered = transformer.inverse_tensor(torch.from_numpy(transformed)).numpy()
    assert np.allclose(recovered, values, atol=1e-6)


def test_trainer_raises_when_num_outputs_mismatch(tmp_path):
    dataset_path = tmp_path / "dataset.npy"
    dtype = [
        ("X", np.float32, (2, 6, 4)),
        ("y_br", np.float32, (3,)),
    ]
    data = np.zeros(2, dtype=dtype)
    data["y_br"] = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32)
    data["X"] = 0.25
    np.save(dataset_path, data)

    payload = {
        "seed": 1,
        "label_transform": {"strategy": "sqrt"},
        "data": {
            "dataset_file": str(dataset_path.resolve()),
            "batch_size": 1,
            "num_workers": 0,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
        },
        "trainer": {"epochs": 1, "patience": 1, "learning_rate": 0.001, "weight_decay": 0.0},
        "model": {"num_outputs": 4},
        "outputs": {"branch_plot_dir": str((tmp_path / "plots").resolve())},
    }

    config = TrainingConfig.from_mapping(payload, base_path=tmp_path)
    trainer = Trainer(config)

    with pytest.raises(ValueError):
        trainer.run()
