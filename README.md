# Phylogenetic ML Models

This project provides two phylogenetic branch-length regressors: a configurable convolutional neural network (CNN) and a Kolmogorov-Arnold Network (KAN). Bring prepared sequence datasets (`.npy` files) to train either architecture.

## Setup

- Python 3.10+
- Create and activate a preferred virtual environment (pyenv, Conda, `venv`, etc.)
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Configuration

Training configurations share a common schema with model-specific sub-sections. Key fields:

- Common model keys: `in_channels`, `rooted`, `num_outputs`, `topology_classification`, `topology_weight`.
- `model.cnn`: convolution blocks, linear layers, global pooling, and related CNN hyperparameters.
- `model.kan`: hidden layer widths, spline/grid parameters, regularisation toggles, and other KAN options.
- `data`, `trainer`, `label_transform`, `outputs`: dataset location, training schedule, label transforms, and artifact directories.

Sample configurations live in `sample_config/training.yaml` and `sample_config/training.json`.

## Training

### CNN workflow

The CNN trainer expects structured NumPy records with encoded sequences (`X`), branch-length targets (`y_br`), optional topology labels (`y_top`), and masks. Populate the `model.cnn` block and launch training:

```bash
python -m src.cnn --config path/to/training.yaml
```

If `model.topology_classification` is enabled, the trainer uses the additional topology targets for multi-task learning.

### KAN workflow

The KAN trainer flattens the same dataset format and optimises a Kolmogorov-Arnold Network via `pykan`. Populate the `model.kan` block and run:

```bash
python -m src.kan --config path/to/training.yaml
```

KAN training is focused on branch-length regression; topology classification must remain disabled.

## Testing

Execute the automated checks with:

```bash
pytest
```

Test suites cover CNN architecture behaviours (`tests/test_model.py`) and data splitting plus trainer invariants (`tests/test_train.py`).
