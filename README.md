# Phylogenetic ML Models

This project provides two phylogenetic branch-length regressors: a CNN and a Kolmogorov-Arnold Network (KAN). Bring prepared sequence datasets (`.npy` files) to train either architecture.

## Setup

- Python 3.10+
- Create and activate a preferred virtual environment (pyenv, Conda, `venv`, etc.)
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Configuration

Training configurations share a common schema with model-specific sub-sections. Key fields:

- `model`: `in_channels`, `rooted`, `num_outputs`, `topology_classification`, `topology_weight` (used by CNN/KAN where applicable).
- `model.kan`: hidden layer widths, spline/grid parameters, regularisation toggles, and other KAN options.
- `data`, `trainer`, `label_transform`, `outputs`: dataset location, training schedule, label transforms, and output directories (`outputs.results_dir`).

Sample configurations live in `sample_config/training.yaml` and `sample_config/training.json`.

## Training

### CNN workflow

The CNN trainer expects structured NumPy records with encoded sequences (`X`) and branch-length targets (`y_br`). Populate the common `model` fields and launch training:

```bash
python -m src.cnn --config path/to/training.yaml
```

Topology classification targets are not used by the CNN.

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
