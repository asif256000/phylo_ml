# Phylogenetic ML Models (Tests)

Tests for the phylogenetic ML repository. Data simulation and XML parsing have been removed; bring your own prepared datasets. The current module under test is a convolutional neural network for branch-length prediction.

## Setup

- **Python:** 3.10 or newer.
- **Environment:** create and activate a virtual environment (Conda, `venv`, etc.).
- **Install:**

  ```bash
  pip install -r requirements.txt
  ```

## Training (CNN module)

The CNN trainer consumes a NumPy `.npy` dataset containing encoded sequences and branch-length targets. Point the `data.dataset_file` field in a training config to your prepared file, then launch training:

```bash
python -m src.cnn --config path/to/training.yaml
```

Reference configs live in `sample_config/training.yaml` and `sample_config/training.json`. Key sections:

- `data`: dataset path, batch size, worker count, and split ratios.
- `trainer`: epochs, patience, learning rate, weight decay.
- `model`: convolutional/linear layers plus optional topology classification head.
- `label_transform`: `sqrt` (default) or `log` transform for branch-length targets.
- `outputs`: directory for diagnostic plots.

## Testing

Run the remaining model-focused tests with:

```bash
pytest
```

Suites:

- `tests/test_model.py`: CNN architecture and forward-pass coverage.
- `tests/test_train.py`: data splitting, label transforms, and trainer checks.
