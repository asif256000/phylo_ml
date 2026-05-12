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

Set `model.topology_classification: true` to enable the optional topology classification head. This does **not** change `model.num_outputs`; branch-length regression still uses `y_br`, while topology classification uses `y_top`.

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

## Data sanity checks

Use the scripts under `scripts/` to validate dataset labels and run a topology overfit check.

- Label verification (one-hot validation + class balance):

  ```bash
  python scripts/verify.py
  ```

- Overfit check for topology classification (balanced subset):

  ```bash
  python scripts/overfit_check.py --topology-only --full-batch --learning-rate 1e-2 --epochs 300 --disable-dropout
  ```

- Linear probe sanity test (separability):

  ```bash
  python scripts/overfit_check.py --topology-only --linear-probe --full-batch --learning-rate 1e-2 --epochs 300
  ```

### Output artifacts

 `br_predictions.csv`:
  - Regression rows: `Sample,Branch,Actual,Predicted`.
 `top_predictions.csv` (only when enabled):
  - Topology rows: `Sample,TrueClass,PredClass`.
 `metrics.txt`:
  - Regression metrics: per-branch MAE/MSE/RMSE/R2, total-branch metrics, and overall metrics.
  - Topology metrics (only when enabled): Accuracy, Macro F1, full classification report, and confusion matrix.
