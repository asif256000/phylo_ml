# Configuration Reference

This document describes all configuration fields used by the CNN and KAN modules. Configuration files are YAML/JSON mappings with the sections below.

## Top-level

- `seed` (int, optional, default: `42`)
  - Global RNG seed for NumPy/Torch.
- `label_transform` (string or mapping, optional, default: `sqrt`)
  - If a string: one of `sqrt`, `log`.
  - If a mapping: `label_transform.strategy` with the same allowed values.
  - Mapping form is supported to allow future extensions (e.g., per-transform settings) without changing the top-level schema. Example:
    - YAML:
      - `label_transform: sqrt`
      - `label_transform: { strategy: sqrt }`

## `data`

- `data.dataset_file` (string, required)
  - Path to `.npy` dataset file (structured array). This is the only required field in `data`.
- `data.batch_size` (int, optional, default: `32`)
  - Mini-batch size used by the DataLoader. Must be positive.
- `data.num_workers` (int, optional, default: `0`)
  - Number of DataLoader subprocesses used for data loading.
  - `0` means **load data in the main process** (simpler, fewer multiprocessing issues; slower on large datasets).
  - `>0` uses worker processes (faster input pipeline but higher memory usage).
- `data.train_ratio` (float, optional, default: `0.70`)
  - Fraction of samples used for training. Must be in `(0, 1)`.
- `data.val_ratio` (float, optional, default: `0.15`)
  - Fraction of samples used for validation. Must be in `[0, 1)` and `train_ratio + val_ratio < 1`.
- `data.seed` (int, optional)
  - Overrides `seed` for dataset split generation so you can keep global seeding stable but vary splits.

## `trainer`

- `trainer.epochs` (int, optional, default: `20`)
  - Maximum training epochs. Must be positive.
- `trainer.patience` (int, optional, default: `20`)
  - Early stopping patience (number of epochs without validation improvement). Must be positive.
- `trainer.learning_rate` (float, optional, default: `1e-3`)
  - Initial learning rate for Adam. Must be positive.
- `trainer.weight_decay` (float, optional, default: `0.0`)
  - L2 weight decay for Adam. Must be >= 0.

## `model` (shared)

- `model.in_channels` (int, optional, default: `4`)
  - Number of input channels per taxon (must match the last dimension of `X`).
- `model.num_outputs` (int, optional)
  - Number of branch-length outputs. If set, must match the `y_br` width in the dataset.
- `model.num_taxa` (int, optional)
  - Optional override for taxa count (usually inferred from `X`).
- `model.rooted` (bool, optional, default: `true`)
  - Whether the tree is rooted. Used in model metadata and output summaries.
- `model.topology_classification` (bool, optional, default: `false`)
  - If `true`, CNN adds a topology classification head and expects `y_top` in the dataset.
- `model.topology_weight` (float, optional, default: `1.0`)
  - Multiplier for topology classification loss when enabled.

### CNN-only behavior

- If `model.topology_classification: true`, the CNN outputs:
  - Regression head: `y_br` predictions (size = `model.num_outputs`).
  - Classification head: logits for topology classes (size inferred from `y_top`).
- **Important:** Enabling topology classification does **not** change `model.num_outputs`.

### KAN-only behavior (`model.kan` required)

`model.kan` is required for KAN training and ignored by CNN.

- `model.kan.hidden_layers` (list[int], required)
  - Sizes of hidden layers. Must be non-empty and positive.
- `model.kan.grid` (int, optional, default: `5`)
  - Must be positive.
- `model.kan.spline_order` (int, optional, default: `3`)
  - Must be positive.
- `model.kan.mult_arity` (int, optional, default: `2`)
  - Must be positive.
- `model.kan.noise_scale` (float, optional, default: `0.3`)
  - Must be >= 0.
- `model.kan.base_function` (string, optional, default: `silu`)
- `model.kan.symbolic_enabled` (bool, optional, default: `true`)
- `model.kan.affine_trainable` (bool, optional, default: `false`)
- `model.kan.grid_eps` (float, optional, default: `0.02`)
  - Must be in `[0, 1]`.
- `model.kan.grid_range` (list[float, float], optional, default: `[-1.0, 1.0]`)
  - Must be two values with lower < upper.
- `model.kan.sp_trainable` (bool, optional, default: `true`)
- `model.kan.sb_trainable` (bool, optional, default: `true`)
- `model.kan.sparse_init` (bool, optional, default: `false`)
- `model.kan.auto_save` (bool, optional, default: `false`)

## `outputs`

- `outputs.results_dir` (string, optional)
  - Directory for metrics/predictions/plots.
  - If omitted, a folder named `latest_results` is created at the workspace root. If it already exists, it is suffixed (`latest_results_1`, `latest_results_2`, ...).
- `outputs.zoomed_plots` (bool, optional, default: `false`)
  - Whether to produce high-definition zoomed scatter plots for each branch.
- `outputs.individual_branch_plots` (bool, optional, default: `false`)
  - Whether to produce per-branch prediction vs. truth scatter plots.
- `outputs.branch_sum_plots` (bool, optional, default: `false`)
  - Whether to produce scatter plots of summed branch lengths.

## Dataset fields

### CNN (regression only)
- Required fields: `X`, `y_br`.

### CNN (with topology classification)
- Required fields: `X`, `y_br`, `y_top`.
- `y_top` is expected as one-hot or class-probability vectors per sample.

### KAN
- Required fields: `X`, `y_br`, `branch_mask`, `tree_index`.

## Output artifacts

- `br_predictions.csv`
  - Regression rows: `Sample,Branch,Actual,Predicted`.
- `top_predictions.csv` (only when topology classification is enabled)
  - Classification rows: `Sample,TrueClass,PredClass`.
- `metrics.txt`
  - Regression metrics: per-branch MAE/MSE/RMSE/R2, total-branch metrics, and overall metrics.
  - Topology metrics (if enabled): Accuracy, Macro F1, classification report, and confusion matrix.
