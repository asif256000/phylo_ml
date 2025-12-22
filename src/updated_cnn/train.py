from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

from .model import CNNModel


def _get_model_summary(model: CNNModel, num_clades: int, seq_length: int, num_channels: int, batch_size: int = 1) -> str:
    """Get model details via torchinfo if available; fallback to __str__ otherwise."""
    try:
        from torchinfo import summary
        from io import StringIO
        import sys

        # Capture torchinfo output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        summary(
            model,
            input_size=(batch_size, num_channels, num_clades, seq_length),
            col_names=("input_size", "output_size", "num_params"),
            depth=3,
            verbose=0,
        )
        
        sys.stdout = old_stdout
        summary_str = captured_output.getvalue()
        return summary_str
    except Exception as exc:
        # Fallback preserves prior behavior if torchinfo is missing or fails
        return f"(torchinfo unavailable or failed: {exc})\n{str(model)}"


def _warn_if_negative(values, context: str, source: str) -> None:
    """Emit an uppercase warning if any branch length is negative."""
    if isinstance(values, torch.Tensor):
        if torch.any(values < 0):
            min_val = float(torch.min(values).item())
            warnings.warn(
                f"{context} DETECTED NEGATIVE {source} BRANCH LENGTH: {min_val:.6f}",
                RuntimeWarning,
            )
    else:
        array = np.asarray(values)
        if np.any(array < 0):
            min_val = float(np.min(array))
            warnings.warn(
                f"{context} DETECTED NEGATIVE {source} BRANCH LENGTH: {min_val:.6f}",
                RuntimeWarning,
            )


def _get_next_results_dir(base_path: Path) -> Path:
    """Find next available results directory with intelligent suffix numbering.
    
    If base_path exists, returns base_path_1, base_path_2, etc.
    Looks for existing suffixes and increments to find the next available number.
    
    Args:
        base_path: The desired results directory path
        
    Returns:
        Path object for the next available directory (doesn't create it)
    """
    if not base_path.exists():
        return base_path
    
    # Base path exists, find highest existing suffix
    parent = base_path.parent
    base_name = base_path.name
    
    # Find all directories matching base_name or base_name_N pattern
    existing_suffixes = [0]  # 0 represents the base directory itself
    
    if parent.exists():
        for item in parent.iterdir():
            if item.is_dir():
                name = item.name
                if name == base_name:
                    existing_suffixes.append(0)
                elif name.startswith(base_name + "_"):
                    suffix_part = name[len(base_name) + 1:]
                    if suffix_part.isdigit():
                        existing_suffixes.append(int(suffix_part))
    
    # Find next available suffix
    next_suffix = max(existing_suffixes) + 1
    return parent / f"{base_name}_{next_suffix}"


class LabelTransformer:
    """Applies optional transforms to branch-length targets."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy.lower()

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        if self.strategy == "none":
            return np.asarray(values, dtype=np.float32)
        data = np.array(values, dtype=np.float32, copy=True)
        if self.strategy == "sqrt":
            if np.any(data < 0):
                raise ValueError("Square-root transform requires non-negative branch lengths")
            np.sqrt(data, out=data)
            return data
        if self.strategy == "log":
            if np.any(data <= 0):
                raise ValueError("Log transform requires strictly positive branch lengths")
            np.log(data, out=data)
            return data
        raise ValueError(f"Unsupported label transform strategy '{self.strategy}'")

    def inverse_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.strategy == "none":
            return tensor
        if self.strategy == "sqrt":
            return torch.square(tensor)
        if self.strategy == "log":
            return torch.exp(tensor)
        raise ValueError(f"Unsupported label transform strategy '{self.strategy}'")


@dataclass(frozen=True)
class TrainingConfig:
    dataset_file: Path
    batch_size: int = 32
    num_workers: int = 0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    epochs: int = 20
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    label_transform: str = "sqrt"
    device: str | None = None
    model_in_channels: int = 4
    model_rooted: bool = True
    model_num_outputs: int | None = None
    results_dir: Path | None = None
    zoomed_plots: bool = False
    individual_branch_plots: bool = False
    branch_sum_plots: bool = False


@dataclass
class TrainingResult:
    best_val_loss: float
    test_loss: float
    train_losses: list[float]
    val_losses: list[float]
    runtime_seconds: float
    metrics: dict[str, float]


class SequenceDataset(Dataset):
    """Wrap structured NumPy memmap and expose tensors."""

    def __init__(
        self,
        data: np.memmap,
        indices: np.ndarray,
        y_br: np.ndarray,
        num_clades: int,
        seq_length: int,
        num_channels: int,
    ) -> None:
        self.data = data
        self.indices = indices
        self.num_clades = num_clades
        self.seq_length = seq_length
        self.num_channels = num_channels
        if y_br.dtype != np.float32:
            y_br = y_br.astype(np.float32, copy=False)
        self.y_br = np.ascontiguousarray(y_br)

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.data[self.indices[idx]]
        encoded = record["X"]
        if encoded.ndim != 3:
            raise ValueError("Encoded features must have shape (taxa, seq_length, channels)")
        if encoded.shape != (self.num_clades, self.seq_length, self.num_channels):
            raise ValueError(
                "Encoded feature matrix shape mismatch: "
                f"expected {(self.num_clades, self.seq_length, self.num_channels)}, got {encoded.shape}"
            )
        encoded = np.transpose(encoded, (2, 0, 1))
        features = torch.from_numpy(np.ascontiguousarray(encoded, dtype=np.float32))
        y_br = torch.from_numpy(self.y_br[self.indices[idx]])
        return features, y_br


def split_indices(total_size: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio > 0.95:
        raise ValueError("train_ratio + val_ratio cannot exceed 0.95")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_size)

    train_size = max(1, int(total_size * train_ratio))
    val_min = 1 if val_ratio > 0 else 0
    val_size = max(val_min, int(total_size * val_ratio))
    test_size = total_size - train_size - val_size

    while test_size <= 0 and (train_size > 1 or val_size > val_min):
        if train_size > 1:
            train_size -= 1
        elif val_size > val_min:
            val_size -= 1
        test_size = total_size - train_size - val_size

    if train_size <= 0:
        raise ValueError("Train split is empty; increase dataset size or adjust ratios")
    if val_ratio > 0 and val_size <= 0:
        raise ValueError("Validation split is empty; adjust ratios or increase dataset size")
    if test_size <= 0:
        raise ValueError("Test split is empty; adjust ratios or increase dataset size")

    train_end = train_size
    val_end = train_end + val_size
    train_idx = permutation[:train_end]
    val_idx = permutation[train_end:val_end]
    test_idx = permutation[val_end:val_end + test_size]
    return train_idx, val_idx, test_idx


def _build_dataloaders(
    data: np.memmap,
    y_br: np.ndarray,
    num_clades: int,
    seq_length: int,
    num_channels: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_idx, val_idx, test_idx = split_indices(len(data), train_ratio, val_ratio, seed)

    train_dataset = SequenceDataset(data, train_idx, y_br, num_clades, seq_length, num_channels)
    val_dataset = SequenceDataset(data, val_idx, y_br, num_clades, seq_length, num_channels)
    test_dataset = SequenceDataset(data, test_idx, y_br, num_clades, seq_length, num_channels)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def _train_epoch(
    model: CNNModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    for features, y_br in loader:
        features = features.to(device, non_blocking=True)
        y_br = y_br.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred_br = model(features)
        loss = criterion(pred_br, y_br)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def _evaluate(
    model: CNNModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, y_br in loader:
            features = features.to(device, non_blocking=True)
            y_br = y_br.to(device, non_blocking=True)
            pred_br = model(features)
            loss = criterion(pred_br, y_br)
            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
    return running_loss / total_samples


@torch.no_grad()
def _collect_predictions(
    model: CNNModel,
    loader: DataLoader,
    device: torch.device,
    transformer: LabelTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    model.eval()
    for features, y_br in loader:
        features = features.to(device, non_blocking=True)
        y_br = y_br.to(device, non_blocking=True)
        pred_br = model(features)
        preds.append(transformer.inverse_tensor(pred_br).cpu().numpy())
        trues.append(transformer.inverse_tensor(y_br).cpu().numpy())
    if not preds:
        return np.array([]), np.array([])
    
    preds_arr = np.vstack(preds)
    trues_arr = np.vstack(trues)
    
    # Check for negative predictions after inverse transform
    _warn_if_negative(preds_arr, "POST-TRANSFORM:", "PREDICTED")
    
    return preds_arr, trues_arr


def _plot_branch_pair(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    output_path: Path,
    title: str,
    bins: int = 60,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    combined = np.concatenate([true_vals, pred_vals])
    lower = float(np.percentile(combined, 0.5))
    upper = float(np.percentile(combined, 99.5))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        if lower == upper:
            upper = lower + 1.0

    ax.set_facecolor("white")
    base_cmap = plt.cm.get_cmap("magma", 256)
    cmap_colors = base_cmap(np.linspace(0, 1, 256))
    cmap_colors[0, :3] = 1.0
    density_cmap = colors.ListedColormap(cmap_colors)
    hist = ax.hist2d(
        true_vals,
        pred_vals,
        bins=bins,
        range=[[lower, upper], [lower, upper]],
        cmap=density_cmap,
        vmin=0.5,
    )
    fig.colorbar(hist[3], ax=ax, label="Count")
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("True branch length")
    ax.set_ylabel("Predicted branch length")
    ax.set_title(title)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_branch_pair_high_def(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    output_path: Path,
    title: str,
    fraction: float = 0.25,
) -> None:
    combined = np.concatenate([true_vals, pred_vals])
    finite_mask = np.isfinite(combined)
    if not np.any(finite_mask):
        return
    finite_vals = combined[finite_mask]
    min_val = float(np.min(finite_vals))
    max_val = float(np.max(finite_vals))
    span = max_val - min_val
    if span <= 0 or not np.isfinite(span):
        return

    cutoff = min_val + fraction * span
    branch_mask = (
        np.isfinite(true_vals)
        & np.isfinite(pred_vals)
        & (true_vals <= cutoff)
        & (pred_vals <= cutoff)
    )
    if not np.any(branch_mask):
        return

    _plot_branch_pair(true_vals[branch_mask], pred_vals[branch_mask], output_path, title, bins=150)


def _summarize_branch_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(true_vals, pred_vals))
    mse = float(mean_squared_error(true_vals, pred_vals))
    rmse = float(root_mean_squared_error(true_vals, pred_vals))
    r2 = float(r2_score(true_vals, pred_vals))
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def _save_predictions_and_metrics(
    results_dir: Path,
    preds: np.ndarray,
    trues: np.ndarray,
    train_losses: list[float],
    val_losses: list[float],
    test_loss: float,
    branch_metrics: list[dict[str, float]] | None = None,
    model_arch: str | None = None,
    training_status: str | None = None,
    sum_metrics: dict[str, float] | None = None,
    overall_metrics: dict[str, float] | None = None,
) -> None:
    """Save predictions and metrics to text files in results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions vs actual values
    predictions_file = results_dir / "predictions.txt"
    with open(predictions_file, "w") as f:
        f.write("Sample,Branch,Actual,Predicted\n")
        num_samples = preds.shape[0]
        num_branches = preds.shape[1] if preds.ndim > 1 else 1
        
        if preds.ndim == 1:
            for sample_idx in range(num_samples):
                f.write(f"{sample_idx},0,{trues[sample_idx]:.6f},{preds[sample_idx]:.6f}\n")
        else:
            for sample_idx in range(num_samples):
                for branch_idx in range(num_branches):
                    f.write(
                        f"{sample_idx},{branch_idx},"
                        f"{trues[sample_idx, branch_idx]:.6f},"
                        f"{preds[sample_idx, branch_idx]:.6f}\n"
                    )
    
    # Save metrics
    metrics_file = results_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        # Model architecture at the top
        if model_arch:
            f.write("=== Model Architecture ===\n")
            f.write(model_arch.rstrip("\n") + "\n\n")

        if branch_metrics:
            f.write("\n=== Per-Branch Metrics ===\n")
            for idx, bm in enumerate(branch_metrics, start=1):
                f.write(
                    "Branch b{idx} | MAE: {mae:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | R2: {r2:.6f}\n".format(
                        idx=idx,
                        mae=bm.get("mae", float("nan")),
                        mse=bm.get("mse", float("nan")),
                        rmse=bm.get("rmse", float("nan")),
                        r2=bm.get("r2", float("nan")),
                    )
                )
        if overall_metrics:
            f.write("\n=== Overall Metrics ===\n")
            f.write(
                "MAE: {mae:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | R2: {r2:.6f}\n".format(
                    mae=overall_metrics.get("mae", float("nan")),
                    mse=overall_metrics.get("mse", float("nan")),
                    rmse=overall_metrics.get("rmse", float("nan")),
                    r2=overall_metrics.get("r2", float("nan")),
                )
            )
        if sum_metrics:
            f.write("\n=== Total Branch Metrics ===\n")
            f.write(
                "MAE: {mae:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | R2: {r2:.6f}\n".format(
                    mae=sum_metrics.get("mae", float("nan")),
                    mse=sum_metrics.get("mse", float("nan")),
                    rmse=sum_metrics.get("rmse", float("nan")),
                    r2=sum_metrics.get("r2", float("nan")),
                )
            )
        # Training summary before losses
        f.write("\n=== Training Summary ===\n")
        if training_status:
            f.write(training_status + "\n")
        else:
            f.write("Training status not available.\n")

        f.write("\n=== Loss History ===\n")
        f.write(f"Best Val Loss: {min(val_losses) if val_losses else 'N/A'}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Val Loss: {val_losses[-1]:.6f}\n")


def _plot_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    if not train_losses and not val_losses:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, max(len(train_losses), len(val_losses)) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    if train_losses:
        ax.plot(epochs[: len(train_losses)], train_losses, label="Train Loss", color="tab:blue")
    if val_losses:
        ax.plot(epochs[: len(val_losses)], val_losses, label="Val Loss", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (transformed space)")
    ax.set_title("Training vs Validation Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


class Trainer:
    """Minimal trainer for regression-only CNN."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        device_str = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.transformer = LabelTransformer(config.label_transform)

    def run(self) -> TrainingResult:
        start = time.time()
        cfg = self.config

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        dataset = np.load(cfg.dataset_file, mmap_mode="r")
        expected_fields = {"X", "y_br"}
        if dataset.dtype.names is None or expected_fields - set(dataset.dtype.names):
            raise ValueError("Dataset must contain structured fields 'X' and 'y_br'")

        example = dataset[0]
        if example["X"].ndim != 3:
            raise ValueError("Encoded feature matrix must have shape (taxa, seq_length, channels)")
        num_clades, seq_length, num_channels = example["X"].shape

        # Load raw targets and normalize shape before any transformation
        y_br_raw = dataset["y_br"]
        if y_br_raw.ndim == 1:
            y_br_raw = y_br_raw[:, None]
        y_br_raw = y_br_raw.astype(np.float32, copy=False)

        target_width = int(y_br_raw.shape[1])
        configured_outputs = cfg.model_num_outputs
        if configured_outputs is not None and int(configured_outputs) != target_width:
            raise ValueError(
                "Configured 'model.num_outputs' does not match dataset target width: "
                f"{configured_outputs} != {target_width}"
            )
        effective_outputs = configured_outputs or target_width

        # Apply label transform after shape checks
        y_br = self.transformer.transform_numpy(y_br_raw)

        model = CNNModel(
            num_taxa=num_clades,
            num_outputs=effective_outputs,
            in_channels=cfg.model_in_channels or num_channels,
            label_transform=cfg.label_transform,
            tree_rooted=cfg.model_rooted,
        ).to(self.device)

        model_summary = _get_model_summary(
            model,
            num_clades=num_clades,
            seq_length=seq_length,
            num_channels=num_channels,
            batch_size=min(cfg.batch_size, 4),
        )
        print("\n" + "=" * 60)
        print(model_summary)
        print("=" * 60 + "\n")

        train_loader, val_loader, test_loader = _build_dataloaders(
            data=dataset,
            y_br=y_br,
            num_clades=num_clades,
            seq_length=seq_length,
            num_channels=num_channels,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
        )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        train_losses: list[float] = []
        val_losses: list[float] = []
        epochs_without_improvement = 0
        last_epoch = 0
        last_train_loss = 0.0
        last_val_loss = 0.0

        early_stopped = False
        for epoch in range(1, cfg.epochs + 1):
            train_loss = _train_epoch(model, train_loader, criterion, optimizer, self.device)
            val_loss = _evaluate(model, val_loader, criterion, self.device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            last_epoch = epoch
            last_train_loss = train_loss
            last_val_loss = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epoch % 5 == 0:
                print(f"Epoch {epoch:02d}/{cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if epochs_without_improvement >= cfg.patience:
                print(
                    "Early stopping triggered after "
                    f"{epoch} epochs (no val improvement for {cfg.patience} epochs)."
                )
                early_stopped = True
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss = _evaluate(model, test_loader, criterion, self.device)
        runtime = time.time() - start

        print("\n" + "=" * 60)
        print(f"Final Epoch {last_epoch} | Train Loss: {last_train_loss:.6f} | Test Loss: {test_loss:.6f} | Best Val Loss: {last_val_loss:.6f}")
        print("=" * 60 + "\n")

        base_results_dir = (cfg.results_dir or Path("results")).expanduser().resolve()
        results_dir = _get_next_results_dir(base_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        _plot_loss_curve(train_losses, val_losses, plots_dir / "loss_curve.png")

        preds, trues = _collect_predictions(model, test_loader, self.device, self.transformer)
        branch_metrics_list: list[dict[str, float]] = []
        sum_metrics: dict[str, float] | None = None
        overall_metrics: dict[str, float] | None = None
        if preds.size > 0 and trues.size > 0:
            num_branches = preds.shape[1]
            for idx in range(num_branches):
                pred_branch = preds[:, idx]
                true_branch = trues[:, idx]
                if cfg.individual_branch_plots:
                    _plot_branch_pair(
                        true_branch,
                        pred_branch,
                        plots_dir / f"branch_b{idx + 1}_scatter.png",
                        f"Branch b{idx + 1}: true vs predicted",
                    )
                    if cfg.zoomed_plots:
                        _plot_branch_pair_high_def(
                            true_branch,
                            pred_branch,
                            plots_dir / f"branch_b{idx + 1}_scatter_hd.png",
                            f"Branch b{idx + 1} HD: true vs predicted",
                        )

                branch_metrics = _summarize_branch_metrics(true_branch, pred_branch)
                branch_metrics_list.append(branch_metrics)
                print(
                    "Branch b{idx} Metrics | MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}".format(
                        idx=idx + 1,
                        mae=branch_metrics["mae"],
                        mse=branch_metrics["mse"],
                        rmse=branch_metrics["rmse"],
                        r2=branch_metrics["r2"],
                    )
                )

            # Compute total/sum metrics regardless of plotting preference
            true_sum = np.sum(trues, axis=1)
            pred_sum = np.sum(preds, axis=1)
            sum_metrics = _summarize_branch_metrics(true_sum, pred_sum)
            print(
                "Total Branch Metrics | MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}".format(
                    mae=sum_metrics["mae"],
                    mse=sum_metrics["mse"],
                    rmse=sum_metrics["rmse"],
                    r2=sum_metrics["r2"],
                )
            )
            if cfg.branch_sum_plots:
                _plot_branch_pair(
                    true_sum,
                    pred_sum,
                    plots_dir / "branch_sum_scatter.png",
                    "Branch sum: true vs predicted",
                )

            # Overall metrics across all branches (flattened):
            # We treat each branch-length value independently by flattening
            # the (num_samples, num_branches) arrays into 1D vectors.
            # This yields a branch-wise accuracy view across the dataset.
            flat_true = trues.reshape(-1)
            flat_pred = preds.reshape(-1)
            overall_metrics = _summarize_branch_metrics(flat_true, flat_pred)
            print(
                "Overall Branch Metrics | MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}".format(
                    mae=overall_metrics["mae"],
                    mse=overall_metrics["mse"],
                    rmse=overall_metrics["rmse"],
                    r2=overall_metrics["r2"],
                )
            )

        print(f"Results saved to: {results_dir }")
        print(f"Total runtime: {runtime:.2f} seconds")

        # Compose training status string
        if early_stopped:
            training_status = (
                f"Early stopping triggered after {last_epoch} epochs (no val improvement for {cfg.patience} epochs)"
            )
        else:
            training_status = f"Trained for full {cfg.epochs} epochs"

        # Save predictions and metrics to files (no overall metrics)
        _save_predictions_and_metrics(
            results_dir ,
            preds,
            trues,
            train_losses,
            val_losses,
            test_loss,
            branch_metrics=branch_metrics_list if branch_metrics_list else None,
            model_arch=model_summary,
            training_status=training_status,
            sum_metrics=sum_metrics,
            overall_metrics=overall_metrics,
        )

        return TrainingResult(
            best_val_loss=best_val_loss,
            test_loss=test_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            runtime_seconds=runtime,
            metrics={},
        )


def run_training(config: TrainingConfig) -> TrainingResult:
    trainer = Trainer(config)
    return trainer.run()


def build_config_from_mapping(payload: dict[str, Any]) -> TrainingConfig:
    # Accept a full training config payload similar to the original YAML structure.
    data_section = payload.get("data", payload)
    trainer_section = payload.get("trainer", payload)
    model_section = payload.get("model", {})
    outputs_section = payload.get("outputs", {})

    dataset_value = data_section.get("dataset_file") or payload.get("dataset_file")
    if dataset_value is None:
        raise ValueError("'data.dataset_file' must be provided")
    dataset_path = Path(str(dataset_value)).expanduser().resolve()

    label_cfg = payload.get("label_transform", {})
    label_strategy = label_cfg.get("strategy") if isinstance(label_cfg, dict) else label_cfg or "sqrt"

    return TrainingConfig(
        dataset_file=dataset_path,
        batch_size=int(data_section.get("batch_size", 32)),
        num_workers=int(data_section.get("num_workers", 0)),
        train_ratio=float(data_section.get("train_ratio", 0.7)),
        val_ratio=float(data_section.get("val_ratio", 0.15)),
        seed=int(data_section.get("seed", payload.get("seed", 42))),
        epochs=int(trainer_section.get("epochs", 20)),
        patience=int(trainer_section.get("patience", 20)),
        learning_rate=float(trainer_section.get("learning_rate", 1e-3)),
        weight_decay=float(trainer_section.get("weight_decay", 0.0)),
        label_transform=str(label_strategy),
        device=trainer_section.get("device") or payload.get("device"),
        model_in_channels=int(model_section.get("in_channels", 4)),
        model_rooted=bool(model_section.get("rooted", True)),
        model_num_outputs=(
            int(model_section["num_outputs"])
            if model_section.get("num_outputs") is not None
            else None
        ),
        results_dir=Path(str(outputs_section.get("results_dir", outputs_section.get("results_dir ", "branch_plots")))).expanduser().resolve(),
        zoomed_plots=bool(outputs_section.get("zoomed_plots", False)),
        individual_branch_plots=bool(outputs_section.get("individual_branch_plots", False)),
        branch_sum_plots=bool(outputs_section.get("branch_sum_plots", False)),
    )
