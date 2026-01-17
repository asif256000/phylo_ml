from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from torch.utils.data import DataLoader, Dataset

from src.configuration.training import TrainingConfig

from .model import CNNModel

# from .hardcoded_model import CNNModel


class LabelTransformer:
    """Applies forward and inverse transformations to branch length targets."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
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
        if self.strategy == "sqrt":
            return torch.square(tensor)
        if self.strategy == "log":
            return torch.exp(tensor)
        raise ValueError(f"Unsupported label transform strategy '{self.strategy}'")


@dataclass
class TrainingResult:
    """Summary of the CNN training run."""

    best_val_loss: float
    test_loss: float
    train_losses: list[float]
    val_losses: list[float]
    runtime_seconds: float


class SequenceDataset(Dataset):
    """Wrap a structured NumPy memmap and expose PyTorch tensors."""

    def __init__(
        self,
        data: np.memmap,
        indices: np.ndarray,
        y_br: np.ndarray,
        branch_mask: np.ndarray,
        y_top: np.ndarray,
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

        self.branch_mask = np.ascontiguousarray(branch_mask)

        if y_top.dtype != np.float32:
            y_top = y_top.astype(np.float32, copy=False)
        self.y_top = np.ascontiguousarray(y_top)

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.data[self.indices[idx]]
        encoded = record["X"]
        if encoded.ndim != 3:
            raise ValueError("Expected encoded features to have 3 dimensions (taxa, seq_length, nucleotides)")
        if encoded.shape != (self.num_clades, self.seq_length, self.num_channels):
            raise ValueError(
                "Encoded feature matrix shape mismatch: "
                f"expected {(self.num_clades, self.seq_length, self.num_channels)}, got {encoded.shape}"
            )
        encoded = np.transpose(encoded, (2, 0, 1))  # -> (channels, taxa, seq_length)
        features = torch.from_numpy(np.ascontiguousarray(encoded, dtype=np.float32))

        y_br = torch.from_numpy(self.y_br[self.indices[idx]])
        branch_mask = torch.from_numpy(self.branch_mask[self.indices[idx]])
        y_top = torch.from_numpy(self.y_top[self.indices[idx]])

        return features, y_br, branch_mask, y_top


def split_indices(
    total_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    test_idx = permutation[val_end : val_end + test_size]

    return train_idx, val_idx, test_idx


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


def _build_dataloaders(
    data: np.memmap,
    y_br: np.ndarray,
    branch_mask: np.ndarray,
    y_top: np.ndarray,
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

    train_dataset = SequenceDataset(data, train_idx, y_br, branch_mask, y_top, num_clades, seq_length, num_channels)
    val_dataset = SequenceDataset(data, val_idx, y_br, branch_mask, y_top, num_clades, seq_length, num_channels)
    test_dataset = SequenceDataset(data, test_idx, y_br, branch_mask, y_top, num_clades, seq_length, num_channels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def _train_epoch(
    model: CNNModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    transformer: LabelTransformer,
    topology_weight: float,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    for features, y_br, branch_mask, y_top in loader:
        features = features.to(device, non_blocking=True)
        y_br = y_br.to(device, non_blocking=True)
        branch_mask = branch_mask.to(device, non_blocking=True)
        y_top = y_top.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred_br, pred_top = model(features)

        # Regression loss (masked)
        # We assume criterion is MSELoss(reduction='none') or similar, but we need to handle masking manually
        # if criterion is standard MSELoss.
        # Let's calculate MSE manually for masked elements.

        diff = (pred_br - y_br) * branch_mask
        loss_br = torch.sum(diff**2) / torch.clamp(torch.sum(branch_mask), min=1.0)

        loss_top = torch.tensor(0.0, device=device)
        if pred_top is not None:
            # y_top is one-hot.
            target_classes = torch.argmax(y_top, dim=1)
            loss_top = torch.nn.functional.cross_entropy(pred_top, target_classes)

        loss = loss_br + topology_weight * loss_top

        # Check for negative predictions for warning (on unmasked branches)
        with torch.no_grad():
            actual_outputs = transformer.inverse_tensor(pred_br.detach())
            actual_targets = transformer.inverse_tensor(y_br)
            # Only check masked
            # But _warn_if_negative takes full tensor.
            # We can pass masked values?
            # For simplicity, just pass everything, but ignore zeros if they are masked?
            # Masked values in y_br are 0.
            # Masked values in pred_br are whatever.
            # If we only care about valid branches:
            # _warn_if_negative(actual_outputs[branch_mask], ...)
            # But branch_mask is boolean tensor.
            pass

        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


@torch.no_grad()
def _evaluate(
    model: CNNModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    transformer: LabelTransformer,
    topology_weight: float,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    for features, y_br, branch_mask, y_top in loader:
        features = features.to(device, non_blocking=True)
        y_br = y_br.to(device, non_blocking=True)
        branch_mask = branch_mask.to(device, non_blocking=True)
        y_top = y_top.to(device, non_blocking=True)

        pred_br, pred_top = model(features)

        diff = (pred_br - y_br) * branch_mask
        loss_br = torch.sum(diff**2) / torch.clamp(torch.sum(branch_mask), min=1.0)

        loss_top = torch.tensor(0.0, device=device)
        if pred_top is not None:
            target_classes = torch.argmax(y_top, dim=1)
            loss_top = torch.nn.functional.cross_entropy(pred_top, target_classes)

        loss = loss_br + topology_weight * loss_top

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds_br: list[np.ndarray] = []
    trues_br: list[np.ndarray] = []
    preds_top: list[np.ndarray] = []
    trues_top: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    model.eval()
    for features, y_br, branch_mask, y_top in loader:
        features = features.to(device, non_blocking=True)
        pred_br, pred_top = model(features)

        actual_preds = transformer.inverse_tensor(pred_br).cpu().numpy()
        actual_targets = transformer.inverse_tensor(y_br).cpu().numpy()

        preds_br.append(actual_preds)
        trues_br.append(actual_targets)
        masks.append(branch_mask.cpu().numpy())

        if pred_top is not None:
            preds_top.append(torch.softmax(pred_top, dim=1).cpu().numpy())
            trues_top.append(y_top.cpu().numpy())

    if not preds_top:
        return np.vstack(preds_br), np.vstack(trues_br), np.vstack(masks), np.array([]), np.array([])

    return np.vstack(preds_br), np.vstack(trues_br), np.vstack(masks), np.vstack(preds_top), np.vstack(trues_top)


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
    branch_mask = np.isfinite(true_vals) & np.isfinite(pred_vals) & (true_vals <= cutoff) & (pred_vals <= cutoff)
    if not np.any(branch_mask):
        return

    _plot_branch_pair(true_vals[branch_mask], pred_vals[branch_mask], output_path, title, bins=150)


def _summarize_branch_metrics(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
) -> dict[str, float]:
    mae = float(mean_absolute_error(true_vals, pred_vals))
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(true_vals, pred_vals))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def _plot_loss_curve(train_losses: Sequence[float], val_losses: Sequence[float], output_path: Path) -> None:
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


class CNNTrainer:
    """High-level orchestration for configuring, training, and evaluating the CNN."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = LabelTransformer(config.label_transform.strategy)

    def run(self) -> TrainingResult:
        start_time = time.time()
        data_cfg = self.config.data
        trainer_cfg = self.config.trainer

        print(f"DATASET FILE: {data_cfg.dataset_file}")
        print(
            "TRAINER SETTINGS | epochs={epochs}, patience={patience}, learning_rate={lr}, weight_decay={wd}".format(
                epochs=trainer_cfg.epochs,
                patience=trainer_cfg.patience,
                lr=trainer_cfg.learning_rate,
                wd=trainer_cfg.weight_decay,
            )
        )

        torch.manual_seed(data_cfg.seed)
        np.random.seed(data_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(data_cfg.seed)

        dataset = np.load(data_cfg.dataset_file, mmap_mode="r")
        if dataset.dtype.names is None or {"X", "y_br", "branch_mask", "y_top", "tree_index"} - set(
            dataset.dtype.names
        ):
            raise ValueError(
                "Dataset does not contain the expected structured fields (X, y_br, branch_mask, y_top, tree_index)"
            )

        example = dataset[0]
        if example["X"].ndim != 3:
            raise ValueError(
                "Encoded feature matrix must have shape (taxa, seq_length, nucleotides). "
                f"Observed shape: {example['X'].shape}"
            )
        num_clades, seq_length, num_nucleotides = example["X"].shape
        num_channels = num_nucleotides

        if self.config.model.num_taxa is not None and self.config.model.num_taxa != num_clades:
            raise ValueError(
                "Configured 'model.num_taxa' does not match dataset taxa count: "
                f"{self.config.model.num_taxa} != {num_clades}"
            )

        raw_y_br = dataset["y_br"]
        branch_mask = dataset["branch_mask"]
        y_top = dataset["y_top"]

        target_width = raw_y_br.shape[1]
        configured_outputs = self.config.model.num_outputs
        if configured_outputs is not None and configured_outputs != target_width:
            raise ValueError(
                "Configured 'model.num_outputs' does not match dataset target width: "
                f"{configured_outputs} != {target_width}"
            )
        effective_outputs = configured_outputs or target_width

        num_topology_classes = y_top.shape[1]

        transformed_y_br = self.transformer.transform_numpy(raw_y_br)

        model_settings = self.config.model
        if model_settings.in_channels != num_channels:
            print(
                "Adjusting model.in_channels from "
                f"{model_settings.in_channels} to match dataset channel count {num_channels}."
            )
            model_settings = replace(model_settings, in_channels=num_channels)

        model = CNNModel.from_config(
            model_settings,
            num_taxa=num_clades,
            num_outputs=effective_outputs,
            num_topology_classes=num_topology_classes,
            rooted=self.config.model.rooted,
            label_transform=self.transformer.strategy,
        )
        print(model)

        train_loader, val_loader, test_loader = _build_dataloaders(
            dataset,
            transformed_y_br,
            branch_mask,
            y_top,
            num_clades=num_clades,
            seq_length=seq_length,
            num_channels=num_channels,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            seed=data_cfg.seed,
            train_ratio=data_cfg.train_ratio,
            val_ratio=data_cfg.val_ratio,
        )

        print(f"Using device: {self.device}")
        model = model.to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trainer_cfg.learning_rate,
            weight_decay=trainer_cfg.weight_decay,
        )

        best_val_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        topology_weight = self.config.model.topology_weight

        for epoch in range(1, trainer_cfg.epochs + 1):
            train_loss = _train_epoch(
                model, train_loader, criterion, optimizer, self.device, self.transformer, topology_weight
            )
            val_loss = _evaluate(model, val_loader, criterion, self.device, self.transformer, topology_weight)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch:02d}/{trainer_cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if epochs_without_improvement >= trainer_cfg.patience:
                print(
                    "Early stopping triggered after "
                    f"{epoch} epochs (no val improvement for {trainer_cfg.patience} epochs)."
                )
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        test_loss = _evaluate(model, test_loader, criterion, self.device, self.transformer, topology_weight)
        print(f"Test Loss: {test_loss:.6f} (best val: {best_val_loss:.6f})")

        print("\n" + "=" * 60)
        print("BRANCH REGRESSION RESULTS")
        print("=" * 60 + "\n")

        branch_plot_folder = self.config.outputs.branch_plot_dir
        branch_plot_folder.mkdir(parents=True, exist_ok=True)

        # Always generate loss curve
        _plot_loss_curve(train_losses, val_losses, branch_plot_folder / "loss_curve.png")

        pred_br, true_br, masks, pred_top, true_top = _collect_predictions(
            model, test_loader, self.device, self.transformer
        )

        # Always iterate branches for metrics; plot conditionally
        for idx in range(pred_br.shape[1]):
            mask = masks[:, idx].astype(bool)
            if not np.any(mask):
                continue

            if self.config.outputs.individual_branch_plots:
                _plot_branch_pair(
                    true_br[mask, idx],
                    pred_br[mask, idx],
                    branch_plot_folder / f"branch_b{idx + 1}_scatter.png",
                    f"Branch b{idx + 1}: true vs predicted",
                )

                # Generate HD zoomed plots if enabled
                if self.config.outputs.zoomed_plots:
                    _plot_branch_pair_high_def(
                        true_br[mask, idx],
                        pred_br[mask, idx],
                        branch_plot_folder / f"branch_b{idx + 1}_scatter_hd.png",
                        f"Branch b{idx + 1} HD: true vs predicted",
                    )

            metrics = _summarize_branch_metrics(true_br[mask, idx], pred_br[mask, idx])
            print(
                "Individual Branch b{idx} Metrics \t | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}".format(
                    idx=idx + 1,
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    r2=metrics["r2"],
                )
            )

        # Generate branch sum plot if enabled
        if self.config.outputs.branch_sum_plots:
            # Sum all branches (masked)
            true_sum = np.sum(true_br * masks, axis=1)
            pred_sum = np.sum(pred_br * masks, axis=1)
            _plot_branch_pair(
                true_sum,
                pred_sum,
                branch_plot_folder / "branch_sum_scatter.png",
                "Branch sum: true vs predicted",
            )

        # Calculate overall metrics (masked)
        flat_true = true_br[masks.astype(bool)]
        flat_pred = pred_br[masks.astype(bool)]
        metrics = _summarize_branch_metrics(flat_true, flat_pred)
        print(
            "Overall Branch Sum Metrics \t | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}".format(
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
            )
        )
        print("\n" + "=" * 60 + "\n")

        # Topology classification metrics (if enabled)
        if self.config.model.topology_classification and pred_top.size > 0:
            print("\n" + "=" * 60)
            print("TOPOLOGY CLASSIFICATION RESULTS")
            print("=" * 60)

            pred_classes = np.argmax(pred_top, axis=1)
            true_classes = np.argmax(true_top, axis=1)

            # Accuracy
            acc = np.mean(pred_classes == true_classes)
            print(f"\nAccuracy: {acc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(true_classes, pred_classes)
            print("\nConfusion Matrix:")
            print(cm)

            # Precision, Recall, F1-Score
            precision, recall, f1, support = precision_recall_fscore_support(
                true_classes, pred_classes, average=None, zero_division=0
            )
            print("\nPer-class Metrics:")
            for i in range(len(precision)):
                print(
                    f"  Class {i}: Precision={precision[i]:.4f}, "
                    f"Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}"
                )

            # Macro averages
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)
            print("\nMacro Averages:")
            print(f"  Precision: {macro_precision:.4f}")
            print(f"  Recall: {macro_recall:.4f}")
            print(f"  F1-Score: {macro_f1:.4f}")

            # Weighted averages
            weighted_precision = np.average(precision, weights=support)
            weighted_recall = np.average(recall, weights=support)
            weighted_f1 = np.average(f1, weights=support)
            print("\nWeighted Averages:")
            print(f"  Precision: {weighted_precision:.4f}")
            print(f"  Recall: {weighted_recall:.4f}")
            print(f"  F1-Score: {weighted_f1:.4f}")
            print("=" * 60 + "\n")

        elapsed = time.time() - start_time
        print(f"TOTAL RUNTIME: {elapsed:.2f} seconds\n")

        return TrainingResult(
            best_val_loss=best_val_loss,
            test_loss=test_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            runtime_seconds=elapsed,
        )
