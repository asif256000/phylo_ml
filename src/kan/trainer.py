from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from kan import KAN
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from torch.utils.data import DataLoader, Dataset

from src.configuration.training import KANSettings, TrainingConfig


class LabelTransformer:
    """Applies optional transforms to branch-length targets."""

    def __init__(self, strategy: str) -> None:
        strategy_normalized = strategy.lower()
        if strategy_normalized not in {"sqrt", "log", "none"}:
            raise ValueError("KAN trainer supports 'sqrt', 'log', or 'none' label transforms")
        self.strategy = strategy_normalized

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


@dataclass
class TrainingResult:
    best_val_loss: float
    test_loss: float
    train_losses: list[float]
    val_losses: list[float]
    runtime_seconds: float
    metrics: dict[str, float]


class FlatSequenceDataset(Dataset):
    """Wrap structured NumPy records and expose flattened tensors suitable for KAN."""

    def __init__(
        self,
        data: np.memmap,
        indices: np.ndarray,
        y_br: np.ndarray,
        branch_mask: np.ndarray,
        flattened_length: int,
    ) -> None:
        self.data = data
        self.indices = indices
        self.y_br = np.ascontiguousarray(y_br.astype(np.float32, copy=False))
        self.branch_mask = np.ascontiguousarray(branch_mask.astype(np.float32, copy=False))
        self.flattened_length = flattened_length

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.data[self.indices[idx]]
        features = np.asarray(record["X"], dtype=np.float32)
        flat = np.reshape(features, (self.flattened_length,))
        flat_tensor = torch.from_numpy(flat)
        target = torch.from_numpy(self.y_br[self.indices[idx]])
        mask = torch.from_numpy(self.branch_mask[self.indices[idx]])
        return flat_tensor, target, mask


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
    branch_mask: np.ndarray,
    flattened_length: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_idx, val_idx, test_idx = split_indices(len(data), train_ratio, val_ratio, seed)

    train_dataset = FlatSequenceDataset(data, train_idx, y_br, branch_mask, flattened_length)
    val_dataset = FlatSequenceDataset(data, val_idx, y_br, branch_mask, flattened_length)
    test_dataset = FlatSequenceDataset(data, test_idx, y_br, branch_mask, flattened_length)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def _ensure_tensor_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(1)
    return tensor


def _train_epoch(
    model: KAN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    for features, target, mask in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = _ensure_tensor_2d(model(features))
        masked_diff = (preds - target) * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        loss = torch.sum(masked_diff ** 2) / denom
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


@torch.no_grad()
def _evaluate(
    model: KAN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    for features, target, mask in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        preds = _ensure_tensor_2d(model(features))
        masked_diff = (preds - target) * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        loss = torch.sum(masked_diff ** 2) / denom

        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
    return running_loss / total_samples


@torch.no_grad()
def _collect_predictions(
    model: KAN,
    loader: DataLoader,
    device: torch.device,
    transformer: LabelTransformer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    model.eval()
    for features, target, mask in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        outputs = _ensure_tensor_2d(model(features))
        preds.append(transformer.inverse_tensor(outputs).cpu().numpy())
        trues.append(transformer.inverse_tensor(target).cpu().numpy())
        masks.append(mask.cpu().numpy())
    if not preds:
        return np.array([]), np.array([]), np.array([])
    return np.vstack(preds), np.vstack(trues), np.vstack(masks)


def _flatten_metrics(preds: np.ndarray, trues: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    if preds.size == 0:
        return {}
    mask_flat = masks.astype(bool).reshape(-1)
    true_flat = trues.reshape(-1)
    pred_flat = preds.reshape(-1)
    valid_true = true_flat[mask_flat]
    valid_pred = pred_flat[mask_flat]
    if valid_true.size == 0:
        return {}
    mae = mean_absolute_error(valid_true, valid_pred)
    mse = mean_squared_error(valid_true, valid_pred)
    rmse = root_mean_squared_error(valid_true, valid_pred)
    r2 = r2_score(valid_true, valid_pred)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


class KANTrainer:
    """Orchestrates training for the Kolmogorov-Arnold Network module."""

    def __init__(self, config: TrainingConfig) -> None:
        if config.model.kan_settings is None:
            raise ValueError("model.kan section must be provided for KAN training.")
        if config.model.topology_classification:
            raise ValueError("KANTrainer does not yet support topology classification targets.")

        self.config = config
        self.kan_settings = config.model.kan_settings
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
        expected_fields = {"X", "y_br", "branch_mask", "tree_index"}
        if dataset.dtype.names is None or expected_fields - set(dataset.dtype.names):
            raise ValueError("Dataset must contain fields X, y_br, branch_mask, tree_index")

        example = dataset[0]
        if example["X"].ndim != 3:
            raise ValueError("Encoded features must have shape (taxa, seq_length, channels)")
        num_clades, seq_length, num_channels = example["X"].shape
        flattened_length = num_clades * seq_length * num_channels

        raw_y_br = dataset["y_br"].astype(np.float32, copy=False)
        branch_mask = dataset["branch_mask"].astype(np.float32, copy=False)

        target_width = raw_y_br.shape[1]
        configured_outputs = self.config.model.num_outputs
        if configured_outputs is not None and configured_outputs != target_width:
            raise ValueError(
                "Configured 'model.num_outputs' does not match dataset target width: "
                f"{configured_outputs} != {target_width}"
            )
        effective_outputs = configured_outputs or target_width

        transformed_targets = self.transformer.transform_numpy(raw_y_br)

        train_loader, val_loader, test_loader = _build_dataloaders(
            dataset,
            transformed_targets,
            branch_mask,
            flattened_length,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            seed=data_cfg.seed,
            train_ratio=data_cfg.train_ratio,
            val_ratio=data_cfg.val_ratio,
        )

        kan_model = self._initialize_model(flattened_length, effective_outputs)
        kan_model.to(self.device)

        optimizer = torch.optim.Adam(
            kan_model.parameters(),
            lr=trainer_cfg.learning_rate,
            weight_decay=trainer_cfg.weight_decay,
        )

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, trainer_cfg.epochs + 1):
            train_loss = _train_epoch(kan_model, train_loader, optimizer, self.device)
            val_loss = _evaluate(kan_model, val_loader, self.device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {key: val.detach().clone() for key, val in kan_model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch:02d}/{trainer_cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if epochs_without_improvement >= trainer_cfg.patience:
                print(
                    "Early stopping triggered after "
                    f"{epoch} epochs (no val improvement for {trainer_cfg.patience} epochs)."
                )
                break

        if best_state is not None:
            kan_model.load_state_dict(best_state)

        test_loss = _evaluate(kan_model, test_loader, self.device)
        print(f"Test Loss: {test_loss:.6f} (best val: {best_val_loss:.6f})")

        preds, trues, masks = _collect_predictions(kan_model, test_loader, self.device, self.transformer)
        metrics = _flatten_metrics(preds, trues, masks)
        runtime = time.time() - start_time

        return TrainingResult(
            best_val_loss=best_val_loss,
            test_loss=test_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            runtime_seconds=runtime,
            metrics=metrics,
        )

    def _initialize_model(self, input_dim: int, output_dim: int) -> KAN:
        settings: KANSettings = self.kan_settings
        width = [input_dim, *settings.hidden_layers, output_dim]
        return KAN(
            width=width,
            grid=settings.grid,
            k=settings.spline_order,
            mult_arity=settings.mult_arity,
            noise_scale=settings.noise_scale,
            base_fun=settings.base_function,
            symbolic_enabled=settings.symbolic_enabled,
            affine_trainable=settings.affine_trainable,
            grid_eps=settings.grid_eps,
            grid_range=list(settings.grid_range),
            sp_trainable=settings.sp_trainable,
            sb_trainable=settings.sb_trainable,
            sparse_init=settings.sparse_init,
            auto_save=settings.auto_save,
            device=str(self.device),
        )


def run_training(config: TrainingConfig) -> TrainingResult:
    trainer = KANTrainer(config)
    return trainer.run()
***