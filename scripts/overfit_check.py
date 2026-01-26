from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from src.cnn.config import ConfigurationError, load_training_config
from src.cnn.model import CNNModel
from src.cnn.train import LabelTransformer, SequenceDataset


def _split_outputs(outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]
    return outputs, None


class LinearProbe(torch.nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_outputs: int, num_classes: int) -> None:
        super().__init__()
        channels, taxa, seq_len = input_shape
        self.flatten = torch.nn.Flatten()
        self.topology_head = torch.nn.Linear(channels * taxa * seq_len, num_classes)
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.topology_head(self.flatten(x))
        zeros = torch.zeros((x.size(0), self.num_outputs), device=x.device, dtype=logits.dtype)
        return zeros, logits


def _build_balanced_subset(
    labels: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {cls}: requested {samples_per_class}, found {cls_idx.size}"
            )
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:samples_per_class].tolist())
    indices_arr = np.array(indices, dtype=int)
    rng.shuffle(indices_arr)
    return indices_arr


def _report_duplicate_inputs(
    dataset: np.memmap,
    indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    seen: dict[str, int] = {}
    conflicts = 0
    duplicates = 0
    for idx in indices:
        x = np.ascontiguousarray(dataset[idx]["X"]).tobytes()
        digest = hashlib.md5(x).hexdigest()
        if digest in seen:
            duplicates += 1
            if seen[digest] != int(labels[idx]):
                conflicts += 1
        else:
            seen[digest] = int(labels[idx])
    if duplicates == 0:
        print("No duplicate X samples found in subset.")
    else:
        print(
            f"Duplicate X samples in subset: {duplicates} (label conflicts: {conflicts})"
        )


def _report_duplicate_inputs_full(dataset: np.memmap, labels: np.ndarray) -> None:
    seen: dict[str, int] = {}
    duplicates = 0
    conflicts = 0
    for idx in range(len(dataset)):
        x = np.ascontiguousarray(dataset[idx]["X"]).tobytes()
        digest = hashlib.md5(x).hexdigest()
        label = int(labels[idx])
        if digest in seen:
            duplicates += 1
            if seen[digest] != label:
                conflicts += 1
        else:
            seen[digest] = label
    if duplicates == 0:
        print("No duplicate X samples found in full dataset.")
    else:
        print(
            f"Duplicate X samples in full dataset: {duplicates} (label conflicts: {conflicts})"
        )


def _train_val_split(indices: np.ndarray, labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels):
        cls_indices = indices[labels[indices] == cls]
        rng.shuffle(cls_indices)
        val_size = max(1, int(len(cls_indices) * val_ratio))
        val_idx.extend(cls_indices[:val_size])
        train_idx.extend(cls_indices[val_size:])
    train_idx_arr = np.array(train_idx, dtype=int)
    val_idx_arr = np.array(val_idx, dtype=int)
    rng.shuffle(train_idx_arr)
    rng.shuffle(val_idx_arr)
    return train_idx_arr, val_idx_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit check for topology classification")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/training_test_4t.yaml"),
        help="Path to YAML/JSON training config",
    )
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--topology-weight", type=float, default=None)
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--disable-dropout", action="store_true")
    parser.add_argument("--check-duplicates-full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    args = parser.parse_args()

    try:
        cfg = load_training_config(args.config)
    except (FileNotFoundError, ConfigurationError, ValueError) as exc:
        raise SystemExit(f"Failed to load training configuration: {exc}") from exc

    if not cfg.model.topology_classification:
        raise SystemExit("Topology classification is disabled in config. Enable model.topology_classification first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = np.load(cfg.data.dataset_file, mmap_mode="r")
    expected_fields = {"X", "y_br", "y_top"}
    if dataset.dtype.names is None or expected_fields - set(dataset.dtype.names):
        raise ValueError(f"Dataset must contain structured fields {sorted(expected_fields)}")

    example = dataset[0]
    if example["X"].ndim != 3:
        raise ValueError("Encoded feature matrix must have shape (taxa, seq_length, channels)")
    num_clades, seq_length, num_channels = example["X"].shape

    y_br_raw = dataset["y_br"]
    if y_br_raw.ndim == 1:
        y_br_raw = y_br_raw[:, None]
    y_br_raw = y_br_raw.astype(np.float32, copy=False)

    transformer = LabelTransformer(cfg.label_transform.strategy)
    y_br = transformer.transform_numpy(y_br_raw)

    y_top_raw = dataset["y_top"].astype(np.float32, copy=False)
    if y_top_raw.ndim == 1:
        y_top_raw = y_top_raw[:, None]
    num_topology_classes = int(y_top_raw.shape[1])
    labels = np.argmax(y_top_raw, axis=1)

    if args.check_duplicates_full:
        _report_duplicate_inputs_full(dataset, labels)

    subset_indices = _build_balanced_subset(labels, args.samples_per_class, cfg.seed)
    train_idx, val_idx = _train_val_split(subset_indices, labels, args.val_ratio, cfg.seed)

    print(f"DATASET FILE: {cfg.data.dataset_file}")
    print(f"Using {args.samples_per_class} samples per class (total {len(subset_indices)}).")
    unique, counts = np.unique(labels[subset_indices], return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")
    _report_duplicate_inputs(dataset, subset_indices, labels)

    train_dataset = SequenceDataset(dataset, train_idx, y_br, y_top_raw, num_clades, seq_length, num_channels)
    val_dataset = SequenceDataset(dataset, val_idx, y_br, y_top_raw, num_clades, seq_length, num_channels)

    if args.full_batch:
        batch_size = len(train_dataset)
    else:
        batch_size = args.batch_size or cfg.data.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.linear_probe and not args.topology_only:
        raise SystemExit("--linear-probe requires --topology-only (no regression head).")

    if args.linear_probe:
        model = LinearProbe(
            (cfg.model.in_channels or num_channels, num_clades, seq_length),
            y_br.shape[1],
            num_topology_classes,
        ).to(device)
    else:
        model = CNNModel(
            num_taxa=num_clades,
            num_outputs=y_br.shape[1],
            in_channels=cfg.model.in_channels or num_channels,
            label_transform=cfg.label_transform.strategy,
            tree_rooted=cfg.model.rooted,
            topology_classification=True,
            num_topology_classes=num_topology_classes,
        ).to(device)

    if args.disable_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    lr = args.learning_rate if args.learning_rate is not None else cfg.trainer.learning_rate
    topo_weight = args.topology_weight if args.topology_weight is not None else cfg.model.topology_weight
    reg_weight = 0.0 if args.topology_only else 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.trainer.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for features, y_br_batch, y_top_batch in train_loader:
            features = features.to(device, non_blocking=True)
            y_br_batch = y_br_batch.to(device, non_blocking=True)
            y_top_batch = y_top_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(features)
            pred_br, pred_top = _split_outputs(outputs)
            if pred_top is None:
                raise RuntimeError("Topology head missing")

            target_classes = torch.argmax(y_top_batch, dim=1)
            loss_top = ce(pred_top, target_classes)
            loss = topo_weight * loss_top
            if reg_weight > 0:
                loss = loss + reg_weight * mse(pred_br, y_br_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            total += features.size(0)
            preds = torch.argmax(pred_top, dim=1)
            correct += (preds == target_classes).sum().item()

        train_acc = correct / max(1, total)
        train_loss = running_loss / max(1, total)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for features, _, y_top_batch in val_loader:
                    features = features.to(device, non_blocking=True)
                    y_top_batch = y_top_batch.to(device, non_blocking=True)
                    outputs = model(features)
                    _, pred_top = _split_outputs(outputs)
                    if pred_top is None:
                        raise RuntimeError("Topology head missing")
                    target_classes = torch.argmax(y_top_batch, dim=1)
                    preds = torch.argmax(pred_top, dim=1)
                    val_total += features.size(0)
                    val_correct += (preds == target_classes).sum().item()
            val_acc = val_correct / max(1, val_total)
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

    print("\nOverfit check done. If Train Acc is near 1.0, the classifier can fit the data.")


if __name__ == "__main__":
    main()
