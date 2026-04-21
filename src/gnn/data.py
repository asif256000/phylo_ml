"""PyTorch Geometric data pipeline for phylogenetic tree datasets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .simulate import SimulatedTree


def _to_onehot(int_seq: np.ndarray) -> torch.Tensor:
    """Convert integer-encoded sequences to one-hot float tensors."""
    tensor = torch.from_numpy(int_seq).long()
    return torch.nn.functional.one_hot(tensor, num_classes=4).float()


def tree_to_pyg_data(sample: SimulatedTree) -> Data:
    """Convert a SimulatedTree into a PyG Data object."""
    return Data(
        leaf_seqs=_to_onehot(sample.leaf_sequences),
        true_ancestral=torch.from_numpy(sample.ancestral_sequences).long(),
        true_branches=torch.from_numpy(sample.branch_lengths).float(),
        n_taxa=torch.tensor(sample.n_taxa, dtype=torch.long),
        merge_order=torch.tensor(sample.merge_order, dtype=torch.long),  # (n_taxa-1, 2)
    )


def create_dataloaders(
    samples: Sequence[SimulatedTree],
    train_fraction: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split samples into train/val sets and return DataLoaders."""
    data_list = [tree_to_pyg_data(s) for s in samples]

    generator = torch.Generator().manual_seed(seed)
    total = len(data_list)
    indices = torch.randperm(total, generator=generator).tolist()

    train_size = int(total * train_fraction)
    train_data = [data_list[i] for i in indices[:train_size]]
    val_data = [data_list[i] for i in indices[train_size:]]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
