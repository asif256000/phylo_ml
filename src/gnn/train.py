"""Training loop for the AgglomerativePhyloGNN with unsupervised tree construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .model import AgglomerativePhyloGNN


@dataclass
class EpochMetrics:
    """Aggregated metrics for a single training or evaluation epoch."""

    total_loss: float
    ancestral_loss: float
    branch_loss: float
    topology_loss: float
    ancestral_accuracy: float
    topology_accuracy: float


def _unbatch(batch_data: object) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int,
]:
    """Reshape PyG concatenated tensors back to (batch, ...) form."""
    n_taxa = batch_data.n_taxa[0].item()
    n_internal = n_taxa - 1
    bs = batch_data.n_taxa.size(0)

    leaf_seqs = batch_data.leaf_seqs.reshape(bs, n_taxa, -1, 4)
    true_anc = batch_data.true_ancestral.reshape(bs, n_internal, -1)
    true_br = batch_data.true_branches.reshape(bs, 2 * n_internal)
    true_merge_order = batch_data.merge_order.reshape(bs, n_internal, 2)

    return leaf_seqs, true_anc, true_br, true_merge_order, bs, n_internal


def _topo_acc_from_logits(
    all_merge_logits: list[list[torch.Tensor]],
    true_merge_order: torch.Tensor,
    n_taxa: int,
) -> float:
    """Fraction of trees in a batch whose hard-decision topology matches the true topology."""
    all_leaves = frozenset(range(n_taxa))
    correct = 0

    for b, sample_logits in enumerate(all_merge_logits):
        # Build true splits from simulator merge_order
        desc: dict[int, frozenset] = {i: frozenset({i}) for i in range(n_taxa)}
        for step in range(n_taxa - 1):
            left = true_merge_order[b, step, 0].item()
            right = true_merge_order[b, step, 1].item()
            desc[n_taxa + step] = desc[left] | desc[right]
        true_splits: set[frozenset] = {
            frozenset({desc[n_taxa + s], all_leaves - desc[n_taxa + s]})
            for s in range(n_taxa - 1)
            if len(desc[n_taxa + s]) >= 2 and len(all_leaves - desc[n_taxa + s]) >= 2
        }

        # Replay hard argmax decisions to recover predicted splits
        leaf_sets: list[frozenset] = [frozenset({i}) for i in range(n_taxa)]
        pred_splits: set[frozenset] = set()
        for scores in sample_logits:
            k = len(leaf_sets)
            pair_indices = [(i, j) for i in range(k) for j in range(i + 1, k)]
            si, sj = pair_indices[scores.argmax().item()]
            clade = leaf_sets[si] | leaf_sets[sj]
            complement = all_leaves - clade
            if len(clade) >= 2 and len(complement) >= 2:
                pred_splits.add(frozenset({clade, complement}))
            merged = leaf_sets[si] | leaf_sets[sj]
            leaf_sets = [leaf_sets[idx] for idx in range(k) if idx != si and idx != sj]
            leaf_sets.append(merged)

        if frozenset(pred_splits) == frozenset(true_splits):
            correct += 1

    return correct / len(all_merge_logits)


def compute_loss(
    ancestral_logits: torch.Tensor,
    branch_lengths: torch.Tensor,
    true_ancestral: torch.Tensor,
    true_branches: torch.Tensor,
    topology_loss: torch.Tensor,
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reconstruction loss (ancestral CE + branch MSE) plus direct topology supervision."""
    anc_loss = F.cross_entropy(
        ancestral_logits.reshape(-1, 4),
        true_ancestral.reshape(-1),
    )
    br_loss = F.mse_loss(branch_lengths, true_branches)
    total = beta * anc_loss + gamma * br_loss + delta * topology_loss
    return total, anc_loss, br_loss, topology_loss


def train_epoch(
    model: AgglomerativePhyloGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> EpochMetrics:
    """Run one training epoch and return aggregated metrics."""
    model.train()
    total_loss = 0.0
    total_anc = 0.0
    total_br = 0.0
    total_topo = 0.0
    correct_anc = 0
    total_anc_sites = 0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        leaf_seqs, true_anc, true_br, true_merge_order, bs, n_internal = _unbatch(batch)

        _merge_logits, anc_logits, br_pred, topo_loss = model(leaf_seqs, true_merge_order)

        loss, a_loss, b_loss, t_loss = compute_loss(
            anc_logits, br_pred,
            true_anc, true_br,
            topo_loss,
            beta=beta, gamma=gamma, delta=delta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        num_samples += bs
        total_loss += loss.item() * bs
        total_anc += a_loss.item() * bs
        total_br += b_loss.item() * bs
        total_topo += t_loss.item() * bs

        pred_anc = anc_logits.argmax(dim=-1)
        correct_anc += (pred_anc == true_anc).sum().item()
        total_anc_sites += true_anc.numel()

    return EpochMetrics(
        total_loss=total_loss / num_samples,
        ancestral_loss=total_anc / num_samples,
        branch_loss=total_br / num_samples,
        topology_loss=total_topo / num_samples,
        ancestral_accuracy=correct_anc / max(total_anc_sites, 1),
        topology_accuracy=0.0,
    )


@torch.no_grad()
def evaluate(
    model: AgglomerativePhyloGNN,
    loader: DataLoader,
    device: torch.device,
    *,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> EpochMetrics:
    """Evaluate the model on a validation set and return metrics."""
    model.eval()
    total_loss = 0.0
    total_anc = 0.0
    total_br = 0.0
    total_topo = 0.0
    correct_anc = 0
    total_anc_sites = 0
    correct_topo = 0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        leaf_seqs, true_anc, true_br, true_merge_order, bs, n_internal = _unbatch(batch)
        n_taxa = n_internal + 1

        merge_logits, anc_logits, br_pred, topo_loss = model(leaf_seqs, true_merge_order)

        loss, a_loss, b_loss, t_loss = compute_loss(
            anc_logits, br_pred,
            true_anc, true_br,
            topo_loss,
            beta=beta, gamma=gamma, delta=delta,
        )

        num_samples += bs
        total_loss += loss.item() * bs
        total_anc += a_loss.item() * bs
        total_br += b_loss.item() * bs
        total_topo += t_loss.item() * bs

        pred_anc = anc_logits.argmax(dim=-1)
        correct_anc += (pred_anc == true_anc).sum().item()
        total_anc_sites += true_anc.numel()

        batch_topo_acc = _topo_acc_from_logits(merge_logits, true_merge_order, n_taxa)
        correct_topo += round(batch_topo_acc * bs)

    return EpochMetrics(
        total_loss=total_loss / num_samples,
        ancestral_loss=total_anc / num_samples,
        branch_loss=total_br / num_samples,
        topology_loss=total_topo / num_samples,
        ancestral_accuracy=correct_anc / max(total_anc_sites, 1),
        topology_accuracy=correct_topo / max(num_samples, 1),
    )


def run_training(
    *,
    n_taxa: int = 4,
    num_samples: int = 10_000,
    seq_length: int = 200,
    branch_length_range: tuple[float, float] = (0.005, 0.2),
    hidden_dim: int = 64,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    seed: int = 42,
) -> AgglomerativePhyloGNN:
    """Generate data, train the AgglomerativePhyloGNN, and report results."""
    from .data import create_dataloaders
    from .simulate import generate_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Generating {num_samples} simulated trees "
        f"(n_taxa={n_taxa}, seq_length={seq_length})...",
    )
    samples = generate_dataset(
        num_samples, n_taxa=n_taxa, seq_length=seq_length,
        branch_length_range=branch_length_range, seed=seed,
    )

    train_loader, val_loader = create_dataloaders(
        samples, batch_size=batch_size, seed=seed,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = AgglomerativePhyloGNN(
        hidden_dim=hidden_dim, temperature=temperature_start,
    ).to(device)
    print(f"\n{model}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        frac = (epoch - 1) / max(num_epochs - 1, 1)
        model.temperature = temperature_start + frac * (temperature_end - temperature_start)

        train_m = train_epoch(
            model, train_loader, optimizer, device,
            beta=beta, gamma=gamma, delta=delta,
        )
        val_m = evaluate(
            model, val_loader, device,
            beta=beta, gamma=gamma, delta=delta,
        )

        scheduler.step(val_m.total_loss)
        marker = ""
        if val_m.total_loss < best_val_loss:
            best_val_loss = val_m.total_loss
            marker = " *"

        print(
            f"Epoch {epoch:3d} "
            f"(tau={model.temperature:.2f}) | "
            f"Train {train_m.total_loss:.4f} | "
            f"Val {val_m.total_loss:.4f} | "
            f"Anc {val_m.ancestral_accuracy:.3f} | "
            f"Topo {val_m.topology_loss:.4f} | "
            f"BL {val_m.branch_loss:.6f}{marker}"
        )

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return model
