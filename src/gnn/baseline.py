"""Neighbor-joining baseline and GNN comparison for phylogenetic topology accuracy."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

from .simulate import SimulatedTree, generate_dataset
from .data import create_dataloaders
from .model import AgglomerativePhyloGNN
from .train import train_epoch, evaluate


# ---------------------------------------------------------------------------
# Topology representation: frozenset of frozensets (unrooted splits)
# For n_taxa leaves an unrooted binary tree has n_taxa-3 non-trivial splits
# (both sides have >= 2 leaves). e.g. n_taxa=4 → 1 split, n_taxa=8 → 5 splits.
# ---------------------------------------------------------------------------

def get_true_topology(sample: SimulatedTree) -> frozenset:
    """Extract unrooted topology from a SimulatedTree as a frozenset of splits."""
    n_taxa = sample.n_taxa
    all_leaves = frozenset(range(n_taxa))

    desc: dict[int, frozenset] = {i: frozenset({i}) for i in range(n_taxa)}
    for step, (left, right) in enumerate(sample.merge_order):
        desc[n_taxa + step] = desc[left] | desc[right]

    splits: set[frozenset] = set()
    for step in range(n_taxa - 1):
        clade = desc[n_taxa + step]
        complement = all_leaves - clade
        if len(clade) >= 2 and len(complement) >= 2:
            splits.add(frozenset({clade, complement}))

    return frozenset(splits)


def run_nj(sample: SimulatedTree) -> frozenset:
    """Run neighbor-joining on leaf sequences and return unrooted topology as frozenset of splits."""
    n_taxa = sample.n_taxa
    seqs = sample.leaf_sequences
    names = [str(i) for i in range(n_taxa)]

    # Lower-triangular p-distance matrix required by BioPython
    matrix: list[list[float]] = []
    for i in range(n_taxa):
        row: list[float] = []
        for j in range(i + 1):
            if i == j:
                row.append(0.0)
            else:
                row.append(float((seqs[i] != seqs[j]).mean()))
        matrix.append(row)

    dm = DistanceMatrix(names, matrix)
    tree = DistanceTreeConstructor().nj(dm)

    all_leaves_int = frozenset(range(n_taxa))
    splits: set[frozenset] = set()
    for clade in tree.get_nonterminals():
        clade_leaves = frozenset(int(leaf.name) for leaf in clade.get_terminals())
        complement = all_leaves_int - clade_leaves
        if len(clade_leaves) >= 2 and len(complement) >= 2:
            splits.add(frozenset({clade_leaves, complement}))

    return frozenset(splits)


def _gnn_predict_topology(
    model: AgglomerativePhyloGNN,
    sample: SimulatedTree,
    device: torch.device,
) -> frozenset:
    """Run one forward pass and extract predicted topology via hard argmax merges."""
    model.eval()
    seqs = torch.from_numpy(sample.leaf_sequences).long()
    # (n_taxa, seq_len, 4)
    one_hot = F.one_hot(seqs, num_classes=4).float().to(device)

    n_taxa = sample.n_taxa
    all_leaves = frozenset(range(n_taxa))

    with torch.no_grad():
        embeds = model.leaf_encoder(one_hot)  # (n_taxa, seq_len, hidden_dim)

        pool: list[torch.Tensor] = [embeds[i] for i in range(n_taxa)]
        leaf_sets: list[frozenset] = [frozenset({i}) for i in range(n_taxa)]
        merges: list[tuple[frozenset, frozenset]] = []

        for _step in range(n_taxa - 1):
            k = len(pool)
            pair_indices = [(i, j) for i in range(k) for j in range(i + 1, k)]
            scores_list = []
            for i, j in pair_indices:
                site_pairs = torch.cat([pool[i], pool[j]], dim=-1)  # (seq_len, 2*hidden_dim)
                scores_list.append(model.merge_scorer(site_pairs).mean())
            scores = torch.stack(scores_list)

            si, sj = pair_indices[scores.argmax().item()]
            merged_set = leaf_sets[si] | leaf_sets[sj]
            merges.append((leaf_sets[si], leaf_sets[sj]))

            parent_input = torch.cat([pool[si], pool[sj]], dim=-1)
            new_parent = model.message_fn(parent_input)

            pool = [pool[idx] for idx in range(k) if idx != si and idx != sj]
            pool.append(new_parent)

            leaf_sets = [leaf_sets[idx] for idx in range(k) if idx != si and idx != sj]
            leaf_sets.append(merged_set)

    splits: set[frozenset] = set()
    for left_set, right_set in merges:
        clade = left_set | right_set
        complement = all_leaves - clade
        if len(clade) >= 2 and len(complement) >= 2:
            splits.add(frozenset({clade, complement}))

    return frozenset(splits)


def _train_gnn(
    train_samples: list[SimulatedTree],
    *,
    hidden_dim: int = 64,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    early_stopping_patience: int = 15,
    temperature_start: float = 2.0,
    temperature_end: float = 0.3,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    seed: int = 99,
) -> AgglomerativePhyloGNN:
    """Train an AgglomerativePhyloGNN on pre-split training samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        train_samples, batch_size=batch_size, seed=seed,
    )
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = AgglomerativePhyloGNN(
        hidden_dim=hidden_dim, temperature=temperature_start,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        frac = (epoch - 1) / max(num_epochs - 1, 1)
        model.temperature = temperature_start + frac * (temperature_end - temperature_start)

        train_m = train_epoch(model, train_loader, optimizer, device, beta=beta, gamma=gamma, delta=delta)
        val_m = evaluate(model, val_loader, device, beta=beta, gamma=gamma, delta=delta)
        scheduler.step(val_m.total_loss)

        marker = ""
        if val_m.total_loss < best_val_loss:
            best_val_loss = val_m.total_loss
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1

        print(
            f"  Epoch {epoch:3d} (tau={model.temperature:.2f}) | "
            f"Train {train_m.total_loss:.4f} | "
            f"Val {val_m.total_loss:.4f} | "
            f"Anc {val_m.ancestral_accuracy:.3f} | "
            f"Topo {val_m.topology_loss:.4f} | "
            f"TopoAcc {val_m.topology_accuracy:.3f}{marker}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"  Early stop: no improvement for {early_stopping_patience} epochs.")
            break

    print(f"  Best val loss: {best_val_loss:.4f}")
    return model


def run_comparison(
    n_samples: int = 20000,
    n_taxa: int = 8,
    seq_length: int = 300,
    branch_length_range: tuple[float, float] = (0.005, 0.2),
    seed: int = 99,
    hidden_dim: int = 64,
    batch_size: int = 32,
    num_epochs: int = 100,
    early_stopping_patience: int = 15,
    temperature_start: float = 2.0,
    temperature_end: float = 0.3,
) -> dict:
    """Train GNN and evaluate both GNN and NJ on held-out trees."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating {n_samples} trees (n_taxa={n_taxa}, seq_length={seq_length}, seed={seed})...")
    all_samples = generate_dataset(
        n_samples, n_taxa=n_taxa, seq_length=seq_length,
        branch_length_range=branch_length_range, seed=seed,
    )

    split_idx = int(n_samples * 0.8)
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    print(f"Split: {len(train_samples)} train / {len(eval_samples)} eval\n")

    print(f"Training GNN (max {num_epochs} epochs, early stop patience={early_stopping_patience})...")
    model = _train_gnn(
        train_samples,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        seed=seed,
    )
    model.eval()
    print()

    print("Evaluating NJ and GNN on held-out trees...")
    per_tree_results: list[dict] = []

    for idx, sample in enumerate(eval_samples):
        true_topo = get_true_topology(sample)
        nj_topo = run_nj(sample)
        gnn_topo = _gnn_predict_topology(model, sample, device)

        per_tree_results.append({
            "tree_idx": idx,
            "nj_correct": nj_topo == true_topo,
            "gnn_correct": gnn_topo == true_topo,
            "branch_lengths_mean": float(sample.branch_lengths.mean()),
        })

    n_eval = len(eval_samples)
    nj_acc = sum(r["nj_correct"] for r in per_tree_results) / n_eval
    gnn_acc = sum(r["gnn_correct"] for r in per_tree_results) / n_eval

    return {
        "nj_accuracy": nj_acc,
        "gnn_accuracy": gnn_acc,
        "n_eval_trees": n_eval,
        "n_taxa": n_taxa,
        "seq_length": seq_length,
        "branch_length_range": branch_length_range,
        "per_tree_results": per_tree_results,
    }


if __name__ == "__main__":
    results = run_comparison()

    print("\n" + "=" * 50)
    print("TOPOLOGY ACCURACY COMPARISON")
    print("=" * 50)
    print(f"  {'Method':<20} {'Accuracy':>10}  ({'n=' + str(results['n_eval_trees'])})")
    print(f"  {'-'*42}")
    print(f"  {'Neighbor-Joining':<20} {results['nj_accuracy']:>9.1%}")
    print(f"  {'GNN (trained)':<20} {results['gnn_accuracy']:>9.1%}")
    print("=" * 50)
    print(f"  n_taxa={results['n_taxa']}, seq_length={results['seq_length']}, "
          f"branch_length_range={results['branch_length_range']}")
