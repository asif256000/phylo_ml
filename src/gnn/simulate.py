"""JC69 sequence evolution simulator for arbitrary tree sizes with full ancestral ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUCLEOTIDES = "ACGT"


@dataclass
class SimulatedTree:
    """Container for a simulated tree with ground truth for all training targets."""

    n_taxa: int
    leaf_sequences: np.ndarray
    ancestral_sequences: np.ndarray
    branch_lengths: np.ndarray
    merge_order: list[tuple[int, int]]


def jc69_evolve(
    parent: np.ndarray,
    branch_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve a sequence along a single branch under the Jukes-Cantor 69 model."""
    p_same = 0.25 + 0.75 * np.exp(-4.0 * branch_length / 3.0)
    child = parent.copy()
    mutate_mask = rng.random(len(parent)) >= p_same
    num_mutations = int(mutate_mask.sum())
    if num_mutations > 0:
        offsets = rng.integers(1, 4, size=num_mutations)
        child[mutate_mask] = (parent[mutate_mask] + offsets) % 4
    return child


def simulate_tree(
    n_taxa: int,
    seq_length: int,
    branch_length_range: tuple[float, float],
    rng: np.random.Generator,
) -> SimulatedTree:
    """Simulate a random binary tree with n_taxa leaves and evolve sequences."""
    lo, hi = branch_length_range
    n_internal = n_taxa - 1
    total_nodes = 2 * n_taxa - 1

    pool = list(range(n_taxa))
    merge_order: list[tuple[int, int]] = []

    for step in range(n_internal):
        k = len(pool)
        idx_i, idx_j = sorted(rng.choice(k, size=2, replace=False))

        node_left = pool[idx_i]
        node_right = pool[idx_j]
        new_node = n_taxa + step
        merge_order.append((node_left, node_right))

        pool = [pool[p] for p in range(k) if p != idx_i and p != idx_j]
        pool.append(new_node)

    branch_lengths = rng.uniform(lo, hi, size=2 * n_internal).astype(np.float32)

    sequences = np.empty((total_nodes, seq_length), dtype=np.int64)
    root_id = n_taxa + n_internal - 1
    sequences[root_id] = rng.integers(0, 4, size=seq_length)

    for step in reversed(range(n_internal)):
        parent_id = n_taxa + step
        left_child, right_child = merge_order[step]
        sequences[left_child] = jc69_evolve(
            sequences[parent_id], branch_lengths[2 * step], rng,
        )
        sequences[right_child] = jc69_evolve(
            sequences[parent_id], branch_lengths[2 * step + 1], rng,
        )

    return SimulatedTree(
        n_taxa=n_taxa,
        leaf_sequences=sequences[:n_taxa],
        ancestral_sequences=sequences[n_taxa:],
        branch_lengths=branch_lengths,
        merge_order=merge_order,
    )


def generate_dataset(
    num_samples: int,
    n_taxa: int = 4,
    seq_length: int = 200,
    branch_length_range: tuple[float, float] = (0.005, 0.2),
    seed: int = 42,
) -> list[SimulatedTree]:
    """Generate a dataset of simulated trees for supervised training."""
    rng = np.random.default_rng(seed)
    return [
        simulate_tree(n_taxa, seq_length, branch_length_range, rng)
        for _ in range(num_samples)
    ]


def sequences_to_string(int_sequence: np.ndarray) -> str:
    """Convert an integer-encoded sequence back to a nucleotide string."""
    return "".join(NUCLEOTIDES[i] for i in int_sequence)
