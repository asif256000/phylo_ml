"""Agglomerative GNN that constructs phylogenetic trees through learned pairwise merging."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgglomerativePhyloGNN(nn.Module):
    """Builds phylogenetic trees bottom-up by learning which nodes to merge at each step."""

    def __init__(self, hidden_dim: int = 64, temperature: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.leaf_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
        )

        self.message_fn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.merge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.ancestral_decoder = nn.Linear(hidden_dim, 4)

        self.branch_length_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def _build_tree_single(
        self,
        leaf_embeds: torch.Tensor,
        sample_merge_order: torch.Tensor,  # (n_internal, 2) — simulator node IDs
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Construct one tree via agglomerative merging with straight-through estimation."""
        n_taxa = leaf_embeds.size(0)
        n_internal = n_taxa - 1
        pool: list[torch.Tensor] = [leaf_embeds[i] for i in range(n_taxa)]

        # Precompute which leaves each internal node covers (using simulator node IDs)
        desc: dict[int, frozenset] = {i: frozenset({i}) for i in range(n_taxa)}
        for step in range(n_internal):
            left = sample_merge_order[step, 0].item()
            right = sample_merge_order[step, 1].item()
            desc[n_taxa + step] = desc[left] | desc[right]
        true_clades: set[frozenset] = {desc[n_taxa + step] for step in range(n_internal)}

        leaf_sets: list[frozenset] = [frozenset({i}) for i in range(n_taxa)]
        topology_loss = torch.tensor(0.0, device=leaf_embeds.device)

        merge_logits_list: list[torch.Tensor] = []
        ancestral_logits_list: list[torch.Tensor] = []
        branch_lengths_list: list[torch.Tensor] = []

        for _step in range(n_internal):
            k = len(pool)
            pair_indices = [
                (i, j) for i in range(k) for j in range(i + 1, k)
            ]

            scores_list = []
            for i, j in pair_indices:
                site_pairs = torch.cat([pool[i], pool[j]], dim=-1)  # (seq_len, 2*hidden_dim)
                site_scores = self.merge_scorer(site_pairs)          # (seq_len, 1)
                scores_list.append(site_scores.mean())
            scores = torch.stack(scores_list)
            merge_logits_list.append(scores)

            # Direct topology supervision: scores vs true clade membership
            correct_mask = torch.tensor(
                [1.0 if (leaf_sets[i] | leaf_sets[j]) in true_clades else 0.0
                 for (i, j) in pair_indices],
                device=leaf_embeds.device,
            )
            if correct_mask.sum() > 0:
                label_dist = correct_mask / correct_mask.sum()
                topology_loss = topology_loss + F.cross_entropy(
                    scores.unsqueeze(0), label_dist.unsqueeze(0),
                )

            soft = F.softmax(scores / self.temperature, dim=0)
            hard = torch.zeros_like(soft)
            hard[soft.argmax()] = 1.0
            selection = (hard - soft).detach() + soft

            candidate_parents = []
            candidate_branches = []
            for i, j in pair_indices:
                parent_input = torch.cat([pool[i], pool[j]], dim=-1)
                parent = self.message_fn(parent_input)
                candidate_parents.append(parent)

                pm = parent.mean(dim=0)
                bl_l = self.branch_length_head(torch.cat([pm, pool[i].mean(dim=0)]))
                bl_r = self.branch_length_head(torch.cat([pm, pool[j].mean(dim=0)]))
                candidate_branches.append(torch.cat([bl_l, bl_r]))

            parents_stacked = torch.stack(candidate_parents)
            branches_stacked = torch.stack(candidate_branches)

            new_parent = (selection.view(-1, 1, 1) * parents_stacked).sum(dim=0)
            new_branches = (selection.view(-1, 1) * branches_stacked).sum(dim=0)

            ancestral_logits_list.append(self.ancestral_decoder(new_parent))
            branch_lengths_list.append(new_branches)

            selected_idx = hard.argmax().item()
            si, sj = pair_indices[selected_idx]
            new_pool = [pool[idx] for idx in range(k) if idx != si and idx != sj]
            new_pool.append(new_parent)
            pool = new_pool

            merged = leaf_sets[si] | leaf_sets[sj]
            leaf_sets = [leaf_sets[idx] for idx in range(k) if idx != si and idx != sj]
            leaf_sets.append(merged)

        return merge_logits_list, ancestral_logits_list, branch_lengths_list, topology_loss

    def forward(
        self,
        leaf_seqs: torch.Tensor,
        true_merge_order: torch.Tensor,  # (batch, n_internal, 2)
    ) -> tuple[list[list[torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build trees for a batch of samples and return merge logits, ancestral logits, branch lengths, topology loss."""
        batch_size, n_taxa = leaf_seqs.shape[0], leaf_seqs.shape[1]

        all_merge_logits: list[list[torch.Tensor]] = []
        all_ancestral: list[torch.Tensor] = []
        all_branches: list[torch.Tensor] = []
        topology_loss_total = torch.tensor(0.0, device=leaf_seqs.device)

        for b in range(batch_size):
            embeds = self.leaf_encoder(leaf_seqs[b])
            ml, al, bl, tl = self._build_tree_single(embeds, true_merge_order[b])
            all_merge_logits.append(ml)
            all_ancestral.append(torch.stack(al))
            all_branches.append(torch.cat(bl))
            topology_loss_total = topology_loss_total + tl

        ancestral_logits = torch.stack(all_ancestral)
        branch_lengths = torch.stack(all_branches)
        return all_merge_logits, ancestral_logits, branch_lengths, topology_loss_total / batch_size

    def __str__(self) -> str:
        """Summarize model architecture."""
        lines = [
            "AgglomerativePhyloGNN architecture:",
            f"  Hidden dim: {self.hidden_dim}",
            f"  Temperature: {self.temperature}",
            f"  Leaf encoder: 4 \u2192 {self.hidden_dim}",
            f"  Message function: {2 * self.hidden_dim} \u2192 {self.hidden_dim}",
            f"  Merge scorer: {2 * self.hidden_dim} \u2192 1 (symmetric)",
            f"  Ancestral decoder: {self.hidden_dim} \u2192 4",
            f"  Branch length head: {2 * self.hidden_dim} \u2192 1 (softplus)",
            f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}",
        ]
        return "\n".join(lines)
