"""Step through the PhyloGNN on a single sample, printing everything that happens."""

import torch
import torch.nn.functional as F
from src.gnn.simulate import generate_dataset, sequences_to_string
from src.gnn.model import AgglomerativePhyloGNN


def step_through():
    """Trace one sample through the entire model with detailed annotations."""

    print("=" * 70)
    print("STEP 0: GENERATE ONE SAMPLE")
    print("=" * 70)

    samples = generate_dataset(1, n_taxa=4, seq_length=20, seed=0)
    sample = samples[0]

    print(f"\nWe simulated a tree with {sample.n_taxa} leaves, each 20bp long.")
    print(f"The simulator evolved sequences from root to leaves and recorded")
    print(f"the true sequence at every internal node.\n")

    for i in range(sample.n_taxa):
        label = chr(65 + i)
        print(f"  Leaf {label}: {sequences_to_string(sample.leaf_sequences[i])}")

    print(f"\nTrue ancestral sequences (what the model tries to predict):")
    for i in range(sample.ancestral_sequences.shape[0]):
        name = "Root" if i == sample.ancestral_sequences.shape[0] - 1 else f"Internal {i}"
        print(f"  {name}: {sequences_to_string(sample.ancestral_sequences[i])}")

    print(f"\nTrue branch lengths: {sample.branch_lengths}")
    print(f"  ({len(sample.branch_lengths)} branches = 2 per internal node)")

    print("\n" + "=" * 70)
    print("STEP 1: ENCODE LEAVES")
    print("=" * 70)

    model = AgglomerativePhyloGNN(hidden_dim=8, temperature=0.5)
    model.eval()

    leaf_onehot = torch.nn.functional.one_hot(
        torch.from_numpy(sample.leaf_sequences).long(), num_classes=4,
    ).float()

    print(f"\nEach nucleotide becomes a one-hot vector:")
    print(f"  A=[1,0,0,0]  C=[0,1,0,0]  G=[0,0,1,0]  T=[0,0,0,1]")
    print(f"\nleaf_onehot shape: {leaf_onehot.shape}")
    print(f"  = ({sample.n_taxa} leaves, {20} sites, 4 channels)")

    nuc_char = "ACGT"[sample.leaf_sequences[0, 0]]
    print(f"\nSite 0 of Leaf A (nucleotide '{nuc_char}'):")
    print(f"  one-hot = {leaf_onehot[0, 0].tolist()}")

    leaf_embeds = model.leaf_encoder(leaf_onehot)
    print(f"\nAfter leaf_encoder (Linear 4->8 + ReLU), applied per-site:")
    print(f"  leaf_embeds shape: {leaf_embeds.shape}")
    print(f"  = ({sample.n_taxa} leaves, 20 sites, 8 hidden dims)")
    print(f"\n  Site 0 of Leaf A embedding (8 numbers):")
    print(f"  {leaf_embeds[0, 0].detach().tolist()}")

    print("\n" + "=" * 70)
    print("STEP 2: SCORE ALL PAIRS (merge step 1 of 3)")
    print("=" * 70)

    pool = [leaf_embeds[i] for i in range(sample.n_taxa)]
    labels = ["A", "B", "C", "D"]

    k = len(pool)
    pair_indices = [(i, j) for i in range(k) for j in range(i + 1, k)]

    print(f"\nPool has {k} nodes -> C({k},2) = {len(pair_indices)} possible pairs:")

    pooled = [node.mean(dim=0) for node in pool]

    print(f"\nFirst, mean-pool each node over its 20 sites -> one vector of 8 dims.")
    print(f"  pooled[A] shape: {pooled[0].shape}")

    scores = []
    for i, j in pair_indices:
        combined = torch.cat([pooled[i] + pooled[j], (pooled[i] - pooled[j]).abs()])
        score = model.merge_scorer(combined).squeeze(-1)
        scores.append(score)
        pair_label = f"({labels[i]},{labels[j]})"
        print(f"  merge_scorer{pair_label} = {score.item():.4f}")

    scores_tensor = torch.stack(scores)
    soft = F.softmax(scores_tensor / model.temperature, dim=0)
    winner = soft.argmax().item()
    wi, wj = pair_indices[winner]

    print(f"\nSoftmax(scores / tau={model.temperature}) -> selection weights:")
    for idx, (i, j) in enumerate(pair_indices):
        marker = " <-- SELECTED" if idx == winner else ""
        print(f"  ({labels[i]},{labels[j]}): {soft[idx].item():.4f}{marker}")

    print(f"\nThe model picks ({labels[wi]},{labels[wj]}) to merge first.")

    print("\n" + "=" * 70)
    print("STEP 3: MERGE THE SELECTED PAIR")
    print("=" * 70)

    parent_input = torch.cat([pool[wi], pool[wj]], dim=-1)
    print(f"\nConcatenate {labels[wi]} and {labels[wj]} embeddings site-wise:")
    print(f"  concat shape: {parent_input.shape}")
    print(f"  = (20 sites, 16 dims)  [8 from {labels[wi]} + 8 from {labels[wj]}]")

    new_parent = model.message_fn(parent_input)
    print(f"\nmessage_fn (MLP: 16->8->8 with ReLU) produces the parent embedding:")
    print(f"  parent shape: {new_parent.shape}")
    print(f"  = (20 sites, 8 dims)")

    anc_logits = model.ancestral_decoder(new_parent)
    anc_probs = F.softmax(anc_logits, dim=-1)
    anc_pred = anc_logits.argmax(dim=-1)

    print(f"\nancestral_decoder (Linear 8->4) predicts nucleotide at each site:")
    print(f"  Output shape: {anc_logits.shape} = (20 sites, 4 nucleotide logits)")
    print(f"\n  First 5 sites predicted ancestor of ({labels[wi]},{labels[wj]}):")
    print(f"  {'Site':<6} {'P(A)':<8} {'P(C)':<8} {'P(G)':<8} {'P(T)':<8} {'Pred':<6}")
    for site in range(5):
        p = anc_probs[site].detach().tolist()
        pred_nuc = "ACGT"[anc_pred[site].item()]
        print(f"  {site:<6} {p[0]:<8.3f} {p[1]:<8.3f} {p[2]:<8.3f} {p[3]:<8.3f} {pred_nuc:<6}")

    pm = new_parent.mean(dim=0).detach()
    cm = pooled[wi].detach()
    bl_left = model.branch_length_head(torch.cat([pm, cm]))
    print(f"\nbranch_length_head predicts edge lengths:")
    print(f"  concat(pool(parent), pool({labels[wi]})) -> MLP + softplus -> {bl_left.item():.6f}")

    print("\n" + "=" * 70)
    print("STEP 4: UPDATE THE POOL")
    print("=" * 70)

    merged_label = f"({labels[wi]}{labels[wj]})"
    remaining = [labels[idx] for idx in range(k) if idx != wi and idx != wj]
    remaining.append(merged_label)
    print(f"\nRemove {labels[wi]} and {labels[wj]}, add their parent {merged_label}.")
    print(f"Pool was: {labels}")
    print(f"Pool now: {remaining}")
    print(f"Pool size: {k} -> {k - 1}")
    print(f"\nRepeat: score {len(remaining)}C2 = {len(remaining) * (len(remaining)-1) // 2} pairs, merge best, shrink pool.")
    print(f"After {sample.n_taxa - 1} total merges, pool has 1 node = the root.")

    print("\n" + "=" * 70)
    print("STEP 5: HOW LEARNING WORKS")
    print("=" * 70)

    print(f"""
The loss has two parts:

1. ANCESTRAL LOSS (cross-entropy):
   At each merge step, the model predicted an ancestral sequence.
   Compare to the true simulated ancestor -> high loss if wrong.

2. BRANCH LENGTH LOSS (MSE):
   At each merge step, the model predicted 2 branch lengths.
   Compare to the true simulated lengths -> high loss if wrong.

There is NO merge-order supervision. The model is never told which
pair to merge. Instead, if it merges the wrong pair:
  -> the "ancestor" of two unrelated sequences will look like noise
  -> the ancestral loss will be high
  -> gradients flow back through the straight-through estimator
  -> the merge_scorer learns to assign higher scores to truly related pairs

The straight-through estimator (STE) is the key trick:
  FORWARD:  hard argmax (pick one pair, discrete decision)
  BACKWARD: pretend it was softmax (gradients flow to all pair scores)

This means the model gets clean discrete trees during inference,
but can still learn via gradient descent during training.
""")

    print("=" * 70)
    print("SUMMARY: THE FULL PIPELINE")
    print("=" * 70)
    print(f"""
  leaf_encoder:      DNA one-hot (4) -> hidden embedding (8)
                     Same weights for every leaf, every site.

  merge_scorer:      For each pair in pool, score how "related" they are.
                     Input: sum and abs-diff of mean-pooled embeddings.
                     Output: scalar score. Symmetric by construction.

  message_fn:        Merge two children into a parent, site by site.
                     Input: concat of child embeddings (16 dims).
                     Output: parent embedding (8 dims).

  ancestral_decoder: From parent embedding, predict ancestor nucleotide.
                     Input: parent embedding at each site (8 dims).
                     Output: probability over A,C,G,T (4 dims).

  branch_length_head: From parent-child pair, predict edge length.
                     Input: concat of mean-pooled parent and child (16 dims).
                     Output: positive scalar via softplus.

  The model builds the tree bottom-up through N-1 merge steps.
  No topology is ever enumerated. The tree emerges from the merge decisions.
  Learning signal comes entirely from reconstruction quality.
""")


if __name__ == "__main__":
    step_through()