# PhyloGNN — An Intuitive Guide to the Codebase

> **What this document is:** A deep walkthrough of the `AgglomerativePhyloGNN` — what each piece does, why it's shaped the way it is, and exactly where in the code it lives. Aimed at someone who already understands the biological goal (reconstruct ancestral DNA sequences + tree topology) but wants to see the machinery clearly.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Data Pipeline — `simulate.py` and `data.py`](#2-the-data-pipeline)
3. [The Five Learnable Modules — `model.py`](#3-the-five-learnable-modules)
4. [The Build Loop — `_build_tree_single`](#4-the-build-loop)
5. [The Straight-Through Estimator — the Key Trick](#5-the-straight-through-estimator)
6. [The Training Loop — `train.py`](#6-the-training-loop)
7. [How the Model Learns Topology Without Being Told](#7-how-the-model-learns-topology-without-being-told)
8. [Tensor Shape Cheat Sheet](#8-tensor-shape-cheat-sheet)
9. [File Map](#9-file-map)

---

## 1. The Big Picture

### What is phylogenetic inference?

You have DNA sequences from, say, 4 modern species. You want to know:
- **What is the tree structure?** (which species share a more recent common ancestor?)
- **What did the ancestral DNA look like** at each internal node?
- **How long was each branch** (i.e., how much evolution happened)?

Traditional methods (Neighbor-Joining, Maximum Likelihood, MCMC) solve this with explicit substitution models and search over a space of possible trees. This model solves it with a GNN that *builds* the tree bottom-up by learning which sequences are most closely related.

### The core idea in one sentence

> Start with a pool of leaf embeddings. Repeatedly pick the two most "related" nodes, merge them into a parent, predict the ancestor and branch lengths at that node, remove the two children, add the parent to the pool. After N−1 merges, you've built the whole tree.

This is called **agglomerative** merging — the same conceptual family as hierarchical clustering, but the "distance" function is a learned neural network, and the output includes predicted ancestral sequences.

### Why not enumerate all topologies?

For 4 taxa there are 3 possible unrooted topologies — easy to try all three (V1 did this, achieving 99.8% topology accuracy). But the number of topologies grows super-exponentially:

| N taxa | Topologies |
|--------|------------|
| 4 | 3 |
| 10 | 945 |
| 20 | ~2.2 × 10²⁰ |

The agglomerative approach scales as O(N²) per merge step, O(N³) total — completely tractable.

---

## 2. The Data Pipeline

### `simulate.py` — Generating Ground Truth Trees

This file creates synthetic training data using the **Jukes-Cantor 69 (JC69)** DNA substitution model.

**`simulate_tree(n_taxa, seq_length, branch_length_range, rng)`**

The key function. It does three things:

1. **Builds a random tree topology** by repeatedly sampling two nodes from a pool and merging them (bottom-up construction, same logic the GNN will later try to learn).

2. **Assigns branch lengths** by drawing from a uniform distribution over `branch_length_range`.

3. **Evolves sequences** from root to leaves using `jc69_evolve`. Crucially, it stores the sequence at *every node*, not just the leaves. These internal node sequences become the training targets.

```python
# simulate.py, line ~50
# The tree is built as a list of (left_child, right_child) merge events.
# Sequences are evolved from root downward (reversed order):
for step in reversed(range(n_internal)):
    parent_id = n_taxa + step
    left_child, right_child = merge_order[step]
    sequences[left_child] = jc69_evolve(sequences[parent_id], branch_lengths[2 * step], rng)
    sequences[right_child] = jc69_evolve(sequences[parent_id], branch_lengths[2 * step + 1], rng)
```

**`jc69_evolve(parent, branch_length, rng)`**

Under JC69, each site mutates to one of the other 3 nucleotides with equal probability. The probability of *staying the same* is:

```
p_same = 0.25 + 0.75 * exp(-4 * branch_length / 3)
```

As `branch_length → 0`, `p_same → 1.0` (no change). As `branch_length → ∞`, `p_same → 0.25` (random nucleotide). This is implemented directly in lines ~30–38 of `simulate.py`.

**`SimulatedTree` dataclass** (returned by `simulate_tree`):

```
leaf_sequences:      shape (n_taxa, seq_length)          — what the model sees as input
ancestral_sequences: shape (n_taxa - 1, seq_length)      — what the model tries to predict
branch_lengths:      shape (2 * (n_taxa - 1),)           — 2 per internal node (left + right child edges)
```

---

### `data.py` — Converting Trees to PyTorch Geometric Objects

**`tree_to_pyg_data(sample)`**

Converts a `SimulatedTree` into a `torch_geometric.data.Data` object. The main transformation is `_to_onehot`, which converts integer-encoded nucleotides (0–3) into one-hot vectors:

```
A → [1, 0, 0, 0]
C → [0, 1, 0, 0]
G → [0, 0, 1, 0]
T → [0, 0, 0, 1]
```

This is the form the neural network receives. Why one-hot rather than a scalar? Because A=0, C=1, G=2, T=3 would imply an ordering relationship that doesn't exist biologically.

**`create_dataloaders(samples, ...)`**

Splits samples 80/20 into train/val, wraps them in PyG `DataLoader`s. Nothing unusual here.

**Important shape note:** PyG's DataLoader *concatenates* samples along the batch dimension rather than stacking them. This is why `train.py` has an `_unbatch()` function that reshapes everything back to `(batch, ...)` form before passing to the model.

---

## 3. The Five Learnable Modules

All five live in `model.py` inside `AgglomerativePhyloGNN.__init__`. Together they have ~29,638 parameters (with `hidden_dim=64`) and handle *any* number of taxa — the same weights are reused at every merge step.

---

### Module 1: `leaf_encoder`

**File:** `model.py`, lines 18–21  
**Purpose:** Transform raw one-hot nucleotides into learned continuous embeddings.

```python
self.leaf_encoder = nn.Sequential(
    nn.Linear(4, hidden_dim),
    nn.ReLU(),
)
```

**Input:** One-hot nucleotide at a single site — shape `(4,)`  
**Output:** A learned embedding — shape `(hidden_dim,)`  
**Applied to:** Every site of every leaf independently (same weights throughout)

**Intuition:** The one-hot vectors live in a very impoverished space — four corners of a unit cube, all equidistant from each other. The `leaf_encoder` projects them into a richer `hidden_dim`-dimensional space where the network can represent meaningful biological similarity. Think of it as learning "what it means for a site to be an A vs. a G, *in context of downstream computation*."

**In the forward pass (`model.py`, line 120):**
```python
embeds = self.leaf_encoder(leaf_seqs[b])
# leaf_seqs[b] shape: (n_taxa, seq_length, 4)
# embeds shape:       (n_taxa, seq_length, hidden_dim)
```
PyTorch's Linear layer broadcasts over all leading dimensions, so this single call encodes every site of every leaf simultaneously.

---

### Module 2: `merge_scorer`

**File:** `model.py`, lines 30–34  
**Purpose:** Score how "evolutionarily related" two nodes are — this is how the model decides which nodes to merge.

```python
self.merge_scorer = nn.Sequential(
    nn.Linear(2 * hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),
)
```

**Input:** A single vector of shape `(2 * hidden_dim,)`  
**Output:** A single scalar score

**Critical detail — symmetry:** The input to `merge_scorer` is *not* `[emb_i, emb_j]` directly (which would score (A, B) differently than (B, A)). Instead, in `_score_pair` (line 47):

```python
combined = torch.cat([emb_i + emb_j, (emb_i - emb_j).abs()])
```

`emb_i + emb_j` is commutative. `|emb_i - emb_j|` is also symmetric. So the score of (A, B) always equals the score of (B, A) — which is exactly what you want for an unrooted tree.

**Also note:** The input embeddings are *mean-pooled* over sites first (line 67: `pooled = [node.mean(dim=0) for node in pool]`), collapsing the `(seq_length, hidden_dim)` node representation down to a single `(hidden_dim,)` vector before scoring. The scorer sees a sequence-level summary, not individual sites.

---

### Module 3: `message_fn`

**File:** `model.py`, lines 23–28  
**Purpose:** Given two child nodes, compute their parent node's embedding — this is the "message passing" step, and where ancestral reconstruction happens.

```python
self.message_fn = nn.Sequential(
    nn.Linear(2 * hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
)
```

**Input:** Concatenation of two child embeddings, site-wise — shape `(seq_length, 2 * hidden_dim)`  
**Output:** Parent embedding — shape `(seq_length, hidden_dim)`

**Intuition:** Unlike the merger scorer (which works on mean-pooled summaries), `message_fn` works **site by site**. For each position in the alignment, it looks at what child A has at that site and what child B has at that site, and computes what their common ancestor most likely had. This is the GNN's learned version of Felsenstein's pruning algorithm.

The two-layer MLP gives the network enough capacity to capture non-linear interactions between child nucleotide states.

**Used at line 84:**
```python
parent_input = torch.cat([pool[i], pool[j]], dim=-1)  # (seq_length, 2*hidden_dim)
parent = self.message_fn(parent_input)                 # (seq_length, hidden_dim)
```

---

### Module 4: `ancestral_decoder`

**File:** `model.py`, line 36  
**Purpose:** Convert a parent node embedding into nucleotide probability predictions.

```python
self.ancestral_decoder = nn.Linear(hidden_dim, 4)
```

**Input:** Parent embedding at one site — shape `(hidden_dim,)`  
**Output:** Logits over {A, C, G, T} — shape `(4,)`  
**Applied to:** Every site of the parent, so full output is `(seq_length, 4)`

This is the most direct output of the whole model — the predicted ancestral sequence. During training, these logits are compared to the true ancestral sequences via cross-entropy loss (in `train.py`). During inference, you'd take `argmax` (or `softmax` for uncertainty).

It's intentionally a single linear layer. The heavy lifting is already done by `message_fn`; the decoder just maps the learned representation to the 4-nucleotide output space.

**Used at line 98:**
```python
ancestral_logits_list.append(self.ancestral_decoder(new_parent))
# new_parent shape: (seq_length, hidden_dim)
# output shape:     (seq_length, 4)
```

---

### Module 5: `branch_length_head`

**File:** `model.py`, lines 38–43  
**Purpose:** Predict the edge lengths between a parent node and each of its two children.

```python
self.branch_length_head = nn.Sequential(
    nn.Linear(2 * hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),
    nn.Softplus(),
)
```

**Input:** Concatenation of *mean-pooled* parent and *mean-pooled* child embeddings — shape `(2 * hidden_dim,)`  
**Output:** A single positive scalar (the branch length)  
**Called twice per merge:** once for the left child edge, once for the right child edge

**Why Softplus?** Branch lengths must be strictly positive. `Softplus(x) = log(1 + exp(x))` is a smooth approximation to `ReLU` that is always > 0 and differentiable everywhere — ideal for this.

**Used at lines 88–90:**
```python
pm = parent.mean(dim=0)                  # mean-pool parent over sites
bl_l = self.branch_length_head(torch.cat([pm, pooled[i]]))   # left edge
bl_r = self.branch_length_head(torch.cat([pm, pooled[j]]))   # right edge
candidate_branches.append(torch.cat([bl_l, bl_r]))
```

---

## 4. The Build Loop

**File:** `model.py`, `_build_tree_single` (lines 50–107)

This is the heart of the model. It's called once per sample in the batch. Here's what happens at each of the N−1 steps:

```
Step 1: Pool = [A, B, C, D]   (4 leaves)  → Score 6 pairs, merge best → Pool = [A, C, (BD)]
Step 2: Pool = [A, C, (BD)]   (3 nodes)   → Score 3 pairs, merge best → Pool = [A, (C(BD))]
Step 3: Pool = [A, (C(BD))]   (2 nodes)   → Score 1 pair,  merge best → Pool = [root]
```

At each step:

1. **Enumerate all pairs** (`pair_indices`) — all C(k, 2) combinations of the current pool.

2. **Mean-pool each node** over sites → one summary vector per node.

3. **Score all pairs** using `_score_pair` → one scalar per pair.

4. **Apply Straight-Through Estimator** (see Section 5) to select one pair discretely but maintain differentiability.

5. **Compute all candidate parents** — run `message_fn` on *every* possible pair (not just the winner). This is necessary so gradients can flow to all pair scores.

6. **Soft-blend** candidate parents using the STE weights → `new_parent`.

7. **Decode** `new_parent` → ancestral logits and branch lengths.

8. **Update the pool:** remove the two selected children, add `new_parent`.

The outputs (`ancestral_logits_list`, `branch_lengths_list`) accumulate across all steps and are returned to `forward()`.

---

## 5. The Straight-Through Estimator

**File:** `model.py`, lines 75–78

This is the single most important trick in the whole codebase. Three lines:

```python
soft = F.softmax(scores / self.temperature, dim=0)  # (1) differentiable selection weights
hard = torch.zeros_like(soft)                        # (2a) winner-takes-all
hard[soft.argmax()] = 1.0                            # (2b)
selection = (hard - soft).detach() + soft            # (3) THE TRICK
```

### The problem it solves

Tree construction requires a **discrete decision** at each step: pick *one* pair to merge. But `argmax` has zero gradient everywhere — you can't backpropagate through it. If you can't get gradients to the `merge_scorer`, it can never learn.

### How it works

The STE makes a split-brain tensor:

- **Forward pass:** `selection ≈ hard` (the `.detach()` kills `soft`'s contribution to the forward values, leaving only `hard`). The model behaves as if it made a crisp discrete choice.

- **Backward pass:** The gradient of `selection` with respect to `scores` flows through `soft` (since `hard.detach()` contributes zero gradient). The model "pretends" the forward pass was a softmax, and gradients reach all pair scores.

Think of it as: *"When deciding what to do, act decisively. When learning from mistakes, apportion blame to every option you considered."*

### Temperature annealing

The `temperature` parameter controls how "sharp" the softmax is:
- **High temperature (1.0, early training):** soft distribution, spread gradients widely, exploration
- **Low temperature (0.1, late training):** near-hard distribution, more committed decisions, exploitation

Temperature anneals linearly from `temperature_start=1.0` to `temperature_end=0.1` in `train.py` (line ~162):
```python
frac = (epoch - 1) / max(num_epochs - 1, 1)
model.temperature = temperature_start + frac * (temperature_end - temperature_start)
```

---

## 6. The Training Loop

**File:** `train.py`

### Loss function — `compute_loss` (lines 40–55)

Two components:

**1. Ancestral reconstruction loss (cross-entropy):**
```python
anc_loss = F.cross_entropy(
    ancestral_logits.reshape(-1, 4),   # predicted: (batch * n_internal * seq_length, 4)
    true_ancestral.reshape(-1),        # true:      (batch * n_internal * seq_length,)
)
```
This is the *main signal*. The model is penalized for predicting the wrong nucleotide at any ancestral site.

**2. Branch length loss (MSE):**
```python
br_loss = F.mse_loss(branch_lengths, true_branches)
```
The model is penalized for predicting the wrong edge lengths.

**Combined:**
```python
total = beta * anc_loss + gamma * br_loss
```
`beta` and `gamma` are weighting hyperparameters (both default to 1.0). If branch length loss destabilizes training, try `gamma=0.5`.

### What is NOT in the loss

There is **no merge-order supervision**. The model is never told which pairs to merge in which order. This is intentional — merge order is non-unique (different orderings can produce the same tree), so supervising it directly would be like training on coin flips (~50% accuracy baseline).

### `_unbatch` (lines 17–30)

PyG concatenates samples in a batch rather than stacking them. `_unbatch` reshapes everything back:
```python
leaf_seqs = batch_data.leaf_seqs.reshape(bs, n_taxa, -1, 4)
true_anc   = batch_data.true_ancestral.reshape(bs, n_internal, -1)
true_br    = batch_data.true_branches.reshape(bs, 2 * n_internal)
```

### Optimizer and scheduler

- **Adam** with `lr=1e-3`
- **ReduceLROnPlateau:** halves the learning rate when val loss stops improving for 5 epochs
- **Gradient clipping:** `max_norm=5.0` prevents exploding gradients (especially relevant when branch length MSE spikes)

---

## 7. How the Model Learns Topology Without Being Told

This is the conceptually deepest part of the system. Here's the full causal chain:

### What happens when the model merges the *wrong* pair

Suppose the true tree has A and B as sisters, but the model merges A and C first.

1. `message_fn` computes the "parent" of A and C. Since they aren't actually closely related, this parent embedding is a confused mixture.

2. `ancestral_decoder` tries to predict the ancestral sequence of this fake parent. The true ancestral sequence doesn't exist for this (A+C) merge, so the loss compares against whichever internal node is used — the prediction will be bad.

3. **High cross-entropy loss** → large gradient.

4. Gradients flow backward through `ancestral_decoder` → `message_fn` → `_score_pair` → `merge_scorer`.

5. But wait — `argmax` is not differentiable! This is where the **STE** saves us: because `selection = (hard - soft).detach() + soft`, the backward pass treats the merge decision as if it were a softmax over all pairs. So the gradient reaches the *score* for the (A+C) merge and says "lower this score."

6. Over many examples, the `merge_scorer` learns to assign high scores to truly related pairs and low scores to unrelated pairs.

### The key insight

**The model never has to be told the correct topology.** It discovers it because:

> *Merging the correct pair → good ancestral reconstruction → low loss*  
> *Merging the wrong pair → bad ancestral reconstruction → high loss → gradients fix the merge scorer*

Ancestral reconstruction is both the output objective *and* the implicit topology learning signal. This is what makes the design principled: the two tasks are not just jointly trained, they are causally linked.

---

## 8. Tensor Shape Cheat Sheet

For a concrete case: `batch_size=2`, `n_taxa=4`, `seq_length=200`, `hidden_dim=64`.

| Variable | Shape | Where |
|---|---|---|
| `leaf_seqs` (one-hot) | `(2, 4, 200, 4)` | `data.py: _to_onehot` |
| `leaf_embeds` | `(4, 200, 64)` | `model.py:120` after `leaf_encoder` |
| `pool[i]` (one node) | `(200, 64)` | `model.py:55` |
| `pooled[i]` (mean over sites) | `(64,)` | `model.py:67` |
| `scores` at step 1 | `(6,)` | `model.py:69` — 6 pairs for 4 nodes |
| `selection` | `(6,)` | `model.py:78` |
| `parent_input` | `(200, 128)` | `model.py:83` — 64+64 concat |
| `new_parent` | `(200, 64)` | `model.py:95` |
| `ancestral_logits` (all steps) | `(2, 3, 200, 4)` | `model.py:126` — 3 internal nodes |
| `branch_lengths` (all steps) | `(2, 6)` | `model.py:127` — 2 edges × 3 steps |
| `true_ancestral` | `(2, 3, 200)` | `data.py: tree_to_pyg_data` |
| `true_branches` | `(2, 6)` | `data.py: tree_to_pyg_data` |

---

## 9. File Map

```
src/gnn/
│
├── simulate.py       DATA GENERATION
│                     - SimulatedTree dataclass
│                     - jc69_evolve()        ← mutation model
│                     - simulate_tree()      ← one tree with ground truth
│                     - generate_dataset()   ← N trees for training
│
├── data.py           PYTORCH PIPELINE
│                     - _to_onehot()         ← int sequences → one-hot tensors
│                     - tree_to_pyg_data()   ← SimulatedTree → PyG Data object
│                     - create_dataloaders() ← train/val split + DataLoaders
│
├── model.py          THE MODEL
│                     - AgglomerativePhyloGNN
│                       ├── leaf_encoder        (4 → hidden_dim)
│                       ├── merge_scorer        (2*hidden_dim → 1)
│                       ├── message_fn          (2*hidden_dim → hidden_dim)
│                       ├── ancestral_decoder   (hidden_dim → 4)
│                       ├── branch_length_head  (2*hidden_dim → 1)
│                       ├── _score_pair()       ← symmetric scoring util
│                       ├── _build_tree_single()← the N-1 merge loop + STE
│                       └── forward()           ← batch loop over samples
│
├── train.py          TRAINING LOOP
│                     - EpochMetrics dataclass
│                     - _unbatch()           ← undo PyG batch concatenation
│                     - compute_loss()       ← CE (ancestral) + MSE (branches)
│                     - train_epoch()        ← one pass over train loader
│                     - evaluate()           ← one pass over val loader
│                     - run_training()       ← generate data → train → report
│
├── step_through.py   INTERACTIVE DEBUGGER
│                     - step_through()       ← trace one sample, print everything
│
├── __init__.py       PUBLIC API
└── __main__.py       CLI entry point (python -m src.gnn)
```

---

## Putting It All Together: One Training Step

```
1. simulate.py:   Generate tree with random topology, branch lengths, and sequences.
                  Store sequences at ALL nodes (leaves + internals = ground truth).

2. data.py:       Convert to PyG Data: one-hot encode leaves, tensorize targets.

3. model.py forward():
   a. leaf_encoder:       DNA one-hot → learned embeddings  [per site, per leaf]
   b. _build_tree_single() loops N-1 times:
      ├─ merge_scorer:    Score all C(k,2) pairs             [on mean-pooled embeddings]
      ├─ STE:             Discrete merge choice + gradients  [hard argmax + soft backward]
      ├─ message_fn:      Two children → parent embedding    [per site]
      ├─ ancestral_decoder: Parent → nucleotide logits       [per site → 4-class]
      └─ branch_length_head: Parent+child → edge length × 2 [mean-pooled, softplus]

4. train.py compute_loss():
   ├─ Cross-entropy:  predicted ancestral logits vs. true ancestral sequences
   └─ MSE:            predicted branch lengths vs. true branch lengths

5. loss.backward():  Gradients flow back through decoder → message_fn → STE → scorer.
                     The STE lets gradients reach merge_scorer even though we used argmax.

6. optimizer.step(): All five modules update their weights.
```

The elegant result: a model that learns to infer tree topology, reconstruct ancestral DNA, and estimate branch lengths — simultaneously, from a single loss signal, with no topology supervision at all.
