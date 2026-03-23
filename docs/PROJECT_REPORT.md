# Project Report: Sparse Perceiver for fMRI Voxel Decoding

## 1. Project Overview

This project develops a Perceiver-based deep learning architecture for **task-fMRI brain decoding** on the Human Connectome Project (HCP) 1200 dataset. The model classifies cognitive task conditions from whole-brain fMRI volumes at the voxel level, while simultaneously predicting subject identity to disentangle individual brain variability from task-relevant neural representations.

### Key Contributions

1. **Sparse Perceiver (Sparceiver)**: Top-k attention thresholding adapted for fMRI, selectively attending to the most informative voxels/patches
2. **Subject-Aware Dual-Token Design**: Dedicated classification and subject-prediction tokens within the latent array, with configurable early extraction for subject disentanglement
3. **Perceiver IO Extension**: Task-specific learned query tokens with shared output cross-attention for multi-task decoding
4. **Scalable Pipeline**: 3D patch embedding, 4D Fourier positional encoding, DDP multi-GPU training

---

## 2. Architecture Evolution

### Phase 1: Base Sparceiver (2025.06)

Initial adaptation of the Perceiver architecture for fMRI data with subject-aware auxiliary loss.

- **Input**: Raw fMRI volumes (X, Y, Z) with Fourier positional encoding
- **Architecture**: Cross-attention (voxels → latents) + self-attention
- **Sparse attention**: Top-k thresholding per key — keeps only the k most relevant queries, setting the rest to -inf before softmax
- **Dual tokens**: `latents[-2]` for task classification, `latents[-1]` for subject prediction
- **Training**: Single GPU, single-label classification, K-fold CV

### Phase 2: ViT-Style Patch Embedding (2026.01 - 2026.02)

Addressed the computational bottleneck of voxel-level tokenization (~271K tokens per volume).

- **3D Patch Embedding**: 4×4×4 spatial patches reduce token count from 271K to ~4,864
- **4D Fourier Encoding**: Extended positional encoding to (x, y, z, t) for spatio-temporal volumes
- **Attention Gate Propagation**: Keys zeroed out by sparse attention in one layer remain dead in subsequent layers

### Phase 3: Multi-Label Task-Specific Heads (2026.02)

Extended the single-label classifier to handle all 7 HCP tasks simultaneously.

- **Per-task classifiers**: Independent MLP heads for each task (different condition counts)
- **Shared backbone**: Single Perceiver processes all tasks
- **Joint loss**: `L = L_cls + λ · L_subj` with task-specific cross-entropy

### Phase 4: Multi-GPU DDP Training (2026.02 - 2026.03)

Scaled training to multiple GPUs for the full HCP dataset.

- **DDP over FSDP**: Weight-tying in Perceiver layers breaks FSDP's parameter sharding; DDP is compatible
- **Unused parameter handling**: Zero-multiply trick for task parameters absent in a batch
- **Data infrastructure**: Explored HDF5 (hierarchical, LZ4-compressed), reverted to folder-based .npy for NFS performance
- **Resolution**: Experiments at both 4mm and 3mm spatial resolution

### Phase 5: Perceiver IO Output Attention (2026.03)

Latest architecture — replaces direct MLP heads with learned output queries.

- **Per-task query tokens**: Each task has a learned query token (1, D)
- **Shared output cross-attention**: All task queries attend to the same latent array through a shared attention module
- **Lightweight condition heads**: Simple Linear layers after output cross-attention
- **Design principle**: "N students (tasks) with different questions ask the same librarian (shared cross-attention) from the same library (shared latent)"

```
fMRI Volume (B, T, X, Y, Z)
        │
    3D Patching (4³)
        │
    4D Fourier Encoding
        │
    Patch Embedding → (B, T×S, D)
        │
  ┌─────────────────────────────┐
  │  Shared Perceiver Backbone  │
  │  Cross-Attn (sparse top-k)  │
  │  + Self-Attn × N            │
  │  (D iterations, weight-tied)│
  └─────────────────────────────┘
        │
   Shared Latent (B, L, D)
        │
  ┌─────┴──────────────────────┐
  │                            │
  │  Per-Task Query Tokens     │  Subject Token
  │  → Shared Output CrossAttn │  → Subject Head
  │  → Per-Task Linear Head    │
  │                            │
  └────────────────────────────┘
```

---

## 3. Dataset

**Human Connectome Project S1200** — task-fMRI from ~1,200 healthy adults.

| Task | Conditions | Description |
|------|-----------|-------------|
| GAMBLING | win, loss | Reward processing |
| SOCIAL | mental, random | Theory of mind |
| MOTOR | rh, lh, rf, lf, tongue | Motor execution |
| EMOTION | fear, neutral | Affect processing |
| LANGUAGE | story, math | Language comprehension |
| WM | 8 conditions (0bk/2bk × body/face/place/tool) | Working memory |
| RELATIONAL | match, relation | Relational reasoning |

**Preprocessing pipeline:**
1. Start from z-scored fMRI (`epi_final_zscore.nii.gz`)
2. Extract trial-level volumes using EV onset files (variable temporal length)
3. Save as individual `.npy` files per trial
4. Custom collate function handles variable-length padding with temporal masks

**Spatial resolution**: 3mm and 4mm variants tested.

---

## 4. Experimental Results

### 4.1 Overall Accuracy

| Configuration | Val Acc (Max) | Val Acc (Final) | Subject Acc | Resolution | Notes |
|--------------|---------------|-----------------|-------------|------------|-------|
| Base Sparceiver (small) | 49% | 47% | 98% | 4mm | Phase 1 baseline |
| Patch + Sparse (medium) | 45% | 39% | 99.6% | 4mm | HDF5 era |
| Multi-label + 3mm | ~45% | ~40% | 98%+ | 3mm | Better spatial info |
| **Perceiver IO (λ=0.1)** | **54.0%** | **54.0%** | **96.7%** | 3mm | **Best overall** |

### 4.2 Per-Task Performance (Perceiver IO, Best Configuration)

| Task | Conditions | Accuracy | Chance Level | Notes |
|------|-----------|----------|--------------|-------|
| LANGUAGE | 2 | **99.7%** | 50% | Trivially separable |
| EMOTION | 2 | **69.6%** | 50% | Strong signal |
| SOCIAL | 2 | **58.6%** | 50% | Above chance |
| RELATIONAL | 2 | **56.4%** | 50% | Above chance |
| GAMBLING | 2 | **50.7%** | 50% | Near chance |
| MOTOR | 5 | **20.7%** | 20% | At chance (hardest) |
| WM | 8 | **12.2%** | 12.5% | Near chance (most conditions) |

### 4.3 Key Observations

- **Subject classification consistently >96%**: The model easily captures individual brain signatures, confirming strong subject-specific patterns in fMRI
- **Task difficulty hierarchy**: LANGUAGE >> EMOTION > SOCIAL ≈ RELATIONAL > GAMBLING > MOTOR > WM
- **λ_subj tuning matters**: λ=0.1 (reduced subject loss weight) outperformed λ=1.0, suggesting that overweighting subject prediction can compete with task learning
- **Perceiver IO > flat MLP heads**: Shared output cross-attention with task-specific queries improved overall accuracy from ~45% to 54%

---

## 5. Key Design Decisions

### 5.1 Why Perceiver over CNN / ViT?

| Architecture | Issue for fMRI |
|-------------|----------------|
| CNN | Local receptive fields miss distributed brain patterns |
| ViT (standard) | 271K voxel tokens → O(n²) attention is infeasible |
| **Perceiver** | Latent bottleneck (L << N) makes whole-brain attention tractable |

### 5.2 Why Sparse Attention?

fMRI has strong spatial localization — task-relevant information is concentrated in specific brain regions, not distributed across all voxels. Top-k attention provides an inductive bias that matches this neuroscience prior, while reducing effective complexity from O(n²) to O(n×k).

### 5.3 Why Subject-Aware Auxiliary Loss?

Without explicit subject supervision, subject-specific patterns leak into task representations, hurting generalization. The auxiliary subject head forces the model to explicitly capture "what's unique to this brain" in a dedicated token, freeing the classification token to focus on generalizable task signals.

### 5.4 Why DDP over FSDP?

The Perceiver uses weight-tying across depth iterations (shared parameters). FSDP shards parameters across GPUs, which breaks gradient flow through weight-tied layers. DDP replicates the full model and only all-reduces gradients — compatible with any model topology.

---

## 6. Model Variants

| Model | File | Input | Output | Key Feature |
|-------|------|-------|--------|-------------|
| `SubjectAwareSparceiver` | `sparceiver.py` | (B, X, Y, Z, C) | Single-label logits | Base model, per-voxel Fourier |
| `MultiLabelSparceiver` | `sparceiver_multilabel.py` | (B, T, X, Y, Z, C) | Per-task logits | 3D patches, per-task MLP heads |
| `SparceiverIO` | `sparceiver_io.py` | (B, T, X, Y, Z, C) | Per-task logits | Output cross-attention (best) |

---

## 7. Infrastructure

### Training Pipeline
- **Cross-validation**: Stratified K-fold on condition labels
- **Optimizer**: AdamW with OneCycleLR (cosine annealing + warmup)
- **Checkpointing**: Per-fold best model saved; full state (model + optimizer + scheduler + history)
- **Metrics**: Per-task and aggregate accuracy, classification and subject loss tracked separately

### Data Pipeline
- **Format**: Per-trial .npy files organized as `{subject}/{task}/{condition}/{trial}.npy`
- **Variable-length handling**: Custom collate with zero-padding + boolean temporal mask
- **Scaling**: Tested with HDF5 (LZ4-compressed, hierarchical) for large-scale experiments

---

## 8. Future Directions

- [ ] Hierarchical loss: Task-level supervision in addition to condition-level
- [ ] Contrastive learning: Pull same-task samples closer in latent space
- [ ] Spatial augmentation: Brain-aware warping (preserve anatomy)
- [ ] Cross-subject generalization: Train on N-1 subjects, test on held-out subject
- [ ] Attention map analysis: Visualize which brain regions the sparse attention selects per task
