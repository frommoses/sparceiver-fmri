# Sparceiver-fMRI: Subject-Aware Sparse Perceiver for fMRI Voxel Decoding

A Perceiver-based architecture with **sparse attention** and **subject-aware auxiliary prediction** for task-fMRI voxel-level decoding on the Human Connectome Project (HCP) 1200 dataset.

## Overview

Standard brain decoding models treat all voxels equally and ignore subject-specific variation. Sparceiver-fMRI addresses this with two key design choices:

1. **Sparse Attention** — Top-k thresholding in the Perceiver's cross-attention selectively attends to the most informative voxels, reducing noise from non-task-relevant brain regions.

2. **Subject-Aware Dual Tokens** — Two dedicated latent tokens serve distinct roles:
   - **Classification token** (`latents[-2]`): Predicts the task condition
   - **Subject token** (`latents[-1]`): Predicts subject identity via an auxiliary head, encouraging the model to disentangle subject-specific signals from task-relevant representations

The subject token can be extracted at an intermediate layer to promote early separation of subject identity from task content.

## Architecture

```
fMRI Volume (X, Y, Z)
        │
        ▼
  Fourier Positional Encoding
        │
        ▼
  ┌─────────────────────────────────┐
  │  Cross-Attention (sparse top-k) │◄── Learnable Latent Tokens
  │  + Self-Attention × N           │    (includes CLS + SUBJ tokens)
  │  (repeated D times)             │
  └─────────────────────────────────┘
        │                    │
   CLS Token [-2]      SUBJ Token [-1]
        │                    │
        ▼                    ▼
  Task Condition Head   Subject ID Head
```

### Multi-Label Variant

The `MultiLabelSparceiver` extends the base model with:
- **3-D patch embedding** (ViT-style spatial tokenization)
- **4-D Fourier encoding** (space + time)
- **Per-task classification heads** (independent condition classifiers for each HCP task)

## Dataset

Uses the [Human Connectome Project S1200](https://www.humanconnectome.org/study/hcp-young-adult) task-fMRI data:

| Task | Conditions | Examples |
|------|-----------|----------|
| GAMBLING | 2 | win, loss |
| SOCIAL | 2 | mental, random |
| MOTOR | 5 | left hand, right hand, tongue, left foot, right foot |
| EMOTION | 2 | fear, neutral |
| LANGUAGE | 2 | math, story |
| WM | 8 | 0-back/2-back × body/face/place/tool |
| RELATIONAL | 2 | match, relation |

**Total: 7 tasks, 24 conditions**

### Data Preprocessing

1. Start from z-scored fMRI volumes (`epi_final_zscore.nii.gz`)
2. Extract trial-level volumes using event onset files (EVs)
3. Save as individual `.npy` files per trial

```bash
python -m data.preprocess --data_dir /path/to/HCP/results --save_dir ./data/trials
```

## Installation

```bash
git clone https://github.com/<your-username>/sparceiver-fmri.git
cd sparceiver-fmri
pip install -r requirements.txt
```

## Training

```bash
# Basic training with default config
python train.py --data_dir ./data/sample_data --output_dir ./results

# Custom configuration
python train.py \
    --data_dir ./data/data_3mm \
    --output_dir ./results/exp01 \
    --num_latents 64 \
    --latent_dim 32 \
    --topk 16 \
    --num_epochs 20 \
    --batch_size 2 \
    --lr 0.003
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_latents` | 64 | Number of learnable latent tokens |
| `--latent_dim` | 32 | Latent token dimensionality |
| `--depth` | 1 | Cross-attention iterations |
| `--topk` | 16 | Sparse attention: keep top-k queries per key |
| `--lambda_subj` | 1.0 | Weight for subject prediction loss |
| `--subj_extraction_layer` | 0 | Layer for early subject token extraction |
| `--subj_extraction_block` | 1 | Block for early subject token extraction |
| `--num_folds` | 2 | K-fold cross-validation splits |

## Project Structure

```
sparceiver-fmri/
├── models/
│   ├── modules.py              # Attention, FeedForward, sparse gating
│   ├── sparceiver.py           # SubjectAwareSparceiver (single-label)
│   └── sparceiver_multilabel.py # MultiLabelSparceiver (per-task heads)
├── data/
│   ├── dataset.py              # HCPDataset + collate function
│   └── preprocess.py           # Trial extraction from NIfTI
├── configs/
│   └── default.yaml            # Default hyperparameters
├── utils/
│   └── checkpoint.py           # Save/load checkpoints
├── train.py                    # Main training script
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@misc{sparceiver-fmri,
  title={Subject-Aware Sparse Perceiver for fMRI Voxel Decoding},
  year={2025},
  url={https://github.com/<your-username>/sparceiver-fmri}
}
```

## Acknowledgments

- Architecture based on the [Perceiver](https://arxiv.org/abs/2103.03206) (Jaegle et al., 2021)
- Data from the [Human Connectome Project](https://www.humanconnectome.org/)
