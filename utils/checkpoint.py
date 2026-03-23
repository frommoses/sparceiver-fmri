"""Checkpoint save/load utilities."""

import os
import torch


def save_checkpoint(directory, state, filename):
    """Save model checkpoint to disk."""
    os.makedirs(directory, exist_ok=True)
    torch.save(state, os.path.join(directory, filename))


def load_checkpoint(directory, filename):
    """Load model checkpoint from disk. Returns None if not found."""
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        print(f"=> Loading checkpoint: {path}")
        return torch.load(path, weights_only=False)
    return None
