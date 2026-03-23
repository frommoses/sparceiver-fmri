"""Perceiver IO variant of Subject-Aware Sparceiver for fMRI decoding.

Key difference from MultiLabelSparceiver:
  - Instead of routing cls_token through per-task MLP heads, each task has a
    learned **query token** that cross-attends into the shared latent array via
    a shared OutputCrossAttention module.
  - "N students (tasks) with different questions ask the same librarian
    (shared output cross-attention) from the same library (shared latent)."
  - Lightweight per-task Linear heads follow the output cross-attention.

This design encourages the backbone to learn a unified representation while
task-specific queries selectively extract what each task needs.
"""

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .modules import (
    Attention, PreNorm, FeedForward,
    cache_fn, fourier_encode,
)


class OutputCrossAttention(nn.Module):
    """Perceiver IO-style output cross-attention.

    A shared module where task-specific query tokens attend to the
    shared latent array to extract task-relevant information.

    Args:
        latent_dim: Dimensionality of latent and query tokens.
        cross_heads: Number of attention heads.
        cross_dim_head: Dimension per attention head.
        dropout: Dropout rate.
    """

    def __init__(self, latent_dim, cross_heads=1, cross_dim_head=64, dropout=0.):
        super().__init__()
        self.cross_attn = PreNorm(
            latent_dim,
            Attention(latent_dim, latent_dim, heads=cross_heads,
                      dim_head=cross_dim_head, dropout=dropout),
            context_dim=latent_dim,
        )
        self.ff = PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout))

    def forward(self, query, latent):
        """
        Args:
            query: (B, 1, D) task-specific query token.
            latent: (B, N, D) shared backbone output.

        Returns:
            (B, 1, D) task-specific output.
        """
        out, _ = self.cross_attn(query, context=latent)
        out = out + query
        out = self.ff(out) + out
        return out


class SparceiverIO(nn.Module):
    """Perceiver IO-style Sparceiver with task-specific output queries.

    Architecture:
        1. Input: (B, T, X, Y, Z, C) -> 3D patching -> Fourier encoding
        2. Shared Perceiver backbone (cross-attn + self-attn)
        3. Per-task learned query tokens -> shared OutputCrossAttention -> latent
        4. Per-task lightweight Linear heads for condition classification
        5. Shared subject prediction head from latent[-1]

    Args:
        num_freq_bands: Fourier frequency bands.
        depth: Cross-attention iterations.
        max_freq: Maximum Fourier frequency.
        input_channels: Input channels per voxel.
        input_axis: Positional axes (4: x, y, z, t).
        num_latents: Learnable latent tokens.
        latent_dim: Latent dimensionality.
        cross_heads: Cross-attention heads.
        latent_heads: Latent self-attention heads.
        cross_dim_head: Dimension per cross-attention head.
        latent_dim_head: Dimension per latent self-attention head.
        num_tasks: Number of HCP tasks.
        num_conditions: List of condition counts per task.
        num_subjects: Total subjects.
        attn_dropout: Attention dropout.
        ff_dropout: Feed-forward dropout.
        weight_tie_layers: Share weights across iterations.
        self_per_cross_attn: Self-attention blocks per cross-attention.
        subj_extraction_layer: Layer for early subject token extraction.
        subj_extraction_block: Block for early subject token extraction.
        topk: Sparse attention top-k.
        maxprop: Sparse attention max-proportion.
        patch_size: 3D spatial patch size.
    """

    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels=1,
        input_axis=4,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_tasks=7,
        num_conditions=None,
        num_subjects=1000,
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False,
        self_per_cross_attn=2,
        subj_extraction_layer=None,
        subj_extraction_block=None,
        topk=None,
        maxprop=None,
        patch_size=4,
    ):
        super().__init__()

        self.subj_extraction_layer = subj_extraction_layer
        self.subj_extraction_block = subj_extraction_block

        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.num_tasks = num_tasks
        self.patch_size = patch_size

        # Patch + Fourier dimensions
        self.patch_dim = (patch_size ** 3) * input_channels
        self.fourier_dim = input_axis * ((num_freq_bands * 2) + 1)
        input_dim = self.patch_dim + self.fourier_dim

        self.patch_to_embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.maxprop = maxprop
        self.topk = topk

        # Per-task learned query tokens
        if num_conditions is None:
            num_conditions = [10] * num_tasks
        self.num_conditions = num_conditions

        self.task_queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, latent_dim))
            for _ in range(num_tasks)
        ])

        # Shared output cross-attention
        self.output_cross_attn = OutputCrossAttention(
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            cross_dim_head=cross_dim_head,
            dropout=attn_dropout,
        )

        # Lightweight per-task condition heads
        self.condition_heads = nn.ModuleList([
            nn.Linear(latent_dim, num_conds, bias=False)
            for num_conds in num_conditions
        ])

        # Shared subject head
        self.subj_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, num_subjects, bias=False),
        )

        # Backbone transformer layers
        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, latent_dim, heads=cross_heads,
                      dim_head=cross_dim_head, dropout=attn_dropout),
            context_dim=latent_dim,
        )
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, heads=latent_heads,
                      dim_head=latent_dim_head, dropout=attn_dropout),
        )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])
            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=block_ind),
                    get_latent_ff(**cache_args, key=block_ind),
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns,
            ]))

    def forward(self, data, task_labels, v_mask=None, subject_ids=None,
                mask=None, return_embeddings=False):
        """
        Args:
            data: (B, T, X, Y, Z, C) spatio-temporal fMRI.
            task_labels: (B,) task index per sample.
            mask: (B, T) temporal validity mask.
            return_embeddings: Return full latent embeddings.

        Returns:
            dict with logits_cls (list), logits_subj (tensor).
        """
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        t, x, y, z = axis
        p = self.patch_size

        # -- Pad + Patch + Fourier --
        pad_x = (p - x % p) % p
        pad_y = (p - y % p) % p
        pad_z = (p - z % p) % p

        data = data.permute(0, 1, 5, 4, 3, 2)
        data = F.pad(data, (0, pad_x, 0, pad_y, 0, pad_z))
        data = data.permute(0, 1, 5, 4, 3, 2)

        data = rearrange(
            data, 'b t (h p1) (w p2) (d p3) c -> b t (h w d) (p1 p2 p3 c)',
            p1=p, p2=p, p3=p,
        )
        num_spatial = data.shape[2]

        n_x = (x + pad_x) // p
        n_y = (y + pad_y) // p
        n_z = (z + pad_z) // p

        axis_x = torch.linspace(-1., 1., steps=n_x, device=device, dtype=dtype)
        axis_y = torch.linspace(-1., 1., steps=n_y, device=device, dtype=dtype)
        axis_z = torch.linspace(-1., 1., steps=n_z, device=device, dtype=dtype)
        axis_t = torch.linspace(-1., 1., steps=t, device=device, dtype=dtype)

        grid_t, grid_x, grid_y, grid_z = torch.meshgrid(
            axis_t, axis_x, axis_y, axis_z, indexing='ij',
        )
        for g in [grid_x, grid_y, grid_z, grid_t]:
            g = rearrange(g, 't h w d -> t (h w d)')

        grid_x = rearrange(grid_x, 't h w d -> t (h w d)')
        grid_y = rearrange(grid_y, 't h w d -> t (h w d)')
        grid_z = rearrange(grid_z, 't h w d -> t (h w d)')
        grid_t = rearrange(grid_t, 't h w d -> t (h w d)')

        pos = torch.stack([grid_x, grid_y, grid_z, grid_t], dim=-1)
        pos = repeat(pos, 't s c -> b t s c', b=b)

        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        data = torch.cat((data, enc_pos), dim=-1)
        data = self.patch_to_embedding(data)

        if mask is not None:
            mask = mask.to(device)
            mask = repeat(mask, 'b t -> b t s', s=num_spatial)
            mask = rearrange(mask, 'b t s -> b (t s)')

        data = rearrange(data, 'b t s d -> b (t s) d')

        # -- Perceiver Backbone --
        x = repeat(self.latents, 'n d -> b n d', b=b)
        early_subj_token = None

        for i, (cross_attn, cross_ff, self_attns) in enumerate(self.layers):
            cross_out, attn_gate = cross_attn(
                x, context=data, mask=mask, topk=self.topk, maxprop=self.maxprop,
            )
            x = cross_out + x
            x = cross_ff(x) + x

            for j, (self_attn, self_ff) in enumerate(self_attns):
                self_out, attn_gate = self_attn(
                    x, topk=self.topk, maxprop=self.maxprop, attn_gate=attn_gate,
                )
                x = self_out + x
                x = self_ff(x) + x

                if (self.subj_extraction_layer is not None
                        and self.subj_extraction_block is not None
                        and i == self.subj_extraction_layer
                        and j == self.subj_extraction_block):
                    early_subj_token = x[:, -1].clone()

        # -- Perceiver IO: Output Cross-Attention --
        logits_cls_list = [None] * b
        task_to_indices = {}
        for i in range(b):
            task_id = task_labels[i].item()
            if task_id not in task_to_indices:
                task_to_indices[task_id] = []
            task_to_indices[task_id].append(i)

        for task_id, indices in task_to_indices.items():
            indices_tensor = torch.tensor(indices, device=device)
            q = self.task_queries[task_id]
            q_batch = repeat(q, '1 d -> k 1 d', k=len(indices))
            latent_batch = x[indices_tensor]

            out = self.output_cross_attn(q_batch, latent_batch)
            out = out.squeeze(1)
            batch_logits = self.condition_heads[task_id](out)

            for out_i, orig_i in enumerate(indices):
                logits_cls_list[orig_i] = batch_logits[out_i]

        # -- Subject Prediction --
        subj_token = early_subj_token if early_subj_token is not None else x[:, -1]
        logits_subj = self.subj_head(subj_token)

        # DDP compatibility: ensure unused task params are in compute graph
        used_tasks = set(task_to_indices.keys())
        missing_tasks = [t for t in range(self.num_tasks) if t not in used_tasks]
        if missing_tasks:
            aux = sum(
                self.task_queries[t].sum() + self.condition_heads[t].weight.sum()
                for t in missing_tasks
            )
            logits_subj = logits_subj + aux * 0

        outputs = {
            "logits_cls": logits_cls_list,
            "logits_subj": logits_subj,
        }
        if return_embeddings:
            outputs["embeddings"] = x

        return outputs
