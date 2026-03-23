"""Subject-Aware Sparceiver for fMRI voxel-level decoding.

Single-label variant: one classification head + one subject-prediction head.
Input is a 3-D fMRI volume (X, Y, Z) with Fourier positional encoding.
"""

import torch
from torch import nn

from einops import rearrange, repeat

from .modules import (
    Attention, PreNorm, FeedForward, Decoder,
    cache_fn, fourier_encode,
)


class SubjectAwareSparceiver(nn.Module):
    """Perceiver with sparse attention and subject-aware auxiliary head.

    Latent tokens include two special tokens appended at the end:
        latents[-2]  ->  classification token
        latents[-1]  ->  subject prediction token

    The subject token can optionally be extracted at an intermediate layer
    (``subj_extraction_layer``, ``subj_extraction_block``) to encourage
    early disentanglement of subject identity from task content.

    Args:
        num_freq_bands: Number of Fourier frequency bands for positional encoding.
        depth: Number of cross-attention layers (iterative refinement).
        max_freq: Maximum frequency for Fourier encoding.
        input_channels: Number of input channels per voxel (default 1 for z-scored).
        input_axis: Number of spatial axes (3 for volumetric fMRI).
        num_latents: Number of learnable latent tokens.
        latent_dim: Dimensionality of latent tokens.
        cross_heads: Number of heads in cross-attention.
        latent_heads: Number of heads in latent self-attention.
        cross_dim_head: Dimension per head in cross-attention.
        latent_dim_head: Dimension per head in latent self-attention.
        num_classes: Number of task conditions to classify.
        num_subjects: Number of subjects for subject prediction head.
        attn_dropout: Dropout rate for attention weights.
        ff_dropout: Dropout rate for feed-forward layers.
        weight_tie_layers: Share weights across depth iterations (after first).
        fourier_encode_data: Whether to apply Fourier positional encoding.
        self_per_cross_attn: Number of self-attention blocks per cross-attention.
        subj_extraction_layer: Cross-attention layer index for early subject extraction.
        subj_extraction_block: Self-attention block index for early subject extraction.
        topk: If set, keep only top-k queries per key (sparse attention).
        maxprop: If set, keep queries above max_val * maxprop per key.
    """

    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels=1,
        input_axis=3,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1000,
        num_subjects=30,
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False,
        fourier_encode_data=True,
        self_per_cross_attn=2,
        subj_extraction_layer=None,
        subj_extraction_block=None,
        topk=None,
        maxprop=None,
    ):
        super().__init__()

        self.subj_extraction_layer = subj_extraction_layer
        self.subj_extraction_block = subj_extraction_block

        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data

        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.maxprop = maxprop
        self.topk = topk

        # Classification head (from cls token)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, num_classes, bias=False),
        )

        # Subject prediction head (from subj token)
        self.subj_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim * 2, num_subjects, bias=False),
        )

        # Voxel-space decoder
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=2599)

        # Build transformer layers
        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout),
            context_dim=input_dim,
        )
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout),
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

    def forward(self, data, v_mask=None, subject_ids=None, mask=None, return_embeddings=False):
        """
        Args:
            data: (B, X, Y, Z, C) fMRI volume.
            v_mask: (B, X, Y, Z, 1) optional voxel mask (brain region).
            subject_ids: (B,) subject indices (unused in forward, for API compat).
            mask: Optional key mask for cross-attention.
            return_embeddings: If True, include full latent embeddings in output.

        Returns:
            dict with keys: logits_cls, logits_subj, and optionally embeddings.
        """
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, \
            f'Expected {self.input_axis} spatial axes, got {len(axis)}'

        # Fourier positional encoding
        if self.fourier_encode_data:
            axis_pos = list(map(
                lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype),
                axis,
            ))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)
            data = torch.cat((data, enc_pos), dim=-1)

        # Apply voxel mask (select brain voxels only)
        if v_mask is not None:
            v_mask = v_mask.to(device=device, dtype=torch.bool)
            data_flattened = rearrange(data, 'b w h d dim -> b (w h d) dim')
            v_mask_flattened = rearrange(v_mask, 'b w h d 1 -> b (w h d)')
            data = data_flattened[v_mask_flattened]
            data = data.view(b, -1, data.shape[-1])
        else:
            data = rearrange(data, 'b w h d dim -> b (w h d) dim')

        # Initialize latent tokens
        x = repeat(self.latents, 'n d -> b n d', b=b)

        early_subj_token = None

        # Iterative cross-attention + self-attention
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

                # Early subject token extraction
                if (self.subj_extraction_layer is not None
                        and self.subj_extraction_block is not None
                        and i == self.subj_extraction_layer
                        and j == self.subj_extraction_block):
                    early_subj_token = x[:, -1].clone()

        # Extract special tokens
        cls_token = x[:, -2]
        subj_token = early_subj_token if early_subj_token is not None else x[:, -1]

        outputs = {
            "logits_cls": self.cls_head(cls_token),
            "logits_subj": self.subj_head(subj_token),
        }

        if return_embeddings:
            outputs["embeddings"] = x

        return outputs
