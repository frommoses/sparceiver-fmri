"""Shared modules for Sparceiver architectures."""

from math import pi
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    """Cache layer constructors for weight-tying across depth iterations."""
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn


def fourier_encode(x, max_freq, num_bands=4):
    """Fourier positional encoding for continuous coordinates."""
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    """Pre-LayerNorm wrapper with optional context normalization."""

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    """Gated GELU activation (Shazeer 2020)."""

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2, bias=False),
            GEGLU(),
            nn.Linear(dim * mult, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention with optional sparse top-k / max-proportion gating.

    Sparse attention keeps only the top-k (or top maxprop-fraction) queries
    per key, setting the rest to -inf before softmax.  An ``attn_gate``
    tensor from the previous layer can further restrict which keys are live.
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, context=None, mask=None, topk=None, maxprop=None, attn_gate=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Apply attention gate from previous layer (zero out dead keys)
        if attn_gate is not None:
            attn_gate_expanded = attn_gate.unsqueeze(-1).expand_as(k)
            k = k * attn_gate_expanded

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        epsilon = 1e-7

        # Sparse attention: keep top-k queries per key
        if topk:
            threshold_vals, _ = torch.topk(sim, topk, dim=-2)

        # Sparse attention: keep queries above max_val * maxprop per key
        if maxprop:
            max_vals, _ = torch.max(sim, dim=-2)
            threshold_vals = max_vals.unsqueeze(1) * maxprop

        if topk or maxprop:
            min_vals = threshold_vals[:, -1, :].unsqueeze(1).expand_as(sim)
            min_vals = torch.where(
                min_vals <= 0,
                torch.tensor(epsilon, device=sim.device),
                min_vals,
            )
            sim_k = torch.where(sim >= min_vals, sim, float('-inf'))

            # Restore full attention for special tokens (cls, subj) on live keys
            if x.shape[1] >= 2:
                active_k = torch.ones_like(sim[:, 0, :], dtype=torch.bool)
                if attn_gate is not None:
                    active_k = repeat(attn_gate, 'b j -> (b h) j', h=h)
                for i in [-2, -1]:
                    sim_k[:, i, :] = torch.where(active_k, sim[:, i, :], float('-inf'))
        else:
            sim_k = sim

        attn = sim_k.softmax(dim=-1)

        # Propagate gate: keys that received zero attention everywhere are dead
        next_attn_gate = ~torch.all(torch.isnan(attn), dim=-1)
        attn = attn.nan_to_num(nan=0.0)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out), next_attn_gate


class Decoder(nn.Module):
    """Simple linear decoder from latent space to voxel space."""

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim, bias=False),
            Reduce('b n v -> b v', 'mean'),
        )

    def forward(self, x):
        return self.decoder_layers(x)
