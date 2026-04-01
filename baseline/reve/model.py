"""REVE model architecture — ported from EEG-FM-Bench (xw1216/EEG-FM-Bench).

Original: https://github.com/xw1216/EEG-FM-Bench/blob/main/baseline/reve/model.py
"""

import math
from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .reve_config import ReveModelConfig


# ──────────────────── Layers ────────────────────


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, geglu: bool):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────── Attention ────────────────────


class ClassicalAttention(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = ClassicalAttention(self.heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


# ──────────────────── Transformer ────────────────────


class TransformerBackbone(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, head_dim: int, mlp_dim: int, geglu: bool):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads=heads, head_dim=head_dim),
                    FeedForward(dim, mlp_dim, geglu),
                ])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ──────────────────── 4D Fourier PE ────────────────────


class FourierEmb4D(nn.Module):
    """Fourier positional embedding for 4D positions (x, y, z, t)."""

    def __init__(self, dimension: int, freqs: int, increment_time: float = 0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *U, _ = positions.shape

        freqs_w = torch.arange(self.freqs, device=positions.device, dtype=positions.dtype)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        ).view(*U, -1)
        if self.dimension != 512:
            _, _, hd = loc.shape
            diff = hd - self.dimension // 2
            if diff > 0:
                loc = loc[:, :, :-diff]
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @staticmethod
    def add_time_patch(pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Expand (B, C, 3) positions to (B, C*num_patches, 4) with time dimension."""
        B, C, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)  # (B, C, T, 3)
        time_values = torch.arange(0, num_patches, 1, device=pos.device, dtype=pos.dtype)
        time_values = time_values.view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)  # (B, C, T, 4)
        return pos_with_time.view(B, C * num_patches, 4)


# ──────────────────── REVE Model ────────────────────


class Reve(nn.Module):
    """REVE: Representation of EEG Via Encoders."""

    def __init__(self, cfg: ReveModelConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.overlap_size = cfg.patch_overlap

        self.transformer = TransformerBackbone(
            dim=cfg.embed_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            head_dim=cfg.head_dim,
            mlp_dim=int(cfg.embed_dim * cfg.mlp_dim_ratio),
            geglu=cfg.use_geglu,
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(cfg.patch_size, cfg.embed_dim),
        )
        self.fourier4d = FourierEmb4D(cfg.embed_dim, freqs=cfg.freqs)
        self.mlp4d = nn.Sequential(
            nn.Linear(4, cfg.embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(cfg.embed_dim),
        )
        self.ln = nn.LayerNorm(cfg.embed_dim)
        self.final_layer = nn.Identity()
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim))

    def forward(self, eeg: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: (B, C, T) raw EEG signal
            pos: (B, C, 3) electrode 3D coordinates

        Returns:
            (B, C, n_patches, embed_dim) transformer output
        """
        eeg = eeg.float()
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        _b, c, h, _p = patches.shape

        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))

        x = rearrange(
            self.to_patch_embedding(patches),
            "b c h e -> b (c h) e", c=c, h=h, e=self.embed_dim,
        ) + pos_embed

        x = self.transformer(x)
        x = rearrange(x, "b (c h) e -> b c h e", b=_b, c=c, h=h, e=self.embed_dim)
        x = self.final_layer(x)
        return x

    def attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Attention pooling: (B, C, S, E) → (B, E)."""
        b, c, s, e = x.shape
        x = rearrange(x, "b c s e -> b (c s) e")
        query = self.cls_query_token.expand(b, -1, -1)
        scores = torch.matmul(query, x.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, x).squeeze(1)
        return out
