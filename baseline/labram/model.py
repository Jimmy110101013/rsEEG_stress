"""LaBraM model architecture — ported from 935963004/LaBraM (ICLR 2024).

Key names match braindecode/labram-pretrained HuggingFace weights.
Original: https://github.com/935963004/LaBraM/blob/main/modeling_finetune.py
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_prob
        return x / (1 - self.drop_prob) * keep


class Mlp(nn.Module):
    """MLP with keys matching braindecode weights: mlp.0 (fc1), mlp.2 (fc2)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        # Sequential so keys are .0, .1, .2, .3 matching braindecode weights
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.q_norm = qk_norm(head_dim) if qk_norm else None
        self.k_norm = qk_norm(head_dim) if qk_norm else None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ])
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden, drop=drop)

        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TemporalConv(nn.Module):
    """Temporal convolution tokenizer for raw EEG.

    Key names match braindecode weights: patch_embed.temporal_conv.*
    """
    def __init__(self, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x):
        # x: (B, n_channels, n_patches, patch_size) e.g. (B, 30, 5, 200)
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)  # (B, 1, N*A, T)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class NeuralTransformer(nn.Module):
    """LaBraM Neural Transformer.

    Input: (B, n_channels, n_patches, patch_size) e.g. (B, 30, 5, 200)
    Output: (B, embed_dim) mean-pooled features (when use_mean_pooling=True)
    """
    def __init__(self, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.,
                 qkv_bias=False, qk_norm=None, out_chans=8,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Temporal convolution tokenizer
        self.patch_embed = nn.Module()
        self.patch_embed.temporal_conv = TemporalConv(out_chans=out_chans)

        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position & time embeddings
        # 128 channel slots + 1 CLS = 129
        self.position_embedding = nn.Parameter(torch.zeros(1, 129, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 16, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_norm=qk_norm, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)
        ])

        # Normalization: pretrained weights use CLS pooling (norm on all tokens),
        # but we mean-pool patch tokens with fc_norm for feature extraction
        self.norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim)

        # Init
        trunc_normal_(self.position_embedding, std=0.02)
        trunc_normal_(self.temporal_embedding, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.layers[2].weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward_features(self, x, input_chans=None):
        """
        Args:
            x: (B, n_channels, n_patches, patch_size)
            input_chans: list of ints — position embedding indices
                         [0, idx1, idx2, ...] where 0=CLS

        Returns:
            (B, embed_dim) mean-pooled features
        """
        B, n_ch, n_patches, patch_size = x.shape

        # Tokenize: (B, n_ch, n_patches, patch_size) → (B, n_ch*n_patches, embed_dim)
        x = self.patch_embed.temporal_conv(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding (channel-level)
        if input_chans is not None:
            pos_embed = self.position_embedding[:, input_chans]
        else:
            pos_embed = self.position_embedding[:, :n_ch + 1]

        # Expand pos_embed per time patch: (B, n_ch+1, E) → (B, 1 + n_ch*n_patches, E)
        pos_cls = pos_embed[:, 0:1, :].expand(B, -1, -1)
        pos_ch = pos_embed[:, 1:, :].unsqueeze(2).expand(B, -1, n_patches, -1).flatten(1, 2)
        x = x + torch.cat([pos_cls, pos_ch], dim=1)

        # Add temporal embedding
        time_embed = self.temporal_embedding[:, :n_patches, :]
        time_embed = time_embed.unsqueeze(1).expand(B, n_ch, -1, -1).flatten(1, 2)
        x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Mean pooling over patch tokens (exclude CLS)
        return self.fc_norm(x[:, 1:, :].mean(dim=1))

    def forward(self, x, input_chans=None):
        return self.forward_features(x, input_chans=input_chans)
