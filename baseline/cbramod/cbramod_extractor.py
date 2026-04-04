"""CBraMod Extractor — wraps CBraMod into BaseExtractor interface.

Loads pretrained weights from weighting666/CBraMod (HuggingFace).
Uses Asymmetric Conditional Positional Encoding (no channel mapping needed).
"""

import os

import torch
from torch import Tensor, nn

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .cbramod_config import CBraModModelConfig
from .model import CBraMod


@register_extractor("cbramod")
class CBraModExtractor(BaseExtractor):
    """CBraMod Foundation Model extractor.

    Maps (B, 30, T) -> (B, 200) via:
        1. Reshape to (B, 30, n_patches, 200)
        2. CNN + spectral patch embedding + ACPE
        3. 12-layer Criss-Cross Transformer (parallel spatial + temporal attention)
        4. Mean pooling over channels and patches
    """

    CONFIG_CLASS = CBraModModelConfig

    def __init__(self, config: CBraModModelConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.config = config
        self.patch_size = config.patch_size

        # Build model
        self.model = CBraMod(
            in_dim=config.patch_size,
            out_dim=config.embed_dim,
            d_model=config.embed_dim,
            dim_feedforward=config.dim_feedforward,
            seq_len=30,  # not used in forward pass
            n_layer=config.depth,
            nhead=config.nhead,
        )

        # Remove reconstruction head — we only need encoder features
        self.model.proj_out = nn.Identity()

        # Pooling: (B, C, n_patches, D) -> (B, D)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Load pretrained weights
        self._load_weights(config)

    def _load_weights(self, config: CBraModModelConfig):
        pth_path = os.path.join(config.pretrained_path, "pretrained_weights.pth")
        if os.path.isfile(pth_path):
            state = torch.load(pth_path, map_location="cpu", weights_only=True)
            # Filter out proj_out keys since we replaced it with Identity
            filtered = {k: v for k, v in state.items()
                        if not k.startswith("proj_out")}
            missing, unexpected = self.model.load_state_dict(filtered, strict=False)
            n_loaded = len(filtered) - len(unexpected)
            n_total = len([k for k in self.model.state_dict()
                           if not k.startswith("proj_out")])
            print(f"[CBraMod] Loaded {n_loaded}/{n_total} params from {pth_path}")
            if missing:
                non_proj = [m for m in missing if not m.startswith("proj_out")]
                if non_proj:
                    print(f"[CBraMod] Missing: {non_proj[:5]}")
        else:
            print(f"[CBraMod] No pretrained weights found at {pth_path}, using random init")

    def get_layer_groups(self) -> list[list["torch.nn.Parameter"]]:
        """CBraMod layer groups: [patch_embedding, layer_0, ..., layer_11]."""
        groups = []
        # Group 0: patch embedding (CNN + spectral + positional encoding)
        groups.append(list(self.model.patch_embedding.parameters()))
        # Groups 1..depth: transformer layers
        for layer in self.model.encoder.layers:
            groups.append(list(layer.parameters()))
        return groups

    def forward(self, x: Tensor) -> Tensor:
        """Extract features from EEG.

        Args:
            x: (B, C=30, T) raw EEG signal where T is divisible by patch_size=200

        Returns:
            (B, embed_dim=200) feature vectors
        """
        B, C, T = x.shape
        n_patches = T // self.patch_size

        # Reshape to CBraMod expected format: (B, n_channels, n_patches, patch_size)
        x = x.reshape(B, C, n_patches, self.patch_size)

        # Normalize: CBraMod expects data / 100 (uV scale to ~[-1, 1])
        x = x / 100.0

        # Forward through encoder (proj_out is Identity)
        # Output: (B, C, n_patches, embed_dim)
        feats = self.model(x)

        # Pool: (B, C, n_patches, D) -> (B, D)
        # Rearrange to (B, D, C, n_patches) for AdaptiveAvgPool2d
        feats = feats.permute(0, 3, 1, 2)  # (B, D, C, n_patches)
        pooled = self.pool(feats).squeeze(-1).squeeze(-1)  # (B, D)

        return pooled
