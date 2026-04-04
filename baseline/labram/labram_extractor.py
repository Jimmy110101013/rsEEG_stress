"""LaBraM Extractor — wraps NeuralTransformer into BaseExtractor interface.

Loads pretrained weights from braindecode/labram-pretrained (HuggingFace).
Maps our 30-channel montage to LaBraM's 128-slot position embedding.
"""

import os
from functools import partial

import torch
from torch import Tensor, nn

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .labram_config import LaBraMModelConfig
from .model import NeuralTransformer
from .channel_map import get_input_chans


@register_extractor("labram")
class LaBraMExtractor(BaseExtractor):
    """LaBraM Foundation Model extractor.

    Maps (B, 30, T) → (B, 200) via:
        1. Reshape to (B, 30, n_patches, 200)
        2. TemporalConv tokenizer
        3. 12-layer Transformer with channel + time position embeddings
        4. Mean pooling over patch tokens
    """

    CONFIG_CLASS = LaBraMModelConfig

    def __init__(self, config: LaBraMModelConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.config = config
        self.patch_size = config.patch_size

        # Build model
        self.model = NeuralTransformer(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            out_chans=config.out_chans,
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            init_values=config.init_values,
        )

        # Load pretrained weights
        self._load_weights(config)

        # Precompute channel indices for our 30-channel montage
        self.input_chans = get_input_chans()

    def _load_weights(self, config: LaBraMModelConfig):
        safetensor_path = os.path.join(config.pretrained_path, "model.safetensors")
        if os.path.isfile(safetensor_path):
            from safetensors.torch import load_file
            state = load_file(safetensor_path)
            # Map braindecode weight keys to our model
            mapped = self._map_weights(state)
            missing, unexpected = self.model.load_state_dict(mapped, strict=False)
            n_loaded = len(mapped) - len(unexpected)
            n_total = len(self.model.state_dict())
            print(f"[LaBraM] Loaded {n_loaded}/{n_total} params from {safetensor_path}")
            if missing:
                print(f"[LaBraM] Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
        else:
            pth_path = os.path.join(config.pretrained_path, "labram-base.pth")
            if os.path.isfile(pth_path):
                state = torch.load(pth_path, map_location="cpu", weights_only=True)
                if "model" in state:
                    state = state["model"]
                mapped = self._map_weights(state)
                self.model.load_state_dict(mapped, strict=False)
                print(f"[LaBraM] Loaded weights from {pth_path}")
            else:
                print("[LaBraM] No pretrained weights found, using random init")

    def _map_weights(self, state: dict) -> dict:
        """Map weight keys from braindecode/original format to our model."""
        mapped = {}
        model_state = self.model.state_dict()

        for key, val in state.items():
            # Strip common prefixes (braindecode may add 'model.' etc.)
            clean = key
            for prefix in ["model.", "encoder.", "neural_transformer."]:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]

            # Braindecode MLP keys: mlp.0.* → mlp.layers.0.*
            if ".mlp." in clean and ".mlp.layers." not in clean:
                clean = clean.replace(".mlp.", ".mlp.layers.")

            if clean in model_state and val.shape == model_state[clean].shape:
                mapped[clean] = val

        return mapped

    def get_layer_groups(self) -> list[list["torch.nn.Parameter"]]:
        """LaBraM layer groups: [tokenizer+embeddings, block_0, ..., block_11, norm]."""
        groups = []
        # Group 0: tokenizer + embeddings
        embed_params = []
        for name, p in self.model.named_parameters():
            if "blocks." not in name and "fc_norm" not in name and "norm." not in name:
                embed_params.append(p)
        groups.append(embed_params)
        # Groups 1..depth: transformer blocks
        for block in self.model.blocks:
            groups.append(list(block.parameters()))
        # Final group: norms
        norm_params = list(self.model.norm.parameters()) + list(self.model.fc_norm.parameters())
        groups.append(norm_params)
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

        # Reshape to LaBraM expected format: (B, n_channels, n_patches, patch_size)
        x = x.reshape(B, C, n_patches, self.patch_size)

        # Forward with our channel position indices
        return self.model(x, input_chans=self.input_chans)
