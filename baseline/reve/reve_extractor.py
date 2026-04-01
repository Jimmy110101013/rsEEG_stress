"""REVE Extractor — wraps the REVE model into our BaseExtractor interface.

Loads pretrained weights from brain-bzh/reve-base and brain-bzh/reve-positions.
Falls back to random initialization if weights are unavailable.
"""

import os

import torch
from torch import Tensor

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .reve_config import ReveModelConfig
from .model import Reve
from .pos_bank import RevePositionBank, OUR_CHANNELS


@register_extractor("reve")
class ReveExtractor(BaseExtractor):
    """REVE Foundation Model extractor.

    Maps (B, 30, 2000) → (B, 512) via:
        1. Patch embedding + 4D Fourier PE
        2. 22-layer Transformer
        3. Attention pooling (CLS query)
    """

    CONFIG_CLASS = ReveModelConfig

    def __init__(self, config: ReveModelConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.config = config

        # Position bank for 3D electrode coordinates
        self.pos_bank = RevePositionBank()

        # Core REVE model
        self.reve = Reve(config)

        # Load pretrained weights
        self._load_weights(config)

        # Precompute and cache the 3D positions for our 30-channel montage
        pos_3d = self.pos_bank(OUR_CHANNELS)  # (30, 3)
        self.register_buffer("channel_positions", pos_3d)

    def _load_weights(self, config: ReveModelConfig):
        """Load pretrained weights if available, otherwise use random init."""
        loaded_encoder = False
        loaded_posbank = False

        # Try loading position bank
        if os.path.isdir(config.pos_bank_path):
            try:
                self.pos_bank.load_pretrained(config.pos_bank_path)
                loaded_posbank = True
            except Exception as e:
                print(f"[REVE] Failed to load position bank: {e}")

        if not loaded_posbank:
            self.pos_bank.init_fallback(OUR_CHANNELS)

        # Try loading encoder weights
        if os.path.isdir(config.pretrained_path):
            safetensor_path = os.path.join(config.pretrained_path, "model.safetensors")
            if os.path.isfile(safetensor_path):
                try:
                    from safetensors.torch import load_file
                    state = load_file(safetensor_path)

                    # FM-Bench and HF may use different key prefixes — try to align
                    reve_state = self.reve.state_dict()
                    mapped_state = {}
                    for key, val in state.items():
                        # Strip common prefixes
                        clean_key = key
                        for prefix in ["model.", "encoder.", "reve."]:
                            if clean_key.startswith(prefix):
                                clean_key = clean_key[len(prefix):]
                        if clean_key in reve_state and val.shape == reve_state[clean_key].shape:
                            mapped_state[clean_key] = val

                    if mapped_state:
                        self.reve.load_state_dict(mapped_state, strict=False)
                        loaded_encoder = True
                        print(f"[REVE] Loaded {len(mapped_state)}/{len(reve_state)} encoder params from {safetensor_path}")
                    else:
                        print(f"[REVE] No matching keys found in {safetensor_path}")
                except Exception as e:
                    print(f"[REVE] Failed to load encoder weights: {e}")

        if not loaded_encoder:
            print("[REVE] Using randomly initialized encoder weights")

    def forward(self, x: Tensor) -> Tensor:
        """Extract features from EEG epochs.

        Args:
            x: (B, C=30, T=2000) raw EEG signal

        Returns:
            (B, embed_dim=512) feature vectors
        """
        B = x.shape[0]

        # Expand cached positions to batch: (30, 3) → (B, 30, 3)
        pos = self.channel_positions.unsqueeze(0).expand(B, -1, -1)

        # Forward through REVE: (B, C, T) + (B, C, 3) → (B, C, n_patches, E)
        features = self.reve(x, pos)

        # Attention pooling over all channel-patch tokens → (B, E)
        pooled = self.reve.attention_pooling(features)

        return pooled
