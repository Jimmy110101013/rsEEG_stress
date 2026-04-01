"""REVE Position Bank — maps EEG channel names to 3D electrode coordinates.

The pretrained position bank from brain-bzh/reve-positions contains learned
3D embeddings for hundreds of standard electrode names. We load those weights
and look up positions for our 30-channel montage at runtime.
"""

import json
import os

import torch
from torch import nn


# Standard 10-20 system 3D coordinates (unit sphere approximation) for our 30 channels.
# Used as fallback when pretrained position bank is unavailable.
CHANNEL_COORDS_30 = {
    "FP1": (-0.309, 0.951, 0.0),
    "FP2": (0.309, 0.951, 0.0),
    "F7": (-0.809, 0.588, 0.0),
    "F3": (-0.545, 0.673, 0.5),
    "FZ": (0.0, 0.719, 0.695),
    "F4": (0.545, 0.673, 0.5),
    "F8": (0.809, 0.588, 0.0),
    "FT7": (-0.951, 0.309, 0.0),
    "FC3": (-0.673, 0.345, 0.654),
    "FCZ": (0.0, 0.391, 0.921),
    "FC4": (0.673, 0.345, 0.654),
    "FT8": (0.951, 0.309, 0.0),
    "T3": (-1.0, 0.0, 0.0),
    "C3": (-0.719, 0.0, 0.695),
    "CZ": (0.0, 0.0, 1.0),
    "C4": (0.719, 0.0, 0.695),
    "T4": (1.0, 0.0, 0.0),
    "TP7": (-0.951, -0.309, 0.0),
    "CP3": (-0.673, -0.345, 0.654),
    "CPZ": (0.0, -0.391, 0.921),
    "CP4": (0.673, -0.345, 0.654),
    "TP8": (0.951, -0.309, 0.0),
    "T5": (-0.809, -0.588, 0.0),
    "P3": (-0.545, -0.673, 0.5),
    "PZ": (0.0, -0.719, 0.695),
    "P4": (0.545, -0.673, 0.5),
    "T6": (0.809, -0.588, 0.0),
    "O1": (-0.309, -0.951, 0.0),
    "OZ": (0.0, -1.0, 0.0),
    "O2": (0.309, -0.951, 0.0),
}

OUR_CHANNELS = list(CHANNEL_COORDS_30.keys())


class RevePositionBank(nn.Module):
    """Loads the pretrained REVE position bank and looks up 3D coords by channel name."""

    def __init__(self):
        super().__init__()
        self.position_names: list[str] = []
        self.mapping: dict[str, int] = {}
        # Placeholder — will be overwritten by load_pretrained or fallback
        self.register_buffer("embedding", torch.zeros(1, 3))

    def load_pretrained(self, pos_bank_dir: str):
        """Load position bank from brain-bzh/reve-positions download directory."""
        from safetensors.torch import load_file

        # Load positions.json for the name → index mapping
        json_path = os.path.join(pos_bank_dir, "positions.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                positions_data = json.load(f)
            self.position_names = positions_data if isinstance(positions_data, list) else list(positions_data.keys())
        else:
            # Fallback: try config.json
            config_path = os.path.join(pos_bank_dir, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    config_data = json.load(f)
                self.position_names = config_data.get("position_names", [])

        # Load safetensors weights
        safetensor_path = os.path.join(pos_bank_dir, "model.safetensors")
        if os.path.isfile(safetensor_path):
            state = load_file(safetensor_path)
            # The embedding tensor key varies — find it
            for key, tensor in state.items():
                if tensor.ndim == 2 and tensor.shape[1] == 3:
                    self.embedding = tensor
                    break

        self.mapping = {name.upper(): i for i, name in enumerate(self.position_names)}
        print(f"[PosBank] Loaded {len(self.position_names)} positions from {pos_bank_dir}")

    def init_fallback(self, channel_names: list[str]):
        """Initialize with hardcoded 10-20 coordinates when pretrained bank unavailable."""
        coords = []
        for ch in channel_names:
            ch_upper = ch.upper()
            if ch_upper in CHANNEL_COORDS_30:
                coords.append(CHANNEL_COORDS_30[ch_upper])
            else:
                coords.append((0.0, 0.0, 0.0))
        self.embedding = torch.tensor(coords, dtype=torch.float32)
        self.position_names = [ch.upper() for ch in channel_names]
        self.mapping = {name: i for i, name in enumerate(self.position_names)}
        print(f"[PosBank] Fallback: using hardcoded 10-20 coords for {len(channel_names)} channels")

    def forward(self, channel_names: list[str]) -> torch.Tensor:
        """Look up 3D positions for given channel names.

        Returns: (n_channels, 3) tensor of 3D coordinates.
        """
        channel_names = [cn.upper() for cn in channel_names]
        # Handle common aliases
        channel_names = ["TP7" if cn == "T1" else "TP8" if cn == "T2" else cn for cn in channel_names]

        indices = []
        for cn in channel_names:
            if cn in self.mapping:
                indices.append(self.mapping[cn])
            else:
                print(f"[PosBank] Warning: channel '{cn}' not found in position bank")
                indices.append(0)  # fallback to first position

        idx_tensor = torch.tensor(indices, device=self.embedding.device)
        return self.embedding[idx_tensor]
