from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class ReveModelConfig(BaseModelConfig):
    """REVE Foundation Model configuration."""

    model_name: str = "reve"
    embed_dim: int = 512
    sample_rate: int = 200
    patch_size: int = 200       # 1 second at 200Hz
    patch_overlap: int = 20

    # Transformer architecture
    depth: int = 22
    heads: int = 8
    head_dim: int = 64
    mlp_dim_ratio: float = 2.66
    use_geglu: bool = True

    # Positional encoding
    freqs: int = 4              # Fourier PE frequency count
    noise_ratio: float = 0.0025

    # Pretrained weight paths (local dirs from HF download)
    pretrained_path: str = "weights/reve-base"
    pos_bank_path: str = "weights/reve-positions"
