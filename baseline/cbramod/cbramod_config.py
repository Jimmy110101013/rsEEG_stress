from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class CBraModModelConfig(BaseModelConfig):
    """CBraMod Foundation Model configuration (base variant)."""

    model_name: str = "cbramod"
    embed_dim: int = 200
    sample_rate: int = 200
    patch_size: int = 200       # 1 second at 200Hz
    window_sec: float = 5.0     # 5s windows per stress paper

    # Transformer
    depth: int = 12
    nhead: int = 8
    dim_feedforward: int = 800

    # Pretrained weights
    pretrained_path: str = "weights/cbramod-base"
