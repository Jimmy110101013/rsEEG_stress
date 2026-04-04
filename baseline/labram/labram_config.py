from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class LaBraMModelConfig(BaseModelConfig):
    """LaBraM Foundation Model configuration (base variant)."""

    model_name: str = "labram"
    embed_dim: int = 200
    sample_rate: int = 200
    patch_size: int = 200       # 1 second at 200Hz
    window_sec: float = 5.0     # 5s windows per stress paper

    # Transformer
    depth: int = 12
    heads: int = 10
    mlp_ratio: float = 4.0
    out_chans: int = 8          # TemporalConv output channels
    init_values: float = 0.1
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    # Pretrained weights
    pretrained_path: str = "weights/labram-base"
