from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class MockModelConfig(BaseModelConfig):
    model_name: str = "mock_fm"
    embed_dim: int = 512
