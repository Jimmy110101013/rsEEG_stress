from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    """Per-model configuration declaring FM-specific parameters."""

    model_name: str = "base"
    embed_dim: int = 512
    sample_rate: int = 200
    n_channels: int = 30
    patch_size: int = 200  # REVE default: 1 second at 200Hz
    window_sec: float = 10.0  # epoch length in seconds

    @property
    def n_samples(self) -> int:
        """Number of time samples per epoch window."""
        return int(self.sample_rate * self.window_sec)
