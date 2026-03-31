from typing import Dict, Tuple, Type

from .base_extractor import BaseExtractor
from .base_config import BaseModelConfig

EXTRACTOR_REGISTRY: Dict[str, Tuple[Type[BaseExtractor], Type[BaseModelConfig]]] = {}


def register_extractor(name: str):
    """Decorator to register an extractor class with its config."""

    def decorator(cls: Type[BaseExtractor]):
        config_cls = getattr(cls, "CONFIG_CLASS", BaseModelConfig)
        EXTRACTOR_REGISTRY[name] = (cls, config_cls)
        return cls

    return decorator


def create_extractor(name: str, **config_overrides) -> BaseExtractor:
    """Instantiate a registered extractor by name."""
    if name not in EXTRACTOR_REGISTRY:
        available = list(EXTRACTOR_REGISTRY.keys())
        raise ValueError(f"Unknown extractor '{name}'. Available: {available}")

    extractor_cls, config_cls = EXTRACTOR_REGISTRY[name]
    config = config_cls(model_name=name, **config_overrides)
    return extractor_cls(config)
