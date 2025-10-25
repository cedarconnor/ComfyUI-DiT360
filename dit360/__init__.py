"""
DiT360 model implementation

This package contains the core DiT360 diffusion transformer implementation:
- Model architecture (FLUX.1-dev based)
- Sampling algorithms (flow matching)
- Text conditioning (T5-XXL)
- Geometric losses (yaw, cube)
"""

from .model import (
    DiT360Model,
    DiT360Wrapper,
    load_dit360_model,
    download_dit360_from_huggingface,
    get_model_info,
)

from .vae import (
    DiT360VAE,
    load_vae,
    download_vae_from_huggingface,
)

from .conditioning import (
    T5TextEncoder,
    load_t5_encoder,
    download_t5_from_huggingface,
    text_preprocessing,
)

__all__ = [
    # Model
    'DiT360Model',
    'DiT360Wrapper',
    'load_dit360_model',
    'download_dit360_from_huggingface',
    'get_model_info',
    # VAE
    'DiT360VAE',
    'load_vae',
    'download_vae_from_huggingface',
    # Text Encoding
    'T5TextEncoder',
    'load_t5_encoder',
    'download_t5_from_huggingface',
    'text_preprocessing',
]
