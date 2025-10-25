"""
DiT360 model implementation

This package contains the core DiT360 diffusion transformer implementation:
- Model architecture (FLUX.1-dev based)
- Sampling algorithms (flow matching)
- Text conditioning (T5-XXL)
- Geometric losses (yaw, cube)
- Advanced features (LoRA, inpainting, projections)
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

from .scheduler import (
    FlowMatchScheduler,
    CFGFlowMatchScheduler,
    get_timestep_schedule,
    compute_snr,
)

from .losses import (
    YawLoss,
    CubeLoss,
    rotate_equirect_yaw,
    equirect_to_cubemap,
    cubemap_to_equirect,
    compute_yaw_consistency,
)

from .projection import (
    create_equirect_to_cube_grid,
    equirect_to_cubemap_fast,
    cubemap_to_equirect_fast,
    compute_projection_distortion,
    apply_distortion_weighted_loss,
    split_cubemap_horizontal,
    split_cubemap_cross,
)

from .lora import (
    LoRALayer,
    LoRACollection,
    load_lora_from_safetensors,
    merge_lora_into_model,
    unmerge_lora_from_model,
    combine_loras,
)

from .inpainting import (
    prepare_inpaint_mask,
    gaussian_blur_mask,
    expand_mask,
    create_latent_noise_mask,
    blend_latents,
    apply_inpainting_conditioning,
    create_circular_mask,
    create_rectangle_mask,
    create_horizon_mask,
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
    # Scheduler
    'FlowMatchScheduler',
    'CFGFlowMatchScheduler',
    'get_timestep_schedule',
    'compute_snr',
    # Losses
    'YawLoss',
    'CubeLoss',
    'rotate_equirect_yaw',
    'equirect_to_cubemap',
    'cubemap_to_equirect',
    'compute_yaw_consistency',
    # Projection
    'create_equirect_to_cube_grid',
    'equirect_to_cubemap_fast',
    'cubemap_to_equirect_fast',
    'compute_projection_distortion',
    'apply_distortion_weighted_loss',
    'split_cubemap_horizontal',
    'split_cubemap_cross',
    # LoRA
    'LoRALayer',
    'LoRACollection',
    'load_lora_from_safetensors',
    'merge_lora_into_model',
    'unmerge_lora_from_model',
    'combine_loras',
    # Inpainting
    'prepare_inpaint_mask',
    'gaussian_blur_mask',
    'expand_mask',
    'create_latent_noise_mask',
    'blend_latents',
    'apply_inpainting_conditioning',
    'create_circular_mask',
    'create_rectangle_mask',
    'create_horizon_mask',
]
