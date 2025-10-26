"""
DiT360 utilities for 360° panorama generation

This module contains utilities for geometric losses and projection operations
used in panoramic image generation.

Note: DiT360 is a LoRA for FLUX.1-dev, not a standalone model.
Use standard ComfyUI nodes to load FLUX and apply the DiT360 LoRA.
"""

from .losses import (
    YawLoss,
    CubeLoss,
    rotate_equirect_yaw,
    compute_yaw_consistency,
)

from .projection import (
    equirect_to_cubemap,
    cubemap_to_equirect,
    create_equirect_to_cube_grid,
    equirect_to_cubemap_fast,
    cubemap_to_equirect_fast,
    compute_projection_distortion,
    apply_distortion_weighted_loss,
    split_cubemap_horizontal,
    split_cubemap_cross,
)

__all__ = [
    # Geometric Losses
    'YawLoss',
    'CubeLoss',
    'rotate_equirect_yaw',
    'compute_yaw_consistency',
    # Projection Utilities
    'equirect_to_cubemap',
    'cubemap_to_equirect',
    'create_equirect_to_cube_grid',
    'equirect_to_cubemap_fast',
    'cubemap_to_equirect_fast',
    'compute_projection_distortion',
    'apply_distortion_weighted_loss',
    'split_cubemap_horizontal',
    'split_cubemap_cross',
]
