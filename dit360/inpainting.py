"""
Inpainting support for DiT360 panoramic images.

This module provides utilities for masked generation where specific regions
of a panorama can be regenerated while keeping other regions fixed.

Author: DiT360 Team
License: Apache 2.0
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


def prepare_inpaint_mask(
    mask: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
    blur_radius: int = 0,
    invert: bool = False
) -> torch.Tensor:
    """
    Prepare a mask for inpainting.

    Args:
        mask: Input mask (B, 1, H, W) or (B, H, W, 1) or (H, W)
            Values: 0 = keep original, 1 = inpaint
        target_size: Resize mask to (height, width) if provided
        blur_radius: Blur mask edges for smooth blending (0 = no blur)
        invert: If True, invert mask (1 -> 0, 0 -> 1)

    Returns:
        Prepared mask (B, 1, H, W) with values in [0, 1]

    Example:
        >>> mask = torch.zeros(1, 1, 1024, 2048)
        >>> mask[:, :, 400:600, 800:1200] = 1.0  # Inpaint center region
        >>> prepared = prepare_inpaint_mask(mask, blur_radius=10)
    """
    # Normalize shape to (B, 1, H, W)
    if mask.dim() == 2:  # (H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:  # (B, H, W)
        mask = mask.unsqueeze(1)
    elif mask.dim() == 4:  # (B, H, W, C) or (B, C, H, W)
        if mask.shape[-1] == 1:  # (B, H, W, 1)
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] != 1:  # (B, C, H, W) with C > 1
            # Take first channel or average
            if mask.shape[1] == 3:  # RGB
                mask = mask.mean(dim=1, keepdim=True)
            else:
                mask = mask[:, 0:1, :, :]

    # Ensure float and [0, 1] range
    mask = mask.float()
    if mask.max() > 1.0:
        mask = mask / 255.0

    # Invert if requested
    if invert:
        mask = 1.0 - mask

    # Resize if needed
    if target_size is not None:
        height, width = target_size
        mask = F.interpolate(
            mask,
            size=(height, width),
            mode='bilinear',
            align_corners=True
        )

    # Blur edges for smooth blending
    if blur_radius > 0:
        mask = gaussian_blur_mask(mask, radius=blur_radius)

    # Clamp to [0, 1]
    mask = torch.clamp(mask, 0.0, 1.0)

    return mask


def gaussian_blur_mask(mask: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """
    Apply Gaussian blur to a mask for smooth edge blending.

    Args:
        mask: Input mask (B, 1, H, W)
        radius: Blur radius in pixels

    Returns:
        Blurred mask (B, 1, H, W)

    Example:
        >>> mask = torch.zeros(1, 1, 256, 256)
        >>> mask[:, :, 100:150, 100:150] = 1.0
        >>> blurred = gaussian_blur_mask(mask, radius=10)
    """
    if radius <= 0:
        return mask

    # Create Gaussian kernel
    kernel_size = radius * 2 + 1
    sigma = radius / 3.0

    # Generate 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - radius
    gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    # Create 2D kernel
    gaussian_2d = gaussian_1d.view(-1, 1) @ gaussian_1d.view(1, -1)
    gaussian_2d = gaussian_2d.view(1, 1, kernel_size, kernel_size)

    # Apply convolution
    padding = radius
    blurred = F.conv2d(mask, gaussian_2d, padding=padding)

    return blurred


def expand_mask(
    mask: torch.Tensor,
    expand_pixels: int,
    circular: bool = True
) -> torch.Tensor:
    """
    Expand mask by dilating (growing) the masked region.

    This is useful for ensuring complete coverage of objects and smooth edges.

    Args:
        mask: Input mask (B, 1, H, W), values in [0, 1]
        expand_pixels: Number of pixels to expand
        circular: Use circular padding for panoramas (default: True)

    Returns:
        Expanded mask (B, 1, H, W)

    Example:
        >>> mask = torch.zeros(1, 1, 256, 512)
        >>> mask[:, :, 100:150, 200:250] = 1.0
        >>> expanded = expand_mask(mask, expand_pixels=10)
    """
    if expand_pixels <= 0:
        return mask

    # Use max pooling for dilation
    kernel_size = expand_pixels * 2 + 1
    padding = expand_pixels

    if circular:
        # Apply circular padding for panoramas
        try:
            from utils.padding import apply_circular_padding
        except (ImportError, ModuleNotFoundError):
            # Fallback: manual circular padding
            left_pad = mask[:, :, :, -expand_pixels:]
            right_pad = mask[:, :, :, :expand_pixels]
            mask_padded = torch.cat([left_pad, mask, right_pad], dim=3)
            expanded = F.max_pool2d(mask_padded, kernel_size=kernel_size, stride=1, padding=0)
            # Remove padding
            expanded = expanded[:, :, :, expand_pixels:-expand_pixels]
        else:
            mask_padded = apply_circular_padding(mask, padding=expand_pixels)
            expanded = F.max_pool2d(mask_padded, kernel_size=kernel_size, stride=1, padding=0)
            # Remove padding
            expanded = expanded[:, :, :, expand_pixels:-expand_pixels]
    else:
        # Standard padding
        expanded = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)

    return expanded


def create_latent_noise_mask(
    image_mask: torch.Tensor,
    latent_size: Tuple[int, int],
    vae_scale_factor: int = 8
) -> torch.Tensor:
    """
    Create a mask in latent space from an image-space mask.

    Args:
        image_mask: Mask in image space (B, 1, H, W)
        latent_size: Size of latent space (height, width)
        vae_scale_factor: VAE downscaling factor (default: 8)

    Returns:
        Mask in latent space (B, 1, latent_H, latent_W)

    Example:
        >>> image_mask = torch.ones(1, 1, 2048, 4096)
        >>> latent_mask = create_latent_noise_mask(image_mask, (256, 512), vae_scale_factor=8)
        >>> latent_mask.shape
        torch.Size([1, 1, 256, 512])
    """
    latent_height, latent_width = latent_size

    # Downsample mask to latent resolution
    latent_mask = F.interpolate(
        image_mask,
        size=(latent_height, latent_width),
        mode='bilinear',
        align_corners=True
    )

    # Threshold to binary
    latent_mask = (latent_mask > 0.5).float()

    return latent_mask


def blend_latents(
    original_latent: torch.Tensor,
    generated_latent: torch.Tensor,
    mask: torch.Tensor,
    blend_mode: str = "linear"
) -> torch.Tensor:
    """
    Blend original and generated latents using a mask.

    Args:
        original_latent: Original latent (B, C, H, W)
        generated_latent: Generated/inpainted latent (B, C, H, W)
        mask: Blending mask (B, 1, H, W), 0 = original, 1 = generated
        blend_mode: Blending mode - "linear", "cosine", or "smooth"

    Returns:
        Blended latent (B, C, H, W)

    Example:
        >>> orig = torch.randn(1, 4, 256, 512)
        >>> gen = torch.randn(1, 4, 256, 512)
        >>> mask = torch.ones(1, 1, 256, 512) * 0.5
        >>> blended = blend_latents(orig, gen, mask, blend_mode="cosine")
    """
    if mask.shape[2:] != original_latent.shape[2:]:
        # Resize mask to match latent size
        mask = F.interpolate(
            mask,
            size=original_latent.shape[2:],
            mode='bilinear',
            align_corners=True
        )

    # Apply blending mode
    if blend_mode == "cosine":
        # Cosine interpolation (smoother at edges)
        weight = (1 - torch.cos(mask * math.pi)) / 2
    elif blend_mode == "smooth":
        # Smoothstep interpolation
        weight = mask * mask * (3 - 2 * mask)
    else:  # linear
        weight = mask

    # Blend
    blended = original_latent * (1 - weight) + generated_latent * weight

    return blended


def apply_inpainting_conditioning(
    latent: torch.Tensor,
    mask: torch.Tensor,
    original_image_latent: Optional[torch.Tensor] = None,
    fill_mode: str = "noise"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare latent and conditioning for inpainting.

    This function masks the latent and optionally fills masked regions with
    noise or other content for conditioning the generation process.

    Args:
        latent: Input latent (B, C, H, W)
        mask: Inpainting mask (B, 1, H, W), 1 = inpaint, 0 = keep
        original_image_latent: Optional original image latent for reference
        fill_mode: How to fill masked regions - "noise", "zero", "edge"

    Returns:
        Tuple of (conditioned_latent, conditioning_mask)

    Example:
        >>> latent = torch.randn(1, 4, 256, 512)
        >>> mask = torch.ones(1, 1, 256, 512)
        >>> mask[:, :, :, 200:300] = 0.0  # Keep middle section
        >>> cond_latent, cond_mask = apply_inpainting_conditioning(latent, mask)
    """
    # Ensure mask matches latent resolution
    if mask.shape[2:] != latent.shape[2:]:
        mask = F.interpolate(
            mask,
            size=latent.shape[2:],
            mode='bilinear',
            align_corners=True
        )

    # Threshold mask to binary
    binary_mask = (mask > 0.5).float()

    # Fill masked regions based on fill_mode
    if fill_mode == "noise":
        # Fill with random noise
        noise = torch.randn_like(latent)
        conditioned_latent = latent * (1 - binary_mask) + noise * binary_mask
    elif fill_mode == "zero":
        # Fill with zeros
        conditioned_latent = latent * (1 - binary_mask)
    elif fill_mode == "edge":
        # Fill using edge information (average of boundary pixels)
        # This is more complex, use noise as fallback
        noise = torch.randn_like(latent) * 0.1  # Small noise
        conditioned_latent = latent * (1 - binary_mask) + noise * binary_mask
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    return conditioned_latent, mask


def create_circular_mask(
    size: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a circular mask.

    Args:
        size: Mask size (height, width)
        center: Center coordinates (x, y) in range [0, 1]
        radius: Radius in range [0, 1] relative to min(height, width)
        device: Target device
        dtype: Data type

    Returns:
        Circular mask (1, 1, H, W)

    Example:
        >>> mask = create_circular_mask((1024, 2048), center=(0.5, 0.5), radius=0.2)
        >>> mask.shape
        torch.Size([1, 1, 1024, 2048])
    """
    height, width = size
    if device is None:
        device = torch.device('cpu')

    # Create coordinate grids
    y = torch.linspace(0, 1, height, device=device, dtype=dtype)
    x = torch.linspace(0, 1, width, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    # Compute distance from center
    center_x, center_y = center
    dx = x_grid - center_x
    dy = y_grid - center_y
    distance = torch.sqrt(dx ** 2 + dy ** 2)

    # Create mask
    mask = (distance <= radius).float()
    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


def create_rectangle_mask(
    size: Tuple[int, int],
    top_left: Tuple[float, float],
    bottom_right: Tuple[float, float],
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a rectangular mask.

    Args:
        size: Mask size (height, width)
        top_left: Top-left corner (x, y) in range [0, 1]
        bottom_right: Bottom-right corner (x, y) in range [0, 1]
        device: Target device
        dtype: Data type

    Returns:
        Rectangular mask (1, 1, H, W)

    Example:
        >>> mask = create_rectangle_mask((1024, 2048), (0.25, 0.25), (0.75, 0.75))
        >>> mask.shape
        torch.Size([1, 1, 1024, 2048])
    """
    height, width = size
    if device is None:
        device = torch.device('cpu')

    # Convert normalized coordinates to pixels
    x1, y1 = top_left
    x2, y2 = bottom_right

    x1_pix = int(x1 * width)
    x2_pix = int(x2 * width)
    y1_pix = int(y1 * height)
    y2_pix = int(y2 * height)

    # Create mask
    mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    mask[:, :, y1_pix:y2_pix, x1_pix:x2_pix] = 1.0

    return mask


def create_horizon_mask(
    size: Tuple[int, int],
    horizon_y: float = 0.5,
    height: float = 0.2,
    feather: float = 0.05,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a mask for the horizon region (useful for sky/ground inpainting).

    Args:
        size: Mask size (height, width)
        horizon_y: Horizon position (0 = top, 1 = bottom)
        height: Height of masked region around horizon (0-1)
        feather: Feathering amount for smooth edges (0-1)
        device: Target device
        dtype: Data type

    Returns:
        Horizon mask (1, 1, H, W)

    Example:
        >>> # Mask the horizon region for sky replacement
        >>> mask = create_horizon_mask((1024, 2048), horizon_y=0.5, height=0.3)
    """
    h, w = size
    if device is None:
        device = torch.device('cpu')

    # Create vertical gradient
    y = torch.linspace(0, 1, h, device=device, dtype=dtype)
    y_grid = y.view(-1, 1).expand(h, w)

    # Compute distance from horizon
    distance = torch.abs(y_grid - horizon_y)

    # Create mask with feathering
    half_height = height / 2
    mask = torch.zeros_like(y_grid)

    if feather > 0:
        # Smooth transition
        inner_edge = half_height - feather
        outer_edge = half_height + feather

        # Inside inner edge: full mask
        mask = torch.where(distance <= inner_edge, torch.ones_like(mask), mask)

        # Between inner and outer: gradient
        in_feather = (distance > inner_edge) & (distance < outer_edge)
        feather_value = 1.0 - (distance - inner_edge) / (2 * feather)
        mask = torch.where(in_feather, feather_value, mask)
    else:
        # Hard edge
        mask = (distance <= half_height).float()

    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


# Export all
__all__ = [
    'prepare_inpaint_mask',
    'gaussian_blur_mask',
    'expand_mask',
    'create_latent_noise_mask',
    'blend_latents',
    'apply_inpainting_conditioning',
    'create_circular_mask',
    'create_rectangle_mask',
    'create_horizon_mask'
]
