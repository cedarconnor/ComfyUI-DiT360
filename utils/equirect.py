"""
Equirectangular projection utilities for panoramic images

Functions for handling 360-degree equirectangular panoramas:
- Aspect ratio validation (2:1 requirement)
- Aspect ratio fixing (crop/pad/stretch)
- Edge blending for seamless wraparound
- Continuity checking
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Literal


def validate_aspect_ratio(width: int, height: int, tolerance: float = 0.01) -> bool:
    """
    Validate if dimensions are 2:1 ratio (equirectangular requirement)

    Equirectangular panoramas must be exactly 2:1 ratio (width:height) to properly
    map a sphere to a rectangular image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        tolerance: Acceptable deviation from 2:1 ratio (default 0.01 = 1%)

    Returns:
        True if aspect ratio is within tolerance of 2:1

    Example:
        >>> validate_aspect_ratio(2048, 1024)  # Exactly 2:1
        True
        >>> validate_aspect_ratio(2000, 1000)  # Exactly 2:1
        True
        >>> validate_aspect_ratio(1920, 1080)  # 16:9, not 2:1
        False
    """
    ratio = width / height
    return abs(ratio - 2.0) < tolerance


def get_equirect_dimensions(
    width: int,
    alignment: int = 16
) -> Tuple[int, int]:
    """
    Calculate valid equirectangular dimensions with proper alignment

    Args:
        width: Desired width in pixels
        alignment: Pixel alignment requirement (16 for FLUX)

    Returns:
        (width, height) tuple with correct 2:1 ratio and alignment

    Example:
        >>> get_equirect_dimensions(2048, alignment=16)
        (2048, 1024)
        >>> get_equirect_dimensions(2055, alignment=16)  # Auto-aligns
        (2048, 1024)
    """
    # Ensure width is multiple of alignment
    width = (width // alignment) * alignment

    # Height is exactly half for 2:1 ratio
    height = width // 2

    return width, height


def fix_aspect_ratio(
    image: torch.Tensor,
    mode: Literal["pad", "crop", "stretch"] = "pad",
    target_width: int = None
) -> torch.Tensor:
    """
    Fix image to 2:1 aspect ratio using specified mode

    Args:
        image: Input image tensor in ComfyUI format (B, H, W, C)
        mode: How to fix ratio:
            - 'pad': Add black bars top/bottom to reach 2:1 (preserves content)
            - 'crop': Center crop to 2:1 (loses content at top/bottom)
            - 'stretch': Resize to 2:1 (distorts content vertically)
        target_width: Optional target width after fixing. If None, keeps current width

    Returns:
        Fixed image with 2:1 aspect ratio

    Example:
        >>> image = torch.rand(1, 1080, 1920, 3)  # 16:9 image
        >>> fixed = fix_aspect_ratio(image, mode="pad")
        >>> fixed.shape
        torch.Size([1, 960, 1920, 3])  # Now 2:1 ratio
    """
    B, H, W, C = image.shape

    # Check if already 2:1
    if validate_aspect_ratio(W, H):
        if target_width and W != target_width:
            target_height = target_width // 2
            return F.interpolate(
                image.permute(0, 3, 1, 2),  # (B, C, H, W)
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        return image

    # Calculate target dimensions
    if target_width:
        target_height = target_width // 2
    else:
        # Use current width, calculate height for 2:1
        target_width = W
        target_height = W // 2

    if mode == "pad":
        # Add black bars top/bottom
        if H < target_height:
            # Need to add padding
            pad_total = target_height - H
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            # Padding format: (left, right, top, bottom)
            result = F.pad(
                image.permute(0, 3, 1, 2),  # (B, C, H, W)
                (0, 0, pad_top, pad_bottom),
                mode='constant',
                value=0
            ).permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        else:
            # Height too large, crop first then pad if needed
            crop_start = (H - target_height) // 2
            result = image[:, crop_start:crop_start+target_height, :, :]

        return result

    elif mode == "crop":
        # Center crop to 2:1
        if H > target_height:
            # Crop height
            crop_start = (H - target_height) // 2
            result = image[:, crop_start:crop_start+target_height, :, :]
        else:
            # Height too small, need to crop width instead and resize
            crop_width = H * 2
            crop_start_w = (W - crop_width) // 2
            result = image[:, :, crop_start_w:crop_start_w+crop_width, :]
            # Resize to target
            result = F.interpolate(
                result.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)

        return result

    elif mode == "stretch":
        # Resize to exact 2:1 (distorts content)
        return F.interpolate(
            image.permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)

    else:
        raise ValueError(f"Unknown fix mode: {mode}. Use 'pad', 'crop', or 'stretch'")


def blend_edges(
    image: torch.Tensor,
    blend_width: int = 10,
    mode: Literal["linear", "cosine", "smooth"] = "cosine"
) -> torch.Tensor:
    """
    Blend left and right edges for seamless wraparound

    Creates a smooth transition between the left and right edges of a panorama
    to ensure seamless wraparound when viewed in 360° viewers.

    Args:
        image: Input image in ComfyUI format (B, H, W, C)
        blend_width: Width of blend region in pixels (default 10)
        mode: Blending function:
            - 'linear': Simple linear interpolation
            - 'cosine': Smooth cosine interpolation (recommended)
            - 'smooth': Quadratic smooth interpolation

    Returns:
        Image with blended edges

    Example:
        >>> panorama = torch.rand(1, 1024, 2048, 3)
        >>> blended = blend_edges(panorama, blend_width=20, mode="cosine")
        >>> # Left and right edges now transition smoothly
    """
    B, H, W, C = image.shape

    # Validate blend width
    if blend_width <= 0 or blend_width >= W // 2:
        print(f"Warning: blend_width {blend_width} invalid, must be 0 < width < {W//2}")
        return image

    # Extract edge regions
    left_edge = image[:, :, :blend_width, :]
    right_edge = image[:, :, -blend_width:, :]

    # Create blend weights based on mode
    if mode == "linear":
        # Simple linear ramp: 0 -> 1
        weights = torch.linspace(0, 1, blend_width, device=image.device)

    elif mode == "cosine":
        # Smooth cosine curve: 0 -> 1
        # Uses (1 - cos(πx)) / 2 for smooth S-curve
        t = torch.linspace(0, math.pi, blend_width, device=image.device)
        weights = (1 - torch.cos(t)) / 2

    elif mode == "smooth":
        # Quadratic smooth: x²
        weights = torch.linspace(0, 1, blend_width, device=image.device) ** 2

    else:
        raise ValueError(f"Unknown blend mode: {mode}. Use 'linear', 'cosine', or 'smooth'")

    # Reshape weights for broadcasting: (1, 1, blend_width, 1)
    weights = weights.view(1, 1, -1, 1)

    # Blend edges using weighted average
    # Left edge: transitions from left_edge to right_edge
    # Right edge: transitions from right_edge to left_edge
    blended_left = left_edge * (1 - weights) + right_edge * weights
    blended_right = right_edge * (1 - weights) + left_edge * weights

    # Apply blending to image
    result = image.clone()
    result[:, :, :blend_width, :] = blended_left
    result[:, :, -blend_width:, :] = blended_right

    return result


def check_edge_continuity(
    image: torch.Tensor,
    threshold: float = 0.05
) -> bool:
    """
    Check if left and right edges are continuous (for validation)

    Measures the average pixel difference between the leftmost and rightmost
    columns to determine if the panorama wraps seamlessly.

    Args:
        image: Input image in ComfyUI format (B, H, W, C)
        threshold: Maximum allowed difference (0-1 scale). Default 0.05 = 5%

    Returns:
        True if edges are continuous within threshold

    Example:
        >>> panorama = torch.rand(1, 1024, 2048, 3)
        >>> panorama = blend_edges(panorama)
        >>> check_edge_continuity(panorama)
        True
        >>> # Without blending, likely returns False
    """
    # Get leftmost and rightmost columns
    left_edge = image[:, :, 0, :]   # (B, H, C)
    right_edge = image[:, :, -1, :]  # (B, H, C)

    # Calculate mean absolute difference
    diff = torch.abs(left_edge - right_edge).mean()

    return diff.item() < threshold


def equirect_to_cubemap(
    image: torch.Tensor,
    face_size: int = 512
) -> torch.Tensor:
    """
    Convert equirectangular panorama to cubemap (6 faces)

    NOTE: This is a placeholder for Phase 7 (advanced features)

    Args:
        image: Equirectangular image (B, H, W, C)
        face_size: Size of each cube face in pixels

    Returns:
        Cubemap faces tensor (6, face_size, face_size, C)
    """
    # TODO: Implement in Phase 7 for cube loss calculation
    raise NotImplementedError("Cubemap conversion coming in Phase 7")


def calculate_yaw_consistency(
    latent: torch.Tensor,
    shift_pixels: int = None
) -> float:
    """
    Calculate rotational consistency metric for yaw loss

    NOTE: This is a placeholder for Phase 7 (advanced features)

    Args:
        latent: Latent tensor to check (B, C, H, W)
        shift_pixels: Number of pixels to shift for comparison

    Returns:
        Consistency score (lower is better)
    """
    # TODO: Implement in Phase 7 for yaw loss
    raise NotImplementedError("Yaw consistency calculation coming in Phase 7")
