"""
Panoramic projection utilities for DiT360.

This module provides conversion functions between different panoramic projections:
- Equirectangular (standard 2:1 spherical projection)
- Cubemap (6 cube faces)
- Additional utilities for projection quality analysis

Author: DiT360 Team
License: Apache 2.0
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


def create_equirect_to_cube_grid(
    face_size: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> List[torch.Tensor]:
    """
    Pre-compute sampling grids for equirect->cubemap conversion.

    This function creates reusable grids that can be cached for efficiency.

    Args:
        face_size: Size of each cube face in pixels
        device: Target device for tensors
        dtype: Data type for coordinates

    Returns:
        List of 6 grids (front, back, right, left, top, bottom), each (face_size, face_size, 2)

    Example:
        >>> grids = create_equirect_to_cube_grid(512)
        >>> len(grids)
        6
    """
    if device is None:
        device = torch.device('cpu')

    # Generate face coordinates (u, v) in range [-1, 1]
    u = torch.linspace(-1, 1, face_size, device=device, dtype=dtype)
    v = torch.linspace(-1, 1, face_size, device=device, dtype=dtype)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

    grids = []

    # Define transformations for each cube face
    face_transforms = [
        # Front (Z+): x=u, y=-v, z=1
        lambda u, v: (u, -v, torch.ones_like(u)),
        # Back (Z-): x=-u, y=-v, z=-1
        lambda u, v: (-u, -v, -torch.ones_like(u)),
        # Right (X+): x=1, y=-v, z=-u
        lambda u, v: (torch.ones_like(u), -v, -u),
        # Left (X-): x=-1, y=-v, z=u
        lambda u, v: (-torch.ones_like(u), -v, u),
        # Top (Y+): x=u, y=1, z=v
        lambda u, v: (u, torch.ones_like(u), v),
        # Bottom (Y-): x=u, y=-1, z=-v
        lambda u, v: (u, -torch.ones_like(u), -v)
    ]

    for transform in face_transforms:
        x, y, z = transform(u_grid, v_grid)

        # Convert to spherical coordinates
        lon = torch.atan2(x, z)
        lat = torch.atan2(y, torch.sqrt(x**2 + z**2))

        # Normalize to equirectangular UV coordinates [0, 1]
        u_equirect = (lon + math.pi) / (2 * math.pi)
        v_equirect = (lat + math.pi / 2) / math.pi

        # Convert to grid_sample format [-1, 1]
        grid_u = u_equirect * 2 - 1
        grid_v = v_equirect * 2 - 1

        # Stack: (H, W, 2)
        grid = torch.stack([grid_u, grid_v], dim=-1)
        grids.append(grid)

    return grids


def equirect_to_cubemap_fast(
    equirect: torch.Tensor,
    face_size: int = 512,
    grids: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Fast equirectangular to cubemap conversion using pre-computed grids.

    Args:
        equirect: Equirectangular image (B, C, H, W)
        face_size: Size of each cube face
        grids: Pre-computed grids from create_equirect_to_cube_grid() (optional)

    Returns:
        Tuple of 6 face tensors (front, back, right, left, top, bottom)

    Example:
        >>> equirect = torch.randn(1, 3, 1024, 2048)
        >>> grids = create_equirect_to_cube_grid(512, device=equirect.device)
        >>> faces = equirect_to_cubemap_fast(equirect, 512, grids)
        >>> len(faces)
        6
    """
    B, C, H, W = equirect.shape
    device = equirect.device

    # Create grids if not provided
    if grids is None:
        grids = create_equirect_to_cube_grid(face_size, device=device, dtype=equirect.dtype)
    else:
        # Ensure grids are on correct device
        grids = [g.to(device) for g in grids]

    faces = []
    for grid in grids:
        # Expand grid for batch
        grid_batch = grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Sample from equirectangular image
        face = F.grid_sample(
            equirect,
            grid_batch,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        faces.append(face)

    return tuple(faces)


def cubemap_to_equirect_fast(
    faces: Tuple[torch.Tensor, ...],
    height: int = 1024,
    width: int = 2048,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Fast cubemap to equirectangular conversion using vectorized operations.

    Args:
        faces: Tuple of 6 face tensors (front, back, right, left, top, bottom)
        height: Output equirectangular height
        width: Output equirectangular width
        device: Target device (uses first face's device if None)

    Returns:
        Equirectangular image (B, C, H, W)

    Example:
        >>> faces = tuple(torch.randn(1, 3, 512, 512) for _ in range(6))
        >>> equirect = cubemap_to_equirect_fast(faces, 1024, 2048)
        >>> equirect.shape
        torch.Size([1, 3, 1024, 2048])
    """
    if len(faces) != 6:
        raise ValueError(f"Expected 6 cube faces, got {len(faces)}")

    front, back, right, left, top, bottom = faces
    B, C, face_size, _ = front.shape

    if device is None:
        device = front.device
    dtype = front.dtype

    # Create equirectangular coordinate grid
    u = torch.linspace(0, 1, width, device=device, dtype=dtype)
    v = torch.linspace(0, 1, height, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')  # (H, W)

    # Convert to spherical coordinates
    lon = u_grid * 2 * math.pi - math.pi  # [-π, π]
    lat = v_grid * math.pi - math.pi / 2  # [-π/2, π/2]

    # Convert to 3D Cartesian coordinates
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.sin(lon)
    y = torch.sin(lat)
    z = cos_lat * torch.cos(lon)

    # Stack all faces: (6, B, C, face_size, face_size)
    all_faces = torch.stack(faces, dim=0)

    # Determine which face to sample from for each pixel
    abs_x = torch.abs(x)
    abs_y = torch.abs(y)
    abs_z = torch.abs(z)

    # Initialize output
    output = torch.zeros(B, C, height, width, device=device, dtype=dtype)

    # Create masks for each face
    # Right face (X+): abs_x largest and x > 0
    mask_right = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
    # Left face (X-): abs_x largest and x < 0
    mask_left = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
    # Top face (Y+): abs_y largest and y > 0
    mask_top = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
    # Bottom face (Y-): abs_y largest and y < 0
    mask_bottom = (abs_y >= abs_x) & (abs_y >= abs_z) & (y < 0)
    # Front face (Z+): abs_z largest and z > 0
    mask_front = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
    # Back face (Z-): abs_z largest and z < 0
    mask_back = (abs_z >= abs_x) & (abs_z >= abs_y) & (z < 0)

    masks = [mask_front, mask_back, mask_right, mask_left, mask_top, mask_bottom]
    face_uvs = []

    # Compute UV coordinates for each face
    # Front (Z+): uc = x/z, vc = -y/z
    uc_front = x / (z + 1e-8)
    vc_front = -y / (z + 1e-8)
    face_uvs.append((uc_front, vc_front))

    # Back (Z-): uc = -x/(-z), vc = -y/(-z)
    uc_back = -x / (-z + 1e-8)
    vc_back = -y / (-z + 1e-8)
    face_uvs.append((uc_back, vc_back))

    # Right (X+): uc = -z/x, vc = -y/x
    uc_right = -z / (x + 1e-8)
    vc_right = -y / (x + 1e-8)
    face_uvs.append((uc_right, vc_right))

    # Left (X-): uc = z/(-x), vc = -y/(-x)
    uc_left = z / (-x + 1e-8)
    vc_left = -y / (-x + 1e-8)
    face_uvs.append((uc_left, vc_left))

    # Top (Y+): uc = x/y, vc = z/y
    uc_top = x / (y + 1e-8)
    vc_top = z / (y + 1e-8)
    face_uvs.append((uc_top, vc_top))

    # Bottom (Y-): uc = x/(-y), vc = -z/(-y)
    uc_bottom = x / (-y + 1e-8)
    vc_bottom = -z / (-y + 1e-8)
    face_uvs.append((uc_bottom, vc_bottom))

    # Sample from each face
    for face_idx, (face, mask, (uc, vc)) in enumerate(zip(faces, masks, face_uvs)):
        if not mask.any():
            continue

        # Convert UV from [-1, 1] to grid_sample format
        grid_u = uc
        grid_v = vc

        # Clamp to valid range
        grid_u = torch.clamp(grid_u, -1, 1)
        grid_v = torch.clamp(grid_v, -1, 1)

        # Create grid: (H, W, 2)
        grid = torch.stack([grid_u, grid_v], dim=-1)
        grid_batch = grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Sample from face
        sampled = F.grid_sample(
            face,
            grid_batch,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (B, C, H, W)

        # Apply mask
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        output = torch.where(mask_expanded, sampled, output)

    return output


def compute_projection_distortion(
    equirect: torch.Tensor,
    return_map: bool = False
) -> torch.Tensor:
    """
    Compute pixel area distortion in equirectangular projection.

    In equirectangular projection, pixels near the poles represent much smaller
    areas on the sphere than pixels near the equator. This function computes
    the distortion factor for each pixel.

    Args:
        equirect: Equirectangular image (B, C, H, W)
        return_map: If True, return full distortion map; if False, return average

    Returns:
        Distortion factor(s). Range [0, 1] where 1 = no distortion (equator)

    Example:
        >>> img = torch.randn(1, 3, 1024, 2048)
        >>> avg_distortion = compute_projection_distortion(img, return_map=False)
        >>> distortion_map = compute_projection_distortion(img, return_map=True)
    """
    B, C, H, W = equirect.shape
    device = equirect.device
    dtype = equirect.dtype

    # Create latitude grid
    v = torch.linspace(0, 1, H, device=device, dtype=dtype)
    lat = v * math.pi - math.pi / 2  # [-π/2, π/2]

    # Distortion factor = cos(latitude)
    # At equator (lat=0): cos(0) = 1 (no distortion)
    # At poles (lat=±π/2): cos(±π/2) = 0 (infinite distortion)
    distortion = torch.cos(lat)

    if return_map:
        # Expand to full image
        distortion_map = distortion.view(1, 1, H, 1).expand(B, C, H, W)
        return distortion_map
    else:
        # Return average distortion
        return distortion.mean()


def apply_distortion_weighted_loss(
    loss_map: torch.Tensor,
    equirect_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Weight a loss map by equirectangular distortion to give equal importance to all sphere regions.

    Args:
        loss_map: Per-pixel loss values (B, C, H, W) or (B, H, W) or (H, W)
        equirect_shape: (height, width) of the equirectangular projection

    Returns:
        Distortion-weighted loss (scalar)

    Example:
        >>> # Compute per-pixel loss
        >>> per_pixel_loss = F.mse_loss(pred, target, reduction='none')
        >>> # Weight by distortion
        >>> weighted_loss = apply_distortion_weighted_loss(per_pixel_loss, (1024, 2048))
    """
    H, W = equirect_shape
    device = loss_map.device
    dtype = loss_map.dtype

    # Create latitude weights
    v = torch.linspace(0, 1, H, device=device, dtype=dtype)
    lat = v * math.pi - math.pi / 2
    weights = torch.cos(lat)

    # Normalize weights so they sum to 1
    weights = weights / weights.sum()

    # Expand to match loss_map shape
    if loss_map.dim() == 2:  # (H, W)
        weights = weights.view(H, 1).expand(H, W)
    elif loss_map.dim() == 3:  # (B, H, W)
        weights = weights.view(1, H, 1).expand_as(loss_map)
    elif loss_map.dim() == 4:  # (B, C, H, W)
        weights = weights.view(1, 1, H, 1).expand_as(loss_map)
    else:
        raise ValueError(f"Unexpected loss_map shape: {loss_map.shape}")

    # Apply weights and sum
    weighted_loss = (loss_map * weights).sum()

    return weighted_loss


def split_cubemap_horizontal(faces: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Arrange cubemap faces in horizontal strip layout for visualization.

    Layout: [Right, Front, Left, Back, Top, Bottom]

    Args:
        faces: Tuple of 6 face tensors (front, back, right, left, top, bottom)

    Returns:
        Horizontal strip image (B, C, face_size, face_size*6)

    Example:
        >>> faces = tuple(torch.randn(1, 3, 512, 512) for _ in range(6))
        >>> strip = split_cubemap_horizontal(faces)
        >>> strip.shape
        torch.Size([1, 3, 512, 3072])
    """
    front, back, right, left, top, bottom = faces

    # Horizontal layout: right, front, left, back, top, bottom
    horizontal = torch.cat([right, front, left, back, top, bottom], dim=3)

    return horizontal


def split_cubemap_cross(faces: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Arrange cubemap faces in cross layout for visualization.

    Layout:
        [    Top    ]
        [Left Front Right Back]
        [  Bottom   ]

    Args:
        faces: Tuple of 6 face tensors (front, back, right, left, top, bottom)

    Returns:
        Cross layout image (B, C, face_size*3, face_size*4)

    Example:
        >>> faces = tuple(torch.randn(1, 3, 512, 512) for _ in range(6))
        >>> cross = split_cubemap_cross(faces)
        >>> cross.shape
        torch.Size([1, 3, 1536, 2048])
    """
    front, back, right, left, top, bottom = faces
    B, C, face_size, _ = front.shape
    device = front.device
    dtype = front.dtype

    # Create empty canvas
    output = torch.zeros(B, C, face_size * 3, face_size * 4, device=device, dtype=dtype)

    # Top row: [empty, top, empty, empty]
    output[:, :, 0:face_size, face_size:face_size*2] = top

    # Middle row: [left, front, right, back]
    output[:, :, face_size:face_size*2, 0:face_size] = left
    output[:, :, face_size:face_size*2, face_size:face_size*2] = front
    output[:, :, face_size:face_size*2, face_size*2:face_size*3] = right
    output[:, :, face_size:face_size*2, face_size*3:face_size*4] = back

    # Bottom row: [empty, bottom, empty, empty]
    output[:, :, face_size*2:face_size*3, face_size:face_size*2] = bottom

    return output


# Export all
__all__ = [
    'create_equirect_to_cube_grid',
    'equirect_to_cubemap_fast',
    'cubemap_to_equirect_fast',
    'compute_projection_distortion',
    'apply_distortion_weighted_loss',
    'split_cubemap_horizontal',
    'split_cubemap_cross'
]
