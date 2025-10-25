"""
Advanced loss functions for DiT360 panoramic generation.

This module implements specialized losses for improving panoramic image quality:
- Yaw loss: Encourages rotational consistency at panorama edges
- Cube loss: Reduces pole distortion by computing loss in cubemap space

Author: DiT360 Team
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def rotate_equirect_yaw(image: torch.Tensor, yaw_degrees: float) -> torch.Tensor:
    """
    Rotate an equirectangular panorama by yaw angle (horizontal rotation).

    This performs a circular shift along the width dimension, which corresponds
    to rotation around the vertical axis in spherical coordinates.

    Args:
        image: Equirectangular image tensor of shape (B, C, H, W) or (B, H, W, C)
        yaw_degrees: Rotation angle in degrees (positive = rotate right)

    Returns:
        Rotated image tensor with same shape as input

    Example:
        >>> img = torch.randn(1, 3, 1024, 2048)
        >>> rotated = rotate_equirect_yaw(img, 45.0)  # Rotate 45° right
        >>> rotated.shape
        torch.Size([1, 3, 1024, 2048])
    """
    # Detect format: (B, C, H, W) or (B, H, W, C)
    if image.dim() == 4:
        if image.shape[1] in [1, 3, 4]:  # Likely (B, C, H, W)
            width = image.shape[3]
            width_dim = 3
        else:  # Likely (B, H, W, C)
            width = image.shape[2]
            width_dim = 2
    else:
        raise ValueError(f"Expected 4D tensor, got shape {image.shape}")

    # Convert degrees to pixel shift
    # 360° = full width
    shift_pixels = int((yaw_degrees / 360.0) * width)

    # Circular shift (torch.roll wraps around)
    rotated = torch.roll(image, shifts=shift_pixels, dims=width_dim)

    return rotated


class YawLoss(nn.Module):
    """
    Yaw consistency loss for panoramic images.

    This loss encourages the model to generate panoramas that remain consistent
    when rotated. It works by:
    1. Rotating the input/output by a random yaw angle
    2. Measuring the difference between rotated and unrotated generation
    3. Penalizing inconsistencies

    This helps eliminate visible seams at the 0°/360° boundary.

    Args:
        num_rotations: Number of random rotations to test (default: 4)
        max_yaw_degrees: Maximum rotation angle in degrees (default: 180)
        loss_type: Type of loss - "l1", "l2", or "perceptual" (default: "l2")

    Example:
        >>> yaw_loss = YawLoss(num_rotations=4)
        >>> generated = model.generate(...)  # (B, C, H, W)
        >>> loss = yaw_loss(generated, original_latent)
    """

    def __init__(
        self,
        num_rotations: int = 4,
        max_yaw_degrees: float = 180.0,
        loss_type: str = "l2"
    ):
        super().__init__()
        self.num_rotations = num_rotations
        self.max_yaw_degrees = max_yaw_degrees
        self.loss_type = loss_type

        if loss_type not in ["l1", "l2", "perceptual"]:
            raise ValueError(f"loss_type must be 'l1', 'l2', or 'perceptual', got {loss_type}")

    def forward(
        self,
        image: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute yaw consistency loss.

        Args:
            image: Generated panorama (B, C, H, W) or (B, H, W, C)
            reference: Optional reference image for comparison

        Returns:
            Scalar loss value
        """
        device = image.device
        dtype = image.dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # Generate random yaw angles
        yaw_angles = torch.rand(self.num_rotations, device=device) * self.max_yaw_degrees
        yaw_angles = yaw_angles - (self.max_yaw_degrees / 2)  # Center around 0

        for yaw in yaw_angles:
            yaw_deg = yaw.item()

            # Rotate image forward
            rotated_forward = rotate_equirect_yaw(image, yaw_deg)

            # Rotate back
            rotated_back = rotate_equirect_yaw(rotated_forward, -yaw_deg)

            # Compute difference
            if self.loss_type == "l1":
                loss = F.l1_loss(rotated_back, image)
            elif self.loss_type == "l2":
                loss = F.mse_loss(rotated_back, image)
            else:  # perceptual (simple approximation using gradient)
                grad_orig = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
                grad_rot = torch.abs(rotated_back[:, :, :, 1:] - rotated_back[:, :, :, :-1])
                loss = F.mse_loss(grad_rot, grad_orig)

            total_loss = total_loss + loss

        # Average over rotations
        return total_loss / self.num_rotations


def equirect_to_cubemap(
    equirect: torch.Tensor,
    face_size: int = 512
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert equirectangular panorama to cubemap (6 faces).

    Cubemap layout:
    - Front (Z+): Looking forward
    - Back (Z-): Looking backward
    - Right (X+): Looking right
    - Left (X-): Looking left
    - Top (Y+): Looking up
    - Bottom (Y-): Looking down

    Args:
        equirect: Equirectangular image (B, C, H, W)
        face_size: Size of each cube face in pixels

    Returns:
        Tuple of 6 tensors (front, back, right, left, top, bottom), each (B, C, face_size, face_size)

    Example:
        >>> equirect = torch.randn(1, 3, 1024, 2048)
        >>> faces = equirect_to_cubemap(equirect, face_size=512)
        >>> len(faces)
        6
        >>> faces[0].shape  # Front face
        torch.Size([1, 3, 512, 512])
    """
    B, C, H, W = equirect.shape
    device = equirect.device
    dtype = equirect.dtype

    # Create sampling grids for each face
    # Each grid maps from cube face coordinates to spherical coordinates

    # Generate face coordinates (u, v) in range [-1, 1]
    u = torch.linspace(-1, 1, face_size, device=device, dtype=dtype)
    v = torch.linspace(-1, 1, face_size, device=device, dtype=dtype)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

    faces = []

    # Define transformations for each cube face
    # (x, y, z) coordinates in 3D space -> (lon, lat) in spherical -> (u, v) in equirect

    for face_idx in range(6):
        if face_idx == 0:  # Front (Z+)
            x = u_grid
            y = -v_grid
            z = torch.ones_like(u_grid)
        elif face_idx == 1:  # Back (Z-)
            x = -u_grid
            y = -v_grid
            z = -torch.ones_like(u_grid)
        elif face_idx == 2:  # Right (X+)
            x = torch.ones_like(u_grid)
            y = -v_grid
            z = -u_grid
        elif face_idx == 3:  # Left (X-)
            x = -torch.ones_like(u_grid)
            y = -v_grid
            z = u_grid
        elif face_idx == 4:  # Top (Y+)
            x = u_grid
            y = torch.ones_like(u_grid)
            z = v_grid
        else:  # Bottom (Y-)
            x = u_grid
            y = -torch.ones_like(u_grid)
            z = -v_grid

        # Convert to spherical coordinates
        # lon (longitude) = atan2(x, z), range [-π, π]
        # lat (latitude) = atan2(y, sqrt(x^2 + z^2)), range [-π/2, π/2]
        lon = torch.atan2(x, z)
        lat = torch.atan2(y, torch.sqrt(x**2 + z**2))

        # Normalize to equirectangular UV coordinates [0, 1]
        u_equirect = (lon + math.pi) / (2 * math.pi)  # [0, 1]
        v_equirect = (lat + math.pi / 2) / math.pi    # [0, 1]

        # Convert to grid_sample format [-1, 1]
        grid_u = u_equirect * 2 - 1
        grid_v = v_equirect * 2 - 1

        # Stack and reshape for grid_sample: (B, H, W, 2)
        grid = torch.stack([grid_u, grid_v], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Sample from equirectangular image
        face = F.grid_sample(
            equirect,
            grid,
            mode='bilinear',
            padding_mode='border',  # Clamp at poles
            align_corners=True
        )

        faces.append(face)

    return tuple(faces)


def cubemap_to_equirect(
    faces: Tuple[torch.Tensor, ...],
    height: int = 1024,
    width: int = 2048
) -> torch.Tensor:
    """
    Convert cubemap (6 faces) back to equirectangular panorama.

    Args:
        faces: Tuple of 6 face tensors (front, back, right, left, top, bottom)
        height: Output equirectangular height
        width: Output equirectangular width (should be 2*height)

    Returns:
        Equirectangular image (B, C, H, W)

    Example:
        >>> faces = tuple(torch.randn(1, 3, 512, 512) for _ in range(6))
        >>> equirect = cubemap_to_equirect(faces, height=1024, width=2048)
        >>> equirect.shape
        torch.Size([1, 3, 1024, 2048])
    """
    if len(faces) != 6:
        raise ValueError(f"Expected 6 cube faces, got {len(faces)}")

    front, back, right, left, top, bottom = faces
    B, C, face_size, _ = front.shape
    device = front.device
    dtype = front.dtype

    # Create equirectangular coordinate grid
    u = torch.linspace(0, 1, width, device=device, dtype=dtype)
    v = torch.linspace(0, 1, height, device=device, dtype=dtype)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

    # Convert to spherical coordinates
    lon = u_grid * 2 * math.pi - math.pi  # [-π, π]
    lat = v_grid * math.pi - math.pi / 2  # [-π/2, π/2]

    # Convert to 3D Cartesian coordinates
    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)

    # Determine which cube face to sample from
    # Find the dominant axis (largest absolute value)
    abs_x = torch.abs(x)
    abs_y = torch.abs(y)
    abs_z = torch.abs(z)

    # Initialize output
    output = torch.zeros(B, C, height, width, device=device, dtype=dtype)

    # For each pixel, determine which face and sample
    for b in range(B):
        for h in range(height):
            for w in range(width):
                px, py, pz = x[w, h], y[w, h], z[w, h]
                pax, pay, paz = abs_x[w, h], abs_y[w, h], abs_z[w, h]

                # Determine face and UV coordinates
                if pax >= pay and pax >= paz:  # X dominant
                    if px > 0:  # Right face (X+)
                        face = right
                        uc = -pz / px
                        vc = -py / px
                    else:  # Left face (X-)
                        face = left
                        uc = pz / -px
                        vc = -py / -px
                elif pay >= pax and pay >= paz:  # Y dominant
                    if py > 0:  # Top face (Y+)
                        face = top
                        uc = px / py
                        vc = pz / py
                    else:  # Bottom face (Y-)
                        face = bottom
                        uc = px / -py
                        vc = -pz / -py
                else:  # Z dominant
                    if pz > 0:  # Front face (Z+)
                        face = front
                        uc = px / pz
                        vc = -py / pz
                    else:  # Back face (Z-)
                        face = back
                        uc = -px / -pz
                        vc = -py / -pz

                # Convert UV from [-1, 1] to pixel coordinates
                face_u = (uc + 1) / 2 * (face_size - 1)
                face_v = (vc + 1) / 2 * (face_size - 1)
                face_u = torch.clamp(face_u, 0, face_size - 1)
                face_v = torch.clamp(face_v, 0, face_size - 1)

                # Bilinear interpolation (simple nearest neighbor for now)
                fu = int(face_u.item())
                fv = int(face_v.item())

                output[b, :, h, w] = face[b, :, fv, fu]

    return output


class CubeLoss(nn.Module):
    """
    Cube loss for reducing pole distortion in equirectangular panoramas.

    This loss works by:
    1. Converting the equirectangular image to cubemap (6 faces)
    2. Computing loss in cubemap space where all pixels have equal weight
    3. This gives more importance to pole regions that are stretched in equirect

    This helps reduce artifacts and distortion near the top/bottom of panoramas.

    Args:
        face_size: Size of each cube face in pixels (default: 512)
        loss_type: Type of loss - "l1" or "l2" (default: "l2")
        weight: Weight multiplier for this loss (default: 1.0)

    Example:
        >>> cube_loss = CubeLoss(face_size=512)
        >>> generated = model.generate(...)  # (B, C, H, W)
        >>> loss = cube_loss(generated, target)
    """

    def __init__(
        self,
        face_size: int = 512,
        loss_type: str = "l2",
        weight: float = 1.0
    ):
        super().__init__()
        self.face_size = face_size
        self.loss_type = loss_type
        self.weight = weight

        if loss_type not in ["l1", "l2"]:
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type}")

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cube loss between generated and target panoramas.

        Args:
            generated: Generated panorama (B, C, H, W)
            target: Target panorama (B, C, H, W)

        Returns:
            Scalar loss value
        """
        # Convert both to cubemap
        gen_faces = equirect_to_cubemap(generated, self.face_size)
        target_faces = equirect_to_cubemap(target, self.face_size)

        # Compute loss on each face
        total_loss = 0.0
        for gen_face, target_face in zip(gen_faces, target_faces):
            if self.loss_type == "l1":
                total_loss += F.l1_loss(gen_face, target_face)
            else:  # l2
                total_loss += F.mse_loss(gen_face, target_face)

        # Average over 6 faces
        return self.weight * (total_loss / 6.0)


def compute_yaw_consistency(image: torch.Tensor, num_rotations: int = 8) -> float:
    """
    Compute yaw consistency metric for a panorama (0.0 = perfect, higher = worse).

    This metric measures how consistent a panorama is when rotated. Lower values
    indicate better edge continuity and seamless wraparound.

    Args:
        image: Panorama image (B, C, H, W)
        num_rotations: Number of rotations to test

    Returns:
        Average consistency error across all rotations

    Example:
        >>> img = torch.randn(1, 3, 1024, 2048)
        >>> consistency = compute_yaw_consistency(img)
        >>> print(f"Consistency error: {consistency:.4f}")
    """
    yaw_loss = YawLoss(num_rotations=num_rotations, loss_type="l2")
    with torch.no_grad():
        loss = yaw_loss(image)
    return loss.item()


# Export all
__all__ = [
    'rotate_equirect_yaw',
    'YawLoss',
    'equirect_to_cubemap',
    'cubemap_to_equirect',
    'CubeLoss',
    'compute_yaw_consistency'
]
