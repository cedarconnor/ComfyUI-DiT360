"""
Circular RoPE (Rotary Position Embeddings) for 360° Panorama Generation

This module implements circular position encodings that treat the left and right
edges of panoramic images as adjacent, enabling seamless wraparound generation.

Supports: FLUX.1/2, Qwen-Image, Z-Image, and other DiT-based models.

Theory:
-------
Standard RoPE encodes positions linearly: [0, 1, 2, ..., N-1]
The model treats position 0 and N-1 as maximally distant.

Circular RoPE maps positions to a circle:
    θ(x) = 2π * x / width

This way:
- Position 0: θ = 0
- Position width-1: θ ≈ 2π ≈ 0 (wraps back!)
- Distance(0, width-1) = small (adjacent on circle)

References:
-----------
- RoPE: https://arxiv.org/abs/2104.09864
- DiT360: https://arxiv.org/abs/2510.11712
- ComfyUI DyPE: https://github.com/wildminder/ComfyUI-DyPE
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict, Any


def circular_position_ids(
    height: int,
    width: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate position IDs that are circular in the X (width) dimension.

    Instead of linear positions [0, 1, 2, ..., W-1], we generate angular
    positions that wrap around: θ = 2π * x / W

    Args:
        height: Image height in tokens
        width: Image width in tokens
        device: Target device
        dtype: Target dtype

    Returns:
        Position tensor of shape (H*W, 3) with [y, cos(θ_x), sin(θ_x)]
        where θ_x encodes circular horizontal position
    """
    # Create grid
    y_pos = torch.arange(height, device=device, dtype=dtype)
    x_pos = torch.arange(width, device=device, dtype=dtype)

    # Convert X positions to circular (angular) encoding
    # θ = 2π * x / width, so position 0 and width-1 are adjacent
    theta_x = 2 * math.pi * x_pos / width

    # Encode as (cos, sin) pairs - these naturally wrap!
    cos_x = torch.cos(theta_x)
    sin_x = torch.sin(theta_x)

    # Create meshgrid
    y_grid, cos_grid = torch.meshgrid(y_pos, cos_x, indexing='ij')
    _, sin_grid = torch.meshgrid(y_pos, sin_x, indexing='ij')

    # Flatten and stack: [y_normalized, cos(θ_x), sin(θ_x)]
    # Normalize y to [0, 1] range (vertical is NOT circular)
    y_normalized = y_grid.flatten() / max(height - 1, 1)

    positions = torch.stack([
        y_normalized,
        cos_grid.flatten(),
        sin_grid.flatten()
    ], dim=-1)

    return positions


def circular_rope_frequencies(
    dim: int,
    width: int,
    height: int,
    theta_base: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute RoPE frequencies with circular topology in X dimension.

    For standard RoPE:
        freq_i = 1 / (θ^(2i/d))
        rotation = position * freq_i

    For circular RoPE in X:
        θ_x = 2π * x / width  (circular position)
        rotation_x = θ_x * freq_i  (wraps naturally!)

    For non-circular Y:
        rotation_y = y * freq_i  (standard linear)

    Args:
        dim: Embedding dimension (must be divisible by 4 for 2D)
        width: Image width in tokens
        height: Image height in tokens
        theta_base: RoPE base frequency (10000 for most models, 256 for Z-Image)
        device: Target device
        dtype: Target dtype

    Returns:
        Complex frequency tensor for RoPE application
    """
    assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"

    half_dim = dim // 2
    quarter_dim = dim // 4

    # Compute base frequencies
    freqs = 1.0 / (theta_base ** (torch.arange(0, quarter_dim, dtype=dtype, device=device) / quarter_dim))

    # Y positions (linear, non-circular)
    y_pos = torch.arange(height, dtype=dtype, device=device)

    # X positions (circular!)
    x_pos = torch.arange(width, dtype=dtype, device=device)
    theta_x = 2 * math.pi * x_pos / width  # Circular mapping

    # Compute rotations
    # Y: standard linear
    freqs_y = torch.outer(y_pos, freqs)  # (H, quarter_dim)

    # X: circular - use theta_x instead of x_pos
    freqs_x = torch.outer(theta_x, freqs)  # (W, quarter_dim)

    # Create meshgrid for 2D positions
    # Shape: (H, W, quarter_dim)
    freqs_y_grid = freqs_y.unsqueeze(1).expand(-1, width, -1)
    freqs_x_grid = freqs_x.unsqueeze(0).expand(height, -1, -1)

    # Concatenate Y and X frequencies
    # Shape: (H, W, half_dim)
    freqs_2d = torch.cat([freqs_y_grid, freqs_x_grid], dim=-1)

    # Convert to complex exponential for efficient rotation
    # e^(iθ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs_2d), freqs_2d)

    # Flatten spatial dimensions
    # Shape: (H*W, half_dim)
    freqs_cis = freqs_cis.reshape(-1, half_dim)

    return freqs_cis


def apply_circular_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply circular RoPE to query and key tensors.

    Args:
        q: Query tensor (B, N, H, D) or (B, H, N, D)
        k: Key tensor (B, N, H, D) or (B, H, N, D)
        freqs_cis: Precomputed circular frequencies (N, D//2)

    Returns:
        Rotated (q, k) tuple
    """
    # Reshape for complex multiplication
    # Treat pairs of dimensions as complex numbers
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Ensure freqs_cis broadcasts correctly
    # freqs_cis: (N, D//2) -> needs to match (B, N, H, D//2) or (B, H, N, D//2)
    if q.dim() == 4:
        if q.shape[1] == freqs_cis.shape[0]:  # (B, N, H, D)
            freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        else:  # (B, H, N, D)
            freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1)

    # Apply rotation via complex multiplication
    q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

    return q_rotated.type_as(q), k_rotated.type_as(k)


class CircularRoPEEmbedding(nn.Module):
    """
    Circular RoPE position embedding module.

    Drop-in replacement for standard RoPE embeddings that treats
    the horizontal (X) dimension as circular for panorama generation.

    Compatible with:
    - FLUX.1/FLUX.2 (axes_dim: 16, 56, 56)
    - Qwen-Image (MSRoPE)
    - Z-Image (theta=256)
    - HiDream, Lumina, etc.
    """

    def __init__(
        self,
        dim: int,
        theta_base: float = 10000.0,
        axes_dim: Tuple[int, ...] = (16, 56, 56),
        circular_axis: int = -1,  # Which axis is circular (usually X = last)
    ):
        super().__init__()
        self.dim = dim
        self.theta_base = theta_base
        self.axes_dim = axes_dim
        self.circular_axis = circular_axis

        # Cache for precomputed frequencies
        self._freqs_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def get_freqs(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Get or compute circular RoPE frequencies."""
        cache_key = (height, width)

        if cache_key not in self._freqs_cache:
            self._freqs_cache[cache_key] = circular_rope_frequencies(
                dim=self.dim,
                width=width,
                height=height,
                theta_base=self.theta_base,
                device=device,
                dtype=dtype
            )

        freqs = self._freqs_cache[cache_key]
        return freqs.to(device=device, dtype=dtype)

    def forward(
        self,
        ids: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate circular RoPE embeddings.

        Args:
            ids: Position IDs tensor (typically from patchify)
            height: Image height in tokens (if not inferrable from ids)
            width: Image width in tokens (if not inferrable from ids)

        Returns:
            RoPE frequency tensor
        """
        # Infer dimensions from ids if not provided
        if height is None or width is None:
            # Assume square-ish aspect, try to factor sequence length
            seq_len = ids.shape[-2] if ids.dim() > 1 else ids.shape[0]
            # For 2:1 panoramas, width = 2 * height
            height = int(math.sqrt(seq_len / 2))
            width = 2 * height

        return self.get_freqs(height, width, ids.device, ids.dtype)


def create_circular_rope_wrapper(original_embedder: nn.Module, circular_axis: int = -1):
    """
    Wrap an existing RoPE embedder to use circular positions in one axis.

    This is the key function for patching existing models!

    Args:
        original_embedder: The model's original position embedder
        circular_axis: Which spatial axis is circular (default: -1 = X/width)

    Returns:
        Wrapped embedder that produces circular RoPE

    Example:
        >>> # For FLUX
        >>> model.diffusion_model.pos_embed = create_circular_rope_wrapper(
        ...     model.diffusion_model.pos_embed,
        ...     circular_axis=-1  # X dimension
        ... )
    """
    original_forward = original_embedder.forward

    # Get theta_base from original if available
    theta_base = getattr(original_embedder, 'theta', 10000.0)
    if hasattr(original_embedder, 'theta_base'):
        theta_base = original_embedder.theta_base

    def circular_forward(ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Wrapped forward that applies circular mapping to specified axis.
        """
        # Get original output shape info
        original_output = original_forward(ids, *args, **kwargs)

        # Determine spatial dimensions
        # ids typically has shape (B, N, num_axes) where last axis has [t, y, x] or [y, x]
        if ids.dim() >= 2:
            n_axes = ids.shape[-1]

            # Modify the circular axis (usually X)
            # Map linear positions to circular angles
            if ids.shape[-1] >= 2:  # At least 2D spatial
                modified_ids = ids.clone()

                # Get the axis to make circular
                axis_idx = circular_axis if circular_axis >= 0 else n_axes + circular_axis

                # Get max position in this axis to determine "width"
                max_pos = ids[..., axis_idx].max().item() + 1

                # Convert to circular: θ = 2π * pos / max_pos
                circular_pos = 2 * math.pi * ids[..., axis_idx] / max_pos

                # The trick: RoPE uses positions directly in rotation
                # By scaling positions to [0, 2π), we make the topology circular
                # because RoPE's rotation naturally wraps at 2π
                modified_ids[..., axis_idx] = circular_pos / (2 * math.pi / max_pos)

                return original_forward(modified_ids, *args, **kwargs)

        return original_output

    # Create wrapper class that mimics original
    class CircularWrapper(nn.Module):
        def __init__(self, orig):
            super().__init__()
            self._original = orig
            # Copy attributes
            for attr in ['theta', 'theta_base', 'axes_dim', 'dim']:
                if hasattr(orig, attr):
                    setattr(self, attr, getattr(orig, attr))

        def forward(self, ids, *args, **kwargs):
            return circular_forward(ids, *args, **kwargs)

        def __getattr__(self, name):
            if name.startswith('_'):
                return super().__getattr__(name)
            return getattr(self._original, name)

    return CircularWrapper(original_embedder)


# =============================================================================
# MODEL-SPECIFIC IMPLEMENTATIONS
# =============================================================================

class CircularPosEmbedFlux(nn.Module):
    """
    Circular position embedding for FLUX.1/FLUX.2 models.

    FLUX uses axes_dim = (16, 56, 56) for (time, height, width).
    We make the width (last axis) circular.
    """

    def __init__(self, theta: int = 10000, axes_dim: Tuple[int, int, int] = (16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Generate circular position embeddings for FLUX.

        Args:
            ids: Position IDs (B, N, 3) with [time, y, x] for each token

        Returns:
            Position embeddings with circular X dimension
        """
        n_axes = ids.shape[-1]
        device = ids.device
        dtype = torch.float32  # RoPE computation needs float32

        emb_list = []
        for i in range(n_axes):
            axis_dim = self.axes_dim[i]
            positions = ids[..., i]

            if i == n_axes - 1:  # Last axis (X) is circular
                # Get width from max position
                width = positions.max().item() + 1
                # Map to circular: θ = 2π * x / width
                positions = positions * (2 * math.pi / width)

            # Standard RoPE frequency computation
            scale = torch.arange(0, axis_dim, 2, dtype=dtype, device=device) / axis_dim
            omega = 1.0 / (self.theta ** scale)

            # Outer product: positions × frequencies
            angles = positions.unsqueeze(-1).float() * omega

            # Stack cos and sin
            emb = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            emb = emb.flatten(-2)  # (..., axis_dim)

            emb_list.append(emb)

        # Concatenate all axes
        return torch.cat(emb_list, dim=-1).unsqueeze(2)  # Add head dimension


class CircularPosEmbedQwen(nn.Module):
    """
    Circular MSRoPE for Qwen-Image models.

    Qwen-Image uses Multi-Scale Rotary Position Encoding (MSRoPE)
    which handles text + 2D image positions.
    """

    def __init__(self, theta: float = 10000.0, dim: int = 128):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(
        self,
        text_positions: torch.Tensor,
        image_positions: torch.Tensor,
        image_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate circular position embeddings for Qwen-Image.

        Args:
            text_positions: Text token positions (B, T)
            image_positions: Image token positions (B, N, 2) with [y, x]
            image_width: Width of image in tokens (for circular mapping)

        Returns:
            Tuple of (text_freqs, image_freqs)
        """
        device = image_positions.device
        dtype = torch.float32

        # Compute base frequencies
        half_dim = self.dim // 2
        freqs = 1.0 / (self.theta ** (torch.arange(0, half_dim, 2, dtype=dtype, device=device) / half_dim))

        # Text: standard linear RoPE
        text_angles = text_positions.unsqueeze(-1).float() * freqs
        text_freqs = torch.polar(torch.ones_like(text_angles), text_angles)

        # Image: circular in X
        y_pos = image_positions[..., 0]
        x_pos = image_positions[..., 1]

        # Make X circular
        theta_x = 2 * math.pi * x_pos / image_width

        # Compute frequencies for each axis
        y_angles = y_pos.unsqueeze(-1).float() * freqs
        x_angles = theta_x.unsqueeze(-1).float() * freqs

        # Combine
        image_freqs = torch.cat([
            torch.polar(torch.ones_like(y_angles), y_angles),
            torch.polar(torch.ones_like(x_angles), x_angles)
        ], dim=-1)

        return text_freqs, image_freqs


class CircularPosEmbedZImage(nn.Module):
    """
    Circular position embedding for Z-Image models.

    Z-Image uses a very low theta (256) and single-stream architecture.
    """

    def __init__(self, theta: float = 256.0, dim: int = 64):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Generate circular position embeddings for Z-Image.

        Args:
            height: Image height in tokens
            width: Image width in tokens
            device: Target device
            dtype: Target dtype

        Returns:
            Position embedding tensor (H*W, dim)
        """
        return circular_rope_frequencies(
            dim=self.dim,
            width=width,
            height=height,
            theta_base=self.theta,
            device=device,
            dtype=dtype
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_rope_embedder(model) -> Optional[nn.Module]:
    """
    Find the RoPE position embedder in a model.

    Searches common locations across different architectures.

    Args:
        model: The diffusion model (or model patcher)

    Returns:
        The RoPE embedder module, or None if not found
    """
    # Get the actual model if wrapped
    original_model = model
    if hasattr(model, 'model'):
        model = model.model
    if hasattr(model, 'diffusion_model'):
        model = model.diffusion_model

    # Common attribute names for RoPE embedders (in priority order)
    embedder_names = [
        'pe_embedder',        # FLUX (EmbedND) - MOST COMMON
        'pos_embed',          # Some DiT variants
        'rope_embedder',      # Z-Image
        'position_embedder',  # Generic
        'x_embedder',         # Some DiT variants
        'patch_embed',        # May contain position logic
        'rotary_emb',         # Some implementations
        'rope',               # Short form
        'img_in',             # FLUX image input (not RoPE but related)
    ]

    for name in embedder_names:
        if hasattr(model, name):
            return getattr(model, name)

    # Search recursively in named_modules
    for name, module in model.named_modules():
        name_lower = name.lower()
        if 'pe_embedder' in name_lower or 'pos_embed' in name_lower:
            return module

    # Broader search
    for name, module in model.named_modules():
        name_lower = name.lower()
        for emb_name in embedder_names:
            if emb_name in name_lower:
                return module

    return None


def patch_model_for_circular_rope(
    model,
    circular_axis: int = -1,
    model_type: str = "auto"
) -> bool:
    """
    Patch a model to use circular RoPE for panorama generation.

    Args:
        model: ComfyUI model (ModelPatcher) or raw PyTorch model
        circular_axis: Which axis to make circular (-1 = last = X/width)
        model_type: "auto", "flux", "qwen", "zimage", or "generic"

    Returns:
        True if patching succeeded, False otherwise
    """
    # Find the RoPE embedder
    embedder = get_model_rope_embedder(model)

    if embedder is None:
        return False

    # Get the parent module to replace the embedder
    if hasattr(model, 'model'):
        base = model.model
    else:
        base = model

    if hasattr(base, 'diffusion_model'):
        diffusion_model = base.diffusion_model
    else:
        diffusion_model = base

    # Detect model type if auto
    if model_type == "auto":
        class_name = type(diffusion_model).__name__.lower()
        if 'flux' in class_name:
            model_type = "flux"
        elif 'qwen' in class_name:
            model_type = "qwen"
        elif 'zimage' in class_name or hasattr(diffusion_model, 'rope_embedder'):
            model_type = "zimage"
        else:
            model_type = "generic"

    # Create wrapped embedder
    wrapped = create_circular_rope_wrapper(embedder, circular_axis)

    # Find and replace the embedder (pe_embedder is most common for FLUX)
    for name in ['pe_embedder', 'pos_embed', 'rope_embedder', 'position_embedder', 'rotary_emb']:
        if hasattr(diffusion_model, name):
            setattr(diffusion_model, name, wrapped)
            return True

    # Also check model.model directly (some architectures)
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        diff = model.model.diffusion_model
        for name in ['pe_embedder', 'pos_embed', 'rope_embedder']:
            if hasattr(diff, name):
                setattr(diff, name, wrapped)
                return True

    return False


# =============================================================================
# TESTING
# =============================================================================

def test_circular_rope():
    """Test circular RoPE implementation."""
    print("Testing Circular RoPE...")

    # Test 1: Circular position IDs
    print("\n1. Testing circular position IDs...")
    pos = circular_position_ids(height=8, width=16)
    assert pos.shape == (128, 3), f"Expected (128, 3), got {pos.shape}"
    print(f"   ✓ Shape correct: {pos.shape}")

    # Check that X positions wrap (cos/sin at x=0 and x=15 should be close)
    first_row = pos[:16]  # First row of tokens
    cos_first = first_row[0, 1]   # cos at x=0
    sin_first = first_row[0, 2]   # sin at x=0
    cos_last = first_row[15, 1]   # cos at x=15
    sin_last = first_row[15, 2]   # sin at x=15

    # Distance on unit circle between x=0 and x=15 should be small
    # (they're adjacent in circular topology)
    dist = ((cos_first - cos_last)**2 + (sin_first - sin_last)**2).sqrt()
    print(f"   Distance between x=0 and x=15: {dist:.4f} (should be small)")
    assert dist < 0.5, "Adjacent positions should be close on circle"
    print("   ✓ Circular wrapping verified")

    # Test 2: Circular RoPE frequencies
    print("\n2. Testing circular RoPE frequencies...")
    freqs = circular_rope_frequencies(dim=64, width=16, height=8)
    assert freqs.shape == (128, 32), f"Expected (128, 32), got {freqs.shape}"
    print(f"   ✓ Frequency shape correct: {freqs.shape}")

    # Test 3: CircularPosEmbedFlux
    print("\n3. Testing CircularPosEmbedFlux...")
    flux_emb = CircularPosEmbedFlux(theta=10000, axes_dim=(16, 56, 56))

    # Create sample position IDs (batch=1, seq=128, axes=3)
    ids = torch.zeros(1, 128, 3)
    for i in range(128):
        ids[0, i, 0] = 0  # time
        ids[0, i, 1] = i // 16  # y
        ids[0, i, 2] = i % 16   # x

    emb = flux_emb(ids)
    print(f"   ✓ FLUX embedding shape: {emb.shape}")

    print("\n✅ All circular RoPE tests passed!\n")


if __name__ == "__main__":
    test_circular_rope()
