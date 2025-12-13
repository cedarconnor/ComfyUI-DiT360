"""
Utility modules for ComfyUI-DiT360

This package contains utility functions for:
- Equirectangular projection handling
- Circular padding for seamless panoramas
- Circular RoPE for attention-level circularity
- Edge blending for perfect wraparound
- Path handling for Windows compatibility
- Validation and error checking
"""

from .equirect import (
    validate_aspect_ratio,
    get_equirect_dimensions,
    fix_aspect_ratio,
    blend_edges,
    check_edge_continuity,
)

from .padding import (
    apply_circular_padding,
    remove_circular_padding,
    create_circular_padding_wrapper,
)

# Import circular RoPE utilities
try:
    from .circular_rope import (
        circular_position_ids,
        circular_rope_frequencies,
        apply_circular_rope,
        CircularRoPEEmbedding,
        CircularPosEmbedFlux,
        CircularPosEmbedQwen,
        CircularPosEmbedZImage,
        create_circular_rope_wrapper,
        get_model_rope_embedder,
        patch_model_for_circular_rope,
    )
    CIRCULAR_ROPE_AVAILABLE = True
except ImportError:
    CIRCULAR_ROPE_AVAILABLE = False


def validate_circular_continuity(image, threshold: float = 0.05) -> bool:
    """
    Backwards-compatible alias for :func:`check_edge_continuity`.

    Several internal docs and examples still reference the older helper name.
    Keeping the alias avoids breaking those references while steering new code
    toward ``check_edge_continuity``.
    """
    return check_edge_continuity(image, threshold)

__all__ = [
    # Aspect Ratio
    'validate_aspect_ratio',
    'get_equirect_dimensions',
    'fix_aspect_ratio',
    # Edge Blending
    'blend_edges',
    'check_edge_continuity',
    'validate_circular_continuity',
    # Circular Padding
    'apply_circular_padding',
    'remove_circular_padding',
    'create_circular_padding_wrapper',
    # Circular RoPE (if available)
    'CIRCULAR_ROPE_AVAILABLE',
]

# Add circular RoPE exports if available
if CIRCULAR_ROPE_AVAILABLE:
    __all__.extend([
        'circular_position_ids',
        'circular_rope_frequencies',
        'apply_circular_rope',
        'CircularRoPEEmbedding',
        'CircularPosEmbedFlux',
        'CircularPosEmbedQwen',
        'CircularPosEmbedZImage',
        'create_circular_rope_wrapper',
        'get_model_rope_embedder',
        'patch_model_for_circular_rope',
    ])
