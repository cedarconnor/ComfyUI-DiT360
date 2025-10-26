"""
Utility modules for ComfyUI-DiT360

This package contains utility functions for:
- Equirectangular projection handling
- Circular padding for seamless panoramas
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
]
