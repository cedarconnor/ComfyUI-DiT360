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

__all__ = [
    # Aspect Ratio
    'validate_aspect_ratio',
    'get_equirect_dimensions',
    'fix_aspect_ratio',
    # Edge Blending
    'blend_edges',
    'check_edge_continuity',
    # Circular Padding
    'apply_circular_padding',
    'remove_circular_padding',
    'create_circular_padding_wrapper',
]
