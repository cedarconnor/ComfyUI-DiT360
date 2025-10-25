"""
Utility modules for ComfyUI-DiT360

This package contains utility functions for:
- Equirectangular projection handling
- Circular padding for seamless panoramas
- Path handling for Windows compatibility
- Validation and error checking
"""

from .equirect import (
    validate_aspect_ratio,
    fix_aspect_ratio,
    blend_edges,
    check_edge_continuity,
)

from .padding import (
    apply_circular_padding,
    remove_circular_padding,
)

__all__ = [
    'validate_aspect_ratio',
    'fix_aspect_ratio',
    'blend_edges',
    'check_edge_continuity',
    'apply_circular_padding',
    'remove_circular_padding',
]
