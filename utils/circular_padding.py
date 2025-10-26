"""
Compatibility wrapper for circular padding utilities.

Historically the project referenced ``utils/circular_padding.py`` in the
documentation.  The implementation was later consolidated under
``utils/padding.py``.  This module re-exports the public helpers so that both
paths remain valid.
"""

from .padding import (  # noqa: F401
    apply_circular_padding,
    remove_circular_padding,
    create_circular_padding_wrapper,
)

__all__ = [
    "apply_circular_padding",
    "remove_circular_padding",
    "create_circular_padding_wrapper",
]
