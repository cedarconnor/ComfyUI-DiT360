"""
ComfyUI-DiT360: 360° Panorama Generation Enhancement Nodes

Enhancement nodes for generating seamless equirectangular panoramic images
using FLUX.1-dev, FLUX.2, Qwen-Image, Z-Image, and other DiT-based models.

This node pack provides:

CORE NODES (5):
1. Equirect360EmptyLatent - 2:1 aspect ratio helper
2. Equirect360KSampler - Sampling with circular padding + optional losses
3. Equirect360VAEDecode - VAE decode with circular padding
4. Equirect360EdgeBlender - Post-processing edge blending
5. Equirect360Viewer - Interactive 360° preview (Three.js)

CIRCULAR ROPE NODES (4) - For true panorama topology:
6. ApplyCircularRoPE - Make position embeddings circular
7. CircularRoPESamplerWrapper - Dynamic RoPE for sampling
8. CircularRoPEPositionOverride - Advanced position control
9. ApplyCircularPanorama - All-in-one panorama patching

Supported Models:
- FLUX.1, FLUX.2 (Black Forest Labs)
- Qwen-Image (Alibaba)
- Z-Image, Z-Image-Turbo (Alibaba)
- HiDream, Lumina, and other DiT variants

Author: ComfyUI-DiT360 Contributors
License: Apache 2.0
Version: 2.1.0
"""

import os

# Import base nodes
from .nodes import (
    NODE_CLASS_MAPPINGS as BASE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BASE_NODE_DISPLAY_NAME_MAPPINGS
)

# Import circular RoPE nodes
try:
    from .nodes_circular_rope import (
        NODE_CLASS_MAPPINGS as ROPE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as ROPE_NODE_DISPLAY_NAME_MAPPINGS
    )
    ROPE_NODES_AVAILABLE = True
except ImportError as e:
    print(f"[DiT360] Warning: Circular RoPE nodes not loaded: {e}")
    ROPE_NODE_CLASS_MAPPINGS = {}
    ROPE_NODE_DISPLAY_NAME_MAPPINGS = {}
    ROPE_NODES_AVAILABLE = False

# Merge all node mappings
NODE_CLASS_MAPPINGS = {**BASE_NODE_CLASS_MAPPINGS, **ROPE_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**BASE_NODE_DISPLAY_NAME_MAPPINGS, **ROPE_NODE_DISPLAY_NAME_MAPPINGS}

# Version info
__version__ = "2.1.0"
__author__ = "ComfyUI-DiT360 Contributors"
__license__ = "Apache 2.0"

# Register web directory for Three.js viewer
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Startup message
print(f"\n{'='*60}")
print(f"ComfyUI-DiT360 v{__version__} loaded")
print(f"   - 5 core enhancement nodes")
if ROPE_NODES_AVAILABLE:
    print(f"   - 4 circular RoPE nodes (FLUX/Qwen/Z-Image)")
print(f"   - Supports: FLUX, Qwen-Image, Z-Image, HiDream")
print(f"{'='*60}\n")
