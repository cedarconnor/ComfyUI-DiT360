"""
ComfyUI-DiT360: 360° Panorama Generation Enhancement Nodes

Enhancement nodes for generating seamless equirectangular panoramic images
using FLUX.1-dev with the DiT360 LoRA adapter.

DiT360 is a LoRA adapter (~2-5GB), not a full model. Users load FLUX.1-dev
normally via standard ComfyUI nodes, then apply the DiT360 LoRA using the
standard Load LoRA node.

This node pack provides 5 enhancement nodes:
1. Equirect360EmptyLatent - 2:1 aspect ratio helper
2. Equirect360KSampler - Sampling with circular padding + optional losses
3. Equirect360VAEDecode - VAE decode with circular padding
4. Equirect360EdgeBlender - Post-processing edge blending
5. Equirect360Viewer - Interactive 360° preview (Three.js)

Author: ComfyUI-DiT360 Contributors
License: Apache 2.0
Version: 2.0.0
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import os

# Version info
__version__ = "2.0.0"
__author__ = "ComfyUI-DiT360 Contributors"
__license__ = "Apache 2.0"

# Register web directory for Three.js viewer
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print(f"\n{'='*60}")
print(f"✅ ComfyUI-DiT360 v{__version__} loaded")
print(f"   • 5 enhancement nodes for 360° panoramas")
print(f"   • Works with FLUX.1-dev + DiT360 LoRA")
print(f"   • Circular padding for seamless edges")
print(f"{'='*60}\n")
