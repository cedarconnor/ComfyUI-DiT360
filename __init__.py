"""
ComfyUI-DiT360: Panoramic image generation using DiT360 model

A custom node pack for generating high-fidelity 360-degree equirectangular
panoramic images using the DiT360 diffusion transformer model.

Author: ComfyUI-DiT360 Contributors
License: Apache 2.0
Version: 0.1.0
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Register custom model paths
import folder_paths
import os
from pathlib import Path

# Add dit360 model directory
dit360_models_dir = Path(folder_paths.models_dir) / "dit360"
dit360_models_dir.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path("dit360", str(dit360_models_dir))

# Version info
__version__ = "0.1.0"
__author__ = "ComfyUI-DiT360 Contributors"
__license__ = "Apache 2.0"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"\n{'='*60}")
print(f"ComfyUI-DiT360 v{__version__} loaded")
print(f"Model directory: {dit360_models_dir}")
print(f"{'='*60}\n")
