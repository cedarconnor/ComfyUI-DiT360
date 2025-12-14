"""
ComfyUI Nodes for Circular RoPE (Rotary Position Embeddings)

These nodes enable seamless 360° panorama generation by making the
horizontal position encoding circular, treating left and right edges
as adjacent.

Supports: FLUX.1/2, Qwen-Image, Z-Image, and other DiT-based models.
"""

import torch
import torch.nn as nn
from typing import Optional

# ComfyUI imports
try:
    import comfy.model_patcher
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False

# Local imports
try:
    from .utils.circular_rope import (
        patch_model_for_circular_rope,
    )
except ImportError:
    from utils.circular_rope import (
        patch_model_for_circular_rope,
    )


class ApplyCircularRoPE:
    """
    Apply Circular RoPE (Rotary Position Embeddings) to a model.

    This makes the model treat horizontal positions as circular,
    enabling seamless 360° panorama generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": ([
                    "auto",
                    "flux",
                    "qwen",
                    "zimage",
                    "generic",
                ], {"default": "auto"}),
            },
            "optional": {
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_circular_rope"
    CATEGORY = "DiT360/position_encoding"

    def apply_circular_rope(
        self,
        model,
        model_type: str = "auto",
        enable: bool = True
    ):
        if not enable:
            return (model,)

        model_clone = model.clone()

        success = patch_model_for_circular_rope(
            model_clone,
            circular_axis=-1,  # X (horizontal) axis
            model_type=model_type
        )

        if success:
            print(f"[CircularRoPE] Patched for circular topology")
        else:
            print("[CircularRoPE] Warning: Could not patch model")

        return (model_clone,)


class ApplyCircularPanorama:
    """
    Combined node that applies both circular Conv2d padding AND circular RoPE.

    This is the recommended all-in-one solution for panorama generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": ([
                    "auto",
                    "flux",
                    "qwen",
                    "zimage",
                    "generic",
                ], {"default": "auto"}),
            },
            "optional": {
                "patch_conv2d": ("BOOLEAN", {"default": True}),
                "patch_rope": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_all"
    CATEGORY = "DiT360"

    def apply_all(
        self,
        model,
        model_type: str = "auto",
        patch_conv2d: bool = True,
        patch_rope: bool = True,
    ):
        model_clone = model.clone()

        # Detect model type
        if model_type == "auto":
            model_type = self._detect_model_type(model_clone)

        patches = []

        # Patch Conv2d layers for circular padding
        if patch_conv2d:
            conv_count = self._patch_conv2d(model_clone)
            if conv_count > 0:
                patches.append(f"{conv_count} Conv2d")

        # Patch RoPE for circular topology
        if patch_rope:
            if patch_model_for_circular_rope(model_clone, circular_axis=-1, model_type=model_type):
                patches.append("RoPE")

        if patches:
            print(f"[CircularPanorama] Patched: {', '.join(patches)}")

        return (model_clone,)

    def _detect_model_type(self, model) -> str:
        base = model.model if hasattr(model, 'model') else model
        if hasattr(base, 'diffusion_model'):
            class_name = type(base.diffusion_model).__name__.lower()
        else:
            class_name = type(base).__name__.lower()

        if 'flux' in class_name:
            return "flux"
        elif 'qwen' in class_name:
            return "qwen"
        elif 'zimage' in class_name:
            return "zimage"
        return "generic"

    def _patch_conv2d(self, model) -> int:
        """Patch Conv2d layers to use circular padding."""
        patched = 0
        base = model.model if hasattr(model, 'model') else model

        targets = []
        if hasattr(base, 'first_stage_model'):
            targets.append(base.first_stage_model)
        if hasattr(base, 'diffusion_model'):
            targets.append(base.diffusion_model)

        for target in targets:
            for name, module in target.named_modules():
                if isinstance(module, nn.Conv2d):
                    if hasattr(module, 'padding') and module.padding[0] > 0:
                        try:
                            module.padding_mode = 'circular'
                            patched += 1
                        except Exception:
                            pass

        return patched


# Node registration
NODE_CLASS_MAPPINGS = {
    "ApplyCircularRoPE": ApplyCircularRoPE,
    "ApplyCircularPanorama": ApplyCircularPanorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyCircularRoPE": "Apply Circular RoPE",
    "ApplyCircularPanorama": "Apply Circular Panorama (All-in-One)",
}
