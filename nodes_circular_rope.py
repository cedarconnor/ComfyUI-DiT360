"""
ComfyUI Nodes for Circular RoPE (Rotary Position Embeddings)

These nodes enable seamless 360° panorama generation by making the
horizontal position encoding circular, treating left and right edges
as adjacent.

Supports: FLUX.1/2, Qwen-Image, Z-Image, and other DiT-based models.

Usage:
------
1. Load your model (FLUX, Qwen-Image, Z-Image, etc.)
2. Connect to "Apply Circular RoPE" node
3. Use the patched model with your sampler

The patched model will treat horizontal positions as circular,
so the attention mechanism "knows" that left and right edges are adjacent.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

# ComfyUI imports
try:
    import comfy.model_patcher
    import comfy.samplers
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("[CircularRoPE] Warning: ComfyUI not found, running in standalone mode")

# Local imports
from .utils.circular_rope import (
    CircularRoPEEmbedding,
    CircularPosEmbedFlux,
    CircularPosEmbedQwen,
    CircularPosEmbedZImage,
    create_circular_rope_wrapper,
    get_model_rope_embedder,
    patch_model_for_circular_rope,
)


# =============================================================================
# NODE: APPLY CIRCULAR ROPE
# =============================================================================

class ApplyCircularRoPE:
    """
    Apply Circular RoPE (Rotary Position Embeddings) to a model.

    This makes the model treat horizontal positions as circular,
    enabling seamless 360° panorama generation.

    The key insight: standard RoPE encodes positions linearly,
    so position 0 and position N-1 are maximally distant.
    Circular RoPE maps positions to a circle, making 0 and N-1 adjacent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": ([
                    "auto",      # Auto-detect model type
                    "flux",      # FLUX.1, FLUX.2
                    "qwen",      # Qwen-Image
                    "zimage",    # Z-Image, Z-Image-Turbo
                    "hidream",   # HiDream
                    "lumina",    # Lumina
                    "generic",   # Try generic patching
                ], {"default": "auto"}),
                "circular_axis": ([
                    "x_only",    # Only X (horizontal) is circular
                    "y_only",    # Only Y (vertical) is circular
                    "both",      # Both axes circular (unusual)
                ], {"default": "x_only"}),
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
        circular_axis: str = "x_only",
        enable: bool = True
    ):
        """
        Apply circular RoPE to the model.

        Args:
            model: ComfyUI MODEL (ModelPatcher)
            model_type: Type of model for architecture-specific patching
            circular_axis: Which axis to make circular
            enable: Whether to enable circular RoPE

        Returns:
            Patched model tuple
        """
        if not enable:
            print("[CircularRoPE] Disabled, returning original model")
            return (model,)

        # Clone to avoid modifying the original
        model_clone = model.clone()

        # Convert axis string to index
        axis_map = {
            "x_only": -1,    # Last axis (width)
            "y_only": -2,    # Second-to-last (height)
            "both": None,    # Special handling
        }
        axis = axis_map.get(circular_axis, -1)

        # Attempt to patch the model
        if axis is None:  # both axes
            # For "both", we'd need to modify the underlying logic
            # For now, just use X (horizontal is what matters for panoramas)
            print("[CircularRoPE] Note: 'both' axes uses horizontal only (standard for panoramas)")
            axis = -1

        success = self._patch_model(model_clone, model_type, axis)

        if success:
            print(f"[CircularRoPE] Successfully patched model for circular topology")
            print(f"   Model type: {model_type}")
            print(f"   Circular axis: {circular_axis}")
        else:
            print("[CircularRoPE] Warning: Patching may not have fully succeeded")
            print("   The model will still work, but seamless edges may not be perfect")

        return (model_clone,)

    def _patch_model(self, model, model_type: str, circular_axis: int) -> bool:
        """
        Internal method to patch the model's RoPE.

        This uses ComfyUI's transformer_options mechanism when possible,
        falling back to direct module replacement.
        """
        # Try using ComfyUI's built-in rope_options if available
        # This is the cleanest approach as it doesn't modify model structure
        if hasattr(model, 'model_options'):
            # Store circular RoPE flag in transformer options
            if 'transformer_options' not in model.model_options:
                model.model_options['transformer_options'] = {}

            model.model_options['transformer_options']['circular_rope'] = {
                'enabled': True,
                'axis': circular_axis,
                'model_type': model_type,
            }

        # Also try direct patching as a backup
        try:
            return patch_model_for_circular_rope(
                model,
                circular_axis=circular_axis,
                model_type=model_type
            )
        except Exception as e:
            print(f"[CircularRoPE] Direct patching failed: {e}")
            return False


# =============================================================================
# NODE: CIRCULAR ROPE SAMPLER WRAPPER
# =============================================================================

class CircularRoPESamplerWrapper:
    """
    Wrapper node that applies circular RoPE during sampling.

    This node wraps the sampling process to inject circular position
    information dynamically, which can be more reliable than static patching.

    Use this if ApplyCircularRoPE doesn't work for your specific model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blending strength between standard and circular RoPE"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "wrap_for_sampling"
    CATEGORY = "DiT360/position_encoding"

    def wrap_for_sampling(self, model, latent, strength: float = 1.0):
        """
        Prepare model and latent for circular RoPE sampling.

        This injects the latent dimensions into the model's transformer options
        so the circular RoPE calculation knows the correct width.
        """
        model_clone = model.clone()
        latent_samples = latent["samples"]

        # Get latent dimensions
        if latent_samples.dim() == 4:
            B, C, H, W = latent_samples.shape
        else:
            H, W = 64, 128  # Default for 1024x2048 with 16x compression

        # Store dimensions for circular RoPE calculation
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}

        model_clone.model_options['transformer_options']['circular_rope_dims'] = {
            'latent_height': H,
            'latent_width': W,
            'strength': strength,
        }

        print(f"[CircularRoPE] Prepared for sampling: {W}x{H} latent")
        print(f"   Circular RoPE strength: {strength}")

        return (model_clone, latent)


# =============================================================================
# NODE: CIRCULAR ROPE POSITION OVERRIDE
# =============================================================================

class CircularRoPEPositionOverride:
    """
    Advanced node for manually controlling circular RoPE parameters.

    Use this for fine-grained control over the circular position encoding,
    such as adjusting the "wrap point" or applying partial circularity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "wrap_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Width at which positions wrap (0 = auto from latent)"
                }),
                "theta_base": ("FLOAT", {
                    "default": 10000.0,
                    "min": 1.0,
                    "max": 100000.0,
                    "step": 100.0,
                    "tooltip": "RoPE base frequency (10000 for FLUX, 256 for Z-Image)"
                }),
                "circular_blend": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "0 = linear RoPE, 1 = fully circular RoPE"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "override_positions"
    CATEGORY = "DiT360/position_encoding"

    def override_positions(
        self,
        model,
        wrap_width: int = 0,
        theta_base: float = 10000.0,
        circular_blend: float = 1.0
    ):
        """
        Override RoPE position parameters.

        Args:
            model: ComfyUI MODEL
            wrap_width: Width at which X positions wrap to 0
            theta_base: RoPE base frequency
            circular_blend: Interpolation between linear (0) and circular (1)

        Returns:
            Model with overridden position parameters
        """
        model_clone = model.clone()

        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}

        # Set custom RoPE options
        model_clone.model_options['transformer_options']['rope_options'] = {
            'circular': True,
            'wrap_width': wrap_width,
            'theta_base': theta_base,
            'circular_blend': circular_blend,
        }

        print(f"[CircularRoPE] Position override:")
        print(f"   Wrap width: {wrap_width if wrap_width > 0 else 'auto'}")
        print(f"   Theta base: {theta_base}")
        print(f"   Circular blend: {circular_blend}")

        return (model_clone,)


# =============================================================================
# NODE: PATCH CONV2D + ROPE (COMBINED)
# =============================================================================

class ApplyCircularPanorama:
    """
    Combined node that applies both circular Conv2d padding AND circular RoPE.

    This is the recommended "all-in-one" solution for panorama generation.
    It patches:
    1. Conv2d layers in VAE and PatchEmbed (for convolution-level circularity)
    2. RoPE position embeddings (for attention-level circularity)

    This gives the best results for seamless 360° panorama generation.
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
                "patch_vae_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only patch VAE Conv2d (safer but less effective)"
                }),
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
        patch_vae_only: bool = False
    ):
        """
        Apply all circular patches for panorama generation.
        """
        model_clone = model.clone()

        # Detect model type
        if model_type == "auto":
            model_type = self._detect_model_type(model_clone)

        patches_applied = []

        # 1. Patch Conv2d layers
        if patch_conv2d:
            conv_count = self._patch_conv2d(model_clone, patch_vae_only)
            if conv_count > 0:
                patches_applied.append(f"{conv_count} Conv2d layers")

        # 2. Patch RoPE
        if patch_rope:
            rope_success = patch_model_for_circular_rope(
                model_clone,
                circular_axis=-1,
                model_type=model_type
            )
            if rope_success:
                patches_applied.append("RoPE embedder")

        # Report results
        if patches_applied:
            print(f"[CircularPanorama] Applied patches: {', '.join(patches_applied)}")
            print(f"   Model type: {model_type}")
        else:
            print("[CircularPanorama] Warning: No patches applied")

        return (model_clone,)

    def _detect_model_type(self, model) -> str:
        """Auto-detect the model architecture type."""
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
        else:
            return "generic"

    def _patch_conv2d(self, model, vae_only: bool) -> int:
        """Patch Conv2d layers to use circular padding."""
        patched = 0
        base = model.model if hasattr(model, 'model') else model

        targets = []

        # Always include VAE if available
        if hasattr(base, 'first_stage_model'):
            targets.append(base.first_stage_model)

        # Include diffusion model if not vae_only
        if not vae_only and hasattr(base, 'diffusion_model'):
            targets.append(base.diffusion_model)

        for target in targets:
            for name, module in target.named_modules():
                if isinstance(module, nn.Conv2d):
                    if hasattr(module, 'padding') and module.padding[0] > 0:
                        try:
                            module.padding_mode = 'circular'
                            patched += 1
                        except Exception:
                            pass  # Some modules don't support this

        return patched


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "ApplyCircularRoPE": ApplyCircularRoPE,
    "CircularRoPESamplerWrapper": CircularRoPESamplerWrapper,
    "CircularRoPEPositionOverride": CircularRoPEPositionOverride,
    "ApplyCircularPanorama": ApplyCircularPanorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyCircularRoPE": "Apply Circular RoPE",
    "CircularRoPESamplerWrapper": "Circular RoPE Sampler Wrapper",
    "CircularRoPEPositionOverride": "Circular RoPE Position Override",
    "ApplyCircularPanorama": "Apply Circular Panorama (All-in-One)",
}
