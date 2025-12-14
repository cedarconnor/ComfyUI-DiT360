"""
ComfyUI Nodes for Circular RoPE (Rotary Position Embeddings)

These nodes enable seamless 360° panorama generation by making the
horizontal position encoding circular, treating left and right edges
as adjacent.

Supports: FLUX.1/2, Qwen-Image, Z-Image, and other DiT-based models.
"""

import torch
import torch.nn as nn

# Local imports
try:
    from .utils.circular_rope import create_circular_rope_wrapper
except ImportError:
    from utils.circular_rope import create_circular_rope_wrapper


def find_rope_embedder(model):
    """Find the RoPE position embedder in a model."""
    # Get the actual model
    base = model.model if hasattr(model, 'model') else model
    diff = base.diffusion_model if hasattr(base, 'diffusion_model') else base

    # Check common attribute names (in priority order for FLUX)
    for name in ['pe_embedder', 'pos_embed', 'rope_embedder', 'position_embedder']:
        if hasattr(diff, name):
            return name, getattr(diff, name)

    return None, None


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
            },
            "optional": {
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_circular_rope"
    CATEGORY = "DiT360/position_encoding"

    def apply_circular_rope(self, model, enable: bool = True):
        if not enable:
            return (model,)

        model_clone = model.clone()

        # Find the embedder
        attr_name, embedder = find_rope_embedder(model_clone)

        if embedder is None:
            print("[CircularRoPE] Warning: Could not find RoPE embedder")
            return (model_clone,)

        # Create wrapped embedder
        wrapped = create_circular_rope_wrapper(embedder, circular_axis=-1)

        # Try add_object_patch first (preferred ComfyUI method)
        patched = False
        if hasattr(model_clone, 'add_object_patch'):
            try:
                model_clone.add_object_patch(f"diffusion_model.{attr_name}", wrapped)
                patched = True
                print(f"[CircularRoPE] Patched via add_object_patch: {attr_name}")
            except Exception as e:
                print(f"[CircularRoPE] add_object_patch failed: {e}")

        # Fallback to direct setattr
        if not patched:
            base = model_clone.model if hasattr(model_clone, 'model') else model_clone
            diff = base.diffusion_model if hasattr(base, 'diffusion_model') else base
            setattr(diff, attr_name, wrapped)
            print(f"[CircularRoPE] Patched via setattr: {attr_name}")

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
        patch_conv2d: bool = True,
        patch_rope: bool = True,
    ):
        model_clone = model.clone()
        patches = []

        # Patch Conv2d layers for circular padding
        if patch_conv2d:
            conv_count = self._patch_conv2d(model_clone)
            if conv_count > 0:
                patches.append(f"{conv_count} Conv2d")

        # Patch RoPE for circular topology
        if patch_rope:
            attr_name, embedder = find_rope_embedder(model_clone)

            if embedder is not None:
                wrapped = create_circular_rope_wrapper(embedder, circular_axis=-1)

                # Try add_object_patch first
                patched = False
                if hasattr(model_clone, 'add_object_patch'):
                    try:
                        model_clone.add_object_patch(f"diffusion_model.{attr_name}", wrapped)
                        patched = True
                    except Exception:
                        pass

                if not patched:
                    base = model_clone.model if hasattr(model_clone, 'model') else model_clone
                    diff = base.diffusion_model if hasattr(base, 'diffusion_model') else base
                    setattr(diff, attr_name, wrapped)

                patches.append("RoPE")

        if patches:
            print(f"[CircularPanorama] Patched: {', '.join(patches)}")
        else:
            print("[CircularPanorama] Warning: No patches applied")

        return (model_clone,)

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
