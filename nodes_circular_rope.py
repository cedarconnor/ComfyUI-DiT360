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
        """
        Patch Conv2d layers to use X-only circular padding.

        NOTE: PyTorch's padding_mode='circular' wraps both X and Y.
        For panoramas, we need X-only (width wraps, height doesn't).
        This implementation wraps the forward method to apply custom padding.
        """
        import torch.nn.functional as F

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
                            # Store original forward
                            if not hasattr(module, '_original_forward'):
                                module._original_forward = module.forward

                                # Get padding values
                                if isinstance(module.padding, int):
                                    pad_h = pad_w = module.padding
                                else:
                                    pad_h, pad_w = module.padding

                                # Create custom forward with X-only circular padding
                                def make_circular_forward(conv_module, orig_forward, pad_h, pad_w):
                                    def circular_x_forward(x):
                                        # Apply Y padding (top/bottom) with zeros
                                        if pad_h > 0:
                                            x = F.pad(x, (0, 0, pad_h, pad_h), mode='constant', value=0)

                                        # Apply X padding (left/right) with circular wrapping
                                        if pad_w > 0:
                                            left_edge = x[:, :, :, :pad_w]
                                            right_edge = x[:, :, :, -pad_w:]
                                            x = torch.cat([right_edge, x, left_edge], dim=3)

                                        # Call original conv with padding=0 (we've already padded)
                                        return F.conv2d(
                                            x,
                                            conv_module.weight,
                                            conv_module.bias,
                                            conv_module.stride,
                                            padding=0,  # We've done padding manually
                                            dilation=conv_module.dilation,
                                            groups=conv_module.groups
                                        )
                                    return circular_x_forward

                                module.forward = make_circular_forward(module, module._original_forward, pad_h, pad_w)
                                patched += 1
                        except Exception as e:
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
