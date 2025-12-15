"""
ComfyUI Nodes for Circular RoPE (Rotary Position Embeddings)

These nodes enable seamless 360° panorama generation by making the
horizontal position encoding circular, treating left and right edges
as adjacent.

Supports: FLUX.1/2, Qwen-Image, Z-Image, and other DiT-based models.
"""

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

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


def _dit360_conv2d_forward_x_circular_y_constant(
    self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Conv2d forward that applies circular padding on X and zero padding on Y.

    This is equivalent to the approach used by ComfyUI_pytorch360convert.
    """
    x = F.pad(x, self._dit360_padding_values_x, mode="circular")
    x = F.pad(x, self._dit360_padding_values_y, mode="constant", value=0)
    return F.conv2d(
        x,
        weight,
        bias,
        self.stride,
        (0, 0),  # padding is applied manually above
        self.dilation,
        self.groups,
    )


def _apply_circular_conv2d_padding(
    root: nn.Module,
    x_axis_only: bool = True,
) -> int:
    """
    Patch Conv2d layers under `root` to use circular padding on the width axis.

    Returns the number of Conv2d layers patched.
    """
    patched = 0

    for layer in root.modules():
        if not isinstance(layer, nn.Conv2d):
            continue

        if x_axis_only:
            if hasattr(layer, "_dit360_original_conv_forward"):
                continue

            padding = getattr(layer, "_reversed_padding_repeated_twice", None)
            if not (isinstance(padding, (tuple, list)) and len(padding) == 4):
                # Fallback: derive symmetric padding from .padding
                if isinstance(layer.padding, int):
                    pad_h = pad_w = int(layer.padding)
                else:
                    pad_h = int(layer.padding[0])
                    pad_w = int(layer.padding[1])
                padding = (pad_w, pad_w, pad_h, pad_h)

            pad_w = max(int(padding[0]), int(padding[1]))
            if pad_w <= 0:
                continue

            layer._dit360_original_conv_forward = layer._conv_forward
            layer._dit360_padding_values_x = (int(padding[0]), int(padding[1]), 0, 0)
            layer._dit360_padding_values_y = (0, 0, int(padding[2]), int(padding[3]))
            layer._conv_forward = _dit360_conv2d_forward_x_circular_y_constant.__get__(
                layer, nn.Conv2d
            )
            patched += 1
        else:
            # NOTE: This wraps both axes, which is not ideal for equirectangular,
            # but is provided for compatibility/testing.
            if getattr(layer, "padding_mode", "zeros") != "circular":
                layer.padding_mode = "circular"
                patched += 1

    return patched


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
                "model": ("MODEL", {"tooltip": "Connect your diffusion MODEL (e.g., FLUX UNet + DiT360 LoRA)."}),
            },
            "optional": {
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Toggle the RoPE patch on/off (useful for A/B testing without rewiring). Recommended: OFF unless you are testing attention-level seam handling."
                }),
                "mode": (["shift", "angle"], {
                    "default": "shift",
                    "tooltip": "RoPE circularization strategy. Recommended: shift (safer for planar-trained models like FLUX). Angle is aggressive/experimental and may degrade results."
                }),
                "seam_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Only for mode='shift': token columns near the right edge to wrap. Recommended: 0 (auto) or 4-16. Larger values can distort the image."
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print one-time diagnostic info (inferred token width, mode, seam width). Enable for debugging."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_circular_rope"
    CATEGORY = "DiT360/position_encoding"

    def apply_circular_rope(
        self,
        model,
        enable: bool = True,
        mode: str = "shift",
        seam_width: int = 0,
        verbose: bool = False,
    ):
        if not enable:
            return (model,)

        model_clone = model.clone()

        # Find the embedder
        attr_name, embedder = find_rope_embedder(model_clone)

        if embedder is None:
            print("[CircularRoPE] Warning: Could not find RoPE embedder")
            return (model_clone,)

        # Create wrapped embedder
        wrapped = create_circular_rope_wrapper(
            embedder,
            circular_axis=-1,
            mode=mode,
            seam_width=None if seam_width <= 0 else seam_width,
            verbose=verbose,
        )

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
                "model": ("MODEL", {"tooltip": "Connect your diffusion MODEL (e.g., FLUX UNet + DiT360 LoRA)."}),
            },
            "optional": {
                "patch_conv2d": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Patch Conv2d layers for X-only circular padding inside the model. For FLUX, this can reduce stability/quality; prefer 360° KSampler/VAE padding or Apply Circular Padding VAE. Enable only if you know you need model-internal padding."
                }),
                "patch_rope": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Patch RoPE (attention-level wrap). Recommended: OFF by default; enable only if you are specifically testing RoPE seam handling."
                }),
                "rope_mode": (["shift", "angle"], {
                    "default": "shift",
                    "tooltip": "RoPE strategy. Recommended: shift. Angle is aggressive/experimental and may degrade results."
                }),
                "rope_seam_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Only for rope_mode='shift': token columns near the right edge to wrap. Recommended: 0 (auto) or 4-16. Larger values can distort the image."
                }),
                "rope_verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print one-time diagnostic info about the RoPE patch. Enable for debugging."
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
        patch_conv2d: bool = True,
        patch_rope: bool = True,
        rope_mode: str = "shift",
        rope_seam_width: int = 0,
        rope_verbose: bool = False,
    ):
        model_clone = model.clone()
        patches = []

        # Legacy workflow compatibility: some older saved workflows pass model_type
        # (e.g. "flux") into the first optional slot.
        if isinstance(patch_conv2d, str):
            patch_conv2d = True

        # Patch Conv2d layers for circular padding
        if patch_conv2d:
            conv_count = self._patch_conv2d(model_clone)
            if conv_count > 0:
                patches.append(f"{conv_count} Conv2d")

        # Patch RoPE for circular topology
        if patch_rope:
            attr_name, embedder = find_rope_embedder(model_clone)

            if embedder is not None:
                wrapped = create_circular_rope_wrapper(
                    embedder,
                    circular_axis=-1,
                    mode=rope_mode,
                    seam_width=None if rope_seam_width <= 0 else rope_seam_width,
                    verbose=rope_verbose,
                )

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
                    # Get padding values
                    if isinstance(module.padding, int):
                        pad_h = pad_w = module.padding
                    else:
                        pad_h, pad_w = module.padding

                    # Only patch if there's width padding (we need to make it circular)
                    if pad_w > 0:
                        try:
                            # Store original forward (avoid double-wrapping)
                            if not hasattr(module, '_original_forward'):
                                module._original_forward = module.forward

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


class ApplyCircularPaddingVAE:
    """
    Patch a ComfyUI VAE so Conv2d layers use circular padding on the X axis.

    This helps reduce the visible seam at the left/right boundary after decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "Connect the VAE to patch (from VAELoader). If used, set 360° VAE Decode circular_padding to 0 to avoid double-padding."}),
                "inplace": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Modify the loaded VAE (True) or a deep-copied VAE (False). Recommended: True (faster/less memory). Use False only if you need both patched and unpatched VAEs in one workflow."
                }),
                "x_axis_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply circular padding only on X (recommended for equirectangular) or on both X and Y (generally not recommended)."
                }),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "apply"
    CATEGORY = "DiT360/vae"

    def apply(self, vae, inplace: bool = True, x_axis_only: bool = True):
        if inplace:
            use_vae = vae
        else:
            try:
                use_vae = copy.deepcopy(vae)
            except Exception as e:
                print(f"[CircularVAE] deepcopy failed, falling back to inplace=True: {e}")
                use_vae = vae

        # ComfyUI VAEs typically expose a `first_stage_model` containing Conv2d layers.
        candidates = [
            getattr(use_vae, "first_stage_model", None),
            getattr(use_vae, "vae", None),
            getattr(use_vae, "model", None),
        ]
        target = next((c for c in candidates if isinstance(c, nn.Module)), None)
        if target is None:
            if isinstance(use_vae, nn.Module):
                target = use_vae
            else:
                print("[CircularVAE] Warning: Could not find a torch.nn.Module to patch on this VAE")
                return (use_vae,)

        patched = _apply_circular_conv2d_padding(target, x_axis_only=x_axis_only)
        mode = "x-only" if x_axis_only else "x+y"
        print(f"[CircularVAE] Patched {patched} Conv2d layers ({mode}).")

        return (use_vae,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ApplyCircularRoPE": ApplyCircularRoPE,
    "ApplyCircularPanorama": ApplyCircularPanorama,
    "ApplyCircularPaddingVAE": ApplyCircularPaddingVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyCircularRoPE": "Apply Circular RoPE",
    "ApplyCircularPanorama": "Apply Circular Panorama (All-in-One)",
    "ApplyCircularPaddingVAE": "Apply Circular Padding VAE",
}
