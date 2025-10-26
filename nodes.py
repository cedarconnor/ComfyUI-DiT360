"""
ComfyUI-DiT360 Node Implementations

5 enhancement nodes for generating seamless 360° equirectangular panoramas
using FLUX.1-dev with the DiT360 LoRA adapter.

Nodes:
1. Equirect360EmptyLatent - Create 2:1 aspect ratio latents
2. Equirect360KSampler - Sample with circular padding
3. Equirect360VAEDecode - Decode with circular padding
4. Equirect360EdgeBlender - Post-process edge blending
5. Equirect360Viewer - Interactive 360° preview
"""

import torch
import math
import numpy as np
from PIL import Image
import io
import base64

# ComfyUI imports
import comfy.samplers
import comfy.sample
import comfy.utils

# Our utilities
from .utils import (
    get_equirect_dimensions,
    validate_aspect_ratio,
    apply_circular_padding,
    remove_circular_padding,
    create_circular_padding_wrapper,
    blend_edges,
    check_edge_continuity,
)


# ====================================================================
# NODE 1: EQUIRECT360EMPTYLATENT
# ====================================================================

class Equirect360EmptyLatent:
    """
    Create empty latent with enforced 2:1 aspect ratio for equirectangular panoramas.

    This replaces EmptyLatentImage and ensures users create proper equirectangular
    dimensions automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 16,  # FLUX requires 16-pixel alignment
                    "tooltip": "Width in pixels (height will be automatically calculated as width/2 for 2:1 ratio)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4096
                })
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "DiT360/latent"

    def generate(self, width, batch_size):
        """Generate empty latent with 2:1 aspect ratio"""

        # Get valid equirectangular dimensions
        width, height = get_equirect_dimensions(width, alignment=16)

        # FLUX uses 16x compression factor, 16 channels
        latent_width = width // 16
        latent_height = height // 16

        # Create empty latent (16 channels for FLUX)
        latent = torch.zeros(
            [batch_size, 16, latent_height, latent_width],
            dtype=torch.float32
        )

        print(f"✅ Created equirectangular latent: {width}×{height} image → {latent_width}×{latent_height} latent (2:1 ratio)")

        return ({"samples": latent},)


# ====================================================================
# NODE 2: EQUIRECT360KSAMPLER
# ====================================================================

class Equirect360KSampler:
    """
    KSampler with circular padding for seamless 360° panoramas.

    This is the CORE node. It applies circular padding at each sampling step
    to ensure seamless wraparound at the left/right edges.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Circular padding
                "circular_padding": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 128,
                    "tooltip": "Padding width for seamless edges (16-32 recommended, 0 to disable)"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DiT360/sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise, circular_padding):
        """
        Sample with circular padding for seamless panoramas
        """

        # Clone model to avoid affecting other nodes
        model_clone = model.clone()

        # Wrap model to add circular padding if enabled
        if circular_padding > 0:
            model_clone = create_circular_padding_wrapper(model_clone, circular_padding)
            print(f"🔄 Circular padding enabled: {circular_padding} pixels in latent space")
        else:
            print("⚠️ Circular padding disabled (circular_padding=0)")

        # Get latent samples
        latent = latent_image["samples"]

        # Use ComfyUI's standard sample function
        samples = comfy.sample.sample(
            model_clone,
            comfy.utils.common_upscale(latent, latent.shape[3] * 8, latent.shape[2] * 8, "nearest-exact", "center")
                if denoise < 1.0 else latent,  # Handle img2img
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
            disable_noise=(denoise < 1.0),
            start_step=0,
            last_step=steps,
            force_full_denoise=True,
            seed=seed
        )

        print(f"✅ Sampling complete: {samples.shape}")

        return ({"samples": samples},)


# ====================================================================
# NODE 3: EQUIRECT360VAEDECODE
# ====================================================================

class Equirect360VAEDecode:
    """
    VAE decode with circular padding for smooth edges.

    Applies circular padding during VAE upscaling for extra edge smoothness.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "circular_padding": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 128,
                    "tooltip": "VAE decode padding (16 recommended, 0 to disable)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "DiT360/vae"

    def decode(self, samples, vae, circular_padding):
        """
        Decode latent with circular padding
        """

        latent = samples["samples"]

        if circular_padding > 0:
            # Apply padding before decode
            latent_padded = apply_circular_padding(latent, circular_padding)

            # Decode with VAE
            image_padded = vae.decode(latent_padded)

            # Remove padding (16x upscale factor for FLUX VAE)
            padding_pixels = circular_padding * 8  # VAE upscales 8x
            image = remove_circular_padding(image_padded, padding_pixels)

            print(f"🔄 VAE decoded with circular padding: {circular_padding} latent → {padding_pixels} pixels")
        else:
            # Standard decode
            image = vae.decode(latent)
            print("⚠️ VAE decoded without circular padding")

        # Ensure ComfyUI format: (B, H, W, C)
        if image.ndim == 4 and image.shape[1] == 3:  # (B, 3, H, W) → (B, H, W, 3)
            image = image.permute(0, 2, 3, 1)

        print(f"✅ Decoded to {image.shape[2]}×{image.shape[1]} panorama")

        return (image,)


# ====================================================================
# NODE 4: EQUIRECT360EDGEBLENDER
# ====================================================================

class Equirect360EdgeBlender:
    """
    Post-processing edge blending for perfect wraparound.

    This is the final polish step - blends left and right edges to ensure
    perfect seamless wraparound.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend_width": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 200,
                    "tooltip": "Blend region width in pixels (10-20 recommended)"
                }),
                "blend_mode": (["cosine", "linear", "smooth"], {
                    "default": "cosine",
                    "tooltip": "Blending curve (cosine is smoothest)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "DiT360/post_process"

    def blend(self, image, blend_width, blend_mode):
        """Apply edge blending"""

        if blend_width == 0:
            print("⚠️ blend_width=0, skipping edge blending")
            return (image,)

        blended = blend_edges(image, blend_width, blend_mode)

        # Validate seamlessness
        is_seamless = check_edge_continuity(blended, threshold=0.05)

        if is_seamless:
            print(f"✅ Edges blended seamlessly (mode: {blend_mode}, width: {blend_width})")
        else:
            print(f"⚠️ Edges may have visible seam (try increasing blend_width)")

        return (blended,)


# ====================================================================
# NODE 5: EQUIRECT360VIEWER
# ====================================================================

class Equirect360Viewer:
    """
    Interactive 360° panorama viewer.

    Prepares panorama for viewing with Three.js in the ComfyUI interface.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_resolution": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 8192,
                    "step": 16,
                    "tooltip": "Max width for preview (lower = faster loading)"
                })
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "DiT360/preview"

    def preview(self, images, max_resolution):
        """
        Prepare panorama for 360° viewing
        """

        results = []

        for idx, image in enumerate(images):
            # Convert tensor to PIL Image
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # Resize if needed
            W, H = img_pil.size
            if W > max_resolution:
                new_W = max_resolution
                new_H = max_W // 2  # Maintain 2:1 ratio
                img_pil = img_pil.resize((new_W, new_H), Image.LANCZOS)
                print(f"📐 Resized for preview: {W}×{H} → {new_W}×{new_H}")

            # Validate aspect ratio
            W, H = img_pil.size
            if not validate_aspect_ratio(W, H, tolerance=0.05):
                print(f"⚠️ Warning: Image {idx} is not 2:1 aspect ratio ({W}×{H} = {W/H:.2f}:1)")

            # Convert to base64 JPEG for web display
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=90)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            results.append({
                "type": "equirect360",
                "image": f"data:image/jpeg;base64,{img_base64}",
                "width": img_pil.size[0],
                "height": img_pil.size[1]
            })

        print(f"🌐 Prepared {len(results)} panorama(s) for 360° viewing")

        return {"ui": {"images": results}}


# ====================================================================
# NODE REGISTRATION
# ====================================================================

NODE_CLASS_MAPPINGS = {
    "Equirect360EmptyLatent": Equirect360EmptyLatent,
    "Equirect360KSampler": Equirect360KSampler,
    "Equirect360VAEDecode": Equirect360VAEDecode,
    "Equirect360EdgeBlender": Equirect360EdgeBlender,
    "Equirect360Viewer": Equirect360Viewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Equirect360EmptyLatent": "360° Empty Latent",
    "Equirect360KSampler": "360° KSampler",
    "Equirect360VAEDecode": "360° VAE Decode",
    "Equirect360EdgeBlender": "360° Edge Blender",
    "Equirect360Viewer": "360° Viewer",
}
