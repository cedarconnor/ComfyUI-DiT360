"""
Core node implementations for ComfyUI-DiT360

This module contains all 6 core nodes for the DiT360 panorama generation workflow:
1. DiT360Loader - Load model, VAE, and text encoder
2. DiT360TextEncode - Encode text prompts
3. DiT360Sampler - Generate panoramic latents
4. DiT360Decode - Decode latents to images
5. Equirect360Process - Validate and process panoramas
6. Equirect360Preview - Interactive 360° viewer

No dependencies on external node packs - completely self-contained.
"""

import torch
import folder_paths
from pathlib import Path
import comfy.model_management as mm
import comfy.utils
import gc
import os

# ====================================================================
# NODE 1: DiT360 Loader (Loads Model, VAE, and Text Encoder)
# ====================================================================

class DiT360Loader:
    """
    Load DiT360 model components (Transformer, VAE, Text Encoder) in a single node.

    This consolidates model loading to simplify workflows. Models can be manually
    downloaded or auto-downloaded from HuggingFace.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "dit360_model.safetensors",
                    "tooltip": "Path to DiT360 model file. Can be filename (searches in models/dit360/) "
                               "or full path. Download from: https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation"
                }),
                "precision": (["fp16", "bf16", "fp32", "fp8"], {
                    "default": "fp16",
                    "tooltip": "Model precision. fp16 (recommended for 16-24GB VRAM), bf16 (better quality), "
                               "fp8 (experimental, lowest VRAM), fp32 (highest quality, requires 48GB+ VRAM)"
                }),
                "vae_path": ("STRING", {
                    "default": "dit360_vae.safetensors",
                    "tooltip": "Path to VAE model. Searches in models/vae/ or models/dit360/"
                }),
                "t5_path": ("STRING", {
                    "default": "t5-v1_1-xxl",
                    "tooltip": "Path to T5 text encoder. Searches in models/t5/ directory"
                }),
                "offload_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload models to CPU when not in use to save VRAM. "
                               "Recommended: True (slower but uses less VRAM)"
                }),
            }
        }

    RETURN_TYPES = ("DIT360_PIPE",)
    RETURN_NAMES = ("dit360_pipe",)
    FUNCTION = "load_models"
    CATEGORY = "DiT360/loaders"

    def load_models(self, model_path, precision, vae_path, t5_path, offload_to_cpu):
        """Load all DiT360 components and return as pipeline object"""
        from .dit360 import load_dit360_model, load_vae, load_t5_encoder
        from pathlib import Path

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device() if offload_to_cpu else device

        print(f"\n{'='*60}")
        print(f"Loading DiT360 Pipeline...")
        print(f"Precision: {precision}")
        print(f"Device: {device}")
        print(f"Offload: {offload_to_cpu}")
        print(f"{'='*60}\n")

        # Prepare paths
        import folder_paths
        models_dir = Path(folder_paths.models_dir)

        # Resolve model path
        if model_path and model_path.strip():
            model_full_path = Path(model_path)
            if not model_full_path.is_absolute():
                # Try in dit360 folder
                model_full_path = models_dir / "dit360" / model_path
        else:
            # Use default
            model_full_path = models_dir / "dit360" / "dit360_model.safetensors"

        # Resolve VAE path
        if vae_path and vae_path.strip():
            vae_full_path = Path(vae_path)
            if not vae_full_path.is_absolute():
                # Try in vae folder
                vae_full_path = models_dir / "vae" / vae_path
        else:
            # Use default
            vae_full_path = models_dir / "vae" / "dit360_vae.safetensors"

        # Resolve T5 path
        if t5_path and t5_path.strip():
            t5_full_path = Path(t5_path)
            if not t5_full_path.is_absolute():
                # Try in t5 folder
                t5_full_path = models_dir / "t5" / t5_path
        else:
            # Use default
            t5_full_path = models_dir / "t5" / "t5-v1_1-xxl"

        # Load components (with placeholder implementations for now)
        try:
            # Load DiT360 model
            print(f"[1/3] Loading DiT360 model...")
            if model_full_path.exists():
                dit360_model = load_dit360_model(
                    model_full_path,
                    precision=precision,
                    device=device,
                    offload_device=offload_device
                )
            else:
                print(f"  Model not found at: {model_full_path}")
                print(f"  Skipping model load (manual download required)")
                dit360_model = None

            # Load VAE
            print(f"\n[2/3] Loading VAE...")
            if vae_full_path.exists():
                vae = load_vae(
                    vae_full_path,
                    precision=precision,
                    device=device,
                    offload_device=offload_device
                )
            else:
                print(f"  VAE not found at: {vae_full_path}")
                print(f"  Skipping VAE load (manual download required)")
                vae = None

            # Load T5 encoder
            print(f"\n[3/3] Loading T5 text encoder...")
            if t5_full_path.exists():
                text_encoder = load_t5_encoder(
                    t5_full_path,
                    precision=precision,
                    device=device,
                    offload_device=offload_device
                )
            else:
                print(f"  T5 not found at: {t5_full_path}")
                print(f"  Skipping T5 load (manual download required)")
                text_encoder = None

        except Exception as e:
            print(f"\n⚠ Error loading models: {e}")
            print(f"  Continuing with placeholder pipeline...")
            dit360_model = None
            vae = None
            text_encoder = None

        # Create pipeline object
        pipeline = {
            "model": dit360_model,
            "vae": vae,
            "text_encoder": text_encoder,
            "dtype": dit360_model.dtype if dit360_model else torch.float16,
            "device": device,
            "offload_device": offload_device,
            "model_path": str(model_full_path),
            "vae_path": str(vae_full_path),
            "t5_path": str(t5_full_path),
        }

        print(f"\n{'='*60}")
        print(f"✓ Pipeline ready!")
        print(f"  Model: {'✓ Loaded' if dit360_model else '✗ Not loaded'}")
        print(f"  VAE: {'✓ Loaded' if vae else '✗ Not loaded'}")
        print(f"  T5: {'✓ Loaded' if text_encoder else '✗ Not loaded'}")
        print(f"{'='*60}\n")

        return (pipeline,)


# ====================================================================
# NODE 2: DiT360 Text Encoder
# ====================================================================

class DiT360TextEncode:
    """
    Encode text prompts using T5-XXL encoder.

    Supports positive and negative prompts for classifier-free guidance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit360_pipe": ("DIT360_PIPE", {
                    "tooltip": "DiT360 pipeline from DiT360Loader node"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Positive prompt describing the desired panoramic scene. "
                               "Example: 'A beautiful sunset over the ocean with palm trees'"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt for things to avoid. "
                               "Example: 'blurry, distorted, low quality'"
                }),
                "max_length": ("INT", {
                    "default": 512,
                    "min": 77,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Maximum token length for text encoding. "
                               "Longer prompts provide more detail but use more VRAM"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_text"
    CATEGORY = "DiT360/conditioning"

    def encode_text(self, dit360_pipe, prompt, negative_prompt, max_length):
        """Encode text prompts to conditioning embeddings"""
        from .dit360 import text_preprocessing

        print(f"\n{'='*60}")
        print(f"Encoding text prompts...")
        print(f"{'='*60}")
        print(f"Positive: {prompt[:100]}..." if len(prompt) > 100 else f"Positive: {prompt}")
        if negative_prompt:
            print(f"Negative: {negative_prompt[:100]}..." if len(negative_prompt) > 100 else f"Negative: {negative_prompt}")

        # Get text encoder from pipeline
        text_encoder = dit360_pipe.get("text_encoder")

        if text_encoder is None:
            # Fallback to placeholder if encoder not loaded
            print("\n⚠ Text encoder not loaded, using placeholder embeddings")
            device = dit360_pipe["device"]
            dtype = dit360_pipe["dtype"]

            conditioning = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "embeddings": torch.randn(1, max_length, 4096, device=device, dtype=dtype),
                "negative_embeddings": torch.randn(1, max_length, 4096, device=device, dtype=dtype) if negative_prompt else None,
            }
        else:
            # Use actual text encoder
            # Preprocess prompts
            clean_prompt = text_preprocessing(prompt)
            clean_negative = text_preprocessing(negative_prompt) if negative_prompt else ""

            print(f"\nProcessed prompts:")
            print(f"  Positive: {clean_prompt[:80]}...")
            if negative_prompt:
                print(f"  Negative: {clean_negative[:80]}...")

            # Encode
            result = text_encoder.encode(
                prompts=clean_prompt,
                negative_prompts=clean_negative if negative_prompt else None
            )

            conditioning = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "embeddings": result["prompt_embeds"],
                "negative_embeddings": result.get("negative_prompt_embeds"),
            }

        print(f"\n✓ Text encoding complete")
        print(f"  Embedding shape: {conditioning['embeddings'].shape}")
        print(f"{'='*60}\n")

        return (conditioning,)


# ====================================================================
# NODE 3: DiT360 Sampler (Main Generation Node)
# ====================================================================

class DiT360Sampler:
    """
    Generate panoramic latents using DiT360 diffusion transformer.

    Includes advanced options for yaw loss (rotational consistency) and
    cube loss (pole distortion reduction).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit360_pipe": ("DIT360_PIPE", {
                    "tooltip": "DiT360 pipeline from DiT360Loader"
                }),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Text conditioning from DiT360TextEncode"
                }),
                "width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Output width in pixels. MUST be 2× height for equirectangular format. "
                               "Recommended: 2048 (standard), 4096 (high quality), 1024 (fast/low VRAM)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Output height in pixels. MUST be width÷2 for equirectangular format. "
                               "Recommended: 1024 (standard), 2048 (high quality), 512 (fast/low VRAM)"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 150,
                    "step": 1,
                    "tooltip": "Number of denoising steps. More steps = higher quality but slower. "
                               "Recommended: 20-30 (fast), 50 (balanced), 100+ (high quality)"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Higher = more prompt adherence. "
                               "Recommended: 3.0-5.0 for DiT360"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility. Same seed = same result"
                }),
                "circular_padding": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Pixels of circular padding for seamless wraparound. "
                               "Recommended: 10 (default), 0 (disable), 20+ (very seamless)"
                }),
            },
            "optional": {
                "latent_image": ("LATENT", {
                    "tooltip": "Optional input latent for img2img. Leave empty for text-to-image"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising strength for img2img. 1.0 = full denoise (text-to-image), "
                               "0.5 = half denoise (img2img style transfer)"
                }),
                "enable_yaw_loss": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable yaw loss for enhanced rotational consistency. "
                               "Makes panorama look same when rotated. Slower but higher quality."
                }),
                "yaw_loss_weight": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Weight for yaw loss. Higher = more rotational consistency. "
                               "Recommended: 0.1 (default)"
                }),
                "enable_cube_loss": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable cube loss for reduced pole distortion. "
                               "Improves quality at top/bottom of panorama. Slower but higher quality."
                }),
                "cube_loss_weight": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Weight for cube loss. Higher = less pole distortion. "
                               "Recommended: 0.1 (default)"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "DiT360"

    def generate(self, dit360_pipe, conditioning, width, height, steps, cfg_scale, seed,
                 circular_padding, latent_image=None, denoise=1.0, enable_yaw_loss=False,
                 yaw_loss_weight=0.1, enable_cube_loss=False, cube_loss_weight=0.1):
        """Generate panoramic latents using DiT360 model"""

        from .utils.equirect import validate_aspect_ratio

        # Validate aspect ratio
        if not validate_aspect_ratio(width, height):
            ratio = width / height
            print(f"\n⚠ WARNING: Aspect ratio {ratio:.2f}:1 is not 2:1!")
            print(f"   Equirectangular panoramas require 2:1 ratio.")
            print(f"   Recommended resolution: {width}×{width//2}")
            print(f"   Output may have distortion.\n")

        print(f"\n{'='*60}")
        print(f"Generating {width}×{height} panorama")
        print(f"Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
        print(f"Circular padding: {circular_padding}px")
        if enable_yaw_loss:
            print(f"Yaw loss: enabled (weight={yaw_loss_weight})")
        if enable_cube_loss:
            print(f"Cube loss: enabled (weight={cube_loss_weight})")
        print(f"{'='*60}\n")

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # TODO: Implement actual generation in Phase 4
        # For now, create dummy latent output

        device = dit360_pipe["device"]
        dtype = dit360_pipe["dtype"]

        # Calculate latent dimensions (8x downscale for VAE)
        latent_height = height // 8
        latent_width = width // 8

        # Initialize or use existing latent
        if latent_image is not None:
            latent = latent_image["samples"].clone()
            print(f"Using input latent: {latent.shape}")
        else:
            latent = torch.randn(1, 4, latent_height, latent_width, device=device, dtype=dtype)
            print(f"Initialized random latent: {latent.shape}")

        # Placeholder: simulate sampling with progress bar
        pbar = comfy.utils.ProgressBar(steps)
        for step in range(steps):
            # TODO: Actual sampling loop with circular padding
            pbar.update(1)

        print("✓ Generation complete (placeholder - Phase 4)")

        # Return in ComfyUI LATENT format
        return ({"samples": latent},)


# ====================================================================
# NODE 4: DiT360 Decode
# ====================================================================

class DiT360Decode:
    """
    Decode latents to panoramic images using VAE.

    Includes optional automatic edge blending for seamless wraparound.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "Latent samples from DiT360Sampler"
                }),
                "dit360_pipe": ("DIT360_PIPE", {
                    "tooltip": "DiT360 pipeline (contains VAE)"
                }),
                "auto_blend_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically blend left/right edges for seamless wraparound. "
                               "Recommended: True for panoramas, False for normal images"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "DiT360"

    def decode(self, samples, dit360_pipe, auto_blend_edges):
        """Decode latents to images"""

        print(f"\nDecoding latents to image...")
        print(f"Auto blend edges: {auto_blend_edges}")

        # TODO: Implement actual VAE decoding in Phase 5
        # For now, create dummy image output

        latent = samples["samples"]
        batch, channels, latent_h, latent_w = latent.shape

        # Image is 8x upscale of latent
        height = latent_h * 8
        width = latent_w * 8

        # Create dummy image in ComfyUI format (B, H, W, C)
        image = torch.rand(batch, height, width, 3)

        # Apply edge blending if enabled
        if auto_blend_edges:
            from .utils.equirect import blend_edges
            image = blend_edges(image, blend_width=10, mode="cosine")
            print("✓ Edges blended for seamless wraparound")

        print(f"✓ Decoded to {width}×{height} image (placeholder - Phase 5)")

        return (image,)


# ====================================================================
# NODE 5: Equirect360 Process (Validation & Post-Processing)
# ====================================================================

class Equirect360Process:
    """
    Validate and process equirectangular panoramas.

    - Enforces 2:1 aspect ratio (crop/pad/stretch)
    - Blends edges for seamless wraparound
    - Validates edge continuity
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to process"
                }),
                "enforce_2_1_ratio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enforce 2:1 aspect ratio required for equirectangular panoramas"
                }),
                "fix_mode": (["none", "crop", "pad", "stretch"], {
                    "default": "pad",
                    "tooltip": "How to fix aspect ratio if not 2:1. "
                               "crop: center crop to 2:1, pad: add black bars, "
                               "stretch: resize (distorts), none: no fixing"
                }),
                "blend_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Blend left/right edges for seamless wraparound"
                }),
                "blend_width": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Width of edge blend region in pixels. "
                               "Recommended: 10 (default), 20+ (very smooth)"
                }),
                "blend_mode": (["linear", "cosine", "smooth"], {
                    "default": "cosine",
                    "tooltip": "Blending curve. cosine (smooth), linear (simple), smooth (quadratic)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "DiT360/utils"

    def process(self, image, enforce_2_1_ratio, fix_mode, blend_edges, blend_width, blend_mode):
        """Process and validate equirectangular image"""

        from .utils.equirect import validate_aspect_ratio, fix_aspect_ratio, blend_edges as blend_edges_fn, check_edge_continuity

        batch, height, width, channels = image.shape

        print(f"\n{'='*60}")
        print(f"Processing equirectangular image: {width}×{height}")

        # Check aspect ratio
        if enforce_2_1_ratio:
            if not validate_aspect_ratio(width, height):
                print(f"⚠ Aspect ratio {width/height:.2f}:1 is not 2:1")
                if fix_mode != "none":
                    print(f"  Fixing with mode: {fix_mode}")
                    image = fix_aspect_ratio(image, mode=fix_mode)
                    _, height, width, _ = image.shape
                    print(f"  New dimensions: {width}×{height}")
            else:
                print(f"✓ Aspect ratio 2:1 validated")

        # Blend edges for seamless wraparound
        if blend_edges and blend_width > 0:
            print(f"Blending edges (width={blend_width}px, mode={blend_mode})")
            image = blend_edges_fn(image, blend_width=blend_width, mode=blend_mode)

            # Check edge continuity
            is_continuous = check_edge_continuity(image, threshold=0.05)
            if is_continuous:
                print(f"✓ Edges are continuous (seamless wraparound)")
            else:
                print(f"⚠ Edges may have visible seam")

        print(f"{'='*60}\n")

        return (image,)


# ====================================================================
# NODE 6: Equirect360 Preview (Interactive 360° Viewer)
# ====================================================================

class Equirect360Preview:
    """
    Preview panoramic images with metadata.

    Future enhancement: Interactive 360° viewer using Three.js
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Panoramic images to preview"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "DiT360/utils"

    def preview(self, images, prompt=None, extra_pnginfo=None):
        """Preview panoramic images"""

        batch, height, width, channels = images.shape

        print(f"\n{'='*60}")
        print(f"Previewing {batch} panoramic image(s)")
        print(f"Resolution: {width}×{height}")
        print(f"{'='*60}\n")

        # TODO: Phase 6 - Implement interactive 360° viewer
        # For now, use standard ComfyUI preview

        # Convert to format ComfyUI expects for preview
        # This is a simplified preview - full viewer comes in Phase 6

        return {"ui": {"images": []}}


# ====================================================================
# NODE REGISTRATION
# ====================================================================

NODE_CLASS_MAPPINGS = {
    "DiT360Loader": DiT360Loader,
    "DiT360TextEncode": DiT360TextEncode,
    "DiT360Sampler": DiT360Sampler,
    "DiT360Decode": DiT360Decode,
    "Equirect360Process": Equirect360Process,
    "Equirect360Preview": Equirect360Preview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360Loader": "DiT360 Loader",
    "DiT360TextEncode": "DiT360 Text Encode",
    "DiT360Sampler": "DiT360 Sampler",
    "DiT360Decode": "DiT360 Decode",
    "Equirect360Process": "Equirect360 Process",
    "Equirect360Preview": "Equirect360 Preview",
}
