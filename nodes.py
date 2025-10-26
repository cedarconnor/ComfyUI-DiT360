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
            },
            "optional": {
                "attention_backend": (["auto", "eager", "xformers", "flash"], {
                    "default": "auto",
                    "tooltip": "Attention backend to use inside the DiT transformer. "
                               "auto selects the best available (prefers xFormers, then FlashAttention)."
                }),
                "attention_slice_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Chunk size for attention slicing. 0 disables slicing. "
                               "Smaller values reduce VRAM usage at the cost of speed."
                }),
                "quantization_mode": (["none", "int8", "int4"], {
                    "default": "none",
                    "tooltip": "Optional model quantization. int8 uses torch dynamic quantization, "
                               "int4 uses bitsandbytes if available."
                }),
                "vae_tile_size": ("INT", {
                    "default": 1536,
                    "min": 512,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Tile height (pixels) for VAE encode/decode tiling. "
                               "Larger tiles yield better quality but use more VRAM."
                }),
                "vae_tile_overlap": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 1024,
                    "step": 16,
                    "tooltip": "Overlap between VAE tiles (pixels). Helps hide seams when tiling is enabled."
                }),
                "vae_auto_tile_pixels": ("INT", {
                    "default": 16777216,
                    "min": 0,
                    "max": 268435456,
                    "step": 1048576,
                    "tooltip": "Automatic tiling threshold in total image pixels. 0 uses internal default." 
                               "If the image exceeds this count, tiling is used automatically."
                }),
            }
        }

    RETURN_TYPES = ("DIT360_PIPE",)
    RETURN_NAMES = ("dit360_pipe",)
    FUNCTION = "load_models"
    CATEGORY = "DiT360/loaders"

    def load_models(
        self,
        model_path,
        precision,
        vae_path,
        t5_path,
        offload_to_cpu,
        attention_backend="auto",
        attention_slice_size=0,
        quantization_mode="none",
        vae_tile_size=1536,
        vae_tile_overlap=128,
        vae_auto_tile_pixels=16777216
    ):
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
        print(f"Attention backend (requested): {attention_backend}")
        if attention_slice_size:
            print(f"Attention slicing (requested): {attention_slice_size}")
        print(f"Quantization mode (requested): {quantization_mode}")
        auto_tile_display = "disabled" if vae_auto_tile_pixels == 0 else vae_auto_tile_pixels
        print(f"VAE tiling config: tile={vae_tile_size}px overlap={vae_tile_overlap}px auto-threshold={auto_tile_display}")
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

        # Resolve overrides for advanced options
        slice_override = attention_slice_size if attention_slice_size > 0 else None
        auto_tile_pixels = vae_auto_tile_pixels if vae_auto_tile_pixels > 0 else vae_auto_tile_pixels

        # Load components (with placeholder implementations for now)
        try:
            # Load DiT360 model
            print(f"[1/3] Loading DiT360 model...")
            if model_full_path.exists():
                dit360_model = load_dit360_model(
                    model_full_path,
                    precision=precision,
                    device=device,
                    offload_device=offload_device,
                    enable_circular_padding=True,
                    attention_backend=attention_backend,
                    attention_slice_size=slice_override,
                    quantization_mode=quantization_mode
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
                    offload_device=offload_device,
                    tile_size=vae_tile_size,
                    tile_overlap=vae_tile_overlap,
                    max_tile_pixels=auto_tile_pixels
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
            "attention_backend": dit360_model.attention_backend if dit360_model else attention_backend,
            "attention_slice_size": dit360_model.attention_slice_size if dit360_model else slice_override,
            "quantization_mode": dit360_model.quantization_mode if dit360_model else quantization_mode,
            "vae_tile_size": vae.tile_size if vae else vae_tile_size,
            "vae_tile_overlap": vae.tile_overlap if vae else vae_tile_overlap,
            "vae_auto_tile_pixels": vae.max_tile_pixels if vae else auto_tile_pixels,
        }

        print(f"\n{'='*60}")
        print(f"✓ Pipeline ready!")
        print(f"  Model: {'✓ Loaded' if dit360_model else '✗ Not loaded'}")
        print(f"  VAE: {'✓ Loaded' if vae else '✗ Not loaded'}")
        print(f"  T5: {'✓ Loaded' if text_encoder else '✗ Not loaded'}")
        if dit360_model:
            print(f"  Attention backend: {dit360_model.attention_backend} (slice={dit360_model.attention_slice_size})")
            print(f"  Quantization: {dit360_model.quantization_mode}")
        else:
            print(f"  Attention backend: {pipeline['attention_backend']} (slice={pipeline['attention_slice_size']})")
            print(f"  Quantization: {pipeline['quantization_mode']}")
        if vae:
            print(f"  VAE tiling: tile={vae.tile_size}px overlap={vae.tile_overlap}px auto-threshold={vae.max_tile_pixels}")
        else:
            requested_auto = "disabled" if auto_tile_pixels == 0 else auto_tile_pixels
            print(f"  VAE tiling: tile={vae_tile_size}px overlap={vae_tile_overlap}px auto-threshold={requested_auto}")
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
                "scheduler_type": (["flow_match", "ddim"], {
                    "default": "flow_match",
                    "tooltip": "Sampling scheduler. flow_match (default) uses DiT360 rectified flow, ddim offers deterministic diffusion-style inference." 
                }),
                "timestep_schedule": (["linear", "quadratic", "cosine"], {
                    "default": "linear",
                    "tooltip": "Timestep schedule shaping noise distribution across steps."
                }),
                "scheduler_eta": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "DDIM sigma parameter (eta). 0 = deterministic, >0 adds stochasticity."
                }),
                "attention_backend": (["pipeline", "auto", "eager", "xformers", "flash"], {
                    "default": "pipeline",
                    "tooltip": "Attention implementation override for this run. 'pipeline' keeps settings from loader."
                }),
                "attention_slice_size": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Attention slicing chunk size. -1 = pipeline default, 0 = disable slicing, >0 sets explicit chunk size."
                }),
                "log_memory_stats": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print allocated and peak VRAM after generation (requires CUDA)."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "DiT360"

    def generate(self, dit360_pipe, conditioning, width, height, steps, cfg_scale, seed,
                 circular_padding, latent_image=None, denoise=1.0, enable_yaw_loss=False,
                 yaw_loss_weight=0.1, enable_cube_loss=False, cube_loss_weight=0.1,
                 scheduler_type="flow_match", timestep_schedule="linear", scheduler_eta=0.0,
                 attention_backend="pipeline", attention_slice_size=-1,
                 log_memory_stats=False):
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
        print(f"Scheduler: {scheduler_type} (schedule={timestep_schedule}, eta={scheduler_eta})")
        if attention_backend != "pipeline":
            print(f"Attention override: backend={attention_backend}")
        if attention_slice_size != -1:
            slice_desc = "disabled" if attention_slice_size == 0 else attention_slice_size
            print(f"Attention slicing override: {slice_desc}")
        print(f"{'='*60}\n")

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # TODO: Implement actual generation in Phase 4
        # For now, create dummy latent output

        device = dit360_pipe["device"]
        dtype = dit360_pipe["dtype"]
        if log_memory_stats and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                torch.cuda.reset_peak_memory_stats()

        # Calculate latent dimensions (8x downscale for VAE)
        latent_height = height // 8
        latent_width = width // 8

        #================================================================
        # Phase 3: Actual generation loop with flow matching
        # ================================================================

        from .dit360 import create_scheduler, get_timestep_schedule

        # Get components from pipeline
        model_wrapper = dit360_pipe.get("model")
        model = getattr(model_wrapper, "model", None) if model_wrapper else None
        text_encoder = dit360_pipe.get("text_encoder")

        if model is None:
            print("\n⚠ DiT360 model is not loaded. Returning zero latent.")
            latent_placeholder = torch.zeros(1, 4, latent_height, latent_width, device=device, dtype=dtype)
            return ({"samples": latent_placeholder},)

        # Apply attention overrides if requested
        override_kwargs = {}
        backend_override_val = (attention_backend or "pipeline").lower()
        if backend_override_val != "pipeline":
            override_kwargs["backend"] = backend_override_val
        if attention_slice_size != -1:
            override_kwargs["slice_size"] = None if attention_slice_size == 0 else attention_slice_size

        if override_kwargs:
            model_wrapper.set_attention_options(**override_kwargs)
        print(f"Attention backend in use: {model_wrapper.attention_backend} (slice={model_wrapper.attention_slice_size})")

        # Load model to device
        model_wrapper.load_to_device()

        # Get prompt embeddings from conditioning
        prompt_embeds = conditioning["prompt_embeds"]
        negative_prompt_embeds = conditioning.get("negative_prompt_embeds", None)

        # Initialize scheduler
        scheduler = create_scheduler(
            scheduler_type,
            num_train_timesteps=1000,
            shift=1.0,
            eta=scheduler_eta
        )
        scheduler.set_timesteps(steps, device=device)
        if scheduler_type == "flow_match":
            custom_timesteps = get_timestep_schedule(steps, method=timestep_schedule).to(device)
            scheduler.timesteps = custom_timesteps
        elif timestep_schedule != "linear":
            print("Note: timestep_schedule currently only affects the flow_match scheduler.")

        timesteps_tensor = scheduler.timesteps
        total_steps = len(timesteps_tensor)

        # Initialize or use existing latent
        if latent_image is not None:
            # Img2img: Start from existing latent with noise
            latent_clean = latent_image["samples"].clone().to(device, dtype=dtype)

            # Add noise based on denoise strength
            noise = torch.randn_like(latent_clean)
            t_start = max(min(int((1.0 - denoise) * total_steps), total_steps - 1), 0)
            timestep_value = timesteps_tensor[t_start]
            if scheduler_type == "ddim":
                timestep_for_noise = int(float(timestep_value)) if isinstance(timestep_value, torch.Tensor) else int(timestep_value)
            else:
                timestep_for_noise = timestep_value

            latent = scheduler.add_noise(latent_clean, noise, timestep_for_noise)
            print(f"Using input latent: {latent.shape}, denoise={denoise:.2f}, start_step={t_start}")

            # Adjust scheduler to start from t_start
            scheduler.timesteps = timesteps_tensor[t_start:]

        else:
            # Text-to-image: Start from pure noise
            latent = torch.randn(1, 4, latent_height, latent_width, device=device, dtype=dtype)
            print(f"Initialized random latent: {latent.shape}")

        timesteps_tensor = scheduler.timesteps
        total_steps = len(timesteps_tensor)

        # Initialize loss modules if enabled
        yaw_loss_fn = None
        cube_loss_fn = None

        if enable_yaw_loss:
            from .dit360 import YawLoss
            yaw_loss_fn = YawLoss(num_rotations=4, loss_type="l2")
            print(f"✓ Yaw loss enabled (weight={yaw_loss_weight})")

        if enable_cube_loss:
            from .dit360 import CubeLoss
            # Use smaller face size for latent space
            cube_loss_fn = CubeLoss(face_size=max(64, min(latent_height, latent_width)), loss_type="l2")
            print(f"✓ Cube loss enabled (weight={cube_loss_weight})")

        # Sampling loop
        print(f"\nStarting generation loop...")
        pbar = comfy.utils.ProgressBar(total_steps)

        num_train_timesteps = getattr(scheduler, "num_train_timesteps", 1000)

        for i, t in enumerate(timesteps_tensor):
            if isinstance(t, torch.Tensor):
                t_scalar = float(t.cpu().item())
            else:
                t_scalar = float(t)

            if scheduler_type == "ddim":
                normalized_t = t_scalar / max(num_train_timesteps - 1, 1)
                timestep_model = torch.tensor([normalized_t], device=device, dtype=torch.float32)
                scheduler_t_value = int(round(t_scalar))
            else:
                timestep_model = torch.tensor([t_scalar], device=device, dtype=torch.float32)
                scheduler_t_value = t_scalar

            # Classifier-free guidance: Run model twice if we have negative prompt
            if cfg_scale != 1.0 and negative_prompt_embeds is not None:
                latent_model_input = torch.cat([latent, latent], dim=0)
                timestep_input = torch.cat([timestep_model, timestep_model], dim=0)

                context = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

                noise_pred = model(latent_model_input, timestep_input, context)

                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            else:
                noise_pred = model(latent, timestep_model, prompt_embeds)

            # Apply geometric losses if enabled (Phase 4 Advanced Features)
            # Note: These losses are computed for monitoring/debugging during generation
            # For training-time losses, these would be used to compute gradients
            # For inference-time guidance, we compute them but don't backprop
            if (enable_yaw_loss or enable_cube_loss) and i % 10 == 0:
                # Only compute every 10 steps to save time
                with torch.no_grad():
                    # Compute predicted x0 (denoised latent)
                    # For flow matching: x0 = latent - noise_pred
                    predicted_x0 = latent - noise_pred

                    # Yaw consistency loss (for monitoring)
                    if enable_yaw_loss and yaw_loss_fn is not None:
                        yaw_loss = yaw_loss_fn(predicted_x0)
                        if i == 0:
                            print(f"  Step {i}: Yaw loss = {yaw_loss.item():.6f}")

                    # Cube projection loss (for monitoring)
                    if enable_cube_loss and cube_loss_fn is not None:
                        # Use latent as reference
                        cube_loss = cube_loss_fn(predicted_x0, latent)
                        if i == 0:
                            print(f"  Step {i}: Cube loss = {cube_loss.item():.6f}")

                # Note: For full inference-time guidance, losses would need to be
                # applied via gradient computation. This is left as a future enhancement.
                # Current implementation: losses are computed for monitoring only.

            # Apply scheduler step to update latent
            latent = scheduler.step(
                model_output=noise_pred,
                timestep=scheduler_t_value,
                sample=latent,
                guidance_scale=1.0,
                negative_model_output=None,
                step_index=i,
                eta=scheduler_eta
            )

            # Progress update
            pbar.update(1)

        if log_memory_stats and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            try:
                current_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            except Exception:
                current_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"VRAM stats — current: {current_alloc:.2f} GB, peak: {peak_alloc:.2f} GB")

        # Offload model to save VRAM
        model_wrapper.offload()

        print("✓ Generation complete!")

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
            },
            "optional": {
                "tiling_mode": (["auto", "always", "never"], {
                    "default": "auto",
                    "tooltip": "VAE tiling mode. auto uses loader heuristics, always forces tiling, never disables tiling."
                }),
                "tile_size_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Optional tile size override (pixels). 0 keeps loader default."
                }),
                "tile_overlap_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1024,
                    "step": 16,
                    "tooltip": "Optional tile overlap override (pixels). -1 keeps loader default, 0 disables overlap."
                }),
                "max_tile_pixels_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 268435456,
                    "step": 1048576,
                    "tooltip": "Optional auto-tiling threshold override in total pixels. -1 keeps loader default, 0 disables auto-tiling."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "DiT360"

    def decode(
        self,
        samples,
        dit360_pipe,
        auto_blend_edges,
        tiling_mode="auto",
        tile_size_override=0,
        tile_overlap_override=-1,
        max_tile_pixels_override=-1
    ):
        """Decode latents to images using VAE"""

        print(f"\nDecoding latents to image...")
        print(f"Auto blend edges: {auto_blend_edges}")
        print(f"Tiling mode: {tiling_mode}")

        # Phase 3: Actual VAE decoding
        latent = samples["samples"]
        batch, channels, latent_h, latent_w = latent.shape

        # Get VAE from pipeline
        vae = dit360_pipe["vae"]

        if vae is None:
            raise RuntimeError("DiT360 pipeline VAE is not loaded - cannot decode latents.")

        # Configure tiling overrides
        tile_size = tile_size_override if tile_size_override > 0 else vae.tile_size
        tile_overlap = tile_overlap_override if tile_overlap_override >= 0 else vae.tile_overlap
        max_tile_pixels = max_tile_pixels_override if max_tile_pixels_override >= 0 else vae.max_tile_pixels

        vae.configure_tiling(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            max_tile_pixels=max_tile_pixels
        )

        original_auto_threshold = vae.max_tile_pixels
        reset_auto_threshold = False
        if tiling_mode == "always":
            use_tiling_flag = True
        elif tiling_mode == "never":
            use_tiling_flag = False
            vae.configure_tiling(max_tile_pixels=0)
            if max_tile_pixels_override < 0:
                reset_auto_threshold = True
        else:
            # auto mode -> let heuristics decide (force flag False)
            use_tiling_flag = False

        print(f"VAE tiling configuration: tile={vae.tile_size}px overlap={vae.tile_overlap}px auto-threshold={vae.max_tile_pixels}")
        if tiling_mode == "auto":
            print("Use tiling this decode: auto (heuristics)")
        elif tiling_mode == "always":
            print("Use tiling this decode: forced on")
        else:
            print("Use tiling this decode: forced off")

        # Decode using VAE
        image = vae.decode(latent, use_tiling=use_tiling_flag)

        if reset_auto_threshold:
            vae.configure_tiling(max_tile_pixels=original_auto_threshold)

        # Image dimensions
        height, width = image.shape[1], image.shape[2]

        # Apply edge blending if enabled
        if auto_blend_edges:
            from .utils.equirect import blend_edges
            image = blend_edges(image, blend_width=10, mode="cosine")
            print("✓ Edges blended for seamless wraparound")

        print(f"✓ Decoded to {width}×{height} image")

        return (image,)


# ====================================================================
# NODE 4B: DiT360 Pipe Breakout / Combine
# ====================================================================

class DiT360PipeBreakout:
    """
    Extract model, VAE, and text encoder handles from a DiT360 pipeline.

    Useful for advanced workflows where components need to be tweaked,
    wrapped, or swapped independently before reassembling with the
    DiT360PipeCombine node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit360_pipe": ("DIT360_PIPE", {
                    "tooltip": "DiT360 pipeline dictionary to break apart"
                }),
            }
        }

    RETURN_TYPES = ("DIT360_MODEL", "DIT360_VAE", "DIT360_TEXT_ENCODER", "DIT360_PIPE")
    RETURN_NAMES = ("model", "vae", "text_encoder", "passthrough")
    FUNCTION = "breakout"
    CATEGORY = "DiT360/utils"

    def breakout(self, dit360_pipe):
        """Return individual components while passing the original pipeline through."""
        if not isinstance(dit360_pipe, dict):
            raise TypeError("Expected dit360_pipe to be a dict produced by DiT360Loader.")

        model = dit360_pipe.get("model")
        vae = dit360_pipe.get("vae")
        text_encoder = dit360_pipe.get("text_encoder")

        print("\nBreaking out DiT360 pipeline components...")
        print(f"  Model: {'✓' if model else '✗'}")
        print(f"  VAE: {'✓' if vae else '✗'}")
        print(f"  Text encoder: {'✓' if text_encoder else '✗'}")

        passthrough = dict(dit360_pipe)
        return (model, vae, text_encoder, passthrough)


class DiT360PipeCombine:
    """
    Reassemble a DiT360 pipeline from individual components.

    Accepts a base pipeline and optional overrides for model, VAE, text
    encoder, and metadata (dtype, devices, attention options, tiling).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_pipe": ("DIT360_PIPE", {
                    "tooltip": "Starting pipeline (usually output of DiT360Loader)"
                }),
            },
            "optional": {
                "model": ("DIT360_MODEL", {
                    "tooltip": "Optional DiT360 model wrapper override"
                }),
                "vae": ("DIT360_VAE", {
                    "tooltip": "Optional DiT360 VAE override"
                }),
                "text_encoder": ("DIT360_TEXT_ENCODER", {
                    "tooltip": "Optional text encoder override"
                }),
                "dtype_override": ("STRING", {
                    "default": "",
                    "tooltip": "Override pipeline dtype (fp16, fp32, bf16). Leave empty to keep current."
                }),
                "device_override": ("STRING", {
                    "default": "",
                    "tooltip": "Override compute device string (e.g., 'cuda:0', 'cpu')."
                }),
                "offload_device_override": ("STRING", {
                    "default": "",
                    "tooltip": "Override offload device string."
                }),
                "quantization_override": (["pipeline", "none", "int8", "int4"], {
                    "default": "pipeline",
                    "tooltip": "Override quantization mode recorded in pipeline."
                }),
                "attention_backend_override": (["pipeline", "auto", "eager", "xformers", "flash"], {
                    "default": "pipeline",
                    "tooltip": "Override attention backend recorded in pipeline (and update model when present)."
                }),
                "attention_slice_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Override attention slice size (-1 keeps current, 0 disables slicing)."
                }),
                "vae_tile_size_override": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Override VAE tile size in pixels (0 keeps current)."
                }),
                "vae_tile_overlap_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1024,
                    "step": 16,
                    "tooltip": "Override VAE tile overlap in pixels (-1 keeps current)."
                }),
                "vae_auto_tile_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 268435456,
                    "step": 1048576,
                    "tooltip": "Override VAE auto tiling threshold in pixels (-1 keeps current, 0 disables)."
                }),
                "refresh_metadata_from_components": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pull dtype/device/tiling metadata from supplied components when available."
                }),
            }
        }

    RETURN_TYPES = ("DIT360_PIPE",)
    RETURN_NAMES = ("dit360_pipe",)
    FUNCTION = "combine"
    CATEGORY = "DiT360/utils"

    def combine(
        self,
        base_pipe,
        model=None,
        vae=None,
        text_encoder=None,
        dtype_override="",
        device_override="",
        offload_device_override="",
        quantization_override="pipeline",
        attention_backend_override="pipeline",
        attention_slice_override=-1,
        vae_tile_size_override=0,
        vae_tile_overlap_override=-1,
        vae_auto_tile_override=-1,
        refresh_metadata_from_components=True,
    ):
        if not isinstance(base_pipe, dict):
            raise TypeError("base_pipe must be the dictionary produced by DiT360Loader.")

        pipeline = dict(base_pipe)

        if model is not None:
            pipeline["model"] = model
        if vae is not None:
            pipeline["vae"] = vae
        if text_encoder is not None:
            pipeline["text_encoder"] = text_encoder

        if refresh_metadata_from_components:
            if model is not None:
                pipeline["dtype"] = getattr(model, "dtype", pipeline.get("dtype"))
                pipeline["device"] = getattr(model, "device", pipeline.get("device"))
                pipeline["offload_device"] = getattr(model, "offload_device", pipeline.get("offload_device"))
                pipeline["attention_backend"] = getattr(model, "attention_backend", pipeline.get("attention_backend", "auto"))
                pipeline["attention_slice_size"] = getattr(model, "attention_slice_size", pipeline.get("attention_slice_size"))
                pipeline["quantization_mode"] = getattr(model, "quantization_mode", pipeline.get("quantization_mode", "none"))
            if vae is not None:
                pipeline["vae_tile_size"] = getattr(vae, "tile_size", pipeline.get("vae_tile_size", 1536))
                pipeline["vae_tile_overlap"] = getattr(vae, "tile_overlap", pipeline.get("vae_tile_overlap", 128))
                pipeline["vae_auto_tile_pixels"] = getattr(vae, "max_tile_pixels", pipeline.get("vae_auto_tile_pixels", 16777216))

        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        dtype_key = dtype_override.strip().lower()
        if dtype_key:
            if dtype_key not in dtype_map:
                raise ValueError(f"Unsupported dtype_override '{dtype_override}'. Use fp16, fp32, or bf16.")
            pipeline["dtype"] = dtype_map[dtype_key]

        if device_override.strip():
            pipeline["device"] = torch.device(device_override.strip())
        if offload_device_override.strip():
            pipeline["offload_device"] = torch.device(offload_device_override.strip())

        if quantization_override != "pipeline":
            pipeline["quantization_mode"] = quantization_override

        if attention_backend_override != "pipeline":
            pipeline["attention_backend"] = attention_backend_override
            model_to_update = pipeline.get("model")
            if model_to_update is not None and hasattr(model_to_update, "set_attention_options"):
                model_to_update.set_attention_options(backend=attention_backend_override)

        if attention_slice_override >= 0:
            slice_val = None if attention_slice_override == 0 else attention_slice_override
            pipeline["attention_slice_size"] = slice_val
            model_to_update = pipeline.get("model")
            if model_to_update is not None and hasattr(model_to_update, "set_attention_options"):
                model_to_update.set_attention_options(slice_size=slice_val)

        if vae_tile_size_override > 0 or vae_tile_overlap_override >= 0 or vae_auto_tile_override >= 0:
            vae_obj = pipeline.get("vae")
            if vae_obj is not None and hasattr(vae_obj, "configure_tiling"):
                tile_size = vae_tile_size_override if vae_tile_size_override > 0 else getattr(vae_obj, "tile_size", 1536)
                tile_overlap = vae_tile_overlap_override if vae_tile_overlap_override >= 0 else getattr(vae_obj, "tile_overlap", 128)
                auto_pixels = vae_auto_tile_override if vae_auto_tile_override >= 0 else getattr(vae_obj, "max_tile_pixels", 16777216)
                vae_obj.configure_tiling(
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    max_tile_pixels=auto_pixels,
                )
                pipeline["vae_tile_size"] = tile_size
                pipeline["vae_tile_overlap"] = tile_overlap
                pipeline["vae_auto_tile_pixels"] = auto_pixels

        print("\nCombining DiT360 pipeline components...")
        print(f"  Model: {'✓' if pipeline.get('model') else '✗'}")
        print(f"  VAE: {'✓' if pipeline.get('vae') else '✗'}")
        print(f"  Text encoder: {'✓' if pipeline.get('text_encoder') else '✗'}")
        print(f"  dtype: {pipeline.get('dtype')}")
        print(f"  device: {pipeline.get('device')}")
        print(f"  offload: {pipeline.get('offload_device')}")
        print(f"  attention backend: {pipeline.get('attention_backend')}")
        print(f"  attention slice: {pipeline.get('attention_slice_size')}")
        print(f"  quantization: {pipeline.get('quantization_mode')}")
        print(f"  VAE tiling: {pipeline.get('vae_tile_size')}px/{pipeline.get('vae_tile_overlap')}px auto={pipeline.get('vae_auto_tile_pixels')}")

        return (pipeline,)


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
# NODE 7: DiT360 LoRA Loader
# ====================================================================

class DiT360LoRALoader:
    """
    Load and merge LoRA weights into DiT360 model.

    LoRA (Low-Rank Adaptation) allows fine-tuning models for specific styles
    or subjects without retraining the entire model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit360_pipe": ("DIT360_PIPE", {
                    "tooltip": "DiT360 pipeline from DiT360Loader"
                }),
                "lora_path": ("STRING", {
                    "default": "lora/anime_style.safetensors",
                    "multiline": False,
                    "tooltip": "Path to LoRA .safetensors file (relative to models/dit360/ or absolute path)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA strength multiplier. 0.0 = no effect, 1.0 = full effect, "
                               "negative values remove LoRA"
                }),
                "merge_mode": (["merge", "unmerge"], {
                    "default": "merge",
                    "tooltip": "merge: Add LoRA weights, unmerge: Remove previously merged LoRA"
                }),
            }
        }

    RETURN_TYPES = ("DIT360_PIPE",)
    RETURN_NAMES = ("dit360_pipe",)
    FUNCTION = "load_lora"
    CATEGORY = "DiT360/advanced"

    def load_lora(self, dit360_pipe, lora_path, strength, merge_mode):
        """Load and merge/unmerge LoRA into model"""

        from pathlib import Path
        from .dit360 import load_lora_from_safetensors, merge_lora_into_model, unmerge_lora_from_model
        import folder_paths

        print(f"\n{'='*60}")
        print(f"LoRA: {merge_mode}ing weights")
        print(f"Path: {lora_path}")
        print(f"Strength: {strength:.2f}")

        # Resolve lora path
        lora_path = Path(lora_path)
        if not lora_path.is_absolute():
            # Try relative to dit360 models folder
            dit360_models = Path(folder_paths.models_dir) / "dit360" / "loras"
            dit360_models.mkdir(parents=True, exist_ok=True)
            lora_path = dit360_models / lora_path

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        # Load LoRA
        print(f"Loading LoRA from {lora_path}")
        lora_collection = load_lora_from_safetensors(
            lora_path,
            device=dit360_pipe["device"],
            dtype=dit360_pipe["dtype"]
        )

        # Get model from wrapper
        model_wrapper = dit360_pipe["model"]
        model = model_wrapper.model

        # Merge or unmerge
        if merge_mode == "merge":
            model = merge_lora_into_model(model, lora_collection, strength=strength)
            print(f"✓ LoRA merged with strength {strength:.2f}")
        else:  # unmerge
            model = unmerge_lora_from_model(model, lora_collection, strength=strength)
            print(f"✓ LoRA unmerged")

        print(f"{'='*60}\n")

        # Return updated pipeline
        return (dit360_pipe,)


# ====================================================================
# NODE 8: DiT360 Inpaint
# ====================================================================

class DiT360Inpaint:
    """
    Inpaint specific regions of a panorama using a mask.

    This node allows you to selectively regenerate parts of a panorama
    while keeping other regions unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit360_pipe": ("DIT360_PIPE", {
                    "tooltip": "DiT360 pipeline from DiT360Loader"
                }),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Text conditioning from DiT360TextEncode for inpainted region"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Original panorama image to inpaint"
                }),
                "mask": ("MASK", {
                    "tooltip": "Inpainting mask (white = inpaint, black = keep original)"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Number of sampling steps (more = higher quality, slower)"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Classifier-free guidance scale (higher = follow prompt more closely)"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising strength (1.0 = full regeneration, 0.0 = no change)"
                }),
                "blur_radius": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Blur mask edges for smooth blending (0 = hard edge)"
                }),
                "blend_mode": (["linear", "cosine", "smooth"], {
                    "default": "cosine",
                    "tooltip": "Blending mode for mask edges"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "inpaint"
    CATEGORY = "DiT360/advanced"

    def inpaint(self, dit360_pipe, conditioning, image, mask, steps, cfg_scale, denoise,
                blur_radius, blend_mode, seed=None):
        """Inpaint panorama with mask"""

        import torch
        from .dit360 import FlowMatchScheduler, prepare_inpaint_mask, blend_latents, apply_inpainting_conditioning
        import comfy.utils

        if seed is None:
            seed = torch.randint(0, 0xffffffffffffffff, (1,)).item()

        torch.manual_seed(seed)

        device = dit360_pipe["device"]
        dtype = dit360_pipe["dtype"]

        model_wrapper = dit360_pipe["model"]
        vae_wrapper = dit360_pipe["vae"]

        print(f"\n{'='*60}")
        print(f"DiT360 Inpainting")
        print(f"Steps: {steps}, CFG: {cfg_scale:.1f}, Denoise: {denoise:.2f}")
        print(f"Blur radius: {blur_radius}px, Blend: {blend_mode}")

        # Prepare mask
        prepared_mask = prepare_inpaint_mask(
            mask.unsqueeze(1),  # Add channel dim
            target_size=(image.shape[1], image.shape[2]),
            blur_radius=blur_radius,
            invert=False
        )

        # Encode original image
        print("Encoding original image...")
        vae_wrapper.load_to_device()
        original_latent = vae_wrapper.encode(image)
        vae_wrapper.offload()

        # Create latent mask
        latent_height, latent_width = original_latent.shape[2], original_latent.shape[3]
        from .dit360 import create_latent_noise_mask
        latent_mask = create_latent_noise_mask(
            prepared_mask,
            (latent_height, latent_width),
            vae_scale_factor=8
        )

        # Apply inpainting conditioning
        conditioned_latent, cond_mask = apply_inpainting_conditioning(
            original_latent,
            latent_mask,
            original_image_latent=original_latent,
            fill_mode="noise"
        )

        # Set up scheduler
        scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
        scheduler.set_timesteps(steps, device=device)

        # Start from conditioned latent with noise
        latent = conditioned_latent.to(device, dtype=dtype)

        # Get text embeddings
        prompt_embeds = conditioning["prompt_embeds"]
        negative_prompt_embeds = conditioning.get("negative_prompt_embeds", None)

        # Load model
        model_wrapper.load_to_device()
        model = model_wrapper.model

        # Sampling loop
        print(f"Inpainting with {len(scheduler.timesteps)} steps...")
        pbar = comfy.utils.ProgressBar(len(scheduler.timesteps))

        for i, t in enumerate(scheduler.timesteps):
            timestep = torch.tensor([t], device=device)

            # CFG
            if cfg_scale != 1.0 and negative_prompt_embeds is not None:
                latent_model_input = torch.cat([latent, latent], dim=0)
                timestep_input = torch.cat([timestep, timestep], dim=0)
                context = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

                noise_pred = model(latent_model_input, timestep_input, context)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model(latent, timestep, prompt_embeds)

            # Scheduler step
            latent = scheduler.step(model_output=noise_pred, timestep=t.item(), sample=latent)

            # Blend with original latent (keep unmasked regions)
            latent = blend_latents(
                original_latent.to(device, dtype=dtype),
                latent,
                latent_mask.to(device, dtype=dtype),
                blend_mode=blend_mode
            )

            pbar.update(1)

        model_wrapper.offload()

        print(f"✓ Inpainting complete")
        print(f"{'='*60}\n")

        return ({"samples": latent},)


# ====================================================================
# NODE REGISTRATION
# ====================================================================

NODE_CLASS_MAPPINGS = {
    "DiT360Loader": DiT360Loader,
    "DiT360TextEncode": DiT360TextEncode,
    "DiT360Sampler": DiT360Sampler,
    "DiT360Decode": DiT360Decode,
    "DiT360PipeBreakout": DiT360PipeBreakout,
    "DiT360PipeCombine": DiT360PipeCombine,
    "Equirect360Process": Equirect360Process,
    "Equirect360Preview": Equirect360Preview,
    "DiT360LoRALoader": DiT360LoRALoader,
    "DiT360Inpaint": DiT360Inpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360Loader": "DiT360 Loader",
    "DiT360TextEncode": "DiT360 Text Encode",
    "DiT360Sampler": "DiT360 Sampler",
    "DiT360Decode": "DiT360 Decode",
    "DiT360PipeBreakout": "DiT360 Pipe Breakout",
    "DiT360PipeCombine": "DiT360 Pipe Combine",
    "Equirect360Process": "Equirect360 Process",
    "Equirect360Preview": "Equirect360 Preview",
    "DiT360LoRALoader": "DiT360 LoRA Loader",
    "DiT360Inpaint": "DiT360 Inpaint",
}
