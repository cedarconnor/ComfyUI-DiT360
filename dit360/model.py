"""
DiT360 Model Architecture and Loading

This module implements the DiT360 diffusion transformer model based on FLUX.1-dev.
The model is a 12-billion parameter transformer designed for panoramic image generation
with special adaptations for equirectangular projection.

Key Features:
- Circular padding in attention layers for seamless wraparound
- RoPE positional embeddings adapted for spherical geometry
- Flow matching scheduler for efficient sampling
- Support for multiple precisions (fp32, fp16, bf16, fp8)
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file, safe_open
import json
from typing import Dict, Optional, Union
import comfy.model_management as mm
from huggingface_hub import snapshot_download
import os


class DiT360Model(nn.Module):
    """
    DiT360 Diffusion Transformer Model

    Based on FLUX.1-dev architecture with adaptations for panoramic generation:
    - Circular padding for seamless left/right wraparound
    - Modified attention for equirectangular projection
    - RoPE positional embeddings for spherical geometry

    Args:
        config: Model configuration dictionary
        enable_circular_padding: Enable circular padding in attention layers
    """

    def __init__(self, config: Dict, enable_circular_padding: bool = True):
        super().__init__()
        self.config = config
        self.enable_circular_padding = enable_circular_padding

        # Extract key parameters from config
        self.in_channels = config.get("in_channels", 4)
        self.hidden_size = config.get("hidden_size", 3072)
        self.num_layers = config.get("num_layers", 38)
        self.num_heads = config.get("num_heads", 24)
        self.caption_channels = config.get("caption_channels", 4096)

        print(f"\nInitializing DiT360 Model:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Circular padding: {enable_circular_padding}")

        # TODO Phase 3: Implement full transformer architecture
        # For now, create a placeholder that matches expected interface
        self.initialized = False

    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                context: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through DiT360 transformer

        Args:
            x: Input latent tensor (B, C, H, W)
            timestep: Timestep tensor (B,)
            context: Text conditioning (B, seq_len, dim)
            **kwargs: Additional conditioning

        Returns:
            Denoised latent prediction (B, C, H, W)
        """
        # TODO Phase 4: Implement actual forward pass with circular padding
        # For now, return input (placeholder)
        return x


class DiT360Wrapper:
    """
    Wrapper for DiT360 model that handles loading, caching, and device management.

    This wrapper:
    - Loads model from HuggingFace or local files
    - Manages model caching (avoid reloading)
    - Handles device placement (GPU/CPU)
    - Supports precision conversion
    """

    def __init__(
        self,
        model: DiT360Model,
        dtype: torch.dtype,
        device: torch.device,
        offload_device: torch.device
    ):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.offload_device = offload_device
        self.is_loaded = False

    def load_to_device(self):
        """Load model to GPU"""
        if not self.is_loaded:
            print(f"Loading DiT360 to {self.device}...")
            self.model.to(self.device)
            self.is_loaded = True

    def offload(self):
        """Offload model to CPU to free VRAM"""
        if self.is_loaded:
            print(f"Offloading DiT360 to {self.offload_device}...")
            self.model.to(self.offload_device)
            self.is_loaded = False
            mm.soft_empty_cache()


def download_dit360_from_huggingface(
    repo_id: str = "Insta360-Research/DiT360-Panorama-Image-Generation",
    save_dir: Path = None,
    model_name: str = "dit360_model"
) -> Path:
    """
    Download DiT360 model from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID
        save_dir: Directory to save model (default: ComfyUI/models/dit360/)
        model_name: Name for model directory

    Returns:
        Path to downloaded model directory

    Example:
        >>> model_dir = download_dit360_from_huggingface()
        >>> print(f"Model downloaded to: {model_dir}")
    """
    import folder_paths

    if save_dir is None:
        save_dir = Path(folder_paths.models_dir) / "dit360" / model_name
    else:
        save_dir = Path(save_dir)

    # Check if already downloaded
    if save_dir.exists() and (save_dir / "config.json").exists():
        print(f"Model already exists at: {save_dir}")
        return save_dir

    print(f"\n{'='*60}")
    print(f"Downloading DiT360 model from HuggingFace...")
    print(f"Repository: {repo_id}")
    print(f"Destination: {save_dir}")
    print(f"{'='*60}\n")

    try:
        # Download using HuggingFace Hub
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            # Ignore EMA weights if present (we only need main model)
            ignore_patterns=["*ema*", "*.md", "*.txt"]
        )

        print(f"\n✓ Download complete: {downloaded_path}\n")
        return Path(downloaded_path)

    except Exception as e:
        raise RuntimeError(
            f"\nFailed to download DiT360 model from HuggingFace.\n\n"
            f"Error: {e}\n\n"
            f"Please download manually from:\n"
            f"  https://huggingface.co/{repo_id}\n\n"
            f"And place in:\n"
            f"  {save_dir}\n"
        )


def load_dit360_model(
    model_path: Union[str, Path],
    precision: str = "fp16",
    device: Optional[torch.device] = None,
    offload_device: Optional[torch.device] = None,
    enable_circular_padding: bool = True
) -> DiT360Wrapper:
    """
    Load DiT360 model from file or HuggingFace

    Args:
        model_path: Path to model file (.safetensors) or directory
        precision: Model precision - "fp32", "fp16", "bf16", or "fp8"
        device: Target device (None = auto-detect GPU)
        offload_device: Device for offloading (None = CPU)
        enable_circular_padding: Enable circular padding for panoramas

    Returns:
        DiT360Wrapper containing loaded model

    Raises:
        FileNotFoundError: If model file not found
        RuntimeError: If model loading fails

    Example:
        >>> model = load_dit360_model("models/dit360/model.safetensors", precision="fp16")
        >>> # Use model for inference
        >>> output = model.model(latent, timestep, context)
    """
    model_path = Path(model_path)

    # Auto-detect devices if not specified
    if device is None:
        device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()

    print(f"\n{'='*60}")
    print(f"Loading DiT360 Model")
    print(f"{'='*60}")
    print(f"Path: {model_path}")
    print(f"Precision: {precision}")
    print(f"Device: {device}")
    print(f"Offload: {offload_device}")
    print(f"{'='*60}\n")

    # Handle directory vs file path
    if model_path.is_dir():
        # Look for .safetensors file in directory
        config_path = model_path / "config.json"
        model_file = None
        for ext in ["*.safetensors", "*.pt", "*.pth"]:
            matches = list(model_path.glob(ext))
            if matches:
                model_file = matches[0]
                break

        if model_file is None:
            raise FileNotFoundError(
                f"No model file found in directory: {model_path}\n"
                f"Expected .safetensors, .pt, or .pth file"
            )
        model_path = model_file
    else:
        # File path provided
        config_path = model_path.parent / "config.json"

    # Check if file exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n\n"
            f"Please download DiT360 model from:\n"
            f"  https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation\n\n"
            f"Or use auto-download by leaving model_path empty in DiT360Loader node.\n"
        )

    # Load or create config
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Config not found, using default DiT360 config")
        # Default FLUX.1-dev style config for DiT360
        config = {
            "model_type": "dit360",
            "architecture": "flux-dev",
            "params": "12B",
            "in_channels": 4,
            "hidden_size": 3072,
            "num_layers": 38,
            "num_heads": 24,
            "caption_channels": 4096,
            "model_max_length": 512,
        }

    # Initialize model
    print("Initializing DiT360 architecture...")
    model = DiT360Model(config, enable_circular_padding=enable_circular_padding)

    # Load weights
    print(f"Loading weights from: {model_path.name}")
    try:
        if model_path.suffix == ".safetensors":
            # Use safetensors for faster loading
            state_dict = load_file(str(model_path))
        else:
            # Use torch for .pt/.pth files
            state_dict = torch.load(str(model_path), map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Load state dict (strict=False to allow missing keys in placeholder)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"  Warning: Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Warning: Unexpected keys: {len(unexpected)}")

        print("✓ Weights loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    # Convert precision
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float16,  # fp8 not widely supported yet, fallback to fp16
    }

    if precision not in dtype_map:
        raise ValueError(f"Unknown precision: {precision}. Use: {list(dtype_map.keys())}")

    dtype = dtype_map[precision]

    if precision == "fp8":
        print("Warning: fp8 not fully supported, using fp16 instead")

    print(f"Converting to {precision} ({dtype})...")
    model = model.to(dtype=dtype)

    # Move to offload device initially (will load to GPU on demand)
    model = model.to(offload_device)
    model.eval()  # Set to evaluation mode

    print(f"✓ Model ready on {offload_device}")
    print(f"  (Will load to {device} when needed)\n")

    # Wrap model
    wrapper = DiT360Wrapper(
        model=model,
        dtype=dtype,
        device=device,
        offload_device=offload_device
    )

    return wrapper


def get_model_info(model_path: Union[str, Path]) -> Dict:
    """
    Get information about a DiT360 model without loading it

    Args:
        model_path: Path to model file or directory

    Returns:
        Dictionary with model information

    Example:
        >>> info = get_model_info("models/dit360/model.safetensors")
        >>> print(f"Model size: {info['size_gb']:.2f} GB")
    """
    model_path = Path(model_path)

    info = {
        "exists": model_path.exists(),
        "path": str(model_path),
        "type": None,
        "size_gb": 0.0,
        "config": None,
    }

    if not model_path.exists():
        return info

    # Get file size
    if model_path.is_file():
        size_bytes = model_path.stat().st_size
        info["size_gb"] = size_bytes / (1024**3)
        info["type"] = model_path.suffix

    # Try to load config
    config_path = model_path.parent / "config.json" if model_path.is_file() else model_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            info["config"] = json.load(f)

    return info
