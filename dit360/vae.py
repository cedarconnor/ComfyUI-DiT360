"""
VAE (Variational Autoencoder) for DiT360

Handles encoding images to latent space and decoding latents to images.
Compatible with FLUX.1-dev VAE with 8x downscaling.

Key Features:
- 8x spatial downscaling (2048×1024 → 256×128 latent)
- Tiling support for large panoramas (>4096px)
- Maintains aspect ratio for equirectangular format
- Optimized for panoramic images
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file
from typing import Union, Optional, Tuple
import comfy.model_management as mm
from huggingface_hub import snapshot_download


class DiT360VAE:
    """
    VAE Wrapper for DiT360

    Provides encoding and decoding with proper device management
    and support for panoramic aspect ratios.

    Args:
        vae_model: The actual VAE model
        dtype: Data type (fp16, bf16, fp32)
        device: Computation device (GPU)
        offload_device: Storage device (CPU)
        scale_factor: Latent space scale factor (default 8)
    """

    def __init__(
        self,
        vae_model: nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        offload_device: torch.device,
        scale_factor: int = 8
    ):
        self.vae = vae_model
        self.dtype = dtype
        self.device = device
        self.offload_device = offload_device
        self.scale_factor = scale_factor
        self.is_loaded = False

    def load_to_device(self):
        """Load VAE to GPU"""
        if not self.is_loaded:
            print(f"Loading VAE to {self.device}...")
            self.vae.to(self.device)
            self.is_loaded = True

    def offload(self):
        """Offload VAE to CPU"""
        if self.is_loaded:
            print(f"Offloading VAE to {self.offload_device}...")
            self.vae.to(self.offload_device)
            self.is_loaded = False
            mm.soft_empty_cache()

    def encode(
        self,
        images: torch.Tensor,
        use_tiling: bool = False
    ) -> torch.Tensor:
        """
        Encode images to latent space

        Args:
            images: Input images in ComfyUI format (B, H, W, C), range [0, 1]
            use_tiling: Use tiling for large images (>4096px)

        Returns:
            Latent tensor (B, 4, H//8, W//8)

        Example:
            >>> images = torch.rand(1, 1024, 2048, 3)  # 2048×1024 panorama
            >>> latent = vae.encode(images)
            >>> latent.shape
            torch.Size([1, 4, 128, 256])
        """
        self.load_to_device()

        # Convert from ComfyUI format (B, H, W, C) to VAE format (B, C, H, W)
        x = images.permute(0, 3, 1, 2).to(self.device, dtype=self.dtype)

        # Normalize from [0, 1] to [-1, 1] for VAE
        x = (x * 2.0) - 1.0

        with torch.no_grad():
            # TODO Phase 3: Implement actual VAE encoding
            # For now, simulate latent space (8x downscale, 4 channels)
            B, C, H, W = x.shape
            latent_h = H // self.scale_factor
            latent_w = W // self.scale_factor

            # Placeholder: Create dummy latent
            latent = torch.randn(B, 4, latent_h, latent_w,
                                device=self.device, dtype=self.dtype)

        return latent

    def decode(
        self,
        latent: torch.Tensor,
        use_tiling: bool = False
    ) -> torch.Tensor:
        """
        Decode latent to image

        Args:
            latent: Latent tensor (B, 4, H, W)
            use_tiling: Use tiling for large latents

        Returns:
            Image tensor in ComfyUI format (B, H, W, C), range [0, 1]

        Example:
            >>> latent = torch.randn(1, 4, 128, 256)
            >>> image = vae.decode(latent)
            >>> image.shape
            torch.Size([1, 1024, 2048, 3])
        """
        self.load_to_device()

        latent = latent.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            # TODO Phase 3: Implement actual VAE decoding
            # For now, simulate image (8x upscale, 3 channels)
            B, C, H, W = latent.shape
            image_h = H * self.scale_factor
            image_w = W * self.scale_factor

            # Placeholder: Create dummy image in [-1, 1]
            image = torch.randn(B, 3, image_h, image_w,
                               device=self.device, dtype=self.dtype)

            # Denormalize from [-1, 1] to [0, 1]
            image = (image + 1.0) / 2.0
            image = torch.clamp(image, 0.0, 1.0)

        # Convert to ComfyUI format (B, H, W, C)
        image = image.permute(0, 2, 3, 1).cpu().float()

        return image

    def get_latent_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate latent dimensions for given image size

        Args:
            image_size: (height, width) of image

        Returns:
            (latent_height, latent_width)

        Example:
            >>> vae.get_latent_size((1024, 2048))
            (128, 256)
        """
        h, w = image_size
        return (h // self.scale_factor, w // self.scale_factor)


def download_vae_from_huggingface(
    repo_id: str = "black-forest-labs/FLUX.1-dev",
    save_dir: Path = None,
    vae_name: str = "ae.safetensors"
) -> Path:
    """
    Download VAE from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository
        save_dir: Save directory (default: ComfyUI/models/vae/)
        vae_name: VAE filename

    Returns:
        Path to downloaded VAE file
    """
    import folder_paths

    if save_dir is None:
        save_dir = Path(folder_paths.models_dir) / "vae"
        save_dir.mkdir(parents=True, exist_ok=True)

    vae_path = save_dir / vae_name

    if vae_path.exists():
        print(f"VAE already exists at: {vae_path}")
        return vae_path

    print(f"\n{'='*60}")
    print(f"Downloading VAE from HuggingFace...")
    print(f"Repository: {repo_id}")
    print(f"Destination: {vae_path}")
    print(f"{'='*60}\n")

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=vae_name,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False
        )

        print(f"\n✓ VAE downloaded: {downloaded_path}\n")
        return Path(downloaded_path)

    except Exception as e:
        raise RuntimeError(
            f"\nFailed to download VAE from HuggingFace.\n\n"
            f"Error: {e}\n\n"
            f"Please download manually from:\n"
            f"  https://huggingface.co/{repo_id}\n\n"
            f"And place in:\n"
            f"  {vae_path}\n"
        )


def load_vae(
    vae_path: Union[str, Path],
    precision: str = "fp16",
    device: Optional[torch.device] = None,
    offload_device: Optional[torch.device] = None
) -> DiT360VAE:
    """
    Load VAE model for DiT360

    Args:
        vae_path: Path to VAE file (.safetensors)
        precision: Model precision (fp32/fp16/bf16)
        device: Target device (None = auto)
        offload_device: Offload device (None = CPU)

    Returns:
        DiT360VAE wrapper

    Example:
        >>> vae = load_vae("models/vae/ae.safetensors", precision="fp16")
        >>> latent = vae.encode(images)
    """
    vae_path = Path(vae_path)

    # Auto-detect devices
    if device is None:
        device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()

    print(f"\n{'='*60}")
    print(f"Loading VAE")
    print(f"{'='*60}")
    print(f"Path: {vae_path}")
    print(f"Precision: {precision}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Check file exists
    if not vae_path.exists():
        raise FileNotFoundError(
            f"VAE file not found: {vae_path}\n\n"
            f"Please download FLUX.1-dev VAE from:\n"
            f"  https://huggingface.co/black-forest-labs/FLUX.1-dev\n\n"
            f"Or use auto-download in DiT360Loader.\n"
        )

    # TODO Phase 3: Load actual VAE model
    # For now, create placeholder VAE
    class PlaceholderVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.initialized = True

        def encode(self, x):
            return x

        def decode(self, x):
            return x

    vae_model = PlaceholderVAE()
    print("✓ Placeholder VAE created (actual VAE loading in Phase 3)")

    # Convert precision
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(precision, torch.float16)

    vae_model = vae_model.to(dtype=dtype, device=offload_device)
    vae_model.eval()

    # Wrap VAE
    wrapper = DiT360VAE(
        vae_model=vae_model,
        dtype=dtype,
        device=device,
        offload_device=offload_device,
        scale_factor=8  # FLUX VAE uses 8x downscale
    )

    print(f"✓ VAE ready\n")
    return wrapper
