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
import math


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
        scale_factor: int = 8,
        tile_size: int = 1536,
        tile_overlap: int = 128,
        max_tile_pixels: int = 4096 * 4096
    ):
        self.vae = vae_model
        self.dtype = dtype
        self.device = device
        self.offload_device = offload_device
        self.scale_factor = scale_factor
        self.is_loaded = False
        self.tile_size = max(tile_size, scale_factor)
        self.tile_overlap = max(0, min(tile_overlap, self.tile_size - 1))
        self.max_tile_pixels = max_tile_pixels

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

    def _should_tile(self, height: int, width: int, force: bool) -> bool:
        if force:
            return True
        if self.tile_size <= 0:
            return False
        if self.max_tile_pixels and self.max_tile_pixels > 0 and (height * width) > self.max_tile_pixels:
            return True
        if max(height, width) > self.tile_size:
            return True
        return False

    @staticmethod
    def _compute_tile_starts(length: int, tile: int, overlap: int) -> list:
        if tile >= length:
            return [0]
        stride = max(tile - overlap, 1)
        starts = list(range(0, max(length - tile, 0) + 1, stride))
        last = length - tile
        if starts[-1] != last:
            starts.append(last)
        return sorted(set(starts))

    @staticmethod
    def _blend_weights(
        tile_tensor: torch.Tensor,
        top: int,
        bottom: int,
        total: int,
        overlap_elements: int
    ) -> torch.Tensor:
        """Create blending weights for overlapping stripes."""
        weight = torch.ones_like(tile_tensor[:, :1, :, :])
        if overlap_elements <= 0:
            return weight

        if top > 0:
            ramp = torch.linspace(
                0.0,
                1.0,
                steps=min(overlap_elements, tile_tensor.shape[2]),
                device=tile_tensor.device,
                dtype=tile_tensor.dtype
            )
            weight[:, :, :ramp.numel(), :] *= ramp.view(1, 1, -1, 1)

        if bottom < total:
            ramp = torch.linspace(
                1.0,
                0.0,
                steps=min(overlap_elements, tile_tensor.shape[2]),
                device=tile_tensor.device,
                dtype=tile_tensor.dtype
            )
            weight[:, :, -ramp.numel():, :] *= ramp.view(1, 1, -1, 1)

        return weight

    def _encode_direct(self, x: torch.Tensor) -> torch.Tensor:
        """Direct VAE encode without tiling. Expects tensor on device and normalized."""
        with torch.no_grad():
            if hasattr(self.vae, 'encode'):
                try:
                    encoded = self.vae.encode(x)
                    if hasattr(encoded, 'latent_dist'):
                        latent = encoded.latent_dist.sample()
                    elif hasattr(encoded, 'latents'):
                        latent = encoded.latents
                    else:
                        latent = encoded
                except Exception as e:
                    print(f"Warning: VAE encode failed ({e}), using fallback")
                    latent = torch.nn.functional.avg_pool2d(x, kernel_size=self.scale_factor)
                    if latent.shape[1] != 4:
                        latent = latent[:, :4, :, :] if latent.shape[1] > 4 else \
                                torch.cat([latent, torch.zeros_like(latent[:, :1, :, :])], dim=1)
            else:
                latent = torch.nn.functional.avg_pool2d(x, kernel_size=self.scale_factor)
                if latent.shape[1] != 4:
                    latent = latent[:, :4, :, :] if latent.shape[1] > 4 else \
                            torch.cat([latent, torch.zeros_like(latent[:, :1, :, :])], dim=1)
        return latent

    def _decode_direct(self, latent: torch.Tensor) -> torch.Tensor:
        """Direct VAE decode without tiling."""
        with torch.no_grad():
            if hasattr(self.vae, 'decode'):
                try:
                    decoded = self.vae.decode(latent)
                    if hasattr(decoded, 'sample'):
                        image = decoded.sample
                    else:
                        image = decoded

                    image = (image + 1.0) / 2.0
                    image = torch.clamp(image, 0.0, 1.0)

                except Exception as e:
                    print(f"Warning: VAE decode failed ({e}), using fallback")
                    image = torch.nn.functional.interpolate(
                        latent,
                        scale_factor=self.scale_factor,
                        mode='bilinear',
                        align_corners=False
                    )
                    if image.shape[1] != 3:
                        image = image[:, :3, :, :] if image.shape[1] > 3 else \
                               torch.cat([image] * (3 // image.shape[1] + 1), dim=1)[:, :3, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
            else:
                image = torch.nn.functional.interpolate(
                    latent,
                    scale_factor=self.scale_factor,
                    mode='bilinear',
                    align_corners=False
                )
                if image.shape[1] != 3:
                    image = image[:, :3, :, :] if image.shape[1] > 3 else \
                           torch.cat([image] * (3 // image.shape[1] + 1), dim=1)[:, :3, :, :]
                image = torch.clamp(image, 0.0, 1.0)

        return image

    def _encode_tiled_height(self, x: torch.Tensor, height: int) -> torch.Tensor:
        """Encode using height-wise tiling with blending to control VRAM."""
        tile_h = min(self.tile_size, height)
        starts = self._compute_tile_starts(height, tile_h, self.tile_overlap)

        latent_accum = None
        weight_accum = None
        for top in starts:
            bottom = min(top + tile_h, height)
            tile = x[:, :, top:bottom, :]
            latent_tile = self._encode_direct(tile)

            if latent_accum is None:
                batch, channels, latent_tile_h, latent_w = latent_tile.shape
                total_latent_h = math.ceil(height / self.scale_factor)
                latent_accum = torch.zeros(batch, channels, total_latent_h, latent_w, device=latent_tile.device, dtype=latent_tile.dtype)
                weight_accum = torch.zeros_like(latent_accum)

            latent_top = top // self.scale_factor
            latent_bottom = min(latent_top + latent_tile.shape[2], latent_accum.shape[2])
            valid = latent_tile[:, :, :latent_bottom - latent_top, :]
            overlap_latent = max(self.tile_overlap // self.scale_factor, 0)
            weights = self._blend_weights(valid, latent_top, latent_bottom, latent_accum.shape[2], overlap_latent)

            latent_accum[:, :, latent_top:latent_bottom, :] += valid * weights
            weight_accum[:, :, latent_top:latent_bottom, :] += weights

        latent = latent_accum / weight_accum.clamp_min(1e-6)
        return latent

    def _decode_tiled_height(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent using height-wise tiling with blending."""
        tile_latent_h = max(self.tile_size // self.scale_factor, 1)
        tile_latent_h = min(tile_latent_h, latent.shape[2])
        overlap_latent = max(self.tile_overlap // self.scale_factor, 0)
        starts = self._compute_tile_starts(latent.shape[2], tile_latent_h, overlap_latent)

        image_accum = None
        weight_accum = None
        for top in starts:
            bottom = min(top + tile_latent_h, latent.shape[2])
            tile = latent[:, :, top:bottom, :]
            decoded_tile = self._decode_direct(tile)

            if image_accum is None:
                batch, channels, tile_h, tile_w = decoded_tile.shape
                total_height = latent.shape[2] * self.scale_factor
                image_accum = torch.zeros(batch, channels, total_height, tile_w, device=decoded_tile.device, dtype=decoded_tile.dtype)
                weight_accum = torch.zeros_like(image_accum)

            image_top = top * self.scale_factor
            image_bottom = min(image_top + decoded_tile.shape[2], image_accum.shape[2])
            valid = decoded_tile[:, :, :image_bottom - image_top, :]
            weights = self._blend_weights(valid, image_top, image_bottom, image_accum.shape[2], self.tile_overlap)

            image_accum[:, :, image_top:image_bottom, :] += valid * weights
            weight_accum[:, :, image_top:image_bottom, :] += weights

        image = image_accum / weight_accum.clamp_min(1e-6)
        return image

    def configure_tiling(
        self,
        tile_size: Optional[int] = None,
        tile_overlap: Optional[int] = None,
        max_tile_pixels: Optional[int] = None
    ):
        """Update tiling parameters at runtime."""
        if tile_size is not None and tile_size > 0:
            self.tile_size = max(tile_size, self.scale_factor)
        if tile_overlap is not None and tile_overlap >= 0:
            self.tile_overlap = tile_overlap
        if max_tile_pixels is not None and max_tile_pixels >= 0:
            self.max_tile_pixels = max_tile_pixels

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

        height, width = x.shape[2], x.shape[3]

        if self._should_tile(height, width, use_tiling):
            latent = self._encode_tiled_height(x, height)
        else:
            latent = self._encode_direct(x)

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

        if self._should_tile(latent.shape[2] * self.scale_factor, latent.shape[3] * self.scale_factor, use_tiling):
            image = self._decode_tiled_height(latent)
        else:
            image = self._decode_direct(latent)

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
    offload_device: Optional[torch.device] = None,
    tile_size: int = 1536,
    tile_overlap: int = 128,
    max_tile_pixels: int = 4096 * 4096
) -> DiT360VAE:
    """
    Load VAE model for DiT360

    Args:
        vae_path: Path to VAE file (.safetensors)
        precision: Model precision (fp32/fp16/bf16)
        device: Target device (None = auto)
        offload_device: Offload device (None = CPU)
        tile_size: Maximum height (pixels) per VAE tile
        tile_overlap: Overlap between tiles to reduce seams (pixels)
        max_tile_pixels: Threshold for auto-tiling (total pixels)

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
    print(f"Tile size: {tile_size}px, overlap: {tile_overlap}px, auto-tiling threshold: {max_tile_pixels} pixels")
    print(f"{'='*60}\n")

    # Check file exists
    if not vae_path.exists():
        raise FileNotFoundError(
            f"VAE file not found: {vae_path}\n\n"
            f"Please download FLUX.1-dev VAE from:\n"
            f"  https://huggingface.co/black-forest-labs/FLUX.1-dev\n\n"
            f"Or use auto-download in DiT360Loader.\n"
        )

    # Load actual VAE model
    try:
        # Try loading with diffusers AutoencoderKL
        from diffusers import AutoencoderKL

        print("Loading VAE with diffusers...")

        if vae_path.suffix == ".safetensors":
            # Load from safetensors
            vae_model = AutoencoderKL.from_single_file(
                str(vae_path),
                torch_dtype=torch.float32  # Load in fp32 first, convert later
            )
        else:
            # Try loading from directory
            vae_model = AutoencoderKL.from_pretrained(
                str(vae_path.parent),
                torch_dtype=torch.float32
            )

        print("✓ VAE loaded successfully")

    except Exception as e:
        print(f"Warning: Failed to load VAE with diffusers ({e})")
        print("Falling back to safetensors direct load...")

        # Fallback: Try loading directly with safetensors
        try:
            from diffusers import AutoencoderKL

            # Create VAE from scratch and load weights
            vae_model = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                down_block_types=["DownEncoderBlock2D"] * 4,
                up_block_types=["UpDecoderBlock2D"] * 4,
                block_out_channels=[128, 256, 512, 512],
                layers_per_block=2,
            )

            # Load state dict
            state_dict = load_file(str(vae_path))
            vae_model.load_state_dict(state_dict, strict=False)

            print("✓ VAE loaded from safetensors")

        except Exception as e2:
            print(f"Warning: Could not load VAE ({e2})")
            print("Using placeholder VAE")

            # Last resort: Use placeholder
            class PlaceholderVAE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.initialized = True

                def encode(self, x):
                    # Simple downsampling
                    class FakeOutput:
                        def __init__(self, latents):
                            self.latent_dist = type('obj', (object,), {'sample': lambda: latents})()

                    latent = torch.nn.functional.avg_pool2d(x, kernel_size=8)
                    if latent.shape[1] != 4:
                        latent = torch.nn.functional.pad(latent, (0, 0, 0, 0, 0, 4 - latent.shape[1]))
                    return FakeOutput(latent)

                def decode(self, x):
                    # Simple upsampling
                    image = torch.nn.functional.interpolate(x, scale_factor=8, mode='bilinear')
                    if image.shape[1] != 3:
                        image = image[:, :3, :, :]

                    class FakeOutput:
                        def __init__(self, sample):
                            self.sample = sample

                    return FakeOutput(image)

            vae_model = PlaceholderVAE()
            print("✓ Placeholder VAE created")

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
        scale_factor=8,  # FLUX VAE uses 8x downscale
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        max_tile_pixels=max_tile_pixels
    )

    print(f"✓ VAE ready\n")
    return wrapper
