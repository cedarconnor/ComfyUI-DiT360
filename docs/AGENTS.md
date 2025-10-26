# Agents.md - ComfyUI-DiT360 Implementation Guide for Claude Code

## Project Context

You are building **ComfyUI-DiT360**, a custom node pack that integrates the DiT360 panoramic image generation model into ComfyUI. DiT360 is a 12-billion-parameter diffusion transformer based on FLUX.1-dev that generates high-fidelity 360-degree equirectangular panoramic images.

**Key Requirements**:
- Windows compatibility (primary platform)
- ComfyUI node-based architecture integration
- Efficient memory management (16-24GB VRAM)
- Support for text-to-panorama, inpainting, and outpainting
- Interactive 360° panorama viewing
- Seamless edge wrapping for equirectangular format

**Critical Constraints**:
- Must use pathlib.Path for ALL path operations (Windows compatibility)
- Must enforce 2:1 aspect ratio for equirectangular panoramas
- Must implement circular padding for seamless wraparound
- Must integrate with ComfyUI's memory management (don't override)
- Must avoid PyTorch version conflicts (loose version constraints)

**Reference Documents**:
- Full technical specifications in `TECHNICAL_DESIGN.md`
- DiT360 paper: https://arxiv.org/abs/2510.11712
- DiT360 repo: https://github.com/Insta360-Research-Team/DiT360
- ComfyUI docs: https://docs.comfy.org/

---

## Implementation Strategy

### Development Approach
1. **Start minimal, iterate progressively**: Begin with basic structure and add complexity
2. **Test early and often**: Validate each component before building the next
3. **Follow ComfyUI patterns**: Study existing node packs (OpenDiTWrapper, WanVideoWrapper)
4. **Windows-first development**: Use pathlib, test paths, avoid Linux-specific code
5. **Document as you build**: Add docstrings, comments, and README sections immediately

### Critical Success Factors
- ✅ Nodes load without errors in ComfyUI
- ✅ Models download/load successfully
- ✅ Basic generation produces 2048×1024 panoramas
- ✅ Windows paths work correctly
- ✅ Memory stays within 24GB VRAM bounds
- ✅ Edge wrapping is seamless

---

## Phase-by-Phase Implementation Guide

## PHASE 1: Foundation Setup

### Objective
Create basic project structure that loads in ComfyUI without errors.

### Steps

#### 1.1 Create Directory Structure
```bash
mkdir -p ComfyUI-DiT360
cd ComfyUI-DiT360
mkdir -p dit360 utils web/js examples tests docs
touch __init__.py nodes.py requirements.txt install.py README.md LICENSE
```

#### 1.2 Implement `__init__.py` (Entry Point)
```python
"""
ComfyUI-DiT360: Panoramic image generation using DiT360 model
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Optional: Register custom model paths
import folder_paths
import os
from pathlib import Path

# Add dit360 model directory
dit360_models_dir = Path(folder_paths.models_dir) / "dit360"
dit360_models_dir.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path("dit360", str(dit360_models_dir))

# Version info
__version__ = "0.1.0"
__author__ = "Your Name"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"ComfyUI-DiT360 v{__version__} loaded")
print(f"Model directory: {dit360_models_dir}")
```

#### 1.3 Implement `nodes.py` (Skeleton Nodes)
```python
"""
Core node implementations for ComfyUI-DiT360
"""

import torch
import folder_paths
from pathlib import Path
import comfy.model_management as mm
import comfy.utils

# ====================================================================
# MODEL LOADER NODES
# ====================================================================

class DiT360ModelLoader:
    """Load DiT360 transformer model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("dit360"), {"default": ""}),
                "precision": (["fp32", "fp16", "bf16", "fp8"], {"default": "fp16"}),
            }
        }
    
    RETURN_TYPES = ("DIT360_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "DiT360/loaders"
    
    def load_model(self, model_name, precision):
        """Load DiT360 model from file"""
        # TODO: Implement actual loading
        print(f"Loading DiT360 model: {model_name} ({precision})")
        return ({"model_name": model_name, "precision": precision},)


class DiT360TextEncoderLoader:
    """Load text encoder for prompt conditioning"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder_name": (["t5-xxl", "clip-l"], {"default": "t5-xxl"}),
                "precision": (["fp32", "fp16"], {"default": "fp16"}),
            }
        }
    
    RETURN_TYPES = ("TEXT_ENCODER",)
    FUNCTION = "load_encoder"
    CATEGORY = "DiT360/loaders"
    
    def load_encoder(self, encoder_name, precision):
        """Load text encoding model"""
        # TODO: Implement actual loading
        print(f"Loading text encoder: {encoder_name} ({precision})")
        return ({"encoder_name": encoder_name},)


class DiT360VAELoader:
    """Load VAE for latent encoding/decoding"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), {"default": ""}),
                "precision": (["fp32", "fp16"], {"default": "fp16"}),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "DiT360/loaders"
    
    def load_vae(self, vae_name, precision):
        """Load VAE model"""
        # TODO: Implement actual loading
        print(f"Loading VAE: {vae_name} ({precision})")
        return ({"vae_name": vae_name},)


# ====================================================================
# GENERATION NODES
# ====================================================================

class DiT360Sampler:
    """Core panorama generation node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DIT360_MODEL",),
                "text_encoder": ("TEXT_ENCODER",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 150}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "generate"
    CATEGORY = "DiT360"
    
    def generate(self, model, text_encoder, vae, prompt, negative_prompt,
                 width, height, steps, cfg_scale, seed):
        """Generate panoramic image"""
        # TODO: Implement actual generation
        print(f"Generating {width}×{height} panorama: '{prompt[:50]}...'")
        
        # Create dummy output for testing
        latent = torch.zeros(1, 4, height // 8, width // 8)
        image = torch.zeros(1, height, width, 3)
        
        return ({"samples": latent}, image)


# ====================================================================
# UTILITY NODES
# ====================================================================

class Equirect360Validator:
    """Validate and fix equirectangular format"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enforce_ratio": ("BOOLEAN", {"default": True}),
                "fix_mode": (["none", "crop", "pad", "stretch"], {"default": "pad"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "validate"
    CATEGORY = "DiT360/utils"
    
    def validate(self, image, enforce_ratio, fix_mode):
        """Validate equirectangular format"""
        # TODO: Implement validation
        print(f"Validating equirectangular image: {image.shape}")
        return (image,)


class Equirect360EdgeBlender:
    """Apply seamless edge blending"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend_width": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_edges"
    CATEGORY = "DiT360/utils"
    
    def blend_edges(self, image, blend_width):
        """Blend left and right edges for seamless wraparound"""
        # TODO: Implement edge blending
        print(f"Blending edges with width: {blend_width}")
        return (image,)


# ====================================================================
# NODE REGISTRATION
# ====================================================================

NODE_CLASS_MAPPINGS = {
    "DiT360ModelLoader": DiT360ModelLoader,
    "DiT360TextEncoderLoader": DiT360TextEncoderLoader,
    "DiT360VAELoader": DiT360VAELoader,
    "DiT360Sampler": DiT360Sampler,
    "Equirect360Validator": Equirect360Validator,
    "Equirect360EdgeBlender": Equirect360EdgeBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360ModelLoader": "DiT360 Model Loader",
    "DiT360TextEncoderLoader": "DiT360 Text Encoder Loader",
    "DiT360VAELoader": "DiT360 VAE Loader",
    "DiT360Sampler": "DiT360 Sampler",
    "Equirect360Validator": "Equirect360 Validator",
    "Equirect360EdgeBlender": "Equirect360 Edge Blender",
}
```

#### 1.4 Create `requirements.txt`
```txt
# Core dependencies (loose constraints for compatibility)
torch>=2.0.0,<3.0.0
torchvision>=0.15.0
transformers>=4.28.1
diffusers>=0.25.0
safetensors>=0.4.2
accelerate>=0.26.0
huggingface-hub>=0.20.0

# Optional dependencies
# opencv-python>=4.8.0
# Pillow>=10.0.0
```

#### 1.5 Create Basic `README.md`
```markdown
# ComfyUI-DiT360

Generate high-fidelity 360-degree panoramic images using DiT360 in ComfyUI.

## Installation

1. Navigate to ComfyUI's custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/ComfyUI-DiT360.git
```

3. Install dependencies:
```bash
cd ComfyUI-DiT360
pip install -r requirements.txt
```

4. Restart ComfyUI

## Usage

Coming soon...

## Requirements

- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8 or newer
- Python 3.9 - 3.12

## License

Apache License 2.0
```

#### 1.6 Validation Checklist
- [ ] Files created in correct structure
- [ ] `__init__.py` imports work without errors
- [ ] Copy folder to `ComfyUI/custom_nodes/`
- [ ] Start ComfyUI
- [ ] Check console for "ComfyUI-DiT360 v0.1.0 loaded"
- [ ] Nodes appear in node menu under "DiT360" category
- [ ] Right-click → Add Node → DiT360/ shows all nodes
- [ ] Nodes can be added to workflow (even if non-functional)

---

## PHASE 2: Model Loading Infrastructure

### Objective
Implement actual model loading with Hugging Face integration.

### Critical Implementation Details

#### 2.1 Windows Path Handling Pattern (USE EVERYWHERE)
```python
from pathlib import Path

# ✅ CORRECT - Cross-platform compatible
model_dir = Path(folder_paths.models_dir) / "dit360"
config_file = model_dir / "config.json"

# ✅ CORRECT - Windows-specific if needed
windows_path = Path(r"C:\Users\Name\ComfyUI\models\dit360")

# ❌ WRONG - Will break on Windows
model_dir = "C:\Users\Name\ComfyUI\models\dit360"  # Escape sequences!
```

#### 2.2 Create `dit360/model.py`
```python
"""
DiT360 model architecture and loading utilities
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file
import json
from typing import Dict, Optional
import comfy.model_management as mm


class DiT360Model(nn.Module):
    """
    DiT360 Transformer Model
    Based on FLUX.1-dev architecture with panoramic adaptations
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # TODO: Initialize transformer architecture
        # This will be a FLUX.1-dev style transformer with:
        # - 12B parameters
        # - Attention layers with circular padding support
        # - RoPE positional embeddings adapted for spherical geometry
        
        print(f"Initialized DiT360Model with config: {config}")
    
    def forward(self, latent, conditioning, timestep):
        """Forward pass through transformer"""
        # TODO: Implement forward pass
        return latent


def load_dit360_model(
    model_path: Path,
    precision: str = "fp16",
    device: Optional[torch.device] = None
) -> DiT360Model:
    """
    Load DiT360 model from safetensors file
    
    Args:
        model_path: Path to model file (.safetensors)
        precision: Model precision (fp32, fp16, bf16, fp8)
        device: Target device (None = auto-detect)
    
    Returns:
        Loaded DiT360Model instance
    """
    if device is None:
        device = mm.get_torch_device()
    
    print(f"Loading DiT360 model from: {model_path}")
    print(f"Precision: {precision}, Device: {device}")
    
    # Load config
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config if not found
        config = {
            "model_type": "dit360",
            "params": "12B",
            "latent_channels": 4,
            "attention_heads": 16,
        }
        print(f"Warning: config.json not found, using defaults")
    
    # Load state dict
    print("Loading model weights...")
    state_dict = load_file(str(model_path))
    
    # Initialize model
    model = DiT360Model(config)
    model.load_state_dict(state_dict, strict=False)
    
    # Apply precision conversion
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
    }
    
    target_dtype = dtype_map.get(precision, torch.float16)
    model = model.to(dtype=target_dtype)
    
    print(f"Model loaded successfully ({precision})")
    
    return model


def download_dit360_model(model_name: str, save_dir: Path):
    """
    Download DiT360 model from Hugging Face Hub
    
    Args:
        model_name: Name of model on HF Hub
        save_dir: Directory to save model
    """
    from huggingface_hub import hf_hub_download
    
    print(f"Downloading {model_name} from Hugging Face Hub...")
    
    repo_id = "Insta360-Research/DiT360-Panorama-Image-Generation"
    
    try:
        # Download main model file
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_name}.safetensors",
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        
        # Download config if exists
        try:
            config_file = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                local_dir=save_dir,
                local_dir_use_symlinks=False
            )
        except:
            print("Config file not found, will use defaults")
        
        print(f"Download complete: {model_file}")
        return Path(model_file)
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model: {e}\n"
            f"Please download manually from:\n"
            f"https://huggingface.co/{repo_id}"
        )
```

#### 2.3 Update `DiT360ModelLoader` in `nodes.py`
```python
from .dit360.model import load_dit360_model, download_dit360_model

class DiT360ModelLoader:
    """Load DiT360 transformer model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("dit360"), {"default": ""}),
                "precision": (["fp32", "fp16", "bf16", "fp8"], {"default": "fp16"}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("DIT360_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "DiT360/loaders"
    
    def load_model(self, model_name, precision, auto_download):
        """Load DiT360 model from file"""
        from pathlib import Path
        
        # Get model path
        model_path = folder_paths.get_full_path("dit360", model_name)
        
        if not model_path or not Path(model_path).exists():
            if auto_download:
                print(f"Model not found locally, downloading...")
                model_dir = Path(folder_paths.get_folder_paths("dit360")[0])
                model_path = download_dit360_model(model_name, model_dir)
            else:
                raise FileNotFoundError(
                    f"Model not found: {model_name}\n"
                    f"Enable auto_download or download manually from:\n"
                    f"https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation\n"
                    f"Place in: {folder_paths.get_folder_paths('dit360')[0]}"
                )
        
        # Load model
        model = load_dit360_model(Path(model_path), precision=precision)
        
        return ({"model": model, "precision": precision},)
```

#### 2.4 Validation Checklist
- [ ] Model downloads from Hugging Face (if not present)
- [ ] Model loads without errors
- [ ] Correct precision conversion applied
- [ ] Memory usage reasonable (<5GB for model loading)
- [ ] Works on Windows (path handling correct)
- [ ] Clear error messages for download failures

---

## PHASE 3: Core Generation Pipeline

### Objective
Implement basic panorama generation (no geometric losses yet).

### Critical Implementation Details

#### 3.1 Circular Padding Implementation

Create `utils/padding.py`:
```python
"""
Circular padding utilities for seamless panorama wraparound
"""

import torch
import torch.nn.functional as F


def apply_circular_padding(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Apply circular padding to create seamless wraparound at panorama edges
    
    Args:
        tensor: Input tensor (B, C, H, W) or (B, H, W, C)
        padding: Number of pixels to pad on left/right
    
    Returns:
        Padded tensor with wraparound continuity
    """
    # Handle different tensor formats
    if tensor.ndim == 4:
        if tensor.shape[1] <= 4:  # (B, C, H, W) - latent format
            left_edge = tensor[:, :, :, :padding]
            right_edge = tensor[:, :, :, -padding:]
            padded = torch.cat([right_edge, tensor, left_edge], dim=3)
        else:  # (B, H, W, C) - image format
            left_edge = tensor[:, :, :padding, :]
            right_edge = tensor[:, :, -padding:, :]
            padded = torch.cat([right_edge, tensor, left_edge], dim=2)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    return padded


def remove_circular_padding(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """Remove circular padding after processing"""
    if tensor.ndim == 4:
        if tensor.shape[1] <= 4:  # (B, C, H, W)
            return tensor[:, :, :, padding:-padding]
        else:  # (B, H, W, C)
            return tensor[:, :, padding:-padding, :]
    return tensor


def circular_conv2d(input: torch.Tensor, weight: torch.Tensor, 
                    bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    2D convolution with circular padding on width dimension
    Ensures convolutions respect panorama wraparound
    """
    # Apply circular padding on width (left/right edges)
    if padding > 0:
        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        else:
            pad_h, pad_w = padding
        
        # Pad height normally, width circularly
        input = F.pad(input, (0, 0, pad_h, pad_h), mode='constant', value=0)
        input = apply_circular_padding(input, pad_w)
        padding = 0  # Already padded
    
    # Standard convolution
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
```

#### 3.2 Equirectangular Validation

Create `utils/equirect.py`:
```python
"""
Equirectangular projection utilities
"""

import torch
import math
from typing import Tuple


def validate_aspect_ratio(width: int, height: int, tolerance: float = 0.01) -> bool:
    """
    Validate if dimensions are 2:1 ratio (equirectangular requirement)
    
    Args:
        width: Image width
        height: Image height
        tolerance: Acceptable deviation from 2:1 ratio
    
    Returns:
        True if valid 2:1 ratio
    """
    ratio = width / height
    return abs(ratio - 2.0) < tolerance


def fix_aspect_ratio(image: torch.Tensor, mode: str = "pad", 
                     target_width: int = None) -> torch.Tensor:
    """
    Fix image to 2:1 aspect ratio
    
    Args:
        image: Input image tensor (B, H, W, C)
        mode: How to fix ratio - 'pad', 'crop', or 'stretch'
        target_width: Desired width (height will be width/2)
    
    Returns:
        Fixed image with 2:1 ratio
    """
    B, H, W, C = image.shape
    
    if validate_aspect_ratio(W, H):
        if target_width and W != target_width:
            target_height = target_width // 2
            return torch.nn.functional.interpolate(
                image.permute(0, 3, 1, 2),  # (B, C, H, W)
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        return image
    
    target_height = W // 2
    
    if mode == "pad":
        # Add black bars top/bottom
        pad_total = target_height - H
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return torch.nn.functional.pad(
            image.permute(0, 3, 1, 2),
            (0, 0, pad_top, pad_bottom),
            mode='constant',
            value=0
        ).permute(0, 2, 3, 1)
    
    elif mode == "crop":
        # Center crop
        crop_start = (H - target_height) // 2
        return image[:, crop_start:crop_start+target_height, :, :]
    
    elif mode == "stretch":
        # Resize (distorts content)
        return torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),
            size=(target_height, W),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def blend_edges(image: torch.Tensor, blend_width: int = 10, 
                mode: str = "cosine") -> torch.Tensor:
    """
    Blend left and right edges for seamless wraparound
    
    Args:
        image: Input image (B, H, W, C)
        blend_width: Width of blend region in pixels
        mode: Blending function - 'linear', 'cosine', or 'smooth'
    
    Returns:
        Image with blended edges
    """
    B, H, W, C = image.shape
    
    if blend_width <= 0 or blend_width >= W // 2:
        return image
    
    left_edge = image[:, :, :blend_width, :]
    right_edge = image[:, :, -blend_width:, :]
    
    # Create blend weights
    if mode == "linear":
        weights = torch.linspace(0, 1, blend_width, device=image.device)
    elif mode == "cosine":
        t = torch.linspace(0, math.pi, blend_width, device=image.device)
        weights = (1 - torch.cos(t)) / 2
    elif mode == "smooth":
        weights = torch.linspace(0, 1, blend_width, device=image.device) ** 2
    else:
        raise ValueError(f"Unknown blend mode: {mode}")
    
    weights = weights.view(1, 1, -1, 1)
    
    # Blend edges
    blended_left = left_edge * (1 - weights) + right_edge * weights
    blended_right = right_edge * (1 - weights) + left_edge * weights
    
    # Apply blending
    result = image.clone()
    result[:, :, :blend_width, :] = blended_left
    result[:, :, -blend_width:, :] = blended_right
    
    return result


def check_edge_continuity(image: torch.Tensor, threshold: float = 0.05) -> bool:
    """
    Check if left and right edges are continuous (for validation)
    
    Args:
        image: Input image (B, H, W, C)
        threshold: Maximum allowed difference (0-1 scale)
    
    Returns:
        True if edges are continuous within threshold
    """
    left_edge = image[:, :, 0, :]
    right_edge = image[:, :, -1, :]
    
    diff = torch.abs(left_edge - right_edge).mean()
    return diff.item() < threshold
```

#### 3.3 Update `DiT360Sampler` with Basic Generation

Update in `nodes.py`:
```python
from .utils.padding import apply_circular_padding, remove_circular_padding
from .utils.equirect import validate_aspect_ratio, blend_edges
import comfy.utils

class DiT360Sampler:
    """Core panorama generation node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DIT360_MODEL",),
                "text_encoder": ("TEXT_ENCODER",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 150}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "circular_padding": ("INT", {"default": 10, "min": 0, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "generate"
    CATEGORY = "DiT360"
    
    def generate(self, model, text_encoder, vae, prompt, negative_prompt,
                 width, height, steps, cfg_scale, seed, circular_padding):
        """Generate panoramic image"""
        
        # Validate aspect ratio
        if not validate_aspect_ratio(width, height):
            print(f"Warning: {width}×{height} is not 2:1 ratio. "
                  f"Recommended: {width}×{width//2}")
        
        # Get device
        device = mm.get_torch_device()
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # TODO: Encode prompts using text_encoder
        # For now, create dummy conditioning
        batch_size = 1
        conditioning = torch.randn(batch_size, 77, 768, device=device)
        
        # Initialize latent noise
        latent_height = height // 8  # VAE 8x downscale
        latent_width = width // 8
        latent = torch.randn(
            batch_size, 4, latent_height, latent_width,
            device=device,
            dtype=torch.float16
        )
        
        # Sampling loop with progress bar
        pbar = comfy.utils.ProgressBar(steps)
        
        for step in range(steps):
            # Apply circular padding
            if circular_padding > 0:
                latent_padded = apply_circular_padding(latent, circular_padding)
            else:
                latent_padded = latent
            
            # TODO: Model forward pass
            # noise_pred = model['model'](latent_padded, conditioning, timestep)
            
            # For now, just add small noise
            noise = torch.randn_like(latent) * 0.01
            latent = latent + noise
            
            pbar.update(1)
        
        # Remove padding
        if circular_padding > 0:
            latent = remove_circular_padding(latent, circular_padding)
        
        # TODO: VAE decode
        # image = vae.decode(latent)
        
        # For now, create dummy image
        image = torch.rand(batch_size, height, width, 3)
        
        # Apply edge blending for seamless wraparound
        image = blend_edges(image, blend_width=10)
        
        return ({"samples": latent}, image)
```

#### 3.4 Validation Checklist
- [ ] Aspect ratio validation works
- [ ] Circular padding applies correctly
- [ ] Edge blending produces seamless result
- [ ] Progress bar displays during generation
- [ ] Seed produces reproducible results
- [ ] Memory usage stays reasonable
- [ ] No CUDA errors

---

## PHASE 4: Windows Testing & Path Validation

### Objective
Ensure all code works correctly on Windows.

### Critical Testing Points

#### 4.1 Path Handling Test Script

Create `tests/test_windows_paths.py`:
```python
"""
Windows path handling tests
"""

import sys
import os
from pathlib import Path

def test_path_handling():
    """Test various path scenarios"""
    
    print("=" * 60)
    print("Windows Path Handling Tests")
    print("=" * 60)
    
    # Test 1: Backslash paths
    print("\n1. Testing backslash paths...")
    try:
        # These should all work
        path1 = Path(r"C:\Users\Test\ComfyUI\models")
        path2 = Path("C:/Users/Test/ComfyUI/models")
        path3 = Path("C:\\Users\\Test\\ComfyUI\\models")
        print("   ✓ All path formats accepted")
    except Exception as e:
        print(f"   ✗ Path format error: {e}")
    
    # Test 2: Path concatenation
    print("\n2. Testing path concatenation...")
    try:
        base = Path("C:/ComfyUI")
        models = base / "models" / "dit360"
        print(f"   Result: {models}")
        print("   ✓ Path concatenation works")
    except Exception as e:
        print(f"   ✗ Concatenation error: {e}")
    
    # Test 3: Spaces in paths
    print("\n3. Testing paths with spaces...")
    try:
        path_with_spaces = Path("C:/Program Files/ComfyUI/models/dit360")
        print(f"   Result: {path_with_spaces}")
        print("   ✓ Spaces handled correctly")
    except Exception as e:
        print(f"   ✗ Spaces error: {e}")
    
    # Test 4: Long paths
    print("\n4. Testing long paths...")
    long_name = "a" * 200
    long_path = Path("C:/ComfyUI/models") / long_name / "model.safetensors"
    if len(str(long_path)) > 260:
        print(f"   Path length: {len(str(long_path))} (exceeds 260)")
        print("   ⚠ May require long path support enabled in Windows")
    else:
        print(f"   Path length: {len(str(long_path))} (OK)")
    
    # Test 5: Case sensitivity
    print("\n5. Testing case sensitivity...")
    test_dir = Path("./test_case")
    test_dir.mkdir(exist_ok=True)
    
    file1 = test_dir / "TestFile.txt"
    file1.touch()
    
    # Try to find with different case
    file2 = test_dir / "testfile.txt"
    if file2.exists():
        print("   ✓ Case-insensitive filesystem detected (Windows)")
    else:
        print("   ✗ Case-sensitive filesystem (Linux/Mac)")
    
    # Cleanup
    file1.unlink()
    test_dir.rmdir()
    
    print("\n" + "=" * 60)
    print("All path tests completed")
    print("=" * 60)


if __name__ == "__main__":
    test_path_handling()
```

Run on Windows:
```bash
python tests/test_windows_paths.py
```

#### 4.2 File Locking Prevention Pattern

**ALWAYS use context managers for file operations**:

```python
# ✅ CORRECT - File automatically closed
with open(file_path, 'r') as f:
    content = f.read()

# ✅ CORRECT - Using pathlib
content = Path(file_path).read_text()

# ✅ CORRECT - Safetensors with context manager
from safetensors.torch import safe_open
with safe_open(model_path, framework="pt") as f:
    state_dict = {k: f.get_tensor(k) for k in f.keys()}

# ❌ WRONG - File may stay locked
f = open(file_path, 'r')
content = f.read()
# f.close() might not be reached if error occurs
```

#### 4.3 Windows Validation Checklist
- [ ] All paths use pathlib.Path
- [ ] No hardcoded backslash strings
- [ ] Files use context managers (with statement)
- [ ] Long path support documented
- [ ] Case-insensitive file finding works
- [ ] No file locking errors
- [ ] Works with portable ComfyUI
- [ ] Spaces in paths handled correctly

---

## Common Pitfalls & Solutions

### Pitfall 1: PyTorch Version Conflicts

**Problem**: Users have different PyTorch versions installed, causing conflicts.

**Solution**:
```python
# In requirements.txt - use loose constraints
torch>=2.0.0,<3.0.0  # Not torch==2.1.2

# In install.py - check existing version
import torch
import subprocess
import sys

def smart_install():
    """Install dependencies without breaking existing PyTorch"""
    
    torch_version = torch.__version__
    print(f"Existing PyTorch: {torch_version}")
    
    # Don't reinstall PyTorch
    deps = [
        "transformers>=4.28.1",
        "diffusers>=0.25.0",
        "safetensors>=0.4.2",
    ]
    
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep])
```

### Pitfall 2: CUDA Out of Memory

**Problem**: Model exceeds available VRAM.

**Solution**:
```python
def handle_oom(func):
    """Decorator to gracefully handle OOM errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Insufficient VRAM. Try:\n"
                "• Lower precision (fp16 → fp8)\n"
                "• Smaller resolution (2048×1024 → 1024×512)\n"
                "• Enable model offloading\n"
                f"Current VRAM: {torch.cuda.memory_allocated()/(1024**3):.1f}GB"
            )
    return wrapper
```

### Pitfall 3: Incorrect Tensor Shapes

**Problem**: ComfyUI expects specific tensor formats.

**Solution**:
```python
def ensure_comfyui_format(image: torch.Tensor) -> torch.Tensor:
    """
    Ensure image is in ComfyUI format: (B, H, W, C)
    Values in range [0, 1]
    """
    # Handle different input formats
    if image.ndim == 3:  # (H, W, C)
        image = image.unsqueeze(0)  # Add batch dim
    
    if image.shape[1] <= 4:  # (B, C, H, W) - need to transpose
        image = image.permute(0, 2, 3, 1)  # → (B, H, W, C)
    
    # Ensure [0, 1] range
    if image.min() < 0 or image.max() > 1:
        image = torch.clamp(image, 0, 1)
    
    return image
```

### Pitfall 4: Missing Error Context

**Problem**: Errors don't provide enough information to debug.

**Solution**:
```python
def load_model_with_context(model_path):
    """Load model with detailed error context"""
    try:
        state_dict = load_file(str(model_path))
        return state_dict
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Directory exists: {model_path.parent.exists()}\n"
            f"Directory contents: {list(model_path.parent.glob('*')) if model_path.parent.exists() else 'N/A'}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model: {model_path}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {e}\n"
            f"File size: {model_path.stat().st_size if model_path.exists() else 'N/A'}"
        )
```

---

## Code Quality Standards

### Docstring Format
```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Brief one-line description
    
    Detailed explanation if needed. Can be multiple paragraphs.
    Explain the purpose, algorithm, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ErrorType: When this error occurs
    
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        "output"
    """
    # Implementation
    pass
```

### Type Hints
```python
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path

# ✅ CORRECT - Type hints provided
def load_model(
    model_path: Path,
    precision: str = "fp16",
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    pass

# ❌ WRONG - No type hints
def load_model(model_path, precision="fp16", device=None):
    pass
```

### Error Handling
```python
# ✅ CORRECT - Specific exceptions with context
def validate_config(config: Dict):
    if "model_type" not in config:
        raise KeyError(
            "Missing required key 'model_type' in config\n"
            f"Available keys: {list(config.keys())}"
        )

# ❌ WRONG - Generic exception without context
def validate_config(config):
    if "model_type" not in config:
        raise Exception("Bad config")
```

### Comments
```python
# ✅ CORRECT - Explain WHY, not WHAT
# Use circular padding to ensure seamless wraparound at panorama edges
# Without this, left and right edges won't match when viewer wraps
padded = apply_circular_padding(latent, padding=10)

# ❌ WRONG - Restating the obvious
# Apply circular padding with padding of 10
padded = apply_circular_padding(latent, padding=10)
```

---

## Testing Workflow

### Before Each Commit
1. **Run basic tests**:
   ```bash
   python -m pytest tests/
   ```

2. **Test in ComfyUI**:
   - Restart ComfyUI
   - Add nodes to workflow
   - Execute workflow
   - Check console for errors

3. **Memory check**:
   ```python
   # Add to test
   import torch
   torch.cuda.reset_peak_memory_stats()
   # ... run generation ...
   peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
   print(f"Peak VRAM: {peak_mem:.2f} GB")
   assert peak_mem < 24, "VRAM usage too high"
   ```

4. **Windows path check**:
   ```bash
   python tests/test_windows_paths.py
   ```

### Before Release
- [ ] All tests pass
- [ ] Example workflows work
- [ ] Documentation complete
- [ ] No hardcoded paths
- [ ] Memory usage within limits
- [ ] Windows compatibility verified
- [ ] README has installation instructions
- [ ] Changelog updated

---

## Implementation Priority Order

### High Priority (Must Have)
1. ✅ Basic structure that loads in ComfyUI
2. ⬜ Model loading from HuggingFace
3. ⬜ Text encoding (prompt conditioning)
4. ⬜ VAE encode/decode
5. ⬜ Basic sampling loop
6. ⬜ Circular padding
7. ⬜ Edge blending
8. ⬜ 2:1 aspect ratio validation
9. ⬜ Windows path handling

### Medium Priority (Should Have)
10. ⬜ Advanced sampler with geometric losses
11. ⬜ Inpainting support
12. ⬜ Multiple precision support (fp8, bf16)
13. ⬜ Progress reporting
14. ⬜ Model offloading
15. ⬜ 360° preview viewer

### Low Priority (Nice to Have)
16. ⬜ Outpainting
17. ⬜ Image-to-panorama
18. ⬜ LoRA support
19. ⬜ Cubemap conversion
20. ⬜ Video panorama (future)

---

## Quick Reference: Key Patterns

### Path Handling
```python
from pathlib import Path
model_dir = Path(folder_paths.models_dir) / "dit360"
```

### Tensor Format
```python
# ComfyUI IMAGE format: (B, H, W, C), values [0, 1]
# ComfyUI LATENT format: {"samples": tensor (B, C, H, W)}
```

### Progress Bar
```python
pbar = comfy.utils.ProgressBar(total_steps)
for step in range(total_steps):
    # ... work ...
    pbar.update(1)
```

### Memory Management
```python
device = mm.get_torch_device()
model.to(device)
# ComfyUI handles caching automatically
```

### Error Messages
```python
raise ValueError(
    f"Clear description of what went wrong\n"
    f"Current state: {state}\n"
    f"Expected: {expected}\n"
    f"Suggestion: Try doing X instead"
)
```

---

## Final Notes

**Remember**:
- Start simple, add complexity progressively
- Test on Windows frequently (primary platform)
- Use pathlib.Path EVERYWHERE
- Follow ComfyUI patterns from existing nodes
- Document as you go
- Clear error messages save debugging time

**When Stuck**:
1. Check existing node packs (OpenDiTWrapper, WanVideoWrapper)
2. Read ComfyUI source code for patterns
3. Test isolated components separately
4. Add debug prints liberally
5. Check TECHNICAL_DESIGN.md for specifications

**Success Metrics**:
- Loads without errors ✓
- Generates 2048×1024 panoramas ✓
- Edges wrap seamlessly ✓
- Works on Windows ✓
- Memory < 24GB ✓

Good luck! 🚀
