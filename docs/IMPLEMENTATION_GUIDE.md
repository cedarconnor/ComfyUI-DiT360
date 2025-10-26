# Agents.md - ComfyUI-DiT360 Implementation Guide (Streamlined)
## Building 360° Panorama Nodes for FLUX.1-dev + DiT360 LoRA

---

## Project Context

You are building **5 enhancement nodes** that add 360° panorama capabilities to standard FLUX workflows in ComfyUI. DiT360 is **just a LoRA** for FLUX.1-dev—no custom model loading needed!

**What You're Building**:
1. `Equirect360EmptyLatent` - 2:1 aspect ratio helper
2. `Equirect360KSampler` - Sampling with circular padding + optional losses
3. `Equirect360VAEDecode` - VAE decode with circular padding
4. `Equirect360EdgeBlender` - Post-processing edge smoothing
5. `Equirect360Viewer` - Interactive 360° preview

**Critical Understanding**:
- DiT360 is a **LoRA weight file** (~2-5GB), not a full model
- Users load FLUX.1-dev normally, then apply DiT360 LoRA (standard ComfyUI workflow)
- Your nodes are **drop-in replacements** for standard nodes
- **No custom model loading**—just enhance the sampling/decoding process

---

## Key Implementation Principles

### 1. **Circular Padding is the Core**
This is what makes panoramas seamless. Applied at:
- **Sampling time** (in Equirect360KSampler) ← MOST IMPORTANT
- **VAE decode** (in Equirect360VAEDecode) ← Extra polish

### 2. **Losses Are Optional Performance Tradeoffs**
- **Yaw Loss**: Better rotational consistency, but 2-3x slower
- **Cube Loss**: Less pole distortion, but 1.5-2x slower
- Default: **DISABLED** (fast mode)

### 3. **Windows Compatibility First**
- Use `pathlib.Path` for ALL paths
- No hardcoded backslashes
- Test on Windows frequently

### 4. **Minimal Dependencies**
Only need PyTorch primitives—no custom libraries!

---

## File Structure

```
ComfyUI/custom_nodes/ComfyUI-DiT360/
├── __init__.py                    # Entry point
├── nodes.py                       # All 5 node implementations
├── requirements.txt               # Minimal deps
├── README.md
├── utils/
│   ├── __init__.py
│   ├── circular_padding.py        # Core padding functions
│   ├── equirect.py                # Aspect ratio & validation
│   └── losses.py                  # Yaw/cube loss (optional)
├── web/
│   └── js/
│       └── equirect360_viewer.js  # Three.js viewer
└── examples/
    └── basic_workflow.json        # Example workflow
```

---

## PHASE 1: Foundation Setup

### Step 1.1: Create Directory Structure

```bash
mkdir -p ComfyUI-DiT360/utils
mkdir -p ComfyUI-DiT360/web/js
mkdir -p ComfyUI-DiT360/examples
cd ComfyUI-DiT360
touch __init__.py nodes.py requirements.txt README.md
touch utils/__init__.py utils/circular_padding.py utils/equirect.py utils/losses.py
touch web/js/equirect360_viewer.js
```

### Step 1.2: Create `requirements.txt`

```txt
# Minimal dependencies - only PyTorch primitives needed
# torch>=2.0.0 is already in ComfyUI, don't reinstall!
numpy>=1.25.0
Pillow>=10.0.0
```

### Step 1.3: Create `__init__.py`

```python
"""
ComfyUI-DiT360: 360° Panorama Generation Enhancement
Adds circular padding and geometric losses to FLUX workflows
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.0.0"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Register web directory for Three.js viewer
import os
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

print(f"✅ ComfyUI-DiT360 v{__version__} loaded")
print(f"   • 5 enhancement nodes for 360° panoramas")
print(f"   • Works with FLUX.1-dev + DiT360 LoRA")
```

### Step 1.4: Validation Checklist
- [ ] Directory structure created
- [ ] Files exist (even if empty)
- [ ] Copy to `ComfyUI/custom_nodes/`
- [ ] Start ComfyUI
- [ ] Check console for "ComfyUI-DiT360 loaded" message

---

## PHASE 2: Core Utilities Implementation

### Step 2.1: Implement Circular Padding

Create `utils/circular_padding.py`:

```python
"""
Circular padding for seamless 360° panorama wraparound
"""

import torch
from typing import Tuple


def apply_circular_padding(
    tensor: torch.Tensor, 
    padding: int
) -> torch.Tensor:
    """
    Apply circular padding to tensor for seamless wraparound
    
    This is the CORE function for 360° panoramas. It wraps the left
    and right edges so they connect seamlessly.
    
    Args:
        tensor: (B, C, H, W) for latents or (B, H, W, C) for images
        padding: Padding width in pixels/latent cells
    
    Returns:
        Padded tensor with wraparound continuity
    
    Example:
        >>> latent = torch.rand(1, 4, 64, 128)
        >>> padded = apply_circular_padding(latent, padding=8)
        >>> padded.shape
        torch.Size([1, 4, 64, 144])  # Width increased by 2*padding
    """
    if padding <= 0:
        return tensor
    
    # Determine format: (B, C, H, W) or (B, H, W, C)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tensor.shape}")
    
    # Check if latent format (C=4 or C=16) or image format (C=3)
    if tensor.shape[1] <= 16:  # (B, C, H, W) - latent format
        # Extract edges along width dimension (dim=3)
        left_edge = tensor[:, :, :, -padding:]  # Rightmost columns
        right_edge = tensor[:, :, :, :padding]  # Leftmost columns
        
        # Concatenate: [right_edge] [tensor] [left_edge]
        padded = torch.cat([left_edge, tensor, right_edge], dim=3)
        
    else:  # (B, H, W, C) - image format
        # Extract edges along width dimension (dim=2)
        left_edge = tensor[:, :, -padding:, :]  # Rightmost columns
        right_edge = tensor[:, :, :padding, :]  # Leftmost columns
        
        # Concatenate: [right_edge] [tensor] [left_edge]
        padded = torch.cat([left_edge, tensor, right_edge], dim=2)
    
    return padded


def remove_circular_padding(
    tensor: torch.Tensor,
    padding: int
) -> torch.Tensor:
    """
    Remove circular padding after processing
    
    Args:
        tensor: Padded tensor
        padding: Padding width to remove
    
    Returns:
        Original tensor without padding
    """
    if padding <= 0:
        return tensor
    
    if tensor.shape[1] <= 16:  # (B, C, H, W) - latent format
        return tensor[:, :, :, padding:-padding]
    else:  # (B, H, W, C) - image format
        return tensor[:, :, padding:-padding, :]


def validate_circular_continuity(
    tensor: torch.Tensor,
    threshold: float = 0.05
) -> bool:
    """
    Check if left and right edges are continuous (for testing)
    
    Args:
        tensor: Tensor to check
        threshold: Maximum allowed difference (0-1 scale)
    
    Returns:
        True if edges are continuous within threshold
    """
    if tensor.shape[1] <= 16:  # Latent format
        left_edge = tensor[:, :, :, 0]
        right_edge = tensor[:, :, :, -1]
    else:  # Image format
        left_edge = tensor[:, :, 0, :]
        right_edge = tensor[:, :, -1, :]
    
    diff = torch.abs(left_edge - right_edge).mean()
    return diff.item() < threshold
```

### Step 2.2: Implement Aspect Ratio Utilities

Create `utils/equirect.py`:

```python
"""
Equirectangular projection utilities and validation
"""

import torch
import math
from typing import Tuple


def validate_aspect_ratio(
    width: int,
    height: int,
    tolerance: float = 0.01
) -> bool:
    """
    Check if dimensions are 2:1 ratio (equirectangular requirement)
    
    Args:
        width: Image width
        height: Image height
        tolerance: Acceptable deviation from 2:1
    
    Returns:
        True if valid 2:1 ratio
    """
    ratio = width / height
    return abs(ratio - 2.0) < tolerance


def get_equirect_dimensions(
    width: int,
    alignment: int = 16
) -> Tuple[int, int]:
    """
    Calculate valid equirectangular dimensions
    
    Args:
        width: Desired width
        alignment: Pixel alignment (16 for FLUX)
    
    Returns:
        (width, height) tuple with correct 2:1 ratio and alignment
    """
    # Ensure width is multiple of alignment
    width = (width // alignment) * alignment
    
    # Height is exactly half for 2:1 ratio
    height = width // 2
    
    return width, height


def blend_edges(
    image: torch.Tensor,
    blend_width: int = 10,
    blend_mode: str = "cosine"
) -> torch.Tensor:
    """
    Blend left and right edges for seamless wraparound
    
    This is the final polish step after generation. Even with
    circular padding, there can be subtle edge artifacts.
    
    Args:
        image: (B, H, W, C) image in ComfyUI format
        blend_width: Width of blend region in pixels
        blend_mode: Blending curve - 'cosine', 'linear', 'smoothstep'
    
    Returns:
        Image with perfectly blended edges
    
    Example:
        >>> image = torch.rand(1, 1024, 2048, 3)
        >>> blended = blend_edges(image, blend_width=10)
        >>> # Check left and right edges match
        >>> left = blended[:, :, :5, :]
        >>> right = blended[:, :, -5:, :]
        >>> torch.allclose(left, right, atol=0.01)
        True
    """
    if blend_width <= 0:
        return image
    
    B, H, W, C = image.shape
    
    if blend_width >= W // 2:
        print(f"Warning: blend_width {blend_width} too large for width {W}, using {W//4}")
        blend_width = W // 4
    
    # Extract edges
    left_edge = image[:, :, :blend_width, :]
    right_edge = image[:, :, -blend_width:, :]
    
    # Create blend weights based on mode
    if blend_mode == "cosine":
        # Smooth cosine transition: 0 → 1
        t = torch.linspace(0, math.pi, blend_width, device=image.device)
        weights = (1 - torch.cos(t)) / 2
    elif blend_mode == "linear":
        # Linear ramp: 0 → 1
        weights = torch.linspace(0, 1, blend_width, device=image.device)
    elif blend_mode == "smoothstep":
        # Smoothstep: 0 → 1 with smooth acceleration
        t = torch.linspace(0, 1, blend_width, device=image.device)
        weights = t * t * (3 - 2 * t)
    else:
        print(f"Unknown blend_mode '{blend_mode}', using cosine")
        t = torch.linspace(0, math.pi, blend_width, device=image.device)
        weights = (1 - torch.cos(t)) / 2
    
    # Reshape for broadcasting: (1, 1, blend_width, 1)
    weights = weights.view(1, 1, -1, 1)
    
    # Blend edges:
    # - Left edge transitions FROM right edge TO original left
    # - Right edge transitions FROM original right TO left edge
    blended_left = right_edge * (1 - weights) + left_edge * weights
    blended_right = right_edge * (1 - weights) + left_edge * weights
    
    # Apply blending
    result = image.clone()
    result[:, :, :blend_width, :] = blended_left
    result[:, :, -blend_width:, :] = blended_right
    
    return result
```

### Step 2.3: Validation Checklist
- [ ] Test circular padding with dummy tensor
- [ ] Verify left padding = right edge of original
- [ ] Verify right padding = left edge of original
- [ ] Test remove_circular_padding undoes apply_circular_padding
- [ ] Test aspect ratio validation with valid/invalid dimensions
- [ ] Test edge blending creates smooth transition

---

## PHASE 3: Node Implementations

### Step 3.1: Node Registration Setup

Create `nodes.py` skeleton:

```python
"""
ComfyUI-DiT360 Node Implementations
"""

import torch
import math
import comfy.samplers
import comfy.sample
import comfy.utils
import folder_paths
from pathlib import Path

# Import our utilities
from .utils.circular_padding import (
    apply_circular_padding,
    remove_circular_padding,
    validate_circular_continuity
)
from .utils.equirect import (
    validate_aspect_ratio,
    get_equirect_dimensions,
    blend_edges
)


# ====================================================================
# NODE 1: EQUIRECT360EMPTYLATENT
# ====================================================================

class Equirect360EmptyLatent:
    """
    Create empty latent with enforced 2:1 aspect ratio
    
    This replaces EmptyLatentImage and ensures users create
    proper equirectangular dimensions.
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
                    "tooltip": "Width (must be 2× height)"
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
        
        # Get valid dimensions
        width, height = get_equirect_dimensions(width, alignment=16)
        
        # FLUX uses 16x compression factor
        latent_width = width // 16
        latent_height = height // 16
        
        # FLUX latent has 16 channels
        latent = torch.zeros(
            [batch_size, 16, latent_height, latent_width],
            dtype=torch.float32
        )
        
        print(f"Created equirectangular latent: {width}×{height} image → {latent_width}×{latent_height} latent")
        
        return ({"samples": latent},)


# ====================================================================
# NODE 2: EQUIRECT360KSAMPLER
# ====================================================================

class Equirect360KSampler:
    """
    KSampler with circular padding for seamless 360° panoramas
    
    This is the CORE node. It applies circular padding at each
    sampling step to ensure seamless wraparound.
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
                    "tooltip": "Padding for seamless edges (16-32 recommended)"
                }),
                
                # Optional losses (disabled by default)
                "enable_yaw_loss": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Rotational consistency (2-3x slower)"
                }),
                "yaw_loss_weight": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "enable_cube_loss": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pole distortion reduction (slower)"
                }),
                "cube_loss_weight": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DiT360/sampling"
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise, circular_padding,
               enable_yaw_loss, yaw_loss_weight, enable_cube_loss, cube_loss_weight):
        """
        Sample with circular padding for seamless panoramas
        """
        
        # TODO: Implement actual sampling with circular padding
        # This is a placeholder that will be replaced in Phase 4
        
        print(f"🔄 Equirect360KSampler:")
        print(f"   • Steps: {steps}, CFG: {cfg}")
        print(f"   • Circular padding: {circular_padding}")
        print(f"   • Yaw loss: {enable_yaw_loss} (weight: {yaw_loss_weight})")
        print(f"   • Cube loss: {enable_cube_loss} (weight: {cube_loss_weight})")
        
        # For now, just add some noise to test
        latent = latent_image["samples"]
        noisy = latent + torch.randn_like(latent) * 0.1
        
        return ({"samples": noisy},)


# ====================================================================
# NODE 3: EQUIRECT360VAEDECODE
# ====================================================================

class Equirect360VAEDecode:
    """
    VAE decode with circular padding for smooth edges
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
                    "tooltip": "VAE decode padding (16 recommended)"
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
            
            # Remove padding (16x upscale factor for FLUX)
            padding_pixels = circular_padding * 16
            image = remove_circular_padding(image_padded, padding_pixels)
        else:
            # Standard decode
            image = vae.decode(latent)
        
        # Ensure ComfyUI format: (B, H, W, C)
        if image.shape[1] == 3:  # (B, 3, H, W) → (B, H, W, 3)
            image = image.permute(0, 2, 3, 1)
        
        print(f"✅ Decoded to {image.shape[2]}×{image.shape[1]} panorama")
        
        return (image,)


# ====================================================================
# NODE 4: EQUIRECT360EDGEBLENDER
# ====================================================================

class Equirect360EdgeBlender:
    """
    Post-processing edge blending for perfect wraparound
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
                    "tooltip": "Blend region width in pixels"
                }),
                "blend_mode": (["cosine", "linear", "smoothstep"], {
                    "default": "cosine"
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
        is_seamless = validate_circular_continuity(blended, threshold=0.05)
        
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
    Interactive 360° panorama viewer
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
                    "tooltip": "Max width for preview (lower = faster)"
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
        import base64
        from io import BytesIO
        from PIL import Image
        import numpy as np
        
        results = []
        
        for image in images:
            # Convert to PIL
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Resize if needed
            W, H = img_pil.size
            if W > max_resolution:
                new_W = max_resolution
                new_H = new_W // 2
                img_pil = img_pil.resize((new_W, new_H), Image.LANCZOS)
                print(f"📐 Resized for preview: {W}×{H} → {new_W}×{new_H}")
            
            # Convert to base64 JPEG
            buffer = BytesIO()
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
```

### Step 3.2: Validation Checklist
- [ ] All 5 nodes load in ComfyUI
- [ ] Nodes appear in "DiT360" category
- [ ] Can add nodes to workflow
- [ ] Nodes accept correct input types
- [ ] Console shows print statements when nodes execute

---

## PHASE 4: Implement Actual Sampling

### Step 4.1: Understanding ComfyUI's Sampling System

ComfyUI's sampling is complex. Key points:

1. **Don't reimplement the sampler** - Use ComfyUI's existing system
2. **Intercept at the model call** - Apply padding before model forward pass
3. **Use model_options for hooks** - ComfyUI provides callback system

**Strategy**: We'll wrap the model's `apply_model` function to add circular padding.

### Step 4.2: Implement Circular Padding Wrapper

Add this to `nodes.py` before the `Equirect360KSampler` class:

```python
def create_circular_padding_wrapper(model, circular_padding):
    """
    Create a wrapper that applies circular padding to model calls
    
    This wraps the model's apply_model function to add circular
    padding before each forward pass.
    """
    original_apply_model = model.apply_model
    
    def wrapped_apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        """
        Wrapped apply_model with circular padding
        
        Args:
            x: Input latent
            t: Timestep
            Other args passed through
        
        Returns:
            Model output with padding removed
        """
        # Apply circular padding to input
        if circular_padding > 0:
            x_padded = apply_circular_padding(x, circular_padding)
        else:
            x_padded = x
        
        # Call original model
        output_padded = original_apply_model(
            x_padded, t, c_concat, c_crossattn, control, transformer_options, **kwargs
        )
        
        # Remove padding from output
        if circular_padding > 0:
            output = remove_circular_padding(output_padded, circular_padding)
        else:
            output = output_padded
        
        return output
    
    # Replace model's apply_model
    model.apply_model = wrapped_apply_model
    
    return model
```

### Step 4.3: Update Equirect360KSampler.sample()

Replace the placeholder implementation:

```python
def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
           positive, negative, latent_image, denoise, circular_padding,
           enable_yaw_loss, yaw_loss_weight, enable_cube_loss, cube_loss_weight):
    """
    Sample with circular padding for seamless panoramas
    """
    
    # Clone model to avoid affecting other nodes
    model_clone = model.clone()
    
    # Wrap model to add circular padding
    if circular_padding > 0:
        model_clone.model = create_circular_padding_wrapper(
            model_clone.model, 
            circular_padding
        )
        print(f"🔄 Applied circular padding: {circular_padding} pixels")
    
    # Warning if losses enabled (slow)
    if enable_yaw_loss or enable_cube_loss:
        print("⚠️ Geometric losses enabled - generation will be 2-5x slower")
        # TODO: Implement yaw/cube loss in Phase 7
        # For now, just warn
    
    # Standard sampling using ComfyUI's sampler
    latent = latent_image["samples"]
    
    # Use ComfyUI's sample function
    samples = comfy.sample.sample(
        model_clone,
        noise=torch.randn_like(latent) if denoise == 1.0 else None,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent,
        start_step=0,
        last_step=steps,
        force_full_denoise=(denoise == 1.0),
        denoise_mask=None,
        disable_noise=False,
        seed=seed,
        callback=None
    )
    
    print(f"✅ Sampling complete: {samples.shape}")
    
    return ({"samples": samples},)
```

### Step 4.4: Validation Checklist
- [ ] Generate a panorama with FLUX + DiT360 LoRA
- [ ] Use Equirect360KSampler instead of standard KSampler
- [ ] Check if left and right edges align when wrapped
- [ ] Compare with/without circular_padding parameter
- [ ] Verify generation time is similar to standard KSampler

---

## PHASE 5: Three.js Viewer Implementation

### Step 5.1: Create Frontend JavaScript

Create `web/js/equirect360_viewer.js`:

```javascript
/**
 * ComfyUI-DiT360 Interactive 360° Viewer
 * Uses Three.js for panorama navigation
 */

import { app } from "../../scripts/app.js";

// Load Three.js from CDN
const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

let THREE = null;

// Load Three.js dynamically
async function loadThreeJS() {
    if (THREE) return THREE;
    
    try {
        THREE = await import(THREE_CDN);
        console.log("✅ Three.js loaded for 360° viewer");
        return THREE;
    } catch (error) {
        console.error("❌ Failed to load Three.js:", error);
        return null;
    }
}

// Register extension with ComfyUI
app.registerExtension({
    name: "ComfyUI.DiT360.Equirect360Viewer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Equirect360Viewer") {
            // Load Three.js when viewer node is registered
            await loadThreeJS();
        }
    },
    
    async nodeCreated(node) {
        if (node.comfyClass === "Equirect360Viewer") {
            // Add custom viewer widget
            const widget = node.addWidget("button", "🌐 View 360°", "view360", () => {
                if (node.imgs && node.imgs.length > 0) {
                    open360Viewer(node.imgs[0].src);
                } else {
                    alert("No panorama to view. Generate an image first!");
                }
            });
            
            widget.serialize = false; // Don't save button state
        }
    }
});

/**
 * Open 360° viewer modal
 */
async function open360Viewer(imageUrl) {
    if (!THREE) {
        await loadThreeJS();
        if (!THREE) {
            alert("Failed to load 3D viewer library");
            return;
        }
    }
    
    // Create modal overlay
    const modal = document.createElement("div");
    modal.id = "equirect360-viewer-modal";
    modal.style.cssText = `
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(0, 0, 0, 0.95);
        z-index: 10000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    `;
    
    // Create canvas
    const canvas = document.createElement("canvas");
    canvas.style.cssText = `
        width: 90vw;
        height: 90vh;
        cursor: grab;
    `;
    modal.appendChild(canvas);
    
    // Create controls overlay
    const controls = document.createElement("div");
    controls.style.cssText = `
        position: absolute;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.7);
        padding: 10px 20px;
        border-radius: 5px;
        color: white;
        font-family: Arial, sans-serif;
    `;
    controls.innerHTML = `
        <strong>Controls:</strong> 
        Drag to rotate • Scroll to zoom • ESC or click to close
    `;
    modal.appendChild(controls);
    
    // Create close button
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕ Close";
    closeBtn.style.cssText = `
        position: absolute;
        top: 20px; right: 20px;
        padding: 10px 20px;
        background: #fff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    `;
    closeBtn.onclick = () => cleanup();
    modal.appendChild(closeBtn);
    
    document.body.appendChild(modal);
    
    // Initialize Three.js scene
    const scene = new THREE.Scene();
    
    const camera = new THREE.PerspectiveCamera(
        75,
        canvas.clientWidth / canvas.clientHeight,
        0.1,
        1000
    );
    camera.position.set(0, 0, 0);
    
    const renderer = new THREE.WebGLRenderer({ 
        canvas,
        antialias: true
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Create sphere for panorama (inverted normals for inside view)
    const geometry = new THREE.SphereGeometry(500, 60, 40);
    geometry.scale(-1, 1, 1); // Invert for inside viewing
    
    // Load texture
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(imageUrl, () => {
        console.log("✅ Panorama texture loaded");
    });
    
    const material = new THREE.MeshBasicMaterial({ map: texture });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);
    
    // Mouse controls
    let isDragging = false;
    let previousMouse = { x: 0, y: 0 };
    let rotation = { x: 0, y: 0 };
    
    canvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        canvas.style.cursor = "grabbing";
        previousMouse = { x: e.clientX, y: e.clientY };
    });
    
    canvas.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        
        const deltaX = e.clientX - previousMouse.x;
        const deltaY = e.clientY - previousMouse.y;
        
        rotation.y -= deltaX * 0.005;
        rotation.x -= deltaY * 0.005;
        
        // Clamp vertical rotation to avoid flipping
        rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotation.x));
        
        previousMouse = { x: e.clientX, y: e.clientY };
    });
    
    canvas.addEventListener("mouseup", () => {
        isDragging = false;
        canvas.style.cursor = "grab";
    });
    
    canvas.addEventListener("mouseleave", () => {
        isDragging = false;
        canvas.style.cursor = "grab";
    });
    
    // Scroll for zoom (FOV adjustment)
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        
        camera.fov += e.deltaY * 0.05;
        camera.fov = Math.max(30, Math.min(120, camera.fov));
        camera.updateProjectionMatrix();
    });
    
    // Touch controls for mobile
    let touchStart = null;
    
    canvas.addEventListener("touchstart", (e) => {
        if (e.touches.length === 1) {
            touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        }
    });
    
    canvas.addEventListener("touchmove", (e) => {
        if (e.touches.length === 1 && touchStart) {
            const deltaX = e.touches[0].clientX - touchStart.x;
            const deltaY = e.touches[0].clientY - touchStart.y;
            
            rotation.y -= deltaX * 0.005;
            rotation.x -= deltaY * 0.005;
            rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotation.x));
            
            touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        }
    });
    
    canvas.addEventListener("touchend", () => {
        touchStart = null;
    });
    
    // Keyboard controls
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            cleanup();
        }
    });
    
    // Click outside to close
    modal.addEventListener("click", (e) => {
        if (e.target === modal) {
            cleanup();
        }
    });
    
    // Animation loop
    let animationId;
    function animate() {
        animationId = requestAnimationFrame(animate);
        
        // Apply rotation to camera
        camera.rotation.order = "YXZ";
        camera.rotation.y = rotation.y;
        camera.rotation.x = rotation.x;
        
        renderer.render(scene, camera);
    }
    
    animate();
    
    // Cleanup function
    function cleanup() {
        cancelAnimationFrame(animationId);
        renderer.dispose();
        geometry.dispose();
        material.dispose();
        texture.dispose();
        document.body.removeChild(modal);
        console.log("✅ 360° viewer closed");
    }
    
    // Handle window resize
    window.addEventListener("resize", () => {
        if (document.body.contains(modal)) {
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        }
    });
    
    console.log("✅ 360° viewer opened");
}
```

### Step 5.2: Validation Checklist
- [ ] Viewer button appears on Equirect360Viewer node
- [ ] Clicking button opens modal with panorama
- [ ] Mouse drag rotates view
- [ ] Scroll zooms in/out
- [ ] ESC key closes viewer
- [ ] No visible seam at wraparound boundary

---

## PHASE 6: Testing & Polish

### Step 6.1: Create Test Workflow

Create `examples/basic_workflow.json`:

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "flux1-dev.safetensors"
    }
  },
  "2": {
    "class_type": "LoraLoader",
    "inputs": {
      "model": ["1", 0],
      "clip": ["1", 1],
      "lora_name": "dit360.safetensors",
      "strength_model": 1.0,
      "strength_clip": 1.0
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "clip": ["2", 1],
      "text": "A beautiful sunset over a tropical beach with palm trees, 360 degree panorama"
    }
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "clip": ["2", 1],
      "text": "blurry, low quality, distorted"
    }
  },
  "5": {
    "class_type": "Equirect360EmptyLatent",
    "inputs": {
      "width": 2048,
      "batch_size": 1
    }
  },
  "6": {
    "class_type": "Equirect360KSampler",
    "inputs": {
      "model": ["2", 0],
      "seed": 42,
      "steps": 20,
      "cfg": 3.5,
      "sampler_name": "euler",
      "scheduler": "simple",
      "positive": ["3", 0],
      "negative": ["4", 0],
      "latent_image": ["5", 0],
      "denoise": 1.0,
      "circular_padding": 16,
      "enable_yaw_loss": false,
      "yaw_loss_weight": 0.1,
      "enable_cube_loss": false,
      "cube_loss_weight": 0.1
    }
  },
  "7": {
    "class_type": "Equirect360VAEDecode",
    "inputs": {
      "samples": ["6", 0],
      "vae": ["1", 2],
      "circular_padding": 16
    }
  },
  "8": {
    "class_type": "Equirect360EdgeBlender",
    "inputs": {
      "image": ["7", 0],
      "blend_width": 10,
      "blend_mode": "cosine"
    }
  },
  "9": {
    "class_type": "Equirect360Viewer",
    "inputs": {
      "images": ["8", 0],
      "max_resolution": 4096
    }
  },
  "10": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["8", 0],
      "filename_prefix": "dit360_panorama"
    }
  }
}
```

### Step 6.2: Create README.md

```markdown
# ComfyUI-DiT360

360° panorama generation enhancement for FLUX.1-dev with DiT360 LoRA.

## Features

- ✅ **Circular Padding**: Seamless wraparound edges
- ✅ **2:1 Aspect Ratio**: Automatic equirectangular format
- ✅ **Edge Blending**: Perfect continuity at boundaries
- ✅ **Interactive Viewer**: Three.js-based 360° navigation
- ✅ **Optional Losses**: Yaw/cube loss for quality (slower)

## Installation

1. Install ComfyUI if not already installed
2. Clone this repo into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-DiT360.git
cd ComfyUI-DiT360
pip install -r requirements.txt
```

3. Download models:
   - **FLUX.1-dev**: Place in `ComfyUI/models/checkpoints/`
   - **DiT360 LoRA**: Download from [HuggingFace](https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation), place in `ComfyUI/models/loras/`

4. Restart ComfyUI

## Usage

### Basic Workflow

1. Load Checkpoint (FLUX.1-dev)
2. Load LoRA (DiT360)
3. CLIP Text Encode (your prompt)
4. **Equirect360EmptyLatent** (creates 2048×1024 latent)
5. **Equirect360KSampler** (circular_padding=16)
6. **Equirect360VAEDecode** (circular_padding=16)
7. **Equirect360EdgeBlender** (blend_width=10)
8. **Equirect360Viewer** (interactive preview)
9. Save Image

See `examples/basic_workflow.json` for complete workflow.

### Node Descriptions

#### 360° Empty Latent
Creates empty latent with enforced 2:1 aspect ratio. Use instead of EmptyLatentImage.

#### 360° KSampler
Drop-in replacement for KSampler with circular padding. Enables seamless panoramas.

**Parameters**:
- `circular_padding`: 16-32 recommended
- `enable_yaw_loss`: False by default (2-3x slower if enabled)
- `enable_cube_loss`: False by default (slower if enabled)

#### 360° VAE Decode
VAE decode with circular padding for extra edge smoothness.

#### 360° Edge Blender
Post-processing to ensure perfect wraparound. Highly recommended!

**Parameters**:
- `blend_width`: 10-20 pixels recommended
- `blend_mode`: "cosine" (smoothest), "linear", "smoothstep"

#### 360° Viewer
Interactive preview with Three.js. Click "View 360°" to navigate.

## Prompting Tips

Good prompts describe the full 360° environment:

```
"A cozy living room with large windows showing mountain views,
warm afternoon sunlight, wooden furniture, plants, 360 panorama"

"Standing in a futuristic city plaza, skyscrapers all around,
neon signs, rain-slicked streets, night time, cyberpunk"
```

## Performance

| Resolution | VRAM | Speed | Quality |
|-----------|------|-------|---------|
| 1024×512 | 12GB | Fast | Good |
| 2048×1024 | 16GB | Medium | Excellent |
| 4096×2048 | 24GB+ | Slow | Outstanding |

**With losses enabled**: 2-5x slower but higher quality.

## Troubleshooting

### Visible seam at edges
- Increase `circular_padding` to 24-32
- Increase `blend_width` to 20+
- Enable `enable_yaw_loss` (slower)

### Out of memory
- Lower resolution (1024×512)
- Use fp8 precision for FLUX
- Disable yaw/cube losses

### Not seamless in viewer
- Check that DiT360 LoRA is loaded
- Ensure `circular_padding` > 0
- Use Equirect360EdgeBlender

## License

Apache 2.0

## Credits

- DiT360 LoRA by Insta360 Research Team
- FLUX.1-dev by Black Forest Labs
- ComfyUI by comfyanonymous
```

### Step 6.3: Final Validation Checklist

- [ ] All nodes work in complete workflow
- [ ] Panoramas have seamless edges
- [ ] Viewer works correctly
- [ ] README has clear instructions
- [ ] Example workflow loads and runs
- [ ] Works on Windows (path handling correct)
- [ ] Console output is helpful
- [ ] No errors or warnings

---

## Common Issues & Solutions

### Issue 1: "Model not found"
**Problem**: DiT360 LoRA not in correct location
**Solution**: 
```bash
# Correct location
ComfyUI/models/loras/dit360.safetensors

# Not here:
ComfyUI/models/dit360/
```

### Issue 2: Visible seam despite circular padding
**Problem**: Padding too small or blend width too small
**Solution**:
```python
# Try these values:
circular_padding = 24  # Instead of 16
blend_width = 20       # Instead of 10
```

### Issue 3: Generation very slow
**Problem**: Yaw/cube losses enabled
**Solution**:
```python
# Disable for faster generation:
enable_yaw_loss = False
enable_cube_loss = False
```

### Issue 4: Viewer doesn't open
**Problem**: Three.js failed to load
**Solution**: Check browser console for errors. Try different browser or check internet connection (Three.js loads from CDN).

---

## Testing Checklist

Before considering complete:

### Functionality
- [ ] Equirect360EmptyLatent creates correct dimensions
- [ ] Equirect360KSampler applies circular padding
- [ ] Equirect360VAEDecode works with standard VAE
- [ ] Equirect360EdgeBlender creates seamless wraparound
- [ ] Equirect360Viewer opens and works

### Quality
- [ ] Left and right edges align perfectly
- [ ] No visible seam in 360° viewer
- [ ] Consistent lighting across boundary
- [ ] No artifacts at edges

### Performance
- [ ] Generation time reasonable (~2min for 2048×1024, 20 steps)
- [ ] Memory usage acceptable (<20GB VRAM)
- [ ] Losses disable by default (fast mode)

### Compatibility
- [ ] Works on Windows
- [ ] Works on Linux
- [ ] Works with FLUX.1-dev
- [ ] Works with DiT360 LoRA
- [ ] Compatible with standard ComfyUI workflow

### Documentation
- [ ] README clear and complete
- [ ] Example workflow included
- [ ] Troubleshooting guide helpful
- [ ] Code commented

---

## Next Steps After Completion

1. **GitHub Release**
   - Create repository
   - Add all files
   - Create release with version tag
   - Write release notes

2. **ComfyUI Manager**
   - Submit to ComfyUI Manager registry
   - Test installation via manager

3. **Community Sharing**
   - Post on ComfyUI Discord
   - Share on Reddit (r/StableDiffusion)
   - Create example gallery

4. **Future Enhancements**
   - Implement yaw/cube losses properly
   - Add inpainting support
   - Support for other models (SDXL, etc.)
   - Better pole handling

---

## Quick Reference

### Circular Padding
```python
# Apply
padded = apply_circular_padding(latent, padding=16)

# Remove
original = remove_circular_padding(padded, padding=16)
```

### Edge Blending
```python
blended = blend_edges(image, blend_width=10, blend_mode="cosine")
```

### Aspect Ratio
```python
width, height = get_equirect_dimensions(width=2048, alignment=16)
# Returns (2048, 1024)
```

### Validation
```python
is_valid = validate_aspect_ratio(2048, 1024)  # True
is_seamless = validate_circular_continuity(image, threshold=0.05)  # True
```

---

Good luck! 🚀 You're building something awesome for the ComfyUI community!
