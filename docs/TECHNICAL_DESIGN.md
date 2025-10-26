# ComfyUI-DiT360 Technical Design Document (Streamlined)
## 360° Panorama Generation with FLUX.1-dev + DiT360 LoRA

---

## 1. Executive Summary

### 1.1 Project Overview
ComfyUI-DiT360 adds 360-degree equirectangular panorama capabilities to standard ComfyUI workflows using FLUX.1-dev with the DiT360 LoRA adapter. This is **not** a full model wrapper—DiT360 is simply a LoRA weight file that enhances FLUX for panoramic generation.

**Key Insight**: DiT360 is distributed as a **LoRA adapter** (~2-5GB), not a full 12B parameter model. Users load FLUX.1-dev normally, then apply the DiT360 LoRA like any other LoRA in ComfyUI.

### 1.2 Core Features
- **Circular Padding**: Seamless wraparound at panorama edges (applied in sampling + VAE)
- **Yaw Loss**: Optional rotational consistency improvement (2-3x slower)
- **Cube Loss**: Optional pole distortion reduction (2-3x slower)  
- **2:1 Aspect Ratio Enforcement**: Helpers for equirectangular format
- **Interactive 360° Viewer**: Three.js-based panorama navigation

### 1.3 Integration Approach
Works as **drop-in enhancements** to standard FLUX workflows:

```
Standard FLUX Workflow:
Load Checkpoint → Load LoRA → CLIP Encode → KSampler → VAE Decode → Save

With DiT360 Enhancements:
Load Checkpoint → Load LoRA (DiT360) → CLIP Encode → 
  [Equirect360EmptyLatent] → [Equirect360KSampler] → 
  [Equirect360VAEDecode] → [Equirect360EdgeBlender] → 
  [Equirect360Viewer]
```

### 1.4 Node Count
**Only 5 new nodes** (minimal, composable):
1. `Equirect360EmptyLatent` - 2:1 aspect ratio helper
2. `Equirect360KSampler` - Sampling with circular padding + losses
3. `Equirect360VAEDecode` - VAE decode with circular padding
4. `Equirect360EdgeBlender` - Post-processing edge blending
5. `Equirect360Viewer` - Interactive preview

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Standard ComfyUI FLUX Workflow                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Load Checkpoint (FLUX.1-dev)                              │
│         ↓                                                   │
│  Load LoRA (DiT360.safetensors)  ← Standard LoRA!         │
│         ↓                                                   │
│  CLIP Text Encode (Positive/Negative)                      │
│         ↓                                                   │
│  ┌──────────────────────────────────────┐                  │
│  │  DiT360 Enhancement Layer (NEW)      │                  │
│  ├──────────────────────────────────────┤                  │
│  │  Equirect360EmptyLatent              │                  │
│  │    ↓                                 │                  │
│  │  Equirect360KSampler                 │                  │
│  │    ↓                                 │                  │
│  │  Equirect360VAEDecode                │                  │
│  │    ↓                                 │                  │
│  │  Equirect360EdgeBlender              │                  │
│  │    ↓                                 │                  │
│  │  Equirect360Viewer                   │                  │
│  └──────────────────────────────────────┘                  │
│         ↓                                                   │
│  Save Image (Standard)                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### Standard FLUX Components (User Provides)
- **Load Checkpoint**: Load FLUX.1-dev model (standard node)
- **Load LoRA**: Load DiT360.safetensors (standard node)
- **CLIP Text Encode**: Encode prompts (standard node)
- **VAE** (optional): Custom VAE or use FLUX's built-in

#### New DiT360 Enhancement Nodes

**1. Equirect360EmptyLatent**
- Purpose: Create empty latent with enforced 2:1 aspect ratio
- Replaces: `EmptyLatentImage`
- Why needed: Prevents user errors with wrong dimensions

**2. Equirect360KSampler** ⭐ CORE NODE
- Purpose: Sampling with circular padding for seamless edges
- Replaces: `KSampler` / `KSamplerAdvanced`
- Features:
  - Circular padding applied to latent at each step
  - Optional yaw loss (rotational consistency)
  - Optional cube loss (pole handling)
  - Works with any sampler/scheduler combination

**3. Equirect360VAEDecode**
- Purpose: VAE decode with circular padding
- Replaces: `VAEDecode`
- Why needed: Extra edge smoothing during upscaling

**4. Equirect360EdgeBlender**
- Purpose: Post-process edge blending
- Replaces: Nothing (new functionality)
- Why needed: Final polish for perfect wraparound

**5. Equirect360Viewer**
- Purpose: Interactive 360° preview
- Replaces: `PreviewImage` (for panoramas)
- Why needed: Standard preview doesn't show 360° properly

---

## 3. Technical Specifications

### 3.1 System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11 (64-bit), Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with 12GB VRAM (RTX 3060 12GB, 3080, 4070)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16GB system memory
- **Storage**: 30GB free SSD space (FLUX + LoRA + VAE)
- **Python**: 3.9 - 3.12

#### Recommended Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4080, 4090, A5000)
- **RAM**: 32GB system memory
- **Storage**: 50GB NVMe SSD

### 3.2 Dependencies

#### Core Dependencies (Minimal)
```
torch>=2.0.0,<3.0.0
numpy>=1.25.0
Pillow>=10.0.0
```

**No additional dependencies** beyond standard ComfyUI! All functionality uses PyTorch primitives.

#### Optional Dependencies
```
# For faster attention (if not already in ComfyUI)
xformers>=0.0.22  # Optional for memory-efficient attention
```

### 3.3 Model Specifications

#### FLUX.1-dev Base Model
- **Size**: ~24GB (fp16), ~12GB (fp8)
- **Source**: Black Forest Labs via Hugging Face
- **License**: FLUX.1-dev license (non-commercial research)
- **Location**: Standard ComfyUI `models/checkpoints/`

#### DiT360 LoRA
- **Size**: ~2-5GB (typical LoRA size)
- **Source**: Insta360-Research via Hugging Face
- **Format**: Standard safetensors LoRA
- **Location**: Standard ComfyUI `models/loras/`
- **Hugging Face**: `Insta360-Research/DiT360-Panorama-Image-Generation`

#### Output Specifications
- **Format**: Equirectangular projection (2:1 aspect ratio)
- **Common Resolutions**: 
  - Fast: 1024×512 (12GB VRAM)
  - Standard: 2048×1024 (16GB VRAM)
  - High Quality: 4096×2048 (24GB+ VRAM)
- **Latent Space**: Standard FLUX latent (16x compression)
- **Color Space**: RGB, values [0, 1]

---

## 4. Node Specifications

### 4.1 Equirect360EmptyLatent

**Purpose**: Create empty latent with enforced 2:1 aspect ratio

**Category**: `DiT360/latent`

**Inputs**:
```python
{
    "required": {
        "width": ("INT", {
            "default": 2048,
            "min": 512,
            "max": 8192,
            "step": 16  # FLUX requires 16-pixel alignment
        }),
        "batch_size": ("INT", {
            "default": 1,
            "min": 1,
            "max": 4096
        })
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("LATENT",)
```

**Behavior**:
- Automatically calculates `height = width // 2` (enforces 2:1)
- Creates latent with shape `(batch, 16, height//16, width//16)` for FLUX
- Validates dimensions are multiples of 16
- Initializes with zeros (standard empty latent)

**Example Usage**:
```
Equirect360EmptyLatent
  width: 2048
  → Creates latent for 2048×1024 image
  → Latent shape: (1, 16, 64, 128)
```

---

### 4.2 Equirect360KSampler ⭐

**Purpose**: KSampler with circular padding and optional geometric losses

**Category**: `DiT360/sampling`

**Inputs**:
```python
{
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
        
        # Circular padding settings
        "circular_padding": ("INT", {
            "default": 16,
            "min": 0,
            "max": 128,
            "tooltip": "Padding width for seamless edges (16-32 recommended)"
        }),
        
        # Optional geometric losses
        "enable_yaw_loss": ("BOOLEAN", {
            "default": False,
            "tooltip": "Enable rotational consistency (slower, higher quality)"
        }),
        "yaw_loss_weight": ("FLOAT", {
            "default": 0.1,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": "Yaw loss strength (0.05-0.2 recommended)"
        }),
        "enable_cube_loss": ("BOOLEAN", {
            "default": False,
            "tooltip": "Enable pole distortion reduction (slower)"
        }),
        "cube_loss_weight": ("FLOAT", {
            "default": 0.1,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": "Cube loss strength (0.05-0.2 recommended)"
        })
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("LATENT",)
```

**Key Implementation Details**:

**Circular Padding Application**:
```python
def apply_circular_padding(latent: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Apply circular padding to latent for seamless wraparound
    
    Args:
        latent: (B, C, H, W) tensor
        padding: Padding width in pixels
    
    Returns:
        Padded latent (B, C, H, W+2*padding)
    """
    if padding <= 0:
        return latent
    
    # Extract left and right edges
    left_edge = latent[:, :, :, -padding:]  # Rightmost columns
    right_edge = latent[:, :, :, :padding]  # Leftmost columns
    
    # Concatenate: [right_edge][latent][left_edge]
    padded = torch.cat([left_edge, latent, right_edge], dim=3)
    
    return padded

def remove_circular_padding(latent: torch.Tensor, padding: int) -> torch.Tensor:
    """Remove circular padding after processing"""
    if padding <= 0:
        return latent
    return latent[:, :, :, padding:-padding]
```

**Sampling Loop Integration**:
```python
def sample_with_circular_padding(
    model, latent, positive, negative, steps, cfg, 
    sampler_name, scheduler, seed, denoise,
    circular_padding=16,
    enable_yaw_loss=False, yaw_loss_weight=0.1,
    enable_cube_loss=False, cube_loss_weight=0.1
):
    """
    Modified sampling loop with circular padding
    
    Key differences from standard KSampler:
    1. Applies circular padding before each model call
    2. Removes padding from noise prediction
    3. Optionally applies geometric losses
    """
    
    # Initialize sampler (use ComfyUI's sampler factory)
    sampler = comfy.samplers.KSampler(
        model, steps, device, sampler_name, scheduler, 
        denoise, model_options={}
    )
    
    # Custom callback to apply circular padding
    def circular_padding_callback(step, x0, x, total_steps):
        """Called before each model forward pass"""
        
        # Apply circular padding
        x_padded = apply_circular_padding(x, circular_padding)
        
        # Get noise prediction
        noise_pred = model(x_padded, ...)
        
        # Remove padding from prediction
        noise_pred = remove_circular_padding(noise_pred, circular_padding)
        
        # Optional: Apply yaw loss
        if enable_yaw_loss and step % 5 == 0:  # Every 5 steps to save time
            yaw_grad = compute_yaw_loss(x, model, circular_padding)
            noise_pred = noise_pred + yaw_loss_weight * yaw_grad
        
        # Optional: Apply cube loss
        if enable_cube_loss and step % 5 == 0:
            cube_grad = compute_cube_loss(x, model)
            noise_pred = noise_pred + cube_loss_weight * cube_grad
        
        return noise_pred
    
    # Run sampling with callback
    samples = sampler.sample(
        latent, positive, negative, seed,
        callback=circular_padding_callback
    )
    
    return samples
```

**Yaw Loss Implementation** (Optional):
```python
def compute_yaw_loss(
    latent: torch.Tensor,
    model: torch.nn.Module,
    circular_padding: int,
    shift_amount: int = None
) -> torch.Tensor:
    """
    Compute yaw loss for rotational consistency
    
    Yaw loss ensures the panorama looks consistent when rotated
    horizontally. We generate predictions for the original and 
    horizontally-shifted versions and minimize their difference.
    
    Args:
        latent: Current latent (B, C, H, W)
        model: Diffusion model
        circular_padding: Padding width for model calls
        shift_amount: How far to shift (default: width // 4)
    
    Returns:
        Gradient to apply to latent
    """
    if shift_amount is None:
        shift_amount = latent.shape[3] // 4
    
    # Shift latent horizontally (circular shift)
    latent_shifted = torch.roll(latent, shifts=shift_amount, dims=3)
    
    # Get predictions for both
    with torch.no_grad():
        # Pad and predict original
        latent_padded = apply_circular_padding(latent, circular_padding)
        pred_original = model(latent_padded, ...)
        pred_original = remove_circular_padding(pred_original, circular_padding)
        
        # Pad and predict shifted
        latent_shifted_padded = apply_circular_padding(latent_shifted, circular_padding)
        pred_shifted = model(latent_shifted_padded, ...)
        pred_shifted = remove_circular_padding(pred_shifted, circular_padding)
    
    # Shift prediction back to align
    pred_shifted_aligned = torch.roll(pred_shifted, shifts=-shift_amount, dims=3)
    
    # Compute difference (this is the loss gradient)
    yaw_gradient = pred_original - pred_shifted_aligned
    
    return yaw_gradient
```

**Cube Loss Implementation** (Optional):
```python
def compute_cube_loss(
    latent: torch.Tensor,
    model: torch.nn.Module
) -> torch.Tensor:
    """
    Compute cube loss for pole distortion reduction
    
    Cube loss projects the latent to cubemap faces and checks
    consistency at face boundaries. This reduces distortion
    near poles (top/bottom of panorama).
    
    Note: This is a simplified approximation. Full implementation
    would require proper equirectangular-to-cubemap projection.
    
    Args:
        latent: Current latent (B, C, H, W)
        model: Diffusion model
    
    Returns:
        Gradient to apply to latent
    """
    B, C, H, W = latent.shape
    
    # Sample key regions (poles, equator)
    top_region = latent[:, :, :H//4, :]      # North pole region
    bottom_region = latent[:, :, -H//4:, :]  # South pole region
    equator_region = latent[:, :, H//2-H//8:H//2+H//8, :]  # Equator
    
    # Get predictions for each region
    with torch.no_grad():
        pred_top = model(top_region, ...)
        pred_bottom = model(bottom_region, ...)
        pred_equator = model(equator_region, ...)
    
    # Compute consistency losses
    # Poles should have lower frequency (less detail) than equator
    top_freq = compute_frequency_content(pred_top)
    bottom_freq = compute_frequency_content(pred_bottom)
    equator_freq = compute_frequency_content(pred_equator)
    
    # Gradient encourages pole regions to match expected frequency
    gradient = torch.zeros_like(latent)
    gradient[:, :, :H//4, :] = (top_freq - equator_freq * 0.5) * pred_top
    gradient[:, :, -H//4:, :] = (bottom_freq - equator_freq * 0.5) * pred_bottom
    
    return gradient

def compute_frequency_content(tensor: torch.Tensor) -> torch.Tensor:
    """Compute frequency content using FFT"""
    fft = torch.fft.fft2(tensor)
    magnitude = torch.abs(fft)
    return magnitude.mean()
```

**Performance Notes**:
- **Circular padding**: ~5% overhead (negligible)
- **Yaw loss**: ~2-3x slower (extra model forward pass)
- **Cube loss**: ~1.5-2x slower (partial forward passes)
- **Both losses**: ~4-5x slower total

**Recommendations**:
- Default: `circular_padding=16`, losses disabled (fast, good quality)
- Balanced: `circular_padding=24`, `enable_yaw_loss=True, weight=0.1` (2x slower, better quality)
- Maximum quality: Both losses enabled at weight 0.1-0.2 (5x slower, best quality)

---

### 4.3 Equirect360VAEDecode

**Purpose**: VAE decode with circular padding for smooth edges

**Category**: `DiT360/vae`

**Inputs**:
```python
{
    "required": {
        "samples": ("LATENT",),
        "vae": ("VAE",),
        "circular_padding": ("INT", {
            "default": 16,
            "min": 0,
            "max": 128,
            "tooltip": "Padding for VAE decode (16 recommended)"
        })
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("IMAGE",)
```

**Implementation**:
```python
def vae_decode_with_circular_padding(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    circular_padding: int = 16
) -> torch.Tensor:
    """
    VAE decode with circular padding for smooth edges
    
    Why this helps: VAE upsamples latent 16x. Circular padding
    ensures the upsampling respects wraparound at edges.
    
    Args:
        vae: VAE decoder
        latent: Latent tensor (B, C, H, W)
        circular_padding: Padding width in latent space
    
    Returns:
        Decoded image (B, H*16, W*16, 3) in ComfyUI format
    """
    
    if circular_padding > 0:
        # Apply padding in latent space
        latent_padded = apply_circular_padding(latent, circular_padding)
        
        # Decode with padding
        image_padded = vae.decode(latent_padded)
        
        # Remove padding in image space (16x upscale factor)
        padding_pixels = circular_padding * 16
        image = remove_circular_padding(image_padded, padding_pixels)
    else:
        # Standard decode without padding
        image = vae.decode(latent)
    
    # Convert to ComfyUI format if needed
    # FLUX VAE outputs (B, C, H, W), ComfyUI expects (B, H, W, C)
    if image.shape[1] == 3:  # (B, 3, H, W)
        image = image.permute(0, 2, 3, 1)  # → (B, H, W, 3)
    
    return image
```

**Performance**: ~5% overhead (negligible)

---

### 4.4 Equirect360EdgeBlender

**Purpose**: Post-processing edge blending for perfect wraparound

**Category**: `DiT360/post_process`

**Inputs**:
```python
{
    "required": {
        "image": ("IMAGE",),
        "blend_width": ("INT", {
            "default": 10,
            "min": 0,
            "max": 200,
            "tooltip": "Width of blend region in pixels"
        }),
        "blend_mode": (["cosine", "linear", "smoothstep"], {
            "default": "cosine",
            "tooltip": "Blending curve shape"
        })
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("IMAGE",)
```

**Implementation**:
```python
def blend_edges(
    image: torch.Tensor,
    blend_width: int = 10,
    blend_mode: str = "cosine"
) -> torch.Tensor:
    """
    Blend left and right edges for seamless wraparound
    
    This is the final polish step. Even with circular padding,
    there can be subtle discontinuities at the edge. This blends
    the leftmost and rightmost pixels to ensure perfect continuity.
    
    Args:
        image: (B, H, W, C) image in ComfyUI format
        blend_width: Width of blend region
        blend_mode: Blending curve
    
    Returns:
        Image with blended edges
    """
    if blend_width <= 0:
        return image
    
    B, H, W, C = image.shape
    
    # Extract edges
    left_edge = image[:, :, :blend_width, :]
    right_edge = image[:, :, -blend_width:, :]
    
    # Create blend weights based on mode
    if blend_mode == "cosine":
        # Smooth cosine curve: 0 → 1
        t = torch.linspace(0, math.pi, blend_width, device=image.device)
        weights = (1 - torch.cos(t)) / 2
    elif blend_mode == "linear":
        # Linear ramp: 0 → 1
        weights = torch.linspace(0, 1, blend_width, device=image.device)
    elif blend_mode == "smoothstep":
        # Smoothstep: 0 → 1 with ease in/out
        t = torch.linspace(0, 1, blend_width, device=image.device)
        weights = t * t * (3 - 2 * t)
    else:
        weights = torch.linspace(0, 1, blend_width, device=image.device)
    
    # Reshape for broadcasting: (1, 1, blend_width, 1)
    weights = weights.view(1, 1, -1, 1)
    
    # Blend: left transitions from right edge, right transitions to left edge
    blended_left = left_edge * (1 - weights) + right_edge * weights
    blended_right = right_edge * (1 - weights) + left_edge * weights
    
    # Apply blending
    result = image.clone()
    result[:, :, :blend_width, :] = blended_left
    result[:, :, -blend_width:, :] = blended_right
    
    return result
```

**Visual Explanation**:
```
Before blending:
Left edge:  [A A A A A]        Right edge: [B B B B B]
                                            ↓
After blending (blend_width=5):
Left edge:  [B B→A A A]        Right edge: [B A→A A A]
              ↑                            ↑
        Smooth transition            Smooth transition
```

**Performance**: <1% overhead (very fast)

---

### 4.5 Equirect360Viewer

**Purpose**: Interactive 360° panorama viewer

**Category**: `DiT360/preview`

**Inputs**:
```python
{
    "required": {
        "images": ("IMAGE",),
        "max_resolution": ("INT", {
            "default": 4096,
            "min": 512,
            "max": 8192,
            "tooltip": "Max width for preview (lower = faster loading)"
        })
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ()  # Preview node, no output
OUTPUT_NODE = True
```

**Implementation Overview**:

The viewer consists of:
1. **Backend (Python)**: Converts image to base64, returns UI data
2. **Frontend (JavaScript)**: Three.js viewer embedded in ComfyUI

**Backend Implementation**:
```python
def preview_360(
    images: torch.Tensor,
    max_resolution: int = 4096
) -> dict:
    """
    Prepare panorama for 360° preview
    
    Args:
        images: (B, H, W, C) tensor
        max_resolution: Max width for preview
    
    Returns:
        UI data dict for ComfyUI frontend
    """
    import base64
    from io import BytesIO
    from PIL import Image
    
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
        
        # Convert to base64 JPEG
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        results.append({
            "type": "equirect360",
            "image_data": f"data:image/jpeg;base64,{img_base64}",
            "width": img_pil.size[0],
            "height": img_pil.size[1]
        })
    
    return {"ui": {"images": results}}
```

**Frontend Implementation** (`web/js/equirect360_viewer.js`):
```javascript
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Register extension
app.registerExtension({
    name: "ComfyUI.DiT360.Equirect360Viewer",
    
    async nodeCreated(node) {
        if (node.comfyClass === "Equirect360Viewer") {
            // Add custom widget for 360° viewing
            node.addCustomWidget({
                name: "360_viewer",
                type: "360_viewer",
                
                draw: function(ctx, node, width, y) {
                    // Draw "View 360°" button
                    ctx.fillStyle = "#4CAF50";
                    ctx.fillRect(10, y, width - 20, 30);
                    ctx.fillStyle = "#FFFFFF";
                    ctx.font = "14px Arial";
                    ctx.fillText("🔄 View 360°", width/2 - 40, y + 20);
                },
                
                mouse: function(event, pos, node) {
                    if (event.type === "click" && node.images?.[0]) {
                        // Open 360° viewer modal
                        open360Viewer(node.images[0].image_data);
                    }
                }
            });
        }
    }
});

function open360Viewer(imageData) {
    // Create modal overlay
    const modal = document.createElement("div");
    modal.style.cssText = `
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.9);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    // Create canvas for Three.js
    const canvas = document.createElement("canvas");
    canvas.width = window.innerWidth * 0.9;
    canvas.height = window.innerHeight * 0.9;
    modal.appendChild(canvas);
    
    // Add close button
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕ Close";
    closeBtn.style.cssText = `
        position: absolute;
        top: 20px; right: 20px;
        padding: 10px 20px;
        background: #fff;
        border: none;
        cursor: pointer;
    `;
    closeBtn.onclick = () => {
        document.body.removeChild(modal);
        renderer.dispose();
    };
    modal.appendChild(closeBtn);
    
    document.body.appendChild(modal);
    
    // Initialize Three.js scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
        75, canvas.width / canvas.height, 0.1, 1000
    );
    const renderer = new THREE.WebGLRenderer({ canvas });
    
    // Create sphere for panorama
    const geometry = new THREE.SphereGeometry(500, 60, 40);
    geometry.scale(-1, 1, 1); // Invert for inside viewing
    
    // Load texture
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(imageData);
    
    const material = new THREE.MeshBasicMaterial({ map: texture });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);
    
    camera.position.set(0, 0, 0);
    
    // Mouse controls
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let rotation = { x: 0, y: 0 };
    
    canvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
    });
    
    canvas.addEventListener("mousemove", (e) => {
        if (isDragging) {
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;
            
            rotation.y += deltaX * 0.005;
            rotation.x += deltaY * 0.005;
            
            // Clamp vertical rotation
            rotation.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotation.x));
            
            previousMousePosition = { x: e.clientX, y: e.clientY };
        }
    });
    
    canvas.addEventListener("mouseup", () => {
        isDragging = false;
    });
    
    // Scroll for zoom (FOV adjustment)
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        camera.fov += e.deltaY * 0.05;
        camera.fov = Math.max(30, Math.min(120, camera.fov));
        camera.updateProjectionMatrix();
    });
    
    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        
        // Apply rotation to camera
        camera.rotation.order = "YXZ";
        camera.rotation.y = rotation.y;
        camera.rotation.x = rotation.x;
        
        renderer.render(scene, camera);
    }
    
    animate();
}
```

**Features**:
- Mouse drag to rotate view
- Scroll to zoom (adjust FOV)
- Fullscreen-like modal
- Works in ComfyUI web interface
- No external dependencies (Three.js loaded from CDN)

---

## 5. Implementation Phases

### Phase 1: Basic Structure (Week 1)
**Deliverables**:
- Project structure with 5 node files
- `__init__.py` with node registration
- `requirements.txt` (minimal)
- Basic README

**Validation**:
- Nodes load in ComfyUI without errors
- Appear in "DiT360" category

---

### Phase 2: Aspect Ratio Helper (Week 1)
**Deliverables**:
- `Equirect360EmptyLatent` node
- 2:1 ratio enforcement
- Input validation

**Validation**:
- Creates correct latent dimensions
- Rejects invalid dimensions
- Works with standard FLUX workflow

---

### Phase 3: Circular Padding Implementation (Week 1-2)
**Deliverables**:
- Core circular padding functions
- `Equirect360KSampler` basic version (no losses)
- Integration with ComfyUI sampler system

**Validation**:
- Panoramas have seamless left/right edges
- Generation time similar to standard KSampler
- Works with all sampler types (euler, dpmpp, etc.)

---

### Phase 4: VAE Decode Enhancement (Week 2)
**Deliverables**:
- `Equirect360VAEDecode` node
- Circular padding during VAE decode

**Validation**:
- Improved edge quality vs standard VAE decode
- No artifacts at boundaries
- <10% performance overhead

---

### Phase 5: Edge Blending Post-Process (Week 2)
**Deliverables**:
- `Equirect360EdgeBlender` node
- Multiple blend modes (cosine, linear, smoothstep)

**Validation**:
- Perfect wraparound in final image
- Check with `torch.allclose(left_edge, right_edge)`
- Visual inspection: no visible seam

---

### Phase 6: Interactive Viewer (Week 3)
**Deliverables**:
- `Equirect360Viewer` backend node
- Three.js frontend viewer
- Modal interface in ComfyUI

**Validation**:
- Viewer loads panoramas correctly
- Mouse controls work smoothly
- Works in ComfyUI web interface
- Proper texture wrapping (no seam in viewer)

---

### Phase 7: Geometric Losses (Week 3-4)
**Deliverables**:
- Yaw loss implementation in KSampler
- Cube loss implementation in KSampler
- Optional parameters (disabled by default)

**Validation**:
- Yaw loss improves rotational consistency
- Cube loss reduces pole distortion
- Performance impact: 2-3x slower when enabled
- Quality improvement visible in A/B testing

---

### Phase 8: Testing & Documentation (Week 4)
**Deliverables**:
- Unit tests for all nodes
- Example workflows
- Comprehensive README
- Troubleshooting guide

**Validation**:
- All tests pass on Windows and Linux
- Example workflows work out-of-box
- Clear installation instructions

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Test: Circular Padding**
```python
def test_circular_padding():
    """Test circular padding creates wraparound continuity"""
    latent = torch.rand(1, 4, 64, 128)
    padded = apply_circular_padding(latent, padding=10)
    
    # Check shape
    assert padded.shape == (1, 4, 64, 148)
    
    # Check left padding matches right edge of original
    assert torch.allclose(
        padded[:, :, :, :10],
        latent[:, :, :, -10:],
        atol=1e-6
    )
    
    # Check right padding matches left edge of original
    assert torch.allclose(
        padded[:, :, :, -10:],
        latent[:, :, :, :10],
        atol=1e-6
    )
```

**Test: Edge Blending**
```python
def test_edge_blending():
    """Test edge blending creates seamless wraparound"""
    image = torch.rand(1, 1024, 2048, 3)
    blended = blend_edges(image, blend_width=10)
    
    # Check left and right edges are similar after blending
    left_edge = blended[:, :, :10, :]
    right_edge = blended[:, :, -10:, :]
    
    # Should be very close (not identical due to blending)
    diff = torch.abs(left_edge - right_edge).mean()
    assert diff < 0.01  # Less than 1% difference
```

**Test: 2:1 Aspect Ratio**
```python
def test_aspect_ratio_enforcement():
    """Test Equirect360EmptyLatent enforces 2:1 ratio"""
    node = Equirect360EmptyLatent()
    
    # Test valid width
    latent = node.create_latent(width=2048, batch_size=1)
    samples = latent["samples"]
    
    # Latent is 16x compressed
    assert samples.shape == (1, 16, 64, 128)  # 1024×2048 image
    
    # Height should be exactly half of width
    assert samples.shape[2] * 2 == samples.shape[3]
```

### 6.2 Integration Tests

**Test: Full Workflow**
```python
def test_full_panorama_workflow():
    """Test complete workflow from latent to panorama"""
    
    # 1. Create empty latent
    latent_node = Equirect360EmptyLatent()
    latent = latent_node.create_latent(width=2048, batch_size=1)
    
    # 2. Sample (mock model)
    sampler_node = Equirect360KSampler()
    samples = sampler_node.sample(
        model=mock_model,
        latent_image=latent,
        steps=10,
        circular_padding=16
    )
    
    # 3. Decode
    vae_node = Equirect360VAEDecode()
    image = vae_node.decode(samples, mock_vae, circular_padding=16)
    
    # 4. Blend edges
    blender_node = Equirect360EdgeBlender()
    final_image = blender_node.blend(image, blend_width=10)
    
    # Validate final output
    assert final_image.shape == (1, 1024, 2048, 3)
    assert final_image.min() >= 0 and final_image.max() <= 1
    
    # Check seamlessness
    left = final_image[:, :, :5, :]
    right = final_image[:, :, -5:, :]
    diff = torch.abs(left - right).mean()
    assert diff < 0.05  # Less than 5% difference
```

### 6.3 Visual Quality Tests

**Checklist for Manual Testing**:
- [ ] Left and right edges align perfectly when wrapped
- [ ] No visible seam at edge boundary
- [ ] Top and bottom edges have appropriate distortion (more at poles)
- [ ] 360° viewer shows seamless rotation
- [ ] No artifacts or discontinuities
- [ ] Consistent lighting across wraparound
- [ ] Prompt is followed throughout panorama

---

## 7. Known Limitations & Future Work

### 7.1 Current Limitations

**Performance**:
- Yaw loss: 2-3x slower (requires extra model forward passes)
- Cube loss: 1.5-2x slower (additional computations)
- Limited to single-batch generation (no batch support yet)

**Compatibility**:
- Requires FLUX.1-dev (not compatible with SD1.5/SDXL without modifications)
- Needs 12GB+ VRAM for standard resolution (2048×1024)
- Windows/Linux only (macOS support untested)

**Features**:
- Yaw/cube loss implementations are simplified (could be more sophisticated)
- No inpainting support yet (would need mask handling)
- No ControlNet integration yet

### 7.2 Future Enhancements

**Performance**:
- [ ] Implement attention slicing for lower VRAM
- [ ] Add batch generation support
- [ ] Optimize yaw/cube loss (run less frequently, cache results)

**Features**:
- [ ] Inpainting support (mask-guided generation)
- [ ] Outpainting (extend existing panoramas)
- [ ] ControlNet integration (depth/edge-guided panoramas)
- [ ] Img2img workflow support
- [ ] Video panorama generation (360° videos)

**Quality**:
- [ ] More sophisticated yaw loss (multiple rotation angles)
- [ ] Better cube loss (proper cubemap projection)
- [ ] Lighting consistency checks
- [ ] Automatic horizon leveling

---

## 8. Appendix

### 8.1 Circular Padding Mathematical Basis

Circular padding works because:
1. Equirectangular projection wraps horizontally (0° = 360°)
2. Left edge (0°) and right edge (360°) represent the same direction
3. Padding ensures model "sees" this wraparound during generation
4. Without padding, model treats edges as independent boundaries

**Visual Representation**:
```
Standard Padding (wrong):
[0][0][0] [image data] [0][0][0]
   ↑                        ↑
Artificial boundaries create seam

Circular Padding (correct):
[R][R][R] [image data] [L][L][L]
   ↑                        ↑
Right edge   Left edge wraps to right
wraps to left
```

### 8.2 Why Losses Are Optional

**Yaw Loss**: Improves rotational consistency but:
- Requires 2-3 extra model forward passes per step
- Most panoramas look good without it
- Main benefit: Very large panoramas (4096×2048+)

**Cube Loss**: Reduces pole distortion but:
- Requires additional computations
- Effect most visible near top/bottom 20% of image
- Most users won't notice difference

**Recommendation**: Enable for final renders, disable for testing.

### 8.3 Reference Links

- **DiT360 LoRA**: https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation
- **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **Three.js**: https://threejs.org/
- **Equirectangular Projection**: https://en.wikipedia.org/wiki/Equirectangular_projection

### 8.4 License

This project is licensed under Apache License 2.0.

FLUX.1-dev and DiT360 LoRA are subject to their respective license terms.
