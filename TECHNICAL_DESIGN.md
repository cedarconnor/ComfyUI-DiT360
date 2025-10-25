# ComfyUI-DiT360 Technical Design Document

## 1. Executive Summary

### 1.1 Project Overview
ComfyUI-DiT360 is a custom node pack that integrates the DiT360 panoramic image generation model into ComfyUI's node-based workflow system. DiT360 is a 12-billion-parameter diffusion transformer built on FLUX.1-dev that generates high-fidelity 360-degree equirectangular panoramic images through hybrid training combining synthetic panoramic data with perspective images.

### 1.2 Key Objectives
- Provide seamless integration of DiT360 into ComfyUI workflows
- Support text-to-panorama, image-to-panorama, inpainting, and outpainting
- Ensure Windows compatibility with proper path handling and CUDA management
- Implement efficient memory management for 12B parameter model
- Support multiple precision levels (fp32, fp16, bf16, fp8)
- Provide interactive 360° panorama viewing capabilities
- Maintain compatibility with existing ComfyUI node ecosystem

### 1.3 Success Criteria
- Loads and runs on Windows 10/11 with NVIDIA GPUs (16GB+ VRAM)
- Generates 2048×1024 panoramas in under 2 minutes (50 steps)
- Properly handles equirectangular format with seamless edge wrapping
- Compatible with ComfyUI Manager for easy installation
- Clear error messages and comprehensive documentation
- No conflicts with major node packs (Impact-Pack, Manager, WAS Suite)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyUI Core System                      │
│                 (Node Discovery & Execution)                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Loads and Registers
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              ComfyUI-DiT360 Node Pack                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Model Loader │  │Text Encoder  │  │  VAE Loader  │     │
│  │    Nodes     │  │    Nodes     │  │    Nodes     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                  ┌──────────────────┐                       │
│                  │  Sampler Node    │                       │
│                  │  (Generation)    │                       │
│                  └────────┬─────────┘                       │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         ▼                 ▼                 ▼               │
│  ┌──────────┐      ┌──────────┐     ┌──────────┐          │
│  │Validator │      │ Decoder  │     │ Preview  │          │
│  │  Nodes   │      │  Nodes   │     │  Nodes   │          │
│  └──────────┘      └──────────┘     └──────────┘          │
└─────────────────────────────────────────────────────────────┘
                 │
                 │ Manages
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Model & Data Layer                             │
├─────────────────────────────────────────────────────────────┤
│  • DiT360 Model Weights (Hugging Face Hub)                 │
│  • FLUX.1-dev Base Model                                    │
│  • Text Encoders (T5, CLIP)                                │
│  • VAE (Encoder/Decoder)                                    │
│  • LoRA Weights                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 Model Loading Components
- **DiT360ModelLoader**: Main transformer model loader
- **DiT360TextEncoderLoader**: T5/CLIP text encoder loader
- **DiT360VAELoader**: VAE autoencoder loader
- **DiT360LoRALoader**: LoRA weight loader (optional)

#### 2.2.2 Generation Components
- **DiT360Sampler**: Core sampling/generation node
- **DiT360SamplerAdvanced**: Advanced sampler with geometric losses
- **DiT360ConditioningCombine**: Combine multiple conditioning inputs

#### 2.2.3 Processing Components
- **DiT360Decode**: Latent to image decoder
- **DiT360Encode**: Image to latent encoder
- **DiT360InpaintPrep**: Prepare images/masks for inpainting

#### 2.2.4 Validation & Utility Components
- **Equirect360Validator**: Validate and fix equirectangular format
- **Equirect360EdgeBlender**: Seamless edge blending
- **Equirect360Preview**: Interactive 360° viewer
- **Equirect360ToRectilinear**: Convert to flat perspective view
- **Equirect360ToCubemap**: Convert to cubemap format

---

## 3. Technical Specifications

### 3.1 System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11 (64-bit), Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with 16GB VRAM (RTX 3090, 4080, etc.)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16GB system memory
- **Storage**: 100GB free SSD space
- **Python**: 3.9 - 3.12

#### Recommended Requirements
- **GPU**: NVIDIA GPU with 24GB VRAM (RTX 4090, A5000, etc.)
- **RAM**: 32GB system memory
- **Storage**: 250GB NVMe SSD
- **Python**: 3.12

### 3.2 Dependencies

#### Core Dependencies
```
torch>=2.0.0,<3.0.0
torchvision>=0.15.0
transformers>=4.28.1
diffusers>=0.25.0
safetensors>=0.4.2
accelerate>=0.26.0
huggingface-hub>=0.20.0
```

#### Optional Dependencies
```
opencv-python>=4.8.0  # For advanced image processing
Pillow>=10.0.0  # Image handling
numpy>=1.25.0  # Numerical operations
```

### 3.3 Model Specifications

#### DiT360 Model Architecture
- **Base Model**: FLUX.1-dev (12B parameters)
- **Architecture**: Diffusion Transformer (DiT)
- **Training**: Hybrid (Matterport3D panoramas + perspective images)
- **Input Resolution**: Text prompts (up to 512 tokens)
- **Output Resolution**: 2048×1024 (default), supports 1024×512 to 4096×2048
- **Aspect Ratio**: 2:1 (equirectangular requirement)
- **Latent Channels**: 4 (VAE compression)
- **Latent Downscale**: 8x (256×128 latent for 2048×1024 image)

#### Key Features
- **Circular Padding**: Seamless wraparound at panorama edges
- **Yaw Loss**: Rotational invariance for consistent panoramas
- **Cube Loss**: Multi-scale distortion awareness (pole handling)
- **Flow Matching**: Advanced sampling with RoPE for spherical geometry

---

## 4. Data Flow & Processing Pipeline

### 4.1 Text-to-Panorama Pipeline

```
User Input (Text Prompt)
        ↓
Text Encoding (T5/CLIP)
        ↓
Conditioning Embeddings
        ↓
Initial Latent Noise (256×128×4)
        ↓
Iterative Denoising Loop (50 steps)
│   ├─ Apply Circular Padding
│   ├─ Transformer Forward Pass
│   ├─ Calculate Yaw Loss (optional)
│   ├─ Calculate Cube Loss (optional)
│   └─ Update Latent
        ↓
Denoised Latent (256×128×4)
        ↓
VAE Decode
        ↓
Equirectangular Image (2048×1024×3)
        ↓
Edge Blending & Validation
        ↓
Final 360° Panorama
```

### 4.2 Image-to-Panorama Pipeline

```
Input Image (Any Resolution)
        ↓
Resize/Pad to 2:1 Ratio
        ↓
VAE Encode to Latent
        ↓
Add Noise (Strength-based)
        ↓
[Same as Text-to-Panorama from Denoising Loop]
```

### 4.3 Inpainting Pipeline

```
Input Image (2048×1024) + Mask
        ↓
VAE Encode Image & Mask
        ↓
Initialize Latent with Image Latent
        ↓
Denoising Loop
│   ├─ Generate Noise for Masked Regions
│   ├─ Preserve Unmasked Regions
│   └─ Apply DiT360 Sampling
        ↓
VAE Decode
        ↓
Blend with Original (Feathering)
        ↓
Final Inpainted Panorama
```

---

## 5. Node Specifications

### 5.1 DiT360ModelLoader Node

**Purpose**: Load the main DiT360 transformer model

**Inputs**:
```python
{
    "required": {
        "model_name": (folder_paths.get_filename_list("dit360"),),
        "precision": (["fp32", "fp16", "bf16", "fp8"], {"default": "fp16"}),
        "offload_device": (["cuda", "cpu"], {"default": "cpu"})
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("DIT360_MODEL",)
```

**Functionality**:
- Locate model file in `ComfyUI/models/dit360/`
- Load safetensors/checkpoint with specified precision
- Initialize DiT360 architecture with FLUX.1-dev config
- Apply circular padding modifications to attention layers
- Load LoRA weights if present
- Configure offloading strategy for memory management
- Return wrapped model object with ComfyUI integration

**Error Handling**:
- Missing model files → Clear error with download instructions
- Insufficient VRAM → Suggest lower precision or offloading
- Corrupt model files → Validate checksums, suggest re-download

### 5.2 DiT360TextEncoderLoader Node

**Purpose**: Load text encoding models (T5/CLIP)

**Inputs**:
```python
{
    "required": {
        "encoder_name": (folder_paths.get_filename_list("text_encoders"),),
        "precision": (["fp32", "fp16"], {"default": "fp16"}),
        "max_length": ("INT", {"default": 512, "min": 77, "max": 1024})
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("TEXT_ENCODER",)
```

**Functionality**:
- Load T5-XXL or CLIP text encoder
- Configure tokenizer with max_length
- Handle both Hugging Face and local models
- Apply precision conversion
- Enable gradient checkpointing for memory efficiency

### 5.3 DiT360VAELoader Node

**Purpose**: Load VAE for latent encoding/decoding

**Inputs**:
```python
{
    "required": {
        "vae_name": (folder_paths.get_filename_list("vae"),),
        "precision": (["fp32", "fp16"], {"default": "fp16"})
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("VAE",)
```

**Functionality**:
- Load FLUX.1-dev compatible VAE
- Configure 8x downscale factor (matches latent dimensions)
- Enable tiling for large panoramas (>4096px width)
- Apply precision conversion

### 5.4 DiT360Sampler Node

**Purpose**: Core panorama generation node

**Inputs**:
```python
{
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
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
    },
    "optional": {
        "latent_image": ("LATENT",),
        "mask": ("MASK",)
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("LATENT", "IMAGE")
RETURN_NAMES = ("latent", "image")
```

**Functionality**:
1. **Validate aspect ratio** (enforce 2:1 or warn)
2. **Encode prompts** using text encoder
3. **Initialize latents**:
   - If latent_image provided: use as starting point
   - Else: generate random noise with seed
4. **Sampling loop** (50 steps default):
   - Apply circular padding to latent edges
   - Forward pass through DiT360 transformer
   - Calculate noise prediction with CFG
   - Update latent using flow matching scheduler
   - Report progress via ComfyUI progress bar
5. **VAE decode** latent to image
6. **Apply edge blending** for seamless wraparound
7. **Return** both latent and decoded image

**Key Implementation Details**:
```python
# Circular padding implementation
def apply_circular_padding(latent, padding=10):
    """Apply circular padding for seamless panorama edges"""
    left_edge = latent[:, :, :, :padding]
    right_edge = latent[:, :, :, -padding:]
    
    # Concatenate for wraparound
    padded = torch.cat([right_edge, latent, left_edge], dim=3)
    return padded

# Yaw loss calculation (optional)
def calculate_yaw_loss(latent, model, shift_amount=512):
    """Ensure rotational consistency"""
    shifted = torch.roll(latent, shifts=shift_amount, dims=3)
    loss = F.mse_loss(model(latent), model(shifted))
    return loss

# Cube loss calculation (optional)
def calculate_cube_loss(latent, model):
    """Multi-scale distortion awareness"""
    # Project to cubemap faces
    cubemap = equirect_to_cubemap(latent)
    # Calculate consistency across faces
    face_losses = []
    for i in range(6):
        for j in range(i+1, 6):
            if adjacent_faces(i, j):
                loss = edge_consistency_loss(cubemap[i], cubemap[j])
                face_losses.append(loss)
    return sum(face_losses) / len(face_losses)
```

### 5.5 DiT360SamplerAdvanced Node

**Purpose**: Advanced sampler with geometric loss options

**Additional Inputs** (extends DiT360Sampler):
```python
{
    "optional": {
        "enable_yaw_loss": ("BOOLEAN", {"default": False}),
        "yaw_loss_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
        "enable_cube_loss": ("BOOLEAN", {"default": False}),
        "cube_loss_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
        "circular_padding_width": ("INT", {"default": 10, "min": 0, "max": 100})
    }
}
```

**Functionality**:
- All functionality of DiT360Sampler
- Optional yaw loss for enhanced rotational consistency
- Optional cube loss for improved pole handling
- Configurable circular padding width
- Gradient-based optimization when losses enabled

### 5.6 Equirect360Validator Node

**Purpose**: Validate and fix equirectangular format issues

**Inputs**:
```python
{
    "required": {
        "image": ("IMAGE",),
        "enforce_ratio": ("BOOLEAN", {"default": True}),
        "fix_ratio": (["crop", "pad", "stretch", "none"], {"default": "pad"}),
        "target_width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64})
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("IMAGE",)
```

**Functionality**:
- Check if image is 2:1 aspect ratio
- If not 2:1 and enforce_ratio=True:
  - **Crop**: Center crop to 2:1
  - **Pad**: Add black bars to reach 2:1
  - **Stretch**: Resize to 2:1 (distorts content)
- Resize to target_width while maintaining 2:1
- Validate image range [0, 1] for ComfyUI compatibility
- Return validated image

### 5.7 Equirect360EdgeBlender Node

**Purpose**: Apply seamless edge blending for wraparound continuity

**Inputs**:
```python
{
    "required": {
        "image": ("IMAGE",),
        "blend_width": ("INT", {"default": 10, "min": 1, "max": 100}),
        "blend_mode": (["linear", "cosine", "smooth"], {"default": "cosine"})
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ("IMAGE",)
```

**Functionality**:
```python
def blend_edges(image, blend_width, mode="cosine"):
    """Blend left and right edges for seamless wraparound"""
    B, H, W, C = image.shape
    
    left_edge = image[:, :, :blend_width, :]
    right_edge = image[:, :, -blend_width:, :]
    
    if mode == "linear":
        weights = torch.linspace(0, 1, blend_width)
    elif mode == "cosine":
        weights = (1 - torch.cos(torch.linspace(0, math.pi, blend_width))) / 2
    elif mode == "smooth":
        weights = torch.linspace(0, 1, blend_width) ** 2
    
    weights = weights.view(1, 1, -1, 1).to(image.device)
    
    blended_left = left_edge * (1 - weights) + right_edge * weights
    blended_right = right_edge * (1 - weights) + left_edge * weights
    
    result = image.clone()
    result[:, :, :blend_width, :] = blended_left
    result[:, :, -blend_width:, :] = blended_right
    
    return result
```

### 5.8 Equirect360Preview Node

**Purpose**: Interactive 360° panorama viewer

**Inputs**:
```python
{
    "required": {
        "images": ("IMAGE",),
        "max_width": ("INT", {"default": 4096, "min": -1, "max": 8192})
    }
}
```

**Outputs**:
```python
RETURN_TYPES = ()
OUTPUT_NODE = True
```

**Functionality**:
- Convert ComfyUI IMAGE tensor to base64 JPEG
- Downsample to max_width for web viewing
- Return UI data with embedded viewer
- JavaScript viewer uses Three.js for 360° rendering
- Mouse drag for rotation, scroll for zoom
- Fullscreen mode support

**Frontend Implementation** (`web/js/equirect360_preview.js`):
```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.DiT360.Equirect360Preview",
    async nodeCreated(node) {
        if (node.comfyClass === "Equirect360Preview") {
            // Create Three.js scene for 360 viewing
            node.addWidget("button", "View 360°", "view", () => {
                createPanoramaViewer(node.imageData);
            });
        }
    }
});

function createPanoramaViewer(imageData) {
    // Three.js implementation
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    
    // Create sphere geometry with panorama texture
    const geometry = new THREE.SphereGeometry(500, 60, 40);
    geometry.scale(-1, 1, 1); // Invert for inside viewing
    
    const texture = new THREE.TextureLoader().load(imageData);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    const sphere = new THREE.Mesh(geometry, material);
    
    scene.add(sphere);
    camera.position.set(0, 0, 0);
    
    // Mouse controls for rotation
    // ... (implementation details)
}
```

---

## 6. File Structure

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-DiT360/
        ├── __init__.py                 # Entry point & node registration
        ├── nodes.py                    # Core node implementations
        ├── requirements.txt            # Python dependencies
        ├── install.py                  # Custom installation script
        ├── README.md                   # User documentation
        ├── LICENSE                     # Apache 2.0 license
        │
        ├── dit360/                     # Core DiT360 implementation
        │   ├── __init__.py
        │   ├── model.py                # DiT360 model architecture
        │   ├── sampler.py              # Sampling algorithms
        │   ├── conditioning.py         # Text encoding & conditioning
        │   ├── losses.py               # Yaw loss, cube loss implementations
        │   └── utils.py                # Utility functions
        │
        ├── utils/                      # Utility modules
        │   ├── __init__.py
        │   ├── equirect.py             # Equirectangular utilities
        │   ├── padding.py              # Circular padding implementations
        │   ├── validation.py           # Input validation
        │   └── paths.py                # Windows path handling
        │
        ├── web/                        # Frontend resources
        │   └── js/
        │       ├── equirect360_preview.js  # 360° viewer
        │       └── node_widgets.js         # Custom node widgets
        │
        ├── examples/                   # Example workflows
        │   ├── text_to_panorama.json
        │   ├── image_to_panorama.json
        │   ├── inpainting.json
        │   └── advanced_workflow.json
        │
        ├── tests/                      # Unit tests
        │   ├── __init__.py
        │   ├── test_nodes.py
        │   ├── test_equirect.py
        │   ├── test_padding.py
        │   └── test_windows_paths.py
        │
        └── docs/                       # Documentation
            ├── installation.md
            ├── usage.md
            ├── troubleshooting.md
            └── api_reference.md
```

---

## 7. Implementation Phases

### Phase 1: Foundation Setup (Week 1)
**Deliverables**:
- Project structure created
- `__init__.py` with NODE_CLASS_MAPPINGS
- Basic requirements.txt
- Model folder registration with folder_paths
- Empty node classes that load in ComfyUI

**Validation**:
- ComfyUI loads without errors
- Nodes appear in node menu under "DiT360" category
- No dependency conflicts with fresh ComfyUI install

### Phase 2: Model Loading Infrastructure (Week 1-2)
**Deliverables**:
- DiT360ModelLoader implementation
- DiT360TextEncoderLoader implementation
- DiT360VAELoader implementation
- Hugging Face Hub integration
- Automatic model download
- Model caching system

**Validation**:
- Models download automatically when missing
- Models load without errors
- Memory usage stays within expected bounds
- Device placement (GPU/CPU) works correctly
- Progress bars show during loading

### Phase 3: Core Generation Pipeline (Week 2-3)
**Deliverables**:
- DiT360Sampler node implementation
- Circular padding implementation
- Basic sampling loop (no geometric losses yet)
- VAE encode/decode integration
- Prompt encoding pipeline
- Progress reporting

**Validation**:
- Can generate 2048×1024 panoramas
- Generation completes without OOM errors
- Images have correct tensor shape [B, H, W, C]
- Basic prompt following works
- Seed produces reproducible results

### Phase 4: Geometric Features (Week 3-4)
**Deliverables**:
- DiT360SamplerAdvanced node
- Yaw loss implementation
- Cube loss implementation
- Configurable circular padding
- Edge blending utilities

**Validation**:
- Panoramas have seamless wraparound
- Rotational consistency improved with yaw loss
- Pole distortion reduced with cube loss
- Advanced sampler produces higher quality than basic

### Phase 5: Format Validation & Utilities (Week 4)
**Deliverables**:
- Equirect360Validator node
- Equirect360EdgeBlender node
- Aspect ratio fixing (crop/pad/stretch)
- Format conversion utilities
- Image range validation

**Validation**:
- Non-2:1 images correctly fixed
- Edge blending produces seamless results
- Validation catches common format errors
- Clear error messages for invalid inputs

### Phase 6: Interactive Viewing (Week 5)
**Deliverables**:
- Equirect360Preview node
- Three.js 360° viewer
- Frontend JavaScript integration
- Fullscreen mode
- Export functionality

**Validation**:
- Viewer loads panoramas correctly
- Mouse controls work smoothly
- Works in ComfyUI web interface
- Fullscreen mode functional
- Export produces valid panorama files

### Phase 7: Windows Compatibility (Week 5-6)
**Deliverables**:
- Path handling with pathlib
- Case-insensitive file search
- Long path support validation
- File locking prevention
- Windows-specific install.py
- CUDA installation validation

**Validation**:
- All tests pass on Windows 10/11
- Works with portable ComfyUI
- Paths with spaces and special chars work
- No file locking issues
- CUDA properly detected

### Phase 8: Advanced Features (Week 6-7)
**Deliverables**:
- Inpainting support
- Outpainting support
- Image-to-panorama pipeline
- LoRA loading
- Multiple precision support (fp8, bf16)
- Model offloading configuration

**Validation**:
- Inpainting produces coherent results
- Outpainting extends panoramas naturally
- Img2img maintains panoramic format
- LoRA weights apply correctly
- FP8 reduces VRAM as expected

### Phase 9: Testing & Documentation (Week 7-8)
**Deliverables**:
- Comprehensive test suite
- Unit tests for all modules
- Integration tests for workflows
- Windows-specific tests
- README.md with examples
- Troubleshooting guide
- API documentation
- Example workflows

**Validation**:
- All tests pass
- Documentation covers common issues
- Example workflows work out-of-box
- Clear installation instructions
- Known limitations documented

### Phase 10: Release & Distribution (Week 8)
**Deliverables**:
- GitHub repository setup
- LICENSE file (Apache 2.0)
- CHANGELOG.md
- ComfyUI Manager integration
- Release builds
- Issue templates
- Contributing guidelines

**Validation**:
- Installs via ComfyUI Manager
- GitHub releases work
- Issue tracker configured
- License compliance verified

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Module: utils/equirect.py**
```python
def test_validate_aspect_ratio():
    """Test 2:1 ratio validation"""
    assert validate_aspect_ratio(2048, 1024) == True
    assert validate_aspect_ratio(2000, 1000) == True
    assert validate_aspect_ratio(1920, 1080) == False

def test_blend_edges():
    """Test edge blending produces seamless wraparound"""
    image = torch.rand(1, 1024, 2048, 3)
    blended = blend_edges(image, blend_width=10)
    
    # Check left and right edges match
    left = blended[:, :, :10, :]
    right = blended[:, :, -10:, :]
    assert torch.allclose(left, right, atol=1e-4)

def test_equirect_to_cubemap():
    """Test equirectangular to cubemap conversion"""
    equirect = torch.rand(1, 1024, 2048, 3)
    cubemap = equirect_to_cubemap(equirect, face_size=512)
    
    assert cubemap.shape == (6, 512, 512, 3)
    assert cubemap.min() >= 0 and cubemap.max() <= 1
```

**Module: utils/padding.py**
```python
def test_circular_padding():
    """Test circular padding maintains continuity"""
    latent = torch.rand(1, 4, 128, 256)
    padded = apply_circular_padding(latent, padding=10)
    
    # Check wraparound continuity
    assert torch.allclose(
        padded[:, :, :, :10],
        latent[:, :, :, -10:],
        atol=1e-6
    )
```

**Module: dit360/model.py**
```python
def test_model_loading():
    """Test DiT360 model loads correctly"""
    model = load_dit360_model("test_model.safetensors", precision="fp16")
    assert model is not None
    assert next(model.parameters()).dtype == torch.float16

def test_model_inference():
    """Test model produces correct output shapes"""
    model = load_dit360_model("test_model.safetensors")
    latent = torch.rand(1, 4, 128, 256)
    conditioning = torch.rand(1, 77, 768)
    
    output = model(latent, conditioning)
    assert output.shape == latent.shape
```

### 8.2 Integration Tests

**Workflow: Text-to-Panorama**
```python
def test_text_to_panorama_workflow():
    """Test complete text-to-panorama generation"""
    # Load models
    model = DiT360ModelLoader.load("dit360_model.safetensors")
    encoder = DiT360TextEncoderLoader.load("t5_encoder.safetensors")
    vae = DiT360VAELoader.load("vae.safetensors")
    
    # Generate panorama
    latent, image = DiT360Sampler.generate(
        model=model,
        text_encoder=encoder,
        vae=vae,
        prompt="A beautiful sunset over the ocean",
        width=2048,
        height=1024,
        steps=50,
        seed=42
    )
    
    # Validate output
    assert image.shape == (1, 1024, 2048, 3)
    assert image.min() >= 0 and image.max() <= 1
    assert validate_aspect_ratio(2048, 1024)
    
    # Check wraparound continuity
    assert check_edge_continuity(image, threshold=0.05)
```

**Workflow: Inpainting**
```python
def test_inpainting_workflow():
    """Test inpainting workflow"""
    # Load models and base image
    base_image = load_test_panorama("test_panorama.png")
    mask = create_test_mask(width=512, height=512, x=1000, y=400)
    
    # Run inpainting
    result = DiT360Sampler.generate(
        model=model,
        vae=vae,
        prompt="A hot air balloon",
        latent_image=encode_image(base_image, vae),
        mask=mask,
        denoise=0.8
    )
    
    # Validate
    assert result.shape == base_image.shape
    assert not torch.allclose(result[:, 400:912, 1000:1512, :], 
                              base_image[:, 400:912, 1000:1512, :])
```

### 8.3 Windows-Specific Tests

```python
def test_windows_paths():
    """Test path handling on Windows"""
    test_paths = [
        "C:\\Users\\Test User\\ComfyUI\\models\\dit360",
        "C:/Users/Test User/ComfyUI/models/dit360",
        r"C:\Users\Test User\ComfyUI\models\dit360",
    ]
    
    for path in test_paths:
        p = Path(path)
        assert p.exists() or not p.exists()  # Should not error

def test_long_paths():
    """Test handling of long Windows paths"""
    long_path = "C:\\Users\\TestUser\\ComfyUI\\models\\" + "a" * 240 + "\\model.safetensors"
    assert len(long_path) > 260
    
    # Should handle gracefully
    try:
        p = Path(long_path)
        result = validate_path_length(str(p))
    except Exception as e:
        pytest.fail(f"Long path handling failed: {e}")

def test_case_insensitive_search():
    """Test case-insensitive file finding"""
    # Create test files
    test_dir = Path("test_models")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "TestModel.safetensors").touch()
    
    # Should find with different case
    result = find_file_case_insensitive(test_dir, "testmodel.safetensors")
    assert result is not None
    assert result.name == "TestModel.safetensors"
```

### 8.4 Performance Tests

```python
def test_memory_usage():
    """Test memory stays within expected bounds"""
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Baseline memory
    gc.collect()
    torch.cuda.empty_cache()
    baseline_vram = torch.cuda.memory_allocated()
    
    # Load model and generate
    model = DiT360ModelLoader.load("dit360_model.safetensors", precision="fp16")
    latent, image = DiT360Sampler.generate(
        model=model,
        prompt="Test panorama",
        steps=20
    )
    
    # Check peak memory
    peak_vram = torch.cuda.max_memory_allocated()
    vram_used_gb = (peak_vram - baseline_vram) / (1024**3)
    
    assert vram_used_gb < 20, f"VRAM usage {vram_used_gb:.2f}GB exceeds 20GB limit"

def test_generation_speed():
    """Test generation completes within time limits"""
    import time
    
    start = time.time()
    latent, image = DiT360Sampler.generate(
        model=model,
        prompt="Speed test panorama",
        width=2048,
        height=1024,
        steps=50
    )
    duration = time.time() - start
    
    assert duration < 180, f"Generation took {duration:.1f}s, exceeds 3min limit"
```

---

## 9. Error Handling & Validation

### 9.1 Common Error Scenarios

**Insufficient VRAM**
```python
def handle_oom_error():
    """Gracefully handle CUDA out of memory errors"""
    try:
        latent, image = model.generate(...)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "Insufficient VRAM for generation. Try:\n"
            "• Lower precision (fp16 → fp8)\n"
            "• Smaller resolution (2048×1024 → 1024×512)\n"
            "• Enable model offloading\n"
            f"Current VRAM: {torch.cuda.memory_allocated()/(1024**3):.1f}GB"
        )
```

**Missing Model Files**
```python
def check_model_exists(model_name):
    """Check if model exists, provide download instructions if not"""
    model_path = folder_paths.get_full_path("dit360", model_name)
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_name}\n\n"
            f"Download from: https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation\n"
            f"Place in: {folder_paths.get_folder_paths('dit360')[0]}"
        )
```

**Invalid Aspect Ratio**
```python
def validate_dimensions(width, height):
    """Validate panorama dimensions"""
    ratio = width / height
    
    if abs(ratio - 2.0) > 0.01:
        raise ValueError(
            f"Invalid aspect ratio: {width}×{height} ({ratio:.2f}:1)\n"
            f"Equirectangular panoramas must be 2:1 ratio.\n"
            f"Valid resolutions: 2048×1024, 4096×2048, 1024×512, etc."
        )
    
    if width % 64 != 0 or height % 64 != 0:
        raise ValueError(
            f"Dimensions must be multiples of 64\n"
            f"Got: {width}×{height}\n"
            f"Try: {(width//64)*64}×{(height//64)*64}"
        )
```

**CUDA Not Available**
```python
def check_cuda_available():
    """Check CUDA availability and version"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. DiT360 requires NVIDIA GPU with CUDA support.\n"
            "Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads\n"
            "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    
    cuda_version = torch.version.cuda
    if cuda_version < "11.8":
        print(f"Warning: CUDA {cuda_version} detected. Recommended: 11.8 or newer")
```

### 9.2 Input Validation

```python
def validate_inputs(self, **kwargs):
    """Validate all inputs before processing"""
    errors = []
    
    # Check dimensions
    width = kwargs.get("width")
    height = kwargs.get("height")
    if width and height:
        try:
            validate_dimensions(width, height)
        except ValueError as e:
            errors.append(str(e))
    
    # Check steps range
    steps = kwargs.get("steps")
    if steps and (steps < 1 or steps > 150):
        errors.append(f"Steps must be 1-150, got {steps}")
    
    # Check CFG scale
    cfg = kwargs.get("cfg_scale")
    if cfg and (cfg < 0 or cfg > 20):
        errors.append(f"CFG scale must be 0-20, got {cfg}")
    
    # Check prompt length
    prompt = kwargs.get("prompt", "")
    if len(prompt) > 1000:
        errors.append(f"Prompt too long: {len(prompt)} chars (max 1000)")
    
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))
```

---

## 10. Windows Compatibility Checklist

### 10.1 Path Handling
- [ ] All paths use `pathlib.Path` or `os.path.join`
- [ ] No hardcoded backslash paths (`C:\...`)
- [ ] Forward slashes used where possible (`C:/...`)
- [ ] Raw strings for Windows paths (`r"C:\..."`)
- [ ] Case-insensitive file search implemented
- [ ] Long path support validated (>260 chars)
- [ ] Path length validation warnings

### 10.2 File Operations
- [ ] Context managers for all file operations
- [ ] Files properly closed after reading
- [ ] No file locking issues
- [ ] Temp files cleaned up properly
- [ ] Safe concurrent file access

### 10.3 Dependencies
- [ ] Requirements.txt has loose version constraints
- [ ] No PyTorch reinstallation unless necessary
- [ ] CUDA version compatibility checked
- [ ] Visual C++ redistributables documented
- [ ] Portable Python support tested

### 10.4 Environment
- [ ] PYTORCH_CUDA_ALLOC_CONF configuration documented
- [ ] Environment variable handling works
- [ ] Portable installations supported
- [ ] Virtual environments supported

### 10.5 Testing
- [ ] All tests pass on Windows 10
- [ ] All tests pass on Windows 11
- [ ] Portable ComfyUI tested
- [ ] Standard installation tested
- [ ] Paths with spaces tested
- [ ] Special characters in paths tested

---

## 11. Deployment & Distribution

### 11.1 GitHub Repository Setup

**Repository Structure**:
```
ComfyUI-DiT360/
├── .github/
│   ├── workflows/
│   │   ├── tests.yml          # CI/CD tests
│   │   └── release.yml        # Release automation
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── [all node pack files]
├── LICENSE                     # Apache 2.0
├── README.md
├── CHANGELOG.md
└── CONTRIBUTING.md
```

### 11.2 ComfyUI Manager Integration

**pyproject.toml** (for ComfyUI Manager):
```toml
[project]
name = "ComfyUI-DiT360"
version = "1.0.0"
description = "DiT360 panoramic image generation for ComfyUI"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.28.1",
    "diffusers>=0.25.0",
    "safetensors>=0.4.2",
    "accelerate>=0.26.0",
    "huggingface-hub>=0.20.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/ComfyUI-DiT360"
Repository = "https://github.com/yourusername/ComfyUI-DiT360"
Issues = "https://github.com/yourusername/ComfyUI-DiT360/issues"
```

### 11.3 Release Process

1. **Version Bump**: Update version in `__init__.py` and `pyproject.toml`
2. **Update CHANGELOG.md**: Document all changes
3. **Run Tests**: Ensure all tests pass on Windows and Linux
4. **Create Tag**: `git tag -a v1.0.0 -m "Release v1.0.0"`
5. **Push Tag**: `git push origin v1.0.0`
6. **GitHub Release**: Create release with binaries and changelog
7. **Announce**: Post in ComfyUI Discord and Reddit

---

## 12. Known Limitations & Future Work

### 12.1 Current Limitations

**Performance**:
- Requires 16-24GB VRAM for inference
- Generation takes 1-2 minutes for 50 steps
- Model loading takes 30-60 seconds

**Features**:
- No multi-batch generation support
- Limited LoRA compatibility
- No ControlNet support yet
- Single precision per session

**Compatibility**:
- Tested only on NVIDIA GPUs (no AMD/Intel)
- Windows 10/11 and Linux only (no macOS)
- CUDA 11.8+ required

### 12.2 Future Enhancements

**Performance Optimizations**:
- [ ] Implement attention slicing for lower VRAM
- [ ] Add xFormers memory efficient attention
- [ ] Support model quantization (int8, int4)
- [ ] Batch generation support
- [ ] Persistent model caching

**Features**:
- [ ] ControlNet integration for guided generation
- [ ] Depth-to-panorama pipeline
- [ ] Video panorama generation (360° videos)
- [ ] HDR panorama support
- [ ] Lighting control (time of day, weather)
- [ ] Style transfer for panoramas

**Compatibility**:
- [ ] AMD ROCm support
- [ ] Intel Arc GPU support
- [ ] Apple Silicon (MPS) support
- [ ] CPU-only mode (for testing)

**Usability**:
- [ ] Web UI for quick generation
- [ ] Preset library (indoor, outdoor, sci-fi, etc.)
- [ ] Prompt templates and examples
- [ ] One-click model installer
- [ ] Quality presets (fast/balanced/quality)

---

## 13. Appendix

### 13.1 Glossary

- **DiT**: Diffusion Transformer architecture
- **Equirectangular**: Spherical projection mapping sphere to 2D rectangle (2:1 ratio)
- **Circular Padding**: Wrapping edges for seamless panorama boundaries
- **Yaw Loss**: Loss function ensuring rotational consistency
- **Cube Loss**: Loss function for multi-scale distortion awareness
- **VAE**: Variational Autoencoder for latent space compression
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **CFG**: Classifier-Free Guidance for prompt adherence

### 13.2 Reference Links

- **DiT360 Paper**: https://arxiv.org/abs/2510.11712
- **DiT360 GitHub**: https://github.com/Insta360-Research-Team/DiT360
- **DiT360 Models**: https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation
- **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **ComfyUI Documentation**: https://docs.comfy.org/

### 13.3 License

This project is licensed under Apache License 2.0.

DiT360 model is subject to FLUX.1-dev license terms.

### 13.4 Acknowledgments

- Insta360 Research Team for DiT360 model
- Black Forest Labs for FLUX.1-dev base model
- ComfyUI community for node pack patterns
- Kijai for OpenDiTWrapper and WanVideoWrapper examples
