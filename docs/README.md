# ComfyUI-DiT360

Generate high-fidelity 360-degree panoramic images using the DiT360 diffusion transformer model in ComfyUI.

![DiT360 Banner](docs/banner.png)
*High-quality equirectangular panoramas with seamless wraparound*

## 🌟 Features

- **Text-to-Panorama**: Generate 360° panoramas from text prompts
- **Image-to-Panorama**: Transform existing images into panoramas
- **Seamless Wraparound**: Automatic edge blending for perfect 360° continuity
- **Advanced Quality Options**: Yaw loss and cube loss for enhanced consistency
- **Multiple Precisions**: Support for fp16, bf16, fp32, and fp8
- **Optimization Controls**: Switchable attention backends, attention slicing, optional int8/int4 quantization, and VAE tiling
- **Windows Compatible**: Proper path handling for Windows systems
- **Minimal Node Count**: Only 6 nodes for simple workflows

## 📋 Requirements

### Minimum System Requirements

- **OS**: Windows 10/11 (64-bit) or Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, 4080, etc.)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16GB system memory
- **Storage**: 100GB free SSD space
- **Python**: 3.9 - 3.12

### Recommended System Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A5000, etc.)
- **RAM**: 32GB system memory
- **Storage**: 250GB NVMe SSD

## 🚀 Installation

### Step 1: Install ComfyUI

If you don't have ComfyUI installed:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### Step 2: Install ComfyUI-DiT360

Navigate to ComfyUI's custom_nodes directory and clone this repository:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-DiT360.git
cd ComfyUI-DiT360
pip install -r requirements.txt
```

### Step 3: Download Models

DiT360 requires three model components. Create the following directory structure:

```
ComfyUI/models/
├── dit360/
│   └── dit360_model.safetensors  (Download from HuggingFace)
├── vae/
│   └── dit360_vae.safetensors    (Download from HuggingFace)
└── t5/
    └── t5-v1_1-xxl/               (Download from HuggingFace)
```

**Download Links**:

1. **DiT360 Model** (~24GB):
   ```
   https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation
   ```
   Download and place in `ComfyUI/models/dit360/`

2. **VAE** (~400MB):
   ```
   https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation
   ```
   Download and place in `ComfyUI/models/vae/`

3. **T5-XXL Text Encoder** (~5GB):
   ```
   https://huggingface.co/city96/t5-v1_1-xxl-encoder-bf16
   ```
   Download entire folder and place in `ComfyUI/models/t5/`

### Step 4: Restart ComfyUI

Restart ComfyUI to load the new nodes:

```bash
cd ComfyUI
python main.py
```

You should see nodes under the "DiT360" category in the node menu.

## 📖 Quick Start

### Basic Text-to-Panorama Workflow

1. **Add DiT360Loader** node
   - Set paths to your downloaded models
   - Choose precision (fp16 recommended)

2. **Add DiT360TextEncode** node
   - Connect to DiT360Loader
   - Enter your prompt: `"A beautiful sunset over the ocean with palm trees"`

3. **Add DiT360Sampler** node
   - Connect pipeline and conditioning
   - Set resolution: 2048×1024 (must be 2:1 ratio)
   - Set steps: 50 (higher = better quality)

4. **Add DiT360Decode** node
   - Connect sampler output
   - Enable auto_blend_edges

5. **Add SaveImage** node (stock ComfyUI)
   - Save your panorama!

See `examples/text_to_panorama_basic.json` for a complete workflow.

## 🎨 Node Reference

### 1. DiT360Loader

Loads the DiT360 model, VAE, and text encoder.

**Parameters**:
- `model_path`: Path to DiT360 model (searches in models/dit360/)
- `precision`: fp16 (recommended), bf16, fp32, or fp8
- `vae_path`: Path to VAE model
- `t5_path`: Path to T5 text encoder
- `offload_to_cpu`: True = save VRAM, False = faster

**Outputs**: `dit360_pipe` - Pipeline object containing all models

---

### 2. DiT360TextEncode

Encodes text prompts using T5-XXL.

**Parameters**:
- `dit360_pipe`: Pipeline from DiT360Loader
- `prompt`: Positive prompt (what you want)
- `negative_prompt`: Negative prompt (what to avoid)
- `max_length`: Max tokens (default 512)

**Outputs**: `conditioning` - Encoded prompt embeddings

---

### 3. DiT360Sampler

Generates panoramic latents using DiT360.

**Parameters**:
- `width`: Output width (must be 2× height)
- `height`: Output height (must be width ÷ 2)
- `steps`: Denoising steps (20-30 fast, 50 balanced, 100+ quality)
- `cfg_scale`: Guidance scale (3.0-5.0 recommended)
- `seed`: Random seed for reproducibility
- `circular_padding`: Edge padding for seamlessness (10 recommended)

**Advanced Options** (optional):
- `enable_yaw_loss`: Rotational consistency
- `enable_cube_loss`: Pole distortion reduction
- `latent_image`: Input latent for img2img
- `denoise`: Denoising strength (1.0 = full, 0.5 = half)

**Outputs**: `latent` - Generated latent tensor

---

### 4. DiT360Decode

Decodes latents to panoramic images.

**Parameters**:
- `samples`: Latent from DiT360Sampler
- `dit360_pipe`: Pipeline (contains VAE)
- `auto_blend_edges`: Automatic edge blending (recommended: True)

**Outputs**: `image` - Panoramic image (ComfyUI format)

---

### 5. Equirect360Process

Validates and processes equirectangular panoramas.

**Parameters**:
- `enforce_2_1_ratio`: Force 2:1 aspect ratio
- `fix_mode`: How to fix ratio (pad/crop/stretch/none)
- `blend_edges`: Enable edge blending
- `blend_width`: Blend region width (10 recommended)
- `blend_mode`: Blend curve (cosine/linear/smooth)

**Outputs**: `image` - Processed panorama

---

### 6. Equirect360Preview

Preview panoramic images (standard ComfyUI preview for now).

**Parameters**:
- `images`: Panorama to preview

*Note: Interactive 360° viewer coming in future update*

## 💡 Usage Tips

### Resolution Recommendations

| Use Case | Resolution | VRAM | Quality |
|----------|-----------|------|---------|
| **Fast Preview** | 1024×512 | ~12GB | Good |
| **Standard** | 2048×1024 | ~16GB | Excellent |
| **High Quality** | 4096×2048 | ~24GB+ | Outstanding |

### Prompt Tips

**Good prompts**:
- Describe the full 360° environment
- Mention specific directions: "mountains to the north, ocean to the south"
- Include lighting: "sunset lighting from the west"

**Examples**:
```
"A cozy living room with large windows showing a forest view,
warm afternoon sunlight, wooden furniture, plants"

"Standing in a futuristic city square, tall skyscrapers all around,
neon lights, night time, cyberpunk style"

"Inside a medieval castle throne room, stone walls, torches,
tapestries, high vaulted ceiling, dramatic lighting"
```

### Performance Optimization

1. **Lower VRAM usage**:
   - Use fp16 or fp8 precision
   - Enable `offload_to_cpu`
   - Use smaller resolution (1024×512)
    - Set `attention_slice_size` to 256–512
    - Load with `quantization_mode=int8` (or `int4` if bitsandbytes is available)
    - Decode with `tiling_mode=always`

2. **Higher quality**:
   - Use bf16 or fp32 precision
   - Enable yaw_loss and cube_loss
   - Increase steps to 100+
   - Use higher resolution (4096×2048)

3. **Faster generation**:
   - Reduce steps to 20-30
   - Disable yaw_loss and cube_loss
    - Use fp16 precision
    - Switch to `scheduler_type=ddim` with `scheduler_eta=0.0`
    - Choose `attention_backend=flash` or `xformers` when available

4. **Memory diagnostics**:
   - Toggle `log_memory_stats` in the sampler to print allocated/peak VRAM
   - Record results in `docs/benchmarks/phase4_baseline.md` for regression tracking

## 🔧 Troubleshooting

### "Model not found" error

**Solution**: Check that models are in the correct directories:
```
ComfyUI/models/dit360/dit360_model.safetensors
ComfyUI/models/vae/dit360_vae.safetensors
ComfyUI/models/t5/t5-v1_1-xxl/
```

### "CUDA out of memory" error

**Solutions**:
1. Lower precision: fp16 → fp8
2. Enable `offload_to_cpu` in DiT360Loader
3. Reduce resolution: 2048×1024 → 1024×512
4. Close other GPU applications
5. Restart ComfyUI to clear CUDA cache

### "Invalid aspect ratio" warning

**Solution**: Ensure width is exactly 2× height:
- ✅ Valid: 2048×1024, 4096×2048, 1024×512
- ❌ Invalid: 1920×1080, 1024×768

### Visible seam at edges

**Solutions**:
1. Enable `auto_blend_edges` in DiT360Decode
2. Use Equirect360Process node with edge blending
3. Increase `blend_width` to 20+
4. Enable `circular_padding` in sampler

## 📚 Example Workflows

Located in the `examples/` directory:

1. **text_to_panorama_basic.json** - Simple text-to-panorama
2. **text_to_panorama_advanced.json** - With quality enhancements
3. **image_to_panorama.json** - Transform images to panoramas
4. **batch_generation.json** - Generate multiple panoramas

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0.

The DiT360 model is subject to FLUX.1-dev license terms.

## 🙏 Acknowledgments

- Insta360 Research Team for the DiT360 model
- Black Forest Labs for FLUX.1-dev base model
- ComfyUI community for the amazing framework
- Kijai for OpenDiTWrapper reference implementation

## 📞 Support

- **Issues**: https://github.com/yourusername/ComfyUI-DiT360/issues
- **Discussions**: https://github.com/yourusername/ComfyUI-DiT360/discussions
- **Discord**: [ComfyUI Discord](https://discord.gg/comfyui)

## 🗺️ Roadmap

- [x] Phase 1: Basic structure and nodes
- [x] Phase 2: Model loading and HuggingFace integration
- [x] Phase 3: Text encoding with T5-XXL
- [x] Phase 4: Core generation with circular padding
- [ ] Phase 5: Optimization & memory efficiency (attention backends, tiling)
- [ ] Phase 6: Interactive 360° viewer (Three.js)
- [ ] Phase 7: Advanced features (yaw loss, cube loss)
- [ ] Phase 8: Inpainting and outpainting support
- [ ] Phase 9: LoRA support
- [ ] Phase 10: ControlNet integration

---

**Version**: 0.1.0 (Alpha)
**Status**: Phase 1 Complete - Basic Structure ✅

For detailed documentation, see the `docs/` directory.
