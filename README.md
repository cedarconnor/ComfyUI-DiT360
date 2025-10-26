# ComfyUI-DiT360

**360° Panorama Generation Enhancement Nodes for ComfyUI**

Generate seamless equirectangular panoramic images using FLUX.1-dev with the DiT360 LoRA adapter.

---

## 🌟 Features

- ✅ **Circular Padding**: Seamless wraparound at panorama edges
- ✅ **2:1 Aspect Ratio Enforcement**: Automatic equirectangular format
- ✅ **Edge Blending**: Perfect continuity at boundaries
- ✅ **Interactive 360° Viewer**: Three.js-based panorama navigation (coming soon)
- ✅ **Optional Geometric Losses**: Yaw/cube loss for quality improvements (slower)

---

## 📦 What is This?

**DiT360 is a LoRA adapter** (~2-5GB) for FLUX.1-dev that enables high-quality 360° panorama generation. This node pack provides enhancement nodes that add the necessary circular padding and post-processing to make panoramas seamless.

**This is NOT a full model loader**—you use standard ComfyUI nodes to load FLUX and apply the DiT360 LoRA, then use our enhancement nodes for 360° specific features.

---

## 🚀 Installation

### 1. Install ComfyUI
If you haven't already, install [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

### 2. Clone This Repository
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-DiT360.git
cd ComfyUI-DiT360
pip install -r requirements.txt
```

### 3. Download Models
- **FLUX.1-dev**: Place in `ComfyUI/models/checkpoints/`
  Download from [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev)

- **DiT360 LoRA**: Place in `ComfyUI/models/loras/`
  Download from [Hugging Face](https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation)

### 4. Restart ComfyUI
```bash
python main.py
```

You should see: `✅ ComfyUI-DiT360 v2.0.0 loaded`

---

## 📖 Usage

### Basic Workflow

1. **Load Checkpoint** → Select FLUX.1-dev *(standard node)*
2. **Load LoRA** → Select dit360.safetensors, strength 1.0 *(standard node)*
3. **CLIP Text Encode** → Enter your prompt *(standard node)*
4. **Equirect360EmptyLatent** → Creates 2048×1024 latent *(our node)*
5. **Equirect360KSampler** → Sample with circular padding *(our node)*
6. **Equirect360VAEDecode** → Decode with circular padding *(our node)*
7. **Equirect360EdgeBlender** → Blend edges for perfect wraparound *(our node)*
8. **Equirect360Viewer** → Interactive 360° preview *(our node, coming soon)*
9. **Save Image** → Standard save *(standard node)*

### Example Workflow
See `examples/basic_workflow.json` *(coming soon)*

---

## 🎨 Node Descriptions

### 1. Equirect360EmptyLatent
Creates empty latent with enforced 2:1 aspect ratio.
- **Input**: Width (e.g., 2048), Batch size
- **Output**: Latent (auto-calculated height = width/2)
- **Use instead of**: EmptyLatentImage

### 2. Equirect360KSampler
Standard KSampler with circular padding for seamless edges.
- **Inputs**: Model, conditioning, latent, sampler settings
- **Key Parameter**: `circular_padding` (16-32 recommended)
- **Optional**: `enable_yaw_loss`, `enable_cube_loss` (slower, higher quality)
- **Use instead of**: KSampler

### 3. Equirect360VAEDecode
VAE decode with circular padding for smooth edges.
- **Inputs**: Samples (latent), VAE, circular_padding
- **Use instead of**: VAEDecode

### 4. Equirect360EdgeBlender
Post-processing to ensure perfect wraparound.
- **Inputs**: Image, blend_width (10-20 px), blend_mode (cosine/linear/smooth)
- **Highly recommended** for best results!

### 5. Equirect360Viewer *(coming soon)*
Interactive Three.js viewer for 360° navigation.
- **Input**: Image
- **Features**: Mouse drag, scroll zoom, fullscreen

---

## 💡 Prompting Tips

Describe the full 360° environment in your prompts:

**Good Examples**:
```
"A cozy mountain cabin interior, large windows showing snowy peaks,
warm fireplace, wooden furniture, morning light, 360 degree panorama"

"Standing in a futuristic city plaza, skyscrapers all around,
neon signs, rain-slicked streets, night time, cyberpunk, 360 panorama"

"Ancient library with towering bookshelves on all sides, spiral staircases,
warm lighting from chandeliers, dusty atmosphere, 360 degree view"
```

---

## ⚙️ System Requirements

### Minimum
- **GPU**: NVIDIA with 12GB VRAM (RTX 3060 12GB, 3080, 4070)
- **RAM**: 16GB system memory
- **Storage**: 30GB free space (FLUX + LoRA)
- **OS**: Windows 10/11 or Linux

### Recommended
- **GPU**: NVIDIA with 16GB+ VRAM (RTX 4080, 4090)
- **RAM**: 32GB system memory
- **Storage**: 50GB NVMe SSD

### Resolution Guide
| Resolution   | VRAM  | Speed  | Quality     |
|-------------|-------|--------|-------------|
| 1024×512    | 12GB  | Fast   | Good        |
| 2048×1024   | 16GB  | Medium | Excellent   |
| 4096×2048   | 24GB+ | Slow   | Outstanding |

---

## 🐛 Troubleshooting

### Visible Seam at Edges
- **Solution**: Increase `circular_padding` to 24-32
- **Solution**: Increase `blend_width` to 20+
- **Solution**: Enable `enable_yaw_loss` (slower)

### Out of Memory
- **Solution**: Lower resolution (1024×512)
- **Solution**: Use fp8 precision for FLUX
- **Solution**: Disable yaw/cube losses

### Not Seamless in Viewer
- **Check**: DiT360 LoRA is loaded and strength = 1.0
- **Check**: `circular_padding` > 0
- **Check**: Using Equirect360EdgeBlender

---

## 📚 Documentation

- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Step-by-step development guide
- **[Technical Design](docs/TECHNICAL_DESIGN.md)** - Architecture and specifications
- **[Migration from V1](docs/MIGRATION_FROM_V1.md)** - Upgrading from old architecture
- **[Docs Overview](docs/README.md)** - Documentation summary

---

## 🗺️ Roadmap

### v2.0.0 (Current - In Progress)
- [x] Documentation cleanup
- [x] Core utilities (circular padding, edge blending)
- [ ] 5 clean nodes (refactoring in progress)
- [ ] Three.js 360° viewer
- [ ] Example workflows

### v2.1.0 (Future)
- [ ] Yaw loss implementation
- [ ] Cube loss implementation
- [ ] ControlNet integration
- [ ] Inpainting support

### v2.2.0 (Future)
- [ ] Optimization (xFormers, attention slicing)
- [ ] Batch generation support
- [ ] Video panorama generation

---

## 📄 License

Apache License 2.0

**Models**:
- FLUX.1-dev: Subject to Black Forest Labs license
- DiT360 LoRA: Subject to Insta360 Research license

---

## 🙏 Credits

- **DiT360 LoRA**: [Insta360 Research Team](https://huggingface.co/Insta360-Research)
- **FLUX.1-dev**: [Black Forest Labs](https://huggingface.co/black-forest-labs)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ComfyUI-DiT360/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ComfyUI-DiT360/discussions)
- **Discord**: ComfyUI Discord #custom-nodes

---

**⭐ If you find this useful, please star the repository!**
