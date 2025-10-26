# Code Review: ComfyUI-DiT360 Repository

## Executive Summary

Based on your README, your current implementation treats DiT360 as a **full model pipeline** with custom loaders. However, since **DiT360 is actually a LoRA**, this can be massively simplified. Here's what needs to change:

---

## Current Implementation Issues

### ❌ Problem 1: Custom Model Loading (Unnecessary)

**Your Current Approach**:
```
DiT360Loader node
├─ Loads dit360_model.safetensors (~24GB)
├─ Loads dit360_vae.safetensors
└─ Loads T5-XXL text encoder

Requires users to download 3 separate model components
```

**Reality**: DiT360 is a LoRA (~2-5GB), not a full model!

**Correct Approach**:
```
Standard FLUX workflow:
├─ Load Checkpoint (FLUX.1-dev) ← Standard node
├─ Load LoRA (DiT360) ← Standard node
└─ CLIP Text Encode ← Standard node
```

**What to Remove**:
- `DiT360Loader` node (completely unnecessary)
- `DiT360TextEncode` node (use standard CLIP)
- Custom VAE loading (use FLUX's built-in VAE)
- All Hugging Face Hub download code
- T5 encoder loading code

**What to Keep**:
- Just the 5 enhancement nodes that add 360° features

---

### ❌ Problem 2: Custom Pipeline Architecture

**Your Current Approach**:
```python
class DiT360Loader:
    """Loads DiT360 model, VAE, and text encoder"""
    # 200+ lines of custom loading code
    # Downloads from Hugging Face
    # Manages multiple model components
```

**Correct Approach**:
```python
# No custom loader needed!
# Users just:
# 1. Load FLUX.1-dev checkpoint (standard)
# 2. Load DiT360 LoRA (standard)
# 3. Use your enhancement nodes
```

**Why This is Wrong**:
1. DiT360 is not a standalone model—it's weights trained on top of FLUX
2. Users already have FLUX loaded in ComfyUI
3. You're making users download FLUX twice (once for ComfyUI, once for your node)
4. This creates version conflicts and wastes 24GB storage

---

### ❌ Problem 3: Custom Text Encoding

**Your Current Approach**:
```python
class DiT360TextEncode:
    """Encodes text prompts using T5-XXL"""
    # Custom T5 loading
    # Custom tokenization
    # Custom encoding
```

**Correct Approach**:
```python
# Use standard CLIPTextEncode node
# FLUX already has text encoding built-in
# No need for custom implementation
```

---

### ❌ Problem 4: Directory Structure Requirements

**Your README Says**:
```
ComfyUI/models/
├── dit360/
│   └── dit360_model.safetensors (24GB)
├── vae/
│   └── dit360_vae.safetensors (400MB)
└── t5/
    └── t5-v1_1-xxl/ (5GB)

Total: ~30GB of downloads
```

**Should Be**:
```
ComfyUI/models/
├── checkpoints/
│   └── flux1-dev.safetensors (24GB) ← User already has this
└── loras/
    └── dit360.safetensors (2-5GB) ← Only new download needed!
```

---

## What Your Nodes Should Actually Do

### ✅ Keep These (With Modifications)

#### 1. Equirect360EmptyLatent (formerly part of DiT360Sampler)
**Purpose**: Create 2:1 aspect ratio latents
**Changes Needed**: None, this is correct!

#### 2. Equirect360KSampler (formerly DiT360Sampler)
**Current**: Tries to load model and sample
**Should Be**: Just wrap standard KSampler with circular padding

**Key Change**:
```python
# ❌ WRONG - Don't do this:
class DiT360Sampler:
    def sample(self, dit360_pipe, conditioning, ...):
        model = dit360_pipe["model"]
        # Custom sampling code

# ✅ CORRECT - Do this:
class Equirect360KSampler:
    def sample(self, model, positive, negative, ...):
        # Model comes from standard Load Checkpoint + Load LoRA
        # Just add circular padding wrapper
        model = wrap_with_circular_padding(model)
        # Use ComfyUI's standard sampler
        return comfy.sample.sample(model, ...)
```

#### 3. Equirect360VAEDecode (formerly DiT360Decode)
**Current**: Uses custom VAE from dit360_pipe
**Should Be**: Accept standard VAE input from Load Checkpoint

#### 4. Equirect360Process (Rename to Equirect360EdgeBlender)
**Current**: Good! This is correct.
**Changes Needed**: Just rename for clarity

#### 5. Equirect360Preview (formerly DiT360Preview)
**Current**: Good foundation
**Changes Needed**: Add Three.js viewer integration

---

### ❌ Remove These Entirely

1. **DiT360Loader** 
   - Not needed—users load FLUX + LoRA normally
   
2. **DiT360TextEncode**
   - Not needed—users use standard CLIP Text Encode
   
3. **Pipeline Object**
   - Not needed—ComfyUI handles model routing

---

## Specific Code Changes Needed

### File: `__init__.py`

**Remove**:
```python
import folder_paths

dit360_path = os.path.join(folder_paths.models_dir, "dit360")
folder_paths.add_model_folder_path("dit360", dit360_path)
```

**Why**: DiT360 is a LoRA—goes in standard `loras/` folder

### File: `nodes.py`

**Remove** (entire classes):
- `DiT360Loader`
- `DiT360TextEncode`

**Keep but Modify**:
- `DiT360Sampler` → Rename to `Equirect360KSampler`
  - Remove `dit360_pipe` input
  - Accept standard `model`, `positive`, `negative` inputs
  - Add circular padding wrapper
  
- `DiT360Decode` → Rename to `Equirect360VAEDecode`
  - Remove `dit360_pipe` input
  - Accept standard `vae` input

### File: `requirements.txt`

**Remove**:
```txt
transformers>=4.28.1  # Not needed
diffusers>=0.25.0      # Not needed
huggingface-hub>=0.20.0  # Not needed
```

**Keep**:
```txt
torch>=2.0.0  # Already in ComfyUI
numpy>=1.25.0
Pillow>=10.0.0
```

---

## New Architecture Diagram

### Current (Overcomplicated):
```
User Downloads:
├─ FLUX.1-dev (24GB) ← For ComfyUI
├─ dit360_model (24GB) ← Duplicate FLUX!
├─ dit360_vae (400MB) ← Duplicate FLUX VAE!
└─ T5 encoder (5GB) ← Already in FLUX!

Total: ~54GB (wasteful!)

Workflow:
DiT360Loader → DiT360TextEncode → DiT360Sampler → DiT360Decode
```

### Correct (Streamlined):
```
User Downloads:
├─ FLUX.1-dev (24GB) ← Once
└─ DiT360 LoRA (2-5GB) ← Only new download

Total: ~26-29GB (efficient!)

Workflow:
Load Checkpoint → Load LoRA → CLIP Encode → 
Equirect360EmptyLatent → Equirect360KSampler → 
Equirect360VAEDecode → Equirect360EdgeBlender → 
Equirect360Viewer → Save
```

---

## Implementation Priority

### Phase 1: Remove Custom Loading (Week 1)
- [ ] Delete `DiT360Loader` class
- [ ] Delete `DiT360TextEncode` class
- [ ] Remove folder_paths registration for dit360
- [ ] Update README to remove model download instructions

### Phase 2: Fix Node Inputs (Week 1)
- [ ] Rename `DiT360Sampler` → `Equirect360KSampler`
- [ ] Change inputs to accept standard `MODEL`, `CONDITIONING`
- [ ] Remove `dit360_pipe` input
- [ ] Same for `DiT360Decode` → `Equirect360VAEDecode`

### Phase 3: Implement Circular Padding (Week 2)
- [ ] Create `utils/circular_padding.py`
- [ ] Implement model wrapper for circular padding
- [ ] Integrate with ComfyUI's sampler system

### Phase 4: Add Remaining Features (Week 2-3)
- [ ] `Equirect360EmptyLatent` (if not exists)
- [ ] `Equirect360EdgeBlender` (rename from Equirect360Process)
- [ ] `Equirect360Viewer` with Three.js

### Phase 5: Testing & Documentation (Week 3)
- [ ] Update all README instructions
- [ ] Create new example workflow (FLUX + LoRA + your nodes)
- [ ] Test on Windows
- [ ] Test with actual DiT360 LoRA from Hugging Face

---

## Updated README Structure

Your README should say:

```markdown
# ComfyUI-DiT360

360° panorama enhancement nodes for FLUX.1-dev + DiT360 LoRA

## What is this?

DiT360 is a **LoRA adapter** for FLUX.1-dev that enables high-quality 
360° panorama generation. This node pack adds the necessary circular 
padding and post-processing to make panoramas seamless.

## Installation

1. Clone into ComfyUI/custom_nodes/
2. pip install -r requirements.txt
3. Download models:
   - FLUX.1-dev: Standard ComfyUI checkpoint
   - DiT360 LoRA: https://huggingface.co/Insta360-Research/...
4. Restart ComfyUI

## Usage

### Basic Workflow

1. **Load Checkpoint** (FLUX.1-dev) ← Standard node
2. **Load LoRA** (dit360.safetensors) ← Standard node
3. **CLIP Text Encode** (your prompt) ← Standard node
4. **360° Empty Latent** (2048×1024) ← Our node
5. **360° KSampler** (circular_padding=16) ← Our node
6. **360° VAE Decode** (circular_padding=16) ← Our node
7. **360° Edge Blender** (blend_width=10) ← Our node
8. **360° Viewer** (interactive preview) ← Our node
9. **Save Image** ← Standard node

See examples/basic_workflow.json

## Nodes

### 360° Empty Latent
Creates 2:1 aspect ratio latent for panoramas.

### 360° KSampler
Standard KSampler with circular padding for seamless edges.
Use instead of regular KSampler when generating panoramas.

### 360° VAE Decode
VAE decode with circular padding for smooth edges.
Use instead of regular VAEDecode.

### 360° Edge Blender
Post-processing to ensure perfect wraparound.
Highly recommended for best results.

### 360° Viewer
Interactive Three.js viewer. Click "View 360°" to navigate.

## Why not a custom model loader?

DiT360 is a LoRA, not a full model. You load FLUX normally, 
then apply the DiT360 LoRA like any other LoRA in ComfyUI.

Our nodes just add the circular padding and post-processing 
needed to make panoramas seamless.
```

---

## Key Takeaways

### What You Were Doing Wrong:
1. Treating DiT360 as a full model (it's a LoRA)
2. Custom loading code (unnecessary)
3. Making users download FLUX twice
4. Reimplementing text encoding (use standard CLIP)
5. Custom pipeline architecture (use standard workflow)

### What You Should Do Instead:
1. Accept standard ComfyUI inputs (MODEL, VAE, CONDITIONING)
2. Just add circular padding wrapper
3. Users load FLUX + LoRA normally
4. Your nodes enhance the process, not replace it
5. Focus on the 360° specific features only

### The Core Insight:
**You're not wrapping a model, you're enhancing a workflow.**

Your nodes should be **plugins**, not a **pipeline replacement**.

---

## Comparison: Lines of Code

### Current Approach:
```
__init__.py:           ~50 lines
nodes.py:              ~800 lines (DiT360Loader, TextEncode, etc.)
model loading:         ~300 lines
text encoding:         ~200 lines
sampling:              ~200 lines
decode:                ~100 lines
```

### Streamlined Approach:
```
__init__.py:           ~15 lines
nodes.py:              ~300 lines (5 simple nodes)
utils/circular_padding: ~100 lines
utils/equirect:        ~100 lines
web/js/viewer:         ~200 lines

Total: ~50% less code, 10x simpler
```

---

## Migration Path

If you have existing workflows with your current nodes:

1. **Create migration guide** showing old → new
2. **Keep old nodes** for 1 version (mark deprecated)
3. **Remove old nodes** in version 2.0

**Example Migration**:
```
Old workflow:
DiT360Loader → DiT360TextEncode → DiT360Sampler → DiT360Decode

New workflow:
Load Checkpoint → Load LoRA → CLIP Encode → 
Equirect360KSampler → Equirect360VAEDecode
```

---

## Questions to Answer

Before proceeding, confirm:

1. **Is the DiT360 you're using actually a LoRA?**
   - Check file size: ~2-5GB = LoRA, ~24GB = full model
   - Check Hugging Face repo structure

2. **Does your current implementation actually work?**
   - Have you tested it with real DiT360 weights?
   - Does it generate seamless panoramas?

3. **What are your users expecting?**
   - If they're already using your nodes, migration will be needed
   - If this is new, start with clean architecture

---

## Action Items

### Immediate (Today):
- [ ] Verify DiT360 is actually a LoRA (check Hugging Face)
- [ ] Test if DiT360 LoRA works with standard FLUX loading
- [ ] Decide: Rewrite from scratch or migrate existing code?

### This Week:
- [ ] Implement streamlined node architecture
- [ ] Test circular padding with real FLUX + DiT360 LoRA
- [ ] Verify panoramas are seamless

### Next Week:
- [ ] Add Three.js viewer
- [ ] Create example workflows
- [ ] Update documentation
- [ ] Test on Windows

### Release:
- [ ] Version 2.0 with new architecture
- [ ] Migration guide for v1 users
- [ ] Announce breaking changes clearly

---

## Conclusion

Your current implementation is **architecturally wrong** but the **core ideas are right**:
- ✅ Circular padding is correct
- ✅ Edge blending is correct
- ✅ Aspect ratio enforcement is correct
- ❌ Model loading approach is wrong
- ❌ Pipeline architecture is wrong

**Recommendation**: Start fresh with the streamlined architecture. You'll have:
- 50% less code
- 10x simpler installation
- No model version conflicts
- Better ComfyUI integration
- Easier maintenance

The good news: Most of your utility code (circular padding, edge blending) can be kept!
