# Circular Padding Research for DiT360 in ComfyUI

## Executive Summary

**The current implementation won't work properly because it only pads the input/output latents but doesn't address the core issue: the transformer's attention mechanism doesn't understand circular topology.**

FLUX.1 is a Diffusion Transformer (DiT) architecture - it uses attention between tokens, not just convolutions. Simply padding the input doesn't make the attention "wrap around."

## The Problem

### Why the Current Approach Fails

The current `create_circular_padding_wrapper` in `utils/padding.py`:
```python
def wrapped_apply_model(x, t, **kwargs):
    x_padded = apply_circular_padding(x, circular_padding)
    output_padded = original_apply_model(x_padded, t, **kwargs)
    return remove_circular_padding(output_padded, circular_padding)
```

This pads the **latent tensor** before feeding it to the model, but:

1. **FLUX.1 uses transformers with self-attention**, not UNets with convolutions
2. The attention mechanism computes relationships between ALL tokens (positions)
3. **Tokens at position 0 (left edge) don't know they should attend to tokens at position N-1 (right edge) as neighbors**
4. The positional embeddings (RoPE in FLUX) encode absolute positions, not circular topology

### FLUX.1 Architecture (DiT)

From [arXiv research](https://arxiv.org/html/2507.09595v1):

- FLUX.1 is a **12 billion parameter rectified flow transformer**
- Uses **MM-DiT (Multi-Modal DiT) blocks** - double-stream (19 blocks) and single-stream (38 blocks)
- Each block uses **joint self-attention** over concatenated text + image tokens
- Uses **Rotary Position Embeddings (RoPE)** for positional encoding
- Very few convolutions (mostly in the VAE, not the main transformer)

### What DiT360 Actually Does

From the [DiT360 paper](https://arxiv.org/abs/2510.11712) and [project page](https://fenghora.github.io/DiT360-Page/):

> "At the token level, hybrid supervision is applied across multiple modules, which include **circular padding for boundary continuity**, yaw loss for rotational robustness, and cube loss for distortion awareness."

DiT360 applies circular padding at the **token/attention level** during training, not just at the input/output. This requires modifying how attention is computed.

---

## Working Solutions for ComfyUI Custom Nodes

### Solution 1: Patch Conv2d Layers to Circular Mode (Partial Fix)

**Status: WORKS for UNet models, PARTIAL for FLUX**

This is what [ComfyUI_Seamless_Patten](https://github.com/moyi7712/ComfyUI_Seamless_Patten) and [comfy_mtb](https://github.com/melMass/comfy_mtb) do:

```python
import torch.nn as nn

def set_circular_padding(model):
    """Set all Conv2d layers to circular padding mode"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Only apply to width (horizontal) dimension
            module.padding_mode = 'circular'
```

**Why it's partial for FLUX:**
- FLUX's main transformer has minimal Conv2d layers
- The VAE does have Conv2d layers - this will help VAE encode/decode
- Won't fix the attention-based core diffusion process

### Solution 2: Use ComfyUI's ModelPatcher Attention Patches

**Status: POSSIBLE but COMPLEX**

ComfyUI provides methods to patch attention:
- `set_model_attn1_patch(patch)` - Patch self-attention (attn1)
- `set_model_attn2_patch(patch)` - Patch cross-attention (attn2)
- `set_model_attn1_replace(patch, block_name, number)` - Replace specific attention

From [ComfyUI model_patcher.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py):

```python
def set_model_attn1_patch(self, patch):
    self.set_model_patch(patch, "attn1_patch")

def set_model_attn2_patch(self, patch):
    self.set_model_patch(patch, "attn2_patch")
```

**Implementation approach:**
```python
def circular_attention_patch(q, k, v, extra_options):
    """
    Custom attention that handles circular topology.

    The idea: when computing attention scores, add connections
    between leftmost and rightmost tokens.
    """
    # Get image token dimensions from extra_options
    # Modify attention to wrap around horizontally
    # This is complex because we need to know which tokens
    # correspond to which spatial positions
    pass
```

**Challenges:**
1. Need to understand FLUX's token layout (how 2D positions map to 1D token sequence)
2. Need to modify attention scores or use masking
3. May conflict with other nodes that patch attention

### Solution 3: Register Forward Hooks on Attention Modules

**Status: POSSIBLE, Similar to Solution 2**

```python
def register_circular_hooks(model):
    """Register hooks to modify attention behavior"""

    def attention_pre_hook(module, inputs):
        # Modify inputs before attention
        q, k, v = inputs
        # Wrap tokens circularly
        return (q, k, v)

    def attention_post_hook(module, inputs, output):
        # Modify output after attention
        return output

    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            module.register_forward_pre_hook(attention_pre_hook)
            module.register_forward_hook(attention_post_hook)
```

**The core algorithm for circular attention:**
```python
def make_attention_circular(attn_scores, width_tokens, wrap_amount=8):
    """
    Modify attention scores to create circular connectivity.

    For each token at x=0, add attention to tokens at x=width-1
    and vice versa, treating them as adjacent.
    """
    B, heads, N, N = attn_scores.shape

    # Create circular mask or modify scores
    # Tokens at position x should attend to:
    # - Normal positions
    # - Position (x + width) % width (the wrapped version)

    # This requires knowing the spatial layout of tokens
    pass
```

### Solution 4: VAE-Only Circular Padding (Easiest, Partial)

**Status: WORKS, but only fixes VAE**

Focus circular padding only on the VAE encode/decode since those have convolutions:

```python
class CircularVAE:
    """Wrap VAE to use circular padding in all Conv2d layers"""

    def __init__(self, vae):
        self.vae = vae
        self._set_circular_mode()

    def _set_circular_mode(self):
        """Set all Conv2d to circular padding"""
        for module in self.vae.modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.padding[1] > 0:  # Has horizontal padding
                    module.padding_mode = 'circular'

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, x):
        return self.vae.decode(x)
```

This helps with seam artifacts in the final decoded image but doesn't fix the core generation.

### Solution 5: Circular Tile/Blend During Sampling (Current Best Option)

**Status: WORKS as workaround**

Generate slightly overlapping tiles and blend them:

```python
def circular_sample(model, latent, overlap=64):
    """
    Sample with circular blending strategy.

    1. Pad input latent circularly
    2. Generate
    3. Blend the overlapping regions using weighted average
    4. Crop to original size
    """
    # Circular pad
    padded = apply_circular_padding(latent, overlap)

    # Sample normally
    output = sample(model, padded, ...)

    # Blend overlap regions
    left_region = output[:, :, :, :overlap]
    right_region = output[:, :, :, -overlap:]

    # Weighted blend
    weights = torch.linspace(0, 1, overlap)
    blended = left_region * (1 - weights) + right_region * weights

    # Place blended region
    output[:, :, :, :overlap] = blended

    # Crop
    return output[:, :, :, overlap:-overlap]
```

This is what the current implementation does, and while not perfect, it helps.

### Solution 6: Modify Position Embeddings (Complex)

**Status: THEORETICALLY BEST, but VERY COMPLEX**

The ideal solution would modify FLUX's Rotary Position Embeddings (RoPE) to encode circular topology:

```python
def circular_rope(positions, width):
    """
    Modified RoPE that encodes circular positions.

    Instead of linear positions [0, 1, 2, ..., N-1],
    use circular encoding where position 0 is adjacent to position N-1.
    """
    # Convert linear positions to circular
    theta = 2 * np.pi * positions / width

    # Encode as (cos, sin) which naturally wraps
    pos_embed = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    return pos_embed
```

This would require:
1. Finding where position embeddings are computed in FLUX
2. Replacing them with circular versions
3. Potentially fine-tuning to adjust for the new embeddings

---

## Recommended Implementation Strategy

### Phase 1: VAE Circular Padding (Easy Win)

```python
class Equirect360VAEDecode:
    def decode(self, samples, vae, circular_padding):
        # Temporarily set VAE Conv2d to circular mode
        original_modes = {}
        for name, module in vae.first_stage_model.named_modules():
            if isinstance(module, nn.Conv2d):
                original_modes[name] = module.padding_mode
                module.padding_mode = 'circular'

        try:
            # Decode with circular padding
            image = vae.decode(samples["samples"])
        finally:
            # Restore original modes
            for name, module in vae.first_stage_model.named_modules():
                if name in original_modes:
                    module.padding_mode = original_modes[name]

        return (image,)
```

### Phase 2: Attention Patch (Medium Complexity)

Use ComfyUI's `set_model_attn1_patch` to add circular attention awareness:

```python
def create_circular_attention_patch():
    def circular_attn_patch(q, k, v, extra_options):
        """Patch attention to handle circular topology"""
        # Implementation depends on FLUX's token layout
        # Need to identify image tokens and their 2D positions
        # Then modify attention to wrap horizontally
        pass
    return circular_attn_patch

class Equirect360KSampler:
    def sample(self, model, ...):
        model_patched = model.clone()
        model_patched.set_model_attn1_patch(create_circular_attention_patch())
        # ... rest of sampling
```

### Phase 3: Full Circular Transformer (Complex)

Implement proper circular position embeddings and attention masks. This may require forking FLUX components.

---

## Key Files to Modify

1. **`utils/padding.py`** - Update `create_circular_padding_wrapper` to use proper patching
2. **`nodes.py`** - Update `Equirect360VAEDecode` to patch Conv2d padding modes
3. **NEW: `utils/circular_attention.py`** - Implement attention patches

---

## References

### ComfyUI Patching
- [ComfyUI model_patcher.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py)
- [ComfyUI Discussion #408 - Patching models](https://github.com/comfyanonymous/ComfyUI/discussions/408)
- [ComfyUI Discussion #982 - Undoing patches](https://github.com/comfyanonymous/ComfyUI/discussions/982)

### Existing Seamless Implementations
- [ComfyUI_Seamless_Patten](https://github.com/moyi7712/ComfyUI_Seamless_Patten) - Conv2d circular padding
- [comfy_mtb Model Patch Seamless](https://github.com/melMass/comfy_mtb/wiki/nodes-model-patch-seamless)
- [ComfyUI-seamless-tiling](https://github.com/spinagon/ComfyUI-seamless-tiling)

### DiT360
- [DiT360 Paper](https://arxiv.org/abs/2510.11712)
- [DiT360 Project Page](https://fenghora.github.io/DiT360-Page/)
- [DiT360 GitHub](https://github.com/Insta360-Research-Team/DiT360)
- [DiT360 HuggingFace](https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation)

### FLUX Architecture
- [Demystifying Flux Architecture](https://arxiv.org/html/2507.09595v1)
- [FLUX.1-dev on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev)

---

## Conclusion

**Can circular padding work as a ComfyUI custom node without modifying core code? YES, but with limitations:**

| Approach | Works Without Core Changes | Effectiveness for FLUX |
|----------|---------------------------|----------------------|
| Conv2d padding_mode='circular' | YES | LOW (FLUX has few convs) |
| VAE circular padding | YES | MEDIUM (helps decode only) |
| Input/output latent padding | YES | LOW (doesn't fix attention) |
| Attention patches via ModelPatcher | YES | MEDIUM-HIGH (if implemented right) |
| Position embedding modification | MAYBE | HIGH (but very complex) |
| Edge blending post-process | YES | MEDIUM (hides seams, doesn't fix generation) |

**Recommended path forward:**
1. Implement VAE circular padding (Conv2d mode change)
2. Investigate FLUX's attention structure to design proper attention patches
3. Use edge blending as a final polish step
4. Document that full circular padding requires the DiT360 LoRA (which was trained with circular awareness)

The DiT360 LoRA itself was trained with circular padding in the attention mechanism, so using the LoRA with proper circular VAE decode and edge blending should give reasonable results even without full attention patching.
