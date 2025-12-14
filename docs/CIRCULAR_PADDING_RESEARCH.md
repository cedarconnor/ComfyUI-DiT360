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

---

## Analysis: ChatGPT vs Gemini Approaches

Both AI assistants provided valid approaches. Here's how they compare:

### ChatGPT's Approach: Module Iteration + Conv2d Wrapping

```python
# ChatGPT's recommended pattern
def patch_model_convs(model, tiling_mode="x_only"):
    for name, module in model.model.diffusion_model.modules():
        if isinstance(module, nn.Conv2d):
            # Wrap with circular padding
            wrapper = CircularConvWrapper(module)
            # Replace in-place or return patched model
```

**Pros:**
- Simple to implement
- Works well for UNet architectures (SD 1.5, SDXL)
- Catches ALL Conv2d layers

**Cons for FLUX/DiT:**
- FLUX has very few Conv2d layers (only PatchEmbed + VAE)
- 95%+ of computation is in Linear layers and Attention
- Won't fix the core topology issue

### Gemini's Approach: `add_object_patch` API

```python
# Gemini's recommended pattern using ComfyUI API
def apply_circular(model, tiling_mode, patch_embed_only):
    model_clone = model.clone()

    for name, module in search_target.named_modules():
        if isinstance(module, nn.Conv2d):
            if "patch_embed" in name:  # Focus on critical layer
                patch_key = f"diffusion_model.{name}"
                wrapper = CircularWrapper(module, tiling_mode)
                model_clone.add_object_patch(patch_key, wrapper)

    return (model_clone,)
```

**Pros:**
- Uses official ComfyUI API
- Creates isolated clone (doesn't corrupt global state)
- Focuses on the critical `PatchEmbed` layer

**Cons:**
- `add_object_patch` behavior depends on ComfyUI version
- May not work for all model architectures
- Still doesn't address attention/RoPE

### What Both Miss: The RoPE Problem

Neither fully addresses that FLUX uses **Rotary Position Embeddings (RoPE)**:

```python
# The problem:
# Token at x=0 gets RoPE for position 0
# Token at x=255 gets RoPE for position 255
# Even with circular padding, attention "knows" they're far apart

# RoPE formula:
# q_rotated = q * cos(θ * position) + rotate(q) * sin(θ * position)
# k_rotated = k * cos(θ * position) + rotate(k) * sin(θ * position)

# For position 0 and 255 to be "adjacent", you'd need:
# position 0 → θ = 0
# position 255 → θ ≈ 2π (wraps back to ≈ 0)
```

**True circular RoPE would require:**
```python
def circular_rope(positions, width, dim):
    """RoPE that treats position 0 and width-1 as adjacent"""
    # Map linear position to angle on circle
    theta = 2 * np.pi * positions / width  # Now 0 and width-1 are close!

    # Standard RoPE with circular positions
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    angles = theta.unsqueeze(-1) * freqs.unsqueeze(0)

    return torch.cos(angles), torch.sin(angles)
```

### Synthesis: Recommended Implementation Order

| Priority | Approach | Effectiveness | Complexity | Source |
|----------|----------|---------------|------------|--------|
| 1 | VAE Conv2d circular | HIGH for decode | LOW | Both |
| 2 | PatchEmbed circular | MEDIUM | LOW | Gemini |
| 3 | All Conv2d circular | LOW for DiT | LOW | ChatGPT |
| 4 | Edge blending | MEDIUM (cosmetic) | LOW | Original |
| 5 | Attention patches | HIGH | HIGH | Neither fully |
| 6 | Circular RoPE | HIGHEST | VERY HIGH | Neither |

### Practical Code: Combining Best of Both

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularConv2d(nn.Module):
    """Wrapper that applies circular padding in X, zeros in Y"""

    def __init__(self, original_conv):
        super().__init__()
        self.conv = original_conv
        self.padding = original_conv.padding

    def forward(self, x):
        pad_h, pad_w = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)

        if pad_w > 0:
            # Circular pad width (horizontal wraparound)
            left = x[..., -pad_w:]
            right = x[..., :pad_w]
            x = torch.cat([left, x, right], dim=-1)

        if pad_h > 0:
            # Zero pad height (top/bottom)
            x = F.pad(x, (0, 0, pad_h, pad_h), mode='constant', value=0)

        # Run conv with padding=0 (we already padded)
        return F.conv2d(
            x, self.conv.weight, self.conv.bias,
            self.conv.stride, padding=0,
            self.conv.dilation, self.conv.groups
        )


class ApplyCircularPadding:
    """ComfyUI node that patches Conv2d layers for panorama generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scope": (["vae_only", "patch_embed_only", "all_conv2d"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "DiT360"

    def apply(self, model, scope):
        # CRITICAL: Clone to avoid corrupting global state
        model_clone = model.clone()

        # Get the underlying PyTorch model
        base = model_clone.model
        targets = []

        if scope == "vae_only":
            # VAE has most Conv2d layers
            if hasattr(base, 'first_stage_model'):
                targets.append(('vae', base.first_stage_model))
        elif scope == "patch_embed_only":
            # Just the PatchEmbed layer (most critical for DiT)
            if hasattr(base, 'diffusion_model'):
                for name, mod in base.diffusion_model.named_modules():
                    if 'patch_embed' in name.lower() or 'x_embedder' in name.lower():
                        targets.append((name, mod))
        else:
            # All Conv2d layers
            if hasattr(base, 'diffusion_model'):
                targets.append(('diffusion_model', base.diffusion_model))
            if hasattr(base, 'first_stage_model'):
                targets.append(('vae', base.first_stage_model))

        # Patch the layers
        patched = 0
        for prefix, target in targets:
            for name, module in target.named_modules():
                if isinstance(module, nn.Conv2d) and module.padding[0] > 0:
                    # Store original and set circular mode
                    # Note: Direct padding_mode change may not work for all ops
                    # Using wrapper is more reliable
                    try:
                        module.padding_mode = 'circular'
                        patched += 1
                    except:
                        pass  # Some modules don't support this

        print(f"[DiT360] Patched {patched} Conv2d layers to circular mode")
        return (model_clone,)
```

### Bottom Line

**Both ChatGPT and Gemini are correct that you CAN do this without core mods.** The key insights:

1. **ChatGPT's insight**: Iterate modules, wrap Conv2d - works for UNet
2. **Gemini's insight**: Use `add_object_patch`, focus on PatchEmbed - more targeted for DiT
3. **What both miss**: For FLUX/DiT, the attention mechanism and RoPE are the bigger issues
4. **Reality**: The DiT360 LoRA was trained WITH circular awareness baked in, so it compensates for some of the inference-time limitations
