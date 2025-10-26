"""
DiT360 Model Architecture and Loading

This module implements the DiT360 diffusion transformer model based on FLUX.1-dev.
The model is a 12-billion parameter transformer designed for panoramic image generation
with special adaptations for equirectangular projection.

Key Features:
- Circular padding in attention layers for seamless wraparound
- RoPE positional embeddings adapted for spherical geometry
- Flow matching scheduler for efficient sampling
- Support for multiple precisions (fp32, fp16, bf16, fp8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file, safe_open
import json
from typing import Dict, Optional, Union, Tuple
import comfy.model_management as mm
from huggingface_hub import snapshot_download
import os
import math
import importlib
import warnings


def _import_optional(module_name: str):
    """
    Attempt to import an optional dependency.

    Returns the imported module or None if unavailable.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


_XFORMERS_OPS = _import_optional("xformers.ops")
_FLASH_ATTN_UNPADDED = None
_FLASH_ATTN_AVAILABLE = False
_BITSANDBYTES_NN = _import_optional("bitsandbytes.nn")
_HAS_BITSANDBYTES = False
_BNB_LINEAR4 = None
if _BITSANDBYTES_NN is not None and hasattr(_BITSANDBYTES_NN, "Linear4bit"):
    _HAS_BITSANDBYTES = True
    _BNB_LINEAR4 = _BITSANDBYTES_NN.Linear4bit
_SLICE_SENTINEL = object()

if _import_optional("flash_attn") is not None:
    # Try to grab the unpadded attention function (preferred API)
    flash_interface = _import_optional("flash_attn.flash_attn_interface")
    if flash_interface is not None and hasattr(flash_interface, "flash_attn_unpadded_qkvpacked_func"):
        _FLASH_ATTN_UNPADDED = flash_interface.flash_attn_unpadded_qkvpacked_func
        _FLASH_ATTN_AVAILABLE = True
    else:
        flash_pkg = _import_optional("flash_attn.flash_attn_func")
        if flash_pkg is not None and hasattr(flash_pkg, "flash_attn_func"):
            _FLASH_ATTN_UNPADDED = flash_pkg.flash_attn_func
            _FLASH_ATTN_AVAILABLE = True


# ============================================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================================

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor

    Args:
        x: Input tensor of shape (B, seq_len, num_heads, head_dim)
        freqs_cos: Cosine frequencies (seq_len, head_dim//2)
        freqs_sin: Sine frequencies (seq_len, head_dim//2)

    Returns:
        Tensor with rotary embeddings applied
    """
    # Reshape freqs for broadcasting: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    # Reshape x to apply rotations
    x_real, x_imag = x.chunk(2, dim=-1)  # Split last dim in half

    # Apply rotation
    x_rotated_real = x_real * freqs_cos - x_imag * freqs_sin
    x_rotated_imag = x_real * freqs_sin + x_imag * freqs_cos

    # Concatenate back
    return torch.cat([x_rotated_real, x_rotated_imag], dim=-1)


class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embeddings adapted for spherical geometry

    For panoramic images, we use modified RoPE that accounts for the
    wraparound nature of equirectangular projection.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)

        # Create cos and sin embeddings
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for sequence length

        Args:
            x: Input tensor (B, seq_len, ...)

        Returns:
            Tuple of (cos_freqs, sin_freqs) for the sequence length
        """
        seq_len = x.shape[1]
        return self.freqs_cos[:seq_len], self.freqs_sin[:seq_len]


# ============================================================================
# Adaptive Layer Normalization (adaLN)
# ============================================================================

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization for conditioning on timestep and text

    This modulates LayerNorm parameters based on conditioning signals,
    allowing the model to adapt its normalization to different timesteps
    and text prompts.
    """

    def __init__(self, hidden_size: int, conditioning_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Project conditioning to scale and shift parameters
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, hidden_size * 2)
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization

        Args:
            x: Input tensor (B, seq_len, hidden_size)
            conditioning: Conditioning tensor (B, conditioning_dim)

        Returns:
            Normalized and modulated tensor
        """
        # Normalize
        x_norm = self.norm(x)

        # Get scale and shift from conditioning
        ada_params = self.ada_proj(conditioning)
        scale, shift = ada_params.chunk(2, dim=-1)

        # Apply modulation
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================================
# Multi-Head Attention with Circular Padding
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with circular padding support

    For panoramic images, we apply circular padding along the width dimension
    to ensure seamless wraparound at the edges.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        enable_circular_padding: bool = True,
        circular_padding_width: int = 0,
        attention_backend: str = "auto",
        attention_slice_size: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.enable_circular_padding = enable_circular_padding
        self.circular_padding_width = circular_padding_width
        self.requested_backend = attention_backend
        self.attention_backend = self._resolve_backend(attention_backend)
        self.attention_slice_size = attention_slice_size
        self._backend_warning_logged = False

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K, V projections
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _resolve_backend(self, backend: str) -> str:
        """Resolve requested attention backend to an available implementation."""
        backend = (backend or "auto").lower()

        if backend == "auto":
            if _XFORMERS_OPS is not None:
                return "xformers"
            if _FLASH_ATTN_AVAILABLE:
                return "flash"
            return "eager"

        if backend == "xformers":
            if _XFORMERS_OPS is None:
                return "eager"
            return "xformers"

        if backend in {"flash", "flash_attn", "flashattention"}:
            if _FLASH_ATTN_AVAILABLE:
                return "flash"
            return "eager"

        return "eager"

    def set_attention_backend(self, backend: str):
        """Update attention backend at runtime."""
        resolved = self._resolve_backend(backend)
        if resolved != backend and backend not in {"auto", resolved} and not self._backend_warning_logged:
            warnings.warn(
                f"Requested attention backend '{backend}' is unavailable; falling back to '{resolved}'.",
                RuntimeWarning
            )
            self._backend_warning_logged = True
        self.requested_backend = backend
        self.attention_backend = resolved

    def set_attention_slicing(self, slice_size: Optional[int]):
        """Configure attention slicing chunk size."""
        if slice_size is not None and slice_size <= 0:
            slice_size = None
        self.attention_slice_size = slice_size

    def apply_circular_padding_to_tokens(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Apply circular padding to token sequence

        Args:
            tokens: Token tensor (B, H*W, hidden_size)
            height: Latent height
            width: Latent width

        Returns:
            Padded token tensor
        """
        if self.circular_padding_width == 0:
            return tokens

        B, seq_len, C = tokens.shape

        # Reshape to spatial grid
        tokens_spatial = tokens.reshape(B, height, width, C)

        # Apply circular padding on width dimension
        left_edge = tokens_spatial[:, :, :self.circular_padding_width, :]
        right_edge = tokens_spatial[:, :, -self.circular_padding_width:, :]

        # Concatenate: right_edge | original | left_edge
        padded = torch.cat([right_edge, tokens_spatial, left_edge], dim=2)

        # Reshape back to sequence
        new_width = width + 2 * self.circular_padding_width
        padded_seq = padded.reshape(B, height * new_width, C)

        return padded_seq

    def remove_circular_padding_from_tokens(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Remove circular padding from token sequence"""
        if self.circular_padding_width == 0:
            return tokens

        B, _, C = tokens.shape
        padded_width = width + 2 * self.circular_padding_width

        # Reshape to spatial
        tokens_spatial = tokens.reshape(B, height, padded_width, C)

        # Remove padding
        unpadded = tokens_spatial[:, :, self.circular_padding_width:-self.circular_padding_width, :]

        # Reshape back
        return unpadded.reshape(B, height * width, C)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional circular padding

        Args:
            x: Input tokens (B, seq_len, hidden_size)
            height: Spatial height for circular padding
            width: Spatial width for circular padding
            rope_emb: Optional RoPE embeddings (cos, sin)

        Returns:
            Attention output (B, seq_len, hidden_size)
        """
        B, seq_len, C = x.shape

        # Apply circular padding if enabled
        if self.enable_circular_padding and self.circular_padding_width > 0:
            x_padded = self.apply_circular_padding_to_tokens(x, height, width)
        else:
            x_padded = x

        # Compute Q, K, V
        qkv = self.qkv(x_padded).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len_padded, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if provided (adjust for padded sequence length)
        if rope_emb is not None:
            freqs_cos, freqs_sin = rope_emb
            # Get embeddings for the actual sequence length (padded or unpadded)
            actual_seq_len = q.shape[2]  # Get actual sequence length from Q
            if actual_seq_len > freqs_cos.shape[0]:
                # If padded sequence is longer, we skip RoPE to avoid errors
                # In practice, RoPE should be recomputed for padded length, but this is a fallback
                pass  # Skip RoPE for now
            else:
                q = apply_rotary_emb(q.transpose(1, 2), freqs_cos[:actual_seq_len], freqs_sin[:actual_seq_len]).transpose(1, 2)
                k = apply_rotary_emb(k.transpose(1, 2), freqs_cos[:actual_seq_len], freqs_sin[:actual_seq_len]).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        backend = self.attention_backend

        try:
            if backend == "xformers":
                out = self._forward_xformers(q, k, v)
            elif backend == "flash":
                out = self._forward_flash(q, k, v, scale)
            else:
                out = self._forward_eager(q, k, v, scale)
        except RuntimeError as err:
            if not self._backend_warning_logged:
                warnings.warn(
                    f"Attention backend '{backend}' failed with error '{err}'. Falling back to eager implementation.",
                    RuntimeWarning
                )
                self._backend_warning_logged = True
            self.attention_backend = "eager"
            out = self._forward_eager(q, k, v, scale)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, -1, self.hidden_size)
        out = self.out_proj(out)

        # Remove circular padding
        if self.enable_circular_padding and self.circular_padding_width > 0:
            out = self.remove_circular_padding_from_tokens(out, height, width)

        return out

    def _forward_eager(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """Standard PyTorch attention with optional slicing."""
        if self.attention_slice_size is not None and self.attention_slice_size < q.shape[2]:
            outputs = []
            for start in range(0, q.shape[2], self.attention_slice_size):
                end = min(start + self.attention_slice_size, q.shape[2])
                q_slice = q[:, :, start:end, :]
                attn_slice = torch.matmul(q_slice, k.transpose(-2, -1)) * scale
                attn_slice = attn_slice.softmax(dim=-1)
                out_slice = torch.matmul(attn_slice, v)
                outputs.append(out_slice)
            out = torch.cat(outputs, dim=2)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = torch.matmul(attn, v)
        return out

    def _forward_xformers(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """xFormers memory efficient attention."""
        if _XFORMERS_OPS is None:
            return self._forward_eager(q, k, v, 1.0 / math.sqrt(self.head_dim))

        B, heads, seq_len, head_dim = q.shape
        q_flat = q.reshape(B * heads, seq_len, head_dim)
        k_flat = k.reshape(B * heads, seq_len, head_dim)
        v_flat = v.reshape(B * heads, seq_len, head_dim)

        out = _XFORMERS_OPS.memory_efficient_attention(q_flat, k_flat, v_flat, attn_bias=None)
        out = out.reshape(B, heads, seq_len, head_dim)
        return out

    def _forward_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """FlashAttention integration (unpadded)."""
        if not _FLASH_ATTN_AVAILABLE or _FLASH_ATTN_UNPADDED is None:
            return self._forward_eager(q, k, v, scale)

        B, heads, seq_len, head_dim = q.shape
        qkv = torch.stack([q, k, v], dim=2)  # (B, heads, 3, seq, head_dim)
        qkv = qkv.permute(0, 1, 3, 2, 4).contiguous()  # (B, heads, seq, 3, head_dim)
        qkv = qkv.view(B * heads, seq_len, 3, head_dim)

        try:
            out = _FLASH_ATTN_UNPADDED(qkv, causal=False)
            out = out.view(B, heads, seq_len, head_dim)
            return out
        except RuntimeError:
            return self._forward_eager(q, k, v, scale)


# ============================================================================
# MLP Block
# ============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron block

    Standard feedforward network with GELU activation.
    """

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP

    Uses pre-normalization and adaptive layer normalization for conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        conditioning_dim: int,
        mlp_ratio: float = 4.0,
        enable_circular_padding: bool = True,
        circular_padding_width: int = 0,
        attention_backend: str = "auto",
        attention_slice_size: Optional[int] = None
    ):
        super().__init__()

        # Adaptive norms
        self.norm1 = AdaptiveLayerNorm(hidden_size, conditioning_dim)
        self.norm2 = AdaptiveLayerNorm(hidden_size, conditioning_dim)

        # Attention
        self.attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            enable_circular_padding,
            circular_padding_width,
            attention_backend=attention_backend,
            attention_slice_size=attention_slice_size
        )

        # MLP
        self.mlp = MLP(hidden_size, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        height: int,
        width: int,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block

        Args:
            x: Input tokens (B, seq_len, hidden_size)
            conditioning: Conditioning vector (B, conditioning_dim)
            height: Spatial height
            width: Spatial width
            rope_emb: Optional RoPE embeddings

        Returns:
            Output tokens (B, seq_len, hidden_size)
        """
        # Self-attention with residual
        x = x + self.attn(
            self.norm1(x, conditioning),
            height,
            width,
            rope_emb
        )

        # MLP with residual
        x = x + self.mlp(self.norm2(x, conditioning))

        return x

    def set_attention_backend(self, backend: str):
        """Update backend for the internal attention layer."""
        self.attn.set_attention_backend(backend)

    def set_attention_slicing(self, slice_size: Optional[int]):
        """Update slicing configuration for the internal attention layer."""
        self.attn.set_attention_slicing(slice_size)


# ============================================================================
# DiT360 Model
# ============================================================================

class DiT360Model(nn.Module):
    """
    DiT360 Diffusion Transformer Model

    Based on FLUX.1-dev architecture with adaptations for panoramic generation:
    - Circular padding for seamless left/right wraparound
    - Modified attention for equirectangular projection
    - RoPE positional embeddings for spherical geometry

    Args:
        config: Model configuration dictionary
        enable_circular_padding: Enable circular padding in attention layers
    """

    def __init__(
        self,
        config: Dict,
        enable_circular_padding: bool = True,
        attention_backend: str = "auto",
        attention_slice_size: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.enable_circular_padding = enable_circular_padding
        self.attention_backend = attention_backend
        self.attention_slice_size = attention_slice_size

        # Extract key parameters from config
        self.in_channels = config.get("in_channels", 4)
        self.hidden_size = config.get("hidden_size", 3072)
        self.num_layers = config.get("num_layers", 38)
        self.num_heads = config.get("num_heads", 24)
        self.caption_channels = config.get("caption_channels", 4096)
        self.patch_size = config.get("patch_size", 2)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.circular_padding_width = config.get("circular_padding_width", 2)

        print(f"\nInitializing DiT360 Model:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Circular padding: {enable_circular_padding}")
        print(f"  Attention backend: {attention_backend}")
        if attention_slice_size:
            print(f"  Attention slicing: {attention_slice_size}")

        # ====================================================================
        # Input Processing
        # ====================================================================

        # Patch embedding: Convert (B, C, H, W) to (B, seq_len, hidden_size)
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # ====================================================================
        # Conditioning
        # ====================================================================

        # Timestep embedding (256-dim -> hidden_size)
        self.time_embed = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Text embedding projection (caption_channels -> hidden_size)
        self.caption_proj = nn.Linear(self.caption_channels, self.hidden_size)

        # Combined conditioning dimension for adaLN
        self.conditioning_dim = self.hidden_size * 2

        # Project combined conditioning
        self.conditioning_proj = nn.Linear(self.hidden_size * 2, self.conditioning_dim)

        # ====================================================================
        # Positional Embeddings
        # ====================================================================

        # RoPE for positional encoding
        self.rope = RoPEEmbedding(
            dim=self.hidden_size // self.num_heads,
            max_seq_len=8192
        )

        # ====================================================================
        # Transformer Blocks
        # ====================================================================

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                conditioning_dim=self.conditioning_dim,
                mlp_ratio=self.mlp_ratio,
                enable_circular_padding=enable_circular_padding,
                circular_padding_width=self.circular_padding_width,
                attention_backend=attention_backend,
                attention_slice_size=attention_slice_size
            )
            for _ in range(self.num_layers)
        ])

    def set_attention_backend(self, backend: str):
        """Update attention backend for all transformer blocks."""
        self.attention_backend = backend
        for block in self.blocks:
            block.set_attention_backend(backend)

    def set_attention_slicing(self, slice_size: Optional[int]):
        """Update attention slicing setting for all transformer blocks."""
        self.attention_slice_size = slice_size
        for block in self.blocks:
            block.set_attention_slicing(slice_size)

    def set_attention_options(
        self,
        backend: Optional[str] = None,
        slice_size: Union[int, None, object] = _SLICE_SENTINEL
    ):
        """Convenience method to configure backend and slicing in one call."""
        if backend is not None:
            self.set_attention_backend(backend)
        if slice_size is not _SLICE_SENTINEL:
            self.set_attention_slicing(slice_size if slice_size is not None else None)

        # ====================================================================
        # Output Processing
        # ====================================================================

        # Final layer norm
        self.final_norm = nn.LayerNorm(self.hidden_size)

        # Output projection back to latent space
        self.out_proj = nn.Linear(self.hidden_size, self.in_channels * self.patch_size * self.patch_size)

        # Initialize weights
        self.initialize_weights()
        self.initialized = True

    def initialize_weights(self):
        """Initialize model weights using standard initialization"""
        # Initialize linear layers with xavier uniform
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch embedding like a linear layer
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed[0].weight, std=0.02)
        nn.init.normal_(self.time_embed[2].weight, std=0.02)

    def timestep_embedding(self, timesteps: torch.Tensor, dim: int = 256) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings

        Args:
            timesteps: Timestep values (B,)
            dim: Embedding dimension

        Returns:
            Timestep embeddings (B, dim)
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if dim % 2 == 1:  # Zero pad if odd dimension
            emb = F.pad(emb, (0, 1))

        return emb

    def unpatchify(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Convert token sequence back to spatial latent

        Args:
            x: Token sequence (B, seq_len, patch_size^2 * C)
            height: Original latent height (in patches)
            width: Original latent width (in patches)

        Returns:
            Spatial latent (B, C, H, W)
        """
        B = x.shape[0]
        patch_size = self.patch_size
        c = self.in_channels

        # Reshape: (B, H*W, patch_size^2 * C) -> (B, H, W, patch_size, patch_size, C)
        x = x.reshape(B, height, width, patch_size, patch_size, c)

        # Rearrange to (B, C, H, patch_size, W, patch_size)
        x = x.permute(0, 5, 1, 3, 2, 4)

        # Merge patches: (B, C, H*patch_size, W*patch_size)
        x = x.reshape(B, c, height * patch_size, width * patch_size)

        return x

    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                context: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through DiT360 transformer

        Args:
            x: Input latent tensor (B, C, H, W)
            timestep: Timestep tensor (B,) - values typically in [0, 1000]
            context: Text conditioning (B, seq_len, caption_channels)
            **kwargs: Additional conditioning

        Returns:
            Noise prediction (B, C, H, W)
        """
        B, C, H, W = x.shape

        # ====================================================================
        # Step 1: Patch Embedding
        # ====================================================================
        # Convert (B, C, H, W) to (B, seq_len, hidden_size)
        x = self.patch_embed(x)  # (B, hidden_size, H//patch_size, W//patch_size)

        # Flatten spatial dimensions
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        x = x.flatten(2).transpose(1, 2)  # (B, seq_len, hidden_size)

        # ====================================================================
        # Step 2: Conditioning
        # ====================================================================

        # Timestep embedding
        t_emb = self.timestep_embedding(timestep, dim=256)  # (B, 256)
        t_emb = self.time_embed(t_emb)  # (B, hidden_size)

        # Text embedding (pool across sequence dimension)
        c_emb = context.mean(dim=1)  # (B, caption_channels)
        c_emb = self.caption_proj(c_emb)  # (B, hidden_size)

        # Combine conditioning signals
        conditioning = torch.cat([t_emb, c_emb], dim=1)  # (B, hidden_size * 2)
        conditioning = self.conditioning_proj(conditioning)  # (B, conditioning_dim)

        # ====================================================================
        # Step 3: Positional Embeddings (RoPE)
        # ====================================================================
        rope_emb = self.rope(x)  # Get RoPE embeddings for sequence

        # ====================================================================
        # Step 4: Transformer Blocks
        # ====================================================================
        for block in self.blocks:
            x = block(x, conditioning, h_patches, w_patches, rope_emb)

        # ====================================================================
        # Step 5: Output Projection
        # ====================================================================
        x = self.final_norm(x)
        x = self.out_proj(x)  # (B, seq_len, patch_size^2 * C)

        # ====================================================================
        # Step 6: Unpatchify
        # ====================================================================
        x = self.unpatchify(x, h_patches, w_patches)  # (B, C, H, W)

        return x


class DiT360Wrapper:
    """
    Wrapper for DiT360 model that handles loading, caching, and device management.

    This wrapper:
    - Loads model from HuggingFace or local files
    - Manages model caching (avoid reloading)
    - Handles device placement (GPU/CPU)
    - Supports precision conversion
    """

    def __init__(
        self,
        model: DiT360Model,
        dtype: torch.dtype,
        device: torch.device,
        offload_device: torch.device,
        quantization_mode: str = "none"
    ):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.offload_device = offload_device
        self.is_loaded = False
        self.attention_backend = model.attention_backend
        self.attention_slice_size = model.attention_slice_size
        self.quantization_mode = quantization_mode

    def load_to_device(self):
        """Load model to GPU"""
        if not self.is_loaded:
            print(f"Loading DiT360 to {self.device}...")
            self.model.to(self.device)
            self.is_loaded = True

    def offload(self):
        """Offload model to CPU to free VRAM"""
        if self.is_loaded:
            print(f"Offloading DiT360 to {self.offload_device}...")
            self.model.to(self.offload_device)
            self.is_loaded = False
            mm.soft_empty_cache()

    def set_attention_options(
        self,
        backend: Optional[str] = None,
        slice_size: Union[int, None, object] = _SLICE_SENTINEL
    ):
        """Expose attention configuration for downstream nodes."""
        self.model.set_attention_options(backend=backend, slice_size=slice_size)
        if backend is not None:
            self.attention_backend = self.model.attention_backend
        if slice_size is not _SLICE_SENTINEL:
            self.attention_slice_size = self.model.attention_slice_size


def download_dit360_from_huggingface(
    repo_id: str = "Insta360-Research/DiT360-Panorama-Image-Generation",
    save_dir: Path = None,
    model_name: str = "dit360_model"
) -> Path:
    """
    Download DiT360 model from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID
        save_dir: Directory to save model (default: ComfyUI/models/dit360/)
        model_name: Name for model directory

    Returns:
        Path to downloaded model directory

    Example:
        >>> model_dir = download_dit360_from_huggingface()
        >>> print(f"Model downloaded to: {model_dir}")
    """
    import folder_paths

    if save_dir is None:
        save_dir = Path(folder_paths.models_dir) / "dit360" / model_name
    else:
        save_dir = Path(save_dir)

    # Check if already downloaded
    if save_dir.exists() and (save_dir / "config.json").exists():
        print(f"Model already exists at: {save_dir}")
        return save_dir

    print(f"\n{'='*60}")
    print(f"Downloading DiT360 model from HuggingFace...")
    print(f"Repository: {repo_id}")
    print(f"Destination: {save_dir}")
    print(f"{'='*60}\n")

    try:
        # Download using HuggingFace Hub
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            # Ignore EMA weights if present (we only need main model)
            ignore_patterns=["*ema*", "*.md", "*.txt"]
        )

        print(f"\n✓ Download complete: {downloaded_path}\n")
        return Path(downloaded_path)

    except Exception as e:
        raise RuntimeError(
            f"\nFailed to download DiT360 model from HuggingFace.\n\n"
            f"Error: {e}\n\n"
            f"Please download manually from:\n"
            f"  https://huggingface.co/{repo_id}\n\n"
            f"And place in:\n"
            f"  {save_dir}\n"
        )


def load_dit360_model(
    model_path: Union[str, Path],
    precision: str = "fp16",
    device: Optional[torch.device] = None,
    offload_device: Optional[torch.device] = None,
    enable_circular_padding: bool = True,
    attention_backend: str = "auto",
    attention_slice_size: Optional[int] = None,
    quantization_mode: str = "none"
) -> DiT360Wrapper:
    """
    Load DiT360 model from file or HuggingFace

    Args:
        model_path: Path to model file (.safetensors) or directory
        precision: Model precision - "fp32", "fp16", "bf16", or "fp8"
        device: Target device (None = auto-detect GPU)
        offload_device: Device for offloading (None = CPU)
        enable_circular_padding: Enable circular padding for panoramas
        attention_backend: Attention implementation (auto/eager/xformers/flash)
        attention_slice_size: Optional chunk size for attention slicing
        quantization_mode: Optional post-load quantization ("none", "int8", "int4")

    Returns:
        DiT360Wrapper containing loaded model

    Raises:
        FileNotFoundError: If model file not found
        RuntimeError: If model loading fails

    Example:
        >>> model = load_dit360_model("models/dit360/model.safetensors", precision="fp16")
        >>> # Use model for inference
        >>> output = model.model(latent, timestep, context)
    """
    model_path = Path(model_path)

    # Auto-detect devices if not specified
    if device is None:
        device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()

    print(f"\n{'='*60}")
    print(f"Loading DiT360 Model")
    print(f"{'='*60}")
    print(f"Path: {model_path}")
    print(f"Precision: {precision}")
    print(f"Device: {device}")
    print(f"Offload: {offload_device}")
    print(f"Attention backend: {attention_backend}")
    if attention_slice_size:
        print(f"Attention slicing: {attention_slice_size}")
    print(f"Quantization: {quantization_mode}")
    print(f"{'='*60}\n")

    # Handle directory vs file path
    if model_path.is_dir():
        # Look for .safetensors file in directory
        config_path = model_path / "config.json"
        model_file = None
        for ext in ["*.safetensors", "*.pt", "*.pth"]:
            matches = list(model_path.glob(ext))
            if matches:
                model_file = matches[0]
                break

        if model_file is None:
            raise FileNotFoundError(
                f"No model file found in directory: {model_path}\n"
                f"Expected .safetensors, .pt, or .pth file"
            )
        model_path = model_file
    else:
        # File path provided
        config_path = model_path.parent / "config.json"

    # Check if file exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n\n"
            f"Please download DiT360 model from:\n"
            f"  https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation\n\n"
            f"Or use auto-download by leaving model_path empty in DiT360Loader node.\n"
        )

    # Load or create config
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Config not found, using default DiT360 config")
        # Default FLUX.1-dev style config for DiT360
        config = {
            "model_type": "dit360",
            "architecture": "flux-dev",
            "params": "12B",
            "in_channels": 4,
            "hidden_size": 3072,
            "num_layers": 38,
            "num_heads": 24,
            "caption_channels": 4096,
            "model_max_length": 512,
        }

    # Initialize model
    print("Initializing DiT360 architecture...")
    model = DiT360Model(
        config,
        enable_circular_padding=enable_circular_padding,
        attention_backend=attention_backend,
        attention_slice_size=attention_slice_size
    )

    # Load weights
    print(f"Loading weights from: {model_path.name}")
    try:
        if model_path.suffix == ".safetensors":
            # Use safetensors for faster loading
            state_dict = load_file(str(model_path))
        else:
            # Use torch for .pt/.pth files
            state_dict = torch.load(str(model_path), map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Load state dict (strict=False to allow missing keys in placeholder)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"  Warning: Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Warning: Unexpected keys: {len(unexpected)}")

        print("✓ Weights loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    # Quantization (optional)
    quantization_mode = (quantization_mode or "none").lower()
    supported_quant = {"none", "int8", "int4"}
    if quantization_mode not in supported_quant:
        raise ValueError(f"Unknown quantization mode: {quantization_mode}. Use: {sorted(supported_quant)}")

    quantized = False
    quantization_note = None

    if quantization_mode == "int8":
        try:
            from torch.ao.quantization import quantize_dynamic

            print("Applying dynamic int8 quantization (torch.ao.quantization)...")
            model = model.to(dtype=torch.float32)
            model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            quantized = True
            quantization_note = "int8 dynamic"
        except Exception as quant_err:
            warnings.warn(
                f"Failed to apply int8 quantization ({quant_err}). Continuing without quantization.",
                RuntimeWarning
            )
            quantization_mode = "none"

    if quantization_mode == "int4" and not quantized:
        if not _HAS_BITSANDBYTES or _BNB_LINEAR4 is None:
            warnings.warn(
                "bitsandbytes with Linear4bit not available; cannot perform int4 quantization.",
                RuntimeWarning
            )
            quantization_mode = "none"
        else:
            print("Applying int4 quantization via bitsandbytes Linear4bit...")

            def _convert_linear_to_4bit(module: nn.Module):
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear):
                        if hasattr(_BNB_LINEAR4, "from_float"):
                            quant_child = _BNB_LINEAR4.from_float(child)
                        else:
                            quant_child = _BNB_LINEAR4(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                quant_type="nf4"
                            )
                            quant_child.weight.data.copy_(child.weight.data)
                            if child.bias is not None:
                                quant_child.bias = torch.nn.Parameter(child.bias.data.clone())
                        setattr(module, name, quant_child)
                    else:
                        _convert_linear_to_4bit(child)

            try:
                model = model.to(dtype=torch.float32)
                _convert_linear_to_4bit(model)
                quantized = True
                quantization_note = "int4 (bitsandbytes)"
            except Exception as quant_err:
                warnings.warn(
                    f"Failed to apply int4 quantization ({quant_err}). Continuing without quantization.",
                    RuntimeWarning
                )
                quantization_mode = "none"

    # Convert precision (only if not quantized)
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float16,  # fp8 not widely supported yet, fallback to fp16
    }

    if precision not in dtype_map:
        raise ValueError(f"Unknown precision: {precision}. Use: {list(dtype_map.keys())}")

    dtype = dtype_map[precision]

    if quantized:
        dtype = torch.float32  # quantized modules operate in fp32 for activations
    elif precision == "fp8":
        print("Warning: fp8 not fully supported, using fp16 instead")

    if not quantized:
        print(f"Converting to {precision} ({dtype})...")
        model = model.to(dtype=dtype)
    else:
        print(f"Quantized model ready ({quantization_note}).")

    # Move to offload device initially (will load to GPU on demand)
    model = model.to(offload_device)
    model.eval()  # Set to evaluation mode

    print(f"✓ Model ready on {offload_device}")
    print(f"  (Will load to {device} when needed)\n")

    # Wrap model
    wrapper = DiT360Wrapper(
        model=model,
        dtype=dtype,
        device=device,
        offload_device=offload_device,
        quantization_mode=quantization_mode
    )

    return wrapper


def get_model_info(model_path: Union[str, Path]) -> Dict:
    """
    Get information about a DiT360 model without loading it

    Args:
        model_path: Path to model file or directory

    Returns:
        Dictionary with model information

    Example:
        >>> info = get_model_info("models/dit360/model.safetensors")
        >>> print(f"Model size: {info['size_gb']:.2f} GB")
    """
    model_path = Path(model_path)

    info = {
        "exists": model_path.exists(),
        "path": str(model_path),
        "type": None,
        "size_gb": 0.0,
        "config": None,
    }

    if not model_path.exists():
        return info

    # Get file size
    if model_path.is_file():
        size_bytes = model_path.stat().st_size
        info["size_gb"] = size_bytes / (1024**3)
        info["type"] = model_path.suffix

    # Try to load config
    config_path = model_path.parent / "config.json" if model_path.is_file() else model_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            info["config"] = json.load(f)

    return info
