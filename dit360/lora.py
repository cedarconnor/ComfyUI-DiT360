"""
LoRA (Low-Rank Adaptation) support for DiT360.

This module provides functionality for loading and merging LoRA weights into DiT360 models,
enabling fine-tuning and style customization without retraining the entire model.

Author: DiT360 Team
License: Apache 2.0
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from safetensors.torch import load_file
import re


class LoRALayer:
    """
    Represents a single LoRA layer with down and up projection matrices.

    LoRA adds trainable low-rank matrices to frozen pretrained weights:
    W' = W + α/r * (B @ A)

    Where:
    - W: Original frozen weight matrix
    - A: Down projection (rank reduction)
    - B: Up projection (rank expansion)
    - r: Rank of the adaptation
    - α: Scaling factor

    Args:
        down: Down projection matrix (out_features, rank)
        up: Up projection matrix (rank, in_features)
        alpha: LoRA scaling factor
        rank: Rank of the low-rank matrices

    Example:
        >>> down = torch.randn(768, 8)
        >>> up = torch.randn(8, 768)
        >>> lora = LoRALayer(down, up, alpha=8.0, rank=8)
    """

    def __init__(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        alpha: float = 1.0,
        rank: Optional[int] = None
    ):
        self.down = down
        self.up = up
        self.alpha = alpha
        self.rank = rank if rank is not None else down.shape[1]

        # Compute scale factor: α/r
        self.scale = alpha / self.rank

    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the weight delta to add to the original weight.

        Returns:
            Delta weight matrix: scale * (up @ down)
        """
        return self.scale * (self.up @ self.down)

    def to(self, device: torch.device, dtype: torch.dtype = None):
        """Move LoRA matrices to device and dtype."""
        self.down = self.down.to(device=device, dtype=dtype)
        self.up = self.up.to(device=device, dtype=dtype)
        return self


class LoRACollection:
    """
    Collection of LoRA layers for a model.

    This class manages multiple LoRA layers and provides methods for merging
    them into a model at different strengths.

    Args:
        lora_layers: Dictionary mapping layer names to LoRALayer objects
        name: Optional name for this LoRA collection

    Example:
        >>> layers = {
        ...     "blocks.0.attn.qkv": LoRALayer(down, up, alpha=8.0),
        ...     "blocks.1.attn.qkv": LoRALayer(down, up, alpha=8.0)
        ... }
        >>> lora = LoRACollection(layers, name="anime_style")
    """

    def __init__(
        self,
        lora_layers: Dict[str, LoRALayer],
        name: Optional[str] = None
    ):
        self.lora_layers = lora_layers
        self.name = name or "untitled"

    def get_layer_names(self) -> List[str]:
        """Get list of all layer names in this collection."""
        return list(self.lora_layers.keys())

    def to(self, device: torch.device, dtype: torch.dtype = None):
        """Move all LoRA layers to device and dtype."""
        for layer in self.lora_layers.values():
            layer.to(device=device, dtype=dtype)
        return self

    def __len__(self) -> int:
        """Return number of LoRA layers."""
        return len(self.lora_layers)

    def __repr__(self) -> str:
        return f"LoRACollection(name='{self.name}', layers={len(self)})"


def load_lora_from_safetensors(
    lora_path: Union[str, Path],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> LoRACollection:
    """
    Load LoRA weights from a safetensors file.

    Supports standard LoRA naming conventions:
    - "lora_down" and "lora_up" keys
    - "lora.down" and "lora.up" keys
    - "lora_A" and "lora_B" keys

    Args:
        lora_path: Path to LoRA .safetensors file
        device: Target device (default: CPU)
        dtype: Target dtype (default: float32)

    Returns:
        LoRACollection with loaded layers

    Example:
        >>> lora = load_lora_from_safetensors("style.safetensors")
        >>> print(f"Loaded {len(lora)} LoRA layers")

    Raises:
        FileNotFoundError: If LoRA file doesn't exist
        ValueError: If file format is invalid
    """
    lora_path = Path(lora_path)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    if device is None:
        device = torch.device('cpu')

    if dtype is None:
        dtype = torch.float32

    # Load all tensors
    try:
        state_dict = load_file(str(lora_path))
    except Exception as e:
        raise ValueError(f"Failed to load LoRA file {lora_path}: {e}")

    # Parse LoRA layers
    lora_layers = {}

    # Group keys by layer name
    layer_groups = {}
    for key in state_dict.keys():
        # Extract base layer name and type (down/up)
        # Support patterns like:
        # - "model.blocks.0.attn.qkv.lora_down"
        # - "lora.blocks.0.attn.qkv.down"
        # - "blocks.0.attn.qkv_lora_A"

        # Try to match different patterns
        patterns = [
            r'(.+)\.lora_(down|up|A|B)',
            r'(.+)\.(down|up)',
            r'lora\.(.+)\.(down|up)'
        ]

        layer_name = None
        lora_type = None

        for pattern in patterns:
            match = re.search(pattern, key)
            if match:
                layer_name = match.group(1)
                lora_type_str = match.group(2).lower()

                # Normalize type names
                if lora_type_str in ['down', 'a']:
                    lora_type = 'down'
                elif lora_type_str in ['up', 'b']:
                    lora_type = 'up'

                break

        if layer_name and lora_type:
            if layer_name not in layer_groups:
                layer_groups[layer_name] = {}
            layer_groups[layer_name][lora_type] = state_dict[key]

        # Also check for alpha values
        if 'alpha' in key.lower():
            # Extract layer name from alpha key
            alpha_pattern = r'(.+)\.(?:lora_)?alpha'
            match = re.search(alpha_pattern, key)
            if match:
                layer_name = match.group(1)
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = {}
                layer_groups[layer_name]['alpha'] = state_dict[key].item()

    # Create LoRALayer objects
    for layer_name, tensors in layer_groups.items():
        if 'down' not in tensors or 'up' not in tensors:
            print(f"Warning: Skipping incomplete LoRA layer {layer_name}")
            continue

        down = tensors['down'].to(device=device, dtype=dtype)
        up = tensors['up'].to(device=device, dtype=dtype)

        # Get alpha and rank
        alpha = tensors.get('alpha', None)
        rank = down.shape[1] if len(down.shape) > 1 else down.shape[0]

        if alpha is None:
            alpha = float(rank)  # Default: α = r

        lora_layer = LoRALayer(down, up, alpha=alpha, rank=rank)
        lora_layers[layer_name] = lora_layer

    if not lora_layers:
        raise ValueError(f"No valid LoRA layers found in {lora_path}")

    # Create collection
    collection = LoRACollection(lora_layers, name=lora_path.stem)

    print(f"Loaded LoRA '{collection.name}' with {len(collection)} layers")

    return collection


def merge_lora_into_model(
    model: nn.Module,
    lora_collection: LoRACollection,
    strength: float = 1.0,
    key_map: Optional[Dict[str, str]] = None
) -> nn.Module:
    """
    Merge LoRA weights into a model in-place.

    This adds the LoRA delta weights to the model's existing weights:
    W_new = W_old + strength * delta_weight

    Args:
        model: Target model to merge LoRA into
        lora_collection: LoRA weights to merge
        strength: Multiplier for LoRA strength (0.0 = no effect, 1.0 = full effect)
        key_map: Optional mapping from LoRA keys to model parameter names

    Returns:
        Modified model (same object, modified in-place)

    Example:
        >>> model = DiT360Model(...)
        >>> lora = load_lora_from_safetensors("style.safetensors")
        >>> model = merge_lora_into_model(model, lora, strength=0.8)

    Note:
        This modifies the model in-place! Create a copy first if you need the original.
    """
    if strength == 0.0:
        print("LoRA strength is 0.0, skipping merge")
        return model

    # Get model state dict
    model_state = model.state_dict()

    # Track successful merges
    merged_count = 0
    skipped_count = 0

    for lora_key, lora_layer in lora_collection.lora_layers.items():
        # Map LoRA key to model key
        if key_map and lora_key in key_map:
            model_key = key_map[lora_key]
        else:
            # Try direct mapping
            model_key = lora_key

        # Check if key exists in model
        if model_key not in model_state:
            # Try adding common prefixes
            for prefix in ['model.', 'dit360.', '']:
                candidate_key = prefix + lora_key
                if candidate_key in model_state:
                    model_key = candidate_key
                    break
            else:
                print(f"Warning: LoRA key '{lora_key}' not found in model, skipping")
                skipped_count += 1
                continue

        # Get original weight
        orig_weight = model_state[model_key]

        # Compute delta weight
        delta_weight = lora_layer.get_delta_weight()

        # Ensure shapes match
        if orig_weight.shape != delta_weight.shape:
            print(f"Warning: Shape mismatch for {lora_key}: "
                  f"model={orig_weight.shape}, lora={delta_weight.shape}, skipping")
            skipped_count += 1
            continue

        # Merge with strength scaling
        new_weight = orig_weight + strength * delta_weight.to(orig_weight.device, orig_weight.dtype)

        # Update model
        model_state[model_key] = new_weight
        merged_count += 1

    # Load updated state dict
    model.load_state_dict(model_state, strict=False)

    print(f"Merged {merged_count}/{len(lora_collection)} LoRA layers "
          f"(skipped {skipped_count}) with strength {strength:.2f}")

    return model


def unmerge_lora_from_model(
    model: nn.Module,
    lora_collection: LoRACollection,
    strength: float = 1.0,
    key_map: Optional[Dict[str, str]] = None
) -> nn.Module:
    """
    Remove previously merged LoRA weights from a model.

    This subtracts the LoRA delta weights:
    W_new = W_old - strength * delta_weight

    Args:
        model: Model with merged LoRA
        lora_collection: LoRA weights to remove
        strength: Strength that was used during merge
        key_map: Optional mapping from LoRA keys to model parameter names

    Returns:
        Modified model (same object)

    Example:
        >>> model = merge_lora_into_model(model, lora, strength=0.8)
        >>> # ... use model ...
        >>> model = unmerge_lora_from_model(model, lora, strength=0.8)  # Restore original
    """
    # Unmerging is just merging with negative strength
    return merge_lora_into_model(model, lora_collection, strength=-strength, key_map=key_map)


def combine_loras(
    lora_collections: List[Tuple[LoRACollection, float]],
    name: Optional[str] = None
) -> LoRACollection:
    """
    Combine multiple LoRA collections with different strengths.

    This creates a new LoRA collection that represents the weighted sum of
    multiple LoRAs, useful for blending styles.

    Args:
        lora_collections: List of (LoRACollection, strength) tuples
        name: Name for the combined collection

    Returns:
        New LoRACollection with combined weights

    Example:
        >>> lora1 = load_lora_from_safetensors("anime.safetensors")
        >>> lora2 = load_lora_from_safetensors("realistic.safetensors")
        >>> combined = combine_loras([(lora1, 0.7), (lora2, 0.3)], name="mixed_style")

    Note:
        Only layers present in ALL collections will be included in the output.
    """
    if not lora_collections:
        raise ValueError("Must provide at least one LoRA collection")

    # Find common layer names
    common_layers = set(lora_collections[0][0].get_layer_names())
    for lora, _ in lora_collections[1:]:
        common_layers &= set(lora.get_layer_names())

    if not common_layers:
        raise ValueError("No common layers found across all LoRA collections")

    # Combine layers
    combined_layers = {}

    for layer_name in common_layers:
        # Get first layer for dimensions
        first_lora, first_strength = lora_collections[0]
        first_layer = first_lora.lora_layers[layer_name]

        # Initialize combined matrices
        combined_down = torch.zeros_like(first_layer.down)
        combined_up = torch.zeros_like(first_layer.up)

        # Sum weighted contributions
        total_alpha = 0.0
        for lora, strength in lora_collections:
            layer = lora.lora_layers[layer_name]
            combined_down += strength * layer.down
            combined_up += strength * layer.up
            total_alpha += strength * layer.alpha

        # Create combined layer
        combined_layer = LoRALayer(
            combined_down,
            combined_up,
            alpha=total_alpha,
            rank=first_layer.rank
        )
        combined_layers[layer_name] = combined_layer

    # Create collection
    combined_name = name or "_".join([lora.name for lora, _ in lora_collections])
    return LoRACollection(combined_layers, name=combined_name)


# Export all
__all__ = [
    'LoRALayer',
    'LoRACollection',
    'load_lora_from_safetensors',
    'merge_lora_into_model',
    'unmerge_lora_from_model',
    'combine_loras'
]
