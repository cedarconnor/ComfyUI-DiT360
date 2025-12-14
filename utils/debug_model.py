"""
Debug utilities to inspect model structure and find RoPE embedder.

Run this to see what attributes your model has.
"""

import torch
import torch.nn as nn


def inspect_model_structure(model, max_depth=3):
    """
    Print the structure of a ComfyUI model to find RoPE embedder.

    Usage in ComfyUI:
        # In a custom node or Python console:
        from utils.debug_model import inspect_model_structure
        inspect_model_structure(model)
    """
    print("\n" + "="*80)
    print("MODEL STRUCTURE INSPECTION")
    print("="*80)

    # Get the actual model
    if hasattr(model, 'model'):
        base = model.model
        print(f"\nmodel.model type: {type(base).__name__}")
    else:
        base = model
        print(f"\nmodel type: {type(base).__name__}")

    # Check for diffusion_model
    if hasattr(base, 'diffusion_model'):
        diff_model = base.diffusion_model
        print(f"\nmodel.model.diffusion_model type: {type(diff_model).__name__}")

        # List top-level attributes
        print("\nTop-level attributes of diffusion_model:")
        for name in dir(diff_model):
            if not name.startswith('_'):
                attr = getattr(diff_model, name, None)
                if isinstance(attr, nn.Module):
                    print(f"  - {name}: {type(attr).__name__}")
                elif isinstance(attr, (int, float, str, tuple, list)):
                    print(f"  - {name}: {type(attr).__name__} = {attr}")

        # Search for RoPE-related modules
        print("\nSearching for RoPE/position-related modules:")
        rope_keywords = ['rope', 'pos', 'embed', 'rotary', 'position', 'pe_']
        found_any = False

        for name, module in diff_model.named_modules():
            name_lower = name.lower()
            if any(kw in name_lower for kw in rope_keywords):
                print(f"  FOUND: {name} -> {type(module).__name__}")
                found_any = True

                # Print module attributes
                if hasattr(module, 'theta'):
                    print(f"         theta: {module.theta}")
                if hasattr(module, 'axes_dim'):
                    print(f"         axes_dim: {module.axes_dim}")
                if hasattr(module, 'dim'):
                    print(f"         dim: {module.dim}")

        if not found_any:
            print("  No RoPE-related modules found by keyword search")
            print("\n  All module names (first 50):")
            for i, (name, module) in enumerate(diff_model.named_modules()):
                if i >= 50:
                    print(f"  ... and more")
                    break
                if name:  # Skip root module
                    print(f"    {name}: {type(module).__name__}")

    # Check for VAE
    if hasattr(base, 'first_stage_model'):
        print(f"\nVAE (first_stage_model) type: {type(base.first_stage_model).__name__}")

    print("\n" + "="*80)


def find_rope_embedder_candidates(model):
    """
    Return a list of candidate RoPE embedder modules.
    """
    candidates = []

    if hasattr(model, 'model'):
        base = model.model
    else:
        base = model

    if hasattr(base, 'diffusion_model'):
        diff_model = base.diffusion_model
    else:
        diff_model = base

    # Direct attributes to check
    direct_attrs = [
        'pos_embed', 'rope_embedder', 'position_embedder',
        'x_embedder', 'patch_embed', 'rotary_emb', 'rope',
        'img_in', 'pe_embedder', 'positional_embedding'
    ]

    for attr in direct_attrs:
        if hasattr(diff_model, attr):
            candidates.append((attr, getattr(diff_model, attr)))

    # Search in named_modules
    rope_keywords = ['rope', 'pos_embed', 'rotary', 'position']
    for name, module in diff_model.named_modules():
        name_lower = name.lower()
        if any(kw in name_lower for kw in rope_keywords):
            if (name, module) not in candidates:
                candidates.append((name, module))

    return candidates


class DebugModelStructure:
    """
    ComfyUI node to inspect model structure and find RoPE embedder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "debug"
    CATEGORY = "DiT360/debug"
    OUTPUT_NODE = True

    def debug(self, model):
        """Print model structure and return unchanged."""
        inspect_model_structure(model)

        print("\n" + "-"*40)
        print("RoPE EMBEDDER CANDIDATES:")
        print("-"*40)
        candidates = find_rope_embedder_candidates(model)

        if candidates:
            for name, module in candidates:
                print(f"  {name}: {type(module).__name__}")
        else:
            print("  No candidates found!")
            print("  The model may use a different architecture.")

        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DebugModelStructure": DebugModelStructure,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DebugModelStructure": "Debug Model Structure (DiT360)",
}


if __name__ == "__main__":
    print("Run this in ComfyUI with a loaded model to inspect structure")
