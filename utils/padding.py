"""
Circular padding utilities for seamless panorama wraparound

Implements circular padding on the width dimension to ensure panoramas
wrap seamlessly at the left/right edges.
"""

import torch
import torch.nn.functional as F


def apply_circular_padding(
    tensor: torch.Tensor,
    padding: int
) -> torch.Tensor:
    """
    Apply circular padding to create seamless wraparound at panorama edges

    Takes pixels from the right edge and wraps them to the left side,
    and vice versa, creating continuity for 360° panoramas.

    Args:
        tensor: Input tensor, can be:
            - Latent format: (B, C, H, W) - padding applied to W dimension
            - Image format: (B, H, W, C) - padding applied to W dimension
        padding: Number of pixels to pad on left/right edges

    Returns:
        Padded tensor with wraparound continuity
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")

    # Early return if no padding needed
    if padding <= 0:
        return tensor

    # Detect format by checking last dimension
    # Image format has C=3 or 4 (RGB/RGBA) as last dim
    # Latent format has W (width, typically large) as last dim
    is_image_format = tensor.shape[-1] in (3, 4) and tensor.shape[1] > 4

    if not is_image_format:
        # Latent format: (B, C, H, W) - includes FLUX 16-channel
        # Pad width dimension (dim 3)
        left_edge = tensor[:, :, :, :padding]
        right_edge = tensor[:, :, :, -padding:]
        padded = torch.cat([right_edge, tensor, left_edge], dim=3)
    else:
        # Image format: (B, H, W, C)
        # Pad width dimension (dim 2)
        left_edge = tensor[:, :, :padding, :]
        right_edge = tensor[:, :, -padding:, :]
        padded = torch.cat([right_edge, tensor, left_edge], dim=2)

    return padded


def remove_circular_padding(
    tensor: torch.Tensor,
    padding: int
) -> torch.Tensor:
    """
    Remove circular padding after processing

    Removes the padding added by apply_circular_padding to return
    to original dimensions.

    Args:
        tensor: Padded tensor from apply_circular_padding
        padding: Number of pixels to remove from each side

    Returns:
        Tensor with padding removed
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")

    # Early return if no padding to remove
    if padding <= 0:
        return tensor

    # Detect format by checking last dimension
    is_image_format = tensor.shape[-1] in (3, 4) and tensor.shape[1] > 4

    if not is_image_format:
        # Latent format: (B, C, H, W)
        # Use explicit slicing to avoid -0 issue
        width = tensor.shape[3]
        return tensor[:, :, :, padding:width-padding]
    else:
        # Image format: (B, H, W, C)
        width = tensor.shape[2]
        return tensor[:, :, padding:width-padding, :]


def circular_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    2D convolution with circular padding on width dimension

    Ensures convolution operations respect panorama wraparound by applying
    circular padding before convolution.

    NOTE: This is a utility for Phase 4+ when implementing custom conv layers
    in the DiT360 architecture.

    Args:
        input: Input tensor (B, C, H, W)
        weight: Convolution weights
        bias: Optional bias
        stride: Convolution stride
        padding: Padding amount
        dilation: Dilation factor
        groups: Number of groups for grouped convolution

    Returns:
        Convolved tensor with circular padding applied

    Example:
        >>> input = torch.rand(1, 64, 128, 256)
        >>> weight = torch.rand(64, 64, 3, 3)
        >>> output = circular_conv2d(input, weight, padding=1)
    """
    # Apply circular padding on width dimension
    if padding > 0:
        if isinstance(padding, int):
            pad_h, pad_w = padding, padding
        else:
            pad_h, pad_w = padding

        # Pad height normally (top/bottom with zeros)
        if pad_h > 0:
            input = F.pad(input, (0, 0, pad_h, pad_h), mode='constant', value=0)

        # Pad width circularly (left/right wraparound)
        if pad_w > 0:
            input = apply_circular_padding(input, pad_w)

        padding = 0  # Already padded manually

    # Standard convolution (now with circular padding applied)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


def create_circular_padding_wrapper(model, circular_padding: int):
    """
    Create a wrapper that applies circular padding to model forward passes

    IMPORTANT: Only call this on a model.clone(), never on the original model!
    This function mutates model.model.apply_model and can leak if not used carefully.

    Args:
        model: ComfyUI ModelPatcher object (must be a clone!)
        circular_padding: Padding width in latent space

    Returns:
        Model with circular padding wrapper applied

    Example:
        >>> model_clone = model.clone()  # CRITICAL: always clone first!
        >>> model_clone = create_circular_padding_wrapper(model_clone, 16)
    """
    if circular_padding <= 0:
        return model

    # Get the underlying model
    underlying_model = model.model

    # Check if already wrapped (avoid double-wrapping)
    if hasattr(underlying_model, '_original_apply_model_circ'):
        return model  # Already wrapped

    # Store original apply_model function
    original_apply_model = underlying_model.apply_model

    def wrapped_apply_model(x, t, **kwargs):
        """Wrapped apply_model with circular padding"""
        # Apply circular padding to input
        x_padded = apply_circular_padding(x, circular_padding)

        # Call original model
        output_padded = original_apply_model(x_padded, t, **kwargs)

        # Remove padding from output
        output = remove_circular_padding(output_padded, circular_padding)

        return output

    # Replace model's apply_model and store original for restoration
    underlying_model.apply_model = wrapped_apply_model
    underlying_model._original_apply_model_circ = original_apply_model

    return model


def test_circular_padding():
    """Test circular padding implementation."""
    print("Testing circular padding...")

    # Test 1: SD/SDXL 4-channel latent
    print("\n1. Testing 4-channel latent (B, C, H, W)...")
    latent = torch.rand(1, 4, 128, 256)
    padded = apply_circular_padding(latent, padding=10)
    unpadded = remove_circular_padding(padded, padding=10)
    assert padded.shape == (1, 4, 128, 276), f"Expected (1,4,128,276), got {padded.shape}"
    assert torch.allclose(unpadded, latent), "Data corrupted"
    print("   ✓ 4-channel latent works")

    # Test 2: FLUX 16-channel latent
    print("\n2. Testing FLUX 16-channel latent (B, C, H, W)...")
    latent = torch.rand(1, 16, 128, 256)
    padded = apply_circular_padding(latent, padding=16)
    unpadded = remove_circular_padding(padded, padding=16)
    assert padded.shape == (1, 16, 128, 288), f"Expected (1,16,128,288), got {padded.shape}"
    assert torch.allclose(unpadded, latent), "Data corrupted"
    print("   ✓ FLUX 16-channel latent works")

    # Test 3: Image format
    print("\n3. Testing image format (B, H, W, C)...")
    image = torch.rand(1, 1024, 2048, 3)
    padded = apply_circular_padding(image, padding=20)
    unpadded = remove_circular_padding(padded, padding=20)
    assert padded.shape == (1, 1024, 2088, 3), f"Expected (1,1024,2088,3), got {padded.shape}"
    assert torch.allclose(unpadded, image), "Data corrupted"
    print("   ✓ Image format works")

    # Test 4: Wraparound continuity
    print("\n4. Testing wraparound continuity...")
    latent = torch.rand(1, 16, 128, 256)
    padded = apply_circular_padding(latent, padding=10)
    assert torch.allclose(padded[:, :, :, :10], latent[:, :, :, -10:]), "Left padding mismatch"
    assert torch.allclose(padded[:, :, :, -10:], latent[:, :, :, :10]), "Right padding mismatch"
    print("   ✓ Wraparound verified")

    print("\n✅ All circular padding tests passed!\n")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_circular_padding()
