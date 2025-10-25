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
            - Latent format: (B, C, H, W) where C <= 4
            - Image format: (B, H, W, C) where C = 3
        padding: Number of pixels to pad on left/right edges

    Returns:
        Padded tensor with wraparound continuity

    Example:
        >>> latent = torch.rand(1, 4, 128, 256)  # Latent
        >>> padded = apply_circular_padding(latent, padding=10)
        >>> padded.shape
        torch.Size([1, 4, 128, 276])  # Width increased by 20 (10 each side)

        >>> image = torch.rand(1, 1024, 2048, 3)  # Image
        >>> padded = apply_circular_padding(image, padding=20)
        >>> padded.shape
        torch.Size([1, 1024, 2088, 3])  # Width increased by 40 (20 each side)
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")

    # Detect format: (B, C, H, W) vs (B, H, W, C)
    if tensor.shape[1] <= 4:
        # Latent format: (B, C, H, W)
        # Take from width dimension (dim 3)
        left_edge = tensor[:, :, :, :padding]      # Leftmost columns
        right_edge = tensor[:, :, :, -padding:]    # Rightmost columns

        # Concatenate: right_edge | tensor | left_edge
        # This wraps the right edge to the left side and vice versa
        padded = torch.cat([right_edge, tensor, left_edge], dim=3)

    else:
        # Image format: (B, H, W, C)
        # Take from width dimension (dim 2)
        left_edge = tensor[:, :, :padding, :]      # Leftmost columns
        right_edge = tensor[:, :, -padding:, :]    # Rightmost columns

        # Concatenate: right_edge | tensor | left_edge
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

    Example:
        >>> padded = torch.rand(1, 4, 128, 276)
        >>> original = remove_circular_padding(padded, padding=10)
        >>> original.shape
        torch.Size([1, 4, 128, 256])
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")

    # Detect format and remove padding from width dimension
    if tensor.shape[1] <= 4:
        # Latent format: (B, C, H, W)
        return tensor[:, :, :, padding:-padding]
    else:
        # Image format: (B, H, W, C)
        return tensor[:, :, padding:-padding, :]


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


def test_circular_padding():
    """
    Test function to validate circular padding implementation

    Run this to ensure circular padding works correctly.
    """
    print("Testing circular padding...")

    # Test 1: Latent format
    print("\n1. Testing latent format (B, C, H, W)...")
    latent = torch.rand(1, 4, 128, 256)
    padded = apply_circular_padding(latent, padding=10)
    unpadded = remove_circular_padding(padded, padding=10)

    assert padded.shape == (1, 4, 128, 276), f"Expected (1,4,128,276), got {padded.shape}"
    assert unpadded.shape == latent.shape, f"Shape mismatch after removing padding"
    assert torch.allclose(unpadded, latent), "Data corrupted after padding/unpadding"
    print("   ✓ Latent format works correctly")

    # Test 2: Image format
    print("\n2. Testing image format (B, H, W, C)...")
    image = torch.rand(1, 1024, 2048, 3)
    padded = apply_circular_padding(image, padding=20)
    unpadded = remove_circular_padding(padded, padding=20)

    assert padded.shape == (1, 1024, 2088, 3), f"Expected (1,1024,2088,3), got {padded.shape}"
    assert unpadded.shape == image.shape, f"Shape mismatch after removing padding"
    assert torch.allclose(unpadded, image), "Data corrupted after padding/unpadding"
    print("   ✓ Image format works correctly")

    # Test 3: Wraparound continuity
    print("\n3. Testing wraparound continuity...")
    latent = torch.rand(1, 4, 128, 256)
    padded = apply_circular_padding(latent, padding=10)

    # Check that left padding matches original right edge
    left_pad = padded[:, :, :, :10]
    original_right = latent[:, :, :, -10:]
    assert torch.allclose(left_pad, original_right), "Left padding doesn't match original right edge"

    # Check that right padding matches original left edge
    right_pad = padded[:, :, :, -10:]
    original_left = latent[:, :, :, :10]
    assert torch.allclose(right_pad, original_left), "Right padding doesn't match original left edge"
    print("   ✓ Wraparound continuity verified")

    print("\n✅ All circular padding tests passed!\n")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_circular_padding()
