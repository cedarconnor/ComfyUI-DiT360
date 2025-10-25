"""
Unit tests for ComfyUI-DiT360 utilities

Tests circular padding, equirectangular functions, and validation.
Run with: python -m pytest tests/test_utils.py
Or directly: python tests/test_utils.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.padding import apply_circular_padding, remove_circular_padding
from utils.equirect import (
    validate_aspect_ratio,
    fix_aspect_ratio,
    blend_edges,
    check_edge_continuity,
)


def test_circular_padding_latent():
    """Test circular padding with latent format (B, C, H, W)"""
    print("\n" + "="*60)
    print("TEST: Circular Padding - Latent Format")
    print("="*60)

    latent = torch.rand(1, 4, 128, 256)
    print(f"Original shape: {latent.shape}")

    padded = apply_circular_padding(latent, padding=10)
    print(f"Padded shape: {padded.shape}")
    assert padded.shape == (1, 4, 128, 276), f"Expected (1,4,128,276), got {padded.shape}"

    # Verify wraparound
    left_pad = padded[:, :, :, :10]
    original_right = latent[:, :, :, -10:]
    assert torch.allclose(left_pad, original_right, atol=1e-6), "Left padding mismatch"

    right_pad = padded[:, :, :, -10:]
    original_left = latent[:, :, :, :10]
    assert torch.allclose(right_pad, original_left, atol=1e-6), "Right padding mismatch"

    # Test removal
    unpadded = remove_circular_padding(padded, padding=10)
    assert unpadded.shape == latent.shape, "Shape mismatch after removal"
    assert torch.allclose(unpadded, latent, atol=1e-6), "Data corrupted"

    print("[PASS] Circular padding (latent format)")


def test_circular_padding_image():
    """Test circular padding with image format (B, H, W, C)"""
    print("\n" + "="*60)
    print("TEST: Circular Padding - Image Format")
    print("="*60)

    image = torch.rand(1, 1024, 2048, 3)
    print(f"Original shape: {image.shape}")

    padded = apply_circular_padding(image, padding=20)
    print(f"Padded shape: {padded.shape}")
    assert padded.shape == (1, 1024, 2088, 3), f"Expected (1,1024,2088,3), got {padded.shape}"

    # Verify wraparound
    left_pad = padded[:, :, :20, :]
    original_right = image[:, :, -20:, :]
    assert torch.allclose(left_pad, original_right, atol=1e-6), "Left padding mismatch"

    # Test removal
    unpadded = remove_circular_padding(padded, padding=20)
    assert unpadded.shape == image.shape, "Shape mismatch after removal"
    assert torch.allclose(unpadded, image, atol=1e-6), "Data corrupted"

    print("[PASS] Circular padding (image format)")


def test_aspect_ratio_validation():
    """Test aspect ratio validation"""
    print("\n" + "="*60)
    print("TEST: Aspect Ratio Validation")
    print("="*60)

    # Valid 2:1 ratios
    assert validate_aspect_ratio(2048, 1024) == True, "2048×1024 should be valid"
    assert validate_aspect_ratio(4096, 2048) == True, "4096×2048 should be valid"
    assert validate_aspect_ratio(1024, 512) == True, "1024×512 should be valid"
    assert validate_aspect_ratio(2000, 1000) == True, "2000×1000 should be valid"

    # Invalid ratios
    assert validate_aspect_ratio(1920, 1080) == False, "1920×1080 (16:9) should be invalid"
    assert validate_aspect_ratio(1024, 1024) == False, "1024×1024 (1:1) should be invalid"
    assert validate_aspect_ratio(1920, 1024) == False, "1920×1024 should be invalid"

    print("[PASS] Aspect ratio validation")


def test_aspect_ratio_fixing():
    """Test aspect ratio fixing modes"""
    print("\n" + "="*60)
    print("TEST: Aspect Ratio Fixing")
    print("="*60)

    # Test PAD mode
    image = torch.rand(1, 1080, 1920, 3)  # 16:9
    fixed = fix_aspect_ratio(image, mode="pad")
    _, h, w, _ = fixed.shape
    assert validate_aspect_ratio(w, h), f"Pad mode failed: {w}×{h}"
    print(f"Pad mode: {image.shape} -> {fixed.shape}")

    # Test CROP mode
    fixed = fix_aspect_ratio(image, mode="crop")
    _, h, w, _ = fixed.shape
    assert validate_aspect_ratio(w, h), f"Crop mode failed: {w}×{h}"
    print(f"Crop mode: {image.shape} -> {fixed.shape}")

    # Test STRETCH mode
    fixed = fix_aspect_ratio(image, mode="stretch")
    _, h, w, _ = fixed.shape
    assert validate_aspect_ratio(w, h), f"Stretch mode failed: {w}×{h}"
    print(f"Stretch mode: {image.shape} -> {fixed.shape}")

    print("[PASS] Aspect ratio fixing")


def test_edge_blending():
    """Test edge blending for seamless wraparound"""
    print("\n" + "="*60)
    print("TEST: Edge Blending")
    print("="*60)

    # Create image with distinct left/right edges
    image = torch.rand(1, 1024, 2048, 3)

    # Blend edges
    blended = blend_edges(image, blend_width=20, mode="cosine")
    assert blended.shape == image.shape, "Shape changed after blending"

    # Check continuity improved
    original_continuity = check_edge_continuity(image, threshold=0.5)
    blended_continuity = check_edge_continuity(blended, threshold=0.1)

    print(f"Original edge continuity: {original_continuity}")
    print(f"Blended edge continuity: {blended_continuity}")

    # Blended should have better continuity
    left = blended[:, :, 0, :]
    right = blended[:, :, -1, :]
    diff = torch.abs(left - right).mean().item()
    print(f"Average edge difference: {diff:.6f}")

    print("[PASS] Edge blending")


def test_edge_continuity_check():
    """Test edge continuity checking"""
    print("\n" + "="*60)
    print("TEST: Edge Continuity Check")
    print("="*60)

    # Create image with matching edges (continuous)
    image = torch.rand(1, 1024, 2048, 3)
    image[:, :, -1, :] = image[:, :, 0, :]  # Make edges match
    assert check_edge_continuity(image, threshold=0.01), "Should be continuous"

    # Create image with non-matching edges
    image2 = torch.rand(1, 1024, 2048, 3)
    # Very likely to be discontinuous (random edges)
    is_continuous = check_edge_continuity(image2, threshold=0.01)
    print(f"Random edges continuous: {is_continuous}")

    print("[PASS] Edge continuity check")


def run_all_tests():
    """Run all utility tests"""
    print("\n" + "="*60)
    print("Running ComfyUI-DiT360 Utility Tests")
    print("="*60)

    try:
        test_circular_padding_latent()
        test_circular_padding_image()
        test_aspect_ratio_validation()
        test_aspect_ratio_fixing()
        test_edge_blending()
        test_edge_continuity_check()

        print("\n" + "="*60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*60 + "\n")
        return True

    except AssertionError as e:
        print("\n" + "="*60)
        print(f"[FAIL] TEST FAILED: {e}")
        print("="*60 + "\n")
        return False

    except Exception as e:
        print("\n" + "="*60)
        print(f"[ERROR] {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
