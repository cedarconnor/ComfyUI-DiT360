"""
Test DiT360 model loading infrastructure

Tests that model, VAE, and text encoder loaders work correctly
even without actual model files present.

Run with: python tests/test_model_loading.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_model_imports():
    """Test that all model modules import correctly"""
    print("\n" + "="*60)
    print("TEST: Module Imports")
    print("="*60)

    try:
        from dit360 import (
            DiT360Model,
            DiT360Wrapper,
            load_dit360_model,
            DiT360VAE,
            load_vae,
            T5TextEncoder,
            load_t5_encoder,
            text_preprocessing,
        )
        print("[PASS] All modules imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_text_preprocessing():
    """Test text preprocessing function"""
    print("\n" + "="*60)
    print("TEST: Text Preprocessing")
    print("="*60)

    from dit360 import text_preprocessing

    # Test cases
    test_cases = [
        ("  A   BEAUTIFUL    sunset  ", "a beautiful sunset"),
        ("PANORAMA!!!", "panorama!"),
        ("  Mixed   Case   Text  ", "mixed case text"),
        ("", ""),
    ]

    for input_text, expected in test_cases:
        result = text_preprocessing(input_text)
        print(f"  Input: '{input_text}'")
        print(f"  Output: '{result}'")
        print(f"  Expected: '{expected}'")

        if result == expected:
            print("  [PASS]")
        else:
            print("  [FAIL]")
            return False

    print("\n[PASS] Text preprocessing works correctly")
    return True


def test_dit360_model_init():
    """Test DiT360 model initialization"""
    print("\n" + "="*60)
    print("TEST: DiT360 Model Initialization")
    print("="*60)

    from dit360 import DiT360Model

    config = {
        "in_channels": 4,
        "hidden_size": 3072,
        "num_layers": 38,
        "num_heads": 24,
        "caption_channels": 4096,
    }

    try:
        model = DiT360Model(config, enable_circular_padding=True)
        print(f"  Model created with config: {config}")
        print(f"  Circular padding: {model.enable_circular_padding}")
        print("[PASS] DiT360Model initialized")
        return True
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        return False


def test_vae_wrapper():
    """Test VAE wrapper initialization"""
    print("\n" + "="*60)
    print("TEST: VAE Wrapper")
    print("="*60)

    from dit360.vae import DiT360VAE
    import torch.nn as nn

    # Create dummy VAE model
    class DummyVAE(nn.Module):
        def __init__(self):
            super().__init__()

    dummy_vae = DummyVAE()

    try:
        vae_wrapper = DiT360VAE(
            vae_model=dummy_vae,
            dtype=torch.float16,
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            scale_factor=8
        )

        # Test latent size calculation
        latent_size = vae_wrapper.get_latent_size((1024, 2048))
        expected = (128, 256)

        print(f"  Image size: (1024, 2048)")
        print(f"  Latent size: {latent_size}")
        print(f"  Expected: {expected}")

        if latent_size == expected:
            print("[PASS] VAE wrapper works correctly")
            return True
        else:
            print("[FAIL] Latent size mismatch")
            return False

    except Exception as e:
        print(f"[FAIL] VAE wrapper failed: {e}")
        return False


def test_t5_encoder_wrapper():
    """Test T5 encoder wrapper initialization"""
    print("\n" + "="*60)
    print("TEST: T5 Encoder Wrapper")
    print("="*60)

    from dit360.conditioning import T5TextEncoder
    import torch.nn as nn

    # Create dummy T5 model and tokenizer
    class DummyT5(nn.Module):
        def __init__(self):
            super().__init__()

    class DummyTokenizer:
        def __call__(self, *args, **kwargs):
            class Tokens:
                def __init__(self):
                    self.input_ids = torch.zeros(1, 512, dtype=torch.long)
                    self.attention_mask = torch.ones(1, 512, dtype=torch.long)
            return Tokens()

    dummy_model = DummyT5()
    dummy_tokenizer = DummyTokenizer()

    try:
        encoder = T5TextEncoder(
            model=dummy_model,
            tokenizer=dummy_tokenizer,
            dtype=torch.float16,
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            max_length=512
        )

        # Test text preprocessing
        processed = encoder.preprocess_text("  A   BEAUTIFUL    SUNSET  ")
        expected = "a beautiful sunset"

        print(f"  Input: '  A   BEAUTIFUL    SUNSET  '")
        print(f"  Output: '{processed}'")
        print(f"  Expected: '{expected}'")

        if processed == expected:
            print("[PASS] T5 encoder wrapper works correctly")
            return True
        else:
            print("[FAIL] Preprocessing mismatch")
            return False

    except Exception as e:
        print(f"[FAIL] T5 encoder wrapper failed: {e}")
        return False


def test_model_info_function():
    """Test get_model_info function"""
    print("\n" + "="*60)
    print("TEST: Model Info Function")
    print("="*60)

    from dit360 import get_model_info

    # Test with non-existent path
    info = get_model_info("nonexistent_model.safetensors")

    print(f"  Path exists: {info['exists']}")
    print(f"  Expected: False")

    if info['exists'] == False:
        print("[PASS] get_model_info works correctly")
        return True
    else:
        print("[FAIL] Unexpected result")
        return False


def run_all_tests():
    """Run all model loading tests"""
    print("\n" + "="*60)
    print("Running ComfyUI-DiT360 Model Loading Tests")
    print("="*60)

    tests = [
        test_model_imports,
        test_text_preprocessing,
        test_dit360_model_init,
        test_vae_wrapper,
        test_t5_encoder_wrapper,
        test_model_info_function,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"[SUCCESS] ALL TESTS PASSED! ({passed}/{total})")
        print("="*60 + "\n")
        return True
    else:
        print(f"[FAIL] {total - passed}/{total} tests failed")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
