"""
Test Phase 3: Model Inference Components

Tests for DiT360 transformer, flow matching scheduler, VAE, and T5 text encoding.
These tests validate the core inference pipeline without requiring actual model weights.

Run with: python tests/test_phase3_inference.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock comfy module for standalone testing
if 'comfy' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['comfy'] = MagicMock()
    sys.modules['comfy.model_management'] = MagicMock()
    sys.modules['comfy.utils'] = MagicMock()

    # Mock folder_paths
    sys.modules['folder_paths'] = MagicMock()
    sys.modules['folder_paths'].models_dir = str(Path(__file__).parent.parent / "models")

import torch
import torch.nn as nn


def test_dit360_model_architecture():
    """Test DiT360 model initialization and forward pass"""
    print("\n" + "="*60)
    print("TEST: DiT360 Model Architecture")
    print("="*60)

    from dit360.model import DiT360Model

    # Create model with test config
    config = {
        "in_channels": 4,
        "hidden_size": 512,  # Smaller for testing
        "num_layers": 4,     # Fewer layers for testing
        "num_heads": 8,
        "caption_channels": 512,
        "patch_size": 2,
        "mlp_ratio": 4.0,
        "circular_padding_width": 2,
    }

    try:
        model = DiT360Model(config, enable_circular_padding=True)
        print(f"  [OK] Model initialized with {config['num_layers']} layers")

        # Test forward pass
        batch_size = 1
        latent_h, latent_w = 16, 32  # Small test size
        x = torch.randn(batch_size, 4, latent_h, latent_w)
        timestep = torch.tensor([0.5])
        context = torch.randn(batch_size, 77, config['caption_channels'])

        output = model(x, timestep, context)

        print(f"  [OK] Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        # Verify output shape matches input
        if output.shape == x.shape:
            print("  [OK] Output shape matches input")
        else:
            print(f"  [FAIL] Shape mismatch: {output.shape} != {x.shape}")
            return False

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")

        print("[PASS] DiT360 model architecture test")
        return True

    except Exception as e:
        print(f"[FAIL] DiT360 model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flow_matching_scheduler():
    """Test flow matching scheduler"""
    print("\n" + "="*60)
    print("TEST: Flow Matching Scheduler")
    print("="*60)

    from dit360.scheduler import FlowMatchScheduler

    try:
        # Initialize scheduler
        scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=1.0
        )
        print("  [OK] Scheduler initialized")

        # Set timesteps
        num_steps = 50
        scheduler.set_timesteps(num_steps, device='cpu')
        print(f"  [OK] Timesteps set: {len(scheduler.timesteps)} steps")

        # Verify timesteps are decreasing (from 1 to 0)
        if scheduler.timesteps[0] > scheduler.timesteps[-1]:
            print("  [OK] Timesteps decrease from noise to data")
        else:
            print("  [FAIL] Timesteps should decrease")
            return False

        # Test scheduler step
        sample = torch.randn(1, 4, 16, 32)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0].item()

        prev_sample = scheduler.step(model_output, timestep, sample)

        print(f"  [OK] Scheduler step successful")
        print(f"  Sample shape: {prev_sample.shape}")

        # Test add_noise
        clean = torch.randn(1, 4, 16, 32)
        noise = torch.randn_like(clean)
        t = torch.tensor([0.5])

        noisy = scheduler.add_noise(clean, noise, t)
        print(f"  [OK] Noise addition successful")

        print("[PASS] Flow matching scheduler test")
        return True

    except Exception as e:
        print(f"[FAIL] Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rope_embeddings():
    """Test rotary positional embeddings"""
    print("\n" + "="*60)
    print("TEST: RoPE Embeddings")
    print("="*60)

    from dit360.model import RoPEEmbedding, apply_rotary_emb

    try:
        dim = 64
        max_seq_len = 512
        rope = RoPEEmbedding(dim, max_seq_len)

        print("  [OK] RoPE initialized")

        # Test embeddings
        x = torch.randn(2, 128, 8, dim)  # (B, seq_len, num_heads, head_dim)
        freqs_cos, freqs_sin = rope(x)

        print(f"  [OK] RoPE embeddings generated")
        print(f"  Cos shape: {freqs_cos.shape}")
        print(f"  Sin shape: {freqs_sin.shape}")

        # Apply to tensor
        x_rope = apply_rotary_emb(x, freqs_cos, freqs_sin)

        if x_rope.shape == x.shape:
            print("  [OK] RoPE application preserves shape")
        else:
            print("  [FAIL] Shape mismatch after RoPE")
            return False

        print("[PASS] RoPE embeddings test")
        return True

    except Exception as e:
        print(f"[FAIL] RoPE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_layer_norm():
    """Test adaptive layer normalization"""
    print("\n" + "="*60)
    print("TEST: Adaptive Layer Norm")
    print("="*60)

    from dit360.model import AdaptiveLayerNorm

    try:
        hidden_size = 256
        conditioning_dim = 128

        adaln = AdaptiveLayerNorm(hidden_size, conditioning_dim)
        print("  [OK] AdaLN initialized")

        # Test forward
        x = torch.randn(2, 50, hidden_size)
        conditioning = torch.randn(2, conditioning_dim)

        output = adaln(x, conditioning)

        if output.shape == x.shape:
            print("  [OK] AdaLN output shape correct")
        else:
            print("  [FAIL] AdaLN shape mismatch")
            return False

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        print("[PASS] Adaptive layer norm test")
        return True

    except Exception as e:
        print(f"[FAIL] AdaLN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_circular_padding_attention():
    """Test multi-head attention with circular padding"""
    print("\n" + "="*60)
    print("TEST: Circular Padding Attention")
    print("="*60)

    from dit360.model import MultiHeadAttention

    try:
        hidden_size = 256
        num_heads = 8
        padding_width = 2

        attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            enable_circular_padding=True,
            circular_padding_width=padding_width
        )
        print("  [OK] Attention initialized with circular padding")

        # Test forward
        batch_size = 1
        height, width = 8, 16
        seq_len = height * width
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = attn(x, height, width)

        if output.shape == x.shape:
            print("  [OK] Attention output shape correct")
        else:
            print("  [FAIL] Attention shape mismatch")
            return False

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        # Test without circular padding
        attn_no_pad = MultiHeadAttention(
            hidden_size,
            num_heads,
            enable_circular_padding=False
        )

        output_no_pad = attn_no_pad(x, height, width)
        print("  [OK] Attention works without circular padding too")

        print("[PASS] Circular padding attention test")
        return True

    except Exception as e:
        print(f"[FAIL] Attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_wrapper():
    """Test VAE wrapper functionality"""
    print("\n" + "="*60)
    print("TEST: VAE Wrapper")
    print("="*60)

    from dit360.vae import DiT360VAE

    try:
        # Create dummy VAE
        class DummyVAE(nn.Module):
            def encode(self, x):
                class Output:
                    def __init__(self, latent):
                        class LatentDist:
                            def __init__(self, latent):
                                self.latent = latent
                            def sample(self):
                                return self.latent
                        self.latent_dist = LatentDist(latent)
                # 8x downsample
                latent = torch.nn.functional.avg_pool2d(x, kernel_size=8)
                # Ensure 4 channels
                if latent.shape[1] != 4:
                    latent = torch.cat([latent, torch.zeros_like(latent[:, :1, :, :])], dim=1)[:, :4, :, :]
                return Output(latent)

            def decode(self, x):
                class Output:
                    def __init__(self, sample):
                        self.sample = sample
                # 8x upsample
                image = torch.nn.functional.interpolate(x, scale_factor=8, mode='bilinear')
                # Ensure 3 channels
                if image.shape[1] != 3:
                    image = image[:, :3, :, :]
                return Output(image)

        dummy_vae = DummyVAE()
        vae_wrapper = DiT360VAE(
            vae_model=dummy_vae,
            dtype=torch.float32,
            device=torch.device('cpu'),
            offload_device=torch.device('cpu'),
            scale_factor=8
        )

        print("  [OK] VAE wrapper initialized")

        # Test encoding
        image = torch.rand(1, 1024, 2048, 3)  # ComfyUI format
        latent = vae_wrapper.encode(image)

        expected_shape = (1, 4, 128, 256)  # 8x downscale
        if latent.shape == expected_shape:
            print(f"  [OK] Encode: {image.shape} -> {latent.shape}")
        else:
            print(f"  [FAIL] Encode shape mismatch: {latent.shape} != {expected_shape}")
            return False

        # Test decoding
        decoded = vae_wrapper.decode(latent)

        expected_decode_shape = (1, 1024, 2048, 3)
        if decoded.shape == expected_decode_shape:
            print(f"  [OK] Decode: {latent.shape} -> {decoded.shape}")
        else:
            print(f"  [FAIL] Decode shape mismatch: {decoded.shape} != {expected_decode_shape}")
            return False

        # Test latent size calculation
        latent_size = vae_wrapper.get_latent_size((1024, 2048))
        if latent_size == (128, 256):
            print("  [OK] Latent size calculation correct")
        else:
            print(f"  [FAIL] Latent size mismatch: {latent_size} != (128, 256)")
            return False

        print("[PASS] VAE wrapper test")
        return True

    except Exception as e:
        print(f"[FAIL] VAE wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_encoder_wrapper():
    """Test T5 text encoder wrapper"""
    print("\n" + "="*60)
    print("TEST: T5 Text Encoder Wrapper")
    print("="*60)

    from dit360.conditioning import T5TextEncoder

    try:
        # Create dummy T5 model and tokenizer
        class DummyT5(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
                batch_size, seq_len = input_ids.shape
                class Output:
                    def __init__(self, hidden_states):
                        self.last_hidden_state = hidden_states
                hidden_states = torch.randn(batch_size, seq_len, 4096)
                return Output(hidden_states)

        class DummyTokenizer:
            def __init__(self):
                self.model_max_length = 512

            def __call__(self, texts, padding="max_length", max_length=None, truncation=True, return_tensors="pt"):
                max_len = max_length or self.model_max_length
                batch_size = len(texts) if isinstance(texts, list) else 1

                class Tokens:
                    def __init__(self, batch_size, max_len):
                        self.input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
                        self.attention_mask = torch.ones(batch_size, max_len, dtype=torch.long)

                return Tokens(batch_size, max_len)

            def __len__(self):
                return 32000

        dummy_model = DummyT5()
        dummy_tokenizer = DummyTokenizer()

        encoder = T5TextEncoder(
            model=dummy_model,
            tokenizer=dummy_tokenizer,
            dtype=torch.float32,
            device=torch.device('cpu'),
            offload_device=torch.device('cpu'),
            max_length=512
        )

        print("  [OK] T5 encoder wrapper initialized")

        # Test encoding
        result = encoder.encode(
            prompts="A beautiful sunset over the ocean",
            negative_prompts="blurry, low quality"
        )

        if "prompt_embeds" in result and "negative_prompt_embeds" in result:
            print("  [OK] Encoding returns both positive and negative embeddings")
        else:
            print("  [FAIL] Missing embeddings in result")
            return False

        print(f"  Prompt embeds shape: {result['prompt_embeds'].shape}")
        print(f"  Negative embeds shape: {result['negative_prompt_embeds'].shape}")

        # Test text preprocessing
        processed = encoder.preprocess_text("  A   BEAUTIFUL    SUNSET  ")
        expected = "a beautiful sunset"
        if processed == expected:
            print(f"  [OK] Text preprocessing: '{processed}'")
        else:
            print(f"  [FAIL] Text preprocessing mismatch: '{processed}' != '{expected}'")
            return False

        print("[PASS] T5 text encoder wrapper test")
        return True

    except Exception as e:
        print(f"[FAIL] T5 encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_preprocessing():
    """Test global text preprocessing function"""
    print("\n" + "="*60)
    print("TEST: Text Preprocessing")
    print("="*60)

    from dit360.conditioning import text_preprocessing

    test_cases = [
        ("  A   BEAUTIFUL    sunset  ", "a beautiful sunset"),
        ("PANORAMA!!!", "panorama!"),
        ("  Mixed   Case   Text  ", "mixed case text"),
        ("", ""),
    ]

    try:
        for input_text, expected in test_cases:
            result = text_preprocessing(input_text)
            if result == expected:
                print(f"  [OK] '{input_text}' -> '{result}'")
            else:
                print(f"  [FAIL] '{input_text}' -> '{result}' (expected '{expected}')")
                return False

        print("[PASS] Text preprocessing test")
        return True

    except Exception as e:
        print(f"[FAIL] Text preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_forward_pass():
    """Test complete forward pass integration"""
    print("\n" + "="*60)
    print("TEST: Integration - Complete Forward Pass")
    print("="*60)

    try:
        from dit360.model import DiT360Model
        from dit360.scheduler import FlowMatchScheduler

        # Small model for testing
        config = {
            "in_channels": 4,
            "hidden_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "caption_channels": 256,
            "patch_size": 2,
        }

        model = DiT360Model(config)
        scheduler = FlowMatchScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(10, device='cpu')

        print("  [OK] Model and scheduler initialized")

        # Simulate one sampling step
        latent = torch.randn(1, 4, 16, 32)
        timestep = scheduler.timesteps[0]
        context = torch.randn(1, 77, config['caption_channels'])

        # Forward pass
        noise_pred = model(latent, timestep.unsqueeze(0), context)

        print(f"  [OK] Forward pass successful: {noise_pred.shape}")

        # Scheduler step
        prev_latent = scheduler.step(noise_pred, timestep.item(), latent)

        print(f"  [OK] Scheduler step successful: {prev_latent.shape}")

        print("[PASS] Integration forward pass test")
        return True

    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 3 validation tests"""
    print("\n" + "="*60)
    print("Running ComfyUI-DiT360 Phase 3 Validation Tests")
    print("="*60)

    tests = [
        test_dit360_model_architecture,
        test_flow_matching_scheduler,
        test_rope_embeddings,
        test_adaptive_layer_norm,
        test_circular_padding_attention,
        test_vae_wrapper,
        test_text_encoder_wrapper,
        test_text_preprocessing,
        test_integration_forward_pass,
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
