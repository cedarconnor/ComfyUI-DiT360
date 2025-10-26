"""Phase 5 optimization regression tests."""

import math
import sys
from pathlib import Path
import types

import torch

# Add repository root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

# Provide lightweight stubs when ComfyUI isn't available
if "comfy" not in sys.modules:
    comfy_stub = types.ModuleType("comfy")
    mm_stub = types.ModuleType("comfy.model_management")
    mm_stub.get_torch_device = lambda: torch.device("cpu")
    mm_stub.unet_offload_device = lambda: torch.device("cpu")
    mm_stub.soft_empty_cache = lambda: None

    utils_stub = types.ModuleType("comfy.utils")

    class _ProgressBar:
        def __init__(self, total):
            self.total = total

        def update(self, _):
            return None

    utils_stub.ProgressBar = _ProgressBar

    sys.modules["comfy"] = comfy_stub
    sys.modules["comfy.model_management"] = mm_stub
    sys.modules["comfy.utils"] = utils_stub

if "folder_paths" not in sys.modules:
    folder_stub = types.ModuleType("folder_paths")
    models_root = Path(__file__).parent.parent / "models"
    folder_stub.models_dir = str(models_root)
    folder_stub.get_filename_list = lambda *_args, **_kwargs: []
    folder_stub.get_folder_paths = lambda *_args, **_kwargs: [str(models_root)]
    folder_stub.get_full_path = lambda *_args, **_kwargs: ""
    folder_stub.add_model_folder_path = lambda *_args, **_kwargs: None
    sys.modules["folder_paths"] = folder_stub

from dit360.model import DiT360Model
from dit360.scheduler import create_scheduler
from dit360.vae import DiT360VAE


class _DummyPipeModel:
    def __init__(self):
        self.dtype = torch.float16
        self.device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.attention_backend = "eager"
        self.attention_slice_size = None
        self.quantization_mode = "none"

    def set_attention_options(self, backend=None, slice_size=None):
        if backend is not None:
            self.attention_backend = backend
        if slice_size is not None:
            self.attention_slice_size = slice_size


class _DummyPipeVAE:
    def __init__(self):
        self.tile_size = 1536
        self.tile_overlap = 128
        self.max_tile_pixels = 16_777_216

    def configure_tiling(self, tile_size=None, tile_overlap=None, max_tile_pixels=None):
        if tile_size is not None:
            self.tile_size = tile_size
        if tile_overlap is not None:
            self.tile_overlap = tile_overlap
        if max_tile_pixels is not None:
            self.max_tile_pixels = max_tile_pixels


class _DummyVAE(torch.nn.Module):
    """Lightweight VAE stub for tiling tests."""

    def encode(self, x):
        class _Output:
            def __init__(self, latents):
                self.latent_dist = type(
                    "_Latent", (), {"sample": staticmethod(lambda latents=latents: latents)}
                )()

        latents = torch.nn.functional.avg_pool2d(x, kernel_size=8)
        return _Output(latents)

    def decode(self, z):
        class _Output:
            def __init__(self, sample):
                self.sample = sample

        upsampled = torch.nn.functional.interpolate(
            z,
            scale_factor=8,
            mode="nearest",
        )
        upsampled = upsampled.clamp(0.0, 1.0)
        return _Output(upsampled)


def test_scheduler_factory_ddim_step():
    scheduler = create_scheduler("ddim", num_train_timesteps=50, eta=0.0)
    scheduler.set_timesteps(10)
    sample = torch.zeros(1, 4, 8, 8)
    noise = torch.randn_like(sample)
    timestep = scheduler.timesteps[0]
    noised = scheduler.add_noise(sample, noise, timestep)
    assert torch.allclose(noised, noise * math.sqrt(1 - scheduler.alphas_cumprod[int(timestep)]), atol=1e-5)
    result = scheduler.step(noise, timestep, sample, step_index=0)
    assert result.shape == sample.shape


def test_dit360_model_attention_override():
    config = {
        "in_channels": 4,
        "hidden_size": 64,
        "num_layers": 1,
        "num_heads": 4,
        "caption_channels": 16,
        "patch_size": 2,
    }
    model = DiT360Model(config, enable_circular_padding=False, attention_backend="eager")
    model.set_attention_options(backend="auto", slice_size=4)
    assert model.attention_backend in {"eager", "xformers", "flash"}
    assert model.attention_slice_size == 4
    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([0.5])
    context = torch.randn(1, 8, 16)
    out = model(x, t, context)
    assert out.shape == x.shape


def test_vae_tiling_configuration():
    dummy = _DummyVAE()
    vae = DiT360VAE(
        vae_model=dummy,
        dtype=torch.float32,
        device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        scale_factor=8,
        tile_size=64,
        tile_overlap=16,
        max_tile_pixels=64 * 64,
    )
    vae.configure_tiling(tile_size=128, tile_overlap=8, max_tile_pixels=128 * 128)
    assert vae.tile_size == 128
    assert vae.tile_overlap == 8
    assert vae.max_tile_pixels == 128 * 128
    image = torch.rand(1, 1024, 2048, 3)
    latent = vae.encode(image, use_tiling=True)
    decoded = vae.decode(latent, use_tiling=True)
    assert decoded.shape == image.shape


def test_pipeline_breakout_and_combine_roundtrip():
    from nodes import DiT360PipeBreakout, DiT360PipeCombine
    from types import SimpleNamespace

    dummy_model = _DummyPipeModel()
    dummy_vae = _DummyPipeVAE()
    dummy_text_encoder = SimpleNamespace(name="dummy")

    base_pipe = {
        "model": dummy_model,
        "vae": dummy_vae,
        "text_encoder": dummy_text_encoder,
        "dtype": torch.float16,
        "device": torch.device("cpu"),
        "offload_device": torch.device("cpu"),
        "attention_backend": "eager",
        "attention_slice_size": None,
        "quantization_mode": "none",
        "vae_tile_size": dummy_vae.tile_size,
        "vae_tile_overlap": dummy_vae.tile_overlap,
        "vae_auto_tile_pixels": dummy_vae.max_tile_pixels,
    }

    breakout_node = DiT360PipeBreakout()
    model_out, vae_out, text_out, passthrough = breakout_node.breakout(base_pipe)

    assert model_out is dummy_model
    assert vae_out is dummy_vae
    assert text_out is dummy_text_encoder
    assert passthrough is not base_pipe

    combine_node = DiT360PipeCombine()
    combined_pipe, = combine_node.combine(
        base_pipe=passthrough,
        model=model_out,
        vae=vae_out,
        text_encoder=text_out,
        attention_backend_override="xformers",
        attention_slice_override=32,
        quantization_override="int8",
        vae_tile_size_override=1024,
        vae_tile_overlap_override=64,
        vae_auto_tile_override=8_388_608,
    )

    assert combined_pipe["model"] is dummy_model
    assert dummy_model.attention_backend == "xformers"
    assert combined_pipe["attention_backend"] == "xformers"
    assert combined_pipe["attention_slice_size"] == 32
    assert combined_pipe["quantization_mode"] == "int8"
    assert combined_pipe["vae_tile_size"] == 1024
    assert combined_pipe["vae_tile_overlap"] == 64
    assert combined_pipe["vae_auto_tile_pixels"] == 8_388_608
    assert combined_pipe is not base_pipe
