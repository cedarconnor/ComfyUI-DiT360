"""
Text Conditioning for DiT360 using T5-XXL Encoder

Handles text prompt encoding to conditioning embeddings for the diffusion model.
Uses T5-XXL (4.7B parameters) for high-quality text understanding.

Key Features:
- T5-XXL model integration
- Support for long prompts (up to 512 tokens)
- Positive and negative prompt encoding
- Classifier-free guidance (CFG) support
- Device management and caching
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional, Dict, List
import comfy.model_management as mm
from transformers import T5EncoderModel, T5Tokenizer
from huggingface_hub import snapshot_download


class T5TextEncoder:
    """
    T5-XXL Text Encoder for DiT360

    Encodes text prompts to embeddings for conditioning the diffusion model.

    Args:
        model: T5 encoder model
        tokenizer: T5 tokenizer
        dtype: Data type
        device: Computation device
        offload_device: Storage device
        max_length: Maximum token length
    """

    def __init__(
        self,
        model: T5EncoderModel,
        tokenizer: T5Tokenizer,
        dtype: torch.dtype,
        device: torch.device,
        offload_device: torch.device,
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.offload_device = offload_device
        self.max_length = max_length
        self.is_loaded = False

    def load_to_device(self):
        """Load text encoder to GPU"""
        if not self.is_loaded:
            print(f"Loading T5 encoder to {self.device}...")
            self.model.to(self.device)
            self.is_loaded = True

    def offload(self):
        """Offload text encoder to CPU"""
        if self.is_loaded:
            print(f"Offloading T5 encoder to {self.offload_device}...")
            self.model.to(self.offload_device)
            self.is_loaded = False
            mm.soft_empty_cache()

    def encode(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts to embeddings

        Args:
            prompts: Positive prompt(s)
            negative_prompts: Negative prompt(s) for CFG

        Returns:
            Dictionary with 'prompt_embeds' and 'negative_prompt_embeds'

        Example:
            >>> encoder = T5TextEncoder(...)
            >>> result = encoder.encode("A beautiful sunset over the ocean")
            >>> prompt_embeds = result['prompt_embeds']  # (1, seq_len, 4096)
        """
        self.load_to_device()

        # Ensure lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if negative_prompts is not None and isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Encode positive prompts
        print(f"Encoding {len(prompts)} prompt(s)...")
        prompt_embeds = self._encode_batch(prompts)

        # Encode negative prompts
        if negative_prompts is not None:
            print(f"Encoding {len(negative_prompts)} negative prompt(s)...")
            negative_embeds = self._encode_batch(negative_prompts)
        else:
            # Use empty prompt as negative
            print("Using empty negative prompt...")
            negative_embeds = self._encode_batch([""] * len(prompts))

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_embeds,
        }

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts

        Args:
            texts: List of text strings

        Returns:
            Embeddings tensor (B, seq_len, hidden_size)
        """
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Move to device
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        # Encode
        with torch.no_grad():
            # TODO Phase 3: Use actual T5 model
            # For now, create placeholder embeddings
            batch_size = len(texts)
            hidden_size = 4096  # T5-XXL hidden size

            # Placeholder embeddings
            embeddings = torch.randn(
                batch_size,
                self.max_length,
                hidden_size,
                device=self.device,
                dtype=self.dtype
            )

            print(f"  Created placeholder embeddings: {embeddings.shape}")

        return embeddings

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text prompt (clean, normalize)

        Args:
            text: Raw text prompt

        Returns:
            Processed text

        Example:
            >>> processed = encoder.preprocess_text("A   BEAUTIFUL    sunset")
            >>> print(processed)
            "A beautiful sunset"
        """
        # Remove extra whitespace
        text = " ".join(text.split())

        # Lowercase (T5 is case-insensitive)
        text = text.lower()

        return text


def download_t5_from_huggingface(
    repo_id: str = "google/t5-v1_1-xxl",
    save_dir: Path = None
) -> Path:
    """
    Download T5-XXL model from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository
        save_dir: Save directory

    Returns:
        Path to downloaded model directory
    """
    import folder_paths

    if save_dir is None:
        model_name = repo_id.split("/")[-1]
        save_dir = Path(folder_paths.models_dir) / "t5" / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "config.json").exists():
        print(f"T5 model already exists at: {save_dir}")
        return save_dir

    print(f"\n{'='*60}")
    print(f"Downloading T5 model from HuggingFace...")
    print(f"Repository: {repo_id}")
    print(f"Destination: {save_dir}")
    print(f"This is a large download (~5GB)...")
    print(f"{'='*60}\n")

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*tensorflow*"]
        )

        print(f"\n✓ T5 model downloaded: {downloaded_path}\n")
        return Path(downloaded_path)

    except Exception as e:
        raise RuntimeError(
            f"\nFailed to download T5 model from HuggingFace.\n\n"
            f"Error: {e}\n\n"
            f"Please download manually from:\n"
            f"  https://huggingface.co/{repo_id}\n\n"
            f"Or try: https://huggingface.co/city96/t5-v1_1-xxl-encoder-bf16 (optimized)\n\n"
            f"And place in:\n"
            f"  {save_dir}\n"
        )


def load_t5_encoder(
    model_path: Union[str, Path],
    precision: str = "fp16",
    device: Optional[torch.device] = None,
    offload_device: Optional[torch.device] = None,
    max_length: int = 512
) -> T5TextEncoder:
    """
    Load T5-XXL text encoder

    Args:
        model_path: Path to T5 model directory
        precision: Model precision (fp32/fp16/bf16)
        device: Target device
        offload_device: Offload device
        max_length: Maximum token length

    Returns:
        T5TextEncoder wrapper

    Example:
        >>> encoder = load_t5_encoder("models/t5/t5-v1_1-xxl", precision="fp16")
        >>> result = encoder.encode("A beautiful panorama")
    """
    model_path = Path(model_path)

    # Auto-detect devices
    if device is None:
        device = mm.get_torch_device()
    if offload_device is None:
        offload_device = mm.unet_offload_device()

    print(f"\n{'='*60}")
    print(f"Loading T5-XXL Text Encoder")
    print(f"{'='*60}")
    print(f"Path: {model_path}")
    print(f"Precision: {precision}")
    print(f"Max length: {max_length}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Check if path exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"T5 model not found: {model_path}\n\n"
            f"Please download T5-XXL from:\n"
            f"  https://huggingface.co/google/t5-v1_1-xxl\n"
            f"  Or (optimized): https://huggingface.co/city96/t5-v1_1-xxl-encoder-bf16\n\n"
            f"Or use auto-download in DiT360Loader.\n"
        )

    # Convert precision
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(precision, torch.float16)

    try:
        # TODO Phase 3: Load actual T5 model using transformers
        # For now, create placeholder

        print("Loading tokenizer...")
        # Placeholder tokenizer
        class PlaceholderTokenizer:
            def __call__(self, *args, **kwargs):
                class Tokens:
                    def __init__(self):
                        self.input_ids = torch.zeros(1, max_length, dtype=torch.long)
                        self.attention_mask = torch.ones(1, max_length, dtype=torch.long)
                return Tokens()

        tokenizer = PlaceholderTokenizer()
        print("✓ Placeholder tokenizer created")

        print("Loading T5 encoder model...")
        # Placeholder model
        class PlaceholderT5(nn.Module):
            def __init__(self):
                super().__init__()
                self.initialized = True

            def forward(self, *args, **kwargs):
                return None

        model = PlaceholderT5()
        print("✓ Placeholder T5 model created (actual T5 loading in Phase 3)")

        model = model.to(dtype=dtype, device=offload_device)
        model.eval()

    except Exception as e:
        raise RuntimeError(f"Failed to load T5 model: {e}")

    # Wrap encoder
    encoder = T5TextEncoder(
        model=model,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        offload_device=offload_device,
        max_length=max_length
    )

    print(f"✓ T5 encoder ready\n")
    return encoder


def text_preprocessing(text: str) -> str:
    """
    Preprocess text for T5 encoding

    Cleans and normalizes text for better encoding quality.

    Args:
        text: Raw text prompt

    Returns:
        Preprocessed text

    Example:
        >>> clean = text_preprocessing("  A    BEAUTIFUL  panorama!!! ")
        >>> print(clean)
        "a beautiful panorama"
    """
    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove excessive punctuation
    import re
    text = re.sub(r'([!?.])\1+', r'\1', text)

    # Lowercase
    text = text.lower().strip()

    return text
