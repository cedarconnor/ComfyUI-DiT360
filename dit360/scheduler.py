"""
Flow Matching Scheduler for DiT360

Implements the flow matching (rectified flow) sampling algorithm used by FLUX.1-dev.
This is a simpler and more efficient alternative to traditional DDPM/DDIM schedulers.

Key Concepts:
- Linear interpolation between noise and data
- Optimal transport path for efficient sampling
- Euler method for numerical integration
- Compatible with classifier-free guidance (CFG)
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple
import numpy as np


class FlowMatchScheduler:
    """
    Flow Matching Scheduler (Rectified Flow)

    This scheduler implements the sampling process for flow matching models.
    Unlike traditional diffusion models, flow matching uses a straight-line
    path between noise and data, making it more efficient.

    The key equation is:
        x(t) = t * x_data + (1 - t) * x_noise

    Where t goes from 0 (pure noise) to 1 (pure data).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False
    ):
        """
        Initialize Flow Matching Scheduler

        Args:
            num_train_timesteps: Number of timesteps during training
            shift: Shift parameter for timestep schedule (higher = more noise earlier)
            use_dynamic_shifting: Use dynamic shifting based on resolution
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting

        # Will be set when calling set_timesteps()
        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """
        Set the discrete timesteps used for inference

        Args:
            num_inference_steps: Number of steps to use for sampling
            device: Device to place timesteps on
        """
        self.num_inference_steps = num_inference_steps

        # Create linearly spaced timesteps from 1 to 0
        # We reverse so that we go from noise (t=1) to data (t=0)
        timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

        # Apply shifting if enabled
        if self.shift != 1.0:
            # Shift the timestep schedule to control noise distribution
            # Higher shift = more time spent in high noise regime
            timesteps = self._apply_shift(timesteps, self.shift)

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.float32)

    def _apply_shift(self, timesteps: np.ndarray, shift: float) -> np.ndarray:
        """
        Apply shifting to timestep schedule

        Args:
            timesteps: Original timesteps
            shift: Shift parameter

        Returns:
            Shifted timesteps
        """
        # Apply power law shifting
        return timesteps ** shift

    def scale_noise(self, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Scale the input sample based on timestep

        In flow matching, this is simply:
            x(t) = t * x_data + (1 - t) * x_noise

        Args:
            sample: Input sample
            timesteps: Current timesteps

        Returns:
            Scaled sample
        """
        # For flow matching, the noise schedule is linear
        # This is a no-op during inference (used during training)
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        guidance_scale: float = 1.0,
        negative_model_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform one step of the sampling process using Euler method

        Args:
            model_output: Predicted velocity from the model
            timestep: Current timestep (float in [0, 1])
            sample: Current sample x(t)
            guidance_scale: Classifier-free guidance scale
            negative_model_output: Model output for negative prompt (for CFG)

        Returns:
            Updated sample x(t - dt)
        """
        # Apply classifier-free guidance if negative output provided
        if negative_model_output is not None and guidance_scale != 1.0:
            # CFG: output = uncond + scale * (cond - uncond)
            model_output = negative_model_output + guidance_scale * (model_output - negative_model_output)

        # Calculate step size (dt)
        # For flow matching, we're integrating dx/dt = v(x, t)
        # where v is the velocity field predicted by the model
        dt = 1.0 / self.num_inference_steps

        # Euler method: x(t - dt) = x(t) - v(x, t) * dt
        prev_sample = sample - model_output * dt

        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples for training

        For flow matching:
            x(t) = t * x_data + (1 - t) * x_noise

        Args:
            original_samples: Original clean samples
            noise: Noise to add
            timesteps: Timesteps (values in [0, 1])

        Returns:
            Noised samples
        """
        # Ensure timesteps is the right shape for broadcasting
        timesteps = timesteps.to(original_samples.device)

        # Reshape timesteps for broadcasting: (B,) -> (B, 1, 1, 1)
        while len(timesteps.shape) < len(original_samples.shape):
            timesteps = timesteps.unsqueeze(-1)

        # Linear interpolation between data and noise
        noisy_samples = timesteps * original_samples + (1.0 - timesteps) * noise

        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the target velocity for training

        For flow matching, the velocity is:
            v(x, t) = x_data - x_noise

        This is constant and doesn't depend on t for the straight-line path.

        Args:
            sample: Clean data sample
            noise: Noise sample
            timesteps: Timesteps (not used for straight-line flow)

        Returns:
            Target velocity
        """
        # For rectified flow (straight-line path), velocity is constant
        velocity = sample - noise

        return velocity


# ============================================================================
# Advanced Scheduler with CFG Support
# ============================================================================

class CFGFlowMatchScheduler(FlowMatchScheduler):
    """
    Flow Matching Scheduler with built-in Classifier-Free Guidance support

    Extends the base scheduler to handle CFG during sampling more efficiently.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        guidance_scale: float = 3.0
    ):
        super().__init__(num_train_timesteps, shift, use_dynamic_shifting)
        self.guidance_scale = guidance_scale

    def step_with_cfg(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        timestep: float,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Perform one step with built-in CFG

        Args:
            model: The DiT360 model
            sample: Current sample
            timestep: Current timestep
            prompt_embeds: Positive prompt embeddings
            negative_prompt_embeds: Negative prompt embeddings
            guidance_scale: Override the default guidance scale

        Returns:
            Updated sample
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # Prepare timestep tensor
        t = torch.tensor([timestep], device=sample.device)

        if guidance_scale == 1.0 or negative_prompt_embeds is None:
            # No guidance, single forward pass
            model_output = model(sample, t, prompt_embeds)
        else:
            # CFG: Run model twice (conditional and unconditional)
            # Concatenate samples for batched inference
            sample_combined = torch.cat([sample, sample], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            embeds_combined = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

            # Single batched forward pass
            output_combined = model(sample_combined, t_combined, embeds_combined)

            # Split outputs
            cond_output, uncond_output = output_combined.chunk(2, dim=0)

            # Apply CFG
            model_output = uncond_output + guidance_scale * (cond_output - uncond_output)

        # Euler step
        dt = 1.0 / self.num_inference_steps
        prev_sample = sample - model_output * dt

        return prev_sample


# ============================================================================
# Utility Functions
# ============================================================================

def get_timestep_schedule(
    num_steps: int,
    method: str = "linear",
    **kwargs
) -> torch.Tensor:
    """
    Get various timestep schedules

    Args:
        num_steps: Number of sampling steps
        method: Schedule type - "linear", "quadratic", "cosine"
        **kwargs: Additional parameters for specific schedules

    Returns:
        Timesteps tensor
    """
    if method == "linear":
        # Linear schedule from 1 to 0
        return torch.linspace(1.0, 0.0, num_steps + 1)[:-1]

    elif method == "quadratic":
        # Quadratic schedule (spend more time in high noise)
        t = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
        return t ** 2

    elif method == "cosine":
        # Cosine schedule
        steps = torch.arange(num_steps)
        t = 1.0 - (steps / num_steps)
        return ((torch.cos(t * np.pi) + 1.0) / 2.0)

    else:
        raise ValueError(f"Unknown schedule method: {method}")


def compute_snr(timesteps: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio for given timesteps

    Args:
        timesteps: Timesteps in [0, 1]
        shift: Shift parameter

    Returns:
        SNR values
    """
    # For flow matching with linear interpolation:
    # SNR(t) = (t / (1 - t))^2
    t = timesteps.clamp(min=1e-8, max=1.0 - 1e-8)
    snr = (t / (1.0 - t)) ** (2.0 * shift)
    return snr
