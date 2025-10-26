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
        negative_model_output: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        eta: Optional[float] = None
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


class DDIMSchedulerLite:
    """
    Minimal DDIM scheduler implementation for inference-time experimentation.

    This intentionally keeps the interface similar to FlowMatchScheduler so the
    sampler can switch between them without large code changes.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        eta: float = 0.0
    ):
        self.num_train_timesteps = num_train_timesteps
        self.eta = eta

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            self.alphas_cumprod.new_tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """Set timesteps descending from training range."""
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device
        )
        self.timesteps = timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        guidance_scale: float = 1.0,
        negative_model_output: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        eta: Optional[float] = None
    ) -> torch.Tensor:
        if negative_model_output is not None and guidance_scale != 1.0:
            model_output = negative_model_output + guidance_scale * (model_output - negative_model_output)

        if isinstance(timestep, torch.Tensor):
            timestep_index = int(timestep[0].item()) if timestep.numel() == 1 else int(timestep.item())
        else:
            timestep_index = int(timestep)

        if step_index is None and self.timesteps is not None:
            # locate current timestep in the list
            matches = (self.timesteps == timestep_index).nonzero()
            if matches.numel() > 0:
                step_index = int(matches[0].item())

        if step_index is None:
            step_index = 0

        is_last = step_index == len(self.timesteps) - 1
        prev_timestep_index = 0 if is_last else int(self.timesteps[step_index + 1].item())

        alpha_prod_t = self.alphas_cumprod[timestep_index]
        alpha_prod_prev = self.alphas_cumprod_prev[prev_timestep_index]
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_alpha_prod_prev = torch.sqrt(alpha_prod_prev)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1.0 - alpha_prod_t)

        # Predict original sample from noise prediction
        pred_original_sample = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t

        # Compute variance
        eta = self.eta if eta is None else eta
        sigma = 0.0
        if eta > 0 and not is_last:
            var = (1.0 - alpha_prod_prev) / (1.0 - alpha_prod_t) * (1.0 - alpha_prod_t / alpha_prod_prev)
            sigma = eta * torch.sqrt(var)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prod_prev - sigma ** 2, min=0.0)) * model_output

        noise = sigma * torch.randn_like(sample) if sigma > 0 else 0.0

        prev_sample = sqrt_alpha_prod_prev * pred_original_sample + dir_xt + noise
        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(timesteps, torch.Tensor):
            t_indices = timesteps.to(dtype=torch.long).view(-1)
        else:
            t_indices = torch.tensor([int(timesteps)], device=original_samples.device, dtype=torch.long)

        alphas = self.alphas_cumprod.to(original_samples.device)
        sqrt_alphas = torch.sqrt(alphas[t_indices])
        sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas[t_indices])

        while sqrt_alphas.ndim < original_samples.ndim:
            sqrt_alphas = sqrt_alphas.view(-1, *([1] * (original_samples.ndim - 1)))
            sqrt_one_minus_alphas = sqrt_one_minus_alphas.view(-1, *([1] * (original_samples.ndim - 1)))

        return sqrt_alphas * original_samples + sqrt_one_minus_alphas * noise


def create_scheduler(
    scheduler_type: str,
    **kwargs
):
    scheduler_type = scheduler_type.lower()
    if scheduler_type in {"flow_match", "flow"}:
        return FlowMatchScheduler(
            num_train_timesteps=kwargs.get("num_train_timesteps", 1000),
            shift=kwargs.get("shift", 1.0),
            use_dynamic_shifting=kwargs.get("use_dynamic_shifting", False)
        )
    if scheduler_type == "ddim":
        return DDIMSchedulerLite(
            num_train_timesteps=kwargs.get("num_train_timesteps", 1000),
            beta_start=kwargs.get("beta_start", 0.00085),
            beta_end=kwargs.get("beta_end", 0.012),
            eta=kwargs.get("eta", 0.0)
        )
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")

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
