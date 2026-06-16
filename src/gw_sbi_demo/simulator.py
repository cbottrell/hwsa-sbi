"""Pedagogical gravitational-wave-like chirp simulator for SBI demos.

The waveform here is intentionally lightweight. It is inspired by compact binary
chirps, but it is not a production waveform model. That is a feature for the
workshop: participants can inspect and modify the whole simulator in a few
minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional, Sequence, Tuple

import torch


DEFAULT_PARAMETER_NAMES: Tuple[str, ...] = (
    "chirp_mass",
    "amplitude",
    "merger_time",
    "phase",
)

DEFAULT_PRIOR_LOW = torch.tensor([18.0, 0.35, 0.58, 0.0], dtype=torch.float32)
DEFAULT_PRIOR_HIGH = torch.tensor([45.0, 2.00, 0.86, 2.0 * pi], dtype=torch.float32)


@dataclass(frozen=True)
class GWConfig:
    """Configuration for the toy chirp simulator."""

    n_time: int = 256
    duration: float = 1.0
    noise_std: float = 0.35
    noise_knee_hz: float = 35.0
    device: str = "cpu"


def time_grid(config: GWConfig = GWConfig()) -> torch.Tensor:
    """Return the time grid used by the simulator."""

    return torch.linspace(
        0.0,
        config.duration,
        config.n_time,
        dtype=torch.float32,
        device=config.device,
    )


def build_prior(
    low: torch.Tensor = DEFAULT_PRIOR_LOW,
    high: torch.Tensor = DEFAULT_PRIOR_HIGH,
    device: str = "cpu",
):
    """Build an `sbi` BoxUniform prior over the simulator parameters."""

    from sbi.utils import BoxUniform

    return BoxUniform(low=low.to(device), high=high.to(device))


def _as_batch(theta: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    theta = torch.as_tensor(theta, dtype=torch.float32)
    single = theta.ndim == 1
    if single:
        theta = theta.unsqueeze(0)
    if theta.shape[-1] != len(DEFAULT_PARAMETER_NAMES):
        raise ValueError(
            f"Expected theta with {len(DEFAULT_PARAMETER_NAMES)} parameters "
            f"({DEFAULT_PARAMETER_NAMES}), got shape {tuple(theta.shape)}."
        )
    return theta, single


def clean_chirp(theta: torch.Tensor, config: GWConfig = GWConfig()) -> torch.Tensor:
    """Generate a clean chirp-like strain time series.

    Parameters are ordered as:

    1. chirp mass, in arbitrary solar-mass-like units.
    2. amplitude, in arbitrary strain units.
    3. merger time, as a fraction of the 1-second segment.
    4. phase, in radians.
    """

    theta, single = _as_batch(theta)
    theta = theta.to(config.device)

    chirp_mass, amplitude, merger_time, phase0 = theta.T
    t = time_grid(config)
    dt = config.duration / (config.n_time - 1)

    tau = torch.clamp(merger_time[:, None] - t[None, :], min=dt)
    tau_start = torch.clamp(merger_time[:, None], min=dt)

    mass_scale = (chirp_mass[:, None] / 30.0).pow(5.0 / 8.0)
    frequency = 18.0 + 20.0 * mass_scale * tau.pow(-3.0 / 8.0)
    nyquist = 0.5 * config.n_time / config.duration
    frequency = torch.clamp(frequency, max=0.88 * nyquist)

    phase = phase0[:, None] + 2.0 * pi * torch.cumsum(frequency, dim=1) * dt

    growth = (tau / tau_start).pow(-0.25)
    onset = torch.sigmoid((t[None, :] - 0.08) / 0.025)
    inspiral = amplitude[:, None] * growth * onset * torch.sin(phase)

    post_time = torch.clamp(t[None, :] - merger_time[:, None], min=0.0)
    ring_frequency = 92.0 + 0.65 * (chirp_mass[:, None] - 30.0)
    ringdown = (
        2.2
        * amplitude[:, None]
        * torch.exp(-post_time / 0.055)
        * torch.sin(2.0 * pi * ring_frequency * post_time + phase0[:, None])
    )

    waveform = torch.where(t[None, :] <= merger_time[:, None], inspiral, ringdown)
    waveform = waveform - waveform.mean(dim=1, keepdim=True)

    return waveform.squeeze(0) if single else waveform


def colored_noise(
    batch_size: int,
    config: GWConfig = GWConfig(),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate simple colored Gaussian noise for strain segments."""

    white = torch.randn(
        batch_size,
        config.n_time,
        generator=generator,
        dtype=torch.float32,
        device=config.device,
    )
    dt = config.duration / (config.n_time - 1)
    freqs = torch.fft.rfftfreq(config.n_time, d=dt).to(config.device)
    spectrum = torch.fft.rfft(white, dim=1)

    color = 1.0 / torch.sqrt(1.0 + (freqs / config.noise_knee_hz).pow(2))
    noise = torch.fft.irfft(spectrum * color[None, :], n=config.n_time, dim=1)
    noise = noise - noise.mean(dim=1, keepdim=True)
    noise = noise / noise.std(dim=1, keepdim=True).clamp_min(1e-6)
    return config.noise_std * noise


def simulate(
    theta: torch.Tensor,
    config: GWConfig = GWConfig(),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Simulate noisy strain data for one or more parameter vectors."""

    theta, single = _as_batch(theta)
    waveform = clean_chirp(theta, config=config)
    noise = colored_noise(theta.shape[0], config=config, generator=generator)
    x = waveform + noise
    return x.squeeze(0) if single else x


def make_observation(
    theta: Sequence[float],
    config: GWConfig = GWConfig(),
    seed: int = 123,
) -> torch.Tensor:
    """Generate one reproducible noisy observation."""

    generator = torch.Generator(device=config.device).manual_seed(seed)
    return simulate(torch.tensor(theta, dtype=torch.float32), config, generator)


def standardize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Apply feature-wise standardization with a safe denominator."""

    return (x - mean) / std.clamp_min(1e-6)
