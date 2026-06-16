# %% [markdown]
# # Optional Extension: Sparse VLBI Ring Inference With SBI
#
# This notebook is a toy model inspired by Event Horizon Telescope style imaging.
# It does **not** reproduce the M87 pipeline. The goal is smaller and more useful
# for an SBI workshop: infer simple ring-image parameters from sparse noisy
# Fourier measurements.
#
# We simulate:
#
# ```text
# ring parameters -> image -> sparse complex visibilities -> noisy data
# ```
#
# Then we infer:
#
# - ring diameter
# - ring width
# - azimuthal brightness asymmetry
# - asymmetry angle

# %%
from pathlib import Path
from math import pi
import sys

# Allow imports from the local `src/` directory whether JupyterLab was launched
# from the repository root or from inside `notebooks/`.
ROOT = Path.cwd()
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))
elif (ROOT.parent / "src").exists():
    sys.path.insert(0, str(ROOT.parent / "src"))

import matplotlib.pyplot as plt
import torch

from sbi.inference import NPE

# BoxUniform gives a rectangular prior over the ring parameters.
from sbi.utils import BoxUniform

from gw_sbi_demo.plotting import plot_corner

# Fixed seeds keep live workshop runs reproducible.
torch.manual_seed(11)
plt.rcParams.update({"figure.dpi": 120})

# %% [markdown]
# ## 1. Define a Sparse Interferometer
#
# A radio interferometer samples Fourier components of the sky brightness. Sparse
# coverage means the image is not directly observed.

# %%
def make_uv_coverage(n_visibilities=96, uv_radius=18.0, seed=3):
    # `torch.Generator` keeps this function's randomness local and repeatable.
    generator = torch.Generator().manual_seed(seed)

    # The square root makes points uniformly distributed over a disc rather than
    # over-representing the centre.
    radius = uv_radius * torch.sqrt(torch.rand(n_visibilities, generator=generator))
    angle = 2.0 * pi * torch.rand(n_visibilities, generator=generator)

    # Convert polar uv coordinates into Cartesian baseline coordinates.
    u = radius * torch.cos(angle)
    v = radius * torch.sin(angle)
    return torch.stack([u, v], dim=1)


uv = make_uv_coverage(n_visibilities=96)

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(uv[:, 0], uv[:, 1], s=18)

# Real sky brightness has conjugate-symmetric Fourier samples, so plotting the
# reflected points gives a more interferometer-like visual.
ax.scatter(-uv[:, 0], -uv[:, 1], s=18, alpha=0.35)
ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_title("Toy sparse uv coverage")
ax.set_aspect("equal")
plt.show()

# %% [markdown]
# ## 2. Build the Ring Simulator
#
# Parameters are ordered as:
#
# ```text
# theta = (diameter, width, asymmetry, angle)
# ```

# %%
VLBI_PARAMETER_NAMES = ("diameter", "width", "asymmetry", "angle")
VLBI_LOW = torch.tensor([0.34, 0.035, 0.00, 0.0])
VLBI_HIGH = torch.tensor([0.72, 0.120, 0.70, 2.0 * pi])


def _image_grid(n_pix=32, fov=1.0):
    # Image coordinates span the field of view. `meshgrid` returns 2D x/y arrays
    # that we flatten so matrix operations can treat each pixel as one column.
    axis = torch.linspace(-0.5 * fov, 0.5 * fov, n_pix)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def ring_image(theta, n_pix=32, fov=1.0):
    # Accept either one parameter vector with shape (4,) or a batch with shape
    # (N, 4). The `single` flag lets us return the same style we received.
    theta = torch.as_tensor(theta, dtype=torch.float32)
    single = theta.ndim == 1
    if single:
        theta = theta.unsqueeze(0)

    # `.T` lets us unpack columns from the batch: one tensor per parameter.
    diameter, width, asymmetry, angle = theta.T
    x, y = _image_grid(n_pix=n_pix, fov=fov)
    radius = torch.sqrt(x.pow(2) + y.pow(2))
    phi = torch.atan2(y, x)

    ring_radius = 0.5 * diameter[:, None]

    # Broadcasting does most of the work here:
    # - radius[None, :] has one row shared across all simulated images.
    # - ring_radius and width have one value per simulated image.
    image = torch.exp(-0.5 * ((radius[None, :] - ring_radius) / width[:, None]).pow(2))

    # The cosine term brightens one side of the ring. `angle` sets where that
    # bright side points; `asymmetry` controls how strong the contrast is.
    image = image * (1.0 + asymmetry[:, None] * torch.cos(phi[None, :] - angle[:, None]))

    # Clamp numerical negatives and normalise total flux to one, so the inference
    # focuses on ring shape rather than arbitrary image scale.
    image = torch.clamp(image, min=0.0)
    image = image / image.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Restore image geometry for plotting and for later Fourier measurements.
    image = image.reshape(theta.shape[0], n_pix, n_pix)
    return image.squeeze(0) if single else image


def observe_ring(theta, uv_points, n_pix=32, fov=1.0, noise_std=0.015):
    # Same single-versus-batch handling as `ring_image`.
    theta = torch.as_tensor(theta, dtype=torch.float32)
    single = theta.ndim == 1
    if single:
        theta = theta.unsqueeze(0)

    # Flatten images so each row is one image and each column is one pixel.
    images = ring_image(theta, n_pix=n_pix, fov=fov).reshape(theta.shape[0], -1)
    x, y = _image_grid(n_pix=n_pix, fov=fov)

    # A visibility is a Fourier component of the image. This phase matrix says
    # how each pixel contributes to each uv point.
    phase = 2.0 * pi * (uv_points[:, 0, None] * x[None, :] + uv_points[:, 1, None] * y[None, :])

    # `@` is matrix multiplication. Multiplying by cos/sin implements the real
    # and imaginary parts of the discrete Fourier transform at our sparse points.
    real = images @ torch.cos(phase).T
    imag = -(images @ torch.sin(phase).T)

    # Concatenate real and imaginary parts into one data vector for SBI, then add
    # independent Gaussian measurement noise.
    visibilities = torch.cat([real, imag], dim=1)
    visibilities = visibilities + noise_std * torch.randn_like(visibilities)
    return visibilities.squeeze(0) if single else visibilities

# %% [markdown]
# Inspect one truth image and its sparse visibility amplitudes.

# %%
true_theta = torch.tensor([0.52, 0.065, 0.45, 2.35])
x_obs = observe_ring(true_theta, uv)
truth_image = ring_image(true_theta)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].imshow(truth_image, origin="lower", cmap="afmhot")
axes[0].set_title("True toy ring image")
axes[0].set_xticks([])
axes[0].set_yticks([])

n_vis = uv.shape[0]

# The first half of `x_obs` is the real part and the second half is the
# imaginary part, so sqrt(real^2 + imag^2) gives visibility amplitude.
amp = torch.sqrt(x_obs[:n_vis].pow(2) + x_obs[n_vis:].pow(2))
axes[1].scatter(torch.linalg.norm(uv, dim=1), amp, s=18)
axes[1].set_xlabel("uv radius")
axes[1].set_ylabel("visibility amplitude")
axes[1].set_title("Sparse noisy measurements")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train NPE On Sparse Measurements
#
# This example is a little more abstract than the chirp example, but the SBI
# pattern is exactly the same.

# %%
NUM_SIMULATIONS = 6000

# The prior is rectangular: every parameter is sampled independently between
# the corresponding entries in VLBI_LOW and VLBI_HIGH.
prior = BoxUniform(low=VLBI_LOW, high=VLBI_HIGH)
theta_train = prior.sample((NUM_SIMULATIONS,))

# Vectorised simulator call: all images and visibilities are generated in one
# batch, which is much faster than a Python loop.
x_train = observe_ring(theta_train, uv)

# Standardise each visibility feature using training simulations only. The same
# transform must be applied to the observation before posterior sampling.
x_mean = x_train.mean(dim=0)
x_std = x_train.std(dim=0)
x_train_z = (x_train - x_mean) / x_std.clamp_min(1e-6)
x_obs_z = (x_obs - x_mean) / x_std.clamp_min(1e-6)

inference = NPE(prior=prior)
density_estimator = inference.append_simulations(theta_train, x_train_z).train(
    training_batch_size=256,  # simulations per optimiser step
    max_num_epochs=80,       # hard training limit
    stop_after_epochs=10,    # early-stopping patience
)
posterior = inference.build_posterior(density_estimator)

# %% [markdown]
# ## 4. Infer Ring Parameters

# %%
# Draw ring-parameter samples from p(theta | sparse visibilities).
samples = posterior.sample((5000,), x=x_obs_z)

for name, truth, mean, std in zip(
    VLBI_PARAMETER_NAMES,
    true_theta,
    samples.mean(dim=0),
    samples.std(dim=0),
):
    print(f"{name:>10s}: truth={truth:6.3f}  posterior={mean:6.3f} +/- {std:6.3f}")

fig = plot_corner(samples, labels=VLBI_PARAMETER_NAMES, truths=true_theta)
plt.show()

# %% [markdown]
# ## 5. Posterior Image Ensemble
#
# Plotting image draws makes the parameter posterior feel concrete. Sparse data
# can constrain some image properties while leaving others broad.

# %%
# Choose a handful of posterior samples and render their implied images.
draw_ids = torch.randperm(samples.shape[0])[:8]
image_draws = ring_image(samples[draw_ids])

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for ax, image in zip(axes.ravel(), image_draws):
    ax.imshow(image, origin="lower", cmap="afmhot")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Posterior image draws")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Suggested Exercises
#
# 1. Reduce `n_visibilities` from `96` to `32` and retrain.
# 2. Increase `noise_std` in `observe_ring`.
# 3. Fix the asymmetry to zero in the simulator. Which posterior dimensions disappear?
# 4. Replace random uv coverage with a few radial tracks.
# 5. Compare posterior image draws to a regularised maximum likelihood image.
