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
# Choose `"points"` for independent random uv samples, or `"arcs"` for
# Earth-rotation-like tracks. In arc mode, `N_UV_ARCS` is the only workshop knob:
# the helper below chooses realistic track lengths, curvature, and sampling.
UV_COVERAGE_MODE = "points"
N_UV_POINTS = 96
N_UV_ARCS = 6


def make_point_uv_coverage(n_visibilities=N_UV_POINTS, uv_radius=18.0, seed=3):
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


def make_arc_uv_coverage(n_arcs):
    generator = torch.Generator().manual_seed(3)

    # These internal choices keep the arc geometry realistic without adding more
    # user-facing controls. The formula is a simplified Earth-rotation synthesis
    # projection for a source at a low northern declination.
    samples_per_arc = 16
    uv_radius = 18.0
    source_declination = torch.tensor(0.22)
    hour_angle_offsets = torch.linspace(-0.5, 0.5, samples_per_arc)

    tracks = []
    for _ in range(n_arcs):
        baseline_length = uv_radius * (0.35 + 0.65 * torch.rand((), generator=generator))
        baseline_angle = 2.0 * pi * torch.rand((), generator=generator)
        vertical_fraction = 0.12 * (2.0 * torch.rand((), generator=generator) - 1.0)

        baseline_east = baseline_length * torch.cos(baseline_angle)
        baseline_north = baseline_length * torch.sin(baseline_angle)
        baseline_up = baseline_length * vertical_fraction

        centre_hour_angle = 2.0 * pi * torch.rand((), generator=generator)
        hour_angle_span = 0.45 + 0.35 * torch.rand((), generator=generator)
        hour_angle = centre_hour_angle + hour_angle_span * hour_angle_offsets

        # Project one physical baseline through a short hour-angle range. The
        # sine/cosine terms naturally produce curved tracks across the uv plane.
        u = baseline_east * torch.sin(hour_angle) + baseline_north * torch.cos(hour_angle)
        v = (
            -baseline_east * torch.sin(source_declination) * torch.cos(hour_angle)
            + baseline_north * torch.sin(source_declination) * torch.sin(hour_angle)
            + baseline_up * torch.cos(source_declination)
        )
        tracks.append(torch.stack([u, v], dim=1))

    arc_points = torch.cat(tracks, dim=0)

    # For real sky brightness, V(-u, -v) is the complex conjugate of V(u, v).
    # Include both halves in the simulated observation rather than only drawing
    # the reflected points in the plot.
    return torch.cat([arc_points, -arc_points], dim=0)


if UV_COVERAGE_MODE == "points":
    uv = make_point_uv_coverage()
elif UV_COVERAGE_MODE == "arcs":
    uv = make_arc_uv_coverage(n_arcs=N_UV_ARCS)
else:
    raise ValueError('UV_COVERAGE_MODE must be either "points" or "arcs"')

fig, ax = plt.subplots(figsize=(4, 4))
if UV_COVERAGE_MODE == "arcs":
    samples_per_arc = uv.shape[0] // (2 * N_UV_ARCS)
    tracks = uv[: N_UV_ARCS * samples_per_arc].reshape(N_UV_ARCS, samples_per_arc, 2)
    conjugate_tracks = uv[N_UV_ARCS * samples_per_arc :].reshape(N_UV_ARCS, samples_per_arc, 2)

    for track, conjugate_track in zip(tracks, conjugate_tracks):
        ax.plot(track[:, 0], track[:, 1], color="tab:blue", lw=1.2)
        ax.scatter(track[:, 0], track[:, 1], color="tab:blue", s=14)
        ax.plot(conjugate_track[:, 0], conjugate_track[:, 1], color="tab:orange", lw=1.2, alpha=0.6)
        ax.scatter(conjugate_track[:, 0], conjugate_track[:, 1], color="tab:orange", s=14, alpha=0.6)
else:
    ax.scatter(uv[:, 0], uv[:, 1], s=18)

    # Real sky brightness has conjugate-symmetric Fourier samples, so plotting
    # the reflected points gives a more interferometer-like visual.
    ax.scatter(-uv[:, 0], -uv[:, 1], s=18, alpha=0.35)

ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_title("Toy sparse uv coverage")
ax.set_aspect("auto")
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
VLBI_HIGH = torch.tensor([0.72, 0.120, 0.99, 2.0 * pi])

# Number of pixels along each image axis. Try `24`, `32`, or `48` in the
# workshop. Larger values make smoother images but each simulation is slower,
# because every visibility sums over `N_PIX ** 2` pixels.
N_PIX = 128

def _image_grid(n_pix=None, fov=1.0):
    n_pix = N_PIX if n_pix is None else n_pix

    # Image coordinates span the field of view. `meshgrid` returns 2D x/y arrays
    # that we flatten so matrix operations can treat each pixel as one column.
    axis = torch.linspace(-0.5 * fov, 0.5 * fov, n_pix)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def ring_image(theta, n_pix=None, fov=1.0):
    n_pix = N_PIX if n_pix is None else n_pix

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


def observe_ring(theta, uv_points, n_pix=None, fov=1.0, noise_std=0.015):
    n_pix = N_PIX if n_pix is None else n_pix

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
true_theta = torch.tensor([0.52, 0.065, 0.77, 2.35])
x_obs = observe_ring(true_theta, uv, n_pix=N_PIX)
truth_image = ring_image(true_theta, n_pix=N_PIX)

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
NUM_SIMULATIONS = 5000

# The prior is rectangular: every parameter is sampled independently between
# the corresponding entries in VLBI_LOW and VLBI_HIGH.
prior = BoxUniform(low=VLBI_LOW, high=VLBI_HIGH)
theta_train = prior.sample((NUM_SIMULATIONS,))

# Vectorised simulator call: all images and visibilities are generated in one
# batch, which is much faster than a Python loop.
x_train = observe_ring(theta_train, uv, n_pix=N_PIX)

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
# Render every posterior sample once. This gives us both individual image draws
# and pixel-wise summaries of the whole posterior image ensemble.
posterior_images = ring_image(samples, n_pix=N_PIX)

# Choose a handful of posterior samples and render their implied images.
draw_ids = torch.randperm(samples.shape[0])[:8]
image_draws = posterior_images[draw_ids]

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for ax, image in zip(axes.ravel(), image_draws):
    ax.imshow(image, origin="lower", cmap="afmhot")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Posterior image draws")
fig.tight_layout()
plt.show()

# Pixel-wise posterior summaries. `median(dim=0)` collapses the sample axis and
# keeps the N_PIX x N_PIX image geometry.
median_image = posterior_images.median(dim=0).values

# A posterior spread map is more useful than the Monte Carlo standard error here:
# it shows where the inferred image itself is uncertain. The 16th-to-84th
# percentile interval is the central 68% credible interval for each pixel.
q16, q84 = torch.quantile(
    posterior_images,
    torch.tensor([0.16, 0.84], device=posterior_images.device),
    dim=0,
)
fractional_credible_half_width = 0.5 * (q84 - q16) / median_image.clamp_min(1e-8)

# Only show the fractional uncertainty where the median image is meaningfully
# bright. Outside the ring, the median is nearly zero, so any fractional ratio is
# visually dominated by division noise rather than image uncertainty.
bright_pixels = median_image > 0.02 * median_image.max()
masked_fractional_credible_half_width = fractional_credible_half_width.clone()
masked_fractional_credible_half_width[~bright_pixels] = torch.nan
credible_vmax = torch.quantile(
    masked_fractional_credible_half_width[bright_pixels],
    0.98,
).item()

uncertainty_cmap = plt.get_cmap("magma").copy()
uncertainty_cmap.set_bad(color="0.92")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
summary_panels = [
    (truth_image, "True image", "afmhot", None),
    (median_image, "Posterior median", "afmhot", None),
    (
        masked_fractional_credible_half_width,
        "68% half-width / median",
        uncertainty_cmap,
        credible_vmax,
    ),
]
for ax, (image, title, cmap, vmax) in zip(axes, summary_panels):
    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Suggested Exercises
#
# 1. Reduce `N_UV_POINTS` from `96` to `32` and retrain.
# 2. Increase `noise_std` in `observe_ring`.
# 3. Fix the asymmetry to zero in the simulator. Which posterior dimensions disappear?
# 4. Set `UV_COVERAGE_MODE = "arcs"` and change `N_UV_ARCS`.
# 5. Compare posterior image draws to a regularised maximum likelihood image.
