# %% [markdown]
# # Recovering a Noisy Gravitational-Wave-Like Chirp With SBI
#
# This notebook is the main worked example for the workshop.
#
# We will:
#
# 1. Define a prior over source parameters.
# 2. Simulate noisy strain data from those parameters.
# 3. Train neural posterior estimation with `sbi`.
# 4. Infer the parameters of one observed noisy signal.
# 5. Check the result with posterior predictive waveforms.
#
# The waveform is deliberately lightweight and pedagogical. It is inspired by
# compact binary chirps, but it is not a production LIGO/Virgo/KAGRA waveform
# model.

# %%
from pathlib import Path
import sys

# Jupyter may be launched either from the repository root or from `notebooks/`.
# Adding `src/` to the front of `sys.path` lets this notebook import the local
# workshop package without requiring `pip install -e .`.
ROOT = Path.cwd()
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))
elif (ROOT.parent / "src").exists():
    sys.path.insert(0, str(ROOT.parent / "src"))

import matplotlib.pyplot as plt
import torch

# NPE = Neural Posterior Estimation: learn p(theta | x) from simulations.
from sbi.inference import NPE

from gw_sbi_demo import (
    DEFAULT_PARAMETER_NAMES,
    GWConfig,
    build_prior,
    clean_chirp,
    make_observation,
    simulate,
    standardize,
    time_grid,
)
from gw_sbi_demo.plotting import plot_corner, plot_posterior_predictive, plot_signal

# Fixing the seed makes the workshop reproducible: students should see the same
# random training examples and posterior plots when they rerun the notebook.
torch.manual_seed(7)

# Increase inline figure resolution without changing the underlying data.
plt.rcParams.update({"figure.dpi": 120})

# %% [markdown]
# ## 1. Configure the Simulator
#
# The simulator maps source parameters to a noisy time series:
#
# ```text
# theta = (chirp_mass, amplitude, merger_time, phase)
# theta -> clean chirp -> detector-like coloured noise -> observed strain
# ```
#
# The next cell will configure the format of the data (e.g. time series sampling
# rate, duration, noise properties) and also the type of device to be used to
# generate samples.
#
# Larger `noise_std` makes the inverse problem harder. Return to this later and
# modify the "observation" to make inference harder or easier. This
# "observation" is characterised by a true `theta`, the unknown ground truth to
# be inferred.

# %%
config = GWConfig(
    n_time=256,       # number of samples in each strain time series
    duration=1.0,     # seconds covered by the synthetic detector segment
    noise_std=0.35,   # larger values make the recovery problem harder
    noise_knee_hz=35.0,  # sets where the coloured-noise spectrum turns over
    device="cpu",     # keep the demo laptop-friendly; use "cuda" if available
)

# `time` is used only for plotting; the simulator itself knows the same config.
time = time_grid(config)

# The prior is a BoxUniform over (chirp_mass, amplitude, merger_time, phase).
# SBI will learn only within this parameter range, so prior limits matter.
prior = build_prior(device=config.device)

# One known source used to generate a synthetic "observed" event. In real use,
# this would be the unknown event we are trying to infer.
true_theta = torch.tensor([32.0, 1.15, 0.74, 1.25])

# `seed` fixes just this observation's noise realisation, separate from the
# training simulations below.
x_obs = make_observation(true_theta, config=config, seed=42)

# The clean signal is not available in real data; we keep it here for teaching
# overlays so students can see what the noisy observation is hiding.
x_clean = clean_chirp(true_theta, config=config)

ax = plot_signal(time, x_obs, x_clean, title="Synthetic observed strain")
plt.show()

# %% [markdown]
# ## 2. Draw Training Simulations
#
# SBI learns from examples. We sample parameters from the prior, run the
# simulator, and train a neural density estimator to approximate
# `p(theta | x)`.
#
# For a live session, start with `NUM_SIMULATIONS = 3000`. For a cleaner result,
# use `8000` or more.

# %%
NUM_SIMULATIONS = 5000

# `prior.sample((N,))` returns an N x 4 tensor: one row per simulated source.
theta_train = prior.sample((NUM_SIMULATIONS,))

# A separate generator makes the simulated noise reproducible without changing
# other random draws, such as posterior samples later in the notebook.
noise_generator = torch.Generator(device=config.device).manual_seed(2026)

# `simulate` is vectorised: it accepts all parameter rows and returns an
# N x n_time matrix of noisy strain segments.
x_train = simulate(theta_train, config=config, generator=noise_generator)

# Neural density estimators train more stably when input features have similar
# scales. We standardise each time bin using the training simulations only.
x_mean = x_train.mean(dim=0)
x_std = x_train.std(dim=0)
x_train_z = standardize(x_train, x_mean, x_std)

# The observed event must be transformed with the same training mean/std. Do not
# recompute statistics from `x_obs`, because that would leak observation-specific
# information into the preprocessing.
x_obs_z = standardize(x_obs, x_mean, x_std)

print(f"theta_train shape: {tuple(theta_train.shape)}")
print(f"x_train shape:     {tuple(x_train.shape)}")

# %% [markdown]
# Let us inspect a few random simulations. This is a useful teaching pause:
# students can connect parameters to observable waveform structure before the
# neural network appears.

# %%
fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True)

# `torch.randperm(NUM_SIMULATIONS)[:4]` chooses four unique random rows without
# replacement, so every plotted example is a different simulated event.
for ax, idx in zip(axes, torch.randperm(NUM_SIMULATIONS)[:4]):
    ax.plot(time, x_train[idx], color="0.25", lw=1.0)

    # `zip(names, values)` pairs each parameter value with its label; the
    # generator expression keeps the label formatting compact.
    label = ", ".join(
        f"{name}={value:.2f}"
        for name, value in zip(DEFAULT_PARAMETER_NAMES, theta_train[idx])
    )
    ax.text(0.01, 0.88, label, transform=ax.transAxes, fontsize=8)
    ax.set_ylabel("strain")
axes[-1].set_xlabel("time [s]")
fig.suptitle("Random simulator draws")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train Neural Posterior Estimation
#
# `NPE` learns a conditional density estimator for `p(theta | x)`. Once trained,
# it can be reused for any observation generated under the same prior and
# simulator setup.

# %%
inference = NPE(prior=prior)

# `append_simulations(theta, x)` gives SBI paired examples of causes and
# observations. `theta_train` is unstandardised because the prior is defined in
# physical parameter units; only the data vector `x` was standardised.
density_estimator = inference.append_simulations(theta_train, x_train_z).train(
    training_batch_size=256,  # simulations per optimiser step
    max_num_epochs=80,       # hard upper limit; training may stop earlier
    stop_after_epochs=10,    # early stopping patience after validation stalls
)

# The trained density estimator becomes a reusable posterior object. We can now
# condition it on any standardised observation from the same simulator/prior.
posterior = inference.build_posterior(density_estimator)

# %% [markdown]
# ## 4. Condition On One Observation
#
# We now ask for samples from:
#
# ```text
# p(theta | x_obs)
# ```
#
# This is the central SBI object. It is not one best fit; it is a distribution
# over all parameter values compatible with the noisy signal and the prior.

# %%
# The keyword `x=` is the observation we condition on. The shape `(5000,)` asks
# for 5000 independent draws from p(theta | x_obs).
posterior_samples = posterior.sample((5000,), x=x_obs_z)

# Means and standard deviations are summaries only; the full sample cloud is the
# posterior approximation we care about.
posterior_mean = posterior_samples.mean(dim=0)
posterior_std = posterior_samples.std(dim=0)

for name, truth, mean, std in zip(
    DEFAULT_PARAMETER_NAMES,
    true_theta,
    posterior_mean,
    posterior_std,
):
    print(f"{name:>12s}: truth={truth:6.3f}  posterior={mean:6.3f} +/- {std:6.3f}")

# %%
fig = plot_corner(
    posterior_samples,
    labels=DEFAULT_PARAMETER_NAMES,
    truths=true_theta,
)
plt.show()

# %% [markdown]
# ## 5. Posterior Predictive Check
#
# A posterior is only useful if it can explain the observation. We draw parameter
# samples from the posterior, generate the implied clean chirps, and overlay them
# on the noisy observation.
#
# If the observed signal lies outside this predictive cloud, the posterior may be
# overconfident, the simulator may be wrong, or the observation may contain
# structure that the simulator cannot express.

# %%
# Use a subset of posterior samples so the plot remains readable.
draw_ids = torch.randperm(posterior_samples.shape[0])[:160]

# We draw clean chirps, not noisy simulations, to show which signal shapes the
# posterior believes could be hiding under the observed detector noise.
predictive_clean = clean_chirp(posterior_samples[draw_ids], config=config)

ax = plot_posterior_predictive(
    time,
    x_obs,
    predictive_clean,
    title="Posterior implied clean chirps over the observed data",
)
ax.plot(time, x_clean, color="tab:orange", lw=2.0, label="true clean signal")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 6. A Simple Misspecification Stress Test
#
# Now add a short glitch to the observation but keep using the same posterior
# estimator. The point is not to get a perfect answer; the point is to show that
# posterior predictive checks can expose when the simulator is missing part of
# the data-generating process.

# %%
# A narrow Gaussian bump is a simple stand-in for an unmodelled detector glitch.
# The expression is amplitude * exp[-0.5 * ((t - centre) / width)^2].
glitch_amplitude = 1.2
glitch_center = 0.48
glitch_width = 0.012
glitch = glitch_amplitude * torch.exp(-0.5 * ((time - glitch_center) / glitch_width) ** 2)
x_glitch = x_obs + glitch

# Keep the same training-set standardisation. Changing preprocessing here would
# make this observation inconsistent with what the neural posterior learned.
x_glitch_z = standardize(x_glitch, x_mean, x_std)

# The posterior object is reused without retraining: only the conditioning
# observation changes.
glitch_samples = posterior.sample((3000,), x=x_glitch_z)
glitch_draw_ids = torch.randperm(glitch_samples.shape[0])[:160]
glitch_predictive = clean_chirp(glitch_samples[glitch_draw_ids], config=config)

fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
plot_signal(time, x_glitch, x_clean, ax=axes[0], title="Observation with an unmodelled glitch")
plot_posterior_predictive(
    time,
    x_glitch,
    glitch_predictive,
    ax=axes[1],
    title="Predictive check exposes the mismatch",
)

# Shade the part of the time series where the injected glitch is concentrated.
# Three Gaussian widths captures the visibly affected region without covering
# unrelated parts of the signal.
glitch_window = 3.0 * glitch_width
for ax in axes:
    ax.axvspan(
        glitch_center - glitch_window,
        glitch_center + glitch_window,
        color="tab:red",
        alpha=0.12,
        lw=0,
        label="glitch window",
    )
    ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Suggested Workshop Exercises
#
# 1. Change `config.noise_std` to `0.55`. Which parameters become uncertain?
# 2. Reduce `NUM_SIMULATIONS` to `1000`. What changes in the posterior?
# 3. Increase the prior width on `merger_time`. Does the posterior still find the signal?
# 4. Replace coloured noise with white noise in `src/gw_sbi_demo/simulator.py`.
# 5. Add a second chirp to the observation. Can the current simulator explain it?
