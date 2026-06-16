"""Plotting helpers for the SBI workshop notebooks."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def to_numpy(value):
    """Convert torch tensors and arrays to NumPy arrays for plotting."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def plot_signal(
    time,
    observed,
    clean=None,
    ax: Optional[plt.Axes] = None,
    title: str = "Observed strain segment",
):
    """Plot an observed strain segment with an optional clean signal overlay."""

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    ax.plot(to_numpy(time), to_numpy(observed), color="0.25", lw=1.2, label="observed")
    if clean is not None:
        ax.plot(to_numpy(time), to_numpy(clean), color="tab:orange", lw=2.0, label="clean signal")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("strain [arb.]")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def plot_posterior_predictive(
    time,
    observed,
    waveforms,
    ax: Optional[plt.Axes] = None,
    title: str = "Posterior predictive waveforms",
):
    """Overlay posterior predictive waveforms on the observed data."""

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    time_np = to_numpy(time)
    observed_np = to_numpy(observed)
    waveforms_np = to_numpy(waveforms)

    for waveform in waveforms_np:
        ax.plot(time_np, waveform, color="tab:blue", alpha=0.08, lw=0.8)
    ax.plot(time_np, observed_np, color="0.1", lw=1.4, label="observed")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("strain [arb.]")
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def plot_corner(
    samples,
    labels: Sequence[str],
    truths=None,
    bins: int = 35,
):
    """Plot posterior samples with `corner` when available, otherwise fallback."""

    samples_np = to_numpy(samples)
    truths_np = None if truths is None else to_numpy(truths)

    try:
        import corner

        return corner.corner(
            samples_np,
            labels=list(labels),
            truths=truths_np,
            bins=bins,
            color="tab:blue",
            truth_color="tab:orange",
            show_titles=True,
            title_fmt=".3g",
        )
    except ImportError:
        return _fallback_corner(samples_np, labels, truths_np, bins)


def _fallback_corner(samples_np, labels, truths_np, bins):
    n_dim = samples_np.shape[1]
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(2.3 * n_dim, 2.3 * n_dim))

    for row in range(n_dim):
        for col in range(n_dim):
            ax = axes[row, col]
            if row == col:
                ax.hist(samples_np[:, col], bins=bins, color="tab:blue", alpha=0.8)
                if truths_np is not None:
                    ax.axvline(truths_np[col], color="tab:orange", lw=2)
            elif row > col:
                ax.scatter(samples_np[:, col], samples_np[:, row], s=3, alpha=0.12)
                if truths_np is not None:
                    ax.axvline(truths_np[col], color="tab:orange", lw=1)
                    ax.axhline(truths_np[row], color="tab:orange", lw=1)
            else:
                ax.axis("off")

            if row == n_dim - 1:
                ax.set_xlabel(labels[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row])
            elif col != 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig

