# Simulation-Based Inference Workshop: Astrophysical Signals

This repository contains a Python workshop on simulation-based inference (SBI)
for astrophysical inverse problems. The main worked example recovers a
gravitational-wave-like chirp signal embedded in noisy strain data. An optional
extension shows the same SBI workflow on a sparse VLBI ring-imaging problem,
inspired by Event Horizon Telescope style data.

The material is designed for a hands-on workshop:

1. Start with a simulator and a prior.
2. Generate many synthetic observations.
3. Train a neural posterior estimator.
4. Condition on one noisy observation.
5. Diagnose the posterior with posterior predictive checks.

## Quick Start

Current `sbi` releases require Python 3.10 or newer. The system Python in this
workspace is 3.9, so use a fresh environment.

With conda or mamba:

```bash
conda env create -f environment.yml
conda activate hwsa-sbi
jupyter lab
```

With `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open the Jupytext notebooks in `notebooks/`:

- `01_gravitational_wave_sbi.py`: main workshop example.
- `02_sparse_vlbi_ring_sbi.py`: optional M87/VLBI-style extension.

If your Jupyter installation does not automatically treat `.py` files as
notebooks, install Jupytext from the requirements and open them through the
JupyterLab text/notebook pairing workflow, or run the cells in an editor that
supports `# %%` cell markers.

## Workshop Flow

Recommended timing for a 2-3 hour workshop:

| Segment | Time | Goal |
| --- | ---: | --- |
| Bayesian inverse problems without a likelihood | 15 min | Motivate SBI as simulator inversion |
| Build the chirp simulator | 25 min | Show priors, parameters, stochastic noise |
| Generate training simulations | 20 min | Connect simulation budget to posterior quality |
| Train neural posterior estimation | 30 min | Use `sbi.inference.NPE` |
| Infer one observed signal | 25 min | Sample and visualize `p(theta | x_o)` |
| Posterior predictive checks | 25 min | Ask whether inferred parameters reproduce the data |
| Extensions and exercises | 20 min | Noise, glitches, sparsity, VLBI |

## Main Example

The gravitational-wave example uses a deliberately lightweight chirp simulator.
It is not a replacement for production waveform models from LIGO/Virgo/KAGRA
analysis tools. The pedagogical goal is to make the SBI workflow transparent:
students can see how source parameters shape the waveform, how noise obscures
the signal, and how the posterior captures uncertainty.

Default inferred parameters:

- `chirp_mass`: controls how quickly the frequency sweeps upward.
- `amplitude`: controls signal strength.
- `merger_time`: controls where the chirp peaks in the time series.
- `phase`: controls the waveform phase.

## Optional VLBI Extension

The VLBI notebook uses a toy ring image and sparse Fourier measurements. It is
intended to echo the structure of EHT imaging without claiming to reproduce the
M87 reconstruction pipeline. It is useful for showing how the same SBI pattern
applies when the data are sparse, indirect measurements rather than a time
series.

## References And Tools

- [`sbi` documentation](https://sbi.readthedocs.io/en/latest/)
- [`bilby` documentation](https://bilby-dev.github.io/bilby/)
- [`GWpy` documentation](https://gwpy.github.io/docs/stable/)
- [`eht-imaging` documentation](https://achael.github.io/eht-imaging/)

