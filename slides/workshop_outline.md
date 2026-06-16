# Workshop Outline: Simulation-Based Inference In Astrophysics

## Learning Objectives

By the end of the workshop, participants should be able to:

- Explain SBI as Bayesian inference with a simulator but no tractable likelihood.
- Define priors, simulator parameters, observations, and posterior targets.
- Train a neural posterior estimator with `sbi`.
- Interpret posterior samples and parameter degeneracies.
- Run posterior predictive checks to test whether inferred parameters explain an observation.
- Translate the workflow to other astrophysical inverse problems.

## Core Narrative

The workshop follows one repeated sentence:

> We can simulate the universe forward, then learn how to infer it backward.

For the live example, the "universe" is a compact-binary-like chirp signal. The
observation is noisy strain data. The posterior tells us which source parameters
could plausibly have produced that noisy data.

## Session Plan

1. **Motivation**
   - Astrophysics often has excellent simulators and awkward likelihoods.
   - Examples: gravitational waves, VLBI, supernova populations, stellar streams,
     weak lensing, exoplanet transits.

2. **Bayesian Target**
   - Prior: `p(theta)`.
   - Simulator: `x ~ p(x | theta)`.
   - Posterior: `p(theta | x_o)`.
   - SBI replaces explicit likelihood evaluation with learned inference from
     simulations.

3. **Worked Example: Noisy Chirp Recovery**
   - Choose physically interpretable parameters.
   - Simulate clean chirps.
   - Add coloured detector-like noise.
   - Train neural posterior estimation.
   - Condition on one observed strain segment.

4. **Diagnostics**
   - Marginal and joint posterior plots.
   - Posterior predictive waveforms.
   - Sensitivity to simulation budget.
   - Sensitivity to signal-to-noise ratio.
   - Failure mode: data generated outside the simulator family.

5. **Extension: Sparse VLBI Ring**
   - Image plane to sparse Fourier samples.
   - Infer ring diameter, width, asymmetry, and angle.
   - Discuss why full EHT imaging is more complex.

## Teaching Prompts

- Where does prior knowledge enter?
- Which parameters are identifiable from the data?
- What degeneracies appear in the posterior?
- Does the posterior predictive distribution cover the observed signal?
- What happens when the simulator is wrong?
- How many simulations are enough?

## Exercises

1. Increase the noise level and retrain. Which parameters degrade first?
2. Reduce the number of simulations. How does the posterior change?
3. Generate an observation with a phase outside the assumed prior. What breaks?
4. Add a short glitch to the observed data. Does the posterior predictive check catch it?
5. In the VLBI notebook, reduce the number of baselines. Which image parameters remain identifiable?
