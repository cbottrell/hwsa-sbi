"""Small teaching utilities for the HWSA SBI astrophysics workshop."""

from .simulator import (
    DEFAULT_PARAMETER_NAMES,
    DEFAULT_PRIOR_HIGH,
    DEFAULT_PRIOR_LOW,
    GWConfig,
    build_prior,
    clean_chirp,
    make_observation,
    simulate,
    standardize,
    time_grid,
)

__all__ = [
    "DEFAULT_PARAMETER_NAMES",
    "DEFAULT_PRIOR_HIGH",
    "DEFAULT_PRIOR_LOW",
    "GWConfig",
    "build_prior",
    "clean_chirp",
    "make_observation",
    "simulate",
    "standardize",
    "time_grid",
]

