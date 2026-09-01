from dataclasses import dataclass

import numpy as np

from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class PhaseProfileMetrics:
    max_abs_phase: float
    max_abs_detuning: float
    max_abs_detuning_slope: float
    rms_fit_error: float


def phase_profile_metrics(
    reference_profile: RealVector,
    candidate_profile: RealVector,
    duration: float,
) -> PhaseProfileMetrics:
    if duration <= 0.0:
        raise ValueError(f"Duration must be positive, got {duration}.")
    reference = validate_profile(profile=reference_profile, profile_label="reference profile")
    candidate = validate_profile(profile=candidate_profile, profile_label="candidate profile")
    if candidate.shape != reference.shape:
        raise ValueError(f"Candidate profile shape {candidate.shape} does not match reference shape {reference.shape}.")
    unwrapped_reference = np.unwrap(reference)
    unwrapped_candidate = np.unwrap(candidate)
    if candidate.size == 1:
        detuning = np.zeros(candidate.shape, dtype=np.float64)
        detuning_slope = np.zeros(candidate.shape, dtype=np.float64)
    else:
        sample_spacing = duration / candidate.size
        detuning = np.gradient(unwrapped_candidate, sample_spacing)
        detuning_slope = np.gradient(detuning, sample_spacing)
    fit_error = unwrapped_candidate - unwrapped_reference
    return PhaseProfileMetrics(
        max_abs_phase=float(np.max(np.abs(unwrapped_candidate))),
        max_abs_detuning=float(np.max(np.abs(detuning))),
        max_abs_detuning_slope=float(np.max(np.abs(detuning_slope))),
        rms_fit_error=float(np.sqrt(np.mean(np.square(fit_error)))),
    )


def validate_profile(profile: RealVector, profile_label: str) -> RealVector:
    vector = np.asarray(profile, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"Expected {profile_label} to be a vector, got shape {vector.shape}.")
    if vector.size == 0:
        raise ValueError(f"Expected {profile_label} to be non-empty.")
    return vector.copy()
