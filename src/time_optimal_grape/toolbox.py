from collections.abc import Mapping, Sequence

import numpy as np

from time_optimal_grape.controls import ControlLayout
from time_optimal_grape.typing import ComplexVector, RealVector


def basis_state(dimension: int, index: int) -> ComplexVector:
    if dimension <= 0:
        raise ValueError(f"Basis dimension must be positive, got {dimension}.")
    if index < 0 or index >= dimension:
        raise ValueError(f"Basis index {index} is outside [0, {dimension}).")
    state = np.zeros(dimension, dtype=np.complex128)
    state[index] = 1.0
    return state


def normalized_state(amplitudes: Sequence[complex]) -> ComplexVector:
    state = np.asarray(amplitudes, dtype=np.complex128)
    if state.ndim != 1:
        raise ValueError(f"State amplitudes must define a vector, got shape {state.shape}.")
    if state.size == 0:
        raise ValueError("State amplitudes cannot be empty.")
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    return state / norm


def zero_phase_profile(samples: int) -> RealVector:
    return zero_control_profile(samples=samples)


def zero_control_profile(samples: int) -> RealVector:
    validate_samples(samples)
    return np.zeros(samples, dtype=np.float64)


def constant_phase_profile(samples: int, phase: float) -> RealVector:
    return constant_control_profile(samples=samples, value=phase)


def constant_control_profile(samples: int, value: float) -> RealVector:
    validate_samples(samples)
    return np.full(samples, value, dtype=np.float64)


def random_phase_profile(samples: int, low: float, high: float, rng: np.random.Generator) -> RealVector:
    return random_control_profile(samples=samples, low=low, high=high, rng=rng)


def random_control_profile(samples: int, low: float, high: float, rng: np.random.Generator) -> RealVector:
    validate_samples(samples)
    if low >= high:
        raise ValueError(f"Expected low < high for random control profile, got low={low}, high={high}.")
    return np.asarray(rng.uniform(low=low, high=high, size=samples), dtype=np.float64)


def assemble_parameters(
    layout: ControlLayout,
    time_controls: Mapping[str, RealVector],
    local_phases: Mapping[str, float],
) -> RealVector:
    parameters = np.zeros(layout.parameter_count, dtype=np.float64)
    for time_control_name in layout.time_control_names:
        if time_control_name not in time_controls:
            raise ValueError(f"Missing time control profile {time_control_name!r}.")
        time_control_profile = np.asarray(time_controls[time_control_name], dtype=np.float64)
        if time_control_profile.shape != (layout.samples,):
            raise ValueError(
                f"Time control profile {time_control_name!r} must have shape "
                f"{(layout.samples,)}, got {time_control_profile.shape}."
            )
        for sample_index in range(layout.samples):
            parameter_index = layout.time_control_parameter_index(time_control_name, sample_index)
            parameters[parameter_index] = time_control_profile[sample_index]

    for local_phase_name in layout.local_phase_names:
        if local_phase_name not in local_phases:
            raise ValueError(f"Missing local phase {local_phase_name!r}.")
        parameters[layout.local_phase_parameter_index(local_phase_name)] = float(local_phases[local_phase_name])

    return parameters


def time_control_parameter_indices(layout: ControlLayout, time_control_name: str) -> tuple[int, ...]:
    return tuple(
        layout.time_control_parameter_index(time_control_name=time_control_name, sample_index=sample_index)
        for sample_index in range(layout.samples)
    )


def local_phase_parameter_indices(layout: ControlLayout, local_phase_names: Sequence[str]) -> tuple[int, ...]:
    return tuple(
        layout.local_phase_parameter_index(local_phase_name=local_phase_name)
        for local_phase_name in local_phase_names
    )


def locked_parameter_indices(
    layout: ControlLayout,
    time_control_names: Sequence[str],
    local_phase_names: Sequence[str],
) -> tuple[int, ...]:
    indices: list[int] = []
    for time_control_name in time_control_names:
        indices.extend(time_control_parameter_indices(layout=layout, time_control_name=time_control_name))
    indices.extend(local_phase_parameter_indices(layout=layout, local_phase_names=local_phase_names))
    if len(set(indices)) != len(indices):
        raise ValueError(f"Locked parameter names produced duplicate indices: {tuple(indices)}.")
    return tuple(indices)


def dimensionless_time_axis(duration: float, samples: int) -> RealVector:
    if duration <= 0.0:
        raise ValueError(f"Duration must be positive, got {duration}.")
    validate_samples(samples)
    return np.linspace(0.0, duration, samples + 1, dtype=np.float64)


def phase_sample_axis(duration: float, samples: int) -> RealVector:
    if duration <= 0.0:
        raise ValueError(f"Duration must be positive, got {duration}.")
    validate_samples(samples)
    delta_t = duration / samples
    return np.linspace(0.5 * delta_t, duration - 0.5 * delta_t, samples, dtype=np.float64)


def unwrap_phase_profile(phase_profile: RealVector) -> RealVector:
    profile = np.asarray(phase_profile, dtype=np.float64)
    if profile.ndim != 1:
        raise ValueError(f"Phase profile must be a vector, got shape {profile.shape}.")
    return np.asarray(np.unwrap(profile), dtype=np.float64)


def validate_samples(samples: int) -> None:
    if samples <= 0:
        raise ValueError(f"Sample count must be positive, got {samples}.")
