from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from time_optimal_grape.typing import RealMatrix, RealVector


@dataclass(frozen=True)
class ProfileAndJacobian:
    profile: RealVector
    jacobian: RealMatrix

    def __post_init__(self) -> None:
        profile = np.asarray(self.profile, dtype=np.float64)
        jacobian = np.asarray(self.jacobian, dtype=np.float64)
        if profile.ndim != 1:
            raise ValueError(f"Profile must be a vector, got shape {profile.shape}.")
        if jacobian.ndim != 2:
            raise ValueError(f"Jacobian must be a matrix, got shape {jacobian.shape}.")
        if jacobian.shape[0] != profile.size:
            raise ValueError(
                f"Jacobian row count {jacobian.shape[0]} does not match profile size {profile.size}."
            )
        object.__setattr__(self, "profile", profile.copy())
        object.__setattr__(self, "jacobian", jacobian.copy())


@dataclass(frozen=True)
class TimeControlParameterization:
    time_control_name: str
    parameter_count: int
    evaluate: Callable[[RealVector], ProfileAndJacobian]

    def __post_init__(self) -> None:
        if not self.time_control_name.strip():
            raise ValueError("Time control name must not be empty.")
        if self.parameter_count <= 0:
            raise ValueError(f"Parameter count must be positive, got {self.parameter_count}.")


def phase_sample_times(duration: float, samples: int) -> RealVector:
    if duration <= 0.0:
        raise ValueError(f"Duration must be positive, got {duration}.")
    if samples <= 0:
        raise ValueError(f"Sample count must be positive, got {samples}.")
    delta_t = duration / samples
    return np.linspace(0.5 * delta_t, duration - 0.5 * delta_t, samples, dtype=np.float64)


def direct_fourier_parameter_count(order: int) -> int:
    validate_order(order=order)
    return 1 + 2 * order


def integrated_detuning_parameter_count(order: int) -> int:
    validate_order(order=order)
    return 1 + order


def single_harmonic_detuning_offset_parameter_count() -> int:
    return 2


def two_sine_phase_parameter_count() -> int:
    return 6


def spline_parameter_count(knot_count: int) -> int:
    validate_knot_count(knot_count=knot_count)
    return knot_count


def direct_fourier_phase_and_jacobian(
    parameters: RealVector,
    duration: float,
    samples: int,
    order: int,
) -> ProfileAndJacobian:
    coefficient_vector = validate_parameter_vector(
        parameters=parameters,
        parameter_count=direct_fourier_parameter_count(order=order),
        parameter_label="direct Fourier coefficients",
    )
    times = phase_sample_times(duration=duration, samples=samples)
    normalized_times = times / duration
    jacobian = np.ones((samples, coefficient_vector.size), dtype=np.float64)
    column_index = 1
    for harmonic in range(1, order + 1):
        angle = 2.0 * np.pi * harmonic * normalized_times
        jacobian[:, column_index] = np.sin(angle)
        column_index += 1
        jacobian[:, column_index] = np.cos(angle)
        column_index += 1
    profile = jacobian @ coefficient_vector
    return ProfileAndJacobian(profile=profile, jacobian=jacobian)


def fit_direct_fourier_phase(
    phase_profile: RealVector,
    duration: float,
    order: int,
) -> RealVector:
    profile = validate_profile(profile=phase_profile, profile_label="phase profile")
    basis = direct_fourier_phase_and_jacobian(
        parameters=np.zeros(direct_fourier_parameter_count(order=order), dtype=np.float64),
        duration=duration,
        samples=profile.size,
        order=order,
    ).jacobian
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(basis, np.unwrap(profile), rcond=None)
    return np.asarray(coefficients, dtype=np.float64)


def integrated_detuning_phase_and_jacobian(
    parameters: RealVector,
    duration: float,
    samples: int,
    order: int,
) -> ProfileAndJacobian:
    coefficient_vector = validate_parameter_vector(
        parameters=parameters,
        parameter_count=integrated_detuning_parameter_count(order=order),
        parameter_label="integrated detuning coefficients",
    )
    times = phase_sample_times(duration=duration, samples=samples)
    jacobian = np.ones((samples, coefficient_vector.size), dtype=np.float64)
    for harmonic in range(1, order + 1):
        column_index = harmonic
        angle = np.pi * harmonic * times / duration
        jacobian[:, column_index] = duration * (1.0 - np.cos(angle)) / (np.pi * harmonic)
    profile = jacobian @ coefficient_vector
    return ProfileAndJacobian(profile=profile, jacobian=jacobian)


def fit_integrated_detuning_phase(
    phase_profile: RealVector,
    duration: float,
    order: int,
) -> RealVector:
    profile = validate_profile(profile=phase_profile, profile_label="phase profile")
    basis = integrated_detuning_phase_and_jacobian(
        parameters=np.zeros(integrated_detuning_parameter_count(order=order), dtype=np.float64),
        duration=duration,
        samples=profile.size,
        order=order,
    ).jacobian
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(basis, np.unwrap(profile), rcond=None)
    return np.asarray(coefficients, dtype=np.float64)


def single_harmonic_detuning_offset_phase_and_jacobian(
    parameters: RealVector,
    duration: float,
    samples: int,
    detuning_offset: float,
) -> ProfileAndJacobian:
    coefficient_vector = validate_parameter_vector(
        parameters=parameters,
        parameter_count=single_harmonic_detuning_offset_parameter_count(),
        parameter_label="single-harmonic detuning-offset coefficients",
    )
    times = phase_sample_times(duration=duration, samples=samples)
    jacobian = np.ones((samples, coefficient_vector.size), dtype=np.float64)
    jacobian[:, 1] = duration * (1.0 - np.cos(np.pi * times / duration)) / np.pi
    profile = coefficient_vector[0] + detuning_offset * times + coefficient_vector[1] * jacobian[:, 1]
    return ProfileAndJacobian(profile=profile, jacobian=jacobian)


def fit_single_harmonic_detuning_offset_phase(
    phase_profile: RealVector,
    duration: float,
    detuning_offset: float,
) -> RealVector:
    profile = validate_profile(profile=phase_profile, profile_label="phase profile")
    basis = single_harmonic_detuning_offset_phase_and_jacobian(
        parameters=np.zeros(single_harmonic_detuning_offset_parameter_count(), dtype=np.float64),
        duration=duration,
        samples=profile.size,
        detuning_offset=detuning_offset,
    ).jacobian
    times = phase_sample_times(duration=duration, samples=profile.size)
    target = np.unwrap(profile) - detuning_offset * times
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(basis, target, rcond=None)
    return np.asarray(coefficients, dtype=np.float64)


def two_sine_phase_and_jacobian(
    parameters: RealVector,
    duration: float,
    samples: int,
) -> ProfileAndJacobian:
    coefficient_vector = validate_parameter_vector(
        parameters=parameters,
        parameter_count=two_sine_phase_parameter_count(),
        parameter_label="two-sine phase coefficients",
    )
    times = phase_sample_times(duration=duration, samples=samples)
    fixed_angle = 2.0 * np.pi * times / duration
    fitted_omega = coefficient_vector[5]
    fitted_angle = fitted_omega * times
    jacobian = np.ones((samples, coefficient_vector.size), dtype=np.float64)
    jacobian[:, 1] = np.sin(fixed_angle)
    jacobian[:, 2] = np.cos(fixed_angle)
    jacobian[:, 3] = np.sin(fitted_angle)
    jacobian[:, 4] = np.cos(fitted_angle)
    jacobian[:, 5] = times * (
        coefficient_vector[3] * np.cos(fitted_angle) - coefficient_vector[4] * np.sin(fitted_angle)
    )
    profile = (
        coefficient_vector[0]
        + coefficient_vector[1] * jacobian[:, 1]
        + coefficient_vector[2] * jacobian[:, 2]
        + coefficient_vector[3] * jacobian[:, 3]
        + coefficient_vector[4] * jacobian[:, 4]
    )
    return ProfileAndJacobian(profile=profile, jacobian=jacobian)


def fit_two_sine_phase(
    phase_profile: RealVector,
    duration: float,
    omega_min: float,
    omega_max: float,
    omega_grid_size: int,
) -> RealVector:
    profile = validate_profile(profile=phase_profile, profile_label="phase profile")
    validate_omega_scan(
        omega_min=omega_min,
        omega_max=omega_max,
        omega_grid_size=omega_grid_size,
    )
    target = np.unwrap(profile)
    omega_grid = np.linspace(omega_min, omega_max, omega_grid_size, dtype=np.float64)
    grid_errors = np.asarray(
        [
            two_sine_fit_error(
                omega=float(omega),
                target=target,
                duration=duration,
            )
            for omega in omega_grid
        ],
        dtype=np.float64,
    )
    best_grid_index = int(np.argmin(grid_errors))
    bracket_radius = (omega_max - omega_min) / float(omega_grid_size - 1)
    local_min = max(omega_min, float(omega_grid[best_grid_index] - bracket_radius))
    local_max = min(omega_max, float(omega_grid[best_grid_index] + bracket_radius))
    optimization_result = minimize_scalar(
        fun=lambda omega: two_sine_fit_error(
            omega=float(omega),
            target=target,
            duration=duration,
        ),
        bounds=(local_min, local_max),
        method="bounded",
    )
    if not optimization_result.success:
        raise RuntimeError(f"Two-sine omega fit failed: {optimization_result.message}")
    fitted_omega = float(optimization_result.x)
    coefficients = two_sine_linear_coefficients(
        omega=fitted_omega,
        target=target,
        duration=duration,
    )
    return np.asarray((*coefficients, fitted_omega), dtype=np.float64)


def spline_phase_and_jacobian(
    parameters: RealVector,
    duration: float,
    samples: int,
    knot_count: int,
) -> ProfileAndJacobian:
    knot_values = validate_parameter_vector(
        parameters=parameters,
        parameter_count=spline_parameter_count(knot_count=knot_count),
        parameter_label="spline knot values",
    )
    times = phase_sample_times(duration=duration, samples=samples)
    knot_times = np.linspace(0.0, duration, knot_count, dtype=np.float64)
    jacobian = spline_basis_matrix(times=times, knot_times=knot_times)
    profile = jacobian @ knot_values
    return ProfileAndJacobian(profile=profile, jacobian=jacobian)


def fit_spline_phase(
    phase_profile: RealVector,
    duration: float,
    knot_count: int,
) -> RealVector:
    profile = validate_profile(profile=phase_profile, profile_label="phase profile")
    times = phase_sample_times(duration=duration, samples=profile.size)
    knot_times = np.linspace(0.0, duration, knot_count, dtype=np.float64)
    basis = spline_basis_matrix(times=times, knot_times=knot_times)
    knot_values, _residuals, _rank, _singular_values = np.linalg.lstsq(basis, np.unwrap(profile), rcond=None)
    return np.asarray(knot_values, dtype=np.float64)


def make_direct_fourier_parameterization(
    time_control_name: str,
    duration: float,
    samples: int,
    order: int,
) -> TimeControlParameterization:
    parameter_count = direct_fourier_parameter_count(order=order)

    def evaluate(parameters: RealVector) -> ProfileAndJacobian:
        return direct_fourier_phase_and_jacobian(
            parameters=parameters,
            duration=duration,
            samples=samples,
            order=order,
        )

    return TimeControlParameterization(
        time_control_name=time_control_name,
        parameter_count=parameter_count,
        evaluate=evaluate,
    )


def make_integrated_detuning_parameterization(
    time_control_name: str,
    duration: float,
    samples: int,
    order: int,
) -> TimeControlParameterization:
    parameter_count = integrated_detuning_parameter_count(order=order)

    def evaluate(parameters: RealVector) -> ProfileAndJacobian:
        return integrated_detuning_phase_and_jacobian(
            parameters=parameters,
            duration=duration,
            samples=samples,
            order=order,
        )

    return TimeControlParameterization(
        time_control_name=time_control_name,
        parameter_count=parameter_count,
        evaluate=evaluate,
    )


def make_single_harmonic_detuning_offset_parameterization(
    time_control_name: str,
    duration: float,
    samples: int,
    detuning_offset: float,
) -> TimeControlParameterization:
    parameter_count = single_harmonic_detuning_offset_parameter_count()

    def evaluate(parameters: RealVector) -> ProfileAndJacobian:
        return single_harmonic_detuning_offset_phase_and_jacobian(
            parameters=parameters,
            duration=duration,
            samples=samples,
            detuning_offset=detuning_offset,
        )

    return TimeControlParameterization(
        time_control_name=time_control_name,
        parameter_count=parameter_count,
        evaluate=evaluate,
    )


def make_two_sine_phase_parameterization(
    time_control_name: str,
    duration: float,
    samples: int,
) -> TimeControlParameterization:
    parameter_count = two_sine_phase_parameter_count()

    def evaluate(parameters: RealVector) -> ProfileAndJacobian:
        return two_sine_phase_and_jacobian(
            parameters=parameters,
            duration=duration,
            samples=samples,
        )

    return TimeControlParameterization(
        time_control_name=time_control_name,
        parameter_count=parameter_count,
        evaluate=evaluate,
    )


def make_spline_parameterization(
    time_control_name: str,
    duration: float,
    samples: int,
    knot_count: int,
) -> TimeControlParameterization:
    parameter_count = spline_parameter_count(knot_count=knot_count)

    def evaluate(parameters: RealVector) -> ProfileAndJacobian:
        return spline_phase_and_jacobian(
            parameters=parameters,
            duration=duration,
            samples=samples,
            knot_count=knot_count,
        )

    return TimeControlParameterization(
        time_control_name=time_control_name,
        parameter_count=parameter_count,
        evaluate=evaluate,
    )


def two_sine_linear_coefficients(
    omega: float,
    target: RealVector,
    duration: float,
) -> RealVector:
    target_profile = validate_profile(profile=target, profile_label="target profile")
    basis = two_sine_linear_basis(
        omega=omega,
        duration=duration,
        samples=target_profile.size,
    )
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(basis, target_profile, rcond=None)
    return np.asarray(coefficients, dtype=np.float64)


def two_sine_fit_error(
    omega: float,
    target: RealVector,
    duration: float,
) -> float:
    target_profile = validate_profile(profile=target, profile_label="target profile")
    coefficients = two_sine_linear_coefficients(
        omega=omega,
        target=target_profile,
        duration=duration,
    )
    basis = two_sine_linear_basis(
        omega=omega,
        duration=duration,
        samples=target_profile.size,
    )
    residual = basis @ coefficients - target_profile
    return float(np.mean(np.square(residual)))


def two_sine_linear_basis(
    omega: float,
    duration: float,
    samples: int,
) -> RealMatrix:
    if duration <= 0.0:
        raise ValueError(f"Duration must be positive, got {duration}.")
    times = phase_sample_times(duration=duration, samples=samples)
    fixed_angle = 2.0 * np.pi * times / duration
    fitted_angle = omega * times
    return np.column_stack(
        (
            np.ones(samples, dtype=np.float64),
            np.sin(fixed_angle),
            np.cos(fixed_angle),
            np.sin(fitted_angle),
            np.cos(fitted_angle),
        )
    ).astype(np.float64)


def spline_basis_matrix(times: RealVector, knot_times: RealVector) -> RealMatrix:
    sample_times = validate_profile(profile=times, profile_label="sample times")
    knots = validate_profile(profile=knot_times, profile_label="knot times")
    if knots.size < 4:
        raise ValueError(f"Cubic spline parameterization requires at least 4 knots, got {knots.size}.")
    if not np.all(np.diff(knots) > 0.0):
        raise ValueError("Spline knot times must be strictly increasing.")
    basis = np.zeros((sample_times.size, knots.size), dtype=np.float64)
    for column_index in range(knots.size):
        unit_values = np.zeros(knots.size, dtype=np.float64)
        unit_values[column_index] = 1.0
        spline = CubicSpline(knots, unit_values, bc_type="natural")
        basis[:, column_index] = np.asarray(spline(sample_times), dtype=np.float64)
    return basis


def validate_order(order: int) -> None:
    if order <= 0:
        raise ValueError(f"Basis order must be positive, got {order}.")


def validate_knot_count(knot_count: int) -> None:
    if knot_count < 4:
        raise ValueError(f"Cubic spline knot count must be at least 4, got {knot_count}.")


def validate_omega_scan(omega_min: float, omega_max: float, omega_grid_size: int) -> None:
    if omega_min >= omega_max:
        raise ValueError(f"Expected omega_min < omega_max, got {omega_min} >= {omega_max}.")
    if omega_grid_size < 2:
        raise ValueError(f"Omega grid size must be at least 2, got {omega_grid_size}.")


def validate_parameter_vector(parameters: RealVector, parameter_count: int, parameter_label: str) -> RealVector:
    vector = np.asarray(parameters, dtype=np.float64)
    if vector.shape != (parameter_count,):
        raise ValueError(f"Expected {parameter_label} shape {(parameter_count,)}, got {vector.shape}.")
    return vector.copy()


def validate_profile(profile: RealVector, profile_label: str) -> RealVector:
    vector = np.asarray(profile, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"Expected {profile_label} to be a vector, got shape {vector.shape}.")
    if vector.size == 0:
        raise ValueError(f"Expected {profile_label} to be non-empty.")
    return vector.copy()
