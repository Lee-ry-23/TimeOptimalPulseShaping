from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult

from time_optimal_grape.controls import ControlLayout
from time_optimal_grape.optimization import GrapeOptimizer, GrapeProblem, optimize_parameter_vector
from time_optimal_grape.parameterization import TimeControlParameterization
from time_optimal_grape.results import OptimizationResult
from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class ParameterizedControlProjection:
    layout: ControlLayout
    base_parameters: RealVector
    time_control_parameterizations: tuple[TimeControlParameterization, ...]
    direct_parameter_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        base_vector = np.asarray(self.base_parameters, dtype=np.float64)
        if base_vector.shape != (self.layout.parameter_count,):
            raise ValueError(
                f"Base parameter vector must have shape {(self.layout.parameter_count,)}, got {base_vector.shape}."
            )
        names = tuple(parameterization.time_control_name for parameterization in self.time_control_parameterizations)
        if len(set(names)) != len(names):
            raise ValueError(f"Time control parameterizations contain duplicate names: {names}.")
        for name in names:
            if name not in self.layout.time_control_names:
                raise ValueError(f"Cannot parameterize unknown time control {name!r}.")
        for index in self.direct_parameter_indices:
            if index < 0 or index >= self.layout.parameter_count:
                raise ValueError(
                    f"Direct parameter index {index} is outside [0, {self.layout.parameter_count})."
                )
        if len(set(self.direct_parameter_indices)) != len(self.direct_parameter_indices):
            raise ValueError(f"Direct parameter indices contain duplicates: {self.direct_parameter_indices}.")
        parameterized_time_indices = parameterized_control_indices(
            layout=self.layout,
            time_control_parameterizations=self.time_control_parameterizations,
        )
        overlap = set(parameterized_time_indices).intersection(self.direct_parameter_indices)
        if overlap:
            raise ValueError(
                "A full control parameter cannot be both ansatz-controlled and direct-controlled: "
                f"{tuple(sorted(overlap))}."
            )
        object.__setattr__(self, "base_parameters", base_vector.copy())


@dataclass(frozen=True)
class ParameterizedOptimizationResult:
    ansatz_parameters: RealVector
    full_parameters: RealVector
    fidelity: float
    infidelity: float
    success: bool
    message: str
    iterations: int
    raw_result: OptimizeResult


def parameterized_parameter_count(projection: ParameterizedControlProjection) -> int:
    time_parameter_count = sum(
        parameterization.parameter_count for parameterization in projection.time_control_parameterizations
    )
    return time_parameter_count + len(projection.direct_parameter_indices)


def expand_parameterized_controls(
    projection: ParameterizedControlProjection,
    ansatz_parameters: RealVector,
) -> RealVector:
    parameters = validate_ansatz_parameters(projection=projection, ansatz_parameters=ansatz_parameters)
    full_parameters = projection.base_parameters.copy()
    offset = 0
    for parameterization in projection.time_control_parameterizations:
        next_offset = offset + parameterization.parameter_count
        profile_result = parameterization.evaluate(parameters[offset:next_offset])
        if profile_result.profile.shape != (projection.layout.samples,):
            raise ValueError(
                f"Parameterized profile {parameterization.time_control_name!r} must have shape "
                f"{(projection.layout.samples,)}, got {profile_result.profile.shape}."
            )
        for sample_index, value in enumerate(profile_result.profile):
            full_index = projection.layout.time_control_parameter_index(
                time_control_name=parameterization.time_control_name,
                sample_index=sample_index,
            )
            full_parameters[full_index] = value
        offset = next_offset
    direct_values = parameters[offset:]
    for direct_value, full_index in zip(direct_values, projection.direct_parameter_indices, strict=True):
        full_parameters[full_index] = direct_value
    return full_parameters


def project_full_gradient(
    projection: ParameterizedControlProjection,
    ansatz_parameters: RealVector,
    full_gradient: RealVector,
) -> RealVector:
    parameters = validate_ansatz_parameters(projection=projection, ansatz_parameters=ansatz_parameters)
    gradient = np.zeros(parameters.shape, dtype=np.float64)
    full_gradient_vector = np.asarray(full_gradient, dtype=np.float64)
    if full_gradient_vector.shape != (projection.layout.parameter_count,):
        raise ValueError(
            f"Full gradient must have shape {(projection.layout.parameter_count,)}, got {full_gradient_vector.shape}."
        )

    offset = 0
    for parameterization in projection.time_control_parameterizations:
        next_offset = offset + parameterization.parameter_count
        profile_result = parameterization.evaluate(parameters[offset:next_offset])
        sample_indices = tuple(
            projection.layout.time_control_parameter_index(
                time_control_name=parameterization.time_control_name,
                sample_index=sample_index,
            )
            for sample_index in range(projection.layout.samples)
        )
        profile_gradient = full_gradient_vector[np.asarray(sample_indices, dtype=np.int64)]
        gradient[offset:next_offset] = profile_result.jacobian.T @ profile_gradient
        offset = next_offset

    for ansatz_index, full_index in enumerate(projection.direct_parameter_indices, start=offset):
        gradient[ansatz_index] = full_gradient_vector[full_index]
    return gradient


def evaluate_parameterized_controls(
    problem: GrapeProblem,
    projection: ParameterizedControlProjection,
    ansatz_parameters: RealVector,
) -> tuple[float, RealVector]:
    validate_problem_projection(problem=problem, projection=projection)
    full_parameters = expand_parameterized_controls(
        projection=projection,
        ansatz_parameters=ansatz_parameters,
    )
    cost, full_gradient = GrapeOptimizer(problem).evaluate(full_parameters)
    gradient = project_full_gradient(
        projection=projection,
        ansatz_parameters=ansatz_parameters,
        full_gradient=full_gradient,
    )
    return float(cost), gradient


def optimize_parameterized_controls(
    problem: GrapeProblem,
    projection: ParameterizedControlProjection,
    initial_ansatz_parameters: RealVector,
    locked_ansatz_parameter_indices: Sequence[int],
) -> ParameterizedOptimizationResult:
    validate_problem_projection(problem=problem, projection=projection)
    raw_result, optimized_ansatz_parameters = optimize_parameter_vector(
        evaluate=lambda parameters: evaluate_parameterized_controls(
            problem=problem,
            projection=projection,
            ansatz_parameters=parameters,
        ),
        initial_parameters=initial_ansatz_parameters,
        parameter_count=parameterized_parameter_count(projection=projection),
        locked_parameter_indices=locked_ansatz_parameter_indices,
        optimizer_settings=problem.optimizer_settings,
    )
    return build_parameterized_result(
        problem=problem,
        projection=projection,
        raw_result=raw_result,
        optimized_ansatz_parameters=optimized_ansatz_parameters,
    )


def build_parameterized_result(
    problem: GrapeProblem,
    projection: ParameterizedControlProjection,
    raw_result: OptimizeResult,
    optimized_ansatz_parameters: RealVector,
) -> ParameterizedOptimizationResult:
    full_parameters = expand_parameterized_controls(
        projection=projection,
        ansatz_parameters=optimized_ansatz_parameters,
    )
    infidelity, _gradient = GrapeOptimizer(problem).evaluate(full_parameters)
    return ParameterizedOptimizationResult(
        ansatz_parameters=np.asarray(optimized_ansatz_parameters, dtype=np.float64),
        full_parameters=full_parameters,
        fidelity=1.0 - infidelity,
        infidelity=infidelity,
        success=bool(raw_result.success),
        message=str(raw_result.message),
        iterations=int(raw_result.nit),
        raw_result=raw_result,
    )


def as_optimization_result(result: ParameterizedOptimizationResult) -> OptimizationResult:
    return OptimizationResult(
        parameters=result.full_parameters,
        fidelity=result.fidelity,
        infidelity=result.infidelity,
        success=result.success,
        message=result.message,
        iterations=result.iterations,
        raw_result=result.raw_result,
    )


def parameterized_control_indices(
    layout: ControlLayout,
    time_control_parameterizations: tuple[TimeControlParameterization, ...],
) -> tuple[int, ...]:
    indices: list[int] = []
    for parameterization in time_control_parameterizations:
        indices.extend(
            layout.time_control_parameter_index(
                time_control_name=parameterization.time_control_name,
                sample_index=sample_index,
            )
            for sample_index in range(layout.samples)
        )
    return tuple(indices)


def validate_problem_projection(problem: GrapeProblem, projection: ParameterizedControlProjection) -> None:
    if problem.control_layout != projection.layout:
        raise ValueError("Parameterized projection layout does not match problem layout.")


def validate_ansatz_parameters(
    projection: ParameterizedControlProjection,
    ansatz_parameters: RealVector,
) -> RealVector:
    parameter_count = parameterized_parameter_count(projection=projection)
    vector = np.asarray(ansatz_parameters, dtype=np.float64)
    if vector.shape != (parameter_count,):
        raise ValueError(f"Ansatz parameter vector must have shape {(parameter_count,)}, got {vector.shape}.")
    return vector.copy()
