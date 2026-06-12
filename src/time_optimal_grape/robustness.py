from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult

from time_optimal_grape.optimization import GrapeOptimizer, GrapeProblem, optimize_parameter_vector
from time_optimal_grape.results import OptimizationResult
from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class CurvatureRobustnessSettings:
    eta: float
    finite_difference_step: float

    def __post_init__(self) -> None:
        if self.eta < 0.0:
            raise ValueError(f"CurvatureRobustnessSettings.eta must be non-negative, got {self.eta}.")
        if self.finite_difference_step <= 0.0:
            raise ValueError(
                "CurvatureRobustnessSettings.finite_difference_step must be positive, "
                f"got {self.finite_difference_step}."
            )


class RobustGrapeOptimizer:
    def __init__(
        self,
        nominal_problem: GrapeProblem,
        plus_problem: GrapeProblem,
        minus_problem: GrapeProblem,
        robustness_settings: CurvatureRobustnessSettings,
    ) -> None:
        validate_compatible_problem_layouts(
            nominal_problem=nominal_problem,
            plus_problem=plus_problem,
            minus_problem=minus_problem,
        )
        self.nominal_problem = nominal_problem
        self.plus_problem = plus_problem
        self.minus_problem = minus_problem
        self.robustness_settings = robustness_settings

    def evaluate(self, parameters: RealVector) -> tuple[float, RealVector]:
        nominal_fidelity, nominal_gradient = GrapeOptimizer(self.nominal_problem).evaluate_fidelity(parameters)
        plus_fidelity, plus_gradient = GrapeOptimizer(self.plus_problem).evaluate_fidelity(parameters)
        minus_fidelity, minus_gradient = GrapeOptimizer(self.minus_problem).evaluate_fidelity(parameters)

        step_squared = self.robustness_settings.finite_difference_step**2
        curvature = (plus_fidelity - 2.0 * nominal_fidelity + minus_fidelity) / step_squared
        curvature_gradient = (plus_gradient - 2.0 * nominal_gradient + minus_gradient) / step_squared
        cost = 1.0 - nominal_fidelity - self.robustness_settings.eta * curvature
        gradient = -nominal_gradient - self.robustness_settings.eta * curvature_gradient
        return float(cost), np.asarray(gradient, dtype=np.float64)

    def optimize(self, initial_parameters: RealVector) -> OptimizationResult:
        raw_result, optimized_parameters = optimize_parameter_vector(
            evaluate=self.evaluate,
            initial_parameters=initial_parameters,
            parameter_count=self.nominal_problem.control_layout.parameter_count,
            locked_parameter_indices=(),
            optimizer_settings=self.nominal_problem.optimizer_settings,
        )
        return self.build_result(raw_result=raw_result, optimized_parameters=optimized_parameters)

    def optimize_with_locked_parameters(
        self,
        initial_parameters: RealVector,
        locked_parameter_indices: Sequence[int],
    ) -> OptimizationResult:
        raw_result, optimized_parameters = optimize_parameter_vector(
            evaluate=self.evaluate,
            initial_parameters=initial_parameters,
            parameter_count=self.nominal_problem.control_layout.parameter_count,
            locked_parameter_indices=locked_parameter_indices,
            optimizer_settings=self.nominal_problem.optimizer_settings,
        )
        return self.build_result(raw_result=raw_result, optimized_parameters=optimized_parameters)

    def build_result(self, raw_result: OptimizeResult, optimized_parameters: RealVector) -> OptimizationResult:
        cost, _gradient = self.evaluate(optimized_parameters)
        nominal_fidelity, _nominal_gradient = GrapeOptimizer(self.nominal_problem).evaluate_fidelity(optimized_parameters)
        return OptimizationResult(
            parameters=np.asarray(optimized_parameters, dtype=np.float64),
            fidelity=nominal_fidelity,
            infidelity=cost,
            success=bool(raw_result.success),
            message=str(raw_result.message),
            iterations=int(raw_result.nit),
            raw_result=raw_result,
        )


def scan_parameter_sensitivity(
    problem_factory: Callable[[float], GrapeProblem],
    parameters: RealVector,
    parameter_values: RealVector,
) -> tuple[RealVector, RealVector]:
    values = np.asarray(parameter_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Parameter scan values must be a vector, got shape {values.shape}.")
    infidelities = np.zeros(values.shape, dtype=np.float64)
    for index, parameter_value in enumerate(values):
        problem = problem_factory(float(parameter_value))
        infidelity, _gradient = GrapeOptimizer(problem).evaluate(parameters)
        infidelities[index] = infidelity
    return values, infidelities


def validate_compatible_problem_layouts(
    nominal_problem: GrapeProblem,
    plus_problem: GrapeProblem,
    minus_problem: GrapeProblem,
) -> None:
    nominal_layout = nominal_problem.control_layout
    for label, problem in (("plus", plus_problem), ("minus", minus_problem)):
        layout = problem.control_layout
        if layout.time_control_names != nominal_layout.time_control_names:
            raise ValueError(
                f"{label} problem time control names {layout.time_control_names} do not match "
                f"nominal time control names {nominal_layout.time_control_names}."
            )
        if layout.local_phase_names != nominal_layout.local_phase_names:
            raise ValueError(
                f"{label} problem local phase names {layout.local_phase_names} do not match "
                f"nominal local phase names {nominal_layout.local_phase_names}."
            )
        if layout.samples != nominal_layout.samples:
            raise ValueError(
                f"{label} problem sample count {layout.samples} does not match "
                f"nominal sample count {nominal_layout.samples}."
            )
        if problem.duration != nominal_problem.duration:
            raise ValueError(
                f"{label} problem duration {problem.duration} does not match "
                f"nominal duration {nominal_problem.duration}."
            )
