from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

from time_optimal_grape.blocks import ManualHamiltonianBlock
from time_optimal_grape.controls import ControlLayout
from time_optimal_grape.evolution import evaluate_block_evolution
from time_optimal_grape.fidelity import FidelityObjective
from time_optimal_grape.results import OptimizationResult
from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class OptimizerSettings:
    method: str
    max_iterations: int
    gradient_tolerance: float
    display: bool

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(f"OptimizerSettings.max_iterations must be positive, got {self.max_iterations}.")
        if self.gradient_tolerance <= 0.0:
            raise ValueError(
                f"OptimizerSettings.gradient_tolerance must be positive, got {self.gradient_tolerance}."
            )


@dataclass(frozen=True)
class GrapeProblem:
    blocks: tuple[ManualHamiltonianBlock, ...]
    control_layout: ControlLayout
    duration: float
    objective: FidelityObjective
    optimizer_settings: OptimizerSettings

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("GrapeProblem.blocks cannot be empty.")
        if not self.control_layout.time_control_names:
            raise ValueError("GrapeProblem.control_layout must define at least one time control.")
        if self.duration <= 0.0:
            raise ValueError(f"GrapeProblem.duration must be positive, got {self.duration}.")
        for block in self.blocks:
            for time_control_name in self.control_layout.time_control_names:
                if time_control_name not in block.derivatives:
                    raise ValueError(
                        f"Block {block.name!r} is missing derivative for time control {time_control_name!r}."
                    )

    def with_duration(self, duration: float) -> "GrapeProblem":
        return GrapeProblem(
            blocks=self.blocks,
            control_layout=self.control_layout,
            duration=duration,
            objective=self.objective,
            optimizer_settings=self.optimizer_settings,
        )


class GrapeOptimizer:
    def __init__(self, problem: GrapeProblem) -> None:
        self.problem = problem

    def evaluate(self, parameters: RealVector) -> tuple[float, RealVector]:
        fidelity, fidelity_gradient = self.evaluate_fidelity(parameters)
        infidelity = 1.0 - fidelity
        infidelity_gradient = -fidelity_gradient
        return float(infidelity), np.asarray(infidelity_gradient, dtype=np.float64)

    def evaluate_fidelity(self, parameters: RealVector) -> tuple[float, RealVector]:
        control_values = self.problem.control_layout.unpack(parameters)
        block_evolutions = tuple(
            evaluate_block_evolution(
                block=block,
                control_values=control_values,
                time_control_names=self.problem.control_layout.time_control_names,
                duration=self.problem.duration,
                samples=self.problem.control_layout.samples,
            )
            for block in self.problem.blocks
        )
        fidelity_evaluation = self.problem.objective.evaluate(
            block_evolutions=block_evolutions,
            control_layout=self.problem.control_layout,
            control_values=control_values,
        )
        return float(fidelity_evaluation.fidelity), np.asarray(fidelity_evaluation.gradient, dtype=np.float64)

    def optimize(self, initial_parameters: RealVector) -> OptimizationResult:
        raw_result, optimized_parameters = optimize_parameter_vector(
            evaluate=self.evaluate,
            initial_parameters=initial_parameters,
            parameter_count=self.problem.control_layout.parameter_count,
            locked_parameter_indices=(),
            optimizer_settings=self.problem.optimizer_settings,
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
            parameter_count=self.problem.control_layout.parameter_count,
            locked_parameter_indices=locked_parameter_indices,
            optimizer_settings=self.problem.optimizer_settings,
        )
        return self.build_result(raw_result=raw_result, optimized_parameters=optimized_parameters)

    def build_result(self, raw_result: OptimizeResult, optimized_parameters: RealVector) -> OptimizationResult:
        infidelity, _gradient = self.evaluate(optimized_parameters)
        return OptimizationResult(
            parameters=np.asarray(optimized_parameters, dtype=np.float64),
            fidelity=1.0 - infidelity,
            infidelity=infidelity,
            success=bool(raw_result.success),
            message=str(raw_result.message),
            iterations=int(raw_result.nit),
            raw_result=raw_result,
        )


def optimize_parameter_vector(
    evaluate: Callable[[RealVector], tuple[float, RealVector]],
    initial_parameters: RealVector,
    parameter_count: int,
    locked_parameter_indices: Sequence[int],
    optimizer_settings: OptimizerSettings,
) -> tuple[OptimizeResult, RealVector]:
    initial_vector = validate_initial_vector(initial_parameters=initial_parameters, parameter_count=parameter_count)
    free_indices = build_free_parameter_indices(
        parameter_count=parameter_count,
        locked_parameter_indices=locked_parameter_indices,
    )

    def evaluate_free_parameters(free_parameters: RealVector) -> tuple[float, RealVector]:
        full_parameters = initial_vector.copy()
        full_parameters[free_indices] = np.asarray(free_parameters, dtype=np.float64)
        cost, gradient = evaluate(full_parameters)
        return float(cost), np.asarray(gradient, dtype=np.float64)[free_indices]

    raw_result = minimize(
        fun=evaluate_free_parameters,
        x0=initial_vector[free_indices],
        jac=True,
        method=optimizer_settings.method,
        options={
            "maxiter": optimizer_settings.max_iterations,
            "gtol": optimizer_settings.gradient_tolerance,
            "disp": optimizer_settings.display,
        },
    )
    optimized_parameters = initial_vector.copy()
    optimized_parameters[free_indices] = np.asarray(raw_result.x, dtype=np.float64)
    return raw_result, optimized_parameters


def validate_initial_vector(initial_parameters: RealVector, parameter_count: int) -> RealVector:
    initial_vector = np.asarray(initial_parameters, dtype=np.float64)
    if initial_vector.shape != (parameter_count,):
        raise ValueError(f"Expected initial vector shape {(parameter_count,)}, got {initial_vector.shape}.")
    return initial_vector.copy()


def build_free_parameter_indices(parameter_count: int, locked_parameter_indices: Sequence[int]) -> np.ndarray:
    if parameter_count <= 0:
        raise ValueError(f"parameter_count must be positive, got {parameter_count}.")

    locked_indices = tuple(int(index) for index in locked_parameter_indices)
    if len(set(locked_indices)) != len(locked_indices):
        raise ValueError(f"Locked parameter indices contain duplicates: {locked_indices}.")

    locked_mask = np.zeros(parameter_count, dtype=np.bool_)
    for index in locked_indices:
        if index < 0 or index >= parameter_count:
            raise ValueError(f"Locked parameter index {index} is outside [0, {parameter_count}).")
        locked_mask[index] = True

    free_indices = np.flatnonzero(~locked_mask)
    if free_indices.size == 0:
        raise ValueError("At least one optimizer parameter must remain unlocked.")
    return np.asarray(free_indices, dtype=np.int64)
