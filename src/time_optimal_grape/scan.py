from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from time_optimal_grape.optimization import GrapeOptimizer, GrapeProblem
from time_optimal_grape.results import TimeScanPoint
from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class TimeScanSettings:
    start_duration: float
    stop_duration: float
    duration_step: float
    infidelity_threshold: float
    stop_after_threshold_failure: bool

    def __post_init__(self) -> None:
        if self.start_duration <= 0.0:
            raise ValueError(f"TimeScanSettings.start_duration must be positive, got {self.start_duration}.")
        if self.stop_duration <= 0.0:
            raise ValueError(f"TimeScanSettings.stop_duration must be positive, got {self.stop_duration}.")
        if self.duration_step <= 0.0:
            raise ValueError(f"TimeScanSettings.duration_step must be positive, got {self.duration_step}.")
        if self.start_duration < self.stop_duration:
            raise ValueError(
                "TimeScanSettings.start_duration must be greater than or equal to stop_duration "
                "for a decreasing time scan."
            )
        if self.infidelity_threshold <= 0.0:
            raise ValueError(
                f"TimeScanSettings.infidelity_threshold must be positive, got {self.infidelity_threshold}."
            )


class TimeContinuationScanner:
    def __init__(self, problem_template: GrapeProblem, settings: TimeScanSettings) -> None:
        self.problem_template = problem_template
        self.settings = settings

    def scan(self, initial_parameters: RealVector) -> tuple[TimeScanPoint, ...]:
        return self.scan_with_locked_parameters(
            initial_parameters=initial_parameters,
            locked_parameter_indices=(),
        )

    def scan_with_locked_parameters(
        self,
        initial_parameters: RealVector,
        locked_parameter_indices: Sequence[int],
    ) -> tuple[TimeScanPoint, ...]:
        current_parameters = np.asarray(initial_parameters, dtype=np.float64)
        points: list[TimeScanPoint] = []
        duration = self.settings.start_duration

        while duration >= self.settings.stop_duration:
            problem = self.problem_template.with_duration(duration)
            result = GrapeOptimizer(problem).optimize_with_locked_parameters(
                initial_parameters=current_parameters,
                locked_parameter_indices=locked_parameter_indices,
            )
            point = TimeScanPoint(duration=duration, result=result)
            points.append(point)

            if result.infidelity > self.settings.infidelity_threshold and self.settings.stop_after_threshold_failure:
                break

            current_parameters = result.parameters
            duration -= self.settings.duration_step

        return tuple(points)
