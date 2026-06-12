from dataclasses import dataclass
from typing import Any

from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class OptimizationResult:
    parameters: RealVector
    fidelity: float
    infidelity: float
    success: bool
    message: str
    iterations: int
    raw_result: Any


@dataclass(frozen=True)
class TimeScanPoint:
    duration: float
    result: OptimizationResult

