from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealVector = NDArray[np.float64]


class HamiltonianCallable(Protocol):
    def __call__(self, control_values: "ControlValuesProtocol", sample_index: int) -> ComplexMatrix:
        ...


class HamiltonianDerivativeCallable(Protocol):
    def __call__(self, control_values: "ControlValuesProtocol", sample_index: int) -> ComplexMatrix:
        ...


class ControlValuesProtocol(Protocol):
    time_controls: Mapping[str, RealVector]
    local_phases: Mapping[str, float]


PhaseDerivativeMap = Mapping[str, HamiltonianDerivativeCallable]
LocalPhaseCoefficients = Mapping[str, float]
ObjectiveCallable = Callable[[RealVector], tuple[float, RealVector]]
