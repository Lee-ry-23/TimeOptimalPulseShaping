from dataclasses import dataclass

import numpy as np

from time_optimal_grape.typing import (
    ComplexVector,
    HamiltonianCallable,
    LocalPhaseCoefficients,
    PhaseDerivativeMap,
)


@dataclass(frozen=True)
class ManualHamiltonianBlock:
    name: str
    weight: int
    initial_state: ComplexVector
    target_state: ComplexVector
    hamiltonian: HamiltonianCallable
    derivatives: PhaseDerivativeMap
    phase_offset: float
    local_phase_coefficients: LocalPhaseCoefficients

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ManualHamiltonianBlock.name cannot be empty.")
        if self.weight <= 0:
            raise ValueError(f"Block {self.name!r} has non-positive weight {self.weight}.")

        initial_state = np.asarray(self.initial_state, dtype=np.complex128)
        target_state = np.asarray(self.target_state, dtype=np.complex128)
        if initial_state.ndim != 1:
            raise ValueError(f"Block {self.name!r} initial_state must be a vector.")
        if target_state.ndim != 1:
            raise ValueError(f"Block {self.name!r} target_state must be a vector.")
        if initial_state.shape != target_state.shape:
            raise ValueError(
                f"Block {self.name!r} initial and target shapes differ: "
                f"{initial_state.shape} != {target_state.shape}."
            )
        if initial_state.size == 0:
            raise ValueError(f"Block {self.name!r} state dimension cannot be zero.")
        if not self.derivatives:
            raise ValueError(f"Block {self.name!r} must define at least one Hamiltonian derivative.")

        object.__setattr__(self, "initial_state", initial_state.copy())
        object.__setattr__(self, "target_state", target_state.copy())

    @property
    def dimension(self) -> int:
        return int(self.initial_state.shape[0])

    def phase(self, local_phases: dict[str, float]) -> float:
        phase_value = self.phase_offset
        for local_phase_name, coefficient in self.local_phase_coefficients.items():
            if local_phase_name not in local_phases:
                raise ValueError(f"Block {self.name!r} requires missing local phase {local_phase_name!r}.")
            phase_value += coefficient * local_phases[local_phase_name]
        return phase_value

