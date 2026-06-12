from dataclasses import dataclass
from typing import Protocol

import numpy as np

from time_optimal_grape.controls import ControlLayout, ControlValues
from time_optimal_grape.evolution import BlockEvolution
from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class FidelityEvaluation:
    fidelity: float
    gradient: RealVector


class FidelityObjective(Protocol):
    def evaluate(
        self,
        block_evolutions: tuple[BlockEvolution, ...],
        control_layout: ControlLayout,
        control_values: ControlValues,
    ) -> FidelityEvaluation:
        ...


@dataclass(frozen=True)
class WeightedOverlapFidelity:
    normalizer: float

    def __post_init__(self) -> None:
        if self.normalizer <= 0.0:
            raise ValueError(f"WeightedOverlapFidelity.normalizer must be positive, got {self.normalizer}.")

    def evaluate(
        self,
        block_evolutions: tuple[BlockEvolution, ...],
        control_layout: ControlLayout,
        control_values: ControlValues,
    ) -> FidelityEvaluation:
        weighted_terms = build_weighted_terms(block_evolutions, control_values)
        coherent_sum = sum(term for term in weighted_terms)
        fidelity = float(np.abs(coherent_sum) ** 2 / self.normalizer)
        gradient = np.zeros(control_layout.parameter_count, dtype=np.float64)

        for block_evolution, weighted_term in zip(block_evolutions, weighted_terms, strict=True):
            phase_factor = np.exp(-1j * block_evolution.block.phase(control_values.local_phases))
            for time_control_name in control_layout.time_control_names:
                overlap_gradient = block_evolution.overlap_gradients[time_control_name]
                for sample_index in range(control_layout.samples):
                    derivative_index = control_layout.time_control_parameter_index(time_control_name, sample_index)
                    term_derivative = block_evolution.block.weight * phase_factor * overlap_gradient[sample_index]
                    gradient[derivative_index] += 2.0 * float(np.real(np.conj(coherent_sum) * term_derivative))

            for local_phase_name, coefficient in block_evolution.block.local_phase_coefficients.items():
                derivative_index = control_layout.local_phase_parameter_index(local_phase_name)
                term_derivative = -1j * coefficient * weighted_term
                gradient[derivative_index] += 2.0 * float(np.real(np.conj(coherent_sum) * term_derivative))

        gradient /= self.normalizer
        return FidelityEvaluation(fidelity=fidelity, gradient=gradient)


@dataclass(frozen=True)
class AveragePhaseGateFidelity:
    hilbert_dimension: int

    def __post_init__(self) -> None:
        if self.hilbert_dimension <= 0:
            raise ValueError(
                f"AveragePhaseGateFidelity.hilbert_dimension must be positive, got {self.hilbert_dimension}."
            )

    @property
    def normalizer(self) -> float:
        dimension = float(self.hilbert_dimension)
        return dimension * dimension + dimension

    def evaluate(
        self,
        block_evolutions: tuple[BlockEvolution, ...],
        control_layout: ControlLayout,
        control_values: ControlValues,
    ) -> FidelityEvaluation:
        weighted_terms = build_weighted_terms(block_evolutions, control_values)
        coherent_sum = sum(term for term in weighted_terms)
        incoherent_sum = 0.0
        for block_evolution in block_evolutions:
            incoherent_sum += block_evolution.block.weight * float(np.abs(block_evolution.overlap) ** 2)

        fidelity = float((np.abs(coherent_sum) ** 2 + incoherent_sum) / self.normalizer)
        gradient = np.zeros(control_layout.parameter_count, dtype=np.float64)

        for block_evolution, weighted_term in zip(block_evolutions, weighted_terms, strict=True):
            phase_factor = np.exp(-1j * block_evolution.block.phase(control_values.local_phases))
            overlap = block_evolution.overlap
            for time_control_name in control_layout.time_control_names:
                overlap_gradient = block_evolution.overlap_gradients[time_control_name]
                for sample_index in range(control_layout.samples):
                    derivative_index = control_layout.time_control_parameter_index(time_control_name, sample_index)
                    term_derivative = block_evolution.block.weight * phase_factor * overlap_gradient[sample_index]
                    incoherent_derivative = block_evolution.block.weight * np.conj(overlap) * overlap_gradient[
                        sample_index
                    ]
                    gradient[derivative_index] += 2.0 * float(
                        np.real(np.conj(coherent_sum) * term_derivative + incoherent_derivative)
                    )

            for local_phase_name, coefficient in block_evolution.block.local_phase_coefficients.items():
                derivative_index = control_layout.local_phase_parameter_index(local_phase_name)
                term_derivative = -1j * coefficient * weighted_term
                gradient[derivative_index] += 2.0 * float(np.real(np.conj(coherent_sum) * term_derivative))

        gradient /= self.normalizer
        return FidelityEvaluation(fidelity=fidelity, gradient=gradient)


def build_weighted_terms(
    block_evolutions: tuple[BlockEvolution, ...],
    control_values: ControlValues,
) -> list[complex]:
    terms: list[complex] = []
    for block_evolution in block_evolutions:
        phase_factor = np.exp(-1j * block_evolution.block.phase(control_values.local_phases))
        terms.append(block_evolution.block.weight * phase_factor * block_evolution.overlap)
    return terms
