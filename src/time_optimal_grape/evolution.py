from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm_frechet

from time_optimal_grape.blocks import ManualHamiltonianBlock
from time_optimal_grape.controls import ControlValues
from time_optimal_grape.typing import ComplexMatrix, ComplexVector


@dataclass(frozen=True)
class BlockEvolution:
    block: ManualHamiltonianBlock
    overlap: complex
    overlap_gradients: dict[str, ComplexVector]


def evaluate_block_evolution(
    block: ManualHamiltonianBlock,
    control_values: ControlValues,
    time_control_names: tuple[str, ...],
    duration: float,
    samples: int,
) -> BlockEvolution:
    if duration <= 0.0:
        raise ValueError(f"Pulse duration must be positive, got {duration}.")
    if samples <= 0:
        raise ValueError(f"Sample count must be positive, got {samples}.")

    delta_t = duration / samples
    unitaries: list[ComplexMatrix] = []
    unitary_derivatives: dict[str, list[ComplexMatrix]] = {
        time_control_name: [] for time_control_name in time_control_names
    }

    for sample_index in range(samples):
        hamiltonian = np.asarray(block.hamiltonian(control_values, sample_index), dtype=np.complex128)
        validate_square_matrix(block.name, hamiltonian, block.dimension)
        scaled_hamiltonian = -1j * delta_t * hamiltonian

        first_time_control_name = time_control_names[0]
        if first_time_control_name not in block.derivatives:
            raise ValueError(
                f"Block {block.name!r} is missing derivative for time control {first_time_control_name!r}."
            )
        first_derivative = np.asarray(
            block.derivatives[first_time_control_name](control_values, sample_index),
            dtype=np.complex128,
        )
        validate_square_matrix(block.name, first_derivative, block.dimension)
        first_scaled_derivative = -1j * delta_t * first_derivative
        unitary, first_frechet = expm_frechet(
            scaled_hamiltonian,
            first_scaled_derivative,
            compute_expm=True,
        )
        unitaries.append(np.asarray(unitary, dtype=np.complex128))
        unitary_derivatives[first_time_control_name].append(np.asarray(first_frechet, dtype=np.complex128))

        for time_control_name in time_control_names[1:]:
            if time_control_name not in block.derivatives:
                raise ValueError(
                    f"Block {block.name!r} is missing derivative for time control {time_control_name!r}."
                )
            derivative = np.asarray(
                block.derivatives[time_control_name](control_values, sample_index),
                dtype=np.complex128,
            )
            validate_square_matrix(block.name, derivative, block.dimension)
            scaled_derivative = -1j * delta_t * derivative
            frechet = expm_frechet(
                scaled_hamiltonian,
                scaled_derivative,
                compute_expm=False,
            )
            unitary_derivatives[time_control_name].append(np.asarray(frechet, dtype=np.complex128))

    forward_states = build_forward_states(block.initial_state, unitaries)
    backward_states = build_backward_states(block.target_state, unitaries)
    overlap = complex(np.vdot(block.target_state, forward_states[-1]))

    overlap_gradients: dict[str, ComplexVector] = {}
    for time_control_name in time_control_names:
        gradient = np.zeros(samples, dtype=np.complex128)
        for sample_index in range(samples):
            derivative_state = unitary_derivatives[time_control_name][sample_index] @ forward_states[sample_index]
            gradient[sample_index] = np.vdot(backward_states[sample_index + 1], derivative_state)
        overlap_gradients[time_control_name] = np.asarray(gradient, dtype=np.complex128)

    return BlockEvolution(block=block, overlap=overlap, overlap_gradients=overlap_gradients)


def build_forward_states(initial_state: ComplexVector, unitaries: list[ComplexMatrix]) -> list[ComplexVector]:
    states: list[ComplexVector] = [np.asarray(initial_state, dtype=np.complex128).copy()]
    for unitary in unitaries:
        states.append(unitary @ states[-1])
    return states


def build_backward_states(target_state: ComplexVector, unitaries: list[ComplexMatrix]) -> list[ComplexVector]:
    states: list[ComplexVector] = [np.zeros_like(target_state) for _ in range(len(unitaries) + 1)]
    states[-1] = np.asarray(target_state, dtype=np.complex128).copy()
    for sample_index in range(len(unitaries) - 1, -1, -1):
        states[sample_index] = unitaries[sample_index].conj().T @ states[sample_index + 1]
    return states


def validate_square_matrix(block_name: str, matrix: ComplexMatrix, dimension: int) -> None:
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"Block {block_name!r} expected matrix shape {(dimension, dimension)}, got {matrix.shape}."
        )
