from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from time_optimal_grape.blocks import ManualHamiltonianBlock
from time_optimal_grape.controls import ControlValues
from time_optimal_grape.toolbox import dimensionless_time_axis
from time_optimal_grape.typing import ComplexVector, RealVector


@dataclass(frozen=True)
class PopulationTrace:
    times: RealVector
    populations: dict[str, RealVector]


def trace_populations(
    block: ManualHamiltonianBlock,
    control_values: ControlValues,
    duration: float,
    samples: int,
    states: Mapping[str, ComplexVector],
) -> PopulationTrace:
    if not states:
        raise ValueError("At least one population state must be requested.")
    normalized_states = normalize_population_states(block, states)
    state_history = trace_states(
        block=block,
        control_values=control_values,
        duration=duration,
        samples=samples,
    )
    populations: dict[str, RealVector] = {}
    for label, state in normalized_states.items():
        values = np.zeros(samples + 1, dtype=np.float64)
        for time_index, evolved_state in enumerate(state_history):
            values[time_index] = float(np.abs(np.vdot(state, evolved_state)) ** 2)
        populations[label] = values

    return PopulationTrace(
        times=dimensionless_time_axis(duration=duration, samples=samples),
        populations=populations,
    )


def trace_states(
    block: ManualHamiltonianBlock,
    control_values: ControlValues,
    duration: float,
    samples: int,
) -> tuple[ComplexVector, ...]:
    if duration <= 0.0:
        raise ValueError(f"Pulse duration must be positive, got {duration}.")
    if samples <= 0:
        raise ValueError(f"Sample count must be positive, got {samples}.")

    delta_t = duration / samples
    states: list[ComplexVector] = [block.initial_state.copy()]
    current_state = block.initial_state.copy()
    for sample_index in range(samples):
        hamiltonian = np.asarray(block.hamiltonian(control_values, sample_index), dtype=np.complex128)
        if hamiltonian.shape != (block.dimension, block.dimension):
            raise ValueError(
                f"Block {block.name!r} expected Hamiltonian shape "
                f"{(block.dimension, block.dimension)}, got {hamiltonian.shape}."
            )
        current_state = expm(-1j * delta_t * hamiltonian) @ current_state
        states.append(np.asarray(current_state, dtype=np.complex128))
    return tuple(states)


def normalize_population_states(
    block: ManualHamiltonianBlock,
    states: Mapping[str, ComplexVector],
) -> dict[str, ComplexVector]:
    normalized_states: dict[str, ComplexVector] = {}
    for label, state in states.items():
        state_vector = np.asarray(state, dtype=np.complex128)
        if state_vector.shape != (block.dimension,):
            raise ValueError(
                f"Population state {label!r} must have shape {(block.dimension,)}, got {state_vector.shape}."
            )
        norm = float(np.linalg.norm(state_vector))
        if norm == 0.0:
            raise ValueError(f"Population state {label!r} cannot be the zero vector.")
        normalized_states[label] = state_vector / norm
    return normalized_states

