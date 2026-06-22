from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import add

import numpy as np
import qutip as qt


@dataclass(frozen=True)
class ReadoutParameters:
    omega_rb: float
    b_rb_cs: float
    b_edge: float
    b_diag: float
    rb_stark_shift: float
    cs_stark_shift: float


VERTICES: tuple[int, ...] = (0, 1, 2, 3)
EDGE_PAIRS: frozenset[frozenset[int]] = frozenset(
    {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3)), frozenset((3, 0))}
)
DIAG_PAIRS: frozenset[frozenset[int]] = frozenset({frozenset((0, 2)), frozenset((1, 3))})
RB_DIMENSION: int = 3
CS_DIMENSION: int = 2


def tensor_operator(rb_operators: tuple[qt.Qobj, ...], cs_operator: qt.Qobj) -> qt.Qobj:
    if len(rb_operators) != len(VERTICES):
        raise ValueError(f"Expected {len(VERTICES)} Rb operators, got {len(rb_operators)}.")
    return qt.tensor((*rb_operators, cs_operator))


def rb_single_operator(operator: qt.Qobj, vertex: int) -> qt.Qobj:
    if vertex not in VERTICES:
        raise ValueError(f"Unknown Rb vertex {vertex}.")
    identity = qt.qeye(RB_DIMENSION)
    rb_operators = tuple(operator if rb_vertex == vertex else identity for rb_vertex in VERTICES)
    return tensor_operator(rb_operators=rb_operators, cs_operator=qt.qeye(CS_DIMENSION))


def cs_single_operator(operator: qt.Qobj) -> qt.Qobj:
    return tensor_operator(
        rb_operators=tuple(qt.qeye(RB_DIMENSION) for _vertex in VERTICES),
        cs_operator=operator,
    )


def rb_pair_operator(operator: qt.Qobj, first_vertex: int, second_vertex: int) -> qt.Qobj:
    if first_vertex == second_vertex:
        raise ValueError(f"Pair vertices must differ, got {first_vertex}.")
    if first_vertex not in VERTICES or second_vertex not in VERTICES:
        raise ValueError(f"Unknown Rb pair ({first_vertex}, {second_vertex}).")
    identity = qt.qeye(RB_DIMENSION)
    rb_operators = tuple(
        operator if rb_vertex in (first_vertex, second_vertex) else identity for rb_vertex in VERTICES
    )
    return tensor_operator(rb_operators=rb_operators, cs_operator=qt.qeye(CS_DIMENSION))


def rb_cs_pair_operator(rb_operator: qt.Qobj, cs_operator: qt.Qobj, vertex: int) -> qt.Qobj:
    if vertex not in VERTICES:
        raise ValueError(f"Unknown Rb vertex {vertex}.")
    identity = qt.qeye(RB_DIMENSION)
    rb_operators = tuple(rb_operator if rb_vertex == vertex else identity for rb_vertex in VERTICES)
    return tensor_operator(rb_operators=rb_operators, cs_operator=cs_operator)


def qobj_sum(operators: tuple[qt.Qobj, ...]) -> qt.Qobj:
    if not operators:
        raise ValueError("Cannot sum an empty operator tuple.")
    return reduce(add, operators)


def build_full_hamiltonian(
    parameters: ReadoutParameters,
    phi_rb: float,
    phi_cs: float,
    ratio_cs: float,
) -> qt.Qobj:
    rb_one = qt.basis(RB_DIMENSION, 1)
    rb_rydberg = qt.basis(RB_DIMENSION, 2)
    cs_one = qt.basis(CS_DIMENSION, 0)
    cs_rydberg = qt.basis(CS_DIMENSION, 1)

    rb_projector_rydberg = rb_rydberg * rb_rydberg.dag()
    cs_projector_rydberg = cs_rydberg * cs_rydberg.dag()
    rb_drive = parameters.omega_rb * np.exp(1j * phi_rb) * (rb_one * rb_rydberg.dag()) / 2.0
    rb_drive = rb_drive + rb_drive.dag()
    cs_drive = ratio_cs * parameters.omega_rb * np.exp(1j * phi_cs) * (cs_one * cs_rydberg.dag()) / 2.0
    cs_drive = cs_drive + cs_drive.dag()
    cs_hamiltonian = cs_drive + parameters.cs_stark_shift * cs_projector_rydberg

    rb_drive_terms = tuple(rb_single_operator(operator=rb_drive, vertex=vertex) for vertex in VERTICES)
    rb_stark_terms = tuple(
        parameters.rb_stark_shift * rb_single_operator(operator=rb_projector_rydberg, vertex=vertex)
        for vertex in VERTICES
    )
    rb_edge_terms = tuple(
        parameters.b_edge * rb_pair_operator(operator=rb_projector_rydberg, first_vertex=first, second_vertex=second)
        for first, second in (tuple(pair) for pair in EDGE_PAIRS)
    )
    rb_diag_terms = tuple(
        parameters.b_diag * rb_pair_operator(operator=rb_projector_rydberg, first_vertex=first, second_vertex=second)
        for first, second in (tuple(pair) for pair in DIAG_PAIRS)
    )
    rb_cs_terms = tuple(
        parameters.b_rb_cs
        * rb_cs_pair_operator(rb_operator=rb_projector_rydberg, cs_operator=cs_projector_rydberg, vertex=vertex)
        for vertex in VERTICES
    )
    cs_terms = (cs_single_operator(operator=cs_hamiltonian),)

    return qobj_sum(operators=rb_drive_terms + rb_stark_terms + rb_edge_terms + rb_diag_terms + rb_cs_terms + cs_terms)


def computational_state(bitstring: str, cs_state: qt.Qobj) -> qt.Qobj:
    if len(bitstring) != len(VERTICES):
        raise ValueError(f"Expected {len(VERTICES)} Rb bits, got {bitstring!r}.")
    rb_states = []
    for bit in bitstring:
        if bit == "0":
            rb_states.append(qt.basis(RB_DIMENSION, 0))
        elif bit == "1":
            rb_states.append(qt.basis(RB_DIMENSION, 1))
        else:
            raise ValueError(f"Rb bitstring must contain only 0 and 1, got {bitstring!r}.")
    return qt.tensor((*tuple(rb_states), cs_state))


def parity(bitstring: str) -> int:
    return sum(1 for bit in bitstring if bit == "1") % 2


def active_count(bitstring: str) -> int:
    return sum(1 for bit in bitstring if bit == "1")


def all_bitstrings() -> tuple[str, ...]:
    return tuple(format(index, f"0{len(VERTICES)}b") for index in range(2 ** len(VERTICES)))


def overlap_for_bitstring(unitary: qt.Qobj, bitstring: str) -> complex:
    cs_one = qt.basis(CS_DIMENSION, 0)
    cs_rydberg = qt.basis(CS_DIMENSION, 1)
    cs_plus = (cs_one + cs_rydberg).unit()
    cs_minus = (cs_one - cs_rydberg).unit()
    initial_state = computational_state(bitstring=bitstring, cs_state=cs_plus)
    target_cs_state = cs_minus if parity(bitstring=bitstring) == 1 else cs_plus
    target_state = computational_state(bitstring=bitstring, cs_state=target_cs_state)
    return complex(target_state.overlap(unitary * initial_state))


def bare_fidelity(
    parameters: ReadoutParameters,
    duration: float,
    phi_rb: float,
    phi_cs: float,
    ratio_cs: float,
    theta_rb: float,
    theta_cs: float,
) -> float:
    hamiltonian = build_full_hamiltonian(
        parameters=parameters,
        phi_rb=phi_rb,
        phi_cs=phi_cs,
        ratio_cs=ratio_cs,
    )
    unitary = (-1j * duration * hamiltonian).expm()
    coherent_sum = 0.0 + 0.0j
    for bitstring in all_bitstrings():
        phase = active_count(bitstring=bitstring) * theta_rb + theta_cs
        coherent_sum += np.exp(-1j * phase) * overlap_for_bitstring(unitary=unitary, bitstring=bitstring)
    return float(np.abs(coherent_sum) ** 2 / 16.0**2)


def main() -> None:
    duration: float = 2.0 * np.pi
    parameters = ReadoutParameters(
        omega_rb=1.0,
        b_rb_cs=10.0,
        b_edge=0.1,
        b_diag=0.0,
        rb_stark_shift=0.0,
        cs_stark_shift=0.0,
    )
    fidelity = bare_fidelity(
        parameters=parameters,
        duration=duration,
        phi_rb=0.0,
        phi_cs=0.0,
        ratio_cs=0.0,
        theta_rb=np.pi,
        theta_cs=0.0,
    )
    print(f"initial infidelity: {1.0 - fidelity:.6e}")


if __name__ == "__main__":
    main()
