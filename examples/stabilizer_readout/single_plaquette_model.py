from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from time_optimal_grape import ControlValues, ManualHamiltonianBlock, basis_state, normalized_state


@dataclass(frozen=True)
class ReadoutParameters:
    omega_rb: float
    b_rb_cs: float
    b_edge: float
    b_diag: float
    rb_stark_shift: float
    cs_stark_shift: float


@dataclass(frozen=True)
class ReducedVector:
    label: str
    amplitudes: dict[frozenset[int], complex]


@dataclass(frozen=True)
class ClassSpec:
    name: str
    representative: str
    active_vertices: tuple[int, ...]
    weight: int
    target_cs_state: str
    reduced_vectors: tuple[ReducedVector, ...]


VERTICES: tuple[int, ...] = (0, 1, 2, 3)
EDGE_PAIRS: frozenset[frozenset[int]] = frozenset(
    {
        frozenset((0, 1)),
        frozenset((1, 2)),
        frozenset((2, 3)),
        frozenset((3, 0)),
    }
)
DIAG_PAIRS: frozenset[frozenset[int]] = frozenset(
    {
        frozenset((0, 2)),
        frozenset((1, 3)),
    }
)
CS_STATES: dict[str, int] = {"1": 0, "r": 1}


def rv(label: str, terms: tuple[tuple[tuple[int, ...], complex], ...]) -> ReducedVector:
    return ReducedVector(
        label=label,
        amplitudes={frozenset(vertices): amplitude for vertices, amplitude in terms},
    )


CLASS_SPECS: tuple[ClassSpec, ...] = (
    ClassSpec(
        name="m0",
        representative="0000",
        active_vertices=(),
        weight=1,
        target_cs_state="+",
        reduced_vectors=(rv("G", (((), 1.0),)),),
    ),
    ClassSpec(
        name="m1",
        representative="1000",
        active_vertices=(0,),
        weight=4,
        target_cs_state="-",
        reduced_vectors=(
            rv("G", (((), 1.0),)),
            rv("R", (((0,), 1.0),)),
        ),
    ),
    ClassSpec(
        name="m2_edge",
        representative="1100",
        active_vertices=(0, 1),
        weight=4,
        target_cs_state="+",
        reduced_vectors=(
            rv("G", (((), 1.0),)),
            rv("W", (((0,), 1 / np.sqrt(2)), ((1,), 1 / np.sqrt(2)))),
            rv("D_e", (((0, 1), 1.0),)),
        ),
    ),
    ClassSpec(
        name="m2_diag",
        representative="1010",
        active_vertices=(0, 2),
        weight=2,
        target_cs_state="+",
        reduced_vectors=(
            rv("G", (((), 1.0),)),
            rv("W", (((0,), 1 / np.sqrt(2)), ((2,), 1 / np.sqrt(2)))),
            rv("D_d", (((0, 2), 1.0),)),
        ),
    ),
    ClassSpec(
        name="m3",
        representative="1110",
        active_vertices=(0, 1, 3),
        weight=4,
        target_cs_state="-",
        reduced_vectors=(
            rv("G", (((), 1.0),)),
            rv("S_a", (((0,), 1.0),)),
            rv("S_b", (((1,), 1 / np.sqrt(2)), ((3,), 1 / np.sqrt(2)))),
            rv("D_e", (((0, 1), 1 / np.sqrt(2)), ((0, 3), 1 / np.sqrt(2)))),
            rv("D_d", (((1, 3), 1.0),)),
            rv("T", (((0, 1, 3), 1.0),)),
        ),
    ),
    ClassSpec(
        name="m4",
        representative="1111",
        active_vertices=(0, 1, 2, 3),
        weight=1,
        target_cs_state="+",
        reduced_vectors=(
            rv("G", (((), 1.0),)),
            rv("W_1", tuple(((vertex,), 0.5) for vertex in VERTICES)),
            rv("D_e", tuple((tuple(pair), 0.5) for pair in EDGE_PAIRS)),
            rv("D_d", tuple((tuple(pair), 1 / np.sqrt(2)) for pair in DIAG_PAIRS)),
            rv("W_3", tuple((tuple(vertex for vertex in VERTICES if vertex != missing), 0.5) for missing in VERTICES)),
            rv("Q", (((0, 1, 2, 3), 1.0),)),
        ),
    ),
)


def all_subsets(vertices: tuple[int, ...]) -> tuple[frozenset[int], ...]:
    subsets: list[frozenset[int]] = []
    for mask in range(2 ** len(vertices)):
        subset = frozenset(vertex for bit, vertex in enumerate(vertices) if mask & (1 << bit))
        subsets.append(subset)
    return tuple(subsets)


def count_pairs(subset: frozenset[int], pairs: frozenset[frozenset[int]]) -> int:
    return sum(1 for pair in pairs if pair.issubset(subset))


def rb_interaction_energy(subset: frozenset[int], parameters: ReadoutParameters) -> float:
    rydberg_count = len(subset)
    return (
        parameters.rb_stark_shift * rydberg_count
        + parameters.b_edge * count_pairs(subset=subset, pairs=EDGE_PAIRS)
        + parameters.b_diag * count_pairs(subset=subset, pairs=DIAG_PAIRS)
    )


def reduced_projection(spec: ClassSpec) -> np.ndarray:
    raw_subsets = all_subsets(vertices=spec.active_vertices)
    raw_index = {subset: index for index, subset in enumerate(raw_subsets)}
    projection = np.zeros((len(raw_subsets), len(spec.reduced_vectors)), dtype=np.complex128)
    for column_index, vector in enumerate(spec.reduced_vectors):
        for subset, amplitude in vector.amplitudes.items():
            projection[raw_index[subset], column_index] = amplitude
    gram = projection.conj().T @ projection
    if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-12):
        raise ValueError(f"Reduced basis for {spec.name} is not orthonormal.")
    return projection


def raw_rb_matrices(
    spec: ClassSpec,
    control_values: ControlValues,
    sample_index: int,
    parameters: ReadoutParameters,
) -> tuple[np.ndarray, np.ndarray]:
    raw_subsets = all_subsets(vertices=spec.active_vertices)
    raw_index = {subset: index for index, subset in enumerate(raw_subsets)}
    dimension = len(raw_subsets)
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    derivative_phi_rb = np.zeros_like(hamiltonian)
    phi_rb = control_values.time_controls["phi_rb"][sample_index]
    drive = parameters.omega_rb * np.exp(1j * phi_rb) / 2.0
    drive_derivative = 1j * drive

    for subset in raw_subsets:
        subset_index = raw_index[subset]
        hamiltonian[subset_index, subset_index] = rb_interaction_energy(
            subset=subset,
            parameters=parameters,
        )
        for vertex in spec.active_vertices:
            if vertex not in subset:
                coupled_subset = frozenset(set(subset) | {vertex})
                coupled_index = raw_index[coupled_subset]
                hamiltonian[subset_index, coupled_index] = drive
                hamiltonian[coupled_index, subset_index] = np.conj(drive)
                derivative_phi_rb[subset_index, coupled_index] = drive_derivative
                derivative_phi_rb[coupled_index, subset_index] = np.conj(drive_derivative)

    return hamiltonian, derivative_phi_rb


def reduced_rb_matrices(
    spec: ClassSpec,
    control_values: ControlValues,
    sample_index: int,
    parameters: ReadoutParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projection = reduced_projection(spec=spec)
    raw_hamiltonian, raw_derivative_phi_rb = raw_rb_matrices(
        spec=spec,
        control_values=control_values,
        sample_index=sample_index,
        parameters=parameters,
    )
    raw_subsets = all_subsets(vertices=spec.active_vertices)
    raw_number_operator = np.diag([len(subset) for subset in raw_subsets]).astype(np.complex128)
    return (
        projection.conj().T @ raw_hamiltonian @ projection,
        projection.conj().T @ raw_derivative_phi_rb @ projection,
        projection.conj().T @ raw_number_operator @ projection,
    )


def cs_hamiltonian(control_values: ControlValues, sample_index: int, parameters: ReadoutParameters) -> np.ndarray:
    phi_cs = control_values.time_controls["phi_cs"][sample_index]
    ratio_cs = control_values.time_controls["ratio_cs"][sample_index]
    drive = ratio_cs * parameters.omega_rb * np.exp(1j * phi_cs) / 2.0
    matrix = np.zeros((2, 2), dtype=np.complex128)
    matrix[CS_STATES["r"], CS_STATES["r"]] = parameters.cs_stark_shift
    matrix[CS_STATES["1"], CS_STATES["r"]] = drive
    matrix[CS_STATES["r"], CS_STATES["1"]] = np.conj(drive)
    return matrix


def d_cs_d_phi_cs(control_values: ControlValues, sample_index: int, parameters: ReadoutParameters) -> np.ndarray:
    phi_cs = control_values.time_controls["phi_cs"][sample_index]
    ratio_cs = control_values.time_controls["ratio_cs"][sample_index]
    drive_derivative = 1j * ratio_cs * parameters.omega_rb * np.exp(1j * phi_cs) / 2.0
    matrix = np.zeros((2, 2), dtype=np.complex128)
    matrix[CS_STATES["1"], CS_STATES["r"]] = drive_derivative
    matrix[CS_STATES["r"], CS_STATES["1"]] = np.conj(drive_derivative)
    return matrix


def d_cs_d_ratio_cs(control_values: ControlValues, sample_index: int, parameters: ReadoutParameters) -> np.ndarray:
    phi_cs = control_values.time_controls["phi_cs"][sample_index]
    drive_derivative = parameters.omega_rb * np.exp(1j * phi_cs) / 2.0
    matrix = np.zeros((2, 2), dtype=np.complex128)
    matrix[CS_STATES["1"], CS_STATES["r"]] = drive_derivative
    matrix[CS_STATES["r"], CS_STATES["1"]] = np.conj(drive_derivative)
    return matrix


def tensor_state(rb_vector: np.ndarray, cs_vector: np.ndarray) -> np.ndarray:
    return np.kron(rb_vector, cs_vector).astype(np.complex128)


def make_block(spec: ClassSpec, parameters: ReadoutParameters) -> ManualHamiltonianBlock:
    rb_dimension = len(spec.reduced_vectors)
    cs_plus = normalized_state((1.0, 1.0))
    cs_minus = normalized_state((1.0, -1.0))
    initial_state = tensor_state(rb_vector=basis_state(rb_dimension, 0), cs_vector=cs_plus)
    target_cs = cs_plus if spec.target_cs_state == "+" else cs_minus
    target_state = tensor_state(rb_vector=basis_state(rb_dimension, 0), cs_vector=target_cs)

    def hamiltonian(control_values: ControlValues, sample_index: int) -> np.ndarray:
        rb_hamiltonian, _d_rb, rb_number = reduced_rb_matrices(
            spec=spec,
            control_values=control_values,
            sample_index=sample_index,
            parameters=parameters,
        )
        cs_h = cs_hamiltonian(
            control_values=control_values,
            sample_index=sample_index,
            parameters=parameters,
        )
        rb_cs = parameters.b_rb_cs * np.kron(rb_number, np.diag([0.0, 1.0]))
        return np.kron(rb_hamiltonian, np.eye(2)) + np.kron(np.eye(rb_dimension), cs_h) + rb_cs

    def d_hamiltonian_d_phi_rb(control_values: ControlValues, sample_index: int) -> np.ndarray:
        _rb_hamiltonian, d_rb, _rb_number = reduced_rb_matrices(
            spec=spec,
            control_values=control_values,
            sample_index=sample_index,
            parameters=parameters,
        )
        return np.kron(d_rb, np.eye(2))

    def d_hamiltonian_d_phi_cs(control_values: ControlValues, sample_index: int) -> np.ndarray:
        return np.kron(
            np.eye(rb_dimension),
            d_cs_d_phi_cs(
                control_values=control_values,
                sample_index=sample_index,
                parameters=parameters,
            ),
        )

    def d_hamiltonian_d_ratio_cs(control_values: ControlValues, sample_index: int) -> np.ndarray:
        return np.kron(
            np.eye(rb_dimension),
            d_cs_d_ratio_cs(
                control_values=control_values,
                sample_index=sample_index,
                parameters=parameters,
            ),
        )

    return ManualHamiltonianBlock(
        name=spec.name,
        weight=spec.weight,
        initial_state=initial_state,
        target_state=target_state,
        hamiltonian=hamiltonian,
        derivatives={
            "phi_rb": d_hamiltonian_d_phi_rb,
            "phi_cs": d_hamiltonian_d_phi_cs,
            "ratio_cs": d_hamiltonian_d_ratio_cs,
        },
        phase_offset=0.0,
        local_phase_coefficients={
            "theta_rb": float(len(spec.active_vertices)),
            "theta_cs": 1.0,
        },
    )


def make_blocks(parameters: ReadoutParameters) -> tuple[ManualHamiltonianBlock, ...]:
    return tuple(make_block(spec=spec, parameters=parameters) for spec in CLASS_SPECS)
