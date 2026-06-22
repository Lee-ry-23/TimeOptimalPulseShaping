from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]
RbSector = tuple[int, int, int, int]
AtomPair = tuple[int, int]

RB_COUNT: int = 4
CS_COUNT: int = 4
RB_NEAREST_PAIRS: tuple[AtomPair, ...] = ((0, 1), (1, 2), (2, 3), (3, 0))
RB_DIAGONAL_PAIRS: tuple[AtomPair, ...] = ((0, 2), (1, 3))
RB_CS_PAIRS: tuple[AtomPair, ...] = (
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 3),
    (3, 3),
    (3, 0),
)
CS_RB_NEIGHBORS: tuple[tuple[int, int], ...] = ((3, 0), (0, 1), (1, 2), (2, 3))


class PulseProfile(NamedTuple):
    duration: float
    phases: RealVector


class Parameters(NamedTuple):
    omega_rb: float
    omega_cs: float
    b_rb_cs: float
    b_rb_nearest: float
    b_rb_diagonal: float


class SectorModel(NamedTuple):
    sector: RbSector
    active_rb_vertices: tuple[int, ...]
    rb_hamiltonian: ComplexMatrix
    cs_hamiltonian: ComplexMatrix
    rb_rydberg_counts: RealVector
    initial_state: ComplexVector
    target_state: ComplexVector


class FidelityResult(NamedTuple):
    fidelity: float
    theta_rb: float
    mean_sector_fidelity: float
    overlaps: ComplexVector


def validate_parameters(parameters: Parameters) -> None:
    values: tuple[tuple[str, float], ...] = (
        ("omega_rb", parameters.omega_rb),
        ("omega_cs", parameters.omega_cs),
        ("b_rb_cs", parameters.b_rb_cs),
        ("b_rb_nearest", parameters.b_rb_nearest),
        ("b_rb_diagonal", parameters.b_rb_diagonal),
    )
    for name, value in values:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if parameters.omega_rb == 0.0 or parameters.omega_cs == 0.0:
        raise ValueError("Both Rabi frequencies must be positive.")


def load_phase_profile(path: Path, omega_rb: float) -> PulseProfile:
    if not path.exists():
        raise FileNotFoundError(f"Missing shaped Rb phase profile: {path}")
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Phase profile must contain a JSON object, got {type(payload).__name__}.")
    required_keys = {"duration", "phase_values", "reference_omega_rb"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise KeyError(f"Phase profile {path} is missing keys {sorted(missing_keys)}.")
    reference_omega = float(payload["reference_omega_rb"])
    if not np.isclose(reference_omega, omega_rb, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Profile reference omega {reference_omega} does not match omega_rb {omega_rb}."
        )
    duration = float(payload["duration"])
    phases = np.asarray(payload["phase_values"], dtype=np.float64)
    if duration <= 0.0:
        raise ValueError(f"Profile duration must be positive, got {duration}.")
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError(f"Profile phases must be a non-empty vector, got {phases.shape}.")
    return PulseProfile(duration=duration, phases=phases.copy())


def enumerate_rb_sectors() -> tuple[RbSector, ...]:
    return tuple(
        tuple((mask >> vertex) & 1 for vertex in range(RB_COUNT))
        for mask in range(2**RB_COUNT)
    )


def transition_matrix(qubit_indices: tuple[int, ...], qubit_count: int) -> ComplexMatrix:
    dimension = 2**qubit_count
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis_index in range(dimension):
        for qubit_index in qubit_indices:
            if basis_index & (1 << qubit_index):
                continue
            rydberg_index = basis_index | (1 << qubit_index)
            matrix[basis_index, rydberg_index] += 1.0
            matrix[rydberg_index, basis_index] += 1.0
    return matrix


def interaction_diagonal(
    active_rb_vertices: tuple[int, ...],
    parameters: Parameters,
) -> RealVector:
    rb_qubit_by_vertex: dict[int, int] = {
        vertex: qubit_index for qubit_index, vertex in enumerate(active_rb_vertices)
    }
    rb_qubit_count = len(active_rb_vertices)
    dimension = 2 ** (rb_qubit_count + CS_COUNT)
    diagonal = np.zeros(dimension, dtype=np.float64)
    for basis_index in range(dimension):
        energy = 0.0
        for first_vertex, second_vertex in RB_NEAREST_PAIRS:
            if first_vertex not in rb_qubit_by_vertex or second_vertex not in rb_qubit_by_vertex:
                continue
            first_bit = 1 << rb_qubit_by_vertex[first_vertex]
            second_bit = 1 << rb_qubit_by_vertex[second_vertex]
            if basis_index & first_bit and basis_index & second_bit:
                energy += parameters.b_rb_nearest
        for first_vertex, second_vertex in RB_DIAGONAL_PAIRS:
            if first_vertex not in rb_qubit_by_vertex or second_vertex not in rb_qubit_by_vertex:
                continue
            first_bit = 1 << rb_qubit_by_vertex[first_vertex]
            second_bit = 1 << rb_qubit_by_vertex[second_vertex]
            if basis_index & first_bit and basis_index & second_bit:
                energy += parameters.b_rb_diagonal
        for rb_vertex, cs_vertex in RB_CS_PAIRS:
            if rb_vertex not in rb_qubit_by_vertex:
                continue
            rb_bit = 1 << rb_qubit_by_vertex[rb_vertex]
            cs_bit = 1 << (rb_qubit_count + cs_vertex)
            if basis_index & rb_bit and basis_index & cs_bit:
                energy += parameters.b_rb_cs
        diagonal[basis_index] = energy
    return diagonal


def build_initial_and_target_states(
    sector: RbSector,
    active_rb_count: int,
) -> tuple[ComplexVector, ComplexVector]:
    dimension = 2 ** (active_rb_count + CS_COUNT)
    initial_state = np.zeros(dimension, dtype=np.complex128)
    target_state = np.zeros(dimension, dtype=np.complex128)
    target_signs = tuple(
        1.0 if (sector[first_rb] + sector[second_rb]) % 2 == 0 else -1.0
        for first_rb, second_rb in CS_RB_NEIGHBORS
    )
    normalization = 1.0 / np.sqrt(2**CS_COUNT)
    for cs_basis_index in range(2**CS_COUNT):
        basis_index = cs_basis_index << active_rb_count
        target_amplitude = normalization
        for cs_vertex, sign in enumerate(target_signs):
            if cs_basis_index & (1 << cs_vertex):
                target_amplitude *= sign
        initial_state[basis_index] = normalization
        target_state[basis_index] = target_amplitude
    return initial_state, target_state


def build_sector_model(sector: RbSector, parameters: Parameters) -> SectorModel:
    if any(bit not in (0, 1) for bit in sector):
        raise ValueError(f"Rb sector must contain only zero and one, got {sector}.")
    active_rb_vertices = tuple(vertex for vertex, bit in enumerate(sector) if bit == 1)
    active_rb_count = len(active_rb_vertices)
    qubit_count = active_rb_count + CS_COUNT
    interaction = np.diag(
        interaction_diagonal(active_rb_vertices=active_rb_vertices, parameters=parameters)
    ).astype(np.complex128)
    rb_drive = transition_matrix(tuple(range(active_rb_count)), qubit_count)
    cs_drive = transition_matrix(
        tuple(range(active_rb_count, active_rb_count + CS_COUNT)), qubit_count
    )
    rb_rydberg_counts = np.array(
        [
            sum((basis_index >> qubit_index) & 1 for qubit_index in range(active_rb_count))
            for basis_index in range(2**qubit_count)
        ],
        dtype=np.float64,
    )
    initial_state, target_state = build_initial_and_target_states(
        sector=sector,
        active_rb_count=active_rb_count,
    )
    return SectorModel(
        sector=sector,
        active_rb_vertices=active_rb_vertices,
        rb_hamiltonian=interaction + 0.5 * parameters.omega_rb * rb_drive,
        cs_hamiltonian=interaction + 0.5 * parameters.omega_cs * cs_drive,
        rb_rydberg_counts=rb_rydberg_counts,
        initial_state=initial_state,
        target_state=target_state,
    )


def propagate_constant_rb_pulse(
    state: ComplexVector,
    zero_phase_slice_unitary: ComplexMatrix,
    samples: int,
) -> ComplexVector:
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}.")
    return np.linalg.matrix_power(zero_phase_slice_unitary, samples) @ state


def propagate_shaped_rb_pulse(
    state: ComplexVector,
    zero_phase_slice_unitary: ComplexMatrix,
    rb_rydberg_counts: RealVector,
    phases: RealVector,
) -> ComplexVector:
    evolved_state = np.asarray(state, dtype=np.complex128).copy()
    for phase in phases:
        phase_diagonal = np.exp(-1j * float(phase) * rb_rydberg_counts)
        evolved_state = phase_diagonal * (
            zero_phase_slice_unitary @ (np.conj(phase_diagonal) * evolved_state)
        )
    return evolved_state


def sector_overlaps(
    model: SectorModel,
    profile: PulseProfile,
    parameters: Parameters,
) -> tuple[complex, complex]:
    rb_delta_t = profile.duration / profile.phases.size
    rb_slice_unitary = expm(-1j * rb_delta_t * model.rb_hamiltonian)
    cs_pi_duration = np.pi / parameters.omega_cs
    cs_pi_unitary = expm(-1j * cs_pi_duration * model.cs_hamiltonian)

    constant_state = propagate_constant_rb_pulse(
        state=model.initial_state,
        zero_phase_slice_unitary=rb_slice_unitary,
        samples=int(profile.phases.size),
    )
    constant_state = cs_pi_unitary @ constant_state
    constant_state = propagate_constant_rb_pulse(
        state=constant_state,
        zero_phase_slice_unitary=rb_slice_unitary,
        samples=int(profile.phases.size),
    )

    shaped_state = propagate_shaped_rb_pulse(
        state=model.initial_state,
        zero_phase_slice_unitary=rb_slice_unitary,
        rb_rydberg_counts=model.rb_rydberg_counts,
        phases=profile.phases,
    )
    shaped_state = cs_pi_unitary @ shaped_state
    shaped_state = propagate_shaped_rb_pulse(
        state=shaped_state,
        zero_phase_slice_unitary=rb_slice_unitary,
        rb_rydberg_counts=model.rb_rydberg_counts,
        phases=profile.phases,
    )
    return (
        complex(np.vdot(model.target_state, constant_state)),
        complex(np.vdot(model.target_state, shaped_state)),
    )


def compute_overlaps(
    parameters: Parameters,
    profile: PulseProfile,
) -> tuple[tuple[RbSector, ...], ComplexVector, ComplexVector]:
    validate_parameters(parameters=parameters)
    sectors = enumerate_rb_sectors()
    constant_overlaps = np.zeros(len(sectors), dtype=np.complex128)
    shaped_overlaps = np.zeros(len(sectors), dtype=np.complex128)
    for sector_index, sector in enumerate(sectors):
        model = build_sector_model(sector=sector, parameters=parameters)
        constant_overlap, shaped_overlap = sector_overlaps(
            model=model,
            profile=profile,
            parameters=parameters,
        )
        constant_overlaps[sector_index] = constant_overlap
        shaped_overlaps[sector_index] = shaped_overlap
    return sectors, constant_overlaps, shaped_overlaps


def coherent_fidelity(
    overlaps: ComplexVector,
    sectors: tuple[RbSector, ...],
    theta_rb: float,
) -> float:
    overlap_vector = np.asarray(overlaps, dtype=np.complex128)
    if overlap_vector.shape != (len(sectors),):
        raise ValueError(
            f"Overlap shape {overlap_vector.shape} does not match {len(sectors)} sectors."
        )
    active_counts = np.asarray([sum(sector) for sector in sectors], dtype=np.float64)
    corrected_sum = np.sum(np.exp(-1j * theta_rb * active_counts) * overlap_vector)
    return float(np.abs(corrected_sum) ** 2 / len(sectors) ** 2)


def optimize_local_rb_phase(
    overlaps: ComplexVector,
    sectors: tuple[RbSector, ...],
    grid_size: int,
) -> FidelityResult:
    if grid_size < 8:
        raise ValueError(f"grid_size must be at least eight, got {grid_size}.")
    phase_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False, dtype=np.float64)
    grid_fidelities = np.array(
        [coherent_fidelity(overlaps=overlaps, sectors=sectors, theta_rb=phase) for phase in phase_grid],
        dtype=np.float64,
    )
    best_index = int(np.argmax(grid_fidelities))
    grid_step = 2.0 * np.pi / grid_size
    center = float(phase_grid[best_index])
    optimization = minimize_scalar(
        lambda phase: -coherent_fidelity(
            overlaps=overlaps,
            sectors=sectors,
            theta_rb=float(phase),
        ),
        bounds=(center - grid_step, center + grid_step),
        method="bounded",
        options={"xatol": 1e-14},
    )
    if not optimization.success:
        raise RuntimeError(f"Local-phase optimization failed: {optimization.message}")
    theta_rb = float((optimization.x + np.pi) % (2.0 * np.pi) - np.pi)
    return FidelityResult(
        fidelity=coherent_fidelity(overlaps=overlaps, sectors=sectors, theta_rb=theta_rb),
        theta_rb=theta_rb,
        mean_sector_fidelity=float(np.mean(np.abs(overlaps) ** 2)),
        overlaps=np.asarray(overlaps, dtype=np.complex128).copy(),
    )


def main() -> None:
    omega_rb = 1.0
    parameters = Parameters(
        omega_rb=omega_rb,
        omega_cs=2.0 * omega_rb,
        b_rb_cs=10.0 * omega_rb,
        b_rb_nearest=0.1 * omega_rb,
        b_rb_diagonal=0.01 * omega_rb,
    )
    profile_path = Path(__file__).resolve().parent / "rb_phase_profile_original.json"
    profile = load_phase_profile(path=profile_path, omega_rb=omega_rb)
    if not np.isclose(profile.duration * omega_rb, 2.0 * np.pi, rtol=0.0, atol=1e-10):
        raise ValueError(
            "The validation requires a 2*pi Rb profile: "
            f"duration*omega_rb={profile.duration * omega_rb}."
        )
    sectors, constant_overlaps, shaped_overlaps = compute_overlaps(
        parameters=parameters,
        profile=profile,
    )
    constant_result = optimize_local_rb_phase(
        overlaps=constant_overlaps,
        sectors=sectors,
        grid_size=4096,
    )
    shaped_result = optimize_local_rb_phase(
        overlaps=shaped_overlaps,
        sectors=sectors,
        grid_size=4096,
    )
    print("Sequence: Rb 2*pi -> simultaneous Cs X(pi) -> Rb 2*pi")
    print(f"sectors: {len(sectors)}")
    print(
        f"constant coherent fidelity: {constant_result.fidelity:.12f} "
        f"(theta_rb={constant_result.theta_rb:.12f})"
    )
    print(f"constant mean sector fidelity: {constant_result.mean_sector_fidelity:.12f}")
    print(
        f"shaped coherent fidelity:   {shaped_result.fidelity:.12f} "
        f"(theta_rb={shaped_result.theta_rb:.12f})"
    )
    print(f"shaped mean sector fidelity:   {shaped_result.mean_sector_fidelity:.12f}")


if __name__ == "__main__":
    main()
