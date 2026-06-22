from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply


ComplexVector = NDArray[np.complex128]
RealVector = NDArray[np.float64]
RbSector = tuple[int, ...]
AtomPair = tuple[int, int]


class Geometry(NamedTuple):
    rb_count: int
    cs_count: int
    rb_cs_pairs: tuple[AtomPair, ...]
    cs_rb_neighbors: tuple[tuple[int, ...], ...]
    rb_nearest_pairs: tuple[AtomPair, ...]


class PulseProfile(NamedTuple):
    label: str
    omega0: float
    reference_omega_rb: float
    duration: float
    phases: RealVector


class Parameters(NamedTuple):
    omega_rb: float
    omega_cs: float
    b_rb_cs: float
    b_rb_nearest: float


class SectorModel(NamedTuple):
    sector: RbSector
    active_rb_count: int
    rb_hamiltonian: csr_matrix
    cs_hamiltonian: csr_matrix
    rb_rydberg_counts: RealVector
    initial_state: ComplexVector
    target_state: ComplexVector


class FidelityResult(NamedTuple):
    fidelity: float
    theta_rb: float
    mean_sector_fidelity: float


def six_cs_geometry() -> Geometry:
    return Geometry(
        rb_count=6,
        cs_count=6,
        rb_cs_pairs=(
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 4),
            (5, 5),
        ),
        cs_rb_neighbors=((0,), (0, 1, 3), (1, 2, 4), (2,), (3, 5), (4, 5)),
        rb_nearest_pairs=((0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)),
    )


def validate_geometry(geometry: Geometry) -> None:
    if geometry.rb_count <= 0 or geometry.cs_count <= 0:
        raise ValueError(
            f"Atom counts must be positive, got Rb={geometry.rb_count}, Cs={geometry.cs_count}."
        )
    if len(geometry.cs_rb_neighbors) != geometry.cs_count:
        raise ValueError(
            f"Expected {geometry.cs_count} Cs neighbor lists, got {len(geometry.cs_rb_neighbors)}."
        )
    for rb_vertex, cs_vertex in geometry.rb_cs_pairs:
        if rb_vertex not in range(geometry.rb_count) or cs_vertex not in range(geometry.cs_count):
            raise ValueError(f"Invalid Rb-Cs pair {(rb_vertex, cs_vertex)} for {geometry}.")
    for first_rb, second_rb in geometry.rb_nearest_pairs:
        invalid_pair = (
            first_rb == second_rb
            or first_rb not in range(geometry.rb_count)
            or second_rb not in range(geometry.rb_count)
        )
        if invalid_pair:
            raise ValueError(f"Invalid nearest Rb pair {(first_rb, second_rb)} for {geometry}.")
    for cs_vertex, neighbors in enumerate(geometry.cs_rb_neighbors):
        expected_neighbors = {
            rb_vertex
            for rb_vertex, pair_cs_vertex in geometry.rb_cs_pairs
            if pair_cs_vertex == cs_vertex
        }
        if set(neighbors) != expected_neighbors:
            raise ValueError(
                f"Cs {cs_vertex} neighbors {neighbors} do not match Rb-Cs pairs {expected_neighbors}."
            )


def validate_parameters(parameters: Parameters) -> None:
    values: tuple[tuple[str, float], ...] = (
        ("omega_rb", parameters.omega_rb),
        ("omega_cs", parameters.omega_cs),
        ("b_rb_cs", parameters.b_rb_cs),
        ("b_rb_nearest", parameters.b_rb_nearest),
    )
    for name, value in values:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if parameters.omega_rb == 0.0 or parameters.omega_cs == 0.0:
        raise ValueError("Both Rabi frequencies must be positive.")


def load_phase_profile(path: Path) -> PulseProfile:
    if not path.exists():
        raise FileNotFoundError(f"Missing shaped Rb phase profile: {path}")
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Phase profile must contain a JSON object, got {type(payload).__name__}.")
    required_keys = {
        "profile_label",
        "omega0",
        "duration",
        "phase_values",
        "reference_omega_rb",
    }
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise KeyError(f"Phase profile {path} is missing keys {sorted(missing_keys)}.")
    label = str(payload["profile_label"])
    omega0 = float(payload["omega0"])
    reference_omega = float(payload["reference_omega_rb"])
    duration = float(payload["duration"])
    phases = np.asarray(payload["phase_values"], dtype=np.float64)
    if not label:
        raise ValueError(f"Profile label must be non-empty in {path}.")
    if omega0 <= 0.0 or reference_omega <= 0.0:
        raise ValueError(
            "Profile frequencies must be positive: "
            f"omega0={omega0}, reference_omega_rb={reference_omega}."
        )
    if duration <= 0.0:
        raise ValueError(f"Profile duration must be positive, got {duration}.")
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError(f"Profile phases must be a non-empty vector, got {phases.shape}.")
    return PulseProfile(
        label=label,
        omega0=omega0,
        reference_omega_rb=reference_omega,
        duration=duration,
        phases=phases.copy(),
    )


def parameters_for_profile(profile: PulseProfile) -> Parameters:
    return Parameters(
        omega_rb=profile.reference_omega_rb,
        omega_cs=2.0 * profile.reference_omega_rb,
        b_rb_cs=10.0 * profile.omega0,
        b_rb_nearest=0.1 * profile.omega0,
    )


def enumerate_rb_sectors(geometry: Geometry) -> tuple[RbSector, ...]:
    validate_geometry(geometry=geometry)
    return tuple(
        tuple((mask >> vertex) & 1 for vertex in range(geometry.rb_count))
        for mask in range(2**geometry.rb_count)
    )


def transition_matrix(qubit_indices: tuple[int, ...], qubit_count: int) -> csr_matrix:
    dimension = 2**qubit_count
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for basis_index in range(dimension):
        for qubit_index in qubit_indices:
            if basis_index & (1 << qubit_index):
                continue
            rydberg_index = basis_index | (1 << qubit_index)
            rows.extend((basis_index, rydberg_index))
            columns.extend((rydberg_index, basis_index))
            values.extend((1.0 + 0.0j, 1.0 + 0.0j))
    return csr_matrix((values, (rows, columns)), shape=(dimension, dimension), dtype=np.complex128)


def interaction_diagonal(
    active_rb_vertices: tuple[int, ...],
    parameters: Parameters,
    geometry: Geometry,
) -> RealVector:
    rb_qubit_by_vertex: dict[int, int] = {
        vertex: qubit_index for qubit_index, vertex in enumerate(active_rb_vertices)
    }
    active_rb_count = len(active_rb_vertices)
    dimension = 2 ** (active_rb_count + geometry.cs_count)
    diagonal = np.zeros(dimension, dtype=np.float64)
    for basis_index in range(dimension):
        energy = 0.0
        for first_vertex, second_vertex in geometry.rb_nearest_pairs:
            if first_vertex not in rb_qubit_by_vertex or second_vertex not in rb_qubit_by_vertex:
                continue
            first_bit = 1 << rb_qubit_by_vertex[first_vertex]
            second_bit = 1 << rb_qubit_by_vertex[second_vertex]
            if basis_index & first_bit and basis_index & second_bit:
                energy += parameters.b_rb_nearest
        for rb_vertex, cs_vertex in geometry.rb_cs_pairs:
            if rb_vertex not in rb_qubit_by_vertex:
                continue
            rb_bit = 1 << rb_qubit_by_vertex[rb_vertex]
            cs_bit = 1 << (active_rb_count + cs_vertex)
            if basis_index & rb_bit and basis_index & cs_bit:
                energy += parameters.b_rb_cs
        diagonal[basis_index] = energy
    return diagonal


def build_initial_and_target_states(
    sector: RbSector,
    active_rb_count: int,
    geometry: Geometry,
) -> tuple[ComplexVector, ComplexVector]:
    dimension = 2 ** (active_rb_count + geometry.cs_count)
    initial_state = np.zeros(dimension, dtype=np.complex128)
    target_state = np.zeros(dimension, dtype=np.complex128)
    target_signs = tuple(
        1.0 if sum(sector[rb_vertex] for rb_vertex in neighbors) % 2 == 0 else -1.0
        for neighbors in geometry.cs_rb_neighbors
    )
    normalization = 1.0 / np.sqrt(2**geometry.cs_count)
    for cs_basis_index in range(2**geometry.cs_count):
        basis_index = cs_basis_index << active_rb_count
        target_amplitude = normalization
        for cs_vertex, sign in enumerate(target_signs):
            if cs_basis_index & (1 << cs_vertex):
                target_amplitude *= sign
        initial_state[basis_index] = normalization
        target_state[basis_index] = target_amplitude
    return initial_state, target_state


def build_sector_model(
    sector: RbSector,
    parameters: Parameters,
    geometry: Geometry,
) -> SectorModel:
    validate_geometry(geometry=geometry)
    if len(sector) != geometry.rb_count:
        raise ValueError(f"Expected {geometry.rb_count} Rb bits, got {len(sector)}.")
    if any(bit not in (0, 1) for bit in sector):
        raise ValueError(f"Rb sector must contain only zero and one, got {sector}.")
    active_rb_vertices = tuple(vertex for vertex, bit in enumerate(sector) if bit == 1)
    active_rb_count = len(active_rb_vertices)
    qubit_count = active_rb_count + geometry.cs_count
    interaction = diags(
        interaction_diagonal(
            active_rb_vertices=active_rb_vertices,
            parameters=parameters,
            geometry=geometry,
        ),
        offsets=0,
        format="csr",
        dtype=np.complex128,
    )
    rb_drive = transition_matrix(tuple(range(active_rb_count)), qubit_count)
    cs_drive = transition_matrix(
        tuple(range(active_rb_count, active_rb_count + geometry.cs_count)), qubit_count
    )
    dimension = 2**qubit_count
    rb_rydberg_counts = np.array(
        [
            sum((basis_index >> qubit_index) & 1 for qubit_index in range(active_rb_count))
            for basis_index in range(dimension)
        ],
        dtype=np.float64,
    )
    initial_state, target_state = build_initial_and_target_states(
        sector=sector,
        active_rb_count=active_rb_count,
        geometry=geometry,
    )
    return SectorModel(
        sector=sector,
        active_rb_count=active_rb_count,
        rb_hamiltonian=interaction + 0.5 * parameters.omega_rb * rb_drive,
        cs_hamiltonian=interaction + 0.5 * parameters.omega_cs * cs_drive,
        rb_rydberg_counts=rb_rydberg_counts,
        initial_state=initial_state,
        target_state=target_state,
    )


def propagate_constant_rb_pulse(
    state: ComplexVector,
    rb_hamiltonian: csr_matrix,
    duration: float,
) -> ComplexVector:
    return np.asarray(expm_multiply(-1j * duration * rb_hamiltonian, state), dtype=np.complex128)


def propagate_shaped_rb_pulse(
    state: ComplexVector,
    rb_hamiltonian: csr_matrix,
    rb_rydberg_counts: RealVector,
    profile: PulseProfile,
) -> ComplexVector:
    delta_t = profile.duration / profile.phases.size
    generator = -1j * delta_t * rb_hamiltonian
    evolved_state = np.asarray(state, dtype=np.complex128).copy()
    for phase in profile.phases:
        phase_diagonal = np.exp(-1j * float(phase) * rb_rydberg_counts)
        rotated_state = np.conj(phase_diagonal) * evolved_state
        evolved_state = phase_diagonal * expm_multiply(generator, rotated_state)
    return np.asarray(evolved_state, dtype=np.complex128)


def constant_sector_overlap(
    model: SectorModel,
    duration: float,
    parameters: Parameters,
) -> complex:
    cs_pi_duration = np.pi / parameters.omega_cs
    constant_state = propagate_constant_rb_pulse(
        state=model.initial_state,
        rb_hamiltonian=model.rb_hamiltonian,
        duration=duration,
    )
    constant_state = np.asarray(
        expm_multiply(-1j * cs_pi_duration * model.cs_hamiltonian, constant_state),
        dtype=np.complex128,
    )
    constant_state = propagate_constant_rb_pulse(
        state=constant_state,
        rb_hamiltonian=model.rb_hamiltonian,
        duration=duration,
    )
    return complex(np.vdot(model.target_state, constant_state))


def shaped_sector_overlap(
    model: SectorModel,
    profile: PulseProfile,
    parameters: Parameters,
) -> complex:
    cs_pi_duration = np.pi / parameters.omega_cs
    shaped_state = propagate_shaped_rb_pulse(
        state=model.initial_state,
        rb_hamiltonian=model.rb_hamiltonian,
        rb_rydberg_counts=model.rb_rydberg_counts,
        profile=profile,
    )
    shaped_state = np.asarray(
        expm_multiply(-1j * cs_pi_duration * model.cs_hamiltonian, shaped_state),
        dtype=np.complex128,
    )
    shaped_state = propagate_shaped_rb_pulse(
        state=shaped_state,
        rb_hamiltonian=model.rb_hamiltonian,
        rb_rydberg_counts=model.rb_rydberg_counts,
        profile=profile,
    )
    return complex(np.vdot(model.target_state, shaped_state))


def compute_constant_overlaps(
    parameters: Parameters,
    duration: float,
    geometry: Geometry,
    sectors: tuple[RbSector, ...],
) -> tuple[tuple[RbSector, ...], ComplexVector]:
    validate_parameters(parameters=parameters)
    validate_geometry(geometry=geometry)
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}.")
    constant_overlaps = np.zeros(len(sectors), dtype=np.complex128)
    for sector_index, sector in enumerate(sectors):
        model = build_sector_model(sector=sector, parameters=parameters, geometry=geometry)
        constant_overlaps[sector_index] = constant_sector_overlap(
            model=model,
            duration=duration,
            parameters=parameters,
        )
    return sectors, constant_overlaps


def compute_shaped_overlaps(
    parameters: Parameters,
    profile: PulseProfile,
    geometry: Geometry,
    sectors: tuple[RbSector, ...],
) -> tuple[tuple[RbSector, ...], ComplexVector]:
    validate_parameters(parameters=parameters)
    validate_geometry(geometry=geometry)
    shaped_overlaps = np.zeros(len(sectors), dtype=np.complex128)
    for sector_index, sector in enumerate(sectors):
        model = build_sector_model(sector=sector, parameters=parameters, geometry=geometry)
        shaped_overlaps[sector_index] = shaped_sector_overlap(
            model=model,
            profile=profile,
            parameters=parameters,
        )
    return sectors, shaped_overlaps


def coherent_fidelity(
    overlaps: ComplexVector,
    sectors: tuple[RbSector, ...],
    theta_rb: float,
) -> float:
    active_counts = np.asarray([sum(sector) for sector in sectors], dtype=np.float64)
    corrected_sum = np.sum(np.exp(-1j * theta_rb * active_counts) * overlaps)
    return float(np.abs(corrected_sum) ** 2 / len(sectors) ** 2)


def optimize_local_rb_phase(
    overlaps: ComplexVector,
    sectors: tuple[RbSector, ...],
    grid_size: int,
) -> FidelityResult:
    if overlaps.shape != (len(sectors),):
        raise ValueError(f"Expected {len(sectors)} overlaps, got shape {overlaps.shape}.")
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
    )


def main() -> None:
    example_directory = Path(__file__).resolve().parent
    geometry = six_cs_geometry()
    sectors = enumerate_rb_sectors(geometry=geometry)
    profile_paths: tuple[Path, ...] = (
        example_directory / "rb_phase_profile_original.json",
        example_directory / "rb_phase_profile_high_omega.json",
        example_directory / "rb_phase_profile_extended_duration.json",
    )
    profiles = tuple(load_phase_profile(path=path) for path in profile_paths)
    omega0_values = np.asarray([profile.omega0 for profile in profiles], dtype=np.float64)
    if not np.allclose(omega0_values, omega0_values[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"All profiles must use one omega0, got {omega0_values.tolist()}.")
    original_profile = profiles[0]
    original_parameters = parameters_for_profile(profile=original_profile)
    sectors, constant_overlaps = compute_constant_overlaps(
        parameters=original_parameters,
        duration=original_profile.duration,
        geometry=geometry,
        sectors=sectors,
    )
    constant_result = optimize_local_rb_phase(
        overlaps=constant_overlaps,
        sectors=sectors,
        grid_size=4096,
    )
    shaped_results: list[tuple[PulseProfile, FidelityResult]] = []
    for profile in profiles:
        profile_sectors, shaped_overlaps = compute_shaped_overlaps(
            parameters=parameters_for_profile(profile=profile),
            profile=profile,
            geometry=geometry,
            sectors=sectors,
        )
        if profile_sectors != sectors:
            raise RuntimeError("Profile simulations produced inconsistent Rb sector ordering.")
        shaped_results.append(
            (
                profile,
                optimize_local_rb_phase(
                    overlaps=shaped_overlaps,
                    sectors=sectors,
                    grid_size=4096,
                ),
            )
        )
    print("Sequence: Rb 2*pi -> simultaneous Cs X(pi) -> Rb 2*pi")
    print("Geometry: six Cs, six edge-centered Rb, one square below the middle top edge")
    print(f"sectors: {len(sectors)}")
    print(
        f"constant coherent fidelity: {constant_result.fidelity:.12f} "
        f"(theta_rb={constant_result.theta_rb:.12f})"
    )
    print(f"constant mean sector fidelity: {constant_result.mean_sector_fidelity:.12f}")
    for profile, result in shaped_results:
        print(
            f"{profile.label} coherent fidelity: {result.fidelity:.12f} "
            f"(theta_rb={result.theta_rb:.12f}, "
            f"omega_rb/omega0={profile.reference_omega_rb / profile.omega0:.6f}, "
            f"T*omega0={profile.duration * profile.omega0:.12f})"
        )
        print(f"{profile.label} mean sector fidelity: {result.mean_sector_fidelity:.12f}")


if __name__ == "__main__":
    main()
