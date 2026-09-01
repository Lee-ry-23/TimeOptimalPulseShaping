from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from six_cs_extended_square_validation import (
    ComplexVector,
    FidelityResult,
    Geometry,
    PulseProfile,
    RbSector,
    compute_constant_overlaps,
    compute_shaped_overlaps,
    enumerate_rb_sectors,
    load_phase_profile,
    optimize_local_rb_phase,
    parameters_for_profile,
    validate_geometry,
)


RealVector = NDArray[np.float64]


class SampledSectors(NamedTuple):
    sectors: tuple[RbSector, ...]
    weights: RealVector


class SampledFidelity(NamedTuple):
    result: FidelityResult
    bootstrap_standard_error: float


class SampledCommand(NamedTuple):
    samples_per_active_count: int
    bootstrap_samples: int
    random_seed: int


class ExhaustiveCommand(NamedTuple):
    pass


Command = SampledCommand | ExhaustiveCommand


def eight_cs_geometry() -> Geometry:
    return Geometry(
        rb_count=9,
        cs_count=8,
        rb_cs_pairs=(
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (4, 1),
            (4, 4),
            (5, 2),
            (5, 5),
            (6, 6),
            (6, 7),
            (7, 1),
            (7, 6),
            (8, 2),
            (8, 7),
        ),
        rb_cs_next_nearest_pairs=(
            (0, 4),
            (0, 6),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 5),
            (2, 7),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 2),
            (4, 5),
            (5, 1),
            (5, 3),
            (5, 4),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 2),
            (7, 7),
            (8, 1),
            (8, 3),
            (8, 6),
        ),
        cs_rb_neighbors=(
            (0,),
            (0, 1, 4, 7),
            (1, 2, 5, 8),
            (2,),
            (3, 4),
            (3, 5),
            (6, 7),
            (6, 8),
        ),
        rb_nearest_pairs=(
            (0, 4),
            (0, 7),
            (1, 4),
            (1, 7),
            (1, 5),
            (1, 8),
            (2, 5),
            (2, 8),
            (3, 4),
            (3, 5),
            (6, 7),
            (6, 8),
        ),
    )


def stratified_sector_sample(
    geometry: Geometry,
    samples_per_active_count: int,
    rng: np.random.Generator,
) -> SampledSectors:
    validate_geometry(geometry=geometry)
    if samples_per_active_count <= 0:
        raise ValueError(
            f"samples_per_active_count must be positive, got {samples_per_active_count}."
        )
    all_sectors = enumerate_rb_sectors(geometry=geometry)
    selected_sectors: list[RbSector] = []
    selected_weights: list[float] = []
    total_sector_count = len(all_sectors)
    for active_count in range(geometry.rb_count + 1):
        stratum = tuple(sector for sector in all_sectors if sum(sector) == active_count)
        selected_count = min(samples_per_active_count, len(stratum))
        selected_indices = np.sort(
            rng.choice(len(stratum), size=selected_count, replace=False)
        )
        sector_weight = len(stratum) / (total_sector_count * selected_count)
        for selected_index in selected_indices:
            selected_sectors.append(stratum[int(selected_index)])
            selected_weights.append(sector_weight)
    weights = np.asarray(selected_weights, dtype=np.float64)
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Stratified weights must sum to one, got {np.sum(weights)}.")
    return SampledSectors(sectors=tuple(selected_sectors), weights=weights)


def weighted_coherent_fidelity(
    overlaps: ComplexVector,
    sectors: tuple[RbSector, ...],
    weights: RealVector,
    theta_rb: float,
) -> float:
    if overlaps.shape != weights.shape or overlaps.shape != (len(sectors),):
        raise ValueError(
            f"Expected matching sector, overlap, and weight sizes; got "
            f"{len(sectors)}, {overlaps.shape}, and {weights.shape}."
        )
    active_counts = np.asarray([sum(sector) for sector in sectors], dtype=np.float64)
    corrected_sum = np.sum(weights * np.exp(-1j * theta_rb * active_counts) * overlaps)
    return float(np.abs(corrected_sum) ** 2)


def optimize_weighted_local_phase(
    overlaps: ComplexVector,
    sampled_sectors: SampledSectors,
    grid_size: int,
) -> FidelityResult:
    if grid_size < 8:
        raise ValueError(f"grid_size must be at least eight, got {grid_size}.")
    phase_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False, dtype=np.float64)
    grid_fidelities = np.array(
        [
            weighted_coherent_fidelity(
                overlaps=overlaps,
                sectors=sampled_sectors.sectors,
                weights=sampled_sectors.weights,
                theta_rb=phase,
            )
            for phase in phase_grid
        ],
        dtype=np.float64,
    )
    best_index = int(np.argmax(grid_fidelities))
    grid_step = 2.0 * np.pi / grid_size
    center = float(phase_grid[best_index])
    optimization = minimize_scalar(
        lambda phase: -weighted_coherent_fidelity(
            overlaps=overlaps,
            sectors=sampled_sectors.sectors,
            weights=sampled_sectors.weights,
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
        fidelity=weighted_coherent_fidelity(
            overlaps=overlaps,
            sectors=sampled_sectors.sectors,
            weights=sampled_sectors.weights,
            theta_rb=theta_rb,
        ),
        theta_rb=theta_rb,
        mean_sector_fidelity=float(np.sum(sampled_sectors.weights * np.abs(overlaps) ** 2)),
    )


def bootstrap_standard_error(
    overlaps: ComplexVector,
    sampled_sectors: SampledSectors,
    bootstrap_samples: int,
    grid_size: int,
    rng: np.random.Generator,
) -> float:
    if bootstrap_samples < 2:
        raise ValueError(f"bootstrap_samples must be at least two, got {bootstrap_samples}.")
    active_counts = np.asarray([sum(sector) for sector in sampled_sectors.sectors], dtype=np.int64)
    fidelity_samples = np.zeros(bootstrap_samples, dtype=np.float64)
    for bootstrap_index in range(bootstrap_samples):
        resampled_indices: list[int] = []
        for active_count in np.unique(active_counts):
            stratum_indices = np.flatnonzero(active_counts == active_count)
            resampled_indices.extend(
                int(index)
                for index in rng.choice(
                    stratum_indices,
                    size=stratum_indices.size,
                    replace=True,
                )
            )
        index_array = np.asarray(resampled_indices, dtype=np.int64)
        resampled = SampledSectors(
            sectors=tuple(sampled_sectors.sectors[index] for index in index_array),
            weights=sampled_sectors.weights[index_array],
        )
        fidelity_samples[bootstrap_index] = optimize_weighted_local_phase(
            overlaps=overlaps[index_array],
            sampled_sectors=resampled,
            grid_size=grid_size,
        ).fidelity
    return float(np.std(fidelity_samples, ddof=1))


def evaluate_sampled_fidelity(
    overlaps: ComplexVector,
    sampled_sectors: SampledSectors,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> SampledFidelity:
    result = optimize_weighted_local_phase(
        overlaps=overlaps,
        sampled_sectors=sampled_sectors,
        grid_size=4096,
    )
    standard_error = bootstrap_standard_error(
        overlaps=overlaps,
        sampled_sectors=sampled_sectors,
        bootstrap_samples=bootstrap_samples,
        grid_size=512,
        rng=rng,
    )
    return SampledFidelity(result=result, bootstrap_standard_error=standard_error)


def print_sampled_result(
    label: str,
    sampled_fidelity: SampledFidelity,
    elapsed_seconds: float,
) -> None:
    result = sampled_fidelity.result
    print(
        f"{label} coherent fidelity: {result.fidelity:.12f} "
        f"+/- {sampled_fidelity.bootstrap_standard_error:.3e} "
        f"(theta_rb={result.theta_rb:.12f})"
    )
    print(f"{label} mean sector fidelity: {result.mean_sector_fidelity:.12f}")
    print(f"{label} elapsed seconds: {elapsed_seconds:.3f}")


def print_exhaustive_result(
    label: str,
    result: FidelityResult,
    elapsed_seconds: float,
) -> None:
    print(
        f"{label} coherent fidelity: {result.fidelity:.12f} "
        f"(theta_rb={result.theta_rb:.12f})"
    )
    print(f"{label} mean sector fidelity: {result.mean_sector_fidelity:.12f}")
    print(f"{label} elapsed seconds: {elapsed_seconds:.3f}")


def load_profiles(example_directory: Path) -> tuple[PulseProfile, ...]:
    return tuple(
        load_phase_profile(path=example_directory / filename)
        for filename in (
            "rb_phase_profile_original.json",
            "rb_phase_profile_high_omega.json",
            "rb_phase_profile_extended_duration.json",
        )
    )


def parse_command() -> Command:
    parser = argparse.ArgumentParser(
        description="Validate four pulse choices on the eight-Cs, nine-Rb geometry."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    sampled_parser = subparsers.add_parser(
        "sampled",
        help="Use active-count-stratified sector sampling with bootstrap uncertainty.",
    )
    sampled_parser.add_argument("--samples-per-active-count", type=int, required=True)
    sampled_parser.add_argument("--bootstrap-samples", type=int, required=True)
    sampled_parser.add_argument("--random-seed", type=int, required=True)
    subparsers.add_parser(
        "exhaustive",
        help="Evaluate all 512 Rb sectors without sampling uncertainty.",
    )
    arguments = parser.parse_args()
    if arguments.command == "sampled":
        return SampledCommand(
            samples_per_active_count=int(arguments.samples_per_active_count),
            bootstrap_samples=int(arguments.bootstrap_samples),
            random_seed=int(arguments.random_seed),
        )
    if arguments.command == "exhaustive":
        return ExhaustiveCommand()
    raise ValueError(f"Unknown command {arguments.command!r}.")


def run_sampled(command: SampledCommand) -> None:
    rng = np.random.default_rng(command.random_seed)
    geometry = eight_cs_geometry()
    sampled_sectors = stratified_sector_sample(
        geometry=geometry,
        samples_per_active_count=command.samples_per_active_count,
        rng=rng,
    )
    example_directory = Path(__file__).resolve().parent
    profiles = load_profiles(example_directory=example_directory)
    print("mode: sampled")
    print("Geometry: eight Cs and nine edge-centered Rb around two stacked squares")
    print(f"full sector count: {2**geometry.rb_count}")
    print(f"stratified simulated sector count: {len(sampled_sectors.sectors)}")

    original_profile = profiles[0]
    original_parameters = parameters_for_profile(profile=original_profile)
    start_time = time.perf_counter()
    constant_sectors, constant_overlaps = compute_constant_overlaps(
        parameters=original_parameters,
        duration=original_profile.duration,
        geometry=geometry,
        sectors=sampled_sectors.sectors,
    )
    if constant_sectors != sampled_sectors.sectors:
        raise RuntimeError("Constant simulation changed the sampled sector ordering.")
    constant_fidelity = evaluate_sampled_fidelity(
        overlaps=constant_overlaps,
        sampled_sectors=sampled_sectors,
        bootstrap_samples=command.bootstrap_samples,
        rng=rng,
    )
    print_sampled_result(
        label="constant",
        sampled_fidelity=constant_fidelity,
        elapsed_seconds=time.perf_counter() - start_time,
    )

    for profile in profiles:
        start_time = time.perf_counter()
        profile_sectors, shaped_overlaps = compute_shaped_overlaps(
            parameters=parameters_for_profile(profile=profile),
            profile=profile,
            geometry=geometry,
            sectors=sampled_sectors.sectors,
        )
        if profile_sectors != sampled_sectors.sectors:
            raise RuntimeError(f"Profile {profile.label} changed the sampled sector ordering.")
        sampled_fidelity = evaluate_sampled_fidelity(
            overlaps=shaped_overlaps,
            sampled_sectors=sampled_sectors,
            bootstrap_samples=command.bootstrap_samples,
            rng=rng,
        )
        print_sampled_result(
            label=profile.label,
            sampled_fidelity=sampled_fidelity,
            elapsed_seconds=time.perf_counter() - start_time,
        )


def run_exhaustive(command: ExhaustiveCommand) -> None:
    _ = command
    geometry = eight_cs_geometry()
    sectors = enumerate_rb_sectors(geometry=geometry)
    example_directory = Path(__file__).resolve().parent
    profiles = load_profiles(example_directory=example_directory)
    print("mode: exhaustive")
    print("Geometry: eight Cs and nine edge-centered Rb around two stacked squares")
    print(f"exhaustive sector count: {len(sectors)}")

    original_profile = profiles[0]
    start_time = time.perf_counter()
    constant_sectors, constant_overlaps = compute_constant_overlaps(
        parameters=parameters_for_profile(profile=original_profile),
        duration=original_profile.duration,
        geometry=geometry,
        sectors=sectors,
    )
    if constant_sectors != sectors:
        raise RuntimeError("Constant simulation changed the exhaustive sector ordering.")
    constant_result = optimize_local_rb_phase(
        overlaps=constant_overlaps,
        sectors=sectors,
        grid_size=4096,
    )
    print_exhaustive_result(
        label="constant",
        result=constant_result,
        elapsed_seconds=time.perf_counter() - start_time,
    )

    for profile in profiles:
        start_time = time.perf_counter()
        profile_sectors, shaped_overlaps = compute_shaped_overlaps(
            parameters=parameters_for_profile(profile=profile),
            profile=profile,
            geometry=geometry,
            sectors=sectors,
        )
        if profile_sectors != sectors:
            raise RuntimeError(f"Profile {profile.label} changed the exhaustive sector ordering.")
        result = optimize_local_rb_phase(
            overlaps=shaped_overlaps,
            sectors=sectors,
            grid_size=4096,
        )
        print_exhaustive_result(
            label=profile.label,
            result=result,
            elapsed_seconds=time.perf_counter() - start_time,
        )


def main() -> None:
    command = parse_command()
    if isinstance(command, SampledCommand):
        run_sampled(command=command)
        return
    run_exhaustive(command=command)


if __name__ == "__main__":
    main()
