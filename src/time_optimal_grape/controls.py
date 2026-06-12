from dataclasses import dataclass

import numpy as np

from time_optimal_grape.typing import RealVector


@dataclass(frozen=True)
class ControlValues:
    time_controls: dict[str, RealVector]
    local_phases: dict[str, float]

    @property
    def phases(self) -> dict[str, RealVector]:
        return self.time_controls


@dataclass(frozen=True)
class ControlLayout:
    time_control_names: tuple[str, ...]
    local_phase_names: tuple[str, ...]
    samples: int

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("ControlLayout.samples must be positive.")
        if not self.time_control_names:
            raise ValueError("ControlLayout.time_control_names cannot be empty.")
        if len(set(self.time_control_names)) != len(self.time_control_names):
            raise ValueError("ControlLayout.time_control_names contains duplicate names.")
        if len(set(self.local_phase_names)) != len(self.local_phase_names):
            raise ValueError("ControlLayout.local_phase_names contains duplicate names.")
        duplicated_names = set(self.time_control_names).intersection(self.local_phase_names)
        if duplicated_names:
            raise ValueError(f"Control names cannot be both time control and local phase: {sorted(duplicated_names)}.")

    @property
    def parameter_count(self) -> int:
        return len(self.time_control_names) * self.samples + len(self.local_phase_names)

    @property
    def phase_names(self) -> tuple[str, ...]:
        return self.time_control_names

    def unpack(self, vector: RealVector) -> ControlValues:
        real_vector = np.asarray(vector, dtype=np.float64)
        if real_vector.shape != (self.parameter_count,):
            raise ValueError(
                f"Expected control vector with shape {(self.parameter_count,)}, got {real_vector.shape}."
            )

        time_controls: dict[str, RealVector] = {}
        offset = 0
        for time_control_name in self.time_control_names:
            time_controls[time_control_name] = real_vector[offset : offset + self.samples].copy()
            offset += self.samples

        local_phases: dict[str, float] = {}
        for local_phase_name in self.local_phase_names:
            local_phases[local_phase_name] = float(real_vector[offset])
            offset += 1

        return ControlValues(time_controls=time_controls, local_phases=local_phases)

    def time_control_parameter_index(self, time_control_name: str, sample_index: int) -> int:
        if time_control_name not in self.time_control_names:
            raise ValueError(f"Unknown time control {time_control_name!r}.")
        if sample_index < 0 or sample_index >= self.samples:
            raise ValueError(f"Sample index {sample_index} is outside [0, {self.samples}).")
        time_control_offset = self.time_control_names.index(time_control_name) * self.samples
        return time_control_offset + sample_index

    def phase_parameter_index(self, phase_name: str, sample_index: int) -> int:
        return self.time_control_parameter_index(time_control_name=phase_name, sample_index=sample_index)

    def local_phase_parameter_index(self, local_phase_name: str) -> int:
        if local_phase_name not in self.local_phase_names:
            raise ValueError(f"Unknown local phase {local_phase_name!r}.")
        return len(self.time_control_names) * self.samples + self.local_phase_names.index(local_phase_name)
