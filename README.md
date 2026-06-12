# Time-Optimal Gate Pulse Shaping

A Python library for manual-block GRAPE optimization of phase-modulated Rydberg gates.

The core idea is simple: the user defines independent Hamiltonian blocks, and the library handles time evolution, fidelity evaluation, gradients, optimization, time-continuation scans, population traces, and robustness scans.

The first implementation intentionally focuses on manual input. It does not try to automatically discover symmetry sectors for arbitrary atom geometries. Notebook examples may use symmetry-reduced bases, but those Hamiltonians are still written explicitly by the user.

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m ipykernel install --user --name time-optimal-grape --display-name "Time Optimal GRAPE"
```

Then open the notebooks and choose the `Time Optimal GRAPE` kernel.

## Examples

- `examples/manual_cz_optimization.ipynb`
  Manual CZ Hamiltonian blocks, initial infidelity evaluation, fixed-time optimization, population tracing, and time-continuation scan.

- `examples/manual_ccz_optimization.ipynb`
  Manual collective CCZ blocks, fixed-time optimization, population tracing, and time-continuation scan.

- `examples/manual_cz_robust_optimization.ipynb`
  Curvature-based robustness optimization against Rabi-frequency fluctuations and a post-optimization Omega sensitivity scan.

- `examples/stabilizer_readout/README.md`
  Design notes for the future five-atom Rb-Cs stabilizer-readout notebook.

- `examples/stabilizer_readout/half_plaquet.ipynb`
  Manual three-atom half-plaquette Rb-Cs stabilizer-readout optimization with the Cs laser fixed off.

- `examples/stabilizer_readout/single_plaquet.ipynb`
  Manual single-plaquette Rb-Cs stabilizer-readout optimization with six symmetry-reduced Hamiltonian blocks.

## Core Workflow

1. Define one or more `ManualHamiltonianBlock` objects.
2. Define a `ControlLayout` that maps the flat optimizer vector to time-dependent controls and local phases.
3. Choose a fidelity objective.
4. Build a `GrapeProblem`.
5. Evaluate the current infidelity or run `GrapeOptimizer`.
6. Optionally scan time, trace populations, or run robust optimization.

## Manual Hamiltonian Blocks

A block describes one independent subspace of the full evolution. It contains:

- `name`: readable block label;
- `weight`: integer multiplicity in the gate fidelity;
- `initial_state`: vector in this block basis;
- `target_state`: vector in this block basis;
- `hamiltonian(control_values, sample_index)`;
- `derivatives`: derivatives such as `{"phi": dH_dphi}`;
- `phase_offset`: fixed target phase, for example `pi` for a CZ/CCZ sector;
- `local_phase_coefficients`: coefficients for optimized local phases.

The block phase used by the fidelity layer is

```text
xi_q = phase_offset_q + sum_j coefficient_{q,j} theta_j
```

For a CZ example:

```text
xi_00 = 0
xi_01 = xi_10 = theta
xi_11 = pi + 2 theta
```

The local phase `theta` does not enter the Hamiltonian. It is an optimization variable used only when the block overlaps are combined into the gate fidelity.

## Controls

`ControlLayout` defines how one flat optimizer vector is unpacked:

```python
layout = ControlLayout(
    time_control_names=("phi",),
    local_phase_names=("theta",),
    samples=50,
)
```

This means the optimizer vector is

```text
[phi_0, phi_1, ..., phi_49, theta]
```

For dual-species or multi-laser problems, use more time controls:

```python
layout = ControlLayout(
    time_control_names=("phi_rb", "phi_cs", "ratio_cs"),
    local_phase_names=("theta_rb", "theta_cs"),
    samples=100,
)
```

The optimizer vector is then ordered as:

```text
[phi_rb[0:N], phi_cs[0:N], ratio_cs[0:N], theta_rb, theta_cs]
```

The core optimizer does not treat `phi_rb`, `phi_cs`, and `ratio_cs` differently. They are all time-dependent controls. The block tells the optimizer how each one enters the Hamiltonian through `dH/dcontrol`.

For example, if

```text
Omega_cs(k) = ratio_cs(k) * Omega_rb
H_ij = Omega_cs(k) exp(i phi_cs(k)) / 2
```

then the manual block should provide:

```text
dH_ij / d phi_cs = i * ratio_cs(k) * Omega_rb * exp(i phi_cs(k)) / 2
dH_ij / d ratio_cs = Omega_rb * exp(i phi_cs(k)) / 2
```

## Toolbox Helpers

The `toolbox` helpers are exported from `time_optimal_grape`:

```python
from time_optimal_grape import (
    assemble_parameters,
    basis_state,
    constant_control_profile,
    dimensionless_time_axis,
    locked_parameter_indices,
    normalized_state,
    phase_sample_axis,
    random_control_profile,
    unwrap_phase_profile,
    zero_control_profile,
)
```

Common uses:

```python
ket_0 = basis_state(3, 0)
psi = normalized_state((1, 0, 0, 0))

initial_parameters = assemble_parameters(
    layout=layout,
    time_controls={"phi": random_control_profile(samples=50, low=-0.05, high=0.05, rng=rng)},
    local_phases={"theta": 0.0},
)
```

`phase_sample_axis(duration, samples)` returns the midpoint time of each pulse slice. `dimensionless_time_axis(duration, samples)` returns the state-evolution times including both endpoints. Durations are treated as dimensionless `T Omega`, so notebook plots label axes as `t Omega` or `T Omega`.

## Evaluate Without Optimization

The same optimizer object can evaluate the current pulse without changing it:

```python
infidelity, gradient = GrapeOptimizer(problem).evaluate(parameters)
fidelity, fidelity_gradient = GrapeOptimizer(problem).evaluate_fidelity(parameters)
```

Use this to check a zero control profile, a random initial profile, or an externally supplied protocol.

## Fixed-Time Optimization

```python
result = GrapeOptimizer(problem).optimize(initial_parameters)

print(result.fidelity)
print(result.infidelity)
print(result.parameters)
```

For standard `GrapeOptimizer`, `result.infidelity` is `1 - result.fidelity`.

## Locked Parameters

Sometimes a control should be held fixed while the other controls are optimized. For example, to test the limit \(\Omega_{\rm Cs}=0\) in a dual-species problem, set the initial `ratio_cs` profile to zero and lock the whole `ratio_cs` time control:

```python
locked_indices = locked_parameter_indices(
    layout=layout,
    time_control_names=("ratio_cs",),
    local_phase_names=(),
)

result = GrapeOptimizer(problem).optimize_with_locked_parameters(
    initial_parameters=initial_parameters,
    locked_parameter_indices=locked_indices,
)
```

Locked entries keep their values from `initial_parameters`. The same mechanism can lock local phases:

```python
locked_indices = locked_parameter_indices(
    layout=layout,
    time_control_names=(),
    local_phase_names=("theta_cs",),
)
```

## Time-Continuation Scan

The time-continuation scanner starts from a long duration and repeatedly decreases `T Omega`, using each optimized profile as the initial guess for the next shorter duration:

```python
scan_settings = TimeScanSettings(
    start_duration=9.0,
    stop_duration=6.0,
    duration_step=0.5,
    infidelity_threshold=1e-3,
    stop_after_threshold_failure=True,
)

scan_points = TimeContinuationScanner(problem, scan_settings).scan(initial_parameters)
```

For a scan with locked controls, use:

```python
scan_points = TimeContinuationScanner(problem, scan_settings).scan_with_locked_parameters(
    initial_parameters=initial_parameters,
    locked_parameter_indices=locked_indices,
)
```

For plotting, sort durations before drawing so the horizontal axis increases from left to right.

## Population Tracing

After a protocol is defined, selected block populations can be traced:

```python
controls = layout.unpack(result.parameters)
trace = trace_populations(
    block=block_11,
    control_values=controls,
    duration=duration,
    samples=samples,
    states={
        "|11>": basis_state(3, 0),
        "|W>": basis_state(3, 1),
        "|rr>": basis_state(3, 2),
    },
)
```

The returned `PopulationTrace` has:

- `times`: dimensionless times from `0` to `T Omega`;
- `populations`: a dictionary from labels to population arrays.

## Robustness Optimization

Robustness optimization is implemented as a finite-difference curvature penalty. The cost is

```text
C = 1 - F(Omega) - eta * d2F/dOmega2
```

with

```text
d2F/dOmega2 ~= [F(Omega + delta) - 2 F(Omega) + F(Omega - delta)] / delta^2
```

The user supplies three compatible problems:

```python
nominal_problem = make_problem(omega)
plus_problem = make_problem(omega + omega_delta)
minus_problem = make_problem(omega - omega_delta)
```

Then optimize:

```python
robust_result = RobustGrapeOptimizer(
    nominal_problem=nominal_problem,
    plus_problem=plus_problem,
    minus_problem=minus_problem,
    robustness_settings=CurvatureRobustnessSettings(
        eta=2e-4,
        finite_difference_step=omega_delta,
    ),
).optimize(initial_parameters)
```

For `RobustGrapeOptimizer`, `result.fidelity` is the nominal fidelity at the optimized parameters. `result.infidelity` stores the robust optimizer cost, not simply `1 - result.fidelity`.

To test a fixed profile against amplitude fluctuations:

```python
omega_values, infidelities = scan_parameter_sensitivity(
    problem_factory=make_problem,
    parameters=robust_result.parameters,
    parameter_values=omega * np.linspace(0.97, 1.03, 17),
)
```

## Numerical Method

For each block and time slice,

```text
psi_{k+1} = exp(-i H_k dt) psi_k
```

Gradients use the Frechet derivative of the matrix exponential via SciPy. For one time-dependent control,

```text
d a_q / d x_k =
<target_q|U_N ... U_{k+1} (dU_k/dx_k) U_{k-1} ... U_1|initial_q>
```

The fidelity layer maps overlap gradients to the final objective gradient.

## Current Non-Goals

- Automatic symmetry discovery for arbitrary atom geometries.
- Open-system Lindblad evolution.
- Experimental calibration loops.
- GPU acceleration.

These can be added later without changing the manual-block workflow.
