# Stabilizer Readout Examples

This folder contains dual-species stabilizer-readout examples. The smaller half-plaquette case is described first because it uses the same physics as the five-atom plaquette with fewer symmetry classes.

## Files

- `half_plaquet.ipynb`: three-atom diagonal half-plaquette readout with the Cs drive locked off.
- `single_plaquet.ipynb`: five-atom single-plaquette readout, time scan, population traces, and robustness diagnostics.
- `rb_gaussian_stark_dictionary.ipynb`: ARC-based preprocessing notebook that computes `omega_stark_values.json` from Gaussian-beam intensity variations.
- `two_plaquette/`: nine-atom, 128-sector forward validation of shaped and unshaped `Rb 2*pi -> Cs pi -> Rb 2*pi` sequences.

The generated `omega_stark_values.json` is read by `single_plaquet.ipynb`. Its keys are displacements $r/\sigma$, and each value is already normalized to GRAPE units:

```math
\omega_{\rm rb}=\frac{|\Omega_R(r)|}{|\Omega_R(0)|},
\qquad
\Delta_{\rm rb}=\frac{[\Delta_g(r)+\Delta_r(r)]-[\Delta_g(0)+\Delta_r(0)]}{|\Omega_R(0)|}.
```

The preprocessing notebook fixes the single-photon detuning to $\Delta/2\pi=2.3\,\mathrm{GHz}$, computes the center Stark shift as the reference, and subtracts that reference while scanning position.

# Three-Atom Half-Plaquette Readout

The half-plaquette geometry keeps only two diagonal Rb data atoms and the central Cs ancilla:

- two Rb atoms on opposite vertices of the square;
- one Cs atom at the square center;
- all active Rb Rydberg transitions are driven simultaneously;
- the Cs laser is off throughout the pulse, so $\Omega_{\rm Cs}=0$ and $\phi_{\rm Cs}$ is not an optimizer variable.

The target operation is still a parity readout. The two Rb atoms should return to their initial computational state. The Cs atom starts in

```math
|+\rangle_{\rm Cs}=\frac{|1\rangle_{\rm Cs}+|r\rangle_{\rm Cs}}{\sqrt{2}},
```

and should end in

```math
\begin{cases}
|+\rangle_{\rm Cs}, & \text{even Rb parity},\\
|-\rangle_{\rm Cs}=\frac{|1\rangle_{\rm Cs}-|r\rangle_{\rm Cs}}{\sqrt{2}}, & \text{odd Rb parity}.
\end{cases}
```

There are $2^2=4$ Rb computational bitstrings. The two diagonal Rb atoms are equivalent under exchange, so the four cases reduce to three classes:

| Class | Representative Rb bitstring | Active Rb atoms | Weight | Parity |
| --- | --- | ---: | ---: | --- |
| `m0` | `00` | 0 | 1 | even |
| `m1` | `10` | 1 | 2 | odd |
| `m2_diag` | `11` | 2 diagonal | 1 | even |

The weights add to 4.

## Half-Plaquette Controls

The control layout is intentionally smaller than the five-atom example:

```python
layout = ControlLayout(
    time_control_names=("phi_rb",),
    local_phase_names=("theta_rb",),
    samples=samples,
)
```

The only time-dependent Hamiltonian control is $\phi_{\rm Rb}(k)$. The Rb Rabi frequency is constant:

```math
\Omega_{\rm Rb}(k)=\Omega_{\rm Rb},
```

and the Rb drive is

```math
v(k)=\frac{\Omega_{\rm Rb}e^{i\phi_{\rm Rb}(k)}}{2}.
```

The Cs Hamiltonian is zero in the ordered basis $(|1\rangle,|r\rangle)$:

```math
H_{\rm Cs}=0_{2\times 2}.
```

The Cs state still participates in the block through the Rb-Cs interaction.

## Half-Plaquette Shared Hamiltonian

For each class and time slice $k$,

```math
H_k = H_{\rm Rb}(k)\otimes I_{\rm Cs}+H_{\rm int}.
```

The interaction energy for a reduced Rb state with $n_R$ Rydberg excitations and Cs state $c\in\{1,r\}$ is

```math
E_{\rm int}(n_R,c)=n_R B_{\rm RbCs}\mathbf{1}_{c=r}.
```

For the two-active-Rb class, the diagonal Rb-Rb interaction is

```math
B_d
```

when both diagonal Rb atoms are in $|r\rangle$.

The only Hamiltonian derivative required by GRAPE is

```math
\frac{\partial}{\partial \phi_{\rm Rb}}
\left(v|\alpha\rangle\langle\beta|+v^*|\beta\rangle\langle\alpha|\right)
=
iv|\alpha\rangle\langle\beta|
-iv^*|\beta\rangle\langle\alpha|.
```

## Half-Plaquette Class `m0`: Representative `00`

There are no active Rb atoms. The reduced Rb basis is

```math
|G\rangle.
```

The full Hamiltonian is zero:

```math
H^{(m0)}=0_{2\times 2},
```

where the dimension 2 is the Cs $(|1\rangle,|r\rangle)$ branch. The initial and target states are both

```math
|G\rangle_{\rm Rb}\otimes |+\rangle_{\rm Cs}.
```

## Half-Plaquette Class `m1`: Representative `10`

There is one active Rb atom. The reduced Rb basis is

```math
|G\rangle,\quad |R\rangle.
```

The Rb Hamiltonian is

```math
H_{\rm Rb}^{(m1)}(k)
=
\begin{pmatrix}
0 & v(k)\\
v^*(k) & 0
\end{pmatrix}.
```

The Rb-Cs interaction adds

```math
0,\quad B_{\rm RbCs}\mathbf{1}_{c=r}
```

to the diagonal entries for $|G,c\rangle$ and $|R,c\rangle$, respectively. The target Cs state is $|-\rangle_{\rm Cs}$.

## Half-Plaquette Class `m2_diag`: Representative `11`

There are two active diagonal Rb atoms. Use the symmetric Rb basis

```math
|G\rangle=|11\rangle,
\qquad
|W\rangle=\frac{|r1\rangle+|1r\rangle}{\sqrt{2}},
\qquad
|D_d\rangle=|rr\rangle.
```

The reduced Rb Hamiltonian is

```math
H_{\rm Rb}^{(m2d)}(k)=
\begin{pmatrix}
0 & \sqrt{2}v(k) & 0\\
\sqrt{2}v^*(k) & 0 & \sqrt{2}v(k)\\
0 & \sqrt{2}v^*(k) & B_d
\end{pmatrix}.
```

The Rb-Cs interaction adds

```math
0,\quad B_{\rm RbCs}\mathbf{1}_{c=r},\quad 2B_{\rm RbCs}\mathbf{1}_{c=r}
```

to the diagonal entries for $|G,c\rangle$, $|W,c\rangle$, and $|D_d,c\rangle$, respectively. The target Cs state is $|+\rangle_{\rm Cs}$.

## Half-Plaquette Fidelity

The half-plaquette fidelity uses the coherent weighted overlap

```math
F=\frac{
\left|\sum_q w_q e^{-i\xi_q}
\langle\psi_q^{\rm target}|U_q|\psi_q^0\rangle\right|^2
}{4^2},
```

with weights $1,2,1$. The local phase convention used in the notebook is

```math
\xi_q=n_{\rm Rb}(q)\theta_{\rm Rb}.
```

The parity-dependent Cs sign is part of the target state, not part of $\xi_q$.

# Five-Atom Single-Plaquette Readout

The physical layout is a square plaquette:

- four Rb atoms at the square vertices;
- one Cs atom at the square center;
- all Rb Rydberg transitions are driven simultaneously;
- the Cs Rydberg transition is also driven simultaneously;
- the Rb laser phase, Cs laser phase, and Cs/Rb amplitude ratio may all be time-dependent controls.

The target operation is a parity readout on the Cs ancilla. The four Rb atoms should return to their initial computational state. The Cs atom starts in

```math
|+\rangle_{\rm Cs}=\frac{|1\rangle_{\rm Cs}+|r\rangle_{\rm Cs}}{\sqrt{2}},
```

and should end in

```math
\begin{cases}
|+\rangle_{\rm Cs}, & \text{even Rb parity},\\
|-\rangle_{\rm Cs}=\frac{|1\rangle_{\rm Cs}-|r\rangle_{\rm Cs}}{\sqrt{2}}, & \text{odd Rb parity}.
\end{cases}
```

This is the surface-code stabilizer readout condition: the four Rb data atoms collectively apply a parity-controlled operation to the Cs ancilla.

## Computational Cases

There are $2^4=16$ Rb computational bitstrings.

Using the square symmetry, these 16 cases reduce to 6 inequivalent classes:

| Class | Representative Rb bitstring | Active Rb atoms | Weight | Parity |
| --- | --- | ---: | ---: | --- |
| `m0` | `0000` | 0 | 1 | even |
| `m1` | `1000` | 1 | 4 | odd |
| `m2_edge` | `1100` | 2 adjacent | 4 | even |
| `m2_diag` | `1010` | 2 diagonal | 2 | even |
| `m3` | `1110` | 3 | 4 | odd |
| `m4` | `1111` | 4 | 1 | even |

The weights add to 16.

Only active Rb atoms, meaning atoms initially in $|1\rangle$, participate in Rydberg dynamics. Rb atoms initially in $|0\rangle$ are frozen and are omitted from each block basis.

## Controls

The planned control layout is:

```python
layout = ControlLayout(
    time_control_names=("phi_rb", "phi_cs", "ratio_cs"),
    local_phase_names=("theta_rb", "theta_cs"),
    samples=samples,
)
```

The Rabi frequencies are interpreted as

```math
\Omega_{\rm Rb}(k)=\Omega_{\rm Rb},
\qquad
\Omega_{\rm Cs}(k)=\rho(k)\Omega_{\rm Rb},
```

where

```math
\rho(k)=\texttt{ratio\_cs}[k].
```

The complex drives are

```math
\Omega_{\rm Rb}e^{i\phi_{\rm Rb}(k)},
\qquad
\rho(k)\Omega_{\rm Rb}e^{i\phi_{\rm Cs}(k)}.
```

## Interaction Parameters

There are three interaction strengths and two differential AC Stark shifts:

```math
B_{\rm RbCs},\qquad B_e,\qquad B_d,\qquad
\Delta_{\rm Rb},\qquad \Delta_{\rm Cs}.
```

Here:

- $B_{\rm RbCs}$ is the Rb-Cs Rydberg interaction;
- $B_e$ is the Rb-Rb nearest-neighbor edge interaction;
- $B_d$ is the Rb-Rb diagonal interaction.
- $\Delta_{\rm Rb}$ is the Rb differential AC Stark shift on $|r\rangle_{\rm Rb}$;
- $\Delta_{\rm Cs}$ is the Cs differential AC Stark shift on $|r\rangle_{\rm Cs}$.

For a raw Rb Rydberg subset $A$ and Cs state $c\in\{1,r\}$, the diagonal interaction energy is

```math
E(A,c)
= \Delta_{\rm Rb}|A|
+ B_e N_e(A)
+ B_d N_d(A)
+ \Delta_{\rm Cs}\mathbf{1}_{c=r}
+ B_{\rm RbCs}|A|\mathbf{1}_{c=r}.
```

Here $N_e(A)$ is the number of Rb edge pairs in $A$, $N_d(A)$ is the number of Rb diagonal pairs in $A$, and $\mathbf{1}_{c=r}$ is 1 only when Cs is in $|r\rangle$. In the notebook, $\Delta_{\rm Rb}$ and $\Delta_{\rm Cs}$ are `rb_stark_shift` and `cs_stark_shift`.

## Symmetry-Reduced Block Bases

The previous raw basis $|A,c\rangle$ has dimension $2\times 2^m$ for $m$ active Rb atoms. This is valid but unnecessarily large. Because all active Rb atoms in one symmetry class are driven identically, and because the square geometry preserves subgroup symmetries, each class can be reduced to symmetry-adapted Rb states.

The Cs branch is then attached as

```math
|\alpha,c\rangle = |\alpha\rangle_{\rm Rb}\otimes |c\rangle_{\rm Cs},
\qquad c\in\{1,r\}.
```

The Cs block is a two-level $|1\rangle\leftrightarrow |r\rangle$ system.

### Dimension Summary

| Class | Raw dimension | Reduced Rb dimension | Reduced total dimension |
| --- | ---: | ---: | ---: |
| `m0` | 2 | 1 | 2 |
| `m1` | 4 | 2 | 4 |
| `m2_edge` | 8 | 3 | 6 |
| `m2_diag` | 8 | 3 | 6 |
| `m3` | 16 | 6 | 12 |
| `m4` | 32 | 6 | 12 |

The reduced total dimension is $2d_{\rm Rb}$, where $d_{\rm Rb}$ is the reduced Rb-sector dimension.

## Shared Hamiltonian Form

For each class and each time slice $k$,

```math
H_k = H_{\rm Rb}(k)\otimes I_{\rm Cs}
+ I_{\rm Rb}\otimes H_{\rm Cs}(k)
+ H_{\rm int}.
```

The Cs single-atom Hamiltonian in the ordered basis $(|1\rangle, |r\rangle)$ is

```math
H_{\rm Cs}(k)=
\begin{pmatrix}
0 & \Omega_{\rm Cs}(k)e^{i\phi_{\rm Cs}(k)}/2\\
\Omega_{\rm Cs}(k)e^{-i\phi_{\rm Cs}(k)}/2 & \Delta_{\rm Cs}
\end{pmatrix}.
```

The Rb Stark shift contributes $\Delta_{\rm Rb}N_{\rm Rb}$ to $H_{\rm Rb}$. Equivalently, every reduced Rb basis state with $n_R$ Rb Rydberg excitations receives the extra diagonal shift $n_R\Delta_{\rm Rb}$.

The derivatives are

```math
\frac{\partial H_{\rm Cs}}{\partial \phi_{\rm Cs}}
=
\begin{pmatrix}
0 & i\Omega_{\rm Cs}e^{i\phi_{\rm Cs}}/2\\
-i\Omega_{\rm Cs}e^{-i\phi_{\rm Cs}}/2 & 0
\end{pmatrix},
```

and

```math
\frac{\partial H_{\rm Cs}}{\partial \rho}
=
\begin{pmatrix}
0 & \Omega_{\rm Rb}e^{i\phi_{\rm Cs}}/2\\
\Omega_{\rm Rb}e^{-i\phi_{\rm Cs}}/2 & 0
\end{pmatrix}.
```

The Rb drive derivative has the same structure inside each reduced Rb block:

```math
\frac{\partial}{\partial \phi_{\rm Rb}}
\left(\frac{\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}}{2}|\alpha\rangle\langle\beta|+\text{h.c.}\right)
=
\frac{i\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}}{2}|\alpha\rangle\langle\beta|
-\frac{i\Omega_{\rm Rb}e^{-i\phi_{\rm Rb}}}{2}|\beta\rangle\langle\alpha|.
```

The following sections specify the reduced Rb bases and $H_{\rm Rb}$ for each computational class.

## Class `m0`: Representative `0000`

There are no active Rb atoms. The reduced Rb basis is

```math
|g_0\rangle.
```

The Rb Hamiltonian is zero:

```math
H_{\rm Rb}^{(m0)} = (0).
```

The full block is only the Cs two-level system:

```math
H^{(m0)}_k = H_{\rm Cs}(k).
```

The initial and target states are both

```math
|g_0\rangle\otimes |+\rangle_{\rm Cs}.
```

## Class `m1`: Representative `1000`

There is one active Rb atom. The reduced Rb basis is

```math
|G\rangle,\quad |R\rangle.
```

Here $|G\rangle$ means the active Rb atom is in $|1\rangle$, and $|R\rangle$ means it is in $|r\rangle$.

The Rb Hamiltonian is

```math
H_{\rm Rb}^{(m1)}(k)
=
\begin{pmatrix}
0 & \Omega_{\rm Rb}e^{i\phi_{\rm Rb}(k)}/2\\
\Omega_{\rm Rb}e^{-i\phi_{\rm Rb}(k)}/2 & \Delta_{\rm Rb}
\end{pmatrix}.
```

The Rb-Cs interaction contribution is diagonal:

```math
\Delta E_{G,c}=0,\qquad
\Delta E_{R,c}=B_{\rm RbCs}\mathbf{1}_{c=r}.
```

The $\Delta_{\rm Rb}$ term is already included in $H_{\rm Rb}^{(m1)}$.

The target Cs state is $|-\rangle_{\rm Cs}$, because the Rb parity is odd.

## Class `m2_edge`: Representative `1100`

There are two adjacent active Rb atoms. Use the symmetric Rb basis

```math
|G\rangle=|11\rangle,
\qquad
|W\rangle=\frac{|r1\rangle+|1r\rangle}{\sqrt{2}},
\qquad
|D_e\rangle=|rr\rangle.
```

The Rb Hamiltonian is

```math
H_{\rm Rb}^{(m2e)}(k)=
\begin{pmatrix}
0 & \sqrt{2}\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}/2 & 0\\
\sqrt{2}\Omega_{\rm Rb}e^{-i\phi_{\rm Rb}}/2 & \Delta_{\rm Rb} & \sqrt{2}\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}/2\\
0 & \sqrt{2}\Omega_{\rm Rb}e^{-i\phi_{\rm Rb}}/2 & B_e+2\Delta_{\rm Rb}
\end{pmatrix}.
```

The Rb-Cs interaction adds $n_R B_{\rm RbCs}\mathbf{1}_{c=r}$, where $n_R=(0,1,2)$ for $|G,c\rangle$, $|W,c\rangle$, and $|D_e,c\rangle$, respectively. The $\Delta_{\rm Rb}n_R$ and $B_e$ terms are already included in $H_{\rm Rb}^{(m2e)}$.

The target Cs state is $|+\rangle_{\rm Cs}$, because the Rb parity is even.

## Class `m2_diag`: Representative `1010`

This class is identical to `m2_edge` except the two active Rb atoms are diagonal neighbors. The reduced Rb basis is

```math
|G\rangle,\quad |W\rangle,\quad |D_d\rangle=|rr\rangle.
```

The Rb Hamiltonian is

```math
H_{\rm Rb}^{(m2d)}(k)=
\begin{pmatrix}
0 & \sqrt{2}\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}/2 & 0\\
\sqrt{2}\Omega_{\rm Rb}e^{-i\phi_{\rm Rb}}/2 & \Delta_{\rm Rb} & \sqrt{2}\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}/2\\
0 & \sqrt{2}\Omega_{\rm Rb}e^{-i\phi_{\rm Rb}}/2 & B_d+2\Delta_{\rm Rb}
\end{pmatrix}.
```

The Rb-Cs diagonal shifts are again $n_R B_{\rm RbCs}\mathbf{1}_{c=r}$, where $n_R=(0,1,2)$. The $\Delta_{\rm Rb}n_R$ and $B_d$ terms are already included in $H_{\rm Rb}^{(m2d)}$.

The target Cs state is $|+\rangle_{\rm Cs}$.

## Class `m3`: Representative `1110`

There are three active Rb atoms forming an L shape. Label the corner atom $a$, and the two atoms adjacent to it $b,c$. The pairs $(a,b)$ and $(a,c)$ are edges, while $(b,c)$ is a diagonal.

A symmetry-adapted Rb basis is

```math
\begin{aligned}
|G\rangle &= |111\rangle,\\
|S_a\rangle &= |r11\rangle,\\
|S_b\rangle &= \frac{|1r1\rangle+|11r\rangle}{\sqrt{2}},\\
|D_e\rangle &= \frac{|rr1\rangle+|r1r\rangle}{\sqrt{2}},\\
|D_d\rangle &= |1rr\rangle,\\
|T\rangle &= |rrr\rangle.
\end{aligned}
```

The reduced block dimension is therefore $6\times 2=12$. In the ordered basis

```math
\left(|G\rangle, |S_a\rangle, |S_b\rangle, |D_e\rangle, |D_d\rangle, |T\rangle\right),
```

the Rb Hamiltonian is

```math
H_{\rm Rb}^{(m3)}=
\begin{pmatrix}
0 & v & \sqrt{2}v & 0 & 0 & 0\\
v^* & \Delta_{\rm Rb} & 0 & \sqrt{2}v & 0 & 0\\
\sqrt{2}v^* & 0 & \Delta_{\rm Rb} & v & \sqrt{2}v & 0\\
0 & \sqrt{2}v^* & v^* & B_e+2\Delta_{\rm Rb} & 0 & \sqrt{2}v\\
0 & 0 & \sqrt{2}v^* & 0 & B_d+2\Delta_{\rm Rb} & v\\
0 & 0 & 0 & \sqrt{2}v^* & v^* & 2B_e+B_d+3\Delta_{\rm Rb}
\end{pmatrix},
```

where

```math
v=\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}/2.
```

The diagonal Rb energies are

```math
E_G=0,\qquad
E_{S_a}=\Delta_{\rm Rb},\qquad
E_{S_b}=\Delta_{\rm Rb},\qquad
E_{D_e}=B_e+2\Delta_{\rm Rb},\qquad
E_{D_d}=B_d+2\Delta_{\rm Rb},\qquad
E_T=2B_e+B_d+3\Delta_{\rm Rb}.
```

The Rb-Cs interaction adds

```math
n_R B_{\rm RbCs}\mathbf{1}_{c=r}
```

to a basis state with $n_R$ Rb Rydberg excitations. For the ordered basis above,

```math
n_R=(0,1,1,2,2,3).
```

The target Cs state is $|-\rangle_{\rm Cs}$.

## Class `m4`: Representative `1111`

All four Rb atoms are active. The full square symmetry gives a compact six-state Rb basis:

```math
\begin{aligned}
|G\rangle &= |1111\rangle,\\
|W_1\rangle &= \frac{1}{2}\sum_{j=1}^4 |r_j\rangle,\\
|D_e\rangle &= \frac{1}{2}\sum_{\langle i,j\rangle_{\rm edge}} |r_i r_j\rangle,\\
|D_d\rangle &= \frac{1}{\sqrt{2}}\sum_{\langle i,j\rangle_{\rm diag}} |r_i r_j\rangle,\\
|W_3\rangle &= \frac{1}{2}\sum_{j=1}^4 |\text{all Rb in }r\text{ except }j\rangle,\\
|Q\rangle &= |rrrr\rangle.
\end{aligned}
```

The reduced block dimension is $6\times 2=12$.

In this basis, the Rb Hamiltonian is

```math
H_{\rm Rb}^{(m4)}=
\begin{pmatrix}
0 & 2v & 0 & 0 & 0 & 0\\
2v^* & \Delta_{\rm Rb} & 2v & \sqrt{2}v & 0 & 0\\
0 & 2v^* & B_e+2\Delta_{\rm Rb} & 0 & 2v & 0\\
0 & \sqrt{2}v^* & 0 & B_d+2\Delta_{\rm Rb} & \sqrt{2}v & 0\\
0 & 0 & 2v^* & \sqrt{2}v^* & 2B_e+B_d+3\Delta_{\rm Rb} & 2v\\
0 & 0 & 0 & 0 & 2v^* & 4B_e+2B_d+4\Delta_{\rm Rb}
\end{pmatrix}.
```

Again,

```math
v=\Omega_{\rm Rb}e^{i\phi_{\rm Rb}}/2.
```

The diagonal Rb energies are:

```math
\begin{aligned}
E_G &= 0,\\
E_{W_1} &= \Delta_{\rm Rb},\\
E_{D_e} &= B_e+2\Delta_{\rm Rb},\\
E_{D_d} &= B_d+2\Delta_{\rm Rb},\\
E_{W_3} &= 2B_e+B_d+3\Delta_{\rm Rb},\\
E_Q &= 4B_e+2B_d+4\Delta_{\rm Rb}.
\end{aligned}
```

The Rb-Cs interaction adds $n_R B_{\rm RbCs}\mathbf{1}_{c=r}$, where $n_R=0,1,2,2,3,4$ for the six basis states above.

The target Cs state is $|+\rangle_{\rm Cs}$.

## Initial and Target States

For every class, the initial state is

```math
|\psi_0\rangle = |G\rangle_{\rm Rb}\otimes |+\rangle_{\rm Cs}.
```

The target state is

```math
|\psi_{\rm target}\rangle =
\begin{cases}
|G\rangle_{\rm Rb}\otimes |+\rangle_{\rm Cs}, & \text{even parity},\\
|G\rangle_{\rm Rb}\otimes |-\rangle_{\rm Cs}, & \text{odd parity}.
\end{cases}
```

All Rydberg population should return to zero at the end of the pulse.

## Fidelity and Local Phases

The fidelity layer should allow two independent local phases:

```math
\theta_{\rm Rb},\qquad \theta_{\rm Cs}.
```

A possible phase convention for one block is

```math
\xi_q=\xi_q^{\rm target}
+ n_{\rm Rb}(q)\theta_{\rm Rb}
+ n_{\rm Cs}(q)\theta_{\rm Cs}.
```

For this readout task, the target state already includes the parity-dependent Cs sign. The exact local phase convention should be chosen carefully in the notebook so that local single-qubit phases are optimized but the required parity flip is not gauged away.

One useful diagnostic is the limit $T\Omega_{\rm Rb}=2\pi$, $\Omega_{\rm Cs}=0$, and no Rb-Rb interaction. In that case the Rb drive gives an approximately parity-dependent $(-1)^m$ phase. If the coherent objective is evaluated at $\theta_{\rm Rb}=0$, the weighted terms can cancel as $1-4+4+2-4+1=0$, even though each block has large overlap. Starting the local phase near $\theta_{\rm Rb}=\pi$ aligns this trivial Rb phase.

## Implementation Plan

The future notebook should:

1. Define square vertex labels and edge/diagonal pair sets.
2. Build raw Rb basis states for each class.
3. Project the raw Hamiltonian into the reduced symmetry basis.
4. Construct `H`, `dH/dphi_rb`, `dH/dphi_cs`, and `dH/dratio_cs`.
5. Define initial and target states for all 6 weighted classes.
6. Evaluate the initial infidelity without optimization.
7. Run fixed-time optimization.
8. Plot the three time controls: `phi_rb`, `phi_cs`, and `ratio_cs`.
9. Trace selected populations for representative classes.
10. Optionally run a time-continuation scan.
