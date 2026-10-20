# Unitary Foundation Microgrant Proposal

## Title

Structure-aware compiler passes for the Unitary Compiler Collection (UCC)

## Summary

Upstream two exact compiler passes into UCC that keep high-level circuit
structure alive until it can be exploited, and add the benchmark circuits that
make their effect visible in ucc-bench. On official Qiskit library circuits
they cut phase estimation from 166,805 to 450 gates and QAOA CNOTs by 47%
versus Qiskit `optimization_level=3`, without calling it; on UCC's existing
ucc-bench suite they change no 2-qubit count.

## The problem

Compilers usually lower a circuit to basis gates before optimizing it. By then
exact structure is gone: a `QFT` followed by its inverse (UCC issue #662), a
block repeated many times, a controlled power `C-U^(2^k)` of a diagonal
unitary (Qiskit's `PhaseEstimation` emits `2^k` controlled copies of `U`), or
a layer of commuting `ZZ` rotations.

## What exists today

Two Qiskit `TransformationPass`es in `ucc/transpilers/`, run before
translation by `UCCDefault1(structural_prepass=True)` (default off; with a
backend they run in the level-3 preset's `pre_init` stage):

- `PreBasisStructuralSimplification`: exact inverse-pair and adjacent
  `U U^-1` cancellation, repeated-block simplification, commuting
  rotation/phase merging on high-level instructions.
- `PhasePolynomialSynthesis`: CNOT + phase structure as a phase polynomial
  plus a linear map, resynthesized with a GraySynth parity network (PMH for
  the linear part). A rewrite is kept only if it lowers the CNOT count after
  two-qubit consolidation.

Both are exact including global phase; unit tests compare full operators.

## Evidence

Basis `{cx, rx, ry, rz, h}`; gates / CNOTs; "UCC main" is upstream UCC. Scripts
and full tables are in `research/`.

| Circuit | Qiskit O3 | UCC main | with the passes |
|---|---:|---:|---:|
| `PhaseEstimation`, 10 evaluation qubits, diagonal 5-qubit `U` | 166,805 / 58,491 | 469,792 / 58,491 | **450 / 249** |
| `QAOAAnsatz`, MaxCut on K32, 24 layers | 36,512 / 23,808 | 167,833 / 23,808 | **25,352 / 12,648** |
| `GroverOperator`, 11 qubits, 36 iterations | 79,029 / 32,544 | 280,240 / 32,544 | **74,994 / 30,960** |
| repeated `QFT QFT^-1`, 100k gates | timeout (>120 s) | 968,787 | **0** |
| QPE-style, 100k gates | 212,006 (110 s; timed out at 120 s on a rerun) | 464,011 | **168,004** (5 s) |
| 20-qubit Grover, line backend | 7,195,089 (210 s) | 7,198,051 (212 s) | **6,896,129 (7.0 s)** |

Ablations attribute every gain beyond Qiskit O3 to one of the two passes;
circuits without such structure (repeated `QFT QFT`, a QAOA ring) tie.

ucc-bench compilation suite, run locally against UCC main, 2-qubit gates:
qaoa 1176 → 1176, qv 14856 → 14856, qft 2740 → 2740, square_heisenberg
540 → 540, prep_select 9702 → 9702, qcnn 388 → 388. These circuits are
already basis-level, so no change is expected; they confirm no regression.

Known limitations: compile time on those basis-level inputs is higher than UCC
main (quantum volume 5.4 s vs 0.4 s, QFT 1.4 s vs 0.15 s), and ucc-bench has
no high-level circuits that would show the gains above.

## Proposed work (about 8 weeks, part time)

1. **Upstream the passes.** UCC's New Compiler Pass discussion (DISCUSSION_URL),
   then focused pull requests with tests and documentation; maintainers
   decide whether `ucc.compile()` enables them by default.
2. **Compile time.** Bring the overhead on basis-level ucc-bench circuits
   close to UCC main (profiling points at Python-level structure scans and
   parity-network synthesis on blocks that are later rejected).
3. **Benchmarks.** Contribute high-level circuits (phase estimation, QAOA,
   Grover, `QFT QFT^-1`) to ucc-bench so this class of optimization is
   tracked for every future pass.
4. **Upstream reports.** File the Qiskit `CommutativeInverseCancellation`
   scaling issue found during this work (quadratic in the number of custom
   composite gates; reproduces on Qiskit 2.6.0).

## Use of funds ($4,000)

Developer time for the four items above; no hardware or paid services are
needed.

## Why Unitary Foundation

UCC is a Unitary Foundation project and this work targets a reported UCC
issue, follows UCC's pass architecture and contribution process, and adds
benchmark coverage that other UCC contributors can use.
