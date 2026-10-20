# Benchmarks for the structure-aware passes

Scripts behind the numbers in `proposal/`. Every method runs in its own
subprocess with a timeout; results are written as JSON and Markdown.

| Script | Circuits |
|---|---|
| `compare_real_instances.py` | Qiskit library instances: `PhaseEstimation` (10 evaluation qubits, diagonal 5-qubit unitary), `GroverOperator` (11 qubits, 36 iterations), `QAOAAnsatz` (MaxCut on K32, 24 layers) |
| `compare_external_baselines.py` | 100,000-gate structured families: repeated `QFT QFT^-1`, `QFT QFT`, QPE-style, QAOA ring, mirrored Grover |
| `compare_hardware_aware.py` | MQT Bench QPE, QAOA, Grover on a 20-qubit line backend (seed 12345) |

Methods: `translation_only`, Qiskit `optimization_level` 0/1/3, Qiskit
`CommutativeInverseCancellation`, `baseline_ucc` (upstream UCC), and
`optimized_ucc` (this branch's `ucc.compile`). The first two scripts also run
ablations (`ucc_methods.py`) that drop passes from the real
`UCCDefault1(structural_prepass=True)` pass list: `ucc_local_only` (neither
structure-aware pass) and `ucc_plus_structural` (structural simplification
only). Rows identical to `qiskit_opt3` in gates, depth and CX are marked
`(= qiskit_opt3)`.

## Reproduce

```bash
uv sync --all-groups
uv pip install "mqt.bench>=2.2.1,<3"          # hardware instances only
git worktree add /tmp/ucc_662issue_baseline feaf2411   # upstream UCC baseline
cd research
uv run python compare_real_instances.py --json-out real.json --md-out real.md
uv run python compare_external_baselines.py --target-gates 100000 --json-out ext.json --md-out ext.md
uv run python compare_hardware_aware.py --json-out hw.json --md-out hw.md
```

`--baseline-repo` overrides the baseline checkout and `--experimental-repo`
the repository under test (default: this one). Results from the runs cited in
`proposal/` are the `*_2026-10-20.*` files here, and the local ucc-bench runs
are in `ucc_bench_2026-10-20/`.
