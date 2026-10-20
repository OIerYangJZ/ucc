# Hardware-Aware Comparison

Target backend: 20-qubit bidirectional line backend

Seed transpiler: `12345`

## hw_mqt_qpeexact_20

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| qiskit_opt3 | ok | 1,572 | 407 | 812 | 0.046 s |
| baseline_ucc | ok | 1,912 | 603 | 1,174 | 0.069 s |
| optimized_ucc | ok | 1,572 | 407 | 812 | 0.119 s |

## hw_mqt_qaoa_20

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| qiskit_opt3 | ok | 2,091 | 532 | 1,530 | 0.055 s |
| baseline_ucc | ok | 2,060 | 487 | 1,382 | 0.054 s |
| optimized_ucc | ok | 2,091 | 532 | 1,530 | 0.089 s |

## hw_mqt_grover_20

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| qiskit_opt3 | ok | 7,195,089 | 4,430,283 | 3,680,048 | 209.702 s |
| baseline_ucc | ok | 7,198,051 | 4,434,466 | 3,682,673 | 212.003 s |
| optimized_ucc | ok | 6,896,129 | 4,312,276 | 3,400,048 | 7.032 s |
