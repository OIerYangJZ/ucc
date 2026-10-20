# Real Algorithm-Family Baselines

Target basis: `['cx', 'rx', 'ry', 'rz', 'h']`

## phase_estimation_real

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only | ok | 159,838 | 121,214 | 67,608 | 0.058 s |
| qiskit_opt1 | ok | 159,838 | 121,214 | 67,608 | 0.099 s |
| qiskit_opt3 | ok | 166,805 | 119,204 | 58,491 | 0.549 s |
| qiskit_commutative_inverse | ok | 159,838 | 121,214 | 67,608 | 0.281 s |
| baseline_ucc | ok | 469,792 | 326,241 | 58,491 | 37.796 s |
| optimized_ucc | ok | 450 | 203 | 249 | 0.724 s |
| ucc_local_only | ok | 167,826 | 122,243 | 58,491 | 77.783 s |
| ucc_plus_structural | ok | 167,826 | 122,243 | 58,491 | 81.713 s |

## grover_real

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only | ok | 85,403 | 55,892 | 33,408 | 0.069 s |
| qiskit_opt1 | ok | 80,718 | 54,307 | 32,544 | 0.11 s |
| qiskit_opt3 | ok | 79,029 | 54,163 | 32,544 | 0.295 s |
| qiskit_commutative_inverse | ok | 85,403 | 55,892 | 33,408 | 0.06 s |
| baseline_ucc | ok | 280,240 | 151,722 | 32,544 | 1.119 s |
| optimized_ucc | ok | 74,994 | 53,355 | 30,960 | 1.828 s |
| ucc_local_only | ok | 79,102 | 54,163 | 32,544 | 2.132 s |
| ucc_plus_structural | ok | 79,102 | 54,163 | 32,544 | 2.198 s |

## qaoa_real

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only (= qiskit_opt3) | ok | 36,512 | 2,416 | 23,808 | 0.02 s |
| qiskit_opt1 (= qiskit_opt3) | ok | 36,512 | 2,416 | 23,808 | 0.029 s |
| qiskit_opt3 | ok | 36,512 | 2,416 | 23,808 | 0.15 s |
| qiskit_commutative_inverse (= qiskit_opt3) | ok | 36,512 | 2,416 | 23,808 | 0.276 s |
| baseline_ucc | ok | 167,833 | 7,133 | 23,808 | 0.76 s |
| optimized_ucc | ok | 25,352 | 1,590 | 12,648 | 1.644 s |
| ucc_local_only (= qiskit_opt3) | ok | 36,512 | 2,416 | 23,808 | 1.644 s |
| ucc_plus_structural (= qiskit_opt3) | ok | 36,512 | 2,416 | 23,808 | 1.663 s |
