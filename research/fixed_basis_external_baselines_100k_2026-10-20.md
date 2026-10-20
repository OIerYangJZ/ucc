# Fixed-Basis External Baselines

Target basis: `['cx', 'rx', 'ry', 'rz', 'h']`

## qft_inverse

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only | ok | 400,000 | 142,500 | 170,000 | 0.168 s |
| qiskit_opt0 | ok | 400,000 | 142,500 | 170,000 | 0.16 s |
| qiskit_opt1 | ok | 365,002 | 130,002 | 140,000 | 0.419 s |
| qiskit_opt3 | timeout | - | - | - | > 120 s |
| qiskit_commutative_inverse | ok | 0 | 0 | 0 | 1.742 s |
| tket_full_peephole | unavailable | - | - | - | - |
| baseline_ucc | ok | 968,787 | 276,269 | 135,004 | 4.075 s |
| optimized_ucc | ok | 0 | 0 | 0 | 0.466 s |
| ucc_local_only | ok | 282,526 | 105,010 | 135,004 | 5.741 s |
| ucc_plus_structural | ok | 0 | 0 | 0 | 1.573 s |

## qft_control

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only | ok | 400,000 | 140,000 | 170,000 | 0.158 s |
| qiskit_opt0 | ok | 400,000 | 140,000 | 170,000 | 0.192 s |
| qiskit_opt1 | ok | 400,000 | 140,000 | 170,000 | 0.294 s |
| qiskit_opt3 | ok | 347,500 | 125,000 | 170,000 | 1.144 s |
| qiskit_commutative_inverse | ok | 400,000 | 140,000 | 170,000 | 1.417 s |
| tket_full_peephole | unavailable | - | - | - | - |
| baseline_ucc | ok | 1,207,504 | 317,501 | 170,000 | 4.133 s |
| optimized_ucc (= qiskit_opt3) | ok | 347,500 | 125,000 | 170,000 | 10.087 s |
| ucc_local_only (= qiskit_opt3) | ok | 347,500 | 125,000 | 170,000 | 5.815 s |
| ucc_plus_structural (= qiskit_opt3) | ok | 347,500 | 125,000 | 170,000 | 6.856 s |

## qpe_style

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only | ok | 396,000 | 296,001 | 156,000 | 0.152 s |
| qiskit_opt0 | ok | 396,000 | 296,001 | 156,000 | 0.158 s |
| qiskit_opt1 | ok | 396,000 | 296,001 | 156,000 | 0.327 s |
| qiskit_opt3 | timeout | - | - | - | > 120 s |
| qiskit_commutative_inverse | ok | 396,000 | 296,001 | 156,000 | 1.289 s |
| tket_full_peephole | unavailable | - | - | - | - |
| baseline_ucc | ok | 464,011 | 238,011 | 60,002 | 1.989 s |
| optimized_ucc | ok | 168,004 | 110,002 | 60,002 | 5.042 s |
| ucc_local_only | ok | 222,003 | 140,004 | 60,002 | 4.353 s |
| ucc_plus_structural | ok | 168,004 | 110,002 | 60,002 | 4.319 s |

## qaoa_ring

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only (= qiskit_opt3) | ok | 100,000 | 62,000 | 40,000 | 0.016 s |
| qiskit_opt0 (= qiskit_opt3) | ok | 100,000 | 62,000 | 40,000 | 0.016 s |
| qiskit_opt1 (= qiskit_opt3) | ok | 100,000 | 62,000 | 40,000 | 0.046 s |
| qiskit_opt3 | ok | 100,000 | 62,000 | 40,000 | 0.219 s |
| qiskit_commutative_inverse (= qiskit_opt3) | ok | 100,000 | 62,000 | 40,000 | 0.376 s |
| tket_full_peephole | unavailable | - | - | - | - |
| baseline_ucc | ok | 300,055 | 163,055 | 40,000 | 0.899 s |
| optimized_ucc (= qiskit_opt3) | ok | 100,000 | 62,000 | 40,000 | 2.882 s |
| ucc_local_only (= qiskit_opt3) | ok | 100,000 | 62,000 | 40,000 | 2.205 s |
| ucc_plus_structural (= qiskit_opt3) | ok | 100,000 | 62,000 | 40,000 | 2.913 s |

## grover_mirrored

| Method | Status | Output Gates | Output Depth | CX Count | Runtime |
|---|---|---:|---:|---:|---:|
| translation_only | ok | 1,370,000 | 840,009 | 492,000 | 0.96 s |
| qiskit_opt0 | ok | 1,370,000 | 840,009 | 492,000 | 1.237 s |
| qiskit_opt1 | ok | 1,184,009 | 772,009 | 468,000 | 2.686 s |
| qiskit_opt3 | ok | 1,158,011 | 762,010 | 468,000 | 5.469 s |
| qiskit_commutative_inverse | ok | 1,334,004 | 836,009 | 492,000 | 1.396 s |
| tket_full_peephole | unavailable | - | - | - | - |
| baseline_ucc | ok | 4,000,009 | 2,154,021 | 468,000 | 16.003 s |
| optimized_ucc | ok | 1,082,011 | 750,010 | 440,000 | 28.78 s |
| ucc_local_only | ok | 1,162,010 | 762,010 | 468,000 | 19.2 s |
| ucc_plus_structural | ok | 1,160,410 | 762,010 | 468,000 | 20.764 s |
