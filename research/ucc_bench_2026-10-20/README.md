# ucc-bench compilation suite (local run, 2026-10-20)

ucc-bench at `b05d9ce`, `benchmarks/compilation_benchmarks.toml`, `-j 4` on an
Apple-silicon laptop. `ucc_main.json` uses the UCC pinned by ucc-bench
(UCC main `b1b4e384`); `structure_aware_passes.json` the same environment
with this fork's `structure-aware-passes` branch installed. Times include
machine noise; compare within a column.

| Benchmark | raw 2q | Qiskit default 2q | UCC main 2q | this branch 2q | time ms (Qiskit / UCC main / this branch) |
|---|---:|---:|---:|---:|---|
| qaoa | 1,176 | 1,176 | 1,176 | 1,176 | 15 / 51 / 240 |
| qv | 15,000 | 14,854 | 14,856 | 14,856 | 535 / 426 / 5424 |
| qft | 10,050 | 3,570 | 2,740 | 2,740 | 35 / 154 / 1394 |
| square_heisenberg | 2,160 | 540 | 540 | 540 | 83 / 87 / 239 |
| prep_select | 9,744 | 9,708 | 9,702 | 9,702 | 104 / 269 / 1229 |
| qcnn | 388 | 388 | 388 | 388 | 5 / 11 / 94 |
