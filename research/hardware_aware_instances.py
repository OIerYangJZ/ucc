from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit


@dataclass
class CircuitCase:
    family: str
    circuit: QuantumCircuit
    description: str


TARGET_SIZES = {
    "hw_mqt_qpeexact_20": ("qpeexact", 20),
    "hw_mqt_qaoa_20": ("qaoa", 20),
    "hw_mqt_grover_20": ("grover", 20),
}


def _bind_deterministic_parameters(circuit: QuantumCircuit) -> QuantumCircuit:
    if not circuit.parameters:
        return circuit
    assignments = {
        param: (index + 1) * 0.1
        for index, param in enumerate(sorted(circuit.parameters, key=str))
    }
    return circuit.assign_parameters(assignments)


def _build_case(
    family: str, benchmark_name: str, circuit_size: int
) -> CircuitCase:
    circuit = get_benchmark(
        benchmark_name,
        BenchmarkLevel.ALG,
        circuit_size=circuit_size,
        random_parameters=False,
    )
    circuit = _bind_deterministic_parameters(circuit)
    circuit.name = family
    return CircuitCase(
        family=family,
        circuit=circuit,
        description=(
            f"MQT Bench `{benchmark_name}` with circuit_size={circuit_size} "
            "compiled against a 20-qubit bidirectional line backend."
        ),
    )


def build_cases() -> list[CircuitCase]:
    return [
        _build_case(family, benchmark_name, circuit_size)
        for family, (benchmark_name, circuit_size) in TARGET_SIZES.items()
    ]


def build_case(family: str) -> CircuitCase:
    for case in build_cases():
        if case.family == family:
            return case
    raise KeyError(f"Unknown hardware-aware family: {family}")


def metrics(circuit: QuantumCircuit) -> dict:
    count_ops = circuit.count_ops()
    multi_qubit_gates = sum(
        1
        for instruction in circuit.data
        if instruction.operation.num_qubits > 1
    )
    return {
        "num_qubits": circuit.num_qubits,
        "total_gates": int(sum(count_ops.values())),
        "depth": int(circuit.depth()),
        "multi_qubit_gates": int(multi_qubit_gates),
        "gate_types": sorted(str(name) for name in count_ops.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hardware-aware benchmark circuits for #662(issue)."
    )
    parser.parse_args()
    payload = {
        case.family: {
            "description": case.description,
            "metrics": metrics(case.circuit),
        }
        for case in build_cases()
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
