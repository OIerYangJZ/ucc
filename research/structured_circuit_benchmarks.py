import argparse
import json
import math
from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass
class CircuitCase:
    family: str
    circuit: QuantumCircuit


def qft_forward_block(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for target in range(num_qubits):
        qc.h(target)
        for control in range(target + 1, num_qubits):
            qc.cp(math.pi / (2 ** (control - target)), control, target)

    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)

    return qc


def qft_inverse_block(num_qubits: int) -> QuantumCircuit:
    qc = qft_forward_block(num_qubits)
    inverse_qft = QuantumCircuit(num_qubits)
    for i in range(num_qubits // 2):
        inverse_qft.swap(i, num_qubits - 1 - i)

    for target in reversed(range(num_qubits)):
        for control in reversed(range(target + 1, num_qubits)):
            inverse_qft.cp(
                -math.pi / (2 ** (control - target)), control, target
            )
        inverse_qft.h(target)

    qc.compose(inverse_qft, inplace=True)
    return qc


def qft_control_block(num_qubits: int) -> QuantumCircuit:
    qc = qft_forward_block(num_qubits)
    qc.compose(qft_forward_block(num_qubits), inplace=True)
    return qc


def qpe_roundtrip_block(eval_qubits: int) -> QuantumCircuit:
    total_qubits = eval_qubits + 1
    phase_qubit = eval_qubits
    qc = QuantumCircuit(total_qubits)

    for q in range(eval_qubits):
        qc.h(q)

    for q in range(eval_qubits):
        repetitions = 2**q
        for _ in range(repetitions):
            qc.cp(math.pi / 8, q, phase_qubit)

    for q in reversed(range(eval_qubits)):
        for k in reversed(range(q + 1, eval_qubits)):
            qc.cp(-math.pi / (2 ** (k - q)), k, q)
        qc.h(q)

    for q in range(eval_qubits // 2):
        qc.swap(q, eval_qubits - 1 - q)

    for q in range(eval_qubits):
        qc.h(q)

    for q in reversed(range(eval_qubits)):
        repetitions = 2**q
        for _ in range(repetitions):
            qc.cp(-math.pi / 8, q, phase_qubit)

    return qc


def qaoa_ring_block(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    gamma_z = math.pi / 7
    gamma_zz = math.pi / 11
    beta = math.pi / 13

    for q in range(num_qubits):
        qc.rz(gamma_z, q)

    for i in range(num_qubits):
        j = (i + 1) % num_qubits
        qc.cx(i, j)
        qc.rz(gamma_zz, j)
        qc.cx(i, j)

    for q in range(num_qubits):
        qc.rx(beta, q)

    return qc


def grover_mirrored_block(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    oracle_qubits = list(range(num_qubits - 1))
    ancilla = num_qubits - 1

    for q in oracle_qubits:
        qc.h(q)
    qc.x(ancilla)
    qc.h(ancilla)

    qc.mcx(oracle_qubits, ancilla)

    for q in oracle_qubits:
        qc.h(q)
        qc.x(q)
    qc.h(oracle_qubits[-1])
    qc.mcx(oracle_qubits[:-1], oracle_qubits[-1])
    qc.h(oracle_qubits[-1])
    for q in oracle_qubits:
        qc.x(q)
        qc.h(q)

    for q in reversed(oracle_qubits):
        qc.h(q)
    qc.h(ancilla)
    qc.x(ancilla)

    return qc


def repeat_to_target(
    block: QuantumCircuit, target_gates: int
) -> QuantumCircuit:
    block_gate_count = len(block.data)
    if block_gate_count == 0:
        raise ValueError("Block must contain at least one gate")
    if target_gates % block_gate_count != 0:
        raise ValueError(
            f"Target gate count {target_gates} is not divisible by block size {block_gate_count}"
        )

    repeats = target_gates // block_gate_count
    circuit = QuantumCircuit(block.num_qubits)
    for _ in range(repeats):
        circuit.compose(block, inplace=True)
    return circuit


def build_cases(target_gates: int) -> list[CircuitCase]:
    return [
        CircuitCase(
            "qft_inverse",
            repeat_to_target(qft_inverse_block(8), target_gates),
        ),
        CircuitCase(
            "qft_control",
            repeat_to_target(qft_control_block(8), target_gates),
        ),
        CircuitCase(
            "qpe_style",
            repeat_to_target(qpe_roundtrip_block(4), target_gates),
        ),
        CircuitCase(
            "qaoa_ring",
            repeat_to_target(qaoa_ring_block(20), target_gates),
        ),
        CircuitCase(
            "grover_mirrored",
            repeat_to_target(grover_mirrored_block(8), target_gates),
        ),
    ]


def build_case(family: str, target_gates: int) -> CircuitCase:
    for case in build_cases(target_gates):
        if case.family == family:
            return case
    raise KeyError(f"Unknown benchmark family: {family}")


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured benchmark circuits for pre-basis simplification experiments."
    )
    parser.add_argument(
        "--target-gates",
        type=int,
        default=100_000,
        help="Exact gate count for each generated benchmark.",
    )
    args = parser.parse_args()

    payload = {
        case.family: metrics(case.circuit)
        for case in build_cases(args.target_gates)
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
