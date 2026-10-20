from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, PhaseEstimation, QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"qiskit\.circuit\.library\.(phase_estimation|grover_operator)",
)


@dataclass
class CircuitCase:
    family: str
    circuit: QuantumCircuit
    description: str


def _diagonal_phase_unitary(num_qubits: int) -> QuantumCircuit:
    """A diagonal phase unitary suitable for a concrete QPE instance."""
    qc = QuantumCircuit(num_qubits, name="diag_phase_step")
    for q in range(num_qubits):
        qc.rz(math.pi / (q + 2), q)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
        qc.rz(math.pi / (q + 3), q + 1)
        qc.cx(q, q + 1)
    return qc


def phase_estimation_case() -> CircuitCase:
    eval_qubits = 10
    target_qubits = 5
    unitary = _diagonal_phase_unitary(target_qubits)

    circuit = QuantumCircuit(
        eval_qubits + target_qubits, name="phase_estimation_real"
    )
    # Prepare a computational-basis eigenstate of the diagonal unitary.
    for q in range(target_qubits):
        circuit.x(eval_qubits + q)
    circuit.compose(
        PhaseEstimation(eval_qubits, unitary),
        qubits=list(range(eval_qubits + target_qubits)),
        inplace=True,
    )
    return CircuitCase(
        family="phase_estimation_real",
        circuit=circuit,
        description=(
            "Official Qiskit PhaseEstimation circuit over a concrete diagonal "
            "5-qubit phase unitary with 10 evaluation qubits."
        ),
    )


def _phase_oracle(bitstring: str) -> QuantumCircuit:
    num_qubits = len(bitstring)
    oracle = QuantumCircuit(num_qubits, name="phase_oracle")
    for qubit, bit in enumerate(reversed(bitstring)):
        if bit == "0":
            oracle.x(qubit)
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)
    for qubit, bit in enumerate(reversed(bitstring)):
        if bit == "0":
            oracle.x(qubit)
    return oracle


def grover_real_case() -> CircuitCase:
    num_qubits = 11
    marked_state = "10101100101"
    oracle = _phase_oracle(marked_state)
    grover_op = GroverOperator(oracle)
    iterations = max(1, round(math.pi / 4 * math.sqrt(2**num_qubits)))

    circuit = QuantumCircuit(num_qubits, name="grover_real")
    circuit.h(range(num_qubits))
    for _ in range(iterations):
        circuit.compose(grover_op, inplace=True)

    return CircuitCase(
        family="grover_real",
        circuit=circuit,
        description=(
            "Official Qiskit GroverOperator circuit with a concrete marked "
            f"bitstring oracle on {num_qubits} qubits and {iterations} Grover iterations."
        ),
    )


def _maxcut_complete_graph_operator(num_qubits: int) -> SparsePauliOp:
    terms: list[tuple[str, float]] = []
    for i, j in combinations(range(num_qubits), 2):
        label = ["I"] * num_qubits
        label[num_qubits - 1 - i] = "Z"
        label[num_qubits - 1 - j] = "Z"
        terms.append(("".join(label), 1.0))
    return SparsePauliOp.from_list(terms)


def qaoa_real_case() -> CircuitCase:
    num_qubits = 32
    reps = 24
    operator = _maxcut_complete_graph_operator(num_qubits)
    ansatz = QAOAAnsatz(operator, reps=reps, flatten=True)
    values = {
        param: (index + 1) * 0.01
        for index, param in enumerate(ansatz.parameters)
    }
    circuit = ansatz.assign_parameters(values)
    circuit.name = "qaoa_real"
    return CircuitCase(
        family="qaoa_real",
        circuit=circuit,
        description=(
            "Official Qiskit QAOAAnsatz for MaxCut on a 32-node complete graph "
            "with 24 layers and concrete bound parameters."
        ),
    )


def build_cases() -> list[CircuitCase]:
    return [
        phase_estimation_case(),
        grover_real_case(),
        qaoa_real_case(),
    ]


def build_case(family: str) -> CircuitCase:
    for case in build_cases():
        if case.family == family:
            return case
    raise KeyError(f"Unknown real-instance family: {family}")


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
        description="Generate real algorithm-family benchmark circuits for #662(issue)."
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
