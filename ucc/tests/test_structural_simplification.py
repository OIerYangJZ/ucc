import random

import pytest
from qiskit import QuantumCircuit
from qiskit import transpile as qiskit_transpile
from qiskit.circuit.library import QFTGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

from ucc import UCCDefault1
from ucc.transpilers.structural_simplification import (
    PreBasisStructuralSimplification,
)

TARGET_BASIS = ["cx", "rz", "rx", "ry", "h"]


def qft_round_trips(num_qubits, repeats):
    circuit = QuantumCircuit(num_qubits)
    for _ in range(repeats):
        circuit.append(QFTGate(num_qubits), range(num_qubits))
        circuit.append(QFTGate(num_qubits).inverse(), range(num_qubits))
    circuit.h(0)
    return circuit


def random_structured_circuit(rng, num_qubits):
    """Random circuit mixing primitive gates, a custom gate, and inverses."""
    body = QuantumCircuit(num_qubits)
    for _ in range(rng.randint(2, 5)):
        if rng.random() < 0.5:
            control, target = rng.sample(range(num_qubits), 2)
            body.cx(control, target)
        else:
            body.rz(rng.uniform(-3, 3), rng.randrange(num_qubits))
            body.h(rng.randrange(num_qubits))
    gate = body.to_gate(label="blk")

    circuit = QuantumCircuit(num_qubits)
    for _ in range(rng.randint(1, 12)):
        choice = rng.random()
        if choice < 0.35:
            circuit.append(gate, range(num_qubits))
        elif choice < 0.55:
            circuit.append(gate, range(num_qubits))
            circuit.append(gate.inverse(), range(num_qubits))
        elif choice < 0.75:
            angle = rng.uniform(-3, 3)
            qubit = rng.randrange(num_qubits)
            circuit.rz(angle, qubit)
            circuit.rz(-angle / 2, qubit)
        else:
            control, target = rng.sample(range(num_qubits), 2)
            circuit.cx(control, target)
            circuit.cx(control, target)
    return circuit


def test_pass_removes_qft_inverse_qft_pairs():
    circuit = qft_round_trips(5, 20)

    result = PassManager([PreBasisStructuralSimplification()]).run(circuit)

    assert dict(result.count_ops()) == {"h": 1}
    assert Operator(result).equiv(Operator(circuit))


def test_pass_leaves_irreducible_circuit_unchanged():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.3, 2)

    result = PassManager([PreBasisStructuralSimplification()]).run(circuit)

    assert result == circuit


@pytest.mark.parametrize("seed", range(25))
def test_pass_preserves_unitary_on_random_structured_circuits(seed):
    rng = random.Random(seed)
    circuit = random_structured_circuit(rng, rng.choice([2, 3, 4]))

    result = PassManager([PreBasisStructuralSimplification()]).run(circuit)

    assert len(result.data) <= len(circuit.data)
    assert Operator(result).equiv(Operator(circuit))


def test_ucc_default_structural_prepass_beats_basis_first_pipeline():
    circuit = qft_round_trips(5, 20)
    basis_first = UCCDefault1().run(
        qiskit_transpile(
            circuit, basis_gates=TARGET_BASIS, optimization_level=0
        )
    )

    result = UCCDefault1(structural_prepass=True).run(circuit)

    assert set(result.count_ops()) <= set(TARGET_BASIS)
    assert result.size() < basis_first.size()
    assert Operator(result).equiv(Operator(circuit))


def test_ucc_default_structural_prepass_is_off_by_default():
    pass_names = [
        type(task).__name__
        for task in UCCDefault1().pass_manager.to_flow_controller().tasks
    ]

    assert "PreBasisStructuralSimplification" not in pass_names


def test_ucc_default_structural_prepass_with_backend():
    coupling_map = [[0, 1], [1, 0], [1, 2], [2, 1]]
    backend = GenericBackendV2(num_qubits=3, coupling_map=coupling_map, seed=1)
    circuit = qft_round_trips(3, 10)

    without_prepass = UCCDefault1(
        target_backend=backend, seed_transpiler=7
    ).run(circuit)

    result = UCCDefault1(
        target_backend=backend, seed_transpiler=7, structural_prepass=True
    ).run(circuit)

    assert Operator.from_circuit(result).equiv(Operator(circuit))
    assert result.size() < without_prepass.size()


def test_ucc_default_structural_prepass_translates_standard_gates():
    """Regression: HighLevelSynthesis without the equivalence library
    rejected standard gates such as ``u`` that BasisTranslator handles."""
    circuit = QuantumCircuit(2)
    circuit.u(0.1, 0.2, 0.3, 0)
    circuit.cp(0.4, 0, 1)
    circuit.append(QFTGate(2), [0, 1])

    result = UCCDefault1(structural_prepass=True).run(circuit)
    result = qiskit_transpile(
        result, basis_gates=TARGET_BASIS, optimization_level=0
    )

    assert set(result.count_ops()) <= set(TARGET_BASIS)
    assert Operator(result).equiv(Operator(circuit))


def test_pass_cancels_cz_pairs_in_diagonal_spans():
    """Regression: two cz on the same pair were merged into a single cz."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.rz(0.3, 1)
    circuit.cz(0, 1)
    circuit.h(1)

    result = PassManager([PreBasisStructuralSimplification()]).run(circuit)

    assert Operator(result).equiv(Operator(circuit))


def test_pass_simplifies_large_dominant_repeated_prefix_blockwise():
    # Above 10,000 instructions the pass simplifies one copy of a dominant
    # repeated prefix and reuses it (and uses the fast prefix scan).
    block = QuantumCircuit(3)
    block.h(0)
    block.rz(0.2, 1)
    block.cx(0, 1)
    block.cx(0, 1)
    block.rz(-0.2, 1)
    block.t(2)
    circuit = QuantumCircuit(3)
    for _ in range(2000):
        circuit.compose(block, inplace=True)
    circuit.cx(1, 2)
    circuit.h(2)

    result = PassManager([PreBasisStructuralSimplification()]).run(circuit)

    assert len(result.data) < len(circuit.data) // 2
    assert Operator(result).equiv(Operator(circuit))


def _two_qubit_count(circuit):
    return sum(
        1 for instruction in circuit.data if len(instruction.qubits) > 1
    )


def test_lowest_cost_local_optimization_keeps_the_cheaper_pipeline():
    from qiskit.transpiler.passes import CommutativeCancellation

    from ucc.transpilers.ucc_defaults import LowestCostLocalOptimization

    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    circuit.h(0)

    result = PassManager(
        [
            LowestCostLocalOptimization(
                [lambda: [], lambda: [CommutativeCancellation()]],
                TARGET_BASIS,
            )
        ]
    ).run(circuit)

    assert _two_qubit_count(result) == 0
    assert Operator(result).equiv(Operator(circuit))
