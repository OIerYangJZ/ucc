import random

import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator

from ucc.transpilers.repeated_run import (
    compile_repeated_run_on_backend,
    find_dominant_repeated_run,
)


def line_backend(num_qubits):
    coupling_map = []
    for qubit in range(num_qubits - 1):
        coupling_map += [[qubit, qubit + 1], [qubit + 1, qubit]]
    return GenericBackendV2(
        num_qubits=num_qubits, coupling_map=coupling_map, seed=1
    )


def repeated_run_circuit(body, repeats):
    gate = body.to_gate(label="blk")
    circuit = QuantumCircuit(body.num_qubits)
    circuit.h(0)
    circuit.cx(0, body.num_qubits - 1)
    for _ in range(repeats):
        circuit.append(gate, range(body.num_qubits))
    circuit.rx(0.3, 1)
    return circuit


def padded(circuit, width):
    result = QuantumCircuit(width)
    result.compose(circuit, range(circuit.num_qubits), inplace=True)
    return result


def test_finds_dominant_composite_run():
    body = QuantumCircuit(2)
    body.cx(0, 1)
    body.h(0)

    assert find_dominant_repeated_run(repeated_run_circuit(body, 600)) == (
        2,
        1,
        600,
    )


def test_ignores_short_or_primitive_circuits():
    body = QuantumCircuit(2)
    body.cx(0, 1)
    body.h(0)
    primitive = QuantumCircuit(2)
    for _ in range(600):
        primitive.compose(body, inplace=True)

    assert find_dominant_repeated_run(repeated_run_circuit(body, 10)) is None
    assert find_dominant_repeated_run(primitive) is None


def test_block_that_needs_routing_is_returned_to_the_shared_layout():
    # A triangle of interactions cannot be placed on a line without SWAPs,
    # so each mapped copy must be returned to the shared layout.
    body = QuantumCircuit(3)
    body.cx(0, 1)
    body.cx(1, 2)
    body.cx(0, 2)
    body.t(2)
    circuit = repeated_run_circuit(body, 600)

    result = compile_repeated_run_on_backend(
        circuit, line_backend(3), seed_transpiler=7
    )

    assert result is not None
    assert result.layout.final_layout is None
    assert Operator.from_circuit(result).equiv(Operator(circuit))


@pytest.mark.parametrize("seed", range(8))
def test_random_blocks_on_wider_backends_are_equivalent(seed):
    rng = random.Random(seed)
    num_qubits = rng.choice([3, 4])
    width = rng.choice([num_qubits, num_qubits + 1])
    body = QuantumCircuit(num_qubits)
    for _ in range(rng.randint(3, 7)):
        if rng.random() < 0.5:
            control, target = rng.sample(range(num_qubits), 2)
            body.cx(control, target)
        else:
            body.rz(rng.uniform(-3, 3), rng.randrange(num_qubits))
            body.h(rng.randrange(num_qubits))
    circuit = repeated_run_circuit(body, 600)

    result = compile_repeated_run_on_backend(
        circuit, line_backend(width), seed_transpiler=rng.choice([None, 7])
    )

    assert result is not None
    assert Operator.from_circuit(result).equiv(
        Operator(padded(circuit, width))
    )
