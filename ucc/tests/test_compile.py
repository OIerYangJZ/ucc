import pytest
import math
import importlib
import cirq
from cirq import Circuit as CirqCircuit
from cirq import CNOT, H, X, LineQubit, NamedQubit
from pytket import Circuit as TketCircuit
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import transpile as qiskit_transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, Statevector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.passes import GatesInBasis, CountOps
from qiskit.transpiler.passes.utils import CheckMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library import HGate, XGate
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from ucc.tests.mock_backends import Mybackend
from ucc import compile
from ucc.transpilers.structural_simplification import (
    _merge_adjacent_parameterized_gates,
    _structural_pre_simplify,
)
from ucc.transpilers.ucc_defaults import UCCDefault1
from ucc.transpilers.aqc.mps_pass import MPSPass
import numpy as np


def repeated_block(block, repeats):
    circuit = QiskitCircuit(block.num_qubits)
    for _ in range(repeats):
        circuit.compose(block, inplace=True)
    return circuit


def qft_forward_block(num_qubits):
    qc = QiskitCircuit(num_qubits)
    for target in range(num_qubits):
        qc.h(target)
        for control in range(target + 1, num_qubits):
            qc.cp(math.pi / (2 ** (control - target)), control, target)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
    return qc


def qft_inverse_block(num_qubits):
    qc = QiskitCircuit(num_qubits)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)
    for target in reversed(range(num_qubits)):
        for control in reversed(range(target + 1, num_qubits)):
            qc.cp(-math.pi / (2 ** (control - target)), control, target)
        qc.h(target)
    return qc


def qaoa_ring_circuit(num_qubits, layers, gamma=0.125, beta=0.25):
    qc = QiskitCircuit(num_qubits)
    for _ in range(layers):
        for qubit in range(num_qubits):
            qc.h(qubit)
        for qubit in range(num_qubits):
            neighbour = (qubit + 1) % num_qubits
            qc.cx(qubit, neighbour)
            qc.rz(gamma, neighbour)
            qc.cx(qubit, neighbour)
        for qubit in range(num_qubits):
            qc.rx(beta, qubit)
    return qc


def qpe_roundtrip_block(eval_qubits):
    total_qubits = eval_qubits + 1
    phase_qubit = eval_qubits
    qc = QiskitCircuit(total_qubits)

    for qubit in range(eval_qubits):
        qc.h(qubit)

    for qubit in range(eval_qubits):
        for _ in range(2**qubit):
            qc.cp(math.pi / 8, qubit, phase_qubit)

    for qubit in reversed(range(eval_qubits)):
        for control in reversed(range(qubit + 1, eval_qubits)):
            qc.cp(-math.pi / (2 ** (control - qubit)), control, qubit)
        qc.h(qubit)

    for qubit in range(eval_qubits // 2):
        qc.swap(qubit, eval_qubits - 1 - qubit)

    for qubit in range(eval_qubits):
        qc.h(qubit)

    for qubit in reversed(range(eval_qubits)):
        for _ in range(2**qubit):
            qc.cp(-math.pi / 8, qubit, phase_qubit)

    return qc


def conjugation_block():
    qc = QiskitCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.25, 1)
    qc.cx(0, 1)
    qc.h(0)
    return qc


def mirrored_self_inverse_block():
    qc = QiskitCircuit(2)
    qc.cz(0, 1)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.x(1)
    qc.h(0)
    return qc


def grover_mirrored_block(num_qubits):
    qc = QiskitCircuit(num_qubits)
    oracle_qubits = list(range(num_qubits - 1))
    ancilla = num_qubits - 1

    for qubit in oracle_qubits:
        qc.h(qubit)
    qc.x(ancilla)
    qc.h(ancilla)
    qc.mcx(oracle_qubits, ancilla)

    for qubit in oracle_qubits:
        qc.h(qubit)
        qc.x(qubit)
    qc.h(oracle_qubits[-1])
    qc.mcx(oracle_qubits[:-1], oracle_qubits[-1])
    qc.h(oracle_qubits[-1])
    for qubit in oracle_qubits:
        qc.x(qubit)
        qc.h(qubit)

    for qubit in reversed(oracle_qubits):
        qc.h(qubit)
    qc.h(ancilla)
    qc.x(ancilla)

    return qc


def complete_graph_maxcut_operator(num_qubits):
    terms = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            label = ["I"] * num_qubits
            label[num_qubits - 1 - i] = "Z"
            label[num_qubits - 1 - j] = "Z"
            terms.append(("".join(label), 1.0))
    return SparsePauliOp.from_list(terms)


def random_area_law_circuit(N, seed=12345):
    """A circuit to generate a random area-law statevector.

    Parameters:
        N (int): Number of qubits

    Returns:
        QiskitCircuit: Output circuit
    """
    np.random.seed(seed)

    state = np.random.rand(2**N) + 1j * np.random.rand(2**N)
    state /= np.linalg.norm(state)

    circuit = QiskitCircuit(N)
    circuit.initialize(state, range(N))

    return circuit


def qcnn_circuit(N, seed=12345):
    """A circuit to generate a Quantum Convolutional Neural Network

    Parameters:
        N (int): Number of qubits

    Returns:
        QiskitCircuit: Output circuit
    """
    rng = np.random.default_rng(seed=seed)

    qc = QiskitCircuit(N)
    num_layers = int(np.ceil(np.log2(N)))
    i_conv = 0
    for i_layer in range(num_layers):
        for i_sub_layer in [0, 2**i_layer]:
            for i_q1 in range(i_sub_layer, N, 2 ** (i_layer + 1)):
                i_q2 = 2**i_layer + i_q1
                if i_q2 < N:
                    qc.rxx(rng.random(), i_q1, i_q2)
                    qc.ry(rng.random(), i_q1)
                    qc.ry(rng.random(), i_q2)
                    i_conv += 1

    return qc


def random_clifford_circuit(num_qubits, seed=12345):
    """Generate a random clifford circuit
    Parameters:
        num_qubits (int): Number of qubits
        seed (int): Optional. Seed the random number generator, default=12345

    Returns:
        QuantumCircuit: Clifford circuit
    """
    # This code is used to generate the QASM file
    from qiskit.circuit.random import random_clifford_circuit

    gates = ["cx", "cz", "cy", "swap", "x", "y", "z", "s", "sdg", "h"]
    qc = random_clifford_circuit(
        num_qubits,
        gates=gates,
        num_gates=10 * num_qubits * num_qubits,
        seed=seed,
    )
    return qc


def test_qiskit_compile():
    circuit = QiskitCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    result_circuit = compile(circuit, return_format="original")
    assert isinstance(result_circuit, QiskitCircuit)


def test_cirq_compile():
    qubits = LineQubit.range(2)
    circuit = CirqCircuit(H(qubits[0]), CNOT(qubits[0], qubits[1]))
    result_circuit = compile(circuit, return_format="original")
    assert isinstance(result_circuit, CirqCircuit)


def test_tket_compile():
    circuit = TketCircuit(2)
    circuit.H(0)
    circuit.CX(0, 1)
    result_circuit = compile(circuit, return_format="original")
    assert isinstance(result_circuit, TketCircuit)


def test_callback():
    was_called = False

    def my_callback(**kwargs):
        nonlocal was_called
        was_called = True

    circuit = QiskitCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    _ = compile(circuit, callback=my_callback)
    assert was_called


def test_custom_pass():
    """Verify that a custom pass works with a non-qiskit input circuit"""

    class HtoX(TransformationPass):
        """Toy transformation that converts all H gates to X gates"""

        def run(self, dag):
            for node in dag.op_nodes():
                if isinstance(node.op, HGate):
                    dag.substitute_node(node, XGate())
            return dag

    # Example usage with a cirq circuit, still showcasing the cross-frontend compatibility

    qubit = NamedQubit("q_0")
    cirq_circuit = CirqCircuit(H(qubit))

    post_compiler_circuit = compile(cirq_circuit, custom_passes=[HtoX()])
    assert cirq.equal_up_to_global_phase(
        cirq.unitary(post_compiler_circuit),
        cirq.unitary(CirqCircuit(X(qubit))),
    )


def test_qiskit_roundtrip_bypasses_qbraid_translate(monkeypatch):
    compile_module = importlib.import_module("ucc.compile")
    circuit = QiskitCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    monkeypatch.setattr(
        compile_module,
        "translate",
        lambda *args, **kwargs: pytest.fail(
            "qBraid translate should not run for qiskit->qiskit compile"
        ),
    )

    result_circuit = compile(circuit, return_format="qiskit")

    assert isinstance(result_circuit, QiskitCircuit)


def test_compile_target_backend_opset():
    circuit = QiskitCircuit(3)
    circuit.cz(0, 1)
    circuit.cz(0, 2)

    # Create a simple backend that does not have direct CX between 0 and 2
    t = Mybackend()
    # Check that the gates in the original circuit are not support by the target
    # to ensure this isn't a trivial check
    assert set(op.name for op in circuit).issubset(t.operation_names) is False

    result_circuit = compile(
        circuit, return_format="original", target_backend=t
    )
    # Check that the gates in the final circuit are all supported on the target device
    assert set(op.name for op in result_circuit).issubset(t.operation_names)


def test_compile_target_backend_coupling_map():
    circuit = QiskitCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(0, 2)

    # Create a simple target that does not have direct CX between 0 and 2
    t = Mybackend()
    result_circuit = compile(
        circuit, return_format="original", target_backend=t
    )
    # Check that the compiled circuit respects the coupling map of the target device
    analysis_pass = CheckMap(
        t.target.build_coupling_map(), property_set_field="check_map"
    )

    dag = circuit_to_dag(result_circuit)
    analysis_pass.run(dag)
    assert analysis_pass.property_set["check_map"]


def test_compile_with_no_target_gateset_or_device():
    """Test that the final circuit is in the default gateset if no `target_gateset` or `target_backend` is provided."""

    # Circuit not in the default target_gateset {"cx", "rz", "rx", "ry", "h"}
    circuit = QiskitCircuit(2)
    circuit.cz(0, 1)
    circuit.h(0)

    result_circuit = compile(
        circuit,
    )

    assert set(op.name for op in result_circuit).issubset(
        {"cx", "rz", "rx", "ry", "h"}
    )

    # Circuit already in the default target_gateset {"cx", "rz", "rx", "ry", "h"}
    circuit = QiskitCircuit(2)
    circuit.cx(0, 1)
    circuit.h(0)

    result_circuit = compile(
        circuit,
    )

    assert set(op.name for op in result_circuit).issubset(
        {"cx", "rz", "rx", "ry", "h"}
    )


def test_raise_error_on_bad_backend():
    circuit = QiskitCircuit(2)
    circuit.cx(0, 1)
    circuit.h(0)

    class BadBackend:
        pass

    with pytest.raises(ValueError):
        _ = compile(circuit, target_backend=BadBackend())


def test_bqskit_compile():
    from ucc.transpilers.ucc_bqskit import BQSKitTransformationPass

    bqskit_pass = BQSKitTransformationPass()
    qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[0];
        cp(1.5707963267948966) q[1], q[0];
        h q[1];
        cp(0.7853981633974483) q[2], q[0];
        cp(1.5707963267948966) q[2], q[1];
        h q[2];
        swap q[0], q[2];
        h q[0];
        cp(-1.5707963267948966) q[1], q[0];
        h q[1];
        cp(-0.7853981633974483) q[2], q[0];
        cp(-1.5707963267948966) q[2], q[1];
        h q[2];
        swap q[0], q[2];
        """
    # This qasm describes a 3-qubit QFT followed by a 3-qubit inverse QFT.
    # This circuit resolves to the identity, but that's not obvious to
    # most synthesis tools.
    # BQSKit using LEAP will usually remove all 2-qubit gates
    # from the circuit, leaving 3 u3 gates that don't do anything
    # because it just focuses on the 2-qubit gates. A further
    # post processing step would remove these 1-qubit gates, but in
    # more realistic use cases, its often not worth the extra processing.

    def get_post_cx_count(circuit, custom_passes=[]):
        post_compiler_circuit = compile(qasm, custom_passes=custom_passes)
        analysis_pass = CountOps()
        dag = circuit_to_dag(
            QiskitCircuit.from_qasm_str(post_compiler_circuit)
        )
        analysis_pass.run(dag)
        if "cx" in analysis_pass.property_set["count_ops"]:
            return analysis_pass.property_set["count_ops"]["cx"]
        else:
            return 0

    assert get_post_cx_count(qasm, [bqskit_pass]) < get_post_cx_count(qasm)


@pytest.mark.parametrize("N", [5, 8, 10, 11])
def test_compile_with_mps_pass(N):
    """Test that the circuit compiled by `MPSPass` works as expected."""
    circuit = random_area_law_circuit(N)
    circuit = qiskit_transpile(circuit, basis_gates=["u3", "cx"])

    compiled_circuit = compile(
        circuit, target_gateset=["u3", "cx"], custom_passes=[MPSPass()]
    )

    fidelity = np.abs(
        np.vdot(Statevector(circuit).data, Statevector(compiled_circuit).data)
    )

    assert np.abs(fidelity) > 0.9

    assert circuit.depth() > compiled_circuit.depth()
    assert circuit.count_ops().get("cx", 0) > compiled_circuit.count_ops().get(
        "cx", 0
    )


def test_compile_trivial_state_with_mps_pass():
    """Test that `MPSPass` does not use CX gates when state has no entanglement."""
    qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[10];
        ry(pi/2) q[9];
        rx(pi) q[9];
        rz(pi/4) q[9];
        cx q[9],q[8];
        rz(-pi/4) q[8];
        cx q[9],q[8];
        rz(pi/4) q[8];
        ry(pi/2) q[8];
        rx(pi) q[8];
        rz(pi/4) q[8];
        rz(pi/8) q[9];
        cx q[9],q[7];
        rz(-pi/8) q[7];
        cx q[9],q[7];
        rz(pi/8) q[7];
        cx q[8],q[7];
        rz(-pi/4) q[7];
        cx q[8],q[7];
        rz(pi/4) q[7];
        ry(pi/2) q[7];
        rx(pi) q[7];
        rz(pi/4) q[7];
        rz(pi/8) q[8];
        rz(pi/16) q[9];
        cx q[9],q[6];
        rz(-pi/16) q[6];
        cx q[9],q[6];
        rz(pi/16) q[6];
        cx q[8],q[6];
        rz(-pi/8) q[6];
        cx q[8],q[6];
        rz(pi/8) q[6];
        cx q[7],q[6];
        rz(-pi/4) q[6];
        cx q[7],q[6];
        rz(pi/4) q[6];
        ry(pi/2) q[6];
        rx(pi) q[6];
        rz(pi/4) q[6];
        rz(pi/8) q[7];
        rz(pi/16) q[8];
        rz(pi/32) q[9];
        cx q[9],q[5];
        rz(-pi/32) q[5];
        cx q[9],q[5];
        rz(pi/32) q[5];
        cx q[8],q[5];
        rz(-pi/16) q[5];
        cx q[8],q[5];
        rz(pi/16) q[5];
        cx q[7],q[5];
        rz(-pi/8) q[5];
        cx q[7],q[5];
        rz(pi/8) q[5];
        cx q[6],q[5];
        rz(-pi/4) q[5];
        cx q[6],q[5];
        rz(pi/4) q[5];
        ry(pi/2) q[5];
        rx(pi) q[5];
        rz(pi/4) q[5];
        rz(pi/8) q[6];
        rz(pi/16) q[7];
        rz(pi/32) q[8];
        rz(pi/64) q[9];
        cx q[9],q[4];
        rz(-pi/64) q[4];
        cx q[9],q[4];
        rz(pi/64) q[4];
        cx q[8],q[4];
        rz(-pi/32) q[4];
        cx q[8],q[4];
        rz(pi/32) q[4];
        cx q[7],q[4];
        rz(-pi/16) q[4];
        cx q[7],q[4];
        rz(pi/16) q[4];
        cx q[6],q[4];
        rz(-pi/8) q[4];
        cx q[6],q[4];
        rz(pi/8) q[4];
        cx q[5],q[4];
        rz(-pi/4) q[4];
        cx q[5],q[4];
        rz(pi/4) q[4];
        ry(pi/2) q[4];
        rx(pi) q[4];
        rz(pi/4) q[4];
        rz(pi/8) q[5];
        rz(pi/16) q[6];
        rz(pi/32) q[7];
        rz(pi/64) q[8];
        rz(pi/128) q[9];
        cx q[9],q[3];
        rz(-pi/128) q[3];
        cx q[9],q[3];
        rz(pi/128) q[3];
        cx q[8],q[3];
        rz(-pi/64) q[3];
        cx q[8],q[3];
        rz(pi/64) q[3];
        cx q[7],q[3];
        rz(-pi/32) q[3];
        cx q[7],q[3];
        rz(pi/32) q[3];
        cx q[6],q[3];
        rz(-pi/16) q[3];
        cx q[6],q[3];
        rz(pi/16) q[3];
        cx q[5],q[3];
        rz(-pi/8) q[3];
        cx q[5],q[3];
        rz(pi/8) q[3];
        cx q[4],q[3];
        rz(-pi/4) q[3];
        cx q[4],q[3];
        rz(pi/4) q[3];
        ry(pi/2) q[3];
        rx(pi) q[3];
        rz(pi/4) q[3];
        rz(pi/8) q[4];
        rz(pi/16) q[5];
        rz(pi/32) q[6];
        rz(pi/64) q[7];
        rz(pi/128) q[8];
        rz(pi/256) q[9];
        cx q[9],q[2];
        rz(-pi/256) q[2];
        cx q[9],q[2];
        rz(pi/256) q[2];
        cx q[8],q[2];
        rz(-pi/128) q[2];
        cx q[8],q[2];
        rz(pi/128) q[2];
        cx q[7],q[2];
        rz(-pi/64) q[2];
        cx q[7],q[2];
        rz(pi/64) q[2];
        cx q[6],q[2];
        rz(-pi/32) q[2];
        cx q[6],q[2];
        rz(pi/32) q[2];
        cx q[5],q[2];
        rz(-pi/16) q[2];
        cx q[5],q[2];
        rz(pi/16) q[2];
        cx q[4],q[2];
        rz(-pi/8) q[2];
        cx q[4],q[2];
        rz(pi/8) q[2];
        cx q[3],q[2];
        rz(-pi/4) q[2];
        cx q[3],q[2];
        rz(pi/4) q[2];
        ry(pi/2) q[2];
        rx(pi) q[2];
        rz(pi/4) q[2];
        rz(pi/8) q[3];
        rz(pi/16) q[4];
        rz(pi/32) q[5];
        rz(pi/64) q[6];
        rz(pi/128) q[7];
        rz(pi/256) q[8];
        rz(pi/512) q[9];
        cx q[9],q[1];
        rz(-pi/512) q[1];
        cx q[9],q[1];
        rz(pi/512) q[1];
        cx q[8],q[1];
        rz(-pi/256) q[1];
        cx q[8],q[1];
        rz(pi/256) q[1];
        cx q[7],q[1];
        rz(-pi/128) q[1];
        cx q[7],q[1];
        rz(pi/128) q[1];
        cx q[6],q[1];
        rz(-pi/64) q[1];
        cx q[6],q[1];
        rz(pi/64) q[1];
        cx q[5],q[1];
        rz(-pi/32) q[1];
        cx q[5],q[1];
        rz(pi/32) q[1];
        cx q[4],q[1];
        rz(-pi/16) q[1];
        cx q[4],q[1];
        rz(pi/16) q[1];
        cx q[3],q[1];
        rz(-pi/8) q[1];
        cx q[3],q[1];
        rz(pi/8) q[1];
        cx q[2],q[1];
        rz(-pi/4) q[1];
        cx q[2],q[1];
        rz(pi/4) q[1];
        ry(pi/2) q[1];
        rx(pi) q[1];
        rz(pi/4) q[1];
        rz(pi/8) q[2];
        rz(pi/16) q[3];
        rz(pi/32) q[4];
        rz(pi/64) q[5];
        rz(pi/128) q[6];
        rz(pi/256) q[7];
        rz(pi/512) q[8];
        rz(pi/1024) q[9];
        cx q[9],q[0];
        rz(-pi/1024) q[0];
        cx q[9],q[0];
        rz(pi/1024) q[0];
        cx q[8],q[0];
        rz(-pi/512) q[0];
        cx q[8],q[0];
        rz(pi/512) q[0];
        cx q[7],q[0];
        rz(-pi/256) q[0];
        cx q[7],q[0];
        rz(pi/256) q[0];
        cx q[6],q[0];
        rz(-pi/128) q[0];
        cx q[6],q[0];
        rz(pi/128) q[0];
        cx q[5],q[0];
        rz(-pi/64) q[0];
        cx q[5],q[0];
        rz(pi/64) q[0];
        cx q[4],q[0];
        rz(-pi/32) q[0];
        cx q[4],q[0];
        rz(pi/32) q[0];
        cx q[3],q[0];
        rz(-pi/16) q[0];
        cx q[3],q[0];
        rz(pi/16) q[0];
        cx q[2],q[0];
        rz(-pi/8) q[0];
        cx q[2],q[0];
        rz(pi/8) q[0];
        cx q[1],q[0];
        rz(-pi/4) q[0];
        cx q[1],q[0];
        rz(pi/4) q[0];
        ry(pi/2) q[0];
        rx(pi) q[0];
        cx q[0],q[9];
        cx q[1],q[8];
        cx q[2],q[7];
        cx q[3],q[6];
        cx q[4],q[5];
        cx q[5],q[4];
        cx q[4],q[5];
        cx q[6],q[3];
        cx q[3],q[6];
        cx q[7],q[2];
        cx q[2],q[7];
        cx q[8],q[1];
        cx q[1],q[8];
        cx q[9],q[0];
        cx q[0],q[9];
    """
    circuit = QiskitCircuit.from_qasm_str(qasm)

    compiled_circuit = compile(
        circuit, target_gateset=["u3", "cx"], custom_passes=[MPSPass()]
    )

    fidelity = np.abs(
        np.vdot(Statevector(circuit).data, Statevector(compiled_circuit).data)
    )

    assert compiled_circuit.count_ops().get("cx", 0) == 0
    assert np.round(fidelity, decimals=10) == 1.0


def repeated_qft_inverse_block(repeats=4):
    circuit = QiskitCircuit(8)
    base_block = QiskitCircuit(8)

    for target in range(8):
        base_block.h(target)
        for control in range(target + 1, 8):
            base_block.cp(np.pi / (2 ** (control - target)), control, target)

    for i in range(4):
        base_block.swap(i, 7 - i)

    for i in range(4):
        base_block.swap(i, 7 - i)

    for target in reversed(range(8)):
        for control in reversed(range(target + 1, 8)):
            base_block.cp(-np.pi / (2 ** (control - target)), control, target)
        base_block.h(target)

    for _ in range(repeats):
        circuit.compose(base_block, inplace=True)
    return circuit


def test_compile_cancels_repeated_qft_inverse_pairs():
    circuit = repeated_qft_inverse_block()

    compiled_circuit = compile(circuit, return_format="qiskit")

    assert len(circuit.data) > 0
    assert len(compiled_circuit.data) == 0


def test_compile_prefers_preset_reference_for_repeated_qft_structure():
    circuit = repeated_block(qft_forward_block(8), repeats=8)

    basis_translated_circuit = qiskit_transpile(
        circuit,
        basis_gates=UCCDefault1.DEFAULT_GATESET,
        optimization_level=0,
    )
    preset_reference = qiskit_transpile(
        circuit,
        basis_gates=list(UCCDefault1.DEFAULT_GATESET),
        optimization_level=3,
        layout_method="trivial",
        routing_method="none",
    )
    result_circuit = compile(circuit, return_format="qiskit")

    assert len(preset_reference.data) < len(basis_translated_circuit.data)
    assert result_circuit.count_ops() == preset_reference.count_ops()
    assert result_circuit.depth() == preset_reference.depth()


def test_compile_keeps_qaoa_ring_at_basis_translation_scale():
    circuit = qaoa_ring_circuit(8, layers=6)

    basis_translated_circuit = qiskit_transpile(
        circuit,
        basis_gates=UCCDefault1.DEFAULT_GATESET,
        optimization_level=0,
    )
    result_circuit = compile(circuit, return_format="qiskit")

    assert result_circuit.count_ops() == basis_translated_circuit.count_ops()
    assert result_circuit.depth() == basis_translated_circuit.depth()


def test_compile_qpe_style_prefers_non_regressing_candidate():
    circuit = repeated_block(qpe_roundtrip_block(3), repeats=6)

    basis_translated_circuit = qiskit_transpile(
        circuit,
        basis_gates=UCCDefault1.DEFAULT_GATESET,
        optimization_level=0,
    )
    result_circuit = compile(circuit, return_format="qiskit")

    assert len(result_circuit.data) < len(basis_translated_circuit.data)
    assert result_circuit.depth() < basis_translated_circuit.depth()


def test_structural_pre_simplify_reduces_grover_mirrored_prefix():
    circuit = repeated_block(grover_mirrored_block(6), repeats=4)

    simplified_circuit = _structural_pre_simplify(circuit)

    assert len(simplified_circuit.data) < len(circuit.data)
    assert simplified_circuit.depth() < circuit.depth()


def test_merge_adjacent_parameterized_gates_collapses_repeated_cp():
    circuit = QiskitCircuit(2)
    for _ in range(4):
        circuit.cp(math.pi / 8, 0, 1)

    merged_circuit = _merge_adjacent_parameterized_gates(circuit)

    assert len(merged_circuit.data) == 1
    assert merged_circuit.data[0].operation.name == "cp"
    assert merged_circuit.data[0].operation.params[0] == pytest.approx(
        math.pi / 2
    )


def test_compile_large_in_basis_identity_is_not_passed_through():
    """Regression: >=4000-gate in-basis sources used to be returned unchanged."""
    block = QiskitCircuit(3)
    block.h(0)
    block.h(0)
    block.cx(0, 1)
    block.cx(0, 1)
    circuit = repeated_block(block, 1000)

    result_circuit = compile(circuit, return_format="qiskit")

    assert result_circuit.size() == 0


def _line_backend(num_qubits):
    coupling_map = []
    for qubit in range(num_qubits - 1):
        coupling_map += [[qubit, qubit + 1], [qubit + 1, qubit]]
    return GenericBackendV2(
        num_qubits=num_qubits, coupling_map=coupling_map, seed=1
    )


def _repeated_composite_run_circuit(block_body, repeats):
    gate = block_body.to_gate(label="blk")
    circuit = QiskitCircuit(block_body.num_qubits)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(0.7, 2)
    for _ in range(repeats):
        circuit.append(gate, range(block_body.num_qubits))
    return circuit


def test_enforce_target_constraints_keeps_layout_of_routed_circuit():
    """Regression: re-transpiling a routed circuit reset its layout."""
    compile_module = importlib.import_module("ucc.compile")
    backend = _line_backend(3)
    compiler = UCCDefault1(target_backend=backend, seed_transpiler=7)

    logical = QiskitCircuit(3)
    logical.h(0)
    for _ in range(3):
        logical.cx(0, 1)
        logical.cx(1, 2)
        logical.cx(0, 2)
        logical.t(2)
    routed = qiskit_transpile(
        logical, backend=backend, optimization_level=1, seed_transpiler=7
    )
    assert routed.layout.routing_permutation() != [0, 1, 2]

    unchanged = compile_module._enforce_target_constraints(routed, compiler)
    assert Operator.from_circuit(unchanged).equiv(Operator(logical))

    # Append an identity pair that violates line connectivity, forcing a
    # second routing pass whose permutation must compose with the first.
    broken = routed.copy()
    broken.cx(0, 2)
    broken.cx(0, 2)
    rerouted = compile_module._enforce_target_constraints(broken, compiler)
    assert Operator.from_circuit(rerouted).equiv(Operator(logical))


def test_compile_with_target_gateset():
    """Test that the final circuit respects the user-defined gateset, no target device"""
    circuit = QiskitCircuit(2)
    circuit.cx(0, 1)
    circuit.h(0)

    target_gateset = {
        "ry",
        "rx",
        "cz",
    }
    result_circuit = compile(
        circuit,
        target_gateset=target_gateset,
    )

    assert set(op.name for op in result_circuit).issubset(target_gateset)


@pytest.mark.parametrize(
    "circuit_function", [qcnn_circuit, random_clifford_circuit]
)
@pytest.mark.parametrize("num_qubits", [6, 7, 8, 9, 10])
@pytest.mark.parametrize("seed", [1, 326, 5678, 12345])
def test_compilation_retains_gateset(circuit_function, num_qubits, seed):
    circuit = circuit_function(num_qubits, seed)
    transpiler = UCCDefault1()
    target_basis = transpiler.target_gateset
    transpiled_circuit = transpiler.run(circuit)
    dag = circuit_to_dag(transpiled_circuit)
    analysis_pass = GatesInBasis(basis_gates=target_basis)
    analysis_pass.run(dag)
    assert analysis_pass.property_set["all_gates_in_basis"]


# Test compilation accepts QASM circuits containing IF-ELSE
def test_compile_if_else():
    qasm = """
    OPENQASM 3;
    include "stdgates.inc";
    bit[3] data;
    bit[2] syndrome;
    qubit[3] q0;
    qubit[2] q1;

    syndrome[0] = measure q0[0];
    syndrome[1] = measure q1[1];
    if (syndrome[0]) {
        x q1[0];
    }
    if (syndrome[1]) {
        x q1[0];
    }
    """
    transpiled = compile(qasm, return_format="qiskit")
    assert isinstance(transpiled, QiskitCircuit)


@pytest.mark.parametrize(
    "circuit_function", [qcnn_circuit, random_clifford_circuit]
)
@pytest.mark.parametrize("num_qubits", [6, 7, 8, 9, 10, 15])
@pytest.mark.parametrize("seed", [1, 326, 5678, 12345])
def test_compiled_circuits_equivalent(circuit_function, num_qubits, seed):
    circuit = circuit_function(num_qubits, seed)
    transpiled = compile(circuit, return_format="qiskit")
    sv1 = Statevector(circuit)
    sv2 = Statevector(transpiled)
    assert sv1.equiv(sv2)


def test_compile_without_backend_never_requests_qiskit_opt3(monkeypatch):
    """The no-backend path is UCC's own pipeline, not Qiskit's O3 preset."""
    compile_module = importlib.import_module("ucc.compile")
    requested_levels = []
    real_transpile = compile_module.qiskit_transpile

    def spy_transpile(*args, **kwargs):
        requested_levels.append(kwargs.get("optimization_level"))
        return real_transpile(*args, **kwargs)

    monkeypatch.setattr(compile_module, "qiskit_transpile", spy_transpile)
    for circuit in (
        repeated_qft_inverse_block(),
        random_clifford_circuit(6, 1),
        QAOAAnsatz(SparsePauliOp(["ZZI", "IZZ"]), reps=2).assign_parameters(
            [0.1, 0.2, 0.3, 0.4]
        ),
    ):
        compile(circuit, return_format="qiskit")

    assert 3 not in requested_levels


def test_compile_without_backend_matches_structure_aware_ucc_default():
    circuit = repeated_qft_inverse_block()

    result_circuit = compile(circuit, return_format="qiskit")
    expected_circuit = UCCDefault1(structural_prepass=True).run(circuit)

    assert result_circuit.count_ops() == expected_circuit.count_ops()
    assert Operator(result_circuit).equiv(Operator(circuit))


def test_compile_collapses_controlled_powers_of_a_diagonal_unitary():
    """Phase estimation over a diagonal unitary: C-U^(2^k) is synthesized as
    one controlled phase polynomial instead of 2^k controlled copies."""
    from qiskit.circuit.library import PhaseEstimation

    step = QiskitCircuit(3)
    for qubit in range(3):
        step.rz(0.2 * (qubit + 1), qubit)
    step.cx(0, 1)
    step.rz(0.7, 1)
    step.cx(0, 1)
    circuit = QiskitCircuit(8)
    circuit.x([5, 6, 7])
    circuit.compose(PhaseEstimation(5, step), range(8), inplace=True)

    result_circuit = compile(circuit, return_format="qiskit")
    translated = qiskit_transpile(
        circuit,
        basis_gates=["cx", "rz", "rx", "ry", "h"],
        optimization_level=0,
    )

    assert result_circuit.size() * 5 < translated.size()
    assert Operator(result_circuit).equiv(Operator(circuit))


@pytest.mark.parametrize("block_needs_routing", [False, True])
def test_compile_with_backend_preserves_unitary_of_repeated_composite_run(
    block_needs_routing,
):
    """Regression for the old controller's shortcut, which composed
    separately routed pieces with unrelated layouts."""
    block_body = QiskitCircuit(3)
    if block_needs_routing:
        # A triangle of interactions cannot be laid out on a line.
        block_body.cx(0, 1)
        block_body.cx(1, 2)
        block_body.cx(0, 2)
        block_body.t(2)
    else:
        block_body.h(0)
        block_body.cx(0, 2)
        block_body.t(2)
        block_body.cx(2, 0)
        block_body.rz(0.3, 1)
        block_body.cx(0, 1)
    circuit = _repeated_composite_run_circuit(block_body, 600)

    for width in (3, 4):
        result_circuit = compile(
            circuit,
            return_format="qiskit",
            target_backend=_line_backend(width),
            seed_transpiler=7,
        )
        padded = QiskitCircuit(width)
        padded.compose(circuit, range(3), inplace=True)
        assert Operator.from_circuit(result_circuit).equiv(Operator(padded))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_compile_preserves_mirrored_and_conjugated_structure(seed):
    from qiskit.circuit.random import random_circuit

    shell = random_circuit(3, 3, max_operands=2, seed=seed)
    center = QiskitCircuit(3)
    center.x(1)
    center.cp(0.4, 0, 2)
    circuit = QiskitCircuit(3)
    for _ in range(4):
        circuit.compose(shell, inplace=True)
        circuit.compose(center, inplace=True)
        circuit.compose(shell.inverse(), inplace=True)

    result_circuit = compile(circuit, return_format="qiskit")
    translated = qiskit_transpile(
        circuit,
        basis_gates=["cx", "rz", "rx", "ry", "h"],
        optimization_level=0,
    )

    assert result_circuit.size() <= translated.size()
    assert Operator(result_circuit).equiv(Operator(circuit))
