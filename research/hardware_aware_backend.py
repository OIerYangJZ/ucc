from __future__ import annotations

from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CXGate, IGate, PhaseGate, SXGate, UGate
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target


class LineBackend(BackendV2):
    """A simple backend-aware target with bidirectional linear connectivity."""

    def __init__(self, num_qubits: int, name: str | None = None) -> None:
        super().__init__(name=name or f"line_backend_{num_qubits}")
        self._num_qubits = num_qubits
        self._target = Target(f"Line backend with {num_qubits} qubits")

        lam = Parameter("λ")
        phi = Parameter("φ")
        theta = Parameter("ϴ")

        single_qubit_props = {(qubit,): None for qubit in range(num_qubits)}
        self._target.add_instruction(PhaseGate(lam), single_qubit_props)
        self._target.add_instruction(SXGate(), single_qubit_props)
        self._target.add_instruction(
            UGate(theta, phi, lam), single_qubit_props
        )
        self._target.add_instruction(Measure(), single_qubit_props)
        self._target.add_instruction(IGate(), single_qubit_props)

        cx_props = {
            edge: None
            for qubit in range(num_qubits - 1)
            for edge in ((qubit, qubit + 1), (qubit + 1, qubit))
        }
        self._target.add_instruction(CXGate(), cx_props)

        self.options.set_validator("shots", (1, 4096))
        self.options.set_validator("memory", bool)

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return 1024

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @classmethod
    def _default_options(cls):
        return Options(shots=1024, memory=False)

    def run(self, circuit, **kwargs):
        return None
