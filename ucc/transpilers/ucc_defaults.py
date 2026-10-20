# Construct a custom compiler
import os

try:
    from qiskit.utils.parallel import default_num_processes
except ImportError:
    # Qiskit 1.0.0 doesn't have this function, so we make it ourselves
    from qiskit.utils.parallel import CPU_COUNT

    def default_num_processes():
        return CPU_COUNT


from qiskit.circuit import CONTROL_FLOW_OP_NAMES
from qiskit.providers import Backend
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit import user_config
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.circuit.equivalence_library import (
    SessionEquivalenceLibrary,
)
from qiskit.transpiler.passes import (
    BasisTranslator,
    ConsolidateBlocks,
    CollectCliffords,
    HighLevelSynthesis,
    HLSConfig,
    CommutativeCancellation,
    Collect2qBlocks,
    UnitarySynthesis,
    Optimize1qGatesDecomposition,
)
from typing import Optional

from .phase_polynomial import PhasePolynomialSynthesis
from .structural_simplification import PreBasisStructuralSimplification


CONFIG = user_config.get_config()


class LowestCostLocalOptimization(TransformationPass):
    """Run alternative local-pass pipelines and keep the cheapest result.

    Each pipeline is a zero-argument callable returning fresh passes. Results
    are translated to ``basis_gates`` and compared by (two-qubit gates, all
    gates, depth). Passes inside the pipelines do not report to an outer
    ``callback``.
    """

    def __init__(self, pipelines, basis_gates):
        super().__init__()
        self._pipelines = pipelines
        self._basis_gates = _with_control_flow(basis_gates)

    def run(self, dag):
        circuit = dag_to_circuit(dag)
        best = None
        for pipeline in self._pipelines:
            candidate = PassManager(
                pipeline()
                + [
                    BasisTranslator(
                        SessionEquivalenceLibrary, self._basis_gates
                    )
                ]
            ).run(circuit)
            two_qubit = sum(
                1
                for instruction in candidate.data
                if len(instruction.qubits) > 1
            )
            cost = (two_qubit, candidate.size(), candidate.depth())
            if best is None or cost < best[0]:
                best = (cost, candidate)
        return circuit_to_dag(best[1])


def backend_pass_manager(
    backend,
    seed_transpiler=None,
    structural_prepass=False,
    initial_layout=None,
):
    """Qiskit's level-3 preset for ``backend``, optionally with UCC's
    structure-aware passes in the ``pre_init`` stage.

    The structure-aware passes run before layout: phase-polynomial synthesis
    emits CNOTs between arbitrary qubits, so it must not run on a routed
    circuit.
    """
    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=seed_transpiler,
        initial_layout=initial_layout,
    )
    if structural_prepass:
        pass_manager.pre_init = PassManager(
            [
                PreBasisStructuralSimplification(),
                PhasePolynomialSynthesis(
                    max_block_qubits=_PRE_TRANSLATION_MAX_BLOCK_QUBITS
                ),
            ]
        )
    return pass_manager


class UCCDefault1:
    DEFAULT_GATESET = {"cx", "rz", "rx", "ry", "h"}

    def __init__(
        self,
        local_iterations: int = 1,
        target_backend: Optional[Backend] = None,
        target_gateset: Optional[set] = None,
        seed_transpiler: Optional[int] = None,
        structural_prepass: bool = False,
    ):
        """
        Create a new instance of UCCDefault1 compiler

            Args:
                local_iterations (int): Number of times to run the local passes
                target_backend (qiskit.providers.Backend): (Optional) The       target backend device to compile the circuit for
                target_gateset (set[str]): (Optional) The gateset to compile the circuit to. e.g. {"cx", "rx",...}. `target_backend` takes precedence if it provides a basis gateset.
                seed_transpiler (int): (Optional) Seed forwarded to Qiskit's preset pass manager when a backend is given.
                structural_prepass (bool): (Optional) Run
                    :class:`~ucc.transpilers.structural_simplification.PreBasisStructuralSimplification`
                    before basis translation, so exact inverse / repeated
                    structure in high-level instructions is simplified before
                    it is expanded. Off by default. When enabled, ``run`` also
                    translates its input to the target gateset itself, so it
                    can be given the original high-level circuit.

        If neither target_backend or target_gateset resolve to a gateset, defaults to {"cx", "rz", "rx", "ry", "h"}.

        """
        self.pass_manager = PassManager()
        self.target_backend = target_backend
        self.seed_transpiler = seed_transpiler

        if self.target_backend is None:
            # If no backend is provided, use the provided gateset or default gateset
            self.target_gateset = (
                target_gateset
                if target_gateset is not None
                else self.DEFAULT_GATESET
            )
        elif hasattr(self.target_backend, "target") and hasattr(
            self.target_backend.target, "operation_names"
        ):
            # If a backend is provided, use its target's operation names as the gateset
            self.target_gateset = self.target_backend.target.operation_names
        else:
            raise ValueError(
                "Provided backend does not provide a target with operation names"
            )

        if self.target_backend is None:
            if structural_prepass:
                self._add_pre_basis_passes()
                self._add_structure_aware_local_passes(local_iterations)
            else:
                self._add_local_passes(local_iterations)
        else:
            self.pass_manager = backend_pass_manager(
                self.target_backend,
                seed_transpiler=self.seed_transpiler,
                structural_prepass=structural_prepass,
            )

    @property
    def default_passes(self):
        return

    def _add_pre_basis_passes(self):
        basis_gates = list(self.target_gateset)
        self.pass_manager.append(PreBasisStructuralSimplification())
        self.pass_manager.append(
            PhasePolynomialSynthesis(
                max_block_qubits=_PRE_TRANSLATION_MAX_BLOCK_QUBITS
            )
        )
        self.pass_manager.append(UnitarySynthesis(basis_gates=basis_gates))
        self.pass_manager.append(
            HighLevelSynthesis(
                basis_gates=_with_control_flow(basis_gates),
                equivalence_library=SessionEquivalenceLibrary,
                qubits_initially_zero=False,
            )
        )
        self.pass_manager.append(
            BasisTranslator(
                SessionEquivalenceLibrary, _with_control_flow(basis_gates)
            )
        )

    def _add_structure_aware_local_passes(self, local_iterations):
        # CNOT + phase runs left by translation are resynthesized as phase
        # polynomials. UCC's default local passes force every two-qubit
        # block through unitary synthesis: that minimizes CNOTs (QFT 3,570
        # -> 2,740) but can triple the single-qubit gate count; replacing a
        # block only when it is cheaper keeps the circuit compact. Neither
        # wins everywhere, so both run and the cheaper result is kept.
        basis_gates = list(self.target_gateset)
        self.pass_manager.append(
            PhasePolynomialSynthesis(
                max_block_qubits=_POST_TRANSLATION_MAX_BLOCK_QUBITS
            )
        )
        self.pass_manager.append(
            BasisTranslator(
                SessionEquivalenceLibrary, _with_control_flow(basis_gates)
            )
        )

        def forced_consolidation():
            passes = []
            for _ in range(local_iterations):
                passes += [
                    Optimize1qGatesDecomposition(),
                    CommutativeCancellation(),
                    Collect2qBlocks(),
                    ConsolidateBlocks(force_consolidate=True),
                    UnitarySynthesis(basis_gates=basis_gates),
                    # The default pipeline's greedy Clifford resynthesis is
                    # left out: on every benchmark circuit it changed no
                    # gate count but took 2.6x longer.
                ]
            return passes

        def cost_checked_consolidation():
            passes = []
            for _ in range(local_iterations):
                passes += [
                    Optimize1qGatesDecomposition(basis=basis_gates),
                    CommutativeCancellation(basis_gates=basis_gates),
                    Collect2qBlocks(),
                    ConsolidateBlocks(basis_gates=basis_gates),
                    UnitarySynthesis(basis_gates=basis_gates),
                    Optimize1qGatesDecomposition(basis=basis_gates),
                ]
            return passes

        self.pass_manager.append(
            LowestCostLocalOptimization(
                [forced_consolidation, cost_checked_consolidation],
                basis_gates,
            )
        )

    def _add_local_passes(self, local_iterations):
        for _ in range(local_iterations):
            self.pass_manager.append(Optimize1qGatesDecomposition())
            self.pass_manager.append(CommutativeCancellation())
            self.pass_manager.append(Collect2qBlocks())
            self.pass_manager.append(ConsolidateBlocks(force_consolidate=True))
            self.pass_manager.append(
                UnitarySynthesis(basis_gates=self.target_gateset)
            )
            # self.pass_manager.append(Optimize1qGatesDecomposition(basis=self._1q_basis))
            self.pass_manager.append(CollectCliffords())
            self.pass_manager.append(
                HighLevelSynthesis(hls_config=HLSConfig(clifford=["greedy"]))
            )

            # Add following passes if merging single qubit rotations that are interrupted by a commuting 2 qubit gate is desired
            # self.pass_manager.append(Optimize1qGatesSimpleCommutation(basis=self._1q_basis))

    def run(self, circuits, callback=None):
        """
        Run the pass manager on the given circuit(s).

            Args:
                circuits (QuantumCircuit or list[QuantumCircuit]): Circuit(s) to transpile
                callback: A callback function that will be called after each pass execution. The
                function will be called with 5 keyword arguments::

                    pass_ (Pass): the pass being run
                    dag (DAGCircuit): the dag output of the pass
                    time (float): the time to execute the pass
                    property_set (PropertySet): the property set
                    count (int): the index for the pass execution
        """
        return self.pass_manager.run(circuits, callback=callback)


# Parity-network synthesis dominates compile time on wide blocks, which on the
# benchmark circuits were always rejected (e.g. 98-qubit runs in a basis-level
# QFT). Accepted blocks: up to 32 qubits before translation (a QAOA cost layer
# on K32) and up to 8 after it.
_PRE_TRANSLATION_MAX_BLOCK_QUBITS = 32
_POST_TRANSLATION_MAX_BLOCK_QUBITS = 16


def _with_control_flow(basis_gates):
    # Like qiskit.transpile, keep control-flow ops so translation recurses
    # into their blocks instead of rejecting them.
    return list(basis_gates) + sorted(CONTROL_FLOW_OP_NAMES)


def _get_trial_count(default_trials=5):
    if CONFIG.get("sabre_all_threads", None) or os.getenv(
        "QISKIT_SABRE_ALL_THREADS"
    ):
        return default_num_processes()
    return default_trials
