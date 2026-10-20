import dataclasses
import sys
import warnings

from qbraid.programs.alias_manager import get_program_type_alias
from qbraid.transpiler import ConversionGraph
from qbraid.transpiler import transpile as translate
from qiskit import transpile as qiskit_transpile
from qiskit.circuit import QuantumCircuit as QiskitCircuit
from qiskit.transpiler import PassManager

from .transpilers.repeated_run import compile_repeated_run_on_backend
from .transpilers.structural_simplification import simplify_circuit
from .transpilers.ucc_defaults import UCCDefault1

# Specify the supported Python version range
REQUIRED_MAJOR = 3
MINOR_VERSION_MIN = 12
MINOR_VERSION_MAX = 13

current_major = sys.version_info.major
current_minor = sys.version_info.minor

if current_major != REQUIRED_MAJOR or not (
    MINOR_VERSION_MIN <= current_minor <= MINOR_VERSION_MAX
):
    warnings.warn(
        f"Warning: This package is designed for Python {REQUIRED_MAJOR}.{MINOR_VERSION_MIN}-{REQUIRED_MAJOR}.{MINOR_VERSION_MAX}. "
        f"You are using Python {current_major}.{current_minor}."
    )
supported_circuit_formats = ConversionGraph().nodes()


def _translate_to_qiskit(circuit):
    """Translate to Qiskit, but skip qBraid when already in Qiskit format."""

    if isinstance(circuit, QiskitCircuit):
        return circuit
    return translate(circuit, "qiskit")


def _translate_from_qiskit(circuit, return_format):
    """Translate from Qiskit only when the caller asked for another format."""

    if return_format == "qiskit":
        return circuit
    return translate(circuit, return_format)


def _satisfies_backend_target(circuit, backend):
    """Return True when every instruction is supported on its physical qubits."""

    target = getattr(backend, "target", None)
    if target is None or circuit.num_qubits != target.num_qubits:
        return False
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == "barrier":
            continue
        qargs = tuple(
            circuit.find_bit(qubit).index for qubit in instruction.qubits
        )
        if not target.instruction_supported(name, qargs):
            return False
    return True


def _enforce_backend_constraints_keeping_layout(circuit, compiler):
    """Re-translate an already laid-out circuit without losing its layout.

    Re-transpiling a physical circuit makes Qiskit treat it as virtual and
    attach a fresh layout, which silently discards the original initial layout
    and routing permutation. Keep the circuit when it already fits the
    backend; otherwise pin the trivial layout and compose any extra routing
    permutation onto the original one.
    """

    if _satisfies_backend_target(circuit, compiler.target_backend):
        return circuit

    retranslated = qiskit_transpile(
        circuit,
        backend=compiler.target_backend,
        optimization_level=0,
        initial_layout=list(range(circuit.num_qubits)),
        seed_transpiler=getattr(compiler, "seed_transpiler", None),
    )
    original_layout = circuit.layout
    extra_final_layout = (
        None
        if retranslated.layout is None
        else retranslated.layout.final_layout
    )
    if extra_final_layout is None:
        final_layout = original_layout.final_layout
    elif original_layout.final_layout is None:
        final_layout = extra_final_layout
    else:
        final_layout = original_layout.final_layout.compose(
            extra_final_layout, retranslated.qubits
        )
    retranslated._layout = dataclasses.replace(
        original_layout,
        final_layout=final_layout,
        _output_qubit_list=retranslated.qubits,
    )
    return retranslated


def _enforce_target_constraints(circuit, compiler):
    """Re-translate a circuit into the requested backend or basis.

    Custom passes run after the default pipeline and may introduce operations
    outside the requested target. Run a final, non-optimizing translation so
    the public ``compile()`` contract still holds for the returned circuit.
    """

    if compiler.target_backend is not None:
        if circuit.layout is None:
            return qiskit_transpile(
                circuit,
                backend=compiler.target_backend,
                optimization_level=0,
                seed_transpiler=getattr(compiler, "seed_transpiler", None),
            )
        return _enforce_backend_constraints_keeping_layout(circuit, compiler)

    return qiskit_transpile(
        circuit,
        basis_gates=list(compiler.target_gateset),
        optimization_level=0,
    )


def compile(
    circuit,
    return_format="original",
    target_gateset=None,
    target_backend=None,
    custom_passes=None,
    callback=None,
    seed_transpiler=None,
):
    """Compiles the provided quantum `circuit` by translating it to a Qiskit
    circuit, transpiling it, and returning the optimized circuit in the
    specified `return_format`.

    Compilation uses ``UCCDefault1(structural_prepass=True)``: structure-aware
    passes (``PreBasisStructuralSimplification`` and
    ``PhasePolynomialSynthesis``) run on the high-level circuit before it is
    lowered. Without a ``target_backend`` the rest is UCC's local-pass
    pipeline; with one, layout, routing and hardware optimization use
    Qiskit's level-3 preset, and a circuit dominated by one repeated
    composite block is mapped one block at a time
    (``ucc.transpilers.repeated_run``).

    Args:
        circuit (object): The quantum circuit to be compiled.
        return_format (str): The format in which your circuit will be returned.
            e.g., "TKET", "OpenQASM2". Check ``ucc.supported_circuit_formats``.
            Defaults to the format of the input circuit.
        target_gateset (set[str]): (optional) The gateset to compile the circuit to.
            e.g. {"cx", "rx",...}. Defaults to the gate set of the target device if available. If no `target_gateset` or ` target_backend` is provided, defaults to {"cx", "rz", "rx", "ry", "h"}.
        target_backend (qiskit.providers.backend): (optional)
            The target device  to compile the circuit for. Can be specified as a Qiskit backend. If None, all-to-all connectivity is assumed. If a `target_backend` is specified, `target_backend.operation_names` supersedes the `target_gateset`.
        custom_passes (list[qiskit.transpiler.TransformationPass]): (optional)
            A list of custom passes to apply after the default set
            of passes. Defaults to None.
        callback: A callback function that will be called after each pass execution. The
                function will be called with 5 keyword arguments::

                    pass_ (Pass): the pass being run
                    dag (DAGCircuit): the dag output of the pass
                    time (float): the time to execute the pass
                    property_set (PropertySet): the property set
                    count (int): the index for the pass execution
        seed_transpiler (int): (optional)
            Fixed seed forwarded to Qiskit's preset pass manager when a
            ``target_backend`` is given, for reproducible layout and routing.

    Returns:
        object: The compiled circuit in the specified format.
    """
    if return_format == "original":
        return_format = get_program_type_alias(circuit)

    source_circuit = _translate_to_qiskit(circuit)
    compiler = UCCDefault1(
        target_backend=target_backend,
        target_gateset=target_gateset,
        seed_transpiler=seed_transpiler,
        structural_prepass=True,
    )

    compiled_circuit = None
    if target_backend is not None and callback is None:
        # Simplify first so cancellations are visible to run detection; if
        # no run is reused, the simplified circuit is what gets compiled.
        source_circuit = simplify_circuit(source_circuit)
        compiled_circuit = compile_repeated_run_on_backend(
            source_circuit, target_backend, seed_transpiler=seed_transpiler
        )
    if compiled_circuit is None:
        compiled_circuit = compiler.run(source_circuit, callback=callback)

    if custom_passes:
        custom_pass_manager = PassManager()
        custom_pass_manager.append(custom_passes)
        compiled_circuit = custom_pass_manager.run(
            compiled_circuit, callback=callback
        )

    compiled_circuit = _enforce_target_constraints(compiled_circuit, compiler)
    return _translate_from_qiskit(compiled_circuit, return_format)
