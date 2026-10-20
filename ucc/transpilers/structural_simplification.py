"""Pre-basis structural simplification.

Generic optimizers see a circuit only after it has been lowered to basis
gates, by which point exact structure such as ``U U^-1`` pairs, repeated
blocks, and commuting diagonal-phase layers has been expanded into many
primitive gates (UCC issue #662). The helpers in this module exploit that
structure on the original high-level instructions instead. Every rewrite
either removes an exact inverse pair, merges commuting rotations/phases, or
simplifies one copy of a repeated block and reuses it, so the circuit's
unitary is preserved by construction.

:class:`PreBasisStructuralSimplification` wraps them as a Qiskit
``TransformationPass``; run it before basis translation.
"""

import math
import os
from dataclasses import dataclass

from qiskit.circuit import CircuitInstruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import CommutativeInverseCancellation

_MAX_STRUCTURAL_BLOCK_SIZE = 256
_MIN_REPEATED_BLOCK_REPEATS = 4
_MAX_GLOBAL_STRUCTURAL_SCAN_SIZE = 50000
_FAST_PREFIX_SCAN_SIZE = 4096
_MIN_DOMINANT_REPEAT_FRACTION = 0.8
_PARAMETER_MERGE_TOLERANCE = 1e-12
_MERGEABLE_PARAMETERIZED_GATES = {"rx", "ry", "rz", "p", "cp"}
_DIAGONAL_PHASE_GATES = {"rz", "p", "cp", "cz"}
_MIN_BLOCKWISE_REPEAT_PRESIMPLIFY_SIZE = 10000
_MAX_CUSTOM_GATES_FOR_COMMUTATION = 32
_STANDARD_GATE_NAMES = frozenset(get_standard_gate_name_mapping()) | {
    "measure",
    "barrier",
    "reset",
    "delay",
}


@dataclass(frozen=True)
class _DiagonalPhaseSpanNode:
    """Semantic IR node for a contiguous commuting diagonal-phase span."""

    start_index: int
    end_index: int


def _build_circuit_from_instructions(template_circuit, instructions):
    """Build a new circuit with the same registers from a list of instructions."""

    rebuilt_circuit = template_circuit.copy_empty_like()
    rebuilt_circuit.global_phase = template_circuit.global_phase
    for instruction in instructions:
        rebuilt_circuit.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )
    return rebuilt_circuit


def _normalize_param(param):
    """Return a hashable representation for instruction parameters."""

    if isinstance(param, (int, float)):
        return round(float(param), 12)
    return str(param)


def _instruction_signature(circuit, instruction):
    """Compute a stable signature for repeated-block detection."""

    return (
        instruction.operation.name,
        tuple(
            _normalize_param(param) for param in instruction.operation.params
        ),
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
        tuple(circuit.find_bit(clbit).index for clbit in instruction.clbits),
    )


def _inverse_instruction_signature(circuit, instruction):
    """Compute the signature of an instruction's inverse when available."""

    try:
        inverse_operation = instruction.operation.inverse()
    except Exception:
        return None

    return (
        inverse_operation.name,
        tuple(_normalize_param(param) for param in inverse_operation.params),
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
        tuple(circuit.find_bit(clbit).index for clbit in instruction.clbits),
    )


def _copy_operation_with_parameter(operation, parameter):
    """Return a mutable copy of a single-parameter operation."""

    try:
        copied_operation = type(operation)(parameter)
    except Exception:
        copied_operation = (
            operation.to_mutable()
            if hasattr(operation, "to_mutable")
            else operation.copy()
        )
        copied_operation.params = [parameter]
    return copied_operation


def _can_merge_adjacent_parameterized_gates(
    left_instruction, right_instruction
):
    """Return True when two adjacent single-parameter gates can be merged."""

    left_operation = left_instruction.operation
    right_operation = right_instruction.operation
    return (
        left_operation.name == right_operation.name
        and left_operation.name in _MERGEABLE_PARAMETERIZED_GATES
        and len(left_operation.params) == 1
        and len(right_operation.params) == 1
        and isinstance(left_operation.params[0], (int, float))
        and isinstance(right_operation.params[0], (int, float))
        and left_instruction.qubits == right_instruction.qubits
        and left_instruction.clbits == right_instruction.clbits
    )


def _is_diagonal_phase_instruction(instruction):
    """Return True when an instruction is a commuting diagonal phase gate."""

    operation = instruction.operation
    if operation.name not in _DIAGONAL_PHASE_GATES:
        return False
    return not getattr(operation, "condition", None)


def _diagonal_phase_signature(circuit, instruction):
    """Return a stable signature for commuting diagonal phase aggregation."""

    return (
        instruction.operation.name,
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
        tuple(circuit.find_bit(clbit).index for clbit in instruction.clbits),
    )


def _diagonal_phase_parameter(operation):
    """Return the additive phase parameter for a diagonal gate when supported."""

    if operation.name == "cz":
        return math.pi
    if len(operation.params) == 1 and isinstance(
        operation.params[0], (int, float)
    ):
        return float(operation.params[0])
    return None


def _build_diagonal_phase_span_nodes(circuit):
    """Build contiguous semantic nodes for commuting diagonal-phase spans."""

    nodes = []
    start_index = None
    for index, instruction in enumerate(circuit.data):
        if _is_diagonal_phase_instruction(instruction):
            if start_index is None:
                start_index = index
            continue
        if start_index is not None and index > start_index:
            nodes.append(_DiagonalPhaseSpanNode(start_index, index))
            start_index = None
    if start_index is not None and len(circuit.data) > start_index:
        nodes.append(_DiagonalPhaseSpanNode(start_index, len(circuit.data)))
    return nodes


def _canonicalize_diagonal_phase_span(circuit, node):
    """Canonicalize a commuting diagonal-phase span by aggregating equal terms."""

    phase_terms = {}
    ordered_keys = []
    for instruction in circuit.data[node.start_index : node.end_index]:
        parameter = _diagonal_phase_parameter(instruction.operation)
        if parameter is None:
            return _build_circuit_from_instructions(
                circuit, list(circuit.data[node.start_index : node.end_index])
            )
        signature = _diagonal_phase_signature(circuit, instruction)
        if signature not in phase_terms:
            phase_terms[signature] = 0.0
            ordered_keys.append(signature)
        phase_terms[signature] += parameter

    rebuilt_span = circuit.copy_empty_like()
    rebuilt_span.global_phase = 0
    for gate_name, qubit_indices, clbit_indices in ordered_keys:
        parameter = phase_terms[(gate_name, qubit_indices, clbit_indices)]
        if abs(parameter) <= _PARAMETER_MERGE_TOLERANCE:
            continue
        if gate_name == "rz":
            rebuilt_span.rz(parameter, qubit_indices[0])
        elif gate_name == "p":
            rebuilt_span.p(parameter, qubit_indices[0])
        elif gate_name == "cp":
            rebuilt_span.cp(parameter, qubit_indices[0], qubit_indices[1])
        elif gate_name == "cz":
            # Each cz contributes pi; an even count cancels to the identity.
            residue = parameter % (2 * math.pi)
            if (
                min(residue, 2 * math.pi - residue)
                <= _PARAMETER_MERGE_TOLERANCE
            ):
                continue
            rebuilt_span.cz(qubit_indices[0], qubit_indices[1])
        else:
            return _build_circuit_from_instructions(
                circuit, list(circuit.data[node.start_index : node.end_index])
            )
    return rebuilt_span


def _canonicalize_diagonal_phase_spans(circuit):
    """Aggregate every contiguous commuting diagonal-phase span once."""

    if os.environ.get("UCC_DISABLE_FOURIER_LAYER_IR") == "1":
        return circuit

    diagonal_nodes = _build_diagonal_phase_span_nodes(circuit)
    if not diagonal_nodes:
        return circuit

    rebuilt_circuit = circuit.copy_empty_like()
    rebuilt_circuit.global_phase = circuit.global_phase
    next_index = 0
    used_semantic_node = False

    for node in diagonal_nodes:
        _append_instruction_sequence(
            rebuilt_circuit, circuit.data[next_index : node.start_index]
        )
        simplified_span = _canonicalize_diagonal_phase_span(circuit, node)
        for instruction in simplified_span.data:
            _append_mapped_instruction(
                rebuilt_circuit, simplified_span, instruction
            )
        used_semantic_node = True
        next_index = node.end_index

    _append_instruction_sequence(rebuilt_circuit, circuit.data[next_index:])

    if not used_semantic_node:
        return circuit

    return rebuilt_circuit


def _merge_adjacent_parameterized_gates(circuit):
    """Merge adjacent additive rotation/phase gates on the same wires."""

    rebuilt_circuit = circuit.copy_empty_like()
    rebuilt_circuit.global_phase = circuit.global_phase
    pending_instruction = None

    def flush_pending_instruction():
        nonlocal pending_instruction
        if pending_instruction is None:
            return
        rebuilt_circuit.append(
            pending_instruction.operation,
            pending_instruction.qubits,
            pending_instruction.clbits,
        )
        pending_instruction = None

    for instruction in circuit.data:
        if pending_instruction is None:
            pending_instruction = instruction
            continue

        if _can_merge_adjacent_parameterized_gates(
            pending_instruction, instruction
        ):
            merged_parameter = float(
                pending_instruction.operation.params[0]
            ) + float(instruction.operation.params[0])
            if abs(merged_parameter) <= _PARAMETER_MERGE_TOLERANCE:
                pending_instruction = None
            else:
                pending_instruction = CircuitInstruction(
                    _copy_operation_with_parameter(
                        pending_instruction.operation, merged_parameter
                    ),
                    pending_instruction.qubits,
                    pending_instruction.clbits,
                )
            continue

        flush_pending_instruction()
        pending_instruction = instruction

    flush_pending_instruction()
    if len(rebuilt_circuit.data) >= len(circuit.data):
        return circuit

    return rebuilt_circuit


def _run_commutative_inverse_cancellation(circuit):
    """Run Qiskit's commutative inverse cancellation when it is affordable.

    Qiskit's commutation checker caches standard gates only; for custom
    composite gates it rebuilds both operators on every pairwise check,
    which is quadratic in their number (600 copies of one custom gate took
    35 s). Exact adjacent ``U U^-1`` pairs are still removed by
    ``_cancel_adjacent_inverse_blocks``.
    """

    custom_instructions = sum(
        1
        for instruction in circuit.data
        if instruction.operation.name not in _STANDARD_GATE_NAMES
    )
    if custom_instructions > _MAX_CUSTOM_GATES_FOR_COMMUTATION:
        return circuit
    try:
        return PassManager([CommutativeInverseCancellation()]).run(circuit)
    except (QiskitError, ValueError):
        return circuit


def _find_repeated_run(
    circuit,
    max_block_size=_MAX_STRUCTURAL_BLOCK_SIZE,
    min_repeats=_MIN_REPEATED_BLOCK_REPEATS,
):
    """Detect the largest repeated exact block anywhere in a circuit."""

    instruction_count = len(circuit.data)
    max_candidate_size = min(max_block_size, instruction_count // min_repeats)
    if max_candidate_size < 1:
        return None

    signatures = [
        _instruction_signature(circuit, instruction)
        for instruction in circuit.data
    ]
    best_match = None
    best_coverage = 0

    for block_size in range(1, max_candidate_size + 1):
        start_index = 0
        while start_index + min_repeats * block_size <= instruction_count:
            block = signatures[start_index : start_index + block_size]
            repeat_count = 1
            while (
                start_index + (repeat_count + 1) * block_size
                <= instruction_count
            ):
                candidate_start = start_index + repeat_count * block_size
                candidate_end = candidate_start + block_size
                if signatures[candidate_start:candidate_end] != block:
                    break
                repeat_count += 1

            coverage = repeat_count * block_size
            if repeat_count >= min_repeats and coverage > best_coverage:
                best_match = (start_index, block_size, repeat_count)
                best_coverage = coverage

            start_index += coverage if repeat_count > 1 else 1

    return best_match


def _find_repeated_prefix(
    circuit,
    max_block_size=_MAX_STRUCTURAL_BLOCK_SIZE,
    min_repeats=_MIN_REPEATED_BLOCK_REPEATS,
):
    """Detect the largest repeated exact prefix block in a circuit."""

    instruction_count = len(circuit.data)
    max_candidate_size = min(max_block_size, instruction_count // min_repeats)
    if max_candidate_size < 1:
        return None

    signatures = [
        _instruction_signature(circuit, instruction)
        for instruction in circuit.data
    ]

    if instruction_count > _FAST_PREFIX_SCAN_SIZE:
        full_coverage_divisors = [
            block_size
            for block_size in range(max_candidate_size, 0, -1)
            if instruction_count % block_size == 0
        ]
        for block_size in full_coverage_divisors:
            repeat_count = instruction_count // block_size
            if repeat_count < min_repeats:
                continue

            prefix = signatures[:block_size]
            if all(
                signatures[start_index : start_index + block_size] == prefix
                for start_index in range(
                    block_size, instruction_count, block_size
                )
            ):
                return (block_size, repeat_count)

        best_match = None
        best_coverage = 0

        for block_size in range(max_candidate_size, 0, -1):
            max_coverage = block_size * (instruction_count // block_size)
            if max_coverage <= best_coverage:
                continue

            prefix = signatures[:block_size]
            repeat_count = 1
            while (repeat_count + 1) * block_size <= instruction_count:
                start_index = repeat_count * block_size
                end_index = start_index + block_size
                if signatures[start_index:end_index] != prefix:
                    break
                repeat_count += 1

            coverage = repeat_count * block_size
            if repeat_count >= min_repeats and coverage > best_coverage:
                best_match = (block_size, repeat_count)
                best_coverage = coverage
                if coverage == instruction_count:
                    break

        return best_match

    best_match = None
    best_coverage = 0

    for block_size in range(1, max_candidate_size + 1):
        prefix = signatures[:block_size]
        repeat_count = 1
        while (repeat_count + 1) * block_size <= instruction_count:
            start_index = repeat_count * block_size
            end_index = start_index + block_size
            if signatures[start_index:end_index] != prefix:
                break
            repeat_count += 1

        coverage = repeat_count * block_size
        if repeat_count >= min_repeats and coverage > best_coverage:
            best_match = (block_size, repeat_count)
            best_coverage = coverage

    return best_match


def _simplify_repeated_prefix(circuit):
    """Simplify a repeated leading block once and reuse the result."""

    repeated_prefix = _find_repeated_prefix(circuit)
    if repeated_prefix is None:
        return circuit

    block_size, repeat_count = repeated_prefix
    run_end = block_size * repeat_count
    block_instructions = list(circuit.data[:block_size])
    block_circuit = _build_circuit_from_instructions(
        circuit, block_instructions
    )
    simplified_block = _run_commutative_inverse_cancellation(block_circuit)

    if len(simplified_block.data) >= block_size:
        return circuit

    rebuilt_circuit = circuit.copy_empty_like()
    rebuilt_circuit.global_phase = circuit.global_phase
    for _ in range(repeat_count):
        for instruction in simplified_block.data:
            rebuilt_circuit.append(
                instruction.operation,
                instruction.qubits,
                instruction.clbits,
            )

    for instruction in circuit.data[run_end:]:
        rebuilt_circuit.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )

    if len(rebuilt_circuit.data) >= len(circuit.data):
        return circuit

    return rebuilt_circuit


def _find_self_inverse_prefix_suffix_overlap(circuit, block_size):
    """Return the overlap size when a block suffix cancels the next prefix."""

    signatures = [
        _instruction_signature(circuit, instruction)
        for instruction in circuit.data[:block_size]
    ]
    inverse_signatures = [
        _inverse_instruction_signature(circuit, instruction)
        for instruction in circuit.data[:block_size]
    ]
    max_overlap = min(block_size // 2, 64)
    for overlap in range(max_overlap, 0, -1):
        suffix = signatures[block_size - overlap : block_size]
        inverse_prefix = list(reversed(inverse_signatures[:overlap]))
        if any(signature is None for signature in inverse_prefix):
            continue
        if suffix == inverse_prefix:
            return overlap
    return 0


def _compress_repeated_prefix_boundary_cancellation(circuit):
    """Compress repeated prefixes when adjacent copies cancel at the boundary."""

    repeated_prefix = _find_repeated_prefix(circuit)
    if repeated_prefix is None:
        return circuit

    block_size, repeat_count = repeated_prefix
    if repeat_count < 2:
        return circuit

    overlap = _find_self_inverse_prefix_suffix_overlap(circuit, block_size)
    if overlap == 0:
        return circuit

    middle_start = overlap
    middle_end = block_size - overlap
    rebuilt_circuit = circuit.copy_empty_like()
    rebuilt_circuit.global_phase = circuit.global_phase

    _append_instruction_sequence(rebuilt_circuit, circuit.data[:overlap])
    middle_instructions = circuit.data[middle_start:middle_end]
    for _ in range(repeat_count):
        _append_instruction_sequence(rebuilt_circuit, middle_instructions)
    _append_instruction_sequence(
        rebuilt_circuit, circuit.data[block_size - overlap : block_size]
    )
    _append_instruction_sequence(
        rebuilt_circuit, circuit.data[block_size * repeat_count :]
    )

    if len(rebuilt_circuit.data) >= len(circuit.data):
        return circuit

    return rebuilt_circuit


def _find_adjacent_inverse_blocks(
    circuit, max_block_size=_MAX_STRUCTURAL_BLOCK_SIZE
):
    """Detect the largest adjacent exact block followed by its inverse."""

    instruction_count = len(circuit.data)
    max_candidate_size = min(max_block_size, instruction_count // 2)
    if max_candidate_size < 1:
        return None

    signatures = [
        _instruction_signature(circuit, instruction)
        for instruction in circuit.data
    ]
    inverse_signatures = [
        _inverse_instruction_signature(circuit, instruction)
        for instruction in circuit.data
    ]
    # A block X followed by X^-1 is centered at the boundary c between them:
    # instruction c-1-k must equal the inverse of instruction c+k for every
    # k < len(X). Grow that radius from each center; this is linear in the
    # circuit plus the matched length, instead of re-slicing every
    # (size, start) pair. Ties keep the earliest start, as before.
    best_match = None
    best_size = 0
    for center in range(1, instruction_count):
        limit = min(max_candidate_size, center, instruction_count - center)
        radius = 0
        while (
            radius < limit
            and inverse_signatures[center + radius] is not None
            and signatures[center - 1 - radius]
            == inverse_signatures[center + radius]
        ):
            radius += 1
        if radius > best_size:
            best_size = radius
            best_match = (center - radius, radius)

    return best_match


def _simplify_repeated_run(circuit):
    """Simplify a repeated block once and reuse the result."""

    repeated_run = _find_repeated_run(circuit)
    if repeated_run is None:
        return circuit

    start_index, block_size, repeat_count = repeated_run
    run_end = start_index + block_size * repeat_count
    block_instructions = list(
        circuit.data[start_index : start_index + block_size]
    )
    block_circuit = _build_circuit_from_instructions(
        circuit, block_instructions
    )
    simplified_block = _run_commutative_inverse_cancellation(block_circuit)

    if len(simplified_block.data) >= block_size:
        return circuit

    rebuilt_circuit = circuit.copy_empty_like()
    rebuilt_circuit.global_phase = circuit.global_phase
    for instruction in circuit.data[:start_index]:
        rebuilt_circuit.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )

    for _ in range(repeat_count):
        for instruction in simplified_block.data:
            rebuilt_circuit.append(
                instruction.operation,
                instruction.qubits,
                instruction.clbits,
            )

    for instruction in circuit.data[run_end:]:
        rebuilt_circuit.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )

    if len(rebuilt_circuit.data) >= len(circuit.data):
        return circuit

    return rebuilt_circuit


def _cancel_adjacent_inverse_blocks(circuit):
    """Remove an adjacent block and its exact inverse."""

    inverse_block = _find_adjacent_inverse_blocks(circuit)
    if inverse_block is None:
        return circuit

    start_index, block_size = inverse_block
    rebuilt_circuit = circuit.copy_empty_like()
    rebuilt_circuit.global_phase = circuit.global_phase

    for instruction in circuit.data[:start_index]:
        rebuilt_circuit.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )

    for instruction in circuit.data[start_index + 2 * block_size :]:
        rebuilt_circuit.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )

    return rebuilt_circuit


def _structural_pre_simplify(circuit):
    """Apply cheap structure-aware simplifications before basis lowering.

    Layer 1 uses commutative inverse cancellation directly on the whole
    circuit. Layer 2 simplifies large repeated exact blocks anywhere in the
    circuit. Layer 3 removes adjacent exact inverse blocks as whole units.
    """

    simplified_circuit = circuit
    for _ in range(3):
        previous_instruction_count = len(simplified_circuit.data)
        simplified_circuit = _canonicalize_diagonal_phase_spans(
            simplified_circuit
        )
        simplified_circuit = _merge_adjacent_parameterized_gates(
            simplified_circuit
        )
        simplified_circuit = _compress_repeated_prefix_boundary_cancellation(
            simplified_circuit
        )
        prefix_simplified_circuit = _simplify_repeated_prefix(
            simplified_circuit
        )
        if len(prefix_simplified_circuit.data) < len(simplified_circuit.data):
            simplified_circuit = prefix_simplified_circuit
        elif len(simplified_circuit.data) <= _MAX_GLOBAL_STRUCTURAL_SCAN_SIZE:
            simplified_circuit = _simplify_repeated_run(simplified_circuit)
            simplified_circuit = _cancel_adjacent_inverse_blocks(
                simplified_circuit
            )
        simplified_circuit = _run_commutative_inverse_cancellation(
            simplified_circuit
        )
        if len(simplified_circuit.data) >= previous_instruction_count:
            break
    return simplified_circuit


def _blockwise_structural_pre_simplify_repeated_prefix(circuit):
    """Pre-simplify a dominant repeated prefix by simplifying one block once."""

    if len(circuit.data) < _MIN_BLOCKWISE_REPEAT_PRESIMPLIFY_SIZE:
        return None

    repeated_prefix = _find_repeated_prefix(circuit)
    if repeated_prefix is None:
        return None

    block_size, repeat_count = repeated_prefix
    coverage = block_size * repeat_count
    if coverage < _MIN_DOMINANT_REPEAT_FRACTION * len(circuit.data):
        return None

    block_circuit = _build_circuit_from_instructions(
        circuit, list(circuit.data[:block_size])
    )
    simplified_block = _structural_pre_simplify(block_circuit)

    rebuilt_circuit = circuit.copy_empty_like()
    # Any global phase the simplification picked up applies once per copy.
    rebuilt_circuit.global_phase = circuit.global_phase + repeat_count * (
        simplified_block.global_phase - block_circuit.global_phase
    )
    for _ in range(repeat_count):
        for instruction in simplified_block.data:
            _append_mapped_instruction(
                rebuilt_circuit, simplified_block, instruction
            )

    _append_instruction_sequence(rebuilt_circuit, circuit.data[coverage:])
    return rebuilt_circuit


def _append_instruction_sequence(circuit, instructions):
    """Append a sequence of instructions to a circuit."""

    for instruction in instructions:
        circuit.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )


def _append_mapped_instruction(
    destination_circuit, source_template, instruction
):
    """Append an instruction while remapping qubits/clbits by index."""

    destination_circuit.append(
        instruction.operation,
        [
            destination_circuit.qubits[source_template.find_bit(qubit).index]
            for qubit in instruction.qubits
        ],
        [
            destination_circuit.clbits[source_template.find_bit(clbit).index]
            for clbit in instruction.clbits
        ],
    )


class PreBasisStructuralSimplification(TransformationPass):
    """Simplify exact inverse, repeated, and diagonal-phase structure.

    Intended to run before basis translation, while composite instructions
    (e.g. ``QFT`` and its inverse, or a repeated custom gate) are still
    intact. The pass only removes or merges operations and never routes, so
    it is valid on both virtual and physical circuits. The input DAG is
    returned unchanged unless the simplified circuit has strictly fewer
    instructions.
    """

    def run(self, dag):
        circuit = _circuit_in_insertion_order(dag)
        simplified = simplify_circuit(circuit)
        if simplified is circuit:
            return dag
        return circuit_to_dag(simplified)


def simplify_circuit(circuit):
    """Apply the structural simplification to a circuit in its given order.

    Repetition detection compares instructions in sequence, so callers that
    hold a ``QuantumCircuit`` should use this rather than a ``PassManager``,
    whose DAG round trip may reorder independent gates. Returns ``circuit``
    itself when nothing is removed.
    """
    simplified = _blockwise_structural_pre_simplify_repeated_prefix(circuit)
    if simplified is None:
        simplified = _structural_pre_simplify(circuit)
    if len(simplified.data) >= len(circuit.data):
        return circuit
    return simplified


def _circuit_in_insertion_order(dag):
    """Convert a DAG to a circuit, breaking topological ties by node id.

    ``dag_to_circuit`` breaks ties by qubit labels, which moves independent
    gates around (e.g. every ``t`` on a spectator qubit to the end) and hides
    repeated blocks. Node ids grow in insertion order, so for a DAG built
    from a circuit this restores that circuit's order exactly.
    """
    circuit = dag_to_circuit(dag)
    ordered = circuit.copy_empty_like()
    for node in dag.topological_op_nodes(
        key=lambda node: f"{node._node_id:012d}"
    ):
        ordered._append(node.op, node.qargs, node.cargs)
    return ordered
