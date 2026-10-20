"""Compile a dominant repeated run once and reuse it on a hardware target.

Circuits such as ``k`` Grover iterations are mostly one composite block
repeated many times. Mapping the whole expanded circuit onto hardware can
take minutes; mapping one copy of the block and repeating it takes seconds.

Separately mapped pieces only compose correctly if they agree on where each
virtual qubit sits at every seam. So every piece (prefix, block, suffix) is
mapped from one shared initial layout, and any routing permutation it leaves
behind is undone with SWAPs along the coupling graph (token swapping). Each
piece then starts and ends in the shared layout, and the pieces can be
concatenated.
"""

import dataclasses

from qiskit import QuantumCircuit
from qiskit import transpile as qiskit_transpile
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper

from .structural_simplification import (
    _build_circuit_from_instructions,
    _find_repeated_run,
)
from .ucc_defaults import backend_pass_manager

_MIN_CIRCUIT_SIZE = 512
_MIN_REPEATS = 8
_MAX_BLOCK_SIZE = 4
_MIN_COVERAGE = 0.8


def find_dominant_repeated_run(circuit):
    """Return ``(start, block_size, repeats)`` for a run worth reusing.

    The run must cover most of the circuit and contain a composite
    instruction; runs of primitive gates are cheap to map as a whole.
    """
    if len(circuit.data) < _MIN_CIRCUIT_SIZE:
        return None
    run = _find_repeated_run(
        circuit, max_block_size=_MAX_BLOCK_SIZE, min_repeats=_MIN_REPEATS
    )
    if run is None:
        return None
    start, block_size, repeats = run
    if block_size * repeats < _MIN_COVERAGE * len(circuit.data):
        return None
    block = circuit.data[start : start + block_size]
    if not any(
        instruction.operation.definition is not None
        and len(instruction.operation.definition.data) > 1
        for instruction in block
    ):
        return None
    return run


def compile_repeated_run_on_backend(circuit, backend, seed_transpiler=None):
    """Map a dominant repeated run once and reuse it; None if not applicable.

    The result carries a ``TranspileLayout`` with the shared initial layout
    and no final permutation.
    """
    run = find_dominant_repeated_run(circuit)
    coupling_map = getattr(backend, "coupling_map", None)
    if run is None or coupling_map is None:
        return None
    start, block_size, repeats = run
    run_end = start + block_size * repeats

    def piece(instructions):
        # The source global phase is added once, when concatenating.
        part = _build_circuit_from_instructions(circuit, list(instructions))
        part.global_phase = 0
        return part

    prefix = list(circuit.data[:start])
    block = list(circuit.data[start : start + block_size])
    suffix = list(circuit.data[run_end:])

    swapper = ApproximateTokenSwapper(
        coupling_map.graph.to_undirected(multigraph=False),
        seed=seed_transpiler,
    )
    layouts = _candidate_layouts(
        backend, seed_transpiler, piece(prefix + block + suffix), piece(block)
    )

    pieces = [piece(prefix), piece(block), piece(suffix)]
    best = None
    for layout in layouts:
        mapped = _map_pieces(pieces, layout, swapper, backend, seed_transpiler)
        if mapped is None:
            continue
        cost = _repeated_cost(mapped, repeats)
        if best is None or cost < best[0]:
            best = (cost, mapped)
    if best is None:
        return None

    prefix_mapped, block_mapped, suffix_mapped = best[1]
    result = prefix_mapped.copy_empty_like()
    result.global_phase = circuit.global_phase
    result.compose(prefix_mapped, inplace=True)
    for _ in range(repeats):
        result.compose(block_mapped, inplace=True)
    result.compose(suffix_mapped, inplace=True)
    result._layout = dataclasses.replace(
        prefix_mapped.layout,
        final_layout=None,
        _output_qubit_list=result.qubits,
    )
    return result


def _map_pieces(pieces, layout, swapper, backend, seed_transpiler):
    """Map each piece from ``layout`` and return it there; None on failure."""
    pass_manager = backend_pass_manager(
        backend,
        seed_transpiler=seed_transpiler,
        structural_prepass=True,
        initial_layout=layout,
    )
    mapped = []
    for part in pieces:
        part = _return_to_layout(
            pass_manager.run(part), layout, swapper, backend, seed_transpiler
        )
        if part is None:
            return None
        mapped.append(part)
    return mapped


def _candidate_layouts(backend, seed_transpiler, representative, block):
    """Shared layouts to try: those Qiskit picks for prefix + one block +
    suffix and for the block alone, and the trivial layout."""
    pass_manager = backend_pass_manager(
        backend, seed_transpiler=seed_transpiler, structural_prepass=True
    )
    layouts = []
    for circuit in (representative, block):
        mapped = pass_manager.run(circuit)
        if mapped.layout is not None:
            layouts.append(
                mapped.layout.initial_index_layout(filter_ancillas=True)
            )
    layouts.append(list(range(representative.num_qubits)))
    unique = []
    for layout in layouts:
        if layout not in unique:
            unique.append(layout)
    return unique


def _repeated_cost(mapped, repeats):
    """(two-qubit gates, all gates) of prefix + repeats * block + suffix."""
    prefix, block, suffix = mapped

    def counts(circuit):
        two_qubit = sum(
            1 for instruction in circuit.data if len(instruction.qubits) == 2
        )
        return two_qubit, len(circuit.data)

    (p2, p), (b2, b), (s2, s) = counts(prefix), counts(block), counts(suffix)
    return p2 + repeats * b2 + s2, p + repeats * b + s


def _return_to_layout(mapped, layout, swapper, backend, seed_transpiler):
    """Append SWAPs that move every virtual qubit back to ``layout``."""
    if mapped.layout is None:
        return None
    if mapped.layout.initial_index_layout(filter_ancillas=True) != layout:
        return None
    final = mapped.layout.final_index_layout(filter_ancillas=True)
    if final == layout:
        return mapped

    # Virtual qubits already in place are fixed points; ancilla positions
    # (all still |0>, only ever swapped) may move freely.
    swaps = swapper.map(
        {position: layout[qubit] for qubit, position in enumerate(final)}
    )
    positions = list(final)
    restore = QuantumCircuit(mapped.num_qubits)
    for first, second in swaps:
        restore.swap(first, second)
        positions = [
            second if p == first else first if p == second else p
            for p in positions
        ]
    if positions != layout:
        return None

    restore = qiskit_transpile(
        restore,
        backend=backend,
        optimization_level=0,
        initial_layout=list(range(mapped.num_qubits)),
        routing_method="none",
        seed_transpiler=seed_transpiler,
    )
    returned = mapped.copy()
    returned.compose(restore, inplace=True)
    return returned
