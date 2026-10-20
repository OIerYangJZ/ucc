"""Shared UCC ablation methods for the fixed-basis comparison scripts.

``ucc.compile`` without a backend runs ``UCCDefault1(structural_prepass=True)``.
The ablations take that exact pass list and drop components, to attribute
its gains:

* ``ucc_local_only``: translation and UCC's structure-aware local passes,
  with neither structure-aware pass;
* ``ucc_plus_structural``: adds pre-basis structural simplification;
* full pipeline (``optimized_ucc``): also phase-polynomial synthesis.

None of them calls Qiskit's optimization_level=3 preset.
"""

from __future__ import annotations

from qiskit import transpile as qiskit_transpile
from qiskit.transpiler import PassManager

ABLATIONS = {
    "ucc_local_only": {"structural": False, "phase_polynomial": False},
    "ucc_plus_structural": {"structural": True, "phase_polynomial": False},
}


def compile_ablation(circuit, target_basis, structural, phase_polynomial):
    """UCC structure-aware pipeline with selected components removed."""
    from ucc import UCCDefault1
    from ucc.transpilers.phase_polynomial import PhasePolynomialSynthesis
    from ucc.transpilers.structural_simplification import (
        PreBasisStructuralSimplification,
    )

    removed = []
    if not structural:
        removed.append(PreBasisStructuralSimplification)
    if not phase_polynomial:
        removed.append(PhasePolynomialSynthesis)
    tasks = (
        UCCDefault1(target_gateset=set(target_basis), structural_prepass=True)
        .pass_manager.to_flow_controller()
        .tasks
    )
    kept = [task for task in tasks if not isinstance(task, tuple(removed))]
    compiled = PassManager(kept).run(circuit)
    return qiskit_transpile(
        compiled, basis_gates=list(target_basis), optimization_level=0
    )


def same_output_as(result, reference):
    """True when two ok results have identical gate, depth and CX counts."""
    if result.get("status") != "ok" or reference.get("status") != "ok":
        return False
    keys = ("total_gates", "depth", "cx_count")
    return all(
        result["output"][key] == reference["output"][key] for key in keys
    )
