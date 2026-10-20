from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from qiskit import transpile as qiskit_transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeInverseCancellation

from structured_circuit_benchmarks import build_case
from ucc_methods import ABLATIONS, compile_ablation, same_output_as


THIS_FILE = Path(__file__).resolve()
TARGET_BASIS = ["cx", "rx", "ry", "rz", "h"]
METHODS = (
    "translation_only",
    "qiskit_opt0",
    "qiskit_opt1",
    "qiskit_opt3",
    "qiskit_commutative_inverse",
    "tket_full_peephole",
    "baseline_ucc",
    "optimized_ucc",
    "ucc_local_only",
    "ucc_plus_structural",
)


def circuit_metrics(circuit) -> dict:
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
        "cx_count": int(count_ops.get("cx", 0)),
        "gate_types": sorted(str(name) for name in count_ops.keys()),
    }


def _baseline_or_optimized_ucc(circuit, repo_root: Path) -> dict:
    import ucc

    start = time.perf_counter()
    compiled = ucc.compile(
        circuit,
        return_format="qiskit",
        target_gateset=set(TARGET_BASIS),
    )
    runtime_s = round(time.perf_counter() - start, 3)
    return {
        "status": "ok",
        "output": circuit_metrics(compiled),
        "runtime_s": runtime_s,
    }


def run_worker(
    method: str,
    family: str,
    target_gates: int,
    repo_root: str | None = None,
) -> dict:
    circuit = build_case(family, target_gates).circuit
    input_metrics = circuit_metrics(circuit)

    if method == "translation_only":
        start = time.perf_counter()
        compiled = qiskit_transpile(
            circuit, basis_gates=TARGET_BASIS, optimization_level=0
        )
        runtime_s = round(time.perf_counter() - start, 3)
        return {
            "method": method,
            "family": family,
            "input": input_metrics,
            "status": "ok",
            "output": circuit_metrics(compiled),
            "runtime_s": runtime_s,
        }

    if method in {"qiskit_opt0", "qiskit_opt1", "qiskit_opt3"}:
        optimization_level = int(method[-1])
        start = time.perf_counter()
        compiled = qiskit_transpile(
            circuit,
            basis_gates=TARGET_BASIS,
            optimization_level=optimization_level,
            layout_method="trivial",
            routing_method="none",
        )
        runtime_s = round(time.perf_counter() - start, 3)
        return {
            "method": method,
            "family": family,
            "input": input_metrics,
            "status": "ok",
            "output": circuit_metrics(compiled),
            "runtime_s": runtime_s,
        }

    if method == "qiskit_commutative_inverse":
        start = time.perf_counter()
        simplified = PassManager([CommutativeInverseCancellation()]).run(
            circuit
        )
        compiled = qiskit_transpile(
            simplified, basis_gates=TARGET_BASIS, optimization_level=0
        )
        runtime_s = round(time.perf_counter() - start, 3)
        return {
            "method": method,
            "family": family,
            "input": input_metrics,
            "status": "ok",
            "output": circuit_metrics(compiled),
            "runtime_s": runtime_s,
        }

    if method == "tket_full_peephole":
        try:
            from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
            from pytket.passes import FullPeepholeOptimise
        except Exception as exc:
            return {
                "method": method,
                "family": family,
                "input": input_metrics,
                "status": "unavailable",
                "error": str(exc),
            }

        start = time.perf_counter()
        tk_circuit = qiskit_to_tk(circuit)
        FullPeepholeOptimise().apply(tk_circuit)
        compiled = qiskit_transpile(
            tk_to_qiskit(tk_circuit),
            basis_gates=TARGET_BASIS,
            optimization_level=0,
        )
        runtime_s = round(time.perf_counter() - start, 3)
        return {
            "method": method,
            "family": family,
            "input": input_metrics,
            "status": "ok",
            "output": circuit_metrics(compiled),
            "runtime_s": runtime_s,
        }

    if method == "baseline_ucc":
        result = _baseline_or_optimized_ucc(circuit, Path(repo_root))
    elif method == "optimized_ucc":
        result = _baseline_or_optimized_ucc(circuit, Path(repo_root))
    elif method in ABLATIONS:
        start = time.perf_counter()
        compiled = compile_ablation(circuit, TARGET_BASIS, **ABLATIONS[method])
        result = {
            "status": "ok",
            "output": circuit_metrics(compiled),
            "runtime_s": round(time.perf_counter() - start, 3),
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    return {
        "method": method,
        "family": family,
        "input": input_metrics,
        **result,
    }


def launch_worker(
    python_executable: str,
    repo_root: Path | None,
    method: str,
    family: str,
    target_gates: int,
    timeout_s: int,
) -> dict:
    env = os.environ.copy()
    env.setdefault("XDG_CONFIG_HOME", "/tmp/xdg-config")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    if repo_root is not None:
        env["PYTHONPATH"] = str(repo_root)

    cmd = [
        python_executable,
        str(THIS_FILE),
        "--worker",
        "--method",
        method,
        "--family",
        family,
        "--target-gates",
        str(target_gates),
    ]
    if repo_root is not None:
        cmd.extend(["--repo-root", str(repo_root)])

    try:
        completed = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return {
            "method": method,
            "family": family,
            "status": "timeout",
            "timeout_s": timeout_s,
        }


def _timeout_for_method(method: str) -> int:
    if method == "qiskit_opt3":
        return 120
    if method == "tket_full_peephole":
        return 60
    return 90


def run_parent(
    python_executable: str,
    baseline_repo: Path,
    experimental_repo: Path,
    target_gates: int,
) -> dict:
    families = [
        "qft_inverse",
        "qft_control",
        "qpe_style",
        "qaoa_ring",
        "grover_mirrored",
    ]
    repo_for_method = {
        "baseline_ucc": baseline_repo,
        "optimized_ucc": experimental_repo,
        "ucc_local_only": experimental_repo,
        "ucc_plus_structural": experimental_repo,
    }
    payload: dict[str, dict[str, dict]] = {}
    for family in families:
        payload[family] = {}
        for method in METHODS:
            repo_root = repo_for_method.get(method)
            payload[family][method] = launch_worker(
                python_executable,
                repo_root,
                method,
                family,
                target_gates,
                _timeout_for_method(method),
            )
    return payload


def markdown_summary(payload: dict) -> str:
    lines = [
        "# Fixed-Basis External Baselines",
        "",
        f"Target basis: `{TARGET_BASIS}`",
        "",
    ]
    for family, results in payload.items():
        lines.extend(
            [
                f"## {family}",
                "",
                "| Method | Status | Output Gates | Output Depth | CX Count | Runtime |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            result = results[method]
            status = result["status"]
            label = method
            if method != "qiskit_opt3" and same_output_as(
                result, results.get("qiskit_opt3", {})
            ):
                label += " (= qiskit_opt3)"
            if status == "ok":
                output = result["output"]
                lines.append(
                    f"| {label} | ok | {output['total_gates']:,} | {output['depth']:,} | {output['cx_count']:,} | {result['runtime_s']} s |"
                )
            elif status == "timeout":
                lines.append(
                    f"| {label} | timeout | - | - | - | > {result['timeout_s']} s |"
                )
            else:
                lines.append(f"| {label} | {status} | - | - | - | - |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-basis external baselines for structured circuits."
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--family")
    parser.add_argument("--target-gates", type=int, default=100_000)
    parser.add_argument("--repo-root")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
    )
    parser.add_argument(
        "--baseline-repo",
        type=Path,
        default=Path("/tmp/ucc_662issue_baseline"),
    )
    parser.add_argument(
        "--experimental-repo",
        type=Path,
        default=THIS_FILE.parents[1],
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.method is None or args.family is None:
            raise ValueError("Worker mode requires --method and --family")
        print(
            json.dumps(
                run_worker(
                    args.method,
                    args.family,
                    args.target_gates,
                    args.repo_root,
                )
            )
        )
        return

    payload = run_parent(
        args.python_executable,
        args.baseline_repo,
        args.experimental_repo,
        args.target_gates,
    )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.md_out is not None:
        args.md_out.write_text(markdown_summary(payload))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
