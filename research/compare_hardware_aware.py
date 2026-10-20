from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from qiskit import transpile as qiskit_transpile

from hardware_aware_backend import LineBackend
from hardware_aware_instances import build_case


THIS_FILE = Path(__file__).resolve()
METHODS = ("qiskit_opt3", "baseline_ucc", "optimized_ucc")
FAMILIES = (
    "hw_mqt_qpeexact_20",
    "hw_mqt_qaoa_20",
    "hw_mqt_grover_20",
)
BACKEND = LineBackend(20, name="line20")
DEFAULT_SEED = 12345


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


def _compile_with_ucc(circuit, seed_transpiler: int | None = None) -> dict:
    import ucc

    start = time.perf_counter()
    try:
        compiled = ucc.compile(
            circuit,
            return_format="qiskit",
            target_backend=BACKEND,
            seed_transpiler=seed_transpiler,
        )
    except TypeError:
        compiled = ucc.compile(
            circuit,
            return_format="qiskit",
            target_backend=BACKEND,
        )
    runtime_s = round(time.perf_counter() - start, 3)
    return {
        "status": "ok",
        "output": circuit_metrics(compiled),
        "runtime_s": runtime_s,
    }


def run_worker(
    method: str, family: str, seed_transpiler: int | None = None
) -> dict:
    circuit = build_case(family).circuit
    input_metrics = circuit_metrics(circuit)

    if method == "qiskit_opt3":
        start = time.perf_counter()
        compiled = qiskit_transpile(
            circuit,
            backend=BACKEND,
            optimization_level=3,
            seed_transpiler=seed_transpiler,
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

    if method in {"baseline_ucc", "optimized_ucc"}:
        result = _compile_with_ucc(circuit, seed_transpiler=seed_transpiler)
        return {
            "method": method,
            "family": family,
            "input": input_metrics,
            **result,
        }

    raise ValueError(f"Unsupported method: {method}")


def launch_worker(
    python_executable: str,
    repo_root: Path | None,
    method: str,
    family: str,
    timeout_s: int,
    seed_transpiler: int | None = None,
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
    ]
    if seed_transpiler is not None:
        cmd.extend(["--seed-transpiler", str(seed_transpiler)])
    if repo_root is not None:
        cmd.extend(["--repo-root", str(repo_root)])

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(timeout=timeout_s)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                cmd,
                output=stdout,
                stderr=_stderr,
            )
        return json.loads(stdout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return {
            "method": method,
            "family": family,
            "status": "timeout",
            "timeout_s": timeout_s,
        }


def _timeout_for_method(method: str, family: str) -> int:
    if family == "hw_mqt_grover_20":
        return 240
    return 120


def run_parent(
    python_executable: str,
    baseline_repo: Path,
    experimental_repo: Path,
    seed_transpiler: int | None = None,
) -> dict:
    repo_for_method = {
        "baseline_ucc": baseline_repo,
        "optimized_ucc": experimental_repo,
    }
    payload: dict[str, dict[str, dict]] = {}
    for family in FAMILIES:
        payload[family] = {}
        for method in METHODS:
            # Every method runs in its own worker so the timeout applies
            # and ucc is imported from the intended repository.
            payload[family][method] = launch_worker(
                python_executable,
                repo_for_method.get(method),
                method,
                family,
                _timeout_for_method(method, family),
                seed_transpiler=seed_transpiler,
            )
    return payload


def markdown_summary(payload: dict) -> str:
    lines = [
        "# Hardware-Aware Comparison",
        "",
        "Target backend: 20-qubit bidirectional line backend",
        "",
        f"Seed transpiler: `{DEFAULT_SEED}`",
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
            if status == "ok":
                output = result["output"]
                lines.append(
                    f"| {method} | ok | {output['total_gates']:,} | {output['depth']:,} | {output['cx_count']:,} | {result['runtime_s']} s |"
                )
            elif status == "timeout":
                lines.append(
                    f"| {method} | timeout | - | - | - | > {result['timeout_s']} s |"
                )
            else:
                lines.append(f"| {method} | {status} | - | - | - | - |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare backend-aware baselines for structured circuits."
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--family")
    parser.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED)
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
                    seed_transpiler=args.seed_transpiler,
                )
            )
        )
        return

    payload = run_parent(
        args.python_executable,
        args.baseline_repo,
        args.experimental_repo,
        seed_transpiler=args.seed_transpiler,
    )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.md_out is not None:
        args.md_out.write_text(markdown_summary(payload))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
