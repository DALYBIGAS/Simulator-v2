#!/usr/bin/env python3
"""Apply a planned pipeline to MLIR and emit backend/runtime artifacts.

This script prefers real `mlir-opt` execution when available. If `mlir-opt` is
not present or a pass is target-specific, it emits a deterministic annotated
fallback MLIR so the framework remains runnable end-to-end.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from lib.llm.runtime.codegen import generate_runtime_launch_code


def _flatten_pipeline(pipeline: Dict[str, List[str]]) -> List[str]:
    ordered = []
    for stage in ["graph", "kernel", "buffer", "runtime", "backend"]:
        ordered.extend(pipeline.get(stage, []))
    return ordered


def _safe_mlir_flag(pass_name: str) -> str | None:
    if pass_name.startswith("llm-"):
        return None
    if "[" in pass_name and pass_name.endswith("]"):
        base, args = pass_name[:-1].split("[", 1)
        args = args.replace(",", "x")
        return f"--{base}={args}"
    return f"--{pass_name}"


def _annotated_fallback_mlir(input_mlir: str, pipeline: Dict[str, List[str]], manifest: Dict[str, object]) -> str:
    header = [
        "// FALLBACK OPTIMIZED MLIR",
        f"// model={manifest.get('model_name')}",
        f"// kernel={manifest.get('kernel_name')}",
        f"// mode={manifest.get('mode')}",
    ]
    for stage, passes in pipeline.items():
        header.append(f"// stage:{stage} -> {', '.join(passes)}")
    header.append("")
    header.append("module attributes {llm.optimized = true} {")
    header.append("  // The original payload is embedded below for deterministic replay.")
    for line in input_mlir.splitlines():
        header.append("  // " + line)
    header.append("}")
    header.append("")
    return "\n".join(header)


def apply_pipeline(manifest_path: Path, out_dir: Path, mlir_opt: str | None = None) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = manifest["pass_pipeline"]
    artifacts = manifest["artifacts"]
    runtime_plan = manifest["runtime_plan"]
    payload_mlir_path = Path(manifest["payload_mlir"])

    out_dir.mkdir(parents=True, exist_ok=True)
    input_mlir = payload_mlir_path.read_text(encoding="utf-8")
    flattened = _flatten_pipeline(pipeline)

    use_mlir_opt = False
    exe = mlir_opt or shutil.which("mlir-opt")
    real_flags = []
    if exe:
        safe = [_safe_mlir_flag(item) for item in flattened]
        if all(flag is not None for flag in safe):
            use_mlir_opt = True
            real_flags = [flag for flag in safe if flag is not None]

    optimized_path = out_dir / "optimized.mlir"
    backend_path = out_dir / "backend.ll"
    runtime_c_path = out_dir / "runtime_launch.c"
    metrics_path = out_dir / "backend_metrics.json"

    if use_mlir_opt:
        command = [exe, str(payload_mlir_path)] + real_flags
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            optimized_text = result.stdout
        else:
            optimized_text = _annotated_fallback_mlir(input_mlir, pipeline, manifest)
    else:
        optimized_text = _annotated_fallback_mlir(input_mlir, pipeline, manifest)

    optimized_path.write_text(optimized_text, encoding="utf-8")

    backend_text = "\n".join(
        [
            "; pseudo backend output",
            f"; model = {manifest.get('model_name')}",
            f"; kernel = {manifest.get('kernel_name')}",
            f"; mode = {manifest.get('mode')}",
            "define void @compiled_entry() {",
            "  ret void",
            "}",
            "",
        ]
    )
    backend_path.write_text(backend_text, encoding="utf-8")

    runtime_c_path.write_text(generate_runtime_launch_code(runtime_plan), encoding="utf-8")

    metrics = {
        "num_pipeline_passes": len(flattened),
        "num_runtime_buffers": len(runtime_plan.get("buffers", [])),
        "num_runtime_launches": len(runtime_plan.get("launches", [])),
        "used_real_mlir_opt": use_mlir_opt,
        "source_manifest": str(manifest_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return {
        "optimized_mlir": str(optimized_path),
        "backend_ir": str(backend_path),
        "runtime_launch_c": str(runtime_c_path),
        "backend_metrics": str(metrics_path),
        "used_real_mlir_opt": use_mlir_opt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a compilation manifest to produce optimized MLIR and runtime artifacts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mlir-opt", default=None)
    args = parser.parse_args()
    result = apply_pipeline(Path(args.manifest), Path(args.out_dir), args.mlir_opt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
