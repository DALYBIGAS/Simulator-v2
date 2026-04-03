#!/usr/bin/env python3
"""Apply a planned pipeline to MLIR and emit backend/runtime artifacts.

The script prefers external ``mlir-opt`` when it can faithfully execute the
planned pipeline. Otherwise it uses the in-repository LLM pass engine, which
parses a constrained MLIR subset and runs real graph-rewrite passes for decode
attention / MoE kernels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from lib.llm.passes.executor import ModuleIR, execute_pass_pipeline
from lib.llm.runtime.codegen import generate_runtime_launch_code


def _flatten_pipeline(pipeline: Dict[str, List[str]]) -> List[str]:
    ordered: List[str] = []
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


def _backend_ir(module: ModuleIR, manifest: Dict[str, Any], runtime_plan: Dict[str, Any]) -> str:
    compile_spec = manifest.get("compile_spec", {})
    metadata = runtime_plan.get("metadata", {})
    declarations: List[str] = []
    body: List[str] = []
    seen_decls = set()

    intrinsic_map = {
        "decode_attention": "void @legend_decode_attention(i8*, i8*, i8*)",
        "prefill_attention": "void @legend_prefill_attention(i8*)",
        "qkv_projection": "void @legend_qkv_projection(i8*)",
        "moe_decode": "void @legend_moe_decode(i8*)",
        "identity": "void @legend_identity(i8*)",
        "async_dma": "void @legend_async_dma(i8*)",
        "event_record": "void @legend_event_record(i8*)",
        "launch": "void @legend_launch(i8*)",
        "kv_cache_alloc": "void @legend_kv_cache_alloc(i8*)",
    }

    for fn in module.functions:
        for op in fn.ops:
            if op.name != "llvm.call":
                continue
            callee = op.attrs.get("callee", '"legend_unknown"').strip('"')
            short = callee.removeprefix("legend_")
            decl = intrinsic_map.get(short, f"void @{callee}(i8*)")
            if decl not in seen_decls:
                declarations.append(f"declare {decl}")
                seen_decls.add(decl)
            arg_comment = ", ".join(op.operands) if op.operands else "none"
            body.append(f"  ; {short} operands: {arg_comment}")
            if short == "decode_attention":
                body.append(f"  call void @{callee}(i8* null, i8* null, i8* null)")
            else:
                body.append(f"  call void @{callee}(i8* null)")

    lines = [
        "; generated backend output from in-repo pass engine",
        f"; model = {manifest.get('model_name')}",
        f"; kernel = {manifest.get('kernel_name')}",
        f"; mode = {manifest.get('mode')}",
        f"; dtype = {compile_spec.get('dtype', 'unknown')}",
        f"; is_moe = {int(bool(metadata.get('is_moe', False)))}",
        f"; kv_cache = {int(bool(metadata.get('kv_cache', False)))}",
        "",
    ]
    lines.extend(declarations)
    if declarations:
        lines.append("")
    lines.append("define void @compiled_entry() {")
    if body:
        lines.extend(body)
    else:
        lines.append("  ret void")
    if not body or body[-1] != "  ret void":
        lines.append("  ret void")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def apply_pipeline(manifest_path: Path, out_dir: Path, mlir_opt: str | None = None) -> Dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = manifest["pass_pipeline"]
    runtime_plan = manifest["runtime_plan"]
    payload_mlir_path = Path(manifest["payload_mlir"])

    out_dir.mkdir(parents=True, exist_ok=True)
    input_mlir = payload_mlir_path.read_text(encoding="utf-8")
    flattened = _flatten_pipeline(pipeline)

    use_mlir_opt = False
    exe = mlir_opt or shutil.which("mlir-opt")
    real_flags: List[str] = []
    if exe:
        safe = [_safe_mlir_flag(item) for item in flattened]
        if all(flag is not None for flag in safe):
            use_mlir_opt = True
            real_flags = [flag for flag in safe if flag is not None]

    optimized_path = out_dir / "optimized.mlir"
    backend_path = out_dir / "backend.ll"
    runtime_c_path = out_dir / "runtime_launch.c"
    metrics_path = out_dir / "backend_metrics.json"

    engine_stats: Dict[str, Any] = {}
    if use_mlir_opt:
        command = [exe, str(payload_mlir_path)] + real_flags
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            optimized_text = result.stdout
            from lib.llm.passes.executor import parse_mlir_module
            parsed_module = parse_mlir_module(optimized_text)
        else:
            optimized_text, parsed_module, engine_stats = execute_pass_pipeline(input_mlir, pipeline, manifest)
            use_mlir_opt = False
    else:
        optimized_text, parsed_module, engine_stats = execute_pass_pipeline(input_mlir, pipeline, manifest)

    optimized_path.write_text(optimized_text, encoding="utf-8")
    backend_path.write_text(_backend_ir(parsed_module, manifest, runtime_plan), encoding="utf-8")
    runtime_c_path.write_text(generate_runtime_launch_code(runtime_plan), encoding="utf-8")

    counts: Dict[str, int] = {}
    for fn in parsed_module.functions:
        for op in fn.ops:
            counts[op.name] = counts.get(op.name, 0) + 1

    metrics = {
        "num_pipeline_passes": len(flattened),
        "num_runtime_buffers": len(runtime_plan.get("buffers", [])),
        "num_runtime_launches": len(runtime_plan.get("launches", [])),
        "num_runtime_events": len(runtime_plan.get("events", [])),
        "used_real_mlir_opt": use_mlir_opt,
        "used_inrepo_pass_engine": not use_mlir_opt,
        "source_manifest": str(manifest_path),
        "pipeline_stages": {stage: len(passes) for stage, passes in pipeline.items()},
        "optimized_op_counts": counts,
        **engine_stats,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return {
        "optimized_mlir": str(optimized_path),
        "backend_ir": str(backend_path),
        "runtime_launch_c": str(runtime_c_path),
        "backend_metrics": str(metrics_path),
        "used_real_mlir_opt": use_mlir_opt,
        "used_inrepo_pass_engine": not use_mlir_opt,
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
