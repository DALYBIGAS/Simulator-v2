#!/usr/bin/env python3
"""Compiler entrypoint for LLM-oriented AI-chip flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from lib.config_parser import load_compilation_context
from lib.driver_gen.driver_gen import build_stub
from lib.llm.kernels.registry import choose_kernel_variant, default_kernel_registry
from lib.llm.models.profiles import default_model_profiles, resolve_model_profile
from lib.llm.passes.pipeline import build_llm_pipeline
from lib.llm.runtime.plan import build_runtime_plan
from lib.llm.profiler.estimator import estimate_metrics
from lib.transform_gen.common import build_fuse_script, build_outline_script, build_tile_script


def _tensor_names(spec_items) -> List[str]:
    return [item.name for item in spec_items]


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile planning entrypoint for LLM AI-chip workflows.")
    parser.add_argument("--hardware", required=True, help="YAML file describing hardware capabilities.")
    parser.add_argument("--compile-spec", required=True, help="YAML file describing the model/kernel compile request.")
    parser.add_argument("--payload-mlir", required=True, help="Input MLIR payload file.")
    parser.add_argument("--out-dir", required=True, help="Directory where compiler outputs are written.")
    args = parser.parse_args()

    hw_caps, options, compile_spec = load_compilation_context(args.hardware, args.compile_spec)
    # profile = resolve_model_profile(compile_spec.model_name, compile_spec.architecture or compile_spec.model_name)
    profile = resolve_model_profile(
        compile_spec.model_name,
        getattr(compile_spec, "architecture", "") or compile_spec.model_name,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_path = Path(args.payload_mlir)
    all_tags = list(dict.fromkeys(list(compile_spec.tags) + list(profile.tags) + [profile.family]))
    match_op = compile_spec.match_op or profile.match_op or options.preferred_match_op
    kernel_variant = choose_kernel_variant(
        match_op,
        compile_spec.mode,
        compile_spec.dtype or profile.default_dtype,
        compile_spec.kv_cache,
        all_tags,
    )

    tile_sizes = kernel_variant.preferred_tile_sizes or options.tile_sizes

    tile_script_path = out_dir / f"{compile_spec.kernel_name}.tile.mlir"
    build_tile_script(match_op, tile_sizes).write(tile_script_path)

    fuse_script_path = None
    if compile_spec.op_chain:
        fuse_script_path = out_dir / f"{compile_spec.kernel_name}.fuse.mlir"
        build_fuse_script(compile_spec.op_chain, compile_spec.kernel_name, tile_sizes).write(fuse_script_path)

    outline_target = compile_spec.outline_function or f"{compile_spec.kernel_name}_{compile_spec.mode}"
    outline_script_path = out_dir / f"{compile_spec.kernel_name}.outline.mlir"
    build_outline_script(match_op, outline_target).write(outline_script_path)

    stub_path = out_dir / f"{compile_spec.kernel_name}_driver.c"
    stub_path.write_text(
        build_stub(
            compile_spec.kernel_name,
            compile_spec.mode,
            compile_spec.dtype,
            _tensor_names(compile_spec.inputs),
            _tensor_names(compile_spec.outputs),
        ),
        encoding="utf-8",
    )

    pipeline = build_llm_pipeline(hw_caps, options, compile_spec=compile_spec, kernel=kernel_variant)
    runtime_plan = build_runtime_plan(hw_caps, options, compile_spec)
    estimated_metrics = estimate_metrics(hw_caps, compile_spec, profile.family)

    manifest = {
        "model_name": compile_spec.model_name,
        "kernel_name": compile_spec.kernel_name,
        "mode": compile_spec.mode,
        "payload_mlir": str(payload_path),
        "hardware": hw_caps.to_dict(),
        "compile_spec": compile_spec.to_dict(),
        "model_profile": profile.to_dict(),
        "compiler_options": options.to_dict(),
        "selected_kernel": kernel_variant.to_dict(),
        "pass_pipeline": pipeline,
        "runtime_plan": runtime_plan.to_dict(),
        "estimated_metrics": estimated_metrics.to_dict(),
        "artifacts": {
            "tile_transform": str(tile_script_path),
            "fuse_transform": str(fuse_script_path) if fuse_script_path else None,
            "outline_transform": str(outline_script_path),
            "driver_stub": str(stub_path),
        },
    }
    _write_json(out_dir / "compile_manifest.json", manifest)
    _write_json(out_dir / "kernel_registry.json", {"kernels": [item.to_dict() for item in default_kernel_registry()]})
    _write_json(out_dir / "model_profiles.json", {"profiles": [item.to_dict() for item in default_model_profiles()]})
    _write_json(out_dir / "pass_pipeline.json", pipeline)
    _write_json(out_dir / "runtime_plan.json", runtime_plan.to_dict())
    _write_json(out_dir / "estimated_metrics.json", estimated_metrics.to_dict())

    summary = "\n".join(
        [
            "# Compilation Summary",
            "",
            f"- model: {compile_spec.model_name}",
            f"- model_family: {profile.family}",
            f"- kernel: {compile_spec.kernel_name}",
            f"- mode: {compile_spec.mode}",
            f"- chip: {hw_caps.name}",
            f"- array: {hw_caps.array_shape}",
            f"- dtype: {compile_spec.dtype}",
            f"- selected_kernel: {kernel_variant.name}",
            f"- tile_sizes: {tile_sizes}",
            f"- kv_cache: {compile_spec.kv_cache}",
            f"- payload: {payload_path}",
            f"- est_latency_ms: {estimated_metrics.estimated_latency_ms:.4f}",
            f"- est_tokens_per_sec: {estimated_metrics.estimated_tokens_per_sec:.2f}",
        ]
    ) + "\n"
    _write_text(out_dir / "compilation_summary.md", summary)

    plan_text = "\n".join(
        [
            "Compiler output bundle:",
            f"  manifest: {out_dir / 'compile_manifest.json'}",
            f"  estimated metrics: {out_dir / 'estimated_metrics.json'}",
            f"  runtime plan: {out_dir / 'runtime_plan.json'}",
            f"  pass pipeline: {out_dir / 'pass_pipeline.json'}",
            f"  tile transform: {tile_script_path}",
            f"  outline transform: {outline_script_path}",
            f"  driver stub: {stub_path}",
        ]
    ) + "\n"
    _write_text(out_dir / "README.txt", plan_text)
    print(out_dir)


if __name__ == "__main__":
    main()
