"""Utilities to describe target-aware pass pipelines."""

from __future__ import annotations

from typing import Dict, List

from lib.config_parser.compiler_options import CompilerOptions
from lib.config_parser.hw_caps import HardwareCaps
from lib.config_parser.schema import CompileSpec
from lib.llm.kernels.registry import KernelVariant


def build_llm_pipeline(hw_caps: HardwareCaps, options: CompilerOptions, compile_spec: CompileSpec | None = None, kernel: KernelVariant | None = None) -> Dict[str, List[str]]:
    stages = {
        "graph": ["canonicalize", "cse"],
        "kernel": [],
        "buffer": [],
        "runtime": [],
        "backend": ["lower-affine", "convert-scf-to-cf", "finalize-memref-to-llvm"],
    }

    if options.enable_fusion:
        stages["kernel"].append("llm-fuse-attention-chain")
    if compile_spec and compile_spec.kv_cache:
        stages["kernel"].append("llm-materialize-kv-cache")
    stages["kernel"].append(f"llm-tile[{','.join(str(v) for v in options.tile_sizes)}]")
    if hw_caps.supports_fused_epilogue:
        stages["kernel"].append("llm-fuse-epilogue")
    if kernel and kernel.name != "generic-fallback":
        stages["kernel"].append(f"llm-select-kernel[{kernel.name}]")

    if options.enable_memory_promotion:
        stages["buffer"].append("llm-promote-sram")
    if options.enable_async_dma and hw_caps.dma.supports_async:
        stages["buffer"].append("llm-insert-async-dma")
        stages["runtime"].append("llm-emit-events")
    if hw_caps.supports_kv_cache and compile_spec and compile_spec.kv_cache:
        stages["runtime"].append("llm-allocate-kv-cache")
    stages["runtime"].append("llm-emit-launch-plan")
    return stages
