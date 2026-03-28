"""Compiler option presets for LLM-oriented modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class CompilerOptions:
    mode: str
    tile_sizes: List[int]
    enable_fusion: bool = True
    enable_async_dma: bool = True
    enable_memory_promotion: bool = True
    enable_vectorization: bool = True
    outline_kernels: bool = True
    preferred_match_op: str = "linalg.matmul"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def build_pass_pipeline(self) -> List[str]:
        pipeline = ["canonicalize", "cse"]
        if self.enable_fusion:
            pipeline.append("llm-fuse-producers")
        pipeline.append(f"llm-tile[{','.join(str(v) for v in self.tile_sizes)}]")
        if self.enable_memory_promotion:
            pipeline.append("llm-promote-buffers")
        if self.enable_vectorization:
            pipeline.append("vectorize")
        if self.enable_async_dma:
            pipeline.append("llm-async-dma")
        if self.outline_kernels:
            pipeline.append("llm-outline-kernels")
        pipeline.extend(["canonicalize", "cse"])
        return pipeline


PRESET_OPTIONS = {
    "prefill": CompilerOptions(
        mode="prefill",
        tile_sizes=[128, 128, 64],
        enable_fusion=True,
        enable_async_dma=True,
        preferred_match_op="linalg.batch_matmul",
    ),
    "decode": CompilerOptions(
        mode="decode",
        tile_sizes=[64, 128, 32],
        enable_fusion=True,
        enable_async_dma=True,
        preferred_match_op="linalg.matmul",
    ),
    "train-fwd": CompilerOptions(
        mode="train-fwd",
        tile_sizes=[128, 128, 128],
        enable_fusion=True,
        enable_async_dma=True,
        preferred_match_op="linalg.batch_matmul",
    ),
    "train-bwd": CompilerOptions(
        mode="train-bwd",
        tile_sizes=[128, 64, 128],
        enable_fusion=True,
        enable_async_dma=False,
        preferred_match_op="linalg.generic",
    ),
}

ALIASES = {
    "train_fwd": "train-fwd",
    "train_bwd": "train-bwd",
}


def get_compiler_options(mode: str) -> CompilerOptions:
    normalized = ALIASES.get(mode, mode)
    if normalized not in PRESET_OPTIONS:
        raise ValueError(f"Unsupported compiler mode: {mode}")
    return PRESET_OPTIONS[normalized]
