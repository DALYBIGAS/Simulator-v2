"""Kernel registry and selection helpers for LLM workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List


@dataclass
class KernelVariant:
    name: str
    op_type: str
    modes: List[str]
    supported_dtypes: List[str]
    tags: List[str] = field(default_factory=list)
    preferred_tile_sizes: List[int] = field(default_factory=list)
    uses_kv_cache: bool = False
    notes: str = ""

    def score(self, *, mode: str, dtype: str, kv_cache: bool, tags: Iterable[str]) -> int:
        score = 0
        tag_set = set(tags)
        if mode in self.modes:
            score += 10
        if dtype in self.supported_dtypes:
            score += 8
        if kv_cache and self.uses_kv_cache:
            score += 6
        score += len(tag_set.intersection(self.tags))
        return score

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_kernel_registry() -> List[KernelVariant]:
    return [
        KernelVariant(
            name="qkv-projection",
            op_type="linalg.batch_matmul",
            modes=["prefill", "train-fwd"],
            supported_dtypes=["bf16", "fp16", "f32"],
            tags=["attention", "projection"],
            preferred_tile_sizes=[128, 128, 64],
            notes="Dense QKV projection path optimized for long-sequence prefill.",
        ),
        KernelVariant(
            name="decode-attention",
            op_type="linalg.matmul",
            modes=["decode"],
            supported_dtypes=["bf16", "fp16", "int8"],
            tags=["attention", "decode"],
            preferred_tile_sizes=[64, 128, 32],
            uses_kv_cache=True,
            notes="Decode path tuned for short query and long KV cache access.",
        ),
        KernelVariant(
            name="rmsnorm-epilogue",
            op_type="linalg.generic",
            modes=["prefill", "decode", "train-fwd"],
            supported_dtypes=["bf16", "fp16", "f32"],
            tags=["norm", "epilogue"],
            preferred_tile_sizes=[1, 256, 1],
            notes="Fusable epilogue path for RMSNorm and residual add.",
        ),
        KernelVariant(
            name="matmul-backward",
            op_type="linalg.generic",
            modes=["train-bwd"],
            supported_dtypes=["bf16", "fp16", "f32"],
            tags=["backward", "matmul"],
            preferred_tile_sizes=[128, 64, 128],
            notes="Gradient-oriented matmul path.",
        ),
    ]


def choose_kernel_variant(op_type: str, mode: str, dtype: str, kv_cache: bool, tags: Iterable[str]) -> KernelVariant:
    candidates = [item for item in default_kernel_registry() if item.op_type == op_type]
    if not candidates:
        return KernelVariant(
            name="generic-fallback",
            op_type=op_type,
            modes=[mode],
            supported_dtypes=[dtype],
            preferred_tile_sizes=[],
            notes="Fallback kernel selection.",
        )
    return max(candidates, key=lambda item: item.score(mode=mode, dtype=dtype, kv_cache=kv_cache, tags=tags))
