"""Model family profiles for common LLM architectures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class ModelProfile:
    family: str
    aliases: List[str]
    default_dtype: str
    match_op: str
    attention_impl: str
    kv_cache_layout: str
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def matches(self, model_name: str, architecture: str) -> bool:
        hay = " ".join([model_name.lower(), architecture.lower()])
        return self.family.lower() in hay or any(alias.lower() in hay for alias in self.aliases)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_model_profiles() -> List[ModelProfile]:
    return [
        ModelProfile(
            family="llama",
            aliases=["llama2", "llama3", "llama-2", "llama-3"],
            default_dtype="bf16",
            match_op="linalg.batch_matmul",
            attention_impl="grouped-query-attention",
            kv_cache_layout="head-major",
            tags=["attention", "rope", "rmsnorm"],
            notes="Decoder-only dense transformer with RoPE and RMSNorm.",
        ),
        ModelProfile(
            family="deepseek",
            aliases=["deepseek-v2", "deepseek-v3"],
            default_dtype="bf16",
            match_op="linalg.batch_matmul",
            attention_impl="mla-or-moe",
            kv_cache_layout="latent-head-major",
            tags=["moe", "rope", "attention"],
            notes="Supports dense and MoE variants; prefer async DMA for expert dispatch.",
        ),
        ModelProfile(
            family="opt",
            aliases=["facebook/opt", "opt-6.7b"],
            default_dtype="fp16",
            match_op="linalg.batch_matmul",
            attention_impl="multi-head-attention",
            kv_cache_layout="sequence-major",
            tags=["gelu", "layernorm", "attention"],
            notes="OPT-style decoder blocks with LayerNorm and GELU MLP.",
        ),
        ModelProfile(
            family="mixtral",
            aliases=["mixtral-8x7b", "mixtral-8x22b"],
            default_dtype="bf16",
            match_op="linalg.batch_matmul",
            attention_impl="moe-attention",
            kv_cache_layout="head-major",
            tags=["moe", "router", "attention"],
            notes="Mixture-of-experts decoder; route and grouped GEMM are critical.",
        ),
        ModelProfile(
            family="qwen3",
            aliases=["qwen", "qwen2", "qwen2.5", "qwen3"],
            default_dtype="bf16",
            match_op="linalg.batch_matmul",
            attention_impl="grouped-query-attention",
            kv_cache_layout="head-major",
            tags=["attention", "rope", "swiglu"],
            notes="QWEN/QWEN3 family with GQA and SwiGLU-oriented epilogues.",
        ),
    ]


def resolve_model_profile(model_name: str, architecture: str) -> ModelProfile:
    for profile in default_model_profiles():
        if profile.matches(model_name, architecture):
            return profile
    return ModelProfile(
        family="generic",
        aliases=[],
        default_dtype="bf16",
        match_op="linalg.matmul",
        attention_impl="generic-attention",
        kv_cache_layout="generic",
        tags=["attention"],
        notes="Fallback profile.",
    )
