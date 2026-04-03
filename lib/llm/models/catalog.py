"""Unified model-family presets for LLM-oriented compiler planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelProfile:
    family: str
    aliases: List[str] = field(default_factory=list)
    attention_type: str = "mha"
    kv_layout: str = "paged"
    default_dtype: str = "bf16"
    preferred_match_op: str = "linalg.matmul"
    decode_tile_sizes: List[int] = field(default_factory=lambda: [64, 128, 32])
    prefill_tile_sizes: List[int] = field(default_factory=lambda: [128, 128, 64])
    extra_tags: List[str] = field(default_factory=list)
    grouped_query_attention: bool = False
    mixture_of_experts: bool = False
    rotary_embedding: bool = True
    supports_prefix_cache: bool = True
    notes: str = ""

    def matches(self, *candidates: str) -> bool:
        hay = " ".join(item.strip().lower() for item in candidates if item)
        if not hay:
            return False
        return self.family.lower() in hay or any(alias.lower() in hay for alias in self.aliases)

    def recommended_tile_sizes(self, mode: str) -> List[int]:
        return list(self.decode_tile_sizes if mode == "decode" else self.prefill_tile_sizes)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_MODEL_PROFILES = [
    ModelProfile(family="llama", aliases=["llama2", "llama3", "meta-llama", "llama-2", "llama-3"], attention_type="gqa", kv_layout="paged", preferred_match_op="linalg.batch_matmul", extra_tags=["llama", "attention", "rope", "rmsnorm"], grouped_query_attention=True, notes="LLaMA-family preset tuned for RoPE and grouped-query attention decode flows."),
    ModelProfile(family="deepseek", aliases=["deepseek-v2", "deepseek-v3"], attention_type="mla", kv_layout="compressed", preferred_match_op="linalg.batch_matmul", decode_tile_sizes=[64, 128, 64], prefill_tile_sizes=[128, 128, 128], extra_tags=["deepseek", "attention", "mla", "moe"], grouped_query_attention=True, mixture_of_experts=True, notes="DeepSeek-family preset biased toward MLA-style attention and MoE routing."),
    ModelProfile(family="opt", aliases=["facebook-opt", "opt-6.7b"], attention_type="mha", kv_layout="contiguous", preferred_match_op="linalg.batch_matmul", default_dtype="fp16", extra_tags=["opt", "attention", "learned-pos", "gelu", "layernorm"], rotary_embedding=False, supports_prefix_cache=False, notes="OPT-family preset for classic MHA decoder blocks with contiguous KV layout."),
    ModelProfile(family="mixtral", aliases=["mistral-moe", "mixtral-8x7b", "mixtral-8x22b"], attention_type="gqa", kv_layout="paged", preferred_match_op="linalg.batch_matmul", decode_tile_sizes=[64, 128, 64], prefill_tile_sizes=[128, 128, 128], extra_tags=["mixtral", "attention", "moe", "router"], grouped_query_attention=True, mixture_of_experts=True, notes="Mixtral-family preset with GQA attention and expert-routing aware passes."),
    ModelProfile(family="qwen3", aliases=["qwen", "qwen2", "qwen2.5", "qwen3-moe", "qwen3"], attention_type="gqa", kv_layout="paged", preferred_match_op="linalg.batch_matmul", extra_tags=["qwen3", "attention", "rope", "swiglu"], grouped_query_attention=True, notes="Qwen-family preset for paged KV cache decode and RoPE-heavy attention fusion."),
]


def default_model_profiles() -> List[ModelProfile]:
    return list(_MODEL_PROFILES)


def supported_model_families() -> List[str]:
    return [profile.family for profile in _MODEL_PROFILES]


def resolve_model_profile(model_name: str = "", architecture: str = "") -> ModelProfile:
    for profile in _MODEL_PROFILES:
        if profile.matches(model_name, architecture):
            return profile
    return ModelProfile(
        family="generic",
        aliases=[],
        attention_type="mha",
        kv_layout="generic",
        default_dtype="bf16",
        preferred_match_op="linalg.matmul",
        extra_tags=["attention"],
        notes="Fallback profile.",
    )


def get_model_profile(name: Optional[str]) -> Optional[ModelProfile]:
    if not name:
        return None
    resolved = resolve_model_profile(name, name)
    return None if resolved.family == "generic" else resolved
