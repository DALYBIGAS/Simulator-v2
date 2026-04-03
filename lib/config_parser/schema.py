"""Schemas used by the compiler planning entrypoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class TensorSpec:
    name: str
    dtype: str = "bf16"
    shape: List[int] = field(default_factory=list)
    layout: str = "row-major"
    role: str = "activation"

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        total = 1
        for extent in self.shape:
            total *= max(int(extent), 1)
        return total

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["rank"] = self.rank
        payload["numel"] = self.numel
        return payload


@dataclass
class CompileSpec:
    model_name: str
    kernel_name: str
    mode: str
    match_op: str
    dtype: str = "bf16"
    architecture: Optional[str] = None
    model_family: Optional[str] = None
    attention_type: str = "mha"
    kv_layout: str = "paged"
    grouped_query_attention: bool = False
    mixture_of_experts: bool = False
    num_experts: int = 0
    experts_per_token: int = 0
    top_k_experts: int = 0
    quantization: Optional[str] = None
    inputs: List[TensorSpec] = field(default_factory=list)
    outputs: List[TensorSpec] = field(default_factory=list)
    op_chain: List[str] = field(default_factory=list)
    outline_function: Optional[str] = None
    batch_size: int = 1
    seq_len: int = 1
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    kv_cache: bool = False
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def effective_top_k_experts(self) -> int:
        return max(self.top_k_experts, self.experts_per_token, 1 if self.num_experts > 0 else 0)

    @property
    def is_moe(self) -> bool:
        return self.mixture_of_experts or self.num_experts > 0 or self.effective_top_k_experts > 0

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["inputs"] = [item.to_dict() for item in self.inputs]
        payload["outputs"] = [item.to_dict() for item in self.outputs]
        payload["effective_top_k_experts"] = self.effective_top_k_experts
        payload["is_moe"] = self.is_moe
        return payload
