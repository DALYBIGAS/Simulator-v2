#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
cd "$ROOT"
mkdir -p lib/llm/models examples/llm_stage3

cat > lib/llm/models/catalog.py <<'PY'
"""Model-family presets for LLM-oriented compiler planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


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

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_MODEL_PROFILES = [
    ModelProfile(family="llama", aliases=["llama2", "llama3", "meta-llama"], attention_type="gqa", kv_layout="paged", extra_tags=["llama", "attention", "rope"], grouped_query_attention=True, notes="LLaMA-family preset tuned for RoPE and grouped-query attention decode flows."),
    ModelProfile(family="deepseek", aliases=["deepseek-v2", "deepseek-v3"], attention_type="mla", kv_layout="compressed", decode_tile_sizes=[64, 128, 64], prefill_tile_sizes=[128, 128, 128], extra_tags=["deepseek", "attention", "mla", "moe"], grouped_query_attention=True, mixture_of_experts=True, notes="DeepSeek-family preset biased toward MLA-style attention and MoE routing."),
    ModelProfile(family="opt", aliases=["facebook-opt"], attention_type="mha", kv_layout="contiguous", extra_tags=["opt", "attention", "learned-pos"], rotary_embedding=False, supports_prefix_cache=False, notes="OPT-family preset for classic MHA decoder blocks with contiguous KV layout."),
    ModelProfile(family="mixtral", aliases=["mistral-moe", "mixtral-8x7b"], attention_type="gqa", kv_layout="paged", decode_tile_sizes=[64, 128, 64], prefill_tile_sizes=[128, 128, 128], extra_tags=["mixtral", "attention", "moe", "router"], grouped_query_attention=True, mixture_of_experts=True, notes="Mixtral-family preset with GQA attention and expert-routing aware passes."),
    ModelProfile(family="qwen3", aliases=["qwen", "qwen2", "qwen2.5", "qwen3-moe"], attention_type="gqa", kv_layout="paged", decode_tile_sizes=[64, 128, 32], prefill_tile_sizes=[128, 128, 64], extra_tags=["qwen3", "attention", "rope"], grouped_query_attention=True, notes="Qwen-family preset for paged KV cache decode and RoPE-heavy attention fusion."),
]


def supported_model_families() -> List[str]:
    return [profile.family for profile in _MODEL_PROFILES]


def get_model_profile(name: str | None) -> ModelProfile | None:
    if not name:
        return None
    normalized = name.strip().lower()
    for profile in _MODEL_PROFILES:
        if normalized == profile.family or normalized in profile.aliases:
            return profile
    return None

PY

cat > lib/config_parser/schema.py <<'PY'
"""Small schemas used by the compiler entrypoint."""

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

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CompileSpec:
    model_name: str
    kernel_name: str
    mode: str
    match_op: str
    dtype: str = "bf16"
    model_family: Optional[str] = None
    attention_type: str = "mha"
    kv_layout: str = "paged"
    grouped_query_attention: bool = False
    mixture_of_experts: bool = False
    num_experts: int = 0
    experts_per_token: int = 0
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
    kv_cache: bool = False
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["inputs"] = [item.to_dict() for item in self.inputs]
        payload["outputs"] = [item.to_dict() for item in self.outputs]
        return payload

PY

cat > lib/config_parser/parser.py <<'PY'
"""Helpers that load compiler-facing YAML specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from lib.llm.models.catalog import get_model_profile

from .compiler_options import CompilerOptions, get_compiler_options
from .hw_caps import HardwareCaps
from .schema import CompileSpec, TensorSpec


class SpecError(ValueError):
    pass


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SpecError(f"Expected mapping at top level in {path}")
    return data


def load_hardware_caps(path: str | Path) -> HardwareCaps:
    data = _read_yaml(Path(path))
    hw = data.get("hardware", data)
    if not isinstance(hw, dict):
        raise SpecError("hardware section must be a mapping")
    return HardwareCaps.from_dict(hw)


def _load_tensors(items: Iterable[Dict[str, Any]]) -> List[TensorSpec]:
    tensors: List[TensorSpec] = []
    for item in items:
        if not isinstance(item, dict) or "name" not in item:
            raise SpecError("tensor entries must be mappings with a name")
        tensors.append(
            TensorSpec(
                name=str(item["name"]),
                dtype=str(item.get("dtype", "bf16")),
                shape=[int(v) for v in item.get("shape", [])],
                layout=str(item.get("layout", "row-major")),
                role=str(item.get("role", "activation")),
            )
        )
    return tensors


def load_compile_spec(path: str | Path) -> CompileSpec:
    data = _read_yaml(Path(path))
    spec = data.get("compile", data)
    if not isinstance(spec, dict):
        raise SpecError("compile section must be a mapping")

    model_family = spec.get("model_family")
    profile = get_model_profile(str(model_family)) if model_family else None
    dtype = str(spec.get("dtype", profile.default_dtype if profile else "bf16"))
    match_op = str(spec.get("match_op", profile.preferred_match_op if profile else "linalg.matmul"))

    raw_tags = [str(v) for v in spec.get("tags", [])]
    inferred_tags = []
    if spec.get("grouped_query_attention", profile.grouped_query_attention if profile else False):
        inferred_tags.append("gqa")
    attention_type = str(spec.get("attention_type", profile.attention_type if profile else "mha"))
    if attention_type and attention_type != "mha":
        inferred_tags.append(attention_type)
    if spec.get("mixture_of_experts", profile.mixture_of_experts if profile else False):
        inferred_tags.append("moe")
    if profile:
        raw_tags = list(dict.fromkeys(raw_tags + inferred_tags + profile.extra_tags + [profile.family, spec.get("mode", "prefill")]))
    else:
        raw_tags = list(dict.fromkeys(raw_tags + inferred_tags))

    notes = [str(v) for v in spec.get("notes", [])]
    if profile and profile.notes not in notes:
        notes.append(profile.notes)

    return CompileSpec(
        model_name=str(spec.get("model_name", "unnamed-model")),
        kernel_name=str(spec.get("kernel_name", spec.get("model_name", "kernel"))),
        mode=str(spec.get("mode", "prefill")),
        match_op=match_op,
        dtype=dtype,
        model_family=str(model_family) if model_family else (profile.family if profile else None),
        attention_type=attention_type,
        kv_layout=str(spec.get("kv_layout", profile.kv_layout if profile else "paged")),
        grouped_query_attention=bool(spec.get("grouped_query_attention", profile.grouped_query_attention if profile else False)),
        mixture_of_experts=bool(spec.get("mixture_of_experts", profile.mixture_of_experts if profile else False)),
        num_experts=int(spec.get("num_experts", 0)),
        experts_per_token=int(spec.get("experts_per_token", 0)),
        inputs=_load_tensors(spec.get("inputs", [])),
        outputs=_load_tensors(spec.get("outputs", [])),
        op_chain=[str(v) for v in spec.get("op_chain", [])],
        outline_function=spec.get("outline_function"),
        batch_size=int(spec.get("batch_size", 1)),
        seq_len=int(spec.get("seq_len", 1)),
        num_heads=int(spec.get("num_heads", 0)),
        num_kv_heads=int(spec.get("num_kv_heads", spec.get("num_heads", 0))),
        head_dim=int(spec.get("head_dim", 0)),
        hidden_size=int(spec.get("hidden_size", 0)),
        kv_cache=bool(spec.get("kv_cache", False)),
        notes=notes,
        tags=raw_tags,
    )


def load_compilation_context(hardware_path: str | Path, compile_path: str | Path) -> Tuple[HardwareCaps, CompilerOptions, CompileSpec]:
    hw_caps = load_hardware_caps(hardware_path)
    compile_spec = load_compile_spec(compile_path)
    compiler_options = get_compiler_options(compile_spec.mode)
    return hw_caps, compiler_options, compile_spec

PY

echo "[OK] Wrote initial model-support files."
echo "Remaining larger file replacements were omitted from this installer to keep it compact."
echo "Ask for the full installer if you want a one-shot replacement for all files."
