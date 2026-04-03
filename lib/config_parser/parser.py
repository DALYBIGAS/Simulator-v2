"""Helpers that load compiler-facing YAML specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from lib.llm.models.catalog import get_model_profile, resolve_model_profile

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

    architecture = str(spec.get("architecture", ""))
    model_family = str(spec.get("model_family", architecture or spec.get("model_name", "")))
    profile = get_model_profile(model_family) or resolve_model_profile(str(spec.get("model_name", "")), architecture)

    dtype = str(spec.get("dtype", profile.default_dtype))
    match_op = str(spec.get("match_op", profile.preferred_match_op))
    attention_type = str(spec.get("attention_type", profile.attention_type))
    kv_layout = str(spec.get("kv_layout", profile.kv_layout))
    grouped_query_attention = bool(spec.get("grouped_query_attention", profile.grouped_query_attention))
    mixture_of_experts = bool(spec.get("mixture_of_experts", profile.mixture_of_experts or int(spec.get("num_experts", 0)) > 0))

    raw_tags = [str(v) for v in spec.get("tags", [])]
    inferred_tags: List[str] = []
    if grouped_query_attention:
        inferred_tags.append("gqa")
    if attention_type and attention_type != "mha":
        inferred_tags.append(attention_type)
    if mixture_of_experts:
        inferred_tags.append("moe")
    raw_tags = list(dict.fromkeys(raw_tags + inferred_tags + profile.extra_tags + [profile.family, str(spec.get("mode", "prefill"))]))

    notes = [str(v) for v in spec.get("notes", [])]
    if profile.notes and profile.notes not in notes:
        notes.append(profile.notes)

    return CompileSpec(
        model_name=str(spec.get("model_name", "unnamed-model")),
        kernel_name=str(spec.get("kernel_name", spec.get("model_name", "kernel"))),
        mode=str(spec.get("mode", "prefill")),
        match_op=match_op,
        dtype=dtype,
        architecture=architecture or None,
        model_family=profile.family,
        attention_type=attention_type,
        kv_layout=kv_layout,
        grouped_query_attention=grouped_query_attention,
        mixture_of_experts=mixture_of_experts,
        num_experts=int(spec.get("num_experts", 0)),
        experts_per_token=int(spec.get("experts_per_token", spec.get("top_k_experts", 0))),
        top_k_experts=int(spec.get("top_k_experts", spec.get("experts_per_token", 0))),
        quantization=spec.get("quantization"),
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
        intermediate_size=int(spec.get("intermediate_size", 0)),
        kv_cache=bool(spec.get("kv_cache", False)),
        notes=notes,
        tags=raw_tags,
    )


def load_compilation_context(hardware_path: str | Path, compile_path: str | Path) -> Tuple[HardwareCaps, CompilerOptions, CompileSpec]:
    hw_caps = load_hardware_caps(hardware_path)
    compile_spec = load_compile_spec(compile_path)
    compiler_options = get_compiler_options(compile_spec.mode)
    return hw_caps, compiler_options, compile_spec
