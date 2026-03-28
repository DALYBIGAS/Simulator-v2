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

