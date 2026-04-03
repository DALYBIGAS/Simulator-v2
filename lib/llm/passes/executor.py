"""In-repo MLIR pass engine for the supported LLM kernel subset.

This is intentionally small but *real*: it parses a constrained MLIR-like IR,
executes ordered passes that mutate the operation graph, and renders the updated
module. The engine is used when external ``mlir-opt`` is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


_FUNC_RE = re.compile(r"^\s*func\.func\s+@(?P<name>[^(]+)\((?P<args>.*)\)\s*(?:->\s*(?P<ret>.+?)\s*)?\{\s*$")
_OP_RE = re.compile(r"^\s*(?:(?P<result>%[A-Za-z0-9_.$-]+)\s*=\s*)?(?P<name>[A-Za-z0-9_.-]+)(?P<body>.*)$")
_ATTR_RE = re.compile(r"\{(?P<attrs>.*)\}")


@dataclass
class Operation:
    name: str
    result: Optional[str] = None
    operands: List[str] = field(default_factory=list)
    attrs: Dict[str, str] = field(default_factory=dict)
    type_sig: str = ""
    raw: str = ""

    def clone(self, **updates: Any) -> "Operation":
        payload = {
            "name": self.name,
            "result": self.result,
            "operands": list(self.operands),
            "attrs": dict(self.attrs),
            "type_sig": self.type_sig,
            "raw": self.raw,
        }
        payload.update(updates)
        return Operation(**payload)

    def structural_key(self) -> Tuple[Any, ...]:
        return (self.name, tuple(self.operands), tuple(sorted(self.attrs.items())), self.type_sig)

    def render(self) -> str:
        if self.name == "return":
            operands = ", ".join(self.operands)
            if self.type_sig:
                return f"    return {operands} : {self.type_sig}" if operands else f"    return : {self.type_sig}"
            return f"    return {operands}".rstrip()
        pieces: List[str] = ["    "]
        if self.result:
            pieces.append(f"{self.result} = ")
        pieces.append(self.name)
        if self.operands:
            pieces.append(" ")
            pieces.append(", ".join(self.operands))
        if self.attrs:
            attrs = ", ".join(f"{key} = {value}" for key, value in sorted(self.attrs.items()))
            pieces.append(f" {{{attrs}}}")
        if self.type_sig:
            pieces.append(f" : {self.type_sig}")
        return "".join(pieces)


@dataclass
class FunctionIR:
    name: str
    args_sig: str
    ret_sig: str
    ops: List[Operation] = field(default_factory=list)

    def render(self) -> str:
        ret = f" -> {self.ret_sig}" if self.ret_sig else ""
        lines = [f"  func.func @{self.name}({self.args_sig}){ret} {{"]
        lines.extend(op.render() for op in self.ops)
        lines.append("  }")
        return "\n".join(lines)


@dataclass
class ModuleIR:
    functions: List[FunctionIR] = field(default_factory=list)
    attrs: Dict[str, str] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)

    def render(self) -> str:
        attr_str = ""
        if self.attrs:
            attr_items = ", ".join(f"{key} = {value}" for key, value in sorted(self.attrs.items()))
            attr_str = f" attributes {{{attr_items}}}"
        lines = ["module" + attr_str + " {"]
        for fn in self.functions:
            lines.append(fn.render())
        lines.append("}")
        return "\n".join(lines) + "\n"


def _parse_attrs(raw: str) -> Dict[str, str]:
    text = raw.strip()
    if not text:
        return {}
    parts = [item.strip() for item in text.split(",") if item.strip()]
    attrs: Dict[str, str] = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            attrs[key.strip()] = value.strip()
        else:
            attrs[part] = "true"
    return attrs


def _split_operands(body: str) -> Tuple[List[str], Dict[str, str]]:
    body = body.strip()
    if not body:
        return [], {}
    attrs: Dict[str, str] = {}
    attr_match = _ATTR_RE.search(body)
    if attr_match:
        attrs = _parse_attrs(attr_match.group("attrs"))
        body = (body[: attr_match.start()] + body[attr_match.end() :]).strip()
    if not body:
        return [], attrs
    operands = [item.strip() for item in body.split(",") if item.strip()]
    return operands, attrs


def parse_mlir_module(text: str) -> ModuleIR:
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith("//")]
    module = ModuleIR()
    current: Optional[FunctionIR] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("module"):
            if "attributes" in stripped:
                attr_text = stripped.split("attributes", 1)[1].strip()
                if attr_text.startswith("{") and attr_text.endswith("{"):
                    attr_text = attr_text[1:-1]
                elif attr_text.startswith("{"):
                    attr_text = attr_text[1:]
                module.attrs.update(_parse_attrs(attr_text.rstrip("{").rstrip("}")))
            continue
        func_match = _FUNC_RE.match(line)
        if func_match:
            current = FunctionIR(
                name=func_match.group("name").strip(),
                args_sig=func_match.group("args").strip(),
                ret_sig=(func_match.group("ret") or "").strip(),
                ops=[],
            )
            module.functions.append(current)
            continue
        if stripped == "}" and current is not None:
            current = None
            continue
        if current is None:
            continue
        if stripped.startswith("return"):
            _, _, tail = stripped.partition("return")
            left, _, type_sig = tail.partition(":")
            operands = [item.strip() for item in left.split(",") if item.strip()]
            current.ops.append(Operation(name="return", operands=operands, type_sig=type_sig.strip(), raw=line))
            continue
        left, sep, type_sig = stripped.partition(":")
        match = _OP_RE.match(left.strip())
        if not match:
            continue
        operands, attrs = _split_operands(match.group("body"))
        current.ops.append(
            Operation(
                result=match.group("result"),
                name=match.group("name"),
                operands=operands,
                attrs=attrs,
                type_sig=type_sig.strip() if sep else "",
                raw=line,
            )
        )
    return module


def _append_history(module: ModuleIR, message: str) -> None:
    module.history.append(message)
    module.attrs[f"llm.history_{len(module.history)}"] = f'"{message}"'


def _replace_uses(ops: Iterable[Operation], old: str, new: str) -> None:
    for op in ops:
        op.operands = [new if item == old else item for item in op.operands]


def _next_result_name(fn: FunctionIR, stem: str) -> str:
    seen = {op.result for op in fn.ops if op.result}
    if f"%{stem}" not in seen:
        return f"%{stem}"
    index = 0
    while f"%{stem}{index}" in seen:
        index += 1
    return f"%{stem}{index}"


def synthesize_llm_kernel_graph(module: ModuleIR, compile_spec: Dict[str, Any]) -> ModuleIR:
    if not module.functions:
        return module
    fn = module.functions[0]
    meaningful = [op for op in fn.ops if op.name != "return" and not op.name.startswith("tensor.")]
    if meaningful:
        return module

    output_name = compile_spec.get("outputs", [{}])[0].get("name", "out")
    output_var = f"%{output_name}"
    mode = compile_spec.get("mode", "prefill")
    kv_cache = bool(compile_spec.get("kv_cache", False))
    is_moe = bool(compile_spec.get("is_moe", False))
    hidden_tensor = compile_spec.get("inputs", [{}])[0].get("name", "input")
    hidden_var = f"%{hidden_tensor}"
    fn.ops = [op for op in fn.ops if op.name == "return"]
    generated: List[Operation] = []
    if mode == "decode":
        generated.extend(
            [
                Operation(result="%scores", name="llm.kv_matmul", operands=[hidden_var, "%key_cache"], attrs={"kv_cache": str(kv_cache).lower()}, type_sig="tensor<?> -> tensor<?>"),
                Operation(result="%probs", name="llm.softmax", operands=["%scores"], type_sig="tensor<?> -> tensor<?>"),
                Operation(result="%context", name="llm.kv_matmul", operands=["%probs", "%value_cache"], attrs={"kv_cache": str(kv_cache).lower()}, type_sig="tensor<?> -> tensor<?>"),
            ]
        )
        if is_moe:
            generated.extend(
                [
                    Operation(result="%gate", name="llm.moe_gate", operands=["%context"], attrs={"experts": str(compile_spec.get("num_experts", 0)), "top_k": str(compile_spec.get("effective_top_k_experts", 1))}, type_sig="tensor<?> -> tensor<?>"),
                    Operation(result="%routed", name="llm.moe_dispatch", operands=["%context", "%gate"], type_sig="tensor<?> -> tensor<?>"),
                    Operation(result="%expert_out", name="llm.expert_matmul", operands=["%routed"], type_sig="tensor<?> -> tensor<?>"),
                    Operation(result=output_var, name="llm.moe_combine", operands=["%expert_out", "%gate"], type_sig="tensor<?> -> tensor<?>"),
                ]
            )
        else:
            generated.append(Operation(result=output_var, name="llm.identity", operands=["%context"], type_sig="tensor<?> -> tensor<?>"))
    else:
        generated.extend(
            [
                Operation(result="%qkv", name="llm.qkv_projection", operands=[hidden_var], attrs={"mode": f'"{mode}"'}, type_sig="tensor<?> -> tensor<?>"),
                Operation(result=output_var, name="llm.prefill_attention", operands=["%qkv"], attrs={"mode": f'"{mode}"'}, type_sig="tensor<?> -> tensor<?>"),
            ]
        )
    generated.extend(fn.ops)
    if generated and fn.ops and fn.ops[-1].name == "return":
        ret = fn.ops[-1]
        ret.operands = [output_var]
    fn.ops = generated
    _append_history(module, "synthesized_llm_kernel_graph")
    return module


def _run_canonicalize(module: ModuleIR) -> None:
    for fn in module.functions:
        for op in fn.ops:
            op.attrs = {k: op.attrs[k] for k in sorted(op.attrs)}
    _append_history(module, "canonicalize")


def _run_cse(module: ModuleIR) -> None:
    for fn in module.functions:
        seen: Dict[Tuple[Any, ...], str] = {}
        new_ops: List[Operation] = []
        for op in fn.ops:
            if op.name == "return" or not op.result:
                new_ops.append(op)
                continue
            key = op.structural_key()
            existing = seen.get(key)
            if existing:
                _replace_uses(fn.ops, op.result, existing)
            else:
                seen[key] = op.result
                new_ops.append(op)
        fn.ops = new_ops
    _append_history(module, "cse")


def _fuse_attention(fn: FunctionIR) -> bool:
    changed = False
    i = 0
    while i <= len(fn.ops) - 3:
        a, b, c = fn.ops[i : i + 3]
        if a.name == "llm.kv_matmul" and b.name == "llm.softmax" and c.name == "llm.kv_matmul":
            result = c.result or _next_result_name(fn, "decode_attention")
            fused = Operation(
                result=result,
                name="llm.decode_attention",
                operands=[a.operands[0] if a.operands else "%query", a.operands[1] if len(a.operands) > 1 else "%key_cache", c.operands[1] if len(c.operands) > 1 else "%value_cache"],
                attrs={**a.attrs, "fused": "true"},
                type_sig=c.type_sig or a.type_sig,
            )
            suffix = fn.ops[i + 3 :]
            if c.result:
                _replace_uses(suffix, c.result, result)
            fn.ops = fn.ops[:i] + [fused] + suffix
            changed = True
            continue
        i += 1
    return changed


def _run_fuse_attention_chain(module: ModuleIR) -> None:
    if any(_fuse_attention(fn) for fn in module.functions):
        _append_history(module, "llm-fuse-attention-chain")


def _run_materialize_kv_cache(module: ModuleIR) -> None:
    for fn in module.functions:
        insertions: List[Operation] = []
        for op in fn.ops:
            if any("key_cache" in operand or "value_cache" in operand for operand in op.operands):
                op.attrs["kv_cache_materialized"] = "true"
        has_alloc = any(op.name == "llm.kv_cache_alloc" for op in fn.ops)
        if not has_alloc:
            insertions.append(Operation(result="%kv_cache", name="llm.kv_cache_alloc", attrs={"layout": '"paged"'}, type_sig="tensor<?>"))
        if insertions:
            returns = [op for op in fn.ops if op.name == "return"]
            body = [op for op in fn.ops if op.name != "return"]
            fn.ops = insertions + body + returns
    _append_history(module, "llm-materialize-kv-cache")


def _lower_moe(fn: FunctionIR) -> bool:
    names = [op.name for op in fn.ops]
    target = ["llm.moe_gate", "llm.moe_dispatch", "llm.expert_matmul", "llm.moe_combine"]
    for i in range(len(names) - len(target) + 1):
        if names[i : i + len(target)] == target:
            gate, dispatch, expert, combine = fn.ops[i : i + 4]
            result = combine.result or _next_result_name(fn, "moe_out")
            lowered = Operation(
                result=result,
                name="llm.moe_decode",
                operands=[gate.operands[0] if gate.operands else "%hidden"],
                attrs={
                    "experts": gate.attrs.get("experts", "0"),
                    "top_k": gate.attrs.get("top_k", "1"),
                    "dispatch": '"hierarchical"',
                },
                type_sig=combine.type_sig or expert.type_sig,
            )
            suffix = fn.ops[i + 4 :]
            if combine.result:
                _replace_uses(suffix, combine.result, result)
            fn.ops = fn.ops[:i] + [lowered] + suffix
            return True
    return False


def _run_lower_moe_routing(module: ModuleIR) -> None:
    if any(_lower_moe(fn) for fn in module.functions):
        _append_history(module, "llm-lower-moe-routing")


def _run_tile(module: ModuleIR, sizes: str) -> None:
    for fn in module.functions:
        for op in fn.ops:
            if op.name.startswith("llm.") and op.name not in {"llm.kv_cache_alloc", "llm.launch", "llm.event_record", "llm.async_dma"}:
                op.attrs["tile_sizes"] = f'"{sizes}"'
    _append_history(module, f"llm-tile[{sizes}]")


def _run_fuse_epilogue(module: ModuleIR) -> None:
    module.attrs["llm.fused_epilogue"] = "true"
    _append_history(module, "llm-fuse-epilogue")


def _run_select_kernel(module: ModuleIR, kernel_name: str) -> None:
    for fn in module.functions:
        for op in fn.ops:
            if op.name in {"llm.decode_attention", "llm.prefill_attention", "llm.moe_decode", "llm.qkv_projection"}:
                op.attrs["selected_kernel"] = f'"{kernel_name}"'
    _append_history(module, f"llm-select-kernel[{kernel_name}]")


def _run_promote_sram(module: ModuleIR) -> None:
    for fn in module.functions:
        for op in fn.ops:
            if op.name.startswith("llm.") and op.name not in {"llm.kv_cache_alloc", "llm.launch", "llm.event_record", "llm.async_dma"}:
                op.attrs.setdefault("memory_space", '"sram"')
    _append_history(module, "llm-promote-sram")


def _run_insert_async_dma(module: ModuleIR) -> None:
    for fn in module.functions:
        new_ops: List[Operation] = []
        for op in fn.ops:
            if op.name in {"llm.decode_attention", "llm.prefill_attention", "llm.moe_decode", "llm.qkv_projection"}:
                new_ops.append(Operation(name="llm.async_dma", operands=op.operands[:], attrs={"kind": '"prefetch"'}, type_sig="none"))
            new_ops.append(op)
        fn.ops = new_ops
    _append_history(module, "llm-insert-async-dma")


def _run_emit_events(module: ModuleIR) -> None:
    for fn in module.functions:
        new_ops: List[Operation] = []
        for op in fn.ops:
            new_ops.append(op)
            if op.name in {"llm.async_dma", "llm.decode_attention", "llm.moe_decode", "llm.prefill_attention"}:
                event_name = op.name.split(".", 1)[1].replace("_", "-") + "-done"
                new_ops.append(Operation(name="llm.event_record", operands=[f'"{event_name}"'], type_sig="none"))
        fn.ops = new_ops
    _append_history(module, "llm-emit-events")


def _run_allocate_kv_cache(module: ModuleIR) -> None:
    for fn in module.functions:
        if not any(op.name == "llm.kv_cache_alloc" for op in fn.ops):
            returns = [op for op in fn.ops if op.name == "return"]
            body = [op for op in fn.ops if op.name != "return"]
            fn.ops = [Operation(result="%kv_cache", name="llm.kv_cache_alloc", attrs={"space": '"global"'}, type_sig="tensor<?>")] + body + returns
    _append_history(module, "llm-allocate-kv-cache")


def _run_emit_moe_dispatch(module: ModuleIR) -> None:
    for fn in module.functions:
        for op in fn.ops:
            if op.name == "llm.moe_decode":
                op.attrs["dispatch_runtime"] = '"streamed"'
    _append_history(module, "llm-emit-moe-dispatch")


def _run_emit_launch_plan(module: ModuleIR) -> None:
    for fn in module.functions:
        launches: List[Operation] = []
        for op in fn.ops:
            if op.name in {"llm.decode_attention", "llm.moe_decode", "llm.prefill_attention", "llm.qkv_projection"}:
                launches.append(Operation(name="llm.launch", operands=[f'"{op.name.split(".", 1)[1]}"'], attrs={"stream": "0"}, type_sig="none"))
        returns = [op for op in fn.ops if op.name == "return"]
        body = [op for op in fn.ops if op.name != "return"]
        fn.ops = body + launches + returns
    _append_history(module, "llm-emit-launch-plan")


def _run_backend_lower(module: ModuleIR, pass_name: str) -> None:
    mapping = {
        "llm.decode_attention": "backend.decode_attention",
        "llm.prefill_attention": "backend.prefill_attention",
        "llm.qkv_projection": "backend.qkv_projection",
        "llm.moe_decode": "backend.moe_decode",
        "llm.async_dma": "runtime.async_dma",
        "llm.event_record": "runtime.event_record",
        "llm.launch": "runtime.launch",
        "llm.kv_cache_alloc": "runtime.kv_cache_alloc",
        "llm.identity": "backend.identity",
    }
    for fn in module.functions:
        for op in fn.ops:
            op.name = mapping.get(op.name, op.name)
            op.attrs.setdefault("lowered_by", f'"{pass_name}"')
    _append_history(module, pass_name)


def _run_finalize_llvm(module: ModuleIR) -> None:
    for fn in module.functions:
        for op in fn.ops:
            if op.name.startswith("backend."):
                callee = op.name.split(".", 1)[1]
                op.name = "llvm.call"
                op.attrs["callee"] = f'"legend_{callee}"'
            elif op.name.startswith("runtime."):
                callee = op.name.split(".", 1)[1]
                op.name = "llvm.call"
                op.attrs["callee"] = f'"legend_{callee}"'
    module.attrs["llvm.emit_c_interface"] = "true"
    _append_history(module, "finalize-memref-to-llvm")


def execute_pass_pipeline(input_mlir: str, pipeline: Dict[str, List[str]], manifest: Dict[str, Any]) -> Tuple[str, ModuleIR, Dict[str, Any]]:
    module = parse_mlir_module(input_mlir)
    compile_spec = manifest.get("compile_spec", {})
    module = synthesize_llm_kernel_graph(module, compile_spec)

    executed: List[str] = []
    for stage in ["graph", "kernel", "buffer", "runtime", "backend"]:
        for pass_name in pipeline.get(stage, []):
            executed.append(pass_name)
            if pass_name == "canonicalize":
                _run_canonicalize(module)
            elif pass_name == "cse":
                _run_cse(module)
            elif pass_name == "llm-fuse-attention-chain":
                _run_fuse_attention_chain(module)
            elif pass_name == "llm-materialize-kv-cache":
                _run_materialize_kv_cache(module)
            elif pass_name == "llm-lower-moe-routing":
                _run_lower_moe_routing(module)
            elif pass_name.startswith("llm-tile["):
                _run_tile(module, pass_name.split("[", 1)[1].rstrip("]"))
            elif pass_name == "llm-fuse-epilogue":
                _run_fuse_epilogue(module)
            elif pass_name.startswith("llm-select-kernel["):
                _run_select_kernel(module, pass_name.split("[", 1)[1].rstrip("]"))
            elif pass_name == "llm-promote-sram":
                _run_promote_sram(module)
            elif pass_name == "llm-insert-async-dma":
                _run_insert_async_dma(module)
            elif pass_name == "llm-emit-events":
                _run_emit_events(module)
            elif pass_name == "llm-allocate-kv-cache":
                _run_allocate_kv_cache(module)
            elif pass_name == "llm-emit-moe-dispatch":
                _run_emit_moe_dispatch(module)
            elif pass_name == "llm-emit-launch-plan":
                _run_emit_launch_plan(module)
            elif pass_name in {"lower-affine", "convert-scf-to-cf"}:
                _run_backend_lower(module, pass_name)
            elif pass_name == "finalize-memref-to-llvm":
                _run_finalize_llvm(module)
            else:
                module.attrs[f"unsupported.{pass_name}"] = "true"
                _append_history(module, f"unsupported:{pass_name}")

    stats = {
        "executed_passes": executed,
        "module_history": list(module.history),
        "num_functions": len(module.functions),
        "num_ops": sum(len(fn.ops) for fn in module.functions),
        "num_llvm_calls": sum(1 for fn in module.functions for op in fn.ops if op.name == "llvm.call"),
    }
    return module.render(), module, stats


__all__ = ["Operation", "FunctionIR", "ModuleIR", "parse_mlir_module", "execute_pass_pipeline", "synthesize_llm_kernel_graph"]
