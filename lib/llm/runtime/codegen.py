"""Generate C runtime launch code from a runtime plan."""

from __future__ import annotations

from typing import Any, Dict, List


def _normalize_plan(plan: Any) -> Dict[str, Any]:
    if isinstance(plan, dict):
        return plan
    if hasattr(plan, "to_dict"):
        return plan.to_dict()
    raise TypeError("Unsupported runtime plan type")


def _emit_buffer_comment(index: int, item: Dict[str, Any]) -> str:
    tensor = item["tensor"]
    return (
        f"  // buffer[{index}] name={tensor['name']} dtype={tensor['dtype']} "
        f"rank={tensor['rank']} space={item['memory_space']} double_buffered={int(item['double_buffered'])}"
    )


def generate_runtime_launch_code(plan: Any, entry_name: str = "run_compiled_kernel") -> str:
    normalized = _normalize_plan(plan)
    buffers = normalized.get("buffers", [])
    launches = normalized.get("launches", [])
    metadata = normalized.get("metadata", {})
    lines: List[str] = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        "typedef struct {",
        "  void *data;",
        "  const int64_t *shape;",
        "  const int64_t *stride;",
        "  int32_t rank;",
        "} ai_tensor_ref;",
        "",
        "typedef struct {",
        "  uint32_t grid[3];",
        "  uint32_t block[3];",
        "  uint32_t stream_id;",
        "} ai_launch_desc;",
        "",
        "static void ai_wait_event(const char *name) { (void)name; }",
        "static void ai_emit_event(const char *name) { (void)name; }",
        "static void ai_launch_kernel(uint64_t accelerator_base, const char *kernel_name, const ai_launch_desc *desc) {",
        "  (void)accelerator_base;",
        "  (void)kernel_name;",
        "  (void)desc;",
        "}",
        "",
        f"void {entry_name}(uint64_t accelerator_base) {{",
        "  // Auto-generated runtime launch plan.",
        f"  // mode={metadata.get('mode', 'unknown')} kv_cache={int(bool(metadata.get('kv_cache', False)))} is_moe={int(bool(metadata.get('is_moe', False)))}",
    ]
    for index, item in enumerate(buffers):
        lines.append(_emit_buffer_comment(index, item))
    lines.append("")
    for i, launch in enumerate(launches):
        lines.append(f"  // launch[{i}] kind={launch.get('kind', 'compute')}")
        for event in launch.get("waits_on", []):
            lines.append(f'  ai_wait_event("{event}");')
        lines.append("  {")
        lines.append("    ai_launch_desc desc = {")
        lines.append(f"      .grid = {{{launch['grid'][0]}, {launch['grid'][1]}, {launch['grid'][2]}}},")
        lines.append(f"      .block = {{{launch['block'][0]}, {launch['block'][1]}, {launch['block'][2]}}},")
        lines.append(f"      .stream_id = {launch['stream_id']},")
        lines.append("    };")
        lines.append(f'    ai_launch_kernel(accelerator_base, "{launch["kernel_name"]}", &desc);')
        lines.append("  }")
        for event in launch.get("emits", []):
            lines.append(f'  ai_emit_event("{event}");')
        if i + 1 != len(launches):
            lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
