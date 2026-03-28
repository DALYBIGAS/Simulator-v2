#!/usr/bin/env python3
"""Generate reusable C host stubs for AI-chip kernels.

This generator is intentionally simple and standalone so it can be used
independently from the legacy config parser while the repo transitions toward
an LLM compiler stack.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


C_DTYPE_MAP = {
    "f32": "float",
    "fp32": "float",
    "f16": "uint16_t",
    "fp16": "uint16_t",
    "bf16": "uint16_t",
    "i8": "int8_t",
    "int8": "int8_t",
    "u8": "uint8_t",
}


@dataclass
class TensorArg:
    name: str
    dtype: str

    @property
    def c_type(self) -> str:
        try:
            return C_DTYPE_MAP[self.dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {self.dtype}") from exc


def _sanitize_names(names: Iterable[str]) -> List[str]:
    result = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError(f"Invalid C identifier: {name}")
        result.append(name)
    return result


def build_stub(kernel_name: str, mode: str, dtype: str, inputs: List[str], outputs: List[str]) -> str:
    input_args = [TensorArg(name, dtype) for name in inputs]
    output_args = [TensorArg(name, dtype) for name in outputs]
    all_args = input_args + output_args

    abi_struct = """typedef struct {
  void *data;
  int64_t *shape;
  int64_t *stride;
  int32_t rank;
} ai_tensor_ref;\n\n"""

    fn_args = ", ".join([f"ai_tensor_ref {arg.name}" for arg in all_args] + ["volatile uint8_t *dma_flags", "uint64_t accelerator_base"])
    lines = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        abi_struct.rstrip(),
        f"// Auto-generated host stub for mode={mode}, kernel={kernel_name}",
        f"void {kernel_name}_launch({fn_args}) {{",
        "  // TODO: Replace placeholder register map with accelerator-specific layout.",
        "  volatile uint64_t *accel_kernel_id = (volatile uint64_t *)(accelerator_base + 0x00);",
        "  volatile uint64_t *accel_arg_base = (volatile uint64_t *)(accelerator_base + 0x40);",
        "  volatile uint64_t *accel_ctrl = (volatile uint64_t *)(accelerator_base + 0x08);",
        "",
        f"  *accel_kernel_id = 0; // kernel slot for {kernel_name}",
    ]

    for index, arg in enumerate(all_args):
        offset = index * 4
        lines.extend(
            [
                f"  accel_arg_base[{offset + 0}] = (uint64_t){arg.name}.data;",
                f"  accel_arg_base[{offset + 1}] = (uint64_t){arg.name}.shape;",
                f"  accel_arg_base[{offset + 2}] = (uint64_t){arg.name}.stride;",
                f"  accel_arg_base[{offset + 3}] = (uint64_t){arg.name}.rank;",
            ]
        )

    lines.extend(
        [
            "",
            "  (void)dma_flags; // reserved for async DMA orchestration",
            "  *accel_ctrl = 0x1;",
            "  while (((*accel_ctrl) & 0x4) != 0x4) {",
            "    ;",
            "  }",
            "}",
            "",
        ]
    )

    wrappers = []
    for arg in input_args:
        wrappers.append(f"// input tensor: {arg.name} ({arg.c_type})")
    for arg in output_args:
        wrappers.append(f"// output tensor: {arg.name} ({arg.c_type})")

    return "\n".join(lines + wrappers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a reusable C host stub for an AI-chip kernel.")
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--mode", default="inference", choices=["inference", "prefill", "decode", "train-fwd", "train-bwd"])
    parser.add_argument("--dtype", default="f32")
    parser.add_argument("--inputs", default="input")
    parser.add_argument("--outputs", default="output")
    parser.add_argument("--output", required=True, help="Path to generated C file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = _sanitize_names(args.inputs.split(","))
    outputs = _sanitize_names(args.outputs.split(","))
    rendered = build_stub(args.kernel_name, args.mode, args.dtype, inputs, outputs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(output_path)


if __name__ == "__main__":
    main()
