"""Build a runtime launch plan from hardware caps and compile specs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from lib.config_parser.compiler_options import CompilerOptions
from lib.config_parser.hw_caps import HardwareCaps
from lib.config_parser.schema import CompileSpec, TensorSpec
from .abi import TensorABI


@dataclass
class BufferPlan:
    tensor: TensorABI
    memory_space: str
    double_buffered: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "tensor": asdict(self.tensor),
            "memory_space": self.memory_space,
            "double_buffered": self.double_buffered,
        }


@dataclass
class LaunchPlan:
    kernel_name: str
    grid: List[int]
    block: List[int]
    stream_id: int
    waits_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RuntimePlan:
    buffers: List[BufferPlan]
    launches: List[LaunchPlan]
    events: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "buffers": [buf.to_dict() for buf in self.buffers],
            "launches": [launch.to_dict() for launch in self.launches],
            "events": list(self.events),
        }


def _contiguous_stride(shape: List[int]) -> List[int]:
    stride: List[int] = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= max(extent, 1)
    return list(reversed(stride))


def _to_abi(spec: TensorSpec, fallback_dtype: str) -> TensorABI:
    dtype = spec.dtype or fallback_dtype
    return TensorABI(name=spec.name, dtype=dtype, rank=spec.rank, shape=list(spec.shape), stride=_contiguous_stride(spec.shape))


def build_runtime_plan(hw_caps: HardwareCaps, options: CompilerOptions, compile_spec: CompileSpec) -> RuntimePlan:
    buffers: List[BufferPlan] = []

    for spec in compile_spec.inputs:
        memory_space = "global"
        if spec.role in {"weight", "kv-cache"} and hw_caps.supports_kv_cache:
            memory_space = "kv-cache" if spec.role == "kv-cache" else "global"
        elif spec.role == "activation" and hw_caps.sram.size_bytes > 0:
            memory_space = "sram"
        buffers.append(
            BufferPlan(
                tensor=_to_abi(spec, compile_spec.dtype),
                memory_space=memory_space,
                double_buffered=options.enable_async_dma and memory_space == "sram",
            )
        )

    for spec in compile_spec.outputs:
        buffers.append(
            BufferPlan(
                tensor=_to_abi(spec, compile_spec.dtype),
                memory_space="global",
                double_buffered=False,
            )
        )

    block = list(options.tile_sizes[:3]) if options.tile_sizes else [1, 1, 1]
    while len(block) < 3:
        block.append(1)

    seq = max(compile_spec.seq_len, 1)
    grid_x = max(1, (seq + max(block[0], 1) - 1) // max(block[0], 1))
    launches = [
        LaunchPlan(
            kernel_name=compile_spec.kernel_name,
            grid=[grid_x, max(1, compile_spec.batch_size), max(1, compile_spec.num_heads or 1)],
            block=block,
            stream_id=0 if compile_spec.mode != "decode" else 1,
            waits_on=["dma-ready"] if options.enable_async_dma else [],
        )
    ]
    events = ["dma-ready", "kernel-done"] if options.enable_async_dma else ["kernel-done"]
    return RuntimePlan(buffers=buffers, launches=launches, events=events)
