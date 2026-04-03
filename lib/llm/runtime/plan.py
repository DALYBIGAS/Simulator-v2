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
    emits: List[str] = field(default_factory=list)
    kind: str = "compute"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RuntimePlan:
    buffers: List[BufferPlan]
    launches: List[LaunchPlan]
    events: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "buffers": [buf.to_dict() for buf in self.buffers],
            "launches": [launch.to_dict() for launch in self.launches],
            "events": list(self.events),
            "metadata": dict(self.metadata),
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


def _buffer_space(hw_caps: HardwareCaps, spec: TensorSpec) -> str:
    if spec.role == "kv-cache" and hw_caps.supports_kv_cache:
        return "kv-cache"
    if spec.role == "weight":
        return "global"
    if spec.role == "activation" and hw_caps.sram.size_bytes > 0:
        return "sram"
    return "global"


def _block_shape(options: CompilerOptions) -> List[int]:
    block = list(options.tile_sizes[:3]) if options.tile_sizes else [1, 1, 1]
    while len(block) < 3:
        block.append(1)
    return [max(v, 1) for v in block[:3]]


def build_runtime_plan(hw_caps: HardwareCaps, options: CompilerOptions, compile_spec: CompileSpec) -> RuntimePlan:
    buffers: List[BufferPlan] = []
    for spec in list(compile_spec.inputs) + list(compile_spec.outputs):
        memory_space = _buffer_space(hw_caps, spec) if spec in compile_spec.inputs else "global"
        buffers.append(
            BufferPlan(
                tensor=_to_abi(spec, compile_spec.dtype),
                memory_space=memory_space,
                double_buffered=options.enable_async_dma and memory_space == "sram",
            )
        )

    block = _block_shape(options)
    seq = max(compile_spec.seq_len, 1)
    batch = max(compile_spec.batch_size, 1)
    heads = max(compile_spec.num_heads or 1, 1)
    grid_x = max(1, (seq + block[0] - 1) // block[0])
    launches: List[LaunchPlan] = []
    events: List[str] = []

    allow_prefetch_launch = options.enable_async_dma and hw_caps.dma.supports_async and compile_spec.mode != "decode"
    if allow_prefetch_launch:
        launches.append(
            LaunchPlan(
                kernel_name=f"{compile_spec.kernel_name}_prefetch",
                grid=[grid_x, batch, 1],
                block=[min(block[0], 32), 1, 1],
                stream_id=0,
                emits=["dma-ready"],
                kind="dma",
            )
        )
        events.append("dma-ready")

    compute_waits = ["dma-ready"] if allow_prefetch_launch else []
    compute_emits = ["kernel-done"]
    launches.append(
        LaunchPlan(
            kernel_name=compile_spec.kernel_name,
            grid=[grid_x, batch, heads],
            block=block,
            stream_id=1 if compile_spec.mode == "decode" else 0,
            waits_on=compute_waits,
            emits=compute_emits,
            kind="compute",
        )
    )
    events.append("kernel-done")

    if compile_spec.is_moe:
        top_k = max(compile_spec.effective_top_k_experts, 1)
        launches.extend(
            [
                LaunchPlan(
                    kernel_name=f"{compile_spec.kernel_name}_route",
                    grid=[batch, max(seq, 1), 1],
                    block=[min(block[0], 64), 1, 1],
                    stream_id=2,
                    waits_on=["kernel-done"],
                    emits=["experts-ready"],
                    kind="router",
                ),
                LaunchPlan(
                    kernel_name=f"{compile_spec.kernel_name}_experts",
                    grid=[max(top_k, 1), batch, max(seq, 1)],
                    block=[min(block[1], 128), 1, 1],
                    stream_id=3,
                    waits_on=["experts-ready"],
                    emits=["experts-done"],
                    kind="moe-experts",
                ),
            ]
        )
        events.extend(["experts-ready", "experts-done"])

    metadata = {
        "mode": compile_spec.mode,
        "kv_cache": compile_spec.kv_cache,
        "is_moe": compile_spec.is_moe,
        "top_k_experts": compile_spec.effective_top_k_experts,
        "num_launches": len(launches),
    }
    return RuntimePlan(buffers=buffers, launches=launches, events=events, metadata=metadata)
