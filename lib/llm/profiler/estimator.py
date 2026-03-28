"""Simple performance estimators from compile specs and hardware caps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from lib.config_parser.hw_caps import HardwareCaps
from lib.config_parser.schema import CompileSpec


_DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "f16": 2,
    "f32": 4,
    "fp32": 4,
    "int8": 1,
    "i8": 1,
}


@dataclass
class EstimatedMetrics:
    model_family: str
    estimated_flops: float
    estimated_bytes: float
    estimated_latency_ms: float
    estimated_tokens_per_sec: float
    estimated_bandwidth_gbps: float
    estimated_compute_utilization: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def estimate_metrics(hw_caps: HardwareCaps, compile_spec: CompileSpec, model_family: str) -> EstimatedMetrics:
    dtype_bytes = _DTYPE_BYTES.get(compile_spec.dtype, 2)
    hidden = max(compile_spec.hidden_size, 1)
    seq = max(compile_spec.seq_len, 1)
    heads = max(compile_spec.num_heads, 1)
    head_dim = max(compile_spec.head_dim, max(hidden // heads, 1))
    batch = max(compile_spec.batch_size, 1)

    if compile_spec.mode == "decode":
        flops = 4.0 * batch * heads * seq * head_dim * max(seq, 1)
    else:
        flops = 6.0 * batch * seq * hidden * hidden

    if compile_spec.num_experts > 0:
        flops *= max(compile_spec.top_k_experts, 1)

    bytes_moved = float((batch * seq * hidden + heads * seq * head_dim * 2) * dtype_bytes)
    peak_ops = float(max(hw_caps.compute.rows * hw_caps.compute.cols * 2048, 1))
    compute_time_ms = flops / peak_ops * 1e3
    bandwidth_time_ms = 0.0
    if hw_caps.dma.bandwidth_gbps > 0:
        bandwidth_time_ms = (bytes_moved * 8.0) / (hw_caps.dma.bandwidth_gbps * 1e9) * 1e3
    latency_ms = max(compute_time_ms, bandwidth_time_ms)
    tokens_per_sec = 0.0 if latency_ms <= 0 else 1000.0 * batch * max(seq if compile_spec.mode != "decode" else 1, 1) / latency_ms
    util = min(1.0, 0.0 if latency_ms <= 0 else compute_time_ms / latency_ms)

    return EstimatedMetrics(
        model_family=model_family,
        estimated_flops=flops,
        estimated_bytes=bytes_moved,
        estimated_latency_ms=latency_ms,
        estimated_tokens_per_sec=tokens_per_sec,
        estimated_bandwidth_gbps=hw_caps.dma.bandwidth_gbps,
        estimated_compute_utilization=util,
    )
