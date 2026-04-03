"""Performance estimators from compile specs and hardware caps.

The estimator is still analytical, but it is structured to separate dense,
attention, KV-cache, and MoE costs so its outputs are easier to calibrate
against simulators such as gem5.
"""

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
    estimated_memory_utilization: float
    dominant_bound: str
    attention_flops: float
    mlp_flops: float
    moe_flops: float
    kv_cache_bytes: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def _peak_compute_ops(hw_caps: HardwareCaps) -> float:
    native_k = max(hw_caps.compute.native_mma_k or 32, 1)
    return float(max(hw_caps.compute.rows * hw_caps.compute.cols * native_k * 2, 1))


def estimate_metrics(hw_caps: HardwareCaps, compile_spec: CompileSpec, model_family: str) -> EstimatedMetrics:
    dtype_bytes = _DTYPE_BYTES.get(compile_spec.dtype, 2)
    hidden = max(compile_spec.hidden_size, 1)
    seq = max(compile_spec.seq_len, 1)
    batch = max(compile_spec.batch_size, 1)
    heads = max(compile_spec.num_heads, 1)
    kv_heads = max(compile_spec.num_kv_heads or heads, 1)
    head_dim = max(compile_spec.head_dim or hidden // heads, 1)
    top_k = max(compile_spec.effective_top_k_experts, 1) if compile_spec.is_moe else 0
    intermediate = max(compile_spec.intermediate_size or hidden * 4, hidden)

    # Decoder kernels are latency-sensitive and touch the full KV span. Prefill is
    # throughput-oriented and dominated by dense projection + attention score/value ops.
    qkv_projection_flops = 6.0 * batch * seq * hidden * hidden
    attention_score_flops = 2.0 * batch * heads * seq * max(seq if compile_spec.mode != "decode" else 1, 1) * head_dim
    attention_value_flops = attention_score_flops
    attention_flops = attention_score_flops + attention_value_flops
    mlp_flops = 4.0 * batch * seq * hidden * intermediate
    dense_flops = qkv_projection_flops if compile_spec.mode == "prefill" else 0.5 * qkv_projection_flops
    moe_flops = 0.0
    if compile_spec.is_moe:
        moe_flops = 2.0 * batch * max(seq, 1) * hidden * intermediate * max(top_k, 1)

    flops = dense_flops + attention_flops + mlp_flops + moe_flops

    activation_bytes = float(batch * seq * hidden * dtype_bytes * 3)
    kv_cache_bytes = 0.0
    if compile_spec.kv_cache:
        cache_seq = max(seq if compile_spec.mode != "decode" else hidden // max(head_dim, 1), 1)
        kv_cache_bytes = float(batch * kv_heads * cache_seq * head_dim * dtype_bytes * 2)
    weight_stream_bytes = float(hidden * hidden * dtype_bytes)
    moe_router_bytes = float(batch * seq * hidden * dtype_bytes * (1 + max(top_k, 0))) if compile_spec.is_moe else 0.0
    bytes_moved = activation_bytes + kv_cache_bytes + weight_stream_bytes + moe_router_bytes

    peak_ops = _peak_compute_ops(hw_caps)
    compute_time_ms = flops / peak_ops * 1e3

    bandwidth_time_ms = 0.0
    if hw_caps.dma.bandwidth_gbps > 0:
        bandwidth_time_ms = (bytes_moved * 8.0) / (hw_caps.dma.bandwidth_gbps * 1e9) * 1e3

    overlap_factor = 0.8 if hw_caps.dma.supports_async else 1.0
    latency_ms = max(compute_time_ms, bandwidth_time_ms * overlap_factor)
    if compile_spec.mode == "decode":
        produced_tokens = float(batch)
    else:
        produced_tokens = float(batch * seq)
    tokens_per_sec = 0.0 if latency_ms <= 0 else (produced_tokens * 1000.0) / latency_ms

    compute_util = min(1.0, 0.0 if latency_ms <= 0 else compute_time_ms / latency_ms)
    memory_util = min(1.0, 0.0 if latency_ms <= 0 else (bandwidth_time_ms * overlap_factor) / latency_ms)
    dominant_bound = "compute" if compute_time_ms >= bandwidth_time_ms * overlap_factor else "memory"

    return EstimatedMetrics(
        model_family=model_family,
        estimated_flops=flops,
        estimated_bytes=bytes_moved,
        estimated_latency_ms=latency_ms,
        estimated_tokens_per_sec=tokens_per_sec,
        estimated_bandwidth_gbps=hw_caps.dma.bandwidth_gbps,
        estimated_compute_utilization=compute_util,
        estimated_memory_utilization=memory_util,
        dominant_bound=dominant_bound,
        attention_flops=attention_flops,
        mlp_flops=mlp_flops,
        moe_flops=moe_flops,
        kv_cache_bytes=kv_cache_bytes,
    )
