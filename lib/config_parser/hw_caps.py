"""Hardware capability model for AI-chip compiler decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SRAMConfig:
    size_bytes: int
    banks: int = 1
    bank_width_bytes: int = 32


@dataclass
class DMAConfig:
    bandwidth_gbps: float
    max_request_bytes: int
    supports_async: bool = True
    supports_2d: bool = False


@dataclass
class ComputeArrayConfig:
    rows: int
    cols: int
    supported_dtypes: List[str] = field(default_factory=lambda: ["f32"])
    native_mma_m: Optional[int] = None
    native_mma_n: Optional[int] = None
    native_mma_k: Optional[int] = None


@dataclass
class HardwareCaps:
    name: str
    sram: SRAMConfig
    dma: DMAConfig
    compute: ComputeArrayConfig
    supports_kv_cache: bool = False
    supports_fused_epilogue: bool = False
    supports_prefill_decode_split: bool = False

    @property
    def array_shape(self) -> str:
        return f"{self.compute.rows}x{self.compute.cols}"

    def supports_dtype(self, dtype: str) -> bool:
        return dtype in self.compute.supported_dtypes

    def recommended_block_k(self) -> int:
        return self.compute.native_mma_k or min(64, max(16, self.compute.cols))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "HardwareCaps":
        sram_raw = raw.get("sram", {})
        dma_raw = raw.get("dma", {})
        compute_raw = raw.get("compute", {})
        return cls(
            name=raw.get("name", "unnamed-chip"),
            sram=SRAMConfig(
                size_bytes=int(sram_raw.get("size_bytes", 0)),
                banks=int(sram_raw.get("banks", 1)),
                bank_width_bytes=int(sram_raw.get("bank_width_bytes", 32)),
            ),
            dma=DMAConfig(
                bandwidth_gbps=float(dma_raw.get("bandwidth_gbps", 0.0)),
                max_request_bytes=int(dma_raw.get("max_request_bytes", 0)),
                supports_async=bool(dma_raw.get("supports_async", True)),
                supports_2d=bool(dma_raw.get("supports_2d", False)),
            ),
            compute=ComputeArrayConfig(
                rows=int(compute_raw.get("rows", 1)),
                cols=int(compute_raw.get("cols", 1)),
                supported_dtypes=list(compute_raw.get("supported_dtypes", ["f32"])),
                native_mma_m=compute_raw.get("native_mma_m"),
                native_mma_n=compute_raw.get("native_mma_n"),
                native_mma_k=compute_raw.get("native_mma_k"),
            ),
            supports_kv_cache=bool(raw.get("supports_kv_cache", False)),
            supports_fused_epilogue=bool(raw.get("supports_fused_epilogue", False)),
            supports_prefill_decode_split=bool(raw.get("supports_prefill_decode_split", False)),
        )
