"""Parse gem5-like stats and generate compact performance reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


@dataclass
class PerfReport:
    sim_ticks: float
    cycles: float
    insts: float
    dram_bytes_read: float
    dram_bytes_write: float
    token_latency_us: float
    achieved_bandwidth_gbps: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Performance Report",
                "",
                f"- sim_ticks: {self.sim_ticks:.0f}",
                f"- cycles: {self.cycles:.0f}",
                f"- instructions: {self.insts:.0f}",
                f"- dram_bytes_read: {self.dram_bytes_read:.0f}",
                f"- dram_bytes_write: {self.dram_bytes_write:.0f}",
                f"- token_latency_us: {self.token_latency_us:.3f}",
                f"- achieved_bandwidth_gbps: {self.achieved_bandwidth_gbps:.3f}",
            ]
        ) + "\n"


def _parse_stats_text(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        try:
            values[key] = float(parts[1])
        except ValueError:
            continue
    return values


def build_perf_report(stats_path: str | Path, *, clock_ghz: float = 1.0, tokens: int = 1) -> PerfReport:
    text = Path(stats_path).read_text(encoding="utf-8")
    values = _parse_stats_text(text)
    sim_ticks = values.get("simTicks", 0.0)
    cycles = values.get("system.cpu.numCycles", values.get("numCycles", 0.0))
    insts = values.get("simInsts", 0.0)
    dram_read = values.get("system.mem_ctrl.bytesRead", 0.0)
    dram_write = values.get("system.mem_ctrl.bytesWritten", 0.0)
    token_latency_us = 0.0
    if cycles > 0 and clock_ghz > 0 and tokens > 0:
        token_latency_us = cycles / (clock_ghz * 1e3) / tokens
    total_bytes = dram_read + dram_write
    achieved_bandwidth_gbps = 0.0
    if cycles > 0 and clock_ghz > 0:
        seconds = cycles / (clock_ghz * 1e9)
        if seconds > 0:
            achieved_bandwidth_gbps = (total_bytes * 8.0) / seconds / 1e9
    return PerfReport(
        sim_ticks=sim_ticks,
        cycles=cycles,
        insts=insts,
        dram_bytes_read=dram_read,
        dram_bytes_write=dram_write,
        token_latency_us=token_latency_us,
        achieved_bandwidth_gbps=achieved_bandwidth_gbps,
    )
