#!/usr/bin/env python3
"""Generate a compact performance report from gem5-like stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.llm.profiler.report import build_perf_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact JSON/Markdown report from gem5 stats.")
    parser.add_argument("stats_file")
    parser.add_argument("--clock-ghz", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_perf_report(args.stats_file, clock_ghz=args.clock_ghz, tokens=args.tokens)
    (out_dir / "perf_report.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out_dir / "perf_report.md").write_text(report.to_markdown(), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
